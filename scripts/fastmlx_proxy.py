#!/usr/bin/env python3
"""An opt-in reverse proxy ``fastmlx serve --front-port`` starts in front of
the served engine: every request is forwarded unchanged to the engine's own
loopback port, and every response gains a handful of ``X-FastMLX-*``
provenance headers (admission outcome, quality card, fit verdict, residency,
engine build, and ``--mtp`` flag-transfer status) computed once from the
launcher's own admission plan.

Nothing about the wire protocol is reinterpreted: every HTTP method, every
path and query string, and every request/response body byte passes through
unmodified. The one exception is ``GET /fastmlx/provenance``, answered by
the proxy itself with a JSON summary of the plan (never the raw ``argv`` or
any model path).

Stdlib only, by design: this module ships in the release tarball's
``libexec/scripts`` alongside ``fastmlx_launch.py`` and must not add a
runtime dependency the packaging step would need to vendor.
"""

from __future__ import annotations

import json
import re
import socket
import struct
import sys
import threading
import time
import uuid
from http import client as http_client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, List, Optional, Tuple
from urllib.parse import urlsplit


# Headers that are meaningful only for ONE hop of a connection and must
# never be relayed to the other side, in either direction (RFC 7230 6.1).
# ``connection`` itself is handled specially below: any token IT names is
# ALSO stripped, since a peer can nominate its own hop-by-hop headers there.
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

# Printable ASCII, no control characters at all (that also excludes CR/LF) --
# a card id or any other plan field that ends up in a header value is never
# trusted to already be a valid header value (see the CR/LF-in-card-id test
# this sanitizer exists for).
_NON_PRINTABLE_ASCII = re.compile(r"[^\x20-\x7e]")

_UPSTREAM_CONNECT_TIMEOUT_SECONDS = 10
_UPSTREAM_READ_CHUNK_BYTES = 65536

# ``BaseHTTPRequestHandler.setup()`` applies this straight to the request
# socket (``self.connection.settimeout(self.timeout)``, see
# ``ProvenanceProxyHandler.timeout`` below) the instant the socket is set
# up. On its own this is NOT a bound on the whole request:
# ``socket.settimeout`` bounds one blocking call, not a request or a
# connection -- a client that delivers at least one byte before each
# per-``recv`` timeout expires makes every ``recv`` succeed, so a bare
# value here never bounds a client that paces itself just inside it (this
# was, in fact, exactly this proxy's own defect on this path -- 64 such
# sockets at a fraction of a byte per second each was a total outage of
# every concurrency slot, forever). The actual fix is
# ``_DeadlineBoundRfile`` (see ``ProvenanceProxyHandler.setup()`` and
# ``_CLIENT_REQUEST_INITIAL_SECONDS`` below), which re-derives a proper
# wall-clock deadline before every individual read. This constant is kept
# for two narrower, still-real jobs ``_DeadlineBoundRfile`` does not cover:
# the brief window between the socket being created and the wrapper being
# installed, and the backstop timeout restored between reads (see
# ``_DeadlineBoundRfile._raw_read1``) so that RESPONSE writes -- which the
# request-receive deadline below must never bound (a streaming SSE
# response legitimately holds its slot for the whole generation) -- are
# still not literally unbounded forever.
_CLIENT_IDLE_TIMEOUT_SECONDS = 120

# ``_DeadlineBoundRfile`` (defined below, installed by
# ``ProvenanceProxyHandler.setup()``) enforces an absolute, extendable
# deadline on RECEIVING a request (request line, headers, body) --
# Apache ``mod_reqtimeout``'s shape, expressed in the same
# absolute-deadline idiom ``_refuse_over_capacity``'s drain loop already
# uses below (compute the deadline once, re-derive each timeout from what
# is left): a client that delivers bytes slower than
# ``_CLIENT_REQUEST_MIN_BYTES_PER_SECOND`` cannot hold a concurrency slot
# indefinitely by dribbling, the way a bare ``socket.settimeout`` (see
# ``_CLIENT_IDLE_TIMEOUT_SECONDS`` above) could and did.
#
# Grace period, in seconds, for the FIRST bytes of a request to arrive
# after the deadline is armed in ``setup()`` -- generous enough that a
# legitimate client on a slow network is never refused just for being
# slow to START a request.
_CLIENT_REQUEST_INITIAL_SECONDS = 30

# Every byte the handler actually reads off the client socket while
# receiving a request extends the deadline by ``1 / this rate`` seconds --
# so a client transferring at or above this rate always stays ahead of its
# own deadline, and a client that dribbles slower than this never does, no
# matter how it paces itself. 1 KiB/s is far below any legitimate client's
# real transfer rate (even a slow mobile link clears this easily) and far
# above what a dribbling attacker can sustain while trying to hold a slot
# cheaply.
_CLIENT_REQUEST_MIN_BYTES_PER_SECOND = 1024

# Absolute ceiling, in seconds, past which no amount of received data can
# extend the deadline further -- without this, a client that never dips
# below the minimum rate could hold a slot forever just by staying exactly
# at it. 300s comfortably covers the largest legitimate request body this
# proxy accepts (``DEFAULT_MAX_REQUEST_BODY_BYTES`` = 64 MiB, which
# crosses the wire in well under a minute on any real network) with wide
# margin, while still being a finite bound on the worst case.
_CLIENT_REQUEST_HARD_DEADLINE_SECONDS = 300

# ``_respond_streamed``/``_respond_buffered`` relay the upstream response
# body to the client with ``self.wfile.write(chunk); self.wfile.flush()``.
# The only bound on any ONE of those writes used to be the handler's own
# socket timeout (``_CLIENT_IDLE_TIMEOUT_SECONDS``) -- and, exactly like
# ``_CLIENT_IDLE_TIMEOUT_SECONDS``'s documented gap on the REQUEST side
# above, ``_respond_streamed``'s loop re-arms a FRESH one before every
# chunk, so the total wall-clock time a client could hold its
# concurrency slot by reading its response just slowly enough was
# UNBOUNDED, and at the default concurrency cap that is a total outage
# exactly like the request-side dribble this file already fixed once.
#
# The exploit shape here is NOT the request side's, and the difference
# was established by measurement, not by symmetry. A client that dribbles
# a BYTE at a time does NOT hold the slot: ``sendall`` does not re-arm
# its timeout per internal send, so the CURRENT write never returns and
# dies on the single backstop. Measured against this file before the fix:
# a client that never reads at all was released after 123.6s, and one
# reading 1 byte/second after 124.3s -- both simply the backstop.
#
# To actually reset the timer the client must let one whole write
# COMPLETE -- drain a full ``_UPSTREAM_READ_CHUNK_BYTES`` -- and may then
# stall again just under the backstop. Measured, that shape held its slot
# for 400.9s and was still holding when the observation window ended, at
# a cost to the attacker of about 3.3 KiB/s (the floor is nearer 64 KiB
# per 120s). THAT is the unbounded case this budget closes.
#
# The metric here is deliberately CUMULATIVE BLOCKED-WRITE WALL-CLOCK
# TIME, never a byte rate (do not copy
# ``_CLIENT_REQUEST_MIN_BYTES_PER_SECOND``'s shape onto this side, even
# though it looks like the natural mirror): a legitimate SSE generation
# is slow by nature (tens of tokens/sec), so a throughput floor would
# refuse exactly the clients this proxy exists to serve. A legitimate
# client instead drains promptly -- the kernel send buffer absorbs each
# delta as it arrives and the proxy's writes never actually block, no
# matter how long the whole generation runs, because the byte RATE of a
# slow-but-honest client is a property of the GENERATION, while the
# blocked-write time is a property of whether the CLIENT is reading at
# all. Blocked-write time separates a slow-but-honest client from one
# that has stopped reading; a throughput floor cannot.
#
# Cumulative across the WHOLE response (every chunk in
# ``_respond_streamed``'s loop shares one budget; ``_respond_buffered``'s
# single write gets the whole budget to itself), re-derived before every
# individual write the same way ``_DeadlineBoundRfile._raw_read1`` (see
# above) re-derives its own read deadline -- without that, a single
# blocked write would sit for the full ``_CLIENT_IDLE_TIMEOUT_SECONDS``
# backstop before this budget is ever consulted.
_CLIENT_RESPONSE_MAX_BLOCKED_SECONDS = 30.0

# Any header FROM UPSTREAM whose name starts with this prefix
# (case-insensitive) is stripped before the proxy's own provenance headers
# are added: a compromised/misbehaving engine must never be able to spoof
# or duplicate them (see L5 in the front-mode review).
_PROVENANCE_HEADER_PREFIX = "x-fastmlx-"

# Default ceiling on a SINGLE request's DECLARED ``Content-Length``, refused
# in ``_dispatch`` before a single body byte is read (see the check there).
# There is otherwise no upper bound at all: the value comes straight off the
# client's own header, and ``--front-host`` accepts any bind address (a
# supported, documented topology, not only loopback) in front of a serve
# host that holds tens of GiB of wired model weights -- an unbounded read is
# a denial-of-service surface, not just a correctness one.
#
# 64 MiB is sized far above any legitimate chat/completions body and far
# below what threatens the host: the largest supported context (262144
# tokens) is on the order of 1 MB of text even before token-vs-character
# compression is accounted for, and a request layering in several
# base64-encoded images stays well under this ceiling too. This bounds ONE
# request's own buffer, never the AGGREGATE memory many concurrent
# ``ThreadingHTTPServer`` daemon threads could hold at once each just under
# the limit -- that second, aggregate threat model is what
# ``DEFAULT_MAX_CONCURRENT_REQUESTS`` below addresses (see the README).
DEFAULT_MAX_REQUEST_BODY_BYTES = 64 * 1024 * 1024

# Default ceiling on the number of requests concurrently in flight through
# this proxy at once, enforced by ``ProvenanceProxyServer.process_request``
# BEFORE a handler thread is even started (see that method for why a
# non-blocking ``BoundedSemaphore``, never a queue). ``ThreadingHTTPServer``
# has ``daemon_threads = True`` and otherwise starts one thread per accepted
# connection with no bound at all -- verified empirically: 300 concurrent
# connections produce 300 threads, and because ``daemon_threads = True``
# makes ``socketserver._Threads.append`` return early, the server does not
# even keep a reference to them (``server._threads`` stays at its initial
# ``_NoThreads()`` sentinel), so there was nothing to bound this on before.
#
# The real aggregate memory bound this proxy enforces is the PRODUCT of
# this and ``DEFAULT_MAX_REQUEST_BODY_BYTES`` -- 64 * 64 MiB = 4 GiB worst
# case with both defaults -- not either limit alone; an operator tuning
# only one of the two flags is tuning half the actual number, not the
# whole thing. A streaming/SSE response (see ``_respond_streamed``) holds
# its slot for the ENTIRE generation, not just while headers are read, so
# this is also an effective bound on concurrent generations in progress,
# not merely concurrent TCP connections.
DEFAULT_MAX_CONCURRENT_REQUESTS = 64

# Absolute wall-clock ceiling on ``_refuse_over_capacity``'s post-503 drain
# (see that method): a client that sends nothing at all, or a client that
# dribbles bytes one at a time forever, both cost the accept thread AT MOST
# this much, once, no matter how the client paces itself. See the drain
# loop's own comment for why this must be an absolute deadline rather than
# a per-``recv`` timeout, and why a well-behaved refused client's cost stays
# far below this number in practice.
_REFUSAL_DRAIN_DEADLINE_SECONDS = 0.05

# Minimum spacing, in monotonic seconds, between ``_log_capacity_refusal``
# ``stderr`` lines. This bounds ONLY the synchronous ``sys.stderr.write`` on
# ``_refuse_over_capacity``'s single accept thread -- it deliberately does
# NOT touch ``log_hook``, which still fires once per refusal with zero
# information lost (see ``_log_capacity_refusal`` below for why that split
# matters). Before the drain fix in ``_refuse_over_capacity`` removed its own
# accidental 0.2s-per-refusal cost, refusals -- and therefore these log
# writes -- could never happen faster than ~5/second, which incidentally
# capped this write's rate too. With that removed, an attacker who can keep
# the concurrency cap saturated can drive refusals (and therefore ~230-byte
# JSONL writes) as fast as the accept loop can refuse connections: a
# sustained disk-fill vector on the serve host, and, if ``stderr`` is ever a
# pipe whose reader stalls (not true today -- the launcher inherits rather
# than pipes ``stderr`` -- but not guaranteed forever), a synchronous write
# that blocks the ONE accept thread every other connection depends on,
# reintroducing the exact wedge class the drain fix above was written to
# remove.
_REFUSAL_LOG_MIN_INTERVAL_SECONDS = 1.0

# Fixed count of held slots a ``saturation_snapshot`` line names explicitly
# (the K OLDEST -- see ``ProvenanceProxyServer._maybe_emit_saturation_
# snapshot``). Fixed, never derived from ``max_concurrent_requests``, so
# the line's byte size is O(1) regardless of how high an operator sets the
# concurrency cap -- a full per-slot dump was REJECTED on exactly this
# ground (see the task predeclaration this increment implements,
# ``docs/task-inbox/2026-09-21-PREDECLARATION-proxy-saturation-snapshot-
# who-holds-the-slots.md``): an operator-settable cap makes an
# O(max_concurrent) line an attacker-triggered log-amplification vector.
_SATURATION_SNAPSHOT_TOP_K = 8

# Minimum spacing, in monotonic seconds, between ``saturation_snapshot``
# stderr lines -- a DEDICATED interval and a DEDICATED piece of rate-limit
# state (``ProvenanceProxyServer._saturation_last_emitted_monotonic``),
# never ``_REFUSAL_LOG_MIN_INTERVAL_SECONDS``/``_refusal_log_lock`` above.
# The two rate limits protect different emitters with different trigger
# conditions (a FAILED semaphore acquire vs. CROSSING a saturation mark);
# sharing their state would let one feature's burst silently consume the
# other's budget, or vice versa.
_SATURATION_SNAPSHOT_MIN_INTERVAL_SECONDS = 1.0


def _saturation_threshold(max_concurrent_requests: int) -> int:
    """``ceil(0.8 * max_concurrent_requests)``, floored at 1 -- the mark
    ``inflight`` must reach or cross for a ``saturation_snapshot`` to be
    considered (still subject to the rate limit above once it is).
    Computed in exact integer arithmetic, never ``0.8 * n`` in floating
    point, so the threshold cannot drift with float rounding: ``ceil(a /
    b)`` for positive integers is ``-(-a // b)``, since Python's ``//``
    floors -- negating both operands turns that floor into the ceiling of
    the original division.
    """
    return max(1, -(-4 * max_concurrent_requests // 5))


def _sanitize_header_value(value: object) -> str:
    return _NON_PRINTABLE_ASCII.sub("", str(value))


def _response_write_budget_exceeded_message() -> str:
    # A module-level function (not a constant string) so it always reads
    # the CURRENT value of ``_CLIENT_RESPONSE_MAX_BLOCKED_SECONDS`` --
    # tests monkeypatch that module attribute directly (see
    # ``ProxyConcurrencyLimitTests``), and a value baked in at import time
    # would silently go stale under that patch.
    return (
        "client did not drain the response within the "
        f"{_CLIENT_RESPONSE_MAX_BLOCKED_SECONDS}s cumulative blocked-write "
        "budget (_CLIENT_RESPONSE_MAX_BLOCKED_SECONDS) -- see that "
        "constant's own comment for why this is wall-clock blocked time, "
        "never a byte rate"
    )


def _connection_tokens(header_items) -> set:
    tokens = set()
    for name, value in header_items:
        if name.lower() != "connection":
            continue
        for token in value.split(","):
            token = token.strip().lower()
            if token:
                tokens.add(token)
    return tokens


def _strip_hop_by_hop(header_items, also_strip: Tuple[str, ...] = ()) -> List[Tuple[str, str]]:
    """``header_items``, minus every RFC 7230 hop-by-hop header, minus every
    token a ``Connection`` header in ``header_items`` itself names, minus
    any additionally named header (case-insensitive) in ``also_strip``.
    """
    named = _connection_tokens(header_items)
    extra = {name.lower() for name in also_strip}
    result = []
    for name, value in header_items:
        lowered = name.lower()
        if lowered in _HOP_BY_HOP or lowered in named or lowered in extra:
            continue
        result.append((name, value))
    return result


def build_provenance_headers(plan: dict) -> List[Tuple[str, str]]:
    """The fixed set of ``X-FastMLX-*`` headers computed ONCE from ``plan``
    (the same dict ``fastmlx_launch._run_serve`` builds) and attached to
    EVERY response the proxy sends, success or error alike.
    """
    admission = plan.get("admission")
    card = plan.get("card")
    card_id = card.get("id") if isinstance(card, dict) and card.get("id") else "none"
    fit_verdict = (plan.get("fit") or {}).get("verdict")
    residency = plan.get("residency")

    engine_build = plan.get("engineBuild") or {}
    build_status = engine_build.get("status")
    build_commit = engine_build.get("launch") or engine_build.get("card")
    build_value = build_status if build_status is not None else "unrecorded"
    if build_commit:
        build_value = f"{build_value}; commit={build_commit}"

    mtp = plan.get("mtp") or {}
    mtp_status = mtp.get("status")
    divergent = mtp.get("divergentPrompts")
    prompts = mtp.get("prompts")
    mtp_value = mtp_status
    if divergent is not None and prompts is not None:
        mtp_value = f"{mtp_status}; divergent={divergent}/{prompts}"

    headers = [
        ("X-FastMLX-Admission", admission),
        ("X-FastMLX-Card", card_id),
        ("X-FastMLX-Fit", fit_verdict),
        ("X-FastMLX-Residency", residency),
        ("X-FastMLX-Engine-Build", build_value),
        ("X-FastMLX-MTP", mtp_value),
    ]
    return [(name, _sanitize_header_value(value)) for name, value in headers]


def build_provenance_body(plan: dict) -> dict:
    """The JSON body ``GET /fastmlx/provenance`` answers with: fit verdict,
    card id, admission, residency, engine build, mtp and front -- deliberately
    NEVER the raw ``argv`` (it names the resolved engine binary and every
    passthrough argument) and never a raw model path (fit fields / card
    contents are not included wholesale for the same reason).
    """
    card = plan.get("card")
    card_id = card.get("id") if isinstance(card, dict) and card.get("id") else None
    fit = plan.get("fit") or {}
    return {
        "fit": {"verdict": fit.get("verdict")},
        "card": card_id,
        "admission": plan.get("admission"),
        "residency": plan.get("residency"),
        "engineBuild": plan.get("engineBuild"),
        "mtp": plan.get("mtp"),
        "front": plan.get("front"),
    }


class _DeadlineBoundRfile:
    """Wraps the real ``self.rfile`` (an ``io.BufferedReader`` over the
    client socket) so every underlying read is bounded by the time
    REMAINING before an absolute, per-connection deadline -- re-derived
    fresh before each individual read, exactly the "compute the deadline
    once, re-derive each timeout from what is left" idiom
    ``_refuse_over_capacity``'s drain loop uses -- rather than a single
    ``socket.settimeout`` value set once and reused for however many
    underlying reads a caller's ``readline``/``read`` call happens to
    need. That distinction is the entire defect this class exists to fix:
    a timeout value set once and left in place lets a client that
    dribbles one byte per underlying ``recv`` make EVERY one of them
    individually succeed, however many there are and however long that
    takes in total, because a per-call timeout is not a wall-clock budget
    for a compound operation that loops internally.

    ``readline``/``read`` are implemented here on top of ``read1`` --
    deliberately never the base object's own ``readline``/``read``,
    which would loop internally, out of this class's control, doing
    however many raw reads a slowly-arriving line or body needs under a
    single stale timeout value (the same mistake as above, one level
    down). ``io.BufferedReader.read1(size)`` does AT MOST ONE underlying
    raw read when its own buffer is empty (or zero, serving
    already-buffered bytes for free without touching the socket at all)
    -- which is what lets this class re-derive the timeout before EVERY
    such underlying read.

    Every byte actually received extends the deadline by
    ``1 / _CLIENT_REQUEST_MIN_BYTES_PER_SECOND`` seconds, clamped to
    never exceed an absolute hard ceiling computed once, from this
    connection's own start (see ``_CLIENT_REQUEST_HARD_DEADLINE_SECONDS``
    above). Once the deadline has passed, the NEXT read (never a
    currently-blocked one -- there is none, by construction) raises
    ``TimeoutError`` -- the same exception
    ``BaseHTTPRequestHandler.handle_one_request`` already has an
    ``except TimeoutError`` clause for, ending the request and releasing
    this proxy's concurrency slot exactly as the plain idle timeout does
    today for the shapes it already catches.

    The client socket's own timeout is armed ONLY for the duration of
    each individual ``read1`` call, and restored to the plain
    ``_CLIENT_IDLE_TIMEOUT_SECONDS`` backstop immediately afterward in a
    ``finally`` -- never left at the (possibly very small) request-receive
    deadline remainder. That is what keeps RESPONSE writes (and any read
    this class does not itself gate) unbound by the deadline above while
    still not literally unbounded forever: a streaming SSE response
    legitimately holds its slot for the whole generation (see
    ``test_streaming_sse_response_holds_its_slot_for_the_whole_
    generation``) and must never be cut off by this deadline.

    Only the methods this proxy's own request-parsing path actually uses
    are delegated: ``readline`` (``BaseHTTPRequestHandler.handle_one_
    request``'s request-line read, and ``http.client.parse_headers``'s
    header reads), ``read`` (this file's own
    ``self.rfile.read(content_length)`` body read), ``readinto`` (kept
    for any future/stdlib caller that might use it, though none in this
    file's own call graph does today), and ``close``. Not a
    ``socket.SocketIO`` subclass, deliberately: a thin delegating wrapper
    around the real buffered reader is clearer than overriding raw-socket
    internals.
    """

    def __init__(self, raw, connection: socket.socket, started_monotonic: float):
        self._raw = raw
        self._connection = connection
        self._pushback = b""
        # Computed ONCE, from this connection's own start -- no amount of
        # extension below can ever push the deadline past this.
        self._hard_deadline = started_monotonic + _CLIENT_REQUEST_HARD_DEADLINE_SECONDS
        self._deadline = min(
            started_monotonic + _CLIENT_REQUEST_INITIAL_SECONDS, self._hard_deadline
        )

    def _extend_deadline(self, bytes_received: int) -> None:
        if bytes_received <= 0:
            return
        extended = self._deadline + (bytes_received / _CLIENT_REQUEST_MIN_BYTES_PER_SECOND)
        self._deadline = min(extended, self._hard_deadline)

    def _raw_read1(self, size: int) -> bytes:
        """Exactly one underlying read (or zero, if already-buffered
        bytes satisfy it), bounded by the time remaining before the
        CURRENT deadline -- re-derived fresh on every call, never a
        value computed by an earlier call and reused.
        """
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                "client did not finish sending its request (request "
                "line/headers/body) within the request-receive deadline"
            )
        self._connection.settimeout(remaining)
        try:
            chunk = self._raw.read1(size)
        finally:
            # See the class docstring: restored to the plain backstop,
            # never left at the deadline remainder, so writes that follow
            # a read are never bound by it.
            self._connection.settimeout(_CLIENT_IDLE_TIMEOUT_SECONDS)
        self._extend_deadline(len(chunk))
        return chunk

    def _next_chunk(self, size: int) -> bytes:
        if self._pushback:
            chunk, self._pushback = self._pushback[:size], self._pushback[size:]
            return chunk
        return self._raw_read1(size)

    def readline(self, size: int = -1) -> bytes:
        limit = size if isinstance(size, int) and size >= 0 else (1 << 20)
        line = bytearray()
        while len(line) < limit:
            chunk = self._next_chunk(min(8192, limit - len(line)))
            if not chunk:
                break  # EOF: the client closed before sending a newline.
            newline_at = chunk.find(b"\n")
            if newline_at == -1:
                line += chunk
                continue
            line += chunk[: newline_at + 1]
            leftover = chunk[newline_at + 1:]
            if leftover:
                # Belongs to whatever is read NEXT (the next header line,
                # or the body) -- held here rather than discarded.
                self._pushback = leftover + self._pushback
            break
        return bytes(line)

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            # Never used on this handler's client-facing rfile -- the
            # only caller in this file is ``self.rfile.read(content_
            # length)`` with a known, non-negative length (see
            # ``_dispatch``). An unbounded read here would defeat the
            # deadline this class exists to enforce, so it is refused
            # outright rather than silently reading until EOF.
            raise ValueError(
                "_DeadlineBoundRfile.read() requires an explicit, "
                "non-negative size"
            )
        data = bytearray()
        while len(data) < size:
            chunk = self._next_chunk(min(8192, size - len(data)))
            if not chunk:
                break  # EOF before `size` bytes arrived.
            data += chunk
        return bytes(data)

    def readinto(self, b) -> int:
        chunk = self._next_chunk(len(b))
        n = len(chunk)
        b[:n] = chunk
        return n

    def close(self) -> None:
        self._raw.close()


class ProvenanceProxyHandler(BaseHTTPRequestHandler):
    """Forwards every method/path/body to ``self.server``'s upstream
    unchanged, stamping the provenance headers onto every response.
    """

    # HTTP/1.0 on purpose: every response here is either length-delimited
    # (Content-Length, sent as-is when upstream provides one) or otherwise
    # close-delimited -- the client reads until the socket closes. HTTP/1.1
    # would make http.client's own `_check_close` ASSUME a persistent
    # connection (RFC 7230) unless this handler also emitted an explicit
    # `Connection: close` header; HTTP/1.0's opposite default ("closes
    # unless Keep-Alive is negotiated") gets the same close-delimited
    # framing without that extra header, and this proxy never keeps a
    # connection open across requests anyway (see `close_connection = True`
    # below on every response path).
    protocol_version = "HTTP/1.0"

    # ``BaseHTTPRequestHandler.setup()`` applies this straight to the
    # request socket (``self.connection.settimeout(self.timeout)``) the
    # instant the socket is set up -- see ``_CLIENT_IDLE_TIMEOUT_SECONDS``
    # above for why this value ALONE does not, and never did, bound how
    # long the handler waits on the CLIENT for the request line/headers/
    # body (a slow-but-alive upstream is a separate concern regardless --
    # ``upstream_read_timeout`` on the server, unbounded by default). The
    # actual per-request deadline is installed by ``setup()`` below,
    # immediately after calling ``super().setup()``, by wrapping
    # ``self.rfile`` in ``_DeadlineBoundRfile`` (see
    # ``_CLIENT_REQUEST_INITIAL_SECONDS`` / ``_CLIENT_REQUEST_MIN_BYTES_
    # PER_SECOND`` / ``_CLIENT_REQUEST_HARD_DEADLINE_SECONDS`` above).
    timeout = _CLIENT_IDLE_TIMEOUT_SECONDS

    # Identifies PROXY-GENERATED responses only (a 502/400/provenance
    # reply built with ``send_response()``, which calls this) -- never the
    # Python version, and never used for a RELAYED upstream response
    # (those use ``send_response_only`` instead, see ``_respond_buffered``
    # / ``_respond_streamed``, so the upstream's own Server/Date headers
    # pass through untouched rather than being duplicated by these).
    server_version = "fastmlx-proxy"
    sys_version = ""

    def version_string(self) -> str:
        return self.server_version

    def setup(self) -> None:
        """Identical to the base class, PLUS installing
        ``_DeadlineBoundRfile`` around ``self.rfile`` immediately
        afterward -- this is what actually bounds RECEIVING a request
        (request line, headers, body) by wall-clock time (see
        ``_CLIENT_REQUEST_INITIAL_SECONDS`` and friends above), rather
        than the per-``recv`` ``self.timeout`` the base class alone
        applies (see the comment on that class attribute for why that is
        not sufficient by itself).

        ``protocol_version = "HTTP/1.0"`` together with
        ``close_connection = True`` on every response path in this
        handler (see those definitions) means exactly one request is
        ever handled per accepted connection here, so arming the deadline
        once per connection, in ``setup()``, is equivalent to arming it
        once per request -- there is no keep-alive iteration on this
        handler that could reuse an already-expired deadline from an
        earlier request on the same connection.
        """
        super().setup()
        self.rfile = _DeadlineBoundRfile(self.rfile, self.connection, time.monotonic())

    def parse_request(self) -> bool:
        """Identical to the base class, PLUS answering the ``Expect:
        100-continue`` handshake the base class itself never reaches here.

        ``BaseHTTPRequestHandler.parse_request`` gates its own call to
        ``self.handle_expect_100()`` on THREE conditions (see stdlib
        ``http/server.py``): the ``Expect`` header itself, ``self.
        protocol_version >= "HTTP/1.1"``, and ``self.request_version >=
        "HTTP/1.1"``. This override reproduces the Expect check and the
        ``request_version`` check, but DROPS the ``protocol_version`` one:

        - ``protocol_version`` (dropped, deliberately): pinned to
          ``"HTTP/1.0"`` above for the response-framing reason explained in
          the comment on that assignment (see ``:170-180``), which makes
          the base class's gate permanently false and its own Expect
          handling simply dead code on this handler, for every request,
          regardless of what the client sent. A client that sends
          ``Expect: 100-continue`` (curl does this by default for any
          request with a body) then waits for the interim response that
          never arrives, and eventually times out and sends the body
          anyway.
        - ``request_version`` (kept): this is the CLIENT's own declared
          HTTP version, not this server's framing pin, and RFC 7231 5.1.1
          says a server MUST NOT send a ``100 (Continue)`` response to a
          request from an HTTP/1.0 (or earlier) client -- such a client is
          not required to understand an interim 1xx response and could
          mis-frame it as the final one. Dropping ``protocol_version``
          above must not also silently drop this independent,
          client-version-based precondition.

        This override calls the base implementation first (unchanged:
        request-line parsing, header parsing, ``Connection`` handling), then
        re-examines the same ``Expect``/``request_version`` pair WITHOUT the
        ``protocol_version`` gate and answers it via
        ``handle_expect_100()`` below. This happens before ``_dispatch``
        reads the body (``self.rfile.read(content_length)``), which is what
        makes the interim response actually useful rather than a race.
        """
        if not super().parse_request():
            return False
        if (self.headers.get("Expect", "").lower() == "100-continue"
                and self.request_version >= "HTTP/1.1"):
            if not self.handle_expect_100():
                return False
        return True

    def handle_expect_100(self) -> bool:
        """Writes the ``100 Continue`` interim response and FLUSHES it,
        then returns ``True`` to tell ``parse_request`` (see override
        above) to keep processing the request.

        This is safe under the ``protocol_version = "HTTP/1.0"`` pin
        (see the comment on that assignment) precisely because an interim
        1xx response is not "the response" that pin's reasoning is about:
        it carries no ``Content-Length``, negotiates no persistence, and
        is always followed by exactly one real, final response on the
        same connection -- the framing contract described there is
        entirely about that final response, which this method never
        touches (contrast with the anti-regression test in
        ``scripts/tests/test_fastmlx_proxy.py`` pinning that the final
        response is byte-identical whether or not this method ran).
        Using ``self.protocol_version`` for the FINAL response is what
        keeps close-delimited framing simple; the interim line's own
        version token is independent of that and is pinned to
        ``HTTP/1.1`` instead, because RFC 7231 5.1.1 ties ``100
        (Continue)`` to HTTP/1.1 semantics -- a client is only supposed to
        send ``Expect: 100-continue`` when it can handle an HTTP/1.1
        response, so ``HTTP/1.1 100 Continue`` is the correct token for
        THIS line regardless of what the final response is framed as.

        The explicit ``self.wfile.flush()`` below matches this proxy's own
        convention in ``_respond_streamed`` (also flushed after every
        ``write``): ``BaseHTTPRequestHandler.end_headers()`` /
        ``flush_headers()`` write the buffered header bytes to ``wfile``
        but never call ``flush()`` themselves, unlike
        ``handle_one_request()``'s own trailing flush for a NORMAL
        response. This handler's ``wfile`` happens to be unbuffered under
        CPython's stdlib defaults (``StreamRequestHandler.wbufsize == 0``,
        never overridden here), so ``write()`` alone already reaches the
        socket immediately today -- the explicit flush is a defensive,
        essentially free belt-and-suspenders call that keeps this method
        correct even if that buffering default ever changed, not a fix for
        an observed buffering delay in THIS stdlib version (confirmed by
        mutation testing: removing it alone does not reproduce a hang).
        """
        try:
            self.wfile.write(b"HTTP/1.1 100 Continue\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False
        return True

    def log_message(self, *args, **kwargs) -> None:  # noqa: D401
        # The stdlib default writes an access-log line straight to stderr;
        # this proxy does its own structured JSONL logging in `_log`
        # instead (see module docstring), so the default is silenced here
        # rather than left to interleave with it.
        pass

    # -- dispatch ---------------------------------------------------------
    def _dispatch(self) -> None:
        request_id = uuid.uuid4().hex
        started = time.monotonic()
        parsed = urlsplit(self.path)
        path_only = parsed.path

        if self.command == "GET" and path_only == "/fastmlx/provenance":
            bytes_out = self._serve_provenance(request_id)
            self._log(request_id, self.command, path_only, 200, 0.0, bytes_out, False)
            return

        transfer_encoding = self.headers.get("Transfer-Encoding", "")
        if "chunked" in transfer_encoding.lower():
            bytes_out = self._send_json_error(
                411,
                "chunked request bodies are not supported by this proxy; send a "
                "Content-Length request body instead",
                "chunked_request_unsupported",
                request_id,
            )
            self._log(request_id, self.command, path_only, 411, 0.0, bytes_out, False)
            return

        # A negative or non-integer Content-Length is answered 400 without
        # ever attempting ``self.rfile.read(...)`` on it: a negative count
        # (``read(-1)``) means "read until EOF", which would hang a
        # persistent client connection rather than fail fast (see L6 in
        # the front-mode review).
        content_length_header = self.headers.get("Content-Length")
        content_length = 0
        if content_length_header is not None:
            try:
                content_length = int(content_length_header)
            except ValueError:
                content_length = None
            if content_length is None or content_length < 0:
                bytes_out = self._send_json_error(
                    400,
                    f"invalid Content-Length: {content_length_header!r}",
                    "invalid_content_length",
                    request_id,
                )
                self._log(request_id, self.command, path_only, 400, 0.0, bytes_out, False, None)
                return

        # Refused on the DECLARED size ALONE, before ``self.rfile.read(...)``
        # below touches a single body byte -- reading first and rejecting
        # after would already have paid the exact memory cost this check
        # exists to avoid (see ``DEFAULT_MAX_REQUEST_BODY_BYTES`` above for
        # why an unbounded read matters on this host). ``content_length`` is
        # already known non-negative here (the block above refused a
        # negative/non-numeric header first), so this is a plain compare of
        # two non-negative integers, no sign surprises.
        if content_length > self.server.max_request_body_bytes:
            bytes_out = self._send_json_error(
                413,
                f"request body of {content_length} bytes exceeds the configured "
                f"limit of {self.server.max_request_body_bytes} bytes",
                "request_body_too_large",
                request_id,
            )
            self._log(request_id, self.command, path_only, 413, 0.0, bytes_out, False, None)
            return
        body = self.rfile.read(content_length) if content_length else b""

        # Sent via ``putrequest``/``putheader`` below (never a ``dict(...)``,
        # which would collapse duplicate header names into one) -- ``Host``
        # is excluded here because ``putrequest``'s own ``skip_host=False``
        # regenerates it pointed at the UPSTREAM host:port, never the
        # client's original value (see L4(a) in the front-mode review).
        outbound_headers = _strip_hop_by_hop(list(self.headers.items()), also_strip=("host",))

        upstream_ms = 0.0
        try:
            # The constructor's own ``timeout`` bounds ONLY the CONNECT below
            # (``http.client`` also applies it to every socket operation
            # thereafter unless overridden) -- a non-streamed response whose
            # headers take a while, or an SSE reply with a multi-second gap
            # between events, must never be treated as a dead upstream.
            # ``connection.sock.settimeout(...)`` right after ``connect()``
            # re-bounds every READ that follows to ``upstream_read_timeout``
            # instead (``None`` by default: unbounded, matching a normal
            # reverse proxy's behavior for a slow-but-alive backend).
            connection = http_client.HTTPConnection(
                self.server.upstream_host,
                self.server.upstream_port,
                timeout=self.server.upstream_connect_timeout,
            )
            connection.connect()
            connection.sock.settimeout(self.server.upstream_read_timeout)
            connection.putrequest(self.command, self.path, skip_accept_encoding=True)
            header_names_lower = {name.lower() for name, _ in outbound_headers}
            if "accept-encoding" not in header_names_lower:
                # http.client's own default, replicated here since
                # ``skip_accept_encoding=True`` above suppresses it (that
                # flag is the only way to avoid a SECOND Accept-Encoding
                # header when the client already sent one).
                connection.putheader("Accept-Encoding", "identity")
            for name, value in outbound_headers:
                connection.putheader(name, value)
            connection.endheaders(body if body else None)
            headers_started = time.monotonic()
            response = connection.getresponse()
            upstream_ms = (time.monotonic() - headers_started) * 1000.0
        except (OSError, http_client.HTTPException) as exc:
            bytes_out = self._send_json_error(
                502, f"upstream unavailable: {exc}", "upstream_unavailable", request_id
            )
            self._log(request_id, self.command, path_only, 502, upstream_ms, bytes_out, False, None)
            return

        status = response.status
        response_headers = [
            (name, value)
            for name, value in _strip_hop_by_hop(response.getheaders())
            if not name.lower().startswith(_PROVENANCE_HEADER_PREFIX)
        ]
        response_content_length_header = response.getheader("Content-Length")
        # An unparseable Content-Length (a malformed/hand-rolled upstream) is
        # treated exactly like a missing one -- the streamed, close-
        # delimited path -- rather than raising inside the handler and
        # leaving the client with no response at all.
        parsed_content_length: Optional[int] = None
        if response_content_length_header is not None:
            try:
                parsed_content_length = int(response_content_length_header)
            except ValueError:
                parsed_content_length = None

        try:
            if parsed_content_length is not None:
                bytes_out, response_status, error = self._respond_buffered(
                    status, response_headers, response, parsed_content_length, request_id
                )
                streamed = False
            else:
                bytes_out, streamed, error = self._respond_streamed(
                    status, response_headers, response, connection, request_id
                )
                response_status = status
        finally:
            connection.close()

        self._log(
            request_id, self.command, path_only, response_status, upstream_ms, bytes_out,
            streamed, error,
        )

    do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = do_HEAD = do_PATCH = _dispatch

    # -- response helpers ---------------------------------------------------
    def _provenance_headers_with_request_id(self, request_id: str) -> List[Tuple[str, str]]:
        return list(self.server.provenance_headers) + [
            ("X-FastMLX-Request-Id", _sanitize_header_value(request_id))
        ]

    def _serve_provenance(self, request_id: str) -> int:
        body = json.dumps(self.server.provenance_body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in self._provenance_headers_with_request_id(request_id):
            self.send_header(name, value)
        self.close_connection = True
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        return len(body)

    def _send_json_error(self, status: int, message: str, error_type: str, request_id: str) -> int:
        body = json.dumps({"error": {"message": message, "type": error_type}}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in self._provenance_headers_with_request_id(request_id):
            self.send_header(name, value)
        self.close_connection = True
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        return len(body)

    def _write_response_chunk(self, data: bytes, budget_remaining: float) -> float:
        """Writes ``data`` (then flushes) to the client, with
        ``self.connection``'s timeout re-derived from ``budget_remaining``
        immediately before the write and restored to the plain
        ``_CLIENT_IDLE_TIMEOUT_SECONDS`` backstop immediately after -- the
        same idiom ``_DeadlineBoundRfile._raw_read1`` uses for reads (see
        there), applied to writes for the reason documented on
        ``_CLIENT_RESPONSE_MAX_BLOCKED_SECONDS`` above.

        Returns the wall-clock seconds this call spent inside
        ``write``/``flush`` -- the caller accumulates this across every
        chunk of one response to enforce the CUMULATIVE budget, not a
        per-chunk one. Raises ``socket.timeout``/``TimeoutError`` if the
        client did not drain within ``budget_remaining``, or
        ``BrokenPipeError``/``ConnectionResetError``/``OSError`` if the
        client vanished outright -- both are left for the caller to
        handle (they mean different things: one is a slow client to
        abort with an error, the other is an ordinary vanished client),
        so neither is swallowed here.
        """
        # Clamped to a small POSITIVE floor, never passed through raw.
        # ``socket.settimeout`` raises ``ValueError`` on a negative value
        # and switches the socket to NON-BLOCKING mode on exactly 0.0 --
        # and neither is caught by this method's callers, which handle
        # ``socket.timeout`` and the vanished-client ``OSError``s only. A
        # ``ValueError`` here would escape as an unhandled exception in
        # the handler thread. ``_respond_streamed``'s own ``budget_
        # remaining <= 0`` pre-check already prevents that call reaching
        # here, but a guard whose removal crashes the thread should not
        # be the only thing standing between the two (a mutation run
        # confirmed that pre-check has no independent test coverage on
        # its own; see this cycle's record).
        self.connection.settimeout(max(budget_remaining, 0.001))
        started = time.monotonic()
        try:
            self.wfile.write(data)
            self.wfile.flush()
        finally:
            # Restored the instant this write is done, never left at the
            # budget remainder -- nothing that runs after this write (the
            # next upstream read, the next chunk's own budget-derived
            # timeout) may ever inherit it.
            self.connection.settimeout(_CLIENT_IDLE_TIMEOUT_SECONDS)
        return time.monotonic() - started

    def _respond_buffered(
        self,
        status: int,
        response_headers: List[Tuple[str, str]],
        response: http_client.HTTPResponse,
        length: int,
        request_id: str,
    ) -> Tuple[int, int, Optional[str]]:
        # The upstream read happens BEFORE any header is sent to the
        # client: a truncated/dropped upstream (Content-Length promised
        # more than it delivered) must 502, never a "complete" response
        # missing its tail (see M2 in the front-mode review).
        #
        # ``HTTPResponse.read(amt)`` does NOT raise when the underlying
        # socket hits EOF before ``amt`` bytes arrive -- unlike the
        # exact-count ``_safe_read`` its no-argument ``read()`` uses, the
        # explicit-``amt`` form has raw-``io.BufferedReader`` read-up-to
        # semantics and just returns however many bytes it got. A length
        # check after the read (not only the exception handler below,
        # which still covers a genuinely malformed/chunked upstream) is
        # what actually catches THIS truncation.
        try:
            body = response.read(length)
        except (http_client.HTTPException, OSError) as exc:
            bytes_out = self._send_json_error(
                502,
                f"upstream response body truncated: {exc}",
                "upstream_body_truncated",
                request_id,
            )
            return bytes_out, 502, f"upstream_body_truncated: {exc}"
        # A HEAD response never carries a body even when Content-Length
        # names one (http.client already zeroes ``response.length`` for a
        # HEAD request, so ``body`` is always empty here) -- that is
        # correct, never a truncation.
        if self.command != "HEAD" and len(body) < length:
            detail = f"expected {length} bytes, got {len(body)}"
            bytes_out = self._send_json_error(
                502,
                f"upstream response body truncated: {detail}",
                "upstream_body_truncated",
                request_id,
            )
            return bytes_out, 502, f"upstream_body_truncated: {detail}"

        # ``send_response_only``, not ``send_response``: this is a RELAYED
        # response, so the upstream's own Server/Date headers (already in
        # ``response_headers``, never stripped) pass through untouched
        # instead of gaining a second pair from this handler (see L4(b) in
        # the front-mode review).
        self.send_response_only(status)
        for name, value in response_headers:
            self.send_header(name, value)
        for name, value in self._provenance_headers_with_request_id(request_id):
            self.send_header(name, value)
        self.close_connection = True
        self.end_headers()
        # A single write, so it gets the WHOLE budget to itself (never a
        # per-chunk fraction of it -- there is only one chunk here).
        #
        # Ordering matters on the except clauses below: ``socket.timeout``
        # IS a subclass of ``OSError`` (``TimeoutError`` too, since
        # Python 3.10 made ``socket.timeout`` an alias of it), so it MUST
        # be caught by its own clause first -- if the generic
        # ``(BrokenPipeError, ConnectionResetError, OSError)`` clause ran
        # first it would swallow a genuine blocked-write-budget timeout
        # and report it as an ordinary successful (or silently vanished)
        # response, defeating this whole fix.
        error: Optional[str] = None
        bytes_out = len(body)
        try:
            self._write_response_chunk(body, _CLIENT_RESPONSE_MAX_BLOCKED_SECONDS)
        except (socket.timeout, TimeoutError):
            error = _response_write_budget_exceeded_message()
            bytes_out = 0
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            # The client vanished mid-write. ``bytes_out`` was set to
            # ``len(body)`` BEFORE the write for the happy path, so it must
            # fall back to 0 here, matching ``_respond_streamed``'s
            # count-only-after-success convention -- otherwise the single
            # caller logs a failed delivery as a clean full-body success.
            error = f"client_write_failed: {exc}"
            bytes_out = 0
            # Deliberately no ``_abort_client_connection()``: the exception
            # proves the socket already failed, unlike the timeout branch
            # where the connection is still alive and the RST is what makes
            # the abort visible to the client. ``_respond_streamed``'s own
            # vanished-client branch draws the same line.
            return bytes_out, status, error
        if error is not None:
            self._abort_client_connection()
        return bytes_out, status, error

    def _abort_client_connection(self) -> None:
        """Force a TCP RST to the client (SO_LINGER on, linger=0, then
        close) instead of the clean FIN a normal close sends -- used when
        the upstream fails AFTER headers already told the client "200 OK":
        the client must see a connection error, never a clean (silently
        truncated) EOF that looks like the body simply ended (see M2 in
        the front-mode review).
        """
        try:
            self.connection.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
        except OSError:
            pass
        self.close_connection = True
        try:
            self.connection.close()
        except OSError:
            pass

    def _respond_streamed(
        self,
        status: int,
        response_headers: List[Tuple[str, str]],
        response: http_client.HTTPResponse,
        connection: http_client.HTTPConnection,
        request_id: str,
    ) -> Tuple[int, bool, Optional[str]]:
        # No Content-Length from upstream (SSE, chunked, or plain
        # close-delimited): relayed close-delimited to the client too --
        # this handler's own protocol_version already closes the
        # connection after one response, so the client's own read-until-
        # EOF is exactly how it learns the body ended.
        #
        # ``send_response_only``, not ``send_response``: see
        # ``_respond_buffered`` above -- the upstream's own Server/Date
        # headers pass through untouched.
        self.send_response_only(status)
        for name, value in response_headers:
            self.send_header(name, value)
        for name, value in self._provenance_headers_with_request_id(request_id):
            self.send_header(name, value)
        self.close_connection = True
        self.end_headers()

        bytes_out = 0
        streamed = False
        error: Optional[str] = None
        # The CUMULATIVE blocked-write budget for the WHOLE response, not
        # a fresh one per chunk (see ``_CLIENT_RESPONSE_MAX_BLOCKED_
        # SECONDS`` above for why a per-chunk budget would not actually
        # bound anything: a client dribbling its reads just fast enough
        # to keep any ONE write under the budget could still hold its
        # slot forever, exactly the request-side defect this mirrors).
        budget_remaining = _CLIENT_RESPONSE_MAX_BLOCKED_SECONDS
        while True:
            try:
                chunk = response.read1(_UPSTREAM_READ_CHUNK_BYTES)
            except (BrokenPipeError, ConnectionResetError, OSError, http_client.HTTPException) as exc:
                # The upstream failed (or an ``IncompleteRead`` proves it
                # closed mid-chunk) AFTER this handler already sent a
                # success status: a clean ``break`` here would let the
                # client's own read-until-EOF believe the body ended
                # cleanly, so the client connection is aborted with a
                # reset instead (see M2 in the front-mode review).
                error = f"upstream read failed: {exc}"
                self._abort_client_connection()
                break
            if not chunk:
                break
            if budget_remaining <= 0:
                # The budget was already exhausted by an EARLIER chunk in
                # this same response (see the ``except`` below for the
                # other way this budget is spent: a single write blocking
                # past what remained). Either way, no further write is
                # attempted -- the client already proved it cannot keep
                # up within the budget.
                error = _response_write_budget_exceeded_message()
                self._abort_client_connection()
                break
            try:
                budget_remaining -= self._write_response_chunk(chunk, budget_remaining)
            except (socket.timeout, TimeoutError):
                # Ordering matters here exactly as in ``_respond_buffered``
                # (see its own comment): ``socket.timeout``/``TimeoutError``
                # must be caught BEFORE the generic vanished-client clause
                # below, which is also an ``OSError`` superclass match and
                # would otherwise silently report this budget-exhaustion
                # abort as an ordinary vanished-client disconnect.
                error = _response_write_budget_exceeded_message()
                self._abort_client_connection()
                break
            except (BrokenPipeError, ConnectionResetError, OSError):
                # The client vanished mid-stream: close the upstream
                # connection promptly rather than draining it to nowhere.
                connection.close()
                break
            bytes_out += len(chunk)
            streamed = True
        return bytes_out, streamed, error

    def _log(
        self,
        request_id: str,
        method: str,
        path: str,
        status: int,
        upstream_ms_to_headers: float,
        bytes_out: int,
        streamed: bool,
        error: Optional[str] = None,
    ) -> None:
        # One JSONL line per request, to stderr: never a header or body
        # value (the Authorization header value is forwarded, above, but
        # never appears here). Written with a SINGLE ``write`` call (then
        # flushed) rather than ``print()``'s separate string/end writes --
        # under ``ThreadingHTTPServer`` two concurrent requests' calls
        # could otherwise interleave mid-line (see L8 in the front-mode
        # review).
        entry = {
            "ts": time.time(),
            "request_id": request_id,
            "method": method,
            "path": path,
            "status": status,
            "upstream_ms_to_headers": round(upstream_ms_to_headers, 3),
            "bytes_out": bytes_out,
            "streamed": streamed,
        }
        if error is not None:
            entry["error"] = error
        server = getattr(self, "server", None)
        # The LEADING indicator entirely absent before this field: unlike
        # ``_log_capacity_refusal``'s ``inflight`` (only ever seen once a
        # 503 has already happened), this fires on every ordinary request
        # too, so an operator watching stderr sees "60/64 and climbing"
        # BEFORE any refusal exists. Two semantics that make the number
        # read wrong if forgotten:
        #   (1) it COUNTS the request being logged -- this method runs
        #       before ``ProvenanceProxyServer.process_request_thread``'s
        #       own ``finally`` decrements the counter for this same
        #       request -- so a single sequential request logs
        #       ``inflight: 1``, not 0.
        #   (2) it is sampled at response COMPLETION, not arrival -- for a
        #       long streamed (SSE) response this call happens at the END
        #       of that whole generation, so the value describes "what
        #       else was in flight when this response finished", not
        #       "when this request showed up".
        # ``None`` if ``server``/the attribute is unavailable (mirrors the
        # ``log_hook`` lookup just below). stderr-only, like every other
        # field this feature adds -- never a client-visible header or body.
        entry["inflight"] = getattr(server, "inflight_requests", None) if server is not None else None
        line = json.dumps(entry)
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
        log_hook = getattr(server, "log_hook", None) if server is not None else None
        if log_hook is not None:
            log_hook(entry)


class ProvenanceProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: Tuple[str, int],
        handler_cls,
        upstream_host: str,
        upstream_port: int,
        plan: dict,
        upstream_connect_timeout: float = _UPSTREAM_CONNECT_TIMEOUT_SECONDS,
        upstream_read_timeout: Optional[float] = None,
        max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
        max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
        log_hook: Optional[Callable[[dict], None]] = None,
    ):
        super().__init__(server_address, handler_cls)
        self.upstream_host = upstream_host
        self.upstream_port = upstream_port
        self.plan = plan
        # ``upstream_connect_timeout`` bounds only the TCP connect (default
        # 10s); ``upstream_read_timeout`` bounds every read that follows and
        # defaults to ``None`` (unbounded) -- a thinking-mode completion or a
        # long-context prefill can take far longer than any reasonable
        # connect bound before the first byte comes back, and this proxy has
        # no basis of its own for guessing a shorter number.
        self.upstream_connect_timeout = upstream_connect_timeout
        self.upstream_read_timeout = upstream_read_timeout
        # Read by ``_dispatch`` BEFORE it reads any body byte -- see
        # ``DEFAULT_MAX_REQUEST_BODY_BYTES`` for why this exists and why 64
        # MiB is the default a caller (``fastmlx_launch.py``'s
        # ``--front-max-body-bytes``) can override.
        self.max_request_body_bytes = max_request_body_bytes
        self.provenance_headers = build_provenance_headers(plan)
        self.provenance_body = build_provenance_body(plan)
        # Called synchronously, once per request, right after the JSONL log
        # line is written to stderr -- an optional test-only hook so a
        # caller can wait deterministically for a specific request's log
        # line instead of polling stderr on a sleep loop. ``None`` by
        # default: production callers never set this.
        self.log_hook = log_hook

        # See ``DEFAULT_MAX_CONCURRENT_REQUESTS`` above for what this
        # bounds and why. ``BoundedSemaphore``, not ``Semaphore``: a
        # mismatched acquire/release across the handler-thread exit paths
        # in ``process_request``/``process_request_thread`` below would
        # otherwise silently widen the cap past ``max_concurrent_requests``
        # forever instead of raising ``ValueError`` at the first
        # over-release, which is the only way such a bug would ever be
        # noticed.
        self.max_concurrent_requests = max_concurrent_requests
        self._slots = threading.BoundedSemaphore(max_concurrent_requests)
        # Guards both counters below -- read by tests (``peak_inflight_
        # requests`` is the anti-vacuity check that the cap was actually
        # REACHED, not merely never exceeded) and written from whichever
        # thread currently owns a slot. INVARIANT: nothing may ever block
        # while holding this lock -- it is a leaf lock, and every critical
        # section that takes it (here and in ``process_request``/
        # ``process_request_thread`` below) is 2-4 arithmetic statements,
        # no I/O, no nested lock acquisition. The moment an edit puts I/O
        # (a log write, a network call, anything that can stall) under
        # this lock, the single accept thread gains its first real wedge:
        # every other connection's admission decision depends on
        # ``process_request`` returning promptly, and that method takes
        # this same lock on every accepted request.
        self._inflight_lock = threading.Lock()
        self.inflight_requests = 0
        self.peak_inflight_requests = 0

        # Guards the rate-limit state for ``_log_capacity_refusal``'s
        # ``stderr`` emission (see ``_REFUSAL_LOG_MIN_INTERVAL_SECONDS``
        # above). ``_refusal_log_last_emitted_monotonic`` starts at ``None``,
        # not ``time.monotonic()`` at construction -- an operator must see
        # the FIRST refusal immediately, not have it silently swallowed
        # because it happened to land inside the very first interval after
        # the server came up.
        self._refusal_log_lock = threading.Lock()
        self._refusal_log_last_emitted_monotonic: Optional[float] = None
        self._refusal_log_suppressed_since_last_emit = 0
        # Monotonic count of EVERY refusal since this server started,
        # incremented once per refusal regardless of whether that
        # refusal's own line is emitted to stderr (see
        # ``_REFUSAL_LOG_MIN_INTERVAL_SECONDS`` above) -- unlike
        # ``_refusal_log_suppressed_since_last_emit``, which RESETS on
        # every emitted line, this never resets, so it is the only field
        # that can recover the true cumulative refusal count if a stderr
        # line is ever lost or rotated. stderr-only telemetry: never
        # reaches an unauthenticated client (see ``_log_capacity_refusal``
        # for the rest of the new counters and why none of them may ever
        # be added to ``build_provenance_body`` or the 503 body itself).
        self._refusals_total = 0

        # -- Saturation snapshot: WHO holds the concurrency slots --------
        # See ``docs/task-inbox/2026-09-21-PREDECLARATION-proxy-
        # saturation-snapshot-who-holds-the-slots.md``. A monotonic slot
        # id, allocated under ``_inflight_lock`` in the SAME critical
        # section that already adjusts ``inflight_requests`` (see
        # ``process_request``/``process_request_thread`` below), maps to
        # the peer host and acquire time of the slot it names. NEVER
        # keyed on ``client_address``, ``id(request)``, or ``fileno()``:
        # all three are reused across a live server's lifetime (a closed
        # socket's fileno is handed to the next accepted connection; an
        # ``id()`` can be reused once its object is collected), so a
        # stale deregister keyed on any of them could delete a different,
        # still-live slot's entry.
        self._slot_registry: dict = {}
        self._next_slot_id = 0
        # A TRANSIENT correlation from the live ``request`` socket object
        # to the slot id ``process_request`` allocated for it. This is
        # needed only because ``process_request_thread``'s call signature
        # is fixed by ``socketserver.ThreadingMixIn.process_request``
        # (which starts it as ``Thread(target=self.process_request_thread,
        # args=(request, client_address))`` -- exactly two positional
        # args, no room for a third) and because this module's own test
        # suite patches/wraps ``process_request_thread`` directly at the
        # class level and must keep working unmodified. Every insertion
        # here is matched by exactly one removal on EVERY code path (the
        # normal ``finally`` in ``process_request_thread`` and the
        # thread-start-failure ``except`` branch in ``process_request``),
        # always while the very same ``request`` object is still a live,
        # referenced local in that call frame -- unlike the slot registry
        # itself (consulted later, asynchronously, by the saturation-
        # reporter thread), this map is never read after the request it
        # describes could have closed, so it does not carry the "reused
        # after close" hazard a persisted ``fileno()``/``id()`` key would.
        self._pending_slot_by_request: dict = {}
        self._saturation_threshold = _saturation_threshold(max_concurrent_requests)
        self._saturation_event = threading.Event()
        self._saturation_stop_event = threading.Event()
        self._saturation_reporter_thread: Optional[threading.Thread] = None
        self._saturation_reporter_lock = threading.Lock()
        # Dedicated rate-limit state for ``saturation_snapshot`` lines --
        # see ``_SATURATION_SNAPSHOT_MIN_INTERVAL_SECONDS`` above for why
        # this is never shared with ``_refusal_log_lock``/
        # ``_refusal_log_last_emitted_monotonic``. Owned exclusively by
        # the single saturation-reporter thread once started (see
        # ``serve_forever`` below) -- no lock needed around reads/writes
        # of this timestamp, since no other thread ever touches it.
        self._saturation_last_emitted_monotonic: Optional[float] = None

    def process_request(self, request, client_address) -> None:
        """Non-blocking admission gate in front of ``ThreadingMixIn``'s own
        ``process_request``: a ``BoundedSemaphore`` slot must be acquired
        BEFORE a handler thread is even started, or the request is refused
        with a 503 right here, on the accept thread, and a handler is
        never constructed for it at all.

        ``acquire(blocking=False)`` -- never blocking or queueing -- is
        deliberate: this method runs on ``serve_forever``'s own single
        accept thread (see ``socketserver.BaseServer.
        _handle_request_noblock``), so blocking here would stall every
        OTHER pending connection behind this one request, turning a fast,
        cheap refusal into an unbounded queue with the same resource-
        exhaustion shape this cap exists to prevent in the first place.
        """
        if not self._slots.acquire(blocking=False):
            # Captured HERE, in the SAME ``_inflight_lock`` acquisition, at
            # the instant the acquire failed -- not later, inside
            # ``_log_capacity_refusal`` after ``_refuse_over_capacity``'s
            # own post-503 drain loop (budgeted up to
            # ``_REFUSAL_DRAIN_DEADLINE_SECONDS``, measured at ~52ms for
            # the silent-client shape). A handler on another thread can
            # release its slot during that drain, so reading the counters
            # after it would report a number from a different accept-
            # thread cycle than the refusal it claims to describe. See
            # ``_log_capacity_refusal`` for the rest of this field's
            # semantics (it can legitimately be less than the cap).
            with self._inflight_lock:
                inflight_at_refusal = self.inflight_requests
                peak_at_refusal = self.peak_inflight_requests
            self._refuse_over_capacity(
                request, client_address, inflight_at_refusal, peak_at_refusal
            )
            self.shutdown_request(request)
            return
        with self._inflight_lock:
            self.inflight_requests += 1
            if self.inflight_requests > self.peak_inflight_requests:
                self.peak_inflight_requests = self.inflight_requests
            # Registry site 1 of 3 (see ``__init__``'s own comment on
            # ``_slot_registry``): allocated in the SAME critical section
            # that increments ``inflight_requests`` above, so a slot is
            # considered HELD from the instant admission succeeds, not
            # from whenever a handler thread happens to start running.
            slot_id = self._next_slot_id
            self._next_slot_id += 1
            self._slot_registry[slot_id] = (client_address[0], time.monotonic())
            self._pending_slot_by_request[request] = slot_id
            inflight_now = self.inflight_requests
        # Outside the lock (see its own invariant comment above): a single
        # int compare and a non-blocking ``Event.set()`` -- never a format
        # or a ``stderr`` write on this, the single accept thread. See
        # ``_saturation_reporter_loop`` below for the one thread that does
        # both, entirely off this thread.
        if inflight_now >= self._saturation_threshold:
            self._saturation_event.set()
        try:
            super().process_request(request, client_address)
        except BaseException:
            # ``ThreadingMixIn.process_request`` starting a new
            # ``threading.Thread`` can itself raise (``RuntimeError:
            # can't start new thread`` under thread-count/resource
            # pressure) BEFORE ``process_request_thread`` ever runs --
            # which means its own ``finally`` below never fires to
            # release this slot. A slot leaked here would count against
            # the cap forever, and the proxy would eventually refuse
            # EVERY request permanently -- a worse, self-inflicted denial
            # of service than the unbounded-thread bug this cap exists to
            # fix. So the slot and the inflight counter are unwound here
            # and the request is closed the same way a refused request
            # is, before the exception is re-raised unchanged for
            # ``_handle_request_noblock``'s own ``handle_error``/
            # ``shutdown_request`` handling (which is safe to run a
            # second time; see ``socketserver.TCPServer.shutdown_
            # request``).
            self._slots.release()
            with self._inflight_lock:
                self.inflight_requests -= 1
                # Registry site 2 of 3: the thread that would otherwise
                # deregister this slot (``process_request_thread``'s own
                # ``finally``, below) never started at all, and this
                # ``except`` is the ONLY place left that can. Missing
                # this leaves a phantom entry in ``_slot_registry``
                # forever -- telemetry that manufactures a permanent
                # false attack signal (a slot that reads as held, by an
                # ever-aging peer, though nothing is actually holding
                # it).
                pending_slot_id = self._pending_slot_by_request.pop(request, None)
                if pending_slot_id is not None:
                    self._slot_registry.pop(pending_slot_id, None)
            self.shutdown_request(request)
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._inflight_lock:
                self.inflight_requests -= 1
                # Registry site 3 of 3: the ordinary release path. Never
                # "repaired" against ``inflight_requests`` here or
                # anywhere else -- a divergence between
                # ``len(_slot_registry)`` and ``inflight_requests`` is the
                # visible symptom of the very bug class this feature
                # exists to surface (a leaked slot, or a double release),
                # exactly like the never-clamped negative ``inflight`` at
                # ``_log_capacity_refusal``'s own comment. Clamping or
                # auto-resyncing it here would restore the silence.
                slot_id = self._pending_slot_by_request.pop(request, None)
                if slot_id is not None:
                    self._slot_registry.pop(slot_id, None)
            self._slots.release()

    def serve_forever(self, poll_interval: float = 0.5) -> None:
        """Starts the saturation-reporter thread (see
        ``_saturation_reporter_loop`` below) lazily, HERE rather than in
        ``__init__`` -- this module's own test suite constructs servers
        via ``create_server``/``ProvenanceProxyServer(...)`` that are
        sometimes never run at all (bind-only construction) and are not
        uniformly closed the instant a test ends, so an ``__init__``-
        started thread would leak one per such instance for the whole
        test process's life. ``daemon=True`` (see ``_start_saturation_
        reporter``) means it is never a barrier to process exit on its
        own regardless, but starting it lazily still keeps a server that
        never serves from ever spinning it up.
        """
        self._start_saturation_reporter()
        super().serve_forever(poll_interval)

    def server_close(self) -> None:
        """Stops the saturation-reporter thread cleanly: sets the stop
        flag, then wakes the reporter via the SAME ``Event`` the accept
        thread uses to signal it (the reporter may currently be blocked
        in ``Event.wait()`` with nothing else left to wake it), then joins
        it after the socket itself is closed.
        """
        self._saturation_stop_event.set()
        self._saturation_event.set()
        thread = self._saturation_reporter_thread
        super().server_close()
        if thread is not None:
            thread.join(timeout=5)

    def _start_saturation_reporter(self) -> None:
        with self._saturation_reporter_lock:
            if self._saturation_reporter_thread is not None:
                return
            thread = threading.Thread(
                target=self._saturation_reporter_loop,
                name="fastmlx-proxy-saturation-reporter",
                daemon=True,
            )
            self._saturation_reporter_thread = thread
            thread.start()

    def _saturation_reporter_loop(self) -> None:
        """The ONLY thread that ever formats or writes a
        ``saturation_snapshot`` line. The accept thread's only job (see
        ``process_request`` above) is waking this one up with a non-
        blocking ``Event.set()`` -- never a format, never I/O, on the
        accept thread itself.
        """
        while True:
            self._saturation_event.wait()
            self._saturation_event.clear()
            if self._saturation_stop_event.is_set():
                return
            self._maybe_emit_saturation_snapshot()

    def _maybe_emit_saturation_snapshot(self) -> None:
        now_monotonic = time.monotonic()
        last = self._saturation_last_emitted_monotonic
        if (
            last is not None
            and (now_monotonic - last) < _SATURATION_SNAPSHOT_MIN_INTERVAL_SECONDS
        ):
            return
        self._saturation_last_emitted_monotonic = now_monotonic

        # COPY under the lock -- never iterate ``_slot_registry`` live
        # here. The accept thread mutates it on every accepted
        # connection (see ``process_request`` above), and this method
        # runs on a SEPARATE thread from that one, so an unlocked
        # iteration here could both race a concurrent mutation and raise
        # ``RuntimeError: dictionary changed size during iteration``.
        # ``json.dumps`` and the ``stderr`` write happen OUTSIDE the
        # lock, below -- the lock's own invariant comment (see
        # ``__init__``) is about the ACCEPT thread never blocking under
        # it, which copying (not formatting or writing) satisfies.
        with self._inflight_lock:
            slots = list(self._slot_registry.values())
            inflight = self.inflight_requests

        # A slot present in this copy may have already been released by
        # the time this line actually reaches ``stderr`` below -- the
        # snapshot is a point-in-time copy, not a live view, and that is
        # benign: it is documented behavior, not a bug to "fix" by
        # re-checking membership against the live registry.
        now = time.monotonic()
        ages = [(peer, max(0.0, now - acquired)) for peer, acquired in slots]
        # The whole point of this field (see the module docstring
        # reference above): counts DISTINCT HOSTS holding a slot, never
        # connections -- one host holding 40 slots reads ``1`` here, not
        # 40. Computed over ALL held slots, not merely the K reported in
        # ``top_slots`` below.
        distinct_peer_hosts = len({peer for peer, _ in ages})
        max_age_s = round(max((age for _, age in ages), default=0.0), 3)
        oldest_first = sorted(ages, key=lambda pair: pair[1], reverse=True)
        top_slots = [
            {"peer": peer, "age_s": round(age, 3)}
            for peer, age in oldest_first[:_SATURATION_SNAPSHOT_TOP_K]
        ]

        entry = {
            "ts": time.time(),
            "event": "saturation_snapshot",
            "inflight": inflight,
            "max_concurrent": self.max_concurrent_requests,
            "distinct_peer_hosts": distinct_peer_hosts,
            "max_age_s": max_age_s,
            "slots_reported": len(top_slots),
            "top_slots": top_slots,
            "trigger": "threshold",
        }
        line = json.dumps(entry)
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
        if self.log_hook is not None:
            self.log_hook(entry)

    def _refuse_over_capacity(
        self, request, client_address, inflight_at_refusal: int, peak_at_refusal: int
    ) -> None:
        """Writes a complete HTTP/1.0 503 response directly on ``request``
        (the raw accepted socket) and logs it -- there is no
        ``ProvenanceProxyHandler`` instance for a refused connection at
        all (that is the point of refusing here, before a handler thread
        even starts), so this builds the same JSON error shape
        ``ProvenanceProxyHandler._send_json_error`` would by hand instead
        of being able to reuse it.

        ``inflight_at_refusal``/``peak_at_refusal`` are passed down from
        ``process_request``, which captured them at the instant the
        semaphore acquire failed -- never re-read in here, since this
        method's own post-503 drain loop below can run for up to
        ``_REFUSAL_DRAIN_DEADLINE_SECONDS`` and another thread can release
        its slot during that window (see ``process_request``'s own comment
        for why that ordering is load-bearing).

        TRAP this exists to guard against: this write happens on
        ``serve_forever``'s own single accept thread (see
        ``process_request`` above), not a per-request handler thread. A
        client that triggers this refusal and then never reads its
        response would, without the timeout below, eventually block this
        ``sendall`` once that client's own TCP receive buffer fills --
        wedging the ONE accept thread every other client's connection
        also depends on, which is a strictly worse denial of service than
        the unbounded-thread bug this whole cap exists to fix. The 2s
        ``settimeout`` plus ``except OSError: pass`` around the write
        make a stuck or hostile client's refusal response best-effort,
        never blocking; the body is kept small enough (a short, fixed
        JSON error shape) to fit in a single MSS so a healthy client
        reads it in one packet.
        """
        request_id = uuid.uuid4().hex
        body = json.dumps(
            {
                "error": {
                    "message": (
                        "too many concurrent requests: the configured limit "
                        f"of {self.max_concurrent_requests} is already in "
                        "flight"
                    ),
                    "type": "too_many_concurrent_requests",
                }
            }
        ).encode("utf-8")
        headers = [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(body))),
            ("Retry-After", "1"),
            ("Connection", "close"),
        ]
        headers.extend(self.provenance_headers)
        headers.append(("X-FastMLX-Request-Id", _sanitize_header_value(request_id)))
        header_bytes = b"".join(f"{name}: {value}\r\n".encode("utf-8") for name, value in headers)
        response = b"HTTP/1.0 503 Service Unavailable\r\n" + header_bytes + b"\r\n" + body
        try:
            request.settimeout(2.0)
            request.sendall(response)
        except OSError:
            pass
        # Drain whatever the client ALREADY sent (for a refusal, always at
        # least the request line -- refusing happens before any read at
        # all, see ``process_request`` above) before closing. Root cause,
        # confirmed by a minimal repro during this feature's own test-first
        # development: closing a socket that still has unread bytes
        # sitting in the kernel receive buffer makes the OS send a TCP RST
        # instead of an orderly FIN. A ``Content-Length``-aware client
        # (``http.client``, curl, this proxy's own upstream connections)
        # never notices, since it stops reading once it has that many
        # body bytes -- but a close-delimited "read until EOF" reader (the
        # framing this response's own HTTP/1.0 default falls back to, see
        # ``ProvenanceProxyHandler.protocol_version``) can see the RST
        # arrive before -- or instead of -- the clean EOF it is waiting
        # for, and some strict clients treat that as a lost/corrupted
        # response even when every declared body byte already arrived.
        # Bounded by an ABSOLUTE wall-clock deadline
        # (``_REFUSAL_DRAIN_DEADLINE_SECONDS``), never a per-``recv``
        # timeout: a per-``recv`` timeout is NOT a wall-clock bound on this
        # loop (an earlier version of this code used ``settimeout(0.2)``
        # here, believing it was) -- a client that keeps delivering at
        # least one byte before each 0.2s per-``recv`` timeout expires
        # makes every ``recv`` succeed, so the loop never times out and
        # never hits EOF either, and keeps running for as long as the
        # client keeps dribbling, on this same single accept thread, where
        # any wait at all, however short per call, is a denial-of-service
        # lever. Computing ``deadline`` ONCE up front and re-deriving each
        # ``settimeout`` call from the REMAINING time left before it is
        # what turns the timeout into a true budget for the whole loop
        # instead of a per-call one: no sequence of "just in time" bytes
        # can push the total past ``_REFUSAL_DRAIN_DEADLINE_SECONDS``.
        #
        # That same fixed deadline is also what bounds the two other
        # shapes this loop must survive: a client that sends nothing at
        # all (refusal happens before any read at all, see
        # ``process_request`` above, so this is a live case, not a
        # hypothetical) blocks on the very first ``recv`` and is cut off
        # once the deadline passes; a client that closes immediately hits
        # EOF (empty ``chunk``) well before the deadline and exits for
        # free.
        #
        # A single blocking ``settimeout`` for the FULL loop is not used
        # instead, because a genuinely well-behaved refused client (one
        # that sends its request, then reads -- never closes) would then
        # cost every ordinary refusal the entire deadline: the loop's
        # ``recv`` would sit waiting for a FIN or more bytes that never
        # come, on the same single accept thread every other connection
        # depends on. This loop switches to non-blocking reads
        # (``setblocking(False)``) the moment the FIRST ``recv`` returns a
        # non-empty chunk, so an ordinary refusal -- whose request bytes
        # are essentially always already sitting in the kernel receive
        # buffer by the time this drain runs -- returns on its first call
        # and pays close to nothing, while a client whose bytes are merely
        # still in flight (not yet landed in that buffer) still gets up to
        # the deadline to arrive rather than an immediate
        # ``BlockingIOError`` that would drain nothing and leave those
        # bytes unread when the socket closes (see the RST-vs-FIN
        # paragraph above for why unread bytes at close time matter).
        # ``BlockingIOError`` is a subclass of ``OSError``, so the
        # existing ``except OSError`` below still terminates the loop the
        # same way once a non-blocking ``recv`` finds nothing left.
        try:
            deadline = time.monotonic() + _REFUSAL_DRAIN_DEADLINE_SECONDS
            drained = 0
            switched_to_nonblocking = False
            while drained < 1 << 20:
                if not switched_to_nonblocking:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    request.settimeout(remaining)
                chunk = request.recv(65536)
                if not chunk:
                    break
                drained += len(chunk)
                if not switched_to_nonblocking:
                    request.setblocking(False)
                    switched_to_nonblocking = True
        except OSError:
            pass
        self._log_capacity_refusal(request_id, len(body), inflight_at_refusal, peak_at_refusal)

    def _log_capacity_refusal(
        self,
        request_id: str,
        bytes_out: int,
        inflight_at_refusal: int,
        peak_at_refusal: int,
    ) -> None:
        # Same JSONL schema ``ProvenanceProxyHandler._log`` writes (see
        # there) -- NOT reused directly, since that is an instance method
        # on a handler that is never constructed for a refused connection
        # (see ``_refuse_over_capacity`` above). ``method``/``path`` are
        # unknown here: refusing before any handler exists means this
        # proxy never reads a byte of the request line, by design (see
        # ``process_request`` above for why that matters).
        entry = {
            "ts": time.time(),
            "request_id": request_id,
            "method": None,
            "path": None,
            "status": 503,
            "upstream_ms_to_headers": 0.0,
            "bytes_out": bytes_out,
            "streamed": False,
            "error": "too_many_concurrent_requests",
            # Captured by ``process_request`` at the instant the semaphore
            # acquire failed, not re-read here -- see its own comment and
            # ``_refuse_over_capacity``'s docstring. Because that capture
            # and the failed acquire are two different locks (never
            # atomic with each other), a legitimate line CAN read e.g.
            # ``inflight=61`` against ``max_concurrent=64``: a handler on
            # another thread released its slot between the failed acquire
            # and the capture. That is not a bug -- do not "fix" it by
            # clamping or asserting equality with ``max_concurrent``.
            # Also never clamped to 0 if negative: a permanently negative
            # value is the visible symptom of a latent double-release
            # elsewhere, and clamping would hide exactly the defect this
            # field exists to surface. stderr-only: never added to
            # ``build_provenance_body`` (see that function's own
            # docstring) -- a live, pollable gauge in a body returned to
            # unauthenticated remote clients would be a real-time
            # saturation and traffic-analysis oracle.
            "inflight": inflight_at_refusal,
            # Always ``self.max_concurrent_requests``, never a literal --
            # this is what lets an operator relate ``inflight``/
            # ``peak_inflight`` to the cap actually CONFIGURED for this
            # process (a caller can override the 64 default).
            "max_concurrent": self.max_concurrent_requests,
            # Same capture-time semantics as ``inflight`` above.
            # Per-PROCESS and never reset for this server's whole life --
            # after a launcher restart this starts back at 0, so a low
            # value here must never be read as "this process has never
            # come close to saturating" if the process is young.
            "peak_inflight": peak_at_refusal,
        }
        # Monotonic, cumulative, incremented once per refusal regardless
        # of whether THIS line is the one that gets emitted below (see
        # ``_refusals_total``'s own comment in ``__init__``). Under the
        # SAME ``_refusal_log_lock`` used for the rate-limit decision
        # further down -- no new lock -- acquired here, separately and
        # first, so the incremented value is already in ``entry`` before
        # ``log_hook`` (unconditional, see below) sees it.
        with self._refusal_log_lock:
            self._refusals_total += 1
            entry["refusals_total"] = self._refusals_total

        # ``log_hook`` fires UNCONDITIONALLY, once per refusal, exactly as
        # before -- it is an in-process, optional, test-only observability
        # callback (see its own docstring in ``__init__``) whose cost is
        # entirely the caller's choice and which no production caller sets,
        # so rate-limiting it would silently throw away information a test
        # or an operator's own hook might depend on. It is the synchronous
        # ``sys.stderr.write`` below -- unconditional disk I/O on the single
        # accept thread, with no caller-controlled opt-out -- that is the
        # unbounded cost being guarded here (see
        # ``_REFUSAL_LOG_MIN_INTERVAL_SECONDS`` above).
        if self.log_hook is not None:
            self.log_hook(entry)

        # Decide whether THIS refusal's line is allowed onto ``stderr``, but
        # do the actual write/flush OUTSIDE the lock: holding a lock across
        # a write that can itself block (see ``_REFUSAL_LOG_MIN_INTERVAL_
        # SECONDS`` above for the pipe-reader-stalls case) would serialize
        # every refusing thread behind that one write, which is the same
        # single-accept-thread wedge shape this whole rate limit exists to
        # prevent -- only now shared across every caller of this method
        # instead of just this one call.
        now = time.monotonic()
        should_emit = False
        suppressed_since_last_log = 0
        with self._refusal_log_lock:
            last = self._refusal_log_last_emitted_monotonic
            if last is None or (now - last) >= _REFUSAL_LOG_MIN_INTERVAL_SECONDS:
                should_emit = True
                suppressed_since_last_log = self._refusal_log_suppressed_since_last_emit
                self._refusal_log_suppressed_since_last_emit = 0
                self._refusal_log_last_emitted_monotonic = now
            else:
                self._refusal_log_suppressed_since_last_emit += 1

        if should_emit:
            entry["suppressed_since_last_log"] = suppressed_since_last_log
            line = json.dumps(entry)
            sys.stderr.write(line + "\n")
            sys.stderr.flush()


def create_server(
    front_host: str,
    front_port: int,
    upstream_host: str,
    upstream_port: int,
    plan: dict,
    upstream_connect_timeout: float = _UPSTREAM_CONNECT_TIMEOUT_SECONDS,
    upstream_read_timeout: Optional[float] = None,
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
    log_hook: Optional[Callable[[dict], None]] = None,
) -> ProvenanceProxyServer:
    """Construct (bind + listen, not yet serving) the proxy server. Raises
    ``OSError`` if ``front_host``:``front_port`` cannot be bound -- the
    caller (``fastmlx_launch.py``'s front-mode orchestration) is
    responsible for turning that into its own fail-closed exit.
    """
    return ProvenanceProxyServer(
        (front_host, front_port),
        ProvenanceProxyHandler,
        upstream_host,
        upstream_port,
        plan,
        upstream_connect_timeout,
        upstream_read_timeout,
        max_request_body_bytes,
        max_concurrent_requests,
        log_hook,
    )
