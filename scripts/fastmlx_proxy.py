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

# How long the handler waits for the CLIENT to finish sending a request
# line/headers/body before giving up -- ``BaseHTTPRequestHandler.setup()``
# applies its ``timeout`` class attribute straight to the request socket
# (see ``ProvenanceProxyHandler.timeout`` below). This bounds only the
# client-facing socket, never the upstream read (see
# ``upstream_read_timeout`` on ``ProvenanceProxyServer``, which stays
# unbounded by default for exactly the opposite reason).
_CLIENT_IDLE_TIMEOUT_SECONDS = 120

# Any header FROM UPSTREAM whose name starts with this prefix
# (case-insensitive) is stripped before the proxy's own provenance headers
# are added: a compromised/misbehaving engine must never be able to spoof
# or duplicate them (see L5 in the front-mode review).
_PROVENANCE_HEADER_PREFIX = "x-fastmlx-"


def _sanitize_header_value(value: object) -> str:
    return _NON_PRINTABLE_ASCII.sub("", str(value))


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
    # request socket (``self.connection.settimeout(self.timeout)``) --
    # bounds only how long the handler waits on the CLIENT for the
    # request line/headers/body; a slow-but-alive upstream is a separate
    # concern (``upstream_read_timeout`` on the server, unbounded by
    # default). Without this, a client that opens a connection and never
    # finishes sending its request ties up a handler thread forever.
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
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        return len(body), status, None

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
            try:
                self.wfile.write(chunk)
                self.wfile.flush()
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
        line = json.dumps(entry)
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
        server = getattr(self, "server", None)
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
        self.provenance_headers = build_provenance_headers(plan)
        self.provenance_body = build_provenance_body(plan)
        # Called synchronously, once per request, right after the JSONL log
        # line is written to stderr -- an optional test-only hook so a
        # caller can wait deterministically for a specific request's log
        # line instead of polling stderr on a sleep loop. ``None`` by
        # default: production callers never set this.
        self.log_hook = log_hook


def create_server(
    front_host: str,
    front_port: int,
    upstream_host: str,
    upstream_port: int,
    plan: dict,
    upstream_connect_timeout: float = _UPSTREAM_CONNECT_TIMEOUT_SECONDS,
    upstream_read_timeout: Optional[float] = None,
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
        log_hook,
    )
