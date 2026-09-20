#!/usr/bin/env python3
"""``fastmlx bench``: an engine-agnostic decode-throughput measurer.

Speaks OpenAI-compatible ``/v1/chat/completions`` over HTTP to ANY server --
the adopted served engine, ``fastmlx-serve``, or a user's own endpoint. This
script never inspects the server's process, binary, or config beyond what
``--expect-listener-pid`` opts into (see C-owner/C-flags below); it measures
strictly over the wire, the same way any real client would.

Invoked with the ``bench`` word already stripped, exactly like
``fastmlx_pull.main`` (see ``scripts/fastmlx.py``'s dispatcher) -- this
module's own ``build_arg_parser`` has no subcommand of its own.

The metric -- steady-state DECODE throughput, excluding prefill -- is
defined ONLY when the reading is MEASURABLE:

    decode_tok_s = (completion_tokens - 1) / (t_last_chunk - t_first_chunk)
        -- valid only if completion_tokens >= 2 AND
           (t_last_chunk - t_first_chunk) > 0

``completion_tokens`` MUST come from the server's own ``usage`` object
(requested via ``stream_options: {"include_usage": true}`` alongside
``stream: true``), never from counting SSE chunks -- a server can legally
coalesce or split tokens across chunks, and a client-side chunk count is
not a token count. A server that never returns usage yields a reading
whose ``token_source`` is ``"sse_chunk_count"`` and which is UNVERIFIED
(see C-tokens). The ``- 1`` is deliberate: the first token's arrival marks
the START of the decode interval, so it is not itself inside it.

A one-token completion, or any completion whose content-bearing chunks
all land within one observed instant (elapsed == 0 -- e.g. two chunks
delivered in a single TCP flush), has an UNMEASURABLE decode interval,
not a zero-length one: this project's rule is that unreachable/
unmeasurable is not zero, so such a reading's ``decode_tok_s`` is
``None`` (JSON ``null``), never ``0.0``. The reading instead carries
``measurable: false`` and an ``unmeasurable_reason`` naming which
condition failed and the observed value(s). TTFT IS measurable from a
single chunk and is still reported on an unmeasurable reading. An arm's
``medianDecodeTokS`` is the median over its MEASURABLE readings only,
and is itself ``None`` when an arm has zero measurable readings; a ratio
between two arms REFUSES (non-zero exit, reason on stderr and in the
refusing control's own ``reason`` field) if either arm's median is
``None``.

TTFT (``t_first_chunk - t_request_sent``) is its own field and is NEVER
folded into the decode rate: a long prefill stall before the first token
must depress TTFT, not decodeTokS, or a slow-to-first-token server would
look like a slow-DECODING one, which is a different (and differently
actionable) fact.

--warmup N runs N passes per arm measurement and DISCARDS them before
the measured passes begin. A server's first pass after a cold start can
read far below its steady rate (a real run of this command read ~260
tok/s on its first pass against ~400 for the rest), and a median over
few runs does not fully absorb that. The default is 1 because the
published quality cards state their method as "one warmup pass
discarded, median of 3 passes" -- so the command's own default
reproduces the cards rather than relying on the CALLER having warmed the
server first. A discarded pass is not a Reading this module keeps: it
never enters ``readings`` or the median, only a per-arm
``warmupDiscarded`` count. Warmup is per measure_arm() INVOCATION, so in
ratio mode all three arm measurements (reference bookend, candidate,
reference bookend) are identically conditioned -- which is what C-drift
requires, since a bookend comparison is only meaningful between like and
like. An exception raised during a warmup pass PROPAGATES: a server that
fails while warming is real signal, and swallowing it would let the
instrument report a measurement it did not honestly obtain. Every row's
``boundary`` records ``warmup=N``, so a row taken with a warmup is
distinguishable from one taken without; rows published before this flag
existed were effectively warmup=0.

--temperature pins the sampling temperature sent with EVERY pass, warmup
and measured alike, and defaults to 0.0. A timing instrument must hold its
workload fixed; with sampling on, the server's own default is whatever it
is, and completion length varies pass to pass -- measured against a real
OpenAI-compatible endpoint, four IDENTICAL requests at the server's own
default returned completion_tokens = 128, 78, 116, 128, and the same four
requests at temperature 0 returned 124, 124, 124, 124. That matters because
decodeTokS divides by an inter-token interval count that scales with
completion length, so a SHORT completion reads SLOWER (fixed early-token
overhead amortizes over fewer intervals) -- across a real cycle's readings,
Pearson(completionTokens, decodeTokS) measured +0.825 and +0.603 within two
separate arms. An unpinned temperature therefore makes the run-to-run
spread measure sampling variance as much as server speed. Every row's
``boundary`` records ``temperature=N`` for the same reason it records
``warmup=N``: a row that does not state its sampling temperature is not
self-describing. Rows published before this flag existed did not pin
temperature at all and so carry sampling-driven spread that this flag did
not yet exist to remove.

Every row's ``boundary`` also records ``chip=<brand> (<arch>)`` -- e.g.
``chip=Apple M3 Ultra (arm64)`` -- read via ``sysctl -n
machdep.cpu.brand_string``, and NEVER a hostname. The standard library's
``node()`` hostname lookup (formerly used here, via the ``platform``
module) returns the machine's own HOSTNAME, which is internal
infrastructure naming that makes a row unpublishable on its own, and is
not even the fact a benchmark row needs -- the published quality cards
describe their hardware by CHIP ("measured on Apple M3 Ultra"), never by
machine name. ``--host-label TEXT`` is the
deliberate, OPT-IN escape hatch for when an operator still needs to tell
two boxes apart in an INTERNAL-only row: when given, it appends
``hostLabel=<text>`` to the boundary. Its absence is the default -- a row
is publishable by construction unless an operator explicitly forfeits that
by passing ``--host-label``.

Five controls, each an independently-failable field (never a boolean the
command can pass for free):

  C-tokens    -- per-reading token_source; ratio mode REFUSES unless every
                 reading on both arms is verified via usage.
  C-drift     -- ratio mode only: the reference arm is measured FIRST and
                 LAST (candidate sandwiched between); a first/last median
                 ratio outside --drift-tolerance VOIDs the run.
  C-magnitude -- any arm's median below --magnitude-floor (a debug/CPU-
                 fallback build reads 1-25 tok/s on this hardware class)
                 OR above --magnitude-ceiling (a degenerate, too-short
                 decode interval reads implausibly fast and flatters the
                 result -- the direction that actually misleads) is
                 reported loudly but does not by itself void a
                 single-arm run. An arm with no measurable readings is
                 reported as unmeasurable in this control, never
                 compared against either bound.
  C-owner     -- with --expect-listener-pid, the PID actually LISTENing on
                 --base-url's port (via ``lsof``) must match; a mismatch
                 REFUSES (a stale server on a reused port has already
                 fooled a health check in this project once).
  C-flags     -- with --expect-listener-pid, that PID's own argv (via
                 ``ps``) is captured so a row cannot claim a configuration
                 it did not run.

Usage errors (bad/missing flags) exit 64 (EX_USAGE), never argparse's
default of 2 -- this project reserves other exit codes for measurement
outcomes, and a usage error must never be misread as one.

Every row also carries a computed ``publishable`` verdict
(``row["publishable"]``), from ``publishability_control``: it scans the
row's own serialized JSON (excluding this field itself, computed last and
attached after) against the public validator's ``PRIVATE_MARKERS``
(``scripts/validate_public_repository.py``, loaded by file path at CALL
time -- never at this module's own import time, so this command stays
runnable when that sibling is absent; any load failure yields
``refused_sweep_unavailable``, never a crash or a silent clean sweep).
Each marker maps to a stable, publishable CLASS label (e.g.
``private-network-address``), never the matched text itself -- a verdict
that quoted what it found would republish the very string it is
withholding the row for. A marker the imported source carries that this
module has no label for makes the verdict REFUSE
(``refused_unclassified_marker``) rather than report a clean sweep: a
single source of truth that can silently grow past its consumer is not
one. One further marker class, a third-party engine name that this
project's OWN binary name (``fastmlx-serve``) happens to CONTAIN as a
substring, is handled as a documented exception: occurrences of this
project's own binary name are stripped from the scanned text FIRST, so a
row naming only its own binary is still ``publishable``, while a row
naming the bare third-party name is not. The verdict is metadata and
NEVER changes the process exit code -- a row measured over the LAN is a
perfectly valid measurement, only not a publishable one; see the five
controls above for what can actually void a run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence


SCHEMA = "fastmlx-bench-row-v1"

TOKEN_SOURCE_USAGE = "usage.completion_tokens"
TOKEN_SOURCE_SSE_COUNT = "sse_chunk_count"

DEFAULT_PROMPT = (
    "In two or three sentences, describe how the water cycle moves water "
    "between the ocean, the atmosphere, and the land."
)
# The published quality cards' method uses THREE prompts, not one -- see
# DEFAULT_MAX_TOKENS's comment for the matching 256-token axis. These three
# are FIXED and deterministic (never randomly sampled or drawn from a corpus
# at run time): this is a TIMING workload, not a quality probe, so the
# command's own reproducibility depends on every invocation sending the
# exact same bytes. They are English and of broadly comparable length so
# that no single prompt dominates a pass's pooled rate (see measure_arm's
# pooling doc) by having a wildly different completion-length profile than
# its siblings. ``--prompt`` (repeatable) replaces this whole set, never
# merges into it -- see build_arg_parser's help text.
DEFAULT_PROMPTS = (
    DEFAULT_PROMPT,
    "In two or three sentences, explain why the sky looks blue during the "
    "day and can turn orange or red near sunset.",
    "In two or three sentences, explain how a refrigerator keeps food cold "
    "using a compression and evaporation cycle.",
)
# The published quality cards' method requests 256 completion tokens per
# pass, not 128 -- this default is the matching axis to DEFAULT_PROMPTS'
# three-prompt set, so the command's own default reproduces the cards'
# method on both axes without the caller needing to pass either flag.
DEFAULT_MAX_TOKENS = 256
DEFAULT_RUNS = 3
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAGNITUDE_FLOOR = 25.0
DEFAULT_MAGNITUDE_CEILING = 2000.0
DEFAULT_DRIFT_TOLERANCE = 0.05
# The published quality cards' method is "one warmup pass discarded, median
# of 3 passes" -- this default is what makes the command's own default
# reproduce that method without requiring the caller to pass --warmup.
DEFAULT_WARMUP = 1
# A timing instrument must hold its workload fixed. With sampling on (a
# nonzero temperature), every completion can be a different length --
# measured on the fleet, four IDENTICAL requests against a real endpoint
# returned completion_tokens = 128, 78, 116, 128 at the server's own
# default, and 124, 124, 124, 124 with temperature pinned to 0. Since
# decode_tok_s divides by (t_last_chunk - t_first_chunk), a SHORT
# completion reads SLOWER (early-token overhead amortizes over fewer
# inter-token intervals) -- so an unpinned temperature makes the median
# measure sampling variance as much as server speed. 0.0 is the default
# precisely because it is the only value that reproduces the SAME
# completion length every pass.
DEFAULT_TEMPERATURE = 0.0

# sysexits.h EX_USAGE -- see _UsageErrorArgumentParser and the module
# docstring for why this must never collide with argparse's own default.
_EXIT_USAGE_ERROR = 64


class BenchError(Exception):
    """A single reading could not be taken at all (unreachable server, a
    non-2xx response, a stream that produced no SSE data before it ended).
    Always carries an operator-facing reason; never raised for a condition
    this module can classify as one of the five named controls instead.
    """


# ---------------------------------------------------------------------
# Measurement.
# ---------------------------------------------------------------------
class Reading:
    """One streamed request/response, resolved to its four load-bearing
    facts. A plain class, not ``@dataclasses.dataclass``: this module is
    loaded via ``importlib.util.module_from_spec`` (see
    ``scripts/fastmlx.py``'s own ``_load_sibling_module``, which never
    registers the loaded module into ``sys.modules``) -- CPython 3.14's
    ``dataclasses`` looks up ``cls.__module__`` in ``sys.modules`` while
    processing the class body and crashes with an unrelated
    ``AttributeError`` when it is not registered there. A hand-written
    ``__init__`` has no such dependency.
    """

    __slots__ = (
        "decode_tok_s",
        "ttft_s",
        "completion_tokens",
        "token_source",
        "measurable",
        "unmeasurable_reason",
        "elapsed_s",
    )

    def __init__(
        self,
        decode_tok_s: Optional[float],
        ttft_s: float,
        completion_tokens: int,
        token_source: str,
        measurable: bool,
        unmeasurable_reason: Optional[str],
        elapsed_s: float,
    ):
        self.decode_tok_s = decode_tok_s
        self.ttft_s = ttft_s
        self.completion_tokens = completion_tokens
        self.token_source = token_source
        self.measurable = measurable
        self.unmeasurable_reason = unmeasurable_reason
        # (t_last_chunk - t_first_chunk), kept even on an unmeasurable
        # reading -- NOT part of to_json's public shape (decode_tok_s /
        # completion_tokens already state everything a reader needs), but
        # load-bearing internally: _pool_pass sums this across a pass's
        # per-prompt readings to compute the pooled decode rate directly
        # from the same numbers _consume_sse_stream already measured,
        # rather than reconstructing it from decode_tok_s (which would
        # divide, then a caller would multiply back out, for no reason).
        self.elapsed_s = elapsed_s

    def to_json(self) -> dict:
        return {
            "decodeTokS": self.decode_tok_s,
            "ttftS": self.ttft_s,
            "completionTokens": self.completion_tokens,
            "tokenSource": self.token_source,
            "measurable": self.measurable,
            "unmeasurableReason": self.unmeasurable_reason,
        }


class ArmResult:
    """See ``Reading``'s docstring for why this is a plain class."""

    __slots__ = ("model", "readings", "median_decode_tok_s", "warmup_discarded", "pass_rates")

    def __init__(
        self,
        model: str,
        readings: List[Reading],
        median_decode_tok_s: Optional[float],
        warmup_discarded: int = 0,
        pass_rates: Optional[List[Optional[float]]] = None,
    ):
        self.model = model
        self.readings = readings
        self.median_decode_tok_s = median_decode_tok_s
        # Count of warmup passes discarded before these ``readings`` were
        # taken -- NOT itself a Reading (see measure_arm): warmup passes
        # never enter ``readings`` or the median at all.
        self.warmup_discarded = warmup_discarded
        # One pooled rate per MEASURED pass (``_pool_pass``'s output,
        # ``None`` for an unmeasurable pass), in pass order -- NOT part of
        # to_json's public shape (``readings`` already carries every
        # per-request fact a reader needs). Load-bearing internally for
        # ratio mode's bookend combination in run_bench: the reference
        # arm's combined median must be taken over the two bookend
        # measure_arm() calls' own PASS rates, not recomputed from pooled
        # per-reading rates across a multi-prompt set (that would mix
        # different prompts' individual rates into one median instead of
        # each pass's own pooled rate).
        self.pass_rates = list(pass_rates) if pass_rates is not None else []

    def to_json(self) -> dict:
        return {
            "model": self.model,
            "readings": [reading.to_json() for reading in self.readings],
            "medianDecodeTokS": self.median_decode_tok_s,
            "warmupDiscarded": self.warmup_discarded,
        }


def _consume_sse_stream(response, t_request_sent: float) -> Reading:
    """Reads one streamed ``/v1/chat/completions`` response to a
    ``Reading``.

    ``t_first_chunk`` is the arrival of the FIRST SSE data event of any
    kind (content or usage-only) -- this marks the start of the decode
    interval and anchors TTFT. ``t_last_chunk`` tracks only the LAST
    CONTENT-bearing event (a trailing usage-only bookkeeping chunk, or a
    long gap before it, must never inflate the decode interval -- see
    C-tokens' ``- 1`` rationale in the module docstring).

    ``completion_tokens``/``token_source`` prefer the server's own
    ``usage.completion_tokens`` the instant any SSE event carries a
    non-null ``usage`` object; only when the stream never carries one at
    all does this fall back to the raw content-chunk count, marked
    ``sse_chunk_count`` (UNVERIFIED -- see C-tokens).
    """
    t_first_chunk: Optional[float] = None
    t_last_chunk: Optional[float] = None
    chunk_count = 0
    usage_completion_tokens: Optional[int] = None
    while True:
        raw_line = response.readline()
        if not raw_line:
            break
        line = raw_line.decode("utf-8", "replace").rstrip("\r\n")
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        arrival = time.monotonic()
        if payload == "[DONE]":
            break
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if t_first_chunk is None:
            t_first_chunk = arrival
        choices = event.get("choices") or []
        has_content = bool(choices) and choices[0].get("delta", {}).get("content") is not None
        if has_content:
            chunk_count += 1
            t_last_chunk = arrival
        usage = event.get("usage")
        if usage is not None and usage.get("completion_tokens") is not None:
            usage_completion_tokens = usage["completion_tokens"]
    if t_first_chunk is None:
        raise BenchError("no SSE data chunks received before the stream ended")
    if t_last_chunk is None:
        # No content-bearing chunk ever arrived (e.g. an immediate
        # usage-only reply): there is no decode interval to measure at
        # all, not a zero-length one masquerading as a real measurement.
        t_last_chunk = t_first_chunk
    if usage_completion_tokens is not None:
        completion_tokens = usage_completion_tokens
        token_source = TOKEN_SOURCE_USAGE
    else:
        completion_tokens = chunk_count
        token_source = TOKEN_SOURCE_SSE_COUNT
    elapsed = max(t_last_chunk - t_first_chunk, 0.0)
    numerator = max(completion_tokens - 1, 0)
    # MEASURABLE only if BOTH conditions hold -- see module docstring's
    # "unreachable/unmeasurable is not zero" rule. A one-token completion
    # (completion_tokens < 2) and a zero-width interval (elapsed <= 0, e.g.
    # every content chunk landing in one flush) are independent failure
    # modes; report whichever actually failed with its observed value so
    # an operator never has to guess which precondition broke.
    measurable = completion_tokens >= 2 and elapsed > 0
    if measurable:
        decode_tok_s: Optional[float] = numerator / elapsed
        unmeasurable_reason: Optional[str] = None
    else:
        decode_tok_s = None
        failed_conditions = []
        if completion_tokens < 2:
            failed_conditions.append(f"completion_tokens={completion_tokens} (need >= 2)")
        if not (elapsed > 0):
            failed_conditions.append(f"elapsed={elapsed:.6f}s (need > 0)")
        unmeasurable_reason = "decode interval unmeasurable: " + "; ".join(failed_conditions)
    return Reading(
        decode_tok_s=decode_tok_s,
        ttft_s=t_first_chunk - t_request_sent,
        completion_tokens=completion_tokens,
        token_source=token_source,
        measurable=measurable,
        unmeasurable_reason=unmeasurable_reason,
        elapsed_s=elapsed,
    )


def stream_chat_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    api_key: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Reading:
    """One streamed ``/v1/chat/completions`` request/response, resolved to
    exactly one ``Reading``. Raises ``BenchError`` (never crashes) on an
    unreachable server, a non-2xx response, or a stream that produced no
    SSE data at all before it ended.
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        # Requested unconditionally: C-tokens depends on the server's own
        # accounting being available to ask for at all.
        "stream_options": {"include_usage": True},
        # Always included, never conditional on a non-default value: a
        # server sampling with its OWN default when this field is absent is
        # exactly the length-varying-workload defect this flag exists to
        # close (see DEFAULT_TEMPERATURE's comment).
        "temperature": temperature,
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
    )
    t_request_sent = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _consume_sse_stream(response, t_request_sent)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")
        except Exception:
            pass
        raise BenchError(
            f"{base_url} refused the request: HTTP {exc.code} {exc.reason} {detail}".strip()
        )
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise BenchError(f"could not reach {base_url}: {exc}")


def _pool_pass(pass_readings: Sequence[Reading]) -> Optional[float]:
    """Pools ONE pass's per-prompt ``Reading``s into a single pass rate:

        passDecodeTokS = sum_i (completionTokens_i - 1) / sum_i elapsed_i

    the exact generalisation of the single-request formula (the ``i`` = 1
    case) -- summing numerators and denominators SEPARATELY before
    dividing, never averaging the per-prompt rates themselves (that would
    weight a short, fast prompt equally with a long, slow one instead of
    weighting by actual decode time, and is a numerically DIFFERENT
    quantity -- see this function's own test coverage for a fixture where
    the two disagree).

    A pass is measurable ONLY when every one of its per-prompt readings is
    individually measurable (``Reading.measurable``) -- pooling from a
    measurable subset would silently shrink the workload for just that one
    pass relative to its siblings, which is exactly the kind of workload
    drift this instrument exists to prevent (see module docstring's
    --warmup/--temperature discussion for the same principle applied
    elsewhere). Zero passed-in readings, or any unmeasurable one, makes the
    whole pass unmeasurable: ``None``, never ``0.0``.
    """
    if not pass_readings or not all(reading.measurable for reading in pass_readings):
        return None
    numerator = sum(reading.completion_tokens - 1 for reading in pass_readings)
    denominator = sum(reading.elapsed_s for reading in pass_readings)
    if not (denominator > 0):
        # Cannot occur given every reading passed its own measurable check
        # (each individually requires elapsed_s > 0), but guarded the same
        # fail-closed way as _consume_sse_stream's own measurability check
        # rather than trusting that invariant silently.
        return None
    return numerator / denominator


def measure_arm(
    base_url: str,
    model: str,
    prompts: Sequence[str],
    max_tokens: int,
    runs: int,
    api_key: Optional[str],
    timeout: float,
    warmup: int = 0,
    temperature: float = DEFAULT_TEMPERATURE,
) -> ArmResult:
    """Measures one arm over ``prompts`` (the arm's whole prompt SET, in
    order): ``warmup`` PASSES executed and discarded FIRST -- one request
    PER PROMPT per pass (each request a full request/response, so a server
    that fails during warmup raises exactly as it would for a measured
    pass -- see module docstring's --warmup discussion) -- then ``runs``
    measured passes, each likewise one request per prompt.

    A "pass" is the unit the published quality cards' method counts: one
    request per prompt in the set, pooled to a single rate via
    ``_pool_pass`` (see its docstring). The arm's ``readings`` still
    retains EVERY per-request ``Reading`` from every measured pass, in
    request order -- nothing is discarded or summarised away, so a reader
    can see per-prompt variation directly; only the MEDIAN is computed over
    pooled per-pass rates rather than per-reading ones.

    Warmup readings never enter ``readings`` or the median at all: a
    discarded pass is not kept by this module, only counted.

    ``temperature`` is passed identically to every request, warmup and
    measured alike: they must be the IDENTICAL workload, or a warmup at a
    different temperature would warm the wrong thing (see
    DEFAULT_TEMPERATURE's comment on why sampling makes each pass a
    different-length completion).
    """
    # ``prompts`` is a SET, and a bare ``str`` is itself a valid Sequence
    # of single characters -- passing one would silently measure len(s)
    # one-character prompts and return a plausible-looking rate for a
    # workload nobody asked for. A measurement instrument must not have a
    # silently-wrong input shape, so refuse it.
    if isinstance(prompts, str):
        raise TypeError(
            "measure_arm takes a sequence of prompts, not a single string; "
            "pass [prompt] rather than prompt"
        )
    for _ in range(warmup):
        for prompt in prompts:
            stream_chat_completion(base_url, model, prompt, max_tokens, api_key, timeout, temperature)
    readings: List[Reading] = []
    pass_rates: List[Optional[float]] = []
    for _ in range(runs):
        pass_readings = [
            stream_chat_completion(base_url, model, prompt, max_tokens, api_key, timeout, temperature)
            for prompt in prompts
        ]
        readings.extend(pass_readings)
        pass_rates.append(_pool_pass(pass_readings))
    # The median is over MEASURABLE PASSES only -- an unmeasurable pass's
    # pooled rate is None and must never enter a median (not a 0.0 to
    # average in, nor comparable to the measured values at all). Zero
    # measurable passes means the arm's median is itself unmeasurable
    # (None), not 0.0 -- same "unreachable is not zero" rule as a single
    # reading's own decode_tok_s.
    measurable_pass_rates = [rate for rate in pass_rates if rate is not None]
    median = statistics.median(measurable_pass_rates) if measurable_pass_rates else None
    return ArmResult(
        model=model,
        readings=readings,
        median_decode_tok_s=median,
        warmup_discarded=warmup,
        pass_rates=pass_rates,
    )


# ---------------------------------------------------------------------
# C-owner / C-flags: who is actually listening.
# ---------------------------------------------------------------------
def _listening_pid(port: Optional[int]) -> Optional[int]:
    """The PID of the process LISTENing on ``port`` right now, via ``lsof
    -nP -iTCP:<port> -sTCP:LISTEN``, parsed defensively -- lsof's plain-text
    columnar output is not a stable machine format, so any unexpected shape
    (no ``lsof`` on PATH, no LISTEN socket, an extra/missing column) yields
    ``None`` rather than a crash or a wrong PID.
    """
    if port is None:
        return None
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    # lsof's header line is ``COMMAND PID USER FD TYPE ...``, whitespace-
    # separated (lsof does not quote fields) -- PID is the second column.
    fields = lines[1].split()
    if len(fields) < 2:
        return None
    try:
        return int(fields[1])
    except ValueError:
        return None


def _cmdline_for_pid(pid: int) -> Optional[str]:
    """``pid``'s own argv, as one string, via ``ps -o command= -p <pid>``.
    ``None`` (never an exception) if the process is gone or ``ps`` itself
    is unavailable/fails.
    """
    try:
        result = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    return text or None


# ---------------------------------------------------------------------
# The five controls.
# ---------------------------------------------------------------------
def tokens_control(arms: Sequence[ArmResult]) -> dict:
    unverified_models = sorted(
        {
            arm.model
            for arm in arms
            for reading in arm.readings
            if reading.token_source != TOKEN_SOURCE_USAGE
        }
    )
    if unverified_models:
        return {
            "status": "unverified",
            "reason": (
                "reading(s) for "
                + ", ".join(unverified_models)
                + " used sse_chunk_count -- the server never returned "
                "usage.completion_tokens for a stream_options.include_usage "
                "request"
            ),
        }
    return {"status": "verified", "reason": None}


def drift_control(
    reference_model: Optional[str],
    first_arm: Optional[ArmResult],
    last_arm: Optional[ArmResult],
    tolerance: float,
) -> dict:
    if reference_model is None or first_arm is None or last_arm is None:
        return {
            "status": "not_applicable",
            "firstMedianDecodeTokS": None,
            "lastMedianDecodeTokS": None,
            "driftRatio": None,
            "toleranceRatio": tolerance,
            "reason": "no --reference-model given; drift is only measured in ratio mode",
        }
    first_median = first_arm.median_decode_tok_s
    last_median = last_arm.median_decode_tok_s
    if first_median is None or last_median is None:
        # Neither bookend median is comparable at all -- an arm with zero
        # measurable readings has no rate to sandwich a drift ratio
        # around. This REFUSES the ratio, same as any other void status,
        # rather than crashing on `None / float` or `None < tolerance`.
        unmeasurable_bookends = []
        if first_median is None:
            unmeasurable_bookends.append("first")
        if last_median is None:
            unmeasurable_bookends.append("last")
        reason = (
            f"reference model {reference_model!r} drift could not be evaluated: "
            + " and ".join(unmeasurable_bookends)
            + " bookend measurement had zero measurable readings (median decode "
            "rate is unmeasurable, not comparable to a rate)"
        )
        return {
            "status": "void",
            "firstMedianDecodeTokS": first_median,
            "lastMedianDecodeTokS": last_median,
            "driftRatio": None,
            "toleranceRatio": tolerance,
            "reason": reason,
        }
    drift_ratio = (last_median / first_median) if first_median else None
    void = drift_ratio is None or abs(drift_ratio - 1.0) > tolerance
    if void:
        ratio_desc = f"{drift_ratio:.4f}" if drift_ratio is not None else "undefined (first median is 0)"
        reason = (
            f"reference model {reference_model!r} drifted beyond tolerance between "
            f"the first and last bookend measurement: first median "
            f"{first_median:.3f} tok/s, last median {last_median:.3f} tok/s, "
            f"drift ratio {ratio_desc}, tolerance ±{tolerance}"
        )
        return {
            "status": "void",
            "firstMedianDecodeTokS": first_median,
            "lastMedianDecodeTokS": last_median,
            "driftRatio": drift_ratio,
            "toleranceRatio": tolerance,
            "reason": reason,
        }
    return {
        "status": "verified",
        "firstMedianDecodeTokS": first_median,
        "lastMedianDecodeTokS": last_median,
        "driftRatio": drift_ratio,
        "toleranceRatio": tolerance,
        "reason": None,
    }


def magnitude_control(arms: Sequence[ArmResult], floor: float, ceiling: float) -> dict:
    measured = [
        {"model": arm.model, "medianDecodeTokS": arm.median_decode_tok_s} for arm in arms
    ]
    # An arm with an unmeasurable (None) median is reported as unmeasurable
    # in THIS control, never compared against either bound -- comparing
    # `None < floor` raises TypeError in Python 3, and silently treating
    # None as implausible-by-comparison would be its own "unreachable is
    # not zero" defect.
    below_floor = [
        entry for entry in measured
        if entry["medianDecodeTokS"] is not None and entry["medianDecodeTokS"] < floor
    ]
    above_ceiling = [
        entry for entry in measured
        if entry["medianDecodeTokS"] is not None and entry["medianDecodeTokS"] > ceiling
    ]
    reason_parts = []
    if below_floor:
        names = ", ".join(
            f"{entry['model']}={entry['medianDecodeTokS']:.2f} tok/s" for entry in below_floor
        )
        reason_parts.append(
            f"median decode rate below the {floor} tok/s magnitude FLOOR for: {names} "
            "-- a debug or CPU-fallback build reads in this range on this hardware class"
        )
    if above_ceiling:
        names = ", ".join(
            f"{entry['model']}={entry['medianDecodeTokS']:.2f} tok/s" for entry in above_ceiling
        )
        reason_parts.append(
            f"median decode rate above the {ceiling} tok/s magnitude CEILING for: {names} "
            "-- a degenerate, too-short decode interval (e.g. multiple content "
            "chunks delivered in one flush) inflates the rate rather than "
            "reflecting real decode speed"
        )
    implausible = bool(reason_parts)
    reason = (
        "; ".join(reason_parts) + "; this does not by itself void a single-arm run"
        if reason_parts else None
    )
    return {
        "magnitudeImplausible": implausible,
        "floor": floor,
        "ceiling": ceiling,
        "arms": measured,
        "reason": reason,
    }


def owner_control(expect_pid: Optional[int], port: Optional[int]) -> dict:
    if expect_pid is None:
        return {
            "status": "not_applicable",
            "expectedPid": None,
            "actualPid": None,
            "reason": "--expect-listener-pid not given",
        }
    if port is None:
        return {
            "status": "mismatch",
            "expectedPid": expect_pid,
            "actualPid": None,
            "reason": f"--base-url has no port to check a listener against",
        }
    actual_pid = _listening_pid(port)
    if actual_pid is None:
        return {
            "status": "mismatch",
            "expectedPid": expect_pid,
            "actualPid": None,
            "reason": (
                f"could not determine the PID listening on port {port} "
                "(lsof found no LISTEN socket there)"
            ),
        }
    if actual_pid != expect_pid:
        return {
            "status": "mismatch",
            "expectedPid": expect_pid,
            "actualPid": actual_pid,
            "reason": (
                f"expected listener PID {expect_pid} but port {port} is held by "
                f"PID {actual_pid} -- a stale server on a reused port has already "
                "fooled a health check in this project once"
            ),
        }
    return {
        "status": "verified",
        "expectedPid": expect_pid,
        "actualPid": actual_pid,
        "reason": None,
    }


def flags_control(expect_pid: Optional[int]) -> dict:
    if expect_pid is None:
        return {
            "status": "not_applicable",
            "listenerCmdline": None,
            "reason": "--expect-listener-pid not given",
        }
    cmdline = _cmdline_for_pid(expect_pid)
    if cmdline is None:
        return {
            "status": "capture_failed",
            "listenerCmdline": None,
            "reason": f"could not capture argv for PID {expect_pid} (process may have exited)",
        }
    return {"status": "captured", "listenerCmdline": cmdline, "reason": None}


# ---------------------------------------------------------------------
# Publishability verdict: a whole-row, fail-closed sweep for markers that
# make a row internal-only. See module docstring for the design summary,
# and docs/task-inbox/2026-09-20-DECISION-bench-row-publishability-verdict.md
# for the decision this implements.
#
# Every marker below is built by CONCATENATION, never as a single literal:
# this module is itself part of the public projection, so a literal
# occurrence here would match the very scan it exists to drive (the same
# convention as ``scripts/validate_public_repository.py``, which this
# module's marker source is loaded from at call time -- see
# ``_load_private_markers``).
# ---------------------------------------------------------------------
_MARKER_CLASS_LABELS: dict = {
    marker.lower(): label
    for marker, label in (
        (("/" + "Users/"), "absolute-user-path"),
        (("/" + "private/"), "absolute-user-path"),
        (("192" + ".168."), "private-network-address"),
        (("llm" + "bench"), "internal-host-account"),
        (("passwordless" + " sudo"), "privilege-escalation-note"),
        (("docs/" + "superpowers" + "/"), "internal-repository-path"),
        (("spike/" + "scripts" + "/"), "internal-repository-path"),
        (("BEGIN OPENSSH" + " PRIVATE KEY"), "private-key-material"),
        (("BEGIN RSA" + " PRIVATE KEY"), "private-key-material"),
    )
}

# The third-party engine name is its own marker class, not part of the
# imported PRIVATE_MARKERS set -- it has a documented exception (the
# own-binary substring trap below) that no other marker class needs, so it
# is scanned separately rather than folded into the drift-guarded map.
_THIRD_PARTY_ENGINE_MARKER = "mlx" + "-serve"
_THIRD_PARTY_ENGINE_CLASS_LABEL = "third-party-engine-name"

# This project's OWN binary name CONTAINS the third-party engine marker
# above as a proper substring -- prefixing it is all it takes. Occurrences
# of our own name are stripped from the scanned text before the
# third-party scan runs -- see publishability_control's own-binary
# exception.
#
# Note both operands below are built by CONCATENATION, and the comment
# above deliberately does NOT spell either name out. An earlier draft of
# this very comment illustrated the trap by quoting both names literally,
# which put the bare third-party name into a publicly projected file --
# the exact violation a previous cycle spent an increment scrubbing from
# this file's docstring. The repository gitleaks run and the public
# validator BOTH passed while it was present, because that name is not in
# PRIVATE_MARKERS; only the publication sweep caught it. A comment
# explaining a marker guard is still scanned text.
_OWN_BINARY_MARKER = "fastmlx" + "-serve"


def _load_private_markers() -> "Optional[tuple]":
    """Loads ``PRIVATE_MARKERS`` from the sibling
    ``scripts/validate_public_repository.py`` by file path, at CALL time --
    never at this module's own import time, so ``fastmlx bench`` stays
    runnable when that sibling script is missing or broken. ANY failure
    (missing file, import error, a module with no ``PRIVATE_MARKERS``
    attribute, an OSError reading the file) returns ``None`` rather than
    raising -- ``publishability_control`` turns that into the fail-closed
    ``refused_sweep_unavailable`` status, never a crash and never a silent
    clean sweep.
    """
    try:
        path = Path(__file__).resolve().parent / "validate_public_repository.py"
        spec = importlib.util.spec_from_file_location(
            "fastmlx_bench_private_markers_source", path
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        markers = getattr(module, "PRIVATE_MARKERS", None)
        if not markers:
            return None
        return tuple(markers)
    except Exception:
        return None


def _classify_markers(markers: "Sequence[str]") -> "tuple[dict, int]":
    """Splits ``markers`` (case-insensitively) into a ``{marker_lower:
    label}`` map of every marker this module CAN classify, plus a count of
    markers it cannot -- the drift guard's own input (see
    ``publishability_control``).
    """
    labeled: dict = {}
    unlabeled_count = 0
    for marker in markers:
        label = _MARKER_CLASS_LABELS.get(marker.lower())
        if label is None:
            unlabeled_count += 1
            continue
        labeled[marker.lower()] = label
    return labeled, unlabeled_count


def publishability_control(row: dict) -> dict:
    """A whole-row, fail-closed publishability verdict over ``row``,
    scanned as ``json.dumps(row, sort_keys=True, default=str)`` -- ``row``
    must NOT yet carry its own ``publishable`` key when this is called (see
    ``run_bench``, which computes this LAST and attaches it after).

    Returns ``{"status": ..., "markerClasses": [...], "reason": ... or
    None}``. ``markerClasses`` is always sorted and de-duplicated, and
    names stable publishable LABELS only -- never matched text, and never a
    marker literal -- so neither this dict nor its JSON serialization ever
    republishes the thing it is withholding the row for.
    """
    markers = _load_private_markers()
    if markers is None:
        return {
            "status": "refused_sweep_unavailable",
            "markerClasses": [],
            "reason": (
                "the private-marker source module could not be loaded; "
                "refusing rather than reporting an unswept row as clean"
            ),
        }
    labeled_markers, unlabeled_count = _classify_markers(markers)
    if unlabeled_count:
        return {
            "status": "refused_unclassified_marker",
            "markerClasses": [],
            "reason": (
                f"{unlabeled_count} marker(s) from the private-marker source "
                "have no class label in this module; refusing rather than "
                "reporting a clean sweep the label map cannot actually vouch for"
            ),
        }
    text = json.dumps(row, sort_keys=True, default=str).lower()
    hit_classes = set()
    for marker_lower, label in labeled_markers.items():
        if marker_lower in text:
            hit_classes.add(label)
    # Own-binary exception: strip this project's own binary name BEFORE
    # scanning for the third-party engine name it happens to contain as a
    # substring (order matters -- see module docstring).
    swept_text = text.replace(_OWN_BINARY_MARKER.lower(), "")
    if _THIRD_PARTY_ENGINE_MARKER.lower() in swept_text:
        hit_classes.add(_THIRD_PARTY_ENGINE_CLASS_LABEL)
    if hit_classes:
        classes = sorted(hit_classes)
        return {
            "status": "withheld_marker_present",
            "markerClasses": classes,
            "reason": (
                f"{len(classes)} marker class(es) present: " + ", ".join(classes)
            ),
        }
    return {"status": "publishable", "markerClasses": [], "reason": None}


# ---------------------------------------------------------------------
# Row assembly.
# ---------------------------------------------------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _chip_identity() -> str:
    """The CPU chip's own brand string (e.g. ``Apple M3 Ultra``), via
    ``sysctl -n machdep.cpu.brand_string`` -- macOS-only, but this MUST
    NEVER crash and MUST NEVER fall back to anything hostname-derived: the
    defect this closes is exactly a hostname (the ``platform`` module's
    ``node()`` lookup) leaking into a published row, so silently
    reinstating a hostname as a fallback here would just reopen the same
    defect one layer down. Any failure -- no ``sysctl`` on PATH (non-macOS), a
    nonzero return code, a timeout, or empty stdout -- returns the literal
    string ``"unknown"`` instead, the same fail-closed shape as
    ``_listening_pid``/``_cmdline_for_pid`` above.
    """
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    brand = result.stdout.strip()
    return brand or "unknown"


def _boundary(args: argparse.Namespace, prompts: Sequence[str], prompt_is_default: bool) -> str:
    # States the prompt COUNT and each prompt's own length (never the
    # prompt text itself -- a boundary is a compact fingerprint of the
    # workload, not the workload) so a row is readable as stating exactly
    # which workload was measured, for either the default 3-prompt set or
    # an operator-supplied one of any size (a single `--prompt` included).
    prompt_set_label = (
        f"default-{len(prompts)}-prompt-set" if prompt_is_default else "custom-prompt-set"
    )
    chars_desc = "/".join(str(len(prompt)) for prompt in prompts)
    boundary = (
        f"chip={_chip_identity()} ({platform.machine()}); "
        f"promptSet={prompt_set_label} (prompts={len(prompts)}; chars={chars_desc}); "
        f"maxTokens={args.max_tokens}; runs={args.runs}; "
        f"warmup={args.warmup}; temperature={args.temperature}"
    )
    if args.host_label:
        # OPT-IN only -- see module docstring and --host-label's own help
        # text: absence must be the default, and this must never default
        # to anything machine-derived itself (that would just reintroduce
        # the hostname defect through a second door).
        boundary += f"; hostLabel={args.host_label}"
    return boundary


def run_bench(args: argparse.Namespace) -> "tuple[int, Optional[dict]]":
    """Runs the full measurement plan for ``args`` and returns
    ``(exit_code, row)``. ``row`` is ``None`` only when a reading could not
    be taken at all (see ``BenchError``); every other outcome -- including
    every control refusal -- still returns a row, so a refusal's own
    measured facts are never thrown away, only its ratio (or, for C-owner,
    the whole run).
    """
    # ``--prompt`` is repeatable (``action="append"``): supplying it one or
    # more times REPLACES the default 3-prompt set entirely with exactly
    # the prompts given, in the order given -- a single `--prompt` yields a
    # one-prompt set, same as before this module supported repeating it.
    prompt_is_default = args.prompt is None
    prompts = list(args.prompt) if args.prompt is not None else list(DEFAULT_PROMPTS)
    port = urllib.parse.urlsplit(args.base_url).port

    first_reference: Optional[ArmResult] = None
    last_reference: Optional[ArmResult] = None
    reference_arm: Optional[ArmResult] = None

    try:
        if args.reference_model:
            # Sandwiched: the reference arm is measured before AND after the
            # candidate so C-drift can tell a genuinely drifting server from
            # a stable one (see module docstring).
            first_reference = measure_arm(
                args.base_url, args.reference_model, prompts, args.max_tokens,
                args.runs, args.api_key, args.timeout, args.warmup, args.temperature,
            )
            candidate = measure_arm(
                args.base_url, args.model, prompts, args.max_tokens,
                args.runs, args.api_key, args.timeout, args.warmup, args.temperature,
            )
            last_reference = measure_arm(
                args.base_url, args.reference_model, prompts, args.max_tokens,
                args.runs, args.api_key, args.timeout, args.warmup, args.temperature,
            )
            combined_readings = first_reference.readings + last_reference.readings
            # Combined over the two bookends' own PASS rates (each already
            # pooled per pass by measure_arm/_pool_pass), never recomputed
            # from raw per-reading rates -- with a multi-prompt set, a
            # per-reading median would mix different prompts' individual
            # rates into one number instead of each pass's own pooled one.
            combined_pass_rates = first_reference.pass_rates + last_reference.pass_rates
            combined_measurable_rates = [rate for rate in combined_pass_rates if rate is not None]
            reference_median = (
                statistics.median(combined_measurable_rates) if combined_measurable_rates else None
            )
            reference_arm = ArmResult(
                model=args.reference_model,
                readings=combined_readings,
                median_decode_tok_s=reference_median,
                # Both bookend measure_arm() calls discarded their own
                # warmup passes (see design decision 1: every measure_arm
                # invocation warms up independently) -- the combined arm's
                # count is their sum, not either one alone.
                warmup_discarded=first_reference.warmup_discarded + last_reference.warmup_discarded,
                pass_rates=combined_pass_rates,
            )
            arms = [candidate, reference_arm]
        else:
            candidate = measure_arm(
                args.base_url, args.model, prompts, args.max_tokens,
                args.runs, args.api_key, args.timeout, args.warmup, args.temperature,
            )
            arms = [candidate]
    except BenchError as error:
        print(f"fastmlx bench: {error}", file=sys.stderr)
        return 1, None

    tokens = tokens_control(arms)
    drift = drift_control(args.reference_model, first_reference, last_reference, args.drift_tolerance)
    magnitude = magnitude_control(arms, args.magnitude_floor, args.magnitude_ceiling)
    owner = owner_control(args.expect_listener_pid, port)
    flags = flags_control(args.expect_listener_pid)

    ratio = None
    refusal_reasons: List[str] = []
    if args.reference_model:
        if tokens["status"] != "verified":
            refusal_reasons.append("ratio refused: " + tokens["reason"])
        elif drift["status"] != "verified":
            refusal_reasons.append("ratio refused: " + drift["reason"])
        elif candidate.median_decode_tok_s is None or reference_arm.median_decode_tok_s is None:
            # Either arm has zero measurable readings: there is no rate to
            # divide, not a 0.0 or a crash on `None / float`.
            unmeasurable_arms = [
                arm.model
                for arm in (candidate, reference_arm)
                if arm.median_decode_tok_s is None
            ]
            refusal_reasons.append(
                "ratio refused: no measurable decode-rate readings for "
                + ", ".join(unmeasurable_arms)
            )
        else:
            ratio = candidate.median_decode_tok_s / reference_arm.median_decode_tok_s

    if owner["status"] == "mismatch":
        refusal_reasons.append("refused: " + owner["reason"])

    row = {
        "schema": SCHEMA,
        "generatedAt": _utc_now_iso(),
        "baseUrl": args.base_url,
        "boundary": _boundary(args, prompts, prompt_is_default),
        "arms": [arm.to_json() for arm in arms],
        "ratio": ratio,
        "controls": {
            "tokens": tokens,
            "drift": drift,
            "magnitude": magnitude,
            "owner": owner,
            "flags": flags,
        },
    }
    # Computed LAST, over the row as otherwise complete, and attached only
    # after -- this ordering is the ONLY thing keeping the verdict out of
    # its own scanned text; it is a calling convention, not a guard the
    # function enforces (see its own docstring). This
    # verdict is metadata: it is deliberately never added to
    # refusal_reasons, so it never changes the exit code (see module
    # docstring's own-exit-code paragraph and the decision doc this
    # implements).
    row["publishable"] = publishability_control(row)

    if refusal_reasons:
        for reason in refusal_reasons:
            print(f"fastmlx bench: {reason}", file=sys.stderr)
        return 1, row
    return 0, row


# ---------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------
def format_text(row: dict) -> str:
    lines = [f"fastmlx bench: {row['baseUrl']}"]
    for arm in row["arms"]:
        median = arm["medianDecodeTokS"]
        median_desc = f"{median:.2f} tok/s" if median is not None else "UNMEASURABLE"
        lines.append(
            f"  {arm['model']}: median {median_desc} decode "
            f"over {len(arm['readings'])} run(s)"
        )
    if row["ratio"] is not None:
        lines.append(f"  ratio (candidate/reference): {row['ratio']:.3f}x")
    controls = row["controls"]
    for name in ("tokens", "drift", "magnitude", "owner", "flags"):
        control = controls[name]
        status = control.get("status")
        if name == "magnitude":
            status = "implausible" if control["magnitudeImplausible"] else "plausible"
        reason = control.get("reason")
        suffix = f" ({reason})" if reason else ""
        lines.append(f"  {name}: {status}{suffix}")
    publishable = row.get("publishable")
    if publishable is not None:
        parts = []
        if publishable.get("reason"):
            parts.append(f"({publishable['reason']})")
        classes = publishable.get("markerClasses") or []
        if classes:
            parts.append("[" + ", ".join(classes) + "]")
        pub_suffix = (" " + " ".join(parts)) if parts else ""
        lines.append(f"  publishable: {publishable['status']}{pub_suffix}")
    lines.append(f"  boundary: {row['boundary']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------
class _UsageErrorArgumentParser(argparse.ArgumentParser):
    """Same as ``argparse.ArgumentParser``, except a usage error exits 64
    (EX_USAGE) instead of argparse's default of 2 -- see module docstring.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(_EXIT_USAGE_ERROR, f"{self.prog}: error: {message}\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = _UsageErrorArgumentParser(
        prog="fastmlx-bench",
        description=(
            "fastmlx bench: an engine-agnostic decode-throughput measurer "
            "speaking OpenAI-compatible /v1/chat/completions over HTTP."
        ),
    )
    parser.add_argument(
        "--base-url", required=True,
        help="Base URL of the OpenAI-compatible server to measure (e.g. http://127.0.0.1:8080).",
    )
    parser.add_argument(
        "--model", required=True,
        help="Model name requested for the measured (candidate) arm.",
    )
    parser.add_argument(
        "--prompt", action="append", default=None,
        help=(
            "Custom prompt to send instead of this module's fixed default "
            "3-prompt set; repeatable -- each occurrence adds one prompt, "
            "and ANY occurrence REPLACES the default set entirely with "
            "exactly the prompt(s) given, in the order given (a single "
            "--prompt yields a one-prompt set)."
        ),
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"Maximum completion tokens requested per pass (default: {DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS,
        help=f"Measured passes per arm; the arm's median decode rate is taken over these (default: {DEFAULT_RUNS}).",
    )
    parser.add_argument(
        "--reference-model", default=None,
        help="Reference model name; when given, runs ratio mode with the candidate sandwiched between two reference-arm measurements.",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Bearer token sent in the Authorization header, if the server requires one.",
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-request timeout in seconds before a pass is treated as unreachable (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--magnitude-floor", type=float, default=DEFAULT_MAGNITUDE_FLOOR,
        help=(
            f"Median decode rate in tok/s below which a reading is reported as "
            f"implausibly slow; reported only, never voids a run (default: {DEFAULT_MAGNITUDE_FLOOR})."
        ),
    )
    parser.add_argument(
        "--magnitude-ceiling", type=float, default=DEFAULT_MAGNITUDE_CEILING,
        help=(
            f"Median decode rate in tok/s above which a reading is reported as "
            f"implausibly fast; reported only, never voids a run (default: {DEFAULT_MAGNITUDE_CEILING})."
        ),
    )
    parser.add_argument(
        "--drift-tolerance", type=float, default=DEFAULT_DRIFT_TOLERANCE,
        help=(
            "Maximum fractional drift allowed between the first and last "
            f"reference-arm median in ratio mode before the ratio is voided (default: {DEFAULT_DRIFT_TOLERANCE})."
        ),
    )
    parser.add_argument(
        "--expect-listener-pid", type=int, default=None,
        help="PID expected to be LISTENing on --base-url's port; a mismatch refuses the run (C-owner).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit the result row as one line of JSON on stdout instead of human-readable text.",
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP,
        help=(
            "Passes executed and discarded per arm measurement before the "
            f"measured passes begin; 0 opts out (default: {DEFAULT_WARMUP})."
        ),
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help=(
            "Sampling temperature sent with every pass, warmup and measured "
            "alike; a timing instrument must hold its workload fixed, and "
            "with sampling on, decodeTokS's length-sensitivity turns every "
            f"reading into a different workload (default: {DEFAULT_TEMPERATURE})."
        ),
    )
    parser.add_argument(
        "--host-label", default=None,
        help=(
            "Opt-in free-text label identifying this box, appended to the "
            "row's boundary as hostLabel=<text>; supplying this makes the "
            "row INTERNAL-ONLY (default: not set, so the boundary records "
            "only the chip, never a hostname)."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.warmup < 0:
        # argparse's type=int happily accepts a negative value -- this is a
        # usage error (design decision 4), not a measurement outcome, so it
        # must route through parser.error()'s exit-64 path (see
        # _UsageErrorArgumentParser), never a bare ValueError or exit 1.
        parser.error(f"--warmup must be >= 0 (got {args.warmup})")
    if args.temperature < 0:
        # Same shape as --warmup's check above: a negative sampling
        # temperature is nonsensical, so it is a usage error, not a
        # measurement outcome -- exit 64 via parser.error(), never a bare
        # ValueError or exit 1.
        parser.error(f"--temperature must be >= 0 (got {args.temperature})")
    code, row = run_bench(args)
    if row is not None:
        if args.json:
            print(json.dumps(row))
        else:
            print(format_text(row))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
