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
"""

from __future__ import annotations

import argparse
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
from typing import List, Optional, Sequence


SCHEMA = "fastmlx-bench-row-v1"

TOKEN_SOURCE_USAGE = "usage.completion_tokens"
TOKEN_SOURCE_SSE_COUNT = "sse_chunk_count"

DEFAULT_PROMPT = (
    "In two or three sentences, describe how the water cycle moves water "
    "between the ocean, the atmosphere, and the land."
)
DEFAULT_MAX_TOKENS = 128
DEFAULT_RUNS = 3
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAGNITUDE_FLOOR = 25.0
DEFAULT_MAGNITUDE_CEILING = 2000.0
DEFAULT_DRIFT_TOLERANCE = 0.05

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
    )

    def __init__(
        self,
        decode_tok_s: Optional[float],
        ttft_s: float,
        completion_tokens: int,
        token_source: str,
        measurable: bool,
        unmeasurable_reason: Optional[str],
    ):
        self.decode_tok_s = decode_tok_s
        self.ttft_s = ttft_s
        self.completion_tokens = completion_tokens
        self.token_source = token_source
        self.measurable = measurable
        self.unmeasurable_reason = unmeasurable_reason

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

    __slots__ = ("model", "readings", "median_decode_tok_s")

    def __init__(self, model: str, readings: List[Reading], median_decode_tok_s: Optional[float]):
        self.model = model
        self.readings = readings
        self.median_decode_tok_s = median_decode_tok_s

    def to_json(self) -> dict:
        return {
            "model": self.model,
            "readings": [reading.to_json() for reading in self.readings],
            "medianDecodeTokS": self.median_decode_tok_s,
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
    )


def stream_chat_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    api_key: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
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


def measure_arm(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    runs: int,
    api_key: Optional[str],
    timeout: float,
) -> ArmResult:
    readings = [
        stream_chat_completion(base_url, model, prompt, max_tokens, api_key, timeout)
        for _ in range(runs)
    ]
    # The median is over MEASURABLE readings only -- an unmeasurable
    # reading's decode_tok_s is None and must never enter a median (it is
    # not a 0.0 to average in, nor comparable to the measured values at
    # all). Zero measurable readings means the arm's median is itself
    # unmeasurable (None), not 0.0.
    measurable_rates = [reading.decode_tok_s for reading in readings if reading.measurable]
    median = statistics.median(measurable_rates) if measurable_rates else None
    return ArmResult(model=model, readings=readings, median_decode_tok_s=median)


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
# Row assembly.
# ---------------------------------------------------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _boundary(args: argparse.Namespace, prompt: str, prompt_is_default: bool) -> str:
    return (
        f"host={platform.node() or 'unknown'} ({platform.machine()}); "
        f"promptSet={'default-fixed-prompt' if prompt_is_default else 'custom-prompt'} "
        f"(chars={len(prompt)}); maxTokens={args.max_tokens}; runs={args.runs}"
    )


def run_bench(args: argparse.Namespace) -> "tuple[int, Optional[dict]]":
    """Runs the full measurement plan for ``args`` and returns
    ``(exit_code, row)``. ``row`` is ``None`` only when a reading could not
    be taken at all (see ``BenchError``); every other outcome -- including
    every control refusal -- still returns a row, so a refusal's own
    measured facts are never thrown away, only its ratio (or, for C-owner,
    the whole run).
    """
    prompt_is_default = args.prompt is None
    prompt = args.prompt if args.prompt is not None else DEFAULT_PROMPT
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
                args.base_url, args.reference_model, prompt, args.max_tokens,
                args.runs, args.api_key, args.timeout,
            )
            candidate = measure_arm(
                args.base_url, args.model, prompt, args.max_tokens,
                args.runs, args.api_key, args.timeout,
            )
            last_reference = measure_arm(
                args.base_url, args.reference_model, prompt, args.max_tokens,
                args.runs, args.api_key, args.timeout,
            )
            combined_readings = first_reference.readings + last_reference.readings
            # Same measurable-only filtering as measure_arm (see its
            # comment): an unmeasurable reading's decode_tok_s is None and
            # must never enter this median.
            combined_measurable_rates = [
                reading.decode_tok_s for reading in combined_readings if reading.measurable
            ]
            reference_median = (
                statistics.median(combined_measurable_rates) if combined_measurable_rates else None
            )
            reference_arm = ArmResult(
                model=args.reference_model,
                readings=combined_readings,
                median_decode_tok_s=reference_median,
            )
            arms = [candidate, reference_arm]
        else:
            candidate = measure_arm(
                args.base_url, args.model, prompt, args.max_tokens,
                args.runs, args.api_key, args.timeout,
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
        "boundary": _boundary(args, prompt, prompt_is_default),
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
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--reference-model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--magnitude-floor", type=float, default=DEFAULT_MAGNITUDE_FLOOR)
    parser.add_argument("--magnitude-ceiling", type=float, default=DEFAULT_MAGNITUDE_CEILING)
    parser.add_argument("--drift-tolerance", type=float, default=DEFAULT_DRIFT_TOLERANCE)
    parser.add_argument("--expect-listener-pid", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    code, row = run_bench(args)
    if row is not None:
        if args.json:
            print(json.dumps(row))
        else:
            print(format_text(row))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
