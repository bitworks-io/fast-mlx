"""Tests for ``scripts/fastmlx_bench.py``: the engine-agnostic decode-
throughput measurer.

Every test drives the REAL module against a small stdlib
``http.server``-based fake ``/v1/chat/completions`` server, bound to an
ephemeral loopback port in a background thread -- the same idiom
``test_fastmlx_proxy.py`` uses for its fake upstream.

Timing note: this sandbox's ``time.sleep`` has been observed to overshoot
its requested duration by ~0.15-0.25s per call (a scheduler-tick artifact
of the execution environment, not of this module). Tests that need a
"slow" vs. "fast" comparison therefore use gaps an order of magnitude
larger than that overshoot (hundreds of ms to full seconds) and/or derive
their own expected values from ACTUALLY-OBSERVED server-side send
timestamps (a real ``time.monotonic()`` reading, taken in the same
process, immediately after each flush) rather than from the nominal sleep
argument -- so an assertion never depends on ``time.sleep(x)`` actually
sleeping close to ``x``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import shutil
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


BENCH_PATH = Path(__file__).resolve().parents[1] / "fastmlx_bench.py"
_SPEC = importlib.util.spec_from_file_location("fastmlx_bench", BENCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
FASTMLX_BENCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(FASTMLX_BENCH)


# ---------------------------------------------------------------------
# Fake upstream: a stdlib http.server standing in for ANY OpenAI-compatible
# server, dispatching per-request to a test-supplied responder keyed by the
# request's own JSON body -- see start_fake_server / make_dispatch_responder.
# ---------------------------------------------------------------------
class _FakeChatHandler(BaseHTTPRequestHandler):
    def log_message(self, *args, **kwargs):  # noqa: D401 - silence access log
        pass

    def _dispatch(self):
        self.server.responder(self)

    do_POST = _dispatch


def _read_request_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length") or 0)
    raw = handler.rfile.read(length) if length else b""
    return json.loads(raw) if raw else {}


def start_fake_server(responder) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeChatHandler)
    server.responder = responder
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server._test_thread = thread
    return server


def stop_server(server: ThreadingHTTPServer) -> None:
    server.shutdown()
    server.server_close()
    server._test_thread.join(timeout=5)


def _send_stream_headers(handler: BaseHTTPRequestHandler) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.end_headers()


def _write_sse(handler: BaseHTTPRequestHandler, obj: dict) -> None:
    handler.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode("utf-8"))
    handler.wfile.flush()


def _write_done(handler: BaseHTTPRequestHandler) -> None:
    handler.wfile.write(b"data: [DONE]\n\n")
    handler.wfile.flush()


def _content_event(text: str) -> dict:
    return {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}


def _usage_event(completion_tokens: int, prompt_tokens: int = 10) -> dict:
    return {
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def make_dispatch_responder(behaviors: dict):
    """A responder that reads the request body ONCE and hands off to
    ``behaviors[body["model"]]`` -- lets one fake server stand in for both
    a candidate and a reference model with different streamed behavior.
    """

    def responder(handler):
        body = _read_request_body(handler)
        behaviors[body.get("model")](handler)

    return responder


def make_simple_stream(
    content_chunks: int,
    completion_tokens=None,
    inter_chunk_sleep: float = 0.0,
    pre_usage_sleep: float = 0.0,
    send_times: list = None,
):
    """A responder: streams ``content_chunks`` content deltas, optionally
    followed (after ``pre_usage_sleep``) by a usage-bearing chunk, then
    ``[DONE]``. If ``send_times`` is given, the real (observed, same-
    process) ``time.monotonic()`` at each content chunk's flush is appended
    to it -- ground truth for a test to check the client's own timing
    against, independent of whether ``time.sleep`` overshot its argument.
    """

    def fn(handler):
        _send_stream_headers(handler)
        for i in range(content_chunks):
            _write_sse(handler, _content_event(f"tok{i}"))
            if send_times is not None:
                send_times.append(time.monotonic())
            if i < content_chunks - 1:
                time.sleep(inter_chunk_sleep)
        if completion_tokens is not None:
            time.sleep(pre_usage_sleep)
            _write_sse(handler, _usage_event(completion_tokens))
        _write_done(handler)

    return fn


def make_stalled_then_fast_stream(
    content_chunks: int, completion_tokens: int, stall_seconds: float
):
    """A responder: a long stall BEFORE the first content chunk, then every
    remaining chunk written back-to-back with no sleep at all -- isolates
    TTFT (which must absorb the whole stall) from decodeTokS (which must
    not be depressed by it), without needing any small, jitter-prone sleep.
    """

    def fn(handler):
        _send_stream_headers(handler)
        time.sleep(stall_seconds)
        for i in range(content_chunks):
            _write_sse(handler, _content_event(f"tok{i}"))
        _write_sse(handler, _usage_event(completion_tokens))
        _write_done(handler)

    return fn


class FastmlxBenchTestCase(unittest.TestCase):
    def setUp(self):
        self.servers = []
        self.addCleanup(self._stop_all)

    def _stop_all(self):
        for server in self.servers:
            stop_server(server)

    def start(self, responder) -> str:
        server = start_fake_server(responder)
        self.servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}"

    # ------------------------------------------------------------------
    # THE required first test: decode rate derives from usage.
    # completion_tokens, never from the SSE chunk count.
    # ------------------------------------------------------------------
    def test_decode_rate_uses_usage_completion_tokens_not_sse_chunk_count(self):
        send_times: list = []
        content_chunks = 3
        completion_tokens = 21  # deliberately >> content_chunks (M != N)
        base_url = self.start(
            make_simple_stream(
                content_chunks,
                completion_tokens=completion_tokens,
                inter_chunk_sleep=0.3,
                pre_usage_sleep=0.4,
                send_times=send_times,
            )
        )

        reading = FASTMLX_BENCH.stream_chat_completion(
            base_url, "candidate", "hi", max_tokens=32, timeout=10
        )

        self.assertEqual(reading.completion_tokens, completion_tokens)
        self.assertEqual(reading.token_source, FASTMLX_BENCH.TOKEN_SOURCE_USAGE)

        # Ground truth from the SAME real clock, taken server-side at each
        # content chunk's own flush -- independent of time.sleep's overshoot
        # (see module docstring): loopback transit is sub-millisecond, so
        # this is a faithful proxy for the client's own t_first/t_last.
        server_elapsed = send_times[-1] - send_times[0]
        expected_rate = (completion_tokens - 1) / server_elapsed
        self.assertAlmostEqual(
            reading.decode_tok_s, expected_rate, delta=max(0.25 * expected_rate, 1.0)
        )

        # Anti-vacuity: the WRONG (chunk-count-based) rate the skeleton/a
        # mutant would compute is far off from what was actually measured --
        # this proves the assertion above is discriminating, not just loose.
        wrong_rate = (content_chunks - 1) / server_elapsed
        self.assertGreater(abs(reading.decode_tok_s - wrong_rate), 0.5 * expected_rate)

    # ------------------------------------------------------------------
    # No usage at all -> UNVERIFIED reading; ratio mode refuses.
    # ------------------------------------------------------------------
    def test_no_usage_marks_reading_unverified(self):
        base_url = self.start(
            make_simple_stream(3, completion_tokens=None, inter_chunk_sleep=0.05)
        )
        reading = FASTMLX_BENCH.stream_chat_completion(
            base_url, "candidate", "hi", max_tokens=32, timeout=10
        )
        self.assertEqual(reading.token_source, FASTMLX_BENCH.TOKEN_SOURCE_SSE_COUNT)
        self.assertEqual(reading.completion_tokens, 3)

    def test_ratio_mode_refuses_when_any_reading_is_unverified(self):
        behaviors = {
            "candidate": make_simple_stream(3, completion_tokens=None, inter_chunk_sleep=0.02),
            "reference": make_simple_stream(3, completion_tokens=11, inter_chunk_sleep=0.02),
        }
        base_url = self.start(make_dispatch_responder(behaviors))
        argv = [
            "--base-url", base_url,
            "--model", "candidate",
            "--reference-model", "reference",
            "--runs", "1",
            "--timeout", "10",
            "--json",
        ]
        code, stdout, stderr = self._run_main(argv)
        self.assertNotEqual(code, 0)
        doc = json.loads(stdout)
        self.assertIsNone(doc["ratio"])
        self.assertEqual(doc["controls"]["tokens"]["status"], "unverified")
        self.assertIn("candidate", doc["controls"]["tokens"]["reason"])
        self.assertIn("ratio refused", stderr)

    # ------------------------------------------------------------------
    # Drift beyond tolerance VOIDs the ratio, naming both medians.
    # ------------------------------------------------------------------
    def test_drift_beyond_tolerance_voids_the_ratio(self):
        call_index = {"reference": 0}
        lock = threading.Lock()

        def reference_fn(handler):
            with lock:
                call_index["reference"] += 1
                is_first_bookend = call_index["reference"] <= 1
            # First bookend: fast (near-zero gaps). Last bookend: slow
            # (a real, large sleep) -- an unmistakable drift, not a
            # tolerance-edge case that could flake on jitter.
            sleep = 0.0 if is_first_bookend else 1.0
            _send_stream_headers(handler)
            for i in range(3):
                _write_sse(handler, _content_event(f"tok{i}"))
                if i < 2:
                    time.sleep(sleep)
            _write_sse(handler, _usage_event(13))
            _write_done(handler)

        behaviors = {
            "candidate": make_simple_stream(3, completion_tokens=13, inter_chunk_sleep=0.0),
            "reference": reference_fn,
        }
        base_url = self.start(make_dispatch_responder(behaviors))
        argv = [
            "--base-url", base_url,
            "--model", "candidate",
            "--reference-model", "reference",
            "--runs", "1",
            "--timeout", "10",
            "--drift-tolerance", "0.05",
            "--json",
        ]
        code, stdout, stderr = self._run_main(argv)
        self.assertNotEqual(code, 0)
        doc = json.loads(stdout)
        self.assertIsNone(doc["ratio"])
        drift = doc["controls"]["drift"]
        self.assertEqual(drift["status"], "void")
        self.assertIsNotNone(drift["firstMedianDecodeTokS"])
        self.assertIsNotNone(drift["lastMedianDecodeTokS"])
        reason = drift["reason"]
        self.assertIn(f"{drift['firstMedianDecodeTokS']:.3f}", reason)
        self.assertIn(f"{drift['lastMedianDecodeTokS']:.3f}", reason)
        self.assertIn("ratio refused", stderr)

    # ------------------------------------------------------------------
    # A missing required flag exits 64, not argparse's default of 2.
    # ------------------------------------------------------------------
    def test_missing_required_flag_exits_64(self):
        code, stdout, stderr = self._run_main(["--base-url", "http://127.0.0.1:9"])
        self.assertEqual(code, 64)
        self.assertEqual(stdout, "")
        self.assertIn("--model", stderr)

    def test_unknown_flag_exits_64(self):
        code, _, stderr = self._run_main(
            ["--base-url", "http://127.0.0.1:9", "--model", "x", "--not-a-real-flag"]
        )
        self.assertEqual(code, 64)

    # ------------------------------------------------------------------
    # TTFT absorbs a long prefill stall; decodeTokS is NOT depressed by it.
    # ------------------------------------------------------------------
    def test_ttft_excluded_from_decode_rate(self):
        stall_seconds = 1.5
        completion_tokens = 9
        base_url = self.start(
            make_stalled_then_fast_stream(4, completion_tokens, stall_seconds)
        )
        reading = FASTMLX_BENCH.stream_chat_completion(
            base_url, "candidate", "hi", max_tokens=32, timeout=10
        )
        # TTFT must reflect (at least) the stall.
        self.assertGreaterEqual(reading.ttft_s, stall_seconds * 0.8)
        # decodeTokS must NOT be depressed by the stall: the 4 content
        # chunks were flushed back-to-back with no sleep at all, so the
        # true decode interval is a small fraction of a second -- if the
        # stall had leaked into the denominator the rate would be under
        # completion_tokens/stall_seconds ~= 6 tok/s. A back-to-back local
        # flush sequence comfortably clears 100x that on any real machine.
        self.assertGreater(reading.decode_tok_s, 50.0)

    # ------------------------------------------------------------------
    # Defect 1: a single-token (or otherwise zero-width) decode interval
    # is UNMEASURABLE, never reported as a measured 0.0 -- see the module
    # docstring's "unreachable/unmeasurable is not zero" rule.
    # ------------------------------------------------------------------
    def test_single_token_reading_is_unmeasurable_not_zero(self):
        base_url = self.start(
            make_simple_stream(1, completion_tokens=1, inter_chunk_sleep=0.0)
        )
        reading = FASTMLX_BENCH.stream_chat_completion(
            base_url, "candidate", "hi", max_tokens=32, timeout=10
        )
        self.assertIsNone(reading.decode_tok_s)
        self.assertFalse(reading.measurable)
        self.assertTrue(reading.unmeasurable_reason)
        self.assertIn("completion_tokens=1", reading.unmeasurable_reason)

        # TTFT IS measurable from a single chunk and must still be
        # reported -- it must not be nulled out alongside decodeTokS.
        self.assertGreaterEqual(reading.ttft_s, 0.0)

        # The important part: serialize exactly as a row would, and prove
        # no 0.0 decode rate is anywhere in it -- a 0/0 non-measurement
        # dressed up as a measured 0.0 is the defect being fixed.
        payload = reading.to_json()
        self.assertIsNone(payload["decodeTokS"])
        self.assertFalse(payload["measurable"])
        serialized = json.dumps(payload)
        # Narrowly targeted at the decodeTokS field itself (the honest
        # unmeasurableReason text legitimately contains "0.0" elsewhere,
        # e.g. "elapsed=0.000000s") -- the defect is a *decode rate* of
        # 0.0, not the digits "0.0" appearing anywhere in the row.
        self.assertNotIn('"decodeTokS": 0.0', serialized)
        self.assertNotIn('"decodeTokS":0.0', serialized)

    def test_arm_with_no_measurable_readings_refuses_a_ratio(self):
        behaviors = {
            # Every candidate reading is a single-token, unmeasurable one.
            "candidate": make_simple_stream(1, completion_tokens=1, inter_chunk_sleep=0.0),
            "reference": make_simple_stream(3, completion_tokens=11, inter_chunk_sleep=0.05),
        }
        base_url = self.start(make_dispatch_responder(behaviors))
        argv = [
            "--base-url", base_url,
            "--model", "candidate",
            "--reference-model", "reference",
            "--runs", "1",
            "--timeout", "10",
            # Generous on purpose: this test is about the candidate's
            # unmeasurable median refusing the ratio, not about C-drift's
            # own tolerance -- see test_drift_beyond_tolerance_voids_the_ratio
            # for that control's own dedicated coverage.
            "--drift-tolerance", "5.0",
            "--json",
        ]
        code, stdout, stderr = self._run_main(argv)
        self.assertNotEqual(code, 0)
        doc = json.loads(stdout)
        self.assertIsNone(doc["ratio"])
        candidate_arm = next(arm for arm in doc["arms"] if arm["model"] == "candidate")
        self.assertIsNone(candidate_arm["medianDecodeTokS"])
        self.assertIn("ratio refused", stderr)
        self.assertIn("candidate", stderr)

    # ------------------------------------------------------------------
    # not_applicable controls always carry a reason.
    # ------------------------------------------------------------------
    def test_not_applicable_controls_carry_a_reason(self):
        base_url = self.start(make_simple_stream(3, completion_tokens=11, inter_chunk_sleep=0.0))
        argv = ["--base-url", base_url, "--model", "candidate", "--runs", "1", "--timeout", "10", "--json"]
        code, stdout, _ = self._run_main(argv)
        self.assertEqual(code, 0)
        doc = json.loads(stdout)
        controls = doc["controls"]
        self.assertEqual(controls["drift"]["status"], "not_applicable")
        self.assertTrue(controls["drift"]["reason"])
        self.assertEqual(controls["owner"]["status"], "not_applicable")
        self.assertTrue(controls["owner"]["reason"])
        self.assertEqual(controls["flags"]["status"], "not_applicable")
        self.assertTrue(controls["flags"]["reason"])

    # ------------------------------------------------------------------
    # C-magnitude: below the floor is reported loudly but does not void a
    # single-arm run; above the floor is plausible.
    # ------------------------------------------------------------------
    def test_magnitude_below_floor_is_flagged_but_does_not_void(self):
        # Slow enough (large real sleeps) to land under a generous floor,
        # regardless of this sandbox's sleep overshoot in either direction.
        base_url = self.start(
            make_simple_stream(3, completion_tokens=3, inter_chunk_sleep=1.0)
        )
        argv = [
            "--base-url", base_url, "--model", "candidate", "--runs", "1",
            "--timeout", "10", "--magnitude-floor", "1000", "--json",
        ]
        code, stdout, _ = self._run_main(argv)
        self.assertEqual(code, 0)  # magnitude alone never voids a run
        doc = json.loads(stdout)
        magnitude = doc["controls"]["magnitude"]
        self.assertTrue(magnitude["magnitudeImplausible"])
        self.assertEqual(magnitude["floor"], 1000.0)
        self.assertIn("candidate", magnitude["reason"])
        # A below-floor case must say FLOOR, never CEILING -- the two
        # branches (too slow vs. too fast) must be distinguishable in the
        # reason text, not just in the boolean.
        self.assertIn("FLOOR", magnitude["reason"])
        self.assertNotIn("CEILING", magnitude["reason"])

    # ------------------------------------------------------------------
    # C-magnitude, defect 2: an implausibly HIGH rate (a degenerate,
    # too-short decode interval) must also be caught -- the floor alone
    # cannot catch the direction that flatters our own throughput.
    # ------------------------------------------------------------------
    def test_rate_above_magnitude_ceiling_is_flagged(self):
        # A real, measurable, unremarkable rate -- the point of this test
        # is that a LOW --magnitude-ceiling catches it, not that the
        # server produced a microsecond-scale flush (which would be
        # jitter-prone in this sandbox -- see module docstring).
        base_url = self.start(
            make_simple_stream(3, completion_tokens=3, inter_chunk_sleep=0.05)
        )
        argv = [
            "--base-url", base_url, "--model", "candidate", "--runs", "1",
            "--timeout", "10",
            "--magnitude-floor", "0",       # disable the floor so only the
                                             # ceiling branch is exercised
            "--magnitude-ceiling", "0.001", # far below any real rate
            "--json",
        ]
        code, stdout, _ = self._run_main(argv)
        self.assertEqual(code, 0)  # magnitude alone never voids a run
        doc = json.loads(stdout)
        magnitude = doc["controls"]["magnitude"]
        self.assertTrue(magnitude["magnitudeImplausible"])
        self.assertEqual(magnitude["ceiling"], 0.001)
        self.assertIn("candidate", magnitude["reason"])
        # The important part: the reason names the CEILING, not the FLOOR
        # -- a too-fast reading must not be reported as if it were the
        # (unrelated) too-slow failure mode.
        self.assertIn("CEILING", magnitude["reason"])
        self.assertNotIn("FLOOR", magnitude["reason"])

    def test_magnitude_above_floor_is_plausible(self):
        base_url = self.start(
            make_simple_stream(3, completion_tokens=3, inter_chunk_sleep=0.0)
        )
        # A high --magnitude-ceiling isolates this test to the FLOOR check
        # it names: 3 content chunks flushed back-to-back with no sleep at
        # all yield a very high rate (a genuinely too-short interval -- see
        # test_rate_above_magnitude_ceiling_is_flagged, which pins the
        # opposite, CEILING-crossing behaviour deliberately), so without a
        # raised ceiling here this reading would itself be correctly caught
        # by C-magnitude's new upper bound instead of testing the floor.
        argv = [
            "--base-url", base_url, "--model", "candidate", "--runs", "1",
            "--timeout", "10", "--magnitude-floor", "0.001",
            "--magnitude-ceiling", "1000000000", "--json",
        ]
        code, stdout, _ = self._run_main(argv)
        self.assertEqual(code, 0)
        doc = json.loads(stdout)
        magnitude = doc["controls"]["magnitude"]
        self.assertFalse(magnitude["magnitudeImplausible"])
        self.assertIsNone(magnitude["reason"])

    # ------------------------------------------------------------------
    # C-owner / C-flags: verified when the expected PID is this test
    # process's own (the fake server runs on a thread of THIS process, so
    # this process really is the listener) -- a real, non-mocked check.
    # ------------------------------------------------------------------
    @unittest.skipUnless(shutil.which("lsof") and shutil.which("ps"), "lsof/ps not on PATH")
    def test_owner_and_flags_verified_against_the_real_listener(self):
        import os

        base_url = self.start(make_simple_stream(2, completion_tokens=2, inter_chunk_sleep=0.0))
        argv = [
            "--base-url", base_url, "--model", "candidate", "--runs", "1",
            "--timeout", "10", "--expect-listener-pid", str(os.getpid()), "--json",
        ]
        code, stdout, _ = self._run_main(argv)
        self.assertEqual(code, 0)
        doc = json.loads(stdout)
        owner = doc["controls"]["owner"]
        self.assertEqual(owner["status"], "verified")
        self.assertEqual(owner["actualPid"], os.getpid())
        flags = doc["controls"]["flags"]
        self.assertEqual(flags["status"], "captured")
        self.assertTrue(flags["listenerCmdline"])

    @unittest.skipUnless(shutil.which("lsof"), "lsof not on PATH")
    def test_owner_mismatch_refuses_with_a_reason(self):
        base_url = self.start(make_simple_stream(2, completion_tokens=2, inter_chunk_sleep=0.0))
        # PID 1 is a real process (init/launchd) almost certainly NOT this
        # test process and NOT the fake server's listener.
        argv = [
            "--base-url", base_url, "--model", "candidate", "--runs", "1",
            "--timeout", "10", "--expect-listener-pid", "1", "--json",
        ]
        code, stdout, stderr = self._run_main(argv)
        self.assertNotEqual(code, 0)
        doc = json.loads(stdout)
        owner = doc["controls"]["owner"]
        self.assertEqual(owner["status"], "mismatch")
        self.assertIsNotNone(owner["reason"])
        self.assertIn("refused", stderr)

    # ------------------------------------------------------------------
    # --json prints exactly one line of JSON on stdout, nothing else.
    # ------------------------------------------------------------------
    def test_json_output_shape_and_no_extra_stdout_prose(self):
        base_url = self.start(make_simple_stream(3, completion_tokens=11, inter_chunk_sleep=0.0))
        argv = ["--base-url", base_url, "--model", "candidate", "--runs", "1", "--timeout", "10", "--json"]
        code, stdout, _ = self._run_main(argv)
        self.assertEqual(code, 0)
        self.assertEqual(stdout.count("\n"), 1)
        doc = json.loads(stdout)
        self.assertEqual(doc["schema"], "fastmlx-bench-row-v1")
        self.assertIn("boundary", doc)
        self.assertIn("maxTokens", doc["boundary"])
        self.assertIn("runs", doc["boundary"])

    def test_text_output_is_human_readable(self):
        base_url = self.start(make_simple_stream(3, completion_tokens=11, inter_chunk_sleep=0.0))
        argv = ["--base-url", base_url, "--model", "candidate", "--runs", "1", "--timeout", "10"]
        code, stdout, _ = self._run_main(argv)
        self.assertEqual(code, 0)
        self.assertIn("candidate", stdout)
        self.assertIn("tok/s", stdout)

    # ------------------------------------------------------------------
    def _run_main(self, argv: list):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                FASTMLX_BENCH.main(argv)
        return ctx.exception.code, stdout.getvalue(), stderr.getvalue()


if __name__ == "__main__":
    unittest.main()
