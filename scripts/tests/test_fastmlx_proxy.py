"""Tests for ``scripts/fastmlx_proxy.py``: the opt-in provenance reverse proxy
``fastmlx serve --front-port`` starts in front of the served engine.

Every test drives the REAL proxy server against a small stdlib
``http.server``-based fake upstream, both bound to ephemeral loopback ports in
background threads, with bounded timeouts throughout so a regression fails
fast instead of hanging the suite.
"""

from __future__ import annotations

import contextlib
import http.client
import importlib.util
import io
import itertools
import json
import os
import socket
import socketserver
import sys
import threading
import time
import unittest
import unittest.mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PROXY_PATH = Path(__file__).resolve().parents[1] / "fastmlx_proxy.py"
_SPEC = importlib.util.spec_from_file_location("fastmlx_proxy", PROXY_PATH)
assert _SPEC is not None and _SPEC.loader is not None
FASTMLX_PROXY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(FASTMLX_PROXY)


# A fixture plan shaped like the dict fastmlx_launch.py's `_run_serve` builds
# (see its own `plan` local): only the fields the proxy actually reads are
# populated. The `argv`/`model_path` entries below are deliberately
# fixture-only stand-ins for what a real plan would carry -- this file's own
# assertions prove neither ever reaches an HTTP response.
FIXTURE_MODEL_PATH = "/opt/fixture-models/fixture-pack"
FIXTURE_PLAN = {
    "fit": {"verdict": "GREEN", "fields": {"model": "stub"}},
    "card": {"id": "fixture-card@test", "model": {"repo": "example/FixtureModel"}},
    "admission": "admit",
    "argv": ["/opt/fixture-bin/engine", "--model-path", FIXTURE_MODEL_PATH],
    "residency": "resident",
    "engineBuild": {"status": "match", "card": "a" * 40, "launch": "a" * 40},
    "mtp": {"status": "not_exact", "divergentPrompts": 16, "prompts": 40},
}

UNMEASURED_PLAN = {
    **FIXTURE_PLAN,
    "mtp": {"status": "unmeasured", "divergentPrompts": None, "prompts": None},
}


class _FakeUpstreamHandler(BaseHTTPRequestHandler):
    """Delegates every method to ``self.server.responder(self)``, set fresh
    per test -- keeps each scenario's upstream behavior colocated with its
    test instead of one large dispatch table.
    """

    def log_message(self, *args, **kwargs):  # noqa: D401 - silence stdlib access log
        pass

    def _dispatch(self):
        self.server.responder(self)

    do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = do_HEAD = do_PATCH = _dispatch


def _read_request_body(handler: BaseHTTPRequestHandler) -> bytes:
    length = int(handler.headers.get("Content-Length") or 0)
    return handler.rfile.read(length) if length else b""


def start_fake_upstream(responder) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstreamHandler)
    server.responder = responder
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server._test_thread = thread
    return server


def stop_server(server: ThreadingHTTPServer) -> None:
    server.shutdown()
    server.server_close()
    server._test_thread.join(timeout=5)


def start_proxy(plan: dict, upstream_host: str, upstream_port: int, **kwargs):
    server = FASTMLX_PROXY.create_server(
        "127.0.0.1", 0, upstream_host, upstream_port, plan, **kwargs
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server._test_thread = thread
    return server


class ProxyRoundTripTests(unittest.TestCase):
    def setUp(self):
        self._servers = []
        self.addCleanup(self._stop_all)

    def _stop_all(self):
        for server in self._servers:
            stop_server(server)

    def _start_upstream(self, responder) -> ThreadingHTTPServer:
        server = start_fake_upstream(responder)
        self._servers.append(server)
        return server

    def _start_proxy(self, plan: dict, upstream: ThreadingHTTPServer, **kwargs):
        server = start_proxy(plan, "127.0.0.1", upstream.server_address[1], **kwargs)
        self._servers.append(server)
        return server

    def _assert_provenance_headers(self, headers, plan=FIXTURE_PLAN):
        self.assertEqual(headers.get("X-FastMLX-Admission"), plan["admission"])
        self.assertEqual(headers.get("X-FastMLX-Card"), plan["card"]["id"])
        self.assertEqual(headers.get("X-FastMLX-Fit"), plan["fit"]["verdict"])
        self.assertEqual(headers.get("X-FastMLX-Residency"), plan["residency"])
        engine_build = headers.get("X-FastMLX-Engine-Build")
        self.assertIsNotNone(engine_build)
        self.assertIn(plan["engineBuild"]["status"], engine_build)
        request_id = headers.get("X-FastMLX-Request-Id")
        self.assertIsNotNone(request_id)
        self.assertEqual(len(request_id), 32)
        int(request_id, 16)  # a uuid4 hex digest

    # ------------------------------------------------------------------
    # 1. Non-streamed POST round-trips status/body/content-type; every
    #    X-FastMLX-* header present with expected values.
    # ------------------------------------------------------------------
    def test_non_streamed_post_roundtrips_and_carries_provenance_headers(self):
        received = {}

        def responder(handler):
            received["body"] = _read_request_body(handler)
            received["path"] = handler.path
            payload = b'{"ok":true}'
            handler.send_response(201)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)

        upstream = self._start_upstream(responder)
        proxy = self._start_proxy(FIXTURE_PLAN, upstream)

        conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=5)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=b'{"x":1}',
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        body = resp.read()
        conn.close()

        self.assertEqual(resp.status, 201)
        self.assertEqual(body, b'{"ok":true}')
        self.assertEqual(resp.getheader("Content-Type"), "application/json")
        self.assertEqual(received["body"], b'{"x":1}')
        self.assertEqual(received["path"], "/v1/chat/completions")
        self.assertEqual(headers_dict(resp).get("X-FastMLX-MTP"), "not_exact; divergent=16/40")
        self._assert_provenance_headers(headers_dict(resp))

    # ------------------------------------------------------------------
    # 2. SSE, 3 chunks: the client must receive chunk 1 BEFORE the test
    #    releases chunk 3 -- proves per-chunk flushing, not buffering.
    # ------------------------------------------------------------------
    def test_sse_stream_flushes_before_upstream_finishes(self):
        release_rest = threading.Event()

        def responder(handler):
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.end_headers()
            handler.wfile.write(b"data: chunk1\n\n")
            handler.wfile.flush()
            self.assertTrue(release_rest.wait(timeout=5))
            handler.wfile.write(b"data: chunk2\n\n")
            handler.wfile.write(b"data: chunk3\n\n")
            handler.wfile.flush()

        upstream = self._start_upstream(responder)
        proxy = self._start_proxy(FIXTURE_PLAN, upstream)

        received_first = threading.Event()
        result = {}

        def read_client():
            conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=10)
            conn.request("GET", "/v1/stream")
            resp = conn.getresponse()
            first = resp.read1(4096) if hasattr(resp, "read1") else resp.read(1)
            result["first"] = first
            received_first.set()
            rest = resp.read()
            result["rest"] = rest
            conn.close()

        reader = threading.Thread(target=read_client)
        reader.start()
        self.assertTrue(received_first.wait(timeout=5), "client never received the first chunk")
        # The reader thread already has chunk 1 in hand while upstream is
        # still blocked on release_rest -- that ordering is the proof this
        # test exists for.
        self.assertFalse(release_rest.is_set())
        self.assertEqual(result["first"], b"data: chunk1\n\n")
        release_rest.set()
        reader.join(timeout=5)
        self.assertEqual(result["first"] + result["rest"], b"data: chunk1\n\ndata: chunk2\n\ndata: chunk3\n\n")

    # ------------------------------------------------------------------
    # 3. Upstream 400 / 500 pass through unchanged, with headers.
    # ------------------------------------------------------------------
    def test_upstream_error_statuses_pass_through_unchanged(self):
        for status in (400, 500):
            with self.subTest(status=status):

                def responder(handler, status=status):
                    payload = json.dumps({"status": status}).encode("utf-8")
                    handler.send_response(status)
                    handler.send_header("Content-Type", "application/json")
                    handler.send_header("Content-Length", str(len(payload)))
                    handler.end_headers()
                    handler.wfile.write(payload)

                upstream = self._start_upstream(responder)
                proxy = self._start_proxy(FIXTURE_PLAN, upstream)

                conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=5)
                conn.request("GET", "/v1/models")
                resp = conn.getresponse()
                body = resp.read()
                conn.close()

                self.assertEqual(resp.status, status)
                self.assertEqual(json.loads(body), {"status": status})
                self._assert_provenance_headers(headers_dict(resp))

    # ------------------------------------------------------------------
    # 4. Upstream unreachable -> 502 JSON, still carrying headers.
    # ------------------------------------------------------------------
    def test_upstream_unavailable_returns_502_with_headers(self):
        closed = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        closed.bind(("127.0.0.1", 0))
        dead_port = closed.getsockname()[1]
        closed.close()  # nothing listens here now

        server = FASTMLX_PROXY.create_server("127.0.0.1", 0, "127.0.0.1", dead_port, FIXTURE_PLAN)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        server._test_thread = thread
        self._servers.append(server)

        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        conn.request("GET", "/v1/models")
        resp = conn.getresponse()
        body = json.loads(resp.read())
        conn.close()

        self.assertEqual(resp.status, 502)
        self.assertEqual(body["error"]["type"], "upstream_unavailable")
        self._assert_provenance_headers(headers_dict(resp))

    # ------------------------------------------------------------------
    # 5. Client disconnects mid-stream -> upstream observes a write error
    #    (or closed connection) within a bound.
    # ------------------------------------------------------------------
    def test_client_disconnect_mid_stream_is_observed_by_upstream(self):
        disconnect_detected = threading.Event()

        def responder(handler):
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.end_headers()
            try:
                for _ in range(2000):
                    handler.wfile.write(b"x" * 4096)
                    handler.wfile.flush()
                    time.sleep(0.005)
            except (BrokenPipeError, ConnectionResetError, OSError):
                disconnect_detected.set()

        upstream = self._start_upstream(responder)
        proxy = self._start_proxy(FIXTURE_PLAN, upstream)

        sock = socket.create_connection(("127.0.0.1", proxy.server_address[1]), timeout=5)
        sock.sendall(b"GET /v1/stream HTTP/1.1\r\nHost: x\r\n\r\n")
        sock.recv(4096)  # read a little, then vanish
        sock.close()

        self.assertTrue(disconnect_detected.wait(timeout=5), "upstream never observed the disconnect")

    # ------------------------------------------------------------------
    # 6. /fastmlx/provenance: no argv, no fixture model path.
    # ------------------------------------------------------------------
    def test_provenance_endpoint_omits_argv_and_model_path(self):
        upstream = self._start_upstream(lambda handler: None)
        proxy = self._start_proxy(FIXTURE_PLAN, upstream)

        conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=5)
        conn.request("GET", "/fastmlx/provenance")
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()

        self.assertEqual(resp.status, 200)
        self.assertNotIn(b"argv", raw)
        self.assertNotIn(FIXTURE_MODEL_PATH.encode("utf-8"), raw)
        body = json.loads(raw)
        self.assertEqual(body["card"], FIXTURE_PLAN["card"]["id"])
        self.assertEqual(body["admission"], FIXTURE_PLAN["admission"])
        self.assertNotIn("argv", body)

    # ------------------------------------------------------------------
    # 7. Hop-by-hop headers stripped both directions.
    # ------------------------------------------------------------------
    def test_hop_by_hop_headers_are_stripped(self):
        def responder(handler):
            payload = b"ok"
            handler.send_response(200)
            handler.send_header("Connection", "x-foo")
            handler.send_header("Keep-Alive", "timeout=5")
            handler.send_header("Proxy-Authenticate", "Basic")
            handler.send_header("Upgrade", "h2c")
            handler.send_header("X-Foo", "1")
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)

        upstream = self._start_upstream(responder)
        proxy = self._start_proxy(FIXTURE_PLAN, upstream)

        conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=5)
        conn.request("GET", "/v1/models")
        resp = conn.getresponse()
        resp.read()
        headers = headers_dict(resp)
        conn.close()

        self.assertIsNone(headers.get("Keep-Alive"))
        self.assertIsNone(headers.get("Proxy-Authenticate"))
        self.assertIsNone(headers.get("Upgrade"))
        self.assertIsNone(headers.get("X-Foo"))

    # ------------------------------------------------------------------
    # 8. Chunked request body -> 411.
    # ------------------------------------------------------------------
    def test_chunked_request_body_returns_411(self):
        upstream = self._start_upstream(lambda handler: None)
        proxy = self._start_proxy(FIXTURE_PLAN, upstream)

        sock = socket.create_connection(("127.0.0.1", proxy.server_address[1]), timeout=5)
        sock.sendall(
            b"POST /v1/chat HTTP/1.1\r\n"
            b"Host: x\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"Content-Type: application/json\r\n"
            b"\r\n"
            b"5\r\nhello\r\n0\r\n\r\n"
        )
        sock.settimeout(5)
        response = b""
        try:
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
        except socket.timeout:
            pass
        sock.close()

        self.assertIn(b" 411 ", response)
        body_json = json.loads(response.split(b"\r\n\r\n", 1)[1])
        self.assertEqual(body_json["error"]["type"], "chunked_request_unsupported")

    # ------------------------------------------------------------------
    # 9. Authorization passes through, never logged.
    # ------------------------------------------------------------------
    def test_authorization_passthrough_and_not_logged(self):
        received = {}
        logged = threading.Event()
        log_lines = []

        def responder(handler):
            received["authorization"] = handler.headers.get("Authorization")
            payload = b"ok"
            handler.send_response(200)
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)

        def log_hook(entry):
            # A server-level hook the proxy calls synchronously right after
            # writing each JSONL log line -- deterministic, unlike polling
            # stderr on a sleep loop for a line that may not have landed yet.
            log_lines.append(entry)
            logged.set()

        upstream = self._start_upstream(responder)
        proxy = self._start_proxy(FIXTURE_PLAN, upstream, log_hook=log_hook)

        secret = "Bearer fixture-secret-token-9182"
        stderr_buffer = io.StringIO()
        with contextlib.redirect_stderr(stderr_buffer):
            conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=5)
            conn.request("GET", "/v1/models", headers={"Authorization": secret})
            resp = conn.getresponse()
            resp.read()
            conn.close()
            self.assertTrue(logged.wait(timeout=5), "log line was never written")

        self.assertEqual(received["authorization"], secret)
        self.assertEqual(len(log_lines), 1)
        self.assertNotIn(secret, json.dumps(log_lines[0]))
        self.assertNotIn(secret, stderr_buffer.getvalue())

    # ------------------------------------------------------------------
    # 10. MTP header: not_exact w/ counts, and unmeasured w/ no counts.
    # ------------------------------------------------------------------
    def test_mtp_header_reflects_status_and_counts(self):
        headers = FASTMLX_PROXY.build_provenance_headers(FIXTURE_PLAN)
        self.assertEqual(dict(headers)["X-FastMLX-MTP"], "not_exact; divergent=16/40")

        headers = FASTMLX_PROXY.build_provenance_headers(UNMEASURED_PLAN)
        self.assertEqual(dict(headers)["X-FastMLX-MTP"], "unmeasured")

    # ------------------------------------------------------------------
    # 11. CR/LF in a card id is sanitized out of the header value.
    # ------------------------------------------------------------------
    def test_crlf_in_card_id_is_sanitized(self):
        plan = {
            **FIXTURE_PLAN,
            "card": {"id": "evil\r\nX-Injected: 1"},
        }
        headers = dict(FASTMLX_PROXY.build_provenance_headers(plan))
        value = headers["X-FastMLX-Card"]
        self.assertNotIn("\r", value)
        self.assertNotIn("\n", value)

    # ------------------------------------------------------------------
    # 12. L5: X-FastMLX-* headers coming FROM upstream are stripped before
    #    the proxy's own provenance headers are added -- an upstream (or a
    #    misbehaving/compromised engine) must never be able to spoof or
    #    duplicate the proxy's own provenance headers.
    # ------------------------------------------------------------------
    def test_x_fastmlx_headers_from_upstream_are_stripped(self):
        def responder(handler):
            payload = b"ok"
            handler.send_response(200)
            handler.send_header("X-FastMLX-Evil", "spoofed")
            handler.send_header("x-fastmlx-another", "also-spoofed")
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)

        upstream = self._start_upstream(responder)
        proxy = self._start_proxy(FIXTURE_PLAN, upstream)

        conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=5)
        conn.request("GET", "/v1/models")
        resp = conn.getresponse()
        resp.read()
        headers = headers_dict(resp)
        conn.close()

        self.assertIsNone(headers.get("X-FastMLX-Evil"))
        self.assertIsNone(headers.get("x-fastmlx-another"))
        self.assertEqual(headers.get("X-FastMLX-Admission"), FIXTURE_PLAN["admission"])

    # ------------------------------------------------------------------
    # 13. L4(a): duplicate request headers (e.g. two Accept values) reach
    #    upstream as two headers, not collapsed into one by a dict().
    # ------------------------------------------------------------------
    def test_duplicate_request_headers_reach_upstream(self):
        received = {}

        def responder(handler):
            received["accept_all"] = handler.headers.get_all("Accept")
            payload = b"ok"
            handler.send_response(200)
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)

        upstream = self._start_upstream(responder)
        proxy = self._start_proxy(FIXTURE_PLAN, upstream)

        sock = socket.create_connection(("127.0.0.1", proxy.server_address[1]), timeout=5)
        sock.sendall(
            b"GET /v1/models HTTP/1.1\r\n"
            b"Host: x\r\n"
            b"Accept: application/json\r\n"
            b"Accept: text/event-stream\r\n"
            b"\r\n"
        )
        response = http.client.HTTPResponse(sock)
        response.begin()
        response.read()
        sock.close()

        self.assertEqual(received["accept_all"], ["application/json", "text/event-stream"])

    # ------------------------------------------------------------------
    # 14. L4(b)/L4(c): a relayed response carries exactly one Date and one
    #    Server header (the UPSTREAM's own); a proxy-GENERATED response
    #    (502) carries the proxy's own Server, never a Python version.
    # ------------------------------------------------------------------
    def test_relayed_response_has_exactly_one_date_and_server_from_upstream(self):
        def responder(handler):
            payload = b"ok"
            handler.send_response(200)  # BaseHTTPRequestHandler auto Server+Date
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)

        upstream = self._start_upstream(responder)
        proxy = self._start_proxy(FIXTURE_PLAN, upstream)

        conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=5)
        conn.request("GET", "/v1/models")
        resp = conn.getresponse()
        resp.read()
        server_headers = resp.msg.get_all("Server")
        date_headers = resp.msg.get_all("Date")
        conn.close()

        self.assertEqual(len(server_headers), 1)
        self.assertEqual(len(date_headers), 1)
        self.assertNotIn("fastmlx-proxy", server_headers[0])

    def test_proxy_generated_502_has_fastmlx_proxy_server_header(self):
        closed = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        closed.bind(("127.0.0.1", 0))
        dead_port = closed.getsockname()[1]
        closed.close()

        server = FASTMLX_PROXY.create_server("127.0.0.1", 0, "127.0.0.1", dead_port, FIXTURE_PLAN)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        server._test_thread = thread
        self._servers.append(server)

        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        conn.request("GET", "/v1/models")
        resp = conn.getresponse()
        resp.read()
        server_header = resp.getheader("Server")
        conn.close()

        self.assertIsNotNone(server_header)
        self.assertEqual(server_header.strip(), "fastmlx-proxy")
        self.assertNotIn("Python", server_header)

    # ------------------------------------------------------------------
    # 15. Request-side hop-by-hop stripping AND Host rewrite in one test:
    #    a client-named hop-by-hop token (via Connection:) and the standard
    #    Keep-Alive header never reach upstream; an ordinary header does;
    #    the client's own Host header is REPLACED by the proxy's own
    #    upstream host:port, never forwarded verbatim.
    # ------------------------------------------------------------------
    def test_request_hop_by_hop_stripped_and_host_rewritten(self):
        received = {}

        def responder(handler):
            received["headers"] = dict(handler.headers.items())
            payload = b"ok"
            handler.send_response(200)
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)

        upstream = self._start_upstream(responder)
        proxy = self._start_proxy(FIXTURE_PLAN, upstream)

        conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=5)
        conn.request(
            "GET",
            "/v1/models",
            headers={
                "Host": "client-supplied-host.example",
                "Connection": "keep-alive, x-custom-hop",
                "X-Custom-Hop": "should-be-stripped",
                "Keep-Alive": "timeout=5",
                "X-Keep": "kept",
            },
        )
        resp = conn.getresponse()
        resp.read()
        conn.close()

        upstream_headers = received["headers"]
        self.assertEqual(
            upstream_headers.get("Host"), f"127.0.0.1:{upstream.server_address[1]}"
        )
        self.assertNotIn("X-Custom-Hop", upstream_headers)
        self.assertNotIn("Keep-Alive", upstream_headers)
        self.assertEqual(upstream_headers.get("X-Keep"), "kept")

    # ------------------------------------------------------------------
    # 16. HEAD and 204: no body either direction, never mishandled by the
    #    buffered/streamed split.
    # ------------------------------------------------------------------
    def test_head_request_round_trips_without_body(self):
        def responder(handler):
            handler.send_response(200)
            handler.send_header("Content-Length", "5")
            handler.end_headers()

        upstream = self._start_upstream(responder)
        proxy = self._start_proxy(FIXTURE_PLAN, upstream)

        conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=5)
        conn.request("HEAD", "/v1/models")
        resp = conn.getresponse()
        body = resp.read()
        conn.close()

        self.assertEqual(resp.status, 200)
        self.assertEqual(body, b"")
        self.assertEqual(resp.getheader("Content-Length"), "5")

    def test_204_response_has_no_body(self):
        def responder(handler):
            handler.send_response(204)
            handler.end_headers()

        upstream = self._start_upstream(responder)
        proxy = self._start_proxy(FIXTURE_PLAN, upstream)

        conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=5)
        conn.request("GET", "/v1/models")
        resp = conn.getresponse()
        body = resp.read()
        conn.close()

        self.assertEqual(resp.status, 204)
        self.assertEqual(body, b"")


def headers_dict(resp: http.client.HTTPResponse) -> dict:
    return {name: value for name, value in resp.getheaders()}


def _start_raw_socket_upstream(handle_conn):
    """A one-shot TCP listener on an ephemeral loopback port whose single
    accepted connection is handed, raw, to ``handle_conn`` -- used where a
    test needs to control response TIMING or send a deliberately malformed
    status line/header the stdlib ``http.server``-based fake upstream above
    cannot produce (a bare ``Content-Length: abc``, or a multi-second sleep
    between bytes).
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def serve():
        conn, _ = listener.accept()
        try:
            handle_conn(conn)
        except OSError:
            pass
        finally:
            conn.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return listener, thread


def _read_raw_request_headers(conn: socket.socket) -> bytes:
    request = b""
    while b"\r\n\r\n" not in request:
        chunk = conn.recv(65536)
        if not chunk:
            break
        request += chunk
    return request


class ProxyReadTimeoutTests(unittest.TestCase):
    """The proxy's upstream timeout must bound only the CONNECT, never a
    slow-but-alive read: a non-streamed response whose headers take longer
    than a short connect timeout, or an SSE response with a multi-second gap
    between chunks, must still be relayed in full -- never a 502 and never a
    truncated body. Regression coverage for the HIGH finding where
    ``upstream_timeout`` used to bound socket reads too (see the red-check
    reproduction this test's docstring below references).
    """

    def setUp(self):
        self._listeners = []
        self._servers = []
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for server in self._servers:
            server.shutdown()
            server.server_close()
            server._test_thread.join(timeout=5)
        for listener in self._listeners:
            listener.close()

    def _start_upstream(self, handle_conn):
        listener, _thread = _start_raw_socket_upstream(handle_conn)
        self._listeners.append(listener)
        return listener

    def _start_proxy(self, upstream_port: int, **kwargs):
        server = FASTMLX_PROXY.create_server(
            "127.0.0.1", 0, "127.0.0.1", upstream_port, FIXTURE_PLAN, **kwargs
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        server._test_thread = thread
        self._servers.append(server)
        return server

    # A short connect timeout must never bound a slow-but-alive read: a
    # non-streamed upstream that takes 1.5s to even start writing its
    # response (longer than the 0.5s connect timeout below) must still be
    # relayed in full, not turned into a 502.
    def test_slow_non_streamed_headers_are_not_truncated_by_connect_timeout(self):
        def handle(conn):
            _read_raw_request_headers(conn)
            time.sleep(1.5)
            payload = b'{"ok":true}'
            response = (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: " + str(len(payload)).encode("ascii") + b"\r\n"
                b"\r\n" + payload
            )
            conn.sendall(response)

        listener = self._start_upstream(handle)
        server = self._start_proxy(
            listener.getsockname()[1], upstream_connect_timeout=0.5
        )

        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
        conn.request("GET", "/v1/models")
        resp = conn.getresponse()
        body = resp.read()
        conn.close()

        self.assertEqual(resp.status, 200)
        self.assertEqual(body, b'{"ok":true}')

    # Same idea for a close-delimited/SSE response: a 1.5s gap between two
    # chunks (no Content-Length at all) must not truncate the stream even
    # though it also exceeds the 0.5s connect timeout.
    def test_slow_sse_gap_is_not_truncated_by_connect_timeout(self):
        def handle(conn):
            _read_raw_request_headers(conn)
            conn.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"\r\n"
                b"data: chunk1\n\n"
            )
            time.sleep(1.5)
            conn.sendall(b"data: chunk2\n\n")

        listener = self._start_upstream(handle)
        server = self._start_proxy(
            listener.getsockname()[1], upstream_connect_timeout=0.5
        )

        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
        conn.request("GET", "/v1/stream")
        resp = conn.getresponse()
        body = resp.read()
        conn.close()

        self.assertEqual(resp.status, 200)
        self.assertEqual(body, b"data: chunk1\n\ndata: chunk2\n\n")


class ProxyMalformedContentLengthTests(unittest.TestCase):
    """A non-numeric upstream ``Content-Length`` must be treated as absent
    (the streamed path), never raise inside the request handler.
    """

    def setUp(self):
        self._listeners = []
        self._servers = []
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for server in self._servers:
            server.shutdown()
            server.server_close()
            server._test_thread.join(timeout=5)
        for listener in self._listeners:
            listener.close()

    def test_malformed_content_length_falls_back_to_streamed_relay(self):
        def handle(conn):
            _read_raw_request_headers(conn)
            payload = b'{"ok":true}'
            response = (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: abc\r\n"
                b"\r\n" + payload
            )
            conn.sendall(response)

        listener, _thread = _start_raw_socket_upstream(handle)
        self._listeners.append(listener)

        server = FASTMLX_PROXY.create_server(
            "127.0.0.1", 0, "127.0.0.1", listener.getsockname()[1], FIXTURE_PLAN
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        server._test_thread = thread
        self._servers.append(server)

        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        conn.request("GET", "/v1/models")
        resp = conn.getresponse()
        body = resp.read()
        conn.close()

        self.assertEqual(resp.status, 200)
        self.assertEqual(body, b'{"ok":true}')


class ProxyTruncatedUpstreamTests(unittest.TestCase):
    """M2: a truncated upstream response must never look like a clean,
    complete one to the client -- the buffered path (a promised
    Content-Length the upstream doesn't deliver) must 502 before any
    header is sent; the streamed path (chunked/close-delimited) must
    abort the CLIENT connection with a TCP reset once headers are already
    committed, rather than silently closing clean as if the body ended.
    """

    def setUp(self):
        self._listeners = []
        self._servers = []
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for server in self._servers:
            server.shutdown()
            server.server_close()
            server._test_thread.join(timeout=5)
        for listener in self._listeners:
            listener.close()

    def _start_upstream(self, handle_conn):
        listener, _thread = _start_raw_socket_upstream(handle_conn)
        self._listeners.append(listener)
        return listener

    def _start_proxy(self, upstream_port: int, **kwargs):
        server = FASTMLX_PROXY.create_server(
            "127.0.0.1", 0, "127.0.0.1", upstream_port, FIXTURE_PLAN, **kwargs
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        server._test_thread = thread
        self._servers.append(server)
        return server

    # A REAL Transfer-Encoding: chunked upstream (hand-framed on a raw
    # socket, not the stdlib http.server fake, which cannot produce this
    # framing) relaying two chunks correctly: chunk 1 must arrive at the
    # client before chunk 2 is released, and the joined bytes must be
    # identical to what the upstream sent.
    def test_real_chunked_upstream_relays_two_chunks_in_order(self):
        release_second = threading.Event()

        def handle(conn):
            _read_raw_request_headers(conn)
            conn.sendall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
            chunk1 = b"hello-"
            conn.sendall(("%x\r\n" % len(chunk1)).encode("ascii") + chunk1 + b"\r\n")
            self_wait_ok = release_second.wait(timeout=5)
            if not self_wait_ok:  # pragma: no cover - failure path is asserted below
                return
            chunk2 = b"world!"
            conn.sendall(("%x\r\n" % len(chunk2)).encode("ascii") + chunk2 + b"\r\n")
            conn.sendall(b"0\r\n\r\n")

        listener = self._start_upstream(handle)
        server = self._start_proxy(listener.getsockname()[1])

        received_first = threading.Event()
        result = {}

        def read_client():
            conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
            conn.request("GET", "/v1/stream")
            resp = conn.getresponse()
            first = resp.read(6)
            result["first"] = first
            received_first.set()
            result["rest"] = resp.read()
            conn.close()

        reader = threading.Thread(target=read_client)
        reader.start()
        self.assertTrue(received_first.wait(timeout=5), "client never received chunk 1")
        self.assertFalse(release_second.is_set())
        self.assertEqual(result["first"], b"hello-")
        release_second.set()
        reader.join(timeout=5)
        self.assertEqual(result["first"] + result["rest"], b"hello-world!")

    # The same chunked upstream closes the connection mid-chunk (no
    # terminating 0-length chunk): the client's read must raise -- proof
    # it is NOT treated as a clean, complete read.
    def test_chunked_upstream_closing_mid_chunk_makes_client_read_raise(self):
        def handle(conn):
            _read_raw_request_headers(conn)
            conn.sendall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
            chunk1 = b"partial-chunk-body"
            conn.sendall(("%x\r\n" % len(chunk1)).encode("ascii") + chunk1 + b"\r\n")
            # No terminating chunk: close abruptly, mid-stream.

        listener = self._start_upstream(handle)
        server = self._start_proxy(listener.getsockname()[1])

        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
        conn.request("GET", "/v1/stream")
        resp = conn.getresponse()
        with self.assertRaises(
            (ConnectionResetError, http.client.IncompleteRead, http.client.RemoteDisconnected, OSError)
        ):
            resp.read()
        conn.close()

    # A Content-Length upstream that closes short (buffered path): 502
    # JSON before any header is sent -- never a truncated 200.
    def test_content_length_upstream_closing_short_returns_502(self):
        def handle(conn):
            _read_raw_request_headers(conn)
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n" + b"short")
            # Closes with only 5 of the promised 100 bytes delivered.

        listener = self._start_upstream(handle)
        server = self._start_proxy(listener.getsockname()[1])

        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
        conn.request("GET", "/v1/models")
        resp = conn.getresponse()
        body = resp.read()
        conn.close()

        self.assertEqual(resp.status, 502)
        payload = json.loads(body)
        self.assertEqual(payload["error"]["type"], "upstream_body_truncated")
        self.assertIsNotNone(resp.getheader("X-FastMLX-Admission"))


class ProxyRequestValidationTests(unittest.TestCase):
    """L6: a negative or non-integer request Content-Length must answer
    400 JSON quickly (never attempt to read a negative/garbage byte
    count from the client socket, which would hang), and the handler
    carries a bounded idle timeout.
    """

    def setUp(self):
        self._servers = []
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for server in self._servers:
            server.shutdown()
            server.server_close()
            server._test_thread.join(timeout=5)

    def _start_proxy(self):
        upstream = start_fake_upstream(lambda handler: None)
        self._servers.append(upstream)
        server = FASTMLX_PROXY.create_server(
            "127.0.0.1", 0, "127.0.0.1", upstream.server_address[1], FIXTURE_PLAN
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        server._test_thread = thread
        self._servers.append(server)
        return server

    def _send_and_read(self, port: int, raw_request: bytes) -> bytes:
        sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        sock.sendall(raw_request)
        sock.settimeout(5)
        response = b""
        try:
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
        except socket.timeout:
            pass
        sock.close()
        return response

    def test_negative_content_length_returns_400_quickly(self):
        server = self._start_proxy()
        started = time.monotonic()
        response = self._send_and_read(
            server.server_address[1],
            b"POST /v1/models HTTP/1.1\r\nHost: x\r\nContent-Length: -1\r\n\r\n",
        )
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 5, "negative Content-Length must not hang the handler")
        self.assertIn(b" 400 ", response)
        body_json = json.loads(response.split(b"\r\n\r\n", 1)[1])
        self.assertEqual(body_json["error"]["type"], "invalid_content_length")

    def test_non_integer_content_length_returns_400(self):
        server = self._start_proxy()
        response = self._send_and_read(
            server.server_address[1],
            b"POST /v1/models HTTP/1.1\r\nHost: x\r\nContent-Length: abc\r\n\r\n",
        )
        self.assertIn(b" 400 ", response)
        body_json = json.loads(response.split(b"\r\n\r\n", 1)[1])
        self.assertEqual(body_json["error"]["type"], "invalid_content_length")

    def test_handler_has_a_bounded_idle_timeout(self):
        timeout = FASTMLX_PROXY.ProvenanceProxyHandler.timeout
        self.assertIsNotNone(timeout)
        self.assertGreater(timeout, 0)
        self.assertLessEqual(timeout, 300)


class ProxyRequestBodySizeLimitTests(unittest.TestCase):
    """There is no upper bound on a client-declared ``Content-Length``
    otherwise: the serve host holds tens of GiB of wired model weights, and
    ``--front-host`` accepts any bind address (README documents non-loopback
    as a supported topology), so an unbounded body read is a
    denial-of-service surface. Every test here injects a SMALL
    ``max_request_body_bytes`` via ``create_server(...)`` so the suite never
    actually sends anything close to the real 64 MiB default -- the limit
    itself, not its production magnitude, is what is under test.
    """

    def setUp(self):
        self._servers = []
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for server in self._servers:
            server.shutdown()
            server.server_close()
            server._test_thread.join(timeout=5)

    def _start_upstream_counting_requests(self):
        requests = []

        def responder(handler):
            requests.append(handler.path)
            payload = b'{"ok":true}'
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)

        upstream = start_fake_upstream(responder)
        self._servers.append(upstream)
        return upstream, requests

    def _start_proxy(self, upstream, **kwargs):
        server = start_proxy(FIXTURE_PLAN, "127.0.0.1", upstream.server_address[1], **kwargs)
        self._servers.append(server)
        return server

    def _send_and_read(self, port: int, raw_request: bytes) -> bytes:
        sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        sock.sendall(raw_request)
        sock.settimeout(5)
        response = b""
        try:
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
        except socket.timeout:
            pass
        sock.close()
        return response

    # ------------------------------------------------------------------
    # T1: an over-limit declared Content-Length gets 413, never reaches
    # upstream, and the error names both the declared size and the
    # configured limit.
    # ------------------------------------------------------------------
    def test_over_limit_content_length_returns_413_and_never_reaches_upstream(self):
        upstream, requests = self._start_upstream_counting_requests()
        proxy = self._start_proxy(upstream, max_request_body_bytes=10)
        body = b"x" * 11
        raw_request = (
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
            b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n\r\n" + body
        )
        response = self._send_and_read(proxy.server_address[1], raw_request)

        self.assertIn(b" 413 ", response)
        body_json = json.loads(response.split(b"\r\n\r\n", 1)[1])
        self.assertEqual(body_json["error"]["type"], "request_body_too_large")
        self.assertIn("11", body_json["error"]["message"])
        self.assertIn("10", body_json["error"]["message"])
        self.assertEqual(requests, [], "an over-limit request must never reach the upstream engine")

    # ------------------------------------------------------------------
    # T2 (load-bearing): the refusal is on the DECLARED Content-Length, not
    # on what was actually read. A client that declares a huge body but
    # sends almost none of it must still get 413 promptly -- an
    # implementation that reads (or waits to read) the body before checking
    # would instead hang here until the 120 s client idle timeout, which
    # this test's own 5 s socket timeout and elapsed-time bound would catch
    # as a failure rather than as a slow pass.
    # ------------------------------------------------------------------
    def test_refusal_never_reads_a_declared_body_that_was_never_sent(self):
        upstream, requests = self._start_upstream_counting_requests()
        proxy = self._start_proxy(upstream, max_request_body_bytes=1024)
        # Ten million bytes declared, ZERO of them actually sent -- only the
        # request line and headers cross the wire.
        raw_request = (
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
            b"Content-Length: 10000000\r\n\r\n"
        )
        started = time.monotonic()
        response = self._send_and_read(proxy.server_address[1], raw_request)
        elapsed = time.monotonic() - started

        self.assertLess(
            elapsed, 5,
            "a body that was never sent must never be waited on -- the "
            "refusal is on the DECLARED Content-Length alone",
        )
        self.assertIn(b" 413 ", response)
        body_json = json.loads(response.split(b"\r\n\r\n", 1)[1])
        self.assertEqual(body_json["error"]["type"], "request_body_too_large")
        self.assertEqual(requests, [])

    # ------------------------------------------------------------------
    # T3: a large-but-legitimate body UNDER an injected small limit still
    # relays 200 with every X-FastMLX-* header intact.
    # ------------------------------------------------------------------
    def test_body_under_limit_relays_200_with_provenance_headers(self):
        upstream, requests = self._start_upstream_counting_requests()
        proxy = self._start_proxy(upstream, max_request_body_bytes=4096)
        body = b"y" * 4000

        conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=5)
        conn.request("POST", "/v1/chat/completions", body=body)
        resp = conn.getresponse()
        resp_body = resp.read()
        conn.close()

        self.assertEqual(resp.status, 200)
        self.assertEqual(resp_body, b'{"ok":true}')
        self.assertEqual(requests, ["/v1/chat/completions"])
        headers = headers_dict(resp)
        for name in (
            "X-FastMLX-Admission",
            "X-FastMLX-Card",
            "X-FastMLX-Fit",
            "X-FastMLX-Residency",
            "X-FastMLX-Engine-Build",
            "X-FastMLX-MTP",
            "X-FastMLX-Request-Id",
        ):
            self.assertIsNotNone(headers.get(name), f"missing {name}")

    # ------------------------------------------------------------------
    # T4: the boundary is exact. ``limit`` bytes is allowed; ``limit + 1``
    # is refused.
    # ------------------------------------------------------------------
    def test_boundary_is_exact_limit_allowed_limit_plus_one_refused(self):
        upstream, requests = self._start_upstream_counting_requests()
        proxy = self._start_proxy(upstream, max_request_body_bytes=16)

        conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=5)
        conn.request("POST", "/v1/chat/completions", body=b"z" * 16)
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.assertEqual(resp.status, 200, "exactly `limit` bytes must be allowed")

        # A raw socket, not ``http.client``, for the over-limit half: the
        # proxy closes the connection as soon as it sends the 413, and
        # sending the whole (tiny, 17-byte) request in one ``sendall`` call
        # up front -- the same pattern ``test_over_limit_content_length_...``
        # uses above -- means there is no partial-write/early-close race for
        # ``http.client``'s own two-step (headers, then body) send to lose.
        over_limit_body = b"z" * 17
        raw_request = (
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
            b"Content-Length: 17\r\n\r\n" + over_limit_body
        )
        response = self._send_and_read(proxy.server_address[1], raw_request)
        self.assertIn(b" 413 ", response, "`limit + 1` bytes must be refused")
        body_json = json.loads(response.split(b"\r\n\r\n", 1)[1])
        self.assertEqual(body_json["error"]["type"], "request_body_too_large")

        self.assertEqual(requests, ["/v1/chat/completions"], "only the allowed request reached upstream")


class _HandlerAbort(BaseException):
    """A stand-in for ``SystemExit``/``KeyboardInterrupt`` -- deliberately
    a ``BaseException``, not an ``Exception`` subclass, so a handler that
    raises it is NOT caught by ``socketserver.BaseServer.
    process_request_thread``'s own ``except Exception:`` clause. See
    ``ProxyConcurrencyLimitTests.test_handler_raising_a_base_exception_
    does_not_leak_its_slot`` below for why that distinction is the point.
    """


class ProxyConcurrencyLimitTests(unittest.TestCase):
    """``ProvenanceProxyServer`` is a ``ThreadingHTTPServer`` with
    ``daemon_threads = True`` and, before this cap, NO bound at all on
    concurrent connections: 300 connections produced 300 threads, 1:1, and
    because ``daemon_threads = True`` makes ``socketserver._Threads.append``
    return early, the server did not even keep a reference to them. A probe
    that sends only a partial request and no body still parked a thread for
    the full 120s client idle timeout, at near-zero cost to the client.

    Every test here injects a SMALL ``max_concurrent_requests`` via
    ``create_server(...)`` so the suite never approaches the production
    default of 64 -- the cap itself, not its production magnitude, is what
    is under test. Synchronization is Event-based throughout (an upstream
    responder that blocks on a shared ``gate`` Event the test controls, and
    a per-arrival Event so the test can prove a request reached upstream
    without ever sleeping/polling for it).
    """

    # ------------------------------------------------------------------
    # Slot-release assertions are about WHETHER a slot is released, never
    # about how promptly. Asserted directly against a single sequential
    # request, they are timing-coupled: under CPU contention the releasing
    # path may not have run by the time the follow-up is admitted, and the
    # test fails with a 503 even though nothing leaked. That flaked on a
    # loaded CI runner and blocked a publication, which is worse than a
    # slow test -- a flaky DoS-regression test trains readers to dismiss a
    # real failure.
    #
    # So retry each sequential request against a bounded deadline. This
    # does NOT mask a genuinely leaked slot: a leaked slot is never
    # released, so every retry inside the deadline still gets 503 and the
    # assertion still fails. It only removes the requirement that the
    # release beat a fixed wall clock.
    # ------------------------------------------------------------------
    def assert_sequential_requests_admitted(self, proxy, count, context, deadline=10.0):
        for i in range(count):
            end = time.monotonic() + deadline
            status = None
            attempts = 0
            while True:
                conn = http.client.HTTPConnection(
                    "127.0.0.1", proxy.server_address[1], timeout=5
                )
                conn.request("POST", "/v1/chat/completions", body=b"{}")
                resp = conn.getresponse()
                resp.read()
                conn.close()
                status = resp.status
                attempts += 1
                if status == 200 or time.monotonic() >= end:
                    break
                time.sleep(0.02)
            self.assertEqual(
                status,
                200,
                f"sequential request {i} after {context} must be admitted -- a "
                f"leaked slot would refuse it with 503 instead (still {status} "
                f"after {attempts} attempts over {deadline:.0f}s, so the slot "
                "was never released, not merely released late)",
            )


    def setUp(self):
        self._servers = []
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for server in self._servers:
            server.shutdown()
            server.server_close()
            server._test_thread.join(timeout=5)

    def _start_gated_upstream(self, expected_arrivals: int):
        """An upstream that blocks every request on a single shared
        ``gate`` Event until the test releases it, records every request
        that actually arrived (so a test can prove a refused request never
        reached it), and sets ``arrived[n - 1]`` the moment the n-th
        request arrives -- one Event per expected arrival, so a test can
        wait for "exactly N requests are provably in flight" without any
        sleep-based polling.
        """
        requests = []
        lock = threading.Lock()
        gate = threading.Event()
        arrived = [threading.Event() for _ in range(expected_arrivals)]

        def responder(handler):
            with lock:
                requests.append(handler.path)
                n = len(requests)
            if 0 < n <= expected_arrivals:
                arrived[n - 1].set()
            self.assertTrue(gate.wait(timeout=10), "test never released the gate")
            payload = json.dumps({"n": n}).encode("utf-8")
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)

        upstream = start_fake_upstream(responder)
        self._servers.append(upstream)
        return upstream, requests, gate, arrived

    def _start_proxy(self, upstream, **kwargs):
        server = start_proxy(FIXTURE_PLAN, "127.0.0.1", upstream.server_address[1], **kwargs)
        self._servers.append(server)
        return server

    def _drive_request(self, port: int, results: dict, key) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
        conn.request("POST", "/v1/chat/completions", body=b"{}")
        resp = conn.getresponse()
        payload = resp.read()
        results[key] = (resp.status, payload)
        conn.close()

    def _send_and_read(self, port: int, raw_request: bytes, timeout: float = 5) -> bytes:
        sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        sock.sendall(raw_request)
        sock.settimeout(timeout)
        response = b""
        try:
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
        except socket.timeout:
            pass
        sock.close()
        return response

    # ------------------------------------------------------------------
    # The load-bearing test: written and confirmed failing BEFORE the cap
    # was implemented (against the original code, the 3rd connection got
    # 200, not 503 -- see the implementation report for the exact captured
    # failure).
    # ------------------------------------------------------------------
    def test_concurrency_cap_refuses_the_over_cap_request_while_the_cap_is_held(self):
        upstream, requests, gate, arrived = self._start_gated_upstream(expected_arrivals=2)
        proxy = self._start_proxy(upstream, max_concurrent_requests=2)

        results = {}
        t1 = threading.Thread(target=self._drive_request, args=(proxy.server_address[1], results, 1))
        t2 = threading.Thread(target=self._drive_request, args=(proxy.server_address[1], results, 2))
        t1.start()
        t2.start()
        self.assertTrue(arrived[0].wait(timeout=5), "first request never reached upstream")
        self.assertTrue(arrived[1].wait(timeout=5), "second request never reached upstream")
        # Both in-flight requests are still parked on the gate here -- the
        # cap must be observed as OCCUPIED, not merely "about to free up".
        self.assertFalse(gate.is_set())

        raw_request = b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}"
        response = self._send_and_read(proxy.server_address[1], raw_request)

        self.assertIn(
            b" 503 ", response,
            f"expected a 503 refusal while the cap is held; got: {response!r}",
        )
        header_bytes, _, body_bytes = response.partition(b"\r\n\r\n")
        body_json = json.loads(body_bytes)
        self.assertEqual(body_json["error"]["type"], "too_many_concurrent_requests")
        for name in (
            b"X-FastMLX-Admission", b"X-FastMLX-Card", b"X-FastMLX-Fit", b"X-FastMLX-Residency",
            b"X-FastMLX-Engine-Build", b"X-FastMLX-MTP", b"X-FastMLX-Request-Id",
        ):
            self.assertIn(name, header_bytes, f"missing provenance header {name!r}")

        self.assertEqual(len(requests), 2, "the refused request must never reach upstream")

        gate.set()
        t1.join(timeout=5)
        t2.join(timeout=5)
        self.assertEqual(results[1][0], 200)
        self.assertEqual(results[2][0], 200)
        self.assertEqual(json.loads(results[1][1])["n"] in (1, 2), True)
        self.assertEqual(json.loads(results[2][1])["n"] in (1, 2), True)

    # ------------------------------------------------------------------
    # AC1 (anti-vacuity): the peak in-flight counter must actually REACH
    # the cap, not merely never exceed it -- ``== N``, not only ``<= N``.
    # ------------------------------------------------------------------
    def test_peak_inflight_reaches_and_never_exceeds_the_cap(self):
        upstream, requests, gate, arrived = self._start_gated_upstream(expected_arrivals=3)
        proxy = self._start_proxy(upstream, max_concurrent_requests=3)

        results = {}
        threads = [
            threading.Thread(target=self._drive_request, args=(proxy.server_address[1], results, i))
            for i in range(3)
        ]
        for t in threads:
            t.start()
        for ev in arrived:
            self.assertTrue(ev.wait(timeout=5))

        self.assertLessEqual(proxy.peak_inflight_requests, 3)
        self.assertEqual(proxy.peak_inflight_requests, 3, "the cap was never actually reached")

        gate.set()
        for t in threads:
            t.join(timeout=5)
        for i in range(3):
            self.assertEqual(results[i][0], 200)

    # ------------------------------------------------------------------
    # AC3: the cap is enforced on the DECLARED Content-Length alone, same
    # as the body-size limit -- an over-cap client that declares a huge
    # body and sends none of it must still be refused promptly. Kills a
    # mutation that moves the gate into ``_dispatch`` after the body read.
    # ------------------------------------------------------------------
    def test_over_cap_refusal_never_reads_a_declared_body_that_was_never_sent(self):
        upstream, requests, gate, arrived = self._start_gated_upstream(expected_arrivals=2)
        proxy = self._start_proxy(upstream, max_concurrent_requests=2)

        results = {}
        t1 = threading.Thread(target=self._drive_request, args=(proxy.server_address[1], results, 1))
        t2 = threading.Thread(target=self._drive_request, args=(proxy.server_address[1], results, 2))
        t1.start()
        t2.start()
        self.assertTrue(arrived[0].wait(timeout=5))
        self.assertTrue(arrived[1].wait(timeout=5))

        raw_request = b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 10000000\r\n\r\n"
        started = time.monotonic()
        response = self._send_and_read(proxy.server_address[1], raw_request)
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 5, "the cap must be enforced before any body byte is read")
        self.assertIn(b" 503 ", response)
        body_json = json.loads(response.split(b"\r\n\r\n", 1)[1])
        self.assertEqual(body_json["error"]["type"], "too_many_concurrent_requests")
        self.assertEqual(len(requests), 2)

        gate.set()
        t1.join(timeout=5)
        t2.join(timeout=5)

    # ------------------------------------------------------------------
    # AC4: slots actually RELEASE -- after the two held requests complete,
    # ``cap + 1`` further SEQUENTIAL requests must all be admitted. Kills a
    # missing/short ``release()`` on either handler-thread exit path.
    # ------------------------------------------------------------------
    def test_slots_release_after_completion_and_admit_cap_plus_one_more_sequential_requests(self):
        upstream, requests, gate, arrived = self._start_gated_upstream(expected_arrivals=2)
        proxy = self._start_proxy(upstream, max_concurrent_requests=2)

        results = {}
        t1 = threading.Thread(target=self._drive_request, args=(proxy.server_address[1], results, 1))
        t2 = threading.Thread(target=self._drive_request, args=(proxy.server_address[1], results, 2))
        t1.start()
        t2.start()
        self.assertTrue(arrived[0].wait(timeout=5))
        self.assertTrue(arrived[1].wait(timeout=5))
        gate.set()
        t1.join(timeout=5)
        t2.join(timeout=5)
        self.assertEqual(results[1][0], 200)
        self.assertEqual(results[2][0], 200)

        self.assert_sequential_requests_admitted(proxy, 3, "release")  # cap (2) + 1

    # ------------------------------------------------------------------
    # Slot-unwind path 1: ``ThreadingMixIn.process_request`` itself can
    # raise (``RuntimeError: can't start new thread``, under real
    # thread-count/resource pressure) starting the handler thread, BEFORE
    # ``process_request_thread`` ever runs -- so ITS ``finally`` never
    # fires, and only ``process_request``'s own ``except BaseException``
    # block (see its in-code comment) can release the slot and decrement
    # the counter. A leaked slot here counts against the cap FOREVER.
    # Discriminating quantities: the three 200s below (a leaked semaphore
    # permit would turn one of them into a 503) and
    # ``inflight_requests == 0`` (a missing counter decrement would leave
    # it at 2) -- not merely "no exception escaped the test".
    # ------------------------------------------------------------------
    def test_failed_handler_thread_start_does_not_leak_its_slot(self):
        def responder(handler):
            payload = b"{}"
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)

        upstream = start_fake_upstream(responder)
        self._servers.append(upstream)
        proxy = self._start_proxy(upstream, max_concurrent_requests=2)

        handled_errors = []
        # One Event per induced failure, set from INSIDE ``handle_error``
        # (not from the raising function itself): ``handle_error`` runs
        # strictly AFTER ``process_request``'s own except-block has
        # already released the slot and decremented the counter for that
        # connection (see the source's own comment on that except block),
        # so waiting on these events -- rather than on the raise itself --
        # is what makes the later assertions race-free instead of merely
        # usually-true.
        handled_events = [threading.Event(), threading.Event()]

        def record_handle_error(request, client_address):
            # Replaces the stdlib default (a traceback printed to
            # stderr) with a recorded exception TYPE, so the test can
            # assert exactly what reached ``_handle_request_noblock``'s
            # own error handling instead of merely "did not crash".
            idx = len(handled_errors)
            handled_errors.append(sys.exc_info()[0])
            if idx < len(handled_events):
                handled_events[idx].set()

        proxy.handle_error = record_handle_error

        def flaky_process_request(self_server, request, client_address):
            raise RuntimeError("can't start new thread")

        with unittest.mock.patch.object(
            socketserver.ThreadingMixIn, "process_request", new=flaky_process_request
        ):
            conn1 = socket.create_connection(("127.0.0.1", proxy.server_address[1]), timeout=5)
            self.assertTrue(
                handled_events[0].wait(timeout=5),
                "the first induced thread-start failure never reached handle_error",
            )
            conn1.close()

            conn2 = socket.create_connection(("127.0.0.1", proxy.server_address[1]), timeout=5)
            self.assertTrue(
                handled_events[1].wait(timeout=5),
                "the second induced thread-start failure never reached handle_error",
            )
            conn2.close()

        self.assertEqual(
            len(handled_errors), 2,
            f"expected exactly 2 errors reaching handle_error, got {len(handled_errors)}",
        )
        for exc_type in handled_errors:
            self.assertIs(exc_type, RuntimeError, f"expected RuntimeError, got {exc_type}")

        self.assertEqual(
            proxy.inflight_requests, 0,
            "inflight counter leaked across 2 induced thread-start failures: "
            f"{proxy.inflight_requests} != 0",
        )

        self.assert_sequential_requests_admitted(
            proxy, 3, "2 induced thread-start failures"
        )  # cap (2) + 1

    # ------------------------------------------------------------------
    # Slot-unwind path 2: ``socketserver.BaseServer.process_request_thread``
    # (reached via ``ThreadingMixIn.process_request_thread``'s own
    # ``super()`` call) only catches ``Exception``, never
    # ``BaseException`` -- so for a handler that raises something like
    # ``SystemExit``/``KeyboardInterrupt``, the ONLY thing that releases
    # this proxy's slot is this override's own ``finally``. Kills a
    # version of the fix that caught only ``Exception`` (would leak on
    # any ``BaseException``) or that moved the release after the
    # ``super()`` call without a ``finally`` at all.
    # ------------------------------------------------------------------
    def test_handler_raising_a_base_exception_does_not_leak_its_slot(self):
        def responder(handler):
            payload = b"{}"
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)

        upstream = start_fake_upstream(responder)
        self._servers.append(upstream)
        proxy = self._start_proxy(upstream, max_concurrent_requests=1)

        dispatch_entered = threading.Event()

        def aborting_do_post(self_handler):
            dispatch_entered.set()
            raise _HandlerAbort("induced base-exception abort")

        recorded_exc_types = []
        abort_seen = threading.Event()
        original_excepthook = threading.excepthook

        def record_excepthook(args):
            recorded_exc_types.append(args.exc_type)
            abort_seen.set()

        threading.excepthook = record_excepthook
        self.addCleanup(setattr, threading, "excepthook", original_excepthook)

        raw_request = b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}"
        with unittest.mock.patch.object(
            FASTMLX_PROXY.ProvenanceProxyHandler, "do_POST", new=aborting_do_post
        ):
            self._send_and_read(proxy.server_address[1], raw_request, timeout=5)
            self.assertTrue(dispatch_entered.wait(timeout=5), "the aborting handler was never entered")

        # ``threading.excepthook`` fires only AFTER this override's own
        # ``finally`` (slot release, counter decrement) has already run
        # -- the exception must propagate all the way out of
        # ``process_request_thread`` before ``Thread._bootstrap_inner``'s
        # bare ``except:`` invokes it -- so this wait is what makes the
        # counter assertion below race-free.
        self.assertTrue(
            abort_seen.wait(timeout=5),
            "threading.excepthook was never invoked for the induced BaseException",
        )
        self.assertEqual(
            len(recorded_exc_types), 1,
            f"expected exactly 1 thread exception, got {len(recorded_exc_types)}",
        )
        self.assertIs(
            recorded_exc_types[0], _HandlerAbort,
            f"expected _HandlerAbort, got {recorded_exc_types[0]}",
        )

        self.assertEqual(
            proxy.inflight_requests, 0,
            "inflight counter leaked after a BaseException from the handler: "
            f"{proxy.inflight_requests} != 0",
        )

        self.assert_sequential_requests_admitted(
            proxy, 2, "a BaseException from the handler"
        )  # cap (1) + 1

    # ------------------------------------------------------------------
    # Slot-unwind path 3: the handler raises an ORDINARY ``Exception``
    # (``ValueError``) -- the shape ``socketserver.BaseServer.
    # process_request_thread`` already catches internally, swallowing it
    # and returning NORMALLY, before this override's own ``finally`` ever
    # sees anything to unwind. NOT ``finally``-discriminating: because the
    # stdlib itself already caught the exception, a release placed AFTER
    # the ``super()`` call with no ``try/finally`` at all would ALSO run
    # here, so this test alone cannot distinguish that from the correct
    # ``finally``-based release the code actually uses (contrast with
    # ``test_handler_raising_a_base_exception_does_not_leak_its_slot``
    # above, which CAN make that distinction). Its value is guarding the
    # documented behaviour -- an ordinary handler exception must not leak
    # a slot -- and the inflight counter, not the ``finally`` mechanism.
    # ------------------------------------------------------------------
    def test_handler_raising_an_ordinary_exception_releases_its_slot(self):
        def responder(handler):
            payload = b"{}"
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)

        upstream = start_fake_upstream(responder)
        self._servers.append(upstream)
        proxy = self._start_proxy(upstream, max_concurrent_requests=1)
        proxy.handle_error = lambda request, client_address: None  # expected; keep the log quiet

        dispatch_entered = threading.Event()

        def failing_do_post(self_handler):
            dispatch_entered.set()
            raise ValueError("induced ordinary exception")

        # ``socketserver.BaseServer.process_request_thread``'s own
        # ``shutdown_request`` (which closes the client's socket and ends
        # ``_send_and_read`` below) runs INSIDE ``super().
        # process_request_thread()``, strictly before this override's own
        # ``finally`` decrements the counter -- so waiting only for the
        # client socket to close would leave a real, if short, race
        # against the counter update on a loaded machine. This thin
        # wrapper around the REAL (possibly mutated) implementation adds
        # only a completion signal; it changes nothing about slot/counter
        # handling itself.
        real_process_request_thread = FASTMLX_PROXY.ProvenanceProxyServer.process_request_thread
        thread_finished = threading.Event()

        def observed_process_request_thread(self_server, request, client_address):
            try:
                return real_process_request_thread(self_server, request, client_address)
            finally:
                thread_finished.set()

        raw_request = b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}"
        with unittest.mock.patch.object(
            FASTMLX_PROXY.ProvenanceProxyHandler, "do_POST", new=failing_do_post
        ), unittest.mock.patch.object(
            FASTMLX_PROXY.ProvenanceProxyServer, "process_request_thread", new=observed_process_request_thread
        ):
            # Whatever the client actually observes (a closed connection,
            # a partial response, or nothing) is not asserted here -- the
            # point under test is the slot, not the client-visible
            # status; see the docstring above for why this test cannot be
            # ``finally``-discriminating.
            self._send_and_read(proxy.server_address[1], raw_request, timeout=5)
            self.assertTrue(dispatch_entered.wait(timeout=5), "the failing handler was never entered")
            self.assertTrue(thread_finished.wait(timeout=5), "the handler thread never finished")

        self.assertEqual(
            proxy.inflight_requests, 0,
            "inflight counter leaked after an ordinary Exception from the handler: "
            f"{proxy.inflight_requests} != 0",
        )

        self.assert_sequential_requests_admitted(
            proxy, 2, "an ordinary Exception from the handler"
        )  # cap (1) + 1

    # ------------------------------------------------------------------
    # Documented behaviour, previously uncovered: a streaming (SSE)
    # response holds its slot for the WHOLE generation, not just until
    # headers are sent -- the cap counts in-flight streams, not completed
    # requests. Anti-vacuity control: the mid-stream 503's error type is
    # asserted exactly, so a 503 for any unrelated reason cannot pass.
    # ------------------------------------------------------------------
    def test_streaming_sse_response_holds_its_slot_for_the_whole_generation(self):
        release_rest = threading.Event()

        def responder(handler):
            # The verification requests below are ordinary POSTs with a
            # small declared body -- drained here like every other
            # responder in this file does (``_read_request_body``), NOT
            # because this SSE responder cares about the body, but
            # because leaving it unread in the upstream's own kernel
            # receive buffer at connection-close time makes the OS send a
            # TCP RST instead of a clean FIN (the same RST-vs-FIN
            # mechanism ``_refuse_over_capacity`` documents on the
            # proxy's OWN client-facing socket) -- which the proxy's
            # ``_respond_streamed`` correctly treats as an upstream
            # failure and aborts the client connection for, an entirely
            # different (and non-discriminating, for THIS test) failure
            # mode from the one under test here.
            _read_request_body(handler)
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.end_headers()
            handler.wfile.write(b"data: chunk1\n\n")
            handler.wfile.flush()
            self.assertTrue(release_rest.wait(timeout=10), "test never released the gate")
            handler.wfile.write(b"data: chunk2\n\n")
            handler.wfile.flush()

        upstream = start_fake_upstream(responder)
        self._servers.append(upstream)
        proxy = self._start_proxy(upstream, max_concurrent_requests=1)

        received_first = threading.Event()
        stream_result = {}

        def read_stream():
            conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=10)
            conn.request("GET", "/v1/stream")
            resp = conn.getresponse()
            first = resp.read1(4096) if hasattr(resp, "read1") else resp.read(1)
            stream_result["first"] = first
            received_first.set()
            rest = resp.read()
            stream_result["rest"] = rest
            conn.close()

        reader = threading.Thread(target=read_stream)
        reader.start()
        self.assertTrue(received_first.wait(timeout=5), "client never received the first SSE chunk")

        # The sole slot is still held by the in-progress stream here -- a
        # second connection while it is mid-generation must be refused,
        # not merely eventually admitted.
        raw_request = b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}"
        response = self._send_and_read(proxy.server_address[1], raw_request)
        self.assertIn(
            b" 503 ", response,
            f"expected a 503 while the streaming response holds the sole slot; got: {response!r}",
        )
        body_json = json.loads(response.split(b"\r\n\r\n", 1)[1])
        self.assertEqual(body_json["error"]["type"], "too_many_concurrent_requests")

        release_rest.set()
        reader.join(timeout=5)
        self.assertEqual(stream_result["first"] + stream_result["rest"], b"data: chunk1\n\ndata: chunk2\n\n")

        # The slot must have been released once the stream ended, not
        # leaked for the rest of the process's life.
        conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=5)
        conn.request("POST", "/v1/chat/completions", body=b"{}")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.assertEqual(
            resp.status, 200,
            "the slot must be released once the stream finished, not leaked",
        )

    # ------------------------------------------------------------------
    # Shared helper for the two blocked-write-budget tests below: a client
    # with a tiny ``SO_RCVBUF`` (set BEFORE ``connect`` -- the kernel must
    # honor it for the connection's whole lifetime, not just from whenever
    # a test happens to set it) that sends a complete request and then
    # never reads a single byte of the response. Forces the proxy's own
    # outbound writes to eventually block once its send buffer and the
    # client's shrunk receive window both fill, without needing a real
    # slow network to reproduce the defect.
    # ------------------------------------------------------------------
    def _open_stalled_client(self, port: int, raw_request: bytes) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2048)
        sock.connect(("127.0.0.1", port))
        sock.sendall(raw_request)
        return sock

    # ------------------------------------------------------------------
    # This test covers the BOUNDED half of the defect, and says so
    # precisely because an earlier draft of this comment did not. A client
    # that never reads at all does NOT hold its slot forever: measured
    # against the pre-fix code, it was released after 123.6s -- the
    # ``_CLIENT_IDLE_TIMEOUT_SECONDS`` backstop doing its job. What this
    # test pins is that the budget replaces that 123.6s with the budget,
    # which is worth pinning on its own (two minutes of a scarce slot for
    # zero attacker effort) but is NOT the unbounded case.
    #
    # The unbounded case needs a client that lets each write COMPLETE and
    # then stalls, which resets the backstop every time; that is
    # ``test_a_client_that_drains_one_chunk_then_stalls_...`` below.
    # ``_CLIENT_RESPONSE_MAX_BLOCKED_SECONDS`` is patched down to 2.0s
    # here so this test completes in seconds.
    # ------------------------------------------------------------------
    def test_a_client_that_opens_a_stream_and_stops_reading_releases_its_concurrency_slot(self):
        with unittest.mock.patch.object(
            FASTMLX_PROXY, "_CLIENT_RESPONSE_MAX_BLOCKED_SECONDS", 2.0, create=True
        ):

            def responder(handler):
                if handler.path != "/v1/stream":
                    payload = b'{"ok":true}'
                    handler.send_response(200)
                    handler.send_header("Content-Type", "application/json")
                    handler.send_header("Content-Length", str(len(payload)))
                    handler.end_headers()
                    handler.wfile.write(payload)
                    return
                handler.send_response(200)
                handler.send_header("Content-Type", "text/event-stream")
                handler.end_headers()
                chunk = b"data: " + b"x" * 65000 + b"\n\n"
                try:
                    for _ in range(500):
                        handler.wfile.write(chunk)
                        handler.wfile.flush()
                except OSError:
                    pass  # the proxy closed its upstream connection once it aborted the client.

            upstream = start_fake_upstream(responder)
            self._servers.append(upstream)

            logged = []
            aborted = threading.Event()

            def log_hook(entry):
                logged.append(entry)
                if entry.get("error") and "blocked-write" in entry["error"]:
                    aborted.set()

            proxy = self._start_proxy(upstream, max_concurrent_requests=1, log_hook=log_hook)

            stuck = self._open_stalled_client(
                proxy.server_address[1], b"GET /v1/stream HTTP/1.1\r\nHost: x\r\n\r\n"
            )
            self.addCleanup(stuck.close)

            self.assertIsNotNone(
                self._wait_for_inflight(proxy, 1, timeout=5),
                "the stalled stream was never even admitted",
            )

            self.assertTrue(
                aborted.wait(timeout=10),
                "the stalled stream's slot was never reclaimed on the blocked-write budget",
            )
            self.assertIsNotNone(
                self._wait_for_inflight(proxy, 0, timeout=5),
                "the slot was not released once the blocked-write budget was hit",
            )

            abort_entries = [e for e in logged if e.get("error") and "blocked-write" in e["error"]]
            self.assertEqual(len(abort_entries), 1)
            self.assertIn("2.0", abort_entries[0]["error"])

            # The slot must be genuinely usable again, not merely freed.
            conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=5)
            conn.request("GET", "/v1/models")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            self.assertEqual(
                resp.status, 200,
                "a second client must be able to use the reclaimed slot",
            )

    # ------------------------------------------------------------------
    # A mutation run showed ``_respond_streamed``'s ``budget_remaining
    # <= 0`` pre-check is NOT independently covered: deleting it alone
    # left all three budget tests green, because the per-write timeout
    # caught every case they exercise. That made it look redundant. It is
    # not -- without it, ``settimeout`` would be handed a NEGATIVE value,
    # which raises ``ValueError``, which nothing in the write path
    # catches, so the handler thread would die with an unhandled
    # exception instead of aborting the client cleanly.
    #
    # Rather than leave a guard whose removal is invisible to the suite,
    # ``_write_response_chunk`` now clamps to a positive floor, and this
    # test pins THAT directly -- so the crash class is covered by an
    # assertion, not by the accident of another check running first.
    # ------------------------------------------------------------------
    def test_a_spent_write_budget_never_reaches_settimeout_as_a_negative(self):
        real_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addCleanup(real_socket.close)

        handler = FASTMLX_PROXY.ProvenanceProxyHandler.__new__(
            FASTMLX_PROXY.ProvenanceProxyHandler
        )
        handler.connection = real_socket
        handler.wfile = io.BytesIO()

        # A real socket, so the range being asserted is the OS's own and
        # not a mock's willingness to accept anything.
        with self.assertRaises(ValueError):
            real_socket.settimeout(-0.5)

        # Asserting on ``gettimeout()`` AFTER the call would prove
        # nothing: ``_write_response_chunk``'s ``finally`` restores the
        # backstop, so any post-call read is 120.0 whether the clamp
        # exists or not. The value actually handed to ``settimeout``
        # DURING the write is the only discriminating observation, so it
        # is recorded as it happens.
        applied = []

        class RecordingConnection:
            """Records every ``settimeout`` value and forwards it to a
            REAL socket, so an out-of-range value still raises the OS's
            own ``ValueError`` here rather than being quietly accepted by
            a mock (a socket object's attributes are read-only, so this
            cannot be done by patching one in place)."""

            def __init__(self, sock):
                self._sock = sock

            def settimeout(self, value):
                applied.append(value)
                return self._sock.settimeout(value)

        handler.connection = RecordingConnection(real_socket)

        for spent in (-5.0, -0.001, 0.0):
            with self.subTest(budget_remaining=spent):
                applied.clear()
                handler._write_response_chunk(b"x", spent)
                self.assertTrue(applied, "settimeout was never called at all")
                self.assertGreater(
                    applied[0], 0.0,
                    f"a spent budget ({spent}) reached settimeout as {applied[0]!r} -- "
                    "negative raises ValueError, and 0.0 silently switches the socket "
                    "to non-blocking",
                )

    # ------------------------------------------------------------------
    # THE LOAD-BEARING TEST: the genuinely UNBOUNDED shape.
    #
    # Established by measurement against the pre-fix code, after the
    # obvious guess was refuted. Dribbling one byte at a time does NOT
    # hold the slot (released after 124.3s, i.e. just the backstop),
    # because ``sendall`` does not re-arm its timeout per internal send --
    # the CURRENT write simply never returns. To reset the backstop the
    # client must let one whole write COMPLETE, then stall again. That
    # shape held its slot for 400.9s and was still holding when the
    # window closed, for about 3.3 KiB/s of attacker effort.
    #
    # Reproduced here at unit speed by shrinking BOTH clocks: the backstop
    # down to 3.0s (so a 1.2s stall sits comfortably under it and keeps
    # resetting it, exactly as a 119s stall does under the real 120s) and
    # the budget down to 2.0s. Against the unfixed code this test hangs
    # until its own assertion deadline; the budget is the only thing that
    # ends it.
    # ------------------------------------------------------------------
    def test_a_client_that_drains_one_chunk_then_stalls_cannot_hold_its_slot_forever(self):
        with contextlib.ExitStack() as patches:
            patches.enter_context(unittest.mock.patch.object(
                FASTMLX_PROXY, "_CLIENT_RESPONSE_MAX_BLOCKED_SECONDS", 2.0, create=True))
            # Both spellings matter: the module global is what
            # ``_write_response_chunk`` restores after each write, while
            # the handler's ``timeout`` class attribute was bound from it
            # at import time and is what ``setup()`` applies to the
            # socket. Patching only one leaves the other at 120s and the
            # test would pass for the wrong reason.
            patches.enter_context(unittest.mock.patch.object(
                FASTMLX_PROXY, "_CLIENT_IDLE_TIMEOUT_SECONDS", 3.0))
            patches.enter_context(unittest.mock.patch.object(
                FASTMLX_PROXY.ProvenanceProxyHandler, "timeout", 3.0))

            def responder(handler):
                if handler.path != "/v1/stream":
                    payload = b'{"ok":true}'
                    handler.send_response(200)
                    handler.send_header("Content-Type", "application/json")
                    handler.send_header("Content-Length", str(len(payload)))
                    handler.end_headers()
                    handler.wfile.write(payload)
                    return
                handler.send_response(200)
                handler.send_header("Content-Type", "text/event-stream")
                handler.end_headers()
                chunk = b"data: " + b"x" * 65000 + b"\n\n"
                try:
                    for _ in range(500):
                        handler.wfile.write(chunk)
                        handler.wfile.flush()
                except OSError:
                    pass

            upstream = start_fake_upstream(responder)
            self._servers.append(upstream)

            aborted = threading.Event()

            def log_hook(entry):
                if entry.get("error") and "blocked-write" in entry["error"]:
                    aborted.set()

            proxy = self._start_proxy(upstream, max_concurrent_requests=1, log_hook=log_hook)

            stuck = self._open_stalled_client(
                proxy.server_address[1], b"GET /v1/stream HTTP/1.1\r\nHost: x\r\n\r\n"
            )
            self.addCleanup(stuck.close)

            stop = threading.Event()
            self.addCleanup(stop.set)
            drained = [0]

            def drain_a_chunk_then_stall():
                # Each pass drains a whole write's worth, which lets the
                # proxy's current ``sendall`` RETURN -- that return is what
                # resets the backstop. Then it stalls under the backstop.
                stuck.settimeout(1.0)
                while not stop.is_set():
                    got = 0
                    while got < 262144 and not stop.is_set():
                        try:
                            data = stuck.recv(65536)
                        except (socket.timeout, OSError):
                            break
                        if not data:
                            return
                        got += len(data)
                        drained[0] += len(data)
                    stop.wait(1.2)

            reader = threading.Thread(target=drain_a_chunk_then_stall, daemon=True)
            reader.start()

            self.assertIsNotNone(
                self._wait_for_inflight(proxy, 1, timeout=5),
                "the chunk-then-stall stream was never even admitted",
            )
            self.assertTrue(
                aborted.wait(timeout=20),
                "a client that drains one chunk and then stalls held its slot past the "
                "cumulative blocked-write budget -- this is the unbounded shape, and the "
                "backstop cannot end it because every completed write resets it",
            )
            self.assertIsNotNone(
                self._wait_for_inflight(proxy, 0, timeout=5),
                "the slot was not released once the blocked-write budget was hit",
            )
            self.assertGreater(
                drained[0], 0,
                "the client never drained anything, so this ran as the SILENT shape "
                "(which the backstop alone would have ended) and proves nothing here",
            )

    # ------------------------------------------------------------------
    # Same defect, ``_respond_buffered``'s path: a Content-Length upstream
    # response routes ``_dispatch`` there instead, and its single
    # ``self.wfile.write(body)`` call had the exact same unbounded
    # exposure.
    # ------------------------------------------------------------------
    def test_a_slow_reading_client_on_the_buffered_path_releases_its_concurrency_slot(self):
        with unittest.mock.patch.object(
            FASTMLX_PROXY, "_CLIENT_RESPONSE_MAX_BLOCKED_SECONDS", 2.0, create=True
        ):
            large_body = b"y" * (32 * 1024 * 1024)

            def responder(handler):
                if handler.path != "/v1/big":
                    payload = b'{"ok":true}'
                    handler.send_response(200)
                    handler.send_header("Content-Type", "application/json")
                    handler.send_header("Content-Length", str(len(payload)))
                    handler.end_headers()
                    handler.wfile.write(payload)
                    return
                handler.send_response(200)
                handler.send_header("Content-Type", "application/octet-stream")
                handler.send_header("Content-Length", str(len(large_body)))
                handler.end_headers()
                try:
                    handler.wfile.write(large_body)
                except OSError:
                    pass  # the proxy closed its upstream connection once it aborted the client.

            upstream = start_fake_upstream(responder)
            self._servers.append(upstream)

            logged = []
            aborted = threading.Event()

            def log_hook(entry):
                logged.append(entry)
                if entry.get("error") and "blocked-write" in entry["error"]:
                    aborted.set()

            proxy = self._start_proxy(upstream, max_concurrent_requests=1, log_hook=log_hook)

            stuck = self._open_stalled_client(
                proxy.server_address[1], b"GET /v1/big HTTP/1.1\r\nHost: x\r\n\r\n"
            )
            self.addCleanup(stuck.close)

            self.assertIsNotNone(
                self._wait_for_inflight(proxy, 1, timeout=10),
                "the stalled buffered request was never even admitted",
            )

            self.assertTrue(
                aborted.wait(timeout=15),
                "the stalled buffered response's slot was never reclaimed on the "
                "blocked-write budget",
            )
            self.assertIsNotNone(
                self._wait_for_inflight(proxy, 0, timeout=5),
                "the slot was not released once the blocked-write budget was hit",
            )

            abort_entries = [e for e in logged if e.get("error") and "blocked-write" in e["error"]]
            self.assertEqual(len(abort_entries), 1)
            self.assertIn("2.0", abort_entries[0]["error"])
            self.assertFalse(abort_entries[0]["streamed"], "this is the buffered path, not SSE")

            # The slot must be genuinely usable again, not merely freed.
            conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=5)
            conn.request("GET", "/v1/models")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            self.assertEqual(
                resp.status, 200,
                "a second client must be able to use the reclaimed slot",
            )

    # ------------------------------------------------------------------
    # The defect this pins: ``_respond_buffered`` set ``bytes_out =
    # len(body)`` BEFORE attempting the write, and its (BrokenPipeError,
    # ConnectionResetError, OSError) clause used to just ``pass`` -- so a
    # client that vanishes mid-write got logged as a clean delivery of the
    # WHOLE body, with ``error`` left ``None``. This drives a real client
    # that disconnects instead of reading a large buffered (Content-Length)
    # response, and asserts on the LOGGED entry (the surface the defect
    # corrupts), not on anything client-side -- the client is gone before
    # it could observe a response either way.
    # ------------------------------------------------------------------
    def test_a_client_that_disconnects_mid_write_on_the_buffered_path_is_not_logged_as_delivered(self):
        large_body = b"z" * (32 * 1024 * 1024)

        def responder(handler):
            handler.send_response(200)
            handler.send_header("Content-Type", "application/octet-stream")
            handler.send_header("Content-Length", str(len(large_body)))
            handler.end_headers()
            try:
                handler.wfile.write(large_body)
            except OSError:
                pass  # the proxy closed its upstream connection once the client vanished.

        upstream = start_fake_upstream(responder)
        self._servers.append(upstream)

        logged = []
        logged_event = threading.Event()

        def log_hook(entry):
            if entry.get("event") == "saturation_snapshot":
                return  # a separate, unrelated feature's line; see its own tests
            logged.append(entry)
            logged_event.set()

        proxy = self._start_proxy(upstream, max_concurrent_requests=1, log_hook=log_hook)

        # A shrunk receive window, exactly like ``_open_stalled_client``
        # above, so the proxy's write does not complete in one shot before
        # the close below has a chance to matter -- then the socket is
        # closed outright (not merely left stalled) so the proxy's write
        # fails with a vanished-client ``OSError``, not a budget timeout.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2048)
        sock.connect(("127.0.0.1", proxy.server_address[1]))
        sock.sendall(b"GET /v1/big HTTP/1.1\r\nHost: x\r\n\r\n")
        sock.close()

        self.assertTrue(
            logged_event.wait(timeout=15), "the disconnected request was never logged at all"
        )
        entry = logged[0]
        self.assertIsNotNone(
            entry.get("error"),
            "a client that vanished mid-write must not be logged with error=None",
        )
        # Pin WHICH branch produced the error. A response-write-budget
        # timeout also sets an error and also zeroes bytes_out, so asserting
        # only "some error, fewer bytes" would be satisfied by the timeout
        # path and this test would silently stop covering the vanished-client
        # branch it exists for.
        self.assertTrue(
            entry["error"].startswith("client_write_failed:"),
            "expected the vanished-client branch, got: " + repr(entry["error"]),
        )
        self.assertEqual(
            entry["bytes_out"], 0,
            "a write that did not complete contributes 0, matching the streamed path's "
            "count-only-after-success convention",
        )
        self.assertFalse(
            entry["streamed"], "this must be the buffered (Content-Length) path, not SSE, "
            "for the fix under test to be exercised at all"
        )

    # ------------------------------------------------------------------
    # AC5: fd hygiene -- 50 sequential refusals, while one slot stays held,
    # must not grow the process's open-fd count at all. Kills a missing
    # ``shutdown_request`` on the refusal path: a status-only test is blind
    # to this, and a leaked fd per refusal is a strictly worse DoS than the
    # one this whole cap exists to fix.
    # ------------------------------------------------------------------
    def test_fifty_sequential_refusals_do_not_leak_file_descriptors(self):
        if not os.path.isdir("/dev/fd"):
            self.skipTest("/dev/fd not available on this platform")

        upstream, requests, gate, arrived = self._start_gated_upstream(expected_arrivals=1)
        proxy = self._start_proxy(upstream, max_concurrent_requests=1)

        holder_results = {}
        holder = threading.Thread(target=self._drive_request, args=(proxy.server_address[1], holder_results, "holder"))
        holder.start()
        self.assertTrue(arrived[0].wait(timeout=5), "the sole slot was never occupied")

        raw_request = b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n"
        before = len(os.listdir("/dev/fd"))
        for _ in range(50):
            response = self._send_and_read(proxy.server_address[1], raw_request)
            self.assertIn(b" 503 ", response)
        after = len(os.listdir("/dev/fd"))
        self.assertEqual(after, before, "each refusal must close its own request socket")

        gate.set()
        holder.join(timeout=5)
        self.assertEqual(holder_results["holder"][0], 200)

    # ------------------------------------------------------------------
    # Pins the EXPLICIT close on the refusal path, which the fd count above
    # provably cannot: deleting ``shutdown_request`` from the refusal branch
    # was measured NOT to leak an fd, because CPython closes the request
    # socket by refcounting the moment the last reference to it drops (the
    # locals in ``process_request`` and ``socketserver``'s own
    # ``_handle_request_noblock``, neither of which outlives a refusal).
    # That makes the fd assertion above non-discriminating for that one
    # deletion -- verified, not assumed: a control that leaks one unrelated
    # fd per refusal moves it 11 -> 61, so the counter itself does fire.
    # The explicit call still has to be pinned, because the socket closing
    # at all would then depend on a refcounting implementation detail: any
    # future change that retains a reference (a log buffer, a pool, a
    # traceback) silently turns this back into the fd leak the counter was
    # meant to catch, and a non-refcounting runtime would leak immediately.
    # ------------------------------------------------------------------
    def test_refusal_closes_its_socket_explicitly_not_by_refcounting(self):
        upstream, requests, gate, arrived = self._start_gated_upstream(expected_arrivals=1)
        proxy = self._start_proxy(upstream, max_concurrent_requests=1)

        closed = []
        original = proxy.shutdown_request

        def spy(request):
            closed.append(request)
            return original(request)

        proxy.shutdown_request = spy

        holder_results = {}
        holder = threading.Thread(
            target=self._drive_request, args=(proxy.server_address[1], holder_results, "holder")
        )
        holder.start()
        self.assertTrue(arrived[0].wait(timeout=5), "the sole slot was never occupied")

        raw_request = b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n"
        response = self._send_and_read(proxy.server_address[1], raw_request)
        self.assertIn(b" 503 ", response)

        # The holder is still parked on the gate, so its own connection has
        # not been shut down yet -- every call counted here therefore
        # belongs to the refusal, making ``== 1`` exact rather than a lower
        # bound that a later completion could satisfy by accident.
        self.assertFalse(gate.is_set())
        self.assertEqual(
            len(closed), 1,
            "the refusal must close its request socket EXPLICITLY, not leave it to refcounting",
        )

        gate.set()
        holder.join(timeout=5)
        self.assertEqual(holder_results["holder"][0], 200)

    # ------------------------------------------------------------------
    # AC6 (determinism control): the IDENTICAL load that produced a 503 in
    # the very first test above -- two held connections plus one more --
    # must produce THREE 200s when the cap is raised to a production-sized
    # value. Without this control, the earlier 503 could in principle have
    # come from the listen() backlog or from test timing rather than from
    # the cap actually under test.
    # ------------------------------------------------------------------
    def test_identical_load_under_a_high_cap_yields_zero_refusals(self):
        upstream, requests, gate, arrived = self._start_gated_upstream(expected_arrivals=3)
        proxy = self._start_proxy(upstream, max_concurrent_requests=64)

        results = {}
        threads = [
            threading.Thread(target=self._drive_request, args=(proxy.server_address[1], results, i))
            for i in range(3)
        ]
        for t in threads:
            t.start()
        for ev in arrived:
            self.assertTrue(ev.wait(timeout=5))

        gate.set()
        for t in threads:
            t.join(timeout=5)
        for i in range(3):
            self.assertEqual(results[i][0], 200, "identical load under a high cap must never be refused")
        self.assertEqual(len(requests), 3)

    # ------------------------------------------------------------------
    # Refusal delivery: a client that sends its over-cap request and then
    # never reads the 503 response must still be REFUSED (503, correct
    # error type) and have its socket closed rather than left to hang,
    # and must not delay a concurrent, well-behaved connection's own
    # (503) response.
    #
    # What this test does NOT prove, despite its predecessor's name: the
    # accept-loop WEDGE class. The refusal body is 142 bytes, well under
    # any socket's send-buffer size, so ``sendall`` cannot block on a
    # non-reading client regardless of the write ``settimeout`` --
    # deleting that timeout outright leaves this test green (confirmed by
    # mutation during this test's own repair, see
    # ``docs/task-inbox/2026-09-20-refusal-drain-loop-wedges-the-accept-
    # thread.md``). The genuinely wedge-capable shape -- a client that
    # keeps dribbling bytes so no per-``recv`` timeout or EOF ever fires
    # -- is covered by ``test_paced_dribble_client_cannot_wedge_the_
    # accept_loop`` below, not by this test.
    # ------------------------------------------------------------------
    def test_client_that_never_reads_its_refusal_is_still_refused_and_closed(self):
        upstream, requests, gate, arrived = self._start_gated_upstream(expected_arrivals=2)
        proxy = self._start_proxy(upstream, max_concurrent_requests=2)

        results = {}
        t1 = threading.Thread(target=self._drive_request, args=(proxy.server_address[1], results, 1))
        t2 = threading.Thread(target=self._drive_request, args=(proxy.server_address[1], results, 2))
        t1.start()
        t2.start()
        self.assertTrue(arrived[0].wait(timeout=5))
        self.assertTrue(arrived[1].wait(timeout=5))

        raw_request = b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n"
        # Sends the request that triggers the 503, then never calls
        # recv() at all -- a client shape this proxy must still refuse
        # and close promptly, even though (see the docstring above) its
        # small, fixed-size body cannot exercise the send-side wedge.
        hostile = socket.create_connection(("127.0.0.1", proxy.server_address[1]), timeout=5)
        hostile.sendall(raw_request)

        started = time.monotonic()
        response = self._send_and_read(proxy.server_address[1], raw_request)
        elapsed = time.monotonic() - started
        self.assertLess(
            elapsed, 3,
            "a hostile client that never reads its refusal must not delay another connection's refusal",
        )
        self.assertIn(b" 503 ", response)

        hostile.close()
        gate.set()
        t1.join(timeout=5)
        t2.join(timeout=5)

    # ------------------------------------------------------------------
    # Accept-loop liveness, paced-dribble shape: the neighbouring hostile-
    # client test above sends ``Content-Length: 0`` and then goes SILENT,
    # which only exercises the drain loop's ``except OSError`` (timeout)
    # exit. A client that instead keeps delivering ~1 byte every 100ms
    # never lets a per-``recv`` timeout fire and never hits EOF either, so
    # a drain bounded only by a PER-RECV timeout (rather than a wall-clock
    # budget) keeps making forward progress on every single ``recv`` and
    # never gives up -- wedging the one accept thread for as long as the
    # dribble continues. Proves the drain must not wait for bytes at all,
    # only take what is already buffered.
    # ------------------------------------------------------------------
    def test_paced_dribble_client_cannot_wedge_the_accept_loop(self):
        upstream, requests, gate, arrived = self._start_gated_upstream(expected_arrivals=1)
        proxy = self._start_proxy(upstream, max_concurrent_requests=1)

        holder_results = {}
        holder = threading.Thread(
            target=self._drive_request, args=(proxy.server_address[1], holder_results, "holder")
        )
        holder.start()
        self.assertTrue(arrived[0].wait(timeout=5), "the sole slot was never occupied")

        raw_request = b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n"
        # Trips the refusal, then keeps dribbling one byte every 100ms
        # from a background thread instead of closing or going silent --
        # the paced-dribble shape the neighbouring hostile-client test
        # does NOT exercise.
        dribbler = socket.create_connection(("127.0.0.1", proxy.server_address[1]), timeout=5)
        dribbler.sendall(raw_request)

        stop_dribbling = threading.Event()

        def _pace(seconds: float) -> None:
            # A plain ``Event.wait``/``time.sleep`` for a short duration is
            # NOT used here: this test's own sandbox floors any syscall-
            # based sleep/wait to roughly 200-300ms regardless of the
            # requested duration (measured directly against this
            # environment during this test's own development), which would
            # make a 100ms-paced dribble arrive no faster than the drain's
            # own 0.2s per-``recv`` timeout and turn the very race this
            # test exists to pin into a coin flip. A busy-wait against
            # ``time.monotonic()`` is unaffected by that floor and paces
            # accurately.
            deadline = time.monotonic() + seconds
            while not stop_dribbling.is_set() and time.monotonic() < deadline:
                pass

        def dribble():
            while not stop_dribbling.is_set():
                try:
                    dribbler.sendall(b"x")
                except OSError:
                    return
                _pace(0.1)

        dribble_thread = threading.Thread(target=dribble, daemon=True)
        dribble_thread.start()

        def _stop_dribbler():
            stop_dribbling.set()
            dribble_thread.join(timeout=5)
            dribbler.close()

        self.addCleanup(_stop_dribbler)

        started = time.monotonic()
        response = self._send_and_read(proxy.server_address[1], raw_request, timeout=2)
        elapsed = time.monotonic() - started
        self.assertLess(
            elapsed, 1.5,
            "the single accept thread must not be held by a client that keeps dribbling "
            "bytes after its refusal",
        )
        self.assertIn(
            b" 503 ", response,
            "the single accept thread must not be held by a client that keeps dribbling "
            f"bytes after its refusal; got: {response!r}",
        )

        gate.set()
        holder.join(timeout=5)
        self.assertEqual(holder_results["holder"][0], 200)

    # ------------------------------------------------------------------
    # Accept-loop throughput: an ORDINARY refused client -- one that sends
    # its request then reads the response, never dribbles and never goes
    # silent -- must not cost the accept thread a fixed per-refusal delay.
    # Before this fix, the drain used a per-``recv`` ``settimeout(0.2)``:
    # since a benign client's request bytes are already sitting in this
    # proxy's kernel receive buffer before the drain's first ``recv`` even
    # runs, that ``recv`` still had to wait out the FULL 0.2s timeout for a
    # FIN that never arrives (the client is reading, not closing) --
    # measured at ~201ms per ordinary refusal, capping the whole proxy at
    # roughly 5 accepted connections/second. This proves N SEQUENTIAL
    # ordinary refusals complete in well under
    # ``N * _REFUSAL_DRAIN_DEADLINE_SECONDS`` total: on the fixed drain,
    # the first ``recv`` returns almost immediately (its bytes are already
    # buffered) and switches to non-blocking for the rest, so an ordinary
    # refusal costs close to nothing rather than a fixed delay -- see the
    # drain loop's own comment in ``_refuse_over_capacity`` for the full
    # reasoning.
    # ------------------------------------------------------------------
    def test_ordinary_refusal_does_not_cost_the_accept_thread_a_full_timeout(self):
        upstream, requests, gate, arrived = self._start_gated_upstream(expected_arrivals=1)
        proxy = self._start_proxy(upstream, max_concurrent_requests=1)

        holder_results = {}
        holder = threading.Thread(
            target=self._drive_request, args=(proxy.server_address[1], holder_results, "holder")
        )
        holder.start()
        self.assertTrue(arrived[0].wait(timeout=5), "the sole slot was never occupied")

        raw_request = b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n"
        deadline = FASTMLX_PROXY._REFUSAL_DRAIN_DEADLINE_SECONDS
        refusal_count = 20
        # Half of ``refusal_count * deadline``: generous enough not to flake
        # in a slow/loaded CI environment, yet far below what
        # ``refusal_count`` refusals would take if each one paid a fixed
        # ``deadline``-sized (let alone the original 0.2s) cost -- the
        # ORIGINAL ``settimeout(0.2)`` code fails this bound by roughly an
        # order of magnitude (confirmed against a scratch copy of that
        # code during this test's own development).
        budget_seconds = (refusal_count * deadline) / 2

        started = time.monotonic()
        for _ in range(refusal_count):
            response = self._send_and_read(proxy.server_address[1], raw_request, timeout=5)
            self.assertIn(b" 503 ", response)
        elapsed = time.monotonic() - started

        self.assertLess(
            elapsed, budget_seconds,
            f"{refusal_count} ordinary sequential refusals took {elapsed:.3f}s, not well "
            f"under the {budget_seconds:.3f}s budget -- the accept thread appears to be "
            "burning a fixed delay per ordinary refusal instead of returning almost "
            "immediately once the client's already-buffered bytes are drained",
        )

        gate.set()
        holder.join(timeout=5)
        self.assertEqual(holder_results["holder"][0], 200)

    # ------------------------------------------------------------------
    # AC8: each refusal is observable as exactly one JSONL log line
    # carrying status 503 and the error type, via the same ``log_hook``
    # path production logging uses -- lets an operator alert on this the
    # same way every other proxy-generated error status is already
    # observable.
    # ------------------------------------------------------------------
    def test_refusal_writes_one_jsonl_log_line_with_status_and_error_type(self):
        logged = []
        logged_event = threading.Event()

        def log_hook(entry):
            if entry.get("event") == "saturation_snapshot":
                return  # a separate, unrelated feature's line; see its own tests
            logged.append(entry)
            logged_event.set()

        upstream, requests, gate, arrived = self._start_gated_upstream(expected_arrivals=1)
        proxy = self._start_proxy(upstream, max_concurrent_requests=1, log_hook=log_hook)

        holder_results = {}
        holder = threading.Thread(target=self._drive_request, args=(proxy.server_address[1], holder_results, "holder"))
        holder.start()
        self.assertTrue(arrived[0].wait(timeout=5))

        raw_request = b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n"
        response = self._send_and_read(proxy.server_address[1], raw_request)
        self.assertIn(b" 503 ", response)
        self.assertTrue(logged_event.wait(timeout=5), "the refusal was never logged via log_hook")

        self.assertEqual(len(logged), 1)
        self.assertEqual(logged[0]["status"], 503)
        self.assertEqual(logged[0]["error"], "too_many_concurrent_requests")

        gate.set()
        holder.join(timeout=5)
        self.assertEqual(holder_results["holder"][0], 200)

    # ------------------------------------------------------------------
    # Helpers shared by the rate-limit tests below: refuse ``count`` times
    # in a tight sequential loop against a server whose sole slot is held,
    # returning the raw responses so a caller can also assert on the HTTP
    # side if it wants to.
    # ------------------------------------------------------------------
    def _drive_refusal_burst(self, proxy, count: int) -> None:
        raw_request = b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n"
        for _ in range(count):
            response = self._send_and_read(proxy.server_address[1], raw_request)
            self.assertIn(b" 503 ", response)

    @staticmethod
    def _stderr_jsonl_lines(stderr_buffer: io.StringIO) -> list:
        return [json.loads(line) for line in stderr_buffer.getvalue().splitlines() if line]

    # ------------------------------------------------------------------
    # This is the load-bearing rate-limit test: a burst of refusals well
    # within a single ``_REFUSAL_LOG_MIN_INTERVAL_SECONDS`` window must
    # produce exactly ONE JSONL line on ``stderr`` -- proves the accept
    # thread is no longer paying an unconditional synchronous write per
    # refusal (see ``_REFUSAL_LOG_MIN_INTERVAL_SECONDS`` in
    # ``fastmlx_proxy.py`` for the disk-fill/accept-thread-wedge motivation).
    # ------------------------------------------------------------------
    def test_refusal_log_burst_is_rate_limited_on_stderr(self):
        upstream, requests, gate, arrived = self._start_gated_upstream(expected_arrivals=1)
        proxy = self._start_proxy(upstream, max_concurrent_requests=1)

        holder_results = {}
        holder = threading.Thread(target=self._drive_request, args=(proxy.server_address[1], holder_results, "holder"))
        holder.start()
        self.assertTrue(arrived[0].wait(timeout=5), "the sole slot was never occupied")

        stderr_buffer = io.StringIO()
        with contextlib.redirect_stderr(stderr_buffer):
            self._drive_refusal_burst(proxy, 20)

        lines = self._stderr_jsonl_lines(stderr_buffer)
        self.assertEqual(
            len(lines), 1,
            f"expected exactly one stderr line for a burst well inside "
            f"{FASTMLX_PROXY._REFUSAL_LOG_MIN_INTERVAL_SECONDS}s, got {len(lines)}: {lines!r}",
        )

        gate.set()
        holder.join(timeout=5)
        self.assertEqual(holder_results["holder"][0], 200)

    # ------------------------------------------------------------------
    # The rate limit above must not become an observability hole: every
    # refusal in the same burst still has to reach ``log_hook``, which is
    # deliberately NOT rate-limited (see ``_log_capacity_refusal``'s own
    # comment for why the split is load-bearing). Without this test, a
    # broken implementation that dropped entries instead of merely
    # suppressing their stderr line would pass the test above for the
    # wrong reason.
    # ------------------------------------------------------------------
    def test_every_refusal_still_reaches_the_log_hook(self):
        logged = []
        logged_lock = threading.Lock()

        def log_hook(entry):
            if entry.get("event") == "saturation_snapshot":
                return  # a separate, unrelated feature's line; see its own tests
            with logged_lock:
                logged.append(entry)

        upstream, requests, gate, arrived = self._start_gated_upstream(expected_arrivals=1)
        proxy = self._start_proxy(upstream, max_concurrent_requests=1, log_hook=log_hook)

        holder_results = {}
        holder = threading.Thread(target=self._drive_request, args=(proxy.server_address[1], holder_results, "holder"))
        holder.start()
        self.assertTrue(arrived[0].wait(timeout=5), "the sole slot was never occupied")

        stderr_buffer = io.StringIO()
        with contextlib.redirect_stderr(stderr_buffer):
            self._drive_refusal_burst(proxy, 20)

        self.assertEqual(len(self._stderr_jsonl_lines(stderr_buffer)), 1)
        self.assertEqual(
            len(logged), 20,
            "log_hook must fire once per refusal even while stderr is rate-limited",
        )

        gate.set()
        holder.join(timeout=5)
        self.assertEqual(holder_results["holder"][0], 200)

    # ------------------------------------------------------------------
    # The emitted line must carry how many refusals were swallowed since
    # the previous emitted line, so an operator reading ``stderr`` can
    # still recover the true refusal count instead of undercounting by
    # however many were suppressed.
    # ------------------------------------------------------------------
    def test_emitted_refusal_log_reports_how_many_it_suppressed(self):
        upstream, requests, gate, arrived = self._start_gated_upstream(expected_arrivals=1)
        proxy = self._start_proxy(upstream, max_concurrent_requests=1)

        holder_results = {}
        holder = threading.Thread(target=self._drive_request, args=(proxy.server_address[1], holder_results, "holder"))
        holder.start()
        self.assertTrue(arrived[0].wait(timeout=5), "the sole slot was never occupied")

        stderr_buffer = io.StringIO()
        with contextlib.redirect_stderr(stderr_buffer):
            # First refusal always emits (see the next test); the other 4
            # in this burst land inside the same interval and are
            # suppressed.
            self._drive_refusal_burst(proxy, 5)
            lines_before = self._stderr_jsonl_lines(stderr_buffer)
            self.assertEqual(len(lines_before), 1)

            # Force the NEXT refusal to emit regardless of elapsed time, by
            # collapsing the module's own rate-limit window to zero --
            # imported from the module rather than hardcoded, per this
            # suite's own convention (see ``_REFUSAL_DRAIN_DEADLINE_SECONDS``
            # usage elsewhere in this file).
            with unittest.mock.patch.object(FASTMLX_PROXY, "_REFUSAL_LOG_MIN_INTERVAL_SECONDS", 0.0):
                self._drive_refusal_burst(proxy, 1)

        lines_after = self._stderr_jsonl_lines(stderr_buffer)
        self.assertEqual(len(lines_after), 2, f"expected exactly 2 emitted lines total: {lines_after!r}")
        self.assertEqual(
            lines_after[1]["suppressed_since_last_log"], 4,
            "the forced emission must report exactly the 4 refusals suppressed since the first line",
        )

        gate.set()
        holder.join(timeout=5)
        self.assertEqual(holder_results["holder"][0], 200)

    # ------------------------------------------------------------------
    # An operator must see the FIRST refusal on a fresh server immediately,
    # not have it silently suppressed because it happened to land inside
    # what would otherwise be treated as "since construction" -- this is
    # why ``_refusal_log_last_emitted_monotonic`` is initialized to
    # ``None``, not to a timestamp taken at construction time.
    # ------------------------------------------------------------------
    def test_first_refusal_is_logged_immediately(self):
        upstream, requests, gate, arrived = self._start_gated_upstream(expected_arrivals=1)
        proxy = self._start_proxy(upstream, max_concurrent_requests=1)

        holder_results = {}
        holder = threading.Thread(target=self._drive_request, args=(proxy.server_address[1], holder_results, "holder"))
        holder.start()
        self.assertTrue(arrived[0].wait(timeout=5), "the sole slot was never occupied")

        stderr_buffer = io.StringIO()
        with contextlib.redirect_stderr(stderr_buffer):
            self._drive_refusal_burst(proxy, 1)

        lines = self._stderr_jsonl_lines(stderr_buffer)
        self.assertEqual(len(lines), 1, "the very first refusal on a fresh server must be logged immediately")
        self.assertEqual(lines[0]["suppressed_since_last_log"], 0)

        gate.set()
        holder.join(timeout=5)
        self.assertEqual(holder_results["holder"][0], 200)

    # ------------------------------------------------------------------
    # Shared helper for the request-receive-deadline tests below: polls
    # ``proxy.inflight_requests`` (never sleeps a fixed guess) until it
    # reaches ``expected`` or ``timeout`` elapses, returning the
    # monotonic time the condition was FIRST observed (or ``None``).
    # ------------------------------------------------------------------
    @staticmethod
    def _wait_for_inflight(proxy, expected: int, timeout: float):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proxy.inflight_requests == expected:
                return time.monotonic()
            time.sleep(0.01)
        return None

    # ------------------------------------------------------------------
    # T-A (load-bearing regression test for the CONFIRMED defect): a
    # client that dribbles ONE byte at a time, slower than
    # ``_CLIENT_REQUEST_MIN_BYTES_PER_SECOND``, on the ADMITTED path
    # (cap=1, so it holds the sole slot), must have its slot reclaimed
    # once the request-receive deadline passes -- and a legitimate client
    # queued behind that slot must then be served. Before the fix, this
    # never happened: ``socket.settimeout`` bounds one ``recv``, not a
    # request, so a client that delivers a byte before each per-``recv``
    # timeout expires keeps the handler (and its concurrency slot) alive
    # forever (see docs/task-inbox/2026-09-20-dribbling-client-holds-a-
    # concurrency-slot-indefinitely.md for the full record and a 64-socket,
    # ~0.53 B/s reproduction of a total proxy outage).
    #
    # Constants are scaled down by roughly 60x-200x from their real
    # defaults (30s/1024 B/s/300s -> 0.5s/100 B/s/1.5s) so this test
    # completes in a few seconds -- the per-``read1`` deadline
    # recomputation this fix relies on is scale-invariant (see
    # ``_DeadlineBoundRfile`` in fastmlx_proxy.py), the same way
    # ``test_paced_dribble_client_cannot_wedge_the_accept_loop`` above
    # scales ``_REFUSAL_DRAIN_DEADLINE_SECONDS``.
    #
    # REQUIRED CONTROL ARM: a SILENT client (opens a connection, sends the
    # same partial request, then sends nothing more at all) must ALSO be
    # reaped -- proving this test's own machinery, and the deadline
    # mechanism itself, actually work on the shape the OLD per-recv
    # timeout already handled. Without this control, the dribbler simply
    # never being reaped could in principle be a broken harness rather
    # than the mechanism under test.
    # ------------------------------------------------------------------
    def test_dribbling_client_on_the_admitted_path_cannot_hold_its_slot_forever(self):
        with unittest.mock.patch.object(FASTMLX_PROXY, "_CLIENT_REQUEST_INITIAL_SECONDS", 0.5), \
             unittest.mock.patch.object(FASTMLX_PROXY, "_CLIENT_REQUEST_MIN_BYTES_PER_SECOND", 100), \
             unittest.mock.patch.object(FASTMLX_PROXY, "_CLIENT_REQUEST_HARD_DEADLINE_SECONDS", 1.5), \
             unittest.mock.patch.object(FASTMLX_PROXY.ProvenanceProxyHandler, "timeout", 1.5):
            # The ``timeout`` class attribute is ALSO scaled down here (not
            # just the three ``_CLIENT_REQUEST_*`` constants): it is baked
            # onto ``ProvenanceProxyHandler`` at class-definition time from
            # the real, unscaled ``_CLIENT_IDLE_TIMEOUT_SECONDS`` (120s), so
            # on the UNFIXED code path (M1: the wrapper installation
            # removed, see the implementation report) it is the ONLY thing
            # governing this test's own CONTROL arm, and 120s would make
            # that control unobservable inside this test's bounded waits.
            # This patch is a no-op for the FIXED code's own outcome here
            # (immediately superseded by ``_DeadlineBoundRfile`` on the
            # first read either way).

            def responder(handler):
                payload = b"ok"
                handler.send_response(200)
                handler.send_header("Content-Length", str(len(payload)))
                handler.end_headers()
                handler.wfile.write(payload)

            upstream = start_fake_upstream(responder)
            self._servers.append(upstream)
            proxy = self._start_proxy(upstream, max_concurrent_requests=1)

            # Request line + ONE header, deliberately with NO terminating
            # blank line -- the request is never complete.
            partial_request = b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"

            # --- CONTROL: the SILENT shape (the old per-recv timeout's
            # own documented target) must still be reaped by the new
            # deadline. ---
            silent = socket.create_connection(("127.0.0.1", proxy.server_address[1]), timeout=5)
            silent.sendall(partial_request)
            self.assertIsNotNone(
                self._wait_for_inflight(proxy, 1, timeout=2),
                "CONTROL: the silent client's connection was never even admitted",
            )
            silent_freed_at = self._wait_for_inflight(proxy, 0, timeout=5)
            silent.close()
            self.assertIsNotNone(
                silent_freed_at,
                "CONTROL: the silent client was never reaped -- if this control fails, "
                "the dribbler result below is not discriminating",
            )

            # --- ATTACK: the dribbler. One byte every 0.15s -- far slower
            # than the scaled 100 B/s minimum (which would need one byte
            # every 0.01s to keep pace) -- kept up for well longer than
            # the scaled 1.5s hard ceiling.
            dribbler = socket.create_connection(("127.0.0.1", proxy.server_address[1]), timeout=5)
            dribbler.sendall(partial_request)
            self.assertIsNotNone(
                self._wait_for_inflight(proxy, 1, timeout=2),
                "the dribbler's connection was never even admitted",
            )

            stop_dribbling = threading.Event()

            def dribble():
                while not stop_dribbling.is_set():
                    try:
                        dribbler.sendall(b"z")
                    except OSError:
                        return
                    stop_dribbling.wait(0.15)

            dribble_thread = threading.Thread(target=dribble, daemon=True)
            dribble_thread.start()

            def _stop_dribbler():
                stop_dribbling.set()
                dribble_thread.join(timeout=5)
                try:
                    dribbler.close()
                except OSError:
                    pass

            self.addCleanup(_stop_dribbler)

            freed_at = self._wait_for_inflight(proxy, 0, timeout=5)
            self.assertIsNotNone(
                freed_at,
                "ATTACK: the dribbling client on the ADMITTED path was never reaped -- "
                "it held its concurrency slot indefinitely",
            )
            self.assertFalse(
                stop_dribbling.is_set(),
                "the dribbler must still be dribbling at the moment its slot is freed -- "
                "otherwise the release could just be an ordinary close, not the deadline",
            )

            # The slot must be genuinely USABLE again, not merely counted
            # as free: a legitimate client must be served.
            conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=5)
            conn.request("GET", "/v1/models")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            self.assertEqual(
                resp.status, 200,
                "a legitimate client must be served once the dribbler's slot is freed",
            )

    # ------------------------------------------------------------------
    # T-B (anti-vacuity control for T-A/the fix): an HONEST client that
    # sends a large body SLOWLY but AT OR ABOVE
    # ``_CLIENT_REQUEST_MIN_BYTES_PER_SECOND`` must get its normal
    # response, even though its total transfer time exceeds
    # ``_CLIENT_REQUEST_INITIAL_SECONDS``. Without this test, "refuse
    # everything slow" (e.g. deleting the extension logic entirely) would
    # pass T-A for the wrong reason -- see mutation M2 in the
    # implementation report.
    # ------------------------------------------------------------------
    def test_client_at_or_above_minimum_rate_is_not_refused(self):
        with unittest.mock.patch.object(FASTMLX_PROXY, "_CLIENT_REQUEST_INITIAL_SECONDS", 1.0), \
             unittest.mock.patch.object(FASTMLX_PROXY, "_CLIENT_REQUEST_MIN_BYTES_PER_SECOND", 200), \
             unittest.mock.patch.object(FASTMLX_PROXY, "_CLIENT_REQUEST_HARD_DEADLINE_SECONDS", 10.0):

            received = {}

            def responder(handler):
                received["body"] = _read_request_body(handler)
                payload = b'{"ok":true}'
                handler.send_response(200)
                handler.send_header("Content-Type", "application/json")
                handler.send_header("Content-Length", str(len(payload)))
                handler.end_headers()
                handler.wfile.write(payload)

            upstream = start_fake_upstream(responder)
            self._servers.append(upstream)
            proxy = self._start_proxy(upstream, max_concurrent_requests=1)

            # 4 chunks of 250 bytes (1000 bytes total), spaced 1.0s apart
            # -- 250 B/s, comfortably ABOVE the scaled 200 B/s minimum --
            # sent over roughly 3s total, more than 3x the scaled 1.0s
            # initial grace period, and still well inside the scaled 10s
            # hard ceiling.
            chunk = b"y" * 250
            body = chunk * 4
            sock = socket.create_connection(("127.0.0.1", proxy.server_address[1]), timeout=15)
            sock.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n\r\n"
            )
            started = time.monotonic()
            for _ in range(4):
                sock.sendall(chunk)
                time.sleep(1.0)
            elapsed_sending = time.monotonic() - started

            response = b""
            sock.settimeout(10)
            try:
                while True:
                    piece = sock.recv(4096)
                    if not piece:
                        break
                    response += piece
            except socket.timeout:
                pass
            sock.close()

            self.assertGreater(
                elapsed_sending, 1.0,
                "test setup error: this must exceed the scaled initial grace period "
                "for the extension logic to actually be exercised",
            )
            self.assertIn(b" 200 ", response, f"an at-or-above-minimum-rate client must not be refused: {response!r}")
            self.assertEqual(received["body"], body, "the full body must reach the upstream engine intact")

    # ------------------------------------------------------------------
    # T-C: the hard ceiling bounds even a client that never dips below
    # the minimum rate -- without it, the extension rule in T-B would be
    # unbounded (see mutation M3 in the implementation report).
    # ------------------------------------------------------------------
    def test_hard_ceiling_bounds_even_a_compliant_rate_client(self):
        with unittest.mock.patch.object(FASTMLX_PROXY, "_CLIENT_REQUEST_INITIAL_SECONDS", 0.3), \
             unittest.mock.patch.object(FASTMLX_PROXY, "_CLIENT_REQUEST_MIN_BYTES_PER_SECOND", 50), \
             unittest.mock.patch.object(FASTMLX_PROXY, "_CLIENT_REQUEST_HARD_DEADLINE_SECONDS", 1.0):

            def responder(handler):
                payload = b"ok"
                handler.send_response(200)
                handler.send_header("Content-Length", str(len(payload)))
                handler.end_headers()
                handler.wfile.write(payload)

            upstream = start_fake_upstream(responder)
            self._servers.append(upstream)
            proxy = self._start_proxy(upstream, max_concurrent_requests=1)

            sock = socket.create_connection(("127.0.0.1", proxy.server_address[1]), timeout=5)
            sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n")
            connected_at = time.monotonic()
            self.assertIsNotNone(
                self._wait_for_inflight(proxy, 1, timeout=2),
                "the compliant-rate client's connection was never even admitted",
            )

            stop_sending = threading.Event()

            def keep_sending_at_compliant_rate():
                # 15 bytes every 0.2s = 75 B/s, comfortably ABOVE the
                # scaled 50 B/s minimum -- would extend the deadline
                # forever if there were no hard ceiling.
                while not stop_sending.is_set():
                    try:
                        sock.sendall(b"X-Pad: " + b"a" * 8)  # filler bytes, deliberately no newline
                    except OSError:
                        return
                    stop_sending.wait(0.2)

            sender = threading.Thread(target=keep_sending_at_compliant_rate, daemon=True)
            sender.start()

            def _stop_sender():
                stop_sending.set()
                sender.join(timeout=5)
                try:
                    sock.close()
                except OSError:
                    pass

            self.addCleanup(_stop_sender)

            freed_at = self._wait_for_inflight(proxy, 0, timeout=5)
            self.assertIsNotNone(
                freed_at,
                "a client that never dips below the minimum rate was never cut off -- "
                "the hard ceiling appears to be missing or unbounded",
            )
            self.assertFalse(
                stop_sending.is_set(),
                "the client must still be sending at a compliant rate when its slot is freed",
            )
            elapsed = freed_at - connected_at
            hard_deadline = FASTMLX_PROXY._CLIENT_REQUEST_HARD_DEADLINE_SECONDS
            self.assertGreaterEqual(
                elapsed, hard_deadline * 0.8,
                f"cut off too EARLY ({elapsed:.3f}s) relative to the {hard_deadline}s hard "
                "ceiling -- this must be the ceiling firing, not the initial grace period",
            )
            self.assertLess(
                elapsed, hard_deadline + 2.0,
                f"cut off too LATE ({elapsed:.3f}s) relative to the {hard_deadline}s hard ceiling",
            )

            conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=5)
            conn.request("GET", "/v1/models")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            self.assertEqual(
                resp.status, 200,
                "a legitimate client must be served once the compliant-rate client's "
                "slot is freed by the hard ceiling",
            )

    # ------------------------------------------------------------------
    # T-D: the request-receive deadline must NOT cut an SSE stream short
    # -- it bounds RECEIVING the request only. Constants scaled SHORT so
    # a stream whose generation deliberately outlasts the deadline must
    # still complete successfully; reuses the existing gated-SSE fake-
    # upstream pattern from
    # ``test_streaming_sse_response_holds_its_slot_for_the_whole_
    # generation`` above.
    # ------------------------------------------------------------------
    def test_sse_stream_is_not_cut_by_the_request_receive_deadline(self):
        with unittest.mock.patch.object(FASTMLX_PROXY, "_CLIENT_REQUEST_INITIAL_SECONDS", 0.2), \
             unittest.mock.patch.object(FASTMLX_PROXY, "_CLIENT_REQUEST_MIN_BYTES_PER_SECOND", 1), \
             unittest.mock.patch.object(FASTMLX_PROXY, "_CLIENT_REQUEST_HARD_DEADLINE_SECONDS", 0.5):

            release_rest = threading.Event()

            def responder(handler):
                _read_request_body(handler)
                handler.send_response(200)
                handler.send_header("Content-Type", "text/event-stream")
                handler.end_headers()
                handler.wfile.write(b"data: chunk1\n\n")
                handler.wfile.flush()
                # Longer than the scaled 0.5s hard ceiling above -- if the
                # deadline wrongly covered response generation, this
                # would be cut off before chunk2 is ever sent.
                self.assertTrue(release_rest.wait(timeout=10), "test never released the gate")
                handler.wfile.write(b"data: chunk2\n\n")
                handler.wfile.flush()

            upstream = start_fake_upstream(responder)
            self._servers.append(upstream)
            proxy = self._start_proxy(upstream, max_concurrent_requests=1)

            received_first = threading.Event()
            stream_result = {}

            def read_stream():
                conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=10)
                conn.request("GET", "/v1/stream")
                resp = conn.getresponse()
                first = resp.read1(4096) if hasattr(resp, "read1") else resp.read(1)
                stream_result["first"] = first
                received_first.set()
                stream_result["rest"] = resp.read()
                conn.close()

            reader = threading.Thread(target=read_stream)
            reader.start()
            self.assertTrue(received_first.wait(timeout=5), "client never received the first SSE chunk")

            time.sleep(1.0)  # well past the scaled 0.5s hard ceiling
            release_rest.set()
            reader.join(timeout=5)

            self.assertEqual(
                stream_result.get("first", b"") + stream_result.get("rest", b""),
                b"data: chunk1\n\ndata: chunk2\n\n",
                "the SSE stream must complete in full -- the request-receive deadline "
                "must never bound response generation",
            )

    # ------------------------------------------------------------------
    # M1 (capture-point regression, the load-bearing test of this group):
    # a refusal's ``inflight`` MUST be the value ``process_request``
    # captured at the instant its semaphore acquire failed, never a later
    # ``self.inflight_requests`` read taken inside ``_log_capacity_
    # refusal`` -- that method only runs AFTER ``_refuse_over_capacity``'s
    # own post-503 drain loop, which can run for up to
    # ``_REFUSAL_DRAIN_DEADLINE_SECONDS`` (measured at ~52ms for exactly
    # this silent-client shape), and another thread releasing its slot
    # during that window would otherwise be misattributed to a refusal
    # that happened before the release.
    #
    # Proof shape: a SILENT client -- opens a connection and sends
    # nothing at all, ever -- triggers ``process_request``'s failed
    # acquire the moment its TCP handshake completes (accept() does not
    # require any application byte), then sits parked in the drain
    # loop's blocking ``recv``. Once a fixed pause has given the accept
    # thread ample time to have already captured its counters and
    # entered that blocking ``recv`` (a sub-millisecond synchronous
    # prefix in real time -- 200ms is a large multiple of it, not a
    # tight race), the test releases the holder, THEN closes the silent
    # socket to end the drain via EOF. This ordering is deterministic,
    # not merely usually-true: the capture (correct code) or the leak-
    # through read (mutated code) cannot occur any earlier than the
    # already-elapsed pause, and the holder release cannot occur any
    # later than immediately after it -- so the correct code must log
    # ``inflight: 1`` and the mutation described above must log
    # ``inflight: 0``. ``_REFUSAL_DRAIN_DEADLINE_SECONDS`` is patched
    # larger purely so this ordering never has to race a loaded CI
    # runner's scheduler; the mechanism under test (capture-before-drain,
    # not the deadline's magnitude) is unaffected by that value, the same
    # way other tests in this file scale timing constants for
    # determinism (see ``test_dribbling_client_on_the_admitted_path_
    # cannot_hold_its_slot_forever``'s own comment on scale invariance).
    # ------------------------------------------------------------------
    def test_refusal_inflight_is_captured_at_failed_acquire_not_after_drain(self):
        # ``log_hook`` also fires for the HOLDER's own ordinary 200
        # completion (``ProvenanceProxyHandler._log`` shares the same
        # hook -- see its own comment) -- and that 200 entry is logged
        # the moment ``gate.set()`` below releases it, which is BEFORE
        # this test ever closes the silent socket that ends the drain.
        # A single undifferentiated ``logged_event``/``logged[0]`` would
        # therefore let this test's assertions run against the HOLDER's
        # entry instead of the refusal's, passing vacuously regardless of
        # which value ``_log_capacity_refusal`` actually used -- exactly
        # the trap this comment exists to name. So this test tracks the
        # 503 entry specifically, by status, never by list position or
        # "the first/only thing logged".
        logged = []
        refusal_logged_event = threading.Event()

        def log_hook(entry):
            logged.append(entry)
            if entry.get("status") == 503:
                refusal_logged_event.set()

        upstream, requests, gate, arrived = self._start_gated_upstream(expected_arrivals=1)
        proxy = self._start_proxy(upstream, max_concurrent_requests=1, log_hook=log_hook)

        holder_results = {}
        holder = threading.Thread(
            target=self._drive_request, args=(proxy.server_address[1], holder_results, "holder")
        )
        holder.start()
        self.assertTrue(arrived[0].wait(timeout=5), "the sole slot was never occupied")

        with unittest.mock.patch.object(FASTMLX_PROXY, "_REFUSAL_DRAIN_DEADLINE_SECONDS", 5.0):
            silent = socket.create_connection(("127.0.0.1", proxy.server_address[1]), timeout=10)
            self.addCleanup(silent.close)

            # Generous pause: the accept thread's failed-acquire capture
            # and 503 write are synchronous, in-process, and require no
            # I/O wait (the silent client never sends anything, so there
            # is nothing for the accept thread to wait on except this
            # pause itself) -- 200ms is a large multiple of that real
            # cost, not a tight deadline.
            time.sleep(0.2)

            # The holder's slot is released only NOW -- strictly after
            # the pause above, so strictly after the accept thread has
            # already captured its counters (correct code) or is already
            # parked waiting to make its later, post-drain read (mutated
            # code).
            gate.set()
            holder.join(timeout=5)
            self.assertEqual(holder_results["holder"][0], 200)

            # End the drain now (EOF) instead of waiting out the full
            # patched deadline -- the release above already happened
            # before this, which is the only ordering this test needs.
            silent.close()

        self.assertTrue(
            refusal_logged_event.wait(timeout=10), "the refusal was never logged via log_hook"
        )
        refusal_entries = [entry for entry in logged if entry.get("status") == 503]
        self.assertEqual(
            len(refusal_entries), 1,
            f"expected exactly one 503 refusal log entry, got {len(refusal_entries)}: {logged!r}",
        )
        self.assertEqual(
            refusal_entries[0]["inflight"], 1,
            "inflight must reflect the state AT THE FAILED ACQUIRE (the holder still held "
            f"its slot then), not the state after the drain (already released): {refusal_entries[0]}",
        )

    # ------------------------------------------------------------------
    # M3: ``refusals_total`` must count EVERY refusal, not only the ones
    # whose stderr line survives the 1s rate limit. Uses ``log_hook``
    # (fires unconditionally per refusal, never rate-limited -- see its
    # own comment in ``_log_capacity_refusal``) so this test never has to
    # fight ``_REFUSAL_LOG_MIN_INTERVAL_SECONDS``.
    # ------------------------------------------------------------------
    def test_refusals_total_counts_every_refusal_not_only_emitted_ones(self):
        logged = []
        logged_lock = threading.Lock()

        def log_hook(entry):
            if entry.get("event") == "saturation_snapshot":
                return  # a separate, unrelated feature's line; see its own tests
            with logged_lock:
                logged.append(entry)

        upstream, requests, gate, arrived = self._start_gated_upstream(expected_arrivals=1)
        proxy = self._start_proxy(upstream, max_concurrent_requests=1, log_hook=log_hook)

        holder_results = {}
        holder = threading.Thread(
            target=self._drive_request, args=(proxy.server_address[1], holder_results, "holder")
        )
        holder.start()
        self.assertTrue(arrived[0].wait(timeout=5), "the sole slot was never occupied")

        stderr_buffer = io.StringIO()
        with contextlib.redirect_stderr(stderr_buffer):
            # All 20 refusals below land well inside a single
            # ``_REFUSAL_LOG_MIN_INTERVAL_SECONDS`` (1s) window -- only
            # the FIRST gets an emitted stderr line (asserted below), but
            # every one of them must still bump ``refusals_total``.
            self._drive_refusal_burst(proxy, 20)

        self.assertEqual(
            len(self._stderr_jsonl_lines(stderr_buffer)), 1,
            "test setup assumption violated: all 20 refusals must land inside one rate-limit window",
        )
        self.assertEqual(len(logged), 20)
        self.assertGreaterEqual(
            logged[-1].get("refusals_total", 0), 20,
            f"refusals_total must count every refusal, not only the emitted one: {logged[-1]}",
        )

        gate.set()
        holder.join(timeout=5)
        self.assertEqual(holder_results["holder"][0], 200)

    # ------------------------------------------------------------------
    # M5: the normal-path ``inflight`` gauge ``ProvenanceProxyHandler.
    # _log`` writes must be read BEFORE ``ProvenanceProxyServer.
    # process_request_thread``'s own ``finally`` decrements the counter
    # for the very request being logged -- a single sequential request
    # therefore must log ``inflight: 1`` (itself), never 0.
    # ------------------------------------------------------------------
    def test_normal_path_inflight_counts_the_logged_request_itself(self):
        logged = []
        logged_event = threading.Event()

        def log_hook(entry):
            logged.append(entry)
            logged_event.set()

        def responder(handler):
            payload = b"{}"
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)

        upstream = start_fake_upstream(responder)
        self._servers.append(upstream)
        proxy = self._start_proxy(upstream, max_concurrent_requests=2, log_hook=log_hook)

        conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=5)
        conn.request("POST", "/v1/chat/completions", body=b"{}")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.assertEqual(resp.status, 200)

        self.assertTrue(logged_event.wait(timeout=5), "the request was never logged via log_hook")
        self.assertEqual(len(logged), 1)
        self.assertEqual(
            logged[0]["inflight"], 1,
            f"a single sequential request must log inflight=1 (itself), not 0: {logged[0]}",
        )

    # ------------------------------------------------------------------
    # M6: a refusal's ``max_concurrent`` must always be ``self.
    # max_concurrent_requests`` for THIS server, never a literal --
    # constructed with a non-default cap (3) so the field would be green
    # by construction if it were hardcoded to the 64 default.
    # ------------------------------------------------------------------
    def test_refusal_max_concurrent_reflects_the_configured_cap_not_a_literal(self):
        logged = []
        logged_event = threading.Event()

        def log_hook(entry):
            if entry.get("event") == "saturation_snapshot":
                return  # a separate, unrelated feature's line; see its own tests
            logged.append(entry)
            logged_event.set()

        upstream, requests, gate, arrived = self._start_gated_upstream(expected_arrivals=3)
        proxy = self._start_proxy(upstream, max_concurrent_requests=3, log_hook=log_hook)

        results = {}
        holders = [
            threading.Thread(target=self._drive_request, args=(proxy.server_address[1], results, i))
            for i in range(3)
        ]
        for t in holders:
            t.start()
        for ev in arrived:
            self.assertTrue(ev.wait(timeout=5))

        raw_request = b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n"
        response = self._send_and_read(proxy.server_address[1], raw_request)
        self.assertIn(b" 503 ", response)
        self.assertTrue(logged_event.wait(timeout=5), "the refusal was never logged via log_hook")

        self.assertEqual(len(logged), 1)
        self.assertEqual(
            logged[0]["max_concurrent"], 3,
            f"max_concurrent must reflect this server's configured cap (3), not a literal: {logged[0]}",
        )
        # Semantics 1 (see the ``inflight`` field's own comment in
        # ``fastmlx_proxy.py``): the failed acquire and this counter read
        # are different locks, so a legitimate refusal can read less than
        # the cap -- only the range is asserted here, never exact
        # equality, which would be flaky by construction.
        self.assertGreaterEqual(logged[0]["inflight"], 1)
        self.assertLessEqual(logged[0]["inflight"], 3)

        gate.set()
        for t in holders:
            t.join(timeout=5)
        for i in range(3):
            self.assertEqual(results[i][0], 200)


class ProxyExpect100ContinueTests(unittest.TestCase):
    """A client sending ``Expect: 100-continue`` must get an interim ``100
    Continue`` promptly, not hang until ITS OWN timeout fires and it sends
    the body unprompted anyway (curl's behavior: wait ~1s, then send). Root
    cause: ``BaseHTTPRequestHandler.parse_request`` only calls
    ``handle_expect_100`` when ``self.protocol_version >= "HTTP/1.1"`` --
    since ``ProvenanceProxyHandler.protocol_version`` is pinned to
    ``HTTP/1.0`` (deliberately, see the comment above that assignment), the
    base class never even reaches ``handle_expect_100`` for ANY request,
    regardless of what the client sent. See ``ProvenanceProxyHandler``'s own
    ``parse_request``/``handle_expect_100`` overrides for the fix.
    """

    def setUp(self):
        self._listeners = []
        self._servers = []
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for server in self._servers:
            server.shutdown()
            server.server_close()
            server._test_thread.join(timeout=5)
        for listener in self._listeners:
            listener.close()

    def _start_upstream(self, handle_conn):
        listener, _thread = _start_raw_socket_upstream(handle_conn)
        self._listeners.append(listener)
        return listener

    def _start_proxy(self, upstream_port: int, **kwargs):
        server = FASTMLX_PROXY.create_server(
            "127.0.0.1", 0, "127.0.0.1", upstream_port, FIXTURE_PLAN, **kwargs
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        server._test_thread = thread
        self._servers.append(server)
        return server

    # 1. A request with Expect: 100-continue AND a body must complete
    #    promptly (well under curl's own 1s wait-then-send-anyway window),
    #    and the 100 Continue interim line must arrive BEFORE the client
    #    writes the body -- proven with a raw socket so the handshake
    #    order is under this test's control, not http.client's.
    def test_expect_100_continue_with_body_completes_promptly_and_in_order(self):
        def handle(conn):
            _read_raw_request_headers(conn)
            payload = b'{"ok":true}'
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                b"Content-Length: " + str(len(payload)).encode("ascii") + b"\r\n\r\n" + payload
            )

        listener = self._start_upstream(handle)
        server = self._start_proxy(listener.getsockname()[1])

        sock = socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=5)
        sock.settimeout(5)
        body = b'{"x":1}'
        started = time.monotonic()
        sock.sendall(
            b"POST /v1/chat HTTP/1.1\r\n"
            b"Host: x\r\n"
            b"Expect: 100-continue\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n\r\n"
        )

        # Read exactly the interim status line (up to its terminating
        # blank line) BEFORE sending the body -- if the proxy never sends
        # 100 Continue, this recv blocks until the 5s socket timeout,
        # which the wall-clock assertion below would also catch, but the
        # explicit read-before-write here is the ordering proof itself:
        # the body is not written until the interim response is in hand.
        interim = b""
        while b"\r\n\r\n" not in interim:
            chunk = sock.recv(4096)
            if not chunk:
                break
            interim += chunk
        interim_elapsed = time.monotonic() - started
        self.assertIn(b"100 Continue", interim, "no 100 Continue interim response received")

        sock.sendall(body)

        response = b""
        while b"\r\n\r\n" not in response or not response.endswith(b'{"ok":true}'):
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk
        total_elapsed = time.monotonic() - started
        sock.close()

        self.assertLess(
            interim_elapsed, 0.5, "100 Continue interim response was not prompt"
        )
        self.assertLess(
            total_elapsed, 0.5, "request did not complete promptly (curl's own hang window is ~1s)"
        )
        self.assertIn(b"200 OK", response)
        self.assertIn(b'{"ok":true}', response)
        # The FINAL response on THIS SAME request/connection -- the one
        # that already went through the 100-continue handshake -- must
        # still be framed HTTP/1.0, exactly like a request with no Expect
        # header at all (see test_normal_request_without_expect_is_byte_
        # identical for the no-Expect case). Answering the interim
        # handshake must never leak into or alter the final response's own
        # status line.
        self.assertTrue(
            response.startswith(b"HTTP/1.0 200 OK\r\n"),
            f"final response framing changed after the 100-continue handshake: {response[:40]!r}",
        )

    # 2. Pin the exact interim status line this proxy emits. RFC 7231
    #    ties 100 Continue to HTTP/1.1 semantics (a client is only
    #    supposed to SEND Expect: 100-continue when it can handle an
    #    HTTP/1.1 response), so the interim line's version token
    #    intentionally reflects the CLIENT's own request version, not
    #    ``protocol_version`` (the pin that governs only the FINAL
    #    response's framing -- see the comment on ``handle_expect_100``
    #    below for why the two are independent).
    def test_interim_continue_line_is_pinned_http11(self):
        def handle(conn):
            _read_raw_request_headers(conn)
            payload = b"ok"
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(payload)).encode("ascii")
                + b"\r\n\r\n" + payload
            )

        listener = self._start_upstream(handle)
        server = self._start_proxy(listener.getsockname()[1])

        sock = socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=5)
        sock.settimeout(5)
        body = b"x"
        sock.sendall(
            b"POST /v1/chat HTTP/1.1\r\n"
            b"Host: x\r\n"
            b"Expect: 100-continue\r\n"
            b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n\r\n"
        )
        interim = b""
        while b"\r\n\r\n" not in interim:
            chunk = sock.recv(4096)
            if not chunk:
                break
            interim += chunk
        sock.sendall(body)
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
        sock.close()

        self.assertEqual(interim, b"HTTP/1.1 100 Continue\r\n\r\n")

    # 3. Anti-regression: a normal request WITHOUT Expect: is byte-
    #    identical to before this change -- same status line, same header
    #    set, same framing. This is the invariant guarding the deliberate
    #    HTTP/1.0 pin: the interim-response fix must be additive-only.
    def test_normal_request_without_expect_is_byte_identical(self):
        def responder(handler):
            payload = b'{"ok":true}'
            handler.send_response(201)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)

        upstream = start_fake_upstream(responder)
        self._servers.append(upstream)
        server = FASTMLX_PROXY.create_server(
            "127.0.0.1", 0, "127.0.0.1", upstream.server_address[1], FIXTURE_PLAN
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        server._test_thread = thread
        self._servers.append(server)

        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=b'{"x":1}',
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        body = resp.read()

        self.assertEqual(resp.status, 201)
        self.assertEqual(resp.version, 10)  # HTTP/1.0, exactly as before this change
        self.assertEqual(body, b'{"ok":true}')
        self.assertEqual(resp.getheader("Content-Type"), "application/json")
        self.assertEqual(resp.getheader("Content-Length"), str(len(body)))
        self.assertIsNotNone(resp.getheader("X-FastMLX-Admission"))
        # A second request on a NEW connection must see the exact same
        # framing/header shape -- proves the fix has no per-connection
        # state that could leak between requests.
        conn.close()
        conn2 = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        conn2.request("GET", "/v1/models")
        resp2 = conn2.getresponse()
        resp2.read()
        self.assertEqual(resp2.version, 10)
        conn2.close()

    # 4. Expect: 100-continue with NO body (zero-length) must still
    #    complete promptly, never hang waiting for a body that never
    #    comes.
    def test_expect_100_continue_with_no_body_does_not_hang(self):
        def handle(conn):
            _read_raw_request_headers(conn)
            payload = b"ok"
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(payload)).encode("ascii")
                + b"\r\n\r\n" + payload
            )

        listener = self._start_upstream(handle)
        server = self._start_proxy(listener.getsockname()[1])

        sock = socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=5)
        sock.settimeout(5)
        started = time.monotonic()
        sock.sendall(
            b"GET /v1/models HTTP/1.1\r\n"
            b"Host: x\r\n"
            b"Expect: 100-continue\r\n"
            b"Content-Length: 0\r\n\r\n"
        )
        response = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk
        elapsed = time.monotonic() - started
        sock.close()

        self.assertLess(elapsed, 0.5, "zero-length-body Expect: 100-continue must not hang")
        self.assertIn(b"100 Continue", response)
        self.assertIn(b"200 OK", response)
        self.assertIn(b"ok", response)

    # 5. RFC 7231 5.1.1: a server MUST NOT send a 100 (Continue) response
    #    to a request from an HTTP/1.0 (or earlier) client, even when that
    #    client sends an Expect header -- an HTTP/1.0 client is not
    #    required to understand an interim 1xx response and could mis-
    #    frame it as the final one. This is the CLIENT's own
    #    ``request_version`` (stdlib ``self.request_version``), a THIRD
    #    condition distinct from both the Expect-header check above and
    #    ``protocol_version`` (the SERVER's own deliberate HTTP/1.0 pin,
    #    see the comment on that assignment) -- see ``parse_request``'s
    #    docstring for why this override keeps this one while dropping
    #    ``protocol_version``.
    def test_http10_client_sending_expect_gets_no_interim_continue(self):
        def handle(conn):
            _read_raw_request_headers(conn)
            payload = b"ok"
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(payload)).encode("ascii")
                + b"\r\n\r\n" + payload
            )

        listener = self._start_upstream(handle)
        server = self._start_proxy(listener.getsockname()[1])

        sock = socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=5)
        sock.settimeout(5)
        body = b'{"x":1}'
        started = time.monotonic()
        # The full request, including body, in one write: an HTTP/1.0
        # client is not obligated to wait for any interim response before
        # sending its body, so this test never waits on one either -- the
        # bound below is what catches a hang if the fix regresses.
        sock.sendall(
            b"POST /v1/chat HTTP/1.0\r\n"
            b"Host: x\r\n"
            b"Expect: 100-continue\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n\r\n" + body
        )
        response = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk
        elapsed = time.monotonic() - started
        sock.close()

        self.assertLess(
            elapsed, 0.5,
            "an HTTP/1.0 client with Expect: 100-continue must not hang -- the "
            "absence of an interim response must not become a new hang",
        )
        self.assertNotIn(
            b"100 Continue", response,
            "RFC 7231 5.1.1 forbids a 100 (Continue) interim response to an "
            "HTTP/1.0 client",
        )
        self.assertIn(b"200 OK", response)
        self.assertIn(b"ok", response)


class ProxyLogAtomicityTests(unittest.TestCase):
    """L8: the JSONL log line is written with a SINGLE stderr.write call
    (then flushed) -- two concurrent requests' lines must never be able to
    interleave mid-line the way a multi-call print() risks under
    ThreadingHTTPServer.
    """

    def test_log_line_is_a_single_write_call(self):
        calls = []

        class FakeStream:
            def write(self, data):
                calls.append(data)

            def flush(self):
                pass

        with unittest.mock.patch.object(FASTMLX_PROXY.sys, "stderr", FakeStream()):
            FASTMLX_PROXY.ProvenanceProxyHandler._log(
                None, "req-1", "GET", "/x", 200, 1.0, 2, False, None
            )

        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].endswith("\n"))
        entry = json.loads(calls[0])
        self.assertEqual(entry["request_id"], "req-1")


class ProxySaturationSnapshotTests(unittest.TestCase):
    """``saturation_snapshot`` JSONL lines on the operator-only ``stderr``
    channel: the leading indicator of WHO holds the proxy's concurrency
    slots, not merely how many (see ``docs/task-inbox/2026-09-21-
    PREDECLARATION-proxy-saturation-snapshot-who-holds-the-slots.md``).

    ``log_hook`` carries these entries on the SAME channel ordinary
    ``_log``/``_log_capacity_refusal`` entries use -- every assertion below
    selects entries by ``entry.get("event") == "saturation_snapshot"``,
    NEVER by list position, exactly per that predeclaration's own warning.
    """

    def setUp(self):
        self._servers = []
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for server in self._servers:
            server.shutdown()
            server.server_close()
            server._test_thread.join(timeout=5)

    def _start_gated_upstream(self, expected_arrivals: int):
        """Same shape as ``ProxyConcurrencyLimitTests``' own helper (see
        its docstring there): every request blocks on a shared ``gate``
        until released, and ``arrived[n - 1]`` fires the instant the n-th
        request reaches upstream -- the barrier this class uses to prove
        N slots are provably held concurrently without any sleep-based
        polling.
        """
        requests = []
        lock = threading.Lock()
        gate = threading.Event()
        arrived = [threading.Event() for _ in range(expected_arrivals)]

        def responder(handler):
            with lock:
                requests.append(handler.path)
                n = len(requests)
            if 0 < n <= expected_arrivals:
                arrived[n - 1].set()
            self.assertTrue(gate.wait(timeout=10), "test never released the gate")
            payload = b"{}"
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)

        upstream = start_fake_upstream(responder)
        self._servers.append(upstream)
        return upstream, gate, arrived

    def _start_proxy(self, upstream, **kwargs):
        server = start_proxy(FIXTURE_PLAN, "127.0.0.1", upstream.server_address[1], **kwargs)
        self._servers.append(server)
        return server

    def _drive_request(self, port: int) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
        conn.request("POST", "/v1/chat/completions", body=b"{}")
        resp = conn.getresponse()
        resp.read()
        conn.close()

    def _send_and_read(self, port: int, raw_request: bytes, timeout: float = 5) -> bytes:
        sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        sock.sendall(raw_request)
        sock.settimeout(timeout)
        response = b""
        try:
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
        except socket.timeout:
            pass
        sock.close()
        return response

    @staticmethod
    def _patched_get_request(hosts):
        """Context manager: for its duration, every connection THIS
        process's proxy servers accept reports a ``client_address`` host
        drawn from ``hosts`` (cycled) instead of the real loopback source
        address every real test connection actually shares -- the
        source-side seam this class uses to fabricate identical or
        distinct peer hosts deterministically, without any reliance on
        real routing or multiple source machines.
        """
        real_get_request = FASTMLX_PROXY.ProvenanceProxyServer.get_request
        hosts_iter = itertools.cycle(hosts)

        def fake_get_request(self_server):
            request, real_client_address = real_get_request(self_server)
            host = next(hosts_iter)
            return request, (host, real_client_address[1])

        return unittest.mock.patch.object(
            FASTMLX_PROXY.ProvenanceProxyServer, "get_request", new=fake_get_request
        )

    def _hold_n_slots_sequentially_and_capture_first_snapshot(self, n: int, hosts):
        """Admits ``n`` connections ONE AT A TIME -- starting connection
        ``i + 1`` only after ``arrived[i]`` fires -- so the accept order
        (and therefore which ``hosts[i]`` each slot is tagged with) is
        deterministic, then waits for a ``saturation_snapshot`` line and
        returns it (plus cleanup handles) while every slot is still held.
        """
        upstream, gate, arrived = self._start_gated_upstream(expected_arrivals=n)

        snapshots = []
        snapshot_seen = threading.Event()

        def log_hook(entry):
            if entry.get("event") == "saturation_snapshot":
                snapshots.append(entry)
                snapshot_seen.set()

        proxy = self._start_proxy(upstream, max_concurrent_requests=n, log_hook=log_hook)

        threads = []
        with self._patched_get_request(hosts):
            for i in range(n):
                t = threading.Thread(target=self._drive_request, args=(proxy.server_address[1],))
                t.start()
                threads.append(t)
                self.assertTrue(arrived[i].wait(timeout=5), f"request {i} never reached upstream")

            self.assertTrue(
                snapshot_seen.wait(timeout=5),
                "no saturation_snapshot line was emitted while the cap was held",
            )

        return snapshots[0], gate, threads

    # ------------------------------------------------------------------
    # Criterion 1 (the anti-vacuity bar): at IDENTICAL ``inflight``, arm A
    # (N slots, ONE peer host) must report ``distinct_peer_hosts == 1``
    # and arm B (N slots, N distinct hosts) must report ``== N``. Merely
    # asserting the field's PRESENCE is refused by the task predeclaration
    # as vacuous -- this is the two-arm discrimination it requires
    # instead.
    # ------------------------------------------------------------------
    def test_distinct_peer_hosts_discriminates_one_host_from_many_at_identical_inflight(self):
        n = 4

        entry_a, gate_a, threads_a = self._hold_n_slots_sequentially_and_capture_first_snapshot(
            n, ["10.0.0.1"]
        )
        self.assertEqual(entry_a["inflight"], n)
        self.assertEqual(
            entry_a["distinct_peer_hosts"], 1,
            "N slots from ONE peer host must report distinct_peer_hosts == 1",
        )
        gate_a.set()
        for t in threads_a:
            t.join(timeout=5)

        entry_b, gate_b, threads_b = self._hold_n_slots_sequentially_and_capture_first_snapshot(
            n, [f"10.0.0.{i}" for i in range(1, n + 1)]
        )
        self.assertEqual(entry_b["inflight"], n)
        self.assertEqual(
            entry_b["distinct_peer_hosts"], n,
            "N slots from N distinct peer hosts must report distinct_peer_hosts == N",
        )
        gate_b.set()
        for t in threads_b:
            t.join(timeout=5)

        # The point of the field: identical `inflight`, different shape,
        # different reading.
        self.assertEqual(entry_a["inflight"], entry_b["inflight"])
        self.assertNotEqual(entry_a["distinct_peer_hosts"], entry_b["distinct_peer_hosts"])

    # ------------------------------------------------------------------
    # Criterion 2: an intentionally-aged slot reports ``age_s >= hold`` in
    # the SAME snapshot where a just-acquired slot reports ``age_s <
    # hold``. A real (bounded, generous-margin) wall-clock gap between the
    # two admissions -- the same idiom this suite already uses elsewhere
    # (e.g. ``time.sleep(1.5)`` against a scaled-down deadline constant)
    # -- rather than patching ``time.monotonic`` globally, which this
    # module also calls from unrelated request-timeout/deadline code paths
    # a global patch would silently perturb.
    # ------------------------------------------------------------------
    def test_age_discriminates_an_aged_slot_from_a_fresh_one_in_the_same_snapshot(self):
        hold = 0.05
        upstream, gate, arrived = self._start_gated_upstream(expected_arrivals=2)

        snapshots = []
        snapshot_seen = threading.Event()

        def log_hook(entry):
            if entry.get("event") == "saturation_snapshot":
                snapshots.append(entry)
                snapshot_seen.set()

        proxy = self._start_proxy(upstream, max_concurrent_requests=2, log_hook=log_hook)

        with self._patched_get_request(["10.9.0.1"]):
            t1 = threading.Thread(target=self._drive_request, args=(proxy.server_address[1],))
            t1.start()
            self.assertTrue(arrived[0].wait(timeout=5), "the aged request never reached upstream")

        time.sleep(hold * 4)  # generous margin past `hold`

        with self._patched_get_request(["10.9.0.2"]):
            t2 = threading.Thread(target=self._drive_request, args=(proxy.server_address[1],))
            t2.start()
            self.assertTrue(arrived[1].wait(timeout=5), "the fresh request never reached upstream")

        self.assertTrue(
            snapshot_seen.wait(timeout=5), "no saturation_snapshot line was emitted"
        )

        entry = snapshots[0]
        by_peer = {slot["peer"]: slot["age_s"] for slot in entry["top_slots"]}
        self.assertGreaterEqual(
            by_peer["10.9.0.1"], hold, "the aged slot must report age_s >= hold"
        )
        self.assertLess(
            by_peer["10.9.0.2"], hold, "the just-acquired slot must report age_s < hold"
        )

        gate.set()
        t1.join(timeout=5)
        t2.join(timeout=5)

    # ------------------------------------------------------------------
    # Criterion 3: with the cap at K + 20 and every slot held,
    # ``len(top_slots) <= K`` and the reported slots are the K OLDEST, not
    # merely any K -- proving both the O(1) bound and that "oldest" (not
    # "newest" or "arbitrary") is what survives the cut (kills mutation
    # M2: selecting the K NEWEST instead).
    # ------------------------------------------------------------------
    def test_top_k_bound_holds_regardless_of_the_configured_cap_and_reports_the_oldest(self):
        k = FASTMLX_PROXY._SATURATION_SNAPSHOT_TOP_K
        n = k + 20
        upstream, gate, arrived = self._start_gated_upstream(expected_arrivals=n)

        lock = threading.Lock()
        snapshots = []

        def log_hook(entry):
            if entry.get("event") == "saturation_snapshot":
                with lock:
                    snapshots.append(entry)

        proxy = self._start_proxy(upstream, max_concurrent_requests=n, log_hook=log_hook)

        hosts = [f"10.7.0.{i}" for i in range(1, n + 1)]
        threads = []
        with unittest.mock.patch.object(
            FASTMLX_PROXY, "_SATURATION_SNAPSHOT_MIN_INTERVAL_SECONDS", 0.0
        ), self._patched_get_request(hosts):
            for i in range(n):
                t = threading.Thread(target=self._drive_request, args=(proxy.server_address[1],))
                t.start()
                threads.append(t)
                self.assertTrue(arrived[i].wait(timeout=5), f"request {i} never reached upstream")

            deadline = time.monotonic() + 5
            full_snapshot = None
            while full_snapshot is None and time.monotonic() < deadline:
                with lock:
                    for entry in snapshots:
                        if entry["inflight"] == n:
                            full_snapshot = entry
                            break
                if full_snapshot is None:
                    time.sleep(0.02)

        self.assertIsNotNone(
            full_snapshot, f"never observed a saturation_snapshot with inflight == {n}"
        )
        self.assertLessEqual(len(full_snapshot["top_slots"]), k)
        self.assertEqual(full_snapshot["slots_reported"], len(full_snapshot["top_slots"]))

        reported_peers = {slot["peer"] for slot in full_snapshot["top_slots"]}
        # Connections were admitted strictly one at a time (each started
        # only after the PREVIOUS one's arrival fired -- see the loop
        # above), so ``hosts[0]`` is the OLDEST slot and ``hosts[:k]`` are
        # exactly the K oldest.
        oldest_expected = set(hosts[:k])
        self.assertEqual(
            reported_peers, oldest_expected,
            "top_slots must report the K OLDEST slots, not any K of them",
        )

        gate.set()
        for t in threads:
            t.join(timeout=5)

    # ------------------------------------------------------------------
    # Criterion 5: after a forced thread-start failure (the third
    # registry site -- see ``process_request``'s own ``except
    # BaseException`` unwind), the registry must be EMPTY and
    # ``len(registry) == inflight_requests`` -- no phantom entry. Reuses
    # the same induction technique as ``ProxyConcurrencyLimitTests.
    # test_failed_handler_thread_start_does_not_leak_its_slot``.
    # ------------------------------------------------------------------
    def test_registry_has_no_phantom_entry_after_a_forced_thread_start_failure(self):
        def responder(handler):
            payload = b"{}"
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)

        upstream = start_fake_upstream(responder)
        self._servers.append(upstream)
        proxy = self._start_proxy(upstream, max_concurrent_requests=2)

        handled = threading.Event()

        def record_handle_error(request, client_address):
            handled.set()

        proxy.handle_error = record_handle_error

        def flaky_process_request(self_server, request, client_address):
            raise RuntimeError("can't start new thread")

        with unittest.mock.patch.object(
            socketserver.ThreadingMixIn, "process_request", new=flaky_process_request
        ):
            conn = socket.create_connection(("127.0.0.1", proxy.server_address[1]), timeout=5)
            self.assertTrue(
                handled.wait(timeout=5),
                "the induced thread-start failure never reached handle_error",
            )
            conn.close()

        self.assertEqual(
            proxy.inflight_requests, 0,
            "inflight counter leaked after the induced thread-start failure",
        )
        self.assertEqual(
            len(proxy._slot_registry), 0,
            "a phantom slot-registry entry survived the induced thread-start failure",
        )
        self.assertEqual(len(proxy._slot_registry), proxy.inflight_requests)
        self.assertEqual(
            len(proxy._pending_slot_by_request), 0,
            "the transient request-to-slot correlation map was not cleaned up",
        )

    # ------------------------------------------------------------------
    # Criterion 9 / M4: driving many accepts inside one rate-limit
    # interval must yield AT MOST ONE ``saturation_snapshot`` line -- a
    # mutation that emits on every threshold-crossing accept instead of
    # respecting the rate limit turns this RED.
    # ------------------------------------------------------------------
    def test_rate_limit_allows_at_most_one_line_per_interval(self):
        n = 6
        with unittest.mock.patch.object(
            FASTMLX_PROXY, "_SATURATION_SNAPSHOT_MIN_INTERVAL_SECONDS", 10.0
        ):
            upstream, gate, arrived = self._start_gated_upstream(expected_arrivals=n)
            emitted = []
            lock = threading.Lock()

            def log_hook(entry):
                if entry.get("event") == "saturation_snapshot":
                    with lock:
                        emitted.append(entry)

            proxy = self._start_proxy(upstream, max_concurrent_requests=n, log_hook=log_hook)

            threads = []
            for _ in range(n):
                t = threading.Thread(target=self._drive_request, args=(proxy.server_address[1],))
                t.start()
                threads.append(t)
            for i in range(n):
                self.assertTrue(arrived[i].wait(timeout=5), f"request {i} never reached upstream")

            # Bounded window for the reporter thread to process every
            # trigger the accept thread could possibly have queued while
            # all ``n`` connections were being admitted.
            time.sleep(0.3)

            gate.set()
            for t in threads:
                t.join(timeout=5)

        with lock:
            count = len(emitted)
        self.assertLessEqual(
            count, 1,
            f"expected at most one saturation_snapshot line inside the rate-limit "
            f"interval, got {count}",
        )

    # ------------------------------------------------------------------
    # Criterion 4/14: no peer, age, slot, or counter field from this
    # feature may reach an unauthenticated client -- neither the
    # provenance body nor the 503 refusal body/headers gain any new key.
    # ------------------------------------------------------------------
    def test_client_visible_surfaces_carry_no_new_saturation_fields(self):
        upstream, gate, arrived = self._start_gated_upstream(expected_arrivals=1)
        proxy = self._start_proxy(upstream, max_concurrent_requests=1)

        conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=5)
        conn.request("GET", "/fastmlx/provenance")
        resp = conn.getresponse()
        body_bytes = resp.read()
        conn.close()
        body = json.loads(body_bytes)
        self.assertEqual(
            set(body.keys()),
            {"fit", "card", "admission", "residency", "engineBuild", "mtp", "front"},
            "the provenance body's key set must stay byte-unchanged by this increment",
        )
        for forbidden in (
            "peer", "top_slots", "distinct_peer_hosts", "slots_reported",
            "max_age_s", "saturation_snapshot",
        ):
            self.assertNotIn(forbidden, body_bytes.decode("utf-8"))

        holder = threading.Thread(target=self._drive_request, args=(proxy.server_address[1],))
        holder.start()
        self.assertTrue(arrived[0].wait(timeout=5), "the holder request never reached upstream")

        raw_request = b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}"
        response = self._send_and_read(proxy.server_address[1], raw_request)
        self.assertIn(b" 503 ", response)
        header_bytes, _, body_bytes = response.partition(b"\r\n\r\n")
        body_json = json.loads(body_bytes)
        self.assertEqual(set(body_json.keys()), {"error"})
        self.assertEqual(set(body_json["error"].keys()), {"type", "message"})
        for forbidden in (
            b"peer", b"top_slots", b"distinct_peer_hosts", b"slots_reported", b"max_age_s",
        ):
            self.assertNotIn(forbidden, response)

        gate.set()
        holder.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
