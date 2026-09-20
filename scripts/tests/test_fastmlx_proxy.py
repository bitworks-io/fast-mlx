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
import json
import os
import socket
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

        for i in range(3):  # cap (2) + 1
            conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=5)
            conn.request("POST", "/v1/chat/completions", body=b"{}")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            self.assertEqual(resp.status, 200, f"sequential request {i} after release must be admitted")

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
    # Accept-loop liveness: a hostile over-cap client that triggers the
    # refusal and then never reads its response must not delay a
    # concurrent, well-behaved connection's own (503) response -- proves
    # the ``settimeout``/``except OSError`` around the refusal write
    # keeps the single accept thread from wedging behind it.
    # ------------------------------------------------------------------
    def test_hostile_client_that_never_reads_its_refusal_does_not_wedge_the_accept_loop(self):
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
        # recv() at all -- if the refusal write ever blocked here without
        # a timeout, every OTHER connection (including the well-behaved
        # one below) would wedge behind it on the server's single accept
        # thread.
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


if __name__ == "__main__":
    unittest.main()
