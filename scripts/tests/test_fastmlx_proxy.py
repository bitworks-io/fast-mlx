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
