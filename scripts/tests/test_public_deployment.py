from __future__ import annotations

import copy
import io
import json
import os
import re
import stat
import sys
import tempfile
import threading
import unittest
import warnings
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPOSITORY_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import validate_public_deployment_receipt as offline
import verify_public_deployment as online
import build_public_site


class PublicDeploymentReceiptContractTests(unittest.TestCase):
    @staticmethod
    def workflow() -> str:
        return (REPOSITORY_ROOT / ".github/workflows/pages.yml").read_text(
            encoding="utf-8"
        )

    @classmethod
    def job_block(cls, name: str) -> str:
        marker = f"\n  {name}:\n"
        workflow = cls.workflow()
        start = workflow.index(marker) + 1
        remainder = workflow[start:]
        next_job = re.search(
            r"\n  [A-Za-z0-9_-]+:\n", remainder[len(f"  {name}:\n") :]
        )
        if next_job is None:
            return remainder
        end = len(f"  {name}:\n") + next_job.start()
        return remainder[:end]

    def test_online_verifier_script_exists(self) -> None:
        self.assertTrue(
            (REPOSITORY_ROOT / "scripts/verify_public_deployment.py").is_file()
        )

    def test_offline_receipt_validator_script_exists(self) -> None:
        self.assertTrue(
            (
                REPOSITORY_ROOT
                / "scripts/validate_public_deployment_receipt.py"
            ).is_file()
        )

    def test_pages_workflow_deploy_exposes_exact_page_url(self) -> None:
        deploy = self.job_block("deploy")

        self.assertIn(
            "outputs:\n      page_url: ${{ steps.deployment.outputs.page_url }}",
            deploy,
        )
        self.assertIn("id: deployment", deploy)
        self.assertIn("needs: build", deploy)

    def test_pages_workflow_verifier_job_is_exact_sha_and_read_only(self) -> None:
        verifier = self.job_block("verify-publication")

        self.assertIn(
            "if: github.event_name == 'push' && github.ref == 'refs/heads/main'",
            verifier,
        )
        self.assertIn("needs: deploy", verifier)
        self.assertIn("permissions:\n      contents: read", verifier)
        self.assertNotIn("pages: write", verifier)
        self.assertNotIn("id-token: write", verifier)
        self.assertIn("runs-on: ubuntu-latest", verifier)
        self.assertIn("timeout-minutes: 15", verifier)
        self.assertIn(
            "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803 # v6",
            verifier,
        )
        self.assertIn("ref: ${{ github.sha }}", verifier)
        self.assertIn("fetch-depth: 1", verifier)
        self.assertIn("persist-credentials: false", verifier)
        self.assertIn(
            "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1 # v6",
            verifier,
        )
        self.assertIn('python-version: "3.13"', verifier)

    def test_pages_workflow_rebuilds_and_verifies_exact_subject(self) -> None:
        verifier = self.job_block("verify-publication")

        self.assertIn("python3 scripts/build_public_site.py", verifier)
        self.assertIn(
            '--output "$RUNNER_TEMP/fast-mlx-receipt-site"', verifier
        )
        self.assertIn("python3 scripts/validate_public_site.py", verifier)
        self.assertGreaterEqual(
            verifier.count('"$RUNNER_TEMP/fast-mlx-receipt-site"'), 2
        )
        self.assertIn("id: publication_verifier", verifier)
        self.assertIn(
            "DEPLOYMENT_URL: ${{ needs.deploy.outputs.page_url }}", verifier
        )
        self.assertIn("python3 scripts/verify_public_deployment.py", verifier)
        for argument in (
            '--repository-root "$GITHUB_WORKSPACE"',
            '--site "$RUNNER_TEMP/fast-mlx-receipt-site"',
            '--deployment-url "$DEPLOYMENT_URL"',
            '--commit-sha "$GITHUB_SHA"',
            '--workflow-run-id "$GITHUB_RUN_ID"',
            '--workflow-run-attempt "$GITHUB_RUN_ATTEMPT"',
            '--output "$RUNNER_TEMP/fast-mlx-publication-receipt.json"',
        ):
            with self.subTest(argument=argument):
                self.assertIn(argument, verifier)

    def test_pages_workflow_validates_exact_outcome_before_upload(self) -> None:
        verifier = self.job_block("verify-publication")

        self.assertIn(
            "if: ${{ !cancelled() && steps.publication_verifier.outcome == 'success' }}",
            verifier,
        )
        self.assertIn(
            "if: ${{ !cancelled() && steps.publication_verifier.outcome == 'failure' }}",
            verifier,
        )
        self.assertEqual(
            verifier.count("python3 scripts/validate_public_deployment_receipt.py"),
            2,
        )
        self.assertEqual(
            verifier.count("echo 'validated=true' >> \"$GITHUB_OUTPUT\""), 2
        )
        self.assertIn("--require-result PASS", verifier)
        self.assertIn("--require-result FAIL", verifier)
        self.assertIn(
            "steps.pass_receipt_validation.outputs.validated == 'true'", verifier
        )
        self.assertIn(
            "steps.fail_receipt_validation.outputs.validated == 'true'", verifier
        )

    def test_pages_workflow_upload_is_pinned_bounded_and_non_overwriting(self) -> None:
        verifier = self.job_block("verify-publication")

        self.assertIn(
            "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1",
            verifier,
        )
        self.assertIn(
            "name: public-deployment-receipt-${{ github.sha }}-${{ github.run_id }}-${{ github.run_attempt }}",
            verifier,
        )
        self.assertIn(
            "path: ${{ runner.temp }}/fast-mlx-publication-receipt.json", verifier
        )
        for option in (
            "if-no-files-found: error",
            "retention-days: 30",
            "compression-level: 9",
            "overwrite: false",
            "include-hidden-files: false",
            "archive: true",
        ):
            with self.subTest(option=option):
                self.assertIn(option, verifier)

        workflow = self.workflow()
        self.assertNotIn("continue-on-error", workflow)
        self.assertNotIn("pull_request_target", workflow)
        self.assertNotIn("matrix:", verifier)


class _FakeHeaders:
    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        self._pairs = pairs

    def get_all(self, name: str, default: list[str] | None = None) -> list[str]:
        values = [
            value
            for key, value in self._pairs
            if key.lower() == name.lower()
        ]
        return values if values else ([] if default is None else default)


class _FakeResponse:
    def __init__(
        self,
        *,
        status: int,
        headers: list[tuple[str, str]],
        body: bytes,
        read_error: BaseException | None = None,
    ) -> None:
        self.status = status
        self.headers = _FakeHeaders(headers)
        self.body = body
        self.read_error = read_error
        self.read_calls = 0
        self.read_limits: list[int] = []

    def read(self, limit: int) -> bytes:
        self.read_calls += 1
        self.read_limits.append(limit)
        if self.read_error is not None:
            raise self.read_error
        return self.body


class _ManualClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += seconds


class PublicDeploymentBehaviorTests(unittest.TestCase):
    commit_sha = "0123456789abcdef0123456789abcdef01234567"
    run_id = 101
    run_attempt = 2

    def make_route(
        self,
        body: bytes = b"hello",
        *,
        source: str = "index.html",
        path: str = "/",
        request_target: str = "/fast-mlx/",
        status_code: int = 200,
        content_type: str = "text/html; charset=utf-8",
    ) -> online.RouteSubject:
        kwargs: dict[str, object] = {
            "source": source,
            "path": path,
            "request_target": request_target,
            "expected_status": status_code,
            "expected_content_type": content_type,
            "expected_bytes": len(body),
            "expected_sha256": online.sha256_hex(body),
        }
        if "expected_body" in online.RouteSubject._fields:
            kwargs["expected_body"] = body
        else:
            online._EXPECTED_BODIES_BY_SOURCE = {source: body}
        return online.RouteSubject(**kwargs)

    def make_site(self, root: Path) -> Path:
        site = root / "site"
        (site / "assets").mkdir(parents=True)
        (site / "guide").mkdir()
        (site / ".nojekyll").write_bytes(b"")
        (site / "index.html").write_bytes(b"<!doctype html>\n<title>fast-mlx</title>\n")
        (site / "404.html").write_bytes(b"not found\n")
        (site / "assets" / "app.js").write_bytes(b"console.log('ok');\n")
        (site / "assets" / "style.css").write_bytes(b"body{color:#111}\n")
        (site / "guide" / "index.html").write_bytes(b"<h1>Guide</h1>\n")
        return site

    def build_pass_receipt(
        self, site: Path
    ) -> tuple[dict[str, object], bytes, online.Inventory]:
        inventory = online.inventory_site(site, self.commit_sha)
        local = online.LocalInputs(
            repository_root=REPOSITORY_ROOT,
            site=site,
            output=site.parent / "receipt-outside.json",
            commit_sha=self.commit_sha,
            workflow_run_id=self.run_id,
            workflow_run_attempt=self.run_attempt,
        )
        subject = online.build_subject(local, inventory, online.verifier_source_sha256())
        workflow = {"run_attempt": self.run_attempt, "run_id": self.run_id}
        accepted_routes = [route.accepted_object() for route in inventory.routes]
        cohort = online.cohort_sha256(accepted_routes)
        observation = {
            "accepted_cohort_sha256": cohort,
            "attempted_cohorts": 2,
            "attempts": [
                {
                    "attempt": 1,
                    "cohort_sha256": cohort,
                    "completed_routes": len(accepted_routes),
                    "failure": None,
                    "result": "MATCH",
                },
                {
                    "attempt": 2,
                    "cohort_sha256": cohort,
                    "completed_routes": len(accepted_routes),
                    "failure": None,
                    "result": "MATCH",
                },
            ],
            "consecutive_matching_cohorts": online.STABLE_COHORTS_REQUIRED,
            "failures": [],
            "routes": accepted_routes,
        }
        receipt = online.build_receipt(
            result="PASS",
            subject=subject,
            workflow=workflow,
            observation=observation,
        )
        return receipt, online.receipt_json_bytes(receipt), inventory

    @staticmethod
    def build_fail_receipt(
        pass_receipt: dict[str, object],
        failure: dict[str, object],
        *,
        completed_routes: int = 0,
    ) -> dict[str, object]:
        receipt = copy.deepcopy(pass_receipt)
        receipt["result"] = "FAIL"
        receipt["comparison_sha256"] = None
        receipt["observation"] = {
            "accepted_cohort_sha256": None,
            "attempted_cohorts": 1,
            "attempts": [
                {
                    "attempt": 1,
                    "cohort_sha256": None,
                    "completed_routes": completed_routes,
                    "failure": failure,
                    "result": "FAIL",
                }
            ],
            "consecutive_matching_cohorts": 0,
            "failures": [failure],
            "routes": [],
        }
        return receipt

    def assert_offline_validation_refuses(
        self,
        receipt: dict[str, object],
        site: Path,
        expected_fragment: str,
        *,
        require_result: str = "PASS",
    ) -> None:
        subject = offline.build_local_subject(REPOSITORY_ROOT, site, self.commit_sha)
        with self.assertRaisesRegex(offline.ValidationError, expected_fragment):
            offline.validate_receipt(
                receipt,
                subject,
                self.commit_sha,
                self.run_id,
                self.run_attempt,
                require_result,
            )

    def test_route_subject_and_hashes_are_deterministic_across_online_and_offline(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            site = self.make_site(Path(raw))

            first = online.inventory_site(site, self.commit_sha)
            second = online.inventory_site(site, self.commit_sha)
            offline_first = offline.build_local_subject(
                REPOSITORY_ROOT, site, self.commit_sha
            )
            offline_second = offline.build_local_subject(
                REPOSITORY_ROOT, site, self.commit_sha
            )

            self.assertEqual(first, second)
            self.assertEqual(
                [route.manifest_object() for route in first.routes],
                offline_first.route_subject,
            )
            self.assertEqual(offline_first, offline_second)
            self.assertEqual(first.site_ordered_sha256, offline_first.site_ordered_sha256)
            self.assertEqual(
                first.route_manifest_sha256,
                offline_first.route_manifest_sha256,
            )
            self.assertEqual(
                [route.accepted_object() for route in first.routes],
                offline_first.accepted_routes,
            )
            self.assertEqual(
                {
                    "404.html": f"/deployment-verification-not-found-{self.commit_sha[:13]}/",
                    "assets/app.js": "/assets/app.js",
                    "assets/style.css": "/assets/style.css",
                    "guide/index.html": "/guide/",
                    "index.html": "/",
                },
                {route.source: route.path for route in first.routes},
            )

    def test_closed_mime_authority_policy_failure_and_path_sets(self) -> None:
        expected_mime = {
            ".atom": "application/atom+xml",
            ".css": "text/css; charset=utf-8",
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".png": "image/png",
            ".svg": "image/svg+xml",
            ".txt": "text/plain; charset=utf-8",
            ".xml": "application/xml",
        }
        expected_authority = {
            "acquisition_authority",
            "automatic_evidence_intake",
            "automatic_publication_authority",
            "containment_claim",
            "launchability_claim",
            "model_authority",
            "performance_claim",
            "positive_admission",
            "rollback_authority",
            "runtime_authority",
        }
        expected_failures = {
            "body_too_large",
            "byte_count_mismatch",
            "byte_mismatch",
            "cohort_attempts_exhausted",
            "connect_error",
            "connect_timeout",
            "content_encoding_refused",
            "content_length_refused",
            "content_type_refused",
            "deadline_exhausted",
            "dns_policy_refused",
            "http_status_mismatch",
            "internal_error",
            "read_timeout",
            "resolver_error",
            "resolver_timeout",
            "response_protocol_error",
            "tls_error",
            "transfer_encoding_refused",
        }

        self.assertEqual(online.MIME_BY_SUFFIX, expected_mime)
        self.assertEqual(offline.MIME_TYPES, expected_mime)
        self.assertEqual(set(online.AUTHORITY), expected_authority)
        self.assertEqual(offline.AUTHORITY_FLAGS, expected_authority)
        self.assertTrue(all(value is False for value in online.AUTHORITY.values()))
        self.assertEqual(online.POLICY, offline.POLICY)
        self.assertEqual(online.ALLOWED_FAILURE_CODES, expected_failures)
        self.assertEqual(offline.FAILURE_CODES, expected_failures)
        self.assertEqual(online.ACCEPTED_BASE_URL, offline.BASE_URL)
        self.assertEqual(online.ACCEPTED_HOST, "bitworks-io.github.io")
        self.assertEqual(online.BASE_PATH, offline.BASE_PATH)

        refused_sources = (
            "/index.html",
            "dir/",
            "dir//index.html",
            "dir/../index.html",
            "dir/./index.html",
            "dir\\index.html",
            "dir%2Findex.html",
            "dir?x=index.html",
            "dir#index.html",
            "dir/\x7f.html",
            "dir/é.html",
        )
        for source in refused_sources:
            with self.subTest(source=source):
                with self.assertRaises(Exception):
                    online._validate_relative_source(source)
                with self.assertRaises(Exception):
                    offline.validate_relative_source(source)

    def test_global_unicast_policy_refuses_private_mapped_6to4_and_teredo(self) -> None:
        allowed = ("8.8.8.8", "2001:4860:4860::8888")
        refused = (
            "127.0.0.1",
            "10.0.0.1",
            "192" + ".168.1.1",
            "169.254.1.1",
            "224.0.0.1",
            "255.255.255.255",
            "::1",
            "fd00::1",
            "fe80::1",
            "::ffff:8.8.8.8",
            "2002:0808:0808::1",
            "2001:0000:4136:e378:8000:63bf:3fff:fdd2",
        )
        for address in allowed:
            with self.subTest(address=address):
                self.assertTrue(online.is_allowed_global_unicast(address))
        for address in refused:
            with self.subTest(address=address):
                self.assertFalse(online.is_allowed_global_unicast(address))
                resolver = lambda _host, _port, address=address: [address]
                addresses, code = online.resolve_cohort_addresses(
                    resolver=resolver,
                    deadline=10.0,
                    clock=lambda: 0.0,
                )
                self.assertIsNone(addresses)
                self.assertEqual(code, "dns_policy_refused")

    def test_header_status_and_body_refusal_ordering_is_fail_closed(self) -> None:
        route = self.make_route(b"abc")
        ok_headers = [
            ("Content-Type", route.expected_content_type),
            ("Content-Length", str(route.expected_bytes)),
        ]

        response = _FakeResponse(
            status=500,
            headers=[("Content-Type", "text/plain"), ("Content-Length", "999")],
            body=b"too long",
        )
        result = online.evaluate_http_response(route, response)
        self.assertEqual(result.failure["code"], "http_status_mismatch")
        self.assertEqual(result.failure["status"], 500)
        self.assertEqual(response.read_calls, 0)

        response = _FakeResponse(
            status=999,
            headers=[
                ("Content-Type", route.expected_content_type),
                ("Content-Length", "3"),
            ],
            body=b"abc",
        )
        result = online.evaluate_http_response(route, response)
        self.assertEqual(result.failure["code"], "response_protocol_error")
        self.assertIsNone(result.failure["status"])
        self.assertEqual(response.read_calls, 0)

        response = _FakeResponse(
            status=200,
            headers=[("Content-Type", "text/plain"), ("Content-Length", "999")],
            body=b"too long",
        )
        result = online.evaluate_http_response(route, response)
        self.assertEqual(result.failure["code"], "content_type_refused")
        self.assertEqual(response.read_calls, 0)

        response = _FakeResponse(
            status=200,
            headers=[
                ("Content-Type", route.expected_content_type),
                ("Content-Encoding", "gzip"),
                ("Transfer-Encoding", "chunked"),
                ("Content-Length", str(route.expected_bytes)),
            ],
            body=b"abc",
        )
        result = online.evaluate_http_response(route, response)
        self.assertEqual(result.failure["code"], "content_encoding_refused")
        self.assertEqual(response.read_calls, 0)

        response = _FakeResponse(
            status=200,
            headers=[
                ("Content-Type", route.expected_content_type),
                ("Content-Encoding", "identity"),
                ("Transfer-Encoding", "chunked"),
                ("Content-Length", str(route.expected_bytes)),
            ],
            body=b"abc",
        )
        result = online.evaluate_http_response(route, response)
        self.assertEqual(result.failure["code"], "transfer_encoding_refused")
        self.assertEqual(response.read_calls, 0)

        response = _FakeResponse(
            status=200,
            headers=[
                ("Content-Type", route.expected_content_type),
                ("Content-Length", "0003"),
            ],
            body=b"abcd",
        )
        result = online.evaluate_http_response(route, response)
        self.assertEqual(result.failure["code"], "content_length_refused")
        self.assertEqual(response.read_calls, 0)

        response = _FakeResponse(status=200, headers=ok_headers, body=b"abcd")
        result = online.evaluate_http_response(route, response)
        self.assertEqual(result.failure["code"], "body_too_large")
        self.assertEqual(result.failure["observed_bytes"], 4)
        self.assertEqual(response.read_limits, [4])

        response = _FakeResponse(status=200, headers=ok_headers, body=b"ab")
        result = online.evaluate_http_response(route, response)
        self.assertEqual(result.failure["code"], "byte_count_mismatch")
        self.assertEqual(result.failure["observed_bytes"], 2)

        response = _FakeResponse(status=200, headers=ok_headers, body=b"abd")
        result = online.evaluate_http_response(route, response)
        self.assertEqual(result.failure["code"], "byte_mismatch")
        self.assertEqual(result.failure["observed_bytes"], 3)

        response = _FakeResponse(status=200, headers=ok_headers, body=b"abc")
        result = online.evaluate_http_response(route, response)
        self.assertIsNone(result.failure)
        self.assertEqual(result.record, route.accepted_object())

    def test_fake_resolver_transport_clock_cohort_pass_reset_attempt_and_deadline(self) -> None:
        route = self.make_route(b"abc")
        clock = _ManualClock()
        resolver_calls: list[tuple[str, int]] = []
        transport_calls: list[int] = []

        def resolver(host: str, port: int) -> list[str]:
            resolver_calls.append((host, port))
            return ["8.8.8.8"]

        def transport(
            route_subject: online.RouteSubject,
            addresses: tuple[str, ...],
            _deadline: float,
            _clock: object,
        ) -> online.RouteFetch:
            self.assertEqual(addresses, ("8.8.8.8",))
            transport_calls.append(len(transport_calls) + 1)
            if len(transport_calls) == 1:
                return online.RouteFetch(
                    None,
                    online.make_failure("content_type_refused", route_subject.path, status=200),
                )
            return online.RouteFetch(route_subject.accepted_object(), None)

        observation = online.observe_routes(
            [route],
            resolver=resolver,
            transport=transport,
            clock=clock,
            sleeper=clock.sleep,
        )

        self.assertEqual(resolver_calls, [(online.ACCEPTED_HOST, online.HTTPS_PORT)] * 3)
        self.assertEqual(transport_calls, [1, 2, 3])
        self.assertEqual(clock.sleeps, [online.RETRY_DELAY_SECONDS])
        self.assertEqual(observation["attempted_cohorts"], 3)
        self.assertEqual(
            [attempt["result"] for attempt in observation["attempts"]],
            ["FAIL", "MATCH", "MATCH"],
        )
        self.assertEqual(
            observation["consecutive_matching_cohorts"],
            online.STABLE_COHORTS_REQUIRED,
        )
        self.assertIsNotNone(observation["accepted_cohort_sha256"])
        self.assertEqual(observation["failures"], [])

        deadline_clock = _ManualClock(value=0.0)

        def deadline_sleeper(seconds: float) -> None:
            deadline_clock.sleeps.append(seconds)
            deadline_clock.value = online.MAX_OBSERVATION_SECONDS + 1

        deadline_observation = online.observe_routes(
            [route],
            resolver=lambda _host, _port: ["not-an-ip-address"],
            transport=transport,
            clock=deadline_clock,
            sleeper=deadline_sleeper,
        )
        self.assertEqual(deadline_observation["attempted_cohorts"], 1)
        self.assertEqual(deadline_observation["accepted_cohort_sha256"], None)
        self.assertEqual(
            deadline_observation["failures"][0]["code"], "deadline_exhausted"
        )

    def test_canonical_pass_receipt_comparison_hash_is_noncircular(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            site = self.make_site(Path(raw))
            receipt, _receipt_bytes, _inventory = self.build_pass_receipt(site)

            comparison_sha = receipt["comparison_sha256"]
            self.assertIsInstance(comparison_sha, str)
            self.assertEqual(
                comparison_sha,
                offline.compute_comparison_sha256(receipt),
            )

            mutated = copy.deepcopy(receipt)
            mutated["comparison_sha256"] = "0" * 64
            self.assertEqual(
                comparison_sha,
                offline.compute_comparison_sha256(mutated),
            )
            self.assertNotIn(
                "comparison_sha256",
                json.loads(online.canonical_json_bytes({
                    "authority": mutated["authority"],
                    "kind": mutated["kind"],
                    "observation": {
                        "accepted_cohort_sha256": mutated["observation"][
                            "accepted_cohort_sha256"
                        ],
                        "consecutive_matching_cohorts": mutated["observation"][
                            "consecutive_matching_cohorts"
                        ],
                        "routes": mutated["observation"]["routes"],
                    },
                    "policy": mutated["policy"],
                    "result": "PASS",
                    "schema_version": mutated["schema_version"],
                    "subject": mutated["subject"],
                }).decode("utf-8")),
            )

    def test_cli_duplicate_and_unsafe_gates_refuse_without_network(self) -> None:
        base_args = [
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--site",
            "/tmp/site",
            "--deployment-url",
            online.ACCEPTED_BASE_URL,
            "--commit-sha",
            self.commit_sha,
            "--workflow-run-id",
            str(self.run_id),
            "--workflow-run-attempt",
            str(self.run_attempt),
            "--output",
            "/tmp/receipt.json",
        ]
        with self.assertRaises(online.InvocationError):
            online.parse_fixed_cli(base_args + ["--output", "/tmp/other.json"])
        with self.assertRaises(offline.UsageError):
            offline.parse_arguments(
                [
                    "--repository-root",
                    str(REPOSITORY_ROOT),
                    "--repository-root",
                    str(REPOSITORY_ROOT),
                    "--site",
                    "/tmp/site",
                    "--receipt",
                    "/tmp/receipt.json",
                    "--commit-sha",
                    self.commit_sha,
                    "--workflow-run-id",
                    str(self.run_id),
                    "--workflow-run-attempt",
                    str(self.run_attempt),
                    "--require-result",
                    "PASS",
                ]
            )

        with tempfile.TemporaryDirectory() as raw:
            temp = Path(raw)
            site = self.make_site(temp)
            output = temp / "receipt.json"
            args = online.parse_fixed_cli(
                [
                    "--repository-root",
                    str(REPOSITORY_ROOT),
                    "--site",
                    str(site),
                    "--deployment-url",
                    "https://example.invalid/fast-mlx/",
                    "--commit-sha",
                    self.commit_sha,
                    "--workflow-run-id",
                    str(self.run_id),
                    "--workflow-run-attempt",
                    str(self.run_attempt),
                    "--output",
                    str(output),
                ]
            )
            with self.assertRaisesRegex(online.Refusal, "deployment URL"):
                online.validate_local_inputs(args)

            unsafe_output = REPOSITORY_ROOT / "receipt-inside-repo.json"
            args = online.parse_fixed_cli(
                [
                    "--repository-root",
                    str(REPOSITORY_ROOT),
                    "--site",
                    str(site),
                    "--deployment-url",
                    online.ACCEPTED_BASE_URL,
                    "--commit-sha",
                    self.commit_sha,
                    "--workflow-run-id",
                    str(self.run_id),
                    "--workflow-run-attempt",
                    str(self.run_attempt),
                    "--output",
                    str(unsafe_output),
                ]
            )
            with self.assertRaisesRegex(online.Refusal, "output must be outside"):
                online.validate_local_inputs(args)

            receipt_path = temp / "receipt.json"
            receipt_path.write_bytes(b"{}")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = offline.main(
                    [
                        "--repository-root",
                        str(REPOSITORY_ROOT),
                        "--site",
                        str(site),
                        "--receipt",
                        str(receipt_path),
                        "--commit-sha",
                        "not-a-sha",
                        "--workflow-run-id",
                        str(self.run_id),
                        "--workflow-run-attempt",
                        str(self.run_attempt),
                        "--require-result",
                        "PASS",
                    ]
                )
            self.assertEqual(exit_code, 2)
            self.assertIn('"result":"ERROR"', stderr.getvalue())
            self.assertEqual(stdout.getvalue(), "")

    def test_independent_raw_and_zip_offline_validator_pass(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            temp = Path(raw)
            site = self.make_site(temp)
            receipt, receipt_bytes, _inventory = self.build_pass_receipt(site)
            raw_receipt = temp / "raw-receipt.json"
            raw_receipt.write_bytes(receipt_bytes)
            artifact = temp / "receipt-artifact.zip"
            with zipfile.ZipFile(artifact, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(offline.RECEIPT_ARTIFACT_MEMBER, receipt_bytes)

            original_site_validator = offline.validate_public_site.validate
            offline.validate_public_site.validate = lambda _site: []
            try:
                for flag, path in (("--receipt", raw_receipt), ("--artifact-zip", artifact)):
                    with self.subTest(flag=flag):
                        stdout = io.StringIO()
                        stderr = io.StringIO()
                        with redirect_stdout(stdout), redirect_stderr(stderr):
                            exit_code = offline.main(
                                [
                                    "--repository-root",
                                    str(REPOSITORY_ROOT),
                                    "--site",
                                    str(site),
                                    flag,
                                    str(path),
                                    "--commit-sha",
                                    self.commit_sha,
                                    "--workflow-run-id",
                                    str(self.run_id),
                                    "--workflow-run-attempt",
                                    str(self.run_attempt),
                                    "--require-result",
                                    "PASS",
                                ]
                            )
                        self.assertEqual(exit_code, 0, stderr.getvalue())
                        payload = json.loads(stdout.getvalue())
                        self.assertEqual(stderr.getvalue(), "")
                        self.assertEqual(payload["result"], "PASS")
                        self.assertEqual(
                            payload["comparison_sha256"], receipt["comparison_sha256"]
                        )
                        self.assertEqual(
                            payload["route_count"],
                            len(receipt["observation"]["routes"]),
                        )
            finally:
                offline.validate_public_site.validate = original_site_validator

    def test_full_generated_site_receipts_pass_strict_raw_and_zip_validation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            temp = Path(raw)
            site = temp / "site"
            build_public_site.prepare_output(site, REPOSITORY_ROOT)
            build_public_site.build_site(REPOSITORY_ROOT, site)
            build_public_site.scan_generated_output(site)
            self.assertEqual(offline.validate_public_site.validate(site), [])

            receipt, receipt_bytes, _inventory = self.build_pass_receipt(site)
            raw_receipt = temp / offline.RECEIPT_ARTIFACT_MEMBER
            raw_receipt.write_bytes(receipt_bytes)
            artifact = temp / "receipt-artifact.zip"
            with zipfile.ZipFile(
                artifact, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                archive.writestr(offline.RECEIPT_ARTIFACT_MEMBER, receipt_bytes)

            for flag, path in (
                ("--receipt", raw_receipt),
                ("--artifact-zip", artifact),
            ):
                with self.subTest(flag=flag):
                    stdout = io.StringIO()
                    stderr = io.StringIO()
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        exit_code = offline.main(
                            [
                                "--repository-root",
                                str(REPOSITORY_ROOT),
                                "--site",
                                str(site),
                                flag,
                                str(path),
                                "--commit-sha",
                                self.commit_sha,
                                "--workflow-run-id",
                                str(self.run_id),
                                "--workflow-run-attempt",
                                str(self.run_attempt),
                                "--require-result",
                                "PASS",
                            ]
                        )
                    self.assertEqual(exit_code, 0, stderr.getvalue())
                    self.assertEqual(stderr.getvalue(), "")
                    payload = json.loads(stdout.getvalue())
                    self.assertEqual(payload["result"], "PASS")
                    self.assertEqual(
                        payload["comparison_sha256"], receipt["comparison_sha256"]
                    )

    def test_offline_receipt_refuses_canonical_duplicate_wrong_result_authority_hash_and_relations(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            site = self.make_site(Path(raw))
            receipt, receipt_bytes, _inventory = self.build_pass_receipt(site)

            with self.assertRaisesRegex(offline.ValidationError, "canonical pretty"):
                offline.parse_receipt(online.canonical_json_bytes(receipt))
            with self.assertRaisesRegex(offline.ValidationError, "duplicate key"):
                offline.parse_receipt(b'{"result":"PASS","result":"PASS"}\n')

            self.assert_offline_validation_refuses(
                receipt,
                site,
                "result does not match",
                require_result="FAIL",
            )

            authority = copy.deepcopy(receipt)
            authority["authority"]["positive_admission"] = True
            self.assert_offline_validation_refuses(
                authority,
                site,
                "authority.positive_admission",
            )

            subject_hash = copy.deepcopy(receipt)
            subject_hash["subject"]["site_ordered_sha256"] = "0" * 64
            self.assert_offline_validation_refuses(
                subject_hash,
                site,
                "subject.site_ordered_sha256",
            )

            comparison_hash = copy.deepcopy(receipt)
            comparison_hash["comparison_sha256"] = "0" * 64
            self.assert_offline_validation_refuses(
                comparison_hash,
                site,
                "comparison_sha256 does not match",
            )

            pass_without_accepted = copy.deepcopy(receipt)
            pass_without_accepted["observation"]["accepted_cohort_sha256"] = None
            self.assert_offline_validation_refuses(
                pass_without_accepted,
                site,
                "PASS accepted_cohort_sha256",
            )

            fail_with_comparison = copy.deepcopy(receipt)
            fail_with_comparison["result"] = "FAIL"
            fail_with_comparison["comparison_sha256"] = receipt["comparison_sha256"]
            fail_with_comparison["observation"]["accepted_cohort_sha256"] = None
            fail_with_comparison["observation"]["consecutive_matching_cohorts"] = 0
            fail_with_comparison["observation"]["routes"] = []
            failure = online.make_failure(
                "dns_policy_refused",
                receipt["observation"]["routes"][0]["path"],
            )
            fail_with_comparison["observation"]["failures"] = [
                failure
            ]
            fail_with_comparison["observation"]["attempts"] = [
                {
                    "attempt": 1,
                    "cohort_sha256": None,
                    "completed_routes": 0,
                    "failure": failure,
                    "result": "FAIL",
                }
            ]
            fail_with_comparison["observation"]["attempted_cohorts"] = 1
            self.assert_offline_validation_refuses(
                fail_with_comparison,
                site,
                "FAIL receipt comparison_sha256",
                require_result="FAIL",
            )

            parsed = offline.parse_receipt(receipt_bytes)
            self.assertEqual(parsed, receipt)

    def test_hostile_zip_receipts_are_refused_before_extraction(self) -> None:
        receipt_bytes = offline.canonical_pretty(
            {
                "authority": {},
                "comparison_sha256": None,
                "kind": offline.KIND,
                "observation": {},
                "policy": {},
                "result": "FAIL",
                "schema_version": offline.SCHEMA_VERSION,
                "subject": {},
                "workflow": {},
            }
        )

        def write_zip(
            path: Path,
            entries: list[tuple[zipfile.ZipInfo | str, bytes, int | None]],
        ) -> None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Duplicate name:", category=UserWarning
                )
                with zipfile.ZipFile(path, "w") as archive:
                    for info_or_name, data, compression in entries:
                        compression_type = (
                            zipfile.ZIP_DEFLATED
                            if compression is None
                            else compression
                        )
                        archive.writestr(
                            info_or_name, data, compress_type=compression_type
                        )

        with tempfile.TemporaryDirectory() as raw:
            temp = Path(raw)
            cases: list[tuple[str, str]] = []

            wrong_name = temp / "wrong-name.zip"
            write_zip(wrong_name, [("wrong.json", receipt_bytes, None)])
            cases.append(("wrong name", str(wrong_name)))

            extra = temp / "extra.zip"
            write_zip(
                extra,
                [
                    (offline.RECEIPT_ARTIFACT_MEMBER, receipt_bytes, None),
                    ("extra.txt", b"x", None),
                ],
            )
            cases.append(("extra", str(extra)))

            duplicate = temp / "duplicate.zip"
            write_zip(
                duplicate,
                [
                    (offline.RECEIPT_ARTIFACT_MEMBER, receipt_bytes, None),
                    (offline.RECEIPT_ARTIFACT_MEMBER, receipt_bytes, None),
                ],
            )
            cases.append(("duplicate", str(duplicate)))

            stored = temp / "stored.zip"
            write_zip(
                stored,
                [(offline.RECEIPT_ARTIFACT_MEMBER, receipt_bytes, zipfile.ZIP_STORED)],
            )
            cases.append(("non-DEFLATE", str(stored)))

            symlink = temp / "symlink.zip"
            symlink_info = zipfile.ZipInfo(offline.RECEIPT_ARTIFACT_MEMBER)
            symlink_info.compress_type = zipfile.ZIP_DEFLATED
            symlink_info.external_attr = (stat.S_IFLNK | 0o777) << 16
            write_zip(symlink, [(symlink_info, b"target", None)])
            cases.append(("symlink", str(symlink)))

            oversize = temp / "oversize.zip"
            write_zip(
                oversize,
                [
                    (
                        offline.RECEIPT_ARTIFACT_MEMBER,
                        b"0" * (offline.RAW_RECEIPT_LIMIT + 1),
                        None,
                    )
                ],
            )
            cases.append(("oversize", str(oversize)))

            for label, path in cases:
                with self.subTest(label=label):
                    with self.assertRaises(offline.UsageError):
                        offline.read_receipt_from_zip(Path(path))

    def test_offline_validator_source_does_not_import_online_verifier(self) -> None:
        source = (SCRIPTS_DIR / "validate_public_deployment_receipt.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("import verify_public_deployment", source)
        self.assertNotIn("from verify_public_deployment", source)
        self.assertNotIn("production_transport", source)
        self.assertNotIn("production_resolver", source)

    def test_resolver_refuses_bounded_malformed_and_stalled_results(self) -> None:
        cases = (
            (lambda _host, _port: [], "dns_policy_refused"),
            (
                lambda _host, _port: [f"8.8.8.{index}" for index in range(1, 18)],
                "dns_policy_refused",
            ),
            (lambda _host, _port: [object()], "dns_policy_refused"),
            (lambda _host, _port: ["é"], "dns_policy_refused"),
        )
        for resolver, expected_code in cases:
            with self.subTest(expected_code=expected_code, resolver=resolver):
                addresses, code = online.resolve_cohort_addresses(
                    resolver=resolver,
                    deadline=10.0,
                    clock=lambda: 0.0,
                )
                self.assertIsNone(addresses)
                self.assertEqual(code, expected_code)

        def raising_resolver(_host: str, _port: int) -> object:
            raise OSError("sanitized")

        addresses, code = online.resolve_cohort_addresses(
            resolver=raising_resolver,
            deadline=10.0,
            clock=lambda: 0.0,
        )
        self.assertIsNone(addresses)
        self.assertEqual(code, "resolver_error")

        release = threading.Event()

        def stalled_resolver(_host: str, _port: int) -> list[str]:
            release.wait(timeout=0.05)
            return ["8.8.8.8"]

        with mock.patch.object(online, "RESOLVER_TIMEOUT_SECONDS", 0.001):
            addresses, code = online.resolve_cohort_addresses(
                resolver=stalled_resolver,
                deadline=10.0,
                clock=lambda: 0.0,
            )
        release.set()
        self.assertIsNone(addresses)
        self.assertEqual(code, "resolver_timeout")

    def test_direct_https_connection_uses_numeric_socket_and_exact_sni(self) -> None:
        class FakeSocket:
            def __init__(self) -> None:
                self.timeout: float | None = None
                self.endpoint: tuple[object, ...] | None = None
                self.closed = False

            def settimeout(self, timeout: float) -> None:
                self.timeout = timeout

            def connect(self, endpoint: tuple[object, ...]) -> None:
                self.endpoint = endpoint

            def close(self) -> None:
                self.closed = True

        class FakeContext:
            def __init__(self) -> None:
                self.server_hostname: str | None = None

            def wrap_socket(
                self, raw_socket: FakeSocket, *, server_hostname: str
            ) -> FakeSocket:
                self.server_hostname = server_hostname
                return raw_socket

        for address, family, endpoint in (
            ("8.8.8.8", online.socket.AF_INET, ("8.8.8.8", 443)),
            (
                "2001:4860:4860::8888",
                online.socket.AF_INET6,
                ("2001:4860:4860::8888", 443, 0, 0),
            ),
        ):
            with self.subTest(address=address):
                fake_socket = FakeSocket()
                context = FakeContext()
                with mock.patch.object(
                    online.socket, "socket", return_value=fake_socket
                ) as socket_factory, mock.patch.object(
                    online.socket,
                    "create_connection",
                    side_effect=AssertionError("must not resolve during connect"),
                ):
                    connection = online.DirectHTTPSConnection(
                        address,
                        timeout=3.0,
                        context=context,
                        deadline=10.0,
                        clock=lambda: 0.0,
                    )
                    connection.connect()

                socket_factory.assert_called_once_with(family, online.socket.SOCK_STREAM)
                self.assertEqual(fake_socket.timeout, 3.0)
                self.assertEqual(fake_socket.endpoint, endpoint)
                self.assertEqual(context.server_hostname, online.ACCEPTED_HOST)
                self.assertIs(connection.sock._socket, fake_socket)

    def test_direct_https_connection_reduces_tls_timeout_after_connect(self) -> None:
        clock = _ManualClock()

        class SlowConnectSocket:
            def __init__(self) -> None:
                self.timeout: float | None = None
                self.timeouts: list[float] = []

            def settimeout(self, timeout: float) -> None:
                self.timeout = timeout
                self.timeouts.append(timeout)

            def connect(self, _endpoint: tuple[object, ...]) -> None:
                clock.value = 9.5

            def close(self) -> None:
                pass

        class RecordingContext:
            def __init__(self) -> None:
                self.tls_entry_timeout: float | None = None

            def wrap_socket(
                self, raw_socket: SlowConnectSocket, *, server_hostname: str
            ) -> SlowConnectSocket:
                self.asserted_server_hostname = server_hostname
                self.tls_entry_timeout = raw_socket.timeout
                return raw_socket

        fake_socket = SlowConnectSocket()
        context = RecordingContext()
        with mock.patch.object(online.socket, "socket", return_value=fake_socket):
            connection = online.DirectHTTPSConnection(
                "8.8.8.8",
                timeout=10.0,
                context=context,
                deadline=10.0,
                clock=clock,
            )
            connection.connect()

        self.assertEqual(context.asserted_server_hostname, online.ACCEPTED_HOST)
        self.assertEqual(context.tls_entry_timeout, 0.5)
        self.assertEqual(fake_socket.timeouts, [10.0, 0.5, 0.5])

    def test_production_transport_fixes_authority_headers_and_refuses_redirect(self) -> None:
        route = self.make_route(b"abc")
        clock = _ManualClock()
        response_bytes = (
            b"HTTP/1.1 302 Found\r\n"
            b"Content-Type: text/html; charset=utf-8\r\n"
            b"Content-Length: 0\r\n"
            b"Location: https://example.invalid/\r\n"
            b"Connection: close\r\n\r\n"
        )

        class ScriptedSocket:
            def __init__(self) -> None:
                self.endpoint: tuple[object, ...] | None = None
                self.sent: list[bytes] = []
                self.offset = 0
                self.closed = False

            def settimeout(self, _timeout: float) -> None:
                pass

            def connect(self, endpoint: tuple[object, ...]) -> None:
                self.endpoint = endpoint

            def sendall(self, data: bytes, _flags: int = 0) -> None:
                self.sent.append(bytes(data))

            def recv_into(self, buffer: bytearray | memoryview) -> int:
                remaining = response_bytes[self.offset :]
                if not remaining:
                    return 0
                count = min(len(buffer), len(remaining))
                buffer[:count] = remaining[:count]
                self.offset += count
                clock.value = 11.0
                return count

            def close(self) -> None:
                self.closed = True

        class ScriptedContext:
            def __init__(self) -> None:
                self.minimum_version: object | None = None
                self.server_hostname: str | None = None

            def wrap_socket(
                self, raw_socket: ScriptedSocket, *, server_hostname: str
            ) -> ScriptedSocket:
                self.server_hostname = server_hostname
                return raw_socket

        scripted_socket = ScriptedSocket()
        context = ScriptedContext()
        with mock.patch.object(
            online.socket, "socket", return_value=scripted_socket
        ) as socket_factory, mock.patch.object(
            online.ssl, "create_default_context", return_value=context
        ):
            result = online.production_transport(
                route,
                ("8.8.8.8",),
                deadline=10.0,
                clock=clock,
            )

        request = b"".join(scripted_socket.sent)
        socket_factory.assert_called_once_with(
            online.socket.AF_INET, online.socket.SOCK_STREAM
        )
        self.assertEqual(scripted_socket.endpoint, ("8.8.8.8", 443))
        self.assertEqual(context.server_hostname, online.ACCEPTED_HOST)
        self.assertIn(b"GET /fast-mlx/ HTTP/1.1\r\n", request)
        self.assertIn(b"Host: bitworks-io.github.io\r\n", request)
        self.assertIn(b"Accept-Encoding: identity\r\n", request)
        self.assertNotIn(b"example.invalid", request)
        self.assertEqual(result.failure["code"], "http_status_mismatch")
        self.assertEqual(result.failure["status"], 302)
        self.assertIsNone(result.record)

    def test_duplicate_and_malformed_response_headers_refuse_before_body(self) -> None:
        route = self.make_route(b"abc")
        cases = (
            (
                [
                    ("Content-Type", route.expected_content_type),
                    ("Content-Type", route.expected_content_type),
                    ("Content-Length", "3"),
                ],
                "content_type_refused",
            ),
            (
                [
                    ("Content-Type", route.expected_content_type),
                    ("Content-Encoding", "identity"),
                    ("Content-Encoding", "identity"),
                    ("Content-Length", "3"),
                ],
                "content_encoding_refused",
            ),
            (
                [
                    ("Content-Type", route.expected_content_type),
                    ("Content-Length", "3"),
                    ("Content-Length", "3"),
                ],
                "content_length_refused",
            ),
            (
                [
                    ("Content-Type", route.expected_content_type),
                    ("Content-Length", "999999999"),
                ],
                "content_length_refused",
            ),
        )
        for headers, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                response = _FakeResponse(status=200, headers=headers, body=b"abc")
                result = online.evaluate_http_response(route, response)
                self.assertEqual(result.failure["code"], expected_code)
                self.assertEqual(response.read_calls, 0)

        timeout_response = _FakeResponse(
            status=200,
            headers=[
                ("Content-Type", route.expected_content_type),
                ("Content-Length", "3"),
            ],
            body=b"",
            read_error=TimeoutError(),
        )
        self.assertEqual(
            online.evaluate_http_response(route, timeout_response).failure["code"],
            "read_timeout",
        )
        protocol_response = _FakeResponse(
            status=200,
            headers=[
                ("Content-Type", route.expected_content_type),
                ("Content-Length", "3"),
            ],
            body=b"",
            read_error=online.http.client.IncompleteRead(b""),
        )
        self.assertEqual(
            online.evaluate_http_response(route, protocol_response).failure["code"],
            "response_protocol_error",
        )

    def test_deadline_reader_reduces_timeout_across_incremental_reads(self) -> None:
        clock = _ManualClock()

        class TrickleSocket:
            def __init__(self) -> None:
                self.timeouts: list[float] = []

            def settimeout(self, timeout: float) -> None:
                self.timeouts.append(timeout)

            def recv_into(self, buffer: bytearray | memoryview) -> int:
                buffer[0] = 120
                clock.value += 6.0
                return 1

        trickle = TrickleSocket()
        reader = online._DeadlineReader(
            trickle, deadline=10.0, clock=clock
        )
        first = bytearray(1)
        second = bytearray(1)
        self.assertEqual(reader.readinto(first), 1)
        self.assertEqual(reader.readinto(second), 1)
        with self.assertRaises(TimeoutError):
            reader.readinto(bytearray(1))
        self.assertEqual(trickle.timeouts, [10.0, 4.0])
        self.assertEqual(bytes(first + second), b"xx")

    def test_exact_body_comparison_survives_a_mocked_digest_collision(self) -> None:
        route = self.make_route(b"abc")
        response = _FakeResponse(
            status=200,
            headers=[
                ("Content-Type", route.expected_content_type),
                ("Content-Length", "3"),
            ],
            body=b"abd",
        )

        with mock.patch.object(
            online, "sha256_hex", return_value=route.expected_sha256
        ):
            result = online.evaluate_http_response(route, response)

        self.assertIsNone(result.record)
        self.assertEqual(result.failure["code"], "byte_mismatch")
        self.assertEqual(result.failure["observed_sha256"], route.expected_sha256)

    def test_internal_error_receipt_is_independently_valid_fail_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            site = self.make_site(Path(raw))
            inventory = online.inventory_site(site, self.commit_sha)
            local = online.LocalInputs(
                repository_root=REPOSITORY_ROOT,
                site=site,
                output=site.parent / "receipt.json",
                commit_sha=self.commit_sha,
                workflow_run_id=self.run_id,
                workflow_run_attempt=self.run_attempt,
            )
            receipt = online.internal_error_receipt(local, inventory)
            subject = offline.build_local_subject(
                REPOSITORY_ROOT, site, self.commit_sha
            )

            offline.validate_receipt(
                receipt,
                subject,
                self.commit_sha,
                self.run_id,
                self.run_attempt,
                "FAIL",
            )

    def test_attempt_exhaustion_after_final_match_is_non_promotable(self) -> None:
        route = self.make_route(b"abc")
        calls = 0

        def alternating_transport(
            route_subject: online.RouteSubject,
            _addresses: tuple[str, ...],
            _deadline: float,
            _clock: object,
        ) -> online.RouteFetch:
            nonlocal calls
            calls += 1
            if calls % 2:
                return online.RouteFetch(
                    None,
                    online.make_failure("connect_error", route_subject.path),
                )
            return online.RouteFetch(route_subject.accepted_object(), None)

        observation = online.observe_routes(
            [route],
            resolver=lambda _host, _port: ["8.8.8.8"],
            transport=alternating_transport,
            clock=lambda: 0.0,
            sleeper=lambda _seconds: None,
        )

        self.assertEqual(observation["attempted_cohorts"], online.MAX_COHORTS)
        self.assertEqual(observation["attempts"][-1]["result"], "MATCH")
        self.assertEqual(observation["consecutive_matching_cohorts"], 1)
        self.assertEqual(
            observation["failures"][0]["code"], "cohort_attempts_exhausted"
        )
        self.assertIsNone(observation["accepted_cohort_sha256"])
        self.assertEqual(observation["routes"], [])

    def test_receipt_write_is_canonical_exclusive_and_fsynced(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "receipt.json"
            receipt = {"result": "FAIL", "schema_version": 1}
            with mock.patch.object(online.os, "fsync", wraps=online.os.fsync) as fsync:
                online.write_receipt_exclusive(output, receipt)
            self.assertEqual(output.read_bytes(), online.receipt_json_bytes(receipt))
            fsync.assert_called_once()
            with self.assertRaises(FileExistsError):
                online.write_receipt_exclusive(output, receipt)

    def test_equals_form_duplicate_options_are_refused(self) -> None:
        with self.assertRaises(online.InvocationError):
            online.parse_fixed_cli(
                [
                    f"--repository-root={REPOSITORY_ROOT}",
                    "--repository-root",
                    str(REPOSITORY_ROOT),
                    "--site=/tmp/site",
                    f"--deployment-url={online.ACCEPTED_BASE_URL}",
                    f"--commit-sha={self.commit_sha}",
                    f"--workflow-run-id={self.run_id}",
                    f"--workflow-run-attempt={self.run_attempt}",
                    "--output=/tmp/receipt.json",
                ]
            )
        with self.assertRaises(offline.UsageError):
            offline.parse_arguments(
                [
                    f"--repository-root={REPOSITORY_ROOT}",
                    "--site=/tmp/site",
                    "--receipt=/tmp/receipt.json",
                    "--receipt",
                    "/tmp/other.json",
                    f"--commit-sha={self.commit_sha}",
                    f"--workflow-run-id={self.run_id}",
                    f"--workflow-run-attempt={self.run_attempt}",
                    "--require-result=PASS",
                ]
            )

    def test_offline_validator_enforces_exact_cohort_and_failure_relations(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            site = self.make_site(Path(raw))
            receipt, _receipt_bytes, _inventory = self.build_pass_receipt(site)

            three_matches = copy.deepcopy(receipt)
            third = copy.deepcopy(three_matches["observation"]["attempts"][-1])
            third["attempt"] = 3
            three_matches["observation"]["attempts"].append(third)
            three_matches["observation"]["attempted_cohorts"] = 3
            three_matches["observation"]["consecutive_matching_cohorts"] = 3
            self.assert_offline_validation_refuses(
                three_matches, site, "exactly two consecutive"
            )

            non_integer_count = copy.deepcopy(receipt)
            non_integer_count["observation"]["attempts"][0][
                "completed_routes"
            ] = float(len(non_integer_count["observation"]["routes"]))
            self.assert_offline_validation_refuses(
                non_integer_count,
                site,
                "completed_routes must equal route count",
            )

            fail_receipt = copy.deepcopy(receipt)
            path = receipt["observation"]["routes"][0]["path"]
            inconsistent_failure = online.make_failure("byte_mismatch", path)
            fail_receipt["result"] = "FAIL"
            fail_receipt["comparison_sha256"] = None
            fail_receipt["observation"] = {
                "accepted_cohort_sha256": None,
                "attempted_cohorts": 1,
                "attempts": [
                    {
                        "attempt": 1,
                        "cohort_sha256": None,
                        "completed_routes": 0,
                        "failure": inconsistent_failure,
                        "result": "FAIL",
                    }
                ],
                "consecutive_matching_cohorts": 0,
                "failures": [inconsistent_failure],
                "routes": [],
            }
            self.assert_offline_validation_refuses(
                fail_receipt,
                site,
                "byte_mismatch observations are inconsistent",
                require_result="FAIL",
            )

    def test_offline_validator_accepts_one_well_formed_fail_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            site = self.make_site(Path(raw))
            pass_receipt, _receipt_bytes, _inventory = self.build_pass_receipt(site)
            first_path = pass_receipt["observation"]["routes"][0]["path"]
            failure = online.make_failure("dns_policy_refused", first_path)
            fail_receipt = self.build_fail_receipt(pass_receipt, failure)
            subject = offline.build_local_subject(
                REPOSITORY_ROOT, site, self.commit_sha
            )

            offline.validate_receipt(
                fail_receipt,
                subject,
                self.commit_sha,
                self.run_id,
                self.run_attempt,
                "FAIL",
            )

    def test_offline_validator_rejects_false_failure_diagnostics_and_path_count(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            site = self.make_site(Path(raw))
            pass_receipt, _receipt_bytes, _inventory = self.build_pass_receipt(site)
            routes = pass_receipt["observation"]["routes"]

            false_dns_failure = online.make_failure(
                "dns_policy_refused",
                routes[0]["path"],
                status=200,
                observed_bytes=1,
                observed_sha256="0" * 64,
            )
            false_dns_receipt = self.build_fail_receipt(
                pass_receipt, false_dns_failure
            )
            self.assert_offline_validation_refuses(
                false_dns_receipt,
                site,
                "status must be null for pre-response failure",
                require_result="FAIL",
            )

            null_status_mismatch = online.make_failure(
                "http_status_mismatch", routes[0]["path"]
            )
            null_status_receipt = self.build_fail_receipt(
                pass_receipt, null_status_mismatch
            )
            self.assert_offline_validation_refuses(
                null_status_receipt,
                site,
                "status must be a parsed status",
                require_result="FAIL",
            )

            wrong_path_count_failure = online.make_failure(
                "connect_error", routes[1]["path"]
            )
            wrong_path_count_receipt = self.build_fail_receipt(
                pass_receipt,
                wrong_path_count_failure,
                completed_routes=0,
            )
            self.assert_offline_validation_refuses(
                wrong_path_count_receipt,
                site,
                "completed_routes does not match failure path",
                require_result="FAIL",
            )

    def test_missing_attempt_key_is_invalid_not_internal_error(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            temp = Path(raw)
            site = self.make_site(temp)
            pass_receipt, _receipt_bytes, _inventory = self.build_pass_receipt(site)
            first_path = pass_receipt["observation"]["routes"][0]["path"]
            failure = online.make_failure("dns_policy_refused", first_path)
            malformed = self.build_fail_receipt(pass_receipt, failure)
            del malformed["observation"]["attempts"][0]["result"]

            self.assert_offline_validation_refuses(
                malformed,
                site,
                "keys differ",
                require_result="FAIL",
            )

            receipt_path = temp / "missing-attempt-result.json"
            receipt_path.write_bytes(offline.canonical_pretty(malformed))
            stdout = io.StringIO()
            stderr = io.StringIO()
            with mock.patch.object(
                offline.validate_public_site, "validate", return_value=[]
            ), redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = offline.main(
                    [
                        "--repository-root",
                        str(REPOSITORY_ROOT),
                        "--site",
                        str(site),
                        "--receipt",
                        str(receipt_path),
                        "--commit-sha",
                        self.commit_sha,
                        "--workflow-run-id",
                        str(self.run_id),
                        "--workflow-run-attempt",
                        str(self.run_attempt),
                        "--require-result",
                        "FAIL",
                    ]
                )
            self.assertEqual(exit_code, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn('"result":"INVALID"', stderr.getvalue())

    def test_unhashable_receipt_fields_are_deterministically_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            site = self.make_site(Path(raw))
            pass_receipt, _receipt_bytes, _inventory = self.build_pass_receipt(site)
            first_path = pass_receipt["observation"]["routes"][0]["path"]

            cases = (
                ("code", [], "code is not allowed"),
                ("path", [], "path is not an accepted receipt route"),
            )
            for field, value, expected in cases:
                with self.subTest(field=field):
                    case_failure = online.make_failure(
                        "dns_policy_refused", first_path
                    )
                    malformed = self.build_fail_receipt(
                        pass_receipt, case_failure
                    )
                    malformed["observation"]["attempts"][0]["failure"][field] = value
                    self.assert_offline_validation_refuses(
                        malformed,
                        site,
                        expected,
                        require_result="FAIL",
                    )

            malformed_result = copy.deepcopy(pass_receipt)
            malformed_result["result"] = []
            self.assert_offline_validation_refuses(
                malformed_result,
                site,
                "result must be exact PASS or FAIL",
            )

    def test_hostile_json_canonicalization_is_deterministically_invalid(self) -> None:
        hostile_receipts = (
            b'{"schema_version":1e10000}\n',
            b'{"value":"\\ud800"}\n',
            (b'{"value":' + b"[" * 1500 + b"0" + b"]" * 1500 + b"}\n"),
        )
        for raw in hostile_receipts:
            with self.subTest(length=len(raw)):
                with self.assertRaises(offline.ValidationError):
                    offline.parse_receipt(raw)

    def test_raw_receipt_size_cap_refuses_before_json_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "oversize.json"
            path.write_bytes(b"0" * (offline.RAW_RECEIPT_LIMIT + 1))
            with self.assertRaisesRegex(offline.UsageError, "exceeds"):
                offline.read_regular_file(
                    path, "receipt", offline.RAW_RECEIPT_LIMIT
                )


if __name__ == "__main__":
    unittest.main()
