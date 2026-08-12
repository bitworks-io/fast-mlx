from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import validate_public_repository  # noqa: E402


class PublicRepositoryLicenseTests(unittest.TestCase):
    def test_project_license_and_notice_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            failures = validate_public_repository.validate(Path(directory))

        self.assertIn("missing required public file: LICENSE", failures)
        self.assertIn("missing required public file: NOTICE", failures)

    def test_capability_manifest_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            failures = validate_public_repository.validate(Path(directory))

        self.assertIn("missing required public file: site/capabilities.json", failures)

    def test_project_license_must_match_official_apache_2_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            (repository / "LICENSE").write_text("not Apache-2.0\n", encoding="utf-8")
            (repository / "NOTICE").write_text(
                "fast-mlx\nCopyright 2026 bitworks-io\n", encoding="utf-8"
            )

            failures = validate_public_repository.validate(repository)

        self.assertIn("LICENSE does not match the pinned Apache-2.0 text", failures)

    def test_project_notice_must_retain_project_and_copyright_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            (repository / "LICENSE").write_bytes((REPOSITORY_ROOT / "LICENSE").read_bytes())
            (repository / "NOTICE").write_text("incomplete\n", encoding="utf-8")

            failures = validate_public_repository.validate(repository)

        self.assertIn("NOTICE is missing required project identity: fast-mlx", failures)
        self.assertIn(
            "NOTICE is missing required copyright identity: Copyright 2026 bitworks-io",
            failures,
        )

    def test_private_engineering_paths_are_forbidden_in_public_text(self) -> None:
        cases = (
            ("docs/" + "superpowers" + "/verdicts/private.md", "docs/" + "superpowers" + "/"),
            ("docs/" + "superpowers" + "/specs/private.md", "docs/" + "superpowers" + "/"),
            ("spike/" + "scripts" + "/private.py", "spike/" + "scripts" + "/"),
        )
        for source_path, marker in cases:
            with self.subTest(source_path=source_path):
                with tempfile.TemporaryDirectory() as directory:
                    repository = Path(directory)
                    article = repository / "docs/content/example.md"
                    article.parent.mkdir(parents=True)
                    article.write_text(f"See {source_path}\n", encoding="utf-8")

                    failures = validate_public_repository.validate(repository)

                self.assertIn(
                    f"private marker {marker!r} in docs/content/example.md",
                    failures,
                )

    def test_public_manifest_exports_license_notice_and_validator_test(self) -> None:
        manifest_path = REPOSITORY_ROOT / "public/public-repository.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            destinations = {
                entry["destination"]
                for entry in manifest["files"]
                if isinstance(entry, dict) and isinstance(entry.get("destination"), str)
            }
        else:
            destinations = {
                path
                for path in (
                    "LICENSE",
                    "NOTICE",
                    "scripts/tests/test_public_repository.py",
                )
                if (REPOSITORY_ROOT / path).is_file()
            }

        self.assertIn("LICENSE", destinations)
        self.assertIn("NOTICE", destinations)
        self.assertIn("scripts/tests/test_public_repository.py", destinations)

    def test_public_quality_workflow_validates_engineering_export_or_public_checkout(self) -> None:
        workflow = (REPOSITORY_ROOT / ".github/workflows/quality.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("python3 scripts/export_public_repository.py", workflow)
        self.assertIn('"$RUNNER_TEMP/fast-mlx-public"', workflow)
        self.assertIn("python3 scripts/validate_public_repository.py .", workflow)
        self.assertNotIn("Engineering checkout detected", workflow)

    def test_public_quality_workflow_uses_swift_6_3_capable_macos_runner(self) -> None:
        workflow = (REPOSITORY_ROOT / ".github/workflows/quality.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("runs-on: macos-26", workflow)
        self.assertNotIn("runs-on: macos-15", workflow)
        self.assertIn("swift --version", workflow)

    def test_pages_deployment_is_called_after_complete_main_quality(self) -> None:
        pages = (REPOSITORY_ROOT / ".github/workflows/pages.yml").read_text(
            encoding="utf-8"
        )
        quality = (REPOSITORY_ROOT / ".github/workflows/quality.yml").read_text(
            encoding="utf-8"
        )

        trigger_block = pages.split("on:\n", 1)[1].split("\npermissions:", 1)[0]

        self.assertEqual(trigger_block.strip(), "workflow_call:")
        self.assertNotIn("workflow_run:", pages)
        self.assertNotIn("workflow_dispatch:", pages)
        self.assertIn("group: public-source-quality-${{ github.ref }}", quality)
        self.assertIn("cancel-in-progress: true", quality)
        pages_job = quality.split("\n  pages:\n", 1)[1]
        self.assertIn("needs: [public-boundary, pure-swift-targets]", pages_job)
        self.assertIn("github.event_name == 'push'", pages_job)
        self.assertIn("github.ref == 'refs/heads/main'", pages_job)
        self.assertIn("uses: ./.github/workflows/pages.yml", pages_job)
        self.assertIn("contents: read", pages_job)
        self.assertIn("pages: write", pages_job)
        self.assertIn("id-token: write", pages_job)

    def test_pages_jobs_use_least_privilege_and_eligible_concurrency(self) -> None:
        workflow = (REPOSITORY_ROOT / ".github/workflows/pages.yml").read_text(
            encoding="utf-8"
        )
        before_jobs, jobs = workflow.split("\njobs:\n", 1)
        build, deploy = jobs.split("\n  deploy:\n", 1)

        self.assertEqual(
            before_jobs.split("\npermissions:\n", 1)[1].split("\n\n", 1)[0].strip(),
            "contents: read",
        )
        self.assertIn("permissions:\n      contents: read\n      pages: read", build)
        self.assertIn(
            "if: github.event_name == 'push' && github.ref == 'refs/heads/main'",
            build,
        )
        self.assertNotIn("pages: write", build)
        self.assertNotIn("id-token: write", build)
        self.assertNotIn("\nconcurrency:\n", before_jobs)
        self.assertIn(
            "permissions:\n      pages: write\n      id-token: write",
            deploy,
        )
        self.assertIn(
            "if: github.event_name == 'push' && github.ref == 'refs/heads/main'",
            deploy,
        )
        self.assertIn(
            "concurrency:\n      group: github-pages\n      cancel-in-progress: true",
            deploy,
        )
        self.assertNotIn("workflow_run", workflow)

    def test_pages_actions_use_reviewed_node_24_revisions(self) -> None:
        workflow = (REPOSITORY_ROOT / ".github/workflows/pages.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "actions/configure-pages@45bfe0192ca1faeb007ade9deae92b16b8254a0d # v6.0.0",
            workflow,
        )
        self.assertIn(
            "actions/upload-pages-artifact@fc324d3547104276b827a68afc52ff2a11cc49c9 # v5.0.0",
            workflow,
        )
        self.assertIn(
            "actions/deploy-pages@cd2ce8fcbc39b97be8ca5fce6e763baed58fa128 # v5.0.0",
            workflow,
        )


if __name__ == "__main__":
    unittest.main()
