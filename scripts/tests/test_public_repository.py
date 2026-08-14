from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import validate_public_repository  # noqa: E402


class PublicRepositoryLicenseTests(unittest.TestCase):
    def write_identity_fixture(self, repository: Path) -> None:
        readme = repository / "README.md"
        manifest = repository / "public/public-repository.json"
        manifest.parent.mkdir(parents=True)
        readme.write_text("# Fixture\n", encoding="utf-8")

        entries = {
            "README.md": "100644",
            "public/public-repository.json": "100644",
        }
        digest = hashlib.sha256()
        for path in sorted(entries):
            digest.update(entries[path].encode("ascii"))
            digest.update(b"\0")
            digest.update(path.encode("utf-8"))
            digest.update(b"\0")
        manifest.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "publicIndex": {
                        "pathCount": len(entries),
                        "pathModeSha256": digest.hexdigest(),
                    },
                    "files": [
                        {"source": "README.md", "destination": "README.md"},
                        {
                            "source": "public/public-repository.json",
                            "destination": "public/public-repository.json",
                        },
                    ],
                    "trees": [],
                }
            ),
            encoding="utf-8",
        )

    def test_project_license_and_notice_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            failures = validate_public_repository.validate(Path(directory))

        self.assertIn("missing required public file: LICENSE", failures)
        self.assertIn("missing required public file: NOTICE", failures)

    def test_capability_manifest_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            failures = validate_public_repository.validate(Path(directory))

        self.assertIn("missing required public file: site/capabilities.json", failures)

    def test_release_ledger_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            failures = validate_public_repository.validate(Path(directory))

        self.assertIn("missing required public file: site/releases.json", failures)

    def test_benchmark_runtime_fixture_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            failures = validate_public_repository.validate(Path(directory))

        self.assertIn(
            "missing required public file: scripts/tests/benchmark_explorer_node_test.js",
            failures,
        )

    def test_research_explorer_source_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            failures = validate_public_repository.validate(Path(directory))

        self.assertIn(
            "missing required public file: site/assets/research-explorer.js",
            failures,
        )

    def test_operator_quickstart_source_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            failures = validate_public_repository.validate(Path(directory))

        self.assertIn(
            "missing required public file: site/fragments/quickstart.html",
            failures,
        )

    def test_license_page_source_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            failures = validate_public_repository.validate(Path(directory))

        self.assertIn(
            "missing required public file: site/fragments/license.html",
            failures,
        )

    def test_current_status_contract_test_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            failures = validate_public_repository.validate(Path(directory))

        self.assertIn(
            "missing required public file: scripts/tests/test_public_status.py",
            failures,
        )

    def test_post_deploy_receipt_contract_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            failures = validate_public_repository.validate(Path(directory))

        for path in (
            "scripts/verify_public_deployment.py",
            "scripts/validate_public_deployment_receipt.py",
            "scripts/tests/test_public_deployment.py",
        ):
            with self.subTest(path=path):
                self.assertIn(f"missing required public file: {path}", failures)

    def test_public_projection_toolchain_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            failures = validate_public_repository.validate(Path(directory))

        for path in (
            "public/public-repository.json",
            "scripts/export_public_repository.py",
            "scripts/tests/test_public_export.py",
        ):
            with self.subTest(path=path):
                self.assertIn(f"missing required public file: {path}", failures)

    def test_engineering_identity_manifest_is_forbidden_in_public_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            manifest = repository / "public/public-repository-public.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}\n", encoding="utf-8")

            failures = validate_public_repository.validate(repository)

        self.assertIn(
            "forbidden public path exists: public/public-repository-public.json",
            failures,
        )

    def test_public_validator_requires_sealed_identity_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            manifest = repository / "public/public-repository.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "files": [],
                        "trees": [],
                    }
                ),
                encoding="utf-8",
            )

            failures = validate_public_repository.validate(repository)

        self.assertIn("public identity manifest requires publicIndex", failures)

    def test_public_validator_accepts_valid_identity_manifest_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self.write_identity_fixture(repository)

            failures = validate_public_repository.validate(repository)

        self.assertFalse(
            [failure for failure in failures if failure.startswith("public identity")],
            failures,
        )

    def test_public_validator_rejects_extra_candidate_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self.write_identity_fixture(repository)
            (repository / "ambient-extra.txt").write_text("extra\n", encoding="utf-8")

            failures = validate_public_repository.validate(repository)

        self.assertIn(
            "public identity manifest seal does not match candidate path/mode set",
            failures,
        )
        self.assertIn(
            "public identity manifest does not cover the exact candidate path set",
            failures,
        )

    def test_public_validator_rejects_remapped_identity_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self.write_identity_fixture(repository)
            manifest_path = repository / "public/public-repository.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"][0]["destination"] = "docs/README.md"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            failures = validate_public_repository.validate(repository)

        self.assertIn(
            "public identity manifest file entry must use identity paths",
            failures,
        )

    def test_public_validator_rejects_candidate_mode_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self.write_identity_fixture(repository)
            (repository / "README.md").chmod(0o755)

            failures = validate_public_repository.validate(repository)

        self.assertIn(
            "public identity manifest seal does not match candidate path/mode set",
            failures,
        )

    def test_public_validator_matches_git_owner_execute_bit_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self.write_identity_fixture(repository)
            (repository / "README.md").chmod(0o645)

            failures = validate_public_repository.validate(repository)

        self.assertNotIn(
            "public identity manifest seal does not match candidate path/mode set",
            failures,
        )

    def test_public_validator_rejects_special_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self.write_identity_fixture(repository)
            os.mkfifo(repository / "unexpected-pipe")

            failures = validate_public_repository.validate(repository)

        self.assertIn("special file is forbidden: unexpected-pipe", failures)

    def test_public_validator_does_not_skip_nested_git_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self.write_identity_fixture(repository)
            nested_config = repository / "vendor/.git/config"
            nested_config.parent.mkdir(parents=True)
            nested_config.write_text(
                "BEGIN OPENSSH" + " PRIVATE KEY\n",
                encoding="utf-8",
            )

            failures = validate_public_repository.validate(repository)

        self.assertIn(
            "public identity manifest seal does not match candidate path/mode set",
            failures,
        )
        self.assertIn(
            "private marker 'BEGIN OPENSSH"
            + " PRIVATE KEY' in vendor/.git/config",
            failures,
        )

    def test_public_validator_rejects_empty_nested_git_metadata_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self.write_identity_fixture(repository)
            (repository / "vendor/.git").mkdir(parents=True)

            failures = validate_public_repository.validate(repository)

        self.assertIn("nested Git metadata is forbidden: vendor/.git", failures)

    def test_public_validator_rejects_private_marker_in_decoded_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            private_relative = "docs/" + "superpowers/verdicts/private.md"
            private_path = repository / private_relative
            manifest_path = repository / "public/public-repository.json"
            private_path.parent.mkdir(parents=True)
            manifest_path.parent.mkdir(parents=True)
            private_path.write_text("benign fixture\n", encoding="utf-8")
            entries = {
                private_relative: "100644",
                "public/public-repository.json": "100644",
            }
            digest = hashlib.sha256()
            for path in sorted(entries):
                digest.update(entries[path].encode("ascii"))
                digest.update(b"\0")
                digest.update(path.encode("utf-8"))
                digest.update(b"\0")
            manifest = json.dumps(
                {
                    "schemaVersion": 1,
                    "publicIndex": {
                        "pathCount": len(entries),
                        "pathModeSha256": digest.hexdigest(),
                    },
                    "files": [
                        {
                            "source": private_relative,
                            "destination": private_relative,
                        },
                        {
                            "source": "public/public-repository.json",
                            "destination": "public/public-repository.json",
                        },
                    ],
                    "trees": [],
                }
            ).replace("docs/" + "superpowers", "docs\\/" + "superpowers")
            manifest_path.write_text(manifest, encoding="utf-8")

            failures = validate_public_repository.validate(repository)

        self.assertIn(
            "private marker 'docs/"
            + "superpowers/' in candidate path "
            + private_relative,
            failures,
        )

    def test_public_validator_rejects_article_manifest_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            manifest_path = repository / "public/public-repository.json"
            publications_path = repository / "site/publications.json"
            article_path = repository / "docs/content/2026-08-14-note.md"
            manifest_path.parent.mkdir(parents=True)
            publications_path.parent.mkdir(parents=True)
            article_path.parent.mkdir(parents=True)
            article_path.write_text("# Note\n", encoding="utf-8")
            publications_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "articles": [
                            {"source": "docs/content/2026-08-14-note.md"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            entries = {
                "docs/content/2026-08-14-note.md": "100644",
                "public/public-repository.json": "100644",
                "site/publications.json": "100644",
            }
            digest = hashlib.sha256()
            for path in sorted(entries):
                digest.update(entries[path].encode("ascii"))
                digest.update(b"\0")
                digest.update(path.encode("utf-8"))
                digest.update(b"\0")
            manifest_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "publicIndex": {
                            "pathCount": len(entries),
                            "pathModeSha256": digest.hexdigest(),
                        },
                        "files": [
                            {
                                "source": "docs/content/2026-08-14-note.md",
                                "destination": "docs/content/2026-08-14-note.md",
                            },
                            {
                                "source": "public/public-repository.json",
                                "destination": "public/public-repository.json",
                            },
                            {
                                "source": "site/publications.json",
                                "destination": "site/publications.json",
                            },
                        ],
                        "trees": [],
                    }
                ),
                encoding="utf-8",
            )

            failures = validate_public_repository.validate(repository)

        self.assertIn(
            "published article overlaps public identity manifest: "
            "docs/content/2026-08-14-note.md",
            failures,
        )

    def test_public_validator_rejects_article_without_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self.write_identity_fixture(repository)
            publications_path = repository / "site/publications.json"
            publications_path.parent.mkdir(parents=True)
            publications_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "articles": [
                            {
                                "slug": "missing-source",
                                "status": "published",
                                "reviewedAt": "2026-08-14",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            failures = validate_public_repository.validate(repository)

        self.assertIn("invalid published article source: None", failures)

    def test_public_validator_rejects_publications_schema_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self.write_identity_fixture(repository)
            publications_path = repository / "site/publications.json"
            publications_path.parent.mkdir(parents=True)
            publications_path.write_text(
                json.dumps({"schemaVersion": 2, "articles": []}),
                encoding="utf-8",
            )

            failures = validate_public_repository.validate(repository)

        self.assertIn("site/publications.json must use schemaVersion 1", failures)

    def test_public_validator_rejects_invalid_utf8_publications(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self.write_identity_fixture(repository)
            publications_path = repository / "site/publications.json"
            publications_path.parent.mkdir(parents=True)
            publications_path.write_bytes(b"\xff")

            failures = validate_public_repository.validate(repository)

        self.assertTrue(
            any(
                failure.startswith("invalid site/publications.json:")
                for failure in failures
            ),
            failures,
        )

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

    def test_public_manifest_exports_license_notice_release_ledger_and_validator_test(self) -> None:
        manifest_path = REPOSITORY_ROOT / "public/public-repository.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            destinations = {
                entry["destination"]
                for entry in manifest["files"]
                if isinstance(entry, dict) and isinstance(entry.get("destination"), str)
            }
            site_tree_exports_releases = any(
                entry.get("source") == "site"
                and entry.get("destination") == "site"
                and "releases.json" not in entry.get("exclude", [])
                for entry in manifest.get("trees", [])
                if isinstance(entry, dict)
            )
        else:
            destinations = {
                path
                for path in (
                    "LICENSE",
                    "NOTICE",
                    "site/releases.json",
                    "scripts/verify_public_deployment.py",
                    "scripts/validate_public_deployment_receipt.py",
                    "scripts/tests/test_public_deployment.py",
                    "scripts/tests/test_public_repository.py",
                    "scripts/tests/test_public_status.py",
                    "scripts/tests/benchmark_explorer_node_test.js",
                )
                if (REPOSITORY_ROOT / path).is_file()
            }
            site_tree_exports_releases = "site/releases.json" in destinations

        self.assertIn("LICENSE", destinations)
        self.assertIn("NOTICE", destinations)
        self.assertTrue(site_tree_exports_releases)
        self.assertIn("scripts/verify_public_deployment.py", destinations)
        self.assertIn("scripts/validate_public_deployment_receipt.py", destinations)
        self.assertIn("scripts/tests/test_public_deployment.py", destinations)
        self.assertIn("public/public-repository.json", destinations)
        self.assertIn("scripts/export_public_repository.py", destinations)
        self.assertIn("scripts/tests/test_public_export.py", destinations)
        self.assertIn("scripts/tests/test_public_repository.py", destinations)
        self.assertIn("scripts/tests/test_public_status.py", destinations)
        self.assertIn("scripts/tests/benchmark_explorer_node_test.js", destinations)

    def test_public_quality_workflow_validates_checkout_and_reexport(self) -> None:
        workflow = (REPOSITORY_ROOT / ".github/workflows/quality.yml").read_text(
            encoding="utf-8"
        )

        checkout_validation = "python3 scripts/validate_public_repository.py ."
        export_command = "python3 scripts/export_public_repository.py"
        candidate_validation = (
            "python3 scripts/validate_public_repository.py \\\n"
            '            "$RUNNER_TEMP/fast-mlx-public"'
        )
        public_boundary = workflow.split("\n  public-boundary:\n", 1)[1].split(
            "\n  pure-swift-targets:\n", 1
        )[0]

        self.assertIn(
            '    env:\n      PYTHONDONTWRITEBYTECODE: "1"\n    steps:',
            public_boundary,
        )
        self.assertIn(checkout_validation, public_boundary)
        self.assertIn(export_command, public_boundary)
        self.assertIn('"$RUNNER_TEMP/fast-mlx-public"', public_boundary)
        self.assertIn(candidate_validation, public_boundary)
        self.assertLess(
            public_boundary.index(checkout_validation),
            public_boundary.index(export_command),
        )
        self.assertLess(
            public_boundary.index(export_command),
            public_boundary.index(candidate_validation),
        )
        self.assertNotIn("--development-projection", public_boundary)
        self.assertNotIn("projection_mode", public_boundary)
        self.assertNotIn("if [[ -f public/public-repository.json ]]", workflow)

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
