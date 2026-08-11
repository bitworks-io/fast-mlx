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


if __name__ == "__main__":
    unittest.main()
