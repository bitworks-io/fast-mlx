from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import export_public_repository  # noqa: E402
import validate_public_repository  # noqa: E402


PUBLIC_VENDOR_SOURCE_OVERRIDES = {
    "spike/Vendor/mlx-swift-lm/Libraries/MLXLLM/LLMModelFactory.swift": {
        "source": "public/sanitized-projection/spike/Vendor/mlx-swift-lm/Libraries/MLXLLM/LLMModelFactory.swift",
        "sha256": "7185ddfe5847fbbdd5b44b682c645d269ddfc3c33f18e405f6ad00e97e167bd0",
    },
    "spike/Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift": {
        "source": "public/sanitized-projection/spike/Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift",
        "sha256": "e9b3a174ed61aa172b0f1d4a51312ba9536f8d4cc09ccd0e3dbf28f06a94808f",
    },
    "spike/Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen3MoELazyModel.swift": {
        "source": "public/sanitized-projection/spike/Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen3MoELazyModel.swift",
        "sha256": "9f5c926ffe8625b6b17dd7a48056f2d638f29b1a9036b188a6cf1c5e16230d44",
    },
    "spike/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift": {
        "source": "public/sanitized-projection/spike/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift",
        "sha256": "300e934c43eeffbbac0f90d3befdb8818907aa761ed34ef2cec2ed25d808f1c7",
    },
    "spike/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/MTPDrafterModel.swift": {
        "source": "public/sanitized-projection/spike/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/MTPDrafterModel.swift",
        "sha256": "92961eeae7ebf41df1d6139cef4ec2d0120ce568fe1d735d29cb583fa8f42add",
    },
    "spike/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift": {
        "source": "public/sanitized-projection/spike/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift",
        "sha256": "dbc0022d335dc58d701f6e63c2c1b021de52bc0d564fa99939e5a4cdbb3fbbcf",
    },
    "spike/Vendor/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift": {
        "source": "public/sanitized-projection/spike/Vendor/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift",
        "sha256": "0e0cd9862862ab59759c1496db85b217a2adb195f4ae6f25ac18288ea378b275",
    },
    "spike/Vendor/mlx-swift-lm/Tests/MLXLMTests/KVCacheTests.swift": {
        "source": "public/sanitized-projection/spike/Vendor/mlx-swift-lm/Tests/MLXLMTests/KVCacheTests.swift",
        "sha256": "af9608bdc3b308a4e4eb61577408ae0568b05ea0d3e95399e6c3a989aeee182f",
    },
    "spike/Vendor/mlx-swift-lm/Tests/MLXLMTests/MTPSpeculativeTokenIteratorTests.swift": {
        "source": "public/sanitized-projection/spike/Vendor/mlx-swift-lm/Tests/MLXLMTests/MTPSpeculativeTokenIteratorTests.swift",
        "sha256": "8c85863172a0b83fbd0b43403441fcd19ec2a687aa750f8f0907597eedc9a826",
    },
    "spike/Vendor/mlx-swift-lm/Tests/MLXLMTests/Qwen35MTPTests.swift": {
        "source": "public/sanitized-projection/spike/Vendor/mlx-swift-lm/Tests/MLXLMTests/Qwen35MTPTests.swift",
        "sha256": "621e756789d279c062a9b04d40c0ba69ebc65f5c194cc4c0ccdfbe66932e576f",
    },
    "spike/Vendor/mlx-swift-lm/Tests/MLXLMTests/Qwen3MoELazyModelTests.swift": {
        "source": "public/sanitized-projection/spike/Vendor/mlx-swift-lm/Tests/MLXLMTests/Qwen3MoELazyModelTests.swift",
        "sha256": "9d66385dbf031beaf34dffb3d379e7a6e7cd4e5df30b7ad045793a5a2d5e4ac4",
    },
}


def public_index_seal(entries: dict[str, str]) -> dict[str, object]:
    digest = hashlib.sha256()
    for path in sorted(entries):
        digest.update(entries[path].encode("ascii"))
        digest.update(b"\0")
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
    return {
        "pathCount": len(entries),
        "pathModeSha256": digest.hexdigest(),
    }


class PublicExportTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> None:
        files = {
            "public/public-repository.json": json.dumps(
                {
                    "schemaVersion": 1,
                    "files": [
                        {"source": "README.md", "destination": "README.md"},
                        {
                            "source": "public/public-repository-public.json",
                            "destination": "public/public-repository.json",
                        },
                    ],
                    "trees": [
                        {"source": "site", "destination": "site"}
                    ],
                }
            ),
            "public/public-repository-public.json": json.dumps(
                {
                    "schemaVersion": 1,
                    "publicIndex": public_index_seal(
                        {
                            "README.md": "100644",
                            "public/public-repository.json": "100644",
                            "site/publications.json": "100644",
                            "site/assets/site.css": "100644",
                            "docs/content/2026-08-06-note.md": "100644",
                        }
                    ),
                    "files": [
                        {"source": "README.md", "destination": "README.md"},
                        {
                            "source": "public/public-repository.json",
                            "destination": "public/public-repository.json",
                        },
                    ],
                    "trees": [{"source": "site", "destination": "site"}],
                }
            ),
            "site/publications.json": json.dumps(
                {
                    "schemaVersion": 1,
                    "articles": [
                        {
                            "source": "docs/content/2026-08-06-note.md",
                            "slug": "note",
                            "status": "published",
                            "reviewedAt": "2026-08-06",
                        }
                    ],
                }
            ),
            "site/assets/site.css": "body {}\n",
            "docs/content/2026-08-06-note.md": "# Note\n\nPublic body.\n",
            "README.md": "# Fixture\n",
        }
        for name, contents in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
        (root / "untracked-secret.txt").write_text("must not copy", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "add", "README.md", "public", "site", "docs"], cwd=root, check=True)

    def test_project_manifest_exports_projection_toolchain(self) -> None:
        canonical_manifest = json.loads(
            (REPOSITORY_ROOT / "public/public-repository.json").read_text(
                encoding="utf-8"
            )
        )
        public_manifest_path = REPOSITORY_ROOT / "public/public-repository-public.json"
        public_manifest = (
            json.loads(public_manifest_path.read_text(encoding="utf-8"))
            if public_manifest_path.is_file()
            else canonical_manifest
        )
        public_mappings = {
            (entry.get("source"), entry.get("destination"))
            for entry in public_manifest.get("files", [])
            if isinstance(entry, dict)
        }

        if public_manifest_path.is_file():
            engineering_mappings = {
                (entry.get("source"), entry.get("destination"))
                for entry in canonical_manifest.get("files", [])
                if isinstance(entry, dict)
            }
            self.assertIn(
                (
                    "public/public-repository-public.json",
                    "public/public-repository.json",
                ),
                engineering_mappings,
            )
            self.assertIn(
                ("public/docs-content-README.md", "docs/content/README.md"),
                engineering_mappings,
            )
        for path in (
            "scripts/export_public_repository.py",
            "scripts/tests/test_public_export.py",
        ):
            self.assertIn((path, path), public_mappings)
        self.assertIn(
            ("public/public-repository.json", "public/public-repository.json"),
            public_mappings,
        )
        for entry in public_manifest.get("files", []):
            self.assertEqual(entry.get("source"), entry.get("destination"))
        for entry in public_manifest.get("trees", []):
            self.assertEqual(entry.get("source"), entry.get("destination"))
        self.assertEqual(
            set(public_manifest.get("publicIndex", {})),
            {"pathCount", "pathModeSha256"},
        )

    def test_current_development_projection_passes_public_validator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "public"
            output.mkdir()
            export_public_repository.export(
                REPOSITORY_ROOT,
                output,
                allow_development_manifest=(
                    REPOSITORY_ROOT
                    / "public/public-repository-public.json"
                ).is_file(),
            )

            failures = validate_public_repository.validate(output)

        self.assertEqual(failures, [])

    def test_sampled_generation_foundation_is_exported_byte_for_byte(self) -> None:
        development_manifest = json.loads(
            (REPOSITORY_ROOT / "public/public-repository.json").read_text(
                encoding="utf-8"
            )
        )
        tree_entries = {
            entry["source"]: entry
            for entry in development_manifest["trees"]
        }
        source_path = "HarnessCore/Sampling/SamplingContractV1.swift"
        test_path = "HarnessCoreTests/SamplingContractV1Tests.swift"
        self.assertNotIn(
            source_path,
            tree_entries["spike/Sources"].get("exclude", []),
        )
        self.assertNotIn(
            test_path,
            tree_entries["spike/Tests"].get("exclude", []),
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "public"
            output.mkdir()
            export_public_repository.export(
                REPOSITORY_ROOT,
                output,
                allow_development_manifest=(
                    REPOSITORY_ROOT
                    / "public/public-repository-public.json"
                ).is_file(),
            )
            for relative in (
                "spike/Sources/HarnessCore/Sampling/SamplingContractV1.swift",
                "spike/Tests/HarnessCoreTests/SamplingContractV1Tests.swift",
            ):
                self.assertEqual(
                    (output / relative).read_bytes(),
                    (REPOSITORY_ROOT / relative).read_bytes(),
                )

    def test_public_projection_uses_sanitized_vendor_overrides(self) -> None:
        development_manifest = json.loads(
            (REPOSITORY_ROOT / "public/public-repository.json").read_text(
                encoding="utf-8"
            )
        )
        public_manifest_path = REPOSITORY_ROOT / "public/public-repository-public.json"
        has_development_manifest = public_manifest_path.is_file()
        public_manifest = (
            json.loads(public_manifest_path.read_text(encoding="utf-8"))
            if has_development_manifest
            else development_manifest
        )
        self.assertEqual(
            public_manifest.get("publicIndex"),
            {
                "pathCount": 839,
                "pathModeSha256": "316023f4e8bcfa7c6447940643187c1b2ae740fb56981ed25241a16733ca5643",
            },
        )

        if has_development_manifest:
            vendor_root = "spike/Vendor/mlx-swift-lm"
            vendor_tree = next(
                entry
                for entry in development_manifest["trees"]
                if entry.get("source") == vendor_root
                and entry.get("destination") == vendor_root
            )
            excluded = set(vendor_tree.get("exclude", []))
            expected_excludes = {
                destination.removeprefix(vendor_root + "/")
                for destination in PUBLIC_VENDOR_SOURCE_OVERRIDES
            }
            self.assertLessEqual(expected_excludes, excluded)

            file_mappings = {
                (entry.get("source"), entry.get("destination"))
                for entry in development_manifest["files"]
            }
            expected_mappings = {
                (metadata["source"], destination)
                for destination, metadata in PUBLIC_VENDOR_SOURCE_OVERRIDES.items()
            }
            self.assertLessEqual(expected_mappings, file_mappings)

            for destination, metadata in PUBLIC_VENDOR_SOURCE_OVERRIDES.items():
                source_path = REPOSITORY_ROOT / metadata["source"]
                source_bytes = source_path.read_bytes()
                destination_bytes = (REPOSITORY_ROOT / destination).read_bytes()
                self.assertEqual(
                    hashlib.sha256(source_bytes).hexdigest(),
                    metadata["sha256"],
                )
                self.assertNotEqual(source_bytes, destination_bytes)
        else:
            for metadata in PUBLIC_VENDOR_SOURCE_OVERRIDES.values():
                self.assertFalse((REPOSITORY_ROOT / metadata["source"]).exists())

        self.assertEqual(
            set(public_manifest.get("publicIndex", {})),
            {"pathCount", "pathModeSha256"},
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "public"
            reexport = Path(directory) / "reexport"
            output.mkdir()
            export_public_repository.export(
                REPOSITORY_ROOT,
                output,
                allow_development_manifest=has_development_manifest,
            )

            self.assertFalse((output / "public/sanitized-projection").exists())
            internal_family_marker = "Qwen" + "4Exp"
            internal_family_paths = sorted(
                str(path.relative_to(output))
                for path in output.rglob(f"*{internal_family_marker}*")
            )
            self.assertEqual(internal_family_paths, [])
            internal_family_content = sorted(
                str(path.relative_to(output))
                for path in output.rglob("*")
                if path.is_file()
                and internal_family_marker.encode("utf-8") in path.read_bytes()
            )
            self.assertEqual(internal_family_content, [])
            for destination, metadata in PUBLIC_VENDOR_SOURCE_OVERRIDES.items():
                output_bytes = (output / destination).read_bytes()
                self.assertEqual(
                    hashlib.sha256(output_bytes).hexdigest(),
                    metadata["sha256"],
                )
                if has_development_manifest:
                    source_bytes = (REPOSITORY_ROOT / metadata["source"]).read_bytes()
                    self.assertEqual(output_bytes, source_bytes)
                else:
                    checkout_bytes = (REPOSITORY_ROOT / destination).read_bytes()
                    self.assertEqual(output_bytes, checkout_bytes)

            subprocess.run(["git", "init", "-q"], cwd=output, check=True)
            subprocess.run(["git", "add", "."], cwd=output, check=True)
            reexport_count = export_public_repository.export(output, reexport)
            self.assertEqual(reexport_count, 839)
            for destination, metadata in PUBLIC_VENDOR_SOURCE_OVERRIDES.items():
                output_bytes = (output / destination).read_bytes()
                reexport_bytes = (reexport / destination).read_bytes()
                self.assertEqual(reexport_bytes, output_bytes)
                self.assertEqual(
                    hashlib.sha256(reexport_bytes).hexdigest(),
                    metadata["sha256"],
                )
            self.assertFalse((reexport / "public/sanitized-projection").exists())

    def test_export_copies_only_indexed_allowlist_and_published_articles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            output = Path(directory) / "output"
            root.mkdir()
            output.mkdir()
            self.make_fixture(root)
            (root / "README.md").write_text("# Unstaged content\n", encoding="utf-8")

            count = export_public_repository.export(
                root,
                output,
                allow_development_manifest=True,
            )

            self.assertEqual(count, 5)
            self.assertTrue((output / "README.md").is_file())
            self.assertTrue((output / "site/publications.json").is_file())
            self.assertTrue((output / "site/assets/site.css").is_file())
            self.assertTrue(
                (output / "docs/content/2026-08-06-note.md").is_file()
            )
            self.assertFalse((output / "untracked-secret.txt").exists())
            self.assertTrue((output / "public/public-repository.json").is_file())
            self.assertEqual((output / "README.md").read_text(), "# Fixture\n")

    def test_development_projection_is_strictly_self_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            output = Path(directory) / "output"
            reexport = Path(directory) / "reexport"
            root.mkdir()
            output.mkdir()
            reexport.mkdir()
            self.make_fixture(root)

            export_public_repository.export(
                root,
                output,
                allow_development_manifest=True,
            )
            subprocess.run(["git", "init", "-q"], cwd=output, check=True)
            subprocess.run(["git", "add", "."], cwd=output, check=True)

            count = export_public_repository.export(output, reexport)

            self.assertEqual(count, 5)
            expected = {
                path.relative_to(output).as_posix(): (
                    path.read_bytes(),
                    path.stat().st_mode & 0o111,
                )
                for path in output.rglob("*")
                if path.is_file() and ".git" not in path.relative_to(output).parts
            }
            actual = {
                path.relative_to(reexport).as_posix(): (
                    path.read_bytes(),
                    path.stat().st_mode & 0o111,
                )
                for path in reexport.rglob("*")
                if path.is_file()
            }
            self.assertEqual(actual, expected)

    def test_development_manifest_requires_explicit_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            output = Path(directory) / "output"
            root.mkdir()
            self.make_fixture(root)

            with self.assertRaisesRegex(SystemExit, "must use identity paths"):
                export_public_repository.export(root, output)

    def test_development_mode_requires_indexed_identity_manifest_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            output = Path(directory) / "output"
            root.mkdir()
            (root / "README.md").write_text("# Fixture\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "add", "README.md"], cwd=root, check=True)

            with self.assertRaisesRegex(
                SystemExit,
                "--development-projection requires indexed",
            ):
                export_public_repository.export(
                    root,
                    output,
                    allow_development_manifest=True,
                )

    def test_nonempty_output_is_refused_without_deleting_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            output = Path(directory) / "output"
            root.mkdir()
            output.mkdir()
            marker = output / "preserve.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(SystemExit):
                export_public_repository.prepare_output(output, root)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_output_inside_engineering_checkout_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            with self.assertRaises(SystemExit):
                export_public_repository.prepare_output(root / "public-output", root)

    def test_published_article_source_cannot_traverse(self) -> None:
        manifest = {
            "schemaVersion": 1,
            "articles": [{"source": "docs/content/../../README.md"}],
        }
        with self.assertRaises(SystemExit):
            export_public_repository.article_pairs(manifest, {"README.md": "100644"})

    def test_destination_cannot_target_git_metadata(self) -> None:
        manifest = {
            "schemaVersion": 1,
            "files": [
                {
                    "source": "README.md",
                    "destination": ".git/hooks/post-checkout",
                }
            ],
            "trees": [],
        }

        with self.assertRaisesRegex(SystemExit, "Git metadata"):
            export_public_repository.manifest_pairs(
                manifest, {"README.md": "100644"}
            )

    def test_destination_cannot_target_private_public_path(self) -> None:
        private_destination = "docs/" + "superpowers/verdicts/private.md"
        manifest = {
            "schemaVersion": 1,
            "files": [
                {
                    "source": "README.md",
                    "destination": private_destination,
                }
            ],
            "trees": [],
        }

        with self.assertRaisesRegex(SystemExit, "private public path"):
            export_public_repository.manifest_pairs(
                manifest,
                {"README.md": "100644"},
                allow_remapped_manifest=True,
            )

    def test_expanded_tree_destination_cannot_target_private_public_path(self) -> None:
        source = "safe/.github/workflows/ci.yml"
        manifest = {
            "schemaVersion": 1,
            "files": [],
            "trees": [
                {
                    "source": "safe",
                    "destination": "spike/Vendor/mlx-swift-lm",
                }
            ],
        }

        with self.assertRaisesRegex(SystemExit, "private public path"):
            export_public_repository.manifest_pairs(
                manifest,
                {source: "100644"},
                allow_remapped_manifest=True,
            )

    def test_development_source_cannot_read_private_public_path(self) -> None:
        private_source = "docs/" + "superpowers/private.md"
        vendored_github_source = (
            "spike/Vendor/mlx-swift-lm/.github/ISSUE_TEMPLATE/bug_report.md"
        )
        cases = (
            (
                {
                    "schemaVersion": 1,
                    "files": [
                        {
                            "source": private_source,
                            "destination": "docs/content/README.md",
                        }
                    ],
                    "trees": [],
                },
                {private_source: "100644"},
            ),
            (
                {
                    "schemaVersion": 1,
                    "files": [],
                    "trees": [
                        {
                            "source": "docs/" + "superpowers",
                            "destination": "docs/content",
                        }
                    ],
                },
                {private_source: "100644"},
            ),
            (
                {
                    "schemaVersion": 1,
                    "files": [
                        {
                            "source": vendored_github_source,
                            "destination": "docs/content/README.md",
                        }
                    ],
                    "trees": [],
                },
                {vendored_github_source: "100644"},
            ),
            (
                {
                    "schemaVersion": 1,
                    "files": [],
                    "trees": [
                        {
                            "source": "spike/Vendor/mlx-swift-lm/.github",
                            "destination": "docs/content",
                        }
                    ],
                },
                {vendored_github_source: "100644"},
            ),
        )

        for manifest, tracked in cases:
            with self.subTest(manifest=manifest):
                with self.assertRaisesRegex(SystemExit, "private public source"):
                    export_public_repository.manifest_pairs(
                        manifest,
                        tracked,
                        allow_remapped_manifest=True,
                    )

    def test_destination_must_use_canonical_relative_spelling(self) -> None:
        manifest = {
            "schemaVersion": 1,
            "files": [
                {
                    "source": "README.md",
                    "destination": "docs/./content/README.md",
                }
            ],
            "trees": [],
        }

        with self.assertRaisesRegex(SystemExit, "canonical relative path"):
            export_public_repository.manifest_pairs(
                manifest, {"README.md": "100644"}
            )

    def test_tree_exclude_omits_nested_metadata(self) -> None:
        manifest = {
            "schemaVersion": 1,
            "files": [],
            "trees": [
                {
                    "source": "vendor",
                    "destination": "public-vendor",
                    "exclude": [".github"],
                }
            ],
        }
        tracked = {
            "vendor/.github/workflows/test.yml": "100644",
            "vendor/Sources/Library.swift": "100644",
        }
        self.assertEqual(
            export_public_repository.manifest_pairs(
                manifest,
                tracked,
                allow_remapped_manifest=True,
            ),
            [
                (
                    "vendor/Sources/Library.swift",
                    "public-vendor/Sources/Library.swift",
                )
            ],
        )

    def test_tree_exclude_cannot_traverse(self) -> None:
        manifest = {
            "schemaVersion": 1,
            "files": [],
            "trees": [
                {
                    "source": "vendor",
                    "destination": "vendor",
                    "exclude": ["../private"],
                }
            ],
        }
        with self.assertRaises(SystemExit):
            export_public_repository.manifest_pairs(
                manifest, {"vendor/Sources/Library.swift": "100644"}
            )

    def test_manifest_rejects_unknown_top_level_key(self) -> None:
        manifest = {
            "schemaVersion": 1,
            "files": [],
            "trees": [],
            "file": [],
        }

        with self.assertRaisesRegex(SystemExit, "unknown keys"):
            export_public_repository.manifest_pairs(manifest, {})

    def test_tree_manifest_rejects_unknown_key(self) -> None:
        manifest = {
            "schemaVersion": 1,
            "files": [],
            "trees": [
                {
                    "source": "vendor",
                    "destination": "vendor",
                    "excludes": [".github"],
                }
            ],
        }

        with self.assertRaisesRegex(SystemExit, "unknown keys"):
            export_public_repository.manifest_pairs(
                manifest, {"vendor/Sources/Library.swift": "100644"}
            )

    def test_identity_manifest_requires_exact_public_index(self) -> None:
        expected = {"README.md": "100644"}
        manifest = {
            "schemaVersion": 1,
            "publicIndex": public_index_seal(expected),
            "files": [{"source": "README.md", "destination": "README.md"}],
            "trees": [],
        }

        with self.assertRaisesRegex(SystemExit, "public index seal"):
            export_public_repository.manifest_pairs(
                manifest,
                {
                    "README.md": "100644",
                    "development-only.md": "100644",
                },
            )

    def test_identity_manifest_requires_public_index_seal(self) -> None:
        manifest = {
            "schemaVersion": 1,
            "files": [{"source": "README.md", "destination": "README.md"}],
            "trees": [],
        }

        with self.assertRaisesRegex(SystemExit, "requires publicIndex"):
            export_public_repository.manifest_pairs(
                manifest,
                {"README.md": "100644"},
            )

    def test_identity_manifest_seal_rejects_extra_tracked_tree_member(self) -> None:
        expected = {
            "public/public-repository.json": "100644",
            "site/index.html": "100644",
        }
        manifest = {
            "schemaVersion": 1,
            "publicIndex": public_index_seal(expected),
            "files": [
                {
                    "source": "public/public-repository.json",
                    "destination": "public/public-repository.json",
                }
            ],
            "trees": [{"source": "site", "destination": "site"}],
        }
        tracked = dict(expected)
        tracked["site/ambient-extra.txt"] = "100644"

        with self.assertRaisesRegex(SystemExit, "public index seal"):
            export_public_repository.manifest_pairs(manifest, tracked)

    def test_identity_manifest_seal_rejects_mode_drift(self) -> None:
        expected = {"scripts/tool.py": "100644"}
        manifest = {
            "schemaVersion": 1,
            "publicIndex": public_index_seal(expected),
            "files": [
                {
                    "source": "scripts/tool.py",
                    "destination": "scripts/tool.py",
                }
            ],
            "trees": [],
        }

        with self.assertRaisesRegex(SystemExit, "public index seal"):
            export_public_repository.manifest_pairs(
                manifest,
                {"scripts/tool.py": "100755"},
            )

    def test_public_checkout_manifest_rejects_remapped_paths(self) -> None:
        manifest = {
            "schemaVersion": 1,
            "files": [
                {
                    "source": "README.md",
                    "destination": "docs/README.md",
                }
            ],
            "trees": [],
        }

        with self.assertRaisesRegex(SystemExit, "must use identity paths"):
            export_public_repository.manifest_pairs(
                manifest,
                {"README.md": "100644"},
            )

    def test_file_manifest_rejects_public_source_key(self) -> None:
        manifest = {
            "schemaVersion": 1,
            "files": [
                {
                    "source": "public/docs-content-README.md",
                    "destination": "docs/content/README.md",
                    "publicSource": "docs/content/README.md",
                }
            ],
            "trees": [],
        }

        with self.assertRaisesRegex(SystemExit, "unknown keys"):
            export_public_repository.manifest_pairs(
                manifest, {"public/docs-content-README.md": "100644"}
            )

    def test_article_overlap_refuses_before_copying_any_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            output = Path(directory) / "output"
            root.mkdir()
            output.mkdir()
            self.make_fixture(root)
            manifest_path = root / "public/public-repository.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"].append(
                {
                    "source": "docs/content/2026-08-06-note.md",
                    "destination": "docs/content/2026-08-06-note.md",
                }
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            subprocess.run(
                ["git", "add", "public/public-repository.json"],
                cwd=root,
                check=True,
            )

            with self.assertRaisesRegex(
                SystemExit,
                "duplicate destination across public manifests",
            ):
                export_public_repository.export(
                    root,
                    output,
                    allow_development_manifest=True,
                )

            self.assertEqual(list(output.iterdir()), [])

if __name__ == "__main__":
    unittest.main()
