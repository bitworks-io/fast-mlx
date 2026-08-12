from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import build_public_site  # noqa: E402
import validate_public_site  # noqa: E402


class PublicSiteTests(unittest.TestCase):
    @staticmethod
    def capability_manifest() -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "policy": "fast-mlx-owned-results-only",
            "claimBoundary": "fast-mlx-owned-results-only",
            "updatedAt": "2026-08-11",
            "capabilities": [
                {
                    "id": "example-capability",
                    "name": "Example capability",
                    "status": "implemented",
                    "summary": "A bounded example capability for manifest validation.",
                    "scope": "An example only; it makes no broad support claim.",
                    "evidenceSlugs": ["published-note"],
                }
            ],
            "performanceHighlights": [
                {
                    "id": "example-highlight",
                    "metric": "+1.0%",
                    "label": "Example measured result",
                    "model": "Example model",
                    "hardware": "Example hardware",
                    "workload": "Example workload",
                    "date": "2026-08-11",
                    "decision": "promoted-scoped",
                    "caveat": "Example-only result with no general performance guarantee.",
                    "evidenceSlug": "published-note",
                }
            ],
        }

    @staticmethod
    def write_capability_manifest(root: Path, manifest: dict[str, object]) -> None:
        path = root / "site/capabilities.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_manifest_is_explicit_and_bounded(self) -> None:
        manifest = json.loads(
            (REPOSITORY_ROOT / "site/publications.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["schemaVersion"], 1)
        self.assertEqual(manifest["policy"], "fast-mlx-owned-results-only")
        self.assertEqual(len(manifest["articles"]), 7)
        self.assertEqual(
            len({entry["source"] for entry in manifest["articles"]}), 7
        )
        self.assertEqual(
            len({entry["slug"] for entry in manifest["articles"]}), 7
        )

    def test_capability_manifest_is_explicit_and_evidence_backed(self) -> None:
        articles = build_public_site.load_articles(REPOSITORY_ROOT)
        catalog = build_public_site.load_capability_catalog(
            REPOSITORY_ROOT, {article.slug for article in articles}
        )
        self.assertEqual(catalog["schemaVersion"], 1)
        self.assertEqual(catalog["policy"], "fast-mlx-owned-results-only")
        self.assertEqual(catalog["claimBoundary"], "fast-mlx-owned-results-only")
        self.assertEqual(len(catalog["capabilities"]), 6)
        self.assertEqual(len(catalog["performanceHighlights"]), 3)

    def test_unpublished_capability_evidence_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.capability_manifest()
            manifest["capabilities"][0]["evidenceSlugs"] = ["private-diagnostic"]
            self.write_capability_manifest(root, manifest)
            with self.assertRaises(SystemExit):
                build_public_site.load_capability_catalog(root, {"published-note"})

    def test_unknown_capability_status_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.capability_manifest()
            manifest["capabilities"][0]["status"] = "production-default"
            self.write_capability_manifest(root, manifest)
            with self.assertRaises(SystemExit):
                build_public_site.load_capability_catalog(root, {"published-note"})

    def test_performance_highlight_requires_scope_fields(self) -> None:
        for missing in ("model", "hardware", "workload", "date", "caveat"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.capability_manifest()
                del manifest["performanceHighlights"][0][missing]
                self.write_capability_manifest(root, manifest)
                with self.assertRaises(SystemExit):
                    build_public_site.load_capability_catalog(root, {"published-note"})

    def test_build_is_complete_and_links_are_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            articles = build_public_site.build_site(REPOSITORY_ROOT, output)
            build_public_site.scan_generated_output(output)
            self.assertEqual(len(articles), 7)
            self.assertEqual(validate_public_site.validate(output), [])

            public_index = json.loads(
                (output / "research/index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(public_index["articles"]), 7)
            self.assertEqual(
                public_index["claimBoundary"], "fast-mlx-owned-results-only"
            )

            capabilities = json.loads(
                (output / "capabilities/index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(capabilities["schemaVersion"], 1)
            self.assertEqual(
                capabilities["claimBoundary"], "fast-mlx-owned-results-only"
            )
            self.assertEqual(len(capabilities["capabilities"]), 6)
            self.assertEqual(len(capabilities["performanceHighlights"]), 3)
            for capability in capabilities["capabilities"]:
                for evidence in capability["evidence"]:
                    self.assertTrue((output / evidence["path"] / "index.html").is_file())
            for highlight in capabilities["performanceHighlights"]:
                self.assertTrue(
                    (output / highlight["evidence"]["path"] / "index.html").is_file()
                )

            capability_page = (output / "capabilities/index.html").read_text(
                encoding="utf-8"
            )
            self.assertIn("Capabilities &amp; evidence", capability_page)
            self.assertIn("+100.5%", capability_page)
            self.assertIn("Qwen3-32B-4bit", capability_page)
            self.assertIn("SHELVED", capability_page)
            self.assertIn("github.com/bitworks-io/fast-mlx", capability_page)
            self.assertIn(
                '<a href="../capabilities/" aria-current="page">Capabilities</a>',
                capability_page,
            )
            self.assertEqual(capability_page.count('aria-current="page"'), 1)
            self.assertIn('<ul class="status-legend"', capability_page)

            process_page = (output / "process/index.html").read_text(encoding="utf-8")
            self.assertIn(
                '<a href="../process/" aria-current="page">The loop</a>',
                process_page,
            )
            self.assertEqual(process_page.count('aria-current="page"'), 1)

            llms = (output / "llms.txt").read_text(encoding="utf-8")
            self.assertIn("/capabilities/", llms)
            self.assertIn("/capabilities/index.json", llms)

    def test_validator_requires_capability_outputs(self) -> None:
        for missing in ("capabilities/index.html", "capabilities/index.json"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                (output / missing).unlink()
                self.assertIn(
                    f"missing required file: {missing}",
                    validate_public_site.validate(output),
                )

    def test_multiline_whitepaper_theme_is_consumed_once(self) -> None:
        source = (
            REPOSITORY_ROOT
            / "docs/content/2026-07-28-the-proof-did-not-end-when-the-timer-did.md"
        )
        title, date, theme, summary, body = build_public_site.infer_metadata(
            source, source.read_text(encoding="utf-8")
        )
        self.assertEqual(title, "The proof did not end when the timer did")
        self.assertEqual(date, "2026-07-28")
        self.assertEqual(
            theme,
            "Building a high-performance MLX inference engine in Swift; Serving big "
            "models on Apple Silicon; Rapid research integration — the flywheel",
        )
        self.assertTrue(summary.startswith("A short benchmark can show"))

        rendered = build_public_site.render_markdown(body, {}, "research/example/index.html")
        self.assertEqual(rendered.count('class="eyebrow"'), 1)
        self.assertNotIn("<p>models on Apple Silicon", rendered)
        self.assertIn("A short benchmark can show", rendered)

    def test_unpublished_evidence_link_is_not_emitted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)
            article = (
                output
                / "research/the-fastest-request-wasnt-the-fastest-service/index.html"
            ).read_text(encoding="utf-8")
            self.assertNotIn("../superpowers/", article)
            self.assertIn("reviewed evidence ledger", article)

    def test_generated_table_wrapper_has_responsive_style(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)
            article = (
                output
                / "research/when-zero-speculation-costs-two-percent/index.html"
            ).read_text(encoding="utf-8")
            stylesheet = (output / "assets/site.css").read_text(encoding="utf-8")
            self.assertIn('class="table-scroll"', article)
            self.assertIn(".table-scroll", stylesheet)

    def test_inline_renderer_escapes_article_html_and_unsafe_links(self) -> None:
        renderer = build_public_site.InlineRenderer({}, "research/example/index.html")
        rendered = renderer('<script>alert(1)</script> [bad](javascript:alert(1))')
        self.assertIn("&lt;script&gt;", rendered)
        self.assertNotIn("<script>", rendered)
        self.assertNotIn("javascript:", rendered)
        self.assertIn("source-boundary", rendered)

    def test_nonempty_output_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            (output / "owned.txt").write_text("preserve", encoding="utf-8")
            with self.assertRaises(SystemExit):
                build_public_site.prepare_output(output, REPOSITORY_ROOT)
            self.assertEqual((output / "owned.txt").read_text(), "preserve")

    def test_output_inside_repository_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            with self.assertRaises(SystemExit):
                build_public_site.prepare_output(root / "generated-site", root)

    def test_asset_symlink_is_refused_before_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "site/assets"
            assets.mkdir(parents=True)
            source = root / "private.txt"
            source.write_text("private", encoding="utf-8")
            (assets / "leak.txt").symlink_to(source)

            with self.assertRaises(SystemExit):
                build_public_site.validate_asset_tree(assets)

    def test_capability_manifest_symlink_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            external = root / "external.json"
            external.write_text(json.dumps(self.capability_manifest()), encoding="utf-8")
            manifest = root / "site/capabilities.json"
            manifest.parent.mkdir(parents=True)
            manifest.symlink_to(external)

            with self.assertRaises(SystemExit):
                build_public_site.load_capability_catalog(root, {"published-note"})

    def test_article_manifest_cannot_traverse_out_of_content_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "site/publications.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "policy": "fast-mlx-owned-results-only",
                        "articles": [
                            {
                                "source": "docs/content/../../README.md",
                                "slug": "escaped",
                                "status": "published",
                                "reviewedAt": "2026-08-06",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (root / "README.md").write_text("# Private\n", encoding="utf-8")

            with self.assertRaises(SystemExit):
                build_public_site.load_articles(root)


if __name__ == "__main__":
    unittest.main()
