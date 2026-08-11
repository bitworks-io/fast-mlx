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
