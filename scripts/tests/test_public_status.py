from __future__ import annotations

import html
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import build_public_site  # noqa: E402
import validate_public_site  # noqa: E402


class PublicStatusPageTests(unittest.TestCase):
    def test_status_page_is_generated_from_reviewed_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)

            status_path = output / "status/index.html"
            self.assertTrue(
                status_path.is_file(),
                "missing generated current-state status page: status/index.html",
            )

            capabilities = json.loads(
                (REPOSITORY_ROOT / "site/capabilities.json").read_text(
                    encoding="utf-8"
                )
            )
            releases = json.loads(
                (REPOSITORY_ROOT / "site/releases.json").read_text(
                    encoding="utf-8"
                )
            )
            publications = json.loads(
                (REPOSITORY_ROOT / "site/publications.json").read_text(
                    encoding="utf-8"
                )
            )
            page = status_path.read_text(encoding="utf-8")
            latest = releases["releases"][0]
            boundary = releases["currentBoundary"]
            status_counts = {
                status: sum(
                    capability["status"] == status
                    for capability in capabilities["capabilities"]
                )
                for status in (
                    "implemented",
                    "promoted-scoped",
                    "experimental",
                    "shelved",
                )
            }

            self.assertIn(
                "<div data-status-page "
                f'data-latest-release-id="{latest["id"]}" '
                f'data-boundary-id="{boundary["id"]}" '
                f'data-boundary-state="{boundary["state"]}">',
                page,
            )
            self.assertIn("Current state, not a roadmap.", page)
            self.assertIn(
                "does not create new measurement, performance, model, runtime, "
                "acquisition, or publication authority",
                page,
            )
            self.assertIn(latest["title"], page)
            self.assertIn(latest["summary"], page)
            self.assertIn(latest["scope"], page)
            self.assertIn(latest["publishedAt"], page)
            self.assertIn(boundary["label"], page)
            self.assertIn(boundary["summary"], page)
            self.assertIn(
                f'{len(capabilities["capabilities"])} reviewed capabilities',
                page,
            )
            self.assertIn(
                f'{len(capabilities["performanceHighlights"])} measured proof points',
                page,
            )
            self.assertIn(
                f'{len(publications["articles"])} published research notes',
                page,
            )
            self.assertIn(
                f'{len(releases["releases"])} reviewed release records',
                page,
            )
            for status, count in status_counts.items():
                self.assertIn(
                    f'data-status-count="{status}" data-count="{count}"',
                    page,
                )

            for capability in capabilities["capabilities"]:
                self.assertIn(
                    f'data-status-capability="{capability["id"]}" '
                    f'data-capability-state="{capability["status"]}"',
                    page,
                )
                for field in ("name", "summary", "scope"):
                    self.assertIn(html.escape(capability[field]), page)

            for highlight in capabilities["performanceHighlights"]:
                self.assertIn(
                    f'data-status-highlight="{highlight["id"]}" '
                    f'data-highlight-decision="{highlight["decision"]}"',
                    page,
                )
                for field in (
                    "metric",
                    "label",
                    "model",
                    "hardware",
                    "workload",
                    "date",
                    "caveat",
                ):
                    self.assertIn(html.escape(highlight[field]), page)

            self.assertIn(
                '<a href="../status/" aria-current="page">Status</a>',
                page,
            )
            self.assertEqual(page.count('aria-current="page"'), 1)
            self.assertIn(
                '<link rel="canonical" '
                'href="https://bitworks-io.github.io/fast-mlx/status/">',
                page,
            )
            self.assertNotIn("<script", page)
            for href in (
                "../quickstart/",
                "../capabilities/",
                "../benchmarks/",
                "../releases/",
                "../research/",
                "../methodology/",
            ):
                self.assertIn(f'href="{href}"', page)

            sitemap = (output / "sitemap.xml").read_text(encoding="utf-8")
            self.assertIn(
                "https://bitworks-io.github.io/fast-mlx/status/",
                sitemap,
            )
            llms = (output / "llms.txt").read_text(encoding="utf-8")
            self.assertIn("- /status/: reviewed current-state dashboard", llms)
            self.assertEqual(validate_public_site.validate(output), [])

            original_page = page
            mutations = (
                (
                    "authority insertion",
                    "Current state, not a roadmap.",
                    "Current state, not a roadmap. Model promotion is automatic.",
                    "status/index.html does not match the reviewed page seal",
                ),
                (
                    "hidden root",
                    "<div data-status-page ",
                    '<div hidden data-status-page ',
                    "status page root attributes do not match reviewed contract",
                ),
                (
                    "active content",
                    "</body>",
                    '<script src="../assets/research-explorer.js"></script></body>',
                    "status page must not contain scripts or inline styles",
                ),
                (
                    "status drift",
                    'data-status-count="implemented" data-count="3"',
                    'data-status-count="implemented" data-count="99"',
                    "status/index.html does not match the reviewed page seal",
                ),
            )
            for label, before, after, expected_failure in mutations:
                with self.subTest(label=label):
                    self.assertEqual(original_page.count(before), 1)
                    status_path.write_text(
                        original_page.replace(before, after, 1),
                        encoding="utf-8",
                    )
                    self.assertTrue(
                        any(
                            failure.startswith(expected_failure)
                            for failure in validate_public_site.validate(output)
                        )
                    )
                    status_path.write_text(original_page, encoding="utf-8")

            process_path = output / "process/index.html"
            process_page = process_path.read_text(encoding="utf-8")
            status_nav = '<a href="../status/">Status</a>'
            self.assertEqual(process_page.count(status_nav), 1)
            process_path.write_text(
                process_page.replace(
                    status_nav,
                    '<a href="../capabilities/">Status</a>',
                    1,
                ),
                encoding="utf-8",
            )
            self.assertIn(
                "process/index.html status navigation does not match reviewed contract",
                validate_public_site.validate(output),
            )
            process_path.write_text(process_page, encoding="utf-8")

            manifest_mutations = (
                (
                    "capability manifest",
                    "capabilities.json",
                    lambda manifest: manifest["capabilities"][0].__setitem__(
                        "summary", "Jointly drifted status-page authority claim."
                    ),
                ),
                (
                    "release manifest",
                    "releases.json",
                    lambda manifest: manifest["releases"][0].__setitem__(
                        "summary", "Jointly drifted reviewed-release claim."
                    ),
                ),
                (
                    "publication manifest",
                    "publications.json",
                    lambda manifest: manifest["articles"].pop(2),
                ),
            )
            for label, filename, mutate in manifest_mutations:
                with self.subTest(label=label):
                    source_root = Path(directory) / ("source-" + filename)
                    shutil.copytree(
                        REPOSITORY_ROOT / "site", source_root / "site"
                    )
                    shutil.copytree(
                        REPOSITORY_ROOT / "docs/content",
                        source_root / "docs/content",
                    )
                    manifest_path = source_root / "site" / filename
                    drifted = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    mutate(drifted)
                    manifest_path.write_text(
                        json.dumps(drifted, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    drifted_output = Path(directory) / ("drifted-" + filename)
                    drifted_output.mkdir()
                    build_public_site.build_site(source_root, drifted_output)
                    self.assertIn(
                        "status/index.html does not match the reviewed page seal",
                        validate_public_site.validate(drifted_output),
                    )


if __name__ == "__main__":
    unittest.main()
