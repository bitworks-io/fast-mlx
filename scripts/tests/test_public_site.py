from __future__ import annotations

import html.parser
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import build_public_site  # noqa: E402
import validate_public_site  # noqa: E402


class HeadMetadataCollector(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.canonicals: list[str] = []
        self.properties: dict[str, list[str]] = {}

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if tag == "link" and attributes.get("rel") == "canonical":
            href = attributes.get("href")
            if href is not None:
                self.canonicals.append(href)
        if tag == "meta" and attributes.get("property"):
            name = attributes["property"]
            content = attributes.get("content")
            if content is not None:
                self.properties.setdefault(name, []).append(content)


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

    @staticmethod
    def release_manifest() -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "project": "fast-mlx",
            "policy": "reviewed-public-releases-only",
            "claimBoundary": "fast-mlx-owned-results-only",
            "updatedAt": "2026-08-11",
            "currentBoundary": {
                "id": "runtime-model-promotion",
                "label": "Runtime and model promotion",
                "state": "gated",
                "summary": "Reviewed source does not grant runtime authority.",
                "evidence": {
                    "label": "Read the methodology",
                    "path": "methodology/",
                },
            },
            "releases": [
                {
                    "id": "newer-release",
                    "title": "Newer reviewed release",
                    "publishedAt": "2026-08-11T22:00:00-05:00",
                    "category": "product",
                    "state": "released",
                    "summary": "Adds a bounded public surface.",
                    "scope": "No runtime or benchmark authority changes.",
                    "publicCommit": "1" * 40,
                    "publicLinks": [
                        {"label": "Open capabilities", "path": "capabilities/"}
                    ],
                },
                {
                    "id": "older-release",
                    "title": "Older reviewed release",
                    "publishedAt": "2026-08-11T21:00:00-05:00",
                    "category": "foundation",
                    "state": "released",
                    "summary": "Establishes the public foundation.",
                    "scope": "Only reviewed public source is included.",
                    "publicCommit": "2" * 40,
                    "publicLinks": [],
                },
            ],
        }

    @staticmethod
    def write_release_manifest(root: Path, manifest: dict[str, object]) -> None:
        path = root / "site/releases.json"
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

    def test_release_manifest_is_explicit_newest_first_and_non_circular(self) -> None:
        catalog = build_public_site.load_release_catalog(REPOSITORY_ROOT)
        self.assertEqual(catalog["schemaVersion"], 1)
        self.assertEqual(catalog["project"], "fast-mlx")
        self.assertEqual(catalog["policy"], "reviewed-public-releases-only")
        self.assertEqual(catalog["claimBoundary"], "fast-mlx-owned-results-only")
        self.assertEqual(catalog["currentBoundary"]["state"], "gated")
        self.assertEqual(
            catalog["releases"][0],
            {
                "id": "reviewed-social-metadata",
                "title": "Publish reviewed social metadata",
                "publishedAt": "2026-08-12T02:08:02-05:00",
                "category": "product",
                "state": "released",
                "summary": (
                    "Adds exact self-canonical and Open Graph presentation metadata "
                    "to the fourteen reviewed human-facing pages."
                ),
                "scope": (
                    "Presentation and crawl-discovery metadata only; no tracking, "
                    "indexing guarantee, benchmark evidence, runtime authority, or "
                    "positive absorbed-MLA admission."
                ),
                "publicCommit": "1903e76609cc444bdacfaa5d0472804900dbd13c",
                "publicLinks": [
                    {"label": "Open fast-mlx", "path": "index.html"},
                    {"label": "Read the methodology", "path": "methodology/"},
                ],
            },
        )
        commits = [entry["publicCommit"] for entry in catalog["releases"]]
        self.assertEqual(
            commits,
            [
                "1903e76609cc444bdacfaa5d0472804900dbd13c",
                "df2b067391cec755cb9ec0e6f87097b8c8d6537a",
                "35e751a0a867d187014251b519eebbb17291fd88",
                "195963b912ff287fa4f7b7058c08fdd9b75dcd3d",
                "555514986cdd17ca921c9d9607a92d6248734fdd",
                "3894c324dc69baf428b9fe54d1770a234467dfbd",
                "a39393451848eedc62e035ddee5ef00d0364ca62",
                "a701708fe55311b7ea61db957bf37573f5530ce4",
                "51a99af2b11f42e036f88346a4b8873b71f675d5",
            ],
        )
        self.assertEqual(len(commits), len(set(commits)))

    def test_invalid_release_contract_is_refused(self) -> None:
        mutations = {
            "unknown top-level key": lambda manifest: manifest.update({"private": True}),
            "wrong project": lambda manifest: manifest.update({"project": "other"}),
            "wrong policy": lambda manifest: manifest.update({"policy": "automatic"}),
            "wrong claim boundary": lambda manifest: manifest.update(
                {"claimBoundary": "all-results"}
            ),
            "unreviewed boundary state": lambda manifest: manifest["currentBoundary"].update(
                {"state": "promoted"}
            ),
            "unknown release category": lambda manifest: manifest["releases"][0].update(
                {"category": "benchmark-winner"}
            ),
            "unreleased release": lambda manifest: manifest["releases"][0].update(
                {"state": "draft"}
            ),
            "short commit": lambda manifest: manifest["releases"][0].update(
                {"publicCommit": "deadbeef"}
            ),
            "naive timestamp": lambda manifest: manifest["releases"][0].update(
                {"publishedAt": "2026-08-11T22:00:00"}
            ),
            "external public path": lambda manifest: manifest["releases"][0][
                "publicLinks"
            ][0].update({"path": "https://example.invalid/private"}),
            "traversing public path": lambda manifest: manifest["releases"][0][
                "publicLinks"
            ][0].update({"path": ".." + "/" + "private" + "/"}),
            "private marker": lambda manifest: manifest["releases"][0].update(
                {"scope": "/" + "Users" + "/example/private"}
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.release_manifest()
                mutate(manifest)
                self.write_release_manifest(root, manifest)
                with self.assertRaises(SystemExit):
                    build_public_site.load_release_catalog(root)

    def test_duplicate_or_out_of_order_release_contract_is_refused(self) -> None:
        mutations = {
            "duplicate id": lambda manifest: manifest["releases"][1].update(
                {"id": manifest["releases"][0]["id"]}
            ),
            "duplicate commit": lambda manifest: manifest["releases"][1].update(
                {"publicCommit": manifest["releases"][0]["publicCommit"]}
            ),
            "duplicate link": lambda manifest: manifest["releases"][0][
                "publicLinks"
            ].append(dict(manifest["releases"][0]["publicLinks"][0])),
            "not newest first": lambda manifest: manifest["releases"][1].update(
                {"publishedAt": "2026-08-11T23:00:00-05:00"}
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.release_manifest()
                mutate(manifest)
                self.write_release_manifest(root, manifest)
                with self.assertRaises(SystemExit):
                    build_public_site.load_release_catalog(root)

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

    def test_whitespace_padded_capability_values_are_refused(self) -> None:
        for key in ("id", "decision", "evidenceSlug"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.capability_manifest()
                manifest["performanceHighlights"][0][key] = (
                    " " + manifest["performanceHighlights"][0][key]
                )
                self.write_capability_manifest(root, manifest)
                with self.assertRaises(SystemExit):
                    build_public_site.load_capability_catalog(root, {"published-note"})

    def test_home_current_cycle_summary_is_generated_from_reviewed_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)

            capabilities = json.loads(
                (REPOSITORY_ROOT / "site/capabilities.json").read_text(
                    encoding="utf-8"
                )
            )
            publications = json.loads(
                (REPOSITORY_ROOT / "site/publications.json").read_text(
                    encoding="utf-8"
                )
            )
            releases = json.loads(
                (REPOSITORY_ROOT / "site/releases.json").read_text(encoding="utf-8")
            )
            page = (output / "index.html").read_text(encoding="utf-8")
            latest = releases["releases"][0]
            boundary = releases["currentBoundary"]
            status_counts = {
                status: sum(
                    capability["status"] == status
                    for capability in capabilities["capabilities"]
                )
                for status in (
                    "promoted-scoped",
                    "implemented",
                    "experimental",
                    "shelved",
                )
            }

            self.assertIn(
                'data-current-cycle data-latest-release-id="'
                + latest["id"]
                + '" data-boundary-id="'
                + boundary["id"]
                + '" data-boundary-state="gated"',
                page,
            )
            self.assertIn(latest["title"], page)
            self.assertIn(latest["summary"], page)
            self.assertIn(latest["scope"], page)
            self.assertIn(latest["publishedAt"][:10], page)
            self.assertIn(
                'href="releases/#release-' + latest["id"] + '"', page
            )
            self.assertIn(
                "https://github.com/bitworks-io/fast-mlx/commit/"
                + latest["publicCommit"],
                page,
            )
            self.assertIn(boundary["label"], page)
            self.assertIn(boundary["summary"], page)
            self.assertIn('href="methodology/"', page)
            self.assertIn(
                f'{len(capabilities["capabilities"])} reviewed capabilities', page
            )
            self.assertIn(
                f'{len(publications["articles"])} published research notes', page
            )
            self.assertIn(
                f'{len(releases["releases"])} reviewed releases', page
            )
            self.assertIn(
                'class="capability-list" role="list" '
                'aria-label="Current reviewed evidence inventory"',
                page,
            )
            self.assertEqual(page.count('role="listitem"'), 4)
            self.assertIn("</span>; <span", page)
            for status, count in status_counts.items():
                self.assertIn(
                    f'data-capability-status="{status}" data-count="{count}"', page
                )

    def test_validator_rejects_home_current_cycle_tampering(self) -> None:
        releases = json.loads(
            (REPOSITORY_ROOT / "site/releases.json").read_text(encoding="utf-8")
        )
        latest = releases["releases"][0]
        mutations = (
            (
                "boundary state",
                'data-boundary-state="gated"',
                'data-boundary-state="promoted"',
                "home current-cycle boundary state does not match releases/index.json",
            ),
            (
                "release scope",
                latest["scope"],
                "Runtime and model authority granted as the default.",
                "home current-cycle latest release does not bind scope",
            ),
            (
                "additive authority claim",
                latest["scope"],
                latest["scope"]
                + " Runtime and model authority granted as the default.",
                "home current-cycle text does not match reviewed indexes",
            ),
            (
                "visible boundary state",
                ">GATED</span>",
                ">PROMOTED</span>",
                "home current-cycle text does not match reviewed indexes",
            ),
            (
                "hidden section",
                '<section class="section shell split" aria-labelledby="current-cycle-heading" ',
                '<section class="section shell split" aria-labelledby="current-cycle-heading" hidden ',
                "home current-cycle section attributes do not match the reviewed contract",
            ),
            (
                "aria-hidden section",
                '<section class="section shell split" aria-labelledby="current-cycle-heading" ',
                '<section class="section shell split" aria-labelledby="current-cycle-heading" aria-hidden="true" ',
                "home current-cycle section attributes do not match the reviewed contract",
            ),
            (
                "inline-hidden section",
                '<section class="section shell split" aria-labelledby="current-cycle-heading" ',
                '<section class="section shell split" aria-labelledby="current-cycle-heading" style="display:none" ',
                "home current-cycle section attributes do not match the reviewed contract",
            ),
            (
                "duplicate section class",
                '<section class="section shell split" aria-labelledby="current-cycle-heading" ',
                '<section class="benchmark-controls" class="section shell split" aria-labelledby="current-cycle-heading" ',
                "home current-cycle section contains duplicate attributes",
            ),
            (
                "duplicate inventory role",
                '<div class="capability-list" role="list" ',
                '<div class="capability-list" role="presentation" role="list" ',
                "home current-cycle section contains duplicate attributes",
            ),
            (
                "hidden inventory",
                '<div class="capability-list" role="list" ',
                '<div class="capability-list" role="list" hidden ',
                "home current-cycle section contains a visibility suppressor",
            ),
            (
                "aria-hidden card",
                '<article role="listitem">\n<h3>6 reviewed capabilities</h3>',
                '<article role="listitem" aria-hidden="true">\n'
                '<h3>6 reviewed capabilities</h3>',
                "home current-cycle section contains a visibility suppressor",
            ),
            (
                "known hidden class",
                '<div class="capability-list" role="list" ',
                '<div class="capability-list benchmark-controls" role="list" ',
                "home current-cycle section contains a visibility suppressor",
            ),
            (
                "hidden ancestor wrapper",
                '<section class="section shell split" aria-labelledby="current-cycle-heading" ',
                '<div hidden><section class="section shell split" aria-labelledby="current-cycle-heading" ',
                "home current-cycle section ancestry does not match the reviewed contract",
            ),
            (
                "hidden main ancestor",
                '<main id="content">',
                '<main id="content" hidden>',
                "home current-cycle section ancestry does not match the reviewed contract",
            ),
            (
                "nested section bypass",
                '</div>\n</section>\n\n<section class="section shell" '
                'aria-labelledby="loop-heading">',
                '</div>\n<section></section><p>Runtime and model authority granted '
                'as the default.</p></section>\n\n<section class="section shell" '
                'aria-labelledby="loop-heading">',
                "home current-cycle text does not match reviewed indexes",
            ),
            (
                "commit link",
                "https://github.com/bitworks-io/fast-mlx/commit/"
                + latest["publicCommit"],
                "https://github.com/bitworks-io/fast-mlx/commit/" + "0" * 40,
                "home current-cycle latest release has the wrong source link",
            ),
            (
                "capability count",
                'data-capability-status="implemented" data-count="3"',
                'data-capability-status="implemented" data-count="99"',
                "home current-cycle capability status counts do not match capabilities/index.json",
            ),
            (
                "research count",
                "7 published research notes",
                "99 published research notes",
                "home current-cycle research count does not match research/index.json",
            ),
        )
        for label, original, replacement, expected_failure in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                home_path = output / "index.html"
                page = home_path.read_text(encoding="utf-8")
                self.assertEqual(page.count(original), 1)
                home_path.write_text(
                    page.replace(original, replacement, 1), encoding="utf-8"
                )
                self.assertIn(expected_failure, validate_public_site.validate(output))

    def test_validator_rejects_home_current_cycle_stylesheet_visibility_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)
            stylesheet = output / "assets/site.css"
            stylesheet.write_text(
                stylesheet.read_text(encoding="utf-8")
                + "\n[data-current-cycle] { display: none; }\n",
                encoding="utf-8",
            )
            self.assertIn(
                "assets/site.css does not match the reviewed stylesheet",
                validate_public_site.validate(output),
            )

    def test_validator_rejects_home_current_cycle_page_level_visibility_tampering(self) -> None:
        mutations = (
            (
                "inline style rule",
                "</head>",
                "<style>[data-current-cycle]{display:none}</style></head>",
            ),
            (
                "additional stylesheet",
                "</head>",
                '<link rel="stylesheet" href="assets/extra.css"></head>',
            ),
            (
                "closed details wrapper",
                '<p class="eyebrow">Current reviewed cycle</p>',
                '<details><summary></summary><p class="eyebrow">Current reviewed cycle</p>',
            ),
            (
                "closed dialog wrapper",
                '<p class="eyebrow">Current reviewed cycle</p>',
                '<dialog><p class="eyebrow">Current reviewed cycle</p>',
            ),
            (
                "inert inventory",
                '<div class="capability-list" role="list" ',
                '<div class="capability-list" role="list" inert ',
            ),
        )
        for label, original, replacement in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                home_path = output / "index.html"
                page = home_path.read_text(encoding="utf-8")
                self.assertEqual(page.count(original), 1)
                home_path.write_text(
                    page.replace(original, replacement, 1), encoding="utf-8"
                )
                self.assertIn(
                    "index.html does not match the reviewed home page",
                    validate_public_site.validate(output),
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

    def test_release_surface_is_generated_from_reviewed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)

            source = json.loads(
                (REPOSITORY_ROOT / "site/releases.json").read_text(encoding="utf-8")
            )
            public_index = json.loads(
                (output / "releases/index.json").read_text(encoding="utf-8")
            )
            page = (output / "releases/index.html").read_text(encoding="utf-8")

            self.assertEqual(public_index["schemaVersion"], 1)
            self.assertEqual(public_index["project"], "fast-mlx")
            self.assertEqual(
                public_index["policy"], "reviewed-public-releases-only"
            )
            self.assertEqual(
                public_index["claimBoundary"], "fast-mlx-owned-results-only"
            )
            self.assertEqual(public_index["currentBoundary"], source["currentBoundary"])
            self.assertEqual(len(public_index["releases"]), len(source["releases"]))
            self.assertEqual(
                [entry["publicCommit"] for entry in public_index["releases"]],
                [entry["publicCommit"] for entry in source["releases"]],
            )
            self.assertIn("Reviewed releases", page)
            self.assertIn("GATED", page)
            self.assertIn(
                'data-boundary-id="runtime-model-promotion" data-boundary-state="gated"',
                page,
            )
            self.assertEqual(
                page.count('class="release-card"'), len(source["releases"])
            )
            for release in source["releases"]:
                self.assertIn(f'data-release-id="{release["id"]}"', page)
                self.assertIn(
                    f'data-public-commit="{release["publicCommit"]}"', page
                )
                self.assertIn(release["title"], page)
                self.assertIn(release["summary"], page)
                self.assertIn(release["scope"], page)
                self.assertIn(
                    "https://github.com/bitworks-io/fast-mlx/commit/"
                    + release["publicCommit"],
                    page,
                )
                self.assertEqual(
                    public_index["releases"][source["releases"].index(release)][
                        "sourceUrl"
                    ],
                    "https://github.com/bitworks-io/fast-mlx/commit/"
                    + release["publicCommit"],
                )
            self.assertIn(
                '<a href="../releases/" aria-current="page">Releases</a>', page
            )
            self.assertEqual(page.count('aria-current="page"'), 1)
            llms = (output / "llms.txt").read_text(encoding="utf-8")
            self.assertIn("/releases/", llms)
            self.assertIn("/releases/index.json", llms)

    def test_release_atom_feed_is_generated_from_reviewed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)

            source = json.loads(
                (REPOSITORY_ROOT / "site/releases.json").read_text(encoding="utf-8")
            )
            public_index = json.loads(
                (output / "releases/index.json").read_text(encoding="utf-8")
            )
            feed_path = output / "releases/feed.atom"
            feed_text = feed_path.read_text(encoding="utf-8")
            feed = ET.fromstring(feed_text)
            atom_namespace = "http://www.w3.org/2005/Atom"
            atom = lambda name: f"{{{atom_namespace}}}{name}"

            self.assertTrue(feed_text.startswith('<?xml version="1.0" encoding="utf-8"?>'))
            self.assertEqual(feed.tag, atom("feed"))
            self.assertEqual(feed.findtext(atom("title")), "fast-mlx reviewed releases")
            self.assertEqual(
                feed.findtext(atom("id")),
                "https://bitworks-io.github.io/fast-mlx/releases/",
            )
            self.assertEqual(
                feed.findtext(atom("updated")), source["releases"][0]["publishedAt"]
            )
            self.assertEqual(
                feed.findtext(f"{atom('author')}/{atom('name')}"),
                "fast-mlx contributors",
            )
            feed_links = {
                link.attrib["rel"]: link.attrib
                for link in feed.findall(atom("link"))
            }
            self.assertEqual(
                feed_links["self"],
                {
                    "rel": "self",
                    "type": "application/atom+xml",
                    "href": "https://bitworks-io.github.io/fast-mlx/releases/feed.atom",
                },
            )
            self.assertEqual(
                feed_links["alternate"],
                {
                    "rel": "alternate",
                    "type": "text/html",
                    "href": "https://bitworks-io.github.io/fast-mlx/releases/",
                },
            )

            entries = feed.findall(atom("entry"))
            self.assertEqual(len(entries), len(source["releases"]))
            for entry, release, public_release in zip(
                entries, source["releases"], public_index["releases"]
            ):
                self.assertEqual(
                    entry.findtext(atom("id")),
                    "urn:fast-mlx:public-commit:" + release["publicCommit"],
                )
                self.assertEqual(entry.findtext(atom("title")), release["title"])
                self.assertEqual(
                    entry.findtext(atom("published")), release["publishedAt"]
                )
                self.assertEqual(
                    entry.findtext(atom("updated")), release["publishedAt"]
                )
                self.assertEqual(
                    entry.find(atom("category")).attrib, {"term": release["category"]}
                )
                self.assertEqual(
                    entry.findtext(atom("summary")),
                    release["summary"] + " Boundary: " + release["scope"],
                )
                links = {
                    link.attrib["rel"]: link.attrib
                    for link in entry.findall(atom("link"))
                }
                self.assertEqual(
                    links["alternate"],
                    {
                        "rel": "alternate",
                        "type": "text/html",
                        "href": "https://bitworks-io.github.io/fast-mlx/releases/#release-"
                        + release["id"],
                    },
                )
                self.assertEqual(
                    links["via"],
                    {"rel": "via", "href": public_release["sourceUrl"]},
                )

            release_page = (output / "releases/index.html").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                'rel="alternate" type="application/atom+xml" '
                'title="fast-mlx reviewed releases" href="../releases/feed.atom"',
                release_page,
            )
            for release in source["releases"]:
                self.assertIn(f'id="release-{release["id"]}"', release_page)
            llms = (output / "llms.txt").read_text(encoding="utf-8")
            self.assertIn("/releases/feed.atom", llms)

    def test_site_discovery_contract_is_generated_from_reviewed_routes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            articles = build_public_site.build_site(REPOSITORY_ROOT, output)

            sitemap_path = output / "sitemap.xml"
            sitemap_text = sitemap_path.read_text(encoding="utf-8")
            sitemap = ET.fromstring(sitemap_text)
            namespace = "http://www.sitemaps.org/schemas/sitemap/0.9"
            expected_paths = [
                "",
                "process/",
                "methodology/",
                "capabilities/",
                "benchmarks/",
                "releases/",
                "research/",
                *[article.public_path for article in articles],
            ]
            expected_urls = [
                "https://bitworks-io.github.io/fast-mlx/" + path
                for path in expected_paths
            ]

            self.assertTrue(
                sitemap_text.startswith('<?xml version="1.0" encoding="utf-8"?>')
            )
            self.assertEqual(sitemap.tag, f"{{{namespace}}}urlset")
            self.assertEqual(
                [
                    element.text
                    for element in sitemap.findall(
                        f"{{{namespace}}}url/{{{namespace}}}loc"
                    )
                ],
                expected_urls,
            )
            self.assertEqual(len(expected_urls), 14)
            self.assertNotIn("index.json", sitemap_text)
            self.assertNotIn("feed.atom", sitemap_text)
            self.assertNotIn("llms.txt", sitemap_text)
            self.assertNotIn("404.html", sitemap_text)

            self.assertEqual(
                (output / "robots.txt").read_text(encoding="utf-8"),
                "User-agent: *\n"
                "Allow: /\n"
                "Sitemap: https://bitworks-io.github.io/fast-mlx/sitemap.xml\n",
            )
            llms = (output / "llms.txt").read_text(encoding="utf-8")
            self.assertIn("/sitemap.xml", llms)
            self.assertIn("/robots.txt", llms)

    def test_reviewed_pages_publish_exact_social_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            articles = build_public_site.build_site(REPOSITORY_ROOT, output)

            expected: dict[str, tuple[str, str, str, str | None]] = {
                "": (
                    "fast-mlx — evidence-gated MLX inference",
                    "A Swift and MLX inference project that continuously researches, tests, and publishes verified capabilities.",
                    "website",
                    None,
                ),
                "process/": (
                    "The improvement loop — fast-mlx",
                    "How fast-mlx turns research into reviewed, testable inference capabilities.",
                    "website",
                    None,
                ),
                "methodology/": (
                    "Methodology — fast-mlx",
                    "The correctness, comparability, and public-claim boundaries behind fast-mlx results.",
                    "website",
                    None,
                ),
                "capabilities/": (
                    "Capabilities & evidence — fast-mlx",
                    "A status-aware inventory of fast-mlx features and scoped measured results.",
                    "website",
                    None,
                ),
                "benchmarks/": (
                    "Benchmark explorer — fast-mlx",
                    "Filter reviewed fast-mlx measurements without separating results from their scope, caveats, or evidence.",
                    "website",
                    None,
                ),
                "releases/": (
                    "Releases — fast-mlx",
                    "A reviewed ledger of fast-mlx public milestones, exact commits, shipped surfaces, and unchanged boundaries.",
                    "website",
                    None,
                ),
                "research/": (
                    "Research notes — fast-mlx",
                    "Dated fast-mlx investigations and measured negative results.",
                    "website",
                    None,
                ),
            }
            expected.update(
                {
                    article.public_path: (
                        f"{article.title} — fast-mlx",
                        article.summary,
                        "article",
                        article.theme,
                    )
                    for article in articles
                }
            )

            self.assertEqual(len(expected), 14)
            self.assertEqual(
                tuple(validate_public_site.REVIEWED_PAGE_METADATA), tuple(expected)
            )
            image_url = (
                "https://bitworks-io.github.io/fast-mlx/assets/social-card.png"
            )
            image_alt = (
                "Abstract emerald data loop connecting research, implementation, "
                "testing, and verified release checkpoints."
            )
            for public_path, (
                title,
                description,
                page_type,
                section,
            ) in expected.items():
                with self.subTest(public_path=public_path):
                    page = output / public_path / "index.html"
                    collector = HeadMetadataCollector()
                    collector.feed(page.read_text(encoding="utf-8"))
                    canonical = "https://bitworks-io.github.io/fast-mlx/" + public_path
                    self.assertEqual(collector.canonicals, [canonical])
                    self.assertEqual(collector.properties["og:title"], [title])
                    self.assertEqual(collector.properties["og:type"], [page_type])
                    self.assertEqual(collector.properties["og:image"], [image_url])
                    self.assertEqual(collector.properties["og:image:width"], ["1200"])
                    self.assertEqual(collector.properties["og:image:height"], ["630"])
                    self.assertEqual(collector.properties["og:image:alt"], [image_alt])
                    self.assertEqual(collector.properties["og:url"], [canonical])
                    self.assertEqual(
                        collector.properties["og:description"], [description]
                    )
                    self.assertEqual(
                        collector.properties["og:site_name"], ["fast-mlx"]
                    )
                    if page_type == "article":
                        self.assertEqual(
                            collector.properties["article:section"], [section]
                        )
                    else:
                        self.assertFalse(
                            any(
                                property_name.startswith("article:")
                                for property_name in collector.properties
                            )
                        )

            self.assertTrue((output / "assets/social-card.png").is_file())
            not_found = HeadMetadataCollector()
            not_found.feed((output / "404.html").read_text(encoding="utf-8"))
            self.assertEqual(not_found.canonicals, [])
            self.assertEqual(not_found.properties, {})

    def test_validator_rejects_social_metadata_tampering(self) -> None:
        def move_metadata_into_body(text: str) -> str:
            start = text.index('    <link rel="canonical"')
            last_tag = '<meta property="og:site_name" content="fast-mlx">'
            end = text.index(last_tag, start) + len(last_tag)
            block = text[start:end]
            without_metadata = text[:start] + text[end:]
            return without_metadata.replace(
                "  <body>", f"  <body>\n    <head>\n{block}\n    </head>", 1
            )

        mutations = (
            (
                "off-origin canonical",
                "index.html",
                lambda text: text.replace(
                    "https://bitworks-io.github.io/fast-mlx/",
                    "https://example.invalid/unreviewed/",
                    1,
                ),
                "index.html metadata does not match reviewed contract",
            ),
            (
                "metadata in second head after body",
                "index.html",
                move_metadata_into_body,
                "index.html metadata does not match reviewed contract",
            ),
            (
                "off-origin social URL",
                "process/index.html",
                lambda text: text.replace(
                    'property="og:url" content="https://bitworks-io.github.io/fast-mlx/process/"',
                    'property="og:url" content="https://example.invalid/process/"',
                    1,
                ),
                "process/index.html metadata does not match reviewed contract",
            ),
            (
                "duplicate title",
                "methodology/index.html",
                lambda text: text.replace(
                    '<meta property="og:title"',
                    '<meta property="og:title" content="Unreviewed duplicate">\n    '
                    '<meta property="og:title"',
                    1,
                ),
                "methodology/index.html metadata does not match reviewed contract",
            ),
            (
                "article authority on core page",
                "capabilities/index.html",
                lambda text: text.replace(
                    '<meta property="og:site_name" content="fast-mlx">',
                    '<meta property="og:site_name" content="fast-mlx">\n    '
                    '<meta property="article:section" content="Unreviewed">',
                    1,
                ),
                "capabilities/index.html metadata does not match reviewed contract",
            ),
            (
                "metadata on not found",
                "404.html",
                lambda text: text.replace(
                    '<meta name="description" content="The requested fast-mlx page was not found.">',
                    '<meta name="description" content="The requested fast-mlx page was not found.">\n    '
                    '<link rel="canonical" href="https://bitworks-io.github.io/fast-mlx/404.html">',
                    1,
                ),
                "404.html must not publish canonical or social metadata",
            ),
            (
                "canonical rel token and case bypass on not found",
                "404.html",
                lambda text: text.replace(
                    '<meta name="description" content="The requested fast-mlx page was not found.">',
                    '<meta name="description" content="The requested fast-mlx page was not found.">\n    '
                    '<link rel="stylesheet CANONICAL" href="https://bitworks-io.github.io/fast-mlx/404.html">',
                    1,
                ),
                "404.html must not publish canonical or social metadata",
            ),
            (
                "duplicate canonical attribute bypass on not found",
                "404.html",
                lambda text: text.replace(
                    '<meta name="description" content="The requested fast-mlx page was not found.">',
                    '<meta name="description" content="The requested fast-mlx page was not found.">\n    '
                    '<link rel="canonical" rel="stylesheet" '
                    'href="https://bitworks-io.github.io/fast-mlx/404.html">',
                    1,
                ),
                "404.html must not publish canonical or social metadata",
            ),
            (
                "duplicate social property bypass on not found",
                "404.html",
                lambda text: text.replace(
                    '<meta name="description" content="The requested fast-mlx page was not found.">',
                    '<meta name="description" content="The requested fast-mlx page was not found.">\n    '
                    '<meta property="og:title" property="not-og" content="Unreviewed">',
                    1,
                ),
                "404.html must not publish canonical or social metadata",
            ),
        )
        for label, relative, mutate, expected in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                path = output / relative
                original = path.read_text(encoding="utf-8")
                mutated = mutate(original)
                self.assertNotEqual(mutated, original)
                path.write_text(mutated, encoding="utf-8")
                self.assertIn(expected, validate_public_site.validate(output))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)
            unexpected = output / "research/unreviewed-note/index.html"
            unexpected.parent.mkdir(parents=True)
            unexpected.write_bytes((output / "index.html").read_bytes())
            self.assertIn(
                "unexpected HTML page outside the reviewed route set: "
                "research/unreviewed-note/index.html",
                validate_public_site.validate(output),
            )

        for relative in ("assets/unreviewed.HTML", "assets/unreviewed.htm"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                unexpected = output / relative
                unexpected.write_bytes((output / "index.html").read_bytes())
                self.assertIn(
                    "unexpected HTML page outside the reviewed route set: " + relative,
                    validate_public_site.validate(output),
                )

    def test_validator_rejects_social_card_tampering(self) -> None:
        for mutation, expected in (
            (
                lambda raw: raw[:-1] + bytes([raw[-1] ^ 1]),
                "assets/social-card.png has the wrong SHA-256",
            ),
            (
                lambda raw: raw + b"oversized",
                "assets/social-card.png has the wrong byte count",
            ),
        ):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                card = output / "assets/social-card.png"
                raw = card.read_bytes()
                card.write_bytes(mutation(raw))
                self.assertIn(expected, validate_public_site.validate(output))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)
            card = output / "assets/social-card.png"
            external = root / "social-card.png"
            external.write_bytes(card.read_bytes())
            card.unlink()
            card.symlink_to(external)
            self.assertIn(
                "assets/social-card.png must be a regular non-symlink file",
                validate_public_site.validate(output),
            )

    def test_builder_rejects_unreviewed_social_card_source(self) -> None:
        for mutation, expected in (
            (
                lambda raw: raw[:-1] + bytes([raw[-1] ^ 1]),
                "wrong SHA-256",
            ),
            (lambda raw: raw + b"oversized", "wrong byte count"),
        ):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                assets = Path(directory) / "assets"
                shutil.copytree(REPOSITORY_ROOT / "site/assets", assets)
                card = assets / "social-card.png"
                raw = card.read_bytes()
                card.write_bytes(mutation(raw))
                with self.assertRaisesRegex(SystemExit, expected):
                    build_public_site.validate_asset_tree(assets)

    def test_site_discovery_contract_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, first)
            build_public_site.build_site(REPOSITORY_ROOT, second)

            self.assertEqual(
                (first / "sitemap.xml").read_bytes(),
                (second / "sitemap.xml").read_bytes(),
            )
            self.assertEqual(
                (first / "robots.txt").read_bytes(),
                (second / "robots.txt").read_bytes(),
            )

    def test_validator_rejects_joint_index_and_sitemap_route_drift(self) -> None:
        for mutation in ("omit", "substitute"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                index_path = output / "research/index.json"
                index = json.loads(index_path.read_text(encoding="utf-8"))
                articles = index["articles"]
                original_path = articles[-1]["path"]

                if mutation == "omit":
                    articles.pop()
                else:
                    substituted_path = "research/unreviewed-note/"
                    articles[-1]["path"] = substituted_path
                    substituted_page = output / substituted_path / "index.html"
                    substituted_page.parent.mkdir(parents=True)
                    substituted_page.write_bytes(
                        (output / original_path / "index.html").read_bytes()
                    )

                article_paths = [article["path"] for article in articles]
                index_path.write_text(
                    json.dumps(index, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                (output / "sitemap.xml").write_text(
                    validate_public_site.render_expected_sitemap(article_paths),
                    encoding="utf-8",
                )

                self.assertIn(
                    "research/index.json does not match reviewed article routes",
                    validate_public_site.validate(output),
                )

    def test_validator_rejects_discovery_contract_tampering(self) -> None:
        mutations = (
            (
                "sitemap route",
                "sitemap.xml",
                lambda text: text.replace(
                    "https://bitworks-io.github.io/fast-mlx/process/",
                    "https://example.invalid/unreviewed/",
                    1,
                ),
                "sitemap.xml does not match reviewed public routes",
            ),
            (
                "sitemap DTD",
                "sitemap.xml",
                lambda text: text.replace(
                    "?>",
                    '?>\n<!DOCTYPE urlset [<!ENTITY injected "not-reviewed">]>',
                    1,
                ),
                "sitemap.xml contains a forbidden XML declaration",
            ),
            (
                "sitemap size",
                "sitemap.xml",
                lambda _text: " " * 1_048_577,
                "sitemap.xml exceeds the 1048576-byte limit",
            ),
            (
                "robots sitemap",
                "robots.txt",
                lambda text: text.replace(
                    "https://bitworks-io.github.io/fast-mlx/sitemap.xml",
                    "https://example.invalid/sitemap.xml",
                    1,
                ),
                "robots.txt does not match the canonical crawl policy",
            ),
            (
                "robots size",
                "robots.txt",
                lambda _text: " " * 4_097,
                "robots.txt exceeds the 4096-byte limit",
            ),
        )
        for label, relative, mutate, expected in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                path = output / relative
                original = path.read_text(encoding="utf-8")
                mutated = mutate(original)
                self.assertNotEqual(mutated, original)
                path.write_text(mutated, encoding="utf-8")
                self.assertIn(expected, validate_public_site.validate(output))

    def test_validator_rejects_discovery_contract_symlinks_before_read(self) -> None:
        for relative in ("sitemap.xml", "robots.txt"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                path = output / relative
                external = root / ("external-" + relative)
                external.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
                path.unlink()
                path.symlink_to(external)

                self.assertIn(
                    f"{relative} must be a regular non-symlink file",
                    validate_public_site.validate(output),
                )

    def test_validator_rejects_non_utf8_discovery_contract(self) -> None:
        for relative, expected in (
            ("sitemap.xml", "sitemap.xml is not UTF-8"),
            ("robots.txt", "robots.txt is not UTF-8"),
        ):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                (output / relative).write_bytes(b"\xff\xfe\xfd")

                self.assertTrue(
                    any(
                        failure.startswith(expected)
                        for failure in validate_public_site.validate(output)
                    )
                )

    def test_validator_requires_discovery_contract_outputs(self) -> None:
        for missing in ("sitemap.xml", "robots.txt"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                (output / missing).unlink()
                self.assertIn(
                    f"missing required file: {missing}",
                    validate_public_site.validate(output),
                )

    def test_release_atom_feed_is_deterministic_and_xml_safe(self) -> None:
        manifest = self.release_manifest()
        manifest["releases"][0]["title"] = "Reviewed <release> & result"
        manifest["releases"][0]["summary"] = "Keeps A < B & B > C."
        manifest["releases"][0]["scope"] = 'No "ambient" authority & no <HTML>.'
        _body, release_index = build_public_site.render_release_catalog(manifest)

        first = build_public_site.render_release_feed(release_index)
        second = build_public_site.render_release_feed(release_index)
        self.assertEqual(first, second)
        self.assertNotIn("<release>", first)
        self.assertNotIn("<HTML>", first)
        self.assertIn("Reviewed &lt;release&gt; &amp; result", first)
        self.assertIn("Keeps A &lt; B &amp; B &gt; C.", first)

        atom = lambda name: f"{{http://www.w3.org/2005/Atom}}{name}"
        entry = ET.fromstring(first).find(atom("entry"))
        self.assertIsNotNone(entry)
        self.assertEqual(entry.findtext(atom("title")), "Reviewed <release> & result")
        self.assertEqual(
            entry.findtext(atom("summary")),
            'Keeps A < B & B > C. Boundary: No "ambient" authority & no <HTML>.',
        )

    def test_validator_rejects_release_atom_tampering(self) -> None:
        mutations = (
            (
                "entry identity",
                lambda text: text.replace(
                    "urn:fast-mlx:public-commit:195963b912ff287fa4f7b7058c08fdd9b75dcd3d",
                    "urn:fast-mlx:public-commit:0000000000000000000000000000000000000000",
                    1,
                ),
                "releases/feed.atom does not match releases/index.json",
            ),
            (
                "XML declaration",
                lambda text: text.replace(
                    "?>",
                    "?>\n<!DOCTYPE feed [<!ENTITY injected \"not-reviewed\">]>",
                    1,
                ),
                "releases/feed.atom contains a forbidden XML declaration",
            ),
            (
                "size limit",
                lambda _text: " " * 1_048_577,
                "releases/feed.atom exceeds the 1048576-byte limit",
            ),
        )
        for label, mutate, expected in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                feed_path = output / "releases/feed.atom"
                original = feed_path.read_text(encoding="utf-8")
                mutated = mutate(original)
                self.assertNotEqual(mutated, original)
                feed_path.write_text(mutated, encoding="utf-8")
                self.assertIn(expected, validate_public_site.validate(output))

    def test_validator_rejects_release_atom_symlink_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)
            feed_path = output / "releases/feed.atom"
            external = root / "external.atom"
            external.write_text(feed_path.read_text(encoding="utf-8"), encoding="utf-8")
            feed_path.unlink()
            feed_path.symlink_to(external)

            self.assertIn(
                "releases/feed.atom must be a regular non-symlink file",
                validate_public_site.validate(output),
            )

    def test_validator_requires_release_outputs(self) -> None:
        for missing in (
            "releases/index.html",
            "releases/index.json",
            "releases/feed.atom",
        ):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                (output / missing).unlink()
                self.assertIn(
                    f"missing required file: {missing}",
                    validate_public_site.validate(output),
                )

    def test_validator_rejects_release_page_commit_outside_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)
            page_path = output / "releases/index.html"
            page = page_path.read_text(encoding="utf-8")
            page_path.write_text(
                page.replace(
                    'data-public-commit="555514986cdd17ca921c9d9607a92d6248734fdd"',
                    'data-public-commit="0000000000000000000000000000000000000000"',
                    1,
                ),
                encoding="utf-8",
            )
            self.assertIn(
                "release page card set does not match release ledger",
                validate_public_site.validate(output),
            )

    def test_validator_rejects_release_json_route_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)
            index_path = output / "releases/index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["releases"][0]["publicLinks"][0]["path"] = "private/"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            failures = validate_public_site.validate(output)
            self.assertTrue(
                any(
                    failure.startswith("release ledger")
                    or "release page" in failure
                    for failure in failures
                ),
                failures,
            )

    def test_benchmark_explorer_is_generated_from_reviewed_highlights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)

            catalog = json.loads(
                (REPOSITORY_ROOT / "site/capabilities.json").read_text(
                    encoding="utf-8"
                )
            )
            explorer = (output / "benchmarks/index.html").read_text(encoding="utf-8")

            self.assertIn("Benchmark explorer", explorer)
            self.assertEqual(
                explorer.count('class="benchmark-result"'),
                len(catalog["performanceHighlights"]),
            )
            for highlight in catalog["performanceHighlights"]:
                self.assertIn(
                    f'data-highlight-id="{highlight["id"]}"',
                    explorer,
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
                    self.assertIn(highlight[field], explorer)

            self.assertIn('name="model"', explorer)
            self.assertIn('name="hardware"', explorer)
            self.assertIn('name="decision"', explorer)
            self.assertIn('data-benchmark-count aria-live="polite"', explorer)
            self.assertIn('data-benchmark-empty hidden role="status"', explorer)
            self.assertIn('data-benchmark-reset type="reset"', explorer)
            self.assertIn("JavaScript is optional", explorer)
            self.assertIn(
                '<a href="../benchmarks/" aria-current="page">Benchmarks</a>',
                explorer,
            )
            self.assertEqual(explorer.count('aria-current="page"'), 1)
            self.assertIn(
                '<script src="../assets/benchmark-explorer.js" defer></script>',
                explorer,
            )
            self.assertNotIn('class="benchmark-result" hidden', explorer)

            llms = (output / "llms.txt").read_text(encoding="utf-8")
            self.assertIn("/benchmarks/", llms)
            self.assertIn("/capabilities/index.json", explorer)

    def test_benchmark_explorer_progressive_enhancement_is_bounded(self) -> None:
        script = (
            REPOSITORY_ROOT / "site/assets/benchmark-explorer.js"
        ).read_text(encoding="utf-8")
        stylesheet = (REPOSITORY_ROOT / "site/assets/site.css").read_text(
            encoding="utf-8"
        )

        self.assertIn('classList.add("benchmark-enhanced")', script)
        self.assertIn("URLSearchParams", script)
        self.assertIn("history.replaceState", script)
        self.assertIn("card.hidden", script)
        self.assertIn("textContent", script)
        self.assertNotIn("innerHTML", script)
        self.assertNotIn("fetch(", script)
        self.assertNotIn("localStorage", script)
        self.assertIn(".benchmark-controls", stylesheet)
        self.assertIn("html.benchmark-enhanced .benchmark-controls", stylesheet)

    def test_benchmark_explorer_escapes_catalog_text(self) -> None:
        hostile = 'Model "quoted" <script>alert(1)</script>'
        rendered = build_public_site.render_benchmark_explorer(
            [
                {
                    "id": "safe-highlight",
                    "metric": '<img src=x onerror="alert(1)">',
                    "label": "Bounded result <only>",
                    "model": hostile,
                    "hardware": "Hardware & fixture",
                    "workload": "Synthetic > transcript",
                    "date": "2026-08-11",
                    "decision": "promoted-scoped",
                    "caveat": "No <broad> claim.",
                    "evidence": {
                        "path": "research/reviewed-note/",
                        "title": "Reviewed <note>",
                    },
                }
            ]
        )
        self.assertNotIn("<script>", rendered)
        self.assertNotIn("<img ", rendered)
        self.assertNotIn("<broad>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("&lt;img src=x onerror=&quot;alert(1)&quot;&gt;", rendered)
        self.assertIn("Hardware &amp; fixture", rendered)
        self.assertIn('data-model="Model &quot;quoted&quot; &lt;script&gt;', rendered)

    def test_benchmark_explorer_runtime_state_is_bounded(self) -> None:
        completed = subprocess.run(
            [
                "node",
                str(REPOSITORY_ROOT / "scripts/tests/benchmark_explorer_node_test.js"),
                str(REPOSITORY_ROOT / "site/assets/benchmark-explorer.js"),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("benchmark explorer runtime checks passed", completed.stdout)

    def test_validator_requires_benchmark_explorer_outputs(self) -> None:
        for missing in ("benchmarks/index.html", "assets/benchmark-explorer.js"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                (output / missing).unlink()
                self.assertIn(
                    f"missing required file: {missing}",
                    validate_public_site.validate(output),
                )

    def test_validator_rejects_benchmark_card_outside_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)
            explorer_path = output / "benchmarks/index.html"
            explorer = explorer_path.read_text(encoding="utf-8")
            explorer_path.write_text(
                explorer.replace(
                    'data-highlight-id="pld-echo-throughput"',
                    'data-highlight-id="private-diagnostic"',
                    1,
                ),
                encoding="utf-8",
            )

            self.assertIn(
                "benchmark explorer card set does not match performance highlights",
                validate_public_site.validate(output),
            )

    def test_validator_requires_exact_benchmark_card_evidence(self) -> None:
        catalog = json.loads(
            (REPOSITORY_ROOT / "site/capabilities.json").read_text(encoding="utf-8")
        )
        highlight = max(
            catalog["performanceHighlights"], key=lambda entry: entry["date"]
        )
        evidence_href = f'../research/{highlight["evidenceSlug"]}/'
        mutations = {
            "metric": (
                f'<div class="metric">{highlight["metric"]}</div>',
                '<div class="metric">removed metric</div>',
            ),
            "label": (
                f'<h3>{highlight["label"]}</h3>',
                '<h3>removed label</h3>',
            ),
            "model": (
                f'<div><dt>Model</dt><dd>{highlight["model"]}</dd></div>',
                '<div><dt>Model</dt><dd>removed model</dd></div>',
            ),
            "hardware": (
                f'<div><dt>Hardware</dt><dd>{highlight["hardware"]}</dd></div>',
                '<div><dt>Hardware</dt><dd>removed hardware</dd></div>',
            ),
            "workload": (
                f'<div><dt>Workload</dt><dd>{highlight["workload"]}</dd></div>',
                '<div><dt>Workload</dt><dd>removed workload</dd></div>',
            ),
            "date": (
                f'<time datetime="{highlight["date"]}">{highlight["date"]}</time>',
                '<time datetime="1900-01-01">1900-01-01</time>',
            ),
            "caveat": (
                '<p class="scope-note"><strong>Boundary:</strong> '
                f'{highlight["caveat"]}</p>',
                '<p class="scope-note"><strong>Boundary:</strong> removed caveat</p>',
            ),
            "evidence": (
                f'<a class="text-link" href="{evidence_href}">',
                '<a class="text-link" href="../capabilities/">',
            ),
        }

        for field, (before, after) in mutations.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                explorer_path = output / "benchmarks/index.html"
                explorer = explorer_path.read_text(encoding="utf-8")
                self.assertIn(before, explorer)
                explorer_path.write_text(
                    explorer.replace(before, after, 1), encoding="utf-8"
                )
                self.assertIn(
                    f"benchmark explorer card '{highlight['id']}' has the wrong {field}",
                    validate_public_site.validate(output),
                )

    def test_validator_requires_exact_benchmark_filter_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)
            explorer_path = output / "benchmarks/index.html"
            explorer = explorer_path.read_text(encoding="utf-8")
            explorer_path.write_text(
                explorer.replace(
                    '<option value="Apple M3 Ultra">Apple M3 Ultra</option>',
                    '<option value="Private host">Private host</option>',
                    1,
                ),
                encoding="utf-8",
            )
            self.assertIn(
                "benchmark explorer hardware options do not match performance highlights",
                validate_public_site.validate(output),
            )

    def test_validator_preserves_benchmark_no_js_and_live_semantics(self) -> None:
        mutations = (
            (
                '<article class="benchmark-result" role="listitem"',
                '<article class="benchmark-result" hidden role="listitem"',
                "benchmark explorer card 'http-sse-operational-soak' is hidden before enhancement",
            ),
            (
                'data-benchmark-count aria-live="polite"',
                "data-benchmark-count",
                "benchmark explorer has no live result count",
            ),
            (
                'data-benchmark-empty hidden role="status"',
                'data-benchmark-empty role="status"',
                "benchmark explorer has no hidden status empty state",
            ),
            (
                'data-benchmark-empty hidden role="status"',
                "data-benchmark-empty hidden",
                "benchmark explorer has no hidden status empty state",
            ),
        )
        for before, after, expected_failure in mutations:
            with self.subTest(expected_failure=expected_failure), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                explorer_path = output / "benchmarks/index.html"
                explorer = explorer_path.read_text(encoding="utf-8")
                self.assertIn(before, explorer)
                explorer_path.write_text(
                    explorer.replace(before, after, 1), encoding="utf-8"
                )
                self.assertIn(
                    expected_failure,
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

    def test_release_manifest_symlink_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            external = root / "external.json"
            external.write_text(json.dumps(self.release_manifest()), encoding="utf-8")
            manifest = root / "site/releases.json"
            manifest.parent.mkdir(parents=True)
            manifest.symlink_to(external)

            with self.assertRaises(SystemExit):
                build_public_site.load_release_catalog(root)

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
