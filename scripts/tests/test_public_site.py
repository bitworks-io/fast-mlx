from __future__ import annotations

import contextlib
import datetime as dt
import html
import html.parser
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPLORER_RUNTIME_TIMEOUT_SECONDS = 30
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
        article_count = len(manifest["articles"])
        self.assertEqual(
            len({entry["source"] for entry in manifest["articles"]}), article_count
        )
        self.assertEqual(
            len({entry["slug"] for entry in manifest["articles"]}), article_count
        )
        self.assertEqual(
            manifest["articles"][-1],
            {
                "source": "docs/content/2026-09-18-streaming-the-experts-changed-the-answers.md",
                "slug": "streaming-the-experts-changed-the-answers",
                "status": "published",
                "reviewedAt": "2026-09-18",
            },
        )
        self.assertEqual(article_count, 32)

    def test_capability_manifest_is_explicit_and_evidence_backed(self) -> None:
        articles = build_public_site.load_articles(REPOSITORY_ROOT)
        catalog = build_public_site.load_capability_catalog(
            REPOSITORY_ROOT, {article.slug for article in articles}
        )
        self.assertEqual(catalog["schemaVersion"], 1)
        self.assertEqual(catalog["policy"], "fast-mlx-owned-results-only")
        self.assertEqual(catalog["claimBoundary"], "fast-mlx-owned-results-only")
        self.assertEqual(len(catalog["capabilities"]), 9)
        self.assertEqual(len(catalog["performanceHighlights"]), 3)

    def test_sampled_generation_foundation_publication_is_bounded(self) -> None:
        capability_manifest = json.loads(
            (REPOSITORY_ROOT / "site/capabilities.json").read_text(
                encoding="utf-8"
            )
        )
        capability_records = [
            record
            for record in capability_manifest["capabilities"]
            if record["id"] == "deterministic-sampled-generation-foundation"
        ]
        self.assertEqual(len(capability_records), 1)
        self.assertEqual(
            capability_records[0],
            {
                "id": "deterministic-sampled-generation-foundation",
                "name": "Deterministic sampled-generation foundation",
                "status": "implemented",
                "summary": (
                    "A dependency-free CPU oracle defines seeded temperature and "
                    "top-p token selection with stable counter addressing."
                ),
                "scope": (
                    "Internal HarnessCore foundation only. It is not wired to HTTP "
                    "requests, the serving scheduler, MLX arrays, models, tokenizers, "
                    "or automatic entropy, and it makes no performance claim."
                ),
                "evidenceSlugs": ["sampling-before-serving"],
            },
        )

        publication_manifest = json.loads(
            (REPOSITORY_ROOT / "site/publications.json").read_text(
                encoding="utf-8"
            )
        )
        article_records = [
            record
            for record in publication_manifest["articles"]
            if record["slug"] == "sampling-before-serving"
        ]
        self.assertEqual(
            article_records,
            [
                {
                    "source": "docs/content/2026-08-16-sampling-before-serving.md",
                    "slug": "sampling-before-serving",
                    "status": "published",
                    "reviewedAt": "2026-08-16",
                }
            ],
        )
        note = (
            REPOSITORY_ROOT / "docs/content/2026-08-16-sampling-before-serving.md"
        ).read_text(encoding="utf-8")
        normalized_note = " ".join(note.split())
        for required in (
            "deterministic CPU foundation",
            "temperature before top-p",
            "not sampled serving support",
            "not an MLX device path",
            "no model or tokenizer claim",
            "no performance claim",
            "rejected and non-promotable",
        ):
            self.assertIn(required, normalized_note)

        release_manifest = json.loads(
            (REPOSITORY_ROOT / "site/releases.json").read_text(encoding="utf-8")
        )
        release_records = [
            record
            for record in release_manifest["releases"]
            if record["id"] == "deterministic-sampled-generation-foundation"
        ]
        self.assertEqual(
            release_records,
            [
                {
                    "id": "deterministic-sampled-generation-foundation",
                    "title": "Publish deterministic sampled-generation foundation",
                    "publishedAt": "2026-08-16T15:25:00-05:00",
                    "category": "foundation",
                    "state": "released",
                    "summary": (
                        "Publishes the deterministic CPU sampling oracle, exact "
                        "contract tests, capability page, and technical note."
                    ),
                    "scope": (
                        "Internal deterministic CPU foundation only; no sampled "
                        "HTTP/SSE serving, request/scheduler integration, MLX/model/"
                        "tokenizer/automatic-entropy path, fusion, or performance claim."
                    ),
                    "publicCommit": "9e1424b62e3d36108992322664af5ed7c82a1298",
                    "publicLinks": [
                        {
                            "label": "Inspect the capability",
                            "path": (
                                "capabilities/"
                                "deterministic-sampled-generation-foundation/"
                            ),
                        },
                        {
                            "label": "Read the technical note",
                            "path": "research/sampling-before-serving/",
                        },
                        {
                            "label": "Read the release JSON",
                            "path": "releases/index.json",
                        },
                    ],
                }
            ],
        )

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
                "id": "tagged-distribution-v0-1-3",
                "title": "Publish v0.1.3: the engine no longer outlives its launcher",
                "publishedAt": "2026-09-19T21:17:52Z",
                "category": "operations",
                "state": "released",
                "summary": (
                    "Publishes the v0.1.3 arm64 macOS distribution, in which the "
                    "fastmlx serve --front-port launcher no longer leaves its engine "
                    "running: the engine is stopped when the launcher or its guard is "
                    "killed, a requested stop exits 143, and a busy engine port is "
                    "refused rather than adopted."
                ),
                "scope": (
                    "Tagged public distribution and the arm64 macOS archive published "
                    "with its SHA-256 checksum only; the binaries are unsigned and not "
                    "notarized, and no performance, quality, model-support, "
                    "host-qualification, deployment, or production-promotion claim "
                    "follows."
                ),
                "publicCommit": "f418b8e47b35ff030485bd2878ad5dad63be31fd",
                "publicLinks": [
                    {
                        "label": "Start with the operator quickstart",
                        "path": "quickstart/",
                    },
                    {
                        "label": "Read the release ledger",
                        "path": "releases/index.json",
                    },
                ],
            },
        )
        commits = [entry["publicCommit"] for entry in catalog["releases"]]
        self.assertEqual(
            commits,
            [
                "f418b8e47b35ff030485bd2878ad5dad63be31fd",
                "9bb06501a38684c08293d900fcc6e804e59b11e4",
                "8aab34b45b8dbd39163f8c507343105b55de123f",
                "9671a8b39541990201b510c0da2843528e822395",
                "940e247ff62fb66d76adb3f48167a1d58d5e3dd1",
                "b62e0ccd69e6456ad5a824e1a3177b1a4580ad1b",
                "6a44732e6834788c12b569a8dc5fe1d3b1376455",
                "1cb589cc71241cf84c4763f4f37b5ad5b8520bdc",
                "5f070f128d68de97544c912bcde5223a76c97304",
                "137d88d79d15fc65f169ba5dce79db75723ac92e",
                "17ce331930929b1deb100cd6afed60ca993e665d",
                "9e1424b62e3d36108992322664af5ed7c82a1298",
                "c9ba0341a473dccf219f421efefedf3df3e30e2f",
                "1bb670b0d4be82e294f392c4cb35b0c9977a9f89",
                "acaaa522f3b45b34a24c39a1f227534c835331ce",
                "b91ed46e1d88bfadb984b81162165e386d3445b0",
                "9e0c1a159ee5e458573bae67f323915220eb9b90",
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
            "unsupported machine-readable public path": lambda manifest: manifest[
                "releases"
            ][0]["publicLinks"][0].update({"path": "research/feed.xml"}),
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
                'href="releases/' + latest["id"] + '/"', page
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
                "Quality, fit, and exactness evidence, with the performance record beside it.",
                page,
            )
            self.assertIn(
                "what each setting costs, what is shipped, and what remains gated.",
                page,
            )
            self.assertIn(
                "Reviewed throughput, memory, and service-health results stay tied to source, model, and workload identity.",
                page,
            )
            self.assertIn(
                "OpenAI-compatible serving, continuous batching, exact prefix/session-cache controls, and capacity planning.",
                page,
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
        # Derive the count-bearing home markers from a reference render instead of hard-coding the
        # numbers: these track the reviewed manifests (implemented was 4→6, reviewed capabilities
        # 7→9, research notes 9→11, and all will drift again as the loop ships work). Each mutation
        # subtest only needs a marker present exactly once to prove the seal catches a tampered count,
        # so extracting the live values from a generator-faithful render stops the test going stale on
        # every capability or research note that ships. The generator is deterministic on fixed
        # manifests, so the per-mutation rebuilds below re-emit these exact strings.
        with tempfile.TemporaryDirectory() as _ref_dir:
            _ref_output = Path(_ref_dir) / "site"
            _ref_output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, _ref_output)
            _ref_home = (_ref_output / "index.html").read_text(encoding="utf-8")

        def _sole_marker(pattern: str) -> str:
            found = re.findall(pattern, _ref_home)
            self.assertEqual(
                len(found), 1, f"expected exactly one home marker for /{pattern}/, got {found}"
            )
            return found[0]

        implemented_home_marker = _sole_marker(
            r'data-capability-status="implemented" data-count="\d+"'
        )
        implemented_home_marker_tampered = re.sub(
            r'data-count="\d+"', 'data-count="99"', implemented_home_marker
        )
        # This subtest tampers by injecting a visibility suppressor (aria-hidden) on the article,
        # NOT by changing the count — the count in the marker is incidental (it just makes the
        # anchor unique). Preserve the aria-hidden injection; only the count digits are dynamic.
        reviewed_capabilities_marker = _sole_marker(
            r'<article role="listitem">\n<h3>\d+ reviewed capabilities</h3>'
        )
        reviewed_capabilities_marker_tampered = reviewed_capabilities_marker.replace(
            '<article role="listitem">', '<article role="listitem" aria-hidden="true">', 1
        )
        research_notes_marker = _sole_marker(r"\d+ published research notes")
        research_notes_marker_tampered = "0 published research notes"
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
                reviewed_capabilities_marker,
                reviewed_capabilities_marker_tampered,
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
                implemented_home_marker,
                implemented_home_marker_tampered,
                "home current-cycle capability status counts do not match capabilities/index.json",
            ),
            (
                "research count",
                research_notes_marker,
                research_notes_marker_tampered,
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

    def test_operator_quickstart_is_generated_from_reviewed_cli_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)

            quickstart_path = output / "quickstart/index.html"
            self.assertTrue(
                quickstart_path.is_file(),
                "missing reviewed operator quickstart: quickstart/index.html",
            )
            page = quickstart_path.read_text(encoding="utf-8")
            expected_commands = {
                "clone": (
                    "git clone https://github.com/bitworks-io/fast-mlx.git\n"
                    "cd fast-mlx"
                ),
                "serve-scripted": (
                    "swift run --package-path spike fastmlx-serve --scripted"
                ),
                "request-json": (
                    "curl http://127.0.0.1:8080/v1/chat/completions \\\n"
                    "  -H 'content-type: application/json' \\\n"
                    "  -d '{\"model\":\"fastmlx-scripted\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"temperature\":0,\"n\":1,\"stream\":false}'"
                ),
                "request-sse": (
                    "curl -N http://127.0.0.1:8080/v1/chat/completions \\\n"
                    "  -H 'content-type: application/json' \\\n"
                    "  -d '{\"model\":\"fastmlx-scripted\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"temperature\":0,\"n\":1,\"stream\":true}'"
                ),
                "capacity": "swift run --package-path spike fastmlx-capacity",
                "serve-help": "swift run --package-path spike fastmlx-serve --help",
            }
            self.assertIn("data-quickstart", page)
            self.assertEqual(page.count("<h1>"), 1)
            for identifier, command in expected_commands.items():
                with self.subTest(command=identifier):
                    self.assertIn(
                        f'<code data-command="{identifier}">{html.escape(command)}</code>',
                        page,
                    )
            for required_text in (
                "Apple Silicon Mac",
                "macOS 14 or newer",
                "Swift 6",
                "Scripted mode loads no model",
                "open another terminal",
                "POST /v1/chat/completions",
                "application/json",
                "text/event-stream",
                "FASTMLX_API_KEY",
                "temperature zero",
                "n = 1",
                "No model weights are bundled",
                "does not prove model compatibility, output quality, capacity fit, or performance",
            ):
                with self.subTest(required_text=required_text):
                    self.assertIn(required_text, page)
            for href in (
                "../capabilities/",
                "../benchmarks/",
                "../methodology/",
                "https://github.com/bitworks-io/fast-mlx",
            ):
                self.assertIn(f'href="{href}"', page)
            self.assertIn(
                '<a href="../quickstart/" aria-current="page">Quickstart</a>',
                page,
            )
            self.assertNotIn("<script", page)

            home = (output / "index.html").read_text(encoding="utf-8")
            self.assertIn(
                '<a class="button primary" href="quickstart/">Run the transport smoke</a>',
                home,
            )
            sitemap = (output / "sitemap.xml").read_text(encoding="utf-8")
            self.assertIn(
                "https://bitworks-io.github.io/fast-mlx/quickstart/", sitemap
            )
            llms = (output / "llms.txt").read_text(encoding="utf-8")
            self.assertIn(
                "- /quickstart/: model-free HTTP/SSE operator quickstart", llms
            )

            metadata = HeadMetadataCollector()
            metadata.feed(page)
            self.assertEqual(
                metadata.canonicals,
                ["https://bitworks-io.github.io/fast-mlx/quickstart/"],
            )
            self.assertEqual(metadata.properties["og:type"], ["website"])
            self.assertEqual(validate_public_site.validate(output), [])

    def test_apache_license_page_is_generated_for_commercial_evaluators(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)

            license_path = output / "license/index.html"
            self.assertTrue(
                license_path.is_file(),
                "missing reviewed license page: license/index.html",
            )
            page = license_path.read_text(encoding="utf-8")
            self.assertIn("data-license-page", page)
            self.assertEqual(page.count("<h1>"), 1)
            for required_text in (
                "Apache License 2.0",
                "Commercial use",
                "Proprietary extensions",
                "Redistributing fast-mlx",
                "modified files",
                "patent license",
                "trademark rights",
                "Third-party works",
                "not legal advice",
            ):
                with self.subTest(required_text=required_text):
                    self.assertIn(required_text, page)
            for href in (
                "https://github.com/bitworks-io/fast-mlx/blob/main/LICENSE",
                "https://github.com/bitworks-io/fast-mlx/blob/main/NOTICE",
                "https://www.apache.org/licenses/LICENSE-2.0",
                "https://www.apache.org/foundation/license-faq.html",
            ):
                self.assertIn(f'href="{href}"', page)
            self.assertNotIn("<script", page)

            home = (output / "index.html").read_text(encoding="utf-8")
            self.assertIn('href="license/">License</a>', home)
            self.assertIn(
                'href="../license/">Share this orientation</a>', page
            )

            sitemap = (output / "sitemap.xml").read_text(encoding="utf-8")
            self.assertIn(
                "https://bitworks-io.github.io/fast-mlx/license/", sitemap
            )
            llms = (output / "llms.txt").read_text(encoding="utf-8")
            self.assertIn(
                "- /license/: Apache-2.0 commercial-use and redistribution orientation",
                llms,
            )

            metadata = HeadMetadataCollector()
            metadata.feed(page)
            self.assertEqual(
                metadata.canonicals,
                ["https://bitworks-io.github.io/fast-mlx/license/"],
            )
            self.assertEqual(metadata.properties["og:type"], ["website"])
            self.assertEqual(validate_public_site.validate(output), [])

    def test_validator_rejects_license_page_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)
            license_path = output / "license/index.html"
            page = license_path.read_text(encoding="utf-8")
            self.assertEqual(page.count("<h3>Commercial use</h3>"), 1)
            license_path.write_text(
                page.replace(
                    "<h3>Commercial use</h3>",
                    "<h3>Unreviewed permission</h3>",
                    1,
                ),
                encoding="utf-8",
            )
            self.assertIn(
                "license/index.html does not match the reviewed page seal",
                validate_public_site.validate(output),
            )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)
            (output / "license/index.html").unlink()
            failures = validate_public_site.validate(output)
            self.assertIn("missing required file: license/index.html", failures)
            self.assertIn(
                "license/index.html must be a regular non-symlink file", failures
            )

    def test_license_page_size_cap_precedes_content_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory)
            license_path = site / "license/index.html"
            license_path.parent.mkdir(parents=True)
            with license_path.open("wb") as handle:
                handle.truncate(validate_public_site.MAX_LICENSE_PAGE_BYTES + 1)

            with mock.patch.object(
                Path,
                "read_bytes",
                side_effect=AssertionError("oversized license page was read"),
            ):
                self.assertEqual(
                    validate_public_site.validate_license_page(site),
                    ["license/index.html exceeds the 131072-byte limit"],
                )

    def test_validator_rejects_quickstart_contract_and_visibility_drift(self) -> None:
        mutations = (
            (
                "command drift",
                "swift run --package-path spike fastmlx-capacity",
                "swift run --package-path spike fastmlx-capacity --unsafe",
                "quickstart commands do not match the reviewed CLI contract",
            ),
            (
                "hidden root",
                "<div data-quickstart>",
                "<div data-quickstart hidden>",
                "quickstart content contains a visibility suppressor",
            ),
            (
                "script insertion",
                "<div data-quickstart>",
                "<div data-quickstart><script>window.quickstart = true</script>",
                "quickstart page must not contain scripts or inline styles",
            ),
            (
                "event handler insertion",
                '<p class="lede">',
                '<p class="lede" onclick="window.quickstart = true">',
                "quickstart content contains an interactive or executable tag",
            ),
            (
                "active element insertion",
                "<div data-quickstart>",
                '<div data-quickstart><iframe src="https://example.com"></iframe>',
                "quickstart content contains an interactive or executable tag",
            ),
            (
                "stylesheet hiding class",
                'class="section shell" aria-labelledby="quickstart-prerequisites"',
                'class="section shell research-controls" aria-labelledby="quickstart-prerequisites"',
                "quickstart content contains a visibility suppressor",
            ),
            (
                "proof boundary drift",
                "does not prove model compatibility, output quality, capacity fit, or performance",
                "proves model compatibility and performance",
                "quickstart page is missing reviewed text",
            ),
            (
                "contradictory claim insertion",
                "does not prove model compatibility, output quality, capacity fit, or performance.",
                "does not prove model compatibility, output quality, capacity fit, or performance. This proves model compatibility, output quality, capacity fit, and performance.",
                "quickstart/index.html does not match the reviewed page seal",
            ),
        )
        for label, original, replacement, expected_failure in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                quickstart_path = output / "quickstart/index.html"
                page = quickstart_path.read_text(encoding="utf-8")
                self.assertEqual(page.count(original), 1)
                quickstart_path.write_text(
                    page.replace(original, replacement, 1), encoding="utf-8"
                )
                self.assertTrue(
                    any(
                        failure.startswith(expected_failure)
                        for failure in validate_public_site.validate(output)
                    )
                )

    def test_validator_rejects_quickstart_navigation_drift(self) -> None:
        mutations = (
            (
                "href drift",
                lambda page: page.replace(
                    '<a href="../quickstart/">Quickstart</a>',
                    '<a href="../capabilities/">Quickstart</a>',
                    1,
                ),
            ),
            (
                "footer substitution",
                lambda page: page.replace(
                    '<a href="../quickstart/">Quickstart</a>', "", 1
                ).replace(
                    '<a href="../quickstart/">First run</a>',
                    '<a href="../quickstart/">Quickstart</a>',
                    1,
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                process_path = output / "process/index.html"
                page = process_path.read_text(encoding="utf-8")
                original = '<a href="../quickstart/">Quickstart</a>'
                self.assertEqual(page.count(original), 1)
                mutated = mutate(page)
                self.assertNotEqual(mutated, page)
                process_path.write_text(mutated, encoding="utf-8")
                self.assertIn(
                    "process/index.html quickstart navigation does not match reviewed contract",
                    validate_public_site.validate(output),
                )

    def test_build_is_complete_and_links_are_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            articles = build_public_site.build_site(REPOSITORY_ROOT, output)
            releases = json.loads(
                (REPOSITORY_ROOT / "site/releases.json").read_text(encoding="utf-8")
            )
            build_public_site.scan_generated_output(output)
            publications = json.loads(
                (REPOSITORY_ROOT / "site/publications.json").read_text(
                    encoding="utf-8"
                )
            )
            article_count = len(publications["articles"])
            self.assertEqual(len(articles), article_count)
            self.assertEqual(validate_public_site.validate(output), [])

            public_index = json.loads(
                (output / "research/index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(public_index["articles"]), article_count)
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
            reviewed_capabilities = json.loads(
                (REPOSITORY_ROOT / "site/capabilities.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                len(capabilities["capabilities"]),
                len(reviewed_capabilities["capabilities"]),
            )
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
                # The page renders these three through html.escape(str(x)) --
                # quote=True -- so compare against the escaped form. For a field
                # with no special character the two are identical, so this is the
                # same assertion; it differs only for a field that genuinely has
                # to be escaped, where the raw form is the WRONG expectation.
                self.assertIn(html.escape(release["title"]), page)
                self.assertIn(html.escape(release["summary"]), page)
                self.assertIn(html.escape(release["scope"]), page)
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
                self.assertIn(f'href="{release["id"]}/"', page)
            self.assertIn(
                '<a href="../releases/" aria-current="page">Releases</a>', page
            )
            self.assertEqual(page.count('aria-current="page"'), 1)
            llms = (output / "llms.txt").read_text(encoding="utf-8")
            self.assertIn("/releases/", llms)
            self.assertIn("/releases/index.json", llms)

    def test_release_detail_pages_are_generated_from_reviewed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)

            source = json.loads(
                (REPOSITORY_ROOT / "site/releases.json").read_text(encoding="utf-8")
            )
            ledger = (output / "releases/index.html").read_text(encoding="utf-8")
            sitemap = (output / "sitemap.xml").read_text(encoding="utf-8")
            llms = (output / "llms.txt").read_text(encoding="utf-8")

            for release in source["releases"]:
                identifier = release["id"]
                public_path = f"releases/{identifier}/"
                detail_path = output / public_path / "index.html"
                self.assertTrue(
                    detail_path.is_file(),
                    f"missing reviewed release detail page: {public_path}",
                )
                detail = detail_path.read_text(encoding="utf-8")
                self.assertIn(
                    "data-release-detail "
                    f'data-release-id="{identifier}" '
                    f'data-public-commit="{release["publicCommit"]}"',
                    detail,
                )
                self.assertIn(release["state"].upper(), detail)
                self.assertIn(
                    build_public_site.RELEASE_CATEGORY_LABELS[release["category"]],
                    detail,
                )
                self.assertIn(
                    f'<time datetime="{release["publishedAt"]}">'
                    f'{release["publishedAt"][:10]}</time>',
                    detail,
                )
                for field in ("title", "summary", "scope"):
                    self.assertIn(html.escape(release[field]), detail)
                self.assertIn(
                    "https://github.com/bitworks-io/fast-mlx/commit/"
                    + release["publicCommit"],
                    detail,
                )
                for link in release["publicLinks"]:
                    href = build_public_site.relative_href(
                        public_path + "index.html", link["path"]
                    )
                    self.assertIn(
                        f'href="{html.escape(href, quote=True)}"', detail
                    )
                    self.assertIn(html.escape(link["label"]), detail)
                self.assertIn(
                    'href="../">Back to all reviewed releases</a>', detail
                )
                self.assertIn(
                    "This page does not create a new release, measurement, ranking, "
                    "runtime, model, acquisition, or publication authority.",
                    detail,
                )
                self.assertEqual(detail.count("<h1>"), 1)
                self.assertNotIn("<script", detail)
                self.assertIn(f'href="{identifier}/"', ledger)
                self.assertIn(
                    "https://bitworks-io.github.io/fast-mlx/" + public_path,
                    sitemap,
                )
                self.assertIn(f"- /{public_path}:", llms)

    def test_validator_rejects_release_detail_contract_drift(self) -> None:
        identifier = "reviewed-release-detail-permalinks"
        source = json.loads(
            (REPOSITORY_ROOT / "site/releases.json").read_text(encoding="utf-8")
        )
        release = next(
            entry for entry in source["releases"] if entry["id"] == identifier
        )
        public_path = f"releases/{identifier}/"
        first_link = release["publicLinks"][0]
        first_href = build_public_site.relative_href(
            public_path + "index.html", first_link["path"]
        )
        mutations = (
            (
                f'data-release-id="{identifier}"',
                'data-release-id="unreviewed-release"',
                f"release detail '{identifier}' has the wrong id",
            ),
            (
                f'data-public-commit="{release["publicCommit"]}"',
                'data-public-commit="0000000000000000000000000000000000000000"',
                f"release detail '{identifier}' has the wrong commit",
            ),
            (
                html.escape(release["summary"]),
                "Publishes an unreviewed automatic release surface.",
                f"release detail '{identifier}' has drifted reviewed text",
            ),
            (
                html.escape(release["scope"]),
                "Runtime and model authority granted as the default.",
                f"release detail '{identifier}' has drifted reviewed text",
            ),
            (
                "https://github.com/bitworks-io/fast-mlx/commit/"
                + release["publicCommit"],
                "https://github.com/bitworks-io/fast-mlx/commit/"
                + "0" * 40,
                f"release detail '{identifier}' has the wrong action links",
            ),
            (
                f'<a href="{html.escape(first_href, quote=True)}">'
                f'{html.escape(first_link["label"])} →</a>',
                '<a href="../../capabilities/">Changed public link →</a>',
                f"release detail '{identifier}' has the wrong action links",
            ),
            (
                'href="../">Back to all reviewed releases</a>',
                'href="../../">Back to all reviewed releases</a>',
                f"release detail '{identifier}' has the wrong action links",
            ),
            (
                "This page does not create a new release, measurement, ranking, "
                "runtime, model, acquisition, or publication authority.",
                "This page creates a new public runtime and benchmark authority.",
                f"release detail '{identifier}' has drifted reviewed text",
            ),
            (
                '<section class="page-hero shell benchmark-detail release-detail" ',
                '<section class="page-hero shell benchmark-detail release-detail" hidden ',
                f"release detail '{identifier}' is hidden",
            ),
            (
                "</body>",
                '<script src="../../assets/benchmark-explorer.js"></script></body>',
                f"release detail '{identifier}' must not load scripts",
            ),
            (
                f'<meta property="og:title" content="{release["title"]} — fast-mlx release">',
                '<meta property="og:title" content="Unreviewed release — fast-mlx release">',
                f"releases/{identifier}/index.html metadata does not match reviewed contract",
            ),
            (
                "</main>",
                "<p>This release proves live runtime launchability, model admission, "
                "and production benchmark authority.</p></main>",
                f"release detail '{identifier}' does not match the reviewed page seal",
            ),
            (
                "</main>",
                '<img src="../../assets/favicon.svg" onerror="alert(1)"></main>',
                f"release detail '{identifier}' does not match the reviewed page seal",
            ),
        )
        for before, after, expected_failure in mutations:
            with self.subTest(expected_failure=expected_failure), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                path = output / "releases" / identifier / "index.html"
                page = path.read_text(encoding="utf-8")
                self.assertIn(before, page)
                path.write_text(page.replace(before, after, 1), encoding="utf-8")
                self.assertIn(expected_failure, validate_public_site.validate(output))

    def test_validator_rejects_missing_extra_and_symlink_release_details(self) -> None:
        cases = ("missing", "extra", "symlink")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                detail = (
                    output
                    / "releases/reviewed-benchmark-detail-permalinks/index.html"
                )
                if case == "missing":
                    detail.unlink()
                    expected = (
                        "missing required file: "
                        "releases/reviewed-benchmark-detail-permalinks/index.html"
                    )
                elif case == "extra":
                    extra = output / "releases/unreviewed-release"
                    extra.mkdir()
                    (extra / "index.html").write_text(
                        detail.read_text(encoding="utf-8"), encoding="utf-8"
                    )
                    expected = (
                        "unexpected release route outside reviewed set: "
                        "unreviewed-release"
                    )
                else:
                    external = root / "external-release-detail.html"
                    external.write_text(
                        detail.read_text(encoding="utf-8"), encoding="utf-8"
                    )
                    detail.unlink()
                    detail.symlink_to(external)
                    expected = (
                        "releases/reviewed-benchmark-detail-permalinks/index.html "
                        "must be a regular non-symlink file"
                    )
                self.assertIn(expected, validate_public_site.validate(output))

    def test_validator_rejects_joint_release_index_and_detail_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)

            index_path = output / "releases/index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            release = index["releases"][0]
            original_summary = release["summary"]
            changed_summary = original_summary + " Unreviewed automatic authority."
            release["summary"] = changed_summary
            index_path.write_text(
                json.dumps(index, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            for relative in (
                "releases/index.html",
                f"releases/{release['id']}/index.html",
                "releases/feed.atom",
            ):
                path = output / relative
                page = path.read_text(encoding="utf-8")
                self.assertIn(original_summary, page)
                path.write_text(
                    page.replace(original_summary, changed_summary, 1),
                    encoding="utf-8",
                )

            self.assertIn(
                "releases/index.json does not match the reviewed release ledger",
                validate_public_site.validate(output),
            )

    def test_validator_requires_exact_release_detail_links_from_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)
            path = output / "releases/index.html"
            page = path.read_text(encoding="utf-8")
            path.write_text(
                page.replace(
                    'href="reviewed-release-detail-permalinks/"',
                    'href="../capabilities/"',
                    1,
                ),
                encoding="utf-8",
            )
            self.assertIn(
                "release page card 'reviewed-release-detail-permalinks' "
                "has the wrong release detail link",
                validate_public_site.validate(output),
            )

    def test_release_detail_pages_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, first)
            build_public_site.build_site(REPOSITORY_ROOT, second)
            releases = json.loads(
                (REPOSITORY_ROOT / "site/releases.json").read_text(encoding="utf-8")
            )

            for release in releases["releases"]:
                relative = Path("releases") / release["id"] / "index.html"
                self.assertEqual(
                    (first / relative).read_bytes(),
                    (second / relative).read_bytes(),
                )

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
                        "href": "https://bitworks-io.github.io/fast-mlx/releases/"
                        + release["id"]
                        + "/",
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

    def test_research_atom_feed_is_generated_from_reviewed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            articles = build_public_site.build_site(REPOSITORY_ROOT, output)

            feed_path = output / "research/feed.atom"
            self.assertTrue(feed_path.is_file())
            feed_text = feed_path.read_text(encoding="utf-8")
            feed = ET.fromstring(feed_text)
            atom_namespace = "http://www.w3.org/2005/Atom"
            atom = lambda name: f"{{{atom_namespace}}}{name}"
            expected_articles = sorted(
                articles,
                key=lambda article: (
                    article.reviewed_at,
                    article.date,
                    article.slug,
                ),
                reverse=True,
            )

            self.assertTrue(
                feed_text.startswith('<?xml version="1.0" encoding="utf-8"?>')
            )
            self.assertEqual(feed.tag, atom("feed"))
            self.assertEqual(feed.findtext(atom("title")), "fast-mlx reviewed research")
            self.assertEqual(
                feed.findtext(atom("id")),
                "https://bitworks-io.github.io/fast-mlx/research/",
            )
            self.assertEqual(
                feed.findtext(atom("updated")),
                expected_articles[0].reviewed_at + "T00:00:00Z",
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
                    "href": "https://bitworks-io.github.io/fast-mlx/research/feed.atom",
                },
            )
            self.assertEqual(
                feed_links["alternate"],
                {
                    "rel": "alternate",
                    "type": "text/html",
                    "href": "https://bitworks-io.github.io/fast-mlx/research/",
                },
            )

            entries = feed.findall(atom("entry"))
            self.assertEqual(len(entries), len(expected_articles))
            for entry, article in zip(entries, expected_articles):
                canonical = (
                    "https://bitworks-io.github.io/fast-mlx/"
                    + article.public_path
                )
                self.assertEqual(entry.findtext(atom("id")), canonical)
                self.assertEqual(entry.findtext(atom("title")), article.title)
                self.assertEqual(
                    entry.findtext(atom("published")),
                    article.date + "T00:00:00Z",
                )
                self.assertEqual(
                    entry.findtext(atom("updated")),
                    article.reviewed_at + "T00:00:00Z",
                )
                self.assertEqual(
                    entry.find(atom("category")).attrib,
                    {"term": article.theme},
                )
                self.assertEqual(entry.findtext(atom("summary")), article.summary)
                self.assertIsNone(entry.find(atom("content")))
                self.assertEqual(
                    [link.attrib for link in entry.findall(atom("link"))],
                    [
                        {
                            "rel": "alternate",
                            "type": "text/html",
                            "href": canonical,
                        }
                    ],
                )

            research_page = (output / "research/index.html").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                'rel="alternate" type="application/atom+xml" '
                'title="fast-mlx reviewed research" href="feed.atom"',
                research_page,
            )
            self.assertIn(
                '<a class="button secondary" href="feed.atom" '
                'type="application/atom+xml">Subscribe to reviewed research</a>',
                research_page,
            )
            for article in articles:
                article_page = (output / article.output_file).read_text(
                    encoding="utf-8"
                )
                self.assertIn(
                    'rel="alternate" type="application/atom+xml" '
                    'title="fast-mlx reviewed research" href="../feed.atom"',
                    article_page,
                )
            llms = (output / "llms.txt").read_text(encoding="utf-8")
            self.assertIn(
                "/research/feed.atom: Atom feed of reviewed research notes",
                llms,
            )

    def test_reviewed_updates_atom_feed_is_generated_from_reviewed_manifests(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            articles = build_public_site.build_site(REPOSITORY_ROOT, output)

            release_source = json.loads(
                (REPOSITORY_ROOT / "site/releases.json").read_text(
                    encoding="utf-8"
                )
            )
            expected_updates: list[dict[str, str]] = []
            for release in release_source["releases"]:
                published_at = release["publishedAt"]
                expected_updates.append(
                    {
                        "kind": "release",
                        "title": release["title"],
                        "id": "urn:fast-mlx:public-commit:"
                        + release["publicCommit"],
                        "published": published_at,
                        "updated": published_at,
                        "href": (
                            "https://bitworks-io.github.io/fast-mlx/releases/"
                            + release["id"]
                            + "/"
                        ),
                        "via": (
                            "https://github.com/bitworks-io/fast-mlx/commit/"
                            + release["publicCommit"]
                        ),
                        "summary": release["summary"]
                        + " Boundary: "
                        + release["scope"],
                    }
                )
            for article in articles:
                canonical = (
                    "https://bitworks-io.github.io/fast-mlx/"
                    + article.public_path
                )
                expected_updates.append(
                    {
                        "kind": "research",
                        "title": article.title,
                        "id": canonical,
                        "published": article.date + "T00:00:00Z",
                        "updated": article.reviewed_at + "T00:00:00Z",
                        "href": canonical,
                        "via": "",
                        "summary": article.summary,
                    }
                )
            expected_updates.sort(
                key=lambda update: (
                    dt.datetime.fromisoformat(
                        update["updated"].replace("Z", "+00:00")
                    ),
                    update["id"],
                ),
                reverse=True,
            )

            feed_path = output / "feed.atom"
            self.assertTrue(feed_path.is_file())
            feed_text = feed_path.read_text(encoding="utf-8")
            feed = ET.fromstring(feed_text)
            atom_namespace = "http://www.w3.org/2005/Atom"
            atom = lambda name: f"{{{atom_namespace}}}{name}"
            self.assertTrue(
                feed_text.startswith('<?xml version="1.0" encoding="utf-8"?>')
            )
            self.assertEqual(feed.tag, atom("feed"))
            self.assertEqual(
                feed.findtext(atom("title")), "fast-mlx reviewed updates"
            )
            self.assertEqual(
                feed.findtext(atom("id")),
                "https://bitworks-io.github.io/fast-mlx/",
            )
            self.assertEqual(
                feed.findtext(atom("updated")), expected_updates[0]["updated"]
            )
            self.assertEqual(
                feed.findtext(f"{atom('author')}/{atom('name')}"),
                "fast-mlx contributors",
            )
            self.assertEqual(
                [link.attrib for link in feed.findall(atom("link"))],
                [
                    {
                        "rel": "self",
                        "type": "application/atom+xml",
                        "href": "https://bitworks-io.github.io/fast-mlx/feed.atom",
                    },
                    {
                        "rel": "alternate",
                        "type": "text/html",
                        "href": "https://bitworks-io.github.io/fast-mlx/",
                    },
                ],
            )

            entries = feed.findall(atom("entry"))
            self.assertEqual(
                len(expected_updates), len(release_source["releases"]) + len(articles)
            )
            self.assertEqual(len(entries), len(expected_updates))
            self.assertEqual(
                len({entry.findtext(atom("id")) for entry in entries}),
                len(entries),
            )
            for entry, expected in zip(entries, expected_updates):
                self.assertEqual(entry.findtext(atom("title")), expected["title"])
                self.assertEqual(entry.findtext(atom("id")), expected["id"])
                self.assertEqual(
                    entry.findtext(atom("published")), expected["published"]
                )
                self.assertEqual(
                    entry.findtext(atom("updated")), expected["updated"]
                )
                self.assertEqual(
                    entry.find(atom("category")).attrib,
                    {"term": expected["kind"]},
                )
                self.assertEqual(
                    entry.findtext(atom("summary")), expected["summary"]
                )
                self.assertIsNone(entry.find(atom("content")))
                expected_links = [
                    {
                        "rel": "alternate",
                        "type": "text/html",
                        "href": expected["href"],
                    }
                ]
                if expected["via"]:
                    expected_links.append(
                        {"rel": "via", "href": expected["via"]}
                    )
                self.assertEqual(
                    [link.attrib for link in entry.findall(atom("link"))],
                    expected_links,
                )

            home = (output / "index.html").read_text(encoding="utf-8")
            self.assertIn(
                'title="fast-mlx reviewed updates" href="feed.atom"', home
            )
            self.assertIn(
                '<a class="button secondary" href="feed.atom" '
                'type="application/atom+xml">Subscribe to all reviewed updates</a>',
                home,
            )
            for relative in ("releases/index.html", "research/index.html"):
                page = (output / relative).read_text(encoding="utf-8")
                self.assertIn(
                    '<a class="button secondary" href="../feed.atom" '
                    'type="application/atom+xml">Subscribe to all reviewed updates</a>',
                    page,
                )
            llms = (output / "llms.txt").read_text(encoding="utf-8")
            self.assertIn(
                "/feed.atom: combined Atom feed of all reviewed releases and "
                "research notes",
                llms,
            )

    def test_site_discovery_contract_is_generated_from_reviewed_routes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            articles = build_public_site.build_site(REPOSITORY_ROOT, output)

            sitemap_path = output / "sitemap.xml"
            sitemap_text = sitemap_path.read_text(encoding="utf-8")
            sitemap = ET.fromstring(sitemap_text)
            namespace = "http://www.sitemaps.org/schemas/sitemap/0.9"
            releases = json.loads(
                (REPOSITORY_ROOT / "site/releases.json").read_text(encoding="utf-8")
            )
            expected_paths = [
                "",
                "quickstart/",
                "license/",
                "status/",
                "process/",
                "methodology/",
                "capabilities/",
                "benchmarks/",
                "releases/",
                "research/",
                *[
                    f"capabilities/{capability['id']}/"
                    for capability in validate_public_site.reviewed_capability_records()
                ],
                "benchmarks/pld-echo-throughput/",
                "benchmarks/continuous-batch-c2-throughput/",
                "benchmarks/http-sse-operational-soak/",
                *[
                    f"releases/{release['id']}/"
                    for release in releases["releases"]
                ],
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
            self.assertEqual(
                len(expected_urls),
                len(expected_paths[:10])
                + len(validate_public_site.reviewed_capability_records())
                + 3
                + len(releases["releases"])
                + len(articles),
            )
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
            releases = json.loads(
                (REPOSITORY_ROOT / "site/releases.json").read_text(encoding="utf-8")
            )

            expected: dict[str, tuple[str, str, str, str | None]] = {
                "": (
                    "fast-mlx — measured quality for MLX on Apple Silicon",
                    "Measures what speed and memory settings cost in output quality "
                    "for MLX models on Apple Silicon, sizes models before loading, "
                    "and publishes the evidence from a review-gated research loop.",
                    "website",
                    None,
                ),
                "quickstart/": (
                    "Operator quickstart — fast-mlx",
                    "Run fast-mlx's model-free HTTP/JSON and HTTP/SSE transport smoke, inspect capacity, and understand the loaded-serving boundary.",
                    "website",
                    None,
                ),
                "license/": (
                    "Apache-2.0 license — fast-mlx",
                    "Commercial-use, proprietary-extension, redistribution, notice, patent, trademark, and third-party boundaries for the fast-mlx public source.",
                    "website",
                    None,
                ),
                "status/": (
                    "Current status — fast-mlx",
                    "A manifest-derived view of fast-mlx capabilities, measured proof points, reviewed releases, research, and unchanged authority boundaries.",
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
                **{
                    f"capabilities/{capability['id']}/": (
                        f"{capability['name']} — fast-mlx capability",
                        "Reviewed fast-mlx capability state and evidence for "
                        f"{capability['name']}.",
                        "website",
                        None,
                    )
                    for capability in validate_public_site.reviewed_capability_records()
                },
                "benchmarks/": (
                    "Benchmark explorer — fast-mlx",
                    "Filter reviewed fast-mlx measurements without separating results from their scope, caveats, or evidence.",
                    "website",
                    None,
                ),
                "benchmarks/pld-echo-throughput/": (
                    "Repetition-heavy solo PLD, 28.28 → 56.70 tok/s — fast-mlx benchmark evidence",
                    "A reviewed fast-mlx benchmark result with its exact model, hardware, workload, decision, caveat, and evidence.",
                    "website",
                    None,
                ),
                "benchmarks/continuous-batch-c2-throughput/": (
                    "Aggregate C=2 service rate, 29.29 → 42.70 tok/s — fast-mlx benchmark evidence",
                    "A reviewed fast-mlx benchmark result with its exact model, hardware, workload, decision, caveat, and evidence.",
                    "website",
                    None,
                ),
                "benchmarks/http-sse-operational-soak/": (
                    "HTTP/SSE service soak with 10,368 paired request/evidence rows — fast-mlx benchmark evidence",
                    "A reviewed fast-mlx benchmark result with its exact model, hardware, workload, decision, caveat, and evidence.",
                    "website",
                    None,
                ),
                "releases/": (
                    "Releases — fast-mlx",
                    "A reviewed ledger of fast-mlx public milestones, exact commits, shipped surfaces, and unchanged boundaries.",
                    "website",
                    None,
                ),
            }
            expected.update(
                {
                    f"releases/{release['id']}/": (
                        f"{release['title']} — fast-mlx release",
                        build_public_site.RELEASE_DETAIL_DESCRIPTION,
                        "website",
                        None,
                    )
                    for release in releases["releases"]
                }
            )
            expected["research/"] = (
                "Research notes — fast-mlx",
                "Dated fast-mlx investigations and measured negative results.",
                "website",
                None,
            )
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

            self.assertEqual(len(expected), len(validate_public_site.REVIEWED_PAGE_METADATA))
            # Compare the page set rather than dict insertion order: the validator's
            # REVIEWED_PAGE_METADATA literal is keyed by public path and looked up
            # (not iterated in an order-sensitive way), so its authoring order is
            # incidental and drifts independently of the build's article ordering.
            self.assertEqual(
                set(validate_public_site.REVIEWED_PAGE_METADATA), set(expected)
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

    def test_research_atom_feed_is_deterministic_text_only_and_xml_safe(self) -> None:
        article = build_public_site.Article(
            source=Path("docs/content/2026-08-12-reviewed-note.md"),
            source_name="docs/content/2026-08-12-reviewed-note.md",
            slug="reviewed-note",
            title="Reviewed <note> & result",
            date="2026-08-12",
            theme='Exact <research> & "scope"',
            summary="Keeps A < B & B > C without <script> content.",
            reviewed_at="2026-08-13",
            body="This body must not enter the feed. <script>alert(1)</script>",
        )

        first = build_public_site.render_research_feed([article])
        second = build_public_site.render_research_feed([article])
        self.assertEqual(first, second)
        self.assertNotIn("<note>", first)
        self.assertNotIn("<script>", first)
        self.assertNotIn("This body must not enter the feed", first)
        self.assertIn("Reviewed &lt;note&gt; &amp; result", first)
        self.assertIn("Keeps A &lt; B &amp; B &gt; C", first)

        atom = lambda name: f"{{http://www.w3.org/2005/Atom}}{name}"
        entry = ET.fromstring(first).find(atom("entry"))
        self.assertIsNotNone(entry)
        self.assertEqual(entry.findtext(atom("title")), "Reviewed <note> & result")
        self.assertEqual(
            entry.findtext(atom("summary")),
            "Keeps A < B & B > C without <script> content.",
        )
        self.assertIsNone(entry.find(atom("content")))

    def test_reviewed_updates_atom_feed_is_deterministic_text_only_and_xml_safe(
        self,
    ) -> None:
        reviewed_date = max(
            article["reviewedAt"]
            for article in validate_public_site.reviewed_research_articles()
        )
        manifest = self.release_manifest()
        manifest["releases"] = [manifest["releases"][0]]
        manifest["releases"][0]["title"] = "Reviewed <release> & result"
        manifest["releases"][0]["summary"] = "Keeps A < B & B > C."
        manifest["releases"][0]["publishedAt"] = (
            reviewed_date + "T00:30:00+14:00"
        )
        _body, release_index = build_public_site.render_release_catalog(manifest)
        article = build_public_site.Article(
            source=Path("docs/content/2026-08-10-reviewed-note.md"),
            source_name="docs/content/2026-08-10-reviewed-note.md",
            slug="reviewed-note",
            title="Reviewed <note> & result",
            date="2026-08-10",
            theme="Exact research",
            summary="A reviewed summary with A < B & no body content.",
            reviewed_at=reviewed_date,
            body="This body must not enter the combined feed. <script>alert(1)</script>",
        )

        first = build_public_site.render_reviewed_updates_feed(
            release_index, [article]
        )
        second = build_public_site.render_reviewed_updates_feed(
            release_index, [article]
        )
        self.assertEqual(first, second)
        self.assertNotIn("<release>", first)
        self.assertNotIn("<note>", first)
        self.assertNotIn("This body must not enter the combined feed", first)
        self.assertNotIn("<script>", first)
        self.assertIn("Reviewed &lt;release&gt; &amp; result", first)
        self.assertIn("Reviewed &lt;note&gt; &amp; result", first)

        atom = lambda name: f"{{http://www.w3.org/2005/Atom}}{name}"
        entries = ET.fromstring(first).findall(atom("entry"))
        self.assertEqual(len(entries), 2)
        self.assertEqual(
            [entry.find(atom("category")).attrib for entry in entries],
            [{"term": "research"}, {"term": "release"}],
        )
        self.assertEqual(
            [entry.findtext(atom("updated")) for entry in entries],
            [
                reviewed_date + "T00:00:00Z",
                reviewed_date + "T00:30:00+14:00",
            ],
        )
        self.assertTrue(
            all(entry.find(atom("content")) is None for entry in entries)
        )

        expected = validate_public_site.render_expected_reviewed_updates_feed(
            release_index
        )
        expected_entries = ET.fromstring(expected).findall(atom("entry"))
        self.assertEqual(
            expected_entries[0].find(atom("category")).attrib,
            {"term": "research"},
        )
        self.assertNotEqual(
            expected_entries[0].findtext(atom("id")),
            "urn:fast-mlx:public-commit:"
            + manifest["releases"][0]["publicCommit"],
        )

    def test_validator_rejects_reviewed_updates_atom_tampering(self) -> None:
        atom = lambda name: f"{{http://www.w3.org/2005/Atom}}{name}"

        def mutate_entries(text: str, mutation: str) -> str:
            feed = ET.fromstring(text)
            entries = feed.findall(atom("entry"))
            self.assertGreaterEqual(len(entries), 2)
            if mutation == "omit":
                feed.remove(entries[0])
            elif mutation == "reorder":
                feed.remove(entries[0])
                feed.remove(entries[1])
                feed.append(entries[1])
                feed.append(entries[0])
            elif mutation == "duplicate":
                feed.append(
                    ET.fromstring(ET.tostring(entries[0], encoding="unicode"))
                )
            else:
                self.fail(f"unknown reviewed updates mutation: {mutation}")
            ET.register_namespace("", "http://www.w3.org/2005/Atom")
            ET.indent(feed, space="  ")
            return (
                '<?xml version="1.0" encoding="utf-8"?>\n'
                + ET.tostring(feed, encoding="unicode", short_empty_elements=True)
                + "\n"
            )

        mismatch = (
            "feed.atom does not match the reviewed release and research catalogs"
        )
        mutations = (
            (
                "entry identity",
                lambda text: text.replace(
                    "urn:fast-mlx:public-commit:"
                    "1bb670b0d4be82e294f392c4cb35b0c9977a9f89",
                    "urn:fast-mlx:public-commit:"
                    "0000000000000000000000000000000000000000",
                    1,
                ),
                mismatch,
            ),
            ("missing entry", lambda text: mutate_entries(text, "omit"), mismatch),
            (
                "reordered entries",
                lambda text: mutate_entries(text, "reorder"),
                mismatch,
            ),
            (
                "duplicate entry",
                lambda text: mutate_entries(text, "duplicate"),
                mismatch,
            ),
            (
                "active content",
                lambda text: text.replace(
                    "</entry>",
                    '<content type="html">&lt;script&gt;alert(1)&lt;/script&gt;'
                    "</content>\n  </entry>",
                    1,
                ),
                mismatch,
            ),
            (
                "XML declaration",
                lambda text: text.replace(
                    "?>",
                    '?>\n<!DOCTYPE feed [<!ENTITY injected "not-reviewed">]>',
                    1,
                ),
                "feed.atom contains a forbidden XML declaration",
            ),
            (
                "size limit",
                lambda _text: " " * 1_048_577,
                "feed.atom exceeds the 1048576-byte limit",
            ),
            (
                "malformed XML",
                lambda text: text.replace("</feed>", "", 1),
                "invalid feed.atom:",
            ),
        )
        for label, mutate, expected in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                feed_path = output / "feed.atom"
                original = feed_path.read_text(encoding="utf-8")
                mutated = mutate(original)
                self.assertNotEqual(mutated, original)
                feed_path.write_text(mutated, encoding="utf-8")
                failures = validate_public_site.validate(output)
                if expected.endswith(":"):
                    self.assertTrue(
                        any(failure.startswith(expected) for failure in failures),
                        failures,
                    )
                else:
                    self.assertIn(expected, failures)

    def test_validator_rejects_reviewed_updates_non_utf8_and_symlink_before_read(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)
            (output / "feed.atom").write_bytes(b"\xff\xfe\xfd")
            self.assertTrue(
                any(
                    failure.startswith("feed.atom is not UTF-8")
                    for failure in validate_public_site.validate(output)
                )
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)
            feed_path = output / "feed.atom"
            external = root / "external-reviewed-updates.atom"
            external.write_text(
                feed_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            feed_path.unlink()
            feed_path.symlink_to(external)
            self.assertIn(
                "feed.atom must be a regular non-symlink file",
                validate_public_site.validate(output),
            )

    def test_validator_requires_reviewed_updates_feed_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)
            (output / "feed.atom").unlink()
            self.assertIn(
                "missing required file: feed.atom",
                validate_public_site.validate(output),
            )

    def test_validator_requires_reviewed_updates_feed_discovery_surfaces(
        self,
    ) -> None:
        mutations = (
            (
                "home metadata",
                "index.html",
                lambda text: text.replace(
                    'title="fast-mlx reviewed updates" href="feed.atom"',
                    'title="unreviewed updates" href="feed.atom"',
                    1,
                ),
                "index.html metadata does not match reviewed contract",
            ),
            (
                "home action",
                "index.html",
                lambda text: text.replace(
                    '<a class="button secondary" href="feed.atom" ',
                    '<a class="button secondary" hidden href="feed.atom" ',
                    1,
                ),
                "index.html does not match the reviewed home page",
            ),
            (
                "release action",
                "releases/index.html",
                lambda text: text.replace(
                    '<a class="button secondary" href="../feed.atom" ',
                    '<a class="button secondary" hidden href="../feed.atom" ',
                    1,
                ),
                "releases/index.html does not expose the reviewed subscription actions",
            ),
            (
                "research action",
                "research/index.html",
                lambda text: text.replace(
                    '<a class="button secondary" href="../feed.atom" ',
                    '<a class="button secondary" hidden href="../feed.atom" ',
                    1,
                ),
                "research/index.html does not expose the reviewed subscription actions",
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

    def test_validator_rejects_research_atom_tampering(self) -> None:
        atom = lambda name: f"{{http://www.w3.org/2005/Atom}}{name}"

        def mutate_entries(text: str, mutation: str) -> str:
            feed = ET.fromstring(text)
            entries = feed.findall(atom("entry"))
            self.assertGreaterEqual(len(entries), 2)
            if mutation == "omit":
                feed.remove(entries[0])
            elif mutation == "reorder":
                feed.remove(entries[0])
                feed.remove(entries[1])
                feed.append(entries[1])
                feed.append(entries[0])
            elif mutation == "duplicate":
                feed.append(
                    ET.fromstring(ET.tostring(entries[0], encoding="unicode"))
                )
            else:
                self.fail(f"unknown research feed mutation: {mutation}")
            ET.register_namespace("", "http://www.w3.org/2005/Atom")
            ET.indent(feed, space="  ")
            return (
                '<?xml version="1.0" encoding="utf-8"?>\n'
                + ET.tostring(feed, encoding="unicode", short_empty_elements=True)
                + "\n"
            )

        mutations = (
            (
                "entry identity",
                lambda text: text.replace(
                    "https://bitworks-io.github.io/fast-mlx/research/"
                    "the-fastest-request-wasnt-the-fastest-service/",
                    "https://example.invalid/unreviewed-note/",
                    1,
                ),
                "research/feed.atom does not match the reviewed research catalog",
            ),
            (
                "missing entry",
                lambda text: mutate_entries(text, "omit"),
                "research/feed.atom does not match the reviewed research catalog",
            ),
            (
                "reordered entries",
                lambda text: mutate_entries(text, "reorder"),
                "research/feed.atom does not match the reviewed research catalog",
            ),
            (
                "duplicate entry",
                lambda text: mutate_entries(text, "duplicate"),
                "research/feed.atom does not match the reviewed research catalog",
            ),
            (
                "entry summary",
                lambda text: text.replace(
                    "On an Apple M5 Max at one request, our fastest exact path was ",
                    "Unreviewed automatic performance authority says ",
                    1,
                ),
                "research/feed.atom does not match the reviewed research catalog",
            ),
            (
                "active content",
                lambda text: text.replace(
                    "</entry>",
                    '<content type="html">&lt;script&gt;alert(1)&lt;/script&gt;</content>\n  </entry>',
                    1,
                ),
                "research/feed.atom does not match the reviewed research catalog",
            ),
            (
                "XML declaration",
                lambda text: text.replace(
                    "?>",
                    "?>\n<!DOCTYPE feed [<!ENTITY injected \"not-reviewed\">]>",
                    1,
                ),
                "research/feed.atom contains a forbidden XML declaration",
            ),
            (
                "size limit",
                lambda _text: " " * 1_048_577,
                "research/feed.atom exceeds the 1048576-byte limit",
            ),
            (
                "malformed XML",
                lambda text: text.replace("</feed>", "", 1),
                "invalid research/feed.atom:",
            ),
        )
        for label, mutate, expected in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                feed_path = output / "research/feed.atom"
                original = feed_path.read_text(encoding="utf-8")
                mutated = mutate(original)
                self.assertNotEqual(mutated, original)
                feed_path.write_text(mutated, encoding="utf-8")
                failures = validate_public_site.validate(output)
                if expected.endswith(":"):
                    self.assertTrue(
                        any(failure.startswith(expected) for failure in failures),
                        failures,
                    )
                else:
                    self.assertIn(expected, failures)

    def test_validator_requires_research_feed_discovery_surfaces(self) -> None:
        mutations = (
            (
                "archive metadata",
                "research/index.html",
                lambda text: text.replace(
                    'title="fast-mlx reviewed research" href="feed.atom"',
                    'title="unreviewed research" href="feed.atom"',
                    1,
                ),
                "research/index.html metadata does not match reviewed contract",
            ),
            (
                "article metadata",
                "research/the-proof-did-not-end-when-the-timer-did/index.html",
                lambda text: text.replace(
                    'title="fast-mlx reviewed research" href="../feed.atom"',
                    'title="fast-mlx reviewed research" href="../../feed.atom"',
                    1,
                ),
                "research/the-proof-did-not-end-when-the-timer-did/index.html "
                "metadata does not match reviewed contract",
            ),
            (
                "archive action",
                "research/index.html",
                lambda text: text.replace(
                    '<a class="button secondary" href="feed.atom" '
                    'type="application/atom+xml">Subscribe to reviewed research</a>',
                    '<a class="button secondary" href="index.json">'
                    "Subscribe to reviewed research</a>",
                    1,
                ),
                "research/index.html does not expose the reviewed subscription actions",
            ),
            (
                "hidden archive action",
                "research/index.html",
                lambda text: text.replace(
                    '<a class="button secondary" href="feed.atom" ',
                    '<a class="button secondary" hidden href="feed.atom" ',
                    1,
                ),
                "research/index.html does not expose the reviewed subscription actions",
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

    def test_validator_rejects_research_atom_non_utf8_and_symlink_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)
            (output / "research/feed.atom").write_bytes(b"\xff\xfe\xfd")
            self.assertTrue(
                any(
                    failure.startswith("research/feed.atom is not UTF-8")
                    for failure in validate_public_site.validate(output)
                )
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)
            feed_path = output / "research/feed.atom"
            external = root / "external-research.atom"
            external.write_text(feed_path.read_text(encoding="utf-8"), encoding="utf-8")
            feed_path.unlink()
            feed_path.symlink_to(external)
            self.assertIn(
                "research/feed.atom must be a regular non-symlink file",
                validate_public_site.validate(output),
            )

    def test_validator_rejects_joint_research_index_and_feeds_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)
            index_path = output / "research/index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            target = next(
                article
                for article in index["articles"]
                if article["path"]
                == "research/the-fastest-request-wasnt-the-fastest-service/"
            )
            original_summary = target["summary"]
            changed_summary = "Unreviewed automatic authority."
            target["summary"] = changed_summary
            index_path.write_text(
                json.dumps(index, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            feed_path = output / "research/feed.atom"
            feed = feed_path.read_text(encoding="utf-8")
            self.assertIn(original_summary, feed)
            feed_path.write_text(
                feed.replace(original_summary, changed_summary, 1),
                encoding="utf-8",
            )
            combined_path = output / "feed.atom"
            combined = combined_path.read_text(encoding="utf-8")
            self.assertIn(original_summary, combined)
            combined_path.write_text(
                combined.replace(original_summary, changed_summary, 1),
                encoding="utf-8",
            )

            failures = validate_public_site.validate(output)
            self.assertIn(
                "research/index.json does not match the reviewed research catalog",
                failures,
            )
            self.assertIn(
                "research/feed.atom does not match the reviewed research catalog",
                failures,
            )
            self.assertIn(
                "feed.atom does not match the reviewed release and research catalogs",
                failures,
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

    def test_validator_requires_research_feed_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)
            (output / "research/feed.atom").unlink()
            self.assertIn(
                "missing required file: research/feed.atom",
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

    def test_filterable_research_archive_is_generated_from_reviewed_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            articles = build_public_site.build_site(REPOSITORY_ROOT, output)
            archive = (output / "research/index.html").read_text(encoding="utf-8")

            self.assertIn(
                '<form class="research-controls" data-research-controls '
                'aria-label="Filter reviewed research" action="./" method="get">',
                archive,
            )
            self.assertIn(
                '<input id="research-query" name="q" type="search" '
                'maxlength="120" autocomplete="off">',
                archive,
            )
            self.assertIn('<select id="research-theme" name="theme">', archive)
            self.assertIn('data-research-reset type="reset"', archive)
            self.assertIn(
                'data-research-count aria-live="polite" aria-atomic="true"',
                archive,
            )
            self.assertIn(
                'data-research-empty hidden role="status"',
                archive,
            )
            self.assertIn(
                '<div class="research-grid" data-research-results role="list">',
                archive,
            )
            self.assertEqual(
                archive.count('class="note-card" role="listitem" data-research-card'),
                len(articles),
            )
            self.assertNotIn('class="note-card" hidden', archive)
            self.assertIn("JavaScript is optional", archive)
            self.assertIn(
                '<script src="../assets/research-explorer.js" defer></script>',
                archive,
            )

            positions = []
            for article in articles:
                search_text = " ".join(
                    f"{article.title} {article.summary} {article.theme}".split()
                )
                contract = (
                    f'data-research-path="{html.escape(article.public_path, quote=True)}" '
                    f'data-theme="{html.escape(article.theme, quote=True)}" '
                    f'data-search="{html.escape(search_text, quote=True)}"'
                )
                self.assertIn(contract, archive)
                self.assertIn(
                    f'<a href="{html.escape(article.slug, quote=True)}/">Read the note →</a>',
                    archive,
                )
                positions.append(archive.index(contract))
                article_page = (output / article.output_file).read_text(encoding="utf-8")
                self.assertNotIn("research-explorer.js", article_page)
            self.assertEqual(positions, sorted(positions))

            themes = sorted({article.theme for article in articles}, key=str.casefold)
            self.assertIn('<option value="">All themes</option>', archive)
            for theme in themes:
                escaped = html.escape(theme, quote=True)
                self.assertIn(f'<option value="{escaped}">{html.escape(theme)}</option>', archive)

    def test_research_archive_progressive_enhancement_is_bounded(self) -> None:
        script = (REPOSITORY_ROOT / "site/assets/research-explorer.js").read_text(
            encoding="utf-8"
        )
        stylesheet = (REPOSITORY_ROOT / "site/assets/site.css").read_text(
            encoding="utf-8"
        )

        self.assertIn('classList.add("research-enhanced")', script)
        self.assertIn("MAX_QUERY_LENGTH = 120", script)
        self.assertIn("if (raw.length > MAX_QUERY_LENGTH)", script)
        self.assertLess(
            script.index("if (raw.length > MAX_QUERY_LENGTH)"),
            script.index('raw.trim().replace(/\\s+/g, " ")'),
        )
        self.assertIn("URLSearchParams", script)
        self.assertIn("history.replaceState", script)
        self.assertIn("card.hidden", script)
        self.assertIn("textContent", script)
        self.assertNotIn("innerHTML", script)
        self.assertNotIn("fetch(", script)
        self.assertNotIn("localStorage", script)
        self.assertNotIn("sessionStorage", script)
        self.assertNotIn("eval(", script)
        self.assertIn(".research-controls", stylesheet)
        self.assertIn("html.research-enhanced .research-controls", stylesheet)

    def test_research_archive_escapes_catalog_text(self) -> None:
        hostile = build_public_site.Article(
            source=Path("docs/content/hostile.md"),
            source_name="docs/content/hostile.md",
            slug="hostile-note",
            title='Title "quoted" <script>alert(1)</script>',
            date="2026-08-13",
            theme="Theme & <unsafe>",
            summary='<img src=x onerror="alert(1)"> bounded summary.',
            reviewed_at="2026-08-13",
            body="",
        )
        rendered = build_public_site.render_research_archive([hostile])

        self.assertNotIn("<script>", rendered)
        self.assertNotIn("<img ", rendered)
        self.assertNotIn("<unsafe>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("&lt;img src=x onerror=&quot;alert(1)&quot;&gt;", rendered)
        self.assertIn('data-theme="Theme &amp; &lt;unsafe&gt;"', rendered)
        self.assertIn('data-search="Title &quot;quoted&quot; &lt;script&gt;', rendered)

    def test_research_archive_runtime_state_is_bounded(self) -> None:
        completed = subprocess.run(
            [
                "node",
                str(REPOSITORY_ROOT / "scripts/tests/benchmark_explorer_node_test.js"),
                str(REPOSITORY_ROOT / "site/assets/benchmark-explorer.js"),
                str(REPOSITORY_ROOT / "site/assets/research-explorer.js"),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=EXPLORER_RUNTIME_TIMEOUT_SECONDS,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("research explorer runtime checks passed", completed.stdout)

    def test_validator_requires_filterable_research_archive_outputs(self) -> None:
        for missing in ("research/index.html", "assets/research-explorer.js"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                (output / missing).unlink()
                self.assertIn(
                    f"missing required file: {missing}",
                    validate_public_site.validate(output),
                )

    def test_validator_rejects_research_archive_catalog_and_order_drift(self) -> None:
        mutations = (
            (
                'data-research-path="research/the-proof-did-not-end-when-the-timer-did/"',
                'data-research-path="research/unreviewed-diagnostic/"',
                "research archive card set does not match reviewed research catalog",
            ),
            (
                'data-search="The proof did not end when the timer did ',
                'data-search="Private diagnostic ',
                "research archive card 'research/the-proof-did-not-end-when-the-timer-did/' has the wrong search text",
            ),
            (
                'data-theme="Building a high-performance MLX inference engine in Swift; Serving big models on Apple Silicon; Rapid research integration — the flywheel"',
                'data-theme="Unreviewed theme"',
                "research archive card 'research/the-proof-did-not-end-when-the-timer-did/' has the wrong theme",
            ),
            (
                '<option value="The optimization dial — quantified precision-loss tuning">',
                '<option value="Unreviewed theme">',
                "research archive theme options do not match reviewed research catalog",
            ),
        )
        for before, after, expected in mutations:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                path = output / "research/index.html"
                page = path.read_text(encoding="utf-8")
                self.assertIn(before, page)
                path.write_text(page.replace(before, after, 1), encoding="utf-8")
                self.assertIn(expected, validate_public_site.validate(output))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)
            path = output / "research/index.html"
            page = path.read_text(encoding="utf-8")
            first = 'data-research-path="research/sampling-before-serving/"'
            second = 'data-research-path="research/the-repository-that-could-reproduce-itself/"'
            self.assertIn(first, page)
            self.assertIn(second, page)
            page = page.replace(first, "__FIRST_RESEARCH_PATH__", 1)
            page = page.replace(second, first, 1)
            page = page.replace("__FIRST_RESEARCH_PATH__", second, 1)
            path.write_text(page, encoding="utf-8")
            self.assertIn(
                "research archive cards are not ordered by descending article date",
                validate_public_site.validate(output),
            )

    def test_validator_preserves_research_archive_no_js_and_live_semantics(self) -> None:
        # Derive the first archive card's path from a reference render instead of
        # hard-coding it: the newest reviewed article occupies the first card and
        # will keep drifting as the loop publishes work.
        with tempfile.TemporaryDirectory() as reference_directory:
            reference_output = Path(reference_directory) / "site"
            reference_output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, reference_output)
            reference_index = json.loads(
                (reference_output / "research/index.json").read_text(
                    encoding="utf-8"
                )
            )
            first_article_path = reference_index["articles"][0]["path"]

        mutations = (
            (
                '<article class="note-card" role="listitem" data-research-card',
                '<article class="note-card" hidden role="listitem" data-research-card',
                f"research archive card {first_article_path!r} is hidden before enhancement",
            ),
            (
                '<article class="note-card" role="listitem" data-research-card',
                '<article class="note-card" style="display:none" role="listitem" data-research-card',
                f"research archive card {first_article_path!r} is hidden before enhancement",
            ),
            (
                '<article class="note-card" role="listitem" data-research-card',
                '<article class="note-card" aria-hidden="true" role="listitem" data-research-card',
                f"research archive card {first_article_path!r} is hidden before enhancement",
            ),
            (
                '<article class="note-card" role="listitem" data-research-card',
                '<article class="note-card" inert role="listitem" data-research-card',
                f"research archive card {first_article_path!r} is hidden before enhancement",
            ),
            (
                '<div class="research-grid" data-research-results role="list">',
                '<div class="research-grid" data-research-results role="list" hidden>',
                f"research archive card {first_article_path!r} is hidden before enhancement",
            ),
            (
                '<h2>The proof did not end when the timer did</h2>',
                '<h2 style="display:none">The proof did not end when the timer did</h2>',
                "research archive card 'research/the-proof-did-not-end-when-the-timer-did/' is hidden before enhancement",
            ),
            (
                '<button class="button secondary research-reset" data-research-reset type="reset">Clear filters</button>',
                '',
                "research archive has no exact visible reset control",
            ),
            (
                'data-research-count aria-live="polite" aria-atomic="true"',
                'data-research-count aria-atomic="true"',
                "research archive has no live result count",
            ),
            (
                'data-research-empty hidden role="status"',
                'data-research-empty role="status"',
                "research archive has no hidden status empty state",
            ),
            (
                'data-research-empty hidden role="status"',
                'data-research-empty hidden',
                "research archive has no hidden status empty state",
            ),
        )
        for before, after, expected in mutations:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                path = output / "research/index.html"
                page = path.read_text(encoding="utf-8")
                self.assertIn(before, page)
                path.write_text(page.replace(before, after, 1), encoding="utf-8")
                self.assertIn(expected, validate_public_site.validate(output))

    def test_validator_requires_research_script_timing_and_placement(self) -> None:
        script = '<script src="../assets/research-explorer.js" defer></script>'
        mutations = (
            '<script src="../assets/research-explorer.js"></script>',
            None,
        )
        for replacement in mutations:
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                path = output / "research/index.html"
                page = path.read_text(encoding="utf-8")
                self.assertEqual(page.count(script), 1)
                if replacement is None:
                    page = page.replace(script, "", 1).replace(
                        "</head>", script + "\n  </head>", 1
                    )
                else:
                    page = page.replace(script, replacement, 1)
                path.write_text(page, encoding="utf-8")
                self.assertIn(
                    "research archive does not load only its reviewed script",
                    validate_public_site.validate(output),
                )

    def test_validator_rejects_joint_research_index_and_archive_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)

            index_path = output / "research/index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            target = index["articles"][0]
            original = target["summary"]
            changed = "Unreviewed automatic publication authority."
            target["summary"] = changed
            index_path.write_text(
                json.dumps(index, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            archive_path = output / "research/index.html"
            archive = archive_path.read_text(encoding="utf-8")
            escaped_original = html.escape(original, quote=True)
            escaped_changed = html.escape(changed, quote=True)
            self.assertIn(escaped_original, archive)
            archive_path.write_text(
                archive.replace(escaped_original, escaped_changed, 2),
                encoding="utf-8",
            )
            failures = validate_public_site.validate(output)
            self.assertIn(
                "research/index.json does not match the reviewed research catalog",
                failures,
            )
            self.assertIn(
                f"research archive card {target['path']!r} has drifted reviewed text",
                failures,
            )

    def test_validator_rejects_research_explorer_script_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)
            script = output / "assets/research-explorer.js"
            script.write_text(
                script.read_text(encoding="utf-8") + "\n// unreviewed drift\n",
                encoding="utf-8",
            )
            self.assertIn(
                "assets/research-explorer.js does not match the reviewed script",
                validate_public_site.validate(output),
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

    def test_benchmark_detail_pages_are_generated_from_reviewed_highlights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)

            capabilities = json.loads(
                (output / "capabilities/index.json").read_text(encoding="utf-8")
            )
            explorer = (output / "benchmarks/index.html").read_text(encoding="utf-8")
            sitemap = (output / "sitemap.xml").read_text(encoding="utf-8")
            llms = (output / "llms.txt").read_text(encoding="utf-8")
            decision_labels = {
                "promoted-scoped": "PROMOTED · SCOPED",
                "shelved": "SHELVED",
            }

            for highlight in capabilities["performanceHighlights"]:
                identifier = highlight["id"]
                public_path = f"benchmarks/{identifier}/"
                detail_path = output / public_path / "index.html"
                self.assertTrue(
                    detail_path.is_file(),
                    f"missing reviewed benchmark detail page: {public_path}",
                )
                detail = detail_path.read_text(encoding="utf-8")
                self.assertIn(
                    f'data-benchmark-detail data-highlight-id="{identifier}"', detail
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
                    self.assertIn(html.escape(highlight[field]), detail)
                self.assertIn(decision_labels[highlight["decision"]], detail)
                self.assertIn(
                    f'<time datetime="{highlight["date"]}">{highlight["date"]}</time>',
                    detail,
                )
                self.assertIn(
                    f'href="../../{highlight["evidence"]["path"]}"', detail
                )
                self.assertIn('href="../">Back to all reviewed results</a>', detail)
                self.assertIn(
                    "This page does not normalize, rank, aggregate, or recompute results.",
                    detail,
                )
                self.assertEqual(detail.count("<h1>"), 1)
                self.assertNotIn("benchmark-explorer.js", detail)
                self.assertIn(f'href="{identifier}/"', explorer)
                self.assertIn(
                    "https://bitworks-io.github.io/fast-mlx/" + public_path,
                    sitemap,
                )
                self.assertIn(f"- /{public_path}:", llms)

    def test_validator_rejects_benchmark_detail_contract_drift(self) -> None:
        identifier = "pld-echo-throughput"
        mutations = (
            (
                'data-highlight-id="pld-echo-throughput"',
                'data-highlight-id="unreviewed-result"',
                f"benchmark detail '{identifier}' has the wrong id",
            ),
            (
                '<div class="metric">+100.5%</div>',
                '<div class="metric">+999%</div>',
                f"benchmark detail '{identifier}' has drifted reviewed text",
            ),
            (
                '<div><dt>Model</dt><dd>Qwen3-32B-4bit</dd></div>',
                '<div><dt>Model</dt><dd>Unreviewed model</dd></div>',
                f"benchmark detail '{identifier}' has the wrong context fields",
            ),
            (
                'href="../../research/when-zero-speculation-costs-two-percent/"',
                'href="../../research/the-wall-that-wasnt/"',
                f"benchmark detail '{identifier}' has the wrong action links",
            ),
            (
                'href="../../methodology/"',
                'href="../../capabilities/"',
                f"benchmark detail '{identifier}' has the wrong methodology link",
            ),
            (
                "This page does not normalize, rank, aggregate, or recompute results.",
                "This page establishes a ranked production result.",
                f"benchmark detail '{identifier}' has the wrong claim boundary",
            ),
            (
                '<section class="page-hero shell benchmark-detail" ',
                '<section class="page-hero shell benchmark-detail" hidden ',
                f"benchmark detail '{identifier}' is hidden",
            ),
            (
                "</body>",
                '<script src="../../assets/benchmark-explorer.js"></script></body>',
                f"benchmark detail '{identifier}' must not load scripts",
            ),
        )
        for before, after, expected_failure in mutations:
            with self.subTest(expected_failure=expected_failure), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                path = output / "benchmarks" / identifier / "index.html"
                page = path.read_text(encoding="utf-8")
                self.assertIn(before, page)
                path.write_text(page.replace(before, after, 1), encoding="utf-8")
                self.assertIn(expected_failure, validate_public_site.validate(output))

    def test_validator_rejects_missing_extra_and_symlink_benchmark_details(self) -> None:
        cases = ("missing", "extra", "symlink")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "site"
                output.mkdir()
                build_public_site.build_site(REPOSITORY_ROOT, output)
                detail = output / "benchmarks/pld-echo-throughput/index.html"
                if case == "missing":
                    detail.unlink()
                    expected = (
                        "missing required file: "
                        "benchmarks/pld-echo-throughput/index.html"
                    )
                elif case == "extra":
                    extra = output / "benchmarks/unreviewed-result"
                    extra.mkdir()
                    (extra / "index.html").write_text(
                        detail.read_text(encoding="utf-8"), encoding="utf-8"
                    )
                    expected = "unexpected benchmark route outside reviewed set: unreviewed-result"
                else:
                    detail.unlink()
                    detail.symlink_to("../continuous-batch-c2-throughput/index.html")
                    expected = (
                        "benchmarks/pld-echo-throughput/index.html must be a regular "
                        "non-symlink file"
                    )
                self.assertIn(expected, validate_public_site.validate(output))

    def test_validator_rejects_joint_benchmark_index_and_page_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)

            index_path = output / "capabilities/index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["performanceHighlights"][0]["metric"] = "+999%"
            index_path.write_text(
                json.dumps(index, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            for relative in (
                "benchmarks/index.html",
                "benchmarks/pld-echo-throughput/index.html",
            ):
                path = output / relative
                page = path.read_text(encoding="utf-8")
                path.write_text(
                    page.replace(">+100.5%<", ">+999%<", 1), encoding="utf-8"
                )

            self.assertIn(
                "capabilities/index.json performance highlights do not match reviewed benchmark highlights",
                validate_public_site.validate(output),
            )

    def test_validator_requires_exact_benchmark_detail_links_from_explorer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)
            path = output / "benchmarks/index.html"
            page = path.read_text(encoding="utf-8")
            path.write_text(
                page.replace(
                    'href="http-sse-operational-soak/"',
                    'href="../capabilities/"',
                    1,
                ),
                encoding="utf-8",
            )
            self.assertIn(
                "benchmark explorer card 'http-sse-operational-soak' has the wrong detail link",
                validate_public_site.validate(output),
            )

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
                str(REPOSITORY_ROOT / "site/assets/research-explorer.js"),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=EXPLORER_RUNTIME_TIMEOUT_SECONDS,
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

    def test_publication_manifest_symlink_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            external = root / "external.json"
            external.write_text(
                (REPOSITORY_ROOT / "site/publications.json").read_text(
                    encoding="utf-8"
                ),
                encoding="utf-8",
            )
            manifest = root / "site/publications.json"
            manifest.parent.mkdir(parents=True)
            manifest.symlink_to(external)

            with self.assertRaises(SystemExit):
                build_public_site.load_articles(root)

    def test_publication_manifest_is_bounded_utf8_and_exact_schema(self) -> None:
        original = json.loads(
            (REPOSITORY_ROOT / "site/publications.json").read_text(encoding="utf-8")
        )
        mutations = (
            ("oversized", b" " * 1_048_577),
            ("non-UTF-8", b"\xff\xfe\xfd"),
            (
                "extra top-level key",
                json.dumps({**original, "automatic": True}).encode("utf-8"),
            ),
            (
                "extra article key",
                json.dumps(
                    {
                        **original,
                        "articles": [
                            {**original["articles"][0], "externalFeed": True},
                            *original["articles"][1:],
                        ],
                    }
                ).encode("utf-8"),
            ),
        )
        for label, raw in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = root / "site/publications.json"
                manifest.parent.mkdir(parents=True)
                manifest.write_bytes(raw)
                with self.assertRaises(SystemExit):
                    build_public_site.load_articles(root)

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

    @staticmethod
    def quality_guide_manifest() -> dict[str, object]:
        return json.loads(
            (REPOSITORY_ROOT / "scripts/tests/fixtures/quality-guides.sample.json").read_text(
                encoding="utf-8"
            )
        )

    @staticmethod
    def write_quality_guide_manifest(root: Path, manifest: dict[str, object]) -> None:
        path = root / "site/quality-guides.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest), encoding="utf-8")

    @staticmethod
    def quality_card_fragment(page: str, card_id: str) -> str:
        """Isolate one rendered `<article data-quality-card="...">` block so a
        speed-line assertion cannot pass or fail on a different card's copy."""
        match = re.search(
            r'<article class="quality-card" data-quality-card="'
            + re.escape(card_id)
            + r'"[^>]*>(.*?)</article>',
            page,
            re.S,
        )
        assert match is not None, f"card {card_id!r} not found in rendered page"
        return match.group(1)

    @staticmethod
    def expert_stream_card_manifest_document() -> dict[str, object]:
        """A `fast-mlx-quality-card-v1` fixture document holding one NO_GO
        card for the `expert-stream` residency, shaped like a real internal
        measured card but with a synthetic model identity and
        illustrative-only wording."""
        return {
            "schema": "fast-mlx-quality-card-v1",
            "generatedAt": "2026-01-01T00:00:00Z",
            "cards": [
                {
                    "id": "example-moe-q4-stream@m3ultra",
                    "model": {
                        "family": "Example-MoE",
                        "repo": "example-org/example-moe-gguf",
                        "hfPin": "0123456789abcdef0123456789abcdef01234567",
                    },
                    "config": {
                        "quant": {
                            "bits": 4,
                            "groupSize": None,
                            "mixedBit": True,
                            "note": "fixture: quantized routed experts",
                        },
                        "enhancement": "none",
                        "hardwareClass": "example-hardware",
                        "residency": "expert-stream",
                    },
                    "verdict": "NO_GO",
                    "admission": {
                        "default": False,
                        "optIn": True,
                        "reason": "fixture: output differed on 3 of 8 prompts versus the same pack held in memory",
                    },
                    "legible": {
                        "tier": "Unquantified",
                        "headline": "fixture: expert-stream residency changed greedy output on 3 of 8 prompts versus the same pack held in memory.",
                        "nextWordDrift": None,
                        "regressionFocus": "fixture: illustrative regression focus text",
                        "example": {
                            "status": "pending",
                            "prompt": None,
                            "referenceOutput": None,
                            "configOutput": None,
                            "note": "fixture: illustrative placeholder",
                        },
                        "benefit": {
                            "fit": None,
                            "speedX": 0.5,
                            "speedXStatus": "fixture: illustrative only, not a measurement",
                        },
                    },
                    "rawMetrics": {
                        "greedyDivergentPrompts": "3/8",
                        "comparedTokens": 100,
                        "magnitude": "fixture: not collected",
                    },
                    "provenance": {
                        "source": "fast-mlx-measured",
                        "vendor": None,
                        "method": "fixture: synthetic test data, not a real measurement",
                        "confound": None,
                        "hardware": "fixture",
                        "harnessGitSHA": "0000000000000000000000000000000000000000",
                        "corpusId": "fixture-corpus",
                        "sourceVerdict": None,
                        "measuredAt": "2026-01-01T00:00:00Z",
                    },
                    "boundary": {
                        "scope": "fixture: serving-admission quality signal for expert-stream residency",
                        "unmeasured": ["fixture: illustrative unmeasured item"],
                    },
                }
            ],
        }

    def test_quality_guide_is_skipped_gracefully_when_manifest_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "site").mkdir()
            self.assertIsNone(build_public_site.load_quality_guides(root))

    def test_quality_guide_manifest_extended_private_markers_are_refused(self) -> None:
        # Private serving-engine names are refused EXCEPT inside the exact public
        # token "fastmlx-serve"; a bare ".25<0-3>" host shorthand is refused
        # everywhere. The names are assembled from parts so this public source
        # never carries them as literals (the publication leak sweep greps for them).
        engine = "ml" + "x-serve"
        refused = {
            "bare engine name": f"org/{engine}",
            "mixed-case engine name": f"org/{engine.upper()}",
            "embedded engine name": f"org/x-{engine.title()}-y",
            "first component substring": "the " + "om" + "lx engine",
            "second component substring": "an " + "mt" + "plx build",
            "host shorthand .252": "reachable at host .252 today",
            "host shorthand .250": "staged on host .250",
        }
        for label, marker_text in refused.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.quality_guide_manifest()
                manifest["cards"][0]["provenance"]["confound"] = marker_text
                self.write_quality_guide_manifest(root, manifest)
                with self.assertRaises(SystemExit):
                    build_public_site.load_quality_guides(root)

    def test_quality_guide_manifest_allows_fastmlx_serve_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.quality_guide_manifest()
            manifest["cards"][0]["provenance"]["confound"] = "served by fastmlx-serve"
            self.write_quality_guide_manifest(root, manifest)
            loaded = build_public_site.load_quality_guides(root)
            self.assertIsNotNone(loaded)

    def test_real_quality_guide_manifest_passes_extended_private_marker_scan(self) -> None:
        loaded = build_public_site.load_quality_guides(REPOSITORY_ROOT)
        self.assertIsNotNone(loaded)

    def test_quality_guide_renders_legible_cards_from_manifest(self) -> None:
        manifest = self.quality_guide_manifest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_quality_guide_manifest(root, manifest)
            loaded = build_public_site.load_quality_guides(root)

        self.assertIsNotNone(loaded)
        cards = loaded["cards"]
        self.assertEqual(len(cards), 4)
        page = build_public_site.render_quality_guide(cards)

        no_go_card = next(card for card in cards if card["verdict"] == "NO_GO")
        reference_card = next(card for card in cards if card["verdict"] == "REFERENCE")
        vendor_card = next(
            card for card in cards if card["provenance"]["source"] == "vendor-reported"
        )

        # Every card's tier, headline, model family, and provenance method render.
        for card in cards:
            self.assertIn(card["legible"]["tier"].upper(), page)
            self.assertIn(card["legible"]["headline"], page)
            self.assertIn(html.escape(card["model"]["family"]), page)
            self.assertIn(html.escape(card["provenance"]["method"]), page)

        # The NO_GO card reads as an opt-in choice with its stated cost, never "broken".
        self.assertIn(no_go_card["admission"]["reason"], page)
        self.assertNotIn("broken", page.lower())

        # Both a fast-mlx-measured label and a vendor-reported + "Unsloth" label render.
        self.assertIn("fast-mlx-measured", page)
        self.assertIn("vendor-reported", page)
        self.assertIn("Unsloth", page)
        self.assertEqual(vendor_card["provenance"]["vendor"], "Unsloth")

        # rawMetrics is only ever available behind a <details> expander.
        self.assertIn("<details", page)
        self.assertIn("Raw metrics", page)
        for card in cards:
            for value in card["rawMetrics"].values():
                self.assertNotIn(str(value), page.split("<details", 1)[0])

        # A null speedX renders its speedXStatus text and no fabricated multiplier.
        self.assertIsNone(no_go_card["legible"]["benefit"]["speedX"])
        self.assertIn(no_go_card["legible"]["benefit"]["speedXStatus"], page)

        # The boundary scope caveat renders for every card.
        for card in cards:
            self.assertIn(card["boundary"]["scope"], page)

        self.assertIn(reference_card["legible"]["headline"], page)

    def test_quality_speed_line_below_one_never_says_faster(self) -> None:
        """A measured speedX below 1.0 is a slowdown. Rendering it with the
        word "faster" would publish a falsehood — the reader would read a
        regression as a gain."""
        manifest = self.quality_guide_manifest()
        card = manifest["cards"][0]
        card["legible"]["benefit"]["speedX"] = 0.485
        card["legible"]["benefit"]["speedXStatus"] = (
            "measured on m3ultra, 4-bit vs reference, 40-prompt corpus"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_quality_guide_manifest(root, manifest)
            loaded = build_public_site.load_quality_guides(root)

        page = build_public_site.render_quality_guide(loaded["cards"])
        fragment = self.quality_card_fragment(page, str(card["id"]))
        self.assertNotIn("faster", fragment.lower())

    def test_quality_speed_line_always_carries_speedxstatus(self) -> None:
        """speedXStatus is the only place the measurement boundary (host,
        engine build, flags, prompt count) lives. It must render even once a
        numeric speedX lands, not only in the null/not-measured case."""
        manifest = self.quality_guide_manifest()
        card = manifest["cards"][0]
        card["legible"]["benefit"]["speedX"] = 1.6
        card["legible"]["benefit"]["speedXStatus"] = (
            "measured on m3ultra, 4-bit vs reference, 40-prompt corpus"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_quality_guide_manifest(root, manifest)
            loaded = build_public_site.load_quality_guides(root)

        page = build_public_site.render_quality_guide(loaded["cards"])
        fragment = self.quality_card_fragment(page, str(card["id"]))
        self.assertIn(card["legible"]["benefit"]["speedXStatus"], fragment)

    def test_validator_rejects_quality_page_missing_speedxstatus_even_when_measured(
        self,
    ) -> None:
        """Mirrors the speedXStatus rendering check but for the measured case:
        the validator must not skip the speedXStatus requirement just because
        speedX is a number — that is exactly the gap that let the boundary
        vanish once a real measurement lands."""
        manifest = self.quality_guide_manifest()
        target_id = str(manifest["cards"][0]["id"])
        manifest["cards"][0]["legible"]["benefit"]["speedX"] = 1.6
        manifest["cards"][0]["legible"]["benefit"]["speedXStatus"] = (
            "measured on m3ultra, 4-bit vs reference, 40-prompt corpus"
        )
        original_render_quality_guide = build_public_site.render_quality_guide

        def render_without_speed_dd(cards: object) -> str:
            # Strip the Speed <dd> from ONLY the measured (speedX=1.6) card so
            # a pass here cannot be explained by one of the other, still-null
            # cards tripping the existing null-case check instead.
            page = original_render_quality_guide(cards)  # type: ignore[arg-type]
            fragment = self.quality_card_fragment(page, target_id)
            stripped = re.sub(r"<div><dt>Speed</dt><dd>.*?</dd></div>", "", fragment)
            return page.replace(fragment, stripped, 1)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            with mock.patch.object(
                build_public_site, "load_quality_guides", return_value=manifest
            ), mock.patch.object(
                build_public_site,
                "render_quality_guide",
                side_effect=render_without_speed_dd,
            ):
                build_public_site.build_site(REPOSITORY_ROOT, output)

            failures = validate_public_site.validate_quality_guide_page(output)
            self.assertTrue(
                any(
                    target_id in f and "does not render its speedXStatus text" in f
                    for f in failures
                ),
                failures,
            )

    def render_card_fragment_with_example(self, example: dict[str, object]) -> tuple[str, str]:
        """Render card 0 from the sample manifest with its `legible.example`
        replaced, and return (isolated fragment, card id)."""
        manifest = self.quality_guide_manifest()
        card = manifest["cards"][0]
        card["legible"]["example"] = example
        card_id = str(card["id"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_quality_guide_manifest(root, manifest)
            loaded = build_public_site.load_quality_guides(root)
        page = build_public_site.render_quality_guide(loaded["cards"])
        return self.quality_card_fragment(page, card_id), card_id

    def test_quality_example_measured_renders_side_by_side(self) -> None:
        """Acceptance criterion 1: a `measured` example renders its context,
        both outputs, and its note -- the "visceral side-by-side" the schema
        requires and the renderer was silently dropping."""
        example = {
            "status": "measured",
            "prompt": "fixture-context: the reasoning preamble text",
            "referenceOutput": "fixture-reference-output: the peer pack picked so",
            "configOutput": "fixture-config-output: this pack picked let",
            "note": "fixture-note: first disagreement, prompt 0 position 4",
        }
        fragment, _ = self.render_card_fragment_with_example(example)
        self.assertIn(example["prompt"], fragment)
        self.assertIn(example["referenceOutput"], fragment)
        self.assertIn(example["configOutput"], fragment)
        self.assertIn(example["note"], fragment)

    def test_quality_example_pending_renders_explicit_not_yet_extracted_line(self) -> None:
        """Acceptance criterion 2: a `pending` example must not render as
        silence -- silence reads as "nothing differs" -- mirroring the
        Unquantified branch's reasoning at :2380-2386."""
        example = {
            "status": "pending",
            "prompt": None,
            "referenceOutput": None,
            "configOutput": None,
            "note": "fixture: extract from task-evidence jsonl",
        }
        fragment, _ = self.render_card_fragment_with_example(example)
        self.assertIn("no side-by-side", fragment.lower())
        # None of the measured-only labels leak through when there is
        # nothing to show for prompt/referenceOutput/configOutput.
        self.assertNotIn("Other arm said", fragment)
        self.assertNotIn("This pack said", fragment)
        self.assertNotIn("Context:", fragment)

    def test_quality_example_measured_and_pending_fragments_differ_and_are_nonempty(
        self,
    ) -> None:
        """Differ control: a renderer that emits a constant, or emits nothing
        in both arms, must fail this."""
        measured_example = {
            "status": "measured",
            "prompt": "fixture-context: differ-control prompt",
            "referenceOutput": "fixture-reference-output: differ-control reference",
            "configOutput": "fixture-config-output: differ-control config",
            "note": "fixture-note: differ-control note",
        }
        pending_example = {
            "status": "pending",
            "prompt": None,
            "referenceOutput": None,
            "configOutput": None,
            "note": "fixture: differ-control pending note",
        }
        measured_fragment, _ = self.render_card_fragment_with_example(measured_example)
        pending_fragment, _ = self.render_card_fragment_with_example(pending_example)
        self.assertTrue(measured_fragment.strip())
        self.assertTrue(pending_fragment.strip())
        self.assertNotEqual(measured_fragment, pending_fragment)

    def test_quality_example_illustrative_is_labeled_unmistakably_not_measured(self) -> None:
        """Acceptance criterion 3: `illustrative` renders, but labeled so a
        reader can never mistake it for a measured case."""
        example = {
            "status": "illustrative",
            "prompt": None,
            "referenceOutput": None,
            "configOutput": None,
            "note": "fixture: illustrative placeholder note",
        }
        fragment, _ = self.render_card_fragment_with_example(example)
        self.assertIn(example["note"], fragment)
        self.assertIn("illustrative", fragment.lower())
        self.assertIn("not a measured case", fragment.lower())

    def test_quality_example_measured_prompt_is_html_escaped(self) -> None:
        """A real card's context begins with a literal `<think>` token; an
        unescaped render path would inject a tag into the public page."""
        example = {
            "status": "measured",
            "prompt": "<think> Okay,",
            "referenceOutput": "fixture-reference-output: escaping-control reference",
            "configOutput": "fixture-config-output: escaping-control config",
            "note": "fixture-note: escaping-control note",
        }
        fragment, _ = self.render_card_fragment_with_example(example)
        self.assertIn("&lt;think&gt;", fragment)
        self.assertNotIn("<think>", fragment)

    def test_quality_guide_reference_and_exact_cards_render_null_safely(self) -> None:
        """REFERENCE (null drift/regression) and EXACT/MTP (mostly-null card) render
        with no leaked "None" and the documented null-specific copy."""

        manifest = self.quality_guide_manifest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_quality_guide_manifest(root, manifest)
            loaded = build_public_site.load_quality_guides(root)

        cards = loaded["cards"]
        reference_card = next(card for card in cards if card["verdict"] == "REFERENCE")
        exact_card = next(card for card in cards if card["verdict"] == "EXACT")
        self.assertIsNone(reference_card["legible"]["nextWordDrift"])
        self.assertIsNone(reference_card["legible"]["regressionFocus"])
        self.assertIsNone(exact_card["config"]["quant"])
        self.assertIsNone(exact_card["legible"]["nextWordDrift"]["oneInK"])
        self.assertIsNone(exact_card["legible"]["regressionFocus"])
        self.assertIsNone(exact_card["legible"]["benefit"]["fit"])
        self.assertEqual(exact_card["rawMetrics"], {})
        self.assertIsNone(exact_card["provenance"]["harnessGitSHA"])
        self.assertIsNone(exact_card["provenance"]["measuredAt"])

        page = build_public_site.render_quality_guide(cards)

        def card_html(card_id: str) -> str:
            match = re.search(
                r'<article class="quality-card" data-quality-card="'
                + re.escape(card_id)
                + r'"[^>]*>(.*?)</article>',
                page,
                re.S,
            )
            self.assertIsNotNone(match, f"card {card_id!r} not found in rendered page")
            return match.group(1)

        reference_html = card_html(str(reference_card["id"]))
        self.assertNotIn("None", reference_html)
        self.assertNotIn("quality-drift", reference_html)
        self.assertNotIn("Regression focus", reference_html)

        exact_html = card_html(str(exact_card["id"]))
        self.assertNotIn("None", exact_html)
        self.assertIn("Identical output", exact_html)
        self.assertIn("no next-word drift", exact_html)
        self.assertNotIn("Regression focus", exact_html)
        self.assertNotIn("<dt>Fit</dt>", exact_html)
        self.assertIn("<details", exact_html)
        self.assertIn("{}", exact_html)

        # Both cards pass the fail-closed loader and validator.
        manifest_failures = validate_public_site.validate_quality_guide_manifest(loaded)
        self.assertEqual(manifest_failures, [])

    def test_quality_guide_page_is_generated_when_manifest_is_present(self) -> None:
        manifest = self.quality_guide_manifest()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            with mock.patch.object(
                build_public_site, "load_quality_guides", return_value=manifest
            ):
                build_public_site.build_site(REPOSITORY_ROOT, output)

            self.assertTrue((output / "quality/index.html").is_file())
            self.assertTrue((output / "quality/index.json").is_file())
            failures = validate_public_site.validate(output)
            self.assertEqual(failures, [])

    def test_validator_rejects_quality_page_missing_family_or_method(self) -> None:
        """Mirrors the speedXStatus rendering check: a regression that stops the
        renderer emitting a card's model family or provenance method must be
        caught by the fail-closed page validator, not just eyeballed."""

        manifest = self.quality_guide_manifest()
        original_render_quality_guide = build_public_site.render_quality_guide

        def render_without_family_or_method(cards: object) -> str:
            page = original_render_quality_guide(cards)  # type: ignore[arg-type]
            page = re.sub(r'<p class="quality-family">.*?</p>', "", page)
            page = re.sub(r'<p class="quality-method">.*?</p>', "", page)
            return page

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            with mock.patch.object(
                build_public_site, "load_quality_guides", return_value=manifest
            ), mock.patch.object(
                build_public_site,
                "render_quality_guide",
                side_effect=render_without_family_or_method,
            ):
                build_public_site.build_site(REPOSITORY_ROOT, output)

            failures = validate_public_site.validate_quality_guide_page(output)
            self.assertTrue(
                any("does not render its model family" in f for f in failures), failures
            )
            self.assertTrue(
                any("does not render its provenance method" in f for f in failures), failures
            )

    def test_validator_rejects_unknown_quality_verdict_and_provenance_source(self) -> None:
        base_manifest = self.quality_guide_manifest()

        unknown_verdict = json.loads(json.dumps(base_manifest))
        unknown_verdict["cards"][0]["verdict"] = "MAYBE"
        failures = validate_public_site.validate_quality_guide_manifest(unknown_verdict)
        self.assertTrue(
            any("unknown verdict" in failure for failure in failures), failures
        )

        unknown_source = json.loads(json.dumps(base_manifest))
        unknown_source["cards"][2]["provenance"]["source"] = "vibes-based"
        failures = validate_public_site.validate_quality_guide_manifest(unknown_source)
        self.assertTrue(
            any("unknown source" in failure for failure in failures), failures
        )

        # And the build-time loader fails closed the same way.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_quality_guide_manifest(root, unknown_verdict)
            with self.assertRaises(SystemExit):
                build_public_site.load_quality_guides(root)

    # ------------------------------------------------------------------
    # config.residency: optional; "resident" | "expert-stream" when present;
    # unknown values refuse. Existing cards (no residency key) still pass.
    # ------------------------------------------------------------------
    def test_quality_guide_config_accepts_optional_residency(self):
        manifest = self.quality_guide_manifest()
        manifest["cards"][0]["config"]["residency"] = "expert-stream"
        manifest["cards"][1]["config"]["residency"] = "resident"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_quality_guide_manifest(root, manifest)
            loaded = build_public_site.load_quality_guides(root)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["cards"][0]["config"]["residency"], "expert-stream")
        self.assertEqual(loaded["cards"][1]["config"]["residency"], "resident")

    def test_quality_guide_config_rejects_unknown_residency(self):
        manifest = self.quality_guide_manifest()
        manifest["cards"][0]["config"]["residency"] = "bogus"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_quality_guide_manifest(root, manifest)
            with self.assertRaises(SystemExit):
                build_public_site.load_quality_guides(root)

    # ------------------------------------------------------------------
    # "Unquantified" tier: only a NO_GO card with a null nextWordDrift; and
    # a NO_GO/PASS card with a null nextWordDrift must be tiered Unquantified
    # (so PASS can never carry a null nextWordDrift).
    # ------------------------------------------------------------------
    def test_quality_guide_unquantified_tier_requires_no_go_verdict(self):
        manifest = self.quality_guide_manifest()
        pass_card = manifest["cards"][2]
        self.assertEqual(pass_card["verdict"], "PASS")
        pass_card["legible"]["tier"] = "Unquantified"
        pass_card["legible"]["nextWordDrift"] = None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_quality_guide_manifest(root, manifest)
            with self.assertRaises(SystemExit):
                build_public_site.load_quality_guides(root)

    def test_quality_guide_unquantified_tier_requires_null_drift(self):
        manifest = self.quality_guide_manifest()
        no_go_card = manifest["cards"][0]
        self.assertEqual(no_go_card["verdict"], "NO_GO")
        self.assertIsNotNone(no_go_card["legible"]["nextWordDrift"])
        no_go_card["legible"]["tier"] = "Unquantified"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_quality_guide_manifest(root, manifest)
            with self.assertRaises(SystemExit):
                build_public_site.load_quality_guides(root)

    def test_quality_guide_null_drift_on_no_go_or_pass_requires_unquantified_tier(self):
        manifest = self.quality_guide_manifest()
        no_go_card = manifest["cards"][0]
        no_go_card["legible"]["nextWordDrift"] = None
        # tier stays "Noticeable" -- a null drift with a non-Unquantified tier refuses.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_quality_guide_manifest(root, manifest)
            with self.assertRaises(SystemExit):
                build_public_site.load_quality_guides(root)

        manifest2 = self.quality_guide_manifest()
        manifest2["cards"][2]["legible"]["nextWordDrift"] = None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_quality_guide_manifest(root, manifest2)
            with self.assertRaises(SystemExit):
                build_public_site.load_quality_guides(root)

    def test_quality_guide_unquantified_no_go_card_loads_and_renders_honestly(self):
        manifest = self.quality_guide_manifest()
        no_go_card = manifest["cards"][0]
        no_go_card["legible"]["tier"] = "Unquantified"
        no_go_card["legible"]["nextWordDrift"] = None
        # The fixture's stock headline literally says "1 word in 6"; replace it so
        # the "no fabricated 1-in-K line" assertion below tests the drift
        # paragraph, not leftover fixture prose.
        no_go_card["legible"]["headline"] = (
            "Output differed from the reference on SSD expert streaming."
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_quality_guide_manifest(root, manifest)
            loaded = build_public_site.load_quality_guides(root)
        self.assertIsNotNone(loaded)

        page = build_public_site.render_quality_guide(loaded["cards"])
        match = re.search(
            r'<article class="quality-card" data-quality-card="'
            + re.escape(str(no_go_card["id"]))
            + r'"[^>]*>(.*?)</article>',
            page,
            re.S,
        )
        self.assertIsNotNone(match)
        card_html = match.group(1)
        self.assertIn("UNQUANTIFIED", card_html)
        self.assertNotIn("1 word in", card_html)
        self.assertIn("not measured", card_html)
        self.assertNotIn("None", card_html)
        # `validate_public_site.validate_quality_guide_manifest` is a separate
        # mirror (scripts/validate_public_site.py) used to check an
        # already-built `site/quality-guides.json`; it now enforces the same
        # "Unquantified" tier and `config.residency` rules (see the
        # test_secondary_validator_* cases below) so a card of this shape
        # passes both validators identically.
        manifest_failures = validate_public_site.validate_quality_guide_manifest(loaded)
        self.assertEqual(manifest_failures, [])

    def test_shipped_flash_next_cards_state_a_measured_speed_benefit(self) -> None:
        """The two served-model cards must pair their stated quality COST with a
        measured benefit. A card that carries a numeric speedX must also carry a
        boundary naming the engine build it was measured on, and must not still
        list decode throughput as unmeasured -- that combination is a card
        contradicting itself."""
        manifest = json.loads(
            (REPOSITORY_ROOT / "site/quality-guides.json").read_text(encoding="utf-8")
        )
        cards = {card["id"]: card for card in manifest["cards"]}
        for card_id in (
            "qwen38-flash-next-mixed-4-8bit@m3ultra",
            "qwen38-flash-next-iq-3p3bpw@m3ultra",
        ):
            with self.subTest(card=card_id):
                benefit = cards[card_id]["legible"]["benefit"]
                self.assertIsInstance(benefit["speedX"], float)
                self.assertGreater(benefit["speedX"], 0)
                self.assertIn("fa76a4b5", benefit["speedXStatus"])
                self.assertIsNotNone(benefit["fit"])
                unmeasured = cards[card_id]["boundary"]["unmeasured"]
                self.assertNotIn("decode throughput (speedX)", unmeasured)

    # ------------------------------------------------------------------
    # Internal (unpublished) quality-card manifest: item 7's factored card
    # validation must accept it under the exact same rule set the public
    # site manifest uses. Driven by an inline synthetic fixture shaped like
    # a real internal expert-stream card (see
    # `expert_stream_card_manifest_document` above).
    # ------------------------------------------------------------------
    def test_internal_expert_stream_card_manifest_passes_validation(self):
        document = self.expert_stream_card_manifest_document()
        validated = build_public_site.validate_quality_card_document(
            document, "fixture quality-card manifest"
        )
        self.assertEqual(len(validated["cards"]), 1)
        card = validated["cards"][0]
        self.assertEqual(card["id"], "example-moe-q4-stream@m3ultra")
        self.assertEqual(card["config"]["residency"], "expert-stream")
        self.assertEqual(card["legible"]["tier"], "Unquantified")
        self.assertIsNone(card["legible"]["nextWordDrift"])
        self.assertEqual(card["verdict"], "NO_GO")

    # ------------------------------------------------------------------
    # validate_public_site.validate_quality_guide_manifest is an independent
    # mirror of build_public_site's schema (it accumulates failures instead
    # of raising, to check an already-built quality/index.json). It must
    # enforce the same config.residency / "Unquantified" tier rules.
    # ------------------------------------------------------------------
    def test_secondary_validator_accepts_internal_expert_stream_card(self) -> None:
        document = self.expert_stream_card_manifest_document()
        failures = validate_public_site.validate_quality_guide_manifest(document)
        self.assertEqual(failures, [])

    def test_secondary_validator_rejects_unknown_residency(self) -> None:
        document = self.expert_stream_card_manifest_document()
        document["cards"][0]["config"]["residency"] = "bogus"
        failures = validate_public_site.validate_quality_guide_manifest(document)
        self.assertTrue(
            any("config.residency has unknown value" in f for f in failures), failures
        )

    def test_secondary_validator_rejects_unquantified_tier_on_pass_card(self) -> None:
        manifest = self.quality_guide_manifest()
        pass_card = manifest["cards"][2]
        self.assertEqual(pass_card["verdict"], "PASS")
        pass_card["legible"]["tier"] = "Unquantified"
        pass_card["legible"]["nextWordDrift"] = None
        failures = validate_public_site.validate_quality_guide_manifest(manifest)
        self.assertTrue(
            any("tier Unquantified but verdict is not NO_GO" in f for f in failures), failures
        )

    def test_secondary_validator_rejects_null_drift_no_go_with_noticeable_tier(self) -> None:
        manifest = self.quality_guide_manifest()
        no_go_card = manifest["cards"][0]
        self.assertEqual(no_go_card["verdict"], "NO_GO")
        self.assertEqual(no_go_card["legible"]["tier"], "Noticeable")
        no_go_card["legible"]["nextWordDrift"] = None
        failures = validate_public_site.validate_quality_guide_manifest(manifest)
        self.assertTrue(
            any(
                "null nextWordDrift on verdict 'NO_GO' but tier is not Unquantified" in f
                for f in failures
            ),
            failures,
        )

    def test_secondary_validator_rejects_unquantified_tier_with_non_null_drift(self) -> None:
        manifest = self.quality_guide_manifest()
        no_go_card = manifest["cards"][0]
        self.assertIsNotNone(no_go_card["legible"]["nextWordDrift"])
        no_go_card["legible"]["tier"] = "Unquantified"
        failures = validate_public_site.validate_quality_guide_manifest(manifest)
        self.assertTrue(
            any("tier Unquantified but nextWordDrift is not null" in f for f in failures),
            failures,
        )

    def test_secondary_validator_accepts_real_quality_guide_manifest(self) -> None:
        loaded = build_public_site.load_quality_guides(REPOSITORY_ROOT)
        self.assertIsNotNone(loaded)
        failures = validate_public_site.validate_quality_guide_manifest(loaded)
        self.assertEqual(failures, [])

    # ------------------------------------------------------------------
    # provenance.engineBuild (docs/quality-card-schema-v1.md "Engine
    # build"): OPTIONAL {"commit": <lowercase 40-hex>}, no other keys.
    # Both validators (build_public_site.validate_quality_card_document and
    # validate_public_site.validate_quality_guide_manifest) must agree.
    # ------------------------------------------------------------------
    ENGINE_BUILD_VALID_COMMIT = "fa76a4b50b3f54af7e9cd927279f5ba2870f02c6"

    def test_provenance_engine_build_valid_is_accepted_by_both_validators(self) -> None:
        manifest = self.quality_guide_manifest()
        manifest["cards"][0]["provenance"]["engineBuild"] = {
            "commit": self.ENGINE_BUILD_VALID_COMMIT
        }
        validated = build_public_site.validate_quality_card_document(manifest, "test manifest")
        self.assertEqual(
            validated["cards"][0]["provenance"]["engineBuild"],
            {"commit": self.ENGINE_BUILD_VALID_COMMIT},
        )
        failures = validate_public_site.validate_quality_guide_manifest(manifest)
        self.assertEqual(failures, [])

    def test_provenance_engine_build_invalid_variants_rejected_by_both_validators(self) -> None:
        variants = {
            "non-object": "not-an-object",
            "extra-key": {"commit": self.ENGINE_BUILD_VALID_COMMIT, "extra": "x"},
            "uppercase-commit": {"commit": self.ENGINE_BUILD_VALID_COMMIT.upper()},
            "short-commit": {"commit": self.ENGINE_BUILD_VALID_COMMIT[:10]},
            "non-hex-commit": {"commit": "z" * 40},
            "missing-commit": {},
        }
        for label, value in variants.items():
            with self.subTest(label=label):
                manifest = self.quality_guide_manifest()
                manifest["cards"][0]["provenance"]["engineBuild"] = value
                with self.assertRaises(SystemExit):
                    build_public_site.validate_quality_card_document(manifest, "test manifest")
                failures = validate_public_site.validate_quality_guide_manifest(manifest)
                self.assertTrue(failures, f"{label}: expected at least one failure")

    def test_duplicate_identity_residency_engine_build_is_rejected_by_both_validators(
        self,
    ) -> None:
        manifest = self.quality_guide_manifest()
        duplicate = json.loads(json.dumps(manifest["cards"][0]))
        duplicate["id"] = "qwen38-27b-optiq-4bit-dup@m3ultra"
        manifest["cards"].append(duplicate)
        with self.assertRaises(SystemExit):
            build_public_site.validate_quality_card_document(manifest, "test manifest")
        failures = validate_public_site.validate_quality_guide_manifest(manifest)
        self.assertTrue(
            any("duplicates another card" in f for f in failures), failures
        )

    # ------------------------------------------------------------------
    # config.hardwareClass joins the uniqueness key as a fourth component
    # (docs/quality-card-schema-v1.md "Uniqueness"): an Ultra card and an M5
    # card for the same identity/residency/engineBuild are two measurements
    # of two different facts, not duplicates. It must NEVER become a filter
    # (docs/task-inbox/2026-09-22-PREDECLARATION-hardwareclass-joins-identity-never-filters.md).
    # ------------------------------------------------------------------

    def test_hardware_class_variant_of_a_duplicate_card_is_accepted_by_both_validators(
        self,
    ) -> None:
        manifest = self.quality_guide_manifest()
        variant = json.loads(json.dumps(manifest["cards"][0]))
        variant["id"] = "qwen38-27b-optiq-4bit@m5"
        variant["config"]["hardwareClass"] = "apple-m5"
        manifest["cards"].append(variant)
        validated = build_public_site.validate_quality_card_document(manifest, "test manifest")
        self.assertEqual(len(validated["cards"]), 5)
        failures = validate_public_site.validate_quality_guide_manifest(manifest)
        self.assertEqual(failures, [])

    def test_duplicate_including_hardware_class_is_still_rejected_by_both_validators(
        self,
    ) -> None:
        # Control: the card `id` already embeds an `@<hardwareClass>` suffix,
        # so an implementation that keys uniqueness on `id` (instead of the
        # (identity, residency, hardwareClass, engineBuild) tuple) would pass
        # the accept-test above while having silently deleted the whole
        # uniqueness rule. Give the duplicate a DISTINCT id so this test is
        # about the key, not the id.
        manifest = self.quality_guide_manifest()
        duplicate = json.loads(json.dumps(manifest["cards"][0]))
        duplicate["id"] = "qwen38-27b-optiq-4bit-hw-dup@m3ultra"
        manifest["cards"].append(duplicate)
        with self.assertRaises(SystemExit):
            build_public_site.validate_quality_card_document(manifest, "test manifest")
        failures = validate_public_site.validate_quality_guide_manifest(manifest)
        self.assertTrue(
            any("duplicates another card" in f for f in failures), failures
        )

    def test_hardware_class_duplicate_check_agrees_between_both_validators(self) -> None:
        # Nothing else cross-checks the two implementations; a change to one
        # alone must not go unnoticed. Runs the SAME fixture pair through
        # both validators for the accept case and the refuse case.
        cases = {
            "different-hardware-class-accepts": "apple-m5",
            "same-hardware-class-refuses": "apple-m3-ultra",
        }
        for label, hardware_class in cases.items():
            with self.subTest(label=label):
                manifest = self.quality_guide_manifest()
                variant = json.loads(json.dumps(manifest["cards"][0]))
                variant["id"] = "qwen38-27b-optiq-4bit-crosscheck@variant"
                variant["config"]["hardwareClass"] = hardware_class
                manifest["cards"].append(variant)

                build_refused = False
                try:
                    build_public_site.validate_quality_card_document(manifest, "test manifest")
                except SystemExit:
                    build_refused = True

                failures = validate_public_site.validate_quality_guide_manifest(manifest)
                validate_refused = bool(failures)

                self.assertEqual(
                    build_refused,
                    validate_refused,
                    f"{label}: build_public_site refused={build_refused} "
                    f"validate_public_site refused={validate_refused} failures={failures}",
                )

    # ------------------------------------------------------------------
    # config.hardwareClass must be canonical: `^[a-z0-9]+(-[a-z0-9]+)*$`,
    # exactly the shape `fastmlx_launch.host_hardware_class()` produces. A
    # value outside this shape can never host-match by construction (see
    # that function's docstring), so it is a SILENT no-op at serve time
    # rather than an error -- which is why the shape pin is refused here,
    # not merely style
    # (docs/task-inbox/2026-09-22-PREDECLARATION-hardware-class-must-be-canonical.md).
    # ------------------------------------------------------------------

    HARDWARE_CLASS_CANONICAL_VALUES = ("apple-m3-ultra", "apple-m5")
    HARDWARE_CLASS_NONCANONICAL_VALUES = ("Apple M5", "apple_m5", "APPLE-M5", "apple--m5")

    def test_hardware_class_canonical_values_are_accepted_by_both_validators(self) -> None:
        for hardware_class in self.HARDWARE_CLASS_CANONICAL_VALUES:
            with self.subTest(hardware_class=hardware_class):
                manifest = self.quality_guide_manifest()
                manifest["cards"][0]["config"]["hardwareClass"] = hardware_class
                validated = build_public_site.validate_quality_card_document(
                    manifest, "test manifest"
                )
                self.assertEqual(
                    validated["cards"][0]["config"]["hardwareClass"], hardware_class
                )
                failures = validate_public_site.validate_quality_guide_manifest(manifest)
                self.assertEqual(failures, [])

    def test_hardware_class_noncanonical_values_are_refused_naming_field_and_value(
        self,
    ) -> None:
        for hardware_class in self.HARDWARE_CLASS_NONCANONICAL_VALUES:
            with self.subTest(hardware_class=hardware_class):
                manifest = self.quality_guide_manifest()
                manifest["cards"][0]["config"]["hardwareClass"] = hardware_class

                with self.assertRaises(SystemExit) as build_ctx:
                    build_public_site.validate_quality_card_document(manifest, "test manifest")
                build_message = str(build_ctx.exception)
                self.assertIn("hardwareClass", build_message)
                self.assertIn(repr(hardware_class), build_message)

                failures = validate_public_site.validate_quality_guide_manifest(manifest)
                matching = [
                    f for f in failures if "hardwareClass" in f and repr(hardware_class) in f
                ]
                self.assertTrue(matching, failures)

    # Three hand-maintained copies of the hardwareClass shape/sentinel rule
    # exist by design (the projection split): build_public_site.py,
    # validate_public_site.py, and emit_quality_card.py each own a separate
    # constant rather than importing a shared one. Nothing else cross-checks
    # the two PROJECTED copies against drift, so pin their regex source and
    # flags, and their sentinel constants, directly against each other --
    # an added `re.IGNORECASE` (or similar) on only one copy would otherwise
    # drift silently past every other test in this file.
    def test_hardware_class_shape_and_sentinel_constants_agree_between_both_validators(
        self,
    ) -> None:
        self.assertEqual(
            build_public_site.QUALITY_CARD_HARDWARE_CLASS.pattern,
            validate_public_site.QUALITY_CARD_HARDWARE_CLASS.pattern,
        )
        self.assertEqual(
            build_public_site.QUALITY_CARD_HARDWARE_CLASS.flags,
            validate_public_site.QUALITY_CARD_HARDWARE_CLASS.flags,
        )
        self.assertEqual(
            build_public_site.QUALITY_CARD_HARDWARE_CLASS_SENTINEL,
            validate_public_site.QUALITY_CARD_HARDWARE_CLASS_SENTINEL,
        )

    # ------------------------------------------------------------------
    # config.hardwareClass must not carry the chip-probe-failed sentinel
    # "unknown": `ProvenanceCLI.chipBrand()`
    # (spike/Sources/fastmlx-harness/Provenance+CLI.swift:102,105) and
    # `fastmlx_bench._chip_identity` (three sites) return that literal
    # string when chip identification fails, and it feeds
    # `emit_quality_card`'s `context["hardware"]["chip"]` unrefused. It is
    # lowercase alphanumeric, so it PASSES the canonical shape check above
    # -- yet `fastmlx_launch.host_hardware_class()` (the detector) fails
    # closed to `None` and can NEVER return "unknown", so a card carrying
    # it can never host-match: a silent no-op, exactly what this gate
    # exists to refuse. This is a SEPARATE check from the shape check (the
    # shape is fine; the meaning is "the chip probe failed"), with its OWN
    # distinct message.
    # ------------------------------------------------------------------

    HARDWARE_CLASS_SENTINEL_VALUES = ("unknown", "UNKNOWN", "Unknown")

    def test_hardware_class_sentinel_is_refused_by_both_validators_with_sentinel_message(
        self,
    ) -> None:
        for hardware_class in self.HARDWARE_CLASS_SENTINEL_VALUES:
            with self.subTest(hardware_class=hardware_class):
                manifest = self.quality_guide_manifest()
                manifest["cards"][0]["config"]["hardwareClass"] = hardware_class

                with self.assertRaises(SystemExit) as build_ctx:
                    build_public_site.validate_quality_card_document(manifest, "test manifest")
                build_message = str(build_ctx.exception)
                self.assertIn("hardwareClass", build_message)
                # Distinguishing text: the sentinel message, not the generic
                # shape message -- a test that only checked "some failure
                # occurred" would pass on the wrong branch.
                self.assertIn("chip-probe-failed sentinel value", build_message)
                self.assertNotIn("is not a canonical", build_message)

                failures = validate_public_site.validate_quality_guide_manifest(manifest)
                sentinel_failures = [
                    f
                    for f in failures
                    if "hardwareClass" in f and "chip-probe-failed sentinel value" in f
                ]
                self.assertTrue(sentinel_failures, failures)
                self.assertFalse(
                    any(
                        "hardwareClass" in f and "is not a canonical" in f
                        for f in failures
                    ),
                    f"sentinel value must be refused with the sentinel message, not the "
                    f"generic shape message: {failures}",
                )

    def test_hardware_class_casing_variant_is_refused_for_shape_never_admitted_as_identity(
        self,
    ) -> None:
        # Two otherwise-identical cards sharing the SAME non-canonical
        # hardwareClass casing must be refused for the SHAPE, so a casing
        # variant can never be silently admitted as a second (identity,
        # residency, hardwareClass, engineBuild) identity -- that is
        # precisely F1 in the predeclaration: a casing variant
        # manufacturing a tie the duplicate gate exists to prevent.
        #
        # Measured collect-vs-raise asymmetry (do not mistake either half
        # for a bug): `build_public_site.fail()` raises on the FIRST
        # non-canonical card it sees, so for that validator the shape
        # refusal provably happens before the duplicate check ever runs --
        # it never reaches the second card, let alone the duplicate check.
        # `validate_public_site` is collect-style and does NOT
        # short-circuit: it still constructs the identity key from the raw
        # (non-canonical) value for BOTH cards, so with two IDENTICAL
        # non-canonical values the raw strings tie and it ALSO reports
        # "duplicates another card" alongside the two shape failures
        # (measured: 3 failures total). That is not a second bug -- the
        # shape failures already refuse the manifest either way -- but a
        # test must not claim the duplicate branch never fires here.
        manifest = self.quality_guide_manifest()
        manifest["cards"][0]["config"]["hardwareClass"] = "Apple M3 Ultra"
        variant = json.loads(json.dumps(manifest["cards"][0]))
        variant["id"] = "qwen38-27b-optiq-4bit-casing@variant"
        manifest["cards"].append(variant)

        with self.assertRaises(SystemExit) as build_ctx:
            build_public_site.validate_quality_card_document(manifest, "test manifest")
        build_message = str(build_ctx.exception)
        self.assertIn("hardwareClass", build_message)
        self.assertNotIn("duplicates another card", build_message)

        failures = validate_public_site.validate_quality_guide_manifest(manifest)
        self.assertTrue(failures, "casing-variant manifest must be refused")
        shape_failures = [
            f for f in failures if "hardwareClass" in f and "is not a canonical" in f
        ]
        # Both card 0 and the appended variant carry the non-canonical
        # value, so the validate-side collect list carries a shape failure
        # for each -- it never short-circuits after the first.
        variant_index = len(manifest["cards"]) - 1
        self.assertTrue(
            any("quality card entry 0 " in f for f in shape_failures), shape_failures
        )
        self.assertTrue(
            any(f"quality card entry {variant_index} " in f for f in shape_failures),
            shape_failures,
        )

    # The Flash Next cards were measured on a third-party serving engine
    # (fa76a4b5); the consumer-class 0.6B card was measured on fast-mlx's
    # own fastmlx-serve release build. provenance.engineBuild is OPTIONAL
    # on ANY card (docs/quality-card-schema-v1.md:362) -- what this test
    # pins is that no card acquires one by ACCIDENT, which is the live
    # hazard: emit_quality_card.py --engine-commit stamps every card that
    # run emits (scripts/emit_quality_card.py:946).
    EXPECTED_ENGINE_BUILD_BY_CARD_ID = {
        "qwen38-flash-next-mixed-4-8bit@m3ultra": ENGINE_BUILD_VALID_COMMIT,
        "qwen38-flash-next-iq-3p3bpw@m3ultra": ENGINE_BUILD_VALID_COMMIT,
        "qwen3-0p6b-4bit@m5": "21af3abd339a0752319ba6b4abe3b3880b16da3b",
        # Added 2026-09-23 (cycle 144) for the second card cycle 142 shipped. Its build sha
        # differs from the @m5 card's because the two rows were measured on different hosts --
        # that is the point of the pin, so they are listed separately rather than shared.
        "qwen3-0p6b-4bit@m3ultra": "a1f002a00d452a0abf662f7174cae0ae25873b85",
    }

    def test_real_manifest_engine_build_shas_are_exactly_as_expected(self) -> None:
        loaded = build_public_site.load_quality_guides(REPOSITORY_ROOT)
        self.assertIsNotNone(loaded)
        # (1) unchanged: every Flash Next card carries the build it was measured on.
        flash_next_cards = [
            card for card in loaded["cards"] if card["model"]["family"] == "Qwen3.8-Flash-Next"
        ]
        self.assertTrue(flash_next_cards, "expected at least one Qwen3.8-Flash-Next card")
        for card in flash_next_cards:
            self.assertEqual(
                card["provenance"].get("engineBuild"),
                {"commit": self.ENGINE_BUILD_VALID_COMMIT},
                card["id"],
            )
        # (2) every recorded build, on every card, is the expected one -- a new
        # card, a dropped field, or a wrongly-stamped sha all fail here.
        self.assertEqual(
            {
                card["id"]: card["provenance"]["engineBuild"]["commit"]
                for card in loaded["cards"]
                if "engineBuild" in card["provenance"]
            },
            self.EXPECTED_ENGINE_BUILD_BY_CARD_ID,
        )
        # (3) "absent means unrecorded" stays exercised by a real card.
        self.assertTrue(
            [card for card in loaded["cards"] if "engineBuild" not in card["provenance"]],
            "expected at least one card with no recorded engine build",
        )


class ArgparseUsageErrorExitCodeTests(unittest.TestCase):
    """A bad CLI invocation (unknown flag, missing required positional) must
    exit 64 (EX_USAGE), never argparse's own default of 2 -- this script's
    main() already returns 2 for a real RED "not a directory" refusal (see
    ExitCodeContractTests below), so a mistyped flag must never be
    indistinguishable from that refusal.

    validate_public_site.py has exactly one CLI argument (the required
    positional `site`, no `choices=` on anything), so there is no "bad
    choice" usage-error kind to exercise here -- unlike scripts with
    `--residency`-style enum flags (see fastmlx_gguf_fit.py's
    ArgparseUsageErrorExitCodeTests), this validator has nothing for
    argparse to reject as an invalid choice.
    """

    def test_unknown_flag_exits_64(self) -> None:
        # argparse reports the usage error on stderr; captured so it does not
        # read as a real error in the shared suite log (same reason as
        # ExitCodeContractTests._silenced below).
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as context:
                validate_public_site.parse_arguments(["--bogus-flag", "x"])
        self.assertEqual(context.exception.code, 64)

    def test_missing_required_site_argument_exits_64(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as context:
                validate_public_site.parse_arguments([])
        self.assertEqual(context.exception.code, 64)


class TopLevelArgumentHelpTextTests(unittest.TestCase):
    """validate_public_site.py has no subparsers -- pin the top-level `site`
    argument's help= text instead, so it cannot silently disappear."""

    def test_site_argument_help_text_is_pinned(self) -> None:
        help_text = validate_public_site.build_arg_parser().format_help()
        self.assertIn("generated site directory", help_text)


class ExitCodeContractTests(unittest.TestCase):
    """Anti-vacuity: proves the usage-error exit-64 change above did not
    flatten main()'s own pre-existing exit codes together. A test that only
    asserted 64 would still pass even if every code became 64."""

    @staticmethod
    @contextlib.contextmanager
    def _silenced():
        # main() reports each failure on stdout. The failure-path test below
        # provokes ~86 of them, every one prefixed "FAIL:", straight into the
        # shared suite log -- where they read as real test failures and blunt
        # any grep that scans that log for a verdict. The exit code is what
        # these tests assert; the reporting text is not.
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            yield buffer

    def test_success_path_exits_0(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)
            with self._silenced():
                exit_code = validate_public_site.main([str(output)])
            self.assertEqual(exit_code, 0)

    def test_validation_failure_path_exits_1(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            output.mkdir()
            build_public_site.build_site(REPOSITORY_ROOT, output)
            (output / "research/index.html").unlink()
            with self._silenced():
                exit_code = validate_public_site.main([str(output)])
            self.assertEqual(exit_code, 1)

    def test_not_a_directory_path_exits_2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "does-not-exist"
            with self._silenced():
                exit_code = validate_public_site.main([str(missing)])
            self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
