from __future__ import annotations

import json
import subprocess
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
        commits = [entry["publicCommit"] for entry in catalog["releases"]]
        self.assertEqual(
            commits,
            [
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

    def test_validator_requires_release_outputs(self) -> None:
        for missing in ("releases/index.html", "releases/index.json"):
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
