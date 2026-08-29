from __future__ import annotations

import json
import html
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import build_public_site  # noqa: E402
import validate_public_site  # noqa: E402


class PublicCapabilityDetailTests(unittest.TestCase):
    def build_site(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        directory = tempfile.TemporaryDirectory()
        output = Path(directory.name) / "site"
        output.mkdir()
        build_public_site.build_site(REPOSITORY_ROOT, output)
        return directory, output

    def reviewed_capabilities(self) -> list[dict[str, object]]:
        return [dict(record) for record in validate_public_site.reviewed_capability_records()]

    def test_build_generates_one_detail_page_per_reviewed_capability(self) -> None:
        holder, output = self.build_site()
        with holder:
            expected = {
                f"capabilities/{capability['id']}/index.html"
                for capability in self.reviewed_capabilities()
            }
            actual = {
                path.relative_to(output).as_posix()
                for path in (output / "capabilities").glob("*/index.html")
            }
            self.assertEqual(actual, expected)

            capability_index = json.loads(
                (output / "capabilities/index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                capability_index["capabilities"],
                self.reviewed_capabilities(),
            )
            self.assertEqual(validate_public_site.validate(output), [])

    def test_detail_pages_bind_exact_reviewed_content_and_discovery(self) -> None:
        holder, output = self.build_site()
        with holder:
            sitemap = (output / "sitemap.xml").read_text(encoding="utf-8")
            llms = (output / "llms.txt").read_text(encoding="utf-8")
            capabilities_page = (output / "capabilities/index.html").read_text(
                encoding="utf-8"
            )
            status_page = (output / "status/index.html").read_text(encoding="utf-8")

            for capability in self.reviewed_capabilities():
                identifier = str(capability["id"])
                public_path = f"capabilities/{identifier}/"
                detail_path = output / public_path / "index.html"
                detail = detail_path.read_text(encoding="utf-8")
                state = str(capability["status"])
                title = f"{capability['name']} — fast-mlx capability"
                description = (
                    "Reviewed fast-mlx capability state and evidence for "
                    f"{capability['name']}."
                )

                self.assertIn(
                    f'data-capability-detail data-capability-id="{identifier}" '
                    f'data-capability-state="{state}"',
                    detail,
                )
                self.assertIn(f"<h1>{capability['name']}</h1>", detail)
                self.assertIn(html.escape(str(capability["summary"])), detail)
                self.assertIn(f"<strong>Scope:</strong> {capability['scope']}", detail)
                self.assertIn(
                    validate_public_site.CAPABILITY_STATUS_DESCRIPTIONS[state],
                    detail,
                )
                self.assertIn(
                    "This page creates no broader support, measurement, runtime, model, "
                    "acquisition, publication, admission, launchability, or containment "
                    "authority.",
                    detail,
                )
                self.assertNotIn("<script", detail)
                self.assertNotIn("data-capability-detail hidden", detail)
                self.assertIn(
                    f'<link rel="canonical" href="https://bitworks-io.github.io/fast-mlx/{public_path}">',
                    detail,
                )
                self.assertIn(f'<meta property="og:title" content="{title}">', detail)
                self.assertIn(
                    f'<meta property="og:description" content="{description}">',
                    detail,
                )
                self.assertIn(
                    f"https://bitworks-io.github.io/fast-mlx/{public_path}",
                    sitemap,
                )
                self.assertIn(f"- /{public_path}: {capability['name']}", llms)
                self.assertIn(
                    f'href="{identifier}/">Open capability details →</a>',
                    capabilities_page,
                )
                self.assertIn(
                    f'href="../capabilities/{identifier}/">Open capability details →</a>',
                    status_page,
                )

                for evidence in capability["evidence"]:
                    self.assertIsInstance(evidence, dict)
                    path = str(evidence["path"])
                    reviewed_at = str(evidence["reviewedAt"])
                    self.assertIn(f'data-capability-evidence="{path}"', detail)
                    self.assertIn(f'data-reviewed-at="{reviewed_at}"', detail)
                    self.assertIn(html.escape(str(evidence["title"])), detail)
                    self.assertIn(f"Path: {path}", detail)
                    self.assertIn(f"Reviewed: {reviewed_at}", detail)

    def test_validator_rejects_detail_contract_drift(self) -> None:
        holder, output = self.build_site()
        with holder:
            identifier = "openai-http-sse-serving"
            path = output / "capabilities" / identifier / "index.html"
            original = path.read_text(encoding="utf-8")
            cases = (
                (
                    "wrong id",
                    f'data-capability-id="{identifier}"',
                    'data-capability-id="unreviewed-serving"',
                    f"capability detail '{identifier}' has the wrong id",
                ),
                (
                    "wrong state",
                    'data-capability-state="promoted-scoped"',
                    'data-capability-state="implemented"',
                    f"capability detail '{identifier}' has the wrong state",
                ),
                (
                    "hidden root",
                    '<section class="page-hero shell capability-detail" ',
                    '<section class="page-hero shell capability-detail" hidden ',
                    f"capability detail '{identifier}' is hidden",
                ),
                (
                    "active script",
                    "</body>",
                    '<script src="../../assets/research-explorer.js"></script></body>',
                    f"capability detail '{identifier}' must not load scripts",
                ),
                (
                    "duplicate attribute",
                    'data-capability-id="openai-http-sse-serving"',
                    'data-capability-id="openai-http-sse-serving" data-capability-id="openai-http-sse-serving"',
                    f"capability detail '{identifier}' has duplicate attributes",
                ),
                (
                    "nested detail",
                    "</h1>",
                    '</h1><section data-capability-detail data-capability-id="nested" data-capability-state="implemented"></section>',
                    f"capability detail '{identifier}' has the wrong heading structure",
                ),
                (
                    "evidence path",
                    'data-capability-evidence="research/the-proof-did-not-end-when-the-timer-did/"',
                    'data-capability-evidence="research/lossless-wasnt-byte-identical/"',
                    f"capability detail '{identifier}' has the wrong evidence",
                ),
                (
                    "review date",
                    'data-reviewed-at="2026-08-06"',
                    'data-reviewed-at="2026-08-06T00:00:00Z"',
                    f"capability detail '{identifier}' has the wrong evidence",
                ),
                (
                    "action link",
                    'href="../index.json"',
                    'href="../../capabilities/index.json"',
                    f"capability detail '{identifier}' has the wrong action links",
                ),
                (
                    "authority boundary",
                    "This page creates no broader support, measurement, runtime, model, acquisition, publication, admission, launchability, or containment authority.",
                    "This page creates support for every model and runtime.",
                    f"capability detail '{identifier}' has the wrong claim boundary",
                ),
            )
            for label, before, after, expected_failure in cases:
                with self.subTest(label=label):
                    self.assertIn(before, original)
                    path.write_text(original.replace(before, after, 1), encoding="utf-8")
                    self.assertIn(expected_failure, validate_public_site.validate(output))
                    path.write_text(original, encoding="utf-8")

    def test_validator_rejects_missing_extra_symlink_and_joint_tampering(self) -> None:
        holder, output = self.build_site()
        with holder:
            identifier = "openai-http-sse-serving"
            detail_path = output / "capabilities" / identifier / "index.html"
            original_detail = detail_path.read_text(encoding="utf-8")
            detail_path.unlink()
            self.assertIn(
                f"missing required file: capabilities/{identifier}/index.html",
                validate_public_site.validate(output),
            )
            detail_path.write_text(original_detail, encoding="utf-8")

            extra = output / "capabilities" / "unreviewed-capability"
            extra.mkdir()
            (extra / "index.html").write_text("<!doctype html><title>x</title>", encoding="utf-8")
            self.assertIn(
                "unexpected capability route outside reviewed set: unreviewed-capability",
                validate_public_site.validate(output),
            )
            shutil.rmtree(extra)

            symlink = output / "capabilities" / "symlink-capability"
            try:
                os.symlink(output / "capabilities" / identifier, symlink)
            except (OSError, NotImplementedError):
                pass
            else:
                self.assertIn(
                    "unexpected capability route outside reviewed set: symlink-capability",
                    validate_public_site.validate(output),
                )
                symlink.unlink()

            index_path = output / "capabilities/index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["capabilities"][0]["summary"] = "Tampered but jointly reflected."
            index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
            detail_path.write_text(
                original_detail.replace(
                    "A chat-completions transport with streaming, bounded admission, cancellation, model- and host-fit-aware context/completion limits, authenticated capability discovery, and evidence output.",
                    "Tampered but jointly reflected.",
                    1,
                ),
                encoding="utf-8",
            )
            failures = validate_public_site.validate(output)
            self.assertIn(
                "capabilities/index.json capabilities do not match reviewed capability records",
                failures,
            )
            self.assertIn(
                f"capability detail '{identifier}' has drifted reviewed text",
                failures,
            )

    def test_validator_requires_exact_card_detail_links(self) -> None:
        holder, output = self.build_site()
        with holder:
            capability_page = output / "capabilities/index.html"
            capability_text = capability_page.read_text(encoding="utf-8")
            capability_page.write_text(
                capability_text.replace(
                    'href="openai-http-sse-serving/">Open capability details →</a>',
                    'href="../methodology/">Open capability details →</a>',
                    1,
                ),
                encoding="utf-8",
            )
            self.assertIn(
                "capability card 'openai-http-sse-serving' has the wrong action links",
                validate_public_site.validate(output),
            )
            capability_page.write_text(capability_text, encoding="utf-8")

            status_page = output / "status/index.html"
            status_text = status_page.read_text(encoding="utf-8")
            status_page.write_text(
                status_text.replace(
                    'href="../capabilities/openai-http-sse-serving/">Open capability details →</a>',
                    'href="../methodology/">Open capability details →</a>',
                    1,
                ),
                encoding="utf-8",
            )
            self.assertIn(
                "status page links do not match reviewed contract",
                validate_public_site.validate(output),
            )

    def test_card_detail_actions_are_separate_from_published_evidence(self) -> None:
        holder, output = self.build_site()
        with holder:
            capabilities_page = (output / "capabilities/index.html").read_text(
                encoding="utf-8"
            )
            status_page = (output / "status/index.html").read_text(encoding="utf-8")

            for capability in self.reviewed_capabilities():
                identifier = str(capability["id"])
                self.assertIn(
                    "</div>\n"
                    f'<a class="text-link" href="{identifier}/">'
                    "Open capability details →</a>",
                    capabilities_page,
                )
                self.assertIn(
                    "</div>\n"
                    f'<a class="text-link" href="../capabilities/{identifier}/">'
                    "Open capability details →</a>",
                    status_page,
                )

    def test_validator_rejects_catalog_active_hidden_and_duplicate_content(self) -> None:
        holder, output = self.build_site()
        with holder:
            path = output / "capabilities/index.html"
            original = path.read_text(encoding="utf-8")
            cases = (
                (
                    "active script",
                    "</body>",
                    '<script src="../assets/research-explorer.js"></script></body>',
                ),
                (
                    "hidden content",
                    "</main>",
                    "<section hidden>Unreviewed roadmap promise</section></main>",
                ),
                (
                    "duplicate card attribute",
                    'data-capability-card="openai-http-sse-serving"',
                    'data-capability-card="openai-http-sse-serving" '
                    'data-capability-card="openai-http-sse-serving"',
                ),
            )
            for label, before, after in cases:
                with self.subTest(label=label):
                    self.assertIn(before, original)
                    path.write_text(original.replace(before, after, 1), encoding="utf-8")
                    self.assertIn(
                        "capabilities/index.html does not match the reviewed page seal",
                        validate_public_site.validate(output),
                    )
                    path.write_text(original, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
