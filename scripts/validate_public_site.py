#!/usr/bin/env python3
"""Validate generated fast-mlx Pages files and internal links."""

from __future__ import annotations

import argparse
import html.parser
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlsplit


PRIVATE_MARKERS: Tuple[str, ...] = (
    "/" + "Users/",
    "/" + "private/",
    "192" + ".168.",
    "llm" + "bench",
    "passwordless" + " sudo",
    "BEGIN OPENSSH" + " PRIVATE KEY",
    "BEGIN RSA" + " PRIVATE KEY",
)
CAPABILITY_STATUSES = {"implemented", "promoted-scoped", "experimental", "shelved"}
HIGHLIGHT_DECISION_LABELS = {
    "promoted-scoped": "Promoted · scoped",
    "shelved": "Shelved",
}
BENCHMARK_FILTER_NAMES = ("model", "hardware", "decision")


class LinkCollector(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag not in {"a", "link", "script", "img"}:
            return
        attribute = "href" if tag in {"a", "link"} else "src"
        for key, value in attrs:
            if key == attribute and value:
                self.links.append(value)


class BenchmarkCollector(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.cards: List[Dict[str, object]] = []
        self.options: Dict[str, List[Dict[str, Optional[str]]]] = {
            name: [] for name in BENCHMARK_FILTER_NAMES
        }
        self.has_controls = False
        self.has_count = False
        self.has_empty_state = False
        self.has_script = False
        self._current_card: Optional[Dict[str, object]] = None
        self._current_select: Optional[str] = None
        self._current_option: Optional[Dict[str, object]] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "article" and "benchmark-result" in classes:
            self._current_card = {
                "id": attributes.get("data-highlight-id"),
                "model": attributes.get("data-model"),
                "hardware": attributes.get("data-hardware"),
                "decision": attributes.get("data-decision"),
                "hidden": "hidden" in attributes,
                "datetime": None,
                "links": [],
                "text_parts": [],
            }
        elif self._current_card is not None:
            if tag == "time":
                self._current_card["datetime"] = attributes.get("datetime")
            elif tag == "a" and attributes.get("href"):
                links = self._current_card["links"]
                if isinstance(links, list):
                    links.append(attributes["href"])

        if tag == "select" and attributes.get("name") in self.options:
            self._current_select = attributes["name"]
        elif tag == "option" and self._current_select is not None:
            self._current_option = {
                "value": attributes.get("value"),
                "text_parts": [],
            }
        if tag == "form" and "data-benchmark-controls" in attributes:
            self.has_controls = True
        if (
            "data-benchmark-count" in attributes
            and attributes.get("aria-live") == "polite"
        ):
            self.has_count = True
        if (
            "data-benchmark-empty" in attributes
            and "hidden" in attributes
            and attributes.get("role") == "status"
        ):
            self.has_empty_state = True
        if (
            tag == "script"
            and attributes.get("src") == "../assets/benchmark-explorer.js"
        ):
            self.has_script = True

    def handle_data(self, data: str) -> None:
        if self._current_card is not None:
            text_parts = self._current_card["text_parts"]
            if isinstance(text_parts, list):
                text_parts.append(data)
        if self._current_option is not None:
            text_parts = self._current_option["text_parts"]
            if isinstance(text_parts, list):
                text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "article" and self._current_card is not None:
            text_parts = self._current_card.pop("text_parts")
            self._current_card["text"] = " ".join(
                "".join(text_parts).split()
            ) if isinstance(text_parts, list) else ""
            self.cards.append(self._current_card)
            self._current_card = None
        if tag == "option" and self._current_option is not None:
            text_parts = self._current_option.pop("text_parts")
            self._current_option["text"] = " ".join(
                "".join(text_parts).split()
            ) if isinstance(text_parts, list) else ""
            if self._current_select is not None:
                self.options[self._current_select].append(self._current_option)
            self._current_option = None
        elif tag == "select":
            self._current_select = None


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path, help="generated site directory")
    return parser.parse_args(argv)


def resolve_target(site: Path, page: Path, raw_link: str) -> Optional[Path]:
    parsed = urlsplit(raw_link)
    if parsed.scheme or parsed.netloc or raw_link.startswith(("mailto:", "#")):
        return None
    if parsed.path.startswith("/"):
        target = site / parsed.path.lstrip("/")
    else:
        target = page.parent / unquote(parsed.path)
    if not parsed.path or parsed.path.endswith("/"):
        target /= "index.html"
    return target.resolve()


def validate_evidence_path(site: Path, raw_path: object, label: str) -> List[str]:
    if not isinstance(raw_path, str) or not raw_path:
        return [f"{label} has an invalid evidence path: {raw_path!r}"]
    target = (site / raw_path / "index.html").resolve()
    try:
        target.relative_to(site)
    except ValueError:
        return [f"{label} evidence path escapes site root: {raw_path!r}"]
    if not target.is_file():
        return [f"{label} evidence page is missing: {raw_path!r}"]
    return []


def validate(site: Path) -> List[str]:
    failures: List[str] = []
    site = site.resolve()
    required = [
        "index.html",
        "process/index.html",
        "methodology/index.html",
        "capabilities/index.html",
        "capabilities/index.json",
        "benchmarks/index.html",
        "research/index.html",
        "research/index.json",
        "assets/site.css",
        "assets/benchmark-explorer.js",
        "assets/favicon.svg",
        "llms.txt",
        ".nojekyll",
    ]
    for relative in required:
        if not (site / relative).is_file():
            failures.append(f"missing required file: {relative}")

    expected_benchmark_cards: Dict[str, Dict[str, str]] = {}

    index_path = site / "research/index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"invalid research/index.json: {exc}")
        else:
            if index.get("schemaVersion") != 1:
                failures.append("research/index.json does not use schemaVersion 1")
            articles = index.get("articles", [])
            if not isinstance(articles, list) or not articles:
                failures.append("research/index.json has no articles")
            else:
                for article in articles:
                    path = article.get("path") if isinstance(article, dict) else None
                    if not isinstance(path, str) or not (site / path / "index.html").is_file():
                        failures.append(f"missing article page for index entry: {path!r}")

    capability_index_path = site / "capabilities/index.json"
    if capability_index_path.is_file():
        try:
            capability_index = json.loads(
                capability_index_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            failures.append(f"invalid capabilities/index.json: {exc}")
        else:
            if capability_index.get("schemaVersion") != 1:
                failures.append("capabilities/index.json does not use schemaVersion 1")
            if capability_index.get("project") != "fast-mlx":
                failures.append("capabilities/index.json has the wrong project")
            if capability_index.get("claimBoundary") != "fast-mlx-owned-results-only":
                failures.append("capabilities/index.json has the wrong claim boundary")

            status_definitions = capability_index.get("statusDefinitions")
            definition_ids = {
                item.get("id")
                for item in status_definitions
                if isinstance(item, dict)
            } if isinstance(status_definitions, list) else set()
            if definition_ids != CAPABILITY_STATUSES:
                failures.append("capabilities/index.json has incomplete status definitions")

            capabilities = capability_index.get("capabilities")
            if not isinstance(capabilities, list) or not capabilities:
                failures.append("capabilities/index.json has no capabilities")
            else:
                seen_capability_ids: set[str] = set()
                for position, capability in enumerate(capabilities):
                    label = f"capability index entry {position}"
                    if not isinstance(capability, dict):
                        failures.append(f"{label} is not an object")
                        continue
                    identifier = capability.get("id")
                    if not isinstance(identifier, str) or identifier in seen_capability_ids:
                        failures.append(f"{label} has an invalid or duplicate id")
                    else:
                        seen_capability_ids.add(identifier)
                    if capability.get("status") not in CAPABILITY_STATUSES:
                        failures.append(f"{label} has an unknown status")
                    evidence = capability.get("evidence")
                    if not isinstance(evidence, list) or not evidence:
                        failures.append(f"{label} has no evidence")
                    else:
                        for evidence_position, record in enumerate(evidence):
                            raw_path = record.get("path") if isinstance(record, dict) else None
                            failures.extend(
                                validate_evidence_path(
                                    site,
                                    raw_path,
                                    f"{label} evidence {evidence_position}",
                                )
                            )

            highlights = capability_index.get("performanceHighlights")
            if not isinstance(highlights, list) or not highlights:
                failures.append("capabilities/index.json has no performance highlights")
            else:
                seen_highlight_ids: set[str] = set()
                for position, highlight in enumerate(highlights):
                    label = f"performance highlight entry {position}"
                    if not isinstance(highlight, dict):
                        failures.append(f"{label} is not an object")
                        continue
                    identifier = highlight.get("id")
                    if not isinstance(identifier, str) or identifier in seen_highlight_ids:
                        failures.append(f"{label} has an invalid or duplicate id")
                    else:
                        seen_highlight_ids.add(identifier)
                    if highlight.get("decision") not in {"promoted-scoped", "shelved"}:
                        failures.append(f"{label} has an unknown decision")
                    for key in (
                        "metric",
                        "label",
                        "model",
                        "hardware",
                        "workload",
                        "date",
                        "caveat",
                    ):
                        if not isinstance(highlight.get(key), str) or not highlight[key].strip():
                            failures.append(f"{label} has an empty {key}")
                    evidence = highlight.get("evidence")
                    raw_path = evidence.get("path") if isinstance(evidence, dict) else None
                    failures.extend(validate_evidence_path(site, raw_path, label))
                    expected_fields = {
                        key: highlight.get(key)
                        for key in (
                            "metric",
                            "label",
                            "model",
                            "hardware",
                            "workload",
                            "date",
                            "decision",
                            "caveat",
                        )
                    }
                    if (
                        isinstance(identifier, str)
                        and all(isinstance(value, str) for value in expected_fields.values())
                        and expected_fields["decision"] in HIGHLIGHT_DECISION_LABELS
                        and isinstance(raw_path, str)
                    ):
                        expected_benchmark_cards[identifier] = {
                            **expected_fields,
                            "evidence": f"../{raw_path.rstrip('/')}/",
                        }

    benchmark_path = site / "benchmarks/index.html"
    if benchmark_path.is_file():
        collector = BenchmarkCollector()
        try:
            collector.feed(benchmark_path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"cannot parse benchmarks/index.html: {exc}")
        else:
            actual_ids = [card.get("id") for card in collector.cards]
            if (
                len(actual_ids) != len(set(actual_ids))
                or set(actual_ids) != set(expected_benchmark_cards)
            ):
                failures.append(
                    "benchmark explorer card set does not match performance highlights"
                )
            expected_order = [
                identifier
                for identifier, _entry in sorted(
                    expected_benchmark_cards.items(),
                    key=lambda item: item[1]["date"],
                    reverse=True,
                )
            ]
            if (
                len(actual_ids) == len(set(actual_ids))
                and set(actual_ids) == set(expected_benchmark_cards)
                and actual_ids != expected_order
            ):
                failures.append(
                    "benchmark explorer cards are not ordered by descending evidence date"
                )
            for card in collector.cards:
                identifier = card.get("id")
                expected = expected_benchmark_cards.get(
                    identifier if isinstance(identifier, str) else ""
                )
                if expected is None:
                    continue
                if card.get("hidden"):
                    failures.append(
                        f"benchmark explorer card {identifier!r} is hidden before enhancement"
                    )
                for key in ("model", "hardware", "decision"):
                    if card.get(key) != expected[key]:
                        failures.append(
                            f"benchmark explorer card {identifier!r} has the wrong {key}"
                        )
                card_text = card.get("text")
                normalized_text = card_text if isinstance(card_text, str) else ""
                for key in (
                    "metric",
                    "label",
                    "model",
                    "hardware",
                    "workload",
                    "date",
                    "caveat",
                ):
                    normalized_expected = " ".join(expected[key].split())
                    if normalized_expected not in normalized_text:
                        message = (
                            f"benchmark explorer card {identifier!r} has the wrong {key}"
                        )
                        if message not in failures:
                            failures.append(message)
                if card.get("datetime") != expected["date"]:
                    message = (
                        f"benchmark explorer card {identifier!r} has the wrong date"
                    )
                    if message not in failures:
                        failures.append(message)
                links = card.get("links")
                if not isinstance(links, list) or expected["evidence"] not in links:
                    failures.append(
                        f"benchmark explorer card {identifier!r} has the wrong evidence"
                    )
            expected_options = {
                "model": [("", "All models")]
                + [
                    (value, value)
                    for value in sorted(
                        {entry["model"] for entry in expected_benchmark_cards.values()},
                        key=str.casefold,
                    )
                ],
                "hardware": [("", "All hardware")]
                + [
                    (value, value)
                    for value in sorted(
                        {entry["hardware"] for entry in expected_benchmark_cards.values()},
                        key=str.casefold,
                    )
                ],
                "decision": [("", "All decisions")]
                + [
                    (value, HIGHLIGHT_DECISION_LABELS[value])
                    for value in sorted(
                        {entry["decision"] for entry in expected_benchmark_cards.values()},
                        key=lambda value: HIGHLIGHT_DECISION_LABELS[value].casefold(),
                    )
                ],
            }
            for name, expected in expected_options.items():
                actual = [
                    (option.get("value"), option.get("text"))
                    for option in collector.options[name]
                ]
                if actual != expected:
                    failures.append(
                        f"benchmark explorer {name} options do not match performance highlights"
                    )
            if not collector.has_controls:
                failures.append("benchmark explorer has no filter controls")
            if not collector.has_count:
                failures.append("benchmark explorer has no live result count")
            if not collector.has_empty_state:
                failures.append("benchmark explorer has no hidden status empty state")
            if not collector.has_script:
                failures.append("benchmark explorer does not load its reviewed script")

    for path in site.rglob("*"):
        if path.is_symlink():
            failures.append(f"symlink is forbidden in Pages output: {path.relative_to(site)}")
            continue
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"non-UTF-8 output file: {path.relative_to(site)}")
            continue
        for marker in PRIVATE_MARKERS:
            if marker.casefold() in text.casefold():
                failures.append(
                    f"private marker {marker!r} in {path.relative_to(site)}"
                )
        if path.suffix != ".html":
            continue
        collector = LinkCollector()
        try:
            collector.feed(text)
        except Exception as exc:  # HTMLParser should be tolerant; make any failure explicit.
            failures.append(f"cannot parse {path.relative_to(site)}: {exc}")
            continue
        for link in collector.links:
            target = resolve_target(site, path, link)
            if target is None:
                continue
            try:
                target.relative_to(site)
            except ValueError:
                failures.append(
                    f"link escapes site root: {path.relative_to(site)} -> {link}"
                )
                continue
            if not target.exists():
                failures.append(
                    f"broken internal link: {path.relative_to(site)} -> {link}"
                )
    return failures


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_arguments(argv)
    if not arguments.site.is_dir():
        print(f"public-site validation refused: not a directory: {arguments.site}", file=sys.stderr)
        return 2
    failures = validate(arguments.site)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(f"validated public site: {arguments.site.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
