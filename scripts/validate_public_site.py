#!/usr/bin/env python3
"""Validate generated fast-mlx Pages files and internal links."""

from __future__ import annotations

import argparse
import datetime as dt
import html.parser
import json
import posixpath
import re
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
RELEASE_CATEGORIES = {"foundation", "operations", "product"}
RELEASE_CATEGORY_LABELS = {
    "foundation": "Foundation",
    "operations": "Operations",
    "product": "Product",
}
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
RELEASE_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})"
)
PUBLIC_PATH = re.compile(
    r"(?:[a-z0-9][a-z0-9.-]*/)*(?:[a-z0-9][a-z0-9.-]*/|[a-z0-9][a-z0-9.-]*\.(?:html|json))"
)
RELEASE_INDEX_KEYS = {
    "schemaVersion",
    "project",
    "policy",
    "claimBoundary",
    "updatedAt",
    "currentBoundary",
    "releases",
}
RELEASE_BOUNDARY_KEYS = {"id", "label", "state", "summary", "evidence"}
RELEASE_LINK_KEYS = {"label", "path"}
RELEASE_ENTRY_KEYS = {
    "id",
    "title",
    "publishedAt",
    "category",
    "state",
    "summary",
    "scope",
    "publicCommit",
    "publicLinks",
    "sourceUrl",
}


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


class ReleaseCollector(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.boundary: Dict[str, object] = {
            "id": None,
            "state": None,
            "links": [],
            "text_parts": [],
        }
        self.cards: List[Dict[str, object]] = []
        self.has_json_link = False
        self._in_boundary = False
        self._current_card: Optional[Dict[str, object]] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "section" and "data-release-boundary" in attributes:
            self._in_boundary = True
            self.boundary["id"] = attributes.get("data-boundary-id")
            self.boundary["state"] = attributes.get("data-boundary-state")
        elif tag == "article" and "release-card" in classes:
            self._current_card = {
                "id": attributes.get("data-release-id"),
                "publicCommit": attributes.get("data-public-commit"),
                "datetime": None,
                "links": [],
                "text_parts": [],
                "hidden": "hidden" in attributes,
            }

        if tag == "a" and attributes.get("href") == "index.json":
            self.has_json_link = True
        if self._in_boundary and tag == "a" and attributes.get("href"):
            links = self.boundary["links"]
            if isinstance(links, list):
                links.append(attributes["href"])
        if self._current_card is not None:
            if tag == "time":
                self._current_card["datetime"] = attributes.get("datetime")
            elif tag == "a" and attributes.get("href"):
                links = self._current_card["links"]
                if isinstance(links, list):
                    links.append(attributes["href"])

    def handle_data(self, data: str) -> None:
        if self._in_boundary:
            text_parts = self.boundary["text_parts"]
            if isinstance(text_parts, list):
                text_parts.append(data)
        if self._current_card is not None:
            text_parts = self._current_card["text_parts"]
            if isinstance(text_parts, list):
                text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "section" and self._in_boundary:
            text_parts = self.boundary.pop("text_parts")
            self.boundary["text"] = " ".join(
                "".join(text_parts).split()
            ) if isinstance(text_parts, list) else ""
            self._in_boundary = False
        elif tag == "article" and self._current_card is not None:
            text_parts = self._current_card.pop("text_parts")
            self._current_card["text"] = " ".join(
                "".join(text_parts).split()
            ) if isinstance(text_parts, list) else ""
            self.cards.append(self._current_card)
            self._current_card = None


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


def relative_href(current_file: str, target_path: str) -> str:
    current_dir = posixpath.dirname(current_file)
    target = target_path.rstrip("/") or "."
    value = posixpath.relpath(target, current_dir or ".")
    if value == ".":
        value = "./"
    elif target_path.endswith("/"):
        value += "/"
    return value


def key_failures(value: object, expected: set[str], label: str) -> List[str]:
    if not isinstance(value, dict):
        return [f"{label} is not an object"]
    actual = set(value)
    if actual == expected:
        return []
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return [f"{label} keys differ from schema; missing={missing} extra={extra}"]


def require_str(
    entry: Dict[str, object], key: str, label: str, failures: List[str]
) -> Optional[str]:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        failures.append(f"{label} has an empty or non-string {key}")
        return None
    if value != value.strip():
        failures.append(f"{label} {key} contains surrounding whitespace")
    return value


def parse_release_timestamp(
    entry: Dict[str, object], key: str, label: str, failures: List[str]
) -> Optional[dt.datetime]:
    value = require_str(entry, key, label, failures)
    if value is None:
        return None
    if not RELEASE_TIMESTAMP.fullmatch(value):
        failures.append(f"{label} {key} is not an offset-aware release timestamp")
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        failures.append(f"{label} {key} is not an offset-aware release timestamp")
        return None
    if parsed.utcoffset() is None:
        failures.append(f"{label} {key} is not an offset-aware release timestamp")
        return None
    return parsed


def validate_public_path(site: Path, raw_path: object, label: str) -> List[str]:
    failures: List[str] = []
    if not isinstance(raw_path, str) or not raw_path.strip() or raw_path != raw_path.strip():
        return [f"{label} has an invalid public path: {raw_path!r}"]
    if not PUBLIC_PATH.fullmatch(raw_path):
        return [f"{label} has an invalid public path: {raw_path!r}"]
    if raw_path.endswith("/"):
        target = site / raw_path / "index.html"
    else:
        target = site / raw_path
    try:
        target.resolve().relative_to(site)
    except ValueError:
        failures.append(f"{label} public path escapes site root: {raw_path!r}")
    if not target.is_file():
        failures.append(f"{label} public path target is missing: {raw_path!r}")
    return failures


def validate_release_link(site: Path, value: object, label: str) -> List[str]:
    failures = key_failures(value, RELEASE_LINK_KEYS, label)
    if not isinstance(value, dict):
        return failures
    require_str(value, "label", label, failures)
    failures.extend(validate_public_path(site, value.get("path"), label))
    return failures


def load_release_index(site: Path) -> Tuple[Optional[Dict[str, object]], List[str]]:
    failures: List[str] = []
    release_index_path = site / "releases/index.json"
    try:
        release_index = json.loads(release_index_path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, [f"cannot read releases/index.json: {exc}"]
    except json.JSONDecodeError as exc:
        return None, [f"invalid releases/index.json: {exc}"]

    failures.extend(key_failures(release_index, RELEASE_INDEX_KEYS, "releases/index.json"))
    if not isinstance(release_index, dict):
        return None, failures
    if release_index.get("schemaVersion") != 1:
        failures.append("releases/index.json does not use schemaVersion 1")
    if release_index.get("project") != "fast-mlx":
        failures.append("releases/index.json has the wrong project")
    if release_index.get("policy") != "reviewed-public-releases-only":
        failures.append("releases/index.json has the wrong policy")
    if release_index.get("claimBoundary") != "fast-mlx-owned-results-only":
        failures.append("releases/index.json has the wrong claim boundary")

    updated_at = release_index.get("updatedAt")
    if not isinstance(updated_at, str):
        failures.append("releases/index.json has an invalid updatedAt")
    else:
        try:
            dt.date.fromisoformat(updated_at)
        except ValueError:
            failures.append("releases/index.json has an invalid updatedAt")

    boundary = release_index.get("currentBoundary")
    failures.extend(key_failures(boundary, RELEASE_BOUNDARY_KEYS, "current release boundary"))
    if isinstance(boundary, dict):
        if boundary.get("id") != "runtime-model-promotion":
            failures.append("current release boundary must remain runtime-model-promotion")
        if boundary.get("state") != "gated":
            failures.append("current release boundary state must remain gated")
        for key in ("label", "summary"):
            require_str(boundary, key, "current release boundary", failures)
        evidence = boundary.get("evidence")
        failures.extend(
            validate_release_link(site, evidence, "current release boundary evidence")
        )
        if isinstance(evidence, dict) and evidence.get("path") != "methodology/":
            failures.append("current release boundary evidence must remain methodology/")

    releases = release_index.get("releases")
    if not isinstance(releases, list) or not releases:
        failures.append("releases/index.json has no releases")
    else:
        seen_ids: set[str] = set()
        seen_commits: set[str] = set()
        previous_timestamp: Optional[dt.datetime] = None
        for position, release in enumerate(releases):
            label = f"release index entry {position}"
            failures.extend(key_failures(release, RELEASE_ENTRY_KEYS, label))
            if not isinstance(release, dict):
                continue
            identifier = require_str(release, "id", label, failures)
            if identifier is not None:
                if not SLUG.fullmatch(identifier) or identifier in seen_ids:
                    failures.append(f"{label} has an invalid or duplicate id")
                seen_ids.add(identifier)
            for key in ("title", "summary", "scope"):
                require_str(release, key, label, failures)
            category = require_str(release, "category", label, failures)
            if category is not None and category not in RELEASE_CATEGORIES:
                failures.append(f"{label} has unknown category {category!r}")
            state = require_str(release, "state", label, failures)
            if state is not None and state != "released":
                failures.append(f"{label} is not explicitly released")
            parsed_timestamp = parse_release_timestamp(
                release, "publishedAt", label, failures
            )
            if parsed_timestamp is not None:
                if (
                    previous_timestamp is not None
                    and parsed_timestamp >= previous_timestamp
                ):
                    failures.append("release entries are not strictly newest-first")
                previous_timestamp = parsed_timestamp
            commit = require_str(release, "publicCommit", label, failures)
            if commit is not None:
                if not COMMIT_SHA.fullmatch(commit) or commit in seen_commits:
                    failures.append(f"{label} has an invalid or duplicate publicCommit")
                seen_commits.add(commit)
                expected_source_url = (
                    "https://github.com/bitworks-io/fast-mlx/commit/" + commit
                )
                if release.get("sourceUrl") != expected_source_url:
                    failures.append(f"{label} sourceUrl does not match publicCommit")
            links = release.get("publicLinks")
            if not isinstance(links, list):
                failures.append(f"{label} publicLinks is not a list")
            else:
                seen_paths: set[str] = set()
                for link_position, link in enumerate(links):
                    link_label = f"{label} public link {link_position}"
                    failures.extend(validate_release_link(site, link, link_label))
                    if isinstance(link, dict) and isinstance(link.get("path"), str):
                        if link["path"] in seen_paths:
                            failures.append(f"{label} has a duplicate public link path")
                        seen_paths.add(link["path"])
    return release_index, failures


def validate_release_page(site: Path, release_index: Dict[str, object]) -> List[str]:
    failures: List[str] = []
    release_path = site / "releases/index.html"
    collector = ReleaseCollector()
    try:
        collector.feed(release_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"cannot parse releases/index.html: {exc}"]

    if not collector.has_json_link:
        failures.append("releases/index.html does not link to releases/index.json")

    boundary = release_index.get("currentBoundary")
    if isinstance(boundary, dict):
        if collector.boundary.get("id") != boundary.get("id"):
            failures.append("release page boundary id does not match releases/index.json")
        if collector.boundary.get("state") != boundary.get("state"):
            failures.append("release page boundary state does not match releases/index.json")
        boundary_text = collector.boundary.get("text")
        normalized_boundary_text = (
            boundary_text if isinstance(boundary_text, str) else ""
        )
        for key in ("label", "summary"):
            value = boundary.get(key)
            if isinstance(value, str) and " ".join(value.split()) not in normalized_boundary_text:
                failures.append(f"release page boundary has the wrong {key}")
        evidence = boundary.get("evidence")
        if isinstance(evidence, dict):
            expected_href = relative_href("releases/index.html", str(evidence.get("path", "")))
            links = collector.boundary.get("links")
            if not isinstance(links, list) or expected_href not in links:
                failures.append("release page boundary evidence link does not match JSON")

    releases = release_index.get("releases")
    expected_releases = releases if isinstance(releases, list) else []
    expected_by_id = {
        release["id"]: release
        for release in expected_releases
        if isinstance(release, dict) and isinstance(release.get("id"), str)
    }
    actual_ids = [card.get("id") for card in collector.cards]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_by_id):
        failures.append("release page card set does not match release ledger")
    expected_order = [release.get("id") for release in expected_releases]
    if (
        len(actual_ids) == len(set(actual_ids))
        and set(actual_ids) == set(expected_by_id)
        and actual_ids != expected_order
    ):
        failures.append("release page cards are not ordered like releases/index.json")
    seen_card_commits: set[object] = set()
    for card in collector.cards:
        identifier = card.get("id")
        release = expected_by_id.get(identifier if isinstance(identifier, str) else "")
        if release is None:
            continue
        commit = release.get("publicCommit")
        if card.get("publicCommit") != commit:
            message = "release page card set does not match release ledger"
            if message not in failures:
                failures.append(message)
            failures.append(f"release page card {identifier!r} has the wrong commit")
        if card.get("publicCommit") in seen_card_commits:
            failures.append(f"release page card {identifier!r} duplicates a commit")
        seen_card_commits.add(card.get("publicCommit"))
        if card.get("hidden"):
            failures.append(f"release page card {identifier!r} is hidden")
        if card.get("datetime") != release.get("publishedAt"):
            failures.append(f"release page card {identifier!r} has the wrong timestamp")
        text = card.get("text")
        normalized_text = text if isinstance(text, str) else ""
        expected_text_values = [
            release.get("state", "").upper(),
            str(release.get("publishedAt", ""))[:10],
            RELEASE_CATEGORY_LABELS.get(str(release.get("category", "")), ""),
            release.get("title"),
            release.get("summary"),
            release.get("scope"),
        ]
        for value in expected_text_values:
            if isinstance(value, str) and value:
                normalized_expected = " ".join(value.split())
                if normalized_expected not in normalized_text:
                    failures.append(
                        f"release page card {identifier!r} does not bind text {normalized_expected!r}"
                    )
        links = card.get("links")
        actual_links = links if isinstance(links, list) else []
        source_url = release.get("sourceUrl")
        if not isinstance(source_url, str) or source_url not in actual_links:
            failures.append(f"release page card {identifier!r} has the wrong source link")
        for raw_link in release.get("publicLinks", []):
            if not isinstance(raw_link, dict):
                continue
            path = raw_link.get("path")
            if isinstance(path, str):
                expected_href = relative_href("releases/index.html", path)
                if expected_href not in actual_links:
                    failures.append(
                        f"release page card {identifier!r} is missing public link {path!r}"
                    )
    return failures


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
        "releases/index.html",
        "releases/index.json",
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

    release_index: Optional[Dict[str, object]] = None
    if (site / "releases/index.json").is_file():
        release_index, release_failures = load_release_index(site)
        failures.extend(release_failures)
    if release_index is not None and (site / "releases/index.html").is_file():
        failures.extend(validate_release_page(site, release_index))

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
