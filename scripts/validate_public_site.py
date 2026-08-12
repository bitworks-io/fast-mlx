#!/usr/bin/env python3
"""Validate generated fast-mlx Pages files and internal links."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html.parser
import json
import posixpath
import re
import sys
import xml.etree.ElementTree as ET
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
CAPABILITY_STATUS_LABELS = {
    "implemented": "Implemented",
    "promoted-scoped": "Promoted · scoped",
    "experimental": "Experimental",
    "shelved": "Shelved",
}
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
ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"
SITEMAP_NAMESPACE = "http://www.sitemaps.org/schemas/sitemap/0.9"
PUBLIC_SITE_URL = "https://bitworks-io.github.io/fast-mlx/"
SOCIAL_CARD_PATH = "assets/social-card.png"
SOCIAL_CARD_URL = PUBLIC_SITE_URL + SOCIAL_CARD_PATH
SOCIAL_CARD_ALT = (
    "Abstract emerald data loop connecting research, implementation, testing, "
    "and verified release checkpoints."
)
SOCIAL_CARD_SHA256 = (
    "aa4eaaa35a0dc2280752aab92e6731300e63d272cc5ba6340e0b626f5be610e0"
)
SITE_STYLESHEET_PATH = "assets/site.css"
SITE_STYLESHEET_SHA256 = (
    "5601429bd64b6dbd58da20c10f7f8116f27774a285d574152795435b0522ea81"
)
REVIEWED_HOME_PAGE_BYTES = 8_945
REVIEWED_HOME_PAGE_SHA256 = (
    "27482181f5561faf2af25a895d39769d3500c1d3918b41596a4acef5132c7587"
)
SOCIAL_CARD_BYTES = 1_011_297
SOCIAL_CARD_WIDTH = 1_200
SOCIAL_CARD_HEIGHT = 630
MAX_RELEASE_FEED_BYTES = 1_048_576
MAX_SITEMAP_BYTES = 1_048_576
MAX_ROBOTS_BYTES = 4_096
CORE_PUBLIC_PAGE_PATHS = (
    "",
    "process/",
    "methodology/",
    "capabilities/",
    "benchmarks/",
    "releases/",
    "research/",
)
REVIEWED_ARTICLE_PATHS = (
    "research/the-proof-did-not-end-when-the-timer-did/",
    "research/the-fastest-request-wasnt-the-fastest-service/",
    "research/lossless-wasnt-byte-identical/",
    "research/when-zero-speculation-costs-two-percent/",
    "research/turboquant-exact-math-still-lost/",
    "research/trusting-the-instrument/",
    "research/the-wall-that-wasnt/",
)
HTML_VOID_ELEMENTS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
REVIEWED_PAGE_METADATA: Dict[
    str, Tuple[str, str, str, Optional[str]]
] = {
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
    "research/the-proof-did-not-end-when-the-timer-did/": (
        "The proof did not end when the timer did — fast-mlx",
        "A short benchmark can show that continuous batching works. It cannot show that an HTTP service keeps cleaning up after disappearing clients for a full day.",
        "article",
        "Building a high-performance MLX inference engine in Swift; Serving big models on Apple Silicon; Rapid research integration — the flywheel",
    ),
    "research/the-fastest-request-wasnt-the-fastest-service/": (
        "The fastest request wasn't the fastest service — fast-mlx",
        "On an Apple M5 Max at one request, our fastest exact path was prompt-lookup decoding. Qwen3-32B-4bit generated 28.30 tokens per second with PLD, versus 26.72 through the new continuous-batching runtime. If we had…",
        "article",
        "Building a high-performance MLX inference engine in Swift; Rapid research integration — the flywheel",
    ),
    "research/lossless-wasnt-byte-identical/": (
        "“Lossless” wasn't byte-identical: the speculative decoder that failed at generated index seven — fast-mlx",
        "EAGLE-3 looked like the trained speculative decoder we had been waiting for. A public Qwen3-32B checkpoint matched our production-size target. The draft head was only one decoder layer. Its published algorithm was…",
        "article",
        "Rapid research integration — the flywheel; Building a high-performance MLX engine in Swift",
    ),
    "research/when-zero-speculation-costs-two-percent/": (
        "When zero speculation costs 2%: making a 2× decoder safe to leave on — fast-mlx",
        "Prompt-lookup decoding had already given us the result every inference team wants: nearly twice the decode throughput, with byte-identical output. On a repetition-heavy agent prompt, Qwen3-32B-4bit rose from about 28…",
        "article",
        "Building a high-performance MLX inference engine in Swift; Rapid research integration — the flywheel",
    ),
    "research/turboquant-exact-math-still-lost/": (
        "We implemented Google's TurboQuant exactly, matched the paper's error tables — and it still lost to plain 4-bit quantization — fast-mlx",
        "The KV cache is the memory bill for long context. On Qwen3-32B, every token you keep costs 256 KiB of fp16 keys and values — 64 layers × 8 KV heads × 128 dims × 2 tensors. At a 24K-token context that's 6 GB per…",
        "article",
        "The optimization dial — quantified precision-loss tuning; Building a high-performance MLX engine in Swift",
    ),
    "research/trusting-the-instrument/": (
        "Who measures the measurer? Auditing a precision-loss harness that was quietly lying — fast-mlx",
        "fast-mlx's product isn't raw speed — it's a dial: turn up the compression, and see exactly how much accuracy you trade. That promise lives or dies on one thing — the instrument that produces the \"how much accuracy\"…",
        "article",
        "The optimization dial — quantified precision-loss tuning",
    ),
    "research/the-wall-that-wasnt/": (
        "The 7K wall that wasn't: jetsam forensics, a quadratic allocator, and the statistic hiding in the tail — fast-mlx",
        "Our precision-loss harness had just been hardened — teacher-forced KL, perplexity, a versioned corpus, provenance records. Then it hit a wall: any measurement past roughly 7,000 tokens of context died with a SIGKILL…",
        "article",
        "The optimization dial — quantified precision-loss tuning",
    ),
}
SITEMAP_ARTICLE_PATH = re.compile(r"research/[a-z0-9]+(?:-[a-z0-9]+)*/")
HTML_LIKE_SUFFIXES = {".html", ".htm"}
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


class HeadMetadataCollector(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.canonicals: List[str] = []
        self.properties: Dict[str, List[str]] = {}
        self.metadata_outside_head = False
        self.invalid_head_structure = False
        self._in_head = False
        self._head_seen = False
        self._body_started = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag == "head":
            if self._head_seen or self._body_started:
                self.invalid_head_structure = True
                self._in_head = False
            else:
                self._head_seen = True
                self._in_head = True
            return
        if tag == "body":
            self._body_started = True
            self._in_head = False
        attribute_names = [name.casefold() for name, _value in attrs]
        if tag in {"link", "meta"} and len(attribute_names) != len(
            set(attribute_names)
        ):
            self.invalid_head_structure = True
        attributes = dict(attrs)
        rel_tokens = {
            token.casefold()
            for token in (attributes.get("rel") or "").split()
        }
        is_canonical = tag == "link" and "canonical" in rel_tokens
        property_name = attributes.get("property") if tag == "meta" else None
        is_social = isinstance(property_name, str) and (
            property_name.casefold().startswith("og:")
            or property_name.casefold().startswith("article:")
        )
        if not is_canonical and not is_social:
            return
        if not self._in_head:
            self.metadata_outside_head = True
        if is_canonical:
            href = attributes.get("href")
            self.canonicals.append(href if isinstance(href, str) else "")
        if is_social and isinstance(property_name, str):
            content = attributes.get("content")
            self.properties.setdefault(property_name, []).append(
                content if isinstance(content, str) else ""
            )

    def handle_endtag(self, tag: str) -> None:
        if tag == "head":
            self._in_head = False


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
        self.has_atom_link = False
        self.has_atom_action = False
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
                "anchor": attributes.get("id"),
                "publicCommit": attributes.get("data-public-commit"),
                "datetime": None,
                "links": [],
                "text_parts": [],
                "hidden": "hidden" in attributes,
            }

        if tag == "a" and attributes.get("href") == "index.json":
            self.has_json_link = True
        if (
            tag == "a"
            and attributes.get("href") == "feed.atom"
            and attributes.get("type") == "application/atom+xml"
        ):
            self.has_atom_action = True
        if (
            tag == "link"
            and attributes.get("rel") == "alternate"
            and attributes.get("type") == "application/atom+xml"
            and attributes.get("title") == "fast-mlx reviewed releases"
            and attributes.get("href") == "../releases/feed.atom"
        ):
            self.has_atom_link = True
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


class HomeCurrentCycleCollector(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sections: List[Dict[str, object]] = []
        self._current: Optional[Dict[str, object]] = None
        self._section_depth = 0
        self._element_stack: List[
            Tuple[str, Dict[str, Optional[str]], bool]
        ] = []

    @staticmethod
    def _suppresses_visibility(attributes: Dict[str, Optional[str]]) -> bool:
        classes = set((attributes.get("class") or "").split())
        aria_hidden = (attributes.get("aria-hidden") or "").casefold()
        return (
            "hidden" in attributes
            or aria_hidden == "true"
            or "style" in attributes
            or "benchmark-controls" in classes
            or "data-benchmark-controls" in attributes
        )

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        names = [name for name, _value in attrs]
        has_duplicate_attributes = len(names) != len(set(names))
        attributes = dict(attrs)
        if tag == "section" and "data-current-cycle" in attributes:
            if self._current is not None:
                self.sections.append(self._current)
            self._current = {
                "sectionAttributes": attributes,
                "ancestry": list(self._element_stack),
                "hasDuplicateAttributes": has_duplicate_attributes,
                "hasVisibilitySuppressor": self._suppresses_visibility(attributes),
                "latestReleaseId": attributes.get("data-latest-release-id"),
                "boundaryId": attributes.get("data-boundary-id"),
                "boundaryState": attributes.get("data-boundary-state"),
                "time": None,
                "links": [],
                "statusCounts": [],
                "text_parts": [],
                "inventoryRole": None,
                "listItemCount": 0,
            }
            self._section_depth = 1
        elif self._current is not None:
            if has_duplicate_attributes:
                self._current["hasDuplicateAttributes"] = True
            if self._suppresses_visibility(attributes):
                self._current["hasVisibilitySuppressor"] = True
            if tag == "section":
                self._section_depth += 1
            if tag == "time":
                self._current["time"] = attributes.get("datetime")
            elif tag == "a" and attributes.get("href"):
                links = self._current["links"]
                if isinstance(links, list):
                    links.append(attributes["href"])
            if "data-capability-status" in attributes:
                counts = self._current["statusCounts"]
                if isinstance(counts, list):
                    counts.append(
                        (
                            attributes.get("data-capability-status"),
                            attributes.get("data-count"),
                        )
                    )
            classes = set((attributes.get("class") or "").split())
            if "capability-list" in classes:
                self._current["inventoryRole"] = attributes.get("role")
            if attributes.get("role") == "listitem":
                count = self._current["listItemCount"]
                self._current["listItemCount"] = (
                    count + 1 if isinstance(count, int) else 1
                )
        if tag not in HTML_VOID_ELEMENTS:
            self._element_stack.append((tag, attributes, has_duplicate_attributes))

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            text_parts = self._current["text_parts"]
            if isinstance(text_parts, list):
                text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "section" and self._current is not None:
            self._section_depth -= 1
            if self._section_depth == 0:
                text_parts = self._current.pop("text_parts")
                self._current["text"] = " ".join(
                    "".join(text_parts).split()
                ) if isinstance(text_parts, list) else ""
                self.sections.append(self._current)
                self._current = None
        for index in range(len(self._element_stack) - 1, -1, -1):
            if self._element_stack[index][0] == tag:
                del self._element_stack[index:]
                break


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


def render_expected_release_feed(release_index: Dict[str, object]) -> str:
    """Recreate the one canonical Atom document accepted by the Pages validator."""

    ET.register_namespace("", ATOM_NAMESPACE)
    atom = lambda name: f"{{{ATOM_NAMESPACE}}}{name}"
    feed = ET.Element(atom("feed"))
    ET.SubElement(feed, atom("title")).text = "fast-mlx reviewed releases"
    ET.SubElement(feed, atom("id")).text = PUBLIC_SITE_URL + "releases/"
    releases = release_index["releases"]
    ET.SubElement(feed, atom("updated")).text = str(releases[0]["publishedAt"])
    author = ET.SubElement(feed, atom("author"))
    ET.SubElement(author, atom("name")).text = "fast-mlx contributors"
    ET.SubElement(
        feed,
        atom("link"),
        {
            "rel": "self",
            "type": "application/atom+xml",
            "href": PUBLIC_SITE_URL + "releases/feed.atom",
        },
    )
    ET.SubElement(
        feed,
        atom("link"),
        {
            "rel": "alternate",
            "type": "text/html",
            "href": PUBLIC_SITE_URL + "releases/",
        },
    )
    for release in releases:
        entry = ET.SubElement(feed, atom("entry"))
        ET.SubElement(entry, atom("title")).text = str(release["title"])
        ET.SubElement(entry, atom("id")).text = (
            "urn:fast-mlx:public-commit:" + str(release["publicCommit"])
        )
        ET.SubElement(entry, atom("published")).text = str(release["publishedAt"])
        ET.SubElement(entry, atom("updated")).text = str(release["publishedAt"])
        ET.SubElement(
            entry, atom("category"), {"term": str(release["category"])}
        )
        ET.SubElement(
            entry,
            atom("link"),
            {
                "rel": "alternate",
                "type": "text/html",
                "href": (
                    PUBLIC_SITE_URL
                    + "releases/#release-"
                    + str(release["id"])
                ),
            },
        )
        ET.SubElement(
            entry,
            atom("link"),
            {"rel": "via", "href": str(release["sourceUrl"])},
        )
        ET.SubElement(entry, atom("summary")).text = (
            str(release["summary"]) + " Boundary: " + str(release["scope"])
        )
    ET.indent(feed, space="  ")
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        + ET.tostring(feed, encoding="unicode", short_empty_elements=True)
        + "\n"
    )


def validate_release_feed(site: Path, release_index: Dict[str, object]) -> List[str]:
    failures: List[str] = []
    feed_path = site / "releases/feed.atom"
    if feed_path.is_symlink() or not feed_path.is_file():
        return ["releases/feed.atom must be a regular non-symlink file"]
    try:
        feed_size = feed_path.stat().st_size
    except OSError as exc:
        return [f"cannot stat releases/feed.atom: {exc}"]
    if feed_size > MAX_RELEASE_FEED_BYTES:
        return ["releases/feed.atom exceeds the 1048576-byte limit"]
    try:
        raw_feed = feed_path.read_bytes()
    except OSError as exc:
        return [f"cannot read releases/feed.atom: {exc}"]
    if len(raw_feed) > MAX_RELEASE_FEED_BYTES:
        return ["releases/feed.atom exceeds the 1048576-byte limit"]
    try:
        feed_text = raw_feed.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [f"releases/feed.atom is not UTF-8: {exc}"]
    upper_feed = feed_text.upper()
    if "<!DOCTYPE" in upper_feed or "<!ENTITY" in upper_feed:
        return ["releases/feed.atom contains a forbidden XML declaration"]
    try:
        feed = ET.fromstring(feed_text)
    except ET.ParseError as exc:
        return [f"invalid releases/feed.atom: {exc}"]
    if feed.tag != f"{{{ATOM_NAMESPACE}}}feed":
        failures.append("releases/feed.atom is not an Atom 1.0 feed")

    releases = release_index.get("releases")
    if not isinstance(releases, list) or not releases:
        return failures
    if not all(isinstance(release, dict) for release in releases):
        return failures
    try:
        expected_feed = render_expected_release_feed(release_index)
    except (KeyError, IndexError, TypeError):
        return failures
    if feed_text != expected_feed:
        failures.append("releases/feed.atom does not match releases/index.json")
    return failures


def render_expected_sitemap(article_paths: Sequence[str]) -> str:
    """Recreate the one canonical sitemap accepted by the Pages validator."""

    ET.register_namespace("", SITEMAP_NAMESPACE)
    sitemap = ET.Element(f"{{{SITEMAP_NAMESPACE}}}urlset")
    for public_path in (*CORE_PUBLIC_PAGE_PATHS, *article_paths):
        url = ET.SubElement(sitemap, f"{{{SITEMAP_NAMESPACE}}}url")
        ET.SubElement(url, f"{{{SITEMAP_NAMESPACE}}}loc").text = (
            PUBLIC_SITE_URL + public_path
        )
    ET.indent(sitemap, space="  ")
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        + ET.tostring(sitemap, encoding="unicode", short_empty_elements=True)
        + "\n"
    )


def validate_sitemap(site: Path) -> List[str]:
    failures: List[str] = []
    sitemap_path = site / "sitemap.xml"
    if sitemap_path.is_symlink() or not sitemap_path.is_file():
        return ["sitemap.xml must be a regular non-symlink file"]
    try:
        sitemap_size = sitemap_path.stat().st_size
    except OSError as exc:
        return [f"cannot stat sitemap.xml: {exc}"]
    if sitemap_size > MAX_SITEMAP_BYTES:
        return ["sitemap.xml exceeds the 1048576-byte limit"]
    try:
        raw_sitemap = sitemap_path.read_bytes()
    except OSError as exc:
        return [f"cannot read sitemap.xml: {exc}"]
    if len(raw_sitemap) > MAX_SITEMAP_BYTES:
        return ["sitemap.xml exceeds the 1048576-byte limit"]
    try:
        sitemap_text = raw_sitemap.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [f"sitemap.xml is not UTF-8: {exc}"]
    upper_sitemap = sitemap_text.upper()
    if "<!DOCTYPE" in upper_sitemap or "<!ENTITY" in upper_sitemap:
        return ["sitemap.xml contains a forbidden XML declaration"]
    try:
        sitemap = ET.fromstring(sitemap_text)
    except ET.ParseError as exc:
        return [f"invalid sitemap.xml: {exc}"]
    if sitemap.tag != f"{{{SITEMAP_NAMESPACE}}}urlset":
        failures.append("sitemap.xml is not a Sitemap protocol urlset")
    if sitemap_text != render_expected_sitemap(REVIEWED_ARTICLE_PATHS):
        failures.append("sitemap.xml does not match reviewed public routes")
    return failures


def validate_robots(site: Path) -> List[str]:
    robots_path = site / "robots.txt"
    if robots_path.is_symlink() or not robots_path.is_file():
        return ["robots.txt must be a regular non-symlink file"]
    try:
        robots_size = robots_path.stat().st_size
    except OSError as exc:
        return [f"cannot stat robots.txt: {exc}"]
    if robots_size > MAX_ROBOTS_BYTES:
        return ["robots.txt exceeds the 4096-byte limit"]
    try:
        raw_robots = robots_path.read_bytes()
    except OSError as exc:
        return [f"cannot read robots.txt: {exc}"]
    if len(raw_robots) > MAX_ROBOTS_BYTES:
        return ["robots.txt exceeds the 4096-byte limit"]
    try:
        robots_text = raw_robots.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [f"robots.txt is not UTF-8: {exc}"]
    expected = (
        "User-agent: *\n"
        "Allow: /\n"
        f"Sitemap: {PUBLIC_SITE_URL}sitemap.xml\n"
    )
    if robots_text != expected:
        return ["robots.txt does not match the canonical crawl policy"]
    return []


def validate_social_card(site: Path) -> List[str]:
    path = site / SOCIAL_CARD_PATH
    if path.is_symlink() or not path.is_file():
        return [f"{SOCIAL_CARD_PATH} must be a regular non-symlink file"]
    try:
        size = path.stat().st_size
    except OSError as exc:
        return [f"cannot stat {SOCIAL_CARD_PATH}: {exc}"]
    if size != SOCIAL_CARD_BYTES:
        return [f"{SOCIAL_CARD_PATH} has the wrong byte count"]
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [f"cannot read {SOCIAL_CARD_PATH}: {exc}"]
    if len(raw) != SOCIAL_CARD_BYTES:
        return [f"{SOCIAL_CARD_PATH} has the wrong byte count"]
    if hashlib.sha256(raw).hexdigest() != SOCIAL_CARD_SHA256:
        return [f"{SOCIAL_CARD_PATH} has the wrong SHA-256"]
    if (
        raw[:8] != b"\x89PNG\r\n\x1a\n"
        or raw[8:12] != (13).to_bytes(4, "big")
        or raw[12:16] != b"IHDR"
        or int.from_bytes(raw[16:20], "big") != SOCIAL_CARD_WIDTH
        or int.from_bytes(raw[20:24], "big") != SOCIAL_CARD_HEIGHT
        or raw[24] != 8
        or raw[25] != 2
    ):
        return [f"{SOCIAL_CARD_PATH} is not the reviewed 1200x630 RGB PNG"]
    return []


def validate_reviewed_stylesheet(site: Path) -> List[str]:
    path = site / SITE_STYLESHEET_PATH
    if path.is_symlink() or not path.is_file():
        return [f"{SITE_STYLESHEET_PATH} must be a regular non-symlink file"]
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [f"cannot read {SITE_STYLESHEET_PATH}: {exc}"]
    if hashlib.sha256(raw).hexdigest() != SITE_STYLESHEET_SHA256:
        return [f"{SITE_STYLESHEET_PATH} does not match the reviewed stylesheet"]
    return []


def validate_reviewed_home_page(site: Path) -> List[str]:
    path = site / "index.html"
    if path.is_symlink() or not path.is_file():
        return ["index.html must be a regular non-symlink file"]
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [f"cannot read index.html: {exc}"]
    if (
        len(raw) != REVIEWED_HOME_PAGE_BYTES
        or hashlib.sha256(raw).hexdigest() != REVIEWED_HOME_PAGE_SHA256
    ):
        return ["index.html does not match the reviewed home page"]
    return []


def expected_social_properties(
    public_path: str,
    metadata: Tuple[str, str, str, Optional[str]],
) -> Dict[str, List[str]]:
    title, description, page_type, article_section = metadata
    canonical = PUBLIC_SITE_URL + public_path
    properties = {
        "og:title": [title],
        "og:type": [page_type],
        "og:image": [SOCIAL_CARD_URL],
        "og:image:width": [str(SOCIAL_CARD_WIDTH)],
        "og:image:height": [str(SOCIAL_CARD_HEIGHT)],
        "og:image:alt": [SOCIAL_CARD_ALT],
        "og:url": [canonical],
        "og:description": [description],
        "og:site_name": ["fast-mlx"],
    }
    if page_type == "article":
        if article_section is None:
            raise ValueError(f"incomplete reviewed article metadata for {public_path}")
        properties["article:section"] = [article_section]
    elif article_section is not None:
        raise ValueError(f"article metadata attached to core page {public_path}")
    return properties


def validate_reviewed_head_metadata(site: Path) -> List[str]:
    failures: List[str] = []
    expected_files = {
        (public_path + "index.html") if public_path else "index.html": public_path
        for public_path in REVIEWED_PAGE_METADATA
    }
    allowed_html_files = set(expected_files) | {"404.html"}
    actual_html_files = {
        path.relative_to(site).as_posix()
        for path in site.rglob("*")
        if (
            path.is_file()
            and not path.is_symlink()
            and path.suffix.casefold() in HTML_LIKE_SUFFIXES
        )
    }
    for relative in sorted(actual_html_files - allowed_html_files):
        failures.append(
            f"unexpected HTML page outside the reviewed route set: {relative}"
        )

    for relative, public_path in expected_files.items():
        path = site / relative
        if path.is_symlink() or not path.is_file():
            continue
        collector = HeadMetadataCollector()
        try:
            collector.feed(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"cannot parse {relative} metadata: {exc}")
            continue
        canonical = PUBLIC_SITE_URL + public_path
        expected_properties = expected_social_properties(
            public_path, REVIEWED_PAGE_METADATA[public_path]
        )
        if (
            collector.invalid_head_structure
            or collector.metadata_outside_head
            or collector.canonicals != [canonical]
            or collector.properties != expected_properties
        ):
            failures.append(f"{relative} metadata does not match reviewed contract")

    not_found = site / "404.html"
    if not_found.is_file() and not not_found.is_symlink():
        collector = HeadMetadataCollector()
        try:
            collector.feed(not_found.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"cannot parse 404.html metadata: {exc}")
        else:
            if (
                collector.invalid_head_structure
                or collector.metadata_outside_head
                or collector.canonicals
                or collector.properties
            ):
                failures.append(
                    "404.html must not publish canonical or social metadata"
                )
    return failures


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
    if not collector.has_atom_link:
        failures.append("releases/index.html does not advertise releases/feed.atom")
    if not collector.has_atom_action:
        failures.append("releases/index.html does not link to releases/feed.atom")

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
        if card.get("anchor") != "release-" + str(identifier):
            failures.append(f"release page card {identifier!r} has the wrong anchor")
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


def validate_home_current_cycle(
    site: Path,
    release_index: Dict[str, object],
    capability_index: Dict[str, object],
    research_index: Dict[str, object],
) -> List[str]:
    failures: List[str] = []
    collector = HomeCurrentCycleCollector()
    try:
        collector.feed((site / "index.html").read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"cannot parse index.html current cycle: {exc}"]
    if len(collector.sections) != 1:
        return ["index.html must contain exactly one home current-cycle section"]

    current = collector.sections[0]
    releases = release_index.get("releases")
    latest = releases[0] if isinstance(releases, list) and releases else None
    boundary = release_index.get("currentBoundary")
    capabilities = capability_index.get("capabilities")
    articles = research_index.get("articles")
    if not isinstance(latest, dict) or not isinstance(boundary, dict):
        return failures
    if not isinstance(capabilities, list) or not isinstance(articles, list):
        return failures

    expected_section_attributes = {
        "class": "section shell split",
        "aria-labelledby": "current-cycle-heading",
        "data-current-cycle": None,
        "data-latest-release-id": latest.get("id"),
        "data-boundary-id": boundary.get("id"),
        "data-boundary-state": boundary.get("state"),
    }
    if current.get("sectionAttributes") != expected_section_attributes:
        failures.append(
            "home current-cycle section attributes do not match the reviewed contract"
        )
    expected_ancestry = [
        ("html", {"lang": "en"}, False),
        ("body", {}, False),
        ("main", {"id": "content"}, False),
    ]
    if current.get("ancestry") != expected_ancestry:
        failures.append(
            "home current-cycle section ancestry does not match the reviewed contract"
        )
    if current.get("hasDuplicateAttributes") is not False:
        failures.append("home current-cycle section contains duplicate attributes")
    if current.get("hasVisibilitySuppressor") is not False:
        failures.append("home current-cycle section contains a visibility suppressor")

    if current.get("latestReleaseId") != latest.get("id"):
        failures.append(
            "home current-cycle latest release does not match releases/index.json"
        )
    if current.get("boundaryId") != boundary.get("id"):
        failures.append(
            "home current-cycle boundary id does not match releases/index.json"
        )
    if current.get("boundaryState") != boundary.get("state"):
        failures.append(
            "home current-cycle boundary state does not match releases/index.json"
        )
    if current.get("time") != latest.get("publishedAt"):
        failures.append(
            "home current-cycle timestamp does not match releases/index.json"
        )

    text = current.get("text")
    normalized_text = text if isinstance(text, str) else ""
    latest_text = (
        ("state", str(latest.get("state", "")).upper()),
        ("date", str(latest.get("publishedAt", ""))[:10]),
        ("title", latest.get("title")),
        ("summary", latest.get("summary")),
        ("scope", latest.get("scope")),
    )
    for label, value in latest_text:
        if isinstance(value, str) and " ".join(value.split()) not in normalized_text:
            failures.append(f"home current-cycle latest release does not bind {label}")
    for label, value in (
        ("label", boundary.get("label")),
        ("summary", boundary.get("summary")),
    ):
        if isinstance(value, str) and " ".join(value.split()) not in normalized_text:
            failures.append(f"home current-cycle boundary has the wrong {label}")

    expected_capability_text = f"{len(capabilities)} reviewed capabilities"
    if expected_capability_text not in normalized_text:
        failures.append(
            "home current-cycle capability count does not match capabilities/index.json"
        )
    expected_research_text = f"{len(articles)} published research notes"
    if expected_research_text not in normalized_text:
        failures.append(
            "home current-cycle research count does not match research/index.json"
        )
    expected_release_text = f"{len(releases)} reviewed releases"
    if expected_release_text not in normalized_text:
        failures.append(
            "home current-cycle release count does not match releases/index.json"
        )

    expected_status_counts = {
        status: sum(
            isinstance(capability, dict) and capability.get("status") == status
            for capability in capabilities
        )
        for status in CAPABILITY_STATUSES
    }
    actual_status_counts: Dict[str, int] = {}
    raw_status_counts = current.get("statusCounts")
    if isinstance(raw_status_counts, list):
        for raw_status, raw_count in raw_status_counts:
            if (
                not isinstance(raw_status, str)
                or raw_status in actual_status_counts
                or not isinstance(raw_count, str)
            ):
                actual_status_counts = {}
                break
            try:
                actual_status_counts[raw_status] = int(raw_count)
            except ValueError:
                actual_status_counts = {}
                break
    if actual_status_counts != expected_status_counts:
        failures.append(
            "home current-cycle capability status counts do not match capabilities/index.json"
        )

    links = current.get("links")
    actual_links = links if isinstance(links, list) else []
    expected_source = latest.get("sourceUrl")
    if not isinstance(expected_source, str) or expected_source not in actual_links:
        failures.append("home current-cycle latest release has the wrong source link")
    expected_anchor = "releases/#release-" + str(latest.get("id", ""))
    if expected_anchor not in actual_links:
        failures.append("home current-cycle latest release has the wrong ledger link")
    for expected_link, label in (
        ("capabilities/", "capability"),
        ("research/", "research"),
        ("releases/", "release"),
    ):
        if expected_link not in actual_links:
            failures.append(f"home current-cycle has the wrong {label} inventory link")
    evidence = boundary.get("evidence")
    expected_evidence = evidence.get("path") if isinstance(evidence, dict) else None
    if not isinstance(expected_evidence, str) or expected_evidence not in actual_links:
        failures.append(
            "home current-cycle boundary evidence link does not match releases/index.json"
        )
    if current.get("inventoryRole") != "list" or current.get("listItemCount") != 4:
        failures.append("home current-cycle evidence inventory is not an accessible list")

    status_text = "; ".join(
        f"{expected_status_counts[status]} {CAPABILITY_STATUS_LABELS[status].casefold()}"
        for status in (
            "implemented",
            "promoted-scoped",
            "experimental",
            "shelved",
        )
    )
    expected_text = " ".join(
        [
            "Current reviewed cycle",
            str(latest.get("state", "")).upper(),
            str(latest.get("publishedAt", ""))[:10],
            str(latest.get("title", "")),
            str(latest.get("summary", "")),
            "Boundary:",
            str(latest.get("scope", "")),
            "Inspect the latest reviewed milestone →",
            "Inspect commit",
            str(latest.get("publicCommit", ""))[:12],
            "→",
            expected_capability_text,
            status_text,
            "Inspect capability states →",
            expected_research_text,
            "Dated investigations preserve promoted, shelved, rejected, and diagnostic outcomes.",
            "Read the evidence trail →",
            expected_release_text,
            "Each public milestone names its exact commit and unchanged claim boundary.",
            "Follow the release ledger →",
            str(boundary.get("state", "")).upper(),
            str(boundary.get("label", "")),
            str(boundary.get("summary", "")),
            str(evidence.get("label", "")) if isinstance(evidence, dict) else "",
            "→",
        ]
    )
    expected_text = " ".join(expected_text.split())
    if normalized_text != expected_text:
        failures.append("home current-cycle text does not match reviewed indexes")
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
        "releases/feed.atom",
        "research/index.html",
        "research/index.json",
        "sitemap.xml",
        "robots.txt",
        "assets/site.css",
        "assets/benchmark-explorer.js",
        "assets/favicon.svg",
        SOCIAL_CARD_PATH,
        "llms.txt",
        ".nojekyll",
    ]
    for relative in required:
        if not (site / relative).is_file():
            failures.append(f"missing required file: {relative}")

    failures.extend(validate_social_card(site))
    failures.extend(validate_reviewed_stylesheet(site))
    failures.extend(validate_reviewed_home_page(site))
    failures.extend(validate_reviewed_head_metadata(site))

    expected_benchmark_cards: Dict[str, Dict[str, str]] = {}

    release_index: Optional[Dict[str, object]] = None
    if (site / "releases/index.json").is_file():
        release_index, release_failures = load_release_index(site)
        failures.extend(release_failures)
    if release_index is not None and (site / "releases/index.html").is_file():
        failures.extend(validate_release_page(site, release_index))
    if release_index is not None and (site / "releases/feed.atom").is_file():
        failures.extend(validate_release_feed(site, release_index))

    research_index: Optional[Dict[str, object]] = None
    sitemap_article_paths: List[str] = []
    index_path = site / "research/index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"invalid research/index.json: {exc}")
        else:
            if not isinstance(index, dict):
                failures.append("research/index.json is not an object")
            else:
                research_index = index
                if index.get("schemaVersion") != 1:
                    failures.append("research/index.json does not use schemaVersion 1")
                articles = index.get("articles", [])
                if not isinstance(articles, list) or not articles:
                    failures.append("research/index.json has no articles")
                else:
                    seen_article_paths: set[str] = set()
                    for article in articles:
                        path = article.get("path") if isinstance(article, dict) else None
                        if (
                            not isinstance(path, str)
                            or not SITEMAP_ARTICLE_PATH.fullmatch(path)
                            or path in seen_article_paths
                        ):
                            failures.append(
                                f"invalid or duplicate article path in index entry: {path!r}"
                            )
                            continue
                        seen_article_paths.add(path)
                        sitemap_article_paths.append(path)
                        if not (site / path / "index.html").is_file():
                            failures.append(
                                f"missing article page for index entry: {path!r}"
                            )

                    if tuple(sitemap_article_paths) != REVIEWED_ARTICLE_PATHS:
                        failures.append(
                            "research/index.json does not match reviewed article routes"
                        )

    sitemap_path = site / "sitemap.xml"
    if sitemap_path.exists() or sitemap_path.is_symlink():
        failures.extend(validate_sitemap(site))
    robots_path = site / "robots.txt"
    if robots_path.exists() or robots_path.is_symlink():
        failures.extend(validate_robots(site))

    capability_index: Optional[Dict[str, object]] = None
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

    if (
        release_index is not None
        and capability_index is not None
        and research_index is not None
        and (site / "index.html").is_file()
    ):
        failures.extend(
            validate_home_current_cycle(
                site,
                release_index,
                capability_index,
                research_index,
            )
        )

    for path in site.rglob("*"):
        if path.is_symlink():
            failures.append(f"symlink is forbidden in Pages output: {path.relative_to(site)}")
            continue
        if not path.is_file():
            continue
        if path.relative_to(site).as_posix() == SOCIAL_CARD_PATH:
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
        if path.suffix.casefold() not in HTML_LIKE_SUFFIXES:
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
