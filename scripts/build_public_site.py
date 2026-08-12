#!/usr/bin/env python3
"""Build the dependency-free fast-mlx GitHub Pages site.

Only articles named in site/publications.json are rendered. Markdown is escaped and handled by a
small project-owned renderer so a Pages build does not execute article HTML or depend on a remote
package registry.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import posixpath
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PRIVATE_MARKERS: Tuple[str, ...] = (
    "/" + "Users/",
    "/" + "private/",
    "192" + ".168.",
    "llm" + "bench",
    "passwordless" + " sudo",
    "BEGIN OPENSSH" + " PRIVATE KEY",
    "BEGIN RSA" + " PRIVATE KEY",
)

TABLE_SEPARATOR = re.compile(r"^:?-{3,}:?$")
LIST_ITEM = re.compile(r"^(?:[-+*]\s+|\d+\.\s+)(.+)$")
ORDERED_ITEM = re.compile(r"^\d+\.\s+(.+)$")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
CODE_SPAN = re.compile(r"`([^`]+)`")
SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
WHITEPAPER_THEME = re.compile(
    r"^(?:>\s*)?\*\*Whitepaper themes?:\*\*\s*(.*)$", re.IGNORECASE
)

CAPABILITY_STATUS_DEFINITIONS: Tuple[Tuple[str, str, str], ...] = (
    (
        "implemented",
        "Implemented",
        "The public source and regression contracts exist; this is not automatically a supported default.",
    ),
    (
        "promoted-scoped",
        "Promoted · scoped",
        "A bounded route or result crossed its stated evidence gates only for the named scope.",
    ),
    (
        "experimental",
        "Experimental",
        "The surface is active research and has not earned a production support claim.",
    ),
    (
        "shelved",
        "Shelved",
        "The dated result remains useful evidence, but the capability is not the current production route.",
    ),
)
CAPABILITY_STATUSES = {item[0] for item in CAPABILITY_STATUS_DEFINITIONS}
HIGHLIGHT_DECISIONS = {"promoted-scoped", "shelved"}
RELEASE_CATEGORIES = {"foundation", "operations", "product"}
RELEASE_CATEGORY_LABELS = {
    "foundation": "Foundation",
    "operations": "Operations",
    "product": "Product",
}
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
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
SOCIAL_CARD_BYTES = 1_011_297
SOCIAL_CARD_WIDTH = 1_200
SOCIAL_CARD_HEIGHT = 630
CORE_PUBLIC_PAGE_PATHS = (
    "",
    "process/",
    "methodology/",
    "capabilities/",
    "benchmarks/",
    "releases/",
    "research/",
)
PUBLIC_PATH = re.compile(
    r"(?:[a-z0-9][a-z0-9.-]*/)*(?:[a-z0-9][a-z0-9.-]*/|[a-z0-9][a-z0-9.-]*\.(?:html|json))"
)


@dataclass(frozen=True)
class Article:
    source: Path
    source_name: str
    slug: str
    title: str
    date: str
    theme: str
    summary: str
    reviewed_at: str
    body: str

    @property
    def output_file(self) -> str:
        return f"research/{self.slug}/index.html"

    @property
    def public_path(self) -> str:
        return f"research/{self.slug}/"


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="fast-mlx checkout root (defaults to the script's parent checkout)",
    )
    parser.add_argument("--output", type=Path, required=True, help="absent or empty output directory")
    return parser.parse_args(argv)


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"public-site build refused: {message}")


def read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {path}: {exc}")


def prepare_output(output: Path, repository_root: Path) -> None:
    resolved = output.resolve()
    resolved_root = repository_root.resolve()
    if resolved == resolved_root or resolved_root in resolved.parents:
        fail("output must not be inside the repository root")
    if output.exists():
        if not output.is_dir():
            fail(f"output exists and is not a directory: {output}")
        if any(output.iterdir()):
            fail(f"output directory is not empty: {output}")
    else:
        output.mkdir(parents=True)


def validate_asset_tree(assets: Path) -> None:
    if not assets.is_dir() or assets.is_symlink():
        fail("site/assets is missing, not a directory, or a symlink")
    for path in assets.rglob("*"):
        if path.is_symlink():
            fail(f"site asset is a symlink: {path.relative_to(assets)}")
        if not path.is_file() and not path.is_dir():
            fail(f"site asset is not a regular file or directory: {path.relative_to(assets)}")
    validate_social_card(assets / "social-card.png", "site social card")


def validate_social_card(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        fail(f"{label} is missing, not a regular file, or a symlink")
    try:
        size = path.stat().st_size
    except OSError as exc:
        fail(f"cannot stat {label}: {exc}")
    if size != SOCIAL_CARD_BYTES:
        fail(f"{label} has the wrong byte count")
    raw = path.read_bytes()
    if len(raw) != SOCIAL_CARD_BYTES:
        fail(f"{label} has the wrong byte count")
    if hashlib.sha256(raw).hexdigest() != SOCIAL_CARD_SHA256:
        fail(f"{label} has the wrong SHA-256")
    if (
        raw[:8] != b"\x89PNG\r\n\x1a\n"
        or raw[8:12] != (13).to_bytes(4, "big")
        or raw[12:16] != b"IHDR"
        or int.from_bytes(raw[16:20], "big") != SOCIAL_CARD_WIDTH
        or int.from_bytes(raw[20:24], "big") != SOCIAL_CARD_HEIGHT
        or raw[24] != 8
        or raw[25] != 2
    ):
        fail(f"{label} is not the reviewed 1200x630 RGB PNG")


def strip_front_matter(text: str) -> Tuple[Dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        fail("unterminated article front matter")
    metadata: Dict[str, str] = {}
    for raw_line in text[4:end].splitlines():
        if not raw_line.strip():
            continue
        if ":" not in raw_line:
            fail(f"invalid front-matter line: {raw_line}")
        key, value = raw_line.split(":", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        metadata[key.strip()] = value
    return metadata, text[end + 5 :]


def plain_text(markdown: str) -> str:
    value = LINK.sub(lambda match: match.group(1), markdown)
    value = re.sub(r"[*_`>#]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def find_whitepaper_theme_block(lines: Sequence[str]) -> Optional[Tuple[int, int, str]]:
    """Return the half-open line span and joined value for a wrapped theme paragraph."""

    for start, line in enumerate(lines):
        match = WHITEPAPER_THEME.match(line.strip())
        if not match:
            continue
        values = [match.group(1).strip()] if match.group(1).strip() else []
        end = start + 1
        while end < len(lines):
            stripped = lines[end].strip()
            if (
                not stripped
                or stripped.startswith(("#", ">", "- ", "* ", "```", "|"))
                or re.match(r"^\d+\.\s", stripped)
            ):
                break
            values.append(stripped)
            end += 1
        return start, end, plain_text(" ".join(values))
    return None


def infer_metadata(source: Path, text: str) -> Tuple[str, str, str, str, str]:
    metadata, body = strip_front_matter(text)
    title = metadata.get("title", "")
    if not title:
        title_match = re.search(r"^#\s+(.+)$", body, flags=re.MULTILINE)
        if not title_match:
            fail(f"article has no H1 title: {source}")
        title = plain_text(title_match.group(1))

    date = metadata.get("date", source.name[:10])
    try:
        dt.date.fromisoformat(date)
    except ValueError:
        fail(f"article has invalid ISO date: {source}: {date}")

    lines = body.splitlines()
    theme_block = find_whitepaper_theme_block(lines)
    theme = metadata.get("whitepaper_theme", "")
    if not theme:
        theme = theme_block[2] if theme_block else "Inference research"

    paragraphs: List[str] = []
    current: List[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if theme_block and theme_block[0] <= index < theme_block[1]:
            continue
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if stripped.startswith(("#", ">", "- ", "* ", "```", "|")):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if re.match(r"^\d+\.\s", stripped):
            continue
        current.append(stripped)
    if current:
        paragraphs.append(" ".join(current))
    summary = next((plain_text(item) for item in paragraphs if len(plain_text(item)) > 60), "")
    if not summary:
        fail(f"article has no usable summary paragraph: {source}")
    if len(summary) > 220:
        summary = summary[:217].rsplit(" ", 1)[0] + "…"
    return title, date, theme, summary, body


def require_exact_keys(value: object, required: set[str], label: str) -> Dict[str, object]:
    if not isinstance(value, dict):
        fail(f"{label} is not an object")
    keys = set(value)
    if keys != required:
        missing = sorted(required - keys)
        extra = sorted(keys - required)
        fail(f"{label} keys differ from schema; missing={missing} extra={extra}")
    return value


def require_text(entry: Dict[str, object], key: str, label: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        fail(f"{label} has an empty or non-string {key}")
    if value != value.strip():
        fail(f"{label} {key} contains surrounding whitespace")
    return value


def require_iso_date(entry: Dict[str, object], key: str, label: str) -> str:
    value = require_text(entry, key, label)
    try:
        dt.date.fromisoformat(value)
    except ValueError:
        fail(f"{label} {key} is not an ISO date")
    return value


def load_capability_catalog(
    repository_root: Path, published_slugs: Iterable[str]
) -> Dict[str, object]:
    """Load the strict public capability contract and bind it to reviewed article slugs."""

    manifest_path = repository_root / "site/capabilities.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        fail("site/capabilities.json is missing, not a file, or a symlink")
    catalog = require_exact_keys(
        read_json(manifest_path),
        {
            "schemaVersion",
            "policy",
            "claimBoundary",
            "updatedAt",
            "capabilities",
            "performanceHighlights",
        },
        "site/capabilities.json",
    )
    if catalog.get("schemaVersion") != 1:
        fail("site/capabilities.json must use schemaVersion 1")
    if catalog.get("policy") != "fast-mlx-owned-results-only":
        fail("capability policy must remain fast-mlx-owned-results-only")
    if catalog.get("claimBoundary") != "fast-mlx-owned-results-only":
        fail("capability claim boundary must remain fast-mlx-owned-results-only")
    require_iso_date(catalog, "updatedAt", "site/capabilities.json")

    serialized = json.dumps(catalog, ensure_ascii=False)
    for marker in PRIVATE_MARKERS:
        if marker.casefold() in serialized.casefold():
            fail(f"capability catalog contains private marker {marker!r}")

    published = set(published_slugs)
    capabilities = catalog.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities:
        fail("capability catalog must contain at least one capability")
    seen_ids: set[str] = set()
    for index, raw_entry in enumerate(capabilities):
        label = f"capability entry {index}"
        entry = require_exact_keys(
            raw_entry,
            {"id", "name", "status", "summary", "scope", "evidenceSlugs"},
            label,
        )
        identifier = require_text(entry, "id", label)
        if not SLUG.fullmatch(identifier) or identifier in seen_ids:
            fail(f"{label} has an invalid or duplicate id")
        seen_ids.add(identifier)
        require_text(entry, "name", label)
        require_text(entry, "summary", label)
        require_text(entry, "scope", label)
        status = require_text(entry, "status", label)
        if status not in CAPABILITY_STATUSES:
            fail(f"{label} has unknown status {status!r}")
        evidence_slugs = entry.get("evidenceSlugs")
        if (
            not isinstance(evidence_slugs, list)
            or not evidence_slugs
            or any(not isinstance(slug, str) or not SLUG.fullmatch(slug) for slug in evidence_slugs)
            or len(set(evidence_slugs)) != len(evidence_slugs)
        ):
            fail(f"{label} has invalid evidenceSlugs")
        unavailable = sorted(set(evidence_slugs) - published)
        if unavailable:
            fail(f"{label} cites unpublished evidence: {unavailable}")

    highlights = catalog.get("performanceHighlights")
    if not isinstance(highlights, list) or not highlights:
        fail("capability catalog must contain at least one performance highlight")
    seen_highlight_ids: set[str] = set()
    for index, raw_entry in enumerate(highlights):
        label = f"performance highlight {index}"
        entry = require_exact_keys(
            raw_entry,
            {
                "id",
                "metric",
                "label",
                "model",
                "hardware",
                "workload",
                "date",
                "decision",
                "caveat",
                "evidenceSlug",
            },
            label,
        )
        identifier = require_text(entry, "id", label)
        if not SLUG.fullmatch(identifier) or identifier in seen_highlight_ids:
            fail(f"{label} has an invalid or duplicate id")
        seen_highlight_ids.add(identifier)
        for key in ("metric", "label", "model", "hardware", "workload", "caveat"):
            require_text(entry, key, label)
        require_iso_date(entry, "date", label)
        decision = require_text(entry, "decision", label)
        if decision not in HIGHLIGHT_DECISIONS:
            fail(f"{label} has unknown decision {decision!r}")
        evidence_slug = require_text(entry, "evidenceSlug", label)
        if not SLUG.fullmatch(evidence_slug) or evidence_slug not in published:
            fail(f"{label} cites unpublished evidence {evidence_slug!r}")
    return catalog


def require_release_timestamp(
    entry: Dict[str, object], key: str, label: str
) -> Tuple[str, dt.datetime]:
    value = require_text(entry, key, label)
    if not RELEASE_TIMESTAMP.fullmatch(value):
        fail(f"{label} {key} is not an offset-aware release timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        fail(f"{label} {key} is not an offset-aware release timestamp")
    if parsed.utcoffset() is None:
        fail(f"{label} {key} is not an offset-aware release timestamp")
    return value, parsed


def require_public_link(value: object, label: str) -> Dict[str, str]:
    link = require_exact_keys(value, {"label", "path"}, label)
    link_label = require_text(link, "label", label)
    path = require_text(link, "path", label)
    if not PUBLIC_PATH.fullmatch(path):
        fail(f"{label} has an invalid public path")
    return {"label": link_label, "path": path}


def load_release_catalog(repository_root: Path) -> Dict[str, object]:
    """Load the strict reviewed public-release ledger without consulting Git or the network."""

    manifest_path = repository_root / "site/releases.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        fail("site/releases.json is missing, not a file, or a symlink")
    catalog = require_exact_keys(
        read_json(manifest_path),
        {
            "schemaVersion",
            "project",
            "policy",
            "claimBoundary",
            "updatedAt",
            "currentBoundary",
            "releases",
        },
        "site/releases.json",
    )
    if catalog.get("schemaVersion") != 1:
        fail("site/releases.json must use schemaVersion 1")
    if catalog.get("project") != "fast-mlx":
        fail("release project must remain fast-mlx")
    if catalog.get("policy") != "reviewed-public-releases-only":
        fail("release policy must remain reviewed-public-releases-only")
    if catalog.get("claimBoundary") != "fast-mlx-owned-results-only":
        fail("release claim boundary must remain fast-mlx-owned-results-only")
    require_iso_date(catalog, "updatedAt", "site/releases.json")

    serialized = json.dumps(catalog, ensure_ascii=False)
    for marker in PRIVATE_MARKERS:
        if marker.casefold() in serialized.casefold():
            fail(f"release catalog contains private marker {marker!r}")

    boundary = require_exact_keys(
        catalog.get("currentBoundary"),
        {"id", "label", "state", "summary", "evidence"},
        "current release boundary",
    )
    boundary_id = require_text(boundary, "id", "current release boundary")
    if not SLUG.fullmatch(boundary_id):
        fail("current release boundary has an invalid id")
    require_text(boundary, "label", "current release boundary")
    require_text(boundary, "summary", "current release boundary")
    if require_text(boundary, "state", "current release boundary") != "gated":
        fail("current release boundary state must remain gated")
    require_public_link(boundary.get("evidence"), "current release boundary evidence")

    releases = catalog.get("releases")
    if not isinstance(releases, list) or not releases:
        fail("release catalog must contain at least one release")
    seen_ids: set[str] = set()
    seen_commits: set[str] = set()
    previous_timestamp: Optional[dt.datetime] = None
    for index, raw_entry in enumerate(releases):
        label = f"release entry {index}"
        entry = require_exact_keys(
            raw_entry,
            {
                "id",
                "title",
                "publishedAt",
                "category",
                "state",
                "summary",
                "scope",
                "publicCommit",
                "publicLinks",
            },
            label,
        )
        identifier = require_text(entry, "id", label)
        if not SLUG.fullmatch(identifier) or identifier in seen_ids:
            fail(f"{label} has an invalid or duplicate id")
        seen_ids.add(identifier)
        for key in ("title", "summary", "scope"):
            require_text(entry, key, label)
        category = require_text(entry, "category", label)
        if category not in RELEASE_CATEGORIES:
            fail(f"{label} has unknown category {category!r}")
        if require_text(entry, "state", label) != "released":
            fail(f"{label} is not explicitly released")
        _timestamp, parsed_timestamp = require_release_timestamp(
            entry, "publishedAt", label
        )
        if previous_timestamp is not None and parsed_timestamp >= previous_timestamp:
            fail("release entries are not strictly newest-first")
        previous_timestamp = parsed_timestamp
        commit = require_text(entry, "publicCommit", label)
        if not COMMIT_SHA.fullmatch(commit) or commit in seen_commits:
            fail(f"{label} has an invalid or duplicate publicCommit")
        seen_commits.add(commit)
        links = entry.get("publicLinks")
        if not isinstance(links, list):
            fail(f"{label} publicLinks is not a list")
        seen_paths: set[str] = set()
        for link_index, raw_link in enumerate(links):
            link = require_public_link(raw_link, f"{label} public link {link_index}")
            if link["path"] in seen_paths:
                fail(f"{label} has a duplicate public link path")
            seen_paths.add(link["path"])
    return catalog


def load_articles(repository_root: Path) -> List[Article]:
    manifest_path = repository_root / "site/publications.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != 1:
        fail("site/publications.json must use schemaVersion 1")
    if manifest.get("policy") != "fast-mlx-owned-results-only":
        fail("publication policy must remain fast-mlx-owned-results-only")
    entries = manifest.get("articles")
    if not isinstance(entries, list) or not entries:
        fail("publication manifest must contain at least one article")

    seen_sources: set[str] = set()
    seen_slugs: set[str] = set()
    articles: List[Article] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            fail(f"article entry {index} is not an object")
        source_name = entry.get("source")
        slug = entry.get("slug")
        reviewed_at = entry.get("reviewedAt")
        if entry.get("status") != "published":
            fail(f"article entry {index} is not explicitly published")
        if not isinstance(source_name, str):
            fail(f"article entry {index} source is outside docs/content")
        source_path = Path(source_name)
        if (
            source_path.is_absolute()
            or ".." in source_path.parts
            or source_path.parent != Path("docs/content")
            or source_path.suffix != ".md"
        ):
            fail(f"article entry {index} source is outside docs/content")
        if not isinstance(slug, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug):
            fail(f"article entry {index} has an invalid slug")
        if not isinstance(reviewed_at, str):
            fail(f"article entry {index} is missing reviewedAt")
        try:
            dt.date.fromisoformat(reviewed_at)
        except ValueError:
            fail(f"article entry {index} reviewedAt is not an ISO date")
        if source_name in seen_sources or slug in seen_slugs:
            fail(f"duplicate source or slug in publication manifest: {source_name}")
        seen_sources.add(source_name)
        seen_slugs.add(slug)

        source = repository_root / source_name
        if not source.is_file() or source.is_symlink():
            fail(f"published source is missing, not a file, or a symlink: {source_name}")
        if source.resolve().parent != (repository_root / "docs/content").resolve():
            fail(f"published source escapes docs/content: {source_name}")
        text = source.read_text(encoding="utf-8")
        for marker in PRIVATE_MARKERS:
            if marker.casefold() in text.casefold():
                fail(f"published source contains private marker {marker!r}: {source_name}")
        title, date, theme, summary, body = infer_metadata(source, text)
        articles.append(
            Article(
                source=source,
                source_name=source_name,
                slug=slug,
                title=title,
                date=date,
                theme=theme,
                summary=summary,
                reviewed_at=reviewed_at,
                body=body,
            )
        )
    return sorted(articles, key=lambda article: (article.date, article.slug), reverse=True)


def relative_href(current_file: str, target_path: str) -> str:
    current_dir = posixpath.dirname(current_file)
    target = target_path.rstrip("/") or "."
    value = posixpath.relpath(target, current_dir or ".")
    if value == ".":
        value = "./"
    elif target_path.endswith("/"):
        value += "/"
    return value


class InlineRenderer:
    def __init__(self, article_routes: Dict[str, str], current_file: str) -> None:
        self.article_routes = article_routes
        self.current_file = current_file
        self.tokens: Dict[str, str] = {}

    def stash(self, value: str) -> str:
        token = f"@@FASTMLX{len(self.tokens)}TOKEN@@"
        self.tokens[token] = value
        return token

    def render_link(self, match: re.Match[str]) -> str:
        label = match.group(1)
        raw_url = html.unescape(match.group(2)).strip()
        if raw_url.startswith(("https://", "http://", "mailto:")):
            href = html.escape(raw_url, quote=True)
            return self.stash(f'<a href="{href}" rel="noreferrer">{label}</a>')
        if raw_url.startswith("#"):
            href = html.escape(raw_url, quote=True)
            return self.stash(f'<a href="{href}">{label}</a>')

        path, separator, fragment = raw_url.partition("#")
        route = self.article_routes.get(posixpath.basename(path))
        if route:
            href = relative_href(self.current_file, route)
            if separator:
                href += "#" + fragment
            return self.stash(f'<a href="{html.escape(href, quote=True)}">{label}</a>')

        title = html.escape(
            "Supporting project evidence is retained outside the public projection", quote=True
        )
        return self.stash(f'<span class="source-boundary" title="{title}">{label}</span>')

    def __call__(self, text: str) -> str:
        escaped = html.escape(text, quote=False)
        escaped = CODE_SPAN.sub(
            lambda match: self.stash(f"<code>{match.group(1)}</code>"), escaped
        )
        escaped = LINK.sub(self.render_link, escaped)
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
        escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)
        for token, value in self.tokens.items():
            escaped = escaped.replace(token, value)
        return escaped


def table_cells(line: str) -> List[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def is_table(lines: Sequence[str], index: int) -> bool:
    if index + 1 >= len(lines) or "|" not in lines[index]:
        return False
    separators = table_cells(lines[index + 1])
    return bool(separators) and all(TABLE_SEPARATOR.fullmatch(cell) for cell in separators)


def heading_id(value: str) -> str:
    lowered = plain_text(value).casefold()
    return re.sub(r"[^a-z0-9]+", "-", lowered).strip("-") or "section"


def render_markdown(body: str, article_routes: Dict[str, str], current_file: str) -> str:
    lines = body.splitlines()
    theme_block = find_whitepaper_theme_block(lines)
    inline = InlineRenderer(article_routes, current_file)
    output: List[str] = []
    paragraph: List[str] = []
    title_skipped = False

    def flush_paragraph() -> None:
        if paragraph:
            output.append(f"<p>{inline(' '.join(item.strip() for item in paragraph))}</p>")
            paragraph.clear()

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if theme_block and index == theme_block[0]:
            flush_paragraph()
            output.append(
                f'<p class="eyebrow">{inline("**Whitepaper themes:** " + theme_block[2])}</p>'
            )
            index = theme_block[1]
            continue
        if not stripped:
            flush_paragraph()
            index += 1
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            language = re.sub(r"[^a-zA-Z0-9_+-]", "", stripped[3:].strip())
            code: List[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            if index >= len(lines):
                fail("unterminated fenced code block")
            language_class = f' class="language-{language}"' if language else ""
            output.append(
                f"<pre><code{language_class}>{html.escape(chr(10).join(code))}</code></pre>"
            )
            index += 1
            continue

        heading = HEADING.match(stripped)
        if heading:
            flush_paragraph()
            if len(heading.group(1)) == 1 and not title_skipped:
                title_skipped = True
                index += 1
                continue
            level = min(6, max(2, len(heading.group(1))))
            title = heading.group(2)
            output.append(
                f'<h{level} id="{heading_id(title)}">{inline(title)}</h{level}>'
            )
            index += 1
            continue

        if is_table(lines, index):
            flush_paragraph()
            headers = table_cells(lines[index])
            index += 2
            rows: List[List[str]] = []
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                rows.append(table_cells(lines[index]))
                index += 1
            output.append("<div class=\"table-scroll\"><table><thead><tr>")
            output.extend(f"<th>{inline(cell)}</th>" for cell in headers)
            output.append("</tr></thead><tbody>")
            for row in rows:
                padded = row + [""] * (len(headers) - len(row))
                output.append("<tr>")
                output.extend(f"<td>{inline(cell)}</td>" for cell in padded[: len(headers)])
                output.append("</tr>")
            output.append("</tbody></table></div>")
            continue

        if stripped.startswith(">"):
            flush_paragraph()
            quote: List[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote.append(lines[index].strip().lstrip("> "))
                index += 1
            output.append(f"<blockquote><p>{inline(' '.join(quote))}</p></blockquote>")
            continue

        list_match = LIST_ITEM.match(stripped)
        if list_match:
            flush_paragraph()
            ordered = ORDERED_ITEM.match(stripped) is not None
            tag = "ol" if ordered else "ul"
            items: List[str] = []
            while index < len(lines):
                candidate = lines[index].strip()
                match = ORDERED_ITEM.match(candidate) if ordered else re.match(r"^[-+*]\s+(.+)$", candidate)
                if not match:
                    break
                items.append(match.group(1))
                index += 1
            output.append(f"<{tag}>")
            output.extend(f"<li>{inline(item)}</li>" for item in items)
            output.append(f"</{tag}>")
            continue

        if stripped in {"---", "***", "___"}:
            flush_paragraph()
            output.append("<hr>")
            index += 1
            continue

        paragraph.append(stripped)
        index += 1

    flush_paragraph()
    return "\n".join(output)


def render_template(
    template: str,
    title: str,
    description: str,
    root: str,
    body: str,
    current_page: str = "",
    page_script: str = "",
    *,
    public_path: Optional[str] = None,
    article_section: Optional[str] = None,
) -> str:
    nav_root = html.escape(root, quote=True)

    def nav_link(page: str, label: str) -> str:
        current = ' aria-current="page"' if current_page == page else ""
        return f'<a href="{nav_root}{page}/"{current}>{label}</a>'

    replacements = {
        "{{title}}": html.escape(title),
        "{{description}}": html.escape(description, quote=True),
        "{{head_metadata}}": render_head_metadata(
            title,
            description,
            public_path,
            article_section=article_section,
        ),
        "{{root}}": root,
        "{{body}}": body.replace("{{root}}", root),
        "{{capabilities_nav}}": nav_link("capabilities", "Capabilities"),
        "{{benchmarks_nav}}": nav_link("benchmarks", "Benchmarks"),
        "{{releases_nav}}": nav_link("releases", "Releases"),
        "{{process_nav}}": nav_link("process", "The loop"),
        "{{methodology_nav}}": nav_link("methodology", "Methodology"),
        "{{research_nav}}": nav_link("research", "Research notes"),
        "{{page_script}}": page_script,
    }
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    if re.search(r"{{[a-zA-Z0-9_]+}}", rendered):
        fail(f"template contains an unresolved token for {title}")
    return rendered


def render_head_metadata(
    title: str,
    description: str,
    public_path: Optional[str],
    *,
    article_section: Optional[str],
) -> str:
    if public_path is None:
        if article_section is not None:
            fail("article metadata requires a reviewed public path")
        return ""
    if public_path not in CORE_PUBLIC_PAGE_PATHS and not re.fullmatch(
        r"research/[a-z0-9]+(?:-[a-z0-9]+)*/", public_path
    ):
        fail(f"invalid metadata public path: {public_path!r}")
    canonical = PUBLIC_SITE_URL + public_path
    page_type = "article" if article_section is not None else "website"
    tags = [
        f'<link rel="canonical" href="{html.escape(canonical, quote=True)}">',
        f'<meta property="og:title" content="{html.escape(title, quote=True)}">',
        f'<meta property="og:type" content="{page_type}">',
        f'<meta property="og:image" content="{SOCIAL_CARD_URL}">',
        f'<meta property="og:image:width" content="{SOCIAL_CARD_WIDTH}">',
        f'<meta property="og:image:height" content="{SOCIAL_CARD_HEIGHT}">',
        f'<meta property="og:image:alt" content="{html.escape(SOCIAL_CARD_ALT, quote=True)}">',
        f'<meta property="og:url" content="{html.escape(canonical, quote=True)}">',
        f'<meta property="og:description" content="{html.escape(description, quote=True)}">',
        '<meta property="og:site_name" content="fast-mlx">',
    ]
    if article_section is not None:
        tags.append(
            f'<meta property="article:section" content="{html.escape(article_section, quote=True)}">'
        )
    return "\n    ".join(tags)


def write_page(output: Path, relative_file: str, contents: str) -> None:
    destination = output / relative_file
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(contents.rstrip() + "\n", encoding="utf-8")


def render_release_catalog(
    catalog: Dict[str, object]
) -> Tuple[str, Dict[str, object]]:
    boundary = dict(catalog["currentBoundary"])
    boundary["evidence"] = dict(boundary["evidence"])
    releases: List[Dict[str, object]] = []
    for raw_entry in catalog["releases"]:
        entry = dict(raw_entry)
        entry["publicLinks"] = [dict(link) for link in entry["publicLinks"]]
        entry["sourceUrl"] = (
            "https://github.com/bitworks-io/fast-mlx/commit/"
            f"{entry['publicCommit']}"
        )
        releases.append(entry)

    public_index: Dict[str, object] = {
        "schemaVersion": 1,
        "project": "fast-mlx",
        "policy": catalog["policy"],
        "claimBoundary": catalog["claimBoundary"],
        "updatedAt": catalog["updatedAt"],
        "currentBoundary": boundary,
        "releases": releases,
    }

    boundary_evidence = boundary["evidence"]
    boundary_href = relative_href(
        "releases/index.html", str(boundary_evidence["path"])
    )
    body: List[str] = [
        '<section class="page-hero shell release-hero">',
        '<p class="eyebrow">Reviewed releases</p>',
        '<h1>See what shipped—and what it still does not prove.</h1>',
        '<p class="lede">A generated, newest-first ledger of fast-mlx public milestones. Every entry names the exact public commit, the user-facing surface, and the boundary that stayed in force.</p>',
        '<div class="hero-actions">',
        '<a class="button primary" href="index.json">Read the release JSON</a>',
        '<a class="button secondary" href="feed.atom" type="application/atom+xml">Subscribe to reviewed releases</a>',
        '<a class="button secondary" href="https://github.com/bitworks-io/fast-mlx/commits/main" rel="noreferrer">Inspect public history</a>',
        '</div>',
        '</section>',
        '<section class="section shell release-boundary" aria-labelledby="release-boundary-heading" data-release-boundary '
        f'data-boundary-id="{html.escape(str(boundary["id"]), quote=True)}" '
        f'data-boundary-state="{html.escape(str(boundary["state"]), quote=True)}">',
        '<div class="section-heading">',
        '<p class="eyebrow">Current boundary</p>',
        f'<span class="status-badge status-gated">{html.escape(str(boundary["state"]).upper())}</span>',
        f'<h2 id="release-boundary-heading">{html.escape(str(boundary["label"]))}</h2>',
        f'<p class="section-intro">{html.escape(str(boundary["summary"]))}</p>',
        f'<a class="text-link" href="{html.escape(boundary_href, quote=True)}">{html.escape(str(boundary_evidence["label"]))} →</a>',
        '</div>',
        '</section>',
        '<section class="section shell" aria-labelledby="release-ledger-heading">',
        '<div class="section-heading"><p class="eyebrow">Public history</p><h2 id="release-ledger-heading">One reviewed commit at a time.</h2><p class="section-intro">Operational hardening and product surfaces appear together because both determine what users can trust.</p></div>',
        '<ol class="release-list">',
    ]
    for release in releases:
        category = str(release["category"])
        commit = str(release["publicCommit"])
        published_at = str(release["publishedAt"])
        body.extend(
            [
                '<li>',
                '<article class="release-card" '
                f'id="release-{html.escape(str(release["id"]), quote=True)}" '
                f'data-release-id="{html.escape(str(release["id"]), quote=True)}" '
                f'data-public-commit="{html.escape(commit, quote=True)}">',
                '<div class="card-topline">',
                f'<span class="status-badge status-released">{html.escape(str(release["state"]).upper())}</span>',
                f'<time datetime="{html.escape(published_at, quote=True)}">{html.escape(published_at[:10])}</time>',
                '</div>',
                f'<p class="release-category">{html.escape(RELEASE_CATEGORY_LABELS[category])}</p>',
                f'<h3>{html.escape(str(release["title"]))}</h3>',
                f'<p>{html.escape(str(release["summary"]))}</p>',
                f'<p class="scope-note"><strong>Boundary:</strong> {html.escape(str(release["scope"]))}</p>',
                '<div class="release-links">',
                f'<a class="text-link" href="{html.escape(str(release["sourceUrl"]), quote=True)}" rel="noreferrer">Inspect commit {html.escape(commit[:12])} →</a>',
            ]
        )
        for link in release["publicLinks"]:
            href = relative_href("releases/index.html", str(link["path"]))
            body.append(
                f'<a href="{html.escape(href, quote=True)}">{html.escape(str(link["label"]))} →</a>'
            )
        body.extend(['</div>', '</article>', '</li>'])
    body.extend(['</ol>', '</section>'])
    return "\n".join(body), public_index


def render_release_feed(release_index: Dict[str, object]) -> str:
    """Render a deterministic Atom 1.0 feed from the reviewed public release index."""

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
    )


def render_sitemap(articles: Sequence[Article]) -> str:
    """Render canonical human-facing routes without inventing crawl metadata."""

    ET.register_namespace("", SITEMAP_NAMESPACE)
    sitemap = ET.Element(f"{{{SITEMAP_NAMESPACE}}}urlset")
    public_paths = [
        *CORE_PUBLIC_PAGE_PATHS,
        *[article.public_path for article in articles],
    ]
    if len(public_paths) != len(set(public_paths)):
        fail("reviewed sitemap routes are not unique")
    for public_path in public_paths:
        url = ET.SubElement(sitemap, f"{{{SITEMAP_NAMESPACE}}}url")
        ET.SubElement(url, f"{{{SITEMAP_NAMESPACE}}}loc").text = (
            PUBLIC_SITE_URL + public_path
        )
    ET.indent(sitemap, space="  ")
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        + ET.tostring(sitemap, encoding="unicode", short_empty_elements=True)
    )


def render_robots() -> str:
    """Render the exact public crawl policy and canonical sitemap locator."""

    return (
        "User-agent: *\n"
        "Allow: /\n"
        f"Sitemap: {PUBLIC_SITE_URL}sitemap.xml"
    )


def render_capability_catalog(
    catalog: Dict[str, object], articles: Sequence[Article]
) -> Tuple[str, Dict[str, object]]:
    article_by_slug = {article.slug: article for article in articles}

    def evidence(slug: str) -> Dict[str, str]:
        article = article_by_slug[slug]
        return {
            "slug": article.slug,
            "title": article.title,
            "path": article.public_path,
            "reviewedAt": article.reviewed_at,
        }

    capabilities: List[Dict[str, object]] = []
    for raw_entry in catalog["capabilities"]:
        entry = dict(raw_entry)
        evidence_slugs = entry.pop("evidenceSlugs")
        entry["evidence"] = [evidence(slug) for slug in evidence_slugs]
        capabilities.append(entry)

    highlights: List[Dict[str, object]] = []
    for raw_entry in catalog["performanceHighlights"]:
        entry = dict(raw_entry)
        evidence_slug = entry.pop("evidenceSlug")
        entry["evidence"] = evidence(evidence_slug)
        highlights.append(entry)

    status_definitions = [
        {"id": identifier, "label": label, "description": description}
        for identifier, label, description in CAPABILITY_STATUS_DEFINITIONS
    ]
    public_index: Dict[str, object] = {
        "schemaVersion": 1,
        "project": "fast-mlx",
        "claimBoundary": catalog["claimBoundary"],
        "updatedAt": catalog["updatedAt"],
        "statusDefinitions": status_definitions,
        "capabilities": capabilities,
        "performanceHighlights": highlights,
    }

    status_by_id = {item[0]: item[1] for item in CAPABILITY_STATUS_DEFINITIONS}
    body: List[str] = [
        '<section class="page-hero shell capability-hero">',
        '<p class="eyebrow">Capabilities &amp; evidence</p>',
        '<h1>See what exists—and what each claim actually covers.</h1>',
        '<p class="lede">A generated inventory of fast-mlx source, scoped decisions, and measured results. Every evidence link resolves to a reviewed public note; source presence alone is never treated as production support.</p>',
        '<div class="hero-actions">',
        '<a class="button primary" href="https://github.com/bitworks-io/fast-mlx#first-run" rel="noreferrer">Run the scripted server</a>',
        '<a class="button secondary" href="https://github.com/bitworks-io/fast-mlx" rel="noreferrer">Inspect the source</a>',
        '</div>',
        '</section>',
        '<section class="section shell" aria-labelledby="measured-heading">',
        '<div class="section-heading"><p class="eyebrow">Measured highlights</p><h2 id="measured-heading">Useful numbers, attached to their boundaries.</h2><p class="section-intro">These are dated fast-mlx results—not cross-model, cross-hardware, or future-version guarantees.</p></div>',
        '<div class="evidence-grid">',
    ]
    for highlight in highlights:
        evidence_record = highlight["evidence"]
        href = relative_href("capabilities/index.html", evidence_record["path"])
        body.extend(
            [
                '<article class="evidence-card">',
                f'<div class="card-topline"><span class="status-badge status-{html.escape(highlight["decision"])}">{html.escape(status_by_id[highlight["decision"]].upper())}</span><time datetime="{html.escape(highlight["date"], quote=True)}">{html.escape(highlight["date"])}</time></div>',
                f'<div class="metric">{html.escape(highlight["metric"])}</div>',
                f'<h3>{html.escape(highlight["label"])}</h3>',
                '<dl class="evidence-context">',
                f'<div><dt>Model</dt><dd>{html.escape(highlight["model"])}</dd></div>',
                f'<div><dt>Hardware</dt><dd>{html.escape(highlight["hardware"])}</dd></div>',
                f'<div><dt>Workload</dt><dd>{html.escape(highlight["workload"])}</dd></div>',
                '</dl>',
                f'<p class="scope-note"><strong>Boundary:</strong> {html.escape(highlight["caveat"])}</p>',
                f'<a class="text-link" href="{html.escape(href, quote=True)}">Read the measured note →</a>',
                '</article>',
            ]
        )
    body.extend(
        [
            '</div>',
            '</section>',
            '<section class="section shell" aria-labelledby="inventory-heading">',
            '<div class="section-heading"><p class="eyebrow">Current inventory</p><h2 id="inventory-heading">Capabilities carry state, scope, and evidence.</h2></div>',
            '<ul class="status-legend" aria-label="Capability status definitions">',
        ]
    )
    for definition in status_definitions:
        body.append(
            f'<li><span class="status-badge status-{html.escape(definition["id"])}">{html.escape(definition["label"].upper())}</span><p>{html.escape(definition["description"])}</p></li>'
        )
    body.extend(['</ul>', '<div class="capability-grid">'])
    for capability in capabilities:
        body.extend(
            [
                '<article class="capability-card">',
                f'<span class="status-badge status-{html.escape(capability["status"])}">{html.escape(status_by_id[capability["status"]].upper())}</span>',
                f'<h3>{html.escape(capability["name"])}</h3>',
                f'<p>{html.escape(capability["summary"])}</p>',
                f'<p class="scope-note"><strong>Scope:</strong> {html.escape(capability["scope"])}</p>',
                '<div class="evidence-links" aria-label="Published evidence">',
            ]
        )
        for record in capability["evidence"]:
            href = relative_href("capabilities/index.html", record["path"])
            body.append(
                f'<a href="{html.escape(href, quote=True)}">{html.escape(record["title"])} →</a>'
            )
        body.extend(['</div>', '</article>'])
    body.extend(
        [
            '</div>',
            '</section>',
            '<section class="section shell callout capability-callout" aria-labelledby="machine-heading">',
            '<p class="eyebrow">Machine-readable contract</p>',
            '<h2 id="machine-heading">Agents can inspect the same bounded inventory.</h2>',
            '<p>The generated JSON mirrors this page and includes resolved public evidence paths. The build refuses unknown states, missing scope, and unpublished evidence slugs.</p>',
            '<a class="text-link" href="index.json">Open capabilities/index.json →</a>',
            '</section>',
        ]
    )
    return "\n".join(body), public_index


def render_benchmark_explorer(highlights: Sequence[Dict[str, object]]) -> str:
    status_by_id = {item[0]: item[1] for item in CAPABILITY_STATUS_DEFINITIONS}

    def options(values: Iterable[str], all_label: str) -> str:
        rendered = [f'<option value="">{html.escape(all_label)}</option>']
        for value in sorted(set(values), key=str.casefold):
            escaped = html.escape(value, quote=True)
            rendered.append(f'<option value="{escaped}">{html.escape(value)}</option>')
        return "".join(rendered)

    model_options = options(
        (str(highlight["model"]) for highlight in highlights), "All models"
    )
    hardware_options = options(
        (str(highlight["hardware"]) for highlight in highlights), "All hardware"
    )
    decision_options = ['<option value="">All decisions</option>']
    for decision in sorted(
        {str(highlight["decision"]) for highlight in highlights},
        key=lambda value: status_by_id[value].casefold(),
    ):
        decision_options.append(
            f'<option value="{html.escape(decision, quote=True)}">'
            f'{html.escape(status_by_id[decision])}</option>'
        )

    total = len(highlights)
    body: List[str] = [
        '<section class="page-hero shell benchmark-hero">',
        '<p class="eyebrow">Benchmark explorer</p>',
        '<h1>Compare the result, then read its boundary.</h1>',
        '<p class="lede">Filter fast-mlx’s reviewed measurements by model, hardware, or decision. Every result keeps its workload, date, caveat, and evidence attached; these are not cross-system or future-version guarantees.</p>',
        '<div class="hero-actions">',
        '<a class="button secondary" href="../methodology/">Read the methodology</a>',
        '<a class="button secondary" href="../capabilities/index.json">Inspect the source contract</a>',
        '</div>',
        '</section>',
        '<section class="section shell" aria-labelledby="benchmark-results-heading">',
        '<div class="section-heading"><p class="eyebrow">Reviewed fast-mlx evidence only</p><h2 id="benchmark-results-heading">Scope travels with every number.</h2><p class="section-intro">Filters change only what is visible. They do not recompute, normalize, rank, or combine measurements.</p></div>',
        '<form class="benchmark-controls" data-benchmark-controls aria-label="Filter benchmark evidence" action="./" method="get">',
        '<div><label for="benchmark-model">Model</label><select id="benchmark-model" name="model">'
        + model_options
        + '</select></div>',
        '<div><label for="benchmark-hardware">Hardware</label><select id="benchmark-hardware" name="hardware">'
        + hardware_options
        + '</select></div>',
        '<div><label for="benchmark-decision">Decision</label><select id="benchmark-decision" name="decision">'
        + "".join(decision_options)
        + '</select></div>',
        '<button class="button secondary benchmark-reset" data-benchmark-reset type="reset">Clear filters</button>',
        '</form>',
        '<noscript><p class="benchmark-noscript">JavaScript is optional. All reviewed results remain visible below; interactive filters are unavailable.</p></noscript>',
        f'<p class="benchmark-count" data-benchmark-count aria-live="polite">Showing {total} of {total} reviewed results.</p>',
        '<div class="benchmark-results" data-benchmark-results role="list">',
    ]
    for highlight in sorted(
        highlights, key=lambda entry: str(entry["date"]), reverse=True
    ):
        evidence_record = highlight["evidence"]
        href = relative_href("benchmarks/index.html", evidence_record["path"])
        decision = str(highlight["decision"])
        body.extend(
            [
                '<article class="benchmark-result" role="listitem" '
                f'data-benchmark-card data-highlight-id="{html.escape(str(highlight["id"]), quote=True)}" '
                f'data-model="{html.escape(str(highlight["model"]), quote=True)}" '
                f'data-hardware="{html.escape(str(highlight["hardware"]), quote=True)}" '
                f'data-decision="{html.escape(decision, quote=True)}">',
                '<div class="card-topline">'
                f'<span class="status-badge status-{html.escape(decision, quote=True)}">{html.escape(status_by_id[decision].upper())}</span>'
                f'<time datetime="{html.escape(str(highlight["date"]), quote=True)}">{html.escape(str(highlight["date"]))}</time>'
                '</div>',
                f'<div class="metric">{html.escape(str(highlight["metric"]))}</div>',
                f'<h3>{html.escape(str(highlight["label"]))}</h3>',
                '<dl class="evidence-context">',
                f'<div><dt>Model</dt><dd>{html.escape(str(highlight["model"]))}</dd></div>',
                f'<div><dt>Hardware</dt><dd>{html.escape(str(highlight["hardware"]))}</dd></div>',
                f'<div><dt>Workload</dt><dd>{html.escape(str(highlight["workload"]))}</dd></div>',
                '</dl>',
                f'<p class="scope-note"><strong>Boundary:</strong> {html.escape(str(highlight["caveat"]))}</p>',
                f'<a class="text-link" href="{html.escape(href, quote=True)}">Read {html.escape(str(evidence_record["title"]))} →</a>',
                '</article>',
            ]
        )
    body.extend(
        [
            '</div>',
            '<div class="benchmark-empty" data-benchmark-empty hidden role="status"><h3>No reviewed results match those filters.</h3><p>Clear the filters to return to the complete evidence set.</p><a class="text-link" href="./">Reset the benchmark explorer →</a></div>',
            '</section>',
            '<section class="section shell callout benchmark-callout" aria-labelledby="benchmark-boundary-heading">',
            '<p class="eyebrow">Claim boundary</p>',
            '<h2 id="benchmark-boundary-heading">This explorer never manufactures a comparison.</h2>',
            '<p>It presents only the reviewed entries already published in the fast-mlx capability contract. It performs no unit conversion, interpolation, aggregation, competitor comparison, or live benchmark execution.</p>',
            '<a class="text-link" href="../capabilities/">See the complete capability inventory →</a>',
            '</section>',
        ]
    )
    return "\n".join(body)


def render_home_current_cycle(
    catalog: Dict[str, object],
    articles: Sequence[Article],
    release_catalog: Dict[str, object],
) -> str:
    """Render the home-page snapshot from reviewed public manifests only."""

    capabilities = catalog["capabilities"]
    releases = release_catalog["releases"]
    latest = releases[0]
    boundary = release_catalog["currentBoundary"]
    boundary_evidence = boundary["evidence"]
    status_counts = {
        status: sum(capability["status"] == status for capability in capabilities)
        for status in CAPABILITY_STATUSES
    }
    status_labels = {
        status: label for status, label, _description in CAPABILITY_STATUS_DEFINITIONS
    }
    published_at = str(latest["publishedAt"])
    commit = str(latest["publicCommit"])
    latest_id = str(latest["id"])

    status_summary = "; ".join(
        '<span data-capability-status="'
        + html.escape(status, quote=True)
        + '" data-count="'
        + str(status_counts[status])
        + '">'
        + str(status_counts[status])
        + " "
        + html.escape(status_labels[status].casefold())
        + "</span>"
        for status, _label, _description in CAPABILITY_STATUS_DEFINITIONS
    )

    return "\n".join(
        [
            '<section class="section shell split" aria-labelledby="current-cycle-heading" '
            'data-current-cycle data-latest-release-id="'
            + html.escape(latest_id, quote=True)
            + '" data-boundary-id="'
            + html.escape(str(boundary["id"]), quote=True)
            + '" data-boundary-state="'
            + html.escape(str(boundary["state"]), quote=True)
            + '">',
            '<div>',
            '<p class="eyebrow">Current reviewed cycle</p>',
            '<div class="card-topline"><span class="status-badge status-released">RELEASED</span> '
            f'<time datetime="{html.escape(published_at, quote=True)}">{html.escape(published_at[:10])}</time></div>',
            f'<h2 id="current-cycle-heading">{html.escape(str(latest["title"]))}</h2>',
            f'<p>{html.escape(str(latest["summary"]))}</p>',
            '<p class="scope-note"><strong>Boundary:</strong> '
            + html.escape(str(latest["scope"]))
            + "</p>",
            '<div class="release-links">',
            '<a class="text-link" href="releases/#release-'
            + html.escape(latest_id, quote=True)
            + '">Inspect the latest reviewed milestone →</a>',
            '<a href="https://github.com/bitworks-io/fast-mlx/commit/'
            + html.escape(commit, quote=True)
            + '" rel="noreferrer">Inspect commit '
            + html.escape(commit[:12])
            + " →</a>",
            '</div>',
            '</div>',
            '<div class="capability-list" role="list" aria-label="Current reviewed evidence inventory">',
            '<article role="listitem">',
            f'<h3>{len(capabilities)} reviewed capabilities</h3>',
            f'<p>{status_summary}</p>',
            '<a class="text-link" href="capabilities/">Inspect capability states →</a>',
            '</article>',
            '<article role="listitem">',
            f'<h3>{len(articles)} published research notes</h3>',
            '<p>Dated investigations preserve promoted, shelved, rejected, and diagnostic outcomes.</p>',
            '<a class="text-link" href="research/">Read the evidence trail →</a>',
            '</article>',
            '<article role="listitem">',
            f'<h3>{len(releases)} reviewed releases</h3>',
            '<p>Each public milestone names its exact commit and unchanged claim boundary.</p>',
            '<a class="text-link" href="releases/">Follow the release ledger →</a>',
            '</article>',
            '<article role="listitem">',
            f'<span class="status-badge status-gated">{html.escape(str(boundary["state"]).upper())}</span>',
            f'<h3>{html.escape(str(boundary["label"]))}</h3>',
            f'<p>{html.escape(str(boundary["summary"]))}</p>',
            '<a class="text-link" href="'
            + html.escape(str(boundary_evidence["path"]), quote=True)
            + '">'
            + html.escape(str(boundary_evidence["label"]))
            + " →</a>",
            '</article>',
            '</div>',
            '</section>',
        ]
    )


def build_site(repository_root: Path, output: Path) -> List[Article]:
    articles = load_articles(repository_root)
    catalog = load_capability_catalog(repository_root, {article.slug for article in articles})
    release_catalog = load_release_catalog(repository_root)
    template = (repository_root / "site/templates/base.html").read_text(encoding="utf-8")
    assets = repository_root / "site/assets"
    validate_asset_tree(assets)
    shutil.copytree(assets, output / "assets", dirs_exist_ok=True)

    home = (repository_root / "site/fragments/home.html").read_text(encoding="utf-8")
    if home.count("{{current_cycle}}") != 1:
        fail("site/fragments/home.html must contain exactly one current-cycle slot")
    home = home.replace(
        "{{current_cycle}}",
        render_home_current_cycle(catalog, articles, release_catalog),
    )
    write_page(
        output,
        "index.html",
        render_template(
            template,
            "fast-mlx — evidence-gated MLX inference",
            "A Swift and MLX inference project that continuously researches, tests, and publishes verified capabilities.",
            "",
            home,
            public_path="",
        ),
    )

    for name, title, description in (
        (
            "process",
            "The improvement loop — fast-mlx",
            "How fast-mlx turns research into reviewed, testable inference capabilities.",
        ),
        (
            "methodology",
            "Methodology — fast-mlx",
            "The correctness, comparability, and public-claim boundaries behind fast-mlx results.",
        ),
    ):
        fragment = (repository_root / f"site/fragments/{name}.html").read_text(encoding="utf-8")
        write_page(
            output,
            f"{name}/index.html",
            render_template(
                template,
                title,
                description,
                "../",
                fragment,
                name,
                public_path=f"{name}/",
            ),
        )

    capability_body, capability_index = render_capability_catalog(catalog, articles)
    write_page(
        output,
        "capabilities/index.html",
        render_template(
            template,
            "Capabilities & evidence — fast-mlx",
            "A status-aware inventory of fast-mlx features and scoped measured results.",
            "../",
            capability_body,
            "capabilities",
            public_path="capabilities/",
        ),
    )
    write_page(
        output,
        "capabilities/index.json",
        json.dumps(capability_index, indent=2, ensure_ascii=False),
    )
    benchmark_body = render_benchmark_explorer(
        capability_index["performanceHighlights"]
    )
    write_page(
        output,
        "benchmarks/index.html",
        render_template(
            template,
            "Benchmark explorer — fast-mlx",
            "Filter reviewed fast-mlx measurements without separating results from their scope, caveats, or evidence.",
            "../",
            benchmark_body,
            "benchmarks",
            '<script src="../assets/benchmark-explorer.js" defer></script>',
            public_path="benchmarks/",
        ),
    )
    release_body, release_index = render_release_catalog(release_catalog)
    write_page(
        output,
        "releases/index.html",
        render_template(
            template,
            "Releases — fast-mlx",
            "A reviewed ledger of fast-mlx public milestones, exact commits, shipped surfaces, and unchanged boundaries.",
            "../",
            release_body,
            "releases",
            public_path="releases/",
        ),
    )
    write_page(
        output,
        "releases/index.json",
        json.dumps(release_index, indent=2, ensure_ascii=False),
    )
    write_page(output, "releases/feed.atom", render_release_feed(release_index))

    cards = [
        '<section class="page-hero shell"><p class="eyebrow">Research notes</p>'
        '<h1>What the measurements changed.</h1>'
        '<p class="lede">Dated fast-mlx investigations, including negative results. Each note is a scoped historical result, not a timeless performance guarantee.</p></section>',
        '<section class="section shell"><div class="research-grid">',
    ]
    for article in articles:
        cards.append(
            '<article class="note-card">'
            f'<div class="meta">{html.escape(article.date)} · {html.escape(article.theme)}</div>'
            f'<h2>{html.escape(article.title)}</h2>'
            f'<p>{html.escape(article.summary)}</p>'
            f'<a href="{html.escape(article.slug)}/">Read the note →</a>'
            "</article>"
        )
    cards.append("</div></section>")
    write_page(
        output,
        "research/index.html",
        render_template(
            template,
            "Research notes — fast-mlx",
            "Dated fast-mlx investigations and measured negative results.",
            "../",
            "\n".join(cards),
            "research",
            public_path="research/",
        ),
    )

    article_routes = {Path(article.source_name).name: article.public_path for article in articles}
    for article in articles:
        markdown_html = render_markdown(article.body, article_routes, article.output_file)
        article_body = (
            '<article class="article shell">'
            '<header class="article-header">'
            f'<div class="article-meta">{html.escape(article.date)} · {html.escape(article.theme)}</div>'
            f'<h1>{html.escape(article.title)}</h1>'
            f'<p class="article-summary">{html.escape(article.summary)}</p>'
            "</header>"
            f'<div class="article-body">{markdown_html}</div>'
            "</article>"
        )
        write_page(
            output,
            article.output_file,
            render_template(
                template,
                f"{article.title} — fast-mlx",
                article.summary,
                "../../",
                article_body,
                "research",
                public_path=article.public_path,
                article_section=article.theme,
            ),
        )

    public_index = {
        "schemaVersion": 1,
        "project": "fast-mlx",
        "claimBoundary": "fast-mlx-owned-results-only",
        "articles": [
            {
                "title": article.title,
                "date": article.date,
                "theme": article.theme,
                "summary": article.summary,
                "path": article.public_path,
                "reviewedAt": article.reviewed_at,
            }
            for article in articles
        ],
    }
    write_page(output, "research/index.json", json.dumps(public_index, indent=2, ensure_ascii=False))
    write_page(output, "sitemap.xml", render_sitemap(articles))
    write_page(
        output,
        "llms.txt",
        "# fast-mlx\n\n"
        "Evidence-gated Swift/MLX inference research for Apple Silicon.\n\n"
        "## Core pages\n"
        "- /process/: research-to-publication loop\n"
        "- /methodology/: correctness and claim boundaries\n"
        "- /capabilities/: status-aware feature and evidence inventory\n"
        "- /capabilities/index.json: machine-readable capability contract\n"
        "- /benchmarks/: filterable reviewed performance evidence\n"
        "- /releases/: reviewed public milestones and unchanged boundaries\n"
        "- /releases/index.json: machine-readable release ledger\n"
        "- /releases/feed.atom: Atom feed of reviewed public milestones\n"
        "- /sitemap.xml: reviewed human-facing page inventory\n"
        "- /robots.txt: crawl policy and sitemap location\n"
        "- /research/: dated technical notes\n\n"
        "## Published notes\n"
        + "\n".join(f"- /{article.public_path}: {article.title}" for article in articles),
    )
    write_page(output, "robots.txt", render_robots())
    write_page(
        output,
        "404.html",
        render_template(
            template,
            "Not found — fast-mlx",
            "The requested fast-mlx page was not found.",
            "",
            '<section class="page-hero shell narrow"><p class="eyebrow">404</p><h1>That evidence path is not here.</h1><p class="lede">Return to the <a href="./">fast-mlx home page</a>.</p></section>',
        ),
    )
    (output / ".nojekyll").write_text("", encoding="utf-8")
    return articles


def scan_generated_output(output: Path) -> None:
    validate_social_card(output / SOCIAL_CARD_PATH, "generated social card")
    for path in output.rglob("*"):
        if path.is_symlink():
            fail(f"generated site contains a symlink: {path}")
        if not path.is_file():
            continue
        if path.relative_to(output).as_posix() == SOCIAL_CARD_PATH:
            continue
        text = path.read_text(encoding="utf-8")
        for marker in PRIVATE_MARKERS:
            if marker.casefold() in text.casefold():
                fail(f"generated site contains private marker {marker!r}: {path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_arguments(argv)
    repository_root = arguments.repository_root.resolve()
    output = arguments.output.resolve()
    prepare_output(output, repository_root)
    articles = build_site(repository_root, output)
    scan_generated_output(output)
    print(f"built public site: articles={len(articles)} output={output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
