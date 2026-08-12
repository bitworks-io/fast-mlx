#!/usr/bin/env python3
"""Build the dependency-free fast-mlx GitHub Pages site.

Only articles named in site/publications.json are rendered. Markdown is escaped and handled by a
small project-owned renderer so a Pages build does not execute article HTML or depend on a remote
package registry.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import posixpath
import re
import shutil
import sys
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
    return value.strip()


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
) -> str:
    nav_root = html.escape(root, quote=True)

    def nav_link(page: str, label: str) -> str:
        current = ' aria-current="page"' if current_page == page else ""
        return f'<a href="{nav_root}{page}/"{current}>{label}</a>'

    replacements = {
        "{{title}}": html.escape(title),
        "{{description}}": html.escape(description, quote=True),
        "{{root}}": root,
        "{{body}}": body.replace("{{root}}", root),
        "{{capabilities_nav}}": nav_link("capabilities", "Capabilities"),
        "{{process_nav}}": nav_link("process", "The loop"),
        "{{methodology_nav}}": nav_link("methodology", "Methodology"),
        "{{research_nav}}": nav_link("research", "Research notes"),
    }
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    if re.search(r"{{[a-zA-Z0-9_]+}}", rendered):
        fail(f"template contains an unresolved token for {title}")
    return rendered


def write_page(output: Path, relative_file: str, contents: str) -> None:
    destination = output / relative_file
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(contents.rstrip() + "\n", encoding="utf-8")


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


def build_site(repository_root: Path, output: Path) -> List[Article]:
    articles = load_articles(repository_root)
    catalog = load_capability_catalog(repository_root, {article.slug for article in articles})
    template = (repository_root / "site/templates/base.html").read_text(encoding="utf-8")
    assets = repository_root / "site/assets"
    validate_asset_tree(assets)
    shutil.copytree(assets, output / "assets", dirs_exist_ok=True)

    home = (repository_root / "site/fragments/home.html").read_text(encoding="utf-8")
    write_page(
        output,
        "index.html",
        render_template(
            template,
            "fast-mlx — evidence-gated MLX inference",
            "A Swift and MLX inference project that continuously researches, tests, and publishes verified capabilities.",
            "",
            home,
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
            render_template(template, title, description, "../", fragment, name),
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
        ),
    )
    write_page(
        output,
        "capabilities/index.json",
        json.dumps(capability_index, indent=2, ensure_ascii=False),
    )

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
        "- /research/: dated technical notes\n\n"
        "## Published notes\n"
        + "\n".join(f"- /{article.public_path}: {article.title}" for article in articles),
    )
    write_page(output, "robots.txt", "User-agent: *\nAllow: /\n")
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
    for path in output.rglob("*"):
        if path.is_symlink():
            fail(f"generated site contains a symlink: {path}")
        if not path.is_file():
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
