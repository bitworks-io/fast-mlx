#!/usr/bin/env python3
"""Validate generated fast-mlx Pages files and internal links."""

from __future__ import annotations

import argparse
import html.parser
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
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


def validate(site: Path) -> List[str]:
    failures: List[str] = []
    site = site.resolve()
    required = [
        "index.html",
        "process/index.html",
        "methodology/index.html",
        "research/index.html",
        "research/index.json",
        "assets/site.css",
        "assets/favicon.svg",
        "llms.txt",
        ".nojekyll",
    ]
    for relative in required:
        if not (site / relative).is_file():
            failures.append(f"missing required file: {relative}")

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
