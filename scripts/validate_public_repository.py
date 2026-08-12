#!/usr/bin/env python3
"""Validate a generated fast-mlx public-repository candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


PRIVATE_MARKERS: Tuple[str, ...] = (
    "/" + "Users/",
    "/" + "private/",
    "192" + ".168.",
    "llm" + "bench",
    "passwordless" + " sudo",
    "docs/" + "superpowers" + "/",
    "spike/" + "scripts" + "/",
    "BEGIN OPENSSH" + " PRIVATE KEY",
    "BEGIN RSA" + " PRIVATE KEY",
)

APACHE_2_LICENSE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
REQUIRED_NOTICE_LINES: Tuple[str, ...] = (
    "fast-mlx",
    "Copyright 2026 bitworks-io",
)

FORBIDDEN_PATHS: Tuple[str, ...] = (
    ".agents",
    ".claude",
    ".codex",
    ".harness-sha",
    "docs/agent-handoff.md",
    "docs/verification-evidence.md",
    "docs/research",
    "docs/task-inbox",
    "spike" + "/scripts",
    "spike/Vendor/mlx-swift-lm/.github",
)

REQUIRED_PATHS: Tuple[str, ...] = (
    "LICENSE",
    "NOTICE",
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "PUBLICATION.md",
    ".github/workflows/pages.yml",
    ".github/workflows/quality.yml",
    "site/capabilities.json",
    "site/publications.json",
    "site/releases.json",
    "scripts/build_public_site.py",
    "scripts/validate_public_site.py",
    "scripts/tests/benchmark_explorer_node_test.js",
    "spike/Package.swift",
    "spike/Sources/fastmlx-serve/FastMLXServe.swift",
    "spike/Tests/ServingCoreTests/FastMLXServeArgumentsTests.swift",
    "spike/Tests/HarnessCoreTests/Fixtures/GSM8K-LICENSE",
    "spike/Vendor/mlx-swift-lm/LICENSE",
    "spike/Vendor/mlx-swift-lm/FAST_MLX_UPSTREAM.md",
)


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path, help="exported candidate root")
    return parser.parse_args(argv)


def validate(repository: Path) -> List[str]:
    failures: List[str] = []
    repository = repository.resolve()
    for required in REQUIRED_PATHS:
        if not (repository / required).is_file():
            failures.append(f"missing required public file: {required}")

    license_path = repository / "LICENSE"
    if license_path.is_file():
        license_sha256 = hashlib.sha256(license_path.read_bytes()).hexdigest()
        if license_sha256 != APACHE_2_LICENSE_SHA256:
            failures.append("LICENSE does not match the pinned Apache-2.0 text")

    notice_path = repository / "NOTICE"
    if notice_path.is_file():
        notice_lines = set(notice_path.read_text(encoding="utf-8").splitlines())
        for required_line in REQUIRED_NOTICE_LINES:
            if required_line not in notice_lines:
                label = "project" if required_line == "fast-mlx" else "copyright"
                failures.append(
                    f"NOTICE is missing required {label} identity: {required_line}"
                )

    for forbidden in FORBIDDEN_PATHS:
        if (repository / forbidden).exists():
            failures.append(f"forbidden public path exists: {forbidden}")

    publications_path = repository / "site/publications.json"
    approved_sources: set[str] = set()
    if publications_path.is_file():
        try:
            publications = json.loads(publications_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"invalid site/publications.json: {exc}")
        else:
            entries = publications.get("articles", []) if isinstance(publications, dict) else []
            for entry in entries if isinstance(entries, list) else []:
                source = entry.get("source") if isinstance(entry, dict) else None
                if isinstance(source, str):
                    source_path = Path(source)
                    if (
                        source_path.is_absolute()
                        or ".." in source_path.parts
                        or source_path.parent != Path("docs/content")
                        or source_path.suffix != ".md"
                    ):
                        failures.append(f"invalid published article source: {source!r}")
                    else:
                        approved_sources.add(source)
            exported_articles = {
                str(path.relative_to(repository))
                for path in (repository / "docs/content").glob("20*.md")
            }
            if exported_articles != approved_sources:
                failures.append(
                    "exported article set does not exactly match site/publications.json"
                )

    for path in repository.rglob("*"):
        if ".git" in path.relative_to(repository).parts:
            continue
        if path.is_symlink():
            failures.append(f"symlink is forbidden: {path.relative_to(repository)}")
            continue
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for marker in PRIVATE_MARKERS:
            if marker.casefold() in text.casefold():
                failures.append(
                    f"private marker {marker!r} in {path.relative_to(repository)}"
                )
    return failures


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_arguments(argv)
    if not arguments.repository.is_dir():
        print(f"public-repository validation refused: not a directory: {arguments.repository}", file=sys.stderr)
        return 2
    failures = validate(arguments.repository)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(f"validated public repository: {arguments.repository.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
