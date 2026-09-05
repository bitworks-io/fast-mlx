#!/usr/bin/env python3
"""Validate a generated fast-mlx public-repository candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


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

# The internal implementation-family marker below is assembled by string
# concatenation, never written as a literal. This script is itself part of
# the public projection it validates, so a literal occurrence here would
# match its own scan on every future publish. Do not "tidy" this into a
# single literal string.
INTERNAL_FAMILY_MARKER = "Qwen" + "4Exp"

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
    "public/public-repository-public.json",
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
    "public/public-repository.json",
    ".github/workflows/pages.yml",
    ".github/workflows/quality.yml",
    "site/capabilities.json",
    "site/publications.json",
    "site/releases.json",
    "site/fragments/quickstart.html",
    "site/fragments/license.html",
    "scripts/export_public_repository.py",
    "scripts/build_public_site.py",
    "scripts/validate_public_site.py",
    "scripts/verify_public_deployment.py",
    "scripts/validate_public_deployment_receipt.py",
    "scripts/tests/test_public_deployment.py",
    "scripts/tests/test_public_capability_details.py",
    "scripts/tests/test_public_export.py",
    "scripts/tests/test_public_status.py",
    "scripts/tests/benchmark_explorer_node_test.js",
    "site/assets/research-explorer.js",
    "spike/Package.swift",
    "spike/Sources/fastmlx-serve/FastMLXServe.swift",
    "spike/Tests/ServingCoreTests/FastMLXServeArgumentsTests.swift",
    "spike/Tests/HarnessCoreTests/Fixtures/GSM8K-LICENSE",
    "spike/Vendor/mlx-swift-lm/LICENSE",
    "spike/Vendor/mlx-swift-lm/FAST_MLX_UPSTREAM.md",
)


def is_canonical_public_path(value: object) -> bool:
    if not isinstance(value, str):
        return False
    candidate = Path(value)
    return (
        not candidate.is_absolute()
        and ".." not in candidate.parts
        and value not in {"", "."}
        and candidate.as_posix() == value
        and all(part.casefold() != ".git" for part in candidate.parts)
    )


def candidate_index_entries(repository: Path) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    for path in repository.rglob("*"):
        relative = path.relative_to(repository)
        if (
            (relative.parts and relative.parts[0] == ".git")
            or path.is_symlink()
            or not path.is_file()
        ):
            continue
        mode = "100755" if path.stat().st_mode & stat.S_IXUSR else "100644"
        entries[relative.as_posix()] = mode
    return entries


def candidate_index_seal(entries: Dict[str, str]) -> Tuple[int, str]:
    digest = hashlib.sha256()
    for path in sorted(entries):
        digest.update(entries[path].encode("ascii"))
        digest.update(b"\0")
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
    return len(entries), digest.hexdigest()


def validate_public_identity_manifest(repository: Path) -> List[str]:
    failures: List[str] = []
    manifest_path = repository / "public/public-repository.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return failures
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"invalid public identity manifest: {exc}"]
    if not isinstance(manifest, dict):
        return ["public identity manifest must be an object"]
    if manifest.get("schemaVersion") != 1:
        failures.append("public identity manifest must use schemaVersion 1")
    expected_manifest_keys = {"schemaVersion", "publicIndex", "files", "trees"}
    if set(manifest) != expected_manifest_keys:
        failures.append("public identity manifest must use the exact top-level schema")

    public_index = manifest.get("publicIndex")
    if not isinstance(public_index, dict):
        failures.append("public identity manifest requires publicIndex")
    elif set(public_index) != {"pathCount", "pathModeSha256"}:
        failures.append("public identity manifest has an invalid publicIndex schema")
    else:
        path_count = public_index.get("pathCount")
        path_mode_sha256 = public_index.get("pathModeSha256")
        if type(path_count) is not int or path_count < 1:
            failures.append("public identity manifest has an invalid pathCount")
        if (
            not isinstance(path_mode_sha256, str)
            or len(path_mode_sha256) != 64
            or any(character not in "0123456789abcdef" for character in path_mode_sha256)
        ):
            failures.append("public identity manifest has an invalid pathModeSha256")
        if type(path_count) is int and isinstance(path_mode_sha256, str):
            actual_entries = candidate_index_entries(repository)
            actual_count, actual_sha256 = candidate_index_seal(actual_entries)
            if path_count != actual_count or path_mode_sha256 != actual_sha256:
                failures.append(
                    "public identity manifest seal does not match candidate path/mode set"
                )

    covered_paths: Set[str] = set()
    files = manifest.get("files")
    if not isinstance(files, list):
        failures.append("public identity manifest files must be an array")
    else:
        for entry in files:
            if not isinstance(entry, dict) or set(entry) != {"source", "destination"}:
                failures.append("public identity manifest has an invalid file entry")
                continue
            source = entry.get("source")
            destination = entry.get("destination")
            if (
                not is_canonical_public_path(source)
                or source != destination
            ):
                failures.append("public identity manifest file entry must use identity paths")
                continue
            if destination in covered_paths:
                failures.append(f"duplicate public identity destination: {destination}")
            covered_paths.add(destination)

    trees = manifest.get("trees")
    actual_entries = candidate_index_entries(repository)
    if not isinstance(trees, list):
        failures.append("public identity manifest trees must be an array")
    else:
        for entry in trees:
            if not isinstance(entry, dict) or not {"source", "destination"} <= set(entry):
                failures.append("public identity manifest has an invalid tree entry")
                continue
            if set(entry) - {"source", "destination", "exclude"}:
                failures.append("public identity manifest has an invalid tree entry")
                continue
            source = entry.get("source")
            destination = entry.get("destination")
            excludes = entry.get("exclude", [])
            if (
                not is_canonical_public_path(source)
                or source != destination
                or not isinstance(excludes, list)
                or not all(is_canonical_public_path(value) for value in excludes)
            ):
                failures.append("public identity manifest tree entry must use identity paths")
                continue
            prefix = source.rstrip("/") + "/"
            normalized_excludes = [value.rstrip("/") for value in excludes]
            matches = []
            for path in sorted(value for value in actual_entries if value.startswith(prefix)):
                relative = path[len(prefix) :]
                if any(
                    relative == excluded or relative.startswith(excluded + "/")
                    for excluded in normalized_excludes
                ):
                    continue
                matches.append(path)
            if not matches:
                failures.append(f"public identity tree has no candidate files: {source}")
            for path in matches:
                if path in covered_paths:
                    failures.append(f"duplicate public identity destination: {path}")
                covered_paths.add(path)

    publications_path = repository / "site/publications.json"
    if publications_path.is_file() and not publications_path.is_symlink():
        try:
            publications = json.loads(publications_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            failures.append(f"invalid site/publications.json: {exc}")
            publications = None
        if not isinstance(publications, dict):
            if publications is not None:
                failures.append("site/publications.json must be an object")
        else:
            articles = publications.get("articles")
            if not isinstance(articles, list) or not articles:
                failures.append("site/publications.json has no articles")
                articles = []
            seen_article_sources: Set[str] = set()
            for article in articles:
                source = article.get("source") if isinstance(article, dict) else None
                if not is_canonical_public_path(source):
                    failures.append(f"invalid published article source: {source!r}")
                    continue
                if source in seen_article_sources:
                    failures.append(
                        f"duplicate published article source: {source}"
                    )
                seen_article_sources.add(source)
                if source in covered_paths:
                    failures.append(
                        f"published article overlaps public identity manifest: {source}"
                    )
                covered_paths.add(source)

    if covered_paths != set(actual_entries):
        failures.append(
            "public identity manifest does not cover the exact candidate path set"
        )
    return failures


def validate_no_internal_family_marker(repository: Path) -> List[str]:
    failures: List[str] = []
    marker_bytes = INTERNAL_FAMILY_MARKER.encode("utf-8")
    for path in repository.rglob("*"):
        relative = path.relative_to(repository)
        if relative.parts and relative.parts[0] == ".git":
            continue
        relative_text = relative.as_posix()
        if INTERNAL_FAMILY_MARKER in relative_text:
            failures.append(
                "projected path contains an internal implementation-family "
                f"marker: {relative}"
            )
        if path.is_symlink() or not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if marker_bytes in data:
            failures.append(
                f"projected file contains an internal implementation-family marker: {relative}"
            )
    return failures


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

    failures.extend(validate_public_identity_manifest(repository))
    failures.extend(validate_no_internal_family_marker(repository))

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
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            failure = f"invalid site/publications.json: {exc}"
            if failure not in failures:
                failures.append(failure)
        else:
            if (
                isinstance(publications, dict)
                and publications.get("schemaVersion") != 1
            ):
                failures.append("site/publications.json must use schemaVersion 1")
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
        relative = path.relative_to(repository)
        if relative.parts and relative.parts[0] == ".git":
            continue
        relative_text = relative.as_posix()
        if any(part.casefold() == ".git" for part in relative.parts):
            failures.append(f"nested Git metadata is forbidden: {relative}")
        for marker in PRIVATE_MARKERS:
            path_marker = marker.lstrip("/")
            if path_marker and path_marker.casefold() in relative_text.casefold():
                failures.append(
                    f"private marker {marker!r} in candidate path {relative}"
                )
        if path.is_symlink():
            failures.append(f"symlink is forbidden: {relative}")
            continue
        if not path.is_file():
            if not path.is_dir():
                failures.append(
                    f"special file is forbidden: {relative}"
                )
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for marker in PRIVATE_MARKERS:
            if marker.casefold() in text.casefold():
                failures.append(
                    f"private marker {marker!r} in {relative}"
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
