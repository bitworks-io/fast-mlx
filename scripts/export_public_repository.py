#!/usr/bin/env python3
"""Export a committed, allowlisted fast-mlx public-repository candidate.

The exporter copies only paths present in Git's index and in public/public-repository.json, plus
the articles explicitly approved by site/publications.json. The development allowlist exports a
public-ready identity-path manifest at that canonical path, so a public checkout can enforce an
exact index. The exporter never initializes Git, deletes an existing tree, commits, or pushes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple


DEVELOPMENT_PUBLIC_MANIFEST_PATH = "public/public-repository-public.json"
FORBIDDEN_PUBLIC_DESTINATION_PREFIXES: Tuple[str, ...] = (
    ".agents",
    ".claude",
    ".codex",
    ".harness-sha",
    "docs/agent-handoff.md",
    "docs/verification-evidence.md",
    "docs/research",
    "docs/superpowers",
    "docs/task-inbox",
    "public/public-repository-public.json",
    "spike/scripts",
    "spike/Vendor/mlx-swift-lm/.github",
)


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"public-repository export refused: {message}")


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="committed engineering or public checkout root",
    )
    parser.add_argument(
        "--development-projection",
        action="store_true",
        help=(
            "allow the broader development manifest to remap reviewed sources; "
            "never use for a public checkout"
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="absent or empty destination")
    return parser.parse_args(argv)


def decode_object(contents: bytes, path: str) -> Dict[str, object]:
    try:
        value = json.loads(contents.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"manifest must contain a JSON object: {path}")
    return value


def tracked_entries(repository_root: Path) -> Dict[str, str]:
    result = subprocess.run(
        ["git", "ls-files", "--stage", "-z"],
        cwd=str(repository_root),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        fail(f"git ls-files failed: {result.stderr.decode('utf-8', errors='replace').strip()}")
    entries: Dict[str, str] = {}
    for item in result.stdout.split(b"\0"):
        if not item:
            continue
        try:
            metadata, raw_path = item.split(b"\t", 1)
            mode, _object_id, stage = metadata.split(b" ", 2)
            path = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            fail(f"cannot parse git index entry: {exc}")
        if stage != b"0":
            fail(f"unmerged Git index entry is not publishable: {path}")
        entries[path] = mode.decode("ascii")
    return entries


def index_blob(repository_root: Path, source_name: str) -> bytes:
    result = subprocess.run(
        ["git", "cat-file", "blob", f":{source_name}"],
        cwd=str(repository_root),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        fail(
            f"cannot read indexed source {source_name}: "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return result.stdout


def validate_relative(path: str, field: str) -> None:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts or path in {"", "."}:
        fail(f"{field} is not a safe relative path: {path!r}")
    if candidate.as_posix() != path:
        fail(f"{field} is not a canonical relative path: {path!r}")
    if any(part.casefold() == ".git" for part in candidate.parts):
        fail(f"{field} cannot target Git metadata: {path!r}")


def validate_public_destination(path: str, field: str) -> None:
    validate_relative(path, field)
    path_casefold = path.casefold()
    for prefix in FORBIDDEN_PUBLIC_DESTINATION_PREFIXES:
        prefix_casefold = prefix.casefold()
        if path_casefold == prefix_casefold or path_casefold.startswith(
            prefix_casefold + "/"
        ):
            fail(f"{field} targets a private public path: {path!r}")


def validate_public_source(
    path: str,
    field: str,
    *,
    allow_development_manifest_source: bool = False,
) -> None:
    validate_relative(path, field)
    if allow_development_manifest_source and path == DEVELOPMENT_PUBLIC_MANIFEST_PATH:
        return
    path_casefold = path.casefold()
    for prefix in FORBIDDEN_PUBLIC_DESTINATION_PREFIXES:
        prefix_casefold = prefix.casefold()
        if path_casefold == prefix_casefold or path_casefold.startswith(
            prefix_casefold + "/"
        ):
            fail(f"{field} targets a private public source: {path!r}")


def public_index_seal(tracked: Mapping[str, str]) -> Tuple[int, str]:
    digest = hashlib.sha256()
    for path in sorted(tracked):
        digest.update(tracked[path].encode("ascii"))
        digest.update(b"\0")
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
    return len(tracked), digest.hexdigest()


def prepare_output(output: Path, repository_root: Path) -> None:
    resolved_output = output.resolve()
    resolved_root = repository_root.resolve()
    if resolved_output == resolved_root or resolved_root in resolved_output.parents:
        fail("output cannot be inside the source checkout")
    if output.exists():
        if not output.is_dir():
            fail(f"output exists and is not a directory: {output}")
        if any(output.iterdir()):
            fail(f"output directory is not empty: {output}")
    else:
        output.mkdir(parents=True)


def copy_file(
    repository_root: Path,
    output: Path,
    source_name: str,
    destination_name: str,
    entries: Mapping[str, str],
) -> None:
    destination = output / destination_name
    mode = entries.get(source_name)
    if mode not in {"100644", "100755"}:
        fail(f"indexed source is not a regular file: {source_name} (mode {mode!r})")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(index_blob(repository_root, source_name))
    destination.chmod(0o755 if mode == "100755" else 0o644)


def manifest_pairs(
    manifest: Dict[str, object],
    tracked: Mapping[str, str],
    additional_public_paths: Sequence[str] = (),
    allow_remapped_manifest: bool = False,
) -> List[Tuple[str, str]]:
    if manifest.get("schemaVersion") != 1:
        fail("public/public-repository.json must use schemaVersion 1")
    extra_manifest_keys = set(manifest) - {
        "schemaVersion",
        "publicIndex",
        "files",
        "trees",
    }
    if extra_manifest_keys:
        fail(f"public-repository manifest has unknown keys: {sorted(extra_manifest_keys)}")
    pairs: List[Tuple[str, str]] = []
    seen_destinations: Set[str] = set()
    identity_manifest = True

    files = manifest.get("files")
    trees = manifest.get("trees")
    if not isinstance(files, list) or not isinstance(trees, list):
        fail("public-repository manifest requires files and trees arrays")

    for entry in files:
        if not isinstance(entry, dict):
            fail("file manifest entry is not an object")
        extra_keys = set(entry) - {"source", "destination"}
        if extra_keys:
            fail(f"file manifest entry has unknown keys: {sorted(extra_keys)}")
        source = entry.get("source")
        destination = entry.get("destination")
        if not isinstance(source, str) or not isinstance(destination, str):
            fail("file manifest entry needs string source and destination")
        validate_public_destination(destination, "destination")
        validate_public_source(
            source,
            "source",
            allow_development_manifest_source=(
                source == DEVELOPMENT_PUBLIC_MANIFEST_PATH
                and destination == "public/public-repository.json"
            ),
        )
        if source not in tracked:
            fail(f"allowlisted file is not present in Git's index: {source}")
        if source != destination:
            identity_manifest = False
        if destination in seen_destinations:
            fail(f"duplicate public destination: {destination}")
        seen_destinations.add(destination)
        pairs.append((source, destination))

    for entry in trees:
        if not isinstance(entry, dict):
            fail("tree manifest entry is not an object")
        extra_keys = set(entry) - {"source", "destination", "exclude"}
        if extra_keys:
            fail(f"tree manifest entry has unknown keys: {sorted(extra_keys)}")
        source_root = entry.get("source")
        destination_root = entry.get("destination")
        raw_excludes = entry.get("exclude", [])
        if not isinstance(source_root, str) or not isinstance(destination_root, str):
            fail("tree manifest entry needs string source and destination")
        if not isinstance(raw_excludes, list) or not all(
            isinstance(value, str) for value in raw_excludes
        ):
            fail("tree manifest exclude must be an array of strings")
        validate_public_source(source_root, "tree source")
        validate_public_destination(destination_root, "tree destination")
        if source_root != destination_root:
            identity_manifest = False
        excludes = []
        for value in raw_excludes:
            validate_relative(value, "tree exclude")
            excludes.append(value.rstrip("/"))
        prefix = source_root.rstrip("/") + "/"
        matches = []
        for path in sorted(path for path in tracked if path.startswith(prefix)):
            relative = path[len(prefix) :]
            if any(
                relative == excluded or relative.startswith(excluded + "/")
                for excluded in excludes
            ):
                continue
            matches.append(path)
        if not matches:
            fail(f"allowlisted tree has no tracked files: {source_root}")
        for source in matches:
            validate_public_source(source, "tree source member")
            relative = source[len(prefix) :]
            destination = str(Path(destination_root) / relative)
            validate_public_destination(destination, "tree destination member")
            if destination in seen_destinations:
                fail(f"duplicate public destination: {destination}")
            seen_destinations.add(destination)
            pairs.append((source, destination))

    public_index = manifest.get("publicIndex")
    if not identity_manifest:
        if public_index is not None:
            fail("remapped development manifest cannot declare publicIndex")
        if not allow_remapped_manifest:
            fail("public checkout manifest must use identity paths")
    else:
        if not isinstance(public_index, dict):
            fail("identity-path manifest requires publicIndex")
        extra_index_keys = set(public_index) - {"pathCount", "pathModeSha256"}
        if extra_index_keys:
            fail(f"public index seal has unknown keys: {sorted(extra_index_keys)}")
        path_count = public_index.get("pathCount")
        path_mode_sha256 = public_index.get("pathModeSha256")
        if type(path_count) is not int or path_count < 1:
            fail("public index seal pathCount must be a positive integer")
        if (
            not isinstance(path_mode_sha256, str)
            or len(path_mode_sha256) != 64
            or any(character not in "0123456789abcdef" for character in path_mode_sha256)
        ):
            fail("public index seal pathModeSha256 must be lowercase SHA-256")
        actual_count, actual_sha256 = public_index_seal(tracked)
        if path_count != actual_count or path_mode_sha256 != actual_sha256:
            fail(
                "public index seal mismatch "
                f"(expected pathCount={path_count} pathModeSha256={path_mode_sha256}; "
                f"found pathCount={actual_count} pathModeSha256={actual_sha256})"
            )
        expected_public_paths = seen_destinations | set(additional_public_paths)
        if set(tracked) != expected_public_paths:
            fail(
                "identity-path manifest requires an exact public checkout "
                f"index (expected {len(expected_public_paths)} paths, found {len(tracked)})"
            )
    return pairs


def article_pairs(
    manifest: Dict[str, object], tracked: Mapping[str, str]
) -> List[Tuple[str, str]]:
    if manifest.get("schemaVersion") != 1:
        fail("site/publications.json must use schemaVersion 1")
    entries = manifest.get("articles")
    if not isinstance(entries, list) or not entries:
        fail("site/publications.json has no articles")
    pairs: List[Tuple[str, str]] = []
    for entry in entries:
        source = entry.get("source") if isinstance(entry, dict) else None
        if not isinstance(source, str):
            fail(f"invalid published article source: {source!r}")
        validate_relative(source, "published article source")
        source_path = Path(source)
        if source_path.parent != Path("docs/content") or source_path.suffix != ".md":
            fail(f"invalid published article source: {source!r}")
        if source not in tracked:
            fail(f"published article is not present in Git's index: {source}")
        pairs.append((source, source))
    return pairs


def export(
    repository_root: Path,
    output: Path,
    allow_development_manifest: bool = False,
) -> int:
    tracked = tracked_entries(repository_root)
    if (
        allow_development_manifest
        and DEVELOPMENT_PUBLIC_MANIFEST_PATH not in tracked
    ):
        fail(
            "--development-projection requires indexed "
            f"{DEVELOPMENT_PUBLIC_MANIFEST_PATH}"
        )
    manifest_path = "public/public-repository.json"
    publications_path = "site/publications.json"
    manifest = decode_object(index_blob(repository_root, manifest_path), manifest_path)
    publications = decode_object(
        index_blob(repository_root, publications_path), publications_path
    )
    articles = article_pairs(publications, tracked)
    pairs = manifest_pairs(
        manifest,
        tracked,
        additional_public_paths=tuple(destination for _source, destination in articles),
        allow_remapped_manifest=allow_development_manifest,
    ) + articles
    destinations: Set[str] = set()
    for source, destination in pairs:
        if destination in destinations:
            fail(f"duplicate destination across public manifests: {destination}")
        destinations.add(destination)
        mode = tracked.get(source)
        if mode not in {"100644", "100755"}:
            fail(f"indexed source is not a regular file: {source} (mode {mode!r})")
    for source, destination in pairs:
        copy_file(repository_root, output, source, destination, tracked)
    print(f"exported public repository candidate: files={len(pairs)} output={output.resolve()}")
    return len(pairs)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_arguments(argv)
    repository_root = arguments.repository_root.resolve()
    output = arguments.output.resolve()
    prepare_output(output, repository_root)
    export(
        repository_root,
        output,
        allow_development_manifest=arguments.development_projection,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
