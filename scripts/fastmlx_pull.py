#!/usr/bin/env python3
"""``fastmlx pull``: a verified, resumable, pinned Hugging Face snapshot pull.

This is a supervisor around ``hf_pinned_snapshot_download.py`` (the reviewed
pinned-snapshot downloader). That downloader is, by its own design, a
single-shot tool with no resume: on failure it leaves its staging tree
behind under a unique ``.acquiring-*`` name and never looks at it again. All
the resume behavior lives here, one layer up:

- The pinned reference must be ``<repo>/<name>@<40-char-lowercase-hex-sha>``.
  Branch/tag names (``main``), short shas, and uppercase hex are all
  refused with a stated reason before any network request is made.
- A free-space floor is checked before the first attempt, using the
  revision API's total planned size (or an explicit ``--min-free-bytes``
  floor) plus the downloader's own safety margin.
- The downloader is invoked up to ``--max-attempts`` times. After a failed
  attempt, the next attempt is pointed at the failed attempt's preserved
  staging tree via ``--reuse-verified-from`` so files already downloaded
  and verified are moved instead of re-fetched. No staging tree is ever
  deleted by this script.
- On success, ``<dest>.pull-receipt.json`` is written recording the repo,
  revision, per-file sha256 and size, attempt count, how much was reused
  from a previous attempt, and the sha256 of the downloader script itself
  (so the receipt is pinned to the code that produced it). An existing
  receipt at that path is never overwritten.

``--adopt`` (2026-09-19): a second way to end up with a receipt, for a
directory that was staged by hand (rsync, a copy from another host, etc.)
rather than by this script. Given the same ``<repo>@<revision>`` pinned
reference and ``--dest`` pointing at an EXISTING directory, it fetches the
same revision-API file manifest ``pull()`` does (``downloader.fetch_api`` /
``downloader.validated_entries`` -- no separate HTTP path), verifies every
manifest file is present as a real regular file of the exact size and
content identity the manifest declares (LFS files by streamed sha256,
non-LFS files by the same git-blob-sha1 identity ``downloader.hash_file``
already checks for a fresh download or a resumed one), and refuses with no
receipt written on the first mismatch. ``dest`` is then walked recursively:
any regular file, at any depth, whose relative path is not an exact
manifest entry is a refusal (it could be loaded even though it was never
verified -- a pinned revision's manifest can vouch only for the bytes it
actually names, so a receipt must never vouch for more than that). An
unlisted symlink, or any other non-regular path, is refused the same way.
The only paths this walk does not refuse on are directories themselves and
anything whose relative path has a component starting with ``.`` (a
``.cache/`` directory, ``.gitattributes``, a dotfile, ...) -- those are
never descended into and are instead recorded in the receipt's
``ignored_local_paths``. The receipt this writes has the same shape and the
same exclusive, never-overwritten write as a normal pull, plus
``"acquisition": "adopted"`` (a normal ``pull()`` receipt now carries
``"acquisition": "downloaded"`` for the same reason).

This script never executes or imports anything from the pulled repository,
and it does not add any new token/credential handling.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Callable, Optional


_DOWNLOADER_PATH = Path(__file__).resolve().parent / "hf_pinned_snapshot_download.py"
_DOWNLOADER_SPEC = importlib.util.spec_from_file_location(
    "hf_pinned_snapshot_download", _DOWNLOADER_PATH
)
assert _DOWNLOADER_SPEC is not None and _DOWNLOADER_SPEC.loader is not None
downloader = importlib.util.module_from_spec(_DOWNLOADER_SPEC)
_DOWNLOADER_SPEC.loader.exec_module(downloader)


DEFAULT_MAX_ATTEMPTS = 3
_UPPERCASE_HEX_40 = re.compile(r"[0-9A-Fa-f]{40}")
_RECEIPT_HASH_CHUNK_BYTES = 8 << 20


def _stream_sha256(path: Path) -> str:
    """sha256 of ``path``, read in chunks so multi-GB shards never load whole.

    ``downloader.hash_file`` is not reused here: it returns a git blob
    sha1 for non-LFS entries (and only sha256 for LFS entries), whereas
    the receipt always records a plain file sha256 regardless of entry
    kind.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(_RECEIPT_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PinnedReferenceError(Exception):
    """The ``<repo>@<sha>`` command-line argument is not acceptable."""


class PullError(Exception):
    """The pull could not be completed."""


# ---------------------------------------------------------------------
# 1. Pinned reference parsing/validation
# ---------------------------------------------------------------------
def validate_pinned_reference(text: str) -> tuple[str, str]:
    """Parse ``<repo>@<sha>``, refusing anything that is not pinned.

    Only a 40-character lowercase hex commit sha is accepted as the
    revision. Branch/tag names (``main``), short shas, and uppercase hex
    are all refused with a stated reason.
    """
    if text.count("@") != 1:
        raise PinnedReferenceError(
            f"pinned reference {text!r} must contain exactly one '@' "
            "separating the repository id from a 40-character lowercase "
            "hex commit sha, e.g. 'org/name@" + "a" * 40 + "'"
        )
    repo_id, _, revision = text.partition("@")
    if downloader.REPO_PATTERN.fullmatch(repo_id) is None:
        raise PinnedReferenceError(
            f"pinned reference {text!r} has an invalid repository id "
            f"{repo_id!r}; expected '<org>/<name>'"
        )
    if downloader.LOWER_HEX_40.fullmatch(revision) is None:
        if len(revision) == 40 and _UPPERCASE_HEX_40.fullmatch(revision):
            raise PinnedReferenceError(
                f"pinned reference sha {revision!r} is 40 hex characters "
                "but is not all lowercase; fastmlx pull requires the "
                "lowercase commit sha form"
            )
        raise PinnedReferenceError(
            f"pinned reference sha {revision!r} must be a 40-character "
            "lowercase hex commit sha; branch and tag names such as "
            "'main' are refused, and abbreviated shas are refused"
        )
    return repo_id, revision


# ---------------------------------------------------------------------
# 2. Supervisor: attempts + resume via --reuse-verified-from
# ---------------------------------------------------------------------
def _acquiring_staging_dirs(dest: Path) -> list[Path]:
    return sorted(
        (path for path in dest.parent.glob(f".{dest.name}.acquiring-*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )


def _newest_staging_dir(dest: Path) -> Optional[Path]:
    staging_dirs = _acquiring_staging_dirs(dest)
    return staging_dirs[-1] if staging_dirs else None


def manifest_path_for(dest: Path) -> Path:
    return dest.parent / f"{dest.name}.source-api-manifest.json"


def receipt_path_for(dest: Path) -> Path:
    return dest.parent / f"{dest.name}.pull-receipt.json"


def downloader_script_sha256() -> str:
    return hashlib.sha256(_DOWNLOADER_PATH.read_bytes()).hexdigest()


def _snapshot_regular_files(root: Path) -> dict[str, tuple[int, int]]:
    """``{relative name: (st_dev, st_ino)}`` of every regular file under
    ``root``, taken *before* an attempt that will reuse from it.

    v5's reuse moves a verified candidate out of ``root`` (``os.rename``),
    so by the time an attempt that reused from ``root`` has finished, the
    candidate no longer exists there to compare against. The identity it
    would have moved with is recorded here, up front, and matched against
    the published file afterward instead.
    """
    snapshot: dict[str, tuple[int, int]] = {}
    for path in root.rglob("*"):
        try:
            info = path.lstat()
        except OSError:
            continue
        if not stat.S_ISREG(info.st_mode):
            continue
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        snapshot[relative] = (info.st_dev, info.st_ino)
    return snapshot


def pull(
    repo_id: str,
    revision: str,
    dest: Path,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    min_free_bytes: Optional[int] = None,
    disk_usage_bytes: Optional[Callable[[Path], int]] = None,
) -> Path:
    """Pull one pinned snapshot into ``dest``, resuming across attempts.

    ``disk_usage_bytes``, when given, overrides the free-space probe used
    for the preflight floor check (tests inject a fake here). When not
    given, ``downloader.probe_free_space_bytes`` is looked up at call time
    (not bound at import time) so a test that patches that attribute on the
    downloader module is honored even without passing this parameter.
    """
    if max_attempts < 1:
        raise PullError(f"--max-attempts must be >= 1, got {max_attempts}")

    dest = dest.expanduser().absolute()
    receipt_path = receipt_path_for(dest)
    if receipt_path.exists() or receipt_path.is_symlink():
        raise PullError(
            f"refusing to overwrite an existing pull receipt: {receipt_path}"
        )

    document, _api_data = downloader.fetch_api(repo_id, revision)
    entries = downloader.validated_entries(document, repo_id, revision)
    total_bytes = sum(entry["size"] for entry in entries)

    required_free_bytes = (
        min_free_bytes
        if min_free_bytes is not None
        else int(total_bytes * downloader.FREE_SPACE_SAFETY_MULTIPLIER)
    )
    probe = disk_usage_bytes if disk_usage_bytes is not None else downloader.probe_free_space_bytes
    available_free_bytes = probe(dest.parent)
    if available_free_bytes < required_free_bytes:
        raise PullError(
            "insufficient free space at "
            f"{dest.parent}: need >= {required_free_bytes} bytes, "
            f"have {available_free_bytes} bytes"
        )

    manifest_path = manifest_path_for(dest)
    # A staging tree can survive a killed *process* (SIGKILL, laptop
    # sleep), not just a failed in-process attempt: it is never deleted by
    # this script. If one is already sitting beside dest when a fresh
    # pull() call starts, point attempt 1 at the newest one so it resumes
    # instead of re-downloading everything. This is safe because reuse is
    # always hash-verified per file (reusable_verified_candidate), so a
    # stale or corrupt leftover file is silently ignored, not trusted.
    reuse_from: Optional[Path] = _newest_staging_dir(dest)
    if reuse_from is not None:
        print(
            f"pull found a preserved staging tree from an earlier process: "
            f"{reuse_from}",
            flush=True,
        )
    last_error: Optional[BaseException] = None
    attempts_used = 0
    # Snapshotted immediately before the attempt that becomes the
    # *successful* one, since reuse moves files out of `reuse_from` -- by
    # the time acquire() returns, a moved candidate no longer exists there
    # to compare against. Retaken every iteration; only the value from the
    # final (successful) iteration is used below.
    reuse_snapshot: dict[str, tuple[int, int]] = {}

    for attempt_number in range(1, max_attempts + 1):
        attempts_used = attempt_number
        reuse_snapshot = (
            _snapshot_regular_files(reuse_from) if reuse_from is not None else {}
        )
        namespace = argparse.Namespace(
            repo_id=repo_id,
            revision=revision,
            output=dest,
            source_api_manifest=manifest_path,
            plan_only=False,
            include_prefix=None,
            reuse_verified_from=reuse_from,
        )
        print(
            f"pull attempt={attempt_number}/{max_attempts} repo={repo_id} "
            f"revision={revision} reuse_from={reuse_from or 'NONE'}",
            flush=True,
        )
        try:
            downloader.acquire(namespace)
            last_error = None
            break
        except downloader.AcquisitionError as error:
            last_error = error
            # Never delete a staging tree: every failed attempt's tree
            # stays on disk, and the next attempt is pointed at the
            # newest one so it can reuse whatever was already verified.
            reuse_from = _newest_staging_dir(dest)
            print(
                f"pull attempt={attempt_number}/{max_attempts} failed: {error}",
                flush=True,
            )

    if last_error is not None:
        raise PullError(
            f"pull failed after {attempts_used} attempt(s) of {max_attempts}: "
            f"{last_error}"
        )

    files_receipt: dict[str, dict[str, object]] = {}
    reused_files = 0
    reused_bytes = 0
    for entry in entries:
        name = entry["name"]
        file_path = dest / name
        files_receipt[name] = {
            "sha256": _stream_sha256(file_path),
            "size": entry["size"],
        }
        snapshot_identity = reuse_snapshot.get(name)
        if snapshot_identity is not None:
            try:
                dest_stat = file_path.lstat()
            except OSError:
                pass
            else:
                if (dest_stat.st_dev, dest_stat.st_ino) == snapshot_identity:
                    reused_files += 1
                    reused_bytes += entry["size"]

    receipt = {
        "format_version": 1,
        "repo_id": repo_id,
        "revision": revision,
        "dest": str(dest),
        "attempts": attempts_used,
        "max_attempts": max_attempts,
        "total_files": len(entries),
        "total_bytes": total_bytes,
        "reused_files": reused_files,
        "reused_bytes": reused_bytes,
        "downloader_script_sha256": downloader_script_sha256(),
        "files": files_receipt,
        "acquisition": "downloaded",
    }
    receipt_bytes = json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n"
    downloader.write_exclusive(receipt_path, receipt_bytes)
    print(f"pull complete: {receipt_path}", flush=True)
    return receipt_path


# ---------------------------------------------------------------------
# 2b. Adopt: verify an already-staged directory against a pinned revision's
#     manifest instead of downloading it, then write the same receipt shape.
# ---------------------------------------------------------------------
def _verify_adopted_entry(dest: Path, entry: dict) -> dict:
    """Verify one manifest ``entry`` against the file already on disk at
    ``dest / entry['name']``, returning the receipt fragment for it.

    Refuses (``PullError``, naming the file) on: a missing file, a path that
    is not a regular file (a symlink is refused even if it targets a regular
    file -- ``lstat`` is used, never ``stat``, so a symlink escape is never
    silently followed), a size mismatch, or a content-identity mismatch.
    Content identity reuses ``downloader.hash_file`` -- the exact function
    the downloader itself uses to verify a fresh download and a
    ``--reuse-verified-from`` candidate -- so an LFS entry is checked by
    streamed sha256 and a non-LFS entry by the same git-blob-sha1 identity a
    real pull checks; no new verification algorithm is introduced here. Only
    an entry with neither identity (never produced by
    ``downloader.validated_entries`` today, which always requires a valid
    ``blob_id``) falls back to a recorded ``"verified": "size-only"``.
    """
    name = entry["name"]
    file_path = dest / name
    try:
        info = file_path.lstat()
    except OSError as error:
        raise PullError(f"--adopt refused: missing file in {dest}: {name}") from error
    if not stat.S_ISREG(info.st_mode):
        raise PullError(
            f"--adopt refused: {name} in {dest} is not a regular file "
            "(symlink, directory, or device); it is refused even if it "
            "targets correct content"
        )
    if info.st_size != entry["size"]:
        raise PullError(
            f"--adopt refused: size mismatch for {name}: the pinned revision "
            f"declares {entry['size']} bytes, {dest} has {info.st_size} bytes"
        )

    expected_identity = entry["lfs_sha256"] or entry["blob_id"]
    if expected_identity is None:
        return {
            "sha256": _stream_sha256(file_path),
            "size": entry["size"],
            "verified": "size-only",
        }
    try:
        observed_identity = downloader.hash_file(
            file_path, entry["size"], entry["lfs_sha256"]
        )
    except downloader.AcquisitionError as error:
        raise PullError(
            f"--adopt refused: could not verify {name} in {dest}: {error}"
        ) from error
    if observed_identity != expected_identity:
        raise PullError(
            f"--adopt refused: content mismatch for {name}: {dest} does not "
            "hold the bytes the pinned revision declares"
        )
    # An LFS identity IS the plain file sha256, so reuse it rather than read
    # a multi-GB file a second time; a non-LFS identity is a git blob sha1,
    # so the receipt's plain sha256 still needs its own (small-file) pass.
    if entry["lfs_sha256"] is not None:
        return {"sha256": observed_identity, "size": entry["size"], "verified": "lfs-sha256"}
    return {
        "sha256": _stream_sha256(file_path),
        "size": entry["size"],
        "verified": "git-blob-sha1",
    }


def _refuse_unmanifested_loadable_extras(dest: Path, entries: list) -> list:
    """Recursively walk ``dest`` and refuse any regular file, at any depth,
    whose relative path is not an exact manifest entry (it could be loaded
    even though it was never verified against the pinned revision); an
    unlisted symlink or other non-regular path is refused the same way.
    Directories themselves are never refused on, and a path with any
    component starting with ``.`` (a ``.cache/`` directory, a dotfile, a
    hand-copied README someone renamed to start with a dot, ...) is never
    descended into and is returned as ``ignored_local_paths`` for the
    receipt instead -- the only paths this function treats as harmless.

    A top-level-only, suffix-based check (the previous behavior: only
    unlisted ``*.safetensors``/``*.gguf``/``*.bin`` files were refused, and
    subdirectories were never even looked into) let a receipt vouch for
    content the pinned revision never had -- an edited config file, a
    hand-copied script, or any file nested in a manifest-named
    subdirectory. Every non-dot, non-manifest path is now refused,
    regardless of name or depth.
    """
    manifest_names = {entry["name"] for entry in entries}
    ignored_local_paths: list = []

    def _walk(directory: Path) -> None:
        try:
            children = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as error:
            raise PullError(f"--adopt refused: could not list {directory}: {error}") from error
        for child in children:
            relative = Path(child.path).relative_to(dest).as_posix()
            if child.name.startswith("."):
                ignored_local_paths.append(relative)
                continue
            if child.is_symlink():
                raise PullError(
                    f"--adopt refused: {dest} contains a path that is not "
                    f"part of the pinned revision's manifest: {relative} "
                    "(a symlink); it could be loaded even though it was "
                    "never verified -- remove it or use a clean directory"
                )
            if child.is_dir(follow_symlinks=False):
                _walk(Path(child.path))
                continue
            if child.is_file(follow_symlinks=False):
                if relative in manifest_names:
                    continue
                raise PullError(
                    f"--adopt refused: {dest} contains a file that is not "
                    f"part of the pinned revision's manifest: {relative}; "
                    "it could be loaded even though it was never verified "
                    "-- remove it or use a clean directory"
                )
            raise PullError(
                f"--adopt refused: {dest} contains a path that is not part "
                f"of the pinned revision's manifest: {relative} (not a "
                "regular file or directory); remove it or use a clean "
                "directory"
            )

    _walk(dest)
    return ignored_local_paths


def adopt(repo_id: str, revision: str, dest: Path) -> Path:
    """Verify an already-staged ``dest`` against the pinned revision's file
    manifest and write the same receipt a real ``pull()`` would, marked
    ``"acquisition": "adopted"``. Nothing is downloaded; every file must
    already be present and correct. Writes nothing on any refusal.
    """
    dest = dest.expanduser().absolute()
    receipt_path = receipt_path_for(dest)
    if receipt_path.exists() or receipt_path.is_symlink():
        raise PullError(
            f"refusing to overwrite an existing pull receipt: {receipt_path}"
        )
    if dest.is_symlink() or not dest.is_dir():
        raise PullError(
            f"--adopt requires an existing directory to verify (not a fresh "
            f"download destination): {dest}"
        )

    document, _api_data = downloader.fetch_api(repo_id, revision)
    entries = downloader.validated_entries(document, repo_id, revision)

    files_receipt: dict[str, dict[str, object]] = {}
    for index, entry in enumerate(entries, start=1):
        print(
            f"adopt verify file={index}/{len(entries)} name={entry['name']}",
            file=sys.stderr,
            flush=True,
        )
        files_receipt[entry["name"]] = _verify_adopted_entry(dest, entry)

    ignored_local_paths = _refuse_unmanifested_loadable_extras(dest, entries)

    total_bytes = sum(entry["size"] for entry in entries)
    receipt = {
        "format_version": 1,
        "repo_id": repo_id,
        "revision": revision,
        "dest": str(dest),
        "attempts": 1,
        "max_attempts": 1,
        "total_files": len(entries),
        "total_bytes": total_bytes,
        "reused_files": 0,
        "reused_bytes": 0,
        "downloader_script_sha256": downloader_script_sha256(),
        "files": files_receipt,
        "acquisition": "adopted",
        "ignored_local_paths": ignored_local_paths,
    }
    receipt_bytes = json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n"
    downloader.write_exclusive(receipt_path, receipt_bytes)
    print(f"adopt complete: {receipt_path}", flush=True)
    return receipt_path


# ---------------------------------------------------------------------
# 3. CLI
# ---------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fastmlx_pull",
        description=(
            "Pull one pinned Hugging Face snapshot: verified, resumable, "
            "stdlib only."
        ),
    )
    parser.add_argument(
        "pinned_reference",
        help="<org>/<name>@<40-character-lowercase-hex-commit-sha>",
    )
    parser.add_argument(
        "--dest",
        required=True,
        type=Path,
        help=(
            "the destination directory: where a fresh download lands, or "
            "the existing directory --adopt verifies in place"
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=(
            "how many times to invoke the downloader before giving up, "
            f"resuming from the previous attempt's preserved staging tree "
            f"each retry (default {DEFAULT_MAX_ATTEMPTS})"
        ),
    )
    parser.add_argument(
        "--min-free-bytes",
        type=int,
        default=None,
        help=(
            "override the preflight free-space floor, in bytes (default: "
            "the revision's total planned size times the downloader's own "
            "safety multiplier)"
        ),
    )
    parser.add_argument(
        "--adopt",
        action="store_true",
        help=(
            "verify an EXISTING directory at --dest against the pinned "
            "revision's file manifest instead of downloading it, then "
            "write the same .pull-receipt.json a download would (marked "
            "acquisition: adopted); refuses on any missing, wrong-size, "
            "wrong-content, or unverified-but-loadable file"
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        repo_id, revision = validate_pinned_reference(args.pinned_reference)
    except PinnedReferenceError as error:
        print(f"fastmlx pull refused: {error}", file=sys.stderr)
        raise SystemExit(1)
    try:
        if args.adopt:
            adopt(repo_id=repo_id, revision=revision, dest=args.dest)
        else:
            pull(
                repo_id=repo_id,
                revision=revision,
                dest=args.dest,
                max_attempts=args.max_attempts,
                min_free_bytes=args.min_free_bytes,
            )
    except PullError as error:
        print(f"fastmlx pull failed: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
