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
``"acquisition": "downloaded"`` for the same reason). ``--adopt`` downloads
nothing and copies nothing, so ``--max-attempts`` (nothing to retry) and
``--min-free-bytes`` (no pack bytes written, so no free-space floor) are usage
errors with it (exit 2, before any manifest fetch), refused whatever their value.

``--from-hub-cache [DIR]`` (2026-10-06): import a pinned pack that is already
in the local Hugging Face hub cache (``DIR`` defaults to ``$HF_HUB_CACHE``, else
``$HF_HOME/hub``, else ``~/.cache/huggingface/hub``) into a FRESH ``--dest``
without downloading it again. The same pinned manifest is fetched as for
``--adopt``. For every manifest entry,
``DIR/models--<owner>--<name>/snapshots/<revision>/<name>`` must resolve to a
regular file whose real path lies inside that repo's own ``blobs/``; the
resolved blob (never the link) is cloned copy-on-write where the volume
supports it (macOS ``clonefile(2)``), otherwise copied, into a sibling staging
directory, verified with the ``--adopt`` verifier (size + content identity),
and the staging directory is renamed to ``--dest``. The receipt is the usual shape
plus ``"acquisition": "hub-cache"`` and ``"hub_cache_dir"``. Any refusal
removes the staging directory it created and writes no dest and no receipt.
The cache is read-only: no blob is modified, moved, or linked (a clone is an
independent file, never a hard link). Not combinable
with ``--adopt``; ``--kv-reserve-gib`` runs before any copy, then the same
free-space floor as a download (``--min-free-bytes`` or total size times the
downloader's safety multiplier) is checked before the staging directory exists
and does not assume a clone will succeed.
``--max-attempts`` is a usage error here (an import makes one copy).

``--kv-reserve-gib N`` (with optional ``--context C`` and ``--host-use
{shared,dedicated-serving}``) turns on a pre-download judgment: from the
revision manifest ``pull`` has just fetched (never a second fetch), and
before the free-space probe or any staging directory exists, it sizes the
pack exactly as ``fastmlx recommend --pinned`` does (the built-in safetensors
sizer fed the manifest's file sizes, against this host's wired-memory
ceiling) and looks the pinned revision up in the default quality-card store.
A pack that does not fit, or that cannot be sized (a GGUF-only manifest, a
>= 1 GiB unnamed non-safetensors file), or a card store that fails to load, is
refused with exit 1 and nothing written; the message says to drop
``--kv-reserve-gib`` to pull unchecked. A pack that fits prints ONE stderr
line naming its quality card and verdict (``--accept-quality <id>`` when the
card is opt-in, "unmeasured" when it has none) and the download proceeds --
``pull`` never refuses on the card verdict; admission is ``fastmlx serve``'s
job. Without ``--kv-reserve-gib`` nothing changes (staging a pack for another
host stays legitimate); ``--context``/``--host-use`` alone, and ``--adopt``
with ``--kv-reserve-gib``, are usage errors (exit 2).

This script never executes or imports anything from the pulled repository,
and it does not add any new token/credential handling.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.util
import json
import os
import re
import shutil
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


class PreflightRefused(Exception):
    """A pre-download check (``--kv-reserve-gib``) refused this pull; nothing
    has been written. The message is shown verbatim on stderr."""


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


def _check_free_space(
    dest: Path,
    total_bytes: int,
    min_free_bytes: Optional[int],
    disk_usage_bytes: Optional[Callable[[Path], int]],
) -> None:
    """Refuse with ``PullError`` unless the volume ``dest`` will land on has
    the required free bytes: ``min_free_bytes`` when given, else ``total_bytes``
    times the downloader's own safety multiplier.

    ``dest.parent`` may not exist yet (it is created later), and the real probe
    ``statvfs()``es it, so the NEAREST EXISTING ANCESTOR is probed instead: the
    same volume the directory will be created on. The probe is looked up at call
    time so a test patching ``downloader.probe_free_space_bytes`` is honored.
    """
    required_free_bytes = (
        min_free_bytes
        if min_free_bytes is not None
        else int(total_bytes * downloader.FREE_SPACE_SAFETY_MULTIPLIER)
    )
    probe = disk_usage_bytes if disk_usage_bytes is not None else downloader.probe_free_space_bytes
    probe_path = dest.parent
    while not probe_path.exists() and probe_path != probe_path.parent:
        probe_path = probe_path.parent
    available_free_bytes = probe(probe_path)
    if available_free_bytes < required_free_bytes:
        raise PullError(
            "insufficient free space at "
            f"{dest.parent}: need >= {required_free_bytes} bytes, "
            f"have {available_free_bytes} bytes"
        )


def pull(
    repo_id: str,
    revision: str,
    dest: Path,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    min_free_bytes: Optional[int] = None,
    disk_usage_bytes: Optional[Callable[[Path], int]] = None,
    preflight: Optional[Callable[[list], None]] = None,
) -> Path:
    """Pull one pinned snapshot into ``dest``, resuming across attempts.

    ``preflight``, when given, is called with the validated manifest
    ``entries`` right after they are fetched and BEFORE the free-space probe
    and before any staging directory exists; it refuses the pull by raising
    ``PreflightRefused`` (nothing has been written by then).

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
    if preflight is not None:
        preflight(entries)
    total_bytes = sum(entry["size"] for entry in entries)

    _check_free_space(dest, total_bytes, min_free_bytes, disk_usage_bytes)

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
def _verify_adopted_entry(dest: Path, entry: dict, label: str = "--adopt") -> dict:
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
        raise PullError(f"{label} refused: missing file in {dest}: {name}") from error
    if not stat.S_ISREG(info.st_mode):
        raise PullError(
            f"{label} refused: {name} in {dest} is not a regular file "
            "(symlink, directory, or device); it is refused even if it "
            "targets correct content"
        )
    if info.st_size != entry["size"]:
        raise PullError(
            f"{label} refused: size mismatch for {name}: the pinned revision "
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
            f"{label} refused: could not verify {name} in {dest}: {error}"
        ) from error
    if observed_identity != expected_identity:
        raise PullError(
            f"{label} refused: content mismatch for {name}: {dest} does not "
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


def resolve_hub_cache_dir(explicit: Optional[str], environ=None) -> Path:
    """The hub cache directory: ``explicit`` (a non-empty ``--from-hub-cache``
    value) > ``HF_HUB_CACHE`` > ``HF_HOME/hub`` > ``~/.cache/huggingface/hub``.
    Empty environment values count as unset. ``environ`` is a parameter so
    tests need not touch the process environment."""
    env = os.environ if environ is None else environ
    if explicit:
        return Path(explicit).expanduser()
    if env.get("HF_HUB_CACHE"):
        return Path(env["HF_HUB_CACHE"]).expanduser()
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]).expanduser() / "hub"
    home = env.get("HOME") or os.path.expanduser("~")
    return Path(home) / ".cache" / "huggingface" / "hub"


def _resolve_cache_blob(snapshot: Path, blobs_real: str, name: str, repo_label: str) -> str:
    """The real path of the cache blob behind ``snapshot / name``, refusing
    (naming ``name``) a name that could leave the snapshot, a missing entry,
    a real path outside this repo's own ``blobs/``, and a non-regular file."""
    parts = Path(name).parts
    if Path(name).is_absolute() or ".." in parts or not parts:
        raise PullError(
            f"--from-hub-cache refused: manifest name {name!r} is not a "
            "relative path inside the snapshot"
        )
    link = snapshot / name
    if not os.path.lexists(link):
        raise PullError(
            f"--from-hub-cache refused: {name} is missing from the cached "
            f"snapshot {snapshot}; the cache does not hold this pinned "
            f"revision completely -- run a plain 'fastmlx pull {repo_label} "
            "--dest <dir>' instead"
        )
    real = os.path.realpath(link)
    try:
        inside = (
            os.path.commonpath([real, blobs_real]) == blobs_real and real != blobs_real
        )
    except ValueError:
        inside = False
    if not inside:
        raise PullError(
            f"--from-hub-cache refused: {name} resolves to {real}, which is "
            f"not inside this repository's cache blobs/ directory ({blobs_real})"
        )
    try:
        info = os.lstat(real)
    except OSError as error:
        raise PullError(
            f"--from-hub-cache refused: {name} is a dangling cache link "
            f"({real} is missing)"
        ) from error
    if not stat.S_ISREG(info.st_mode):
        raise PullError(
            f"--from-hub-cache refused: {name} does not resolve to a regular "
            f"file in the cache blobs/ directory ({real})"
        )
    return real


# clonefile(2) flag: do not follow a symlink at the source.
_CLONE_NOFOLLOW = 0x0001


def _clone_file(src: str, dst: str) -> None:
    """Clone ``src`` to the new file ``dst`` with libc ``clonefile(2)`` (macOS:
    an independent copy-on-write file, never a link). ``dst`` must not exist.
    Raises ``OSError`` carrying ``errno`` on failure."""
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    clonefile = libc.clonefile
    clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32]
    clonefile.restype = ctypes.c_int
    if clonefile(os.fsencode(src), os.fsencode(dst), _CLONE_NOFOLLOW) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), dst)


def _resolve_clone_primitive() -> Optional[Callable[[str, str], None]]:
    """The copy-on-write clone primitive, or ``None`` when this platform or
    libc has none (not macOS, or ``clonefile`` is missing)."""
    if sys.platform != "darwin":
        return None
    try:
        import ctypes

        getattr(ctypes.CDLL(None), "clonefile")
    except (ImportError, OSError, AttributeError):
        return None
    return _clone_file


def import_from_hub_cache(
    repo_id: str,
    revision: str,
    dest: Path,
    hub_dir: Path,
    preflight: Optional[Callable[[list], None]] = None,
    min_free_bytes: Optional[int] = None,
    disk_usage_bytes: Optional[Callable[[Path], int]] = None,
) -> Path:
    """Import a pinned pack from the local Hugging Face hub cache into the
    fresh ``dest`` (see the module docstring), verified like ``adopt()``, and
    write a receipt marked ``"acquisition": "hub-cache"``. Nothing is
    downloaded; the cache is only read. On any failure the staging directory
    this call created is removed and no dest or receipt is left.

    ``preflight``, when given, is called with the manifest entries before any
    copy (``--kv-reserve-gib``); it refuses by raising ``PreflightRefused``.

    After the preflight and the cache-blob containment checks, and before any
    staging directory or ``dest`` exists, free space on ``dest``'s volume is
    checked exactly as ``pull()`` does (``min_free_bytes`` or the manifest's
    total size times the downloader's safety multiplier); a shortfall refuses
    with ``PullError`` and nothing written.
    """
    dest = dest.expanduser().absolute()
    hub_dir = Path(hub_dir).expanduser()
    receipt_path = receipt_path_for(dest)
    if receipt_path.exists() or receipt_path.is_symlink():
        raise PullError(
            f"refusing to overwrite an existing pull receipt: {receipt_path}"
        )
    if os.path.lexists(dest):
        raise PullError(
            f"--from-hub-cache requires a fresh destination, but {dest} "
            "already exists"
        )

    repo_label = f"{repo_id}@{revision}"
    repo_dir = hub_dir / ("models--" + repo_id.replace("/", "--"))
    snapshot = repo_dir / "snapshots" / revision
    if not snapshot.is_dir():
        raise PullError(
            f"--from-hub-cache refused: no cached snapshot for {repo_label} "
            f"under {hub_dir} (expected {snapshot}); run a plain "
            f"'fastmlx pull {repo_label} --dest {dest}' to download it"
        )
    blobs_real = os.path.realpath(repo_dir / "blobs")

    document, _api_data = downloader.fetch_api(repo_id, revision)
    entries = downloader.validated_entries(document, repo_id, revision)
    if preflight is not None:
        preflight(entries)

    # Resolve and contain every entry BEFORE anything is copied.
    sources = {
        entry["name"]: _resolve_cache_blob(snapshot, blobs_real, entry["name"], repo_label)
        for entry in entries
    }

    _check_free_space(
        dest,
        sum(entry["size"] for entry in entries),
        min_free_bytes,
        disk_usage_bytes,
    )

    dest.parent.mkdir(parents=True, exist_ok=True)
    staging: Optional[Path] = None
    counter = 0
    while staging is None:
        candidate = dest.parent / f".{dest.name}.hub-import-{os.getpid()}-{counter}"
        try:
            candidate.mkdir()
        except FileExistsError:
            counter += 1
            continue
        staging = candidate

    try:
        files_receipt: dict[str, dict[str, object]] = {}
        clone = _resolve_clone_primitive()
        for index, entry in enumerate(entries, start=1):
            name = entry["name"]
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            method = "copy"
            if clone is not None:
                try:
                    clone(str(sources[name]), str(target))
                    method = "clone"
                except OSError as error:
                    if error.errno == errno.EEXIST:
                        raise PullError(
                            f"--from-hub-cache refused: cloning {name} found "
                            f"{target} already present in the private "
                            f"staging directory: {error}"
                        ) from error
                    # Unsupported volume, cross-device, permission, ...: this
                    # file falls back to a byte copy (drop a partial target).
                    if os.path.lexists(target):
                        os.unlink(target)
            print(
                f"hub-cache {method} file={index}/{len(entries)} name={name}",
                file=sys.stderr,
                flush=True,
            )
            if method == "copy":
                try:
                    with open(sources[name], "rb") as source, open(
                        target, "xb"
                    ) as sink:
                        shutil.copyfileobj(source, sink, _RECEIPT_HASH_CHUNK_BYTES)
                except OSError as error:
                    raise PullError(
                        f"--from-hub-cache refused: could not copy {name} from "
                        f"the cache: {error}"
                    ) from error
            # Never a link: the import must be an independent file.
            try:
                imported, cached = os.stat(target), os.stat(sources[name])
            except OSError as error:
                raise PullError(
                    f"--from-hub-cache refused: {name} was not imported "
                    f"from the cache: {error}"
                ) from error
            if (imported.st_dev, imported.st_ino) == (cached.st_dev, cached.st_ino):
                raise PullError(
                    f"--from-hub-cache refused: {name} is a link to the cache "
                    "blob; refusing"
                )
        for entry in entries:
            files_receipt[entry["name"]] = _verify_adopted_entry(
                staging, entry, label="--from-hub-cache"
            )
        if os.path.lexists(dest):
            raise PullError(
                f"--from-hub-cache requires a fresh destination, but {dest} "
                "appeared during the import"
            )
        os.rename(staging, dest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

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
        "acquisition": "hub-cache",
        "hub_cache_dir": str(hub_dir),
        "ignored_local_paths": [],
    }
    receipt_bytes = json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n"
    try:
        downloader.write_exclusive(receipt_path, receipt_bytes)
    except BaseException:
        # The dest was created by this call a moment ago; do not leave it
        # without the receipt that vouches for it.
        shutil.rmtree(dest, ignore_errors=True)
        raise
    print(f"hub-cache import complete: {receipt_path}", flush=True)
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
            "the destination directory: where a fresh download or a "
            "--from-hub-cache import lands, or the existing directory "
            "--adopt verifies in place"
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help=(
            "how many times to invoke the downloader before giving up, "
            f"resuming from the previous attempt's preserved staging tree "
            f"each retry (default {DEFAULT_MAX_ATTEMPTS}; not allowed with "
            "--from-hub-cache, which makes a single copy, or with --adopt, "
            "which downloads nothing)"
        ),
    )
    parser.add_argument(
        "--min-free-bytes",
        type=int,
        default=None,
        help=(
            "override the preflight free-space floor, in bytes (default: "
            "the revision's total planned size times the downloader's own "
            "safety multiplier); applies to a download and to "
            "--from-hub-cache; not allowed with --adopt, which writes no "
            "pack bytes"
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--from-hub-cache",
        nargs="?",
        const="",
        default=None,
        metavar="DIR",
        help=(
            "import the pinned pack from the local Hugging Face hub cache "
            "into the fresh --dest instead of downloading it: each manifest "
            "file must resolve to a regular file inside that repo's cache "
            "blobs/; blobs are cloned copy-on-write where the volume "
            "supports it, otherwise copied; never linked. Each is verified "
            "like --adopt, and a receipt marked acquisition: hub-cache is written. DIR "
            "defaults to $HF_HUB_CACHE, else $HF_HOME/hub, else "
            "~/.cache/huggingface/hub. The cache is only read. Not "
            "combinable with --adopt; --kv-reserve-gib applies"
        ),
    )
    mode.add_argument(
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
    parser.add_argument(
        "--kv-reserve-gib",
        type=float,
        default=None,
        help=(
            "judge fit and the quality card BEFORE downloading: size the "
            "pinned revision's manifest (as 'fastmlx recommend --pinned' "
            "does) against this host's wired-memory ceiling with this "
            "KV-cache reserve (GiB). A pack that does not fit or cannot be "
            "sized is refused (exit 1, nothing written); otherwise one line "
            "names its quality card and verdict and the download proceeds "
            "(the card verdict never refuses a pull). Without this flag no "
            "check runs. Not combinable with --adopt"
        ),
    )
    parser.add_argument(
        "--context",
        type=int,
        default=None,
        help=(
            "the context length the --kv-reserve-gib check sizes for "
            "(needs --kv-reserve-gib)"
        ),
    )
    parser.add_argument(
        "--host-use",
        choices=["shared", "dedicated-serving"],
        default=None,
        help=(
            "the host-sharing mode the --kv-reserve-gib check assumes "
            "(default 'shared'; needs --kv-reserve-gib)"
        ),
    )
    return parser


def _load_recommend_module():
    """Load ``fastmlx_recommend.py`` by file path, on demand.

    ``fastmlx_recommend`` imports ``fastmlx_launch``, which imports this
    module, so a top-level import here would recurse; ``main`` calls this
    only when ``--kv-reserve-gib`` asks for the pre-download check.
    """
    path = Path(__file__).resolve().parent / "fastmlx_recommend.py"
    spec = importlib.util.spec_from_file_location("fastmlx_recommend", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_fit_and_card_preflight(
    repo_id: str,
    revision: str,
    kv_reserve_gib: float,
    context: Optional[int],
    host_use: str,
) -> Callable[[list], None]:
    """The ``--kv-reserve-gib`` pre-download check, as a ``pull(preflight=)``
    hook. Loads ``fastmlx_recommend`` and the default card store NOW (a store
    that cannot be loaded refuses before any network request), and classifies
    the manifest entries ``pull`` hands it with the same pure classifier
    ``recommend --pinned`` uses -- it never fetches the manifest itself."""
    recommend = _load_recommend_module()
    try:
        cards = recommend.load_default_card_store("fastmlx pull")
    except recommend.launch.LaunchRefusal as refusal:
        raise PreflightRefused(
            f"the quality-card store could not be loaded, so fit and card "
            f"cannot be judged: {refusal.message}; nothing was written (drop "
            "--kv-reserve-gib to pull without the check)"
        ) from refusal
    ref = f"{repo_id}@{revision}"

    def preflight(entries: list) -> None:
        row = recommend.classify_pinned_entries(
            ref=ref,
            entries=entries,
            cards=cards,
            host_use=host_use,
            context=context,
            fit_check_args=[],
            kv_reserve_gib=kv_reserve_gib,
        )
        status = row["status"]
        if status in (recommend.STATUS_DOES_NOT_FIT, recommend.STATUS_ERROR):
            raise PreflightRefused(
                f"{row['message']}; nothing was written (drop --kv-reserve-gib "
                "to pull without the check)"
            )
        card = row.get("card") or {}
        card_id = card.get("id")
        verdict = card.get("verdict")
        if status == recommend.STATUS_OPT_IN:
            line = (
                f"fits this host; quality card {card_id} verdict {verdict} -- "
                f"serving it needs {row['accept_quality_flag']}"
            )
        elif status == recommend.STATUS_UNCARDED:
            if card_id:
                line = (
                    f"fits this host; quality card {card_id} verdict {verdict} "
                    "is not a measured verdict -- fastmlx serve admits it as "
                    "unmeasured"
                )
            else:
                line = (
                    "fits this host; no quality card names this pack (uncarded) "
                    "-- fastmlx serve admits it as unmeasured"
                )
        else:
            line = f"fits this host; quality card {card_id} verdict {verdict}"
        print(f"fastmlx pull: {ref}: {line}", file=sys.stderr, flush=True)

    return preflight


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.kv_reserve_gib is None:
        for flag, value in (("--context", args.context), ("--host-use", args.host_use)):
            if value is not None:
                parser.error(f"{flag} has no effect without --kv-reserve-gib")
    elif args.adopt:
        parser.error(
            "--adopt cannot be combined with --kv-reserve-gib: --adopt "
            "verifies a directory that already exists and downloads nothing, "
            "so there is nothing to check before a download"
        )
    if args.from_hub_cache is not None and args.max_attempts is not None:
        parser.error(
            "--max-attempts has no effect with --from-hub-cache: an import "
            "makes a single copy, with no download to retry"
        )
    if args.adopt and args.max_attempts is not None:
        parser.error(
            "--max-attempts has no effect with --adopt: --adopt verifies a "
            "directory that already exists and downloads nothing, so there "
            "is nothing to retry"
        )
    if args.adopt and args.min_free_bytes is not None:
        parser.error(
            "--min-free-bytes has no effect with --adopt: --adopt verifies a "
            "directory in place and writes no pack bytes, so there is no "
            "free-space floor to check"
        )
    try:
        repo_id, revision = validate_pinned_reference(args.pinned_reference)
    except PinnedReferenceError as error:
        print(f"fastmlx pull refused: {error}", file=sys.stderr)
        raise SystemExit(1)
    try:
        if args.adopt:
            adopt(repo_id=repo_id, revision=revision, dest=args.dest)
        else:
            preflight = None
            if args.kv_reserve_gib is not None:
                preflight = _build_fit_and_card_preflight(
                    repo_id,
                    revision,
                    args.kv_reserve_gib,
                    args.context,
                    args.host_use or "shared",
                )
            if args.from_hub_cache is not None:
                import_from_hub_cache(
                    repo_id=repo_id,
                    revision=revision,
                    dest=args.dest,
                    hub_dir=resolve_hub_cache_dir(args.from_hub_cache),
                    preflight=preflight,
                    min_free_bytes=args.min_free_bytes,
                )
                return
            pull(
                repo_id=repo_id,
                revision=revision,
                dest=args.dest,
                max_attempts=(
                    args.max_attempts
                    if args.max_attempts is not None
                    else DEFAULT_MAX_ATTEMPTS
                ),
                min_free_bytes=args.min_free_bytes,
                preflight=preflight,
            )
    except PreflightRefused as error:
        print(f"fastmlx pull refused: {error}", file=sys.stderr)
        raise SystemExit(1)
    except PullError as error:
        print(f"fastmlx pull failed: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
