#!/usr/bin/env python3
"""Download one immutable public Hugging Face snapshot without extra packages.

v3 (2026-09-02): adds --include-prefix subsetting for multi-quantization
repositories; the authenticated source-API manifest still records the full
repository document, and per-file verification plus atomic publication are
unchanged from the reviewed v2.

v4 (2026-09-03, fast-mlx vendoring): two behavior fixes on top of the v3
acquisition tool.

- FIX A: a transient per-file retry no longer leaves earlier failed
  ``attempt-N.partial`` files behind under
  ``.cache/huggingface/failed-downloads/<hash>/``. Those files were part of
  the staging tree that gets atomically published, so a shard that needed a
  retry could ship tens of GB of dead bytes inside the "complete" artifact
  while the receipt kept reporting only the logical (correct) total. Failed
  attempt files are now unlinked as soon as the attempt fails, so at most the
  in-flight attempt file exists at any time and a successful acquisition
  publishes zero ``*.partial`` residue.
- FIX B: a ``statvfs``-based free-space preflight now runs against the
  destination filesystem before any staging directory is created or any
  download is issued, refusing with a clear ``AcquisitionError`` when free
  space is less than ``FREE_SPACE_SAFETY_MULTIPLIER`` times the planned
  total download size. This used to be a manual, easy-to-skip step.

The destination and API-manifest paths must not exist. Downloads land in a
fresh sibling directory and are verified against the revision API's Git blob
or LFS identity before the complete directory is atomically installed. This
script itself still has no resume: a failed acquisition leaves its staging
tree under a unique ``.acquiring-*`` name and the *next call* to this script
starts over from scratch for every file. What changed in v5 is that a
*caller* (``fastmlx_pull.py``'s supervisor) can now point a fresh call at
that leftover staging tree via ``--reuse-verified-from`` so it does not
re-download files that were already downloaded and verified by an earlier,
failed call; this script never resumes or reuses anything on its own.

v5 (2026-09-18, fast-mlx pull supervisor): adds ``--reuse-verified-from
<preserved staging tree>``. When given, each file is first checked against
``<preserved staging tree>/<name>``: it is reused by moving it (``os.rename``,
never a hardlink and never a copy) into the new staging tree, only if that
candidate is a regular file (not a symlink, not a directory, not a device),
its on-disk size matches the revision API's declared size, and it hashes
identically to the revision API's identity (the same ``hash_file`` used to
verify a fresh download). A move, not a hardlink, is required for two
reasons: ``hash_file`` deliberately refuses to hash any candidate whose
``st_nlink != 1`` (a guard against hashing a file another hardlink could be
concurrently mutating), so a hardlinked file could never pass verification a
second time and be reused again by a later call (multi-hop reuse would be
blocked after the first hop); and a hardlink would leave a second,
writable-path link to a published file's bytes sitting inside a staging tree
that a later, failed call might still mutate. If the move raises ``OSError``
(for example ``EXDEV``, when the candidate and destination are not on the
same filesystem) the file is never copied as a fallback; it is downloaded
normally instead, exactly as if the candidate had failed verification. Any
candidate that fails a verification check, or whose move fails, is silently
left exactly where it was and the file is downloaded normally. The
staging-directory name also now includes a random component in addition to
the pid, because a long-lived supervisor process can call ``acquire`` more
than once and the old pid-only name would collide with a staging directory a
previous, failed call left behind in the same process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import ctypes
import errno
import http.client
import time
import uuid
from urllib.parse import quote, urlsplit
from urllib.error import URLError
from urllib.request import (
    HTTPSHandler,
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)


CHUNK_BYTES = 8 << 20
MAX_API_BYTES = 32 << 20
REPO_PATTERN = re.compile(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+")
LOWER_HEX_40 = re.compile(r"[0-9a-f]{40}")
LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}")

# FIX B: required headroom over the planned download total before any
# staging directory is created or any download is issued.
FREE_SPACE_SAFETY_MULTIPLIER = 1.2


class AcquisitionError(Exception):
    pass


class HTTPSOnlyRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        if urlsplit(new_url).scheme.lower() != "https":
            raise AcquisitionError("network request attempted a non-HTTPS redirect")
        return super().redirect_request(
            request, file_pointer, code, message, headers, new_url
        )


def https_opener():
    return build_opener(
        ProxyHandler({}), HTTPSHandler(), HTTPSOnlyRedirectHandler()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-api-manifest", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--include-prefix",
        action="append",
        default=None,
        help=(
            "restrict acquisition to entries equal to this path or under this"
            " directory prefix; repeatable; every given prefix must match at"
            " least one entry (v3 addition; per-file verification and atomic"
            " publication are unchanged from v2)"
        ),
    )
    parser.add_argument(
        "--reuse-verified-from",
        type=Path,
        default=None,
        help=(
            "a preserved staging tree (typically a leftover .acquiring-*"
            " directory from a previous, failed call) to reuse verified"
            " files from by moving them (never a hardlink or a copy)"
            " instead of downloading them again; a move keeps every"
            " published file at a single link, which hash_file requires,"
            " and leaves the file eligible for a later, second hop of"
            " reuse; v5 addition, this script still never resumes on its"
            " own"
        ),
    )
    return parser.parse_args()


def safe_relative_path(raw: str) -> str:
    value = PurePosixPath(raw)
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise AcquisitionError(f"unsafe repository path: {raw!r}")
    normalized = value.as_posix()
    if normalized in {".", ".cache"} or normalized.startswith(".cache/"):
        raise AcquisitionError(f"reserved repository path: {raw!r}")
    return normalized


def require_directory(path: Path, label: str) -> None:
    try:
        result = path.lstat()
    except OSError as error:
        raise AcquisitionError(f"missing {label}: {path}") from error
    if path.is_symlink() or not stat.S_ISDIR(result.st_mode):
        raise AcquisitionError(f"invalid {label}: {path}")


def write_exclusive(path: Path, data: bytes, mode: int = 0o444) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(descriptor)
    os.chmod(path, mode)


def fetch_api(repo_id: str, revision: str) -> tuple[dict, bytes]:
    url = (
        "https://huggingface.co/api/models/"
        f"{quote(repo_id, safe='/')}/revision/{revision}?blobs=true"
    )
    request = Request(url, headers={"User-Agent": "fast-mlx-pinned-acquisition/1"})
    try:
        opener = https_opener()
        with opener.open(request, timeout=60) as response:
            if urlsplit(response.geturl()).scheme.lower() != "https":
                raise AcquisitionError("source API resolved outside HTTPS")
            data = response.read(MAX_API_BYTES + 1)
    except Exception as error:
        raise AcquisitionError("source API request failed") from error
    if len(data) > MAX_API_BYTES:
        raise AcquisitionError("source API response exceeded the bound")
    try:
        document = json.loads(data)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AcquisitionError("source API response is not valid JSON") from error
    if not isinstance(document, dict):
        raise AcquisitionError("source API response has an invalid root")
    return document, data


def validated_entries(document: dict, repo_id: str, revision: str) -> list[dict]:
    if document.get("id") != repo_id:
        raise AcquisitionError("source API repository identity mismatch")
    if document.get("sha") != revision:
        raise AcquisitionError("source API revision identity mismatch")
    if document.get("private") is not False:
        raise AcquisitionError("source API does not identify a public repository")
    siblings = document.get("siblings")
    if not isinstance(siblings, list) or not siblings:
        raise AcquisitionError("source API has no file manifest")
    entries: list[dict] = []
    names: set[str] = set()
    for raw in siblings:
        if not isinstance(raw, dict):
            raise AcquisitionError("source API has an invalid file entry")
        raw_name = raw.get("rfilename")
        if not isinstance(raw_name, str):
            raise AcquisitionError("source API file has no name")
        name = safe_relative_path(raw_name)
        if name in names:
            raise AcquisitionError(f"duplicate repository path: {name}")
        names.add(name)
        size = raw.get("size")
        blob_id = raw.get("blobId")
        if not isinstance(size, int) or size < 0:
            raise AcquisitionError(f"invalid source size: {name}")
        if not isinstance(blob_id, str) or LOWER_HEX_40.fullmatch(blob_id) is None:
            raise AcquisitionError(f"invalid Git blob identity: {name}")
        lfs = raw.get("lfs")
        lfs_sha256: str | None = None
        lfs_size: int | None = None
        if lfs is not None:
            if not isinstance(lfs, dict):
                raise AcquisitionError(f"invalid LFS identity: {name}")
            lfs_sha256 = lfs.get("sha256")
            lfs_size = lfs.get("size")
            if (
                not isinstance(lfs_sha256, str)
                or LOWER_HEX_64.fullmatch(lfs_sha256) is None
                or lfs_size != size
            ):
                raise AcquisitionError(f"invalid LFS identity: {name}")
        entries.append(
            {
                "name": name,
                "size": size,
                "blob_id": blob_id,
                "lfs_sha256": lfs_sha256,
                "lfs_size": lfs_size,
            }
        )
    return sorted(entries, key=lambda entry: entry["name"])


def hash_file(path: Path, size: int, lfs_sha256: str | None) -> str:
    if lfs_sha256 is not None:
        digest = hashlib.sha256()
    else:
        digest = hashlib.sha1()
        digest.update(f"blob {size}\0".encode())
    observed = 0
    with path.open("rb") as source:
        opened = os.fstat(source.fileno())
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise AcquisitionError("download is not a single-link regular file")
        for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
            observed += len(chunk)
            if observed > size:
                raise AcquisitionError("download exceeded its declared size")
            digest.update(chunk)
        final = os.fstat(source.fileno())
    if (
        observed != size
        or final.st_dev != opened.st_dev
        or final.st_ino != opened.st_ino
        or final.st_size != opened.st_size
        or final.st_mtime_ns != opened.st_mtime_ns
    ):
        raise AcquisitionError("download size or identity changed while hashing")
    return digest.hexdigest()


def reusable_verified_candidate(candidate: Path, entry: dict) -> bool:
    """Return whether ``candidate`` may be moved in place of a download.

    v5: backs ``--reuse-verified-from``. Uses ``lstat`` (never follows a
    symlink) so a symlink that happens to point at correct content is still
    refused, then requires an exact on-disk size match before paying for a
    full hash, then reuses the same ``hash_file`` identity check a freshly
    downloaded file is held to.
    """
    try:
        info = candidate.lstat()
    except OSError:
        return False
    if not stat.S_ISREG(info.st_mode):
        return False
    if info.st_size != entry["size"]:
        return False
    try:
        observed_identity = hash_file(candidate, entry["size"], entry["lfs_sha256"])
    except (AcquisitionError, OSError):
        return False
    expected_identity = entry["lfs_sha256"] or entry["blob_id"]
    return observed_identity == expected_identity


class TransientTransferError(Exception):
    pass


def stream_download_attempt(url: str, destination: Path, expected_size: int) -> None:
    request = Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "fast-mlx-pinned-acquisition/1",
        },
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
    try:
        try:
            opener = https_opener()
            with opener.open(request, timeout=60) as response:
                if urlsplit(response.geturl()).scheme.lower() != "https":
                    raise AcquisitionError("download redirected outside HTTPS")
                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    try:
                        declared_length = int(content_length)
                    except ValueError as error:
                        raise AcquisitionError("download has an invalid content length") from error
                    if declared_length > expected_size:
                        raise AcquisitionError("download exceeds its declared source size")
                observed = 0
                with os.fdopen(descriptor, "wb", closefd=False) as output:
                    while True:
                        chunk = response.read(CHUNK_BYTES)
                        if not chunk:
                            break
                        observed += len(chunk)
                        if observed > expected_size:
                            raise AcquisitionError(
                                "download exceeded its declared source size"
                            )
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if observed != expected_size:
                    raise TransientTransferError("download ended before its declared size")
        except (URLError, http.client.IncompleteRead, TimeoutError, ConnectionError) as error:
            raise TransientTransferError("network transfer failed") from error
    finally:
        os.close(descriptor)


def download_entry(
    repo_id: str,
    revision: str,
    root: Path,
    cache_root: Path,
    entry: dict,
    index: int,
    total: int,
    reuse_root: Path | None = None,
) -> None:
    name = entry["name"]
    destination = root / name
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise AcquisitionError(f"download destination is not fresh: {name}")

    expected_identity = entry["lfs_sha256"] or entry["blob_id"]
    if reuse_root is not None:
        candidate = reuse_root / name
        if reusable_verified_candidate(candidate, entry):
            try:
                os.rename(candidate, destination)
            except OSError:
                # The move failed (for example EXDEV: candidate and
                # destination are not on the same filesystem). Never fall
                # back to a copy -- the candidate is left exactly where it
                # was, and this file is downloaded normally below, exactly
                # as if it had failed verification.
                pass
            else:
                metadata = cache_root / f"{name}.metadata"
                metadata.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                write_exclusive(metadata, f"{revision}\n{expected_identity}\n".encode())
                print(
                    f"reused file={index}/{total} bytes={entry['size']} name={name}",
                    flush=True,
                )
                return

    url = (
        f"https://huggingface.co/{quote(repo_id, safe='/')}/resolve/"
        f"{revision}/{quote(name, safe='/')}?download=true"
    )
    transfer_root = (
        root
        / ".cache/huggingface/failed-downloads"
        / hashlib.sha256(name.encode()).hexdigest()
    )
    transfer_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    successful_attempt: Path | None = None
    for attempt in range(1, 6):
        attempt_path = transfer_root / f"attempt-{attempt}.partial"
        try:
            stream_download_attempt(url, attempt_path, entry["size"])
            successful_attempt = attempt_path
            break
        except TransientTransferError:
            # FIX A: never leave a failed attempt's bytes behind under the
            # staging tree that gets atomically published. Without this, a
            # multi-attempt shard could ship every earlier failed attempt's
            # dead bytes inside the "complete" artifact.
            attempt_path.unlink(missing_ok=True)
            if attempt == 5:
                raise AcquisitionError(
                    f"download failed at file {index} of {total}: {name}"
                )
            print(
                f"retry file={index}/{total} attempt={attempt + 1} name={name}",
                flush=True,
            )
            time.sleep(2)
    if successful_attempt is None:
        raise AcquisitionError(f"download failed at file {index} of {total}: {name}")
    observed_identity = hash_file(
        successful_attempt, entry["size"], entry["lfs_sha256"]
    )
    if observed_identity != expected_identity:
        raise AcquisitionError(f"download identity mismatch: {name}")
    os.chmod(successful_attempt, 0o444)
    exclusive_rename(successful_attempt, destination)

    metadata = cache_root / f"{name}.metadata"
    metadata.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_exclusive(metadata, f"{revision}\n{expected_identity}\n".encode())
    print(f"verified file={index}/{total} bytes={entry['size']} name={name}", flush=True)


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exclusive_rename(source: Path, destination: Path) -> None:
    """Rename without replacement on the macOS acquisition host."""
    library = ctypes.CDLL(None, use_errno=True)
    renameatx_np = getattr(library, "renameatx_np", None)
    if renameatx_np is None:
        raise AcquisitionError("exclusive rename is unavailable")
    renameatx_np.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameatx_np.restype = ctypes.c_int
    at_fdcwd = -2
    rename_excl = 0x00000004
    result = renameatx_np(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_excl,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise AcquisitionError("publish destination is no longer fresh")
        raise AcquisitionError(f"exclusive publish failed with errno {error}")


def unlink_if_owned(path: Path, expected: os.stat_result) -> None:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return
    if (
        stat.S_ISREG(observed.st_mode)
        and observed.st_dev == expected.st_dev
        and observed.st_ino == expected.st_ino
    ):
        path.unlink()


def probe_free_space_bytes(path: Path) -> int:
    """Return bytes available to an unprivileged caller on ``path``'s filesystem.

    FIX B: this backs the preflight in ``acquire`` that refuses to start an
    acquisition when the destination filesystem does not have enough
    headroom for the planned download. Kept as a standalone function so
    tests can stub it without touching the real filesystem.
    """
    result = os.statvfs(path)
    return result.f_bavail * result.f_frsize


def acquire(args: argparse.Namespace) -> None:
    if REPO_PATTERN.fullmatch(args.repo_id) is None:
        raise AcquisitionError("repository ID is not canonical")
    if LOWER_HEX_40.fullmatch(args.revision) is None:
        raise AcquisitionError("revision must be 40 lowercase hex")
    document, api_data = fetch_api(args.repo_id, args.revision)
    entries = validated_entries(document, args.repo_id, args.revision)
    if args.include_prefix:
        prefixes = []
        for raw_prefix in args.include_prefix:
            prefixes.append(safe_relative_path(raw_prefix))
        selected = []
        matched: set[str] = set()
        for entry in entries:
            for prefix in prefixes:
                if entry["name"] == prefix or entry["name"].startswith(
                    prefix + "/"
                ):
                    selected.append(entry)
                    matched.add(prefix)
                    break
        unmatched = [prefix for prefix in prefixes if prefix not in matched]
        if unmatched:
            raise AcquisitionError(
                f"include prefixes matched no entries: {unmatched}"
            )
        entries = selected
    total_bytes = sum(entry["size"] for entry in entries)
    print(
        f"plan revision_match=true files={len(entries)} bytes={total_bytes}"
        f" include_prefixes={args.include_prefix or 'ALL'}",
        flush=True,
    )
    if args.plan_only:
        return
    if args.output is None or args.source_api_manifest is None:
        raise AcquisitionError("output and source API manifest are required")

    output = args.output.expanduser().absolute()
    manifest = args.source_api_manifest.expanduser().absolute()
    require_directory(output.parent, "output parent")
    require_directory(manifest.parent, "manifest parent")
    if output.exists() or output.is_symlink():
        raise AcquisitionError("output already exists")
    if manifest.exists() or manifest.is_symlink():
        raise AcquisitionError("source API manifest already exists")

    reuse_root: Path | None = None
    reuse_verified_from = getattr(args, "reuse_verified_from", None)
    if reuse_verified_from is not None:
        reuse_root = reuse_verified_from.expanduser().absolute()
        require_directory(reuse_root, "reuse-verified-from directory")

    # FIX B: refuse before creating any staging directory or issuing any
    # download when the destination filesystem does not have enough free
    # space for the planned total plus safety headroom.
    required_free_bytes = int(total_bytes * FREE_SPACE_SAFETY_MULTIPLIER)
    available_free_bytes = probe_free_space_bytes(output.parent)
    if available_free_bytes < required_free_bytes:
        raise AcquisitionError(
            "insufficient free space at "
            f"{output.parent}: need >= {required_free_bytes} bytes"
            f" ({FREE_SPACE_SAFETY_MULTIPLIER}x of planned {total_bytes} bytes),"
            f" have {available_free_bytes} bytes"
        )

    # v5: the pid alone is no longer guaranteed unique, since a long-lived
    # supervisor process (fastmlx_pull.py) can call acquire() more than once
    # and would otherwise collide with the staging directory a previous,
    # failed call in the same process left behind. The `.<name>.acquiring-*`
    # prefix is unchanged so existing staging-directory discovery keeps
    # working.
    staging_suffix = f"{os.getpid()}-{uuid.uuid4().hex}"
    temp_root = output.parent / f".{output.name}.acquiring-{staging_suffix}"
    temp_manifest = manifest.parent / f".{manifest.name}.acquiring-{staging_suffix}"
    if temp_root.exists() or temp_root.is_symlink() or temp_manifest.exists() or temp_manifest.is_symlink():
        raise AcquisitionError("acquisition staging path already exists")

    os.mkdir(temp_root, 0o700)
    cache_root = temp_root / ".cache/huggingface/download"
    cache_root.mkdir(mode=0o700, parents=True)
    write_exclusive(temp_manifest, api_data)
    manifest_identity = temp_manifest.lstat()
    for index, entry in enumerate(entries, start=1):
        download_entry(
            args.repo_id,
            args.revision,
            temp_root,
            cache_root,
            entry,
            index,
            len(entries),
            reuse_root,
        )

    tree_files = {
        entry["name"]: {
            "size": entry["size"],
            "blob_id": entry["blob_id"],
            "lfs_sha256": entry["lfs_sha256"],
            "lfs_size": entry["lfs_size"],
        }
        for entry in entries
    }
    tree = json.dumps(
        {"format_version": 1, "files": tree_files},
        indent=2,
        sort_keys=True,
    ).encode() + b"\n"
    tree_root = temp_root / ".cache/huggingface/trees"
    tree_root.mkdir(mode=0o700, parents=True)
    write_exclusive(tree_root / f"{args.revision}.json", tree)
    sync_directory(temp_root)
    exclusive_rename(temp_manifest, manifest)
    try:
        exclusive_rename(temp_root, output)
    except BaseException:
        unlink_if_owned(manifest, manifest_identity)
        raise
    sync_directory(output.parent)
    if manifest.parent != output.parent:
        sync_directory(manifest.parent)
    print(
        f"complete revision_match=true files={len(entries)} bytes={total_bytes}",
        flush=True,
    )


def main() -> None:
    try:
        acquire(parse_args())
    except AcquisitionError as error:
        print(f"acquisition failed: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
