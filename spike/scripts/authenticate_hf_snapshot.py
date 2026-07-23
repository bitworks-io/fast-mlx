#!/usr/bin/env python3
"""Authenticate a Hugging Face local-dir snapshot without network access.

The receipt proves that the local regular files match the pinned repository
tree metadata, per-file download metadata, LFS hashes or Git blob identities,
checkpoint index, model geometry, and fast-mlx's own checkpoint/tokenizer
content-manifest contracts. It is source admission only, never runtime or
promotion evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import stat
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable


CHUNK_BYTES = 8 << 20
CHECKPOINT_DOMAIN = b"fastmlx-checkpoint-content-manifest-v2\n"
SNAPSHOT_DOMAIN = b"fastmlx-hf-snapshot-source-lock-v1\n"
TOKENIZER_NAMES = {
    "added_tokens.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.model",
    "vocab.json",
    "vocab.txt",
}


class AuthenticationError(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--hub-cache-path", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source-api-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-model-type", required=True)
    parser.add_argument("--expected-architecture", required=True)
    parser.add_argument("--expected-max-context", type=int, required=True)
    parser.add_argument("--expected-layers", type=int, required=True)
    parser.add_argument("--expected-query-heads", type=int, required=True)
    parser.add_argument("--expected-kv-heads", type=int, required=True)
    parser.add_argument("--expected-head-dim", type=int, required=True)
    parser.add_argument("--expected-weight-bits", type=int, required=True)
    parser.add_argument(
        "--expected-weight-group-size", type=int, required=True
    )
    parser.add_argument("--expected-rope-type")
    return parser.parse_args()


def require_regular_file(path: Path) -> os.stat_result:
    try:
        result = path.lstat()
    except OSError as error:
        raise AuthenticationError(f"missing file: {path}") from error
    if path.is_symlink() or not stat.S_ISREG(result.st_mode):
        raise AuthenticationError(
            f"not a regular non-symlink file: {path}"
        )
    return result


def require_directory(path: Path, label: str) -> None:
    try:
        result = path.lstat()
    except OSError as error:
        raise AuthenticationError(f"missing {label}: {path}") from error
    if path.is_symlink() or not stat.S_ISDIR(result.st_mode):
        raise AuthenticationError(
            f"{label} is not a regular non-symlink directory: {path}"
        )


def safe_tree_name(name: str) -> str:
    value = PurePosixPath(name)
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise AuthenticationError(f"unsafe tree path: {name!r}")
    normalized = value.as_posix()
    if normalized in {".", ".cache"} or normalized.startswith(".cache/"):
        raise AuthenticationError(f"reserved tree path: {name!r}")
    return normalized


def collect_model_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for current, directories, filenames in os.walk(
        root, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        kept_directories = []
        for name in directories:
            candidate = current_path / name
            if current_path == root and name == ".cache":
                continue
            if candidate.is_symlink():
                raise AuthenticationError(
                    f"not a regular non-symlink directory: {candidate}"
                )
            require_directory(candidate, "model subdirectory")
            kept_directories.append(name)
        directories[:] = kept_directories
        for name in filenames:
            candidate = current_path / name
            require_regular_file(candidate)
            relative = candidate.relative_to(root).as_posix()
            files[relative] = candidate
    return files


def read_json_document(path: Path) -> tuple[Any, bytes]:
    try:
        data = read_regular_bytes(path)
        return json.loads(data), data
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuthenticationError(f"invalid JSON: {path}") from error


def read_regular_bytes(path: Path) -> bytes:
    source_stat = require_regular_file(path)
    with path.open("rb") as handle:
        opened_stat = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(opened_stat.st_mode)
            or opened_stat.st_dev != source_stat.st_dev
            or opened_stat.st_ino != source_stat.st_ino
        ):
            raise AuthenticationError(
                f"file identity changed while opening: {path}"
            )
        data = handle.read()
        final_stat = os.fstat(handle.fileno())
    if (
        final_stat.st_dev != opened_stat.st_dev
        or final_stat.st_ino != opened_stat.st_ino
        or final_stat.st_size != opened_stat.st_size
        or final_stat.st_mtime_ns != opened_stat.st_mtime_ns
        or len(data) != opened_stat.st_size
    ):
        raise AuthenticationError(f"file changed while reading: {path}")
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_length_field(digest: Any, data: bytes) -> None:
    digest.update(struct.pack(">Q", len(data)))
    digest.update(data)


def fnv1a64(chunks: Iterable[bytes]) -> str:
    value = 0xCBF29CE484222325
    for chunk in chunks:
        for byte in chunk:
            value ^= byte
            value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def copy_into_digests(
    handle: BinaryIO,
    expected_size: int,
    sha256: Any,
    git_blob: Any,
    checkpoint: Any | None,
    tokenizer: Any | None,
) -> None:
    observed = 0
    for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
        observed += len(chunk)
        if observed > expected_size:
            raise AuthenticationError("file grew while hashing")
        sha256.update(chunk)
        git_blob.update(chunk)
        if checkpoint is not None:
            checkpoint.update(chunk)
        if tokenizer is not None:
            tokenizer.update(chunk)
    if observed != expected_size:
        raise AuthenticationError(
            f"file size changed while hashing: "
            f"expected {expected_size}, observed {observed}"
        )


def compatible_geometry(
    config: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    architectures = config.get("architectures")
    if architectures != [args.expected_architecture]:
        raise AuthenticationError(
            f"architecture mismatch: {architectures!r}"
        )
    if config.get("model_type") != args.expected_model_type:
        raise AuthenticationError(
            f"model type mismatch: {config.get('model_type')!r}"
        )
    query_heads = config.get("num_attention_heads")
    hidden_size = config.get("hidden_size")
    head_dim = config.get("head_dim")
    if head_dim is None:
        if (
            not isinstance(hidden_size, int)
            or not isinstance(query_heads, int)
            or query_heads <= 0
            or hidden_size % query_heads
        ):
            raise AuthenticationError("cannot derive head dimension")
        head_dim = hidden_size // query_heads
    geometry = {
        "modelType": config.get("model_type"),
        "architecture": architectures[0],
        "maxPositionEmbeddings": config.get(
            "max_position_embeddings"
        ),
        "layerCount": config.get("num_hidden_layers"),
        "queryHeadCount": query_heads,
        "kvHeadCount": config.get("num_key_value_heads"),
        "headDimension": head_dim,
        "hiddenSize": hidden_size,
        "ropeScaling": config.get("rope_scaling"),
        "quantization": config.get("quantization"),
        "nativeDType": config.get("torch_dtype") or config.get("dtype"),
    }
    expected = {
        "maxPositionEmbeddings": args.expected_max_context,
        "layerCount": args.expected_layers,
        "queryHeadCount": args.expected_query_heads,
        "kvHeadCount": args.expected_kv_heads,
        "headDimension": args.expected_head_dim,
    }
    for key, value in expected.items():
        if geometry[key] != value:
            raise AuthenticationError(
                f"{key} mismatch: expected {value}, "
                f"observed {geometry[key]!r}"
            )
    quantization = geometry["quantization"]
    if not isinstance(quantization, dict):
        raise AuthenticationError("missing quantization configuration")
    if quantization.get("bits") != args.expected_weight_bits:
        raise AuthenticationError("weight bits mismatch")
    if (
        quantization.get("group_size")
        != args.expected_weight_group_size
    ):
        raise AuthenticationError("weight group size mismatch")
    if args.expected_rope_type is not None:
        rope_scaling = geometry["ropeScaling"]
        if not isinstance(rope_scaling, dict):
            raise AuthenticationError("missing rope scaling")
        rope_type = rope_scaling.get("rope_type") or rope_scaling.get(
            "type"
        )
        if rope_type != args.expected_rope_type:
            raise AuthenticationError(
                f"rope type mismatch: {rope_type!r}"
            )
    return geometry


def validate_source_api_manifest(
    source: Any,
    repo_id: str,
    revision: str,
    tree_files: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(source, dict):
        raise AuthenticationError("invalid source API manifest root")
    if source.get("id") != repo_id:
        raise AuthenticationError(
            f"source API repo mismatch: {source.get('id')!r}"
        )
    if source.get("sha") != revision:
        raise AuthenticationError(
            f"source API revision mismatch: {source.get('sha')!r}"
        )
    if source.get("private") is not False:
        raise AuthenticationError("source API unexpectedly marks repo private")
    siblings = source.get("siblings")
    if not isinstance(siblings, list) or not siblings:
        raise AuthenticationError("source API has no sibling manifest")
    by_name: dict[str, dict[str, Any]] = {}
    for sibling in siblings:
        if not isinstance(sibling, dict):
            raise AuthenticationError("invalid source API sibling")
        raw_name = sibling.get("rfilename")
        if not isinstance(raw_name, str):
            raise AuthenticationError("source API sibling has no path")
        name = safe_tree_name(raw_name)
        if name in by_name:
            raise AuthenticationError(
                f"duplicate source API sibling: {name}"
            )
        by_name[name] = sibling
    if set(by_name) != set(tree_files):
        raise AuthenticationError(
            "source API and cached tree file sets differ"
        )
    for name, expected in tree_files.items():
        sibling = by_name[name]
        if sibling.get("size") != expected.get("size"):
            raise AuthenticationError(
                f"source API size mismatch: {name}"
            )
        if sibling.get("blobId") != expected.get("blob_id"):
            raise AuthenticationError(
                f"source API blob mismatch: {name}"
            )
        expected_lfs = expected.get("lfs_sha256")
        source_lfs = sibling.get("lfs")
        if expected_lfs is None:
            if source_lfs is not None:
                raise AuthenticationError(
                    f"unexpected source API LFS identity: {name}"
                )
        else:
            if not isinstance(source_lfs, dict):
                raise AuthenticationError(
                    f"missing source API LFS identity: {name}"
                )
            if (
                source_lfs.get("sha256") != expected_lfs
                or source_lfs.get("size") != expected.get("lfs_size")
            ):
                raise AuthenticationError(
                    f"source API LFS mismatch: {name}"
                )
    return source


def write_receipt(path: Path, receipt: dict[str, Any]) -> str:
    sidecar = Path(f"{path}.sha256")
    if path.exists() or path.is_symlink():
        raise AuthenticationError(f"output already exists: {path}")
    if sidecar.exists() or sidecar.is_symlink():
        raise AuthenticationError(f"output already exists: {sidecar}")
    require_directory(path.parent, "output directory")
    encoded = (
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    ).encode()
    receipt_sha = hashlib.sha256(encoded).hexdigest()
    sidecar_data = f"{receipt_sha}  {path.name}\n".encode()
    receipt_temp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    sidecar_temp = path.parent / f".{sidecar.name}.{os.getpid()}.tmp"
    if receipt_temp.exists() or sidecar_temp.exists():
        raise AuthenticationError("temporary output already exists")
    receipt_temp_stat: os.stat_result | None = None
    sidecar_temp_stat: os.stat_result | None = None

    def unlink_if_owned(
        candidate: Path, expected: os.stat_result | None
    ) -> None:
        if expected is None:
            return
        try:
            observed = candidate.lstat()
        except FileNotFoundError:
            return
        if (
            stat.S_ISREG(observed.st_mode)
            and observed.st_dev == expected.st_dev
            and observed.st_ino == expected.st_ino
        ):
            candidate.unlink()

    try:
        with receipt_temp.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        receipt_temp_stat = require_regular_file(receipt_temp)
        with sidecar_temp.open("xb") as handle:
            handle.write(sidecar_data)
            handle.flush()
            os.fsync(handle.fileno())
        sidecar_temp_stat = require_regular_file(sidecar_temp)
        os.link(receipt_temp, path, follow_symlinks=False)
        os.link(sidecar_temp, sidecar, follow_symlinks=False)
        receipt_temp.unlink()
        sidecar_temp.unlink()
        path.chmod(0o444)
        sidecar.chmod(0o444)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        unlink_if_owned(sidecar, sidecar_temp_stat)
        unlink_if_owned(path, receipt_temp_stat)
        raise
    finally:
        receipt_temp.unlink(missing_ok=True)
        sidecar_temp.unlink(missing_ok=True)
    return receipt_sha


def authenticate(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    model = Path(args.model_path).expanduser().absolute()
    hub_cache = Path(args.hub_cache_path).expanduser().absolute()
    output = Path(args.output).expanduser().absolute()
    require_directory(model, "model directory")
    require_directory(hub_cache, "hub cache directory")
    expected_cache_name = "models--" + args.repo_id.replace("/", "--")
    if hub_cache.name != expected_cache_name:
        raise AuthenticationError(
            f"hub cache path does not match repo ID: "
            f"expected {expected_cache_name!r}"
        )
    if len(args.revision) != 40 or any(
        character not in "0123456789abcdef"
        for character in args.revision
    ):
        raise AuthenticationError("revision must be 40 lowercase hex")
    if output.exists() or output.is_symlink():
        raise AuthenticationError(f"output already exists: {output}")
    if Path(f"{output}.sha256").exists():
        raise AuthenticationError(
            f"output already exists: {output}.sha256"
        )

    ref_path = hub_cache / "refs/main"
    revision_ref: str | None
    revision_ref_sha256: str | None
    if ref_path.exists() or ref_path.is_symlink():
        require_regular_file(ref_path)
        ref_data = read_regular_bytes(ref_path)
        try:
            revision_ref = ref_data.decode("utf-8").strip()
        except UnicodeDecodeError as error:
            raise AuthenticationError(
                f"invalid revision ref: {ref_path}"
            ) from error
        if revision_ref != args.revision:
            raise AuthenticationError(
                f"revision ref mismatch: expected {args.revision}, "
                f"observed {revision_ref}"
            )
        revision_ref_sha256 = hashlib.sha256(ref_data).hexdigest()
    else:
        revision_ref = None
        revision_ref_sha256 = None

    tree_path = (
        model
        / ".cache/huggingface/trees"
        / f"{args.revision}.json"
    )
    tree, tree_data = read_json_document(tree_path)
    if not isinstance(tree, dict) or tree.get("format_version") != 1:
        raise AuthenticationError("unsupported tree metadata")
    raw_tree_files = tree.get("files")
    if not isinstance(raw_tree_files, dict) or not raw_tree_files:
        raise AuthenticationError("tree metadata has no files")
    tree_files: dict[str, dict[str, Any]] = {}
    for raw_name, value in raw_tree_files.items():
        name = safe_tree_name(raw_name)
        if name in tree_files:
            raise AuthenticationError(
                f"normalized duplicate tree path: {name}"
            )
        tree_files[name] = value
    if any(not isinstance(value, dict) for value in tree_files.values()):
        raise AuthenticationError("invalid tree file entry")
    source_api_path = (
        Path(args.source_api_manifest).expanduser().absolute()
    )
    source_api_document, source_api_data = read_json_document(
        source_api_path
    )
    source_api = validate_source_api_manifest(
        source_api_document,
        args.repo_id,
        args.revision,
        tree_files,
    )

    incomplete = sorted(
        str(path)
        for path in (
            model / ".cache/huggingface/download"
        ).rglob("*.incomplete")
    )
    if incomplete:
        raise AuthenticationError(
            f"incomplete download artifacts remain: {incomplete}"
        )
    lock_files = sorted(
        model.glob(".cache/huggingface/download/**/*.lock")
    )

    actual_files = collect_model_files(model)
    expected_names = set(tree_files)
    actual_names = set(actual_files)
    if expected_names != actual_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise AuthenticationError(
            f"snapshot file set mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )

    config_path = actual_files.get("config.json")
    index_path = actual_files.get("model.safetensors.index.json")
    if config_path is None or index_path is None:
        raise AuthenticationError("config or checkpoint index missing")
    config_data = read_regular_bytes(config_path)
    index_data = read_regular_bytes(index_path)
    try:
        config = json.loads(config_data)
        index = json.loads(index_data)
    except json.JSONDecodeError as error:
        raise AuthenticationError("invalid config or index JSON") from error
    if not isinstance(config, dict) or not isinstance(index, dict):
        raise AuthenticationError("invalid config or index root")
    geometry = compatible_geometry(config, args)

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise AuthenticationError("checkpoint index has no weights")
    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or not value
        for key, value in weight_map.items()
    ):
        raise AuthenticationError("checkpoint index has invalid weights")
    indexed_shards = set(weight_map.values())
    weight_names = sorted(
        name
        for name in actual_names
        if name.endswith(".safetensors")
    )
    if any("/" in name for name in weight_names):
        raise AuthenticationError(
            "nested checkpoint shards are incompatible with "
            "benchmark provenance"
        )
    if indexed_shards != set(weight_names):
        raise AuthenticationError(
            f"checkpoint index shard mismatch: "
            f"index={sorted(indexed_shards)}, files={weight_names}"
        )

    checkpoint_digest = hashlib.sha256()
    checkpoint_digest.update(CHECKPOINT_DOMAIN)
    update_length_field(checkpoint_digest, config_data)
    update_length_field(checkpoint_digest, index_data)
    tokenizer_digest = hashlib.sha256()
    file_results: list[dict[str, Any]] = []
    tokenizer_count = 0
    total_bytes = 0
    initial_content_sha256 = {
        "config.json": hashlib.sha256(config_data).hexdigest(),
        "model.safetensors.index.json":
            hashlib.sha256(index_data).hexdigest(),
    }

    for name in sorted(actual_names):
        path = actual_files[name]
        source_stat = require_regular_file(path)
        expected = tree_files[name]
        expected_size = expected.get("size")
        if not isinstance(expected_size, int) or expected_size < 0:
            raise AuthenticationError(f"invalid expected size: {name}")
        if source_stat.st_size != expected_size:
            raise AuthenticationError(
                f"size mismatch for {name}: expected {expected_size}, "
                f"observed {source_stat.st_size}"
            )
        metadata_path = (
            model
            / ".cache/huggingface/download"
            / f"{name}.metadata"
        )
        metadata_data = read_regular_bytes(metadata_path)
        try:
            metadata_lines = metadata_data.decode(
                "utf-8"
            ).splitlines()
        except UnicodeDecodeError as error:
            raise AuthenticationError(
                f"invalid download metadata: {metadata_path}"
            ) from error
        if len(metadata_lines) < 2:
            raise AuthenticationError(
                f"invalid download metadata: {metadata_path}"
            )
        if metadata_lines[0] != args.revision:
            raise AuthenticationError(
                f"download metadata revision mismatch: {name}"
            )

        is_weight = name in weight_names
        basename = PurePosixPath(name).name.lower()
        is_tokenizer = basename.startswith(
            "tokenizer"
        ) or basename in TOKENIZER_NAMES
        if is_tokenizer and "/" in name:
            raise AuthenticationError(
                f"nested tokenizer file is unsupported: {name}"
            )
        sha256 = hashlib.sha256()
        git_blob = hashlib.sha1()
        git_blob.update(f"blob {expected_size}\0".encode())
        if is_weight:
            update_length_field(
                checkpoint_digest, name.encode("utf-8")
            )
            checkpoint_digest.update(struct.pack(">Q", expected_size))
        if is_tokenizer:
            encoded_name = name.encode("utf-8")
            tokenizer_digest.update(struct.pack(">Q", len(encoded_name)))
            tokenizer_digest.update(encoded_name)
            tokenizer_digest.update(struct.pack(">Q", expected_size))
            tokenizer_count += 1
        with path.open("rb") as handle:
            opened_stat = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(opened_stat.st_mode)
                or opened_stat.st_dev != source_stat.st_dev
                or opened_stat.st_ino != source_stat.st_ino
            ):
                raise AuthenticationError(
                    f"file identity changed while opening: {name}"
                )
            try:
                copy_into_digests(
                    handle,
                    expected_size,
                    sha256,
                    git_blob,
                    checkpoint_digest if is_weight else None,
                    tokenizer_digest if is_tokenizer else None,
                )
            except AuthenticationError as error:
                raise AuthenticationError(f"{name}: {error}") from error
            final_stat = os.fstat(handle.fileno())
        if (
            final_stat.st_dev != opened_stat.st_dev
            or final_stat.st_ino != opened_stat.st_ino
            or final_stat.st_size != opened_stat.st_size
            or final_stat.st_mtime_ns != opened_stat.st_mtime_ns
        ):
            raise AuthenticationError(f"file changed while hashing: {name}")
        observed_sha256 = sha256.hexdigest()
        observed_git_blob = git_blob.hexdigest()
        if (
            name in initial_content_sha256
            and observed_sha256 != initial_content_sha256[name]
        ):
            raise AuthenticationError(
                f"file changed after contract validation: {name}"
            )
        expected_lfs = expected.get("lfs_sha256")
        if expected_lfs is not None:
            if (
                not isinstance(expected_lfs, str)
                or expected.get("lfs_size") != expected_size
                or observed_sha256 != expected_lfs
            ):
                raise AuthenticationError(
                    f"LFS identity mismatch: {name}"
                )
            expected_metadata_identity = expected_lfs
            identity_kind = "lfs-sha256"
        else:
            expected_blob = expected.get("blob_id")
            if (
                not isinstance(expected_blob, str)
                or observed_git_blob != expected_blob
            ):
                raise AuthenticationError(
                    f"Git blob identity mismatch: {name}"
                )
            expected_metadata_identity = expected_blob
            identity_kind = "git-blob-sha1"
        if metadata_lines[1] != expected_metadata_identity:
            raise AuthenticationError(
                f"download metadata identity mismatch: {name}"
            )
        file_results.append(
            {
                "path": name,
                "sizeBytes": expected_size,
                "sha256": observed_sha256,
                "sourceIdentityKind": identity_kind,
                "sourceIdentity": expected_metadata_identity,
                "downloadMetadataSHA256":
                    hashlib.sha256(metadata_data).hexdigest(),
            }
        )
        total_bytes += expected_size

    if tokenizer_count == 0:
        raise AuthenticationError("no tokenizer files authenticated")

    checkpoint_manifest_hash = fnv1a64(
        [
            config_data,
            index_data,
            *[
                f"{name}:{tree_files[name]['size']}\n".encode()
                for name in weight_names
            ],
        ]
    )
    snapshot_digest = hashlib.sha256()
    snapshot_digest.update(SNAPSHOT_DOMAIN)
    for entry in file_results:
        update_length_field(
            snapshot_digest, entry["path"].encode("utf-8")
        )
        snapshot_digest.update(
            struct.pack(">Q", entry["sizeBytes"])
        )
        snapshot_digest.update(bytes.fromhex(entry["sha256"]))

    receipt = {
        "schemaVersion": 1,
        "status": "PASS",
        "purpose": "model-source-lock-admission",
        "promotable": False,
        "runtimeEvidence": False,
        "createdAt": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "host": socket.gethostname(),
        "authenticatorSHA256": sha256_file(
            Path(__file__).resolve()
        ),
        "source": {
            "repoID": args.repo_id,
            "revision": args.revision,
            "revisionRef": revision_ref,
            "revisionRefPresent": revision_ref is not None,
            "revisionRefSHA256": revision_ref_sha256,
            "modelPath": str(model),
            "hubCachePath": str(hub_cache),
            "treeMetadataPath": str(tree_path),
            "treeMetadataSHA256":
                hashlib.sha256(tree_data).hexdigest(),
            "sourceAPIManifestPath": str(source_api_path),
            "sourceAPIManifestSHA256":
                hashlib.sha256(source_api_data).hexdigest(),
            "sourceAPIRepoID": source_api["id"],
            "sourceAPIRevision": source_api["sha"],
            "sourceAPIGated": source_api.get("gated"),
        },
        "geometry": geometry,
        "fileCount": len(file_results),
        "weightShardCount": len(weight_names),
        "tokenizerFileCount": tokenizer_count,
        "totalBytes": total_bytes,
        "modelConfigHash": fnv1a64([config_data]),
        "modelConfigSHA256": hashlib.sha256(
            config_data
        ).hexdigest(),
        "checkpointManifestHash": checkpoint_manifest_hash,
        "checkpointContentSHA256": checkpoint_digest.hexdigest(),
        "tokenizerSHA256": tokenizer_digest.hexdigest(),
        "snapshotManifestSHA256": snapshot_digest.hexdigest(),
        "checkpointIndexSHA256": hashlib.sha256(
            index_data
        ).hexdigest(),
        "downloadLockFileCount": len(lock_files),
        "incompleteDownloadFileCount": 0,
        "files": file_results,
    }
    return receipt, output


def main() -> int:
    try:
        receipt, output = authenticate(parse_args())
        receipt_sha = write_receipt(output, receipt)
    except (AuthenticationError, OSError) as error:
        print(
            f"snapshot authentication failed: {error}",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "status": "PASS",
                "receipt": str(output),
                "receiptSHA256": receipt_sha,
                "snapshotManifestSHA256":
                    receipt["snapshotManifestSHA256"],
                "checkpointContentSHA256":
                    receipt["checkpointContentSHA256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
