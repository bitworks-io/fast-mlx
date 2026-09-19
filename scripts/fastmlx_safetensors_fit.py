#!/usr/bin/env python3
"""fastmlx-safetensors-fit: a safetensors-aware, residency-aware pre-load
fit checker.

Given a local pack directory containing one or more ``*.safetensors``
files anywhere in its tree (a dot-prefixed directory, e.g. ``.cache``, is
never scanned), this binary parses each shard's safetensors header (an
8-byte little-endian u64 header length, followed by that many bytes of
JSON tensor metadata), reconciles each file's declared tensor layout
against its actual bytes on disk (never trusting a tensor list a file's
bytes do not back up), sums the *file sizes* of the safetensors shards,
and compares the result -- plus an operator-declared KV-cache reserve --
against a wired-memory ceiling. It is a drop-in ``--fit-check-bin`` for
``scripts/fastmlx_launch.py``, using the same ceiling rule as
``scripts/fastmlx_gguf_fit.py``: exit 0 with an attestation line means
GREEN (fits), exit 2 means RED (does not fit, with a reason on stderr),
exit 64 is a usage error, and any other exit code (this binary uses 1) is
a format/configuration error -- never a fit verdict. A ``--json`` mode
prints the same fields as a JSON object for other consumers.

``--wired-limit-mib``, ``--wired-margin-gib``, and ``--kv-reserve-gib``
each accept an environment-variable fallback with the exact same names
and precedence (flag, then environment variable, then a computed
default) as ``scripts/fastmlx_gguf_fit.py``: ``FASTMLX_WIRED_LIMIT_MIB``,
``FASTMLX_WIRED_MARGIN_GIB``, and ``FASTMLX_GGUF_KV_RESERVE_GIB`` (the
KV-reserve variable is shared with the GGUF sizer on purpose -- a launch
that sets it once covers whichever sizer ends up on the command line). A
KV-cache reserve given by neither the flag nor the environment variable
is a usage error (exit 64), never a silent zero default.

GREEN is a memory verdict only, not a loadability verdict: this binary
never inspects, refuses on, or reasons about which tensor dtypes or
operators a serving engine implements, so a pack can be fit-GREEN here
and still fail to load under a given engine. Loadability is the engine's
own concern, checked at load time. GREEN is not a quality verdict
either: it says only that the resident weight bytes plus the declared
KV-cache reserve fit under the wired-memory ceiling.

This binary knows nothing about the identity or family of the model it
is checking: it sizes resident weights purely from the safetensors files
present in the pack directory, never by model name, architecture, or any
per-family convention.

Memory-mapped side files: a safetensors pack directory can also contain
large non-safetensors side files that a serving engine memory-maps
rather than loads resident (for example, a large n-gram lookup table
used by a speculative-decoding auxiliary model). Such a file must be
named explicitly with a repeatable ``--mmap-side-file RELPATH`` flag
(the path relative to ``--model-path``); it is then excluded from
resident weight bytes and reported separately (``memory_mapped_side_files``
in ``--json``, and a note on stderr) so an operator can see it was seen
and deliberately excluded, not silently missed. A ``--mmap-side-file``
that does not exist under the pack is a configuration error (exit 1). Any
OTHER non-safetensors regular file that is at least 1 GiB, found anywhere
in the pack and not named by ``--mmap-side-file``, is also a
configuration error (exit 1): silently excluding an unnamed multi-GiB
file would let a pack this checker cannot size pass as GREEN. If a
top-level ``model.safetensors.index.json`` is present, every shard named
in its ``weight_map`` must exist and be among the counted
``*.safetensors`` shards, or this is also a configuration error.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import struct
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------
# Shared helpers, imported from scripts/fastmlx_gguf_fit.py by file path
# (not by package import) so this binary works whether it is invoked
# directly via its shebang, via `python3 fastmlx_safetensors_fit.py`, or
# loaded by tests via importlib -- none of which guarantee `scripts` is
# on sys.path as an importable package.
# ---------------------------------------------------------------------
_GGUF_FIT_PATH = Path(__file__).resolve().parent / "fastmlx_gguf_fit.py"
_GGUF_FIT_SPEC = importlib.util.spec_from_file_location(
    "_fastmlx_safetensors_fit_shared_gguf_helpers", _GGUF_FIT_PATH
)
assert _GGUF_FIT_SPEC is not None and _GGUF_FIT_SPEC.loader is not None
_GGUF = importlib.util.module_from_spec(_GGUF_FIT_SPEC)
_GGUF_FIT_SPEC.loader.exec_module(_GGUF)

GIB = _GGUF.GIB
MIB = _GGUF.MIB

# sysexits.h EX_USAGE, reused from the GGUF sizer's own protocol: a bad
# invocation must never be misread by scripts/fastmlx_launch.py's
# run_fit_check() as "does not fit" (exit 2) or "fits" (exit 0).
_EXIT_USAGE_ERROR = 64

# Header length is read with a single bounded read, never the tensor data
# section (files here are commonly tens of gigabytes). A sane upper cap
# on the declared header length guards against a corrupt or hostile
# length value causing an oversized in-memory JSON parse.
_SAFETENSORS_HEADER_LEN_CAP_BYTES = 100 * MIB

# safetensors dtype -> byte size per element. Unknown/unsupported dtypes
# must error, never be treated as size 0 or silently skipped.
_DTYPE_SIZES = {
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "U16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}

# A non-safetensors regular file is only worth reporting/refusing over as
# a "memory-mapped side file" once it is large enough that silently
# ignoring it (or wrongly counting it) would matter to an operator
# reading the fit check's reasoning.
_SIDE_FILE_MIN_BYTES = GIB

# Hostile-shape guards for a tensor's declared "shape" list: a real model
# tensor never has more than a handful of dimensions, and a legitimate
# element count can never exceed the file's own byte size (every dtype
# here is at least 1 byte/element). Both are cheap, load-bearing checks
# against a header whose dimension list is absurdly long or whose element
# count balloons past anything the file could possibly hold.
_MAX_TENSOR_DIMS = 16


class SafetensorsFormatError(ValueError):
    """A malformed, truncated, or inconsistent safetensors file -- never a fit verdict."""


class FitCheckError(RuntimeError):
    """A configuration/environment problem (bad flags, missing/empty pack) -- never a fit verdict."""


# ---------------------------------------------------------------------
# safetensors header parsing + reconciliation
# ---------------------------------------------------------------------
def parse_safetensors_header(path: Path) -> dict:
    """Reads only the 8-byte header-length prefix and the declared header
    JSON bytes -- never the tensor data section that follows."""
    label = str(path)
    with open(path, "rb") as f:
        file_size = os.fstat(f.fileno()).st_size
        if file_size < 8:
            raise SafetensorsFormatError(
                f"{label}: truncated file ({file_size} byte(s), need at least "
                "8 for the header-length prefix)"
            )
        len_bytes = f.read(8)
        if len(len_bytes) < 8:
            raise SafetensorsFormatError(f"{label}: truncated file (short read of header length)")
        header_len = struct.unpack("<Q", len_bytes)[0]
        if header_len > _SAFETENSORS_HEADER_LEN_CAP_BYTES:
            raise SafetensorsFormatError(
                f"{label}: declared header length {header_len} exceeds the sane "
                f"cap of {_SAFETENSORS_HEADER_LEN_CAP_BYTES} bytes"
            )
        if 8 + header_len > file_size:
            raise SafetensorsFormatError(
                f"{label}: declared header length {header_len} plus the 8-byte "
                f"prefix ({8 + header_len} bytes) exceeds the file size "
                f"({file_size} bytes); truncated file"
            )
        raw = f.read(header_len)
        if len(raw) < header_len:
            raise SafetensorsFormatError(
                f"{label}: truncated file (needed {header_len} header byte(s), "
                f"got {len(raw)})"
            )

    try:
        header_text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SafetensorsFormatError(f"{label}: header is not valid utf-8: {exc}") from None
    try:
        entries = json.loads(header_text)
    except json.JSONDecodeError as exc:
        raise SafetensorsFormatError(f"{label}: header is not valid JSON: {exc}") from None
    if not isinstance(entries, dict):
        raise SafetensorsFormatError(f"{label}: header JSON is not an object")

    return {
        "path": path,
        "label": label,
        "file_size": file_size,
        "header_len": header_len,
        "entries": entries,
    }


def _tensor_entries(header: dict) -> list:
    """Validates and returns (name, begin, end) for every real tensor
    entry (skipping the optional ``__metadata__`` key), checking dtype
    and shape against the declared data_offsets byte length."""
    label = header["label"]
    file_size = header["file_size"]
    tensors = []
    for name, meta in header["entries"].items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict):
            raise SafetensorsFormatError(f"{label}: tensor entry {name!r} is not an object")

        for key in ("dtype", "shape", "data_offsets"):
            if key not in meta:
                raise SafetensorsFormatError(
                    f"{label}: tensor entry {name!r} is missing required key {key!r}"
                )

        dtype = meta["dtype"]
        dtype_size = _DTYPE_SIZES.get(dtype)
        if dtype_size is None:
            raise SafetensorsFormatError(
                f"{label}: tensor entry {name!r} has unknown/unsupported dtype {dtype!r}"
            )

        shape = meta["shape"]
        if not isinstance(shape, list) or not all(
            isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape
        ):
            raise SafetensorsFormatError(
                f"{label}: tensor entry {name!r} has an invalid shape {shape!r}"
            )
        if len(shape) > _MAX_TENSOR_DIMS:
            raise SafetensorsFormatError(
                f"{label}: tensor entry {name!r} has {len(shape)} dimensions, "
                f"exceeding the sane cap of {_MAX_TENSOR_DIMS}"
            )
        numel = 1
        for dim in shape:
            numel *= dim
            if numel > file_size:
                raise SafetensorsFormatError(
                    f"{label}: tensor entry {name!r} shape {shape!r} implies "
                    f"an element count that already exceeds the file size "
                    f"({file_size} byte(s)) partway through the shape"
                )

        offsets = meta["data_offsets"]
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in offsets)
        ):
            raise SafetensorsFormatError(
                f"{label}: tensor entry {name!r} has invalid data_offsets {offsets!r} "
                "(must be a 2-element integer list)"
            )
        begin, end = offsets
        if begin < 0 or end < begin:
            raise SafetensorsFormatError(
                f"{label}: tensor entry {name!r} has invalid data_offsets "
                f"[{begin}, {end}]"
            )

        expected_len = numel * dtype_size
        actual_len = end - begin
        if actual_len != expected_len:
            raise SafetensorsFormatError(
                f"{label}: tensor entry {name!r} data_offsets span "
                f"{actual_len} byte(s) but shape {shape} x dtype {dtype} "
                f"({dtype_size} byte(s)/element) requires {expected_len} byte(s)"
            )

        tensors.append({"name": name, "begin": begin, "end": end})
    return tensors


def reconcile_safetensors(header: dict) -> int:
    """Validates one safetensors file's tensor layout against its actual
    file size.

    Unlike GGUF, safetensors has no inter-tensor alignment padding: the
    spec requires tensors to be packed contiguously in ``data_offsets``
    order with the first tensor's ``begin`` at 0 and every next tensor's
    ``begin`` equal to the previous tensor's ``end`` exactly, and the
    file size must equal ``8 + header_len + highest_end`` exactly (no
    slack). A gap, an overlap, or trailing bytes the header does not
    account for are all rejected; dtype/shape mismatches against the
    declared byte span are caught in ``_tensor_entries``.

    Returns the reconciled ``highest_end`` (informational only -- the
    weight-byte accounting used by this checker is each file's whole
    on-disk size, not the sum of reconciled tensor spans).
    """
    label = header["label"]
    file_size = header["file_size"]
    header_len = header["header_len"]
    data_start = 8 + header_len

    tensors = _tensor_entries(header)
    ordered = sorted(tensors, key=lambda t: t["begin"])

    expected = 0
    highest_end = 0
    previous = None
    for entry in ordered:
        name, begin, end = entry["name"], entry["begin"], entry["end"]
        if begin != expected:
            if previous is None:
                raise SafetensorsFormatError(
                    f"{label}: tensor {name!r} begins at offset {begin}, but "
                    "the data section must begin at offset 0"
                )
            raise SafetensorsFormatError(
                f"{label}: tensor {name!r} begins at offset {begin}, but "
                f"tensor {previous['name']!r} ends at {previous['end']} and "
                f"the next tensor was expected at offset {expected}; "
                "safetensors data_offsets must be packed contiguously with "
                "no gap and no overlap"
            )
        expected = end
        highest_end = end
        previous = entry

    if data_start + highest_end != file_size:
        raise SafetensorsFormatError(
            f"{label}: reconciled tensor data ends at byte {highest_end} but "
            f"the file's data section is {file_size - data_start} byte(s) "
            f"(file size {file_size}, header {data_start} byte(s)); the "
            "tensor list does not account for every byte on disk"
        )

    return highest_end


# ---------------------------------------------------------------------
# Pack-directory discovery (recursive; dot-directories are never scanned)
# ---------------------------------------------------------------------
_SAFETENSORS_INDEX_FILENAME = "model.safetensors.index.json"


def _iter_regular_files(model_path: Path):
    """Yields ``(relative_posix, Path)`` for every regular file under
    ``model_path``, recursively.

    Any entry whose name starts with ``.`` is skipped, and if it is a
    directory it is never descended into (so ``.cache/``, a dotfile, etc.
    are invisible to this scan at any depth). A symlink -- file or
    directory -- is never followed and never counted (this checker only
    sizes real on-disk bytes it can independently verify). Each directory
    level is visited in sorted-name order for deterministic output.
    """
    stack = [model_path]
    while stack:
        current = stack.pop()
        try:
            children = sorted(os.scandir(current), key=lambda e: e.name)
        except OSError as error:
            raise FitCheckError(f"could not list {current}: {error}") from error
        for child in children:
            if child.name.startswith("."):
                continue
            if child.is_symlink():
                continue
            child_path = Path(child.path)
            if child.is_dir(follow_symlinks=False):
                stack.append(child_path)
                continue
            if child.is_file(follow_symlinks=False):
                yield child_path.relative_to(model_path).as_posix(), child_path


def _check_safetensors_index(model_path: Path, weight_by_relative: dict) -> None:
    """If a top-level ``model.safetensors.index.json`` is present, every
    shard filename its ``weight_map`` names must exist and be among the
    ``*.safetensors`` shards this checker actually counted -- otherwise a
    renamed, moved, or missing shard could vouch for weights that were
    never sized."""
    index_path = model_path / _SAFETENSORS_INDEX_FILENAME
    if not index_path.is_file():
        return
    try:
        index_text = index_path.read_text(encoding="utf-8")
    except OSError as error:
        raise FitCheckError(f"could not read {index_path}: {error}") from error
    try:
        index_doc = json.loads(index_text)
    except json.JSONDecodeError as error:
        raise FitCheckError(f"{index_path}: not valid JSON: {error}") from error
    if not isinstance(index_doc, dict) or not isinstance(index_doc.get("weight_map"), dict):
        raise FitCheckError(f"{index_path}: missing or invalid 'weight_map' object")

    named_shards = sorted({str(name) for name in index_doc["weight_map"].values()})
    for shard_name in named_shards:
        if shard_name not in weight_by_relative:
            raise FitCheckError(
                f"{index_path}: weight_map names shard {shard_name!r}, which "
                "is not among the *.safetensors shards this checker found "
                f"and counted under {model_path}"
            )


def compute_model_bytes(model_path: Path, mmap_side_files: Optional[list] = None) -> tuple:
    """Returns ``(weight_bytes, weight_files, side_files)``.

    ``weight_bytes`` is the sum of the *file sizes* of every
    ``*.safetensors`` shard found anywhere under ``model_path`` (each
    individually header-parsed and reconciled first, so a truncated or
    inconsistent shard errors out rather than silently contributing a
    wrong byte count). ``weight_files``/``side_files`` are
    ``[{"name": ..., "bytes": ...}]`` lists, ``name`` being the path
    relative to ``model_path`` (POSIX separators), never a full path.

    ``mmap_side_files`` (relative paths, matching ``--mmap-side-file``)
    are excluded from resident weight bytes and reported in
    ``side_files``; each must actually exist under ``model_path``, or
    this raises. Any OTHER non-safetensors regular file that is at least
    1 GiB and not named by ``mmap_side_files`` is a configuration error:
    silently excluding it (the old behavior) could let a pack this
    checker cannot size pass as GREEN.
    """
    if not model_path.is_dir():
        raise FitCheckError(f"model path {model_path} does not exist or is not a directory")

    requested_side_files = list(mmap_side_files or [])
    requested_side_file_set = set(requested_side_files)

    weight_bytes = 0
    weight_files = []
    weight_by_relative: dict = {}
    side_files = []
    seen_relative_paths: set = set()

    for relative, path in _iter_regular_files(model_path):
        seen_relative_paths.add(relative)
        if relative.endswith(".safetensors"):
            header = parse_safetensors_header(path)
            reconcile_safetensors(header)
            size = header["file_size"]
            weight_bytes += size
            weight_files.append({"name": relative, "bytes": size})
            weight_by_relative[relative] = size
            continue

        size = path.stat().st_size
        if relative in requested_side_file_set:
            side_files.append({"name": relative, "bytes": size})
            continue
        if size >= _SIDE_FILE_MIN_BYTES:
            raise FitCheckError(
                f"{model_path}: {relative} is a non-safetensors file of "
                f"{size} bytes (>= {_SIDE_FILE_MIN_BYTES} byte(s), i.e. "
                ">= 1 GiB) that is not named by --mmap-side-file; pass "
                f"--mmap-side-file {relative} if the serving engine "
                "memory-maps it, or this pack holds weights this sizer "
                "cannot size"
            )

    if not weight_files:
        raise FitCheckError(f"no .safetensors files found under {model_path}")

    for requested in requested_side_files:
        if requested not in seen_relative_paths:
            raise FitCheckError(
                f"--mmap-side-file {requested} does not exist under {model_path}"
            )

    _check_safetensors_index(model_path, weight_by_relative)

    return weight_bytes, weight_files, side_files


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = _GGUF._UsageErrorArgumentParser(
        prog="fastmlx-safetensors-fit",
        description=(
            "Safetensors-aware, residency-aware pre-load fit checker. GREEN "
            "is a memory verdict only, not a loadability or quality "
            "verdict; this checker knows nothing about model identity or "
            "family. Resident weight bytes are the sum of each "
            "*.safetensors shard's actual on-disk size (reconciled "
            "against its own header), found anywhere under --model-path "
            "except inside a dot-prefixed directory. A large "
            "non-safetensors file must be named with --mmap-side-file to "
            "be excluded from resident bytes and reported separately; an "
            "unnamed one that is >= 1 GiB is a configuration error."
        ),
    )
    # Accepted for drop-in --fit-check-bin compatibility with
    # scripts/fastmlx_launch.py's run_fit_check(); validated where a
    # validation is cheap and meaningful, otherwise stored and unused.
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument(
        "--host-use", default="shared", choices=["shared", "dedicated-serving"]
    )
    parser.add_argument("--fit-check-only", action="store_true")
    parser.add_argument("--context", type=int, default=None)

    parser.add_argument("--residency", choices=["resident", "expert-stream"], default=None)
    parser.add_argument(
        "--wired-limit-mib", type=_GGUF._positive_int_type("--wired-limit-mib"), default=None
    )
    parser.add_argument("--wired-margin-gib", type=_GGUF._wired_margin_gib_type, default=None)
    # No `required=True`: a flag, then the shared FASTMLX_GGUF_KV_RESERVE_GIB
    # environment variable (see module docstring), then a usage error (exit
    # 64, checked explicitly in main()) -- a KV-cache reserve must never be
    # silently defaulted to zero.
    parser.add_argument("--kv-reserve-gib", type=float, default=None)
    parser.add_argument(
        "--mmap-side-file",
        action="append",
        default=[],
        metavar="RELPATH",
        dest="mmap_side_file",
        help=(
            "a path, relative to --model-path, of a large non-safetensors "
            "file the serving engine memory-maps rather than loads "
            "resident (e.g. an n-gram table); repeatable. Excludes it from "
            "resident weight bytes. Any other non-safetensors file >= 1 "
            "GiB anywhere in the pack that is not named here is a "
            "configuration error (exit 1)."
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser


class MissingKvReserveError(FitCheckError):
    """``--kv-reserve-gib`` was given by neither the flag nor the shared
    environment variable -- a usage error (exit 64), not a configuration
    error (exit 1)."""


def compute_fit(args: argparse.Namespace) -> dict:
    residency = args.residency or "resident"

    kv_reserve_gib = args.kv_reserve_gib
    if kv_reserve_gib is None:
        kv_reserve_gib = _GGUF._env_float(_GGUF.ENV_KV_RESERVE_GIB)
    if kv_reserve_gib is None:
        raise MissingKvReserveError(
            "--kv-reserve-gib is required (pass --kv-reserve-gib or set "
            f"{_GGUF.ENV_KV_RESERVE_GIB}); a fit check must never silently "
            "assume zero KV-cache reserve"
        )
    if kv_reserve_gib < 0:
        raise FitCheckError("--kv-reserve-gib must be >= 0")

    wired_margin_gib = args.wired_margin_gib
    if wired_margin_gib is None:
        wired_margin_gib = _GGUF._env_int(_GGUF.ENV_WIRED_MARGIN_GIB)
    if wired_margin_gib is None:
        wired_margin_gib = 8
    if not (2 <= wired_margin_gib <= 32):
        raise FitCheckError("--wired-margin-gib must be an integer between 2 and 32")

    wired_limit_mib, wired_limit_source = _GGUF.resolve_wired_limit_mib(
        args.wired_limit_mib, _GGUF._env_int(_GGUF.ENV_WIRED_LIMIT_MIB)
    )

    weight_bytes, weight_files, side_files = compute_model_bytes(
        args.model_path, getattr(args, "mmap_side_file", None)
    )

    kv_reserve_bytes = int(round(kv_reserve_gib * GIB))
    ceiling_bytes = wired_limit_mib * MIB - wired_margin_gib * GIB
    if ceiling_bytes <= 0:
        raise FitCheckError(
            "the wired-memory ceiling is not positive: --wired-limit-mib "
            f"{wired_limit_mib} ({wired_limit_mib * MIB} bytes) minus "
            f"--wired-margin-gib {wired_margin_gib} ({wired_margin_gib * GIB} "
            f"bytes) leaves a ceiling of {ceiling_bytes} bytes; this is a "
            "configuration error, not a fit verdict -- raise "
            "--wired-limit-mib or lower --wired-margin-gib"
        )
    total_bytes = weight_bytes + kv_reserve_bytes

    if total_bytes <= ceiling_bytes:
        fit = "green"
        reason = "fits"
        reason_text = f"total {total_bytes} bytes fits within ceiling {ceiling_bytes} bytes"
    else:
        fit = "red"
        reason = "total_exceeds_ceiling"
        over_by = total_bytes - ceiling_bytes
        reason_text = (
            f"total {total_bytes} bytes exceeds ceiling {ceiling_bytes} bytes "
            f"(over by {over_by} bytes)"
        )

    return {
        "weights_bytes": weight_bytes,
        "kv_reserve_bytes": kv_reserve_bytes,
        "total_bytes": total_bytes,
        "ceiling_bytes": ceiling_bytes,
        "wired_limit_source": wired_limit_source,
        "residency": residency,
        "fit": fit,
        "reason": reason,
        "reason_text": reason_text,
        "context": args.context,
        "weight_files": weight_files,
        "memory_mapped_side_files": side_files,
    }


def format_attestation_line(result: dict) -> str:
    return (
        f"{_GGUF._ATTESTATION_PREFIX} fit_check_only=complete weights_loaded=false "
        f"weights_bytes={result['weights_bytes']} "
        f"kv_reserve_bytes={result['kv_reserve_bytes']} "
        f"total_bytes={result['total_bytes']} "
        f"ceiling_bytes={result['ceiling_bytes']} "
        f"wired_limit_source={result['wired_limit_source']} "
        f"residency={result['residency']} "
        f"fit={result['fit']} "
        f"reason={result['reason']}"
    )


def _print_side_file_notes(side_files: list) -> None:
    for side_file in side_files:
        print(
            f"fastmlx-safetensors-fit: note: {side_file['name']} "
            f"({side_file['bytes']} bytes) is a memory-mapped side file, "
            "not counted as resident (engine maps them)",
            file=sys.stderr,
        )


def main(argv: Optional[list] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.residency == "expert-stream":
        parser.error(
            "safetensors packs have no expert-stream sizing (resident "
            "weights only); omit --residency or pass --residency resident"
        )

    try:
        result = compute_fit(args)
    except MissingKvReserveError as exc:
        # A usage error (exit 64), not a configuration error (exit 1): the
        # equivalent of argparse's own `required=True` handling, done by
        # hand here because presence is only knowable after checking the
        # environment-variable fallback too (see module docstring).
        parser.print_usage(sys.stderr)
        sys.stderr.write(f"{parser.prog}: error: {exc}\n")
        return _EXIT_USAGE_ERROR
    except (FitCheckError, SafetensorsFormatError, _GGUF.FitCheckError) as exc:
        print(f"fastmlx-safetensors-fit: error: {exc}", file=sys.stderr)
        return 1

    _print_side_file_notes(result["memory_mapped_side_files"])

    if args.json:
        payload = dict(result)
        payload["reason"] = result["reason_text"]
        del payload["reason_text"]
        print(json.dumps(payload, sort_keys=True))
    else:
        print(format_attestation_line(result))

    if result["fit"] == "green":
        return 0

    print(result["reason_text"], file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
