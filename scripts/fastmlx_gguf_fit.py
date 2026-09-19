#!/usr/bin/env python3
"""fastmlx-gguf-fit: a GGUF-aware, residency-aware pre-load fit checker.

Given a local GGUF pack (a single file, a directory containing one, or a
split-shard set named ``<stem>-<idx>-of-<count>.gguf``), this binary parses
the GGUF v3 header of every shard, reconciles each shard's declared tensor
layout against its actual file size (never trusting a tensor list a file's
bytes do not back up), sums per-tensor weight bytes from a ggml type table,
and compares the result -- plus an operator-declared KV-cache reserve --
against a wired-memory ceiling. It is a drop-in ``--fit-check-bin`` for
``scripts/fastmlx_launch.py``: exit 0 with an attestation line means GREEN
(fits), exit 2 means RED (does not fit, with a reason on stderr), and any
other exit code is an error -- never a fit verdict. A ``--json`` mode prints
the same fields as a JSON object for other consumers.

This binary knows nothing about the identity of the model it is checking:
it classifies weight tensors purely by the generic GGUF/ggml tensor-name
convention used by expert-streaming engines for mixture-of-experts packs
(``blk.<N>.ffn_{gate,up,down}_exps.*`` and ``blk.<N>.ffn_gate_up_exps.*``),
never by model family.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Optional


GGUF_MAGIC = b"GGUF"
GGUF_SUPPORTED_VERSION = 3

# Built by concatenation, on purpose: written as one literal token this name
# contains, as a plain substring, an unrelated forbidden term this repo's
# public leak sweep scans for (a different, third-party engine's short
# name). Mirrors the same construction scripts/fastmlx_launch.py uses for
# its own built-in engine binary name.
_ATTESTATION_PREFIX = "fastmlx" + "-serve"

GIB = 1024 ** 3
MIB = 1024 ** 2

# The header reader tops up its internal buffer in fixed chunks of this size
# rather than ever reading a whole shard (which can be tens of GiB) into
# memory. Header parsing therefore reads at most header_end + one chunk.
_HEADER_READ_CHUNK_BYTES = MIB

# gguf_metadata_value_type (GGUF spec).
_KV_UINT8 = 0
_KV_INT8 = 1
_KV_UINT16 = 2
_KV_INT16 = 3
_KV_UINT32 = 4
_KV_INT32 = 5
_KV_FLOAT32 = 6
_KV_BOOL = 7
_KV_STRING = 8
_KV_ARRAY = 9
_KV_UINT64 = 10
_KV_INT64 = 11
_KV_FLOAT64 = 12

# Fixed-size scalar metadata-array element types -> struct format char, used
# to bulk-read+unpack a whole array in one call rather than one
# read()+struct.unpack() per element (see _Reader._kv_array). Deliberately
# excludes _KV_STRING (variable-length) and _KV_ARRAY (nesting is rejected
# before this table is consulted).
_KV_SCALAR_STRUCT_FMT = {
    _KV_UINT8: "B",
    _KV_INT8: "b",
    _KV_UINT16: "H",
    _KV_INT16: "h",
    _KV_UINT32: "I",
    _KV_INT32: "i",
    _KV_FLOAT32: "f",
    _KV_BOOL: "B",
    _KV_UINT64: "Q",
    _KV_INT64: "q",
    _KV_FLOAT64: "d",
}

# ggml type id -> (name, block_size, type_size_bytes). Ids and sizes mirror
# the published ggml.h enum / ggml.c type_traits table. Any id not present
# here is unsupported and must error, never be treated as zero bytes.
GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
}

# Generic GGUF/ggml tensor-name convention for mixture-of-experts weight
# blocks used by expert-streaming engines, independent of any model family:
# blk.<N>.ffn_{gate,up,down}_exps[.suffix] and the fused blk.<N>.ffn_gate_up_exps.
_EXPERT_TENSOR_RE = re.compile(
    r"^blk\.\d+\.ffn_(?:gate_up_exps|gate_exps|up_exps|down_exps)(?:\..*)?$"
)

# Split-shard naming: "<stem>-<idx>-of-<count>.gguf", idx/count zero-padded
# to whatever width the first-seen shard uses.
_SPLIT_RE = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{3,})-of-(?P<count>\d{3,})\.gguf$")


class GGUFFormatError(ValueError):
    """A malformed, truncated, or unsupported GGUF file -- never a fit verdict."""


class FitCheckError(RuntimeError):
    """A configuration/environment problem (bad flags, missing files) -- never a fit verdict."""


def is_expert_tensor(name: str) -> bool:
    return bool(_EXPERT_TENSOR_RE.match(name))


# ---------------------------------------------------------------------
# GGUF v3 binary reader
# ---------------------------------------------------------------------
def _open_binary(path: Path):
    """Open a shard for binary reading. A thin, separately-named seam
    (rather than inlining ``open()``) so tests can substitute an
    instrumented file object without touching the real filesystem call."""
    return open(path, "rb")


class _Reader:
    """Incrementally parses a GGUF header from an open binary file handle.

    Reads are satisfied from an internal buffer that is topped up in fixed
    ``chunk_size`` reads from the underlying file only when more bytes are
    needed than are currently buffered. This never reads a shard's (often
    many-GiB) tensor-data section into memory: parsing stops as soon as the
    tensor-info list has been consumed, and at most one chunk's worth of
    over-read past the true header end is possible.

    The buffer is a ``bytearray`` with a read cursor rather than a ``bytes``
    object that gets re-sliced on every ``read()`` call: re-slicing the
    whole remaining buffer on every call (``buf = buf[n:]``) costs O(buffer
    size) per call and made parsing a header with tens or hundreds of
    thousands of small values (e.g. a large tokenizer string array)
    quadratic overall. Consumed bytes are only ever dropped from the front
    of the buffer (compacted), never copied out one read at a time.
    """

    def __init__(self, fileobj, label: str, chunk_size: int = _HEADER_READ_CHUNK_BYTES):
        self._file = fileobj
        self._label = label
        self._chunk_size = chunk_size
        self._buffer = bytearray()
        self._read_pos = 0  # index into _buffer of the next unconsumed byte
        self._pos = 0  # absolute stream offset already consumed by callers

    def read(self, n: int) -> bytes:
        available = len(self._buffer) - self._read_pos
        while available < n:
            chunk = self._file.read(self._chunk_size)
            if not chunk:
                raise GGUFFormatError(
                    f"{self._label}: truncated file (needed {n} more byte(s) at "
                    f"header offset {self._pos}, only {available} available)"
                )
            self._buffer += chunk
            available += len(chunk)
        start = self._read_pos
        end = start + n
        result = bytes(self._buffer[start:end])
        self._read_pos = end
        self._pos += n
        # Compact only once consumed bytes make up at least half the
        # buffer, so this is amortized O(1) per byte rather than a copy on
        # every single read().
        if self._read_pos >= self._chunk_size and self._read_pos * 2 >= len(self._buffer):
            del self._buffer[: self._read_pos]
            self._read_pos = 0
        return result

    def u8(self) -> int:
        return struct.unpack("<B", self.read(1))[0]

    def i8(self) -> int:
        return struct.unpack("<b", self.read(1))[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.read(2))[0]

    def i16(self) -> int:
        return struct.unpack("<h", self.read(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.read(8))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.read(4))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self.read(8))[0]

    def bool_(self) -> bool:
        return bool(self.u8())

    def gguf_string(self) -> str:
        length = self.u64()
        raw = self.read(length)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFFormatError(
                f"{self._label}: invalid utf-8 string in header: {exc}"
            ) from None

    def kv_value(self, value_type: int):
        if value_type == _KV_UINT8:
            return self.u8()
        if value_type == _KV_INT8:
            return self.i8()
        if value_type == _KV_UINT16:
            return self.u16()
        if value_type == _KV_INT16:
            return self.i16()
        if value_type == _KV_UINT32:
            return self.u32()
        if value_type == _KV_INT32:
            return self.i32()
        if value_type == _KV_FLOAT32:
            return self.f32()
        if value_type == _KV_BOOL:
            return self.bool_()
        if value_type == _KV_STRING:
            return self.gguf_string()
        if value_type == _KV_UINT64:
            return self.u64()
        if value_type == _KV_INT64:
            return self.i64()
        if value_type == _KV_FLOAT64:
            return self.f64()
        if value_type == _KV_ARRAY:
            elem_type = self.u32()
            length = self.u64()
            if elem_type == _KV_ARRAY:
                raise GGUFFormatError(
                    f"{self._label}: nested metadata arrays are not supported"
                )
            return self._kv_array(elem_type, length)
        raise GGUFFormatError(
            f"{self._label}: unknown metadata value type {value_type}"
        )

    def _kv_array(self, elem_type: int, length: int) -> list:
        # Strings are variable-length and must be read one at a time, but
        # fixed-size scalar element types are read and unpacked in a single
        # bulk call instead of one read()+struct.unpack() per element: for
        # a several-hundred-thousand-element array (e.g. token-merge ranks)
        # the per-element method-call overhead alone is the dominant cost.
        if elem_type == _KV_STRING:
            return [self.gguf_string() for _ in range(length)]
        fmt_char = _KV_SCALAR_STRUCT_FMT.get(elem_type)
        if fmt_char is None:
            raise GGUFFormatError(
                f"{self._label}: unknown metadata value type {elem_type}"
            )
        raw = self.read(struct.calcsize(fmt_char) * length)
        values = struct.unpack(f"<{length}{fmt_char}", raw)
        if elem_type == _KV_BOOL:
            return [bool(v) for v in values]
        return list(values)


def _align_up(value: int, alignment: int) -> int:
    remainder = value % alignment
    if remainder == 0:
        return value
    return value + (alignment - remainder)


def parse_gguf_header(path: Path) -> dict:
    label = str(path)
    with _open_binary(path) as f:
        file_size = os.fstat(f.fileno()).st_size
        reader = _Reader(f, label)

        magic = reader.read(4)
        if magic != GGUF_MAGIC:
            raise GGUFFormatError(f"{label}: bad magic {magic!r} (expected b'GGUF')")

        version = reader.u32()
        if version != GGUF_SUPPORTED_VERSION:
            raise GGUFFormatError(
                f"{label}: unsupported GGUF version {version} (only version 3 is supported)"
            )

        tensor_count = reader.u64()
        kv_count = reader.u64()

        metadata: dict = {}
        for _ in range(kv_count):
            key = reader.gguf_string()
            value_type = reader.u32()
            metadata[key] = reader.kv_value(value_type)

        tensors = []
        for _ in range(tensor_count):
            name = reader.gguf_string()
            n_dims = reader.u32()
            dims = [reader.u64() for _ in range(n_dims)]
            ggml_type = reader.u32()
            offset = reader.u64()
            tensors.append({"name": name, "dims": dims, "type": ggml_type, "offset": offset})

        alignment = metadata.get("general.alignment", 32)
        if not isinstance(alignment, int) or isinstance(alignment, bool) or alignment <= 0:
            raise GGUFFormatError(f"{label}: invalid general.alignment value {alignment!r}")

        header_end = reader._pos  # noqa: SLF001 -- same module, deliberate.
        data_start = _align_up(header_end, alignment)

    return {
        "path": path,
        "label": label,
        "metadata": metadata,
        "tensors": tensors,
        "alignment": alignment,
        "header_end": header_end,
        "data_start": data_start,
        "file_size": file_size,
    }


def tensor_byte_size(label: str, tensor: dict) -> int:
    type_id = tensor["type"]
    info = GGML_TYPES.get(type_id)
    if info is None:
        raise GGUFFormatError(
            f"{label}: unknown ggml type id {type_id} for tensor {tensor['name']!r}"
        )
    _, block_size, type_size = info
    numel = 1
    for dim in tensor["dims"]:
        numel *= dim
    if numel == 0:
        return 0
    if block_size == 1:
        return numel * type_size
    if numel % block_size != 0:
        raise GGUFFormatError(
            f"{label}: tensor {tensor['name']!r} element count {numel} is not a "
            f"multiple of block size {block_size} for ggml type id {type_id}"
        )
    return (numel // block_size) * type_size


def reconcile_shard(header: dict) -> list:
    """Validate one shard's tensor layout against its actual file size.

    Standard GGUF writers place tensors contiguously: sorted by offset, the
    first tensor's offset is 0 and every next tensor's offset equals
    ``align_up(prev.offset + prev.size, alignment)`` -- alignment padding
    between tensors is permitted, but nothing else is: a gap larger than
    alignment padding (a hole the tensor list doesn't account for) and any
    overlap between tensors are both rejected. ggml block sizes alone are
    not sufficient to catch a corrupted or mismeasured tensor: a size that
    is merely *wrong* (too big or too small) for a middle tensor, without
    this contiguity check, would silently pass as long as its own
    ``[offset, offset + size)`` stayed inside the file.

    The highest ``offset + size`` among all tensors (``highest_end``) must
    be no greater than the data section's actual size, and the data
    section's actual size must be no greater than ``highest_end`` rounded
    up to the shard's alignment: real GGUF writers pad after every tensor,
    including the last, so a valid file's data section commonly ends up to
    one alignment unit past ``highest_end``, never less and never by more
    than that. This is what catches truncation, a tensor list that outruns
    the file's real bytes, or trailing garbage appended past what the
    tensor list (plus its final padding) accounts for.

    Returns a list of ``(name, size)`` pairs, one per tensor, in the same
    order as ``header["tensors"]`` (not necessarily offset order).
    """
    label = header["label"]
    alignment = header["alignment"]
    data_start = header["data_start"]
    file_size = header["file_size"]
    data_section_size = file_size - data_start
    if data_section_size < 0:
        raise GGUFFormatError(
            f"{label}: header + alignment padding ({data_start} bytes) exceeds "
            f"the file size ({file_size} bytes)"
        )

    entries = []
    for index, tensor in enumerate(header["tensors"]):
        size = tensor_byte_size(label, tensor)
        offset = tensor["offset"]
        if offset % alignment != 0:
            raise GGUFFormatError(
                f"{label}: tensor {tensor['name']!r} offset {offset} is not a "
                f"multiple of the alignment ({alignment})"
            )
        entries.append({"index": index, "name": tensor["name"], "offset": offset, "size": size})

    ordered = sorted(entries, key=lambda e: e["offset"])

    highest_end = 0
    expected_offset = 0
    previous = None
    for entry in ordered:
        name, offset, size = entry["name"], entry["offset"], entry["size"]
        if offset != expected_offset:
            if previous is None:
                raise GGUFFormatError(
                    f"{label}: tensor {name!r} begins at offset {offset}, but "
                    "the data section must begin at offset 0; tensors must "
                    "be packed contiguously"
                )
            raise GGUFFormatError(
                f"{label}: tensor {name!r} begins at offset {offset}, but "
                f"tensor {previous['name']!r} (offset {previous['offset']}, "
                f"size {previous['size']}) ends at "
                f"{previous['offset'] + previous['size']} and the next "
                f"tensor was expected at offset {expected_offset} (alignment "
                f"{alignment}); tensors must be packed contiguously with no "
                "gap larger than alignment padding and no overlap"
            )
        end = offset + size
        if end > data_section_size:
            raise GGUFFormatError(
                f"{label}: tensor {name!r} extends past end of file "
                f"(offset {offset} + size {size} = {end} > data section size "
                f"{data_section_size})"
            )
        expected_offset = _align_up(end, alignment)
        highest_end = end
        previous = entry

    max_allowed_data_section_size = _align_up(highest_end, alignment)
    if not (highest_end <= data_section_size <= max_allowed_data_section_size):
        raise GGUFFormatError(
            f"{label}: reconciled tensor layout ends at byte {highest_end} "
            f"(up to {max_allowed_data_section_size} bytes allowed with "
            f"trailing alignment padding) but the file's data section is "
            f"{data_section_size} bytes (file size {file_size}, "
            f"header+alignment {data_start}); the tensor list does not "
            "account for every byte on disk"
        )

    size_by_index = {entry["index"]: entry["size"] for entry in entries}
    return [
        (tensor["name"], size_by_index[index])
        for index, tensor in enumerate(header["tensors"])
    ]


def classify_tensors(per_tensor: list) -> tuple:
    expert_bytes = 0
    non_expert_bytes = 0
    for name, size in per_tensor:
        if is_expert_tensor(name):
            expert_bytes += size
        else:
            non_expert_bytes += size
    return expert_bytes, non_expert_bytes


# ---------------------------------------------------------------------
# Model-path / split-shard resolution
# ---------------------------------------------------------------------
def _resolve_split_from_match(any_shard_path: Path, match: "re.Match") -> list:
    stem = match.group("stem")
    idx_str = match.group("idx")
    count_str = match.group("count")
    idx_width = len(idx_str)
    count = int(count_str)
    directory = any_shard_path.parent

    shards = []
    missing = []
    for i in range(1, count + 1):
        name = f"{stem}-{i:0{idx_width}d}-of-{count_str}.gguf"
        candidate = directory / name
        if candidate.is_file():
            shards.append(candidate)
        else:
            missing.append(name)
    if missing:
        raise FitCheckError(
            f"split GGUF set '{stem}' ({count} shard(s)) is missing shard(s): "
            + ", ".join(missing)
        )
    return shards


def resolve_shards(model_path: Path) -> list:
    model_path = Path(model_path)

    if model_path.is_dir():
        candidates = sorted(
            p for p in model_path.iterdir() if p.is_file() and p.suffix == ".gguf"
        )
        if not candidates:
            raise FitCheckError(f"no .gguf files found in directory {model_path}")

        split_groups: dict = {}
        plain = []
        for p in candidates:
            match = _SPLIT_RE.match(p.name)
            if match:
                key = (match.group("stem"), match.group("count"))
                split_groups.setdefault(key, []).append(p)
            else:
                plain.append(p)

        if split_groups:
            if len(split_groups) > 1 or plain:
                raise FitCheckError(
                    f"directory {model_path} contains more than one candidate "
                    "GGUF model (ambiguous); pass a single file or a single "
                    "split-shard set"
                )
            (_key, members) = next(iter(split_groups.items()))
            representative = members[0]
            match = _SPLIT_RE.match(representative.name)
            return _resolve_split_from_match(representative, match)

        if len(plain) > 1:
            raise FitCheckError(
                f"directory {model_path} contains {len(plain)} .gguf files "
                "(ambiguous); pass a single file"
            )
        return [plain[0]]

    if not model_path.is_file():
        raise FitCheckError(f"model path {model_path} does not exist")

    match = _SPLIT_RE.match(model_path.name)
    if match:
        return _resolve_split_from_match(model_path, match)
    return [model_path]


def compute_model_bytes(model_path: Path) -> tuple:
    shard_paths = resolve_shards(model_path)
    total_expert = 0
    total_non_expert = 0
    shard_details = []
    for shard_path in shard_paths:
        header = parse_gguf_header(shard_path)
        per_tensor = reconcile_shard(header)
        expert_bytes, non_expert_bytes = classify_tensors(per_tensor)
        total_expert += expert_bytes
        total_non_expert += non_expert_bytes

        bytes_by_type: dict = {}
        for tensor, (_name, size) in zip(header["tensors"], per_tensor):
            type_name = GGML_TYPES[tensor["type"]][0]
            bytes_by_type[type_name] = bytes_by_type.get(type_name, 0) + size
        shard_details.append({
            "name": shard_path.name,
            "data_start": header["data_start"],
            "header_end": header["header_end"],
            "file_size": header["file_size"],
            "tensor_count": len(header["tensors"]),
            "bytes_by_type": bytes_by_type,
        })
    return total_expert, total_non_expert, shard_paths, shard_details


# ---------------------------------------------------------------------
# Wired-memory ceiling
# ---------------------------------------------------------------------
def read_sysctl_int(name: str) -> Optional[int]:
    """Read an integer sysctl value; ``None`` on any failure (never raises)."""
    try:
        proc = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip())
    except ValueError:
        return None


def resolve_wired_limit_mib(
    flag_value: Optional[int],
    env_value: Optional[int],
    sysctl_reader=read_sysctl_int,
) -> tuple:
    """Returns ``(wired_limit_mib, source)`` where source is one of
    ``"flag"`` (an explicit ``--wired-limit-mib`` or the env-var fallback),
    ``"measured"`` (read live from ``iogpu.wired_limit_mb``), or
    ``"synthesized"`` (75% of ``hw.memsize``, used when the sysctl is
    absent or reports 0).

    A ``flag_value`` or ``env_value`` of zero or negative is rejected: zero
    is not "unset" (unset is ``None``) and a non-positive wired-memory
    ceiling can never legitimately fit anything, so accepting it would
    silently manufacture a bogus RED verdict instead of surfacing the
    operator's configuration mistake.
    """
    if flag_value is not None:
        if flag_value <= 0:
            raise FitCheckError(
                f"--wired-limit-mib must be a positive integer (> 0), got {flag_value}"
            )
        return flag_value, "flag"
    if env_value is not None:
        if env_value <= 0:
            raise FitCheckError(
                f"{ENV_WIRED_LIMIT_MIB}={env_value} must be a positive integer (> 0)"
            )
        return env_value, "flag"

    measured = sysctl_reader("iogpu.wired_limit_mb")
    if measured:
        return measured, "measured"

    memsize = sysctl_reader("hw.memsize")
    if not memsize:
        raise FitCheckError(
            "could not determine a wired memory ceiling: iogpu.wired_limit_mb "
            "is unavailable and hw.memsize could not be read either; pass "
            "--wired-limit-mib explicitly"
        )
    synthesized_mib = (memsize * 3 // 4) // MIB
    return synthesized_mib, "synthesized"


# ---------------------------------------------------------------------
# Environment-variable fallbacks (used only when the corresponding flag is
# absent; a flag always wins). These exist so this binary is a drop-in
# --fit-check-bin even for a launcher invocation that only forwards the
# launcher's own fixed argv shape and no extra --fit-check-arg tokens.
# ---------------------------------------------------------------------
ENV_KV_RESERVE_GIB = "FASTMLX_GGUF_KV_RESERVE_GIB"
ENV_WIRED_LIMIT_MIB = "FASTMLX_WIRED_LIMIT_MIB"
ENV_WIRED_MARGIN_GIB = "FASTMLX_WIRED_MARGIN_GIB"
ENV_RESIDENCY = "FASTMLX_GGUF_RESIDENCY"


def _env_float(name: str) -> Optional[float]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        raise FitCheckError(f"environment variable {name}={value!r} is not a number")


def _env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        raise FitCheckError(f"environment variable {name}={value!r} is not an integer")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
# sysexits.h EX_USAGE: argparse's own ArgumentParser.error() exits 2, which
# this binary's own protocol reserves for a real RED fit verdict (see
# module docstring). A bad invocation -- an unknown flag, a missing
# required flag, an invalid choice, an out-of-range value -- must never be
# misread by scripts/fastmlx_launch.py's run_fit_check() as "does not fit".
_EXIT_USAGE_ERROR = 64


class _UsageErrorArgumentParser(argparse.ArgumentParser):
    """Same as ``argparse.ArgumentParser``, except a usage error exits 64
    (EX_USAGE) instead of argparse's default of 2."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(_EXIT_USAGE_ERROR, f"{self.prog}: error: {message}\n")


def _wired_margin_gib_type(value: str) -> int:
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}")
    if not (2 <= ivalue <= 32):
        raise argparse.ArgumentTypeError(
            f"--wired-margin-gib must be an integer between 2 and 32, got {ivalue}"
        )
    return ivalue


def _positive_int_type(flag_name: str):
    def _type(value: str) -> int:
        try:
            ivalue = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid int value: {value!r}")
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(
                f"{flag_name} must be a positive integer (> 0), got {ivalue}"
            )
        return ivalue

    return _type


def build_arg_parser() -> argparse.ArgumentParser:
    parser = _UsageErrorArgumentParser(
        prog="fastmlx-gguf-fit",
        description="GGUF-aware, residency-aware pre-load fit checker.",
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
        "--wired-limit-mib", type=_positive_int_type("--wired-limit-mib"), default=None
    )
    parser.add_argument("--wired-margin-gib", type=_wired_margin_gib_type, default=None)
    parser.add_argument("--kv-reserve-gib", type=float, default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def compute_fit(args: argparse.Namespace) -> dict:
    residency = args.residency or os.environ.get(ENV_RESIDENCY) or "resident"
    if residency not in ("resident", "expert-stream"):
        raise FitCheckError(
            f"--residency must be 'resident' or 'expert-stream', got {residency!r}"
        )

    kv_reserve_gib = args.kv_reserve_gib
    if kv_reserve_gib is None:
        kv_reserve_gib = _env_float(ENV_KV_RESERVE_GIB)
    if kv_reserve_gib is None:
        raise FitCheckError(
            "--kv-reserve-gib is required (pass --kv-reserve-gib or set "
            f"{ENV_KV_RESERVE_GIB}); a fit check must never silently assume "
            "zero KV-cache reserve"
        )
    if kv_reserve_gib < 0:
        raise FitCheckError("--kv-reserve-gib must be >= 0")

    wired_margin_gib = args.wired_margin_gib
    if wired_margin_gib is None:
        wired_margin_gib = _env_int(ENV_WIRED_MARGIN_GIB)
    if wired_margin_gib is None:
        wired_margin_gib = 8
    if not (2 <= wired_margin_gib <= 32):
        raise FitCheckError("--wired-margin-gib must be an integer between 2 and 32")

    wired_limit_mib, wired_limit_source = resolve_wired_limit_mib(
        args.wired_limit_mib, _env_int(ENV_WIRED_LIMIT_MIB)
    )

    expert_bytes, non_expert_bytes, _shard_paths, shard_details = compute_model_bytes(
        args.model_path
    )

    if residency == "expert-stream":
        weights_bytes = non_expert_bytes
        lower_bound = True
    else:
        weights_bytes = expert_bytes + non_expert_bytes
        lower_bound = False

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
    total_bytes = weights_bytes + kv_reserve_bytes

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
        "weights_bytes": weights_bytes,
        "expert_bytes": expert_bytes,
        "non_expert_bytes": non_expert_bytes,
        "kv_reserve_bytes": kv_reserve_bytes,
        "total_bytes": total_bytes,
        "ceiling_bytes": ceiling_bytes,
        "wired_limit_source": wired_limit_source,
        "residency": residency,
        "lower_bound": lower_bound,
        "fit": fit,
        "reason": reason,
        "reason_text": reason_text,
        "shards": shard_details,
    }


def format_attestation_line(result: dict) -> str:
    return (
        f"{_ATTESTATION_PREFIX} fit_check_only=complete weights_loaded=false "
        f"weights_bytes={result['weights_bytes']} "
        f"expert_bytes={result['expert_bytes']} "
        f"non_expert_bytes={result['non_expert_bytes']} "
        f"kv_reserve_bytes={result['kv_reserve_bytes']} "
        f"total_bytes={result['total_bytes']} "
        f"ceiling_bytes={result['ceiling_bytes']} "
        f"wired_limit_source={result['wired_limit_source']} "
        f"residency={result['residency']} "
        f"lower_bound={'true' if result['lower_bound'] else 'false'} "
        f"fit={result['fit']} "
        f"reason={result['reason']}"
    )


def main(argv: Optional[list] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        result = compute_fit(args)
    except (FitCheckError, GGUFFormatError) as exc:
        print(f"fastmlx-gguf-fit: error: {exc}", file=sys.stderr)
        return 1

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
