"""Tests for scripts/fastmlx_gguf_fit.py: a GGUF-aware, residency-aware
pre-load fit checker.

The GGUF v3 writer below is independent, minimal, in-test code: it does not
import anything from the module under test for its byte-layout logic, so a
regression in the production parser cannot also corrupt the fixtures that
are meant to catch it. Per-tensor byte sizes used as expected values are
hand-derived literals from the published ggml block-size/type-size table,
not computed by calling into the implementation.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "fastmlx_gguf_fit.py"
LAUNCH_PATH = Path(__file__).resolve().parents[1] / "fastmlx_launch.py"

_FIT_SPEC = importlib.util.spec_from_file_location("fastmlx_gguf_fit", SCRIPT_PATH)
assert _FIT_SPEC is not None and _FIT_SPEC.loader is not None
FIT = importlib.util.module_from_spec(_FIT_SPEC)
_FIT_SPEC.loader.exec_module(FIT)

_LAUNCH_SPEC = importlib.util.spec_from_file_location("fastmlx_launch", LAUNCH_PATH)
assert _LAUNCH_SPEC is not None and _LAUNCH_SPEC.loader is not None
LAUNCH = importlib.util.module_from_spec(_LAUNCH_SPEC)
_LAUNCH_SPEC.loader.exec_module(LAUNCH)


def _make_executable(path: Path) -> Path:
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


# ---------------------------------------------------------------------
# Independent minimal GGUF v3 writer (test-only; not imported from the
# module under test's binary-layout logic).
# ---------------------------------------------------------------------
def _gguf_string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv_u32(key: str, value: int) -> bytes:
    return _gguf_string(key) + struct.pack("<I", 4) + struct.pack("<I", value)


# gguf_metadata_value_type ids used only by the large-metadata-array fixture
# below (independent literals, mirroring the module-under-test's own
# constants on purpose rather than importing them).
_KVTYPE_INT32 = 5
_KVTYPE_STRING = 8
_KVTYPE_ARRAY = 9


def _kv_string_array(key: str, values: list) -> bytes:
    out = bytearray()
    out += _gguf_string(key)
    out += struct.pack("<I", _KVTYPE_ARRAY)
    out += struct.pack("<I", _KVTYPE_STRING)
    out += struct.pack("<Q", len(values))
    for v in values:
        out += _gguf_string(v)
    return bytes(out)


def _kv_int32_array(key: str, values: list) -> bytes:
    out = bytearray()
    out += _gguf_string(key)
    out += struct.pack("<I", _KVTYPE_ARRAY)
    out += struct.pack("<I", _KVTYPE_INT32)
    out += struct.pack("<Q", len(values))
    out += struct.pack(f"<{len(values)}i", *values)
    return bytes(out)


def build_gguf_bytes(
    *,
    tensors: list,
    data_section: bytes,
    version: int = 3,
    magic: bytes = b"GGUF",
    alignment: int = 32,
    write_alignment_kv: bool = True,
) -> bytes:
    """tensors: list of {"name": str, "dims": [int, ...], "type": int, "offset": int}."""
    header = bytearray()
    header += magic
    header += struct.pack("<I", version)
    header += struct.pack("<Q", len(tensors))
    header += struct.pack("<Q", 1 if write_alignment_kv else 0)
    if write_alignment_kv:
        header += _kv_u32("general.alignment", alignment)
    for t in tensors:
        header += _gguf_string(t["name"])
        header += struct.pack("<I", len(t["dims"]))
        for d in t["dims"]:
            header += struct.pack("<Q", d)
        header += struct.pack("<I", t["type"])
        header += struct.pack("<Q", t["offset"])
    header_end = len(header)
    pad_len = (-header_end) % alignment
    return bytes(header) + bytes(pad_len) + data_section


# ggml type ids used by the fixtures below (mirrors GGML_TYPES in the module
# under test, but written as independent literals here on purpose).
_TYPE_F32 = 0
_TYPE_F16 = 1
_TYPE_Q8_0 = 8
_TYPE_Q4_K = 12
_TYPE_I8 = 24
_TYPE_MXFP4 = 39
_TYPE_UNKNOWN = 12345


def write_file(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def run_cli(args: list, env: dict = None) -> subprocess.CompletedProcess:
    argv = [sys.executable, "-B", str(SCRIPT_PATH)] + args
    return subprocess.run(argv, capture_output=True, text=True, timeout=30, env=env)


class MinimalWriterFixtureTests(unittest.TestCase):
    """T1: hand-derived per-tensor byte sizes and whole-file reconciliation."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_per_tensor_sizes_match_hand_derived_constants_and_file_reconciles(self):
        # F32[4]   = 4 * 4                    = 16 bytes  (block=1)
        # F16[4]   = 4 * 2                    = 8 bytes   (block=1)
        # Q8_0[32] = 1 block * 34             = 34 bytes  (block=32, 1 block)
        # Q4_K[256,2] = 2 blocks * 144        = 288 bytes (block=256, 512 elems -> 2 blocks)
        tensors = [
            {"name": "t.f32", "dims": [4], "type": _TYPE_F32, "offset": 0},
            {"name": "t.f16", "dims": [4], "type": _TYPE_F16, "offset": 32},
            {"name": "t.q8_0", "dims": [32], "type": _TYPE_Q8_0, "offset": 64},
            {"name": "t.q4_k", "dims": [256, 2], "type": _TYPE_Q4_K, "offset": 128},
        ]
        data_section = bytes(416)  # 128 + 288 = 416, the reconciled end
        blob = build_gguf_bytes(tensors=tensors, data_section=data_section)
        path = write_file(self.root / "model.gguf", blob)

        header = FIT.parse_gguf_header(path)
        per_tensor = FIT.reconcile_shard(header)
        sizes = dict(per_tensor)

        self.assertEqual(sizes["t.f32"], 16)
        self.assertEqual(sizes["t.f16"], 8)
        self.assertEqual(sizes["t.q8_0"], 34)
        self.assertEqual(sizes["t.q4_k"], 288)

        expected_total_file_size = header["data_start"] + 416
        self.assertEqual(path.stat().st_size, expected_total_file_size)


class MalformedInputTests(unittest.TestCase):
    """T2: bad-input scenarios must never exit 0 or 2, and must never print fit=green."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _assert_error_exit(self, path_arg: str, extra: list = None):
        args = [
            "--model-path", path_arg,
            "--kv-reserve-gib", "0",
            "--wired-limit-mib", "4096",
            "--wired-margin-gib", "2",
        ] + (extra or [])
        result = run_cli(args)
        self.assertNotIn(result.returncode, (0, 2), msg=result.stderr)
        self.assertNotIn("fit=green", result.stdout)
        return result

    def test_unknown_ggml_type_id_errors(self):
        tensors = [{"name": "t.bad", "dims": [1], "type": _TYPE_UNKNOWN, "offset": 0}]
        blob = build_gguf_bytes(tensors=tensors, data_section=b"")
        path = write_file(self.root / "unknown_type.gguf", blob)
        self._assert_error_exit(str(path))

    def test_bad_magic_errors(self):
        blob = build_gguf_bytes(tensors=[], data_section=b"", magic=b"BADG")
        path = write_file(self.root / "bad_magic.gguf", blob)
        self._assert_error_exit(str(path))

    def test_unsupported_version_errors(self):
        blob = build_gguf_bytes(tensors=[], data_section=b"", version=2)
        path = write_file(self.root / "v2.gguf", blob)
        self._assert_error_exit(str(path))

    def test_truncated_header_errors(self):
        # A KV entry that declares a value type but is cut off before the
        # value's bytes are present: this must fail inside header parsing,
        # not the post-header reconciliation pass.
        blob = bytearray()
        blob += b"GGUF"
        blob += struct.pack("<I", 3)
        blob += struct.pack("<Q", 0)  # tensor_count
        blob += struct.pack("<Q", 1)  # kv_count
        blob += _gguf_string("general.alignment")
        blob += struct.pack("<I", 4)  # value_type = UINT32
        # ... the 4-byte UINT32 value itself is missing: truncated.
        path = write_file(self.root / "truncated.gguf", bytes(blob))
        self._assert_error_exit(str(path))

    def test_tensor_extending_past_eof_errors(self):
        tensors = [{"name": "t.f32", "dims": [4], "type": _TYPE_F32, "offset": 0}]
        # t.f32 needs 16 bytes; only 8 are actually present in the file.
        blob = build_gguf_bytes(tensors=tensors, data_section=bytes(8))
        path = write_file(self.root / "past_eof.gguf", blob)
        self._assert_error_exit(str(path))

    def test_missing_split_shard_errors(self):
        shard_dir = self.root / "split_missing"
        shard_dir.mkdir()
        blob = build_gguf_bytes(tensors=[], data_section=b"")
        write_file(shard_dir / "pack-00001-of-00002.gguf", blob)
        # pack-00002-of-00002.gguf deliberately absent.
        self._assert_error_exit(str(shard_dir))


class ExpertClassificationTests(unittest.TestCase):
    """T3: generic ffn_*_exps tensor-name classification."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

        tensors = [
            {"name": "blk.0.ffn_gate_exps.weight", "dims": [4], "type": _TYPE_F32, "offset": 0},
            {"name": "blk.0.ffn_up_exps.weight", "dims": [4], "type": _TYPE_F32, "offset": 32},
            {"name": "blk.0.ffn_down_exps.weight", "dims": [4], "type": _TYPE_F32, "offset": 64},
            {"name": "blk.0.attn_q.weight", "dims": [4], "type": _TYPE_F32, "offset": 96},
            {"name": "token_embd.weight", "dims": [4], "type": _TYPE_F32, "offset": 128},
        ]
        # 5 tensors * 16 bytes each, ending at 128 + 16 = 144.
        blob = build_gguf_bytes(tensors=tensors, data_section=bytes(144))
        self.path = write_file(self.root / "moe.gguf", blob)

    def test_expert_and_non_expert_bytes_split_correctly(self):
        # NOTE for reviewer mutation testing: if is_expert_tensor() is
        # mutated to always return False, expert_bytes collapses to 0 and
        # non_expert_bytes inflates to 80 -- both assertions below catch it.
        (
            expert_bytes,
            non_expert_bytes,
            _shards,
            _shard_details,
            _expert_bytes_by_type,
        ) = FIT.compute_model_bytes(self.path)
        self.assertGreater(expert_bytes, 0)
        self.assertEqual(expert_bytes, 48)
        self.assertEqual(non_expert_bytes, 32)

    def test_expert_stream_residency_reports_non_expert_only_as_lower_bound(self):
        result = run_cli([
            "--model-path", str(self.path),
            "--residency", "expert-stream",
            "--kv-reserve-gib", "0",
            "--wired-limit-mib", "4096",
            "--wired-margin-gib", "2",
        ])
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        fields = LAUNCH._parse_attestation_fields(
            LAUNCH._find_attestation_line(result.stdout)
        )
        self.assertEqual(fields["weights_bytes"], "32")
        self.assertEqual(fields["lower_bound"], "true")


class CeilingBoundaryTests(unittest.TestCase):
    """T4: total == ceiling is GREEN; one byte over is RED with a specific
    reason. The ceiling here is a positive 1 MiB (wired-limit-mib 2049 minus
    a 2 GiB margin = 1 MiB), not zero: a zero-or-negative ceiling is its own
    configuration-error case, covered by WiredLimitPositivityTests."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # wired-limit 2049 MiB - margin 2 GiB (2048 MiB) => ceiling = 1 MiB
        # = 1,048,576 bytes exactly.
        self.ceiling_args = [
            "--wired-limit-mib", "2049",
            "--wired-margin-gib", "2",
        ]
        self.ceiling_bytes = 1048576

    def test_total_equal_to_ceiling_is_green(self):
        blob = build_gguf_bytes(tensors=[], data_section=b"")
        path = write_file(self.root / "empty.gguf", blob)
        # 1,048,576 bytes / GIB == 2**-10, an exact binary fraction, so the
        # round-trip through float is exact and kv_reserve_bytes == ceiling.
        result = run_cli(
            ["--model-path", str(path), "--kv-reserve-gib", "0.0009765625"]
            + self.ceiling_args
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("fit=green", result.stdout)

    def test_one_byte_over_ceiling_is_red_with_specific_reason(self):
        tensors = [
            {"name": "t.i8", "dims": [self.ceiling_bytes + 1], "type": _TYPE_I8, "offset": 0}
        ]
        blob = build_gguf_bytes(tensors=tensors, data_section=bytes(self.ceiling_bytes + 1))
        path = write_file(self.root / "one_byte.gguf", blob)
        result = run_cli(
            ["--model-path", str(path), "--kv-reserve-gib", "0"] + self.ceiling_args
        )
        self.assertEqual(result.returncode, 2, msg=result.stdout + result.stderr)
        self.assertIn(f"exceeds ceiling {self.ceiling_bytes} bytes", result.stderr)
        self.assertIn("over by 1 bytes", result.stderr)


class MissingKvReserveTests(unittest.TestCase):
    """T5: --kv-reserve-gib is required; missing it must never default to 0."""

    def test_missing_kv_reserve_gib_errors_naming_the_flag(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("FASTMLX_GGUF_")}
        env.pop("FASTMLX_WIRED_LIMIT_MIB", None)
        env.pop("FASTMLX_WIRED_MARGIN_GIB", None)
        result = run_cli(
            ["--model-path", "/nonexistent-path-not-read.gguf"], env=env
        )
        self.assertNotIn(result.returncode, (0, 2))
        self.assertIn("--kv-reserve-gib", result.stderr)


class SplitShardTests(unittest.TestCase):
    """T6: a 2-shard split set resolves, sums, and its attestation line parses
    with the launcher's own field parser."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_two_shard_set_sums_and_attestation_line_parses(self):
        shard_dir = self.root / "split_ok"
        shard_dir.mkdir()
        shard1 = [{"name": "token_embd.weight", "dims": [4], "type": _TYPE_F32, "offset": 0}]
        shard2 = [{"name": "output.weight", "dims": [4], "type": _TYPE_F32, "offset": 0}]
        write_file(
            shard_dir / "pack-00001-of-00002.gguf",
            build_gguf_bytes(tensors=shard1, data_section=bytes(16)),
        )
        write_file(
            shard_dir / "pack-00002-of-00002.gguf",
            build_gguf_bytes(tensors=shard2, data_section=bytes(16)),
        )

        result = run_cli([
            "--model-path", str(shard_dir / "pack-00001-of-00002.gguf"),
            "--kv-reserve-gib", "0",
            "--wired-limit-mib", "4096",
            "--wired-margin-gib", "2",
        ])
        self.assertEqual(result.returncode, 0, msg=result.stderr)

        line = LAUNCH._find_attestation_line(result.stdout)
        self.assertIsNotNone(line)
        fields = LAUNCH._parse_attestation_fields(line)
        self.assertEqual(fields["fit"], "green")
        self.assertEqual(fields["weights_bytes"], "32")


class LauncherCallSiteTests(unittest.TestCase):
    """T7: the launcher's run_fit_check() admits/refuses via the real binary,
    using the env-var fallback path (no --fit-check-arg forwarding)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        _make_executable(SCRIPT_PATH)

        tensors = [{"name": "token_embd.weight", "dims": [4], "type": _TYPE_F32, "offset": 0}]
        blob = build_gguf_bytes(tensors=tensors, data_section=bytes(16))
        self.model_path = write_file(self.root / "small.gguf", blob)

    def test_run_fit_check_admits_when_it_fits(self):
        env_overrides = {
            FIT.ENV_KV_RESERVE_GIB: "0",
            FIT.ENV_WIRED_LIMIT_MIB: "4096",
            FIT.ENV_WIRED_MARGIN_GIB: "2",
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            result = LAUNCH.run_fit_check(
                fit_check_bin=str(SCRIPT_PATH),
                model_id="fixture-pack",
                model_path=self.model_path,
                host_use="shared",
                context=None,
                extra_args=[],
            )
        self.assertEqual(result.kind, "green", msg=result.detail)
        self.assertEqual(result.fields.get("fit"), "green")

    def test_run_fit_check_refuses_when_it_does_not_fit(self):
        # A positive but insufficient ceiling (1 MiB, the smallest a 2 GiB
        # margin admits) plus a 1 GiB KV reserve: a legitimate RED verdict,
        # not the zero-or-negative-ceiling *configuration* error case
        # (covered separately by WiredLimitPositivityTests).
        env_overrides = {
            FIT.ENV_KV_RESERVE_GIB: "1",
            FIT.ENV_WIRED_LIMIT_MIB: "2049",
            FIT.ENV_WIRED_MARGIN_GIB: "2",
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            result = LAUNCH.run_fit_check(
                fit_check_bin=str(SCRIPT_PATH),
                model_id="fixture-pack",
                model_path=self.model_path,
                host_use="shared",
                context=None,
                extra_args=[],
            )
        self.assertEqual(result.kind, "red")


class _CountingBinaryFile:
    """Wraps an open binary file object and counts bytes actually read, to
    prove the header parser never reads a shard's (potentially many-GiB)
    tensor-data section into memory."""

    def __init__(self, fileobj):
        self._fileobj = fileobj
        self.bytes_read = 0

    def read(self, n=-1):
        chunk = self._fileobj.read(n)
        self.bytes_read += len(chunk)
        return chunk

    def fileno(self):
        return self._fileobj.fileno()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._fileobj.close()
        return False


class LargeShardIncrementalReadTests(unittest.TestCase):
    """T8 (HIGH): the header parser must stream the header in bounded
    chunks from an open file handle, never reading a shard's multi-GiB
    tensor-data section into memory."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_huge_declared_tensor_is_sized_without_reading_its_bytes(self):
        # F32 dims [1024, 1048576] = 1,073,741,824 elements * 4 bytes
        # = 4,294,967,296 bytes (4 GiB) declared for a single tensor, but
        # the file on disk is sparse: only the header is really written.
        huge_bytes = 1024 * 1048576 * 4
        self.assertEqual(huge_bytes, 4294967296)
        tensors = [
            {"name": "t.big", "dims": [1024, 1048576], "type": _TYPE_F32, "offset": 0},
        ]
        header_only_blob = build_gguf_bytes(tensors=tensors, data_section=b"")
        path = write_file(self.root / "huge.gguf", header_only_blob)
        data_start = len(header_only_blob)
        # Sparse-extend: costs no real disk space, just a bigger apparent
        # file size (os.truncate never allocates the extended range).
        os.truncate(path, data_start + huge_bytes)

        counting_files = []

        def instrumented_open_binary(p):
            counted = _CountingBinaryFile(open(p, "rb"))
            counting_files.append(counted)
            return counted

        with patch.object(FIT, "_open_binary", instrumented_open_binary):
            header = FIT.parse_gguf_header(path)
            per_tensor = FIT.reconcile_shard(header)

        sizes = dict(per_tensor)
        self.assertEqual(sizes["t.big"], huge_bytes)

        self.assertEqual(len(counting_files), 1)
        bytes_actually_read = counting_files[0].bytes_read
        self.assertLessEqual(
            bytes_actually_read,
            header["header_end"] + FIT._HEADER_READ_CHUNK_BYTES,
            msg=(
                f"parser read {bytes_actually_read} bytes but the header "
                f"ends at {header['header_end']}; it must not have read "
                "into the multi-GiB data section"
            ),
        )


class PaddedLastTensorReconciliationTests(unittest.TestCase):
    """T9 (MEDIUM): real GGUF writers pad after every tensor, including the
    last; reconcile_shard must accept trailing alignment padding up to one
    alignment unit past the highest tensor end, and still reject anything
    beyond that or anything short."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # Q8_0[32] = 1 block * 34 bytes; alignment 32; highest_end = 34;
        # align_up(34, 32) = 64.
        self.tensors = [{"name": "t.q8_0", "dims": [32], "type": _TYPE_Q8_0, "offset": 0}]

    def test_trailing_padding_up_to_alignment_is_accepted(self):
        blob = build_gguf_bytes(tensors=self.tensors, data_section=bytes(64))
        path = write_file(self.root / "padded_ok.gguf", blob)
        header = FIT.parse_gguf_header(path)
        per_tensor = FIT.reconcile_shard(header)
        self.assertEqual(dict(per_tensor)["t.q8_0"], 34)

    def test_padding_beyond_one_alignment_unit_errors(self):
        # 30 bytes to reach the alignment boundary, plus a whole extra
        # alignment unit (32) of trailing garbage: 34 + 30 + 32 = 96.
        blob = build_gguf_bytes(tensors=self.tensors, data_section=bytes(96))
        path = write_file(self.root / "padded_too_much.gguf", blob)
        header = FIT.parse_gguf_header(path)
        with self.assertRaises(FIT.GGUFFormatError):
            FIT.reconcile_shard(header)

    def test_data_section_shorter_than_highest_end_errors(self):
        blob = build_gguf_bytes(tensors=self.tensors, data_section=bytes(33))
        path = write_file(self.root / "too_short.gguf", blob)
        header = FIT.parse_gguf_header(path)
        with self.assertRaises(FIT.GGUFFormatError):
            FIT.reconcile_shard(header)


class JsonShardsFieldTests(unittest.TestCase):
    """T10: --json output includes a shards list with basename-only name
    and per-shard header/type statistics (never a full path)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_json_shards_field_reports_name_and_type_bytes(self):
        tensors = [
            {"name": "t.f32.a", "dims": [4], "type": _TYPE_F32, "offset": 0},
            {"name": "t.f32.b", "dims": [4], "type": _TYPE_F32, "offset": 32},
            {"name": "t.q8_0", "dims": [32], "type": _TYPE_Q8_0, "offset": 64},
        ]
        blob = build_gguf_bytes(tensors=tensors, data_section=bytes(98))
        path = write_file(self.root / "typed.gguf", blob)

        result = run_cli([
            "--model-path", str(path),
            "--kv-reserve-gib", "0",
            "--wired-limit-mib", "4096",
            "--wired-margin-gib", "2",
            "--json",
        ])
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(result.stdout)

        self.assertEqual(len(payload["shards"]), 1)
        shard = payload["shards"][0]
        self.assertEqual(shard["name"], "typed.gguf")
        self.assertNotIn(str(self.root), shard["name"])
        self.assertEqual(shard["tensor_count"], 3)
        self.assertEqual(shard["bytes_by_type"]["F32"], 32)
        self.assertEqual(shard["bytes_by_type"]["Q8_0"], 34)
        self.assertEqual(shard["file_size"], shard["data_start"] + 98)
        self.assertLessEqual(shard["header_end"], shard["data_start"])


class ExpertBytesByTypeJsonTests(unittest.TestCase):
    """T15: --json output includes expert_bytes_by_type, a
    {ggml type name: bytes} breakdown over routed-expert tensors only (an
    informational field -- the sizer stays engine-agnostic and never
    refuses on tensor type)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_expert_bytes_by_type_reports_routed_expert_type_only(self):
        # blk.0.ffn_gate_exps: MXFP4[32] = 1 block * 17 bytes = 17 bytes
        # (block=32, type_size=17). A non-expert F32[4] tensor (16 bytes)
        # must not appear in expert_bytes_by_type at all.
        tensors = [
            {
                "name": "blk.0.ffn_gate_exps.weight",
                "dims": [32],
                "type": _TYPE_MXFP4,
                "offset": 0,
            },
            {"name": "token_embd.weight", "dims": [4], "type": _TYPE_F32, "offset": 32},
        ]
        # t0 (17 bytes) padded to 32; t1 (16 bytes) at 32, ends at 48.
        blob = build_gguf_bytes(tensors=tensors, data_section=bytes(48))
        path = write_file(self.root / "expert_typed.gguf", blob)

        result = run_cli([
            "--model-path", str(path),
            "--kv-reserve-gib", "0",
            "--wired-limit-mib", "4096",
            "--wired-margin-gib", "2",
            "--json",
        ])
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(result.stdout)

        self.assertEqual(payload["expert_bytes_by_type"], {"MXFP4": 17})
        self.assertNotIn("F32", payload["expert_bytes_by_type"])

    def test_expert_bytes_by_type_empty_when_no_expert_tensors(self):
        tensors = [
            {"name": "token_embd.weight", "dims": [4], "type": _TYPE_F32, "offset": 0},
        ]
        blob = build_gguf_bytes(tensors=tensors, data_section=bytes(16))
        path = write_file(self.root / "no_experts.gguf", blob)

        result = run_cli([
            "--model-path", str(path),
            "--kv-reserve-gib", "0",
            "--wired-limit-mib", "4096",
            "--wired-margin-gib", "2",
            "--json",
        ])
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(result.stdout)

        self.assertEqual(payload["expert_bytes_by_type"], {})


class ContiguousPackingTests(unittest.TestCase):
    """T11 (HIGH): ggml block sizes alone are not discriminating for
    non-last tensors -- reconcile_shard must additionally require that
    tensors, sorted by offset, are packed contiguously (each tensor's
    offset equals the previous tensor's aligned-up end), rejecting gaps
    larger than alignment padding and rejecting overlap."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _three_tensor_fixture(self, data_section_len=208):
        # t0 [0,16), pad to 32. t1 [32,192) (F32 dims=[40] = 160 bytes,
        # exact multiple of 32, no padding). t2 at 192, real on-disk size 16.
        tensors = [
            {"name": "t0.f32", "dims": [4], "type": _TYPE_F32, "offset": 0},
            {"name": "t1.f32.mid", "dims": [40], "type": _TYPE_F32, "offset": 32},
            {"name": "t2.f32", "dims": [4], "type": _TYPE_F32, "offset": 192},
        ]
        blob = build_gguf_bytes(tensors=tensors, data_section=bytes(data_section_len))
        path = write_file(self.root / f"three_{data_section_len}.gguf", blob)
        return path

    def test_unmutated_three_tensor_fixture_reconciles_ok(self):
        # Sanity check on the fixture itself before proving mutation breaks it.
        path = self._three_tensor_fixture()
        header = FIT.parse_gguf_header(path)
        per_tensor = FIT.reconcile_shard(header)
        self.assertEqual(dict(per_tensor)["t1.f32.mid"], 160)

    def test_enlarging_a_middle_tensor_type_size_is_caught(self):
        # (a) Patch F32's type_size 4 -> 8: the middle tensor's computed
        # size grows from 160 to 320 bytes, so its reconciled end (352)
        # no longer lines up with t2's real on-disk offset (192, fixed on
        # disk regardless of the mutation). The data section is given 352
        # bytes of slack (rather than the tight 208 the unmutated layout
        # needs) so that ggml block-size/EOF/tail bookkeeping alone -- the
        # checks that predate this fix -- cannot coincidentally catch the
        # mutation; only a genuine packing/contiguity check can.
        path = self._three_tensor_fixture(data_section_len=352)
        header = FIT.parse_gguf_header(path)
        mutated_types = dict(FIT.GGML_TYPES)
        mutated_types[_TYPE_F32] = ("F32", 1, 8)
        with patch.dict(FIT.GGML_TYPES, mutated_types, clear=True):
            with self.assertRaises(FIT.GGUFFormatError):
                FIT.reconcile_shard(header)

    def test_shrinking_a_middle_tensor_type_size_is_caught(self):
        # (b) Patch F32's type_size 4 -> 1: the middle tensor's computed
        # size shrinks from 160 to 40 bytes, so the expected next offset
        # (32+40=72, aligned up to 96) no longer matches t2's real offset
        # (192).
        path = self._three_tensor_fixture()
        header = FIT.parse_gguf_header(path)
        mutated_types = dict(FIT.GGML_TYPES)
        mutated_types[_TYPE_F32] = ("F32", 1, 1)
        with patch.dict(FIT.GGML_TYPES, mutated_types, clear=True):
            with self.assertRaises(FIT.GGUFFormatError):
                FIT.reconcile_shard(header)

    def test_gap_larger_than_alignment_padding_errors(self):
        # (c) t0 ends at 16, aligns up to 32 -- but t1 is really written at
        # offset 96, a gap far larger than alignment padding can justify.
        tensors = [
            {"name": "t0.f32", "dims": [4], "type": _TYPE_F32, "offset": 0},
            {"name": "t1.f32", "dims": [4], "type": _TYPE_F32, "offset": 96},
        ]
        blob = build_gguf_bytes(tensors=tensors, data_section=bytes(112))
        path = write_file(self.root / "gap.gguf", blob)
        header = FIT.parse_gguf_header(path)
        with self.assertRaises(FIT.GGUFFormatError):
            FIT.reconcile_shard(header)

    def test_overlapping_offsets_error(self):
        # (d) Both tensors declare offset 0 (a legal multiple of the
        # alignment on its own), but the second tensor overlaps the first's
        # [0, 34) byte range instead of starting where it ends.
        tensors = [
            {"name": "t0.q8_0", "dims": [32], "type": _TYPE_Q8_0, "offset": 0},
            {"name": "t1.overlap", "dims": [4], "type": _TYPE_F32, "offset": 0},
        ]
        blob = build_gguf_bytes(tensors=tensors, data_section=bytes(64))
        path = write_file(self.root / "overlap.gguf", blob)
        header = FIT.parse_gguf_header(path)
        with self.assertRaises(FIT.GGUFFormatError):
            FIT.reconcile_shard(header)

    def test_valid_padded_layout_with_34_byte_middle_tensor_is_ok(self):
        # (e) A real-writer-shaped layout: t0 [0,16) pad to 32; t1 (Q8_0,
        # 34 bytes) [32,66) pad to 96; t2 [96,112). The [66,96) gap is
        # legitimate alignment padding, not a defect.
        tensors = [
            {"name": "t0.f32", "dims": [4], "type": _TYPE_F32, "offset": 0},
            {"name": "t1.q8_0.mid", "dims": [32], "type": _TYPE_Q8_0, "offset": 32},
            {"name": "t2.f32", "dims": [4], "type": _TYPE_F32, "offset": 96},
        ]
        blob = build_gguf_bytes(tensors=tensors, data_section=bytes(112))
        path = write_file(self.root / "padded_middle.gguf", blob)
        header = FIT.parse_gguf_header(path)
        per_tensor = FIT.reconcile_shard(header)
        sizes = dict(per_tensor)
        self.assertEqual(sizes["t0.f32"], 16)
        self.assertEqual(sizes["t1.q8_0.mid"], 34)
        self.assertEqual(sizes["t2.f32"], 16)


class ArgparseUsageErrorExitCodeTests(unittest.TestCase):
    """T12 (HIGH): argparse usage errors must exit 64 (EX_USAGE), never 0 or
    2 -- the launcher's fit-check protocol reserves exit 2 for a real RED
    fit verdict, so a bad invocation must not be misread as "does not fit"."""

    def _assert_usage_exit(self, args):
        result = run_cli(args)
        self.assertEqual(
            result.returncode, 64, msg=f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        self.assertNotIn("fit=green", result.stdout)
        return result

    def test_unknown_flag_exits_64(self):
        self._assert_usage_exit(["--model-path", "/does-not-matter.gguf", "--bogus-flag", "x"])

    def test_missing_model_path_exits_64(self):
        self._assert_usage_exit([])

    def test_bad_residency_value_exits_64(self):
        self._assert_usage_exit(
            ["--model-path", "/does-not-matter.gguf", "--residency", "nonsense"]
        )

    def test_wired_margin_gib_below_range_exits_64(self):
        self._assert_usage_exit(
            ["--model-path", "/does-not-matter.gguf", "--wired-margin-gib", "1"]
        )

    def test_wired_margin_gib_above_range_exits_64(self):
        self._assert_usage_exit(
            ["--model-path", "/does-not-matter.gguf", "--wired-margin-gib", "33"]
        )


class WiredLimitPositivityTests(unittest.TestCase):
    """T13 (HIGH): --wired-limit-mib/env value must be a positive integer,
    and a ceiling (limit - margin) that is <= 0 must be a configuration
    error -- never accepted, and never reported as a fit=red verdict."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        blob = build_gguf_bytes(tensors=[], data_section=b"")
        self.model_path = write_file(self.root / "empty.gguf", blob)

    def test_wired_limit_mib_zero_flag_is_a_usage_error(self):
        result = run_cli([
            "--model-path", str(self.model_path),
            "--kv-reserve-gib", "0",
            "--wired-limit-mib", "0",
            "--wired-margin-gib", "2",
        ])
        self.assertEqual(result.returncode, 64, msg=result.stdout + result.stderr)
        self.assertNotIn("fit=green", result.stdout)

    def test_wired_limit_mib_negative_flag_is_a_usage_error(self):
        result = run_cli([
            "--model-path", str(self.model_path),
            "--kv-reserve-gib", "0",
            "--wired-limit-mib", "-5",
            "--wired-margin-gib", "2",
        ])
        self.assertEqual(result.returncode, 64, msg=result.stdout + result.stderr)

    def test_wired_limit_mib_env_zero_errors_not_red(self):
        env = {k: v for k, v in os.environ.items()}
        env[FIT.ENV_WIRED_LIMIT_MIB] = "0"
        env.pop(FIT.ENV_KV_RESERVE_GIB, None)
        result = run_cli(
            [
                "--model-path", str(self.model_path),
                "--kv-reserve-gib", "0",
                "--wired-margin-gib", "2",
            ],
            env=env,
        )
        self.assertNotIn(result.returncode, (0, 2), msg=result.stdout + result.stderr)
        self.assertIn(FIT.ENV_WIRED_LIMIT_MIB, result.stderr)

    def test_ceiling_at_or_below_zero_is_a_config_error_not_red(self):
        # wired-limit-mib 2048 (2 GiB) - margin 2 GiB => ceiling == 0 bytes:
        # a nonsensical configuration (no headroom at all), refused as a
        # config error rather than silently treated as a real fit verdict.
        result = run_cli([
            "--model-path", str(self.model_path),
            "--kv-reserve-gib", "0",
            "--wired-limit-mib", "2048",
            "--wired-margin-gib", "2",
        ])
        self.assertNotIn(result.returncode, (0, 2), msg=result.stdout + result.stderr)
        self.assertIn("ceiling", result.stderr.lower())

    def test_resolve_wired_limit_falls_back_to_synthesized_when_sysctl_reports_zero(self):
        def stub_sysctl(name):
            if name == "iogpu.wired_limit_mb":
                return 0
            if name == "hw.memsize":
                return 16 * FIT.GIB
            return None

        wired_limit_mib, source = FIT.resolve_wired_limit_mib(
            None, None, sysctl_reader=stub_sysctl
        )
        self.assertEqual(source, "synthesized")
        self.assertEqual(wired_limit_mib, (16 * FIT.GIB * 3 // 4) // FIT.MIB)

    def test_resolve_wired_limit_falls_back_to_synthesized_when_sysctl_absent(self):
        def stub_sysctl(name):
            if name == "iogpu.wired_limit_mb":
                return None
            if name == "hw.memsize":
                return 8 * FIT.GIB
            return None

        wired_limit_mib, source = FIT.resolve_wired_limit_mib(
            None, None, sysctl_reader=stub_sysctl
        )
        self.assertEqual(source, "synthesized")
        self.assertEqual(wired_limit_mib, (8 * FIT.GIB * 3 // 4) // FIT.MIB)

    def test_resolve_wired_limit_rejects_non_positive_flag_value(self):
        with self.assertRaises(FIT.FitCheckError):
            FIT.resolve_wired_limit_mib(0, None)
        with self.assertRaises(FIT.FitCheckError):
            FIT.resolve_wired_limit_mib(-1, None)

    def test_resolve_wired_limit_rejects_non_positive_env_value(self):
        with self.assertRaises(FIT.FitCheckError):
            FIT.resolve_wired_limit_mib(None, 0)


class LargeMetadataArrayPerformanceTests(unittest.TestCase):
    """T14 (MEDIUM): parsing a header with very large metadata arrays (a
    200,000-element string array plus a 400,000-element int32 array) must
    be near-linear in the number of bytes, not quadratic -- a generous 3s
    bound that a quadratic buffer-copying reader blows through by an order
    of magnitude or more on a real ~5 MB tokenizer-heavy shard."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_large_string_and_int32_arrays_parse_under_three_seconds(self):
        n_strings = 200_000
        n_ints = 400_000
        strings = [f"tok{i}" for i in range(n_strings)]
        ints = list(range(n_ints))

        blob = bytearray()
        blob += b"GGUF"
        blob += struct.pack("<I", 3)
        blob += struct.pack("<Q", 0)  # tensor_count
        blob += struct.pack("<Q", 2)  # kv_count
        blob += _kv_string_array("tok.strings", strings)
        blob += _kv_int32_array("tok.ints", ints)
        path = write_file(self.root / "big_meta.gguf", bytes(blob))

        start = time.perf_counter()
        header = FIT.parse_gguf_header(path)
        elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 3.0, msg=f"parse took {elapsed:.2f}s (generous 3s bound)")
        self.assertEqual(len(header["metadata"]["tok.strings"]), n_strings)
        self.assertEqual(len(header["metadata"]["tok.ints"]), n_ints)
        self.assertEqual(header["metadata"]["tok.strings"][0], "tok0")
        self.assertEqual(header["metadata"]["tok.strings"][-1], f"tok{n_strings - 1}")
        self.assertEqual(header["metadata"]["tok.ints"][-1], n_ints - 1)


if __name__ == "__main__":
    unittest.main()
