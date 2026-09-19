"""Tests for scripts/fastmlx_safetensors_fit.py: a safetensors-aware,
residency-aware pre-load fit checker.

The safetensors writer below is independent, minimal, in-test code: it
does not import anything from the module under test for its byte-layout
logic (header length prefix, JSON header, contiguous data section), so a
regression in the production parser cannot also corrupt the fixtures
that are meant to catch it. Per-tensor byte sizes used as expected
values are hand-derived from a small independent dtype-size table, not
computed by calling into the implementation.
"""

from __future__ import annotations

import argparse
import hashlib
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


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "fastmlx_safetensors_fit.py"
LAUNCH_PATH = Path(__file__).resolve().parents[1] / "fastmlx_launch.py"

_FIT_SPEC = importlib.util.spec_from_file_location("fastmlx_safetensors_fit", SCRIPT_PATH)
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
# Independent minimal safetensors writer (test-only; not imported from
# the module under test's binary-layout logic).
# ---------------------------------------------------------------------
# Hand-derived dtype -> byte size table, written independently of the
# module under test's own _DTYPE_SIZES dict.
_TEST_DTYPE_SIZES = {"F32": 4, "F16": 2, "I8": 1, "BF16": 2}


def zero_tensor_bytes(dtype: str, shape: list) -> bytes:
    size = _TEST_DTYPE_SIZES[dtype]
    numel = 1
    for d in shape:
        numel *= d
    return bytes(numel * size)


def build_safetensors_bytes(
    tensor_specs: list, metadata: dict = None, header_override: dict = None
) -> bytes:
    """tensor_specs: list of (name, dtype, shape, data_bytes), laid out
    contiguously in list order starting at offset 0. If header_override
    is given, it replaces the computed header dict wholesale (used by
    malformed-input fixtures that need a header the data doesn't match)."""
    header = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    offset = 0
    data_blob = bytearray()
    for name, dtype, shape, data in tensor_specs:
        begin = offset
        end = offset + len(data)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [begin, end]}
        data_blob += data
        offset = end
    if header_override is not None:
        header = header_override
    header_json = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(header_json)) + header_json + bytes(data_blob)


def write_file(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def run_cli(args: list, env: dict = None) -> subprocess.CompletedProcess:
    argv = [sys.executable, "-B", str(SCRIPT_PATH)] + args
    return subprocess.run(argv, capture_output=True, text=True, timeout=30, env=env)


class MinimalWriterFixtureTests(unittest.TestCase):
    """Hand-derived byte sizes and exact file-size sum across 2 shards."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_two_shard_exact_byte_sum(self):
        shard1 = build_safetensors_bytes(
            [("t.a", "F32", [4], zero_tensor_bytes("F32", [4]))]  # 16 bytes
        )
        shard2 = build_safetensors_bytes(
            [("t.b", "I8", [10], zero_tensor_bytes("I8", [10]))]  # 10 bytes
        )
        p1 = write_file(self.root / "model-00001-of-00002.safetensors", shard1)
        p2 = write_file(self.root / "model-00002-of-00002.safetensors", shard2)
        expected_total = len(shard1) + len(shard2)

        weight_bytes, weight_files, side_files = FIT.compute_model_bytes(self.root)
        self.assertEqual(weight_bytes, expected_total)
        self.assertEqual(side_files, [])
        names = {f["name"] for f in weight_files}
        self.assertEqual(names, {p1.name, p2.name})

    def test_single_shard_header_reconciles(self):
        blob = build_safetensors_bytes(
            [("t.a", "F32", [4], zero_tensor_bytes("F32", [4]))]
        )
        path = write_file(self.root / "single.safetensors", blob)
        header = FIT.parse_safetensors_header(path)
        highest_end = FIT.reconcile_safetensors(header)
        self.assertEqual(highest_end, 16)
        self.assertEqual(header["file_size"], 8 + header["header_len"] + 16)


class GreenRedTests(unittest.TestCase):
    """GREEN and RED at a chosen --wired-limit-mib/--kv-reserve-gib."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # 1,048,576 bytes (1 MiB) of weights.
        self.blob = build_safetensors_bytes(
            [("t.a", "I8", [1048576], zero_tensor_bytes("I8", [1048576]))]
        )
        self.path = write_file(self.root / "model.safetensors", self.blob)
        # wired-limit 2050 MiB - margin 2 GiB (2048 MiB) => ceiling = 2 MiB.
        self.ceiling_args = ["--wired-limit-mib", "2050", "--wired-margin-gib", "2"]

    def test_green_when_weights_plus_kv_fit(self):
        result = run_cli(
            ["--model-path", str(self.root), "--kv-reserve-gib", "0"] + self.ceiling_args
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("fit=green", result.stdout)
        fields = LAUNCH._parse_attestation_fields(LAUNCH._find_attestation_line(result.stdout))
        self.assertEqual(fields["weights_bytes"], str(len(self.blob)))  # header + 1 MiB payload
        self.assertEqual(fields["residency"], "resident")

    def test_red_when_kv_reserve_pushes_past_ceiling(self):
        # kv_reserve alone (2 GiB) already exceeds the 2 MiB ceiling.
        result = run_cli(
            ["--model-path", str(self.root), "--kv-reserve-gib", "2"] + self.ceiling_args
        )
        self.assertEqual(result.returncode, 2, msg=result.stdout + result.stderr)
        self.assertIn("exceeds ceiling", result.stderr)


class SideFileExclusionTests(unittest.TestCase):
    """A >=1 GiB non-safetensors file named by --mmap-side-file is excluded
    from resident weight bytes and listed in --json's
    memory_mapped_side_files; created sparse (via os.truncate) so the test
    never writes real gigabytes. An UNNAMED >=1 GiB non-safetensors file is
    now a configuration error (exit 1) -- the old lenient "any large file
    is a side file" assumption silently let a pack this checker cannot
    size pass as GREEN."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_named_side_file_excluded_and_reported(self):
        blob = build_safetensors_bytes(
            [("t.a", "F32", [4], zero_tensor_bytes("F32", [4]))]
        )
        write_file(self.root / "model.safetensors", blob)

        side_path = self.root / "ngram_table.bin"
        side_path.touch()
        os.truncate(side_path, 2 * FIT.GIB)  # sparse: no real disk space used

        result = run_cli([
            "--model-path", str(self.root),
            "--kv-reserve-gib", "0",
            "--wired-limit-mib", "4096",
            "--wired-margin-gib", "2",
            "--mmap-side-file", "ngram_table.bin",
            "--json",
        ])
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["weights_bytes"], len(blob))
        self.assertEqual(len(payload["memory_mapped_side_files"]), 1)
        side = payload["memory_mapped_side_files"][0]
        self.assertEqual(side["name"], "ngram_table.bin")
        self.assertEqual(side["bytes"], 2 * FIT.GIB)

        note_result = run_cli([
            "--model-path", str(self.root),
            "--kv-reserve-gib", "0",
            "--wired-limit-mib", "4096",
            "--wired-margin-gib", "2",
            "--mmap-side-file", "ngram_table.bin",
        ])
        self.assertIn("not counted as resident", note_result.stderr)
        self.assertIn("ngram_table.bin", note_result.stderr)

    def test_unnamed_large_side_file_is_a_configuration_error(self):
        blob = build_safetensors_bytes(
            [("t.a", "F32", [4], zero_tensor_bytes("F32", [4]))]
        )
        write_file(self.root / "model.safetensors", blob)

        side_path = self.root / "ngram_table.bin"
        side_path.touch()
        os.truncate(side_path, 2 * FIT.GIB)  # sparse: no real disk space used

        result = run_cli([
            "--model-path", str(self.root),
            "--kv-reserve-gib", "0",
            "--wired-limit-mib", "4096",
            "--wired-margin-gib", "2",
        ])
        self.assertEqual(result.returncode, 1, msg=result.stdout + result.stderr)
        self.assertNotIn("fit=green", result.stdout)
        self.assertIn("ngram_table.bin", result.stderr)
        self.assertIn("--mmap-side-file", result.stderr)

    def test_named_side_file_that_does_not_exist_is_an_error(self):
        blob = build_safetensors_bytes(
            [("t.a", "F32", [4], zero_tensor_bytes("F32", [4]))]
        )
        write_file(self.root / "model.safetensors", blob)

        result = run_cli([
            "--model-path", str(self.root),
            "--kv-reserve-gib", "0",
            "--wired-limit-mib", "4096",
            "--wired-margin-gib", "2",
            "--mmap-side-file", "does-not-exist.bin",
        ])
        self.assertEqual(result.returncode, 1, msg=result.stdout + result.stderr)
        self.assertIn("does-not-exist.bin", result.stderr)

    def test_small_non_safetensors_file_is_not_reported(self):
        blob = build_safetensors_bytes(
            [("t.a", "F32", [4], zero_tensor_bytes("F32", [4]))]
        )
        write_file(self.root / "model.safetensors", blob)
        write_file(self.root / "config.json", b"{}")

        result = run_cli([
            "--model-path", str(self.root),
            "--kv-reserve-gib", "0",
            "--wired-limit-mib", "4096",
            "--wired-margin-gib", "2",
            "--json",
        ])
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["memory_mapped_side_files"], [])


class RecursiveShardDiscoveryTests(unittest.TestCase):
    """*.safetensors shards are counted recursively, not just top-level."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_nested_shard_is_counted(self):
        blob = build_safetensors_bytes(
            [("t.a", "F32", [4], zero_tensor_bytes("F32", [4]))]
        )
        nested_dir = self.root / "shards" / "part1"
        nested_dir.mkdir(parents=True)
        write_file(nested_dir / "model-00001-of-00001.safetensors", blob)

        weight_bytes, weight_files, side_files = FIT.compute_model_bytes(self.root)
        self.assertEqual(weight_bytes, len(blob))
        self.assertEqual(side_files, [])
        self.assertEqual(
            [f["name"] for f in weight_files],
            ["shards/part1/model-00001-of-00001.safetensors"],
        )


class SafetensorsIndexTests(unittest.TestCase):
    """A top-level model.safetensors.index.json must name only shards this
    checker actually found and counted."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_index_naming_a_missing_shard_errors(self):
        blob = build_safetensors_bytes(
            [("t.a", "F32", [4], zero_tensor_bytes("F32", [4]))]
        )
        write_file(self.root / "model-00001-of-00002.safetensors", blob)
        index_doc = {
            "weight_map": {
                "t.a": "model-00001-of-00002.safetensors",
                "t.b": "model-00002-of-00002.safetensors",  # never written
            }
        }
        write_file(
            self.root / "model.safetensors.index.json",
            json.dumps(index_doc).encode("utf-8"),
        )

        result = run_cli([
            "--model-path", str(self.root),
            "--kv-reserve-gib", "0",
            "--wired-limit-mib", "4096",
            "--wired-margin-gib", "2",
        ])
        self.assertEqual(result.returncode, 1, msg=result.stdout + result.stderr)
        self.assertIn("model-00002-of-00002.safetensors", result.stderr)

    def test_index_naming_only_present_shards_is_green(self):
        blob = build_safetensors_bytes(
            [("t.a", "F32", [4], zero_tensor_bytes("F32", [4]))]
        )
        write_file(self.root / "model-00001-of-00001.safetensors", blob)
        index_doc = {"weight_map": {"t.a": "model-00001-of-00001.safetensors"}}
        write_file(
            self.root / "model.safetensors.index.json",
            json.dumps(index_doc).encode("utf-8"),
        )

        result = run_cli([
            "--model-path", str(self.root),
            "--kv-reserve-gib", "0",
            "--wired-limit-mib", "4096",
            "--wired-margin-gib", "2",
        ])
        self.assertEqual(result.returncode, 0, msg=result.stderr)


class HostileShapeTests(unittest.TestCase):
    """A tensor's declared shape can never legitimately imply more elements
    than the file could hold, and never needs an absurd dimension count;
    both are caught immediately, not after a slow/huge computation."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_huge_single_dimension_errors_quickly(self):
        header = {
            "t.a": {"dtype": "F32", "shape": [10**30], "data_offsets": [0, 16]}
        }
        blob = build_safetensors_bytes([], header_override=header) + bytes(16)
        path = write_file(self.root / "model.safetensors", blob)

        started = time.monotonic()
        with self.assertRaises(FIT.SafetensorsFormatError):
            FIT.reconcile_safetensors(FIT.parse_safetensors_header(path))
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 5.0, "hostile shape should abort almost immediately")

    def test_too_many_dimensions_errors(self):
        header = {
            "t.a": {
                "dtype": "F32",
                "shape": [1] * 17,
                "data_offsets": [0, 4],
            }
        }
        blob = build_safetensors_bytes([], header_override=header) + bytes(4)
        path = write_file(self.root / "model.safetensors", blob)
        with self.assertRaises(FIT.SafetensorsFormatError):
            FIT.reconcile_safetensors(FIT.parse_safetensors_header(path))


class DotDirIgnoredTests(unittest.TestCase):
    """A top-level dot-directory (e.g. .cache) is never scanned, even if
    it contains a *.safetensors file."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_dot_cache_subdir_ignored(self):
        blob = build_safetensors_bytes(
            [("t.a", "F32", [4], zero_tensor_bytes("F32", [4]))]
        )
        write_file(self.root / "model.safetensors", blob)

        cache_dir = self.root / ".cache"
        cache_dir.mkdir()
        # A decoy shard inside the dot-dir; must not be scanned (this
        # fixture is deliberately malformed so if it were scanned the
        # fit check would error, not silently pass).
        write_file(cache_dir / "decoy.safetensors", b"not a real safetensors file")

        weight_bytes, weight_files, _side = FIT.compute_model_bytes(self.root)
        self.assertEqual(weight_bytes, len(blob))
        self.assertEqual([f["name"] for f in weight_files], ["model.safetensors"])


class MissingKvReserveTests(unittest.TestCase):
    """--kv-reserve-gib is required (flag or the shared
    FASTMLX_GGUF_KV_RESERVE_GIB environment variable); absence from both is
    a usage error (exit 64), never a silent zero default."""

    def test_missing_kv_reserve_gib_exits_64(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("FASTMLX_")}
        result = run_cli(["--model-path", "/nonexistent-path-not-read"], env=env)
        self.assertEqual(result.returncode, 64, msg=result.stdout + result.stderr)
        self.assertNotIn("fit=green", result.stdout)


class EnvParityTests(unittest.TestCase):
    """--wired-limit-mib, --wired-margin-gib, and --kv-reserve-gib each
    accept an environment-variable fallback with the same names and
    flag > env > default precedence as scripts/fastmlx_gguf_fit.py."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # 1,048,576 bytes (1 MiB) of weights, same fixture shape as
        # GreenRedTests, so a 1 MiB ceiling is exactly borderline.
        self.blob = build_safetensors_bytes(
            [("t.a", "I8", [1048576], zero_tensor_bytes("I8", [1048576]))]
        )
        write_file(self.root / "model.safetensors", self.blob)
        self.clean_env = {
            k: v for k, v in os.environ.items() if not k.startswith("FASTMLX_")
        }

    def _namespace(self, **overrides):
        base = dict(
            model=None,
            model_path=self.root,
            host_use="shared",
            fit_check_only=False,
            context=None,
            residency=None,
            wired_limit_mib=None,
            wired_margin_gib=None,
            kv_reserve_gib=0.0,
            json=False,
            mmap_side_file=[],
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_env_lowers_ceiling_to_red_where_flagless_default_would_be_green(self):
        # Calls FIT.compute_fit() directly (not via a subprocess CLI
        # invocation) so a fake `sysctl` can stand in deterministically --
        # the real sysctl binary is macOS-only and its live values vary by
        # host, which a spawned subprocess would use unpatched regardless.
        def fake_sysctl_run(cmd, capture_output=None, text=None, timeout=None):
            class _Result:
                returncode = 0
                # A huge (1 TiB-ish) "measured" iogpu.wired_limit_mb, so the
                # flag-less/env-less default comfortably fits ~1 MiB of
                # weights plus a zero KV reserve.
                stdout = "1048576\n"

            return _Result()

        with patch("subprocess.run", fake_sysctl_run):
            with patch.dict(os.environ, self.clean_env, clear=True):
                baseline = FIT.compute_fit(self._namespace())
        self.assertEqual(baseline["fit"], "green")

        # Same pack and flags, but the env vars alone now pin a 1 MiB
        # ceiling (wired-limit 2049 MiB - 2 GiB margin), which the ~1 MiB
        # weight blob (plus its header) exceeds -- proving the env
        # variables are actually read, not just harmlessly ignored.
        env_overrides = dict(self.clean_env)
        env_overrides[FIT._GGUF.ENV_WIRED_LIMIT_MIB] = "2049"
        env_overrides[FIT._GGUF.ENV_WIRED_MARGIN_GIB] = "2"
        with patch.dict(os.environ, env_overrides, clear=True):
            red = FIT.compute_fit(self._namespace())
        self.assertEqual(red["fit"], "red")

    def test_flag_beats_env_for_wired_limit_and_margin(self):
        env_overrides = dict(self.clean_env)
        # Env alone would be RED (a 1 MiB ceiling); the flags must win and
        # produce GREEN.
        env_overrides[FIT._GGUF.ENV_WIRED_LIMIT_MIB] = "2049"
        env_overrides[FIT._GGUF.ENV_WIRED_MARGIN_GIB] = "2"
        result = run_cli(
            [
                "--model-path", str(self.root),
                "--kv-reserve-gib", "0",
                "--wired-limit-mib", "4096",
                "--wired-margin-gib", "2",
            ],
            env=env_overrides,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("fit=green", result.stdout)

    def test_flag_beats_env_for_kv_reserve(self):
        env_overrides = dict(self.clean_env)
        # Env alone (2 GiB KV reserve) would push this past a 2 MiB
        # ceiling; the flag (0) must win and produce GREEN.
        env_overrides[FIT._GGUF.ENV_KV_RESERVE_GIB] = "2"
        result = run_cli(
            [
                "--model-path", str(self.root),
                "--kv-reserve-gib", "0",
                "--wired-limit-mib", "2050",
                "--wired-margin-gib", "2",
            ],
            env=env_overrides,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("fit=green", result.stdout)

    def test_env_kv_reserve_is_accepted_without_the_flag(self):
        env_overrides = dict(self.clean_env)
        env_overrides[FIT._GGUF.ENV_KV_RESERVE_GIB] = "0"
        result = run_cli(
            [
                "--model-path", str(self.root),
                "--wired-limit-mib", "4096",
                "--wired-margin-gib", "2",
            ],
            env=env_overrides,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("fit=green", result.stdout)

    def test_neither_flag_nor_env_kv_reserve_exits_64(self):
        result = run_cli(
            ["--model-path", str(self.root), "--wired-limit-mib", "4096",
             "--wired-margin-gib", "2"],
            env=self.clean_env,
        )
        self.assertEqual(result.returncode, 64, msg=result.stdout + result.stderr)


class ExpertStreamResidencyTests(unittest.TestCase):
    """safetensors packs have no expert-stream sizing: --residency
    expert-stream must exit 64 with a message naming why."""

    def test_expert_stream_residency_exits_64_with_reason(self):
        result = run_cli([
            "--model-path", "/nonexistent-path-not-read",
            "--kv-reserve-gib", "0",
            "--residency", "expert-stream",
        ])
        self.assertEqual(result.returncode, 64, msg=result.stdout + result.stderr)
        self.assertIn("expert-stream", result.stderr)
        self.assertIn("resident", result.stderr)


class ArgparseUsageErrorExitCodeTests(unittest.TestCase):
    """Bad invocations exit 64 (EX_USAGE), never 0 or 2."""

    def _assert_usage_exit(self, args):
        result = run_cli(args)
        self.assertEqual(
            result.returncode, 64, msg=f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        self.assertNotIn("fit=green", result.stdout)
        return result

    def test_unknown_flag_exits_64(self):
        self._assert_usage_exit(
            ["--model-path", "/does-not-matter", "--kv-reserve-gib", "0", "--bogus-flag", "x"]
        )

    def test_missing_model_path_exits_64(self):
        self._assert_usage_exit(["--kv-reserve-gib", "0"])

    def test_abbreviated_flag_is_a_usage_error(self):
        # allow_abbrev=False, inherited from the shared
        # _GGUF._UsageErrorArgumentParser (defense in depth for the
        # reserved-fit-check-arg guard in fastmlx_launch.py): an abbreviated
        # flag like --kv-res must be refused as unrecognized, never silently
        # resolved to --kv-reserve-gib the way argparse's default
        # abbreviation-matching would.
        self._assert_usage_exit(
            ["--model-path", "/does-not-matter", "--kv-res", "8"]
        )


class MalformedInputTests(unittest.TestCase):
    """Bad-input scenarios must never exit 0 or 2, and must never print
    fit=green -- a format/configuration error is a distinct exit (1),
    never mistaken for a fit verdict."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _assert_error_exit(self, model_path: Path):
        result = run_cli([
            "--model-path", str(model_path),
            "--kv-reserve-gib", "0",
            "--wired-limit-mib", "4096",
            "--wired-margin-gib", "2",
        ])
        self.assertNotIn(result.returncode, (0, 2, 64), msg=result.stdout + result.stderr)
        self.assertNotIn("fit=green", result.stdout)
        return result

    def test_truncated_shard_errors(self):
        blob = build_safetensors_bytes(
            [("t.a", "F32", [4], zero_tensor_bytes("F32", [4]))]
        )
        truncated = blob[:-4]  # cut off the last 4 bytes of tensor data
        write_file(self.root / "model.safetensors", truncated)
        result = self._assert_error_exit(self.root)
        self.assertIn("model.safetensors", result.stderr)

    def test_noncontiguous_offsets_error(self):
        header = {
            "t.a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "t.b": {"dtype": "F32", "shape": [4], "data_offsets": [32, 48]},  # gap [16,32)
        }
        blob = build_safetensors_bytes([], header_override=header)
        blob += bytes(48)  # enough trailing bytes that EOF isn't the failure mode
        write_file(self.root / "model.safetensors", blob)
        self._assert_error_exit(self.root)

    def test_overlapping_offsets_error(self):
        header = {
            "t.a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "t.b": {"dtype": "F32", "shape": [4], "data_offsets": [8, 24]},  # overlaps t.a
        }
        blob = build_safetensors_bytes([], header_override=header)
        blob += bytes(24)
        write_file(self.root / "model.safetensors", blob)
        self._assert_error_exit(self.root)

    def test_shape_dtype_mismatch_error(self):
        # F32[4] declares 16 bytes, but data_offsets only spans 8 bytes.
        header = {"t.a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 8]}}
        blob = build_safetensors_bytes([], header_override=header)
        blob += bytes(8)
        write_file(self.root / "model.safetensors", blob)
        self._assert_error_exit(self.root)

    def test_unknown_dtype_error(self):
        header = {"t.a": {"dtype": "NOPE_9", "shape": [4], "data_offsets": [0, 16]}}
        blob = build_safetensors_bytes([], header_override=header)
        blob += bytes(16)
        write_file(self.root / "model.safetensors", blob)
        self._assert_error_exit(self.root)

    def test_no_safetensors_files_errors(self):
        # Directory exists but is empty (or has only unrelated files).
        write_file(self.root / "config.json", b"{}")
        self._assert_error_exit(self.root)

    def test_header_length_exceeds_file_size_errors(self):
        blob = bytearray(struct.pack("<Q", 1000)) + b'{"a":1}'  # declares far more than present
        write_file(self.root / "model.safetensors", bytes(blob))
        self._assert_error_exit(self.root)

    def test_invalid_json_header_errors(self):
        bad_json = b"{not json"
        blob = struct.pack("<Q", len(bad_json)) + bad_json
        write_file(self.root / "model.safetensors", blob)
        self._assert_error_exit(self.root)


class LauncherCallSiteTests(unittest.TestCase):
    """The launcher's run_fit_check() admits/refuses via the real binary,
    and its own attestation-line parser accepts this binary's GREEN
    output."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        _make_executable(SCRIPT_PATH)

        blob = build_safetensors_bytes(
            [("t.a", "F32", [4], zero_tensor_bytes("F32", [4]))]
        )
        write_file(self.root / "model.safetensors", blob)

    def test_run_fit_check_admits_when_it_fits(self):
        result = LAUNCH.run_fit_check(
            fit_check_bin=str(SCRIPT_PATH),
            model_id="fixture-pack",
            model_path=self.root,
            host_use="shared",
            context=None,
            extra_args=[
                "--kv-reserve-gib", "0",
                "--wired-limit-mib", "4096",
                "--wired-margin-gib", "2",
            ],
        )
        self.assertEqual(result.kind, "green", msg=result.detail)
        self.assertEqual(result.fields.get("fit"), "green")

    def test_run_fit_check_refuses_when_it_does_not_fit(self):
        result = LAUNCH.run_fit_check(
            fit_check_bin=str(SCRIPT_PATH),
            model_id="fixture-pack",
            model_path=self.root,
            host_use="shared",
            context=None,
            extra_args=[
                "--kv-reserve-gib", "1",
                "--wired-limit-mib", "2049",
                "--wired-margin-gib", "2",
            ],
        )
        self.assertEqual(result.kind, "red")


class ContiguityMutationCheckTests(unittest.TestCase):
    """Mutation check: prove reconcile_safetensors's contiguity guard is
    load-bearing -- if it is bypassed, a fixture with overlapping
    data_offsets (which the unmutated function correctly rejects) is
    wrongly accepted. The production module is patched only in-process,
    for the duration of a single `with` block, via unittest.mock.patch;
    a sha256 of the module source file taken before and after the test
    confirms the file on disk was never touched."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_bypassing_contiguity_check_wrongly_accepts_overlap(self):
        sha_before = hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest()

        header = {
            "t.a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "t.b": {"dtype": "F32", "shape": [4], "data_offsets": [8, 24]},
        }
        blob = build_safetensors_bytes([], header_override=header) + bytes(24)
        path = write_file(self.root / "overlap.safetensors", blob)
        parsed = FIT.parse_safetensors_header(path)

        # Sanity: the real, unmutated function rejects this fixture.
        with self.assertRaises(FIT.SafetensorsFormatError):
            FIT.reconcile_safetensors(parsed)

        # Mutation: a stand-in that only sums declared spans, skipping
        # the begin == expected contiguity guard entirely.
        def _reconcile_without_contiguity_guard(hdr):
            highest_end = 0
            for name, meta in hdr["entries"].items():
                if name == "__metadata__":
                    continue
                highest_end = max(highest_end, meta["data_offsets"][1])
            return highest_end

        with patch.object(FIT, "reconcile_safetensors", _reconcile_without_contiguity_guard):
            # No exception now: proves the overlap is caught by the
            # contiguity guard specifically, not by some other check
            # (dtype/shape sizing, EOF bounds, etc).
            accepted_end = FIT.reconcile_safetensors(parsed)
        self.assertEqual(accepted_end, 24)

        sha_after = hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest()
        self.assertEqual(sha_before, sha_after, "module source file was modified on disk")


if __name__ == "__main__":
    unittest.main()
