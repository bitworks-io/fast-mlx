"""A large pack file edited after its pull receipt is hashed (cycle 251).

``verify_receipt_binding`` used to check a listed file larger than
``RECEIPT_HASH_MAX_BYTES`` by size only. It now ALSO sha256-compares such a file
when its ctime (symlinks followed) is >= the sibling receipt file's ctime at
nanosecond resolution, and hashes every listed file when the receipt cannot be
stat'd. Predeclaration:
docs/task-inbox/2026-10-15-PREDECLARATION-large-file-edited-after-the-receipt-is-hashed.md

The pack is built with a lowered threshold so the "large" file is a few bytes.
Real ctimes are used wherever possible; the equal-ctime edge cannot be built
deterministically on disk, so it patches ``_stat_ctime_ns``.
"""

import hashlib
import importlib.util
import json
import os
import shutil
import stat
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FASTMLX_LAUNCH = _load("fastmlx_launch", "fastmlx_launch.py")
FASTMLX_RECOMMEND = _load("fastmlx_recommend", "fastmlx_recommend.py")

REPO = "example-org/Pack"
REVISION = "a" * 40
THRESHOLD = 20
CONFIG = b'{"model_type": "x"}\n'  # 20 bytes: exactly at the threshold
BIG = b"A" * 64  # above the threshold
CARD = {"id": "pack@test", "model": {"repo": REPO, "hfPin": None}, "verdict": "PASS"}


def _ctime_ns(path) -> int:
    return os.stat(path).st_ctime_ns


def _bump(path) -> None:
    mode = stat.S_IMODE(os.stat(path).st_mode)
    os.chmod(path, mode)


def ensure_receipt_after(receipt: Path, others, attempts: int = 500) -> None:
    """Make the receipt's ctime STRICTLY greater than every path in ``others``."""
    for _ in range(attempts):
        if all(_ctime_ns(receipt) > _ctime_ns(other) for other in others):
            return
        time.sleep(0.002)
        _bump(receipt)
    raise AssertionError("could not order the receipt ctime after the pack files")


def ensure_not_before(path: Path, receipt: Path, attempts: int = 500) -> None:
    """Make ``path``'s ctime >= the receipt's."""
    for _ in range(attempts):
        if _ctime_ns(path) >= _ctime_ns(receipt):
            return
        time.sleep(0.002)
        _bump(path)
    raise AssertionError("could not order the file ctime at or after the receipt")


def entry(data: bytes) -> dict:
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


class CtimeGateTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="receipt-ctime-gate-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.pack = self.tmp / "pack"
        self.pack.mkdir()
        (self.pack / "config.json").write_bytes(CONFIG)
        (self.pack / "big.bin").write_bytes(BIG)
        self.receipt = {
            "repo_id": REPO,
            "revision": REVISION,
            "files": {"config.json": entry(CONFIG), "big.bin": entry(BIG)},
        }
        self.receipt_path = FASTMLX_LAUNCH.pull.receipt_path_for(self.pack)
        patcher = mock.patch.object(FASTMLX_LAUNCH, "RECEIPT_HASH_MAX_BYTES", THRESHOLD)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_receipt(self) -> None:
        self.receipt_path.write_text(json.dumps(self.receipt), encoding="utf-8")
        ensure_receipt_after(
            self.receipt_path, [self.pack / "config.json", self.pack / "big.bin"]
        )

    def edit_big_after_receipt(self, new_bytes: bytes) -> None:
        self.assertEqual(len(new_bytes), len(BIG))
        target = self.pack / "big.bin"
        with open(target, "r+b") as handle:  # in place, same size
            handle.write(new_bytes)
        ensure_not_before(target, self.receipt_path)

    def verify(self):
        return FASTMLX_LAUNCH.verify_receipt_binding(self.pack, self.receipt)

    def hashed_names(self, call):
        hashed: list = []
        real = FASTMLX_LAUNCH._receipt_file_sha256

        def counting(path):
            hashed.append(Path(path).name)
            return real(path)

        with mock.patch.object(FASTMLX_LAUNCH, "_receipt_file_sha256", counting):
            result = call()
        return result, hashed

    # T1
    def test_t1_a_large_file_edited_same_size_after_the_receipt_is_a_sha256_mismatch(self):
        self.write_receipt()
        self.edit_big_after_receipt(b"B" + BIG[1:])
        result = self.verify()
        self.assertEqual(result.status, FASTMLX_LAUNCH.RECEIPT_BINDING_MISMATCH)
        self.assertIn("'big.bin' sha256 differs", result.detail)

    # T2
    def test_t2_judge_refuses_with_the_receipt_mismatch_reason_for_a_matching_card(self):
        self.write_receipt()
        self.edit_big_after_receipt(b"B" + BIG[1:])
        judgement = FASTMLX_LAUNCH.judge_receipt_binding(self.pack, CARD)
        self.assertEqual(judgement.kind, FASTMLX_LAUNCH.RECEIPT_JUDGEMENT_REFUSE)
        self.assertIn("quality_card_receipt_mismatch", judgement.message)
        self.assertIn("'big.bin' sha256 differs", judgement.message)

    # T3
    def test_t3_recommend_reports_an_error_row_for_the_edited_pack(self):
        self.write_receipt()
        self.edit_big_after_receipt(b"B" + BIG[1:])
        with mock.patch.object(FASTMLX_RECOMMEND.launch, "RECEIPT_HASH_MAX_BYTES", THRESHOLD):
            row = FASTMLX_RECOMMEND.build_row(
                model_path=self.pack,
                cards=[CARD],
                fit_check_bin="/nonexistent/fit-check",
                host_use="shared",
                context=None,
                fit_check_args=[],
                host_hardware_class=lambda: "apple-m5",
            )
        self.assertEqual(row["status"], FASTMLX_RECOMMEND.STATUS_ERROR)
        self.assertIn("quality_card_receipt_mismatch", row["message"])
        self.assertIn("'big.bin' sha256 differs", row["message"])

    # T4 (control)
    def test_t4_an_untouched_large_file_older_than_the_receipt_is_never_opened(self):
        self.receipt["files"]["big.bin"]["sha256"] = "a" * 64  # wrong, never consulted
        self.write_receipt()
        result, hashed = self.hashed_names(self.verify)
        self.assertEqual(result.status, FASTMLX_LAUNCH.RECEIPT_BINDING_OK, result)
        self.assertEqual(hashed, ["config.json"])

    # T5
    def _with_ctimes(self, file_ctime: int, receipt_ctime: int):
        self.receipt["files"]["big.bin"]["sha256"] = "a" * 64
        self.write_receipt()
        receipt_name = self.receipt_path.name

        def fake(path):
            return receipt_ctime if Path(path).name == receipt_name else file_ctime

        with mock.patch.object(FASTMLX_LAUNCH, "_stat_ctime_ns", fake):
            return self.hashed_names(self.verify)

    def test_t5_equal_ctime_is_hashed(self):
        result, hashed = self._with_ctimes(1_000, 1_000)
        self.assertEqual(result.status, FASTMLX_LAUNCH.RECEIPT_BINDING_MISMATCH)
        self.assertIn("'big.bin' sha256 differs", result.detail)
        self.assertIn("big.bin", hashed)

    def test_t5_one_nanosecond_older_than_the_receipt_is_not_hashed(self):
        result, hashed = self._with_ctimes(999, 1_000)
        self.assertEqual(result.status, FASTMLX_LAUNCH.RECEIPT_BINDING_OK, result)
        self.assertNotIn("big.bin", hashed)

    def test_t5_one_nanosecond_newer_than_the_receipt_is_hashed(self):
        result, hashed = self._with_ctimes(1_001, 1_000)
        self.assertEqual(result.status, FASTMLX_LAUNCH.RECEIPT_BINDING_MISMATCH)
        self.assertIn("big.bin", hashed)

    # T6
    def test_t6_a_receipt_that_cannot_be_statted_hashes_a_large_file(self):
        self.receipt["files"]["big.bin"]["sha256"] = "a" * 64
        self.assertFalse(self.receipt_path.exists())
        result, hashed = self.hashed_names(self.verify)
        self.assertEqual(result.status, FASTMLX_LAUNCH.RECEIPT_BINDING_MISMATCH)
        self.assertIn("'big.bin' sha256 differs", result.detail)
        self.assertIn("big.bin", hashed)

    def test_t6_a_receipt_that_cannot_be_statted_hashes_every_listed_file(self):
        self.assertFalse(self.receipt_path.exists())
        result, hashed = self.hashed_names(self.verify)
        self.assertEqual(result.status, FASTMLX_LAUNCH.RECEIPT_BINDING_OK, result)
        self.assertEqual(sorted(hashed), ["big.bin", "config.json"])

    # T7 (control)
    def test_t7_restoring_the_bytes_after_the_edit_is_ok_because_the_hash_decides(self):
        self.write_receipt()
        self.edit_big_after_receipt(b"B" + BIG[1:])
        self.edit_big_after_receipt(BIG)
        result, hashed = self.hashed_names(self.verify)
        self.assertEqual(result.status, FASTMLX_LAUNCH.RECEIPT_BINDING_OK, result)
        self.assertIn("big.bin", hashed)

    def test_t7_a_chmod_only_touch_after_the_receipt_is_ok_and_hashed(self):
        self.write_receipt()
        ensure_not_before(self.pack / "big.bin", self.receipt_path)
        result, hashed = self.hashed_names(self.verify)
        self.assertEqual(result.status, FASTMLX_LAUNCH.RECEIPT_BINDING_OK, result)
        self.assertIn("big.bin", hashed)

    def test_the_threshold_is_read_at_call_time(self):
        self.receipt["files"]["big.bin"]["sha256"] = "a" * 64
        self.write_receipt()
        with mock.patch.object(FASTMLX_LAUNCH, "RECEIPT_HASH_MAX_BYTES", len(BIG)):
            result = self.verify()
        self.assertEqual(result.status, FASTMLX_LAUNCH.RECEIPT_BINDING_MISMATCH)


if __name__ == "__main__":
    unittest.main()
