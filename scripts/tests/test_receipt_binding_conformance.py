"""Shared receipt-binding conformance table, Python side.

``spike/Tests/HarnessCoreTests/Fixtures/receipt-binding-conformance-v1.json`` is
checked by BOTH this module (``fastmlx_launch.verify_receipt_binding``, used by
``fastmlx serve`` / ``fastmlx recommend``) and the Swift engine
(``PackReceiptBinding.verify``, ``ReceiptBindingConformanceTests``). The two are
one rule in two languages; if they drift, the launcher refuses a pack the
engine admits (or the reverse). The table pins the status and, for a mismatch,
the first difference's kind and path -- including the size-only case above a
lowered hash threshold, which pins the accepted 64 MiB limit in both languages --
and the ctime gate: a file above the threshold is also hashed when its ctime is
at or after the sibling receipt file's (or when the receipt cannot be stat'd).
The runner writes the receipt next to the pack and orders its ctime strictly
after every created file before applying a case's ``editedAfterReceipt`` /
``touchedAfterReceipt`` / ``receiptMissing`` fields.
"""

import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


LAUNCH_PATH = Path(__file__).resolve().parents[1] / "fastmlx_launch.py"
_SPEC = importlib.util.spec_from_file_location("fastmlx_launch", LAUNCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
FASTMLX_LAUNCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(FASTMLX_LAUNCH)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "spike/Tests/HarnessCoreTests/Fixtures/receipt-binding-conformance-v1.json"
)

# (kind, pattern) in the order the launcher's wording is tried. Each pattern's
# group 1 is the name, rendered by Python's repr for the plain names the table uses.
_DETAIL_PATTERNS = (
    ("missing", re.compile(r"^'(.+)' is listed in the receipt but missing$")),
    ("size", re.compile(r"^'(.+)' size differs \(receipt \d+, found \d+\)$")),
    ("sha256", re.compile(r"^'(.+)' sha256 differs \(receipt [0-9a-f]{12}\.\.\., found [0-9a-f]{12}\.\.\.\)$")),
    ("unreadable", re.compile(r"^'(.+)' could not be read \(.*\)$")),
    ("unlisted", re.compile(r"^'(.+)' is in the pack but not listed in the receipt$")),
)


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def kind_and_path(detail: str):
    for kind, pattern in _DETAIL_PATTERNS:
        match = pattern.match(detail)
        if match:
            return kind, match.group(1)
    raise AssertionError(f"unrecognized first-difference wording: {detail!r}")


def build_tree(root: Path, case: dict) -> None:
    for relative, spec in case.get("files", {}).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if "text" in spec:
            path.write_bytes(spec["text"].encode("utf-8"))
        else:
            path.write_bytes(b"x" * spec["size"])
    for relative in case.get("dirs", []):
        (root / relative).mkdir(parents=True, exist_ok=True)
    for link in case.get("symlinks", []):
        os.symlink(link["target"], root / link["link"])


def _ctime_ns(path) -> int:
    return os.stat(path).st_ctime_ns


def _bump_ctime(path) -> None:
    os.chmod(path, stat.S_IMODE(os.stat(path).st_mode))


def _tree_paths(root: Path) -> list:
    """Every created file and every symlink whose target resolves (stat follows)."""
    paths = []
    for current, _dirs, names in os.walk(root):
        for name in names:
            path = Path(current) / name
            if path.exists():
                paths.append(path)
    return paths


def write_receipt_and_order(pack: Path, receipt: dict) -> Path:
    """Write the sibling receipt and make its ctime STRICTLY greater than every
    file/symlink-target ctime in the pack (bounded chmod-bump retries)."""
    receipt_path = FASTMLX_LAUNCH.pull.receipt_path_for(pack)
    assert receipt_path.parent == pack.parent and receipt_path != pack
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    paths = _tree_paths(pack)
    for _ in range(500):
        if all(_ctime_ns(receipt_path) > _ctime_ns(path) for path in paths):
            return receipt_path
        time.sleep(0.002)
        _bump_ctime(receipt_path)
    raise AssertionError("could not order the receipt ctime after the pack files")


def ensure_not_before(path: Path, receipt_path: Path) -> None:
    for _ in range(500):
        if _ctime_ns(path) >= _ctime_ns(receipt_path):
            return
        time.sleep(0.002)
        _bump_ctime(path)
    raise AssertionError("could not order the file ctime at or after the receipt")


def apply_post_receipt_edits(pack: Path, case: dict, receipt_path: Path) -> None:
    for name in case.get("editedAfterReceipt", []):
        path = pack / name
        data = bytearray(path.read_bytes())
        spec = case["files"][name]
        if "size" in spec:
            assert data[0:1] == b"x"
            data[0] = ord("y")
        else:
            data[0] ^= 0x01
        with open(path, "r+b") as handle:  # in place, same size
            handle.write(bytes(data))
        ensure_not_before(path, receipt_path)
    for name in case.get("touchedAfterReceipt", []):
        ensure_not_before(pack / name, receipt_path)


def tree_entry(root: Path, relative: str) -> dict:
    data = (root / relative).read_bytes()
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def build_receipt(root: Path, case: dict) -> dict:
    receipt = {"repo_id": "example-org/Pack", "revision": "a" * 40}
    if "receiptFiles" not in case:
        return receipt
    spec = case["receiptFiles"]
    if spec != "fromTree":
        receipt["files"] = spec
        return receipt
    exclude = set(case.get("receiptExclude", []))
    files = {
        relative: tree_entry(root, relative)
        for relative in case.get("files", {})
        if relative not in exclude
    }
    for relative, edits in case.get("receiptOverrides", {}).items():
        entry = files[relative]
        for key, value in edits.items():
            if key == "upperSha256":
                entry["sha256"] = entry["sha256"].upper()
            elif value is None:
                entry.pop(key, None)
            else:
                entry[key] = value
    for relative, entry in case.get("receiptExtra", {}).items():
        built = tree_entry(root, entry["copyFrom"]) if "copyFrom" in entry else {}
        for key, value in entry.items():
            if key != "copyFrom":
                built[key] = value
        files[relative] = built
    receipt["files"] = files
    return receipt


class ReceiptBindingConformanceTests(unittest.TestCase):
    def test_fixture_is_well_formed(self):
        fixture = load_fixture()
        self.assertEqual(fixture["schema"], "receipt-binding-conformance-v1")
        ids = [case["id"] for case in fixture["cases"]]
        self.assertEqual(len(ids), len(set(ids)), "duplicate case ids")
        for case in fixture["cases"]:
            expect = case["expect"]
            self.assertIn(expect["status"], ("ok", "mismatch", "uncheckable"), case["id"])
            if expect["status"] == "mismatch":
                self.assertIsNotNone(expect["kind"], case["id"])
                self.assertIsNotNone(expect["path"], case["id"])
            else:
                self.assertIsNone(expect["kind"], case["id"])
                self.assertIsNone(expect["path"], case["id"])
        # The categories this table exists to pin must not be silently dropped.
        required = {
            "ok-matching-tree",
            "size-differs",
            "sha256-differs",
            "missing-listed-file",
            "unlisted-top-level-file",
            "unlisted-nested-file",
            "dot-components-are-ignored",
            "nested-listed-file-ok",
            "listed-prefix-is-a-file",
            "size-only-above-hash-threshold",
            "large-file-edited-after-receipt-is-hashed",
            "large-file-restored-after-receipt-is-ok",
            "receipt-unstatable-hashes-large-files",
            "hash-threshold-is-inclusive",
            "uncheckable-no-files-key",
            "uncheckable-name-absolute",
            "uncheckable-entry-is-a-string",
            "uncheckable-size-bool",
            "uncheckable-sha256-short",
        }
        self.assertLessEqual(required, set(ids))
        statuses = {case["expect"]["status"] for case in fixture["cases"]}
        self.assertEqual(statuses, {"ok", "mismatch", "uncheckable"})
        kinds = {case["expect"]["kind"] for case in fixture["cases"]}
        self.assertLessEqual({"missing", "size", "sha256", "unlisted"}, kinds)
        self.assertTrue(any("hashMaxBytes" in case for case in fixture["cases"]))

    def test_python_verify_receipt_binding_matches_the_shared_table(self):
        launch = FASTMLX_LAUNCH
        for case in load_fixture()["cases"]:
            with self.subTest(case=case["id"]):
                root = Path(tempfile.mkdtemp(prefix="receipt-binding-"))
                try:
                    pack = root / "pack"
                    pack.mkdir()
                    build_tree(pack, case)
                    receipt = build_receipt(pack, case)
                    if not case.get("receiptMissing", False):
                        receipt_path = write_receipt_and_order(pack, receipt)
                        apply_post_receipt_edits(pack, case, receipt_path)
                    threshold = case.get("hashMaxBytes", launch.RECEIPT_HASH_MAX_BYTES)
                    with mock.patch.object(launch, "RECEIPT_HASH_MAX_BYTES", threshold):
                        result = launch.verify_receipt_binding(pack, receipt)
                    expect = case["expect"]
                    self.assertEqual(result.status, expect["status"], f"{case['why']} -> {result}")
                    if expect["status"] == "ok":
                        self.assertIsNone(result.detail)
                    elif expect["status"] == "uncheckable":
                        self.assertIn(expect["reasonContains"], result.detail, case["why"])
                    else:
                        self.assertEqual(
                            kind_and_path(result.detail),
                            (expect["kind"], expect["path"]),
                            case["why"],
                        )
                finally:
                    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
