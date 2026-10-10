"""Shared pack-identity conformance table, Python side.

``spike/Tests/HarnessCoreTests/Fixtures/pack-identity-conformance-v1.json`` is
checked by BOTH this module (``fastmlx_launch.derive_pack_identity``, used by
``fastmlx recommend`` / serve before download) and the Swift engine
(``PackIdentity.derive``, ``PackIdentityConformanceTests``). The two are one
rule in two languages; if they drift, recommend vouches for a pack the engine
refuses. The Swift engine is the authority.
"""

import base64
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path


LAUNCH_PATH = Path(__file__).resolve().parents[1] / "fastmlx_launch.py"
_SPEC = importlib.util.spec_from_file_location("fastmlx_launch", LAUNCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
FASTMLX_LAUNCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(FASTMLX_LAUNCH)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "spike/Tests/HarnessCoreTests/Fixtures/pack-identity-conformance-v1.json"
)


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def build_case(root: str, case: dict) -> str:
    """Materialize ``case`` under ``root``; return the input path string."""
    model_dir = f"{root}/{case['modelDir']}"
    if case.get("createModelDir", True):
        os.makedirs(model_dir)
    os.makedirs(f"{root}/other-dir", exist_ok=True)
    for extra in case.get("extraDirs", []):
        os.makedirs(f"{root}/{extra}", exist_ok=True)
    for link in case.get("symlinks", []):
        link_path = f"{root}/{link['link']}"
        os.makedirs(os.path.dirname(link_path), exist_ok=True)
        os.symlink(f"{root}/{link['target']}", link_path)
    receipt = case.get("receipt")
    if receipt is not None:
        if "base64" in receipt:
            data = base64.b64decode(receipt["base64"])
        else:
            text = (
                receipt["text"]
                .replace("{DIR_RESOLVED}", os.path.realpath(model_dir))
                .replace("{DIR}", model_dir)
                .replace("{OTHER}", f"{root}/other-dir")
                .replace("{ROOT}", root)
            )
            data = text.encode("utf-8")
        base = f"{root}/{case.get('receiptBase') or case['modelDir']}"
        with open(f"{base}.pull-receipt.json", "wb") as handle:
            handle.write(data)
    input_path = case.get("inputPath") or case["modelDir"]
    return f"{root}/{input_path}"


class PackIdentityConformanceTests(unittest.TestCase):
    def test_fixture_is_well_formed(self):
        fixture = load_fixture()
        self.assertEqual(fixture["schema"], "pack-identity-conformance-v1")
        ids = [case["id"] for case in fixture["cases"]]
        self.assertEqual(len(ids), len(set(ids)), "duplicate case ids")
        for case in fixture["cases"]:
            self.assertIn("expect", case, case["id"])
        # The categories this table exists to pin must not be silently dropped.
        required = {
            "no-receipt-plain-dir",
            "receipt-dest-other-existing-dir",
            "receipt-invalid-utf8-lead-byte",
            "receipt-utf8-bom",
            "receipt-nan-alongside-repo",
            "receipt-duplicate-keys",
            "hub-snapshot-64-hex",
            "hub-nested-innermost-wins",
            "receipt-and-hub-different-repo-conflict",
        }
        self.assertLessEqual(required, set(ids))
        self.assertTrue(any("conflict" in (c["expect"] or {}) for c in fixture["cases"]))
        for case in fixture["cases"]:
            known = case.get("knownPythonDivergence")
            if known is not None:
                # Only the conservative direction is tolerated: Python may read
                # NO receipt where the engine reads one, never the reverse.
                self.assertIsNone(known["pythonExpect"], case["id"])
                self.assertIsNotNone(case["expect"], case["id"])

    def test_python_derive_pack_identity_matches_the_shared_table(self):
        launch = FASTMLX_LAUNCH
        for case in load_fixture()["cases"]:
            with self.subTest(case=case["id"]):
                root = tempfile.mkdtemp(prefix="packid-")
                try:
                    input_path = build_case(root, case)
                    expect = case["expect"]
                    known = case.get("knownPythonDivergence")
                    if known is not None:
                        # A documented, conservative-direction difference: pin
                        # the Python behavior so a change is noticed.
                        self.assertTrue(known["reason"])
                        expect = known["pythonExpect"]
                    if expect is not None and "conflict" in expect:
                        with self.assertRaises(launch.LaunchRefusal) as caught:
                            launch.derive_pack_identity(Path(input_path))
                        self.assertEqual(caught.exception.exit_code, 2)
                        message = caught.exception.message
                        self.assertIn(launch.PACK_IDENTITY_CONFLICT_REASON, message)
                        self.assertIn(repr(expect["conflict"]["receiptRepo"]), message)
                        self.assertIn(repr(expect["conflict"]["pathRepo"]), message)
                        continue
                    actual = launch.derive_pack_identity(Path(input_path))
                    if expect is None:
                        self.assertIsNone(actual, case["why"])
                    else:
                        self.assertEqual(
                            actual, (expect["repo"], expect["revision"]), case["why"]
                        )
                finally:
                    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
