"""Tests for a recommend row's `mtp` object (`docs/quality-card-schema-v1.md`
"Flag transfer") -- following the exact pattern
`EngineBuildRecommendTestCase` in `test_fastmlx_recommend.py` set for
`engineBuild`. A NEW file that imports the shared fixtures/helpers that
file already exports.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.tests.test_fastmlx_recommend import (
    FASTMLX_RECOMMEND,
    GREEN_FIT_CHECK_BODY,
    write_pull_receipt,
    write_script,
)

MTP_CARD_COMMIT = "77" * 20
MTP_OTHER_COMMIT = "88" * 20

MTP_NOT_EXACT_REPO = "example/RecommendMtpNotExactModel"
MTP_NOT_EXACT_CARD_ID = "recommend-mtp-not-exact@test"

# Unvalidated shape (`load_quality_cards` never validates a card against
# the schema): `greedy` outside the three known values. Exercises the
# fail-open rule in `launch.mtp_transfer_status`, shared by
# `fastmlx_recommend.build_row`.
MTP_MALFORMED_REPO = "example/RecommendMtpMalformedModel"
MTP_MALFORMED_CARD_ID = "recommend-mtp-malformed@test"


def mtp_manifest_for_recommend() -> dict:
    return {
        "schema": "fast-mlx-quality-card-v1",
        "generatedAt": "2026-01-01T00:00:00Z",
        "cards": [
            {
                "id": MTP_NOT_EXACT_CARD_ID,
                "model": {"repo": MTP_NOT_EXACT_REPO, "hfPin": "abcdef12"},
                "verdict": "PASS",
                "legible": {"tier": "Reference", "headline": "recommend mtp headline"},
                "config": {
                    "flagTransfer": {
                        "--mtp": {
                            "greedy": "not_exact",
                            "divergentPrompts": 4,
                            "prompts": 40,
                            "maxTokens": 64,
                            "evidence": "docs/x.json",
                        }
                    }
                },
                "provenance": {"engineBuild": {"commit": MTP_CARD_COMMIT}},
            },
            {
                "id": MTP_MALFORMED_CARD_ID,
                "model": {"repo": MTP_MALFORMED_REPO, "hfPin": "abcdef34"},
                "verdict": "PASS",
                "legible": {"tier": "Reference", "headline": "recommend mtp malformed headline"},
                "config": {
                    "flagTransfer": {
                        "--mtp": {
                            "greedy": "EXACT",  # wrong case; not a recognized value
                            "divergentPrompts": 0,
                            "prompts": 40,
                            "maxTokens": 64,
                            "evidence": "docs/x.json",
                        }
                    }
                },
                "provenance": {"engineBuild": {"commit": MTP_CARD_COMMIT}},
            },
        ],
    }


class MtpRecommendTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest_path = self.root / "mtp-quality-guides.json"
        self.manifest_path.write_text(json.dumps(mtp_manifest_for_recommend()), encoding="utf-8")
        self.green_fit_bin = write_script(self.root / "fit-green.py", GREEN_FIT_CHECK_BODY)

    def make_model_dir(self, name: str, repo: str) -> Path:
        model_dir = self.root / name
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        write_pull_receipt(model_dir, repo_id=repo, revision="e" * 40)
        return model_dir

    def _write_profile(self, name: str, mtp: bool, commit: str = None) -> Path:
        argv = ["{engine_bin}"]
        if mtp:
            argv.append("--mtp")
        document = {
            "schema": "fastmlx-engine-profile-v1",
            "name": name,
            "argv": argv,
        }
        if commit is not None:
            document["engineBuild"] = {"commit": commit}
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def run_main(self, argv: list):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                FASTMLX_RECOMMEND.main(argv)
        return ctx.exception.code, stdout.getvalue(), stderr.getvalue()

    def _run(self, profile_path: Path, model_dir: Path) -> dict:
        argv = [
            "recommend",
            "--quality-cards",
            str(self.manifest_path),
            "--model-path",
            str(model_dir),
            "--fit-check-bin",
            str(self.green_fit_bin),
            "--engine-profile",
            str(profile_path),
            "--json",
        ]
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        return json.loads(stdout)["rows"][0]

    def test_row_mtp_off_when_profile_has_no_mtp_flag(self) -> None:
        model_dir = self.make_model_dir("mtp-off-model", repo=MTP_NOT_EXACT_REPO)
        profile_path = self._write_profile("no-mtp", mtp=False, commit=MTP_CARD_COMMIT)
        row = self._run(profile_path, model_dir)
        self.assertEqual(row["mtp"]["status"], "off")
        self.assertIsNone(row["mtp"]["message"])

    def test_row_mtp_not_exact_at_matching_build_carries_message(self) -> None:
        model_dir = self.make_model_dir("mtp-match-model", repo=MTP_NOT_EXACT_REPO)
        profile_path = self._write_profile("mtp-match", mtp=True, commit=MTP_CARD_COMMIT)
        row = self._run(profile_path, model_dir)
        self.assertEqual(row["mtp"]["status"], "not_exact")
        self.assertEqual(row["mtp"]["divergentPrompts"], 4)
        self.assertEqual(row["mtp"]["prompts"], 40)
        self.assertIsNotNone(row["mtp"]["message"])
        self.assertIn(MTP_NOT_EXACT_CARD_ID, row["mtp"]["message"])
        self.assertIn("differed on 4/40 prompts", row["mtp"]["message"])

    def test_row_mtp_unmeasured_on_build_mismatch(self) -> None:
        model_dir = self.make_model_dir("mtp-mismatch-model", repo=MTP_NOT_EXACT_REPO)
        profile_path = self._write_profile("mtp-mismatch", mtp=True, commit=MTP_OTHER_COMMIT)
        row = self._run(profile_path, model_dir)
        self.assertEqual(row["engineBuild"]["status"], "mismatch")
        self.assertEqual(row["mtp"]["status"], "unmeasured")
        self.assertIsNone(row["mtp"]["divergentPrompts"])

    def test_text_rendering_includes_mtp_message(self) -> None:
        model_dir = self.make_model_dir("mtp-text-model", repo=MTP_NOT_EXACT_REPO)
        profile_path = self._write_profile("mtp-match-text", mtp=True, commit=MTP_CARD_COMMIT)
        argv = [
            "recommend",
            "--quality-cards",
            str(self.manifest_path),
            "--model-path",
            str(model_dir),
            "--fit-check-bin",
            str(self.green_fit_bin),
            "--engine-profile",
            str(profile_path),
        ]
        _, stdout_text, _ = self.run_main(argv)
        self.assertIn("differed on 4/40 prompts", stdout_text)

    # Fail-open on unvalidated flagTransfer data: `load_quality_cards`
    # never validates a card, so a hand-edited/stale card with an unknown
    # `greedy` value must never crash `build_row` (whose docstring promises
    # "Never raises") -- it must classify as "unmeasured" instead.
    def test_unknown_greedy_value_fails_open_without_crashing(self) -> None:
        model_dir = self.make_model_dir("mtp-malformed-model", repo=MTP_MALFORMED_REPO)
        profile_path = self._write_profile("mtp-malformed", mtp=True, commit=MTP_CARD_COMMIT)
        row = self._run(profile_path, model_dir)
        self.assertEqual(row["mtp"]["status"], "unmeasured")
        self.assertIsNone(row["mtp"]["divergentPrompts"])

    # (6d) `--mtp` never gates a row's status/verdict -- it is a notice
    # only, exactly like `engineBuild`.
    def test_status_and_verdict_identical_with_and_without_mtp(self) -> None:
        model_dir = self.make_model_dir("mtp-identical-model", repo=MTP_NOT_EXACT_REPO)
        profile_without_mtp = self._write_profile(
            "identical-no-mtp", mtp=False, commit=MTP_CARD_COMMIT
        )
        profile_with_mtp = self._write_profile(
            "identical-with-mtp", mtp=True, commit=MTP_CARD_COMMIT
        )
        row_without = self._run(profile_without_mtp, model_dir)
        row_with = self._run(profile_with_mtp, model_dir)
        self.assertEqual(row_without["status"], row_with["status"])
        self.assertEqual(
            (row_without.get("card") or {}).get("verdict"),
            (row_with.get("card") or {}).get("verdict"),
        )
        # The rows genuinely differ only in the `mtp` notice -- otherwise
        # this test would pass vacuously even if --mtp secretly changed
        # something else about admission.
        self.assertNotEqual(row_without["mtp"]["status"], row_with["mtp"]["status"])


if __name__ == "__main__":
    unittest.main()
