"""Tests for `config.flagTransfer["--mtp"]` validation
(`docs/quality-card-schema-v1.md` "Flag transfer") in BOTH site validators
`build_public_site.validate_quality_card_document` (fail-fast) and
`validate_public_site.validate_quality_guide_manifest` (accumulate) --
following the exact pattern the `provenance.engineBuild` tests in
`test_public_site.py` set for both validators agreeing. A NEW file (not an
addition to the already-large `test_public_site.py`).
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import build_public_site  # noqa: E402
import validate_public_site  # noqa: E402


def quality_guide_manifest() -> dict:
    return json.loads(
        (REPOSITORY_ROOT / "scripts/tests/fixtures/quality-guides.sample.json").read_text(
            encoding="utf-8"
        )
    )


VALID_MTP_TRANSFER = {
    "greedy": "not_exact",
    "divergentPrompts": 3,
    "prompts": 40,
    "maxTokens": 64,
    "evidence": "mtp-transfer-fixture-2026-09-19",
}


class FlagTransferValidationTests(unittest.TestCase):
    def test_valid_flag_transfer_is_accepted_by_both_validators(self) -> None:
        manifest = quality_guide_manifest()
        manifest["cards"][0]["config"]["flagTransfer"] = {"--mtp": VALID_MTP_TRANSFER}
        validated = build_public_site.validate_quality_card_document(manifest, "test manifest")
        self.assertEqual(
            validated["cards"][0]["config"]["flagTransfer"], {"--mtp": VALID_MTP_TRANSFER}
        )
        failures = validate_public_site.validate_quality_guide_manifest(manifest)
        self.assertEqual(failures, [])

    def test_flag_transfer_without_engine_build_is_rejected_by_both_validators(self) -> None:
        manifest = quality_guide_manifest()
        card = manifest["cards"][1]
        self.assertNotIn("engineBuild", card["provenance"])
        card["config"]["flagTransfer"] = {"--mtp": VALID_MTP_TRANSFER}
        # Each validator names the card by its manifest position, following
        # the existing convention every other `fail`/failure message in
        # these validators uses (e.g. the provenance.engineBuild.commit
        # format check above) -- never the card's `id` string. Each
        # validator uses its own label prefix ("test manifest card entry 1"
        # vs "quality card entry 1").
        with self.assertRaises(SystemExit) as build_ctx:
            build_public_site.validate_quality_card_document(manifest, "test manifest")
        self.assertIn("test manifest card entry 1", str(build_ctx.exception))
        self.assertIn("flagTransfer", str(build_ctx.exception))
        self.assertIn("engineBuild", str(build_ctx.exception))
        failures = validate_public_site.validate_quality_guide_manifest(manifest)
        self.assertTrue(failures, "expected at least one failure")
        self.assertTrue(
            any(
                "quality card entry 1" in failure
                and "flagTransfer" in failure
                and "engineBuild" in failure
                for failure in failures
            ),
            f"expected a failure naming card entry 1, got {failures!r}",
        )

    def test_flag_transfer_with_engine_build_is_accepted_by_both_validators(self) -> None:
        manifest = quality_guide_manifest()
        card = manifest["cards"][1]
        card["provenance"]["engineBuild"] = {
            "commit": "fa76a4b50b3f54af7e9cd927279f5ba2870f02c6"
        }
        card["config"]["flagTransfer"] = {"--mtp": VALID_MTP_TRANSFER}
        validated = build_public_site.validate_quality_card_document(manifest, "test manifest")
        self.assertEqual(
            validated["cards"][1]["config"]["flagTransfer"], {"--mtp": VALID_MTP_TRANSFER}
        )
        failures = validate_public_site.validate_quality_guide_manifest(manifest)
        self.assertEqual(failures, [])

    def test_absent_flag_transfer_is_accepted_by_both_validators(self) -> None:
        manifest = quality_guide_manifest()
        self.assertNotIn("flagTransfer", manifest["cards"][0]["config"])
        build_public_site.validate_quality_card_document(manifest, "test manifest")
        failures = validate_public_site.validate_quality_guide_manifest(manifest)
        self.assertEqual(failures, [])

    def test_exact_with_zero_divergent_is_accepted(self) -> None:
        manifest = quality_guide_manifest()
        manifest["cards"][0]["config"]["flagTransfer"] = {
            "--mtp": {
                "greedy": "exact",
                "divergentPrompts": 0,
                "prompts": 40,
                "maxTokens": 64,
                "evidence": "docs/x.json",
            }
        }
        build_public_site.validate_quality_card_document(manifest, "test manifest")
        failures = validate_public_site.validate_quality_guide_manifest(manifest)
        self.assertEqual(failures, [])

    def _assert_rejected_by_both_validators(self, flag_transfer: object) -> None:
        manifest = quality_guide_manifest()
        manifest["cards"][0]["config"]["flagTransfer"] = flag_transfer
        with self.assertRaises(SystemExit):
            build_public_site.validate_quality_card_document(manifest, "test manifest")
        failures = validate_public_site.validate_quality_guide_manifest(manifest)
        self.assertTrue(failures, f"expected at least one failure for {flag_transfer!r}")

    def test_unknown_key_inside_flag_transfer_is_rejected(self) -> None:
        self._assert_rejected_by_both_validators(
            {"--mtp": VALID_MTP_TRANSFER, "--other-flag": {}}
        )

    def test_unknown_key_inside_mtp_object_is_rejected(self) -> None:
        bad = dict(VALID_MTP_TRANSFER, unknownKey="x")
        self._assert_rejected_by_both_validators({"--mtp": bad})

    def test_missing_key_inside_mtp_object_is_rejected(self) -> None:
        bad = dict(VALID_MTP_TRANSFER)
        del bad["evidence"]
        self._assert_rejected_by_both_validators({"--mtp": bad})

    def test_unknown_greedy_value_is_rejected(self) -> None:
        bad = dict(VALID_MTP_TRANSFER, greedy="sometimes")
        self._assert_rejected_by_both_validators({"--mtp": bad})

    def test_exact_with_divergent_greater_than_zero_is_rejected(self) -> None:
        bad = dict(VALID_MTP_TRANSFER, greedy="exact", divergentPrompts=1)
        self._assert_rejected_by_both_validators({"--mtp": bad})

    def test_not_exact_with_zero_divergent_is_rejected(self) -> None:
        bad = dict(VALID_MTP_TRANSFER, greedy="not_exact", divergentPrompts=0)
        self._assert_rejected_by_both_validators({"--mtp": bad})

    def test_divergent_greater_than_prompts_is_rejected(self) -> None:
        bad = dict(VALID_MTP_TRANSFER, divergentPrompts=41, prompts=40)
        self._assert_rejected_by_both_validators({"--mtp": bad})

    def test_prompts_below_one_is_rejected(self) -> None:
        bad = dict(VALID_MTP_TRANSFER, divergentPrompts=0, prompts=0, greedy="exact")
        self._assert_rejected_by_both_validators({"--mtp": bad})

    def test_max_tokens_below_one_is_rejected(self) -> None:
        bad = dict(VALID_MTP_TRANSFER, maxTokens=0)
        self._assert_rejected_by_both_validators({"--mtp": bad})

    def test_absolute_evidence_path_is_rejected(self) -> None:
        bad = dict(VALID_MTP_TRANSFER, evidence="/opt/evidence/x.json")
        self._assert_rejected_by_both_validators({"--mtp": bad})

    def test_dotdot_evidence_path_is_rejected(self) -> None:
        bad = dict(VALID_MTP_TRANSFER, evidence="../outside/x.json")
        self._assert_rejected_by_both_validators({"--mtp": bad})

    def test_empty_evidence_string_is_rejected(self) -> None:
        bad = dict(VALID_MTP_TRANSFER, evidence="")
        self._assert_rejected_by_both_validators({"--mtp": bad})

    def test_non_object_flag_transfer_is_rejected(self) -> None:
        self._assert_rejected_by_both_validators("not-an-object")

    def test_flag_transfer_missing_mtp_key_is_rejected(self) -> None:
        self._assert_rejected_by_both_validators({})

    def test_private_marker_in_evidence_is_rejected_by_whole_document_scan(self) -> None:
        # Concatenated, never a literal private-marker token in this source
        # file (see validate_public_repository.py's own convention).
        marker = "192" + ".168.1.1"
        bad = dict(VALID_MTP_TRANSFER, evidence=f"docs/{marker}/x.json")
        manifest = quality_guide_manifest()
        manifest["cards"][0]["config"]["flagTransfer"] = {"--mtp": bad}
        with self.assertRaises(SystemExit):
            build_public_site.validate_quality_card_document(manifest, "test manifest")
        failures = validate_public_site.validate_quality_guide_manifest(manifest)
        self.assertTrue(failures, "expected validate_public_site to reject it too")

    def test_private_docs_superpowers_marker_in_evidence_is_rejected_by_both_validators(
        self,
    ) -> None:
        # This repository's own PRIVATE verdict/notes tree marker (see
        # validate_public_repository.py PRIVATE_MARKERS) -- never a
        # legitimate `--mtp-transfer` evidence value. Concatenated for the
        # same reason as the marker above (never a literal token here).
        marker = "docs/" + "superpowers/"
        bad = dict(VALID_MTP_TRANSFER, evidence=f"{marker}verdicts/mtp-transfer.json")
        manifest = quality_guide_manifest()
        manifest["cards"][0]["config"]["flagTransfer"] = {"--mtp": bad}
        with self.assertRaises(SystemExit):
            build_public_site.validate_quality_card_document(manifest, "test manifest")
        failures = validate_public_site.validate_quality_guide_manifest(manifest)
        self.assertTrue(failures, "expected validate_public_site to reject it too")


if __name__ == "__main__":
    unittest.main()
