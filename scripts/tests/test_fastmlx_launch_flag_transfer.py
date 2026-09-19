"""Tests for `config.flagTransfer["--mtp"]` / the `--mtp` launcher notice
(`docs/quality-card-schema-v1.md` "Flag transfer") -- following the exact
pattern `EngineBuildTestCase` in `test_fastmlx_launch.py` set for
`provenance.engineBuild`. A NEW file (not an addition to the already-large
`test_fastmlx_launch.py`) that imports the shared fixtures/helpers that file
already exports.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.tests.test_fastmlx_launch import (
    FAKE_ENGINE_BODY,
    FASTMLX_LAUNCH,
    GREEN_FIT_CHECK_BODY,
    write_script,
)

# ---------------------------------------------------------------------
# Fixture manifest: one card per MTP status this module classifies, all
# measured on the SAME engine build (MTP_CARD_COMMIT) -- so a launch whose
# own profile declares that build gets `match`, and a launch that declares
# a different one gets `mismatch` (-> unmeasured, per the schema rule that
# the transfer is valid only at the card's own build).
# ---------------------------------------------------------------------
MTP_CARD_COMMIT = "e5" * 20
MTP_OTHER_COMMIT = "f6" * 20

MTP_EXACT_CARD_ID = "mtp-exact@test"
MTP_EXACT_REPO = "example/MtpExactModel"

MTP_NOT_EXACT_CARD_ID = "mtp-not-exact@test"
MTP_NOT_EXACT_REPO = "example/MtpNotExactModel"
MTP_NOT_EXACT_DIVERGENT = 3
MTP_NOT_EXACT_PROMPTS = 40

MTP_NONDETERMINISTIC_CARD_ID = "mtp-nondeterministic@test"
MTP_NONDETERMINISTIC_REPO = "example/MtpNondeterministicModel"
MTP_NONDETERMINISTIC_DIVERGENT = 5
MTP_NONDETERMINISTIC_PROMPTS = 40

MTP_UNMEASURED_CARD_ID = "mtp-unmeasured@test"
MTP_UNMEASURED_REPO = "example/MtpUnmeasuredModel"

MTP_NO_GO_CARD_ID = "mtp-no-go@test"
MTP_NO_GO_REPO = "example/MtpNoGoModel"

# (B1) A NO_GO card WHOSE flagTransfer IS present (nondeterministic, at the
# matching build) -- unlike MTP_NO_GO_CARD_ID above (no flagTransfer at
# all), this exercises the refusal-message/admitted-notice path together
# with a still-gating NO_GO verdict.
MTP_NO_GO_NONDETERMINISTIC_CARD_ID = "mtp-no-go-nondeterministic@test"
MTP_NO_GO_NONDETERMINISTIC_REPO = "example/MtpNoGoNondeterministicModel"
MTP_NO_GO_NONDETERMINISTIC_DIVERGENT = 6
MTP_NO_GO_NONDETERMINISTIC_PROMPTS = 40

# A card whose flagTransfer is malformed (unvalidated -- `load_quality_cards`
# never validates a card against the schema): `greedy` outside the three
# known values. Exercises the fail-open rule in `mtp_transfer_status`.
MTP_MALFORMED_GREEDY_CARD_ID = "mtp-malformed-greedy@test"
MTP_MALFORMED_GREEDY_REPO = "example/MtpMalformedGreedyModel"

# A card whose flagTransfer["--mtp"] is present but empty (missing every
# key) -- another unvalidated shape a hand-edited manifest could carry.
MTP_MALFORMED_EMPTY_CARD_ID = "mtp-malformed-empty@test"
MTP_MALFORMED_EMPTY_REPO = "example/MtpMalformedEmptyModel"


def _pass_card(card_id: str, repo: str, flag_transfer: dict | None) -> dict:
    card: dict = {
        "id": card_id,
        "model": {"repo": repo, "hfPin": "deadbeef"},
        "verdict": "PASS",
        "admission": {"default": True, "optIn": False, "reason": "measured pass"},
        "legible": {"tier": "Reference", "headline": f"{card_id} headline"},
        "provenance": {"engineBuild": {"commit": MTP_CARD_COMMIT}},
    }
    if flag_transfer is not None:
        card["config"] = {"flagTransfer": {"--mtp": flag_transfer}}
    return card


def flag_transfer_card_manifest() -> dict:
    return {
        "schema": "fast-mlx-quality-card-v1",
        "generatedAt": "2026-01-01T00:00:00Z",
        "cards": [
            _pass_card(
                MTP_EXACT_CARD_ID,
                MTP_EXACT_REPO,
                {
                    "greedy": "exact",
                    "divergentPrompts": 0,
                    "prompts": 40,
                    "maxTokens": 64,
                    "evidence": "docs/x.json",
                },
            ),
            _pass_card(
                MTP_NOT_EXACT_CARD_ID,
                MTP_NOT_EXACT_REPO,
                {
                    "greedy": "not_exact",
                    "divergentPrompts": MTP_NOT_EXACT_DIVERGENT,
                    "prompts": MTP_NOT_EXACT_PROMPTS,
                    "maxTokens": 64,
                    "evidence": "docs/x.json",
                },
            ),
            _pass_card(
                MTP_NONDETERMINISTIC_CARD_ID,
                MTP_NONDETERMINISTIC_REPO,
                {
                    "greedy": "nondeterministic",
                    "divergentPrompts": MTP_NONDETERMINISTIC_DIVERGENT,
                    "prompts": MTP_NONDETERMINISTIC_PROMPTS,
                    "maxTokens": 64,
                    "evidence": "docs/x.json",
                },
            ),
            _pass_card(MTP_UNMEASURED_CARD_ID, MTP_UNMEASURED_REPO, None),
            {
                "id": MTP_NO_GO_CARD_ID,
                "model": {"repo": MTP_NO_GO_REPO, "hfPin": "cafebabe"},
                "verdict": "NO_GO",
                "admission": {
                    "default": False,
                    "optIn": True,
                    "reason": "quality-degraded vs reference",
                },
                "legible": {"tier": "Noticeable", "headline": "no-go headline"},
                "provenance": {"engineBuild": {"commit": MTP_CARD_COMMIT}},
                # No config.flagTransfer at all.
            },
            {
                "id": MTP_NO_GO_NONDETERMINISTIC_CARD_ID,
                "model": {"repo": MTP_NO_GO_NONDETERMINISTIC_REPO, "hfPin": "deadfeed"},
                "verdict": "NO_GO",
                "admission": {
                    "default": False,
                    "optIn": True,
                    "reason": "quality-degraded vs reference",
                },
                "legible": {"tier": "Significant", "headline": "no-go nondeterministic headline"},
                "provenance": {"engineBuild": {"commit": MTP_CARD_COMMIT}},
                "config": {
                    "flagTransfer": {
                        "--mtp": {
                            "greedy": "nondeterministic",
                            "divergentPrompts": MTP_NO_GO_NONDETERMINISTIC_DIVERGENT,
                            "prompts": MTP_NO_GO_NONDETERMINISTIC_PROMPTS,
                            "maxTokens": 64,
                            "evidence": "docs/x.json",
                        }
                    }
                },
            },
            # Unvalidated shapes (`load_quality_cards` never validates a
            # card against the schema) -- `mtp_transfer_status` must fail
            # OPEN to "unmeasured" for these, never crash.
            _pass_card(
                MTP_MALFORMED_GREEDY_CARD_ID,
                MTP_MALFORMED_GREEDY_REPO,
                {
                    "greedy": "EXACT",  # wrong case; not a recognized value
                    "divergentPrompts": 0,
                    "prompts": 40,
                    "maxTokens": 64,
                    "evidence": "docs/x.json",
                },
            ),
            _pass_card(MTP_MALFORMED_EMPTY_CARD_ID, MTP_MALFORMED_EMPTY_REPO, {}),
        ],
    }


def write_flag_transfer_card_manifest(root: Path) -> Path:
    path = root / "flag-transfer-quality-cards.json"
    path.write_text(json.dumps(flag_transfer_card_manifest()), encoding="utf-8")
    return path


def write_mtp_profile(root: Path, name: str, commit: str | None = None) -> Path:
    """An engine profile whose OWN argv carries the literal `--mtp` token --
    the "final argv after profile expansion" the schema's launcher-status
    rule keys on -- optionally also declaring `engineBuild.commit`.
    """
    document = {
        "schema": "fastmlx-engine-profile-v1",
        "name": name,
        "argv": list(FASTMLX_LAUNCH.BUILT_IN_ENGINE_PROFILE["argv"]) + ["--mtp"],
    }
    if commit is not None:
        document["engineBuild"] = {"commit": commit}
    path = root / f"{name}.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def write_no_mtp_profile(root: Path, name: str, commit: str | None = None) -> Path:
    document = {
        "schema": "fastmlx-engine-profile-v1",
        "name": name,
        "argv": list(FASTMLX_LAUNCH.BUILT_IN_ENGINE_PROFILE["argv"]),
    }
    if commit is not None:
        document["engineBuild"] = {"commit": commit}
    path = root / f"{name}.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class FlagTransferTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        (self.model_dir / "config.json").write_text("{}", encoding="utf-8")

        self.manifest_path = write_flag_transfer_card_manifest(self.root)
        self.green_fit_bin = write_script(self.root / "fit-green.py", GREEN_FIT_CHECK_BODY)
        self.fake_engine_bin = write_script(self.root / "fake-engine.py", FAKE_ENGINE_BODY)

    def base_args(self, **overrides) -> list:
        args = {
            "--model-path": str(self.model_dir),
            "--quality-cards": str(self.manifest_path),
            "--fit-check-bin": str(self.green_fit_bin),
            "--engine-bin": str(self.fake_engine_bin),
        }
        args.update(overrides)
        argv = ["serve"]
        for key, value in args.items():
            if value is None:
                continue
            argv += [key, str(value)]
        return argv

    def run_main(self, argv: list):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                FASTMLX_LAUNCH.main(argv)
        return ctx.exception.code, stdout.getvalue(), stderr.getvalue()

    @staticmethod
    def last_json_line(stdout: str) -> dict:
        lines = [line for line in stdout.splitlines() if line.strip()]
        return json.loads(lines[-1])

    # (i) No --mtp at all: status "off", no notice, regardless of the card.
    def test_off_when_no_mtp_flag(self) -> None:
        profile_path = write_no_mtp_profile(self.root, "no-mtp-profile", commit=MTP_CARD_COMMIT)
        argv = self.base_args(
            **{
                "--model-repo": MTP_NOT_EXACT_REPO,
                "--context": "2048",
                "--engine-profile": str(profile_path),
            }
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        self.assertNotIn("--mtp transfer", stderr)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["mtp"]["status"], "off")
        self.assertIsNone(plan["mtp"]["divergentPrompts"])
        self.assertIsNone(plan["mtp"]["prompts"])

    # (ii) --mtp + exact + build match: status "exact", no notice.
    def test_exact_at_matching_build_has_no_notice(self) -> None:
        profile_path = write_mtp_profile(self.root, "mtp-match-profile", commit=MTP_CARD_COMMIT)
        argv = self.base_args(
            **{
                "--model-repo": MTP_EXACT_REPO,
                "--context": "2048",
                "--engine-profile": str(profile_path),
            }
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        self.assertNotIn("--mtp transfer", stderr)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["mtp"]["status"], "exact")
        self.assertEqual(plan["mtp"]["divergentPrompts"], 0)
        self.assertEqual(plan["mtp"]["prompts"], 40)

    # (ii-real) The REAL (non-dry-run) admitted stderr line carries the
    # actual computed `mtp=<status>` token -- mocking os.execv the same way
    # `test_capacity_execs_the_resolved_binary_with_argv_passthrough` /
    # `test_engine_execs_the_built_in_engine_binary_name` in test_fastmlx.py
    # mock `os.execv` for the engine-build-carrying admitted line, so this
    # runs in-process instead of the separate `test_admitted_run_actually_execs_the_engine`
    # out-of-process subprocess pattern in test_fastmlx_launch.py (which
    # only asserts the literal "fastmlx_launch=admitted" token, never the
    # `mtp=` field).
    def test_real_admitted_line_carries_the_computed_mtp_status(self) -> None:
        profile_path = write_mtp_profile(self.root, "mtp-match-profile", commit=MTP_CARD_COMMIT)
        argv = self.base_args(
            **{
                "--model-repo": MTP_EXACT_REPO,
                "--context": "2048",
                "--engine-profile": str(profile_path),
            }
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(FASTMLX_LAUNCH.os, "execv") as fake_execv:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as ctx:
                    FASTMLX_LAUNCH.main(argv)
        self.assertEqual(ctx.exception.code, 0)
        fake_execv.assert_called_once()
        self.assertIn("fastmlx_launch=admitted", stderr.getvalue())
        self.assertIn("mtp=exact", stderr.getvalue())

    # (iii) --mtp + not_exact + build match: status "not_exact", exact
    # notice text, k/n in both stderr and plan.
    def test_not_exact_prints_exact_notice_text(self) -> None:
        profile_path = write_mtp_profile(self.root, "mtp-match-profile", commit=MTP_CARD_COMMIT)
        argv = self.base_args(
            **{
                "--model-repo": MTP_NOT_EXACT_REPO,
                "--context": "2048",
                "--engine-profile": str(profile_path),
            }
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        expected = (
            f"card {MTP_NOT_EXACT_CARD_ID} was measured without --mtp; with --mtp at "
            f"build {MTP_CARD_COMMIT[:12]}, greedy output differed on "
            f"{MTP_NOT_EXACT_DIVERGENT}/{MTP_NOT_EXACT_PROMPTS} prompts"
        )
        self.assertIn(expected, stderr)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["mtp"]["status"], "not_exact")
        self.assertEqual(plan["mtp"]["divergentPrompts"], MTP_NOT_EXACT_DIVERGENT)
        self.assertEqual(plan["mtp"]["prompts"], MTP_NOT_EXACT_PROMPTS)

    # (iv) --mtp + nondeterministic + build match: exact notice text.
    def test_nondeterministic_prints_exact_notice_text(self) -> None:
        profile_path = write_mtp_profile(self.root, "mtp-match-profile", commit=MTP_CARD_COMMIT)
        argv = self.base_args(
            **{
                "--model-repo": MTP_NONDETERMINISTIC_REPO,
                "--context": "2048",
                "--engine-profile": str(profile_path),
            }
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        expected = (
            f"card {MTP_NONDETERMINISTIC_CARD_ID} was measured without --mtp; with "
            f"--mtp at build {MTP_CARD_COMMIT[:12]}, greedy output differed from "
            f"the card's path on {MTP_NONDETERMINISTIC_DIVERGENT}/{MTP_NONDETERMINISTIC_PROMPTS} "
            "prompts and was not reproducible across processes"
        )
        self.assertIn(expected, stderr)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["mtp"]["status"], "nondeterministic")

    # (v) --mtp + card with no flagTransfer at all: "unmeasured", exact
    # no-flagTransfer notice text.
    def test_unmeasured_card_prints_exact_notice_text(self) -> None:
        profile_path = write_mtp_profile(self.root, "mtp-match-profile", commit=MTP_CARD_COMMIT)
        argv = self.base_args(
            **{
                "--model-repo": MTP_UNMEASURED_REPO,
                "--context": "2048",
                "--engine-profile": str(profile_path),
            }
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        expected = (
            f"card {MTP_UNMEASURED_CARD_ID} was measured without --mtp; the --mtp "
            "transfer is unmeasured"
        )
        self.assertIn(expected, stderr)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["mtp"]["status"], "unmeasured")
        self.assertIsNone(plan["mtp"]["divergentPrompts"])
        self.assertIsNone(plan["mtp"]["prompts"])

    # (vi) --mtp + a card WITH flagTransfer but a build MISMATCH: the
    # transfer is only valid at the card's own build, so this downgrades to
    # "unmeasured" even though flagTransfer is present.
    def test_build_mismatch_downgrades_to_unmeasured(self) -> None:
        profile_path = write_mtp_profile(self.root, "mtp-mismatch-profile", commit=MTP_OTHER_COMMIT)
        argv = self.base_args(
            **{
                "--model-repo": MTP_NOT_EXACT_REPO,
                "--context": "2048",
                "--engine-profile": str(profile_path),
            }
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["engineBuild"]["status"], "mismatch")
        self.assertEqual(plan["mtp"]["status"], "unmeasured")
        self.assertIsNone(plan["mtp"]["divergentPrompts"])
        self.assertIn(f"card {MTP_NOT_EXACT_CARD_ID} was measured without --mtp; the --mtp transfer is unmeasured", stderr)

    # (vii) --mtp + no repo at all (no card resolves): the no-card notice
    # variant.
    def test_no_card_prints_no_card_notice_variant(self) -> None:
        profile_path = write_mtp_profile(self.root, "mtp-match-profile", commit=MTP_CARD_COMMIT)
        argv = self.base_args(
            **{
                "--model-repo": "example/NoCardAtAllModel",
                "--context": "2048",
                "--engine-profile": str(profile_path),
            }
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        self.assertIn(
            "no quality card was resolved for this launch; the --mtp transfer is unmeasured",
            stderr,
        )
        plan = self.last_json_line(stdout)
        self.assertIsNone(plan["card"])
        self.assertEqual(plan["mtp"]["status"], "unmeasured")

    # (viii) MTP status/notice NEVER changes admission: a NO_GO card with no
    # flagTransfer, launched WITH --mtp and with NO --accept-quality, still
    # refuses (exit 2) -- never "admit_unmeasured".
    def test_no_go_card_still_refused_under_mtp_with_no_flag_transfer(self) -> None:
        profile_path = write_mtp_profile(self.root, "mtp-match-profile", commit=MTP_CARD_COMMIT)
        argv = self.base_args(
            **{
                "--model-repo": MTP_NO_GO_REPO,
                "--context": "2048",
                "--engine-profile": str(profile_path),
            }
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2, stderr)
        self.assertIn("re-run with --accept-quality", stderr)

    # (ix) The passthrough path (`-- --mtp`) is equivalent to a profile
    # whose own argv carries `--mtp` -- both are "the final engine argv".
    def test_mtp_via_passthrough_is_equivalent_to_profile_argv(self) -> None:
        profile_path = write_no_mtp_profile(self.root, "no-mtp-profile", commit=MTP_CARD_COMMIT)
        argv = self.base_args(
            **{
                "--model-repo": MTP_EXACT_REPO,
                "--context": "2048",
                "--engine-profile": str(profile_path),
            }
        ) + ["--dry-run", "--", "--mtp"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["mtp"]["status"], "exact")
        self.assertIn("--mtp", plan["argv"])

    # (x) Fail-open on unvalidated flagTransfer data: `load_quality_cards`
    # never validates a card, so a hand-edited/stale card with an unknown
    # `greedy` value must never crash the launcher -- it must classify as
    # "unmeasured" instead, exactly like a card with no flagTransfer at
    # all.
    def test_unknown_greedy_value_fails_open_to_unmeasured_without_crashing(self) -> None:
        profile_path = write_mtp_profile(self.root, "mtp-match-profile", commit=MTP_CARD_COMMIT)
        argv = self.base_args(
            **{
                "--model-repo": MTP_MALFORMED_GREEDY_REPO,
                "--context": "2048",
                "--engine-profile": str(profile_path),
            }
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["mtp"]["status"], "unmeasured")
        self.assertIsNone(plan["mtp"]["divergentPrompts"])
        self.assertIsNone(plan["mtp"]["prompts"])

    # (xi) Same fail-open rule for a flagTransfer object missing every key
    # (e.g. `{}` -- another shape a hand-edited manifest could carry).
    def test_empty_flag_transfer_object_fails_open_to_unmeasured_without_crashing(self) -> None:
        profile_path = write_mtp_profile(self.root, "mtp-match-profile", commit=MTP_CARD_COMMIT)
        argv = self.base_args(
            **{
                "--model-repo": MTP_MALFORMED_EMPTY_REPO,
                "--context": "2048",
                "--engine-profile": str(profile_path),
            }
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["mtp"]["status"], "unmeasured")

    # (xii, "B1") A NO_GO card WHOSE flagTransfer IS present (nondeterministic,
    # matching build): without --accept-quality, still refused (exit 2) and
    # the mtp notice is folded into the refusal message; with
    # --accept-quality, admitted, and the notice appears on stderr.
    def test_b1_no_go_with_flag_transfer_refuses_with_notice_in_refusal_message(self) -> None:
        profile_path = write_mtp_profile(self.root, "mtp-match-profile", commit=MTP_CARD_COMMIT)
        argv = self.base_args(
            **{
                "--model-repo": MTP_NO_GO_NONDETERMINISTIC_REPO,
                "--context": "2048",
                "--engine-profile": str(profile_path),
            }
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2, stderr)
        self.assertIn("re-run with --accept-quality", stderr)
        expected_notice = (
            f"card {MTP_NO_GO_NONDETERMINISTIC_CARD_ID} was measured without --mtp; "
            f"with --mtp at build {MTP_CARD_COMMIT[:12]}, greedy output differed "
            f"from the card's path on {MTP_NO_GO_NONDETERMINISTIC_DIVERGENT}/"
            f"{MTP_NO_GO_NONDETERMINISTIC_PROMPTS} prompts and was not reproducible "
            "across processes"
        )
        self.assertIn(expected_notice, stderr)

    def test_b1_no_go_with_flag_transfer_admitted_with_notice_on_stderr(self) -> None:
        profile_path = write_mtp_profile(self.root, "mtp-match-profile", commit=MTP_CARD_COMMIT)
        argv = self.base_args(
            **{
                "--model-repo": MTP_NO_GO_NONDETERMINISTIC_REPO,
                "--context": "2048",
                "--engine-profile": str(profile_path),
                "--accept-quality": MTP_NO_GO_NONDETERMINISTIC_CARD_ID,
            }
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["admission"], "admit_with_quality_flag")
        expected_notice = (
            f"card {MTP_NO_GO_NONDETERMINISTIC_CARD_ID} was measured without --mtp; "
            f"with --mtp at build {MTP_CARD_COMMIT[:12]}, greedy output differed "
            f"from the card's path on {MTP_NO_GO_NONDETERMINISTIC_DIVERGENT}/"
            f"{MTP_NO_GO_NONDETERMINISTIC_PROMPTS} prompts and was not reproducible "
            "across processes"
        )
        self.assertIn(expected_notice, stderr)

    # (xiii) `--mtp` arriving ONLY via `residencyArgs['expert-stream']`
    # (never in the profile's own `argv`) must still be detected -- this is
    # a direct unit test of the pure helper, since routing a full expert-
    # stream launch through card resolution needs its own residency-aware
    # fixture (see `ResidencyTestCase` in test_fastmlx_launch.py) that is
    # out of scope for this notice-focused module.
    def test_mtp_only_in_residency_args_is_detected(self) -> None:
        profile = {"argv": ["{engine_bin}"], "residencyArgs": {"expert-stream": ["--mtp"]}}
        self.assertTrue(FASTMLX_LAUNCH.mtp_launch_requested(profile, "expert-stream"))
        self.assertFalse(FASTMLX_LAUNCH.mtp_launch_requested(profile, "resident"))
        self.assertFalse("--mtp" in profile["argv"])


if __name__ == "__main__":
    unittest.main()
