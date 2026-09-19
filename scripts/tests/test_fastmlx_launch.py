import contextlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.tests.test_fastmlx_gguf_fit import build_gguf_bytes


LAUNCH_PATH = Path(__file__).resolve().parents[1] / "fastmlx_launch.py"
_SPEC = importlib.util.spec_from_file_location("fastmlx_launch", LAUNCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
FASTMLX_LAUNCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(FASTMLX_LAUNCH)

GGUF_FIT_CHECK_PATH = Path(__file__).resolve().parents[1] / "fastmlx_gguf_fit.py"


NO_GO_CARD_ID = "fixture-no-go@test"
NO_GO_REPO = "example/NoGoModel"
NO_GO_TIER = "Noticeable"
NO_GO_HEADLINE = "About 1 word in 6 differs from the reference model."

PASS_CARD_ID = "fixture-pass@test"
PASS_REPO = "example/PassModel"

EXACT_CARD_ID = "fixture-exact@test"
EXACT_REPO = "example/ExactModel"

# A public card identifying its pack only by a pinned-revision prefix, with
# no repo at all -- the pin-based matching path the spec amendment adds.
PIN_ONLY_CARD_ID = "fixture-pin-only@test"
PIN_ONLY_HF_PIN = "abc123ef"  # exactly the 8-char minimum, valid hex
PIN_ONLY_REVISION = PIN_ONLY_HF_PIN + "0" * (40 - len(PIN_ONLY_HF_PIN))

# An hfPin-only NO_GO card (no repo at all): identifying a pulled pack this
# way only works if the launcher resolves the revision from the pull
# receipt -- the integration path this file's pull/launch receipt-path bug
# fix covers.
PIN_ONLY_NO_GO_CARD_ID = "fixture-pin-only-no-go@test"
PIN_ONLY_NO_GO_HF_PIN = "9fed1234"
PIN_ONLY_NO_GO_REVISION = PIN_ONLY_NO_GO_HF_PIN + "0" * (40 - len(PIN_ONLY_NO_GO_HF_PIN))
PIN_ONLY_NO_GO_TIER = "Noticeable"
PIN_ONLY_NO_GO_HEADLINE = "Pin-identified NO_GO pack for pull-receipt integration coverage."


def fixture_manifest() -> dict:
    return {
        "schema": "fast-mlx-quality-card-v1",
        "generatedAt": "2026-01-01T00:00:00Z",
        "cards": [
            {
                "id": NO_GO_CARD_ID,
                "model": {"repo": NO_GO_REPO, "hfPin": "deadbeef"},
                "verdict": "NO_GO",
                "admission": {
                    "default": False,
                    "optIn": True,
                    "reason": "quality-degraded vs reference",
                },
                "legible": {"tier": NO_GO_TIER, "headline": NO_GO_HEADLINE},
            },
            {
                "id": PASS_CARD_ID,
                "model": {"repo": PASS_REPO, "hfPin": "cafebabe"},
                "verdict": "PASS",
                "admission": {
                    "default": True,
                    "optIn": False,
                    "reason": "measured pass",
                },
                "legible": {"tier": "Reference", "headline": "Matches the reference closely."},
            },
            {
                "id": EXACT_CARD_ID,
                "model": {"repo": EXACT_REPO, "hfPin": "0123abcd"},
                "verdict": "EXACT",
                "admission": {
                    "default": True,
                    "optIn": False,
                    "reason": "token-exact",
                },
                "legible": {"tier": "Exact", "headline": "Token-exact with the reference."},
            },
            {
                "id": PIN_ONLY_CARD_ID,
                "model": {"repo": None, "hfPin": PIN_ONLY_HF_PIN},
                "verdict": "PASS",
                "admission": {
                    "default": True,
                    "optIn": False,
                    "reason": "measured pass (pin-identified pack, no repo)",
                },
                "legible": {
                    "tier": "Reference",
                    "headline": "Matches the reference closely (pin-identified).",
                },
            },
            {
                "id": PIN_ONLY_NO_GO_CARD_ID,
                "model": {"repo": None, "hfPin": PIN_ONLY_NO_GO_HF_PIN},
                "verdict": "NO_GO",
                "admission": {
                    "default": False,
                    "optIn": True,
                    "reason": "quality-degraded vs reference (pin-identified pack, no repo)",
                },
                "legible": {
                    "tier": PIN_ONLY_NO_GO_TIER,
                    "headline": PIN_ONLY_NO_GO_HEADLINE,
                },
            },
        ],
    }


def write_script(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def write_pull_receipt(
    model_dir: Path, repo_id, revision: str, dest: Path = None, recorded_dest: Path = None
) -> Path:
    """Write a receipt at exactly the path and in exactly the shape
    ``fastmlx_pull.pull()`` writes one for ``dest`` (defaulting to
    ``model_dir`` itself): the SAME ``receipt_path_for`` naming rule and the
    SAME ``downloader.write_exclusive`` call pull's own code uses, so this
    fixture can never drift from what a real pull actually produces.
    """
    dest = dest if dest is not None else model_dir
    receipt_path = FASTMLX_LAUNCH.pull.receipt_path_for(dest)
    receipt = {
        "format_version": 1,
        "repo_id": repo_id,
        "revision": revision,
        "dest": str(recorded_dest if recorded_dest is not None else dest),
        "attempts": 1,
        "max_attempts": 3,
        "total_files": 0,
        "total_bytes": 0,
        "reused_files": 0,
        "reused_bytes": 0,
        "downloader_script_sha256": "0" * 64,
        "files": {},
    }
    receipt_bytes = json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n"
    FASTMLX_LAUNCH.pull.downloader.write_exclusive(receipt_path, receipt_bytes)
    return receipt_path


GREEN_FIT_CHECK_BODY = f"""#!{sys.executable}
import sys
# The launcher's attestation search matches on the "fit_check_only=complete"
# substring alone (see _find_attestation_line); no other prefix is required.
print(
    "fit_check_only=complete weights_loaded=false route=scalar "
    "model=stub max_context_tokens=4096 memory_limit_bytes=1000 "
    "cache_limit_bytes=100 fit_check=green fit_binding=none "
    "weights_measured=True wired_limit_measured=True fit_estimate_measured=True "
    "fit_quant_bits=none fit_served_context=4096 fit_context_ceiling=4096 "
    "fit_context_capped=False"
)
sys.exit(0)
"""

RED_FIT_CHECK_BODY = f"""#!{sys.executable}
import sys
sys.stderr.write("fit refused: requested context exceeds the computed ceiling\\n")
sys.exit(2)
"""

UNKNOWN_EXIT_FIT_CHECK_BODY = f"""#!{sys.executable}
import sys
sys.stderr.write("unexpected crash\\n")
sys.exit(1)
"""

NO_ATTESTATION_FIT_CHECK_BODY = f"""#!{sys.executable}
import sys
print("some_other_line=true")
sys.exit(0)
"""

FAKE_ENGINE_BODY = f"""#!{sys.executable}
import json
import os
import sys

capture_path = os.environ["FAKE_ENGINE_CAPTURE_PATH"]
with open(capture_path, "w", encoding="utf-8") as handle:
    json.dump(sys.argv, handle)
sys.exit(0)
"""


class FastmlxLaunchTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        (self.model_dir / "config.json").write_text("{}", encoding="utf-8")

        self.manifest_path = self.root / "quality-guides.json"
        self.manifest_path.write_text(json.dumps(fixture_manifest()), encoding="utf-8")

        self.green_fit_bin = write_script(self.root / "fit-green.py", GREEN_FIT_CHECK_BODY)
        self.red_fit_bin = write_script(self.root / "fit-red.py", RED_FIT_CHECK_BODY)
        self.unknown_fit_bin = write_script(
            self.root / "fit-unknown.py", UNKNOWN_EXIT_FIT_CHECK_BODY
        )
        self.no_attestation_fit_bin = write_script(
            self.root / "fit-no-attestation.py", NO_ATTESTATION_FIT_CHECK_BODY
        )
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
        """The dry-run plan is always the LAST stdout line: an opt-in admission
        prints its one-line quality-flag message to stdout first, before the
        plan.
        """
        lines = [line for line in stdout.splitlines() if line.strip()]
        return json.loads(lines[-1])

    # ------------------------------------------------------------------
    # Green + PASS card: admitted, argv built from the built-in profile.
    # ------------------------------------------------------------------
    def test_green_and_pass_card_admits_with_correct_argv(self):
        argv = self.base_args(
            **{
                "--model-repo": PASS_REPO,
                "--context": "2048",
                "--host": "127.0.0.1",
                "--port": "9001",
            }
        ) + ["--dry-run"]
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        plan = json.loads(stdout)
        self.assertEqual(plan["admission"], "admit")
        self.assertEqual(plan["card"]["id"], PASS_CARD_ID)
        self.assertEqual(plan["fit"]["verdict"], "GREEN")
        engine_bin_abs = str(self.fake_engine_bin.resolve())
        self.assertEqual(
            plan["argv"],
            [
                engine_bin_abs,
                "--model",
                self.model_dir.resolve().name,
                "--model-path",
                str(self.model_dir.resolve()),
                "--host",
                "127.0.0.1",
                "--port",
                "9001",
                "--context",
                "2048",
            ],
        )

    def test_yellow_fit_verdict_is_reported_as_yellow_and_admitted(self):
        yellow_bin = write_script(
            self.root / "fit-yellow.py",
            GREEN_FIT_CHECK_BODY.replace("fit_check=green", "fit_check=yellow"),
        )
        argv = self.base_args(
            **{"--fit-check-bin": str(yellow_bin), "--model-repo": PASS_REPO, "--context": "2048"}
        ) + ["--dry-run"]
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout)["fit"]["verdict"], "YELLOW")

    # ------------------------------------------------------------------
    # NO_GO, no opt-in: refused, message names tier + headline + card id.
    # ------------------------------------------------------------------
    def test_no_go_card_without_opt_in_is_refused(self):
        # --dry-run is defensive here, not load-bearing: a correct refusal
        # happens before the dry-run/exec branch is even reached, but it
        # keeps a regression that fails to refuse from calling os.execv()
        # inside this in-process test run instead of failing cleanly.
        argv = self.base_args(**{"--model-repo": NO_GO_REPO, "--context": "2048"}) + [
            "--dry-run"
        ]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn(NO_GO_TIER, stderr)
        self.assertIn(NO_GO_HEADLINE, stderr)
        self.assertIn(NO_GO_CARD_ID, stderr)
        self.assertIn("--accept-quality", stderr)

    # ------------------------------------------------------------------
    # NO_GO, opted in by card id: admitted.
    # ------------------------------------------------------------------
    def test_no_go_card_opt_in_by_card_id_admits(self):
        argv = self.base_args(
            **{
                "--model-repo": NO_GO_REPO,
                "--context": "2048",
                "--accept-quality": NO_GO_CARD_ID,
            }
        ) + ["--dry-run"]
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["admission"], "admit_with_quality_flag")
        self.assertEqual(plan["card"]["id"], NO_GO_CARD_ID)
        self.assertIn(NO_GO_HEADLINE, stdout)

    # ------------------------------------------------------------------
    # NO_GO, opted in by repo: admitted.
    # ------------------------------------------------------------------
    def test_no_go_card_opt_in_by_repo_admits(self):
        argv = self.base_args(
            **{
                "--model-repo": NO_GO_REPO,
                "--context": "2048",
                "--accept-quality": NO_GO_REPO,
            }
        ) + ["--dry-run"]
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["admission"], "admit_with_quality_flag")

    # ------------------------------------------------------------------
    # EXACT card: admitted silently, same as PASS.
    # ------------------------------------------------------------------
    def test_exact_card_admits_silently(self):
        argv = self.base_args(
            **{"--model-repo": EXACT_REPO, "--context": "2048"}
        ) + ["--dry-run"]
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        plan = json.loads(stdout)
        self.assertEqual(plan["admission"], "admit")

    # ------------------------------------------------------------------
    # RED verdict, no --force: refused exit 2.
    # ------------------------------------------------------------------
    def test_red_fit_check_without_force_is_refused(self):
        # --dry-run is defensive: see the comment in
        # test_no_go_card_without_opt_in_is_refused.
        argv = self.base_args(
            **{"--fit-check-bin": str(self.red_fit_bin), "--context": "2048"}
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("RED", stderr)
        self.assertIn("requested context exceeds the computed ceiling", stderr)

    # ------------------------------------------------------------------
    # RED verdict + --force: admitted, fit recorded as RED-forced.
    # ------------------------------------------------------------------
    def test_red_fit_check_with_force_admits(self):
        argv = self.base_args(
            **{
                "--fit-check-bin": str(self.red_fit_bin),
                "--context": "2048",
                "--force": "",
            }
        )
        argv = [a for a in argv if a != ""] + ["--dry-run"]
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        plan = json.loads(stdout)
        self.assertEqual(plan["fit"]["verdict"], "RED-forced")

    # ------------------------------------------------------------------
    # Unknown/unrunnable fit outcome: exit 3, --force does not override.
    # ------------------------------------------------------------------
    def test_unknown_fit_outcome_refuses_even_with_force(self):
        argv = self.base_args(
            **{
                "--fit-check-bin": str(self.unknown_fit_bin),
                "--context": "2048",
                "--force": "",
            }
        )
        argv = [a for a in argv if a != ""] + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3)
        self.assertIn("fit check could not run", stderr)

    def test_green_exit_without_attestation_line_is_an_unknown_fit_outcome(self):
        argv = self.base_args(
            **{"--fit-check-bin": str(self.no_attestation_fit_bin), "--context": "2048"}
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3)
        self.assertIn("fit check could not run", stderr)
        self.assertIn("attestation", stderr)

    # ------------------------------------------------------------------
    # Missing config.json: refused exit 2.
    # ------------------------------------------------------------------
    def test_missing_config_json_is_refused(self):
        bare_dir = self.root / "bare-model"
        bare_dir.mkdir()
        argv = self.base_args(**{"--model-path": str(bare_dir), "--context": "2048"}) + [
            "--dry-run"
        ]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("config.json", stderr)

    # ------------------------------------------------------------------
    # A GGUF pack (no config.json, at least one top-level *.gguf file) is
    # an accepted model-directory layout: the precondition no longer
    # refuses it. (The stub fit-check binary used here ignores
    # --model-path entirely, so this isolates the precondition change from
    # the real GGUF fit checker, which is covered separately below.)
    # ------------------------------------------------------------------
    def test_gguf_only_model_dir_is_not_refused_at_the_precondition(self):
        gguf_dir = self.root / "gguf-model"
        gguf_dir.mkdir()
        (gguf_dir / "pack.gguf").write_bytes(b"not a real gguf file, just a marker")
        argv = self.base_args(**{"--model-path": str(gguf_dir), "--context": "2048"}) + [
            "--dry-run"
        ]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        self.assertNotIn("config.json", stderr)
        plan = json.loads(stdout)
        self.assertIn("argv", plan)

    def test_model_dir_with_neither_config_json_nor_gguf_is_refused_naming_both_layouts(self):
        bare_dir = self.root / "bare-model"
        bare_dir.mkdir()
        argv = self.base_args(**{"--model-path": str(bare_dir), "--context": "2048"}) + [
            "--dry-run"
        ]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("does not contain config.json or any .gguf file", stderr)

    def test_model_dir_with_only_a_gguf_named_directory_is_refused(self):
        # A directory named like a GGUF shard is not a GGUF shard: the
        # precondition only accepts a top-level .gguf REGULAR FILE.
        gguf_dir_trap = self.root / "gguf-dir-trap"
        gguf_dir_trap.mkdir()
        (gguf_dir_trap / "shard.gguf").mkdir()
        argv = self.base_args(**{"--model-path": str(gguf_dir_trap), "--context": "2048"}) + [
            "--dry-run"
        ]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("does not contain config.json or any .gguf file", stderr)

    def test_missing_model_directory_is_refused(self):
        argv = self.base_args(
            **{"--model-path": str(self.root / "does-not-exist"), "--context": "2048"}
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("does not exist", stderr)

    # ------------------------------------------------------------------
    # Explicit --card-id not found: refused exit 2 with a specific reason.
    # ------------------------------------------------------------------
    def test_explicit_card_id_not_found_is_refused(self):
        argv = self.base_args(
            **{"--card-id": "no-such-card@test", "--context": "2048"}
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("no-such-card@test", stderr)
        self.assertIn("no quality card", stderr)

    def test_explicit_quality_cards_path_missing_is_refused_exit_3(self):
        missing = self.root / "no-such-manifest.json"
        argv = self.base_args(
            **{"--quality-cards": str(missing), "--context": "2048"}
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3)
        self.assertIn(str(missing), stderr)
        self.assertIn("is missing or is not a quality-card manifest", stderr)

    def test_explicit_quality_cards_path_malformed_is_refused_exit_3(self):
        self.manifest_path.write_text("{\"schema\": 1}", encoding="utf-8")
        argv = self.base_args(**{"--context": "2048"}) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3)
        self.assertIn("is missing or is not a quality-card manifest", stderr)

    # ------------------------------------------------------------------
    # --model-repo defaults from the SIBLING pull receipt
    # (<dir-name>.pull-receipt.json, written by fastmlx_pull.receipt_path_for)
    # when omitted.
    # ------------------------------------------------------------------
    def test_model_repo_defaults_from_pull_receipt(self):
        write_pull_receipt(self.model_dir, repo_id=PASS_REPO, revision="e" * 40)
        argv = self.base_args(**{"--context": "2048"}) + ["--dry-run"]
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        plan = json.loads(stdout)
        self.assertEqual(plan["card"]["id"], PASS_CARD_ID)

    # ------------------------------------------------------------------
    # A receipt written the OLD in-dir way (DIR/.pull-receipt.json, which
    # nothing writes anymore) must no longer identify the model: without a
    # sibling receipt or explicit --model-repo, the model has no resolved
    # identity, so the PASS card never applies and the plan admits unmeasured
    # (a NO_GO card would instead have to be silently forced through).
    # ------------------------------------------------------------------
    def test_in_dir_pull_receipt_alone_no_longer_identifies_model(self):
        receipt = {"repo_id": PASS_REPO, "revision": "e" * 40}
        (self.model_dir / ".pull-receipt.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )
        argv = self.base_args(**{"--context": "2048"}) + ["--dry-run"]
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        plan = json.loads(stdout)
        self.assertEqual(plan["admission"], "admit_unmeasured")
        self.assertIsNone(plan["card"])

    # ------------------------------------------------------------------
    # A sibling receipt whose recorded "dest" resolves to a DIFFERENT
    # directory than this model path must be ignored -- a copied or stale
    # receipt must never identify the wrong pack.
    # ------------------------------------------------------------------
    def test_mismatched_destination_receipt_is_ignored(self):
        # The receipt sits where this model's receipt belongs, and its revision
        # would match the hfPin-only NO_GO card -- but it records a different
        # directory, so it must not identify this one.
        other_dest = self.root / "some-other-model-dir"
        write_pull_receipt(
            self.model_dir,
            repo_id=None,
            revision=PIN_ONLY_NO_GO_REVISION,
            recorded_dest=other_dest,
        )
        argv = self.base_args(**{"--context": "2048"}) + ["--dry-run"]
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        plan = json.loads(stdout)
        self.assertEqual(plan["admission"], "admit_unmeasured")
        self.assertIsNone(plan["card"])

    # ------------------------------------------------------------------
    # End-to-end: a receipt produced by pull's own receipt-writing code path
    # resolves the model's pinned revision, which matches an hfPin-only
    # NO_GO card -- refuses without --accept-quality, admits with it.
    # ------------------------------------------------------------------
    def test_pull_receipt_resolved_revision_gates_hf_pin_only_no_go_card(self):
        write_pull_receipt(
            self.model_dir, repo_id=None, revision=PIN_ONLY_NO_GO_REVISION
        )
        argv = self.base_args(**{"--context": "2048"}) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn(PIN_ONLY_NO_GO_TIER, stderr)
        self.assertIn(PIN_ONLY_NO_GO_HEADLINE, stderr)
        self.assertIn(PIN_ONLY_NO_GO_CARD_ID, stderr)
        self.assertIn("--accept-quality", stderr)

        argv_accepted = self.base_args(
            **{"--context": "2048", "--accept-quality": PIN_ONLY_NO_GO_CARD_ID}
        ) + ["--dry-run"]
        code, stdout, _ = self.run_main(argv_accepted)
        self.assertEqual(code, 0)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["admission"], "admit_with_quality_flag")
        self.assertEqual(plan["card"]["id"], PIN_ONLY_NO_GO_CARD_ID)

    # ------------------------------------------------------------------
    # --context omitted: use the fit check's context ceiling.
    # ------------------------------------------------------------------
    def test_context_defaults_from_fit_ceiling_when_omitted(self):
        argv = self.base_args() + ["--dry-run"]
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        plan = json.loads(stdout)
        self.assertEqual(plan["fit"]["fields"]["fit_context_ceiling"], "4096")
        self.assertIn("--context", plan["argv"])
        self.assertEqual(plan["argv"][plan["argv"].index("--context") + 1], "4096")

    def test_red_forced_without_context_and_without_ceiling_refuses_exit_3(self):
        argv = self.base_args(
            **{"--fit-check-bin": str(self.red_fit_bin), "--force": ""}
        )
        argv = [a for a in argv if a != ""] + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3)
        self.assertIn("context could not be determined", stderr)

    # ------------------------------------------------------------------
    # Unknown placeholder in a custom engine profile: refused exit 3.
    # ------------------------------------------------------------------
    def test_unknown_placeholder_in_engine_profile_is_refused(self):
        profile_path = self.root / "bad-profile.json"
        profile_path.write_text(
            json.dumps(
                {
                    "schema": "fastmlx-engine-profile-v1",
                    "name": "custom",
                    "argv": ["{engine_bin}", "--bogus", "{bogus}"],
                }
            ),
            encoding="utf-8",
        )
        argv = self.base_args(
            **{"--engine-profile": str(profile_path), "--context": "2048"}
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3)
        self.assertIn("{bogus}", stderr)
        self.assertIn("unknown", stderr.lower())

    def test_malformed_engine_profile_schema_is_refused(self):
        profile_path = self.root / "not-a-profile.json"
        profile_path.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
        argv = self.base_args(
            **{"--engine-profile": str(profile_path), "--context": "2048"}
        )
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3)
        self.assertIn("fastmlx-engine-profile-v1", stderr)

    # ------------------------------------------------------------------
    # `--` passthrough args are appended to the exec'd argv.
    # ------------------------------------------------------------------
    def test_passthrough_args_after_double_dash_are_appended(self):
        argv = self.base_args(**{"--context": "2048"}) + [
            "--dry-run",
            "--",
            "--extra-flag",
            "value",
        ]
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        plan = json.loads(stdout)
        self.assertEqual(plan["argv"][-2:], ["--extra-flag", "value"])

    # ------------------------------------------------------------------
    # --dry-run: prints the plan, never execs.
    # ------------------------------------------------------------------
    def test_dry_run_prints_plan_and_does_not_exec(self):
        capture_path = self.root / "captured-argv.json"
        self.assertFalse(capture_path.exists())
        os.environ["FAKE_ENGINE_CAPTURE_PATH"] = str(capture_path)
        try:
            argv = self.base_args(**{"--context": "2048"}) + ["--dry-run"]
            code, stdout, _ = self.run_main(argv)
        finally:
            del os.environ["FAKE_ENGINE_CAPTURE_PATH"]
        self.assertEqual(code, 0)
        plan = json.loads(stdout)
        self.assertIn("argv", plan)
        self.assertFalse(capture_path.exists(), "dry-run must never exec the engine")

    # ------------------------------------------------------------------
    # Built-in profile argv shape (this repo's own in-tree serving CLI).
    # ------------------------------------------------------------------
    def test_built_in_profile_argv_shape(self):
        argv = self.base_args(**{"--context": "1234", "--host": "0.0.0.0", "--port": "9000"}) + [
            "--dry-run"
        ]
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        plan = json.loads(stdout)
        engine_bin_abs = str(self.fake_engine_bin.resolve())
        self.assertEqual(
            plan["argv"],
            [
                engine_bin_abs,
                "--model",
                self.model_dir.resolve().name,
                "--model-path",
                str(self.model_dir.resolve()),
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
                "--context",
                "1234",
            ],
        )

    # ------------------------------------------------------------------
    # A real exec: run the launcher out-of-process so os.execv does not
    # replace the test runner, and confirm the fake engine actually ran
    # with the argv the launcher built.
    # ------------------------------------------------------------------
    def test_admitted_run_actually_execs_the_engine(self):
        capture_path = self.root / "captured-argv.json"
        env = dict(os.environ)
        env["FAKE_ENGINE_CAPTURE_PATH"] = str(capture_path)
        argv = [
            sys.executable,
            str(LAUNCH_PATH),
        ] + self.base_args(**{"--context": "2048", "--model-repo": PASS_REPO})
        result = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fastmlx_launch=admitted", result.stderr)
        captured_argv = json.loads(capture_path.read_text(encoding="utf-8"))
        self.assertEqual(captured_argv[0], str(self.fake_engine_bin.resolve()))
        self.assertIn("--context", captured_argv)
        self.assertEqual(captured_argv[captured_argv.index("--context") + 1], "2048")

    # ------------------------------------------------------------------
    # A public card with model.repo == null, identified only by an hfPin
    # prefix of the model's pinned revision.
    # ------------------------------------------------------------------
    def test_pin_prefix_match_with_repo_null_admits(self):
        argv = self.base_args(
            **{"--model-revision": PIN_ONLY_REVISION, "--context": "2048"}
        ) + ["--dry-run"]
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["admission"], "admit")
        self.assertEqual(plan["card"]["id"], PIN_ONLY_CARD_ID)

    def test_pin_match_also_works_from_pull_receipt_revision(self):
        write_pull_receipt(self.model_dir, repo_id=None, revision=PIN_ONLY_REVISION)
        argv = self.base_args(**{"--context": "2048"}) + ["--dry-run"]
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["card"]["id"], PIN_ONLY_CARD_ID)

    # ------------------------------------------------------------------
    # Opt-in by hfPin.
    # ------------------------------------------------------------------
    def test_opt_in_by_hf_pin_admits_a_no_go_card(self):
        no_go_hf_pin = "deadbeef"  # matches the NO_GO fixture card's hfPin
        argv = self.base_args(
            **{
                "--model-repo": NO_GO_REPO,
                "--context": "2048",
                "--accept-quality": no_go_hf_pin,
            }
        ) + ["--dry-run"]
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["admission"], "admit_with_quality_flag")
        self.assertEqual(plan["card"]["id"], NO_GO_CARD_ID)

    # ------------------------------------------------------------------
    # Ambiguous lookup: a repo match and a pin match name different cards.
    # ------------------------------------------------------------------
    def test_ambiguous_repo_and_pin_match_refuses_exit_3(self):
        # --model-repo resolves the PASS card by repo; --model-revision,
        # built from the NO_GO card's own hfPin ("deadbeef"), resolves a
        # DIFFERENT card by pin. Neither identity is wrong on its own --
        # the manifest is simply naming the same run two ways.
        conflicting_revision = "deadbeef" + "0" * 32
        argv = self.base_args(
            **{
                "--model-repo": PASS_REPO,
                "--model-revision": conflicting_revision,
                "--context": "2048",
            }
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3)
        self.assertIn("ambiguous", stderr)
        self.assertIn(PASS_CARD_ID, stderr)
        self.assertIn(NO_GO_CARD_ID, stderr)

    # ------------------------------------------------------------------
    # --card-id is honoured only if it can be verified against the
    # resolved model identity.
    # ------------------------------------------------------------------
    def test_card_id_mismatched_with_model_identity_is_refused(self):
        argv = self.base_args(
            **{
                "--card-id": PASS_CARD_ID,
                "--model-repo": NO_GO_REPO,
                "--context": "2048",
            }
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn(PASS_CARD_ID, stderr)
        self.assertIn("cannot be verified", stderr)

    def test_card_id_with_no_model_identity_is_refused(self):
        argv = self.base_args(
            **{"--card-id": PASS_CARD_ID, "--context": "2048"}
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn(PASS_CARD_ID, stderr)
        self.assertIn("no model identity is known", stderr)

    def test_card_id_verified_by_matching_repo_admits(self):
        argv = self.base_args(
            **{
                "--card-id": PASS_CARD_ID,
                "--model-repo": PASS_REPO,
                "--context": "2048",
            }
        ) + ["--dry-run"]
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["card"]["id"], PASS_CARD_ID)

    def test_card_id_verified_by_matching_pin_admits(self):
        argv = self.base_args(
            **{
                "--card-id": PIN_ONLY_CARD_ID,
                "--model-revision": PIN_ONLY_REVISION,
                "--context": "2048",
            }
        ) + ["--dry-run"]
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["card"]["id"], PIN_ONLY_CARD_ID)


class CardMatchingHelperTests(unittest.TestCase):
    """Pure-function edge cases for the hfPin-prefix matching the spec
    amendment adds: a short (<8 hex chars) pin, and a non-hex pin, must
    never match even when the textual prefix would otherwise line up.
    """

    def test_short_hf_pin_is_not_matched(self):
        short_pin = "abc1"  # 4 hex chars: below the 8-char minimum
        revision = short_pin + "0" * 36
        self.assertFalse(FASTMLX_LAUNCH._hf_pin_matches_revision(short_pin, revision))

    def test_non_hex_hf_pin_is_not_matched(self):
        non_hex_pin = "zzzzzzzz"  # 8 chars, but not hex digits
        revision = "a" * 40
        self.assertFalse(FASTMLX_LAUNCH._hf_pin_matches_revision(non_hex_pin, revision))

    def test_non_full_length_revision_is_not_matched(self):
        pin = "abc123ef"
        short_revision = pin + "0" * 10  # valid hex, but only 18 chars long
        self.assertFalse(FASTMLX_LAUNCH._hf_pin_matches_revision(pin, short_revision))

    def test_case_insensitive_prefix_match(self):
        pin = "ABC123EF"
        revision = "abc123ef" + "0" * 32
        self.assertTrue(FASTMLX_LAUNCH._hf_pin_matches_revision(pin, revision))

    def test_find_card_by_pin_ignores_cards_with_no_hf_pin(self):
        cards = [{"id": "no-pin", "model": {"repo": None}}]
        self.assertIsNone(
            FASTMLX_LAUNCH.find_card_by_pin(cards, "a" * 40)
        )


class GgufFitCheckCallSiteTests(unittest.TestCase):
    """CALL-SITE coverage for the GGUF-pack admission fix: drives the real
    ``scripts/fastmlx_gguf_fit.py`` binary through the launcher's CLI entry
    point (``fastmlx serve``, i.e. ``FASTMLX_LAUNCH.main``), not through
    ``run_fit_check`` directly -- proving a GGUF-only model directory can
    reach and be classified by the fit check at all, now that the
    ``config.json`` precondition no longer refuses it first.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.assertTrue(
            GGUF_FIT_CHECK_PATH.is_file(), f"missing {GGUF_FIT_CHECK_PATH}"
        )
        GGUF_FIT_CHECK_PATH.chmod(
            GGUF_FIT_CHECK_PATH.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
        )

        self.gguf_dir = self.root / "gguf-pack"
        self.gguf_dir.mkdir()
        tensors = [
            {"name": "token_embd.weight", "dims": [4], "type": 0, "offset": 0}  # F32
        ]
        blob = build_gguf_bytes(tensors=tensors, data_section=bytes(16))
        (self.gguf_dir / "pack.gguf").write_bytes(blob)

        self.manifest_path = self.root / "quality-guides.json"
        self.manifest_path.write_text(json.dumps(fixture_manifest()), encoding="utf-8")
        self.fake_engine_bin = write_script(self.root / "fake-engine.py", FAKE_ENGINE_BODY)

    def _argv(self, fit_check_args: list) -> list:
        argv = [
            "serve",
            "--model-path", str(self.gguf_dir),
            "--quality-cards", str(self.manifest_path),
            "--engine-bin", str(self.fake_engine_bin),
            "--fit-check-bin", str(GGUF_FIT_CHECK_PATH),
            "--context", "2048",
        ]
        for value in fit_check_args:
            # "=" form on purpose: a bare "--fit-check-arg VALUE" pair makes
            # argparse treat a VALUE that itself starts with "--" (e.g.
            # "--kv-reserve-gib") as a new option rather than this flag's
            # argument.
            argv.append(f"--fit-check-arg={value}")
        argv.append("--dry-run")
        return argv

    def run_main(self, argv: list):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                FASTMLX_LAUNCH.main(argv)
        return ctx.exception.code, stdout.getvalue(), stderr.getvalue()

    @staticmethod
    def _env_without_stray_fastmlx_vars() -> dict:
        # Isolate the run from any FASTMLX_* variable already present in the
        # test process's environment (e.g. FASTMLX_WIRED_LIMIT_MIB,
        # FASTMLX_WIRED_MARGIN_GIB): the fit-check subprocess inherits
        # os.environ, and a stray value there would make the ceiling
        # computed below nondeterministic.
        return {k: v for k, v in os.environ.items() if not k.startswith("FASTMLX_")}

    def test_gguf_pack_that_fits_admits_and_would_exec_the_engine(self):
        # A wired-limit comfortably above the default 8 GiB margin and a
        # zero KV reserve: the 16-byte fixture pack fits easily -> GREEN.
        argv = self._argv(
            ["--kv-reserve-gib", "0", "--wired-limit-mib", "16384"]
        )
        with patch.dict(os.environ, self._env_without_stray_fastmlx_vars(), clear=True):
            code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        plan = json.loads(stdout.splitlines()[-1])
        self.assertEqual(plan["fit"]["fields"].get("fit"), "green")
        self.assertEqual(plan["argv"][0], str(self.fake_engine_bin.resolve()))

    def test_gguf_pack_with_overflowing_kv_reserve_is_refused_as_does_not_fit(self):
        # A wired-limit just above the default 8 GiB margin (ceiling is
        # only 1 MiB) plus a 1 GiB KV reserve: legitimately exceeds the
        # ceiling -> RED, refused without --force at the launcher's own
        # RED exit code (2).
        argv = self._argv(
            ["--kv-reserve-gib", "1", "--wired-limit-mib", "8193"]
        )
        with patch.dict(os.environ, self._env_without_stray_fastmlx_vars(), clear=True):
            code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("fit_check=RED", stderr)
        self.assertIn("exceeds ceiling", stderr)


if __name__ == "__main__":
    unittest.main()
