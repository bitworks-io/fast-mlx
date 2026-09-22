import contextlib
import http.client
import importlib.util
import io
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.tests.test_fastmlx_gguf_fit import build_gguf_bytes
from scripts.tests.test_fastmlx_safetensors_fit import build_safetensors_bytes, zero_tensor_bytes


LAUNCH_PATH = Path(__file__).resolve().parents[1] / "fastmlx_launch.py"
_SPEC = importlib.util.spec_from_file_location("fastmlx_launch", LAUNCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
FASTMLX_LAUNCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(FASTMLX_LAUNCH)

GGUF_FIT_CHECK_PATH = Path(__file__).resolve().parents[1] / "fastmlx_gguf_fit.py"
SAFETENSORS_FIT_CHECK_PATH = Path(__file__).resolve().parents[1] / "fastmlx_safetensors_fit.py"


NO_GO_CARD_ID = "fixture-no-go@test"
NO_GO_REPO = "example/NoGoModel"
NO_GO_TIER = "Noticeable"
NO_GO_HEADLINE = "About 1 word in 6 differs from the reference model."

PASS_CARD_ID = "fixture-pass@test"
PASS_REPO = "example/PassModel"

EXACT_CARD_ID = "fixture-exact@test"
EXACT_REPO = "example/ExactModel"

# A NO_GO card WITH a measured legible.benefit -- the opt-in decision this
# repository's cycle-104 defect hid the speed upside from (see
# card_benefit_line). speedXStatus deliberately names a referent that must
# never be dropped or paraphrased.
NO_GO_WITH_BENEFIT_CARD_ID = "fixture-no-go-benefit@test"
NO_GO_WITH_BENEFIT_REPO = "example/NoGoBenefitModel"
NO_GO_WITH_BENEFIT_TIER = "Noticeable"
NO_GO_WITH_BENEFIT_HEADLINE = (
    "About 1 word in 6 differs from the reference model, but it is faster."
)
NO_GO_WITH_BENEFIT_REFERENT = "not against the full-precision model"

# A NO_GO card whose legible.benefit carries BOTH a fit clause and a
# speedXStatus -- the exact shape of the real published cards (e.g.
# qwen38-flash-next-mixed-4-8bit@m3ultra in site/quality-guides.json) that
# exposed the mislabeling defect: card_benefit_line used to concatenate the
# fit clause into the same string as the speed clause, and both call sites
# prefixed the combined string with "speed: ", presenting a fit fact as a
# speed fact. FIT_TEXT is deliberately size/host language that could never
# be mistaken for a speed measurement.
NO_GO_WITH_FIT_CARD_ID = "fixture-no-go-fit@test"
NO_GO_WITH_FIT_REPO = "example/NoGoFitModel"
NO_GO_WITH_FIT_TIER = "Noticeable"
NO_GO_WITH_FIT_HEADLINE = (
    "About 1 word in 6 differs from the reference model, but it is faster."
)
NO_GO_WITH_FIT_TEXT = (
    "70.1 GiB resident weights plus a memory-mapped n-gram table. "
    "GREEN on a 128 GB Mac."
)
NO_GO_WITH_FIT_STATUS = "measured on Apple M3 Ultra: ratio against the 8-bit reference pack"

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
                "id": NO_GO_WITH_BENEFIT_CARD_ID,
                "model": {"repo": NO_GO_WITH_BENEFIT_REPO, "hfPin": "beadfeed"},
                "verdict": "NO_GO",
                "admission": {
                    "default": False,
                    "optIn": True,
                    "reason": "quality-degraded vs reference",
                },
                "legible": {
                    "tier": NO_GO_WITH_BENEFIT_TIER,
                    "headline": NO_GO_WITH_BENEFIT_HEADLINE,
                    "benefit": {
                        "speedX": 1.19,
                        "speedXStatus": (
                            "measured on Apple M3 Ultra: ratio against the 8-bit "
                            "reference pack, " + NO_GO_WITH_BENEFIT_REFERENT + "."
                        ),
                    },
                },
            },
            {
                "id": NO_GO_WITH_FIT_CARD_ID,
                "model": {"repo": NO_GO_WITH_FIT_REPO, "hfPin": "beadfeed1"},
                "verdict": "NO_GO",
                "admission": {
                    "default": False,
                    "optIn": True,
                    "reason": "quality-degraded vs reference",
                },
                "legible": {
                    "tier": NO_GO_WITH_FIT_TIER,
                    "headline": NO_GO_WITH_FIT_HEADLINE,
                    "benefit": {
                        "fit": NO_GO_WITH_FIT_TEXT,
                        "speedX": 1.19,
                        "speedXStatus": NO_GO_WITH_FIT_STATUS,
                    },
                },
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


def GREEN_ATTESTATION_WITH_RESIDENCY_BODY(residency: str) -> str:
    """A GREEN fit-check stub whose attestation line carries its own
    ``residency=`` field (both real sizers do this) -- used to exercise the
    attestation-residency-mismatch fail-closed check.
    """
    return f"""#!{sys.executable}
import sys
print(
    "fit_check_only=complete weights_loaded=false route=scalar "
    "model=stub max_context_tokens=4096 memory_limit_bytes=1000 "
    "cache_limit_bytes=100 fit_check=green fit_binding=none "
    "weights_measured=True wired_limit_measured=True fit_estimate_measured=True "
    "fit_quant_bits=none fit_served_context=4096 fit_context_ceiling=4096 "
    "fit_context_capped=False residency={residency}"
)
sys.exit(0)
"""

UNKNOWN_EXIT_FIT_CHECK_BODY = f"""#!{sys.executable}
import sys
sys.stderr.write("unexpected crash\\n")
sys.exit(1)
"""

# A distinctive stderr reason an unrunnable sizer might print (e.g. the
# ngram-table sizer refusing a pack that has no ngram_table.bin) -- used to
# verify the refusal/error-row detail surfaces the sizer's OWN reason, not
# just its exit code.
DISTINCTIVE_UNRUNNABLE_REASON = "distinctive-reason: pack is missing ngram_table.bin"

DISTINCTIVE_REASON_FIT_CHECK_BODY = f"""#!{sys.executable}
import sys
sys.stderr.write({DISTINCTIVE_UNRUNNABLE_REASON!r} + "\\n")
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

# A fit-check stub that captures its own argv (to the path named by
# FIT_CHECK_CAPTURE_PATH) before emitting the same GREEN attestation line
# GREEN_FIT_CHECK_BODY does -- used to verify exactly which binary and argv
# an engine profile's `fitCheck` produced, without needing the real
# fastmlx_safetensors_fit.py/fastmlx_gguf_fit.py sizers to succeed against a
# fixture model directory that carries no real weights.
CAPTURING_FIT_CHECK_BODY = f"""#!{sys.executable}
import json
import os
import sys

capture_path = os.environ["FIT_CHECK_CAPTURE_PATH"]
with open(capture_path, "w", encoding="utf-8") as handle:
    json.dump(sys.argv, handle)
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
        self.distinctive_reason_fit_bin = write_script(
            self.root / "fit-distinctive-reason.py", DISTINCTIVE_REASON_FIT_CHECK_BODY
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
    # legible.benefit must reach the operator on BOTH the refusal path and
    # the opt-in path -- the user deciding whether to elect a NO_GO card is
    # exactly who needs to see what it buys, not only what it costs.
    # ------------------------------------------------------------------
    def test_opt_in_admission_states_the_measured_benefit(self):
        # Refusal path.
        argv = self.base_args(
            **{"--model-repo": NO_GO_WITH_BENEFIT_REPO, "--context": "2048"}
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("faster", stderr)
        self.assertIn(NO_GO_WITH_BENEFIT_REFERENT, stderr)
        self.assertIn("--accept-quality", stderr)

        # Opt-in path: the same benefit line still reaches the operator.
        argv = self.base_args(
            **{
                "--model-repo": NO_GO_WITH_BENEFIT_REPO,
                "--context": "2048",
                "--accept-quality": NO_GO_WITH_BENEFIT_CARD_ID,
            }
        ) + ["--dry-run"]
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        self.assertIn("faster", stdout)
        self.assertIn(NO_GO_WITH_BENEFIT_REFERENT, stdout)

    # ------------------------------------------------------------------
    # The defect this task repairs: a card whose legible.benefit carries
    # BOTH a fit clause and a speedXStatus must never present the fit
    # clause as if it were a speed fact. The fit clause must appear under
    # its own "fit: " label, and the "speed: " clause must not contain the
    # fit text at all. This is specific enough that the old "; "-joined
    # single string (prefixed once with "speed: ") fails it: that form
    # places the fit text right after "; " with no "fit: " label.
    # ------------------------------------------------------------------
    def test_fit_is_not_labelled_as_speed(self):
        argv = self.base_args(
            **{"--model-repo": NO_GO_WITH_FIT_REPO, "--context": "2048"}
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        # The fit text must appear, under its own "fit: " label.
        self.assertIn(f"fit: {NO_GO_WITH_FIT_TEXT}", stderr)
        # The old defect joined speed and fit with "; " into one string --
        # that exact join must be gone.
        self.assertNotIn(f"; {NO_GO_WITH_FIT_TEXT}", stderr)
        # The "speed: " clause itself (everything between "speed: " and
        # "fit: ") must not contain the fit text.
        speed_clause = stderr.split("speed: ", 1)[1].split("fit: ", 1)[0]
        self.assertNotIn(NO_GO_WITH_FIT_TEXT, speed_clause)

    # ------------------------------------------------------------------
    # Unit-level pin of the same defect: card_benefit_line itself must
    # never return the fit text -- it is the speed line ONLY.
    # ------------------------------------------------------------------
    def test_card_benefit_line_excludes_fit(self):
        card = {
            "legible": {
                "benefit": {
                    "fit": NO_GO_WITH_FIT_TEXT,
                    "speedX": 1.19,
                    "speedXStatus": NO_GO_WITH_FIT_STATUS,
                }
            }
        }
        line = FASTMLX_LAUNCH.card_benefit_line(card)
        self.assertIsNotNone(line)
        self.assertNotIn(NO_GO_WITH_FIT_TEXT, line)

    # ------------------------------------------------------------------
    # Anti-regression pin: a card with no legible.benefit at all
    # (NO_GO_CARD_ID has no "benefit" key) must refuse byte-identically to
    # today -- no "speed:" line appears out of nowhere.
    # ------------------------------------------------------------------
    def test_card_without_benefit_renders_unchanged(self):
        argv = self.base_args(**{"--model-repo": NO_GO_REPO, "--context": "2048"}) + [
            "--dry-run"
        ]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn(NO_GO_TIER, stderr)
        self.assertIn(NO_GO_HEADLINE, stderr)
        self.assertNotIn("speed:", stderr)

    # ------------------------------------------------------------------
    # Structural guard: the CLI helper must agree with the site renderer's
    # polarity on every direction, so this defect class cannot reappear on
    # a third surface. Strengthened to EXACT STRING EQUALITY (not merely
    # polarity agreement) and exercised with a non-null "fit" among the
    # cases -- exactly the shape that used to diverge, since the old
    # card_benefit_line concatenated fit into the returned string while
    # quality_speed_line never does.
    # ------------------------------------------------------------------
    def test_speed_direction_matches_site_renderer(self):
        scripts_dir = str(Path(__file__).resolve().parents[1])
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import build_public_site  # noqa: E402  (test-only; never imported by the CLI)

        cases = (
            (1.19, None),
            (1.00, None),
            (0.485, None),
            (None, None),
            (1.19, "20.7 GB — fits a 24 GB Mac"),
        )
        for speed_x, fit in cases:
            benefit = {
                "speedX": speed_x,
                "speedXStatus": "measured on Apple M3 Ultra",
                "fit": fit,
            }
            card = {"legible": {"benefit": benefit}}
            expected = build_public_site.quality_speed_line(benefit)
            actual = FASTMLX_LAUNCH.card_benefit_line(card)
            self.assertEqual(actual, expected, msg=f"speedX={speed_x!r} fit={fit!r}")

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

    def test_unknown_fit_outcome_detail_includes_sizers_own_stderr_reason(self):
        # A sizer that exits 1 (neither the GREEN 0 nor the RED 2 the fit
        # check protocol defines) is still fail-closed, but the operator
        # should see WHY the sizer itself gave up, not only its exit code.
        argv = self.base_args(
            **{"--fit-check-bin": str(self.distinctive_reason_fit_bin), "--context": "2048"}
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3)
        self.assertIn("fit check could not run", stderr)
        self.assertIn(DISTINCTIVE_UNRUNNABLE_REASON, stderr)

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
    # W2 part B: a hand-staged pack with NO resolved model identity at all
    # (no --model-repo, no --model-revision, no sibling pull receipt) means
    # no card could ever match this launch -- made visible with one stderr
    # line, without changing the (still admit_unmeasured) outcome.
    # ------------------------------------------------------------------
    def test_no_identity_prints_visible_line_and_still_admits_unmeasured(self):
        argv = self.base_args(**{"--context": "2048"}) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["admission"], "admit_unmeasured")
        self.assertIsNone(plan["card"])
        self.assertIn("no model identity", stderr)
        self.assertIn("no pull receipt", stderr)
        self.assertIn("--model-revision", stderr)
        self.assertIn("fastmlx pull", stderr)
        self.assertIn("--adopt", stderr)

    def test_no_identity_line_is_absent_when_model_revision_is_given(self):
        argv = self.base_args(
            **{"--model-revision": "e" * 40, "--context": "2048"}
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["admission"], "admit_unmeasured")
        self.assertNotIn("no model identity", stderr)

    def test_no_identity_line_is_absent_when_a_card_resolves_by_repo(self):
        argv = self.base_args(
            **{"--model-repo": PASS_REPO, "--context": "2048"}
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["card"]["id"], PASS_CARD_ID)
        self.assertNotIn("no model identity", stderr)

    def test_no_identity_line_is_absent_when_sibling_receipt_supplies_identity(self):
        write_pull_receipt(self.model_dir, repo_id=PASS_REPO, revision="e" * 40)
        argv = self.base_args(**{"--context": "2048"}) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["card"]["id"], PASS_CARD_ID)
        self.assertNotIn("no model identity", stderr)

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
    # Engine profile `fitCheck`: an optional pointer to the sizer this
    # profile's own engine needs, resolved either from a `builtin:` name
    # (a sibling of fastmlx_launch.py) or an absolute path.
    # ------------------------------------------------------------------
    def _write_fit_check_profile(self, fit_check: dict, name: str = "served-engine") -> Path:
        profile_path = self.root / "fit-check-profile.json"
        profile_path.write_text(
            json.dumps(
                {
                    "schema": "fastmlx-engine-profile-v1",
                    "name": name,
                    "argv": list(FASTMLX_LAUNCH.BUILT_IN_ENGINE_PROFILE["argv"]),
                    "fitCheck": fit_check,
                }
            ),
            encoding="utf-8",
        )
        return profile_path

    def test_builtin_safetensors_fit_check_resolves_to_sibling_and_runs_profile_args_before_cli_args(
        self,
    ):
        profile_path = self._write_fit_check_profile(
            {"bin": "builtin:safetensors", "args": ["--kv-reserve-gib", "8"]}
        )
        captured: dict = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=(
                    "fit_check_only=complete fit_check=green "
                    "fit_context_ceiling=4096 fit_served_context=4096"
                ),
                stderr="",
            )

        argv = self.base_args(
            **{
                "--engine-profile": str(profile_path),
                "--fit-check-bin": None,
                "--context": "2048",
            }
        ) + ["--fit-check-arg=--mmap-side-file", "--dry-run"]
        with patch.object(FASTMLX_LAUNCH.subprocess, "run", side_effect=fake_run):
            code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        expected_bin = str(
            (LAUNCH_PATH.parent / "fastmlx_safetensors_fit.py").resolve()
        )
        self.assertEqual(captured["argv"][0], expected_bin)
        # Profile args come before the CLI's own --fit-check-arg.
        self.assertEqual(
            captured["argv"][-3:], ["--kv-reserve-gib", "8", "--mmap-side-file"]
        )

    def test_cli_fit_check_bin_overrides_profile_fit_check_and_drops_profile_args(self):
        profile_path = self._write_fit_check_profile(
            {"bin": "builtin:safetensors", "args": ["--kv-reserve-gib", "8"]}
        )
        capture_path = self.root / "captured-fit-argv.json"
        capturing_bin = write_script(self.root / "fit-capture.py", CAPTURING_FIT_CHECK_BODY)
        os.environ["FIT_CHECK_CAPTURE_PATH"] = str(capture_path)
        try:
            argv = self.base_args(
                **{
                    "--engine-profile": str(profile_path),
                    "--fit-check-bin": str(capturing_bin),
                    "--context": "2048",
                }
            ) + ["--dry-run"]
            code, stdout, stderr = self.run_main(argv)
        finally:
            del os.environ["FIT_CHECK_CAPTURE_PATH"]
        self.assertEqual(code, 0, stderr)
        self.assertIn("overrides", stderr)
        self.assertIn("served-engine", stderr)
        captured_argv = json.loads(capture_path.read_text(encoding="utf-8"))
        self.assertNotIn("--kv-reserve-gib", captured_argv)

    def test_env_fit_check_bin_overrides_profile_fit_check_and_drops_profile_args(self):
        profile_path = self._write_fit_check_profile(
            {"bin": "builtin:safetensors", "args": ["--kv-reserve-gib", "8"]}
        )
        capture_path = self.root / "captured-fit-argv.json"
        capturing_bin = write_script(self.root / "fit-capture.py", CAPTURING_FIT_CHECK_BODY)
        env_patch = {"FASTMLX_FIT_CHECK_BIN": str(capturing_bin), "FIT_CHECK_CAPTURE_PATH": str(capture_path)}
        argv = self.base_args(
            **{
                "--engine-profile": str(profile_path),
                "--fit-check-bin": None,
                "--context": "2048",
            }
        ) + ["--dry-run"]
        with patch.dict(os.environ, env_patch):
            code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        self.assertIn("overrides", stderr)
        captured_argv = json.loads(capture_path.read_text(encoding="utf-8"))
        self.assertNotIn("--kv-reserve-gib", captured_argv)

    def test_fit_check_unknown_builtin_name_is_refused(self):
        profile_path = self._write_fit_check_profile({"bin": "builtin:bogus"})
        argv = self.base_args(
            **{"--engine-profile": str(profile_path), "--context": "2048"}
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3)
        self.assertIn("builtin:bogus", stderr)
        self.assertIn("unknown", stderr.lower())

    def test_fit_check_relative_path_bin_is_refused(self):
        profile_path = self._write_fit_check_profile({"bin": "relative/fit.py"})
        argv = self.base_args(
            **{"--engine-profile": str(profile_path), "--context": "2048"}
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3)
        self.assertIn("relative/fit.py", stderr)
        self.assertIn("absolute", stderr.lower())

    def test_fit_check_reserved_arg_is_refused(self):
        profile_path = self._write_fit_check_profile(
            {"bin": "builtin:gguf", "args": ["--model"]}
        )
        argv = self.base_args(
            **{"--engine-profile": str(profile_path), "--context": "2048"}
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3)
        self.assertIn("--model", stderr)

    def test_fit_check_unknown_key_is_refused(self):
        profile_path = self._write_fit_check_profile(
            {"bin": "builtin:gguf", "bogusKey": True}
        )
        argv = self.base_args(
            **{"--engine-profile": str(profile_path), "--context": "2048"}
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3)
        self.assertIn("bogusKey", stderr)

    def test_fit_check_empty_arg_is_refused(self):
        profile_path = self._write_fit_check_profile(
            {"bin": "builtin:gguf", "args": [""]}
        )
        argv = self.base_args(
            **{"--engine-profile": str(profile_path), "--context": "2048"}
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3)
        self.assertIn("empty", stderr.lower())

    def test_fit_check_reserved_arg_with_equals_form_is_refused(self):
        profile_path = self._write_fit_check_profile(
            {"bin": "builtin:gguf", "args": ["--context=1024"]}
        )
        argv = self.base_args(
            **{"--engine-profile": str(profile_path), "--context": "2048"}
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3)
        self.assertIn("--context=1024", stderr)
        self.assertIn("--context", stderr)

    def test_fit_check_reserved_arg_host_use_with_equals_form_is_refused(self):
        profile_path = self._write_fit_check_profile(
            {"bin": "builtin:gguf", "args": ["--host-use=dedicated-serving"]}
        )
        argv = self.base_args(
            **{"--engine-profile": str(profile_path), "--context": "2048"}
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3)
        self.assertIn("--host-use", stderr)

    def test_fit_check_reserved_arg_abbreviation_is_refused(self):
        profile_path = self._write_fit_check_profile(
            {"bin": "builtin:gguf", "args": ["--cont", "1024"]}
        )
        argv = self.base_args(
            **{"--engine-profile": str(profile_path), "--context": "2048"}
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3)
        self.assertIn("--cont", stderr)
        self.assertIn("--context", stderr)

    def test_fit_check_reserved_arg_model_path_abbreviation_is_refused(self):
        profile_path = self._write_fit_check_profile(
            {"bin": "builtin:gguf", "args": ["--model-p", "/x"]}
        )
        argv = self.base_args(
            **{"--engine-profile": str(profile_path), "--context": "2048"}
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3)
        self.assertIn("--model-p", stderr)
        self.assertIn("--model-path", stderr)

    def test_fit_check_reserved_arg_host_use_abbreviation_is_refused(self):
        profile_path = self._write_fit_check_profile(
            {"bin": "builtin:gguf", "args": ["--host", "dedicated-serving"]}
        )
        argv = self.base_args(
            **{"--engine-profile": str(profile_path), "--context": "2048"}
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3)
        self.assertIn("--host", stderr)
        self.assertIn("--host-use", stderr)

    def test_fit_check_arg_residency_conflict_from_profile_args_is_refused(self):
        profile_path = self._write_fit_check_profile(
            {"bin": "builtin:gguf", "args": ["--residency", "expert-stream"]}
        )
        argv = self.base_args(
            **{
                "--engine-profile": str(profile_path),
                "--fit-check-bin": None,
                "--context": "2048",
            }
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("resident", stderr)
        self.assertIn("expert-stream", stderr)
        # Fix #3: the refusal must name the ACTUAL source of the conflicting
        # value (the engine profile's own fitCheck.args), never blame
        # --fit-check-arg for a conflict that came from the profile.
        self.assertIn("engine profile", stderr.lower())
        self.assertIn("fitcheck.args", stderr.lower())

    def test_fit_check_arg_residency_conflict_from_cli_names_cli_as_source(self):
        argv = self.base_args(
            **{
                "--residency": "resident",
                "--fit-check-arg": "--residency=expert-stream",
            }
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("--fit-check-arg", stderr)
        self.assertNotIn("engine profile", stderr.lower())

    def test_fit_check_arg_residency_conflict_scans_every_occurrence_not_just_first(self):
        # A first occurrence that agrees, followed by a later conflicting
        # one, must still be caught -- the old guard only checked the first.
        # "=" form throughout on purpose: a bare "--fit-check-arg VALUE"
        # pair makes argparse treat a VALUE that itself starts with "--" as
        # a new option rather than this flag's argument.
        argv = self.base_args(**{"--residency": "resident"}) + [
            "--fit-check-arg=--residency",
            "--fit-check-arg=resident",
            "--fit-check-arg=--residency=expert-stream",
            "--dry-run",
        ]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("resident", stderr)
        self.assertIn("expert-stream", stderr)

    def test_fit_check_arg_residency_conflict_via_abbreviation_is_refused(self):
        # "=" form on purpose: a bare "--fit-check-arg VALUE" pair makes
        # argparse treat a VALUE that itself starts with "--" as a new
        # option rather than this flag's argument.
        argv = self.base_args(**{"--residency": "resident"}) + [
            "--fit-check-arg=--resid=expert-stream",
            "--dry-run",
        ]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("--resid=expert-stream", stderr)
        self.assertIn("expert-stream", stderr)

    def test_fit_check_arg_residency_short_abbreviation_is_not_treated_as_residency(self):
        # "--res" (5 chars) is deliberately the shortest abbreviation this
        # guard recognizes; a 4-char-or-shorter prefix like "--re" must NOT
        # be misread as a residency assertion at all.
        argv = self.base_args(**{"--residency": "resident"}) + [
            "--fit-check-arg=--re",
            "--dry-run",
        ]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)

    # ------------------------------------------------------------------
    # Fix #2(b): a GREEN attestation's own residency= field must agree
    # with the launch's own --residency; a mismatch fails closed, even
    # under --force.
    # ------------------------------------------------------------------
    def test_attestation_residency_mismatch_is_refused(self):
        mismatched_bin = write_script(
            self.root / "fit-residency-mismatch.py",
            GREEN_ATTESTATION_WITH_RESIDENCY_BODY("expert-stream"),
        )
        argv = self.base_args(
            **{"--fit-check-bin": str(mismatched_bin), "--residency": "resident", "--context": "2048"}
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3)
        self.assertIn("resident", stderr)
        self.assertIn("expert-stream", stderr)

    def test_attestation_residency_mismatch_not_overridable_by_force(self):
        # --force only overrides a RED verdict; a mismatched-residency
        # attestation is a distinct, unconditional refusal.
        mismatched_bin = write_script(
            self.root / "fit-residency-mismatch-force.py",
            GREEN_ATTESTATION_WITH_RESIDENCY_BODY("expert-stream"),
        )
        argv = self.base_args(
            **{
                "--fit-check-bin": str(mismatched_bin),
                "--residency": "resident",
                "--context": "2048",
                "--force": "",
            }
        )
        argv = [a for a in argv if a != ""] + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3)

    def test_attestation_matching_residency_admits(self):
        matching_bin = write_script(
            self.root / "fit-residency-match.py",
            GREEN_ATTESTATION_WITH_RESIDENCY_BODY("resident"),
        )
        argv = self.base_args(
            **{"--fit-check-bin": str(matching_bin), "--residency": "resident", "--context": "2048"}
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)

    def test_attestation_without_residency_field_is_not_checked(self):
        # GREEN_FIT_CHECK_BODY prints no residency= field at all -- absence
        # must never be treated as a mismatch.
        argv = self.base_args(**{"--residency": "resident", "--context": "2048"}) + [
            "--dry-run"
        ]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)

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


def _free_tcp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _kill_pid_if_alive(pid: int) -> None:
    """L7 test cleanup: kill a recorded stub-engine pid even when the test
    that recorded it fails partway through (an ``addCleanup`` callback, not
    the ``finally`` block that only ever kills the LAUNCHER's own pid) --
    the launcher's own SIGKILL/crash leaves an unrecoverable, unforwarded
    engine child behind otherwise.
    """
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _minimal_front_plan(front_port: int, upstream_port: int) -> dict:
    """The smallest plan dict ``_run_front_mode``/``fastmlx_proxy.create_server``
    need: every field ``build_provenance_headers``/``build_provenance_body``
    read, plus a ``front`` key naming this launch's own (fake) ports.
    """
    return {
        "fit": {"verdict": "GREEN", "fields": {}},
        "card": None,
        "admission": "admit_unmeasured",
        "argv": [],
        "residency": "resident",
        "engineBuild": {"status": "unrecorded", "card": None, "launch": None},
        "mtp": {"status": "off", "divergentPrompts": None, "prompts": None},
        "front": {
            "host": "127.0.0.1",
            "port": front_port,
            "upstream": f"http://127.0.0.1:{upstream_port}",
        },
    }


class _FakeChild:
    """A ``subprocess.Popen``-shaped stand-in for ``_run_front_mode``'s exit-
    code-mapping and bind/Popen-ordering unit tests: ``wait()`` returns a
    fixed, caller-chosen code (never actually spawns anything), and
    ``send_signal``/``kill`` just record/no-op.
    """

    def __init__(self, wait_return: int):
        self._wait_return = wait_return
        self.signals_received: list = []

    def wait(self):
        return self._wait_return

    def send_signal(self, signum):
        self.signals_received.append(signum)

    def kill(self):
        pass


# A stub OpenAI-compatible engine, spawned as a real subprocess by the
# front-mode e2e test below: parses --host/--port off its own argv (the same
# shape the built-in engine profile substitutes), writes its own pid to
# STUB_ENGINE_PID_PATH (so the test can confirm the launcher's SIGTERM
# actually reaches and kills it), then serves one fixed JSON route.
STUB_ENGINE_BODY = (
    "#!" + sys.executable + "\n"
    "import http.server\n"
    "import os\n"
    "import sys\n"
    "\n"
    "argv = sys.argv[1:]\n"
    "host = argv[argv.index('--host') + 1]\n"
    "port = int(argv[argv.index('--port') + 1])\n"
    "\n"
    "with open(os.environ['STUB_ENGINE_PID_PATH'], 'w', encoding='utf-8') as handle:\n"
    "    handle.write(str(os.getpid()))\n"
    "\n"
    "class Handler(http.server.BaseHTTPRequestHandler):\n"
    "    def log_message(self, *a, **k):\n"
    "        pass\n"
    "    def do_GET(self):\n"
    "        payload = b'{\"stub\": true}'\n"
    "        self.send_response(200)\n"
    "        self.send_header('Content-Type', 'application/json')\n"
    "        self.send_header('Content-Length', str(len(payload)))\n"
    "        self.end_headers()\n"
    "        self.wfile.write(payload)\n"
    "\n"
    "if os.environ.get('STUB_ENGINE_GRACEFUL') == '1':\n"
    "    import signal\n"
    "    signal.signal(signal.SIGTERM, lambda *a: os._exit(0))\n"
    "\n"
    "server = http.server.ThreadingHTTPServer((host, port), Handler)\n"
    "server.serve_forever()\n"
)


# A generic stub engine for the --internal-engine-guard tests below: driven
# entirely by environment variables (never its own argv, which in front mode
# carries --host/--port/... it does not need to parse) so one script covers
# "exit with code N", "sleep and ignore SIGTERM", and "sleep and honor the
# default SIGTERM disposition" -- optionally recording its own pid first, so
# a test can confirm the guard actually reached (or killed) this exact
# process.
GUARD_TEST_ENGINE_BODY = (
    "#!" + sys.executable + "\n"
    "import os\n"
    "import signal\n"
    "import sys\n"
    "import time\n"
    "\n"
    "mode = os.environ.get('GUARD_TEST_ENGINE_MODE', '0')\n"
    "pid_path = os.environ.get('GUARD_TEST_ENGINE_PID_PATH')\n"
    "if pid_path:\n"
    "    with open(pid_path, 'w', encoding='utf-8') as handle:\n"
    "        handle.write(str(os.getpid()))\n"
    "if mode == 'ignore-sigterm':\n"
    "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "    time.sleep(30)\n"
    "elif mode == 'sleep':\n"
    "    time.sleep(30)\n"
    "else:\n"
    "    sys.exit(int(mode))\n"
)


class FrontProxyModeTests(FastmlxLaunchTestCase):
    """``--front-port``: opt-in provenance proxy in front of the engine.

    Reuses ``FastmlxLaunchTestCase.setUp``/``base_args``/``run_main`` so
    these cases start from exactly the same fixtures (model dir, quality
    manifest, GREEN fit-check stub) as every other launcher test.
    """

    # ------------------------------------------------------------------
    # Without --front-port: nothing changes. The plan carries no "front"
    # key at all, so every pre-existing dry-run assertion (byte-exact argv,
    # admission, fit verdict, ...) in FastmlxLaunchTestCase is untouched by
    # this feature -- this is the regression guard for that claim.
    # ------------------------------------------------------------------
    def test_dry_run_without_front_port_carries_no_front_key(self):
        argv = self.base_args(**{"--model-repo": PASS_REPO, "--context": "2048"}) + ["--dry-run"]
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        plan = json.loads(stdout)
        self.assertNotIn("front", plan)
        self.assertEqual(
            plan["argv"][plan["argv"].index("--host") + 1],
            "127.0.0.1",
            "without --front-port the engine keeps binding the launch's own --host",
        )

    # ------------------------------------------------------------------
    # With --front-port: the engine's {host} placeholder becomes 127.0.0.1
    # (loopback) regardless of --front-host, --port stays the backend port,
    # and the plan gains a "front" key.
    # ------------------------------------------------------------------
    def test_front_mode_dry_run_binds_engine_to_loopback_and_adds_front_key(self):
        argv = self.base_args(
            **{
                "--model-repo": PASS_REPO,
                "--context": "2048",
                "--host": "0.0.0.0",
                "--port": "8080",
                "--front-port": "9090",
                "--front-host": "0.0.0.0",
            }
        ) + ["--dry-run"]
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        plan = json.loads(stdout)
        self.assertEqual(
            plan["argv"][plan["argv"].index("--host") + 1],
            "127.0.0.1",
            "front mode must bind the engine to loopback, never --front-host",
        )
        self.assertEqual(plan["argv"][plan["argv"].index("--port") + 1], "8080")
        self.assertEqual(
            plan["front"],
            {"host": "0.0.0.0", "port": 9090, "upstream": "http://127.0.0.1:8080"},
        )

    # ------------------------------------------------------------------
    # Refusal: --front-port must not equal --port.
    # ------------------------------------------------------------------
    def test_front_port_equal_to_port_is_refused(self):
        argv = self.base_args(
            **{
                "--model-repo": PASS_REPO,
                "--context": "2048",
                "--port": "8080",
                "--front-port": "8080",
            }
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("--front-port", stderr)
        self.assertIn("cannot equal", stderr)
        self.assertIn("--port", stderr)

    # ------------------------------------------------------------------
    # Refusal: a passthrough argument that would let the engine bind a
    # different host/port than the proxy expects, bypassing it.
    # ------------------------------------------------------------------
    def test_front_mode_passthrough_host_override_is_refused(self):
        argv = self.base_args(
            **{
                "--model-repo": PASS_REPO,
                "--context": "2048",
                "--front-port": "9090",
            }
        ) + ["--dry-run", "--", "--host", "0.0.0.0"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("--host", stderr)
        self.assertIn("bypass", stderr)

    def test_front_mode_passthrough_port_equals_form_is_refused(self):
        argv = self.base_args(
            **{
                "--model-repo": PASS_REPO,
                "--context": "2048",
                "--front-port": "9090",
            }
        ) + ["--dry-run", "--", "--port=9999"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("--port=9999", stderr)
        self.assertIn("bypass", stderr)

    # ------------------------------------------------------------------
    # M1: the loopback guarantee depends on the PROFILE, not only the
    # passthrough-args refusal above. A custom engine profile whose argv
    # has no {host}/{port} placeholder at all, has a literal host/port-
    # bypass flag alongside the placeholders, or puts one in
    # residencyArgs, would let the engine bind wider than loopback even
    # though front mode believes it fixed the bind. Refuse all of these,
    # and the wider passthrough-flag family (--hostname/--bind/-H), while
    # still accepting the shipped example profiles and the built-in one.
    # ------------------------------------------------------------------
    def _write_profile(self, argv, residency_args=None, name="front-mode-profile"):
        document = {
            "schema": "fastmlx-engine-profile-v1",
            "name": name,
            "argv": argv,
        }
        if residency_args is not None:
            document["residencyArgs"] = residency_args
        profile_path = self.root / f"{name}.json"
        profile_path.write_text(json.dumps(document), encoding="utf-8")
        return profile_path

    def test_front_mode_refuses_profile_with_no_host_port_placeholder(self):
        profile_path = self._write_profile(
            ["{engine_bin}", "--serve", "--model", "{model_path}", "--ctx-size", "{context}"]
        )
        argv = self.base_args(
            **{
                "--model-repo": PASS_REPO,
                "--context": "2048",
                "--engine-profile": str(profile_path),
                "--front-port": "9090",
            }
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("{host}", stderr)
        self.assertIn("{port}", stderr)

    def test_front_mode_refuses_profile_argv_literal_hostname_override(self):
        profile_path = self._write_profile(
            [
                "{engine_bin}", "--serve", "--model", "{model_path}",
                "--host", "{host}", "--port", "{port}",
                "--hostname", "0.0.0.0",
            ]
        )
        argv = self.base_args(
            **{
                "--model-repo": PASS_REPO,
                "--context": "2048",
                "--engine-profile": str(profile_path),
                "--front-port": "9090",
            }
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("--hostname", stderr)
        self.assertIn("bypass", stderr)

    def test_front_mode_refuses_profile_argv_bind_equals_form(self):
        profile_path = self._write_profile(
            [
                "{engine_bin}", "--serve", "--model", "{model_path}",
                "--host", "{host}", "--port", "{port}",
                "--bind=0.0.0.0",
            ]
        )
        argv = self.base_args(
            **{
                "--model-repo": PASS_REPO,
                "--context": "2048",
                "--engine-profile": str(profile_path),
                "--front-port": "9090",
            }
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("--bind=0.0.0.0", stderr)
        self.assertIn("bypass", stderr)

    def test_front_mode_refuses_profile_argv_short_h_flag(self):
        profile_path = self._write_profile(
            [
                "{engine_bin}", "--serve", "--model", "{model_path}",
                "--host", "{host}", "--port", "{port}",
                "-H", "0.0.0.0",
            ]
        )
        argv = self.base_args(
            **{
                "--model-repo": PASS_REPO,
                "--context": "2048",
                "--engine-profile": str(profile_path),
                "--front-port": "9090",
            }
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("-H", stderr)
        self.assertIn("bypass", stderr)

    def test_front_mode_refuses_residency_args_host_override(self):
        profile_path = self._write_profile(
            [
                "{engine_bin}", "--serve", "--model", "{model_path}",
                "--host", "{host}", "--port", "{port}",
            ],
            residency_args={"expert-stream": ["--ssd-streaming", "--host", "0.0.0.0"]},
        )
        argv = self.base_args(
            **{
                "--model-repo": PASS_REPO,
                "--context": "2048",
                "--engine-profile": str(profile_path),
                "--front-port": "9090",
                "--residency": "expert-stream",
            }
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("residencyArgs", stderr)
        self.assertIn("--host", stderr)
        self.assertIn("bypass", stderr)

    def test_front_mode_passthrough_hostname_and_bind_and_short_h_are_refused(self):
        cases = [
            ["--hostname", "0.0.0.0"],
            ["--hostname=0.0.0.0"],
            ["--bind", "0.0.0.0"],
            ["--bind=0.0.0.0"],
            ["-H", "0.0.0.0"],
        ]
        for tokens in cases:
            with self.subTest(tokens=tokens):
                argv = self.base_args(
                    **{
                        "--model-repo": PASS_REPO,
                        "--context": "2048",
                        "--front-port": "9090",
                    }
                ) + ["--dry-run", "--"] + tokens
                code, _, stderr = self.run_main(argv)
                self.assertEqual(code, 2)
                self.assertIn(tokens[0], stderr)
                self.assertIn("bypass", stderr)

    def test_front_mode_accepts_shipped_example_profiles_and_builtin(self):
        example_dir = Path(__file__).resolve().parents[2] / "examples" / "engine-profiles"
        example_profiles = sorted(example_dir.glob("*.json"))
        self.assertTrue(example_profiles, "expected at least one shipped example profile")
        profile_choices = [None] + example_profiles
        for profile_path in profile_choices:
            with self.subTest(profile=profile_path):
                overrides = {
                    "--model-repo": PASS_REPO,
                    "--context": "2048",
                    "--front-port": "9090",
                }
                overrides["--engine-profile"] = str(profile_path) if profile_path else None
                argv = self.base_args(**overrides) + ["--dry-run"]
                code, stdout, stderr = self.run_main(argv)
                self.assertEqual(code, 0, stderr)
                plan = json.loads(stdout)
                self.assertEqual(plan["front"]["port"], 9090)

    # ------------------------------------------------------------------
    # --front-max-body-bytes: the proxy's ceiling on a single request's
    # declared Content-Length (see fastmlx_proxy.DEFAULT_MAX_REQUEST_BODY_
    # BYTES). 0, a negative count, and a non-numeric string are all
    # equally nonsensical as a body-size ceiling and must all refuse at
    # exit 2 with a named reason -- the SAME LaunchRefusal fail-closed
    # style as every other front-mode argument check above, never
    # argparse's own "invalid int value" usage error.
    # ------------------------------------------------------------------
    def test_front_max_body_bytes_invalid_values_are_refused(self):
        # The expected substring differs per case ("not an integer" vs.
        # "positive number of bytes") DELIBERATELY: both are unique to this
        # feature's own ``LaunchRefusal`` message text, unlike a bare
        # ``self.assertIn("--front-max-body-bytes", stderr)`` would be --
        # argparse's own "unrecognized arguments: --front-max-body-bytes 0"
        # usage error (if the flag were not registered at all) ALSO
        # contains the flag name and would make a weaker assertion here
        # pass for entirely the wrong reason.
        cases = {
            "zero": ("0", "positive number of bytes"),
            "negative": ("-1", "positive number of bytes"),
            "non-numeric": ("abc", "not an integer"),
        }
        for label, (value, expected_text) in cases.items():
            with self.subTest(label=label, value=value):
                argv = self.base_args(
                    **{
                        "--model-repo": PASS_REPO,
                        "--context": "2048",
                        "--front-port": "9090",
                        "--front-max-body-bytes": value,
                    }
                ) + ["--dry-run"]
                code, _, stderr = self.run_main(argv)
                self.assertEqual(code, 2, stderr)
                self.assertIn("--front-max-body-bytes", stderr)
                self.assertIn(expected_text, stderr)

    # ------------------------------------------------------------------
    # A valid --front-max-body-bytes reaches fastmlx_proxy.create_server:
    # asserted on the actual plumbed keyword value, not merely that the
    # process started -- an implementation that validated the flag but
    # never threaded it through would pass a weaker assertion here.
    # ------------------------------------------------------------------
    def test_front_max_body_bytes_valid_value_reaches_create_server(self):
        argv = self.base_args(
            **{
                "--model-repo": PASS_REPO,
                "--context": "2048",
                "--front-port": "9090",
                "--front-max-body-bytes": "12345",
            }
        )
        captured = {}

        def recording_create_server(*args, **kwargs):
            captured["kwargs"] = kwargs
            raise OSError("address already in use")

        with patch.object(
            FASTMLX_LAUNCH.fastmlx_proxy, "create_server", side_effect=recording_create_server
        ):
            code, _, stderr = self.run_main(argv)

        self.assertEqual(code, 3, stderr)
        self.assertEqual(captured["kwargs"].get("max_request_body_bytes"), 12345)

    # ------------------------------------------------------------------
    # --front-max-concurrent: the proxy's own ceiling on the number of
    # in-flight requests it will accept at once (see fastmlx_proxy.
    # DEFAULT_MAX_CONCURRENT_REQUESTS). 0, a negative count, and a
    # non-numeric string are all equally nonsensical as a concurrency
    # ceiling and must all refuse at exit 2 with a named reason -- the
    # SAME LaunchRefusal fail-closed style as --front-max-body-bytes
    # above, never argparse's own "invalid int value" usage error.
    # ------------------------------------------------------------------
    def test_front_max_concurrent_invalid_values_are_refused(self):
        cases = {
            "zero": ("0", "positive number of requests"),
            "negative": ("-1", "positive number of requests"),
            "non-numeric": ("abc", "not an integer"),
        }
        for label, (value, expected_text) in cases.items():
            with self.subTest(label=label, value=value):
                argv = self.base_args(
                    **{
                        "--model-repo": PASS_REPO,
                        "--context": "2048",
                        "--front-port": "9090",
                        "--front-max-concurrent": value,
                    }
                ) + ["--dry-run"]
                code, _, stderr = self.run_main(argv)
                self.assertEqual(code, 2, stderr)
                self.assertIn("--front-max-concurrent", stderr)
                self.assertIn(expected_text, stderr)

    # ------------------------------------------------------------------
    # A valid --front-max-concurrent reaches fastmlx_proxy.create_server:
    # asserted on the actual plumbed keyword value, not merely that the
    # process started -- an implementation that validated the flag but
    # never threaded it through would pass a weaker assertion here.
    # ------------------------------------------------------------------
    def test_front_max_concurrent_valid_value_reaches_create_server(self):
        argv = self.base_args(
            **{
                "--model-repo": PASS_REPO,
                "--context": "2048",
                "--front-port": "9090",
                "--front-max-concurrent": "7",
            }
        )
        captured = {}

        def recording_create_server(*args, **kwargs):
            captured["kwargs"] = kwargs
            raise OSError("address already in use")

        with patch.object(
            FASTMLX_LAUNCH.fastmlx_proxy, "create_server", side_effect=recording_create_server
        ):
            code, _, stderr = self.run_main(argv)

        self.assertEqual(code, 3, stderr)
        self.assertEqual(captured["kwargs"].get("max_concurrent_requests"), 7)

    # ------------------------------------------------------------------
    # Ordering: the proxy must bind FIRST -- a bind failure must never
    # start (and then have to kill) the engine child at all.
    # ------------------------------------------------------------------
    def test_bind_failure_never_starts_the_engine(self):
        front_port = _free_tcp_port()
        upstream_port = _free_tcp_port()
        plan = _minimal_front_plan(front_port, upstream_port)
        popen_calls = []

        def recording_popen(argv, **kwargs):
            popen_calls.append(argv)
            return _FakeChild(0)

        with patch.object(
            FASTMLX_LAUNCH.fastmlx_proxy,
            "create_server",
            side_effect=OSError("address already in use"),
        ):
            result = FASTMLX_LAUNCH._run_front_mode([], plan, popen=recording_popen)

        self.assertEqual(result, 3)
        self.assertEqual(popen_calls, [], "a bind failure must never invoke Popen")

    # ------------------------------------------------------------------
    # Orphan-engine guard: a listener already on the engine's loopback port
    # (e.g. left behind by a SIGKILLed earlier launcher) must refuse the
    # NEW launch before the engine is spawned, rather than silently proxy
    # to the old process during the new model's load.
    # ------------------------------------------------------------------
    def test_busy_engine_port_never_starts_the_engine(self):
        front_port = _free_tcp_port()
        engine_port = _free_tcp_port()
        busy_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        busy_listener.bind(("127.0.0.1", engine_port))
        busy_listener.listen(1)
        self.addCleanup(busy_listener.close)

        plan = _minimal_front_plan(front_port, engine_port)
        popen_calls = []

        def recording_popen(argv, **kwargs):
            popen_calls.append(argv)
            return _FakeChild(0)

        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = FASTMLX_LAUNCH._run_front_mode([], plan, popen=recording_popen)

        self.assertEqual(result, 3)
        self.assertEqual(popen_calls, [], "a busy engine port must never invoke Popen")
        self.assertIn(str(engine_port), stderr.getvalue())

        # The front socket must have been released on refusal: rebindable.
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", front_port))
        except OSError as exc:  # pragma: no cover - failure path itself is the assertion
            self.fail(f"proxy socket on {front_port} was never closed: {exc}")
        finally:
            probe.close()

    # ------------------------------------------------------------------
    # A port that was recently used (bound, connected to, then closed --
    # leaving it in a TIME_WAIT-ish state) but has no live listener must
    # NOT be mistaken for busy: front mode should proceed normally.
    # ------------------------------------------------------------------
    def test_recently_closed_engine_port_is_accepted(self):
        front_port = _free_tcp_port()

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        engine_port = listener.getsockname()[1]

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", engine_port))
        accepted, _ = listener.accept()
        accepted.close()
        client.close()
        listener.close()

        plan = _minimal_front_plan(front_port, engine_port)
        popen_calls = []

        def recording_popen(argv, **kwargs):
            popen_calls.append(argv)
            return _FakeChild(0)

        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            FASTMLX_LAUNCH._run_front_mode([], plan, popen=recording_popen)

        self.assertEqual(len(popen_calls), 1, "a free (recently closed) engine port must proceed")

    # ------------------------------------------------------------------
    # Ordering: if Popen raises AFTER a successful bind, the bound proxy
    # socket must be closed (not leaked) -- proven by re-binding the same
    # port immediately after.
    # ------------------------------------------------------------------
    def test_popen_oserror_after_bind_closes_the_proxy_socket(self):
        front_port = _free_tcp_port()
        upstream_port = _free_tcp_port()
        plan = _minimal_front_plan(front_port, upstream_port)

        def failing_popen(argv, **kwargs):
            raise OSError("engine binary not found")

        result = FASTMLX_LAUNCH._run_front_mode([], plan, popen=failing_popen)
        self.assertEqual(result, 3)

        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", front_port))
        except OSError as exc:  # pragma: no cover - failure path itself is the assertion
            self.fail(f"proxy socket on {front_port} was never closed: {exc}")
        finally:
            probe.close()

    # ------------------------------------------------------------------
    # Exit code: a signal-killed child (negative os.waitpid code) maps to
    # 128+signum; a zero-exit child maps to 1 (front mode never exits 0);
    # any other positive code passes through unchanged.
    # ------------------------------------------------------------------
    def test_exit_code_mapping_for_signal_killed_zero_and_nonzero_child(self):
        front_port = _free_tcp_port()
        upstream_port = _free_tcp_port()
        original_term = signal.getsignal(signal.SIGTERM)
        original_int = signal.getsignal(signal.SIGINT)
        self.addCleanup(signal.signal, signal.SIGTERM, original_term)
        self.addCleanup(signal.signal, signal.SIGINT, original_int)

        for wait_return, expected in ((-15, 143), (0, 1), (7, 7)):
            with self.subTest(wait_return=wait_return):
                plan = _minimal_front_plan(front_port, upstream_port)
                child = _FakeChild(wait_return)
                result = FASTMLX_LAUNCH._run_front_mode(
                    [], plan, popen=lambda argv, child=child, **kwargs: child
                )
                self.assertEqual(result, expected)

    # ------------------------------------------------------------------
    # A stop the launcher was asked for maps to 128+signum even when the
    # engine handles the forwarded signal and exits 0 (the served engine
    # shuts down gracefully; cycle-102 live run E8 saw exit 1). A different
    # engine code after the stop passes through, so a crash while shutting
    # down stays visible.
    # ------------------------------------------------------------------
    def test_requested_stop_with_graceful_engine_exit_maps_to_128_plus_signal(self):
        front_port = _free_tcp_port()
        upstream_port = _free_tcp_port()
        original_term = signal.getsignal(signal.SIGTERM)
        original_int = signal.getsignal(signal.SIGINT)
        self.addCleanup(signal.signal, signal.SIGTERM, original_term)
        self.addCleanup(signal.signal, signal.SIGINT, original_int)

        class _StoppedChild(_FakeChild):
            def __init__(self, wait_return, signum):
                super().__init__(wait_return)
                self._signum = signum

            def wait(self):
                # Deliver the stop through the handler _run_front_mode
                # installed, exactly as the OS would, then "exit".
                signal.getsignal(self._signum)(self._signum, None)
                return self._wait_return

        cases = (
            (signal.SIGTERM, 0, 143),
            (signal.SIGINT, 0, 130),
            (signal.SIGTERM, -signal.SIGTERM, 143),
            (signal.SIGTERM, 7, 7),
        )
        for signum, wait_return, expected in cases:
            with self.subTest(signum=signum, wait_return=wait_return):
                plan = _minimal_front_plan(front_port, upstream_port)
                child = _StoppedChild(wait_return, signum)
                result = FASTMLX_LAUNCH._run_front_mode(
                    [], plan, popen=lambda argv, child=child, **kwargs: child
                )
                self.assertEqual(child.signals_received, [signum])
                self.assertEqual(result, expected)

    # ------------------------------------------------------------------
    # F4 CRITICAL TRAP -- the guard's crash-signal fix must not move the
    # exit-status contract `_run_front_mode` maps: a crash still reaches
    # front mode as `child.wait()` returning a plain (non-negative)
    # 128+signum -- the guard's new own encoding, see
    # `_engine_lifeline_guard` -- and must still end up 139 for SIGSEGV; a
    # forwarded SIGTERM still reaches front mode the OLD way, as a negative
    # signal-encoded `wait()` return (the guard self-signalled, unchanged
    # for non-crash signals), and must still end up 143.
    # ------------------------------------------------------------------
    def test_front_mode_exit_contract_unchanged_for_crash_and_forwarded_stop(self):
        front_port = _free_tcp_port()
        upstream_port = _free_tcp_port()
        original_term = signal.getsignal(signal.SIGTERM)
        original_int = signal.getsignal(signal.SIGINT)
        self.addCleanup(signal.signal, signal.SIGTERM, original_term)
        self.addCleanup(signal.signal, signal.SIGINT, original_int)

        # (1) a crash: the guard (per its new F4 fix) returns a plain,
        # non-negative 128+SIGSEGV -- no signal killed the GUARD itself.
        plan = _minimal_front_plan(front_port, upstream_port)
        crash_child = _FakeChild(128 + signal.SIGSEGV)
        result = FASTMLX_LAUNCH._run_front_mode(
            [], plan, popen=lambda argv, child=crash_child, **kwargs: child
        )
        self.assertEqual(result, 139)

        # (2) a requested stop: the guard mirrors SIGTERM onto itself
        # (unchanged, non-crash path), so front mode's own `wait()` sees
        # the OLD negative signal-encoded return.
        class _StoppedChild(_FakeChild):
            def wait(self):
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                return self._wait_return

        plan = _minimal_front_plan(front_port, upstream_port)
        stopped_child = _StoppedChild(-signal.SIGTERM)
        result = FASTMLX_LAUNCH._run_front_mode(
            [], plan, popen=lambda argv, child=stopped_child, **kwargs: child
        )
        self.assertEqual(result, 143)

    # ------------------------------------------------------------------
    # L7: the engine child is started in its OWN session (start_new_session
    # =True), not the launcher's process group -- a terminal Ctrl-C (which
    # sends SIGINT to the whole foreground process group) must reach the
    # engine only ONCE, via this function's own signal forwarding, never a
    # second time directly from the terminal racing the forwarded signal.
    # ------------------------------------------------------------------
    def test_engine_child_starts_in_its_own_session(self):
        front_port = _free_tcp_port()
        upstream_port = _free_tcp_port()
        plan = _minimal_front_plan(front_port, upstream_port)
        popen_kwargs = {}

        def recording_popen(argv, **kwargs):
            popen_kwargs.update(kwargs)
            return _FakeChild(0)

        FASTMLX_LAUNCH._run_front_mode([], plan, popen=recording_popen)
        self.assertTrue(popen_kwargs.get("start_new_session"))

    # ------------------------------------------------------------------
    # End-to-end: a real subprocess launch in front mode actually proxies
    # a request to a stub engine, then a SIGTERM to the launcher stops both
    # the launcher and the engine child within a bound.
    # ------------------------------------------------------------------
    def test_front_mode_e2e_proxies_and_forwards_sigterm(self):
        self._front_mode_e2e_sigterm(graceful_engine=False)

    def test_front_mode_e2e_graceful_engine_stop_exits_143(self):
        # The served engine traps SIGTERM and exits 0 (cycle-102 E8). An
        # operator stop must still exit 128+SIGTERM, not front mode's 1.
        self._front_mode_e2e_sigterm(graceful_engine=True)

    def _front_mode_e2e_sigterm(self, graceful_engine: bool):
        stub_bin = write_script(self.root / "stub-engine.py", STUB_ENGINE_BODY)
        pid_path = self.root / "stub-engine.pid"
        env = dict(os.environ)
        env["STUB_ENGINE_PID_PATH"] = str(pid_path)
        if graceful_engine:
            env["STUB_ENGINE_GRACEFUL"] = "1"

        front_port = _free_tcp_port()
        backend_port = _free_tcp_port()

        argv = [
            sys.executable,
            str(LAUNCH_PATH),
            "serve",
            "--model-path", str(self.model_dir),
            "--quality-cards", str(self.manifest_path),
            "--fit-check-bin", str(self.green_fit_bin),
            "--engine-bin", str(stub_bin),
            "--model-repo", PASS_REPO,
            "--context", "2048",
            "--port", str(backend_port),
            "--front-port", str(front_port),
        ]
        proc = subprocess.Popen(
            argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        try:
            resp = None
            body = None
            deadline = time.time() + 15
            while time.time() < deadline:
                if proc.poll() is not None:
                    self.fail(
                        "launcher exited early: "
                        f"code={proc.returncode} stderr={proc.stderr.read()}"
                    )
                try:
                    conn = http.client.HTTPConnection("127.0.0.1", front_port, timeout=2)
                    conn.request("GET", "/v1/models")
                    resp = conn.getresponse()
                    body = resp.read()
                    conn.close()
                    if resp.status == 200:
                        break
                    # The proxy can come up and answer 502 briefly before the
                    # stub engine has finished binding its own backend port;
                    # keep polling until it reports success or time runs out.
                    resp = None
                    time.sleep(0.1)
                except OSError:
                    time.sleep(0.1)
            self.assertIsNotNone(resp, "front proxy never came up")
            self.assertEqual(resp.status, 200)
            self.assertEqual(json.loads(body), {"stub": True})
            self.assertIsNotNone(resp.getheader("X-FastMLX-Admission"))
            self.assertIsNotNone(resp.getheader("X-FastMLX-Request-Id"))

            deadline = time.time() + 5
            while not pid_path.exists() and time.time() < deadline:
                time.sleep(0.05)
            self.assertTrue(pid_path.exists(), "stub engine never recorded its pid")
            child_pid = int(pid_path.read_text().strip())
            # Guarantees the stub engine dies even if an assertion below
            # fails mid-test -- the outer ``finally`` only ever kills the
            # LAUNCHER's pid (``proc``), never the engine CHILD's.
            self.addCleanup(_kill_pid_if_alive, child_pid)

            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                self.fail("launcher did not exit within 10s of SIGTERM")

            # Without STUB_ENGINE_GRACEFUL the stub installs no SIGTERM
            # handler, so the forwarded SIGTERM kills it by signal (wait()
            # reports -15). With it, the stub exits 0 like the served engine.
            # Either way the requested stop maps to 128+15 = 143: never the
            # raw negative code (which would wrap to 241) and never the 1 a
            # zero exit gives when nobody asked the engine to stop.
            self.assertEqual(
                proc.returncode,
                143,
                "launcher must exit 143 (128+SIGTERM) after a forwarded SIGTERM "
                f"(graceful_engine={graceful_engine})",
            )

            deadline = time.time() + 5
            child_gone = False
            while time.time() < deadline:
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    child_gone = True
                    break
                time.sleep(0.1)
            self.assertTrue(
                child_gone, "engine child process survived the launcher's SIGTERM"
            )
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

    # ------------------------------------------------------------------
    # (a) e2e: SIGKILL the launcher process itself (spawned directly with
    # Popen, never through a shell or `&`) and confirm the orphaned engine
    # is gone within grace+2s. Uses a short grace override (internal env
    # var) so the test does not have to wait out the real 30s default.
    # ------------------------------------------------------------------
    def test_e2e_launcher_sigkill_stops_orphaned_engine(self):
        engine_bin = write_script(self.root / "guard-test-engine.py", GUARD_TEST_ENGINE_BODY)
        pid_path = self.root / "engine.pid"
        env = dict(os.environ)
        env["GUARD_TEST_ENGINE_MODE"] = "sleep"
        env["GUARD_TEST_ENGINE_PID_PATH"] = str(pid_path)
        env["_FASTMLX_ENGINE_LIFELINE_GRACE_SECONDS_INTERNAL"] = "1"

        front_port = _free_tcp_port()
        backend_port = _free_tcp_port()
        argv = [
            sys.executable,
            str(LAUNCH_PATH),
            "serve",
            "--model-path", str(self.model_dir),
            "--quality-cards", str(self.manifest_path),
            "--fit-check-bin", str(self.green_fit_bin),
            "--engine-bin", str(engine_bin),
            "--model-repo", PASS_REPO,
            "--context", "2048",
            "--port", str(backend_port),
            "--front-port", str(front_port),
        ]
        launcher = subprocess.Popen(
            argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        try:
            deadline = time.time() + 10
            while not pid_path.exists() and time.time() < deadline:
                if launcher.poll() is not None:
                    self.fail(
                        "launcher exited early: "
                        f"code={launcher.returncode} stderr={launcher.stderr.read()}"
                    )
                time.sleep(0.05)
            self.assertTrue(pid_path.exists(), "engine never recorded its pid")
            engine_pid = int(pid_path.read_text().strip())
            self.addCleanup(_kill_pid_if_alive, engine_pid)

            os.kill(launcher.pid, signal.SIGKILL)
            launcher.wait(timeout=5)

            deadline = time.time() + 3  # grace override (1s) + 2s
            gone = False
            while time.time() < deadline:
                try:
                    os.kill(engine_pid, 0)
                except ProcessLookupError:
                    gone = True
                    break
                time.sleep(0.05)
            self.assertTrue(gone, "engine survived the launcher's SIGKILL past grace+2s")
        finally:
            if launcher.poll() is None:
                launcher.kill()
                launcher.wait(timeout=5)


    # ------------------------------------------------------------------
    # A SIGKILL of the GUARD (e.g. `pkill -9 -f fastmlx_launch`, which
    # matches the guard's argv too) must not orphan the engine either: the
    # launcher stops whatever is left in the guard's process group, SIGTERM
    # then SIGKILL after the grace. The engine here ignores SIGTERM, so it
    # is only gone if the escalation happens.
    # ------------------------------------------------------------------
    def test_e2e_guard_sigkill_does_not_orphan_the_engine(self):
        engine_bin = write_script(self.root / "guard-test-engine.py", GUARD_TEST_ENGINE_BODY)
        pid_path = self.root / "engine.pid"
        env = dict(os.environ)
        env["GUARD_TEST_ENGINE_MODE"] = "ignore-sigterm"
        env["GUARD_TEST_ENGINE_PID_PATH"] = str(pid_path)
        env["_FASTMLX_ENGINE_LIFELINE_GRACE_SECONDS_INTERNAL"] = "1"

        argv = [
            sys.executable,
            str(LAUNCH_PATH),
            "serve",
            "--model-path", str(self.model_dir),
            "--quality-cards", str(self.manifest_path),
            "--fit-check-bin", str(self.green_fit_bin),
            "--engine-bin", str(engine_bin),
            "--model-repo", PASS_REPO,
            "--context", "2048",
            "--port", str(_free_tcp_port()),
            "--front-port", str(_free_tcp_port()),
        ]
        launcher = subprocess.Popen(
            argv, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        try:
            deadline = time.time() + 10
            while not pid_path.exists() and time.time() < deadline:
                self.assertIsNone(launcher.poll(), "launcher exited early")
                time.sleep(0.05)
            self.assertTrue(pid_path.exists(), "engine never recorded its pid")
            engine_pid = int(pid_path.read_text().strip())
            self.addCleanup(_kill_pid_if_alive, engine_pid)
            children = subprocess.run(
                ["pgrep", "-P", str(launcher.pid)], capture_output=True, text=True
            ).stdout.split()
            self.assertEqual(len(children), 1, children)
            guard_pid = int(children[0])
            self.assertNotEqual(guard_pid, engine_pid)

            os.kill(guard_pid, signal.SIGKILL)

            # grace override (1s) + 2s for the launcher to escalate and exit
            self.assertEqual(launcher.wait(timeout=3), 128 + signal.SIGKILL)
            gone = False
            try:
                os.kill(engine_pid, 0)
            except ProcessLookupError:
                gone = True
            self.assertTrue(gone, "engine survived its guard's SIGKILL")
        finally:
            if launcher.poll() is None:
                launcher.kill()
                launcher.wait(timeout=5)

class EngineLifelineGuardTests(unittest.TestCase):
    """``--internal-engine-guard``: front mode's engine child must not be
    orphaned by a SIGKILLed launcher (a launchd ExitTimeOut or a memory-
    pressure kill neither of which the launcher can catch or clean up
    after). Front mode Popens this guard instead of the engine directly
    (see the module-level guard section above ``_run_front_mode`` in
    ``fastmlx_launch.py``); the guard Popens the engine itself and watches
    a pipe only the launcher's process holds open, so the OS itself signals
    the guard the instant the launcher dies for any reason.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _write_guard_test_engine(self) -> Path:
        return write_script(self.root / "guard-test-engine.py", GUARD_TEST_ENGINE_BODY)

    def test_guard_forwards_a_stop_that_arrives_while_the_engine_is_starting(self):
        # The guard's handlers must be installed BEFORE it spawns the engine.
        # Otherwise a SIGTERM forwarded by the launcher right then meets the
        # default disposition, kills the guard, and orphans the new engine.
        # The sentinel stands in for that default: with handlers installed
        # late, it swallows the signal and nothing reaches the engine.
        received = []

        class FakeEngine:
            def send_signal(self, signum):
                received.append(signum)

            def wait(self):
                return 0

            def poll(self):
                return 0

            def terminate(self):
                pass

            def kill(self):
                pass

        def popen_that_is_interrupted(_argv):
            os.kill(os.getpid(), signal.SIGTERM)
            return FakeEngine()

        watched = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
        saved = {sig: signal.getsignal(sig) for sig in watched}
        read_fd, write_fd = os.pipe()
        try:
            for sig in watched:
                signal.signal(sig, lambda *_: None)
            code = FASTMLX_LAUNCH._engine_lifeline_guard(
                read_fd, 1.0, ["engine"], popen=popen_that_is_interrupted
            )
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)
            os.close(write_fd)
        self.assertEqual(code, 0)
        self.assertEqual(received, [signal.SIGTERM])

    # ------------------------------------------------------------------
    # (b) guard run directly, with a stub engine that ignores SIGTERM:
    # closing the lifeline's write end must still get the engine SIGKILLed
    # within roughly the guard's own (short, test-supplied) grace window.
    # ------------------------------------------------------------------
    def test_guard_sigkills_an_engine_that_ignores_sigterm(self):
        engine_bin = self._write_guard_test_engine()
        pid_path = self.root / "engine.pid"
        env = dict(os.environ)
        env["GUARD_TEST_ENGINE_MODE"] = "ignore-sigterm"
        env["GUARD_TEST_ENGINE_PID_PATH"] = str(pid_path)

        lifeline_read_fd, lifeline_write_fd = os.pipe()
        guard_argv = [
            sys.executable, str(LAUNCH_PATH), "--internal-engine-guard",
            "--lifeline-fd", str(lifeline_read_fd), "--grace", "0.5", "--",
            sys.executable, str(engine_bin),
        ]
        guard = subprocess.Popen(guard_argv, env=env, pass_fds=(lifeline_read_fd,))
        os.close(lifeline_read_fd)
        self.addCleanup(lambda: guard.poll() is None and guard.kill())

        deadline = time.time() + 5
        while not pid_path.exists() and time.time() < deadline:
            time.sleep(0.05)
        self.assertTrue(pid_path.exists(), "stub engine never recorded its pid")
        engine_pid = int(pid_path.read_text().strip())
        self.addCleanup(_kill_pid_if_alive, engine_pid)

        os.close(lifeline_write_fd)  # simulate the launcher dying

        deadline = time.time() + 2
        gone = False
        while time.time() < deadline:
            try:
                os.kill(engine_pid, 0)
            except ProcessLookupError:
                gone = True
                break
            time.sleep(0.05)
        self.assertTrue(gone, "engine (ignoring SIGTERM) survived the guard's grace+SIGKILL")
        guard.wait(timeout=3)

    # ------------------------------------------------------------------
    # (c) exit-status propagation: the guard's own exit mirrors the
    # engine's exactly, whether a plain exit code or a signal kill.
    # ------------------------------------------------------------------
    def test_guard_propagates_plain_engine_exit_codes(self):
        engine_bin = self._write_guard_test_engine()
        for mode, expected in (("0", 0), ("7", 7)):
            with self.subTest(mode=mode):
                env = dict(os.environ)
                env["GUARD_TEST_ENGINE_MODE"] = mode
                lifeline_read_fd, lifeline_write_fd = os.pipe()
                guard_argv = [
                    sys.executable, str(LAUNCH_PATH), "--internal-engine-guard",
                    "--lifeline-fd", str(lifeline_read_fd), "--grace", "5", "--",
                    sys.executable, str(engine_bin),
                ]
                guard = subprocess.Popen(guard_argv, env=env, pass_fds=(lifeline_read_fd,))
                os.close(lifeline_read_fd)
                try:
                    self.assertEqual(guard.wait(timeout=5), expected)
                finally:
                    os.close(lifeline_write_fd)
                    if guard.poll() is None:
                        guard.kill()

    def test_guard_signal_killed_engine_mirrors_the_signal(self):
        engine_bin = self._write_guard_test_engine()
        pid_path = self.root / "engine.pid"
        env = dict(os.environ)
        env["GUARD_TEST_ENGINE_MODE"] = "sleep"
        env["GUARD_TEST_ENGINE_PID_PATH"] = str(pid_path)

        lifeline_read_fd, lifeline_write_fd = os.pipe()
        guard_argv = [
            sys.executable, str(LAUNCH_PATH), "--internal-engine-guard",
            "--lifeline-fd", str(lifeline_read_fd), "--grace", "5", "--",
            sys.executable, str(engine_bin),
        ]
        guard = subprocess.Popen(guard_argv, env=env, pass_fds=(lifeline_read_fd,))
        os.close(lifeline_read_fd)
        try:
            deadline = time.time() + 5
            while not pid_path.exists() and time.time() < deadline:
                time.sleep(0.05)
            self.assertTrue(pid_path.exists(), "engine never recorded its pid")

            # Signalling the GUARD (not the engine) exercises the launcher
            # -> guard -> engine forwarding chain, and then the guard's own
            # exit-status mirroring below.
            os.kill(guard.pid, signal.SIGTERM)
            returncode = guard.wait(timeout=5)
            self.assertEqual(returncode, -signal.SIGTERM)
        finally:
            os.close(lifeline_write_fd)
            if guard.poll() is None:
                guard.kill()

    # ------------------------------------------------------------------
    # (d) a missing engine binary is refused through the guard exactly the
    # way front mode always refused it directly: exit 3, same message.
    # ------------------------------------------------------------------
    def test_guard_missing_engine_binary_exits_3_with_message(self):
        missing_bin = self.root / "does-not-exist-engine"
        lifeline_read_fd, lifeline_write_fd = os.pipe()
        guard_argv = [
            sys.executable, str(LAUNCH_PATH), "--internal-engine-guard",
            "--lifeline-fd", str(lifeline_read_fd), "--grace", "5", "--",
            str(missing_bin),
        ]
        guard = subprocess.Popen(
            guard_argv, pass_fds=(lifeline_read_fd,), stderr=subprocess.PIPE, text=True
        )
        os.close(lifeline_read_fd)
        try:
            _, stderr = guard.communicate(timeout=5)
            self.assertEqual(guard.returncode, 3)
            self.assertIn("front mode could not start the engine", stderr)
        finally:
            os.close(lifeline_write_fd)

    # ------------------------------------------------------------------
    # F2 -- exactly-once forwarding per signum: `pkill -f fastmlx_launch`
    # matches this guard's own argv too (see `_stop_leftover_engine_group`'s
    # docstring), so a real stop can reach the guard's SIGTERM handler
    # twice. The engine must only ever be told to stop once for each of
    # those deliveries, not once per delivery -- a second SIGTERM arriving
    # mid-graceful-shutdown is the path that can abort an orderly release
    # of a large wired-memory allocation.
    # ------------------------------------------------------------------
    def test_guard_forwards_a_repeated_signal_to_the_engine_only_once(self):
        class FakeEngine:
            def __init__(self):
                self.signals_received = []

            def send_signal(self, signum):
                self.signals_received.append(signum)

            def wait(self):
                # Two SIGTERMs land on the guard AFTER the engine exists --
                # e.g. once from `pkill -f fastmlx_launch` and once
                # forwarded by the launcher.
                os.kill(os.getpid(), signal.SIGTERM)
                os.kill(os.getpid(), signal.SIGTERM)
                return 0

            def poll(self):
                return 0

            def terminate(self):
                pass

            def kill(self):
                pass

        engine_holder = []

        def popen_returns_fake_engine(_argv):
            engine = FakeEngine()
            engine_holder.append(engine)
            return engine

        watched = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
        saved = {sig: signal.getsignal(sig) for sig in watched}
        read_fd, write_fd = os.pipe()
        try:
            code = FASTMLX_LAUNCH._engine_lifeline_guard(
                read_fd, 1.0, ["engine"], popen=popen_returns_fake_engine
            )
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)
            os.close(write_fd)
        self.assertEqual(code, 0)
        self.assertEqual(engine_holder[0].signals_received, [signal.SIGTERM])

    # ------------------------------------------------------------------
    # F2 -- the same dedupe must apply to the PRE-SPAWN `pending` drain: a
    # signal received twice before the engine exists must still reach it
    # only once once it does, proving the dedupe state is shared between
    # `_forward` and the drain rather than being a half-applied fix.
    # ------------------------------------------------------------------
    def test_guard_dedupes_a_repeated_signal_received_before_the_engine_exists(self):
        received = []

        class FakeEngine:
            def send_signal(self, signum):
                received.append(signum)

            def wait(self):
                return 0

            def poll(self):
                return 0

            def terminate(self):
                pass

            def kill(self):
                pass

        def popen_delivers_pending_sigterm_twice(_argv):
            # Both signals land on the guard while it is still inside
            # `popen` (i.e. before `engine_holder["engine"]` is set), so
            # both go through the `pending` list, not `_forward`'s
            # already-spawned branch.
            os.kill(os.getpid(), signal.SIGTERM)
            os.kill(os.getpid(), signal.SIGTERM)
            return FakeEngine()

        watched = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
        saved = {sig: signal.getsignal(sig) for sig in watched}
        read_fd, write_fd = os.pipe()
        try:
            code = FASTMLX_LAUNCH._engine_lifeline_guard(
                read_fd, 1.0, ["engine"], popen=popen_delivers_pending_sigterm_twice
            )
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)
            os.close(write_fd)
        self.assertEqual(code, 0)
        self.assertEqual(received, [signal.SIGTERM])

    # ------------------------------------------------------------------
    # F2 -- escalation must survive the per-signum dedupe: SIGINT then
    # SIGTERM are two DIFFERENT signums, so both must still reach the
    # engine, in order -- the fix must never collapse to "a stop was
    # already sent" once any signal has been forwarded.
    # ------------------------------------------------------------------
    def test_guard_still_escalates_a_different_signal(self):
        class FakeEngine:
            def __init__(self):
                self.signals_received = []

            def send_signal(self, signum):
                self.signals_received.append(signum)

            def wait(self):
                os.kill(os.getpid(), signal.SIGINT)
                os.kill(os.getpid(), signal.SIGTERM)
                return 0

            def poll(self):
                return 0

            def terminate(self):
                pass

            def kill(self):
                pass

        engine_holder = []

        def popen_returns_fake_engine(_argv):
            engine = FakeEngine()
            engine_holder.append(engine)
            return engine

        watched = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
        saved = {sig: signal.getsignal(sig) for sig in watched}
        read_fd, write_fd = os.pipe()
        try:
            code = FASTMLX_LAUNCH._engine_lifeline_guard(
                read_fd, 1.0, ["engine"], popen=popen_returns_fake_engine
            )
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)
            os.close(write_fd)
        self.assertEqual(code, 0)
        self.assertEqual(
            engine_holder[0].signals_received, [signal.SIGINT, signal.SIGTERM]
        )

    # ------------------------------------------------------------------
    # F2 CRITICAL TRAP -- the lifeline path (`_watch_lifeline` calling
    # `engine.terminate()`/`kill()` directly, never through `_forward`)
    # must keep stopping an orphaned engine even after a signal was
    # already forwarded to it: dedupe state must live inside `_forward`/
    # the drain, never on the engine object, or a SIGKILLed launcher would
    # stop stopping its engine -- regressing cycle 103's headline fix.
    # ------------------------------------------------------------------
    def test_lifeline_watch_still_stops_the_engine_after_a_signal_was_forwarded(self):
        class FakeEngine:
            def __init__(self):
                self.signals_received = []
                self.terminated = False

            def send_signal(self, signum):
                self.signals_received.append(signum)

            def wait(self):
                # A SIGTERM is forwarded (and deduped) first, then this
                # blocks until the lifeline watcher (running on its own
                # thread) terminates the engine, exactly as an orphaned
                # engine would once the launcher's write end closes.
                os.kill(os.getpid(), signal.SIGTERM)
                deadline = time.monotonic() + 5
                while not self.terminated and time.monotonic() < deadline:
                    time.sleep(0.01)
                return 0

            def poll(self):
                return 0 if self.terminated else None

            def terminate(self):
                self.terminated = True

            def kill(self):
                pass

        engine_holder = []

        def popen_returns_fake_engine(_argv):
            engine = FakeEngine()
            engine_holder.append(engine)
            return engine

        watched = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
        saved = {sig: signal.getsignal(sig) for sig in watched}
        read_fd, write_fd = os.pipe()

        def close_write_end_shortly():
            time.sleep(0.2)
            os.close(write_fd)

        closer = threading.Thread(target=close_write_end_shortly, daemon=True)
        closer.start()
        try:
            code = FASTMLX_LAUNCH._engine_lifeline_guard(
                read_fd, 1.0, ["engine"], popen=popen_returns_fake_engine
            )
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)
            closer.join(timeout=2)
        self.assertEqual(code, 0)
        self.assertTrue(engine_holder[0].terminated, "lifeline watch never terminated the engine")
        self.assertEqual(engine_holder[0].signals_received, [signal.SIGTERM])

    # ------------------------------------------------------------------
    # F4 -- a crash signal (SIGSEGV et al.) must never be mirrored onto
    # this guard: it must instead return the same 128+signum encoding
    # `_run_front_mode` already produces for a signal-killed child, name
    # the signal and that the ENGINE (not the guard) died from it on
    # stderr, and never self-signal (which would write a SECOND, false
    # crash report attributed to this guard's own `python3` process).
    # ------------------------------------------------------------------
    def test_crash_signal_returns_128_plus_signum_without_self_signalling(self):
        class FakeEngine:
            def send_signal(self, signum):
                pass

            def wait(self):
                return -signal.SIGSEGV

            def poll(self):
                return 0

            def terminate(self):
                pass

            def kill(self):
                pass

        read_fd, write_fd = os.pipe()
        stderr = io.StringIO()
        try:
            with contextlib.redirect_stderr(stderr):
                code = FASTMLX_LAUNCH._engine_lifeline_guard(
                    read_fd, 1.0, ["engine"], popen=lambda _argv: FakeEngine()
                )
        finally:
            os.close(write_fd)
        self.assertEqual(code, 128 + signal.SIGSEGV)
        self.assertIn("SIGSEGV", stderr.getvalue())
        self.assertIn("engine", stderr.getvalue().lower())


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

    def test_engine_profile_builtin_gguf_resolves_to_sibling_and_runs(self):
        # SUCCESS path: an engine profile naming fitCheck.bin="builtin:gguf"
        # (no --fit-check-bin at all) resolves to the real, SIBLING
        # fastmlx_gguf_fit.py and actually runs it against a real GGUF pack.
        profile_path = self.root / "gguf-profile.json"
        profile_path.write_text(
            json.dumps(
                {
                    "schema": "fastmlx-engine-profile-v1",
                    "name": "gguf-served-engine",
                    "argv": list(FASTMLX_LAUNCH.BUILT_IN_ENGINE_PROFILE["argv"]),
                    "fitCheck": {"bin": "builtin:gguf"},
                }
            ),
            encoding="utf-8",
        )
        argv = [
            "serve",
            "--model-path", str(self.gguf_dir),
            "--quality-cards", str(self.manifest_path),
            "--engine-bin", str(self.fake_engine_bin),
            "--engine-profile", str(profile_path),
            "--fit-check-arg=--kv-reserve-gib",
            "--fit-check-arg=0",
            "--fit-check-arg=--wired-limit-mib",
            "--fit-check-arg=16384",
            "--context", "2048",
            "--dry-run",
        ]
        with patch.dict(os.environ, self._env_without_stray_fastmlx_vars(), clear=True):
            code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        plan = json.loads(stdout.splitlines()[-1])
        self.assertEqual(plan["fit"]["fields"].get("fit"), "green")
        expected_bin = str(
            (LAUNCH_PATH.parent / "fastmlx_gguf_fit.py").resolve()
        )
        self.assertTrue(GGUF_FIT_CHECK_PATH.samefile(expected_bin))


# ---------------------------------------------------------------------
# 0b/kv-reserve coverage: the "0b. Built-in sizer auto-selection" fallback
# in _select_builtin_fit_check_bin (reached only when no --fit-check-bin,
# no FASTMLX_FIT_CHECK_BIN, and no engine profile fitCheck named a sizer,
# AND the built-in Swift engine (fastmlx-serve) is not on PATH), and the
# --kv-reserve-gib requirement/forwarding/refusal that comes with any
# built-in sizer -- whichever way it resolved (auto-select, an explicit
# builtin: CLI/env value, or an engine profile's own fitCheck.bin). Real
# safetensors/GGUF packs (via build_safetensors_bytes/build_gguf_bytes,
# never a stub) drive the auto-select cases through the launcher's real
# CLI entry point, the same call-site-coverage posture
# GgufFitCheckCallSiteTests above uses, so a regression that breaks the
# real sibling sizer's own argument parsing is still caught here.
# ---------------------------------------------------------------------
class BuiltinAutoSelectAndKvReserveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.assertTrue(
            SAFETENSORS_FIT_CHECK_PATH.is_file(), f"missing {SAFETENSORS_FIT_CHECK_PATH}"
        )
        SAFETENSORS_FIT_CHECK_PATH.chmod(
            SAFETENSORS_FIT_CHECK_PATH.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
        )
        self.assertTrue(
            GGUF_FIT_CHECK_PATH.is_file(), f"missing {GGUF_FIT_CHECK_PATH}"
        )
        GGUF_FIT_CHECK_PATH.chmod(
            GGUF_FIT_CHECK_PATH.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
        )

        # A real, tiny safetensors pack: config.json (so is_model_dir
        # admits it) plus one countable *.safetensors shard (so
        # _pack_has_safetensors is True).
        self.safetensors_dir = self.root / "safetensors-pack"
        self.safetensors_dir.mkdir()
        (self.safetensors_dir / "config.json").write_text("{}", encoding="utf-8")
        blob = build_safetensors_bytes(
            [("t.a", "I8", [1024], zero_tensor_bytes("I8", [1024]))]
        )
        (self.safetensors_dir / "model.safetensors").write_bytes(blob)

        # A real, tiny GGUF pack: one top-level *.gguf file, no config.json
        # -- is_model_dir admits it on the GGUF arm alone.
        self.gguf_dir = self.root / "gguf-pack"
        self.gguf_dir.mkdir()
        tensors = [
            {"name": "token_embd.weight", "dims": [4], "type": 0, "offset": 0}  # F32
        ]
        gguf_blob = build_gguf_bytes(tensors=tensors, data_section=bytes(16))
        (self.gguf_dir / "pack.gguf").write_bytes(gguf_blob)

        # A pack with NEITHER layout: config.json alone (admits is_model_dir)
        # but zero *.safetensors and zero *.gguf entries anywhere -- the
        # auto-select fallback's own "neither layout" refusal.
        self.neither_dir = self.root / "neither-pack"
        self.neither_dir.mkdir()
        (self.neither_dir / "config.json").write_text("{}", encoding="utf-8")

        self.manifest_path = self.root / "quality-guides.json"
        self.manifest_path.write_text(json.dumps(fixture_manifest()), encoding="utf-8")
        self.fake_engine_bin = write_script(self.root / "fake-engine.py", FAKE_ENGINE_BODY)

    @staticmethod
    def _env_without_stray_fastmlx_vars() -> dict:
        # See GgufFitCheckCallSiteTests._env_without_stray_fastmlx_vars: the
        # fit-check subprocess inherits os.environ, and a stray FASTMLX_*
        # value already present in the test process would make the ceiling
        # computed below nondeterministic.
        return {k: v for k, v in os.environ.items() if not k.startswith("FASTMLX_")}

    def _argv(self, model_dir: Path, extra: list, dry_run: bool = True) -> list:
        # --context is always given explicitly: neither real sizer's own
        # attestation line carries a fit_context_ceiling field (unlike the
        # GREEN_FIT_CHECK_BODY stub used elsewhere in this file), so a
        # GREEN-classified launch with no --context would otherwise refuse
        # at the separate "context could not be determined" check.
        argv = [
            "serve",
            "--model-path", str(model_dir),
            "--quality-cards", str(self.manifest_path),
            "--engine-bin", str(self.fake_engine_bin),
            "--context", "2048",
        ] + extra
        if dry_run:
            argv.append("--dry-run")
        return argv

    def run_main(self, argv: list):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                FASTMLX_LAUNCH.main(argv)
        return ctx.exception.code, stdout.getvalue(), stderr.getvalue()

    def _engine_not_on_path(self):
        # The auto-select fallback is only reached when
        # shutil.which(_BUILT_IN_ENGINE_BINARY_NAME) already returned None
        # -- patched directly (rather than mutating PATH) so this can never
        # be defeated by a real fastmlx-serve binary the host happens to
        # have built, the same reliability reason the subprocess.run/
        # shutil.which patches elsewhere in this file exist.
        return patch.object(FASTMLX_LAUNCH.shutil, "which", return_value=None)

    # ------------------------------------------------------------------
    # 1/2: auto-select fallback reached, real fit check actually runs.
    # ------------------------------------------------------------------
    def test_auto_select_resolves_builtin_safetensors_and_runs_it(self):
        argv = self._argv(
            self.safetensors_dir,
            [
                "--kv-reserve-gib", "0",
                "--fit-check-arg=--wired-limit-mib",
                "--fit-check-arg=16384",
            ],
        )
        with self._engine_not_on_path(), patch.object(
            FASTMLX_LAUNCH.subprocess, "run", wraps=FASTMLX_LAUNCH.subprocess.run
        ) as spy, patch.dict(os.environ, self._env_without_stray_fastmlx_vars(), clear=True):
            code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        called_argv = spy.call_args.args[0]
        expected_bin = str(SAFETENSORS_FIT_CHECK_PATH.resolve())
        self.assertEqual(called_argv[0], expected_bin)
        plan = json.loads(stdout.splitlines()[-1])
        self.assertEqual(plan["fit"]["fields"].get("fit"), "green")

    def test_auto_select_resolves_builtin_gguf_and_runs_it(self):
        argv = self._argv(
            self.gguf_dir,
            [
                "--kv-reserve-gib", "0",
                "--fit-check-arg=--wired-limit-mib",
                "--fit-check-arg=16384",
            ],
        )
        with self._engine_not_on_path(), patch.object(
            FASTMLX_LAUNCH.subprocess, "run", wraps=FASTMLX_LAUNCH.subprocess.run
        ) as spy, patch.dict(os.environ, self._env_without_stray_fastmlx_vars(), clear=True):
            code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        called_argv = spy.call_args.args[0]
        expected_bin = str(GGUF_FIT_CHECK_PATH.resolve())
        self.assertEqual(called_argv[0], expected_bin)
        plan = json.loads(stdout.splitlines()[-1])
        self.assertEqual(plan["fit"]["fields"].get("fit"), "green")

    # ------------------------------------------------------------------
    # 3: auto-select refusal when neither layout is present.
    # ------------------------------------------------------------------
    def test_auto_select_refuses_pack_with_neither_layout(self):
        argv = self._argv(self.neither_dir, [])
        with self._engine_not_on_path():
            code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3)
        self.assertIn("fit check could not run", stderr)
        self.assertIn(
            "neither a *.safetensors layout nor a top-level *.gguf file", stderr
        )

    # ------------------------------------------------------------------
    # 4: auto-select is a LAST resort -- an on-PATH fastmlx-serve still
    # wins, so a built-in sizer is never auto-selected out from under it.
    # ------------------------------------------------------------------
    def test_engine_on_path_wins_over_builtin_auto_select(self):
        stub_bin = write_script(self.root / "fastmlx-serve-stub.py", GREEN_FIT_CHECK_BODY)
        argv = self._argv(self.neither_dir, [])
        with patch.object(
            FASTMLX_LAUNCH.shutil, "which", return_value=str(stub_bin)
        ), patch.object(
            FASTMLX_LAUNCH.subprocess, "run", wraps=FASTMLX_LAUNCH.subprocess.run
        ) as spy:
            code, stdout, stderr = self.run_main(argv)
        # No --kv-reserve-gib was given: if a built-in sizer had been
        # auto-selected instead of the on-PATH engine, this would refuse
        # with exit 3 (see test 5 below). A clean exit 0 here proves the
        # on-PATH engine -- not a built-in sizer -- was actually used.
        self.assertEqual(code, 0, stderr)
        called_argv = spy.call_args.args[0]
        self.assertEqual(called_argv[0], str(stub_bin))
        self.assertNotIn(called_argv[0], FASTMLX_LAUNCH._BUILTIN_FIT_CHECK_BIN_PATHS)

    # ------------------------------------------------------------------
    # 5: --kv-reserve-gib is required once a built-in sizer resolves.
    # ------------------------------------------------------------------
    def test_kv_reserve_gib_required_for_auto_selected_builtin(self):
        argv = self._argv(self.safetensors_dir, [])
        with self._engine_not_on_path():
            code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3)
        self.assertIn("requires --kv-reserve-gib", stderr)
        self.assertIn("builtin:safetensors", stderr)

    # ------------------------------------------------------------------
    # 6: --kv-reserve-gib, once given, is forwarded to the fit check.
    # ------------------------------------------------------------------
    def test_kv_reserve_gib_is_forwarded_to_builtin_fit_check_argv(self):
        argv = self._argv(
            self.safetensors_dir,
            [
                "--kv-reserve-gib", "0",
                "--fit-check-arg=--wired-limit-mib",
                "--fit-check-arg=16384",
            ],
        )
        with self._engine_not_on_path(), patch.object(
            FASTMLX_LAUNCH.subprocess, "run", wraps=FASTMLX_LAUNCH.subprocess.run
        ) as spy, patch.dict(os.environ, self._env_without_stray_fastmlx_vars(), clear=True):
            code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        called_argv = spy.call_args.args[0]
        self.assertIn("--kv-reserve-gib", called_argv)
        idx = called_argv.index("--kv-reserve-gib")
        self.assertEqual(called_argv[idx + 1], "0.0")

    # ------------------------------------------------------------------
    # 7: --kv-reserve-gib is refused against a non-built-in fit-check bin.
    # ------------------------------------------------------------------
    def test_kv_reserve_gib_refused_for_non_builtin_fit_check_bin(self):
        green_bin = write_script(self.root / "fit-green-explicit.py", GREEN_FIT_CHECK_BODY)
        argv = self._argv(
            self.safetensors_dir,
            ["--fit-check-bin", str(green_bin), "--kv-reserve-gib", "4"],
        )
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("is not one of this repository's built-in sizers", stderr)

    # ------------------------------------------------------------------
    # 8/9: `builtin:` on the serve front door (CLI flag and env var) must
    # resolve to the real sibling .py sizer, never get exec'd raw.
    # ------------------------------------------------------------------
    def test_cli_fit_check_bin_builtin_safetensors_resolves_to_sibling(self):
        argv = self._argv(
            self.safetensors_dir,
            [
                "--fit-check-bin", "builtin:safetensors",
                "--kv-reserve-gib", "0",
                "--fit-check-arg=--wired-limit-mib",
                "--fit-check-arg=16384",
            ],
        )
        with patch.object(
            FASTMLX_LAUNCH.subprocess, "run", wraps=FASTMLX_LAUNCH.subprocess.run
        ) as spy, patch.dict(os.environ, self._env_without_stray_fastmlx_vars(), clear=True):
            code, stdout, stderr = self.run_main(argv)
        self.assertNotIn("fit check binary not found: builtin:safetensors", stderr)
        self.assertEqual(code, 0, stderr)
        called_argv = spy.call_args.args[0]
        self.assertEqual(called_argv[0], str(SAFETENSORS_FIT_CHECK_PATH.resolve()))

    def test_env_fit_check_bin_builtin_safetensors_resolves_to_sibling(self):
        argv = self._argv(
            self.safetensors_dir,
            [
                "--kv-reserve-gib", "0",
                "--fit-check-arg=--wired-limit-mib",
                "--fit-check-arg=16384",
            ],
        )
        env_patch = dict(self._env_without_stray_fastmlx_vars())
        env_patch["FASTMLX_FIT_CHECK_BIN"] = "builtin:safetensors"
        with patch.object(
            FASTMLX_LAUNCH.subprocess, "run", wraps=FASTMLX_LAUNCH.subprocess.run
        ) as spy, patch.dict(os.environ, env_patch, clear=True):
            code, stdout, stderr = self.run_main(argv)
        self.assertNotIn("fit check binary not found: builtin:safetensors", stderr)
        self.assertEqual(code, 0, stderr)
        called_argv = spy.call_args.args[0]
        self.assertEqual(called_argv[0], str(SAFETENSORS_FIT_CHECK_PATH.resolve()))

    # ------------------------------------------------------------------
    # 10/11: --fit-check-arg can itself satisfy the --kv-reserve-gib
    # requirement, both the two-token and the "="-joined form.
    # ------------------------------------------------------------------
    def test_fit_check_arg_two_token_form_satisfies_kv_reserve_requirement(self):
        argv = self._argv(
            self.safetensors_dir,
            [
                "--fit-check-arg=--kv-reserve-gib",
                "--fit-check-arg=8",
                "--fit-check-arg=--wired-limit-mib",
                "--fit-check-arg=32768",
            ],
        )
        with self._engine_not_on_path(), patch.object(
            FASTMLX_LAUNCH.subprocess, "run", wraps=FASTMLX_LAUNCH.subprocess.run
        ) as spy, patch.dict(os.environ, self._env_without_stray_fastmlx_vars(), clear=True):
            code, stdout, stderr = self.run_main(argv)
        self.assertNotIn("requires --kv-reserve-gib", stderr)
        self.assertEqual(code, 0, stderr)
        called_argv = spy.call_args.args[0]
        occurrences = [
            item for item in called_argv
            if item == "--kv-reserve-gib" or item.startswith("--kv-reserve-gib=")
        ]
        self.assertEqual(len(occurrences), 1)

    def test_fit_check_arg_equals_form_satisfies_kv_reserve_requirement(self):
        argv = self._argv(
            self.safetensors_dir,
            [
                "--fit-check-arg=--kv-reserve-gib=8",
                "--fit-check-arg=--wired-limit-mib",
                "--fit-check-arg=32768",
            ],
        )
        with self._engine_not_on_path(), patch.object(
            FASTMLX_LAUNCH.subprocess, "run", wraps=FASTMLX_LAUNCH.subprocess.run
        ) as spy, patch.dict(os.environ, self._env_without_stray_fastmlx_vars(), clear=True):
            code, stdout, stderr = self.run_main(argv)
        self.assertNotIn("requires --kv-reserve-gib", stderr)
        self.assertEqual(code, 0, stderr)
        called_argv = spy.call_args.args[0]
        occurrences = [
            item for item in called_argv
            if item == "--kv-reserve-gib" or item.startswith("--kv-reserve-gib=")
        ]
        self.assertEqual(len(occurrences), 1)

    # ------------------------------------------------------------------
    # 12: ordering -- the residency-conflict refusal (exit 2) is reported
    # BEFORE the kv-reserve requirement (exit 3) when a launch trips both.
    # ------------------------------------------------------------------
    def test_residency_conflict_reported_before_kv_reserve_requirement(self):
        argv = self._argv(
            self.safetensors_dir,
            ["--fit-check-arg=--residency=expert-stream"],
        )
        # Neither --kv-reserve-gib nor an equivalent --fit-check-arg is
        # given, so this launch ALSO trips the kv-reserve requirement (see
        # test 5) -- proving which refusal actually surfaces first.
        with self._engine_not_on_path():
            code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("resident", stderr)
        self.assertIn("expert-stream", stderr)
        self.assertNotIn("requires --kv-reserve-gib", stderr)


# ---------------------------------------------------------------------
# Residency-aware quality-card matching (fast-mlx-quality-card-v1's
# config.residency): expert streaming can change greedy output versus the
# same pack held resident, so a card measured under one residency must
# never admit a launch of the other. This exercises that rule with an
# inline synthetic fixture card shaped like a real measured NO_GO
# expert-stream card -- no real model, repository, or measurement.
# ---------------------------------------------------------------------
SYNTHETIC_CARD_ID = "example-moe-q4-stream@m3ultra"
SYNTHETIC_CARD_REPO = "example-org/example-moe-gguf"
SYNTHETIC_CARD_REVISION = "0123456789abcdef0123456789abcdef01234567"


def expert_stream_card_manifest() -> dict:
    """A `fast-mlx-quality-card-v1` fixture manifest holding one NO_GO card
    for the `expert-stream` residency, shaped like a real measured card but
    with a synthetic model identity and illustrative-only wording."""
    return {
        "schema": "fast-mlx-quality-card-v1",
        "generatedAt": "2026-01-01T00:00:00Z",
        "cards": [
            {
                "id": SYNTHETIC_CARD_ID,
                "model": {
                    "family": "Example-MoE",
                    "repo": SYNTHETIC_CARD_REPO,
                    "hfPin": SYNTHETIC_CARD_REVISION,
                },
                "config": {
                    "quant": {
                        "bits": 4,
                        "groupSize": None,
                        "mixedBit": True,
                        "note": "fixture: quantized routed experts",
                    },
                    "enhancement": "none",
                    "hardwareClass": "example-hardware",
                    "residency": "expert-stream",
                },
                "verdict": "NO_GO",
                "admission": {
                    "default": False,
                    "optIn": True,
                    "reason": "fixture: output differed on 3 of 8 prompts versus the same pack held in memory",
                },
                "legible": {
                    "tier": "Unquantified",
                    "headline": "fixture: expert-stream residency changed greedy output on 3 of 8 prompts versus the same pack held in memory.",
                    "nextWordDrift": None,
                    "regressionFocus": "fixture: illustrative regression focus text",
                    "example": {
                        "status": "pending",
                        "prompt": None,
                        "referenceOutput": None,
                        "configOutput": None,
                        "note": "fixture: illustrative placeholder",
                    },
                    "benefit": {
                        "fit": None,
                        "speedX": 0.5,
                        "speedXStatus": "fixture: illustrative only, not a measurement",
                    },
                },
                "rawMetrics": {
                    "greedyDivergentPrompts": "3/8",
                    "comparedTokens": 100,
                    "magnitude": "fixture: not collected",
                },
                "provenance": {
                    "source": "fast-mlx-measured",
                    "vendor": None,
                    "method": "fixture: synthetic test data, not a real measurement",
                    "confound": None,
                    "hardware": "fixture",
                    "harnessGitSHA": "0000000000000000000000000000000000000000",
                    "corpusId": "fixture-corpus",
                    "sourceVerdict": None,
                    "measuredAt": "2026-01-01T00:00:00Z",
                },
                "boundary": {
                    "scope": "fixture: serving-admission quality signal for expert-stream residency",
                    "unmeasured": ["fixture: illustrative unmeasured item"],
                },
            }
        ],
    }


def write_expert_stream_card_manifest(root: Path) -> Path:
    """Write `expert_stream_card_manifest()` to `root` and return its path."""
    path = root / "quality-cards.json"
    path.write_text(json.dumps(expert_stream_card_manifest()), encoding="utf-8")
    return path


# ---------------------------------------------------------------------
# provenance.engineBuild / engine profile engineBuild: fastmlx serve reports
# whether the engine build a launch runs matches the one a quality card was
# measured on (docs/quality-card-schema-v1.md "Engine build"). NEVER used to
# filter card admission -- only to classify a status and surface a stderr
# notice / refusal-message addendum.
# ---------------------------------------------------------------------
EB_CARD_COMMIT = "a1" * 20
EB_OTHER_COMMIT = "b2" * 20

EB_NO_GO_CARD_ID = "eb-no-go@test"
EB_NO_GO_REPO = "example/EbNoGoModel"

EB_PASS_CARD_ID = "eb-pass@test"
EB_PASS_REPO = "example/EbPassModel"

EB_UNRECORDED_CARD_ID = "eb-unrecorded@test"
EB_UNRECORDED_REPO = "example/EbUnrecordedModel"


def engine_build_card_manifest() -> dict:
    return {
        "schema": "fast-mlx-quality-card-v1",
        "generatedAt": "2026-01-01T00:00:00Z",
        "cards": [
            {
                "id": EB_NO_GO_CARD_ID,
                "model": {"repo": EB_NO_GO_REPO, "hfPin": "deadbee1"},
                "verdict": "NO_GO",
                "admission": {
                    "default": False,
                    "optIn": True,
                    "reason": "quality-degraded vs reference",
                },
                "legible": {"tier": "Noticeable", "headline": "eb no-go headline"},
                "provenance": {"engineBuild": {"commit": EB_CARD_COMMIT}},
            },
            {
                "id": EB_PASS_CARD_ID,
                "model": {"repo": EB_PASS_REPO, "hfPin": "cafebab1"},
                "verdict": "PASS",
                "admission": {"default": True, "optIn": False, "reason": "measured pass"},
                "legible": {"tier": "Reference", "headline": "eb pass headline"},
                "provenance": {"engineBuild": {"commit": EB_CARD_COMMIT}},
            },
            {
                "id": EB_UNRECORDED_CARD_ID,
                "model": {"repo": EB_UNRECORDED_REPO, "hfPin": "0badc0d1"},
                "verdict": "PASS",
                "admission": {"default": True, "optIn": False, "reason": "measured pass"},
                "legible": {"tier": "Reference", "headline": "eb unrecorded headline"},
                # No provenance at all: engine build "unrecorded".
            },
        ],
    }


def write_engine_build_card_manifest(root: Path) -> Path:
    path = root / "eb-quality-cards.json"
    path.write_text(json.dumps(engine_build_card_manifest()), encoding="utf-8")
    return path


DUAL_BUILD_REPO = "example/DualBuildModel"
DUAL_BUILD_CARD_A_ID = "dual-build-a@test"
DUAL_BUILD_CARD_B_ID = "dual-build-b@test"
DUAL_BUILD_COMMIT_A = "c3" * 20
DUAL_BUILD_COMMIT_B = "d4" * 20


def dual_engine_build_card_manifest() -> dict:
    return {
        "schema": "fast-mlx-quality-card-v1",
        "generatedAt": "2026-01-01T00:00:00Z",
        "cards": [
            {
                "id": DUAL_BUILD_CARD_A_ID,
                "model": {"repo": DUAL_BUILD_REPO, "hfPin": "aaaaaaaa"},
                "verdict": "PASS",
                "admission": {"default": True, "optIn": False, "reason": "pass build A"},
                "legible": {"tier": "Reference", "headline": "Build A pass."},
                "provenance": {"engineBuild": {"commit": DUAL_BUILD_COMMIT_A}},
            },
            {
                "id": DUAL_BUILD_CARD_B_ID,
                "model": {"repo": DUAL_BUILD_REPO, "hfPin": "bbbbbbbb"},
                "verdict": "PASS",
                "admission": {"default": True, "optIn": False, "reason": "pass build B"},
                "legible": {"tier": "Reference", "headline": "Build B pass."},
                "provenance": {"engineBuild": {"commit": DUAL_BUILD_COMMIT_B}},
            },
        ],
    }


def write_dual_engine_build_card_manifest(root: Path) -> Path:
    path = root / "dual-eb-quality-cards.json"
    path.write_text(json.dumps(dual_engine_build_card_manifest()), encoding="utf-8")
    return path


def write_engine_build_profile(
    root: Path, name: str, commit: str = None, binary_sha256: str = None
) -> Path:
    document = {
        "schema": "fastmlx-engine-profile-v1",
        "name": name,
        "argv": list(FASTMLX_LAUNCH.BUILT_IN_ENGINE_PROFILE["argv"]),
    }
    engine_build = {}
    if commit is not None:
        engine_build["commit"] = commit
    if binary_sha256 is not None:
        engine_build["binarySha256"] = binary_sha256
    if engine_build:
        document["engineBuild"] = engine_build
    path = root / f"{name}.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class EngineBuildTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        (self.model_dir / "config.json").write_text("{}", encoding="utf-8")

        self.manifest_path = write_engine_build_card_manifest(self.root)
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

    # (i) NO_GO + mismatch, no opt-in: refuses exit 2 AND the refusal
    # message itself carries the engine-build notice -- engine build never
    # filters a card the way residency does.
    def test_no_go_card_mismatch_refuses_with_notice_in_message(self):
        profile_path = write_engine_build_profile(
            self.root, "mismatch-profile", commit=EB_OTHER_COMMIT
        )
        argv = self.base_args(
            **{
                "--model-repo": EB_NO_GO_REPO,
                "--context": "2048",
                "--engine-profile": str(profile_path),
            }
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn(EB_CARD_COMMIT[:12], stderr)
        self.assertIn(EB_OTHER_COMMIT[:12], stderr)
        self.assertIn("transfer unmeasured", stderr)

    # (ii) default-admit (PASS) card + mismatch: admits; stderr has both
    # 12-char shas; dry-run plan status is "mismatch".
    def test_pass_card_mismatch_admits_with_notice_and_plan_status(self):
        profile_path = write_engine_build_profile(
            self.root, "mismatch-profile", commit=EB_OTHER_COMMIT
        )
        argv = self.base_args(
            **{
                "--model-repo": EB_PASS_REPO,
                "--context": "2048",
                "--engine-profile": str(profile_path),
            }
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        self.assertIn(EB_CARD_COMMIT[:12], stderr)
        self.assertIn(EB_OTHER_COMMIT[:12], stderr)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["engineBuild"]["status"], "mismatch")
        self.assertEqual(plan["engineBuild"]["card"], EB_CARD_COMMIT)
        self.assertEqual(plan["engineBuild"]["launch"], EB_OTHER_COMMIT)

    # (iii) match: no notice on stderr; plan status "match".
    def test_pass_card_match_admits_with_no_notice(self):
        profile_path = write_engine_build_profile(self.root, "match-profile", commit=EB_CARD_COMMIT)
        argv = self.base_args(
            **{
                "--model-repo": EB_PASS_REPO,
                "--context": "2048",
                "--engine-profile": str(profile_path),
            }
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        self.assertNotIn("transfer unmeasured", stderr)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["engineBuild"]["status"], "match")

    # (iv) undeclared: card has a build, launch (built-in profile) does not.
    def test_pass_card_undeclared_admits_with_notice(self):
        argv = self.base_args(**{"--model-repo": EB_PASS_REPO, "--context": "2048"}) + [
            "--dry-run"
        ]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        self.assertIn("undeclared", stderr)
        self.assertIn(EB_CARD_COMMIT[:12], stderr)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["engineBuild"]["status"], "undeclared")
        self.assertIsNone(plan["engineBuild"]["launch"])

    # (v) unrecorded: card has no engineBuild at all -- no notice regardless
    # of the launch's own build.
    def test_unrecorded_card_never_prints_notice(self):
        profile_path = write_engine_build_profile(
            self.root, "some-profile", commit=EB_OTHER_COMMIT
        )
        argv = self.base_args(
            **{
                "--model-repo": EB_UNRECORDED_REPO,
                "--context": "2048",
                "--engine-profile": str(profile_path),
            }
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        self.assertNotIn("transfer unmeasured", stderr)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["engineBuild"]["status"], "unrecorded")

    # (vi) explicit --card-id gets the same status/notice, and never
    # refuses due to build.
    def test_explicit_card_id_gets_same_status_and_never_refuses_on_build(self):
        profile_path = write_engine_build_profile(
            self.root, "mismatch-profile", commit=EB_OTHER_COMMIT
        )
        argv = self.base_args(
            **{
                "--model-repo": EB_PASS_REPO,
                "--card-id": EB_PASS_CARD_ID,
                "--context": "2048",
                "--engine-profile": str(profile_path),
            }
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["engineBuild"]["status"], "mismatch")

    # (vii) invalid profile engineBuild variants: refused exit 3.
    def test_invalid_engine_build_variants_refuse_exit_3(self):
        variants = [
            {"engineBuild": "not-an-object"},
            {"engineBuild": {"commit": EB_CARD_COMMIT, "unknown": "x"}},
            {"engineBuild": {"commit": "TOOSHORTORUPPERCASE"}},
            {"engineBuild": {"commit": EB_CARD_COMMIT.upper()}},
            {"engineBuild": {"binarySha256": "not-64-hex"}},
        ]
        for extra in variants:
            with self.subTest(extra=extra):
                document = {
                    "schema": "fastmlx-engine-profile-v1",
                    "name": "bad-profile",
                    "argv": list(FASTMLX_LAUNCH.BUILT_IN_ENGINE_PROFILE["argv"]),
                }
                document.update(extra)
                profile_path = self.root / "bad-profile.json"
                profile_path.write_text(json.dumps(document), encoding="utf-8")
                argv = self.base_args(
                    **{"--context": "2048", "--engine-profile": str(profile_path)}
                ) + ["--dry-run"]
                code, _, stderr = self.run_main(argv)
                self.assertEqual(code, 3, stderr)
                self.assertIn("engineBuild", stderr)

    # (viii) binarySha256: wrong hash refuses exit 3 EVEN with --force;
    # correct hash admits.
    def test_binary_sha256_mismatch_refuses_even_with_force(self):
        actual_sha256 = FASTMLX_LAUNCH._sha256_file(str(self.fake_engine_bin))
        wrong_sha256 = ("0" if actual_sha256[0] != "0" else "1") + actual_sha256[1:]
        profile_path = write_engine_build_profile(
            self.root, "sha-profile", binary_sha256=wrong_sha256
        )
        argv = self.base_args(
            **{"--context": "2048", "--engine-profile": str(profile_path), "--force": ""}
        )
        argv = [a for a in argv if a != ""] + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3, stderr)
        self.assertIn("binarySha256", stderr)

    def test_binary_sha256_match_admits(self):
        actual_sha256 = FASTMLX_LAUNCH._sha256_file(str(self.fake_engine_bin))
        profile_path = write_engine_build_profile(
            self.root, "sha-profile", binary_sha256=actual_sha256
        )
        argv = self.base_args(
            **{"--context": "2048", "--engine-profile": str(profile_path)}
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)

    # (ix) two cards for the same pack (same repo/residency, different
    # engine builds): the launch whose engineBuild.commit matches one
    # exactly is picked; an undeclared launch refuses exit 3.
    def test_two_cards_same_pack_picks_exact_build_match(self):
        manifest_path = write_dual_engine_build_card_manifest(self.root)
        profile_path = write_engine_build_profile(
            self.root, "dual-a-profile", commit=DUAL_BUILD_COMMIT_A
        )
        argv = self.base_args(
            **{
                "--quality-cards": str(manifest_path),
                "--model-repo": DUAL_BUILD_REPO,
                "--context": "2048",
                "--engine-profile": str(profile_path),
            }
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["card"]["id"], DUAL_BUILD_CARD_A_ID)

    def test_two_cards_same_pack_undeclared_launch_refuses_exit_3(self):
        manifest_path = write_dual_engine_build_card_manifest(self.root)
        argv = self.base_args(
            **{
                "--quality-cards": str(manifest_path),
                "--model-repo": DUAL_BUILD_REPO,
                "--context": "2048",
            }
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3, stderr)
        self.assertIn("cards for this pack", stderr)
        self.assertIn("--card-id", stderr)

    def test_two_cards_same_pack_picks_the_second_listed_build(self):
        # Build B is listed second, so a first-wins lookup would return A.
        cards = dual_engine_build_card_manifest()["cards"]
        card = FASTMLX_LAUNCH.resolve_card(
            cards, DUAL_BUILD_REPO, None, engine_build_commit=DUAL_BUILD_COMMIT_B
        )
        self.assertEqual(card["id"], DUAL_BUILD_CARD_B_ID)

    def test_undeclared_launch_never_picks_the_unrecorded_card_of_several(self):
        cards = dual_engine_build_card_manifest()["cards"]
        del cards[1]["provenance"]["engineBuild"]
        with self.assertRaises(FASTMLX_LAUNCH.LaunchRefusal) as ctx:
            FASTMLX_LAUNCH.resolve_card(cards, DUAL_BUILD_REPO, None)
        self.assertEqual(ctx.exception.exit_code, 3)
        self.assertIn("undeclared", ctx.exception.message)


# ---------------------------------------------------------------------
# hardwareClass: docs/task-inbox/2026-09-22-PREDECLARATION-hardwareclass-
# joins-identity-never-filters.md. hardwareClass JOINS the multi-candidate
# tiebreak (like engineBuild.commit) but must NEVER filter admission (like
# residency does NOT apply here) -- an Ultra card is still the sole,
# reachable, resolved card on an M5 host whenever it is the only candidate
# for the pack. host_hardware_class is an injectable seam (default the
# real function), exactly like _run_front_mode/_wait_for_lifeline_signal's
# ``popen=subprocess.Popen`` seam -- tests always pass it explicitly rather
# than monkeypatching the module attribute, since a keyword default is
# bound once at function-definition time.
# ---------------------------------------------------------------------
SAFETY_ULTRA_REPO = "example/SafetyUltraModel"
SAFETY_ULTRA_CARD_ID = "safety-ultra@test"


def single_ultra_card(verdict: str = "PASS", opt_in: bool = False, default: bool = True) -> dict:
    return {
        "id": SAFETY_ULTRA_CARD_ID,
        "model": {"repo": SAFETY_ULTRA_REPO, "hfPin": "cccccccc"},
        "verdict": verdict,
        "config": {"hardwareClass": "apple-m3-ultra"},
        "admission": {"default": default, "optIn": opt_in, "reason": "measured on Ultra"},
        "legible": {"tier": "Reference" if verdict == "PASS" else "Unquantified",
                    "headline": "Ultra card."},
    }


HWC_REPO = "example/HwClassModel"
HWC_CARD_ULTRA_ID = "hwc-ultra@test"
HWC_CARD_M5_ID = "hwc-m5@test"


def hardware_class_card_manifest() -> dict:
    return {
        "schema": "fast-mlx-quality-card-v1",
        "generatedAt": "2026-01-01T00:00:00Z",
        "cards": [
            {
                "id": HWC_CARD_ULTRA_ID,
                "model": {"repo": HWC_REPO, "hfPin": "aaaaaaaa"},
                "verdict": "PASS",
                "config": {"hardwareClass": "apple-m3-ultra"},
                "admission": {"default": True, "optIn": False, "reason": "pass ultra"},
                "legible": {"tier": "Reference", "headline": "Ultra pass."},
            },
            {
                "id": HWC_CARD_M5_ID,
                "model": {"repo": HWC_REPO, "hfPin": "bbbbbbbb"},
                "verdict": "PASS",
                "config": {"hardwareClass": "apple-m5"},
                "admission": {"default": True, "optIn": False, "reason": "pass m5"},
                "legible": {"tier": "Reference", "headline": "M5 pass."},
            },
        ],
    }


class HardwareClassTieBreakTestCase(unittest.TestCase):
    # Criterion 4 -- THE SAFETY TEST. A single card is the only candidate,
    # so resolve_card must return it UNCHANGED regardless of the host's
    # class: hardwareClass never filters. Asserted on the returned card's
    # id, not merely "no exception", so the mutation this guards against
    # (turning this into a filter) is provably reached.
    def test_single_ultra_card_resolves_unchanged_on_m5_host(self):
        cards = [single_ultra_card()]
        card = FASTMLX_LAUNCH.resolve_card(
            cards, SAFETY_ULTRA_REPO, None, host_hardware_class=lambda: "apple-m5"
        )
        self.assertEqual(card["id"], SAFETY_ULTRA_CARD_ID)

    # The refusal that would be silently lost if hardwareClass filtered: a
    # NO_GO Ultra card must still be the card resolve_card hands back on an
    # M5 host, so the caller's admission gate still sees it and still
    # refuses it.
    def test_single_no_go_ultra_card_still_resolves_on_m5_host(self):
        cards = [single_ultra_card(verdict="NO_GO", opt_in=True, default=False)]
        card = FASTMLX_LAUNCH.resolve_card(
            cards, SAFETY_ULTRA_REPO, None, host_hardware_class=lambda: "apple-m5"
        )
        self.assertEqual(card["id"], SAFETY_ULTRA_CARD_ID)
        self.assertEqual(card["verdict"], "NO_GO")

    # Structural: the non-filter property isn't incidental -- the
    # single-candidate path never even calls the host-class seam.
    def test_single_candidate_path_never_consults_host_hardware_class(self):
        cards = [single_ultra_card()]
        calls = []

        def spy():
            calls.append(True)
            return "apple-m5"

        card = FASTMLX_LAUNCH.resolve_card(
            cards, SAFETY_ULTRA_REPO, None, host_hardware_class=spy
        )
        self.assertEqual(card["id"], SAFETY_ULTRA_CARD_ID)
        self.assertEqual(calls, [])

    # Criterion 5a: two candidates differ ONLY in hardwareClass; the host
    # matches exactly one -> that one is selected.
    def test_two_candidates_differ_only_in_hardware_class_host_match_selects_one(self):
        cards = hardware_class_card_manifest()["cards"]
        card = FASTMLX_LAUNCH.resolve_card(
            cards, HWC_REPO, None, host_hardware_class=lambda: "apple-m5"
        )
        self.assertEqual(card["id"], HWC_CARD_M5_ID)

    def test_two_candidates_differ_only_in_hardware_class_host_match_selects_the_other(self):
        cards = hardware_class_card_manifest()["cards"]
        card = FASTMLX_LAUNCH.resolve_card(
            cards, HWC_REPO, None, host_hardware_class=lambda: "apple-m3-ultra"
        )
        self.assertEqual(card["id"], HWC_CARD_ULTRA_ID)

    # Criterion 5b: an unknown host class (None) never narrows -- falls
    # through to the existing exit-3 refusal, whose message names the
    # hardware classes so the refusal is diagnosable.
    def test_two_candidates_differ_only_in_hardware_class_unknown_host_refuses(self):
        cards = hardware_class_card_manifest()["cards"]
        with self.assertRaises(FASTMLX_LAUNCH.LaunchRefusal) as ctx:
            FASTMLX_LAUNCH.resolve_card(
                cards, HWC_REPO, None, host_hardware_class=lambda: None
            )
        self.assertEqual(ctx.exception.exit_code, 3)
        self.assertIn("apple-m3-ultra", ctx.exception.message)
        self.assertIn("apple-m5", ctx.exception.message)

    # Same refusal when the host class matches NEITHER candidate.
    def test_two_candidates_differ_only_in_hardware_class_host_matches_neither_refuses(self):
        cards = hardware_class_card_manifest()["cards"]
        with self.assertRaises(FASTMLX_LAUNCH.LaunchRefusal) as ctx:
            FASTMLX_LAUNCH.resolve_card(
                cards, HWC_REPO, None, host_hardware_class=lambda: "apple-m1"
            )
        self.assertEqual(ctx.exception.exit_code, 3)
        self.assertIn("apple-m3-ultra", ctx.exception.message)
        self.assertIn("apple-m5", ctx.exception.message)

    # When candidates all share one class (or all carry none), the host
    # seam is never consulted at all -- the existing engine-build tiebreak
    # (dual_engine_build_card_manifest, both cards carrying no
    # hardwareClass) is untouched by this change.
    def test_candidates_sharing_one_hardware_class_never_consult_host(self):
        cards = dual_engine_build_card_manifest()["cards"]
        calls = []

        def spy():
            calls.append(True)
            return "apple-m5"

        card = FASTMLX_LAUNCH.resolve_card(
            cards,
            DUAL_BUILD_REPO,
            None,
            engine_build_commit=DUAL_BUILD_COMMIT_B,
            host_hardware_class=spy,
        )
        self.assertEqual(card["id"], DUAL_BUILD_CARD_B_ID)
        self.assertEqual(calls, [])


# ---------------------------------------------------------------------
# NO_GO fail-closed tiebreak: an exact host-class match must NEVER convert
# a NO_GO refusal into an admission. hardwareClass is a tiebreak WITHIN a
# verdict class (see resolve_card's docstring), never across one -- a
# same-pack PASS measured on THIS host is not "better evidence" than a
# NO_GO also on record for this pack; it is simply evidence about a
# DIFFERENT configuration the launch is not using. docs/task-inbox/
# 2026-09-22-PREDECLARATION-hardwareclass-joins-identity-never-filters.md.
# ---------------------------------------------------------------------
MIXED_VERDICT_REPO = "example/MixedVerdictModel"
MIXED_VERDICT_PASS_M5_ID = "mixed-pass-m5@test"
MIXED_VERDICT_NOGO_ULTRA_ID = "mixed-nogo-ultra@test"
MIXED_VERDICT_NOGO_M5_ID = "mixed-nogo-m5@test"


def mixed_verdict_card(card_id: str, hardware_class: str, verdict: str) -> dict:
    return {
        "id": card_id,
        "model": {"repo": MIXED_VERDICT_REPO, "hfPin": card_id},
        "verdict": verdict,
        "config": {"hardwareClass": hardware_class},
        "admission": {
            "default": verdict != "NO_GO",
            "optIn": verdict == "NO_GO",
            "reason": f"{verdict.lower()} on {hardware_class}",
        },
        "legible": {
            "tier": "Reference" if verdict == "PASS" else "Unquantified",
            "headline": f"{verdict} on {hardware_class}.",
        },
    }


def mixed_pass_m5_no_go_ultra_cards() -> list:
    # PASS@apple-m5, NO_GO@apple-m3-ultra -- the exact shape from the
    # defect report: an apple-m5 host has an exact hardwareClass match
    # against the PASS card, which must NOT outrank the NO_GO sibling.
    return [
        mixed_verdict_card(MIXED_VERDICT_PASS_M5_ID, "apple-m5", "PASS"),
        mixed_verdict_card(MIXED_VERDICT_NOGO_ULTRA_ID, "apple-m3-ultra", "NO_GO"),
    ]


class NoGoFailClosedTieBreakTestCase(unittest.TestCase):
    # T1: the flip is unreachable. Without the fix, an apple-m5 host's
    # exact hardwareClass match against the PASS card silently disarms the
    # published NO_GO. With the fix, NO_GO candidates are narrowed to
    # FIRST, so the single remaining NO_GO candidate is returned outright
    # and the caller's admission gate still refuses the launch.
    def test_no_go_outranks_exact_host_class_match_on_pass_sibling(self):
        cards = mixed_pass_m5_no_go_ultra_cards()
        card = FASTMLX_LAUNCH.resolve_card(
            cards, MIXED_VERDICT_REPO, None, host_hardware_class=lambda: "apple-m5"
        )
        self.assertEqual(card["id"], MIXED_VERDICT_NOGO_ULTRA_ID)
        self.assertEqual(card["verdict"], "NO_GO")
        outcome, _message = FASTMLX_LAUNCH.decide_admission(card, opted_in=False)
        self.assertEqual(outcome, "refuse_quality_flagged")

    # T2: two-arm discrimination -- host class still works when no
    # candidate is NO_GO. MANDATORY alongside T1: deleting the host-class
    # branch entirely would still pass T1 for free (the NO_GO pool narrows
    # to one candidate without ever consulting host class), so this test
    # is what actually proves the tiebreak logic, not just the narrowing,
    # survived the fix.
    def test_host_class_still_selects_between_two_pass_candidates(self):
        cards = hardware_class_card_manifest()["cards"]
        card = FASTMLX_LAUNCH.resolve_card(
            cards, HWC_REPO, None, host_hardware_class=lambda: "apple-m5"
        )
        self.assertEqual(card["id"], HWC_CARD_M5_ID)

    # T3: host class still disambiguates WITHIN the NO_GO pool -- proves
    # subordination (host class demoted to a within-verdict tiebreak), not
    # a bypass of the tiebreak altogether.
    def test_host_class_disambiguates_within_the_no_go_pool(self):
        cards = [
            mixed_verdict_card(MIXED_VERDICT_NOGO_M5_ID, "apple-m5", "NO_GO"),
            mixed_verdict_card(MIXED_VERDICT_NOGO_ULTRA_ID, "apple-m3-ultra", "NO_GO"),
        ]
        card = FASTMLX_LAUNCH.resolve_card(
            cards, MIXED_VERDICT_REPO, None, host_hardware_class=lambda: "apple-m5"
        )
        self.assertEqual(card["id"], MIXED_VERDICT_NOGO_M5_ID)
        self.assertEqual(card["verdict"], "NO_GO")

    # T4: an unknown host class (None) is unchanged by the fix. The NO_GO
    # pool here narrows to a single candidate BEFORE hardwareClass is ever
    # consulted (same as the single-candidate short circuit above it), so
    # the outcome does not depend on whether the host is known.
    def test_unknown_host_class_still_returns_the_sole_no_go_candidate(self):
        cards = mixed_pass_m5_no_go_ultra_cards()
        card = FASTMLX_LAUNCH.resolve_card(
            cards, MIXED_VERDICT_REPO, None, host_hardware_class=lambda: None
        )
        self.assertEqual(card["id"], MIXED_VERDICT_NOGO_ULTRA_ID)
        self.assertEqual(card["verdict"], "NO_GO")


class CardHardwareClassTestCase(unittest.TestCase):
    def test_returns_config_hardware_class(self):
        card = {"config": {"hardwareClass": "apple-m5"}}
        self.assertEqual(FASTMLX_LAUNCH.card_hardware_class(card), "apple-m5")

    def test_none_when_card_is_none(self):
        self.assertIsNone(FASTMLX_LAUNCH.card_hardware_class(None))

    def test_none_when_no_config(self):
        self.assertIsNone(FASTMLX_LAUNCH.card_hardware_class({"id": "x"}))

    def test_none_when_config_not_a_dict(self):
        self.assertIsNone(FASTMLX_LAUNCH.card_hardware_class({"config": "nope"}))

    def test_none_when_empty_string(self):
        self.assertIsNone(
            FASTMLX_LAUNCH.card_hardware_class({"config": {"hardwareClass": ""}})
        )

    def test_none_when_non_string(self):
        self.assertIsNone(
            FASTMLX_LAUNCH.card_hardware_class({"config": {"hardwareClass": 5}})
        )


class HostHardwareClassTestCase(unittest.TestCase):
    """``host_hardware_class`` composes ``sysctl -n
    machdep.cpu.brand_string`` with the SAME normalization
    ``emit_quality_card._hardware_class`` applies, and fails closed to
    ``None`` on any failure -- never a guess, never a hostname, never a
    raised exception.
    """

    # The normalization contract, stated ONCE and shared by both tests below.
    # These are the exact strings the published cards carry, so this table is the
    # contract itself rather than a convenience fixture.
    REPRESENTATIVE_CHIPS = {
        "Apple M3 Ultra": "apple-m3-ultra",
        "Apple M5": "apple-m5",
    }

    @staticmethod
    def _import_emit_quality_card():
        """The emitter module, or ``None`` when it is not importable.

        ``scripts/emit_quality_card.py`` is NOT part of the sanitized public
        projection, so the PROJECTED copy of this test file runs in a tree where
        the module genuinely does not exist. Returning ``None`` (rather than
        letting ``ModuleNotFoundError`` escape) is what lets the agreement test
        below skip only on that tree, while the contract test still asserts real
        behavior everywhere. See
        ``docs/task-inbox/2026-09-22-projected-test-imported-an-unprojected-module.md``.
        """
        scripts_dir = str(LAUNCH_PATH.parent)
        inserted = scripts_dir not in sys.path
        if inserted:
            sys.path.insert(0, scripts_dir)
        try:
            import emit_quality_card
        except ModuleNotFoundError:
            return None
        finally:
            if inserted:
                sys.path.remove(scripts_dir)
        return emit_quality_card

    # Criterion 6, half one: the normalization contract itself. Runs on EVERY
    # tree, including the projected public one -- never skipped, so this can
    # never silently vanish.
    def test_normalization_matches_representative_chips(self):
        for chip, expected in self.REPRESENTATIVE_CHIPS.items():

            def fake_run(argv, chip=chip, **kwargs):
                return subprocess.CompletedProcess(argv, 0, stdout=chip + "\n", stderr="")

            with patch.object(FASTMLX_LAUNCH.subprocess, "run", side_effect=fake_run):
                actual = FASTMLX_LAUNCH.host_hardware_class()
            self.assertEqual(actual, expected)

    # Criterion 6, half two: the ANTI-DRIFT pin. `host_hardware_class` is a
    # deliberate second copy of `emit_quality_card._hardware_class`'s rule, so
    # the two must agree -- pinned against the SAME table above, which is what
    # makes this a real pin rather than a tautology (asserting the emitter
    # against itself would pass however either side drifted).
    def test_emitter_normalization_agrees_with_the_same_contract(self):
        emit_quality_card = self._import_emit_quality_card()
        if emit_quality_card is None:
            self.skipTest(
                "emit_quality_card is not in the sanitized public projection; "
                "the contract itself is asserted by "
                "test_normalization_matches_representative_chips, which never skips"
            )
        for chip, expected in self.REPRESENTATIVE_CHIPS.items():
            self.assertEqual(
                emit_quality_card._hardware_class(chip),
                expected,
                "the emitter's normalization drifted from host_hardware_class's contract",
            )

    def test_nonzero_returncode_fails_closed_to_none(self):
        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="no sysctl")

        with patch.object(FASTMLX_LAUNCH.subprocess, "run", side_effect=fake_run):
            self.assertIsNone(FASTMLX_LAUNCH.host_hardware_class())

    def test_empty_stdout_fails_closed_to_none(self):
        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, stdout="   \n", stderr="")

        with patch.object(FASTMLX_LAUNCH.subprocess, "run", side_effect=fake_run):
            self.assertIsNone(FASTMLX_LAUNCH.host_hardware_class())

    def test_missing_sysctl_fails_closed_to_none_never_raises(self):
        def fake_run(argv, **kwargs):
            raise FileNotFoundError("sysctl not found")

        with patch.object(FASTMLX_LAUNCH.subprocess, "run", side_effect=fake_run):
            self.assertIsNone(FASTMLX_LAUNCH.host_hardware_class())

    def test_timeout_fails_closed_to_none_never_raises(self):
        def fake_run(argv, **kwargs):
            raise subprocess.TimeoutExpired(cmd=argv, timeout=5)

        with patch.object(FASTMLX_LAUNCH.subprocess, "run", side_effect=fake_run):
            self.assertIsNone(FASTMLX_LAUNCH.host_hardware_class())


# ---------------------------------------------------------------------
# Engine build DERIVATION from a release tree's own provenance.json, with
# NO operator-written --engine-profile engineBuild.commit at all -- see
# `derive_engine_build_from_release`. A release-layout engine binary lives
# at <root>/bin/<name>, with a sibling <root>/provenance.json (exactly the
# shape scripts/package-release.sh stages; see write_release_provenance).
# ---------------------------------------------------------------------
def write_release_engine_binary(
    root: Path, name: str = "fastmlx-serve", content: bytes = b"#!/bin/sh\nexit 0\n"
) -> Path:
    bin_dir = root / "bin"
    bin_dir.mkdir(exist_ok=True)
    path = bin_dir / name
    path.write_bytes(content)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def write_release_provenance(
    root: Path,
    *,
    source_commit=None,
    source_dirty=False,
    engine_binary_sha256=None,
    raw_text: str = None,
    omit: bool = False,
) -> Path:
    """Write <root>/provenance.json in the shape
    scripts/package-release.sh emits it. ``raw_text``, when given, is
    written verbatim (malformed-JSON fixture); ``omit`` skips writing the
    file at all (missing-provenance fixture).
    """
    path = root / "provenance.json"
    if omit:
        return path
    if raw_text is not None:
        path.write_text(raw_text, encoding="utf-8")
        return path
    document = {"source_commit": source_commit, "source_dirty": source_dirty}
    if engine_binary_sha256 is not None:
        document["engine_binary_sha256"] = engine_binary_sha256
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class EngineBuildDerivationTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        (self.model_dir / "config.json").write_text("{}", encoding="utf-8")

        self.manifest_path = write_engine_build_card_manifest(self.root)
        self.green_fit_bin = write_script(self.root / "fit-green.py", GREEN_FIT_CHECK_BODY)

    def base_args(self, engine_bin: Path, **overrides) -> list:
        args = {
            "--model-path": str(self.model_dir),
            "--quality-cards": str(self.manifest_path),
            "--fit-check-bin": str(self.green_fit_bin),
            "--engine-bin": str(engine_bin),
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

    # (1) The headline: no --engine-profile engineBuild.commit at all, but a
    # correctly-formed release layout whose provenance.json names the SAME
    # commit the card was measured on -- match, reachable with NO operator
    # action.
    def test_release_layout_derives_match_with_no_operator_action(self):
        engine_bin = write_release_engine_binary(self.root)
        actual_sha256 = FASTMLX_LAUNCH._sha256_file(str(engine_bin))
        write_release_provenance(
            self.root,
            source_commit=EB_CARD_COMMIT,
            source_dirty=False,
            engine_binary_sha256=actual_sha256,
        )
        argv = self.base_args(
            engine_bin, **{"--model-repo": EB_PASS_REPO, "--context": "2048"}
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        self.assertNotIn("transfer unmeasured", stderr)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["engineBuild"]["status"], "match")
        self.assertEqual(plan["engineBuild"]["launch"], EB_CARD_COMMIT)

    # (2) source_dirty: true -> undeclared, notice printed.
    def test_dirty_source_stays_undeclared(self):
        engine_bin = write_release_engine_binary(self.root)
        actual_sha256 = FASTMLX_LAUNCH._sha256_file(str(engine_bin))
        write_release_provenance(
            self.root,
            source_commit=EB_CARD_COMMIT,
            source_dirty=True,
            engine_binary_sha256=actual_sha256,
        )
        argv = self.base_args(
            engine_bin, **{"--model-repo": EB_PASS_REPO, "--context": "2048"}
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        self.assertIn("transfer unmeasured", stderr)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["engineBuild"]["status"], "undeclared")
        self.assertIsNone(plan["engineBuild"]["launch"])

    # (2b) source_dirty is checked with `is not False`, so anything that is
    # not literally the bool False fails closed the same way True does --
    # including a truthy/falsy non-bool and an absent key. Covered
    # explicitly because the identity comparison is the ONLY thing standing
    # between "unexpected shape" and "treated as a clean build": `== False`
    # would accept 0, and `not source_dirty` would accept a missing key.
    def test_non_bool_or_absent_source_dirty_stays_undeclared(self):
        engine_bin = write_release_engine_binary(self.root)
        actual_sha256 = FASTMLX_LAUNCH._sha256_file(str(engine_bin))
        documents = {
            "zero": {
                "source_commit": EB_CARD_COMMIT,
                "source_dirty": 0,
                "engine_binary_sha256": actual_sha256,
            },
            "string-false": {
                "source_commit": EB_CARD_COMMIT,
                "source_dirty": "false",
                "engine_binary_sha256": actual_sha256,
            },
            "null": {
                "source_commit": EB_CARD_COMMIT,
                "source_dirty": None,
                "engine_binary_sha256": actual_sha256,
            },
            "absent": {
                "source_commit": EB_CARD_COMMIT,
                "engine_binary_sha256": actual_sha256,
            },
        }
        for label, document in documents.items():
            with self.subTest(source_dirty=label):
                write_release_provenance(
                    self.root, raw_text=json.dumps(document)
                )
                argv = self.base_args(
                    engine_bin, **{"--model-repo": EB_PASS_REPO, "--context": "2048"}
                ) + ["--dry-run"]
                code, stdout, stderr = self.run_main(argv)
                self.assertEqual(code, 0, stderr)
                plan = self.last_json_line(stdout)
                self.assertEqual(plan["engineBuild"]["status"], "undeclared")

    # (3) engine_binary_sha256 present but wrong -> undeclared, and the run
    # is NOT refused.
    def test_binary_sha256_mismatch_stays_undeclared_and_does_not_refuse(self):
        engine_bin = write_release_engine_binary(self.root)
        actual_sha256 = FASTMLX_LAUNCH._sha256_file(str(engine_bin))
        wrong_sha256 = ("0" if actual_sha256[0] != "0" else "1") + actual_sha256[1:]
        write_release_provenance(
            self.root,
            source_commit=EB_CARD_COMMIT,
            source_dirty=False,
            engine_binary_sha256=wrong_sha256,
        )
        argv = self.base_args(
            engine_bin, **{"--model-repo": EB_PASS_REPO, "--context": "2048"}
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        self.assertIn("transfer unmeasured", stderr)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["engineBuild"]["status"], "undeclared")

    # (4) provenance.json absent -> undeclared (today's behaviour, unchanged).
    def test_missing_provenance_stays_undeclared(self):
        engine_bin = write_release_engine_binary(self.root)
        write_release_provenance(self.root, omit=True)
        argv = self.base_args(
            engine_bin, **{"--model-repo": EB_PASS_REPO, "--context": "2048"}
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["engineBuild"]["status"], "undeclared")

    # (5) provenance.json malformed JSON -> undeclared, no traceback, no
    # refusal.
    def test_malformed_provenance_json_stays_undeclared(self):
        engine_bin = write_release_engine_binary(self.root)
        write_release_provenance(self.root, raw_text="{not-json")
        argv = self.base_args(
            engine_bin, **{"--model-repo": EB_PASS_REPO, "--context": "2048"}
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["engineBuild"]["status"], "undeclared")

    # (6) source_commit missing / not 40-hex -> undeclared.
    def test_source_commit_missing_or_malformed_stays_undeclared(self):
        engine_bin = write_release_engine_binary(self.root)
        actual_sha256 = FASTMLX_LAUNCH._sha256_file(str(engine_bin))
        for source_commit in (None, "TOOSHORTORUPPERCASE"):
            with self.subTest(source_commit=source_commit):
                write_release_provenance(
                    self.root,
                    source_commit=source_commit,
                    source_dirty=False,
                    engine_binary_sha256=actual_sha256,
                )
                argv = self.base_args(
                    engine_bin, **{"--model-repo": EB_PASS_REPO, "--context": "2048"}
                ) + ["--dry-run"]
                code, stdout, stderr = self.run_main(argv)
                self.assertEqual(code, 0, stderr)
                plan = self.last_json_line(stdout)
                self.assertEqual(plan["engineBuild"]["status"], "undeclared")

    # (7) engine_binary_sha256 missing -> undeclared.
    def test_missing_binary_sha256_stays_undeclared(self):
        engine_bin = write_release_engine_binary(self.root)
        write_release_provenance(
            self.root,
            source_commit=EB_CARD_COMMIT,
            source_dirty=False,
            engine_binary_sha256=None,
        )
        argv = self.base_args(
            engine_bin, **{"--model-repo": EB_PASS_REPO, "--context": "2048"}
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["engineBuild"]["status"], "undeclared")

    # (8) Precedence: an operator-declared engineBuild.commit in the engine
    # profile wins over a well-formed, sha-verified release provenance.json
    # that names a DIFFERENT commit.
    def test_profile_declared_commit_wins_over_derivation(self):
        engine_bin = write_release_engine_binary(self.root)
        actual_sha256 = FASTMLX_LAUNCH._sha256_file(str(engine_bin))
        write_release_provenance(
            self.root,
            source_commit=EB_CARD_COMMIT,
            source_dirty=False,
            engine_binary_sha256=actual_sha256,
        )
        profile_path = write_engine_build_profile(
            self.root, "declared-profile", commit=EB_OTHER_COMMIT
        )
        argv = self.base_args(
            engine_bin,
            **{
                "--model-repo": EB_PASS_REPO,
                "--context": "2048",
                "--engine-profile": str(profile_path),
            },
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["engineBuild"]["launch"], EB_OTHER_COMMIT)
        self.assertEqual(plan["engineBuild"]["status"], "mismatch")

    # (9) The engine binary is NOT inside a `bin/` parent directory -> the
    # layout guard alone forces undeclared, even with a well-formed sibling
    # provenance.json right next to it.
    def test_engine_binary_outside_bin_dir_stays_undeclared(self):
        engine_bin = self.root / "fastmlx-serve"
        engine_bin.write_bytes(b"#!/bin/sh\nexit 0\n")
        engine_bin.chmod(engine_bin.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        actual_sha256 = FASTMLX_LAUNCH._sha256_file(str(engine_bin))
        write_release_provenance(
            self.root,
            source_commit=EB_CARD_COMMIT,
            source_dirty=False,
            engine_binary_sha256=actual_sha256,
        )
        argv = self.base_args(
            engine_bin, **{"--model-repo": EB_PASS_REPO, "--context": "2048"}
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["engineBuild"]["status"], "undeclared")

    # (10) A card whose commit differs from a successfully DERIVED commit ->
    # mismatch, with the notice naming both short shas.
    def test_derived_commit_mismatch_names_both_short_shas(self):
        engine_bin = write_release_engine_binary(self.root)
        actual_sha256 = FASTMLX_LAUNCH._sha256_file(str(engine_bin))
        write_release_provenance(
            self.root,
            source_commit=EB_OTHER_COMMIT,
            source_dirty=False,
            engine_binary_sha256=actual_sha256,
        )
        argv = self.base_args(
            engine_bin, **{"--model-repo": EB_PASS_REPO, "--context": "2048"}
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        self.assertIn(EB_CARD_COMMIT[:12], stderr)
        self.assertIn(EB_OTHER_COMMIT[:12], stderr)
        self.assertIn("transfer unmeasured", stderr)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["engineBuild"]["status"], "mismatch")
        self.assertEqual(plan["engineBuild"]["launch"], EB_OTHER_COMMIT)


# Part 1 (scripts/package-release.sh's new provenance.json keys:
# engine_binary_sha256 / capacity_binary_sha256) is covered by the real
# release-packaging harness in scripts/tests/test_release_package.py
# (test_provenance_binary_sha256_fields_match_the_staged_binaries), which
# builds an actual tarball and asserts each recorded digest equals the
# real sha256 of the staged binary bytes -- not re-covered here.


class ResidencyTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.quality_cards_path = write_expert_stream_card_manifest(self.root)

        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        (self.model_dir / "config.json").write_text("{}", encoding="utf-8")

        self.green_fit_bin = write_script(self.root / "fit-green.py", GREEN_FIT_CHECK_BODY)
        self.fake_engine_bin = write_script(self.root / "fake-engine.py", FAKE_ENGINE_BODY)

        self.streaming_profile_path = self.root / "streaming-profile.json"
        self.streaming_profile_path.write_text(
            json.dumps(
                {
                    "schema": "fastmlx-engine-profile-v1",
                    "name": "streaming-engine",
                    "argv": [
                        "{engine_bin}",
                        "--model",
                        "{model_id}",
                        "--model-path",
                        "{model_path}",
                        "--host",
                        "{host}",
                        "--port",
                        "{port}",
                        "--context",
                        "{context}",
                    ],
                    "residencyArgs": {"expert-stream": ["--ssd-streaming"]},
                }
            ),
            encoding="utf-8",
        )

    def base_args(self, **overrides) -> list:
        args = {
            "--model-path": str(self.model_dir),
            "--quality-cards": str(self.quality_cards_path),
            "--fit-check-bin": str(self.green_fit_bin),
            "--engine-bin": str(self.fake_engine_bin),
            "--engine-profile": str(self.streaming_profile_path),
            "--model-repo": SYNTHETIC_CARD_REPO,
            "--model-revision": SYNTHETIC_CARD_REVISION,
            "--context": "2048",
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

    # (i) expert-stream, no opt-in: refused, names the card id and tier.
    def test_expert_stream_without_opt_in_is_refused(self):
        argv = self.base_args(**{"--residency": "expert-stream"}) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn(SYNTHETIC_CARD_ID, stderr)
        self.assertIn("Unquantified", stderr)

    # (ii) expert-stream + opt-in: admits, streaming argv appended, residency recorded.
    def test_expert_stream_with_opt_in_admits_and_appends_streaming_argv(self):
        argv = self.base_args(
            **{"--residency": "expert-stream", "--accept-quality": SYNTHETIC_CARD_ID}
        ) + ["--dry-run"]
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        plan = self.last_json_line(stdout)
        self.assertEqual(plan["admission"], "admit_with_quality_flag")
        self.assertIn("--ssd-streaming", plan["argv"])
        self.assertEqual(plan["residency"], "expert-stream")

    # (iii) resident: admits unmeasured (the streaming-only card is invisible
    # to a resident lookup), no streaming argv appended.
    def test_resident_launch_admits_unmeasured_with_no_card(self):
        argv = self.base_args(**{"--residency": "resident"}) + ["--dry-run"]
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        plan = self.last_json_line(stdout)
        self.assertIsNone(plan["card"])
        self.assertEqual(plan["admission"], "admit_unmeasured")
        self.assertEqual(plan["residency"], "resident")
        self.assertNotIn("--ssd-streaming", plan["argv"])

    # (iv) resident + a streaming flag as passthrough: refused (the operator
    # cannot bypass the residency gate this way).
    def test_resident_launch_with_streaming_passthrough_is_refused(self):
        argv = self.base_args(**{"--residency": "resident"}) + [
            "--dry-run",
            "--",
            "--ssd-streaming",
        ]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("--residency expert-stream", stderr)

    # (v) expert-stream with the built-in profile: refused, exit 3 (the
    # built-in engine cannot stream).
    def test_expert_stream_with_built_in_profile_is_refused_exit_3(self):
        argv = self.base_args(
            **{"--residency": "expert-stream", "--engine-profile": None}
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3)
        self.assertIn("cannot stream", stderr)

    # (vi) a fit-check-arg residency mismatch is refused.
    def test_fit_check_arg_residency_mismatch_is_refused(self):
        argv = self.base_args(
            **{
                "--residency": "resident",
                "--fit-check-arg": "--residency=expert-stream",
            }
        ) + ["--dry-run"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("resident", stderr)
        self.assertIn("expert-stream", stderr)

    # (vii) a manifest holding a resident PASS card and a streaming NO_GO
    # card for the SAME repo resolves by residency with no ambiguity error.
    def test_resident_and_streaming_cards_for_same_repo_resolve_by_residency(self):
        repo = "example/DualResidencyModel"
        manifest = {
            "schema": "fast-mlx-quality-card-v1",
            "generatedAt": "2026-01-01T00:00:00Z",
            "cards": [
                {
                    "id": "dual-resident@test",
                    "model": {"repo": repo, "hfPin": "aaaaaaaa"},
                    "verdict": "PASS",
                    "config": {"residency": "resident"},
                    "admission": {
                        "default": True,
                        "optIn": False,
                        "reason": "measured pass",
                    },
                    "legible": {"tier": "Reference", "headline": "Resident pass."},
                },
                {
                    "id": "dual-streaming@test",
                    "model": {"repo": repo, "hfPin": "bbbbbbbb"},
                    "verdict": "NO_GO",
                    "config": {"residency": "expert-stream"},
                    "admission": {
                        "default": False,
                        "optIn": True,
                        "reason": "streaming drift",
                    },
                    "legible": {"tier": "Unquantified", "headline": "Streaming drift."},
                },
            ],
        }
        manifest_path = self.root / "dual-manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        resident_argv = self.base_args(
            **{
                "--quality-cards": str(manifest_path),
                "--model-repo": repo,
                "--model-revision": None,
                "--residency": "resident",
            }
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(resident_argv)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(self.last_json_line(stdout)["card"]["id"], "dual-resident@test")

        streaming_argv = self.base_args(
            **{
                "--quality-cards": str(manifest_path),
                "--model-repo": repo,
                "--model-revision": None,
                "--residency": "expert-stream",
                "--accept-quality": "dual-streaming@test",
            }
        ) + ["--dry-run"]
        code, stdout, stderr = self.run_main(streaming_argv)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(self.last_json_line(stdout)["card"]["id"], "dual-streaming@test")

    # (viii) a card with an unrecognized residency never matches, at either
    # launch residency.
    def test_unrecognized_card_residency_never_matches(self):
        cards = [{"id": "bogus@test", "model": {"repo": "example/BogusModel"}, "config": {"residency": "bogus"}}]
        self.assertIsNone(
            FASTMLX_LAUNCH.resolve_card(cards, "example/BogusModel", None, residency="resident")
        )
        self.assertIsNone(
            FASTMLX_LAUNCH.resolve_card(
                cards, "example/BogusModel", None, residency="expert-stream"
            )
        )


REPO_ROOT = LAUNCH_PATH.parents[1]
EXAMPLE_ENGINE_PROFILES_DIR = REPO_ROOT / "examples" / "engine-profiles"
README_PATH = REPO_ROOT / "README.md"

# The exact set of example profiles this repo ships. A glob-count assertion below pins this to
# exactly these two files, so the loader test cannot pass vacuously on an empty/missing directory.
EXPECTED_EXAMPLE_ENGINE_PROFILE_NAMES = (
    "served-engine-safetensors-ngram.json",
    "served-engine-safetensors.json",
)

# Hard public-safety rule: no shipped example may name a specific served engine product, a
# private host, or a machine-local path. Each marker is assembled by concatenation, never written
# as a literal: this test file is itself projected, and a literal would trip the public
# validator's own private-marker scan (the same convention as validate_public_repository.py).
_FORBIDDEN_PUBLIC_SAFETY_SUBSTRINGS = (
    "mlx" + "-serve",
    "om" + "lx",
    "MTP" + "LX",
    "192" + ".168.",
    "llm" + "bench",
    "/" + "Users/",
)


def _find_readme_engine_profile_json_block(readme_text: str) -> dict:
    """Return the parsed JSON of the README's engine-profile example fence -- the one whose body
    contains a `"schema": "fastmlx-engine-profile-v1"` key, not any other fenced ```json block."""
    import re

    for match in re.finditer(r"```json\n(.*?)\n```", readme_text, re.DOTALL):
        block = match.group(1)
        if '"schema": "fastmlx-engine-profile-v1"' in block:
            return json.loads(block)
    raise AssertionError(
        "README.md has no fenced ```json block containing "
        '"schema": "fastmlx-engine-profile-v1"'
    )


class EngineProfileExampleFilesTests(unittest.TestCase):
    """Acceptance: the public-safe example engine profiles this repo ships under
    examples/engine-profiles/ load through the REAL load_engine_profile(), name the real sizer,
    stay in sync with the README's own worked example, and carry no forbidden internal-engine or
    machine-local string."""

    def test_example_profiles_load_and_glob_finds_exactly_the_two_expected_files(self):
        paths = sorted(EXAMPLE_ENGINE_PROFILES_DIR.glob("*.json"))
        names = tuple(path.name for path in paths)
        # Pinning the exact expected set (not just a count) means this test cannot pass
        # vacuously on an empty directory, and fails loudly if a file is renamed or an extra one
        # is added without updating this test.
        self.assertEqual(names, EXPECTED_EXAMPLE_ENGINE_PROFILE_NAMES)

        expected_sizer = str((LAUNCH_PATH.parent / "fastmlx_safetensors_fit.py").resolve())
        for path in paths:
            profile, is_built_in = FASTMLX_LAUNCH.load_engine_profile(str(path))
            self.assertFalse(is_built_in, f"{path.name} must not resolve to the built-in profile")
            self.assertIsNotNone(profile["fitCheck"], f"{path.name} must declare a fitCheck")
            self.assertEqual(
                profile["fitCheck"]["bin"],
                expected_sizer,
                f"{path.name} fitCheck.bin must resolve to this repo's real safetensors sizer",
            )
            self.assertIn(
                "{model_path}", profile["argv"], f"{path.name} argv must place the model path"
            )
            self.assertIn(
                "{context}", profile["argv"], f"{path.name} argv must place the context"
            )

    def test_readme_worked_example_equals_the_ngram_example_file(self):
        # The README text says "This exact file ships as ...ngram.json" -- so the WHOLE parsed
        # document (schema, name, argv, fitCheck) must match, not just argv; a drift in `name` or
        # `fitCheck` would make that sentence false while an argv-only check stayed green.
        readme_doc = _find_readme_engine_profile_json_block(
            README_PATH.read_text(encoding="utf-8")
        )
        ngram_doc = json.loads(
            (EXAMPLE_ENGINE_PROFILES_DIR / "served-engine-safetensors-ngram.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            readme_doc,
            ngram_doc,
            "README's worked engine-profile example has drifted from "
            "examples/engine-profiles/served-engine-safetensors-ngram.json -- the README claims "
            '"This exact file ships as..." so the whole document must match, not just its argv',
        )

    def test_sibling_profile_equals_the_ngram_profile_minus_the_side_file(self):
        # served-engine-safetensors.json is documented as "identical argv, no --mmap-side-file in
        # its fitCheck.args" -- pin that relationship structurally (not just by eyeball) so the two
        # files can never silently diverge in `argv`, `schema`, or the surviving `--kv-reserve-gib`
        # pair.
        ngram_doc = json.loads(
            (EXAMPLE_ENGINE_PROFILES_DIR / "served-engine-safetensors-ngram.json").read_text(
                encoding="utf-8"
            )
        )
        sibling_doc = json.loads(
            (EXAMPLE_ENGINE_PROFILES_DIR / "served-engine-safetensors.json").read_text(
                encoding="utf-8"
            )
        )

        ngram_args = list(ngram_doc["fitCheck"]["args"])
        side_file_index = ngram_args.index("--mmap-side-file")
        del ngram_args[side_file_index : side_file_index + 2]

        expected_sibling = dict(ngram_doc)
        expected_sibling["name"] = sibling_doc["name"]
        expected_sibling["fitCheck"] = dict(ngram_doc["fitCheck"], args=ngram_args)

        # The exclusions above must actually exclude something real, or this test would pass
        # vacuously even if the two files were identical.
        self.assertNotEqual(sibling_doc["name"], ngram_doc["name"])
        self.assertIn("--mmap-side-file", ngram_doc["fitCheck"]["args"])
        self.assertNotIn("--mmap-side-file", sibling_doc["fitCheck"]["args"])

        self.assertEqual(sibling_doc, expected_sibling)

    def test_example_profiles_contain_no_forbidden_public_safety_strings(self):
        paths = sorted(EXAMPLE_ENGINE_PROFILES_DIR.glob("*.json"))
        self.assertTrue(paths, "expected at least one example engine profile to scan")
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for forbidden in _FORBIDDEN_PUBLIC_SAFETY_SUBSTRINGS:
                self.assertNotIn(
                    forbidden,
                    text,
                    f"{path.name} must not contain the forbidden string {forbidden!r}",
                )


# ---------------------------------------------------------------------
# A launch sized by a built-in sizer needs TWO flags a built-in-engine
# launch does not: --kv-reserve-gib and --context. Until this class
# landed, an operator discovered them ONE REFUSAL AT A TIME -- and the
# --context one only AFTER paying for a full fit-check run, since it was
# enforced from the fit check's (absent) context ceiling. Both are now
# collected and reported in a single up-front refusal. See
# docs/task-inbox/2026-09-21-builtin-sizer-yields-no-context-ceiling.md
# option (c).
# ---------------------------------------------------------------------
class BuiltinSizerCombinedRequirementRefusalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

        self.model_dir = self.root / "safetensors-pack"
        self.model_dir.mkdir()
        (self.model_dir / "config.json").write_text("{}", encoding="utf-8")
        blob = build_safetensors_bytes(
            [("t.a", "I8", [1024], zero_tensor_bytes("I8", [1024]))]
        )
        (self.model_dir / "model.safetensors").write_bytes(blob)

        self.manifest_path = self.root / "quality-guides.json"
        self.manifest_path.write_text(json.dumps(fixture_manifest()), encoding="utf-8")
        self.fake_engine_bin = write_script(self.root / "fake-engine.py", FAKE_ENGINE_BODY)

    def run_main(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                FASTMLX_LAUNCH.main(argv)
        return ctx.exception.code, stdout.getvalue(), stderr.getvalue()

    def _argv(self, extra):
        return [
            "serve",
            "--model-path", str(self.model_dir),
            "--quality-cards", str(self.manifest_path),
            "--engine-bin", str(self.fake_engine_bin),
            "--fit-check-bin", "builtin:safetensors",
            "--dry-run",
        ] + extra

    def test_both_missing_requirements_named_in_one_refusal(self):
        code, _, stderr = self.run_main(self._argv([]))
        self.assertEqual(code, 3, stderr)
        self.assertIn("--kv-reserve-gib", stderr)
        self.assertIn("--context", stderr)
        # It must name which sizer, as the reserve-only refusal always did.
        self.assertIn("builtin:safetensors", stderr)

    def test_missing_context_alone_refuses_before_running_the_fit_check(self):
        # The reserve IS supplied, so the only thing missing is --context.
        # This must still refuse UP FRONT rather than after a fit check: a
        # built-in sizer never emits a ceiling, so running it first only
        # burns a fit check the operator was always going to have to redo.
        code, _, stderr = self.run_main(self._argv(["--kv-reserve-gib", "8"]))
        self.assertEqual(code, 3, stderr)
        self.assertIn("--context", stderr)
        # The OLD late check ran only after the fit check and said this;
        # seeing it would mean the refusal is still the post-fit-check one.
        self.assertNotIn("context could not be determined", stderr)

    def test_missing_reserve_alone_still_refuses_naming_the_reserve(self):
        code, _, stderr = self.run_main(self._argv(["--context", "2048"]))
        self.assertEqual(code, 3, stderr)
        self.assertIn("--kv-reserve-gib", stderr)

    def test_both_supplied_passes_the_requirement_check(self):
        # Positive control: with both flags the launch gets PAST this
        # refusal. Without it, the three tests above would pass even if the
        # check refused unconditionally.
        code, _, stderr = self.run_main(
            self._argv(["--kv-reserve-gib", "8", "--context", "2048"])
        )
        self.assertNotIn("which requires", stderr)
        self.assertNotIn("context could not be determined", stderr)


# ---------------------------------------------------------------------
# ANTI-DRIFT. The up-front --context requirement above rests on a fact
# about a DIFFERENT program: this repository's built-in sizers answer only
# "does this pack fit at the KV reserve you named" and emit no context
# ceiling. If that ever stops being true -- option (b) of the same record
# proposes exactly that -- the up-front refusal becomes WRONG: it would
# refuse a launch whose context could in fact have been derived.
# ---------------------------------------------------------------------
class BuiltinSizersEmitNoContextCeilingTests(unittest.TestCase):
    """Pins the assumption the up-front --context refusal depends on.

    This is a DECIDED invariant, not a pending gap. Emitting a context
    ceiling from a built-in sizer was evaluated and REJECTED in cycle 128
    (2026-09-21); see
    docs/task-inbox/2026-09-21-DECISION-builtin-sizer-context-ceiling-REJECTED.md.
    In short: context is not a variable in these sizers' arithmetic at all
    (weight_bytes + kv_reserve_bytes), so "inverting" it yields bytes, not
    tokens; producing tokens would require porting the seven-way Swift KV
    geometry model into Python, where a single formula is already measured
    wrong for 5 of 14 catalog models -- including our own production
    qwen4_exp. Nothing revalidates a derived context before execv, so a
    plausible-but-wrong ceiling is a runtime OOM rather than a clean refusal.

    If this test fails, a built-in sizer has started emitting a context
    ceiling. Do NOT adjust this test to match, and do NOT treat the failure
    as permission to proceed -- reopen the decision record first. Only if
    that record is superseded should you rework the up-front --context
    requirement in scripts/fastmlx_launch.py (search for
    _builtin_sizer_missing_requirements) so `fastmlx serve` derives the
    context from the ceiling instead of refusing, and only then update
    this test.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

        self.safetensors_dir = self.root / "safetensors-pack"
        self.safetensors_dir.mkdir()
        (self.safetensors_dir / "config.json").write_text("{}", encoding="utf-8")
        (self.safetensors_dir / "model.safetensors").write_bytes(
            build_safetensors_bytes(
                [("t.a", "I8", [1024], zero_tensor_bytes("I8", [1024]))]
            )
        )

        self.gguf_dir = self.root / "gguf-pack"
        self.gguf_dir.mkdir()
        (self.gguf_dir / "pack.gguf").write_bytes(
            build_gguf_bytes(
                tensors=[
                    {"name": "token_embd.weight", "dims": [4], "type": 0, "offset": 0}
                ],
                data_section=bytes(16),
            )
        )

    def _attestation_fields(self, sizer_path, model_dir):
        # Run the REAL sizer, not a stub: the whole point is to detect
        # drift in what the actual shipped program emits.
        #
        # --wired-limit-mib is passed explicitly because these tests must
        # run on the Linux CI runner as well as on macOS. Left to itself a
        # sizer derives the ceiling from iogpu.wired_limit_mb / hw.memsize,
        # neither of which exists on Linux, and it then refuses (exit 1)
        # with "could not determine a wired memory ceiling" -- a refusal
        # about the HOST, not about the pack. Naming the ceiling removes
        # the host dependency without weakening what is asserted below:
        # the sizer still runs for real against a real pack, and a context
        # ceiling would still show up in its attested fields if it ever
        # emitted one.
        proc = subprocess.run(
            [
                sys.executable,
                str(sizer_path),
                "--model-path",
                str(model_dir),
                "--kv-reserve-gib",
                "1",
                "--wired-limit-mib",
                "16384",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"real sizer {sizer_path.name} did not run: {proc.stderr}",
        )
        fields = {}
        for line in (proc.stdout + proc.stderr).splitlines():
            for token in line.split():
                if "=" in token:
                    key, _, value = token.partition("=")
                    fields[key] = value
        return fields, proc

    def test_real_safetensors_sizer_emits_no_context_ceiling(self):
        fields, proc = self._attestation_fields(
            SAFETENSORS_FIT_CHECK_PATH, self.safetensors_dir
        )
        # Guard against a vacuous pass: the sizer must have emitted SOME
        # attested fields, or "no ceiling" would be true of empty output.
        self.assertTrue(
            fields, f"real sizer emitted no key=value fields at all: {proc.stdout!r}"
        )
        self.assertNotIn("fit_context_ceiling", fields)

    def test_real_gguf_sizer_emits_no_context_ceiling(self):
        fields, proc = self._attestation_fields(GGUF_FIT_CHECK_PATH, self.gguf_dir)
        self.assertTrue(
            fields, f"real sizer emitted no key=value fields at all: {proc.stdout!r}"
        )
        self.assertNotIn("fit_context_ceiling", fields)


if __name__ == "__main__":
    unittest.main()
