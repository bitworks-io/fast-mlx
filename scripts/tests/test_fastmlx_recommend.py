import contextlib
import importlib.util
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.tests.test_fastmlx_launch import (
    CAPTURING_FIT_CHECK_BODY,
    GREEN_ATTESTATION_WITH_RESIDENCY_BODY,
    SYNTHETIC_CARD_ID,
    SYNTHETIC_CARD_REPO,
    SYNTHETIC_CARD_REVISION,
    write_expert_stream_card_manifest,
)


RECOMMEND_PATH = Path(__file__).resolve().parents[1] / "fastmlx_recommend.py"
_SPEC = importlib.util.spec_from_file_location("fastmlx_recommend", RECOMMEND_PATH)
assert _SPEC is not None and _SPEC.loader is not None
FASTMLX_RECOMMEND = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(FASTMLX_RECOMMEND)


PASS_REPO = "example/PassModel"
PASS_CARD_ID = "fixture-pass@test"

REFERENCE_REPO = "example/ReferenceModel"
REFERENCE_CARD_ID = "fixture-reference@test"

EXACT_REPO = "example/ExactModel"
EXACT_CARD_ID = "fixture-exact@test"

NO_GO_REPO = "example/NoGoModel"
NO_GO_CARD_ID = "fixture-no-go@test"
NO_GO_TIER = "Noticeable"
NO_GO_HEADLINE = "About 1 word in 6 differs from the reference model."

UNMEASURED_REPO = "example/UnmeasuredModel"
UNMEASURED_CARD_ID = "fixture-unmeasured@test"

UNCARDED_REPO = "example/NoCardModel"  # no matching card in the manifest at all

# An hfPin-only NO_GO card (no repo at all), identifying a pulled pack only
# through the revision a pull receipt records.
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
                "id": PASS_CARD_ID,
                "model": {"repo": PASS_REPO, "hfPin": "cafebabe"},
                "verdict": "PASS",
                "legible": {
                    "tier": "Reference",
                    "headline": "Matches the reference closely.",
                    "nextWordDrift": {"top1AgreementPct": 91.2},
                    "benefit": {"speedX": None, "speedXStatus": "not-measured"},
                },
            },
            {
                "id": REFERENCE_CARD_ID,
                "model": {"repo": REFERENCE_REPO, "hfPin": "0badc0de"},
                "verdict": "REFERENCE",
                "legible": {
                    "tier": "Reference",
                    "headline": "This IS the reference.",
                    "nextWordDrift": {"top1AgreementPct": 99.9},
                    "benefit": {"speedX": 1.35, "speedXStatus": "measured"},
                },
            },
            {
                "id": EXACT_CARD_ID,
                "model": {"repo": EXACT_REPO, "hfPin": "0123abcd"},
                "verdict": "EXACT",
                "legible": {
                    "tier": "Exact",
                    "headline": "Token-exact with the reference.",
                },
            },
            {
                "id": NO_GO_CARD_ID,
                "model": {"repo": NO_GO_REPO, "hfPin": "deadbeef"},
                "verdict": "NO_GO",
                "legible": {"tier": NO_GO_TIER, "headline": NO_GO_HEADLINE},
            },
            {
                "id": UNMEASURED_CARD_ID,
                "model": {"repo": UNMEASURED_REPO, "hfPin": "abadcafe"},
                "verdict": "UNMEASURED",
                "legible": {"tier": None, "headline": None},
            },
            {
                "id": PIN_ONLY_NO_GO_CARD_ID,
                "model": {"repo": None, "hfPin": PIN_ONLY_NO_GO_HF_PIN},
                "verdict": "NO_GO",
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


def write_pull_receipt(model_dir: Path, repo_id, revision: str, dest: Path = None) -> Path:
    """Write a receipt at exactly the path and in exactly the shape
    ``fastmlx_pull.pull()`` writes one for ``dest`` (defaulting to
    ``model_dir`` itself): the SAME ``receipt_path_for`` naming rule and the
    SAME ``downloader.write_exclusive`` call pull's own code uses.
    """
    dest = dest if dest is not None else model_dir
    receipt_path = FASTMLX_RECOMMEND.launch.pull.receipt_path_for(dest)
    receipt = {
        "format_version": 1,
        "repo_id": repo_id,
        "revision": revision,
        "dest": str(dest),
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
    FASTMLX_RECOMMEND.launch.pull.downloader.write_exclusive(receipt_path, receipt_bytes)
    return receipt_path


GREEN_FIT_CHECK_BODY = f"""#!{sys.executable}
import sys
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

# A distinctive stderr reason an unrunnable sizer might print -- used to
# verify the error row's message surfaces the sizer's OWN reason, not just
# its exit code (see the sibling fastmlx_launch fail-closed detail test).
DISTINCTIVE_UNRUNNABLE_REASON = "distinctive-reason: pack is missing ngram_table.bin"

DISTINCTIVE_REASON_FIT_CHECK_BODY = f"""#!{sys.executable}
import sys
sys.stderr.write({DISTINCTIVE_UNRUNNABLE_REASON!r} + "\\n")
sys.exit(1)
"""


class FastmlxRecommendTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

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

    def make_model_dir(self, name: str, repo: str = None, revision: str = None) -> Path:
        model_dir = self.root / name
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        if repo is not None or revision is not None:
            write_pull_receipt(model_dir, repo_id=repo, revision=revision or ("e" * 40))
        return model_dir

    def base_argv(self, model_paths, **overrides) -> list:
        argv = ["recommend", "--quality-cards", str(self.manifest_path)]
        for path in model_paths:
            argv += ["--model-path", str(path)]
        args = {"--fit-check-bin": str(self.green_fit_bin)}
        args.update(overrides)
        for key, value in args.items():
            if value is None:
                continue
            argv += [key, str(value)]
        return argv

    def run_main(self, argv: list):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                FASTMLX_RECOMMEND.main(argv)
        return ctx.exception.code, stdout.getvalue(), stderr.getvalue()

    def run_json(self, argv: list):
        code, stdout, stderr = self.run_main(argv + ["--json"])
        return code, json.loads(stdout), stderr

    # ------------------------------------------------------------------
    # Classification: recommended (PASS card, green fit).
    # ------------------------------------------------------------------
    def test_pass_card_and_green_fit_is_recommended(self):
        model_dir = self.make_model_dir("pass-model", repo=PASS_REPO)
        code, doc, _ = self.run_json(self.base_argv([model_dir]))
        self.assertEqual(code, 0)
        rows = doc["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "recommended")
        self.assertEqual(rows[0]["card"]["id"], PASS_CARD_ID)
        self.assertEqual(rows[0]["fit"]["verdict"], "GREEN")

    # ------------------------------------------------------------------
    # No speed printed when the card carries no numeric speedX.
    # ------------------------------------------------------------------
    def test_no_speed_printed_when_card_has_no_numeric_speedx(self):
        model_dir = self.make_model_dir("pass-model", repo=PASS_REPO)
        code, doc, _ = self.run_json(self.base_argv([model_dir]))
        self.assertEqual(code, 0)
        self.assertNotIn("speedX", doc["rows"][0]["card"])
        _, stdout, _ = self.run_main(self.base_argv([model_dir]))
        self.assertNotIn("speedX", stdout)

    def test_speed_printed_when_card_has_numeric_speedx(self):
        model_dir = self.make_model_dir("reference-model", repo=REFERENCE_REPO)
        code, doc, _ = self.run_json(self.base_argv([model_dir]))
        self.assertEqual(code, 0)
        self.assertEqual(doc["rows"][0]["card"]["speedX"], 1.35)
        _, stdout, _ = self.run_main(self.base_argv([model_dir]))
        self.assertIn("speedX=1.35x", stdout)

    # ------------------------------------------------------------------
    # opt-in: NO_GO card carries the exact accept flag.
    # ------------------------------------------------------------------
    def test_no_go_card_is_opt_in_with_accept_flag(self):
        model_dir = self.make_model_dir("no-go-model", repo=NO_GO_REPO)
        code, doc, _ = self.run_json(self.base_argv([model_dir]))
        self.assertEqual(code, 1)
        row = doc["rows"][0]
        self.assertEqual(row["status"], "opt-in")
        self.assertEqual(row["accept_quality_flag"], f"--accept-quality {NO_GO_CARD_ID}")
        self.assertIn(NO_GO_TIER, row["message"])
        self.assertIn(NO_GO_HEADLINE, row["message"])
        _, stdout, _ = self.run_main(self.base_argv([model_dir]))
        self.assertIn(f"--accept-quality {NO_GO_CARD_ID}", stdout)

    # ------------------------------------------------------------------
    # opt-in via a pulled layout: no --model-repo/--model-revision at all,
    # only a sibling pull receipt (written the way fastmlx_pull.py actually
    # writes one) resolving the pinned revision that an hfPin-only NO_GO
    # card matches.
    # ------------------------------------------------------------------
    def test_pulled_layout_with_hf_pin_no_go_card_is_opt_in_with_accept_flag(self):
        model_dir = self.root / "pinned-pull-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        write_pull_receipt(model_dir, repo_id=None, revision=PIN_ONLY_NO_GO_REVISION)
        code, doc, _ = self.run_json(self.base_argv([model_dir]))
        self.assertEqual(code, 1)
        row = doc["rows"][0]
        self.assertEqual(row["status"], "opt-in")
        self.assertEqual(row["revision"], PIN_ONLY_NO_GO_REVISION)
        self.assertEqual(
            row["accept_quality_flag"], f"--accept-quality {PIN_ONLY_NO_GO_CARD_ID}"
        )
        self.assertIn(PIN_ONLY_NO_GO_TIER, row["message"])
        self.assertIn(PIN_ONLY_NO_GO_HEADLINE, row["message"])

    # ------------------------------------------------------------------
    # uncarded: no card at all, and a recognized-but-UNMEASURED verdict.
    # ------------------------------------------------------------------
    def test_no_card_at_all_is_uncarded(self):
        model_dir = self.make_model_dir("uncarded-model", repo=UNCARDED_REPO)
        code, doc, _ = self.run_json(self.base_argv([model_dir]))
        self.assertEqual(code, 1)
        row = doc["rows"][0]
        self.assertEqual(row["status"], "uncarded")
        self.assertIsNone(row["card"])
        self.assertIn("no measured quality card", row["message"])

    def test_unmeasured_verdict_card_is_uncarded(self):
        model_dir = self.make_model_dir("unmeasured-model", repo=UNMEASURED_REPO)
        code, doc, _ = self.run_json(self.base_argv([model_dir]))
        self.assertEqual(code, 1)
        self.assertEqual(doc["rows"][0]["status"], "uncarded")

    # ------------------------------------------------------------------
    # does-not-fit: RED fit verdict.
    # ------------------------------------------------------------------
    def test_red_fit_is_does_not_fit(self):
        model_dir = self.make_model_dir("pass-model", repo=PASS_REPO)
        code, doc, _ = self.run_json(
            self.base_argv([model_dir], **{"--fit-check-bin": str(self.red_fit_bin)})
        )
        self.assertEqual(code, 2)
        row = doc["rows"][0]
        self.assertEqual(row["status"], "does-not-fit")
        self.assertIn("requested context exceeds", row["message"])

    # ------------------------------------------------------------------
    # error: an unrunnable fit check never crashes, becomes an error row.
    # ------------------------------------------------------------------
    def test_unrunnable_fit_check_is_an_error_row_not_a_crash(self):
        model_dir = self.make_model_dir("pass-model", repo=PASS_REPO)
        code, doc, _ = self.run_json(
            self.base_argv([model_dir], **{"--fit-check-bin": str(self.unknown_fit_bin)})
        )
        self.assertEqual(code, 2)
        row = doc["rows"][0]
        self.assertEqual(row["status"], "error")
        self.assertIn("fit check could not run", row["message"])

    def test_unrunnable_fit_check_error_row_includes_sizers_own_stderr_reason(self):
        model_dir = self.make_model_dir("pass-model", repo=PASS_REPO)
        code, doc, _ = self.run_json(
            self.base_argv(
                [model_dir], **{"--fit-check-bin": str(self.distinctive_reason_fit_bin)}
            )
        )
        self.assertEqual(code, 2)
        row = doc["rows"][0]
        self.assertEqual(row["status"], "error")
        self.assertIn("fit check could not run", row["message"])
        self.assertIn(DISTINCTIVE_UNRUNNABLE_REASON, row["message"])

    def test_missing_fit_check_binary_path_is_an_error_row_not_a_crash(self):
        model_dir = self.make_model_dir("pass-model", repo=PASS_REPO)
        code, doc, _ = self.run_json(
            self.base_argv(
                [model_dir], **{"--fit-check-bin": str(self.root / "does-not-exist")}
            )
        )
        self.assertEqual(code, 2)
        self.assertEqual(doc["rows"][0]["status"], "error")

    # ------------------------------------------------------------------
    # error: an ambiguous card lookup never crashes, becomes an error row.
    # ------------------------------------------------------------------
    def test_ambiguous_card_lookup_is_an_error_row_not_a_crash(self):
        # PASS_REPO resolves the PASS card by repo; a revision built from the
        # NO_GO card's own hfPin ("deadbeef") resolves a DIFFERENT card by
        # pin -- the same ambiguity fastmlx_launch.resolve_card refuses on.
        conflicting_revision = "deadbeef" + "0" * 32
        model_dir = self.make_model_dir(
            "ambiguous-model", repo=PASS_REPO, revision=conflicting_revision
        )
        code, doc, _ = self.run_json(self.base_argv([model_dir]))
        self.assertEqual(code, 2)
        row = doc["rows"][0]
        self.assertEqual(row["status"], "error")
        self.assertIn("ambiguous", row["message"])

    # ------------------------------------------------------------------
    # Bad candidate paths never crash either.
    # ------------------------------------------------------------------
    def test_missing_model_directory_is_an_error_row(self):
        code, doc, _ = self.run_json(self.base_argv([self.root / "does-not-exist"]))
        self.assertEqual(code, 2)
        self.assertEqual(doc["rows"][0]["status"], "error")
        self.assertIn("does not exist", doc["rows"][0]["message"])

    def test_missing_config_json_is_an_error_row(self):
        bare_dir = self.root / "bare-model"
        bare_dir.mkdir()
        code, doc, _ = self.run_json(self.base_argv([bare_dir]))
        self.assertEqual(code, 2)
        self.assertIn("config.json", doc["rows"][0]["message"])

    # ------------------------------------------------------------------
    # Ranking: recommended > opt-in > uncarded > error; within recommended,
    # REFERENCE/EXACT before PASS; uncarded never ranked above a carded row.
    # (does-not-fit vs. the others is covered separately below: the fit
    # check binary is one global flag for the whole run, so a single
    # invocation cannot mix a RED verdict for one candidate with GREEN for
    # another.)
    # ------------------------------------------------------------------
    def test_ranking_order_across_recommended_opt_in_uncarded_and_error(self):
        pass_dir = self.make_model_dir("z-pass", repo=PASS_REPO)
        ref_dir = self.make_model_dir("a-reference", repo=REFERENCE_REPO)
        exact_dir = self.make_model_dir("b-exact", repo=EXACT_REPO)
        no_go_dir = self.make_model_dir("no-go", repo=NO_GO_REPO)
        uncarded_dir = self.make_model_dir("uncarded", repo=UNCARDED_REPO)
        missing_dir = self.root / "missing"

        argv = self.base_argv(
            [pass_dir, ref_dir, exact_dir, no_go_dir, uncarded_dir, missing_dir]
        )
        code, doc, _ = self.run_json(argv)
        self.assertEqual(code, 0)
        statuses = [row["status"] for row in doc["rows"]]
        self.assertEqual(
            statuses,
            [
                "recommended",  # a-reference (REFERENCE)
                "recommended",  # b-exact (EXACT)
                "recommended",  # z-pass (PASS) -- ranked after REFERENCE/EXACT
                "opt-in",
                "uncarded",
                "error",
            ],
        )
        # REFERENCE/EXACT rank ahead of PASS within "recommended", in the
        # stable discovery order they were passed in (ref before exact).
        recommended_ids = [row["card"]["id"] for row in doc["rows"][:3]]
        self.assertEqual(
            recommended_ids, [REFERENCE_CARD_ID, EXACT_CARD_ID, PASS_CARD_ID]
        )
        # Never ranked above a carded row: uncarded sits after opt-in.
        self.assertLess(statuses.index("opt-in"), statuses.index("uncarded"))

    # ------------------------------------------------------------------
    # --models-dir discovery: every immediate subdir with a config.json.
    # ------------------------------------------------------------------
    def test_models_dir_discovers_immediate_subdirs_with_config_json(self):
        models_root = self.root / "models"
        models_root.mkdir()
        pass_dir = models_root / "pass-model"
        pass_dir.mkdir()
        (pass_dir / "config.json").write_text("{}", encoding="utf-8")
        write_pull_receipt(pass_dir, repo_id=PASS_REPO, revision="e" * 40)
        no_config_dir = models_root / "not-a-model"
        no_config_dir.mkdir()  # no config.json: must not be discovered

        argv = [
            "recommend",
            "--quality-cards",
            str(self.manifest_path),
            "--models-dir",
            str(models_root),
            "--fit-check-bin",
            str(self.green_fit_bin),
            "--json",
        ]
        code, stdout, _ = self.run_main(argv)
        doc = json.loads(stdout)
        self.assertEqual(code, 0)
        self.assertEqual(len(doc["rows"]), 1)
        self.assertEqual(doc["rows"][0]["name"], "pass-model")

    # ------------------------------------------------------------------
    # --models-dir discovery must also pick up a GGUF-only subdir (no
    # config.json, at least one top-level *.gguf file) -- the same
    # accepted-layout fix fastmlx serve's precondition gets.
    # ------------------------------------------------------------------
    def test_models_dir_discovers_gguf_only_subdirs(self):
        models_root = self.root / "models"
        models_root.mkdir()
        gguf_dir = models_root / "gguf-model"
        gguf_dir.mkdir()
        (gguf_dir / "pack.gguf").write_bytes(b"not a real gguf file, just a marker")
        no_model_dir = models_root / "not-a-model"
        no_model_dir.mkdir()  # neither config.json nor .gguf: must not be discovered

        argv = [
            "recommend",
            "--quality-cards",
            str(self.manifest_path),
            "--models-dir",
            str(models_root),
            "--fit-check-bin",
            str(self.green_fit_bin),
            "--json",
        ]
        code, stdout, _ = self.run_main(argv)
        doc = json.loads(stdout)
        self.assertEqual(code, 1)  # no card for this pack: uncarded, not recommended
        self.assertEqual(len(doc["rows"]), 1)
        self.assertEqual(doc["rows"][0]["name"], "gguf-model")
        self.assertEqual(doc["rows"][0]["status"], "uncarded")

    # ------------------------------------------------------------------
    # A GGUF-only model directory's row still runs the fit check and is
    # classified exactly like a config.json-layout row: identity resolved
    # from its sibling pull receipt, fit check run, card matched.
    # ------------------------------------------------------------------
    def test_gguf_only_model_dir_row_runs_fit_check_and_is_classified(self):
        gguf_dir = self.root / "gguf-pack"
        gguf_dir.mkdir()
        (gguf_dir / "shard-00001-of-00001.gguf").write_bytes(b"marker")
        write_pull_receipt(gguf_dir, repo_id=PASS_REPO, revision="e" * 40)

        code, doc, _ = self.run_json(self.base_argv([gguf_dir]))
        self.assertEqual(code, 0)
        rows = doc["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "gguf-pack")
        self.assertEqual(rows[0]["status"], "recommended")
        self.assertEqual(rows[0]["card"]["id"], PASS_CARD_ID)
        self.assertEqual(rows[0]["fit"]["verdict"], "GREEN")

    def test_model_path_and_models_dir_dedupe_the_same_directory(self):
        models_root = self.root / "models"
        models_root.mkdir()
        pass_dir = models_root / "pass-model"
        pass_dir.mkdir()
        (pass_dir / "config.json").write_text("{}", encoding="utf-8")
        write_pull_receipt(pass_dir, repo_id=PASS_REPO, revision="e" * 40)

        argv = [
            "recommend",
            "--quality-cards",
            str(self.manifest_path),
            "--model-path",
            str(pass_dir),
            "--models-dir",
            str(models_root),
            "--fit-check-bin",
            str(self.green_fit_bin),
            "--json",
        ]
        code, stdout, _ = self.run_main(argv)
        doc = json.loads(stdout)
        self.assertEqual(code, 0)
        self.assertEqual(len(doc["rows"]), 1)

    # ------------------------------------------------------------------
    # Exit codes.
    # ------------------------------------------------------------------
    def test_exit_code_2_when_no_candidates_at_all(self):
        argv = ["recommend", "--quality-cards", str(self.manifest_path), "--json"]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("no model candidates found", stderr)

    def test_exit_code_1_when_nothing_recommended_but_something_fits(self):
        model_dir = self.make_model_dir("no-go-model", repo=NO_GO_REPO)
        code, _, _ = self.run_main(self.base_argv([model_dir]))
        self.assertEqual(code, 1)

    def test_exit_code_2_when_nothing_fits_at_all(self):
        model_dir = self.make_model_dir("pass-model", repo=PASS_REPO)
        code, _, _ = self.run_main(
            self.base_argv([model_dir], **{"--fit-check-bin": str(self.red_fit_bin)})
        )
        self.assertEqual(code, 2)

    def test_missing_explicit_quality_cards_manifest_is_a_usage_error(self):
        missing_manifest = self.root / "no-such-manifest.json"
        model_dir = self.make_model_dir("pass-model", repo=PASS_REPO)
        argv = [
            "recommend",
            "--quality-cards",
            str(missing_manifest),
            "--model-path",
            str(model_dir),
            "--fit-check-bin",
            str(self.green_fit_bin),
        ]
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn(str(missing_manifest), stderr)

    # ------------------------------------------------------------------
    # --json shape: exactly the documented envelope, no stray stdout prose.
    # ------------------------------------------------------------------
    def test_json_output_shape_and_no_extra_stdout_prose(self):
        model_dir = self.make_model_dir("pass-model", repo=PASS_REPO)
        code, stdout, _ = self.run_main(self.base_argv([model_dir]) + ["--json"])
        self.assertEqual(code, 0)
        self.assertEqual(stdout.count("\n"), 1)  # one JSON line, nothing else
        doc = json.loads(stdout)
        self.assertEqual(doc["schema"], "fastmlx-recommend-v0")
        self.assertIsInstance(doc["rows"], list)


    def test_does_not_fit_ranks_before_error(self):
        pass_dir = self.make_model_dir("pass-model", repo=PASS_REPO)
        missing_dir = self.root / "missing"
        argv = self.base_argv(
            [pass_dir, missing_dir], **{"--fit-check-bin": str(self.red_fit_bin)}
        )
        code, doc, _ = self.run_json(argv)
        self.assertEqual(code, 2)
        self.assertEqual(
            [row["status"] for row in doc["rows"]], ["does-not-fit", "error"]
        )

    # ------------------------------------------------------------------
    # `--engine-profile`: recommend reuses the profile's own `fitCheck`
    # exactly like `fastmlx serve` does (CLI `--fit-check-bin` still wins).
    # ------------------------------------------------------------------
    def _write_fit_check_profile(self, fit_check: dict, name: str = "served-engine") -> Path:
        profile_path = self.root / "fit-check-profile.json"
        profile_path.write_text(
            json.dumps(
                {
                    "schema": "fastmlx-engine-profile-v1",
                    "name": name,
                    "argv": ["{engine_bin}"],
                    "fitCheck": fit_check,
                }
            ),
            encoding="utf-8",
        )
        return profile_path

    def test_recommend_honors_engine_profile_fit_check(self):
        model_dir = self.make_model_dir("pass-model", repo=PASS_REPO)
        capture_path = self.root / "captured-fit-argv.json"
        capturing_bin = write_script(self.root / "fit-capture.py", CAPTURING_FIT_CHECK_BODY)
        profile_path = self._write_fit_check_profile(
            {"bin": str(capturing_bin.resolve()), "args": ["--kv-reserve-gib", "8"]}
        )
        os.environ["FIT_CHECK_CAPTURE_PATH"] = str(capture_path)
        try:
            argv = [
                "recommend",
                "--quality-cards",
                str(self.manifest_path),
                "--model-path",
                str(model_dir),
                "--engine-profile",
                str(profile_path),
                "--json",
            ]
            code, stdout, _ = self.run_main(argv)
        finally:
            del os.environ["FIT_CHECK_CAPTURE_PATH"]
        doc = json.loads(stdout)
        self.assertEqual(code, 0)
        self.assertEqual(doc["rows"][0]["status"], "recommended")
        captured_argv = json.loads(capture_path.read_text(encoding="utf-8"))
        self.assertEqual(captured_argv[0], str(capturing_bin.resolve()))
        self.assertIn("--kv-reserve-gib", captured_argv)

    def test_recommend_cli_fit_check_bin_overrides_profile_fit_check(self):
        model_dir = self.make_model_dir("pass-model", repo=PASS_REPO)
        profile_path = self._write_fit_check_profile(
            {"bin": "builtin:safetensors", "args": ["--kv-reserve-gib", "8"]}
        )
        argv = [
            "recommend",
            "--quality-cards",
            str(self.manifest_path),
            "--model-path",
            str(model_dir),
            "--engine-profile",
            str(profile_path),
            "--fit-check-bin",
            str(self.green_fit_bin),
            "--json",
        ]
        code, stdout, stderr = self.run_main(argv)
        doc = json.loads(stdout)
        self.assertEqual(code, 0)
        self.assertEqual(doc["rows"][0]["status"], "recommended")

    def test_recommend_malformed_engine_profile_exits_3_with_reason(self):
        model_dir = self.make_model_dir("pass-model", repo=PASS_REPO)
        profile_path = self.root / "bad-profile.json"
        profile_path.write_text(
            json.dumps({"schema": "fastmlx-engine-profile-v1", "name": "x"}),
            encoding="utf-8",
        )
        argv = [
            "recommend",
            "--quality-cards",
            str(self.manifest_path),
            "--model-path",
            str(model_dir),
            "--engine-profile",
            str(profile_path),
        ]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 3)
        self.assertIn("argv", stderr)

    # ------------------------------------------------------------------
    # No-identity hint: recommend prints the same one-line hint `fastmlx
    # serve` does, prefixed for recommend, when a pack has no resolvable
    # identity at all (no pull receipt, no revision).
    # ------------------------------------------------------------------
    def test_recommend_prints_no_identity_hint_when_pack_has_no_identity(self):
        model_dir = self.root / "no-identity-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        argv = self.base_argv([model_dir])
        _, _, stderr = self.run_main(argv)
        self.assertIn("fastmlx recommend:", stderr)
        self.assertIn("no model identity", stderr)
        # recommend walks many candidates, so each hint must say which one.
        self.assertIn(f"fastmlx recommend: {model_dir}: no model identity", stderr)

    def test_recommend_no_identity_hint_absent_when_identity_resolves(self):
        model_dir = self.make_model_dir("pass-model", repo=PASS_REPO)
        argv = self.base_argv([model_dir])
        _, _, stderr = self.run_main(argv)
        self.assertNotIn("no model identity", stderr)


class RecommendResidencyTestCase(unittest.TestCase):
    """Residency-aware card matching (fast-mlx-quality-card-v1's
    config.residency), driven through a synthetic expert-streaming fixture
    card (see `test_fastmlx_launch.expert_stream_card_manifest`).
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.quality_cards_path = write_expert_stream_card_manifest(self.root)
        self.green_fit_bin = write_script(self.root / "fit-green.py", GREEN_FIT_CHECK_BODY)

    def make_model_dir(self, name: str, repo: str = None, revision: str = None) -> Path:
        model_dir = self.root / name
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        if repo is not None or revision is not None:
            write_pull_receipt(model_dir, repo_id=repo, revision=revision or ("e" * 40))
        return model_dir

    def base_argv(self, model_paths, **overrides) -> list:
        argv = ["recommend"]
        for path in model_paths:
            argv += ["--model-path", str(path)]
        args = {"--fit-check-bin": str(self.green_fit_bin)}
        args.update(overrides)
        for key, value in args.items():
            if value is None:
                continue
            argv += [key, str(value)]
        return argv

    def run_main(self, argv: list):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                FASTMLX_RECOMMEND.main(argv)
        return ctx.exception.code, stdout.getvalue(), stderr.getvalue()

    def run_json(self, argv: list):
        code, stdout, stderr = self.run_main(argv + ["--json"])
        return code, json.loads(stdout), stderr

    def make_synthetic_card_model_dir(self) -> Path:
        return self.make_model_dir(
            "expert-stream-model", repo=SYNTHETIC_CARD_REPO, revision=SYNTHETIC_CARD_REVISION
        )

    def test_expert_stream_row_for_synthetic_card_is_opt_in(self):
        model_dir = self.make_synthetic_card_model_dir()
        argv = self.base_argv(
            [model_dir],
            **{
                "--quality-cards": str(self.quality_cards_path),
                "--residency": "expert-stream",
            },
        )
        code, doc, _ = self.run_json(argv)
        self.assertEqual(code, 1)
        row = doc["rows"][0]
        self.assertEqual(row["status"], "opt-in")
        self.assertEqual(row["residency"], "expert-stream")
        self.assertEqual(row["accept_quality_flag"], f"--accept-quality {SYNTHETIC_CARD_ID}")

        _, stdout, _ = self.run_main(argv)
        self.assertIn("residency=expert-stream", stdout)

    def test_resident_row_for_synthetic_card_is_uncarded(self):
        model_dir = self.make_synthetic_card_model_dir()
        argv = self.base_argv(
            [model_dir],
            **{
                "--quality-cards": str(self.quality_cards_path),
                "--residency": "resident",
            },
        )
        code, doc, _ = self.run_json(argv)
        self.assertEqual(code, 1)
        row = doc["rows"][0]
        self.assertEqual(row["status"], "uncarded")
        self.assertIsNone(row["card"])
        self.assertEqual(row["residency"], "resident")

        _, stdout, _ = self.run_main(argv)
        self.assertNotIn("residency=", stdout)

    def test_fit_check_arg_residency_mismatch_is_a_usage_error(self):
        model_dir = self.make_synthetic_card_model_dir()
        argv = self.base_argv(
            [model_dir],
            **{
                "--quality-cards": str(self.quality_cards_path),
                "--residency": "resident",
                "--fit-check-arg": "--residency=expert-stream",
            },
        )
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("resident", stderr)
        self.assertIn("expert-stream", stderr)
        # Fix #3: names the actual source of the conflict.
        self.assertIn("--fit-check-arg", stderr)
        self.assertNotIn("engine profile", stderr.lower())

    def test_fit_check_arg_residency_mismatch_from_profile_names_profile_as_source(self):
        model_dir = self.make_synthetic_card_model_dir()
        profile_path = self.root / "fit-check-profile.json"
        profile_path.write_text(
            json.dumps(
                {
                    "schema": "fastmlx-engine-profile-v1",
                    "name": "served-engine",
                    "argv": ["{engine_bin}"],
                    "fitCheck": {
                        "bin": "builtin:gguf",
                        "args": ["--residency", "expert-stream"],
                    },
                }
            ),
            encoding="utf-8",
        )
        argv = self.base_argv(
            [model_dir],
            **{
                "--quality-cards": str(self.quality_cards_path),
                "--residency": "resident",
                "--engine-profile": str(profile_path),
                "--fit-check-bin": None,
            },
        )
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("resident", stderr)
        self.assertIn("expert-stream", stderr)
        self.assertIn("engine profile", stderr.lower())
        self.assertIn("fitcheck.args", stderr.lower())

    def test_fit_check_arg_residency_mismatch_scans_every_occurrence(self):
        model_dir = self.make_synthetic_card_model_dir()
        argv = self.base_argv(
            [model_dir],
            **{
                "--quality-cards": str(self.quality_cards_path),
                "--residency": "resident",
            },
        ) + [
            "--fit-check-arg=--residency",
            "--fit-check-arg=resident",
            "--fit-check-arg=--residency=expert-stream",
        ]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("resident", stderr)
        self.assertIn("expert-stream", stderr)

    def test_fit_check_arg_residency_mismatch_via_abbreviation_is_refused(self):
        model_dir = self.make_synthetic_card_model_dir()
        argv = self.base_argv(
            [model_dir],
            **{
                "--quality-cards": str(self.quality_cards_path),
                "--residency": "resident",
            },
        ) + ["--fit-check-arg=--resid=expert-stream"]
        code, _, stderr = self.run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("expert-stream", stderr)

    # ------------------------------------------------------------------
    # Fix #2(b): a GREEN attestation's own residency= field must agree
    # with the requested --residency; a row whose attestation disagrees
    # must never come out recommended/green.
    # ------------------------------------------------------------------
    def test_attestation_residency_mismatch_row_is_an_error_not_recommended(self):
        model_dir = self.make_model_dir("pass-model", repo=SYNTHETIC_CARD_REPO)
        mismatched_bin = write_script(
            self.root / "fit-residency-mismatch.py",
            GREEN_ATTESTATION_WITH_RESIDENCY_BODY("expert-stream"),
        )
        argv = self.base_argv(
            [model_dir],
            **{
                "--quality-cards": str(self.quality_cards_path),
                "--residency": "resident",
                "--fit-check-bin": str(mismatched_bin),
            },
        )
        code, doc, stderr = self.run_json(argv)
        row = doc["rows"][0]
        self.assertEqual(row["status"], "error")
        self.assertIn("resident", row["message"])
        self.assertIn("expert-stream", row["message"])
        self.assertNotEqual(code, 0)


class RankingHelperTests(unittest.TestCase):
    """Pure-function tests for the ranking key, independent of the CLI."""

    def test_uncarded_never_outranks_a_carded_row(self):
        uncarded = {"status": "uncarded", "card": None}
        opt_in = {"status": "opt-in", "card": {"verdict": "NO_GO"}}
        ranked = FASTMLX_RECOMMEND.rank_rows([uncarded, opt_in])
        self.assertEqual([row["status"] for row in ranked], ["opt-in", "uncarded"])

    def test_reference_and_exact_outrank_pass_within_recommended(self):
        pass_row = {"status": "recommended", "card": {"verdict": "PASS"}}
        reference_row = {"status": "recommended", "card": {"verdict": "REFERENCE"}}
        exact_row = {"status": "recommended", "card": {"verdict": "EXACT"}}
        ranked = FASTMLX_RECOMMEND.rank_rows([pass_row, reference_row, exact_row])
        self.assertEqual(
            [row["card"]["verdict"] for row in ranked], ["REFERENCE", "EXACT", "PASS"]
        )

    def test_exit_code_prefers_recommended_over_everything_else(self):
        rows = [{"status": "error"}, {"status": "uncarded"}, {"status": "recommended"}]
        self.assertEqual(FASTMLX_RECOMMEND.exit_code_for(rows), 0)

    def test_exit_code_is_2_when_only_errors_and_does_not_fit(self):
        rows = [{"status": "error"}, {"status": "does-not-fit"}]
        self.assertEqual(FASTMLX_RECOMMEND.exit_code_for(rows), 2)


if __name__ == "__main__":
    unittest.main()
