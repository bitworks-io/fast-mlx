import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.tests.test_fastmlx_gguf_fit import build_gguf_bytes
from scripts.tests.test_fastmlx_launch import (
    CAPTURING_FIT_CHECK_BODY,
    GREEN_ATTESTATION_WITH_RESIDENCY_BODY,
    P3_SHARED_REPO_AND_PIN_REPO,
    P3_SHARED_REPO_AND_PIN_REVISION,
    SYNTHETIC_CARD_ID,
    SYNTHETIC_CARD_REPO,
    SYNTHETIC_CARD_REVISION,
    _load_real_quality_guides_manifest,
    _synthetic_card,
    write_expert_stream_card_manifest,
    write_release_engine_binary,
    write_release_provenance,
)
from scripts.tests.test_fastmlx_safetensors_fit import build_safetensors_bytes, zero_tensor_bytes


REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"

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

# A carded pack with a MEASURED speedX > 1.0 whose speedXStatus names a
# referent that must never be dropped or paraphrased -- the cycle-104
# defect this file's new tests pin (see fastmlx_launch.card_benefit_line).
MIXED_SPEED_REPO = "example/MixedSpeedModel"
MIXED_SPEED_CARD_ID = "fixture-mixed-speed@test"
MIXED_SPEED_REFERENT = "not against the full-precision model"
MIXED_SPEED_HEADLINE = "Mixed 4/8-bit pack; near-lossless drift."

# A carded pack with a MEASURED speedX < 1.0: the polarity case a naive
# "{n}x slower" phrasing would invert.
SLOWDOWN_REPO = "example/SlowdownModel"
SLOWDOWN_CARD_ID = "fixture-slowdown@test"
SLOWDOWN_HEADLINE = "Slower mixed pack; near-lossless drift."

# The cycle-106 defect this file's new tests pin: `legible.benefit.fit` (a
# footprint + which Mac classes a pack fits) never survived into a
# `recommend` row at all -- for FOUR of the six real published cards, `fit`
# is the pack's ONLY benefit (their `speedX` is null), so those packs were
# presented as pure cost with their entire upside dropped. Fit text below
# is deliberately distinct from any headline/tier string already in this
# file, so a clause split can never accidentally match the wrong text.
PASS_FIT_TEXT = "20.7 GB — fits a 24 GB Mac"
MIXED_SPEED_FIT_TEXT = "12.3 GB — fits an 18 GB Mac"

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
            {
                "id": MIXED_SPEED_CARD_ID,
                "model": {"repo": MIXED_SPEED_REPO, "hfPin": "1234abcd"},
                "verdict": "PASS",
                "legible": {
                    "tier": "Near-lossless",
                    "headline": MIXED_SPEED_HEADLINE,
                    "benefit": {
                        "speedX": 1.19,
                        "speedXStatus": (
                            "measured on Apple M3 Ultra: this is a ratio against "
                            "the 8-bit reference pack, " + MIXED_SPEED_REFERENT + "."
                        ),
                    },
                },
            },
            {
                "id": SLOWDOWN_CARD_ID,
                "model": {"repo": SLOWDOWN_REPO, "hfPin": "5678beef"},
                "verdict": "PASS",
                "legible": {
                    "tier": "Near-lossless",
                    "headline": SLOWDOWN_HEADLINE,
                    "benefit": {
                        "speedX": 0.485,
                        "speedXStatus": (
                            "measured on Apple M3 Ultra: ratio against the 8-bit "
                            "reference pack."
                        ),
                    },
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

    def write_manifest_with_card_fit(self, card_id: str, fit_text: str) -> Path:
        """A manifest byte-identical to ``fixture_manifest()`` except that
        one named card's ``legible.benefit.fit`` is set -- written to its
        own file rather than mutating ``self.manifest_path`` so every other
        test in this class keeps running against the unmodified fixture.
        """
        manifest = fixture_manifest()
        found = False
        for card in manifest["cards"]:
            if card["id"] == card_id:
                card.setdefault("legible", {}).setdefault("benefit", {})["fit"] = fit_text
                found = True
        assert found, f"no fixture card {card_id!r} to attach a fit clause to"
        path = self.root / f"manifest-with-fit-{card_id.replace('@', '-')}.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

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
        # A bare ratio with no referent is exactly the defect this file's
        # cycle-104 tests pin (see test_speedx_row_never_prints_a_bare_ratio
        # below); the direction+status line replaces it.
        self.assertNotIn("speedX=1.35x", stdout)
        self.assertIn("1.35x faster on this engine (measured)", stdout)

    # ------------------------------------------------------------------
    # A speed ratio must NEVER be shown without its speedXStatus scope --
    # the only measurement boundary (host, engine build, flags, prompt
    # count, and the referent the ratio is against) fast-mlx has for the
    # number. See docs/agent-handoff.md cycle 104 and
    # build_public_site.quality_speed_line, which this CLI helper mirrors.
    # ------------------------------------------------------------------
    def test_speedx_row_never_prints_a_bare_ratio(self):
        model_dir = self.make_model_dir("mixed-speed-model", repo=MIXED_SPEED_REPO)
        _, stdout, _ = self.run_main(self.base_argv([model_dir]))
        self.assertIn(MIXED_SPEED_REFERENT, stdout)
        self.assertNotIn("speedX=1.19x", stdout)

    def test_sub_one_speedx_says_slowdown_not_faster(self):
        model_dir = self.make_model_dir("slowdown-model", repo=SLOWDOWN_REPO)
        _, stdout, _ = self.run_main(self.base_argv([model_dir]))
        self.assertIn("slowdown", stdout)
        self.assertNotIn("faster", stdout)

    def test_json_row_carries_speedx_status_whenever_speedx(self):
        model_a = self.make_model_dir("mixed-speed-model", repo=MIXED_SPEED_REPO)
        model_b = self.make_model_dir("reference-model", repo=REFERENCE_REPO)
        model_c = self.make_model_dir("pass-model", repo=PASS_REPO)
        code, doc, _ = self.run_json(self.base_argv([model_a, model_b, model_c]))
        self.assertEqual(code, 0)
        saw_numeric_speedx = False
        for row in doc["rows"]:
            card = row.get("card") or {}
            if "speedX" in card:
                saw_numeric_speedx = True
                self.assertIn("speedXStatus", card, msg=card)
        self.assertTrue(
            saw_numeric_speedx, "fixture produced no row with a numeric speedX to check"
        )

    # ------------------------------------------------------------------
    # Anti-regression pin: a card with no legible.benefit at all (EXACT_CARD_ID
    # has no "benefit" key) must render byte-identically to today.
    # ------------------------------------------------------------------
    def test_card_without_benefit_renders_unchanged(self):
        model_dir = self.make_model_dir("exact-model", repo=EXACT_REPO)
        code, doc, _ = self.run_json(self.base_argv([model_dir]))
        self.assertEqual(code, 0)
        card = doc["rows"][0]["card"]
        self.assertNotIn("speedX", card)
        self.assertNotIn("speedXStatus", card)
        self.assertNotIn("benefitFit", card)
        _, stdout, _ = self.run_main(self.base_argv([model_dir]))
        self.assertNotIn("speed:", stdout)
        self.assertNotIn("card fit:", stdout)

    # ------------------------------------------------------------------
    # Structural guard: the CLI helper must agree with the site renderer's
    # polarity on every direction (faster / no difference / slowdown /
    # status-only), so this defect class cannot reappear on a third surface.
    # ------------------------------------------------------------------
    def test_speed_direction_matches_site_renderer(self):
        scripts_dir = str(Path(__file__).resolve().parents[1])
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import build_public_site  # noqa: E402  (test-only; never imported by the CLI)

        launch = FASTMLX_RECOMMEND.launch
        for speed_x in (1.19, 1.00, 0.485, None):
            benefit = {"speedX": speed_x, "speedXStatus": "measured on Apple M3 Ultra"}
            card = {"legible": {"benefit": benefit}}
            expected = build_public_site.quality_speed_line(benefit)
            actual = launch.card_benefit_line(card)
            self.assertEqual(actual, expected, msg=f"speedX={speed_x!r}")

    # ------------------------------------------------------------------
    # cycle-106: legible.benefit.fit must survive into a recommend row and
    # render under its own "card fit: " clause, independent of speedX.
    # ------------------------------------------------------------------
    def test_text_row_shows_card_fit_clause(self):
        manifest_path = self.write_manifest_with_card_fit(
            MIXED_SPEED_CARD_ID, MIXED_SPEED_FIT_TEXT
        )
        model_dir = self.make_model_dir("mixed-speed-model", repo=MIXED_SPEED_REPO)
        _, stdout, _ = self.run_main(
            self.base_argv([model_dir], **{"--quality-cards": str(manifest_path)})
        )
        self.assertIn(f"card fit: {MIXED_SPEED_FIT_TEXT}", stdout)
        # The "speed: " clause (everything between "speed: " and the next
        # "card fit: " label) must not itself contain the fit text -- the
        # same clause-separation guard test_fastmlx_launch.py:545 uses.
        speed_clause = stdout.split("speed: ", 1)[1].split("card fit: ", 1)[0]
        self.assertNotIn(MIXED_SPEED_FIT_TEXT, speed_clause)

    def test_json_row_carries_benefit_fit(self):
        manifest_path = self.write_manifest_with_card_fit(
            MIXED_SPEED_CARD_ID, MIXED_SPEED_FIT_TEXT
        )
        model_dir = self.make_model_dir("mixed-speed-model", repo=MIXED_SPEED_REPO)
        code, doc, _ = self.run_json(
            self.base_argv([model_dir], **{"--quality-cards": str(manifest_path)})
        )
        self.assertEqual(code, 0)
        self.assertEqual(doc["rows"][0]["card"]["benefitFit"], MIXED_SPEED_FIT_TEXT)

    # ------------------------------------------------------------------
    # boundary.measuredNewTokens: the generation length a card was measured
    # over is surfaced (strict reader only) in the summary, the text row,
    # and --json; any non-strict value leaves it absent everywhere.
    # ------------------------------------------------------------------
    def write_manifest_with_measured_new_tokens(self, value, present: bool = True) -> Path:
        manifest = fixture_manifest()
        for card in manifest["cards"]:
            if card["id"] == PASS_CARD_ID and present:
                card["boundary"] = {"measuredNewTokens": value}
        path = self.root / "manifest-measured-new-tokens.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_summary_and_text_row_carry_measured_new_tokens(self):
        card = {
            "id": "mnt@test",
            "verdict": "PASS",
            "legible": {"tier": "Reference", "nextWordDrift": {"top1AgreementPct": 91.2}},
            "boundary": {"measuredNewTokens": 64},
        }
        self.assertEqual(FASTMLX_RECOMMEND._card_summary(card)["measuredNewTokens"], 64)

        manifest_path = self.write_manifest_with_measured_new_tokens(64)
        model_dir = self.make_model_dir("pass-model", repo=PASS_REPO)
        code, stdout, _ = self.run_main(
            self.base_argv([model_dir], **{"--quality-cards": str(manifest_path)})
        )
        self.assertEqual(code, 0)
        self.assertIn("measured_tokens=64", stdout)
        self.assertLess(stdout.index("top1="), stdout.index("measured_tokens=64"))

    def test_non_strict_measured_new_tokens_is_absent_everywhere(self):
        for index, bad in enumerate((True, 0, "64", 64.0)):
            with self.subTest(value=repr(bad)):
                card = {
                    "id": "mnt@test",
                    "verdict": "PASS",
                    "legible": {"tier": "Reference"},
                    "boundary": {"measuredNewTokens": bad},
                }
                self.assertNotIn("measuredNewTokens", FASTMLX_RECOMMEND._card_summary(card))
                manifest_path = self.write_manifest_with_measured_new_tokens(bad)
                model_dir = self.root / f"pass-model-bad-{index}"
                model_dir.mkdir()
                (model_dir / "config.json").write_text("{}", encoding="utf-8")
                write_pull_receipt(model_dir, repo_id=PASS_REPO, revision="e" * 40)
                code, stdout, _ = self.run_main(
                    self.base_argv([model_dir], **{"--quality-cards": str(manifest_path)})
                )
                self.assertEqual(code, 0)
                self.assertNotIn("measured_tokens=", stdout)
                code, doc, _ = self.run_json(
                    self.base_argv([model_dir], **{"--quality-cards": str(manifest_path)})
                )
                self.assertNotIn("measuredNewTokens", doc["rows"][0]["card"])

    def test_card_without_boundary_has_no_measured_new_tokens(self):
        model_dir = self.make_model_dir("pass-model", repo=PASS_REPO)
        code, doc, _ = self.run_json(self.base_argv([model_dir]))
        self.assertEqual(code, 0)
        self.assertNotIn("measuredNewTokens", doc["rows"][0]["card"])

    def test_json_row_carries_measured_new_tokens(self):
        manifest_path = self.write_manifest_with_measured_new_tokens(64)
        model_dir = self.make_model_dir("pass-model", repo=PASS_REPO)
        code, doc, _ = self.run_json(
            self.base_argv([model_dir], **{"--quality-cards": str(manifest_path)})
        )
        self.assertEqual(code, 0)
        self.assertEqual(doc["rows"][0]["card"]["measuredNewTokens"], 64)

    # ------------------------------------------------------------------
    # README drift pin: the --json field emitted by `_card_summary` for a
    # card's own fit sentence must stay documented, under its REAL name --
    # not a name hardcoded on both sides, which could drift with the code
    # and still "pass". Derive the emitted key from a real _card_summary()
    # call (diffing a card with legible.benefit.fit set against one
    # without) so a future rename fails this test instead of silently
    # leaving README.md describing a key `_card_summary` no longer emits.
    # ------------------------------------------------------------------
    def test_readme_documents_the_real_card_benefit_fit_key(self):
        card_with_fit = {
            "id": "readme-pin@test",
            "verdict": "PASS",
            "legible": {"benefit": {"fit": "fits a 24 GB Mac"}},
        }
        card_without_fit = {"id": "readme-pin@test", "verdict": "PASS", "legible": {}}
        summary_with = FASTMLX_RECOMMEND._card_summary(card_with_fit)
        summary_without = FASTMLX_RECOMMEND._card_summary(card_without_fit)
        new_keys = set(summary_with) - set(summary_without)
        self.assertEqual(
            len(new_keys),
            1,
            msg=f"expected exactly one new key when only benefit.fit is set, got {new_keys}",
        )
        emitted_key = new_keys.pop()
        readme_text = README.read_text(encoding="utf-8")
        self.assertIn(
            f"`{emitted_key}`",
            readme_text,
            msg=(
                f"README.md must document the --json field `{emitted_key}` "
                "(the name _card_summary currently emits for a card's own fit "
                "sentence) so a --json consumer is not forced to read source"
            ),
        )
        # It must also be documented as a fact DISTINCT from the row's own
        # live fit-check verdict key -- not merged/aliased to plain `fit`.
        # Clamp the window start at 0: a negative slice start would wrap to
        # the END of README.md and assert against unrelated prose, which is a
        # silent false pass rather than a failure.
        mention = readme_text.index(f"`{emitted_key}`")
        benefit_fit_paragraph = readme_text[max(0, mention - 200) : mention + 200]
        self.assertIn(
            "`fit`",
            benefit_fit_paragraph,
            msg=(
                f"README.md's `{emitted_key}` documentation must call out the row's own "
                "live `fit` verdict as the distinct fact it is not merged into"
            ),
        )

    def test_fit_clause_independent_of_speedx(self):
        # PASS_CARD_ID's benefit has speedX=None (see fixture_manifest) --
        # four of the six real published cards look exactly like this: a
        # fit clause with no measured speed at all. The clause must still
        # render.
        manifest_path = self.write_manifest_with_card_fit(PASS_CARD_ID, PASS_FIT_TEXT)
        model_dir = self.make_model_dir("pass-model", repo=PASS_REPO)
        _, stdout, _ = self.run_main(
            self.base_argv([model_dir], **{"--quality-cards": str(manifest_path)})
        )
        self.assertIn(f"card fit: {PASS_FIT_TEXT}", stdout)

    def test_card_fit_clause_matches_launch_helper(self):
        # Exact string equality against launch.card_fit_line, mirroring the
        # cross-surface pin test_speed_direction_matches_site_renderer uses
        # for the speed clause above.
        launch = FASTMLX_RECOMMEND.launch
        row = {
            "name": "x",
            "repo": None,
            "revision": None,
            "residency": "resident",
            "fit": None,
            "card": {
                "id": "fixture-fit-helper@test",
                "verdict": "PASS",
                "tier": "Reference",
                "benefitFit": MIXED_SPEED_FIT_TEXT,
            },
            "status": "recommended",
            "message": None,
            "accept_quality_flag": None,
            "engineBuild": None,
            "mtp": None,
        }
        text = FASTMLX_RECOMMEND._format_row_text(1, row)
        expected = launch.card_fit_line(
            {"legible": {"benefit": {"fit": MIXED_SPEED_FIT_TEXT}}}
        )
        actual = text.split("card fit: ", 1)[1].splitlines()[0]
        self.assertEqual(actual, expected)

    def test_fit_clause_is_distinguishable_from_host_verdict(self):
        # The real repro: this HOST's live fit-check verdict (RED) and the
        # card's own sentence ("fits a 24 GB Mac") are two different facts
        # that can disagree on one row. Both must render, and the card
        # sentence must appear ONLY under "card fit: ", never under a bare
        # "fit="/"fit: " token.
        card = FASTMLX_RECOMMEND._card_summary(
            {
                "id": "fixture-fit-mismatch@test",
                "verdict": "NO_GO",
                "legible": {
                    "tier": "Noticeable",
                    "headline": "About 1 word in 6 differs.",
                    "benefit": {"fit": PASS_FIT_TEXT},
                },
            }
        )
        row = {
            "name": "optiq-pack",
            "repo": None,
            "revision": None,
            "residency": "resident",
            "fit": {"verdict": "RED", "context": 262144},
            "card": card,
            "status": "does-not-fit",
            "message": None,
            "accept_quality_flag": None,
            "engineBuild": None,
            "mtp": None,
        }
        text = FASTMLX_RECOMMEND._format_row_text(1, row)
        self.assertIn("fit=RED", text)
        self.assertIn(f"card fit: {PASS_FIT_TEXT}", text)
        # Everything before the "card fit: " label (the head + any speed
        # clause) must not itself contain the card's fit sentence.
        before_card_fit = text.split("card fit: ", 1)[0]
        self.assertNotIn(PASS_FIT_TEXT, before_card_fit)

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
    # The RED-fit defect this cycle's tests pin: `build_row` resolves the
    # pack's quality card long before the fit check runs (see
    # `launch.resolve_card` above), but a live RED verdict used to return
    # before `row["card"]` was ever set -- the ONE resolved fact `build_row`
    # threw away on this path (`engineBuild`/`mtp` already survive it). A
    # does-not-fit pack is exactly the case where the discarded card
    # (headline/tier/verdict) matters most: it is the artifact that would
    # tell the operator what this pack costs in quality and point them at a
    # different one. Two arms, same pack/host, differing ONLY in whether
    # the quality-card manifest has a matching card -- proving the carry is
    # conditioned on a REAL resolved card, not just always-None turning
    # into some other unconditional placeholder.
    # ------------------------------------------------------------------
    def test_does_not_fit_row_carries_resolved_card_when_one_resolves(self):
        model_dir = self.make_model_dir("pass-model", repo=PASS_REPO)
        code, doc, _ = self.run_json(
            self.base_argv([model_dir], **{"--fit-check-bin": str(self.red_fit_bin)})
        )
        self.assertEqual(code, 2)
        row = doc["rows"][0]
        self.assertEqual(row["status"], "does-not-fit")
        card = row["card"]
        self.assertIsNotNone(card, msg="does-not-fit row dropped its resolved card")
        self.assertEqual(card["id"], PASS_CARD_ID)
        self.assertEqual(card["verdict"], "PASS")
        self.assertEqual(card["tier"], "Reference")
        self.assertEqual(card["headline"], "Matches the reference closely.")

    def test_does_not_fit_row_card_is_none_when_no_card_resolves(self):
        # The discrimination arm: same pack shape, same host, same RED fit
        # check -- only the manifest lookup differs (this repo has no
        # matching card at all). Without this arm, a test asserting only
        # the carded case above would pass equally well against a bug that
        # always sets `row["card"] = {}` or similar.
        model_dir = self.make_model_dir("uncarded-pack", repo=UNCARDED_REPO)
        code, doc, _ = self.run_json(
            self.base_argv([model_dir], **{"--fit-check-bin": str(self.red_fit_bin)})
        )
        self.assertEqual(code, 2)
        row = doc["rows"][0]
        self.assertEqual(row["status"], "does-not-fit")
        self.assertIsNone(row["card"])

    def test_does_not_fit_text_rendering_keeps_live_fit_and_card_fit_in_order(self):
        # Integration-level (through the real CLI/build_row, not a
        # hand-built row dict): the head line must still show the LIVE
        # fit-check verdict this host measured, and the card's own fit
        # sentence must render under the existing "card fit: " label
        # (never a bare "fit:") -- _format_row_text is unchanged, this only
        # proves it already covers a does-not-fit row with a card attached.
        manifest_path = self.write_manifest_with_card_fit(PASS_CARD_ID, PASS_FIT_TEXT)
        model_dir = self.make_model_dir("pass-model", repo=PASS_REPO)
        _, stdout, _ = self.run_main(
            self.base_argv(
                [model_dir],
                **{
                    "--fit-check-bin": str(self.red_fit_bin),
                    "--quality-cards": str(manifest_path),
                },
            )
        )
        self.assertIn("fit=RED", stdout)
        self.assertIn("card fit:", stdout)
        self.assertIn(f"card fit: {PASS_FIT_TEXT}", stdout)
        # Order: the live verdict is in the head line; the card's sentence
        # is on a later line under its own label.
        self.assertLess(stdout.index("fit=RED"), stdout.index("card fit:"))

    def test_does_not_fit_status_and_ranking_unchanged_by_the_card_carry(self):
        # No behavioral change to classification, ranking, or exit code:
        # a does-not-fit row with a now-carried card must still rank
        # exactly like it did before (ahead of error, behind everything
        # recommendable) and must never itself become admitted.
        pass_dir = self.make_model_dir("pass-model", repo=PASS_REPO)
        missing_dir = self.root / "missing"
        argv = self.base_argv(
            [pass_dir, missing_dir], **{"--fit-check-bin": str(self.red_fit_bin)}
        )
        code, doc, _ = self.run_json(argv)
        self.assertEqual(code, 2)
        rows = doc["rows"]
        self.assertEqual([row["status"] for row in rows], ["does-not-fit", "error"])
        does_not_fit_row = rows[0]
        self.assertEqual(does_not_fit_row["status"], FASTMLX_RECOMMEND.STATUS_DOES_NOT_FIT)
        self.assertIsNotNone(does_not_fit_row["card"])
        # Direct unit-level pin on the ranking/exit-code helpers themselves,
        # confirming a resolved card on a does-not-fit row cannot promote it
        # (rank_rows keys does-not-fit purely on status; exit_code_for never
        # treats does-not-fit as recommendable).
        ranked = FASTMLX_RECOMMEND.rank_rows(list(reversed(rows)))
        self.assertEqual(
            [row["status"] for row in ranked], ["does-not-fit", "error"]
        )
        self.assertEqual(FASTMLX_RECOMMEND.exit_code_for(rows), 2)

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
        # A8: a user with nothing downloaded is pointed at the offline card catalog.
        self.assertIn("fastmlx cards list", stderr)

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


# ---------------------------------------------------------------------
# provenance.engineBuild: a recommend row carries the same status/notice
# fastmlx serve computes (docs/quality-card-schema-v1.md "Engine build"),
# never gating the row's status/verdict.
# ---------------------------------------------------------------------
EB_PASS_REPO_RECOMMEND = "example/EbRecommendPassModel"
EB_PASS_CARD_ID_RECOMMEND = "eb-recommend-pass@test"
EB_CARD_COMMIT_RECOMMEND = "5555555555555555555555555555555555555555"
EB_OTHER_COMMIT_RECOMMEND = "6666666666666666666666666666666666666666"


def engine_build_manifest_for_recommend() -> dict:
    return {
        "schema": "fast-mlx-quality-card-v1",
        "generatedAt": "2026-01-01T00:00:00Z",
        "cards": [
            {
                "id": EB_PASS_CARD_ID_RECOMMEND,
                "model": {"repo": EB_PASS_REPO_RECOMMEND, "hfPin": "eeeeeeee"},
                "verdict": "PASS",
                "legible": {"tier": "Reference", "headline": "eb recommend pass headline"},
                "provenance": {"engineBuild": {"commit": EB_CARD_COMMIT_RECOMMEND}},
            }
        ],
    }


class EngineBuildRecommendTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest_path = self.root / "eb-quality-guides.json"
        self.manifest_path.write_text(
            json.dumps(engine_build_manifest_for_recommend()), encoding="utf-8"
        )
        self.green_fit_bin = write_script(self.root / "fit-green.py", GREEN_FIT_CHECK_BODY)

    def make_model_dir(self, name: str, repo: str = None) -> Path:
        model_dir = self.root / name
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        if repo is not None:
            write_pull_receipt(model_dir, repo_id=repo, revision="e" * 40)
        return model_dir

    def _write_profile(self, commit: str = None) -> Path:
        document = {
            "schema": "fastmlx-engine-profile-v1",
            "name": "eb-profile",
            "argv": ["{engine_bin}"],
        }
        if commit is not None:
            document["engineBuild"] = {"commit": commit}
        path = self.root / "eb-profile.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def run_main(self, argv: list):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                FASTMLX_RECOMMEND.main(argv)
        return ctx.exception.code, stdout.getvalue(), stderr.getvalue()

    def test_recommend_row_carries_engine_build_object_and_text_line(self):
        model_dir = self.make_model_dir("eb-pass-model", repo=EB_PASS_REPO_RECOMMEND)
        profile_path = self._write_profile(commit=EB_OTHER_COMMIT_RECOMMEND)
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
        doc = json.loads(stdout)
        row = doc["rows"][0]
        self.assertEqual(row["engineBuild"]["status"], "mismatch")
        self.assertEqual(row["engineBuild"]["card"], EB_CARD_COMMIT_RECOMMEND)
        self.assertEqual(row["engineBuild"]["launch"], EB_OTHER_COMMIT_RECOMMEND)
        self.assertIsNotNone(row["engineBuild"]["message"])
        self.assertIn(EB_CARD_COMMIT_RECOMMEND[:12], row["engineBuild"]["message"])

        argv_text = [a for a in argv if a != "--json"]
        _, stdout_text, _ = self.run_main(argv_text)
        self.assertIn("transfer unmeasured", stdout_text)

    def test_recommend_row_match_has_no_message(self):
        model_dir = self.make_model_dir("eb-pass-model", repo=EB_PASS_REPO_RECOMMEND)
        profile_path = self._write_profile(commit=EB_CARD_COMMIT_RECOMMEND)
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
        doc = json.loads(stdout)
        row = doc["rows"][0]
        self.assertEqual(row["engineBuild"]["status"], "match")
        self.assertIsNone(row["engineBuild"]["message"])


# ---------------------------------------------------------------------
# engineBuild DERIVATION: no operator-written --engine-profile
# engineBuild.commit at all, resolved instead from a release layout's own
# provenance.json -- the SAME two launch helpers fastmlx serve already
# calls (derive_engine_build_from_release,
# _guarded_engine_bin_abs_for_engine_build_derivation), reused unchanged
# here via --engine-bin. recommend still execs nothing: --engine-bin only
# NAMES the binary these rows describe, it is never run.
# ---------------------------------------------------------------------
class EngineBuildDerivationRecommendTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest_path = self.root / "eb-derive-quality-guides.json"
        self.manifest_path.write_text(
            json.dumps(engine_build_manifest_for_recommend()), encoding="utf-8"
        )
        self.green_fit_bin = write_script(self.root / "fit-green.py", GREEN_FIT_CHECK_BODY)

    def make_model_dir(self, name: str, repo: str = None) -> Path:
        model_dir = self.root / name
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        if repo is not None:
            write_pull_receipt(model_dir, repo_id=repo, revision="e" * 40)
        return model_dir

    def base_argv(self, model_dir: Path, engine_bin: Path, **overrides) -> list:
        args = {
            "--quality-cards": str(self.manifest_path),
            "--model-path": str(model_dir),
            "--fit-check-bin": str(self.green_fit_bin),
            "--engine-bin": str(engine_bin),
        }
        args.update(overrides)
        argv = ["recommend"]
        for key, value in args.items():
            if value is None:
                continue
            argv += [key, str(value)]
        return argv + ["--json"]

    def _write_profile(self, commit: str = None) -> Path:
        document = {
            "schema": "fastmlx-engine-profile-v1",
            "name": "eb-derive-profile",
            "argv": ["{engine_bin}"],
        }
        if commit is not None:
            document["engineBuild"] = {"commit": commit}
        path = self.root / "eb-derive-profile.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def run_main(self, argv: list):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                FASTMLX_RECOMMEND.main(argv)
        return ctx.exception.code, stdout.getvalue(), stderr.getvalue()

    # (1) The headline case: release layout + matching commit + matching
    # binary sha, no profile commit at all -- match, reachable with NO
    # operator-written engineBuild.commit.
    def test_release_layout_derives_match_with_no_profile_commit(self):
        model_dir = self.make_model_dir(
            "eb-derive-match-model", repo=EB_PASS_REPO_RECOMMEND
        )
        engine_bin = write_release_engine_binary(self.root)
        actual_sha256 = FASTMLX_RECOMMEND.launch._sha256_file(str(engine_bin))
        write_release_provenance(
            self.root,
            source_commit=EB_CARD_COMMIT_RECOMMEND,
            source_dirty=False,
            engine_binary_sha256=actual_sha256,
        )
        argv = self.base_argv(model_dir, engine_bin)
        code, stdout, stderr = self.run_main(argv)
        self.assertEqual(code, 0, stderr)
        doc = json.loads(stdout)
        row = doc["rows"][0]
        self.assertEqual(row["engineBuild"]["status"], "match")
        self.assertIsNone(row["engineBuild"]["message"])

    # (2) A dirty source tree's commit does not faithfully name the exact
    # bytes that were built -- stays undeclared, notice present.
    def test_dirty_source_stays_undeclared(self):
        model_dir = self.make_model_dir(
            "eb-derive-dirty-model", repo=EB_PASS_REPO_RECOMMEND
        )
        engine_bin = write_release_engine_binary(self.root)
        actual_sha256 = FASTMLX_RECOMMEND.launch._sha256_file(str(engine_bin))
        write_release_provenance(
            self.root,
            source_commit=EB_CARD_COMMIT_RECOMMEND,
            source_dirty=True,
            engine_binary_sha256=actual_sha256,
        )
        argv = self.base_argv(model_dir, engine_bin)
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        doc = json.loads(stdout)
        row = doc["rows"][0]
        self.assertEqual(row["engineBuild"]["status"], "undeclared")
        self.assertIsNotNone(row["engineBuild"]["message"])

    # (3) provenance.json names a binary hash that does not match the
    # ACTUAL binary at --engine-bin: stays undeclared (never an error row,
    # never a changed exit code) -- a sibling text file is not a seal.
    def test_binary_sha256_mismatch_stays_undeclared_not_an_error(self):
        model_dir = self.make_model_dir(
            "eb-derive-sha-mismatch-model", repo=EB_PASS_REPO_RECOMMEND
        )
        engine_bin = write_release_engine_binary(self.root)
        actual_sha256 = FASTMLX_RECOMMEND.launch._sha256_file(str(engine_bin))
        wrong_sha256 = ("0" if actual_sha256[0] != "0" else "1") + actual_sha256[1:]
        write_release_provenance(
            self.root,
            source_commit=EB_CARD_COMMIT_RECOMMEND,
            source_dirty=False,
            engine_binary_sha256=wrong_sha256,
        )
        argv = self.base_argv(model_dir, engine_bin)
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        doc = json.loads(stdout)
        row = doc["rows"][0]
        self.assertEqual(row["engineBuild"]["status"], "undeclared")
        self.assertNotEqual(row["status"], "error")

    # (4) Absent / malformed / non-dict provenance.json: undeclared, no
    # traceback, no refusal -- an underivable build is a fact this script
    # cannot assert, never a crash.
    def test_absent_or_malformed_provenance_stays_undeclared(self):
        cases = {
            "absent": lambda: write_release_provenance(self.root, omit=True),
            "malformed-json": lambda: write_release_provenance(
                self.root, raw_text="{not-json"
            ),
            "non-dict": lambda: write_release_provenance(
                self.root, raw_text=json.dumps(["not", "a", "dict"])
            ),
        }
        for label, write_provenance in cases.items():
            with self.subTest(provenance=label):
                # A carded candidate (repo=EB_PASS_REPO_RECOMMEND) so the
                # card's OWN commit is known: this makes a bad
                # provenance.json read as "undeclared" (card names a
                # commit, launch has none), not "unrecorded" (no card
                # commit at all, the status an uncarded candidate would
                # get regardless of provenance -- a weaker, less
                # discriminating control).
                model_dir = self.make_model_dir(
                    f"eb-derive-{label}-model", repo=EB_PASS_REPO_RECOMMEND
                )
                engine_bin = write_release_engine_binary(self.root, name=f"{label}-serve")
                write_provenance()
                argv = self.base_argv(model_dir, engine_bin)
                code, stdout, _ = self.run_main(argv)
                self.assertEqual(code, 0)
                doc = json.loads(stdout)
                row = doc["rows"][0]
                self.assertEqual(row["engineBuild"]["status"], "undeclared")
                self.assertNotEqual(row["status"], "error")

    # (5) Precedence: an operator-declared engineBuild.commit in the
    # profile always wins over a derivable one, even when a valid release
    # layout is present.
    def test_profile_declared_commit_beats_derived_commit(self):
        model_dir = self.make_model_dir(
            "eb-derive-precedence-model", repo=EB_PASS_REPO_RECOMMEND
        )
        engine_bin = write_release_engine_binary(self.root)
        actual_sha256 = FASTMLX_RECOMMEND.launch._sha256_file(str(engine_bin))
        write_release_provenance(
            self.root,
            source_commit=EB_OTHER_COMMIT_RECOMMEND,
            source_dirty=False,
            engine_binary_sha256=actual_sha256,
        )
        profile_path = self._write_profile(commit=EB_CARD_COMMIT_RECOMMEND)
        argv = self.base_argv(
            model_dir, engine_bin, **{"--engine-profile": str(profile_path)}
        )
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        doc = json.loads(stdout)
        row = doc["rows"][0]
        self.assertEqual(row["engineBuild"]["launch"], EB_CARD_COMMIT_RECOMMEND)

    # (6) A derived commit that disagrees with the card's own commit is a
    # mismatch, not a match -- and the notice names both 12-char shas so
    # an operator can tell the two builds apart.
    def test_derived_commit_mismatch_names_both_shas(self):
        model_dir = self.make_model_dir(
            "eb-derive-mismatch-model", repo=EB_PASS_REPO_RECOMMEND
        )
        engine_bin = write_release_engine_binary(self.root)
        actual_sha256 = FASTMLX_RECOMMEND.launch._sha256_file(str(engine_bin))
        write_release_provenance(
            self.root,
            source_commit=EB_OTHER_COMMIT_RECOMMEND,
            source_dirty=False,
            engine_binary_sha256=actual_sha256,
        )
        argv = self.base_argv(model_dir, engine_bin)
        code, stdout, _ = self.run_main(argv)
        self.assertEqual(code, 0)
        doc = json.loads(stdout)
        row = doc["rows"][0]
        self.assertEqual(row["engineBuild"]["status"], "mismatch")
        self.assertIn(EB_CARD_COMMIT_RECOMMEND[:12], row["engineBuild"]["message"])
        self.assertIn(EB_OTHER_COMMIT_RECOMMEND[:12], row["engineBuild"]["message"])

    # (7) Non-interference control: for the SAME candidate set, the
    # derivation-available arm (--engine-bin naming a valid release
    # layout) and the derivation-unavailable arm (no --engine-bin at all)
    # must differ in the `engineBuild` sub-object ONLY -- ranking order
    # and every other field (`fit`, `status`, ...) stay byte-identical.
    # Proven with a structural per-row dict diff, not a test-count delta.
    def test_derivation_changes_only_the_engineBuild_sub_object(self):
        carded_dir = self.make_model_dir(
            "eb-derive-noninterference-carded", repo=EB_PASS_REPO_RECOMMEND
        )
        uncarded_dir = self.make_model_dir("eb-derive-noninterference-uncarded")
        engine_bin = write_release_engine_binary(self.root)
        actual_sha256 = FASTMLX_RECOMMEND.launch._sha256_file(str(engine_bin))
        write_release_provenance(
            self.root,
            source_commit=EB_CARD_COMMIT_RECOMMEND,
            source_dirty=False,
            engine_binary_sha256=actual_sha256,
        )

        def argv_for(model_dirs, engine_bin_arg):
            args = {
                "--quality-cards": str(self.manifest_path),
                "--fit-check-bin": str(self.green_fit_bin),
                "--engine-bin": engine_bin_arg,
            }
            argv = ["recommend"]
            for model_dir in model_dirs:
                argv += ["--model-path", str(model_dir)]
            for key, value in args.items():
                if value is None:
                    continue
                argv += [key, str(value)]
            return argv + ["--json"]

        model_dirs = [carded_dir, uncarded_dir]
        code_with, stdout_with, _ = self.run_main(
            argv_for(model_dirs, str(engine_bin))
        )
        code_without, stdout_without, _ = self.run_main(
            argv_for(model_dirs, None)
        )
        self.assertEqual(code_with, code_without)
        rows_with = json.loads(stdout_with)["rows"]
        rows_without = json.loads(stdout_without)["rows"]
        self.assertEqual(len(rows_with), len(rows_without))
        self.assertEqual(
            [row["name"] for row in rows_with], [row["name"] for row in rows_without]
        )
        for row_with, row_without in zip(rows_with, rows_without):
            stripped_with = {k: v for k, v in row_with.items() if k != "engineBuild"}
            stripped_without = {
                k: v for k, v in row_without.items() if k != "engineBuild"
            }
            self.assertEqual(stripped_with, stripped_without)
            self.assertEqual(row_with["status"], row_without["status"])
            self.assertEqual(row_with.get("fit"), row_without.get("fit"))
        # The one field allowed to differ: the carded row's engineBuild
        # status flips from undeclared (no --engine-bin) to match (a
        # derivable release layout).
        self.assertEqual(rows_without[0]["engineBuild"]["status"], "undeclared")
        self.assertEqual(rows_with[0]["engineBuild"]["status"], "match")


# ---------------------------------------------------------------------
# AC1/AC2: built-in pure-Python sizer auto-selected from a candidate
# pack's own contents when neither --fit-check-bin nor an engine-profile
# fitCheck named one -- recommend must be able to answer with Python
# alone, never falling back to searching PATH for the Swift engine binary.
#
# A standalone TestCase (NOT a subclass of FastmlxRecommendTestCase, which
# already carries 40+ of its own test_ methods) -- inheriting it here
# would silently re-run its entire suite under this class's name too,
# following this file's own established convention (see
# RecommendResidencyTestCase below, which duplicates its own small setUp
# rather than subclassing for the same reason).
# ---------------------------------------------------------------------
class BuiltinSizerAutoSelectionTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

        self.manifest_path = self.root / "quality-guides.json"
        self.manifest_path.write_text(json.dumps(fixture_manifest()), encoding="utf-8")

        self.green_fit_bin = write_script(self.root / "fit-green.py", GREEN_FIT_CHECK_BODY)

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

    def _safetensors_model_dir(
        self, name: str = "auto-safetensors-model", repo: str = None
    ) -> Path:
        model_dir = self.root / name
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        blob = build_safetensors_bytes(
            [("t.a", "F32", [4], zero_tensor_bytes("F32", [4]))]
        )
        (model_dir / "model.safetensors").write_bytes(blob)
        if repo is not None:
            write_pull_receipt(model_dir, repo_id=repo, revision="e" * 40)
        return model_dir

    def _write_gguf_file(self, model_dir: Path, name: str = "model.gguf") -> Path:
        tensors = [{"name": "t.f32", "dims": [4], "type": 0, "offset": 0}]
        blob = build_gguf_bytes(tensors=tensors, data_section=bytes(16))
        path = model_dir / name
        path.write_bytes(blob)
        return path

    def _gguf_model_dir(self, name: str = "auto-gguf-model") -> Path:
        model_dir = self.root / name
        model_dir.mkdir()
        self._write_gguf_file(model_dir)
        return model_dir

    # ------------------------------------------------------------------
    # AC1: selection by pack contents.
    # ------------------------------------------------------------------
    def test_safetensors_pack_auto_selects_builtin_safetensors_sizer(self):
        model_dir = self._safetensors_model_dir()
        argv = self.base_argv([model_dir], **{"--fit-check-bin": None})
        code, doc, _ = self.run_json(argv)
        self.assertEqual(code, 2)
        row = doc["rows"][0]
        self.assertEqual(row["status"], "error")
        self.assertIn("builtin:safetensors", row["message"])
        self.assertIn("--kv-reserve-gib", row["message"])

    # ------------------------------------------------------------------
    # Same defect class as the RED-fit does-not-fit row (see
    # FastmlxRecommendTestCase.test_does_not_fit_row_carries_resolved_card_when_one_resolves):
    # the --kv-reserve-gib-missing error row is a DIFFERENT early return in
    # build_row (the auto-selected-builtin-sizer branch, before the fit
    # check even runs) that resolves a card first and must carry it too.
    # ------------------------------------------------------------------
    def test_kv_reserve_gib_missing_error_row_carries_resolved_card(self):
        model_dir = self._safetensors_model_dir(repo=PASS_REPO)
        argv = self.base_argv([model_dir], **{"--fit-check-bin": None})
        code, doc, _ = self.run_json(argv)
        self.assertEqual(code, 2)
        row = doc["rows"][0]
        self.assertEqual(row["status"], "error")
        self.assertIn("--kv-reserve-gib", row["message"])
        card = row["card"]
        self.assertIsNotNone(card, msg="kv-reserve-gib error row dropped its resolved card")
        self.assertEqual(card["id"], PASS_CARD_ID)

    def test_gguf_pack_auto_selects_builtin_gguf_sizer(self):
        model_dir = self._gguf_model_dir()
        argv = self.base_argv([model_dir], **{"--fit-check-bin": None})
        code, doc, _ = self.run_json(argv)
        self.assertEqual(code, 2)
        row = doc["rows"][0]
        self.assertEqual(row["status"], "error")
        self.assertIn("builtin:gguf", row["message"])
        self.assertIn("--kv-reserve-gib", row["message"])

    def test_pack_with_neither_layout_refuses_naming_what_it_looked_for_and_the_remedy(self):
        model_dir = self.make_model_dir("bare-config-only", repo=PASS_REPO)
        argv = self.base_argv([model_dir], **{"--fit-check-bin": None})
        code, doc, _ = self.run_json(argv)
        self.assertEqual(code, 2)
        row = doc["rows"][0]
        self.assertEqual(row["status"], "error")
        self.assertIn("*.safetensors", row["message"])
        self.assertIn("*.gguf", row["message"])
        self.assertIn("--fit-check-bin", row["message"])
        # Never a silent fall-back to naming the Swift engine binary.
        self.assertNotIn("fastmlx-serve", row["message"])

    def test_pack_with_both_layouts_prefers_safetensors_builtin(self):
        model_dir = self._safetensors_model_dir("both-layouts-model")
        self._write_gguf_file(model_dir, name="also-present.gguf")
        argv = self.base_argv([model_dir], **{"--fit-check-bin": None})
        code, doc, _ = self.run_json(argv)
        self.assertEqual(code, 2)
        row = doc["rows"][0]
        self.assertIn("builtin:safetensors", row["message"])
        self.assertNotIn("builtin:gguf", row["message"])

    # ------------------------------------------------------------------
    # AC1 (this cycle): pack-has-safetensors detection must count only
    # entries the safetensors sizer's OWN scan (_iter_regular_files) would
    # actually count -- never a symlink, never anything under (or named
    # with) a dot-prefixed path component. Fixtures below reproduce the
    # REAL defect shape: a real blob file plus a *symlink* named
    # `model.safetensors` pointing at it, exactly the canonical Hugging
    # Face hub cache layout
    # (~/.cache/huggingface/hub/models--<repo>/snapshots/<rev>/).
    # ------------------------------------------------------------------
    def _symlinked_safetensors_model_dir(
        # Deliberately does NOT contain the substring "symlink" -- an
        # earlier draft of this fixture used a name containing it, which
        # made `self.assertIn("symlink", message)` pass for the wrong
        # reason (the directory's OWN path was embedded in even the
        # generic, unrelated fallback message) rather than because the
        # refusal text itself explained the symlink. See the mutation
        # check in the cycle report for how this was caught.
        self, name: str = "hf-cache-style-model", repo: str = None
    ) -> Path:
        model_dir = self.root / name
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        blob_dir = self.root / f"{name}-blobs"
        blob_dir.mkdir(exist_ok=True)
        blob = build_safetensors_bytes(
            [("t.a", "F32", [4], zero_tensor_bytes("F32", [4]))]
        )
        real_blob = blob_dir / "deadbeef0123456789"
        real_blob.write_bytes(blob)
        (model_dir / "model.safetensors").symlink_to(real_blob)
        if repo is not None:
            write_pull_receipt(model_dir, repo_id=repo, revision="e" * 40)
        return model_dir

    def _dot_hidden_safetensors_model_dir(
        self, name: str = "dot-hidden-safetensors-model", repo: str = None
    ) -> Path:
        model_dir = self.root / name
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        hidden_dir = model_dir / ".cache"
        hidden_dir.mkdir()
        blob = build_safetensors_bytes(
            [("t.a", "F32", [4], zero_tensor_bytes("F32", [4]))]
        )
        # A REAL, non-symlink regular file -- the miss here is purely the
        # dot-named directory it lives under, isolating that half of AC1
        # from the symlink half exercised by
        # _symlinked_safetensors_model_dir.
        (hidden_dir / "model.safetensors").write_bytes(blob)
        if repo is not None:
            write_pull_receipt(model_dir, repo_id=repo, revision="e" * 40)
        return model_dir

    def test_pack_has_safetensors_is_false_for_symlink_only_pack(self):
        # Direct unit-level pin on the detection function itself: a
        # symlinked model.safetensors must never register as "has
        # safetensors" -- that is exactly what routed the sizer at a
        # guaranteed "no .safetensors files found" refusal before this fix.
        model_dir = self._symlinked_safetensors_model_dir()
        self.assertFalse(FASTMLX_RECOMMEND._pack_has_safetensors(model_dir))

    def test_pack_has_safetensors_is_false_for_dot_hidden_only_pack(self):
        model_dir = self._dot_hidden_safetensors_model_dir()
        self.assertFalse(FASTMLX_RECOMMEND._pack_has_safetensors(model_dir))

    def test_pack_has_safetensors_is_true_for_a_real_regular_file(self):
        # The happy path AC1 must not break: a normal pack of REAL,
        # non-symlink files still registers as "has safetensors".
        model_dir = self._safetensors_model_dir("plain-real-file-model")
        self.assertTrue(FASTMLX_RECOMMEND._pack_has_safetensors(model_dir))

    def test_symlinked_only_pack_still_selects_builtin_safetensors_through_build_row(self):
        # Reachability through the real CLI entry point (build_row), not
        # just the bare detection function: a symlink-only pack must not
        # silently fall through to "neither layout" either -- it is
        # recognized as a safetensors pack that failed for a SPECIFIC,
        # honest reason (see the next test for the message contents).
        model_dir = self._symlinked_safetensors_model_dir()
        argv = self.base_argv([model_dir], **{"--fit-check-bin": None})
        code, doc, _ = self.run_json(argv)
        self.assertEqual(code, 2)
        row = doc["rows"][0]
        self.assertEqual(row["status"], "error")
        self.assertNotIn("--kv-reserve-gib", row["message"])
        self.assertNotIn("neither a *.safetensors layout", row["message"])

    def test_symlinked_only_pack_refusal_names_symlink_and_a_verified_remedy(self):
        model_dir = self._symlinked_safetensors_model_dir()
        argv = self.base_argv([model_dir], **{"--fit-check-bin": None})
        code, doc, _ = self.run_json(argv)
        self.assertEqual(code, 2)
        message = doc["rows"][0]["message"]
        # The OLD, misleading message a bare rglob() match used to route
        # the sizer into (see fastmlx_safetensors_fit.compute_model_bytes)
        # must be gone: the pack visibly contains model.safetensors, and
        # this message must never assert its absence.
        self.assertNotIn("no .safetensors files found", message)
        self.assertIn("model.safetensors", message)
        self.assertIn("symlink", message)
        # The verified remedy: this repository's OWN downloader writes
        # real, non-symlink files (see hf_pinned_snapshot_download.py); a
        # bare `--adopt` is explicitly NOT offered as the remedy since it
        # refuses a symlinked source directory outright
        # (fastmlx_pull.py's own adopt() verification walk).
        self.assertIn("fastmlx pull", message)
        self.assertIn("--adopt", message)

    def test_symlinked_only_pack_with_kv_reserve_never_reaches_sizers_own_no_files_message(self):
        # The EXACT reported repro: --kv-reserve-gib IS supplied (so the
        # old code's earlier "requires --kv-reserve-gib" gate cannot mask
        # anything), and the OLD detection would still route to the real
        # safetensors sizer subprocess, which refuses with "no
        # .safetensors files found under <dir>" even though the directory
        # visibly contains model.safetensors. This pins the whole
        # end-to-end path: with the fix, this candidate must never reach
        # that sizer subprocess at all.
        model_dir = self._symlinked_safetensors_model_dir()
        argv = self.base_argv(
            [model_dir], **{"--fit-check-bin": None, "--kv-reserve-gib": "1"}
        )
        code, doc, _ = self.run_json(argv)
        self.assertEqual(code, 2)
        message = doc["rows"][0]["message"]
        self.assertNotIn("no .safetensors files found", message)
        self.assertIn("symlink", message)

    def test_dot_hidden_only_pack_refusal_names_dot_directory_and_a_remedy(self):
        model_dir = self._dot_hidden_safetensors_model_dir()
        argv = self.base_argv([model_dir], **{"--fit-check-bin": None})
        code, doc, _ = self.run_json(argv)
        self.assertEqual(code, 2)
        message = doc["rows"][0]["message"]
        self.assertNotIn("no .safetensors files found", message)
        self.assertIn(".cache/model.safetensors", message)
        self.assertIn("dot-named", message)

    # ------------------------------------------------------------------
    # AC2: --kv-reserve-gib forwarding, and the acceptance-level case this
    # whole cycle exists for: recommend answers a real safetensors pack
    # with NO fastmlx-serve on PATH at all.
    # ------------------------------------------------------------------
    def test_kv_reserve_gib_forwarded_to_auto_selected_builtin_and_fit_check_succeeds(self):
        # This is the exact user story this cycle exists for: a real
        # safetensors pack, no fastmlx-serve on PATH anywhere in this
        # process, and recommend still answers "recommended" with Python
        # alone (see the manual repro in scripts/fastmlx_recommend.py's
        # module docstring / cycle's acceptance criteria).
        model_dir = self._safetensors_model_dir(repo=PASS_REPO)
        argv = (
            self.base_argv(
                [model_dir],
                **{"--fit-check-bin": None, "--kv-reserve-gib": "0.5"},
            )
            + [
                # `=` form: a flag-shaped value token (e.g. `--wired-limit-mib`)
                # passed as a separate argv entry would make argparse treat it
                # as a NEW option rather than this --fit-check-arg's value
                # (see the sibling pattern in test_fastmlx_launch.py's
                # "--fit-check-arg=--mmap-side-file" usage).
                "--fit-check-arg=--wired-limit-mib",
                "--fit-check-arg",
                "4096",
                "--fit-check-arg=--wired-margin-gib",
                "--fit-check-arg",
                "2",
            ]
        )
        code, doc, _ = self.run_json(argv)
        self.assertEqual(code, 0, doc)
        row = doc["rows"][0]
        self.assertEqual(row["status"], "recommended")
        self.assertEqual(row["fit"]["verdict"], "GREEN")

    def test_kv_reserve_gib_forwarded_to_explicit_fit_check_bin(self):
        model_dir = self.make_model_dir("pass-model-explicit", repo=PASS_REPO)
        capture_path = self.root / "captured-fit-argv.json"
        capturing_bin = write_script(self.root / "fit-capture.py", CAPTURING_FIT_CHECK_BODY)
        os.environ["FIT_CHECK_CAPTURE_PATH"] = str(capture_path)
        try:
            argv = self.base_argv(
                [model_dir],
                **{"--fit-check-bin": str(capturing_bin), "--kv-reserve-gib": "1.5"},
            )
            code, doc, _ = self.run_json(argv)
        finally:
            del os.environ["FIT_CHECK_CAPTURE_PATH"]
        self.assertEqual(code, 0, doc)
        captured_argv = json.loads(capture_path.read_text(encoding="utf-8"))
        self.assertIn("--kv-reserve-gib", captured_argv)
        self.assertIn("1.5", captured_argv)


# ---------------------------------------------------------------------
# AC3: the no-model-identity hint must never name a flag `recommend`
# itself does not accept (unlike the shared launch.NO_MODEL_IDENTITY_HINT,
# which names --model-revision -- a `fastmlx serve`-only flag). The
# accepted-flag set is derived from recommend's OWN parser, never
# hand-copied, so this test cannot rot independently of the real CLI.
#
# A standalone TestCase, not a subclass of FastmlxRecommendTestCase --
# see the comment on BuiltinSizerAutoSelectionTestCase above for why.
# ---------------------------------------------------------------------
class RecommendNoIdentityHintFlagsTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest_path = self.root / "quality-guides.json"
        self.manifest_path.write_text(json.dumps(fixture_manifest()), encoding="utf-8")
        self.green_fit_bin = write_script(self.root / "fit-green.py", GREEN_FIT_CHECK_BODY)

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

    @staticmethod
    def _recommend_subparser():
        parser = FASTMLX_RECOMMEND.build_arg_parser()
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                return action.choices["recommend"]
        raise AssertionError("fastmlx_recommend's parser has no 'recommend' subparser")

    @classmethod
    def _recommend_accepted_flags(cls) -> set:
        flags = set()
        for action in cls._recommend_subparser()._actions:
            flags.update(action.option_strings)
        return flags

    def _no_identity_model_dir(self, name: str) -> Path:
        model_dir = self.root / name
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        return model_dir

    def test_no_identity_hint_names_no_flag_recommend_does_not_accept(self):
        model_dir = self._no_identity_model_dir("no-identity-flag-check")
        argv = self.base_argv([model_dir])
        _, _, stderr = self.run_main(argv)
        accepted = self._recommend_accepted_flags()
        # Strip any single-quoted example command (e.g. a suggested
        # `fastmlx pull ...` invocation for a DIFFERENT command) before
        # scanning for flag-looking tokens -- a flag that legitimately
        # belongs to a different named command inside a quoted example is
        # never a claim about THIS command's own flags.
        without_examples = re.sub(r"'[^']*'", "", stderr)
        claimed_flags = set(re.findall(r"--[A-Za-z][A-Za-z-]*", without_examples))
        unaccepted = claimed_flags - accepted
        self.assertEqual(
            unaccepted,
            set(),
            msg=(
                f"no-identity hint claims flag(s) recommend does not accept: "
                f"{unaccepted} (stderr={stderr!r})"
            ),
        )

    def test_no_identity_hint_no_longer_mentions_model_revision(self):
        model_dir = self._no_identity_model_dir("no-identity-model-revision-check")
        argv = self.base_argv([model_dir])
        _, _, stderr = self.run_main(argv)
        self.assertNotIn("--model-revision", stderr)
        self.assertIn("no model identity", stderr)
        self.assertIn("fastmlx pull", stderr)
        self.assertIn("--adopt", stderr)


# ---------------------------------------------------------------------
# Shared pack identity: `fastmlx recommend` must judge a candidate by the
# SAME `(repo, revision)` `fastmlx serve` and the engine do
# (`launch.derive_pack_identity`: pull receipt, else the Hugging Face
# hub-cache path; a receipt and a path naming different repos refuse with
# `quality_card_identity_conflict`). Before this, recommend read the
# receipt only, so a hub snapshot was listed uncarded ("no model identity")
# while the gate that would then run refused it.
# ---------------------------------------------------------------------
HUB_SNAPSHOT_REVISION = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
HUB_QWEN_DIR_NAME = "models--mlx-community--Qwen3-0.6B-4bit"


class RecommendSharedPackIdentityTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest_path = self.root / "quality-guides.json"
        self.manifest_path.write_text(json.dumps(fixture_manifest()), encoding="utf-8")
        self.green_fit_bin = write_script(self.root / "fit-green.py", GREEN_FIT_CHECK_BODY)
        self.real_cards = _load_real_quality_guides_manifest()["cards"]

    def hub_dir(self, repo_dir_name: str, revision: str = HUB_SNAPSHOT_REVISION) -> Path:
        """``<root>/hub/models--<org>--<name>/snapshots/<revision>`` with a config.json."""
        path = self.root / "hub" / repo_dir_name / "snapshots" / revision
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        return path

    def plain_dir(self, name: str) -> Path:
        path = self.root / name
        path.mkdir()
        (path / "config.json").write_text("{}", encoding="utf-8")
        return path

    def build_row(self, model_dir: Path, cards, host_class: str = "apple-m5") -> tuple:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            row = FASTMLX_RECOMMEND.build_row(
                model_path=model_dir,
                cards=cards,
                fit_check_bin=str(self.green_fit_bin),
                host_use="shared",
                context=None,
                fit_check_args=[],
                host_hardware_class=lambda: host_class,
            )
        return row, stderr.getvalue()

    def run_json(self, model_dirs: list) -> tuple:
        argv = ["recommend", "--quality-cards", str(self.manifest_path)]
        for path in model_dirs:
            argv += ["--model-path", str(path)]
        argv += ["--fit-check-bin", str(self.green_fit_bin), "--json"]
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                FASTMLX_RECOMMEND.main(argv)
        return ctx.exception.code, json.loads(stdout.getvalue()), stderr.getvalue()

    # A1: a receipt-less hub snapshot of a bundled NO_GO repo is carded.
    def test_a1_hub_snapshot_without_receipt_is_judged_by_its_path_identity(self):
        model_dir = self.hub_dir(HUB_QWEN_DIR_NAME)
        row, stderr = self.build_row(model_dir, self.real_cards)
        self.assertEqual(row["repo"], P3_SHARED_REPO_AND_PIN_REPO)
        self.assertEqual(row["revision"], HUB_SNAPSHOT_REVISION)
        self.assertEqual(row["status"], FASTMLX_RECOMMEND.STATUS_OPT_IN)
        self.assertEqual(row["card"]["id"], "qwen3-0p6b-4bit@m5")
        self.assertEqual(row["card"]["verdict"], "NO_GO")
        self.assertEqual(row["accept_quality_flag"], "--accept-quality qwen3-0p6b-4bit@m5")
        self.assertNotIn("no model identity", stderr)

    def test_a1_hub_snapshot_through_main_is_carded_and_prints_no_identity_line(self):
        model_dir = self.hub_dir("models--example--NoGoModel")
        code, doc, stderr = self.run_json([model_dir])
        self.assertEqual(code, 1)
        row = doc["rows"][0]
        self.assertEqual(row["repo"], NO_GO_REPO)
        self.assertEqual(row["revision"], HUB_SNAPSHOT_REVISION)
        self.assertEqual(row["status"], "opt-in")
        self.assertEqual(row["card"]["id"], NO_GO_CARD_ID)
        self.assertNotIn("no model identity", stderr)

    # A2: a receipt and a hub path naming different repos refuse THAT row only.
    def test_a2_receipt_and_path_naming_different_repos_make_only_that_row_an_error(self):
        conflict_dir = self.hub_dir("models--example--NoGoModel")
        write_pull_receipt(conflict_dir, repo_id=PASS_REPO, revision="e" * 40)
        healthy_dir = self.plain_dir("healthy-pass")
        write_pull_receipt(healthy_dir, repo_id=PASS_REPO, revision="e" * 40)
        code, doc, _ = self.run_json([conflict_dir, healthy_dir])
        rows = {row["path"]: row for row in doc["rows"]}
        conflict = rows[str(conflict_dir)]
        self.assertEqual(conflict["status"], "error")
        self.assertIn("quality_card_identity_conflict", conflict["message"])
        self.assertIn(PASS_REPO, conflict["message"])
        self.assertIn(NO_GO_REPO, conflict["message"])
        self.assertIsNone(conflict["card"])
        healthy = rows[str(healthy_dir)]
        self.assertEqual(healthy["status"], "recommended")
        self.assertEqual(healthy["repo"], PASS_REPO)
        self.assertEqual(healthy["card"]["id"], PASS_CARD_ID)
        self.assertEqual(code, 0)

    def test_a2_conflict_row_never_reaches_the_fit_check(self):
        conflict_dir = self.hub_dir("models--example--NoGoModel")
        write_pull_receipt(conflict_dir, repo_id=PASS_REPO, revision="e" * 40)
        row, _ = self.build_row(conflict_dir, fixture_manifest()["cards"])
        self.assertEqual(row["status"], FASTMLX_RECOMMEND.STATUS_ERROR)
        self.assertIsNone(row["fit"])

    # A3: receipt behavior is unchanged; a missing receipt half comes from the path.
    def test_a3_receipt_in_a_non_hub_directory_is_unchanged(self):
        model_dir = self.plain_dir("receipted")
        write_pull_receipt(model_dir, repo_id=NO_GO_REPO, revision="e" * 40)
        row, stderr = self.build_row(model_dir, fixture_manifest()["cards"])
        self.assertEqual(row["repo"], NO_GO_REPO)
        self.assertEqual(row["revision"], "e" * 40)
        self.assertEqual(row["status"], FASTMLX_RECOMMEND.STATUS_OPT_IN)
        self.assertNotIn("no model identity", stderr)

    def test_a3_receipt_without_revision_inside_a_hub_snapshot_takes_the_snapshot_revision(self):
        model_dir = self.hub_dir("models--example--NoGoModel")
        write_pull_receipt(model_dir, repo_id=NO_GO_REPO, revision=None)
        row, stderr = self.build_row(model_dir, fixture_manifest()["cards"])
        self.assertEqual(row["repo"], NO_GO_REPO)
        self.assertEqual(row["revision"], HUB_SNAPSHOT_REVISION)
        self.assertEqual(row["status"], FASTMLX_RECOMMEND.STATUS_OPT_IN)
        self.assertNotIn("no model identity", stderr)

    # A4: a plain directory outside the hub layout is still uncarded, hint once.
    def test_a4_plain_directory_without_receipt_prints_the_hint_once_and_is_uncarded(self):
        model_dir = self.plain_dir("plain-no-receipt")
        code, doc, stderr = self.run_json([model_dir])
        row = doc["rows"][0]
        self.assertEqual(row["status"], "uncarded")
        self.assertIsNone(row["repo"])
        self.assertIsNone(row["revision"])
        self.assertEqual(stderr.count("no model identity"), 1)
        self.assertIn(f"fastmlx recommend: {model_dir}: no model identity", stderr)

    def test_a4_hint_text_mentions_the_hub_cache_alternative(self):
        self.assertIn(
            "no pull receipt, not a Hugging Face cache snapshot",
            FASTMLX_RECOMMEND.RECOMMEND_NO_MODEL_IDENTITY_HINT,
        )


# ---------------------------------------------------------------------
# tiebreakNotice: `fastmlx recommend` must surface the SAME same-verdict
# lowest-id tiebreak notice `fastmlx serve` already surfaces (see
# `resolve_card`'s docstring and `tiebreak_notice` in fastmlx_launch.py) --
# `build_row` previously called `launch.resolve_card(...)` without
# `notices=`, so the pick was silently reported as though it were
# unambiguous. `build_row`'s own `host_hardware_class=` override lets these
# tests simulate a specific host WITHOUT touching the real sysctl call and
# without relying on subprocess patching (unlike fastmlx_launch's own
# `MainEndToEndSameVerdictTiebreakTestCase`, which has to patch
# `subprocess.run` because `resolve_card`'s default parameter is bound at
# *def* time -- `build_row` resolves its own default at *call* time
# instead, precisely to avoid that trap; see build_row's docstring).
# ---------------------------------------------------------------------
class RecommendTiebreakNoticeTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.green_fit_bin = write_script(self.root / "fit-green.py", GREEN_FIT_CHECK_BODY)
        self.red_fit_bin = write_script(self.root / "fit-red.py", RED_FIT_CHECK_BODY)
        self.real_cards = _load_real_quality_guides_manifest()["cards"]

    def make_model_dir(self, name: str, repo: str, revision: str) -> Path:
        model_dir = self.root / name
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        write_pull_receipt(model_dir, repo_id=repo, revision=revision)
        return model_dir

    def qwen_row(self, fit_check_bin: Path, host_hardware_class) -> dict:
        model_dir = self.make_model_dir(
            "qwen-tiebreak-model",
            P3_SHARED_REPO_AND_PIN_REPO,
            P3_SHARED_REPO_AND_PIN_REVISION,
        )
        return FASTMLX_RECOMMEND.build_row(
            model_path=model_dir,
            cards=self.real_cards,
            fit_check_bin=str(fit_check_bin),
            host_use="shared",
            context=None,
            fit_check_args=[],
            host_hardware_class=host_hardware_class,
        )

    # (a) Real shipped manifest, host class apple-m4-max (matches NEITHER
    # sibling card's own hardwareClass): the lowest-id tiebreak fires, and
    # the exact notice fastmlx_launch.tiebreak_notice would build is
    # present verbatim in both the JSON row and the text rendering, naming
    # both tied card ids and the host class. Status/flag are unchanged
    # from what the pre-existing tiebreak behavior already produces for
    # this NO_GO pack (opt-in, naming the resolved lowest-id card).
    def test_json_and_text_carry_the_exact_tiebreak_notice_on_m4_max(self):
        row = self.qwen_row(self.green_fit_bin, host_hardware_class=lambda: "apple-m4-max")

        expected = FASTMLX_RECOMMEND.launch.tiebreak_notice(
            "qwen3-0p6b-4bit@m3ultra",
            ["qwen3-0p6b-4bit@m3ultra", "qwen3-0p6b-4bit@m5"],
            "apple-m4-max",
        )
        self.assertEqual(row["tiebreakNotice"], expected)
        self.assertIn("qwen3-0p6b-4bit@m3ultra", row["tiebreakNotice"])
        self.assertIn("qwen3-0p6b-4bit@m5", row["tiebreakNotice"])
        self.assertIn("apple-m4-max", row["tiebreakNotice"])

        self.assertEqual(row["status"], FASTMLX_RECOMMEND.STATUS_OPT_IN)
        self.assertEqual(row["accept_quality_flag"], "--accept-quality qwen3-0p6b-4bit@m3ultra")

        text = FASTMLX_RECOMMEND._format_row_text(1, row)
        self.assertIn(expected, text)

    # (b) host apple-m5: hardwareClass narrows to exactly one candidate
    # (the m5 card itself), so the lowest-id tiebreak never fires --
    # tiebreakNotice is null, no notice line in the text rendering, and
    # the resolved card is the host-matched one (not the lowest id).
    def test_host_class_match_wins_no_notice(self):
        row = self.qwen_row(self.green_fit_bin, host_hardware_class=lambda: "apple-m5")

        self.assertIsNone(row["tiebreakNotice"])
        self.assertEqual(row["accept_quality_flag"], "--accept-quality qwen3-0p6b-4bit@m5")

        text = FASTMLX_RECOMMEND._format_row_text(1, row)
        self.assertNotIn("chosen by lowest id", text)
        self.assertNotIn("--card-id", text)

    # (c) host class None (unrecognized): the lowest-id tiebreak fires
    # exactly like (a), but the notice names the host as 'unrecognized'
    # (tiebreak_notice's own rendering for a None host), never the
    # literal string "None".
    def test_unrecognized_host_notice_says_unrecognized(self):
        row = self.qwen_row(self.green_fit_bin, host_hardware_class=lambda: None)

        self.assertIsNotNone(row["tiebreakNotice"])
        self.assertIn("unrecognized", row["tiebreakNotice"])
        self.assertNotIn("None", row["tiebreakNotice"])
        self.assertEqual(row["accept_quality_flag"], "--accept-quality qwen3-0p6b-4bit@m3ultra")

    # (d) A malformed-`model` card elsewhere in the manifest must never be
    # reported as the tiebreak notice: `build_row` reads `notices[0]` as
    # `tiebreakNotice`, so the malformed-card notice is kept out of that
    # list. Both arms first assert the malformed card is present and
    # really is skipped by `find_cards_by_repo` (reachability).
    def _cards_with_a_malformed_model_card(self) -> list:
        malformed = {"id": "zz-malformed-model", "model": None, "verdict": "PASS"}
        shape_notices: list = []
        FASTMLX_RECOMMEND.launch.find_cards_by_repo(
            [malformed], P3_SHARED_REPO_AND_PIN_REPO, notices=shape_notices
        )
        self.assertEqual(len(shape_notices), 1)
        self.assertIn("zz-malformed-model", shape_notices[0])
        return [malformed, *self.real_cards]

    def test_malformed_model_card_is_not_reported_as_the_tiebreak(self):
        self.real_cards = self._cards_with_a_malformed_model_card()
        row = self.qwen_row(self.green_fit_bin, host_hardware_class=lambda: "apple-m4-max")

        expected = FASTMLX_RECOMMEND.launch.tiebreak_notice(
            "qwen3-0p6b-4bit@m3ultra",
            ["qwen3-0p6b-4bit@m3ultra", "qwen3-0p6b-4bit@m5"],
            "apple-m4-max",
        )
        self.assertEqual(row["tiebreakNotice"], expected)

    def test_malformed_model_card_without_a_tie_leaves_tiebreak_null(self):
        self.real_cards = self._cards_with_a_malformed_model_card()
        row = self.qwen_row(self.green_fit_bin, host_hardware_class=lambda: "apple-m5")

        self.assertIsNone(row["tiebreakNotice"])
        self.assertEqual(row["accept_quality_flag"], "--accept-quality qwen3-0p6b-4bit@m5")

    # (e) A does-not-fit row (RED fit check verdict) still carries the
    # notice: it must be set BEFORE the fit-check logic runs, so a card
    # resolved via the tiebreak is never silently dropped from a
    # does-not-fit row.
    def test_does_not_fit_row_still_carries_the_notice(self):
        row = self.qwen_row(self.red_fit_bin, host_hardware_class=lambda: "apple-m4-max")

        self.assertEqual(row["status"], FASTMLX_RECOMMEND.STATUS_DOES_NOT_FIT)
        expected = FASTMLX_RECOMMEND.launch.tiebreak_notice(
            "qwen3-0p6b-4bit@m3ultra",
            ["qwen3-0p6b-4bit@m3ultra", "qwen3-0p6b-4bit@m5"],
            "apple-m4-max",
        )
        self.assertEqual(row["tiebreakNotice"], expected)

    # (d1) Refusal path, unchanged: cards sharing the lowest id refuse
    # (status=error) exactly like fastmlx serve's identical refusal (see
    # fastmlx_launch's SameVerdictLowestIdTiebreakTestCase.
    # test_duplicated_lowest_id_refuses_naming_the_duplicate), and never
    # populate tiebreakNotice (the row returns before it would be set).
    def test_duplicated_lowest_id_refusal_is_unchanged(self):
        repo = "example/RecommendDuplicateLowestIdModel"
        cards = [
            _synthetic_card("dup", repo, "NO_GO", hardware_class="apple-m5"),
            _synthetic_card("dup", repo, "NO_GO", hardware_class="apple-m3-ultra"),
            _synthetic_card("zzz", repo, "NO_GO"),
        ]
        model_dir = self.make_model_dir("dup-lowest-id-model", repo, "d" * 40)
        row = FASTMLX_RECOMMEND.build_row(
            model_path=model_dir,
            cards=cards,
            fit_check_bin=str(self.green_fit_bin),
            host_use="shared",
            context=None,
            fit_check_args=[],
            host_hardware_class=lambda: None,
        )
        self.assertEqual(row["status"], FASTMLX_RECOMMEND.STATUS_ERROR)
        self.assertIn("dup", row["message"])
        self.assertIn("cannot be told apart", row["message"])
        self.assertIsNone(row["tiebreakNotice"])

    # (d2) Refusal path, unchanged: a genuinely mixed-verdict tie (only
    # reachable when none of the tied candidates is NO_GO) still refuses,
    # naming both tied ids, and never populates tiebreakNotice.
    def test_mixed_verdict_refusal_is_unchanged(self):
        repo = "example/RecommendMixedVerdictModel"
        cards = [
            _synthetic_card("a", repo, "PASS", hardware_class="apple-m5"),
            _synthetic_card("b", repo, "EXACT", hardware_class="apple-m3-ultra"),
        ]
        model_dir = self.make_model_dir("mixed-verdict-model", repo, "e" * 40)
        row = FASTMLX_RECOMMEND.build_row(
            model_path=model_dir,
            cards=cards,
            fit_check_bin=str(self.green_fit_bin),
            host_use="shared",
            context=None,
            fit_check_args=[],
            host_hardware_class=lambda: None,
        )
        self.assertEqual(row["status"], FASTMLX_RECOMMEND.STATUS_ERROR)
        self.assertIn("['a', 'b']", row["message"])
        self.assertIn("different verdicts", row["message"])
        self.assertIn("--card-id", row["message"])
        self.assertIsNone(row["tiebreakNotice"])

    # The default (no explicit host_hardware_class passed to build_row)
    # resolves `launch.host_hardware_class` AT CALL TIME, not at def time
    # -- monkeypatching the module attribute reaches it, closing the same
    # trap `resolve_card`'s own default parameter has. This is the seam
    # `_run_recommend` (the real CLI path) relies on; it is exercised here
    # directly rather than via `main()` since that is the narrowest test
    # of the actual fix.
    def test_default_host_hardware_class_resolved_at_call_time(self):
        model_dir = self.make_model_dir(
            "qwen-default-host-model",
            P3_SHARED_REPO_AND_PIN_REPO,
            P3_SHARED_REPO_AND_PIN_REVISION,
        )
        with patch.object(FASTMLX_RECOMMEND.launch, "host_hardware_class", lambda: "apple-m5"):
            row = FASTMLX_RECOMMEND.build_row(
                model_path=model_dir,
                cards=self.real_cards,
                fit_check_bin=str(self.green_fit_bin),
                host_use="shared",
                context=None,
                fit_check_args=[],
            )
        self.assertIsNone(row["tiebreakNotice"])
        self.assertEqual(row["accept_quality_flag"], "--accept-quality qwen3-0p6b-4bit@m5")


# ---------------------------------------------------------------------
# PREDECLARATION 2026-09-27: `fastmlx recommend` surfaces the SAME
# unrecognized-verdict notice `fastmlx serve` does (AC2), via a new
# ``verdictNotice`` row field (JSON) and its text-row rendering. Row
# status/exit-code classification is UNCHANGED -- see
# ``fastmlx_launch.unrecognized_verdict_notice`` for the notice-only
# contract this reuses verbatim.
# ---------------------------------------------------------------------
UNRECOGNIZED_VERDICT_RAW_VALUES = ("no_go", "NO-GO", "NO_G", 123, ["NO_GO"])


def _unrecognized_verdict_manifest_cards() -> list:
    cards = []
    for index, raw_verdict in enumerate(UNRECOGNIZED_VERDICT_RAW_VALUES):
        cards.append(
            {
                "id": f"fixture-unrecognized-verdict-{index}@test",
                "model": {"repo": f"example/UnrecognizedVerdict{index}Model"},
                "verdict": raw_verdict,
            }
        )
    # AC1b: the verdict key is entirely absent.
    cards.append(
        {
            "id": "fixture-missing-verdict@test",
            "model": {"repo": "example/MissingVerdictModel"},
        }
    )
    return cards


class UnrecognizedVerdictRecommendTestCase(FastmlxRecommendTestCase):
    def setUp(self):
        super().setUp()
        self.extra_cards = _unrecognized_verdict_manifest_cards()
        manifest = fixture_manifest()
        manifest["cards"] = manifest["cards"] + self.extra_cards
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # AC2 (json): every unrecognized raw value's row carries a
    # ``verdictNotice`` string naming the card, and the generic "not
    # recommended" message is REPLACED by it (still status "uncarded").
    def test_json_row_carries_verdict_notice_for_each_unrecognized_value(self):
        for card in self.extra_cards:
            if "verdict" not in card:
                continue
            repo = card["model"]["repo"]
            card_id = card["id"]
            with self.subTest(card_id=card_id):
                model_dir = self.make_model_dir(card_id.replace("@", "-"), repo=repo)
                code, doc, stderr = self.run_json(self.base_argv([model_dir]))
                self.assertEqual(code, 1, stderr)
                row = doc["rows"][0]
                # Reachability: the row's card really is the malformed one
                # under test.
                self.assertIsNotNone(row["card"])
                self.assertEqual(row["card"]["id"], card_id)
                self.assertEqual(row["status"], "uncarded")
                self.assertIsNotNone(row["verdictNotice"])
                self.assertIn(card_id, row["verdictNotice"])
                self.assertIn("treated as unmeasured", row["verdictNotice"])
                self.assertIn("upgrade fastmlx or fix the card", row["verdictNotice"])
                self.assertEqual(row["message"], row["verdictNotice"])

    # AC2 (text): the same notice reaches the text rendering.
    def test_text_row_carries_verdict_notice(self):
        card = self.extra_cards[0]
        repo = card["model"]["repo"]
        card_id = card["id"]
        model_dir = self.make_model_dir("text-row-model", repo=repo)
        code, stdout, stderr = self.run_main(self.base_argv([model_dir]))
        self.assertEqual(code, 1, stderr)
        self.assertIn(card_id, stdout)
        self.assertIn("treated as unmeasured", stdout)

    # AC1b, at the recommend level: the missing-verdict wording differs
    # from the unrecognized-verdict wording, exactly like on the serve
    # side.
    def test_missing_verdict_key_json_row_has_its_own_notice(self):
        card = next(c for c in self.extra_cards if "verdict" not in c)
        repo = card["model"]["repo"]
        self.assertNotIn("verdict", card)  # reachability
        model_dir = self.make_model_dir("missing-verdict-model", repo=repo)
        code, doc, stderr = self.run_json(self.base_argv([model_dir]))
        self.assertEqual(code, 1, stderr)
        row = doc["rows"][0]
        self.assertEqual(row["card"]["id"], card["id"])
        self.assertIn("no verdict field", row["verdictNotice"])
        self.assertNotIn("unrecognized verdict", row["verdictNotice"])

    # AC3 controls: no card, an explicit UNMEASURED, and each of the four
    # real verdicts never carry a verdictNotice -- status/message stay
    # exactly as they were before this change.
    def test_controls_have_no_verdict_notice(self):
        cases = (
            (UNCARDED_REPO, "uncarded"),
            (UNMEASURED_REPO, "uncarded"),
            (PASS_REPO, "recommended"),
            (REFERENCE_REPO, "recommended"),
            (EXACT_REPO, "recommended"),
            (NO_GO_REPO, "opt-in"),
        )
        for repo, expected_status in cases:
            with self.subTest(repo=repo):
                model_dir = self.make_model_dir(
                    f"control-{repo.rsplit('/', 1)[-1]}", repo=repo
                )
                code, doc, stderr = self.run_json(self.base_argv([model_dir]))
                row = doc["rows"][0]
                self.assertEqual(row["status"], expected_status, stderr)
                self.assertIsNone(row["verdictNotice"])


# ---------------------------------------------------------------------
# `fastmlx recommend` names the card store it ranked against and can pin it
# by sha256 (`--quality-cards-sha256`), exactly like `fastmlx serve`.
# Predeclaration:
# docs/task-inbox/2026-10-02-PREDECLARATION-launcher-names-and-pins-its-card-store.md
# ---------------------------------------------------------------------
class RecommendCardStoreTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

        self.manifest_path = self.root / "quality-guides.json"
        self.manifest_bytes = json.dumps(fixture_manifest()).encode("utf-8")
        self.manifest_path.write_bytes(self.manifest_bytes)
        self.digest = hashlib.sha256(self.manifest_bytes).hexdigest()
        self.n_cards = len(fixture_manifest()["cards"])

        self.green_fit_bin = write_script(self.root / "fit-green.py", GREEN_FIT_CHECK_BODY)
        self.model_dir = self.root / "pass-model"
        self.model_dir.mkdir()
        (self.model_dir / "config.json").write_text("{}", encoding="utf-8")
        write_pull_receipt(self.model_dir, repo_id=PASS_REPO, revision="e" * 40)

        # An empty stand-in repo root, so the conventional DEFAULT manifest
        # path is absent unless a test writes it.
        self.fake_repo_root = self.root / "repo"
        (self.fake_repo_root / "site").mkdir(parents=True)

    def argv(self, *extra, cards="explicit") -> list:
        argv = ["recommend", "--model-path", str(self.model_dir),
                "--fit-check-bin", str(self.green_fit_bin)]
        if cards == "explicit":
            argv += ["--quality-cards", str(self.manifest_path)]
        return argv + list(extra)

    def run_main(self, argv: list):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with patch.object(FASTMLX_RECOMMEND.launch, "REPO_ROOT", self.fake_repo_root):
                with self.assertRaises(SystemExit) as ctx:
                    FASTMLX_RECOMMEND.main(argv)
        return ctx.exception.code, stdout.getvalue(), stderr.getvalue()

    def identity(self) -> dict:
        return {"sha256": self.digest, "generatedAt": "2026-01-01T00:00:00Z",
                "cards": self.n_cards}

    def test_json_carries_card_store_identity(self):
        code, stdout, stderr = self.run_main(self.argv("--json"))
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["cardStore"], self.identity())

    def test_json_card_store_is_null_when_default_store_absent(self):
        code, stdout, stderr = self.run_main(self.argv("--json", cards="default"))
        self.assertEqual(code, 1, stderr)  # uncarded: fits, nothing measured-good
        doc = json.loads(stdout)
        self.assertIn("cardStore", doc)
        self.assertIsNone(doc["cardStore"])

    def test_text_output_has_one_header_line_naming_the_store(self):
        code, stdout, stderr = self.run_main(self.argv())
        self.assertEqual(code, 0, stderr)
        header = (
            f"card store: sha256={self.digest} "
            f"generated_at=2026-01-01T00:00:00Z cards={self.n_cards}"
        )
        lines = stdout.splitlines()
        self.assertEqual(lines[0], header)
        self.assertEqual(stdout.count("card store:"), 1)
        self.assertTrue(lines[1].startswith("1. "), lines[1])

    def test_text_header_says_none_when_default_store_absent(self):
        _, stdout, _ = self.run_main(self.argv(cards="default"))
        self.assertEqual(stdout.splitlines()[0], "card store: none")

    def test_matching_pin_ranks_normally_case_insensitively(self):
        for pin in (self.digest, self.digest.upper()):
            with self.subTest(pin=pin[:8]):
                code, stdout, stderr = self.run_main(
                    self.argv("--quality-cards-sha256", pin, "--json")
                )
                self.assertEqual(code, 0, stderr)
                self.assertEqual(json.loads(stdout)["rows"][0]["status"], "recommended")

    def test_pin_mismatch_refuses_exit_3_naming_both_digests(self):
        wrong = "0" * 64
        code, stdout, stderr = self.run_main(
            self.argv("--quality-cards-sha256", wrong, "--json")
        )
        self.assertEqual(code, 3, stderr)
        self.assertEqual(stdout, "")
        self.assertIn(wrong, stderr)
        self.assertIn(self.digest, stderr)

    def test_malformed_pin_refuses_exit_3_not_argparse_exit_2(self):
        for bad in ("xyz", "", "g" * 64, self.digest[:63], self.digest + "0"):
            with self.subTest(pin=bad):
                code, stdout, stderr = self.run_main(
                    self.argv("--quality-cards-sha256", bad, "--json")
                )
                self.assertEqual(code, 3, stderr)
                self.assertEqual(stdout, "")
                self.assertIn("--quality-cards-sha256", stderr)
                self.assertNotIn("usage:", stderr)

    def test_pin_with_missing_default_store_refuses_exit_3(self):
        code, stdout, stderr = self.run_main(
            self.argv("--quality-cards-sha256", self.digest, "--json", cards="default")
        )
        self.assertEqual(code, 3, stderr)
        self.assertEqual(stdout, "")

    def test_pin_matching_non_manifest_bytes_refuses_exit_3(self):
        not_a_manifest = b'{"schema": 1}'
        self.manifest_path.write_bytes(not_a_manifest)
        code, stdout, stderr = self.run_main(
            self.argv(
                "--quality-cards-sha256",
                hashlib.sha256(not_a_manifest).hexdigest(),
                "--json",
            )
        )
        self.assertEqual(code, 3, stderr)
        self.assertEqual(stdout, "")
        self.assertIn("not a quality-card manifest", stderr)

    def test_one_changed_byte_refuses_under_the_original_pin(self):
        flipped = bytearray(self.manifest_bytes)
        flipped[flipped.index(b"2026-01-01") + 3] ^= 0x01
        json.loads(bytes(flipped))
        self.manifest_path.write_bytes(bytes(flipped))
        code, stdout, stderr = self.run_main(
            self.argv("--quality-cards-sha256", self.digest, "--json")
        )
        self.assertEqual(code, 3, stderr)
        self.assertEqual(stdout, "")
        self.assertIn(hashlib.sha256(bytes(flipped)).hexdigest(), stderr)


# ---------------------------------------------------------------------
# `recommend` resolves a pulled card store by default, exactly like `serve`.
# Predeclaration rows D1-D8:
# docs/task-inbox/2026-10-03-PREDECLARATION-serve-and-recommend-resolve-a-pulled-card-store-by-default.md
# ---------------------------------------------------------------------
PULLED_NEWER = "2026-06-01T00:00:00Z"
PULLED_OLDER = "2025-12-01T00:00:00Z"
PULLED_ADDED_ID = "pulled-no-go@test"
PULLED_ADDED_REPO = "example/PulledNoGoModel"
BUNDLED_GENERATED_AT = "2026-01-01T00:00:00Z"


def pulled_added_card(card_id=PULLED_ADDED_ID, repo=PULLED_ADDED_REPO, verdict="NO_GO") -> dict:
    return {
        "id": card_id,
        "model": {"repo": repo, "hfPin": "5eed1234"},
        "verdict": verdict,
        "legible": {"tier": "Noticeable", "headline": "Pulled card headline."},
    }


def pulled_store_bytes(generated_at=PULLED_NEWER, extra=None, mutate=None) -> bytes:
    document = fixture_manifest()
    document["generatedAt"] = generated_at
    document["cards"] = document["cards"] + (
        [pulled_added_card()] if extra is None else list(extra)
    )
    if mutate is not None:
        mutate(document)
    return json.dumps(document).encode("utf-8")


_MODULE_STATE = {}
_REAL_PULLED_CARDS_DIR = FASTMLX_RECOMMEND.launch.pulled_cards_dir


def setUpModule():
    # Hermeticity (row D11): no test in this module may read the real
    # ``~/.fastmlx/cards``. The D tests re-patch this to their own directory.
    tmp = tempfile.TemporaryDirectory()
    empty = Path(tmp.name) / "no-pulled-cards"
    patcher = patch.object(FASTMLX_RECOMMEND.launch, "pulled_cards_dir", lambda: empty)
    patcher.start()
    _MODULE_STATE["tmp"], _MODULE_STATE["patcher"] = tmp, patcher


def tearDownModule():
    _MODULE_STATE["patcher"].stop()
    _MODULE_STATE["tmp"].cleanup()


class RecommendPulledCardStoreTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

        self.fit_marker = self.root / "fit-ran"
        self.fit_bin = write_script(
            self.root / "fit.py",
            GREEN_FIT_CHECK_BODY.replace(
                "import sys\n", f"import sys\nopen({str(self.fit_marker)!r}, 'w').close()\n", 1
            ),
        )
        self.model_dir = self.root / "pass-model"
        self.model_dir.mkdir()
        (self.model_dir / "config.json").write_text("{}", encoding="utf-8")
        write_pull_receipt(self.model_dir, repo_id=PASS_REPO, revision="e" * 40)
        self.added_dir = self.root / "added-model"
        self.added_dir.mkdir()
        (self.added_dir / "config.json").write_text("{}", encoding="utf-8")
        write_pull_receipt(self.added_dir, repo_id=PULLED_ADDED_REPO, revision="f" * 40)

        self.fake_repo_root = self.root / "repo"
        (self.fake_repo_root / "site").mkdir(parents=True)
        self.bundled_path = self.fake_repo_root / "site" / "quality-guides.json"
        self.bundled_bytes = json.dumps(fixture_manifest()).encode("utf-8")
        self.bundled_path.write_bytes(self.bundled_bytes)
        self.bundled_digest = hashlib.sha256(self.bundled_bytes).hexdigest()
        self.n_cards = len(fixture_manifest()["cards"])

        self.pulled = self.root / "pulled"  # not created: D1 starts with no directory

    def write_pulled(self, raw: bytes, name: str = None) -> Path:
        self.pulled.mkdir(exist_ok=True)
        path = self.pulled / (name if name is not None else f"{hashlib.sha256(raw).hexdigest()}.json")
        path.write_bytes(raw)
        return path

    def argv(self, *extra, models=None) -> list:
        argv = ["recommend", "--fit-check-bin", str(self.fit_bin)]
        for model in (models if models is not None else [self.model_dir]):
            argv += ["--model-path", str(model)]
        return argv + list(extra)

    def run_main(self, argv: list, pulled_dir=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        pulled_dir = self.pulled if pulled_dir is None else pulled_dir
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with patch.object(FASTMLX_RECOMMEND.launch, "REPO_ROOT", self.fake_repo_root), patch.object(
                FASTMLX_RECOMMEND.launch, "pulled_cards_dir", lambda: pulled_dir
            ):
                with self.assertRaises(SystemExit) as ctx:
                    FASTMLX_RECOMMEND.main(argv)
        return ctx.exception.code, stdout.getvalue(), stderr.getvalue()

    def assert_refused_before_fit(self, code, stdout, stderr):
        self.assertEqual(code, 3, stderr)
        self.assertEqual(stdout, "")
        self.assertFalse(self.fit_marker.exists(), "the fit-check ran before the refusal")

    # --- D1 ---------------------------------------------------------------
    def test_d1_missing_and_empty_directory_are_the_bundled_store_unchanged(self):
        results = []
        for create in (False, True):
            if create:
                self.pulled.mkdir()
            code, stdout, stderr = self.run_main(self.argv("--json"))
            self.assertEqual(code, 0, stderr)
            doc = json.loads(stdout)
            self.assertEqual(
                doc["cardStore"],
                {"sha256": self.bundled_digest, "generatedAt": BUNDLED_GENERATED_AT,
                 "cards": self.n_cards},
            )
            self.assertNotIn("skipping", stderr)
            code_t, text, stderr_t = self.run_main(self.argv())
            self.assertEqual(
                text.splitlines()[0],
                f"card store: sha256={self.bundled_digest} "
                f"generated_at={BUNDLED_GENERATED_AT} cards={self.n_cards}",
            )
            results.append((code, stdout, stderr, code_t, text, stderr_t))
        self.assertEqual(results[0], results[1])

    # --- D2 ---------------------------------------------------------------
    def test_d2_newer_store_is_ranked_against_and_named(self):
        raw = pulled_store_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        self.write_pulled(raw)
        # Control: with no pulled store the pack is uncarded.
        code, stdout, stderr = self.run_main(
            self.argv("--json", models=[self.added_dir]), pulled_dir=self.root / "none"
        )
        self.assertEqual(json.loads(stdout)["rows"][0]["status"], "uncarded", stderr)
        code, stdout, stderr = self.run_main(self.argv("--json", models=[self.added_dir]))
        doc = json.loads(stdout)
        self.assertEqual(doc["cardStore"]["sha256"], digest)
        self.assertEqual(doc["cardStore"]["cards"], self.n_cards + 1)
        row = doc["rows"][0]
        self.assertEqual(row["status"], "opt-in", stderr)
        self.assertEqual(row["accept_quality_flag"], f"--accept-quality {PULLED_ADDED_ID}")
        _, text, _ = self.run_main(self.argv(models=[self.added_dir]))
        self.assertEqual(
            text.splitlines()[0],
            f"card store: sha256={digest} generated_at={PULLED_NEWER} cards={self.n_cards + 1}",
        )

    # --- D3 ---------------------------------------------------------------
    def test_d3_an_older_store_is_skipped_with_exactly_one_notice(self):
        older = self.write_pulled(pulled_store_bytes(PULLED_OLDER))
        self.write_pulled(self.bundled_bytes)  # bundled-identical: silent
        code, stdout, stderr = self.run_main(self.argv("--json"))
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["cardStore"]["sha256"], self.bundled_digest)
        self.assertEqual(
            [l for l in stderr.splitlines() if "skipping" in l],
            [
                f"fastmlx recommend: skipping pulled card store {older}: generatedAt "
                f"{PULLED_OLDER} is older than the bundled store's {BUNDLED_GENERATED_AT}"
            ],
        )

    # --- D4 ---------------------------------------------------------------
    def test_d4_a_name_hash_mismatch_refuses_naming_the_file_without_fallback(self):
        bad = self.write_pulled(pulled_store_bytes(), name="a" * 64 + ".json")
        code, stdout, stderr = self.run_main(self.argv("--json"))
        self.assert_refused_before_fit(code, stdout, stderr)
        self.assertIn(str(bad), stderr)

    def test_d4_an_unreadable_candidate_refuses_naming_the_file(self):
        path = self.write_pulled(pulled_store_bytes())
        path.chmod(0)
        self.addCleanup(path.chmod, 0o644)
        if os.access(path, os.R_OK):
            self.skipTest("running as a user that can read a mode-000 file")
        code, stdout, stderr = self.run_main(self.argv("--json"))
        self.assert_refused_before_fit(code, stdout, stderr)
        self.assertIn(str(path), stderr)

    # --- D5 ---------------------------------------------------------------
    def test_d5_rule_violations_refuse_with_the_rule_and_the_remedy(self):
        def drop(document):
            document["cards"] = [c for c in document["cards"] if c["id"] != NO_GO_CARD_ID]

        def relax(document):
            for card in document["cards"]:
                if card["id"] == NO_GO_CARD_ID:
                    card["verdict"] = "PASS"

        for mutate, rule_text in (
            (drop, f"R3 drops bundled card id(s): {NO_GO_CARD_ID}"),
            (relax, f"R6 card {NO_GO_CARD_ID} relaxes verdict NO_GO -> PASS"),
        ):
            with self.subTest(rule=rule_text[:2]):
                path = self.write_pulled(pulled_store_bytes(mutate=mutate))
                code, stdout, stderr = self.run_main(self.argv("--json"))
                self.assert_refused_before_fit(code, stdout, stderr)
                self.assertIn(rule_text, stderr)
                self.assertIn(
                    f"pass --quality-cards {self.bundled_path} to use the bundled store, "
                    f"or remove {path}",
                    stderr,
                )
                path.unlink()

    def test_d5_control_the_same_store_without_the_violation_is_used(self):
        raw = pulled_store_bytes()
        self.write_pulled(raw)
        code, stdout, stderr = self.run_main(self.argv("--json"))
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["cardStore"]["sha256"], hashlib.sha256(raw).hexdigest())

    # --- D6 ---------------------------------------------------------------
    def test_d6_greatest_generated_at_wins_and_equal_is_ambiguous(self):
        older = pulled_store_bytes("2026-05-01T00:00:00Z", extra=[pulled_added_card("a@test", "example/A")])
        newer = pulled_store_bytes("2026-07-01T00:00:00Z", extra=[pulled_added_card("b@test", "example/B")])
        self.write_pulled(older)
        self.write_pulled(newer)
        code, stdout, stderr = self.run_main(self.argv("--json"))
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["cardStore"]["sha256"], hashlib.sha256(newer).hexdigest())
        twin = pulled_store_bytes("2026-07-01T00:00:00Z", extra=[pulled_added_card("c@test", "example/C")])
        self.write_pulled(twin)
        self.fit_marker.unlink()  # the run above reached the fit-check
        code, stdout, stderr = self.run_main(self.argv("--json"))
        self.assert_refused_before_fit(code, stdout, stderr)
        self.assertIn("ambiguous", stderr)

    # --- D7 ---------------------------------------------------------------
    def test_d7_an_explicit_store_never_reads_the_directory(self):
        self.write_pulled(b"corrupt", name="c" * 64 + ".json")
        explicit = self.root / "explicit.json"
        explicit.write_bytes(self.bundled_bytes)

        def forbidden():
            raise AssertionError("the pulled-cards directory was read")

        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with patch.object(FASTMLX_RECOMMEND.launch, "REPO_ROOT", self.fake_repo_root), patch.object(
                FASTMLX_RECOMMEND.launch, "pulled_cards_dir", forbidden
            ):
                with self.assertRaises(SystemExit) as ctx:
                    FASTMLX_RECOMMEND.main(self.argv("--quality-cards", str(explicit), "--json"))
        self.assertEqual(ctx.exception.code, 0, stderr.getvalue())
        self.assertEqual(json.loads(stdout.getvalue())["cardStore"]["sha256"], self.bundled_digest)
        code, out, err = self.run_main(self.argv("--quality-cards", str(explicit), "--json"))
        self.assertEqual(code, 0, err)

    # --- D8 ---------------------------------------------------------------
    def test_d8_other_entries_are_ignored_silently(self):
        valid_raw = pulled_store_bytes()
        valid_digest = hashlib.sha256(valid_raw).hexdigest()
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "target.json").write_bytes(valid_raw)
        self.pulled.mkdir()
        (self.pulled / ".fastmlx-cards-abc123.tmp").write_bytes(b"half written")
        (self.pulled / "notes.json").write_bytes(b"{}")
        (self.pulled / ("e" * 64 + ".json")).mkdir()
        os.symlink(elsewhere / "target.json", self.pulled / f"{valid_digest}.json")
        code, stdout, stderr = self.run_main(self.argv("--json"))
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["cardStore"]["sha256"], self.bundled_digest)
        self.assertNotIn("skipping", stderr)

    # --- D9 (recommend side) ----------------------------------------------
    def test_d9_pin_alone_pins_the_resolved_store(self):
        raw = pulled_store_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        self.write_pulled(raw)
        code, stdout, stderr = self.run_main(self.argv("--quality-cards-sha256", digest, "--json"))
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["cardStore"]["sha256"], digest)
        self.fit_marker.unlink()  # the run above reached the fit-check
        code, stdout, stderr = self.run_main(
            self.argv("--quality-cards-sha256", self.bundled_digest, "--json")
        )
        self.assert_refused_before_fit(code, stdout, stderr)
        self.assertIn(self.bundled_digest, stderr)
        self.assertIn(digest, stderr)

    # --- review follow-ups ---------------------------------------------------
    def test_a_a_store_swapped_after_verification_is_refused_before_the_fit_check(self):
        path = self.write_pulled(pulled_store_bytes())
        real = FASTMLX_RECOMMEND.launch.resolve_quality_card_store
        evil = pulled_store_bytes(extra=[pulled_added_card("evil@test", "example/Evil")])

        def resolve_then_swap(*args, **kwargs):
            result = real(*args, **kwargs)
            path.write_bytes(evil)
            return result

        with patch.object(FASTMLX_RECOMMEND.launch, "resolve_quality_card_store", resolve_then_swap):
            code, stdout, stderr = self.run_main(self.argv("--json"))
        self.assert_refused_before_fit(code, stdout, stderr)
        self.assertIn(f"pulled card store {path} changed after it was verified", stderr)

    def test_b1_a_sole_store_with_an_unparsable_generated_at_is_refused_by_r2(self):
        path = self.write_pulled(pulled_store_bytes("not-a-date"))
        code, stdout, stderr = self.run_main(self.argv("--json"))
        self.assert_refused_before_fit(code, stdout, stderr)
        self.assertIn("R2 generatedAt 'not-a-date' is not strictly", stderr)
        self.assertIn(str(path), stderr)

    def test_b2_an_unparsable_store_beside_a_parsable_newer_one_selects_the_newer(self):
        self.write_pulled(pulled_store_bytes("not-a-date"))
        good = pulled_store_bytes("2026-05-01T00:00:00Z")
        self.write_pulled(good)
        code, stdout, stderr = self.run_main(self.argv("--json"))
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["cardStore"]["sha256"], hashlib.sha256(good).hexdigest())

    def test_b3_the_bundled_generated_at_with_different_bytes_is_refused_by_r2(self):
        self.write_pulled(pulled_store_bytes(BUNDLED_GENERATED_AT))
        code, stdout, stderr = self.run_main(self.argv("--json"))
        self.assert_refused_before_fit(code, stdout, stderr)
        self.assertIn("equals the bundled store's but the bytes differ", stderr)
        self.assertNotIn("skipping", stderr)

    def test_b4_a_future_dated_store_is_refused_by_r2_with_no_fallback(self):
        self.write_pulled(pulled_store_bytes("2099-01-01T00:00:00Z"))
        code, stdout, stderr = self.run_main(self.argv("--json"))
        self.assert_refused_before_fit(code, stdout, stderr)
        self.assertIn("R2 generatedAt 2099-01-01T00:00:00Z is in the future", stderr)

    def test_c_a_hash_mismatched_store_beside_a_good_newer_store_refuses_naming_it(self):
        self.write_pulled(pulled_store_bytes("2026-07-01T00:00:00Z"))
        bad = self.write_pulled(
            pulled_store_bytes("2026-05-01T00:00:00Z"), name="b" * 64 + ".json"
        )
        code, stdout, stderr = self.run_main(self.argv("--json"))
        self.assert_refused_before_fit(code, stdout, stderr)
        self.assertIn(str(bad), stderr)
        self.assertIn("does not match its file name", stderr)

    def test_d_an_lstat_permission_error_refuses_and_file_not_found_is_skipped(self):
        raw = pulled_store_bytes()
        victim = self.write_pulled(raw)
        real = os.lstat

        def raising(error):
            def lstat(path, *args, **kwargs):
                if str(path) == str(victim):
                    raise error
                return real(path, *args, **kwargs)

            return patch.object(FASTMLX_RECOMMEND.launch.os, "lstat", lstat)

        with raising(PermissionError(13, "Permission denied")):
            code, stdout, stderr = self.run_main(self.argv("--json"))
        self.assert_refused_before_fit(code, stdout, stderr)
        self.assertIn(str(victim), stderr)
        with raising(FileNotFoundError(2, "No such file")):
            code, stdout, stderr = self.run_main(self.argv("--json"))
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["cardStore"]["sha256"], self.bundled_digest)

    def test_e_no_home_directory_resolves_the_bundled_store(self):
        self.write_pulled(b"corrupt", name="c" * 64 + ".json")  # must never be reached
        real_dir = _REAL_PULLED_CARDS_DIR

        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with patch.object(FASTMLX_RECOMMEND.launch, "REPO_ROOT", self.fake_repo_root), patch.object(
                FASTMLX_RECOMMEND.launch, "pulled_cards_dir", real_dir
            ), patch.object(FASTMLX_RECOMMEND.launch.Path, "home", side_effect=RuntimeError("no home")):
                with self.assertRaises(SystemExit) as ctx:
                    FASTMLX_RECOMMEND.main(self.argv("--json"))
        self.assertEqual(ctx.exception.code, 0, stderr.getvalue())
        self.assertEqual(json.loads(stdout.getvalue())["cardStore"]["sha256"], self.bundled_digest)



# ---------------------------------------------------------------------
# `recommend --pinned <repo>@<sha>`: a fit + card verdict from the Hugging
# Face revision manifest, BEFORE any download. Standalone TestCase (see the
# comment on BuiltinSizerAutoSelectionTestCase above for why).
# ---------------------------------------------------------------------
PINNED_SHA = "a" * 40
PINNED_REF = f"{PASS_REPO}@{PINNED_SHA}"
PINNED_SIZER_ARGS = [
    "--fit-check-arg=--wired-limit-mib",
    "--fit-check-arg",
    "4096",
    "--fit-check-arg=--wired-margin-gib",
    "--fit-check-arg",
    "2",
]


def manifest_document(repo: str, sha: str, files: dict) -> dict:
    """A revision-API document shaped like Hugging Face's, ``files`` being
    ``{relative_path: size}`` (the real ``validated_entries`` validates it)."""
    return {
        "id": repo,
        "sha": sha,
        "private": False,
        "siblings": [
            {"rfilename": name, "size": size, "blobId": "b" * 40}
            for name, size in files.items()
        ],
    }


class RecommendPinnedTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest_path = self.root / "quality-guides.json"
        self.manifest_path.write_text(json.dumps(fixture_manifest()), encoding="utf-8")
        self.fetch_calls = []
        self.documents = {}
        patcher = patch.object(
            FASTMLX_RECOMMEND.downloader, "fetch_api", side_effect=self._fake_fetch
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _fake_fetch(self, repo_id, revision):
        self.fetch_calls.append((repo_id, revision))
        document = self.documents.get((repo_id, revision))
        if isinstance(document, Exception):
            raise document
        return document, b""

    def serve_manifest(self, repo: str, sha: str, files: dict) -> str:
        self.documents[(repo, sha)] = manifest_document(repo, sha, files)
        return f"{repo}@{sha}"

    def argv(self, refs, *extra, kv="0.5", sizer_args=True) -> list:
        argv = ["recommend", "--quality-cards", str(self.manifest_path)]
        for ref in refs:
            argv += ["--pinned", ref]
        if kv is not None:
            argv += ["--kv-reserve-gib", kv]
        if sizer_args:
            argv += PINNED_SIZER_ARGS
        return argv + list(extra)

    def run_main(self, argv: list):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                FASTMLX_RECOMMEND.main(argv)
        return ctx.exception.code, stdout.getvalue(), stderr.getvalue()

    def run_json(self, argv: list):
        code, stdout, stderr = self.run_main(argv + ["--json"])
        return code, json.loads(stdout), stderr

    def small_pack(self, repo=PASS_REPO, sha=PINNED_SHA, **overrides) -> str:
        files = {
            "config.json": 10,
            "model-00001-of-00002.safetensors": 1000,
            "model-00002-of-00002.safetensors": 2000,
        }
        files.update(overrides)
        return self.serve_manifest(repo, sha, files)

    # --- A1 --------------------------------------------------------
    def test_pinned_row_statuses_and_marking(self):
        ref = self.small_pack()
        code, doc, _ = self.run_json(self.argv([ref]))
        self.assertEqual(code, 0, doc)
        row = doc["rows"][0]
        self.assertEqual(row["status"], "recommended")
        self.assertEqual(row["source"], "revision-manifest")
        self.assertIs(row["downloaded"], False)
        self.assertEqual(row["ref"], ref)
        self.assertEqual(row["repo"], PASS_REPO)
        self.assertEqual(row["revision"], PINNED_SHA)
        self.assertEqual(row["fit"]["verdict"], "GREEN")
        self.assertEqual(row["card"]["id"], PASS_CARD_ID)
        self.assertEqual(self.fetch_calls, [(PASS_REPO, PINNED_SHA)])

    def test_pinned_text_says_not_downloaded_and_prints_pull_next_step(self):
        ref = self.small_pack()
        code, stdout, _ = self.run_main(self.argv([ref]))
        self.assertEqual(code, 0)
        self.assertIn("not downloaded", stdout)
        self.assertIn(f"fastmlx pull {ref} --dest <dir>", stdout)
        self.assertNotIn("--accept-quality", stdout)

    def test_pinned_pack_over_the_ceiling_is_does_not_fit(self):
        ref = self.small_pack(**{"model-00001-of-00002.safetensors": 3 << 30})
        code, doc, _ = self.run_json(self.argv([ref]))
        self.assertEqual(code, 2)
        row = doc["rows"][0]
        self.assertEqual(row["status"], "does-not-fit")
        self.assertEqual(row["fit"]["verdict"], "RED")
        self.assertIsNone(row["next_step"])
        code, stdout, _ = self.run_main(self.argv([ref]))
        self.assertIn("not downloaded", stdout)
        self.assertNotIn("fastmlx pull", stdout)

    def test_pinned_uncarded_row(self):
        ref = self.small_pack(repo=UNCARDED_REPO)
        code, doc, _ = self.run_json(self.argv([ref]))
        self.assertEqual(code, 1)
        self.assertEqual(doc["rows"][0]["status"], "uncarded")
        self.assertEqual(doc["rows"][0]["source"], "revision-manifest")

    def test_pinned_row_combines_with_a_local_candidate_and_ranks_with_it(self):
        ref = self.small_pack()
        local = self.root / "local-model"
        local.mkdir()
        (local / "config.json").write_text("{}", encoding="utf-8")
        blob = build_safetensors_bytes([("t.a", "F32", [4], zero_tensor_bytes("F32", [4]))])
        (local / "model.safetensors").write_bytes(blob)
        argv = self.argv([ref]) + ["--model-path", str(local)]
        code, doc, _ = self.run_json(argv)
        self.assertEqual(code, 0, doc)
        by_source = {row.get("source"): row for row in doc["rows"]}
        self.assertEqual(by_source["revision-manifest"]["status"], "recommended")
        self.assertEqual(by_source[None]["status"], "uncarded")
        self.assertEqual(doc["rows"][0]["source"], "revision-manifest")  # recommended first

    # --- A2 (differential) ------------------------------------------
    def _local_fixture_dir(self) -> Path:
        pack = self.root / "local-diff-pack"
        pack.mkdir()
        (pack / "config.json").write_text("{}", encoding="utf-8")
        (pack / "tokenizer.json").write_text('{"a": 1}', encoding="utf-8")
        for index, count in enumerate((4, 9), start=1):
            blob = build_safetensors_bytes(
                [("t.a", "F32", [count], zero_tensor_bytes("F32", [count]))]
            )
            (pack / f"model-0000{index}-of-00002.safetensors").write_bytes(blob)
        (pack / ".hidden").mkdir()
        (pack / ".hidden" / "stray.safetensors").write_bytes(
            build_safetensors_bytes([("t.z", "F32", [64], zero_tensor_bytes("F32", [64]))])
        )
        return pack

    def _manifest_files_of(self, pack: Path) -> dict:
        return {
            path.relative_to(pack).as_posix(): path.stat().st_size
            for path in sorted(pack.rglob("*"))
            if path.is_file()
        }

    def _local_compute_fit(self, pack: Path, kv: str) -> dict:
        sizer = FASTMLX_RECOMMEND.launch._safetensors_fit
        args = sizer.build_arg_parser().parse_args(
            [
                "--model-path", str(pack),
                "--kv-reserve-gib", kv,
                "--wired-limit-mib", "4096",
                "--wired-margin-gib", "2",
            ]
        )
        return sizer.compute_fit(args)

    def test_pinned_row_matches_local_fit_verdict(self):
        pack = self._local_fixture_dir()
        files = self._manifest_files_of(pack)
        ref = self.serve_manifest(PASS_REPO, PINNED_SHA, files)
        for kv, expected_fit in (("0.5", "green"), ("3", "red")):
            with self.subTest(kv=kv):
                local = self._local_compute_fit(pack, kv)
                self.assertEqual(local["fit"], expected_fit)
                code, doc, _ = self.run_json(self.argv([ref], kv=kv))
                row = doc["rows"][0]
                self.assertEqual(row["sizing"]["weights_bytes"], local["weights_bytes"])
                self.assertEqual(row["sizing"]["ceiling_bytes"], local["ceiling_bytes"])
                self.assertEqual(row["sizing"]["total_bytes"], local["total_bytes"])
                self.assertEqual(row["fit"]["verdict"], local["fit"].upper())
                self.assertEqual(
                    row["status"], "recommended" if expected_fit == "green" else "does-not-fit"
                )
        # the .hidden shard is excluded on both sides, and sizes are non-trivial
        self.assertGreater(local["weights_bytes"], 0)

    # --- A3 ---------------------------------------------------------
    def test_invalid_refs_are_usage_errors_before_any_fetch(self):
        bad_refs = (
            f"{PASS_REPO}@main",
            f"{PASS_REPO}@{'a' * 12}",
            f"{PASS_REPO}@{'A' * 40}",
            f"{PASS_REPO}",
            f"{PASS_REPO}@{'a' * 40}@{'b' * 40}",
            f"noslash@{'a' * 40}",
        )
        good = self.small_pack()
        for bad in bad_refs:
            with self.subTest(ref=bad):
                self.fetch_calls.clear()
                code, stdout, stderr = self.run_main(self.argv([good, bad]))
                self.assertEqual(code, 2)
                self.assertEqual(stdout, "")
                self.assertIn("fastmlx recommend", stderr)
                self.assertIn(bad, stderr)
                self.assertEqual(self.fetch_calls, [])

    # --- A4 ---------------------------------------------------------
    def _assert_error_row_with_other_rows_unaffected(self, bad_ref, argv_extra=(), kv="0.5"):
        good = self.small_pack(repo=PASS_REPO, sha="c" * 40)
        code, doc, _ = self.run_json(self.argv([good, bad_ref], *argv_extra, kv=kv))
        rows = {row["ref"]: row for row in doc["rows"]}
        self.assertEqual(rows[good]["status"], "recommended")
        self.assertEqual(rows[bad_ref]["status"], "error")
        self.assertEqual(code, 0)
        return rows[bad_ref]

    def test_error_row_when_the_manifest_fetch_raises(self):
        bad = f"{NO_GO_REPO}@{'d' * 40}"
        self.documents[(NO_GO_REPO, "d" * 40)] = RuntimeError("network down")
        row = self._assert_error_row_with_other_rows_unaffected(bad)
        self.assertIn("revision manifest", row["message"])
        self.assertIn("network down", row["message"])
        self.assertEqual(row["source"], "revision-manifest")
        self.assertIs(row["downloaded"], False)

    def test_error_row_when_the_manifest_identity_does_not_validate(self):
        bad = f"{NO_GO_REPO}@{'d' * 40}"
        self.documents[(NO_GO_REPO, "d" * 40)] = manifest_document(
            NO_GO_REPO, "e" * 40, {"model.safetensors": 5}
        )
        row = self._assert_error_row_with_other_rows_unaffected(bad)
        self.assertIn("identity mismatch", row["message"])

    def test_error_row_when_the_manifest_has_no_safetensors(self):
        bad = self.serve_manifest(NO_GO_REPO, "d" * 40, {"config.json": 5, "README.md": 9})
        row = self._assert_error_row_with_other_rows_unaffected(bad)
        self.assertIn("no .safetensors files", row["message"])

    def test_error_row_for_an_unnamed_big_non_safetensors_file(self):
        bad = self.serve_manifest(
            NO_GO_REPO,
            "d" * 40,
            {"model.safetensors": 100, "model-q4.gguf": 2 << 30},
        )
        row = self._assert_error_row_with_other_rows_unaffected(bad)
        self.assertIn("model-q4.gguf", row["message"])
        self.assertIn("--mmap-side-file", row["message"])

    def test_big_file_named_by_mmap_side_file_is_not_resident(self):
        ref = self.serve_manifest(
            PASS_REPO, PINNED_SHA, {"model.safetensors": 100, "ngram.bin": 2 << 30}
        )
        code, doc, _ = self.run_json(
            self.argv([ref], "--fit-check-arg=--mmap-side-file", "--fit-check-arg", "ngram.bin")
        )
        self.assertEqual(doc["rows"][0]["status"], "recommended", doc)
        self.assertEqual(doc["rows"][0]["sizing"]["weights_bytes"], 100)

    def test_error_row_for_expert_stream_residency(self):
        ref = self.small_pack()
        code, doc, _ = self.run_json(self.argv([ref], "--residency", "expert-stream"))
        self.assertEqual(code, 2)
        row = doc["rows"][0]
        self.assertEqual(row["status"], "error")
        self.assertIn("expert-stream", row["message"])
        self.assertEqual(self.fetch_calls, [])

    def test_error_row_for_an_explicit_fit_check_bin(self):
        ref = self.small_pack()
        code, doc, _ = self.run_json(self.argv([ref], "--fit-check-bin", "/bin/true"))
        row = doc["rows"][0]
        self.assertEqual(row["status"], "error")
        self.assertIn("--fit-check-bin", row["message"])
        self.assertEqual(self.fetch_calls, [])

    def test_error_row_for_a_profile_fit_check_bin(self):
        ref = self.small_pack()
        profile = FASTMLX_RECOMMEND.launch.load_engine_profile(None)[0]
        profile = dict(profile, fitCheck={"bin": "/bin/true", "args": []})
        with patch.object(
            FASTMLX_RECOMMEND.launch, "load_engine_profile", return_value=(profile, False)
        ):
            code, doc, _ = self.run_json(self.argv([ref]))
        row = doc["rows"][0]
        self.assertEqual(row["status"], "error")
        self.assertIn("fitCheck.bin", row["message"])
        self.assertEqual(self.fetch_calls, [])

    def test_error_row_for_a_missing_kv_reserve_matches_the_local_message(self):
        ref = self.small_pack()
        code, doc, _ = self.run_json(self.argv([ref], kv=None))
        self.assertEqual(code, 2)
        row = doc["rows"][0]
        self.assertEqual(row["status"], "error")
        self.assertIn("requires --kv-reserve-gib", row["message"])
        self.assertIn("never silently assume a zero KV-cache reserve", row["message"])
        self.assertEqual(self.fetch_calls, [])
        # the same sentence the local auto-selected-sizer refusal prints
        local = self.root / "kv-local"
        local.mkdir()
        (local / "config.json").write_text("{}", encoding="utf-8")
        (local / "model.safetensors").write_bytes(
            build_safetensors_bytes([("t.a", "F32", [4], zero_tensor_bytes("F32", [4]))])
        )
        _, local_doc, _ = self.run_json(
            ["recommend", "--quality-cards", str(self.manifest_path), "--model-path", str(local)]
        )
        self.assertEqual(local_doc["rows"][0]["message"], row["message"])

    def test_a_bad_fit_check_arg_is_an_error_row_not_a_crash(self):
        ref = self.small_pack()
        code, doc, _ = self.run_json(
            self.argv([ref], "--fit-check-arg=--no-such-sizer-flag", sizer_args=False)
        )
        self.assertEqual(doc["rows"][0]["status"], "error")

    # --- A5 ---------------------------------------------------------
    def test_dot_prefixed_manifest_paths_are_excluded(self):
        ref = self.serve_manifest(
            PASS_REPO,
            PINNED_SHA,
            {
                "model.safetensors": 1000,
                ".hidden/shard.safetensors": 5 << 30,
                "sub/.inner/shard.safetensors": 5 << 30,
                ".gitattributes.safetensors": 5 << 30,
                ".big-dotfile.bin": 5 << 30,
            },
        )
        code, doc, _ = self.run_json(self.argv([ref]))
        row = doc["rows"][0]
        self.assertEqual(row["status"], "recommended", row)
        self.assertEqual(row["sizing"]["weights_bytes"], 1000)

    def test_only_dot_prefixed_safetensors_is_the_no_safetensors_error(self):
        ref = self.serve_manifest(
            PASS_REPO, PINNED_SHA, {"config.json": 3, ".hidden/model.safetensors": 10}
        )
        _, doc, _ = self.run_json(self.argv([ref]))
        self.assertEqual(doc["rows"][0]["status"], "error")

    # --- A6 ---------------------------------------------------------
    def test_card_lookup_by_repo_and_by_hf_pin(self):
        by_repo = self.small_pack(repo=NO_GO_REPO, sha="1" * 40)
        by_pin = self.small_pack(repo="example/Unrelated", sha=PIN_ONLY_NO_GO_REVISION)
        _, doc, _ = self.run_json(self.argv([by_repo, by_pin]))
        rows = {row["ref"]: row for row in doc["rows"]}
        self.assertEqual(rows[by_repo]["card"]["id"], NO_GO_CARD_ID)
        self.assertEqual(rows[by_pin]["card"]["id"], PIN_ONLY_NO_GO_CARD_ID)

    def test_no_go_card_is_an_opt_in_row_with_the_accept_quality_next_step(self):
        ref = self.small_pack(repo=NO_GO_REPO, sha="1" * 40)
        code, doc, _ = self.run_json(self.argv([ref]))
        self.assertEqual(code, 1)
        row = doc["rows"][0]
        self.assertEqual(row["status"], "opt-in")
        self.assertEqual(row["accept_quality_flag"], f"--accept-quality {NO_GO_CARD_ID}")
        self.assertEqual(row["source"], "revision-manifest")
        self.assertIs(row["downloaded"], False)
        self.assertIn(f"fastmlx pull {ref} --dest <dir>", row["next_step"])
        self.assertIn(f"--accept-quality {NO_GO_CARD_ID}", row["next_step"])
        _, stdout, _ = self.run_main(self.argv([ref]))
        self.assertIn(f"fastmlx pull {ref} --dest <dir>", stdout)
        self.assertIn(f"--accept-quality {NO_GO_CARD_ID}", stdout)
        self.assertIn("not downloaded", stdout)

    def test_exit_code_for_is_unchanged_for_pinned_rows(self):
        statuses = {
            "recommended": 0,
            "opt-in": 1,
            "uncarded": 1,
            "does-not-fit": 2,
            "error": 2,
        }
        for status, expected in statuses.items():
            rows = [{"status": status, "source": "revision-manifest"}]
            self.assertEqual(FASTMLX_RECOMMEND.exit_code_for(rows), expected)

    # --- no disk writes ---------------------------------------------
    def test_pinned_path_writes_nothing_to_disk(self):
        ref = self.small_pack()
        before = sorted(p.name for p in self.root.rglob("*"))
        cwd_before = sorted(os.listdir("."))
        self.run_json(self.argv([ref]))
        self.assertEqual(sorted(p.name for p in self.root.rglob("*")), before)
        self.assertEqual(sorted(os.listdir(".")), cwd_before)


class WeightsFromManifestTestCase(unittest.TestCase):
    """The pure manifest -> weights-tuple helper (rules mirror the local
    ``compute_model_bytes`` / ``_iter_regular_files``)."""

    SIZER = FASTMLX_RECOMMEND.launch._safetensors_fit

    def test_sums_safetensors_and_ignores_small_other_files(self):
        entries = [
            {"name": "config.json", "size": 10},
            {"name": "a.safetensors", "size": 7},
            {"name": "sub/b.safetensors", "size": 5},
        ]
        weight_bytes, weight_files, side_files = self.SIZER.weights_from_manifest(entries)
        self.assertEqual(weight_bytes, 12)
        self.assertEqual(
            weight_files,
            [{"name": "a.safetensors", "bytes": 7}, {"name": "sub/b.safetensors", "bytes": 5}],
        )
        self.assertEqual(side_files, [])

    def test_dot_component_excludes_a_path_at_any_depth(self):
        entries = [
            {"name": "a.safetensors", "size": 7},
            {"name": ".x/b.safetensors", "size": 100},
            {"name": "d/.y/c.safetensors", "size": 100},
        ]
        self.assertEqual(self.SIZER.weights_from_manifest(entries)[0], 7)

    def test_big_unnamed_file_refused_and_named_one_is_a_side_file(self):
        entries = [{"name": "a.safetensors", "size": 7}, {"name": "t.bin", "size": 1 << 30}]
        with self.assertRaises(self.SIZER.FitCheckError):
            self.SIZER.weights_from_manifest(entries)
        _, _, side = self.SIZER.weights_from_manifest(entries, ["t.bin"])
        self.assertEqual(side, [{"name": "t.bin", "bytes": 1 << 30}])
        with self.assertRaises(self.SIZER.FitCheckError):
            self.SIZER.weights_from_manifest(entries, ["missing.bin"])

    def test_compute_fit_with_weights_walks_nothing(self):
        args = self.SIZER.build_arg_parser().parse_args(
            [
                "--model-path", "/nonexistent/never/touched",
                "--kv-reserve-gib", "1",
                "--wired-limit-mib", "4096",
                "--wired-margin-gib", "2",
            ]
        )
        result = self.SIZER.compute_fit(args, weights=(123, [{"name": "x", "bytes": 123}], []))
        self.assertEqual(result["weights_bytes"], 123)
        self.assertEqual(result["fit"], "green")


if __name__ == "__main__":
    unittest.main()
