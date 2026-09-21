import argparse
import contextlib
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

from scripts.tests.test_fastmlx_gguf_fit import build_gguf_bytes
from scripts.tests.test_fastmlx_launch import (
    CAPTURING_FIT_CHECK_BODY,
    GREEN_ATTESTATION_WITH_RESIDENCY_BODY,
    SYNTHETIC_CARD_ID,
    SYNTHETIC_CARD_REPO,
    SYNTHETIC_CARD_REVISION,
    write_expert_stream_card_manifest,
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


if __name__ == "__main__":
    unittest.main()
