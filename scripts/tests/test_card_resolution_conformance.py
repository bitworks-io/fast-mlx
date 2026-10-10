"""Shared card-resolution conformance table, Python side.

``spike/Tests/HarnessCoreTests/Fixtures/card-resolution-conformance-v1.json``
is checked by BOTH this module (``fastmlx_launch.resolve_card``, which
``fastmlx recommend`` uses to judge a pack before any download) and the Swift
engine (``QualityCardStore.resolve``, ``CardResolutionConformanceTests``). The
two are one selection rule in two languages; if they drift, recommend vouches
for (or refuses) a pack the engine judges differently. The Swift engine is the
authority.

Scope held fixed: Python's ``engine_build_commit`` tiebreak and ``residency``
argument have no Swift counterpart in ``resolve``, so every case runs with
``residency="resident"`` and ``engine_build_commit=None``; the host class is
injected through ``host_hardware_class``.

``knownPythonDivergence`` is tolerated in ONE direction only: Python refuses
(``LaunchRefusal`` exit 3) where the engine picks a card.
"""

import importlib.util
import json
import re
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
    / "spike/Tests/HarnessCoreTests/Fixtures/card-resolution-conformance-v1.json"
)


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def load_cards(case: dict) -> list:
    """Round-trip the case's cards through the launcher's own manifest loader."""
    manifest = {"schema": "fast-mlx-quality-card-v1", "cards": case["cards"]}
    with tempfile.TemporaryDirectory(prefix="cardres-") as root:
        path = Path(root) / "quality-guides.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        cards = FASTMLX_LAUNCH.load_quality_cards(path)
    assert cards is not None
    return cards


def resolve(case: dict):
    host = case["hostHardwareClass"]
    return FASTMLX_LAUNCH.resolve_card(
        load_cards(case),
        case["repo"],
        case["revision"],
        residency="resident",
        engine_build_commit=None,
        host_hardware_class=lambda: host,
    )


def outcome_kind(expect) -> str:
    if expect == "none":
        return "none"
    if "card" in expect:
        return "card"
    if "ambiguous" in expect:
        return "ambiguous"
    raise AssertionError(f"unknown expect shape: {expect!r}")


class CardResolutionConformanceTests(unittest.TestCase):
    def test_fixture_is_well_formed(self):
        fixture = load_fixture()
        self.assertEqual(fixture["schema"], "card-resolution-conformance-v1")
        cases = fixture["cases"]
        self.assertGreater(len(cases), 50, "the table must not be silently truncated")
        ids = [case["id"] for case in cases]
        self.assertEqual(len(ids), len(set(ids)), "duplicate case ids")
        text = FIXTURE_PATH.read_text(encoding="utf-8")
        # Built by concatenation so this file does not itself carry the marker the
        # public validator refuses.
        self.assertNotIn("/" + "private/", text)
        self.assertIsNone(re.search(r'"/(Users|home|tmp|var)/', text), "absolute path in fixture")
        for case in cases:
            for key in ("why", "repo", "revision", "hostHardwareClass", "cards", "expect"):
                self.assertIn(key, case, case["id"])
            self.assertTrue(case["why"], case["id"])
            outcome_kind(case["expect"])
            known = case.get("knownPythonDivergence")
            if known is not None:
                # The ONLY tolerated direction: Python refuses where the engine
                # picks a card. Anything else (Python returns a card the engine
                # does not, or a different one) is a defect, never a divergence.
                self.assertEqual(known["pythonOutcome"], "refuses", case["id"])
                self.assertEqual(outcome_kind(case["expect"]), "card", case["id"])
                self.assertTrue(known["reason"], case["id"])
                self.assertEqual(set(known), {"pythonOutcome", "reason"}, case["id"])

    def test_table_covers_every_outcome_kind(self):
        cases = load_fixture()["cases"]
        kinds = {outcome_kind(case["expect"]) for case in cases}
        self.assertEqual(kinds, {"none", "card", "ambiguous"})
        self.assertTrue(
            any("knownPythonDivergence" in case for case in cases),
            "no python-refuses case",
        )
        # Each R1 shape named by the predeclaration must stay in the table.
        required = {
            "repo-only-single-card",
            "pin-only-8-hex-prefix",
            "repo-and-pin-agree-on-one-card",
            "repo-and-pin-name-different-cards",
            "residency-absent-config-missing",
            "residency-null",
            "residency-resident",
            "residency-expert-stream-only",
            "residency-unknown-only",
            "no-go-beside-pass-host-on-pass-class",
            "no-go-beside-pass-host-on-no-go-class",
            "no-go-beside-pass-host-on-neither",
            "no-go-beside-pass-host-nil",
            "unique-host-match-beats-lowest-id",
            "several-host-matches-use-lowest-id-of-whole-pool",
            "same-verdict-pass-lowest-id-manifest-order-scrambled",
            "pass-vs-exact-host-nil",
            "unrecognized-verdict-single-card",
            "pin-7-hex-too-short",
            "pin-non-hex-8-chars",
            "pin-40-hex-equal-to-revision",
            "pin-uppercase-hex-matches-lowercase-revision",
            "revision-39-hex-never-matches",
            "revision-41-hex-never-matches",
            "revision-uppercase-matches-lowercase-pin",
        }
        self.assertLessEqual(required, {case["id"] for case in cases})

    def test_divergence_check_rejects_a_python_card_where_the_engine_has_none(self):
        # Guard on the guard: a divergence entry may never excuse Python
        # returning a card the engine does not. Feed the checker a table whose
        # engine expectation is "none" while Python resolves a card.
        case = {
            "id": "synthetic-wrong-direction",
            "repo": "org/pack-a",
            "revision": None,
            "hostHardwareClass": None,
            "cards": [
                {
                    "id": "card-a",
                    "model": {"repo": "org/pack-a"},
                    "verdict": "PASS",
                    "legible": {"tier": "t", "headline": "h"},
                }
            ],
            "expect": "none",
            "knownPythonDivergence": {"pythonOutcome": "refuses", "reason": "synthetic"},
        }
        with self.assertRaises(AssertionError):
            check_case(self, case)

    def test_python_resolve_card_matches_the_shared_table(self):
        for case in load_fixture()["cases"]:
            with self.subTest(case=case["id"]):
                check_case(self, case)


def check_case(test: unittest.TestCase, case: dict) -> None:
    launch = FASTMLX_LAUNCH
    expect = case["expect"]
    known = case.get("knownPythonDivergence")
    if known is not None:
        if known["pythonOutcome"] != "refuses" or outcome_kind(expect) != "card":
            raise AssertionError(f"{case['id']}: divergence may only be python-refuses-where-engine-picks-a-card")
        with test.assertRaises(launch.LaunchRefusal, msg=case["why"]) as caught:
            resolve(case)
        test.assertEqual(caught.exception.exit_code, 3, case["id"])
        return
    kind = outcome_kind(expect)
    if kind == "ambiguous":
        with test.assertRaises(launch.LaunchRefusal, msg=case["why"]) as caught:
            resolve(case)
        test.assertEqual(caught.exception.exit_code, 3, case["id"])
        message = caught.exception.message
        test.assertIn("ambiguous", message)
        test.assertIn(f"card(s) {sorted(expect['ambiguous']['repo'])}", message)
        test.assertIn(f"different card(s) {sorted(expect['ambiguous']['pin'])}", message)
        return
    actual = resolve(case)
    if kind == "none":
        test.assertIsNone(actual, case["why"])
    else:
        test.assertIsNotNone(actual, case["why"])
        test.assertEqual(actual.get("id"), expect["card"], case["why"])


if __name__ == "__main__":
    unittest.main()
