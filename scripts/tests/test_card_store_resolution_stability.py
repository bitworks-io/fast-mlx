"""Publish-time gate: a published quality-card store must not change the card
that any released binary resolves.

Invariant under test (verbatim): a published card store must not change the
card that any released binary resolves, for any pack, host class or engine
build.

How it works
------------
* Baseline = ``fixtures/published-quality-guides-baseline.json``, a byte copy
  of the last PUBLISHED store (the v0.1.17..v0.1.21 bundled store). It is
  immutable test data; it may only be replaced by a store that is itself
  already published (a develop-side check proves its bytes equal a release
  tag's store or ``origin/main``'s, and skips when those refs are absent).
* Candidate = ``site/quality-guides.json`` in the tree under test.
* Resolvers are the RELEASED semantics, frozen:

  - Python: the launcher's ``resolve_card`` (one variant across v0.1.6 ..
    v0.1.21; ``fastmlx cards pull`` first shipped in v0.1.9). A pin test fixes
    the sha256 of its source and of every helper it calls.
  - Swift: ``swift_resolve`` below is a pure-Python model of
    ``QualityCardStore.resolve`` / ``select`` (one variant across v0.1.5 ..
    v0.1.21: no engine-build input; a unique host-class match, else the lowest
    id). The model must reproduce the engine expectation of EVERY case in
    ``card-resolution-conformance-v1.json``, so it cannot drift from the Swift
    engine unnoticed.

* Matrix: repo x revision x host class x engine build x residency. The outcome
  per cell and resolver is ``card:<id>``, ``none`` or ``refused:<kind>``. A
  cell whose baseline outcome is ``card:*`` or ``refused:*`` and whose
  candidate outcome differs is a violation. ``none -> card:*`` (a card for a
  pack that had none) is an allowed addition. ``none -> refused:*`` is a
  violation too: a pack that used to launch must not start refusing.

Scope note: the outcome carries the resolved card id, not its content. A
verdict edit on a pack's only card keeps ``card:<id>`` and is the pull rules'
business (R6), not this gate's; a verdict edit inside a multi-card pack moves
the NO_GO narrowing and is caught here.
"""

import hashlib
import importlib.util
import inspect
import json
import re
import subprocess
import unittest
from copy import deepcopy
from pathlib import Path
from typing import NamedTuple, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCH_PATH = REPO_ROOT / "scripts" / "fastmlx_launch.py"
_SPEC = importlib.util.spec_from_file_location("fastmlx_launch", LAUNCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
LAUNCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(LAUNCH)

BASELINE_PATH = REPO_ROOT / "scripts/tests/fixtures/published-quality-guides-baseline.json"
CANDIDATE_PATH = REPO_ROOT / "site/quality-guides.json"
LEDGER_PATH = REPO_ROOT / "site/releases.json"
CONFORMANCE_PATH = (
    REPO_ROOT / "spike/Tests/HarnessCoreTests/Fixtures/card-resolution-conformance-v1.json"
)

BASELINE_SHA256 = "d96156a2363399785fa97fa46e2c90a5c4a908529e6d96f52c8326bc8db6bdb5"
RELEASED_TAG = "v0.1.21"

# Matrix axes.
HOST_CLASSES = ("apple-m5", "apple-m3-ultra", "apple-m9-unrecognized", None)
RESIDENCY_AXIS = ("resident", "expert-stream")
UNKNOWN_REPO = "org/unknown-pack"
UNKNOWN_REVISION = "f" * 40
UNKNOWN_BUILD = "0" * 40

# The released Python resolver surface: ``resolve_card`` and everything it
# calls. Module constants are pinned by their defining source line.
PINNED_FUNCTIONS = (
    "resolve_card",
    "find_cards_by_repo",
    "find_cards_by_pin",
    "_card_identity",
    "card_hardware_class",
    "card_engine_build_commit",
    "card_residency",
    "_is_full_hex_revision",
    "_is_usable_hf_pin",
    "_hf_pin_matches_revision",
    "_as_dict",
    "malformed_model_card_notice",
    "tiebreak_notice",
)
PINNED_CONSTANTS = (
    "RESIDENCIES",
    "_HEX_DIGITS_RE",
    "_MIN_HF_PIN_LENGTH",
    "_FULL_REVISION_LENGTH",
)
# sha256 over ``pinned_resolver_sources()`` of the released resolver
# (identical in every tag v0.1.6 .. v0.1.21).
PINNED_RESOLVER_SHA256 = "44955bc6fe2fe8c3e3d8ab31b737764e0f90d74a0340bbdfdf87043f737e463f"
PIN_FAILURE_MESSAGE = (
    "the released resolver changed; vendor the released variant into this test "
    "before changing the live one"
)


# --------------------------------------------------------------------------
# Resolvers
# --------------------------------------------------------------------------


def python_outcome(cards: list, repo, revision, host, build, residency) -> str:
    """Outcome of the launcher's own ``resolve_card`` for one cell."""
    try:
        card = LAUNCH.resolve_card(
            cards,
            repo,
            revision,
            residency=residency,
            engine_build_commit=build,
            host_hardware_class=lambda: host,
        )
    except LAUNCH.LaunchRefusal as refusal:
        return f"refused:{refusal.exit_code}"
    return "none" if card is None else f"card:{card.get('id')}"


class _SwiftCard(NamedTuple):
    id: str
    repo: Optional[str]
    hf_pin: Optional[str]
    verdict: str
    hardware_class: Optional[str]
    residency: str


_SWIFT_VERDICTS = ("NO_GO", "PASS", "REFERENCE", "EXACT", "UNMEASURED")


def _swift_decode(card) -> Optional[_SwiftCard]:
    """The engine's card decode, narrowed to the fields resolution reads.

    A card that fails the engine's strict decode (no string id / verdict, no
    ``model`` or ``legible`` object) is dropped, as ``loadManifestDetailed``
    drops it. An unrecognized verdict string reads ``UNMEASURED``.
    """
    if not isinstance(card, dict):
        return None
    card_id, model, verdict = card.get("id"), card.get("model"), card.get("verdict")
    if not isinstance(card_id, str) or not isinstance(model, dict):
        return None
    if not isinstance(verdict, str) or not isinstance(card.get("legible"), dict):
        return None
    config = card.get("config")
    config = config if isinstance(config, dict) else {}
    repo, hf_pin = model.get("repo"), model.get("hfPin")
    hardware_class = config.get("hardwareClass")
    residency = config.get("residency")
    return _SwiftCard(
        id=card_id,
        repo=repo if isinstance(repo, str) else None,
        hf_pin=hf_pin if isinstance(hf_pin, str) else None,
        verdict=verdict if verdict in _SWIFT_VERDICTS else "UNMEASURED",
        hardware_class=hardware_class if isinstance(hardware_class, str) else None,
        residency=residency if isinstance(residency, str) else "resident",
    )


_HEX = re.compile(r"^[0-9a-fA-F]+$")


def _swift_pin_matches(hf_pin: Optional[str], revision: Optional[str]) -> bool:
    if revision is None or len(revision) != 40 or not _HEX.fullmatch(revision):
        return False
    if hf_pin is None or len(hf_pin) < 8 or not _HEX.fullmatch(hf_pin):
        return False
    return revision.lower().startswith(hf_pin.lower())


def _swift_select(candidates: list, host: Optional[str]) -> Optional[_SwiftCard]:
    """``QualityCardStore.select``: no engine-build input."""
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    no_go = [card for card in candidates if card.verdict == "NO_GO"]
    pool = no_go if no_go else candidates
    if host is not None:
        host_matches = [card for card in pool if card.hardware_class == host]
        if len(host_matches) == 1:
            return host_matches[0]
    return min(pool, key=lambda card: card.id)


def swift_resolve(cards: list, repo, revision, host) -> str:
    """Pure-Python model of ``QualityCardStore.resolve(repo:revision:
    hostHardwareClass:in:)``, as a canonical outcome string.

    The Swift engine only serves resident weights, so the residency axis does
    not exist for it: every cell resolves the resident cards.
    """
    decoded = [card for card in map(_swift_decode, cards) if card is not None]
    resident = [card for card in decoded if card.residency == "resident"]
    repo_cards = [card for card in resident if repo is not None and card.repo == repo]
    pin_cards = [card for card in resident if _swift_pin_matches(card.hf_pin, revision)]
    if repo_cards and pin_cards:
        identity = lambda c: (c.id, c.repo, c.hf_pin, c.verdict, c.hardware_class)  # noqa: E731
        if {identity(c) for c in repo_cards} != {identity(c) for c in pin_cards}:
            return "refused:ambiguous"
    candidates = repo_cards if repo_cards else pin_cards
    if not candidates:
        return "none"
    selected = _swift_select(candidates, host)
    return "none" if selected is None else f"card:{selected.id}"


# --------------------------------------------------------------------------
# Matrix and comparison
# --------------------------------------------------------------------------


def _padded_pin(pin) -> Optional[str]:
    """A 40-hex revision that a card's ``hfPin`` prefix-matches, or ``None``."""
    if not isinstance(pin, str) or len(pin) < 8 or not _HEX.fullmatch(pin):
        return None
    if len(pin) > 40:
        return None
    return pin.lower() + "0" * (40 - len(pin))


def _card_repo(card) -> Optional[str]:
    model = card.get("model") if isinstance(card, dict) else None
    repo = model.get("repo") if isinstance(model, dict) else None
    return repo if isinstance(repo, str) else None


def _card_pin(card) -> Optional[str]:
    model = card.get("model") if isinstance(card, dict) else None
    return _padded_pin(model.get("hfPin")) if isinstance(model, dict) else None


def _ledger_release_commits() -> list:
    """Every release commit the public ledger names (the engine builds a
    released binary can carry). Empty when the ledger is absent."""
    try:
        ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    commits = []
    for release in ledger.get("releases", []) if isinstance(ledger, dict) else []:
        commit = release.get("publicCommit") if isinstance(release, dict) else None
        if isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit):
            commits.append(commit)
    return commits


def matrix_axes(*stores: list) -> dict:
    """The cell axes for the union of ``stores`` (so a baseline and a
    candidate are always compared over exactly the same cells)."""
    all_cards = [card for store in stores for card in store if isinstance(card, dict)]
    repos = sorted({repo for repo in map(_card_repo, all_cards) if repo is not None})
    all_pins = sorted({pin for pin in map(_card_pin, all_cards) if pin is not None})
    revisions = {}
    for repo in repos:
        own = sorted(
            {_card_pin(card) for card in all_cards if _card_repo(card) == repo} - {None}
        )
        revisions[repo] = [None, *own, UNKNOWN_REVISION]
    # Pin-only cards name no repo: reach them through a repo-less launch and
    # through an unknown repo, under every pin in either store.
    for repo in (None, UNKNOWN_REPO):
        revisions[repo] = [None, *all_pins, UNKNOWN_REVISION]
    builds = {None, UNKNOWN_BUILD, *_ledger_release_commits()}
    for card in all_cards:
        commit = LAUNCH.card_engine_build_commit(card)
        if commit is not None:
            builds.add(commit)
    return {
        "revisions": revisions,
        "builds": sorted(builds, key=lambda b: (b is not None, b or "")),
        "hosts": list(HOST_CLASSES),
        "residencies": list(RESIDENCY_AXIS),
    }


def resolution_matrix(store_cards: list, axes: Optional[dict] = None) -> dict:
    """``{(repo, revision, host, build, residency): {"python": outcome,
    "swift": outcome}}`` for one store. ``axes`` defaults to this store's own."""
    axes = axes if axes is not None else matrix_axes(store_cards)
    matrix = {}
    swift_memo = {}
    for repo, revisions in axes["revisions"].items():
        for revision in revisions:
            for host in axes["hosts"]:
                swift_key = (repo, revision, host)
                swift_memo[swift_key] = swift_resolve(store_cards, repo, revision, host)
                for build in axes["builds"]:
                    for residency in axes["residencies"]:
                        matrix[(repo, revision, host, build, residency)] = {
                            "python": python_outcome(
                                store_cards, repo, revision, host, build, residency
                            ),
                            "swift": swift_memo[swift_key],
                        }
    return matrix


class Violation(NamedTuple):
    repo: Optional[str]
    revision: Optional[str]
    host: Optional[str]
    build: Optional[str]
    residency: str
    resolver: str
    baseline: str
    candidate: str

    def __str__(self) -> str:
        return (
            f"repo={self.repo!r} revision={self.revision!r} host={self.host!r} "
            f"engineBuild={self.build!r} residency={self.residency} "
            f"resolver={self.resolver}: baseline {self.baseline} -> candidate {self.candidate}"
        )


RESOLVERS = ("python", "swift")


def _changed_cells(baseline: dict, candidate: dict):
    if set(baseline) != set(candidate):
        raise AssertionError("baseline and candidate matrices must share one cell set")
    for cell in sorted(baseline, key=repr):
        for resolver in RESOLVERS:
            before, after = baseline[cell][resolver], candidate[cell][resolver]
            if before != after:
                yield cell, resolver, before, after


def compare(baseline: dict, candidate: dict) -> list:
    """Every cell where a resolved ``card:*`` or a ``refused:*`` outcome of the
    baseline store changed in the candidate store (``none -> refused:*``
    included). ``none -> card:*`` is an addition, not a violation."""
    violations = []
    for cell, resolver, before, after in _changed_cells(baseline, candidate):
        if before == "none" and after.startswith("card:"):
            continue
        violations.append(Violation(*cell, resolver, before, after))
    return violations


def additions(baseline: dict, candidate: dict) -> list:
    """The allowed ``none -> card:*`` cells."""
    return [
        Violation(*cell, resolver, before, after)
        for cell, resolver, before, after in _changed_cells(baseline, candidate)
        if before == "none" and after.startswith("card:")
    ]


def gate(baseline_cards: list, candidate_cards: list) -> tuple:
    """``(violations, additions)`` for a candidate store against a baseline."""
    axes = matrix_axes(baseline_cards, candidate_cards)
    baseline = resolution_matrix(baseline_cards, axes)
    candidate = resolution_matrix(candidate_cards, axes)
    return compare(baseline, candidate), additions(baseline, candidate)


# --------------------------------------------------------------------------
# Pin of the released Python resolver
# --------------------------------------------------------------------------


def _constant_line(source: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}\b.*$", source, re.M)
    assert match is not None, f"constant {name} not found"
    return match.group(0)


def pinned_resolver_sources(
    functions: tuple = PINNED_FUNCTIONS, constants: tuple = PINNED_CONSTANTS
) -> list:
    """``[(name, source text)]`` for the released resolver surface of the
    launcher under test."""
    launch_source = LAUNCH_PATH.read_text(encoding="utf-8")
    sources = [(name, inspect.getsource(getattr(LAUNCH, name))) for name in functions]
    sources += [(name, _constant_line(launch_source, name)) for name in constants]
    return sources


def pinned_resolver_digest(sources: list) -> str:
    digest = hashlib.sha256()
    for name, text in sources:
        digest.update(f"{name}\n{text}\n".encode("utf-8"))
    return digest.hexdigest()


def _git_bytes(*args: str) -> Optional[bytes]:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _release_tag_available() -> bool:
    return _git_bytes("rev-parse", "--verify", "--quiet", f"{RELEASED_TAG}^{{commit}}") is not None


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def load_store(path: Path) -> list:
    document = json.loads(path.read_text(encoding="utf-8"))
    return document["cards"]


def find_card(cards: list, card_id: str) -> dict:
    return next(card for card in cards if card["id"] == card_id)


def released_v0120_commit() -> str:
    """The v0.1.20 engine build (``96c77660``) from the ledger, else a stand-in."""
    for commit in _ledger_release_commits():
        if commit.startswith("96c77660"):
            return commit
    return "96c77660" + "1" * 32


M5_REPO = "mlx-community/Qwen3-0.6B-4bit"


def with_v0120_sibling(cards: list) -> list:
    """The cycle-270 shape: a clone of ``qwen3-0p6b-4bit@m5`` with another id
    and engine build, appended beside it."""
    candidate = deepcopy(cards)
    clone = deepcopy(find_card(candidate, "qwen3-0p6b-4bit@m5"))
    clone["id"] = "qwen3-0p6b-4bit@m5-v0120"
    clone["provenance"]["engineBuild"]["commit"] = released_v0120_commit()
    candidate.append(clone)
    return candidate


def find_violation(violations: list, *, repo, host, resolver, baseline, candidate, **extra):
    wanted = dict(repo=repo, host=host, resolver=resolver, baseline=baseline, candidate=candidate)
    wanted.update(extra)
    return [v for v in violations if all(getattr(v, key) == value for key, value in wanted.items())]


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


class CardStoreResolutionStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = load_store(BASELINE_PATH)
        cls.candidate = load_store(CANDIDATE_PATH)

    # A1 -------------------------------------------------------------------
    def test_a1_current_store_does_not_change_any_baseline_resolution(self):
        self.assertEqual(hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest(), BASELINE_SHA256)
        axes = matrix_axes(self.baseline, self.candidate)
        baseline = resolution_matrix(self.baseline, axes)
        candidate = resolution_matrix(self.candidate, axes)
        violations = compare(baseline, candidate)
        self.assertEqual(
            violations,
            [],
            "published store changes a released binary's resolution:\n"
            + "\n".join(str(v) for v in violations[:20]),
        )
        for resolver in RESOLVERS:
            resolved = sum(1 for cell in baseline.values() if cell[resolver] != "none")
            self.assertGreater(resolved, 0, f"{resolver}: the matrix exercised no non-none cell")

    # A2 -------------------------------------------------------------------
    def test_a2_sibling_on_the_host_class_fails_the_gate_on_the_named_cell(self):
        candidate = with_v0120_sibling(self.baseline)
        violations, _ = gate(self.baseline, candidate)
        swift = find_violation(
            violations,
            repo=M5_REPO,
            host="apple-m5",
            resolver="swift",
            baseline="card:qwen3-0p6b-4bit@m5",
            candidate="card:qwen3-0p6b-4bit@m3ultra",
        )
        self.assertTrue(
            swift,
            "no Swift violation on the qwen3-0p6b-4bit / apple-m5 cell; got:\n"
            + "\n".join(str(v) for v in violations[:10]),
        )
        # Swift has no engine-build input, so the change reaches every build.
        self.assertEqual({v.build for v in swift} >= {None, UNKNOWN_BUILD}, True)
        python = find_violation(
            violations,
            repo=M5_REPO,
            host="apple-m5",
            resolver="python",
            baseline="card:qwen3-0p6b-4bit@m5",
            candidate="card:qwen3-0p6b-4bit@m3ultra",
            build=None,
        )
        self.assertTrue(python, "no Python violation on the apple-m5 / no-engine-build cell")

    # A3 -------------------------------------------------------------------
    def test_a3_card_for_a_pack_with_no_baseline_card_is_an_allowed_addition(self):
        candidate = deepcopy(self.baseline)
        clone = deepcopy(find_card(candidate, "qwen3-8b-4bit@m5"))
        clone["id"] = "brand-new-pack-4bit@m5"
        clone["model"] = {"repo": "org/brand-new-pack-4bit", "hfPin": "abcdef0123456789"}
        candidate.append(clone)
        violations, added = gate(self.baseline, candidate)
        self.assertEqual(violations, [], "\n".join(str(v) for v in violations[:10]))
        self.assertTrue(added, "the new pack produced no none -> card addition")
        self.assertTrue(
            {v.candidate for v in added} == {"card:brand-new-pack-4bit@m5"}
            and {v.baseline for v in added} == {"none"}
        )
        self.assertEqual({v.resolver for v in added}, {"python", "swift"})

    # A4 -------------------------------------------------------------------
    def test_a4_editing_hardware_class_of_an_existing_card_fails_the_gate(self):
        candidate = deepcopy(self.baseline)
        find_card(candidate, "qwen3-0p6b-4bit@m5")["config"]["hardwareClass"] = "apple-m3-ultra"
        violations, _ = gate(self.baseline, candidate)
        self.assertTrue(
            find_violation(
                violations,
                repo=M5_REPO,
                host="apple-m5",
                resolver="swift",
                baseline="card:qwen3-0p6b-4bit@m5",
                candidate="card:qwen3-0p6b-4bit@m3ultra",
            ),
            "\n".join(str(v) for v in violations[:10]),
        )

    def test_a4_editing_verdict_inside_a_multi_card_pack_fails_the_gate(self):
        candidate = deepcopy(self.baseline)
        # NO_GO narrowing: the m3ultra card stops being NO_GO, so the m3ultra
        # host now resolves the remaining NO_GO card (the m5 one).
        find_card(candidate, "qwen3-0p6b-4bit@m3ultra")["verdict"] = "PASS"
        violations, _ = gate(self.baseline, candidate)
        self.assertTrue(
            find_violation(
                violations,
                repo=M5_REPO,
                host="apple-m3-ultra",
                resolver="swift",
                baseline="card:qwen3-0p6b-4bit@m3ultra",
                candidate="card:qwen3-0p6b-4bit@m5",
            ),
            "\n".join(str(v) for v in violations[:10]),
        )

    def test_a4_removing_a_card_fails_the_gate(self):
        candidate = [c for c in self.baseline if c["id"] != "qwen3-8b-4bit@m5"]
        violations, _ = gate(self.baseline, candidate)
        self.assertTrue(
            find_violation(
                violations,
                repo="mlx-community/Qwen3-8B-4bit",
                host="apple-m5",
                resolver="swift",
                baseline="card:qwen3-8b-4bit@m5",
                candidate="none",
            ),
            "\n".join(str(v) for v in violations[:10]),
        )

    # A5 -------------------------------------------------------------------
    def test_a5_swift_model_reproduces_every_conformance_case(self):
        cases = json.loads(CONFORMANCE_PATH.read_text(encoding="utf-8"))["cases"]
        self.assertGreater(len(cases), 50)
        for case in cases:
            with self.subTest(case=case["id"]):
                expect = case["expect"]
                if expect == "none":
                    want = "none"
                elif "card" in expect:
                    want = f"card:{expect['card']}"
                else:
                    want = "refused:ambiguous"
                got = swift_resolve(
                    case["cards"], case["repo"], case["revision"], case["hostHardwareClass"]
                )
                self.assertEqual(got, want, case["why"])

    def test_a5_python_outcomes_match_the_conformance_table_in_canonical_form(self):
        for case in json.loads(CONFORMANCE_PATH.read_text(encoding="utf-8"))["cases"]:
            with self.subTest(case=case["id"]):
                expect = case["expect"]
                if case.get("knownPythonDivergence") is not None:
                    want = "refused:3"
                elif expect == "none":
                    want = "none"
                elif "card" in expect:
                    want = f"card:{expect['card']}"
                else:
                    want = "refused:3"
                got = python_outcome(
                    case["cards"],
                    case["repo"],
                    case["revision"],
                    case["hostHardwareClass"],
                    None,
                    "resident",
                )
                self.assertEqual(got, want, case["why"])

    def test_a5_conformance_table_carries_the_sibling_on_host_class_case(self):
        cases = {c["id"]: c for c in json.loads(CONFORMANCE_PATH.read_text(encoding="utf-8"))["cases"]}
        case = cases["sibling-on-host-class-hides-the-unique-host-match"]
        classes = sorted(c["config"]["hardwareClass"] for c in case["cards"])
        self.assertEqual(classes, ["apple-m3-ultra", "apple-m5", "apple-m5"])
        self.assertEqual(case["hostHardwareClass"], "apple-m5")
        self.assertEqual(case["expect"], {"card": "pack-a@m3ultra"})

    # A6 -------------------------------------------------------------------
    def test_a6_released_python_resolver_is_pinned(self):
        digest = pinned_resolver_digest(pinned_resolver_sources())
        self.assertEqual(digest, PINNED_RESOLVER_SHA256, PIN_FAILURE_MESSAGE)

    @unittest.skipUnless(_release_tag_available(), f"{RELEASED_TAG} is not available (shallow clone)")
    def test_a6_pinned_resolver_sources_appear_verbatim_in_the_released_launcher(self):
        released = _git_bytes("show", f"{RELEASED_TAG}:scripts/fastmlx_launch.py")
        self.assertIsNotNone(released)
        released_text = released.decode("utf-8")
        for name, text in pinned_resolver_sources():
            with self.subTest(name=name):
                self.assertIn(text, released_text, f"{name} differs from {RELEASED_TAG}")

    @unittest.skipUnless(_release_tag_available(), f"{RELEASED_TAG} is not available (shallow clone)")
    def test_a6_baseline_bytes_equal_a_published_store(self):
        published = [_git_bytes("show", f"{RELEASED_TAG}:site/quality-guides.json")]
        published.append(_git_bytes("show", "origin/main:site/quality-guides.json"))
        published = [blob for blob in published if blob is not None]
        self.assertTrue(published)
        baseline = BASELINE_PATH.read_bytes()
        self.assertIn(baseline, published, "baseline is not a store that was published")

    def test_a6_conformance_fixture_and_baseline_carry_no_machine_paths(self):
        for path in (BASELINE_PATH, CONFORMANCE_PATH):
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("/" + "private/", text)
                self.assertIsNone(re.search(r'"/(Users|home|tmp|var)/', text))


if __name__ == "__main__":
    unittest.main()
