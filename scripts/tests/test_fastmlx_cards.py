"""Tests for ``fastmlx cards pull`` (``scripts/fastmlx_cards.py``).

Predeclaration (frozen contract, rules R1-R7, acceptance P1-P7):
docs/task-inbox/2026-10-03-PREDECLARATION-fastmlx-cards-pull-fetches-a-pinned-monotonic-card-store.md

Every test uses an injected fake opener (``CARDS.https_opener`` patched) and
a temporary HOME / ``--output``; none touches the network. Test stores are
built from the REAL bundled ``site/quality-guides.json`` so they track it.
"""

import contextlib
import copy
import hashlib
import http.client
import importlib.util
import io
import json
import os
import shlex
import stat
import tempfile
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from scripts.tests import test_fastmlx_launch as launch_tests
from scripts.tests import test_fastmlx_recommend as recommend_tests


CARDS_PATH = Path(__file__).resolve().parents[1] / "fastmlx_cards.py"
_SPEC = importlib.util.spec_from_file_location("fastmlx_cards", CARDS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
CARDS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(CARDS)

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_STORE_PATH = REPO_ROOT / "site" / "quality-guides.json"

COMMIT = "a" * 40
EXPECTED_URL = (
    f"https://raw.githubusercontent.com/bitworks-io/fast-mlx/{COMMIT}/site/quality-guides.json"
)
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
FOUR_MIB = 4 << 20

NO_GO_ID = "qwen38-27b-optiq-4bit@m3ultra"
NO_GO_REPO = "mlx-community/Qwen3.8-27B-OptiQ-4bit"
NO_GO_REVISION = "b04599de" + "0" * 32  # 40-hex revision whose prefix is the card's hfPin

NEW_CARD_ID = "pulled-fixture-no-go@test"
NEW_CARD_REPO = "example/PulledNoGoModel"


def bundled_document() -> dict:
    return json.loads(BUNDLED_STORE_PATH.read_bytes().decode("utf-8"))


def serialize(document: dict) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=False) + "\n").encode("utf-8")


def later(document: dict, days: int = 1) -> str:
    parsed = datetime.strptime(document["generatedAt"], TIME_FORMAT)
    return (parsed + timedelta(days=days)).strftime(TIME_FORMAT)


def new_card(card_id: str = NEW_CARD_ID, repo: str = NEW_CARD_REPO, verdict: str = "NO_GO") -> dict:
    return launch_tests._synthetic_card(card_id, repo, verdict)


def honest_document() -> dict:
    """Bundled store + one new card, strictly newer ``generatedAt``."""
    document = bundled_document()
    document["generatedAt"] = later(document)
    document["cards"].append(new_card())
    return document


class FakeResponse:
    def __init__(self, body: bytes, url: str):
        self._body = body
        self._url = url
        self.read_sizes = []

    def read(self, size=-1):
        self.read_sizes.append(size)
        if size is None or size < 0:
            return self._body
        return self._body[:size]

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        pass


class FakeOpener:
    def __init__(self, body=b"", final_url=EXPECTED_URL, raises=None):
        self.body = body
        self.final_url = final_url
        self.raises = raises
        self.calls = []
        self.response = None

    def open(self, request, timeout=None):
        url = request if isinstance(request, str) else request.full_url
        self.calls.append((url, timeout))
        if self.raises is not None:
            raise self.raises
        self.response = FakeResponse(self.body, self.final_url)
        return self.response


class CardsPullTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.home = self.root / "home"
        self.home.mkdir()
        self.out_dir = self.root / "out"
        self.out_dir.mkdir()
        self.output = self.out_dir / "store.json"

    def run_pull(self, body, *, sha=None, commit=COMMIT, output="default", extra=(), opener=None,
                 baseline_root=None, pin_args=True, now=None):
        opener = opener if opener is not None else FakeOpener(body)
        self.opener = opener
        argv = ["pull", "--commit", commit]
        if pin_args:
            argv += ["--sha256", sha if sha is not None else hashlib.sha256(body).hexdigest()]
        if output == "default":
            argv += ["--output", str(self.output)]
        elif output is not None:
            argv += ["--output", str(output)]
        argv += list(extra)
        stdout, stderr = io.StringIO(), io.StringIO()
        patches = [
            patch.dict(os.environ, {"HOME": str(self.home)}),
            patch.object(CARDS, "https_opener", lambda: opener),
        ]
        if baseline_root is not None:
            patches.append(patch.object(CARDS.launch, "REPO_ROOT", baseline_root))
        if now is not None:
            patches.append(patch.object(CARDS, "utc_now", lambda: now))
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            stack.enter_context(contextlib.redirect_stdout(stdout))
            stack.enter_context(contextlib.redirect_stderr(stderr))
            with self.assertRaises(SystemExit) as ctx:
                CARDS.main(argv)
        return ctx.exception.code, stdout.getvalue(), stderr.getvalue()

    def all_files_under(self, root: Path) -> list:
        return sorted(str(p) for p in root.rglob("*") if p.is_file())

    def assert_refused_nothing_written(self, code, stderr, expected_code=3, reason=None):
        self.assertEqual(code, expected_code, stderr)
        if reason is not None:
            self.assertIn(reason, stderr)
        self.assertEqual(self.all_files_under(self.out_dir), [], "output or temp file left behind")
        self.assertFalse((self.home / ".fastmlx").exists(), "default store dir created")


# ---------------------------------------------------------------------
# P1 / P6: honest pull, idempotence, conflicting output
# ---------------------------------------------------------------------
class HonestPullTests(CardsPullTestCase):
    def test_p1_honest_store_pulls_and_prints_one_flag_line(self):
        body = serialize(honest_document())
        digest = hashlib.sha256(body).hexdigest()
        code, stdout, stderr = self.run_pull(body)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(self.output.read_bytes(), body)
        self.assertEqual(hashlib.sha256(self.output.read_bytes()).hexdigest(), digest)
        self.assertEqual(
            stdout,
            f"--quality-cards {shlex.quote(str(self.output))} --quality-cards-sha256 {digest}\n",
        )
        document = honest_document()
        self.assertEqual(
            stderr,
            f"fastmlx_cards=pulled commit={COMMIT} sha256={digest} "
            f"generated_at={document['generatedAt']} cards={len(document['cards'])} added=1\n",
        )
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o644)
        self.assertEqual(self.all_files_under(self.out_dir), [str(self.output)])

    def test_fetches_the_fixed_https_url_with_a_timeout_and_bounded_read(self):
        body = serialize(honest_document())
        code, _, stderr = self.run_pull(body)
        self.assertEqual(code, 0, stderr)
        ((url, timeout),) = self.opener.calls
        self.assertEqual(url, EXPECTED_URL)
        self.assertIsNotNone(timeout)
        self.assertGreater(timeout, 0)
        self.assertEqual(self.opener.response.read_sizes, [FOUR_MIB + 1])

    def test_default_output_is_content_addressed_under_home(self):
        body = serialize(honest_document())
        digest = hashlib.sha256(body).hexdigest()
        code, stdout, stderr = self.run_pull(body, output=None)
        self.assertEqual(code, 0, stderr)
        expected = self.home / ".fastmlx" / "cards" / f"{digest}.json"
        self.assertEqual(expected.read_bytes(), body)
        self.assertEqual(stdout.split()[1], str(expected))

    def test_pin_is_accepted_case_insensitively(self):
        body = serialize(honest_document())
        code, stdout, stderr = self.run_pull(body, sha=hashlib.sha256(body).hexdigest().upper())
        self.assertEqual(code, 0, stderr)
        self.assertTrue(stdout.rstrip().endswith(hashlib.sha256(body).hexdigest()))

    def test_pulling_the_bundled_store_itself_is_allowed_with_zero_added(self):
        body = BUNDLED_STORE_PATH.read_bytes()
        code, _, stderr = self.run_pull(body)
        self.assertEqual(code, 0, stderr)
        self.assertTrue(stderr.strip().endswith("added=0"), stderr)

    def test_tightening_a_verdict_is_allowed(self):
        document = honest_document()
        flipped = [c for c in document["cards"] if c["verdict"] == "PASS"]
        self.assertTrue(flipped)
        flipped[0]["verdict"] = "NO_GO"
        code, _, stderr = self.run_pull(serialize(document))
        self.assertEqual(code, 0, stderr)

    def test_p6_idempotent_repull_leaves_file_untouched(self):
        body = serialize(honest_document())
        code1, stdout1, _ = self.run_pull(body)
        self.assertEqual(code1, 0)
        before = self.output.stat()
        code2, stdout2, stderr2 = self.run_pull(body)
        self.assertEqual(code2, 0, stderr2)
        after = self.output.stat()
        self.assertEqual((before.st_ino, before.st_mtime_ns), (after.st_ino, after.st_mtime_ns))
        self.assertEqual(stdout1, stdout2)
        self.assertEqual(self.all_files_under(self.out_dir), [str(self.output)])

    def test_p6_conflicting_output_refuses_exit_3_and_keeps_the_file(self):
        self.output.write_bytes(b"something else entirely\n")
        before = self.output.stat()
        body = serialize(honest_document())
        code, stdout, stderr = self.run_pull(body)
        self.assertEqual(code, 3, stderr)
        self.assertEqual(stdout, "")
        self.assertIn(str(self.output), stderr)
        self.assertEqual(self.output.read_bytes(), b"something else entirely\n")
        after = self.output.stat()
        self.assertEqual((before.st_ino, before.st_mtime_ns), (after.st_ino, after.st_mtime_ns))
        self.assertEqual(self.all_files_under(self.out_dir), [str(self.output)])

    def test_unwritable_output_location_exits_1(self):
        blocker = self.out_dir / "blocker"
        blocker.write_bytes(b"x")
        code, stdout, stderr = self.run_pull(
            serialize(honest_document()), output=blocker / "store.json"
        )
        self.assertEqual(code, 1, stderr)
        self.assertEqual(stdout, "")


# ---------------------------------------------------------------------
# P3: digest mismatch
# ---------------------------------------------------------------------
class DigestTests(CardsPullTestCase):
    def test_p3_byte_flipped_body_exits_3_naming_both_digests_and_writes_nothing(self):
        body = serialize(honest_document())
        pin = hashlib.sha256(body).hexdigest()
        flipped = bytearray(body)
        flipped[body.index(b"2026-") + 3] ^= 0x01
        flipped = bytes(flipped)
        json.loads(flipped)  # still valid JSON: only the digest can catch it
        code, stdout, stderr = self.run_pull(flipped, sha=pin)
        self.assert_refused_nothing_written(code, stderr)
        self.assertEqual(stdout, "")
        self.assertIn(pin, stderr)
        self.assertIn(hashlib.sha256(flipped).hexdigest(), stderr)

    def test_p3_default_output_dir_is_not_even_created_on_mismatch(self):
        body = serialize(honest_document())
        code, _, stderr = self.run_pull(body, sha="0" * 64, output=None)
        self.assert_refused_nothing_written(code, stderr)


# ---------------------------------------------------------------------
# P4: one test (group) per rule R1-R7; each asserts reason text and that
# nothing (output or temp file) is written.
# ---------------------------------------------------------------------
class RuleTests(CardsPullTestCase):
    def refuse(self, document_or_bytes, reason, **kwargs):
        body = (
            document_or_bytes
            if isinstance(document_or_bytes, bytes)
            else serialize(document_or_bytes)
        )
        code, stdout, stderr = self.run_pull(body, **kwargs)
        self.assert_refused_nothing_written(code, stderr, reason=reason)
        self.assertEqual(stdout, "")

    # R1
    def test_r1_not_json(self):
        self.refuse(b"this is not json", "R1")

    def test_r1_not_utf8(self):
        self.refuse(b"\xff\xfe\x00 not utf8", "R1")

    def test_r1_json_but_not_a_card_store(self):
        self.refuse(b'["a", "list"]', "R1")
        self.refuse(b'{"schema": "fast-mlx-quality-card-v1", "cards": "nope"}', "R1")

    def test_r1_wrong_schema(self):
        document = honest_document()
        document["schema"] = "fast-mlx-quality-card-v2"
        self.refuse(document, "R1")
        del document["schema"]
        self.refuse(document, "R1")

    # R2
    def test_r2_generated_at_must_be_strict_format(self):
        for bad in (
            "2026-10-02T00:00:00+00:00",
            "2026-10-02",
            "2026-10-02T00:00:00z",
            "2026-10-02 00:00:00Z",
            "2026-10-2T0:0:0Z",
            "2026-13-40T00:00:00Z",
            None,
            12345,
        ):
            with self.subTest(generatedAt=bad):
                document = honest_document()
                document["generatedAt"] = bad
                self.refuse(document, "R2")

    def test_r2_missing_generated_at(self):
        document = honest_document()
        del document["generatedAt"]
        self.refuse(document, "R2")

    def test_r2_older_than_bundled_even_when_a_superset(self):
        # M3 case: a strict superset of the bundled store, but older.
        document = honest_document()
        document["generatedAt"] = later(document, days=-3)
        self.refuse(document, "R2")

    def test_r2_equal_generated_at_with_different_bytes(self):
        document = honest_document()
        document["generatedAt"] = bundled_document()["generatedAt"]
        self.refuse(document, "R2")

    def test_r2_generated_at_more_than_24_hours_in_the_future_refuses(self):
        now = datetime(2026, 10, 3, 12, 0, 0)
        document = honest_document()
        document["generatedAt"] = (now + timedelta(hours=25)).strftime(TIME_FORMAT)
        body = serialize(document)
        code, stdout, stderr = self.run_pull(body, now=now)
        self.assert_refused_nothing_written(
            code, stderr, reason=f"R2 generatedAt {document['generatedAt']} is in the future"
        )
        self.assertEqual(stdout, "")

    def test_r2_generated_at_within_24_hours_ahead_is_allowed(self):
        # One hour past the bundled store's own stamp, so a document dated 23-24 h ahead
        # of `now` is newer than the bundled store (R2's rollback check) whatever it holds.
        now = datetime.strptime(bundled_document()["generatedAt"], TIME_FORMAT) + timedelta(hours=1)
        for delta in (timedelta(hours=23), timedelta(hours=24)):
            with self.subTest(delta=delta):
                document = honest_document()
                document["generatedAt"] = (now + delta).strftime(TIME_FORMAT)
                code, _, stderr = self.run_pull(serialize(document), now=now, output=None)
                self.assertEqual(code, 0, stderr)

    def test_r2_now_is_injectable_and_defaults_to_the_real_utc_clock(self):
        real = CARDS.utc_now()
        self.assertIsInstance(real, datetime)
        self.assertIsNone(real.tzinfo)  # naive UTC, comparable with strptime output
        self.assertLess(abs((datetime.now(timezone.utc).replace(tzinfo=None) - real).total_seconds()), 60)

    # R3
    def test_r3_dropping_a_bundled_no_go_card_refuses(self):
        # M2 case: newer, otherwise honest, but a NO_GO card vanished.
        document = honest_document()
        dropped = next(c for c in document["cards"] if c["verdict"] == "NO_GO")
        document["cards"] = [c for c in document["cards"] if c is not dropped]
        self.refuse(document, "R3")

    def test_r3_names_the_dropped_id(self):
        document = honest_document()
        dropped = document["cards"].pop(0)
        body = serialize(document)
        code, _, stderr = self.run_pull(body)
        self.assert_refused_nothing_written(code, stderr, reason="R3")
        self.assertIn(dropped["id"], stderr)

    # R4
    def test_r4_duplicate_card_ids(self):
        document = honest_document()
        document["cards"].append(copy.deepcopy(document["cards"][-1]))
        self.refuse(document, "R4")

    # R5
    def test_r5_non_dict_model(self):
        for bad_model in ("a string", ["list"], None, 7):
            with self.subTest(model=bad_model):
                document = honest_document()
                document["cards"][-1]["model"] = bad_model
                self.refuse(document, "R5")

    def test_r5_missing_model_or_missing_repo_key(self):
        document = honest_document()
        del document["cards"][-1]["model"]
        self.refuse(document, "R5")
        document = honest_document()
        del document["cards"][-1]["model"]["repo"]
        self.refuse(document, "R5")

    def test_r5_non_string_repo(self):
        document = honest_document()
        document["cards"][-1]["model"]["repo"] = 12
        self.refuse(document, "R5")

    def test_r5_non_dict_card_or_non_string_id(self):
        document = honest_document()
        document["cards"].append("not a card")
        self.refuse(document, "R5")
        document = honest_document()
        document["cards"][-1]["id"] = ["unhashable"]
        self.refuse(document, "R5")

    def test_r5_unrecognized_or_missing_verdict(self):
        for bad in ("pass", "MAYBE", "", None, 3):
            with self.subTest(verdict=bad):
                document = honest_document()
                document["cards"][-1]["verdict"] = bad
                self.refuse(document, "R5")
        document = honest_document()
        del document["cards"][-1]["verdict"]
        self.refuse(document, "R5")

    # R6: a kept id must be DEEP-EQUAL to the bundled card; the one allowed
    # difference is ``verdict`` going from an admitting verdict to NO_GO.
    def test_r6_relaxing_no_go_to_an_admitted_verdict(self):
        for relaxed in ("PASS", "REFERENCE", "EXACT", "UNMEASURED"):
            with self.subTest(to=relaxed):
                document = honest_document()
                target = next(c for c in document["cards"] if c["verdict"] == "NO_GO")
                target["verdict"] = relaxed
                code, _, stderr = self.run_pull(serialize(document))
                self.assert_refused_nothing_written(
                    code, stderr, reason=f"R6 card {target['id']} relaxes verdict NO_GO -> {relaxed}"
                )

    def test_r6_changing_between_admitted_verdicts_refuses(self):
        # Formerly allowed as "not relaxing"; a within-tier swap can undo
        # resolve_card's exit-3 mixed-verdict refusal, so it is refused now.
        for swapped in ("EXACT", "REFERENCE", "UNMEASURED"):
            with self.subTest(to=swapped):
                document = honest_document()
                target = next(c for c in document["cards"] if c["verdict"] == "PASS")
                target["verdict"] = swapped
                code, _, stderr = self.run_pull(serialize(document))
                self.assert_refused_nothing_written(
                    code,
                    stderr,
                    reason=f"R6 card {target['id']} changes verdict relative to the bundled store",
                )

    def test_r6_no_go_is_never_changed_to_anything_else_even_with_other_keys(self):
        document = honest_document()
        target = next(c for c in document["cards"] if c["id"] == NO_GO_ID)
        target["verdict"] = "PASS"
        target["admission"] = {"default": True, "optIn": True, "reason": "edited"}
        code, _, stderr = self.run_pull(serialize(document))
        self.assert_refused_nothing_written(
            code, stderr, reason=f"R6 card {NO_GO_ID} changes admission,verdict relative to the bundled store"
        )

    def test_r6_tightening_with_everything_else_equal_is_allowed(self):
        document = honest_document()
        target = next(c for c in document["cards"] if c["verdict"] == "PASS")
        before = copy.deepcopy(target)
        target["verdict"] = "NO_GO"
        self.assertEqual({k for k in before if before[k] != target[k]}, {"verdict"})
        code, _, stderr = self.run_pull(serialize(document))
        self.assertEqual(code, 0, stderr)

    def test_r6_tightening_that_also_changes_another_key_refuses(self):
        document = honest_document()
        target = next(c for c in document["cards"] if c["verdict"] == "PASS")
        target["verdict"] = "NO_GO"
        target["model"]["hfPin"] = "deadbeef"
        code, _, stderr = self.run_pull(serialize(document))
        self.assert_refused_nothing_written(
            code, stderr, reason=f"R6 card {target['id']} changes model,verdict relative to the bundled store"
        )

    def test_r6_adding_new_ids_stays_allowed(self):
        document = honest_document()
        document["cards"].append(new_card("another-new@test", "example/AnotherNew", "PASS"))
        code, _, stderr = self.run_pull(serialize(document))
        self.assertEqual(code, 0, stderr)
        self.assertTrue(stderr.strip().endswith("added=2"), stderr)

    # The hole R6 closes: the edited NO_GO card passed every other rule and
    # turned a resident launch from refuse_quality_flagged into
    # admit_unmeasured. Each test (1) proves the edited store WOULD admit in
    # the launcher (control) while the bundled one refuses, then (2) proves
    # the pull refuses it.
    def launcher_outcome(self, cards, repo=NO_GO_REPO, revision=NO_GO_REVISION):
        card = CARDS.launch.resolve_card(cards, repo, revision, residency="resident")
        return CARDS.launch.decide_admission(card, False)[0]

    def assert_hole_is_real_then_refused(self, document, keys, revision=NO_GO_REVISION):
        self.assertEqual(
            self.launcher_outcome(bundled_document()["cards"], revision=revision),
            "refuse_quality_flagged",
        )
        self.assertEqual(
            self.launcher_outcome(document["cards"], revision=revision), "admit_unmeasured"
        )
        code, stdout, stderr = self.run_pull(serialize(document))
        self.assert_refused_nothing_written(
            code, stderr, reason=f"R6 card {NO_GO_ID} changes {keys} relative to the bundled store"
        )
        self.assertEqual(stdout, "")

    def test_r6_a_residency_edit_on_the_bundled_no_go_card_refuses(self):
        document = honest_document()
        target = next(c for c in document["cards"] if c["id"] == NO_GO_ID)
        target["config"]["residency"] = "expert-stream"
        self.assert_hole_is_real_then_refused(document, "config")

    def test_r6_a_model_repo_edit_on_the_bundled_no_go_card_refuses(self):
        document = honest_document()
        target = next(c for c in document["cards"] if c["id"] == NO_GO_ID)
        target["model"]["repo"] = "example/Renamed-Qwen3.8-27B"
        # A repo-only edit leaves the unchanged hfPin matching a launch that
        # knows its revision, so the hole is a repo-keyed (no revision) launch.
        self.assertEqual(
            self.launcher_outcome(document["cards"]), "refuse_quality_flagged"
        )
        self.assert_hole_is_real_then_refused(document, "model", revision=None)

    def test_r6_a_repo_and_hfpin_edit_on_the_bundled_no_go_card_refuses(self):
        document = honest_document()
        target = next(c for c in document["cards"] if c["id"] == NO_GO_ID)
        target["model"]["repo"] = "example/Renamed-Qwen3.8-27B"
        target["model"]["hfPin"] = "c0ffee00"
        self.assert_hole_is_real_then_refused(document, "model")

    def test_r6_a_pass_to_exact_swap_on_a_bundled_pass_card_refuses(self):
        document = honest_document()
        target = next(c for c in document["cards"] if c["verdict"] == "PASS")
        target["verdict"] = "EXACT"
        code, stdout, stderr = self.run_pull(serialize(document))
        self.assert_refused_nothing_written(
            code, stderr, reason=f"R6 card {target['id']} changes verdict relative to the bundled store"
        )
        self.assertEqual(stdout, "")

    def test_r6_an_added_extra_key_on_a_kept_card_refuses(self):
        document = honest_document()
        target = next(c for c in document["cards"] if c["verdict"] == "PASS")
        target["extraKey"] = "anything"
        code, stdout, stderr = self.run_pull(serialize(document))
        self.assert_refused_nothing_written(
            code, stderr, reason=f"R6 card {target['id']} changes extraKey relative to the bundled store"
        )
        self.assertEqual(stdout, "")

    def test_r6_a_removed_key_and_a_json_type_change_on_a_kept_card_refuse(self):
        document = honest_document()
        target = next(c for c in document["cards"] if c["verdict"] == "PASS")
        del target["legible"]
        self.refuse(document, f"R6 card {target['id']} changes legible relative to the bundled store")
        document = honest_document()
        target = next(c for c in document["cards"] if c["id"] == NO_GO_ID)
        # json-equal, not Python-equal: 1 == 1.0 == True in Python.
        target["config"]["quant"]["bits"] = float(target["config"]["quant"]["bits"])
        self.refuse(document, f"R6 card {NO_GO_ID} changes config relative to the bundled store")

    def test_r6_tightening_verdict_is_declared_over_the_launcher_vocabulary(self):
        self.assertIn(CARDS.TIGHTENING_VERDICT, CARDS.launch._RECOGNIZED_VERDICTS)
        self.assertEqual(CARDS.TIGHTENING_VERDICT, "NO_GO")

    # R7
    def test_r7_no_resolvable_bundled_baseline(self):
        empty_root = self.root / "empty-repo"
        (empty_root / "site").mkdir(parents=True)
        self.refuse(serialize(honest_document()), "R7", baseline_root=empty_root)

    def test_r7_unusable_bundled_baseline(self):
        for bad in (b"not json", b'{"schema": "wrong", "generatedAt": "2026-01-01T00:00:00Z", "cards": []}',
                    b'{"schema": "fast-mlx-quality-card-v1", "generatedAt": "nope", "cards": []}'):
            with self.subTest(bad=bad[:20]):
                root = self.root / f"repo-{hashlib.sha256(bad).hexdigest()[:6]}"
                (root / "site").mkdir(parents=True)
                (root / "site" / "quality-guides.json").write_bytes(bad)
                self.refuse(serialize(honest_document()), "R7", baseline_root=root)


# ---------------------------------------------------------------------
# P5: argument validation and transport failures
# ---------------------------------------------------------------------
class ArgumentAndTransportTests(CardsPullTestCase):
    def test_p5_malformed_commit_exits_2_without_fetching(self):
        body = serialize(honest_document())
        for bad in ("main", "a" * 39, "a" * 41, "A" * 40, "g" * 40, "", "a" * 40 + "\n"):
            with self.subTest(commit=bad):
                code, stdout, stderr = self.run_pull(body, commit=bad)
                self.assertEqual(code, 2, stderr)
                self.assertEqual(stdout, "")
                self.assertEqual(self.opener.calls, [])
                self.assertEqual(self.all_files_under(self.out_dir), [])

    def test_p5_malformed_sha_exits_3_without_fetching(self):
        body = serialize(honest_document())
        digest = hashlib.sha256(body).hexdigest()
        for bad in ("xyz", "", "g" * 64, digest[:63], digest + "0"):
            with self.subTest(sha=bad):
                code, stdout, stderr = self.run_pull(body, sha=bad)
                self.assertEqual(code, 3, stderr)
                self.assertEqual(stdout, "")
                self.assertIn("--sha256", stderr)
                self.assertEqual(self.opener.calls, [])
                self.assertEqual(self.all_files_under(self.out_dir), [])

    def test_missing_required_flags_are_argparse_exit_2(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as ctx:
            CARDS.main(["pull", "--commit", COMMIT])
        self.assertEqual(ctx.exception.code, 2)
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as ctx:
            CARDS.main([])
        self.assertEqual(ctx.exception.code, 2)

    def test_p5_cross_host_final_url_exits_1_and_writes_nothing(self):
        body = serialize(honest_document())
        opener = FakeOpener(body, final_url="https://example.com/site/quality-guides.json")
        code, stdout, stderr = self.run_pull(body, opener=opener)
        self.assert_refused_nothing_written(code, stderr, expected_code=1)
        self.assertEqual(stdout, "")

    def test_p5_non_https_final_url_exits_1_and_writes_nothing(self):
        body = serialize(honest_document())
        opener = FakeOpener(
            body, final_url="http://raw.githubusercontent.com/x/quality-guides.json"
        )
        code, _, stderr = self.run_pull(body, opener=opener)
        self.assert_refused_nothing_written(code, stderr, expected_code=1)

    def test_p5_redirect_refusal_raised_by_the_opener_exits_1(self):
        body = serialize(honest_document())
        opener = FakeOpener(body, raises=CARDS.CardsFetchError("refused a redirect"))
        code, _, stderr = self.run_pull(body, opener=opener)
        self.assert_refused_nothing_written(code, stderr, expected_code=1, reason="refused a redirect")

    def test_p5_oversize_body_exits_1_and_writes_nothing(self):
        body = b" " * (FOUR_MIB + 1)
        code, stdout, stderr = self.run_pull(body, sha=hashlib.sha256(body).hexdigest())
        self.assert_refused_nothing_written(code, stderr, expected_code=1)
        self.assertEqual(stdout, "")
        self.assertEqual(self.opener.response.read_sizes, [FOUR_MIB + 1])

    def test_a_body_of_exactly_four_mib_is_not_refused_for_size(self):
        document = honest_document()
        body = serialize(document)
        body += b" " * (FOUR_MIB - len(body))
        self.assertEqual(len(body), FOUR_MIB)
        code, _, stderr = self.run_pull(body)
        self.assertEqual(code, 0, stderr)

    def test_p5_network_failures_exit_1(self):
        body = serialize(honest_document())
        for error in (
            urllib.error.URLError("no route"),
            urllib.error.HTTPError(EXPECTED_URL, 404, "Not Found", {}, io.BytesIO(b"")),
            TimeoutError("timed out"),
            OSError("connection reset"),
        ):
            with self.subTest(error=type(error).__name__):
                code, stdout, stderr = self.run_pull(body, opener=FakeOpener(raises=error))
                self.assert_refused_nothing_written(code, stderr, expected_code=1)
                self.assertEqual(stdout, "")

    def test_p5_http_exceptions_exit_1_and_write_nothing(self):
        body = serialize(honest_document())
        for error in (http.client.IncompleteRead(b"abc", 10), http.client.BadStatusLine("x")):
            with self.subTest(error=type(error).__name__):
                code, stdout, stderr = self.run_pull(body, opener=FakeOpener(raises=error))
                self.assert_refused_nothing_written(code, stderr, expected_code=1)
                self.assertEqual(stdout, "")

    def test_p5_http_exception_while_reading_the_body_exits_1(self):
        body = serialize(honest_document())
        opener = FakeOpener(body)
        real_open = opener.open

        def open_then_fail(request, timeout=None):
            response = real_open(request, timeout)
            response.read = lambda size=-1: (_ for _ in ()).throw(
                http.client.IncompleteRead(b"par", 5)
            )
            return response

        opener.open = open_then_fail
        code, stdout, stderr = self.run_pull(body, opener=opener)
        self.assert_refused_nothing_written(code, stderr, expected_code=1)
        self.assertEqual(stdout, "")

    def test_r1_pathologically_nested_json_is_unparsable_not_a_crash(self):
        body = b'{"schema": "fast-mlx-quality-card-v1", "cards": ' + b"[" * 200000 + b"]" * 200000 + b"}"
        with self.assertRaises(RecursionError):
            json.loads(body.decode("utf-8"))  # the input really recurses
        code, stdout, stderr = self.run_pull(body)
        self.assert_refused_nothing_written(code, stderr, reason="R1")
        self.assertEqual(stdout, "")
        self.assertNotIn("Traceback", stderr)

    # The REAL redirect handler (no network: called directly).
    def test_redirect_handler_refuses_non_https_and_cross_host(self):
        handler = CARDS.CardsRedirectHandler()
        request = urllib.request.Request(EXPECTED_URL)
        for new_url in (
            "http://raw.githubusercontent.com/x",
            "https://example.com/x",
            "https://raw.githubusercontent.com.evil.example/x",
            "https://user@evil.example/x",
            "https://user@raw.githubusercontent.com/x",  # same host, with userinfo
            "https://raw.githubusercontent.com:443/x",  # same host, explicit port
            "file:///etc/passwd",
        ):
            with self.subTest(url=new_url):
                with self.assertRaises(CARDS.CardsFetchError):
                    handler.redirect_request(request, None, 302, "Found", {}, new_url)

    def test_p5_same_host_userinfo_or_explicit_port_final_url_exits_1_and_writes_nothing(self):
        body = serialize(honest_document())
        for final_url in (
            "https://user@raw.githubusercontent.com/x",
            "https://raw.githubusercontent.com:443/x",
        ):
            with self.subTest(final_url=final_url):
                code, stdout, stderr = self.run_pull(body, opener=FakeOpener(body, final_url=final_url))
                self.assert_refused_nothing_written(code, stderr, expected_code=1)
                self.assertEqual(stdout, "")

    def test_redirect_handler_follows_same_host_https(self):
        handler = CARDS.CardsRedirectHandler()
        request = urllib.request.Request(EXPECTED_URL)
        followed = handler.redirect_request(
            request, None, 302, "Found", {}, "https://raw.githubusercontent.com/other/path"
        )
        self.assertIsNotNone(followed)
        self.assertEqual(followed.full_url, "https://raw.githubusercontent.com/other/path")

    def test_real_opener_is_built_with_the_confined_redirect_handler(self):
        opener = CARDS.https_opener()
        self.assertTrue(
            any(isinstance(h, CARDS.CardsRedirectHandler) for h in opener.handlers)
        )


# ---------------------------------------------------------------------
# P2: the printed flags drive the real launcher / recommend entry points.
# ---------------------------------------------------------------------
class PulledStoreDrivesLauncherTests(CardsPullTestCase):
    def setUp(self):
        super().setUp()
        self.body = serialize(honest_document())
        self.digest = hashlib.sha256(self.body).hexdigest()
        code, stdout, stderr = self.run_pull(self.body)
        self.assertEqual(code, 0, stderr)
        self.flags = shlex.split(stdout)
        self.assertEqual(
            self.flags, ["--quality-cards", str(self.output), "--quality-cards-sha256", self.digest]
        )

    def test_p2_serve_dry_run_admits_against_the_new_card(self):
        model_dir = self.root / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        fit_bin = launch_tests.write_script(
            self.root / "fit-green.py", launch_tests.GREEN_FIT_CHECK_BODY
        )
        engine_bin = launch_tests.write_script(
            self.root / "fake-engine.py", launch_tests.FAKE_ENGINE_BODY
        )
        empty_root = self.root / "empty-repo"
        (empty_root / "site").mkdir(parents=True)

        def serve(*extra, dry_run=True):
            argv = [
                "serve", "--model-path", str(model_dir), "--model-repo", NEW_CARD_REPO,
                "--fit-check-bin", str(fit_bin), "--engine-bin", str(engine_bin),
                "--context", "2048", *(["--dry-run"] if dry_run else []),
                *self.flags, *extra,
            ]
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                with patch.object(launch_tests.FASTMLX_LAUNCH.os, "execv", lambda *a, **k: None):
                    with patch.object(launch_tests.FASTMLX_LAUNCH, "REPO_ROOT", empty_root):
                        with self.assertRaises(SystemExit) as ctx:
                            launch_tests.FASTMLX_LAUNCH.main(argv)
            return ctx.exception.code, stdout.getvalue(), stderr.getvalue()

        # The new card is NO_GO: admission refuses without opt-in ...
        code, _, stderr = serve()
        self.assertEqual(code, 2, stderr)  # quality refusal (exit 2, like a RED verdict)
        self.assertIn(f"--accept-quality {NEW_CARD_ID}", stderr)
        # ... and admits with it, naming the pulled store in the plan.
        code, stdout, stderr = serve("--accept-quality", NEW_CARD_ID)
        self.assertEqual(code, 0, stderr)
        plan = json.loads(stdout.strip().splitlines()[-1])
        self.assertEqual(plan["cardStore"]["sha256"], self.digest)
        self.assertEqual(plan["cardStore"]["cards"], len(honest_document()["cards"]))
        self.assertEqual(plan["cardStore"]["generatedAt"], honest_document()["generatedAt"])
        # A real (non-dry-run) launch names the store as explicit, by digest.
        code, _, stderr = serve("--accept-quality", NEW_CARD_ID, dry_run=False)
        self.assertEqual(code, 0, stderr)
        self.assertIn(
            f"fastmlx_launch=card_store source=explicit sha256={self.digest} ", stderr
        )

    def test_p2_recommend_json_ranks_against_the_new_card(self):
        model_dir = self.root / "rec-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        recommend_tests.write_pull_receipt(model_dir, repo_id=NEW_CARD_REPO, revision="e" * 40)
        fit_bin = recommend_tests.write_script(
            self.root / "rec-fit-green.py", recommend_tests.GREEN_FIT_CHECK_BODY
        )
        empty_root = self.root / "empty-repo"
        (empty_root / "site").mkdir(parents=True)
        argv = [
            "recommend", "--model-path", str(model_dir), "--fit-check-bin", str(fit_bin),
            "--json", *self.flags,
        ]
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with patch.object(recommend_tests.FASTMLX_RECOMMEND.launch, "REPO_ROOT", empty_root):
                with self.assertRaises(SystemExit) as ctx:
                    recommend_tests.FASTMLX_RECOMMEND.main(argv)
        self.assertEqual(ctx.exception.code, 1, stderr.getvalue())  # opt-in only: nothing recommended
        document = json.loads(stdout.getvalue())
        self.assertEqual(document["cardStore"]["sha256"], self.digest)
        (row,) = document["rows"]
        self.assertEqual(row["status"], "opt-in")
        self.assertEqual(row["accept_quality_flag"], f"--accept-quality {NEW_CARD_ID}")


# ---------------------------------------------------------------------
# Decode refusal at pull time (predeclaration rows P1-P5):
# docs/task-inbox/2026-10-04-PREDECLARATION-cards-pull-refuses-a-store-the-built-in-engine-cannot-decode.md
#
# A store the built-in engine cannot decode is refused at PULL time, naming
# the card and the field, before anything is written. The shape tables are the
# launcher's own (cycle 201), imported -- never re-declared here.
# ---------------------------------------------------------------------
# Fields R5 (id / model / model.repo / verdict / object-ness) already refuses
# for the shapes below; for those the refusal is R5's, not the decode check's.
R5_REFUSED_FIELDS = {"id", "model", "model.repo", "verdict", "element"}
# Decodable shapes R5 itself refuses (unrecognized verdict spellings; a model
# object with no ``repo`` key), so they cannot be pulled for another reason.
R5_REFUSED_DECODABLE_SHAPES = {
    "verdict no_go lowercase",
    "verdict unrecognized string",
    "verdict empty string",
    "repo missing",
}


class UndecodablePullRefusalTests(CardsPullTestCase):
    def body_with(self, card) -> bytes:
        document = bundled_document()
        document["generatedAt"] = later(document)
        document["cards"].append(card)
        return serialize(document)

    def element_index(self) -> int:
        return len(bundled_document()["cards"])

    # --- P1 ---------------------------------------------------------------
    def test_p1_each_undecodable_shape_is_refused_before_anything_is_written(self):
        for index, (name, build, field) in enumerate(launch_tests.UNDECODABLE_CARD_SHAPES):
            with self.subTest(shape=name):
                # A fresh output directory per shape: one shape's leftover
                # file must not read as the next shape's.
                self.out_dir = self.root / f"out-{index}"
                self.out_dir.mkdir()
                self.output = self.out_dir / "store.json"
                body = self.body_with(build())
                code, stdout, stderr = self.run_pull(body)
                self.assert_refused_nothing_written(code, stderr)
                self.assertEqual(stdout, "")
                self.assertFalse(self.output.exists())
                if field in R5_REFUSED_FIELDS:
                    self.assertIn("R5", stderr)  # already refused, earlier, by R5
                    continue
                self.assertNotIn("R5", stderr)
                self.assertIn("cannot be decoded by the built-in engine", stderr)
                self.assertIn(field, stderr)
                self.assertIn(repr(launch_tests.OTHER_CARD_ID), stderr)
                self.assertIn(f"element {self.element_index()}", stderr)

    def test_p1_the_refusal_is_one_line_and_never_advises_a_repull(self):
        body = self.body_with(launch_tests._other_card(legible={"tier": "T"}))
        code, _, stderr = self.run_pull(body)
        self.assert_refused_nothing_written(code, stderr)
        self.assertEqual(len(stderr.strip().splitlines()), 1, stderr)
        self.assertNotIn("re-pull", stderr)

    def test_p1_a_hostile_card_id_is_bounded_to_one_line(self):
        hostile = "evil\nfastmlx_cards=pulled forged " + "x" * 500
        body = self.body_with(launch_tests._other_card(id=hostile, legible=None))
        code, stdout, stderr = self.run_pull(body)
        self.assert_refused_nothing_written(code, stderr)
        self.assertEqual(stdout, "")
        self.assertEqual(len(stderr.strip().splitlines()), 1, stderr)
        self.assertIn("evil", stderr)
        self.assertNotIn("x" * 100, stderr)

    def test_p1_the_first_undecodable_card_is_the_one_named(self):
        document = bundled_document()
        document["generatedAt"] = later(document)
        document["cards"] += [
            launch_tests._other_card(id="second-ok@test"),
            launch_tests._other_card(id="third-bad@test", legible="x"),
            launch_tests._other_card(id="fourth-bad@test", legible=None),
        ]
        code, _, stderr = self.run_pull(serialize(document))
        self.assert_refused_nothing_written(code, stderr)
        self.assertIn("third-bad@test", stderr)
        self.assertIn(f"element {self.element_index() + 1}", stderr)
        self.assertNotIn("fourth-bad@test", stderr)

    def test_p1_refused_with_the_default_output_too(self):
        body = self.body_with(launch_tests._other_card(legible=None))
        code, stdout, stderr = self.run_pull(body, output=None)
        self.assert_refused_nothing_written(code, stderr)
        self.assertEqual(stdout, "")

    # --- P2 ---------------------------------------------------------------
    def test_p2_each_decodable_shape_r5_accepts_is_pulled(self):
        pulled_any = False
        for name, build in launch_tests.DECODABLE_CARD_SHAPES:
            if name in R5_REFUSED_DECODABLE_SHAPES:
                continue
            with self.subTest(shape=name):
                pulled_any = True
                output = self.out_dir / f"{name.replace(' ', '-')}.json"
                body = self.body_with(build())
                digest = hashlib.sha256(body).hexdigest()
                code, stdout, stderr = self.run_pull(body, output=output)
                self.assertEqual(code, 0, stderr)
                self.assertNotIn("cannot be decoded", stderr)
                self.assertEqual(output.read_bytes(), body)
                self.assertEqual(
                    stdout,
                    f"--quality-cards {shlex.quote(str(output))} --quality-cards-sha256 {digest}\n",
                )
        self.assertTrue(pulled_any)

    def test_p2_the_excluded_shapes_really_are_refused_by_r5(self):
        for name, build in launch_tests.DECODABLE_CARD_SHAPES:
            if name not in R5_REFUSED_DECODABLE_SHAPES:
                continue
            with self.subTest(shape=name):
                code, _, stderr = self.run_pull(self.body_with(build()))
                self.assert_refused_nothing_written(code, stderr, reason="R5")

    # --- P5 ---------------------------------------------------------------
    def test_p5_the_shipped_store_pulls_and_every_card_is_engine_decodable(self):
        body = BUNDLED_STORE_PATH.read_bytes()
        digest = hashlib.sha256(body).hexdigest()
        code, stdout, stderr = self.run_pull(body)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(self.output.read_bytes(), body)
        self.assertEqual(
            stdout,
            f"--quality-cards {shlex.quote(str(self.output))} --quality-cards-sha256 {digest}\n",
        )
        for card in bundled_document()["cards"]:
            self.assertIsNone(CARDS.launch.engine_undecodable_card_reason(card))


# ---------------------------------------------------------------------
# P3: the pull-time check is NOT a card-store rule. An already-present
# undecodable pulled store (one that passed R1-R7 before this change) is
# still read by a custom-engine serve and by ``recommend`` exactly as before.
# ---------------------------------------------------------------------
class UndecodableStoreOtherConsumersTests(CardsPullTestCase):
    def setUp(self):
        super().setUp()
        document = honest_document()
        document["cards"].append(launch_tests._other_card(legible=None))
        self.body = serialize(document)
        self.digest = hashlib.sha256(self.body).hexdigest()
        self.pulled = self.root / "pulled"
        self.pulled.mkdir()
        (self.pulled / f"{self.digest}.json").write_bytes(self.body)
        self.generated_at = document["generatedAt"]

    def test_p3_the_store_passes_rules_r1_r7_though_the_pull_check_refuses_it(self):
        # The store is a pull the new check refuses; it must still pass the
        # rules R1-R7 that gate every other consumer (the check is not a rule).
        document, _ = launch_tests.FASTMLX_LAUNCH.check_card_store_rules(
            self.body,
            baseline_path=BUNDLED_STORE_PATH,
            now=datetime.strptime(self.generated_at, TIME_FORMAT),
        )
        self.assertEqual(document["generatedAt"], self.generated_at)
        code, _, stderr = self.run_pull(self.body)
        self.assert_refused_nothing_written(code, stderr)
        self.assertIn("cannot be decoded by the built-in engine", stderr)

    def test_p3_custom_engine_profile_serve_admits_against_the_store(self):
        launch = launch_tests.FASTMLX_LAUNCH
        model_dir = self.root / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        fit_bin = launch_tests.write_script(
            self.root / "fit-green.py", launch_tests.GREEN_FIT_CHECK_BODY
        )
        engine_bin = launch_tests.write_script(
            self.root / "fake-engine.py", launch_tests.FAKE_ENGINE_BODY
        )
        profile = launch_tests.write_custom_engine_profile(self.root)
        argv = [
            "serve", "--model-path", str(model_dir), "--model-repo", NEW_CARD_REPO,
            "--fit-check-bin", str(fit_bin), "--engine-bin", str(engine_bin),
            "--engine-profile", str(profile), "--context", "2048",
            "--accept-quality", NEW_CARD_ID,
        ]
        stdout, stderr = io.StringIO(), io.StringIO()
        execed = []
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with patch.object(launch.os, "execv", lambda path, a: execed.append(list(a))):
                with patch.object(launch, "REPO_ROOT", REPO_ROOT):
                    with patch.object(launch, "pulled_cards_dir", lambda: self.pulled):
                        with self.assertRaises(SystemExit) as ctx:
                            launch.main(argv)
        self.assertEqual(ctx.exception.code, 0, stderr.getvalue())
        self.assertNotIn("cannot be decoded", stderr.getvalue())
        self.assertIn(f"source=pulled sha256={self.digest} ", stderr.getvalue())
        self.assertEqual(len(execed), 1, stderr.getvalue())

    def test_p3_recommend_reads_the_store_without_a_decode_refusal(self):
        model_dir = self.root / "rec-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        recommend_tests.write_pull_receipt(model_dir, repo_id=NEW_CARD_REPO, revision="e" * 40)
        fit_bin = recommend_tests.write_script(
            self.root / "rec-fit-green.py", recommend_tests.GREEN_FIT_CHECK_BODY
        )
        recommend = recommend_tests.FASTMLX_RECOMMEND
        argv = [
            "recommend", "--model-path", str(model_dir), "--fit-check-bin", str(fit_bin), "--json",
        ]
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with patch.object(recommend.launch, "REPO_ROOT", REPO_ROOT):
                with patch.object(recommend.launch, "pulled_cards_dir", lambda: self.pulled):
                    with self.assertRaises(SystemExit) as ctx:
                        recommend.main(argv)
        self.assertEqual(ctx.exception.code, 1, stderr.getvalue())  # opt-in only
        self.assertNotIn("cannot be decoded", stderr.getvalue())
        document = json.loads(stdout.getvalue())
        self.assertEqual(document["cardStore"]["sha256"], self.digest)
        (row,) = document["rows"]
        self.assertEqual(row["status"], "opt-in")


# ---------------------------------------------------------------------
# P7: serve / recommend never fetch.
# ---------------------------------------------------------------------
class ServeAndRecommendNeverFetchTests(unittest.TestCase):
    def test_p7_card_store_tests_pass_with_every_fetch_path_patched_to_raise(self):
        def boom(*args, **kwargs):
            raise AssertionError("network fetch attempted by serve/recommend")

        suites = unittest.TestSuite()
        loader = unittest.TestLoader()
        suites.addTests(
            loader.loadTestsFromTestCase(launch_tests.QualityCardStoreIdentityTestCase)
        )
        suites.addTests(
            loader.loadTestsFromTestCase(recommend_tests.RecommendCardStoreTestCase)
        )
        self.assertGreater(suites.countTestCases(), 20)
        with patch.object(CARDS, "https_opener", boom), patch.object(
            urllib.request, "urlopen", boom
        ), patch.object(urllib.request.OpenerDirector, "open", boom):
            result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suites)
        self.assertTrue(
            result.wasSuccessful(), [str(f[0]) + f[1] for f in result.failures + result.errors]
        )

    def test_p7_serve_and_recommend_sources_never_reference_the_fetcher(self):
        scripts = Path(__file__).resolve().parents[1]
        for name in ("fastmlx_launch.py", "fastmlx_recommend.py"):
            with self.subTest(module=name):
                text = (scripts / name).read_text(encoding="utf-8")
                self.assertNotIn("fastmlx_cards", text)
                self.assertNotIn("raw.githubusercontent.com", text)



# ---------------------------------------------------------------------
# ``fastmlx cards list``: which models carry a quality card, offline.
# Predeclaration rows A1-A8:
# docs/task-inbox/2026-10-06-PREDECLARATION-cards-list-answers-which-models-are-carded-before-download.md
# ---------------------------------------------------------------------
GUIDES_PATH = REPO_ROOT / "site" / "quality-guides.json"
QWEN3_8B_8BIT_REPO = "mlx-community/Qwen3-8B-8bit"
HEADER_PREFIX = "card store: "


def _raise_network(*args, **kwargs):
    raise AssertionError("cards list touched the network")


class CardsListTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.home = self.root / "home"
        self.home.mkdir()

    def run_list(self, *argv, host_class="apple-m3-ultra"):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"HOME": str(self.home)}))
            # `list` is offline: any fetch path raising proves it never calls one.
            stack.enter_context(patch.object(CARDS, "fetch", _raise_network))
            stack.enter_context(patch.object(CARDS, "https_opener", _raise_network))
            stack.enter_context(patch.object(urllib.request, "urlopen", _raise_network))
            stack.enter_context(
                patch.object(CARDS.launch, "host_hardware_class", lambda: host_class)
            )
            stack.enter_context(contextlib.redirect_stdout(stdout))
            stack.enter_context(contextlib.redirect_stderr(stderr))
            with self.assertRaises(SystemExit) as ctx:
                CARDS.main(["list", *argv])
        return ctx.exception.code, stdout.getvalue(), stderr.getvalue()

    @staticmethod
    def split_output(stdout):
        lines = [line for line in stdout.splitlines() if line.strip()]
        header = lines[0] if lines else ""
        return header, lines[1:]

    @staticmethod
    def row_for(rows, card_id):
        found = [row for row in rows if row.startswith(card_id + " ")]
        assert len(found) == 1, (card_id, rows)
        return found[0]

    def write_store(self, cards, name="store.json"):
        document = bundled_document()
        document["cards"] = cards
        path = self.root / name
        path.write_bytes(serialize(document))
        return path


class CardsListBundledStoreTests(CardsListTestCase):
    def test_a1_bundled_store_lists_every_card_with_the_store_identity(self):
        code, stdout, stderr = self.run_list()
        self.assertEqual(code, 0, stderr)
        header, rows = self.split_output(stdout)
        guides = json.loads(GUIDES_PATH.read_text(encoding="utf-8"))
        ids = [card["id"] for card in guides["cards"]]
        self.assertEqual(len(rows), 15)
        self.assertEqual([row.split(" ")[0] for row in rows], ids)  # store order
        self.assertTrue(header.startswith(HEADER_PREFIX + "source=default "), header)
        self.assertIn(
            "sha256=" + hashlib.sha256(GUIDES_PATH.read_bytes()).hexdigest(), header
        )
        self.assertIn("cards=15", header)
        self.assertEqual(stderr, "")

    def test_a1_every_row_names_verdict_admission_repo_revision_residency_hardware(self):
        _, stdout, _ = self.run_list()
        _, rows = self.split_output(stdout)
        no_go = self.row_for(rows, "qwen3-8b-4bit@m5")
        self.assertIn("| NO_GO |", no_go)
        self.assertIn("opt-in: --accept-quality qwen3-8b-4bit@m5", no_go)
        self.assertNotIn("default-eligible", no_go)
        self.assertIn("mlx-community/Qwen3-8B-4bit", no_go)
        self.assertIn("revision begins 545dc425", no_go)
        self.assertIn("residency resident", no_go)
        self.assertIn("hardware apple-m5", no_go)
        # A local pack carries no repo; a card with no residency says so.
        local = self.row_for(rows, "qwen38-flash-next-mixed-4-8bit@m3ultra")
        self.assertIn("repo=none (local pack)", local)
        self.assertIn("residency unrecorded", local)
        self.assertIn("revision begins ef5b919d31534faa1997666f1a22d362cd6383cd", local)
        # A card with no hfPin says so.
        unpinned = self.row_for(rows, "qwen38-27b-mtp@m3ultra")
        self.assertIn("revision unrecorded", unpinned)
        self.assertIn("| EXACT |", unpinned)
        self.assertIn("default-eligible", unpinned)

    def test_a2_model_filter_selects_the_one_measured_pass_card(self):
        code, stdout, stderr = self.run_list("--model", QWEN3_8B_8BIT_REPO)
        self.assertEqual(code, 0, stderr)
        header, rows = self.split_output(stdout)
        self.assertEqual(len(rows), 1, rows)
        self.assertTrue(rows[0].startswith("qwen3-8b-8bit@m3ultra "), rows[0])
        self.assertIn("| PASS |", rows[0])
        self.assertIn("default-eligible", rows[0])
        self.assertIn("revision begins 48a0b75b", rows[0])
        self.assertIn("cards=15", header)  # the header names the whole store

    def test_a3_model_with_no_card_exits_1_and_does_not_claim_unmeasured_on_bundled(self):
        # The bundled store holds pin-only (repo null) Flash Next cards, so a
        # repo miss cannot prove serve admits UNMEASURED; it points at --revision.
        code, stdout, stderr = self.run_list("--model", "nobody/x")
        self.assertEqual(code, 1)
        self.assertIn("no quality card names nobody/x by repo", stderr)
        self.assertIn("--revision", stderr)
        self.assertNotIn("UNMEASURED", stderr)
        self.assertEqual(self.split_output(stdout)[1], [])  # no card rows

    def test_a3_model_filter_is_exact_not_prefix_or_substring(self):
        for model in ("mlx-community/Qwen3-8B", "Qwen3-8B-8bit", "8bit", "mlx-community/"):
            with self.subTest(model=model):
                code, stdout, stderr = self.run_list("--model", model)
                self.assertEqual(code, 1, stdout)
                self.assertIn("no quality card names", stderr)
                self.assertEqual(self.split_output(stdout)[1], [])

    def test_a4_hardware_class_is_tagged_against_this_host(self):
        code, stdout, stderr = self.run_list(host_class="apple-m5")
        self.assertEqual(code, 0, stderr)
        _, rows = self.split_output(stdout)
        self.assertIn("hardware apple-m5 (this host)", self.row_for(rows, "qwen3-0p6b-4bit@m5"))
        other = self.row_for(rows, "qwen3-0p6b-4bit@m3ultra")
        self.assertIn("hardware apple-m3-ultra (other hardware)", other)
        self.assertNotIn("this host", other)

    def test_a4_unknown_host_is_not_called_other_hardware(self):
        _, stdout, _ = self.run_list(host_class=None)
        _, rows = self.split_output(stdout)
        row = self.row_for(rows, "qwen3-0p6b-4bit@m5")
        self.assertIn("hardware apple-m5 (host unknown)", row)
        self.assertNotIn("other hardware", row)

    def test_a4_card_without_hardware_class_reads_unrecorded(self):
        card = new_card("nohw@test", "example/NoHw", "PASS")
        store = self.write_store([card])
        _, stdout, _ = self.run_list("--quality-cards", str(store))
        _, rows = self.split_output(stdout)
        self.assertIn("hardware unrecorded", self.row_for(rows, "nohw@test"))


class CardsListFixtureStoreTests(CardsListTestCase):
    def test_a5_unrecognized_verdict_is_flagged_and_admits_as_unmeasured(self):
        card = new_card("odd@test", "example/Odd", "no_go")
        store = self.write_store([card])
        code, stdout, stderr = self.run_list("--quality-cards", str(store))
        self.assertEqual(code, 0, stderr)
        header, rows = self.split_output(stdout)
        self.assertIn("source=explicit", header)
        self.assertEqual(len(rows), 1)
        self.assertIn("UNRECOGNIZED 'no_go' (admits as UNMEASURED)", rows[0])
        self.assertIn("default-eligible", rows[0])
        self.assertNotIn("opt-in", rows[0])

    def test_a5_hostile_verdict_is_bounded_to_a_single_line(self):
        card = new_card("hostile@test", "example/Hostile", "x\nFORGED row\r" + "A" * 500)
        store = self.write_store([card])
        code, stdout, stderr = self.run_list("--quality-cards", str(store))
        self.assertEqual(code, 0, stderr)
        _, rows = self.split_output(stdout)
        self.assertEqual(len(rows), 1, stdout)
        self.assertNotIn("\r", stdout)
        self.assertLess(len(rows[0]), 400)
        self.assertIn("UNRECOGNIZED", rows[0])

    def test_a5_undecodable_card_prints_its_reason_and_exits_3_after_the_listing(self):
        good = new_card("good@test", "example/Good", "PASS")
        junk = new_card("junk@test", "example/Junk", "PASS")
        junk["config"] = "junk"
        store = self.write_store([good, junk])
        code, stdout, stderr = self.run_list("--quality-cards", str(store))
        self.assertEqual(code, 3)
        _, rows = self.split_output(stdout)
        self.assertEqual(len(rows), 2)  # both listed first
        self.assertNotIn("UNDECODABLE", self.row_for(rows, "good@test"))
        self.assertIn("UNDECODABLE config is not an object", self.row_for(rows, "junk@test"))
        self.assertIn("cannot be decoded by the built-in engine", stderr)

    def test_a5_non_object_element_is_undecodable_not_a_crash(self):
        store = self.write_store([new_card("good@test", "example/Good", "PASS"), 7])
        code, stdout, stderr = self.run_list("--quality-cards", str(store))
        self.assertEqual(code, 3, stderr)
        _, rows = self.split_output(stdout)
        self.assertEqual(len(rows), 2)
        self.assertIn("UNDECODABLE element is not a JSON object", rows[1])

    def test_a5_model_filter_is_exact_on_repos_that_share_a_prefix(self):
        store = self.write_store(
            [
                new_card("x@test", "example/X", "PASS"),
                new_card("x-8bit@test", "example/X-8bit", "NO_GO"),
            ]
        )
        common = ["--quality-cards", str(store)]
        code, stdout, stderr = self.run_list(*common, "--model", "example/X")
        self.assertEqual(code, 0, stderr)
        _, rows = self.split_output(stdout)
        self.assertEqual([row.split(" ")[0] for row in rows], ["x@test"])
        code, stdout, stderr = self.run_list(*common, "--model", "example/X-8bit")
        self.assertEqual(code, 0, stderr)
        _, rows = self.split_output(stdout)
        self.assertEqual([row.split(" ")[0] for row in rows], ["x-8bit@test"])
        self.assertIn("opt-in: --accept-quality x-8bit@test", rows[0])
        for partial in ("example/X-", "example/X-8", "X"):
            with self.subTest(partial=partial):
                code, _, stderr = self.run_list(*common, "--model", partial)
                self.assertEqual(code, 1)
                self.assertIn("serve would admit it UNMEASURED", stderr)


class CardsListRevisionAndStoreWideTests(CardsListTestCase):
    """Review defects D1-D3: pin-only cards, store-wide undecodable, wording."""

    PIN = "abcdef12"
    REVISION = "abcdef12" + "0123456789abcdef0123456789abcdef"[:32]

    def pin_only_card(self, card_id="pinonly@test"):
        card = new_card(card_id, "example/PinOnly", "PASS")
        card["model"]["repo"] = None
        card["model"]["hfPin"] = self.PIN
        return card

    def test_t1_revision_selects_a_pin_only_card(self):
        self.assertEqual(len(self.REVISION), 40)
        store = self.write_store(
            [new_card("other@test", "example/Other", "PASS"), self.pin_only_card()]
        )
        code, stdout, stderr = self.run_list(
            "--quality-cards", str(store), "--revision", self.REVISION
        )
        self.assertEqual(code, 0, stderr)
        _, rows = self.split_output(stdout)
        self.assertEqual([row.split(" ")[0] for row in rows], ["pinonly@test"])

    def test_t1_revision_and_model_select_the_union_in_store_order_without_duplicates(self):
        both = new_card("both@test", "example/Both", "PASS")
        both["model"]["hfPin"] = self.PIN
        store = self.write_store(
            [self.pin_only_card(), new_card("other@test", "example/Other", "PASS"), both]
        )
        code, stdout, stderr = self.run_list(
            "--quality-cards", str(store), "--revision", self.REVISION,
            "--model", "example/Both",
        )
        self.assertEqual(code, 0, stderr)
        _, rows = self.split_output(stdout)
        self.assertEqual([row.split(" ")[0] for row in rows], ["pinonly@test", "both@test"])

    def test_revision_alone_matching_nothing_points_at_model_not_unmeasured(self):
        # serve also matches by repo, so a pin miss alone cannot prove UNMEASURED.
        store = self.write_store([new_card("other@test", "example/Other", "PASS")])
        code, _, stderr = self.run_list(
            "--quality-cards", str(store), "--revision", "f" * 40
        )
        self.assertEqual(code, 1)
        self.assertNotIn("UNMEASURED", stderr)
        self.assertIn("--model", stderr)
        # Both axes given and both miss: now UNMEASURED is provable.
        code, _, stderr = self.run_list(
            "--quality-cards", str(store), "--revision", "f" * 40,
            "--model", "nobody/x",
        )
        self.assertEqual(code, 1)
        self.assertIn("serve would admit it UNMEASURED", stderr)

    def test_t1_model_matching_nothing_by_repo_points_at_revision_not_unmeasured(self):
        store = self.write_store(
            [new_card("other@test", "example/Other", "PASS"), self.pin_only_card()]
        )
        code, stdout, stderr = self.run_list(
            "--quality-cards", str(store), "--model", "some/Repo"
        )
        self.assertEqual(code, 1)
        self.assertNotIn("UNMEASURED", stderr)
        self.assertIn("--revision", stderr)
        self.assertIn("some/Repo", stderr)
        self.assertIn("1 card(s)", stderr)
        self.assertEqual(self.split_output(stdout)[1], [])

    def test_t1_revision_matching_nothing_keeps_the_unmeasured_message(self):
        store = self.write_store([self.pin_only_card()])
        code, _, stderr = self.run_list(
            "--quality-cards", str(store), "--revision", "f" * 40
        )
        self.assertEqual(code, 1)
        self.assertIn("serve would admit it UNMEASURED", stderr)
        self.assertIn("f" * 40, stderr)

    def test_t2_bundled_store_model_with_a_card_still_exits_0_with_one_row(self):
        code, stdout, stderr = self.run_list("--model", QWEN3_8B_8BIT_REPO)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(len(self.split_output(stdout)[1]), 1)

    def test_t2_bundled_store_unknown_model_points_at_revision(self):
        code, stdout, stderr = self.run_list("--model", "nobody/x")
        self.assertEqual(code, 1)
        self.assertIn("--revision", stderr)
        self.assertNotIn("UNMEASURED", stderr)
        self.assertIn("no quality card names nobody/x", stderr)
        self.assertEqual(self.split_output(stdout)[1], [])

    def test_t2_bundled_flash_next_card_is_found_by_its_full_revision(self):
        code, stdout, stderr = self.run_list(
            "--revision", "ef5b919d31534faa1997666f1a22d362cd6383cd"
        )
        self.assertEqual(code, 0, stderr)
        ids = [row.split(" ")[0] for row in self.split_output(stdout)[1]]
        self.assertIn("qwen38-flash-next-mixed-4-8bit@m3ultra", ids)

    def test_t3_store_without_pin_only_cards_keeps_the_unmeasured_message(self):
        store = self.write_store([new_card("other@test", "example/Other", "PASS")])
        code, _, stderr = self.run_list(
            "--quality-cards", str(store), "--model", "nobody/x"
        )
        self.assertEqual(code, 1)
        self.assertIn("serve would admit it UNMEASURED", stderr)
        self.assertNotIn("--revision", stderr)

    def test_t3_a_repo_less_card_with_no_pin_is_not_pin_only(self):
        card = new_card("nopin@test", "example/NoPin", "PASS")
        card["model"]["repo"] = None
        del card["model"]["hfPin"]
        store = self.write_store([card])
        code, _, stderr = self.run_list(
            "--quality-cards", str(store), "--model", "nobody/x"
        )
        self.assertEqual(code, 1)
        self.assertIn("serve would admit it UNMEASURED", stderr)

    def test_t4_malformed_revision_is_a_usage_error_before_any_row(self):
        for bad in ("nothex", "abcdef12", "g" * 40, "a" * 41):
            with self.subTest(revision=bad):
                code, stdout, stderr = self.run_list("--revision", bad)
                self.assertEqual(code, 2)
                self.assertEqual(stdout, "")
                self.assertIn("fastmlx cards list:", stderr)

    def test_t4_uppercase_hex_revision_is_accepted(self):
        store = self.write_store([self.pin_only_card()])
        code, stdout, stderr = self.run_list(
            "--quality-cards", str(store), "--revision", self.REVISION.upper()
        )
        self.assertEqual(code, 0, stderr)
        self.assertEqual(len(self.split_output(stdout)[1]), 1)

    def test_t5_filter_does_not_hide_a_store_wide_undecodable_card(self):
        good = new_card("good@test", "good/Repo", "PASS")
        junk = new_card("junk@test", "example/Junk", "PASS")
        junk["config"] = "junk"
        store = self.write_store([good, junk])
        code, stdout, stderr = self.run_list(
            "--quality-cards", str(store), "--model", "good/Repo"
        )
        self.assertEqual(code, 3)
        _, rows = self.split_output(stdout)
        self.assertEqual([row.split(" ")[0] for row in rows], ["good@test"])
        self.assertNotIn("UNDECODABLE", rows[0])
        self.assertIn("junk@test", stderr)
        self.assertIn("config is not an object", stderr)
        code, _, stderr = self.run_list("--quality-cards", str(store))
        self.assertEqual(code, 3)

    def test_t5_undecodable_takes_precedence_over_no_match(self):
        good = new_card("good@test", "good/Repo", "PASS")
        junk = new_card("junk@test", "example/Junk", "PASS")
        junk["config"] = "junk"
        store = self.write_store([good, junk])
        code, stdout, stderr = self.run_list(
            "--quality-cards", str(store), "--model", "nobody/x"
        )
        self.assertEqual(code, 3)
        self.assertEqual(self.split_output(stdout)[1], [])
        self.assertIn("junk@test", stderr)

    def test_t5_json_mode_stdout_stays_one_document_when_undecodable_hidden(self):
        good = new_card("good@test", "good/Repo", "PASS")
        junk = new_card("junk@test", "example/Junk", "PASS")
        junk["config"] = "junk"
        store = self.write_store([good, junk])
        code, stdout, _ = self.run_list(
            "--json", "--quality-cards", str(store), "--model", "good/Repo"
        )
        self.assertEqual(code, 3)
        self.assertEqual([c["id"] for c in json.loads(stdout)["cards"]], ["good@test"])

    def test_t6_undecodable_message_scopes_the_refusal_to_the_built_in_engine(self):
        junk = new_card("junk@test", "example/Junk", "PASS")
        junk["config"] = "junk"
        store = self.write_store([junk])
        code, _, stderr = self.run_list("--quality-cards", str(store))
        self.assertEqual(code, 3)
        self.assertIn("built-in engine", stderr)
        self.assertIn("without a custom engine profile", stderr)
        self.assertNotIn("`fastmlx serve` would refuse every launch", stderr)


class CardsListPinAndStoreTests(CardsListTestCase):
    def test_a6_wrong_pin_exits_3_before_any_row(self):
        code, stdout, stderr = self.run_list("--quality-cards-sha256", "0" * 64)
        self.assertEqual(code, 3)
        self.assertEqual(stdout, "")
        self.assertIn("does not match --quality-cards-sha256", stderr)

    def test_a6_malformed_pin_exits_3(self):
        code, stdout, stderr = self.run_list("--quality-cards-sha256", "xyz")
        self.assertEqual(code, 3)
        self.assertEqual(stdout, "")
        self.assertIn("must be exactly 64 hexadecimal characters", stderr)

    def test_a6_matching_pin_lists_the_store(self):
        digest = hashlib.sha256(GUIDES_PATH.read_bytes()).hexdigest()
        code, stdout, stderr = self.run_list("--quality-cards-sha256", digest.upper())
        self.assertEqual(code, 0, stderr)
        self.assertEqual(len(self.split_output(stdout)[1]), 15)

    def test_a6_pin_refusal_text_is_the_launcher_text_recommend_prints(self):
        launch = CARDS.launch
        raw, cards, _ = launch._inspect_quality_card_store(GUIDES_PATH)
        with self.assertRaises(launch.LaunchRefusal) as expected:
            launch.enforce_quality_cards_pin("0" * 64, GUIDES_PATH, raw, cards)
        code, _, stderr = self.run_list("--quality-cards-sha256", "0" * 64)
        self.assertEqual(code, expected.exception.exit_code)
        self.assertEqual(stderr.strip(), "fastmlx cards list: " + expected.exception.message)

    def test_a6_unreadable_or_non_manifest_explicit_store_exits_3(self):
        missing = self.root / "missing.json"
        code, stdout, stderr = self.run_list("--quality-cards", str(missing))
        self.assertEqual(code, 3)
        self.assertEqual(stdout, "")
        self.assertIn("missing or is not a quality-card manifest", stderr)
        junk = self.root / "junk.json"
        junk.write_text("[1, 2]", encoding="utf-8")
        code, stdout, stderr = self.run_list("--quality-cards", str(junk))
        self.assertEqual(code, 3)
        self.assertEqual(stdout, "")

    def test_a6_pulled_store_is_read_like_recommend_reads_it(self):
        body = serialize(honest_document())
        digest = hashlib.sha256(body).hexdigest()
        pulled_dir = self.home / ".fastmlx" / "cards"
        pulled_dir.mkdir(parents=True)
        (pulled_dir / f"{digest}.json").write_bytes(body)
        code, stdout, stderr = self.run_list()
        self.assertEqual(code, 0, stderr)
        header, rows = self.split_output(stdout)
        self.assertIn("source=pulled", header)
        self.assertIn(f"sha256={digest}", header)
        self.assertEqual(len(rows), 16)
        self.assertIn(NEW_CARD_ID + " ", rows[-1])

    def test_a6_pulled_store_that_fails_its_integrity_check_refuses_with_exit_3(self):
        body = serialize(honest_document())
        pulled_dir = self.home / ".fastmlx" / "cards"
        pulled_dir.mkdir(parents=True)
        (pulled_dir / ("f" * 64 + ".json")).write_bytes(body)  # name != digest
        code, stdout, stderr = self.run_list()
        self.assertEqual(code, 3)
        self.assertEqual(stdout, "")
        self.assertIn("does not match its file name", stderr)

    def test_a6_list_never_calls_a_network_function(self):
        # run_list patches fetch/https_opener/urlopen to raise AssertionError;
        # a clean exit 0 proves none was reached.
        code, _, stderr = self.run_list()
        self.assertEqual(code, 0, stderr)
        text = Path(CARDS.__file__).read_text(encoding="utf-8")
        body = text[text.index("def _run_list") :]
        body = body[: body.index("\ndef ", 1)] if "\ndef " in body[1:] else body
        for forbidden in ("fetch(", "https_opener", "urlopen"):
            self.assertNotIn(forbidden, body)


class CardsListDispatchAndJsonTests(CardsListTestCase):
    def test_a7_cards_with_no_subcommand_still_exits_2(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                CARDS.main([])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("list", stderr.getvalue())
        self.assertIn("pull", stderr.getvalue())

    def test_a7_list_rejects_unknown_flags_with_usage_exit_2(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                CARDS.main(["list", "--nope"])
        self.assertEqual(ctx.exception.code, 2)

    def test_a8_json_parses_and_ids_match_the_text_rows(self):
        _, text_out, _ = self.run_list(host_class="apple-m5")
        _, rows = self.split_output(text_out)
        code, stdout, stderr = self.run_list("--json", host_class="apple-m5")
        self.assertEqual(code, 0, stderr)
        document = json.loads(stdout)
        self.assertEqual(
            [card["id"] for card in document["cards"]], [row.split(" ")[0] for row in rows]
        )
        guide_sha = hashlib.sha256(GUIDES_PATH.read_bytes()).hexdigest()
        self.assertEqual(document["cardStore"]["sha256"], guide_sha)
        self.assertEqual(document["cardStore"]["cards"], 15)
        by_id = {card["id"]: card for card in document["cards"]}
        pass_card = by_id["qwen3-8b-8bit@m3ultra"]
        self.assertEqual(pass_card["verdict"], "PASS")
        self.assertEqual(pass_card["admission"], {"default": True, "optIn": False})
        self.assertEqual(pass_card["repo"], QWEN3_8B_8BIT_REPO)
        self.assertEqual(pass_card["hfPin"], "48a0b75b")
        self.assertIsNone(pass_card["residency"])
        self.assertEqual(pass_card["hardwareClass"], "apple-m3-ultra")
        self.assertEqual(pass_card["host"], "other hardware")
        no_go = by_id["qwen3-0p6b-4bit@m5"]
        self.assertEqual(no_go["admission"], {"default": False, "optIn": True})
        self.assertEqual(no_go["residency"], "resident")
        self.assertEqual(no_go["host"], "this host")
        self.assertIsNone(by_id["qwen38-flash-next-mixed-4-8bit@m3ultra"]["repo"])

    def test_a8_json_model_filter_and_no_match(self):
        code, stdout, _ = self.run_list("--json", "--model", QWEN3_8B_8BIT_REPO)
        self.assertEqual(code, 0)
        self.assertEqual([c["id"] for c in json.loads(stdout)["cards"]], ["qwen3-8b-8bit@m3ultra"])
        code, stdout, stderr = self.run_list("--json", "--model", "nobody/x")
        self.assertEqual(code, 1)
        self.assertIn("no quality card names nobody/x by repo", stderr)
        self.assertNotIn("UNMEASURED", stderr)
        self.assertEqual(json.loads(stdout)["cards"], [])
        # stdout stays one parseable JSON document even when nothing matches.
        self.assertEqual(json.loads(stdout)["cards"], [])

    def test_a8_json_undecodable_card_exits_3_after_printing(self):
        junk = new_card("junk@test", "example/Junk", "PASS")
        junk["config"] = "junk"
        store = self.write_store([junk])
        code, stdout, _ = self.run_list("--json", "--quality-cards", str(store))
        self.assertEqual(code, 3)
        row = json.loads(stdout)["cards"][0]
        self.assertEqual(row["undecodable"], "config is not an object")

    def test_a8_dispatcher_usage_mentions_cards_list(self):
        spec = importlib.util.spec_from_file_location(
            "fastmlx_dispatch", Path(__file__).resolve().parents[1] / "fastmlx.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertIn("cards list", module.USAGE)
        self.assertIn("cards pull", module.USAGE)


if __name__ == "__main__":
    unittest.main()
