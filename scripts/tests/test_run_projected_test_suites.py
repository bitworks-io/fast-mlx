"""Tests for the parser and refusal behaviour behind `run_projected_test_suites.py`.

Every verdict that gate produces flows through `parse_summary`, `class_selectors`,
`evaluate_target`'s OK/MISMATCH/REFUSED classification, and `run_gate`'s once-per-run package
build. This module runs on CPU only, never shells out to `swift`/`xcrun`, and never touches the
network -- it drives those functions with recorded XCTest/`swift test list` output shapes and
stubbed subprocess results, which is why it lives in the fast `scripts/tests` suite rather than
beside the gate itself.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import run_projected_test_suites as suites  # noqa: E402


def summary_block(name: str, tests: int, failures: int, skipped: int | None = None) -> str:
    """Build one `Test Suite '<name>'` XCTest summary block in the exact shape `parse_summary`
    anchors on: a `Test Suite '<name>' passed|failed at ...` line immediately followed by the
    `Executed N tests, with ... failures ...` counts line. `<name>` is passed through verbatim so
    this can build either a bundle-level `'<Target>.xctest'` block or a `'Selected tests'` block."""
    if skipped is not None:
        counts = (
            f"Executed {tests} tests, with {skipped} tests skipped and {failures} failures "
            "(0 unexpected) in 0.100 (0.105) seconds"
        )
    else:
        counts = f"Executed {tests} tests, with {failures} failures (0 unexpected) in 0.100 (0.105) seconds"
    return (
        f"Test Suite '{name}' passed at 2026-09-17 10:00:00.000.\n"
        f"\t {counts}\n"
    )


def bundle_summary(target: str, tests: int, failures: int, skipped: int | None = None) -> str:
    """A per-target-bundle summary block, anchored on `'<target>.xctest'`."""
    return summary_block(f"{target}.xctest", tests, failures, skipped)


def selected_tests_summary(tests: int, failures: int, skipped: int | None = None) -> str:
    """A combined-bundle-mode summary block, anchored on `'Selected tests'`, exactly as
    `xcrun xctest -XCTest <selectors> <bundle>` prints last."""
    return summary_block("Selected tests", tests, failures, skipped)


class ParseSummaryTests(unittest.TestCase):
    """Direct tests of the pure parser, independent of the build/run plumbing around it."""

    def test_parses_summary_without_skipped_clause(self) -> None:
        """Older/plain XCTest output omits the skipped clause entirely; a regex that assumes it
        is always present would fail to match this shape at all and silently REFUSE every run on
        a toolchain that doesn't print it."""
        output = bundle_summary("HarnessCoreTests", tests=5, failures=0)
        self.assertEqual(suites.parse_summary(output, "HarnessCoreTests"), (5, 0, 0))

    def test_parses_summary_with_skipped_clause(self) -> None:
        """The common case on this project's toolchain: skipped count must land in the middle
        slot, not get merged into tests or failures."""
        output = bundle_summary("HarnessCoreTests", tests=12, failures=1, skipped=2)
        self.assertEqual(suites.parse_summary(output, "HarnessCoreTests"), (12, 2, 1))

    def test_multi_bundle_output_does_not_read_a_neighbouring_bundles_summary(self) -> None:
        """Reproduces the exact hazard the module docstring names: a run that prints more than
        one bundle's summary (e.g. from `swift test --filter`) puts an unrelated bundle's
        `Executed 0 tests` line right next to the requested target's real one. A parser that
        isn't anchored precisely to `'<target>.xctest'` could read the wrong line and report a
        confident, wrong count instead of the real one."""
        blob = bundle_summary("UnrelatedTests", tests=0, failures=0) + bundle_summary(
            "HarnessCoreTests", tests=42, failures=3
        )
        self.assertEqual(suites.parse_summary(blob, "HarnessCoreTests"), (42, 0, 3))

    def test_multi_bundle_output_resolves_each_of_two_real_targets_correctly(self) -> None:
        """Same hazard, but with two targets that both have real (nonzero) counts, so a swapped
        read would still look plausible instead of obviously wrong -- pin which summary belongs
        to which target in both directions."""
        blob = bundle_summary("AlphaTests", tests=40, failures=3) + bundle_summary(
            "BetaTests", tests=7, failures=0
        )
        self.assertEqual(suites.parse_summary(blob, "AlphaTests"), (40, 0, 3))
        self.assertEqual(suites.parse_summary(blob, "BetaTests"), (7, 0, 0))

    def test_crash_truncated_output_yields_none(self) -> None:
        """A bundle that crashes mid-run before ever printing its summary line must be reported
        as unreadable (None), never coerced into a passing 0/0/0 result."""
        output = (
            "Test Suite 'HarnessCoreTests.xctest' started at 2026-09-17 10:00:00.000.\n"
            "Test Case '-[HarnessCoreTests.SomeTest testFoo]' started.\n"
        )
        self.assertIsNone(suites.parse_summary(output, "HarnessCoreTests"))

    def test_missing_marker_entirely_yields_none(self) -> None:
        """No `Test Suite '<target>.xctest'` line at all (e.g. a toolchain-launch failure) must
        refuse, not fall through to scanning the output for any stray `Executed` line."""
        output = "dyld: Library not loaded\nAbort trap: 6\n"
        self.assertIsNone(suites.parse_summary(output, "HarnessCoreTests"))

    def test_marker_present_but_summary_beyond_lookahead_window_yields_none(self) -> None:
        """The lookahead tolerates exactly one blank line between the marker and its counts line;
        anything further away must be refused rather than scanned for indefinitely."""
        output = (
            "Test Suite 'HarnessCoreTests.xctest' passed at 2026-09-17 10:00:00.000.\n"
            "\n\n"
            "Executed 5 tests, with 0 failures (0 unexpected) in 0.100 (0.105) seconds\n"
        )
        self.assertIsNone(suites.parse_summary(output, "HarnessCoreTests"))

    def test_zero_executed_tests_still_parses_to_a_real_tuple(self) -> None:
        """`xcrun xctest` exits 0 on a bundle that finds nothing to run, printing a well-formed
        `Executed 0 tests` summary. The parser must not special-case this away -- it is
        `evaluate_target`'s job (tested below) to treat it as a mismatch, not the parser's job to
        hide it."""
        output = bundle_summary("HarnessCoreTests", tests=0, failures=0)
        self.assertEqual(suites.parse_summary(output, "HarnessCoreTests"), (0, 0, 0))

    def test_selected_tests_anchor_reads_its_own_block_not_a_preceding_bundle_block(self) -> None:
        """Combined-bundle mode: `xcrun xctest -XCTest <selectors> <bundle>` prints the whole
        combined bundle's `'<bundle>.xctest'` summary BEFORE the `'Selected tests'` summary that
        covers only the requested classes. Those two blocks are constructed here to differ on
        purpose -- a parser that fell through to the first `Executed` line it saw, instead of
        anchoring on `anchor="Selected tests"` specifically, would silently read the whole
        bundle's (wrong, larger) counts instead of the selected classes' own counts."""
        blob = bundle_summary(
            "fast-mlx-spikePackageTests", tests=2565, failures=75
        ) + selected_tests_summary(tests=12, failures=1, skipped=2)
        self.assertEqual(
            suites.parse_summary(blob, "HarnessCoreTests", anchor="Selected tests"), (12, 2, 1)
        )

    def test_selected_tests_anchor_absent_yields_none(self) -> None:
        """If the combined-bundle run never reaches a `'Selected tests'` block (crash before it),
        anchoring on it must refuse rather than silently fall back to the whole-bundle block."""
        blob = bundle_summary("fast-mlx-spikePackageTests", tests=2565, failures=75)
        self.assertIsNone(suites.parse_summary(blob, "HarnessCoreTests", anchor="Selected tests"))


class ClassSelectorsTests(unittest.TestCase):
    """Direct tests of the pure `swift test list` -> selector-list derivation."""

    LISTING = (
        "[0/1] Planning build\n"
        "HarnessCoreTests.ForcedScoringPlanTests/everyPositionAppearsExactlyOnce\n"
        "HarnessCoreTests.ForcedScoringPlanTests/anotherCase()\n"
        "HarnessCoreTests.TailStatisticTests/basicCase\n"
        "ServingCoreTests.RequestRouterTests/routesToBackend\n"
        "ServingCoreTestsExtra.SomethingTests/unrelatedCase\n"
        "not a test listing line\n"
        "\n"
    )

    def test_exact_prefix_match_returns_sorted_deduped_classes(self) -> None:
        self.assertEqual(
            suites.class_selectors(self.LISTING, "HarnessCoreTests"),
            ["HarnessCoreTests.ForcedScoringPlanTests", "HarnessCoreTests.TailStatisticTests"],
        )

    def test_does_not_match_a_sibling_module_that_shares_a_string_prefix(self) -> None:
        """`ServingCoreTests.` must not match `ServingCoreTestsExtra.x/y` -- a naive
        `startswith(target)` without the trailing dot would incorrectly pull that sibling
        module's classes into ServingCoreTests' selector list."""
        selectors = suites.class_selectors(self.LISTING, "ServingCoreTests")
        self.assertEqual(selectors, ["ServingCoreTests.RequestRouterTests"])
        self.assertNotIn("ServingCoreTestsExtra.SomethingTests", selectors)

    def test_ignores_lines_without_a_slash(self) -> None:
        """Lines like `[0/1] Planning build` (stderr/status noise that can end up interleaved
        with stdout) must never be mistaken for a test identifier."""
        selectors = suites.class_selectors("HarnessCoreTests\nnot a test line\n", "HarnessCoreTests")
        self.assertEqual(selectors, [])

    def test_unknown_target_returns_empty_list(self) -> None:
        self.assertEqual(suites.class_selectors(self.LISTING, "NoSuchTarget"), [])

    def test_dedupes_repeated_classes(self) -> None:
        listing = (
            "HarnessCoreTests.ForcedScoringPlanTests/methodOne\n"
            "HarnessCoreTests.ForcedScoringPlanTests/methodTwo\n"
        )
        self.assertEqual(
            suites.class_selectors(listing, "HarnessCoreTests"),
            ["HarnessCoreTests.ForcedScoringPlanTests"],
        )

    def test_strips_trailing_parens_and_whitespace_robustly(self) -> None:
        listing = "  HarnessCoreTests.SpacedTests  /  method()  \n"
        self.assertEqual(
            suites.class_selectors(listing, "HarnessCoreTests"),
            ["HarnessCoreTests.SpacedTests"],
        )


def _stub_run(
    list_returncode: int = 0,
    list_stdout: str = "",
    xctest_returncode: int = 0,
    xctest_stdout: str = "",
    xctest_stderr: str = "",
):
    """A `run_projected_test_suites.run` stand-in for `evaluate_target`-level tests, which never
    call the package build (that happens once in `run_gate`, before `evaluate_target` runs).
    Answers `swift test list ...` and `xcrun xctest ...` without touching a real toolchain."""

    def _fake(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["swift", "test", "list"]:
            return subprocess.CompletedProcess(
                args=command, returncode=list_returncode, stdout=list_stdout, stderr=""
            )
        if command[0] == "xcrun":
            return subprocess.CompletedProcess(
                args=command, returncode=xctest_returncode, stdout=xctest_stdout, stderr=xctest_stderr
            )
        raise AssertionError(f"unexpected command in evaluate_target stub: {command}")

    return _fake


class EvaluateTargetTests(unittest.TestCase):
    """Tests the OK/MISMATCH/REFUSED classification in `evaluate_target`, with `run` and bundle
    resolution stubbed so no real `swift`/`xcrun` toolchain is required. Assumes the package build
    already succeeded (that step lives in `run_gate`, tested separately below)."""

    COMBINED_BUNDLE = Path(
        "/fake/dest/spike/.build/arm64-apple-macosx/debug/fast-mlx-spikePackageTests.xctest"
    )
    PER_TARGET_BUNDLE = Path(
        "/fake/dest/spike/.build/out/Products/Debug/HarnessCoreTests.xctest"
    )
    LISTING = "HarnessCoreTests.HarnessCoreTests/testSomething\n"

    def _evaluate(
        self,
        expected_tests: int = 5,
        expected_failures: int = 0,
        per_target_bundle_found: bool = False,
        per_target_executable_found: bool = True,
        combined_bundle: Path | None = None,
        combined_candidates: list[Path] | None = None,
        list_returncode: int = 0,
        list_stdout: str | None = None,
        xctest_stdout: str = "",
        xctest_stderr: str = "",
        listing_cache: dict | None = None,
    ) -> suites.TargetResult:
        expected = {"tests": expected_tests, "skipped": 0, "failures": expected_failures}
        if combined_candidates is None:
            combined_candidates = [self.COMBINED_BUNDLE] if combined_bundle else []
        if list_stdout is None:
            list_stdout = self.LISTING
        with mock.patch.object(
            suites,
            "run",
            side_effect=_stub_run(
                list_returncode=list_returncode,
                list_stdout=list_stdout,
                xctest_stdout=xctest_stdout,
                xctest_stderr=xctest_stderr,
            ),
        ), mock.patch.object(
            suites,
            "resolve_bundle",
            return_value=(self.PER_TARGET_BUNDLE if per_target_bundle_found else None),
        ), mock.patch.object(
            suites, "bundle_has_executable", return_value=per_target_executable_found
        ), mock.patch.object(
            suites,
            "resolve_combined_bundle",
            return_value=(combined_bundle, combined_candidates),
        ), mock.patch.object(
            suites, "swift_toolchain_version", return_value="Apple Swift version 6.3.3 (fake)"
        ):
            return suites.evaluate_target(
                Path("/fake/dest"),
                "HarnessCoreTests",
                expected,
                log_dir=None,
                listing_cache=(listing_cache if listing_cache is not None else {}),
            )

    # -- Mode 1: per-target bundle preferred over combined -----------------------------------

    def test_per_target_bundle_with_executable_is_preferred_and_xctest_run_directly(self) -> None:
        calls: list[list[str]] = []

        def _fake(command, cwd):
            calls.append(command)
            if command[0] == "xcrun":
                return subprocess.CompletedProcess(
                    command, 0, bundle_summary("HarnessCoreTests", tests=5, failures=0), ""
                )
            raise AssertionError(f"unexpected command: {command}")

        with mock.patch.object(suites, "run", side_effect=_fake), mock.patch.object(
            suites, "resolve_bundle", return_value=self.PER_TARGET_BUNDLE
        ), mock.patch.object(suites, "bundle_has_executable", return_value=True):
            result = suites.evaluate_target(
                Path("/fake/dest"),
                "HarnessCoreTests",
                {"tests": 5, "skipped": 0, "failures": 0},
                log_dir=None,
                listing_cache={},
            )
        self.assertEqual(result.verdict, suites.OK)
        # No -XCTest selector run, and no `swift test list` call, when the per-target bundle
        # already answers the question directly.
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], ["xcrun", "xctest", str(self.PER_TARGET_BUNDLE)])

    # -- Mode 2: combined bundle fallback ------------------------------------------------------

    def test_combined_bundle_fallback_invokes_xctest_with_selectors(self) -> None:
        calls: list[list[str]] = []

        def _fake(command, cwd):
            calls.append(command)
            if command[:3] == ["swift", "test", "list"]:
                return subprocess.CompletedProcess(command, 0, self.LISTING, "")
            if command[0] == "xcrun":
                return subprocess.CompletedProcess(
                    command, 0, selected_tests_summary(tests=5, failures=0), ""
                )
            raise AssertionError(f"unexpected command: {command}")

        with mock.patch.object(suites, "run", side_effect=_fake), mock.patch.object(
            suites, "resolve_bundle", return_value=None
        ), mock.patch.object(
            suites, "resolve_combined_bundle", return_value=(self.COMBINED_BUNDLE, [self.COMBINED_BUNDLE])
        ):
            result = suites.evaluate_target(
                Path("/fake/dest"),
                "HarnessCoreTests",
                {"tests": 5, "skipped": 0, "failures": 0},
                log_dir=None,
                listing_cache={},
            )
        self.assertEqual(result.verdict, suites.OK)
        xctest_calls = [c for c in calls if c[0] == "xcrun"]
        self.assertEqual(len(xctest_calls), 1)
        self.assertIn("-XCTest", xctest_calls[0])
        self.assertIn("HarnessCoreTests.HarnessCoreTests", xctest_calls[0])
        self.assertEqual(xctest_calls[0][-1], str(self.COMBINED_BUNDLE))

    def test_skeleton_per_target_bundle_falls_through_to_combined_mode(self) -> None:
        """A per-target bundle directory exists but has no executable: this must NOT refuse
        immediately -- it must fall back to combined mode and succeed if that resolves."""
        result = self._evaluate(
            per_target_bundle_found=True,
            per_target_executable_found=False,
            combined_bundle=self.COMBINED_BUNDLE,
            xctest_stdout=selected_tests_summary(tests=5, failures=0),
        )
        self.assertEqual(result.verdict, suites.OK)

    def test_combined_mode_count_mismatch_is_mismatch(self) -> None:
        result = self._evaluate(
            expected_tests=5,
            expected_failures=0,
            combined_bundle=self.COMBINED_BUNDLE,
            xctest_stdout=selected_tests_summary(tests=4, failures=1),
        )
        self.assertEqual(result.verdict, suites.MISMATCH)

    def test_combined_mode_exact_match_is_ok(self) -> None:
        result = self._evaluate(
            expected_tests=5,
            expected_failures=0,
            combined_bundle=self.COMBINED_BUNDLE,
            xctest_stdout=selected_tests_summary(tests=5, failures=0),
        )
        self.assertEqual(result.verdict, suites.OK)

    # -- Refusal shapes ------------------------------------------------------------------------

    def test_both_per_target_and_combined_bundle_present_is_refused_and_neither_tool_runs(
        self,
    ) -> None:
        """A build tree that holds BOTH a per-target bundle with an executable AND a combined
        bundle with an executable is evidence of a mixed-toolchain build tree: Swift 6.4 was
        measured to leave only per-target bundles and Swift 6.3.3 was measured to leave only the
        combined one, in different directories, so a tree built by one and then the other can
        hold a stale bundle of the first kind next to a fresh one of the second. This gate must
        refuse rather than silently prefer mode 1, and must not invoke `xcrun xctest` (either
        mode) or `swift test list` while doing so."""
        calls: list[list[str]] = []

        def _fake(command, cwd):
            calls.append(command)
            raise AssertionError(
                f"neither xctest nor swift test list may run when both bundle kinds exist: "
                f"{command}"
            )

        with mock.patch.object(suites, "run", side_effect=_fake), mock.patch.object(
            suites, "resolve_bundle", return_value=self.PER_TARGET_BUNDLE
        ), mock.patch.object(
            suites, "bundle_has_executable", return_value=True
        ), mock.patch.object(
            suites,
            "resolve_combined_bundle",
            return_value=(self.COMBINED_BUNDLE, [self.COMBINED_BUNDLE]),
        ), mock.patch.object(
            suites, "swift_toolchain_version", return_value="Apple Swift version 6.3.3 (fake)"
        ):
            result = suites.evaluate_target(
                Path("/fake/dest"),
                "HarnessCoreTests",
                {"tests": 5, "skipped": 0, "failures": 0},
                log_dir=None,
                listing_cache={},
            )
        self.assertEqual(result.verdict, suites.REFUSED)
        self.assertIn(str(self.PER_TARGET_BUNDLE), result.detail)
        self.assertIn(str(self.COMBINED_BUNDLE), result.detail)
        self.assertIn("Apple Swift version 6.3.3 (fake)", result.detail)
        self.assertEqual(calls, [])

    def test_neither_per_target_nor_combined_bundle_is_refused(self) -> None:
        result = self._evaluate(per_target_bundle_found=False, combined_bundle=None)
        self.assertEqual(result.verdict, suites.REFUSED)
        self.assertIn("no HarnessCoreTests.xctest bundle exists", result.detail)
        self.assertIn("no combined", result.detail)

    def test_skeleton_per_target_with_no_combined_available_mentions_both_in_detail(self) -> None:
        result = self._evaluate(
            per_target_bundle_found=True, per_target_executable_found=False, combined_bundle=None
        )
        self.assertEqual(result.verdict, suites.REFUSED)
        self.assertIn("no executable", result.detail)
        self.assertIn("Contents/MacOS/HarnessCoreTests", result.detail)
        self.assertIn("no combined", result.detail)

    def test_more_than_one_combined_candidate_with_executable_is_refused_distinctly(self) -> None:
        other = Path("/fake/dest/spike/.build/other/OtherPackageTests.xctest")
        result = self._evaluate(
            combined_bundle=None, combined_candidates=[self.COMBINED_BUNDLE, other]
        )
        self.assertEqual(result.verdict, suites.REFUSED)
        self.assertIn("more than one combined", result.detail)
        self.assertIn(str(self.COMBINED_BUNDLE), result.detail)
        self.assertIn(str(other), result.detail)

    def test_zero_selectors_is_refused_and_xctest_not_invoked(self) -> None:
        calls: list[list[str]] = []

        def _fake(command, cwd):
            calls.append(command)
            if command[:3] == ["swift", "test", "list"]:
                return subprocess.CompletedProcess(command, 0, "SomeOtherTarget.Foo/bar\n", "")
            raise AssertionError(f"xctest must not be invoked with zero selectors: {command}")

        with mock.patch.object(suites, "run", side_effect=_fake), mock.patch.object(
            suites, "resolve_bundle", return_value=None
        ), mock.patch.object(
            suites,
            "resolve_combined_bundle",
            return_value=(self.COMBINED_BUNDLE, [self.COMBINED_BUNDLE]),
        ):
            result = suites.evaluate_target(
                Path("/fake/dest"),
                "HarnessCoreTests",
                {"tests": 5, "skipped": 0, "failures": 0},
                log_dir=None,
                listing_cache={},
            )
        self.assertEqual(result.verdict, suites.REFUSED)
        self.assertIn("zero class selectors", result.detail)
        self.assertFalse(any(c[0] == "xcrun" for c in calls))

    def test_swift_test_list_failure_is_refused_distinctly(self) -> None:
        result = self._evaluate(combined_bundle=self.COMBINED_BUNDLE, list_returncode=3)
        self.assertEqual(result.verdict, suites.REFUSED)
        self.assertIn("swift test list", result.detail)

    def test_crash_with_no_selected_tests_summary_is_refused(self) -> None:
        result = self._evaluate(
            combined_bundle=self.COMBINED_BUNDLE,
            xctest_stdout="Test Case '-[HarnessCoreTests.SomeTest testFoo]' started.\n",
        )
        self.assertEqual(result.verdict, suites.REFUSED)
        self.assertIn("crash or truncated run", result.detail)

    def test_refusal_shapes_are_pairwise_distinct(self) -> None:
        """Anti-vacuity: a test that only checked `verdict == REFUSED` for every shape would keep
        passing even if they were collapsed back into one indistinguishable message."""
        neither = self._evaluate(per_target_bundle_found=False, combined_bundle=None)
        skeleton_no_combined = self._evaluate(
            per_target_bundle_found=True, per_target_executable_found=False, combined_bundle=None
        )
        too_many_combined = self._evaluate(
            combined_bundle=None,
            combined_candidates=[self.COMBINED_BUNDLE, Path("/fake/dest/other.xctest")],
        )
        list_failed = self._evaluate(combined_bundle=self.COMBINED_BUNDLE, list_returncode=3)
        crashed = self._evaluate(
            combined_bundle=self.COMBINED_BUNDLE,
            xctest_stdout="Test Case '-[HarnessCoreTests.SomeTest testFoo]' started.\n",
        )
        both_present = self._evaluate(
            per_target_bundle_found=True,
            per_target_executable_found=True,
            combined_bundle=self.COMBINED_BUNDLE,
        )
        details = {
            neither.detail,
            skeleton_no_combined.detail,
            too_many_combined.detail,
            list_failed.detail,
            crashed.detail,
            both_present.detail,
        }
        self.assertEqual(len(details), 6, f"expected six distinct refusal details, got: {details}")
        for result in (
            neither,
            skeleton_no_combined,
            too_many_combined,
            list_failed,
            crashed,
            both_present,
        ):
            self.assertEqual(result.verdict, suites.REFUSED)

    def test_toolchain_refusal_surfaces_the_captured_swift_version(self) -> None:
        result = self._evaluate(per_target_bundle_found=False, combined_bundle=None)
        self.assertIn("Apple Swift version 6.3.3 (fake)", result.detail)

    # -- swift test list caching ----------------------------------------------------------------

    def test_swift_test_list_is_reused_from_a_shared_cache_across_calls(self) -> None:
        """`evaluate_target` itself must not re-run `swift test list` if the cache already has an
        answer -- `run_gate`'s once-per-gate-run guarantee (tested below) depends on
        `evaluate_target` honouring a pre-populated cache rather than always calling
        `swift_test_listing` fresh."""
        calls: list[list[str]] = []

        def _fake(command, cwd):
            calls.append(command)
            if command[:3] == ["swift", "test", "list"]:
                raise AssertionError("swift test list must not run again: cache was pre-populated")
            if command[0] == "xcrun":
                return subprocess.CompletedProcess(
                    command, 0, selected_tests_summary(tests=5, failures=0), ""
                )
            raise AssertionError(f"unexpected command: {command}")

        shared_cache = {"done": True, "listing": self.LISTING, "error": None}
        with mock.patch.object(suites, "run", side_effect=_fake), mock.patch.object(
            suites, "resolve_bundle", return_value=None
        ), mock.patch.object(
            suites,
            "resolve_combined_bundle",
            return_value=(self.COMBINED_BUNDLE, [self.COMBINED_BUNDLE]),
        ):
            result = suites.evaluate_target(
                Path("/fake/dest"),
                "HarnessCoreTests",
                {"tests": 5, "skipped": 0, "failures": 0},
                log_dir=None,
                listing_cache=shared_cache,
            )
        self.assertEqual(result.verdict, suites.OK)


class SwiftTestListingTests(unittest.TestCase):
    """Direct tests of the caching wrapper around `swift test list --skip-build`."""

    def test_runs_once_and_caches_across_repeated_calls(self) -> None:
        call_count = {"n": 0}

        def _fake(command, cwd):
            call_count["n"] += 1
            return subprocess.CompletedProcess(command, 0, "listing text\n", "")

        cache: dict = {}
        with mock.patch.object(suites, "run", side_effect=_fake):
            first = suites.swift_test_listing(Path("/fake/dest"), cache)
            second = suites.swift_test_listing(Path("/fake/dest"), cache)
        self.assertEqual(first, ("listing text\n", None))
        self.assertEqual(second, ("listing text\n", None))
        self.assertEqual(call_count["n"], 1)

    def test_failure_caches_the_error_not_the_listing(self) -> None:
        def _fake(command, cwd):
            return subprocess.CompletedProcess(command, 1, "", "boom")

        cache: dict = {}
        with mock.patch.object(suites, "run", side_effect=_fake):
            listing, error = suites.swift_test_listing(Path("/fake/dest"), cache)
        self.assertIsNone(listing)
        self.assertIsNotNone(error)


class RunGateTests(unittest.TestCase):
    """Tests `run_gate`'s once-per-run package build and its interaction with `evaluate_target`
    across multiple targets, with `run` stubbed so no real toolchain is required."""

    EXPECTATIONS = {
        "targets": {
            "AlphaTests": {"tests": 3, "skipped": 0, "failures": 0},
            "BetaTests": {"tests": 2, "skipped": 0, "failures": 0},
        }
    }

    def test_package_build_failure_refuses_every_target_and_never_invokes_xctest(self) -> None:
        calls: list[list[str]] = []

        def _fake(command, cwd):
            calls.append(command)
            if command[:2] == ["swift", "build"]:
                return subprocess.CompletedProcess(command, 1, "", "compile error")
            raise AssertionError(f"no command should run after a failed package build: {command}")

        with mock.patch.object(suites, "run", side_effect=_fake):
            exit_code = suites.run_gate(
                Path("/fake/dest"),
                ["AlphaTests", "BetaTests"],
                self.EXPECTATIONS,
                log_dir=None,
                update=False,
                expectations_path=Path("/fake/expectations.json"),
            )
        self.assertEqual(exit_code, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0], ["swift", "build", "--package-path", "spike", "--build-tests"]
        )

    def test_package_build_failure_writes_a_package_build_log(self) -> None:
        import tempfile

        def _fake(command, cwd):
            if command[:2] == ["swift", "build"]:
                return subprocess.CompletedProcess(command, 1, "compile output", "compile error")
            raise AssertionError(f"unexpected command: {command}")

        with tempfile.TemporaryDirectory() as scratch:
            log_dir = Path(scratch) / "logs"
            with mock.patch.object(suites, "run", side_effect=_fake):
                suites.run_gate(
                    Path("/fake/dest"),
                    ["AlphaTests"],
                    self.EXPECTATIONS,
                    log_dir=log_dir,
                    update=False,
                    expectations_path=Path("/fake/expectations.json"),
                )
            build_log = (log_dir / "_package-build.log").read_text(encoding="utf-8")
            self.assertIn("compile output", build_log)
            self.assertIn("compile error", build_log)

    def test_swift_test_list_runs_at_most_once_across_two_combined_mode_targets(self) -> None:
        listing = (
            "AlphaTests.AlphaTests/testOne\n"
            "BetaTests.BetaTests/testOne\n"
        )
        list_calls: list[list[str]] = []

        def _fake(command, cwd):
            if command[:2] == ["swift", "build"]:
                return subprocess.CompletedProcess(command, 0, "", "")
            if command[:3] == ["swift", "test", "list"]:
                list_calls.append(command)
                return subprocess.CompletedProcess(command, 0, listing, "")
            if command[0] == "xcrun":
                selector_arg = command[command.index("-XCTest") + 1]
                if selector_arg.startswith("AlphaTests."):
                    return subprocess.CompletedProcess(
                        command, 0, selected_tests_summary(tests=3, failures=0), ""
                    )
                return subprocess.CompletedProcess(
                    command, 0, selected_tests_summary(tests=2, failures=0), ""
                )
            raise AssertionError(f"unexpected command: {command}")

        with mock.patch.object(suites, "run", side_effect=_fake), mock.patch.object(
            suites, "resolve_bundle", return_value=None
        ), mock.patch.object(
            suites,
            "resolve_combined_bundle",
            return_value=(
                Path("/fake/dest/spike/.build/x/fast-mlx-spikePackageTests.xctest"),
                [Path("/fake/dest/spike/.build/x/fast-mlx-spikePackageTests.xctest")],
            ),
        ):
            exit_code = suites.run_gate(
                Path("/fake/dest"),
                ["AlphaTests", "BetaTests"],
                self.EXPECTATIONS,
                log_dir=None,
                update=False,
                expectations_path=Path("/fake/expectations.json"),
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(list_calls), 1)


if __name__ == "__main__":
    unittest.main()
