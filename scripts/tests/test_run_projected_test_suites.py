"""Tests for the parser and refusal behaviour behind `run_projected_test_suites.py`.

Every verdict that gate produces flows through `parse_summary` and `evaluate_target`'s
OK/MISMATCH/REFUSED classification. This module runs on CPU only, never shells out to
`swift`/`xcrun`, and never touches the network -- it drives those two functions with
recorded XCTest output shapes and stubbed subprocess results, which is why it lives in the
fast `scripts/tests` suite rather than beside the gate itself.
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


def summary_block(target: str, tests: int, failures: int, skipped: int | None = None) -> str:
    """Build one bundle-level XCTest summary block in the exact shape `parse_summary` anchors
    on: a `Test Suite '<target>.xctest' passed|failed at ...` line immediately followed by the
    `Executed N tests, with ... failures ...` counts line."""
    if skipped is not None:
        counts = (
            f"Executed {tests} tests, with {skipped} tests skipped and {failures} failures "
            "(0 unexpected) in 0.100 (0.105) seconds"
        )
    else:
        counts = f"Executed {tests} tests, with {failures} failures (0 unexpected) in 0.100 (0.105) seconds"
    return (
        f"Test Suite '{target}.xctest' passed at 2026-09-17 10:00:00.000.\n"
        f"\t {counts}\n"
    )


class ParseSummaryTests(unittest.TestCase):
    """Direct tests of the pure parser, independent of the build/run plumbing around it."""

    def test_parses_summary_without_skipped_clause(self) -> None:
        """Older/plain XCTest output omits the skipped clause entirely; a regex that assumes it
        is always present would fail to match this shape at all and silently REFUSE every run on
        a toolchain that doesn't print it."""
        output = summary_block("HarnessCoreTests", tests=5, failures=0)
        self.assertEqual(suites.parse_summary(output, "HarnessCoreTests"), (5, 0, 0))

    def test_parses_summary_with_skipped_clause(self) -> None:
        """The common case on this project's toolchain: skipped count must land in the middle
        slot, not get merged into tests or failures."""
        output = summary_block("HarnessCoreTests", tests=12, failures=1, skipped=2)
        self.assertEqual(suites.parse_summary(output, "HarnessCoreTests"), (12, 2, 1))

    def test_multi_bundle_output_does_not_read_a_neighbouring_bundles_summary(self) -> None:
        """Reproduces the exact hazard the module docstring names: a run that prints more than
        one bundle's summary (e.g. from `swift test --filter`) puts an unrelated bundle's
        `Executed 0 tests` line right next to the requested target's real one. A parser that
        isn't anchored precisely to `'<target>.xctest'` could read the wrong line and report a
        confident, wrong count instead of the real one."""
        blob = summary_block("UnrelatedTests", tests=0, failures=0) + summary_block(
            "HarnessCoreTests", tests=42, failures=3
        )
        self.assertEqual(suites.parse_summary(blob, "HarnessCoreTests"), (42, 0, 3))

    def test_multi_bundle_output_resolves_each_of_two_real_targets_correctly(self) -> None:
        """Same hazard, but with two targets that both have real (nonzero) counts, so a swapped
        read would still look plausible instead of obviously wrong -- pin which summary belongs
        to which target in both directions."""
        blob = summary_block("AlphaTests", tests=40, failures=3) + summary_block(
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
        output = summary_block("HarnessCoreTests", tests=0, failures=0)
        self.assertEqual(suites.parse_summary(output, "HarnessCoreTests"), (0, 0, 0))


def _stub_run(build_returncode: int = 0, xctest_stdout: str = "", xctest_stderr: str = ""):
    """A `run_projected_test_suites.run` stand-in that answers the two subprocess calls
    `evaluate_target` makes (`swift build ...` then `xcrun xctest ...`) without touching a real
    toolchain."""

    def _fake(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        if command[:2] == ["swift", "build"]:
            return subprocess.CompletedProcess(
                args=command, returncode=build_returncode, stdout="", stderr=""
            )
        if command[0] == "xcrun":
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout=xctest_stdout, stderr=xctest_stderr
            )
        raise AssertionError(f"unexpected command in evaluate_target stub: {command}")

    return _fake


class EvaluateTargetTests(unittest.TestCase):
    """Tests the OK/MISMATCH/REFUSED classification in `evaluate_target`, with `run` and
    `resolve_bundle` stubbed so no real `swift`/`xcrun` toolchain is required."""

    def _evaluate(
        self,
        expected_tests: int,
        expected_failures: int,
        build_returncode: int = 0,
        xctest_stdout: str = "",
        bundle_found: bool = True,
        executable_found: bool = True,
    ) -> suites.TargetResult:
        expected = {"tests": expected_tests, "skipped": 0, "failures": expected_failures}
        bundle = Path("/fake/dest/spike/.build/out/Products/Debug/HarnessCoreTests.xctest")
        with mock.patch.object(
            suites, "run", side_effect=_stub_run(build_returncode, xctest_stdout)
        ), mock.patch.object(
            suites, "resolve_bundle", return_value=(bundle if bundle_found else None)
        ), mock.patch.object(
            suites, "bundle_has_executable", return_value=executable_found
        ), mock.patch.object(
            suites, "swift_toolchain_version", return_value="Apple Swift version 6.3.3 (fake)"
        ):
            return suites.evaluate_target(
                Path("/fake/dest"), "HarnessCoreTests", expected, log_dir=None
            )

    def test_exact_match_is_ok(self) -> None:
        result = self._evaluate(
            expected_tests=5,
            expected_failures=0,
            xctest_stdout=summary_block("HarnessCoreTests", tests=5, failures=0),
        )
        self.assertEqual(result.verdict, suites.OK)

    def test_failure_count_higher_than_pinned_is_mismatch(self) -> None:
        """A regression: more failures than the pinned baseline recorded."""
        result = self._evaluate(
            expected_tests=5,
            expected_failures=0,
            xctest_stdout=summary_block("HarnessCoreTests", tests=5, failures=1),
        )
        self.assertEqual(result.verdict, suites.MISMATCH)

    def test_failure_count_lower_than_pinned_is_also_mismatch(self) -> None:
        """Anti-vacuity: a target quietly starting to pass more than it used to must ALSO trip
        the gate, so the improvement gets recorded on purpose via --update-expectations instead
        of drifting unnoticed. This is the "lower is also a failure" rule from the module
        docstring -- if `evaluate_target` ever loosened this to only flag increases, this test
        would start passing against that regression instead of catching it."""
        result = self._evaluate(
            expected_tests=5,
            expected_failures=2,
            xctest_stdout=summary_block("HarnessCoreTests", tests=5, failures=0),
        )
        self.assertEqual(result.verdict, suites.MISMATCH)

    def test_test_count_higher_than_pinned_is_mismatch(self) -> None:
        """An unrecorded addition must force a conscious baseline update, not drift through."""
        result = self._evaluate(
            expected_tests=5,
            expected_failures=0,
            xctest_stdout=summary_block("HarnessCoreTests", tests=6, failures=0),
        )
        self.assertEqual(result.verdict, suites.MISMATCH)

    def test_test_count_lower_than_pinned_is_mismatch(self) -> None:
        """Catches "green by deletion": quietly removing tests must not look like success."""
        result = self._evaluate(
            expected_tests=5,
            expected_failures=0,
            xctest_stdout=summary_block("HarnessCoreTests", tests=3, failures=0),
        )
        self.assertEqual(result.verdict, suites.MISMATCH)

    def test_zero_executed_tests_is_mismatch_not_ok(self) -> None:
        """A silently-broken test discovery exits 0 and prints a well-formed zero-count summary;
        this must never be indistinguishable from a real pass."""
        result = self._evaluate(
            expected_tests=5,
            expected_failures=0,
            xctest_stdout=summary_block("HarnessCoreTests", tests=0, failures=0),
        )
        self.assertNotEqual(result.verdict, suites.OK)
        self.assertEqual(result.verdict, suites.MISMATCH)

    def test_build_failure_is_refused(self) -> None:
        result = self._evaluate(expected_tests=5, expected_failures=0, build_returncode=1)
        self.assertEqual(result.verdict, suites.REFUSED)

    def test_missing_bundle_after_successful_build_is_refused(self) -> None:
        """Case (a): the build reported success but no `.xctest` bundle exists anywhere. This must
        name the toolchain gap, not blame a crash or truncated run -- distinguished from the other
        two REFUSED shapes below by `test_the_three_refusal_shapes_are_distinguishable_not_lumped`."""
        result = self._evaluate(expected_tests=5, expected_failures=0, bundle_found=False)
        self.assertEqual(result.verdict, suites.REFUSED)
        self.assertIn("no", result.detail)
        self.assertIn("HarnessCoreTests.xctest bundle exists", result.detail)
        self.assertIn("does not", result.detail.lower())
        self.assertNotIn("crash or truncated run", result.detail)

    def test_bundle_present_but_executable_missing_is_refused(self) -> None:
        """Case (b): the bundle directory exists (an incremental-tree skeleton), but
        `Contents/MacOS/<target>` is missing or empty, so nothing can be run. This must be a
        DIFFERENT message from both the no-bundle-at-all case and the crash/truncated-run case --
        it is the same underlying toolchain gap caught one resolution stage later."""
        result = self._evaluate(expected_tests=5, expected_failures=0, executable_found=False)
        self.assertEqual(result.verdict, suites.REFUSED)
        self.assertIn("no executable", result.detail)
        self.assertIn("Contents/MacOS/HarnessCoreTests", result.detail)
        self.assertNotIn("crash or truncated run", result.detail)

    def test_crash_with_no_summary_line_is_refused_not_a_pass(self) -> None:
        """Case (c): the bundle existed, had an executable, and ran -- but printed no parseable
        summary line. This is the ONLY case where "crash or truncated run" is the correct
        diagnosis, and it must stay distinct from cases (a) and (b) above, which are toolchain
        gaps, not crashes."""
        result = self._evaluate(
            expected_tests=5,
            expected_failures=0,
            xctest_stdout="Test Case '-[HarnessCoreTests.SomeTest testFoo]' started.\n",
        )
        self.assertEqual(result.verdict, suites.REFUSED)
        self.assertIn("crash or truncated run", result.detail)

    def test_the_three_refusal_shapes_are_distinguishable_not_lumped(self) -> None:
        """Anti-vacuity: a test that only checked `verdict == REFUSED` for all three shapes would
        keep passing even if they were collapsed back into one indistinguishable message. Pin that
        the three details are pairwise different, not merely that all three refuse."""
        no_bundle = self._evaluate(expected_tests=5, expected_failures=0, bundle_found=False)
        no_executable = self._evaluate(
            expected_tests=5, expected_failures=0, executable_found=False
        )
        crash = self._evaluate(
            expected_tests=5,
            expected_failures=0,
            xctest_stdout="Test Case '-[HarnessCoreTests.SomeTest testFoo]' started.\n",
        )
        details = {no_bundle.detail, no_executable.detail, crash.detail}
        self.assertEqual(len(details), 3, f"expected three distinct refusal details, got: {details}")
        for result in (no_bundle, no_executable, crash):
            self.assertEqual(result.verdict, suites.REFUSED)

    def test_toolchain_refusals_surface_the_captured_swift_version(self) -> None:
        """Requirement: where the cause is the toolchain, the refusal should surface the active
        Swift/Xcode version if cheaply available. Pin that the stubbed version string actually
        lands in both toolchain-caused refusal details (cases a and b), not just that a version
        function exists and is unused."""
        no_bundle = self._evaluate(expected_tests=5, expected_failures=0, bundle_found=False)
        no_executable = self._evaluate(
            expected_tests=5, expected_failures=0, executable_found=False
        )
        self.assertIn("Apple Swift version 6.3.3 (fake)", no_bundle.detail)
        self.assertIn("Apple Swift version 6.3.3 (fake)", no_executable.detail)


if __name__ == "__main__":
    unittest.main()
