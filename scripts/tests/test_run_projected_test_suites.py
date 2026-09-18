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
import tempfile
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


class SwiftTestingFilterPatternTests(unittest.TestCase):
    def test_anchored_with_exact_trailing_dot_prefix(self) -> None:
        self.assertEqual(suites.swift_testing_filter_pattern("HarnessCoreTests"), r"^HarnessCoreTests\.")

    def test_escapes_regex_metacharacters_in_target_name(self) -> None:
        # No pinned target name has regex metacharacters today, but the pattern must not silently
        # misbehave if one ever did (e.g. a target literally named "A.B" must not have its dot
        # treated as "any character").
        self.assertEqual(suites.swift_testing_filter_pattern("A.B"), r"^A\.B\.")


class ParseSwiftTestingXunitTests(unittest.TestCase):
    """Direct tests of the Swift Testing xunit parser, using REAL `--xunit-output` artifacts
    captured 2026-09-17 from `swift test --package-path spike --skip-build --disable-xctest
    --filter '^HarnessCoreTests\\.' --xunit-output <path>`, trimmed to a manageable number of
    `<testcase>` lines while preserving the exact tag/attribute shape each toolchain produced.
    Expected counts below are recomputed for the trimmed subset actually kept, not the full
    49-case run the untrimmed capture reported."""

    # Captured on the consumer host, Swift 6.3.3 / Xcode 26.6: exactly one `<testsuite>` element
    # for the whole filtered run (that toolchain builds one combined bundle -- see the module
    # docstring). Trimmed to 3 real `HarnessCoreTests.*` testcases, plus one synthetic sibling
    # line (`HarnessCoreTestsExtra.Foo`, never actually observed in this project) appended to
    # prove the trailing-dot exact-prefix rule, the same hazard `class_selectors` guards against.
    SIX_3_3_PASS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="TestResults" errors="0" tests="4" failures="0" skipped="0" time="0.007172792">
    <testcase classname="HarnessCoreTests.SystemProfileOperatorBudgetTests" name="sameHostWithNilBudgetStaysMeasured()" time="0.001071166" />
    <testcase classname="HarnessCoreTests.ForcedScoringPlanTests" name="singleChunkCoversAllPositions()" time="0.001074125" />
    <testcase classname="HarnessCoreTests.TailStatisticTests" name="quantileUsesCeilingIndexConvention()" time="0.000628084" />
    <testcase classname="HarnessCoreTestsExtra.Foo" name="unrelatedSiblingModule()" time="0.000100000" />
  </testsuite>
</testsuites>
"""

    # Captured on the same host/toolchain after a scratch `@Test func zzzScratchAlwaysFails()`
    # was added to force a real failure shape: the `<failure message="...">` child element.
    SIX_3_3_FAIL_XML = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="TestResults" errors="0" tests="3" failures="1" skipped="0" time="0.007270791">
    <testcase classname="HarnessCoreTests.WiredCeilingOvercommitGuardTests" name="boundaryIsExclusiveAtThresholdAndInclusiveOneByteAbove()" time="0.00110725" />
    <testcase classname="HarnessCoreTests.SystemProfileOperatorBudgetTests" name="operatorBudgetDoesNotDegradeHostProvenance()" time="0.000926792" />
    <testcase classname="HarnessCoreTests.ZZZScratchFailingTest" name="zzzScratchAlwaysFails()" time="0.00127475" >
      <failure message="Expectation failed: 1 == 2 (error)" />
    </testcase>
  </testsuite>
</testsuites>
"""

    # Captured for a target with genuinely zero Swift Testing cases (`ServingCoreTests`, Swift
    # 6.3.3): a real, well-formed, empty `<testsuite>` -- must parse to (0, 0), not None.
    SIX_3_3_ZERO_XML = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<testsuites>\n"
        '  <testsuite name="TestResults" errors="0" tests="0" failures="0" skipped="0" '
        'time="0.000129834">\n\n  </testsuite>\n</testsuites>\n'
    )

    # Captured on the dev box, Swift 6.4 / Xcode 27: `--build-tests` produces per-target bundles
    # there (see the module docstring), so the SAME filtered command still walks every OTHER
    # bundle in the package and contributes one empty sibling `<testsuite>` per bundle walked,
    # all identically named "TestResults". Trimmed to 3 empty siblings (the real capture had ~11)
    # plus the one holding real `HarnessCoreTests.*` testcases (trimmed to 2 of its own).
    SIX_4_MULTI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
    <testsuite name="TestResults" errors="0" tests="0" failures="0" skipped="0" time="0.002097083"/>
    <testsuite name="TestResults" errors="0" tests="0" failures="0" skipped="0" time="0.000338083"/>
    <testsuite name="TestResults" errors="0" tests="2" failures="0" skipped="0" time="0.037702458">
        <testcase classname="HarnessCoreTests.ForcedScoringPlanTests" name="promptLongerThanChunkYieldsRowlessLeadingChunks()" time="0.005697666"/>
        <testcase classname="HarnessCoreTests.WiredCeilingOvercommitGuardTests" name="advisoryLinesCarrySystemCeilingBudgetExcessAndMachineToken()" time="0.005290959"/>
    </testsuite>
    <testsuite name="TestResults" errors="0" tests="0" failures="0" skipped="0" time="0.000290792"/>
</testsuites>
"""

    def test_six_3_3_pass_counts_only_the_exact_module_prefix(self) -> None:
        """Also proves the trailing-dot exact-prefix rule: the appended
        `HarnessCoreTestsExtra.Foo` sibling testcase must not be counted towards
        `HarnessCoreTests`, the same hazard `class_selectors` guards against for XCTest."""
        self.assertEqual(
            suites.parse_swift_testing_xunit(self.SIX_3_3_PASS_XML, "HarnessCoreTests"), (3, 0)
        )

    def test_six_3_3_fail_counts_the_failure_child_element(self) -> None:
        self.assertEqual(
            suites.parse_swift_testing_xunit(self.SIX_3_3_FAIL_XML, "HarnessCoreTests"), (3, 1)
        )

    def test_six_3_3_zero_is_a_real_tuple_not_none(self) -> None:
        self.assertEqual(
            suites.parse_swift_testing_xunit(self.SIX_3_3_ZERO_XML, "ServingCoreTests"), (0, 0)
        )

    def test_six_4_multi_testsuite_sums_only_the_matching_sibling(self) -> None:
        """The empty sibling `<testsuite>` elements from other bundles must not corrupt the
        count, and must not need to be located by name or position -- `parse_swift_testing_xunit`
        never looks at which `<testsuite>` a `<testcase>` belongs to."""
        self.assertEqual(
            suites.parse_swift_testing_xunit(self.SIX_4_MULTI_XML, "HarnessCoreTests"), (2, 0)
        )

    def test_truncated_document_yields_none(self) -> None:
        """A process that crashed mid-write leaves an invalid XML document -- must refuse, never
        coerce to (0, 0)."""
        truncated = '<?xml version="1.0"?><testsuites><testsuite name="TestResults" tests="1">'
        self.assertIsNone(suites.parse_swift_testing_xunit(truncated, "HarnessCoreTests"))

    def test_well_formed_document_with_no_testsuite_at_all_yields_none(self) -> None:
        """Measured 2026-09-17: even a filter matching nothing still produces one empty
        `<testsuite>` per bundle walked. A document with ZERO `<testsuite>` elements anywhere
        means `swift test` never got far enough to report anything at all."""
        empty = '<?xml version="1.0"?><testsuites></testsuites>'
        self.assertIsNone(suites.parse_swift_testing_xunit(empty, "HarnessCoreTests"))

    def test_not_xml_at_all_yields_none(self) -> None:
        self.assertIsNone(
            suites.parse_swift_testing_xunit("dyld: Library not loaded\nAbort trap: 6\n", "HarnessCoreTests")
        )


class SwiftTestingTargetIsKnownTests(unittest.TestCase):
    """Direct tests of the zero-case vacuity guard, reusing `ClassSelectorsTests.LISTING`."""

    LISTING = ClassSelectorsTests.LISTING

    def test_target_with_xctest_cases_is_known_even_with_zero_swift_testing_cases(self) -> None:
        self.assertTrue(suites.swift_testing_target_is_known(self.LISTING, "HarnessCoreTests"))
        self.assertTrue(suites.swift_testing_target_is_known(self.LISTING, "ServingCoreTests"))

    def test_unknown_target_is_not_known(self) -> None:
        self.assertFalse(suites.swift_testing_target_is_known(self.LISTING, "NoSuchTarget"))

    def test_sibling_module_does_not_falsely_confirm_the_real_target(self) -> None:
        """`ServingCoreTestsExtra` existing in the listing must not make a caller asking about
        `ServingCoreTests` itself get a false True from that unrelated module alone -- it is
        True here only because a genuine `ServingCoreTests.` line also exists."""
        listing_without_the_real_module = (
            "ServingCoreTestsExtra.SomethingTests/unrelatedCase\n"
        )
        self.assertFalse(
            suites.swift_testing_target_is_known(listing_without_the_real_module, "ServingCoreTests")
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


class EvaluateSwiftTestingTargetTests(unittest.TestCase):
    """Tests `evaluate_swift_testing_target`'s OK/MISMATCH/REFUSED classification, with `run`
    stubbed and a REAL temporary directory standing in for `xunit_dir` -- the function reads the
    xunit artifact from disk, so tests exercise that real read/parse path rather than mocking it
    away, while still requiring no real `swift` toolchain."""

    LISTING = "HarnessCoreTests.SomeXCTestClass/testSomething\n"

    def _run_with_xunit(
        self,
        target: str = "HarnessCoreTests",
        expected_tests: int = 0,
        expected_failures: int = 0,
        xunit_xml: str | None = None,
        write_xunit: bool = True,
        list_returncode: int = 0,
        list_stdout: str | None = None,
        listing_cache: dict | None = None,
    ) -> suites.SwiftTestingResult:
        if list_stdout is None:
            list_stdout = self.LISTING

        def _fake(command, cwd):
            if command[:3] == ["swift", "test", "list"]:
                return subprocess.CompletedProcess(command, list_returncode, list_stdout, "")
            if command[:2] == ["swift", "test"]:
                if write_xunit:
                    xunit_path = Path(command[command.index("--xunit-output") + 1])
                    xunit_path.write_text(xunit_xml or "", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, "", "")
            raise AssertionError(f"unexpected command: {command}")

        with tempfile.TemporaryDirectory() as scratch:
            xunit_dir = Path(scratch)
            with mock.patch.object(suites, "run", side_effect=_fake):
                return suites.evaluate_swift_testing_target(
                    Path("/fake/dest"),
                    target,
                    {"tests": expected_tests, "failures": expected_failures},
                    log_dir=None,
                    listing_cache=(listing_cache if listing_cache is not None else {}),
                    xunit_dir=xunit_dir,
                )

    def test_command_uses_anchored_filter_and_disables_xctest(self) -> None:
        captured: list[list[str]] = []

        def _fake(command, cwd):
            captured.append(command)
            if command[:2] == ["swift", "test"]:
                xunit_path = Path(command[command.index("--xunit-output") + 1])
                xunit_path.write_text(
                    '<?xml version="1.0"?><testsuites><testsuite name="TestResults" tests="0" '
                    'failures="0"/></testsuites>',
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, "", "")
            raise AssertionError(f"unexpected command: {command}")

        with tempfile.TemporaryDirectory() as scratch, mock.patch.object(
            suites, "run", side_effect=_fake
        ):
            suites.evaluate_swift_testing_target(
                Path("/fake/dest"),
                "HarnessCoreTests",
                {"tests": 0, "failures": 0},
                log_dir=None,
                listing_cache={"done": True, "listing": self.LISTING, "error": None},
                xunit_dir=Path(scratch),
            )
        self.assertEqual(len(captured), 1)
        command = captured[0]
        self.assertIn("--disable-xctest", command)
        self.assertIn("--skip-build", command)
        self.assertEqual(command[command.index("--filter") + 1], r"^HarnessCoreTests\.")

    def test_nonzero_match_ok(self) -> None:
        xml = (
            '<?xml version="1.0"?><testsuites><testsuite name="TestResults" tests="1" '
            'failures="0"><testcase classname="HarnessCoreTests.Foo" name="bar()"/>'
            "</testsuite></testsuites>"
        )
        result = self._run_with_xunit(expected_tests=1, expected_failures=0, xunit_xml=xml)
        self.assertEqual(result.verdict, suites.OK)
        self.assertEqual((result.actual_tests, result.actual_failures), (1, 0))

    def test_nonzero_count_mismatch(self) -> None:
        xml = (
            '<?xml version="1.0"?><testsuites><testsuite name="TestResults" tests="1" '
            'failures="0"><testcase classname="HarnessCoreTests.Foo" name="bar()"/>'
            "</testsuite></testsuites>"
        )
        result = self._run_with_xunit(expected_tests=2, expected_failures=0, xunit_xml=xml)
        self.assertEqual(result.verdict, suites.MISMATCH)

    def test_nonzero_failure_count_mismatch(self) -> None:
        xml = (
            '<?xml version="1.0"?><testsuites><testsuite name="TestResults" tests="1" '
            'failures="1"><testcase classname="HarnessCoreTests.Foo" name="bar()">'
            '<failure message="x"/></testcase></testsuite></testsuites>'
        )
        result = self._run_with_xunit(expected_tests=1, expected_failures=0, xunit_xml=xml)
        self.assertEqual(result.verdict, suites.MISMATCH)

    def test_legitimate_zero_is_ok_when_target_known_in_listing(self) -> None:
        xml = '<?xml version="1.0"?><testsuites><testsuite name="TestResults" tests="0" failures="0"/></testsuites>'
        result = self._run_with_xunit(
            target="HarnessCoreTests", expected_tests=0, expected_failures=0, xunit_xml=xml,
            list_stdout="HarnessCoreTests.SomeXCTestClass/testSomething\n",
        )
        self.assertEqual(result.verdict, suites.OK)
        self.assertEqual((result.actual_tests, result.actual_failures), (0, 0))

    def test_zero_with_target_unknown_in_listing_is_refused(self) -> None:
        """Closes the zero-case vacuity trap: a zero result for a target that does not otherwise
        appear in `swift test list` must not be silently accepted as a legitimate pass, even
        though it is pinned at zero."""
        xml = '<?xml version="1.0"?><testsuites><testsuite name="TestResults" tests="0" failures="0"/></testsuites>'
        result = self._run_with_xunit(
            target="TotallyUnbuiltTarget", expected_tests=0, expected_failures=0, xunit_xml=xml,
            list_stdout="SomeOtherModule.Foo/bar\n",
        )
        self.assertEqual(result.verdict, suites.REFUSED)
        self.assertIn("does not otherwise appear in", result.detail)

    def test_zero_pin_mismatch_when_actual_is_nonzero(self) -> None:
        xml = (
            '<?xml version="1.0"?><testsuites><testsuite name="TestResults" tests="1" '
            'failures="0"><testcase classname="HarnessCoreTests.Foo" name="bar()"/>'
            "</testsuite></testsuites>"
        )
        result = self._run_with_xunit(expected_tests=0, expected_failures=0, xunit_xml=xml)
        self.assertEqual(result.verdict, suites.MISMATCH)

    def test_missing_artifact_is_refused(self) -> None:
        result = self._run_with_xunit(write_xunit=False)
        self.assertEqual(result.verdict, suites.REFUSED)
        self.assertIn("no Swift Testing xunit artifact", result.detail)

    def test_malformed_artifact_is_refused(self) -> None:
        result = self._run_with_xunit(xunit_xml="<?xml version=\"1.0\"?><testsuites><testsuite>")
        self.assertEqual(result.verdict, suites.REFUSED)
        self.assertIn("malformed", result.detail)

    def test_listing_failure_on_a_zero_result_is_refused_distinctly(self) -> None:
        xml = '<?xml version="1.0"?><testsuites><testsuite name="TestResults" tests="0" failures="0"/></testsuites>'
        result = self._run_with_xunit(
            expected_tests=0, expected_failures=0, xunit_xml=xml, list_returncode=3
        )
        self.assertEqual(result.verdict, suites.REFUSED)
        self.assertIn("could not be corroborated", result.detail)

    def test_refusal_shapes_are_pairwise_distinct(self) -> None:
        missing = self._run_with_xunit(write_xunit=False)
        malformed = self._run_with_xunit(xunit_xml="<broken")
        zero_unknown = self._run_with_xunit(
            target="TotallyUnbuiltTarget",
            xunit_xml='<?xml version="1.0"?><testsuites><testsuite name="TestResults" tests="0" failures="0"/></testsuites>',
            list_stdout="SomeOtherModule.Foo/bar\n",
        )
        listing_failed = self._run_with_xunit(
            xunit_xml='<?xml version="1.0"?><testsuites><testsuite name="TestResults" tests="0" failures="0"/></testsuites>',
            list_returncode=3,
        )
        details = {missing.detail, malformed.detail, zero_unknown.detail, listing_failed.detail}
        self.assertEqual(len(details), 4, f"expected four distinct refusal details, got: {details}")
        for result in (missing, malformed, zero_unknown, listing_failed):
            self.assertEqual(result.verdict, suites.REFUSED)

    def test_does_not_invoke_swift_test_list_when_actual_tests_is_nonzero(self) -> None:
        """The listing cross-check only matters for a zero result -- a nonzero, correctly
        attributed count is already self-evidently real and must not pay for an extra
        subprocess."""
        xml = (
            '<?xml version="1.0"?><testsuites><testsuite name="TestResults" tests="1" '
            'failures="0"><testcase classname="HarnessCoreTests.Foo" name="bar()"/>'
            "</testsuite></testsuites>"
        )

        def _fake(command, cwd):
            if command[:3] == ["swift", "test", "list"]:
                raise AssertionError("swift test list must not run for a nonzero result")
            if command[:2] == ["swift", "test"]:
                xunit_path = Path(command[command.index("--xunit-output") + 1])
                xunit_path.write_text(xml, encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, "", "")
            raise AssertionError(f"unexpected command: {command}")

        with tempfile.TemporaryDirectory() as scratch, mock.patch.object(
            suites, "run", side_effect=_fake
        ):
            result = suites.evaluate_swift_testing_target(
                Path("/fake/dest"),
                "HarnessCoreTests",
                {"tests": 1, "failures": 0},
                log_dir=None,
                listing_cache={},
                xunit_dir=Path(scratch),
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
            "AlphaTests": {
                "tests": 3, "skipped": 0, "failures": 0,
                "swift_testing": {"tests": 0, "failures": 0},
            },
            "BetaTests": {
                "tests": 2, "skipped": 0, "failures": 0,
                "swift_testing": {"tests": 0, "failures": 0},
            },
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
            if command[:2] == ["swift", "test"] and "--disable-xctest" in command:
                xunit_path = Path(command[command.index("--xunit-output") + 1])
                xunit_path.write_text(
                    '<?xml version="1.0"?><testsuites>'
                    '<testsuite name="TestResults" tests="0" failures="0" errors="0" '
                    'skipped="0"/></testsuites>',
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, "", "")
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
