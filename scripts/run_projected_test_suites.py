#!/usr/bin/env python3
"""Run the PUBLIC projection's test targets, which no other gate has ever done.

The public repository ships `spike/Tests/**` -- roughly 300 projected test files -- but until this
gate existed, nothing even compiled them, and compiling is not the same as running. Cycle 81 found
the published engine shipping a fail-open speculation-gate safety defect that a projected test would
have caught: the test compiled cleanly and only failed, silently, at runtime, because the sanitized
override it exercised was missing a safety requirement (an `XCTAssertThrowsError` that did not
throw). A compile-only check would have stayed green through that defect; only actually running the
suite catches it, which is why this gate runs bundles instead of stopping at a successful build.

This gate exports (or reuses) a public projection, builds the package's tests ONCE with `swift build
--package-path spike --build-tests`, then resolves and runs each named test target's `.xctest`
bundle, and compares the parsed result against a PINNED baseline in
`scripts/projected_test_expectations.json`.

TWO RUN MODES, AND WHY BOTH EXIST: a per-target `swift build --package-path spike --target <T>`
build was measured, on this project's original toolchain, to produce a runnable,
individually-attributable `<T>.xctest` bundle. It produces no runnable bundle at all on Swift 6.3.3 /
Xcode 26.6 (the GitHub macos-26 CI runner) -- an eight-target REFUSED gate in CI, not a build failure
the runner could retry past. Measured 2026-09-17 on an Apple M5 24 GiB consumer host on
that same 6.3.3 toolchain, fresh projection with no `.build`: `swift build --package-path spike
--build-tests` -- the whole package, every target together -- is a clean 133 s wall time / 2.1 GB max
RSS build, and it produces exactly ONE combined bundle,
`spike/.build/arm64-apple-macosx/debug/fast-mlx-spikePackageTests.xctest`, with a single executable
inside it covering every target; there are no per-target bundles alongside it on this toolchain. That
measurement made the per-target design's original rationale -- avoiding paying for the MLX/Metal
build on every run -- moot: the whole package already builds in about two minutes on ordinary
consumer hardware, so this gate now always builds the whole package exactly once per run instead of
maintaining a separate cheap-but-toolchain-fragile per-target build path.

MEASURED PER-TOOLCHAIN LAYOUTS, AND THE MIXED-TREE REFUSAL: `swift build --package-path spike
--build-tests` was measured to produce two mutually exclusive shapes depending on the active
toolchain, each written under a different `.build` subdirectory. Swift 6.4 / Xcode 27 produces ONLY
per-target bundles, `spike/.build/out/Products/Debug/<Target>.xctest`, and NO combined bundle at all.
Swift 6.3.3 / Xcode 26.6 produces ONLY the one combined bundle,
`spike/.build/arm64-apple-macosx/debug/fast-mlx-spikePackageTests.xctest`, and NO per-target bundles.
Because the two shapes live in different directories, neither toolchain's build ever deletes the
other's leftovers: a `.build` tree that was built once by each toolchain -- most commonly by
switching Xcode versions locally without removing `spike/.build` in between -- can hold a stale
bundle of one kind sitting right next to a fresh one of the other, and nothing on disk marks which is
stale. If this gate found a per-target bundle with a runnable executable AND a combined bundle with a
runnable executable at the same time, it refuses (REFUSED) rather than silently preferring mode 1 and
possibly running the stale bundle; the detail names both paths and the active toolchain, and neither
`xcrun xctest` nor `swift test list` is invoked. Removing `spike/.build` and re-running clears it.

Per target, this gate resolves a bundle to run in this order:

1. Per-target bundle: if a runnable `<T>.xctest` with `Contents/MacOS/<T>` already exists under
   `spike/.build` -- the shape a per-target `swift build --target <T>` build used to produce, and
   which some toolchains may still leave behind from `--build-tests` or a prior incremental build --
   this gate first checks that no combined bundle with a runnable executable also exists (see the
   mixed-tree refusal above); if none does, it runs the per-target bundle directly with
   `xcrun xctest <bundle>`, exactly as before. This gate no longer runs a per-target
   `swift build --target` itself; it only checks whether one is already sitting there.
2. Combined bundle + class selectors: otherwise, locate the combined `*PackageTests.xctest` bundle
   `--build-tests` produced and run `xcrun xctest -XCTest <selectors> <bundle>`, where `<selectors>`
   is a comma-joined, sorted, deduped list of `Module.Class` names belonging to exactly this target.
   Selectors are derived from `swift test list --skip-build --package-path spike` (exact
   module-prefix match, e.g. `ServingCoreTests.` never matches a sibling like
   `ServingCoreTestsExtra.x/y`); that listing command is run at MOST once per gate invocation and its
   result is shared across every target that needs it, not repeated per target. Per-target
   attribution, which used to come from building and running one target in isolation, now comes from
   this class-selector filter applied to the one shared combined bundle instead.
3. Neither: REFUSED, naming both the per-target and combined-bundle absence (mentioning a skeleton
   per-target bundle with no executable, if one was found) and the active Swift/Xcode version.

Because the whole package now builds once, up front, a compile error ANYWHERE in `spike/Tests` or
`spike/Sources` now refuses every selected target for this run, not just the target whose source
broke it -- the `_package-build.log` this gate writes (when `--log-dir` is given) is what attributes
which target's source actually caused it.

SWIFT TESTING (`@Test`) CASES GET THEIR OWN LEG, SEPARATE FROM THE XCTEST MODES ABOVE: `swift test
list` also enumerates Swift Testing suites/cases -- 49 of them in HarnessCoreTests as of
2026-09-17, e.g. `ForcedScoringPlanTests`, `SystemProfileOperatorBudgetTests`,
`TailStatisticTests`, `WiredCeilingOvercommitGuardTests` -- that neither a direct `xcrun xctest
<bundle>` run nor an `-XCTest <selectors>` run ever executes: XCTest only runs `XCTestCase`
subclasses, in both modes. For each pinned target, after its XCTest leg above, this gate also
runs `swift test --package-path spike --skip-build --disable-xctest --filter <pattern>
--xunit-output <path>`, where `<pattern>` is `^<Target>\\.` -- anchored, with an exact
module-prefix trailing dot, exactly like `class_selectors`' XCTest selectors, so a hypothetical
sibling module cannot be mistaken for the real one. Every target is pinned a `swift_testing`
count (0 for the seven targets that have none today) in `projected_test_expectations.json`,
independent of its XCTest `tests`/`failures` pin, and it gates exactly the same way: MISMATCH on
either direction, REFUSED on an unreadable result.

THE FILTER MUST BE ANCHORED PER TARGET, AND THIS IS A SAFETY PROPERTY, NOT ONLY AN ATTRIBUTION
ONE: an UNFILTERED `swift test --skip-build --disable-xctest` (no `--filter` at all) walks every
bundle built for the whole package, including the MLX/Metal targets this gate deliberately does
not cover (`SpikeCoreTests`, `FastMLXHarnessTests`, `ExactPrefixMLXTests`,
`SpikeServingAdaptersTests` -- see `projected_test_expectations.json`'s `_provenance.scope`).
Measured 2026-09-17 on the Swift 6.4 / Xcode 27 dev box: running unfiltered reaches a Swift
Testing case inside `SpikeServingAdaptersTests` and crashes with `MLX error: Failed to load the
default metallib ... library not found`, because this gate's projection never stages a Metal
library. That crash also corrupted the shared `--xunit-output` file for every other bundle
walked in the same invocation (a truncated, unparseable document) -- one MLX target's missing
Metal library turned into eight unrelated REFUSED verdicts. A `--filter` anchored to exactly one
of the eight MLX-free pinned targets was measured to never select anything belonging to an
MLX/Metal target, so it never reaches that crash; this gate therefore always calls `swift test`
once per target with an anchored filter, never once for the whole package.

MEASURED XUNIT SHAPE ALSO DIFFERS BY TOOLCHAIN, THE SAME WAY THE BUNDLE LAYOUT DOES: on Swift
6.3.3, a per-target filtered `swift test` run produces exactly one `<testsuite
name="TestResults">` element in the xunit document, containing only the matched target's
`<testcase>` elements. On Swift 6.4, the SAME command still produces one `<testsuite
name="TestResults">` element PER BUNDLE walked (there is no per-bundle name to distinguish them
by -- every one is literally named `"TestResults"`), almost all of them empty, alongside the one
holding the target's real `<testcase>` elements. `parse_swift_testing_xunit` does not pick a
`<testsuite>` by name or position for this reason: it collects every `<testcase>` in the whole
document and keeps only the ones whose `classname` starts with the exact module prefix
`f"{target}."`, so it produces the same attributed count regardless of how many empty sibling
`<testsuite>` elements a given toolchain happens to interleave alongside the real one.

THE SWIFT TESTING LEG'S OWN ZERO-CASE VACUITY TRAP, AND HOW IT IS CLOSED: seven of the eight
pinned targets are pinned at zero Swift Testing cases, and a filtered run that legitimately finds
none produces a well-formed report indistinguishable, BY ITS OWN OUTPUT ALONE, from a filtered
run whose target name was mistyped, renamed, or never got built -- measured 2026-09-17 with a
deliberately bogus `--filter` target: the resulting xunit document has exactly the same
all-testsuites-empty shape as a real target with genuinely zero Swift Testing cases. Comparing
the observed count only against the pinned expectation cannot close this, because a target
permanently mis-pinned at zero and a tool permanently (and silently) unable to find that target's
module would agree with each other forever. This gate closes it by requiring independent
evidence that the target's module was actually part of this build before it will accept an
observed zero: whenever a target's Swift Testing count comes back zero, it additionally checks
that the target appears in the same `swift test list --skip-build` listing the XCTest
combined-bundle mode already uses (cached and shared, not a second invocation) -- every one of
the eight pinned targets has a substantial XCTest suite of its own, so a target that is really
part of the build always appears there regardless of how many (if any) Swift Testing cases it
has. A zero that cannot be corroborated this way is REFUSED, not accepted as a pass.

Running the bundle by name/selector, not `swift test --package-path spike --filter '<Target>\\.'`,
still matters because a `--filter` run walks EVERY `.xctest` bundle in the package: a non-matching
bundle prints its own `Executed 0 tests, ...` summary alongside the target's real one, and a parser
that is not careful about which summary line belongs to which bundle will silently read the wrong
one. `parse_summary` anchors on an exact `Test Suite '<name>'` marker -- `'<target>.xctest'` for mode
1, or `'Selected tests'` for mode 2 -- to remove that ambiguity by construction.

WHAT THIS GATE PROVES, AND WHAT IT DOES NOT: a passing run proves the named projected suites still
RUN and still match a recorded XCTest test/failure count on the host that ran them. It does NOT prove
that recorded baseline is CORRECT. Every target is currently pinned at 0 failures; that was not
always true, and it does not need to stay true for this gate to still be doing its job. The gate's
only job is to notice when a target's test count or failure count moves, in either direction, so the
move gets a conscious decision (via `--update-expectations`) instead of silently drifting.

ANTI-VACUITY RULES -- every one of these exists because this project has been burned by exactly this
shape of false-green before, and every one of them is a hard FAIL (non-zero exit), never a skip:

- No XCTest summary line found for a target (crash, truncation, unexpected output shape) -> FAIL,
  reported as REFUSED. A run that produced no readable result must never be mistaken for a pass.
- Executed test count is zero -> FAIL. `xcrun xctest` exits 0 on a bundle that finds nothing to run,
  so a silently-broken test discovery would otherwise look identical to success.
- Executed test count does not equal the pinned expectation, in EITHER direction -> FAIL. Lower
  catches "green by deletion" (quietly removing a failing test); higher catches an unrecorded
  addition and forces the baseline to be updated on purpose, with review, via --update-expectations,
  rather than by drifting.
- Failure count does not equal the pinned expectation, in EITHER direction -> FAIL. The pinned number
  is whatever was last measured and consciously recorded, whether that is zero or not; a target that
  silently starts passing must ALSO trip this gate, so the improvement gets recorded on purpose
  instead of going unnoticed, the same as a regression would.
- Zero class selectors resolved for a target that needs combined mode -> FAIL, reported as REFUSED,
  and `xcrun xctest` is never invoked for it: an empty `-XCTest` selector list is not the same
  request as "run nothing" to XCTest on every toolchain -- it can be read as "run everything" -- so
  this gate refuses before running it rather than guess which behaviour it would get.
- More than one combined `*PackageTests.xctest` bundle with a runnable executable exists under
  `spike/.build` -> FAIL, reported as REFUSED: this gate does not guess which one to run.
- A per-target bundle with a runnable executable AND a combined bundle with a runnable executable
  both exist under `spike/.build` for the same run -> FAIL, reported as REFUSED, naming both paths
  and the active toolchain, with NEITHER `xcrun xctest` nor `swift test list` invoked: this is a
  build tree touched by two different Swift toolchains (see the measured layouts above), and one of
  the two bundles is stale -- this gate does not guess which.
- A package build that reports success but leaves no runnable bundle for a target in EITHER mode is
  its own REFUSED, naming both the per-target and combined-bundle absence in one detail. A package
  build failure, or an unavailable `swift`/`xcrun` toolchain, are their own FAILs, reported as
  REFUSED. There is deliberately no "toolchain unavailable, skip and return success" branch anywhere
  in this file.

THE SAME RULES APPLY TO THE SWIFT TESTING LEG, WITH TWO LEG-SPECIFIC ADDITIONS:

- No `--xunit-output` artifact exists after the `swift test` invocation, or it exists but is not
  parseable XML, or it parses but contains zero `<testsuite>` elements anywhere -> FAIL, reported as
  REFUSED: exactly the "a run that produced no readable result must never be mistaken for a pass"
  rule above, applied to this leg's own artifact shape.
- A target whose Swift Testing count comes back zero, and which does NOT also appear in the shared
  `swift test list` listing -> FAIL, reported as REFUSED: see "THE SWIFT TESTING LEG'S OWN ZERO-CASE
  VACUITY TRAP" above. This is what makes the zero-executed rule meaningful for the seven targets
  legitimately pinned at zero Swift Testing cases, instead of exempting them from it.

This gate runs in CI on every push and pull request, as its own job. It is deliberately kept out of
the fast `scripts/tests` Python suite (`python3 -m unittest discover -s scripts/tests`): exporting
plus building and running eight targets takes tens of minutes, which would make that fast suite
unusable for quick local iteration. To run it locally against a checkout, from the repository root:
`python3 scripts/run_projected_test_suites.py --projection .`. Run it after any change to
`spike/Tests`, `spike/Sources`, or a `public/sanitized-projection/` override that a projected test
depends on, even if you also rely on CI to catch it.

Exit codes: 0 = every selected target matched its pinned expectation, XCTest leg and Swift Testing
leg alike (test count and failure count); 1 = at least one target mismatched or was refused, in
either leg; 2 = setup error (export failed, the projection path or expectations file is missing, or
the `swift`/`xcrun` toolchain is not on PATH).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_EXPECTATIONS = REPO_ROOT / "scripts" / "projected_test_expectations.json"

# Matches the counts line XCTest prints immediately after a `Test Suite '<name>' passed|failed at
# <date>.` line, e.g. "Executed 721 tests, with 2 tests skipped and 0 failures (0 unexpected) in
# 21.170 (21.351) seconds". The "with N test(s) skipped and" clause is optional because older/plain
# XCTest output omits it entirely when there is nothing to report there; making it optional handles
# both shapes with one pattern instead of guessing which one a given toolchain will print.
EXECUTED_LINE = re.compile(
    r"Executed (?P<tests>\d+) tests?, with (?:(?P<skipped>\d+) tests? skipped and )?"
    r"(?P<failures>\d+) failures?"
)

OK = "OK"
MISMATCH = "MISMATCH"
REFUSED = "REFUSED"

MODE_PER_TARGET = "per-target bundle"
MODE_COMBINED = "combined bundle + class selectors"


@dataclasses.dataclass
class TargetResult:
    target: str
    expected_tests: int
    expected_skipped: int
    expected_failures: int
    actual_tests: int | None
    actual_skipped: int | None
    actual_failures: int | None
    verdict: str
    detail: str


@dataclasses.dataclass
class SwiftTestingResult:
    """One target's Swift Testing (`@Test`) leg result -- entirely separate from `TargetResult`
    (the XCTest leg) because the two legs run independent tools (`swift test --disable-xctest`
    vs. `xcrun xctest`) against independent artifacts, and neither leg's bundle resolution or
    refusal affects the other's."""

    target: str
    expected_tests: int
    expected_failures: int
    actual_tests: int | None
    actual_failures: int | None
    verdict: str
    detail: str


def run(command: list[str], cwd: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


def parse_summary(output: str, target: str, anchor: str | None = None) -> tuple[int, int, int] | None:
    """Return (tests, skipped, failures) parsed from one summary block, or None if no usable
    summary was found.

    `anchor` selects which `Test Suite '<name>'` marker to read: the default (None) anchors on
    `'<target>.xctest'`, the per-target bundle's own bundle-level summary line, the same behaviour
    this function has always had. Passing `anchor="Selected tests"` instead anchors on the
    `Test Suite 'Selected tests'` block `xcrun xctest -XCTest <selectors> <bundle>` prints last,
    after the individual class summaries and the whole-bundle
    `Test Suite '<combined-bundle-name>.xctest'` summary -- reading any of those other blocks
    instead would silently report the WRONG target's counts (or the whole combined bundle's
    counts, covering every target at once) rather than the selected classes' own counts.
    """
    marker = f"Test Suite '{anchor}'" if anchor is not None else f"Test Suite '{target}.xctest'"
    lines = output.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith(marker):
            continue
        if " passed at " not in stripped and " failed at " not in stripped:
            continue
        # The counts line is normally the very next line. Tolerate one blank line beyond that (seen
        # to vary slightly across toolchain versions) but no further -- if nothing usable is found
        # within that narrow window, this is treated exactly like a missing summary.
        for candidate in lines[index + 1 : index + 3]:
            candidate_stripped = candidate.strip()
            if not candidate_stripped:
                continue
            match = EXECUTED_LINE.search(candidate_stripped)
            if match is None:
                break
            tests = int(match.group("tests"))
            skipped = int(match.group("skipped")) if match.group("skipped") else 0
            failures = int(match.group("failures"))
            return tests, skipped, failures
        return None
    return None


def class_selectors(listing: str, target: str) -> list[str]:
    """Derive the sorted, deduped `Module.Class` selector list for one target from a
    `swift test list --skip-build` listing. Pure function; no subprocess, no filesystem.

    Each XCTest line in that listing has shape `Module.Class/method` (occasionally with a
    trailing `()` on the method). This keeps only lines that contain a '/', splits on the FIRST
    '/', and requires the part before it to start with the exact module prefix `f"{target}."` --
    an exact prefix match with the trailing dot, so `ServingCoreTests.` never matches a sibling
    module like `ServingCoreTestsExtra.x/y` just because it shares a string prefix. Matches are
    deduped and sorted for a stable, reproducible selector list and log line. Returns an empty
    list, never raises, when the target has no matching lines (an unknown target, or one whose
    listing lines were all filtered out) -- the caller is responsible for treating an empty list
    as its own REFUSED case rather than passing it to `xcrun xctest` as an empty selector.
    """
    prefix = f"{target}."
    selectors: set[str] = set()
    for raw_line in listing.splitlines():
        line = raw_line.strip()
        if "/" not in line:
            continue
        selector = line.split("/", 1)[0].strip().rstrip("()").strip()
        if selector.startswith(prefix):
            selectors.add(selector)
    return sorted(selectors)


def swift_testing_filter_pattern(target: str) -> str:
    """The anchored `swift test --filter` regular expression for exactly one target's Swift
    Testing cases. Anchored with `^` and an exact module-prefix trailing dot
    (`re.escape(target) + r"\\."`), the same rule `class_selectors` uses for XCTest selectors, so
    an unanchored pattern cannot accidentally select a sibling module that merely shares a string
    prefix. Running this filter, rather than an unfiltered `swift test`, is also what keeps this
    gate away from the MLX/Metal targets it does not cover -- see the module docstring."""
    return f"^{re.escape(target)}\\."


def parse_swift_testing_xunit(xml_text: str, target: str) -> tuple[int, int] | None:
    """Parse one `swift test --xunit-output` artifact and return `(tests, failures)`
    attributable to exactly one target, or None if the artifact cannot be trusted at all.

    Deliberately does not pick a `<testsuite>` element by name or position -- see the module
    docstring for why every `<testsuite>` in this artifact is named identically
    (`"TestResults"`) on every measured toolchain, and why Swift 6.4 was measured to emit one
    such element per bundle walked while Swift 6.3.3 emits exactly one for the whole document.
    Instead this collects every `<testcase>` anywhere in the document and keeps only the ones
    whose `classname` attribute starts with the exact module prefix `f"{target}."` (the same
    trailing-dot exact match `class_selectors` uses for XCTest selectors), so the attributed
    count does not depend on how many empty sibling `<testsuite>` elements happen to be
    interleaved alongside the real one. A matched `<testcase>` counts as a failure if it has a
    `<failure>` or `<error>` child element, the two failure shapes Swift Testing's xUnit writer
    was measured to use.

    Returns None -- never `(0, 0)` -- when the artifact itself is not readable: invalid XML
    (`ET.ParseError`, e.g. a process that crashed mid-write and left a truncated document), or a
    well-formed document with zero `<testsuite>` elements anywhere. The latter was measured to
    not happen even for a filter that matches nothing at all -- every walked bundle still reports
    its own empty `<testsuite>` -- so a document with none of them at all means `swift test` never
    got far enough to report anything, not that it legitimately found nothing.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    if not root.findall(".//testsuite"):
        return None
    prefix = f"{target}."
    tests = 0
    failures = 0
    for testcase in root.findall(".//testcase"):
        classname = testcase.get("classname", "")
        if not classname.startswith(prefix):
            continue
        tests += 1
        if testcase.find("failure") is not None or testcase.find("error") is not None:
            failures += 1
    return tests, failures


def swift_testing_target_is_known(listing: str, target: str) -> bool:
    """Whether `target` appears anywhere in a `swift test list --skip-build` listing, as an exact
    module-prefix match (the same trailing-dot rule as `class_selectors`).

    Closes one specific vacuity trap named in the module docstring: a Swift Testing filtered run
    that finds zero matching cases produces a well-formed report indistinguishable, by its own
    output alone, from one whose target name was mistyped, renamed, or never got built. Every
    pinned target has a substantial XCTest suite of its own, so a target that is genuinely part
    of the build always appears in this listing regardless of how many (if any) Swift Testing
    cases it has. Reuses the listing `swift_test_listing` already fetched and cached for the
    XCTest combined-bundle mode rather than invoking `swift test list` a second time.
    """
    prefix = f"{target}."
    return any(line.strip().startswith(prefix) for line in listing.splitlines())


def swift_test_listing(destination: pathlib.Path, cache: dict) -> tuple[str | None, str | None]:
    """Run `swift test list --skip-build --package-path spike` at most once per gate invocation,
    caching the result (and its raw stdout/stderr, for per-target logging) in `cache`, a plain
    dict the caller owns and shares across every target evaluated in one `run_gate` call. Every
    target that needs combined-mode selectors would otherwise repeat an identical, non-trivial
    subprocess call for no new information -- this makes that call idempotent for the run.

    Returns (listing_text, error_detail); exactly one of the two is not None.
    """
    if "done" not in cache:
        result = run(
            ["swift", "test", "list", "--skip-build", "--package-path", "spike"], cwd=destination
        )
        cache["done"] = True
        cache["listing_stdout"] = result.stdout
        cache["listing_stderr"] = result.stderr
        if result.returncode != 0:
            cache["listing"] = None
            cache["error"] = (
                f"`swift test list --skip-build --package-path spike` failed (exit "
                f"{result.returncode})"
            )
        else:
            cache["listing"] = result.stdout
            cache["error"] = None
    return cache.get("listing"), cache.get("error")


def primary_bundle_path(destination: pathlib.Path, target: str) -> pathlib.Path:
    """The bundle path a per-target `swift build --package-path spike --target <Target>` build has
    been observed to produce on a Swift 6.4 toolchain. Factored out of `resolve_bundle` so a
    refusal message can name the exact path this gate looked for, even when nothing was found
    there. This gate no longer runs that build command itself (see the module docstring); this is
    only where it checks for a bundle that may already be sitting there."""
    return destination / "spike" / ".build" / "out" / "Products" / "Debug" / f"{target}.xctest"


def resolve_bundle(destination: pathlib.Path, target: str) -> pathlib.Path | None:
    """Locate a built per-target `.xctest` bundle for one target, if one already exists.

    The primary path is where a per-target `swift build --package-path spike --target <Target>`
    build has been observed to place the bundle. The recursive fallback exists because that exact
    layout is a build-system implementation detail this gate does not control; it should not
    hard-refuse on that detail alone if a different toolchain places the bundle somewhere else
    under the same `.build` tree. Returning None is not itself a refusal -- the caller falls back
    to combined-bundle mode next.
    """
    primary = primary_bundle_path(destination, target)
    if primary.exists():
        return primary
    build_root = destination / "spike" / ".build"
    if not build_root.exists():
        return None
    matches = sorted(build_root.rglob(f"{target}.xctest"))
    return matches[0] if matches else None


def bundle_executable_path(bundle: pathlib.Path, target: str) -> pathlib.Path:
    """The path an already-resolved per-target `.xctest` bundle's runnable executable is expected
    at."""
    return bundle / "Contents" / "MacOS" / target


def bundle_has_executable(bundle: pathlib.Path, target: str) -> bool:
    """Whether an already-resolved per-target `.xctest` bundle actually contains a runnable
    executable.

    A bundle directory existing is not sufficient on its own: an incremental build tree can leave
    a skeleton `<Target>.xctest/Contents/MacOS/` directory with nothing inside it, or with
    `Contents/MacOS/` missing entirely. When that happens this gate falls back to combined-bundle
    mode rather than refusing immediately -- see `evaluate_target`.
    """
    return bundle_executable_path(bundle, target).is_file()


def combined_bundle_candidates(destination: pathlib.Path) -> list[pathlib.Path]:
    """Every `*PackageTests.xctest` bundle directory under `spike/.build`, sorted for a stable
    order. `swift build --package-path spike --build-tests` was measured (2026-09-17, Swift
    6.3.3) to produce exactly one such bundle,
    `arm64-apple-macosx/debug/fast-mlx-spikePackageTests.xctest`, but this globs the whole
    `.build` tree rather than hard-coding that path, the same reasoning as `resolve_bundle`'s
    fallback: the exact location is a build-system detail this gate does not control."""
    build_root = destination / "spike" / ".build"
    if not build_root.exists():
        return []
    return sorted(build_root.rglob("*PackageTests.xctest"))


def resolve_combined_bundle(
    destination: pathlib.Path,
) -> tuple[pathlib.Path | None, list[pathlib.Path]]:
    """Locate the one combined test bundle to run selectors against.

    Returns (bundle_or_None, candidates_with_a_runnable_executable). A bundle is only returned
    when EXACTLY one candidate has a runnable `Contents/MacOS/<bundle-stem>` executable -- zero
    candidates is the ordinary "not built yet" case, and more than one is refused by the caller
    rather than guessed at, since silently picking one could run the wrong package's tests.
    """
    candidates = combined_bundle_candidates(destination)
    with_executable = [
        candidate
        for candidate in candidates
        if (candidate / "Contents" / "MacOS" / candidate.stem).is_file()
    ]
    if len(with_executable) == 1:
        return with_executable[0], with_executable
    return None, with_executable


_SWIFT_TOOLCHAIN_VERSION: str | None = None


def swift_toolchain_version() -> str:
    """Best-effort first line of `swift --version`, cached for the life of this process.

    Purely diagnostic: surfaced inside toolchain-caused REFUSED details so a reader does not have
    to separately re-run `swift --version` on the host that produced the failure. Never raises and
    never affects a verdict -- if this cannot be obtained for any reason, the refusal still fires,
    just without this extra detail.
    """
    global _SWIFT_TOOLCHAIN_VERSION
    if _SWIFT_TOOLCHAIN_VERSION is None:
        try:
            result = subprocess.run(
                ["swift", "--version"], capture_output=True, text=True, check=False
            )
            combined = (result.stdout or "") + (result.stderr or "")
            first_line = combined.splitlines()[0].strip() if combined.strip() else "unknown"
        except OSError:
            first_line = "unknown"
        _SWIFT_TOOLCHAIN_VERSION = first_line
    return _SWIFT_TOOLCHAIN_VERSION


def _classify(
    parsed: tuple[int, int, int],
    expected_tests: int,
    expected_failures: int,
    finish,
    mode: str,
):
    """Shared OK/MISMATCH classification for both run modes, so the actual-vs-expected comparison
    lives in exactly one place regardless of which bundle produced the counts. `mode` is recorded
    in the detail either way, so a reader of the table/log can see which path ran without a
    separate column."""
    actual_tests, _actual_skipped, actual_failures = parsed
    if actual_tests != expected_tests or actual_failures != expected_failures:
        detail = (
            f"tests {actual_tests} (expected {expected_tests}), "
            f"failures {actual_failures} (expected {expected_failures}) [{mode}]"
        )
        return finish(parsed, MISMATCH, detail)
    return finish(parsed, OK, f"[{mode}]")


def evaluate_target(
    destination: pathlib.Path,
    target: str,
    expected: dict,
    log_dir: pathlib.Path | None,
    listing_cache: dict,
) -> TargetResult:
    """Resolve and run one target's tests, assuming the package-wide `--build-tests` build (done
    once, by the caller, in `run_gate`) already succeeded. See the module docstring for the two
    resolution modes this tries, in order."""
    expected_tests = int(expected["tests"])
    expected_skipped = int(expected.get("skipped", 0))
    expected_failures = int(expected["failures"])

    log_parts: list[str] = []

    def finish(
        actual: tuple[int, int, int] | None, verdict: str, detail: str
    ) -> TargetResult:
        if log_dir is not None:
            # Never truncate: failures attach at the tail of a raw XCTest log, and a truncated log
            # is worse than no log because it looks complete.
            (log_dir / f"{target}.log").write_text("".join(log_parts), encoding="utf-8")
        actual_tests, actual_skipped, actual_failures = actual if actual else (None, None, None)
        return TargetResult(
            target=target,
            expected_tests=expected_tests,
            expected_skipped=expected_skipped,
            expected_failures=expected_failures,
            actual_tests=actual_tests,
            actual_skipped=actual_skipped,
            actual_failures=actual_failures,
            verdict=verdict,
            detail=detail,
        )

    # Mode 1: a per-target bundle already sitting under spike/.build, with a runnable executable.
    per_target_bundle = resolve_bundle(destination, target)
    per_target_note: str | None = None
    if per_target_bundle is not None:
        if bundle_has_executable(per_target_bundle, target):
            # Before trusting this per-target bundle, check whether a combined bundle with a
            # runnable executable ALSO exists under spike/.build. The two shapes were measured to
            # come from different Swift toolchains (see the module docstring): Swift 6.4 leaves
            # only per-target bundles, Swift 6.3.3 leaves only the combined one, in different
            # directories. A tree built once by each toolchain -- e.g. from switching Xcode
            # versions locally without cleaning `.build` -- can therefore hold a stale bundle of
            # one kind sitting next to a fresh one of the other, and nothing here can tell which
            # is which. Refuse rather than silently preferring mode 1, and do not invoke
            # `xcrun xctest` (either mode) or `swift test list` while refusing.
            _mixed_combined_bundle, mixed_candidates = resolve_combined_bundle(destination)
            if mixed_candidates:
                names = ", ".join(str(candidate) for candidate in mixed_candidates)
                return finish(
                    None,
                    REFUSED,
                    "build tree holds both a per-target test bundle "
                    f"({per_target_bundle}) and a combined test bundle ({names}) -- this is a "
                    "tree built by two different Swift toolchains, so one of the two is stale, "
                    "and this gate refuses to guess which; remove spike/.build and re-run -- "
                    f"active toolchain: {swift_toolchain_version()}",
                )
            xctest = run(["xcrun", "xctest", str(per_target_bundle)], cwd=destination)
            log_parts.append(f"$ xcrun xctest {per_target_bundle}\n")
            log_parts.append(xctest.stdout)
            log_parts.append(xctest.stderr)
            # Deliberately not gating on xctest.returncode: XCTest exits non-zero whenever any
            # test fails, an expected, already-recorded shape for a target like HarnessCoreTests.
            # The only thing that makes a run unreadable is the absence of a parseable summary
            # line, checked next.
            parsed = parse_summary(xctest.stdout + "\n" + xctest.stderr, target)
            if parsed is None:
                return finish(
                    None,
                    REFUSED,
                    f"no XCTest summary line found for this target (crash or truncated run) "
                    f"[{MODE_PER_TARGET}]",
                )
            return _classify(parsed, expected_tests, expected_failures, finish, MODE_PER_TARGET)
        expected_executable = bundle_executable_path(per_target_bundle, target)
        per_target_note = (
            f"{target}.xctest bundle exists at {per_target_bundle} but has no executable at "
            f"{expected_executable} (a skeleton bundle from an incremental build tree) -- falling "
            "back to the combined bundle"
        )

    # Mode 2: the combined bundle `--build-tests` produced, filtered to this target's classes.
    combined_bundle, candidates_with_executable = resolve_combined_bundle(destination)
    if combined_bundle is None:
        if len(candidates_with_executable) > 1:
            names = ", ".join(str(candidate) for candidate in candidates_with_executable)
            return finish(
                None,
                REFUSED,
                "more than one combined *PackageTests.xctest bundle with a runnable executable "
                f"found under spike/.build, refusing to guess which one to run: {names}",
            )
        reasons = [per_target_note] if per_target_note else [
            f"no {target}.xctest bundle exists anywhere under spike/.build"
        ]
        reasons.append(
            "no combined *PackageTests.xctest bundle with a runnable executable exists under "
            "spike/.build after `swift build --package-path spike --build-tests`"
        )
        detail = (
            "; ".join(reasons)
            + f" -- active toolchain: {swift_toolchain_version()}"
        )
        return finish(None, REFUSED, detail)

    listing, listing_error = swift_test_listing(destination, listing_cache)
    log_parts.append("$ swift test list --skip-build --package-path spike\n")
    log_parts.append(listing_cache.get("listing_stdout", ""))
    log_parts.append(listing_cache.get("listing_stderr", ""))
    if listing_error is not None:
        return finish(
            None,
            REFUSED,
            f"could not derive class selectors for combined-bundle mode: {listing_error}",
        )

    selectors = class_selectors(listing or "", target)
    if not selectors:
        return finish(
            None,
            REFUSED,
            f"`swift test list` produced zero class selectors for target {target} -- refusing to "
            "run `xcrun xctest -XCTest` with an empty selector list rather than guess whether an "
            "empty selector means \"run nothing\" or \"run everything\" on this toolchain",
        )

    selector_arg = ",".join(selectors)
    xctest = run(
        ["xcrun", "xctest", "-XCTest", selector_arg, str(combined_bundle)], cwd=destination
    )
    log_parts.append(f"$ xcrun xctest -XCTest {selector_arg} {combined_bundle}\n")
    log_parts.append(xctest.stdout)
    log_parts.append(xctest.stderr)

    parsed = parse_summary(xctest.stdout + "\n" + xctest.stderr, target, anchor="Selected tests")
    if parsed is None:
        return finish(
            None,
            REFUSED,
            f"no XCTest summary line found for this target (crash or truncated run) "
            f"[{MODE_COMBINED}]",
        )
    return _classify(parsed, expected_tests, expected_failures, finish, MODE_COMBINED)


def evaluate_swift_testing_target(
    destination: pathlib.Path,
    target: str,
    expected: dict,
    log_dir: pathlib.Path | None,
    listing_cache: dict,
    xunit_dir: pathlib.Path,
) -> SwiftTestingResult:
    """Run and classify one target's Swift Testing leg, independent of the XCTest leg's own
    bundle resolution or verdict for the same target -- see the module docstring for why this
    always runs `swift test --filter <anchored pattern> --disable-xctest`, never `xcrun xctest`,
    and never an unfiltered `swift test`.

    `xunit_dir` is a scratch directory the caller owns (created once per `run_gate` call, shared
    across every target so each gets its own `<target>.xml` inside it) -- passed in rather than
    created here so a test can control exactly what does or does not land at that path without
    a real `swift` subprocess.
    """
    expected_tests = int(expected["tests"])
    expected_failures = int(expected["failures"])
    xunit_path = xunit_dir / f"{target}.xml"
    log_parts: list[str] = []

    def finish(
        actual: tuple[int, int] | None, verdict: str, detail: str
    ) -> SwiftTestingResult:
        if log_dir is not None:
            (log_dir / f"{target}.swift-testing.log").write_text(
                "".join(log_parts), encoding="utf-8"
            )
        actual_tests, actual_failures = actual if actual else (None, None)
        return SwiftTestingResult(
            target=target,
            expected_tests=expected_tests,
            expected_failures=expected_failures,
            actual_tests=actual_tests,
            actual_failures=actual_failures,
            verdict=verdict,
            detail=detail,
        )

    command = [
        "swift",
        "test",
        "--package-path",
        "spike",
        "--skip-build",
        "--disable-xctest",
        "--filter",
        swift_testing_filter_pattern(target),
        "--xunit-output",
        str(xunit_path),
    ]
    result = run(command, cwd=destination)
    log_parts.append(f"$ {' '.join(command)}\n")
    log_parts.append(result.stdout)
    log_parts.append(result.stderr)

    # Deliberately not gating on result.returncode: `swift test` exits non-zero whenever any
    # selected case fails, an expected, already-recorded shape -- exactly the same reasoning the
    # XCTest leg above uses. The only thing that makes a run unreadable is an absent or
    # unparseable artifact, checked next.
    if not xunit_path.exists():
        return finish(
            None,
            REFUSED,
            "no Swift Testing xunit artifact was produced by `swift test` (exit "
            f"{result.returncode}) [swift testing]",
        )
    xml_text = xunit_path.read_text(encoding="utf-8")
    parsed = parse_swift_testing_xunit(xml_text, target)
    if parsed is None:
        return finish(
            None,
            REFUSED,
            "Swift Testing xunit artifact was malformed or contained no <testsuite> element at "
            "all (crash or truncated run) [swift testing]",
        )
    actual_tests, actual_failures = parsed

    if actual_tests == 0:
        # Close the zero-case vacuity trap named in the module docstring: a legitimate zero and
        # a mistyped/unbuilt target look identical from this artifact alone.
        listing, listing_error = swift_test_listing(destination, listing_cache)
        if listing_error is not None:
            return finish(
                None,
                REFUSED,
                "Swift Testing reported zero cases and this target's existence could not be "
                f"corroborated: {listing_error} [swift testing]",
            )
        if not swift_testing_target_is_known(listing or "", target):
            return finish(
                None,
                REFUSED,
                f"Swift Testing reported zero cases for {target}, and {target} does not "
                "otherwise appear in `swift test list` -- refusing to treat this as a "
                "legitimate zero rather than a typo'd or unbuilt target [swift testing]",
            )

    if actual_tests != expected_tests or actual_failures != expected_failures:
        detail = (
            f"tests {actual_tests} (expected {expected_tests}), "
            f"failures {actual_failures} (expected {expected_failures}) [swift testing]"
        )
        return finish((actual_tests, actual_failures), MISMATCH, detail)
    return finish((actual_tests, actual_failures), OK, "[swift testing]")


def print_table(results: list[TargetResult]) -> None:
    columns = (
        f"{'TARGET':<28} {'EXP TESTS':>9} {'ACT TESTS':>9} {'EXP FAIL':>8} {'ACT FAIL':>8} "
        f"{'SKIP act/exp':>14}  VERDICT"
    )
    print(columns)
    print("-" * len(columns))
    for result in results:
        act_tests = "-" if result.actual_tests is None else str(result.actual_tests)
        act_fail = "-" if result.actual_failures is None else str(result.actual_failures)
        act_skip = "-" if result.actual_skipped is None else str(result.actual_skipped)
        skip_field = f"{act_skip}/{result.expected_skipped}"
        line = (
            f"{result.target:<28} {result.expected_tests:>9} {act_tests:>9} "
            f"{result.expected_failures:>8} {act_fail:>8} {skip_field:>14}  {result.verdict}"
        )
        if result.detail:
            line += f"  ({result.detail})"
        print(line)

    ok_count = sum(1 for r in results if r.verdict == OK)
    mismatch_count = sum(1 for r in results if r.verdict == MISMATCH)
    refused_count = sum(1 for r in results if r.verdict == REFUSED)
    print(
        f"\n{ok_count}/{len(results)} target(s) OK, {mismatch_count} MISMATCH, "
        f"{refused_count} REFUSED"
    )


def print_swift_testing_table(results: list[SwiftTestingResult]) -> None:
    """Same table style as `print_table`, for the Swift Testing leg -- no skip column, since
    that leg's pin has no skipped-count concept."""
    columns = f"{'TARGET':<28} {'EXP TESTS':>9} {'ACT TESTS':>9} {'EXP FAIL':>8} {'ACT FAIL':>8}  VERDICT"
    print("\nSwift Testing (`@Test`) leg:")
    print(columns)
    print("-" * len(columns))
    for result in results:
        act_tests = "-" if result.actual_tests is None else str(result.actual_tests)
        act_fail = "-" if result.actual_failures is None else str(result.actual_failures)
        line = (
            f"{result.target:<28} {result.expected_tests:>9} {act_tests:>9} "
            f"{result.expected_failures:>8} {act_fail:>8}  {result.verdict}"
        )
        if result.detail:
            line += f"  ({result.detail})"
        print(line)

    ok_count = sum(1 for r in results if r.verdict == OK)
    mismatch_count = sum(1 for r in results if r.verdict == MISMATCH)
    refused_count = sum(1 for r in results if r.verdict == REFUSED)
    print(
        f"\n{ok_count}/{len(results)} target(s) OK, {mismatch_count} MISMATCH, "
        f"{refused_count} REFUSED"
    )


def update_expectations(
    expectations_path: pathlib.Path,
    expectations_data: dict,
    results: list[TargetResult],
    swift_testing_results: list[SwiftTestingResult] | None = None,
) -> None:
    print("\n" + "!" * 78, file=sys.stderr)
    print(
        "WARNING: --update-expectations is about to rewrite the pinned baseline in\n"
        f"{expectations_path} from THIS RUN's observed counts. This RECORDS whatever is\n"
        "currently true; it does not validate that the new numbers are correct. A dropped\n"
        "count can mean a test was fixed, or that a failing test was quietly deleted -- this\n"
        "flag cannot tell the difference. Diff this file against the previous commit and\n"
        "review every changed number by hand before committing.",
        file=sys.stderr,
    )
    print("!" * 78 + "\n", file=sys.stderr)

    not_updated: list[str] = []
    for result in results:
        if result.actual_tests is None:
            not_updated.append(result.target)
            continue
        entry = expectations_data["targets"][result.target]
        entry["tests"] = result.actual_tests
        entry["skipped"] = result.actual_skipped
        entry["failures"] = result.actual_failures

    if not_updated:
        print(
            "NOTE: left unchanged (REFUSED this run, no observed counts to record): "
            + ", ".join(not_updated),
            file=sys.stderr,
        )

    if swift_testing_results is not None:
        swift_testing_not_updated = update_swift_testing_expectations(
            expectations_data, swift_testing_results
        )
        if swift_testing_not_updated:
            print(
                "NOTE: swift_testing left unchanged (REFUSED this run, no observed counts to "
                "record): " + ", ".join(swift_testing_not_updated),
                file=sys.stderr,
            )

    expectations_path.write_text(json.dumps(expectations_data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote updated expectations to {expectations_path}", file=sys.stderr)


def update_swift_testing_expectations(
    expectations_data: dict, results: list[SwiftTestingResult]
) -> list[str]:
    """Update the `swift_testing` block of `expectations_data["targets"][*]` in place from this
    run's observed counts. Returns the list of targets left unchanged (REFUSED this run). Writing
    the file and printing the shared warning stay in `update_expectations`/`run_gate`, which
    calls this before writing -- both legs' updates land in one write, one warning, one diff."""
    not_updated: list[str] = []
    for result in results:
        if result.actual_tests is None:
            not_updated.append(result.target)
            continue
        entry = expectations_data["targets"][result.target]
        entry["swift_testing"] = {
            "tests": result.actual_tests,
            "failures": result.actual_failures,
        }
    return not_updated


def run_gate(
    destination: pathlib.Path,
    selected: list[str],
    expectations_data: dict,
    log_dir: pathlib.Path | None,
    update: bool,
    expectations_path: pathlib.Path,
) -> int:
    all_targets = expectations_data["targets"]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    # Build the whole package's tests exactly once, before evaluating any target -- see the module
    # docstring for why this replaced a per-target `swift build --target` step.
    package_build = run(
        ["swift", "build", "--package-path", "spike", "--build-tests"], cwd=destination
    )
    if log_dir is not None:
        (log_dir / "_package-build.log").write_text(
            "$ swift build --package-path spike --build-tests\n"
            + package_build.stdout
            + package_build.stderr,
            encoding="utf-8",
        )

    if package_build.returncode != 0:
        detail = (
            "package test build failed before any target could be evaluated (`swift build "
            f"--package-path spike --build-tests` exit {package_build.returncode})"
        )
        if log_dir is not None:
            detail += f" -- see {log_dir / '_package-build.log'}"
        results = [
            TargetResult(
                target=target,
                expected_tests=int(all_targets[target]["tests"]),
                expected_skipped=int(all_targets[target].get("skipped", 0)),
                expected_failures=int(all_targets[target]["failures"]),
                actual_tests=None,
                actual_skipped=None,
                actual_failures=None,
                verdict=REFUSED,
                detail=detail,
            )
            for target in selected
        ]
        swift_testing_results = [
            SwiftTestingResult(
                target=target,
                expected_tests=int(all_targets[target]["swift_testing"]["tests"]),
                expected_failures=int(all_targets[target]["swift_testing"]["failures"]),
                actual_tests=None,
                actual_failures=None,
                verdict=REFUSED,
                detail=detail + " [swift testing]",
            )
            for target in selected
        ]
        print_table(results)
        print_swift_testing_table(swift_testing_results)
        if update:
            update_expectations(expectations_path, expectations_data, results, swift_testing_results)
        return 1

    listing_cache: dict = {}
    results = [
        evaluate_target(destination, target, all_targets[target], log_dir, listing_cache)
        for target in selected
    ]
    print_table(results)

    with tempfile.TemporaryDirectory(prefix="swift-testing-xunit-") as swift_testing_scratch:
        swift_testing_xunit_dir = pathlib.Path(swift_testing_scratch)
        swift_testing_results = [
            evaluate_swift_testing_target(
                destination,
                target,
                all_targets[target]["swift_testing"],
                log_dir,
                listing_cache,
                swift_testing_xunit_dir,
            )
            for target in selected
        ]
    print_swift_testing_table(swift_testing_results)

    if update:
        update_expectations(expectations_path, expectations_data, results, swift_testing_results)

    return (
        0
        if all(result.verdict == OK for result in results)
        and all(result.verdict == OK for result in swift_testing_results)
        else 1
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--projection",
        type=pathlib.Path,
        default=None,
        help="Use an existing exported projection tree (skip exporting a new one). If omitted, "
        "one is exported via scripts/export_public_repository.py --development-projection into "
        "a temporary directory that is removed afterwards.",
    )
    parser.add_argument(
        "--expectations",
        type=pathlib.Path,
        default=DEFAULT_EXPECTATIONS,
        help=f"Path to the pinned baseline JSON. Default: {DEFAULT_EXPECTATIONS}",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        help="Restrict the run to this target (repeatable). Default: every target listed in "
        "--expectations.",
    )
    parser.add_argument(
        "--log-dir",
        type=pathlib.Path,
        default=None,
        help="Write each target's full raw build+test output to <log-dir>/<Target>.log, plus the "
        "shared package build output to <log-dir>/_package-build.log. Logs are never truncated -- "
        "a failure's detail is usually at the tail.",
    )
    parser.add_argument(
        "--update-expectations",
        action="store_true",
        default=False,
        help="Rewrite --expectations from this run's observed counts. RECORDS whatever is "
        "currently true; does not validate it. Only use this with the resulting diff reviewed "
        "by hand before committing.",
    )
    arguments = parser.parse_args()

    # Fail fast with a clear setup error rather than let every target refuse individually with a
    # confusing "command not found" -- a missing toolchain is a global problem, not a per-target one.
    missing_tools = [
        name for name in ("swift", "xcrun") if shutil.which(name) is None
    ]
    if missing_tools:
        print(
            f"FAIL: required toolchain not found on PATH: {', '.join(missing_tools)}",
            file=sys.stderr,
        )
        return 2

    if not arguments.expectations.exists():
        print(f"FAIL: expectations file not found: {arguments.expectations}", file=sys.stderr)
        return 2
    try:
        expectations_data = json.loads(arguments.expectations.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        print(f"FAIL: expectations file is not valid JSON: {error}", file=sys.stderr)
        return 2

    all_targets = expectations_data.get("targets", {})
    if not all_targets:
        print(f"FAIL: expectations file has no \"targets\": {arguments.expectations}", file=sys.stderr)
        return 2

    selected = arguments.target if arguments.target else sorted(all_targets.keys())
    unknown = [target for target in selected if target not in all_targets]
    if unknown:
        print(
            "FAIL: --target named target(s) absent from the expectations file: "
            + ", ".join(unknown),
            file=sys.stderr,
        )
        return 2

    if arguments.projection is not None:
        if not arguments.projection.exists():
            print(f"FAIL: --projection path does not exist: {arguments.projection}", file=sys.stderr)
            return 2
        return run_gate(
            arguments.projection,
            selected,
            expectations_data,
            arguments.log_dir,
            arguments.update_expectations,
            arguments.expectations,
        )

    with tempfile.TemporaryDirectory() as scratch:
        destination = pathlib.Path(scratch) / "public"
        export = run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "export_public_repository.py"),
                "--development-projection",
                "--output",
                str(destination),
            ],
            cwd=REPO_ROOT,
        )
        if export.returncode != 0:
            print("FAIL: could not export the projection", file=sys.stderr)
            print(export.stdout + export.stderr, file=sys.stderr)
            return 2

        return run_gate(
            destination,
            selected,
            expectations_data,
            arguments.log_dir,
            arguments.update_expectations,
            arguments.expectations,
        )


if __name__ == "__main__":
    sys.exit(main())
