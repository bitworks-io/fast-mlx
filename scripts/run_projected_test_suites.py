#!/usr/bin/env python3
"""Run the PUBLIC projection's test targets, which no other gate has ever done.

The public repository ships `spike/Tests/**` -- roughly 300 projected test files -- but until this
gate existed, nothing even compiled them, and compiling is not the same as running. Cycle 81 found
the published engine shipping a fail-open speculation-gate safety defect that a projected test would
have caught: the test compiled cleanly and only failed, silently, at runtime, because the sanitized
override it exercised was missing a safety requirement (an `XCTAssertThrowsError` that did not
throw). A compile-only check would have stayed green through that defect; only actually running the
suite catches it, which is why this gate runs bundles instead of stopping at a successful build.

This gate exports (or reuses) a public projection, builds each named test target INDIVIDUALLY with
`swift build --package-path spike --target <Target>` -- this compiles only that target's dependency
closure, not the whole package (measured: 2.28s warm for one target, versus minutes for the full
package with MLX/Metal) -- then runs the resulting `.xctest` bundle DIRECTLY with `xcrun xctest
<bundle>`, and compares the parsed result against a PINNED baseline in
`scripts/projected_test_expectations.json`.

Both of those choices matter and are not incidental:

- Building per-target, not `swift build --package-path spike --build-tests`, keeps this gate usable
  without paying for the MLX/Metal targets every run, and lets one target's build failure be
  attributed precisely instead of stopping the whole package build at the first broken target.
- Running the bundle directly, not `swift test --package-path spike --filter '<Target>\\.'`, matters
  because a `--filter` run still walks EVERY `.xctest` bundle in the package: each non-matching
  bundle prints its own `Executed 0 tests, ...` summary alongside the target's real one, and a
  parser that is not careful about which summary line belongs to which bundle will silently read
  the wrong one. Running one bundle at a time removes that ambiguity by construction.

WHAT THIS GATE PROVES, AND WHAT IT DOES NOT: a passing run proves the named projected suites still
RUN and still match a recorded test/failure count on the host that ran them. It does NOT prove that
recorded baseline is CORRECT. Every target is currently pinned at 0 failures; that was not always
true, and it does not need to stay true for this gate to still be doing its job. The gate's only job
is to notice when a target's test count or failure count moves, in either direction, so the move gets
a conscious decision (via `--update-expectations`) instead of silently drifting.

ANTI-VACUITY RULES -- every one of these exists because this project has been burned by exactly this
shape of false-green before, and every one of them is a hard FAIL (non-zero exit), never a skip:

- No XCTest summary line found for a target (crash, truncation, unexpected output shape) -> FAIL,
  reported as REFUSED. A run that produced no readable result must never be mistaken for a pass.
- Executed test count is zero -> FAIL. `xcrun xctest` exits 0 on a bundle that finds nothing to run,
  so a silently-broken test discovery would otherwise look identical to success.
- Executed test count does not equal the pinned expectation, in EITHER direction -> FAIL. Lower
  catches "green by deletion" (quietly removing a failing test); higher catches an unrecorded
  addition and forces the baseline to be updated on purpose, with review, via
  --update-expectations, rather than by drifting.
- Failure count does not equal the pinned expectation, in EITHER direction -> FAIL. The pinned number
  is whatever was last measured and consciously recorded, whether that is zero or not; a target that
  silently starts passing must ALSO trip this gate, so the improvement gets recorded on purpose
  instead of going unnoticed, the same as a regression would.
- A build that reports success but produces no `.xctest` bundle, and a bundle that exists but
  contains no executable to run, are reported as two DISTINCT refusals, not folded into one vague
  "bundle missing" message -- they are the same measured toolchain gap surfacing at two different
  resolution stages (a toolchain that does not assemble a runnable bundle for a single-target build
  can either skip producing the bundle directory entirely, or leave a skeleton bundle from an
  incremental build with an empty `Contents/MacOS/`), and telling them apart is what pointed at the
  toolchain instead of a phantom crash. A build failure or an unavailable `swift`/`xcrun` toolchain
  are their own FAILs, reported as REFUSED. There is deliberately no "toolchain unavailable, skip
  and return success" branch anywhere in this file.

This gate runs in CI on every push and pull request, as its own job. It is deliberately kept out of
the fast `scripts/tests` Python suite (`python3 -m unittest discover -s scripts/tests`): exporting
plus building and running eight targets takes tens of minutes, which would make that fast suite
unusable for quick local iteration. To run it locally against a checkout, from the repository root:
`python3 scripts/run_projected_test_suites.py --projection .`. Run it after any change to
`spike/Tests`, `spike/Sources`, or a `public/sanitized-projection/` override that a projected test
depends on, even if you also rely on CI to catch it.

Exit codes: 0 = every selected target matched its pinned expectation (test count and failure count);
1 = at least one target mismatched or was refused; 2 = setup error (export failed, the projection
path or expectations file is missing, or the `swift`/`xcrun` toolchain is not on PATH).
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

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_EXPECTATIONS = REPO_ROOT / "scripts" / "projected_test_expectations.json"

# Matches the counts line XCTest prints immediately after a `Test Suite '<name>.xctest' passed|failed
# at <date>.` line, e.g. "Executed 721 tests, with 2 tests skipped and 0 failures (0 unexpected) in
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


def run(command: list[str], cwd: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


def parse_summary(output: str, target: str) -> tuple[int, int, int] | None:
    """Return (tests, skipped, failures) parsed from the summary for exactly this target's bundle,
    or None if no usable summary was found.

    A single-bundle `xcrun xctest` run prints a `Test Suite '<ClassName>'` line per test class, plus
    an aggregate `Test Suite 'All tests'` summary, in addition to the bundle-level
    `Test Suite '<Target>.xctest'` line this function anchors on. Scanning the whole output for the
    first "Executed N tests" line would risk reading one of those other summaries instead of this
    target's own -- anchoring to the exact `'<target>.xctest'` marker removes that ambiguity.
    """
    marker = f"Test Suite '{target}.xctest'"
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


def primary_bundle_path(destination: pathlib.Path, target: str) -> pathlib.Path:
    """The bundle path a per-target `swift build --package-path spike --target <Target>` has been
    observed to produce on this project's toolchain. Factored out of `resolve_bundle` so a refusal
    message can name the exact path this gate looked for, even when nothing was found there."""
    return destination / "spike" / ".build" / "out" / "Products" / "Debug" / f"{target}.xctest"


def resolve_bundle(destination: pathlib.Path, target: str) -> pathlib.Path | None:
    """Locate the built `.xctest` bundle for one target.

    The primary path is where a per-target `swift build --package-path spike --target <Target>` has
    been observed to place the bundle on this project's toolchain. The recursive fallback exists
    because that exact layout is a build-system implementation detail this gate does not control;
    it should not hard-refuse on that detail alone if a different toolchain places the bundle
    somewhere else under the same `.build` tree. Returning None either way is a REFUSAL upstream,
    never a silent pass.
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
    """The path an already-resolved `.xctest` bundle's runnable executable is expected at."""
    return bundle / "Contents" / "MacOS" / target


def bundle_has_executable(bundle: pathlib.Path, target: str) -> bool:
    """Whether an already-resolved `.xctest` bundle actually contains a runnable executable.

    A bundle directory existing is not sufficient on its own: an incremental build tree, on a
    toolchain that does not assemble runnable single-target test bundles (measured: Swift 6.3.3),
    can leave a skeleton `<Target>.xctest/Contents/MacOS/` directory with nothing inside it, or
    with `Contents/MacOS/` missing entirely. This is the same underlying toolchain gap as a wholly
    missing bundle, caught at a later resolution stage, so it needs its own check rather than being
    assumed away once `resolve_bundle` finds a directory.
    """
    return bundle_executable_path(bundle, target).is_file()


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


def evaluate_target(
    destination: pathlib.Path,
    target: str,
    expected: dict,
    log_dir: pathlib.Path | None,
) -> TargetResult:
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

    build = run(
        ["swift", "build", "--package-path", "spike", "--target", target], cwd=destination
    )
    log_parts.append(f"$ swift build --package-path spike --target {target}\n")
    log_parts.append(build.stdout)
    log_parts.append(build.stderr)
    if build.returncode != 0:
        return finish(None, REFUSED, f"build failed (swift build exit {build.returncode})")

    bundle = resolve_bundle(destination, target)
    if bundle is None:
        expected_bundle = primary_bundle_path(destination, target)
        return finish(
            None,
            REFUSED,
            f"build reported success but no {target}.xctest bundle exists at {expected_bundle} (or "
            "anywhere else under spike/.build) -- likely cause: this toolchain does not "
            "assemble a runnable test bundle for a single-target `swift build --target` build "
            "(measured: works on Swift 6.4, does not on Swift 6.3.3); check the active "
            f"Swift/Xcode version ({swift_toolchain_version()})",
        )

    if not bundle_has_executable(bundle, target):
        expected_executable = bundle_executable_path(bundle, target)
        return finish(
            None,
            REFUSED,
            f"{target}.xctest bundle exists but has no executable at {expected_executable} -- "
            "likely cause: the same toolchain gap as a missing bundle, this time leaving an "
            "incomplete/empty bundle from an incremental build (measured: works on Swift 6.4, "
            f"does not on Swift 6.3.3); check the active Swift/Xcode version "
            f"({swift_toolchain_version()})",
        )

    xctest = run(["xcrun", "xctest", str(bundle)], cwd=destination)
    log_parts.append(f"\n$ xcrun xctest {bundle}\n")
    log_parts.append(xctest.stdout)
    log_parts.append(xctest.stderr)

    # Deliberately not gating on xctest.returncode here: XCTest exits non-zero whenever any test
    # fails, which is an expected, already-recorded shape for a target like HarnessCoreTests. The
    # only thing that makes a run unreadable is the absence of a parseable summary line, checked
    # next -- that is the actual REFUSAL signal, not the process exit code.
    parsed = parse_summary(xctest.stdout + "\n" + xctest.stderr, target)
    if parsed is None:
        return finish(
            None, REFUSED, "no XCTest summary line found for this target (crash or truncated run)"
        )

    actual_tests, actual_skipped, actual_failures = parsed
    # A zero-test run needs no separate branch: every pinned expectation has tests > 0, so
    # actual_tests == 0 always trips the mismatch check below. It is called out here in comments,
    # not code, because it is exactly the failure shape this gate exists to catch -- a run that
    # silently found nothing to execute, exited 0, and would otherwise look identical to success.
    if actual_tests != expected_tests or actual_failures != expected_failures:
        detail = (
            f"tests {actual_tests} (expected {expected_tests}), "
            f"failures {actual_failures} (expected {expected_failures})"
        )
        return finish(parsed, MISMATCH, detail)

    return finish(parsed, OK, "")


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


def update_expectations(
    expectations_path: pathlib.Path, expectations_data: dict, results: list[TargetResult]
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

    expectations_path.write_text(json.dumps(expectations_data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote updated expectations to {expectations_path}", file=sys.stderr)


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

    results = [
        evaluate_target(destination, target, all_targets[target], log_dir) for target in selected
    ]
    print_table(results)

    if update:
        update_expectations(expectations_path, expectations_data, results)

    return 0 if all(result.verdict == OK for result in results) else 1


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
        help="Write each target's full raw build+test output to <log-dir>/<Target>.log. Logs are "
        "never truncated -- a failure's detail is usually at the tail.",
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
