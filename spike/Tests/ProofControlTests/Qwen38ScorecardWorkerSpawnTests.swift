import CryptoKit
import Darwin
import Foundation
import HarnessCore
import XCTest

@testable import ProofControl

/// Slice 3: runner-side spawn/observe. The evidence recipe here must stay
/// byte-identical to the worker-side HarnessCore recipe — the Slice 4
/// external-equality gate check compares the two independently minted
/// evidence IDs by equality. HarnessCore is imported as a TEST-ONLY
/// dependency to pin that cross-recipe equality for real; the golden
/// constants additionally pin the ProofControl recipe against an
/// out-of-band Python oracle (`hashlib`/`json` with sorted keys), so if
/// either side drifts, the failing pin identifies which one moved.
final class Qwen38ScorecardWorkerSpawnTests: XCTestCase {
    // Golden fixture values. The JSON text, byte count, and sha256 below
    // were computed OUT OF BAND with Python (json.dumps sort_keys compact
    // + hashlib.sha256), independent of the Swift code under test.
    private static let goldenExecutableSHA256 =
        "dc367c27a51a1984acbc0ba74be0c781c131607468be63f00778070a0cc8cb16"
    private static let goldenHarnessGitSHA =
        "9ce731e83999d51a2b86ff569fc9b2815e99c75b"
    private static let goldenSourceID =
        "0c84df486dd162c97bb65f2683e51a048562089d3ceb453423554bc72ee1bfc6"
    private static let goldenCanonicalJSON =
        #"{"bootTimeUnixSeconds":1700000123,"executableIdentitySource":"proc_pidpath","executableSHA256":"dc367c27a51a1984acbc0ba74be0c781c131607468be63f00778070a0cc8cb16","gdnMode":"gdn-on","harnessGitSHA":"9ce731e83999d51a2b86ff569fc9b2815e99c75b","observedEnv":"enabled","parentProcessID":4241,"processID":4242,"processStartUptimeNanoseconds":123456789012,"sourceID":"0c84df486dd162c97bb65f2683e51a048562089d3ceb453423554bc72ee1bfc6"}"#
    private static let goldenCanonicalByteCount = 427
    private static let goldenEvidenceID =
        "8cac9c594898f6e611ff7475958ef129f50f73279b237a4b194728f824d24d44"

    private static func goldenEvidence()
        -> Qwen38ScorecardRunnerProcessIsolationEvidence
    {
        Qwen38ScorecardRunnerProcessIsolationEvidence(
            processID: 4242,
            parentProcessID: 4241,
            processStartUptimeNanoseconds: 123_456_789_012,
            bootTimeUnixSeconds: 1_700_000_123,
            executableIdentitySource: .procPIDPath,
            executableSHA256: goldenExecutableSHA256,
            harnessGitSHA: goldenHarnessGitSHA,
            sourceID: goldenSourceID,
            gdnMode: .on,
            observedEnv: .enabled
        )
    }

    // MARK: - Pinned constants and derivation

    func testGDNEnvironmentKeyPinned() {
        // Duplicated from the worker's private constant in
        // Qwen38MTPScorecardLiveProcessWorker.swift; the two must move
        // together.
        XCTAssertEqual(
            Qwen38ScorecardWorkerSpawner.gdnEnvironmentKey,
            "MLX_QWEN_FOUR_GDN"
        )
    }

    func testObservedEnvDerivationTable() {
        let cases: [(
            Qwen38ScorecardRunnerGDNMode,
            String?,
            Qwen38ScorecardRunnerGDNObservedEnv
        )] = [
            // gdn-on observes enabled ONLY for the exact string "1".
            (.on, "1", .enabled),
            (.on, nil, .disabled),
            (.on, "0", .disabled),
            (.on, "", .disabled),
            // gdn-off observes disabled ONLY when absent; any present
            // value, including the empty string, observes enabled.
            (.off, nil, .disabled),
            (.off, "1", .enabled),
            (.off, "", .enabled),
            (.off, "0", .enabled),
        ]
        for (mode, value, expected) in cases {
            XCTAssertEqual(
                Qwen38ScorecardWorkerSpawner.observedEnv(
                    gdnMode: mode,
                    environmentValue: value
                ),
                expected,
                "\(mode) \(value ?? "<absent>")"
            )
        }
    }

    // MARK: - Canonical recipe pins

    func testGoldenCanonicalEvidenceBytesMatchIndependentOracle() {
        let bytes = Qwen38ScorecardWorkerSpawner.canonicalEvidenceBytes(
            Self.goldenEvidence()
        )
        XCTAssertEqual(bytes, Data(Self.goldenCanonicalJSON.utf8))
        XCTAssertEqual(bytes.count, Self.goldenCanonicalByteCount)
        XCTAssertEqual(
            Qwen38ScorecardWorkerSpawner.evidenceID(
                for: Self.goldenEvidence()
            ),
            Self.goldenEvidenceID
        )
    }

    func testCrossRecipeEvidenceIDMatchesHarnessCoreWorkerRecipe() {
        let workerSide = Qwen38MTPLiveExactnessProcessIsolationEvidence(
            processID: 4242,
            parentProcessID: 4241,
            processStartUptimeNanoseconds: 123_456_789_012,
            bootTimeUnixSeconds: 1_700_000_123,
            executableIdentitySource: .procPIDPath,
            executableSHA256: Self.goldenExecutableSHA256,
            harnessGitSHA: Self.goldenHarnessGitSHA,
            sourceID: Self.goldenSourceID,
            gdnMode: .gdnOn,
            observedEnv: .enabled
        )
        let workerSideID = Qwen38MTPLiveExactnessGate.processIsolationEvidenceID(
            for: workerSide
        )
        let runnerSideID = Qwen38ScorecardWorkerSpawner.evidenceID(
            for: Self.goldenEvidence()
        )
        // Triple pin: worker recipe == runner recipe == independent oracle.
        XCTAssertEqual(workerSideID, runnerSideID)
        XCTAssertEqual(workerSideID, Self.goldenEvidenceID)
    }

    // MARK: - Kernel parent observation

    func testParentProcessIDCollector() throws {
        XCTAssertEqual(
            try ProcessIdentity.parentProcessID(pid: getpid()),
            getppid()
        )
        // The test process's own parent is a same-user process this test
        // can observe but did NOT spawn (proc_pidinfo on system daemons
        // like pid 1 is permission-denied for unprivileged callers, so
        // launchd is not usable as the fixture). A process's grandparent
        // can never be the process itself.
        XCTAssertNotEqual(
            try ProcessIdentity.parentProcessID(pid: getppid()),
            getpid()
        )

        XCTAssertThrowsError(
            try ProcessIdentity.parentProcessID(pid: 9_999_999)
        ) { error in
            XCTAssertEqual(
                error as? ProcessIdentityError,
                .processInfoUnavailable(pid: 9_999_999)
            )
        }
    }

    /// The pid-reuse race itself is not deterministically constructible in
    /// a test, so the guard is exercised through the internal observation
    /// seam with a pid whose kernel-reported parent is provably not the
    /// runner: the test process's own parent.
    func testObserveSpawnedChildRejectsNonChildPID() {
        XCTAssertThrowsError(
            try Qwen38ScorecardWorkerSpawner.observeSpawnedChild(
                processID: getppid(),
                runnerProcessID: getpid()
            )
        ) { error in
            guard
                case .childParentMismatch(let observedParent, let runner) =
                    error as? Qwen38ScorecardWorkerSpawnError
            else {
                return XCTFail("expected childParentMismatch, got \(error)")
            }
            XCTAssertEqual(runner, getpid())
            XCTAssertNotEqual(observedParent, getpid())
        }
    }

    // MARK: - Spawn + observe

    func testSpawnAndObserveMintsKernelObservedEvidence() throws {
        let stdinPipe = Pipe()
        let worker = try Qwen38ScorecardWorkerSpawner.spawnAndObserve(
            executableURL: URL(fileURLWithPath: "/bin/cat"),
            arguments: [],
            environment: [
                Qwen38ScorecardWorkerSpawner.gdnEnvironmentKey: "1"
            ],
            standardInput: stdinPipe,
            harnessGitSHA: Self.goldenHarnessGitSHA,
            sourceID: Self.goldenSourceID,
            gdnMode: .on
        )
        defer {
            try? stdinPipe.fileHandleForWriting.close()
            worker.process.waitUntilExit()
        }

        XCTAssertEqual(
            worker.evidence.processID,
            Int(worker.process.processIdentifier)
        )
        XCTAssertEqual(worker.evidence.parentProcessID, Int(getpid()))
        XCTAssertGreaterThan(
            worker.evidence.processStartUptimeNanoseconds,
            0
        )
        XCTAssertGreaterThan(worker.evidence.bootTimeUnixSeconds, 0)
        XCTAssertEqual(
            worker.evidence.executableIdentitySource,
            .procPIDPath
        )
        // Independent hash of the same on-disk binary the child executes.
        let catBytes = try Data(
            contentsOf: URL(fileURLWithPath: "/bin/cat")
        )
        XCTAssertEqual(
            worker.evidence.executableSHA256,
            SHA256.hash(data: catBytes)
                .map { String(format: "%02x", $0) }
                .joined()
        )
        XCTAssertEqual(
            worker.evidence.harnessGitSHA,
            Self.goldenHarnessGitSHA
        )
        XCTAssertEqual(worker.evidence.sourceID, Self.goldenSourceID)
        XCTAssertEqual(worker.evidence.gdnMode, .on)
        XCTAssertEqual(worker.evidence.observedEnv, .enabled)
        XCTAssertEqual(
            worker.evidenceID,
            Qwen38ScorecardWorkerSpawner.evidenceID(for: worker.evidence)
        )
        XCTAssertTrue(worker.process.isRunning)

        try stdinPipe.fileHandleForWriting.close()
        worker.process.waitUntilExit()
        XCTAssertEqual(worker.process.terminationStatus, 0)
    }

    func testSpawnValidatesInputsBeforeProcessCreation() {
        // The executable deliberately does not exist: if validation runs
        // first (as required), the input error is thrown and no spawn is
        // ever attempted.
        let missing = URL(fileURLWithPath: "/nonexistent-fastmlx-worker")
        let badHarness = [
            "XYZ",
            String(repeating: "d", count: 39),
            String(repeating: "D", count: 40),
            String(repeating: "0", count: 40),
        ]
        for value in badHarness {
            XCTAssertThrowsError(
                try Qwen38ScorecardWorkerSpawner.spawnAndObserve(
                    executableURL: missing,
                    arguments: [],
                    environment: [:],
                    harnessGitSHA: value,
                    sourceID: Self.goldenSourceID,
                    gdnMode: .off
                ),
                value
            ) { error in
                XCTAssertEqual(
                    error as? Qwen38ScorecardWorkerSpawnError,
                    .invalidHarnessGitSHA,
                    value
                )
            }
        }

        let badSource = [
            "short",
            String(repeating: "e", count: 63),
            String(repeating: "E", count: 64),
            String(repeating: "0", count: 64),
        ]
        for value in badSource {
            XCTAssertThrowsError(
                try Qwen38ScorecardWorkerSpawner.spawnAndObserve(
                    executableURL: missing,
                    arguments: [],
                    environment: [:],
                    harnessGitSHA: Self.goldenHarnessGitSHA,
                    sourceID: value,
                    gdnMode: .off
                ),
                value
            ) { error in
                XCTAssertEqual(
                    error as? Qwen38ScorecardWorkerSpawnError,
                    .invalidSourceID,
                    value
                )
            }
        }
    }

    func testSpawnRejectsMissingExecutable() {
        XCTAssertThrowsError(
            try Qwen38ScorecardWorkerSpawner.spawnAndObserve(
                executableURL: URL(
                    fileURLWithPath: "/nonexistent-fastmlx-worker"
                ),
                arguments: [],
                environment: [:],
                harnessGitSHA: Self.goldenHarnessGitSHA,
                sourceID: Self.goldenSourceID,
                gdnMode: .off
            )
        ) { error in
            XCTAssertEqual(
                error as? Qwen38ScorecardWorkerSpawnError,
                .workerExecutableNotFound(
                    path: "/nonexistent-fastmlx-worker"
                )
            )
        }
    }

    /// A fast-exiting child races the observation; both outcomes are
    /// legitimate and both must be truthful — either a complete
    /// observation that honestly describes the real binary, or a typed
    /// fail-closed observation error (with the child reaped inside the
    /// spawner). Nondeterminism here is bounded to that either/or.
    func testFastExitingChildYieldsTruthfulObservationOrTypedError() throws {
        do {
            let worker = try Qwen38ScorecardWorkerSpawner.spawnAndObserve(
                executableURL: URL(fileURLWithPath: "/usr/bin/true"),
                arguments: [],
                environment: [:],
                harnessGitSHA: Self.goldenHarnessGitSHA,
                sourceID: Self.goldenSourceID,
                gdnMode: .off
            )
            worker.process.waitUntilExit()
            let trueBytes = try Data(
                contentsOf: URL(fileURLWithPath: "/usr/bin/true")
            )
            XCTAssertEqual(
                worker.evidence.executableSHA256,
                SHA256.hash(data: trueBytes)
                    .map { String(format: "%02x", $0) }
                    .joined()
            )
            XCTAssertEqual(worker.evidence.parentProcessID, Int(getpid()))
            XCTAssertEqual(worker.evidence.observedEnv, .disabled)
        } catch let error as Qwen38ScorecardWorkerSpawnError {
            switch error {
            case .observationFailed, .childParentMismatch,
                .observationUnstable:
                break
            default:
                XCTFail("unexpected spawn error: \(error)")
            }
        }
    }
}
