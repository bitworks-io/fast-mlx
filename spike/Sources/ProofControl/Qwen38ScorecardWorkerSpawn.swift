import CryptoKit
import Darwin
import Foundation

/// Qwen38 scorecard chain runner-side worker spawn/observe (chain Slice 3).
///
/// Binding finding 1 of the reviewed chain design
/// (docs/task-inbox/2026-09-01-qwen38-proof-runner-scope-and-chain-design.md
/// and its 2026-09-02 security-review addendum): the live worker today mints
/// its OWN process-isolation evidence and launch binding, so the gate check
/// is pure self-consistency. The runner must therefore SPAWN the worker
/// itself and mint independent evidence over the child; Slice 4 adds the
/// external-equality gate check
/// `launchBinding.processIsolationEvidenceID == runner-minted evidenceID`.
/// A caller-supplied worker pid (`--worker-pid`) is ruled out.
///
/// Trust properties and their limits:
///   - The child pid comes from the spawn API, never from a caller.
///   - The child's parent pid is OBSERVED from the kernel
///     (`proc_bsdinfo.pbi_ppid`) and REQUIRED to equal the runner's own
///     pid; a recycled or foreign pid whose kernel-reported parent is
///     someone else fails closed (security-review finding B1). The worker
///     must be spawned DIRECTLY — any shell/`env` wrapper makes the real
///     worker a grandchild and correctly fails this check.
///   - Start-uptime plus boot-time defeat pid reuse across the observation:
///     after hashing the executable, the observation is RE-VERIFIED
///     (same kernel parent and start time, child still running) so the
///     whole observation is bound to one live process instance
///     (security-review finding S1).
///   - DOCUMENTED TOCTOU RESIDUAL (accepted, per binding finding 1): the
///     executable hash is computed from the on-disk file at the
///     kernel-reported `proc_pidpath` after exec; if the file is replaced
///     between exec and hash, the hash describes the new on-disk bytes,
///     not the loaded image. The worker's own self-hash has the same gap.
///     Closing it requires code-signing / Endpoint Security attestation,
///     which is out of scope for this chain.
///   - KNOWN RETRYABLE FAILURE MODE: both sides read `sysctl
///     KERN_BOOTTIME` at different instants; a wall-clock step (e.g. NTP)
///     between the two mints shifts `bootTimeUnixSeconds` and the derived
///     start-uptime, producing a Slice 4 evidence-ID mismatch. That is a
///     fail-closed flake, not a hole: re-run the pair.
///
/// The evidence struct below deliberately mirrors the worker-side
/// `Qwen38MTPLiveExactnessProcessIsolationEvidence` in HarnessCore FIELD
/// NAME by FIELD NAME and type by type, so that `JSONEncoder` with
/// `.sortedKeys` produces byte-identical canonical bytes for equal values
/// and the two independently minted evidence IDs are comparable by
/// equality. The duplication (rather than a shared type) follows binding
/// item 4; the cross-recipe equality test in ProofControlTests pins both
/// recipes against drift. The identity-source enum here has ONLY the
/// `proc_pidpath` case: the runner is structurally unable to express a
/// weaker identity source.
public enum Qwen38ScorecardRunnerGDNMode: String, Encodable, Equatable, Sendable {
    case on = "gdn-on"
    case off = "gdn-off"
}

public enum Qwen38ScorecardRunnerGDNObservedEnv: String, Encodable, Equatable, Sendable {
    case disabled
    case enabled
}

public enum Qwen38ScorecardRunnerExecutableIdentitySource: String, Encodable, Equatable, Sendable {
    case procPIDPath = "proc_pidpath"
}

public struct Qwen38ScorecardRunnerProcessIsolationEvidence: Encodable, Equatable, Sendable {
    public let processID: Int
    public let parentProcessID: Int
    public let processStartUptimeNanoseconds: UInt64
    public let bootTimeUnixSeconds: Int64
    public let executableIdentitySource: Qwen38ScorecardRunnerExecutableIdentitySource
    public let executableSHA256: String
    public let harnessGitSHA: String
    public let sourceID: String
    public let gdnMode: Qwen38ScorecardRunnerGDNMode
    public let observedEnv: Qwen38ScorecardRunnerGDNObservedEnv

    public init(
        processID: Int,
        parentProcessID: Int,
        processStartUptimeNanoseconds: UInt64,
        bootTimeUnixSeconds: Int64,
        executableIdentitySource: Qwen38ScorecardRunnerExecutableIdentitySource,
        executableSHA256: String,
        harnessGitSHA: String,
        sourceID: String,
        gdnMode: Qwen38ScorecardRunnerGDNMode,
        observedEnv: Qwen38ScorecardRunnerGDNObservedEnv
    ) {
        self.processID = processID
        self.parentProcessID = parentProcessID
        self.processStartUptimeNanoseconds = processStartUptimeNanoseconds
        self.bootTimeUnixSeconds = bootTimeUnixSeconds
        self.executableIdentitySource = executableIdentitySource
        self.executableSHA256 = executableSHA256
        self.harnessGitSHA = harnessGitSHA
        self.sourceID = sourceID
        self.gdnMode = gdnMode
        self.observedEnv = observedEnv
    }
}

public enum Qwen38ScorecardWorkerSpawnError: Error, Equatable, Sendable {
    case invalidHarnessGitSHA
    case invalidSourceID
    case workerExecutableNotFound(path: String)
    case spawnFailed(reason: String)
    case observationFailed(ProcessIdentityError)
    case childParentMismatch(observedParent: Int32, runner: Int32)
    case observationUnstable
}

extension Qwen38ScorecardWorkerSpawnError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .invalidHarnessGitSHA:
            "qwen38 worker spawn harness git sha is not canonical nonzero lowercase 40-hex"
        case .invalidSourceID:
            "qwen38 worker spawn source id is not canonical nonzero lowercase 64-hex"
        case .workerExecutableNotFound(let path):
            "qwen38 worker executable is not an executable regular file: \(path)"
        case .spawnFailed(let reason):
            "qwen38 worker spawn failed: \(reason)"
        case .observationFailed(let error):
            "qwen38 worker observation failed: \(error)"
        case .childParentMismatch(let observedParent, let runner):
            "qwen38 worker kernel-reported parent \(observedParent) is not the runner \(runner)"
        case .observationUnstable:
            "qwen38 worker process identity changed or exited during observation"
        }
    }
}

/// A spawned, kernel-observed worker. Not Equatable: it owns the live
/// `Process` handle, which the caller (the runner) manages to completion.
public struct Qwen38ScorecardSpawnedWorker {
    public let process: Process
    public let evidence: Qwen38ScorecardRunnerProcessIsolationEvidence
    public let evidenceID: String
}

public enum Qwen38ScorecardWorkerSpawner {
    /// Duplicated from the worker's private
    /// `qwen38MTPScorecardGDNEnvironmentKey`
    /// (spike/Sources/fastmlx-harness/Qwen38MTPScorecardLiveProcessWorker
    /// .swift, private constant, not importable here). Pinned by a golden
    /// test; if the worker key ever changes, both must move together.
    public static let gdnEnvironmentKey = "MLX_QWEN_FOUR_GDN"

    /// Duplicates the worker's `observedGDNEnv` logic exactly
    /// (Qwen38MTPScorecardLiveProcessWorker.swift): gdn-on observes
    /// `enabled` only for the exact string "1"; gdn-off observes
    /// `disabled` only when the variable is absent (any present value,
    /// including the empty string, observes `enabled`). The spawner
    /// derives this from the environment it actually passes to the child,
    /// so a caller cannot assert an inconsistent mode/env pair.
    public static func observedEnv(
        gdnMode: Qwen38ScorecardRunnerGDNMode,
        environmentValue: String?
    ) -> Qwen38ScorecardRunnerGDNObservedEnv {
        switch gdnMode {
        case .on:
            return environmentValue == "1" ? .enabled : .disabled
        case .off:
            return environmentValue == nil ? .disabled : .enabled
        }
    }

    /// Canonical evidence bytes: `JSONEncoder` with `.sortedKeys`, exactly
    /// mirroring the worker-side recipe in
    /// `Qwen38MTPLiveExactnessGate.canonicalData` so equal field values
    /// produce byte-identical JSON on both sides. No floats exist in the
    /// evidence, and no field value may contain "/" (all are pids, times,
    /// hex digests, or pinned enum tokens), so the two known
    /// JSONSerialization variability points (float formatting, slash
    /// escaping) cannot arise.
    public static func canonicalEvidenceBytes(
        _ evidence: Qwen38ScorecardRunnerProcessIsolationEvidence
    ) -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        // Encoding a pure in-memory value type with no throwing custom
        // encoder cannot fail; mirrors the worker-side `canonicalData`.
        return try! encoder.encode(evidence)
    }

    public static func evidenceID(
        for evidence: Qwen38ScorecardRunnerProcessIsolationEvidence
    ) -> String {
        sha256Hex(canonicalEvidenceBytes(evidence))
    }

    /// Spawns the worker DIRECTLY (no shell) and mints runner-side
    /// process-isolation evidence over the child. See the file comment for
    /// the trust properties, the accepted TOCTOU residual, and the
    /// retryable boot-time-jitter failure mode.
    ///
    /// `environment` is passed to the child EXACTLY (full replacement, no
    /// inheritance from the runner); `observedEnv` in the evidence is
    /// derived from it, never asserted. `harnessGitSHA` and `sourceID` are
    /// validated fail-closed BEFORE any process is created. Slice 4 is
    /// REQUIRED (recorded binding item, 2026-09-02 addendum) to derive
    /// them from a `Qwen38ScorecardResolvedRunAuthorization`, never accept
    /// them independently.
    public static func spawnAndObserve(
        executableURL: URL,
        arguments: [String],
        environment: [String: String],
        standardInput: Pipe? = nil,
        standardOutput: Pipe? = nil,
        standardError: Pipe? = nil,
        harnessGitSHA: String,
        sourceID: String,
        gdnMode: Qwen38ScorecardRunnerGDNMode
    ) throws -> Qwen38ScorecardSpawnedWorker {
        // Fail closed on every input BEFORE any process is created (the
        // same nonzero-hex rules the worker-side gate enforces).
        guard
            isLowerHex(harnessGitSHA, count: 40),
            harnessGitSHA != String(repeating: "0", count: 40)
        else {
            throw Qwen38ScorecardWorkerSpawnError.invalidHarnessGitSHA
        }
        guard
            isLowerHex(sourceID, count: 64),
            sourceID != String(repeating: "0", count: 64)
        else {
            throw Qwen38ScorecardWorkerSpawnError.invalidSourceID
        }
        var isDirectory: ObjCBool = false
        guard
            FileManager.default.fileExists(
                atPath: executableURL.path,
                isDirectory: &isDirectory
            ),
            !isDirectory.boolValue,
            FileManager.default.isExecutableFile(atPath: executableURL.path)
        else {
            throw Qwen38ScorecardWorkerSpawnError.workerExecutableNotFound(
                path: executableURL.path
            )
        }

        let process = Process()
        process.executableURL = executableURL
        process.arguments = arguments
        process.environment = environment
        if let standardInput { process.standardInput = standardInput }
        if let standardOutput { process.standardOutput = standardOutput }
        if let standardError { process.standardError = standardError }
        do {
            try process.run()
        } catch {
            throw Qwen38ScorecardWorkerSpawnError.spawnFailed(
                reason: String(describing: error)
            )
        }

        let childPID = process.processIdentifier
        do {
            let observed = try observeSpawnedChild(
                processID: childPID,
                runnerProcessID: getpid()
            )

            // Post-hash re-verification (security-review finding S1):
            // the executable hash above is only evidence about THIS child
            // if the same kernel identity (parent + start time) still
            // describes a live process after hashing completes.
            let parentAgain: pid_t
            let startAgain: UInt64
            do {
                parentAgain = try ProcessIdentity.parentProcessID(
                    pid: childPID
                )
                startAgain = try ProcessIdentity.processStartUptimeNanoseconds(
                    pid: childPID
                )
            } catch let error as ProcessIdentityError {
                throw Qwen38ScorecardWorkerSpawnError.observationFailed(error)
            }
            guard
                parentAgain == observed.parentProcessID,
                startAgain == observed.startUptimeNanoseconds,
                process.isRunning
            else {
                throw Qwen38ScorecardWorkerSpawnError.observationUnstable
            }

            let evidence = Qwen38ScorecardRunnerProcessIsolationEvidence(
                processID: Int(childPID),
                parentProcessID: Int(observed.parentProcessID),
                processStartUptimeNanoseconds: observed.startUptimeNanoseconds,
                bootTimeUnixSeconds: Int64(observed.bootTimeUnixSeconds),
                executableIdentitySource: .procPIDPath,
                executableSHA256: observed.executableSHA256,
                harnessGitSHA: harnessGitSHA,
                sourceID: sourceID,
                gdnMode: gdnMode,
                observedEnv: observedEnv(
                    gdnMode: gdnMode,
                    environmentValue: environment[gdnEnvironmentKey]
                )
            )
            return Qwen38ScorecardSpawnedWorker(
                process: process,
                evidence: evidence,
                evidenceID: evidenceID(for: evidence)
            )
        } catch {
            reap(process)
            throw error
        }
    }

    /// Kernel observation of an already-spawned child. `internal` ONLY as
    /// a test seam for the parent-mismatch guard (the pid-reuse race is
    /// not otherwise constructible deterministically in a test); the
    /// public trust API remains spawn-owned — this is NOT a caller-pid
    /// entry point and must never become public or CLI-reachable.
    static func observeSpawnedChild(
        processID: pid_t,
        runnerProcessID: pid_t
    ) throws -> (
        parentProcessID: pid_t,
        startUptimeNanoseconds: UInt64,
        bootTimeUnixSeconds: Int,
        executableSHA256: String
    ) {
        let parentProcessID: pid_t
        do {
            parentProcessID = try ProcessIdentity.parentProcessID(
                pid: processID
            )
        } catch let error as ProcessIdentityError {
            throw Qwen38ScorecardWorkerSpawnError.observationFailed(error)
        }
        // Security-review finding B1: the parent is OBSERVED from the
        // kernel and REQUIRED to be the runner; a recycled or foreign pid
        // fails closed here before any evidence exists.
        guard parentProcessID == runnerProcessID else {
            throw Qwen38ScorecardWorkerSpawnError.childParentMismatch(
                observedParent: Int32(parentProcessID),
                runner: Int32(runnerProcessID)
            )
        }

        do {
            let startUptimeNanoseconds =
                try ProcessIdentity.processStartUptimeNanoseconds(
                    pid: processID
                )
            let bootTimeUnixSeconds = try ProcessIdentity.bootTimeUnixSeconds()
            let executableSHA256 = try ProcessIdentity.executableSHA256(
                pid: processID
            )
            return (
                parentProcessID: parentProcessID,
                startUptimeNanoseconds: startUptimeNanoseconds,
                bootTimeUnixSeconds: bootTimeUnixSeconds,
                executableSHA256: executableSHA256
            )
        } catch let error as ProcessIdentityError {
            throw Qwen38ScorecardWorkerSpawnError.observationFailed(error)
        }
    }
}

private extension Qwen38ScorecardWorkerSpawner {
    /// Escalated termination for a child whose observation failed,
    /// mirroring the existing terminate/force-kill precedent in the live
    /// worker adapter. The SIGKILL is guarded by `isRunning` so a child
    /// Foundation has already observed exiting is not signaled; the
    /// residual micro-window (exit + reap + pid recycle between the check
    /// and the signal) is accepted — a zombie pid cannot be recycled
    /// until reaped, and `Process` reaping is what flips `isRunning`.
    static func reap(_ process: Process) {
        if process.isRunning {
            process.terminate()
            Darwin.kill(process.processIdentifier, SIGKILL)
        }
        process.waitUntilExit()
    }

    static func isLowerHex(_ value: String, count: Int) -> Bool {
        value.utf8.count == count && value.utf8.allSatisfy {
            (0x30...0x39).contains($0) || (0x61...0x66).contains($0)
        }
    }

    static func sha256Hex(_ bytes: Data) -> String {
        SHA256.hash(data: bytes)
            .map { String(format: "%02x", $0) }
            .joined()
    }
}
