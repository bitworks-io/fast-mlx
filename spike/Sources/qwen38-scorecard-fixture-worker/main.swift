import Darwin
import Foundation
import HarnessCore
import ProofControl
import ScorecardPairControl

/// Slice 4b two-process integration FIXTURE child (design verdict P8).
///
/// Self-mints its process-isolation evidence with the worker-side
/// HarnessCore recipe over its OWN kernel-observed facts, taking the two
/// claim identity fields via argv. That argv seam is acceptable ONLY in
/// this test fixture — the production worker self-observes both fields —
/// and this target must never ship as a product. No signing capability
/// exists here (the structural no-self-sign gate scans this target too).
///
/// Protocol: emit exactly one sorted-keys JSON line on stdout carrying the
/// self-minted evidence and its ID, then block reading stdin until EOF and
/// exit 0 — the parent controls lifetime. With `--ignore-termination`
/// the fixture ignores BOTH SIGTERM and SIGINT, so only the transport's
/// final SIGKILL escalation rung can end it — exercising the full
/// escalation against a real child.

struct FixtureWorkerReport: Codable {
    let role: String
    let processIsolation: Qwen38MTPLiveExactnessProcessIsolationEvidence
    let processIsolationEvidenceID: String
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(
        Data("qwen38-scorecard-fixture-worker: \(message)\n".utf8))
    exit(64)
}

var role: String?
var harnessGitSHA: String?
var sourceID: String?
var ignoreTermination = false

var index = 1
let argv = CommandLine.arguments
while index < argv.count {
    switch argv[index] {
    case "--role":
        guard index + 1 < argv.count else { fail("missing --role value") }
        role = argv[index + 1]
        index += 2
    case "--harness-git-sha":
        guard index + 1 < argv.count else { fail("missing --harness-git-sha value") }
        harnessGitSHA = argv[index + 1]
        index += 2
    case "--source-id":
        guard index + 1 < argv.count else { fail("missing --source-id value") }
        sourceID = argv[index + 1]
        index += 2
    case "--ignore-termination":
        ignoreTermination = true
        index += 1
    default:
        fail("unknown argument \(argv[index])")
    }
}

guard let role, role == "candidate" || role == "reference" else {
    fail("--role must be candidate or reference")
}
guard let harnessGitSHA, let sourceID else {
    fail("--harness-git-sha and --source-id are required")
}

if ignoreTermination {
    signal(SIGTERM, SIG_IGN)
    signal(SIGINT, SIG_IGN)
}

let pid = getpid()
let gdnEnvironmentValue = ProcessInfo.processInfo.environment[
    Qwen38ScorecardWorkerSpawner.gdnEnvironmentKey]
let evidence: Qwen38MTPLiveExactnessProcessIsolationEvidence
do {
    evidence = Qwen38MTPLiveExactnessProcessIsolationEvidence(
        processID: Int(pid),
        parentProcessID: Int(getppid()),
        processStartUptimeNanoseconds:
            try ProcessIdentity.processStartUptimeNanoseconds(pid: pid),
        bootTimeUnixSeconds: Int64(try ProcessIdentity.bootTimeUnixSeconds()),
        executableIdentitySource: .procPIDPath,
        executableSHA256: try ProcessIdentity.executableSHA256(pid: pid),
        harnessGitSHA: harnessGitSHA,
        sourceID: sourceID,
        gdnMode: .gdnOn,
        observedEnv: gdnEnvironmentValue == "1" ? .enabled : .disabled)
} catch {
    fail("evidence collection failed: \(error)")
}

let report = FixtureWorkerReport(
    role: role,
    processIsolation: evidence,
    processIsolationEvidenceID:
        Qwen38MTPLiveExactnessGate.processIsolationEvidenceID(for: evidence))

do {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    var line = try encoder.encode(report)
    line.append(0x0a)
    try FileHandle.standardOutput.write(contentsOf: line)
} catch {
    fail("report encoding failed: \(error)")
}

// Block until the parent closes stdin (or sends any EOF); the transport's
// terminate() closes stdin first, so a cooperative fixture exits here.
while true {
    let chunk = FileHandle.standardInput.availableData
    if chunk.isEmpty {
        break
    }
}
exit(0)
