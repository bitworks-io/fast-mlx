import Foundation
import HarnessCore
import ProofControl
import ScorecardPairControl

/// Chain Slice 4b (increment C): the Qwen38 scorecard proof-runner
/// pipeline. The runner is the trusted parent: it admits the operator key
/// policy against the invoker-chosen trust anchor, resolves the
/// operator-signed claim (real Ed25519 verification — the ONLY way a
/// `Qwen38ScorecardResolvedRunAuthorization` can exist), spawns BOTH leg
/// workers itself through the authorized claim-bound entry point, mints
/// the launch observations at spawn time from kernel-observed evidence,
/// drives the shared pair coordinator, and accepts a verdict only through
/// the combined validation entry — so the external launch-equality check
/// can never be skipped. Stops at the operator-signing credential
/// boundary: nothing here can sign; fixture-signed claims exist only in
/// tests.
func runQwen38ScorecardProofRunner(
    arguments: [String]
) async throws -> Qwen38ScorecardRunnerStatus {
    let parsed = try parseQwen38ScorecardRunnerArguments(arguments)
    try qwen38ScorecardRunnerPreflightDeployTreeRoot()

    // Hardened input captures (single-link regular files, bounded sizes).
    let anchorFile = try AdmittedFile.capture(
        absolutePath: parsed.trustAnchorPath,
        maximumBytes: Qwen38ScorecardRunnerInputBounds.trustAnchorMaximumBytes)
    let policyFile = try AdmittedFile.capture(
        absolutePath: parsed.policyPath,
        maximumBytes: Qwen38ScorecardRunnerInputBounds.policyMaximumBytes)
    let policySignatureFile = try AdmittedFile.capture(
        absolutePath: parsed.policySignaturePath,
        maximumBytes: Qwen38ScorecardRunnerInputBounds.signatureMaximumBytes)
    let claimFile = try AdmittedFile.capture(
        absolutePath: parsed.claimPath,
        maximumBytes: Qwen38ScorecardRunnerInputBounds.claimMaximumBytes)
    let claimSignatureFile = try AdmittedFile.capture(
        absolutePath: parsed.claimSignaturePath,
        maximumBytes: Qwen38ScorecardRunnerInputBounds.signatureMaximumBytes)

    // Slice 2 admission: root-signed key policy against the trust anchor.
    // The verification time is runner-minted (see the anchor parser).
    let trustAnchor = try parseQwen38ScorecardRunnerTrustAnchor(
        anchorFile.bytes,
        verificationUnixSeconds: UInt64(max(0, Date().timeIntervalSince1970)))
    let policy = try Qwen38ScorecardKeyPolicyVerifier.admit(
        policyFile: policyFile,
        rootSignatureBase64: try parseQwen38ScorecardRunnerSignatureLine(
            policySignatureFile.bytes,
            label: "policy"),
        trustAnchor: trustAnchor)

    // Slice 3 resolution: real Ed25519 operator-signature verification.
    let authorization = try Qwen38ScorecardRunAuthorizationResolver.resolve(
        claimBytes: claimFile.bytes,
        claimSignatureBase64: try parseQwen38ScorecardRunnerSignatureLine(
            claimSignatureFile.bytes,
            label: "claim"),
        policy: policy)

    // Fail-fast pre-spawn diagnostic (design verdict P3d): a claim signed
    // for a different source identity fails here with a clear error rather
    // than as a distant evidence-ID mismatch. Transitively enforced later,
    // so this is diagnostics-quality, not load-bearing.
    guard authorization.authorizedSourceID
        == Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID
    else {
        throw Qwen38ScorecardRunnerError.claimSourceIdentityMismatch
    }

    // Reserve fresh outputs and admit the two-worker memory plan before
    // any child exists (shared single-copy admission arithmetic — design
    // verdict P6).
    let outputs = try Qwen38MTPScorecardFreshOutputSet(
        scorecardPath: parsed.run.outputPath,
        authorityPath: parsed.run.authorityOutputPath)
    try parsed.run.memoryBudget.validateTwoWorkerAdmission(
        .live(),
        expectedChip: parsed.run.expectedChip)

    // Spawn BOTH legs through the authorized entry point only: identity
    // inputs come from the signed claim, and the GDN environment variable
    // is forced per leg. Fixed-GDN executionMode axis (design verdict P3):
    // both legs launch GDN-on; candidate/reference differ in the exact-MTP
    // vs scalar route, which the workers self-select by role.
    let workerURL = URL(fileURLWithPath: parsed.workerExecutablePath)
    let environment = ProcessInfo.processInfo.environment
    let candidateInput = Pipe()
    let candidateOutput = Pipe()
    let candidateError = Pipe()
    let candidateWorker = try Qwen38ScorecardWorkerSpawner.spawnAndObserveAuthorized(
        authorization: authorization,
        executableURL: workerURL,
        arguments: qwen38MTPScorecardWorkerLaunchArguments(
            role: .candidate,
            arguments: parsed.run),
        environment: environment,
        standardInput: candidateInput,
        standardOutput: candidateOutput,
        standardError: candidateError,
        gdnMode: .on)

    let referenceInput = Pipe()
    let referenceOutput = Pipe()
    let referenceError = Pipe()
    let referenceWorker: Qwen38ScorecardSpawnedWorker
    do {
        referenceWorker = try Qwen38ScorecardWorkerSpawner.spawnAndObserveAuthorized(
            authorization: authorization,
            executableURL: workerURL,
            arguments: qwen38MTPScorecardWorkerLaunchArguments(
                role: .reference,
                arguments: parsed.run),
            environment: environment,
            standardInput: referenceInput,
            standardOutput: referenceOutput,
            standardError: referenceError,
            gdnMode: .on)
    } catch {
        // fd hygiene on the failure path (design verdict P2 rider): reap
        // the already-spawned candidate and close the runner-held ends.
        qwen38MTPScorecardTerminateChild(candidateWorker.process, deadlineChecks: 20)
        try? candidateInput.fileHandleForWriting.close()
        try? candidateOutput.fileHandleForReading.close()
        try? candidateError.fileHandleForReading.close()
        throw error
    }

    // The launch observations are minted HERE, at spawn time, directly and
    // only from the runner's kernel-observed evidence IDs, BEFORE any
    // worker byte is read (design verdict hazard C1 — echoing the
    // worker-reported IDs back would make the equality check vacuous; the
    // structural gate pins this initializer to exactly this one site).
    let observations = Qwen38MTPPerformanceScorecardRunnerLaunchObservations(
        candidateProcessIsolationEvidenceID: candidateWorker.evidenceID,
        referenceProcessIsolationEvidenceID: referenceWorker.evidenceID)

    let releaseBuildObserved: Bool
    #if DEBUG
    // A debug runner stays runnable for fixture-driven integration, but
    // the gate's evidence-level release enforcement keeps it
    // non-promotable, and the real worker service refuses DEBUG builds.
    releaseBuildObserved = false
    #else
    releaseBuildObserved = true
    #endif

    let hostFacts = Qwen38ScorecardRunnerHostFacts.collect()
    let runIdentity = qwen38ScorecardRunnerTrustedRunIdentity(
        authorization: authorization,
        hostFacts: hostFacts)
    let coordinator = Qwen38MTPScorecardLiveCoordinator(
        candidate: Qwen38MTPScorecardLineProtocolClient(
            role: .candidate,
            transport: Qwen38MTPScorecardProcessPipesTransport(
                role: .candidate,
                child: candidateWorker.process,
                stdin: candidateInput.fileHandleForWriting,
                stdout: candidateOutput.fileHandleForReading,
                stderr: candidateError.fileHandleForReading)),
        reference: Qwen38MTPScorecardLineProtocolClient(
            role: .reference,
            transport: Qwen38MTPScorecardProcessPipesTransport(
                role: .reference,
                child: referenceWorker.process,
                stdin: referenceInput.fileHandleForWriting,
                stdout: referenceOutput.fileHandleForReading,
                stderr: referenceError.fileHandleForReading)),
        runIdentity: runIdentity,
        provenance: qwen38ScorecardRunnerProvenance(identity: runIdentity),
        releaseBuildObserved: releaseBuildObserved,
        expectations: .fixedGDNAxis)
    let result = try await coordinator.run()

    // Combined validation entry ONLY (design verdict P4): full scorecard
    // validation plus the runner launch-equality check; a Verdict exists
    // only if both passed, and the published record is the exact record
    // whose payload went through this entry.
    let verdict = try Qwen38MTPPerformanceScorecardGate.validateWithRunnerLaunchObservations(
        result.record.payload,
        authority: result.authority,
        observations: observations)

    let scorecardData = Data((try result.record.jsonLine() + "\n").utf8)
    let authorityData = try qwen38MTPScorecardCanonicalJSON(result.authority)
    try writeFreshQwen38MTPScorecardOutputSet(
        scorecardData: scorecardData,
        authorityData: authorityData,
        outputs: outputs)

    return Qwen38ScorecardRunnerStatus(
        schema: ProofControl.schema,
        program: Qwen38ScorecardRunnerStatus.programName,
        status: "OK",
        promotable: releaseBuildObserved,
        qualified: verdict.qualified,
        claimSHA256: authorization.claimSHA256,
        authorizationID: authorization.authorizationID,
        operatorKeyID: authorization.operatorKeyID,
        policyAdmissionID: authorization.policyAdmissionID.rawValue,
        candidateProcessIsolationEvidenceID: observations
            .candidateProcessIsolationEvidenceID,
        referenceProcessIsolationEvidenceID: observations
            .referenceProcessIsolationEvidenceID,
        error: nil)
}

let runnerArguments = Array(CommandLine.arguments.dropFirst())
do {
    let status = try await runQwen38ScorecardProofRunner(
        arguments: runnerArguments)
    print(try status.jsonLine())
    exit(status.status == "OK" && status.promotable ? 0 : 1)
} catch {
    let status = Qwen38ScorecardRunnerStatus.failure(String(describing: error))
    if let line = try? status.jsonLine() {
        print(line)
    }
    FileHandle.standardError.write(
        Data("qwen38-scorecard-proof-runner: \(error)\n".utf8))
    exit(1)
}
