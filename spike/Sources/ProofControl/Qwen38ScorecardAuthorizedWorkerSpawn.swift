import Foundation

/// Qwen38 scorecard chain authorized worker spawn (chain Slice 4a).
///
/// Closes the binding Slice 4 item from the 2026-09-02 security-review
/// addendum (finding S2): the Slice 3 primitive
/// `Qwen38ScorecardWorkerSpawner.spawnAndObserve` takes free
/// `harnessGitSHA`/`sourceID` parameters, which leaves an unbound seam if a
/// pipeline wires them independently of the operator-signed claim. The
/// authorized entry point below DERIVES those identity inputs from a
/// `Qwen38ScorecardResolvedRunAuthorization` — a value that can only exist
/// after real Ed25519 verification — and additionally FORCES the child
/// environment's GDN variable to match the requested leg, so a caller of
/// the authorized path cannot express a claim-inconsistent worker identity
/// or a mode/environment mismatch. The runner pipeline (Slice 4b) must use
/// this entry point, never the free primitive.
extension Qwen38ScorecardResolvedRunAuthorization {
    /// Claim-derived worker identity inputs (signed fields of the resolved
    /// claim). Exposed read-only so callers can log/compare them; they are
    /// claim contents, not secrets.
    public var authorizedHarnessGitSHA1: String {
        claim.fields.harnessGitSHA1
    }

    public var authorizedSourceID: String {
        claim.fields.sourceID
    }
}

extension Qwen38ScorecardWorkerSpawner {
    /// Spawns a worker whose evidence identity fields come from the
    /// resolved authorization's SIGNED claim — never from the caller:
    ///   - `harnessGitSHA` := claim `harness_git_sha1`
    ///   - `sourceID`      := claim `source_id`
    ///   - the GDN environment variable is forced to the requested leg
    ///     (`gdn-on` ⇒ exactly "1"; `gdn-off` ⇒ variable absent), so the
    ///     derived `observedEnv` is always consistent with `gdnMode`
    ///     regardless of what the caller put in `environment`.
    /// All Slice 3 trust properties (kernel-observed parentage, post-hash
    /// re-verification, escalated reap, documented TOCTOU residual) apply
    /// unchanged.
    public static func spawnAndObserveAuthorized(
        authorization: Qwen38ScorecardResolvedRunAuthorization,
        executableURL: URL,
        arguments: [String],
        environment: [String: String],
        standardInput: Pipe? = nil,
        standardOutput: Pipe? = nil,
        standardError: Pipe? = nil,
        gdnMode: Qwen38ScorecardRunnerGDNMode
    ) throws -> Qwen38ScorecardSpawnedWorker {
        var forcedEnvironment = environment
        switch gdnMode {
        case .on:
            forcedEnvironment[gdnEnvironmentKey] = "1"
        case .off:
            forcedEnvironment.removeValue(forKey: gdnEnvironmentKey)
        }
        return try spawnAndObserve(
            executableURL: executableURL,
            arguments: arguments,
            environment: forcedEnvironment,
            standardInput: standardInput,
            standardOutput: standardOutput,
            standardError: standardError,
            harnessGitSHA: authorization.authorizedHarnessGitSHA1,
            sourceID: authorization.authorizedSourceID,
            gdnMode: gdnMode
        )
    }
}
