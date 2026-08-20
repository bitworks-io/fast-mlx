import Foundation

/// The outcome of resolving a list of on-disk quant candidate directories: the full `QuantPickResult`
/// plus the winning directory the serve path should load and its already-decoded arch (so the serve
/// path need not decode it a second time). `winnerDirectory`/`winnerParsed` are `nil` exactly when the
/// pick refused (no candidate fits).
public struct QuantResolution: Sendable {
    public let pick: QuantPickResult
    public let winnerDirectory: URL?
    public let winnerParsed: ParsedModelArch?

    public var shouldProceed: Bool { pick.shouldProceed }
    public func summaryLines() -> [String] { pick.summaryLines() }
    /// The `--quant-pick-only` machine line (`nil` when the pick refused). Winner path comes from the
    /// resolver's directory mapping (== `pick.winnerRepoID`, which is the candidate's absolute path).
    public func machineReadableWinnerLine() -> String? { pick.machineReadableWinnerLine() }
}

/// Serve-path glue for quant auto-pick (fit-checked-serve, differentiator #2 full shape): turn a list
/// of already-downloaded local checkpoint directories into a pick. Each directory is decoded from its
/// real `config.json` + summed safetensors bytes; an undecodable directory becomes an
/// excluded-with-reason candidate (never fatal); the winner is mapped back to the directory the serve
/// path loads. Pure and MLX-free, so the whole resolution is unit-tested off-box; the live serve only
/// confirms the winning directory loads.
///
/// Candidate *sourcing* is deliberately local-directories-only — enumerating HF repo names + probing
/// remote metadata is the unresolved policy decision #3 in
/// docs/task-inbox/2026-08-18-quant-auto-pick-policy.md and is not built here. A red-only candidate
/// set refuses regardless of any `--force`; force-serving the best red candidate in candidates mode is
/// a separate, not-yet-wired feature (see the same policy doc).
public enum QuantCandidateResolver {

    public static func resolve(
        candidateDirectories: [URL], host: SystemProfile,
        requestedContext: Int? = nil, kvQuant: KVQuantTier = .fp16,
        concurrency: Int = 1, thresholds: CapacityThresholds = .default,
        allowedKVTiers: [KVQuantTier]? = nil, allowContextCapping: Bool = true,
        preference: QuantPickPreference = .context
    ) -> QuantResolution {
        // Use the absolute path as the candidate identity: it is unique across the list, so the winner
        // maps back to exactly one directory (a display-name like the last path component could collide).
        var directoryByID: [String: URL] = [:]
        var parsedByID: [String: ParsedModelArch] = [:]

        let candidates: [QuantServeCandidate] = candidateDirectories.map { dir in
            let id = dir.path
            directoryByID[id] = dir
            do {
                let parsed = try ModelConfigDecoder.decodeModelDirectory(dir, id: id)
                parsedByID[id] = parsed
                return QuantServeCandidate(repoID: id, parsed: parsed)
            } catch {
                return QuantServeCandidate(repoID: id, parsed: nil, exclusionReason: "\(error)")
            }
        }

        let pick = QuantAutoPicker.pick(
            candidates: candidates, host: host, requestedContext: requestedContext,
            kvQuant: kvQuant, concurrency: concurrency, thresholds: thresholds,
            allowedKVTiers: allowedKVTiers, allowContextCapping: allowContextCapping,
            preference: preference)

        return QuantResolution(
            pick: pick,
            winnerDirectory: pick.winnerRepoID.flatMap { directoryByID[$0] },
            winnerParsed: pick.winnerRepoID.flatMap { parsedByID[$0] })
    }
}
