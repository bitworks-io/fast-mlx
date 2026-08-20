import HarnessCore
import ServingCore

/// MLX-free seam that pairs a model's serving-side snapshot-reuse classification with the cold-plane
/// restore arithmetic, so a consumer physically cannot apply the wrong (dense) block-floor math to a
/// layout that can't support it.
///
/// `ServingCore.ScalarServingSnapshotReuse` (classified from the native cache layout by
/// `snapshotReuseGranularity(for:)`) and `HarnessCore.ColdSnapshotReuseGranularity` (consumed by
/// `ExactPrefixColdSnapshotPlanner.planRestore`) are deliberately parallel enums declared in two
/// zero-dependency targets that don't import each other. Their doc-comments say "a future consumer
/// maps between them"; today that mapping lives nowhere, so the two planes can silently drift — e.g. a
/// new cache kind, or a consumer defaulting a recurrent model to `.blockAligned`, would produce exactly
/// the wrong-arithmetic failure the `.wholeSnapshotOnly` rule exists to prevent. This target is the one
/// tested, fail-closed mapping, mirroring `ServingSamplingBridge`: it imports both pure targets and
/// lives in its own so both stay dependency-free and the bridge builds+runs off-box with `swift test`.
public enum ServingSnapshotBridge {
    /// Map the serving-side reuse classification to the cold-plane restore granularity. Total on the
    /// two supported cases; `.unsupported` FAILS CLOSED (throws) rather than silently defaulting to
    /// `.blockAligned` — an unaudited or mixed layout has no restore arithmetic we can prove correct,
    /// so a consumer must not be handed one. The `switch` is exhaustive, so a new
    /// `ScalarServingSnapshotReuse` case breaks compilation here rather than drifting silently.
    public static func coldSnapshotGranularity(
        for reuse: ScalarServingSnapshotReuse
    ) throws -> ColdSnapshotReuseGranularity {
        switch reuse {
        case .blockAligned:
            return .blockAligned
        case .wholeSnapshotOnly:
            return .wholeSnapshotOnly
        case .unsupported:
            throw ServingSnapshotBridgeError.unsupportedSnapshotReuseLayout
        }
    }

    /// Compose the whole safe path for restoring ONE stored snapshot: classify the model's native
    /// cache layout, map it to the restore granularity, then run `planRestore`. A consumer that goes
    /// through this overload cannot pair a layout with the wrong arithmetic — an `.unsupported` layout
    /// throws before any reuse is planned. Fail-closed inputs (`blockSize <= 0`, a non-whole-block
    /// stored snapshot) surface as the planner's own `ExactPrefixColdSnapshotError`.
    public static func planRestore(
        storedPrefixTokens: [Int], promptTokens: [Int], blockSize: Int,
        cacheKinds: [ScalarServingNativeCacheKind]
    ) throws -> ColdSnapshotRestorePlan {
        let granularity = try coldSnapshotGranularity(for: snapshotReuseGranularity(for: cacheKinds))
        return try ExactPrefixColdSnapshotPlanner.planRestore(
            storedPrefixTokens: storedPrefixTokens, promptTokens: promptTokens, blockSize: blockSize,
            granularity: granularity)
    }

    /// Compose the safe path for choosing among SEVERAL stored snapshots for one key: classify + map,
    /// then run `planBestRestore`. Same fail-closed guarantee — an `.unsupported` layout throws before
    /// any candidate is considered, and a corrupt candidate fails the whole selection.
    public static func planBestRestore(
        candidateStoredPrefixes: [[Int]], promptTokens: [Int], blockSize: Int,
        cacheKinds: [ScalarServingNativeCacheKind]
    ) throws -> ColdSnapshotRestoreSelection? {
        let granularity = try coldSnapshotGranularity(for: snapshotReuseGranularity(for: cacheKinds))
        return try ExactPrefixColdSnapshotPlanner.planBestRestore(
            candidateStoredPrefixes: candidateStoredPrefixes, promptTokens: promptTokens,
            blockSize: blockSize, granularity: granularity)
    }

    /// Config-plane SAFE reuse policy: predict the snapshot-reuse granularity for a decoded
    /// `ArchClass` WITHOUT a live cache instance, so the off-box plane can plan restore before load.
    /// FAIL-CLOSED — returns a non-`.unsupported` granularity ONLY for arch classes whose
    /// cold-restore arithmetic is audited safe. The `switch` is exhaustive over `ArchClass`, so a new
    /// arch breaks compilation here.
    public static func predictedSnapshotReuse(for archClass: ArchClass) -> ScalarServingSnapshotReuse {
        switch archClass {
        case .uniformGQA:
            // All-dense StandardKVCache; agrees with the live classifier.
            return .blockAligned
        case .hybridLinear:
            // MambaCache (recurrent) + dense; audited (Qwen3.5/3.6); agrees with the live classifier.
            return .wholeSnapshotOnly
        case .interleavedSWA:
            // RotatingKVCache windows can't restore at an arbitrary boundary; agrees with the live
            // classifier (.rotatingAttention -> .unsupported).
            return .unsupported
        case .dualGeometrySWA:
            // Same RotatingKVCache window constraint as .interleavedSWA (plus a per-layer recurrent
            // conv state); can't restore at an arbitrary boundary. Fail closed.
            return .unsupported
        case .mlaAsImplemented:
            // Plausibly block-aligned as implemented (decompressed dense K/V), but its cold-restore
            // arithmetic is NOT audited off-box; fail closed pending a live cache-kind audit on the
            // M5. This is a DELIBERATE conservative divergence from what a dense-only live layout
            // would classify — safe direction (refuses reuse; never applies wrong arithmetic).
            return .unsupported
        case .hybridMamba2MoE:
            // Unaudited family; fail closed.
            return .unsupported
        case .novelCompressedUnsupported:
            // Unsupported by definition.
            return .unsupported
        }
    }

    /// Operator-facing serve-startup announce for the PREDICTED snapshot-reuse granularity of this
    /// arch class, computed off-box from config alone (no live cache instance). Evidence-labeled
    /// `predicted`: it is the pre-load prediction that the live cache-kind classifier confirms after
    /// load — never presented as a measured fact. Delegates to `predictedSnapshotReuse(for:)`, so the
    /// announced value and the planner's fail-closed policy can never diverge.
    public static func snapshotReuseAnnounceLine(for archClass: ArchClass) -> String {
        "snapshot_reuse=\(predictedSnapshotReuse(for: archClass).rawValue) predicted "
            + "arch_class=\(archClass.rawValue)"
    }
}

/// Bridge-level rejection raised when a serving reuse classification cannot be mapped to a cold-plane
/// restore granularity.
public enum ServingSnapshotBridgeError: Error, Equatable, Sendable {
    /// The layout classified as `.unsupported` (rotating/composite/unknown/empty) — no audited restore
    /// arithmetic exists for it, so the bridge refuses to invent one.
    case unsupportedSnapshotReuseLayout
}
