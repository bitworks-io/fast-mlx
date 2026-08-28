import HarnessCore
import MLXLMCommon
import ServingCore

/// The KV storage a route should actually build for a requested tier.
public enum KVCacheQuantDecision: Equatable, Sendable {
    /// Build native fp16 caches — the runtime's unchanged, always-valid default.
    case fp16
    /// Wrap dense native caches in a `QuantizedKVCache` with the given group size / bit width.
    case int8(groupSize: Int, bits: Int)
}

public enum KVQuantSelectionError: Error, CustomStringConvertible, Equatable {
    /// The requested tier is not in the runtime-wired set — the serving runtime cannot actually
    /// store this format today, regardless of route shape.
    case tierNotRuntimeWired(KVQuantTier)
    /// The runtime has a construction capability, but the tier has not passed the quality gate and
    /// therefore cannot be selected for a live route.
    case tierNotQualityApproved(KVQuantTier)
    /// The route's native cache kinds include at least one non-dense-attention cache, which the
    /// requested tier's quantized wrapper cannot apply to.
    case tierIncompatibleWithRoute(tier: KVQuantTier, kinds: [ScalarServingNativeCacheKind])
    /// The route classified zero native caches — nothing to quantize, so there is nothing to
    /// validate compatibility against; fail closed rather than assume compatibility.
    case emptyCacheRoute(KVQuantTier)

    public var description: String {
        switch self {
        case .tierNotRuntimeWired(let tier):
            return "KV tier \(tier) is not runtime-wired; the serving runtime cannot store it yet"
        case .tierNotQualityApproved(let tier):
            return "KV tier \(tier) is not quality-approved; the serving runtime refuses to apply it"
        case .tierIncompatibleWithRoute(let tier, let kinds):
            return "KV tier \(tier) requires all-dense-attention native caches; route caches were \(kinds)"
        case .emptyCacheRoute(let tier):
            return "KV tier \(tier) requested against a route with zero classified native caches"
        }
    }
}

/// Pure fail-closed decision: which KV storage to build for `requested`, given the route's
/// classified native cache kinds. Never silently downgrades; throws rather than serve a tier the
/// runtime can't honor or a route int8 can't apply to. Inert for int8 today because
/// runtimeWiredKVTiers == [.fp16] (int8 always throws tierNotRuntimeWired) — activates only after a
/// dated quality gate flips runtimeWiredKVTiers. groupSize 32 / bits 8 matches mlx-swift.
public func selectKVCacheQuant(
    requested: KVQuantTier,
    nativeKinds: [ScalarServingNativeCacheKind],
    runtimeWired: Set<KVQuantTier> = Set(ServeTierPolicy.runtimeWiredKVTiers),
    qualityApproved: Set<KVQuantTier> = Set(ServeTierPolicy.qualityApprovedKVTiers)
) throws -> KVCacheQuantDecision {
    switch requested {
    case .fp16:
        return .fp16
    case .int8:
        guard runtimeWired.contains(.int8) else {
            throw KVQuantSelectionError.tierNotRuntimeWired(.int8)
        }
        guard qualityApproved.contains(.int8) else {
            throw KVQuantSelectionError.tierNotQualityApproved(.int8)
        }
        guard !nativeKinds.isEmpty else {
            throw KVQuantSelectionError.emptyCacheRoute(.int8)
        }
        guard nativeKinds.allSatisfy({ $0 == .denseAttention }) else {
            throw KVQuantSelectionError.tierIncompatibleWithRoute(tier: .int8, kinds: nativeKinds)
        }
        return .int8(groupSize: 32, bits: 8)
    case .turbo4, .tq2_5, .tq3_5:
        throw KVQuantSelectionError.tierNotRuntimeWired(requested)
    }
}

/// Build the actual KV caches a route serves from a `selectKVCacheQuant` decision.
///
/// `.fp16` returns the native caches UNCHANGED (identity — the same instances, not copies), so this is
/// a provable no-op in front of every route today: `runtimeWiredKVTiers == [.fp16]` makes int8
/// unreachable at selection, and the hybrid recurrent route is additionally int8-incompatible there.
/// The `.int8` branch builds one vendored `QuantizedKVCache(groupSize:bits:mode:.affine)` per native
/// cache — the measured int8 runtime path (`KVCache.swift`). That branch is exercised by unit tests and
/// remains production-inert until a dated quality PASS. It does NOT quantize recurrent state — selection
/// rejects int8 on any non-dense route, so every element handed here is a dense attention cache.
public func buildRouteKVCaches(
    decision: KVCacheQuantDecision,
    nativeCaches: [any KVCache]
) -> [any KVCache] {
    switch decision {
    case .fp16:
        return nativeCaches
    case .int8(let groupSize, let bits):
        return nativeCaches.map { _ in
            QuantizedKVCache(groupSize: groupSize, bits: bits, mode: .affine)
        }
    }
}
