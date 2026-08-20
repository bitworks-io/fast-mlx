import Foundation

/// The per-layer cache KIND for a decoder layer — the structural core of continuous batching over a
/// heterogeneous (hybrid-linear) arch, where dense-attention KV layers are interleaved with fixed-size
/// recurrent-state (GatedDeltaNet/Mamba) layers.
///
/// Geometry payloads (KV head/dim bytes, recurrent conv/SSM state shape) are DELIBERATELY not carried
/// here yet: stage S1 (this type) pins only *which layer is which kind* and the invariants that follow
/// from that structure. The cache-family stage (S2) attaches geometry when it actually allocates caches.
/// See the continuous-batching heterogeneous-cache design.
public enum LayerCacheKind: Equatable, Sendable {
    /// A full-attention layer: a growing K/V cache, batchable by sequence-axis concatenation.
    case denseAttention
    /// A linear/recurrent layer: a fixed-size recurrent state (conv tail + SSM state), length-independent.
    case recurrentState

    public var isDenseAttention: Bool { self == .denseAttention }
    public var isRecurrentState: Bool { self == .recurrentState }
}

/// One `LayerCacheKind` per decoder layer, in layer order. Pure value type — no MLX, no filesystem.
///
/// This is the map the continuous-batch runtime consults to dispatch per-layer cache handling and to
/// key its capacity/geometry accounting on the RIGHT layer. The load-bearing landmine it exists to
/// prevent: for the qwen3_5 family (`full_attention_interval` default 4, attention layer where
/// `(i+1) % interval == 0`), **layer 0 is a recurrent layer, not attention** — so any code that keys KV
/// capacity/geometry off "layer 0" silently breaks. `firstDenseLayerIndex` is the correct key and is
/// pinned by tests.
public struct HybridLayerKindMap: Equatable, Sendable {
    /// Per-layer kinds, `kinds.count == num_hidden_layers`.
    public let kinds: [LayerCacheKind]

    /// Construct from an explicit per-layer kind list (`count` must be > 0).
    public init(kinds: [LayerCacheKind]) {
        precondition(!kinds.isEmpty, "HybridLayerKindMap requires at least one layer")
        self.kinds = kinds
    }

    /// Number of decoder layers.
    public var layerCount: Int { kinds.count }

    /// Indices of the dense-attention (growing-KV) layers, ascending.
    public var denseLayerIndices: [Int] {
        kinds.indices.filter { kinds[$0].isDenseAttention }
    }

    /// Indices of the recurrent-state (fixed) layers, ascending.
    public var recurrentLayerIndices: [Int] {
        kinds.indices.filter { kinds[$0].isRecurrentState }
    }

    /// The FIRST dense-attention layer — the correct key for KV capacity/geometry accounting (layer 0
    /// is NOT dense on qwen3_5). `nil` iff the model has no attention layers at all (a pure-recurrent
    /// model, which this hybrid runtime does not target — callers fail closed on nil).
    public var firstDenseLayerIndex: Int? { denseLayerIndices.first }

    /// True when at least one layer is recurrent — i.e. this genuinely needs the heterogeneous path
    /// rather than the uniform dense-KV path. A map with no recurrent layers is honestly just a dense
    /// model and should route through the existing uniform family, not `.hybridFP16`.
    public var isHeterogeneous: Bool { !recurrentLayerIndices.isEmpty }

    /// The kind of a given layer (traps out-of-range as a programmer error, like `Array` subscript).
    public func kind(atLayer index: Int) -> LayerCacheKind { kinds[index] }
}

public extension HybridLayerKindMap {
    /// Derive the qwen3_5 / qwen3_5_text / qwen3_next family layer pattern: a layer is full-attention
    /// when `(layerIndex + 1) % fullAttentionInterval == 0`, else recurrent (GatedDeltaNet linear). This
    /// mirrors the vendored constructor exactly (`Qwen35.swift`: `isLinear = (layerIdx + 1) % interval
    /// != 0`, interval default 4), so the derived map agrees with the model's own `newCache` layout.
    ///
    /// Fails closed on non-positive inputs (an invalid config must never yield a silently-empty map).
    static func qwen35(layerCount: Int, fullAttentionInterval: Int = 4) -> HybridLayerKindMap? {
        guard layerCount > 0, fullAttentionInterval > 0 else { return nil }
        let kinds: [LayerCacheKind] = (0..<layerCount).map { i in
            (i + 1) % fullAttentionInterval == 0 ? .denseAttention : .recurrentState
        }
        return HybridLayerKindMap(kinds: kinds)
    }

    /// Build a map from an explicit set of attention-layer indices (the shape the interval-select and
    /// index-list hybrid families resolve to — e.g. `ModelConfigDecoder`'s `resolveHybridAttnLayers`,
    /// which reads `layer_types` / `full_attn_idxs` / offset+period across lfm2/jamba/granite/etc.). Any
    /// layer in `attentionLayerIndices` is dense; every other in-range layer is recurrent. This is the
    /// general seam the S2/S3 decoder integration threads through, so the kind map never re-derives a
    /// family-specific pattern that could drift from the decoder's already-audited resolution.
    ///
    /// Fails closed if `layerCount <= 0` or any attention index is out of `0..<layerCount`.
    static func from(attentionLayerIndices: Set<Int>, layerCount: Int) -> HybridLayerKindMap? {
        guard layerCount > 0 else { return nil }
        guard attentionLayerIndices.allSatisfy({ (0..<layerCount).contains($0) }) else { return nil }
        let kinds: [LayerCacheKind] = (0..<layerCount).map {
            attentionLayerIndices.contains($0) ? .denseAttention : .recurrentState
        }
        return HybridLayerKindMap(kinds: kinds)
    }
}
