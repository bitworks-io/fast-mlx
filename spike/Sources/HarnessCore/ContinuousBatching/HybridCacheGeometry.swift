import Foundation

/// S2 of the hybrid continuous-batching build (design of record: the continuous-batching
/// heterogeneous-cache design). Attaches the
/// per-layer cache GEOMETRY that S1's `HybridLayerKindMap` deliberately omitted (structure only). Pure
/// value types — no MLX, no filesystem; derived fail-closed from `config.json` by
/// `ModelConfigDecoder.qwen35HybridGeometry`.
///
/// Geometry is a SEPARATE composed type, not payloads on the `LayerCacheKind` enum: for qwen3_5 every
/// dense layer shares one `DenseKVGeometry` and every recurrent layer shares one `RecurrentStateGeometry`,
/// so per-case payloads would be N redundant copies and would entangle structural equality with geometry.
/// See docs/task-inbox/2026-08-20-continuous-batching-s2-geometry-plan.md for the decision record.

/// The growing K/V geometry of a full-attention layer.
public struct DenseKVGeometry: Equatable, Sendable {
    /// Number of key/value heads (GQA `num_key_value_heads`).
    public let kvHeads: Int
    /// Per-head dimension (`head_dim`).
    public let headDim: Int
    /// Element width of the stored K/V (2 for fp16/bf16, 4 for fp32).
    public let elementBytes: Int

    public init(kvHeads: Int, headDim: Int, elementBytes: Int) {
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.elementBytes = elementBytes
    }

    /// Bytes added per token per layer: K and V, each `kvHeads × headDim × elementBytes`.
    public var bytesPerLayerPerToken: Int { 2 * kvHeads * headDim * elementBytes }
}

/// The fixed-size recurrent state of a GatedDeltaNet linear-attention layer (qwen3_5 / qwen3_next),
/// reconciled against the vendored MLX-Swift `MambaCache` allocation:
///   - conv state `[B, convKernelSize-1, convDim]` at activation precision (`convElementBytes`),
///     `Qwen35.swift:245,254`.
///   - SSM state `[B, valueHeads, valueHeadDim, keyHeadDim]` **fp32 by construction**
///     (`GatedDelta.swift:243,296-298`) — 4 bytes, never quantized, independent of the compute dtype or
///     any KV-quant tier. Exposed as the constant `ssmStateElementBytes`, NOT a settable field, so no
///     caller can misconfigure it.
public struct RecurrentStateGeometry: Equatable, Sendable {
    /// `linear_conv_kernel_dim`.
    public let convKernelSize: Int
    /// `2·(linear_key_head_dim·linear_num_key_heads) + (linear_value_head_dim·linear_num_value_heads)`
    /// (`Qwen35.swift:195`). Carried directly (rather than re-derived) so the derivation is the single
    /// authority for the transposable formula.
    public let convDim: Int
    /// `linear_num_value_heads` (Hv).
    public let valueHeads: Int
    /// `linear_value_head_dim` (Dv).
    public let valueHeadDim: Int
    /// `linear_key_head_dim` (Dk).
    public let keyHeadDim: Int
    /// Activation precision of the conv tail (2 for fp16/bf16).
    public let convElementBytes: Int

    /// SSM state element width: fp32 (4 B) by vendored construction, not configurable.
    public static let ssmStateElementBytes: Int = 4

    public init(
        convKernelSize: Int, convDim: Int, valueHeads: Int, valueHeadDim: Int,
        keyHeadDim: Int, convElementBytes: Int
    ) {
        self.convKernelSize = convKernelSize
        self.convDim = convDim
        self.valueHeads = valueHeads
        self.valueHeadDim = valueHeadDim
        self.keyHeadDim = keyHeadDim
        self.convElementBytes = convElementBytes
    }

    /// Conv-tail bytes for one layer: `(convKernelSize-1) × convDim × convElementBytes`.
    public var convBytesPerLayer: Int { (convKernelSize - 1) * convDim * convElementBytes }

    /// SSM-state bytes for one layer: `valueHeads × valueHeadDim × keyHeadDim × 4` (fp32).
    public var ssmBytesPerLayer: Int {
        valueHeads * valueHeadDim * keyHeadDim * Self.ssmStateElementBytes
    }

    /// Total fixed recurrent state for one layer (conv tail + SSM state). This mirrors the audited
    /// per-layer term inside `ModelConfigDecoder.resolveHybridFixedStateBytes` (MCD.swift:1042-1047);
    /// the sizer-agreement test pins the two in lockstep.
    public var bytesPerLayer: Int { convBytesPerLayer + ssmBytesPerLayer }
}

/// A structural kind map plus the two shared geometries — the whole per-layer cache description the S2
/// `.hybridFP16` family case and the S3 runtime consume.
public struct HybridCacheGeometry: Equatable, Sendable {
    public let map: HybridLayerKindMap
    public let dense: DenseKVGeometry
    public let recurrent: RecurrentStateGeometry

    /// Fails (nil) when `map` is not heterogeneous: an all-dense map is honestly a uniform model and must
    /// route the existing `.fp16` path, never `.hybridFP16` (mirrors `HybridLayerKindMap.isHeterogeneous`).
    public init?(map: HybridLayerKindMap, dense: DenseKVGeometry, recurrent: RecurrentStateGeometry) {
        guard map.isHeterogeneous else { return nil }
        self.map = map
        self.dense = dense
        self.recurrent = recurrent
    }

    /// Per-layer cache kind (passthrough to the structural map).
    public func kind(atLayer index: Int) -> LayerCacheKind { map.kind(atLayer: index) }

    /// Total fixed recurrent state across all linear layers — the per-sequence constant term the hybrid
    /// capacity accounting adds on top of the growing dense KV.
    public var recurrentBytesTotal: Int {
        recurrent.bytesPerLayer * map.recurrentLayerIndices.count
    }
}
