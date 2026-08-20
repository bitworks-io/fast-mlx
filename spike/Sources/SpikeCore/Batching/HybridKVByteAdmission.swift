import Foundation

import HarnessCore

/// Byte-admission plan for a heterogeneous (hybrid dense/recurrent) cache geometry.
///
/// A hybrid model interleaves growing dense-attention KV layers with fixed-size recurrent-state
/// (GatedDeltaNet/Mamba) layers. Two accounting choices are load-bearing here:
///
/// 1. `bytesPerToken` is computed from the DENSE layers only, never `geometry.map.kinds.count`.
///    Multiplying by the total layer count would charge every admitted token as if it grew a
///    K/V buffer on every layer, over-charging by roughly the dense:total layer ratio inverse
///    (e.g. ~4x for a qwen3_5-style 1-in-4 attention pattern) and starving admission of headroom
///    it doesn't need. Recurrent layers carry no growing per-token state, so they contribute
///    nothing to the per-token term.
/// 2. The recurrent state is fixed-size per row (it does not grow with tokens), so it is charged
///    separately as `recurrentBytesPerRow` in `transitionEnvelopeBytes`, under the same
///    conservative five-copy transition envelope as the dense term, and NEVER dropped. This is
///    deliberately fail-closed: `ModelConfigDecoder.resolveHybridFixedStateBytes` (the sizer) is
///    allowed to fail OPEN with 0 when it cannot resolve geometry, because the sizer is advisory.
///    Admission-grade accounting has no such license — under-charging here means admitting more
///    concurrent rows than fit, which is a correctness violation, not an advisory miss.
struct HybridKVByteAdmissionPlan: Sendable, Equatable, ContinuousKVByteAdmissionPlanning {
    let bytesPerToken: Int
    let recurrentBytesPerRow: Int
    let metadataBytesPerRow: Int
    let allocationChunk: Int
    let maxContextTokens: Int

    init(
        geometry: HybridCacheGeometry,
        allocationChunk: Int,
        maxContextTokens: Int
    ) throws {
        let denseCount = geometry.map.denseLayerIndices.count
        let recurrentCount = geometry.map.recurrentLayerIndices.count
        guard denseCount > 0, recurrentCount > 0,
            geometry.dense.kvHeads > 0, geometry.dense.headDim > 0, geometry.dense.elementBytes > 0,
            allocationChunk > 0, maxContextTokens > 0,
            allocationChunk <= maxContextTokens
        else {
            throw DenseKVByteAdmissionPlanError.invalidGeometry
        }

        var bytes = denseCount
        for factor in [2, geometry.dense.kvHeads, geometry.dense.headDim, geometry.dense.elementBytes] {
            let (next, overflow) = bytes.multipliedReportingOverflow(by: factor)
            guard !overflow else { throw DenseKVByteAdmissionPlanError.arithmeticOverflow }
            bytes = next
        }
        self.bytesPerToken = bytes

        self.recurrentBytesPerRow = geometry.recurrentBytesTotal

        let (metadataWords, metadataWordOverflow) = denseCount.addingReportingOverflow(1)
        guard !metadataWordOverflow else {
            throw DenseKVByteAdmissionPlanError.arithmeticOverflow
        }
        let (metadataBytes, metadataOverflow) = metadataWords.multipliedReportingOverflow(by: 4)
        guard !metadataOverflow else {
            throw DenseKVByteAdmissionPlanError.arithmeticOverflow
        }
        self.metadataBytesPerRow = metadataBytes

        self.allocationChunk = allocationChunk
        self.maxContextTokens = maxContextTokens
    }

    func reservedCapacity(promptTokens: Int, outputTokens: Int) throws -> Int {
        guard promptTokens > 0, outputTokens > 0 else {
            throw DenseKVByteAdmissionPlanError.invalidRequest(
                promptTokens: promptTokens, outputTokens: outputTokens)
        }
        let (total, totalOverflow) = promptTokens.addingReportingOverflow(outputTokens)
        guard !totalOverflow, total <= maxContextTokens else {
            throw DenseKVByteAdmissionPlanError.invalidRequest(
                promptTokens: promptTokens, outputTokens: outputTokens)
        }
        let (rounding, roundingOverflow) = total.addingReportingOverflow(allocationChunk - 1)
        guard !roundingOverflow else { throw DenseKVByteAdmissionPlanError.arithmeticOverflow }
        return min(
            maxContextTokens,
            max(allocationChunk, (rounding / allocationChunk) * allocationChunk))
    }

    func transitionEnvelopeBytes(capacities: [Int]) throws -> Int {
        guard !capacities.isEmpty else { return 0 }
        guard capacities.allSatisfy({ $0 > 0 && $0 <= maxContextTokens }) else {
            throw DenseKVByteAdmissionPlanError.invalidCapacity(
                capacities.first(where: { $0 <= 0 || $0 > maxContextTokens })!)
        }
        let maxCapacity = capacities.max()!

        let (batchUnits, batchOverflow) = maxCapacity.multipliedReportingOverflow(
            by: capacities.count)
        guard !batchOverflow else { throw DenseKVByteAdmissionPlanError.arithmeticOverflow }
        let (totalUnits, totalOverflow) = batchUnits.multipliedReportingOverflow(by: 5)
        guard !totalOverflow else { throw DenseKVByteAdmissionPlanError.arithmeticOverflow }
        let (denseData, denseDataOverflow) = totalUnits.multipliedReportingOverflow(
            by: bytesPerToken)
        guard !denseDataOverflow else { throw DenseKVByteAdmissionPlanError.arithmeticOverflow }

        let (metadataRows, metadataRowOverflow) = capacities.count.multipliedReportingOverflow(
            by: 5)
        guard !metadataRowOverflow else {
            throw DenseKVByteAdmissionPlanError.arithmeticOverflow
        }
        let (denseMetadata, denseMetadataOverflow) = metadataRows.multipliedReportingOverflow(
            by: metadataBytesPerRow)
        guard !denseMetadataOverflow else {
            throw DenseKVByteAdmissionPlanError.arithmeticOverflow
        }

        // The recurrent state is fixed-size, but a merge/extract transition can transiently hold
        // multiple copies at once (old batch, extracted rows, new batch), so it is bounded by the
        // same conservative five-copy envelope, charged per row, and never dropped.
        let (recurrentRows, recurrentRowOverflow) = capacities.count.multipliedReportingOverflow(
            by: 5)
        guard !recurrentRowOverflow else {
            throw DenseKVByteAdmissionPlanError.arithmeticOverflow
        }
        let (recurrentEnvelope, recurrentEnvelopeOverflow) = recurrentRows
            .multipliedReportingOverflow(by: recurrentBytesPerRow)
        guard !recurrentEnvelopeOverflow else {
            throw DenseKVByteAdmissionPlanError.arithmeticOverflow
        }

        let (dataPlusMetadata, dataPlusMetadataOverflow) = denseData.addingReportingOverflow(
            denseMetadata)
        guard !dataPlusMetadataOverflow else {
            throw DenseKVByteAdmissionPlanError.arithmeticOverflow
        }
        let (total, totalEnvelopeOverflow) = dataPlusMetadata.addingReportingOverflow(
            recurrentEnvelope)
        guard !totalEnvelopeOverflow else {
            throw DenseKVByteAdmissionPlanError.arithmeticOverflow
        }
        return total
    }
}
