import Foundation

enum DenseKVByteAdmissionPlanError: Error, Equatable {
    case invalidGeometry
    case invalidRequest(promptTokens: Int, outputTokens: Int)
    case invalidCapacity(Int)
    case arithmeticOverflow
}

/// Exact dense-Qwen KV array geometry plus a conservative membership-transition envelope.
///
/// Each capacity unit represents one token across every layer's K and V buffers. Once rows have
/// shared a batch, extraction preserves that batch's largest padded capacity even for shorter
/// requests. A rebuild can temporarily retain five full padded copies: the old batch,
/// extracted scalar rows, tail-zero inputs, padded-row concatenations, and the final batch.
/// The envelope intentionally over-reserves initial joins so later membership changes cannot
/// exceed admission without a new request.
struct DenseKVByteAdmissionPlan: Sendable, Equatable {
    let bytesPerToken: Int
    let metadataBytesPerRow: Int
    let allocationChunk: Int
    let maxContextTokens: Int

    init(
        layerCount: Int,
        keyValueHeadCount: Int,
        headDimension: Int,
        elementBytes: Int,
        allocationChunk: Int,
        maxContextTokens: Int
    ) throws {
        guard layerCount > 0, keyValueHeadCount > 0, headDimension > 0,
            elementBytes > 0, allocationChunk > 0, maxContextTokens > 0,
            allocationChunk <= maxContextTokens
        else {
            throw DenseKVByteAdmissionPlanError.invalidGeometry
        }
        var bytes = layerCount
        for factor in [keyValueHeadCount, headDimension, 2, elementBytes] {
            let (next, overflow) = bytes.multipliedReportingOverflow(by: factor)
            guard !overflow else { throw DenseKVByteAdmissionPlanError.arithmeticOverflow }
            bytes = next
        }
        self.bytesPerToken = bytes
        let (metadataWords, metadataWordOverflow) = layerCount.addingReportingOverflow(1)
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
        let (kvBytes, byteOverflow) = totalUnits.multipliedReportingOverflow(by: bytesPerToken)
        guard !byteOverflow else { throw DenseKVByteAdmissionPlanError.arithmeticOverflow }
        // Per-row metadata is one int32 offset per layer plus one int32 staged token. Carry it
        // under the same five-copy transition envelope as the large K/V arrays.
        let (metadataRows, metadataRowOverflow) = capacities.count.multipliedReportingOverflow(by: 5)
        guard !metadataRowOverflow else {
            throw DenseKVByteAdmissionPlanError.arithmeticOverflow
        }
        let (metadataBytes, metadataOverflow) = metadataRows.multipliedReportingOverflow(
            by: metadataBytesPerRow)
        guard !metadataOverflow else {
            throw DenseKVByteAdmissionPlanError.arithmeticOverflow
        }
        let (totalBytes, finalOverflow) = kvBytes.addingReportingOverflow(metadataBytes)
        guard !finalOverflow else { throw DenseKVByteAdmissionPlanError.arithmeticOverflow }
        return totalBytes
    }
}
