import Foundation

enum DenseKVByteAdmissionPlanError: Error, Equatable {
    case invalidGeometry
    case invalidRequest(promptTokens: Int, outputTokens: Int)
    case invalidCapacity(Int)
    case arithmeticOverflow
}

protocol ContinuousKVByteAdmissionPlanning {
    var bytesPerToken: Int { get }
    func reservedCapacity(promptTokens: Int, outputTokens: Int) throws -> Int
    func transitionEnvelopeBytes(capacities: [Int]) throws -> Int
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

extension DenseKVByteAdmissionPlan: ContinuousKVByteAdmissionPlanning {}

/// Exact affine payload/metadata bytes plus a conservative membership-transition envelope.
///
/// Unlike dense accounting, every layer can have independent K/V bits and group sizes. The
/// envelope also carries one logical offset per row/layer, one staged token per row, and one
/// shared physical-end word per layer under the same five-copy transition bound.
struct AffineKVByteAdmissionPlan: Sendable, Equatable,
    ContinuousKVByteAdmissionPlanning
{
    let bytesPerToken: Int
    let layerCount: Int
    let allocationChunk: Int
    let maxContextTokens: Int

    init(
        configurations: [AffineKVCacheConfiguration],
        keyValueHeadCount: Int,
        headDimension: Int,
        metadataScalarBytes: Int,
        allocationChunk: Int,
        maxContextTokens: Int
    ) throws {
        guard !configurations.isEmpty,
            keyValueHeadCount > 0,
            headDimension > 0,
            metadataScalarBytes > 0,
            allocationChunk > 0,
            maxContextTokens > 0,
            allocationChunk <= maxContextTokens
        else {
            throw DenseKVByteAdmissionPlanError.invalidGeometry
        }

        var totalBytesPerToken = 0
        for configuration in configurations {
            let (keyPackedBits, keyBitsOverflow) = headDimension
                .multipliedReportingOverflow(by: configuration.keyBits)
            let (valuePackedBits, valueBitsOverflow) = headDimension
                .multipliedReportingOverflow(by: configuration.valueBits)
            guard !keyBitsOverflow, !valueBitsOverflow else {
                throw DenseKVByteAdmissionPlanError.arithmeticOverflow
            }
            guard headDimension.isMultiple(of: configuration.keyGroupSize),
                headDimension.isMultiple(of: configuration.valueGroupSize),
                keyPackedBits.isMultiple(of: 8),
                valuePackedBits.isMultiple(of: 8)
            else {
                throw DenseKVByteAdmissionPlanError.invalidGeometry
            }

            func multiply(_ factors: [Int]) throws -> Int {
                var result = 1
                for factor in factors {
                    let (next, overflow) = result.multipliedReportingOverflow(by: factor)
                    guard !overflow else {
                        throw DenseKVByteAdmissionPlanError.arithmeticOverflow
                    }
                    result = next
                }
                return result
            }

            let keyPayload = try multiply([
                keyValueHeadCount,
                keyPackedBits / 8,
            ])
            let valuePayload = try multiply([
                keyValueHeadCount,
                valuePackedBits / 8,
            ])
            let keyMetadata = try multiply([
                keyValueHeadCount,
                headDimension / configuration.keyGroupSize,
                2,
                metadataScalarBytes,
            ])
            let valueMetadata = try multiply([
                keyValueHeadCount,
                headDimension / configuration.valueGroupSize,
                2,
                metadataScalarBytes,
            ])
            for component in [keyPayload, valuePayload, keyMetadata, valueMetadata] {
                let (next, overflow) = totalBytesPerToken.addingReportingOverflow(component)
                guard !overflow else {
                    throw DenseKVByteAdmissionPlanError.arithmeticOverflow
                }
                totalBytesPerToken = next
            }
        }

        self.bytesPerToken = totalBytesPerToken
        self.layerCount = configurations.count
        self.allocationChunk = allocationChunk
        self.maxContextTokens = maxContextTokens
    }

    func reservedCapacity(promptTokens: Int, outputTokens: Int) throws -> Int {
        guard promptTokens > 0, outputTokens > 0 else {
            throw DenseKVByteAdmissionPlanError.invalidRequest(
                promptTokens: promptTokens,
                outputTokens: outputTokens)
        }
        let (total, totalOverflow) = promptTokens.addingReportingOverflow(outputTokens)
        guard !totalOverflow, total <= maxContextTokens else {
            throw DenseKVByteAdmissionPlanError.invalidRequest(
                promptTokens: promptTokens,
                outputTokens: outputTokens)
        }
        let (rounding, roundingOverflow) = total.addingReportingOverflow(allocationChunk - 1)
        guard !roundingOverflow else {
            throw DenseKVByteAdmissionPlanError.arithmeticOverflow
        }
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
        let (batchTokens, batchOverflow) = maxCapacity.multipliedReportingOverflow(
            by: capacities.count)
        guard !batchOverflow else {
            throw DenseKVByteAdmissionPlanError.arithmeticOverflow
        }
        let (transitionTokens, transitionOverflow) = batchTokens
            .multipliedReportingOverflow(by: 5)
        guard !transitionOverflow else {
            throw DenseKVByteAdmissionPlanError.arithmeticOverflow
        }
        let (dataBytes, dataOverflow) = transitionTokens.multipliedReportingOverflow(
            by: bytesPerToken)
        guard !dataOverflow else {
            throw DenseKVByteAdmissionPlanError.arithmeticOverflow
        }

        let (rowControlWords, rowOverflow) = layerCount.addingReportingOverflow(1)
        guard !rowOverflow else {
            throw DenseKVByteAdmissionPlanError.arithmeticOverflow
        }
        let (allRowWords, allRowOverflow) = capacities.count.multipliedReportingOverflow(
            by: rowControlWords)
        guard !allRowOverflow else {
            throw DenseKVByteAdmissionPlanError.arithmeticOverflow
        }
        let (controlWords, controlOverflow) = allRowWords.addingReportingOverflow(layerCount)
        guard !controlOverflow else {
            throw DenseKVByteAdmissionPlanError.arithmeticOverflow
        }
        let (transitionControlWords, transitionControlOverflow) = controlWords
            .multipliedReportingOverflow(by: 5)
        guard !transitionControlOverflow else {
            throw DenseKVByteAdmissionPlanError.arithmeticOverflow
        }
        let (controlBytes, controlByteOverflow) = transitionControlWords
            .multipliedReportingOverflow(by: 4)
        guard !controlByteOverflow else {
            throw DenseKVByteAdmissionPlanError.arithmeticOverflow
        }
        let (total, totalOverflow) = dataBytes.addingReportingOverflow(controlBytes)
        guard !totalOverflow else {
            throw DenseKVByteAdmissionPlanError.arithmeticOverflow
        }
        return total
    }
}
