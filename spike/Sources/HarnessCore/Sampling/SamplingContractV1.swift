import CryptoKit
import Foundation

enum SamplingContractFailure: Error, Equatable, Sendable {
    enum NonFiniteKind: Equatable, Sendable {
        case nan
        case positiveInfinity
        case negativeInfinity
    }

    case nonFiniteTemperature(NonFiniteKind)
    case temperatureOutOfRange(Double)
    case nonFiniteTopP(NonFiniteKind)
    case topPOutOfRange(Double)
    case zeroTemperatureSamplingConflict
    case unsupportedDrawDomain(UInt32)
    case committedSampleOrdinalOverflow
    case emptyLogits
    case vocabularyCountExceedsMaximum(actual: Int, maximum: Int)
    case nonFiniteLogit(tokenID: Int, kind: NonFiniteKind)
    case invalidFullProbabilityMass
    case invalidRetainedProbabilityMass
    case invalidSelectedTokenIndex(Int)
}

enum SamplingContractV1 {
    static let contractVersionTag = "fast-mlx-sampling-contract-v1"
    static let randomWordPreimagePrefix = "fast-mlx-sampling-rng-v1\n"
    static let tokenSelectionDomain: UInt32 = 0
    static let maximumVocabularyCount = 262_144
    static let uniformDenominator = 4_503_599_627_370_496.0
}

struct SamplingSeedV1: Equatable, Sendable {
    let resolvedSeedBitPattern: UInt64

    private init(resolvedSeedBitPattern: UInt64) {
        self.resolvedSeedBitPattern = resolvedSeedBitPattern
    }

    static func callerSupplied(_ seed: Int64) -> SamplingSeedV1 {
        SamplingSeedV1(resolvedSeedBitPattern: UInt64(bitPattern: seed))
    }
}

struct SamplingPolicyV1: Equatable, Sendable {
    let contractVersionTag: String
    let temperature: Double
    let topP: Double
    let seed: SamplingSeedV1?

    private init(
        contractVersionTag: String,
        temperature: Double,
        topP: Double,
        seed: SamplingSeedV1?
    ) {
        self.contractVersionTag = contractVersionTag
        self.temperature = temperature
        self.topP = topP
        self.seed = seed
    }

    static func greedy() -> SamplingPolicyV1 {
        SamplingPolicyV1(
            contractVersionTag: SamplingContractV1.contractVersionTag,
            temperature: 0.0,
            topP: 1.0,
            seed: nil)
    }

    static func sampled(
        temperature: Double = 1,
        topP: Double = 1,
        seed: SamplingSeedV1
    ) throws -> SamplingPolicyV1 {
        if !temperature.isFinite {
            if temperature.isNaN {
                throw SamplingContractFailure.nonFiniteTemperature(.nan)
            }
            if temperature == .infinity {
                throw SamplingContractFailure.nonFiniteTemperature(
                    .positiveInfinity)
            }
            throw SamplingContractFailure.nonFiniteTemperature(
                .negativeInfinity)
        }

        if temperature < 0 || temperature > 2 {
            throw SamplingContractFailure.temperatureOutOfRange(temperature)
        }

        if !topP.isFinite {
            if topP.isNaN {
                throw SamplingContractFailure.nonFiniteTopP(.nan)
            }
            if topP == .infinity {
                throw SamplingContractFailure.nonFiniteTopP(.positiveInfinity)
            }
            throw SamplingContractFailure.nonFiniteTopP(.negativeInfinity)
        }

        if topP < 0 || topP > 1 {
            throw SamplingContractFailure.topPOutOfRange(topP)
        }

        let canonicalTemperature = temperature == 0 ? 0.0 : temperature
        let canonicalTopP = topP == 0 ? 0.0 : topP

        if canonicalTemperature == 0 {
            throw SamplingContractFailure.zeroTemperatureSamplingConflict
        }

        return SamplingPolicyV1(
            contractVersionTag: SamplingContractV1.contractVersionTag,
            temperature: canonicalTemperature,
            topP: canonicalTopP,
            seed: seed)
    }
}

struct SamplingDrawAddressV1: Equatable, Sendable {
    let contractVersionTag: String
    let seedBitPattern: UInt64
    let domain: UInt32
    let committedSampleOrdinal: UInt64

    init(
        seed: SamplingSeedV1,
        domain: UInt32 = SamplingContractV1.tokenSelectionDomain,
        committedSampleOrdinal: UInt64
    ) throws {
        guard domain == SamplingContractV1.tokenSelectionDomain else {
            throw SamplingContractFailure.unsupportedDrawDomain(domain)
        }

        contractVersionTag = SamplingContractV1.contractVersionTag
        seedBitPattern = seed.resolvedSeedBitPattern
        self.domain = domain
        self.committedSampleOrdinal = committedSampleOrdinal
    }

    func randomWord() -> UInt64 {
        var preimage = Data(
            SamplingContractV1.randomWordPreimagePrefix.utf8)

        for shift in stride(from: 56, through: 0, by: -8) {
            preimage.append(UInt8(truncatingIfNeeded: seedBitPattern >> shift))
        }
        for shift in stride(from: 24, through: 0, by: -8) {
            preimage.append(UInt8(truncatingIfNeeded: domain >> shift))
        }
        for shift in stride(from: 56, through: 0, by: -8) {
            preimage.append(
                UInt8(
                    truncatingIfNeeded:
                        committedSampleOrdinal >> shift))
        }

        let digest = SHA256.hash(data: preimage)
        var word: UInt64 = 0
        for byte in digest.prefix(8) {
            word = (word << 8) | UInt64(byte)
        }
        return word
    }

    func uniformOpenInterval() -> Double {
        (Double(randomWord() >> 12) + 0.5)
            / SamplingContractV1.uniformDenominator
    }

    func afterCommittedSelection() throws -> SamplingDrawAddressV1 {
        let (nextOrdinal, overflow) =
            committedSampleOrdinal.addingReportingOverflow(1)
        guard !overflow else {
            throw SamplingContractFailure.committedSampleOrdinalOverflow
        }

        return try SamplingDrawAddressV1(
            seed: SamplingSeedV1.callerSupplied(
                Int64(bitPattern: seedBitPattern)),
            domain: domain,
            committedSampleOrdinal: nextOrdinal)
    }
}

struct SamplingSelectionV1: Equatable, Sendable {
    let tokenID: Int
    let drawAddress: SamplingDrawAddressV1?
    let randomWord: UInt64?

    fileprivate enum ConstructionSeal: Equatable, Sendable {
        case oracle
    }

    fileprivate init(
        tokenID: Int,
        drawAddress: SamplingDrawAddressV1?,
        randomWord: UInt64?,
        constructionSeal: ConstructionSeal
    ) {
        switch constructionSeal {
        case .oracle:
            break
        }
        self.tokenID = tokenID
        self.drawAddress = drawAddress
        self.randomWord = randomWord
    }
}

enum SamplingRank1OracleV1 {
    static func selectToken(
        logits: [Double],
        policy: SamplingPolicyV1,
        committedSampleOrdinal: UInt64 = 0
    ) throws -> SamplingSelectionV1 {
        guard !logits.isEmpty else {
            throw SamplingContractFailure.emptyLogits
        }

        guard logits.count <= SamplingContractV1.maximumVocabularyCount else {
            throw SamplingContractFailure.vocabularyCountExceedsMaximum(
                actual: logits.count,
                maximum: SamplingContractV1.maximumVocabularyCount)
        }

        for (tokenID, logit) in logits.enumerated() {
            if !logit.isFinite {
                let kind: SamplingContractFailure.NonFiniteKind
                if logit.isNaN {
                    kind = .nan
                } else if logit == .infinity {
                    kind = .positiveInfinity
                } else {
                    kind = .negativeInfinity
                }
                throw SamplingContractFailure.nonFiniteLogit(
                    tokenID: tokenID,
                    kind: kind)
            }
        }

        if policy.temperature == 0 {
            var selectedTokenID = 0
            var selectedLogit = logits[0]
            for tokenID in logits.indices.dropFirst() {
                if logits[tokenID] > selectedLogit {
                    selectedTokenID = tokenID
                    selectedLogit = logits[tokenID]
                }
            }

            guard logits.indices.contains(selectedTokenID) else {
                throw SamplingContractFailure.invalidSelectedTokenIndex(
                    selectedTokenID)
            }

            return SamplingSelectionV1(
                tokenID: selectedTokenID,
                drawAddress: nil,
                randomWord: nil,
                constructionSeal: .oracle)
        }

        let (logitBytes, logitBytesOverflow) =
            logits.count.multipliedReportingOverflow(
                by: MemoryLayout<Double>.stride)
        let (candidateBytes, candidateBytesOverflow) =
            logits.count.multipliedReportingOverflow(
                by: MemoryLayout<(Int, Double)>.stride)
        let (twoLogitArraysBytes, twoLogitArraysOverflow) =
            logitBytes.multipliedReportingOverflow(by: 2)
        let (fourCandidateArraysBytes, fourCandidateArraysOverflow) =
            candidateBytes.multipliedReportingOverflow(by: 4)
        let (aggregateBytes, aggregateBytesOverflow) =
            twoLogitArraysBytes.addingReportingOverflow(
                fourCandidateArraysBytes)
        precondition(
            !logitBytesOverflow
                && !candidateBytesOverflow
                && !twoLogitArraysOverflow
                && !fourCandidateArraysOverflow
                && !aggregateBytesOverflow
                && aggregateBytes <= 268_435_456,
            "SamplingContractV1 resource ledger invariant")

        var maximumLogit = logits[0]
        for logit in logits.dropFirst() where logit > maximumLogit {
            maximumLogit = logit
        }

        var weights = [Double]()
        weights.reserveCapacity(logits.count)
        for logit in logits {
            let scaled = (logit - maximumLogit) / policy.temperature
            let exponentiated = Foundation.exp(scaled)
            if scaled == -.infinity || exponentiated == -.infinity {
                weights.append(0.0)
            } else {
                weights.append(exponentiated)
            }
        }

        let fullMass = compensatedFiniteSum(weights)
        guard fullMass.isFinite && fullMass > 0 else {
            throw SamplingContractFailure.invalidFullProbabilityMass
        }

        var candidates = [(tokenID: Int, weight: Double)]()
        candidates.reserveCapacity(logits.count)
        for (tokenID, weight) in weights.enumerated() where weight > 0 {
            candidates.append((tokenID: tokenID, weight: weight))
        }
        candidates.sort { left, right in
            if left.weight == right.weight {
                return left.tokenID < right.tokenID
            }
            return left.weight > right.weight
        }

        var retained = [(tokenID: Int, weight: Double)]()
        retained.reserveCapacity(candidates.count)
        if policy.topP == 0 {
            retained.append(candidates[0])
        } else if policy.topP == 1 {
            retained = candidates
        } else {
            let cutoff = policy.topP * fullMass
            var sum = 0.0
            var compensation = 0.0
            for candidate in candidates {
                let next = sum + candidate.weight
                if abs(sum) >= abs(candidate.weight) {
                    compensation += (sum - next) + candidate.weight
                } else {
                    compensation += (candidate.weight - next) + sum
                }
                sum = next
                retained.append(candidate)
                if sum + compensation >= cutoff {
                    break
                }
            }
        }
        retained.sort { left, right in
            left.tokenID < right.tokenID
        }

        var retainedSum = 0.0
        var retainedCompensation = 0.0
        for candidate in retained {
            let next = retainedSum + candidate.weight
            if abs(retainedSum) >= abs(candidate.weight) {
                retainedCompensation +=
                    (retainedSum - next) + candidate.weight
            } else {
                retainedCompensation +=
                    (candidate.weight - next) + retainedSum
            }
            retainedSum = next
        }
        let retainedTotal = retainedSum + retainedCompensation
        guard retainedTotal.isFinite && retainedTotal > 0 else {
            throw SamplingContractFailure.invalidRetainedProbabilityMass
        }

        let seed = policy.seed!
        let drawAddress = try SamplingDrawAddressV1(
            seed: seed,
            committedSampleOrdinal: committedSampleOrdinal)
        let word = drawAddress.randomWord()
        let uniform = (Double(word >> 12) + 0.5)
            / SamplingContractV1.uniformDenominator
        let threshold = uniform * retainedTotal

        var selectedTokenID: Int?
        var cumulativeSum = 0.0
        var cumulativeCompensation = 0.0
        for candidate in retained {
            let next = cumulativeSum + candidate.weight
            if abs(cumulativeSum) >= abs(candidate.weight) {
                cumulativeCompensation +=
                    (cumulativeSum - next) + candidate.weight
            } else {
                cumulativeCompensation +=
                    (candidate.weight - next) + cumulativeSum
            }
            cumulativeSum = next
            if cumulativeSum + cumulativeCompensation > threshold {
                selectedTokenID = candidate.tokenID
                break
            }
        }

        let finalTokenID = selectedTokenID ?? retained[retained.count - 1].tokenID
        guard logits.indices.contains(finalTokenID) else {
            throw SamplingContractFailure.invalidSelectedTokenIndex(
                finalTokenID)
        }

        return SamplingSelectionV1(
            tokenID: finalTokenID,
            drawAddress: drawAddress,
            randomWord: word,
            constructionSeal: .oracle)
    }

    static func compensatedFiniteSum(_ values: [Double]) -> Double {
        var sum = 0.0
        var compensation = 0.0

        for value in values {
            let next = sum + value
            if abs(sum) >= abs(value) {
                compensation += (sum - next) + value
            } else {
                compensation += (value - next) + sum
            }
            sum = next
        }

        return sum + compensation
    }
}
