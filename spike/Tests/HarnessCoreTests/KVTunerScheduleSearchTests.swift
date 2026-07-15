import Foundation
import XCTest

@testable import HarnessCore

final class KVTunerScheduleSearchTests: XCTestCase {
    private let pairs = KVTunerSensitivityArtifact.canonicalPrecisionPairs
    private let environmentSHA256 = String(repeating: "e", count: 64)

    private func runtimePolicySHA256(_ ordinal: Int) -> String {
        String(repeating: String(ordinal), count: 64)
    }

    private func analysis() -> KVTunerSensitivityAnalysis {
        let outputErrors = [
            [0.01, 0.05, 0.20],
            [0.01, 0.05, 0.20],
            [0.02, 0.06, 0.10],
            [0.02, 0.06, 0.10],
        ]
        let layers = outputErrors.indices.map { layer in
            KVTunerLayerSensitivity(
                layer: layer,
                aggregates: pairs.indices.map { pairIndex in
                    KVTunerPairSensitivity(
                        pair: pairs[pairIndex],
                        relativeKeyError: outputErrors[layer][pairIndex],
                        relativeValueError: outputErrors[layer][pairIndex],
                        attentionScoreError: outputErrors[layer][pairIndex],
                        relativeAttentionOutputError:
                            outputErrors[layer][pairIndex])
                },
                paretoPairs: pairs)
        }
        return KVTunerSensitivityAnalysis(
            layers: layers,
            groups: [
                KVTunerSensitivityGroup(
                    id: 0, layers: [0, 1], allowedPairs: pairs),
                KVTunerSensitivityGroup(
                    id: 1, layers: [2, 3], allowedPairs: pairs),
            ])
    }

    func testExactBudgetEnumerationMatchesHandCalculatedOracle() throws {
        let candidates = try KVTunerScheduleSearch.enumerate(
            analysis: analysis(),
            targetPairBitTotal: 36,
            maxCandidates: 10)

        XCTAssertEqual(candidates.count, 2)
        XCTAssertEqual(candidates.map(\.ordinal), [0, 1])
        XCTAssertEqual(candidates.map(\.totalPairBits), [36, 36])
        XCTAssertEqual(candidates[0].layers.map(\.keyBits), [8, 8, 4, 4])
        XCTAssertEqual(candidates[0].layers.map(\.valueBits), [4, 4, 2, 2])
        XCTAssertEqual(candidates[0].meanAttentionOutputError, 0.055, accuracy: 1e-15)
        XCTAssertEqual(candidates[1].layers.map(\.keyBits), [4, 4, 8, 8])
        XCTAssertEqual(candidates[1].layers.map(\.valueBits), [2, 2, 4, 4])
        XCTAssertEqual(candidates[1].meanAttentionOutputError, 0.11, accuracy: 1e-15)
    }

    func testSearchFailsClosedWhenBudgetIsUnreachableOrCandidateCapIsExceeded() {
        XCTAssertThrowsError(try KVTunerScheduleSearch.enumerate(
            analysis: analysis(),
            targetPairBitTotal: 35,
            maxCandidates: 10)) { error in
                XCTAssertEqual(
                    error as? KVTunerScheduleSearchError,
                    .noFeasibleSchedule)
            }

        XCTAssertThrowsError(try KVTunerScheduleSearch.enumerate(
            analysis: analysis(),
            targetPairBitTotal: 36,
            maxCandidates: 1)) { error in
                XCTAssertEqual(
                    error as? KVTunerScheduleSearchError,
                    .candidateLimitExceeded(limit: 1))
            }
    }

    func testSearchRejectsMalformedGroupCoverageBeforeEnumeration() {
        let base = analysis()
        var malformedGroups = base.groups
        malformedGroups[1] = KVTunerSensitivityGroup(
            id: 1, layers: [1, 3], allowedPairs: pairs)
        var malformed = KVTunerSensitivityAnalysis(
            layers: base.layers, groups: malformedGroups)

        XCTAssertThrowsError(try KVTunerScheduleSearch.enumerate(
            analysis: malformed,
            targetPairBitTotal: 36,
            maxCandidates: 10)) { error in
                XCTAssertEqual(
                    error as? KVTunerScheduleSearchError,
                    .invalidAnalysis)
            }

        malformedGroups = Array(base.groups.reversed())
        malformed = KVTunerSensitivityAnalysis(
            layers: base.layers, groups: malformedGroups)
        XCTAssertThrowsError(try KVTunerScheduleSearch.enumerate(
            analysis: malformed,
            targetPairBitTotal: 36,
            maxCandidates: 10))
    }

    func testAccuracySelectionWinsFirstThenSensitivityBreaksTies() throws {
        let candidates = try KVTunerScheduleSearch.enumerate(
            analysis: analysis(),
            targetPairBitTotal: 36,
            maxCandidates: 10)
        let outputA = String(repeating: "a", count: 64)
        let outputB = String(repeating: "b", count: 64)

        var evaluations = [
            KVTunerCandidateEvaluation(
                candidateOrdinal: 0,
                candidateSHA256:
                    try KVTunerScheduleSearch.candidateSHA256(candidates[0]),
                runtimePolicySHA256: runtimePolicySHA256(0),
                environmentSHA256: environmentSHA256,
                correctCount: 10,
                totalCount: 200,
                outputSHA256: outputA),
            KVTunerCandidateEvaluation(
                candidateOrdinal: 1,
                candidateSHA256:
                    try KVTunerScheduleSearch.candidateSHA256(candidates[1]),
                runtimePolicySHA256: runtimePolicySHA256(1),
                environmentSHA256: environmentSHA256,
                correctCount: 11,
                totalCount: 200,
                outputSHA256: outputB),
        ]
        XCTAssertEqual(
            try KVTunerScheduleSearch.select(
                candidates: candidates,
                evaluations: evaluations,
                requiredRuntimePolicySHA256ByCandidate: [
                    runtimePolicySHA256(0), runtimePolicySHA256(1),
                ]).ordinal,
            1)

        evaluations[1] = KVTunerCandidateEvaluation(
            candidateOrdinal: 1,
            candidateSHA256:
                try KVTunerScheduleSearch.candidateSHA256(candidates[1]),
            runtimePolicySHA256: runtimePolicySHA256(1),
            environmentSHA256: environmentSHA256,
            correctCount: 10,
            totalCount: 200,
            outputSHA256: outputB)
        XCTAssertEqual(
            try KVTunerScheduleSearch.select(
                candidates: candidates,
                evaluations: evaluations,
                requiredRuntimePolicySHA256ByCandidate: [
                    runtimePolicySHA256(0), runtimePolicySHA256(1),
                ]).ordinal,
            0)

        var equalSensitivity = candidates
        equalSensitivity[1] = KVTunerScheduleCandidate(
            ordinal: 1,
            analysisSHA256: candidates[1].analysisSHA256,
            totalPairBits: candidates[1].totalPairBits,
            meanAttentionOutputError:
                candidates[0].meanAttentionOutputError,
            layers: candidates[1].layers)
        var equalEvaluations = evaluations
        equalEvaluations[1] = KVTunerCandidateEvaluation(
            candidateOrdinal: 1,
            candidateSHA256: try KVTunerScheduleSearch.candidateSHA256(
                equalSensitivity[1]),
            runtimePolicySHA256: runtimePolicySHA256(1),
            environmentSHA256: environmentSHA256,
            correctCount: 10,
            totalCount: 200,
            outputSHA256: outputB)
        XCTAssertEqual(
            try KVTunerScheduleSearch.select(
                candidates: equalSensitivity,
                evaluations: equalEvaluations,
                requiredRuntimePolicySHA256ByCandidate: [
                    runtimePolicySHA256(0), runtimePolicySHA256(1),
                ]).ordinal,
            1)
    }

    func testQwen3SixtyFourLayerBudgetSatisfiesExactB45Invariant() throws {
        let layers = (0..<64).map { layer in
            let errors = layer < 32
                ? [0.01, 0.05, 0.20]
                : [0.20, 0.20, 0.20]
            let aggregates = pairs.indices.map { pairIndex in
                KVTunerPairSensitivity(
                    pair: pairs[pairIndex],
                    relativeKeyError: errors[pairIndex],
                    relativeValueError: errors[pairIndex],
                    attentionScoreError: errors[pairIndex],
                    relativeAttentionOutputError: errors[pairIndex])
            }
            return KVTunerLayerSensitivity(
                layer: layer,
                aggregates: aggregates,
                paretoPairs: layer < 32 ? pairs : [pairs[2]])
        }
        let qwenAnalysis = KVTunerSensitivityAnalysis(
            layers: layers,
            groups: [
                KVTunerSensitivityGroup(
                    id: 0,
                    layers: Array(0..<32),
                    allowedPairs: pairs),
                KVTunerSensitivityGroup(
                    id: 1,
                    layers: Array(32..<64),
                    allowedPairs: [pairs[2]]),
            ])

        let candidate = try XCTUnwrap(KVTunerScheduleSearch.enumerate(
            analysis: qwenAnalysis,
            targetPairBitTotal: 576,
            maxCandidates: 10).first)
        let a = candidate.layers.filter {
            $0.keyBits == 8 && $0.valueBits == 4
        }.count
        let b = candidate.layers.filter {
            $0.keyBits == 8 && $0.valueBits == 2
        }.count
        let c = candidate.layers.filter {
            $0.keyBits == 4 && $0.valueBits == 2
        }.count

        XCTAssertEqual(a + b + c, 64)
        XCTAssertEqual(3 * a + 2 * b, 96)
        XCTAssertEqual(candidate.totalPairBits, 576)
    }

    func testCandidateIdentityUsesVersionedTypedTranscriptAndNormalizesSignedZero() throws {
        let base = KVTunerScheduleCandidate(
            ordinal: 0,
            analysisSHA256: String(repeating: "a", count: 64),
            totalPairBits: 12,
            meanAttentionOutputError: 0.0,
            layers: [
                KVLayerPrecision(layer: 0, keyBits: 8, valueBits: 4),
            ])
        let negativeZero = KVTunerScheduleCandidate(
            ordinal: base.ordinal,
            analysisSHA256: base.analysisSHA256,
            totalPairBits: base.totalPairBits,
            meanAttentionOutputError: -0.0,
            layers: base.layers)

        let digest = try KVTunerScheduleSearch.candidateSHA256(base)
        XCTAssertEqual(
            digest,
            try KVTunerScheduleSearch.candidateSHA256(negativeZero))
        XCTAssertEqual(
            digest,
            "12772c1f7f5f5b585a091958362030316c59b6a208f2a13df8837ad5ed55a15f")
        XCTAssertEqual(
            try KVTunerScheduleSearch.candidateListSHA256([base]),
            "a2c42fdeae55a4c42a79fcbef77fa86a84557751ac6ee7e928f120bcb82c1814")
    }

    func testSelectionRejectsIncompleteDuplicateAndMalformedEvaluations() throws {
        let candidates = try KVTunerScheduleSearch.enumerate(
            analysis: analysis(),
            targetPairBitTotal: 36,
            maxCandidates: 10)
        let valid = KVTunerCandidateEvaluation(
            candidateOrdinal: 0,
            candidateSHA256:
                try KVTunerScheduleSearch.candidateSHA256(candidates[0]),
            runtimePolicySHA256: runtimePolicySHA256(0),
            environmentSHA256: environmentSHA256,
            correctCount: 10,
            totalCount: 200,
            outputSHA256: String(repeating: "a", count: 64))

        XCTAssertThrowsError(try KVTunerScheduleSearch.select(
            candidates: candidates,
            evaluations: [valid],
            requiredRuntimePolicySHA256ByCandidate: [
                runtimePolicySHA256(0), runtimePolicySHA256(1),
            ])) { error in
                XCTAssertEqual(
                    error as? KVTunerScheduleSearchError,
                    .incompleteEvaluations(expected: 2, actual: 1))
            }
        XCTAssertThrowsError(try KVTunerScheduleSearch.select(
            candidates: candidates,
            evaluations: [valid, valid],
            requiredRuntimePolicySHA256ByCandidate: [
                runtimePolicySHA256(0), runtimePolicySHA256(1),
            ]))

        let invalidCount = KVTunerCandidateEvaluation(
            candidateOrdinal: 1,
            candidateSHA256:
                try KVTunerScheduleSearch.candidateSHA256(candidates[1]),
            runtimePolicySHA256: runtimePolicySHA256(1),
            environmentSHA256: environmentSHA256,
            correctCount: 201,
            totalCount: 200,
            outputSHA256: String(repeating: "b", count: 64))
        XCTAssertThrowsError(try KVTunerScheduleSearch.select(
            candidates: candidates,
            evaluations: [valid, invalidCount],
            requiredRuntimePolicySHA256ByCandidate: [
                runtimePolicySHA256(0), runtimePolicySHA256(1),
            ]))
    }

    func testSelectionRejectsMixedEnvironmentAndRuntimePolicySubstitution() throws {
        let candidates = try KVTunerScheduleSearch.enumerate(
            analysis: analysis(),
            targetPairBitTotal: 36,
            maxCandidates: 10)
        let requiredPolicies = [
            runtimePolicySHA256(0), runtimePolicySHA256(1),
        ]
        let valid = try candidates.map { candidate in
            KVTunerCandidateEvaluation(
                candidateOrdinal: candidate.ordinal,
                candidateSHA256:
                    try KVTunerScheduleSearch.candidateSHA256(candidate),
                runtimePolicySHA256:
                    requiredPolicies[candidate.ordinal],
                environmentSHA256: environmentSHA256,
                correctCount: 10,
                totalCount: 200,
                outputSHA256: String(repeating: "a", count: 64))
        }

        var mixedEnvironment = valid
        mixedEnvironment[1] = KVTunerCandidateEvaluation(
            candidateOrdinal: valid[1].candidateOrdinal,
            candidateSHA256: valid[1].candidateSHA256,
            runtimePolicySHA256: valid[1].runtimePolicySHA256,
            environmentSHA256: String(repeating: "f", count: 64),
            correctCount: valid[1].correctCount,
            totalCount: valid[1].totalCount,
            outputSHA256: valid[1].outputSHA256)
        XCTAssertThrowsError(try KVTunerScheduleSearch.select(
            candidates: candidates,
            evaluations: mixedEnvironment,
            requiredRuntimePolicySHA256ByCandidate: requiredPolicies
        )) { error in
            XCTAssertEqual(
                error as? KVTunerScheduleSearchError,
                .mixedEvaluationEnvironments)
        }

        var substitutedPolicy = valid
        substitutedPolicy[1] = KVTunerCandidateEvaluation(
            candidateOrdinal: valid[1].candidateOrdinal,
            candidateSHA256: valid[1].candidateSHA256,
            runtimePolicySHA256: requiredPolicies[0],
            environmentSHA256: valid[1].environmentSHA256,
            correctCount: valid[1].correctCount,
            totalCount: valid[1].totalCount,
            outputSHA256: valid[1].outputSHA256)
        XCTAssertThrowsError(try KVTunerScheduleSearch.select(
            candidates: candidates,
            evaluations: substitutedPolicy,
            requiredRuntimePolicySHA256ByCandidate: requiredPolicies
        )) { error in
            XCTAssertEqual(
                error as? KVTunerScheduleSearchError,
                .runtimePolicyMismatch(1))
        }
    }

}
