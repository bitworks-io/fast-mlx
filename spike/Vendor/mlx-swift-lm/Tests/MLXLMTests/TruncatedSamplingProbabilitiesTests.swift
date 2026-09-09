// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import XCTest

/// Covers `truncatedSamplingProbabilities`, the public entry point that lets
/// code outside `TopPSampler` (for example a speculative decoding path)
/// compute the exact same truncated target distribution the scalar sampler
/// draws from, by calling the same extracted filter functions rather than
/// reimplementing them.
///
/// Every expectation below is computed independently in plain Swift
/// arithmetic from a small, explicit fixture — never by calling the function
/// under test — so a bug in the implementation cannot also be baked into the
/// expectation.
final class TruncatedSamplingProbabilitiesTests: XCTestCase {

    private func softmax(_ logits: [Double]) -> [Double] {
        let m = logits.max()!
        let exps = logits.map { Foundation.exp($0 - m) }
        let sum = exps.reduce(0, +)
        return exps.map { $0 / sum }
    }

    // MARK: - Identity control

    func testIdentityAtFullRangeMatchesPlainSoftmax() {
        let logitsValues: [Float] = [1.0, 2.0, 0.5, -1.0, 3.0]
        let logits = MLXArray(logitsValues)[.newAxis, .ellipsis]

        let result = truncatedSamplingProbabilities(
            logits: logits, temperature: 1, topP: 1, topK: 0, minP: 0
        ).asArray(Float.self)

        let expected = softmax(logitsValues.map(Double.init))

        XCTAssertEqual(result.count, expected.count)
        for (actual, wanted) in zip(result, expected) {
            XCTAssertEqual(Double(actual), wanted, accuracy: 1e-6)
        }
    }

    // MARK: - Top-k truncates and renormalizes

    func testTopKTruncatesAndRenormalizes() {
        // softmax(logitsValues) puts the two largest masses on indices 1 and 2.
        let logitsValues: [Float] = [0.5, 2.0, 1.0, -0.5, 0.2]
        let logits = MLXArray(logitsValues)[.newAxis, .ellipsis]

        let result = truncatedSamplingProbabilities(
            logits: logits, temperature: 1, topP: 1, topK: 2, minP: 0
        ).asArray(Float.self)

        let fullSoftmax = softmax(logitsValues.map(Double.init))
        let keptIndices = [1, 2]
        let keptSum = keptIndices.reduce(0.0) { $0 + fullSoftmax[$1] }
        let expectedKept = keptIndices.map { fullSoftmax[$0] / keptSum }

        XCTAssertEqual(Double(result[1]), expectedKept[0], accuracy: 1e-6)
        XCTAssertEqual(Double(result[2]), expectedKept[1], accuracy: 1e-6)

        // Every entry outside the top-k set must be exactly zero, not merely small.
        XCTAssertEqual(result[0], 0.0)
        XCTAssertEqual(result[3], 0.0)
        XCTAssertEqual(result[4], 0.0)
    }

    // MARK: - Top-p truncates

    func testTopPTruncatesToIndependentlyComputedNucleus() {
        let logitsValues: [Float] = [0.0, 2.0, 1.0, -1.0, 0.5, -2.0]
        let logits = MLXArray(logitsValues)[.newAxis, .ellipsis]
        let topP: Float = 0.9

        let result = truncatedSamplingProbabilities(
            logits: logits, temperature: 1, topP: topP, topK: 0, minP: 0
        ).asArray(Float.self)

        let fullSoftmax = softmax(logitsValues.map(Double.init))

        // Nucleus set: sort ascending, keep entries whose ascending cumulative
        // probability exceeds 1 - topP (mirrors `apply_top_p` in mlx_lm).
        let ascendingIndices = (0 ..< fullSoftmax.count).sorted { fullSoftmax[$0] < fullSoftmax[$1] }
        var running = 0.0
        var keptIndices: [Int] = []
        for index in ascendingIndices {
            running += fullSoftmax[index]
            if running > (1 - Double(topP)) {
                keptIndices.append(index)
            }
        }
        XCTAssertEqual(Set(keptIndices), Set([0, 1, 2, 4]))

        let keptSum = keptIndices.reduce(0.0) { $0 + fullSoftmax[$1] }
        for index in keptIndices {
            XCTAssertEqual(Double(result[index]), fullSoftmax[index] / keptSum, accuracy: 1e-6)
        }

        let maskedIndices = Set(0 ..< fullSoftmax.count).subtracting(keptIndices)
        for index in maskedIndices {
            XCTAssertEqual(result[index], 0.0)
        }
    }

    // MARK: - Temperature applied after truncation

    func testTemperatureIsAppliedAfterTruncation() {
        let logitsValues: [Float] = [0.5, 2.0, 1.0, -0.5, 0.2]
        let logits = MLXArray(logitsValues)[.newAxis, .ellipsis]

        // Independently computed: mask everything but the top-2 log-probabilities,
        // divide the survivors by temperature, then softmax.
        let fullSoftmax = softmax(logitsValues.map(Double.init))
        let logSoftmaxValues = fullSoftmax.map(Foundation.log)
        let keptIndices = [1, 2]
        func maskedLogprobsDividedByTemperature(_ temperature: Double) -> [Double] {
            (0 ..< logSoftmaxValues.count).map { index in
                keptIndices.contains(index) ? logSoftmaxValues[index] / temperature : -Double.infinity
            }
        }

        func softmaxWithInfinities(_ values: [Double]) -> [Double] {
            let finite = values.filter { $0.isFinite }
            let m = finite.max()!
            let exps = values.map { $0.isFinite ? Foundation.exp($0 - m) : 0.0 }
            let sum = exps.reduce(0, +)
            return exps.map { $0 / sum }
        }

        let expectedAtTemp1 = softmaxWithInfinities(maskedLogprobsDividedByTemperature(1))
        let expectedAtTemp05 = softmaxWithInfinities(maskedLogprobsDividedByTemperature(0.5))

        let resultAtTemp1 = truncatedSamplingProbabilities(
            logits: logits, temperature: 1, topP: 1, topK: 2, minP: 0
        ).asArray(Float.self)
        let resultAtTemp05 = truncatedSamplingProbabilities(
            logits: logits, temperature: 0.5, topP: 1, topK: 2, minP: 0
        ).asArray(Float.self)

        for index in 0 ..< logitsValues.count {
            XCTAssertEqual(Double(resultAtTemp1[index]), expectedAtTemp1[index], accuracy: 1e-6)
            XCTAssertEqual(Double(resultAtTemp05[index]), expectedAtTemp05[index], accuracy: 1e-6)
        }

        // Anti-vacuity control: without temperature scaling this test would
        // pass even if `truncatedSamplingProbabilities` silently ignored
        // `temperature`, since both calls would then produce the same result.
        var maxAbsoluteDifference = 0.0
        for index in 0 ..< logitsValues.count {
            maxAbsoluteDifference = Swift.max(
                maxAbsoluteDifference,
                abs(Double(resultAtTemp1[index]) - Double(resultAtTemp05[index]))
            )
        }
        XCTAssertGreaterThan(maxAbsoluteDifference, 1e-3)
    }

    // MARK: - TopPSampler unchanged by the extraction

    func testTopPSamplerStillDrawsArgmaxUnderTopKOne() {
        let logitsValues: [Float] = [0.1, 2.0, 1.0]
        let logits = MLXArray(logitsValues)[.newAxis, .ellipsis]
        let sampler = TopPSampler(temperature: 1.0, topK: 1, seed: 42)

        for _ in 0 ..< 20 {
            let token = sampler.sample(logits: logits).item(Int.self)
            XCTAssertEqual(token, 1)
        }
    }
}
