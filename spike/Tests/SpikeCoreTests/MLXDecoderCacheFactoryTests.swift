import MLX
import MLXLMCommon
import MLXNN
import XCTest

import HarnessCore

@testable import SpikeCore

private final class CacheFactorySpyModel: Module, LanguageModel {
    private(set) var nativeCacheBuildCount = 0

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        nativeCacheBuildCount += 1
        return [KVCacheSimple()]
    }

    func prepare(
        _ input: LMInput,
        cache: [KVCache],
        windowSize: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray.zeros([inputs.dim(0), inputs.dim(1), 8])
    }
}

private final class CacheFactorySpy {
    private(set) var callCount = 0

    func makeInt8Caches() -> [KVCache] {
        callCount += 1
        return [QuantizedKVCache(groupSize: 32, bits: 8, mode: .affine)]
    }
}

final class MLXDecoderCacheFactoryTests: XCTestCase {
    func testResetRebuildsTheConfiguredCacheKindWithoutCallingNativeFactory() {
        let model = CacheFactorySpyModel()
        let factory = CacheFactorySpy()
        var decoder = MLXDecoder(model: model, cacheFactory: { factory.makeInt8Caches() })

        XCTAssertEqual(factory.callCount, 1)
        XCTAssertEqual(model.nativeCacheBuildCount, 0)

        decoder.reset()

        XCTAssertEqual(factory.callCount, 2)
        XCTAssertEqual(model.nativeCacheBuildCount, 0)
    }
}

/// Order- and cache-state-sensitive fixture used to prove that chunked prefill is token- and
/// cache-state-equivalent to a single whole-prompt forward. The emitted logits are a function of
/// the FULL, position-weighted set of keys resident in the cache at the moment of the call — not
/// merely of the current chunk's input — so a chunk that is skipped, duplicated, or delivered out
/// of order changes both the sampled token and the cache contents at every downstream position. A
/// fixture that ignored the cache (or returned a constant) would make the equivalence assertions
/// below vacuous.
private final class OrderSensitiveCacheModel: Module, LanguageModel {
    static let vocabSize = 32

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [KVCacheSimple()]
    }

    func prepare(
        _ input: LMInput,
        cache: [KVCache],
        windowSize: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        guard let cache, let kv = cache.first else {
            fatalError("OrderSensitiveCacheModel requires a cache")
        }
        let seqLen = inputs.dim(1)
        let floatIds = inputs.asType(.float32).reshaped([1, 1, seqLen, 1])
        let (allKeys, _) = kv.update(keys: floatIds, values: floatIds)
        let offset = kv.offset
        // Position-weighted so the aggregate depends on WHERE in the cache each token landed,
        // not merely on the multiset of tokens seen — this is what makes reordered chunks
        // detectable, not only dropped ones.
        let positions = MLXArray((1...offset).map { Float($0) }, [1, 1, offset, 1])
        let aggregate = (allKeys * positions).sum().item(Float.self)
        let index =
            (((Int(aggregate.rounded()) % Self.vocabSize) + Self.vocabSize) % Self.vocabSize)
        var row = [Float](repeating: 0, count: Self.vocabSize)
        row[index] = 1_000
        let logitsRow = MLXArray(row, [1, 1, Self.vocabSize])
        return broadcast(logitsRow, to: [1, seqLen, Self.vocabSize])
    }
}

final class MLXDecoderPrefillChunkingEquivalenceTests: XCTestCase {
    /// All mutable state lives in the per-decoder cache array passed in by each test, so reusing
    /// one stateless model instance across decoders under test is safe.
    private let model = OrderSensitiveCacheModel()

    /// A chunk size exactly equal to the prompt length forces exactly one loop iteration
    /// covering the whole prompt — the chunked loop's single-iteration path is byte-identical in
    /// control flow to the prior single whole-prompt forward, so this is a faithful "today's
    /// behavior" baseline to compare against. (Deliberately NOT "prompt length + slack": a
    /// larger-than-needed chunk size is a distinct edge case from "one chunk covers the whole
    /// prompt" and conflating the two would make this baseline itself sensitive to bugs in the
    /// loop's upper bound, rather than being an independent reference.)
    private func wholePromptChunkSize(for promptLength: Int) -> Int {
        promptLength
    }

    private func runPrefill(
        promptTokens: [Int], chunkSize: Int
    ) throws -> (nextToken: Int, offset: Int, keys: [Float], values: [Float]) {
        let caches: [KVCache] = [KVCacheSimple()]
        var decoder = MLXDecoder(model: model, cache: caches, prefillChunkSize: chunkSize)
        let next = try decoder.prefill(promptTokens)
        let kv = caches[0]
        return (next, kv.offset, kv.state[0].asArray(Float.self), kv.state[1].asArray(Float.self))
    }

    private func assertChunkedPrefillMatchesWholePromptForward(
        promptTokens: [Int], chunkSize: Int,
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let chunked = try runPrefill(promptTokens: promptTokens, chunkSize: chunkSize)
        let reference = try runPrefill(
            promptTokens: promptTokens,
            chunkSize: wholePromptChunkSize(for: promptTokens.count))

        XCTAssertEqual(
            chunked.nextToken, reference.nextToken,
            "sampled token diverged between chunked and whole-prompt prefill", file: file,
            line: line)
        XCTAssertEqual(
            chunked.offset, reference.offset, "cache offset diverged", file: file, line: line)
        XCTAssertEqual(chunked.keys, reference.keys, "cache keys diverged", file: file, line: line)
        XCTAssertEqual(
            chunked.values, reference.values, "cache values diverged", file: file, line: line)
    }

    func testPromptShorterThanOneChunkMatchesWholePromptForward() throws {
        try assertChunkedPrefillMatchesWholePromptForward(
            promptTokens: [3, 1, 4], chunkSize: 4)
    }

    func testPromptExactlyOneChunkMatchesWholePromptForward() throws {
        try assertChunkedPrefillMatchesWholePromptForward(
            promptTokens: [3, 1, 4, 1], chunkSize: 4)
    }

    func testPromptExactlyNChunksMatchesWholePromptForward() throws {
        try assertChunkedPrefillMatchesWholePromptForward(
            promptTokens: [3, 1, 4, 1, 5, 9, 2, 6], chunkSize: 4)
    }

    func testRaggedTailPromptMatchesWholePromptForward() throws {
        try assertChunkedPrefillMatchesWholePromptForward(
            promptTokens: [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5], chunkSize: 4)
    }

    func testSingleTokenPromptMatchesWholePromptForward() throws {
        try assertChunkedPrefillMatchesWholePromptForward(
            promptTokens: [7], chunkSize: 4)
    }
}

/// The capacity model prices the prefill transient as if prefill were chunked at
/// `CapacityModel.defaultPrefillChunkTokens`; `MLXDecoder` is what actually chunks it. If those two
/// constants drift apart, the fit check silently stops describing what the runtime does — and
/// because the fit verdict is what gates whether a model is admitted at all, the failure mode is a
/// green verdict for a configuration that OOMs.
///
/// That is not hypothetical: before cycle 46 the runtime chunked nothing at all while this term
/// priced a 2048-token chunk, and `transientPrefillPeakBytes`'s own comment records that dropping
/// the term is what killed processes at the 7K wall. This test is the guard that the pairing stays
/// deliberate.
final class MLXDecoderPrefillChunkSizeMatchesCapacityModelTests: XCTestCase {
    func testDecoderPrefillChunkSizeEqualsCapacityModelPricedChunkTokens() {
        XCTAssertEqual(
            MLXDecoder.defaultPrefillChunkSize,
            CapacityModel.defaultPrefillChunkTokens,
            """
            MLXDecoder chunks prefill at \(MLXDecoder.defaultPrefillChunkSize) tokens while \
            CapacityModel prices the transient peak for \(CapacityModel.defaultPrefillChunkTokens). \
            Change both together, or the fit check no longer describes the runtime.
            """)
    }
}
