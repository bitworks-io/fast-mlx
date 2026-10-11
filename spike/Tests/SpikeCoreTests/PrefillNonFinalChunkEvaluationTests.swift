import HarnessCore
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import SpikeCore

/// Non-final prefill chunks evaluate the caches' state instead of their logits.
///
/// Prefill runs the prompt through fixed-size chunks and only the FINAL chunk's last row is ever
/// sampled from. Evaluating every chunk's logits therefore forced a `[1, chunk, vocab]` lm_head
/// projection per chunk whose result was discarded. The vendored `LLMModel.prepare` idiom
/// evaluates the cache on non-final chunks; `MLXDecoder.prefillChunkEvaluationTargets` is the seam
/// both `MLXDecoder` and `CompiledMLXDecoder`'s split-attention loop use for that choice.
///
/// Two layers of proof, because they fail differently:
/// - the SEAM tests assert WHAT is evaluated (logits excluded, every layer's state included) —
///   an exactness test alone cannot see a head projection that is merely wasted;
/// - the EXACTNESS tests assert that dropping the logits evaluation changed no observable value
///   (last logits, cache contents, greedy continuation) against the unchunked/old path.

// MARK: - Seam

final class PrefillChunkEvaluationTargetsTests: XCTestCase {
    private func populatedSimpleCaches(layers: Int) -> [KVCache] {
        (0..<layers).map { layer in
            let cache = KVCacheSimple()
            // Distinct contents per layer so a seam that returned only one layer's arrays (or one
            // array reused for every layer) cannot satisfy the identity checks below.
            let keys = MLXArray.ones([1, 2, 4, 8]) * Float(layer + 1)
            let values = MLXArray.ones([1, 2, 4, 8]) * Float(-(layer + 1))
            _ = cache.update(keys: keys, values: values)
            return cache
        }
    }

    private func contains(_ targets: [MLXArray], _ array: MLXArray) -> Bool {
        targets.contains { $0 === array }
    }

    func testNonFinalChunkExcludesLogitsAndIncludesEveryLayersCacheState() {
        let caches = populatedSimpleCaches(layers: 3)
        let logits = MLXArray.zeros([1, 4, 32])

        let targets = MLXDecoder.prefillChunkEvaluationTargets(
            chunkLogits: logits, caches: caches, isFinal: false)

        XCTAssertFalse(
            contains(targets, logits),
            "a non-final chunk must not evaluate its logits (the lm_head projection is unread)")
        for (layer, cache) in caches.enumerated() {
            let state = cache.innerState()
            XCTAssertFalse(state.isEmpty, "layer \(layer) fixture has no state to evaluate")
            for array in state {
                XCTAssertTrue(
                    contains(targets, array),
                    "layer \(layer)'s cache state is missing from the non-final targets")
            }
        }
    }

    func testFinalChunkEvaluatesExactlyItsLogits() {
        let caches = populatedSimpleCaches(layers: 3)
        let logits = MLXArray.zeros([1, 4, 32])

        let targets = MLXDecoder.prefillChunkEvaluationTargets(
            chunkLogits: logits, caches: caches, isFinal: true)

        XCTAssertEqual(targets.count, 1, "the final chunk keeps today's behavior: logits only")
        XCTAssertTrue(contains(targets, logits))
    }

    func testRecurrentStateCachesAreCoveredAlongsideAttentionCaches() {
        let recurrent = ArraysCache(size: 2)
        recurrent[0] = MLXArray.ones([1, 4])
        recurrent[1] = MLXArray.ones([1, 4]) * 2
        let attention = populatedSimpleCaches(layers: 1)[0]
        let caches: [KVCache] = [recurrent, attention]
        let logits = MLXArray.zeros([1, 4, 32])

        let targets = MLXDecoder.prefillChunkEvaluationTargets(
            chunkLogits: logits, caches: caches, isFinal: false)

        XCTAssertFalse(contains(targets, logits))
        for cache in caches {
            for array in cache.innerState() {
                XCTAssertTrue(contains(targets, array))
            }
        }
    }

    /// `CompiledMLXDecoder` holds `[any CompiledCache]`. Those caches keep the in-graph position
    /// in `offsetArr` and the K/V payload in packed buffers; if either were missing from
    /// `innerState()` the next chunk would read a stale (or still-lazy) position. Each cache
    /// type's `innerState()` must therefore include the offset array and every allocated buffer.
    func testCompiledCacheTypesExposeOffsetAndBuffersToTheSeam() throws {
        let fp16 = CompiledKVCache(capacity: 8)
        let keys = MLXArray.ones([1, 1, 3, 128], dtype: .float16)
        _ = fp16.update(keys: keys, values: keys)
        let affine = try XCTUnwrap(
            KVCacheKind.affine(.k4v2G64).makeCache(
                capacity: 8, affineAttentionMode: .splitQuantizedMM) as? AffineKVCache)
        _ = affine.update(keys: keys, values: keys)

        let caches: [any CompiledCache] = [fp16, affine]
        let logits = MLXArray.zeros([1, 3, 32])
        let targets = MLXDecoder.prefillChunkEvaluationTargets(
            chunkLogits: logits, caches: caches, isFinal: false)

        XCTAssertFalse(contains(targets, logits))
        XCTAssertTrue(contains(targets, fp16.offsetArr), "fp16 in-graph offset not evaluated")
        XCTAssertTrue(contains(targets, try XCTUnwrap(fp16.keysBuf)))
        XCTAssertTrue(contains(targets, try XCTUnwrap(fp16.valuesBuf)))
        XCTAssertTrue(contains(targets, affine.offsetArr), "affine in-graph offset not evaluated")
        for array in affine.innerState() {
            XCTAssertTrue(contains(targets, array), "an affine cache buffer is not evaluated")
        }
        XCTAssertGreaterThan(affine.innerState().count, 1)
    }
}

// MARK: - Exactness fixtures

/// Tiny random-weight decoder-only model with a REAL lm_head matmul, RoPE, causal attention over
/// the cache, and (optionally) a recurrent mixing layer whose state lives in an `ArraysCache`.
/// Unlike the order-sensitive fixture in `MLXDecoderCacheFactoryTests`, its forward is fully lazy
/// (no host readback inside), so it genuinely exercises "which arrays does the loop evaluate":
/// a cache that is not evaluated between chunks would stay a lazy graph and still be correct, but
/// a cache that were evaluated WRONGLY (stale offset, dropped write) would change values.
private final class LazyProbeModel: Module, LanguageModel {
    enum LayerKind { case attention, recurrent }

    static let vocab = 32
    static let dim = 16
    static let heads = 2
    static let headDim = 8

    let kinds: [LayerKind]
    private let embedding: MLXArray
    private let wq: [MLXArray]
    private let wk: [MLXArray]
    private let wv: [MLXArray]
    private let wo: [MLXArray]
    private let wr: [MLXArray]
    private let head: MLXArray

    /// Every forward's logits, in call order — lets a test read the final prompt chunk's last row
    /// without the decoder exposing it.
    private(set) var recordedLogits: [MLXArray] = []

    init(kinds: [LayerKind], seed: UInt64) {
        self.kinds = kinds
        MLXRandom.seed(seed)
        func weight(_ shape: [Int]) -> MLXArray {
            let array = MLXRandom.normal(shape) * 0.4
            eval(array)
            return array
        }
        let d = Self.dim
        embedding = weight([Self.vocab, d])
        wq = kinds.map { _ in weight([d, d]) }
        wk = kinds.map { _ in weight([d, d]) }
        wv = kinds.map { _ in weight([d, d]) }
        wo = kinds.map { _ in weight([d, d]) }
        wr = kinds.map { _ in weight([d, d]) }
        head = weight([d, Self.vocab])
        super.init()
    }

    func resetRecording() { recordedLogits = [] }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        kinds.map { kind -> KVCache in
            switch kind {
            case .attention: return KVCacheSimple()
            case .recurrent: return ArraysCache(size: 1)
            }
        }
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        guard let cache, cache.count == kinds.count else {
            fatalError("LazyProbeModel requires one cache per layer")
        }
        let t = inputs.dim(1)
        let d = Self.dim
        let heads = Self.heads
        let headDim = Self.headDim
        var h = embedding[inputs] // [1, T, D]
        for (i, kind) in kinds.enumerated() {
            switch kind {
            case .attention:
                let layerCache = cache[i]
                func split(_ x: MLXArray) -> MLXArray {
                    x.reshaped([1, t, heads, headDim]).transposed(0, 2, 1, 3)
                }
                var q = split(matmul(h, wq[i]))
                var k = split(matmul(h, wk[i]))
                let v = split(matmul(h, wv[i]))
                q = MLXFast.RoPE(
                    q, dimensions: headDim, traditional: false, base: 10_000, scale: 1,
                    offset: layerCache.offset)
                k = MLXFast.RoPE(
                    k, dimensions: headDim, traditional: false, base: 10_000, scale: 1,
                    offset: layerCache.offset)
                let mask = createAttentionMask(h: h, cache: layerCache)
                let attended = attentionWithCacheUpdate(
                    queries: q, keys: k, values: v, cache: layerCache,
                    scale: 1 / Float(headDim).squareRoot(), mask: mask)
                let merged = attended.transposed(0, 2, 1, 3).reshaped([1, t, d])
                h = h + matmul(merged, wo[i])
            case .recurrent:
                guard let state = cache[i] as? ArraysCache else {
                    fatalError("recurrent layer requires an ArraysCache")
                }
                let inputsToScan = matmul(h, wr[i]) // [1, T, D]
                var carried = state[0] ?? MLXArray.zeros([1, d])
                var outputs: [MLXArray] = []
                for step in 0..<t {
                    carried = tanh(inputsToScan[0..., step, 0...] + 0.5 * carried)
                    outputs.append(carried)
                }
                state[0] = carried
                h = h + stacked(outputs, axis: 1)
            }
        }
        let logits = matmul(h, head) // [1, T, V] — the lm_head projection
        recordedLogits.append(logits)
        return logits
    }
}

private func cacheContents(_ caches: [KVCache]) -> [[[Float]]] {
    caches.map { cache in cache.state.map { $0.asArray(Float.self) } }
}

// MARK: - Exactness: MLXDecoder

final class MLXDecoderNonFinalChunkExactnessTests: XCTestCase {
    private static let chunkSize = 4
    private static let continuation = 8

    private struct Outcome {
        var greedyTokens: [Int]
        var firstLogprob: DecodedTokenLogprob
        var lastPromptLogits: [Float]
        var cache: [[[Float]]]
    }

    private func run(
        model: LazyProbeModel, prompt: [Int], chunkSize: Int
    ) throws -> Outcome {
        model.resetRecording()
        let caches = model.newCache(parameters: nil)
        var decoder = MLXDecoder(model: model, cache: caches, prefillChunkSize: chunkSize)
        let (first, logprob) = try decoder.prefillWithLogprob(prompt, topN: 5)
        let chunkCount = (prompt.count + chunkSize - 1) / chunkSize
        // `prefillWithLogprob` also submits the first decode forward, so read the prompt's final
        // chunk logits (forward number `chunkCount`) BEFORE stepping further.
        let lastPromptLogits = model.recordedLogits[chunkCount - 1][0, -1, 0...]
            .asArray(Float.self)
        // Cache contents at the end of prefill+first lookahead are compared after the identical
        // number of decode steps on both sides, below.
        var tokens = [first]
        while tokens.count < Self.continuation {
            tokens.append(try decoder.step(last: tokens.last!))
        }
        return Outcome(
            greedyTokens: tokens, firstLogprob: logprob, lastPromptLogits: lastPromptLogits,
            cache: cacheContents(caches))
    }

    /// The pre-change prefill: evaluate every chunk's logits at the SAME chunk size, then the same
    /// greedy continuation the decoder performs (its `prefill` submits the first token's forward
    /// before returning, so `continuation` decode forwards are in the cache at the end).
    /// Independent of the production loop.
    private func oldPathOutcome(
        model: LazyProbeModel, prompt: [Int], chunkSize: Int
    ) -> (lastLogits: [Float], tokens: [Int], cache: [[[Float]]]) {
        let caches = model.newCache(parameters: nil)
        var start = 0
        var last: MLXArray!
        while start < prompt.count {
            let end = min(start + chunkSize, prompt.count)
            let ids = MLXArray(Array(prompt[start..<end])).reshaped([1, end - start])
            let logits = model(ids, cache: caches)
            eval(logits)
            last = logits
            start = end
        }
        let lastLogits = last[0, -1, 0...].asArray(Float.self)
        var tokens = [argMax(last[0..., -1, 0...], axis: -1).item(Int.self)]
        for _ in 0..<Self.continuation {
            let y = MLXArray([tokens.last!]).reshaped([1, 1])
            let logits = model(y, cache: caches)
            tokens.append(argMax(logits[0..., -1, 0...], axis: -1).item(Int.self))
        }
        tokens.removeLast() // the last forward only fed the cache
        return (lastLogits, tokens, cacheContents(caches))
    }

    private func assertExact(
        kinds: [LazyProbeModel.LayerKind], seed: UInt64,
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let model = LazyProbeModel(kinds: kinds, seed: seed)
        let prompt = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9] // ragged: 13 = 4 + 4 + 4 + 1
        XCTAssertEqual(prompt.count % Self.chunkSize, 1, file: file, line: line)

        let chunked = try run(model: model, prompt: prompt, chunkSize: Self.chunkSize)
        let old = oldPathOutcome(model: model, prompt: prompt, chunkSize: Self.chunkSize)

        // EXACT (array equality, no tolerance) against the evaluate-every-chunk loop at the same
        // chunk size: the only difference between the two is whether non-final logits were
        // evaluated, which must change no value.
        XCTAssertEqual(
            chunked.lastPromptLogits, old.lastLogits,
            "final prompt logits diverged from the evaluate-every-chunk loop", file: file,
            line: line)
        XCTAssertEqual(
            chunked.greedyTokens, old.tokens,
            "greedy continuation diverged from the evaluate-every-chunk loop", file: file,
            line: line)
        XCTAssertEqual(
            chunked.cache, old.cache,
            "final cache contents diverged from the evaluate-every-chunk loop", file: file,
            line: line)

        // The reported first-token logprob is computed from the row `prefill` actually sampled
        // from (the loop's RETURN value, which the recorded model logits above do not cover), so
        // pin it against an independent log-softmax of the old loop's final row.
        let reference = logSoftmax(MLXArray(old.lastLogits), axis: -1)
        let referenceValues = reference.asArray(Float.self)
        XCTAssertEqual(chunked.firstLogprob.tokenID, old.tokens[0], file: file, line: line)
        XCTAssertEqual(
            chunked.firstLogprob.logprob, referenceValues[old.tokens[0]], accuracy: 1e-6,
            "first-token logprob does not describe the final chunk's last row", file: file,
            line: line)
        for candidate in chunked.firstLogprob.top {
            XCTAssertEqual(
                candidate.logprob, referenceValues[candidate.tokenID], accuracy: 1e-6,
                "top-N candidate logprob does not describe the final chunk's last row",
                file: file, line: line)
        }

        // Sanity vs the whole-prompt forward. NOT bit-exact on this fixture (different sequence
        // lengths pick different fused kernels/tilings; measured ~1e-2 on logits of magnitude ~3,
        // consistent with the repo's earlier chunking measurements), so exactness is asserted
        // against the same-chunk-size reference above and this is a coarse tolerance check on the
        // logits (a wrong mask/offset would move them by O(1)) plus an exact check on the discrete
        // outputs.
        let whole = try run(model: model, prompt: prompt, chunkSize: prompt.count)
        XCTAssertEqual(chunked.greedyTokens, whole.greedyTokens, file: file, line: line)
        XCTAssertEqual(chunked.firstLogprob.tokenID, whole.firstLogprob.tokenID, file: file, line: line)
        for (a, b) in zip(chunked.lastPromptLogits, whole.lastPromptLogits) {
            XCTAssertEqual(a, b, accuracy: 0.1, file: file, line: line)
        }

        // Anti-vacuity: the fixture's greedy continuation must not be a constant, or the token
        // comparison above could not see a wrong cache.
        XCTAssertGreaterThan(
            Set(chunked.greedyTokens).count, 1, "degenerate greedy continuation",
            file: file, line: line)
    }

    func testDenseModelNonFinalChunkSkipLeavesEveryValueUnchanged() throws {
        try assertExact(kinds: [.attention, .attention], seed: 11)
    }

    func testRecurrentHybridModelNonFinalChunkSkipLeavesEveryValueUnchanged() throws {
        try assertExact(kinds: [.recurrent, .attention, .recurrent, .attention], seed: 23)
    }

    /// A prompt that fits in one chunk has only a final chunk: control flow (eval of the logits)
    /// is unchanged, so it must match the unchunked path trivially.
    func testPromptWithinOneChunkIsUnchanged() throws {
        let model = LazyProbeModel(kinds: [.attention, .recurrent], seed: 5)
        let prompt = [2, 7, 1]
        let a = try run(model: model, prompt: prompt, chunkSize: Self.chunkSize)
        let b = try run(model: model, prompt: prompt, chunkSize: prompt.count)
        XCTAssertEqual(a.lastPromptLogits, b.lastPromptLogits)
        XCTAssertEqual(a.greedyTokens, b.greedyTokens)
        XCTAssertEqual(a.cache, b.cache)
    }
}

// MARK: - Exactness: CompiledMLXDecoder split-attention loop

/// Two-layer attention model for the split-attention route (head dim 128, one KV head, quantized
/// affine cache). Logits come from a real lm_head matmul over a state that depends on the cache
/// contents of BOTH layers, so a stale or dropped cache write changes the greedy tokens.
private final class SplitRouteProbeModel: Module, LanguageModel, KVCacheDimensionProvider {
    let kvHeads = [1, 1]
    static let vocab = 32
    static let dim = 32
    static let headDim = 128

    private let embedding: MLXArray
    private let wq: [MLXArray]
    private let wk: [MLXArray]
    private let wv: [MLXArray]
    private let wo: [MLXArray]
    private let head: MLXArray

    override init() {
        MLXRandom.seed(77)
        func weight(_ shape: [Int]) -> MLXArray {
            let array = MLXRandom.normal(shape) * 0.2
            eval(array)
            return array
        }
        let d = Self.dim
        // Embedding scale >> attention scale so the current token dominates the logits: the greedy
        // token then tracks WHICH row was read, not just the (nearly position-independent)
        // attended average.
        embedding = weight([Self.vocab, d]) * 5
        wq = (0..<2).map { _ in weight([d, 2 * Self.headDim]) }
        wk = (0..<2).map { _ in weight([d, Self.headDim]) }
        wv = (0..<2).map { _ in weight([d, Self.headDim]) }
        wo = (0..<2).map { _ in weight([2 * Self.headDim, d]) }
        head = weight([d, Self.vocab])
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        guard let cache, cache.count == 2 else {
            fatalError("SplitRouteProbeModel requires two caches")
        }
        let t = inputs.dim(1)
        var h = embedding[inputs] // [1, T, D]
        for layer in 0..<2 {
            let queries = matmul(h, wq[layer])
                .reshaped([1, t, 2, Self.headDim]).transposed(0, 2, 1, 3)
            let keys = matmul(h, wk[layer])
                .reshaped([1, t, 1, Self.headDim]).transposed(0, 2, 1, 3)
            let values = matmul(h, wv[layer])
                .reshaped([1, t, 1, Self.headDim]).transposed(0, 2, 1, 3)
            let mask = cache[layer].makeMask(n: t, windowSize: nil, returnArray: true)
            let attended = attentionWithCacheUpdate(
                queries: queries.asType(.float16), keys: keys.asType(.float16),
                values: values.asType(.float16), cache: cache[layer],
                scale: 1 / Float(Self.headDim).squareRoot(), mask: mask)
            let merged = attended.transposed(0, 2, 1, 3).reshaped([1, t, 2 * Self.headDim])
            h = h + matmul(merged.asType(.float32), wo[layer])
        }
        return matmul(h, head)
    }
}

final class CompiledSplitRouteNonFinalChunkExactnessTests: XCTestCase {
    func testSplitChunkLoopMatchesTheEvaluateEveryChunkLoop() throws {
        // Several prompts so a wrong-row read cannot hide behind a coincidentally equal argmax.
        for multiplier in [7, 11, 13] {
            try assertSplitLoopMatches(
                prompt: (0..<1030).map { ($0 * multiplier % 31) + 1 }) // 512 + 512 + 6
        }
    }

    private func assertSplitLoopMatches(
        prompt: [Int], file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let model = SplitRouteProbeModel()
        // 512-token split chunks: two non-final chunks and a ragged tail.
        let continuation = 8

        var decoder = CompiledMLXDecoder(
            model: model, reserve: 16, kvCache: .affine(.k4v2G64),
            affineAttentionMode: .splitQuantizedMM)
        var tokens = [try decoder.prefill(prompt)]
        while tokens.count < continuation {
            tokens.append(try decoder.step(last: tokens.last!))
        }

        // Independent reference: the pre-change loop (eval every chunk's logits) over the same
        // cache kind, then uncompiled greedy decode.
        let chunk = KVTunerCandidateRuntimeContract.cacheGrowthChunkTokens
        let capacity = ((prompt.count + 16 + chunk - 1) / chunk) * chunk
        let caches = try KVCacheKind.affine(.k4v2G64).makeCaches(
            layerCount: 2, capacity: capacity, affineAttentionMode: .splitQuantizedMM)
        var start = 0
        var lastLogits: MLXArray!
        while start < prompt.count {
            let end = min(start + 512, prompt.count)
            let ids = MLXArray(Array(prompt[start..<end]).map { Int32($0) })
                .reshaped([1, end - start])
            let logits = model(ids, cache: caches)
            eval(logits)
            lastLogits = logits
            start = end
        }
        var expected = [argMax(lastLogits[0..., -1, 0...], axis: -1).item(Int.self)]
        while expected.count < continuation {
            let y = MLXArray([Int32(expected.last!)]).reshaped([1, 1])
            let logits = model(y, cache: caches)
            expected.append(argMax(logits[0..., -1, 0...], axis: -1).item(Int.self))
        }

        XCTAssertEqual(
            tokens, expected,
            "split-attention chunked prefill diverged from the evaluate-every-chunk loop",
            file: file, line: line)
    }
}
