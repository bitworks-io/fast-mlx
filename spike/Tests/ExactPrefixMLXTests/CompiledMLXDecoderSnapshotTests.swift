import HarnessCore
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import SpikeCore

private struct FixedSnapshotDrafter: SpecDrafter {
    func propose(context: [Int], maxDraft: Int) -> [Int] {
        Array(repeating: 1, count: min(1, maxDraft))
    }
}

private final class TinyPrefixSnapshotModel:
    Module, LanguageModel, KVCacheDimensionProvider
{
    let kvHeads: [Int]
    private let cacheDType: DType
    private let vocabularySize = 4_096
    private(set) var forwardTokenCounts: [Int] = []

    init(
        layerCount: Int = 1,
        cacheDType: DType = .float16
    ) {
        // Two KV heads keep this compiled-test signature distinct from the one-head
        // continuous-batching fixture that runs later in the same MLX test process.
        kvHeads = Array(repeating: 2, count: layerCount)
        self.cacheDType = cacheDType
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?
    ) -> MLXArray {
        forwardTokenCounts.append(inputs.dim(1))
        guard let caches = cache, caches.count == kvHeads.count,
            let firstCache = caches.first
        else {
            preconditionFailure("tiny prefix model requires its exact cache list")
        }
        let scalar = inputs.asType(cacheDType).reshaped([
            inputs.dim(0), 1, inputs.dim(1), 1,
        ])
        let values = broadcast(
            scalar,
            to: [inputs.dim(0), 2, inputs.dim(1), 1])
        let (firstKeys, _) = firstCache.update(
            keys: values, values: values)
        for layerCache in caches.dropFirst() {
            _ = layerCache.update(keys: values, values: values)
        }

        let nextToken = firstKeys.sum(axes: [1, 2, 3])
            .asType(.int32) + 1
        let target = broadcast(
            nextToken.reshaped([inputs.dim(0), 1, 1]),
            to: [inputs.dim(0), inputs.dim(1), 1])
        let vocabulary = MLXArray(Int32(0) ..< Int32(vocabularySize))
            .reshaped([1, 1, vocabularySize])
        return (target .== vocabulary).asType(.float32) * 100
    }
}

final class CompiledMLXDecoderSnapshotTests: XCTestCase {
    override func tearDown() {
        // Submit-first decode intentionally leaves one lazy token in flight. Drain the shared
        // Metal stream before the decoder's compiled closures are reused by a later test suite.
        Stream().synchronize()
        super.tearDown()
    }

    private func assertSnapshotError(
        _ expected: CompiledMLXDecoderSnapshotError,
        _ body: () throws -> Void,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(try body(), file: file, line: line) {
            XCTAssertEqual(
                $0 as? CompiledMLXDecoderSnapshotError, expected,
                file: file, line: line)
        }
    }

    private func decode(
        _ decoder: inout CompiledMLXDecoder,
        first: Int,
        additionalTokens: Int
    ) -> [Int] {
        var output = [first]
        var token = first
        for _ in 0 ..< additionalTokens {
            token = decoder.step(last: token)
            output.append(token)
        }
        return output
    }

    func testPromptSnapshotExactHitSkipsPromptPrefillAndMatchesColdControl() throws {
        let prompt = [2, 3, 4]
        let model = TinyPrefixSnapshotModel()
        var decoder = CompiledMLXDecoder(
            model: model, reserve: 1, kvCache: .fp16)

        let staged = try decoder.prefillCapturingPromptSnapshot(prompt)
        XCTAssertEqual(staged.snapshot.logicalTokenCount, prompt.count)
        XCTAssertEqual(staged.snapshot.nextTokenID, staged.firstToken)
        XCTAssertEqual(staged.snapshot.layerCount, 1)
        XCTAssertGreaterThan(staged.snapshot.arrayNBytes, 0)
        XCTAssertGreaterThan(staged.snapshot.controlNBytes, 0)
        XCTAssertEqual(
            staged.snapshot.totalNBytes,
            staged.snapshot.arrayNBytes + staged.snapshot.controlNBytes)

        _ = decode(&decoder, first: staged.firstToken, additionalTokens: 2)
        decoder.reset()
        let callsBeforeRestore = model.forwardTokenCounts.count
        let restoredFirst = try decoder.prefillRestoredPrefix(
            staged.snapshot, tailTokens: [])
        let restored = decode(
            &decoder, first: restoredFirst, additionalTokens: 3)

        let controlModel = TinyPrefixSnapshotModel()
        var control = CompiledMLXDecoder(
            model: controlModel, reserve: 1, kvCache: .fp16)
        let controlFirst = control.prefill(prompt)
        let expected = decode(
            &control, first: controlFirst, additionalTokens: 3)

        XCTAssertEqual(restored, expected)
        XCTAssertEqual(
            Array(model.forwardTokenCounts.dropFirst(callsBeforeRestore)),
            [],
            "an exact hit must not rebuild the prompt graph; compiled decode replays do not re-enter Swift")
    }

    func testBFloat16PromptSnapshotRoundTripsExactly() throws {
        let prompt = [2, 3, 4]
        let model = TinyPrefixSnapshotModel(
            cacheDType: .bfloat16)
        var decoder = CompiledMLXDecoder(
            model: model, reserve: 1, kvCache: .fp16)

        let staged =
            try decoder.prefillCapturingPromptSnapshot(prompt)
        XCTAssertEqual(
            staged.snapshot.layerSnapshots.first?.keyDType,
            .bfloat16)
        decoder.reset()
        let restoredFirst = try decoder.prefillRestoredPrefix(
            staged.snapshot, tailTokens: [5])
        let restored = decode(
            &decoder, first: restoredFirst, additionalTokens: 2)

        let controlModel = TinyPrefixSnapshotModel(
            cacheDType: .bfloat16)
        var control = CompiledMLXDecoder(
            model: controlModel, reserve: 1, kvCache: .fp16)
        let controlFirst = control.prefill(prompt + [5])
        let expected = decode(
            &control, first: controlFirst, additionalTokens: 2)
        XCTAssertEqual(restored, expected)
    }

    func testPromptSnapshotPlusTailMatchesUninterruptedControlAndEvaluatesOnlyTail() throws {
        let prefix = [1, 2, 3]
        let tail = [4, 5, 6, 7]
        let model = TinyPrefixSnapshotModel()
        var decoder = CompiledMLXDecoder(
            model: model, reserve: 1, kvCache: .fp16)

        let staged = try decoder.prefillCapturingPromptSnapshot(prefix)
        _ = decode(&decoder, first: staged.firstToken, additionalTokens: 2)
        decoder.reset()
        let callsBeforeRestore = model.forwardTokenCounts.count

        let restoredFirst = try decoder.prefillRestoredPrefix(
            staged.snapshot, tailTokens: tail)
        let restored = decode(
            &decoder, first: restoredFirst, additionalTokens: 3)

        let controlModel = TinyPrefixSnapshotModel()
        var control = CompiledMLXDecoder(
            model: controlModel, reserve: 1, kvCache: .fp16)
        let controlFirst = control.prefill(prefix + tail)
        let expected = decode(
            &control, first: controlFirst, additionalTokens: 3)

        XCTAssertEqual(restored, expected)
        XCTAssertEqual(
            Array(model.forwardTokenCounts.dropFirst(callsBeforeRestore)),
            [tail.count],
            "the restored path must physically prefill only the new tail; compiled decode replays do not re-enter Swift")
    }

    func testFinalContextSnapshotRestoresGeneratedContextBeforeNewTail() throws {
        let prompt = [2, 2]
        let tail = [3, 1]
        let model = TinyPrefixSnapshotModel()
        var decoder = CompiledMLXDecoder(
            model: model, reserve: 1, kvCache: .fp16)

        let first = decoder.prefill(prompt)
        let generated = decode(
            &decoder, first: first, additionalTokens: 1)
        let snapshot = try decoder.captureContinuationSnapshot()
        XCTAssertEqual(
            snapshot.logicalTokenCount,
            prompt.count + generated.count)

        decoder.reset()
        let restoredFirst = try decoder.prefillRestoredPrefix(
            snapshot, tailTokens: tail)
        let restored = decode(
            &decoder, first: restoredFirst, additionalTokens: 2)

        let controlModel = TinyPrefixSnapshotModel()
        var control = CompiledMLXDecoder(
            model: controlModel, reserve: 1, kvCache: .fp16)
        let controlFirst = control.prefill(prompt + generated + tail)
        let expected = decode(
            &control, first: controlFirst, additionalTokens: 2)
        XCTAssertEqual(restored, expected)
    }

    func testRestoredPrefixPlusTailGrowsCapacityWithoutChangingOutput() throws {
        let prefix = [1, 1]
        let tail = Array(repeating: 1, count: 255)
        let model = TinyPrefixSnapshotModel()
        var decoder = CompiledMLXDecoder(
            model: model, reserve: 1, kvCache: .fp16)
        let staged = try decoder.prefillCapturingPromptSnapshot(prefix)

        decoder.reset()
        let restored = try decoder.prefillRestoredPrefix(
            staged.snapshot, tailTokens: tail)

        let controlModel = TinyPrefixSnapshotModel()
        var control = CompiledMLXDecoder(
            model: controlModel, reserve: 1, kvCache: .fp16)
        XCTAssertEqual(restored, control.prefill(prefix + tail))
    }

    func testSnapshotRoutesAndMalformedRestoreFailClosed() throws {
        let unsupportedModel = TinyPrefixSnapshotModel()
        var unsupported = CompiledMLXDecoder(
            model: unsupportedModel,
            reserve: 1,
            kvCache: .affine(.k4v2G64))
        assertSnapshotError(.unsupportedCacheKind) {
            _ = try unsupported.prefillCapturingPromptSnapshot([1, 2])
        }
        XCTAssertTrue(unsupportedModel.forwardTokenCounts.isEmpty)

        let uninitialized = CompiledMLXDecoder(
            model: TinyPrefixSnapshotModel(),
            reserve: 1,
            kvCache: .fp16)
        var fresh = uninitialized
        let sourceModel = TinyPrefixSnapshotModel(layerCount: 2)
        var source = CompiledMLXDecoder(
            model: sourceModel, reserve: 1, kvCache: .fp16)
        let valid = try source.prefillCapturingPromptSnapshot([1, 2]).snapshot
        assertSnapshotError(.decoderNotInitialized) {
            _ = try fresh.prefillRestoredPrefix(
                valid, tailTokens: [])
        }

        source.reset()
        let malformed = CompiledMLXDecoderSnapshot(
            logicalTokenCount: valid.logicalTokenCount,
            nextTokenID: valid.nextTokenID,
            layerSnapshots: Array(valid.layerSnapshots.dropLast()))
        assertSnapshotError(.layerCountMismatch) {
            _ = try source.prefillRestoredPrefix(
                malformed, tailTokens: [])
        }

        let secondLayer = valid.layerSnapshots[1]
        let wrongGeometryLayer = CompiledKVCacheSnapshot(
            rank: secondLayer.rank,
            batchSize: secondLayer.batchSize,
            kvHeadCount: secondLayer.kvHeadCount + 1,
            tokenCount: secondLayer.tokenCount,
            headDimension: secondLayer.headDimension,
            keyDType: secondLayer.keyDType,
            valueDType: secondLayer.valueDType,
            keyNBytes: secondLayer.keyNBytes,
            valueNBytes: secondLayer.valueNBytes,
            keys: secondLayer.keys,
            values: secondLayer.values)
        let heterogeneous = CompiledMLXDecoderSnapshot(
            logicalTokenCount: valid.logicalTokenCount,
            nextTokenID: valid.nextTokenID,
            layerSnapshots: [
                valid.layerSnapshots[0],
                wrongGeometryLayer,
            ])
        assertSnapshotError(.layerGeometryMismatch(layer: 1)) {
            _ = try source.prefillRestoredPrefix(
                heterogeneous, tailTokens: [])
        }

        let control = CompiledMLXDecoder(
            model: TinyPrefixSnapshotModel(),
            reserve: 1,
            kvCache: .fp16)
        var recovered = control
        let recoveredFirst = source.prefill([1, 2])
        let controlFirst = recovered.prefill([1, 2])
        XCTAssertEqual(recoveredFirst, controlFirst)
    }

    func testInvalidPromptTailAndPipelineStateFailBeforeReusableStateLeaks() throws {
        let model = TinyPrefixSnapshotModel()
        var decoder = CompiledMLXDecoder(
            model: model, reserve: 1, kvCache: .fp16)

        assertSnapshotError(.invalidPrompt) {
            _ = try decoder.prefillCapturingPromptSnapshot([])
        }
        assertSnapshotError(.invalidPrompt) {
            _ = try decoder.prefillCapturingPromptSnapshot([1, -1])
        }
        XCTAssertTrue(model.forwardTokenCounts.isEmpty)

        let staged = try decoder.prefillCapturingPromptSnapshot([1, 2])
        assertSnapshotError(.invalidPipelineState) {
            _ = try decoder.prefillCapturingPromptSnapshot([1, 2])
        }

        let callsBeforeInvalidTail = model.forwardTokenCounts.count
        assertSnapshotError(.invalidTailToken(position: 1)) {
            _ = try decoder.prefillRestoredPrefix(
                staged.snapshot, tailTokens: [3, -1])
        }
        XCTAssertEqual(
            model.forwardTokenCounts.count,
            callsBeforeInvalidTail,
            "invalid input must fail before prompt or tail evaluation")

        let controlModel = TinyPrefixSnapshotModel()
        var control = CompiledMLXDecoder(
            model: controlModel, reserve: 1, kvCache: .fp16)
        XCTAssertEqual(
            decoder.prefill([1, 2]),
            control.prefill([1, 2]),
            "failure recovery must leave the decoder reusable")
    }

    func testResetClearsLiveSpeculativeBlockWhileRetainingDecoderReuse() throws {
        let model = TinyPrefixSnapshotModel()
        var decoder = CompiledMLXDecoder(
            model: model, reserve: 1, kvCache: .fp16)
        let speculative = decoder.generateSpec(
            prompt: [1, 1],
            maxTokens: 2,
            eos: 4_095,
            spec: SpecDecodeConfig(
                drafter: FixedSnapshotDrafter(),
                maxDraft: 1,
                lookback: 8,
                compiledVerify: true))
        XCTAssertGreaterThan(speculative.stats.verifySteps, 0)
        assertSnapshotError(.speculativeStateUnsupported) {
            _ = try decoder.captureContinuationSnapshot()
        }

        decoder.reset()
        let staged = try decoder.prefillCapturingPromptSnapshot([1, 1])
        decoder.reset()
        XCTAssertEqual(
            try decoder.prefillRestoredPrefix(
                staged.snapshot, tailTokens: []),
            staged.firstToken)
    }
}
