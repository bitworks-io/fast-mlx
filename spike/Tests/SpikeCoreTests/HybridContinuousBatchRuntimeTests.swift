import HarnessCore
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import SpikeCore

// S3 of the hybrid continuous-batching build (design of record: the continuous-batching
// heterogeneous-cache design;
// increment plan: docs/task-inbox/2026-08-20-hybrid-continuous-batch-s3-runtime.md).
// Hybrid runtime tests live in this file, NOT in DenseContinuousBatchRuntimeTests.swift —
// the dense suite is the byte-for-byte regression gate for the unchanged fp16/affine path.

/// Init-only 48-layer toy matching the real qwen3_5 VL proof's layer count. Never invoked:
/// the selection/guard tests only construct the runtime (which calls `newCache` for the
/// layer count) and read diagnostics.
private final class TinyHybrid48LayerInitModel: Module, LanguageModel, KVCacheDimensionProvider {
    let kvHeads = Array(repeating: 1, count: 48)

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        preconditionFailure("init-only toy model must never be invoked")
    }
}

/// 2-layer toy hybrid for the S3 kind-aware runtime seam: layer 0 is a toy recurrence over a REAL
/// `MambaCache` (consumed exactly the way Qwen35 consumes it: concrete `cache as? MambaCache`
/// downcast, zero-state fallback when it fails, write-back through the same object via subscript),
/// layer 1 is real KV attention over the runtime's dense row. Layer 0 being RECURRENT means every
/// "layer 0 is dense" assumption in the runtime fails this test loudly — qwen3_5's real interval-4
/// map has the same property (first dense layer is 3, not 0).
///
/// Token arithmetic (deterministic, fp32, greedy):
///   emitted(t) = (sum of all session tokens through position t) + 3 × (1-based position) + 1
/// The position term lives ONLY in the MambaCache SSM state, so a stateless/stale recurrent path
/// (failed downcast running the zero-state fallback, a dropped write-back, or a fresh cache per
/// step) resets positions and emits DIFFERENT tokens — the silent-stateless trap is caught by
/// exact token equality, not by absence of a crash.
private final class TinyHybridLanguageModel: Module, LanguageModel, KVCacheDimensionProvider {
    let kvHeads = [1, 1]
    private let vocabularySize = 2_048

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        guard let cache, cache.count == 2 else {
            preconditionFailure("tiny hybrid model requires exactly two caches")
        }
        let batch = inputs.dim(0)
        let length = inputs.dim(1)

        // Layer 0 — toy recurrence mirroring the vendored GatedDeltaNet consumption pattern
        // (Qwen35.swift:487,537 downcast; :242-246 zero fallback; :254,:289 write-backs). The
        // SSM slot [B,1,1,1] fp32 accumulates the committed token count (the position clock);
        // the conv slot [B,1,3] carries the last token value so both slots are initialized the
        // way BatchedRecurrentStateCache.merging later requires.
        let mamba = cache[0] as? MambaCache
        let ssmState = mamba?[1] ?? MLXArray.zeros([batch, 1, 1, 1], dtype: .float32)
        let positionsBefore = ssmState.sum(axes: [1, 2, 3]).asType(.int32)
        let chunkPositions = MLXArray(Int32(1) ..< Int32(length + 1)).reshaped([1, length])
        let positions = positionsBefore.reshaped([batch, 1]) + chunkPositions
        if let mamba {
            mamba[1] = ssmState + Float(length)
            mamba[0] = broadcast(
                inputs.asType(.float32)[0..., (length - 1)...].reshaped([batch, 1, 1]),
                to: [batch, 1, 3])
        }

        // Layer 1 — real KV attention history over the dense row (TinyDenseLanguageModel pattern).
        let values = inputs.asType(.float32).reshaped([batch, 1, length, 1])
        let (keys, _) = cache[1].update(keys: values, values: values)
        let flatInputs = inputs.asType(.int32)
        let historyBeforeChunk =
            keys.sum(axes: [1, 2, 3]).asType(.int32) - flatInputs.sum(axis: 1)
        let cumulative = historyBeforeChunk.reshaped([batch, 1]) + flatInputs.cumsum(axis: 1)

        let target = (cumulative + 3 * positions + 1).expandedDimensions(axis: -1)
        let vocabulary = MLXArray(Int32(0) ..< Int32(vocabularySize))
            .reshaped([1, 1, vocabularySize])
        return (target .== vocabulary).asType(.float32) * 100
    }
}

final class HybridContinuousBatchRuntimeTests: XCTestCase {

    private func writeConfig(_ json: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(json.utf8).write(to: dir.appendingPathComponent("config.json"))
        return dir
    }

    /// Same VL-wrapped qwen3_5 fixture as `HybridQwen35ProofArmTests`: 48 layers, interval 4 →
    /// 12 dense layers at {3,7,...,47}, 36 recurrent layers, layer 0 recurrent.
    private func qwen35VLConfigJSON() -> String {
        #"""
        {"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"],
         "text_config":{"model_type":"qwen3_5_text","max_position_embeddings":262144,
           "vocab_size":248320,"num_hidden_layers":48,"full_attention_interval":4,
           "num_key_value_heads":8,"head_dim":128,"torch_dtype":"bfloat16",
           "linear_num_key_heads":16,"linear_num_value_heads":32,
           "linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4}}
        """#
    }

    private func makeHybridProof() throws -> DenseContinuousBatchModelProof {
        let dir = try writeConfig(qwen35VLConfigJSON())
        defer { try? FileManager.default.removeItem(at: dir) }
        return try DenseContinuousBatchModelProof.verifying(
            modelDirectory: dir, allowHybridQwen35: true)
    }

    // MARK: hybrid family + admission-plan selection

    func testHybridProofSelectsHybridAdmissionPlan() throws {
        let proof = try makeHybridProof()
        let runtime = try DenseContinuousBatchRuntime(
            model: TinyHybrid48LayerInitModel(),
            verifiedBy: proof,
            allocationChunk: 256,
            maxContextTokens: 4_096)
        // Dense-only bytes/token: 12 dense layers × 2 (K+V) × 8 heads × 128 dim × 2 bytes.
        // The uniform dense plan would charge all 48 layers (196,608 B/token) — a 4x
        // over-charge that starves admission; the hybrid plan must not.
        XCTAssertEqual(runtime.diagnostics().kvBytesPerToken, 49_152)
        XCTAssertEqual(runtime.resourceSnapshot()?.kvBytesPerToken, 49_152)
    }

    // MARK: hybrid toy runtime: kind-aware prefill + solo decode

    /// Same qwen3_5 shape as the 48-layer fixture but sized to the toy: 2 layers, interval 2 →
    /// layer 0 recurrent, layer 1 dense. fp32 (toy computes in fp32; dense geometry validation
    /// checks element bytes against torch_dtype). Recurrent geometry matches the toy's state:
    /// convDim = 2·(1·1)+(1·1) = 3, conv tail [1,1,3], SSM [1,1,1,1].
    private func tinyHybridConfigJSON() -> String {
        #"""
        {"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"],
         "text_config":{"model_type":"qwen3_5_text","max_position_embeddings":64,
           "vocab_size":2048,"num_hidden_layers":2,"full_attention_interval":2,
           "num_key_value_heads":1,"head_dim":1,"torch_dtype":"float32",
           "linear_num_key_heads":1,"linear_num_value_heads":1,
           "linear_key_head_dim":1,"linear_value_head_dim":1,"linear_conv_kernel_dim":2}}
        """#
    }

    func testHybridToyPrefillAndSoloDecodeAdvancesRecurrentState() throws {
        let dir = try writeConfig(tinyHybridConfigJSON())
        defer { try? FileManager.default.removeItem(at: dir) }
        let proof = try DenseContinuousBatchModelProof.verifying(
            modelDirectory: dir, allowHybridQwen35: true)
        let runtime = try DenseContinuousBatchRuntime(
            model: TinyHybridLanguageModel(),
            verifiedBy: proof,
            allocationChunk: 4,
            maxContextTokens: 64)

        // Chunked prefill [1,2] then [3]: recurrent + dense state must both carry across chunks.
        try runtime.prefill(
            ContinuousBatchRuntimePrefill(
                id: BatchRequestID(1), startToken: 0, tokens: [1, 2],
                isFinal: false, totalPromptTokens: 3, maxOutputTokens: 8))
        try runtime.prefill(
            ContinuousBatchRuntimePrefill(
                id: BatchRequestID(1), startToken: 2, tokens: [3],
                isFinal: true, totalPromptTokens: 3, maxOutputTokens: 8))

        // emitted(t) = session token sum + 3×position + 1 (see TinyHybridLanguageModel):
        //   prefill end: sum 6, pos 3 → staged 16
        //   step 1: sum 22, pos 4 → 35;  step 2: sum 57, pos 5 → 73
        // The position clock exists ONLY in the recurrent MambaCache state. A stateless or
        // stale recurrent layer (failed `cache as? MambaCache` downcast → zero-state fallback,
        // dropped write-back, or a fresh cache per step) resets it and step 2 emits 26, not 35 —
        // exact equality below is the silent-stateless-downcast trap guard.
        let first = try runtime.decode(.solo(BatchRequestID(1), speculationAllowed: false))
        XCTAssertEqual(first.map(\.tokens), [[16]])
        let second = try runtime.decode(.solo(BatchRequestID(1), speculationAllowed: false))
        XCTAssertEqual(second.map(\.tokens), [[35]])
        let third = try runtime.decode(.solo(BatchRequestID(1), speculationAllowed: false))
        XCTAssertEqual(third.map(\.tokens), [[73]])
    }

    // MARK: fail-closed construction guards

    func testHybridRejectsSoloSpeculationAtConstruction() throws {
        // Speculation × recurrent SSM state is physically incompatible: truncate/rollback
        // rewinds dense KV rows but cannot rewind a fixed-size recurrent state.
        let proof = try makeHybridProof()
        XCTAssertThrowsError(
            try DenseContinuousBatchRuntime(
                model: TinyHybrid48LayerInitModel(),
                verifiedBy: proof,
                allocationChunk: 256,
                maxContextTokens: 4_096,
                soloPLDConfiguration: SpecDecodeConfig(
                    drafter: TinyHybridPromptEchoDrafter(),
                    maxDraft: 2,
                    lookback: 8,
                    compiledVerify: true))
        ) {
            XCTAssertEqual(
                $0 as? DenseContinuousBatchRuntimeError, .speculationUnsupported)
        }
    }

    func testHybridRejectsNonFP16CacheKindAtConstruction() throws {
        // Must fail as unsupportedCompressedBatchCache BEFORE the compressed-admission
        // machinery can demand an admission record for a cache kind hybrid will never run.
        let proof = try makeHybridProof()
        XCTAssertThrowsError(
            try DenseContinuousBatchRuntime(
                model: TinyHybrid48LayerInitModel(),
                verifiedBy: proof,
                allocationChunk: 256,
                maxContextTokens: 4_096,
                kvCacheKind: .affine(.k4v2G64))
        ) {
            XCTAssertEqual(
                $0 as? DenseContinuousBatchRuntimeError, .unsupportedCompressedBatchCache)
        }
        XCTAssertThrowsError(
            try DenseContinuousBatchRuntime(
                model: TinyHybrid48LayerInitModel(),
                verifiedBy: proof,
                allocationChunk: 256,
                maxContextTokens: 4_096,
                kvCacheKind: .turboQuant(.tqB3))
        ) {
            XCTAssertEqual(
                $0 as? DenseContinuousBatchRuntimeError, .unsupportedCompressedBatchCache)
        }
    }

    // MARK: hybrid multi-request parity (T0 gate assertions b + c)

    /// Two requests with DIFFERENT prompts and lengths (defeats row-swap blindness: their token
    /// sums and positions differ) whose per-request decode streams are precomputed closed-form
    /// (NOT by running a second in-process path, which could share the same bug). At each decode
    /// step a request emits `sessionTokenSum + 3·position + 1`; the position clock lives only in
    /// that request's recurrent MambaCache state.
    ///   A = prompt [1,2,3] → [16, 35, 73, 149]
    ///   B = prompt [5,6]   → [18, 39, 81, 165]
    private func makeTinyHybridRuntime() throws -> DenseContinuousBatchRuntime {
        let dir = try writeConfig(tinyHybridConfigJSON())
        defer { try? FileManager.default.removeItem(at: dir) }
        let proof = try DenseContinuousBatchModelProof.verifying(
            modelDirectory: dir, allowHybridQwen35: true)
        return try DenseContinuousBatchRuntime(
            model: TinyHybridLanguageModel(),
            verifiedBy: proof,
            allocationChunk: 4,
            maxContextTokens: 64)
    }

    private func prefillFinal(
        _ runtime: DenseContinuousBatchRuntime, id: UInt64, tokens: [Int]
    ) throws {
        try runtime.prefill(
            ContinuousBatchRuntimePrefill(
                id: BatchRequestID(id), startToken: 0, tokens: tokens,
                isFinal: true, totalPromptTokens: tokens.count, maxOutputTokens: 8))
    }

    /// (b) Batch-of-2 lockstep decode emits EXACTLY the tokens each request emits solo, and a
    /// mid-stream spill (batch → scalar, via `remove` of the cohort peer then `.solo`) round-trips
    /// the survivor's recurrent state so it keeps emitting its solo sequence. Without the
    /// driver-side batched-offset advance, the extract at spill validates the survivor's recurrent
    /// row against its now-larger `cachedTokens` and throws `cacheLengthMismatch` — the frozen-offset
    /// trap this test pins.
    func testHybridBatchOfTwoLockstepMatchesSoloThenSpills() throws {
        let runtime = try makeTinyHybridRuntime()
        try prefillFinal(runtime, id: 1, tokens: [1, 2, 3])
        try prefillFinal(runtime, id: 2, tokens: [5, 6])

        // Lockstep: results are returned in `ids` order (A first, B second).
        let step1 = try runtime.decode(
            .batch([BatchRequestID(1), BatchRequestID(2)], speculationAllowed: false))
        XCTAssertEqual(step1.map(\.tokens), [[16], [18]])
        let step2 = try runtime.decode(
            .batch([BatchRequestID(1), BatchRequestID(2)], speculationAllowed: false))
        XCTAssertEqual(step2.map(\.tokens), [[35], [39]])

        // Spill A to scalar: drop the cohort peer, then decode A solo. A's recurrent state (position
        // clock) must survive the batch→scalar extract bit-identically, so A continues its solo run.
        runtime.remove(BatchRequestID(2))
        let step3 = try runtime.decode(.solo(BatchRequestID(1), speculationAllowed: false))
        XCTAssertEqual(step3.map(\.tokens), [[73]])
        let step4 = try runtime.decode(.solo(BatchRequestID(1), speculationAllowed: false))
        XCTAssertEqual(step4.map(\.tokens), [[149]])
    }

    /// (c) Mid-stream rejoin (scalar → batch) round-trips recurrent state. Each request first runs
    /// solo (which advances `cachedTokens`), drains its lookahead pipeline (the mandatory
    /// solo→batch transition step, itself a decode that emits the next token), then rejoins a batch.
    /// Without the driver-side scalar-offset advance in `decodeScalar`, the rejoin validates the
    /// scalar recurrent row against its advanced `cachedTokens` and throws `cacheLengthMismatch` —
    /// the scalar-side half of the frozen-offset trap. Token equality against the solo oracle proves
    /// the round-tripped state is the live state the batched step actually consumes.
    func testHybridSoloThenDrainThenRejoinBatchRoundTrips() throws {
        let runtime = try makeTinyHybridRuntime()
        try prefillFinal(runtime, id: 1, tokens: [1, 2, 3])
        try prefillFinal(runtime, id: 2, tokens: [5, 6])

        XCTAssertEqual(
            try runtime.decode(.solo(BatchRequestID(1), speculationAllowed: false)).map(\.tokens),
            [[16]])
        XCTAssertEqual(
            try runtime.decode(.solo(BatchRequestID(2), speculationAllowed: false)).map(\.tokens),
            [[18]])
        XCTAssertEqual(
            try runtime.decode(.drainSoloPipeline(BatchRequestID(1))).map(\.tokens), [[35]])
        XCTAssertEqual(
            try runtime.decode(.drainSoloPipeline(BatchRequestID(2))).map(\.tokens), [[39]])

        let rejoin = try runtime.decode(
            .batch([BatchRequestID(1), BatchRequestID(2)], speculationAllowed: false))
        XCTAssertEqual(rejoin.map(\.tokens), [[73], [81]])
        let after = try runtime.decode(
            .batch([BatchRequestID(1), BatchRequestID(2)], speculationAllowed: false))
        XCTAssertEqual(after.map(\.tokens), [[149], [165]])
    }
}

/// Minimal drafter for construction-guard tests only; never consulted because the hybrid
/// runtime must reject speculation before any drafting can happen.
private struct TinyHybridPromptEchoDrafter: SpecDrafter {
    func propose(context: [Int], maxDraft: Int) -> [Int] {
        guard let last = context.last, maxDraft > 0 else { return [] }
        return [last]
    }
}
