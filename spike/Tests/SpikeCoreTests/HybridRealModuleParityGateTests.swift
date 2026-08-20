import HarnessCore
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXRandom
import XCTest

@testable import SpikeCore

// T0.5 of the hybrid continuous-batching build: the REAL-MODULE parity gate.
//
// The T0 gate (HybridContinuousBatchRuntimeTests.swift) proves the runtime seam against
// `TinyHybridLanguageModel` — a toy recurrence. That toy deliberately does NOT call
// `MambaCache.advance(_:)`; it defers all offset bookkeeping to the driver. The vendored
// `Qwen35GatedDeltaNet` (Qwen35.swift:290-291) DOES call `cache.advance(S)` inside its forward,
// so the "advance touches only transient lengths, never offset — the driver owns offset" claim
// that BatchedRecurrentStateCache depends on (BatchedRecurrentStateCache.swift:23-25) is UNTESTED
// against the real code path. This file converts that from an M5-handoff discovery into a
// seconds-fast on-box gate by driving the real `MLXLLM.Qwen35TextModel` at tiny dims.
//
// Measurement discipline for a RANDOM-WEIGHT fixture (differs from the M5 real-checkpoint gate):
// random-init tiny logits are near-uniform, so batch-vs-solo argmax flips are float-order noise,
// not seam bugs. The correctness signal here is therefore max logit-DELTA magnitude (bounded near
// float-epsilon → seam correct), NOT exact token equality. Exact token parity as a SHELVE trigger
// belongs to the real-checkpoint M5 gate where trained weights give sharp logits.
final class HybridRealModuleParityGateTests: XCTestCase {

    // MARK: fixtures

    /// Tiny but real-model-viable qwen3_5 text geometry. 2 layers, interval 2 → layer 0 recurrent
    /// (GatedDeltaNet), layer 1 dense (attention): the first-layer-recurrent property the runtime's
    /// kind-aware path must honor (real interval-4 checkpoints share it). Dims are the smallest that
    /// keep the real GatedDeltaNet + attention kernels well-formed. `linear_key_head_dim` MUST be a
    /// multiple of 32: the vendored gated-delta Metal kernel dispatches 32 threads over Dk and sizes
    /// its per-thread state array as `n_per_t = Dk / 32`, so Dk < 32 compiles a zero-length array and
    /// fails at kernel-build time (GatedDelta.swift:29,50). fp32 for deterministic parity.
    private func tinyRealTextConfigJSON() -> String {
        #"""
        {"model_type":"qwen3_5_text","hidden_size":16,"num_hidden_layers":2,
         "intermediate_size":32,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,
         "linear_num_value_heads":2,"linear_num_key_heads":1,"linear_key_head_dim":32,
         "linear_value_head_dim":32,"linear_conv_kernel_dim":4,"vocab_size":32,
         "full_attention_interval":2,"max_position_embeddings":64,"torch_dtype":"float32",
         "num_experts":0,"num_experts_per_tok":0}
        """#
    }

    private func makeTinyRealModel() throws -> Qwen35TextModel {
        let cfg = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(tinyRealTextConfigJSON().utf8))
        return Qwen35TextModel(cfg)
    }

    // MARK: Incr 1 — real module instantiates and forwards at tiny dims

    /// The real vendored hybrid model must instantiate from a synthetic tiny config and run a
    /// forward over its own `newCache()` on THIS box (no checkpoint, no M5). Layer 0's cache is a
    /// real `MambaCache`; after the forward its conv + SSM slots are populated — the recurrent path
    /// actually ran (a failed downcast or a stateless fallback leaves them nil). This is the on-box
    /// existence proof the toy gate cannot give.
    func testRealQwen35ModuleInstantiatesAndForwardsAtTinyDims() throws {
        MLXRandom.seed(0)
        let model = try makeTinyRealModel()

        let caches = model.newCache(parameters: nil)
        XCTAssertEqual(caches.count, 2)
        XCTAssertTrue(caches[0] is MambaCache, "layer 0 (interval 2) must be recurrent")
        XCTAssertTrue(caches[1] is KVCacheSimple, "layer 1 (interval 2) must be dense")

        let tokens = MLXArray([Int32(1), 2, 3]).reshaped([1, 3])
        let logits = model(tokens, cache: caches)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 3, 32])

        // Statefulness: the recurrent layer wrote back both slots through the real MambaCache.
        let mamba = try XCTUnwrap(caches[0] as? MambaCache)
        let conv = try XCTUnwrap(mamba[0], "conv tail must be populated after a recurrent forward")
        let ssm = try XCTUnwrap(mamba[1], "SSM state must be populated after a recurrent forward")
        XCTAssertEqual(conv.dim(0), 1)
        XCTAssertEqual(ssm.dim(0), 1)
    }

    // MARK: Incr 2 — real module through the continuous-batch runtime (structural)

    /// VL-wrapped form of the tiny text config, the shape the hybrid proof consumes (family under
    /// `text_config`). The proof's recurrent geometry is derived from these fields and MUST match the
    /// model instantiated from the same text config — true by construction here.
    private func tinyRealWrappedConfigJSON() -> String {
        #"""
        {"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"],
         "text_config":
        """#
            + tinyRealTextConfigJSON() + "}"
    }

    private func writeConfig(_ json: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(json.utf8).write(to: dir.appendingPathComponent("config.json"))
        return dir
    }

    private func makeRealRuntime() throws -> DenseContinuousBatchRuntime {
        let dir = try writeConfig(tinyRealWrappedConfigJSON())
        defer { try? FileManager.default.removeItem(at: dir) }
        let proof = try DenseContinuousBatchModelProof.verifying(
            modelDirectory: dir, allowHybridQwen35: true)
        MLXRandom.seed(0)
        let model = try makeTinyRealModel()
        return try DenseContinuousBatchRuntime(
            model: model, verifiedBy: proof, allocationChunk: 4, maxContextTokens: 64)
    }

    private func prefillFinal(
        _ runtime: DenseContinuousBatchRuntime, id: UInt64, tokens: [Int]
    ) throws {
        try runtime.prefill(
            ContinuousBatchRuntimePrefill(
                id: BatchRequestID(id), startToken: 0, tokens: tokens,
                isFinal: true, totalPromptTokens: tokens.count, maxOutputTokens: 8))
    }

    /// The real GatedDeltaNet's `cache.advance(S)` (Qwen35.swift:290-291) must NOT corrupt the
    /// driver-maintained recurrent offset: the runtime relies on `advance` touching only transient
    /// lengths, never `BaseKVCache.offset`. Driving the real module through prefill + solo decode
    /// proves the kind-aware runtime feeds the real recurrent seam and keeps decoding across steps.
    /// (Token VALUES are random-weight-dependent and not asserted; token COUNT and no-throw are the
    /// structural, noise-immune signal.)
    func testRealModuleSoloDecodeThroughRuntime() throws {
        let runtime = try makeRealRuntime()
        try prefillFinal(runtime, id: 1, tokens: [1, 2, 3])

        for _ in 0 ..< 3 {
            let r = try runtime.decode(.solo(BatchRequestID(1), speculationAllowed: false))
            XCTAssertEqual(r.count, 1)
            XCTAssertEqual(r[0].tokens.count, 1, "each solo step emits exactly one token")
        }
    }

    /// Batch→scalar spill and scalar→batch rejoin must round-trip the real module's recurrent state
    /// WITHOUT offset drift. A frozen or double-counted offset throws `cacheLengthMismatch` at the
    /// `extract`/`merging` boundary (BatchedRecurrentStateCache validates each row's `offset` against
    /// its committed length) — so a clean round-trip that keeps emitting is the offset-consistency
    /// proof for the real `advance(S)` path the toy T0 gate could not exercise.
    func testRealModuleBatchSpillAndRejoinRoundTrip() throws {
        // (a) spill: batch of two, then drop the peer and continue the survivor solo.
        let spillRuntime = try makeRealRuntime()
        try prefillFinal(spillRuntime, id: 1, tokens: [1, 2, 3])
        try prefillFinal(spillRuntime, id: 2, tokens: [5, 6])
        let b1 = try spillRuntime.decode(
            .batch([BatchRequestID(1), BatchRequestID(2)], speculationAllowed: false))
        XCTAssertEqual(b1.map { $0.tokens.count }, [1, 1])
        _ = try spillRuntime.decode(
            .batch([BatchRequestID(1), BatchRequestID(2)], speculationAllowed: false))
        spillRuntime.remove(BatchRequestID(2))
        let survivor = try spillRuntime.decode(.solo(BatchRequestID(1), speculationAllowed: false))
        XCTAssertEqual(survivor[0].tokens.count, 1, "spill survivor keeps decoding — offset round-tripped")

        // (b) rejoin: each runs solo, drains its lookahead pipeline, then rejoins a batch.
        let rejoinRuntime = try makeRealRuntime()
        try prefillFinal(rejoinRuntime, id: 1, tokens: [1, 2, 3])
        try prefillFinal(rejoinRuntime, id: 2, tokens: [5, 6])
        _ = try rejoinRuntime.decode(.solo(BatchRequestID(1), speculationAllowed: false))
        _ = try rejoinRuntime.decode(.solo(BatchRequestID(2), speculationAllowed: false))
        _ = try rejoinRuntime.decode(.drainSoloPipeline(BatchRequestID(1)))
        _ = try rejoinRuntime.decode(.drainSoloPipeline(BatchRequestID(2)))
        let rejoined = try rejoinRuntime.decode(
            .batch([BatchRequestID(1), BatchRequestID(2)], speculationAllowed: false))
        XCTAssertEqual(rejoined.map { $0.tokens.count }, [1, 1], "rejoin merged scalar rows — offsets consistent")
    }

    // MARK: Incr 3 — numeric batch correctness (runtime batch-vs-solo token parity)

    private func decodeTokens(
        _ runtime: DenseContinuousBatchRuntime, _ action: BatchDecodeAction, resultIndex: Int
    ) throws -> [Int] {
        try runtime.decode(action)[resultIndex].tokens
    }

    /// The numeric correctness signal for hybrid continuous batching on the REAL module: a cohort
    /// decoded together must emit the SAME tokens as each request decoded solo — for EVERY cohort
    /// slot, not just slot 0. Two runtimes built from the same seed hold identical weights, so exact
    /// token parity is well-defined.
    ///
    /// The valid oracle here is the RUNTIME, not a hand-rolled model-level batch. Batched decode of
    /// the raw vendored `Qwen35TextModel` requires the per-row cache mask / offset preparation the
    /// runtime performs (leftPadding / lengths): without it, raw B>1 single-token decode is wrong for
    /// every row beyond slot 0 — content-independent, in BOTH the dense-attention and GatedDeltaNet
    /// layers. That is a property of naive batching, not a runtime defect; this test pins that the
    /// runtime path is correct. Full investigation:
    /// docs/task-inbox/2026-08-20-real-module-batched-decode-parity.md.
    func testRealModuleBatchedDecodeMatchesSoloTokens() throws {
        let steps = 4

        // Cohort of two different-length requests. req2 occupies slot 1 — the row a naive
        // model-level batch decodes incorrectly, and therefore the row that most needs pinning.
        let batched = try makeRealRuntime()
        try prefillFinal(batched, id: 1, tokens: [1, 2, 3])
        try prefillFinal(batched, id: 2, tokens: [5, 6])
        var batchReq1: [Int] = []
        var batchReq2: [Int] = []
        for _ in 0 ..< steps {
            let d = try batched.decode(
                .batch([BatchRequestID(1), BatchRequestID(2)], speculationAllowed: false))
            batchReq1 += d[0].tokens
            batchReq2 += d[1].tokens
        }

        // Solo oracles (same seed → identical weights via makeRealRuntime).
        let solo1 = try makeRealRuntime()
        try prefillFinal(solo1, id: 1, tokens: [1, 2, 3])
        var soloReq1: [Int] = []
        for _ in 0 ..< steps {
            soloReq1 += try decodeTokens(
                solo1, .solo(BatchRequestID(1), speculationAllowed: false), resultIndex: 0)
        }

        let solo2 = try makeRealRuntime()
        try prefillFinal(solo2, id: 2, tokens: [5, 6])
        var soloReq2: [Int] = []
        for _ in 0 ..< steps {
            soloReq2 += try decodeTokens(
                solo2, .solo(BatchRequestID(2), speculationAllowed: false), resultIndex: 0)
        }

        XCTAssertEqual(batchReq1, soloReq1, "slot-0 request: batched decode must match solo tokens")
        XCTAssertEqual(
            batchReq2, soloReq2,
            "slot-1 request: batched decode must match solo tokens (the row naive model-level batching gets wrong)")
    }
}
