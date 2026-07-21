import Foundation
import HarnessCore
import MLXLMCommon
import XCTest

@testable import fastmlx_harness

private enum PromptTokenizerError: Error {
    case invalidChatRequest
}

private struct PromptTokenizer: MLXLMCommon.Tokenizer {
    let bosToken: String? = nil
    let eosToken: String? = "<eos>"
    let unknownToken: String? = nil

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        (addSpecialTokens ? [7] : []) + text.utf8.map(Int.init)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        ""
    }

    func convertTokenToId(_ token: String) -> Int? {
        token == "<eos>" ? 2 : nil
    }

    func convertIdToToken(_ id: Int) -> String? {
        id == 2 ? "<eos>" : nil
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        guard tools == nil, messages.count == 1,
            messages[0]["role"] as? String == "user",
            let content = messages[0]["content"] as? String,
            additionalContext?["enable_thinking"] as? Bool == false
        else { throw PromptTokenizerError.invalidChatRequest }
        return [9_001]
            + encode(text: content, addSpecialTokens: false)
            + [9_002]
    }
}

final class TaskCoherencePromptFormattingTests: XCTestCase {
    func testKVTunerImplicitAttentionConfigurationBindsMaterializeAndExactCheckpoint() throws {
        let checkpoint = String(repeating: "d", count: 64)
        let implicit = try effectiveCompressedKVAttentionConfiguration(
            explicitRequest: nil,
            explicitCheckpointContentSHA256: nil,
            authenticatedKVTunerCheckpointContentSHA256: checkpoint)
        XCTAssertEqual(implicit.request, .materialize)
        XCTAssertEqual(implicit.checkpointContentSHA256, checkpoint)

        let explicit = try effectiveCompressedKVAttentionConfiguration(
            explicitRequest: .splitAffineQuantizedMM,
            explicitCheckpointContentSHA256: checkpoint,
            authenticatedKVTunerCheckpointContentSHA256: checkpoint)
        XCTAssertEqual(explicit.request, .splitAffineQuantizedMM)

        XCTAssertThrowsError(try
            effectiveCompressedKVAttentionConfiguration(
                explicitRequest: .materialize,
                explicitCheckpointContentSHA256:
                    String(repeating: "e", count: 64),
                authenticatedKVTunerCheckpointContentSHA256: checkpoint)
        ) { error in
            XCTAssertEqual(
                error as? KVTunerCLIError,
                .checkpointContentIdentityMismatch)
        }

        XCTAssertEqual(
            try effectiveCompressedKVAttentionConfiguration(
                explicitRequest: nil,
                explicitCheckpointContentSHA256: nil,
                authenticatedKVTunerCheckpointContentSHA256: nil),
            EffectiveCompressedKVAttentionConfiguration(
                request: nil,
                checkpointContentSHA256: nil))
    }

    func testRestrictedChoiceStaysRawWhileStructuredToolUsesCheckpointChatTemplate() throws {
        let corpus = try TaskCoherenceCorpusV2.make()
        let restricted = try XCTUnwrap(
            corpus.items.first { $0.scoringMode == .restrictedChoice })
        let structured = try XCTUnwrap(
            corpus.items.first { $0.scoringMode == .structuredTool })
        let tokenizer = PromptTokenizer()
        let configuration = TaskCoherenceRunConfiguration.qualificationV3(
            structuredToolMaxTokens: 96)

        let restrictedSegments = try taskCoherencePromptTokenSegments(
            item: restricted, runConfiguration: configuration,
            tokenizer: tokenizer)
        XCTAssertEqual(
            restrictedSegments.prompt,
            tokenizer.encode(
                text: restricted.prompt, addSpecialTokens: true))
        XCTAssertEqual(
            restrictedSegments.prefix,
            tokenizer.encode(
                text: restricted.prefix, addSpecialTokens: true))

        let structuredSegments = try taskCoherencePromptTokenSegments(
            item: structured, runConfiguration: configuration,
            tokenizer: tokenizer)
        XCTAssertEqual(
            structuredSegments.prompt,
            [9_001]
                + tokenizer.encode(
                    text: structured.prompt, addSpecialTokens: false)
                + [9_002])
        XCTAssertEqual(
            structuredSegments.prefixAndMaterial,
            [9_001]
                + tokenizer.encode(
                    text: structured.prefix + structured.material,
                    addSpecialTokens: false)
                + [9_002])
        XCTAssertEqual(
            structuredSegments.suffixAndQuery,
            [9_001]
                + tokenizer.encode(
                    text: structured.suffix + structured.query,
                    addSpecialTokens: false)
                + [9_002])

        let layout = try TaskCoherencePromptLayoutEvidence.derive(
            prefixTokenIDs: structuredSegments.prefix,
            prefixAndMaterialTokenIDs:
                structuredSegments.prefixAndMaterial,
            suffixAndQueryTokenIDs:
                structuredSegments.suffixAndQuery,
            promptTokenIDs: structuredSegments.prompt)
        XCTAssertEqual(
            layout.promptTokens, structuredSegments.prompt.count)
        XCTAssertGreaterThan(
            layout.materialEndToken, layout.materialStartToken)
        XCTAssertGreaterThan(
            layout.minimumCompletedTileCount, 0)
    }

    func testTaskCompressedAttentionBindingUsesPostForwardGenerationAndScoringReceipts() throws {
        let admission = try compressedAttentionAdmission()
        let split = EngagementCounters([
            "affine_attention_split": 1,
            "affine_attention_materialized": 0,
        ])
        let scoringSplit = EngagementCounters([
            "scoring_attention_split": 1,
            "scoring_attention_materialized": 0,
        ])

        let binding = try taskCompressedKVAttentionBinding(
            tier: "affine-k4v2-g64",
            request: .splitAffineQuantizedMM,
            admission: admission,
            generated: split,
            scoring: scoringSplit)

        XCTAssertEqual(binding?.request, .splitAffineQuantizedMM)
        XCTAssertEqual(binding?.observedOperation, .splitQuantizedMM)
        XCTAssertEqual(binding?.admission, admission)
    }

    func testTaskCompressedAttentionBindingAuthenticatesDirectKVarNGenerationAndScoring() throws {
        let admission = try compressedAttentionAdmission()
        let generated = EngagementCounters([
            "kvarn_attention_split": 1,
            "kvarn_attention_materialized": 0,
        ])
        let scoring = EngagementCounters([
            "scoring_kvarn_attention_split": 1,
            "scoring_kvarn_attention_materialized": 0,
        ])

        let binding = try taskCompressedKVAttentionBinding(
            tier: "kvarn-k4v2-g128",
            request: .splitKVarNQuantizedMM,
            admission: admission,
            generated: generated,
            scoring: scoring)

        XCTAssertEqual(binding?.request, .splitKVarNQuantizedMM)
        XCTAssertEqual(
            binding?.observedOperation, .splitKVarNQuantizedMM)
        XCTAssertEqual(binding?.admission, admission)
    }

    func testTaskEngagementRecordsCompiledGenerationAndUncompiledDirectScoring() throws {
        let generated = EngagementCounters([
            "kvarn_tokens": 513,
            "kvarn_completed_tiles": 3,
            "kvarn_compressed_tokens": 384,
            "kvarn_codec_iterations": 8,
            "kvarn_compiled": 1,
            "kvarn_uncompiled_correctness": 0,
        ])
        let scoring = EngagementCounters([
            "scoring_cached_tokens": 512,
            "scoring_kvarn_completed_tiles": 3,
            "scoring_kvarn_compressed_tokens": 384,
            "scoring_kvarn_attention_split": 1,
            "scoring_kvarn_attention_materialized": 0,
        ])

        let evidence = try taskEngagement(
            tier: "kvarn-k4v2-g128",
            kvtunerSchedule: nil,
            compressedKVAttentionRequest: .splitKVarNQuantizedMM,
            generated: generated,
            scoring: scoring)

        XCTAssertEqual(evidence.cachedTokens, 513)
        XCTAssertEqual(evidence.kvarnCompletedTileCount, 3)
        XCTAssertEqual(evidence.kvarnCompressedTokens, 384)
        XCTAssertEqual(evidence.kvarnExecutionMode, "compiled")
        XCTAssertEqual(evidence.scoringCachedTokens, 512)
        XCTAssertEqual(evidence.scoringKVarNCompletedTileCount, 3)
        XCTAssertEqual(evidence.scoringKVarNCompressedTokens, 384)
    }

    func testTaskCompressedAttentionBindingRejectsTelemetryOrScoringDisagreement() throws {
        let admission = try compressedAttentionAdmission()
        let split = EngagementCounters([
            "affine_attention_split": 1,
            "affine_attention_materialized": 0,
        ])
        let materialized = EngagementCounters([
            "affine_attention_split": 0,
            "affine_attention_materialized": 1,
        ])
        let scoringMaterialized = EngagementCounters([
            "scoring_attention_split": 0,
            "scoring_attention_materialized": 1,
        ])

        XCTAssertThrowsError(try taskCompressedKVAttentionBinding(
            tier: "affine-k4v2-g64",
            request: .splitAffineQuantizedMM,
            admission: admission,
            generated: materialized,
            scoring: nil))
        XCTAssertThrowsError(try taskCompressedKVAttentionBinding(
            tier: "affine-k4v2-g64",
            request: .splitAffineQuantizedMM,
            admission: admission,
            generated: split,
            scoring: scoringMaterialized))
        XCTAssertThrowsError(try taskCompressedKVAttentionBinding(
            tier: "affine-k4v2-g64",
            request: .splitAffineQuantizedMM,
            admission: nil,
            generated: split,
            scoring: nil))
    }

    func testTaskCompressedAttentionBindingKeepsHistoricalDefaultAbsent() throws {
        XCTAssertNil(try taskCompressedKVAttentionBinding(
            tier: "fp16",
            request: nil,
            admission: nil,
            generated: EngagementCounters(),
            scoring: nil))
    }

    private func compressedAttentionAdmission() throws
        -> CompressedKVAttentionRuntimeAdmission
    {
        let config = Data(
            #"{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"hidden_size":5120,"num_hidden_layers":64,"num_attention_heads":64,"num_key_value_heads":8,"head_dim":128,"max_position_embeddings":40960,"use_sliding_window":false}"#.utf8)
        return try CompressedKVAttentionRuntimeAdmission.load(
            sourceSnapshot: .load(
                exactModelConfigData: config,
                checkpointManifestHash: "0123456789abcdef",
                checkpointContentSHA256:
                    String(repeating: "d", count: 64),
                tokenizerSHA256: String(repeating: "a", count: 64)))
    }
}
