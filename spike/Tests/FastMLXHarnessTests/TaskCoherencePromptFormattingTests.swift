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
}
