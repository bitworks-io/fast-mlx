import XCTest
@testable import HarnessCore

final class TemplateTokenCacheTests: XCTestCase {
    private func digest(_ character: Character) -> String {
        String(repeating: String(character), count: 64)
    }

    private func key(
        namespace: Character = "a",
        tokenizer: Character = "b",
        template: Character = "c",
        tools: Character = "d",
        content: Character = "e",
        options: Character = "f"
    ) throws -> TemplateTokenCacheKey {
        try TemplateTokenCacheKey(
            isolationNamespaceSHA256: digest(namespace),
            tokenizerSHA256: digest(tokenizer),
            promptTemplateSHA256: digest(template),
            toolsSHA256: digest(tools),
            promptContentSHA256: digest(content),
            formattingOptionsSHA256: digest(options))
    }

    func testKeyBindsEveryHostRenderingAxisAndNamespace() throws {
        let baseline = try key()
        let variants = [
            try key(namespace: "0"),
            try key(tokenizer: "0"),
            try key(template: "0"),
            try key(tools: "0"),
            try key(content: "0"),
            try key(options: "0"),
        ]
        XCTAssertEqual(Set(variants).count, variants.count)
        XCTAssertTrue(variants.allSatisfy { $0 != baseline })

        let encoded = try JSONEncoder().encode(baseline)
        var decodedObject = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoded)
                as? [String: Any])
        decodedObject["tokenizerSHA256"] = "not-a-digest"
        let invalidEncoded = try JSONSerialization.data(
            withJSONObject: decodedObject)
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                TemplateTokenCacheKey.self,
                from: invalidEncoded))
    }

    func testExactLookupAndLRUEvictionAreIndependentFromModelPrefixCache() throws {
        var cache = TemplateTokenCache(
            policy: try TemplateTokenCachePolicy(
                maxEntries: 2,
                maxRetainedBytes: 16 * 1024))
        let one = try key(content: "1")
        let two = try key(content: "2")
        let three = try key(content: "3")

        XCTAssertEqual(
            try cache.insert(key: one, tokenIDs: [1, 2]).inserted,
            true)
        XCTAssertEqual(
            try cache.insert(key: two, tokenIDs: [3, 4]).inserted,
            true)
        XCTAssertEqual(try cache.lookup(key: one), [1, 2])

        let third = try cache.insert(key: three, tokenIDs: [5, 6])
        XCTAssertTrue(third.inserted)
        XCTAssertEqual(third.evictedEntryCount, 1)
        XCTAssertEqual(try cache.lookup(key: one), [1, 2])
        XCTAssertNil(try cache.lookup(key: two))
        XCTAssertEqual(try cache.lookup(key: three), [5, 6])
    }

    func testReplacementAndByteBudgetRemainBounded() throws {
        let cacheKey = try key()
        let oneEntryBytes = try TemplateTokenCache.retainedBytes(
            key: cacheKey,
            tokenIDs: [1, 2, 3])
        var cache = TemplateTokenCache(
            policy: try TemplateTokenCachePolicy(
                maxEntries: 2,
                maxRetainedBytes: oneEntryBytes))

        XCTAssertTrue(
            try cache.insert(
                key: cacheKey,
                tokenIDs: [1, 2, 3]
            ).inserted)
        XCTAssertTrue(
            try cache.insert(
                key: cacheKey,
                tokenIDs: [4, 5, 6]
            ).inserted)
        XCTAssertEqual(cache.snapshot.entryCount, 1)
        XCTAssertLessThanOrEqual(
            cache.snapshot.retainedBytes,
            oneEntryBytes)
        XCTAssertEqual(try cache.lookup(key: cacheKey), [4, 5, 6])

        let skipped = try cache.insert(
            key: try key(content: "9"),
            tokenIDs: Array(repeating: 7, count: 10_000))
        XCTAssertFalse(skipped.inserted)
        XCTAssertEqual(skipped.skipReason, .entryExceedsBudget)
        XCTAssertEqual(cache.snapshot.entryCount, 1)
    }

    func testDisabledEmptyAndOutOfRangeInputsFailClosed() throws {
        var disabled = TemplateTokenCache(policy: .disabled)
        let decision = try disabled.insert(
            key: key(),
            tokenIDs: [1, 2])
        XCTAssertFalse(decision.inserted)
        XCTAssertEqual(decision.skipReason, .disabled)

        var enabled = TemplateTokenCache(
            policy: try TemplateTokenCachePolicy(
                maxEntries: 2,
                maxRetainedBytes: 4096))
        XCTAssertThrowsError(
            try enabled.insert(key: key(), tokenIDs: []))
        XCTAssertThrowsError(
            try enabled.insert(key: key(), tokenIDs: [Int.max]))
        XCTAssertThrowsError(
            try enabled.insert(key: key(), tokenIDs: [-1])
        ) { error in
            XCTAssertEqual(
                error as? TemplateTokenCacheError,
                .negativeTokenID(position: 0))
        }

        let invalidPolicy = Data(
            """
            {"isEnabled":true,"maxEntries":0,"maxRetainedBytes":1}
            """.utf8)
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                TemplateTokenCachePolicy.self,
                from: invalidPolicy))
    }
}
