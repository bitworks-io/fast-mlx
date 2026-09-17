import Foundation
import XCTest

import MLXLMCommon
import ServingCore
import SpikeCore
@testable import SpikeServingAdapters

/// A minimal `MLXLMCommon.Tokenizer` fixture: `decode(tokenIds:)` simply looks up the vocab string
/// for a single id, which is correct for THIS fixture's vocab (every `.bytes`-classified entry is
/// plain printable ASCII, whose byte-level inversion is the identity — see
/// `ByteLevelTokenBytes`'s doc comment on the GPT-2 mapping's identity range), so the self-check
/// this fixture exercises is a REAL check, not one trivially vacuous by construction.
private struct FakeByteLevelTokenizer: MLXLMCommon.Tokenizer {
    let vocabStringsByID: [Int: String]
    let vocabIDsByString: [String: Int]
    var eosToken: String?
    var bosToken: String? { nil }
    var unknownToken: String? { nil }
    /// When non-nil, `convertTokenToId` returns THIS id for any string not literally present in
    /// `vocabIDsByString` — simulating a real BPE tokenizer's UNKNOWN-token fallback, which
    /// conventionally returns a fixed id rather than `nil` for an unresolvable string. `nil`
    /// (the default) keeps the plain "exact lookup or nil" behavior every OTHER test in this file
    /// relies on.
    var unknownFallbackID: Int?

    init(vocabStringsByID: [Int: String], eosToken: String?, unknownFallbackID: Int? = nil) {
        self.vocabStringsByID = vocabStringsByID
        self.eosToken = eosToken
        self.unknownFallbackID = unknownFallbackID
        var inverse: [String: Int] = [:]
        for (id, string) in vocabStringsByID { inverse[string] = id }
        self.vocabIDsByString = inverse
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.compactMap { vocabStringsByID[$0] }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? { vocabIDsByString[token] ?? unknownFallbackID }
    func convertIdToToken(_ id: Int) -> String? { vocabStringsByID[id] }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

/// Writes a minimal fixture model directory (`tokenizer.json` + optionally `config.json` /
/// `generation_config.json`) under a fresh temp directory, returning its URL. Each test controls
/// exactly which files exist / what they contain, mirroring
/// `loadScalarServingJSONObjectConstraintSupport`'s own disablement enumeration.
private func makeFixtureModelDirectory(
    tokenizerJSON: [String: Any]?,
    configJSON: [String: Any]? = ["vocab_size": 10],
    generationConfigJSON: [String: Any]? = nil
) throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("json-object-constraint-fixture-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    if let tokenizerJSON {
        let data = try JSONSerialization.data(withJSONObject: tokenizerJSON)
        try data.write(to: directory.appendingPathComponent("tokenizer.json"))
    }
    if let configJSON {
        let data = try JSONSerialization.data(withJSONObject: configJSON)
        try data.write(to: directory.appendingPathComponent("config.json"))
    }
    if let generationConfigJSON {
        let data = try JSONSerialization.data(withJSONObject: generationConfigJSON)
        try data.write(to: directory.appendingPathComponent("generation_config.json"))
    }
    return directory
}

private func byteLevelTokenizerJSON() -> [String: Any] {
    [
        "decoder": ["type": "ByteLevel"],
        "model": ["vocab": ["a": 0, "b": 1, "</s>": 2, "</think>": 3]],
        "added_tokens": [["id": 3, "content": "</think>", "special": true]],
    ]
}

final class ScalarServingJSONObjectConstraintSupportTests: XCTestCase {
    func testSucceedsForByteLevelTokenizerWithResolvableEOSAndVocabSize() async throws {
        let directory = try makeFixtureModelDirectory(tokenizerJSON: byteLevelTokenizerJSON())
        let tokenizer = FakeByteLevelTokenizer(
            vocabStringsByID: ["a": 0, "b": 1, "</s>": 2, "</think>": 3].reduce(into: [:]) {
                $0[$1.value] = $1.key
            },
            eosToken: "</s>")

        let support = loadScalarServingJSONObjectConstraintSupport(
            modelDirectory: directory, tokenizer: tokenizer)

        let unwrapped = try XCTUnwrap(support, "expected support for a byte-level tokenizer")
        XCTAssertEqual(unwrapped.thinkEndTokenID, 3)
        // The table build starts eagerly, off-actor, at `init` (see `table`'s doc comment); this
        // await must be REACHABLE without crashing/throwing/hanging.
        _ = await unwrapped.table
    }

    func testReturnsNilForNonByteLevelDecoder() throws {
        let directory = try makeFixtureModelDirectory(
            tokenizerJSON: [
                "decoder": ["type": "Metaspace"],
                "model": ["vocab": ["a": 0, "</s>": 1]],
            ])
        let tokenizer = FakeByteLevelTokenizer(
            vocabStringsByID: [0: "a", 1: "</s>"], eosToken: "</s>")

        XCTAssertNil(
            loadScalarServingJSONObjectConstraintSupport(modelDirectory: directory, tokenizer: tokenizer))
    }

    func testReturnsNilWhenTokenizerJSONIsMissing() throws {
        let directory = try makeFixtureModelDirectory(tokenizerJSON: nil)
        let tokenizer = FakeByteLevelTokenizer(vocabStringsByID: [0: "a"], eosToken: "</s>")

        XCTAssertNil(
            loadScalarServingJSONObjectConstraintSupport(modelDirectory: directory, tokenizer: tokenizer))
    }

    func testReturnsNilWhenModelVocabSizeIsUnresolved() throws {
        let directory = try makeFixtureModelDirectory(
            tokenizerJSON: byteLevelTokenizerJSON(), configJSON: nil)
        let tokenizer = FakeByteLevelTokenizer(
            vocabStringsByID: ["a": 0, "b": 1, "</s>": 2, "</think>": 3].reduce(into: [:]) {
                $0[$1.value] = $1.key
            },
            eosToken: "</s>")

        XCTAssertNil(
            loadScalarServingJSONObjectConstraintSupport(modelDirectory: directory, tokenizer: tokenizer))
    }

    /// Neither the tokenizer's own `eosToken` NOR `generation_config.json` resolves an EOS id: the
    /// automaton could never legally terminate, so the feature must be disabled rather than built
    /// against an empty EOS set.
    func testReturnsNilWhenNoEOSIsResolvable() throws {
        let directory = try makeFixtureModelDirectory(tokenizerJSON: byteLevelTokenizerJSON())
        let tokenizer = FakeByteLevelTokenizer(
            vocabStringsByID: ["a": 0, "b": 1, "</s>": 2, "</think>": 3].reduce(into: [:]) {
                $0[$1.value] = $1.key
            },
            eosToken: nil)

        XCTAssertNil(
            loadScalarServingJSONObjectConstraintSupport(modelDirectory: directory, tokenizer: tokenizer))
    }

    /// `generation_config.json`'s `eos_token_id` is enough on its own, even when the tokenizer's
    /// own `eosToken` resolves to nothing — the union contract (response-format design), not an
    /// AND of both sources.
    func testGenerationConfigEOSAloneIsSufficient() throws {
        let directory = try makeFixtureModelDirectory(
            tokenizerJSON: byteLevelTokenizerJSON(),
            generationConfigJSON: ["eos_token_id": 2])
        let tokenizer = FakeByteLevelTokenizer(
            vocabStringsByID: ["a": 0, "b": 1, "</s>": 2, "</think>": 3].reduce(into: [:]) {
                $0[$1.value] = $1.key
            },
            eosToken: nil)

        XCTAssertNotNil(
            loadScalarServingJSONObjectConstraintSupport(modelDirectory: directory, tokenizer: tokenizer))
    }

    // MARK: - Nested `text_config` vocab_size (VL-wrapped hybrid checkpoints)

    /// `config.json`'s `vocab_size` lives under `text_config` on a VL-wrapped hybrid checkpoint,
    /// with NO root-level `vocab_size` at all — `scalarServingModelVocabSize` must still resolve
    /// it, or this checkpoint class would always disable the feature with
    /// `model_vocab_size_unresolved` despite having a perfectly good byte-level tokenizer.
    func testResolvesVocabSizeFromNestedTextConfig() throws {
        let directory = try makeFixtureModelDirectory(
            tokenizerJSON: byteLevelTokenizerJSON(),
            configJSON: ["text_config": ["vocab_size": 10], "some_other_root_field": 1])
        let tokenizer = FakeByteLevelTokenizer(
            vocabStringsByID: ["a": 0, "b": 1, "</s>": 2, "</think>": 3].reduce(into: [:]) {
                $0[$1.value] = $1.key
            },
            eosToken: "</s>")

        XCTAssertNotNil(
            loadScalarServingJSONObjectConstraintSupport(modelDirectory: directory, tokenizer: tokenizer))
    }

    // MARK: - EOS out of range (should-fix #7)

    /// Every resolved EOS id is `>= vocab_size` (here: `</s>` is id 2, but `vocab_size` is 2, so
    /// valid ids are only `0` and `1`) — the automaton could never legally terminate, so this must
    /// disable identically to `no_eos_token_id`, not silently build a table that can never emit EOS.
    func testReturnsNilWhenEOSIdIsAtOrAboveVocabSize() throws {
        let directory = try makeFixtureModelDirectory(
            tokenizerJSON: byteLevelTokenizerJSON(), configJSON: ["vocab_size": 2])
        let tokenizer = FakeByteLevelTokenizer(
            vocabStringsByID: ["a": 0, "b": 1, "</s>": 2, "</think>": 3].reduce(into: [:]) {
                $0[$1.value] = $1.key
            },
            eosToken: "</s>")

        XCTAssertNil(
            loadScalarServingJSONObjectConstraintSupport(modelDirectory: directory, tokenizer: tokenizer))
    }

    // MARK: - `</think>` resolution (must-fix #3): never trust `convertTokenToId` alone

    private func nonThinkingByteLevelTokenizerJSON(addedThinkEndTokenID: Int? = nil) -> [String: Any] {
        var root: [String: Any] = [
            "decoder": ["type": "ByteLevel"],
            "model": ["vocab": ["a": 0, "b": 1, "</s>": 2]],
        ]
        if let addedThinkEndTokenID {
            root["added_tokens"] = [
                ["id": addedThinkEndTokenID, "content": "</think>", "special": true]
            ]
        }
        return root
    }

    /// A BPE tokenizer's `convertTokenToId` conventionally returns its UNKNOWN-token id (never
    /// `nil`) for a string it cannot resolve. Here `convertTokenToId("</think>")` falls back to id
    /// `0` ("a"), which `convertIdToToken` round-trips back to `"a"`, NOT `"</think>"` — the
    /// mismatch `resolveThinkEndTokenID`'s round-trip check exists to catch. Must resolve to `nil`,
    /// not silently phase-switch on the wrong id.
    func testThinkEndTokenIsNilWhenConvertTokenToIdReturnsUnknownIDFallback() throws {
        let directory = try makeFixtureModelDirectory(
            tokenizerJSON: nonThinkingByteLevelTokenizerJSON())
        let tokenizer = FakeByteLevelTokenizer(
            vocabStringsByID: [0: "a", 1: "b", 2: "</s>"],
            eosToken: "</s>",
            unknownFallbackID: 0)

        let support = loadScalarServingJSONObjectConstraintSupport(
            modelDirectory: directory, tokenizer: tokenizer)

        XCTAssertNil(try XCTUnwrap(support, "expected support to still build").thinkEndTokenID)
    }

    /// `convertTokenToId("</think>")` returns `nil` outright (this fake tokenizer has no unknown
    /// fallback and no such entry), but `tokenizer.json`'s own `added_tokens` table has an entry
    /// whose `content` is `"</think>"` — `resolveThinkEndTokenID` must resolve it from THAT source
    /// instead of giving up.
    func testThinkEndTokenIsResolvedViaAddedTokensWhenConvertTokenToIdCannotResolveIt() throws {
        let directory = try makeFixtureModelDirectory(
            tokenizerJSON: nonThinkingByteLevelTokenizerJSON(addedThinkEndTokenID: 3))
        let tokenizer = FakeByteLevelTokenizer(
            vocabStringsByID: [0: "a", 1: "b", 2: "</s>"],
            eosToken: "</s>")

        let support = loadScalarServingJSONObjectConstraintSupport(
            modelDirectory: directory, tokenizer: tokenizer)

        XCTAssertEqual(try XCTUnwrap(support, "expected support to still build").thinkEndTokenID, 3)
    }
}

// MARK: - ScalarServingBackend capability composition

/// A codec that is never actually invoked by these tests (they read
/// `ScalarServingBackend.supportsJSONObjectResponseFormat` directly without ever calling `start`),
/// so every method can be an unreachable trap — reaching one would itself be a test bug.
private struct UnusedScalarServingTextCodec: ScalarServingTextCodec {
    func render(
        messages: [OpenAIChatMessage], tools: [OpenAIToolSpec], enableThinking: Bool?,
        reasoningEffort: String?
    ) throws -> [Int] {
        XCTFail("render should not be called by capability-composition tests")
        return []
    }

    func makeDetokenizer() -> any ScalarServingDetokenizer { UnusedScalarServingDetokenizer() }
}

private struct UnusedScalarServingDetokenizer: ScalarServingDetokenizer {
    mutating func append(token: Int) { XCTFail("should not be called") }
    mutating func next() -> String? {
        XCTFail("should not be called")
        return nil
    }
}

private func makeCapabilityTestBackend(
    jsonObjectConstraintSupport: ScalarServingJSONObjectConstraintSupport?,
    isNonSpeculativeScalarRoute: Bool,
    decoderSupportsResponseFormatConstraint: Bool
) -> ScalarServingBackend {
    ScalarServingBackend(
        launchedModel: "fixture-model",
        inference: InferenceActor(decoder: ScriptedDecoder(script: [99], eos: 99)),
        codec: UnusedScalarServingTextCodec(),
        stopTokenIDs: [99],
        modelStopStrings: [],
        configuration: .init(
            defaultMaximumCompletionTokens: 8,
            maximumQueuedRequests: 2,
            queueRetryAfterSeconds: 2,
            mailboxCapacity: .init(maxDeltas: 4, maxBytes: 1_024),
            jsonObjectConstraintSupport: jsonObjectConstraintSupport,
            isNonSpeculativeScalarRoute: isNonSpeculativeScalarRoute,
            decoderSupportsResponseFormatConstraint: decoderSupportsResponseFormatConstraint))
}

final class ScalarServingBackendJSONObjectCapabilityTests: XCTestCase {
    private func fakeSupport() -> ScalarServingJSONObjectConstraintSupport {
        ScalarServingJSONObjectConstraintSupport(
            classifications: [.eos], thinkEndTokenID: nil)
    }

    /// The bound decoder in this backend (`ScriptedDecoder`) does NOT itself support the
    /// constraint (see `testCapabilityFalseWhenDecoderDoesNotSupportConstraintEvenWithRouteAndSupportPresent`
    /// immediately below), but the capability property reads ONLY `configuration`'s three fields —
    /// this test pins that `configuration.decoderSupportsResponseFormatConstraint: true` is what
    /// admits here, not anything this fixture's `InferenceActor`/decoder itself reports.
    func testCapabilityTrueOnlyWhenSupportPresentAndRouteIsNonSpeculative() {
        XCTAssertTrue(
            makeCapabilityTestBackend(
                jsonObjectConstraintSupport: fakeSupport(), isNonSpeculativeScalarRoute: true,
                decoderSupportsResponseFormatConstraint: true
            ).supportsJSONObjectResponseFormat)
    }

    /// A decoder that cannot take the constraint must yield `false` even with the support table
    /// built and the route flag `true` — the third, independent AND-term
    /// `decoderSupportsResponseFormatConstraint` exists precisely so this combination cannot
    /// silently admit `json_object` against a decoder that does not actually honor it.
    func testCapabilityFalseWhenDecoderDoesNotSupportConstraintEvenWithRouteAndSupportPresent() {
        XCTAssertFalse(
            makeCapabilityTestBackend(
                jsonObjectConstraintSupport: fakeSupport(), isNonSpeculativeScalarRoute: true,
                decoderSupportsResponseFormatConstraint: false
            ).supportsJSONObjectResponseFormat)
    }

    /// Every SPECULATIVE/compiled route stays `false` even when the tokenizer would otherwise
    /// support the constraint — the response-format design's "MTP eligibility cannot see this
    /// processor" rule.
    func testCapabilityFalseWhenRouteIsSpeculativeEvenWithSupport() {
        XCTAssertFalse(
            makeCapabilityTestBackend(
                jsonObjectConstraintSupport: fakeSupport(), isNonSpeculativeScalarRoute: false,
                decoderSupportsResponseFormatConstraint: true
            ).supportsJSONObjectResponseFormat)
    }

    func testCapabilityFalseWhenSupportIsNilEvenOnTheScalarRoute() {
        XCTAssertFalse(
            makeCapabilityTestBackend(
                jsonObjectConstraintSupport: nil, isNonSpeculativeScalarRoute: true,
                decoderSupportsResponseFormatConstraint: true
            ).supportsJSONObjectResponseFormat)
    }

    /// Every EXISTING construction site (predating this feature) leaves both fields at their
    /// defaults — must compute the SAME `false` capability as before this feature existed.
    func testDefaultConfigurationCapabilityIsFalse() {
        let backend = ScalarServingBackend(
            launchedModel: "fixture-model",
            inference: InferenceActor(decoder: ScriptedDecoder(script: [99], eos: 99)),
            codec: UnusedScalarServingTextCodec(),
            stopTokenIDs: [99],
            modelStopStrings: [],
            configuration: .init(
                defaultMaximumCompletionTokens: 8,
                maximumQueuedRequests: 2,
                queueRetryAfterSeconds: 2,
                mailboxCapacity: .init(maxDeltas: 4, maxBytes: 1_024)))
        XCTAssertFalse(backend.supportsJSONObjectResponseFormat)
    }
}
