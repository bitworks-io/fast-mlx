import XCTest

@testable import ServingCore

final class ServingModelCapabilitiesTests: XCTestCase {
    func testAutoMaximumUsesEffectiveHostFitContextAndPreservesNativeContext() throws {
        let capabilities = try makeCapabilities(
            native: 262_144,
            effective: 131_072)

        XCTAssertEqual(capabilities.model, "qwen-test")
        XCTAssertEqual(capabilities.nativeMaxContextTokens, 262_144)
        XCTAssertEqual(capabilities.effectiveMaxContextTokens, 131_072)
        XCTAssertEqual(capabilities.defaultCompletionTokens, 4_096)
        XCTAssertEqual(capabilities.maximumCompletionTokens, 131_071)
        XCTAssertEqual(capabilities.maximumNonStreamingCompletionTokens, 16_384)
        XCTAssertEqual(capabilities.maximumRequestBodyBytes, 8 * 1_048_576)
        XCTAssertEqual(capabilities.maximumNonStreamingResponseBytes, 16 * 1_048_576)
        XCTAssertEqual(capabilities.completionLimitPolicy, .reject)
        XCTAssertTrue(capabilities.reasoningTokensCountTowardCompletion)
    }

    func testImplicitDefaultAndNonStreamingMaximumReduceForSmallContext() throws {
        let capabilities = try makeCapabilities(
            native: 2_048,
            effective: 1_024)

        XCTAssertEqual(capabilities.defaultCompletionTokens, 1_023)
        XCTAssertEqual(capabilities.maximumCompletionTokens, 1_023)
        XCTAssertEqual(capabilities.maximumNonStreamingCompletionTokens, 1_023)
    }

    func testExplicitOperatorMaximumAndDefaultArePreserved() throws {
        let capabilities = try makeCapabilities(
            native: 32_768,
            effective: 16_384,
            requestedDefault: 8_192,
            defaultWasExplicit: true,
            maximum: 12_000,
            nonStreamingMaximum: 10_000,
            requestBodyMaximum: 32 * 1_048_576,
            nonStreamingResponseMaximum: 64 * 1_048_576,
            policy: .clamp)

        XCTAssertEqual(capabilities.defaultCompletionTokens, 8_192)
        XCTAssertEqual(capabilities.maximumCompletionTokens, 12_000)
        XCTAssertEqual(capabilities.maximumNonStreamingCompletionTokens, 10_000)
        XCTAssertEqual(capabilities.maximumRequestBodyBytes, 32 * 1_048_576)
        XCTAssertEqual(capabilities.maximumNonStreamingResponseBytes, 64 * 1_048_576)
        XCTAssertEqual(capabilities.completionLimitPolicy, .clamp)
    }

    func testInvalidCapabilitiesFailClosed() {
        XCTAssertThrowsError(try makeCapabilities(native: 0, effective: 0))
        XCTAssertThrowsError(try makeCapabilities(native: 32, effective: 33))
        XCTAssertThrowsError(try makeCapabilities(native: 1, effective: 1))
        XCTAssertThrowsError(
            try makeCapabilities(
                native: 32,
                effective: 16,
                maximum: 16))
        XCTAssertThrowsError(
            try makeCapabilities(
                native: 32,
                effective: 16,
                requestedDefault: 15,
                defaultWasExplicit: true,
                maximum: 8))
        XCTAssertThrowsError(
            try makeCapabilities(
                native: 32,
                effective: 16,
                nonStreamingMaximum: 0))
        XCTAssertThrowsError(
            try makeCapabilities(
                native: 32,
                effective: 16,
                requestBodyMaximum: 0))
        XCTAssertThrowsError(
            try makeCapabilities(
                native: 32,
                effective: 16,
                nonStreamingResponseMaximum: 0))
    }

    func testAutomaticRequestBodyLimitScalesForLongContextAndRemainsBounded() throws {
        XCTAssertEqual(
            try makeCapabilities(native: 4_096, effective: 4_096)
                .maximumRequestBodyBytes,
            1_048_576)
        XCTAssertEqual(
            try makeCapabilities(native: 262_144, effective: 262_144)
                .maximumRequestBodyBytes,
            16 * 1_048_576)
        XCTAssertEqual(
            try makeCapabilities(native: 1_048_576, effective: 1_048_576)
                .maximumRequestBodyBytes,
            64 * 1_048_576)
        XCTAssertEqual(
            try makeCapabilities(native: 4_194_304, effective: 4_194_304)
                .maximumRequestBodyBytes,
            64 * 1_048_576)
    }

    func testResponseMailboxCapacityUsesAdvertisedResponseByteLimit() throws {
        let capabilities = try makeCapabilities(
            native: 262_144,
            effective: 131_072,
            nonStreamingResponseMaximum: 32 * 1_048_576)

        XCTAssertEqual(
            capabilities.responseMailboxCapacity(maxDeltas: 8),
            .init(maxDeltas: 8, maxBytes: 32 * 1_048_576))
    }

    func testExplicitBudgetWithinPromptAwareLimitIsUnchanged() throws {
        let capabilities = try makeCapabilities(native: 16_384, effective: 8_192)

        let resolution = try capabilities.resolveCompletionBudget(
            requestedCompletionTokens: 6_000,
            renderedPromptTokens: 2_000,
            stream: true)

        XCTAssertEqual(resolution.requestedCompletionTokens, 6_000)
        XCTAssertEqual(resolution.appliedCompletionTokens, 6_000)
        XCTAssertEqual(resolution.maximumAllowedCompletionTokens, 6_192)
        XCTAssertEqual(resolution.renderedPromptTokens, 2_000)
        XCTAssertFalse(resolution.wasClamped)
        XCTAssertEqual(resolution.limitingFactor, .contextWindow)
    }

    func testRejectModeReturnsTypedErrorAbovePromptAwareLimit() throws {
        let capabilities = try makeCapabilities(native: 16_384, effective: 8_192)

        XCTAssertThrowsError(
            try capabilities.resolveCompletionBudget(
                requestedCompletionTokens: 6_193,
                renderedPromptTokens: 2_000,
                stream: true)
        ) { error in
            XCTAssertEqual(
                error as? OpenAIServingError,
                .invalidRequestWithCode(
                    "max_completion_tokens 6193 exceeds the maximum allowed 6192 for this rendered prompt",
                    param: "max_completion_tokens",
                    code: "completion_limit_exceeded"))
        }
    }

    func testClampModeAppliesExactPromptAwareLimit() throws {
        let capabilities = try makeCapabilities(
            native: 16_384,
            effective: 8_192,
            policy: .clamp)

        let resolution = try capabilities.resolveCompletionBudget(
            requestedCompletionTokens: 7_000,
            renderedPromptTokens: 2_000,
            stream: true)

        XCTAssertEqual(resolution.requestedCompletionTokens, 7_000)
        XCTAssertEqual(resolution.appliedCompletionTokens, 6_192)
        XCTAssertEqual(resolution.maximumAllowedCompletionTokens, 6_192)
        XCTAssertTrue(resolution.wasClamped)
    }

    func testOmittedBudgetUsesRemainingContextWithoutTreatingItAsClientClamp() throws {
        let capabilities = try makeCapabilities(native: 8_192, effective: 8_192)

        let resolution = try capabilities.resolveCompletionBudget(
            requestedCompletionTokens: nil,
            renderedPromptTokens: 7_000,
            stream: false)

        XCTAssertNil(resolution.requestedCompletionTokens)
        XCTAssertEqual(resolution.appliedCompletionTokens, 1_192)
        XCTAssertEqual(resolution.maximumAllowedCompletionTokens, 1_192)
        XCTAssertFalse(resolution.wasClamped)
    }

    func testOneTokenContextRemainderIsAdmitted() throws {
        let capabilities = try makeCapabilities(native: 8_192, effective: 8_192)

        let resolution = try capabilities.resolveCompletionBudget(
            requestedCompletionTokens: 1,
            renderedPromptTokens: 8_191,
            stream: false)

        XCTAssertEqual(resolution.appliedCompletionTokens, 1)
        XCTAssertEqual(resolution.maximumAllowedCompletionTokens, 1)
    }

    func testExhaustedContextReturnsTypedError() throws {
        let capabilities = try makeCapabilities(native: 8_192, effective: 8_192)

        for promptTokens in [8_192, 8_193] {
            XCTAssertThrowsError(
                try capabilities.resolveCompletionBudget(
                    requestedCompletionTokens: 1,
                    renderedPromptTokens: promptTokens,
                    stream: true)
            ) { error in
                XCTAssertEqual(
                    error as? OpenAIServingError,
                    .invalidRequestWithCode(
                        "The rendered prompt uses \(promptTokens) tokens and exceeds the effective context limit 8192",
                        param: "messages",
                        code: "context_length_exceeded"))
            }
        }
    }

    func testLongNonStreamingBudgetRequiresStreamingBeforeGeneration() throws {
        let capabilities = try makeCapabilities(
            native: 65_536,
            effective: 65_536,
            nonStreamingMaximum: 16_384)

        XCTAssertThrowsError(
            try capabilities.resolveCompletionBudget(
                requestedCompletionTokens: 16_385,
                renderedPromptTokens: 100,
                stream: false)
        ) { error in
            XCTAssertEqual(
                error as? OpenAIServingError,
                .invalidRequestWithCode(
                    "Non-streaming responses are limited to 16384 completion tokens; set stream=true for 16385 tokens",
                    param: "stream",
                    code: "stream_required"))
        }

        XCTAssertEqual(
            try capabilities.resolveCompletionBudget(
                requestedCompletionTokens: 16_385,
                renderedPromptTokens: 100,
                stream: true)
                .appliedCompletionTokens,
            16_385)
    }

    private func makeCapabilities(
        native: Int,
        effective: Int,
        requestedDefault: Int = 4_096,
        defaultWasExplicit: Bool = false,
        maximum: Int? = nil,
        nonStreamingMaximum: Int = 16_384,
        requestBodyMaximum: Int? = nil,
        nonStreamingResponseMaximum: Int = 16 * 1_048_576,
        policy: ServingCompletionLimitPolicy = .reject
    ) throws -> ServingModelCapabilities {
        try ServingModelCapabilities(
            model: "qwen-test",
            nativeMaxContextTokens: native,
            effectiveMaxContextTokens: effective,
            requestedDefaultCompletionTokens: requestedDefault,
            defaultCompletionTokensWasExplicit: defaultWasExplicit,
            maximumCompletionTokens: maximum,
            maximumNonStreamingCompletionTokens: nonStreamingMaximum,
            maximumRequestBodyBytes: requestBodyMaximum,
            maximumNonStreamingResponseBytes: nonStreamingResponseMaximum,
            completionLimitPolicy: policy)
    }
}
