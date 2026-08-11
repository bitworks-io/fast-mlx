import XCTest
@testable import HarnessCore

final class ExactPrefixRequestTests: XCTestCase {
    private func digest(_ character: Character) -> String {
        String(repeating: String(character), count: 64)
    }

    func testRequestContextValidatesEveryCallerOwnedIsolationAxis()
        throws
    {
        let request = try ExactPrefixRequestContext(
            isolationNamespaceSHA256: digest("a"),
            promptTemplateSHA256: digest("b"),
            toolsSHA256: digest("c"))

        XCTAssertEqual(
            request.isolationNamespaceSHA256, digest("a"))
        XCTAssertEqual(request.promptTemplateSHA256, digest("b"))
        XCTAssertEqual(request.toolsSHA256, digest("c"))

        for (field, namespace, template, tools) in [
            ("isolationNamespaceSHA256", "bad", digest("b"), digest("c")),
            ("promptTemplateSHA256", digest("a"), "bad", digest("c")),
            ("toolsSHA256", digest("a"), digest("b"), "bad"),
        ] {
            XCTAssertThrowsError(
                try ExactPrefixRequestContext(
                    isolationNamespaceSHA256: namespace,
                    promptTemplateSHA256: template,
                    toolsSHA256: tools)
            ) { error in
                XCTAssertEqual(
                    error as? ExactPrefixRequestError,
                    .invalidSemanticIdentity(field))
            }
        }
    }

    func testRequestContextCodableCannotBypassValidation() throws {
        let valid = try ExactPrefixRequestContext(
            isolationNamespaceSHA256: digest("a"),
            promptTemplateSHA256: digest("b"),
            toolsSHA256: digest("c"))
        let data = try JSONEncoder().encode(valid)
        XCTAssertEqual(
            try JSONDecoder().decode(
                ExactPrefixRequestContext.self, from: data),
            valid)

        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data)
                as? [String: Any])
        object["toolsSHA256"] = "not-a-digest"
        let malformed = try JSONSerialization.data(
            withJSONObject: object)
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                ExactPrefixRequestContext.self, from: malformed))
    }

    func testLoadConfigurationIsDisabledByDefaultAndWarmupNeedsBudget()
        throws
    {
        XCTAssertEqual(ExactPrefixCacheConfiguration.disabled.policy, .disabled)
        XCTAssertFalse(
            ExactPrefixCacheConfiguration.disabled.eagerWarmupEnabled)

        XCTAssertThrowsError(
            try ExactPrefixCacheConfiguration(
                policy: .disabled,
                eagerWarmupEnabled: true)
        ) { error in
            XCTAssertEqual(
                error as? ExactPrefixRequestError,
                .warmupRequiresEnabledCache)
        }

        let enabled = try ExactPrefixCacheConfiguration(
            policy: ExactPrefixCachePolicy(
                maxEntries: 3,
                maxRetainedBytes: 1 << 20,
                minimumReusableTokens: 2),
            eagerWarmupEnabled: true)
        XCTAssertTrue(enabled.policy.isEnabled)
        XCTAssertTrue(enabled.eagerWarmupEnabled)
    }

    func testRunConfigPrefixRequestDefaultsOffAndParticipatesInIdentity()
        throws
    {
        let baseline = RunConfig.greedy(maxTokens: 8)
        XCTAssertNil(baseline.exactPrefixRequest)

        var enabled = baseline
        enabled.exactPrefixRequest = try ExactPrefixRequestContext(
            isolationNamespaceSHA256: digest("a"),
            promptTemplateSHA256: digest("b"),
            toolsSHA256: digest("c"))
        XCTAssertNotEqual(enabled, baseline)
        XCTAssertNotEqual(enabled.hashValue, baseline.hashValue)
    }
}
