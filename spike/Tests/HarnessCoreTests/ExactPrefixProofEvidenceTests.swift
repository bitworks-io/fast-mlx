import XCTest
@testable import HarnessCore

final class ExactPrefixProofEvidenceTests: XCTestCase {
    func testCompleteProofRoundTripsAndPromotesWhenWarmStartsBeatControls()
        throws
    {
        let evidence = try proofEvidence(warmRequestStartSeconds: [
            .exactHitA: 0.010,
            .returnHitA: 0.011,
            .postWarmupHit: 0.012,
        ])

        XCTAssertEqual(evidence.schemaVersion, 2)
        XCTAssertEqual(
            evidence.cases[1].requestStartMetrics?
                .runtimeIdentity?.observedDenseHalfDType,
            .float16)
        XCTAssertEqual(
            evidence.cases[1].requestStartMetrics?
                .runtimeIdentity?.kvRouteSHA256,
            digest(
                """
                fast-mlx-exact-prefix-kv-route-v2
                scalar
                compiled
                observed-dense-half=float16
                full-attention
                """))
        XCTAssertTrue(evidence.byteIdentityPassed)
        XCTAssertTrue(evidence.engagementPassed)
        XCTAssertTrue(evidence.boundedPassed)
        XCTAssertTrue(evidence.warmBenefitPassed)
        XCTAssertTrue(evidence.promotable)
        XCTAssertEqual(
            evidence.cases.map(\.caseID),
            ExactPrefixProofCaseID.requiredOrder)
        XCTAssertEqual(
            try JSONDecoder().decode(
                ExactPrefixProofEvidence.self,
                from: JSONEncoder().encode(evidence)),
            evidence)
    }

    func testNegativeWarmBenefitIsValidButNotPromotable() throws {
        let evidence = try proofEvidence(warmRequestStartSeconds: [
            .exactHitA: 0.060,
            .returnHitA: 0.061,
            .postWarmupHit: 0.062,
        ])

        XCTAssertTrue(evidence.byteIdentityPassed)
        XCTAssertTrue(evidence.engagementPassed)
        XCTAssertTrue(evidence.boundedPassed)
        XCTAssertFalse(evidence.warmBenefitPassed)
        XCTAssertFalse(evidence.promotable)
    }

    func testProofBindsAndValidatesTheConfiguredWorkloadShape() throws {
        let evidence = try proofEvidence()

        XCTAssertEqual(evidence.maxTokens, 16)
        XCTAssertEqual(evidence.promptRepeat, 8)
        XCTAssertThrowsError(try proofEvidence(maxTokens: 1)) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofEvidenceError,
                .invalidMaxTokens)
        }
        XCTAssertThrowsError(try proofEvidence(promptRepeat: 0)) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofEvidenceError,
                .invalidPromptRepeat)
        }
    }

    func testRuntimeIdentityTamperingFailsClosed()
        throws
    {
        let evidence = try proofEvidence()
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: JSONEncoder().encode(evidence))
                as? [String: Any])
        var cases = try XCTUnwrap(
            object["cases"] as? [[String: Any]])
        var requestStartMetrics = try XCTUnwrap(
            cases[1]["requestStartMetrics"]
                as? [String: Any])
        var runtimeIdentity = try XCTUnwrap(
            requestStartMetrics["runtimeIdentity"]
                as? [String: Any])
        runtimeIdentity["kvRouteSHA256"] =
            String(repeating: "0", count: 64)
        requestStartMetrics["runtimeIdentity"] =
            runtimeIdentity
        cases[1]["requestStartMetrics"] =
            requestStartMetrics
        object["cases"] = cases
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                ExactPrefixProofEvidence.self,
                from: JSONSerialization.data(
                    withJSONObject: object,
                    options: [.sortedKeys])))

    }

    func testSchemaOneNegativeDiagnosticsRemainReadableButCannotPromote()
        throws
    {
        var outcomeMiss = try proofEvidence().cases
        outcomeMiss[2] = outcomeMiss[2]
            .replacing(requestStartMetrics: metrics(
                outcome: .miss,
                templateHit: true,
                promptTokens: 64))
            .replacing(
                reusedPrefixTokenIDsSHA256: nil,
                reusedPrefixTokenCount: 0)
        let negative = try proofEvidence(cases: outcomeMiss)
        let decodedNegative = try JSONDecoder().decode(
            ExactPrefixProofEvidence.self,
            from: JSONSerialization.data(
                withJSONObject:
                    try legacySchemaOneObject(from: negative),
                options: [.sortedKeys]))

        XCTAssertEqual(decodedNegative.schemaVersion, 1)
        XCTAssertFalse(decodedNegative.engagementPassed)
        XCTAssertFalse(decodedNegative.promotable)

        XCTAssertThrowsError(
            try JSONDecoder().decode(
                ExactPrefixProofEvidence.self,
                from: JSONSerialization.data(
                    withJSONObject:
                        try legacySchemaOneObject(
                            from: proofEvidence()),
                    options: [.sortedKeys])))
        {
            XCTAssertEqual(
                $0 as? ExactPrefixProofEvidenceError,
                .legacyPromotableEvidence)
        }
    }

    func testEveryRequiredWarmCellMustBeatItsPairedControl() throws {
        var cases = try proofEvidence().cases
        cases[4] = cases[4].replacing(
            timing: try ExactPrefixProofCaseTiming(
                requestStartSeconds: 0.056,
                ttftSeconds: 0.076,
                decodeSeconds: 0.040,
                totalSeconds: 0.116))

        let evidence = try proofEvidence(cases: cases)
        XCTAssertFalse(evidence.warmBenefitPassed)
        XCTAssertFalse(evidence.promotable)

        var sameTemplateMissWins = try proofEvidence().cases
        sameTemplateMissWins[7] = sameTemplateMissWins[7].replacing(
            timing: try ExactPrefixProofCaseTiming(
                requestStartSeconds: 0.005,
                ttftSeconds: 0.025,
                decodeSeconds: 0.040,
                totalSeconds: 0.065))
        let sameTemplateNegative = try proofEvidence(
            cases: sameTemplateMissWins)
        XCTAssertFalse(sameTemplateNegative.warmBenefitPassed)
        XCTAssertFalse(sameTemplateNegative.promotable)

        var postWarmupMissWins = try proofEvidence().cases
        postWarmupMissWins[9] = postWarmupMissWins[9].replacing(
            timing: try ExactPrefixProofCaseTiming(
                requestStartSeconds: 0.005,
                ttftSeconds: 0.025,
                decodeSeconds: 0.040,
                totalSeconds: 0.065))
        let postWarmupNegative = try proofEvidence(
            cases: postWarmupMissWins)
        XCTAssertFalse(postWarmupNegative.warmBenefitPassed)
        XCTAssertFalse(postWarmupNegative.promotable)
    }

    func testProofFailsClosedForIdentityCaseAndReceiptMismatches()
        throws
    {
        XCTAssertThrowsError(try proofEvidence(modelID: "/tmp/model"))
        XCTAssertThrowsError(try proofEvidence(sourceRevision: "abc123"))
        XCTAssertThrowsError(try proofEvidence(
            sourceRevision: digest("other-checkpoint")))
        {
            XCTAssertEqual(
                $0 as? ExactPrefixProofEvidenceError,
                .sourceRevisionIdentityMismatch)
        }
        XCTAssertThrowsError(try proofEvidence(expectedHarnessSHA: "ABC123"))
        XCTAssertThrowsError(try proofEvidence(expectedExecutableSHA256: "abc"))

        var missingCase = try proofEvidence().cases
        missingCase.removeLast()
        XCTAssertThrowsError(try proofEvidence(cases: missingCase)) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofEvidenceError,
                .caseOrderMismatch)
        }

        var duplicateCase = try proofEvidence().cases
        duplicateCase[1] = duplicateCase[0]
        XCTAssertThrowsError(try proofEvidence(cases: duplicateCase)) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofEvidenceError,
                .caseOrderMismatch)
        }

        var outOfOrder = try proofEvidence().cases
        outOfOrder.swapAt(1, 2)
        XCTAssertThrowsError(try proofEvidence(cases: outOfOrder)) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofEvidenceError,
                .caseOrderMismatch)
        }

        var badReference = try proofEvidence().cases
        badReference[2] = badReference[2].replacing(
            referenceOutputSHA256: digest("different-output"))
        XCTAssertThrowsError(try proofEvidence(cases: badReference)) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofEvidenceError,
                .referenceHashMismatch(.exactHitA))
        }

        var badPrompt = try proofEvidence().cases
        badPrompt[2] = badPrompt[2].replacing(
            promptTokenIDsSHA256: digest("different-prompt"))
        XCTAssertThrowsError(try proofEvidence(cases: badPrompt)) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofEvidenceError,
                .groupReferenceMismatch(.exactHitA))
        }

        var missingPromptCount = try proofEvidence().cases
        missingPromptCount[2] = missingPromptCount[2].replacing(
            promptTokenCount: 0)
        XCTAssertThrowsError(
            try proofEvidence(cases: missingPromptCount))
        {
            XCTAssertEqual(
                $0 as? ExactPrefixProofEvidenceError,
                .invalidPromptTokenCount(.exactHitA))
        }

        var controlWithMetrics = try proofEvidence().cases
        controlWithMetrics[0] = controlWithMetrics[0].replacing(
            requestStartMetrics: metrics(outcome: .miss, templateHit: false))
        XCTAssertThrowsError(try proofEvidence(cases: controlWithMetrics)) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofEvidenceError,
                .controlCarriesRequestMetrics(.coldControlA))
        }

        var coldCommitHit = try proofEvidence().cases
        coldCommitHit[1] = coldCommitHit[1].replacing(
            requestStartMetrics: metrics(
                outcome: .exactHit,
                templateHit: false,
                promptTokens: 64))
        XCTAssertThrowsError(try proofEvidence(cases: coldCommitHit)) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofEvidenceError,
                .reusedPrefixMismatch(.coldCommitA))
        }

        var templateMismatch = try proofEvidence().cases
        templateMismatch[2] = templateMismatch[2].replacing(
            templateTokenCacheReceipt: .miss)
        XCTAssertThrowsError(try proofEvidence(cases: templateMismatch)) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofEvidenceError,
                .templateTokenCacheMismatch(.exactHitA))
        }

        var missingMemory = try proofEvidence().cases
        missingMemory[3] = missingMemory[3].replacing(memoryEvidence: nil)
        XCTAssertThrowsError(try proofEvidence(cases: missingMemory)) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofEvidenceError,
                .partialEvidence(.partialControl))
        }

        var missingWarmupBinding = try proofEvidence().cases
        missingWarmupBinding[10] = missingWarmupBinding[10].replacing(
            requestStartMetrics: metrics(
                outcome: .exactHit,
                templateHit: true))
        XCTAssertThrowsError(
            try proofEvidence(cases: missingWarmupBinding))
        {
            XCTAssertEqual(
                $0 as? ExactPrefixProofEvidenceError,
                .warmupDurationMismatch(.postWarmupHit))
        }

        var wrongPartialPrefix = try proofEvidence().cases
        wrongPartialPrefix[4] = wrongPartialPrefix[4].replacing(
            reusedPrefixTokenIDsSHA256: digest("wrong-prefix"),
            reusedPrefixTokenCount: 64)
        XCTAssertThrowsError(
            try proofEvidence(cases: wrongPartialPrefix)
        ) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofEvidenceError,
                .reusedPrefixMismatch(.partialHit))
        }

        var wrongRequestContext = try proofEvidence().cases
        wrongRequestContext[2] = wrongRequestContext[2].replacing(
            requestContext: try ExactPrefixRequestContext(
                isolationNamespaceSHA256: digest("other-namespace"),
                promptTemplateSHA256: digest("template"),
                toolsSHA256: digest("tools")))
        XCTAssertThrowsError(
            try proofEvidence(cases: wrongRequestContext)
        ) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofEvidenceError,
                .requestContextMismatch(.exactHitA))
        }
    }

    func testObservedNegativeDiagnosticsValidateButNeverPromote()
        throws
    {
        var outputMismatch = try proofEvidence().cases
        outputMismatch[2] = outputMismatch[2].replacing(
            outputSHA256: digest("observed-mismatch"))
        let byteNegative = try proofEvidence(cases: outputMismatch)
        XCTAssertFalse(byteNegative.byteIdentityPassed)
        XCTAssertFalse(byteNegative.promotable)

        var outcomeMiss = try proofEvidence().cases
        outcomeMiss[2] = outcomeMiss[2]
            .replacing(requestStartMetrics: metrics(
                outcome: .miss,
                templateHit: true,
                promptTokens: 64))
            .replacing(
                reusedPrefixTokenIDsSHA256: nil,
                reusedPrefixTokenCount: 0)
        let engagementNegative = try proofEvidence(cases: outcomeMiss)
        XCTAssertFalse(engagementNegative.engagementPassed)
        XCTAssertFalse(engagementNegative.promotable)

        var templateMiss = try proofEvidence().cases
        templateMiss[2] = templateMiss[2]
            .replacing(requestStartMetrics: metrics(
                outcome: .exactHit,
                templateHit: false,
                promptTokens: 64))
            .replacing(templateTokenCacheReceipt: .miss)
        let templateNegative = try proofEvidence(cases: templateMiss)
        XCTAssertFalse(templateNegative.engagementPassed)
        XCTAssertFalse(templateNegative.promotable)

        var noEviction = try proofEvidence().cases
        noEviction[7] = noEviction[7].replacing(
            requestStartMetrics: metrics(
                outcome: .miss,
                templateHit: true,
                promptTokens: 64,
                evictionCount: 0))
        let evictionNegative = try proofEvidence(cases: noEviction)
        XCTAssertFalse(evictionNegative.engagementPassed)
        XCTAssertFalse(evictionNegative.promotable)

        var overBudget = try proofEvidence().cases
        overBudget[0] = overBudget[0].replacing(
            memoryEvidence: memoryEvidence(mlxCacheBytes: 257))
        let boundedNegative = try proofEvidence(cases: overBudget)
        XCTAssertFalse(boundedNegative.boundedPassed)
        XCTAssertFalse(boundedNegative.promotable)

        var overFootprint = try proofEvidence().cases
        overFootprint[0] = overFootprint[0].replacing(
            memoryEvidence: memoryEvidence(
                physicalFootprintBytes: 1_025))
        let footprintNegative = try proofEvidence(cases: overFootprint)
        XCTAssertFalse(footprintNegative.boundedPassed)
        XCTAssertFalse(footprintNegative.promotable)
    }

    func testProofFailsClosedForWarmupBoundsAndClaimedDerivedValues()
        throws
    {
        XCTAssertThrowsError(try proofEvidence(
            warmup: ExactPrefixProofWarmupEvidence(
                durationSeconds: 0,
                before: emptyPrefixSnapshot(),
                after: emptyPrefixSnapshot())))

        XCTAssertThrowsError(try proofEvidence(
            warmup: ExactPrefixProofWarmupEvidence(
                durationSeconds: 1,
                before: emptyPrefixSnapshot(),
                after: prefixSnapshot(entryCount: 1))))
        {
            XCTAssertEqual(
                $0 as? ExactPrefixProofEvidenceError,
                .warmupMutatedCache)
        }

        XCTAssertThrowsError(try proofEvidence(
            terminalNonPartial: false))
        {
            XCTAssertEqual(
                $0 as? ExactPrefixProofEvidenceError,
                .partialEvidence(.postWarmupHit))
        }

        XCTAssertThrowsError(try proofEvidence(
            exactPrefixCachePolicy: ExactPrefixCachePolicy(
                maxEntries: 65,
                maxRetainedBytes: 128,
                minimumReusableTokens: 8)))
        {
            XCTAssertEqual(
                $0 as? ExactPrefixProofEvidenceError,
                .invalidCachePolicy)
        }

        XCTAssertThrowsError(try proofEvidence(
            templateTokenCachePolicy: TemplateTokenCachePolicy(
                maxEntries: 65,
                maxRetainedBytes: 128)))
        {
            XCTAssertEqual(
                $0 as? ExactPrefixProofEvidenceError,
                .invalidTemplateTokenCachePolicy)
        }

        XCTAssertThrowsError(try proofEvidence(
            exactPrefixCachePolicy: ExactPrefixCachePolicy(
                maxEntries: 8,
                maxRetainedBytes: 900,
                minimumReusableTokens: 8),
            templateTokenCachePolicy: TemplateTokenCachePolicy(
                maxEntries: 8,
                maxRetainedBytes: 200)))
        {
            XCTAssertEqual(
                $0 as? ExactPrefixProofEvidenceError,
                .invalidMemoryLimits)
        }

        XCTAssertThrowsError(try proofEvidence(
            derived: ExactPrefixProofDerived(
                byteIdentityPassed: false,
                engagementPassed: true,
                boundedPassed: true,
                warmBenefitPassed: true,
                promotable: true)))
        {
            XCTAssertEqual(
                $0 as? ExactPrefixProofEvidenceError,
                .derivedClaimMismatch)
        }

    }

    func testCommandPlanValidatesPureCLIContract() throws {
        let plan = try ExactPrefixProofCommandPlan(
            modelID: "loaded-qwen3",
            sourceRevision: checkpointSHA,
            expectedHarnessSHA: digest("harness"),
            expectedExecutableSHA256: binarySHA,
            admission: admission(),
            checkpointContentSHA256: checkpointSHA,
            tokenizerSHA256: tokenizerSHA,
            workloadNonce: "proof-nonce_1",
            maxTokens: 16,
            promptRepeat: 8,
            exactPrefixCachePolicy: exactPolicy(),
            templateTokenCachePolicy: templatePolicy(),
            memoryLimitBytes: 1_024,
            cacheLimitBytes: 256,
            outputPath: "exact-prefix-proof.json")

        XCTAssertEqual(plan.modelID, "loaded-qwen3")

        XCTAssertThrowsError(try commandPlan(modelID: "../model")) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofCommandPlanError,
                .invalidModelID)
        }
        XCTAssertThrowsError(try commandPlan(sourceRevision: "123")) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofCommandPlanError,
                .invalidSourceRevision)
        }
        XCTAssertThrowsError(try commandPlan(
            sourceRevision: digest("other-checkpoint")))
        {
            XCTAssertEqual(
                $0 as? ExactPrefixProofCommandPlanError,
                .sourceRevisionIdentityMismatch)
        }
        XCTAssertThrowsError(try commandPlan(expectedHarnessSHA: "ABC")) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofCommandPlanError,
                .invalidExpectedHarnessSHA)
        }
        XCTAssertThrowsError(try commandPlan(expectedExecutableSHA256: "abc")) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofCommandPlanError,
                .invalidExpectedExecutableSHA256)
        }
        XCTAssertThrowsError(try commandPlan(
            checkpointContentSHA256: digest("wrong")))
        {
            XCTAssertEqual(
                $0 as? ExactPrefixProofCommandPlanError,
                .checkpointIdentityMismatch)
        }
        XCTAssertThrowsError(try commandPlan(tokenizerSHA256: digest("wrong")))
        {
            XCTAssertEqual(
                $0 as? ExactPrefixProofCommandPlanError,
                .tokenizerIdentityMismatch)
        }
        XCTAssertThrowsError(try commandPlan(workloadNonce: "-bad")) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofCommandPlanError,
                .invalidWorkloadNonce)
        }
        XCTAssertThrowsError(try commandPlan(maxTokens: 129)) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofCommandPlanError,
                .invalidMaxTokens)
        }
        XCTAssertThrowsError(try commandPlan(promptRepeat: 0)) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofCommandPlanError,
                .invalidPromptRepeat)
        }
        XCTAssertThrowsError(try commandPlan(
            exactPrefixCachePolicy: ExactPrefixCachePolicy(
                maxEntries: 65,
                maxRetainedBytes: 128,
                minimumReusableTokens: 8)))
        {
            XCTAssertEqual(
                $0 as? ExactPrefixProofCommandPlanError,
                .invalidCachePolicy)
        }
        XCTAssertThrowsError(try commandPlan(
            templateTokenCachePolicy: TemplateTokenCachePolicy(
                maxEntries: 65,
                maxRetainedBytes: 128)))
        {
            XCTAssertEqual(
                $0 as? ExactPrefixProofCommandPlanError,
                .invalidTemplateTokenCachePolicy)
        }
        XCTAssertThrowsError(try commandPlan(
            exactPrefixCachePolicy: ExactPrefixCachePolicy(
                maxEntries: 8,
                maxRetainedBytes: 900,
                minimumReusableTokens: 8),
            templateTokenCachePolicy: TemplateTokenCachePolicy(
                maxEntries: 8,
                maxRetainedBytes: 200)))
        {
            XCTAssertEqual(
                $0 as? ExactPrefixProofCommandPlanError,
                .invalidMemoryLimits)
        }
        XCTAssertThrowsError(try commandPlan(cacheLimitBytes: 1_025)) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofCommandPlanError,
                .invalidMemoryLimits)
        }
        XCTAssertThrowsError(try commandPlan(outputPath: " \n ")) {
            XCTAssertEqual(
                $0 as? ExactPrefixProofCommandPlanError,
                .invalidOutputPath)
        }
    }
}

private let fortyHex = String(repeating: "a", count: 40)
private let checkpointSHA = String(repeating: "d", count: 64)
private let tokenizerSHA = String(repeating: "c", count: 64)
private let binarySHA = String(repeating: "b", count: 64)

private func legacySchemaOneObject(
    from evidence: ExactPrefixProofEvidence
) throws -> [String: Any] {
    var object = try XCTUnwrap(
        JSONSerialization.jsonObject(
            with: JSONEncoder().encode(evidence))
            as? [String: Any])
    var cases = try XCTUnwrap(
        object["cases"] as? [[String: Any]])
    for index in cases.indices {
        guard var requestStartMetrics =
            cases[index]["requestStartMetrics"] as? [String: Any]
        else { continue }
        requestStartMetrics.removeValue(forKey: "runtimeIdentity")
        cases[index]["requestStartMetrics"] = requestStartMetrics
    }
    object["schemaVersion"] = 1
    object["cases"] = cases
    return object
}

private func proofEvidence(
    modelID: String = "loaded-qwen3",
    sourceRevision: String = checkpointSHA,
    expectedHarnessSHA: String = digest("harness"),
    expectedExecutableSHA256: String = binarySHA,
    maxTokens: Int = 16,
    promptRepeat: Int = 8,
    exactPrefixCachePolicy: ExactPrefixCachePolicy? = nil,
    templateTokenCachePolicy: TemplateTokenCachePolicy? = nil,
    warmRequestStartSeconds:
        [ExactPrefixProofCaseID: Double] = [:],
    warmup: ExactPrefixProofWarmupEvidence? = nil,
    cases: [ExactPrefixProofCaseEvidence]? = nil,
    terminalNonPartial: Bool = true,
    derived: ExactPrefixProofDerived? = nil
) throws -> ExactPrefixProofEvidence {
    try ExactPrefixProofEvidence(
        modelID: modelID,
        sourceRevision: sourceRevision,
        admission: admission(),
        expectedHarnessSHA: expectedHarnessSHA,
        expectedExecutableSHA256: expectedExecutableSHA256,
        workloadNonce: "proof-nonce_1",
        maxTokens: maxTokens,
        promptRepeat: promptRepeat,
        exactPrefixCachePolicy:
            exactPrefixCachePolicy ?? exactPolicy(),
        templateTokenCachePolicy:
            templateTokenCachePolicy ?? templatePolicy(),
        memoryLimitBytes: 1_024,
        cacheLimitBytes: 256,
        requestContext: requestContext(),
        warmup: warmup ?? ExactPrefixProofWarmupEvidence(
            durationSeconds: 0.25,
            before: emptyPrefixSnapshot(),
            after: emptyPrefixSnapshot()),
        cases: cases ?? proofCases(
            warmRequestStartSeconds: warmRequestStartSeconds),
        terminalNonPartial: terminalNonPartial,
        derived: derived)
}

private func commandPlan(
    modelID: String = "loaded-qwen3",
    sourceRevision: String = checkpointSHA,
    expectedHarnessSHA: String = digest("harness"),
    expectedExecutableSHA256: String = binarySHA,
    checkpointContentSHA256: String = checkpointSHA,
    tokenizerSHA256: String = tokenizerSHA,
    workloadNonce: String = "proof-nonce_1",
    maxTokens: Int = 16,
    promptRepeat: Int = 8,
    exactPrefixCachePolicy: ExactPrefixCachePolicy? = nil,
    templateTokenCachePolicy: TemplateTokenCachePolicy? = nil,
    memoryLimitBytes: Int = 1_024,
    cacheLimitBytes: Int = 256,
    outputPath: String = "exact-prefix-proof.json"
) throws -> ExactPrefixProofCommandPlan {
    try ExactPrefixProofCommandPlan(
        modelID: modelID,
        sourceRevision: sourceRevision,
        expectedHarnessSHA: expectedHarnessSHA,
        expectedExecutableSHA256: expectedExecutableSHA256,
        admission: admission(),
        checkpointContentSHA256: checkpointContentSHA256,
        tokenizerSHA256: tokenizerSHA256,
        workloadNonce: workloadNonce,
        maxTokens: maxTokens,
        promptRepeat: promptRepeat,
        exactPrefixCachePolicy: exactPrefixCachePolicy ?? exactPolicy(),
        templateTokenCachePolicy: templateTokenCachePolicy ?? templatePolicy(),
        memoryLimitBytes: memoryLimitBytes,
        cacheLimitBytes: cacheLimitBytes,
        outputPath: outputPath)
}

private func proofCases(
    warmRequestStartSeconds: [ExactPrefixProofCaseID: Double]
) throws -> [ExactPrefixProofCaseEvidence] {
    let aTokens = digest("a-tokens")
    let aOutput = digest("a-output")
    let partialTokens = digest("partial-tokens")
    let partialOutput = digest("partial-output")
    let bTokens = digest("b-tokens")
    let bOutput = digest("b-output")
    let warmTokens = digest("warm-tokens")
    let warmOutput = digest("warm-output")

    return [
        try caseEvidence(
            .coldControlA,
            tokenSHA: aTokens,
            outputSHA: aOutput,
            referenceTokenSHA: aTokens,
            referenceOutputSHA: aOutput,
            requestStartSeconds: 0.050,
            metrics: nil,
            templateReceipt: .miss),
        try caseEvidence(
            .coldCommitA,
            tokenSHA: aTokens,
            outputSHA: aOutput,
            referenceTokenSHA: aTokens,
            referenceOutputSHA: aOutput,
            requestStartSeconds: 0.052,
            metrics: metrics(
                outcome: .miss,
                templateHit: false,
                promptTokens: 64),
            templateReceipt: .miss),
        try caseEvidence(
            .exactHitA,
            tokenSHA: aTokens,
            outputSHA: aOutput,
            referenceTokenSHA: aTokens,
            referenceOutputSHA: aOutput,
            requestStartSeconds: warmRequestStartSeconds[.exactHitA] ?? 0.010,
            metrics: metrics(
                outcome: .exactHit,
                templateHit: true,
                promptTokens: 64),
            templateReceipt: .hit),
        try caseEvidence(
            .partialControl,
            tokenSHA: partialTokens,
            outputSHA: partialOutput,
            referenceTokenSHA: partialTokens,
            referenceOutputSHA: partialOutput,
            requestStartSeconds: 0.055,
            metrics: nil,
            templateReceipt: .miss),
        try caseEvidence(
            .partialHit,
            tokenSHA: partialTokens,
            outputSHA: partialOutput,
            referenceTokenSHA: partialTokens,
            referenceOutputSHA: partialOutput,
            requestStartSeconds: 0.030,
            metrics: metrics(
                outcome: .partialHit,
                templateHit: true,
                cacheReadTokenCount: 64),
            templateReceipt: .hit),
        try caseEvidence(
            .coldCommitB,
            tokenSHA: bTokens,
            outputSHA: bOutput,
            referenceTokenSHA: bTokens,
            referenceOutputSHA: bOutput,
            requestStartSeconds: 0.051,
            metrics: metrics(
                outcome: .miss,
                templateHit: false,
                promptTokens: 96),
            templateReceipt: .miss),
        try caseEvidence(
            .returnHitA,
            tokenSHA: aTokens,
            outputSHA: aOutput,
            referenceTokenSHA: aTokens,
            referenceOutputSHA: aOutput,
            requestStartSeconds: warmRequestStartSeconds[.returnHitA] ?? 0.011,
            metrics: metrics(
                outcome: .exactHit,
                templateHit: true,
                promptTokens: 64),
            templateReceipt: .hit),
        try caseEvidence(
            .pressureEvictedA,
            tokenSHA: aTokens,
            outputSHA: aOutput,
            referenceTokenSHA: aTokens,
            referenceOutputSHA: aOutput,
            requestStartSeconds: 0.053,
            metrics: metrics(
                outcome: .miss,
                templateHit: true,
                promptTokens: 64,
                evictionCount: 1),
            templateReceipt: .hit),
        try caseEvidence(
            .postWarmupControl,
            tokenSHA: warmTokens,
            outputSHA: warmOutput,
            referenceTokenSHA: warmTokens,
            referenceOutputSHA: warmOutput,
            requestStartSeconds: 0.054,
            metrics: nil,
            templateReceipt: .miss),
        try caseEvidence(
            .postWarmupMiss,
            tokenSHA: warmTokens,
            outputSHA: warmOutput,
            referenceTokenSHA: warmTokens,
            referenceOutputSHA: warmOutput,
            requestStartSeconds: 0.050,
            metrics: metrics(
                outcome: .miss,
                templateHit: false,
                eagerWarmupSeconds: 0.25),
            templateReceipt: .miss),
        try caseEvidence(
            .postWarmupHit,
            tokenSHA: warmTokens,
            outputSHA: warmOutput,
            referenceTokenSHA: warmTokens,
            referenceOutputSHA: warmOutput,
            requestStartSeconds:
                warmRequestStartSeconds[.postWarmupHit] ?? 0.012,
            metrics: metrics(
                outcome: .exactHit,
                templateHit: true,
                eagerWarmupSeconds: 0.25),
            templateReceipt: .hit),
    ]
}

private func caseEvidence(
    _ id: ExactPrefixProofCaseID,
    tokenSHA: String,
    outputSHA: String,
    referenceTokenSHA: String,
    referenceOutputSHA: String,
    requestStartSeconds: Double,
    metrics: RequestStartMetrics?,
    templateReceipt: ExactPrefixProofCacheReceipt
) throws -> ExactPrefixProofCaseEvidence {
    let promptGroup: String
    let promptTokenCount: Int
    switch id {
    case .coldControlA, .coldCommitA, .exactHitA, .returnHitA,
        .pressureEvictedA:
        promptGroup = "prompt-a"
        promptTokenCount = 64
    case .partialControl, .partialHit:
        promptGroup = "prompt-partial"
        promptTokenCount = 128
    case .coldCommitB:
        promptGroup = "prompt-b"
        promptTokenCount = 96
    case .postWarmupControl, .postWarmupMiss, .postWarmupHit:
        promptGroup = "prompt-warm"
        promptTokenCount = 128
    }
    let reusedPrefix: (sha256: String?, count: Int)
    switch id {
    case .exactHitA, .partialHit, .returnHitA:
        reusedPrefix = (digest("prompt-a"), 64)
    case .postWarmupHit:
        reusedPrefix = (digest("prompt-warm"), 128)
    case .coldControlA, .coldCommitA, .partialControl,
        .coldCommitB, .pressureEvictedA, .postWarmupControl,
        .postWarmupMiss:
        reusedPrefix = (nil, 0)
    }
    return try ExactPrefixProofCaseEvidence(
        caseID: id,
        promptTokenIDsSHA256: digest(promptGroup),
        promptTokenCount: promptTokenCount,
        generatedTokenIDsSHA256: tokenSHA,
        outputSHA256: outputSHA,
        referenceGeneratedTokenIDsSHA256: referenceTokenSHA,
        referenceOutputSHA256: referenceOutputSHA,
        generatedTokenCount: 8,
        timing: ExactPrefixProofCaseTiming(
            requestStartSeconds: requestStartSeconds,
            ttftSeconds: requestStartSeconds + 0.020,
            decodeSeconds: 0.040,
            totalSeconds: requestStartSeconds + 0.060),
        requestStartMetrics: metrics,
        memoryEvidence: memoryEvidence(),
        templateTokenCacheReceipt: templateReceipt,
        reusedPrefixTokenIDsSHA256: reusedPrefix.sha256,
        reusedPrefixTokenCount: reusedPrefix.count,
        requestContext: id.isControl ? nil : requestContext())
}

private func metrics(
    outcome: PrefixCacheRequestOutcome,
    templateHit: Bool,
    cacheReadTokenCount: Int = 0,
    promptTokens: Int = 128,
    evictionCount: Int = 0,
    eagerWarmupSeconds: Double? = nil
) -> RequestStartMetrics {
    let readTokens: Int
    switch outcome {
    case .exactHit:
        readTokens = promptTokens
    case .partialHit:
        readTokens = cacheReadTokenCount
    case .disabled, .miss, .rejected:
        readTokens = 0
    }
    return try! RequestStartMetrics(
        promptTokenCount: promptTokens,
        cacheReadTokenCount: readTokens,
        physicalPrefillTokenCount: promptTokens - readTokens,
        prefixCacheOutcome: outcome,
        templateTokenCacheHit: templateHit,
        templateSeconds: 0.001,
        tokenizeSeconds: 0.001,
        lookupSeconds: 0.001,
        restoreSeconds: outcome == .exactHit ? 0.001 : 0,
        prefillSeconds: outcome == .exactHit ? 0.001 : 0.010,
        retainedBytes: outcome == .miss ? 0 : 512,
        entryCount: outcome == .miss ? 0 : 1,
        evictionCount: evictionCount,
        runtimeIdentity:
            try! ExactPrefixDenseRuntimeIdentityEvidence(
                observedDenseHalfDType: .float16),
        eagerWarmupSeconds: eagerWarmupSeconds)
}

private func memoryEvidence(
    mlxCacheBytes: Int = 128,
    physicalFootprintBytes: UInt64 = 900
) -> BenchRunMemoryEvidence {
    try! BenchRunMemoryEvidence(samples: [
        ServiceMemorySample(
            timestamp: 1,
            physicalFootprintBytes: min(
                physicalFootprintBytes, 800),
            mlxActiveBytes: 100,
            mlxCacheBytes: mlxCacheBytes,
            mlxPeakBytes: 200),
        ServiceMemorySample(
            timestamp: 2,
            physicalFootprintBytes: physicalFootprintBytes,
            mlxActiveBytes: 120,
            mlxCacheBytes: mlxCacheBytes,
            mlxPeakBytes: 220),
    ])
}

private func admission() throws -> CompressedKVAttentionRuntimeAdmission {
    try CompressedKVAttentionRuntimeAdmission(
        family: .qwen3,
        modelType: "qwen3",
        architecture: "Qwen3ForCausalLM",
        modelConfigHash: "0123456789abcdef",
        modelConfigSHA256: digest("model-config"),
        checkpointManifestHash: "fedcba9876543210",
        checkpointContentSHA256: checkpointSHA,
        tokenizerSHA256: tokenizerSHA,
        layerCount: 64,
        queryHeadCount: 64,
        kvHeadCount: 8,
        headDimension: 128,
        maxPositionEmbeddings: 40_960,
        modelNativeDType: .bfloat16).validatedForEvidence()
}

private func exactPolicy() throws -> ExactPrefixCachePolicy {
    try ExactPrefixCachePolicy(
        maxEntries: 2,
        maxRetainedBytes: 512,
        minimumReusableTokens: 16)
}

private func templatePolicy() throws -> TemplateTokenCachePolicy {
    try TemplateTokenCachePolicy(maxEntries: 8, maxRetainedBytes: 512)
}

private func requestContext() throws -> ExactPrefixRequestContext {
    try ExactPrefixRequestContext(
        isolationNamespaceSHA256: digest("namespace"),
        promptTemplateSHA256: digest("template"),
        toolsSHA256: digest("tools"))
}

private func emptyPrefixSnapshot() -> ExactPrefixCacheSnapshot {
    prefixSnapshot(entryCount: 0)
}

private func prefixSnapshot(entryCount: Int) -> ExactPrefixCacheSnapshot {
    ExactPrefixCacheSnapshot(
        entryCount: entryCount,
        reservationCount: 0,
        retainedBytes: entryCount * 128,
        reservedBytes: 0,
        hitCount: 0,
        missCount: 0,
        evictionCount: 0)
}

private func digest(_ value: String) -> String {
    sha256Hex(Data(value.utf8))
}
