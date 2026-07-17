import XCTest
@testable import HarnessCore

final class TaskCoherenceTests: XCTestCase {
    func testFrozenCorpusHasTwentyUniqueItemsPerDomain() throws {
        let corpus = try TaskCoherenceCorpusV1.make()

        XCTAssertEqual(corpus.schemaVersion, 1)
        XCTAssertEqual(corpus.id, "kvarn-task-coherence-v1")
        XCTAssertEqual(corpus.items.count, 80)
        XCTAssertEqual(Set(corpus.items.map(\.id)).count, 80)
        XCTAssertEqual(corpus.contentHash.count, 16)
        XCTAssertEqual(corpus.contentHash, "0f16e3abc00ec7c8")
        for domain in TaskCoherenceDomain.allCases {
            XCTAssertEqual(
                corpus.items.filter { $0.domain == domain }.count,
                TaskCoherenceCorpus.requiredItemsPerDomain)
        }
        for item in corpus.items {
            XCTAssertFalse(item.prefix.isEmpty)
            XCTAssertFalse(item.material.isEmpty)
            XCTAssertFalse(item.suffix.isEmpty)
            XCTAssertFalse(item.query.isEmpty)
            XCTAssertEqual(
                item.prompt,
                item.prefix + item.material + item.suffix + item.query)
        }
        XCTAssertEqual(try TaskCoherenceCorpusV1.make(), corpus)
    }

    func testFrozenCorpusV2PreservesControlsAndMakesLongRetrievalExplicit() throws {
        let v1 = try TaskCoherenceCorpusV1.make()
        let v2 = try TaskCoherenceCorpusV2.make()

        XCTAssertEqual(v1.contentHash, "0f16e3abc00ec7c8")
        XCTAssertEqual(v2.schemaVersion, 1)
        XCTAssertEqual(v2.id, "kvarn-task-coherence-v2")
        XCTAssertEqual(v2.contentHash, "1740d0d07f586def")
        XCTAssertEqual(v2.items.count, 80)
        XCTAssertEqual(
            v2.items.filter { $0.domain != .longRetrieval },
            v1.items.filter { $0.domain != .longRetrieval })

        let longRetrieval = v2.items.filter {
            $0.domain == .longRetrieval
        }
        XCTAssertEqual(longRetrieval.count, 20)
        XCTAssertEqual(
            Dictionary(grouping: longRetrieval, by: \.expectedChoice)
                .mapValues(\.count),
            ["A": 5, "B": 5, "C": 5, "D": 5])

        for (index, item) in longRetrieval.enumerated() {
            let archiveID = String(format: "ARCHIVE-%02d", index)
            let expected = try XCTUnwrap(item.expectedChoice)
            XCTAssertTrue(item.material.contains(
                "The correct option label for \(archiveID) is \(expected)."))
            XCTAssertEqual(
                item.query,
                "According to the earlier correct-option-label statement for \(archiveID), copy that single label. Answer:")
        }
        XCTAssertEqual(try TaskCoherenceCorpusV2.make(), v2)
    }

    func testCorpusRejectsDuplicateIDsAndMismatchedScoringContracts() throws {
        let valid = try TaskCoherenceCorpusV1.make()
        var duplicate = valid.items
        duplicate[1] = duplicate[0]
        XCTAssertThrowsError(try TaskCoherenceCorpus(
            schemaVersion: 1, id: valid.id, items: duplicate
        )) {
            XCTAssertEqual(
                $0 as? TaskCoherenceError,
                .duplicateItemID(duplicate[0].id))
        }

        var malformed = valid.items
        let source = malformed[0]
        malformed[0] = TaskCoherenceItem(
            id: source.id, domain: source.domain,
            scoringMode: .structuredTool,
            prefix: source.prefix, material: source.material,
            suffix: source.suffix, query: source.query,
            expectedChoice: source.expectedChoice,
            expectedTool: nil)
        XCTAssertThrowsError(try TaskCoherenceCorpus(
            schemaVersion: 1, id: valid.id, items: malformed
        )) {
            XCTAssertEqual(
                $0 as? TaskCoherenceError,
                .invalidItem(source.id))
        }
    }

    func testCorpusDecodingRevalidatesContentHashAndItems() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let encoded = try JSONEncoder().encode(corpus)
        var object = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: encoded) as? [String: Any])

        object["contentHash"] = "0000000000000000"
        let forgedHash = try JSONSerialization.data(withJSONObject: object)
        XCTAssertThrowsError(
            try JSONDecoder().decode(TaskCoherenceCorpus.self, from: forgedHash))

        object = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        var items = try XCTUnwrap(object["items"] as? [[String: Any]])
        items[1] = items[0]
        object["items"] = items
        let duplicate = try JSONSerialization.data(withJSONObject: object)
        XCTAssertThrowsError(
            try JSONDecoder().decode(TaskCoherenceCorpus.self, from: duplicate))
    }

    func testRestrictedChoiceScorerUsesOnlyFourPinnedTokenIDs() throws {
        XCTAssertEqual(TaskRestrictedChoiceScorer.labelTokenSpellings, [
            "A": " A", "B": " B", "C": " C", "D": " D",
        ])
        let logits: [Float] = [100, 0.2, 0.9, 0.7, 0.1]
        let labels = ["A": 1, "B": 2, "C": 3, "D": 4]

        XCTAssertEqual(
            try TaskRestrictedChoiceScorer.predict(
                logits: logits, labelTokenIDs: labels),
            "B")

        XCTAssertThrowsError(try TaskRestrictedChoiceScorer.predict(
            logits: logits, labelTokenIDs: ["A": 1, "B": 1, "C": 3, "D": 4]
        )) {
            XCTAssertEqual(
                $0 as? TaskCoherenceError,
                .invalidChoiceTokenIDs)
        }
        XCTAssertThrowsError(try TaskRestrictedChoiceScorer.predict(
            logits: [0, .nan, 1, 2, 3], labelTokenIDs: labels
        )) {
            XCTAssertEqual(
                $0 as? TaskCoherenceError,
                .nonFiniteChoiceLogit("A"))
        }
    }

    func testStructuredToolScorerExtractsWrappedJSONAndRequiresExactSemantics() {
        let expected = TaskToolExpectation(
            name: "lookup_record", arguments: ["record": "R-17"])
        let correct = TaskStructuredToolScorer.score(
            "<think>done</think>\n<tool_call>{\"name\":\"lookup_record\",\"arguments\":{\"record\":\"R-17\"}}</tool_call>",
            expected: expected)
        XCTAssertTrue(correct.syntacticallyValid)
        XCTAssertTrue(correct.correct)
        XCTAssertEqual(correct.toolName, "lookup_record")

        let wrong = TaskStructuredToolScorer.score(
            "prefix {\"name\":\"lookup_record\",\"arguments\":{\"record\":\"R-18\"}} suffix",
            expected: expected)
        XCTAssertTrue(wrong.syntacticallyValid)
        XCTAssertFalse(wrong.correct)

        let malformed = TaskStructuredToolScorer.score(
            "{\"name\":\"lookup_record\",", expected: expected)
        XCTAssertFalse(malformed.syntacticallyValid)
        XCTAssertFalse(malformed.correct)
    }

    func testStructuredToolScorerRejectsDuplicateOrAmbiguousJSON() {
        let expected = TaskToolExpectation(
            name: "lookup_record", arguments: ["record": "R-17"])

        let duplicateRootKey = TaskStructuredToolScorer.score(
            "{\"name\":\"wrong\",\"name\":\"lookup_record\",\"arguments\":{\"record\":\"R-17\"}}",
            expected: expected)
        XCTAssertFalse(duplicateRootKey.syntacticallyValid)
        XCTAssertFalse(duplicateRootKey.correct)

        let ambiguous = TaskStructuredToolScorer.score(
            "{\"name\":\"lookup_record\",\"arguments\":{\"record\":\"R-17\"}} {\"name\":\"lookup_record\",\"arguments\":{\"record\":\"R-17\"}}",
            expected: expected)
        XCTAssertFalse(ambiguous.syntacticallyValid)
        XCTAssertFalse(ambiguous.correct)
    }

    func testHardFloorPreservesAggressiveUserChoiceAboveChance() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let reference = scores(correctByDomain: [
            .math: 20, .code: 20, .structuredTool: 20,
            .longRetrieval: 20,
        ], structuredValid: 20)
        let aggressive = scores(correctByDomain: [
            .math: 9, .code: 10, .structuredTool: 8,
            .longRetrieval: 9,
        ], structuredValid: 18)

        let assessment = try TaskCoherenceAssessment.derive(
            candidate: run(aggressive, tier: "kvarn-k4v2-g128", corpus: corpus),
            reference: run(reference, tier: "fp16", corpus: corpus),
            corpus: corpus)

        XCTAssertTrue(assessment.referenceBaselinePassed)
        XCTAssertTrue(assessment.hardFloorPassed)
        XCTAssertFalse(assessment.balancedTaskDeltaPassed)
        XCTAssertEqual(assessment.structuredValidity.rate, 0.9, accuracy: 0)
        XCTAssertTrue(assessment.structuredValidity.passed)
        XCTAssertTrue(assessment.domains.allSatisfy(\.hardFloorPassed))
    }

    func testHardFloorRejectsDomainBelowChanceAndHalfReference() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let reference = scores(correctByDomain: [
            .math: 20, .code: 20, .structuredTool: 20,
            .longRetrieval: 20,
        ], structuredValid: 20)
        let collapsed = scores(correctByDomain: [
            .math: 4, .code: 20, .structuredTool: 20,
            .longRetrieval: 20,
        ], structuredValid: 20)

        let assessment = try TaskCoherenceAssessment.derive(
            candidate: run(collapsed, tier: "kvarn-k4v2-g128", corpus: corpus),
            reference: run(reference, tier: "fp16", corpus: corpus),
            corpus: corpus)
        let math = try XCTUnwrap(
            assessment.domains.first { $0.domain == .math })

        XCTAssertEqual(math.score, 0.2, accuracy: 0)
        XCTAssertFalse(math.hardFloorPassed)
        XCTAssertFalse(assessment.hardFloorPassed)
    }

    func testReferenceAtChanceCannotAdjudicateCandidate() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let weakReference = scores(correctByDomain: [
            .math: 5, .code: 20, .structuredTool: 20,
            .longRetrieval: 20,
        ], structuredValid: 20)

        XCTAssertThrowsError(try TaskCoherenceAssessment.derive(
            candidate: run(
                weakReference, tier: "kvarn-k4v2-g128", corpus: corpus),
            reference: run(weakReference, tier: "fp16", corpus: corpus),
            corpus: corpus
        )) {
            XCTAssertEqual(
                $0 as? TaskCoherenceError,
                .invalidReferenceBaseline(.math))
        }
    }

    func testReferenceStructuredSyntaxBelowNinetyPercentCannotAdjudicateCandidate() throws {
        let corpus = try TaskCoherenceCorpusV2.make()
        let invalidReference = scores(correctByDomain: [
            .math: 16, .code: 10, .structuredTool: 16,
            .longRetrieval: 20,
        ], structuredValid: 16)

        XCTAssertThrowsError(try TaskCoherenceAssessment.derive(
            candidate: run(
                invalidReference, tier: "kvarn-k4v2-g128",
                corpus: corpus),
            reference: run(
                invalidReference, tier: "fp16", corpus: corpus),
            corpus: corpus
        )) {
            XCTAssertEqual(
                $0 as? TaskCoherenceError,
                .invalidReferenceStructuredValidity)
        }
    }

    func testAssessmentRequiresSameCorpusModelAndFP16Reference() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let complete = scores(correctByDomain: [
            .math: 20, .code: 20, .structuredTool: 20,
            .longRetrieval: 20,
        ], structuredValid: 20)
        let candidate = run(
            complete, tier: "kvarn-k4v2-g128", corpus: corpus)

        let wrongTier = TaskCoherenceScoredRun(
            identity: TaskCoherenceRunIdentity(
                corpusID: corpus.id,
                corpusContentHash: corpus.contentHash,
                modelConfigHash: "model-config",
                modelCheckpointManifestHash: "checkpoint-manifest",
                kvQuantTier: "affine-k8v2-g128"),
            scores: complete)
        XCTAssertThrowsError(try TaskCoherenceAssessment.derive(
            candidate: candidate, reference: wrongTier, corpus: corpus
        )) {
            XCTAssertEqual(
                $0 as? TaskCoherenceError,
                .referenceMustBeFP16)
        }

        let wrongModel = TaskCoherenceScoredRun(
            identity: TaskCoherenceRunIdentity(
                corpusID: corpus.id,
                corpusContentHash: corpus.contentHash,
                modelConfigHash: "different-model",
                modelCheckpointManifestHash: "checkpoint-manifest",
                kvQuantTier: "fp16"),
            scores: complete)
        XCTAssertThrowsError(try TaskCoherenceAssessment.derive(
            candidate: candidate, reference: wrongModel, corpus: corpus
        )) {
            XCTAssertEqual(
                $0 as? TaskCoherenceError,
                .mismatchedRunIdentity)
        }
    }

    func testAssessmentRejectsCorrectButSyntacticallyInvalidToolRow() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        let reference = scores(correctByDomain: [
            .math: 20, .code: 20, .structuredTool: 20,
            .longRetrieval: 20,
        ], structuredValid: 20)
        var inconsistent = reference
        let index = try XCTUnwrap(
            inconsistent.firstIndex { $0.domain == .structuredTool })
        let row = inconsistent[index]
        inconsistent[index] = TaskItemScore(
            itemID: row.itemID,
            domain: row.domain,
            correct: true,
            syntacticallyValid: false)

        XCTAssertThrowsError(try TaskCoherenceAssessment.derive(
            candidate: run(
                inconsistent, tier: "kvarn-k4v2-g128", corpus: corpus),
            reference: run(reference, tier: "fp16", corpus: corpus),
            corpus: corpus
        )) {
            XCTAssertEqual(
                $0 as? TaskCoherenceError,
                .invalidScoreSet("candidate"))
        }
    }

    private func scores(
        correctByDomain: [TaskCoherenceDomain: Int],
        structuredValid: Int
    ) -> [TaskItemScore] {
        let corpus = try! TaskCoherenceCorpusV1.make()
        return TaskCoherenceDomain.allCases.flatMap { domain in
            corpus.items.filter { $0.domain == domain }.enumerated().map {
                index, item in
                TaskItemScore(
                    itemID: item.id,
                    domain: domain,
                    correct: index < (correctByDomain[domain] ?? 0),
                    syntacticallyValid: domain == .structuredTool
                        ? index < structuredValid : nil)
            }
        }
    }

    private func run(
        _ scores: [TaskItemScore],
        tier: String,
        corpus: TaskCoherenceCorpus
    ) -> TaskCoherenceScoredRun {
        TaskCoherenceScoredRun(
            identity: TaskCoherenceRunIdentity(
                corpusID: corpus.id,
                corpusContentHash: corpus.contentHash,
                modelConfigHash: "model-config",
                modelCheckpointManifestHash: "checkpoint-manifest",
                kvQuantTier: tier),
            scores: scores)
    }
}
