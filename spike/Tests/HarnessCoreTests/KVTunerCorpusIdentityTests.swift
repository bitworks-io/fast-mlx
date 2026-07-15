import Foundation
import XCTest
@testable import HarnessCore

final class KVTunerCorpusIdentityTests: XCTestCase {
    private func sourceRows(_ label: String, count: Int) -> [String] {
        (0..<count).map {
            sha256Hex(Data("\(label)-source-row-\($0)".utf8))
        }.sorted()
    }

    private func builtInMeasurementCorpus() throws -> MeasurementCorpus {
        var directory = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
        for _ in 0..<8 {
            let candidate = directory.appendingPathComponent(
                "corpus/measurement-corpus-v2.json")
            if FileManager.default.fileExists(atPath: candidate.path) {
                return try MeasurementCorpusLoader.load(
                    from: Data(contentsOf: candidate))
            }
            directory.deleteLastPathComponent()
        }
        throw CocoaError(.fileNoSuchFile)
    }

    func testMeasurementIdentityHashesExactTextIndependentOfEntryID() throws {
        let originalEntries = [
            MeasurementCorpusEntry(
                id: "original-b", tag: .code, text: "let value = 41 + 1"),
            MeasurementCorpusEntry(
                id: "original-a", tag: .prose, text: "The exact prompt text."),
        ]
        let renamedEntries = [
            MeasurementCorpusEntry(
                id: "renamed-y", tag: .code, text: "let value = 41 + 1"),
            MeasurementCorpusEntry(
                id: "renamed-x", tag: .prose, text: "The exact prompt text."),
        ]
        let original = MeasurementCorpus(
            corpusId: "measurement-original-v1",
            entries: originalEntries,
            contentHash: MeasurementCorpusLoader.contentHash(
                entries: originalEntries))
        let renamed = MeasurementCorpus(
            corpusId: "measurement-renamed-v1",
            entries: renamedEntries,
            contentHash: MeasurementCorpusLoader.contentHash(
                entries: renamedEntries))

        let originalIdentity =
            try KVTunerEvaluationCorpusIdentity.measurementCorpus(
                original,
                canonicalSourceItemDigests: sourceRows(
                    "measurement-original", count: originalEntries.count))
        let renamedIdentity =
            try KVTunerEvaluationCorpusIdentity.measurementCorpus(
                renamed,
                canonicalSourceItemDigests: sourceRows(
                    "measurement-renamed", count: renamedEntries.count))

        XCTAssertEqual(originalIdentity.id, original.corpusId)
        XCTAssertEqual(originalIdentity.aggregateDigest, original.contentHash)
        XCTAssertNotEqual(
            originalIdentity.aggregateDigest, renamedIdentity.aggregateDigest)
        XCTAssertEqual(
            originalIdentity.canonicalEntryDigests,
            originalEntries.map { fnv1a64($0.text.utf8) }.sorted())
        XCTAssertEqual(
            originalIdentity.canonicalEntryDigests,
            renamedIdentity.canonicalEntryDigests,
            "renaming an entry must not hide exact prompt overlap")
        XCTAssertEqual(
            originalIdentity.sourceProvenance,
            .canonicalSourceItems)
        XCTAssertEqual(
            originalIdentity.canonicalSourceItemDigests.count,
            originalEntries.count)
        XCTAssertEqual(
            renamedIdentity.sourceProvenance,
            .canonicalSourceItems)
    }

    func testTaskIdentityHashesFullyExpandedPromptIndependentOfItemID() throws {
        let original = try TaskCoherenceCorpusV1.make()
        var renamedItems = original.items
        let item = renamedItems[0]
        renamedItems[0] = TaskCoherenceItem(
            id: "renamed-task-item",
            domain: item.domain,
            scoringMode: item.scoringMode,
            prefix: item.prefix,
            material: item.material,
            suffix: item.suffix,
            query: item.query,
            expectedChoice: item.expectedChoice,
            expectedTool: item.expectedTool)
        let renamed = try TaskCoherenceCorpus(
            schemaVersion: original.schemaVersion,
            id: "renamed-task-corpus-v1",
            items: renamedItems)

        let originalIdentity =
            try KVTunerEvaluationCorpusIdentity.taskCoherenceCorpus(original)
        let renamedIdentity =
            try KVTunerEvaluationCorpusIdentity.taskCoherenceCorpus(
                renamed,
                canonicalSourceItemDigests: sourceRows(
                    "renamed-task", count: renamed.items.count))

        XCTAssertEqual(originalIdentity.id, original.id)
        XCTAssertEqual(originalIdentity.aggregateDigest, original.contentHash)
        XCTAssertNotEqual(
            originalIdentity.aggregateDigest, renamedIdentity.aggregateDigest)
        XCTAssertEqual(
            originalIdentity.canonicalEntryDigests,
            original.items.map { fnv1a64($0.prompt.utf8) }.sorted())
        XCTAssertEqual(
            originalIdentity.canonicalEntryDigests,
            renamedIdentity.canonicalEntryDigests,
            "renaming an item or corpus must not hide exact prompt overlap")
        XCTAssertEqual(
            originalIdentity.sourceProvenance,
            .firstPartyAuditedNoGSM8K)
        XCTAssertEqual(
            originalIdentity.canonicalSourceItemDigests.count,
            original.items.count)
        XCTAssertEqual(
            try JSONDecoder().decode(
                KVTunerEvaluationCorpusIdentity.self,
                from: JSONEncoder().encode(originalIdentity)),
            originalIdentity)
        XCTAssertEqual(
            renamedIdentity.sourceProvenance,
            .canonicalSourceItems)
        XCTAssertEqual(
            renamedIdentity.canonicalSourceItemDigests.count,
            renamed.items.count)
    }

    func testOnlyExactBuiltInMeasurementCorpusReceivesAuditedSourceProvenance() throws {
        let corpus = try builtInMeasurementCorpus()
        let identity = try KVTunerEvaluationCorpusIdentity.measurementCorpus(
            corpus)
        XCTAssertEqual(
            identity.sourceProvenance,
            .firstPartyAuditedNoGSM8K)
        XCTAssertEqual(
            identity.canonicalSourceItemDigests.count,
            corpus.entries.count)
        XCTAssertEqual(
            try JSONDecoder().decode(
                KVTunerEvaluationCorpusIdentity.self,
                from: JSONEncoder().encode(identity)),
            identity)

        var alteredEntries = corpus.entries
        alteredEntries[0] = MeasurementCorpusEntry(
            id: alteredEntries[0].id,
            tag: alteredEntries[0].tag,
            text: "Question: rewrapped calibration source\nAnswer:")
        let forgedAggregate = MeasurementCorpus(
            corpusId: corpus.corpusId,
            entries: alteredEntries,
            contentHash: corpus.contentHash)
        XCTAssertThrowsError(
            try KVTunerEvaluationCorpusIdentity.measurementCorpus(
                forgedAggregate)
        ) { error in
            XCTAssertEqual(
                error as? KVTunerEvaluationCorpusIdentityError,
                .canonicalSourceItemsRequired)
        }

        let customIdentity = try KVTunerEvaluationCorpusIdentity
            .measurementCorpus(
                forgedAggregate,
                canonicalSourceItemDigests: sourceRows(
                    "rewrapped-measurement", count: alteredEntries.count))
        XCTAssertEqual(customIdentity.sourceProvenance, .canonicalSourceItems)
    }

    func testFullContentSHARejectsCollisionStyleMeasurementIdentity() throws {
        let corpus = try builtInMeasurementCorpus()
        var renamedEntries = corpus.entries
        let first = renamedEntries[0]
        renamedEntries[0] = MeasurementCorpusEntry(
            id: first.id + "-renamed",
            tag: first.tag,
            text: first.text)
        let forged = MeasurementCorpus(
            corpusId: corpus.corpusId,
            entries: renamedEntries,
            contentHash: corpus.contentHash)

        XCTAssertEqual(
            corpus.entries.map { KVTunerPromptDigest.exactText($0.text) }.sorted(),
            renamedEntries.map { KVTunerPromptDigest.exactText($0.text) }.sorted(),
            "the collision-style input preserves every old prompt fingerprint")
        XCTAssertThrowsError(
            try KVTunerEvaluationCorpusIdentity.measurementCorpus(forged)
        ) { error in
            XCTAssertEqual(
                error as? KVTunerEvaluationCorpusIdentityError,
                .canonicalSourceItemsRequired)
        }
    }

    func testFullContentSHAIncludesTaskScoringExpectations() throws {
        let corpus = try TaskCoherenceCorpusV1.make()
        var changedItems = corpus.items
        let index = try XCTUnwrap(changedItems.firstIndex {
            $0.scoringMode == .restrictedChoice
        })
        let original = changedItems[index]
        let replacement = ["A", "B", "C", "D"].first {
            $0 != original.expectedChoice
        }!
        changedItems[index] = TaskCoherenceItem(
            id: original.id,
            domain: original.domain,
            scoringMode: original.scoringMode,
            prefix: original.prefix,
            material: original.material,
            suffix: original.suffix,
            query: original.query,
            expectedChoice: replacement,
            expectedTool: nil)
        let changed = try TaskCoherenceCorpus(
            schemaVersion: corpus.schemaVersion,
            id: corpus.id,
            items: changedItems)

        XCTAssertEqual(
            corpus.items.map { KVTunerPromptDigest.exactText($0.prompt) }.sorted(),
            changed.items.map { KVTunerPromptDigest.exactText($0.prompt) }.sorted(),
            "changing the expected answer preserves every prompt fingerprint")
        XCTAssertThrowsError(
            try KVTunerEvaluationCorpusIdentity.taskCoherenceCorpus(changed)
        ) { error in
            XCTAssertEqual(
                error as? KVTunerEvaluationCorpusIdentityError,
                .canonicalSourceItemsRequired)
        }
    }

    func testIdentityAndRuntimePolicyAreCodable() throws {
        let identity = try KVTunerEvaluationCorpusIdentity(
            id: "evaluation-v1",
            aggregateDigest: "1111111111111111",
            canonicalEntryDigests: ["2222222222222222"],
            canonicalSourceItemDigests: sourceRows("evaluation", count: 1))
        let policy = KVTunerRuntimeLayerPolicy(
            layer: 0, keyBits: 8, valueBits: 4)

        XCTAssertEqual(
            try JSONDecoder().decode(
                KVTunerEvaluationCorpusIdentity.self,
                from: JSONEncoder().encode(identity)),
            identity)
        XCTAssertEqual(
            try JSONDecoder().decode(
                KVTunerRuntimeLayerPolicy.self,
                from: JSONEncoder().encode(policy)),
            policy)
    }

    func testCustomConstructionRequiresSourceRowsAndCannotDecodeAuditedAssertion() throws {
        XCTAssertThrowsError(try KVTunerEvaluationCorpusIdentity(
            id: "evaluation-v1",
            aggregateDigest: "1111111111111111",
            canonicalEntryDigests: ["2222222222222222"],
            canonicalSourceItemDigests: []
        )) { error in
            XCTAssertEqual(
                error as? KVTunerEvaluationCorpusIdentityError,
                .canonicalSourceItemsRequired)
        }

        let canonical = try KVTunerEvaluationCorpusIdentity(
            id: "evaluation-v1",
            aggregateDigest: "1111111111111111",
            canonicalEntryDigests: ["2222222222222222"],
            canonicalSourceItemDigests: sourceRows("evaluation", count: 1))
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: JSONEncoder().encode(canonical)) as? [String: Any])
        object["sourceProvenance"] =
            "first-party-audited-no-gsm8k-v1"
        object["canonicalSourceItemDigests"] =
            KVTunerEvaluationSourceProvenance.auditedSourceItemDigests(
                entryDigests: canonical.canonicalEntryDigests)

        XCTAssertThrowsError(try JSONDecoder().decode(
            KVTunerEvaluationCorpusIdentity.self,
            from: JSONSerialization.data(withJSONObject: object)))

        let builtIn = try KVTunerEvaluationCorpusIdentity.measurementCorpus(
            builtInMeasurementCorpus())
        var modifiedAudited = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: JSONEncoder().encode(builtIn)) as? [String: Any])
        modifiedAudited["id"] = "custom-collision-style-identity"
        XCTAssertThrowsError(try JSONDecoder().decode(
            KVTunerEvaluationCorpusIdentity.self,
            from: JSONSerialization.data(withJSONObject: modifiedAudited)))

        let replacementEntryDigest = "3333333333333333"
        modifiedAudited = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: JSONEncoder().encode(builtIn)) as? [String: Any])
        modifiedAudited["canonicalEntryDigests"] = [
            replacementEntryDigest
        ]
        modifiedAudited["canonicalSourceItemDigests"] =
            KVTunerEvaluationSourceProvenance.auditedSourceItemDigests(
                entryDigests: [replacementEntryDigest])
        XCTAssertThrowsError(try JSONDecoder().decode(
            KVTunerEvaluationCorpusIdentity.self,
            from: JSONSerialization.data(withJSONObject: modifiedAudited)))
    }
}
