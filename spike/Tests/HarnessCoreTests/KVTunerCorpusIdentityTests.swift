import Foundation
import XCTest
@testable import HarnessCore

final class KVTunerCorpusIdentityTests: XCTestCase {
    func testMeasurementIdentityHashesExactTextIndependentOfEntryID() {
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
            KVTunerEvaluationCorpusIdentity.measurementCorpus(original)
        let renamedIdentity =
            KVTunerEvaluationCorpusIdentity.measurementCorpus(renamed)

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
            KVTunerEvaluationCorpusIdentity.taskCoherenceCorpus(original)
        let renamedIdentity =
            KVTunerEvaluationCorpusIdentity.taskCoherenceCorpus(renamed)

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
    }

    func testIdentityAndRuntimePolicyAreCodable() throws {
        let identity = KVTunerEvaluationCorpusIdentity(
            id: "evaluation-v1",
            aggregateDigest: "1111111111111111",
            canonicalEntryDigests: ["2222222222222222"])
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
}
