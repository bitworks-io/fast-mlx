import Foundation
import XCTest
@testable import HarnessCore

final class KVTunerCorpusIdentityTests: XCTestCase {
    private func sourceRows(_ label: String, count: Int) -> [String] {
        (0..<count).map {
            sha256Hex(Data("\(label)-source-row-\($0)".utf8))
        }.sorted()
    }

    // builtInMeasurementCorpus() (loaded spike/corpus/measurement-corpus-v2.json off disk) and the
    // three cases that used it — testOnlyExactBuiltInMeasurementCorpusReceivesAuditedSourceProvenance,
    // testFullContentSHARejectsCollisionStyleMeasurementIdentity, and
    // testCustomConstructionRequiresSourceRowsAndCannotDecodeAuditedAssertion — moved verbatim to
    // KVTunerAuditedCorpusProvenanceTests.swift, since the public projection cannot ship that asset.

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

    func testExactBuiltInTaskCorpusV2ReceivesAuditedSourceProvenance() throws {
        let corpus = try TaskCoherenceCorpusV2.make()
        let identity = try KVTunerEvaluationCorpusIdentity
            .taskCoherenceCorpus(corpus)

        XCTAssertEqual(identity.id, corpus.id)
        XCTAssertEqual(identity.aggregateDigest, corpus.contentHash)
        XCTAssertEqual(
            identity.sourceProvenance,
            .firstPartyAuditedNoGSM8K)
        XCTAssertEqual(
            identity.canonicalEntryDigests,
            corpus.items.map {
                KVTunerPromptDigest.exactText($0.prompt)
            }.sorted())
        XCTAssertEqual(
            identity.canonicalSourceItemDigests.count,
            corpus.items.count)
        XCTAssertEqual(
            try JSONDecoder().decode(
                KVTunerEvaluationCorpusIdentity.self,
                from: JSONEncoder().encode(identity)),
            identity)
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

    func testBenchWorkloadIdentityBindsExactSaltedPromptsAndAuditedSource() throws {
        let workload = try BenchWorkloadIdentity(
            basePrompt: defaultBenchPrompt,
            nonce: "kvarn-frontier-20260718",
            iterations: 4)
        let identity = try KVTunerEvaluationCorpusIdentity.benchWorkload(
            workload)

        XCTAssertEqual(identity.id, "fastmlx-bench-decode-v2")
        XCTAssertEqual(
            identity.canonicalEntryDigests,
            workload.prompts.map(KVTunerPromptDigest.exactText).sorted())
        XCTAssertEqual(
            try JSONDecoder().decode(
                KVTunerEvaluationCorpusIdentity.self,
                from: JSONEncoder().encode(identity)),
            identity)
    }

    func testBenchWorkloadIdentityAcceptsExactRepeatedAuditedPrompt() throws {
        let repeatedPrompt = Array(
            repeating: defaultBenchPrompt,
            count: 3
        ).joined(separator: "\n")
        let workload = try BenchWorkloadIdentity(
            basePrompt: repeatedPrompt,
            nonce: "kvarn-frontier-20260718",
            iterations: 2)

        let identity = try KVTunerEvaluationCorpusIdentity.benchWorkload(
            workload)

        XCTAssertEqual(
            identity.canonicalEntryDigests,
            workload.prompts.map(KVTunerPromptDigest.exactText).sorted())
        XCTAssertFalse(identity.canonicalSourceItemDigests.isEmpty)
    }

    func testBenchWorkloadIdentityRejectsAlteredRepeatedAuditedPrompt() throws {
        let unauditedPrompts = [
            defaultBenchPrompt + "\n" + defaultBenchPrompt + " altered",
            defaultBenchPrompt + "\n",
            defaultBenchPrompt + "\n\n" + defaultBenchPrompt,
        ]

        for basePrompt in unauditedPrompts {
            let workload = try BenchWorkloadIdentity(
                basePrompt: basePrompt,
                nonce: "kvarn-frontier-20260718",
                iterations: 2)

            XCTAssertThrowsError(
                try KVTunerEvaluationCorpusIdentity.benchWorkload(workload)
            ) { error in
                XCTAssertEqual(
                    error as? KVTunerEvaluationCorpusIdentityError,
                    .canonicalSourceItemsRequired)
            }
        }
    }

    func testBenchWorkloadIdentityRejectsRepeatAboveCLIBound() throws {
        let workload = try BenchWorkloadIdentity(
            basePrompt: Array(
                repeating: defaultBenchPrompt,
                count: 4_097
            ).joined(separator: "\n"),
            nonce: "kvarn-frontier-20260718",
            iterations: 2)

        XCTAssertThrowsError(
            try KVTunerEvaluationCorpusIdentity.benchWorkload(workload)
        ) { error in
            XCTAssertEqual(
                error as? KVTunerEvaluationCorpusIdentityError,
                .canonicalSourceItemsRequired)
        }
    }

    func testBenchWorkloadIdentityRejectsAnUnauditedCustomPrompt() throws {
        let workload = try BenchWorkloadIdentity(
            basePrompt: "A custom benchmark prompt",
            nonce: "kvarn-frontier-20260718",
            iterations: 4)

        XCTAssertThrowsError(
            try KVTunerEvaluationCorpusIdentity.benchWorkload(workload)
        ) { error in
            XCTAssertEqual(
                error as? KVTunerEvaluationCorpusIdentityError,
                .canonicalSourceItemsRequired)
        }
    }

    // testOnlyExactBuiltInMeasurementCorpusReceivesAuditedSourceProvenance,
    // testFullContentSHARejectsCollisionStyleMeasurementIdentity, and
    // testCustomConstructionRequiresSourceRowsAndCannotDecodeAuditedAssertion moved verbatim to
    // KVTunerAuditedCorpusProvenanceTests.swift (see note near sourceRows above).

    /// Documents and pins the refusal branch of `measurementCorpus(_:)` using only synthetic
    /// data — the admit branch (needs the real, unpublishable corpus bytes) is covered in
    /// KVTunerAuditedCorpusProvenanceTests.swift instead.
    ///
    /// LIMITATION (verified, not assumed): `measurementCorpus(_:)` gates admission on FOUR
    /// ANDed comparisons — corpusId, contentHash, entries.count, and a transcript SHA256 computed
    /// fresh from the corpus's exact entries (id/tag/text) via
    /// `KVTunerEvaluationCorpusAudit.measurementTranscriptSHA256(_:)`. That transcript hash
    /// necessarily also encodes corpusId and entries.count as transcript fields, so it is a
    /// strict superset of the other three checks: no synthetic (non-real) corpus can ever satisfy
    /// it, which means the transcript-mismatch alone already forces refusal for every case below
    /// regardless of whether the corpusId/contentHash/entries.count comparisons still exist in the
    /// source. A mutation that DELETES any one of those three redundant comparisons therefore
    /// cannot be caught by a synthetic-corpus test — doing so would require forging entries whose
    /// SHA256 transcript collides with the pinned literal, which is only possible with the real,
    /// unpublishable corpus content (or a SHA256 preimage break). This was confirmed empirically
    /// by temporarily deleting the `contentHash` comparison from `measurementCorpus(_:)`,
    /// rebuilding, and observing that every case below still passed (still threw) unchanged; the
    /// mutation was then reverted. See the task handoff report for that mutation run.
    /// What IS discriminating here: whether the refusal path exists and fires at all (e.g. a
    /// mutation that always admits, or that stops throwing `.canonicalSourceItemsRequired`,
    /// IS caught by every case below).
    func testMeasurementCorpusThrowsForEachIndependentlyConstructibleRefusalReason() throws {
        // Case 1: corpusId wrong; contentHash forged to the real pinned literal (public — it's
        // already a source-code literal, not private content) and entries.count == 5 to keep the
        // other two surface-level pins as close to "correct" as achievable without the private
        // corpus bytes.
        let wrongCorpusId = MeasurementCorpus(
            corpusId: "not-measurement-corpus-v2",
            entries: (0..<5).map {
                MeasurementCorpusEntry(
                    id: "synthetic-\($0)", tag: .prose, text: "synthetic entry \($0)")
            },
            contentHash: "8dd73ade100742f2")
        XCTAssertThrowsError(
            try KVTunerEvaluationCorpusIdentity.measurementCorpus(wrongCorpusId)
        ) { error in
            XCTAssertEqual(
                error as? KVTunerEvaluationCorpusIdentityError,
                .canonicalSourceItemsRequired)
        }

        // Case 2: corpusId and entries.count correct; contentHash forged to an obviously wrong
        // value (contentHash is a stored, forgeable field on MeasurementCorpus — see the public
        // initializer — so it can be set independently of corpusId/entries.count here).
        let wrongContentHash = MeasurementCorpus(
            corpusId: "measurement-corpus-v2",
            entries: (0..<5).map {
                MeasurementCorpusEntry(
                    id: "synthetic-\($0)", tag: .prose, text: "synthetic entry \($0)")
            },
            contentHash: "0000000000000000")
        XCTAssertThrowsError(
            try KVTunerEvaluationCorpusIdentity.measurementCorpus(wrongContentHash)
        ) { error in
            XCTAssertEqual(
                error as? KVTunerEvaluationCorpusIdentityError,
                .canonicalSourceItemsRequired)
        }

        // Case 3: corpusId and contentHash correct-looking; entries.count wrong (4, not 5).
        let wrongEntryCount = MeasurementCorpus(
            corpusId: "measurement-corpus-v2",
            entries: (0..<4).map {
                MeasurementCorpusEntry(
                    id: "synthetic-\($0)", tag: .prose, text: "synthetic entry \($0)")
            },
            contentHash: "8dd73ade100742f2")
        XCTAssertThrowsError(
            try KVTunerEvaluationCorpusIdentity.measurementCorpus(wrongEntryCount)
        ) { error in
            XCTAssertEqual(
                error as? KVTunerEvaluationCorpusIdentityError,
                .canonicalSourceItemsRequired)
        }

        // Case 4: an entirely empty/degenerate corpus — the coarsest possible refusal check,
        // included so a mutation that special-cases "empty" into acceptance is also caught.
        let empty = MeasurementCorpus(
            corpusId: "measurement-corpus-v2", entries: [], contentHash: "")
        XCTAssertThrowsError(
            try KVTunerEvaluationCorpusIdentity.measurementCorpus(empty)
        ) { error in
            XCTAssertEqual(
                error as? KVTunerEvaluationCorpusIdentityError,
                .canonicalSourceItemsRequired)
        }
    }
}
