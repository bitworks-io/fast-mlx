import XCTest
@testable import HarnessCore

final class ExactPrefixCacheTests: XCTestCase {
    private func digest(_ character: Character) -> String {
        String(repeating: String(character), count: 64)
    }

    private func key(
        namespace: Character = "a",
        modelInstance: String = "model-instance-1",
        revision: Character = "b",
        tokenizer: Character = "c",
        template: Character = "d",
        tools: Character = "e",
        kvRoute: Character = "f",
        position: Character = "0",
        architecture: Character = "1",
        drafter: Character = "2"
    ) throws -> ExactPrefixSemanticKey {
        try ExactPrefixSemanticKey(
            isolationNamespaceSHA256: digest(namespace),
            modelInstanceID: modelInstance,
            modelRevisionSHA256: digest(revision),
            tokenizerSHA256: digest(tokenizer),
            promptTemplateSHA256: digest(template),
            toolsSHA256: digest(tools),
            kvRouteSHA256: digest(kvRoute),
            positionSemanticsSHA256: digest(position),
            architectureStateSHA256: digest(architecture),
            drafterStateSHA256: digest(drafter))
    }

    private func policy(
        entries: Int = 4,
        bytes: Int = 64 * 1024,
        minimumTokens: Int = 2
    ) throws -> ExactPrefixCachePolicy {
        try ExactPrefixCachePolicy(
            maxEntries: entries,
            maxRetainedBytes: bytes,
            minimumReusableTokens: minimumTokens)
    }

    private func commit(
        _ state: String,
        key: ExactPrefixSemanticKey,
        tokens: [Int],
        snapshotBytes: Int = 64,
        into cache: inout ExactPrefixCache<String>
    ) throws -> ExactPrefixCommitDecision {
        let reservation = try XCTUnwrap(
            try cache.reserve(
                key: key,
                tokens: tokens,
                snapshotBytes: snapshotBytes
            ).reservation)
        return try cache.commit(
            reservation,
            state: state,
            actualSnapshotBytes: snapshotBytes,
            disposition: .successfulText(
                generatedTokenCount: 2,
                visibleTokenCount: 2))
    }

    func testSemanticKeyRejectsMalformedOrPathLikeIdentity() throws {
        XCTAssertThrowsError(
            try ExactPrefixSemanticKey(
                isolationNamespaceSHA256: "not-a-digest",
                modelInstanceID: "model-instance-1",
                modelRevisionSHA256: digest("b"),
                tokenizerSHA256: digest("c"),
                promptTemplateSHA256: digest("d"),
                toolsSHA256: digest("e"),
                kvRouteSHA256: digest("f"),
                positionSemanticsSHA256: digest("0"),
                architectureStateSHA256: digest("1"),
                drafterStateSHA256: digest("2"))
        ) { error in
            XCTAssertEqual(
                error as? ExactPrefixCacheError,
                .invalidSemanticIdentity("isolationNamespaceSHA256"))
        }

        XCTAssertThrowsError(try key(modelInstance: "/Users/operator/model")) {
            error in
            XCTAssertEqual(
                error as? ExactPrefixCacheError,
                .invalidSemanticIdentity("modelInstanceID"))
        }

        let encoded = try JSONEncoder().encode(key())
        var decodedObject = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoded)
                as? [String: Any])
        decodedObject["modelInstanceID"] = "/decoded/path"
        let invalidEncoded = try JSONSerialization.data(
            withJSONObject: decodedObject)
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                ExactPrefixSemanticKey.self,
                from: invalidEncoded))
    }

    func testSemanticKeyIncludesEveryIsolationAxis() throws {
        let baseline = try key()
        let variants = [
            try key(namespace: "3"),
            try key(modelInstance: "model-instance-2"),
            try key(revision: "3"),
            try key(tokenizer: "3"),
            try key(template: "3"),
            try key(tools: "3"),
            try key(kvRoute: "3"),
            try key(position: "3"),
            try key(architecture: "3"),
            try key(drafter: "3"),
        ]

        XCTAssertEqual(Set(variants).count, variants.count)
        XCTAssertTrue(variants.allSatisfy { $0 != baseline })
    }

    func testLongestPrefixWinsAndNamespaceNeverCrosses() throws {
        let keyA = try key(namespace: "a")
        let keyB = try key(namespace: "b")
        var cache = ExactPrefixCache<String>(policy: try policy())

        _ = try commit("short", key: keyA, tokens: [10, 20], into: &cache)
        _ = try commit("long", key: keyA, tokens: [10, 20, 30], into: &cache)
        _ = try commit("other-branch", key: keyA, tokens: [10, 99, 30], into: &cache)

        let hit = try XCTUnwrap(
            try cache.lookup(key: keyA, promptTokens: [10, 20, 30, 40]))
        XCTAssertEqual(hit.state, "long")
        XCTAssertEqual(hit.prefixTokenCount, 3)
        XCTAssertNil(
            try cache.lookup(key: keyB, promptTokens: [10, 20, 30, 40]))
    }

    func testInvalidatedHitIsRemovedAndNextLookupIsAColdMiss() throws {
        let semanticKey = try key()
        var cache = ExactPrefixCache<String>(policy: try policy())
        let committed = try commit(
            "corrupt-snapshot",
            key: semanticKey,
            tokens: [10, 20, 30],
            into: &cache)
        let entryID = try XCTUnwrap(committed.entryID)

        let hit = try XCTUnwrap(
            try cache.lookup(
                key: semanticKey,
                promptTokens: [10, 20, 30, 40]))
        XCTAssertEqual(hit.entryID, entryID)

        XCTAssertTrue(cache.invalidate(entryID: hit.entryID))
        XCTAssertFalse(cache.invalidate(entryID: hit.entryID))
        XCTAssertNil(
            try cache.lookup(
                key: semanticKey,
                promptTokens: [10, 20, 30, 40]))
        XCTAssertEqual(cache.snapshot.entryCount, 0)
        XCTAssertEqual(cache.snapshot.evictionCount, 1)
        XCTAssertEqual(cache.snapshot.missCount, 1)
    }

    func testReservationIsInvisibleUntilPositiveCommit() throws {
        let semanticKey = try key()
        var cache = ExactPrefixCache<String>(policy: try policy())
        let reservation = try XCTUnwrap(
            try cache.reserve(
                key: semanticKey,
                tokens: [1, 2, 3],
                snapshotBytes: 64
            ).reservation)

        XCTAssertNil(
            try cache.lookup(
                key: semanticKey,
                promptTokens: [1, 2, 3, 4]))
        XCTAssertEqual(cache.snapshot.reservationCount, 1)

        let decision = try cache.commit(
            reservation,
            state: "must-not-publish",
            actualSnapshotBytes: 64,
            disposition: .failed)
        XCTAssertEqual(decision.skipReason, .generationFailed)
        XCTAssertNil(decision.entryID)
        XCTAssertEqual(cache.snapshot.reservationCount, 0)
        XCTAssertNil(
            try cache.lookup(
                key: semanticKey,
                promptTokens: [1, 2, 3, 4]))
    }

    func testOnlyNonEmptyVisibleTextSuccessCommits() throws {
        let semanticKey = try key()
        let rejected: [
            (ExactPrefixCommitDisposition, ExactPrefixCommitSkipReason)
        ] = [
            (.failed, .generationFailed),
            (.cancelled, .cancelled),
            (.media, .unsupportedMedia),
            (
                .successfulText(
                    generatedTokenCount: 0,
                    visibleTokenCount: 0),
                .zeroGeneratedTokens
            ),
            (
                .successfulText(
                    generatedTokenCount: 2,
                    visibleTokenCount: 0),
                .padOnlyOutput
            ),
        ]

        for (ordinal, pair) in rejected.enumerated() {
            var cache = ExactPrefixCache<String>(policy: try policy())
            let reservation = try XCTUnwrap(
                try cache.reserve(
                    key: semanticKey,
                    tokens: [1, 2, 3],
                    snapshotBytes: 64
                ).reservation)
            let decision = try cache.commit(
                reservation,
                state: "rejected-\(ordinal)",
                actualSnapshotBytes: 64,
                disposition: pair.0)
            XCTAssertEqual(decision.skipReason, pair.1)
            XCTAssertNil(decision.entryID)
            XCTAssertEqual(cache.snapshot.entryCount, 0)
        }

        var cache = ExactPrefixCache<String>(policy: try policy())
        let committed = try commit(
            "published",
            key: semanticKey,
            tokens: [1, 2, 3],
            into: &cache)
        XCTAssertNotNil(committed.entryID)
        XCTAssertNil(committed.skipReason)
        XCTAssertEqual(
            try cache.lookup(
                key: semanticKey,
                promptTokens: [1, 2, 3, 4])?.state,
            "published")
    }

    func testEntryLimitEvictsLeastRecentlyUsedEntry() throws {
        let semanticKey = try key()
        var cache = ExactPrefixCache<String>(
            policy: try policy(entries: 2, bytes: 64 * 1024))

        _ = try commit("one", key: semanticKey, tokens: [1, 1], into: &cache)
        _ = try commit("two", key: semanticKey, tokens: [2, 2], into: &cache)
        XCTAssertEqual(
            try cache.lookup(key: semanticKey, promptTokens: [1, 1, 9])?.state,
            "one")

        let third = try commit(
            "three",
            key: semanticKey,
            tokens: [3, 3],
            into: &cache)
        XCTAssertEqual(third.evictedEntryIDs.count, 1)
        XCTAssertNotNil(
            try cache.lookup(key: semanticKey, promptTokens: [1, 1, 9]))
        XCTAssertNil(
            try cache.lookup(key: semanticKey, promptTokens: [2, 2, 9]))
        XCTAssertNotNil(
            try cache.lookup(key: semanticKey, promptTokens: [3, 3, 9]))
    }

    func testByteLimitEvictsBeforeReservationAndTracksActualBytes() throws {
        let semanticKey = try key()
        let oneEntryBytes = try ExactPrefixCache<String>.retainedBytes(
            key: semanticKey,
            tokens: [1, 2],
            snapshotBytes: 128)
        var cache = ExactPrefixCache<String>(
            policy: try policy(
                entries: 4,
                bytes: oneEntryBytes * 2,
                minimumTokens: 2))

        _ = try commit(
            "one",
            key: semanticKey,
            tokens: [1, 2],
            snapshotBytes: 128,
            into: &cache)
        _ = try commit(
            "two",
            key: semanticKey,
            tokens: [3, 4],
            snapshotBytes: 128,
            into: &cache)

        let decision = try cache.reserve(
            key: semanticKey,
            tokens: [5, 6],
            snapshotBytes: 128)
        XCTAssertNotNil(decision.reservation)
        XCTAssertEqual(decision.evictedEntryIDs.count, 1)
        XCTAssertLessThanOrEqual(
            cache.snapshot.retainedBytes + cache.snapshot.reservedBytes,
            oneEntryBytes * 2)

        let committed = try cache.commit(
            XCTUnwrap(decision.reservation),
            state: "three",
            actualSnapshotBytes: 96,
            disposition: .successfulText(
                generatedTokenCount: 1,
                visibleTokenCount: 1))
        XCTAssertNotNil(committed.entryID)
        XCTAssertEqual(cache.snapshot.reservedBytes, 0)
        XCTAssertLessThan(cache.snapshot.retainedBytes, oneEntryBytes * 2)
    }

    func testOversizedAndTooShortCandidatesSkipWithoutEviction() throws {
        let semanticKey = try key()
        let retained = try ExactPrefixCache<String>.retainedBytes(
            key: semanticKey,
            tokens: [1, 2],
            snapshotBytes: 64)
        var cache = ExactPrefixCache<String>(
            policy: try policy(entries: 2, bytes: retained))
        _ = try commit(
            "existing",
            key: semanticKey,
            tokens: [1, 2],
            snapshotBytes: 64,
            into: &cache)

        let tooShort = try cache.reserve(
            key: semanticKey,
            tokens: [9],
            snapshotBytes: 1)
        XCTAssertEqual(tooShort.skipReason, .prefixTooShort)
        XCTAssertTrue(tooShort.evictedEntryIDs.isEmpty)

        let oversized = try cache.reserve(
            key: semanticKey,
            tokens: [9, 9],
            snapshotBytes: retained + 1)
        XCTAssertEqual(oversized.skipReason, .snapshotExceedsBudget)
        XCTAssertTrue(oversized.evictedEntryIDs.isEmpty)
        XCTAssertEqual(cache.snapshot.entryCount, 1)
    }

    func testRollbackAndFailedActualByteValidationReleaseReservation() throws {
        let semanticKey = try key()
        var cache = ExactPrefixCache<String>(policy: try policy())

        let rolledBack = try XCTUnwrap(
            try cache.reserve(
                key: semanticKey,
                tokens: [1, 2],
                snapshotBytes: 64
            ).reservation)
        try cache.rollback(rolledBack)
        XCTAssertEqual(cache.snapshot.reservationCount, 0)
        XCTAssertEqual(cache.snapshot.reservedBytes, 0)

        let tooSmall = try XCTUnwrap(
            try cache.reserve(
                key: semanticKey,
                tokens: [3, 4],
                snapshotBytes: 64
            ).reservation)
        XCTAssertThrowsError(
            try cache.commit(
                tooSmall,
                state: "oversized-actual",
                actualSnapshotBytes: 65,
                disposition: .successfulText(
                    generatedTokenCount: 1,
                    visibleTokenCount: 1))
        ) { error in
            XCTAssertEqual(
                error as? ExactPrefixCacheError,
                .actualSnapshotExceedsReservation(
                    reserved: 64,
                    actual: 65))
        }
        XCTAssertEqual(cache.snapshot.reservationCount, 0)
        XCTAssertEqual(cache.snapshot.entryCount, 0)
    }

    func testDisabledPolicyAndInvalidInputsFailClosed() throws {
        var disabled = ExactPrefixCache<String>(policy: .disabled)
        let disabledDecision = try disabled.reserve(
            key: key(),
            tokens: [1, 2],
            snapshotBytes: 64)
        XCTAssertEqual(disabledDecision.skipReason, .disabled)
        XCTAssertNil(disabledDecision.reservation)

        XCTAssertThrowsError(
            try ExactPrefixCachePolicy(
                maxEntries: 0,
                maxRetainedBytes: 1,
                minimumReusableTokens: 1)
        )
        XCTAssertThrowsError(
            try ExactPrefixCachePolicy(
                maxEntries: 1,
                maxRetainedBytes: 0,
                minimumReusableTokens: 1)
        )
        XCTAssertThrowsError(
            try ExactPrefixCachePolicy(
                maxEntries: 1,
                maxRetainedBytes: 1,
                minimumReusableTokens: 0)
        )

        var enabled = ExactPrefixCache<String>(policy: try policy())
        XCTAssertThrowsError(
            try enabled.reserve(
                key: key(),
                tokens: [Int.max, 2],
                snapshotBytes: 64)
        ) { error in
            XCTAssertEqual(
                error as? ExactPrefixCacheError,
                .tokenOutOfInt32Range(position: 0))
        }
        XCTAssertThrowsError(
            try enabled.reserve(
                key: key(),
                tokens: [-1, 2],
                snapshotBytes: 64)
        ) { error in
            XCTAssertEqual(
                error as? ExactPrefixCacheError,
                .negativeTokenID(position: 0))
        }
        XCTAssertThrowsError(
            try enabled.reserve(
                key: key(),
                tokens: [1, 2],
                snapshotBytes: Int.max)
        ) { error in
            XCTAssertEqual(
                error as? ExactPrefixCacheError,
                .retainedByteCountOverflow)
        }

        let invalidPolicy = Data(
            """
            {"isEnabled":true,"maxEntries":0,"maxRetainedBytes":1,\
            "minimumReusableTokens":1}
            """.utf8)
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                ExactPrefixCachePolicy.self,
                from: invalidPolicy))
    }

    func testCombinedByteAccountingEvictsWithoutOverflow() throws {
        let semanticKey = try key()
        let minimumRetained = try ExactPrefixCache<String>.retainedBytes(
            key: semanticKey,
            tokens: [1, 2],
            snapshotBytes: 1)
        let snapshotBytes = Int.max - (minimumRetained - 1)
        var cache = ExactPrefixCache<String>(
            policy: try policy(
                entries: 2,
                bytes: Int.max))

        _ = try commit(
            "near-limit",
            key: semanticKey,
            tokens: [1, 2],
            snapshotBytes: snapshotBytes,
            into: &cache)

        let decision = try cache.reserve(
            key: semanticKey,
            tokens: [3, 4],
            snapshotBytes: 1)

        XCTAssertNotNil(decision.reservation)
        XCTAssertEqual(decision.evictedEntryIDs.count, 1)
        XCTAssertEqual(cache.snapshot.entryCount, 0)
        XCTAssertEqual(cache.snapshot.reservationCount, 1)
        XCTAssertLessThanOrEqual(
            cache.snapshot.retainedBytes + cache.snapshot.reservedBytes,
            Int.max)
    }

    func testFailedExactReplacementPreservesPriorEntry() throws {
        let semanticKey = try key()
        var cache = ExactPrefixCache<String>(
            policy: try policy(
                entries: 2,
                bytes: 8 * 1024))
        _ = try commit(
            "prior",
            key: semanticKey,
            tokens: [1, 2],
            into: &cache)

        let replacement = try XCTUnwrap(
            try cache.reserve(
                key: semanticKey,
                tokens: [1, 2],
                snapshotBytes: 64
            ).reservation)
        _ = try cache.commit(
            replacement,
            state: "failed",
            actualSnapshotBytes: 64,
            disposition: .failed)

        let hit = try XCTUnwrap(
            try cache.lookup(
                key: semanticKey,
                promptTokens: [1, 2, 3]))
        XCTAssertEqual(hit.state, "prior")
        XCTAssertEqual(cache.snapshot.entryCount, 1)
    }

    func testProtectedEntrySurvivesBestEffortReservationPressure() throws {
        let semanticKey = try key()
        var cache = ExactPrefixCache<String>(
            policy: try policy(entries: 1, bytes: 8 * 1024))
        let primary = try commit(
            "primary",
            key: semanticKey,
            tokens: [1, 2],
            into: &cache)
        let primaryID = try XCTUnwrap(primary.entryID)

        let decision = try cache.reserve(
            key: semanticKey,
            tokens: [1, 2, 3],
            snapshotBytes: 64,
            protectingEntryIDs: [primaryID])

        XCTAssertNil(decision.reservation)
        XCTAssertEqual(
            decision.skipReason,
            .reservationCapacityExhausted)
        XCTAssertTrue(decision.evictedEntryIDs.isEmpty)
        XCTAssertEqual(
            try cache.lookup(
                key: semanticKey,
                promptTokens: [1, 2, 9])?.state,
            "primary")
    }

    func testRollbackDoesNotReinflatePreallocationEvictions() throws {
        let semanticKey = try key()
        let entryBytes = try ExactPrefixCache<String>.retainedBytes(
            key: semanticKey,
            tokens: [1, 2],
            snapshotBytes: 64)
        var cache = ExactPrefixCache<String>(
            policy: try policy(entries: 1, bytes: entryBytes))
        _ = try commit(
            "evicted-before-allocation",
            key: semanticKey,
            tokens: [1, 2],
            into: &cache)

        let decision = try cache.reserve(
            key: semanticKey,
            tokens: [3, 4],
            snapshotBytes: 64)
        let reservation = try XCTUnwrap(decision.reservation)
        XCTAssertEqual(decision.evictedEntryIDs.count, 1)
        XCTAssertEqual(cache.snapshot.entryCount, 0)

        try cache.rollback(reservation)

        XCTAssertEqual(cache.snapshot.entryCount, 0)
        XCTAssertEqual(cache.snapshot.reservationCount, 0)
        XCTAssertEqual(cache.snapshot.evictionCount, 1)
    }
}
