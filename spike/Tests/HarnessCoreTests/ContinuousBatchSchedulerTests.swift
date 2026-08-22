import XCTest
@testable import HarnessCore

final class ContinuousBatchSchedulerTests: XCTestCase {
    private func id(_ value: UInt64) -> BatchRequestID {
        BatchRequestID(value)
    }

    private func request(
        _ value: UInt64,
        promptTokens: Int,
        maxOutputTokens: Int = 8,
        architecture: BatchArchitectureClass = .denseAttention,
        speculation: Bool = false,
        decodeCohort: BatchDecodeCohort = .unrestricted
    ) -> BatchRequest {
        BatchRequest(
            id: id(value),
            promptTokenCount: promptTokens,
            maxOutputTokens: maxOutputTokens,
            architecture: architecture,
            requestsSpeculation: speculation,
            decodeCohort: decodeCohort
        )
    }

    private func configuration(
        active: Int = 4,
        prefill: Int = 2,
        chunk: Int = 4,
        queued: Int = 256
    ) -> ContinuousBatchConfiguration {
        try! ContinuousBatchConfiguration(
            maxActiveSlots: active,
            maxPrefillSlots: prefill,
            prefillChunkSize: chunk,
            maxQueuedRequests: queued
        )
    }

    @discardableResult
    private func applyNext(
        _ scheduler: inout ContinuousBatchScheduler,
        finished: Set<BatchRequestID> = []
    ) throws -> BatchTickPlan {
        let plan = scheduler.makeTick()
        try scheduler.apply(
            plan,
            decodeOutcomes: defaultDecodeOutcomes(for: plan, finished: finished)
        )
        return plan
    }

    private func defaultDecodeOutcomes(
        for plan: BatchTickPlan,
        finished: Set<BatchRequestID> = []
    ) -> [BatchDecodeOutcome] {
        guard let decode = plan.decode else { return [] }
        let ids: [BatchRequestID]
        let hasPendingSoloLookahead: Bool
        switch decode {
        case .drainSoloPipeline(let id):
            ids = [id]
            hasPendingSoloLookahead = false
        case .solo(let id, _):
            ids = [id]
            hasPendingSoloLookahead = true
        case .batch(let batchIDs, _):
            ids = batchIDs
            hasPendingSoloLookahead = false
        }
        return ids.map {
            BatchDecodeOutcome(
                id: $0,
                emittedTokenCount: 1,
                finished: finished.contains($0),
                hasPendingSoloLookahead: hasPendingSoloLookahead
            )
        }
    }

    func testConfigurationRejectsNonPositiveAndInconsistentLimits() {
        XCTAssertThrowsError(
            try ContinuousBatchConfiguration(
                maxActiveSlots: 0, maxPrefillSlots: 1, prefillChunkSize: 4)
        ) { error in
            XCTAssertEqual(
                error as? ContinuousBatchSchedulerError,
                .invalidMaxActiveSlots(0)
            )
        }

        XCTAssertThrowsError(
            try ContinuousBatchConfiguration(
                maxActiveSlots: 2, maxPrefillSlots: 0, prefillChunkSize: 4)
        ) { error in
            XCTAssertEqual(
                error as? ContinuousBatchSchedulerError,
                .invalidMaxPrefillSlots(0)
            )
        }

        XCTAssertThrowsError(
            try ContinuousBatchConfiguration(
                maxActiveSlots: 2, maxPrefillSlots: 3, prefillChunkSize: 4)
        ) { error in
            XCTAssertEqual(
                error as? ContinuousBatchSchedulerError,
                .prefillSlotsExceedActive(prefill: 3, active: 2)
            )
        }

        XCTAssertThrowsError(
            try ContinuousBatchConfiguration(
                maxActiveSlots: 2, maxPrefillSlots: 1, prefillChunkSize: 0)
        ) { error in
            XCTAssertEqual(
                error as? ContinuousBatchSchedulerError,
                .invalidPrefillChunkSize(0)
            )
        }

        XCTAssertThrowsError(
            try ContinuousBatchConfiguration(
                maxActiveSlots: 2, maxPrefillSlots: 1, prefillChunkSize: 4,
                maxQueuedRequests: 0)
        ) { error in
            XCTAssertEqual(
                error as? ContinuousBatchSchedulerError,
                .invalidMaxQueuedRequests(0)
            )
        }
    }

    func testQueueCapacityIsExplicitAndRejectsOverflowAtomically() throws {
        var scheduler = ContinuousBatchScheduler(
            configuration: configuration(active: 1, prefill: 1, chunk: 1, queued: 2))
        try scheduler.submit(request(1, promptTokens: 1))
        try scheduler.submit(request(2, promptTokens: 1))

        XCTAssertThrowsError(try scheduler.submit(request(3, promptTokens: 1))) { error in
            XCTAssertEqual(
                error as? ContinuousBatchSchedulerError,
                .queueCapacityExceeded(limit: 2)
            )
        }
        XCTAssertEqual(scheduler.queuedRequestIDs, [id(1), id(2)])
        XCTAssertNil(scheduler.snapshot(for: id(3)))
    }

    func testSubmitFailsClosedForInvalidOrUnsupportedRequests() throws {
        var scheduler = ContinuousBatchScheduler(configuration: configuration())

        XCTAssertThrowsError(try scheduler.submit(request(1, promptTokens: 0))) { error in
            XCTAssertEqual(
                error as? ContinuousBatchSchedulerError,
                .emptyPrompt(id(1))
            )
        }

        XCTAssertThrowsError(
            try scheduler.submit(request(2, promptTokens: 8, maxOutputTokens: 0))
        ) { error in
            XCTAssertEqual(
                error as? ContinuousBatchSchedulerError,
                .invalidOutputBudget(id(2), 0)
            )
        }

        for (offset, architecture) in [
            BatchArchitectureClass.mixtureOfExperts,
            .hybridStateSpace,
            .vision,
            .diffusion,
            .unknown,
        ].enumerated() {
            let requestID = id(UInt64(10 + offset))
            XCTAssertThrowsError(
                try scheduler.submit(
                    request(
                        requestID.rawValue,
                        promptTokens: 8,
                        architecture: architecture
                    ))
            ) { error in
                XCTAssertEqual(
                    error as? ContinuousBatchSchedulerError,
                    .unsupportedArchitecture(requestID, architecture)
                )
            }
        }

        try scheduler.submit(request(99, promptTokens: 8))
        XCTAssertThrowsError(try scheduler.submit(request(99, promptTokens: 8))) { error in
            XCTAssertEqual(
                error as? ContinuousBatchSchedulerError,
                .duplicateRequest(id(99))
            )
        }
    }

    func testFIFOAdmissionIsBoundedByActiveAndPrefillSlots() throws {
        var scheduler = ContinuousBatchScheduler(
            configuration: configuration(active: 2, prefill: 1, chunk: 4))
        try scheduler.submit(request(1, promptTokens: 10))
        try scheduler.submit(request(2, promptTokens: 2))
        try scheduler.submit(request(3, promptTokens: 2))

        let first = scheduler.makeTick()
        XCTAssertEqual(first.admissions, [id(1)])
        XCTAssertEqual(first.prefills, [BatchPrefillSlice(id: id(1), startToken: 0, count: 4)])
        try scheduler.apply(first)

        let second = scheduler.makeTick()
        XCTAssertEqual(second.admissions, [])
        XCTAssertEqual(second.prefills, [BatchPrefillSlice(id: id(1), startToken: 4, count: 4)])
        XCTAssertEqual(scheduler.queuedRequestIDs, [id(2), id(3)])
    }

    func testDecodeRunsBeforePrefillAndShortPromptAdvancesBesideLongPrompt() throws {
        var scheduler = ContinuousBatchScheduler(
            configuration: configuration(active: 2, prefill: 2, chunk: 4))
        try scheduler.submit(request(1, promptTokens: 10))
        try scheduler.submit(request(2, promptTokens: 2, speculation: true))

        let first = try applyNext(&scheduler)
        XCTAssertEqual(first.admissions, [id(1), id(2)])
        XCTAssertEqual(
            first.prefills,
            [
                BatchPrefillSlice(id: id(1), startToken: 0, count: 4),
                BatchPrefillSlice(id: id(2), startToken: 0, count: 2),
            ])
        XCTAssertEqual(scheduler.snapshot(for: id(2))?.phase, .ready)

        let second = scheduler.makeTick()
        XCTAssertEqual(second.decode, .solo(id(2), speculationAllowed: false))
        XCTAssertEqual(
            second.prefills,
            [BatchPrefillSlice(id: id(1), startToken: 4, count: 4)]
        )
        XCTAssertEqual(
            second.operations,
            [
                .decode(.solo(id(2), speculationAllowed: false)),
                .prefill(BatchPrefillSlice(id: id(1), startToken: 4, count: 4)),
            ],
            "decode must be ordered before bounded prefill work"
        )
    }

    func testSoloToBatchJoinDrainsWhileCompanionPrefills() throws {
        var scheduler = ContinuousBatchScheduler(
            configuration: configuration(active: 2, prefill: 1, chunk: 1))
        try scheduler.submit(request(1, promptTokens: 1, speculation: true))

        _ = try applyNext(&scheduler) // request 1 -> ready
        let firstSolo = try applyNext(&scheduler)
        XCTAssertEqual(firstSolo.decode, .solo(id(1), speculationAllowed: true))
        XCTAssertEqual(
            scheduler.snapshot(for: id(1))?.phase,
            .decoding(
                emittedTokens: 1,
                soloPipelineState: .pipelinedLookahead)
        )

        try scheduler.submit(request(2, promptTokens: 1, speculation: true))
        let prefillingJoiner = try applyNext(&scheduler)
        XCTAssertEqual(prefillingJoiner.decode, .drainSoloPipeline(id(1)))
        XCTAssertEqual(prefillingJoiner.admissions, [id(2)])
        XCTAssertEqual(scheduler.snapshot(for: id(2))?.phase, .ready)
        XCTAssertEqual(
            scheduler.snapshot(for: id(1))?.phase,
            .decoding(
                emittedTokens: 2,
                soloPipelineState: .canonical)
        )

        let shared = scheduler.makeTick()
        XCTAssertEqual(
            shared.decode,
            .batch([id(1), id(2)], speculationAllowed: false)
        )
        XCTAssertEqual(
            shared.operations.first,
            .decode(.batch([id(1), id(2)], speculationAllowed: false))
        )
    }

    func testSpeculativeSoloRequiresOutputlessCanonicalDrainBeforeSharedBatch() throws {
        var scheduler = ContinuousBatchScheduler(
            configuration: configuration(active: 2, prefill: 1, chunk: 1))
        try scheduler.submit(request(1, promptTokens: 1, speculation: true))

        _ = try applyNext(&scheduler) // request 1 -> ready
        let speculative = scheduler.makeTick()
        XCTAssertEqual(
            speculative.decode,
            .solo(id(1), speculationAllowed: true))
        try scheduler.apply(
            speculative,
            decodeOutcomes: [
                BatchDecodeOutcome(
                    id: id(1),
                    emittedTokenCount: 3,
                    finished: false,
                    soloPipelineState: .speculative),
            ])
        XCTAssertEqual(
            scheduler.snapshot(for: id(1))?.phase,
            .decoding(
                emittedTokens: 3,
                soloPipelineState: .speculative))

        try scheduler.submit(request(2, promptTokens: 1, speculation: true))
        let join = scheduler.makeTick()
        XCTAssertEqual(join.admissions, [id(2)])
        XCTAssertEqual(join.decode, .drainSoloPipeline(id(1)))
        try scheduler.apply(
            join,
            decodeOutcomes: [
                BatchDecodeOutcome(
                    id: id(1),
                    emittedTokenCount: 0,
                    finished: false,
                    soloPipelineState: .canonical),
            ])
        XCTAssertEqual(
            scheduler.snapshot(for: id(1))?.phase,
            .decoding(
                emittedTokens: 3,
                soloPipelineState: .canonical))
        XCTAssertEqual(
            scheduler.makeTick().decode,
            .batch([id(1), id(2)], speculationAllowed: false))
    }

    func testQueuedRequestBehindFullActiveCapacityDoesNotSuppressSoloSpeculation() throws {
        var scheduler = ContinuousBatchScheduler(
            configuration: configuration(active: 1, prefill: 1, chunk: 1))
        try scheduler.submit(request(1, promptTokens: 1, speculation: true))

        _ = try applyNext(&scheduler)
        try scheduler.submit(request(2, promptTokens: 1, speculation: true))
        let solo = scheduler.makeTick()
        XCTAssertEqual(solo.admissions, [])
        XCTAssertEqual(solo.decode, .solo(id(1), speculationAllowed: true))
    }

    func testPrefillingIncompatibleCohortDoesNotSuppressSoloSpeculation() throws {
        var scheduler = ContinuousBatchScheduler(
            configuration: configuration(active: 2, prefill: 1, chunk: 1))
        try scheduler.submit(
            request(
                1,
                promptTokens: 1,
                speculation: true,
                decodeCohort: .fixedKVCapacity(16)))

        _ = try applyNext(&scheduler)
        try scheduler.submit(
            request(
                2,
                promptTokens: 2,
                speculation: true,
                decodeCohort: .fixedKVCapacity(32)))

        let solo = scheduler.makeTick()
        XCTAssertEqual(solo.admissions, [id(2)])
        XCTAssertEqual(solo.decode, .solo(id(1), speculationAllowed: true))
    }

    func testCancellingSpeculativeSoloMakesItsPlannedDrainStale() throws {
        var scheduler = ContinuousBatchScheduler(
            configuration: configuration(active: 2, prefill: 1, chunk: 1))
        try scheduler.submit(request(1, promptTokens: 1, speculation: true))
        _ = try applyNext(&scheduler)
        let speculative = scheduler.makeTick()
        try scheduler.apply(
            speculative,
            decodeOutcomes: [
                BatchDecodeOutcome(
                    id: id(1),
                    emittedTokenCount: 2,
                    finished: false,
                    soloPipelineState: .speculative),
            ])
        try scheduler.submit(request(2, promptTokens: 1, speculation: true))
        let staleDrain = scheduler.makeTick()
        XCTAssertEqual(staleDrain.decode, .drainSoloPipeline(id(1)))
        XCTAssertEqual(staleDrain.admissions, [id(2)])
        XCTAssertEqual(scheduler.cancel(id(1)).id, id(1))
        try scheduler.submit(request(1, promptTokens: 2, speculation: true))

        XCTAssertThrowsError(
            try scheduler.apply(
                staleDrain,
                decodeOutcomes: [
                    BatchDecodeOutcome(
                        id: id(1),
                        emittedTokenCount: 0,
                        finished: false,
                        soloPipelineState: .canonical),
                ])
        ) { error in
            XCTAssertEqual(
                error as? ContinuousBatchSchedulerError,
                .staleTick(
                    expected: staleDrain.sequence + 2,
                    actual: staleDrain.sequence))
        }
        XCTAssertEqual(scheduler.snapshot(for: id(1))?.phase, .queued)
        XCTAssertEqual(scheduler.snapshot(for: id(2))?.phase, .queued)
    }

    func testSharedDecodeDisablesSpeculationForEverySlot() throws {
        var scheduler = ContinuousBatchScheduler(
            configuration: configuration(active: 2, prefill: 2, chunk: 1))
        try scheduler.submit(request(1, promptTokens: 1, speculation: true))
        try scheduler.submit(request(2, promptTokens: 1, speculation: true))

        _ = try applyNext(&scheduler) // both ready, no prior solo pipeline
        let shared = scheduler.makeTick()
        XCTAssertEqual(
            shared.decode,
            .batch([id(1), id(2)], speculationAllowed: false)
        )
    }

    func testIncompatibleCapacityCohortsNeverMixAndRoundRobin() throws {
        var scheduler = ContinuousBatchScheduler(
            configuration: configuration(active: 3, prefill: 3, chunk: 1))
        try scheduler.submit(
            request(
                1,
                promptTokens: 1,
                decodeCohort: .fixedKVCapacity(256)))
        try scheduler.submit(
            request(
                2,
                promptTokens: 1,
                decodeCohort: .fixedKVCapacity(256)))
        try scheduler.submit(
            request(
                3,
                promptTokens: 1,
                decodeCohort: .fixedKVCapacity(512)))

        _ = try applyNext(&scheduler)

        let first = try applyNext(&scheduler)
        XCTAssertEqual(
            first.decode,
            .batch([id(1), id(2)], speculationAllowed: false))

        let second = try applyNext(&scheduler)
        XCTAssertEqual(
            second.decode,
            .solo(id(3), speculationAllowed: false))

        let third = try applyNext(&scheduler)
        XCTAssertEqual(
            third.decode,
            .batch([id(1), id(2)], speculationAllowed: false))

        let fourth = scheduler.makeTick()
        XCTAssertEqual(
            fourth.decode,
            .solo(id(3), speculationAllowed: false))
    }

    func testIsolatedDecodeCohortsNeverShareAForward() throws {
        var scheduler = ContinuousBatchScheduler(
            configuration: configuration(active: 2, prefill: 2, chunk: 1))
        try scheduler.submit(
            request(
                1,
                promptTokens: 1,
                decodeCohort: .isolated(id(1))))
        try scheduler.submit(
            request(
                2,
                promptTokens: 1,
                decodeCohort: .isolated(id(2))))

        _ = try applyNext(&scheduler)

        let first = try applyNext(&scheduler)
        XCTAssertEqual(
            first.decode,
            .solo(id(1), speculationAllowed: false))
        let second = scheduler.makeTick()
        XCTAssertEqual(
            second.decode,
            .solo(id(2), speculationAllowed: false))
    }

    func testFreshBurstRestartsCohortSelectionInFIFOOrderAfterIdle() throws {
        var scheduler = ContinuousBatchScheduler(
            configuration: configuration(active: 2, prefill: 2, chunk: 1))
        try scheduler.submit(
            request(
                1,
                promptTokens: 1,
                maxOutputTokens: 1,
                decodeCohort: .fixedKVCapacity(256)))
        _ = try applyNext(&scheduler)
        let finishing = scheduler.makeTick()
        try scheduler.apply(
            finishing,
            decodeOutcomes: defaultDecodeOutcomes(
                for: finishing,
                finished: [id(1)]))
        XCTAssertTrue(scheduler.isEmpty)

        try scheduler.submit(
            request(
                2,
                promptTokens: 1,
                decodeCohort: .fixedKVCapacity(256)))
        try scheduler.submit(
            request(
                3,
                promptTokens: 1,
                decodeCohort: .fixedKVCapacity(512)))
        _ = try applyNext(&scheduler)

        XCTAssertEqual(
            scheduler.makeTick().decode,
            .solo(id(2), speculationAllowed: false))
    }

    func testSpeculativeSoloOutcomeControlsTokenCountAndLookaheadState() throws {
        var scheduler = ContinuousBatchScheduler(
            configuration: configuration(active: 2, prefill: 2, chunk: 1))
        try scheduler.submit(
            request(1, promptTokens: 1, maxOutputTokens: 10, speculation: true))
        _ = try applyNext(&scheduler)

        let speculative = scheduler.makeTick()
        XCTAssertEqual(speculative.decode, .solo(id(1), speculationAllowed: true))
        try scheduler.apply(
            speculative,
            decodeOutcomes: [
                BatchDecodeOutcome(
                    id: id(1),
                    emittedTokenCount: 3,
                    finished: false,
                    hasPendingSoloLookahead: false
                )
            ]
        )
        XCTAssertEqual(
            scheduler.snapshot(for: id(1))?.phase,
            .decoding(
                emittedTokens: 3,
                soloPipelineState: .canonical)
        )

        try scheduler.submit(request(2, promptTokens: 1, speculation: false))
        let admit = scheduler.makeTick()
        XCTAssertEqual(admit.decode, .solo(id(1), speculationAllowed: false))
        try scheduler.apply(
            admit,
            decodeOutcomes: [
                BatchDecodeOutcome(
                    id: id(1),
                    emittedTokenCount: 1,
                    finished: false,
                    hasPendingSoloLookahead: false
                )
            ]
        )
        XCTAssertEqual(scheduler.snapshot(for: id(2))?.phase, .ready)

        let shared = scheduler.makeTick()
        XCTAssertEqual(
            shared.decode,
            .batch([id(1), id(2)], speculationAllowed: false),
            "an executor-confirmed no-lookahead solo state can join without a false drain"
        )
    }

    func testCancellationRemovesQueuedPrefillingReadyAndDecodingSlots() throws {
        do {
            var scheduler = ContinuousBatchScheduler(
                configuration: configuration(active: 1, prefill: 1, chunk: 2))
            try scheduler.submit(request(1, promptTokens: 4))
            XCTAssertEqual(
                scheduler.cancel(id(1)),
                .cancelled(id: id(1), previousPhase: .queued)
            )
            XCTAssertEqual(scheduler.cancel(id(1)), .notFound(id(1)))
        }

        do {
            var scheduler = ContinuousBatchScheduler(
                configuration: configuration(active: 1, prefill: 1, chunk: 2))
            try scheduler.submit(request(2, promptTokens: 4))
            _ = try applyNext(&scheduler)
            XCTAssertEqual(
                scheduler.cancel(id(2)),
                .cancelled(
                    id: id(2),
                    previousPhase: .prefilling(processedTokens: 2, totalTokens: 4)
                )
            )
        }

        do {
            var scheduler = ContinuousBatchScheduler(
                configuration: configuration(active: 1, prefill: 1, chunk: 4))
            try scheduler.submit(request(3, promptTokens: 4))
            _ = try applyNext(&scheduler)
            XCTAssertEqual(scheduler.snapshot(for: id(3))?.phase, .ready)
            XCTAssertEqual(
                scheduler.cancel(id(3)),
                .cancelled(id: id(3), previousPhase: .ready)
            )
        }

        do {
            var scheduler = ContinuousBatchScheduler(
                configuration: configuration(active: 1, prefill: 1, chunk: 4))
            try scheduler.submit(request(4, promptTokens: 4))
            _ = try applyNext(&scheduler)
            _ = try applyNext(&scheduler)
            XCTAssertEqual(
                scheduler.cancel(id(4)),
                .cancelled(
                    id: id(4),
                    previousPhase: .decoding(
                        emittedTokens: 1,
                        soloPipelineState: .pipelinedLookahead)
                )
            )
        }
    }

    func testCancellationReleasesCapacityForNextFIFORequest() throws {
        var scheduler = ContinuousBatchScheduler(
            configuration: configuration(active: 1, prefill: 1, chunk: 2))
        try scheduler.submit(request(1, promptTokens: 8))
        try scheduler.submit(request(2, promptTokens: 2))

        _ = try applyNext(&scheduler)
        XCTAssertEqual(scheduler.cancel(id(1)).id, id(1))

        let replacement = scheduler.makeTick()
        XCTAssertEqual(replacement.admissions, [id(2)])
        XCTAssertEqual(
            replacement.prefills,
            [BatchPrefillSlice(id: id(2), startToken: 0, count: 2)]
        )
    }

    func testOutputBudgetFinishesAndReleasesSlot() throws {
        var scheduler = ContinuousBatchScheduler(
            configuration: configuration(active: 1, prefill: 1, chunk: 1))
        try scheduler.submit(request(1, promptTokens: 1, maxOutputTokens: 1))
        _ = try applyNext(&scheduler)
        _ = try applyNext(&scheduler)
        XCTAssertNil(scheduler.snapshot(for: id(1)))
        XCTAssertTrue(scheduler.isEmpty)
    }

    func testMultiTokenSoloOutcomeEnforcesOutputBudget() throws {
        var scheduler = ContinuousBatchScheduler(
            configuration: configuration(active: 1, prefill: 1, chunk: 1))
        try scheduler.submit(
            request(1, promptTokens: 1, maxOutputTokens: 3, speculation: true))
        _ = try applyNext(&scheduler)

        let speculative = scheduler.makeTick()
        try scheduler.apply(
            speculative,
            decodeOutcomes: [
                BatchDecodeOutcome(
                    id: id(1),
                    emittedTokenCount: 3,
                    finished: false,
                    hasPendingSoloLookahead: false
                )
            ]
        )
        XCTAssertNil(scheduler.snapshot(for: id(1)))
    }

    func testInvalidDecodeOutcomesFailAtomically() throws {
        var scheduler = ContinuousBatchScheduler(
            configuration: configuration(active: 1, prefill: 1, chunk: 1))
        try scheduler.submit(request(1, promptTokens: 1, speculation: false))
        _ = try applyNext(&scheduler)
        let plan = scheduler.makeTick()
        let before = scheduler.snapshots

        XCTAssertThrowsError(try scheduler.apply(plan, decodeOutcomes: [])) { error in
            XCTAssertEqual(
                error as? ContinuousBatchSchedulerError,
                .invalidDecodeOutcomeIDs(expected: [id(1)], actual: [])
            )
        }
        XCTAssertEqual(scheduler.snapshots, before)

        XCTAssertThrowsError(
            try scheduler.apply(
                plan,
                decodeOutcomes: [
                    BatchDecodeOutcome(
                        id: id(1),
                        emittedTokenCount: 2,
                        finished: false,
                        hasPendingSoloLookahead: true
                    )
                ]
            )
        ) { error in
            XCTAssertEqual(
                error as? ContinuousBatchSchedulerError,
                .invalidEmittedTokenCount(id(1), 2)
            )
        }
        XCTAssertEqual(scheduler.snapshots, before)
    }

    func testSharedDecodeRejectsPendingSoloLookaheadAtomically() throws {
        var scheduler = ContinuousBatchScheduler(
            configuration: configuration(active: 2, prefill: 2, chunk: 1))
        try scheduler.submit(request(1, promptTokens: 1))
        try scheduler.submit(request(2, promptTokens: 1))
        _ = try applyNext(&scheduler)
        let plan = scheduler.makeTick()
        let before = scheduler.snapshots

        XCTAssertThrowsError(
            try scheduler.apply(
                plan,
                decodeOutcomes: [
                    BatchDecodeOutcome(
                        id: id(1),
                        emittedTokenCount: 1,
                        finished: false,
                        hasPendingSoloLookahead: true
                    ),
                    BatchDecodeOutcome(
                        id: id(2),
                        emittedTokenCount: 1,
                        finished: false,
                        hasPendingSoloLookahead: false
                    ),
                ]
            )
        ) { error in
            XCTAssertEqual(
                error as? ContinuousBatchSchedulerError,
                .invalidPendingLookahead(id(1))
            )
        }
        XCTAssertEqual(scheduler.snapshots, before)
    }

    func testApplyingAPlanTwiceFailsAsStaleWithoutChangingState() throws {
        var scheduler = ContinuousBatchScheduler(configuration: configuration())
        try scheduler.submit(request(1, promptTokens: 2))
        let plan = scheduler.makeTick()
        try scheduler.apply(plan)
        let snapshot = scheduler.snapshots

        XCTAssertThrowsError(try scheduler.apply(plan)) { error in
            XCTAssertEqual(
                error as? ContinuousBatchSchedulerError,
                .staleTick(expected: 2, actual: 1)
            )
        }
        XCTAssertEqual(scheduler.snapshots, snapshot)
    }

    func testCancelAndSameIDResubmitInvalidatesPreviouslyPlannedWork() throws {
        var scheduler = ContinuousBatchScheduler(configuration: configuration())
        let original = request(1, promptTokens: 2)
        try scheduler.submit(original)
        let stale = scheduler.makeTick()

        XCTAssertEqual(scheduler.cancel(id(1)).id, id(1))
        try scheduler.submit(original)
        XCTAssertThrowsError(try scheduler.apply(stale)) { error in
            XCTAssertEqual(
                error as? ContinuousBatchSchedulerError,
                .staleTick(expected: 3, actual: 1)
            )
        }
        XCTAssertEqual(scheduler.snapshot(for: id(1))?.phase, .queued)
    }
}
