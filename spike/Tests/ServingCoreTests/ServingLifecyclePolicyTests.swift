import XCTest

@testable import ServingCore

final class ServingLifecyclePolicyTests: XCTestCase {
    func testRequestLeaseAllowsExactlyOneTerminalTransitionAndOneBackendCancel() async {
        let counter = CancelCounter()
        let lease = ServingRequestLease(id: ServingRequestID("req-1")) {
            await counter.cancel()
        }

        let initialState = await lease.state
        let didActivate = await lease.activate()
        let activeState = await lease.state
        let firstCancel = await lease.cancel(.clientDisconnected)
        let secondCancel = await lease.cancel(.clientDisconnected)
        let didComplete = await lease.complete()
        let didFail = await lease.fail("backend failure")
        let cancelCount = await counter.value
        let terminalState = await lease.state

        XCTAssertEqual(initialState, .pending)
        XCTAssertTrue(didActivate)
        XCTAssertEqual(activeState, .active)
        XCTAssertTrue(firstCancel)
        XCTAssertFalse(secondCancel)
        XCTAssertFalse(didComplete)
        XCTAssertFalse(didFail)
        XCTAssertEqual(cancelCount, 1)
        XCTAssertEqual(terminalState, .cancelled(.clientDisconnected))
    }

    func testRequestLeaseDoesNotCancelBackendAfterCompletion() async {
        let counter = CancelCounter()
        let lease = ServingRequestLease(id: ServingRequestID("req-2")) {
            await counter.cancel()
        }

        let didActivate = await lease.activate()
        let didComplete = await lease.complete()
        let didCancel = await lease.cancel(.shutdown)
        let cancelCount = await counter.value
        let terminalState = await lease.state

        XCTAssertTrue(didActivate)
        XCTAssertTrue(didComplete)
        XCTAssertFalse(didCancel)
        XCTAssertEqual(cancelCount, 0)
        XCTAssertEqual(terminalState, .completed)
    }

    func testAdmissionReducerChoosesBatchForSimultaneousHeldRequests() {
        var reducer = ServingAdmissionReducer(configuration: .init(soloPLDQualified: true))

        XCTAssertEqual(reducer.submit(ServingRequestID("a")), .held([ServingRequestID("a")]))
        XCTAssertEqual(
            reducer.submit(ServingRequestID("b")),
            .held([ServingRequestID("a"), ServingRequestID("b")]))
        XCTAssertEqual(
            reducer.coalescingExpired(),
            .start(route: .continuousBatchNoSpec, requests: [ServingRequestID("a"), ServingRequestID("b")]))
        XCTAssertEqual(
            reducer.submit(ServingRequestID("c")),
            .joinedContinuousBatch(
                requests: [ServingRequestID("a"), ServingRequestID("b"), ServingRequestID("c")]))
    }

    func testAdmissionReducerUsesSoloPLDOnlyWhenQualified() {
        var qualified = ServingAdmissionReducer(configuration: .init(soloPLDQualified: true))
        XCTAssertEqual(qualified.submit(ServingRequestID("solo")), .held([ServingRequestID("solo")]))
        XCTAssertEqual(
            qualified.coalescingExpired(),
            .start(route: .soloPLD, requests: [ServingRequestID("solo")]))

        var unqualified = ServingAdmissionReducer(configuration: .init(soloPLDQualified: false))
        XCTAssertEqual(unqualified.submit(ServingRequestID("solo")), .held([ServingRequestID("solo")]))
        XCTAssertEqual(
            unqualified.coalescingExpired(),
            .start(route: .continuousBatchNoSpec, requests: [ServingRequestID("solo")]))
    }

    func testAdmissionReducerCancellationDuringHoldAndSoloQueueing() {
        var reducer = ServingAdmissionReducer(configuration: .init(soloPLDQualified: true))

        XCTAssertEqual(reducer.submit(ServingRequestID("cancelled")), .held([ServingRequestID("cancelled")]))
        XCTAssertEqual(reducer.cancel(ServingRequestID("cancelled")), .removedFromHold([]))
        XCTAssertEqual(reducer.coalescingExpired(), .idle)

        XCTAssertEqual(reducer.submit(ServingRequestID("solo")), .held([ServingRequestID("solo")]))
        XCTAssertEqual(
            reducer.coalescingExpired(),
            .start(route: .soloPLD, requests: [ServingRequestID("solo")]))
        XCTAssertEqual(
            reducer.submit(ServingRequestID("late")),
            .queued([ServingRequestID("late")]))
        XCTAssertEqual(reducer.executionFinished(requests: [ServingRequestID("solo")]), .held([ServingRequestID("late")]))
    }

    func testAdmissionReducerDoesNotChangeRouteAfterExecutionBegins() {
        var reducer = ServingAdmissionReducer(configuration: .init(soloPLDQualified: true))

        XCTAssertEqual(reducer.submit(ServingRequestID("solo")), .held([ServingRequestID("solo")]))
        XCTAssertEqual(
            reducer.coalescingExpired(),
            .start(route: .soloPLD, requests: [ServingRequestID("solo")]))
        XCTAssertEqual(reducer.currentExecutionRoute, .soloPLD)
        XCTAssertEqual(reducer.submit(ServingRequestID("late")), .queued([ServingRequestID("late")]))
        XCTAssertEqual(reducer.currentExecutionRoute, .soloPLD)
        XCTAssertEqual(reducer.coalescingExpired(), .noRouteChange(route: .soloPLD))
    }

    func testAdmissionReducerCancelsExecutingRequestAndReleasesQueuedReplacement() {
        var reducer = ServingAdmissionReducer(
            configuration: .init(
                soloPLDQualified: true,
                maximumBatchRequests: 1,
                maximumQueuedRequests: 2))

        XCTAssertEqual(reducer.submit(ServingRequestID("solo")), .held([ServingRequestID("solo")]))
        XCTAssertEqual(
            reducer.coalescingExpired(),
            .start(route: .soloPLD, requests: [ServingRequestID("solo")]))
        XCTAssertEqual(
            reducer.submit(ServingRequestID("replacement")),
            .queued([ServingRequestID("replacement")]))

        XCTAssertEqual(
            reducer.cancel(ServingRequestID("solo")),
            .removedFromExecution(
                remaining: [],
                replacements: [],
                nextHeld: [ServingRequestID("replacement")]))
        XCTAssertNil(reducer.currentExecutionRoute)
        XCTAssertEqual(
            reducer.coalescingExpired(),
            .start(route: .soloPLD, requests: [ServingRequestID("replacement")]))
    }

    func testAdmissionReducerBoundsHeldAndQueuedRequestsAndRejectsDuplicates() {
        var reducer = ServingAdmissionReducer(
            configuration: .init(
                soloPLDQualified: false,
                maximumBatchRequests: 2,
                maximumQueuedRequests: 1))

        XCTAssertEqual(reducer.submit(ServingRequestID("a")), .held([ServingRequestID("a")]))
        XCTAssertEqual(
            reducer.submit(ServingRequestID("b")),
            .held([ServingRequestID("a"), ServingRequestID("b")]))
        XCTAssertEqual(reducer.submit(ServingRequestID("c")), .queued([ServingRequestID("c")]))
        XCTAssertEqual(
            reducer.submit(ServingRequestID("d")),
            .rejected(request: ServingRequestID("d"), reason: .queueFull))
        XCTAssertEqual(
            reducer.submit(ServingRequestID("a")),
            .rejected(request: ServingRequestID("a"), reason: .duplicateRequest))
    }

    func testAdmissionReducerRefillsHeldRequestsFromQueueAfterCancellation() {
        var reducer = ServingAdmissionReducer(
            configuration: .init(
                soloPLDQualified: false,
                maximumBatchRequests: 2,
                maximumQueuedRequests: 2))

        XCTAssertEqual(reducer.submit(ServingRequestID("a")), .held([ServingRequestID("a")]))
        XCTAssertEqual(
            reducer.submit(ServingRequestID("b")),
            .held([ServingRequestID("a"), ServingRequestID("b")]))
        XCTAssertEqual(reducer.submit(ServingRequestID("c")), .queued([ServingRequestID("c")]))

        XCTAssertEqual(
            reducer.cancel(ServingRequestID("a")),
            .removedFromHold([ServingRequestID("b"), ServingRequestID("c")]))
        XCTAssertEqual(
            reducer.coalescingExpired(),
            .start(
                route: .continuousBatchNoSpec,
                requests: [ServingRequestID("b"), ServingRequestID("c")]))
    }

    func testAdmissionReducerAllowsBoundedJoinOnlyForRunningContinuousBatch() {
        var batch = ServingAdmissionReducer(
            configuration: .init(
                soloPLDQualified: false,
                maximumBatchRequests: 3,
                maximumQueuedRequests: 2))

        XCTAssertEqual(batch.submit(ServingRequestID("a")), .held([ServingRequestID("a")]))
        XCTAssertEqual(batch.submit(ServingRequestID("b")), .held([ServingRequestID("a"), ServingRequestID("b")]))
        XCTAssertEqual(
            batch.coalescingExpired(),
            .start(
                route: .continuousBatchNoSpec,
                requests: [ServingRequestID("a"), ServingRequestID("b")]))
        XCTAssertEqual(
            batch.submit(ServingRequestID("c")),
            .joinedContinuousBatch(
                requests: [ServingRequestID("a"), ServingRequestID("b"), ServingRequestID("c")]))
        XCTAssertEqual(batch.submit(ServingRequestID("d")), .queued([ServingRequestID("d")]))

        var solo = ServingAdmissionReducer(
            configuration: .init(
                soloPLDQualified: true,
                maximumBatchRequests: 3,
                maximumQueuedRequests: 2))
        XCTAssertEqual(solo.submit(ServingRequestID("solo")), .held([ServingRequestID("solo")]))
        XCTAssertEqual(
            solo.coalescingExpired(),
            .start(route: .soloPLD, requests: [ServingRequestID("solo")]))
        XCTAssertEqual(solo.submit(ServingRequestID("late")), .queued([ServingRequestID("late")]))
    }

    func testContinuousBatchPromotesOldestQueuedRequestAfterPartialCancellation() {
        var reducer = ServingAdmissionReducer(
            configuration: .init(
                soloPLDQualified: false,
                maximumBatchRequests: 2,
                maximumQueuedRequests: 3))

        XCTAssertEqual(reducer.submit(ServingRequestID("a")), .held([ServingRequestID("a")]))
        XCTAssertEqual(reducer.submit(ServingRequestID("b")), .held([ServingRequestID("a"), ServingRequestID("b")]))
        XCTAssertEqual(
            reducer.coalescingExpired(),
            .start(
                route: .continuousBatchNoSpec,
                requests: [ServingRequestID("a"), ServingRequestID("b")]))
        XCTAssertEqual(reducer.submit(ServingRequestID("c")), .queued([ServingRequestID("c")]))
        XCTAssertEqual(reducer.submit(ServingRequestID("d")), .queued([ServingRequestID("c"), ServingRequestID("d")]))

        XCTAssertEqual(
            reducer.cancel(ServingRequestID("a")),
            .removedFromExecution(
                remaining: [ServingRequestID("b"), ServingRequestID("c")],
                replacements: [ServingRequestID("c")],
                nextHeld: []))
        XCTAssertEqual(
            reducer.submit(ServingRequestID("e")),
            .queued([ServingRequestID("d"), ServingRequestID("e")]))
    }

    func testContinuousBatchPromotesOldestQueuedRequestAfterPartialCompletion() {
        var reducer = ServingAdmissionReducer(
            configuration: .init(
                soloPLDQualified: false,
                maximumBatchRequests: 2,
                maximumQueuedRequests: 2))

        XCTAssertEqual(reducer.submit(ServingRequestID("a")), .held([ServingRequestID("a")]))
        XCTAssertEqual(reducer.submit(ServingRequestID("b")), .held([ServingRequestID("a"), ServingRequestID("b")]))
        XCTAssertEqual(
            reducer.coalescingExpired(),
            .start(
                route: .continuousBatchNoSpec,
                requests: [ServingRequestID("a"), ServingRequestID("b")]))
        XCTAssertEqual(reducer.submit(ServingRequestID("c")), .queued([ServingRequestID("c")]))

        XCTAssertEqual(
            reducer.executionFinished(requests: [ServingRequestID("a")]),
            .continuedExecution(
                route: .continuousBatchNoSpec,
                remaining: [ServingRequestID("b"), ServingRequestID("c")],
                replacements: [ServingRequestID("c")]))
    }
}

private actor CancelCounter {
    private var count = 0

    var value: Int { count }

    func cancel() {
        count += 1
    }
}
