import XCTest

@testable import ServingCore

final class ServingBoundedDeltaMailboxTests: XCTestCase {
    func testProducerSuspendsAtDeltaCapacityUntilConsumerDrains() async throws {
        let mailbox = BoundedDeltaMailbox(
            capacity: BoundedDeltaMailbox.Capacity(maxDeltas: 1, maxBytes: 64))

        try await mailbox.send(.text("first"))

        let blocked = Task {
            try await mailbox.send(.text("second"))
            return "sent"
        }

        await waitForSnapshot(mailbox) { $0.waitingProducers == 1 }
        let fullSnapshot = await mailbox.snapshot()
        XCTAssertEqual(fullSnapshot.bufferedDeltas, 1)
        XCTAssertEqual(fullSnapshot.bufferedBytes, 5)
        XCTAssertEqual(fullSnapshot.waitingProducers, 1)

        let first = try await mailbox.next()
        let blockedResult = try await blocked.value
        let second = try await mailbox.next()
        XCTAssertEqual(first, .text("first"))
        XCTAssertEqual(blockedResult, "sent")
        XCTAssertEqual(second, .text("second"))
    }

    func testProducerSuspendsAtByteCapacityUntilConsumerDrains() async throws {
        let mailbox = BoundedDeltaMailbox(
            capacity: BoundedDeltaMailbox.Capacity(maxDeltas: 4, maxBytes: 5))

        try await mailbox.send(.text("12345"))

        let blocked = Task {
            try await mailbox.send(.text("x"))
            return "sent"
        }

        await waitForSnapshot(mailbox) { $0.waitingProducers == 1 }
        let fullSnapshot = await mailbox.snapshot()
        XCTAssertEqual(fullSnapshot.bufferedDeltas, 1)
        XCTAssertEqual(fullSnapshot.bufferedBytes, 5)
        XCTAssertEqual(fullSnapshot.waitingProducers, 1)

        let first = try await mailbox.next()
        let blockedResult = try await blocked.value
        let second = try await mailbox.next()
        XCTAssertEqual(first, .text("12345"))
        XCTAssertEqual(blockedResult, "sent")
        XCTAssertEqual(second, .text("x"))
    }

    func testCancelResumesBlockedProducerAndClearsCapacity() async throws {
        let mailbox = BoundedDeltaMailbox(
            capacity: BoundedDeltaMailbox.Capacity(maxDeltas: 1, maxBytes: 4))

        try await mailbox.send(.text("full"))

        let producer = Task {
            do {
                try await mailbox.send(.text("late"))
                return "sent"
            } catch {
                return "cancelled"
            }
        }

        await waitForSnapshot(mailbox) { $0.waitingProducers == 1 }
        await mailbox.cancel(.clientDisconnected)
        await mailbox.cancel(.clientDisconnected)
        await mailbox.finish()
        await mailbox.fail(.backend("late failure"))

        let producerResult = await producer.value
        XCTAssertEqual(producerResult, "cancelled")
        do {
            _ = try await mailbox.next()
            XCTFail("Expected cancelled mailbox to throw")
        } catch let error as ServingMailboxError {
            XCTAssertEqual(error, .cancelled(.clientDisconnected))
        }
        let terminalSnapshot = await mailbox.snapshot()
        XCTAssertEqual(terminalSnapshot.terminal, .cancelled(.clientDisconnected))
        XCTAssertEqual(terminalSnapshot.waitingConsumers, 0)
        XCTAssertEqual(terminalSnapshot.waitingProducers, 0)
    }

    func testFailResumesWaitingConsumerOnce() async throws {
        let mailbox = BoundedDeltaMailbox(
            capacity: BoundedDeltaMailbox.Capacity(maxDeltas: 1, maxBytes: 4))

        let consumer = Task {
            do {
                _ = try await mailbox.next()
                return "value"
            } catch {
                return "failed"
            }
        }

        await waitForSnapshot(mailbox) { $0.waitingConsumers == 1 }
        await mailbox.fail(.backend("backend failure"))
        await mailbox.fail(.backend("late failure"))
        await mailbox.cancel(.shutdown)

        let consumerResult = await consumer.value
        XCTAssertEqual(consumerResult, "failed")
        let terminalSnapshot = await mailbox.snapshot()
        XCTAssertEqual(terminalSnapshot.terminal, .failed("backend failure"))
        XCTAssertEqual(terminalSnapshot.waitingConsumers, 0)
        XCTAssertEqual(terminalSnapshot.waitingProducers, 0)
    }

    func testCancellingBlockedProducerRemovesItsContinuation() async throws {
        let mailbox = BoundedDeltaMailbox(
            capacity: BoundedDeltaMailbox.Capacity(maxDeltas: 1, maxBytes: 8))
        try await mailbox.send(.text("full"))

        let blocked = Task {
            try await mailbox.send(.text("late"))
        }
        await waitForSnapshot(mailbox) { $0.waitingProducers == 1 }

        blocked.cancel()
        await waitForSnapshot(mailbox) { $0.waitingProducers == 0 }

        do {
            try await blocked.value
            XCTFail("Expected producer task cancellation")
        } catch is CancellationError {
            // Expected.
        }
        let snapshot = await mailbox.snapshot()
        XCTAssertEqual(snapshot.bufferedDeltas, 1)
    }

    func testProducerCancellationLinearizesBeforeACompetingDrain() async throws {
        let mailbox = BoundedDeltaMailbox(
            capacity: BoundedDeltaMailbox.Capacity(maxDeltas: 1, maxBytes: 8))
        try await mailbox.send(.text("full"))

        let blocked = Task {
            try await mailbox.send(.text("late"))
        }
        await waitForSnapshot(mailbox) { $0.waitingProducers == 1 }

        blocked.cancel()
        let first = try await mailbox.next()
        XCTAssertEqual(first, .text("full"))
        do {
            try await blocked.value
            XCTFail("Expected producer task cancellation")
        } catch is CancellationError {
            // Expected.
        }

        await mailbox.finish()
        let terminal = try await mailbox.next()
        XCTAssertNil(terminal)
    }

    func testCancellingWaitingConsumerRemovesItsContinuation() async throws {
        let mailbox = BoundedDeltaMailbox(
            capacity: BoundedDeltaMailbox.Capacity(maxDeltas: 1, maxBytes: 8))

        let waiting = Task {
            try await mailbox.next()
        }
        await waitForSnapshot(mailbox) { $0.waitingConsumers == 1 }

        waiting.cancel()
        await waitForSnapshot(mailbox) { $0.waitingConsumers == 0 }

        do {
            _ = try await waiting.value
            XCTFail("Expected consumer task cancellation")
        } catch is CancellationError {
            // Expected.
        }
    }

    func testBackpressureStallDeadlineIsDeterministicAndIdempotent() {
        var stall = BackpressureStallState(maxStallTicks: 3)

        XCTAssertEqual(stall.observe(isFull: false, tick: 10), .none)
        XCTAssertEqual(stall.observe(isFull: true, tick: 11), .started(deadlineTick: 14))
        XCTAssertEqual(stall.observe(isFull: true, tick: 13), .waiting(deadlineTick: 14))
        XCTAssertEqual(stall.observe(isFull: true, tick: 14), .expired(deadlineTick: 14))
        XCTAssertEqual(stall.observe(isFull: true, tick: 20), .expired(deadlineTick: 14))
        XCTAssertEqual(stall.observe(isFull: false, tick: 21), .cleared)
        XCTAssertEqual(stall.observe(isFull: true, tick: 22), .started(deadlineTick: 25))
    }
}

private func waitForSnapshot(
    _ mailbox: BoundedDeltaMailbox,
    matching predicate: (BoundedDeltaMailbox.Snapshot) -> Bool,
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    for _ in 0..<1_000 {
        if predicate(await mailbox.snapshot()) {
            return
        }
        await Task.yield()
    }
    XCTFail("Mailbox did not reach the expected state", file: file, line: line)
}
