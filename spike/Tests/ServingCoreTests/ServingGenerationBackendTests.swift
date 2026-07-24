import XCTest

@testable import ServingCore

final class ServingGenerationBackendTests: XCTestCase {
    func testBackendStartReturnsPureGenerationHandleEnvelope() async throws {
        let request = OpenAIChatCompletionRequest(
            model: "qwen3-32b",
            messages: [OpenAIChatMessage(role: .user, text: "Hello")],
            maxCompletionTokens: 8,
            temperature: 0,
            choiceCount: 1,
            stream: false,
            stop: [])
        let backend = StaticGenerationBackend()

        let handle = try await backend.start(request)

        XCTAssertEqual(handle.responseID, "chatcmpl-core-contract")
        XCTAssertEqual(handle.created, 1_775_000_000)
        XCTAssertEqual(handle.model, "qwen3-32b")
        XCTAssertEqual(handle.route, .continuousBatchNoSpec)
        let leaseState = await handle.lease.state
        let mailboxSnapshot = await handle.mailbox.snapshot()
        XCTAssertEqual(leaseState, .pending)
        XCTAssertEqual(mailboxSnapshot.terminal, nil)
    }

    func testCompletionDeltaCarriesExactTerminalUsageThroughBoundedMailbox() async throws {
        let mailbox = BoundedDeltaMailbox(
            capacity: BoundedDeltaMailbox.Capacity(maxDeltas: 1, maxBytes: 1))
        let completion = ServingGenerationCompletion(
            finishReason: .length,
            usage: OpenAIChatUsage(promptTokens: 13, completionTokens: 8))

        try await mailbox.send(.completion(completion))
        let blockedProducer = Task {
            try await mailbox.send(.text("x"))
            return "sent"
        }

        await waitForGenerationMailboxSnapshot(mailbox) { $0.waitingProducers == 1 }
        let fullSnapshot = await mailbox.snapshot()
        XCTAssertEqual(fullSnapshot.bufferedDeltas, 1)

        let first = try await mailbox.next()
        let producerResult = try await blockedProducer.value

        XCTAssertEqual(first, .completion(completion))
        XCTAssertEqual(producerResult, "sent")
        let second = try await mailbox.next()
        XCTAssertEqual(second, .text("x"))
        XCTAssertEqual(completion.finishReason, .length)
        XCTAssertEqual(completion.usage.promptTokens, 13)
        XCTAssertEqual(completion.usage.completionTokens, 8)
        XCTAssertEqual(completion.usage.totalTokens, 21)
    }
}

private struct StaticGenerationBackend: ServingGenerationBackend {
    func start(_ request: OpenAIChatCompletionRequest) async throws -> ServingGenerationHandle {
        let mailbox = BoundedDeltaMailbox(
            capacity: BoundedDeltaMailbox.Capacity(maxDeltas: 4, maxBytes: 64))
        let lease = ServingRequestLease(id: ServingRequestID("req-core-contract"))
        return ServingGenerationHandle(
            responseID: "chatcmpl-core-contract",
            created: 1_775_000_000,
            model: request.model,
            route: .continuousBatchNoSpec,
            mailbox: mailbox,
            lease: lease)
    }
}

private func waitForGenerationMailboxSnapshot(
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
