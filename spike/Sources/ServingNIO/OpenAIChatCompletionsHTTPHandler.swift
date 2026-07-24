import Foundation
import NIOCore
import NIOHTTP1
import ServingCore
import os

public final class OpenAIChatCompletionsHTTPHandler: ChannelInboundHandler {
    public typealias InboundIn = HTTPServerRequestPart
    public typealias OutboundOut = HTTPServerResponsePart

    private let configuration: ServingHTTPConfiguration
    private let backend: any ServingGenerationBackend

    private var pendingHead: HTTPRequestHead?
    private var pendingBody: ByteBuffer?
    private var discardingRequestBody = false
    private var activeControl: ServingTransportRequestControl?
    private var activeTask: Task<Void, Never>?
    private var activeWritabilityGate: ServingChannelWritabilityGate?

    public init(
        configuration: ServingHTTPConfiguration,
        backend: any ServingGenerationBackend
    ) {
        self.configuration = configuration
        self.backend = backend
    }

    public func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        let part = unwrapInboundIn(data)
        switch part {
        case .head(let head):
            receiveHead(head, context: context)
        case .body(var body):
            receiveBody(&body, context: context)
        case .end:
            receiveEnd(context: context)
        }
    }

    public func channelWritabilityChanged(context: ChannelHandlerContext) {
        if let activeWritabilityGate {
            activeWritabilityGate.update(isWritable: context.channel.isWritable)
        }
        context.fireChannelWritabilityChanged()
    }

    public func channelInactive(context: ChannelHandlerContext) {
        cancelActive(.clientDisconnected)
        context.fireChannelInactive()
    }

    public func userInboundEventTriggered(context: ChannelHandlerContext, event: Any) {
        switch event {
        case let channelEvent as ChannelEvent where channelEvent == .inputClosed:
            cancelActive(.clientDisconnected)
            context.close(promise: nil)
        case is ChannelShouldQuiesceEvent:
            cancelActive(.shutdown)
            context.close(promise: nil)
        default:
            context.fireUserInboundEventTriggered(event)
        }
    }

    public func errorCaught(context: ChannelHandlerContext, error: Error) {
        cancelActive(.clientDisconnected)
        context.close(promise: nil)
    }

    public func handlerRemoved(context: ChannelHandlerContext) {
        cancelActive(.shutdown)
    }

    private func receiveHead(_ head: HTTPRequestHead, context: ChannelHandlerContext) {
        clearTerminalRequestIfNeeded()
        guard activeControl == nil, pendingHead == nil, !discardingRequestBody else {
            cancelActive(.clientDisconnected)
            context.close(promise: nil)
            return
        }

        if let rejection = validateHead(head) {
            discardingRequestBody = true
            writeError(
                rejection.error,
                status: rejection.status,
                keepAlive: head.isKeepAlive,
                context: context)
            return
        }

        pendingHead = head
        pendingBody = context.channel.allocator.buffer(
            capacity: min(
                configuration.requestLimits.maximumBodyBytes,
                contentLength(from: head) ?? 0))
    }

    private func receiveBody(_ body: inout ByteBuffer, context: ChannelHandlerContext) {
        guard !discardingRequestBody else {
            return
        }
        guard pendingHead != nil, var accumulated = pendingBody else {
            context.close(promise: nil)
            return
        }

        let newCount = accumulated.readableBytes + body.readableBytes
        guard newCount <= configuration.requestLimits.maximumBodyBytes else {
            pendingHead = nil
            pendingBody = nil
            discardingRequestBody = true
            writeError(
                .invalidRequest("Request body exceeds the configured byte limit", param: nil),
                status: .payloadTooLarge,
                keepAlive: false,
                context: context)
            return
        }
        accumulated.writeBuffer(&body)
        pendingBody = accumulated
    }

    private func receiveEnd(context: ChannelHandlerContext) {
        if discardingRequestBody {
            discardingRequestBody = false
            return
        }
        guard let head = pendingHead, var body = pendingBody else {
            context.close(promise: nil)
            return
        }
        pendingHead = nil
        pendingBody = nil

        let bodyData = Data(body.readBytes(length: body.readableBytes) ?? [])
        let request: OpenAIChatCompletionRequest
        do {
            request = try OpenAIChatCompletionRequest.decodeStrict(
                from: bodyData,
                limits: configuration.requestLimits)
            try request.requireLaunchedModel(configuration.launchedModel)
        } catch let error as OpenAIServingError {
            writeError(error, status: .badRequest, keepAlive: head.isKeepAlive, context: context)
            return
        } catch {
            writeError(
                .invalidRequest("Request body is not valid JSON", param: nil),
                status: .badRequest,
                keepAlive: head.isKeepAlive,
                context: context)
            return
        }

        let control = ServingTransportRequestControl()
        let writabilityGate = ServingChannelWritabilityGate(
            initiallyWritable: context.channel.isWritable)
        let backend = self.backend
        let configuration = self.configuration
        let channel = context.channel
        activeControl = control
        activeWritabilityGate = writabilityGate
        activeTask = Self.makeGenerationTask(
            request: request,
            keepAlive: head.isKeepAlive,
            configuration: configuration,
            backend: backend,
            channel: channel,
            control: control,
            writabilityGate: writabilityGate)
    }

    private func validateHead(
        _ head: HTTPRequestHead
    ) -> (status: HTTPResponseStatus, error: OpenAIServingError)? {
        guard head.method == .POST else {
            return (
                .methodNotAllowed,
                .invalidRequest("Only POST is supported for this route", param: nil))
        }
        guard head.uri == "/v1/chat/completions" else {
            return (
                .notFound,
                .invalidRequest("Unknown route", param: nil))
        }

        let contentTypes = head.headers["content-type"]
        guard contentTypes.count == 1,
            contentTypes[0]
                .split(separator: ";", maxSplits: 1)
                .first?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased() == "application/json"
        else {
            return (
                .unsupportedMediaType,
                .invalidRequest("Content-Type must be application/json", param: nil))
        }

        let lengthHeaders = head.headers["content-length"]
        if lengthHeaders.count > 1 {
            return (
                .badRequest,
                .invalidRequest("Content-Length must be unique", param: nil))
        }
        if let rawLength = lengthHeaders.first {
            guard let length = Int(rawLength), length >= 0 else {
                return (
                    .badRequest,
                    .invalidRequest("Content-Length must be a non-negative integer", param: nil))
            }
            if length > configuration.requestLimits.maximumBodyBytes {
                return (
                    .payloadTooLarge,
                    .invalidRequest("Request body exceeds the configured byte limit", param: nil))
            }
        }

        if let requiredBearerToken = configuration.requiredBearerToken {
            let authorization = head.headers["authorization"]
            guard authorization.count == 1,
                let suppliedToken = bearerToken(from: authorization[0]),
                constantTimeEqual(suppliedToken, requiredBearerToken)
            else {
                return (
                    .unauthorized,
                    .invalidRequest("Missing or invalid API key", param: nil))
            }
        }
        return nil
    }

    private func writeError(
        _ error: OpenAIServingError,
        status: HTTPResponseStatus,
        keepAlive: Bool,
        context: ChannelHandlerContext
    ) {
        do {
            let data = try JSONEncoder.openAI.encode(
                OpenAIErrorEnvelope(error: error.openAIError))
            var headers = HTTPHeaders()
            headers.add(name: "content-type", value: "application/json")
            headers.add(name: "content-length", value: "\(data.count)")
            if !keepAlive {
                headers.add(name: "connection", value: "close")
            }
            let head = HTTPResponseHead(
                version: .http1_1,
                status: status,
                headers: headers)
            var body = context.channel.allocator.buffer(capacity: data.count)
            body.writeBytes(data)
            context.write(wrapOutboundOut(.head(head)), promise: nil)
            context.write(wrapOutboundOut(.body(.byteBuffer(body))), promise: nil)
            let completion = context.writeAndFlush(wrapOutboundOut(.end(nil)))
            if !keepAlive {
                let channel = context.channel
                completion.whenComplete { _ in
                    channel.close(promise: nil)
                }
            }
        } catch {
            context.close(promise: nil)
        }
    }

    private func clearTerminalRequestIfNeeded() {
        guard activeControl?.isTerminal == true else {
            return
        }
        activeTask = nil
        activeControl = nil
        activeWritabilityGate = nil
    }

    private func cancelActive(_ reason: ServingCancellationReason) {
        guard let control = activeControl, !control.isTerminal else {
            return
        }
        let task = activeTask
        control.markTerminal()
        Task {
            await control.cancellation.cancel(reason)
            task?.cancel()
        }
    }
}

private extension OpenAIChatCompletionsHTTPHandler {
    enum RunError: Error {
        case backpressureTimeout
        case invalidBackendHandle
        case missingCompletion
        case responseLimitExceeded
        case writeFailure
    }

    static func makeGenerationTask(
        request: OpenAIChatCompletionRequest,
        keepAlive: Bool,
        configuration: ServingHTTPConfiguration,
        backend: any ServingGenerationBackend,
        channel: any Channel,
        control: ServingTransportRequestControl,
        writabilityGate: ServingChannelWritabilityGate
    ) -> Task<Void, Never> {
        Task.detached {
            await runGeneration(
                request: request,
                keepAlive: keepAlive,
                configuration: configuration,
                backend: backend,
                channel: channel,
                control: control,
                writabilityGate: writabilityGate)
        }
    }

    static func runGeneration(
        request: OpenAIChatCompletionRequest,
        keepAlive: Bool,
        configuration: ServingHTTPConfiguration,
        backend: any ServingGenerationBackend,
        channel: any Channel,
        control: ServingTransportRequestControl,
        writabilityGate: ServingChannelWritabilityGate
    ) async {
        var handle: ServingGenerationHandle?
        var responseStarted = false
        defer {
            control.markTerminal()
        }

        do {
            try Task.checkCancellation()
            let started = try await backend.start(request)
            handle = started
            await control.cancellation.attach(started.lease)
            try Task.checkCancellation()
            guard started.model == request.model,
                !started.responseID.isEmpty,
                started.created >= 0,
                await started.lease.activate()
            else {
                throw RunError.invalidBackendHandle
            }

            if request.stream {
                try await waitUntilWritable(
                    writabilityGate,
                    timeout: configuration.backpressureStallTimeout)
                let first = try await started.mailbox.next()
                guard let first else {
                    throw RunError.missingCompletion
                }
                try await writeSSEHead(
                    keepAlive: keepAlive,
                    channel: channel,
                    timeout: configuration.backpressureStallTimeout)
                responseStarted = true
                try await writeRoleChunk(
                    for: started,
                    channel: channel,
                    timeout: configuration.backpressureStallTimeout)
                let completion = try await stream(
                    first: first,
                    handle: started,
                    configuration: configuration,
                    channel: channel,
                    writabilityGate: writabilityGate)
                try await writeSSETerminal(
                    completion,
                    handle: started,
                    channel: channel,
                    writabilityGate: writabilityGate,
                    timeout: configuration.backpressureStallTimeout)
            } else {
                let result = try await collectNonStreaming(
                    handle: started,
                    maximumBytes: configuration.maximumNonStreamingResponseBytes)
                let response = OpenAIChatCompletionResponse(
                    id: started.responseID,
                    created: started.created,
                    model: started.model,
                    content: result.text,
                    finishReason: result.completion.finishReason,
                    usage: result.completion.usage)
                try await writeJSONResponse(
                    response,
                    keepAlive: keepAlive,
                    channel: channel,
                    writabilityGate: writabilityGate,
                    timeout: configuration.backpressureStallTimeout)
                responseStarted = true
            }

            _ = await started.lease.complete()
            control.markTerminal()
            if !keepAlive {
                await close(channel)
            }
        } catch let admissionError as ServingBackendAdmissionError {
            await writeAdmissionFailure(
                admissionError,
                keepAlive: keepAlive,
                channel: channel,
                timeout: configuration.backpressureStallTimeout)
        } catch is CancellationError {
            await control.cancellation.cancel(.clientDisconnected)
            await close(channel)
        } catch let gateError as ServingChannelWritabilityGate.GateError {
            if case .backpressureTimeout = gateError {
                await control.cancellation.cancel(.backpressureTimeout)
            }
            await close(channel)
        } catch let mailboxError as ServingMailboxError {
            switch mailboxError {
            case .cancelled(let reason):
                await control.cancellation.cancel(reason)
                await close(channel)
            case .backend(let message):
                _ = await handle?.lease.fail(message)
                await writeFailureIfPossible(
                    responseStarted: responseStarted,
                    keepAlive: keepAlive,
                    channel: channel,
                    timeout: configuration.backpressureStallTimeout)
            }
        } catch let runError as RunError {
            switch runError {
            case .backpressureTimeout:
                await control.cancellation.cancel(.backpressureTimeout)
                await close(channel)
            case .responseLimitExceeded:
                await control.cancellation.cancel(.responseLimitExceeded)
                await writeFailureIfPossible(
                    responseStarted: responseStarted,
                    keepAlive: keepAlive,
                    channel: channel,
                    timeout: configuration.backpressureStallTimeout)
            case .writeFailure:
                await control.cancellation.cancel(.clientDisconnected)
                await close(channel)
            case .invalidBackendHandle, .missingCompletion:
                _ = await handle?.lease.fail("invalid generation contract")
                await writeFailureIfPossible(
                    responseStarted: responseStarted,
                    keepAlive: keepAlive,
                    channel: channel,
                    timeout: configuration.backpressureStallTimeout)
            }
        } catch {
            _ = await handle?.lease.fail("generation failed")
            await writeFailureIfPossible(
                responseStarted: responseStarted,
                keepAlive: keepAlive,
                channel: channel,
                timeout: configuration.backpressureStallTimeout)
        }
    }

    static func stream(
        first: ServingResponseDelta,
        handle: ServingGenerationHandle,
        configuration: ServingHTTPConfiguration,
        channel: any Channel,
        writabilityGate: ServingChannelWritabilityGate
    ) async throws -> ServingGenerationCompletion {
        var pending: ServingResponseDelta? = first
        var completion: ServingGenerationCompletion?

        while let event = pending {
            switch event {
            case .text(let text):
                guard completion == nil else {
                    throw RunError.missingCompletion
                }
                try await waitUntilWritable(
                    writabilityGate,
                    timeout: configuration.backpressureStallTimeout)
                let chunk = OpenAIChatCompletionChunk(
                    id: handle.responseID,
                    created: handle.created,
                    model: handle.model,
                    index: 0,
                    delta: .init(role: nil, content: text),
                    finishReason: nil)
                try await writeBody(
                    chunk.sseEvent(),
                    channel: channel,
                    timeout: configuration.backpressureStallTimeout)
            case .completion(let value):
                guard completion == nil else {
                    throw RunError.missingCompletion
                }
                completion = value
            }
            pending = try await handle.mailbox.next()
        }

        guard let completion else {
            throw RunError.missingCompletion
        }
        return completion
    }

    static func collectNonStreaming(
        handle: ServingGenerationHandle,
        maximumBytes: Int
    ) async throws -> (text: String, completion: ServingGenerationCompletion) {
        var text = ""
        var byteCount = 0
        var completion: ServingGenerationCompletion?
        while let event = try await handle.mailbox.next() {
            switch event {
            case .text(let delta):
                guard completion == nil else {
                    throw RunError.missingCompletion
                }
                byteCount += delta.utf8.count
                guard byteCount <= maximumBytes else {
                    throw RunError.responseLimitExceeded
                }
                text += delta
            case .completion(let value):
                guard completion == nil else {
                    throw RunError.missingCompletion
                }
                completion = value
            }
        }
        guard let completion else {
            throw RunError.missingCompletion
        }
        return (text, completion)
    }

    static func writeSSEHead(
        keepAlive: Bool,
        channel: any Channel,
        timeout: Duration
    ) async throws {
        var headers = HTTPHeaders()
        headers.add(name: "content-type", value: "text/event-stream")
        headers.add(name: "cache-control", value: "no-cache")
        headers.add(name: "x-accel-buffering", value: "no")
        if !keepAlive {
            headers.add(name: "connection", value: "close")
        }
        let head = HTTPResponseHead(
            version: .http1_1,
            status: .ok,
            headers: headers)
        try await writePart(.head(head), channel: channel, timeout: timeout)
    }

    static func writeRoleChunk(
        for handle: ServingGenerationHandle,
        channel: any Channel,
        timeout: Duration
    ) async throws {
        let chunk = OpenAIChatCompletionChunk(
            id: handle.responseID,
            created: handle.created,
            model: handle.model,
            index: 0,
            delta: .init(role: "assistant", content: nil),
            finishReason: nil)
        try await writeBody(chunk.sseEvent(), channel: channel, timeout: timeout)
    }

    static func writeSSETerminal(
        _ completion: ServingGenerationCompletion,
        handle: ServingGenerationHandle,
        channel: any Channel,
        writabilityGate: ServingChannelWritabilityGate,
        timeout: Duration
    ) async throws {
        try await waitUntilWritable(writabilityGate, timeout: timeout)
        let finish = OpenAIChatCompletionChunk(
            id: handle.responseID,
            created: handle.created,
            model: handle.model,
            index: 0,
            delta: .init(role: nil, content: nil),
            finishReason: completion.finishReason,
            usage: completion.usage)
        try await writeBody(finish.sseEvent(), channel: channel, timeout: timeout)
        try await writeBody(
            OpenAIChatCompletionChunk.doneSSEEvent,
            channel: channel,
            timeout: timeout)
        try await writePart(.end(nil), channel: channel, timeout: timeout)
    }

    static func writeJSONResponse(
        _ response: OpenAIChatCompletionResponse,
        keepAlive: Bool,
        channel: any Channel,
        writabilityGate: ServingChannelWritabilityGate,
        timeout: Duration
    ) async throws {
        try await waitUntilWritable(writabilityGate, timeout: timeout)
        let data = try JSONEncoder.openAI.encode(response)
        var headers = HTTPHeaders()
        headers.add(name: "content-type", value: "application/json")
        headers.add(name: "content-length", value: "\(data.count)")
        if !keepAlive {
            headers.add(name: "connection", value: "close")
        }
        let head = HTTPResponseHead(
            version: .http1_1,
            status: .ok,
            headers: headers)
        try await writePart(.head(head), channel: channel, timeout: timeout)
        var buffer = channel.allocator.buffer(capacity: data.count)
        buffer.writeBytes(data)
        try await writePart(.body(.byteBuffer(buffer)), channel: channel, timeout: timeout)
        try await writePart(.end(nil), channel: channel, timeout: timeout)
    }

    static func waitUntilWritable(
        _ gate: ServingChannelWritabilityGate,
        timeout: Duration
    ) async throws {
        do {
            try await gate.waitUntilWritable(timeout: timeout)
        } catch is ServingChannelWritabilityGate.GateError {
            throw RunError.backpressureTimeout
        }
    }

    static func writeBody(
        _ text: String,
        channel: any Channel,
        timeout: Duration
    ) async throws {
        var buffer = channel.allocator.buffer(capacity: text.utf8.count)
        buffer.writeString(text)
        try await writePart(.body(.byteBuffer(buffer)), channel: channel, timeout: timeout)
    }

    static func writePart(
        _ part: HTTPServerResponsePart,
        channel: any Channel,
        timeout: Duration
    ) async throws {
        let race = ServingWriteCompletionRace()
        let result = channel.eventLoop.makePromise(of: Void.self)
        channel.writeAndFlush(part).whenComplete { writeResult in
            if race.claim() {
                result.completeWith(writeResult)
            }
        }
        let deadline = channel.eventLoop.scheduleTask(
            in: servingNIOTimeAmount(for: timeout)
        ) {
            if race.claim() {
                result.fail(RunError.backpressureTimeout)
            }
        }
        defer {
            deadline.cancel()
        }
        do {
            try await result.futureResult.get()
        } catch let error as RunError {
            throw error
        } catch {
            throw RunError.writeFailure
        }
    }

    static func writeFailureIfPossible(
        responseStarted: Bool,
        keepAlive: Bool,
        channel: any Channel,
        timeout: Duration
    ) async {
        let error = OpenAIErrorEnvelope(
            error: OpenAIServingError.server(
                "Generation failed",
                code: "generation_failed").openAIError)
        do {
            let data = try JSONEncoder.openAI.encode(error)
            if responseStarted {
                let event = "data: \(String(decoding: data, as: UTF8.self))\n\n"
                try await writeBody(event, channel: channel, timeout: timeout)
                try await writePart(.end(nil), channel: channel, timeout: timeout)
                await close(channel)
            } else {
                var headers = HTTPHeaders()
                headers.add(name: "content-type", value: "application/json")
                headers.add(name: "content-length", value: "\(data.count)")
                if !keepAlive {
                    headers.add(name: "connection", value: "close")
                }
                let head = HTTPResponseHead(
                    version: .http1_1,
                    status: .internalServerError,
                    headers: headers)
                try await writePart(.head(head), channel: channel, timeout: timeout)
                var body = channel.allocator.buffer(capacity: data.count)
                body.writeBytes(data)
                try await writePart(
                    .body(.byteBuffer(body)),
                    channel: channel,
                    timeout: timeout)
                try await writePart(.end(nil), channel: channel, timeout: timeout)
                if !keepAlive {
                    await close(channel)
                }
            }
        } catch {
            await close(channel)
        }
    }

    static func writeAdmissionFailure(
        _ admissionError: ServingBackendAdmissionError,
        keepAlive: Bool,
        channel: any Channel,
        timeout: Duration
    ) async {
        let payload: OpenAIErrorPayload
        let status: HTTPResponseStatus
        switch admissionError.reason {
        case .queueFull:
            payload = OpenAIServingError.rateLimited(
                "The serving queue is full",
                code: "queue_full").openAIError
            status = .tooManyRequests
        case .capacityExceeded:
            payload = OpenAIServingError.rateLimited(
                "The loaded model has no remaining request capacity",
                code: "capacity_exhausted").openAIError
            status = .tooManyRequests
        case .requestTooLarge:
            payload = OpenAIServingError.invalidRequest(
                "Request exceeds the loaded model or KV limit",
                param: "messages").openAIError
            status = .badRequest
        }

        do {
            let data = try JSONEncoder.openAI.encode(
                OpenAIErrorEnvelope(error: payload))
            var headers = HTTPHeaders()
            headers.add(name: "content-type", value: "application/json")
            headers.add(name: "content-length", value: "\(data.count)")
            if let retryAfterSeconds = admissionError.retryAfterSeconds {
                headers.add(
                    name: "retry-after",
                    value: "\(retryAfterSeconds)")
            }
            if !keepAlive {
                headers.add(name: "connection", value: "close")
            }
            let head = HTTPResponseHead(
                version: .http1_1,
                status: status,
                headers: headers)
            try await writePart(.head(head), channel: channel, timeout: timeout)
            var body = channel.allocator.buffer(capacity: data.count)
            body.writeBytes(data)
            try await writePart(
                .body(.byteBuffer(body)),
                channel: channel,
                timeout: timeout)
            try await writePart(.end(nil), channel: channel, timeout: timeout)
            if !keepAlive {
                await close(channel)
            }
        } catch {
            await close(channel)
        }
    }

    static func close(_ channel: any Channel) async {
        try? await channel.close().get()
    }
}

private final class ServingTransportRequestControl: Sendable {
    let cancellation = ServingTransportCancellationContext()
    private let terminal = OSAllocatedUnfairLock(initialState: false)

    var isTerminal: Bool {
        terminal.withLock { $0 }
    }

    func markTerminal() {
        terminal.withLock { $0 = true }
    }
}

private final class ServingWriteCompletionRace: Sendable {
    private let completed = OSAllocatedUnfairLock(initialState: false)

    func claim() -> Bool {
        completed.withLock { completed in
            guard !completed else {
                return false
            }
            completed = true
            return true
        }
    }
}

private actor ServingTransportCancellationContext {
    private var cancellationReason: ServingCancellationReason?
    private var lease: ServingRequestLease?

    func attach(_ lease: ServingRequestLease) async {
        if let cancellationReason {
            _ = await lease.cancel(cancellationReason)
            return
        }
        self.lease = lease
    }

    func cancel(_ reason: ServingCancellationReason) async {
        guard cancellationReason == nil else {
            return
        }
        cancellationReason = reason
        _ = await lease?.cancel(reason)
    }
}

private func contentLength(from head: HTTPRequestHead) -> Int? {
    guard let raw = head.headers.first(name: "content-length") else {
        return nil
    }
    return Int(raw)
}

private func bearerToken(from authorization: String) -> String? {
    let parts = authorization.split(
        maxSplits: 1,
        omittingEmptySubsequences: true,
        whereSeparator: \.isWhitespace)
    guard parts.count == 2, parts[0].lowercased() == "bearer" else {
        return nil
    }
    return String(parts[1])
}

private func constantTimeEqual(_ lhs: String, _ rhs: String) -> Bool {
    let left = Array(lhs.utf8)
    let right = Array(rhs.utf8)
    let count = max(left.count, right.count)
    var difference = UInt64(left.count ^ right.count)
    for index in 0..<count {
        let leftByte = index < left.count ? left[index] : 0
        let rightByte = index < right.count ? right[index] : 0
        difference |= UInt64(leftByte ^ rightByte)
    }
    return difference == 0
}
