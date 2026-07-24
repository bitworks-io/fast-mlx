import Foundation

public protocol ServingGenerationBackend: Sendable {
    func start(_ request: OpenAIChatCompletionRequest) async throws -> ServingGenerationHandle
}

public struct ServingGenerationHandle: Sendable {
    public let responseID: String
    public let created: Int
    public let model: String
    public let route: ServingExecutionRoute
    public let mailbox: BoundedDeltaMailbox
    public let lease: ServingRequestLease

    public init(
        responseID: String,
        created: Int,
        model: String,
        route: ServingExecutionRoute,
        mailbox: BoundedDeltaMailbox,
        lease: ServingRequestLease
    ) {
        self.responseID = responseID
        self.created = created
        self.model = model
        self.route = route
        self.mailbox = mailbox
        self.lease = lease
    }
}

public struct ServingGenerationCompletion: Equatable, Sendable {
    public let finishReason: OpenAIChatFinishReason
    public let usage: OpenAIChatUsage

    public init(
        finishReason: OpenAIChatFinishReason,
        usage: OpenAIChatUsage
    ) {
        self.finishReason = finishReason
        self.usage = usage
    }
}
