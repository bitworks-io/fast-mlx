import Foundation
import ServingCore

public struct ServingHTTPConfiguration: Sendable {
    public let launchedModel: String
    public let requestLimits: OpenAIChatRequestLimits
    public let requiredBearerToken: String?
    public let maximumNonStreamingResponseBytes: Int
    public let backpressureStallTimeout: Duration

    public init(
        launchedModel: String,
        requestLimits: OpenAIChatRequestLimits = .productionDefault,
        requiredBearerToken: String?,
        maximumNonStreamingResponseBytes: Int,
        backpressureStallTimeout: Duration
    ) {
        precondition(
            !launchedModel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            "launchedModel must be non-empty")
        precondition(
            maximumNonStreamingResponseBytes > 0,
            "maximumNonStreamingResponseBytes must be positive")
        precondition(
            backpressureStallTimeout > .zero,
            "backpressureStallTimeout must be positive")
        if let requiredBearerToken {
            precondition(!requiredBearerToken.isEmpty, "requiredBearerToken must be non-empty")
        }

        self.launchedModel = launchedModel
        self.requestLimits = requestLimits
        self.requiredBearerToken = requiredBearerToken
        self.maximumNonStreamingResponseBytes = maximumNonStreamingResponseBytes
        self.backpressureStallTimeout = backpressureStallTimeout
    }
}
