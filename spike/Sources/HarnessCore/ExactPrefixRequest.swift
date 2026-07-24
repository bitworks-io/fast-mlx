import Foundation

public enum ExactPrefixRequestError:
    Error, Equatable, Sendable
{
    case invalidSemanticIdentity(String)
    case warmupRequiresEnabledCache
}

/// Caller-owned semantic axes for one exact request-start lookup.
///
/// Prompt tokens are compared exactly by `ExactPrefixCache`; this value carries only the
/// privacy/template/tool boundaries that cannot be inferred safely from those token IDs.
/// Loaded-model, tokenizer, KV-route, position, architecture, and drafter identities are added
/// inside the model-owning actor and cannot be supplied by the caller.
public struct ExactPrefixRequestContext:
    Codable, Equatable, Hashable, Sendable
{
    private enum CodingKeys: String, CodingKey {
        case isolationNamespaceSHA256
        case promptTemplateSHA256
        case toolsSHA256
    }

    public let isolationNamespaceSHA256: String
    public let promptTemplateSHA256: String
    public let toolsSHA256: String

    public init(
        isolationNamespaceSHA256: String,
        promptTemplateSHA256: String,
        toolsSHA256: String
    ) throws {
        for (field, value) in [
            ("isolationNamespaceSHA256", isolationNamespaceSHA256),
            ("promptTemplateSHA256", promptTemplateSHA256),
            ("toolsSHA256", toolsSHA256),
        ] where !requestStartIsLowercaseSHA256(value) {
            throw ExactPrefixRequestError
                .invalidSemanticIdentity(field)
        }
        self.isolationNamespaceSHA256 =
            isolationNamespaceSHA256
        self.promptTemplateSHA256 = promptTemplateSHA256
        self.toolsSHA256 = toolsSHA256
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            isolationNamespaceSHA256: values.decode(
                String.self,
                forKey: .isolationNamespaceSHA256),
            promptTemplateSHA256: values.decode(
                String.self,
                forKey: .promptTemplateSHA256),
            toolsSHA256: values.decode(
                String.self,
                forKey: .toolsSHA256))
    }

    public func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(
            isolationNamespaceSHA256,
            forKey: .isolationNamespaceSHA256)
        try values.encode(
            promptTemplateSHA256,
            forKey: .promptTemplateSHA256)
        try values.encode(toolsSHA256, forKey: .toolsSHA256)
    }
}

/// Model-load policy for the actor-owned hot prefix cache. The feature is disabled unless a
/// positive entry and byte budget are supplied explicitly.
public struct ExactPrefixCacheConfiguration:
    Codable, Equatable, Hashable, Sendable
{
    private enum CodingKeys: String, CodingKey {
        case policy
        case eagerWarmupEnabled
    }

    public let policy: ExactPrefixCachePolicy
    public let eagerWarmupEnabled: Bool

    public static let disabled = ExactPrefixCacheConfiguration(
        policy: .disabled,
        eagerWarmupEnabled: false,
        validated: ())

    public init(
        policy: ExactPrefixCachePolicy,
        eagerWarmupEnabled: Bool = false
    ) throws {
        guard policy.isEnabled || !eagerWarmupEnabled else {
            throw ExactPrefixRequestError
                .warmupRequiresEnabledCache
        }
        self.init(
            policy: policy,
            eagerWarmupEnabled: eagerWarmupEnabled,
            validated: ())
    }

    private init(
        policy: ExactPrefixCachePolicy,
        eagerWarmupEnabled: Bool,
        validated: Void
    ) {
        self.policy = policy
        self.eagerWarmupEnabled = eagerWarmupEnabled
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            policy: values.decode(
                ExactPrefixCachePolicy.self,
                forKey: .policy),
            eagerWarmupEnabled: values.decode(
                Bool.self,
                forKey: .eagerWarmupEnabled))
    }

    public func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(policy, forKey: .policy)
        try values.encode(
            eagerWarmupEnabled,
            forKey: .eagerWarmupEnabled)
    }
}
