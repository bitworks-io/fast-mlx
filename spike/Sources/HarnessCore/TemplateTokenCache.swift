import Foundation

public enum TemplateTokenCacheError:
    Error, Equatable, Sendable
{
    case invalidSemanticIdentity(String)
    case invalidPolicy(String)
    case emptyTokenIDs
    case negativeTokenID(position: Int)
    case tokenOutOfInt32Range(position: Int)
    case retainedByteCountOverflow
}

public struct TemplateTokenCacheKey:
    Codable, Equatable, Hashable, Sendable
{
    private enum CodingKeys: String, CodingKey {
        case isolationNamespaceSHA256
        case tokenizerSHA256
        case promptTemplateSHA256
        case toolsSHA256
        case promptContentSHA256
        case formattingOptionsSHA256
    }

    public let isolationNamespaceSHA256: String
    public let tokenizerSHA256: String
    public let promptTemplateSHA256: String
    public let toolsSHA256: String
    public let promptContentSHA256: String
    public let formattingOptionsSHA256: String

    public init(
        isolationNamespaceSHA256: String,
        tokenizerSHA256: String,
        promptTemplateSHA256: String,
        toolsSHA256: String,
        promptContentSHA256: String,
        formattingOptionsSHA256: String
    ) throws {
        let identities = [
            ("isolationNamespaceSHA256", isolationNamespaceSHA256),
            ("tokenizerSHA256", tokenizerSHA256),
            ("promptTemplateSHA256", promptTemplateSHA256),
            ("toolsSHA256", toolsSHA256),
            ("promptContentSHA256", promptContentSHA256),
            ("formattingOptionsSHA256", formattingOptionsSHA256),
        ]
        for (field, value) in identities
        where !requestStartIsLowercaseSHA256(value)
        {
            throw TemplateTokenCacheError.invalidSemanticIdentity(
                field)
        }
        self.isolationNamespaceSHA256 = isolationNamespaceSHA256
        self.tokenizerSHA256 = tokenizerSHA256
        self.promptTemplateSHA256 = promptTemplateSHA256
        self.toolsSHA256 = toolsSHA256
        self.promptContentSHA256 = promptContentSHA256
        self.formattingOptionsSHA256 = formattingOptionsSHA256
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            isolationNamespaceSHA256: values.decode(
                String.self,
                forKey: .isolationNamespaceSHA256),
            tokenizerSHA256: values.decode(
                String.self,
                forKey: .tokenizerSHA256),
            promptTemplateSHA256: values.decode(
                String.self,
                forKey: .promptTemplateSHA256),
            toolsSHA256: values.decode(
                String.self,
                forKey: .toolsSHA256),
            promptContentSHA256: values.decode(
                String.self,
                forKey: .promptContentSHA256),
            formattingOptionsSHA256: values.decode(
                String.self,
                forKey: .formattingOptionsSHA256))
    }

    public func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(
            isolationNamespaceSHA256,
            forKey: .isolationNamespaceSHA256)
        try values.encode(tokenizerSHA256, forKey: .tokenizerSHA256)
        try values.encode(
            promptTemplateSHA256,
            forKey: .promptTemplateSHA256)
        try values.encode(toolsSHA256, forKey: .toolsSHA256)
        try values.encode(
            promptContentSHA256,
            forKey: .promptContentSHA256)
        try values.encode(
            formattingOptionsSHA256,
            forKey: .formattingOptionsSHA256)
    }

    fileprivate var retainedUTF8Bytes: Int {
        isolationNamespaceSHA256.utf8.count
            + tokenizerSHA256.utf8.count
            + promptTemplateSHA256.utf8.count
            + toolsSHA256.utf8.count
            + promptContentSHA256.utf8.count
            + formattingOptionsSHA256.utf8.count
    }
}

public struct TemplateTokenCachePolicy:
    Codable, Equatable, Hashable, Sendable
{
    private enum CodingKeys: String, CodingKey {
        case maxEntries
        case maxRetainedBytes
        case isEnabled
    }

    public let maxEntries: Int
    public let maxRetainedBytes: Int
    public let isEnabled: Bool

    public static let disabled = TemplateTokenCachePolicy(
        maxEntries: 0,
        maxRetainedBytes: 0,
        isEnabled: false)

    public init(
        maxEntries: Int,
        maxRetainedBytes: Int
    ) throws {
        guard maxEntries > 0 else {
            throw TemplateTokenCacheError.invalidPolicy(
                "maxEntries")
        }
        guard maxRetainedBytes > 0 else {
            throw TemplateTokenCacheError.invalidPolicy(
                "maxRetainedBytes")
        }
        self.init(
            maxEntries: maxEntries,
            maxRetainedBytes: maxRetainedBytes,
            isEnabled: true)
    }

    private init(
        maxEntries: Int,
        maxRetainedBytes: Int,
        isEnabled: Bool
    ) {
        self.maxEntries = maxEntries
        self.maxRetainedBytes = maxRetainedBytes
        self.isEnabled = isEnabled
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let maxEntries = try values.decode(
            Int.self,
            forKey: .maxEntries)
        let maxRetainedBytes = try values.decode(
            Int.self,
            forKey: .maxRetainedBytes)
        let isEnabled = try values.decode(
            Bool.self,
            forKey: .isEnabled)
        if isEnabled {
            try self.init(
                maxEntries: maxEntries,
                maxRetainedBytes: maxRetainedBytes)
        } else {
            guard maxEntries == 0, maxRetainedBytes == 0 else {
                throw TemplateTokenCacheError.invalidPolicy(
                    "disabled")
            }
            self = .disabled
        }
    }

    public func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(maxEntries, forKey: .maxEntries)
        try values.encode(
            maxRetainedBytes,
            forKey: .maxRetainedBytes)
        try values.encode(isEnabled, forKey: .isEnabled)
    }
}

public enum TemplateTokenCacheSkipReason:
    String, Codable, Equatable, Sendable
{
    case disabled
    case entryExceedsBudget = "entry-exceeds-budget"
}

public struct TemplateTokenCacheInsertDecision:
    Codable, Equatable, Sendable
{
    public let inserted: Bool
    public let skipReason: TemplateTokenCacheSkipReason?
    public let evictedEntryCount: Int
}

public struct TemplateTokenCacheSnapshot:
    Codable, Equatable, Sendable
{
    public let entryCount: Int
    public let retainedBytes: Int
    public let hitCount: Int
    public let missCount: Int
    public let evictionCount: Int
}

/// Exact host-side cache of the final token IDs produced by template rendering plus tokenization.
/// It stores no prompt text and has an independent LRU/budget from the model-prefix cache.
public struct TemplateTokenCache {
    private static let fixedEntryAccountingBytes = 192

    private struct Entry {
        let id: UInt64
        let key: TemplateTokenCacheKey
        let tokens: [Int32]
        let retainedBytes: Int
        var lastAccess: UInt64
    }

    public let policy: TemplateTokenCachePolicy
    private var entries: [TemplateTokenCacheKey: Entry] = [:]
    private var nextID: UInt64 = 1
    private var accessClock: UInt64 = 0
    private var hitCount = 0
    private var missCount = 0
    private var evictionCount = 0

    public init(policy: TemplateTokenCachePolicy) {
        self.policy = policy
    }

    public static func retainedBytes(
        key: TemplateTokenCacheKey,
        tokenIDs: [Int]
    ) throws -> Int {
        guard !tokenIDs.isEmpty else {
            throw TemplateTokenCacheError.emptyTokenIDs
        }
        do {
            _ = try requestStartTokenLanes(tokenIDs)
            let tokenBytes = try requestStartCheckedMultiply(
                tokenIDs.count,
                MemoryLayout<Int32>.stride)
            var total = try requestStartCheckedAdd(
                tokenBytes,
                key.retainedUTF8Bytes)
            total = try requestStartCheckedAdd(
                total,
                fixedEntryAccountingBytes)
            return total
        } catch let error as ExactPrefixCacheError {
            switch error {
            case .negativeTokenID(let position):
                throw TemplateTokenCacheError
                    .negativeTokenID(position: position)
            case .tokenOutOfInt32Range(let position):
                throw TemplateTokenCacheError
                    .tokenOutOfInt32Range(position: position)
            case .retainedByteCountOverflow:
                throw TemplateTokenCacheError
                    .retainedByteCountOverflow
            default:
                throw TemplateTokenCacheError
                    .retainedByteCountOverflow
            }
        }
    }

    public var snapshot: TemplateTokenCacheSnapshot {
        TemplateTokenCacheSnapshot(
            entryCount: entries.count,
            retainedBytes: entries.values.reduce(0) {
                $0 + $1.retainedBytes
            },
            hitCount: hitCount,
            missCount: missCount,
            evictionCount: evictionCount)
    }

    public mutating func lookup(
        key: TemplateTokenCacheKey
    ) throws -> [Int]? {
        guard policy.isEnabled else {
            missCount += 1
            return nil
        }
        guard var entry = entries[key] else {
            missCount += 1
            return nil
        }
        accessClock &+= 1
        entry.lastAccess = accessClock
        entries[key] = entry
        hitCount += 1
        return entry.tokens.map(Int.init)
    }

    public mutating func insert(
        key: TemplateTokenCacheKey,
        tokenIDs: [Int]
    ) throws -> TemplateTokenCacheInsertDecision {
        guard policy.isEnabled else {
            return TemplateTokenCacheInsertDecision(
                inserted: false,
                skipReason: .disabled,
                evictedEntryCount: 0)
        }
        guard !tokenIDs.isEmpty else {
            throw TemplateTokenCacheError.emptyTokenIDs
        }
        let tokenLanes: [Int32]
        do {
            tokenLanes = try requestStartTokenLanes(tokenIDs)
        } catch let error as ExactPrefixCacheError {
            switch error {
            case .negativeTokenID(let position):
                throw TemplateTokenCacheError
                    .negativeTokenID(position: position)
            case .tokenOutOfInt32Range(let position):
                throw TemplateTokenCacheError
                    .tokenOutOfInt32Range(position: position)
            default:
                throw TemplateTokenCacheError
                    .retainedByteCountOverflow
            }
        }
        let retained = try Self.retainedBytes(
            key: key,
            tokenIDs: tokenIDs)
        guard retained <= policy.maxRetainedBytes else {
            return TemplateTokenCacheInsertDecision(
                inserted: false,
                skipReason: .entryExceedsBudget,
                evictedEntryCount: 0)
        }

        entries.removeValue(forKey: key)
        var evicted = 0
        while true {
            let currentRetainedBytes = snapshot.retainedBytes
            guard currentRetainedBytes
                <= policy.maxRetainedBytes
            else {
                throw TemplateTokenCacheError
                    .retainedByteCountOverflow
            }
            let exceedsEntryLimit =
                entries.count >= policy.maxEntries
            let exceedsByteLimit =
                retained > policy.maxRetainedBytes
                    - currentRetainedBytes
            guard exceedsEntryLimit || exceedsByteLimit else {
                break
            }
            guard let lru = entries.values.min(by: {
                if $0.lastAccess == $1.lastAccess {
                    return $0.id < $1.id
                }
                return $0.lastAccess < $1.lastAccess
            })
            else {
                throw TemplateTokenCacheError
                    .retainedByteCountOverflow
            }
            entries.removeValue(forKey: lru.key)
            evictionCount += 1
            evicted += 1
        }

        accessClock &+= 1
        let entry = Entry(
            id: nextID,
            key: key,
            tokens: tokenLanes,
            retainedBytes: retained,
            lastAccess: accessClock)
        nextID &+= 1
        entries[key] = entry
        return TemplateTokenCacheInsertDecision(
            inserted: true,
            skipReason: nil,
            evictedEntryCount: evicted)
    }
}
