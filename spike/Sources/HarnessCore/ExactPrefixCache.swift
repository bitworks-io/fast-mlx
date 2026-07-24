import Foundation

public enum ExactPrefixCacheError: Error, Equatable, Sendable {
    case invalidSemanticIdentity(String)
    case invalidPolicy(String)
    case invalidSnapshotByteCount(Int)
    case negativeTokenID(position: Int)
    case tokenOutOfInt32Range(position: Int)
    case retainedByteCountOverflow
    case unknownReservation(UInt64)
    case actualSnapshotExceedsReservation(reserved: Int, actual: Int)
}

@inline(__always)
func requestStartIsLowercaseSHA256(_ value: String) -> Bool {
    value.utf8.count == 64
        && value.utf8.allSatisfy {
            (48 ... 57).contains($0) || (97 ... 102).contains($0)
        }
}

@inline(__always)
func requestStartCheckedAdd(_ lhs: Int, _ rhs: Int) throws -> Int {
    let (sum, overflow) = lhs.addingReportingOverflow(rhs)
    guard !overflow else {
        throw ExactPrefixCacheError.retainedByteCountOverflow
    }
    return sum
}

@inline(__always)
func requestStartCheckedMultiply(_ lhs: Int, _ rhs: Int) throws -> Int {
    let (product, overflow) = lhs.multipliedReportingOverflow(by: rhs)
    guard !overflow else {
        throw ExactPrefixCacheError.retainedByteCountOverflow
    }
    return product
}

func requestStartTokenLanes(_ tokens: [Int]) throws -> [Int32] {
    try tokens.enumerated().map { position, token in
        guard token >= 0 else {
            throw ExactPrefixCacheError.negativeTokenID(
                position: position)
        }
        guard let lane = Int32(exactly: token) else {
            throw ExactPrefixCacheError.tokenOutOfInt32Range(
                position: position)
        }
        return lane
    }
}

/// Every field that can change the meaning of one token prefix. Only path-free digests and one
/// ephemeral loaded-model instance identifier are retained; raw prompts, tool schemas, and local
/// checkpoint paths never enter the key.
public struct ExactPrefixSemanticKey:
    Codable, Equatable, Hashable, Sendable
{
    private enum CodingKeys: String, CodingKey {
        case isolationNamespaceSHA256
        case modelInstanceID
        case modelRevisionSHA256
        case tokenizerSHA256
        case promptTemplateSHA256
        case toolsSHA256
        case kvRouteSHA256
        case positionSemanticsSHA256
        case architectureStateSHA256
        case drafterStateSHA256
    }

    public let isolationNamespaceSHA256: String
    public let modelInstanceID: String
    public let modelRevisionSHA256: String
    public let tokenizerSHA256: String
    public let promptTemplateSHA256: String
    public let toolsSHA256: String
    public let kvRouteSHA256: String
    public let positionSemanticsSHA256: String
    public let architectureStateSHA256: String
    public let drafterStateSHA256: String

    public init(
        isolationNamespaceSHA256: String,
        modelInstanceID: String,
        modelRevisionSHA256: String,
        tokenizerSHA256: String,
        promptTemplateSHA256: String,
        toolsSHA256: String,
        kvRouteSHA256: String,
        positionSemanticsSHA256: String,
        architectureStateSHA256: String,
        drafterStateSHA256: String
    ) throws {
        let digests = [
            ("isolationNamespaceSHA256", isolationNamespaceSHA256),
            ("modelRevisionSHA256", modelRevisionSHA256),
            ("tokenizerSHA256", tokenizerSHA256),
            ("promptTemplateSHA256", promptTemplateSHA256),
            ("toolsSHA256", toolsSHA256),
            ("kvRouteSHA256", kvRouteSHA256),
            ("positionSemanticsSHA256", positionSemanticsSHA256),
            ("architectureStateSHA256", architectureStateSHA256),
            ("drafterStateSHA256", drafterStateSHA256),
        ]
        for (field, value) in digests
        where !requestStartIsLowercaseSHA256(value)
        {
            throw ExactPrefixCacheError.invalidSemanticIdentity(field)
        }
        let instanceBytes = modelInstanceID.utf8
        guard !instanceBytes.isEmpty, instanceBytes.count <= 128,
            instanceBytes.allSatisfy({
                (33 ... 126).contains($0) && $0 != 47 && $0 != 92
            })
        else {
            throw ExactPrefixCacheError.invalidSemanticIdentity(
                "modelInstanceID")
        }

        self.isolationNamespaceSHA256 = isolationNamespaceSHA256
        self.modelInstanceID = modelInstanceID
        self.modelRevisionSHA256 = modelRevisionSHA256
        self.tokenizerSHA256 = tokenizerSHA256
        self.promptTemplateSHA256 = promptTemplateSHA256
        self.toolsSHA256 = toolsSHA256
        self.kvRouteSHA256 = kvRouteSHA256
        self.positionSemanticsSHA256 = positionSemanticsSHA256
        self.architectureStateSHA256 = architectureStateSHA256
        self.drafterStateSHA256 = drafterStateSHA256
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            isolationNamespaceSHA256: values.decode(
                String.self,
                forKey: .isolationNamespaceSHA256),
            modelInstanceID: values.decode(
                String.self,
                forKey: .modelInstanceID),
            modelRevisionSHA256: values.decode(
                String.self,
                forKey: .modelRevisionSHA256),
            tokenizerSHA256: values.decode(
                String.self,
                forKey: .tokenizerSHA256),
            promptTemplateSHA256: values.decode(
                String.self,
                forKey: .promptTemplateSHA256),
            toolsSHA256: values.decode(
                String.self,
                forKey: .toolsSHA256),
            kvRouteSHA256: values.decode(
                String.self,
                forKey: .kvRouteSHA256),
            positionSemanticsSHA256: values.decode(
                String.self,
                forKey: .positionSemanticsSHA256),
            architectureStateSHA256: values.decode(
                String.self,
                forKey: .architectureStateSHA256),
            drafterStateSHA256: values.decode(
                String.self,
                forKey: .drafterStateSHA256))
    }

    public func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(
            isolationNamespaceSHA256,
            forKey: .isolationNamespaceSHA256)
        try values.encode(modelInstanceID, forKey: .modelInstanceID)
        try values.encode(
            modelRevisionSHA256,
            forKey: .modelRevisionSHA256)
        try values.encode(tokenizerSHA256, forKey: .tokenizerSHA256)
        try values.encode(
            promptTemplateSHA256,
            forKey: .promptTemplateSHA256)
        try values.encode(toolsSHA256, forKey: .toolsSHA256)
        try values.encode(kvRouteSHA256, forKey: .kvRouteSHA256)
        try values.encode(
            positionSemanticsSHA256,
            forKey: .positionSemanticsSHA256)
        try values.encode(
            architectureStateSHA256,
            forKey: .architectureStateSHA256)
        try values.encode(
            drafterStateSHA256,
            forKey: .drafterStateSHA256)
    }

    fileprivate var retainedUTF8Bytes: Int {
        isolationNamespaceSHA256.utf8.count
            + modelInstanceID.utf8.count
            + modelRevisionSHA256.utf8.count
            + tokenizerSHA256.utf8.count
            + promptTemplateSHA256.utf8.count
            + toolsSHA256.utf8.count
            + kvRouteSHA256.utf8.count
            + positionSemanticsSHA256.utf8.count
            + architectureStateSHA256.utf8.count
            + drafterStateSHA256.utf8.count
    }
}

public struct ExactPrefixCachePolicy:
    Codable, Equatable, Hashable, Sendable
{
    private enum CodingKeys: String, CodingKey {
        case maxEntries
        case maxRetainedBytes
        case minimumReusableTokens
        case isEnabled
    }

    public let maxEntries: Int
    public let maxRetainedBytes: Int
    public let minimumReusableTokens: Int
    public let isEnabled: Bool

    public static let disabled = ExactPrefixCachePolicy(
        maxEntries: 0,
        maxRetainedBytes: 0,
        minimumReusableTokens: 0,
        isEnabled: false)

    public init(
        maxEntries: Int,
        maxRetainedBytes: Int,
        minimumReusableTokens: Int
    ) throws {
        guard maxEntries > 0 else {
            throw ExactPrefixCacheError.invalidPolicy("maxEntries")
        }
        guard maxRetainedBytes > 0 else {
            throw ExactPrefixCacheError.invalidPolicy(
                "maxRetainedBytes")
        }
        guard minimumReusableTokens > 0 else {
            throw ExactPrefixCacheError.invalidPolicy(
                "minimumReusableTokens")
        }
        self.init(
            maxEntries: maxEntries,
            maxRetainedBytes: maxRetainedBytes,
            minimumReusableTokens: minimumReusableTokens,
            isEnabled: true)
    }

    private init(
        maxEntries: Int,
        maxRetainedBytes: Int,
        minimumReusableTokens: Int,
        isEnabled: Bool
    ) {
        self.maxEntries = maxEntries
        self.maxRetainedBytes = maxRetainedBytes
        self.minimumReusableTokens = minimumReusableTokens
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
        let minimumReusableTokens = try values.decode(
            Int.self,
            forKey: .minimumReusableTokens)
        let isEnabled = try values.decode(
            Bool.self,
            forKey: .isEnabled)
        if isEnabled {
            try self.init(
                maxEntries: maxEntries,
                maxRetainedBytes: maxRetainedBytes,
                minimumReusableTokens: minimumReusableTokens)
        } else {
            guard maxEntries == 0, maxRetainedBytes == 0,
                minimumReusableTokens == 0
            else {
                throw ExactPrefixCacheError.invalidPolicy(
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
        try values.encode(
            minimumReusableTokens,
            forKey: .minimumReusableTokens)
        try values.encode(isEnabled, forKey: .isEnabled)
    }
}

public enum ExactPrefixCommitDisposition: Equatable, Sendable {
    case successfulText(
        generatedTokenCount: Int,
        visibleTokenCount: Int)
    case failed
    case cancelled
    case media

    fileprivate var skipReason: ExactPrefixCommitSkipReason? {
        switch self {
        case .failed:
            return .generationFailed
        case .cancelled:
            return .cancelled
        case .media:
            return .unsupportedMedia
        case .successfulText(
            let generatedTokenCount,
            let visibleTokenCount):
            guard generatedTokenCount > 0 else {
                return .zeroGeneratedTokens
            }
            guard visibleTokenCount > 0 else {
                return .padOnlyOutput
            }
            return nil
        }
    }
}

public enum ExactPrefixCommitSkipReason:
    String, Codable, CaseIterable, Equatable, Hashable, Sendable
{
    case disabled
    case prefixTooShort = "prefix-too-short"
    case snapshotExceedsBudget = "snapshot-exceeds-budget"
    case reservationCapacityExhausted = "reservation-capacity-exhausted"
    case snapshotCaptureFailed = "snapshot-capture-failed"
    case snapshotRestoreFailed = "snapshot-restore-failed"
    case snapshotEvidenceMismatch = "snapshot-evidence-mismatch"
    case generationFailed = "generation-failed"
    case cancelled
    case unsupportedMedia = "unsupported-media"
    case zeroGeneratedTokens = "zero-generated-tokens"
    case padOnlyOutput = "pad-only-output"
}

public struct ExactPrefixReservation: Equatable, Hashable, Sendable {
    fileprivate let id: UInt64
    public let reservedSnapshotBytes: Int
    public let reservedRetainedBytes: Int

    fileprivate init(
        id: UInt64,
        reservedSnapshotBytes: Int,
        reservedRetainedBytes: Int
    ) {
        self.id = id
        self.reservedSnapshotBytes = reservedSnapshotBytes
        self.reservedRetainedBytes = reservedRetainedBytes
    }
}

public struct ExactPrefixReservationDecision:
    Equatable, Sendable
{
    public let reservation: ExactPrefixReservation?
    public let skipReason: ExactPrefixCommitSkipReason?
    public let evictedEntryIDs: [UInt64]

    fileprivate init(
        reservation: ExactPrefixReservation?,
        skipReason: ExactPrefixCommitSkipReason?,
        evictedEntryIDs: [UInt64] = []
    ) {
        self.reservation = reservation
        self.skipReason = skipReason
        self.evictedEntryIDs = evictedEntryIDs
    }
}

public struct ExactPrefixCommitDecision: Equatable, Sendable {
    public let entryID: UInt64?
    public let skipReason: ExactPrefixCommitSkipReason?
    public let evictedEntryIDs: [UInt64]

    fileprivate init(
        entryID: UInt64?,
        skipReason: ExactPrefixCommitSkipReason?,
        evictedEntryIDs: [UInt64]
    ) {
        self.entryID = entryID
        self.skipReason = skipReason
        self.evictedEntryIDs = evictedEntryIDs
    }
}

public struct ExactPrefixCacheSnapshot:
    Codable, Equatable, Sendable
{
    public let entryCount: Int
    public let reservationCount: Int
    public let retainedBytes: Int
    public let reservedBytes: Int
    public let hitCount: Int
    public let missCount: Int
    public let evictionCount: Int
}

public struct ExactPrefixCacheHit<State> {
    public let entryID: UInt64
    public let prefixTokenCount: Int
    public let retainedBytes: Int
    public let state: State
}

/// Pure policy/index plane for actor-confined prefix snapshots. `State` is intentionally not
/// constrained to `Sendable`: real MLX payloads stay inside the inference actor, while off-box
/// tests use ordinary fake values.
public struct ExactPrefixCache<State> {
    private static var fixedEntryAccountingBytes: Int { 256 }

    private struct Entry {
        let id: UInt64
        let key: ExactPrefixSemanticKey
        let tokens: [Int32]
        let snapshotBytes: Int
        let retainedBytes: Int
        var lastAccess: UInt64
        let state: State
    }

    private struct PendingReservation {
        let reservation: ExactPrefixReservation
        let key: ExactPrefixSemanticKey
        let tokens: [Int32]
        let replacementEntryIDs: Set<UInt64>
        let evictedEntryIDs: [UInt64]
    }

    public let policy: ExactPrefixCachePolicy
    private var entries: [UInt64: Entry] = [:]
    private var entryIDsByKey: [
        ExactPrefixSemanticKey: Set<UInt64>
    ] = [:]
    private var reservations: [UInt64: PendingReservation] = [:]
    private var nextID: UInt64 = 1
    private var accessClock: UInt64 = 0
    private var hitCount = 0
    private var missCount = 0
    private var evictionCount = 0

    public init(policy: ExactPrefixCachePolicy) {
        self.policy = policy
    }

    public static func retainedBytes(
        key: ExactPrefixSemanticKey,
        tokens: [Int],
        snapshotBytes: Int
    ) throws -> Int {
        guard snapshotBytes > 0 else {
            throw ExactPrefixCacheError.invalidSnapshotByteCount(
                snapshotBytes)
        }
        _ = try requestStartTokenLanes(tokens)
        let tokenBytes = try requestStartCheckedMultiply(
            tokens.count,
            MemoryLayout<Int32>.stride)
        var total = try requestStartCheckedAdd(
            snapshotBytes,
            tokenBytes)
        total = try requestStartCheckedAdd(
            total,
            key.retainedUTF8Bytes)
        total = try requestStartCheckedAdd(
            total,
            fixedEntryAccountingBytes)
        return total
    }

    public var snapshot: ExactPrefixCacheSnapshot {
        ExactPrefixCacheSnapshot(
            entryCount: entries.count,
            reservationCount: reservations.count,
            retainedBytes: entries.values.reduce(0) {
                $0 + $1.retainedBytes
            },
            reservedBytes: reservations.values.reduce(0) {
                $0 + $1.reservation.reservedRetainedBytes
            },
            hitCount: hitCount,
            missCount: missCount,
            evictionCount: evictionCount)
    }

    public mutating func lookup(
        key: ExactPrefixSemanticKey,
        promptTokens: [Int]
    ) throws -> ExactPrefixCacheHit<State>? {
        guard policy.isEnabled else {
            missCount += 1
            return nil
        }
        let prompt = try requestStartTokenLanes(promptTokens)
        guard prompt.count >= policy.minimumReusableTokens,
            let candidateIDs = entryIDsByKey[key]
        else {
            missCount += 1
            return nil
        }

        var best: Entry?
        for id in candidateIDs {
            guard let entry = entries[id],
                entry.tokens.count <= prompt.count,
                prompt.starts(with: entry.tokens)
            else { continue }
            if let current = best {
                if entry.tokens.count < current.tokens.count {
                    continue
                }
                if entry.tokens.count == current.tokens.count,
                    (
                        entry.lastAccess < current.lastAccess
                            || (
                                entry.lastAccess == current.lastAccess
                                    && entry.id < current.id
                            )
                    )
                {
                    continue
                }
            }
            best = entry
        }

        guard var hit = best else {
            missCount += 1
            return nil
        }
        accessClock &+= 1
        hit.lastAccess = accessClock
        entries[hit.id] = hit
        hitCount += 1
        return ExactPrefixCacheHit(
            entryID: hit.id,
            prefixTokenCount: hit.tokens.count,
            retainedBytes: hit.retainedBytes,
            state: hit.state)
    }

    public mutating func reserve(
        key: ExactPrefixSemanticKey,
        tokens: [Int],
        snapshotBytes: Int,
        protectingEntryIDs: Set<UInt64> = []
    ) throws -> ExactPrefixReservationDecision {
        guard policy.isEnabled else {
            return ExactPrefixReservationDecision(
                reservation: nil,
                skipReason: .disabled)
        }
        guard snapshotBytes > 0 else {
            throw ExactPrefixCacheError.invalidSnapshotByteCount(
                snapshotBytes)
        }
        let tokenLanes = try requestStartTokenLanes(tokens)
        guard tokenLanes.count >= policy.minimumReusableTokens else {
            return ExactPrefixReservationDecision(
                reservation: nil,
                skipReason: .prefixTooShort)
        }
        let totalBytes = try Self.retainedBytes(
            key: key,
            tokens: tokens,
            snapshotBytes: snapshotBytes)
        guard totalBytes <= policy.maxRetainedBytes else {
            return ExactPrefixReservationDecision(
                reservation: nil,
                skipReason: .snapshotExceedsBudget)
        }

        let replacementEntryIDs = Set(
            entryIDsByKey[key, default: []].filter {
                entries[$0]?.tokens == tokenLanes
            })
        let pendingReplacementEntryIDs = Set(
            reservations.values.flatMap(\.replacementEntryIDs))
        guard replacementEntryIDs.isDisjoint(
            with: pendingReplacementEntryIDs)
        else {
            return ExactPrefixReservationDecision(
                reservation: nil,
                skipReason: .reservationCapacityExhausted)
        }

        let protectedEntryIDs = replacementEntryIDs
            .union(pendingReplacementEntryIDs)
            .union(protectingEntryIDs)
        var plannedEvictions: [UInt64] = []
        var plannedEvictionIDs: Set<UInt64> = []

        while entries.count - replacementEntryIDs.count
            - plannedEvictions.count + reservations.count + 1
            > policy.maxEntries
        {
            guard let id = leastRecentlyUsedEntryID(
                excluding: protectedEntryIDs.union(
                    plannedEvictionIDs))
            else {
                return ExactPrefixReservationDecision(
                    reservation: nil,
                    skipReason: .reservationCapacityExhausted)
            }
            plannedEvictions.append(id)
            plannedEvictionIDs.insert(id)
        }

        while true {
            let committedAndReserved = try requestStartCheckedAdd(
                snapshot.retainedBytes,
                snapshot.reservedBytes)
            let plannedBytes = try plannedEvictions.reduce(0) {
                try requestStartCheckedAdd(
                    $0,
                    entries[$1]?.retainedBytes ?? 0)
            }
            guard plannedBytes <= committedAndReserved else {
                throw ExactPrefixCacheError
                    .retainedByteCountOverflow
            }
            let retainedAfterEvictions =
                committedAndReserved - plannedBytes
            guard retainedAfterEvictions
                <= policy.maxRetainedBytes
            else {
                throw ExactPrefixCacheError
                    .retainedByteCountOverflow
            }
            if totalBytes
                <= policy.maxRetainedBytes
                    - retainedAfterEvictions
            {
                break
            }
            guard let id = leastRecentlyUsedEntryID(
                excluding: protectedEntryIDs.union(
                    plannedEvictionIDs))
            else {
                return ExactPrefixReservationDecision(
                    reservation: nil,
                    skipReason: .reservationCapacityExhausted)
            }
            plannedEvictions.append(id)
            plannedEvictionIDs.insert(id)
        }

        // Evict before the caller allocates the detached MLX snapshot so committed plus reserved
        // bytes never exceed the hard budget. Rollback therefore releases only this reservation;
        // it cannot recreate evicted payloads without retaining a second over-budget copy.
        for id in plannedEvictions {
            removeEntry(id)
        }

        let id = nextID
        nextID &+= 1
        let reservation = ExactPrefixReservation(
            id: id,
            reservedSnapshotBytes: snapshotBytes,
            reservedRetainedBytes: totalBytes)
        reservations[id] = PendingReservation(
            reservation: reservation,
            key: key,
            tokens: tokenLanes,
            replacementEntryIDs: replacementEntryIDs,
            evictedEntryIDs: plannedEvictions)
        return ExactPrefixReservationDecision(
            reservation: reservation,
            skipReason: nil,
            evictedEntryIDs: plannedEvictions)
    }

    public mutating func commit(
        _ reservation: ExactPrefixReservation,
        state: State,
        actualSnapshotBytes: Int,
        disposition: ExactPrefixCommitDisposition
    ) throws -> ExactPrefixCommitDecision {
        guard let pending = reservations.removeValue(
            forKey: reservation.id)
        else {
            throw ExactPrefixCacheError.unknownReservation(
                reservation.id)
        }
        guard actualSnapshotBytes > 0 else {
            throw ExactPrefixCacheError.invalidSnapshotByteCount(
                actualSnapshotBytes)
        }
        guard actualSnapshotBytes
            <= pending.reservation.reservedSnapshotBytes
        else {
            throw ExactPrefixCacheError
                .actualSnapshotExceedsReservation(
                    reserved:
                        pending.reservation.reservedSnapshotBytes,
                    actual: actualSnapshotBytes)
        }
        if let reason = disposition.skipReason {
            return ExactPrefixCommitDecision(
                entryID: nil,
                skipReason: reason,
                evictedEntryIDs: pending.evictedEntryIDs)
        }

        let actualRetainedBytes = try Self.retainedBytes(
            key: pending.key,
            tokens: pending.tokens.map(Int.init),
            snapshotBytes: actualSnapshotBytes)
        guard actualRetainedBytes
            <= pending.reservation.reservedRetainedBytes
        else {
            throw ExactPrefixCacheError
                .actualSnapshotExceedsReservation(
                    reserved:
                        pending.reservation.reservedSnapshotBytes,
                    actual: actualSnapshotBytes)
        }

        for id in pending.replacementEntryIDs {
            removeEntry(id, countAsEviction: false)
        }
        accessClock &+= 1
        let entry = Entry(
            id: reservation.id,
            key: pending.key,
            tokens: pending.tokens,
            snapshotBytes: actualSnapshotBytes,
            retainedBytes: actualRetainedBytes,
            lastAccess: accessClock,
            state: state)
        entries[entry.id] = entry
        entryIDsByKey[entry.key, default: []].insert(entry.id)
        return ExactPrefixCommitDecision(
            entryID: entry.id,
            skipReason: nil,
            evictedEntryIDs: pending.evictedEntryIDs)
    }

    public mutating func rollback(
        _ reservation: ExactPrefixReservation
    ) throws {
        // Pre-allocation evictions are intentionally durable. See `reserve`: retaining rollback
        // copies would defeat the byte budget the reservation exists to enforce.
        guard reservations.removeValue(forKey: reservation.id) != nil
        else {
            throw ExactPrefixCacheError.unknownReservation(
                reservation.id)
        }
    }

    /// Remove a selected entry after its opaque payload fails actor-side validation or restore.
    ///
    /// The entry ID comes from `lookup`, so callers can quarantine exactly the broken snapshot
    /// without exposing semantic keys or prompt tokens. A repeated invalidation is a no-op. The
    /// removal is counted as an eviction because it releases retained payload bytes and is visible
    /// in the same bounded-cache telemetry as policy-driven eviction.
    @discardableResult
    public mutating func invalidate(entryID: UInt64) -> Bool {
        guard entries[entryID] != nil else {
            return false
        }
        removeEntry(entryID)
        return true
    }

    private func leastRecentlyUsedEntryID(
        excluding excludedIDs: Set<UInt64> = []
    ) -> UInt64? {
        entries.values.filter {
            !excludedIDs.contains($0.id)
        }.min {
            if $0.lastAccess == $1.lastAccess {
                return $0.id < $1.id
            }
            return $0.lastAccess < $1.lastAccess
        }?.id
    }

    private mutating func removeEntry(
        _ id: UInt64,
        countAsEviction: Bool = true
    ) {
        guard let entry = entries.removeValue(forKey: id) else {
            return
        }
        entryIDsByKey[entry.key]?.remove(id)
        if entryIDsByKey[entry.key]?.isEmpty == true {
            entryIDsByKey.removeValue(forKey: entry.key)
        }
        if countAsEviction {
            evictionCount += 1
        }
    }
}
