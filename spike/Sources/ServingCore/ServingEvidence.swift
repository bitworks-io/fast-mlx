import CryptoKit
import Foundation

public struct ServingEvidence: Codable, Equatable, Sendable {
    public static let currentSchemaVersion = 2

    public let schemaVersion: Int
    public let request: Request
    public let response: Response
    public let route: RouteFacts?
    public let cancellation: CancellationFacts?
    public let resources: ResourceFacts?

    public init(
        schemaVersion: Int = ServingEvidence.currentSchemaVersion,
        request: Request,
        response: Response,
        route: RouteFacts? = nil,
        cancellation: CancellationFacts? = nil,
        resources: ResourceFacts? = nil
    ) throws {
        guard schemaVersion == ServingEvidence.currentSchemaVersion else {
            throw ValidationError.unsupportedSchemaVersion(schemaVersion)
        }
        self.schemaVersion = schemaVersion
        self.request = request
        self.response = response
        self.route = route
        self.cancellation = cancellation
        self.resources = resources
    }

    private enum CodingKeys: String, CodingKey, CaseIterable {
        case schemaVersion = "schema_version"
        case request
        case response
        case route
        case cancellation
        case resources
    }

    public init(from decoder: Decoder) throws {
        try ServingEvidence.rejectUnknownKeys(
            from: decoder,
            allowed: CodingKeys.allCases.map(\.rawValue))
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            schemaVersion: container.decode(Int.self, forKey: .schemaVersion),
            request: container.decode(Request.self, forKey: .request),
            response: container.decode(Response.self, forKey: .response),
            route: ServingEvidence.decodeCanonicalOptional(
                RouteFacts.self,
                from: container,
                forKey: .route,
                field: "route"),
            cancellation: ServingEvidence.decodeCanonicalOptional(
                CancellationFacts.self,
                from: container,
                forKey: .cancellation,
                field: "cancellation"),
            resources: ServingEvidence.decodeCanonicalOptional(
                ResourceFacts.self,
                from: container,
                forKey: .resources,
                field: "resources"))
    }

    public func canonicalJSONData() throws -> Data {
        try ServingEvidence.canonicalJSONEncoder.encode(self)
    }

    public static func decodeCanonicalJSONData(_ data: Data) throws -> Self {
        let evidence = try JSONDecoder().decode(Self.self, from: data)
        guard try evidence.canonicalJSONData() == data else {
            throw ValidationError.noncanonicalField("document")
        }
        return evidence
    }

    public static var canonicalJSONEncoder: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }
}

extension ServingEvidence {
    public struct Request: Codable, Equatable, Sendable {
        public let method: String
        public let path: String
        public let headers: [Header]
        public let authorizationPresent: Bool
        public let bodyBytes: Int
        public let bodySHA256: String
        public let stream: Bool
        public let messageCount: Int
        public let maxCompletionTokens: Int?

        public init(
            method: String,
            path: String,
            headers: [Header],
            body: Data,
            stream: Bool,
            messageCount: Int,
            maxCompletionTokens: Int?
        ) throws {
            try ServingEvidence.validateCanonicalHTTPRequest(
                method: method,
                path: path)
            try ServingEvidence.validateNonNegative(body.count, field: "bodyBytes")
            try ServingEvidence.validateNonNegative(messageCount, field: "messageCount")
            if let maxCompletionTokens {
                try ServingEvidence.validateNonNegative(maxCompletionTokens, field: "maxCompletionTokens")
            }

            self.method = method
            self.path = path
            self.headers = Header.redacted(headers)
            self.authorizationPresent = headers.contains { $0.normalizedName == "authorization" }
            self.bodyBytes = body.count
            self.bodySHA256 = SHA256.hexDigest(of: body)
            self.stream = stream
            self.messageCount = messageCount
            self.maxCompletionTokens = maxCompletionTokens
        }

        private init(
            method: String,
            path: String,
            headers: [Header],
            authorizationPresent: Bool,
            bodyBytes: Int,
            bodySHA256: String,
            stream: Bool,
            messageCount: Int,
            maxCompletionTokens: Int?
        ) throws {
            try ServingEvidence.validateCanonicalHTTPRequest(
                method: method,
                path: path)
            try ServingEvidence.validateNonNegative(bodyBytes, field: "bodyBytes")
            try ServingEvidence.validateNonNegative(messageCount, field: "messageCount")
            if let maxCompletionTokens {
                try ServingEvidence.validateNonNegative(maxCompletionTokens, field: "maxCompletionTokens")
            }
            try ServingEvidence.validateSHA256Hex(bodySHA256, field: "bodySHA256")
            guard headers == Header.redacted(headers) else {
                throw ValidationError.noncanonicalField("headers")
            }

            self.method = method
            self.path = path
            self.headers = headers
            self.authorizationPresent = authorizationPresent
            self.bodyBytes = bodyBytes
            self.bodySHA256 = bodySHA256
            self.stream = stream
            self.messageCount = messageCount
            self.maxCompletionTokens = maxCompletionTokens
        }

        private enum CodingKeys: String, CodingKey, CaseIterable {
            case method
            case path
            case headers
            case authorizationPresent = "authorization_present"
            case bodyBytes = "body_bytes"
            case bodySHA256 = "body_sha256"
            case stream
            case messageCount = "message_count"
            case maxCompletionTokens = "max_completion_tokens"
        }

        public init(from decoder: Decoder) throws {
            try ServingEvidence.rejectUnknownKeys(
                from: decoder,
                allowed: CodingKeys.allCases.map(\.rawValue))
            let container = try decoder.container(keyedBy: CodingKeys.self)
            try self.init(
                method: container.decode(String.self, forKey: .method),
                path: container.decode(String.self, forKey: .path),
                headers: container.decode([Header].self, forKey: .headers),
                authorizationPresent: container.decode(Bool.self, forKey: .authorizationPresent),
                bodyBytes: container.decode(Int.self, forKey: .bodyBytes),
                bodySHA256: container.decode(String.self, forKey: .bodySHA256),
                stream: container.decode(Bool.self, forKey: .stream),
                messageCount: container.decode(Int.self, forKey: .messageCount),
                maxCompletionTokens: ServingEvidence.decodeCanonicalOptional(
                    Int.self,
                    from: container,
                    forKey: .maxCompletionTokens,
                    field: "maxCompletionTokens"))
        }
    }

    public struct Response: Codable, Equatable, Sendable {
        public let status: Int?
        public let completed: Bool
        public let durationMilliseconds: Double
        public let chunkCount: Int
        public let bodyBytes: Int
        public let bodySHA256: String

        public init(
            status: Int,
            durationMilliseconds: Double,
            chunkCount: Int,
            body: Data
        ) throws {
            try ServingEvidence.validateHTTPStatus(status)
            try ServingEvidence.validateFiniteNonNegative(durationMilliseconds, field: "durationMilliseconds")
            try ServingEvidence.validateNonNegative(chunkCount, field: "chunkCount")
            try ServingEvidence.validateNonNegative(body.count, field: "bodyBytes")

            self.status = status
            self.completed = true
            self.durationMilliseconds = durationMilliseconds
            self.chunkCount = chunkCount
            self.bodyBytes = body.count
            self.bodySHA256 = SHA256.hexDigest(of: body)
        }

        public init(
            status: Int,
            durationMilliseconds: Double,
            chunkCount: Int,
            bodyBytes: Int,
            bodySHA256: String
        ) throws {
            try ServingEvidence.validateHTTPStatus(status)
            try ServingEvidence.validateFiniteNonNegative(durationMilliseconds, field: "durationMilliseconds")
            try ServingEvidence.validateNonNegative(chunkCount, field: "chunkCount")
            try ServingEvidence.validateNonNegative(bodyBytes, field: "bodyBytes")
            try ServingEvidence.validateSHA256Hex(bodySHA256, field: "bodySHA256")

            self.status = status
            self.completed = true
            self.durationMilliseconds = durationMilliseconds
            self.chunkCount = chunkCount
            self.bodyBytes = bodyBytes
            self.bodySHA256 = bodySHA256
        }

        public init(
            status: Int?,
            completed: Bool,
            durationMilliseconds: Double,
            chunkCount: Int,
            bodyBytes: Int,
            bodySHA256: String
        ) throws {
            if let status {
                try ServingEvidence.validateHTTPStatus(status)
            }
            guard !completed || status != nil else {
                throw ValidationError.noncanonicalField("response.status")
            }
            try ServingEvidence.validateFiniteNonNegative(durationMilliseconds, field: "durationMilliseconds")
            try ServingEvidence.validateNonNegative(chunkCount, field: "chunkCount")
            try ServingEvidence.validateNonNegative(bodyBytes, field: "bodyBytes")
            try ServingEvidence.validateSHA256Hex(bodySHA256, field: "bodySHA256")

            self.status = status
            self.completed = completed
            self.durationMilliseconds = durationMilliseconds
            self.chunkCount = chunkCount
            self.bodyBytes = bodyBytes
            self.bodySHA256 = bodySHA256
        }

        private enum CodingKeys: String, CodingKey, CaseIterable {
            case status
            case completed
            case durationMilliseconds = "duration_milliseconds"
            case chunkCount = "chunk_count"
            case bodyBytes = "body_bytes"
            case bodySHA256 = "body_sha256"
        }

        public init(from decoder: Decoder) throws {
            try ServingEvidence.rejectUnknownKeys(
                from: decoder,
                allowed: CodingKeys.allCases.map(\.rawValue))
            let container = try decoder.container(keyedBy: CodingKeys.self)
            try self.init(
                status: ServingEvidence.decodeCanonicalOptional(
                    Int.self,
                    from: container,
                    forKey: .status,
                    field: "response.status"),
                completed: container.decode(Bool.self, forKey: .completed),
                durationMilliseconds: container.decode(Double.self, forKey: .durationMilliseconds),
                chunkCount: container.decode(Int.self, forKey: .chunkCount),
                bodyBytes: container.decode(Int.self, forKey: .bodyBytes),
                bodySHA256: container.decode(String.self, forKey: .bodySHA256))
        }
    }

    public struct Header: Codable, Equatable, Sendable {
        public let name: String
        public let value: String

        public init(name: String, value: String) {
            self.name = name
            self.value = value
        }

        fileprivate var normalizedName: String {
            name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        }

        fileprivate static func redacted(_ headers: [Header]) -> [Header] {
            headers
                .compactMap { header -> Header? in
                    let name = header.normalizedName
                    guard let value = canonicalValue(for: name) else {
                        return nil
                    }
                    return Header(name: name, value: value)
                }
                .sorted {
                    if $0.name == $1.name {
                        return $0.value < $1.value
                    }
                    return $0.name < $1.name
                }
        }

        private static let allowedNames: Set<String> = [
            "accept",
            "content-type",
        ]

        private static func canonicalValue(for name: String) -> String? {
            guard allowedNames.contains(name) else {
                return nil
            }
            switch name {
            case "accept":
                return "present"
            case "content-type":
                return "application/json"
            default:
                return nil
            }
        }

        private enum CodingKeys: String, CodingKey, CaseIterable {
            case name
            case value
        }

        public init(from decoder: Decoder) throws {
            try ServingEvidence.rejectUnknownKeys(
                from: decoder,
                allowed: CodingKeys.allCases.map(\.rawValue))
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let name = try container.decode(String.self, forKey: .name)
            let value = try container.decode(String.self, forKey: .value)
            guard name == name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
                value == Self.canonicalValue(for: name)
            else {
                throw ValidationError.noncanonicalField("headers")
            }
            self.init(name: name, value: value)
        }
    }

    public struct RouteFacts: Codable, Equatable, Sendable {
        public let kind: ServingExecutionRoute
        public let batchSize: Int?

        public init(kind: ServingExecutionRoute, batchSize: Int? = nil) throws {
            if let batchSize {
                try ServingEvidence.validatePositive(batchSize, field: "batchSize")
            }
            self.kind = kind
            self.batchSize = batchSize
        }

        private enum CodingKeys: String, CodingKey, CaseIterable {
            case kind
            case batchSize = "batch_size"
        }

        public init(from decoder: Decoder) throws {
            try ServingEvidence.rejectUnknownKeys(
                from: decoder,
                allowed: CodingKeys.allCases.map(\.rawValue))
            let container = try decoder.container(keyedBy: CodingKeys.self)
            try self.init(
                kind: container.decode(ServingExecutionRoute.self, forKey: .kind),
                batchSize: ServingEvidence.decodeCanonicalOptional(
                    Int.self,
                    from: container,
                    forKey: .batchSize,
                    field: "batchSize"))
        }
    }

    public struct CancellationFacts: Codable, Equatable, Sendable {
        public let cancelled: Bool
        public let reason: ServingCancellationReason?

        public init(
            cancelled: Bool,
            reason: ServingCancellationReason?
        ) throws {
            guard cancelled == (reason != nil) else {
                throw ValidationError.inconsistentCancellation
            }
            self.cancelled = cancelled
            self.reason = reason
        }

        private enum CodingKeys: String, CodingKey, CaseIterable {
            case cancelled
            case reason
        }

        public init(from decoder: Decoder) throws {
            try ServingEvidence.rejectUnknownKeys(
                from: decoder,
                allowed: CodingKeys.allCases.map(\.rawValue))
            let container = try decoder.container(keyedBy: CodingKeys.self)
            try self.init(
                cancelled: container.decode(Bool.self, forKey: .cancelled),
                reason: ServingEvidence.decodeCanonicalOptional(
                    ServingCancellationReason.self,
                    from: container,
                    forKey: .reason,
                    field: "reason"))
        }
    }

    public struct ResourceFacts: Codable, Equatable, Sendable {
        public let admission: Admission
        public let queueDepth: Int?
        public let before: ResourceSnapshot?
        public let active: ResourceSnapshot?
        public let terminal: ResourceSnapshot?
        public let failedSnapshots: [ResourceSnapshotStage]

        public init(
            admission: Admission,
            queueDepth: Int? = nil,
            before: ResourceSnapshot? = nil,
            active: ResourceSnapshot? = nil,
            terminal: ResourceSnapshot? = nil,
            failedSnapshots: [ResourceSnapshotStage] = []
        ) throws {
            if let queueDepth {
                try ServingEvidence.validateNonNegative(queueDepth, field: "queueDepth")
            }
            guard failedSnapshots == failedSnapshots.sorted(by: { $0.order < $1.order }),
                Set(failedSnapshots).count == failedSnapshots.count
            else {
                throw ValidationError.noncanonicalField(
                    "resources.failedSnapshots")
            }
            var successfulStages: Set<ResourceSnapshotStage> = []
            if before != nil {
                successfulStages.insert(.before)
            }
            if active != nil {
                successfulStages.insert(.active)
            }
            if terminal != nil {
                successfulStages.insert(.terminal)
            }
            guard successfulStages.isDisjoint(with: failedSnapshots) else {
                throw ValidationError.noncanonicalField(
                    "resources.failedSnapshots")
            }
            self.admission = admission
            self.queueDepth = queueDepth
            self.before = before
            self.active = active
            self.terminal = terminal
            self.failedSnapshots = failedSnapshots
        }

        private enum CodingKeys: String, CodingKey, CaseIterable {
            case admission
            case queueDepth = "queue_depth"
            case before
            case active
            case terminal
            case failedSnapshots = "failed_snapshots"
        }

        public init(from decoder: Decoder) throws {
            try ServingEvidence.rejectUnknownKeys(
                from: decoder,
                allowed: CodingKeys.allCases.map(\.rawValue))
            let container = try decoder.container(keyedBy: CodingKeys.self)
            try self.init(
                admission: container.decode(Admission.self, forKey: .admission),
                queueDepth: ServingEvidence.decodeCanonicalOptional(
                    Int.self,
                    from: container,
                    forKey: .queueDepth,
                    field: "queueDepth"),
                before: ServingEvidence.decodeCanonicalOptional(
                    ResourceSnapshot.self,
                    from: container,
                    forKey: .before,
                    field: "resources.before"),
                active: ServingEvidence.decodeCanonicalOptional(
                    ResourceSnapshot.self,
                    from: container,
                    forKey: .active,
                    field: "resources.active"),
                terminal: ServingEvidence.decodeCanonicalOptional(
                    ResourceSnapshot.self,
                    from: container,
                    forKey: .terminal,
                    field: "resources.terminal"),
                failedSnapshots: ServingEvidence.decodeCanonicalOptional(
                    [ResourceSnapshotStage].self,
                    from: container,
                    forKey: .failedSnapshots,
                    field: "resources.failedSnapshots")
                    ?? [])
        }
    }

    public enum ResourceSnapshotStage: String, Codable, Equatable, Hashable, Sendable {
        case before
        case active
        case terminal

        fileprivate var order: Int {
            switch self {
            case .before:
                0
            case .active:
                1
            case .terminal:
                2
            }
        }
    }

    public struct ResourceSnapshot: Codable, Equatable, Sendable {
        public let activeRequests: Int
        public let coordinatorSlots: Int
        public let reservedKVBytes: Int
        public let maxReservedKVBytes: Int
        public let mlxActiveBytes: Int
        public let mlxCacheBytes: Int
        public let mlxPeakBytes: Int

        public init(
            activeRequests: Int,
            coordinatorSlots: Int,
            reservedKVBytes: Int,
            maxReservedKVBytes: Int,
            mlxActiveBytes: Int,
            mlxCacheBytes: Int,
            mlxPeakBytes: Int
        ) throws {
            let values = [
                ("activeRequests", activeRequests),
                ("coordinatorSlots", coordinatorSlots),
                ("reservedKVBytes", reservedKVBytes),
                ("maxReservedKVBytes", maxReservedKVBytes),
                ("mlxActiveBytes", mlxActiveBytes),
                ("mlxCacheBytes", mlxCacheBytes),
                ("mlxPeakBytes", mlxPeakBytes),
            ]
            for (field, value) in values {
                try ServingEvidence.validateNonNegative(value, field: field)
            }
            self.activeRequests = activeRequests
            self.coordinatorSlots = coordinatorSlots
            self.reservedKVBytes = reservedKVBytes
            self.maxReservedKVBytes = maxReservedKVBytes
            self.mlxActiveBytes = mlxActiveBytes
            self.mlxCacheBytes = mlxCacheBytes
            self.mlxPeakBytes = mlxPeakBytes
        }

        private enum CodingKeys: String, CodingKey, CaseIterable {
            case activeRequests = "active_requests"
            case coordinatorSlots = "coordinator_slots"
            case reservedKVBytes = "reserved_kv_bytes"
            case maxReservedKVBytes = "max_reserved_kv_bytes"
            case mlxActiveBytes = "mlx_active_bytes"
            case mlxCacheBytes = "mlx_cache_bytes"
            case mlxPeakBytes = "mlx_peak_bytes"
        }

        public init(from decoder: Decoder) throws {
            try ServingEvidence.rejectUnknownKeys(
                from: decoder,
                allowed: CodingKeys.allCases.map(\.rawValue))
            let container = try decoder.container(keyedBy: CodingKeys.self)
            try self.init(
                activeRequests: container.decode(Int.self, forKey: .activeRequests),
                coordinatorSlots: container.decode(Int.self, forKey: .coordinatorSlots),
                reservedKVBytes: container.decode(Int.self, forKey: .reservedKVBytes),
                maxReservedKVBytes: container.decode(Int.self, forKey: .maxReservedKVBytes),
                mlxActiveBytes: container.decode(Int.self, forKey: .mlxActiveBytes),
                mlxCacheBytes: container.decode(Int.self, forKey: .mlxCacheBytes),
                mlxPeakBytes: container.decode(Int.self, forKey: .mlxPeakBytes))
        }
    }

    public enum Admission: String, Codable, Equatable, Sendable {
        case accepted
        case notAdmitted = "not_admitted"
        case backendFailure = "backend_failure"
        case queueFull = "queue_full"
        case capacityExceeded = "capacity_exceeded"
        case requestTooLarge = "request_too_large"
    }

    public enum SHA256 {
        public static func hexDigest(of data: Data) -> String {
            CryptoKit.SHA256.hash(data: data)
                .map { String(format: "%02x", $0) }
                .joined()
        }
    }

    public enum ValidationError: Error, Equatable, Sendable {
        case negativeField(String)
        case nonfiniteField(String)
        case invalidHTTPStatus(Int)
        case invalidSHA256(String)
        case inconsistentCancellation
        case noncanonicalField(String)
        case unknownField(String)
        case unsupportedSchemaVersion(Int)
    }
}

private struct ServingEvidenceCodingKey: CodingKey {
    let stringValue: String
    let intValue: Int?

    init?(stringValue: String) {
        self.stringValue = stringValue
        self.intValue = nil
    }

    init?(intValue: Int) {
        self.stringValue = String(intValue)
        self.intValue = intValue
    }
}

private extension ServingEvidence {
    static func decodeCanonicalOptional<Value: Decodable, Key: CodingKey>(
        _ type: Value.Type,
        from container: KeyedDecodingContainer<Key>,
        forKey key: Key,
        field: String
    ) throws -> Value? {
        guard container.contains(key) else {
            return nil
        }
        guard try !container.decodeNil(forKey: key) else {
            throw ValidationError.noncanonicalField(field)
        }
        return try container.decode(type, forKey: key)
    }

    static func validateCanonicalHTTPRequest(
        method: String,
        path: String
    ) throws {
        guard method == "POST" else {
            throw ValidationError.noncanonicalField("method")
        }
        guard path == "/v1/chat/completions" else {
            throw ValidationError.noncanonicalField("path")
        }
    }

    static func rejectUnknownKeys(
        from decoder: Decoder,
        allowed: [String]
    ) throws {
        let container = try decoder.container(keyedBy: ServingEvidenceCodingKey.self)
        let allowed = Set(allowed)
        if let unknown = container.allKeys
            .map(\.stringValue)
            .filter({ !allowed.contains($0) })
            .sorted()
            .first
        {
            throw ValidationError.unknownField(unknown)
        }
    }

    static func validateNonNegative(_ value: Int, field: String) throws {
        guard value >= 0 else {
            throw ValidationError.negativeField(field)
        }
    }

    static func validatePositive(_ value: Int, field: String) throws {
        guard value > 0 else {
            throw ValidationError.negativeField(field)
        }
    }

    static func validateFiniteNonNegative(_ value: Double, field: String) throws {
        guard value.isFinite else {
            throw ValidationError.nonfiniteField(field)
        }
        guard value >= 0 else {
            throw ValidationError.negativeField(field)
        }
    }

    static func validateHTTPStatus(_ status: Int) throws {
        guard (100...599).contains(status) else {
            throw ValidationError.invalidHTTPStatus(status)
        }
    }

    static func validateSHA256Hex(_ value: String, field: String) throws {
        let isValid = value.count == 64 && value.allSatisfy { character in
            character >= "0" && character <= "9" || character >= "a" && character <= "f"
        }
        guard isValid else {
            throw ValidationError.invalidSHA256(field)
        }
    }
}
