import Foundation

/// KV cache geometry a cold snapshot was written under. A restore MUST match these against the
/// running model before reusing on-disk bytes — a mismatch means the persisted blocks are laid out
/// for a different model/config and cannot be trusted, even if the token prefix appears to match.
public struct ColdSnapshotKVLayout: Equatable, Sendable, Codable {
    public let layerCount: Int
    public let kvHeadCount: Int
    public let headDim: Int
    /// Bytes per stored element (e.g. 2 for fp16/bf16, 1 for int8-quantized KV).
    public let elementBytes: Int

    public init(layerCount: Int, kvHeadCount: Int, headDim: Int, elementBytes: Int) {
        self.layerCount = layerCount
        self.kvHeadCount = kvHeadCount
        self.headDim = headDim
        self.elementBytes = elementBytes
    }

    /// Bytes for one token's K and V across all layers: 2 · layers · kvHeads · headDim · elementBytes.
    public var bytesPerToken: Int {
        2 * layerCount * kvHeadCount * headDim * elementBytes
    }

    /// True when every field is positive (a layout with a zero/negative dimension is meaningless).
    public var isWellFormed: Bool {
        layerCount > 0 && kvHeadCount > 0 && headDim > 0 && elementBytes > 0
    }
}

public enum ColdSnapshotManifestError: Error, CustomStringConvertible, Equatable {
    /// The manifest declared a `schema` this build does not understand — or omitted it entirely. Unlike
    /// the quant-reliability renderer, an ABSENT tag is rejected: this is a trust boundary, so a
    /// restore must never proceed on an untagged blob. Associated value is the found tag ("<absent>").
    case unsupportedSchema(String)
    /// The bytes were not decodable as a cold-snapshot manifest (associated: a short reason).
    case malformedJSON(String)
    /// A field was structurally present but invalid (associated: which field + why).
    case invalidField(String)
    /// The reused payload's SHA-256 did not match the manifest — the on-disk bytes are corrupt or
    /// belong to a different snapshot. Fail closed: recompute from the prompt instead of trusting them.
    case digestMismatch(expected: String, actual: String)

    public var description: String {
        switch self {
        case .unsupportedSchema(let tag):
            return "unsupported cold-snapshot schema '\(tag)' "
                + "(this build understands '\(ColdSnapshotManifest.supportedSchema)')"
        case .malformedJSON(let reason):
            return "malformed cold-snapshot manifest: \(reason)"
        case .invalidField(let reason):
            return "invalid cold-snapshot manifest field: \(reason)"
        case .digestMismatch(let expected, let actual):
            return "cold-snapshot payload digest mismatch (expected \(expected), got \(actual))"
        }
    }
}

/// Schema-tagged, digest-verified manifest describing one persisted exact-prefix KV snapshot. It pins
/// the block accounting (`ColdSnapshotBlockPlan`/`ColdSnapshotRestorePlan` geometry) plus the KV
/// layout and a payload digest, so a cold restore can validate a snapshot before reusing its bytes:
/// foreign/absent schema, malformed fields, or a digest mismatch all fail closed.
public struct ColdSnapshotManifest: Equatable, Sendable {
    public static let supportedSchema = "exact-prefix-cold-snapshot/v1"

    public let id: UInt64
    public let blockSize: Int
    public let storedBlockCount: Int
    public let storedPrefixTokenCount: Int
    public let bytes: Int
    public let kvLayout: ColdSnapshotKVLayout
    /// Lowercase SHA-256 hex of the persisted KV payload bytes.
    public let payloadDigest: String

    public init(
        id: UInt64,
        blockSize: Int,
        storedBlockCount: Int,
        storedPrefixTokenCount: Int,
        bytes: Int,
        kvLayout: ColdSnapshotKVLayout,
        payloadDigest: String
    ) {
        self.id = id
        self.blockSize = blockSize
        self.storedBlockCount = storedBlockCount
        self.storedPrefixTokenCount = storedPrefixTokenCount
        self.bytes = bytes
        self.kvLayout = kvLayout
        self.payloadDigest = payloadDigest
    }

    /// Build a manifest for `payload`, computing its digest with the shared `sha256Hex` helper.
    public static func make(
        id: UInt64,
        blockSize: Int,
        storedBlockCount: Int,
        storedPrefixTokenCount: Int,
        bytes: Int,
        kvLayout: ColdSnapshotKVLayout,
        payload: Data
    ) -> ColdSnapshotManifest {
        ColdSnapshotManifest(
            id: id,
            blockSize: blockSize,
            storedBlockCount: storedBlockCount,
            storedPrefixTokenCount: storedPrefixTokenCount,
            bytes: bytes,
            kvLayout: kvLayout,
            payloadDigest: sha256Hex(payload))
    }

    /// Verify reused on-disk bytes against the manifest before trusting them. Throws
    /// `digestMismatch` if the payload has been corrupted or swapped.
    public func verifyDigest(payload: Data) throws {
        let actual = sha256Hex(payload)
        guard actual == payloadDigest else {
            throw ColdSnapshotManifestError.digestMismatch(expected: payloadDigest, actual: actual)
        }
    }

    // MARK: - Codec

    private struct Wire: Codable {
        var schema: String?
        var id: UInt64
        var blockSize: Int
        var storedBlockCount: Int
        var storedPrefixTokenCount: Int
        var bytes: Int
        var kvLayout: ColdSnapshotKVLayout
        var payloadDigest: String
    }

    public func encode() throws -> Data {
        let wire = Wire(
            schema: Self.supportedSchema,
            id: id,
            blockSize: blockSize,
            storedBlockCount: storedBlockCount,
            storedPrefixTokenCount: storedPrefixTokenCount,
            bytes: bytes,
            kvLayout: kvLayout,
            payloadDigest: payloadDigest)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(wire)
    }

    public static func decode(from data: Data) throws -> ColdSnapshotManifest {
        let wire: Wire
        do {
            wire = try JSONDecoder().decode(Wire.self, from: data)
        } catch {
            throw ColdSnapshotManifestError.malformedJSON(String(describing: error))
        }
        guard let schema = wire.schema, schema == supportedSchema else {
            throw ColdSnapshotManifestError.unsupportedSchema(wire.schema ?? "<absent>")
        }
        guard wire.blockSize > 0 else {
            throw ColdSnapshotManifestError.invalidField("blockSize must be positive (got \(wire.blockSize))")
        }
        guard wire.storedBlockCount >= 0 else {
            throw ColdSnapshotManifestError.invalidField(
                "storedBlockCount must be non-negative (got \(wire.storedBlockCount))")
        }
        guard wire.storedPrefixTokenCount >= 0 else {
            throw ColdSnapshotManifestError.invalidField(
                "storedPrefixTokenCount must be non-negative (got \(wire.storedPrefixTokenCount))")
        }
        guard wire.bytes >= 0 else {
            throw ColdSnapshotManifestError.invalidField("bytes must be non-negative (got \(wire.bytes))")
        }
        guard wire.kvLayout.isWellFormed else {
            throw ColdSnapshotManifestError.invalidField(
                "kvLayout must have positive dimensions (got \(wire.kvLayout))")
        }
        guard isLowercaseSHA256Hex(wire.payloadDigest) else {
            throw ColdSnapshotManifestError.invalidField(
                "payloadDigest must be 64 lowercase hex chars (got \(wire.payloadDigest.count) chars)")
        }
        return ColdSnapshotManifest(
            id: wire.id,
            blockSize: wire.blockSize,
            storedBlockCount: wire.storedBlockCount,
            storedPrefixTokenCount: wire.storedPrefixTokenCount,
            bytes: wire.bytes,
            kvLayout: wire.kvLayout,
            payloadDigest: wire.payloadDigest)
    }

    private static func isLowercaseSHA256Hex(_ s: String) -> Bool {
        s.count == 64 && s.allSatisfy { $0.isHexDigit && !$0.isUppercase }
    }
}
