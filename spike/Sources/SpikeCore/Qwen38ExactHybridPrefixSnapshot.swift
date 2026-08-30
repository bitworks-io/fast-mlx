import CryptoKit
import Foundation
import MLX
@_spi(FastMLXExactPrefix) import MLXLMCommon

public enum Qwen38ExactHybridPrefixSnapshotError: Error, Equatable, Sendable {
    case invalidIdentityField(String)
    case emptyPrefix
    case negativeTokenID(position: Int)
    case tokenIDOutOfInt32Range(position: Int)
    case emptyCacheList
    case missingDensePrefixWitness
    case unsupportedCacheType(index: Int, typeName: String)
    case invalidCacheOffset(index: Int, typeName: String, offset: Int)
    case emptyCacheState(index: Int, typeName: String)
    case partialCacheState(index: Int, typeName: String, reason: String)
    case invalidCacheMetadata(index: Int, typeName: String, reason: String)
    case invalidStateArray(index: Int, typeName: String, arrayIndex: Int, reason: String)
    case copyChangedFingerprint(index: Int, typeName: String)
    case identityMismatch
    case prefixMismatch
    case digestMismatch
    case restoreFingerprintMismatch(index: Int, typeName: String)
}

public struct Qwen38ExactHybridPrefixIdentity: Equatable, Hashable, Sendable {
    public let modelSHA256: String
    public let sourceSHA256: String
    public let artifactSHA256: String
    public let tokenizerSHA256: String
    public let templateSHA256: String
    public let cacheLayoutSHA256: String
    public let requestNamespaceSHA256: String

    public init(
        modelSHA256: String,
        sourceSHA256: String,
        artifactSHA256: String,
        tokenizerSHA256: String,
        templateSHA256: String,
        cacheLayoutSHA256: String,
        requestNamespaceSHA256: String
    ) throws {
        let fields = [
            ("modelSHA256", modelSHA256),
            ("sourceSHA256", sourceSHA256),
            ("artifactSHA256", artifactSHA256),
            ("tokenizerSHA256", tokenizerSHA256),
            ("templateSHA256", templateSHA256),
            ("cacheLayoutSHA256", cacheLayoutSHA256),
            ("requestNamespaceSHA256", requestNamespaceSHA256),
        ]
        for (field, value) in fields where !Self.isLowercaseSHA256(value) {
            throw Qwen38ExactHybridPrefixSnapshotError.invalidIdentityField(field)
        }

        self.modelSHA256 = modelSHA256
        self.sourceSHA256 = sourceSHA256
        self.artifactSHA256 = artifactSHA256
        self.tokenizerSHA256 = tokenizerSHA256
        self.templateSHA256 = templateSHA256
        self.cacheLayoutSHA256 = cacheLayoutSHA256
        self.requestNamespaceSHA256 = requestNamespaceSHA256
    }

    private static func isLowercaseSHA256(_ value: String) -> Bool {
        value.utf8.count == 64
            && value.utf8.allSatisfy {
                (48 ... 57).contains($0) || (97 ... 102).contains($0)
            }
    }
}

public struct Qwen38ExactHybridPrefixStateArrayFingerprint:
    Equatable, Sendable
{
    public let shape: [Int]
    public let dtypeName: String
    public let dtypeByteWidth: Int
    public let nbytes: Int
    public let contentsSHA256: String
}

public struct Qwen38ExactHybridPrefixCacheFingerprint:
    Equatable, Sendable
{
    public let dynamicTypeName: String
    public let offset: Int
    public let metaState: [String]
    public let stateArrays: [Qwen38ExactHybridPrefixStateArrayFingerprint]
}

/// Actor-confined scalar warm-prefix snapshot for Qwen hybrid cache state.
///
/// This object intentionally has no `Sendable` conformance. Capture and restore must run on the
/// inference actor that owns the MLX cache objects. It snapshots only the scalar target cache;
/// MTP drafter state, live adapter wiring, and production admission are outside this primitive.
public final class Qwen38ExactHybridPrefixSnapshot {
    public let identity: Qwen38ExactHybridPrefixIdentity
    public let prefixTokenIDs: [Int]
    public let prefixTokenCount: Int
    public let cacheFingerprints: [Qwen38ExactHybridPrefixCacheFingerprint]
    public let canonicalDigest: String

    private let sealedCaches: [any KVCache]

    private init(
        identity: Qwen38ExactHybridPrefixIdentity,
        prefixTokenIDs: [Int],
        cacheFingerprints: [Qwen38ExactHybridPrefixCacheFingerprint],
        canonicalDigest: String,
        sealedCaches: [any KVCache]
    ) {
        self.identity = identity
        self.prefixTokenIDs = prefixTokenIDs
        self.prefixTokenCount = prefixTokenIDs.count
        self.cacheFingerprints = cacheFingerprints
        self.canonicalDigest = canonicalDigest
        self.sealedCaches = sealedCaches
    }

    public static func capture(
        identity: Qwen38ExactHybridPrefixIdentity,
        prefixTokenIDs: [Int],
        caches: [any KVCache]
    ) throws -> Qwen38ExactHybridPrefixSnapshot {
        let checkedPrefix = try validatePrefix(prefixTokenIDs)
        guard !caches.isEmpty else {
            throw Qwen38ExactHybridPrefixSnapshotError.emptyCacheList
        }

        var sealedCaches: [any KVCache] = []
        sealedCaches.reserveCapacity(caches.count)
        var fingerprints: [Qwen38ExactHybridPrefixCacheFingerprint] = []
        fingerprints.reserveCapacity(caches.count)
        var hasDensePrefixWitness = false

        for (index, cache) in caches.enumerated() {
            let liveFingerprint = try fingerprint(
                cache: cache,
                index: index,
                prefixTokenCount: checkedPrefix.count)
            if cache is KVCacheSimple {
                hasDensePrefixWitness = true
            }
            let copied = cache.copy()
            let copiedFingerprint = try fingerprint(
                cache: copied,
                index: index,
                prefixTokenCount: checkedPrefix.count)
            guard copiedFingerprint == liveFingerprint else {
                throw Qwen38ExactHybridPrefixSnapshotError.copyChangedFingerprint(
                    index: index,
                    typeName: liveFingerprint.dynamicTypeName)
            }
            sealedCaches.append(copied)
            fingerprints.append(copiedFingerprint)
        }
        guard hasDensePrefixWitness else {
            throw Qwen38ExactHybridPrefixSnapshotError.missingDensePrefixWitness
        }

        let digest = canonicalDigest(
            identity: identity,
            prefixTokenIDs: checkedPrefix,
            cacheFingerprints: fingerprints)
        return Qwen38ExactHybridPrefixSnapshot(
            identity: identity,
            prefixTokenIDs: checkedPrefix,
            cacheFingerprints: fingerprints,
            canonicalDigest: digest,
            sealedCaches: sealedCaches)
    }

    public static func canonicalDigest(
        identity: Qwen38ExactHybridPrefixIdentity,
        prefixTokenIDs: [Int],
        caches: [any KVCache]
    ) throws -> String {
        let checkedPrefix = try validatePrefix(prefixTokenIDs)
        guard !caches.isEmpty else {
            throw Qwen38ExactHybridPrefixSnapshotError.emptyCacheList
        }
        var hasDensePrefixWitness = false
        let fingerprints = try caches.enumerated().map { index, cache in
            if cache is KVCacheSimple {
                hasDensePrefixWitness = true
            }
            return try fingerprint(
                cache: cache,
                index: index,
                prefixTokenCount: checkedPrefix.count)
        }
        guard hasDensePrefixWitness else {
            throw Qwen38ExactHybridPrefixSnapshotError.missingDensePrefixWitness
        }
        return canonicalDigest(
            identity: identity,
            prefixTokenIDs: checkedPrefix,
            cacheFingerprints: fingerprints)
    }

    public func restore(
        expectedIdentity: Qwen38ExactHybridPrefixIdentity,
        expectedPrefixTokenIDs: [Int],
        expectedDigest: String
    ) throws -> [any KVCache] {
        guard expectedIdentity == identity else {
            throw Qwen38ExactHybridPrefixSnapshotError.identityMismatch
        }
        guard expectedPrefixTokenIDs == prefixTokenIDs else {
            throw Qwen38ExactHybridPrefixSnapshotError.prefixMismatch
        }
        guard expectedDigest == canonicalDigest else {
            throw Qwen38ExactHybridPrefixSnapshotError.digestMismatch
        }

        var restored: [any KVCache] = []
        restored.reserveCapacity(sealedCaches.count)
        for (index, cache) in sealedCaches.enumerated() {
            let copied = cache.copy()
            let copiedFingerprint = try Self.fingerprint(
                cache: copied,
                index: index,
                prefixTokenCount: prefixTokenCount)
            guard copiedFingerprint == cacheFingerprints[index] else {
                throw Qwen38ExactHybridPrefixSnapshotError.restoreFingerprintMismatch(
                    index: index,
                    typeName: cacheFingerprints[index].dynamicTypeName)
            }
            restored.append(copied)
        }

        let restoredDigest = try Self.canonicalDigest(
            identity: identity,
            prefixTokenIDs: prefixTokenIDs,
            caches: restored)
        guard restoredDigest == canonicalDigest else {
            throw Qwen38ExactHybridPrefixSnapshotError.digestMismatch
        }
        return restored
    }

    private static func validatePrefix(_ prefixTokenIDs: [Int]) throws -> [Int] {
        guard !prefixTokenIDs.isEmpty else {
            throw Qwen38ExactHybridPrefixSnapshotError.emptyPrefix
        }
        for (position, tokenID) in prefixTokenIDs.enumerated() {
            guard tokenID >= 0 else {
                throw Qwen38ExactHybridPrefixSnapshotError.negativeTokenID(
                    position: position)
            }
            guard Int32(exactly: tokenID) != nil else {
                throw Qwen38ExactHybridPrefixSnapshotError.tokenIDOutOfInt32Range(
                    position: position)
            }
        }
        return prefixTokenIDs
    }

    private static func fingerprint(
        cache: any KVCache,
        index: Int,
        prefixTokenCount: Int
    ) throws -> Qwen38ExactHybridPrefixCacheFingerprint {
        let typeName = String(reflecting: type(of: cache))
        try validateSupportedCacheType(cache, index: index, typeName: typeName)

        guard cache.offset >= 0 else {
            throw Qwen38ExactHybridPrefixSnapshotError.invalidCacheOffset(
                index: index,
                typeName: typeName,
                offset: cache.offset)
        }

        let state = cache.state
        let metaState = cache.metaState
        try validateStateContract(
            cache: cache,
            typeName: typeName,
            index: index,
            prefixTokenCount: prefixTokenCount,
            state: state,
            metaState: metaState)

        MLX.eval(state)
        let stateFingerprints = try state.enumerated().map { arrayIndex, array in
            try arrayFingerprint(
                array,
                cacheIndex: index,
                typeName: typeName,
                arrayIndex: arrayIndex)
        }
        return Qwen38ExactHybridPrefixCacheFingerprint(
            dynamicTypeName: typeName,
            offset: cache.offset,
            metaState: metaState,
            stateArrays: stateFingerprints)
    }

    private static func validateSupportedCacheType(
        _ cache: any KVCache,
        index: Int,
        typeName: String
    ) throws {
        switch cache {
        case is KVCacheSimple, is MambaCache:
            return
        default:
            throw Qwen38ExactHybridPrefixSnapshotError.unsupportedCacheType(
                index: index,
                typeName: typeName)
        }
    }

    private static func validateStateContract(
        cache: any KVCache,
        typeName: String,
        index: Int,
        prefixTokenCount: Int,
        state: [MLXArray],
        metaState: [String]
    ) throws {
        guard !state.isEmpty else {
            throw Qwen38ExactHybridPrefixSnapshotError.emptyCacheState(
                index: index,
                typeName: typeName)
        }

        if cache is KVCacheSimple {
            guard state.count == 2 else {
                throw Qwen38ExactHybridPrefixSnapshotError.partialCacheState(
                    index: index,
                    typeName: typeName,
                    reason: "KVCacheSimple requires key and value state arrays")
            }
            guard metaState == [""] else {
                throw Qwen38ExactHybridPrefixSnapshotError.invalidCacheMetadata(
                    index: index,
                    typeName: typeName,
                    reason: "KVCacheSimple must not carry transient metadata")
            }
            guard state.allSatisfy({ $0.shape.count == 4 }) else {
                throw Qwen38ExactHybridPrefixSnapshotError.partialCacheState(
                    index: index,
                    typeName: typeName,
                    reason: "KVCacheSimple state arrays must be rank 4")
            }
            guard state[0].dim(0) == 1, state[1].dim(0) == 1 else {
                throw Qwen38ExactHybridPrefixSnapshotError.partialCacheState(
                    index: index,
                    typeName: typeName,
                    reason: "KVCacheSimple scalar snapshot requires batch size 1")
            }
            guard state[0].shape.allSatisfy({ $0 > 0 }),
                state[1].shape.allSatisfy({ $0 > 0 })
            else {
                throw Qwen38ExactHybridPrefixSnapshotError.partialCacheState(
                    index: index,
                    typeName: typeName,
                    reason: "KVCacheSimple state arrays require positive dimensions")
            }
            guard isSupportedFloatingDType(state[0].dtype),
                isSupportedFloatingDType(state[1].dtype)
            else {
                throw Qwen38ExactHybridPrefixSnapshotError.invalidStateArray(
                    index: index,
                    typeName: typeName,
                    arrayIndex: isSupportedFloatingDType(state[0].dtype) ? 1 : 0,
                    reason: "KVCacheSimple state arrays require supported floating dtypes")
            }
            guard state[0].dim(2) == prefixTokenCount,
                state[1].dim(2) == prefixTokenCount,
                cache.offset == prefixTokenCount
            else {
                throw Qwen38ExactHybridPrefixSnapshotError.partialCacheState(
                    index: index,
                    typeName: typeName,
                    reason: "KVCacheSimple state must cover the complete prefix")
            }
            return
        }

        if let mamba = cache as? MambaCache {
            try validateResolvedSpeculativeCheckpoint(
                mamba.hasSpeculativeCheckpointForAudit,
                index: index,
                typeName: typeName)
            guard cache.offset == 0 else {
                throw Qwen38ExactHybridPrefixSnapshotError.invalidCacheOffset(
                    index: index,
                    typeName: typeName,
                    offset: cache.offset)
            }
            guard state.count == 2 else {
                throw Qwen38ExactHybridPrefixSnapshotError.partialCacheState(
                    index: index,
                    typeName: typeName,
                    reason: "MambaCache requires conv and recurrent state arrays")
            }
            guard metaState == ["2", "0,1"] else {
                throw Qwen38ExactHybridPrefixSnapshotError.invalidCacheMetadata(
                    index: index,
                    typeName: typeName,
                    reason: "MambaCache must be complete and finalized before snapshot")
            }
            guard state[0].shape.count == 3 else {
                throw Qwen38ExactHybridPrefixSnapshotError.partialCacheState(
                    index: index,
                    typeName: typeName,
                    reason: "MambaCache conv state must be rank 3")
            }
            guard state[1].shape.count == 4 else {
                throw Qwen38ExactHybridPrefixSnapshotError.partialCacheState(
                    index: index,
                    typeName: typeName,
                    reason: "MambaCache recurrent state must be rank 4")
            }
            guard state[0].dim(0) == 1, state[1].dim(0) == 1 else {
                throw Qwen38ExactHybridPrefixSnapshotError.partialCacheState(
                    index: index,
                    typeName: typeName,
                    reason: "MambaCache scalar snapshot requires batch size 1")
            }
            guard isSupportedFloatingDType(state[0].dtype),
                isSupportedFloatingDType(state[1].dtype)
            else {
                throw Qwen38ExactHybridPrefixSnapshotError.invalidStateArray(
                    index: index,
                    typeName: typeName,
                    arrayIndex: isSupportedFloatingDType(state[0].dtype) ? 1 : 0,
                    reason: "MambaCache state arrays require supported floating dtypes")
            }
            guard state[0].shape.allSatisfy({ $0 > 0 }),
                state[1].shape.allSatisfy({ $0 > 0 })
            else {
                throw Qwen38ExactHybridPrefixSnapshotError.partialCacheState(
                    index: index,
                    typeName: typeName,
                    reason: "MambaCache state arrays require positive dimensions")
            }
        }
    }

    /// Internal test seam joining the vendored checkpoint-lifecycle proof to this snapshot's
    /// fail-closed policy without exposing a checkpoint mutation API across package boundaries.
    static func validateResolvedSpeculativeCheckpoint(
        _ hasSpeculativeCheckpoint: Bool,
        index: Int,
        typeName: String
    ) throws {
        guard !hasSpeculativeCheckpoint else {
            throw Qwen38ExactHybridPrefixSnapshotError.invalidCacheMetadata(
                index: index,
                typeName: typeName,
                reason: "MambaCache speculative checkpoint must be resolved before snapshot")
        }
    }

    private static func arrayFingerprint(
        _ array: MLXArray,
        cacheIndex: Int,
        typeName: String,
        arrayIndex: Int
    ) throws -> Qwen38ExactHybridPrefixStateArrayFingerprint {
        guard !array.shape.isEmpty, array.shape.allSatisfy({ $0 > 0 }) else {
            throw Qwen38ExactHybridPrefixSnapshotError.invalidStateArray(
                index: cacheIndex,
                typeName: typeName,
                arrayIndex: arrayIndex,
                reason: "state array shape must be non-empty and positive")
        }
        let elementCount = try checkedElementCount(
            shape: array.shape,
            cacheIndex: cacheIndex,
            typeName: typeName,
            arrayIndex: arrayIndex)
        let (expectedNBytes, overflow) = elementCount.multipliedReportingOverflow(
            by: array.dtype.size)
        guard !overflow, expectedNBytes > 0, expectedNBytes == array.nbytes else {
            throw Qwen38ExactHybridPrefixSnapshotError.invalidStateArray(
                index: cacheIndex,
                typeName: typeName,
                arrayIndex: arrayIndex,
                reason: "state array byte count does not match shape and dtype")
        }
        let host = array.asData(access: .copy)
        guard host.data.count == array.nbytes else {
            throw Qwen38ExactHybridPrefixSnapshotError.invalidStateArray(
                index: cacheIndex,
                typeName: typeName,
                arrayIndex: arrayIndex,
                reason: "state array could not be materialized exactly")
        }
        return Qwen38ExactHybridPrefixStateArrayFingerprint(
            shape: host.shape,
            dtypeName: String(describing: host.dType),
            dtypeByteWidth: host.dType.size,
            nbytes: host.data.count,
            contentsSHA256: sha256Hex(host.data))
    }

    private static func checkedElementCount(
        shape: [Int],
        cacheIndex: Int,
        typeName: String,
        arrayIndex: Int
    ) throws -> Int {
        var count = 1
        for dimension in shape {
            let (next, overflow) = count.multipliedReportingOverflow(by: dimension)
            guard !overflow else {
                throw Qwen38ExactHybridPrefixSnapshotError.invalidStateArray(
                    index: cacheIndex,
                    typeName: typeName,
                    arrayIndex: arrayIndex,
                    reason: "state array element count overflow")
            }
            count = next
        }
        return count
    }

    private static func isSupportedFloatingDType(_ dtype: DType) -> Bool {
        switch dtype {
        case .float16, .bfloat16, .float32:
            true
        default:
            false
        }
    }

    private static func canonicalDigest(
        identity: Qwen38ExactHybridPrefixIdentity,
        prefixTokenIDs: [Int],
        cacheFingerprints: [Qwen38ExactHybridPrefixCacheFingerprint]
    ) -> String {
        var hasher = SHA256()
        update(&hasher, "qwen38-exact-hybrid-prefix-snapshot-v1")
        update(&hasher, "model", identity.modelSHA256)
        update(&hasher, "source", identity.sourceSHA256)
        update(&hasher, "artifact", identity.artifactSHA256)
        update(&hasher, "tokenizer", identity.tokenizerSHA256)
        update(&hasher, "template", identity.templateSHA256)
        update(&hasher, "cache-layout", identity.cacheLayoutSHA256)
        update(&hasher, "request-namespace", identity.requestNamespaceSHA256)
        update(&hasher, "prefix-count", String(prefixTokenIDs.count))
        for tokenID in prefixTokenIDs {
            update(&hasher, "prefix-token", String(tokenID))
        }
        update(&hasher, "cache-count", String(cacheFingerprints.count))
        for cache in cacheFingerprints {
            update(&hasher, "cache-type", cache.dynamicTypeName)
            update(&hasher, "cache-offset", String(cache.offset))
            update(&hasher, "cache-meta-count", String(cache.metaState.count))
            for value in cache.metaState {
                update(&hasher, "cache-meta", value)
            }
            update(&hasher, "cache-state-count", String(cache.stateArrays.count))
            for array in cache.stateArrays {
                update(&hasher, "array-rank", String(array.shape.count))
                for dimension in array.shape {
                    update(&hasher, "array-dim", String(dimension))
                }
                update(&hasher, "array-dtype", array.dtypeName)
                update(&hasher, "array-dtype-width", String(array.dtypeByteWidth))
                update(&hasher, "array-nbytes", String(array.nbytes))
                update(&hasher, "array-sha256", array.contentsSHA256)
            }
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func update(
        _ hasher: inout SHA256,
        _ field: String,
        _ value: String? = nil
    ) {
        let payload = value.map { "\(field)=\($0.utf8.count):\($0)\n" }
            ?? "\(field)\n"
        hasher.update(data: Data(payload.utf8))
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
