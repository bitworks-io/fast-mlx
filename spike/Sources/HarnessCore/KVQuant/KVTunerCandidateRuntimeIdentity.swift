import Foundation

public enum KVTunerCandidateRuntimeIdentityError:
    Error, Equatable, Sendable
{
    case invalidIdentity
    case missingRuntimeIdentity
    case sourceIdentityChangedDuringModelLoad
    case checkpointIdentityMismatch
    case tokenizerIdentityMismatch
    case invalidRuntimeContract(KVTunerCandidateRuntimeContractError)
}

/// Candidate-grade file identity sampled around MLX model loading. The pre-load and post-load
/// snapshots must match exactly before either one can be bound to the live tokenizer EOS token.
public struct KVTunerCandidateRuntimeSourceSnapshot: Equatable, Sendable {
    public let exactModelConfigData: Data
    public let checkpointManifestHash: String
    public let checkpointContentSHA256: String
    public let tokenizerSHA256: String

    private init(
        exactModelConfigData: Data,
        checkpointManifestHash: String,
        checkpointContentSHA256: String,
        tokenizerSHA256: String
    ) {
        self.exactModelConfigData = exactModelConfigData
        self.checkpointManifestHash = checkpointManifestHash
        self.checkpointContentSHA256 = checkpointContentSHA256
        self.tokenizerSHA256 = tokenizerSHA256
    }

    public static func load(
        exactModelConfigData: Data,
        checkpointManifestHash: String,
        checkpointContentSHA256: String,
        tokenizerSHA256: String
    ) throws -> KVTunerCandidateRuntimeSourceSnapshot {
        guard !exactModelConfigData.isEmpty,
            Self.isIdentityDigest(checkpointManifestHash),
            Self.isLowercaseHex(checkpointContentSHA256, length: 64),
            Self.isLowercaseHex(tokenizerSHA256, length: 64)
        else {
            throw KVTunerCandidateRuntimeIdentityError.invalidIdentity
        }
        return KVTunerCandidateRuntimeSourceSnapshot(
            exactModelConfigData: exactModelConfigData,
            checkpointManifestHash: checkpointManifestHash,
            checkpointContentSHA256: checkpointContentSHA256,
            tokenizerSHA256: tokenizerSHA256)
    }

    public static func validateUnchanged(
        before: KVTunerCandidateRuntimeSourceSnapshot,
        after: KVTunerCandidateRuntimeSourceSnapshot
    ) throws -> KVTunerCandidateRuntimeSourceSnapshot {
        guard before == after else {
            throw KVTunerCandidateRuntimeIdentityError
                .sourceIdentityChangedDuringModelLoad
        }
        return after
    }

    private static func isIdentityDigest(_ value: String) -> Bool {
        isLowercaseHex(value, length: 16)
            || isLowercaseHex(value, length: 64)
    }

    private static func isLowercaseHex(
        _ value: String, length: Int
    ) -> Bool {
        value.utf8.count == length
            && value.utf8.allSatisfy {
                (48 ... 57).contains($0) || (97 ... 102).contains($0)
            }
    }
}

/// Exact, path-free identity captured from the files and tokenizer attached to the live model
/// container. Candidate policies are checked against this independently captured value inside the
/// inference actor before any MLX cache is created, so policy-provided provenance can never label
/// execution by itself.
public struct KVTunerCandidateRuntimeIdentity: Equatable, Sendable {
    public let exactModelConfigData: Data
    public let modelConfigHash: String
    public let modelConfigSHA256: String
    public let checkpointManifestHash: String
    public let checkpointContentSHA256: String
    public let tokenizerSHA256: String
    public let eosTokenID: Int

    private init(
        sourceSnapshot: KVTunerCandidateRuntimeSourceSnapshot,
        eosTokenID: Int
    ) {
        self.exactModelConfigData = sourceSnapshot.exactModelConfigData
        self.modelConfigHash = fnv1a64(sourceSnapshot.exactModelConfigData)
        self.modelConfigSHA256 = sha256Hex(
            sourceSnapshot.exactModelConfigData)
        self.checkpointManifestHash = sourceSnapshot.checkpointManifestHash
        self.checkpointContentSHA256 =
            sourceSnapshot.checkpointContentSHA256
        self.tokenizerSHA256 = sourceSnapshot.tokenizerSHA256
        self.eosTokenID = eosTokenID
    }

    public static func load(
        exactModelConfigData: Data,
        checkpointManifestHash: String,
        checkpointContentSHA256: String,
        tokenizerSHA256: String,
        eosTokenID: Int
    ) throws -> KVTunerCandidateRuntimeIdentity {
        let sourceSnapshot = try KVTunerCandidateRuntimeSourceSnapshot.load(
            exactModelConfigData: exactModelConfigData,
            checkpointManifestHash: checkpointManifestHash,
            checkpointContentSHA256: checkpointContentSHA256,
            tokenizerSHA256: tokenizerSHA256)
        return try load(
            sourceSnapshot: sourceSnapshot,
            eosTokenID: eosTokenID)
    }

    public static func load(
        sourceSnapshot: KVTunerCandidateRuntimeSourceSnapshot,
        eosTokenID: Int
    ) throws -> KVTunerCandidateRuntimeIdentity {
        guard eosTokenID >= 0 else {
            throw KVTunerCandidateRuntimeIdentityError.invalidIdentity
        }
        return KVTunerCandidateRuntimeIdentity(
            sourceSnapshot: sourceSnapshot,
            eosTokenID: eosTokenID)
    }

    /// Reconciles the independently captured live identity with an authenticated candidate policy
    /// and returns the config-derived geometry used to validate its eventual cache receipt.
    @discardableResult
    public func validate(
        runtimePolicy: KVTunerCandidateRuntimePolicy
    ) throws -> KVTunerCandidateRuntimeContract {
        guard checkpointManifestHash
                == runtimePolicy.checkpointManifestHash,
            checkpointContentSHA256
                == runtimePolicy.checkpointContentSHA256
        else {
            throw KVTunerCandidateRuntimeIdentityError
                .checkpointIdentityMismatch
        }
        guard tokenizerSHA256 == runtimePolicy.tokenizerSHA256 else {
            throw KVTunerCandidateRuntimeIdentityError
                .tokenizerIdentityMismatch
        }
        do {
            return try KVTunerCandidateRuntimeContract.load(
                exactModelConfigData: exactModelConfigData,
                runtimePolicy: runtimePolicy,
                eosTokenID: eosTokenID)
        } catch let error as KVTunerCandidateRuntimeContractError {
            throw KVTunerCandidateRuntimeIdentityError
                .invalidRuntimeContract(error)
        }
    }
}
