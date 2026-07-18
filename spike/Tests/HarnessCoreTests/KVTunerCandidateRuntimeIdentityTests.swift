import Foundation
import XCTest

@testable import HarnessCore

final class KVTunerCandidateRuntimeIdentityTests: XCTestCase {
    func testSourceSnapshotMustRemainIdenticalAcrossModelLoad() throws {
        let inputs = try KVTunerTestFixtures.candidateRuntimeInputs()
        let before = try KVTunerCandidateRuntimeSourceSnapshot.load(
            exactModelConfigData: inputs.configData,
            checkpointManifestHash: inputs.manifest.checkpointManifestHash,
            tokenizerSHA256: inputs.manifest.tokenizerSHA256)

        XCTAssertEqual(
            try KVTunerCandidateRuntimeSourceSnapshot.validateUnchanged(
                before: before, after: before),
            before)
        let changedConfig = try KVTunerCandidateRuntimeSourceSnapshot.load(
            exactModelConfigData: inputs.configData + Data(" ".utf8),
            checkpointManifestHash: inputs.manifest.checkpointManifestHash,
            tokenizerSHA256: inputs.manifest.tokenizerSHA256)
        XCTAssertThrowsError(
            try KVTunerCandidateRuntimeSourceSnapshot.validateUnchanged(
                before: before, after: changedConfig)
        ) { error in
            XCTAssertEqual(
                error as? KVTunerCandidateRuntimeIdentityError,
                .sourceIdentityChangedDuringModelLoad)
        }
    }

    func testMatchingLiveIdentityDerivesAuthenticatedRuntimeContract() throws {
        let inputs = try KVTunerTestFixtures.candidateRuntimeInputs()
        let policy = try KVTunerTestFixtures.candidateRuntimePolicy(
            inputs, candidateOrdinal: 0)
        let identity = try KVTunerCandidateRuntimeIdentity.load(
            exactModelConfigData: inputs.configData,
            checkpointManifestHash: inputs.manifest.checkpointManifestHash,
            tokenizerSHA256: inputs.manifest.tokenizerSHA256,
            eosTokenID: 255)

        let contract = try identity.validate(runtimePolicy: policy)

        XCTAssertEqual(contract.modelConfigSHA256, policy.modelConfigSHA256)
        XCTAssertEqual(contract.eosTokenID, 255)
        XCTAssertEqual(contract.geometry.layerCount, policy.layers.count)
    }

    func testLiveIdentityRejectsCheckpointTokenizerConfigAndEOSSubstitution() throws {
        let inputs = try KVTunerTestFixtures.candidateRuntimeInputs()
        let policy = try KVTunerTestFixtures.candidateRuntimePolicy(
            inputs, candidateOrdinal: 0)

        XCTAssertThrowsError(try KVTunerCandidateRuntimeIdentity.load(
            exactModelConfigData: inputs.configData,
            checkpointManifestHash: String(repeating: "0", count: 16),
            tokenizerSHA256: inputs.manifest.tokenizerSHA256,
            eosTokenID: 255).validate(runtimePolicy: policy)) { error in
                XCTAssertEqual(
                    error as? KVTunerCandidateRuntimeIdentityError,
                    .checkpointIdentityMismatch)
            }
        XCTAssertThrowsError(try KVTunerCandidateRuntimeIdentity.load(
            exactModelConfigData: inputs.configData,
            checkpointManifestHash: inputs.manifest.checkpointManifestHash,
            tokenizerSHA256: String(repeating: "b", count: 64),
            eosTokenID: 255).validate(runtimePolicy: policy)) { error in
                XCTAssertEqual(
                    error as? KVTunerCandidateRuntimeIdentityError,
                    .tokenizerIdentityMismatch)
            }

        var changedConfig = inputs.configData
        changedConfig.append(0x20)
        XCTAssertThrowsError(try KVTunerCandidateRuntimeIdentity.load(
            exactModelConfigData: changedConfig,
            checkpointManifestHash: inputs.manifest.checkpointManifestHash,
            tokenizerSHA256: inputs.manifest.tokenizerSHA256,
            eosTokenID: 255).validate(runtimePolicy: policy)) { error in
                XCTAssertEqual(
                    error as? KVTunerCandidateRuntimeIdentityError,
                    .invalidRuntimeContract(.modelConfigIdentityMismatch))
            }
        XCTAssertThrowsError(try KVTunerCandidateRuntimeIdentity.load(
            exactModelConfigData: inputs.configData,
            checkpointManifestHash: inputs.manifest.checkpointManifestHash,
            tokenizerSHA256: inputs.manifest.tokenizerSHA256,
            eosTokenID: 254).validate(runtimePolicy: policy)) { error in
                XCTAssertEqual(
                    error as? KVTunerCandidateRuntimeIdentityError,
                    .invalidRuntimeContract(.invalidEOSTokenID))
            }
    }

    func testLiveIdentityRejectsMalformedIndependentInputs() throws {
        XCTAssertThrowsError(try KVTunerCandidateRuntimeIdentity.load(
            exactModelConfigData: Data(),
            checkpointManifestHash: "checkpoint",
            tokenizerSHA256: String(repeating: "a", count: 64),
            eosTokenID: 1)) { error in
                XCTAssertEqual(
                    error as? KVTunerCandidateRuntimeIdentityError,
                    .invalidIdentity)
            }
        XCTAssertThrowsError(try KVTunerCandidateRuntimeIdentity.load(
            exactModelConfigData: Data("{}".utf8),
            checkpointManifestHash: "checkpoint",
            tokenizerSHA256: "NOT-A-SHA",
            eosTokenID: 1))
        XCTAssertThrowsError(try KVTunerCandidateRuntimeIdentity.load(
            exactModelConfigData: Data("{}".utf8),
            checkpointManifestHash: "checkpoint",
            tokenizerSHA256: String(repeating: "a", count: 64),
            eosTokenID: -1))
    }
}
