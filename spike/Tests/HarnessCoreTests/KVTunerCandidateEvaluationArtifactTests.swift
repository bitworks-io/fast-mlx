import Foundation
import XCTest

@testable import HarnessCore

final class KVTunerCandidateEvaluationArtifactTests: XCTestCase {
    private struct Inputs {
        let artifact: KVTunerCandidateEvaluationArtifact
        let data: Data
        let runtimePolicy: KVTunerCandidateRuntimePolicy
        let runtimeContract: KVTunerCandidateRuntimeContract
        let configData: Data
        let manifest: KVTunerCalibrationManifest
        let manifestData: Data
    }

    private func decodeTokenIDs(_ tokenIDs: [Int]) throws -> String {
        guard tokenIDs.allSatisfy({ (0...255).contains($0) }) else {
            throw CocoaError(.coderInvalidValue)
        }
        return String(decoding: tokenIDs.map(UInt8.init), as: UTF8.self)
    }

    private func environment(
        policy: KVTunerCandidateRuntimePolicy
    ) -> KVTunerCandidateExecutionEnvironment {
        KVTunerCandidateExecutionEnvironment(
            harnessGitSHA: String(repeating: "a", count: 40),
            buildConfiguration: "Release",
            mlxSwiftVersion: "0.31.6",
            mlxSwiftLMRevision:
                "702e5a0eaf990e1f6d3db2b6e7d8872858a44055",
            hardwareChip: "Apple M3 Ultra",
            hardwareRAMBytes: 274_877_906_944,
            hardwareOS: "macOS 26.5.2",
            modelConfigHash: policy.modelConfigHash,
            modelConfigSHA256: policy.modelConfigSHA256,
            checkpointManifestHash: policy.checkpointManifestHash,
            tokenizerSHA256: policy.tokenizerSHA256)
    }

    private func receipt(
        policy: KVTunerCandidateRuntimePolicy,
        contract: KVTunerCandidateRuntimeContract,
        promptTokenCount: Int,
        generatedTokenCount: Int
    ) throws -> KVTunerCandidateRuntimeReceipt {
        let cachedTokens = promptTokenCount + generatedTokenCount
        let capacityTokens = try contract.capacityTokens(
            promptTokenCount: promptTokenCount,
            generatedTokenCount: generatedTokenCount,
            previousCapacityTokens: nil)
        let workspaceBytes = try contract.workspaceBytes(
            capacityTokens: capacityTokens)
        let allocation = try KVStorageFormat.kvtunerAllocation(
            layerPolicy: policy.layers.map {
                KVLayerPrecision(
                    layer: $0.layer,
                    keyBits: $0.keyBits,
                    valueBits: $0.valueBits)
            },
            groupSize: policy.groupSize,
            geometry: KVStorageGeometry(
                layerCount: contract.geometry.layerCount,
                kvHeadCount: contract.geometry.kvHeadCount,
                headDimension: contract.geometry.headDimension),
            capacityTokens: capacityTokens,
            sequences: contract.sequenceCount,
            metadataScalarBytes: contract.metadataScalarBytes,
            maximumLayerWorkspaceBytes: workspaceBytes)
        return KVTunerCandidateRuntimeReceipt(
            runtimePolicySHA256: policy.runtimePolicySHA256,
            cachedTokens: cachedTokens,
            capacityTokens: capacityTokens,
            layers: policy.layers,
            geometry: contract.geometry,
            groupSize: policy.groupSize,
            sequenceCount: contract.sequenceCount,
            metadataScalarBytes: contract.metadataScalarBytes,
            actualPayloadBytes: allocation.payloadBytes,
            actualMetadataBytes: allocation.metadataBytes,
            actualControlBytes: allocation.controlBytes,
            actualWorkspaceBytes: allocation.workspaceBytes,
            actualTotalPersistentBytes: allocation.totalPersistentBytes,
            actualTotalBytes: allocation.totalBytes)
    }

    private func inputs(
        wrongAt wrongOrdinal: Int? = nil
    ) throws -> Inputs {
        let runtimeInputs = try KVTunerTestFixtures.candidateRuntimeInputs()
        let policy = try KVTunerTestFixtures.candidateRuntimePolicy(
            runtimeInputs, candidateOrdinal: 0)
        let runtimeContract = try KVTunerCandidateRuntimeContract.load(
            exactModelConfigData: runtimeInputs.configData,
            runtimePolicy: policy,
            eosTokenID: 255)
        let rows = try runtimeInputs.manifest.searchPrompts.enumerated().map {
            ordinal, prompt in
            let target = runtimeInputs.manifest.searchNormalizedTargets[ordinal]
            let answer = ordinal == wrongOrdinal ? "999999999" : target
            // The first stop in protocol order ("Question:") occurs later in the output than
            // `<|im_end|>`. This proves validation chooses the earliest output position rather
            // than the first matching entry in the protocol's stop-sequence array.
            let rawPrefix =
                "reasoning... final answer: $\(answer).<|im_end|>ignoredQuestion:later"
            let generatedTokenIDs = Array(rawPrefix.utf8).map(Int.init) + [255]
            let raw = try decodeTokenIDs(generatedTokenIDs)
            return KVTunerCandidateOutputRow(
                ordinal: ordinal,
                promptSHA256: prompt.promptSHA256,
                promptTokenIDsSHA256: prompt.tokenIDsSHA256,
                generatedTokenIDs: generatedTokenIDs,
                rawDecodedUTF8: Data(raw.utf8),
                outputUTF8: Data(
                    "reasoning... final answer: $\(answer).".utf8),
                finishReason: .endOfSequence,
                runtimeReceipt: try receipt(
                    policy: policy,
                    contract: runtimeContract,
                    promptTokenCount: prompt.tokenIDs.count,
                    generatedTokenCount: generatedTokenIDs.count))
        }
        let artifact = KVTunerCandidateEvaluationArtifact(
            schemaVersion: 2,
            evaluationProtocol: .canonical,
            promptManifestSHA256: sha256Hex(runtimeInputs.manifestData),
            candidateOrdinal: policy.candidateOrdinal,
            candidateSHA256: policy.candidateSHA256,
            runtimePolicySHA256: policy.runtimePolicySHA256,
            executionEnvironment: environment(policy: policy),
            rows: rows)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return Inputs(
            artifact: artifact,
            data: try encoder.encode(artifact),
            runtimePolicy: policy,
            runtimeContract: runtimeContract,
            configData: runtimeInputs.configData,
            manifest: runtimeInputs.manifest,
            manifestData: runtimeInputs.manifestData)
    }

    private func validate(
        _ inputs: Inputs,
        artifact: KVTunerCandidateEvaluationArtifact? = nil,
        data: Data? = nil,
        runtimePolicy: KVTunerCandidateRuntimePolicy? = nil,
        decodeTokenIDs: (([Int]) throws -> String)? = nil
    ) throws -> KVTunerCandidateEvaluation {
        let artifact = artifact ?? inputs.artifact
        return try artifact.validated(
            exactArtifactData: data ?? inputs.data,
            runtimePolicy: runtimePolicy ?? inputs.runtimePolicy,
            runtimeContract: inputs.runtimeContract,
            calibrationManifest: inputs.manifest,
            exactCalibrationManifestData: inputs.manifestData,
            decodeTokenIDs: decodeTokenIDs ?? self.decodeTokenIDs)
    }

    func testScorerMatchesPinnedFlexibleExtractAndExactMatchRules() throws {
        XCTAssertEqual(
            try KVTunerGSM8KScorer.flexibleExtract(
                Data("first 2, then 3".utf8)),
            "3")
        XCTAssertTrue(try KVTunerGSM8KScorer.isCorrect(
            outputUTF8: Data("work... $1,234.".utf8),
            normalizedTarget: "1234"))
        XCTAssertTrue(try KVTunerGSM8KScorer.isCorrect(
            outputUTF8: Data("#### -7".utf8),
            normalizedTarget: "-7"))
        XCTAssertEqual(
            try KVTunerGSM8KScorer.flexibleExtract(
                Data("no numeric answer".utf8)),
            "[invalid]")
        XCTAssertThrowsError(
            try KVTunerGSM8KScorer.flexibleExtract(Data([0xff])))
    }

    func testSchemaTwoReplaysTokensStopTrimmingScoreAndStorageReceipt() throws {
        let inputs = try inputs(wrongAt: 37)
        let summary = try validate(inputs)

        XCTAssertEqual(summary.candidateOrdinal, 0)
        XCTAssertEqual(summary.correctCount, 199)
        XCTAssertEqual(summary.totalCount, 200)
        XCTAssertEqual(summary.outputSHA256, sha256Hex(inputs.data))
        XCTAssertEqual(
            summary.runtimePolicySHA256,
            inputs.runtimePolicy.runtimePolicySHA256)
        XCTAssertEqual(summary.environmentSHA256.count, 64)
    }

    func testDecoderClassifiesSchemaBeforeCurrentShape() {
        XCTAssertThrowsError(try JSONDecoder().decode(
            KVTunerCandidateEvaluationArtifact.self,
            from: Data(#"{"schemaVersion":1}"#.utf8))) { error in
                XCTAssertEqual(
                    error as? KVTunerCandidateEvaluationArtifactError,
                    .unsupportedSchema(1))
            }
    }

    func testEvaluationRejectsTokenDecodeAndCanonicalStopMutations() throws {
        let inputs = try inputs()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]

        var mutated = inputs.artifact
        mutated.rows[0].promptTokenIDsSHA256 = String(repeating: "f", count: 64)
        XCTAssertThrowsError(try validate(
            inputs,
            artifact: mutated,
            data: try encoder.encode(mutated)))

        mutated = inputs.artifact
        mutated.rows[0].rawDecodedUTF8 = Data("not decoded bytes".utf8)
        XCTAssertThrowsError(try validate(
            inputs,
            artifact: mutated,
            data: try encoder.encode(mutated)))

        mutated = inputs.artifact
        mutated.rows[0].outputUTF8 = mutated.rows[0].rawDecodedUTF8
        XCTAssertThrowsError(try validate(
            inputs,
            artifact: mutated,
            data: try encoder.encode(mutated)))

        XCTAssertThrowsError(try validate(
            inputs,
            decodeTokenIDs: { _ in "substituted decoder output" }))
    }

    func testEvaluationRequiresOneThrough256GeneratedTokens() throws {
        let inputs = try inputs()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]

        for generatedTokenIDs in [[], Array(repeating: 65, count: 257)] {
            var mutated = inputs.artifact
            mutated.rows[0].generatedTokenIDs = generatedTokenIDs
            XCTAssertThrowsError(try validate(
                inputs,
                artifact: mutated,
                data: try encoder.encode(mutated)))
        }
    }

    func testEvaluationRequiresAuthenticatedEOSOrExactBudgetTermination() throws {
        let inputs = try inputs()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]

        var mutated = inputs.artifact
        mutated.rows[0].finishReason = .generationBudgetExhausted
        XCTAssertThrowsError(try validate(
            inputs,
            artifact: mutated,
            data: try encoder.encode(mutated))) { error in
                XCTAssertEqual(
                    error as? KVTunerCandidateEvaluationArtifactError,
                    .invalidFinishReason(0))
            }

        mutated = inputs.artifact
        mutated.rows[0].generatedTokenIDs[0] = 255
        mutated.rows[0].rawDecodedUTF8 = Data(
            try decodeTokenIDs(mutated.rows[0].generatedTokenIDs).utf8)
        XCTAssertThrowsError(try validate(
            inputs,
            artifact: mutated,
            data: try encoder.encode(mutated))) { error in
                XCTAssertEqual(
                    error as? KVTunerCandidateEvaluationArtifactError,
                    .invalidFinishReason(0))
            }

        mutated = inputs.artifact
        mutated.rows[0].generatedTokenIDs.removeLast()
        mutated.rows[0].rawDecodedUTF8 = Data(
            try decodeTokenIDs(mutated.rows[0].generatedTokenIDs).utf8)
        mutated.rows[0].runtimeReceipt = try receipt(
            policy: inputs.runtimePolicy,
            contract: inputs.runtimeContract,
            promptTokenCount: inputs.manifest.searchPrompts[0].tokenIDs.count,
            generatedTokenCount: mutated.rows[0].generatedTokenIDs.count)
        XCTAssertThrowsError(try validate(
            inputs,
            artifact: mutated,
            data: try encoder.encode(mutated))) { error in
                XCTAssertEqual(
                    error as? KVTunerCandidateEvaluationArtifactError,
                    .invalidFinishReason(0))
            }

        mutated = inputs.artifact
        mutated.rows[0].generatedTokenIDs = Array(repeating: 65, count: 256)
        mutated.rows[0].rawDecodedUTF8 = Data(repeating: 65, count: 256)
        mutated.rows[0].outputUTF8 = Data(repeating: 65, count: 256)
        mutated.rows[0].finishReason = .generationBudgetExhausted
        mutated.rows[0].runtimeReceipt = try receipt(
            policy: inputs.runtimePolicy,
            contract: inputs.runtimeContract,
            promptTokenCount: inputs.manifest.searchPrompts[0].tokenIDs.count,
            generatedTokenCount: 256)
        _ = try validate(
            inputs,
            artifact: mutated,
            data: try encoder.encode(mutated))

        mutated.rows[0].generatedTokenIDs[0] = 255
        mutated.rows[0].rawDecodedUTF8 = Data(
            try decodeTokenIDs(mutated.rows[0].generatedTokenIDs).utf8)
        XCTAssertThrowsError(try validate(
            inputs,
            artifact: mutated,
            data: try encoder.encode(mutated))) { error in
                XCTAssertEqual(
                    error as? KVTunerCandidateEvaluationArtifactError,
                    .invalidFinishReason(0))
            }

        mutated.rows[0].generatedTokenIDs[0] = 65
        mutated.rows[0].generatedTokenIDs[255] = 255
        mutated.rows[0].rawDecodedUTF8 = Data(
            try decodeTokenIDs(mutated.rows[0].generatedTokenIDs).utf8)
        XCTAssertThrowsError(try validate(
            inputs,
            artifact: mutated,
            data: try encoder.encode(mutated))) { error in
                XCTAssertEqual(
                    error as? KVTunerCandidateEvaluationArtifactError,
                    .invalidFinishReason(0))
            }
    }

    func testEnvironmentRejectsDirtyDebugPathAndIdentitySubstitution() throws {
        let inputs = try inputs()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]

        let mutations: (inout KVTunerCandidateExecutionEnvironment) -> Void = {
            $0.harnessGitSHA += "-dirty"
        }
        var mutated = inputs.artifact
        mutations(&mutated.executionEnvironment)
        XCTAssertThrowsError(try validate(
            inputs,
            artifact: mutated,
            data: try encoder.encode(mutated)))

        for mutate in [
            { (value: inout KVTunerCandidateExecutionEnvironment) in
                value.buildConfiguration = "Debug"
            },
            { (value: inout KVTunerCandidateExecutionEnvironment) in
                value.mlxSwiftVersion = "0.32.0"
            },
            { (value: inout KVTunerCandidateExecutionEnvironment) in
                value.mlxSwiftLMRevision = String(repeating: "b", count: 40)
            },
            { (value: inout KVTunerCandidateExecutionEnvironment) in
                value.hardwareChip = "/tmp/bench-machine"
            },
            { (value: inout KVTunerCandidateExecutionEnvironment) in
                value.modelConfigSHA256 = String(repeating: "d", count: 64)
            },
            { (value: inout KVTunerCandidateExecutionEnvironment) in
                value.checkpointManifestHash = "0000000000000000"
            },
            { (value: inout KVTunerCandidateExecutionEnvironment) in
                value.tokenizerSHA256 = String(repeating: "0", count: 64)
            },
        ] {
            mutated = inputs.artifact
            mutate(&mutated.executionEnvironment)
            XCTAssertThrowsError(try validate(
                inputs,
                artifact: mutated,
                data: try encoder.encode(mutated)))
        }
    }

    func testRuntimeReceiptRejectsPolicyCacheGeometryAndByteSubstitution() throws {
        let inputs = try inputs()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let mutations: [(inout KVTunerCandidateRuntimeReceipt) -> Void] = [
            { $0.runtimePolicySHA256 = String(repeating: "0", count: 64) },
            { $0.cachedTokens += 1 },
            { $0.layers[0] = KVTunerRuntimeLayerPolicy(
                layer: 0, keyBits: 4, valueBits: 2) },
            { $0.geometry.kvHeadCount += 1 },
            { $0.groupSize = 128 },
            { $0.sequenceCount = 2 },
            { $0.metadataScalarBytes = 4 },
            { $0.actualPayloadBytes += 1 },
            { $0.actualMetadataBytes += 1 },
            { $0.actualControlBytes += 1 },
            { $0.actualWorkspaceBytes += 1 },
            { $0.actualTotalPersistentBytes += 1 },
            { $0.actualTotalBytes += 1 },
        ]

        for mutate in mutations {
            var mutated = inputs.artifact
            mutate(&mutated.rows[0].runtimeReceipt)
            XCTAssertThrowsError(try validate(
                inputs,
                artifact: mutated,
                data: try encoder.encode(mutated)))
        }
    }

    func testRuntimeReceiptRejectsCoherentContractAndByteSubstitution() throws {
        let inputs = try inputs()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        var mutated = inputs.artifact
        var receipt = mutated.rows[0].runtimeReceipt
        receipt.capacityTokens += 256
        receipt.geometry = KVTunerCandidateRuntimeGeometry(
            layerCount: receipt.geometry.layerCount,
            kvHeadCount: receipt.geometry.kvHeadCount + 1,
            headDimension: receipt.geometry.headDimension)
        receipt.sequenceCount = 2
        receipt.metadataScalarBytes = 4
        receipt.actualWorkspaceBytes = receipt.sequenceCount
            * receipt.geometry.kvHeadCount
            * receipt.capacityTokens
            * receipt.geometry.headDimension
            * 2
            * receipt.metadataScalarBytes
        let allocation = try KVStorageFormat.kvtunerAllocation(
            layerPolicy: inputs.runtimePolicy.layers.map {
                KVLayerPrecision(
                    layer: $0.layer,
                    keyBits: $0.keyBits,
                    valueBits: $0.valueBits)
            },
            groupSize: inputs.runtimePolicy.groupSize,
            geometry: KVStorageGeometry(
                layerCount: receipt.geometry.layerCount,
                kvHeadCount: receipt.geometry.kvHeadCount,
                headDimension: receipt.geometry.headDimension),
            capacityTokens: receipt.capacityTokens,
            sequences: receipt.sequenceCount,
            metadataScalarBytes: receipt.metadataScalarBytes,
            maximumLayerWorkspaceBytes: receipt.actualWorkspaceBytes)
        receipt.actualPayloadBytes = allocation.payloadBytes
        receipt.actualMetadataBytes = allocation.metadataBytes
        receipt.actualControlBytes = allocation.controlBytes
        receipt.actualTotalPersistentBytes = allocation.totalPersistentBytes
        receipt.actualTotalBytes = allocation.totalBytes
        mutated.rows[0].runtimeReceipt = receipt

        XCTAssertThrowsError(try validate(
            inputs,
            artifact: mutated,
            data: try encoder.encode(mutated))) { error in
                XCTAssertEqual(
                    error as? KVTunerCandidateEvaluationArtifactError,
                    .runtimeReceiptMismatch(0))
            }
    }

    func testRuntimeContractAuthenticatesPinnedQwenGeometryAndEOS() throws {
        let inputs = try inputs()
        XCTAssertEqual(inputs.runtimeContract.geometry.layerCount, 4)
        XCTAssertEqual(inputs.runtimeContract.geometry.kvHeadCount, 8)
        XCTAssertEqual(inputs.runtimeContract.geometry.headDimension, 128)
        XCTAssertEqual(inputs.runtimeContract.metadataScalarBytes, 2)
        XCTAssertEqual(inputs.runtimeContract.sequenceCount, 1)
        XCTAssertEqual(inputs.runtimeContract.eosTokenID, 255)
        XCTAssertEqual(try inputs.runtimeContract.capacityTokens(
            promptTokenCount: 3,
            generatedTokenCount: 256,
            previousCapacityTokens: nil), 512)
        XCTAssertEqual(try inputs.runtimeContract.capacityTokens(
            promptTokenCount: 600,
            generatedTokenCount: 256,
            previousCapacityTokens: 512), 1_024)

        var wrongConfig = try XCTUnwrap(
            JSONSerialization.jsonObject(with: inputs.configData)
                as? [String: Any])
        wrongConfig["num_key_value_heads"] = 3
        let wrongData = try JSONSerialization.data(withJSONObject: wrongConfig)
        XCTAssertThrowsError(try KVTunerCandidateRuntimeContract.load(
            exactModelConfigData: wrongData,
            runtimePolicy: inputs.runtimePolicy,
            eosTokenID: 255))
        XCTAssertThrowsError(try KVTunerCandidateRuntimeContract.load(
            exactModelConfigData: inputs.configData,
            runtimePolicy: inputs.runtimePolicy,
            eosTokenID: -1))
        XCTAssertThrowsError(try KVTunerCandidateRuntimeContract.load(
            exactModelConfigData: inputs.configData,
            runtimePolicy: inputs.runtimePolicy,
            eosTokenID: 254)) { error in
                XCTAssertEqual(
                    error as? KVTunerCandidateRuntimeContractError,
                    .invalidEOSTokenID)
            }
    }
}
