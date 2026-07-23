import Foundation
import XCTest

@testable import HarnessCore

final class KVTunerQualificationProducerTests: XCTestCase {
    private struct PromptFixture: Decodable {
        let sensitivity: [String]
        let search: [String]
    }

    private func resourceData(_ name: String) throws -> Data {
        let url = try XCTUnwrap(Bundle.module.url(
            forResource: name, withExtension: "json"))
        return try Data(contentsOf: url)
    }

    private func manifestInputs() throws -> (
        config: Data,
        promptFixture: Data,
        normalizedTargets: Data,
        tokenIDs: [String: [Int]]
    ) {
        let config = Data(
            #"{"eos_token_id":255,"head_dim":128,"hidden_size":5120,"model_type":"qwen3","num_attention_heads":64,"num_hidden_layers":4,"num_key_value_heads":8,"torch_dtype":"bfloat16"}"#.utf8)
        let promptFixture = try resourceData("kvtuner-gsm8k-prompts")
        let normalizedTargets = try resourceData(
            "kvtuner-gsm8k-normalized-targets")
        let prompts = try JSONDecoder().decode(
            PromptFixture.self, from: promptFixture)
        var tokenIDs: [String: [Int]] = [:]
        for (ordinal, prompt) in prompts.sensitivity.enumerated() {
            tokenIDs[prompt] = [1, ordinal + 10, ordinal + 1_000]
        }
        for (ordinal, prompt) in prompts.search.enumerated() {
            tokenIDs[prompt] = [2, ordinal + 10, ordinal + 1_000]
        }
        return (config, promptFixture, normalizedTargets, tokenIDs)
    }

    private func sensitivitySamples(
        manifest: KVTunerCalibrationManifest,
        layerCount: Int
    ) -> [KVTunerSensitivitySample] {
        var samples: [KVTunerSensitivitySample] = []
        for promptIndex in manifest.sensitivityPrompts.indices {
            for layer in 0..<layerCount {
                for pair in KVTunerSensitivityArtifact.canonicalPrecisionPairs {
                    samples.append(KVTunerSensitivitySample(
                        promptIndex: promptIndex,
                        layer: layer,
                        keyBits: pair.keyBits,
                        valueBits: pair.valueBits,
                        relativeKeyError: 0.01,
                        relativeValueError: 0.02,
                        attentionScoreError: 0.03,
                        relativeAttentionOutputError: 0.04))
                }
            }
        }
        return samples
    }

    private func decodeTokenIDs(_ tokenIDs: [Int]) throws -> String {
        guard tokenIDs.allSatisfy({ (0...255).contains($0) }) else {
            throw CocoaError(.coderInvalidValue)
        }
        return String(decoding: tokenIDs.map(UInt8.init), as: UTF8.self)
    }

    private func executionEnvironment(
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
            memoryCacheLimitBytes:
                KVTunerCandidateExecutionEnvironment
                    .requiredMemoryCacheLimitBytes,
            hardwareOS: "macOS 26.5.2",
            modelConfigHash: policy.modelConfigHash,
            modelConfigSHA256: policy.modelConfigSHA256,
            checkpointManifestHash: policy.checkpointManifestHash,
            checkpointContentSHA256: policy.checkpointContentSHA256,
            tokenizerSHA256: policy.tokenizerSHA256)
    }

    private func runtimeReceipt(
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

    private func candidateEvaluationData(
        inputs: KVTunerTestFixtures.CandidateRuntimeInputs
    ) throws -> [Data] {
        try inputs.candidates.indices.map { candidateOrdinal in
            let policy = try KVTunerTestFixtures.candidateRuntimePolicy(
                inputs, candidateOrdinal: candidateOrdinal)
            let contract = try KVTunerCandidateRuntimeContract.load(
                exactModelConfigData: inputs.configData,
                runtimePolicy: policy,
                eosTokenID: 255)
            let rows = try inputs.manifest.searchPrompts.enumerated().map {
                ordinal, prompt in
                let target = inputs.manifest.searchNormalizedTargets[ordinal]
                let rawOutput = target + "<|im_end|>"
                let generatedTokenIDs =
                    Array(rawOutput.utf8).map(Int.init) + [255]
                return KVTunerCandidateGeneratedRow(
                    promptOrdinal: ordinal,
                    promptTokenIDsSHA256: prompt.tokenIDsSHA256,
                    generatedTokenIDs: generatedTokenIDs,
                    finishReason: .endOfSequence,
                    runtimeReceipt: try runtimeReceipt(
                        policy: policy,
                        contract: contract,
                        promptTokenCount: prompt.tokenIDs.count,
                        generatedTokenCount: generatedTokenIDs.count))
            }
            let artifact = try KVTunerCandidateEvaluationArtifact
                .makeAuthenticated(
                    runtimePolicy: policy,
                    runtimeContract: contract,
                    executionEnvironment: executionEnvironment(policy: policy),
                    calibrationManifest: inputs.manifest,
                    exactCalibrationManifestData: inputs.manifestData,
                    generatedRows: rows,
                    decodeTokenIDs: decodeTokenIDs)
            return try KVTunerArtifactCodec.encode(artifact)
        }
    }

    func testManifestProducerAuthenticatesPinnedSourcesAndLiveTokenization() throws {
        let inputs = try manifestInputs()
        let manifest = try KVTunerCalibrationManifest.makePinnedGSM8K(
            promptFixtureData: inputs.promptFixture,
            normalizedTargetsData: inputs.normalizedTargets,
            exactModelConfigData: inputs.config,
            checkpointManifestHash: "fedcba9876543210",
            checkpointContentSHA256: String(repeating: "e", count: 64),
            tokenizerSHA256: String(repeating: "a", count: 64),
            tokenizePrompt: { try XCTUnwrap(inputs.tokenIDs[$0]) })

        XCTAssertEqual(try manifest.validated(), manifest)
        XCTAssertEqual(manifest.modelConfigHash, fnv1a64(inputs.config))
        XCTAssertEqual(manifest.modelConfigSHA256, sha256Hex(inputs.config))
        XCTAssertEqual(manifest.sensitivityPrompts[0].tokenIDs, [1, 10, 1_000])
        XCTAssertEqual(manifest.searchPrompts[199].tokenIDs, [2, 209, 1_199])

        let encoded = try KVTunerArtifactCodec.encode(manifest)
        XCTAssertEqual(
            try JSONDecoder().decode(
                KVTunerCalibrationManifest.self, from: encoded),
            manifest)
    }

    func testManifestProducerRejectsSourceAndTokenizerSubstitution() throws {
        let inputs = try manifestInputs()
        var fixture = try XCTUnwrap(
            JSONSerialization.jsonObject(with: inputs.promptFixture)
                as? [String: Any])
        var sensitivity = try XCTUnwrap(fixture["sensitivity"] as? [String])
        sensitivity[0].append(" substituted")
        let substitutedPrompt = sensitivity[0]
        fixture["sensitivity"] = sensitivity
        let changedFixture = try JSONSerialization.data(
            withJSONObject: fixture)

        XCTAssertThrowsError(try KVTunerCalibrationManifest.makePinnedGSM8K(
            promptFixtureData: changedFixture,
            normalizedTargetsData: inputs.normalizedTargets,
            exactModelConfigData: inputs.config,
            checkpointManifestHash: "fedcba9876543210",
            checkpointContentSHA256: String(repeating: "e", count: 64),
            tokenizerSHA256: String(repeating: "a", count: 64),
            tokenizePrompt: { prompt in
                if prompt == substitutedPrompt {
                    return [9, 99, 999]
                }
                guard let tokenIDs = inputs.tokenIDs[prompt] else {
                    throw CocoaError(.coderValueNotFound)
                }
                return tokenIDs
            })) { error in
                XCTAssertEqual(
                    error as? KVTunerCalibrationManifestError,
                    .sourcePromptIdentityMismatch("sensitivity-sha256"))
            }

        XCTAssertThrowsError(try KVTunerCalibrationManifest.makePinnedGSM8K(
            promptFixtureData: inputs.promptFixture,
            normalizedTargetsData: inputs.normalizedTargets,
            exactModelConfigData: inputs.config,
            checkpointManifestHash: "fedcba9876543210",
            checkpointContentSHA256: String(repeating: "e", count: 64),
            tokenizerSHA256: String(repeating: "a", count: 64),
            tokenizePrompt: { _ in [] }))
    }

    func testSensitivityProducerBindsExactManifestAndCanonicalProtocol() throws {
        let inputs = try manifestInputs()
        let manifest = try KVTunerCalibrationManifest.makePinnedGSM8K(
            promptFixtureData: inputs.promptFixture,
            normalizedTargetsData: inputs.normalizedTargets,
            exactModelConfigData: inputs.config,
            checkpointManifestHash: "fedcba9876543210",
            checkpointContentSHA256: String(repeating: "e", count: 64),
            tokenizerSHA256: String(repeating: "a", count: 64),
            tokenizePrompt: { try XCTUnwrap(inputs.tokenIDs[$0]) })
        let manifestData = try KVTunerArtifactCodec.encode(manifest)
        let samples = sensitivitySamples(manifest: manifest, layerCount: 4)

        let artifact = try KVTunerSensitivityArtifact.makeAuthenticated(
            matrixID: "kvarn-qwen3-32b-v1",
            groupSize: 128,
            layerCount: 4,
            samples: samples,
            captureEnvironment:
                KVTunerTestFixtures.sensitivityCaptureEnvironment(),
            calibrationManifest: manifest,
            exactCalibrationManifestData: manifestData)

        XCTAssertEqual(artifact.captureMode,
                       "single-prefill-no-error-propagation-v1")
        XCTAssertEqual(artifact.metricAccumulationDType, "float32")
        XCTAssertEqual(artifact.promptManifestSHA256, sha256Hex(manifestData))
        XCTAssertNoThrow(try artifact.validated(
            calibrationManifest: manifest,
            exactCalibrationManifestData: manifestData))

        XCTAssertThrowsError(try KVTunerSensitivityArtifact.makeAuthenticated(
            matrixID: "kvarn-qwen3-32b-v1",
            groupSize: 128,
            layerCount: 4,
            samples: Array(samples.dropLast()),
            captureEnvironment:
                KVTunerTestFixtures.sensitivityCaptureEnvironment(),
            calibrationManifest: manifest,
            exactCalibrationManifestData: manifestData))
    }

    func testSensitivityProducerRejectsExactManifestByteSubstitution() throws {
        let inputs = try KVTunerTestFixtures.candidateRuntimeInputs()
        var substitutedManifest = inputs.manifest
        substitutedManifest.checkpointManifestHash = "0123456789abcdef"
        let substitutedData = try KVTunerArtifactCodec.encode(
            substitutedManifest.validated())

        XCTAssertThrowsError(try KVTunerSensitivityArtifact.makeAuthenticated(
            matrixID: inputs.sensitivity.matrixID,
            groupSize: inputs.sensitivity.groupSize,
            layerCount: inputs.sensitivity.layerCount,
            samples: inputs.sensitivity.samples,
            captureEnvironment:
                KVTunerTestFixtures.sensitivityCaptureEnvironment(),
            calibrationManifest: inputs.manifest,
            exactCalibrationManifestData: substitutedData
        )) { error in
            XCTAssertEqual(
                error as? KVTunerSensitivityError,
                .calibrationManifestMismatch)
        }
    }

    func testCandidateProducerRejectsReorderedAndWrongDigestRows() throws {
        let inputs = try KVTunerTestFixtures.candidateRuntimeInputs()
        let policy = try KVTunerTestFixtures.candidateRuntimePolicy(
            inputs, candidateOrdinal: 0)
        let contract = try KVTunerCandidateRuntimeContract.load(
            exactModelConfigData: inputs.configData,
            runtimePolicy: policy,
            eosTokenID: 255)
        let receipt = try runtimeReceipt(
            policy: policy,
            contract: contract,
            promptTokenCount: inputs.manifest.searchPrompts[0].tokenIDs.count,
            generatedTokenCount: 1)
        let rows = inputs.manifest.searchPrompts.enumerated().map {
            ordinal, prompt in
            KVTunerCandidateGeneratedRow(
                promptOrdinal: ordinal,
                promptTokenIDsSHA256: prompt.tokenIDsSHA256,
                generatedTokenIDs: [255],
                finishReason: .endOfSequence,
                runtimeReceipt: receipt)
        }

        func make(
            _ generatedRows: [KVTunerCandidateGeneratedRow]
        ) throws -> KVTunerCandidateEvaluationArtifact {
            try KVTunerCandidateEvaluationArtifact.makeAuthenticated(
                runtimePolicy: policy,
                runtimeContract: contract,
                executionEnvironment: executionEnvironment(policy: policy),
                calibrationManifest: inputs.manifest,
                exactCalibrationManifestData: inputs.manifestData,
                generatedRows: generatedRows,
                decodeTokenIDs: decodeTokenIDs)
        }

        XCTAssertNoThrow(try make(rows))

        var reordered = rows
        reordered.swapAt(0, 1)
        XCTAssertThrowsError(try make(reordered)) { error in
            XCTAssertEqual(
                error as? KVTunerCandidateEvaluationArtifactError,
                .invalidRow(0))
        }

        var wrongDigest = rows
        wrongDigest[0] = KVTunerCandidateGeneratedRow(
            promptOrdinal: 0,
            promptTokenIDsSHA256: String(repeating: "f", count: 64),
            generatedTokenIDs: [255],
            finishReason: .endOfSequence,
            runtimeReceipt: receipt)
        XCTAssertThrowsError(try make(wrongDigest)) { error in
            XCTAssertEqual(
                error as? KVTunerCandidateEvaluationArtifactError,
                .invalidRow(0))
        }
    }

    func testSearchProducerRejectsIncompleteCandidateArtifacts() throws {
        let inputs = try KVTunerTestFixtures.candidateRuntimeInputs()

        XCTAssertThrowsError(try KVTunerSearchArtifact.makeAuthenticated(
            targetPairBitTotal: 36,
            maxCandidates: 10,
            sensitivityArtifact: inputs.sensitivity,
            exactSensitivityArtifactData: inputs.sensitivityData,
            calibrationManifest: inputs.manifest,
            exactCalibrationManifestData: inputs.manifestData,
            exactCandidateEvaluationArtifactData: [],
            exactModelConfigData: inputs.configData,
            eosTokenID: 255,
            tokenizePrompt:
                KVTunerTestFixtures.tokenizer(for: inputs.manifest),
            decodeTokenIDs: decodeTokenIDs
        )) { error in
            XCTAssertEqual(
                error as? KVTunerScheduleSearchError,
                .incompleteEvaluations(
                    expected: inputs.candidates.count,
                    actual: 0))
        }
    }

    func testBundleProducerRejectsSuppliedScheduleMismatch() throws {
        let inputs = try KVTunerTestFixtures.candidateRuntimeInputs()
        let evaluationData = try candidateEvaluationData(inputs: inputs)
        let search = try KVTunerSearchArtifact.makeAuthenticated(
            targetPairBitTotal: 36,
            maxCandidates: 10,
            sensitivityArtifact: inputs.sensitivity,
            exactSensitivityArtifactData: inputs.sensitivityData,
            calibrationManifest: inputs.manifest,
            exactCalibrationManifestData: inputs.manifestData,
            exactCandidateEvaluationArtifactData: evaluationData,
            exactModelConfigData: inputs.configData,
            eosTokenID: 255,
            tokenizePrompt:
                KVTunerTestFixtures.tokenizer(for: inputs.manifest),
            decodeTokenIDs: decodeTokenIDs)
        let searchData = try KVTunerArtifactCodec.encode(search)
        let runtimePolicies = try search.candidates.indices.map {
            try KVTunerTestFixtures.candidateRuntimePolicy(
                inputs, candidateOrdinal: $0)
        }
        var schedule = try KVTunerScheduleSearch.makeSchedule(
            searchArtifact: search,
            exactSearchArtifactData: searchData,
            sensitivityArtifact: inputs.sensitivity,
            exactSensitivityArtifactData: inputs.sensitivityData,
            calibrationManifest: inputs.manifest,
            exactCalibrationManifestData: inputs.manifestData,
            exactCandidateEvaluationArtifactData: evaluationData,
            candidateRuntimePolicies: runtimePolicies,
            eosTokenID: 255,
            decodeTokenIDs: decodeTokenIDs,
            exactModelConfigData: inputs.configData,
            expectedCheckpointManifestHash:
                inputs.manifest.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                inputs.manifest.checkpointContentSHA256)
        schedule.cellID += "-substituted"

        XCTAssertThrowsError(try KVTunerQualificationBundle.makeAuthenticated(
            exactScheduleData: KVTunerArtifactCodec.encode(schedule),
            exactCalibrationManifestData: inputs.manifestData,
            exactSensitivityArtifactData: inputs.sensitivityData,
            exactSearchArtifactData: searchData,
            exactCandidateEvaluationArtifactData: evaluationData,
            exactModelConfigData: inputs.configData,
            eosTokenID: 255,
            expectedCheckpointManifestHash:
                inputs.manifest.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                inputs.manifest.checkpointContentSHA256,
            tokenizePrompt:
                KVTunerTestFixtures.tokenizer(for: inputs.manifest),
            decodeTokenIDs: decodeTokenIDs
        )) { error in
            XCTAssertEqual(
                error as? KVTunerScheduleSearchError,
                .searchArtifactMismatch)
        }
    }
}
