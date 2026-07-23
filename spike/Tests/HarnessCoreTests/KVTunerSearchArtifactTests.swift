import Foundation
import XCTest

@testable import HarnessCore

final class KVTunerSearchArtifactTests: XCTestCase {
    private struct Inputs {
        let configData: Data
        let manifest: KVTunerCalibrationManifest
        let manifestData: Data
        let sensitivity: KVTunerSensitivityArtifact
        let sensitivityData: Data
        let candidates: [KVTunerScheduleCandidate]
        let runtimePolicies: [KVTunerCandidateRuntimePolicy]
        let runtimeContract: KVTunerCandidateRuntimeContract
        let evaluations: [KVTunerCandidateEvaluation]
        let evaluationData: [Data]
        let search: KVTunerSearchArtifact
        let searchData: Data
    }

    private func decodeTokenIDs(_ tokenIDs: [Int]) throws -> String {
        guard tokenIDs.allSatisfy({ (0...255).contains($0) }) else {
            throw CocoaError(.coderInvalidValue)
        }
        return String(decoding: tokenIDs.map(UInt8.init), as: UTF8.self)
    }

    private func environment(
        _ policy: KVTunerCandidateRuntimePolicy
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

    private func receipt(
        _ policy: KVTunerCandidateRuntimePolicy,
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

    private func inputs() throws -> Inputs {
        let runtimeInputs = try KVTunerTestFixtures.candidateRuntimeInputs()
        let configData = runtimeInputs.configData
        let manifest = runtimeInputs.manifest
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let manifestData = runtimeInputs.manifestData
        let manifestSHA = sha256Hex(manifestData)
        let sensitivity = runtimeInputs.sensitivity
        let sensitivityData = runtimeInputs.sensitivityData
        let candidates = runtimeInputs.candidates
        let runtimePolicies = try candidates.indices.map {
            try KVTunerTestFixtures.candidateRuntimePolicy(
                runtimeInputs, candidateOrdinal: $0)
        }
        let runtimeContract = try KVTunerCandidateRuntimeContract.load(
            exactModelConfigData: configData,
            runtimePolicy: runtimePolicies[0],
            eosTokenID: 255)
        let evaluationArtifacts = try zip(candidates, runtimePolicies).map {
            candidate, policy in
            KVTunerCandidateEvaluationArtifact(
                schemaVersion: 3,
                evaluationProtocol: .canonical,
                promptManifestSHA256: manifestSHA,
                candidateOrdinal: candidate.ordinal,
                candidateSHA256:
                    try KVTunerScheduleSearch.candidateSHA256(candidate),
                runtimePolicySHA256: policy.runtimePolicySHA256,
                executionEnvironment: environment(policy),
                rows: try manifest.searchPrompts.enumerated().map {
                    ordinal, prompt in
                    let target = manifest.searchNormalizedTargets[ordinal]
                    let wrong = candidate.ordinal == 1 && ordinal == 199
                    let output = wrong ? "999999999" : target
                    let rawPrefix = output + "<|im_end|>"
                    let generatedTokenIDs =
                        Array(rawPrefix.utf8).map(Int.init) + [255]
                    let rawOutput = try decodeTokenIDs(generatedTokenIDs)
                    return KVTunerCandidateOutputRow(
                        ordinal: ordinal,
                        promptSHA256: prompt.promptSHA256,
                        promptTokenIDsSHA256: prompt.tokenIDsSHA256,
                        generatedTokenIDs: generatedTokenIDs,
                        rawDecodedUTF8: Data(rawOutput.utf8),
                        outputUTF8: Data(output.utf8),
                        finishReason: .endOfSequence,
                        runtimeReceipt: try receipt(
                            policy,
                            contract: runtimeContract,
                            promptTokenCount: prompt.tokenIDs.count,
                            generatedTokenCount: generatedTokenIDs.count))
                })
        }
        let evaluationData = try evaluationArtifacts.map(encoder.encode)
        let evaluations = try zip(
            evaluationArtifacts,
            zip(evaluationData, runtimePolicies)
        ).map { artifact, inputs in
            try artifact.validated(
                exactArtifactData: inputs.0,
                runtimePolicy: inputs.1,
                runtimeContract: runtimeContract,
                calibrationManifest: manifest,
                exactCalibrationManifestData: manifestData,
                decodeTokenIDs: decodeTokenIDs)
        }
        let search = try KVTunerSearchArtifact.makeAuthenticated(
            targetPairBitTotal: 36,
            maxCandidates: 10,
            sensitivityArtifact: sensitivity,
            exactSensitivityArtifactData: sensitivityData,
            calibrationManifest: manifest,
            exactCalibrationManifestData: manifestData,
            exactCandidateEvaluationArtifactData: evaluationData,
            exactModelConfigData: configData,
            eosTokenID: 255,
            tokenizePrompt: KVTunerTestFixtures.tokenizer(for: manifest),
            decodeTokenIDs: decodeTokenIDs)
        let searchData = try KVTunerArtifactCodec.encode(search)
        return Inputs(
            configData: configData,
            manifest: manifest,
            manifestData: manifestData,
            sensitivity: sensitivity,
            sensitivityData: sensitivityData,
            candidates: candidates,
            runtimePolicies: runtimePolicies,
            runtimeContract: runtimeContract,
            evaluations: evaluations,
            evaluationData: evaluationData,
            search: search,
            searchData: searchData)
    }

    func testCompleteSearchArtifactValidatesAndEmitsSchemaFourSchedule() throws {
        let inputs = try inputs()
        let selected = try inputs.search.validated(
            sensitivityArtifact: inputs.sensitivity,
            exactSensitivityArtifactData: inputs.sensitivityData,
            calibrationManifest: inputs.manifest,
            exactCalibrationManifestData: inputs.manifestData,
            exactCandidateEvaluationArtifactData: inputs.evaluationData,
            candidateRuntimePolicies: inputs.runtimePolicies,
            exactModelConfigData: inputs.configData,
            eosTokenID: 255,
            decodeTokenIDs: decodeTokenIDs)
        XCTAssertEqual(selected.ordinal, 0)

        let schedule = try KVTunerScheduleSearch.makeSchedule(
            searchArtifact: inputs.search,
            exactSearchArtifactData: inputs.searchData,
            sensitivityArtifact: inputs.sensitivity,
            exactSensitivityArtifactData: inputs.sensitivityData,
            calibrationManifest: inputs.manifest,
            exactCalibrationManifestData: inputs.manifestData,
            exactCandidateEvaluationArtifactData: inputs.evaluationData,
            candidateRuntimePolicies: inputs.runtimePolicies,
            eosTokenID: 255,
            decodeTokenIDs: decodeTokenIDs,
            exactModelConfigData: inputs.configData,
            expectedCheckpointManifestHash:
                inputs.manifest.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                inputs.manifest.checkpointContentSHA256)

        XCTAssertEqual(schedule.schemaVersion, 4)
        XCTAssertEqual(
            schedule.tokenizerSHA256,
            inputs.manifest.tokenizerSHA256)
        XCTAssertEqual(schedule.layers, inputs.candidates[0].layers)
        XCTAssertEqual(
            schedule.sourceSearchArtifactSHA256,
            sha256Hex(inputs.searchData))
        XCTAssertEqual(
            schedule.calibrationEntryHashes,
            inputs.manifest.calibrationEntryDigests)
        XCTAssertEqual(schedule.calibrationEntryHashes.count, 220)
        XCTAssertEqual(schedule.calibrationSourceItemDigests.count, 200)
    }

    func testSearchArtifactRejectsReplayedEvaluationAndWrongSelection() throws {
        let inputs = try inputs()
        var tampered = inputs.search
        tampered.evaluations[0] = KVTunerCandidateEvaluation(
            candidateOrdinal: 0,
            candidateSHA256: inputs.evaluations[1].candidateSHA256,
            runtimePolicySHA256:
                inputs.evaluations[1].runtimePolicySHA256,
            environmentSHA256: inputs.evaluations[1].environmentSHA256,
            correctCount: 120,
            totalCount: 200,
            outputSHA256: String(repeating: "c", count: 64))
        XCTAssertThrowsError(try tampered.validated(
            sensitivityArtifact: inputs.sensitivity,
            exactSensitivityArtifactData: inputs.sensitivityData,
            calibrationManifest: inputs.manifest,
            exactCalibrationManifestData: inputs.manifestData,
            exactCandidateEvaluationArtifactData:
                inputs.evaluationData,
            candidateRuntimePolicies: inputs.runtimePolicies,
            exactModelConfigData: inputs.configData,
            eosTokenID: 255,
            decodeTokenIDs: decodeTokenIDs)) { error in
                XCTAssertEqual(
                    error as? KVTunerSearchArtifactError,
                    .evaluationMismatch)
            }

        tampered = inputs.search
        tampered.selectedCandidateOrdinal = 1
        XCTAssertThrowsError(try tampered.validated(
            sensitivityArtifact: inputs.sensitivity,
            exactSensitivityArtifactData: inputs.sensitivityData,
            calibrationManifest: inputs.manifest,
            exactCalibrationManifestData: inputs.manifestData,
            exactCandidateEvaluationArtifactData:
                inputs.evaluationData,
            candidateRuntimePolicies: inputs.runtimePolicies,
            exactModelConfigData: inputs.configData,
            eosTokenID: 255,
            decodeTokenIDs: decodeTokenIDs)) { error in
                XCTAssertEqual(
                    error as? KVTunerSearchArtifactError,
                    .selectedCandidateMismatch)
            }
    }

    func testSearchArtifactRejectsMixedEnvironmentAndCandidatePolicySubstitution() throws {
        let inputs = try inputs()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]

        var mixedArtifact = try JSONDecoder().decode(
            KVTunerCandidateEvaluationArtifact.self,
            from: inputs.evaluationData[1])
        mixedArtifact.executionEnvironment.hardwareChip = "Apple M2 Ultra"
        let mixedArtifactData = try encoder.encode(mixedArtifact)
        let mixedSummary = try mixedArtifact.validated(
            exactArtifactData: mixedArtifactData,
            runtimePolicy: inputs.runtimePolicies[1],
            runtimeContract: inputs.runtimeContract,
            calibrationManifest: inputs.manifest,
            exactCalibrationManifestData: inputs.manifestData,
            decodeTokenIDs: decodeTokenIDs)
        var mixedSearch = inputs.search
        mixedSearch.evaluations[1] = mixedSummary
        var mixedEvaluationData = inputs.evaluationData
        mixedEvaluationData[1] = mixedArtifactData

        XCTAssertThrowsError(try mixedSearch.validated(
            sensitivityArtifact: inputs.sensitivity,
            exactSensitivityArtifactData: inputs.sensitivityData,
            calibrationManifest: inputs.manifest,
            exactCalibrationManifestData: inputs.manifestData,
            exactCandidateEvaluationArtifactData: mixedEvaluationData,
            candidateRuntimePolicies: inputs.runtimePolicies,
            exactModelConfigData: inputs.configData,
            eosTokenID: 255,
            decodeTokenIDs: decodeTokenIDs
        )) { error in
            XCTAssertEqual(
                error as? KVTunerSearchArtifactError,
                .evaluationMismatch)
        }

        XCTAssertThrowsError(try inputs.search.validated(
            sensitivityArtifact: inputs.sensitivity,
            exactSensitivityArtifactData: inputs.sensitivityData,
            calibrationManifest: inputs.manifest,
            exactCalibrationManifestData: inputs.manifestData,
            exactCandidateEvaluationArtifactData: inputs.evaluationData,
            candidateRuntimePolicies: Array(
                inputs.runtimePolicies.reversed()),
            exactModelConfigData: inputs.configData,
            eosTokenID: 255,
            decodeTokenIDs: decodeTokenIDs
        )) { error in
            XCTAssertEqual(
                error as? KVTunerSearchArtifactError,
                .evaluationMismatch)
        }
    }

    func testSearchArtifactRejectsProtocolAndExactSourceSubstitution() throws {
        let inputs = try inputs()
        var tampered = inputs.search
        tampered.evaluationProtocol = KVTunerSearchEvaluationProtocol(
            id: "lm-eval-gsm8k-v3-first200-four-shot-v1",
            promptCount: 200,
            fewShotCount: 4,
            promptMode: "raw-completion-no-chat-template-v1",
            maxGeneratedTokens: 128,
            stopSequences: ["Question:", "</s>", "<|im_end|>"],
            doSample: false,
            temperature: 0,
            scoringFilterID: "exact-match-flexible-extract-v3")
        XCTAssertThrowsError(try tampered.validated(
            sensitivityArtifact: inputs.sensitivity,
            exactSensitivityArtifactData: inputs.sensitivityData,
            calibrationManifest: inputs.manifest,
            exactCalibrationManifestData: inputs.manifestData,
            exactCandidateEvaluationArtifactData:
                inputs.evaluationData,
            candidateRuntimePolicies: inputs.runtimePolicies,
            exactModelConfigData: inputs.configData,
            eosTokenID: 255,
            decodeTokenIDs: decodeTokenIDs)) { error in
                XCTAssertEqual(
                    error as? KVTunerSearchArtifactError,
                    .invalidProtocol("evaluationProtocol"))
            }

        tampered = inputs.search
        tampered.sourceSensitivityArtifactSHA256 =
            String(repeating: "0", count: 64)
        XCTAssertThrowsError(try tampered.validated(
            sensitivityArtifact: inputs.sensitivity,
            exactSensitivityArtifactData: inputs.sensitivityData,
            calibrationManifest: inputs.manifest,
            exactCalibrationManifestData: inputs.manifestData,
            exactCandidateEvaluationArtifactData:
                inputs.evaluationData,
            candidateRuntimePolicies: inputs.runtimePolicies,
            exactModelConfigData: inputs.configData,
            eosTokenID: 255,
            decodeTokenIDs: decodeTokenIDs)) { error in
                XCTAssertEqual(
                    error as? KVTunerSearchArtifactError,
                    .sensitivityArtifactMismatch)
            }
    }

    func testScheduleFactoryRejectsConfigLayerOrExactSearchByteMismatch() throws {
        let inputs = try inputs()
        XCTAssertThrowsError(try KVTunerScheduleSearch
            .validatedModelLayerCount(
                exactModelConfigData: inputs.configData,
                expectedModelConfigHash: inputs.manifest.modelConfigHash,
                expectedModelConfigSHA256: String(repeating: "d", count: 64),
                expectedLayerCount: inputs.sensitivity.layerCount)) { error in
                    XCTAssertEqual(
                        error as? KVTunerScheduleSearchError,
                        .sensitivityArtifactMismatch)
                }
        let wrongLayerConfig = Data(
            #"{"model_type":"qwen3","num_hidden_layers":5}"#.utf8)
        XCTAssertThrowsError(try KVTunerScheduleSearch.makeSchedule(
            searchArtifact: inputs.search,
            exactSearchArtifactData: inputs.searchData,
            sensitivityArtifact: inputs.sensitivity,
            exactSensitivityArtifactData: inputs.sensitivityData,
            calibrationManifest: inputs.manifest,
            exactCalibrationManifestData: inputs.manifestData,
            exactCandidateEvaluationArtifactData: inputs.evaluationData,
            candidateRuntimePolicies: inputs.runtimePolicies,
            eosTokenID: 255,
            decodeTokenIDs: decodeTokenIDs,
            exactModelConfigData: wrongLayerConfig,
            expectedCheckpointManifestHash:
                inputs.manifest.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                inputs.manifest.checkpointContentSHA256))

        XCTAssertThrowsError(try KVTunerScheduleSearch.makeSchedule(
            searchArtifact: inputs.search,
            exactSearchArtifactData: Data("{}".utf8),
            sensitivityArtifact: inputs.sensitivity,
            exactSensitivityArtifactData: inputs.sensitivityData,
            calibrationManifest: inputs.manifest,
            exactCalibrationManifestData: inputs.manifestData,
            exactCandidateEvaluationArtifactData: inputs.evaluationData,
            candidateRuntimePolicies: inputs.runtimePolicies,
            eosTokenID: 255,
            decodeTokenIDs: decodeTokenIDs,
            exactModelConfigData: inputs.configData,
            expectedCheckpointManifestHash:
                inputs.manifest.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                inputs.manifest.checkpointContentSHA256)) { error in
                XCTAssertEqual(
                    error as? KVTunerScheduleSearchError,
                    .searchArtifactMismatch)
            }
    }

    func testQualifiedRuntimeRebuildsCompleteChainAndRejectsScheduleOrTokenizerSubstitution() throws {
        let inputs = try inputs()
        let checkpointContentSHA256 = inputs.manifest.checkpointContentSHA256
        let schedule = try KVTunerScheduleSearch.makeSchedule(
            searchArtifact: inputs.search,
            exactSearchArtifactData: inputs.searchData,
            sensitivityArtifact: inputs.sensitivity,
            exactSensitivityArtifactData: inputs.sensitivityData,
            calibrationManifest: inputs.manifest,
            exactCalibrationManifestData: inputs.manifestData,
            exactCandidateEvaluationArtifactData: inputs.evaluationData,
            candidateRuntimePolicies: inputs.runtimePolicies,
            eosTokenID: 255,
            decodeTokenIDs: decodeTokenIDs,
            exactModelConfigData: inputs.configData,
            expectedCheckpointManifestHash:
                inputs.manifest.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                checkpointContentSHA256)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let scheduleData = try encoder.encode(schedule)
        let bundle = try KVTunerQualificationBundle.makeAuthenticated(
            exactScheduleData: scheduleData,
            exactCalibrationManifestData: inputs.manifestData,
            exactSensitivityArtifactData: inputs.sensitivityData,
            exactSearchArtifactData: inputs.searchData,
            exactCandidateEvaluationArtifactData: inputs.evaluationData,
            exactModelConfigData: inputs.configData,
            eosTokenID: 255,
            expectedCheckpointManifestHash:
                inputs.manifest.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                checkpointContentSHA256,
            tokenizePrompt:
                KVTunerTestFixtures.tokenizer(for: inputs.manifest),
            decodeTokenIDs: decodeTokenIDs)
        let bundleData = try KVTunerArtifactCodec.encode(bundle)
        XCTAssertEqual(bundle.schemaVersion, 2)
        XCTAssertEqual(bundle.modelConfigSHA256, sha256Hex(inputs.configData))
        XCTAssertEqual(
            bundle.checkpointContentSHA256,
            checkpointContentSHA256)
        let evaluationCorpus = try KVTunerEvaluationCorpusIdentity(
            id: "task-coherence-v2",
            aggregateDigest: "0123456789abcdef",
            canonicalEntryDigests: ["fedcba9876543210"],
            canonicalSourceItemDigests: [
                sha256Hex(Data("task-coherence-v2-source".utf8))
            ])
        let liveTokenIDs = Dictionary(uniqueKeysWithValues:
            (inputs.manifest.sensitivityPrompts
                + inputs.manifest.searchPrompts).map {
                    (String(decoding: $0.promptUTF8, as: UTF8.self),
                     $0.tokenIDs)
                })

        let selection = try KVTunerRuntimeSelection.loadQualified(
            artifactData: bundleData,
            exactModelConfigData: inputs.configData,
            expectedMatrixID: schedule.matrixID,
            expectedCellID: schedule.cellID,
            expectedCheckpointManifestHash:
                inputs.manifest.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                checkpointContentSHA256,
            expectedTokenizerSHA256: inputs.manifest.tokenizerSHA256,
            expectedEOSTokenID: 255,
            tokenizePrompt: { try XCTUnwrap(liveTokenIDs[$0]) },
            decodeTokenIDs: decodeTokenIDs,
            evaluationCorpora: [evaluationCorpus])
        XCTAssertEqual(selection.layers.map(\.keyBits), schedule.layers.map(\.keyBits))
        XCTAssertEqual(selection.tokenizerSHA256, inputs.manifest.tokenizerSHA256)
        XCTAssertEqual(selection.modelConfigSHA256, sha256Hex(inputs.configData))
        XCTAssertEqual(
            selection.checkpointContentSHA256,
            checkpointContentSHA256)
        XCTAssertEqual(selection.artifactSHA256, sha256Hex(scheduleData))
        XCTAssertEqual(
            selection.qualificationBundleSHA256,
            sha256Hex(bundleData))

        var substitutedSchedule = schedule
        substitutedSchedule.layers.swapAt(0, 1)
        var substitutedBundle = bundle
        substitutedBundle.scheduleData = try encoder.encode(substitutedSchedule)
        XCTAssertThrowsError(try KVTunerRuntimeSelection.loadQualified(
            artifactData: try encoder.encode(substitutedBundle),
            exactModelConfigData: inputs.configData,
            expectedMatrixID: schedule.matrixID,
            expectedCellID: schedule.cellID,
            expectedCheckpointManifestHash:
                inputs.manifest.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                checkpointContentSHA256,
            expectedTokenizerSHA256: inputs.manifest.tokenizerSHA256,
            expectedEOSTokenID: 255,
            tokenizePrompt: { try XCTUnwrap(liveTokenIDs[$0]) },
            decodeTokenIDs: decodeTokenIDs,
            evaluationCorpora: [evaluationCorpus]))

        XCTAssertThrowsError(try KVTunerRuntimeSelection.loadQualified(
            artifactData: bundleData,
            exactModelConfigData: inputs.configData,
            expectedMatrixID: schedule.matrixID,
            expectedCellID: schedule.cellID,
            expectedCheckpointManifestHash:
                inputs.manifest.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                checkpointContentSHA256,
            expectedTokenizerSHA256: String(repeating: "f", count: 64),
            expectedEOSTokenID: 255,
            tokenizePrompt: { try XCTUnwrap(liveTokenIDs[$0]) },
            decodeTokenIDs: decodeTokenIDs,
            evaluationCorpora: [evaluationCorpus]))

        XCTAssertThrowsError(try KVTunerRuntimeSelection.loadQualified(
            artifactData: bundleData,
            exactModelConfigData: inputs.configData,
            expectedMatrixID: schedule.matrixID,
            expectedCellID: schedule.cellID,
            expectedCheckpointManifestHash:
                inputs.manifest.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                String(repeating: "f", count: 64),
            expectedTokenizerSHA256: inputs.manifest.tokenizerSHA256,
            expectedEOSTokenID: 255,
            tokenizePrompt: { try XCTUnwrap(liveTokenIDs[$0]) },
            decodeTokenIDs: decodeTokenIDs,
            evaluationCorpora: [evaluationCorpus]
        )) { error in
            XCTAssertEqual(
                error as? KVTunerRuntimeSelectionError,
                .qualificationArtifactMismatch(
                    "checkpoint-content-sha256"))
        }

        XCTAssertThrowsError(try
            KVTunerQualificationBundle.makeAuthenticated(
                exactScheduleData: scheduleData,
                exactCalibrationManifestData: inputs.manifestData,
                exactSensitivityArtifactData: inputs.sensitivityData,
                exactSearchArtifactData: inputs.searchData,
                exactCandidateEvaluationArtifactData:
                    inputs.evaluationData,
                exactModelConfigData: inputs.configData,
                eosTokenID: 255,
                expectedCheckpointManifestHash:
                    inputs.manifest.checkpointManifestHash,
                expectedCheckpointContentSHA256:
                    String(repeating: "f", count: 64),
                tokenizePrompt:
                    KVTunerTestFixtures.tokenizer(for: inputs.manifest),
                decodeTokenIDs: decodeTokenIDs)
        ) { error in
            XCTAssertEqual(
                error as? KVTunerCandidateRuntimePolicyError,
                .checkpointIdentityMismatch)
        }

        var wrongLiveTokenIDs = liveTokenIDs
        let firstPrompt = String(
            decoding: inputs.manifest.searchPrompts[0].promptUTF8,
            as: UTF8.self)
        wrongLiveTokenIDs[firstPrompt]?[0] += 1
        XCTAssertThrowsError(try KVTunerRuntimeSelection.loadQualified(
            artifactData: bundleData,
            exactModelConfigData: inputs.configData,
            expectedMatrixID: schedule.matrixID,
            expectedCellID: schedule.cellID,
            expectedCheckpointManifestHash:
                inputs.manifest.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                checkpointContentSHA256,
            expectedTokenizerSHA256: inputs.manifest.tokenizerSHA256,
            expectedEOSTokenID: 255,
            tokenizePrompt: { try XCTUnwrap(wrongLiveTokenIDs[$0]) },
            decodeTokenIDs: decodeTokenIDs,
            evaluationCorpora: [evaluationCorpus])) { error in
                XCTAssertEqual(
                    error as? KVTunerRuntimeSelectionError,
                    .promptTokenizationMismatch(
                        phase: "search", position: 0))
            }
    }

    func testQualifiedRuntimeClassifiesUnsupportedBundleSchemaBeforeCurrentFields() throws {
        let inputs = try inputs()
        let legacy = Data(#"{"schemaVersion":0}"#.utf8)
        let evaluationCorpus = try KVTunerEvaluationCorpusIdentity(
            id: "task-coherence-v2",
            aggregateDigest: "0123456789abcdef",
            canonicalEntryDigests: ["fedcba9876543210"],
            canonicalSourceItemDigests: [
                sha256Hex(Data("task-coherence-v2-source".utf8))
            ])

        XCTAssertThrowsError(try KVTunerRuntimeSelection.loadQualified(
            artifactData: legacy,
            exactModelConfigData: inputs.configData,
            expectedMatrixID: "kvarn-qwen3-32b-v1",
            expectedCellID: "kvtuner-g64-b4.5",
            expectedCheckpointManifestHash:
                inputs.manifest.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                String(repeating: "e", count: 64),
            expectedTokenizerSHA256: inputs.manifest.tokenizerSHA256,
            expectedEOSTokenID: 255,
            tokenizePrompt: { _ in [] },
            decodeTokenIDs: decodeTokenIDs,
            evaluationCorpora: [evaluationCorpus])) { error in
                XCTAssertEqual(
                    error as? KVTunerRuntimeSelectionError,
                    .unsupportedQualificationBundleSchema(0))
            }
    }
}
