import Foundation
import XCTest

@testable import HarnessCore

enum KVTunerTestFixtures {
    struct CandidateRuntimeInputs {
        let configData: Data
        let manifest: KVTunerCalibrationManifest
        let manifestData: Data
        let sensitivity: KVTunerSensitivityArtifact
        let sensitivityData: Data
        let candidates: [KVTunerScheduleCandidate]
    }

    private struct PromptHashFixture: Decodable {
        struct Entry: Decodable {
            let fnv: String
            let sha256: String
        }
        let sensitivity: [Entry]
        let search: [Entry]
    }

    private struct PromptTextFixture: Decodable {
        let schemaVersion: Int
        let fewShotSeed: UInt64
        let sensitivity: [String]
        let search: [String]
    }

    static func sensitivityCaptureEnvironment()
        -> KVTunerSensitivityCaptureEnvironment
    {
        KVTunerSensitivityCaptureEnvironment(
            harnessGitSHA: String(repeating: "a", count: 40),
            buildConfiguration: "Release",
            mlxSwiftVersion: "0.31.6",
            mlxSwiftLMRevision:
                "702e5a0eaf990e1f6d3db2b6e7d8872858a44055",
            hardwareChip: "Apple M3 Ultra",
            hardwareRAMBytes: 274_877_906_944,
            hardwareOS: "macOS 26.5.2",
            memoryCacheLimitBytes: 8 << 30)
    }

    static func calibrationManifest() throws -> KVTunerCalibrationManifest {
        let url = try XCTUnwrap(Bundle.module.url(
            forResource: "kvtuner-gsm8k-prompt-hashes",
            withExtension: "json"))
        let fixture = try JSONDecoder().decode(
            PromptHashFixture.self, from: Data(contentsOf: url))
        let targetsURL = try XCTUnwrap(Bundle.module.url(
            forResource: "kvtuner-gsm8k-normalized-targets",
            withExtension: "json"))
        let normalizedTargets = try JSONDecoder().decode(
            [String].self, from: Data(contentsOf: targetsURL))
        let promptsURL = try XCTUnwrap(Bundle.module.url(
            forResource: "kvtuner-gsm8k-prompts",
            withExtension: "json"))
        let promptText = try JSONDecoder().decode(
            PromptTextFixture.self, from: Data(contentsOf: promptsURL))
        XCTAssertEqual(promptText.schemaVersion, 1)
        XCTAssertEqual(promptText.fewShotSeed, 1234)
        func identities(
            _ entries: [PromptHashFixture.Entry],
            prompts: [String],
            phase: String
        ) -> [KVTunerCalibrationPromptIdentity] {
            XCTAssertEqual(entries.count, prompts.count)
            return entries.enumerated().map { index, entry in
                let tokenIDs = [
                    phase == "sensitivity" ? 1 : 2,
                    index + 10,
                    index + 1000,
                ]
                return KVTunerCalibrationPromptIdentity(
                    ordinal: index,
                    testIndex: index,
                    promptDigest: entry.fnv,
                    promptSHA256: entry.sha256,
                    promptUTF8: Data(prompts[index].utf8),
                    tokenIDs: tokenIDs,
                    tokenIDsSHA256: taskTokenIDsSHA256(tokenIDs))
            }
        }
        return KVTunerCalibrationManifest(
            schemaVersion: 2,
            protocolID: "gsm8k-kvtuner-qwen3-adaptation-v1",
            corpusID: "gsm8k-kvtuner-calibration-v1",
            modelConfigHash: "0123456789abcdef",
            modelConfigSHA256: String(repeating: "c", count: 64),
            checkpointManifestHash: "fedcba9876543210",
            checkpointContentSHA256: String(repeating: "d", count: 64),
            tokenizerSHA256: String(repeating: "a", count: 64),
            datasetSourceRepository: "openai/grade-school-math",
            datasetSourceCommit:
                "b0bb162abedc65e1fdd8e93ed090fd7598ee68bc",
            trainDataSHA256:
                "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
            testDataSHA256:
                "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
            kvtunerSourceCommit:
                "96dd05eb2fe350c72c1a3dfdca04e878506f7c17",
            lmEvalSourceCommit:
                "6ec76a6e28056f5b27715b8a233c13018a6967cc",
            lmEvalSourceRepository:
                "cmd2001/lm-evaluation-harness-X",
            paperVersion: "arxiv-2502.04420v5",
            promptExpansionID: "lm-eval-gsm8k-question-answer-v1",
            promptListEncodingID: "utf8-json-compact-array-v1",
            tokenizationProtocolID:
                "mlx-tokenizer-raw-add-special-tokens-true-v1",
            fewShotSeed: 1234,
            sensitivityPromptListSHA256:
                "18d51be3aa1ac8e6ed7028a96c8c05efed1aa88588a2635e517d27a3e4e01730",
            fewShotIndexTableSHA256:
                "69dc558179253dcee3d87ab0a95b06911737d483af5e1000038044fb0265728e",
            searchPromptListSHA256:
                "5e79ef00e8d8d602ce0b24a9ce49e2522fd5c775ae9e00f1e2c57f84931fb16e",
            sensitivityPromptSHA256ListSHA256:
                "aa4325d7b4d3f1f243b4ab10b4fecc585dcb437b9ee77cd720e3945394b400fa",
            sensitivityPromptDigestListSHA256:
                "5bb0671a74612092b8c9e417392e7ec133a4a06f6bba6944cfcab15a6a202c3c",
            searchPromptSHA256ListSHA256:
                "52504fe6a2d9342e725d80b7f2f76ba500522085c126e656cb30add18bb41290",
            searchPromptDigestListSHA256:
                "8e1bfc50a66e97a8dd4149c6afd109387601d166d341d1a83da06f022443716d",
            searchNormalizedTargetListSHA256:
                "4f17cc1a11082c7cbd8e2002e69a5b4a30056b1155ce49731b66f8ea4553903f",
            searchNormalizedTargets: normalizedTargets,
            sensitivityPrompts: identities(
                fixture.sensitivity,
                prompts: promptText.sensitivity,
                phase: "sensitivity"),
            searchPrompts: identities(
                fixture.search,
                prompts: promptText.search,
                phase: "search"))
    }

    static func encodedCalibrationManifest()
        throws -> (KVTunerCalibrationManifest, Data)
    {
        let manifest = try calibrationManifest().validated()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return (manifest, try encoder.encode(manifest))
    }

    static func candidateRuntimeInputs() throws -> CandidateRuntimeInputs {
        let configData = Data(
            #"{"eos_token_id":255,"head_dim":128,"hidden_size":5120,"model_type":"qwen3","num_attention_heads":64,"num_hidden_layers":4,"num_key_value_heads":8,"torch_dtype":"bfloat16"}"#.utf8)
        var manifest = try calibrationManifest()
        manifest.modelConfigHash = fnv1a64(configData)
        manifest.modelConfigSHA256 = sha256Hex(configData)
        manifest = try manifest.validated()

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let manifestData = try encoder.encode(manifest)
        let manifestSHA256 = sha256Hex(manifestData)
        let pairs = KVTunerSensitivityArtifact.canonicalPrecisionPairs
        let layerOutputs = [
            [0.01, 0.05, 0.20],
            [0.01, 0.05, 0.20],
            [0.02, 0.06, 0.10],
            [0.02, 0.06, 0.10],
        ]
        var samples: [KVTunerSensitivitySample] = []
        for promptIndex in manifest.sensitivityPrompts.indices {
            for layer in layerOutputs.indices {
                for (pairIndex, pair) in pairs.enumerated() {
                    let output = layerOutputs[layer][pairIndex]
                    samples.append(KVTunerSensitivitySample(
                        promptIndex: promptIndex,
                        layer: layer,
                        keyBits: pair.keyBits,
                        valueBits: pair.valueBits,
                        relativeKeyError: output,
                        relativeValueError: output,
                        attentionScoreError: output,
                        relativeAttentionOutputError: output))
                }
            }
        }
        let sensitivity = try KVTunerSensitivityArtifact(
            schemaVersion: 3,
            matrixID: "kvarn-qwen3-32b-v1",
            modelConfigHash: manifest.modelConfigHash,
            modelConfigSHA256: manifest.modelConfigSHA256,
            checkpointManifestHash: manifest.checkpointManifestHash,
            checkpointContentSHA256: manifest.checkpointContentSHA256,
            tokenizerSHA256: manifest.tokenizerSHA256,
            calibrationCorpusID: manifest.corpusID,
            calibrationCorpusHash: manifestSHA256,
            promptManifestSHA256: manifestSHA256,
            promptDigests: manifest.sensitivityPrompts.map(\.promptDigest),
            quantizerID: "mlx-affine-asymmetric-v1",
            captureMode: "single-prefill-no-error-propagation-v1",
            groupSize: 64,
            layerCount: layerOutputs.count,
            precisionPairs: pairs,
            metricProtocolID:
                "kvtuner-v5-elementwise-mean-absolute-v1",
            metricAccumulationDType: "float32",
            denominatorEpsilon: 1e-8,
            aggregationID: "ordered-incremental-mean-v1",
            dbscanEpsilon: 0.05,
            dbscanMinSamples: 2,
            captureEnvironment: sensitivityCaptureEnvironment(),
            samples: samples).validated(
                calibrationManifest: manifest,
                exactCalibrationManifestData: manifestData)
        let sensitivityData = try encoder.encode(sensitivity)
        let candidates = try KVTunerScheduleSearch.enumerate(
            analysis: sensitivity.analyzed(),
            targetPairBitTotal: 36,
            maxCandidates: 10)
        return CandidateRuntimeInputs(
            configData: configData,
            manifest: manifest,
            manifestData: manifestData,
            sensitivity: sensitivity,
            sensitivityData: sensitivityData,
            candidates: candidates)
    }

    static func tokenizer(
        for manifest: KVTunerCalibrationManifest
    ) -> (String) throws -> [Int] {
        let prompts = manifest.sensitivityPrompts + manifest.searchPrompts
        let tokenIDs = Dictionary(uniqueKeysWithValues: prompts.map {
            (String(decoding: $0.promptUTF8, as: UTF8.self), $0.tokenIDs)
        })
        return { prompt in
            guard let tokens = tokenIDs[prompt] else {
                throw CocoaError(.coderValueNotFound)
            }
            return tokens
        }
    }

    static func candidateRuntimePolicy(
        _ inputs: CandidateRuntimeInputs,
        candidateOrdinal: Int
    ) throws -> KVTunerCandidateRuntimePolicy {
        try KVTunerCandidateRuntimePolicy.load(
            exactCalibrationManifestData: inputs.manifestData,
            exactSensitivityArtifactData: inputs.sensitivityData,
            exactModelConfigData: inputs.configData,
            expectedCheckpointManifestHash:
                inputs.manifest.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                inputs.manifest.checkpointContentSHA256,
            expectedTokenizerSHA256: inputs.manifest.tokenizerSHA256,
            targetPairBitTotal: 36,
            maxCandidates: 10,
            candidateOrdinal: candidateOrdinal,
            tokenizePrompt: tokenizer(for: inputs.manifest))
    }
}
