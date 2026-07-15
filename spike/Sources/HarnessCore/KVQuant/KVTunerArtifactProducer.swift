import Foundation

/// One canonical encoding boundary for qualification artifacts. Exact-byte SHA-256 links in the
/// chain are only reproducible when every producer uses the same deterministic JSON form.
public enum KVTunerArtifactCodec: Sendable {
    public static func encode<Value: Encodable>(
        _ value: Value
    ) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(value)
    }
}

/// Scalar-only row returned by the actor-confined candidate runner before durable prompt and text
/// identities are attached. The producer rejects reordering or tokenization substitution rather
/// than trusting orchestration to pair a result with a manifest row.
public struct KVTunerCandidateGeneratedRow: Equatable, Sendable {
    public let promptOrdinal: Int
    public let promptTokenIDsSHA256: String
    public let generatedTokenIDs: [Int]
    public let finishReason: KVTunerCandidateFinishReason
    public let runtimeReceipt: KVTunerCandidateRuntimeReceipt

    public init(
        promptOrdinal: Int,
        promptTokenIDsSHA256: String,
        generatedTokenIDs: [Int],
        finishReason: KVTunerCandidateFinishReason,
        runtimeReceipt: KVTunerCandidateRuntimeReceipt
    ) {
        self.promptOrdinal = promptOrdinal
        self.promptTokenIDsSHA256 = promptTokenIDsSHA256
        self.generatedTokenIDs = generatedTokenIDs
        self.finishReason = finishReason
        self.runtimeReceipt = runtimeReceipt
    }
}

extension KVTunerCalibrationManifest {
    private struct PinnedPromptFixture: Decodable {
        let schemaVersion: Int
        let fewShotSeed: UInt64
        let sensitivity: [String]
        let search: [String]
    }

    /// Constructs the model-specific manifest from the checked, source-derived prompt fixture and
    /// the live tokenizer. All protocol/source constants remain independently enforced by
    /// `validated()`; this factory removes hand-authored JSON from the qualification path.
    public static func makePinnedGSM8K(
        promptFixtureData: Data,
        normalizedTargetsData: Data,
        exactModelConfigData: Data,
        checkpointManifestHash: String,
        tokenizerSHA256: String,
        tokenizePrompt: (String) throws -> [Int]
    ) throws -> KVTunerCalibrationManifest {
        let fixture = try JSONDecoder().decode(
            PinnedPromptFixture.self, from: promptFixtureData)
        let normalizedTargets = try JSONDecoder().decode(
            [String].self, from: normalizedTargetsData)
        guard fixture.schemaVersion == 1 else {
            throw KVTunerCalibrationManifestError.unsupportedSchema(
                fixture.schemaVersion)
        }
        guard fixture.fewShotSeed == KVTunerScheduleSearch.requiredFewShotSeed
        else {
            throw KVTunerCalibrationManifestError.invalidProtocol(
                "fewShotSeed")
        }

        func identities(
            _ prompts: [String]
        ) throws -> [KVTunerCalibrationPromptIdentity] {
            try prompts.enumerated().map { ordinal, prompt in
                let promptData = Data(prompt.utf8)
                let tokenIDs = try tokenizePrompt(prompt)
                return KVTunerCalibrationPromptIdentity(
                    ordinal: ordinal,
                    testIndex: ordinal,
                    promptDigest: KVTunerPromptDigest.exactText(prompt),
                    promptSHA256: sha256Hex(promptData),
                    promptUTF8: promptData,
                    tokenIDs: tokenIDs,
                    tokenIDsSHA256: taskTokenIDsSHA256(tokenIDs))
            }
        }

        return try KVTunerCalibrationManifest(
            schemaVersion: 1,
            protocolID: "gsm8k-kvtuner-qwen3-adaptation-v1",
            corpusID: "gsm8k-kvtuner-calibration-v1",
            modelConfigHash: fnv1a64(exactModelConfigData),
            modelConfigSHA256: sha256Hex(exactModelConfigData),
            checkpointManifestHash: checkpointManifestHash,
            tokenizerSHA256: tokenizerSHA256,
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
            lmEvalSourceRepository: "cmd2001/lm-evaluation-harness-X",
            paperVersion: "arxiv-2502.04420v5",
            promptExpansionID: "lm-eval-gsm8k-question-answer-v1",
            promptListEncodingID: "utf8-json-compact-array-v1",
            tokenizationProtocolID:
                "mlx-tokenizer-raw-add-special-tokens-true-v1",
            fewShotSeed: KVTunerScheduleSearch.requiredFewShotSeed,
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
            sensitivityPrompts: identities(fixture.sensitivity),
            searchPrompts: identities(fixture.search))
            .validated()
    }
}

extension KVTunerSensitivityArtifact {
    /// Wraps actor-produced sensitivity samples in the fixed KVTuner-v5 protocol and binds them
    /// to the exact calibration-manifest bytes before they can be encoded as evidence.
    public static func makeAuthenticated(
        matrixID: String,
        groupSize: Int,
        layerCount: Int,
        samples: [KVTunerSensitivitySample],
        captureEnvironment: KVTunerSensitivityCaptureEnvironment,
        calibrationManifest: KVTunerCalibrationManifest,
        exactCalibrationManifestData: Data
    ) throws -> KVTunerSensitivityArtifact {
        let manifestSHA256 = sha256Hex(exactCalibrationManifestData)
        return try KVTunerSensitivityArtifact(
            schemaVersion: 2,
            matrixID: matrixID,
            modelConfigHash: calibrationManifest.modelConfigHash,
            modelConfigSHA256: calibrationManifest.modelConfigSHA256,
            checkpointManifestHash:
                calibrationManifest.checkpointManifestHash,
            tokenizerSHA256: calibrationManifest.tokenizerSHA256,
            calibrationCorpusID: calibrationManifest.corpusID,
            calibrationCorpusHash: manifestSHA256,
            promptManifestSHA256: manifestSHA256,
            promptDigests:
                calibrationManifest.sensitivityPrompts.map(\.promptDigest),
            quantizerID: "mlx-affine-asymmetric-v1",
            captureMode: "single-prefill-no-error-propagation-v1",
            groupSize: groupSize,
            layerCount: layerCount,
            precisionPairs: canonicalPrecisionPairs,
            metricProtocolID:
                "kvtuner-v5-elementwise-mean-absolute-v1",
            metricAccumulationDType: "float32",
            denominatorEpsilon: 1e-8,
            aggregationID: "ordered-incremental-mean-v1",
            dbscanEpsilon: 0.05,
            dbscanMinSamples: 2,
            captureEnvironment: captureEnvironment,
            samples: samples)
            .validated(
                calibrationManifest: calibrationManifest,
                exactCalibrationManifestData:
                    exactCalibrationManifestData)
    }
}

extension KVTunerCandidateEvaluationArtifact {
    /// Converts actor-produced scalar rows into the exact schema-2 evidence artifact. The result
    /// is encoded and replay-validated before it is returned, so callers cannot persist a row set
    /// whose output text, stop handling, score, or cache receipt is merely self-consistent.
    public static func makeAuthenticated(
        runtimePolicy: KVTunerCandidateRuntimePolicy,
        runtimeContract: KVTunerCandidateRuntimeContract,
        executionEnvironment: KVTunerCandidateExecutionEnvironment,
        calibrationManifest: KVTunerCalibrationManifest,
        exactCalibrationManifestData: Data,
        generatedRows: [KVTunerCandidateGeneratedRow],
        decodeTokenIDs: ([Int]) throws -> String
    ) throws -> KVTunerCandidateEvaluationArtifact {
        guard generatedRows.count
                == KVTunerScheduleSearch.requiredSearchPromptCount,
            generatedRows.count == calibrationManifest.searchPrompts.count
        else {
            throw KVTunerCandidateEvaluationArtifactError.invalidRow(
                generatedRows.count)
        }

        let rows = try generatedRows.enumerated().map {
            ordinal, generated -> KVTunerCandidateOutputRow in
            let prompt = calibrationManifest.searchPrompts[ordinal]
            guard generated.promptOrdinal == ordinal,
                generated.promptTokenIDsSHA256 == prompt.tokenIDsSHA256
            else {
                throw KVTunerCandidateEvaluationArtifactError.invalidRow(
                    ordinal)
            }
            let rawOutput: String
            do {
                rawOutput = try decodeTokenIDs(generated.generatedTokenIDs)
            } catch {
                throw KVTunerCandidateEvaluationArtifactError
                    .outputDecodingFailed(ordinal)
            }
            let stoppedOutput = canonicalStoppedOutput(
                rawOutput,
                stopSequences:
                    KVTunerSearchEvaluationProtocol.canonical.stopSequences)
            return KVTunerCandidateOutputRow(
                ordinal: ordinal,
                promptSHA256: prompt.promptSHA256,
                promptTokenIDsSHA256:
                    generated.promptTokenIDsSHA256,
                generatedTokenIDs: generated.generatedTokenIDs,
                rawDecodedUTF8: Data(rawOutput.utf8),
                outputUTF8: Data(stoppedOutput.utf8),
                finishReason: generated.finishReason,
                runtimeReceipt: generated.runtimeReceipt)
        }
        let artifact = KVTunerCandidateEvaluationArtifact(
            schemaVersion: 2,
            evaluationProtocol: .canonical,
            promptManifestSHA256:
                sha256Hex(exactCalibrationManifestData),
            candidateOrdinal: runtimePolicy.candidateOrdinal,
            candidateSHA256: runtimePolicy.candidateSHA256,
            runtimePolicySHA256: runtimePolicy.runtimePolicySHA256,
            executionEnvironment: executionEnvironment,
            rows: rows)
        let exactArtifactData = try KVTunerArtifactCodec.encode(artifact)
        _ = try artifact.validated(
            exactArtifactData: exactArtifactData,
            runtimePolicy: runtimePolicy,
            runtimeContract: runtimeContract,
            calibrationManifest: calibrationManifest,
            exactCalibrationManifestData:
                exactCalibrationManifestData,
            decodeTokenIDs: decodeTokenIDs)
        return artifact
    }
}

extension KVTunerSearchArtifact {
    /// Reconstructs and scores the complete exhaustive candidate set from exact artifacts. The
    /// selected ordinal is derived here; a caller cannot nominate a preferred schedule or submit
    /// a prefix of the candidate list as a complete search.
    public static func makeAuthenticated(
        targetPairBitTotal: Int,
        maxCandidates: Int,
        sensitivityArtifact: KVTunerSensitivityArtifact,
        exactSensitivityArtifactData: Data,
        calibrationManifest: KVTunerCalibrationManifest,
        exactCalibrationManifestData: Data,
        exactCandidateEvaluationArtifactData: [Data],
        exactModelConfigData: Data,
        eosTokenID: Int,
        tokenizePrompt: (String) throws -> [Int],
        decodeTokenIDs: ([Int]) throws -> String
    ) throws -> KVTunerSearchArtifact {
        _ = try sensitivityArtifact.validated(
            calibrationManifest: calibrationManifest,
            exactCalibrationManifestData:
                exactCalibrationManifestData)
        let candidates = try KVTunerScheduleSearch.enumerate(
            analysis: sensitivityArtifact.analyzed(),
            targetPairBitTotal: targetPairBitTotal,
            maxCandidates: maxCandidates)
        guard exactCandidateEvaluationArtifactData.count == candidates.count
        else {
            throw KVTunerScheduleSearchError.incompleteEvaluations(
                expected: candidates.count,
                actual: exactCandidateEvaluationArtifactData.count)
        }
        let runtimePolicies = try candidates.indices.map {
            try KVTunerCandidateRuntimePolicy.load(
                exactCalibrationManifestData:
                    exactCalibrationManifestData,
                exactSensitivityArtifactData:
                    exactSensitivityArtifactData,
                exactModelConfigData: exactModelConfigData,
                expectedCheckpointManifestHash:
                    calibrationManifest.checkpointManifestHash,
                expectedTokenizerSHA256:
                    calibrationManifest.tokenizerSHA256,
                targetPairBitTotal: targetPairBitTotal,
                maxCandidates: maxCandidates,
                candidateOrdinal: $0,
                tokenizePrompt: tokenizePrompt)
        }
        let evaluations = try candidates.indices.map { ordinal in
            let data = exactCandidateEvaluationArtifactData[ordinal]
            let artifact = try JSONDecoder().decode(
                KVTunerCandidateEvaluationArtifact.self, from: data)
            let runtimeContract = try KVTunerCandidateRuntimeContract.load(
                exactModelConfigData: exactModelConfigData,
                runtimePolicy: runtimePolicies[ordinal],
                eosTokenID: eosTokenID)
            return try artifact.validated(
                exactArtifactData: data,
                runtimePolicy: runtimePolicies[ordinal],
                runtimeContract: runtimeContract,
                calibrationManifest: calibrationManifest,
                exactCalibrationManifestData:
                    exactCalibrationManifestData,
                decodeTokenIDs: decodeTokenIDs)
        }
        let selected = try KVTunerScheduleSearch.select(
            candidates: candidates,
            evaluations: evaluations,
            requiredRuntimePolicySHA256ByCandidate:
                runtimePolicies.map(\.runtimePolicySHA256))
        let artifact = KVTunerSearchArtifact(
            schemaVersion: 2,
            searchMode: "exhaustive-grouped-v1",
            evaluationProtocol: .canonical,
            sourceSensitivityArtifactSHA256:
                sha256Hex(exactSensitivityArtifactData),
            promptManifestSHA256:
                sha256Hex(exactCalibrationManifestData),
            modelConfigHash: calibrationManifest.modelConfigHash,
            modelConfigSHA256: calibrationManifest.modelConfigSHA256,
            checkpointManifestHash:
                calibrationManifest.checkpointManifestHash,
            tokenizerSHA256: calibrationManifest.tokenizerSHA256,
            groupSize: sensitivityArtifact.groupSize,
            targetPairBitTotal: targetPairBitTotal,
            seed: KVTunerScheduleSearch.requiredFewShotSeed,
            candidateListSHA256:
                try KVTunerScheduleSearch.candidateListSHA256(candidates),
            candidates: candidates,
            evaluations: evaluations,
            selectedCandidateOrdinal: selected.ordinal)
        _ = try artifact.validated(
            sensitivityArtifact: sensitivityArtifact,
            exactSensitivityArtifactData:
                exactSensitivityArtifactData,
            calibrationManifest: calibrationManifest,
            exactCalibrationManifestData:
                exactCalibrationManifestData,
            exactCandidateEvaluationArtifactData:
                exactCandidateEvaluationArtifactData,
            candidateRuntimePolicies: runtimePolicies,
            exactModelConfigData: exactModelConfigData,
            eosTokenID: eosTokenID,
            decodeTokenIDs: decodeTokenIDs)
        return artifact
    }
}

extension KVTunerQualificationBundle {
    /// Builds the portable bundle only after independently re-deriving the supplied schedule from
    /// every exact upstream artifact and the live tokenizer/model contract. The schedule bytes are
    /// evidence output, never an authorization input.
    public static func makeAuthenticated(
        exactScheduleData: Data,
        exactCalibrationManifestData: Data,
        exactSensitivityArtifactData: Data,
        exactSearchArtifactData: Data,
        exactCandidateEvaluationArtifactData: [Data],
        exactModelConfigData: Data,
        eosTokenID: Int,
        expectedCheckpointManifestHash: String,
        tokenizePrompt: (String) throws -> [Int],
        decodeTokenIDs: ([Int]) throws -> String
    ) throws -> KVTunerQualificationBundle {
        let manifest = try JSONDecoder().decode(
            KVTunerCalibrationManifest.self,
            from: exactCalibrationManifestData)
        let sensitivity = try JSONDecoder().decode(
            KVTunerSensitivityArtifact.self,
            from: exactSensitivityArtifactData)
        let search = try JSONDecoder().decode(
            KVTunerSearchArtifact.self,
            from: exactSearchArtifactData)
        let suppliedSchedule = try JSONDecoder().decode(
            KVTunerSchedule.self, from: exactScheduleData)
        let runtimePolicies = try search.candidates.indices.map {
            try KVTunerCandidateRuntimePolicy.load(
                exactCalibrationManifestData:
                    exactCalibrationManifestData,
                exactSensitivityArtifactData:
                    exactSensitivityArtifactData,
                exactModelConfigData: exactModelConfigData,
                expectedCheckpointManifestHash:
                    expectedCheckpointManifestHash,
                expectedTokenizerSHA256: manifest.tokenizerSHA256,
                targetPairBitTotal: search.targetPairBitTotal,
                maxCandidates: search.candidates.count,
                candidateOrdinal: $0,
                tokenizePrompt: tokenizePrompt)
        }
        let derivedSchedule = try KVTunerScheduleSearch.makeSchedule(
            searchArtifact: search,
            exactSearchArtifactData: exactSearchArtifactData,
            sensitivityArtifact: sensitivity,
            exactSensitivityArtifactData:
                exactSensitivityArtifactData,
            calibrationManifest: manifest,
            exactCalibrationManifestData:
                exactCalibrationManifestData,
            exactCandidateEvaluationArtifactData:
                exactCandidateEvaluationArtifactData,
            candidateRuntimePolicies: runtimePolicies,
            eosTokenID: eosTokenID,
            decodeTokenIDs: decodeTokenIDs,
            exactModelConfigData: exactModelConfigData,
            expectedCheckpointManifestHash:
                expectedCheckpointManifestHash)
        guard suppliedSchedule == derivedSchedule else {
            throw KVTunerScheduleSearchError.searchArtifactMismatch
        }
        return KVTunerQualificationBundle(
            schemaVersion: 1,
            scheduleData: exactScheduleData,
            calibrationManifestData: exactCalibrationManifestData,
            sensitivityArtifactData: exactSensitivityArtifactData,
            searchArtifactData: exactSearchArtifactData,
            candidateEvaluationArtifactData:
                exactCandidateEvaluationArtifactData)
    }
}
