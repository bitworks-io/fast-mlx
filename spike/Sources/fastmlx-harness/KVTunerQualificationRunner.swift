import Foundation
import HarnessCore
import Tokenizers

#if canImport(Darwin)
import Darwin
#endif

private func kvtunerQualificationData(at path: String) throws -> Data {
    do {
        return try Data(contentsOf: URL(fileURLWithPath: path))
    } catch {
        throw KVTunerQualificationCLIError.invalidArtifact(path)
    }
}

private func existingKVTunerQualificationData(
    at path: String
) throws -> Data? {
    guard !outputPathIsSymbolicLink(path) else {
        throw KVTunerQualificationCLIError.outputNotFresh(path)
    }
    let url = URL(fileURLWithPath: path).standardizedFileURL
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(
        atPath: url.path, isDirectory: &isDirectory)
    else { return nil }
    let attributes = try FileManager.default.attributesOfItem(
        atPath: url.path)
    let type = attributes[.type] as? FileAttributeType
    let size = (attributes[.size] as? NSNumber)?.uint64Value ?? 0
    guard !isDirectory.boolValue, type == .typeRegular else {
        throw KVTunerQualificationCLIError.outputNotFresh(path)
    }
    return size == 0 ? nil : try Data(contentsOf: url)
}

private func decodedKVTunerManifest(
    _ data: Data,
    path: String
) throws -> KVTunerCalibrationManifest {
    do {
        return try JSONDecoder().decode(
            KVTunerCalibrationManifest.self, from: data).validated()
    } catch {
        throw KVTunerQualificationCLIError.invalidArtifact(path)
    }
}

private func decodedKVTunerSensitivity(
    _ data: Data,
    path: String,
    manifest: KVTunerCalibrationManifest,
    manifestData: Data
) throws -> KVTunerSensitivityArtifact {
    do {
        return try JSONDecoder().decode(
            KVTunerSensitivityArtifact.self, from: data)
            .validated(
                calibrationManifest: manifest,
                exactCalibrationManifestData: manifestData)
    } catch {
        throw KVTunerQualificationCLIError.invalidArtifact(path)
    }
}

private func requireKVTunerSnapshot(
    _ snapshot: KVTunerCandidateRuntimeSourceSnapshot,
    matches manifest: KVTunerCalibrationManifest
) throws {
    guard fnv1a64(snapshot.exactModelConfigData)
            == manifest.modelConfigHash,
        sha256Hex(snapshot.exactModelConfigData)
            == manifest.modelConfigSHA256,
        snapshot.checkpointManifestHash
            == manifest.checkpointManifestHash,
        snapshot.checkpointContentSHA256
            == manifest.checkpointContentSHA256,
        snapshot.tokenizerSHA256 == manifest.tokenizerSHA256
    else {
        throw KVTunerQualificationCLIError.sourceIdentityMismatch
    }
}

private func validateKVTunerManifestTokenization(
    _ manifest: KVTunerCalibrationManifest,
    tokenizePrompt: (String) throws -> [Int]
) throws {
    for prompt in manifest.sensitivityPrompts + manifest.searchPrompts {
        let text = String(decoding: prompt.promptUTF8, as: UTF8.self)
        guard try tokenizePrompt(text) == prompt.tokenIDs else {
            throw KVTunerQualificationCLIError.invalidArtifact(
                "live tokenizer replay")
        }
    }
}

private func requireKVTunerReleaseBuild() throws {
    #if DEBUG
    throw KVTunerQualificationCLIError.releaseBuildRequired
    #endif
}

private func kvtunerSensitivityCaptureEnvironment(
    harnessGitSHA: String
) throws -> KVTunerSensitivityCaptureEnvironment {
    try requireKVTunerReleaseBuild()
    return KVTunerSensitivityCaptureEnvironment(
        harnessGitSHA: harnessGitSHA,
        buildConfiguration:
            KVTunerSensitivityCaptureEnvironment
                .requiredBuildConfiguration,
        mlxSwiftVersion: ProvenanceCLI.mlxSwiftVersion,
        mlxSwiftLMRevision: ProvenanceCLI.mlxSwiftLMRevision,
        hardwareChip: ProvenanceCLI.chipBrand(),
        hardwareRAMBytes: ProvenanceCLI.ramBytes(),
        hardwareOS: ProvenanceCLI.osVersion(),
        memoryCacheLimitBytes:
            KVTunerSensitivityCaptureEnvironment
                .requiredMemoryCacheLimitBytes)
}

private func writeKVTunerQualificationResult(
    _ data: Data,
    to path: String,
    label: String
) throws {
    let wrote = try writeOrValidateExactKVTunerArtifact(data, to: path)
    print(
        "\(label): \(wrote ? "wrote" : "already complete") "
            + "\(path) sha256=\(sha256Hex(data))")
}

func runKVTunerManifest(_ flags: Flags) async {
    do {
        let plan = try parseKVTunerManifestPlan(flags)
        try requireKVTunerReleaseBuild()
        _ = try ProvenanceCLI.qualificationHarnessGitSHA()
        let before = try captureKVTunerQualificationRuntimeSourceSnapshot(
            modelPath: plan.modelPath)
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: URL(fileURLWithPath: plan.modelPath))
        let after = try captureKVTunerQualificationRuntimeSourceSnapshot(
            modelPath: plan.modelPath)
        let stable = try KVTunerCandidateRuntimeSourceSnapshot
            .validateUnchanged(before: before, after: after)
        let artifact = try KVTunerCalibrationManifest.makePinnedGSM8K(
            promptFixtureData: try kvtunerQualificationData(
                at: plan.promptFixturePath),
            normalizedTargetsData: try kvtunerQualificationData(
                at: plan.normalizedTargetsPath),
            exactModelConfigData: stable.exactModelConfigData,
            checkpointManifestHash: stable.checkpointManifestHash,
            checkpointContentSHA256: stable.checkpointContentSHA256,
            tokenizerSHA256: stable.tokenizerSHA256,
            tokenizePrompt: {
                tokenizer.encode(text: $0, addSpecialTokens: true)
            })
        let data = try KVTunerArtifactCodec.encode(artifact)
        try writeKVTunerQualificationResult(
            data, to: plan.outputPath, label: "kvtuner-manifest")
    } catch {
        print("kvtuner-manifest FAILED: \(error)")
        exit(1)
    }
}

func runKVTunerSensitivity(_ flags: Flags) async {
    do {
        let plan = try parseKVTunerSensitivityPlan(flags)
        let harnessGitSHA = try ProvenanceCLI.qualificationHarnessGitSHA()
        let captureEnvironment = try kvtunerSensitivityCaptureEnvironment(
            harnessGitSHA: harnessGitSHA)
        let manifestData = try kvtunerQualificationData(
            at: plan.manifestPath)
        let manifest = try decodedKVTunerManifest(
            manifestData, path: plan.manifestPath)
        let before = try captureKVTunerQualificationRuntimeSourceSnapshot(
            modelPath: plan.modelPath)
        try requireKVTunerSnapshot(before, matches: manifest)
        let layerCount = try KVTunerModelConfigPreflight.load(
            from: before.exactModelConfigData)

        let cpuTokenizer = try await AutoTokenizer.from(
            modelFolder: URL(fileURLWithPath: plan.modelPath))
        guard cpuTokenizer.eosToken.flatMap({
            cpuTokenizer.convertTokenToId($0)
        }) != nil else {
            throw KVTunerQualificationCLIError.missingEOSToken
        }
        try validateKVTunerManifestTokenization(
            manifest,
            tokenizePrompt: {
                cpuTokenizer.encode(
                    text: $0, addSpecialTokens: true)
            })
        let afterTokenizer = try
            captureKVTunerQualificationRuntimeSourceSnapshot(
                modelPath: plan.modelPath)
        _ = try KVTunerCandidateRuntimeSourceSnapshot.validateUnchanged(
            before: before, after: afterTokenizer)

        if let existingData = try existingKVTunerQualificationData(
            at: plan.outputPath)
        {
            let existing = try decodedKVTunerSensitivity(
                existingData,
                path: plan.outputPath,
                manifest: manifest,
                manifestData: manifestData)
            guard existing.captureEnvironment == captureEnvironment,
                try KVTunerArtifactCodec.encode(existing) == existingData
            else {
                throw KVTunerQualificationCLIError.invalidArtifact(
                    plan.outputPath)
            }
            print(
                "kvtuner-sensitivity: already complete \(plan.outputPath) "
                    + "sha256=\(sha256Hex(existingData))")
            return
        }

        print(
            "# kvtuner-sensitivity loading model after manifest, source, "
                + "tokenizer, clean-SHA, and Release preflight passed")
        let loaded = try await loadSwiftDriver(
            modelPath: plan.modelPath,
            requireKVTunerQualificationIdentity: true)
        let expectedRuntimeIdentity = try KVTunerCandidateRuntimeIdentity.load(
            sourceSnapshot: before,
            eosTokenID: loaded.eos)
        let samples = try await loaded.driver.captureKVTunerSensitivity(
            prompts: manifest.sensitivityPrompts.map(\.tokenIDs),
            groupSize: plan.groupSize,
            expectedRuntimeIdentity: expectedRuntimeIdentity)
        let afterCapture = try
            captureKVTunerQualificationRuntimeSourceSnapshot(
                modelPath: plan.modelPath)
        _ = try KVTunerCandidateRuntimeSourceSnapshot.validateUnchanged(
            before: before, after: afterCapture)
        let artifact = try KVTunerSensitivityArtifact.makeAuthenticated(
            matrixID: plan.matrixID,
            groupSize: plan.groupSize,
            layerCount: layerCount,
            samples: samples,
            captureEnvironment: captureEnvironment,
            calibrationManifest: manifest,
            exactCalibrationManifestData: manifestData)
        let data = try KVTunerArtifactCodec.encode(artifact)
        try writeKVTunerQualificationResult(
            data, to: plan.outputPath, label: "kvtuner-sensitivity")
    } catch {
        print("kvtuner-sensitivity FAILED: \(error)")
        exit(1)
    }
}

private func requireKVTunerCandidateDirectory(
    _ path: String,
    createIfMissing: Bool = true
) throws {
    guard !path.isEmpty, !outputPathIsSymbolicLink(path) else {
        throw KVTunerQualificationCLIError.invalidCandidateDirectory(path)
    }
    let url = URL(fileURLWithPath: path).standardizedFileURL
    let manager = FileManager.default
    var isDirectory: ObjCBool = false
    if !manager.fileExists(atPath: url.path, isDirectory: &isDirectory) {
        guard createIfMissing else {
            throw KVTunerQualificationCLIError.invalidCandidateDirectory(
                path)
        }
        do {
            try manager.createDirectory(
                at: url, withIntermediateDirectories: false)
        } catch {
            throw KVTunerQualificationCLIError.invalidCandidateDirectory(
                path)
        }
        isDirectory = true
    }
    let attributes = try manager.attributesOfItem(atPath: url.path)
    guard isDirectory.boolValue,
        attributes[.type] as? FileAttributeType == .typeDirectory,
        manager.isWritableFile(atPath: url.path)
    else {
        throw KVTunerQualificationCLIError.invalidCandidateDirectory(path)
    }
}

func exactKVTunerCandidateArtifactData(
    directory: String,
    count: Int
) throws -> [Data] {
    guard count > 0 else {
        throw KVTunerQualificationCLIError.invalidArtifact(
            "candidate count")
    }
    try requireKVTunerCandidateDirectory(
        directory, createIfMissing: false)
    let expectedNames = Set((0..<count).map {
        kvtunerCandidateArtifactURL(
            directory: directory, ordinal: $0).lastPathComponent
    })
    let directoryURL = URL(
        fileURLWithPath: directory, isDirectory: true)
    let candidateNames = try FileManager.default.contentsOfDirectory(
        atPath: directoryURL.path).filter {
            $0.hasPrefix("candidate-") && $0.hasSuffix(".json")
        }
    guard Set(candidateNames) == expectedNames else {
        let missing = expectedNames.subtracting(candidateNames).sorted()
        if let first = missing.first,
            let ordinal = Int(
                first.dropFirst("candidate-".count).dropLast(5))
        {
            throw KVTunerQualificationCLIError.missingCandidateArtifact(
                ordinal)
        }
        throw KVTunerQualificationCLIError.invalidArtifact(
            "candidate directory contents")
    }
    return try (0..<count).map { ordinal in
        let path = kvtunerCandidateArtifactURL(
            directory: directory, ordinal: ordinal).path
        guard let data = try existingKVTunerQualificationData(at: path)
        else {
            throw KVTunerQualificationCLIError.missingCandidateArtifact(
                ordinal)
        }
        return data
    }
}

private func kvtunerCandidateExecutionEnvironment(
    harnessGitSHA: String,
    policy: KVTunerCandidateRuntimePolicy
) throws -> KVTunerCandidateExecutionEnvironment {
    try requireKVTunerReleaseBuild()
    return KVTunerCandidateExecutionEnvironment(
        harnessGitSHA: harnessGitSHA,
        buildConfiguration:
            KVTunerCandidateExecutionEnvironment
                .requiredBuildConfiguration,
        mlxSwiftVersion: ProvenanceCLI.mlxSwiftVersion,
        mlxSwiftLMRevision: ProvenanceCLI.mlxSwiftLMRevision,
        hardwareChip: ProvenanceCLI.chipBrand(),
        hardwareRAMBytes: ProvenanceCLI.ramBytes(),
        memoryCacheLimitBytes:
            KVTunerCandidateExecutionEnvironment
                .requiredMemoryCacheLimitBytes,
        hardwareOS: ProvenanceCLI.osVersion(),
        modelConfigHash: policy.modelConfigHash,
        modelConfigSHA256: policy.modelConfigSHA256,
        checkpointManifestHash: policy.checkpointManifestHash,
        checkpointContentSHA256: policy.checkpointContentSHA256,
        tokenizerSHA256: policy.tokenizerSHA256)
}

private struct KVTunerCandidateJob {
    let policy: KVTunerCandidateRuntimePolicy
    let contract: KVTunerCandidateRuntimeContract
    let environment: KVTunerCandidateExecutionEnvironment
    let outputPath: String
}

private func validateExistingKVTunerCandidate(
    _ data: Data,
    path: String,
    job: KVTunerCandidateJob,
    manifest: KVTunerCalibrationManifest,
    manifestData: Data,
    decodeTokenIDs: ([Int]) throws -> String
) throws {
    do {
        let artifact = try JSONDecoder().decode(
            KVTunerCandidateEvaluationArtifact.self, from: data)
        guard artifact.executionEnvironment == job.environment,
            try KVTunerArtifactCodec.encode(artifact) == data
        else {
            throw KVTunerQualificationCLIError.invalidArtifact(path)
        }
        _ = try artifact.validated(
            exactArtifactData: data,
            runtimePolicy: job.policy,
            runtimeContract: job.contract,
            calibrationManifest: manifest,
            exactCalibrationManifestData: manifestData,
            decodeTokenIDs: decodeTokenIDs)
    } catch let error as KVTunerQualificationCLIError {
        throw error
    } catch {
        throw KVTunerQualificationCLIError.invalidArtifact(path)
    }
}

func runKVTunerCandidate(_ flags: Flags) async {
    do {
        let plan = try parseKVTunerCandidatePlan(flags)
        let harnessGitSHA = try ProvenanceCLI.qualificationHarnessGitSHA()
        try requireKVTunerReleaseBuild()
        try requireKVTunerCandidateDirectory(plan.outputDirectory)
        let manifestData = try kvtunerQualificationData(
            at: plan.manifestPath)
        let manifest = try decodedKVTunerManifest(
            manifestData, path: plan.manifestPath)
        let sensitivityData = try kvtunerQualificationData(
            at: plan.sensitivityPath)
        let sensitivity = try decodedKVTunerSensitivity(
            sensitivityData,
            path: plan.sensitivityPath,
            manifest: manifest,
            manifestData: manifestData)
        let before = try captureKVTunerQualificationRuntimeSourceSnapshot(
            modelPath: plan.modelPath)
        try requireKVTunerSnapshot(before, matches: manifest)

        let cpuTokenizer = try await AutoTokenizer.from(
            modelFolder: URL(fileURLWithPath: plan.modelPath))
        guard let eosTokenID = cpuTokenizer.eosToken.flatMap({
            cpuTokenizer.convertTokenToId($0)
        }) else {
            throw KVTunerQualificationCLIError.missingEOSToken
        }
        let tokenizePrompt: (String) throws -> [Int] = {
            cpuTokenizer.encode(text: $0, addSpecialTokens: true)
        }
        let decodeTokenIDs: ([Int]) throws -> String = {
            cpuTokenizer.decode(tokens: $0)
        }
        try validateKVTunerManifestTokenization(
            manifest, tokenizePrompt: tokenizePrompt)
        let candidates = try KVTunerScheduleSearch.enumerate(
            analysis: sensitivity.analyzed(),
            targetPairBitTotal: plan.targetPairBitTotal,
            maxCandidates: plan.maxCandidates)
        let ordinals: [Int]
        if let ordinal = plan.candidateOrdinal {
            guard candidates.indices.contains(ordinal) else {
                throw KVTunerQualificationCLIError.invalidOrdinal(ordinal)
            }
            ordinals = [ordinal]
        } else {
            ordinals = Array(candidates.indices)
        }

        var pending: [KVTunerCandidateJob] = []
        for ordinal in ordinals {
            let policy = try KVTunerCandidateRuntimePolicy.load(
                exactCalibrationManifestData: manifestData,
                exactSensitivityArtifactData: sensitivityData,
                exactModelConfigData: before.exactModelConfigData,
                expectedCheckpointManifestHash:
                    before.checkpointManifestHash,
                expectedCheckpointContentSHA256:
                    before.checkpointContentSHA256,
                expectedTokenizerSHA256: before.tokenizerSHA256,
                targetPairBitTotal: plan.targetPairBitTotal,
                maxCandidates: plan.maxCandidates,
                candidateOrdinal: ordinal,
                tokenizePrompt: tokenizePrompt)
            let contract = try KVTunerCandidateRuntimeContract.load(
                exactModelConfigData: before.exactModelConfigData,
                runtimePolicy: policy,
                eosTokenID: eosTokenID)
            let outputPath = kvtunerCandidateArtifactURL(
                directory: plan.outputDirectory, ordinal: ordinal).path
            let job = KVTunerCandidateJob(
                policy: policy,
                contract: contract,
                environment: try kvtunerCandidateExecutionEnvironment(
                    harnessGitSHA: harnessGitSHA, policy: policy),
                outputPath: outputPath)
            if let existing = try existingKVTunerQualificationData(
                at: outputPath)
            {
                try validateExistingKVTunerCandidate(
                    existing,
                    path: outputPath,
                    job: job,
                    manifest: manifest,
                    manifestData: manifestData,
                    decodeTokenIDs: decodeTokenIDs)
                print(
                    "kvtuner-candidate: ordinal \(ordinal) already complete "
                        + "sha256=\(sha256Hex(existing))")
            } else {
                pending.append(job)
            }
        }
        guard !pending.isEmpty else {
            print("kvtuner-candidate: all requested ordinals complete")
            return
        }

        let afterPreflight = try
            captureKVTunerQualificationRuntimeSourceSnapshot(
                modelPath: plan.modelPath)
        _ = try KVTunerCandidateRuntimeSourceSnapshot.validateUnchanged(
            before: before, after: afterPreflight)
        print(
            "# kvtuner-candidate loading model after complete artifact, "
                + "tokenizer, source, clean-SHA, and Release preflight passed")
        let loaded = try await loadSwiftDriver(
            modelPath: plan.modelPath,
            requireKVTunerQualificationIdentity: true)
        guard loaded.eos == eosTokenID else {
            throw KVTunerQualificationCLIError.sourceIdentityMismatch
        }
        for job in pending {
            let results = try await loaded.driver
                .evaluateKVTunerCandidateCohort(
                    prompts: manifest.searchPrompts.map(\.tokenIDs),
                    maxTokens: KVTunerSearchEvaluationProtocol.canonical
                        .maxGeneratedTokens,
                    policy: job.policy)
            let afterExecution = try
                captureKVTunerQualificationRuntimeSourceSnapshot(
                    modelPath: plan.modelPath)
            _ = try KVTunerCandidateRuntimeSourceSnapshot
                .validateUnchanged(
                    before: before, after: afterExecution)
            let rows = results.map {
                KVTunerCandidateGeneratedRow(
                    promptOrdinal: $0.promptOrdinal,
                    promptTokenIDsSHA256: $0.promptTokenIDsSHA256,
                    generatedTokenIDs: $0.tokens,
                    finishReason: $0.finishReason,
                    runtimeReceipt: $0.telemetry.evidenceReceipt)
            }
            let artifact = try KVTunerCandidateEvaluationArtifact
                .makeAuthenticated(
                    runtimePolicy: job.policy,
                    runtimeContract: job.contract,
                    executionEnvironment: job.environment,
                    calibrationManifest: manifest,
                    exactCalibrationManifestData: manifestData,
                    generatedRows: rows,
                    decodeTokenIDs: {
                        loaded.tokenizer.decode(
                            tokenIds: $0, skipSpecialTokens: false)
                    })
            let data = try KVTunerArtifactCodec.encode(artifact)
            try writeKVTunerQualificationResult(
                data,
                to: job.outputPath,
                label:
                    "kvtuner-candidate[\(job.policy.candidateOrdinal)]")
        }
    } catch {
        print("kvtuner-candidate FAILED: \(error)")
        exit(1)
    }
}

private func kvtunerRuntimePolicies(
    count: Int,
    manifestData: Data,
    sensitivityData: Data,
    snapshot: KVTunerCandidateRuntimeSourceSnapshot,
    targetPairBitTotal: Int,
    maxCandidates: Int,
    tokenizePrompt: (String) throws -> [Int]
) throws -> [KVTunerCandidateRuntimePolicy] {
    try (0..<count).map { ordinal in
        try KVTunerCandidateRuntimePolicy.load(
            exactCalibrationManifestData: manifestData,
            exactSensitivityArtifactData: sensitivityData,
            exactModelConfigData: snapshot.exactModelConfigData,
            expectedCheckpointManifestHash:
                snapshot.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                snapshot.checkpointContentSHA256,
            expectedTokenizerSHA256: snapshot.tokenizerSHA256,
            targetPairBitTotal: targetPairBitTotal,
            maxCandidates: maxCandidates,
            candidateOrdinal: ordinal,
            tokenizePrompt: tokenizePrompt)
    }
}

func validateKVTunerCandidateExecutionEnvironments(
    _ environments: [KVTunerCandidateExecutionEnvironment],
    currentHarnessGitSHA: String
) throws {
    guard let first = environments.first else {
        throw KVTunerQualificationCLIError.invalidArtifact(
            "candidate environment cohort")
    }
    guard first.harnessGitSHA == currentHarnessGitSHA else {
        throw KVTunerQualificationCLIError.invalidHarnessGitSHA(
            first.harnessGitSHA)
    }
    guard environments.allSatisfy({ $0 == first }) else {
        throw KVTunerQualificationCLIError.invalidArtifact(
            "mixed candidate execution environments")
    }
}

private func requireKVTunerCandidateExecutionEnvironment(
    _ candidateData: [Data],
    currentHarnessGitSHA: String
) throws {
    do {
        let environments = try candidateData.map {
            try JSONDecoder().decode(
                KVTunerCandidateEvaluationArtifact.self, from: $0)
                .executionEnvironment
        }
        try validateKVTunerCandidateExecutionEnvironments(
            environments, currentHarnessGitSHA: currentHarnessGitSHA)
    } catch let error as KVTunerQualificationCLIError {
        throw error
    } catch {
        throw KVTunerQualificationCLIError.invalidArtifact(
            "candidate environment")
    }
}

func runKVTunerSearch(_ flags: Flags) async {
    do {
        let plan = try parseKVTunerSearchPlan(flags)
        let harnessGitSHA = try ProvenanceCLI.qualificationHarnessGitSHA()
        try requireKVTunerReleaseBuild()
        let manifestData = try kvtunerQualificationData(
            at: plan.manifestPath)
        let manifest = try decodedKVTunerManifest(
            manifestData, path: plan.manifestPath)
        let sensitivityData = try kvtunerQualificationData(
            at: plan.sensitivityPath)
        let sensitivity = try decodedKVTunerSensitivity(
            sensitivityData,
            path: plan.sensitivityPath,
            manifest: manifest,
            manifestData: manifestData)
        let before = try captureKVTunerQualificationRuntimeSourceSnapshot(
            modelPath: plan.modelPath)
        try requireKVTunerSnapshot(before, matches: manifest)
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: URL(fileURLWithPath: plan.modelPath))
        guard let eosTokenID = tokenizer.eosToken.flatMap({
            tokenizer.convertTokenToId($0)
        }) else {
            throw KVTunerQualificationCLIError.missingEOSToken
        }
        let tokenizePrompt: (String) throws -> [Int] = {
            tokenizer.encode(text: $0, addSpecialTokens: true)
        }
        let decodeTokenIDs: ([Int]) throws -> String = {
            tokenizer.decode(tokens: $0)
        }
        try validateKVTunerManifestTokenization(
            manifest, tokenizePrompt: tokenizePrompt)
        let candidates = try KVTunerScheduleSearch.enumerate(
            analysis: sensitivity.analyzed(),
            targetPairBitTotal: plan.targetPairBitTotal,
            maxCandidates: plan.maxCandidates)
        let candidateData = try exactKVTunerCandidateArtifactData(
            directory: plan.candidateDirectory,
            count: candidates.count)
        try requireKVTunerCandidateExecutionEnvironment(
            candidateData, currentHarnessGitSHA: harnessGitSHA)
        let policies = try kvtunerRuntimePolicies(
            count: candidates.count,
            manifestData: manifestData,
            sensitivityData: sensitivityData,
            snapshot: before,
            targetPairBitTotal: plan.targetPairBitTotal,
            maxCandidates: plan.maxCandidates,
            tokenizePrompt: tokenizePrompt)
        let search = try KVTunerSearchArtifact.makeAuthenticated(
            targetPairBitTotal: plan.targetPairBitTotal,
            maxCandidates: plan.maxCandidates,
            sensitivityArtifact: sensitivity,
            exactSensitivityArtifactData: sensitivityData,
            calibrationManifest: manifest,
            exactCalibrationManifestData: manifestData,
            exactCandidateEvaluationArtifactData: candidateData,
            exactModelConfigData: before.exactModelConfigData,
            eosTokenID: eosTokenID,
            tokenizePrompt: tokenizePrompt,
            decodeTokenIDs: decodeTokenIDs)
        let searchData = try KVTunerArtifactCodec.encode(search)
        let schedule = try KVTunerScheduleSearch.makeSchedule(
            searchArtifact: search,
            exactSearchArtifactData: searchData,
            sensitivityArtifact: sensitivity,
            exactSensitivityArtifactData: sensitivityData,
            calibrationManifest: manifest,
            exactCalibrationManifestData: manifestData,
            exactCandidateEvaluationArtifactData: candidateData,
            candidateRuntimePolicies: policies,
            eosTokenID: eosTokenID,
            decodeTokenIDs: decodeTokenIDs,
            exactModelConfigData: before.exactModelConfigData,
            expectedCheckpointManifestHash:
                before.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                before.checkpointContentSHA256)
        let scheduleData = try KVTunerArtifactCodec.encode(schedule)
        let after = try captureKVTunerQualificationRuntimeSourceSnapshot(
            modelPath: plan.modelPath)
        _ = try KVTunerCandidateRuntimeSourceSnapshot.validateUnchanged(
            before: before, after: after)

        // Each artifact is independently resumable by exact bytes, so a crash between these two
        // durable moves cannot force replacement or leave an ambiguous mixed transaction.
        try writeKVTunerQualificationResult(
            scheduleData,
            to: plan.scheduleOutputPath,
            label: "kvtuner-schedule")
        try writeKVTunerQualificationResult(
            searchData,
            to: plan.outputPath,
            label: "kvtuner-search")
    } catch {
        print("kvtuner-search FAILED: \(error)")
        exit(1)
    }
}

func runKVTunerBundle(_ flags: Flags) async {
    do {
        let plan = try parseKVTunerBundlePlan(flags)
        let harnessGitSHA = try ProvenanceCLI.qualificationHarnessGitSHA()
        try requireKVTunerReleaseBuild()
        let manifestData = try kvtunerQualificationData(
            at: plan.manifestPath)
        let manifest = try decodedKVTunerManifest(
            manifestData, path: plan.manifestPath)
        let sensitivityData = try kvtunerQualificationData(
            at: plan.sensitivityPath)
        _ = try decodedKVTunerSensitivity(
            sensitivityData,
            path: plan.sensitivityPath,
            manifest: manifest,
            manifestData: manifestData)
        let searchData = try kvtunerQualificationData(
            at: plan.searchPath)
        let search: KVTunerSearchArtifact
        do {
            search = try JSONDecoder().decode(
                KVTunerSearchArtifact.self, from: searchData)
        } catch {
            throw KVTunerQualificationCLIError.invalidArtifact(
                plan.searchPath)
        }
        let scheduleData = try kvtunerQualificationData(
            at: plan.schedulePath)
        let candidateData = try exactKVTunerCandidateArtifactData(
            directory: plan.candidateDirectory,
            count: search.candidates.count)
        try requireKVTunerCandidateExecutionEnvironment(
            candidateData, currentHarnessGitSHA: harnessGitSHA)
        let before = try captureKVTunerQualificationRuntimeSourceSnapshot(
            modelPath: plan.modelPath)
        try requireKVTunerSnapshot(before, matches: manifest)
        let strongBefore = try
            captureCompressedKVAttentionRuntimeSourceSnapshot(
                modelPath: plan.modelPath)
        guard strongBefore.exactModelConfigData
                == before.exactModelConfigData,
            strongBefore.checkpointManifestHash
                == before.checkpointManifestHash,
            strongBefore.tokenizerSHA256 == before.tokenizerSHA256
        else {
            throw KVTunerQualificationCLIError.sourceIdentityMismatch
        }
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: URL(fileURLWithPath: plan.modelPath))
        guard let eosTokenID = tokenizer.eosToken.flatMap({
            tokenizer.convertTokenToId($0)
        }) else {
            throw KVTunerQualificationCLIError.missingEOSToken
        }
        let tokenizePrompt: (String) throws -> [Int] = {
            tokenizer.encode(text: $0, addSpecialTokens: true)
        }
        let decodeTokenIDs: ([Int]) throws -> String = {
            tokenizer.decode(tokens: $0)
        }
        try validateKVTunerManifestTokenization(
            manifest, tokenizePrompt: tokenizePrompt)
        let bundle = try KVTunerQualificationBundle.makeAuthenticated(
            exactScheduleData: scheduleData,
            exactCalibrationManifestData: manifestData,
            exactSensitivityArtifactData: sensitivityData,
            exactSearchArtifactData: searchData,
            exactCandidateEvaluationArtifactData: candidateData,
            exactModelConfigData: before.exactModelConfigData,
            eosTokenID: eosTokenID,
            expectedCheckpointManifestHash:
                before.checkpointManifestHash,
            expectedCheckpointContentSHA256:
                strongBefore.checkpointContentSHA256,
            tokenizePrompt: tokenizePrompt,
            decodeTokenIDs: decodeTokenIDs)
        let after = try captureKVTunerQualificationRuntimeSourceSnapshot(
            modelPath: plan.modelPath)
        _ = try KVTunerCandidateRuntimeSourceSnapshot.validateUnchanged(
            before: before, after: after)
        let strongAfter = try captureCompressedKVAttentionRuntimeSourceSnapshot(
            modelPath: plan.modelPath)
        _ = try CompressedKVAttentionRuntimeSourceSnapshot.validateUnchanged(
            before: strongBefore, after: strongAfter)
        let data = try KVTunerArtifactCodec.encode(bundle)
        try writeKVTunerQualificationResult(
            data, to: plan.outputPath, label: "kvtuner-bundle")
    } catch {
        print("kvtuner-bundle FAILED: \(error)")
        exit(1)
    }
}
