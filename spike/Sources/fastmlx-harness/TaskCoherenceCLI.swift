import Foundation
import HarnessCore
import MLXLMCommon
import SpikeCore

private enum TaskCoherenceCLIError: Error, CustomStringConvertible {
    case missingFlag(String)
    case invalidIdentifier(String, String)
    case unsupportedTierCell(tier: String, cell: String)
    case invalidTokenization(label: String, spelling: String, tokenIDs: [Int])
    case invalidToolTokenBudget(Int)
    case outputNotFresh(String)
    case outputPathCollision(String)
    case invalidReference(String)
    case missingEngagement(String)
    case dirtyHarnessSHA(String)

    var description: String {
        switch self {
        case .missingFlag(let flag):
            return "missing required --\(flag)"
        case .invalidIdentifier(let flag, let value):
            return "--\(flag) is not a stable identifier: \(String(reflecting: value))"
        case .unsupportedTierCell(let tier, let cell):
            return "task-coherence tier/cell is not runnable: tier=\(tier), cell=\(cell)"
        case .invalidTokenization(let label, let spelling, let tokenIDs):
            return "label \(label) spelling \(String(reflecting: spelling)) must encode to one distinct token; got \(tokenIDs)"
        case .invalidToolTokenBudget(let value):
            return "--max-tool-tokens must be in 1...512; got \(value)"
        case .outputNotFresh(let path):
            return "task evidence destination must be a new or empty writable regular file: \(path)"
        case .outputPathCollision(let path):
            return "task evidence paths must be distinct: \(path)"
        case .invalidReference(let reason):
            return "fp16 task reference is incompatible: \(reason)"
        case .missingEngagement(let marker):
            return "requested cache did not return exact task engagement marker \(marker)"
        case .dirtyHarnessSHA(let sha):
            return "task qualification requires a clean 40/64-hex harness SHA; got \(sha)"
        }
    }
}

private struct TaskCoherenceRunPlan {
    let modelPath: String
    let matrixID: String
    let cellID: String
    let tier: String
    let evidencePath: String
    let summaryEvidencePath: String?
    let referenceEvidencePath: String?
    let kvtunerSchedulePath: String?
    let maxToolTokens: Int
}

private struct PreparedTaskCoherenceCase {
    let item: TaskCoherenceItem
    let promptTokens: [Int]
    let tokenization: TaskCoherenceTokenizationEvidence
    let layout: TaskCoherencePromptLayoutEvidence
}

private func requireTaskIdentifier(_ value: String, flag: String) throws {
    let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty, trimmed == value, value != "unknown",
        !value.contains("\n"), !value.contains("\r")
    else {
        throw TaskCoherenceCLIError.invalidIdentifier(flag, value)
    }
}

private func freshTaskOutput(_ path: String) throws {
    guard !path.isEmpty else {
        throw TaskCoherenceCLIError.missingFlag("evidence")
    }
    guard !outputPathIsSymbolicLink(path) else {
        throw TaskCoherenceCLIError.outputNotFresh(path)
    }
    let manager = FileManager.default
    let url = URL(fileURLWithPath: path).standardizedFileURL
    var isDirectory: ObjCBool = false
    if manager.fileExists(atPath: url.path, isDirectory: &isDirectory) {
        let attributes = try manager.attributesOfItem(atPath: url.path)
        let type = attributes[.type] as? FileAttributeType
        let size = (attributes[.size] as? NSNumber)?.uint64Value
        guard !isDirectory.boolValue, type == .typeRegular, size == 0,
            manager.isWritableFile(atPath: url.path)
        else { throw TaskCoherenceCLIError.outputNotFresh(path) }
        return
    }
    let parent = url.deletingLastPathComponent().path
    var parentIsDirectory: ObjCBool = false
    guard manager.fileExists(
        atPath: parent, isDirectory: &parentIsDirectory),
        parentIsDirectory.boolValue,
        manager.isWritableFile(atPath: parent)
    else { throw TaskCoherenceCLIError.outputNotFresh(path) }
}

private func requireDistinctTaskPaths(_ paths: [String]) throws {
    let configured = paths.filter { !$0.isEmpty }
    for leftIndex in configured.indices {
        for rightIndex in configured.indices where rightIndex > leftIndex {
            guard !outputPathsReferToSameFile(
                configured[leftIndex], configured[rightIndex])
            else {
                throw TaskCoherenceCLIError.outputPathCollision(
                    configured[rightIndex])
            }
        }
    }
}

private func parseTaskCoherenceRunPlan(_ flags: Flags) throws
    -> TaskCoherenceRunPlan
{
    let modelPath = try flags.strictString("model", default: "")
    let matrixID = try flags.strictString("matrix-id", default: "")
    let cellID = try flags.strictString("cell-id", default: "")
    let tier = try flags.strictString("kv-quant", default: "fp16")
    let output = try flags.strictString("evidence", default: "")
    let summaryOutput = try flags.strictString(
        "summary-evidence", default: "")
    let reference = try flags.strictString(
        "reference-task-evidence", default: "")
    let kvtunerSchedule = try flags.strictString(
        "kvtuner-schedule", default: "")
    let maxToolTokens = try flags.strictInt("max-tool-tokens", default: 96)

    guard !modelPath.isEmpty else {
        throw TaskCoherenceCLIError.missingFlag("model")
    }
    guard !matrixID.isEmpty else {
        throw TaskCoherenceCLIError.missingFlag("matrix-id")
    }
    guard !cellID.isEmpty else {
        throw TaskCoherenceCLIError.missingFlag("cell-id")
    }
    guard !output.isEmpty else {
        throw TaskCoherenceCLIError.missingFlag("evidence")
    }
    try requireTaskIdentifier(matrixID, flag: "matrix-id")
    try requireTaskIdentifier(cellID, flag: "cell-id")
    try validateKVTunerScheduleFlag(
        tier: tier, cellID: cellID, schedulePath: kvtunerSchedule)
    if !isKVTunerTier(tier) {
        guard TaskCoherenceArtifact.expectedCellID(forTier: tier) == cellID,
            KVCacheKind(kvQuant: tier) != nil
        else {
            throw TaskCoherenceCLIError.unsupportedTierCell(
                tier: tier, cell: cellID)
        }
    }
    guard (1 ... 512).contains(maxToolTokens) else {
        throw TaskCoherenceCLIError.invalidToolTokenBudget(maxToolTokens)
    }
    if tier == "fp16" {
        guard reference.isEmpty else {
            throw TaskCoherenceCLIError.invalidReference(
                "fp16 baseline must not name a reference artifact")
        }
    } else if reference.isEmpty {
        throw TaskCoherenceCLIError.missingFlag("reference-task-evidence")
    }

    try requireDistinctTaskPaths([
        output, summaryOutput, reference, kvtunerSchedule,
    ])

    try freshTaskOutput(output)
    if !summaryOutput.isEmpty { try freshTaskOutput(summaryOutput) }
    return TaskCoherenceRunPlan(
        modelPath: modelPath, matrixID: matrixID, cellID: cellID,
        tier: tier, evidencePath: output,
        summaryEvidencePath: summaryOutput.isEmpty ? nil : summaryOutput,
        referenceEvidencePath: reference.isEmpty ? nil : reference,
        kvtunerSchedulePath:
            kvtunerSchedule.isEmpty ? nil : kvtunerSchedule,
        maxToolTokens: maxToolTokens)
}

private func cleanTaskHarnessSHA(_ value: String) -> Bool {
    [40, 64].contains(value.count) && value.unicodeScalars.allSatisfy {
        CharacterSet(charactersIn: "0123456789abcdefABCDEF").contains($0)
    }
}

private func taskRuntimeMatches(
    reference: TaskCoherenceArtifactSummary,
    candidate: Provenance
) -> Bool {
    let value = reference.provenance
    return value.hardwareChip == candidate.hardwareChip
        && value.hardwareRAMBytes == candidate.hardwareRAMBytes
        && value.hardwareOS == candidate.hardwareOS
        && value.harnessGitSHA == candidate.harnessGitSHA
        && value.mlxSwiftVersion == candidate.mlxSwiftVersion
        && value.modelPath == candidate.modelPath
        && value.modelConfigHash == candidate.modelConfigHash
        && value.modelCheckpointManifestHash
            == candidate.modelCheckpointManifestHash
        && value.modelQuant == candidate.modelQuant
}

private func validateTaskReferenceBeforeModelLoad(
    _ reference: TaskCoherenceArtifactSummary,
    plan: TaskCoherenceRunPlan,
    corpus: TaskCoherenceCorpus,
    identity: KVModelEvidenceIdentity,
    provenance: Provenance,
    tokenizerManifestSHA256: String,
    runConfiguration: TaskCoherenceRunConfiguration
) throws {
    guard reference.matrixID == plan.matrixID else {
        throw TaskCoherenceCLIError.invalidReference("matrix ID mismatch")
    }
    guard reference.cellID == "fp16",
        reference.identity.kvQuantTier == "fp16"
    else {
        throw TaskCoherenceCLIError.invalidReference("not an fp16 task run")
    }
    guard reference.identity.corpusID == corpus.id,
        reference.identity.corpusContentHash == corpus.contentHash
    else {
        throw TaskCoherenceCLIError.invalidReference("corpus mismatch")
    }
    guard reference.identity.modelConfigHash == identity.configHash,
        reference.identity.modelCheckpointManifestHash
            == identity.checkpointManifestHash
    else {
        throw TaskCoherenceCLIError.invalidReference("model identity mismatch")
    }
    guard taskRuntimeMatches(reference: reference, candidate: provenance) else {
        throw TaskCoherenceCLIError.invalidReference(
            "hardware, software, model path, or clean SHA mismatch")
    }
    guard reference.runConfiguration == runConfiguration else {
        throw TaskCoherenceCLIError.invalidReference(
            "run configuration mismatch")
    }
    guard reference.cases.first?.tokenization.tokenizerManifestSHA256
        == tokenizerManifestSHA256
    else {
        throw TaskCoherenceCLIError.invalidReference(
            "tokenizer manifest mismatch")
    }
    for domain in TaskCoherenceDomain.allCases {
        let rows = reference.scores.filter { $0.domain == domain }
        let score = Double(rows.filter(\.correct).count) / Double(rows.count)
        guard score > 0.25 else {
            throw TaskCoherenceCLIError.invalidReference(
                "\(domain.rawValue) baseline is not above chance")
        }
    }
    let structured = reference.scores.filter {
        $0.domain == .structuredTool
    }
    let validRate = Double(structured.filter {
        $0.syntacticallyValid == true
    }.count) / Double(structured.count)
    guard validRate >= 0.90 else {
        throw TaskCoherenceCLIError.invalidReference(
            "structured syntax baseline is below 90%")
    }
}

private func taskLabelTokenIDs(
    tokenizer: MLXLMCommon.Tokenizer
) throws -> [String: Int] {
    var result: [String: Int] = [:]
    for label in ["A", "B", "C", "D"] {
        guard let spelling =
            TaskRestrictedChoiceScorer.labelTokenSpellings[label]
        else {
            throw TaskCoherenceCLIError.invalidTokenization(
                label: label, spelling: "", tokenIDs: [])
        }
        let tokenIDs = tokenizer.encode(
            text: spelling, addSpecialTokens: false)
        guard tokenIDs.count == 1 else {
            throw TaskCoherenceCLIError.invalidTokenization(
                label: label, spelling: spelling, tokenIDs: tokenIDs)
        }
        result[label] = tokenIDs[0]
    }
    guard Set(result.values).count == 4 else {
        throw TaskCoherenceCLIError.invalidTokenization(
            label: "A-D", spelling: "leading-space labels",
            tokenIDs: ["A", "B", "C", "D"].compactMap { result[$0] })
    }
    return result
}

private func prepareTaskCoherenceCases(
    corpus: TaskCoherenceCorpus,
    tokenizer: MLXLMCommon.Tokenizer,
    tokenizerManifestSHA256: String,
    labelTokenIDs: [String: Int]
) throws -> [PreparedTaskCoherenceCase] {
    try corpus.items.map { item in
        let promptTokens = tokenizer.encode(text: item.prompt)
        let layout = try TaskCoherencePromptLayoutEvidence.derive(
            prefixTokenIDs: tokenizer.encode(text: item.prefix),
            prefixAndMaterialTokenIDs: tokenizer.encode(
                text: item.prefix + item.material),
            suffixAndQueryTokenIDs: tokenizer.encode(
                text: item.suffix + item.query),
            promptTokenIDs: promptTokens)
        return PreparedTaskCoherenceCase(
            item: item,
            promptTokens: promptTokens,
            tokenization: TaskCoherenceTokenizationEvidence(
                tokenizerManifestSHA256: tokenizerManifestSHA256,
                promptTokenIDsSHA256: taskTokenIDsSHA256(promptTokens),
                restrictedChoiceLabelTokenIDs:
                    item.scoringMode == .restrictedChoice
                        ? labelTokenIDs : nil),
            layout: layout)
    }
}

private func taskEngagement(
    tier: String,
    kvtunerSchedule: KVTunerScheduleBinding?,
    generated: EngagementCounters,
    scoring: EngagementCounters?
) throws -> TaskCoherenceCacheEngagementEvidence {
    if tier == "fp16" {
        return TaskCoherenceCacheEngagementEvidence(
            cachedTokens: nil, affineTokens: nil,
            kvarnCompletedTileCount: nil,
            kvarnCompressedTokens: nil,
            kvarnCodecIterations: nil,
            kvarnExecutionMode: nil)
    }
    if tier.hasPrefix("affine-") {
        guard let cached = generated.counts["affine_tokens"] else {
            throw TaskCoherenceCLIError.missingEngagement("affine_tokens")
        }
        if scoring != nil,
            scoring?.counts["scoring_cached_tokens"] == nil
        {
            throw TaskCoherenceCLIError.missingEngagement(
                "scoring_cached_tokens")
        }
        return TaskCoherenceCacheEngagementEvidence(
            cachedTokens: cached, affineTokens: cached,
            kvarnCompletedTileCount: nil,
            kvarnCompressedTokens: nil,
            kvarnCodecIterations: nil,
            kvarnExecutionMode: nil,
            scoringCachedTokens:
                scoring?.counts["scoring_cached_tokens"])
    }
    if let kvtunerSchedule {
        guard let cached = generated.counts["kvtuner_tokens"],
            let layers = generated.counts["kvtuner_layers"],
            layers == kvtunerSchedule.layers.count
        else {
            throw TaskCoherenceCLIError.missingEngagement(
                "kvtuner generation")
        }
        if scoring != nil {
            for marker in [
                "scoring_cached_tokens", "scoring_kvtuner_layers",
            ] where scoring?.counts[marker] == nil {
                throw TaskCoherenceCLIError.missingEngagement(marker)
            }
        }
        return TaskCoherenceCacheEngagementEvidence(
            cachedTokens: cached, affineTokens: nil,
            kvtunerTokens: cached,
            kvtunerLayerCount: layers,
            kvarnCompletedTileCount: nil,
            kvarnCompressedTokens: nil,
            kvarnCodecIterations: nil,
            kvarnExecutionMode: nil,
            scoringCachedTokens:
                scoring?.counts["scoring_cached_tokens"],
            scoringKVTunerLayerCount:
                scoring?.counts["scoring_kvtuner_layers"])
    }
    guard let cached = generated.counts["kvarn_tokens"],
        let completed = generated.counts["kvarn_completed_tiles"],
        let compressed = generated.counts["kvarn_compressed_tokens"],
        let iterations = generated.counts["kvarn_codec_iterations"],
        generated.counts["kvarn_uncompiled_correctness"] == 1
    else {
        throw TaskCoherenceCLIError.missingEngagement("kvarn generation")
    }
    if scoring != nil {
        for marker in [
            "scoring_cached_tokens", "scoring_kvarn_completed_tiles",
            "scoring_kvarn_compressed_tokens",
        ] where scoring?.counts[marker] == nil {
            throw TaskCoherenceCLIError.missingEngagement(marker)
        }
    }
    return TaskCoherenceCacheEngagementEvidence(
        cachedTokens: cached, affineTokens: nil,
        kvarnCompletedTileCount: completed,
        kvarnCompressedTokens: compressed,
        kvarnCodecIterations: iterations,
        kvarnExecutionMode: "uncompiled-correctness",
        scoringCachedTokens: scoring?.counts["scoring_cached_tokens"],
        scoringKVarNCompletedTileCount:
            scoring?.counts["scoring_kvarn_completed_tiles"],
        scoringKVarNCompressedTokens:
            scoring?.counts["scoring_kvarn_compressed_tokens"])
}

private func printTaskSummary(_ summary: TaskCoherenceArtifactSummary) {
    print("# task artifact sha256=\(summary.artifactSHA256) cases=\(summary.caseCount)")
    for domain in TaskCoherenceDomain.allCases {
        let rows = summary.scores.filter { $0.domain == domain }
        print("# task \(domain.rawValue): \(rows.filter(\.correct).count)/\(rows.count)")
    }
    let structured = summary.scores.filter {
        $0.domain == .structuredTool
    }
    print("# structured syntax: \(structured.filter { $0.syntacticallyValid == true }.count)/\(structured.count)")
}

private func printTaskAssessment(_ assessment: TaskCoherenceAssessment) {
    for domain in assessment.domains {
        print(
            "# floor \(domain.domain.rawValue): "
                + "\(domain.correct)/\(domain.denominator), "
                + "score=\(fmt(domain.score, 3)), "
                + "fp16=\(fmt(domain.referenceScore, 3)), "
                + "delta_pp=\(fmt(domain.deltaPercentagePoints, 1)), "
                + "chance=\(fmt(domain.chanceBaseline, 2)), "
                + "half_fp16=\(fmt(domain.halfReferenceScore, 3)), "
                + "\(domain.hardFloorPassed ? "PASS" : "FAIL")")
    }
    let structured = assessment.structuredValidity
    print(
        "# floor structured-syntax: \(structured.valid)/\(structured.denominator), "
            + "rate=\(fmt(structured.rate, 3)), "
            + "fp16=\(fmt(structured.referenceRate, 3)), "
            + "\(structured.passed ? "PASS" : "FAIL")")
    print("# task hard floor: \(assessment.hardFloorPassed ? "PASS" : "FAIL")")
    print("# task Balanced <=5pp: \(assessment.balancedTaskDeltaPassed ? "PASS" : "FAIL")")
}

func runTaskCoherence(_ flags: Flags) async {
    do {
        let plan = try parseTaskCoherenceRunPlan(flags)
        let corpus = try TaskCoherenceCorpusV1.make()
        let modelIdentity = try ProvenanceCLI.modelEvidenceIdentity(
            at: plan.modelPath)
        let tokenizerManifestSHA256 =
            try ProvenanceCLI.tokenizerManifestSHA256(at: plan.modelPath)
        let runConfiguration =
            TaskCoherenceRunConfiguration.qualificationV2(
                structuredToolMaxTokens: plan.maxToolTokens)
        let (provenance, _) = ProvenanceCLI.build(
            modelPath: plan.modelPath, referenceVersions: nil,
            taskCorpus: corpus,
            modelCheckpointManifestHash:
                modelIdentity.checkpointManifestHash)
        guard cleanTaskHarnessSHA(provenance.harnessGitSHA) else {
            throw TaskCoherenceCLIError.dirtyHarnessSHA(
                provenance.harnessGitSHA)
        }

        let reference: TaskCoherenceArtifactSummary?
        if let path = plan.referenceEvidencePath {
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            let parsed = try TaskCoherenceArtifact.summarize(
                data, corpus: corpus)
            try validateTaskReferenceBeforeModelLoad(
                parsed, plan: plan, corpus: corpus,
                identity: modelIdentity, provenance: provenance,
                tokenizerManifestSHA256: tokenizerManifestSHA256,
                runConfiguration: runConfiguration)
            reference = parsed
        } else {
            reference = nil
        }

        let preparedKVTuner: PreparedKVTunerRun?
        if let schedulePath = plan.kvtunerSchedulePath {
            preparedKVTuner = try await prepareKVTunerRun(
                schedulePath: schedulePath,
                modelPath: plan.modelPath,
                matrixID: plan.matrixID,
                cellID: plan.cellID,
                modelIdentity: modelIdentity,
                evaluationCorpus:
                    KVTunerEvaluationCorpusIdentity.taskCoherenceCorpus(
                        corpus))
        } else {
            preparedKVTuner = nil
        }

        print("# task-coherence loading model after all artifact preflight gates passed")
        let (driver, tokenizer, _) = try await loadSwiftDriver(
            modelPath: plan.modelPath)
        let labelTokenIDs = try taskLabelTokenIDs(tokenizer: tokenizer)
        print("# restricted-choice token IDs: \(labelTokenIDs)")
        let preparedCases = try prepareTaskCoherenceCases(
            corpus: corpus,
            tokenizer: tokenizer,
            tokenizerManifestSHA256: tokenizerManifestSHA256,
            labelTokenIDs: labelTokenIDs)
        if let reference {
            guard reference.cases.count == preparedCases.count else {
                throw TaskCoherenceCLIError.invalidReference(
                    "case count mismatch")
            }
            for (prepared, referenceCase) in zip(
                preparedCases, reference.cases)
            {
                guard prepared.tokenization == referenceCase.tokenization,
                    prepared.layout == referenceCase.layout
                else {
                    throw TaskCoherenceCLIError.invalidReference(
                        "prompt tokenization or layout mismatch for \(prepared.item.id)")
                }
            }
        }
        let configTier = plan.tier == "fp16" ? nil : plan.tier
        func runConfig(maxTokens: Int) -> RunConfig {
            RunConfig(
                temperature: 0,
                maxTokens: maxTokens,
                kvQuant: configTier,
                kvtunerSelection: preparedKVTuner?.selection)
        }
        let identity = TaskCoherenceRunIdentity(
            corpusID: corpus.id,
            corpusContentHash: corpus.contentHash,
            modelConfigHash: modelIdentity.configHash,
            modelCheckpointManifestHash:
                modelIdentity.checkpointManifestHash,
            kvQuantTier: plan.tier,
            kvtunerSchedule: preparedKVTuner?.binding)

        for (index, prepared) in preparedCases.enumerated() {
            let item = prepared.item
            let promptTokens = prepared.promptTokens
            let layout = prepared.layout

            let scoredOutput: String
            let score: TaskItemScore
            let generation: RunResult
            let scoringEngagement: EngagementCounters?
            switch item.scoringMode {
            case .restrictedChoice:
                let scoring = try await driver.taskChoiceLogits(
                    prompt: promptTokens,
                    config: runConfig(maxTokens: 1))
                scoredOutput = try TaskRestrictedChoiceScorer.predict(
                    logits: scoring.logits,
                    labelTokenIDs: labelTokenIDs)
                generation = try await driver.generate(
                    prompt: promptTokens,
                    config: runConfig(maxTokens: 1))
                scoringEngagement = scoring.engagement
                guard let expected = item.expectedChoice else {
                    throw TaskCoherenceCLIError.invalidReference(
                        "missing expected choice for \(item.id)")
                }
                score = TaskItemScore(
                    itemID: item.id, domain: item.domain,
                    correct: scoredOutput == expected,
                    syntacticallyValid: nil)
            case .structuredTool:
                generation = try await driver.generate(
                    prompt: promptTokens,
                    config: runConfig(maxTokens: plan.maxToolTokens))
                scoredOutput = tokenizer.decode(
                    tokenIds: generation.tokens,
                    skipSpecialTokens: true)
                scoringEngagement = nil
                guard let expected = item.expectedTool else {
                    throw TaskCoherenceCLIError.invalidReference(
                        "missing expected tool for \(item.id)")
                }
                let structured = TaskStructuredToolScorer.score(
                    scoredOutput, expected: expected)
                score = TaskItemScore(
                    itemID: item.id, domain: item.domain,
                    correct: structured.correct,
                    syntacticallyValid: structured.syntacticallyValid)
            }
            guard !generation.tokens.isEmpty else {
                throw TaskCoherenceCLIError.missingEngagement(
                    "generated token")
            }
            let engagement = try taskEngagement(
                tier: plan.tier,
                kvtunerSchedule: preparedKVTuner?.binding,
                generated: generation.engagement,
                scoring: scoringEngagement)
            let payload = TaskCoherenceCasePayload(
                schemaVersion: TaskCoherenceArtifact.schemaVersion,
                matrixID: plan.matrixID,
                cellID: plan.cellID,
                identity: identity,
                referenceArtifactSHA256: reference?.artifactSHA256,
                promptContentHash: fnv1a64(item.prompt.utf8),
                runConfiguration: runConfiguration,
                tokenization: prepared.tokenization,
                layout: layout,
                generatedTokenCount: generation.tokens.count,
                scoredOutput: scoredOutput,
                outputSHA256: sha256Hex(Data(scoredOutput.utf8)),
                score: score,
                engagement: engagement)
            try requireDistinctTaskPaths([
                plan.evidencePath,
                plan.summaryEvidencePath ?? "",
                plan.referenceEvidencePath ?? "",
                plan.kvtunerSchedulePath ?? "",
            ])
            try appendRequiredJSONLRecord(
                ResultRecord(
                    subcommand: "task-coherence",
                    provenance: provenance,
                    payload: payload),
                to: plan.evidencePath)
            print(
                "# task \(index + 1)/\(corpus.items.count) \(item.id): "
                    + "\(score.correct ? "correct" : "incorrect")")
        }

        let rawData = try Data(contentsOf: URL(
            fileURLWithPath: plan.evidencePath))
        let summary = try TaskCoherenceArtifact.summarize(
            rawData, corpus: corpus)
        printTaskSummary(summary)
        if let output = plan.summaryEvidencePath {
            // The raw run has now made any late hard-link/path alias observable. Reassert the
            // destination immediately before its first append so summary bytes can never mix into
            // the authenticated raw JSONL.
            try freshTaskOutput(output)
            try requireDistinctTaskPaths([
                plan.evidencePath,
                output,
                plan.referenceEvidencePath ?? "",
                plan.kvtunerSchedulePath ?? "",
            ])
            try appendRequiredJSONLRecord(
                ResultRecord(
                    subcommand: "task-coherence-summary",
                    provenance: provenance,
                    payload: summary),
                to: output)
        }

        if let reference {
            let promotion = try TaskCoherencePromotionEvidence.derive(
                candidate: summary, reference: reference, corpus: corpus)
            printTaskAssessment(promotion.assessment)
            if let output = plan.summaryEvidencePath {
                try requireDistinctTaskPaths([
                    plan.evidencePath,
                    output,
                    plan.referenceEvidencePath ?? "",
                    plan.kvtunerSchedulePath ?? "",
                ])
                try appendRequiredJSONLRecord(
                    ResultRecord(
                        subcommand: "task-coherence-assessment",
                        provenance: provenance,
                        payload: promotion),
                    to: output)
            }
            guard promotion.assessment.hardFloorPassed else { exit(1) }
        }
        print("task-coherence: PASS")
    } catch {
        print("task-coherence FAILED: \(error)")
        exit(1)
    }
}
