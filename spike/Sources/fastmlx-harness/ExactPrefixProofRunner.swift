import Darwin
import Foundation
import HarnessCore
import MLX
import MLXLMCommon

private enum ExactPrefixProofRunnerError: Error, Equatable {
    case missingAdmission
    case sourceIdentityChanged
    case memorySettingsNotApplied
    case emptyGeneration(ExactPrefixProofCaseID)
    case missingRequestMetrics(ExactPrefixProofCaseID)
    case unexpectedRequestMetrics(ExactPrefixProofCaseID)
    case invalidPressurePolicy
    case unsafeHostEnvironment
    case templateCacheInsertionFailed(ExactPrefixProofCaseID)
    case reusedPrefixUnavailable(ExactPrefixProofCaseID)
    case reusedPrefixDoesNotMatchPrompt(ExactPrefixProofCaseID)
    case invalidValidationArguments
}

func runExactPrefixProof(_ rawArguments: [String]) async throws {
    let raw = try parseExactPrefixProofRawCommand(arguments: rawArguments)
    let startedAt = ProcessInfo.processInfo.systemUptime
    let actualHarnessSHA = try ProvenanceCLI.qualificationHarnessGitSHA()
    guard actualHarnessSHA == raw.expectedHarnessSHA else {
        throw ExactPrefixProofCLIError.harnessIdentityMismatch
    }
    let executableIdentity = try authenticateCurrentExecutable(
        expectedSHA256: raw.expectedExecutableSHA256)
    let boundary = try ExactPrefixProofOutputBoundary.claim(
        directoryPath: raw.outputPath)
    var completedCases = 0

    func status(
        _ state: ExactPrefixProofOutputState,
        completedCases: Int,
        error: String? = nil
    ) -> ExactPrefixProofOutputStatus {
        ExactPrefixProofOutputStatus(
            state: state,
            processID: ProcessInfo.processInfo.processIdentifier,
            completedCases: completedCases,
            totalCases: ExactPrefixProofCaseID.requiredOrder.count,
            elapsedSeconds:
                ProcessInfo.processInfo.systemUptime - startedAt,
            harnessGitSHA: actualHarnessSHA,
            executableSHA256: executableIdentity.sha256,
            modelID: raw.modelID,
            sourceRevision: raw.sourceRevision,
            workloadNonce: raw.workloadNonce,
            error: error)
    }

    do {
        try boundary.writeStatus(status(.starting, completedCases: 0))
        let hostBeforeLoad = try benchQualificationHostSnapshot()
        try requireExactPrefixProofSafeHost(hostBeforeLoad)
        let configuration = try ExactPrefixCacheConfiguration(
            policy: raw.exactPrefixCachePolicy,
            eagerWarmupEnabled: false)
        print(
            "# exact-prefix-proof loading authenticated model with eager warmup disabled")
        let loaded = try await loadSwiftDriver(
            modelPath: raw.modelPath,
            exactPrefixCacheConfiguration: configuration,
            memoryLimitBytes: raw.memoryLimitBytes,
            memoryCacheLimitBytes: raw.cacheLimitBytes)
        guard Memory.memoryLimit == raw.memoryLimitBytes,
            Memory.cacheLimit == raw.cacheLimitBytes
        else {
            throw ExactPrefixProofRunnerError
                .memorySettingsNotApplied
        }
        guard let admission = loaded.driver
            .compressedKVAttentionAdmission
        else {
            throw ExactPrefixProofRunnerError.missingAdmission
        }
        let command = try raw.authenticated(
            actualHarnessSHA: actualHarnessSHA,
            actualExecutableSHA256: executableIdentity.sha256,
            admission: admission)
        try boundary.writeStatus(status(.running, completedCases: 0))
        print(
            "# exact-prefix-proof loaded proof modelID=\(command.plan.modelID) cases=\(ExactPrefixProofCaseID.requiredOrder.count)")

        let context = try exactPrefixProofRequestContext(
            plan: command.plan)
        var runtime = ExactPrefixProofRuntime(
            command: command,
            driver: loaded.driver,
            tokenizer: loaded.tokenizer,
            requestContext: context)
        let cases = try await runtime.runCases { completed in
            completedCases = completed
            try boundary.writeStatus(status(.running, completedCases: completed))
        }
        let warmupBefore = await loaded.driver.engine
            .exactPrefixCacheSnapshot()
        let warmupDuration = try await loaded.driver.engine
            .performExactPrefixWarmup()
        let warmupAfter = await loaded.driver.engine
            .exactPrefixCacheSnapshot()
        let warmup = ExactPrefixProofWarmupEvidence(
            durationSeconds: warmupDuration,
            before: warmupBefore,
            after: warmupAfter)
        let warmCases = try await runtime.runWarmCases(
            warmupSeconds: warmupDuration
        ) { completed in
            completedCases = completed
            try boundary.writeStatus(status(.running, completedCases: completed))
        }

        try revalidateExactPrefixProofSource(
            plan: command.plan,
            modelPath: command.modelPath,
            executableIdentity: executableIdentity)
        let hostAfterProof = try benchQualificationHostSnapshot()
        try requireExactPrefixProofSafeHost(hostAfterProof)
        let runtimeMetadata = try ExactPrefixProofRuntimeMetadata(
            capturedAtUTC: exactPrefixProofTimestampUTC(),
            hardwareChip: ProvenanceCLI.chipBrand(),
            hardwareRAMBytes: ProvenanceCLI.ramBytes(),
            hardwareOS: ProvenanceCLI.osVersion(),
            mlxSwiftVersion: ProvenanceCLI.mlxSwiftVersion,
            mlxSwiftLMRevision: ProvenanceCLI.mlxSwiftLMRevision,
            hostEnvironment: BenchQualificationWarmupEnvironment(
                before: hostBeforeLoad,
                after: hostAfterProof))
        let evidence = try ExactPrefixProofEvidence(
            modelID: command.plan.modelID,
            sourceRevision: command.plan.sourceRevision,
            admission: command.plan.admission,
            expectedHarnessSHA: command.plan.expectedHarnessSHA,
            expectedExecutableSHA256:
                command.plan.expectedExecutableSHA256,
            workloadNonce: command.plan.workloadNonce,
            maxTokens: command.plan.maxTokens,
            promptRepeat: command.plan.promptRepeat,
            exactPrefixCachePolicy:
                command.plan.exactPrefixCachePolicy,
            templateTokenCachePolicy:
                command.plan.templateTokenCachePolicy,
            memoryLimitBytes: command.plan.memoryLimitBytes,
            cacheLimitBytes: command.plan.cacheLimitBytes,
            requestContext: context,
            warmup: warmup,
            cases: cases + warmCases,
            terminalNonPartial: true)
        let artifact = try ExactPrefixProofArtifact(
            runtime: runtimeMetadata,
            evidence: evidence)
        _ = try publishExactPrefixProofArtifact(
            artifact,
            boundary: boundary,
            authenticatedExecutableIdentity: executableIdentity)
        try revalidateExactPrefixProofSource(
            plan: command.plan,
            modelPath: command.modelPath,
            executableIdentity: executableIdentity)
        try boundary.writeStatus(status(
            .complete,
            completedCases: evidence.cases.count))
        _ = try validateExactPrefixProofOutputDirectory(
            at: boundary.directoryURL)
        print(
            "# exact-prefix-proof COMPLETE output=\(boundary.directoryURL.path) promotable=\(evidence.promotable)")
    } catch {
        try? boundary.writeStatus(status(
            .failed,
            completedCases: completedCases,
            error: String(describing: error)))
        throw error
    }
}

@discardableResult
func runExactPrefixProofValidation(
    _ rawArguments: [String]
) throws -> ExactPrefixProofArtifact {
    guard rawArguments.count == 2, rawArguments[0] == "--output",
        !rawArguments[1].isEmpty,
        rawArguments[1] == rawArguments[1].trimmingCharacters(
            in: .whitespacesAndNewlines),
        (rawArguments[1] as NSString).isAbsolutePath
    else {
        throw ExactPrefixProofRunnerError.invalidValidationArguments
    }
    let artifact = try validateExactPrefixProofOutputDirectory(
        at: URL(fileURLWithPath: rawArguments[1], isDirectory: true))
    print(
        "# exact-prefix-proof validation COMPLETE output=\(rawArguments[1]) promotable=\(artifact.evidence.promotable)")
    return artifact
}

private struct ExactPrefixProofRuntime {
    private struct PreparedPrompt {
        let tokens: [Int]
        let receipt: ExactPrefixProofCacheReceipt
        let templateSeconds: Double
        let tokenizeSeconds: Double
    }

    let command: ExactPrefixProofCommand
    let driver: SwiftEngineDriver
    let tokenizer: Tokenizer
    let requestContext: ExactPrefixRequestContext
    var templateCache: TemplateTokenCache
    var references: [String: (
        generatedTokenIDsSHA256: String,
        outputSHA256: String
    )] = [:]
    var committedPrompts:
        [ExactPrefixProofCaseID: [Int]] = [:]

    init(
        command: ExactPrefixProofCommand,
        driver: SwiftEngineDriver,
        tokenizer: Tokenizer,
        requestContext: ExactPrefixRequestContext
    ) {
        self.command = command
        self.driver = driver
        self.tokenizer = tokenizer
        self.requestContext = requestContext
        self.templateCache = TemplateTokenCache(
            policy: command.plan.templateTokenCachePolicy)
    }

    mutating func runCases(
        progress: (Int) throws -> Void
    ) async throws -> [ExactPrefixProofCaseEvidence] {
        var rows: [ExactPrefixProofCaseEvidence] = []
        rows.append(try await runCase(.coldControlA, insertTemplate: false))
        try progress(rows.count)
        rows.append(try await runCase(.coldCommitA, insertTemplate: true))
        try progress(rows.count)
        rows.append(try await runCase(.exactHitA, insertTemplate: false))
        try progress(rows.count)
        rows.append(try await runCase(.partialControl, insertTemplate: true))
        try progress(rows.count)
        rows.append(try await runCase(.partialHit, insertTemplate: false))
        try progress(rows.count)
        try ensureTemplateAIsRetained()
        rows.append(try await runCase(.coldCommitB, insertTemplate: false))
        try progress(rows.count)
        rows.append(try await runCase(.returnHitA, insertTemplate: false))
        try progress(rows.count)
        try await forcePressureEviction {
            try progress(rows.count)
        }
        rows.append(try await runCase(.pressureEvictedA, insertTemplate: false))
        try progress(rows.count)
        return rows
    }

    mutating func runWarmCases(
        warmupSeconds: Double,
        progress: (Int) throws -> Void
    ) async throws -> [ExactPrefixProofCaseEvidence] {
        var rows: [ExactPrefixProofCaseEvidence] = []
        rows.append(try await runCase(.postWarmupControl, insertTemplate: false))
        try progress(8 + rows.count)
        rows.append(try await runCase(
            .postWarmupMiss,
            insertTemplate: true,
            warmupSeconds: warmupSeconds))
        try progress(8 + rows.count)
        rows.append(try await runCase(
            .postWarmupHit,
            insertTemplate: false,
            warmupSeconds: warmupSeconds))
        try progress(8 + rows.count)
        return rows
    }

    private mutating func runCase(
        _ caseID: ExactPrefixProofCaseID,
        insertTemplate: Bool,
        warmupSeconds: Double? = nil
    ) async throws -> ExactPrefixProofCaseEvidence {
        let caseStartedAt = ProcessInfo.processInfo.systemUptime
        let prepared = try preparePrompt(
            caseID: caseID,
            insertOnMiss: insertTemplate)
        Memory.peakMemory = 0
        let memoryStart = serviceMemorySample()
        let result = try await driver.generate(
            prompt: prepared.tokens,
            config: exactPrefixProofRunConfig(
                plan: command.plan,
                requestContext:
                    isControl(caseID) ? nil : requestContext))
        let caseEndedAt = ProcessInfo.processInfo.systemUptime
        let memoryEnd = serviceMemorySample()
        guard !result.tokens.isEmpty else {
            throw ExactPrefixProofRunnerError.emptyGeneration(caseID)
        }
        let requestMetrics = try requestMetrics(
            for: caseID,
            result: result,
            prepared: prepared,
            warmupSeconds: warmupSeconds)
        let reference = try referenceHashes(
            for: caseID,
            generatedTokens: result.tokens)
        let hostRequestSeconds =
            prepared.templateSeconds + prepared.tokenizeSeconds
        let actorTTFT = result.tokenTimes.first.map {
            $0 - result.submitTime
        } ?? result.prefillDurationSeconds ?? 0
        let requestStartSeconds = max(
            hostRequestSeconds + actorTTFT,
            Double.leastNonzeroMagnitude)
        let actorDecodeSeconds: Double
        if let first = result.tokenTimes.first,
            let last = result.tokenTimes.last
        {
            actorDecodeSeconds = last - first
        } else {
            actorDecodeSeconds = 0
        }
        let decodeSeconds = max(
            actorDecodeSeconds,
            Double.leastNonzeroMagnitude)
        let totalSeconds = max(
            caseEndedAt - caseStartedAt,
            requestStartSeconds + decodeSeconds)
        let reusedPrefix = try reusedPrefixBinding(
            for: caseID,
            prompt: prepared.tokens,
            metrics: requestMetrics)
        if caseID == .coldCommitA
            || caseID == .coldCommitB
            || caseID == .postWarmupMiss
        {
            committedPrompts[caseID] = prepared.tokens
        }
        return try ExactPrefixProofCaseEvidence(
            caseID: caseID,
            promptTokenIDsSHA256:
                taskTokenIDsSHA256(prepared.tokens),
            promptTokenCount: prepared.tokens.count,
            generatedTokenIDsSHA256: taskTokenIDsSHA256(result.tokens),
            outputSHA256: outputSHA256(result.tokens),
            referenceGeneratedTokenIDsSHA256:
                reference.generatedTokenIDsSHA256,
            referenceOutputSHA256: reference.outputSHA256,
            generatedTokenCount: result.tokens.count,
            timing: ExactPrefixProofCaseTiming(
                requestStartSeconds: requestStartSeconds,
                ttftSeconds: requestStartSeconds,
                decodeSeconds: decodeSeconds,
                totalSeconds: totalSeconds),
            requestStartMetrics: requestMetrics,
            memoryEvidence: try BenchRunMemoryEvidence(
                samples: [memoryStart, memoryEnd]),
            templateTokenCacheReceipt: prepared.receipt,
            reusedPrefixTokenIDsSHA256: reusedPrefix.sha256,
            reusedPrefixTokenCount: reusedPrefix.count,
            requestContext:
                isControl(caseID) ? nil : requestContext)
    }

    private func requestMetrics(
        for caseID: ExactPrefixProofCaseID,
        result: RunResult,
        prepared: PreparedPrompt,
        warmupSeconds: Double?
    ) throws -> RequestStartMetrics? {
        if isControl(caseID) {
            guard result.requestStartMetrics == nil else {
                throw ExactPrefixProofRunnerError
                    .unexpectedRequestMetrics(caseID)
            }
            return nil
        }
        guard let metrics = result.requestStartMetrics else {
            throw ExactPrefixProofRunnerError
                .missingRequestMetrics(caseID)
        }
        return try RequestStartMetrics(
            promptTokenCount: metrics.promptTokenCount,
            cacheReadTokenCount: metrics.cacheReadTokenCount,
            physicalPrefillTokenCount:
                metrics.physicalPrefillTokenCount,
            prefixCacheOutcome: metrics.prefixCacheOutcome,
            prefixCacheRejectionReason:
                metrics.prefixCacheRejectionReason,
            templateTokenCacheHit: prepared.receipt == .hit,
            templateSeconds: prepared.templateSeconds,
            tokenizeSeconds: prepared.tokenizeSeconds,
            lookupSeconds: metrics.lookupSeconds,
            restoreSeconds: metrics.restoreSeconds,
            prefillSeconds: metrics.prefillSeconds,
            retainedBytes: metrics.retainedBytes,
            entryCount: metrics.entryCount,
            evictionCount: metrics.evictionCount,
            eagerWarmupSeconds: warmupSeconds)
    }

    private func reusedPrefixBinding(
        for caseID: ExactPrefixProofCaseID,
        prompt: [Int],
        metrics: RequestStartMetrics?
    ) throws -> (sha256: String?, count: Int) {
        let sourceID: ExactPrefixProofCaseID?
        guard metrics?.prefixCacheOutcome == .exactHit
            || metrics?.prefixCacheOutcome == .partialHit
        else {
            return (nil, 0)
        }
        switch caseID {
        case .exactHitA, .partialHit, .returnHitA,
            .pressureEvictedA:
            sourceID = .coldCommitA
        case .postWarmupHit:
            sourceID = .postWarmupMiss
        case .coldControlA, .coldCommitA, .partialControl,
            .coldCommitB, .postWarmupControl, .postWarmupMiss:
            sourceID = nil
        }
        guard let sourceID else {
            return (nil, 0)
        }
        guard let sourcePrompt = committedPrompts[sourceID],
            let metrics
        else {
            throw ExactPrefixProofRunnerError
                .reusedPrefixUnavailable(caseID)
        }
        guard prompt.starts(with: sourcePrompt),
            metrics.cacheReadTokenCount == sourcePrompt.count
        else {
            throw ExactPrefixProofRunnerError
                .reusedPrefixDoesNotMatchPrompt(caseID)
        }
        return (
            taskTokenIDsSHA256(sourcePrompt),
            sourcePrompt.count)
    }

    private mutating func forcePressureEviction(
        heartbeat: () throws -> Void
    ) async throws {
        let (requiredFillerCount, overflow) =
            command.plan.exactPrefixCachePolicy.maxEntries
            .addingReportingOverflow(1)
        guard !overflow, requiredFillerCount > 0 else {
            throw ExactPrefixProofRunnerError.invalidPressurePolicy
        }
        for index in 0..<requiredFillerCount {
            let prompt = try pressurePromptTokens(index: index)
            _ = try await driver.generate(
                prompt: prompt,
                config: exactPrefixProofRunConfig(
                    plan: command.plan,
                    requestContext: requestContext))
            try heartbeat()
        }
    }

    private mutating func preparePrompt(
        caseID: ExactPrefixProofCaseID,
        insertOnMiss: Bool
    ) throws -> PreparedPrompt {
        let templateStartedAt =
            ProcessInfo.processInfo.systemUptime
        let key = try templateKey(caseID: caseID)
        if let cached = try templateCache.lookup(key: key) {
            return PreparedPrompt(
                tokens: cached,
                receipt: .hit,
                templateSeconds: max(
                    ProcessInfo.processInfo.systemUptime
                        - templateStartedAt,
                    0),
                tokenizeSeconds: 0)
        }
        let templateSeconds = max(
            ProcessInfo.processInfo.systemUptime
                - templateStartedAt,
            0)
        let tokenizeStartedAt =
            ProcessInfo.processInfo.systemUptime
        let prompt = try promptTokens(for: caseID)
        let tokenizeSeconds = max(
            ProcessInfo.processInfo.systemUptime
                - tokenizeStartedAt,
            0)
        if insertOnMiss {
            let decision = try templateCache.insert(
                key: key,
                tokenIDs: prompt)
            guard decision.inserted else {
                throw ExactPrefixProofRunnerError
                    .templateCacheInsertionFailed(caseID)
            }
        }
        return PreparedPrompt(
            tokens: prompt,
            receipt: .miss,
            templateSeconds: templateSeconds,
            tokenizeSeconds: tokenizeSeconds)
    }

    private mutating func ensureTemplateAIsRetained() throws {
        let key = try templateKey(caseID: .coldCommitA)
        if try templateCache.lookup(key: key) == nil {
            guard let prompt = committedPrompts[.coldCommitA] else {
                throw ExactPrefixProofRunnerError
                    .reusedPrefixUnavailable(.coldCommitA)
            }
            let decision = try templateCache.insert(
                key: key,
                tokenIDs: prompt)
            guard decision.inserted else {
                throw ExactPrefixProofRunnerError
                    .templateCacheInsertionFailed(.coldCommitA)
            }
        }
    }

    private func templateKey(
        caseID: ExactPrefixProofCaseID
    ) throws -> TemplateTokenCacheKey {
        try TemplateTokenCacheKey(
            isolationNamespaceSHA256:
                requestContext.isolationNamespaceSHA256,
            tokenizerSHA256: command.plan.tokenizerSHA256,
            promptTemplateSHA256:
                requestContext.promptTemplateSHA256,
            toolsSHA256: requestContext.toolsSHA256,
            promptContentSHA256: promptContentSHA256(for: caseID),
            formattingOptionsSHA256: sha256Hex(Data(
                "fast-mlx-exact-prefix-proof-format-v1\n".utf8)))
    }

    private mutating func referenceHashes(
        for caseID: ExactPrefixProofCaseID,
        generatedTokens: [Int]
    ) throws -> (
        generatedTokenIDsSHA256: String,
        outputSHA256: String
    ) {
        let group = referenceGroup(for: caseID)
        let current = (
            generatedTokenIDsSHA256: taskTokenIDsSHA256(generatedTokens),
            outputSHA256: outputSHA256(generatedTokens))
        if references[group] == nil || isReferenceCase(caseID) {
            references[group] = current
        }
        guard let reference = references[group] else {
            throw ExactPrefixProofRunnerError.emptyGeneration(caseID)
        }
        return reference
    }

    private func referenceGroup(
        for caseID: ExactPrefixProofCaseID
    ) -> String {
        switch caseID {
        case .coldControlA, .coldCommitA, .exactHitA, .returnHitA,
            .pressureEvictedA:
            return "A"
        case .partialControl, .partialHit:
            return "partial"
        case .coldCommitB:
            return "B"
        case .postWarmupControl, .postWarmupMiss, .postWarmupHit:
            return "warmup"
        }
    }

    private func isReferenceCase(
        _ caseID: ExactPrefixProofCaseID
    ) -> Bool {
        switch caseID {
        case .coldControlA, .partialControl, .coldCommitB,
            .postWarmupControl:
            return true
        case .coldCommitA, .exactHitA, .partialHit, .returnHitA,
            .pressureEvictedA, .postWarmupMiss, .postWarmupHit:
            return false
        }
    }

    private func promptTokens(
        for caseID: ExactPrefixProofCaseID
    ) throws -> [Int] {
        switch caseID {
        case .coldControlA, .coldCommitA, .exactHitA, .returnHitA,
            .pressureEvictedA:
            return try encodedPrompt(label: "A")
        case .partialControl, .partialHit:
            return try encodedPrompt(label: "A")
                + encodedPrompt(label: "partial-tail")
        case .coldCommitB:
            return try encodedPrompt(label: "B")
        case .postWarmupControl, .postWarmupMiss, .postWarmupHit:
            return try encodedPrompt(label: "warmup")
        }
    }

    private func pressurePromptTokens(index: Int) throws -> [Int] {
        try encodedPrompt(label: "pressure-\(index)")
    }

    private func encodedPrompt(label: String) throws -> [Int] {
        let line = """
            exact-prefix-proof \(command.plan.workloadNonce) \(label): deterministic ledger row for cache admission and byte identity.
            """
        let text = Array(repeating: line, count: command.plan.promptRepeat)
            .joined(separator: "\n")
        let tokens = tokenizer.encode(text: text, addSpecialTokens: true)
        guard !tokens.isEmpty else {
            throw ExactPrefixProofRunnerError.emptyGeneration(.coldControlA)
        }
        return tokens
    }

    private func promptContentSHA256(
        for caseID: ExactPrefixProofCaseID
    ) -> String {
        let group: String
        switch caseID {
        case .coldControlA, .coldCommitA, .exactHitA, .returnHitA,
            .pressureEvictedA:
            group = "A"
        case .partialControl, .partialHit:
            group = "partial"
        case .coldCommitB:
            group = "B"
        case .postWarmupControl, .postWarmupMiss, .postWarmupHit:
            group = "warmup"
        }
        return sha256Hex(Data(
            "fast-mlx-exact-prefix-proof-content-v1\n\(command.plan.workloadNonce)\n\(group)\n".utf8))
    }

    private func outputSHA256(_ tokens: [Int]) -> String {
        sha256Hex(Data(tokenizer.decode(
            tokenIds: tokens,
            skipSpecialTokens: false).utf8))
    }
}

func exactPrefixProofRunConfig(
    plan: ExactPrefixProofCommandPlan,
    requestContext: ExactPrefixRequestContext?
) -> RunConfig {
    RunConfig(
        temperature: 0,
        maxTokens: plan.maxTokens,
        exactPrefixRequest: requestContext)
}

private func exactPrefixProofRequestContext(
    plan: ExactPrefixProofCommandPlan
) throws -> ExactPrefixRequestContext {
    try ExactPrefixRequestContext(
        isolationNamespaceSHA256: sha256Hex(Data(
            "fast-mlx-exact-prefix-proof-namespace-v1\n\(plan.workloadNonce)\n".utf8)),
        promptTemplateSHA256: sha256Hex(Data(
            "fast-mlx-exact-prefix-proof-template-v1\nfixed-ledger\n".utf8)),
        toolsSHA256: sha256Hex(Data(
            "fast-mlx-exact-prefix-proof-tools-v1\nnone\n".utf8)))
}

private func revalidateExactPrefixProofSource(
    plan: ExactPrefixProofCommandPlan,
    modelPath: String,
    executableIdentity: ExactPrefixExecutableIdentity
) throws {
    guard try ProvenanceCLI.qualificationHarnessGitSHA()
        == plan.expectedHarnessSHA
    else {
        throw ExactPrefixProofCLIError.harnessIdentityMismatch
    }
    try validateExecutableUnchanged(executableIdentity)
    let source = try captureCompressedKVAttentionRuntimeSourceSnapshot(
        modelPath: modelPath)
    let admission = try CompressedKVAttentionRuntimeAdmission.load(
        sourceSnapshot: source)
    guard admission == plan.admission,
        source.checkpointContentSHA256 == plan.checkpointContentSHA256,
        source.tokenizerSHA256 == plan.tokenizerSHA256
    else {
        throw ExactPrefixProofRunnerError.sourceIdentityChanged
    }
}

private func exactPrefixProofTimestampUTC() -> String {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [
        .withInternetDateTime,
        .withDashSeparatorInDate,
        .withColonSeparatorInTime,
    ]
    formatter.timeZone = TimeZone(secondsFromGMT: 0)
    return formatter.string(from: Date())
}

private func requireExactPrefixProofSafeHost(
    _ snapshot: BenchQualificationHostSnapshot
) throws {
    guard snapshot.monotonicTimestampSeconds.isFinite,
        snapshot.residentSizeBytes > 0,
        snapshot.physicalFootprintBytes > 0,
        snapshot.powerSource == .acPower,
        !snapshot.lowPowerModeEnabled,
        snapshot.thermalState == .nominal
            || snapshot.thermalState == .fair
    else {
        throw ExactPrefixProofRunnerError.unsafeHostEnvironment
    }
}

private func isControl(_ caseID: ExactPrefixProofCaseID) -> Bool {
    switch caseID {
    case .coldControlA, .partialControl, .postWarmupControl:
        return true
    case .coldCommitA, .exactHitA, .partialHit, .coldCommitB,
        .returnHitA, .pressureEvictedA, .postWarmupMiss,
        .postWarmupHit:
        return false
    }
}
