import Foundation

public enum SampledMTPBlockDistributionCacheGate: Equatable, Sendable {
    public struct Configuration: Equatable, Sendable {
        public var blockSeed: UInt64
        public var targetSamplerSeed: UInt64
        public var sampleCount: Int
        public var distributionTolerance: Double
        public var analyticTolerance: Double

        public init(
            blockSeed: UInt64,
            targetSamplerSeed: UInt64,
            sampleCount: Int,
            distributionTolerance: Double,
            analyticTolerance: Double)
        {
            self.blockSeed = blockSeed
            self.targetSamplerSeed = targetSamplerSeed
            self.sampleCount = sampleCount
            self.distributionTolerance = distributionTolerance
            self.analyticTolerance = analyticTolerance
        }

        public static let defaultFixture = Configuration(
            blockSeed: 0x5eed_5eed_2026_0827,
            targetSamplerSeed: 0x7461_7267_6574_0827,
            sampleCount: 200_000,
            distributionTolerance: 0.012,
            analyticTolerance: 1e-12)
    }

    public enum AnalyticOutcome: Equatable, Hashable, Sendable {
        case rejectFirst
        case rejectSecond
        case acceptAll
    }

    public enum DistributionRow: Equatable, Hashable, Sendable {
        case firstToken
        case secondToken(first: Int)
        case bonus(first: Int, second: Int)
    }

    public enum CacheLayoutEvidence: Equatable, Hashable, Sendable {
        case dense
        case qwenStyleHybrid
    }

    public struct AnalyticCheck: Equatable, Sendable {
        public let outcome: AnalyticOutcome
        public let maxAbsoluteDelta: Double
        public let tolerance: Double

        public var passed: Bool {
            maxAbsoluteDelta <= tolerance
        }

        public init(outcome: AnalyticOutcome, maxAbsoluteDelta: Double, tolerance: Double) {
            self.outcome = outcome
            self.maxAbsoluteDelta = maxAbsoluteDelta
            self.tolerance = tolerance
        }
    }

    public struct DistributionCheck: Equatable, Sendable {
        public let row: DistributionRow
        public let expectedDistribution: [Double]
        public let mtpDistribution: [Double]
        public let targetSamplerDistribution: [Double]
        public let mtpObservationCount: Int
        public let targetSamplerObservationCount: Int
        public let maxAbsoluteDelta: Double
        public let targetSamplerMaxAbsoluteDelta: Double
        public let tolerance: Double

        public var passed: Bool {
            maxAbsoluteDelta <= tolerance && targetSamplerMaxAbsoluteDelta <= tolerance
        }

        public init(
            row: DistributionRow,
            expectedDistribution: [Double],
            mtpDistribution: [Double],
            targetSamplerDistribution: [Double],
            mtpObservationCount: Int,
            targetSamplerObservationCount: Int,
            maxAbsoluteDelta: Double,
            targetSamplerMaxAbsoluteDelta: Double,
            tolerance: Double)
        {
            self.row = row
            self.expectedDistribution = expectedDistribution
            self.mtpDistribution = mtpDistribution
            self.targetSamplerDistribution = targetSamplerDistribution
            self.mtpObservationCount = mtpObservationCount
            self.targetSamplerObservationCount = targetSamplerObservationCount
            self.maxAbsoluteDelta = maxAbsoluteDelta
            self.targetSamplerMaxAbsoluteDelta = targetSamplerMaxAbsoluteDelta
            self.tolerance = tolerance
        }
    }

    public struct FixtureCacheSnapshot: Equatable, Sendable, SpeculativeCacheSnapshotProtocol {
        public let kinds: [LayerCacheKind]
        public var tokenLengthsByLayer: [Int]
        public var storedTokensByLayer: [[Int]]
        public var recurrentStateByLayer: [[Int]]

        public init(
            kinds: [LayerCacheKind],
            tokenLengthsByLayer: [Int],
            storedTokensByLayer: [[Int]],
            recurrentStateByLayer: [[Int]])
        {
            self.kinds = kinds
            self.tokenLengthsByLayer = tokenLengthsByLayer
            self.storedTokensByLayer = storedTokensByLayer
            self.recurrentStateByLayer = recurrentStateByLayer
        }

        func appending(tokens: [Int]) -> FixtureCacheSnapshot {
            var copy = self
            copy.append(tokens: tokens)
            return copy
        }

        mutating func append(tokens: [Int]) {
            for layer in kinds.indices {
                tokenLengthsByLayer[layer] += tokens.count
                switch kinds[layer] {
                case .denseAttention:
                    storedTokensByLayer[layer].append(contentsOf: tokens)
                case .recurrentState:
                    recurrentStateByLayer[layer].append(contentsOf: tokens.map { $0 + layer })
                }
            }
        }
    }

    public struct CacheCompositionCheck: Equatable, Sendable {
        public let layout: CacheLayoutEvidence
        public let outcome: AnalyticOutcome
        public let acceptedDraftCount: Int
        public let proposedDraftTokens: [Int]
        public let expectedCommittedInputTokens: [Int]
        public let committedInputTokens: [Int]
        public let expectedRejectedDraftCount: Int
        public let rejectedDraftCount: Int
        public let emittedTerminalToken: Int
        public let terminalOutputWasCommitted: Bool
        public let finalSnapshot: FixtureCacheSnapshot
        public let scalarEquivalentSnapshot: FixtureCacheSnapshot

        public var passed: Bool {
            layoutMatchesSnapshot
                && acceptedDraftCount == expectedAcceptedDraftCount
                && proposedDraftTokens.count == Self.proposalCount
                && expectedRejectedDraftCount == Self.proposalCount - acceptedDraftCount
                && expectedCommittedInputTokens.count == acceptedDraftCount + 1
                && Array(expectedCommittedInputTokens.dropFirst())
                    == Array(proposedDraftTokens.prefix(acceptedDraftCount))
                && committedInputTokens == expectedCommittedInputTokens
                && rejectedDraftCount == expectedRejectedDraftCount
                && !committedInputTokens.contains(emittedTerminalToken)
                && !terminalOutputWasCommitted
                && !finalSnapshot.contains(token: emittedTerminalToken)
                && finalSnapshot == scalarEquivalentSnapshot
        }

        private static let proposalCount = 2

        private var expectedAcceptedDraftCount: Int {
            switch outcome {
            case .rejectFirst:
                return 0
            case .rejectSecond:
                return 1
            case .acceptAll:
                return 2
            }
        }

        private var layoutMatchesSnapshot: Bool {
            switch layout {
            case .dense:
                return !finalSnapshot.kinds.isEmpty
                    && finalSnapshot.kinds.allSatisfy { $0 == .denseAttention }
            case .qwenStyleHybrid:
                return finalSnapshot.kinds.contains(.denseAttention)
                    && finalSnapshot.kinds.contains(.recurrentState)
            }
        }

        public init(
            layout: CacheLayoutEvidence,
            outcome: AnalyticOutcome,
            acceptedDraftCount: Int,
            proposedDraftTokens: [Int],
            expectedCommittedInputTokens: [Int],
            committedInputTokens: [Int],
            expectedRejectedDraftCount: Int,
            rejectedDraftCount: Int,
            emittedTerminalToken: Int,
            terminalOutputWasCommitted: Bool,
            finalSnapshot: FixtureCacheSnapshot,
            scalarEquivalentSnapshot: FixtureCacheSnapshot)
        {
            self.layout = layout
            self.outcome = outcome
            self.acceptedDraftCount = acceptedDraftCount
            self.proposedDraftTokens = proposedDraftTokens
            self.expectedCommittedInputTokens = expectedCommittedInputTokens
            self.committedInputTokens = committedInputTokens
            self.expectedRejectedDraftCount = expectedRejectedDraftCount
            self.rejectedDraftCount = rejectedDraftCount
            self.emittedTerminalToken = emittedTerminalToken
            self.terminalOutputWasCommitted = terminalOutputWasCommitted
            self.finalSnapshot = finalSnapshot
            self.scalarEquivalentSnapshot = scalarEquivalentSnapshot
        }
    }

    public struct Verdict: Equatable, Sendable {
        public let blockSeed: UInt64
        public let targetSamplerSeed: UInt64
        public let sampleCount: Int
        public let distributionTolerance: Double
        public let analyticTolerance: Double
        public let analyticChecks: [AnalyticCheck]
        public let distributionChecks: [DistributionCheck]
        public let cacheChecks: [CacheCompositionCheck]

        public var passed: Bool {
            analyticChecks.allSatisfy(\.passed)
                && distributionChecks.allSatisfy(\.passed)
                && cacheChecks.allSatisfy(\.passed)
        }

        public init(
            blockSeed: UInt64,
            targetSamplerSeed: UInt64,
            sampleCount: Int,
            distributionTolerance: Double,
            analyticTolerance: Double,
            analyticChecks: [AnalyticCheck],
            distributionChecks: [DistributionCheck],
            cacheChecks: [CacheCompositionCheck])
        {
            self.blockSeed = blockSeed
            self.targetSamplerSeed = targetSamplerSeed
            self.sampleCount = sampleCount
            self.distributionTolerance = distributionTolerance
            self.analyticTolerance = analyticTolerance
            self.analyticChecks = analyticChecks
            self.distributionChecks = distributionChecks
            self.cacheChecks = cacheChecks
        }
    }

    public enum Error: Swift.Error, Equatable, Sendable {
        case invalidConfiguration(String)
        case invalidFixture(String)
        case analyticMismatch(AnalyticCheck)
        case distributionMismatch(DistributionCheck)
        case cacheCompositionMismatch(CacheCompositionCheck)
        case cacheTransactionFailed(SpeculativeCacheTransactionError)
    }

    public static func runDefaultFixture() throws -> Verdict {
        try run(configuration: .defaultFixture)
    }

    public static func run(configuration: Configuration) throws -> Verdict {
        try validate(configuration)
        let model = MarkovFixtureModel()

        let analyticChecks = try analyticChecks(model: model, tolerance: configuration.analyticTolerance)
        if let failed = analyticChecks.first(where: { !$0.passed }) {
            throw Error.analyticMismatch(failed)
        }

        let distributionChecks = try distributionChecks(model: model, configuration: configuration)
        if let failed = distributionChecks.first(where: { !$0.passed }) {
            throw Error.distributionMismatch(failed)
        }

        let cacheChecks = try cacheChecks()
        if let failed = cacheChecks.first(where: { !$0.passed }) {
            throw Error.cacheCompositionMismatch(failed)
        }

        return Verdict(
            blockSeed: configuration.blockSeed,
            targetSamplerSeed: configuration.targetSamplerSeed,
            sampleCount: configuration.sampleCount,
            distributionTolerance: configuration.distributionTolerance,
            analyticTolerance: configuration.analyticTolerance,
            analyticChecks: analyticChecks,
            distributionChecks: distributionChecks,
            cacheChecks: cacheChecks)
    }

    private static func validate(_ configuration: Configuration) throws {
        guard configuration.sampleCount > 0 else {
            throw Error.invalidConfiguration("sample count must be positive")
        }
        guard configuration.distributionTolerance.isFinite, configuration.distributionTolerance >= 0.0 else {
            throw Error.invalidConfiguration("distribution tolerance must be finite and non-negative")
        }
        guard configuration.analyticTolerance.isFinite, configuration.analyticTolerance >= 0.0 else {
            throw Error.invalidConfiguration("analytic tolerance must be finite and non-negative")
        }
    }

    private static func analyticChecks(
        model: MarkovFixtureModel,
        tolerance: Double
    ) throws -> [AnalyticCheck] {
        let firstResidual = try SampledMTPResidualCorrection.residualDistribution(
            target: model.firstTarget,
            draft: model.firstDraft)
        let firstAcceptance = try acceptanceProbabilities(target: model.firstTarget, draft: model.firstDraft)
        let rejectFirstMass = rejectedMass(draft: model.firstDraft, acceptance: firstAcceptance)
        let reconstructedFirst = model.firstTarget.indices.map { token in
            model.firstDraft[token] * firstAcceptance[token] + rejectFirstMass * firstResidual[token]
        }

        var secondMaxDelta = 0.0
        for first in model.vocabulary {
            let target = model.secondTarget(after: first)
            let draft = model.secondDraft(after: first)
            let residual = try SampledMTPResidualCorrection.residualDistribution(target: target, draft: draft)
            let acceptance = try acceptanceProbabilities(target: target, draft: draft)
            let mass = rejectedMass(draft: draft, acceptance: acceptance)
            let reconstructed = target.indices.map { token in
                draft[token] * acceptance[token] + mass * residual[token]
            }
            secondMaxDelta = max(secondMaxDelta, maxAbsoluteDelta(reconstructed, target))
        }

        var bonusMaxDelta = 0.0
        for first in model.vocabulary {
            for second in model.vocabulary {
                let bonus = model.bonusTarget(after: first, second: second)
                let reconstructed = try reconstructBonusWithActualAcceptedBlock(
                    first: first,
                    second: second,
                    distribution: bonus,
                    vocabularyCount: model.vocabularyCount)
                bonusMaxDelta = max(bonusMaxDelta, maxAbsoluteDelta(reconstructed, bonus))
            }
        }

        return [
            AnalyticCheck(
                outcome: .rejectFirst,
                maxAbsoluteDelta: maxAbsoluteDelta(reconstructedFirst, model.firstTarget),
                tolerance: tolerance),
            AnalyticCheck(
                outcome: .rejectSecond,
                maxAbsoluteDelta: secondMaxDelta,
                tolerance: tolerance),
            AnalyticCheck(
                outcome: .acceptAll,
                maxAbsoluteDelta: bonusMaxDelta,
                tolerance: tolerance),
        ]
    }

    private static func distributionChecks(
        model: MarkovFixtureModel,
        configuration: Configuration
    ) throws -> [DistributionCheck] {
        var blockRNG = SplitMix64(seed: configuration.blockSeed)
        var targetRNG = SplitMix64(seed: configuration.targetSamplerSeed)
        var mtpRows: [DistributionRow: [Int]] = [
            .firstToken: zeroCounts(model.vocabularyCount),
        ]

        for _ in 0..<configuration.sampleCount {
            let decision = try sampleBlock(model: model, rng: &blockRNG)
            guard let first = decision.tokens.first else {
                throw Error.invalidFixture("block decision emitted no token")
            }
            mtpRows[.firstToken, default: zeroCounts(model.vocabularyCount)][first] += 1

            if decision.acceptedDraftCount >= 1 {
                let second = decision.tokens[1]
                mtpRows[.secondToken(first: first), default: zeroCounts(model.vocabularyCount)][second] += 1
            }

            if decision.acceptedDraftCount == 2 {
                let second = decision.tokens[1]
                let bonus = decision.tokens[2]
                mtpRows[.bonus(first: first, second: second), default: zeroCounts(model.vocabularyCount)][bonus] += 1
            }
        }

        var checks: [DistributionCheck] = []
        let rows = declaredRows(model: model)
        for row in rows {
            let expected = model.targetDistribution(for: row)
            let mtpCounts = mtpRows[row, default: zeroCounts(model.vocabularyCount)]
            let targetCounts = targetSampleCounts(
                distribution: expected,
                sampleCount: configuration.sampleCount,
                rng: &targetRNG)
            let mtpDistribution = frequencies(mtpCounts)
            let targetDistribution = frequencies(targetCounts)
            checks.append(DistributionCheck(
                row: row,
                expectedDistribution: expected,
                mtpDistribution: mtpDistribution,
                targetSamplerDistribution: targetDistribution,
                mtpObservationCount: mtpCounts.reduce(0, +),
                targetSamplerObservationCount: targetCounts.reduce(0, +),
                maxAbsoluteDelta: maxAbsoluteDelta(mtpDistribution, expected),
                targetSamplerMaxAbsoluteDelta: maxAbsoluteDelta(targetDistribution, expected),
                tolerance: configuration.distributionTolerance))
        }
        return checks
    }

    internal static func runCacheBeginFailureFixtureForTesting() throws {
        var backend = FixtureCacheBackend.dense(layerCount: 1, length: 4, prefix: [1])
        do {
            _ = try SpeculativeCacheTransaction.begin(
                layout: .dense(layerCount: 0),
                committedInputToken: 900,
                draftTokens: [0, 1],
                backend: &backend)
        } catch let error as SpeculativeCacheTransactionError {
            throw Error.cacheTransactionFailed(error)
        } catch {
            throw Error.invalidFixture("unexpected cache begin error: \(error)")
        }
    }

    internal static func runCacheFinalizeFailureFixtureForTesting() throws {
        var backend = FixtureCacheBackend.dense(layerCount: 1, length: 4, prefix: [1])
        let preDraft = try backend.currentSnapshot()
        let transaction: SpeculativeCacheTransactionSession<FixtureCacheBackend>
        do {
            transaction = try SpeculativeCacheTransaction.begin(
                layout: .dense(layerCount: 1),
                committedInputToken: 900,
                draftTokens: [0],
                backend: &backend)
        } catch let error as SpeculativeCacheTransactionError {
            throw Error.cacheTransactionFailed(error)
        } catch {
            throw Error.invalidFixture("unexpected cache begin error: \(error)")
        }

        backend.appendVerifySpan(committedInputToken: 900, draftTokens: [0])
        do {
            _ = try transaction.finalize(
                nConfirmed: 1,
                scalarEquivalentSnapshot: preDraft.appending(tokens: [900]),
                backend: &backend)
        } catch let error as SpeculativeCacheTransactionError {
            throw Error.cacheTransactionFailed(error)
        } catch {
            throw Error.invalidFixture("unexpected cache finalize error: \(error)")
        }
    }

    private static func sampleBlock(
        model: MarkovFixtureModel,
        rng: inout SplitMix64
    ) throws -> SampledMTPBlockDecision {
        let firstProposal = sample(distribution: model.firstDraft, rng: &rng)
        let secondDraft = model.secondDraft(after: firstProposal)
        let secondProposal = sample(distribution: secondDraft, rng: &rng)
        let steps = [
            SampledMTPBlockStep(
                targetDistribution: model.firstTarget,
                draftDistribution: model.firstDraft,
                proposedToken: firstProposal),
            SampledMTPBlockStep(
                targetDistribution: model.secondTarget(after: firstProposal),
                draftDistribution: secondDraft,
                proposedToken: secondProposal),
        ]

        let firstAcceptanceUniform = uniform(&rng)
        let firstAcceptanceProbability = try SampledMTPResidualCorrection.acceptanceProbability(
            target: model.firstTarget,
            draft: model.firstDraft,
            proposedToken: firstProposal)

        let plan: PlannedBlockInputs
        if firstAcceptanceUniform >= firstAcceptanceProbability {
            plan = PlannedBlockInputs(
                acceptanceUniforms: [firstAcceptanceUniform],
                terminalDraws: [.residual(uniform(&rng))],
                bonusTargetDistribution: model.bonusTarget(after: firstProposal, second: secondProposal),
                expectedOutcome: .rejectFirst)
        } else {
            let secondAcceptanceUniform = uniform(&rng)
            let secondAcceptanceProbability = try SampledMTPResidualCorrection.acceptanceProbability(
                target: model.secondTarget(after: firstProposal),
                draft: secondDraft,
                proposedToken: secondProposal)

            if secondAcceptanceUniform >= secondAcceptanceProbability {
                plan = PlannedBlockInputs(
                    acceptanceUniforms: [firstAcceptanceUniform, secondAcceptanceUniform],
                    terminalDraws: [.residual(uniform(&rng))],
                    bonusTargetDistribution: model.bonusTarget(after: firstProposal, second: secondProposal),
                    expectedOutcome: .rejectSecond)
            } else {
                plan = PlannedBlockInputs(
                    acceptanceUniforms: [firstAcceptanceUniform, secondAcceptanceUniform],
                    terminalDraws: [.bonus(uniform(&rng))],
                    bonusTargetDistribution: model.bonusTarget(after: firstProposal, second: secondProposal),
                    expectedOutcome: .acceptAll)
            }
        }

        let decision = try SampledMTPBlockAcceptance.decide(
            steps: steps,
            acceptanceUniforms: plan.acceptanceUniforms,
            terminalDraws: plan.terminalDraws,
            bonusTargetDistribution: plan.bonusTargetDistribution)
        try validate(decision: decision, matches: plan.expectedOutcome)
        return decision
    }

    private static func validate(
        decision: SampledMTPBlockDecision,
        matches expectedOutcome: AnalyticOutcome
    ) throws {
        switch (expectedOutcome, decision.outcome) {
        case (.rejectFirst, .rejected(stepIndex: 0, correctionToken: _)):
            guard decision.acceptedDraftCount == 0 else {
                throw Error.invalidFixture("reject-first decision reported wrong accepted draft count")
            }
        case (.rejectSecond, .rejected(stepIndex: 1, correctionToken: _)):
            guard decision.acceptedDraftCount == 1 else {
                throw Error.invalidFixture("reject-second decision reported wrong accepted draft count")
            }
        case (.acceptAll, .acceptedAll):
            guard decision.acceptedDraftCount == 2 else {
                throw Error.invalidFixture("accept-all decision reported wrong accepted draft count")
            }
        default:
            throw Error.invalidFixture("block decision outcome did not match planned terminal draw purpose")
        }
    }

    private static func declaredRows(model: MarkovFixtureModel) -> [DistributionRow] {
        var rows: [DistributionRow] = [.firstToken]
        rows.append(contentsOf: model.vocabulary.map { .secondToken(first: $0) })
        for first in model.vocabulary {
            for second in model.vocabulary {
                rows.append(.bonus(first: first, second: second))
            }
        }
        return rows
    }

    private static func rowSortKey(_ left: DistributionRow, _ right: DistributionRow) -> Bool {
        let leftKey = distributionSortKey(left)
        let rightKey = distributionSortKey(right)
        return leftKey.lexicographicallyPrecedes(rightKey)
    }

    private static func distributionSortKey(_ row: DistributionRow) -> [Int] {
        switch row {
        case .firstToken:
            return [0, 0, 0]
        case .secondToken(let first):
            return [1, first, 0]
        case .bonus(let first, let second):
            return [2, first, second]
        }
    }

    private static func cacheChecks() throws -> [CacheCompositionCheck] {
        let hybridMap = try requireHybridMap()
        let dense = SpeculativeCacheLayout.dense(layerCount: 3)
        let hybrid = SpeculativeCacheLayout.hybrid(hybridMap)
        let cases = try forcedCacheBlockCases()
        var checks: [CacheCompositionCheck] = []

        for blockCase in cases {
            checks.append(try cacheCheck(
                layoutEvidence: .dense,
                layout: dense,
                blockCase: blockCase,
                backend: .dense(layerCount: 3, length: 8, prefix: [11, 22])))
            checks.append(try cacheCheck(
                layoutEvidence: .qwenStyleHybrid,
                layout: hybrid,
                blockCase: blockCase,
                backend: .hybrid(map: hybridMap, length: 8, prefix: [11, 22])))
        }

        return checks.sorted {
            (sortKey($0.layout), $0.acceptedDraftCount) < (sortKey($1.layout), $1.acceptedDraftCount)
        }
    }

    private static func cacheCheck(
        layoutEvidence: CacheLayoutEvidence,
        layout: SpeculativeCacheLayout,
        blockCase: ForcedCacheBlockCase,
        backend: FixtureCacheBackend
    ) throws -> CacheCompositionCheck {
        var backend = backend
        let committedInput = 900
        let preDraft = try backend.currentSnapshot()
        let transaction: SpeculativeCacheTransactionSession<FixtureCacheBackend>
        do {
            transaction = try SpeculativeCacheTransaction.begin(
                layout: layout,
                committedInputToken: committedInput,
                draftTokens: blockCase.proposedDraftTokens,
                backend: &backend)
        } catch let error as SpeculativeCacheTransactionError {
            throw Error.cacheTransactionFailed(error)
        } catch {
            throw Error.invalidFixture("unexpected cache begin error: \(error)")
        }
        backend.appendVerifySpan(
            committedInputToken: committedInput,
            draftTokens: blockCase.proposedDraftTokens)

        let acceptedDraftCount = blockCase.decision.acceptedDraftCount
        let committedPrefix = [committedInput] + Array(blockCase.proposedDraftTokens.prefix(acceptedDraftCount))
        guard let terminalOutput = blockCase.decision.tokens.last else {
            throw Error.invalidFixture("cache fixture decision emitted no terminal token")
        }
        guard !committedPrefix.contains(terminalOutput) else {
            throw Error.invalidFixture("cache fixture terminal token overlaps committed prefix")
        }
        let scalar = preDraft.appending(tokens: committedPrefix)
        let result: SpeculativeCacheTransactionResult<FixtureCacheSnapshot>
        do {
            result = try transaction.finalize(
                nConfirmed: acceptedDraftCount,
                scalarEquivalentSnapshot: scalar,
                backend: &backend)
        } catch let error as SpeculativeCacheTransactionError {
            throw Error.cacheTransactionFailed(error)
        } catch {
            throw Error.invalidFixture("unexpected cache transaction error: \(error)")
        }

        let terminalCommitted = result.finalSnapshot.contains(token: terminalOutput)
        return CacheCompositionCheck(
            layout: layoutEvidence,
            outcome: blockCase.outcome,
            acceptedDraftCount: acceptedDraftCount,
            proposedDraftTokens: blockCase.proposedDraftTokens,
            expectedCommittedInputTokens: committedPrefix,
            committedInputTokens: result.committedInputTokens,
            expectedRejectedDraftCount: blockCase.proposedDraftTokens.count - acceptedDraftCount,
            rejectedDraftCount: result.rejectedDraftCount,
            emittedTerminalToken: terminalOutput,
            terminalOutputWasCommitted: terminalCommitted,
            finalSnapshot: result.finalSnapshot,
            scalarEquivalentSnapshot: scalar)
    }

    private static func requireHybridMap() throws -> HybridLayerKindMap {
        guard let map = HybridLayerKindMap.qwen35(layerCount: 4, fullAttentionInterval: 2) else {
            throw Error.invalidFixture("qwen-style hybrid fixture map could not be built")
        }
        return map
    }

    private static func sortKey(_ layout: CacheLayoutEvidence) -> Int {
        switch layout {
        case .dense:
            return 0
        case .qwenStyleHybrid:
            return 1
        }
    }

    private static func forcedCacheBlockCases() throws -> [ForcedCacheBlockCase] {
        let terminalDistribution = sparseDistribution(count: 9, masses: [
            (0, 0.20),
            (1, 0.30),
            (8, 0.50),
        ])
        let rejectFirstSteps = [
            SampledMTPBlockStep(
                targetDistribution: sparseDistribution(count: 9, masses: [
                    (0, 0.20),
                    (1, 0.30),
                    (8, 0.50),
                ]),
                draftDistribution: sparseDistribution(count: 9, masses: [
                    (0, 0.70),
                    (1, 0.20),
                    (8, 0.10),
                ]),
                proposedToken: 0),
            SampledMTPBlockStep(
                targetDistribution: sparseDistribution(count: 9, masses: [
                    (0, 0.30),
                    (1, 0.40),
                    (8, 0.30),
                ]),
                draftDistribution: sparseDistribution(count: 9, masses: [
                    (0, 0.30),
                    (1, 0.40),
                    (8, 0.30),
                ]),
                proposedToken: 1),
        ]
        let rejectFirst = try SampledMTPBlockAcceptance.decide(
            steps: rejectFirstSteps,
            acceptanceUniforms: [0.90],
            terminalDraws: [.residual(0.50)],
            bonusTargetDistribution: terminalDistribution)

        let rejectSecondSteps = [
            SampledMTPBlockStep(
                targetDistribution: sparseDistribution(count: 9, masses: [
                    (0, 0.50),
                    (1, 0.30),
                    (8, 0.20),
                ]),
                draftDistribution: sparseDistribution(count: 9, masses: [
                    (0, 0.40),
                    (1, 0.40),
                    (8, 0.20),
                ]),
                proposedToken: 0),
            SampledMTPBlockStep(
                targetDistribution: sparseDistribution(count: 9, masses: [
                    (0, 0.10),
                    (1, 0.20),
                    (8, 0.70),
                ]),
                draftDistribution: sparseDistribution(count: 9, masses: [
                    (0, 0.20),
                    (1, 0.70),
                    (8, 0.10),
                ]),
                proposedToken: 1),
        ]
        let rejectSecond = try SampledMTPBlockAcceptance.decide(
            steps: rejectSecondSteps,
            acceptanceUniforms: [0.50, 0.80],
            terminalDraws: [.residual(0.00)],
            bonusTargetDistribution: terminalDistribution)

        let acceptAllSteps = [
            SampledMTPBlockStep(
                targetDistribution: sparseDistribution(count: 9, masses: [
                    (0, 0.60),
                    (1, 0.20),
                    (8, 0.20),
                ]),
                draftDistribution: sparseDistribution(count: 9, masses: [
                    (0, 0.30),
                    (1, 0.40),
                    (8, 0.30),
                ]),
                proposedToken: 0),
            SampledMTPBlockStep(
                targetDistribution: sparseDistribution(count: 9, masses: [
                    (0, 0.20),
                    (1, 0.60),
                    (8, 0.20),
                ]),
                draftDistribution: sparseDistribution(count: 9, masses: [
                    (0, 0.30),
                    (1, 0.30),
                    (8, 0.40),
                ]),
                proposedToken: 1),
        ]
        let acceptAll = try SampledMTPBlockAcceptance.decide(
            steps: acceptAllSteps,
            acceptanceUniforms: [0.50, 0.50],
            terminalDraws: [.bonus(0.60)],
            bonusTargetDistribution: terminalDistribution)

        let cases = [
            ForcedCacheBlockCase(
                outcome: .rejectFirst,
                proposedDraftTokens: rejectFirstSteps.map(\.proposedToken),
                decision: rejectFirst),
            ForcedCacheBlockCase(
                outcome: .rejectSecond,
                proposedDraftTokens: rejectSecondSteps.map(\.proposedToken),
                decision: rejectSecond),
            ForcedCacheBlockCase(
                outcome: .acceptAll,
                proposedDraftTokens: acceptAllSteps.map(\.proposedToken),
                decision: acceptAll),
        ]

        for blockCase in cases {
            try validate(decision: blockCase.decision, matches: blockCase.outcome)
            guard blockCase.decision.tokens.last == 8 else {
                throw Error.invalidFixture("forced cache decision terminal token must be token 8")
            }
        }
        return cases
    }

    private static func sparseDistribution(count: Int, masses: [(Int, Double)]) -> [Double] {
        var distribution = Array(repeating: 0.0, count: count)
        for (token, mass) in masses {
            distribution[token] = mass
        }
        return distribution
    }

    private static func reconstructBonusWithActualAcceptedBlock(
        first: Int,
        second: Int,
        distribution: [Double],
        vocabularyCount: Int
    ) throws -> [Double] {
        var reconstructed = Array(repeating: 0.0, count: vocabularyCount)
        var lowerBound = 0.0
        for (token, probability) in distribution.enumerated() {
            let upperBound = lowerBound + probability
            if probability > 0.0 {
                let uniform = lowerBound + probability / 2.0
                let decision = try SampledMTPBlockAcceptance.decide(
                    steps: [
                        .init(
                            targetDistribution: oneHot(token: first, count: vocabularyCount),
                            draftDistribution: oneHot(token: first, count: vocabularyCount),
                            proposedToken: first),
                        .init(
                            targetDistribution: oneHot(token: second, count: vocabularyCount),
                            draftDistribution: oneHot(token: second, count: vocabularyCount),
                            proposedToken: second),
                    ],
                    acceptanceUniforms: [0.0, 0.0],
                    terminalDraws: [.bonus(uniform)],
                    bonusTargetDistribution: distribution)
                try validate(decision: decision, matches: .acceptAll)
                guard decision.tokens == [first, second, token] else {
                    throw Error.invalidFixture("accept-all CDF reconstruction emitted unexpected token")
                }
                reconstructed[token] += upperBound - lowerBound
            }
            lowerBound = upperBound
        }
        guard abs(lowerBound - 1.0) <= 1e-12 else {
            throw Error.invalidFixture("bonus distribution is not normalized")
        }
        return reconstructed
    }

    private static func oneHot(token: Int, count: Int) -> [Double] {
        (0..<count).map { $0 == token ? 1.0 : 0.0 }
    }

    private static func acceptanceProbabilities(target: [Double], draft: [Double]) throws -> [Double] {
        try target.indices.map { token in
            try SampledMTPResidualCorrection.acceptanceProbability(
                target: target,
                draft: draft,
                proposedToken: token)
        }
    }

    private static func rejectedMass(draft: [Double], acceptance: [Double]) -> Double {
        zip(draft, acceptance).reduce(0.0) { partial, pair in
            partial + pair.0 * (1.0 - pair.1)
        }
    }

    private static func targetSampleCounts(
        distribution: [Double],
        sampleCount: Int,
        rng: inout SplitMix64
    ) -> [Int] {
        var counts = zeroCounts(distribution.count)
        for _ in 0..<sampleCount {
            counts[sample(distribution: distribution, rng: &rng)] += 1
        }
        return counts
    }

    private static func zeroCounts(_ count: Int) -> [Int] {
        Array(repeating: 0, count: count)
    }

    private static func frequencies(_ counts: [Int]) -> [Double] {
        let total = counts.reduce(0, +)
        guard total > 0 else {
            return Array(repeating: 0.0, count: counts.count)
        }
        return counts.map { Double($0) / Double(total) }
    }

    private static func maxAbsoluteDelta(_ actual: [Double], _ expected: [Double]) -> Double {
        zip(actual, expected).reduce(0.0) { partial, pair in
            max(partial, abs(pair.0 - pair.1))
        }
    }

    private static func sample(distribution: [Double], rng: inout SplitMix64) -> Int {
        let draw = uniform(&rng)
        var cumulative = 0.0
        var lastSupportedToken = 0
        for (token, probability) in distribution.enumerated() {
            if probability > 0.0 {
                lastSupportedToken = token
            }
            cumulative += probability
            if draw < cumulative {
                return token
            }
        }
        return lastSupportedToken
    }

    private static func uniform(_ rng: inout SplitMix64) -> Double {
        Double(rng.next() >> 11) * (1.0 / 9_007_199_254_740_992.0)
    }
}

private struct PlannedBlockInputs: Sendable {
    let acceptanceUniforms: [Double]
    let terminalDraws: [SampledMTPBlockTerminalDraw]
    let bonusTargetDistribution: [Double]
    let expectedOutcome: SampledMTPBlockDistributionCacheGate.AnalyticOutcome
}

private struct ForcedCacheBlockCase: Sendable {
    let outcome: SampledMTPBlockDistributionCacheGate.AnalyticOutcome
    let proposedDraftTokens: [Int]
    let decision: SampledMTPBlockDecision
}

private struct MarkovFixtureModel: Sendable {
    let firstTarget = [0.42, 0.35, 0.23]
    let firstDraft = [0.25, 0.50, 0.25]
    let secondTargets = [
        [0.55, 0.20, 0.25],
        [0.15, 0.65, 0.20],
        [0.25, 0.25, 0.50],
    ]
    let secondDrafts = [
        [0.30, 0.45, 0.25],
        [0.45, 0.30, 0.25],
        [0.25, 0.50, 0.25],
    ]
    let bonusTargets = [
        [
            [0.50, 0.30, 0.20],
            [0.20, 0.55, 0.25],
            [0.25, 0.25, 0.50],
        ],
        [
            [0.60, 0.15, 0.25],
            [0.10, 0.70, 0.20],
            [0.30, 0.20, 0.50],
        ],
        [
            [0.45, 0.25, 0.30],
            [0.20, 0.35, 0.45],
            [0.15, 0.30, 0.55],
        ],
    ]

    var vocabularyCount: Int {
        firstTarget.count
    }

    var vocabulary: Range<Int> {
        0..<vocabularyCount
    }

    func secondTarget(after first: Int) -> [Double] {
        secondTargets[first]
    }

    func secondDraft(after first: Int) -> [Double] {
        secondDrafts[first]
    }

    func bonusTarget(after first: Int, second: Int) -> [Double] {
        bonusTargets[first][second]
    }

    func targetDistribution(for row: SampledMTPBlockDistributionCacheGate.DistributionRow) -> [Double] {
        switch row {
        case .firstToken:
            return firstTarget
        case .secondToken(let first):
            return secondTarget(after: first)
        case .bonus(let first, let second):
            return bonusTarget(after: first, second: second)
        }
    }
}

private struct SplitMix64: RandomNumberGenerator, Sendable {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9e37_79b9_7f4a_7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58_476d_1ce4_e5b9
        z = (z ^ (z >> 27)) &* 0x94d0_49bb_1331_11eb
        return z ^ (z >> 31)
    }
}

private struct FixtureCacheBackend: SpeculativeCacheTransactionBackend {
    var snapshot: SampledMTPBlockDistributionCacheGate.FixtureCacheSnapshot

    static func dense(layerCount: Int, length: Int, prefix: [Int]) -> FixtureCacheBackend {
        let kinds = Array(repeating: LayerCacheKind.denseAttention, count: layerCount)
        return FixtureCacheBackend(snapshot: .init(
            kinds: kinds,
            tokenLengthsByLayer: Array(repeating: length, count: layerCount),
            storedTokensByLayer: Array(repeating: prefix, count: layerCount),
            recurrentStateByLayer: Array(repeating: [], count: layerCount)))
    }

    static func hybrid(map: HybridLayerKindMap, length: Int, prefix: [Int]) -> FixtureCacheBackend {
        FixtureCacheBackend(snapshot: .init(
            kinds: map.kinds,
            tokenLengthsByLayer: Array(repeating: length, count: map.layerCount),
            storedTokensByLayer: map.kinds.map { $0 == .denseAttention ? prefix : [] },
            recurrentStateByLayer: map.kinds.enumerated().map { index, kind in
                kind == .recurrentState ? prefix.map { $0 + index } : []
            }))
    }

    mutating func currentSnapshot() throws -> SampledMTPBlockDistributionCacheGate.FixtureCacheSnapshot {
        snapshot
    }

    mutating func restore(_ snapshot: SampledMTPBlockDistributionCacheGate.FixtureCacheSnapshot) throws {
        self.snapshot = snapshot
    }

    mutating func rollbackDenseSuffix(tokenCount: Int) throws {
        for layer in snapshot.kinds.indices where snapshot.kinds[layer] == .denseAttention {
            snapshot.tokenLengthsByLayer[layer] -= tokenCount
            snapshot.storedTokensByLayer[layer].removeLast(tokenCount)
        }
    }

    mutating func replayCommittedInputTokens(_ tokens: [Int]) throws {
        snapshot.append(tokens: tokens)
    }

    mutating func appendVerifySpan(committedInputToken: Int, draftTokens: [Int]) {
        snapshot.append(tokens: [committedInputToken] + draftTokens)
    }
}

private extension SampledMTPBlockDistributionCacheGate.FixtureCacheSnapshot {
    func contains(token: Int) -> Bool {
        for layer in kinds.indices {
            switch kinds[layer] {
            case .denseAttention:
                if storedTokensByLayer[layer].contains(token) {
                    return true
                }
            case .recurrentState:
                if recurrentStateByLayer[layer].contains(token + layer) {
                    return true
                }
            }
        }
        return false
    }
}
