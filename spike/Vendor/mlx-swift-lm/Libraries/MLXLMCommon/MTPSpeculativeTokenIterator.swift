// Copyright © 2026 Apple Inc.

import Foundation
import MLX

private enum SampledMTPBlockRuntimeDecisionValidationError: Error {
    case invalidDecision
}

/// Generator of tokens using MTP (Multi-Token Prediction) speculative
/// decoding.
///
/// Parallels ``SpeculativeTokenIterator`` but for Gemma 4 - style drafters
/// that share K/V with the target model and produce K - 1 candidate tokens
/// per round in a single ``MTPDrafterModel/draftBlock(target:lastToken:lastHidden:sharedKV:positionDeltas:queryOffset:blockSize:sampler:)`` call (rather
/// than K sequential single-token calls). Every per-round input —
/// `lastToken`, `lastHidden`, `sharedKV`, `positionIds` — is threaded as a
/// method argument, with the target's last hidden state and per-`layer_type`
/// shared K/V extracted from the ``LMOutput/State`` emitted by the target on
/// the previous main-model call.
/// If the drafter needs its own KV cache (Qwen MTP), that cache is owned by
/// this iterator, prefilled over the shifted prompt, and reconciled against
/// accepted target tokens after every verify pass; it is never stored on the
/// shared drafter model.
///
/// The iterator pre-populates each main-model call's incoming `state` with
/// ``mtpEmitFlagKey`` set to `true`, opting the target into populating
/// ``mtpLastHiddenStatesKey`` and ``mtpSharedKVStatesKey`` on its returned
/// ``LMOutput/state``. If the target ever returns nil or partial state
/// (for example when a shared-KV drafter can no longer read regular K/V), the
/// iterator transparently switches into a
/// single-token "passthrough" mode for the remainder of generation — a
/// mid-generation capability loss never crashes or corrupts the stream.
///
/// Port of `_speculative_walk` from mlx-vlm/generate.py at SHA `d49d428`,
/// with no-mutation-during-eval idioms (state is threaded through method
/// args; drafter holds no target-derived state — the target and optional
/// per-stream drafter cache are passed as parameters to `draftBlock(...)` so
/// drafter instances are safe to share across iterators).
public struct MTPSpeculativeTokenIterator: TokenIteratorProtocol {

    var y: LMInput.Text

    let mainModel: any LanguageModel
    let drafter: any MTPDrafterModel

    var mainState: LMOutput.State?
    var mainCache: [KVCache]
    var drafterState: MTPDrafterState?
    let quantizeKVCache: (inout [KVCache]) -> Void
    let collectPhaseTelemetry: Bool
    let promptPreparationEvaluationOrder: MTPPromptPreparationEvaluationOrder

    var processor: LogitProcessor?
    let sampler: LogitSampler
    let drafterSampler: LogitSampler
    let sampledBlockDecisionProvider: (any SampledMTPBlockRuntimeDeciding)?

    public var tokenCount: Int { telemetry.emittedTokenCount }
    public let maxTokens: Int?
    /// Total tokens proposed per round (`blockSize - 1` drafted, plus the
    /// bonus token from the previous verify). Mirrors mlx-vlm's
    /// `draft_block_size` parameter.
    public let blockSize: Int

    private var pendingTokens = [Int]()
    private var pendingIndex = 0
    /// Number of pending tokens already represented by `mainCache`. The last
    /// verifier sample is not committed until a later forward pass.
    private var committedPendingTokenCount = 0
    /// Committed pending tokens actually retained by the caller. A stop token
    /// returned by `next()` and then rejected through `discardGeneratedToken()`
    /// must not count as emitted when finalization reconciles the cache.
    private var emittedCommittedPendingTokenCount = 0
    private var lastReturnedTokenWasCommittedPending = false
    /// The verifier's final bonus/correction is sampled but is not represented
    /// in the target cache until the next target forward. If generation ends
    /// immediately after returning it, finalization must commit that retained
    /// token just as ``TokenIterator.next()`` does before returning a token.
    private var lastReturnedTokenNeedsFinalCommit = false

    /// Set to `true` when the iterator detects that the target can no
    /// longer emit drafter state (typically due to KV cache quantization
    /// converting `Gemma4SharedKVState.regular` to `.quantized`). Once set,
    /// `next()` runs single-token generation against the main model only —
    /// no further `speculateRound` calls. Sticky: never reverts to `false`.
    private var passthrough = false
    private var passthroughLoggedOnce = false

    /// Verify-position index in the prior round's emitted hidden that
    /// produced the newly-accepted bonus's logit prediction. Set at the end
    /// of each `speculateRound()`. `nil` on the first round means slice the
    /// last position (round 1's `lastHidden` has shape `[B, 1, hidden]`, so
    /// last-position == only-position == the correct slot). Round 2+ slices
    /// at this index, mirroring mlx-lm's `verify.hidden[:, accepted : accepted + 1, :]`.
    /// Mismatch (e.g. unconditional last-position) is silent: drafter still
    /// produces tokens, but they're conditioned on the wrong slot → less
    /// coherent drafts → lower acceptance, especially at higher blockSize.
    private var lastRoundAccepted: Int? = nil

    public var promptPrefillTime: TimeInterval = 0.0
    private var telemetry = SpeculativeDecodingTelemetry()
    public var speculativeDecodingTelemetry: SpeculativeDecodingTelemetry? {
        telemetry.roundCount > 0 ? telemetry : nil
    }
    public var speculativeDecodingPhaseTelemetry: SpeculativeDecodingPhaseTelemetry {
        telemetry.phases
    }
    public private(set) var promptPreparationTelemetry: MTPPromptPreparationTelemetry?
    public private(set) var promptPreparationPhaseBoundarySynchronizationSeconds = 0.0

    @inline(__always)
    private func phaseClock() -> TimeInterval {
        collectPhaseTelemetry ? ProcessInfo.processInfo.systemUptime : 0
    }

    public mutating func discardGeneratedToken() {
        telemetry.discardGeneratedToken()
        lastReturnedTokenNeedsFinalCommit = false
        if lastReturnedTokenWasCommittedPending {
            emittedCommittedPendingTokenCount -= 1
            lastReturnedTokenWasCommittedPending = false
        }
    }

    // Optional instrumentation used by acceptance-rate floor tests.
    // Public read-only so test cases can compute `acceptedCount /
    // proposedCount` after the stream drains.
    public private(set) var acceptedCount: Int = 0
    public private(set) var proposedCount: Int = 0
    private var verifierTokenReadbackCount = 0

    // Reason recorded the first time sticky-passthrough engaged, or nil if
    // the iterator stayed speculative for the full stream. Surfaced through
    // ``MTPStatsCollecting`` so `generateLoopTask` can include it on the
    // emitted `.info` event.
    public private(set) var passthroughReason: String?

    public init(
        input: LMInput,
        mainModel: any LanguageModel,
        drafter: any MTPDrafterModel,
        mainCache: [KVCache]? = nil,
        parameters: GenerateParameters,
        blockSize: Int,
        collectPhaseTelemetry: Bool = false,
        promptPreparationEvaluationOrder: MTPPromptPreparationEvaluationOrder = .cacheFirst,
        sampledBlockDecisionProvider: (any SampledMTPBlockRuntimeDeciding)? = nil
    ) throws {
        precondition(
            blockSize >= 2,
            "MTPSpeculativeTokenIterator requires blockSize >= 2 (1 bonus + K-1 drafted)")

        self.y = input.text
        self.mainModel = mainModel
        self.drafter = drafter
        self.collectPhaseTelemetry = collectPhaseTelemetry
        self.promptPreparationEvaluationOrder = promptPreparationEvaluationOrder
        guard collectPhaseTelemetry || promptPreparationEvaluationOrder == .cacheFirst else {
            throw MTPPromptPreparationEvaluationOrderError.telemetryRequired
        }

        let providerIsEligible = parameters.temperature != 0
            && parameters.processor() == nil
            && (sampledBlockDecisionProvider?.supports(parameters: parameters) ?? false)
        let initialPassthroughReason: String?
        if !drafter.supportsSpeculation(for: input) {
            initialPassthroughReason = "drafter does not support this prompt input"
        } else if drafter.requiresGreedySampling, parameters.temperature != 0,
            !providerIsEligible
        {
            initialPassthroughReason =
                "Qwen MTP currently requires temperature == 0; generating without speculation"
        } else {
            initialPassthroughReason = nil
        }

        self.mainCache = mainCache ?? mainModel.newCache(parameters: parameters)
        self.drafterState = initialPassthroughReason == nil
            ? (drafter as? any StatefulMTPDrafterModel)?.makeState(parameters: parameters)
            : nil

        let drafterBlockSize = Swift.min(blockSize, drafter.maximumBlockSize ?? blockSize)
        let nativeRewindDepth =
            (mainModel as? any SpeculativeCacheRewindModel)?
            .maximumNativeTargetCacheRewind ?? 0
        let usesNativeHybridRewind =
            nativeRewindDepth >= drafterBlockSize - 1
            && self.mainCache.contains { $0 is MambaCache }
            && self.mainCache.allSatisfy { $0.isTrimmable || $0 is MambaCache }
        guard canTrimPromptCache(self.mainCache) || usesNativeHybridRewind else {
            throw KVCacheError(
                message: "MTP speculative decoding requires a trimmable main KV cache.")
        }

        self.sampler = parameters.sampler()
        self.sampledBlockDecisionProvider = providerIsEligible
            ? sampledBlockDecisionProvider : nil
        self.drafterSampler = providerIsEligible
            ? sampledBlockDecisionProvider!.proposalSampler : self.sampler
        self.processor = parameters.processor()

        self.maxTokens = parameters.maxTokens
        self.blockSize = drafterBlockSize

        self.quantizeKVCache = { cache in
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: parameters.kvBits,
                kvGroupSize: parameters.kvGroupSize,
                quantizedKVStart: parameters.quantizedKVStart
            )
        }

        if let initialPassthroughReason {
            switchToPassthrough(reason: initialPassthroughReason)
        }

        let prefillStart = ProcessInfo.processInfo.systemUptime
        try prepare(input: input, windowSize: parameters.prefillStepSize)
        self.promptPrefillTime = ProcessInfo.processInfo.systemUptime - prefillStart
    }

    /// Prefill the main model with the prompt and, for stateful drafters, prime
    /// iterator-owned private cache only after all fail-closed gates pass.
    mutating func prepare(input: LMInput, windowSize: Int? = nil) throws {
        let targetPrefillStart = phaseClock()
        processor?.prompt(input.text.tokens)

        var prefillState = LMOutput.State()
        var committedReprimeToken: MLXArray?
        var committedReprimeHidden: MLXArray?
        var capturedPromptHidden: MLXArray?
        prefillState[mtpEmitFlagKey] = !passthrough
        // Note: `prepare(_:cache:windowSize:)` does not currently thread
        // state through. To prime drafter state we run an explicit follow-up
        // forward call after prefill (one position, the bonus token).

        let prepareResult: PrepareResult
        let targetPromptPreparation: MTPPromptPreparation?
        if !passthrough, drafter.requiresPromptPrefill, collectPhaseTelemetry,
            let telemetryTarget = mainModel
                as? any MTPPromptHiddenStateEvaluationOrderPreparingModel
        {
            targetPromptPreparation = try telemetryTarget.prepareForMTP(
                input, cache: mainCache, windowSize: windowSize,
                collectTelemetry: true,
                evaluationOrder: promptPreparationEvaluationOrder)
        } else if !passthrough, drafter.requiresPromptPrefill, collectPhaseTelemetry,
            promptPreparationEvaluationOrder != .cacheFirst
        {
            throw MTPPromptPreparationEvaluationOrderError.unsupportedTarget
        } else if !passthrough, drafter.requiresPromptPrefill, collectPhaseTelemetry,
            let telemetryTarget = mainModel
                as? any MTPPromptHiddenStateTelemetryPreparingModel
        {
            targetPromptPreparation = try telemetryTarget.prepareForMTP(
                input, cache: mainCache, windowSize: windowSize,
                collectTelemetry: true)
        } else if !passthrough, drafter.requiresPromptPrefill,
            let reuseTarget = mainModel
            as? any MTPPromptHiddenStatePreparingModel
        {
            targetPromptPreparation = try reuseTarget.prepareForMTP(
                input, cache: mainCache, windowSize: windowSize)
        } else {
            targetPromptPreparation = nil
        }
        if !passthrough, drafter.requiresPromptPrefill,
            let preparation = targetPromptPreparation
        {
            prepareResult = preparation.result
            capturedPromptHidden = preparation.targetHidden
            promptPreparationTelemetry = preparation.telemetry
        } else {
            prepareResult = try mainModel.prepare(
                input, cache: mainCache, windowSize: windowSize)
        }

        switch prepareResult {
        case .tokens(let tokens):
            y = tokens
            // Final prompt position not yet evaluated -- run one forward to
            // produce the bonus token AND prime drafter state.
            let result = mainModel(y[text: .newAxis], cache: mainCache, state: prefillState)
            var logits = result.logits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            processor?.didSample(token: token)
            y = .init(tokens: token)
            mainState = result.state
            // Yield the bonus to the iterator's consumer. Without this,
            // the iterator silently starts 1 position ahead of an
            // equivalent autoregressive run, violating speculative
            // decoding's bit-exact-equivalence-to-greedy guarantee.
            pendingTokens.append(token.item(Int.self))
        case .logits(let prefillResult):
            // Some `prepare` implementations evaluate the final position
            // themselves and return logits directly; their `state` here may
            // or may not carry drafter state depending on whether the model
            // override threads it.
            var logits = prefillResult.logits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            processor?.didSample(token: token)
            y = .init(tokens: token)
            mainState = prefillResult.state

            // If prefill didn't emit drafter state, do one more forward call
            // with the just-sampled bonus token to prime the state. The cost
            // is one extra token's forward pass; acceptable.
            if !passthrough
                && (mainState?[mtpLastHiddenStatesKey] == nil
                    || mainState?[mtpSharedKVStatesKey] == nil)
            {
                let checkpointReady =
                    checkpointSpeculativePromptCacheForOneTokenReprime(mainCache)
                if !checkpointReady {
                    // A recurrent cache without both state arrays cannot be
                    // rolled back exactly. Keep the sampled bonus as the next
                    // scalar input and never mutate the cache speculatively.
                    switchToPassthrough(
                        reason: "hybrid cache cannot checkpoint one-token MTP re-prime")
                    pendingTokens.append(token.item(Int.self))
                } else {
                    // Preserve model-specific continuation state produced by
                    // prepare (for example Qwen VLM RoPE deltas) while opting
                    // this one-token re-prime into MTP state emission.
                    var reprimeState = mainState ?? LMOutput.State()
                    reprimeState[mtpEmitFlagKey] = true
                    let primed = mainModel(
                        y[text: .newAxis], cache: mainCache, state: reprimeState)
                    mainState = primed.state
                    committedReprimeToken = token
                    committedReprimeHidden = primed.state?[mtpLastHiddenStatesKey]
                    // Resample bonus from this forward's logits so the chain stays
                    // coherent at this position (the cache offset moves by 1, so
                    // we must re-pick the bonus from the new step's logits).
                    var newLogits = primed.logits[0..., -1, 0...]
                    newLogits = processor?.process(logits: newLogits) ?? newLogits
                    let newToken = sampler.sample(logits: newLogits)
                    processor?.didSample(token: newToken)
                    y = .init(tokens: newToken)
                    // Yield BOTH bonuses to the consumer, in sample order.
                    // `token` is the prefill-position-N sample (consumed by the
                    // re-prime forward, now committed in cache); `newToken` is
                    // the prefill-position-N+1 sample that becomes the input
                    // to the first speculateRound.
                    pendingTokens.append(token.item(Int.self))
                    pendingTokens.append(newToken.item(Int.self))
                    // The re-prime forward consumed only `token`; `newToken`
                    // remains the uncommitted input for the next round.
                    committedPendingTokenCount = 1
                }
            } else {
                // Prefill state already carried drafter keys; the single
                // bonus is the input to the first speculateRound.
                pendingTokens.append(token.item(Int.self))
            }
        }

        if collectPhaseTelemetry {
            promptPreparationPhaseBoundarySynchronizationSeconds += synchronizeTargetPhaseBoundary(
                state: mainState, capturedPromptHidden: capturedPromptHidden)
            telemetry.recordTargetPrefill(
                seconds: ProcessInfo.processInfo.systemUptime - targetPrefillStart)
        }

        let shouldMeasureDrafterPriming =
            collectPhaseTelemetry && !passthrough && drafter.requiresPromptPrefill
        let drafterPrimingStart = phaseClock()
        if !passthrough, drafter.requiresPromptPrefill,
            let statefulDrafter = drafter as? any StatefulMTPDrafterModel,
            var currentDrafterState = drafterState
        {
            let normalizedPromptTokens = normalizedMTPTokenBatch(input.text.tokens)
            let normalizedPrompt = LMInput.Text(
                tokens: normalizedPromptTokens,
                mask: input.text.mask.map { normalizedMTPTokenBatch($0) })
            let reusablePromptHidden = capturedPromptHidden.flatMap { hidden in
                hidden.ndim == 3
                    && hidden.dim(0) == normalizedPromptTokens.dim(0)
                    && hidden.dim(1) == normalizedPromptTokens.dim(1)
                    ? hidden : nil
            }
            let baseTargetHidden = reusablePromptHidden
                ?? mainModel(normalizedPrompt, cache: nil, state: prefillState)
                    .state?[mtpLastHiddenStatesKey]
            if let baseTargetHidden,
                committedReprimeToken == nil || committedReprimeHidden != nil
            {
                var promptTokens = normalizedPromptTokens
                var targetHidden = baseTargetHidden
                if let committedReprimeToken, let committedReprimeHidden {
                    // `.logits` prepare already consumed the original prompt.
                    // The follow-up re-prime commits its first sampled token
                    // to the target cache before `y` advances to the next
                    // bonus. Prime the private drafter over that same extended
                    // sequence so its shifted token/hidden pairs and absolute
                    // position match the target before the first verify.
                    promptTokens = concatenated(
                        [promptTokens, normalizedMTPColumn(committedReprimeToken)],
                        axis: 1)
                    targetHidden = concatenated(
                        [targetHidden, committedReprimeHidden], axis: 1)
                }
                statefulDrafter.prepareDrafterState(
                    target: mainModel,
                    promptTokens: promptTokens,
                    targetHidden: targetHidden,
                    firstBonus: y.tokens,
                    positionDeltas: mainState?[mtpPositionDeltasKey],
                    state: &currentDrafterState,
                    sampler: drafterSampler)
                drafterState = currentDrafterState
            } else {
                switchToPassthrough(
                    reason: "target did not emit drafter state for Qwen MTP prompt prefill")
            }
        } else if !passthrough, drafter.requiresPromptPrefill {
            switchToPassthrough(
                reason: "target did not emit drafter state for Qwen MTP prompt prefill")
        }
        if shouldMeasureDrafterPriming {
            synchronizeDrafterPhaseBoundary()
            telemetry.recordDrafterPromptPriming(
                seconds: ProcessInfo.processInfo.systemUptime - drafterPrimingStart)
        }
    }

    @discardableResult
    private mutating func synchronizeTargetPhaseBoundary(
        state: LMOutput.State?,
        capturedPromptHidden: MLXArray? = nil
    ) -> TimeInterval {
        guard collectPhaseTelemetry else { return 0 }
        var arrays = mainCache.flatMap { $0.state }
        if let capturedPromptHidden {
            arrays.append(capturedPromptHidden)
        }
        if let hidden = state?[mtpLastHiddenStatesKey] {
            arrays.append(hidden)
        }
        if let sharedKV = state?[mtpSharedKVStatesKey] {
            for value in sharedKV.values {
                arrays.append(value.0)
                arrays.append(value.1)
            }
        }
        if let positionDeltas = state?[mtpPositionDeltasKey] {
            arrays.append(positionDeltas)
        }
        if !arrays.isEmpty {
            let start = ProcessInfo.processInfo.systemUptime
            eval(arrays)
            return ProcessInfo.processInfo.systemUptime - start
        }
        return 0
    }

    private func synchronizeDrafterPhaseBoundary() {
        guard collectPhaseTelemetry, let drafterState else { return }
        var arrays = drafterState.cache.flatMap { $0.state }
        if let seedToken = drafterState.seedToken {
            arrays.append(seedToken)
        }
        if let seedHidden = drafterState.seedHidden {
            arrays.append(seedHidden)
        }
        if !arrays.isEmpty {
            eval(arrays)
        }
    }

    /// Single round: draft `blockSize - 1` tokens, verify with main, accept
    /// the longest matching prefix, emit the bonus correction.
    mutating func speculateRound() {
        guard !passthrough else { return }
        // A prior all-accepted round may keep one recurrent checkpoint until
        // its pending output is drained so early finalization can rewind it.
        discardSpeculativePromptCacheCheckpoints(mainCache)

        // A speculative round can emit up to `numDraft + 1` tokens: the
        // accepted draft prefix plus the verifier's correction/bonus token.
        // Keep the whole pending buffer within the remaining output budget.
        let numDraft: Int
        if let maxTokens {
            let remaining = maxTokens - tokenCount
            guard remaining > 0 else { return }

            let draftBudget = Swift.min(remaining - 1, blockSize - 1)
            guard draftBudget > 0 else {
                if let token = passthroughStep() {
                    pendingTokens.append(token)
                }
                return
            }
            numDraft = draftBudget
        } else {
            numDraft = blockSize - 1
        }

        guard canPreserveSpeculativeRewindAfterAppend(
            mainCache, tokenCount: numDraft + 1
        ) else {
            switchToPassthrough(
                reason: "main cache cannot preserve rollback through MTP verify")
            return
        }

        if mainCache.contains(where: { $0 is MambaCache }) {
            let supportsNativeHybridRewind =
                ((mainModel as? any SpeculativeCacheRewindModel)?
                    .maximumNativeTargetCacheRewind ?? 0) >= numDraft
                && mainCache.allSatisfy { $0.isTrimmable || $0 is MambaCache }
            guard supportsNativeHybridRewind else {
                switchToPassthrough(
                    reason: "hybrid cache no longer supports native speculative rewind")
                return
            }
        }

        guard
            let state = mainState,
            let lastHidden = state[mtpLastHiddenStatesKey]
        else {
            switchToPassthrough(reason: "main model did not emit drafter state")
            return
        }
        let sharedKV = state[mtpSharedKVStatesKey] ?? [:]
        if drafter.requiresSharedTargetKV, sharedKV["full_attention"] == nil {
            switchToPassthrough(reason: "main model did not emit shared target K/V")
            return
        }

        // Slice the hidden at the slot that produced the newly-accepted
        // bonus's prediction. Round 1: last (and only) position. Round 2+:
        // index `lastRoundAccepted`, matching mlx-lm's
        // `verify.hidden[:, accepted : accepted + 1, :]` semantic.
        let bonusSlotHidden: MLXArray
        if let idx = lastRoundAccepted {
            bonusSlotHidden = lastHidden[0..., idx ..< (idx + 1), 0...]
        } else {
            bonusSlotHidden = lastHidden[0..., (-1)..., 0...]
        }

        let cacheOffset =
            state[mtpSharedKVOffsetsKey]?["full_attention"]
            ?? mainCache.first?.offset ?? 0

        // Invariant: the span the drafter attends over describes exactly the
        // true sequence — the rewind site trims the emitted snapshot in
        // lockstep with the cache. `dim()` is shape metadata (no eval, no GPU
        // sync). The check stands down if the cache ever leaves the trimmable
        // regime (post-wrap sliding window), where the rewind machinery
        // itself no-ops.
        assert(
            sharedKV.allSatisfy { $0.value.0.dim(-2) == cacheOffset }
                || !canTrimPromptCache(mainCache),
            "stale sharedKV: spans \(sharedKV.mapValues { $0.0.dim(-2) }) != main cache offset \(cacheOffset)"
        )

        let bonusToken = y.tokens
        let draftPhaseStart = phaseClock()
        let draftTokens: MLXArray
        if let statefulDrafter = drafter as? any StatefulMTPDrafterModel,
            var currentDrafterState = drafterState
        {
            draftTokens = statefulDrafter.draftBlock(
                target: mainModel,
                lastToken: bonusToken,
                lastHidden: bonusSlotHidden,
                sharedKV: sharedKV,
                positionDeltas: state[mtpPositionDeltasKey],
                queryOffset: cacheOffset,
                blockSize: numDraft + 1,
                state: &currentDrafterState,
                sampler: drafterSampler
            )
            drafterState = currentDrafterState
        } else {
            draftTokens = drafter.draftBlock(
                target: mainModel,
                lastToken: bonusToken,
                lastHidden: bonusSlotHidden,
                sharedKV: sharedKV,
                positionDeltas: state[mtpPositionDeltasKey],
                queryOffset: cacheOffset,
                blockSize: numDraft + 1,
                sampler: drafterSampler
            )
        }
        // draftTokens shape [B, numDraft] -> flatten to [numDraft].
        let flatDraftTokens = draftTokens.flattened()
        var draftPhaseSeconds = 0.0
        if collectPhaseTelemetry {
            eval(flatDraftTokens)
            draftPhaseSeconds = ProcessInfo.processInfo.systemUptime - draftPhaseStart
        }

        // Verify pass: main model evaluates [bonus, draft_1, ..., draft_numDraft]
        // in one forward call, emitting state for next round.
        let verificationPhaseStart = phaseClock()
        var verifyState = state
        verifyState[mtpEmitFlagKey] = true
        let verifyTokens = concatenated([bonusToken, flatDraftTokens])
        let verifyInput = LMInput.Text(tokens: verifyTokens)
        let verifyStart = verifyInput.tokens.dim(0) - (numDraft + 1)
        let nativeHybridRewind =
            ((mainModel as? any SpeculativeCacheRewindModel)?
                .maximumNativeTargetCacheRewind ?? 0) >= numDraft
            && mainCache.contains { $0 is MambaCache }
            && mainCache.allSatisfy { $0.isTrimmable || $0 is MambaCache }
        verifyState[mtpCacheCheckpointIndexKey] = nativeHybridRewind ? 1 : nil
        let mainResult = mainModel(
            verifyInput[text: .newAxis], cache: mainCache, state: verifyState)
        let mainLogits = mainResult.logits
        var finalizedMainState = mainResult.state

        let draftTokensList: [Int]
        var accepted = 0
        let emittedFinalToken: MLXArray
        var sampledDecisionFailed = false
        if let sampledBlockDecisionProvider {
            eval(flatDraftTokens)
            draftTokensList = flatDraftTokens.asArray(Int.self)
            let targetLogits = (0 ..< numDraft).map { index in
                mainLogits[0..., verifyStart + index, 0...]
            }
            do {
                let decision = try sampledBlockDecisionProvider.decide(
                    proposedTokens: draftTokensList,
                    targetLogits: targetLogits,
                    bonusTargetLogits: mainLogits[0..., verifyStart + numDraft, 0...])
                guard (0 ... numDraft).contains(decision.acceptedDraftCount),
                    decision.outputTokens.count == decision.acceptedDraftCount + 1,
                    Array(decision.outputTokens.prefix(decision.acceptedDraftCount))
                        == Array(draftTokensList.prefix(decision.acceptedDraftCount)),
                    decision.outputTokens.allSatisfy({
                        $0 >= 0 && $0 < mainLogits.dim(-1)
                            && Int32(exactly: $0) != nil
                    })
                else {
                    throw SampledMTPBlockRuntimeDecisionValidationError.invalidDecision
                }
                accepted = decision.acceptedDraftCount
                pendingTokens.append(contentsOf: decision.outputTokens)
                emittedFinalToken = MLXArray([
                    Int32(decision.outputTokens[decision.acceptedDraftCount])
                ])
            } catch {
                // The verify pass has already appended [bonus, drafts]. Treat a
                // provider failure as zero acceptance, select the next token
                // from the ordinary target sampler, and let the shared cache
                // transaction below remove every draft before going sticky
                // passthrough. Draft RNG never touched this target sampler.
                let targetToken = sampler.sample(logits: targetLogits[0])
                eval(targetToken)
                verifierTokenReadbackCount += 1
                pendingTokens.append(targetToken.item(Int.self))
                emittedFinalToken = targetToken
                sampledDecisionFailed = true
            }
        } else if processor == nil, sampler is ArgMaxSampler {
            // Match the ordinary speculative iterator's processor-free path:
            // select the complete greedy verification block in one operation
            // and materialize its target token IDs once. The target forward,
            // acceptance walk, and cache transaction remain unchanged.
            let verifyLogits = mainLogits[
                0..., verifyStart ..< (verifyStart + numDraft + 1), 0...
            ].squeezed(axis: 0)
            let targetTokens = sampler.sample(logits: verifyLogits)
            eval(targetTokens, flatDraftTokens)
            verifierTokenReadbackCount += 1
            let targetTokensList = targetTokens.asArray(Int.self)
            draftTokensList = flatDraftTokens.asArray(Int.self)

            while accepted < numDraft,
                targetTokensList[accepted] == draftTokensList[accepted]
            {
                pendingTokens.append(targetTokensList[accepted])
                accepted += 1
            }
            pendingTokens.append(targetTokensList[accepted])
            emittedFinalToken = targetTokens[accepted ... accepted]
        } else {
            eval(flatDraftTokens)
            draftTokensList = flatDraftTokens.asArray(Int.self)

            var finalToken: MLXArray?
            for i in 0 ..< numDraft {
                var logits = mainLogits[0..., verifyStart + i, 0...]
                logits = processor?.process(logits: logits) ?? logits
                let targetToken = sampler.sample(logits: logits)
                eval(targetToken)
                verifierTokenReadbackCount += 1
                let targetTokenValue = targetToken.item(Int.self)
                processor?.didSample(token: targetToken)
                pendingTokens.append(targetTokenValue)
                guard targetTokenValue == draftTokensList[i] else {
                    finalToken = targetToken
                    break
                }
                accepted += 1
            }

            // Only the all-accepted path samples the bonus row. On rejection
            // the mismatching target sample above is the emitted correction.
            if finalToken == nil {
                var logits = mainLogits[0..., verifyStart + accepted, 0...]
                logits = processor?.process(logits: logits) ?? logits
                let bonus = sampler.sample(logits: logits)
                eval(bonus)
                verifierTokenReadbackCount += 1
                processor?.didSample(token: bonus)
                pendingTokens.append(bonus.item(Int.self))
                finalToken = bonus
            }
            emittedFinalToken = finalToken!
        }
        committedPendingTokenCount = accepted

        if collectPhaseTelemetry {
            synchronizeTargetPhaseBoundary(state: mainResult.state)
            telemetry.recordTargetVerification(
                seconds: ProcessInfo.processInfo.systemUptime - verificationPhaseStart)
        }

        proposedCount += numDraft
        acceptedCount += accepted
        var nextHiddenIndex = accepted
        telemetry.recordRound(
            drafted: numDraft,
            accepted: accepted,
            targetVerified: numDraft + 1,
            draftModelCalls: 1
        )

        let seedCommitStart = phaseClock()
        if !sampledDecisionFailed,
            let statefulDrafter = drafter as? any StatefulMTPDrafterModel,
            var currentDrafterState = drafterState
        {
            statefulDrafter.commitDrafterState(
                target: mainModel,
                targetHidden: mainResult.state?[mtpLastHiddenStatesKey] ?? lastHidden,
                draftTokens: draftTokens,
                acceptedCount: accepted,
                finalToken: emittedFinalToken,
                positionDeltas: mainResult.state?[mtpPositionDeltasKey],
                state: &currentDrafterState,
                sampler: drafterSampler)
            drafterState = currentDrafterState
        }
        if sampledDecisionFailed {
            drafterState = nil
        }
        if collectPhaseTelemetry {
            synchronizeDrafterPhaseBoundary()
            draftPhaseSeconds += ProcessInfo.processInfo.systemUptime - seedCommitStart
            telemetry.recordDraftBlock(seconds: draftPhaseSeconds)
        }

        let rejected = numDraft - accepted
        let hybridRewindStart = phaseClock()
        var performedHybridRewindReplay = false
        if nativeHybridRewind {
            if rejected > 0 {
                performedHybridRewindReplay = true
                let rewound = rewindSpeculativePromptCache(mainCache, numTokens: numDraft)
                precondition(
                    rewound == numDraft,
                    "Target advertised native speculative rewind, but cache rewind failed")
                if accepted > 0 {
                    let acceptedPrefix = draftTokensList.prefix(accepted)
                    let replayedState = replayAcceptedPrefixAfterHybridRewind(
                        acceptedPrefix,
                        baseState: mainResult.state ?? state,
                        emitDrafterState: true)
                    precondition(
                        replayedState != nil,
                        "Hybrid speculative rewind cannot replay accepted prefix exactly")
                    finalizedMainState = replayedState
                    nextHiddenIndex = accepted - 1
                } else {
                    trimSharedKVState(&finalizedMainState, numTokens: numDraft)
                    nextHiddenIndex = 0
                }
            }
        } else {
            precondition(
                !mainCache.contains(where: { $0 is MambaCache }),
                "Hybrid speculative cache must never use generic KV trimming")
            let trimmed = trimPromptCache(mainCache, numTokens: rejected)
            trimSharedKVState(&finalizedMainState, numTokens: trimmed)
        }
        mainState = finalizedMainState
        lastRoundAccepted = nextHiddenIndex
        if collectPhaseTelemetry, performedHybridRewindReplay {
            synchronizeTargetPhaseBoundary(state: mainState)
            telemetry.recordHybridRewindReplay(
                seconds: ProcessInfo.processInfo.systemUptime - hybridRewindStart)
        }

        // Dynamic cache quantization may convert `.regular` K/V to `.quantized`,
        // at which point the target's emit-hook returns sharedKV: nil and the
        // next round transitions to passthrough.
        quantizeKVCache(&mainCache)

        y = .init(tokens: emittedFinalToken)
        if sampledDecisionFailed {
            switchToPassthrough(reason: "sampled MTP block decision failed")
        }
    }

    /// Switch to single-token generation for the remainder of the stream.
    /// Sticky — once flipped, `next()` never returns to speculation.
    private mutating func switchToPassthrough(reason: String) {
        if !passthroughLoggedOnce {
            // Log one-time only so a quantization-onset round doesn't spam.
            // The Swift stdlib `print` is intentional here: the iterator is
            // a low-level component without access to a logger.
            print("[MTPSpeculativeTokenIterator] passthrough mode: \(reason)")
            passthroughLoggedOnce = true
        }
        passthroughReason = reason
        passthrough = true
        // A prepare-time re-prime may already have committed one pending
        // token to the hybrid cache. Preserve its recurrent checkpoint until
        // that token is accepted by the caller or finalization discards it.
        if committedPendingTokenCount == 0 {
            discardSpeculativePromptCacheCheckpoints(mainCache)
        }
        mainState?[mtpEmitFlagKey] = false
        mainState?[mtpCacheCheckpointIndexKey] = nil
        mainState?[mtpLastHiddenStatesKey] = nil
        mainState?[mtpSharedKVStatesKey] = nil
        mainState?[mtpSharedKVOffsetsKey] = nil
        mainState?[mtpPositionDeltasKey] = nil
    }

    /// One single-token forward step against the main model, used in
    /// passthrough mode. The drafter is not invoked.
    private mutating func passthroughStep() -> Int? {
        if let maxTokens, tokenCount >= maxTokens { return nil }

        let targetTailStart = phaseClock()
        let result = mainModel(y[text: .newAxis], cache: mainCache, state: mainState)
        mainState = result.state
        var logits = result.logits[0..., -1, 0...]
        logits = processor?.process(logits: logits) ?? logits
        let token = sampler.sample(logits: logits)
        processor?.didSample(token: token)
        eval(token)
        let tokenInt = token.item(Int.self)
        y = .init(tokens: token)
        quantizeKVCache(&mainCache)
        if collectPhaseTelemetry {
            synchronizeTargetPhaseBoundary(state: mainState)
            telemetry.recordTargetTail(
                seconds: ProcessInfo.processInfo.systemUptime - targetTailStart)
        }
        lastReturnedTokenNeedsFinalCommit = true
        return tokenInt
    }

    private mutating func replayAcceptedPrefixAfterHybridRewind(
        _ tokenValues: ArraySlice<Int>,
        baseState: LMOutput.State,
        emitDrafterState: Bool
    ) -> LMOutput.State? {
        guard !tokenValues.isEmpty else { return baseState }
        guard checkpointSpeculativePromptCacheBeforeAppend(
            mainCache, tokenCount: tokenValues.count
        ) else {
            return nil
        }

        var replayTokens = [Int32]()
        replayTokens.reserveCapacity(tokenValues.count)
        for token in tokenValues {
            guard let exact = Int32(exactly: token) else { return nil }
            replayTokens.append(exact)
        }

        var replayState = baseState
        replayState[mtpEmitFlagKey] = emitDrafterState
        replayState[mtpCacheCheckpointIndexKey] = nil
        let replayInput = LMInput.Text(tokens: MLXArray(replayTokens))
        let replayResult = mainModel(
            replayInput[text: .newAxis], cache: mainCache, state: replayState)
        if emitDrafterState, replayResult.state == nil {
            return nil
        }
        return replayResult.state ?? baseState
    }

    public mutating func next() -> Int? {
        // Calling `next` again acknowledges the prior token. Once every
        // cache-committed pending token has been retained, its rollback
        // checkpoint is no longer needed. `discardGeneratedToken()` clears
        // the marker first, so a rejected terminal token keeps the snapshot
        // for `finalizeGeneration()`.
        if lastReturnedTokenWasCommittedPending {
            lastReturnedTokenWasCommittedPending = false
            if emittedCommittedPendingTokenCount == committedPendingTokenCount {
                discardSpeculativePromptCacheCheckpoints(mainCache)
            }
        }
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }
        // A subsequent iteration consumes the prior uncommitted token as the
        // next target input. Only a terminal retained token needs the explicit
        // finalizer forward below.
        lastReturnedTokenNeedsFinalCommit = false

        // Drain the pending buffer first.
        if pendingIndex < pendingTokens.count {
            let token = pendingTokens[pendingIndex]
            lastReturnedTokenWasCommittedPending =
                pendingIndex < committedPendingTokenCount
            if lastReturnedTokenWasCommittedPending {
                emittedCommittedPendingTokenCount += 1
            }
            lastReturnedTokenNeedsFinalCommit = !lastReturnedTokenWasCommittedPending
            pendingIndex += 1
            telemetry.recordGeneratedToken()
            return token
        }

        if passthrough {
            if let token = passthroughStep() {
                telemetry.recordGeneratedToken()
                return token
            }
            return nil
        }

        // Run a new speculation round (may transition to passthrough).
        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        committedPendingTokenCount = 0
        emittedCommittedPendingTokenCount = 0
        speculateRound()

        if pendingTokens.isEmpty {
            // speculateRound chose passthrough -- fall through.
            if passthrough {
                if let token = passthroughStep() {
                    telemetry.recordGeneratedToken()
                    return token
                }
            }
            return nil
        }

        let token = pendingTokens[pendingIndex]
        lastReturnedTokenWasCommittedPending =
            pendingIndex < committedPendingTokenCount
        if lastReturnedTokenWasCommittedPending {
            emittedCommittedPendingTokenCount += 1
        }
        lastReturnedTokenNeedsFinalCommit = !lastReturnedTokenWasCommittedPending
        pendingIndex += 1
        telemetry.recordGeneratedToken()
        return token
    }
}

extension MTPSpeculativeTokenIterator: GenerationFinalizingTokenIterator {
    public mutating func finalizeGeneration() {
        let finalizationStart = phaseClock()
        var nestedHybridRewindSeconds = 0.0
        defer {
            discardSpeculativePromptCacheCheckpoints(mainCache)
            if collectPhaseTelemetry {
                telemetry.recordFinalization(seconds: Swift.max(
                    0,
                    ProcessInfo.processInfo.systemUptime - finalizationStart
                        - nestedHybridRewindSeconds))
            }
        }
        let emitted = Swift.min(
            emittedCommittedPendingTokenCount, committedPendingTokenCount)
        let lookahead = committedPendingTokenCount - emitted
        if lookahead > 0 {
            if mainCache.contains(where: { $0 is MambaCache }) {
                let hybridRewindStart = phaseClock()
                let rewound = rewindSpeculativePromptCache(
                    mainCache, numTokens: committedPendingTokenCount)
                precondition(
                    rewound == committedPendingTokenCount,
                    "Hybrid speculative finalization requires an exact recurrent checkpoint")
                if emitted > 0 {
                    let replayedState = replayAcceptedPrefixAfterHybridRewind(
                        pendingTokens.prefix(emitted),
                        baseState: mainState ?? LMOutput.State(),
                        emitDrafterState: false)
                    precondition(
                        replayedState != nil,
                        "Hybrid speculative finalization cannot replay retained prefix exactly")
                    mainState = replayedState
                }
                if collectPhaseTelemetry {
                    synchronizeTargetPhaseBoundary(state: mainState)
                    nestedHybridRewindSeconds =
                        ProcessInfo.processInfo.systemUptime - hybridRewindStart
                    telemetry.recordHybridRewindReplay(seconds: nestedHybridRewindSeconds)
                }
            } else {
                let rewound = trimPromptCache(mainCache, numTokens: lookahead)
                trimSharedKVState(&mainState, numTokens: rewound)
            }
        }

        guard lastReturnedTokenNeedsFinalCommit else { return }
        precondition(
            lookahead == 0,
            "An uncommitted retained token cannot precede committed speculative lookahead")

        // Match scalar TokenIterator semantics at length/stop boundaries: a
        // retained token is present in cache even when there is no next loop
        // iteration to consume it. Disable MTP-only state emission and native
        // rewind checkpointing for this terminal scalar commit.
        var finalState = mainState ?? LMOutput.State()
        finalState[mtpEmitFlagKey] = false
        finalState[mtpCacheCheckpointIndexKey] = nil
        let result = mainModel(y[text: .newAxis], cache: mainCache, state: finalState)
        mainState = result.state
        quantizeKVCache(&mainCache)
        eval(result.logits)
        synchronizeTargetPhaseBoundary(state: mainState)
        lastReturnedTokenNeedsFinalCommit = false
    }
}

extension MTPSpeculativeTokenIterator: MTPStatsCollecting {
    public var proposedDraftTokens: Int { proposedCount }
    public var acceptedDraftTokens: Int { acceptedCount }
}

extension MTPSpeculativeTokenIterator {
    /// Test-only setter for the canonical `LogitProcessor`. Lets regression
    /// tests install a recording probe AFTER `init` (which calls `prepare`
    /// and would otherwise consume the prepare-time bonus before the probe
    /// is observable). Used by the emit-only invariant regression tests in
    /// `MTPSpeculativeTokenIteratorTests` (CI-scoped) and
    /// `MTPIteratorEndToEndDiagnosticTests` (31B end-to-end).
    @_spi(Testing) public mutating func _setProcessorForTesting(
        _ processor: LogitProcessor?
    ) {
        self.processor = processor
    }

    /// Test-only getter for the canonical `LogitProcessor` so regression
    /// tests can inspect its post-drain state (e.g., a recording probe's
    /// accumulated didSample log).
    @_spi(Testing) public var _processorForTesting: LogitProcessor? {
        processor
    }

    /// Number of host materializations used to read target-selected tokens
    /// from speculative verification blocks. Exposed only to regression tests
    /// so the greedy batched-readback contract remains observable.
    @_spi(Testing) public var _verifierTokenReadbackCountForTesting: Int {
        verifierTokenReadbackCount
    }
}

/// Rewinds the emitted MTP shared-K/V snapshot by `numTokens` trailing
/// sequence positions, mirroring `trimPromptCache` on the main cache.
///
/// The verify pass emits K/V spanning the full `[bonus, d_1 ... d_numDraft]`
/// chunk — materialized before acceptance is known. After a partial
/// acceptance, the rejected tail rows describe tokens that are not part of
/// the sequence; without this trim, the next round's `draftBlock` would
/// cross-attend over them. (PR #308 review: discussion_r3391133046,
/// discussion_r3391147261.)
///
/// No-op when `numTokens <= 0`, when `state` is nil, or when the key is
/// absent (e.g. the quantization-onset round, whose fresh verify state
/// carries no sharedKV). Cost is metadata-only: the slices are lazy views
/// consumed by the next `draftBlock` like the rest of the round's inputs;
/// no `eval`. Iterator-internal — `trimPromptCache` is public because
/// caches are a public surface, but this snapshot is the iterator's own
/// cross-round state.
func trimSharedKVState(_ state: inout LMOutput.State?, numTokens: Int) {
    guard numTokens > 0,
        let sharedKV = state?[mtpSharedKVStatesKey]
    else { return }
    state?[mtpSharedKVStatesKey] = sharedKV.mapValues { kv in
        let newLen = kv.0.dim(-2) - numTokens
        return (
            kv.0[.ellipsis, ..<newLen, 0...],
            kv.1[.ellipsis, ..<newLen, 0...]
        )
    }
    if let offsets = state?[mtpSharedKVOffsetsKey] {
        state?[mtpSharedKVOffsetsKey] = offsets.mapValues {
            Swift.max(0, $0 - numTokens)
        }
    }
}
