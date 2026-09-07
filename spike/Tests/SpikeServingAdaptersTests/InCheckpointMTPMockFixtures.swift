import Foundation
import XCTest

import MLX
import MLXLMCommon
import MLXNN
@testable import SpikeServingAdapters

// Extracted from `InCheckpointMTPStartupGateEndToEndTests.swift` (cycle 77, MTP decoder bridge
// increment) so `MTPSpeculativeDecoderTests.swift` can reuse the identical weight-free mock
// target/drafter pair without duplicating them. Access level widened from `private` to the
// target-internal default (`internal`) so both test files in this target can see these types;
// nothing else about them changed.
//
// MARK: - Weight-free fixtures ported from
// spike/Vendor/mlx-swift-lm/Tests/MLXLMTests/MTPSpeculativeTokenIteratorTests.swift
//
// `spike` and `spike/Vendor/mlx-swift-lm` are SEPARATE Swift packages, so that vendored test
// target's PRIVATE fixtures (its `MockMainModel`/`MockDrafter`/`CountingKVCache`) are not
// importable across the package boundary, and that file also gates its target-model mock on
// `@_spi(Testing) @testable import MLXLMCommon` for VLM-style prompt-hidden reuse this port does
// not need. Every protocol/type actually required below IS `public`: `LanguageModel`,
// `MTPDrafterModel`, `KVCache`, `Module`, `LMInput`/`LMOutput`/`PrepareResult`/`GenerateParameters`,
// `TokenIterator`/`MTPSpeculativeTokenIterator`, and the `mtp*Key` `LMOutput.Key` constants
// (`mtpEmitFlagKey`, `mtpLastHiddenStatesKey`, `mtpSharedKVStatesKey`) the target mock reads/writes
// to opt into (and emit) MTP state. So the port below is feasible without widening any access
// level in the vendored package — no vendored source file is touched by this increment.
//
// Simplified from the vendored originals in two ways:
//   1. Only the STATELESS drafter path (`MTPDrafterModel`, not `StatefulMTPDrafterModel`) is
//      ported, because the production drafter `runInCheckpointMTPSpeculativeDecode` actually
//      drives at startup is itself a plain, non-stateful `MTPDrafterModel` with
//      `requiresPromptPrefill == false` (the vendored facade wrapping the in-checkpoint MTP
//      weights). The `MTPPromptHiddenStateSelectivePreparingModel` VLM-style prompt-reuse path
//      is entirely skipped for the same reason — the iterator only consults it when
//      `drafter.requiresPromptPrefill` is true.
//   2. The target mock selects each call's planned logits from the MOCK CACHE'S OWN OFFSET,
//      not a persistent per-instance call counter (unlike the vendored original). This is a
//      deliberate generalization: the same mock instance must drive BOTH the scalar-reference
//      (`TokenIterator`) and speculative (`MTPSpeculativeTokenIterator`) decode paths in the
//      SAME test — exactly what `verifyInCheckpointMTPStartupReadiness` itself does — each
//      starting from its own fresh cache at offset 0. A shared call counter would let the first
//      run's calls silently consume planned tokens the second run then never sees.

/// Minimal `KVCache` mock: tracks only its own offset, which the target mock below uses to pick
/// each call's planned logits and the iterator uses for its own trim/rewind bookkeeping. Port of
/// the vendored suite's private `CountingKVCache`.
final class InCheckpointMTPMockCountingKVCache: KVCache {
    var offset: Int = 0
    var maxSize: Int? { nil }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) { (keys, values) }
    var state: [MLXArray] {
        get { [] }
        set {}
    }
    var metaState: [String] {
        get { [] }
        set {}
    }
    var isTrimmable: Bool { true }
    @discardableResult
    func trim(_ n: Int) -> Int {
        let removed = Swift.min(n, offset)
        offset -= removed
        return removed
    }
    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }
    func copy() -> any KVCache {
        let copy = InCheckpointMTPMockCountingKVCache()
        copy.offset = offset
        return copy
    }
    func innerState() -> [MLXArray] { [] }
}

/// Minimal `LanguageModel` mock. Emits deterministic one-hot logits over a 20-token vocabulary
/// at each absolute cache position, indexed into `plannedTokens` — so an `ArgMaxSampler`
/// deterministically reproduces the planned sequence regardless of whether a run consumes it one
/// token at a time (scalar decode) or in multi-token batches (speculative verify). When the
/// iterator opts in via `mtpEmitFlagKey`, also emits `mtpLastHiddenStatesKey` and
/// `mtpSharedKVStatesKey["full_attention"]` spanning the cache's own (post-update) offset —
/// mirroring the vendored `MockMainModel`'s emit hook exactly, since the iterator asserts that
/// span matches the main cache's offset before drafting.
final class InCheckpointMTPMockTargetModel: Module, LanguageModel {
    let plannedTokens: [Int32]

    /// Counts calls into `prepareForMTP` below — the anti-vacuity signal for
    /// this file's production-shaped tests. `MTPSpeculativeTokenIterator`
    /// only reaches this method when `drafter.requiresPromptPrefill` is true
    /// (see `MTPSpeculativeTokenIterator.swift:275-281`), so it stays 0 for
    /// every test that drives the existing stateless
    /// `InCheckpointMTPMockDrafter` (`requiresPromptPrefill == false`) and
    /// becomes nonzero only when the stateful, production-shaped drafter
    /// below actually engages speculation. Asserting both values (0 for
    /// stateless, nonzero for stateful) is what proves the two
    /// configurations exercise genuinely different iterator branches rather
    /// than the mock silently falling back to the same one.
    private(set) var prepareForMTPCallCount = 0

    /// Every `windowSize` observed at `prepare(_:cache:windowSize:)`, in call order —
    /// `TokenIterator`/`MTPSpeculativeTokenIterator` both thread `parameters.prefillStepSize`
    /// straight through as this parameter (`Evaluate.swift:643,905`, and see
    /// `MTPSpeculativeDecoderPrefillChunkParityTests.swift` for the precedent this mirrors), so
    /// this is the nearest observable proxy for what each gate arm actually set. One entry per
    /// arm when the SAME target instance drives both `runInCheckpointMTPScalarReference` and
    /// `runInCheckpointMTPSpeculativeDecode` in sequence.
    private(set) var observedPrepareWindowSizes: [Int?] = []

    init(plannedTokens: [Int32]) {
        self.plannedTokens = plannedTokens
        super.init()
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        observedPrepareWindowSizes.append(windowSize)
        return .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        makeLogits(startIndex: 0, positions: inputs.dim(-1))
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let positions = input.tokens.dim(-1)
        let countingCache = cache?.first as? InCheckpointMTPMockCountingKVCache
        let startIndex = countingCache?.offset ?? 0
        let logits = makeLogits(startIndex: startIndex, positions: positions)
        countingCache?.offset = startIndex + positions

        guard state?[mtpEmitFlagKey] ?? false else {
            return LMOutput(logits: logits)
        }
        let kvSpan = countingCache?.offset ?? positions
        var out = state ?? LMOutput.State()
        out[mtpLastHiddenStatesKey] = MLXArray.zeros([1, positions, 4])
        out[mtpSharedKVStatesKey] = [
            "full_attention": (
                MLXArray.zeros([1, 1, kvSpan, 4]),
                MLXArray.zeros([1, 1, kvSpan, 4])
            )
        ]
        return LMOutput(logits: logits, state: out)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [InCheckpointMTPMockCountingKVCache()]
    }

    private func makeLogits(startIndex: Int, positions: Int) -> MLXArray {
        let vocab = 20
        var data = [Float](repeating: 0, count: positions * vocab)
        for i in 0..<positions {
            let tokIdx = startIndex + i
            let tok = tokIdx < plannedTokens.count ? Int(plannedTokens[tokIdx]) : 0
            precondition(
                tok >= 0 && tok < vocab,
                "planned token \(tok) at index \(tokIdx) is outside the mock's \(vocab)-token vocabulary")
            data[i * vocab + tok] = 100
        }
        return MLXArray(data, [1, positions, vocab])
    }
}

extension InCheckpointMTPMockTargetModel: MTPPromptHiddenStatePreparingModel {
    /// Production `prepareForMTP` conformance. `MTPSpeculativeTokenIterator.prepare` reaches this
    /// via the FINAL `else if` in its four-branch cascade (`MTPSpeculativeTokenIterator.swift:
    /// 275-281`) — the only one of the four gated on the base `MTPPromptHiddenStatePreparingModel`
    /// protocol rather than one of its richer, `@_spi(Testing)`-adjacent refinements
    /// (`...Selective...`, `...EvaluationOrder...`, `...Telemetry...`), which this mock
    /// deliberately does NOT conform to since those refinements' extra parameters
    /// (`evaluationOrder`, `requiresSharedTargetKV`, `collectTelemetry`) are diagnostic-only and
    /// the production target model this file's background cites also conforms to exactly the base
    /// protocol.
    ///
    /// Batches the ENTIRE prompt through the existing `callAsFunction(_:cache:state:)` overload in
    /// one call with `mtpEmitFlagKey` forced true, exactly mirroring what `mainModel.prepare` +
    /// the iterator's own follow-up forward do together on the stateless (`.tokens`) path — so the
    /// bonus token this samples (from the LAST prompt position, via the iterator's own
    /// `prefillResult.logits[0..., -1, 0...]` slice at `MTPSpeculativeTokenIterator.swift:322`) is
    /// numerically identical to what the stateless path would sample for the same `plannedTokens`.
    /// That equivalence is what lets the production-shaped tests below reuse the exact
    /// `plannedTokens`/`draftedTokenValue` fixtures from the stateless tests and assert the exact
    /// same verdict shape — proving the production branch reproduces the already-trusted stateless
    /// behavior, not merely that it runs.
    ///
    /// Returns `.logits(output)` with `output.state` already carrying `mtpLastHiddenStatesKey` AND
    /// `mtpSharedKVStatesKey` (since the emit flag was forced true), so the iterator's own
    /// re-prime-forward gate (`MTPSpeculativeTokenIterator.swift:337-341`, `mainState?
    /// [mtpLastHiddenStatesKey] == nil || ...`) evaluates false and no second forward call happens
    /// — one `prepareForMTP` call correlates with exactly one target forward over the prompt, kept
    /// deliberately simple so `prepareForMTPCallCount` is a clean, unambiguous anti-vacuity signal.
    func prepareForMTP(
        _ input: LMInput,
        cache: [KVCache],
        windowSize: Int?
    ) throws -> MTPPromptPreparation? {
        prepareForMTPCallCount += 1
        let promptText =
            input.text.tokens.ndim == 1
            ? LMInput.Text(tokens: input.text.tokens[.newAxis, 0...], mask: input.text.mask)
            : input.text
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true
        let output = callAsFunction(promptText, cache: cache, state: state)
        guard let hidden = output.state?[mtpLastHiddenStatesKey] else { return nil }
        return MTPPromptPreparation(result: .logits(output), targetHidden: hidden)
    }
}

// MARK: - Divergent-arm fixtures (Change 1's "degrade FIRES" coverage)
//
// `InCheckpointMTPMockTargetModel` above is deterministic purely by ABSOLUTE CACHE POSITION, so
// scalar decode (`TokenIterator`) and speculative decode (`MTPSpeculativeTokenIterator`) driven
// against the SAME instance always read the SAME planned-token table and therefore always agree
// -- it cannot exercise a genuinely divergent pair. The real divergence this repository measured
// (`docs/task-inbox/2026-09-07-mtp-scalar-route-divergence-DECISION.md`) comes from the two
// arms' DIFFERENT forward call geometry (scalar evaluates one position per step past prefill;
// the speculative verify forward evaluates several at once), which this weight-free mock cannot
// reproduce bit-for-bit. Instead, this fixture manufactures a DIFFERENT, verifiable form of the
// same OBSERVABLE property the gate actually cares about: the same target instance, driven
// through both arms in one gate call, produces two genuinely different resulting token
// sequences. It does this by tagging each arm's own KV cache (`newCache` is called exactly once
// per arm, in a fixed, sequential order — scalar reference first, speculative decode second, per
// `verifyInCheckpointMTPStartupReadiness`'s own call order) and reading a DIFFERENT planned-token
// table depending on which arm's cache is driving the current forward call.

/// Minimal `KVCache` mock identical to `InCheckpointMTPMockCountingKVCache` except it also tags
/// which of the two gate arms it belongs to, fixed at construction.
final class InCheckpointMTPMockArmTaggedKVCache: KVCache {
    var offset: Int = 0
    let isSecondArm: Bool
    var maxSize: Int? { nil }

    init(isSecondArm: Bool) {
        self.isSecondArm = isSecondArm
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) { (keys, values) }
    var state: [MLXArray] {
        get { [] }
        set {}
    }
    var metaState: [String] {
        get { [] }
        set {}
    }
    var isTrimmable: Bool { true }
    @discardableResult
    func trim(_ n: Int) -> Int {
        let removed = Swift.min(n, offset)
        offset -= removed
        return removed
    }
    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }
    func copy() -> any KVCache {
        let copy = InCheckpointMTPMockArmTaggedKVCache(isSecondArm: isSecondArm)
        copy.offset = offset
        return copy
    }
    func innerState() -> [MLXArray] { [] }
}

/// `LanguageModel` mock whose OWN greedy prediction genuinely differs between the two gate arms
/// driven against it, by reading a different table depending on which tagged cache
/// (`InCheckpointMTPMockArmTaggedKVCache.isSecondArm`) is presented at each forward call. See this
/// section's header comment for why this is the deliberate, controllable stand-in for the real
/// forward-geometry divergence this fixture cannot reproduce directly.
final class InCheckpointMTPMockDivergentTargetModel: Module, LanguageModel {
    let firstArmPlannedTokens: [Int32]
    let secondArmPlannedTokens: [Int32]
    /// Counts `newCache` calls so the FIRST call (the scalar reference arm, per
    /// `verifyInCheckpointMTPStartupReadiness`'s fixed call order) tags its cache
    /// `isSecondArm == false` and every call after it tags `isSecondArm == true`.
    private var vendedCacheCount = 0

    init(firstArmPlannedTokens: [Int32], secondArmPlannedTokens: [Int32]) {
        self.firstArmPlannedTokens = firstArmPlannedTokens
        self.secondArmPlannedTokens = secondArmPlannedTokens
        super.init()
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let table = plannedTokens(for: cache)
        return makeLogits(table: table, startIndex: 0, positions: inputs.dim(-1))
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let positions = input.tokens.dim(-1)
        let taggedCache = cache?.first as? InCheckpointMTPMockArmTaggedKVCache
        let startIndex = taggedCache?.offset ?? 0
        let table = plannedTokens(for: cache)
        let logits = makeLogits(table: table, startIndex: startIndex, positions: positions)
        taggedCache?.offset = startIndex + positions

        guard state?[mtpEmitFlagKey] ?? false else {
            return LMOutput(logits: logits)
        }
        let kvSpan = taggedCache?.offset ?? positions
        var out = state ?? LMOutput.State()
        out[mtpLastHiddenStatesKey] = MLXArray.zeros([1, positions, 4])
        out[mtpSharedKVStatesKey] = [
            "full_attention": (
                MLXArray.zeros([1, 1, kvSpan, 4]),
                MLXArray.zeros([1, 1, kvSpan, 4])
            )
        ]
        return LMOutput(logits: logits, state: out)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        vendedCacheCount += 1
        return [InCheckpointMTPMockArmTaggedKVCache(isSecondArm: vendedCacheCount >= 2)]
    }

    private func plannedTokens(for cache: [KVCache]?) -> [Int32] {
        let taggedCache = cache?.first as? InCheckpointMTPMockArmTaggedKVCache
        return (taggedCache?.isSecondArm ?? false) ? secondArmPlannedTokens : firstArmPlannedTokens
    }

    private func makeLogits(table: [Int32], startIndex: Int, positions: Int) -> MLXArray {
        let vocab = 20
        var data = [Float](repeating: 0, count: positions * vocab)
        for i in 0..<positions {
            let tokIdx = startIndex + i
            let tok = tokIdx < table.count ? Int(table[tokIdx]) : 0
            precondition(
                tok >= 0 && tok < vocab,
                "planned token \(tok) at index \(tokIdx) is outside the mock's \(vocab)-token vocabulary")
            data[i * vocab + tok] = 100
        }
        return MLXArray(data, [1, positions, vocab])
    }
}

/// Minimal, stateless `MTPDrafterModel` mock: always proposes `draftedTokenValue`, repeated
/// `blockSize - 1` times, regardless of round or position. `supportsTarget` lets a test force
/// sticky passthrough at iterator init by refusing `supportsSpeculation(for:target:)`. Port of
/// the vendored suite's private `MockDrafter`, with that one addition.
final class InCheckpointMTPMockDrafter: Module, MTPDrafterModel {
    let draftedTokenValue: Int32
    var supportsTarget = true
    private(set) var draftBlockCallCount = 0

    init(draftedTokenValue: Int32) {
        self.draftedTokenValue = draftedTokenValue
        super.init()
    }

    func supportsSpeculation(for input: LMInput, target: any LanguageModel) -> Bool {
        supportsTarget
    }

    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        draftBlockCallCount += 1
        let batch = lastToken.dim(0)
        let vals = Array(repeating: draftedTokenValue, count: (blockSize - 1) * batch)
        return MLXArray(vals, [batch, blockSize - 1])
    }
}

/// Production-shaped `StatefulMTPDrafterModel` mock: `requiresPromptPrefill == true`, exactly
/// like the real in-checkpoint MTP drafter, which the loader facade returns typed as
/// `any StatefulMTPDrafterModel`. Paired
/// with `InCheckpointMTPMockTargetModel`'s `MTPPromptHiddenStatePreparingModel` conformance
/// above, this makes `MTPSpeculativeTokenIterator` take the SAME branch cascade production takes
/// — `drafterState = ...makeState(...)` at init (`MTPSpeculativeTokenIterator.swift:181-183`),
/// `reuseTarget.prepareForMTP` during prepare (`MTPSpeculativeTokenIterator.swift:275-281`), and
/// `statefulDrafter.prepareDrafterState` to prime it (`MTPSpeculativeTokenIterator.swift:
/// 439-446`) — none of which the file's original stateless `InCheckpointMTPMockDrafter` ever
/// reaches, since every one of those call sites is gated on `drafter.requiresPromptPrefill`.
///
/// `draftBlock`'s ACTUAL proposal logic is identical to the stateless mock's (always propose
/// `draftedTokenValue`, `blockSize - 1` times); only the stateful plumbing around it differs.
/// Both the `MTPDrafterModel`-inherited stateless signature and the `StatefulMTPDrafterModel`
/// stateful signature must be implemented (protocol conformance requires both — see the vendored
/// `MockStatefulGreedyDrafter` at `MTPSpeculativeTokenIteratorTests.swift:94-124` for the same
/// pattern); the stateless one is never invoked by the iterator once a drafter conforms to
/// `StatefulMTPDrafterModel` (the iterator always downcasts and calls the stateful overload), but
/// it counts toward the SAME `draftBlockCallCount` so a test cannot be fooled by a drafter that
/// silently answers through the wrong overload.
final class InCheckpointMTPMockStatefulDrafter: Module, StatefulMTPDrafterModel {
    let draftedTokenValue: Int32
    var supportsTarget = true
    let requiresPromptPrefill = true
    private(set) var draftBlockCallCount = 0
    /// Nonzero iff `MTPSpeculativeTokenIterator.init` actually constructed drafter state for this
    /// stream (`MTPSpeculativeTokenIterator.swift:181-183`) — which only happens when the drafter
    /// wasn't refused at init (no `initialPassthroughReason`). Distinguishes "speculated" runs
    /// from "refused before priming" runs at the finest granularity the iterator exposes.
    private(set) var makeStateCallCount = 0
    /// Nonzero iff the iterator's prompt-priming block actually reached
    /// `statefulDrafter.prepareDrafterState` (`MTPSpeculativeTokenIterator.swift:439-446`), which
    /// additionally requires the target to have emitted BOTH `mtpLastHiddenStatesKey` and
    /// `mtpSharedKVStatesKey` from `prepareForMTP` — the single strongest anti-vacuity signal in
    /// this file, since it is unreachable by any code path the stateless mock/drafter pair can hit.
    private(set) var prepareDrafterStateCallCount = 0

    init(draftedTokenValue: Int32) {
        self.draftedTokenValue = draftedTokenValue
        super.init()
    }

    func supportsSpeculation(for input: LMInput, target: any LanguageModel) -> Bool {
        supportsTarget
    }

    func makeState(parameters: GenerateParameters?) -> MTPDrafterState {
        makeStateCallCount += 1
        return MTPDrafterState(cache: [])
    }

    func prepareDrafterState(
        target: any LanguageModel,
        promptTokens: MLXArray,
        targetHidden: MLXArray,
        firstBonus: MLXArray,
        positionDeltas: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) {
        prepareDrafterStateCallCount += 1
    }

    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        draftBlockCallCount += 1
        let batch = lastToken.dim(0)
        let vals = Array(repeating: draftedTokenValue, count: (blockSize - 1) * batch)
        return MLXArray(vals, [batch, blockSize - 1])
    }

    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray {
        draftBlock(
            target: target, lastToken: lastToken, lastHidden: lastHidden, sharedKV: sharedKV,
            positionDeltas: positionDeltas, queryOffset: queryOffset, blockSize: blockSize,
            sampler: sampler)
    }
}
