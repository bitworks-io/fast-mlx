import Foundation
import HarnessCore
import MLX
import MLXLMCommon

public enum CompiledMLXDecoderSnapshotError: Error, Equatable, Sendable {
    case unsupportedCacheKind
    case speculativeStateUnsupported
    case decoderNotInitialized
    case invalidPipelineState
    case invalidPrompt
    case invalidTailToken(position: Int)
    case layerCountMismatch
    case layerTokenCountMismatch(layer: Int)
    case invalidNextToken
    case invalidSnapshotMetadata
    case byteCountOverflow
}

/// Exact request-start state for the dense scalar compiled decoder.
///
/// A snapshot contains the logical fp16 prefix for every layer plus the greedy token that the
/// prefix predicts. Retaining that next-token pipeline state lets an exact hit skip prompt
/// evaluation entirely while preserving the decoder's submit-first lookahead invariant.
public struct CompiledMLXDecoderSnapshot {
    public let logicalTokenCount: Int
    public let nextTokenID: Int
    public let layerCount: Int
    public let arrayNBytes: Int
    public let controlNBytes: Int
    public let totalNBytes: Int

    let layerSnapshots: [CompiledKVCacheSnapshot]

    init(
        logicalTokenCount: Int,
        nextTokenID: Int,
        layerSnapshots: [CompiledKVCacheSnapshot]
    ) {
        self.logicalTokenCount = logicalTokenCount
        self.nextTokenID = nextTokenID
        self.layerSnapshots = layerSnapshots
        layerCount = layerSnapshots.count

        var arrayBytes = 0
        var byteCountOverflow = false
        for snapshot in layerSnapshots {
            let (next, overflow) = arrayBytes.addingReportingOverflow(
                snapshot.totalNBytes)
            if overflow || snapshot.totalNBytes < 0 {
                byteCountOverflow = true
                break
            }
            arrayBytes = next
        }
        let (layerControlBytes, controlMultiplyOverflow) =
            layerSnapshots.count.multipliedReportingOverflow(
                by: MemoryLayout<CompiledKVCacheSnapshot>.stride)
        let (controlBytes, controlAddOverflow) =
            layerControlBytes.addingReportingOverflow(
                MemoryLayout<Int32>.stride)
        let (allBytes, totalOverflow) =
            arrayBytes.addingReportingOverflow(controlBytes)
        if byteCountOverflow || controlMultiplyOverflow || controlAddOverflow
            || totalOverflow
        {
            arrayNBytes = -1
            controlNBytes = -1
            totalNBytes = -1
        } else {
            arrayNBytes = arrayBytes
            controlNBytes = controlBytes
            totalNBytes = allBytes
        }
    }
}

/// Decoder whose single-token step is wrapped in `MLX.compile`.
///
/// Why: Instruments showed the baseline `MLXDecoder`'s 6.7ms/step "submit" phase is
/// ~82% inside `mlx_async_eval` -> C++ `eval_impl` (per-primitive graph traversal +
/// Metal kernel encoding) and ~18% Swift graph construction — repeated for every
/// token. Compiling the step replays a cached, fused graph instead: the Swift forward
/// runs once per (shape signature), elementwise chains fuse into fewer kernels, and
/// per-step traversal shrinks with the node count.
///
/// The KV-cache position is carried by the caches as in-graph state (array offset +
/// fixed-size buffers), which is what makes the trace valid for every step. The cache
/// implementation is selected by `KVCacheKind` — fp16, native affine, or TurboQuant;
/// each satisfies `CompiledCache`, the contract this decoder needs (Updatable state,
/// chunked grow, identity-preserving reset).
/// Keeps MLXDecoder's submit-first lookahead: the next step's compiled call is
/// submitted (asyncEval) BEFORE the current token's blocking `.item()` readback.
public struct CompiledMLXDecoder: Decoder {
    private let model: any LanguageModel
    private let kvCacheKind: KVCacheKind
    private let affineAttentionMode: AffineKVAttentionMode
    private let kvarnAttentionMode: KVarNKVAttentionMode
    private let kvarnStorageDType: KVarNKVScalarDType?
    /// Escape hatch for cache implementations whose update ops fail to trace under
    /// `MLX.compile`: `false` runs the same step closure uncompiled (correctness path;
    /// per-token graph construction returns, so decode perf drops — flagged, never silent).
    private let compileStepEnabled: Bool
    public let executionMode: KVCacheExecutionMode
    private let chunk =
        KVTunerCandidateRuntimeContract.cacheGrowthChunkTokens
    /// Split affine attention exposes its score tensor, so an all-at-once long prefill would
    /// allocate O(prompt²) temporary storage. Bound the query side while retaining one fixed
    /// cache and exact causal semantics; this matches mlx-lm's standard prefill step size.
    private static let splitAttentionPrefillChunkSize = 512
    /// Decode headroom preallocated beyond the prompt (rounded up to `chunk`).
    private let reserve: Int

    private var caches: [any CompiledCache] = []
    // Plain (non-@Sendable) function type: `compile` returns a @Sendable function (a
    // subtype, assigns fine), while the uncompiled fallback closure captures the
    // non-Sendable model/caches and stays confined to this decoder like all MLX state.
    private var compiledStep: (([MLXArray]) -> [MLXArray])?
    /// Separate fixed-K compiled verify forward for the spec-decode path (lazy; survives
    /// resets like `compiledStep` so bench runs don't retrace).
    private var compiledVerifyStep: (([MLXArray]) -> [MLXArray])?
    /// Prefix snapshots are deliberately disjoint from a live speculative request. The compiled
    /// verify closure may survive reset for reuse; this flag tracks request state, not closure
    /// existence, and reset clears it.
    private var snapshotSpeculativeStateActive = false
    private var pendingNext: MLXArray? // [1] token id for the current position (lazy)
    private var cachedTokens = 0 // host-side position; caches' host mirror goes stale

    public init(
        model: any LanguageModel,
        reserve: Int = KVTunerCandidateRuntimeContract.cacheReserveTokens,
        kvCache: KVCacheKind = .fp16,
        affineAttentionMode: AffineKVAttentionMode = .materialize,
        kvarnAttentionMode: KVarNKVAttentionMode = .materialize,
        kvarnStorageDType: KVarNKVScalarDType? = nil,
        compileStep: Bool = true
    ) {
        let executionMode = kvCache.executionMode(
            requestingCompilation: compileStep,
            kvarnAttentionMode: kvarnAttentionMode)
        self.model = model
        self.reserve = reserve
        self.kvCacheKind = kvCache
        self.affineAttentionMode = affineAttentionMode
        self.kvarnAttentionMode = kvarnAttentionMode
        self.kvarnStorageDType = kvarnStorageDType
        self.compileStepEnabled = executionMode == .compiled
        self.executionMode = executionMode
    }

    public mutating func prefill(_ promptTokens: [Int]) -> Int {
        let first = prefillCore(promptTokens)
        return armLookahead(from: first)
    }

    /// Cold fp16 prefill that also captures the exact prompt-only state before the first
    /// generated token is submitted to the compiled decode step.
    ///
    /// This is a consuming request-start transition, not a validation probe. Any thrown
    /// validation or capture error resets the live decoder in place so the cache-enabled
    /// request fails closed and a later request can take the ordinary cold path.
    public mutating func prefillCapturingPromptSnapshot(
        _ promptTokens: [Int]
    ) throws -> (
        firstToken: Int,
        snapshot: CompiledMLXDecoderSnapshot
    ) {
        do {
            try validateSnapshotRoute()
            guard cachedTokens == 0, pendingNext == nil else {
                throw CompiledMLXDecoderSnapshotError.invalidPipelineState
            }
            try Self.validatePromptTokens(promptTokens)

            let first = prefillCore(promptTokens)
            let firstToken = try Self.validatedTokenID(first)
            let snapshot = try captureSnapshot(nextTokenID: firstToken)
            let emitted = armLookahead(from: first)
            precondition(
                emitted == firstToken,
                "captured prompt token changed while arming lookahead")
            return (firstToken, snapshot)
        } catch {
            reset()
            throw error
        }
    }

    /// Capture the exact committed context plus its pending greedy token.
    ///
    /// Plain decode keeps every emitted token committed in KV and `pendingNext` as the next
    /// model pick. Both parts are required to resume without re-evaluating the prefix.
    public func captureContinuationSnapshot()
        throws -> CompiledMLXDecoderSnapshot
    {
        try validateSnapshotRoute()
        guard !caches.isEmpty, compiledStep != nil else {
            throw CompiledMLXDecoderSnapshotError.decoderNotInitialized
        }
        guard cachedTokens > 0, let pendingNext else {
            throw CompiledMLXDecoderSnapshotError.invalidPipelineState
        }
        let nextTokenID = try Self.validatedTokenID(pendingNext)
        return try captureSnapshot(nextTokenID: nextTokenID)
    }

    /// Restore an exact dense prefix, evaluate only a new tail, and re-enter the ordinary
    /// submit-first decode pipeline.
    ///
    /// This is a consuming request-start transition, not a validation probe. Any thrown
    /// validation or restore error resets the live decoder in place; callers must fail the
    /// current cache-enabled request rather than continue an earlier decode pipeline.
    public mutating func prefillRestoredPrefix(
        _ snapshot: CompiledMLXDecoderSnapshot,
        tailTokens: [Int]
    ) throws -> Int {
        do {
            try validateSnapshotRoute()
            guard !caches.isEmpty, compiledStep != nil else {
                throw CompiledMLXDecoderSnapshotError.decoderNotInitialized
            }
            guard cachedTokens == 0, pendingNext == nil else {
                throw CompiledMLXDecoderSnapshotError.invalidPipelineState
            }
            try Self.validateTailTokens(tailTokens)
            try Self.validate(
                snapshot: snapshot, expectedLayerCount: caches.count)

            let (prefixAndTail, prefixOverflow) =
                snapshot.logicalTokenCount.addingReportingOverflow(
                    tailTokens.count)
            let (requiredCapacity, capacityOverflow) =
                prefixAndTail.addingReportingOverflow(1)
            guard !prefixOverflow, !capacityOverflow,
                requiredCapacity > 0,
                requiredCapacity <= Int(Int32.max)
            else {
                throw CompiledMLXDecoderSnapshotError.byteCountOverflow
            }
            var targetCapacity = caches[0].capacity
            while requiredCapacity > targetCapacity {
                let (grown, overflow) =
                    targetCapacity.addingReportingOverflow(chunk)
                guard !overflow, grown <= Int(Int32.max) else {
                    throw CompiledMLXDecoderSnapshotError.byteCountOverflow
                }
                targetCapacity = grown
            }

            let denseCaches = try denseSnapshotCaches()
            guard denseCaches.allSatisfy({
                $0.capacity == denseCaches[0].capacity
            }) else {
                throw CompiledMLXDecoderSnapshotError
                    .invalidSnapshotMetadata
            }
            var restorePlans: [CompiledKVCacheRestorePlan] = []
            restorePlans.reserveCapacity(denseCaches.count)
            for (layer, cache) in denseCaches.enumerated() {
                guard snapshot.layerSnapshots[layer].tokenCount
                    == snapshot.logicalTokenCount
                else {
                    throw CompiledMLXDecoderSnapshotError
                        .layerTokenCountMismatch(layer: layer)
                }
                restorePlans.append(
                    try cache.prepareRestore(
                        from: snapshot.layerSnapshots[layer],
                        targetCapacity: targetCapacity))
            }
            for (cache, plan) in zip(denseCaches, restorePlans) {
                cache.applyPreparedRestore(plan)
            }

            cachedTokens = snapshot.logicalTokenCount
            pendingNext = nil

            let current: MLXArray
            if tailTokens.isEmpty {
                current = MLXArray([Int32(snapshot.nextTokenID)])
            } else {
                let ids = MLXArray(
                    tailTokens.map { Int32($0) }
                ).reshaped([1, tailTokens.count])
                let logits = model(ids, cache: caches)
                current = Self.greedyTokenOrInvalidSentinel(
                    logits[0..., -1, 0...])
                cachedTokens += tailTokens.count
            }
            let currentToken = try Self.validatedTokenID(current)
            let emitted = armLookahead(from: current)
            precondition(
                emitted == currentToken,
                "restored prefix token changed while arming lookahead")
            return emitted
        } catch {
            reset()
            throw error
        }
    }

    /// Shared prefill: caches allocated/grown, uncompiled prompt forward, compiled step
    /// created. Afterwards the KV holds exactly the prompt and the returned `[1]` array is
    /// the first generated token — NOT yet consumed by any forward. `prefill` adds the
    /// submit-first lookahead on top; the spec-decode loop starts from here directly
    /// (its next forward depends on a host-side drafting decision, so there is nothing
    /// to submit ahead of the readback).
    private mutating func prefillCore(_ promptTokens: [Int]) -> MLXArray {
        let promptLength = promptTokens.count
        if caches.isEmpty {
            let layerCount = model.newCache(parameters: nil).count
            let cap = ((promptLength + reserve + chunk - 1) / chunk) * chunk
            do {
                caches = try kvCacheKind.makeCaches(
                    layerCount: layerCount,
                    capacity: cap,
                    affineAttentionMode: affineAttentionMode,
                    kvarnAttentionMode: kvarnAttentionMode,
                    kvarnStorageDType: kvarnStorageDType)
            } catch {
                preconditionFailure(
                    "KV-cache policy does not match the loaded model: \(error)")
            }
        }
        while promptLength + 1 > caches[0].capacity {
            for cache in caches { cache.grow(by: chunk) }
        }

        let logits: MLXArray
        if shouldChunkPrefillForSplitAttention,
            promptLength > Self.splitAttentionPrefillChunkSize
        {
            var lastChunkLogits: MLXArray?
            var start = 0
            while start < promptLength {
                let end = min(
                    start + Self.splitAttentionPrefillChunkSize,
                    promptLength)
                let chunkTokens = Array(promptTokens[start..<end])
                let ids = MLXArray(chunkTokens).reshaped([
                    1, chunkTokens.count,
                ])
                let chunkLogits = model(ids, cache: caches)
                // Bound the lazy graph and make each cache mutation visible before the next
                // chunk reads the in-graph offset and packed buffers.
                eval(chunkLogits)
                lastChunkLogits = chunkLogits
                start = end
            }
            guard let lastChunkLogits else {
                preconditionFailure("prefill requires at least one prompt token")
            }
            logits = lastChunkLogits
        } else {
            let ids = MLXArray(promptTokens).reshaped([1, promptLength])
            logits = model(ids, cache: caches)
        }
        let first = Self.greedyTokenOrInvalidSentinel(
            logits[0..., -1, 0...]) // [1]
        cachedTokens = promptLength

        if compiledStep == nil {
            // Weights are captured as constants; ONLY the caches are mutable state.
            let model = self.model
            let caches = self.caches
            let step: ([MLXArray]) -> [MLXArray] = { args in
                // A prior invalid sentinel must never become an embedding lookup. The caller
                // fails hard before emitting it; clamping only keeps submit-first lookahead from
                // touching an invalid index while that scalar is read back.
                let y = maximum(args[0], MLXArray(Int32(0)))
                    .reshaped([1, 1])
                let logits = model(y, cache: caches)
                return [Self.greedyTokenOrInvalidSentinel(
                    logits[0..., -1, 0...])]
            }
            // Uncompiled variant is the same closure, graph-built fresh every call.
            compiledStep = compileStepEnabled ? compile(inputs: caches, outputs: caches, step) : step
        }
        return first
    }

    private var shouldChunkPrefillForSplitAttention: Bool {
        if affineAttentionMode == .splitQuantizedMM { return true }
        guard case .kvarn = kvCacheKind else { return false }
        return kvarnAttentionMode == .splitQuantizedMM
    }

    private mutating func armLookahead(from current: MLXArray) -> Int {
        guard let compiledStep else {
            fatalError("CompiledMLXDecoder lookahead missing compiled step")
        }
        while cachedTokens + 1 > caches[0].capacity {
            for cache in caches { cache.grow(by: chunk) }
        }
        // submit-first: the compiled forward for the position after `current` is in
        // flight before we block reading `current` back to the CPU.
        let next = compiledStep([current])[0]
        asyncEval(next)
        pendingNext = next
        cachedTokens += 1
        return Self.requireValidGreedyToken(current)
    }

    private func validateSnapshotRoute() throws {
        guard kvCacheKind == .fp16 else {
            throw CompiledMLXDecoderSnapshotError.unsupportedCacheKind
        }
        guard !snapshotSpeculativeStateActive else {
            throw CompiledMLXDecoderSnapshotError
                .speculativeStateUnsupported
        }
    }

    private func denseSnapshotCaches() throws -> [CompiledKVCache] {
        let denseCaches = caches.compactMap { $0 as? CompiledKVCache }
        guard denseCaches.count == caches.count else {
            throw CompiledMLXDecoderSnapshotError.unsupportedCacheKind
        }
        return denseCaches
    }

    private func captureSnapshot(
        nextTokenID: Int
    ) throws -> CompiledMLXDecoderSnapshot {
        guard cachedTokens > 0, cachedTokens <= Int(Int32.max) else {
            throw CompiledMLXDecoderSnapshotError.invalidPipelineState
        }
        guard nextTokenID >= 0, Int32(exactly: nextTokenID) != nil else {
            throw CompiledMLXDecoderSnapshotError.invalidNextToken
        }
        let denseCaches = try denseSnapshotCaches()
        guard !denseCaches.isEmpty else {
            throw CompiledMLXDecoderSnapshotError.decoderNotInitialized
        }
        let layerSnapshots = try denseCaches.map {
            try $0.captureSnapshot(logicalTokenCount: cachedTokens)
        }
        let snapshot = CompiledMLXDecoderSnapshot(
            logicalTokenCount: cachedTokens,
            nextTokenID: nextTokenID,
            layerSnapshots: layerSnapshots)
        try Self.validate(
            snapshot: snapshot, expectedLayerCount: denseCaches.count)
        return snapshot
    }

    private static func validate(
        snapshot: CompiledMLXDecoderSnapshot,
        expectedLayerCount: Int
    ) throws {
        guard snapshot.layerCount == expectedLayerCount,
            snapshot.layerSnapshots.count == expectedLayerCount
        else {
            throw CompiledMLXDecoderSnapshotError.layerCountMismatch
        }
        guard snapshot.logicalTokenCount > 0,
            snapshot.logicalTokenCount <= Int(Int32.max)
        else {
            throw CompiledMLXDecoderSnapshotError.invalidSnapshotMetadata
        }
        guard snapshot.nextTokenID >= 0,
            Int32(exactly: snapshot.nextTokenID) != nil
        else {
            throw CompiledMLXDecoderSnapshotError.invalidNextToken
        }
        guard snapshot.arrayNBytes >= 0,
            snapshot.controlNBytes > 0,
            snapshot.totalNBytes > 0
        else {
            throw CompiledMLXDecoderSnapshotError.byteCountOverflow
        }

        var arrayBytes = 0
        for (layer, layerSnapshot) in snapshot.layerSnapshots.enumerated() {
            guard layerSnapshot.tokenCount == snapshot.logicalTokenCount else {
                throw CompiledMLXDecoderSnapshotError
                    .layerTokenCountMismatch(layer: layer)
            }
            let (next, overflow) = arrayBytes.addingReportingOverflow(
                layerSnapshot.totalNBytes)
            guard !overflow, layerSnapshot.totalNBytes > 0 else {
                throw CompiledMLXDecoderSnapshotError.byteCountOverflow
            }
            arrayBytes = next
        }
        let (expectedControl, controlOverflow) =
            snapshot.layerCount.multipliedReportingOverflow(
                by: MemoryLayout<CompiledKVCacheSnapshot>.stride)
        let (expectedControlWithToken, tokenOverflow) =
            expectedControl.addingReportingOverflow(
                MemoryLayout<Int32>.stride)
        let (expectedTotal, totalOverflow) =
            arrayBytes.addingReportingOverflow(expectedControlWithToken)
        guard !controlOverflow, !tokenOverflow, !totalOverflow else {
            throw CompiledMLXDecoderSnapshotError.byteCountOverflow
        }
        guard snapshot.arrayNBytes == arrayBytes,
            snapshot.controlNBytes == expectedControlWithToken,
            snapshot.totalNBytes == expectedTotal
        else {
            throw CompiledMLXDecoderSnapshotError.invalidSnapshotMetadata
        }
    }

    private static func validatePromptTokens(
        _ tokens: [Int]
    ) throws {
        guard !tokens.isEmpty,
            tokens.allSatisfy({ $0 >= 0 && Int32(exactly: $0) != nil })
        else {
            throw CompiledMLXDecoderSnapshotError.invalidPrompt
        }
    }

    private static func validateTailTokens(
        _ tokens: [Int]
    ) throws {
        for (position, token) in tokens.enumerated() {
            guard token >= 0, Int32(exactly: token) != nil else {
                throw CompiledMLXDecoderSnapshotError
                    .invalidTailToken(position: position)
            }
        }
    }

    private static func validatedTokenID(
        _ token: MLXArray
    ) throws -> Int {
        guard token.shape == [1], token.dtype == .int32 else {
            throw CompiledMLXDecoderSnapshotError.invalidNextToken
        }
        let value = Int(token.item(Int32.self))
        guard value >= 0 else {
            throw CompiledMLXDecoderSnapshotError.invalidNextToken
        }
        return value
    }

    public mutating func step(last: Int) -> Int {
        guard let next = pendingNext, let compiledStep else {
            fatalError("CompiledMLXDecoder.step called before prefill")
        }
        if cachedTokens + 1 > caches[0].capacity {
            for cache in caches { cache.grow(by: chunk) } // next call retraces once
        }
        let following = compiledStep([next])[0]
        asyncEval(following)
        pendingNext = following
        cachedTokens += 1
        return Self.requireValidGreedyToken(next)
    }

    /// The speculative-decoding loop (PLD first). High-yield rounds run one verify forward
    /// over `[last] + draft`, accept the longest target-confirmed prefix plus the bonus token,
    /// and roll KV back by the rejected count. Empty-draft and gate-disabled rounds retain the
    /// plain loop's submit-first `pendingNext` pipeline instead of synchronously forwarding and
    /// reading back one token at a time.
    ///
    /// There are two explicit cache invariants:
    /// - pipelined: committed context is in KV and `pendingNext` is the target's next pick;
    /// - speculative: the last emitted token is `lastArr`, not yet in KV, and no pick is pending.
    /// Re-entering speculation verifies `draft` against the prefetched pick plus the draft
    /// forward's rows, then exits in the efficient speculative invariant. Falling back from a
    /// speculative round seeds the two-deep plain pipeline once; later fallback steps call the
    /// exact same `step(last:)` implementation as PLD-off.
    ///
    /// THE HEADLINE PROPERTY: at temp 0 the emitted stream is byte-identical to the plain
    /// greedy loop — a drafted token is emitted only when it EQUALS the model's own argmax
    /// at its position, the bonus token IS the model's argmax, and `SpecEmit.trim` replays
    /// the plain loop's budget/eos stopping rules over each batch. Speculation changes how
    /// many tokens one forward emits, never which tokens.
    ///
    public mutating func generateSpec(
        prompt: [Int], maxTokens: Int, eos: Int, spec: SpecDecodeConfig
    ) -> (
        tokens: [Int], submitTime: Double, tokenTimes: [Double],
        prefillDurationSeconds: Double?, stats: SpecDecodeStats
    ) {
        precondition(
            kvCacheKind.supportsSpecDecode,
            "speculative decoding is not qualified for the selected KV-cache tier")
        snapshotSpeculativeStateActive = true
        var stats = SpecDecodeStats()
        let submitTime = Date().timeIntervalSinceReferenceDate
        guard maxTokens > 0 else {
            return ([], submitTime, [], nil, stats)
        }

        // Start in the same submit-first state as the plain loop. A cold request can therefore
        // stay on the base pipeline from token one; a hot request pays one transition verify,
        // then remains in the one-forward-per-round speculative invariant.
        let prefillStartedAt = ProcessInfo.processInfo.systemUptime
        var last = prefill(prompt)
        let prefillDurationSeconds =
            ProcessInfo.processInfo.systemUptime - prefillStartedAt
        let prefillEnd = Date().timeIntervalSinceReferenceDate
        var lastArr: MLXArray? = nil
        var tokens = [last]
        var times = [prefillEnd]
        var context = prompt + [last]
        var gate = spec.gate
        var done = tokens.count >= maxTokens || last == eos

        while !done {
            // Bounded backward scan (Phase-1 flag #3): PLD is O(scanned) per call.
            let draft = gate.isEnabled
                ? spec.drafter.propose(
                    context: Array(context.suffix(spec.lookback)), maxDraft: spec.maxDraft)
                : []

            let emitted: [Int]
            if draft.isEmpty {
                if !gate.isEnabled { stats.gateDisabledSteps += 1 }
                stats.normalSteps += 1
                if pendingNext != nil {
                    // Already on the base loop's pipeline: submit the following forward before
                    // reading this token, exactly as PLD-off does.
                    last = step(last: last)
                } else {
                    // One-time speculative -> pipelined transition. The current `lastArr` is
                    // not in KV, so consume it, submit the following token too, and only then
                    // read back the next emitted token. Subsequent cold steps use `step` above.
                    guard let current = lastArr, let compiledStep else {
                        fatalError("speculative fallback missing current token or compiled step")
                    }
                    while cachedTokens + 2 > caches[0].capacity {
                        for cache in caches { cache.grow(by: chunk) }
                    }
                    let next = compiledStep([current])[0]
                    let following = compiledStep([next])[0]
                    asyncEval(following)
                    pendingNext = following
                    cachedTokens += 2
                    last = Self.requireValidGreedyToken(next)
                    lastArr = nil
                }
                // Phase-1 flag #2: the gate is fed on EVERY step (cooldown clock).
                gate.record(accepted: 0)
                emitted = [last]
            } else {
                var verifyDraft = draft
                if spec.compiledVerify && verifyDraft.count < spec.maxDraft {
                    // Fixed-K pad so the compiled verify replays without retracing.
                    // Exact-safe: a padded token is emitted only if it EQUALS the model's
                    // own argmax at its position (the accept-walk's only criterion).
                    verifyDraft += Array(
                        repeating: verifyDraft.last!, count: spec.maxDraft - verifyDraft.count)
                }
                let wasPipelined = pendingNext != nil
                let verifyInput = wasPipelined ? verifyDraft : [last] + verifyDraft
                let n = verifyInput.count
                while cachedTokens + n > caches[0].capacity {
                    for cache in caches { cache.grow(by: chunk) }
                }
                let ids = MLXArray(verifyInput.map(Int32.init)).reshaped([1, n])
                let verifyArgmax: MLXArray // target pick AFTER each forwarded position
                if spec.compiledVerify {
                    if compiledVerifyStep == nil {
                        let model = self.model
                        let caches = self.caches
                        let verify: ([MLXArray]) -> [MLXArray] = { args in
                            let logits = model(args[0], cache: caches)
                            return [Self.greedyTokenOrInvalidSentinel(logits[0])]
                        }
                        compiledVerifyStep = compile(inputs: caches, outputs: caches, verify)
                    }
                    verifyArgmax = compiledVerifyStep!([ids])[0]
                } else {
                    let logits = model(ids, cache: caches)
                    verifyArgmax = Self.greedyTokenOrInvalidSentinel(logits[0])
                }
                let picksAfterInput = verifyArgmax.asType(.int32).asArray(Int32.self).map(Int.init)
                precondition(
                    picksAfterInput.allSatisfy { $0 >= 0 },
                    "non-finite logits reached speculative decode")
                let prefetched = pendingNext
                let prefetchedPick = prefetched.map {
                    Self.requireValidGreedyToken($0)
                }
                cachedTokens += n // the verify forward appended n rows

                let accepted: Int
                let bonus: Int
                let bonusArr: MLXArray
                let keep: Int
                if let prefetched, let prefetchedPick {
                    // Pipelined entry: KV already contains `last`; the verify forwarded only
                    // draft rows. Keep exactly the accepted draft prefix. The target's first
                    // pick came from the in-flight base step, shifting later picks by one.
                    (accepted, bonus) = SpecAccept.walk(
                        draft: verifyDraft,
                        prefetched: prefetchedPick,
                        verifyArgmaxAfterDraft: picksAfterInput)
                    keep = cachedTokens - n + accepted
                    bonusArr = accepted == 0
                        ? prefetched
                        : verifyArgmax[accepted - 1].reshaped([1])
                } else {
                    // Speculative entry: KV lacks `last`; `[last] + draft` produced the canonical
                    // K+1 picks. Keep `last` plus the accepted draft prefix.
                    (accepted, bonus) = SpecAccept.walk(
                        draft: verifyDraft, verifyArgmax: picksAfterInput)
                    keep = cachedTokens - n + 1 + accepted
                    bonusArr = verifyArgmax[accepted].reshaped([1])
                }

                // Rejected rows become unreachable and are overwritten by the next update.
                if keep < cachedTokens {
                    for cache in caches { cache.truncate(to: keep) }
                    cachedTokens = keep
                }
                stats.verifySteps += 1
                stats.drafted += verifyDraft.count
                stats.accepted += accepted
                gate.record(accepted: accepted)
                last = bonus
                lastArr = bonusArr
                pendingNext = nil
                emitted = Array(verifyDraft.prefix(accepted)) + [bonus]
            }

            let (emit, stop) = SpecEmit.trim(
                emitted: emitted, alreadyEmitted: tokens.count, maxTokens: maxTokens, eos: eos)
            tokens += emit
            context += emit
            let t = Date().timeIntervalSinceReferenceDate
            for _ in emit { times.append(t) }
            done = stop
        }
        return (
            tokens, submitTime, times, prefillDurationSeconds, stats)
    }

    /// Reset caches IN PLACE (same MLXArray identities) so the compiled step function
    /// stays valid across bench runs — no rebuild, no retrace, no cross-actor resend.
    public mutating func reset() {
        for cache in caches { cache.resetInPlace() }
        snapshotSpeculativeStateActive = false
        pendingNext = nil
        cachedTokens = 0
    }

    /// The token scalar is already the decode loop's synchronization point. Folding the
    /// all-finite reduction into the lazy graph preserves that one readback while guaranteeing
    /// NaN/Inf can never silently collapse to token zero through `argMax`.
    static func greedyTokenOrInvalidSentinel(
        _ logits: MLXArray
    ) -> MLXArray {
        let token = argMax(logits, axis: -1).asType(.int32)
        return MLX.where(
            isFinite(logits).all(),
            token,
            MLXArray(Int32(-1)))
    }

    private static func requireValidGreedyToken(
        _ token: MLXArray
    ) -> Int {
        let value = token.item(Int.self)
        precondition(value >= 0, "non-finite logits reached greedy decode")
        return value
    }

    /// Engagement probe: the IN-GRAPH cached-token count from layer 0's TurboQuant cache.
    /// `offsetArr` advances inside the (compiled) step itself, so this is proof the
    /// quantized cache's update actually ran — a silently-substituted fp16 cache returns
    /// nil. Forces a sync readback; call after a run, never in the hot loop.
    public func turboQuantCachedTokens() -> Int? {
        guard let cache = caches.first as? TurboQuantKVCache else { return nil }
        return Int(cache.offsetArr.item(Int32.self))
    }

    /// Post-run affine engagement, geometry, and exact storage bytes. The capture happens
    /// while this decoder and its MLX arrays remain in the inference actor; only a Sendable
    /// scalar snapshot crosses back to the harness.
    public func affineKVTelemetry() -> AffineKVCacheTelemetry? {
        guard case .affine(let tier) = kvCacheKind else { return nil }
        let affineCaches = caches.compactMap { $0 as? AffineKVCache }
        precondition(
            affineCaches.count == caches.count,
            "affine tier requested but the decoder contains a different cache type")
        return AffineKVCacheTelemetry.capture(tier: tier, caches: affineCaches)
    }

    /// Post-run exact schedule identity plus heterogeneous affine-array bytes. The schedule
    /// digest is part of `KVCacheKind`, so decoder reuse cannot cross artifact boundaries.
    public func kvtunerKVTelemetry() -> KVTunerKVCacheTelemetry? {
        guard case .kvtuner(let selection) = kvCacheKind else { return nil }
        let affineCaches = caches.compactMap { $0 as? AffineKVCache }
        precondition(
            affineCaches.count == caches.count,
            "KVTuner decoder contains a different cache type")
        do {
            return try KVTunerKVCacheTelemetry.capture(
                selection: selection, caches: affineCaches)
        } catch {
            preconditionFailure(
                "KVTuner decoder telemetry does not match its schedule: \(error)")
        }
    }

    /// Post-run preselection-candidate receipt. The candidate policy is a distinct cache kind
    /// with no user-facing parser route, so a decoder cannot be reused across candidate digests
    /// and only authenticated search orchestration can request this evidence.
    public func kvtunerCandidateKVTelemetry()
        -> KVTunerCandidateKVCacheTelemetry?
    {
        guard case .kvtunerCandidate(let policy) = kvCacheKind else {
            return nil
        }
        let affineCaches = caches.compactMap { $0 as? AffineKVCache }
        precondition(
            affineCaches.count == caches.count,
            "KVTuner candidate decoder contains a different cache type")
        do {
            return try KVTunerCandidateKVCacheTelemetry.capture(
                policy: policy, caches: affineCaches)
        } catch {
            preconditionFailure(
                "KVTuner candidate telemetry does not match its policy: \(error)")
        }
    }

    /// Post-run KVarN engagement, runtime cell, geometry, and exact storage bytes. All MLX
    /// arrays remain inside the decoder's inference actor; only this scalar `Sendable` snapshot
    /// crosses into harness evidence.
    public func kvarnKVTelemetry() -> KVarNKVCacheTelemetry? {
        guard case .kvarn(let cell) = kvCacheKind else { return nil }
        let kvarnCaches = caches.compactMap { $0 as? KVarNKVCache }
        precondition(
            kvarnCaches.count == caches.count,
            "KVarN tier requested but the decoder contains a different cache type")
        let telemetry = KVarNKVCacheTelemetry
            .capture(caches: kvarnCaches)
            .withExecutionMode(executionMode)
        precondition(
            telemetry.tier == cell.tier
                && telemetry.iterations == cell.iterations
                && telemetry.executionMode == executionMode,
            "KVarN decoder telemetry does not match its requested runtime cell")
        return telemetry
    }
}

private extension KVarNKVCacheTelemetry {
    func withExecutionMode(_ executionMode: KVCacheExecutionMode)
        -> KVarNKVCacheTelemetry
    {
        KVarNKVCacheTelemetry(
            tier: tier,
            iterations: iterations,
            executionMode: executionMode,
            cachedTokens: cachedTokens,
            completedTileCount: completedTileCount,
            compressedTokens: compressedTokens,
            layerCount: layerCount,
            capacityTokens: capacityTokens,
            packedTileSlots: packedTileSlots,
            sequences: sequences,
            kvHeadCount: kvHeadCount,
            headDimension: headDimension,
            sourceKeyDTypes: sourceKeyDTypes,
            sourceValueDTypes: sourceValueDTypes,
            storageKeyDType: storageKeyDType,
            storageValueDType: storageValueDType,
            ingressNormalizationApplied: ingressNormalizationApplied,
            metadataScalarBytes: metadataScalarBytes,
            payloadBytes: payloadBytes,
            metadataBytes: metadataBytes,
            alignmentPaddingBytes: alignmentPaddingBytes,
            fp16SinkBytes: fp16SinkBytes,
            fp16TailBytes: fp16TailBytes,
            controlBytes: controlBytes,
            materializationWorkspaceBytes: materializationWorkspaceBytes,
            normalizationWorkspaceBytes: normalizationWorkspaceBytes,
            attentionWorkspaceBytes: attentionWorkspaceBytes,
            workspaceBytes: workspaceBytes,
            attentionOperation: attentionOperation)
    }
}
