import Foundation
import HarnessCore
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import ServingCore
import SpikeCore
import Tokenizers

// MARK: - Contract
//
// Measures the real sampled-MTP per-step acceptance rate `a`, the decide()-wall-time `D`, the
// target-model scalar step wall time `T`, the drafter-forward-cost fraction `delta`, and an
// INDEPENDENTLY re-derived `Sigma_x min(p(x), q(x))` per step for `qwen4_exp` (Qwen3.8-Flash-Next),
// against a real checkpoint plus its in-checkpoint MTP drafter, driven by
// `MTPSpeculativeTokenIterator` with a sampled block decision provider supplied.
//
// Contract: `docs/task-inbox/2026-09-09-sampled-mtp-acceptance-rate-PREDECLARATION.md`. Every
// quantity that document requires is measured IN THIS RUN, on the SAME release build/host/session
// -- never inherited from a prior measurement.
//
// `requiresGreedySampling` is never touched: supplying an eligible
// `SampledMTPBlockRuntimeDeciding` provider is what makes `providerIsEligible` true inside
// `MTPSpeculativeTokenIterator.init`, which is what keeps `initialPassthroughReason` `nil`
// (`MTPSpeculativeTokenIterator.swift:171-186`) -- verified directly at that source location, not
// assumed.

// MARK: - Pinned structural constants (qwen4_exp in-checkpoint MTP)

/// the in-checkpoint MTP draft model's `maximumBlockSize` is pinned to `3` (in-checkpoint MTP model
/// source, line 812), so
/// `MTPSpeculativeTokenIterator.init`'s own `Swift.min(blockSize, drafter.maximumBlockSize ??
/// blockSize)` clamps to this value regardless of what is requested. Passing it explicitly (rather
/// than a larger number that would silently clamp) documents the pin instead of hiding it. Mirrors
/// `inCheckpointMTPStartupGateBlockSize` in `MLXScalarServing.swift` (not imported here -- that
/// symbol is internal to `SpikeServingAdapters` -- but pinned to the identical value for the
/// identical reason).
let inCheckpointSampledMTPBlockSize = 3

/// `numDraft = blockSize - 1 = 2` -- the number of drafted (non-bonus) tokens per block, i.e. the
/// number of sequential accept/reject steps `decide()` walks per block.
let inCheckpointSampledMTPNumDraft = inCheckpointSampledMTPBlockSize - 1

/// Number of sequential drafter FORWARD passes charged to one block's `draftPhaseSeconds`
/// (`MTPSpeculativeTokenIterator.swift`'s `telemetry.recordDraftBlock(seconds: draftPhaseSeconds)`,
/// called once per block with the SUM of two disjoint timed windows): the draft-loop forwards
/// (the in-checkpoint MTP model's draft-block loop, model source ~line 1160/1175 -- runs
/// `blockSize - 2` iterations, one forward each) plus exactly ONE `commitDrafterState` seed forward
/// (source line 1059) that runs AFTER `decide()` returns but is still inside the second timed
/// window (`MTPSpeculativeTokenIterator.swift:833-834`). Derived from its two structural
/// components rather than hardcoded, then asserted equal to `numDraft` below -- for the pinned
/// `blockSize = 3` this is `1 + 1 = 2`, which coincides with `numDraft` (`blockSize - 1`) but is a
/// SEPARATE fact (draftPhaseSeconds' forward count, not the decide()-time accept/reject step
/// count) that would diverge from `numDraft` at a different `blockSize`.
let inCheckpointSampledMTPDraftLoopForwardsPerBlock = inCheckpointSampledMTPBlockSize - 2
let inCheckpointSampledMTPCommitSeedForwardsPerBlock = 1
let inCheckpointSampledMTPDrafterForwardsPerBlock =
    inCheckpointSampledMTPDraftLoopForwardsPerBlock + inCheckpointSampledMTPCommitSeedForwardsPerBlock

/// Asserts the structural relationship documented on `inCheckpointSampledMTPDrafterForwardsPerBlock`
/// above. NOT a top-level `precondition()` call -- Swift only permits bare executable statements at
/// top level in a `main.swift`-style entry file, and this file is not one. Also NOT a lazily-
/// initialized top-level `let` closure -- an unreferenced global initializer may never run at all,
/// silently defeating the assertion. Instead this is an ordinary function, called explicitly from
/// `runInCheckpointSampledMTPAcceptance` before `inCheckpointSampledMTPDrafterForwardsPerBlock` is
/// used in the `deltaPerForward` derivation, so it is guaranteed to execute on every real run.
func inCheckpointSampledMTPAssertDrafterForwardsPerBlockRelationship() {
    precondition(
        inCheckpointSampledMTPDrafterForwardsPerBlock == inCheckpointSampledMTPNumDraft,
        "drafter-forwards-per-block (\(inCheckpointSampledMTPDrafterForwardsPerBlock)) no longer "
            + "coincides with numDraft (\(inCheckpointSampledMTPNumDraft)) at blockSize="
            + "\(inCheckpointSampledMTPBlockSize) -- the per-forward `delta` derivation "
            + "(cleanDraftBlockSeconds / cleanDraftBlockCount / this constant) assumed they matched; "
            + "re-derive `delta` explicitly from the two structural components above instead of "
            + "reusing `numDraft`")
}

/// The predeclaration's sample-size floor (>= 500 proposed draft tokens, >= 250 blocks, >= 5
/// prompts, over the CLEAN pooled sample -- "Sample size and reporting").
let inCheckpointSampledMTPRequiredProposedDraftTokenCount = 500
let inCheckpointSampledMTPRequiredBlockCount = 250
let inCheckpointSampledMTPRequiredCleanPromptCount = 5

/// The only serve-eligible in-checkpoint MTP artifact today (`FastMLXInCheckpointMTPSelection`,
/// `ServingCore/FastMLXServeArguments.swift`). This harness measures against the SAME pinned
/// namespace/revision/expected-key-count production would load -- never a second, independently
/// invented pin that could silently drift from the deployed artifact's own lock.
let inCheckpointSampledMTPArtifactSelection = FastMLXInCheckpointMTPSelection.converted4Bit

// MARK: - CLI arguments

enum InCheckpointSampledMTPAcceptanceProviderKind: String, Equatable, Sendable {
    case seeded
    case nondeterministic
}

struct InCheckpointSampledMTPAcceptanceArguments: Equatable, Sendable {
    let modelPath: String
    let ngramOffloadPlanPath: String
    let promptsFilePath: String
    let maxTokens: Int
    let seed: UInt64
    let provider: InCheckpointSampledMTPAcceptanceProviderKind
    let outputJSONPath: String
    /// `--top-p`/`--top-k`, optional, defaulting to `1`/`0` -- the untruncated identity, i.e. every
    /// existing invocation that omits these two flags behaves EXACTLY as before this option was
    /// added. `temperature` and `minP` are NOT exposed as flags: `supports()`
    /// (`SampledMTPBlockRuntimeBridge.swift`'s `sharedSampledMTPSupportsPredicate`) requires exactly
    /// `temperature == 1` and `minP == 0`, so a flag that could set either to anything else would
    /// only ever produce a run where the provider refuses and speculation never engages.
    let topP: Double
    let topK: Int
}

enum InCheckpointSampledMTPAcceptanceCLIError: Error, Equatable, CustomStringConvertible, Sendable {
    case missingFlag(String)
    case duplicateFlag(String)
    case unknownFlag(String)
    case missingValue(String)
    case unexpectedPositional
    case releaseBuildRequired
    case invalidMaxTokens(String)
    case invalidSeed(String)
    case invalidProvider(String)
    /// `supports()` requires `topP > 0 && topP <= 1` (`SampledMTPBlockRuntimeBridge.swift`'s
    /// `sharedSampledMTPSupportsPredicate`). Refused at parse time -- before any model load -- so a
    /// value outside that range never gets as far as silently producing a no-speculation run.
    case invalidTopP(String)
    /// `supports()` requires `topK >= 0`. Same parse-time refusal rationale as `invalidTopP`.
    case invalidTopK(String)
    case modelPathMustBeAbsolute
    case ngramOffloadPlanMustBeAbsolute
    case promptsFileMustBeAbsolute
    case promptsFileUnavailable(String)
    case promptsFileEmpty
    case outputJSONMustBeAbsolute
    case outputJSONMustBeNew
    case outputJSONWriteFailed
    /// Fires when ANY stream never speculated at all (`proposedCount == 0`): the provider was
    /// silently dropped (unsupported sampling parameters, or the drafter refused this prompt) and
    /// `MTPSpeculativeTokenIterator` ran the ENTIRE stream through single-token passthrough. This
    /// is the predeclaration's single most important control -- a zero-block run must not read
    /// like a clean one -- so it aborts the whole CLI invocation rather than reporting 0/0 for one
    /// prompt among several.
    /// Fires ONLY for `promptIndex == 0`, immediately after that prompt's `MTPSpeculativeTokenIterator`
    /// is constructed and BEFORE a single token is generated: `passthroughReason` was already
    /// non-nil at construction time, meaning `providerIsEligible` was false at `init`
    /// (`MTPSpeculativeTokenIterator.swift:171-186`) -- e.g. the provider's stored
    /// `SampledMTPSamplingTruncation` did not match this run's own `parameters`, so `supports()`
    /// refused it. `passthroughReason` is nil at this exact point in EVERY legitimate engaged run
    /// (it is set unconditionally, and only, when `providerIsEligible` is false) -- so this check
    /// cannot spuriously fire on a healthy stream, and it catches a provider/truncation wiring
    /// defect before the cost of a full decode loop is spent, rather than only after prompt[0]'s
    /// entire stream has already run to completion.
    case providerWiringNotEngaged(passthroughReason: String)
    case zeroProposedDraftTokens(promptIndex: Int, passthroughReason: String?)
    /// Fires when a stream proposed tokens (so the earlier `zeroProposedDraftTokens` control did
    /// NOT catch it) but `decide()` never once succeeded -- e.g. every `decide()` call threw
    /// (previously: the unsatisfiable `expectedCapturedCount = needed + 1` bug, now closed by Fix
    /// 1). `mtpIterator.proposedCount` increments regardless of whether `decide()` throws (it is
    /// incremented unconditionally after the accept/reject block, not gated on success), so a
    /// zero-measured-block stream reads EXACTLY like a healthy one under `proposedCount > 0` alone
    /// -- this is the control that actually catches that failure mode.
    case zeroMeasuredBlocks(promptIndex: Int, proposedCount: Int, passthroughReason: String?)
    /// Every stream either had zero proposed tokens (already fatal above) or went sticky
    /// mid-stream, leaving nothing clean to pool `a`/`D`/`T`/`delta`/`sigmaMinPQ` from.
    case noCleanStreams
    /// Internal consistency failure in the draft-logit interception (`TeeingLogitSampler`): the
    /// number of raw draft logit rows captured since the last successful `decide()` does not equal
    /// `numDraft` -- see the file-header note above `InCheckpointSampledMTPTeeingLogitSampler` for
    /// the verified real call sequence this expects. This would only happen if the drafter's
    /// call-count shape (verified against the in-checkpoint MTP model source, lines 916, 964,
    /// 1059, 1160, 1175) changes in the future -- fails loudly rather than silently mispairing
    /// target and draft rows.
    case draftLogitCaptureShapeMismatch(expectedRows: Int, observedRows: Int)
    case targetDraftVocabularyWidthMismatch(expected: Int, actual: Int)
    case invalidLogits(String)
    /// The predeclaration's sample-size floor (>= 500 proposed draft tokens, >= 250 blocks, >= 5
    /// clean prompts -- `docs/task-inbox/2026-09-09-sampled-mtp-acceptance-rate-PREDECLARATION.md`,
    /// "Sample size and reporting") was not met by the CLEAN (non-excluded) pooled sample. Refuses
    /// to write `--output-json` rather than emit a JSON/PASS/verdict from an underpowered sample --
    /// e.g. a one-line prompts file with `--max-tokens 8` must not exit 0 with an ACCEPT/REJECT/
    /// GATED verdict from ~6 samples.
    case sampleSizeBelowPredeclaredFloor(
        proposedDraftTokenCount: Int, requiredProposedDraftTokenCount: Int,
        blockCount: Int, requiredBlockCount: Int,
        cleanPromptCount: Int, requiredCleanPromptCount: Int)
    /// The predeclaration's independent cross-check (`Sigma_x min(p(x), q(x))` vs. the empirical
    /// accepted/reached ratio) disagreed for at least one step index or the cross-step pooled
    /// sample: the empirical rate's 95% Wilson confidence interval did not contain the sigmaMinPQ
    /// conditioned mean. Refuses to write `--output-json` -- the predeclaration's central claim is
    /// that AGREEMENT between the two numbers is the evidence; printing both without comparing them
    /// is exactly the failure mode that let a mispairing defect ship silently before.
    case sigmaMinPQDisagreement(String)

    var description: String {
        switch self {
        case .missingFlag(let flag): return "missing required \(flag)"
        case .duplicateFlag(let flag): return "duplicate \(flag)"
        case .unknownFlag(let flag): return "unknown flag \(flag)"
        case .missingValue(let flag): return "\(flag) requires a value"
        case .unexpectedPositional: return "unexpected positional argument"
        case .releaseBuildRequired:
            return "qwen4exp-sampled-mtp-acceptance requires a Release build: a DEBUG build "
                + "measures this workload roughly 70x slower and its timing numbers (D, T, delta) "
                + "would be meaningless -- refusing rather than emitting misleading numbers"
        case .invalidMaxTokens(let raw): return "--max-tokens requires a positive integer; actual=\(raw)"
        case .invalidSeed(let raw): return "--seed requires a non-negative UInt64; actual=\(raw)"
        case .invalidProvider(let raw):
            return "--provider requires seeded|nondeterministic; actual=\(raw)"
        case .invalidTopP(let raw):
            return "--top-p requires a Double with 0 < topP <= 1 (what supports() accepts); "
                + "actual=\(raw)"
        case .invalidTopK(let raw):
            return "--top-k requires a non-negative Int (what supports() accepts); actual=\(raw)"
        case .modelPathMustBeAbsolute: return "--model-path must be an absolute path"
        case .ngramOffloadPlanMustBeAbsolute: return "--ngram-offload-plan must be an absolute path"
        case .promptsFileMustBeAbsolute: return "--prompts-file must be an absolute path"
        case .promptsFileUnavailable(let path): return "--prompts-file unavailable at \(path)"
        case .promptsFileEmpty: return "--prompts-file contains no non-empty lines"
        case .outputJSONMustBeAbsolute: return "--output-json must be an absolute path"
        case .outputJSONMustBeNew: return "--output-json must name a new file"
        case .outputJSONWriteFailed: return "failed to write --output-json"
        case .providerWiringNotEngaged(let passthroughReason):
            return "ABORT: prompt[0]'s speculative iterator was already in passthrough "
                + "(passthroughReason=\(passthroughReason)) IMMEDIATELY after construction, before "
                + "any token was generated -- the sampled MTP provider's truncation does not match "
                + "this run's own sampling parameters (or another construction-time ineligibility), "
                + "so supports() refused it; refusing rather than silently measuring a run with no "
                + "speculation at all"
        case .zeroProposedDraftTokens(let promptIndex, let passthroughReason):
            return "ABORT: prompt[\(promptIndex)] proposed ZERO draft tokens "
                + "(passthroughReason=\(passthroughReason ?? "nil")) -- the sampled MTP provider "
                + "was silently dropped for this stream; a zero-block run must not be reported as "
                + "a measurement"
        case .zeroMeasuredBlocks(let promptIndex, let proposedCount, let passthroughReason):
            return "ABORT: prompt[\(promptIndex)] proposed \(proposedCount) draft token(s) but "
                + "measured ZERO blocks (every decide() call failed; "
                + "passthroughReason=\(passthroughReason ?? "nil")) -- proposedCount alone reads "
                + "like a healthy stream here because it increments even when decide() throws; a "
                + "zero-measured-block stream must not be reported as a measurement"
        case .noCleanStreams:
            return "ABORT: every prompt either proposed zero draft tokens or went sticky "
                + "passthrough mid-stream -- no clean stream remains to compute a/D/T/delta/"
                + "sigmaMinPQ from"
        case .draftLogitCaptureShapeMismatch(let expectedRows, let observedRows):
            return "internal error: draft-logit interception captured \(observedRows) row(s) since "
                + "the last decide(), but expected exactly \(expectedRows) (this block's "
                + "leading-aligned seed-plus-draft rows -- see the file-header note above "
                + "InCheckpointSampledMTPTeeingLogitSampler) -- the assumed capture shape broke; "
                + "refusing to report a mispaired sigmaMinPQ"
        case .sampleSizeBelowPredeclaredFloor(
            let proposedDraftTokenCount, let requiredProposedDraftTokenCount,
            let blockCount, let requiredBlockCount,
            let cleanPromptCount, let requiredCleanPromptCount):
            var shortfalls: [String] = []
            if proposedDraftTokenCount < requiredProposedDraftTokenCount {
                shortfalls.append(
                    "proposedDraftTokenCount=\(proposedDraftTokenCount) < required "
                        + "\(requiredProposedDraftTokenCount)")
            }
            if blockCount < requiredBlockCount {
                shortfalls.append("blockCount=\(blockCount) < required \(requiredBlockCount)")
            }
            if cleanPromptCount < requiredCleanPromptCount {
                shortfalls.append(
                    "cleanPromptCount=\(cleanPromptCount) < required \(requiredCleanPromptCount)")
            }
            return "ABORT: sample size below the predeclared floor "
                + "(docs/task-inbox/2026-09-09-sampled-mtp-acceptance-rate-PREDECLARATION.md, "
                + "\"Sample size and reporting\": >= 500 proposed draft tokens, >= 250 blocks, "
                + ">= 5 prompts, all over the CLEAN pooled sample): " + shortfalls.joined(separator: "; ")
                + " -- refusing to write --output-json or report a verdict from an underpowered sample"
        case .sigmaMinPQDisagreement(let detail):
            return "ABORT: independent cross-check DISAGREEMENT -- the empirical per-step "
                + "accepted/reached ratio's 95% Wilson confidence interval did not contain the "
                + "sigmaMinPQ conditioned mean for at least one step or the cross-step pooled "
                + "sample: \(detail) -- refusing to write --output-json; the predeclaration's "
                + "central claim is that AGREEMENT between these two independently-derived numbers "
                + "is the evidence"
        case .targetDraftVocabularyWidthMismatch(let expected, let actual):
            return "target/draft logits vocabulary width mismatch: expected=\(expected) actual=\(actual)"
        case .invalidLogits(let detail): return "invalid logits: \(detail)"
        }
    }
}

func parseInCheckpointSampledMTPAcceptanceArguments(
    _ arguments: [String]
) throws -> InCheckpointSampledMTPAcceptanceArguments {
    let requiredFlags: Set<String> = [
        "--model-path", "--ngram-offload-plan", "--prompts-file", "--max-tokens", "--seed",
        "--provider", "--output-json",
    ]
    // `--top-p`/`--top-k` are OPTIONAL (see the doc comment on
    // `InCheckpointSampledMTPAcceptanceArguments.topP`/`.topK`) -- present in `allowed` (so they are
    // accepted at all) but deliberately absent from `requiredFlags`, so every existing invocation
    // that omits them keeps working unchanged.
    let optionalFlags: Set<String> = ["--top-p", "--top-k"]
    let allowed = requiredFlags.union(optionalFlags)
    var values: [String: String] = [:]
    var index = 0
    while index < arguments.count {
        let flag = arguments[index]
        guard flag.hasPrefix("--") else {
            throw InCheckpointSampledMTPAcceptanceCLIError.unexpectedPositional
        }
        guard allowed.contains(flag) else {
            throw InCheckpointSampledMTPAcceptanceCLIError.unknownFlag(flag)
        }
        guard values[flag] == nil else {
            throw InCheckpointSampledMTPAcceptanceCLIError.duplicateFlag(flag)
        }
        guard index + 1 < arguments.count, !arguments[index + 1].hasPrefix("--") else {
            throw InCheckpointSampledMTPAcceptanceCLIError.missingValue(flag)
        }
        values[flag] = arguments[index + 1]
        index += 2
    }
    for flag in requiredFlags where values[flag] == nil {
        throw InCheckpointSampledMTPAcceptanceCLIError.missingFlag(flag)
    }

    let modelPath = values["--model-path"]!
    guard modelPath.hasPrefix("/") else {
        throw InCheckpointSampledMTPAcceptanceCLIError.modelPathMustBeAbsolute
    }
    let ngramOffloadPlanPath = values["--ngram-offload-plan"]!
    guard ngramOffloadPlanPath.hasPrefix("/") else {
        throw InCheckpointSampledMTPAcceptanceCLIError.ngramOffloadPlanMustBeAbsolute
    }
    let promptsFilePath = values["--prompts-file"]!
    guard promptsFilePath.hasPrefix("/") else {
        throw InCheckpointSampledMTPAcceptanceCLIError.promptsFileMustBeAbsolute
    }
    let outputJSONPath = values["--output-json"]!
    guard outputJSONPath.hasPrefix("/") else {
        throw InCheckpointSampledMTPAcceptanceCLIError.outputJSONMustBeAbsolute
    }

    let rawMaxTokens = values["--max-tokens"]!
    guard let maxTokens = Int(rawMaxTokens), maxTokens > 0 else {
        throw InCheckpointSampledMTPAcceptanceCLIError.invalidMaxTokens(rawMaxTokens)
    }
    let rawSeed = values["--seed"]!
    guard let seed = UInt64(rawSeed) else {
        throw InCheckpointSampledMTPAcceptanceCLIError.invalidSeed(rawSeed)
    }
    let rawProvider = values["--provider"]!
    guard let provider = InCheckpointSampledMTPAcceptanceProviderKind(rawValue: rawProvider) else {
        throw InCheckpointSampledMTPAcceptanceCLIError.invalidProvider(rawProvider)
    }

    // Defaults are the untruncated identity (`topP = 1`, `topK = 0`) -- an invocation that omits
    // both flags measures EXACTLY what this CLI measured before they existed. Validated against
    // precisely what `supports()` (`sharedSampledMTPSupportsPredicate`) accepts, at parse time,
    // BEFORE any model load or measurement is spent: refusing here means a bad value never gets far
    // enough to construct a provider whose truncation cannot possibly match, which would otherwise
    // degrade to a silent no-speculation run (see `providerWiringNotEngaged`).
    let rawTopP = values["--top-p"] ?? "1"
    guard let topP = Double(rawTopP), topP > 0, topP <= 1 else {
        throw InCheckpointSampledMTPAcceptanceCLIError.invalidTopP(rawTopP)
    }
    let rawTopK = values["--top-k"] ?? "0"
    guard let topK = Int(rawTopK), topK >= 0 else {
        throw InCheckpointSampledMTPAcceptanceCLIError.invalidTopK(rawTopK)
    }

    return InCheckpointSampledMTPAcceptanceArguments(
        modelPath: modelPath,
        ngramOffloadPlanPath: ngramOffloadPlanPath,
        promptsFilePath: promptsFilePath,
        maxTokens: maxTokens,
        seed: seed,
        provider: provider,
        outputJSONPath: outputJSONPath,
        topP: topP,
        topK: topK)
}

func inCheckpointSampledMTPAcceptanceExternalDiagnostic(_ error: Error) -> String {
    if let error = error as? InCheckpointSampledMTPAcceptanceCLIError {
        return error.description
    }
    return "qwen4exp-sampled-mtp-acceptance failed: \(error)"
}

// MARK: - Pure arithmetic (independent of MLX; unit-tested directly)

/// Manual, numerically-stabilized softmax over raw logits -- a fresh computation path distinct
/// from `SampledMTPBlockRuntimeBridge`'s own internal `normalizedProbabilities`/
/// `validatingNormalizedProbabilities`. Never reused here: the whole point of this measurement is
/// an INDEPENDENT cross-check of `decide()`'s empirical accept/reject ratio, so this function must
/// not share code with the thing it is checking.
///
/// A raw `-infinity` logit is legitimate (a masked/unsupported vocabulary position -- verified
/// live on this exact model family: `qwen4_exp` logits carry `-inf` at four media-sentinel indices
/// on every forward) and maps to exactly `0.0` probability. `NaN` and `+infinity` are never
/// legitimate and are rejected.
func inCheckpointSampledMTPIndependentSoftmax(_ logits: [Double]) throws -> [Double] {
    guard !logits.isEmpty else {
        throw InCheckpointSampledMTPAcceptanceCLIError.invalidLogits("empty vocabulary")
    }
    guard logits.allSatisfy({ !$0.isNaN && $0 != Double.infinity }) else {
        throw InCheckpointSampledMTPAcceptanceCLIError.invalidLogits("NaN or +infinity logit")
    }
    guard let maxLogit = logits.max(), maxLogit.isFinite else {
        throw InCheckpointSampledMTPAcceptanceCLIError.invalidLogits("every logit is -infinity")
    }
    let exponentials = logits.map { Foundation.exp($0 - maxLogit) }
    let sum = exponentials.reduce(0, +)
    guard sum.isFinite, sum > 0 else {
        throw InCheckpointSampledMTPAcceptanceCLIError.invalidLogits("non-finite or zero softmax sum")
    }
    return exponentials.map { $0 / sum }
}

/// Independently re-derives, in this file's own plain `[Double]` arithmetic, the exact truncated
/// sampling distribution `MLXLMCommon.truncatedSamplingProbabilities` computes from raw logits:
/// `logSoftmax -> top_p -> min_p -> top_k -> softmax(./temperature)`, in that order.
///
/// DELIBERATELY DUPLICATED, NOT SHARED WITH THE RUNTIME'S HELPER. This file's entire reason to
/// exist is an INDEPENDENT cross-check of the runtime's `Sigma_x min(p(x), q(x))` acceptance-rate
/// prediction. If this function called `truncatedSamplingProbabilities` (or imported the runtime's
/// own truncation code from `SampledMTPBlockRuntimeBridge` / `MLXLMCommon`), the cross-check would
/// become tautological: it would AGREE with the runtime even if the runtime's own truncation were
/// wrong, because both sides would be running the identical bug. A later "deduplication" pass MUST
/// NOT replace this body with a call to the shared MLX helper -- doing so would silently delete the
/// only independent check this instrument has.
///
/// Semantics reproduced from `truncatedSamplingProbabilities` / `applyTopPFilter` /
/// `applyMinPFilter` / `applyTopKFilter` (read for reference only, at
/// `spike/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift` -- never imported here):
///   1. `logprobs[i] = logits[i] - logSumExp(logits)`, max-shifted for numerical stability. A raw
///      `-infinity` logit is legitimate (a masked/unsupported vocabulary position, same as
///      `inCheckpointSampledMTPIndependentSoftmax` above) and maps to exactly `0.0` probability.
///   2. top_p (nucleus, applied only when `0 < topP < 1`): sort ascending by log-probability, take
///      the cumulative sum of `exp(sortedLogprob)` in that ascending order, and mask to
///      `-infinity` every entry whose cumulative probability is `<= 1 - topP` (keep the rest,
///      cumulative STRICTLY greater than `1 - topP`) -- EXCEPT the single highest
///      log-probability entry (the last position in ascending sort order, i.e. the row's argmax),
///      which is ALWAYS kept regardless of whether its own cumulative probability clears the
///      threshold. This ascending-cumulative form -- not a descending "keep the top mass first"
///      variant -- is what the vendored `applyTopPFilter` computes; the two disagree at
///      ties/boundaries, so it is reproduced exactly, not approximated. The keep-the-argmax
///      exception mirrors `applyTopPFilter`'s `min_tokens_to_keep=1` fix (HF
///      `TopPLogitsWarper`'s floor): without it, once `1 - topP` rounds to exactly `1.0f` in
///      float32, the cumulative-probability test is false at every position -- including at the
///      row's own maximum, since `cumulativeProbs` is float32-non-decreasing and its last sorted
///      position always holds that maximum -- and the entire row would mask to `-infinity`.
///      Because this oracle's arithmetic is `Double`, not `float32`, that saturation point is far
///      more extreme here (`1 - topP` does not round to exactly `1.0` in `Double` until
///      `topP` is on the order of `Double.ulpOfOne`, roughly `1e-16`, not `float32`'s `~1e-8`) --
///      but the exception is reproduced anyway, both for semantic parity with the fixed vendored
///      helper across their full domains and so the two never silently diverge at the extreme end
///      of `Double`'s own range either (previously, at `topP` this small, this oracle fell through
///      to the "every entry masked" branch below and returned an all-zero vector; with the
///      exception applied, it now returns a one-hot vector on the argmax instead, matching what
///      the fixed vendored helper does at its own, much larger, float32 saturation point).
///   3. min_p (applied only when `minP > 0`): thresholded against the CURRENT (post-top_p-masked)
///      maximum, mirroring `applyMinPFilter` being chained onto the already-top_p-filtered array in
///      the vendored code -- keep entries with `logprob >= currentMax + log(minP)`, mask the rest.
///   4. top_k (applied only when `topK > 0 && topK < vocabularySize`): keep the `topK` highest
///      remaining log-probabilities, mask the rest.
///   5. `softmax(masked / temperature)`, with masked entries mapping to exactly `0.0`.
///
/// `internal` (not `private`), matching every sibling pure function above in this file
/// (`inCheckpointSampledMTPIndependentSoftmax` etc.) that `@testable import fastmlx_harness` calls
/// directly from `InCheckpointSampledMTPAcceptanceArithmeticTests.swift` -- this affects only Swift
/// access control, not the independence contract above: the body still never calls
/// `truncatedSamplingProbabilities` or any other `MLXLMCommon` helper.
func independentTruncatedProbabilities(
    logits: [Double], temperature: Double, topP: Double, topK: Int, minP: Double
) throws -> [Double] {
    guard !logits.isEmpty else {
        throw InCheckpointSampledMTPAcceptanceCLIError.invalidLogits("empty vocabulary")
    }
    guard logits.allSatisfy({ !$0.isNaN && $0 != Double.infinity }) else {
        throw InCheckpointSampledMTPAcceptanceCLIError.invalidLogits("NaN or +infinity logit")
    }
    guard let maxLogit = logits.max(), maxLogit.isFinite else {
        throw InCheckpointSampledMTPAcceptanceCLIError.invalidLogits("every logit is -infinity")
    }

    // Step 1: max-shifted log-sum-exp -> log-probabilities.
    let sumExp = logits.reduce(0.0) { $0 + Foundation.exp($1 - maxLogit) }
    let logSumExp = maxLogit + Foundation.log(sumExp)
    var masked = logits.map { $0 - logSumExp }

    // Step 2: top_p (nucleus), ascending-cumulative form -- matches `applyTopPFilter` exactly,
    // including its `min_tokens_to_keep=1` fix: the last sorted position (this row's argmax) is
    // never masked, even if its own cumulative probability does not clear `keepThreshold`. See the
    // doc comment above for why this oracle mirrors that exception despite its `Double` arithmetic
    // saturating at a far more extreme `topP` than the vendored helper's `float32` arithmetic does.
    if topP > 0, topP < 1 {
        let ascendingOrder = masked.indices.sorted { masked[$0] < masked[$1] }
        let keepThreshold = 1 - topP
        var cumulative = 0.0
        for (position, index) in ascendingOrder.enumerated() {
            cumulative += Foundation.exp(masked[index])
            let isLastSortedPosition = position == ascendingOrder.count - 1
            if cumulative <= keepThreshold, !isLastSortedPosition {
                masked[index] = -Double.infinity
            }
        }
    }

    // Step 3: min_p, thresholded against the CURRENT (post-top_p) max -- matches `applyMinPFilter`
    // being chained onto the already-top_p-filtered array in the vendored code.
    if minP > 0 {
        guard let currentMax = masked.max(), currentMax.isFinite else {
            // Every entry was already masked by top_p -- nothing left for min_p to keep.
            return Array(repeating: 0.0, count: logits.count)
        }
        let threshold = currentMax + Foundation.log(minP)
        for index in masked.indices where masked[index] < threshold {
            masked[index] = -Double.infinity
        }
    }

    // Step 4: top_k -- keep the topK highest remaining log-probabilities.
    if topK > 0, topK < logits.count {
        let descendingOrder = masked.indices.sorted { masked[$0] > masked[$1] }
        for index in descendingOrder.dropFirst(topK) {
            masked[index] = -Double.infinity
        }
    }

    // Step 5: softmax(masked / temperature). `Foundation.exp` of a `-infinity` argument is exactly
    // `0.0` in Swift, so masked entries need no special-casing here.
    let scaled = masked.map { $0 / temperature }
    guard let scaledMax = scaled.max(), scaledMax.isFinite else {
        // Every entry masked (degenerate distribution) -- the caller's DISAGREE cross-check would
        // catch this against the runtime long before this branch would ever legitimately execute.
        return Array(repeating: 0.0, count: logits.count)
    }
    let exponentials = scaled.map { Foundation.exp($0 - scaledMax) }
    let total = exponentials.reduce(0, +)
    guard total.isFinite, total > 0 else {
        throw InCheckpointSampledMTPAcceptanceCLIError.invalidLogits(
            "non-finite or zero truncated softmax sum")
    }
    return exponentials.map { $0 / total }
}

/// Control (mirrors `inCheckpointSampledMTPAssertDrafterForwardsPerBlockRelationship` above): an
/// ordinary function, not a top-level statement or an unreferenced global initializer (either of
/// which could silently never run), called explicitly from `runInCheckpointSampledMTPAcceptance`
/// before `independentTruncatedProbabilities` is ever used to predict a live run's sigmaMinPQ.
///
/// Identity check: at temperature=1, topP=1, topK=0, minP=0 (the CLI's own untruncated
/// configuration, and the only one this file measured before truncated sampling existed), no
/// filter applies, so the function must reduce to the plain softmax within 1e-12.
///
/// Anti-vacuity companion: at a genuinely truncating configuration, the result must differ from
/// the plain softmax by more than a stated threshold on at least one vocabulary entry. Without
/// this, a bug that made truncation silently a no-op (e.g. an inverted `topK > 0` guard) would
/// still pass the identity check above undetected.
func inCheckpointSampledMTPAssertIndependentTruncationControls() {
    let sampleLogits: [Double] = [2.0, 1.0, 0.5, 0.1, -0.3, -1.0, -2.5, -5.0, -Double.infinity, 3.0]

    let plain: [Double]
    let identity: [Double]
    do {
        plain = try inCheckpointSampledMTPIndependentSoftmax(sampleLogits)
        identity = try independentTruncatedProbabilities(
            logits: sampleLogits, temperature: 1, topP: 1, topK: 0, minP: 0)
    } catch {
        preconditionFailure(
            "independentTruncatedProbabilities identity control setup threw unexpectedly: \(error)")
    }
    precondition(
        plain.count == identity.count,
        "independentTruncatedProbabilities identity control vocabulary width mismatch")
    for index in plain.indices {
        precondition(
            abs(plain[index] - identity[index]) <= 1e-12,
            "independentTruncatedProbabilities at temperature=1 topP=1 topK=0 minP=0 must match "
                + "the plain softmax within 1e-12 (index \(index): plain=\(plain[index]) "
                + "identity=\(identity[index])) -- a divergence here means the new truncation path "
                + "changes behavior even when a run's own sampling parameters request no "
                + "truncation at all")
    }

    let truncated: [Double]
    do {
        truncated = try independentTruncatedProbabilities(
            logits: sampleLogits, temperature: 1, topP: 1, topK: 3, minP: 0)
    } catch {
        preconditionFailure(
            "independentTruncatedProbabilities anti-vacuity control setup threw unexpectedly: "
                + "\(error)")
    }
    let maxAbsoluteDelta = zip(plain, truncated).map { abs($0 - $1) }.max() ?? 0
    precondition(
        maxAbsoluteDelta > 0.05,
        "independentTruncatedProbabilities at topK=3 must differ from the untruncated softmax by "
            + "more than 0.05 on at least one entry (observed max|delta|=\(maxAbsoluteDelta)) -- "
            + "otherwise the identity control above cannot distinguish a correct implementation "
            + "from one where truncation is silently a no-op")
}

// MARK: - Truncation binding instrumentation
//
// At the deployed preset (`top_p=0.95, top_k=20`) the two filters run in series and, per drafted
// position, one is almost always inert: if the 0.95 nucleus alone holds more than 20 tokens, top_k
// binds (trims the nucleus down to exactly `topK`) and top_p was the effective filter; if the
// nucleus holds fewer than 20, top_k never removes anything and top_p was the effective filter.
// Nothing about "measured at the deployed preset" is evidence of which filter actually bound
// without recording this per position -- these types/functions make that observable in the output
// instead of merely asserted.

/// Which truncation filter was the one that actually bound at one drafted position, derived from
/// `supportSize` (the count of strictly-nonzero entries in `independentTruncatedProbabilities`'s
/// returned target distribution `p`) and the run's own `topK`. Scoped to the `topK > 0` regime this
/// run measures (`top_p=0.95, top_k=20`-shaped): `topK <= 0` (top_k disabled) always classifies as
/// `.untruncated` here even if `topP` alone is truncating, because that combination is not the
/// regime this instrument exists to characterize.
enum InCheckpointSampledMTPTruncationBindingClass: String, Codable, Equatable, Sendable {
    /// `supportSize == topK`: top_k is the filter that removed the last entries -- the nucleus held
    /// at least `topK` tokens before top_k ran.
    case topKBinding
    /// `supportSize < topK`: top_k found fewer than `topK` surviving candidates and was a no-op --
    /// top_p (or min_p) is what actually shaped the final support.
    case topPBinding
    /// `topK <= 0`: top_k is disabled outright for this run.
    case untruncated
}

/// Classifies one drafted position's truncation outcome. `supportSize` must already be the count of
/// strictly-nonzero entries in the SAME `p` this run measured (see the call site in
/// `InCheckpointMeasuringSampledMTPBlockRuntimeProvider.measuredDecide`) -- never recomputed from a
/// second, independently invented distribution.
func inCheckpointSampledMTPClassifyTruncationBinding(
    supportSize: Int, topK: Int
) -> InCheckpointSampledMTPTruncationBindingClass {
    guard topK > 0 else { return .untruncated }
    return supportSize == topK ? .topKBinding : .topPBinding
}

/// Pooled `min`/`median`/`max`/`count` over a set of per-position `supportSize` values. A dedicated
/// `Int`-keyed summary (not a reuse of `InCheckpointSampledMTPDoubleSummary`, which has no `median`
/// field and is shared by unrelated `Double` timing summaries elsewhere in this file) so adding
/// `median` here cannot ripple into those call sites.
struct InCheckpointSampledMTPTruncationSupportSummary: Codable, Equatable, Sendable {
    let min: Int
    let median: Double
    let max: Int
    let count: Int
}

/// `median` here is the textbook even/odd-count average-of-middle-two-or-take-middle definition,
/// computed on a SORTED COPY of `values` (never mutates the caller's array).
func inCheckpointSampledMTPSummarizeTruncationSupport(
    _ values: [Int]
) -> InCheckpointSampledMTPTruncationSupportSummary? {
    guard !values.isEmpty else { return nil }
    let sorted = values.sorted()
    let mid = sorted.count / 2
    let median: Double =
        sorted.count % 2 == 0
        ? Double(sorted[mid - 1] + sorted[mid]) / 2.0
        : Double(sorted[mid])
    return InCheckpointSampledMTPTruncationSupportSummary(
        min: sorted.first!, median: median, max: sorted.last!, count: sorted.count)
}

/// Classification counts plus the support-size summary for one pool of `supportSize` samples --
/// factored out so the per-stream, per-step-pooled, and run-pooled call sites (all three classify
/// the SAME way against the SAME `topK`) cannot drift into three separate implementations.
struct InCheckpointSampledMTPTruncationBindingCounts: Equatable, Sendable {
    let topKBindingCount: Int
    let topPBindingCount: Int
    let untruncatedCount: Int
    let supportSizeSummary: InCheckpointSampledMTPTruncationSupportSummary?
}

func inCheckpointSampledMTPClassifyTruncationBindings(
    supportSizes: [Int], topK: Int
) -> InCheckpointSampledMTPTruncationBindingCounts {
    var topKBindingCount = 0
    var topPBindingCount = 0
    var untruncatedCount = 0
    for supportSize in supportSizes {
        switch inCheckpointSampledMTPClassifyTruncationBinding(supportSize: supportSize, topK: topK) {
        case .topKBinding: topKBindingCount += 1
        case .topPBinding: topPBindingCount += 1
        case .untruncated: untruncatedCount += 1
        }
    }
    return InCheckpointSampledMTPTruncationBindingCounts(
        topKBindingCount: topKBindingCount,
        topPBindingCount: topPBindingCount,
        untruncatedCount: untruncatedCount,
        supportSizeSummary: inCheckpointSampledMTPSummarizeTruncationSupport(supportSizes))
}

/// `Sigma_x min(p(x), q(x))` -- the exact per-step sampled-MTP acceptance probability under the
/// residual-correction scheme, computed directly from two already-normalized probability vectors
/// over the SAME vocabulary ordering.
func inCheckpointSampledMTPSigmaMinPQ(target p: [Double], draft q: [Double]) -> Double {
    precondition(p.count == q.count, "sigmaMinPQ requires equal-length probability vectors")
    var total = 0.0
    for index in p.indices { total += Swift.min(p[index], q[index]) }
    return total
}

// MARK: - Temperature counterfactual replay
//
// Contract: `docs/task-inbox/2026-09-09-sampled-mtp-temperature-counterfactual-PREDECLARATION.md`.
// A REPLAY over already-computed logits -- zero incremental GPU work, no gate change, no new model
// run. `supports()` only ever admits `temperature == 1` (`SampledMTPBlockRuntimeBridge.swift:174`),
// so every live run this file drives is itself always at `targetTemperature == 1`; the temperatures
// below are a fixed, pinned counterfactual list, entirely independent of that run parameter.

/// Ordered, pinned temperature list this counterfactual replay measures. Order matters:
/// `InCheckpointSampledMTPStepTemperingSample.sigmaByTemperature[i]` corresponds to
/// `inCheckpointSampledMTPCounterfactualTemperatures[i]` -- index `0` is always `T=1.0`, and every
/// call site below (e.g. `sigmaByTemperature[0]` for `sigma_1`, `sigmaByTemperature[3]` for
/// `sigma_0.7`) relies on this exact ordering rather than searching for the value.
let inCheckpointSampledMTPCounterfactualTemperatures: [Double] = [1.0, 0.9, 0.8, 0.7]

/// One drafted position's counterfactual sigma_T reading across every temperature in
/// `inCheckpointSampledMTPCounterfactualTemperatures`, plus the analytic
/// `sigma_argmax`/`sigma_rest` decomposition at `T=1` the predeclaration also requires.
struct InCheckpointSampledMTPStepTemperingSample: Equatable, Sendable {
    /// `sigmaByTemperature[i]` is `sigma_T` at `inCheckpointSampledMTPCounterfactualTemperatures[i]`.
    let sigmaByTemperature: [Double]
    /// `min(p_1(x*), q(x*))` where `x* = argmax p_1` -- the single largest contributor to `sigma_1`.
    let sigmaArgmax: Double
    /// `sigma_1 - sigmaArgmax` -- everything `sigma_1` accumulates outside the target's own argmax.
    let sigmaRest: Double
}

/// Computes the counterfactual `sigma_T = Sigma_x min(p_T(x), q(x))` for every temperature in
/// `inCheckpointSampledMTPCounterfactualTemperatures`, from the SAME `targetRow` logits and
/// UNTRUNCATED, UNTEMPERED drafter proposal `q` the caller already derived for its own `sigma`
/// (`q` is not re-derived per `T` -- see the predeclaration's "the quantity" section for why: it is
/// not the law `decide()` re-derives per temperature either, since `decide()` never runs at
/// `T != 1` in the first place).
///
/// CRITICAL (do not "optimize" this): the surviving support set IS temperature-invariant --
/// `independentTruncatedProbabilities` applies every filter (top_p/min_p/top_k) BEFORE temperature
/// ever enters, at its step 5 (`softmax(masked / temperature)`). That invariance is tempting to
/// exploit by filtering once and re-scaling only the final softmax per temperature. **Do not.**
/// Control C1 (see the call site in `runInCheckpointSampledMTPAcceptance`) asserts that
/// `supportSize == 1` positions show EXACT temperature-invariant `sigma_T` as a genuine check on
/// `independentTruncatedProbabilities`'s own filter-then-temper ordering. Sharing a single filtered
/// array across temperatures would make that assertion true by construction regardless of whether
/// the ordering is actually implemented correctly, and the control would measure nothing. This
/// function therefore calls `independentTruncatedProbabilities` independently, IN FULL (all five
/// steps, including the top_p/min_p/top_k re-filter), once per temperature -- four full passes over
/// the vocabulary instead of one. That CPU cost is accepted deliberately, for C1's sake.
func inCheckpointSampledMTPCounterfactualTempering(
    targetRow: [Double], draft q: [Double],
    targetTopP: Double, targetTopK: Int, targetMinP: Double
) throws -> InCheckpointSampledMTPStepTemperingSample {
    var sigmaByTemperature: [Double] = []
    sigmaByTemperature.reserveCapacity(inCheckpointSampledMTPCounterfactualTemperatures.count)
    var pAtT1: [Double] = []
    for temperature in inCheckpointSampledMTPCounterfactualTemperatures {
        // Independent, full call -- see the CRITICAL note above. Never a shared/cached masked array.
        let pT = try independentTruncatedProbabilities(
            logits: targetRow, temperature: temperature, topP: targetTopP, topK: targetTopK,
            minP: targetMinP)
        if pAtT1.isEmpty, temperature == inCheckpointSampledMTPCounterfactualTemperatures[0] {
            pAtT1 = pT
        }
        sigmaByTemperature.append(inCheckpointSampledMTPSigmaMinPQ(target: pT, draft: q))
    }
    guard
        let xStar = pAtT1.enumerated().max(by: { $0.element < $1.element })?.offset
    else {
        throw InCheckpointSampledMTPAcceptanceCLIError.invalidLogits(
            "counterfactual tempering: empty T=1 target distribution")
    }
    let sigmaArgmax = Swift.min(pAtT1[xStar], q[xStar])
    let sigma1 = sigmaByTemperature[0]
    let sigmaRest = sigma1 - sigmaArgmax
    return InCheckpointSampledMTPStepTemperingSample(
        sigmaByTemperature: sigmaByTemperature, sigmaArgmax: sigmaArgmax, sigmaRest: sigmaRest)
}

/// One block's raw outcome: how many draft tokens this block proposed, and how many the block
/// accepted before falling to a residual correction or the terminal bonus token.
struct InCheckpointSampledMTPBlockOutcome: Equatable, Sendable {
    let proposedCount: Int
    let acceptedDraftCount: Int
}

/// One step index's pooled reached/accepted counts. `stepIndex` is 0-based (CLI/JSON output
/// reports 1-based "step 1"/"step 2" per the predeclaration's own naming).
struct InCheckpointSampledMTPStepAcceptance: Equatable, Sendable {
    let stepIndex: Int
    let reachedCount: Int
    let acceptedCount: Int

    var acceptanceRate: Double? {
        reachedCount > 0 ? Double(acceptedCount) / Double(reachedCount) : nil
    }
}

/// Step `i` (0-based) is "reached" -- its Bernoulli accept/reject trial was actually run -- iff
/// `i == 0` (every block evaluates the first drafted position) or the block accepted at least `i`
/// prior draft tokens (`acceptedDraftCount >= i`). This follows directly from `acceptedDraftCount`
/// counting a CONSECUTIVE accepted prefix (sequential rejection sampling).
///
/// This is the SINGLE definition of "reached" for step `i`: both the empirical per-step acceptance
/// rate (`inCheckpointSampledMTPPerStepAcceptance` below) and the sigmaMinPQ cross-check's conditioned
/// mean (`inCheckpointSampledMTPSummarizeStepSigma` below) must use this exact predicate -- a divergent
/// second copy would silently reintroduce the conditioning mismatch this file was written to close.
func inCheckpointSampledMTPStepReached(step: Int, acceptedDraftCount: Int) -> Bool {
    step == 0 || acceptedDraftCount >= step
}

/// Per-step-index acceptance, pooled across a set of blocks. Step `i` is "accepted" iff
/// `acceptedDraftCount > i`. Never pooled across step indices, matching the predeclaration's
/// "never pooled only -- pooling across step indices hides the depth-decay" requirement.
func inCheckpointSampledMTPPerStepAcceptance(
    _ outcomes: [InCheckpointSampledMTPBlockOutcome],
    stepCount: Int
) -> [InCheckpointSampledMTPStepAcceptance] {
    guard stepCount > 0 else { return [] }
    var reached = [Int](repeating: 0, count: stepCount)
    var accepted = [Int](repeating: 0, count: stepCount)
    for outcome in outcomes {
        for step in 0 ..< stepCount where step < outcome.proposedCount {
            guard inCheckpointSampledMTPStepReached(step: step, acceptedDraftCount: outcome.acceptedDraftCount) else {
                continue
            }
            reached[step] += 1
            if outcome.acceptedDraftCount > step {
                accepted[step] += 1
            }
        }
    }
    return (0 ..< stepCount).map {
        InCheckpointSampledMTPStepAcceptance(stepIndex: $0, reachedCount: reached[$0], acceptedCount: accepted[$0])
    }
}

/// One block's sigmaMinPQ sample for a given step index, paired with that block's
/// `acceptedDraftCount` so a later summary can re-derive the SAME "reached" conditioning the
/// empirical acceptance rate uses (see `inCheckpointSampledMTPStepReached`).
typealias InCheckpointSampledMTPStepSigmaSample = (sigma: Double, acceptedDraftCount: Int)

/// Both readings of the sigmaMinPQ cross-check for one step index:
///
/// - `conditionedMean` is the like-for-like comparison against the empirical `a_step` rate: it
///   pools sigma only over blocks that actually REACHED this step (the same predicate
///   `inCheckpointSampledMTPPerStepAcceptance` uses for `reachedCount`). This estimates
///   `E[accept at step i | block reached step i]`, matching what the empirical rate estimates.
/// - `unconditionalMean` pools sigma over EVERY block, regardless of whether that block's own
///   sequential accept/reject walk actually reached this step -- the target forward verifies every
///   drafted position in one parallel call, so the row pair exists either way. This estimates the
///   unconditional `E[accept at step i]`, a genuinely different (distribution-level) quantity for
///   `i > 0`: if acceptance at earlier steps correlates with distributional overlap at this step
///   (plausible -- "easy" contexts likely have high overlap throughout), conditioning changes the
///   estimate. Step 0 is reached by every block, so its two means always coincide.
///
/// Both are reported -- never just one -- because collapsing to a single number here would silently
/// reintroduce exactly the ambiguity this cross-check exists to remove.
struct InCheckpointSampledMTPStepSigmaSummary: Equatable, Sendable {
    let unconditionalMean: Double?
    let unconditionalCount: Int
    let conditionedMean: Double?
    let conditionedCount: Int
}

func inCheckpointSampledMTPSummarizeStepSigma(
    _ samples: [InCheckpointSampledMTPStepSigmaSample],
    stepIndex: Int
) -> InCheckpointSampledMTPStepSigmaSummary {
    let unconditional = samples.map { $0.sigma }
    let conditioned = samples
        .filter { inCheckpointSampledMTPStepReached(step: stepIndex, acceptedDraftCount: $0.acceptedDraftCount) }
        .map { $0.sigma }
    return InCheckpointSampledMTPStepSigmaSummary(
        unconditionalMean: unconditional.isEmpty ? nil : unconditional.reduce(0, +) / Double(unconditional.count),
        unconditionalCount: unconditional.count,
        conditionedMean: conditioned.isEmpty ? nil : conditioned.reduce(0, +) / Double(conditioned.count),
        conditionedCount: conditioned.count)
}

/// Pooled accepted/**proposed** ratio across a set of blocks. This is NOT the predeclaration's `a`.
///
/// The predeclaration's model is `E = 1 + a + a^2`, a sequential accept/reject walk whose second
/// term is the acceptance probability *conditioned on the step being reached*. Only
/// accepted/**reached** estimates that; see `inCheckpointSampledMTPPerStepAcceptance` and the
/// AGREE/DISAGREE cross-check, which is wired to the reached denominator on both sides so the two
/// cannot drift. Dividing by `proposedCount` pools step-2 positions in blocks that already rejected
/// at step 1 -- trials that were never run, each contributing a guaranteed zero.
///
/// Algebraically this quantity is `(a1 + a1*a2) / numDraft`, i.e. a rescaled `E - 1`. It is a
/// useful summary and is reported, but feeding it into `1 + a + a^2` is a category error.
func inCheckpointSampledMTPPooledAcceptance(_ outcomes: [InCheckpointSampledMTPBlockOutcome]) -> Double? {
    let proposed = outcomes.reduce(0) { $0 + $1.proposedCount }
    guard proposed > 0 else { return nil }
    let accepted = outcomes.reduce(0) { $0 + $1.acceptedDraftCount }
    return Double(accepted) / Double(proposed)
}

struct InCheckpointSampledMTPDoubleSummary: Codable, Equatable, Sendable {
    let mean: Double
    let min: Double
    let max: Double
    let count: Int
}

func inCheckpointSampledMTPSummarize(_ values: [Double]) -> InCheckpointSampledMTPDoubleSummary? {
    guard !values.isEmpty else { return nil }
    return InCheckpointSampledMTPDoubleSummary(
        mean: values.reduce(0, +) / Double(values.count),
        min: values.min()!,
        max: values.max()!,
        count: values.count)
}

/// Solves `a` from `E = 1 + a + a^2 = 1 + 2*delta + D/T` (the predeclaration's break-even
/// arithmetic, specialized to `numDraft == 2`), i.e. the smallest per-step acceptance rate at
/// which this run's own measured `delta` and `D/T` would exactly break even against unspeculated
/// decode. Reported as a derived, informational cross-check -- never a gate this CLI enforces.
func inCheckpointSampledMTPBreakEvenAcceptance(delta: Double, dOverT: Double) -> Double? {
    let rightHandSide = 2 * delta + dOverT
    let discriminant = 1 + 4 * rightHandSide
    guard discriminant >= 0 else { return nil }
    return (-1 + discriminant.squareRoot()) / 2
}

// MARK: - Independent cross-check: binomial confidence interval and AGREE/DISAGREE verdict
//
// Fix 5. The predeclaration's central claim (`docs/task-inbox/2026-09-09-sampled-mtp-acceptance-
// rate-PREDECLARATION.md`, "Independent expected value") is that AGREEMENT between the empirical
// accepted/reached ratio and the independently re-derived `Sigma_x min(p(x), q(x))` mean IS the
// evidence -- printing both without comparing them is exactly the failure mode that let a silent
// mispairing defect ship. This section makes that comparison an explicit, enforced verdict.

/// A two-sided Wilson score confidence interval for a Bernoulli proportion `successCount /
/// trialCount`.
///
/// CHOSEN OVER the naive Wald interval (`p +/- z*sqrt(p(1-p)/n)`) because Wilson stays inside
/// `[0, 1]` and remains well-calibrated even when `n*p` or `n*(1-p)` is small -- a real possibility
/// at the predeclared floor's per-step block counts (as low as ~250 for step 1, fewer still for
/// step 2 after depth decay, where `p` near 0 or 1 makes the Wald normal approximation break down).
/// This is the standard textbook alternative for exactly this regime (small-to-moderate `n`,
/// proportion possibly near the boundary), not a bespoke derivation.
struct InCheckpointSampledMTPWilsonInterval: Equatable, Sendable {
    let lower: Double
    let upper: Double
}

func inCheckpointSampledMTPWilsonInterval(
    successCount: Int, trialCount: Int, confidenceZ: Double = 1.959963985
) -> InCheckpointSampledMTPWilsonInterval? {
    guard trialCount > 0, successCount >= 0, successCount <= trialCount else { return nil }
    let n = Double(trialCount)
    let phat = Double(successCount) / n
    let z2 = confidenceZ * confidenceZ
    let denominator = 1 + z2 / n
    let center = phat + z2 / (2 * n)
    let margin = confidenceZ * ((phat * (1 - phat) / n + z2 / (4 * n * n)).squareRoot())
    let lower = (center - margin) / denominator
    let upper = (center + margin) / denominator
    return InCheckpointSampledMTPWilsonInterval(
        lower: Swift.max(0, lower), upper: Swift.min(1, upper))
}

enum InCheckpointSampledMTPAgreementVerdict: String, Codable, Equatable, Sendable {
    case agree = "AGREE"
    case disagree = "DISAGREE"
    /// Either side of the comparison has no data (`reachedCount == 0` or no sigma samples reached
    /// this step) -- not a computable comparison, and NOT counted as either AGREE or DISAGREE.
    case noData = "NO_DATA"
}

struct InCheckpointSampledMTPAgreementCheck: Codable, Equatable, Sendable {
    /// `nil` for the cross-step pooled check.
    let stepIndex1Based: Int?
    let empiricalAcceptedCount: Int
    let empiricalReachedCount: Int
    let empiricalAcceptanceRate: Double?
    let wilsonLower: Double?
    let wilsonUpper: Double?
    let sigmaMinPQMeanConditionedOnReached: Double?
    let verdict: InCheckpointSampledMTPAgreementVerdict
}

/// Compares the empirical accepted/reached ratio's 95% Wilson interval against the independently
/// re-derived `sigmaMean` (the "reached"-conditioned mean -- the like-for-like comparison, same
/// predicate `inCheckpointSampledMTPStepReached` both quantities already share).
func inCheckpointSampledMTPAgreementCheck(
    stepIndex1Based: Int?,
    acceptedCount: Int,
    reachedCount: Int,
    sigmaMean: Double?
) -> InCheckpointSampledMTPAgreementCheck {
    guard reachedCount > 0, let sigmaMean, let interval = inCheckpointSampledMTPWilsonInterval(
        successCount: acceptedCount, trialCount: reachedCount)
    else {
        return InCheckpointSampledMTPAgreementCheck(
            stepIndex1Based: stepIndex1Based, empiricalAcceptedCount: acceptedCount,
            empiricalReachedCount: reachedCount, empiricalAcceptanceRate: nil,
            wilsonLower: nil, wilsonUpper: nil, sigmaMinPQMeanConditionedOnReached: sigmaMean,
            verdict: .noData)
    }
    let agrees = sigmaMean >= interval.lower && sigmaMean <= interval.upper
    return InCheckpointSampledMTPAgreementCheck(
        stepIndex1Based: stepIndex1Based, empiricalAcceptedCount: acceptedCount,
        empiricalReachedCount: reachedCount,
        empiricalAcceptanceRate: Double(acceptedCount) / Double(reachedCount),
        wilsonLower: interval.lower, wilsonUpper: interval.upper,
        sigmaMinPQMeanConditionedOnReached: sigmaMean,
        verdict: agrees ? .agree : .disagree)
}

/// A one-line, human-readable rendering of one agreement check -- used both for console output and
/// for the `sigmaMinPQDisagreement` abort message.
func inCheckpointSampledMTPDescribeAgreementCheck(_ check: InCheckpointSampledMTPAgreementCheck) -> String {
    let label = check.stepIndex1Based.map { "step\($0)" } ?? "pooled"
    guard let rate = check.empiricalAcceptanceRate, let lower = check.wilsonLower,
        let upper = check.wilsonUpper, let sigmaMean = check.sigmaMinPQMeanConditionedOnReached
    else {
        return "\(label): NO_DATA (reachedCount=\(check.empiricalReachedCount))"
    }
    return "\(label): empirical a=\(String(format: "%.4f", rate)) "
        + "(accepted=\(check.empiricalAcceptedCount)/reached=\(check.empiricalReachedCount)), "
        + "95% Wilson CI=[\(String(format: "%.4f", lower)),\(String(format: "%.4f", upper))], "
        + "sigmaMinPQ(conditioned)=\(String(format: "%.4f", sigmaMean)) -- \(check.verdict.rawValue)"
}

// MARK: - Draft-logit interception
//
// `decide()` receives `targetLogits` directly as a parameter (one row per proposed token) but
// receives draft probabilities only indirectly, via the provider's own private recorder --
// reading THAT would violate the predeclaration's "do NOT read it out of decide's internals"
// requirement. Instead, this wraps the provider's own `proposalSampler` (the `LogitSampler` the
// iterator calls once per drafted position, on the REAL forward pass) and tees the raw logits
// each call receives, before forwarding to the real (stateful, recording) sampler unchanged.
//
// TRUE invariant (verified directly at source THIS cycle, replacing an earlier "`numDraft + 1`
// calls, seed row first" claim that was FALSE and made the queue-size check below unsatisfiable in
// every live run -- `draftLogitCaptureShapeMismatch(expectedRows: 3, observedRows: 2)` on the FIRST
// block of every stream):
//   - `prepareDrafterState` calls `sampleMTPSeed` -> `sampler.sample(logits:)` ONCE, before the
//     first block (in-checkpoint MTP model source, line 916).
//   - `draftBlock`'s seed branch (source line 964) runs on EVERY block, because
//     `commitDrafterState` (source line 1059) re-sets `state.seedToken` on every prior block too --
//     there is no block where the seed branch does not fire. It consumes the seed as drafted
//     position 0 WITHOUT calling `sample` again, computes `remainingDrafts = blockSize - 2` (`= 1`
//     for the pinned `blockSize = 3`), and calls the draft-token loop with `blockSize - 1` (`= 2`);
//     that loop runs `for stepIndex in 0 ..< (blockSize - 1)` (source line 1160), i.e. exactly ONE
//     iteration and ONE `sampler.sample(logits:)` call (source line 1175) for the pinned block size.
//   - `commitDrafterState`'s OWN `sampleMTPSeed` call happens AFTER that block's `decide()` already
//     returned (source line 1059, invoked from `MTPSpeculativeTokenIterator.swift:814`) -- it
//     produces the seed for the NEXT block, not a row belonging to the current one.
// So the queue seen at each `decide()` call is exactly `[seed_row, draft_row]` -- TWO rows, i.e.
// `numDraft` rows, not `numDraft + 1`. `seed_row` is the token sampled at the end of the PRIOR
// block's `commitDrafterState` (or `prepareDrafterState`'s initial seed for the very first block)
// and is LEADING-ALIGNED with `proposedTokens[0]` (the seed IS proposed token 0); `draft_row` is
// this block's own single draft-loop call and pairs with `proposedTokens[1]`. This matches what the
// production provider itself does: `peek(count:)` in `SampledMTPBlockRuntimeBridge.swift:506-513`
// reads the LEADING `count` pending rows, never a trailing slice.
// `InCheckpointMeasuringSampledMTPBlockRuntimeProvider.decide` therefore (a) requires the queue to
// hold EXACTLY `numDraft` rows before consuming it, failing closed with
// `draftLogitCaptureShapeMismatch` otherwise, and (b) reads the LEADING `numDraft` rows (the whole
// queue) via `TeeingLogitSampler.takeLeadingAndReset`, which clears the queue so nothing
// accumulates into the next block's.
final class InCheckpointSampledMTPTeeingLogitSampler: LogitSampler {
    private let inner: any LogitSampler
    /// Queue of raw, FLATTENED-BUT-NOT-YET-EVALUATED draft logit rows (lazy `MLXArray` graph
    /// nodes), in call order: this block's `seed_row` then its `draft_row` (see the file-header
    /// note above). Deliberately left unevaluated here -- Fix 3: both calls that populate this
    /// queue land inside the vendor iterator's `draftPhaseSeconds`-timed windows
    /// (`MTPSpeculativeTokenIterator.swift:655-658` for the draft-loop call, `:823-834` for the
    /// commit-seed call), so forcing a device->host sync (`eval` + `asArray` + `[Double]`
    /// allocation) here would both inflate `draftPhaseSeconds` -- which flows directly into
    /// `delta` -- and destroy GPU pipelining across the drafter's sequential forwards. The
    /// eval/copy/conversion is deferred to `InCheckpointMeasuringSampledMTPBlockRuntimeProvider
    /// .measuredDecide`, AFTER `inner.decide()` has already returned, which is outside both timed
    /// windows. Retaining the `MLXArray` reference (rather than dropping it) is safe: MLX's lazy
    /// graph nodes are ordinary Swift reference-counted objects, so holding this array alive here
    /// keeps its whole upstream computation graph alive until this queue evaluates or clears it --
    /// no different from holding any other un-eval'd `MLXArray` across a function boundary, which
    /// this same file already does for `targetLogits` (a parameter of `decide` itself).
    private(set) var capturedLogits: [MLXArray] = []

    init(inner: any LogitSampler) {
        self.inner = inner
    }

    func sample(logits: MLXArray) -> MLXArray {
        capturedLogits.append(logits.flattened())
        return inner.sample(logits: logits)
    }

    /// Returns the LEADING `count` captured rows (this block's `seed_row` then its `draft_row`,
    /// leading-aligned with `proposedTokens` -- see the file-header note above) and clears the
    /// ENTIRE queue. Only called AFTER the paired `decide()` call has already succeeded, mirroring
    /// the real recorder's own peek-then-commit-on-success discipline. Renamed from
    /// `takeTrailingAndReset` (Fix 1): the correct slice is the LEADING `count` entries, not a
    /// trailing one -- the old name would have been actively misleading after the fix.
    func takeLeadingAndReset(_ count: Int) -> [MLXArray] {
        let taken = Array(capturedLogits.prefix(count))
        capturedLogits.removeAll(keepingCapacity: true)
        return taken
    }
}

/// Wraps a real `SampledMTPBlockRuntimeDeciding` provider (seeded or nondeterministic) to measure,
/// IN THIS RUN, everything the predeclaration requires that the provider itself does not expose:
/// per-block `decide()` wall time (`D`), and an independently re-derived `Sigma_x min(p(x), q(x))`
/// per drafted position. Every accept/reject DECISION is delegated unchanged to `inner` -- this
/// class changes no serving behavior, only observes it.
final class InCheckpointMeasuringSampledMTPBlockRuntimeProvider: SampledMTPBlockRuntimeDeciding {
    private let inner: any SampledMTPBlockRuntimeDeciding
    private let teeingSampler: InCheckpointSampledMTPTeeingLogitSampler
    /// The run's ACTUAL target-side sampling parameters (same values reported in
    /// `InCheckpointSampledMTPAcceptanceSamplingReport`/`GenerateParameters`, never a second,
    /// independently invented set) -- drives whether and how `independentTruncatedProbabilities`
    /// truncates the target distribution `p` below. When these are the identity configuration
    /// (temperature=1, topP=1, topK=0, minP=0), `independentTruncatedProbabilities` reduces to the
    /// plain softmax (see `inCheckpointSampledMTPAssertIndependentTruncationControls`), so this
    /// class's prediction never hardcodes either the truncated or untruncated case -- it always
    /// tracks whatever configuration is actually being measured.
    private let targetTemperature: Double
    private let targetTopP: Double
    private let targetTopK: Int
    private let targetMinP: Double

    private(set) var decideDurationsSeconds: [Double] = []
    private(set) var blockOutcomes: [InCheckpointSampledMTPBlockOutcome] = []
    /// `sigmaMinPQByStep[i]` collects one independently-derived sample per block for drafted
    /// position `i` (0-based), paired with that block's `acceptedDraftCount` so a later summary can
    /// derive BOTH the unconditional mean and the "reached"-conditioned mean that is comparable
    /// like-for-like to the empirical per-step acceptance rate (see
    /// `InCheckpointSampledMTPStepSigmaSummary`). Recorded unconditionally for every block regardless of
    /// where that block's own accept/reject walk stopped -- the target forward verifies every
    /// drafted position in one parallel call, so every step's target/draft row pair exists whether
    /// or not that step was ultimately reached by the sequential walk.
    private(set) var sigmaMinPQByStep: [Int: [InCheckpointSampledMTPStepSigmaSample]] = [:]
    /// `truncationSupportSizeByStep[i]` collects, per block, the count of strictly-nonzero entries
    /// in that block's OWN `p` at drafted position `i` -- i.e. how many vocabulary entries survived
    /// truncation. Recorded alongside `sigmaMinPQByStep` above (same loop, same already-computed
    /// `p`, no second MLX evaluation) so a reader can later classify which filter bound (see
    /// `inCheckpointSampledMTPClassifyTruncationBinding`) instead of only asserting it did.
    private(set) var truncationSupportSizeByStep: [Int: [Int]] = [:]
    /// `temperingByStep[i]` collects, per block, one `InCheckpointSampledMTPStepTemperingSample` for
    /// drafted position `i` -- the counterfactual `sigma_T` replay over the SAME already-computed
    /// `targetRow`/`q` pair `sigmaMinPQByStep`/`truncationSupportSizeByStep` above are recorded from
    /// (same loop iteration, appended in lockstep -- a later reader zips these three by index).
    /// See `docs/task-inbox/2026-09-09-sampled-mtp-temperature-counterfactual-PREDECLARATION.md`.
    private(set) var temperingByStep: [Int: [InCheckpointSampledMTPStepTemperingSample]] = [:]

    init(
        inner: any SampledMTPBlockRuntimeDeciding,
        targetTemperature: Double, targetTopP: Double, targetTopK: Int, targetMinP: Double
    ) {
        self.inner = inner
        self.teeingSampler = InCheckpointSampledMTPTeeingLogitSampler(inner: inner.proposalSampler)
        self.targetTemperature = targetTemperature
        self.targetTopP = targetTopP
        self.targetTopK = targetTopK
        self.targetMinP = targetMinP
    }

    var proposalSampler: any LogitSampler { teeingSampler }

    func supports(parameters: GenerateParameters) -> Bool {
        inner.supports(parameters: parameters)
    }

    func decide(
        proposedTokens: [Int],
        targetLogits: [MLXArray],
        bonusTargetLogits: MLXArray
    ) throws -> SampledMTPBlockRuntimeDecision {
        let needed = proposedTokens.count
        // Fix 1: EXACTLY `needed` (== `numDraft`) rows -- `[seed_row, draft_row]` -- see the
        // file-header note above `InCheckpointSampledMTPTeeingLogitSampler`.
        let expectedCapturedCount = needed
        guard teeingSampler.capturedLogits.count == expectedCapturedCount else {
            throw InCheckpointSampledMTPAcceptanceCLIError.draftLogitCaptureShapeMismatch(
                expectedRows: expectedCapturedCount, observedRows: teeingSampler.capturedLogits.count)
        }
        // Fix 1: the drafted-position rows are the LEADING `needed` entries (the whole queue) --
        // leading-aligned with `proposedTokens` (`peek(count:)` in the production provider reads
        // the same leading slice). Fix 3: this is a non-destructive peek of the raw, UN-EVAL'D
        // `MLXArray` rows -- conversion to `[Double]` is deferred to `measuredDecide`, AFTER
        // `inner.decide()` below has already returned, so this peek itself never forces a GPU sync.
        // The queue is only cleared after `inner.decide()` succeeds (see `measuredDecide`).
        let draftRows = Array(teeingSampler.capturedLogits.prefix(needed))
        guard targetLogits.count == needed else {
            // `inner.decide` will throw its own, more specific mismatch error for this; let it.
            return try measuredDecide(
                proposedTokens: proposedTokens, targetLogits: targetLogits,
                bonusTargetLogits: bonusTargetLogits, draftRows: nil)
        }

        // Target rows ARE flattened/eval'd here, deliberately BEFORE `measuredDecide` opens the
        // `D` timer below -- this pre-syncs `targetLogits` so `inner.decide()`'s own internal
        // touching of those same arrays (during the timed window) is a no-op sync. This is what
        // keeps `D` decide()-CPU-only rather than absorbing the target forward's device->host
        // sync -- see `measuredDecide`'s doc comment for why that is the quantity the
        // predeclaration's break-even formula wants. Moving this to AFTER the timer (mirroring the
        // draft-row deferral below) would instead push that sync onto `inner.decide()` itself,
        // INSIDE the timed window, and defeat the purpose -- do NOT "fix" this the same way as the
        // draft rows.
        var targetRows: [[Double]] = []
        targetRows.reserveCapacity(needed)
        for index in 0 ..< needed {
            targetRows.append(inCheckpointSampledMTPFlattenLogits(targetLogits[index]))
        }

        return try measuredDecide(
            proposedTokens: proposedTokens, targetLogits: targetLogits,
            bonusTargetLogits: bonusTargetLogits, draftRows: (needed, draftRows, targetRows))
    }

    /// Times ONLY `inner.decide()` below -- `D` is decide()-CPU-only, deliberately excluding both
    /// the target forward's device->host sync (already paid in `decide()` above, before this timer
    /// opens) and the draft-row eval/copy/conversion (deferred below, after this timer closes --
    /// Fix 3). This is the quantity the predeclaration's break-even formula
    /// `S = E / (1 + 2*delta + D/T)` wants: the target forward IS the formula's `1` (one
    /// unspeculated target step), so folding its sync cost into `D` too would double-count it. In
    /// the UN-instrumented production path, `mainLogits` stays lazy until the provider's own
    /// `normalizedProbabilities` touches it inside `decide()`, so a naive production timing of
    /// `decide()` would absorb that sync -- this harness's `D` deliberately does NOT match that
    /// naive timing, and the JSON's `decideCostExcludesTargetForward` field names this so a reader
    /// never assumes this run's `D` is the same quantity as an independently-measured `decide()`
    /// wall-clock figure (e.g. the cycle-117 17.2-21.5ms figure on record, which must NOT be
    /// compared against this run's `D` without re-deriving it the same way).
    private func measuredDecide(
        proposedTokens: [Int],
        targetLogits: [MLXArray],
        bonusTargetLogits: MLXArray,
        draftRows: (needed: Int, rows: [MLXArray], targetRows: [[Double]])?
    ) throws -> SampledMTPBlockRuntimeDecision {
        let start = ProcessInfo.processInfo.systemUptime
        let decision = try inner.decide(
            proposedTokens: proposedTokens, targetLogits: targetLogits,
            bonusTargetLogits: bonusTargetLogits)
        let elapsed = ProcessInfo.processInfo.systemUptime - start

        if let draftRows {
            // Clears the whole queue -- it held exactly `draftRows.needed` rows (see `decide()`
            // above).
            _ = teeingSampler.takeLeadingAndReset(draftRows.needed)
            // Fix 3: eval + flatten + `[Double]` conversion of the DRAFT rows happens HERE, after
            // `inner.decide()` has already returned and `elapsed` is already captured -- outside
            // both of `draftPhaseSeconds`'s timed windows (`MTPSpeculativeTokenIterator.swift:
            // 655-658,823-834`) and outside `D` itself. Batch-eval every row in one call rather
            // than one `eval()` per row.
            eval(draftRows.rows)
            for index in 0 ..< draftRows.needed {
                let draftRow = draftRows.rows[index].asArray(Float.self).map(Double.init)
                let targetRow = draftRows.targetRows[index]
                // Deferring draft-row materialization (Fix 3) means this width check can only run
                // AFTER `inner.decide()` has already executed, unlike the old fail-fast placement
                // -- inert in practice (vocabulary width is a checkpoint-fixed shape, not a runtime
                // toggle), and the caller (`MTPSpeculativeTokenIterator`) treats a `decide()` throw
                // identically regardless of where inside this method it originates.
                guard targetRow.count == draftRow.count else {
                    throw InCheckpointSampledMTPAcceptanceCLIError.targetDraftVocabularyWidthMismatch(
                        expected: draftRow.count, actual: targetRow.count)
                }
                // Target side `p`: truncated at whatever configuration this run's OWN
                // `GenerateParameters` actually requests (`targetTemperature`/`targetTopP`/
                // `targetTopK`/`targetMinP`, set in `init` from the same `parameters` the run's
                // scalar and speculative iterators are constructed with) -- never hardcoded to
                // either the truncated or untruncated case. At the identity configuration
                // (temperature=1, topP=1, topK=0, minP=0) this is exactly the plain softmax (see
                // `inCheckpointSampledMTPAssertIndependentTruncationControls`).
                let p = try independentTruncatedProbabilities(
                    logits: targetRow, temperature: targetTemperature, topP: targetTopP,
                    topK: targetTopK, minP: targetMinP)
                // Draft side `q` stays UNTRUNCATED: the runtime's own drafter proposal distribution
                // is not truncated, so re-deriving it with truncation here would predict a
                // different quantity than the one `decide()` actually accepts/rejects against.
                let q = try inCheckpointSampledMTPIndependentSoftmax(draftRow)
                let sigma = inCheckpointSampledMTPSigmaMinPQ(target: p, draft: q)
                sigmaMinPQByStep[index, default: []].append(
                    (sigma: sigma, acceptedDraftCount: decision.acceptedDraftCount))
                // One pass over `p` (already allocated above, no second MLX evaluation): count of
                // strictly-nonzero entries is exactly how many vocabulary entries this position's
                // truncation left standing -- see `InCheckpointSampledMTPTruncationBindingClass`.
                let supportSize = p.reduce(0) { $0 + ($1 > 0 ? 1 : 0) }
                truncationSupportSizeByStep[index, default: []].append(supportSize)
                // Counterfactual temperature replay -- same `targetRow`/`q` this position's `sigma`
                // above was already computed from, independently re-filtered per temperature (see
                // `inCheckpointSampledMTPCounterfactualTempering`'s CRITICAL note on why the filter
                // chain is never shared across temperatures despite the surviving support set being
                // temperature-invariant).
                let tempering = try inCheckpointSampledMTPCounterfactualTempering(
                    targetRow: targetRow, draft: q, targetTopP: targetTopP, targetTopK: targetTopK,
                    targetMinP: targetMinP)
                // `sigma_T` at `T=1.0` is the SAME quantity as `sigma` above only when
                // `targetTemperature == 1` -- true of every live run this file drives (`supports()`
                // pins it), but asserted here rather than assumed: `tempering` was computed from an
                // entirely independent call to `independentTruncatedProbabilities`, and a divergence
                // would invalidate every pooled `sigma_T` comparison downstream.
                if targetTemperature == 1 {
                    precondition(
                        tempering.sigmaByTemperature[0] == sigma,
                        "sigma_T at T=1.0 (independently recomputed by "
                            + "inCheckpointSampledMTPCounterfactualTempering) must equal the sigma "
                            + "already computed above at targetTemperature=1 EXACTLY -- observed "
                            + "sigma_T[T=1.0]=\(tempering.sigmaByTemperature[0]) sigma=\(sigma). A "
                            + "divergence here means the two independently-computed calls to "
                            + "independentTruncatedProbabilities do not agree on identical inputs, "
                            + "which would invalidate every pooled sigma_T comparison below.")
                }
                temperingByStep[index, default: []].append(tempering)
            }
        }
        decideDurationsSeconds.append(elapsed)
        blockOutcomes.append(
            InCheckpointSampledMTPBlockOutcome(
                proposedCount: proposedTokens.count, acceptedDraftCount: decision.acceptedDraftCount))
        return decision
    }
}

func inCheckpointSampledMTPFlattenLogits(_ array: MLXArray) -> [Double] {
    let flat = array.flattened()
    eval(flat)
    return flat.asArray(Float.self).map(Double.init)
}

// MARK: - Namespace mapping (local; mirrors, but does not import,
// `SpikeServingAdapters.inCheckpointMTPRuntimeNamespace`, which is internal to that module)

func inCheckpointSampledMTPRuntimeNamespace(
    _ namespace: FastMLXInCheckpointMTPNamespace
) -> InCheckpointMTPNamespaceSelection {
    switch namespace {
    case .official: return .official
    case .converted: return .converted
    }
}

// MARK: - Output schema

struct InCheckpointSampledMTPAcceptanceSamplingReport: Codable, Equatable, Sendable {
    let temperature: Double
    let topP: Double
    let topK: Int
    let minP: Double
    let repetitionPenaltyIsNil: Bool
    let presencePenaltyIsNil: Bool
    let frequencyPenaltyIsNil: Bool
}

struct InCheckpointSampledMTPAcceptanceStepReport: Codable, Equatable, Sendable {
    let stepIndex1Based: Int
    let reachedCount: Int
    let acceptedCount: Int
    let acceptanceRate: Double?
    /// Unconditional mean over every block that proposed this step, regardless of whether that
    /// block's sequential accept/reject walk reached it -- see `InCheckpointSampledMTPStepSigmaSummary`.
    let sigmaMinPQMean: Double?
    let sigmaMinPQBlockCount: Int
    /// The like-for-like cross-check against `acceptanceRate` above: mean over only the blocks that
    /// actually REACHED this step (same predicate as `reachedCount`). `nil` when `reachedCount ==
    /// 0` -- never fabricated as `0.0`.
    let sigmaMinPQMeanConditionedOnReached: Double?
    let sigmaMinPQConditionedBlockCount: Int
    /// Per-position truncation-binding classification counts at this step, over the SAME blocks
    /// `sigmaMinPQBlockCount` above pools (unconditional, every block regardless of reached) -- see
    /// `InCheckpointSampledMTPTruncationBindingClass`. The three counts always sum to
    /// `sigmaMinPQBlockCount`.
    let topKBindingCount: Int
    let topPBindingCount: Int
    let untruncatedCount: Int
    /// `nil` only when this step has zero blocks (mirrors `inCheckpointSampledMTPSummarize`'s own
    /// `nil`-on-empty convention elsewhere in this file).
    let truncationSupportSize: InCheckpointSampledMTPTruncationSupportSummary?
}

struct InCheckpointSampledMTPAcceptanceStreamReport: Codable, Equatable, Sendable {
    let promptIndex: Int
    let promptPreview: String
    let promptTokenCount: Int
    let blockCount: Int
    let proposedCount: Int
    let acceptedCount: Int
    let passthroughReason: String?
    let excludedFromPooling: Bool
    let acceptanceRate: Double?
    let perStep: [InCheckpointSampledMTPAcceptanceStepReport]
}

struct InCheckpointSampledMTPAcceptancePooledReport: Codable, Equatable, Sendable {
    let cleanStreamCount: Int
    let excludedStreamCount: Int
    let blockCount: Int
    let proposedCount: Int
    let acceptedCount: Int
    let acceptanceRate: Double?
    let perStep: [InCheckpointSampledMTPAcceptanceStepReport]
    let decideSecondsPerBlock: InCheckpointSampledMTPDoubleSummary?
    /// `D` above is decide()-CPU-only: it excludes the target forward's device->host sync
    /// (deliberately pre-synced, and therefore timed, BEFORE this timer opens -- see
    /// `InCheckpointMeasuringSampledMTPBlockRuntimeProvider.decide`'s comment) and excludes the
    /// draft-row eval/copy (deliberately deferred to AFTER this timer closes -- Fix 3). Always
    /// `true` when this field is present; named explicitly (rather than left implicit) so a reader
    /// never assumes this `D` is the same quantity as an independently-measured `decide()`
    /// wall-clock figure from a different instrument or run (e.g. the cycle-117 17.2-21.5ms figure
    /// on record, which must NOT be compared against this run's `D` without re-deriving it the same
    /// way).
    let decideCostExcludesTargetForward: Bool?
    let targetStepSecondsPerToken: InCheckpointSampledMTPDoubleSummary?
    /// Drafter cost as a fraction of one target scalar step, PER DRAFTER FORWARD (not per block).
    /// THIS run's own phase telemetry (`draftBlockSeconds`/`draftBlockCount`) is a PER-BLOCK figure
    /// covering BOTH of a block's drafter forwards (the draft loop AND the post-`decide()`
    /// commit-seed call -- see `inCheckpointSampledMTPDrafterForwardsPerBlock`'s doc comment), so
    /// this field divides by `inCheckpointSampledMTPDrafterForwardsPerBlock` before dividing by
    /// THIS run's own measured `T`. Named `deltaPerForward` (not `delta`) so the unit cannot be
    /// misread as per-block -- the predeclaration's `S = E / (1 + 2*delta + D/T)` formula already
    /// multiplies by 2 to reconstruct the per-block cost; passing a per-block figure here would
    /// double that term.
    let deltaPerForward: Double?
    let dOverT: Double?
    /// Informational only, derived from `deltaPerForward`/`dOverT` above -- never a gate this CLI
    /// enforces.
    let breakEvenAcceptance: Double?
}

struct InCheckpointSampledMTPAcceptanceSampleSizeReport: Codable, Equatable, Sendable {
    let proposedDraftTokenCount: Int
    let requiredProposedDraftTokenCount: Int
    let blockCount: Int
    let requiredBlockCount: Int
    let cleanPromptCount: Int
    let requiredCleanPromptCount: Int
    /// Always `true` when this report was written to `--output-json` -- a run below the floor
    /// refuses to write JSON at all (see `sampleSizeBelowPredeclaredFloor`) rather than emit this
    /// struct with `false`. Present anyway (not merely implied by the run having produced output)
    /// so a reader inspecting only this one struct sees the floor was actually checked, not
    /// silently skipped.
    let meetsPredeclaredSampleSize: Bool
}

struct InCheckpointSampledMTPAcceptanceAgreementReport: Codable, Equatable, Sendable {
    let perStep: [InCheckpointSampledMTPAgreementCheck]
    /// Cross-step pooled check: every step index's accept/reject trials pooled into one Bernoulli
    /// sample (distinct from the headline `pooled.acceptanceRate`, which pools `acceptedDraftCount`
    /// over `proposedCount` rather than per-step reached/accepted trials -- see the call site for
    /// why the trial-pooled framing is the one comparable to a Bernoulli confidence interval).
    let pooled: InCheckpointSampledMTPAgreementCheck
    /// `true` iff every non-`NO_DATA` check above is `AGREE`. Always `true` when this report was
    /// written to `--output-json` -- a run with any `DISAGREE` refuses to write JSON at all (see
    /// `sigmaMinPQDisagreement`).
    let allAgree: Bool
}

struct InCheckpointSampledMTPAcceptanceReport: Codable, Sendable {
    let schemaVersion: Int
    let modelPath: String
    let ngramOffloadPlanPath: String
    let promptsFilePath: String
    let maxTokens: Int
    let seed: UInt64
    let provider: String
    let buildConfiguration: String
    let blockSize: Int
    let numDraft: Int
    let sampling: InCheckpointSampledMTPAcceptanceSamplingReport
    let streams: [InCheckpointSampledMTPAcceptanceStreamReport]
    let pooled: InCheckpointSampledMTPAcceptancePooledReport
    let sampleSize: InCheckpointSampledMTPAcceptanceSampleSizeReport
    let agreement: InCheckpointSampledMTPAcceptanceAgreementReport
}

// MARK: - Entry point

func runInCheckpointSampledMTPAcceptance(arguments: [String]) async throws {
    #if DEBUG
    throw InCheckpointSampledMTPAcceptanceCLIError.releaseBuildRequired
    #else
    inCheckpointSampledMTPAssertDrafterForwardsPerBlockRelationship()
    inCheckpointSampledMTPAssertIndependentTruncationControls()
    let parsed = try parseInCheckpointSampledMTPAcceptanceArguments(arguments)

    let outputURL = URL(fileURLWithPath: parsed.outputJSONPath)
    guard !FileManager.default.fileExists(atPath: outputURL.path) else {
        throw InCheckpointSampledMTPAcceptanceCLIError.outputJSONMustBeNew
    }

    let promptsText: String
    do {
        promptsText = try String(
            contentsOf: URL(fileURLWithPath: parsed.promptsFilePath), encoding: .utf8)
    } catch {
        throw InCheckpointSampledMTPAcceptanceCLIError.promptsFileUnavailable(parsed.promptsFilePath)
    }
    let prompts = promptsText.split(separator: "\n", omittingEmptySubsequences: false)
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { !$0.isEmpty }
    guard !prompts.isEmpty else {
        throw InCheckpointSampledMTPAcceptanceCLIError.promptsFileEmpty
    }

    let modelDirectory = URL(fileURLWithPath: parsed.modelPath, isDirectory: true)
    let planFileURL = URL(fileURLWithPath: parsed.ngramOffloadPlanPath)
    let tokenizerLoader = #huggingFaceTokenizerLoader()
    let context = try await loadOffloadedNGramModelContext(
        modelDirectory: modelDirectory, planFileURL: planFileURL, tokenizerLoader: tokenizerLoader)

    let loadedDrafter = try loadInCheckpointMTPDrafter(
        modelDirectory: modelDirectory,
        expectedNamespace: inCheckpointSampledMTPRuntimeNamespace(
            inCheckpointSampledMTPArtifactSelection.namespace),
        expectedSourceKeyCount: inCheckpointSampledMTPArtifactSelection.expectedSourceKeyCount,
        revision: inCheckpointSampledMTPArtifactSelection.revision)

    // `temperature` and `minP` are PINNED, not a choice: `supports()`
    // (`SampledMTPBlockRuntimeBridge.swift`'s `sharedSampledMTPSupportsPredicate`) requires exactly
    // `temperature == 1` and `minP == 0`, and `providerIsEligible` additionally requires
    // `parameters.processor() == nil` (`MTPSpeculativeTokenIterator.swift:171-173`) -- there is no
    // flag for either. `topP`/`topK` ARE settable, via `--top-p`/`--top-k`, and default to the
    // untruncated identity (`1`/`0`); `parseInCheckpointSampledMTPAcceptanceArguments` already
    // rejects any value `supports()` would refuse, so `parsed.topP`/`parsed.topK` are always within
    // the accepted range by the time they reach here.
    let parameters = GenerateParameters(
        maxTokens: parsed.maxTokens,
        temperature: 1,
        topP: Float(parsed.topP),
        topK: parsed.topK,
        minP: 0,
        seed: parsed.seed)
    let samplingReport = InCheckpointSampledMTPAcceptanceSamplingReport(
        temperature: Double(parameters.temperature),
        topP: Double(parameters.topP),
        topK: parameters.topK,
        minP: Double(parameters.minP),
        repetitionPenaltyIsNil: parameters.repetitionPenalty == nil,
        presencePenaltyIsNil: parameters.presencePenalty == nil,
        frequencyPenaltyIsNil: parameters.frequencyPenalty == nil)

    print(
        "qwen4exp-sampled-mtp-acceptance: build=release provider=\(parsed.provider.rawValue) "
            + "seed=\(parsed.seed) sampling: temperature=\(samplingReport.temperature) "
            + "topP=\(samplingReport.topP) topK=\(samplingReport.topK) minP=\(samplingReport.minP) "
            + "repetitionPenalty=nil presencePenalty=nil frequencyPenalty=nil "
            + "(temperature=1 and minP=0 are PINNED -- supports() requires exactly those; topP/topK "
            + "are settable via --top-p/--top-k, default to the untruncated identity (1/0), and the "
            + "ACTUAL values this run measured are topP=\(samplingReport.topP) "
            + "topK=\(samplingReport.topK), printed above)")

    var streamReports: [InCheckpointSampledMTPAcceptanceStreamReport] = []
    var cleanOutcomes: [InCheckpointSampledMTPBlockOutcome] = []
    var cleanDecideSeconds: [Double] = []
    var cleanTargetStepSeconds: [Double] = []
    var cleanDraftBlockSeconds = 0.0
    var cleanDraftBlockCount = 0
    var cleanSigmaByStep: [Int: [InCheckpointSampledMTPStepSigmaSample]] = [:]
    var cleanTruncationSupportSizeByStep: [Int: [Int]] = [:]
    var cleanTemperingByStep: [Int: [InCheckpointSampledMTPStepTemperingSample]] = [:]

    for (promptIndex, prompt) in prompts.enumerated() {
        let promptTokens = context.tokenizer.encode(text: prompt)

        // --- sampled MTP run ---
        let mtpCache = context.model.newCache(parameters: parameters)
        // Built from the SAME `parameters` value handed to `MTPSpeculativeTokenIterator` below (not
        // a second, independently invented truncation) -- this is what makes it structurally
        // impossible for the provider's stored truncation and this run's own request truncation to
        // disagree. See `SampledMTPSamplingTruncation`'s doc comment for why that cross-check is the
        // load-bearing safety property this depends on.
        let truncation = SampledMTPSamplingTruncation(parameters: parameters)
        let measuring = InCheckpointMeasuringSampledMTPBlockRuntimeProvider(
            inner: inCheckpointSampledMTPMakeProvider(
                parsed.provider, seed: parsed.seed, truncation: truncation),
            // Same `samplingReport` values already printed above and written into the JSON's
            // `sampling` field -- this run's ACTUAL target-side sampling configuration, not a
            // second, independently invented one. Never hardcodes either the truncated or
            // untruncated case: it always tracks whatever `--top-p`/`--top-k` this run was invoked
            // with.
            targetTemperature: samplingReport.temperature, targetTopP: samplingReport.topP,
            targetTopK: samplingReport.topK, targetMinP: samplingReport.minP)
        var mtpIterator = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(promptTokens)),
            mainModel: context.model,
            drafter: loadedDrafter.drafter,
            mainCache: mtpCache,
            parameters: parameters,
            blockSize: inCheckpointSampledMTPBlockSize,
            collectPhaseTelemetry: true,
            sampledBlockDecisionProvider: measuring)
        // Fail fast on prompt[0], BEFORE spending this prompt's full decode loop: see the doc
        // comment on `providerWiringNotEngaged` for why `passthroughReason` at this exact point
        // (immediately after construction, before `nextThrowing()` is ever called) genuinely
        // discriminates a provider/truncation wiring defect from a healthy run.
        if promptIndex == 0, let constructionPassthroughReason = mtpIterator.passthroughReason {
            throw InCheckpointSampledMTPAcceptanceCLIError.providerWiringNotEngaged(
                passthroughReason: constructionPassthroughReason)
        }
        var emitted = 0
        while emitted < parsed.maxTokens, let _ = try mtpIterator.nextThrowing() {
            emitted += 1
        }
        mtpIterator.finalizeGeneration()

        let proposedCount = mtpIterator.proposedCount
        let acceptedCount = mtpIterator.acceptedCount
        let passthroughReason = mtpIterator.passthroughReason
        let measuredBlockCount = measuring.blockOutcomes.count

        // Fix 4: print the observed counts BEFORE either control below can throw -- previously the
        // fatal throw preceded any print for this prompt, so a failing run emitted nothing to
        // diagnose from.
        print(
            "prompt[\(promptIndex)]: proposedCount=\(proposedCount) measuredBlockCount="
                + "\(measuredBlockCount) passthroughReason=\(passthroughReason ?? "nil")")

        guard proposedCount > 0 else {
            throw InCheckpointSampledMTPAcceptanceCLIError.zeroProposedDraftTokens(
                promptIndex: promptIndex, passthroughReason: passthroughReason)
        }
        // Fix 4: `proposedCount` increments unconditionally, even when EVERY `decide()` call threw
        // (`mtpIterator.proposedCount += numDraft` runs after the accept/reject block regardless of
        // outcome) -- so `proposedCount > 0` alone reads like a healthy stream even when zero
        // blocks were actually measured (exactly the state of the failed live run under the Fix 1
        // defect). This is the control that catches that failure mode.
        guard measuredBlockCount > 0 else {
            throw InCheckpointSampledMTPAcceptanceCLIError.zeroMeasuredBlocks(
                promptIndex: promptIndex, proposedCount: proposedCount,
                passthroughReason: passthroughReason)
        }

        let excluded = passthroughReason != nil
        let perStep = inCheckpointSampledMTPPerStepAcceptance(
            measuring.blockOutcomes, stepCount: inCheckpointSampledMTPNumDraft)
        let streamStepReports = perStep.map { step -> InCheckpointSampledMTPAcceptanceStepReport in
            let sigmaSamples = measuring.sigmaMinPQByStep[step.stepIndex] ?? []
            let sigmaSummary = inCheckpointSampledMTPSummarizeStepSigma(sigmaSamples, stepIndex: step.stepIndex)
            let supportSizes = measuring.truncationSupportSizeByStep[step.stepIndex] ?? []
            let bindingCounts = inCheckpointSampledMTPClassifyTruncationBindings(
                supportSizes: supportSizes, topK: samplingReport.topK)
            return InCheckpointSampledMTPAcceptanceStepReport(
                stepIndex1Based: step.stepIndex + 1,
                reachedCount: step.reachedCount,
                acceptedCount: step.acceptedCount,
                acceptanceRate: step.acceptanceRate,
                sigmaMinPQMean: sigmaSummary.unconditionalMean,
                sigmaMinPQBlockCount: sigmaSummary.unconditionalCount,
                sigmaMinPQMeanConditionedOnReached: sigmaSummary.conditionedMean,
                sigmaMinPQConditionedBlockCount: sigmaSummary.conditionedCount,
                topKBindingCount: bindingCounts.topKBindingCount,
                topPBindingCount: bindingCounts.topPBindingCount,
                untruncatedCount: bindingCounts.untruncatedCount,
                truncationSupportSize: bindingCounts.supportSizeSummary)
        }
        let streamAcceptance = inCheckpointSampledMTPPooledAcceptance(measuring.blockOutcomes)

        streamReports.append(
            InCheckpointSampledMTPAcceptanceStreamReport(
                promptIndex: promptIndex,
                promptPreview: String(prompt.prefix(80)),
                promptTokenCount: promptTokens.count,
                blockCount: measuring.blockOutcomes.count,
                proposedCount: proposedCount,
                acceptedCount: acceptedCount,
                passthroughReason: passthroughReason,
                excludedFromPooling: excluded,
                acceptanceRate: streamAcceptance,
                perStep: streamStepReports))

        print(
            "prompt[\(promptIndex)]: blocks=\(measuring.blockOutcomes.count) "
                + "proposed=\(proposedCount) accepted=\(acceptedCount) "
                + "a=\(streamAcceptance.map { String(format: "%.4f", $0) } ?? "n/a") "
                + "passthrough=\(passthroughReason ?? "nil") "
                + (excluded ? "EXCLUDED (sticky mid-stream)" : ""))

        if excluded {
            continue
        }

        cleanOutcomes.append(contentsOf: measuring.blockOutcomes)
        cleanDecideSeconds.append(contentsOf: measuring.decideDurationsSeconds)
        for (step, samples) in measuring.sigmaMinPQByStep {
            cleanSigmaByStep[step, default: []].append(contentsOf: samples)
        }
        for (step, supportSizes) in measuring.truncationSupportSizeByStep {
            cleanTruncationSupportSizeByStep[step, default: []].append(contentsOf: supportSizes)
        }
        for (step, temperingSamples) in measuring.temperingByStep {
            cleanTemperingByStep[step, default: []].append(contentsOf: temperingSamples)
        }
        let phases = mtpIterator.speculativeDecodingPhaseTelemetry
        cleanDraftBlockSeconds += phases.draftBlockSeconds
        cleanDraftBlockCount += phases.draftBlockCount

        // --- plain scalar reference run (measures T, the same run/build/host/session) ---
        let scalarCache = context.model.newCache(parameters: parameters)
        var scalarIterator = try TokenIterator(
            input: LMInput(tokens: MLXArray(promptTokens)),
            model: context.model,
            cache: scalarCache,
            parameters: parameters)
        var scalarCount = 0
        var lastTimestamp = ProcessInfo.processInfo.systemUptime
        while scalarCount < parsed.maxTokens, let _ = try scalarIterator.nextThrowing() {
            let now = ProcessInfo.processInfo.systemUptime
            cleanTargetStepSeconds.append(now - lastTimestamp)
            lastTimestamp = now
            scalarCount += 1
        }
    }

    guard !cleanOutcomes.isEmpty else {
        throw InCheckpointSampledMTPAcceptanceCLIError.noCleanStreams
    }

    let pooledPerStep = inCheckpointSampledMTPPerStepAcceptance(
        cleanOutcomes, stepCount: inCheckpointSampledMTPNumDraft)
    let pooledStepReports = pooledPerStep.map { step -> InCheckpointSampledMTPAcceptanceStepReport in
        let sigmaSamples = cleanSigmaByStep[step.stepIndex] ?? []
        let sigmaSummary = inCheckpointSampledMTPSummarizeStepSigma(sigmaSamples, stepIndex: step.stepIndex)
        let supportSizes = cleanTruncationSupportSizeByStep[step.stepIndex] ?? []
        let bindingCounts = inCheckpointSampledMTPClassifyTruncationBindings(
            supportSizes: supportSizes, topK: samplingReport.topK)
        return InCheckpointSampledMTPAcceptanceStepReport(
            stepIndex1Based: step.stepIndex + 1,
            reachedCount: step.reachedCount,
            acceptedCount: step.acceptedCount,
            acceptanceRate: step.acceptanceRate,
            sigmaMinPQMean: sigmaSummary.unconditionalMean,
            sigmaMinPQBlockCount: sigmaSummary.unconditionalCount,
            sigmaMinPQMeanConditionedOnReached: sigmaSummary.conditionedMean,
            sigmaMinPQConditionedBlockCount: sigmaSummary.conditionedCount,
            topKBindingCount: bindingCounts.topKBindingCount,
            topPBindingCount: bindingCounts.topPBindingCount,
            untruncatedCount: bindingCounts.untruncatedCount,
            truncationSupportSize: bindingCounts.supportSizeSummary)
    }
    let pooledAcceptance = inCheckpointSampledMTPPooledAcceptance(cleanOutcomes)
    let decideSummary = inCheckpointSampledMTPSummarize(cleanDecideSeconds)
    let targetStepSummary = inCheckpointSampledMTPSummarize(cleanTargetStepSeconds)
    let meanTargetStep = targetStepSummary?.mean
    // Fix 2: `cleanDraftBlockSeconds / cleanDraftBlockCount` is a PER-BLOCK figure (`recordDraftBlock`
    // is called once per block with the SUM of that block's draft-loop AND commit-seed forwards --
    // see `inCheckpointSampledMTPDrafterForwardsPerBlock`'s doc comment). Dividing by that constant
    // converts it to PER-FORWARD before dividing by `meanTargetStep`, matching the predeclaration's
    // `delta` (per-drafter-forward cost as a fraction of a target step) -- the break-even formula
    // below already multiplies by 2 to reconstruct the per-block cost; passing the per-block figure
    // directly here would have double-counted it (the defect this fix closes).
    let deltaPerForward: Double? = (cleanDraftBlockCount > 0 && meanTargetStep.map { $0 > 0 } == true)
        ? (cleanDraftBlockSeconds / Double(cleanDraftBlockCount)
            / Double(inCheckpointSampledMTPDrafterForwardsPerBlock)) / meanTargetStep!
        : nil
    let dOverT: Double? = (decideSummary.map { $0.mean } != nil && meanTargetStep.map { $0 > 0 } == true)
        ? decideSummary!.mean / meanTargetStep!
        : nil
    let breakEven: Double? = (deltaPerForward != nil && dOverT != nil)
        ? inCheckpointSampledMTPBreakEvenAcceptance(delta: deltaPerForward!, dOverT: dOverT!)
        : nil

    let pooledReport = InCheckpointSampledMTPAcceptancePooledReport(
        cleanStreamCount: streamReports.filter { !$0.excludedFromPooling }.count,
        excludedStreamCount: streamReports.filter { $0.excludedFromPooling }.count,
        blockCount: cleanOutcomes.count,
        proposedCount: cleanOutcomes.reduce(0) { $0 + $1.proposedCount },
        acceptedCount: cleanOutcomes.reduce(0) { $0 + $1.acceptedDraftCount },
        acceptanceRate: pooledAcceptance,
        perStep: pooledStepReports,
        decideSecondsPerBlock: decideSummary,
        decideCostExcludesTargetForward: decideSummary != nil ? true : nil,
        targetStepSecondsPerToken: targetStepSummary,
        deltaPerForward: deltaPerForward,
        dOverT: dOverT,
        breakEvenAcceptance: breakEven)

    print("--- pooled across \(pooledReport.cleanStreamCount) clean stream(s), "
        + "\(pooledReport.excludedStreamCount) excluded ---")
    print(
        "a_pooled = \(pooledReport.acceptedCount)/\(pooledReport.proposedCount) = "
            + (pooledAcceptance.map { String(format: "%.4f", $0) } ?? "n/a"))
    for step in pooledStepReports {
        // The CONDITIONED mean is the like-for-like comparison against `a_step` (both are computed
        // only over blocks that reached this step); the UNCONDITIONAL mean pools every block
        // regardless of whether it reached this step, and is a genuinely different quantity for
        // step > 1 -- report both, never collapse to one.
        print(
            "a_step\(step.stepIndex1Based) = \(step.acceptedCount)/\(step.reachedCount) = "
                + (step.acceptanceRate.map { String(format: "%.4f", $0) } ?? "n/a")
                + " | sigmaMinPQ_step\(step.stepIndex1Based) conditioned (comparable) mean="
                + (step.sigmaMinPQMeanConditionedOnReached.map { String(format: "%.4f", $0) } ?? "n/a")
                + " n=\(step.sigmaMinPQConditionedBlockCount)"
                + " | unconditional mean="
                + (step.sigmaMinPQMean.map { String(format: "%.4f", $0) } ?? "n/a")
                + " n=\(step.sigmaMinPQBlockCount)")
    }
    // Pooled ACROSS ALL STEPS (not per-step, unlike the loop above) -- makes "measured at
    // topP=\(samplingReport.topP) topK=\(samplingReport.topK)" evidence rather than an assertion:
    // which filter actually bound at each drafted position, and how wide the surviving support was.
    let allTruncationSupportSizes = cleanTruncationSupportSizeByStep.values.flatMap { $0 }
    let pooledBindingCounts = inCheckpointSampledMTPClassifyTruncationBindings(
        supportSizes: allTruncationSupportSizes, topK: samplingReport.topK)
    let pooledPositionCount = allTruncationSupportSizes.count
    if pooledPositionCount > 0 {
        func fraction(_ count: Int) -> String {
            String(format: "%.4f", Double(count) / Double(pooledPositionCount))
        }
        print(
            "truncation binding (pooled across all steps, n=\(pooledPositionCount)): "
                + "topKBinding=\(pooledBindingCounts.topKBindingCount) (\(fraction(pooledBindingCounts.topKBindingCount))) "
                + "topPBinding=\(pooledBindingCounts.topPBindingCount) (\(fraction(pooledBindingCounts.topPBindingCount))) "
                + "untruncated=\(pooledBindingCounts.untruncatedCount) (\(fraction(pooledBindingCounts.untruncatedCount)))")
        if let supportSummary = pooledBindingCounts.supportSizeSummary {
            print(
                "truncation support size (pooled): min=\(supportSummary.min) "
                    + "median=\(String(format: "%.1f", supportSummary.median)) max=\(supportSummary.max) "
                    + "n=\(supportSummary.count)")
        }
    }

    // Temperature counterfactual replay -- see
    // docs/task-inbox/2026-09-09-sampled-mtp-temperature-counterfactual-PREDECLARATION.md. Console-
    // only, mirroring the pooled-across-all-steps truncation-binding block above: an ADDITIONAL view
    // over the SAME per-step storage (`cleanTemperingByStep`/`cleanTruncationSupportSizeByStep`),
    // not a new JSON field.
    let allTempering = cleanTemperingByStep.values.flatMap { $0 }
    if !allTempering.isEmpty {
        let temperingPositionCount = allTempering.count
        let sigmaTMeans = inCheckpointSampledMTPCounterfactualTemperatures.indices.map { index in
            allTempering.reduce(0.0) { $0 + $1.sigmaByTemperature[index] }
                / Double(temperingPositionCount)
        }
        print(
            "sigma_T (pooled across all steps, n=\(temperingPositionCount)): "
                + zip(inCheckpointSampledMTPCounterfactualTemperatures, sigmaTMeans)
                    .map { temperature, mean in "T=\(temperature)=\(String(format: "%.4f", mean))" }
                    .joined(separator: " "))

        let sigma1Mean = sigmaTMeans[0]
        let sigma07Mean = sigmaTMeans[3]
        let delta: Double? = sigma1Mean > 0 ? (sigma1Mean - sigma07Mean) / sigma1Mean : nil
        print(
            "Delta = (sigma_1 - sigma_0.7) / sigma_1 (pooled across all steps) = "
                + (delta.map { String(format: "%.4f", $0) } ?? "n/a"))

        let sigmaArgmaxMean =
            allTempering.reduce(0.0) { $0 + $1.sigmaArgmax } / Double(temperingPositionCount)
        let sigmaRestMean =
            allTempering.reduce(0.0) { $0 + $1.sigmaRest } / Double(temperingPositionCount)
        print(
            "sigma_argmax (pooled) = \(String(format: "%.4f", sigmaArgmaxMean)) "
                + "sigma_rest (pooled) = \(String(format: "%.4f", sigmaRestMean)) "
                + "n=\(temperingPositionCount)")

        // C1 (exact, non-statistical): at any position with a single-token surviving support, the
        // law is a point mass at every temperature, so `sigma_T` must equal `sigma_1` EXACTLY --
        // compared with `!=` on `Double`, never a tolerance. A non-zero count here means the
        // instrument is wrong and the run is void; see the predeclaration's C1 control.
        var c1ViolationCount = 0
        var c1FirstOffender: (supportSize: Int, temperature: Double, sigmaT: Double, sigma1: Double)?
        // C2 (discriminating): a non-zero count here is what distinguishes a genuine measurement
        // from an instrument that silently returns `sigma_T == sigma_1` everywhere.
        var c2Count = 0
        for step in cleanTemperingByStep.keys.sorted() {
            let supportSizes = cleanTruncationSupportSizeByStep[step] ?? []
            let temperingSamples = cleanTemperingByStep[step] ?? []
            precondition(
                supportSizes.count == temperingSamples.count,
                "C1/C2 require supportSize and tempering samples recorded in lockstep per step "
                    + "(step=\(step): supportSizes.count=\(supportSizes.count) "
                    + "temperingSamples.count=\(temperingSamples.count)) -- both are appended in "
                    + "the same loop iteration in measuredDecide, so a mismatch here means that "
                    + "invariant broke")
            for (supportSize, sample) in zip(supportSizes, temperingSamples) {
                let sigma1 = sample.sigmaByTemperature[0]
                if supportSize == 1 {
                    var violation: (temperature: Double, sigmaT: Double)?
                    for (index, temperature) in inCheckpointSampledMTPCounterfactualTemperatures
                        .enumerated()
                    {
                        let sigmaT = sample.sigmaByTemperature[index]
                        if sigmaT != sigma1 {
                            violation = (temperature, sigmaT)
                            break
                        }
                    }
                    if let violation {
                        c1ViolationCount += 1
                        if c1FirstOffender == nil {
                            c1FirstOffender = (
                                supportSize: supportSize, temperature: violation.temperature,
                                sigmaT: violation.sigmaT, sigma1: sigma1)
                        }
                    }
                } else if supportSize > 1 {
                    let sigma07 = sample.sigmaByTemperature[3]
                    if sigma07 != sigma1 {
                        c2Count += 1
                    }
                }
            }
        }
        print(
            "C1 (supportSize==1 temperature-invariance, exact `!=`): violations=\(c1ViolationCount)")
        if let c1FirstOffender {
            print(
                "C1 FIRST OFFENDER: supportSize=\(c1FirstOffender.supportSize) "
                    + "T=\(c1FirstOffender.temperature) sigma_T=\(c1FirstOffender.sigmaT) "
                    + "sigma_1=\(c1FirstOffender.sigma1)")
        }
        print("C2 (supportSize>1 AND sigma_0.7 != sigma_1, discriminating): count=\(c2Count)")
    }

    if let decideSummary {
        print(
            "D (decide wall time/block, decide()-CPU-only -- EXCLUDES the target forward, see "
                + "decideCostExcludesTargetForward): mean=\(String(format: "%.6f", decideSummary.mean))s "
                + "range=[\(String(format: "%.6f", decideSummary.min)),"
                + "\(String(format: "%.6f", decideSummary.max))]s n=\(decideSummary.count)")
    }
    if let targetStepSummary {
        print(
            "T (target scalar step wall time): mean=\(String(format: "%.6f", targetStepSummary.mean))s "
                + "n=\(targetStepSummary.count)")
    }
    if let deltaPerForward {
        print(
            "deltaPerForward (drafter forward / T, PER FORWARD not per block) = "
                + "\(String(format: "%.6f", deltaPerForward))")
    }
    if let dOverT {
        print("D/T = \(String(format: "%.6f", dOverT))")
    }
    if let breakEven {
        print(
            "a* (break-even, informational, not a gate) = \(String(format: "%.4f", breakEven))")
    }
    if let pooledAcceptance {
        let verdict: String
        if pooledAcceptance >= 0.80 {
            verdict = "ACCEPT band (a >= 0.80)"
        } else if pooledAcceptance <= 0.55 {
            verdict = "REJECT band (a <= 0.55)"
        } else {
            verdict = "GATED band (0.55 < a < 0.80)"
        }
        print("predeclared band (informational, not enforced by this CLI): \(verdict)")
    }

    // Fix 4: enforce the predeclared sample-size floor over the CLEAN pooled sample. Refuses
    // (non-zero exit, no --output-json written) rather than silently reporting an underpowered
    // measurement -- e.g. a one-line prompts file with `--max-tokens 8` must not exit 0.
    let sampleSizeReport = InCheckpointSampledMTPAcceptanceSampleSizeReport(
        proposedDraftTokenCount: pooledReport.proposedCount,
        requiredProposedDraftTokenCount: inCheckpointSampledMTPRequiredProposedDraftTokenCount,
        blockCount: pooledReport.blockCount,
        requiredBlockCount: inCheckpointSampledMTPRequiredBlockCount,
        cleanPromptCount: pooledReport.cleanStreamCount,
        requiredCleanPromptCount: inCheckpointSampledMTPRequiredCleanPromptCount,
        meetsPredeclaredSampleSize: pooledReport.proposedCount
            >= inCheckpointSampledMTPRequiredProposedDraftTokenCount
            && pooledReport.blockCount >= inCheckpointSampledMTPRequiredBlockCount
            && pooledReport.cleanStreamCount >= inCheckpointSampledMTPRequiredCleanPromptCount)
    print(
        "sample size: proposedDraftTokenCount=\(sampleSizeReport.proposedDraftTokenCount) "
            + "(required >= \(sampleSizeReport.requiredProposedDraftTokenCount)) "
            + "blockCount=\(sampleSizeReport.blockCount) "
            + "(required >= \(sampleSizeReport.requiredBlockCount)) "
            + "cleanPromptCount=\(sampleSizeReport.cleanPromptCount) "
            + "(required >= \(sampleSizeReport.requiredCleanPromptCount)) "
            + "meetsPredeclaredSampleSize=\(sampleSizeReport.meetsPredeclaredSampleSize)")
    guard sampleSizeReport.meetsPredeclaredSampleSize else {
        throw InCheckpointSampledMTPAcceptanceCLIError.sampleSizeBelowPredeclaredFloor(
            proposedDraftTokenCount: sampleSizeReport.proposedDraftTokenCount,
            requiredProposedDraftTokenCount: sampleSizeReport.requiredProposedDraftTokenCount,
            blockCount: sampleSizeReport.blockCount,
            requiredBlockCount: sampleSizeReport.requiredBlockCount,
            cleanPromptCount: sampleSizeReport.cleanPromptCount,
            requiredCleanPromptCount: sampleSizeReport.requiredCleanPromptCount)
    }

    // Fix 5: the independent cross-check, made an ENFORCED verdict rather than two printed
    // numbers. Per-step: compares each step's empirical accepted/reached ratio (95% Wilson CI)
    // against the like-for-like sigmaMinPQ conditioned mean. Pooled: every step index's
    // reached/accepted TRIALS pooled into one Bernoulli sample (distinct from the headline
    // `a_pooled`, which pools `acceptedDraftCount` over `proposedCount` -- a different quantity not
    // framed as per-step Bernoulli trials) against the trial-weighted mean of the per-step
    // conditioned sigma means.
    let perStepAgreement = pooledStepReports.map { step in
        inCheckpointSampledMTPAgreementCheck(
            stepIndex1Based: step.stepIndex1Based,
            acceptedCount: step.acceptedCount,
            reachedCount: step.reachedCount,
            sigmaMean: step.sigmaMinPQMeanConditionedOnReached)
    }
    let pooledTrialReached = pooledStepReports.reduce(0) { $0 + $1.reachedCount }
    let pooledTrialAccepted = pooledStepReports.reduce(0) { $0 + $1.acceptedCount }
    let pooledSigmaWeightedNumerator = pooledStepReports.reduce(0.0) {
        $0 + ($1.sigmaMinPQMeanConditionedOnReached ?? 0) * Double($1.sigmaMinPQConditionedBlockCount)
    }
    let pooledSigmaWeightedDenominator = pooledStepReports.reduce(0) {
        $0 + $1.sigmaMinPQConditionedBlockCount
    }
    let pooledSigmaMean: Double? =
        pooledSigmaWeightedDenominator > 0
        ? pooledSigmaWeightedNumerator / Double(pooledSigmaWeightedDenominator) : nil
    let pooledAgreement = inCheckpointSampledMTPAgreementCheck(
        stepIndex1Based: nil, acceptedCount: pooledTrialAccepted, reachedCount: pooledTrialReached,
        sigmaMean: pooledSigmaMean)

    for check in perStepAgreement + [pooledAgreement] {
        print("agreement " + inCheckpointSampledMTPDescribeAgreementCheck(check))
    }
    let disagreements = (perStepAgreement + [pooledAgreement]).filter { $0.verdict == .disagree }
    guard disagreements.isEmpty else {
        // Names the EFFECTIVE truncation parameters `p` (the target side) was predicted under, so
        // a future disagreement report states which configuration it was measuring -- e.g.
        // distinguishing "the untruncated cross-check disagreed" from "the truncated cross-check
        // disagreed at this topP/topK/minP", rather than leaving a reader to guess.
        let detail = disagreements.map(inCheckpointSampledMTPDescribeAgreementCheck)
            .joined(separator: "; ")
            + " -- target `p` predicted under temperature=\(samplingReport.temperature) "
            + "topP=\(samplingReport.topP) topK=\(samplingReport.topK) minP=\(samplingReport.minP)"
        throw InCheckpointSampledMTPAcceptanceCLIError.sigmaMinPQDisagreement(detail)
    }
    let agreementReport = InCheckpointSampledMTPAcceptanceAgreementReport(
        perStep: perStepAgreement,
        pooled: pooledAgreement,
        allAgree: (perStepAgreement + [pooledAgreement]).allSatisfy { $0.verdict != .disagree })

    let report = InCheckpointSampledMTPAcceptanceReport(
        schemaVersion: 2,
        modelPath: parsed.modelPath,
        ngramOffloadPlanPath: parsed.ngramOffloadPlanPath,
        promptsFilePath: parsed.promptsFilePath,
        maxTokens: parsed.maxTokens,
        seed: parsed.seed,
        provider: parsed.provider.rawValue,
        buildConfiguration: "release",
        blockSize: inCheckpointSampledMTPBlockSize,
        numDraft: inCheckpointSampledMTPNumDraft,
        sampling: samplingReport,
        streams: streamReports,
        pooled: pooledReport,
        sampleSize: sampleSizeReport,
        agreement: agreementReport)

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data: Data
    do {
        data = try encoder.encode(report)
    } catch {
        throw InCheckpointSampledMTPAcceptanceCLIError.outputJSONWriteFailed
    }
    let descriptor = outputURL.withUnsafeFileSystemRepresentation { path -> Int32 in
        guard let path else { return -1 }
        return open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0o600)
    }
    guard descriptor >= 0 else {
        throw InCheckpointSampledMTPAcceptanceCLIError.outputJSONMustBeNew
    }
    defer { _ = close(descriptor) }
    do {
        try FileHandle(fileDescriptor: descriptor, closeOnDealloc: false).write(contentsOf: data)
    } catch {
        throw InCheckpointSampledMTPAcceptanceCLIError.outputJSONWriteFailed
    }
    print("qwen4exp-sampled-mtp-acceptance PASS output=\(parsed.outputJSONPath)")
    #endif
}

/// `truncation` is REQUIRED (no default): every call site must build it from the SAME
/// `GenerateParameters` value the caller hands to `MTPSpeculativeTokenIterator`
/// (`SampledMTPSamplingTruncation(parameters:)`), so the constructed provider's stored truncation
/// and the run's own request truncation cannot disagree. A default of `.untruncated` here would
/// silently re-introduce exactly the wiring gap this function exists to close if a future call site
/// forgot to pass one.
func inCheckpointSampledMTPMakeProvider(
    _ kind: InCheckpointSampledMTPAcceptanceProviderKind,
    seed: UInt64,
    truncation: SampledMTPSamplingTruncation
) -> any SampledMTPBlockRuntimeDeciding {
    switch kind {
    case .seeded: return SeededSampledMTPBlockRuntimeProvider(seed: seed, truncation: truncation)
    case .nondeterministic: return NondeterministicSampledMTPBlockRuntimeProvider(truncation: truncation)
    }
}
