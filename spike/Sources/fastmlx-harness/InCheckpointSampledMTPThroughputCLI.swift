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
// Measures `S_direct = median over prompts of (decode tok/s [speculative arm] / decode tok/s
// [scalar arm])` for `qwen4_exp` (Qwen3.8-Flash-Next)'s in-checkpoint MTP drafter, driven by
// `MTPSpeculativeTokenIterator` with a sampled block decision provider, against a real checkpoint
// on the SAME release build/host/session -- never composed from a prior measurement.
//
// Contract: `docs/task-inbox/2026-09-09-sampled-mtp-direct-throughput-PREDECLARATION.md`. This is
// a NEW, separate quantity from `qwen4exp-sampled-mtp-acceptance`'s `a`/`D`/`T`/`delta` model --
// see that predeclaration's "Why a NEW predeclaration" section for why the two cannot be merged.
//
// `collectPhaseTelemetry: false` and NO measuring/tee wrapper around the provider are correctness
// requirements for this measurement, not optimizations: telemetry forces GPU sync at phase
// boundaries and the acceptance instrument's tee adds two full-vocabulary `Double` softmaxes per
// block, both of which would perturb the exact quantity being timed
// (predeclaration, "Method, and the parts of it that are forced").

// MARK: - Pinned structural constants (qwen4_exp in-checkpoint MTP)
//
// Identical values to `InCheckpointSampledMTPAcceptanceCLI.swift`'s own pinned constants --
// duplicated here (rather than shared) because that file's constants are `let` globals scoped to
// this same module already; redeclaring under a distinct name would only invite the two to drift.
// Reusing the SAME global names here would be a duplicate-symbol build error, so this file reuses
// `inCheckpointSampledMTPBlockSize` etc. directly rather than redefining them.

/// The predeclaration's sample-size floor (>= 8 prompts, >= 256 decoded tokens per arm per
/// prompt -- "Sample size and reporting").
let inCheckpointSampledMTPThroughputRequiredPromptCount = 8
let inCheckpointSampledMTPThroughputRequiredMaxTokens = 256

/// C2's target interval center: the acceptance measurement's pooled-over-proposed implied
/// acceptance rate on record (`docs/task-inbox/2026-09-09-sampled-mtp-acceptance-MEASURED.md`).
/// Named so the +/-0.05 absolute tolerance below is legible independent of this literal.
let inCheckpointSampledMTPThroughputExpectedImpliedAcceptance = 0.6879
let inCheckpointSampledMTPThroughputImpliedAcceptanceToleranceAbsolute = 0.05

/// C3's target: the reciprocal of the independently measured `T = 0.0381` s/token from the
/// acceptance run on the same host/artifact (predeclaration, control 3).
let inCheckpointSampledMTPThroughputExpectedScalarTokensPerSecond = 26.24
let inCheckpointSampledMTPThroughputScalarTokensPerSecondToleranceFraction = 0.15

/// The predeclared band boundaries (predeclaration, "Predeclared bands"). Both the REJECT
/// median ceiling and the ACCEPT per-prompt floor share this same numeric value (1.10) by the
/// predeclaration's own text; one named constant for both keeps that coincidence visible instead
/// of duplicating the literal.
let inCheckpointSampledMTPThroughputBandLowThreshold = 1.10
let inCheckpointSampledMTPThroughputBandHighThreshold = 1.30

// MARK: - CLI arguments
//
// `provider` reuses `InCheckpointSampledMTPAcceptanceProviderKind` (declared in
// `InCheckpointSampledMTPAcceptanceCLI.swift`, same module) rather than a second, parallel enum:
// this CLI calls that file's own `inCheckpointSampledMTPMakeProvider` factory unchanged, so a
// distinct type here would only require a conversion at every call site for no benefit.

struct InCheckpointSampledMTPThroughputArguments: Equatable, Sendable {
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

enum InCheckpointSampledMTPThroughputCLIError: Error, Equatable, CustomStringConvertible, Sendable {
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
    /// The predeclaration's sample-size floor (>= 8 prompts, `--max-tokens` >= 256) was not met
    /// BEFORE any model load or measurement -- refuses to spend a real run on an underpowered
    /// request rather than measure it and refuse only at the end.
    case sampleSizeBelowPredeclaredFloor(
        promptCount: Int, requiredPromptCount: Int,
        maxTokens: Int, requiredMaxTokens: Int)
    /// An arm's iterator stopped (returned `nil`, e.g. EOS) before emitting the requested
    /// `--max-tokens` tokens. A short run's tok/s is not comparable to a full one and must not be
    /// silently pooled into the ratio -- this is the same "a truncated sample manufactures a
    /// defect" lesson this project has already paid for once (see
    /// `docs/task-inbox/` truncated-sample record).
    case armEndedEarly(
        arm: String, promptIndex: Int, emittedTokenCount: Int, requestedMaxTokens: Int)
    /// Fires ONLY for `promptIndex == 0`'s speculative arm, immediately after that prompt's
    /// `MTPSpeculativeTokenIterator` is constructed and BEFORE a single token is generated:
    /// `passthroughReason` was already non-nil at construction time, meaning `providerIsEligible`
    /// was false at `init` (`MTPSpeculativeTokenIterator.swift:171-186`) -- e.g. the provider's
    /// stored `SampledMTPSamplingTruncation` did not match this run's own `parameters`, so
    /// `supports()` refused it. `passthroughReason` is nil at this exact point in EVERY legitimate
    /// engaged run (set unconditionally, and only, when `providerIsEligible` is false), so this
    /// cannot spuriously fire on a healthy stream, and it catches a provider/truncation wiring
    /// defect before spending the cost of a full `--max-tokens` decode on both arms, rather than
    /// only after C1 below has already run the whole prompt.
    case providerWiringNotEngaged(passthroughReason: String)
    /// C1: the speculative arm never proposed a draft token, or went sticky passthrough, for this
    /// prompt. This is the predeclaration's single most important control -- a silently
    /// non-speculating arm degenerates to the scalar arm and produces a ratio of ~1.00, which
    /// reads identically to an honest "does not help" result unless this control separates them.
    case speculationNotEngaged(promptIndex: Int, proposedCount: Int, passthroughReason: String?)
    /// C2: the pooled implied acceptance rate (accepted/proposed across every prompt) fell outside
    /// the acceptance measurement's interval -- the telemetry-free configuration measured here is
    /// not decoding the same way the measured one did, and the ratio cannot be attributed.
    case impliedAcceptanceOutOfBand(
        observed: Double, proposedCount: Int, acceptedCount: Int,
        expected: Double, toleranceAbsolute: Double)
    /// C3: the scalar arm's pooled mean tok/s is not consistent with the independently measured
    /// `T` from the acceptance run on the same host/artifact -- this run is measuring something
    /// other than what `T` measured.
    case scalarThroughputOutOfBand(
        observed: Double, expected: Double, toleranceFraction: Double)

    var description: String {
        switch self {
        case .missingFlag(let flag): return "missing required \(flag)"
        case .duplicateFlag(let flag): return "duplicate \(flag)"
        case .unknownFlag(let flag): return "unknown flag \(flag)"
        case .missingValue(let flag): return "\(flag) requires a value"
        case .unexpectedPositional: return "unexpected positional argument"
        case .releaseBuildRequired:
            return "qwen4exp-sampled-mtp-throughput requires a Release build: a DEBUG build "
                + "measures this workload roughly 70x slower and a tok/s ratio measured under "
                + "that regime would be meaningless -- refusing rather than emitting misleading "
                + "numbers"
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
        case .sampleSizeBelowPredeclaredFloor(
            let promptCount, let requiredPromptCount, let maxTokens, let requiredMaxTokens):
            return "ABORT: sample size below the predeclared floor "
                + "(docs/task-inbox/2026-09-09-sampled-mtp-direct-throughput-PREDECLARATION.md, "
                + "\"Sample size and reporting\"): promptCount=\(promptCount) "
                + "(required >= \(requiredPromptCount)), maxTokens=\(maxTokens) "
                + "(required >= \(requiredMaxTokens)) -- refusing to write --output-json or "
                + "report a ratio from an underpowered sample"
        case .armEndedEarly(let arm, let promptIndex, let emittedTokenCount, let requestedMaxTokens):
            return "ABORT: prompt[\(promptIndex)] \(arm) arm emitted only "
                + "\(emittedTokenCount)/\(requestedMaxTokens) requested tokens before its "
                + "iterator ended -- a truncated arm's tok/s is not comparable to a full one; "
                + "refusing to pool it into the ratio"
        case .providerWiringNotEngaged(let passthroughReason):
            return "ABORT: prompt[0]'s speculative arm iterator was already in passthrough "
                + "(passthroughReason=\(passthroughReason)) IMMEDIATELY after construction, before "
                + "any token was generated -- the sampled MTP provider's truncation does not match "
                + "this run's own sampling parameters (or another construction-time ineligibility), "
                + "so supports() refused it; refusing rather than silently measuring a ratio built "
                + "on a speculative arm that never actually speculated"
        case .speculationNotEngaged(let promptIndex, let proposedCount, let passthroughReason):
            return "ABORT: prompt[\(promptIndex)] speculative arm proposedCount=\(proposedCount) "
                + "passthroughReason=\(passthroughReason ?? "nil") -- speculation did not engage "
                + "for this prompt; a silently non-speculating arm degenerates to the scalar arm "
                + "and would read as an honest ~1.00x result. Refusing to report a ratio built on "
                + "this prompt."
        case .impliedAcceptanceOutOfBand(
            let observed, let proposedCount, let acceptedCount, let expected, let toleranceAbsolute):
            return "ABORT: pooled implied acceptance \(String(format: "%.4f", observed)) "
                + "(accepted=\(acceptedCount)/proposed=\(proposedCount)) is outside "
                + "\(String(format: "%.4f", expected)) +/- \(toleranceAbsolute) -- the "
                + "telemetry-free configuration measured here is not decoding the same way the "
                + "acceptance measurement did; the ratio cannot be attributed"
        case .scalarThroughputOutOfBand(let observed, let expected, let toleranceFraction):
            return "ABORT: scalar-arm mean tok/s \(String(format: "%.4f", observed)) is outside "
                + "\(String(format: "%.4f", expected)) +/- \(String(format: "%.0f", toleranceFraction * 100))% "
                + "-- this run is measuring something other than what the independently measured "
                + "T = 0.0381 s/token measured; the comparison is void"
        }
    }
}

func parseInCheckpointSampledMTPThroughputArguments(
    _ arguments: [String]
) throws -> InCheckpointSampledMTPThroughputArguments {
    let requiredFlags: Set<String> = [
        "--model-path", "--ngram-offload-plan", "--prompts-file", "--max-tokens", "--seed",
        "--provider", "--output-json",
    ]
    // `--top-p`/`--top-k` are OPTIONAL (see the doc comment on
    // `InCheckpointSampledMTPThroughputArguments.topP`/`.topK`) -- present in `allowed` (so they are
    // accepted at all) but deliberately absent from `requiredFlags`, so every existing invocation
    // that omits them keeps working unchanged.
    let optionalFlags: Set<String> = ["--top-p", "--top-k"]
    let allowed = requiredFlags.union(optionalFlags)
    var values: [String: String] = [:]
    var index = 0
    while index < arguments.count {
        let flag = arguments[index]
        guard flag.hasPrefix("--") else {
            throw InCheckpointSampledMTPThroughputCLIError.unexpectedPositional
        }
        guard allowed.contains(flag) else {
            throw InCheckpointSampledMTPThroughputCLIError.unknownFlag(flag)
        }
        guard values[flag] == nil else {
            throw InCheckpointSampledMTPThroughputCLIError.duplicateFlag(flag)
        }
        guard index + 1 < arguments.count, !arguments[index + 1].hasPrefix("--") else {
            throw InCheckpointSampledMTPThroughputCLIError.missingValue(flag)
        }
        values[flag] = arguments[index + 1]
        index += 2
    }
    for flag in requiredFlags where values[flag] == nil {
        throw InCheckpointSampledMTPThroughputCLIError.missingFlag(flag)
    }

    let modelPath = values["--model-path"]!
    guard modelPath.hasPrefix("/") else {
        throw InCheckpointSampledMTPThroughputCLIError.modelPathMustBeAbsolute
    }
    let ngramOffloadPlanPath = values["--ngram-offload-plan"]!
    guard ngramOffloadPlanPath.hasPrefix("/") else {
        throw InCheckpointSampledMTPThroughputCLIError.ngramOffloadPlanMustBeAbsolute
    }
    let promptsFilePath = values["--prompts-file"]!
    guard promptsFilePath.hasPrefix("/") else {
        throw InCheckpointSampledMTPThroughputCLIError.promptsFileMustBeAbsolute
    }
    let outputJSONPath = values["--output-json"]!
    guard outputJSONPath.hasPrefix("/") else {
        throw InCheckpointSampledMTPThroughputCLIError.outputJSONMustBeAbsolute
    }

    let rawMaxTokens = values["--max-tokens"]!
    guard let maxTokens = Int(rawMaxTokens), maxTokens > 0 else {
        throw InCheckpointSampledMTPThroughputCLIError.invalidMaxTokens(rawMaxTokens)
    }
    let rawSeed = values["--seed"]!
    guard let seed = UInt64(rawSeed) else {
        throw InCheckpointSampledMTPThroughputCLIError.invalidSeed(rawSeed)
    }
    let rawProvider = values["--provider"]!
    guard let provider = InCheckpointSampledMTPAcceptanceProviderKind(rawValue: rawProvider) else {
        throw InCheckpointSampledMTPThroughputCLIError.invalidProvider(rawProvider)
    }

    // Defaults are the untruncated identity (`topP = 1`, `topK = 0`) -- an invocation that omits
    // both flags measures EXACTLY what this CLI measured before they existed. Validated against
    // precisely what `supports()` (`sharedSampledMTPSupportsPredicate`) accepts, at parse time,
    // BEFORE any model load or measurement is spent.
    let rawTopP = values["--top-p"] ?? "1"
    guard let topP = Double(rawTopP), topP > 0, topP <= 1 else {
        throw InCheckpointSampledMTPThroughputCLIError.invalidTopP(rawTopP)
    }
    let rawTopK = values["--top-k"] ?? "0"
    guard let topK = Int(rawTopK), topK >= 0 else {
        throw InCheckpointSampledMTPThroughputCLIError.invalidTopK(rawTopK)
    }

    return InCheckpointSampledMTPThroughputArguments(
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

func inCheckpointSampledMTPThroughputExternalDiagnostic(_ error: Error) -> String {
    if let error = error as? InCheckpointSampledMTPThroughputCLIError {
        return error.description
    }
    return "qwen4exp-sampled-mtp-throughput failed: \(error)"
}

// MARK: - Pure arithmetic (independent of MLX; unit-tested directly)

/// Decode-only tok/s: `(emittedTokenCount - 1) / elapsedSeconds`. The `-1` matches the timing
/// protocol both arms share (predeclaration, "Method"): `t0` is taken AFTER the first token
/// already returned, so the first token's own decode step is excluded from `elapsedSeconds`, and
/// the numerator must exclude it too -- otherwise the rate would be inflated by counting a token
/// whose cost was never in the timed window.
///
/// Returns `nil` (never a fabricated rate) when there are fewer than 2 emitted tokens (nothing
/// timed) or the elapsed window is non-positive (a clock or ordering defect, not a real zero-cost
/// decode).
func inCheckpointSampledMTPThroughputTokensPerSecond(
    emittedTokenCount: Int, elapsedSeconds: Double
) -> Double? {
    guard emittedTokenCount > 1, elapsedSeconds > 0 else { return nil }
    return Double(emittedTokenCount - 1) / elapsedSeconds
}

/// The median of a set of per-prompt speculative/scalar tok/s ratios. `nil` for an empty input --
/// never fabricated as `0` or `1`. Even counts average the two central (sorted) values.
func inCheckpointSampledMTPThroughputMedianRatio(_ ratios: [Double]) -> Double? {
    guard !ratios.isEmpty else { return nil }
    let sorted = ratios.sorted()
    let middle = sorted.count / 2
    if sorted.count % 2 == 1 {
        return sorted[middle]
    }
    return (sorted[middle - 1] + sorted[middle]) / 2
}

enum InCheckpointSampledMTPThroughputBand: String, Codable, Equatable, Sendable {
    case accept = "ACCEPT"
    case reject = "REJECT"
    case gated = "GATED"
}

/// The predeclaration's INFORMATIONAL band classification (never enforced by this CLI as an exit
/// code) -- see "Predeclared bands":
///   - REJECT: `median <= 1.10`.
///   - ACCEPT: `median >= 1.30` AND `minPerPromptRatio >= 1.10`.
///   - GATED: everything else -- including the case where the median clears 1.30 but at least one
///     per-prompt ratio is below 1.10. That case is deliberately GATED, not ACCEPT: "a median
///     carried by two prompts while others regress is not a speedup a user experiences"
///     (predeclaration).
/// REJECT is checked first so the two boundaries (both literally `1.10`) cannot both match and
/// leave the outcome to case-order ambiguity: a median at or below 1.10 is REJECT regardless of
/// what any single per-prompt ratio is.
func inCheckpointSampledMTPThroughputBand(
    medianRatio: Double, minPerPromptRatio: Double
) -> InCheckpointSampledMTPThroughputBand {
    if medianRatio <= inCheckpointSampledMTPThroughputBandLowThreshold {
        return .reject
    }
    if medianRatio >= inCheckpointSampledMTPThroughputBandHighThreshold
        && minPerPromptRatio >= inCheckpointSampledMTPThroughputBandLowThreshold
    {
        return .accept
    }
    return .gated
}

/// C1, per prompt: `proposedCount > 0 AND passthroughReason == nil`. Verified at
/// `MTPSpeculativeTokenIterator.swift:796-797`: both counters increment OUTSIDE the
/// `collectPhaseTelemetry` guard, so this control is valid with telemetry off.
func inCheckpointSampledMTPThroughputSpeculationEngaged(
    proposedCount: Int, passthroughReason: String?
) -> Bool {
    proposedCount > 0 && passthroughReason == nil
}

/// C2, pooled: implied acceptance (`acceptedCount / proposedCount`, summed across every prompt)
/// must land within `+/- toleranceAbsolute` of `expected`. `nil` (never fabricated as failing OR
/// passing) when nothing was proposed at all.
func inCheckpointSampledMTPThroughputImpliedAcceptance(
    proposedCount: Int, acceptedCount: Int
) -> Double? {
    guard proposedCount > 0 else { return nil }
    return Double(acceptedCount) / Double(proposedCount)
}

func inCheckpointSampledMTPThroughputImpliedAcceptanceControlPasses(
    observed: Double,
    expected: Double = inCheckpointSampledMTPThroughputExpectedImpliedAcceptance,
    toleranceAbsolute: Double = inCheckpointSampledMTPThroughputImpliedAcceptanceToleranceAbsolute
) -> Bool {
    abs(observed - expected) <= toleranceAbsolute
}

/// C3, pooled: scalar-arm mean tok/s must be within `+/- toleranceFraction` (relative) of
/// `expected`.
func inCheckpointSampledMTPThroughputScalarControlPasses(
    observedMeanTokensPerSecond: Double,
    expected: Double = inCheckpointSampledMTPThroughputExpectedScalarTokensPerSecond,
    toleranceFraction: Double = inCheckpointSampledMTPThroughputScalarTokensPerSecondToleranceFraction
) -> Bool {
    let lowerBound = expected * (1 - toleranceFraction)
    let upperBound = expected * (1 + toleranceFraction)
    return observedMeanTokensPerSecond >= lowerBound && observedMeanTokensPerSecond <= upperBound
}

/// C4: sample size -- `promptCount >= required` AND `maxTokens >= required`.
func inCheckpointSampledMTPThroughputSampleSizeControlPasses(
    promptCount: Int, maxTokens: Int,
    requiredPromptCount: Int = inCheckpointSampledMTPThroughputRequiredPromptCount,
    requiredMaxTokens: Int = inCheckpointSampledMTPThroughputRequiredMaxTokens
) -> Bool {
    promptCount >= requiredPromptCount && maxTokens >= requiredMaxTokens
}

struct InCheckpointSampledMTPThroughputDoubleSummary: Codable, Equatable, Sendable {
    let mean: Double
    let min: Double
    let max: Double
    let count: Int
}

func inCheckpointSampledMTPThroughputSummarize(
    _ values: [Double]
) -> InCheckpointSampledMTPThroughputDoubleSummary? {
    guard !values.isEmpty else { return nil }
    return InCheckpointSampledMTPThroughputDoubleSummary(
        mean: values.reduce(0, +) / Double(values.count),
        min: values.min()!,
        max: values.max()!,
        count: values.count)
}

/// Even prompt index runs speculative-then-scalar; odd runs scalar-then-speculative
/// (predeclaration, "Method": "Alternating arm order across prompts ... so any monotonic thermal
/// or memory drift across the run cannot be absorbed entirely into one arm").
func inCheckpointSampledMTPThroughputSpeculativeArmRunsFirst(promptIndex: Int) -> Bool {
    promptIndex % 2 == 0
}

// MARK: - Output schema

struct InCheckpointSampledMTPThroughputPromptReport: Codable, Equatable, Sendable {
    let promptIndex: Int
    let promptPreview: String
    let promptTokenCount: Int
    let speculativeArmRanFirst: Bool
    let speculativeTokensPerSecond: Double
    let scalarTokensPerSecond: Double
    let ratio: Double
    let proposedCount: Int
    let acceptedCount: Int
    let passthroughReason: String?
}

struct InCheckpointSampledMTPThroughputPooledReport: Codable, Equatable, Sendable {
    let medianRatio: Double
    let minRatio: Double
    let maxRatio: Double
    let proposedCount: Int
    let acceptedCount: Int
    let impliedAcceptance: Double?
    let scalarTokensPerSecond: InCheckpointSampledMTPThroughputDoubleSummary
    let speculativeTokensPerSecond: InCheckpointSampledMTPThroughputDoubleSummary
}

struct InCheckpointSampledMTPThroughputControlReport: Codable, Equatable, Sendable {
    let impliedAcceptanceObserved: Double?
    let impliedAcceptanceExpected: Double
    let impliedAcceptanceToleranceAbsolute: Double
    let impliedAcceptancePassed: Bool
    let scalarTokensPerSecondObserved: Double
    let scalarTokensPerSecondExpected: Double
    let scalarTokensPerSecondToleranceFraction: Double
    let scalarTokensPerSecondPassed: Bool
    let promptCount: Int
    let requiredPromptCount: Int
    let maxTokens: Int
    let requiredMaxTokens: Int
    let sampleSizePassed: Bool
}

struct InCheckpointSampledMTPThroughputSamplingReport: Codable, Equatable, Sendable {
    let temperature: Double
    let topP: Double
    let topK: Int
    let minP: Double
    let repetitionPenaltyIsNil: Bool
    let presencePenaltyIsNil: Bool
    let frequencyPenaltyIsNil: Bool
}

struct InCheckpointSampledMTPThroughputReport: Codable, Sendable {
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
    let sampling: InCheckpointSampledMTPThroughputSamplingReport
    let prompts: [InCheckpointSampledMTPThroughputPromptReport]
    let pooled: InCheckpointSampledMTPThroughputPooledReport
    let controls: InCheckpointSampledMTPThroughputControlReport
    /// Informational only, per the predeclaration's own framing of the band line -- never
    /// enforced by this CLI's exit code.
    let informationalBand: String
}

// MARK: - Entry point

func runInCheckpointSampledMTPThroughput(arguments: [String]) async throws {
    #if DEBUG
    throw InCheckpointSampledMTPThroughputCLIError.releaseBuildRequired
    #else
    let parsed = try parseInCheckpointSampledMTPThroughputArguments(arguments)

    let outputURL = URL(fileURLWithPath: parsed.outputJSONPath)
    guard !FileManager.default.fileExists(atPath: outputURL.path) else {
        throw InCheckpointSampledMTPThroughputCLIError.outputJSONMustBeNew
    }

    let promptsText: String
    do {
        promptsText = try String(
            contentsOf: URL(fileURLWithPath: parsed.promptsFilePath), encoding: .utf8)
    } catch {
        throw InCheckpointSampledMTPThroughputCLIError.promptsFileUnavailable(parsed.promptsFilePath)
    }
    let prompts = promptsText.split(separator: "\n", omittingEmptySubsequences: false)
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { !$0.isEmpty }
    guard !prompts.isEmpty else {
        throw InCheckpointSampledMTPThroughputCLIError.promptsFileEmpty
    }

    // C4, checked before any model load or measurement is spent: refusing an underpowered request
    // up front is strictly better than refusing only after a full run.
    guard
        inCheckpointSampledMTPThroughputSampleSizeControlPasses(
            promptCount: prompts.count, maxTokens: parsed.maxTokens)
    else {
        throw InCheckpointSampledMTPThroughputCLIError.sampleSizeBelowPredeclaredFloor(
            promptCount: prompts.count,
            requiredPromptCount: inCheckpointSampledMTPThroughputRequiredPromptCount,
            maxTokens: parsed.maxTokens,
            requiredMaxTokens: inCheckpointSampledMTPThroughputRequiredMaxTokens)
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

    // `temperature` and `minP` are PINNED, not a choice -- identical rationale to the acceptance
    // CLI: `supports()` (`SampledMTPBlockRuntimeBridge.swift`'s `sharedSampledMTPSupportsPredicate`)
    // requires exactly `temperature == 1` and `minP == 0`, and there is no flag for either. `topP`/
    // `topK` ARE settable, via `--top-p`/`--top-k`, and default to the untruncated identity (`1`/
    // `0`); `parseInCheckpointSampledMTPThroughputArguments` already rejects any value `supports()`
    // would refuse.
    let parameters = GenerateParameters(
        maxTokens: parsed.maxTokens,
        temperature: 1,
        topP: Float(parsed.topP),
        topK: parsed.topK,
        minP: 0,
        seed: parsed.seed)
    // Built from the SAME `parameters` value handed to `MTPSpeculativeTokenIterator` inside
    // `runSpeculativeArm` below (not a second, independently invented truncation) -- this is what
    // makes it structurally impossible for the provider's stored truncation and this run's own
    // request truncation to disagree.
    let truncation = SampledMTPSamplingTruncation(parameters: parameters)
    let samplingReport = InCheckpointSampledMTPThroughputSamplingReport(
        temperature: Double(parameters.temperature),
        topP: Double(parameters.topP),
        topK: parameters.topK,
        minP: Double(parameters.minP),
        repetitionPenaltyIsNil: parameters.repetitionPenalty == nil,
        presencePenaltyIsNil: parameters.presencePenalty == nil,
        frequencyPenaltyIsNil: parameters.frequencyPenalty == nil)

    print(
        "qwen4exp-sampled-mtp-throughput: build=release provider=\(parsed.provider.rawValue) "
            + "seed=\(parsed.seed) sampling: temperature=\(samplingReport.temperature) "
            + "topP=\(samplingReport.topP) topK=\(samplingReport.topK) minP=\(samplingReport.minP) "
            + "repetitionPenalty=nil presencePenalty=nil frequencyPenalty=nil "
            + "(temperature=1 and minP=0 are PINNED -- supports() requires exactly those; topP/topK "
            + "are settable via --top-p/--top-k, default to the untruncated identity (1/0), and the "
            + "ACTUAL values this run measured are topP=\(samplingReport.topP) "
            + "topK=\(samplingReport.topK), printed above)")

    var promptReports: [InCheckpointSampledMTPThroughputPromptReport] = []
    var pooledProposedCount = 0
    var pooledAcceptedCount = 0
    var scalarRates: [Double] = []
    var speculativeRates: [Double] = []
    var ratios: [Double] = []

    for (promptIndex, prompt) in prompts.enumerated() {
        let promptTokens = context.tokenizer.encode(text: prompt)
        let speculativeFirst = inCheckpointSampledMTPThroughputSpeculativeArmRunsFirst(
            promptIndex: promptIndex)

        // --- speculative arm: fresh cache, telemetry OFF, PLAIN (unwrapped) provider ---
        func runSpeculativeArm() throws -> (
            tokensPerSecond: Double, proposedCount: Int, acceptedCount: Int,
            passthroughReason: String?
        ) {
            let mtpCache = context.model.newCache(parameters: parameters)
            var mtpIterator = try MTPSpeculativeTokenIterator(
                input: LMInput(tokens: MLXArray(promptTokens)),
                mainModel: context.model,
                drafter: loadedDrafter.drafter,
                mainCache: mtpCache,
                parameters: parameters,
                blockSize: inCheckpointSampledMTPBlockSize,
                collectPhaseTelemetry: false,
                sampledBlockDecisionProvider: inCheckpointSampledMTPMakeProvider(
                    parsed.provider, seed: parsed.seed, truncation: truncation))
            // Fail fast on prompt[0]'s speculative arm, BEFORE spending this prompt's full
            // `--max-tokens` decode on both arms: see the doc comment on `providerWiringNotEngaged`
            // for why `passthroughReason` at this exact point (immediately after construction,
            // before `nextThrowing()` is ever called) genuinely discriminates a provider/truncation
            // wiring defect from a healthy run.
            if promptIndex == 0, let constructionPassthroughReason = mtpIterator.passthroughReason {
                throw InCheckpointSampledMTPThroughputCLIError.providerWiringNotEngaged(
                    passthroughReason: constructionPassthroughReason)
            }
            guard try mtpIterator.nextThrowing() != nil else {
                throw InCheckpointSampledMTPThroughputCLIError.armEndedEarly(
                    arm: "speculative", promptIndex: promptIndex, emittedTokenCount: 0,
                    requestedMaxTokens: parsed.maxTokens)
            }
            let t0 = ProcessInfo.processInfo.systemUptime
            var emitted = 1
            while emitted < parsed.maxTokens, try mtpIterator.nextThrowing() != nil {
                emitted += 1
            }
            let tEnd = ProcessInfo.processInfo.systemUptime
            mtpIterator.finalizeGeneration()
            guard emitted == parsed.maxTokens else {
                throw InCheckpointSampledMTPThroughputCLIError.armEndedEarly(
                    arm: "speculative", promptIndex: promptIndex, emittedTokenCount: emitted,
                    requestedMaxTokens: parsed.maxTokens)
            }
            guard
                let tokensPerSecond = inCheckpointSampledMTPThroughputTokensPerSecond(
                    emittedTokenCount: emitted, elapsedSeconds: tEnd - t0)
            else {
                throw InCheckpointSampledMTPThroughputCLIError.armEndedEarly(
                    arm: "speculative", promptIndex: promptIndex, emittedTokenCount: emitted,
                    requestedMaxTokens: parsed.maxTokens)
            }
            return (
                tokensPerSecond, mtpIterator.proposedCount, mtpIterator.acceptedCount,
                mtpIterator.passthroughReason
            )
        }

        // --- scalar arm: fresh cache, plain TokenIterator ---
        func runScalarArm() throws -> Double {
            let scalarCache = context.model.newCache(parameters: parameters)
            var scalarIterator = try TokenIterator(
                input: LMInput(tokens: MLXArray(promptTokens)),
                model: context.model,
                cache: scalarCache,
                parameters: parameters)
            guard try scalarIterator.nextThrowing() != nil else {
                throw InCheckpointSampledMTPThroughputCLIError.armEndedEarly(
                    arm: "scalar", promptIndex: promptIndex, emittedTokenCount: 0,
                    requestedMaxTokens: parsed.maxTokens)
            }
            let t0 = ProcessInfo.processInfo.systemUptime
            var emitted = 1
            while emitted < parsed.maxTokens, try scalarIterator.nextThrowing() != nil {
                emitted += 1
            }
            let tEnd = ProcessInfo.processInfo.systemUptime
            guard emitted == parsed.maxTokens else {
                throw InCheckpointSampledMTPThroughputCLIError.armEndedEarly(
                    arm: "scalar", promptIndex: promptIndex, emittedTokenCount: emitted,
                    requestedMaxTokens: parsed.maxTokens)
            }
            guard
                let tokensPerSecond = inCheckpointSampledMTPThroughputTokensPerSecond(
                    emittedTokenCount: emitted, elapsedSeconds: tEnd - t0)
            else {
                throw InCheckpointSampledMTPThroughputCLIError.armEndedEarly(
                    arm: "scalar", promptIndex: promptIndex, emittedTokenCount: emitted,
                    requestedMaxTokens: parsed.maxTokens)
            }
            return tokensPerSecond
        }

        let speculativeResult: (
            tokensPerSecond: Double, proposedCount: Int, acceptedCount: Int,
            passthroughReason: String?
        )
        let scalarTokensPerSecond: Double
        if speculativeFirst {
            speculativeResult = try runSpeculativeArm()
            scalarTokensPerSecond = try runScalarArm()
        } else {
            scalarTokensPerSecond = try runScalarArm()
            speculativeResult = try runSpeculativeArm()
        }

        // Fix (mirrors the acceptance CLI's own Fix 4): print the observed counts BEFORE C1 can
        // throw, so a failing prompt still emits something to diagnose from.
        print(
            "prompt[\(promptIndex)]: speculativeFirst=\(speculativeFirst) "
                + "speculativeTokS=\(String(format: "%.2f", speculativeResult.tokensPerSecond)) "
                + "scalarTokS=\(String(format: "%.2f", scalarTokensPerSecond)) "
                + "proposedCount=\(speculativeResult.proposedCount) "
                + "acceptedCount=\(speculativeResult.acceptedCount) "
                + "passthroughReason=\(speculativeResult.passthroughReason ?? "nil")")

        // C1
        guard
            inCheckpointSampledMTPThroughputSpeculationEngaged(
                proposedCount: speculativeResult.proposedCount,
                passthroughReason: speculativeResult.passthroughReason)
        else {
            throw InCheckpointSampledMTPThroughputCLIError.speculationNotEngaged(
                promptIndex: promptIndex, proposedCount: speculativeResult.proposedCount,
                passthroughReason: speculativeResult.passthroughReason)
        }

        let ratio = speculativeResult.tokensPerSecond / scalarTokensPerSecond
        print(
            "prompt[\(promptIndex)]: ratio=\(String(format: "%.4f", ratio))")

        promptReports.append(
            InCheckpointSampledMTPThroughputPromptReport(
                promptIndex: promptIndex,
                promptPreview: String(prompt.prefix(80)),
                promptTokenCount: promptTokens.count,
                speculativeArmRanFirst: speculativeFirst,
                speculativeTokensPerSecond: speculativeResult.tokensPerSecond,
                scalarTokensPerSecond: scalarTokensPerSecond,
                ratio: ratio,
                proposedCount: speculativeResult.proposedCount,
                acceptedCount: speculativeResult.acceptedCount,
                passthroughReason: speculativeResult.passthroughReason))

        pooledProposedCount += speculativeResult.proposedCount
        pooledAcceptedCount += speculativeResult.acceptedCount
        scalarRates.append(scalarTokensPerSecond)
        speculativeRates.append(speculativeResult.tokensPerSecond)
        ratios.append(ratio)
    }

    guard let medianRatio = inCheckpointSampledMTPThroughputMedianRatio(ratios),
        let minRatio = ratios.min(), let maxRatio = ratios.max(),
        let scalarSummary = inCheckpointSampledMTPThroughputSummarize(scalarRates),
        let speculativeSummary = inCheckpointSampledMTPThroughputSummarize(speculativeRates)
    else {
        // Unreachable given the C4 floor already enforced (>= 8 prompts) plus every prompt above
        // either populating these arrays or throwing first -- kept as a named, loud failure
        // rather than a force-unwrap so a future refactor that breaks this invariant fails
        // clearly instead of crashing.
        throw InCheckpointSampledMTPThroughputCLIError.sampleSizeBelowPredeclaredFloor(
            promptCount: prompts.count,
            requiredPromptCount: inCheckpointSampledMTPThroughputRequiredPromptCount,
            maxTokens: parsed.maxTokens,
            requiredMaxTokens: inCheckpointSampledMTPThroughputRequiredMaxTokens)
    }

    let impliedAcceptance = inCheckpointSampledMTPThroughputImpliedAcceptance(
        proposedCount: pooledProposedCount, acceptedCount: pooledAcceptedCount)

    print("--- pooled across \(promptReports.count) prompt(s) ---")
    print(
        "ratio: median=\(String(format: "%.4f", medianRatio)) "
            + "min=\(String(format: "%.4f", minRatio)) max=\(String(format: "%.4f", maxRatio))")
    print(
        "implied acceptance = \(pooledAcceptedCount)/\(pooledProposedCount) = "
            + (impliedAcceptance.map { String(format: "%.4f", $0) } ?? "n/a"))
    print(
        "scalar tok/s: mean=\(String(format: "%.2f", scalarSummary.mean)) "
            + "range=[\(String(format: "%.2f", scalarSummary.min)),"
            + "\(String(format: "%.2f", scalarSummary.max))] n=\(scalarSummary.count)")
    print(
        "speculative tok/s: mean=\(String(format: "%.2f", speculativeSummary.mean)) "
            + "range=[\(String(format: "%.2f", speculativeSummary.min)),"
            + "\(String(format: "%.2f", speculativeSummary.max))] n=\(speculativeSummary.count)")

    // C2
    let impliedAcceptancePassed = impliedAcceptance.map {
        inCheckpointSampledMTPThroughputImpliedAcceptanceControlPasses(observed: $0)
    } ?? false
    print(
        "control C2 (implied acceptance vs "
            + "\(String(format: "%.4f", inCheckpointSampledMTPThroughputExpectedImpliedAcceptance)) "
            + "+/- \(inCheckpointSampledMTPThroughputImpliedAcceptanceToleranceAbsolute)): "
            + (impliedAcceptancePassed ? "PASS" : "FAIL"))
    guard let impliedAcceptance, impliedAcceptancePassed else {
        throw InCheckpointSampledMTPThroughputCLIError.impliedAcceptanceOutOfBand(
            observed: impliedAcceptance ?? .nan, proposedCount: pooledProposedCount,
            acceptedCount: pooledAcceptedCount,
            expected: inCheckpointSampledMTPThroughputExpectedImpliedAcceptance,
            toleranceAbsolute: inCheckpointSampledMTPThroughputImpliedAcceptanceToleranceAbsolute)
    }

    // C3
    let scalarControlPassed = inCheckpointSampledMTPThroughputScalarControlPasses(
        observedMeanTokensPerSecond: scalarSummary.mean)
    print(
        "control C3 (scalar tok/s vs "
            + "\(String(format: "%.2f", inCheckpointSampledMTPThroughputExpectedScalarTokensPerSecond)) "
            + "+/- \(String(format: "%.0f", inCheckpointSampledMTPThroughputScalarTokensPerSecondToleranceFraction * 100))%): "
            + (scalarControlPassed ? "PASS" : "FAIL"))
    guard scalarControlPassed else {
        throw InCheckpointSampledMTPThroughputCLIError.scalarThroughputOutOfBand(
            observed: scalarSummary.mean,
            expected: inCheckpointSampledMTPThroughputExpectedScalarTokensPerSecond,
            toleranceFraction: inCheckpointSampledMTPThroughputScalarTokensPerSecondToleranceFraction)
    }

    let band = inCheckpointSampledMTPThroughputBand(
        medianRatio: medianRatio, minPerPromptRatio: minRatio)
    print(
        "predeclared band (informational, not enforced by this CLI): \(band.rawValue) "
            + "(median=\(String(format: "%.4f", medianRatio)), min=\(String(format: "%.4f", minRatio)))")

    let controlReport = InCheckpointSampledMTPThroughputControlReport(
        impliedAcceptanceObserved: impliedAcceptance,
        impliedAcceptanceExpected: inCheckpointSampledMTPThroughputExpectedImpliedAcceptance,
        impliedAcceptanceToleranceAbsolute: inCheckpointSampledMTPThroughputImpliedAcceptanceToleranceAbsolute,
        impliedAcceptancePassed: impliedAcceptancePassed,
        scalarTokensPerSecondObserved: scalarSummary.mean,
        scalarTokensPerSecondExpected: inCheckpointSampledMTPThroughputExpectedScalarTokensPerSecond,
        scalarTokensPerSecondToleranceFraction: inCheckpointSampledMTPThroughputScalarTokensPerSecondToleranceFraction,
        scalarTokensPerSecondPassed: scalarControlPassed,
        promptCount: prompts.count,
        requiredPromptCount: inCheckpointSampledMTPThroughputRequiredPromptCount,
        maxTokens: parsed.maxTokens,
        requiredMaxTokens: inCheckpointSampledMTPThroughputRequiredMaxTokens,
        sampleSizePassed: inCheckpointSampledMTPThroughputSampleSizeControlPasses(
            promptCount: prompts.count, maxTokens: parsed.maxTokens))

    let pooledReport = InCheckpointSampledMTPThroughputPooledReport(
        medianRatio: medianRatio,
        minRatio: minRatio,
        maxRatio: maxRatio,
        proposedCount: pooledProposedCount,
        acceptedCount: pooledAcceptedCount,
        impliedAcceptance: impliedAcceptance,
        scalarTokensPerSecond: scalarSummary,
        speculativeTokensPerSecond: speculativeSummary)

    let report = InCheckpointSampledMTPThroughputReport(
        schemaVersion: 1,
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
        prompts: promptReports,
        pooled: pooledReport,
        controls: controlReport,
        informationalBand: band.rawValue)

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data: Data
    do {
        data = try encoder.encode(report)
    } catch {
        throw InCheckpointSampledMTPThroughputCLIError.outputJSONWriteFailed
    }
    let descriptor = outputURL.withUnsafeFileSystemRepresentation { path -> Int32 in
        guard let path else { return -1 }
        return open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0o600)
    }
    guard descriptor >= 0 else {
        throw InCheckpointSampledMTPThroughputCLIError.outputJSONMustBeNew
    }
    defer { _ = close(descriptor) }
    do {
        try FileHandle(fileDescriptor: descriptor, closeOnDealloc: false).write(contentsOf: data)
    } catch {
        throw InCheckpointSampledMTPThroughputCLIError.outputJSONWriteFailed
    }
    print("qwen4exp-sampled-mtp-throughput PASS output=\(parsed.outputJSONPath)")
    #endif
}
