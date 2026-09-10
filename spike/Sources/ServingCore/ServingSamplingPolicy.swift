import Foundation

/// Serving-boundary sampling decision, resolved from an OpenAI chat request.
///
/// This mirrors — but does not import — the tested `SamplingContractV1` in
/// HarnessCore, because ServingCore is deliberately dependency-free. The
/// adapter layer (which depends on both) bridges a resolved policy to the
/// contract oracle. Range/finiteness validation here matches the contract so
/// the serving boundary rejects the same values the oracle would.
///
/// `seed` is `nil` when the caller did not pin one: a sampled request without a
/// seed still resolves, but the server must assign a seed before the draw is
/// reproducible. Seed assignment is a runtime concern deferred with live decode.
public enum ServingSamplingPolicy: Equatable, Sendable {
    case greedy
    case sampled(temperature: Double, topP: Double, topK: Int?, minP: Double?, seed: Int64?)

    /// Resolve a policy from explicit request fields.
    ///
    /// Temperature `nil` or `0` is greedy (argmax); `topP`/`topK`/`minP`/`seed`
    /// are ignored on that branch. A positive temperature yields `.sampled`,
    /// defaulting `topP` to `1.0` when absent. Non-finite or out-of-range
    /// values throw.
    ///
    /// `defaults`, when non-`nil`, supplies values for sampling fields the
    /// request omitted — sourced from the model artifact rather than the
    /// client. A field the request DID supply always wins over `defaults`.
    /// `defaults` only ever fills an *absent* (`nil`) temperature: an
    /// explicit `temperature: 0` still resolves to `.greedy` regardless of
    /// `defaults`, preserving that value as the client's escape hatch to the
    /// greedy speculative-decoding path. Passing `defaults: nil` reproduces
    /// today's behavior exactly.
    public static func resolve(
        temperature: Double?,
        topP: Double?,
        topK: Int? = nil,
        minP: Double? = nil,
        seed: Int64?,
        defaults: ServingSamplingDefaults? = nil
    ) throws -> ServingSamplingPolicy {
        let temperature = temperature ?? defaults?.temperature
        guard let temperature, temperature != 0 else {
            return .greedy
        }

        guard temperature.isFinite else {
            throw ServingSamplingPolicyError.nonFiniteTemperature
        }
        guard temperature > 0, temperature <= 2 else {
            throw ServingSamplingPolicyError.temperatureOutOfRange(temperature)
        }

        let resolvedTopP = topP ?? defaults?.topP ?? 1.0
        guard resolvedTopP.isFinite else {
            throw ServingSamplingPolicyError.nonFiniteTopP
        }
        guard resolvedTopP > 0, resolvedTopP <= 1 else {
            throw ServingSamplingPolicyError.topPOutOfRange(resolvedTopP)
        }
        // Downstream (GenerateParameters/TopPSampler) computes its nucleus
        // threshold as `1 - topP` and only truncates when that value is
        // strictly less than 1; below roughly `topP < 3e-8`, `1 - topP`
        // rounds to exactly `1.0f` in float32 and every position in the
        // cumulative-mass test reads as "does not clear the threshold."
        //
        // POLICY, not mechanism: `applyTopPFilter`
        // (`spike/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift`)
        // now force-keeps the row's single most-probable token whenever that
        // happens (HF `TopPLogitsWarper`'s `min_tokens_to_keep=1` floor), so
        // this band is no longer a fail-open defect at the vendored-helper
        // level -- it yields a legitimate, correctly-computed single-token
        // nucleus. This guard refuses it anyway, for a different reason: a
        // forced single-token nucleus is operationally indistinguishable
        // from greedy decoding, while still being requested and billed as
        // sampled. Refusing it here, at the serving boundary, tells the
        // caller to ask for `temperature: 0` (the real greedy path) instead
        // of silently substituting one. This refusal is deliberately
        // width-independent and conservative -- it refuses slightly above
        // the float32 onset, not at the exact onset, because the exact
        // onset shifts with vocabulary width and a hardcoded "safe" constant
        // would not be well-defined across models -- and it keeps the
        // serving boundary's correctness independent of the vendored
        // helper's own float32 numerics: this guard would still be the
        // right policy call even if a future vendor sync changed
        // `applyTopPFilter`'s internals again.
        guard Float(1) - Float(resolvedTopP) < 1 else {
            throw ServingSamplingPolicyError.topPTooSmallToTruncate(resolvedTopP)
        }

        let resolvedTopK = topK ?? defaults?.topK
        if let resolvedTopK, resolvedTopK <= 0 {
            throw ServingSamplingPolicyError.topKOutOfRange(resolvedTopK)
        }

        let resolvedMinP = minP ?? defaults?.minP
        if let resolvedMinP {
            guard resolvedMinP.isFinite else {
                throw ServingSamplingPolicyError.nonFiniteMinP
            }
            guard resolvedMinP >= 0, resolvedMinP <= 1 else {
                throw ServingSamplingPolicyError.minPOutOfRange(resolvedMinP)
            }
        }

        return .sampled(
            temperature: temperature, topP: resolvedTopP, topK: resolvedTopK, minP: resolvedMinP, seed: seed)
    }

    /// Convenience resolution from a decoded request, carrying `top_p`, `top_k`,
    /// `min_p`, and `seed` through from the request.
    ///
    /// Equivalent to `resolve(from: request, defaults: nil)`.
    public static func resolve(
        from request: OpenAIChatCompletionRequest
    ) throws -> ServingSamplingPolicy {
        try resolve(from: request, defaults: nil)
    }

    /// Convenience resolution from a decoded request, additionally consulting
    /// artifact-sourced `defaults` for fields the request omitted. See
    /// `resolve(temperature:topP:topK:minP:seed:defaults:)` for the precedence
    /// rules between the request and `defaults`.
    public static func resolve(
        from request: OpenAIChatCompletionRequest,
        defaults: ServingSamplingDefaults?
    ) throws -> ServingSamplingPolicy {
        try resolve(
            temperature: request.temperature,
            topP: request.topP,
            topK: request.topK,
            minP: request.minP,
            seed: request.seed,
            defaults: defaults)
    }
}

/// Typed rejections raised while resolving a `ServingSamplingPolicy`, mirroring
/// the contract's invalid-parameter cases at the serving boundary.
public enum ServingSamplingPolicyError: Error, Equatable, Sendable {
    case nonFiniteTemperature
    case temperatureOutOfRange(Double)
    case nonFiniteTopP
    case topPOutOfRange(Double)
    case topPTooSmallToTruncate(Double)
    case topKOutOfRange(Int)
    case nonFiniteMinP
    case minPOutOfRange(Double)
}

/// Fallback sampling values applied by `ServingSamplingPolicy.resolve` when a
/// request omits the corresponding field. Sourced from the model artifact
/// (deferred with live decode), not from the client. A field the request
/// explicitly supplies always overrides the matching default; `temperature`
/// only fills an *absent* value and never overrides an explicit `0`, which
/// stays the client's escape hatch to the greedy speculative-decoding path.
///
/// `topP`, `topK`, and `minP` are optional so a loader can represent "the
/// artifact did not constrain this field" (`nil`) distinctly from an
/// explicit, validated value -- notably HF's `top_k: 0` convention ("top-k
/// disabled"), which must map to `nil` here rather than to the literal `0`
/// that `resolve` would refuse as out of range.
public struct ServingSamplingDefaults: Equatable, Sendable {
    public var temperature: Double
    public var topP: Double?
    public var topK: Int?
    public var minP: Double?

    public init(temperature: Double, topP: Double? = nil, topK: Int? = nil, minP: Double? = nil) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
    }
}
