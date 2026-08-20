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
    public static func resolve(
        temperature: Double?,
        topP: Double?,
        topK: Int? = nil,
        minP: Double? = nil,
        seed: Int64?
    ) throws -> ServingSamplingPolicy {
        guard let temperature, temperature != 0 else {
            return .greedy
        }

        guard temperature.isFinite else {
            throw ServingSamplingPolicyError.nonFiniteTemperature
        }
        guard temperature > 0, temperature <= 2 else {
            throw ServingSamplingPolicyError.temperatureOutOfRange(temperature)
        }

        let resolvedTopP = topP ?? 1.0
        guard resolvedTopP.isFinite else {
            throw ServingSamplingPolicyError.nonFiniteTopP
        }
        guard resolvedTopP >= 0, resolvedTopP <= 1 else {
            throw ServingSamplingPolicyError.topPOutOfRange(resolvedTopP)
        }

        if let topK, topK <= 0 {
            throw ServingSamplingPolicyError.topKOutOfRange(topK)
        }

        if let minP {
            guard minP.isFinite else {
                throw ServingSamplingPolicyError.nonFiniteMinP
            }
            guard minP >= 0, minP <= 1 else {
                throw ServingSamplingPolicyError.minPOutOfRange(minP)
            }
        }

        return .sampled(temperature: temperature, topP: resolvedTopP, topK: topK, minP: minP, seed: seed)
    }

    /// Convenience resolution from a decoded request, carrying `top_p`, `top_k`,
    /// `min_p`, and `seed` through from the request.
    public static func resolve(
        from request: OpenAIChatCompletionRequest
    ) throws -> ServingSamplingPolicy {
        try resolve(
            temperature: request.temperature,
            topP: request.topP,
            topK: request.topK,
            minP: request.minP,
            seed: request.seed)
    }
}

/// Typed rejections raised while resolving a `ServingSamplingPolicy`, mirroring
/// the contract's invalid-parameter cases at the serving boundary.
public enum ServingSamplingPolicyError: Error, Equatable, Sendable {
    case nonFiniteTemperature
    case temperatureOutOfRange(Double)
    case nonFiniteTopP
    case topPOutOfRange(Double)
    case topKOutOfRange(Int)
    case nonFiniteMinP
    case minPOutOfRange(Double)
}
