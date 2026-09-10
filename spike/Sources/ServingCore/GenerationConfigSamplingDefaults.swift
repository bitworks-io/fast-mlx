import Foundation

/// Decodes the sampling-relevant subset of a HuggingFace `generation_config.json`
/// model artifact into `ServingSamplingDefaults`, the artifact-sourced fallback
/// values `ServingSamplingPolicy.resolve` applies to a request that omits a
/// sampling field.
///
/// ServingCore is deliberately dependency-free. This decoder is local to
/// ServingCore and intentionally does NOT reuse the vendored
/// `GenerationConfigFile` (`spike/Vendor/mlx-swift-lm/...`), which decodes
/// only eos/stop-token fields for a different purpose.
///
/// Every failure mode here FAILS CLOSED -- none falls back to greedy --
/// because a silent greedy fallback would reproduce exactly the defect this
/// type exists to fix: a client that omits `temperature` today silently gets
/// argmax instead of the artifact's intended sampling behavior.
public enum GenerationConfigSamplingDefaults {
    /// Load and validate sampling defaults from a `generation_config.json` at
    /// `url`.
    ///
    /// - `do_sample` absent or `false` throws `.samplingNotEnabled`: HF's own
    ///   default for `do_sample` is `false`, so absence is not consent to
    ///   sample.
    /// - `temperature` absent, or present and `== 0`, throws
    ///   `.noUsableTemperature`: a defaults object built from either would
    ///   resolve every param-less request to `.greedy` anyway, i.e. it would
    ///   be a no-op default not worth building.
    /// - `top_k == 0` maps to `nil` (HF's "top-k disabled" convention), not
    ///   to the literal `0`, which `ServingSamplingPolicy.resolve` refuses as
    ///   out of range. `top_k` absent also maps to `nil`.
    /// - `top_p` and `min_p` absent map to `nil`, letting `resolve`'s own
    ///   fallbacks (`topP ?? defaults?.topP ?? 1.0`, and no forced `minP`)
    ///   apply.
    /// - Before returning, the candidate is round-tripped through
    ///   `ServingSamplingPolicy.resolve` (with no request-side overrides), so
    ///   an out-of-range artifact value throws `.invalidValue` here, at load
    ///   time, instead of surfacing on the first request that omits the
    ///   field. This also guarantees the range rules applied to artifact
    ///   values can never drift from the resolver's own rules.
    public static func load(contentsOf url: URL) throws -> ServingSamplingDefaults {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw GenerationConfigSamplingDefaultsError.unreadable
        }

        let decoded: Fields
        do {
            decoded = try JSONDecoder().decode(Fields.self, from: data)
        } catch {
            throw GenerationConfigSamplingDefaultsError.unparseable
        }

        guard decoded.doSample == true else {
            throw GenerationConfigSamplingDefaultsError.samplingNotEnabled
        }

        guard let temperature = decoded.temperature, temperature != 0 else {
            throw GenerationConfigSamplingDefaultsError.noUsableTemperature
        }

        // HF: top_k == 0 means "disabled", not the literal 0 that `resolve`
        // would refuse as out of range.
        let topK = decoded.topK.flatMap { $0 == 0 ? nil : $0 }

        let candidate = ServingSamplingDefaults(
            temperature: temperature,
            topP: decoded.topP,
            topK: topK,
            minP: decoded.minP)

        do {
            // No request-side overrides: this proves `candidate` alone
            // resolves to a valid `.sampled` policy, the same validation a
            // live request carrying these exact values would undergo.
            _ = try ServingSamplingPolicy.resolve(
                temperature: nil, topP: nil, topK: nil, minP: nil, seed: nil, defaults: candidate)
        } catch let error as ServingSamplingPolicyError {
            throw GenerationConfigSamplingDefaultsError.invalidValue(error)
        }

        return candidate
    }

    private struct Fields: Decodable {
        let doSample: Bool?
        let temperature: Double?
        let topP: Double?
        let topK: Int?
        let minP: Double?

        enum CodingKeys: String, CodingKey {
            case doSample = "do_sample"
            case temperature
            case topP = "top_p"
            case topK = "top_k"
            case minP = "min_p"
        }
    }
}

/// Typed failures raised while loading `ServingSamplingDefaults` from a
/// `generation_config.json` artifact. Every case is a refusal: there is no
/// silent-greedy fallback path here.
public enum GenerationConfigSamplingDefaultsError: Error, Equatable, Sendable {
    case unreadable
    case unparseable
    case samplingNotEnabled
    case noUsableTemperature
    case invalidValue(ServingSamplingPolicyError)
}
