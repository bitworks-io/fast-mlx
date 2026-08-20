import HarnessCore
import ServingCore

/// MLX-free seam that converts a serving-boundary `ServingSamplingPolicy` into
/// the tested `SamplingPolicyV1` sampling contract.
///
/// This target exists so the bridge (and its tests) stay off the MLX toolchain:
/// `ServingCore` is deliberately zero-dependency and `HarnessCore` is MLX-free,
/// but neither depends on the other, so the conversion needs a dedicated home
/// that imports both. The MLX adapters consume this seam once live decode lands.
public enum ServingSamplingBridge {
    /// Convert a resolved serving policy to the contract policy the oracle consumes.
    ///
    /// `.greedy` maps to the greedy (argmax) contract policy. `.sampled` requires
    /// a resolved seed: a sampled policy without one fails closed, because a
    /// reproducible draw needs a server-assigned seed and that assignment is a
    /// runtime concern deferred with live decode. Contract-level rejections
    /// (e.g. an out-of-range temperature on a directly-constructed policy) surface
    /// as the contract's own `SamplingContractFailure`.
    public static func contractPolicy(
        for policy: ServingSamplingPolicy
    ) throws -> SamplingPolicyV1 {
        switch policy {
        case .greedy:
            return .greedy()
        case let .sampled(temperature, topP, _, _, seed):
            guard let seed else {
                throw ServingSamplingBridgeError.seedRequiredForSampledRoute
            }
            return try SamplingPolicyV1.sampled(
                temperature: temperature,
                topP: topP,
                seed: SamplingSeedV1.callerSupplied(seed))
        }
    }
}

/// Bridge-level rejection raised when a serving policy cannot be converted.
public enum ServingSamplingBridgeError: Error, Equatable, Sendable {
    /// A sampled policy reached the bridge without a resolved seed. Live sampling
    /// needs a server-assigned seed first; that wiring lands with live decode.
    case seedRequiredForSampledRoute
}
