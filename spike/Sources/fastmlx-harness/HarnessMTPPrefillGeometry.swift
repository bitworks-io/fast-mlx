import Foundation
import SpikeCore

/// Single source of truth for the fastmlx-harness's MTP prefill chunk size -- the value every
/// `MTPSpeculativeTokenIterator` construction in this target must set explicitly as
/// `GenerateParameters.prefillStepSize`.
///
/// Production's MTP serving route (`MTPSpeculativeDecoder.buildParameters()`,
/// `spike/Sources/SpikeCore/MTPSpeculativeDecoder.swift`) sets
/// `prefillStepSize = MLXDecoder.defaultPrefillChunkSize` (2048) before constructing its iterator.
/// Every harness CLI that also constructs `MTPSpeculativeTokenIterator` defaults to that SAME
/// value: leaving `prefillStepSize` unset instead silently inherits the vendored
/// `GenerateParameters` default (512, `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift`),
/// measuring a prefill geometry production never uses. That silent divergence -- not a hypothetical
/// one, it is what every sampled-MTP measurement taken before this type existed actually did -- is
/// exactly the defect this type exists to prevent. Routing every harness call site through here,
/// rather than each site inventing (or omitting) its own default, makes the two routes' geometry
/// impossible to drift apart by omission.
///
/// A deliberate off-production comparison (e.g. measuring chunk 512 vs 2048) is still expressible,
/// but only EXPLICITLY: set `FASTMLX_MTP_PREFILL_CHUNK` in the process environment. An unparseable
/// or non-positive override fails loudly (throws) rather than silently falling back to the
/// default -- a silent fallback on a bad override is the same class of defect this type exists to
/// prevent.
enum HarnessMTPPrefillGeometry {
    /// Environment variable name for the explicit off-production override. Read directly wherever
    /// needed (including in error messages) rather than re-derived, so renaming this constant can
    /// never leave a stale name behind.
    static let overrideEnvironmentVariable = "FASTMLX_MTP_PREFILL_CHUNK"

    /// Resolves the harness's effective MTP prefill chunk size: `MLXDecoder.defaultPrefillChunkSize`
    /// unless `overrideEnvironmentVariable` is set in `environment`, in which case that value is
    /// used verbatim after validating it parses to a positive `Int`.
    ///
    /// - Parameter environment: Injectable for tests; defaults to the real process environment.
    /// - Throws: `HarnessMTPPrefillGeometryError.invalidOverride` if the environment variable is
    ///   set but does not parse to a positive `Int`.
    static func prefillChunkSize(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> Int {
        guard let raw = environment[overrideEnvironmentVariable] else {
            return MLXDecoder.defaultPrefillChunkSize
        }
        guard let parsed = Int(raw), parsed > 0 else {
            throw HarnessMTPPrefillGeometryError.invalidOverride(
                variable: overrideEnvironmentVariable, value: raw)
        }
        return parsed
    }
}

/// Thrown by `HarnessMTPPrefillGeometry.prefillChunkSize` when
/// `HarnessMTPPrefillGeometry.overrideEnvironmentVariable` is set to a value that does not parse to
/// a positive `Int`. Deliberately a thrown error, not a `fatalError` -- so both the harness CLIs
/// (which surface it as an ordinary CLI failure) and this type's own tests can observe the specific
/// failure rather than a process abort.
enum HarnessMTPPrefillGeometryError: Error, CustomStringConvertible, Equatable {
    case invalidOverride(variable: String, value: String)

    var description: String {
        switch self {
        case .invalidOverride(let variable, let value):
            return
                "\(variable) must parse to a positive Int (harness MTP prefill chunk size); "
                + "got \"\(value)\""
        }
    }
}
