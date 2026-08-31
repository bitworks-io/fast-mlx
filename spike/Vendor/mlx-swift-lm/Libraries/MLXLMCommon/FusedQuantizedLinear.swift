// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Default-off opt-in switch for the Qwen 3.5/3.6 four-projection GDN fusion.
package let qwen35FourGDNEnabled: Bool = {
    let raw = ProcessInfo.processInfo.environment["MLX_QWEN_FOUR_GDN"]?
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .lowercased()
    return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
}()

/// Default-off opt-in switch for Qwen3.8 Flash-Next four-projection GDN fusion.
package let qwen4ExpFourGDNEnabled: Bool = {
    let raw = ProcessInfo.processInfo.environment["MLX_QWEN4_EXP_FOUR_GDN"]?
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .lowercased()
    return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
}()

/// A model whose forward pass reads arrays that are NOT discoverable as module
/// parameters (for example fused projections held in a
/// ``FusedQuantizedLinearProjectionCache``, which module reflection cannot see).
///
/// Compiled serving steps pass module parameters as compile state so traced
/// graphs reference them as tracer inputs instead of captured constants; models
/// with out-of-tree forward arrays must surface them here so those arrays get
/// the same treatment. Captured constants are retained by the compiled trace,
/// and MLX traced graphs can leak permanently on teardown (orphaned sibling
/// reference cycles), pinning every captured constant's buffer.
public protocol AuxiliaryCompiledStateProviding {
    /// Arrays read by the forward pass that module reflection cannot surface.
    /// Order must be deterministic across calls.
    var auxiliaryCompiledState: [MLXArray] { get }
}

extension FusedQuantizedLinearProjectionCache {
    /// The prepared fused projection's arrays, for compiled-step state.
    package var auxiliaryCompiledState: [MLXArray] {
        fused?.innerState() ?? []
    }
}

/// A fused quantized projection and checkpoint-shaped views into its storage.
///
/// The views let a model keep its public/checkpoint module topology without
/// retaining a second physical copy of the quantized weights.
package struct FusedQuantizedLinearProjection {
    package let fused: QuantizedLinear
    package let sourceViews: [QuantizedLinear]
}

package enum FusedQuantizedLinearPrepareResult {
    case skipped
    case alreadyPrepared
    case prepared

    package var hasPreparedProjection: Bool {
        self != .skipped
    }
}

/// A fused projection could not replace its source modules atomically.
///
/// `rollbackError` is non-nil only when restoring the original source modules
/// also failed.
package struct FusedQuantizedLinearPreparationError: Error, CustomStringConvertible {
    package let installationError: any Error
    package let rollbackError: (any Error)?

    package init(installationError: any Error, rollbackError: (any Error)?) {
        self.installationError = installationError
        self.rollbackError = rollbackError
    }

    package var description: String {
        if let rollbackError {
            return "unable to install fused projection views (\(installationError)); "
                + "restoring the original projections also failed (\(rollbackError))"
        }
        return "unable to install fused projection views (\(installationError)); "
            + "the original projections were restored"
    }
}

/// Lifecycle state for a physical projection derived from several registered
/// quantized linears.
///
/// Mutating operations must run while the caller owns exclusive model access.
/// Inference only reads ``fused`` after preparation has completed.
package final class FusedQuantizedLinearProjectionCache {
    private enum State {
        case unprepared
        case preparing
        case ready
        case ineligible
    }

    private var state = State.unprepared
    package private(set) var fused: QuantizedLinear?

    package init() {}

    package var isPrepared: Bool {
        state == .ready && fused != nil
    }

    package func invalidate() {
        guard state != .preparing else { return }
        fused = nil
        state = .unprepared
    }

    @discardableResult
    package func prepare(
        enabled: Bool,
        linears: [Linear],
        expectedOutputDimensions: [Int]? = nil,
        installSourceModules: ([Linear]) throws -> Void
    ) throws -> Bool {
        try prepareResult(
            enabled: enabled,
            linears: linears,
            expectedOutputDimensions: expectedOutputDimensions,
            installSourceModules: installSourceModules
        ).hasPreparedProjection
    }

    @discardableResult
    package func prepareResult(
        enabled: Bool,
        linears: [Linear],
        expectedOutputDimensions: [Int]? = nil,
        installSourceModules: ([Linear]) throws -> Void
    ) throws -> FusedQuantizedLinearPrepareResult {
        guard enabled else { return .skipped }

        switch state {
        case .ready:
            return fused != nil ? .alreadyPrepared : .skipped
        case .preparing, .ineligible:
            return .skipped
        case .unprepared:
            break
        }

        state = .preparing
        guard let projection = fuseQuantizedLinearProjections(
            linears, expectedOutputDimensions: expectedOutputDimensions),
            projection.sourceViews.count == linears.count
        else {
            state = .ineligible
            return .skipped
        }

        do {
            try installSourceModules(projection.sourceViews)
        } catch let installationError {
            let rollbackError: (any Error)?
            do {
                try installSourceModules(linears)
                rollbackError = nil
            } catch let error {
                rollbackError = error
            }
            fused = nil
            state = .ineligible
            throw FusedQuantizedLinearPreparationError(
                installationError: installationError,
                rollbackError: rollbackError)
        }

        fused = projection.fused
        state = .ready
        return .prepared
    }
}

/// Coalesce compatible stock quantized linears along their output dimension.
///
/// Incompatible inputs return `nil` so callers keep their original projections.
package func fuseQuantizedLinearProjections(
    _ linears: [Linear],
    expectedOutputDimensions: [Int]? = nil
) -> FusedQuantizedLinearProjection? {
    guard linears.count > 1 else { return nil }

    if let expectedOutputDimensions {
        guard expectedOutputDimensions.count == linears.count,
            zip(linears, expectedOutputDimensions).allSatisfy({ linear, expected in
                linear.shape.0 == expected
            })
        else {
            return nil
        }
    }

    let projections = linears.compactMap { $0 as? QuantizedLinear }
    guard projections.count == linears.count,
        zip(linears, projections).allSatisfy({ linear, projection in
            ObjectIdentifier(type(of: linear)) == ObjectIdentifier(QuantizedLinear.self)
                && linear === projection
        }),
        let first = projections.first,
        first.bias == nil,
        first.weight.ndim == 2,
        first.weight.dim(0) == first.shape.0,
        first.scales.ndim == 2,
        first.scales.dim(0) == first.shape.0,
        first.biases == nil || first.biases?.shape == first.scales.shape
    else {
        return nil
    }

    let hasQuantizationBiases = first.biases != nil
    guard
        projections.allSatisfy({ projection in
            projection.bias == nil
                && projection.bits == first.bits
                && projection.groupSize == first.groupSize
                && projection.mode == first.mode
                && projection.shape.1 == first.shape.1
                && projection.weight.ndim == 2
                && projection.weight.dim(0) == projection.shape.0
                && projection.weight.dim(1) == first.weight.dim(1)
                && projection.weight.dtype == first.weight.dtype
                && projection.scales.ndim == 2
                && projection.scales.dim(0) == projection.shape.0
                && projection.scales.dim(1) == first.scales.dim(1)
                && projection.scales.dtype == first.scales.dtype
                && (projection.biases != nil) == hasQuantizationBiases
                && (projection.biases == nil || projection.biases?.shape == projection.scales.shape)
                && projection.biases?.dtype == first.biases?.dtype
        })
    else {
        return nil
    }

    let fusedWeight = concatenated(projections.map(\.weight), axis: 0)
    let fusedScales = concatenated(projections.map(\.scales), axis: 0)
    let fusedBiases =
        hasQuantizationBiases
        ? concatenated(projections.compactMap(\.biases), axis: 0)
        : nil

    eval(fusedWeight, fusedScales)
    if let fusedBiases {
        eval(fusedBiases)
    }

    let fused = QuantizedLinear(
        weight: fusedWeight,
        bias: nil,
        scales: fusedScales,
        biases: fusedBiases,
        groupSize: first.groupSize,
        bits: first.bits,
        mode: first.mode)
    fused.freeze()

    var start = 0
    let sourceViews = projections.map { projection in
        let end = start + projection.shape.0
        defer { start = end }

        let rows = start ..< end
        let view = QuantizedLinear(
            weight: fusedWeight[rows],
            bias: nil,
            scales: fusedScales[rows],
            biases: fusedBiases.map { $0[rows] },
            groupSize: first.groupSize,
            bits: first.bits,
            mode: first.mode)
        view.freeze()
        return view
    }

    return FusedQuantizedLinearProjection(fused: fused, sourceViews: sourceViews)
}
