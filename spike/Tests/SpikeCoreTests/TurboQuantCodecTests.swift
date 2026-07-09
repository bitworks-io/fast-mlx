import MLX
import MLXRandom
import XCTest

@testable import SpikeCore

/// Mathematical-property tests for the TurboQuant quantizer core (arXiv:2504.19874).
/// These prove the estimator's math on synthetic tensors at real head_dim before any
/// KV-cache integration is built on top of it (plan Phase 1B; Spike A gate lives here).
final class TurboQuantCodecTests: XCTestCase {

    // MARK: Task 2 — global params (Π, S, scaled codebook)

    func testParamsRotationIsOrthogonal() {
        let p = TurboQuantParams(headDim: 128, baseBits: 3, seed: 0)
        let ident = p.rotation.matmul(p.rotation.transposed())
        let err = (ident - MLXArray.eye(128)).abs().max().item(Float.self)
        XCTAssertLessThan(err, 1e-3, "Π Πᵀ must be ≈ I")
        XCTAssertEqual(p.qjl.shape, [128, 128])
        XCTAssertEqual(p.scaledCentroids.shape, [8])
    }

    // MARK: Task 3 — _mse quantize/dequantize (rotation + LUT)

    func testMSERoundTripErrorDecreasesWithBits() {
        let x = l2normalizeRows(MLXRandom.normal([64, 128], key: MLXRandom.key(7)))
        func recon(_ b: Int) -> Float {
            let p = TurboQuantParams(headDim: 128, baseBits: b, seed: 0)
            let codes = TurboQuantCodec.quantizeMSE(x, params: p)
            let xr = TurboQuantCodec.dequantizeMSE(codes, params: p)
            return (x - xr).square().mean().sqrt().item(Float.self)
        }
        let e2 = recon(2)
        let e3 = recon(3)
        print("turboquant _mse RMSE (unit rows, d=128): b=2 → \(e2), b=3 → \(e3)")
        XCTAssertLessThan(e3, e2, "more base bits → smaller reconstruction error")
        XCTAssertLessThan(e3, 0.2, "3-base-bit reconstruction should be well under 0.2 RMSE on unit vectors")
    }
}

/// Rows scaled to unit L2 norm (test fixture: KV head-vectors are treated as unit-norm
/// for the estimator property tests; norm bookkeeping is the codec's job).
private func l2normalizeRows(_ x: MLXArray) -> MLXArray {
    x / MLX.sqrt((x * x).sum(axis: -1, keepDims: true))
}
