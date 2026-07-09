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
}
