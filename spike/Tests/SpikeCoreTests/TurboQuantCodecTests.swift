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

    // MARK: Task 4 — _prod QJL residual (Spike A: the make-or-break gate, paper-faithful form)

    /// The three properties TurboQuant Theorem 2 actually claims for `_prod`, all at real
    /// head_dim (d=128, float32). This replaces a naive `prodErr < mseErr` assertion on
    /// INDEPENDENT random query/key pairs, which is the wrong gate: the unbiased QJL residual
    /// estimate carries variance ≈ (π/2)·‖r‖²/d — a factor π/2 *worse* than dropping the
    /// residual when q ⊥ x on average, so `_mse` wins that comparison by the paper's own
    /// distortion table (0.122/d vs 0.18/d at 3 total bits). `_prod`'s value is removing
    /// `_mse`'s shrinkage *bias* (⟨q,r⟩ ≈ ‖r‖²·⟨q,x⟩), which dominates once q correlates
    /// with x (crossover ≈ ⟨q,x⟩ ~ 0.2) — and attention IS the correlated regime: softmax
    /// is driven by the high scores. Full derivation + measurements:
    /// docs/reference/turboquant-algorithm.md §"Spike A resolution".
    func testProdInnerProductProperties() {
        let d = 128
        let n = 4096
        let keys = l2normalizeRows(MLXRandom.normal([n, d], key: MLXRandom.key(1)))
        let queries = l2normalizeRows(MLXRandom.normal([n, d], key: MLXRandom.key(2)))
        let exact = (queries * keys).sum(axis: -1)
        let p = TurboQuantParams(headDim: d, baseBits: 2, seed: 0)
        let code = TurboQuantCodec.quantizeProd(keys, params: p)
        let prodK = TurboQuantCodec.dequantizeProd(code, params: p)

        // (a) Unbiasedness: regression slope of ⟨q, x̃_prod⟩ on ⟨q, x⟩ ≈ 1 — this pins the
        // √(π/2)/d dequant scale (any other scale biases the slope).
        let est = (queries * prodK).sum(axis: -1)
        let slope = ((exact * est).mean() / (exact * exact).mean()).item(Float.self)
        print("turboquant Spike A (a) unbiasedness slope: \(slope)")
        XCTAssertEqual(slope, 1.0, accuracy: 0.05, "⟨q, x̃_prod⟩ must be an unbiased estimate of ⟨q, x⟩")

        // (b) Correlated-regime superiority (the KV-relevant property): for queries with a
        // real component along the key (mean ⟨q,x⟩ ≈ 0.71 ≥ 0.45), _prod beats _mse by ≥ 1.8×.
        let base = TurboQuantCodec.dequantizeMSE(code.idx, params: p)
        let z = l2normalizeRows(MLXRandom.normal([n, d], key: MLXRandom.key(3)))
        let qc = l2normalizeRows(keys + z)
        let exactC = (qc * keys).sum(axis: -1)
        let meanCorr = exactC.mean().item(Float.self)
        let mseErrC = ((qc * base).sum(axis: -1) - exactC).abs().mean().item(Float.self)
        let prodErrC = ((qc * prodK).sum(axis: -1) - exactC).abs().mean().item(Float.self)
        print("turboquant Spike A (b) correlated q (mean ⟨q,x⟩ = \(meanCorr)): prodErr \(prodErrC) vs mseErr \(mseErrC), margin \(mseErrC / prodErrC)×")
        XCTAssertGreaterThanOrEqual(meanCorr, 0.45)
        XCTAssertGreaterThanOrEqual(
            mseErrC / prodErrC, 1.8,
            "_prod must beat _mse decisively for correlated queries (the attention regime)")

        // (c) Theorem-2 anchor: d·E[(⟨q,x̃⟩−⟨q,x⟩)²] matches the paper's empirical distortion
        // table within 20% — ≈ 0.18 at 3 total bits (base 2), ≈ 0.047 at 4 total bits (base 3).
        for (baseBits, expected) in [(2, Float(0.18)), (3, Float(0.047))] {
            let pb = TurboQuantParams(headDim: d, baseBits: baseBits, seed: 0)
            let cb = TurboQuantCodec.quantizeProd(keys, params: pb)
            let kb = TurboQuantCodec.dequantizeProd(cb, params: pb)
            let dist = (((queries * kb).sum(axis: -1) - exact).square().mean() * Float(d))
                .item(Float.self)
            print("turboquant Spike A (c) totalBits=\(baseBits + 1): d·D_prod = \(dist) (paper ≈ \(expected))")
            XCTAssertEqual(
                dist, expected, accuracy: expected * 0.2,
                "d·D_prod at \(baseBits + 1) total bits must reproduce the paper's distortion table")
        }
    }

    // MARK: Task 6a — non-unit-norm handling (paper §1.1: store ‖x‖, normalize, rescale)

    /// Real K/V head-vectors are not unit-norm; the codec's codebook is sized for unit vectors.
    /// The normalized wrapper must round-trip rows of varying norm with *relative* RMSE
    /// comparable to the proven unit-norm reconstruction (which Spike A validated).
    func testNormalizedRoundTripHandlesVaryingNorms() {
        let d = 128
        let n = 512
        let unit = l2normalizeRows(MLXRandom.normal([n, d], key: MLXRandom.key(11)))
        // norms in [0.5, 5] — the non-unit regime Phase 1B flagged as the correctness gap
        let scales = MLXRandom.uniform(low: Float(0.5), high: Float(5.0), [n, 1], key: MLXRandom.key(12))
        let x = unit * scales
        let p = TurboQuantParams(headDim: d, baseBits: 3, seed: 0)

        // Reference: the proven codec's reconstruction RMSE on the SAME directions at unit norm.
        let unitCode = TurboQuantCodec.quantizeProd(unit, params: p)
        let unitRMSE = (unit - TurboQuantCodec.dequantizeProd(unitCode, params: p))
            .square().mean().sqrt().item(Float.self)

        let code = TurboQuantCodec.quantizeProdNormalized(x, params: p)
        XCTAssertEqual(code.xNorm?.shape, [n, 1], "per-row ‖x‖ must be stored as [n, 1]")
        let xr = TurboQuantCodec.dequantizeProdNormalized(code, params: p)
        // Per-row relative residual: dividing by the true norm makes this directly comparable
        // to the unit-norm RMSE (for exact normalize/rescale they are the same quantity).
        let relRMSE = ((x - xr) / scales).square().mean().sqrt().item(Float.self)
        print("turboquant 6a: relative RMSE \(relRMSE) vs unit-norm RMSE \(unitRMSE)")
        XCTAssertLessThan(
            relRMSE, unitRMSE * 1.2,
            "normalized round-trip must match the unit-norm reconstruction quality")
        // Stored ‖x‖ must be the actual input norm (this is what the γ-vs-‖x‖ split protects).
        let xNormErr = (code.xNorm! - scales).abs().max().item(Float.self)
        XCTAssertLessThan(xNormErr, 1e-4, "xNorm must be the per-row input L2 norm")
    }

    /// Scaling the input by c must scale the reconstruction by exactly c (up to float rounding):
    /// x/‖x‖ is scale-invariant, so codes are identical and only the stored ‖x‖ changes.
    func testNormalizedReconstructionIsNormCovariant() {
        let d = 64
        let n = 128
        let x = MLXRandom.normal([n, d], key: MLXRandom.key(21))
        let c: Float = 3.7
        let p = TurboQuantParams(headDim: d, baseBits: 2, seed: 0)

        let r1 = TurboQuantCodec.dequantizeProdNormalized(
            TurboQuantCodec.quantizeProdNormalized(x, params: p), params: p)
        let r2 = TurboQuantCodec.dequantizeProdNormalized(
            TurboQuantCodec.quantizeProdNormalized(x * c, params: p), params: p)
        let covErr = (r2 - r1 * c).abs().max().item(Float.self)
        let scale = r1.abs().max().item(Float.self) * c
        print("turboquant 6a covariance: max |recon(c·x) − c·recon(x)| = \(covErr) (scale \(scale))")
        XCTAssertLessThan(covErr, scale * 1e-4, "reconstruction must be norm-covariant")
    }

    // MARK: Task 5 — tier config + honest bits/element accounting

    func testTierBitsPerElementIsHonest() {
        XCTAssertEqual(TurboQuantTier.tqB2.baseBits, 2)
        XCTAssertEqual(TurboQuantTier.tqB3.baseBits, 3)
        // base bits + 1 QJL bit + the fp16 γ=‖r‖₂ amortized over head_dim.
        XCTAssertEqual(TurboQuantTier.tqB2.bitsPerElement(headDim: 128), 3 + 16.0 / 128)
        XCTAssertEqual(TurboQuantTier.tqB3.bitsPerElement(headDim: 128), 4 + 16.0 / 128)
        XCTAssertEqual(TurboQuantTier.tqB2.harnessSlot, "tq2.5")
        XCTAssertEqual(TurboQuantTier.tqB3.harnessSlot, "tq3.5")
    }
}

/// Rows scaled to unit L2 norm (test fixture: KV head-vectors are treated as unit-norm
/// for the estimator property tests; norm bookkeeping is the codec's job).
private func l2normalizeRows(_ x: MLXArray) -> MLXArray {
    x / MLX.sqrt((x * x).sum(axis: -1, keepDims: true))
}
