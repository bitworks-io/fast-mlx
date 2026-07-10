import Foundation

/// Lloyd-Max optimal scalar quantizer levels for the standard normal N(0,1) (TurboQuant §"The
/// algorithm": post-rotation coordinates are ≈ N(0, 1/d), so we solve on N(0,1) once and let the
/// codec rescale by 1/√d). Pure — no MLX. Solved numerically (no closed form; the paper caches a LUT).
public enum LloydMaxCodebook {
    private static func phi(_ x: Double) -> Double {          // N(0,1) pdf
        guard x.isFinite else { return 0 }
        return exp(-x * x / 2) / (2 * Double.pi).squareRoot()
    }
    private static func bigPhi(_ x: Double) -> Double {        // N(0,1) cdf
        if x == .infinity { return 1 }; if x == -.infinity { return 0 }
        return 0.5 * (1 + erf(x / 2.0.squareRoot()))
    }

    /// `2^bits` ascending centroids for N(0,1). Deterministic (fixed init + fixed iteration count).
    public static func gaussian(bits: Int, iterations: Int = 200) -> [Double] {
        precondition(bits >= 1, "bits must be ≥ 1")
        let k = 1 << bits
        // Deterministic init: evenly spaced in [-3, 3].
        var c = (0..<k).map { 3.0 * (2.0 * (Double($0) + 0.5) / Double(k) - 1.0) }
        for _ in 0..<iterations {
            var t = [Double](repeating: 0, count: k + 1)      // decision boundaries
            t[0] = -.infinity; t[k] = .infinity
            for i in 1..<k { t[i] = (c[i - 1] + c[i]) / 2 }
            for i in 0..<k {                                  // centroid = conditional mean
                let den = bigPhi(t[i + 1]) - bigPhi(t[i])
                if den > 1e-12 { c[i] = (phi(t[i]) - phi(t[i + 1])) / den }
            }
        }
        return c
    }
}
