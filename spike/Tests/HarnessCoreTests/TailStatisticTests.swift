import Testing

@testable import HarnessCore

/// Tail-aware long-context KL statistic (sub-task: long-context median reads BELOW the noise
/// floor because easy function-word tokens dominate; the KV-quant divergence signal lives in
/// the TAIL — observed pooled p95 0.69 vs median 6e-05 on the first real lossy run).
struct TailStatisticTests {

    // MARK: quantile — one deterministic convention (index = floor(q * (n-1)) of the sorted
    // values), shared by the pooled-p95 diagnostic and the long-context headline.

    @Test func quantileOfEmptyIsZero() {
        #expect(quantile([], 0.95) == 0)
    }

    @Test func quantileOfSingleValueIsThatValue() {
        #expect(quantile([0.42], 0.95) == 0.42)
        #expect(quantile([0.42], 0.0) == 0.42)
    }

    @Test func quantileUsesCeilingIndexConvention() {
        let values = (1...100).map(Double.init) // sorted 1...100
        // ceil(0.95 * 99) = 95 -> sorted[95] = 96 (ceiling: err toward the tail, never below it)
        #expect(quantile(values, 0.95) == 96)
        // ceil(0.5 * 99) = 50 -> sorted[50] = 51
        #expect(quantile(values, 0.5) == 51)
    }

    /// The boundary case that decides the convention: a divergence tail of EXACTLY 5% mass.
    /// floor(q*(n-1)) lands on the last easy-token value (reads noise, 6e-5); ceiling lands on
    /// the first tail value (reads the divergence). A lossiness instrument must not
    /// under-report at exactly the boundary it exists to catch.
    @Test func quantileCeilingCatchesAnExactBoundaryTail() {
        let kls = (0..<95).map { _ in 6e-5 } + (0..<5).map { _ in 0.69 } // n=100, 5% tail
        #expect(quantile(kls, 0.95) == 0.69)
    }

    @Test func quantileSortsItsInput() {
        #expect(quantile([5, 1, 4, 2, 3], 1.0) == 5)
        #expect(quantile([5, 1, 4, 2, 3], 0.0) == 1)
    }

    @Test func quantileExtremesAreMinAndMax() {
        let values = [0.7, 0.1, 0.4]
        #expect(quantile(values, 0.0) == 0.1)
        #expect(quantile(values, 1.0) == 0.7)
    }

    // MARK: longContextTailKL — per-entry tail quantile, median across entries.

    @Test func singleEntryHeadlineIsItsTailQuantile() {
        let kls = (0..<100).map { _ in 1e-5 } + [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] // 5.7% tail
        let expected = quantile(kls, 0.95)
        #expect(longContextTailKL(perEntryKLs: [kls]) == expected)
        #expect(expected > 0.1) // and that tail is a real signal, not noise-floor mush
    }

    @Test func multipleEntriesCombineByMedianOfPerEntryTails() {
        let a = [0.0, 0.0, 1.0]  // p95 = 1.0
        let b = [0.0, 0.0, 3.0]  // p95 = 3.0
        let c = [0.0, 0.0, 2.0]  // p95 = 2.0
        #expect(longContextTailKL(perEntryKLs: [a, b, c]) == 2.0)
    }

    @Test func noEntriesYieldsZero() {
        #expect(longContextTailKL(perEntryKLs: []) == 0)
    }

    /// THE regression this statistic exists for: a realistic long-context KL profile — the
    /// overwhelming majority of positions are easy tokens both quants agree on (KL below the
    /// same-weights noise floor of ~1.3e-3 nats), while a 5% tail carries the KV-quant
    /// divergence. The median headline reads as noise; the tail headline reads as signal.
    @Test func tailStatisticSeesSignalTheMedianMisses() {
        let noiseFloor = 1.314e-3 // measured same-weights (pipeline-proof) median, nats
        var kls = (0..<9500).map { _ in 6e-5 }   // easy positions: far below the floor
        kls += (0..<500).map { _ in 0.69 }       // the divergence tail
        kls.shuffle()
        #expect(medianOf(kls) < noiseFloor)                            // the flaw
        #expect(longContextTailKL(perEntryKLs: [kls]) > 10 * noiseFloor) // the fix
    }
}
