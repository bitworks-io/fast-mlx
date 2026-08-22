import XCTest
@testable import HarnessCore

/// TDD for the adaptive solo-speculation policy: the first brick of the adaptive concurrency-aware
/// serving mode. The continuous-batch scheduler's `makeTick()` already decides solo-vs-batch from the
/// in-flight candidate count (1 → solo single-stream, ≥2 → batch) — this policy does NOT re-decide that
/// (a second mode authority would be a bug). It ONLY decides whether SOLO SPECULATION is enabled, with
/// hysteresis to prevent thrash at the 1↔2 boundary, and suppression when a companion request is
/// imminent (queued/prefilling) so we don't draft tokens that a drain will immediately throw away. This
/// is also the seam MTP speculative decode will later plug into.
final class AdaptiveSoloSpeculationPolicyTests: XCTestCase {

    // MARK: mode mapping

    func testZeroCandidatesIsIdleRegardlessOfQueue() throws {
        let policy = try AdaptiveSoloSpeculationPolicy(speculationWarmupTicks: 3)
        let signals = AdaptiveSoloSpeculationPolicy.Signals(
            decodeCandidates: 0, prefillingRequests: 3, queuedRequests: 5, ticksSinceMembershipChange: 0)
        XCTAssertEqual(try policy.mode(for: signals), .idle)
    }

    func testTwoOrMoreCandidatesAlwaysBatchEvenInsideWarmup() throws {
        let policy = try AdaptiveSoloSpeculationPolicy(speculationWarmupTicks: 5)
        let signals = AdaptiveSoloSpeculationPolicy.Signals(
            decodeCandidates: 2, prefillingRequests: 0, queuedRequests: 0, ticksSinceMembershipChange: 0)
        XCTAssertEqual(try policy.mode(for: signals), .batch)
    }

    func testStableSoloWithEmptyHorizonEnablesSpeculation() throws {
        let policy = try AdaptiveSoloSpeculationPolicy(speculationWarmupTicks: 3)
        let signals = AdaptiveSoloSpeculationPolicy.Signals(
            decodeCandidates: 1, prefillingRequests: 0, queuedRequests: 0, ticksSinceMembershipChange: 10)
        XCTAssertEqual(try policy.mode(for: signals), .soloSpeculative)
    }

    func testQueuedArrivalSuppressesSpeculation() throws {
        let policy = try AdaptiveSoloSpeculationPolicy(speculationWarmupTicks: 3)
        let signals = AdaptiveSoloSpeculationPolicy.Signals(
            decodeCandidates: 1, prefillingRequests: 0, queuedRequests: 1, ticksSinceMembershipChange: 100)
        XCTAssertEqual(try policy.mode(for: signals), .soloPlain)
    }

    func testPrefillingCompanionSuppressesSpeculation() throws {
        let policy = try AdaptiveSoloSpeculationPolicy(speculationWarmupTicks: 3)
        let signals = AdaptiveSoloSpeculationPolicy.Signals(
            decodeCandidates: 1, prefillingRequests: 1, queuedRequests: 0, ticksSinceMembershipChange: 100)
        XCTAssertEqual(try policy.mode(for: signals), .soloPlain)
    }

    func testWarmupBoundaryIsExactlyAtThreshold() throws {
        let policy = try AdaptiveSoloSpeculationPolicy(speculationWarmupTicks: 3)
        let belowThreshold = AdaptiveSoloSpeculationPolicy.Signals(
            decodeCandidates: 1, prefillingRequests: 0, queuedRequests: 0, ticksSinceMembershipChange: 2)
        XCTAssertEqual(try policy.mode(for: belowThreshold), .soloPlain, "ticks < warmup stays suppressed")
        let atThreshold = AdaptiveSoloSpeculationPolicy.Signals(
            decodeCandidates: 1, prefillingRequests: 0, queuedRequests: 0, ticksSinceMembershipChange: 3)
        XCTAssertEqual(try policy.mode(for: atThreshold), .soloSpeculative, "ticks == warmup flips to speculative")
    }

    func testZeroWarmupEnablesSpeculationImmediately() throws {
        let policy = try AdaptiveSoloSpeculationPolicy(speculationWarmupTicks: 0)
        let signals = AdaptiveSoloSpeculationPolicy.Signals(
            decodeCandidates: 1, prefillingRequests: 0, queuedRequests: 0, ticksSinceMembershipChange: 0)
        XCTAssertEqual(try policy.mode(for: signals), .soloSpeculative)
    }

    // MARK: fail-closed value validation

    func testNegativeConfigurationAndSignalsFailClosed() throws {
        XCTAssertThrowsError(try AdaptiveSoloSpeculationPolicy(speculationWarmupTicks: -1)) { error in
            XCTAssertEqual(error as? AdaptiveSoloSpeculationPolicyError, .negativeWarmupTicks(-1))
        }

        let policy = try AdaptiveSoloSpeculationPolicy(speculationWarmupTicks: 3)
        let negativeCandidates = AdaptiveSoloSpeculationPolicy.Signals(
            decodeCandidates: -1, prefillingRequests: 0, queuedRequests: 0, ticksSinceMembershipChange: 0)
        XCTAssertThrowsError(try policy.mode(for: negativeCandidates)) { error in
            XCTAssertEqual(error as? AdaptiveSoloSpeculationPolicyError, .negativeSignal("decodeCandidates", -1))
        }
    }

    // MARK: anti-thrash acceptance case

    /// The acceptance-shaped anti-thrash case: drive a sequence of candidate counts across the 1↔2
    /// boundary the way the real scheduler will — the TEST maintains `ticksSinceMembershipChange`
    /// itself (reset to 0 on any candidate-count change, else +1), mirroring the caller's contract that
    /// this policy never stores mutable state. Asserts speculation is OFF (.batch) during the 2-candidate
    /// spike, and stays suppressed (.soloPlain) for exactly `warmup` ticks after returning to 1 candidate,
    /// only flipping to .soloSpeculative once the warmup window has fully elapsed.
    func testNoThrashAcrossOneToTwoBoundarySequence() throws {
        let warmup = 3
        let policy = try AdaptiveSoloSpeculationPolicy(speculationWarmupTicks: warmup)

        let candidateSequence = [1, 1, 1, 2, 1, 1, 1, 1, 1, 1]
        var ticksSinceMembershipChange = 0
        var previousCandidates: Int?
        var observedModes: [AdaptiveSoloSpeculationPolicy.Mode] = []

        for candidates in candidateSequence {
            if let previous = previousCandidates, previous != candidates {
                ticksSinceMembershipChange = 0
            } else if previousCandidates != nil {
                ticksSinceMembershipChange += 1
            }
            previousCandidates = candidates

            let signals = AdaptiveSoloSpeculationPolicy.Signals(
                decodeCandidates: candidates, prefillingRequests: 0, queuedRequests: 0,
                ticksSinceMembershipChange: ticksSinceMembershipChange)
            observedModes.append(try policy.mode(for: signals))
        }

        // Indices: 0,1,2 = solo (pre-spike), 3 = the spike, 4..9 = solo (post-spike, 6 ticks).
        XCTAssertEqual(observedModes[3], .batch, "the 2-candidate spike must never permit speculation")
        // Post-spike ticksSinceMembershipChange resets to 0 at index 4, so indices 4,5,6 have ticks
        // 0,1,2 — all below warmup(3) — and must stay suppressed.
        XCTAssertEqual(observedModes[4], .soloPlain)
        XCTAssertEqual(observedModes[5], .soloPlain)
        XCTAssertEqual(observedModes[6], .soloPlain)
        // Index 7 has ticksSinceMembershipChange == 3 == warmup: speculation re-enables.
        XCTAssertEqual(observedModes[7], .soloSpeculative)
        XCTAssertEqual(observedModes[8], .soloSpeculative)
        XCTAssertEqual(observedModes[9], .soloSpeculative)
    }
}
