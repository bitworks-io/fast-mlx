import XCTest

@testable import SpikeServingAdapters

/// Unit tests for `scalarServingIsNonSpeculativeScalarRoute`, the pure function
/// `loadScalarServingModel` factors its `response_format: json_object` route-flag derivation
/// through (response-format design item #4 / must-fix #4). Each test name states the concrete
/// scalar-serving configuration it stands in for; "sampled MTP" and "in-checkpoint MTP" resolve to
/// the SAME two inputs on purpose — see the function's own doc comment for why sampled block
/// decisions do not change which decoder is bound, only how it behaves once bound.
final class ScalarServingIsNonSpeculativeRouteTests: XCTestCase {
    /// Plain scalar (no MTP at all): `.nativeCaches` with no retained drafter.
    func testPlainScalarRouteIsNonSpeculative() {
        XCTAssertTrue(
            scalarServingIsNonSpeculativeScalarRoute(
                decoderStrategy: .nativeCaches(.fp16),
                hasRetainedInCheckpointMTPDrafter: false))
    }

    /// Compiled fp16 with no retained drafter: NOW non-speculative. `.compiledFP16` resolves to a
    /// `RouteSwitchingDecoder` whose general side genuinely honors `setResponseFormatConstraint`
    /// (see that type's doc comment), so it is no longer excluded from this gate the way the
    /// greedy-only `CompiledMLXDecoder` alone used to be.
    func testCompiledRouteWithNoDrafterIsNonSpeculative() {
        XCTAssertTrue(
            scalarServingIsNonSpeculativeScalarRoute(
                decoderStrategy: .compiledFP16,
                hasRetainedInCheckpointMTPDrafter: false))
    }

    /// Compiled fp16 WITH a retained drafter: still `false`. This pairing cannot occur in
    /// production (`scalarServingInCheckpointMTPDecoderStrategyError` refuses it at load), but the
    /// pure function must still answer defensively rather than assume the combination is
    /// unreachable — see the function's own doc comment.
    func testCompiledRouteWithDrafterIsNotNonSpeculative() {
        XCTAssertFalse(
            scalarServingIsNonSpeculativeScalarRoute(
                decoderStrategy: .compiledFP16,
                hasRetainedInCheckpointMTPDrafter: true))
    }

    /// In-checkpoint MTP: `.nativeCaches` WITH a retained drafter routes to
    /// `MTPSpeculativeDecoder`, never the plain `MLXDecoder` this capability requires.
    func testInCheckpointMTPRouteIsNotNonSpeculative() {
        XCTAssertFalse(
            scalarServingIsNonSpeculativeScalarRoute(
                decoderStrategy: .nativeCaches(.fp16),
                hasRetainedInCheckpointMTPDrafter: true))
    }

    /// Sampled MTP block decisions are a RUNTIME flag layered onto the same retained-drafter
    /// `MTPSpeculativeDecoder` composition as in-checkpoint MTP — this function's two inputs
    /// (decoder strategy, retained-drafter presence) cannot see that flag at all, and correctly so:
    /// the flag never changes which decoder type is bound, so it must never change this answer.
    func testSampledMTPRouteIsNotNonSpeculative() {
        XCTAssertFalse(
            scalarServingIsNonSpeculativeScalarRoute(
                decoderStrategy: .nativeCaches(.fp16),
                hasRetainedInCheckpointMTPDrafter: true))
    }

    /// Control: an int8 KV tier on the plain scalar route (no drafter retained) is STILL
    /// non-speculative — proving the function keys on the drafter/strategy SHAPE, not on which
    /// specific `KVCacheQuantDecision` case is inside `.nativeCaches`.
    func testInt8KVTierPlainScalarRouteIsStillNonSpeculative() {
        XCTAssertTrue(
            scalarServingIsNonSpeculativeScalarRoute(
                decoderStrategy: .nativeCaches(.int8(groupSize: 64, bits: 8)),
                hasRetainedInCheckpointMTPDrafter: false))
    }
}
