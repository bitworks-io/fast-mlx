import XCTest
@testable import HarnessCore

/// TDD for the serve-tier dial (differentiator #4 slice): `transparent | balanced | maxfit` is the
/// operator-intent contract that turns raw sizer verdicts into a policy the auto-picker and serve
/// path consult. This test pins the *pure* tier→policy mapping — which KV tiers each tier is willing
/// to auto-select, whether it may cap context to make a model fit, and how an explicit `--kv-quant`
/// override interacts with the tier's default set. No model load, no runtime wiring here.
///
/// Honesty rule pinned below: only KV tiers with a real/measured reference (fp16, int8) may be
/// auto-selected; the unmeasured/unbuilt turbo/tq tiers are excluded from every tier's auto-pick set
/// (selecting one would present an unvalidated footprint as a servable plan).
final class ServeTierPolicyTests: XCTestCase {

    // MARK: tier default mappings

    func testTransparentIsFidelityFirstNeverCaps() {
        let p = ServeTierPolicy.resolve(tier: .transparent)
        XCTAssertEqual(p.tier, .transparent)
        XCTAssertEqual(p.allowedKVTiers, [.fp16], "transparent auto-picks fp16 KV only")
        XCTAssertFalse(p.allowContextCapping, "transparent refuses rather than silently capping context")
        XCTAssertNil(p.conflictAnnotation)
    }

    func testBalancedEscalatesKVButRefusesToCap() {
        let p = ServeTierPolicy.resolve(tier: .balanced)
        XCTAssertEqual(p.allowedKVTiers, [.fp16, .int8], "balanced adds the one measured lossy tier")
        XCTAssertFalse(p.allowContextCapping, "balanced trades KV precision to fit, but never silently shortens context")
        XCTAssertNil(p.conflictAnnotation)
    }

    func testMaxfitAllowsCappingButStillExcludesUnmeasuredTiers() {
        let p = ServeTierPolicy.resolve(tier: .maxfit)
        // maxfit is the most aggressive dial, but auto-pick still may not select an UNMEASURED tier
        // (turbo4/tq2_5/tq3_5 are documented placeholders) — that would violate measured-vs-modeled.
        XCTAssertEqual(p.allowedKVTiers, [.fp16, .int8])
        XCTAssertTrue(p.allowContextCapping, "maxfit is the only tier that silently shortens context")
        XCTAssertFalse(p.allowedKVTiers.contains(.turbo4))
        XCTAssertFalse(p.allowedKVTiers.contains(.tq2_5))
    }

    /// The three tiers must be behaviorally DISTINCT — no two resolve to the same (KV set, capping)
    /// policy, or the dial ships a no-op position.
    func testAllThreeTiersAreDistinct() {
        let policies = [ServeTier.transparent, .balanced, .maxfit].map { ServeTierPolicy.resolve(tier: $0) }
        let signatures = policies.map { "\($0.allowedKVTiers.map(\.rawValue))|\($0.allowContextCapping)" }
        XCTAssertEqual(Set(signatures).count, 3, "each tier is a distinct (KV set, capping) stance: \(signatures)")
    }

    func testFidelityOrderingHighestFirst() {
        // allowedKVTiers must be ordered highest-fidelity first — the auto-picker escalates down it.
        for tier in [ServeTier.transparent, .balanced, .maxfit] {
            let tiers = ServeTierPolicy.resolve(tier: tier).allowedKVTiers
            XCTAssertEqual(tiers.first, .fp16, "\(tier) must try fp16 before any lossy tier")
        }
    }

    // MARK: explicit-override precedence

    func testExplicitKVQuantPinsTheSet() {
        // An explicit `--kv-quant int8` pins the auto-pick set to exactly that tier: the operator
        // asked for a specific KV precision, so the picker must not silently try fp16 above it.
        let p = ServeTierPolicy.resolve(tier: .balanced, explicitKVQuant: .int8)
        XCTAssertEqual(p.allowedKVTiers, [.int8])
    }

    func testExplicitKVQuantConflictingWithTierIsAnnotated() {
        // int8 is NOT in transparent's default set (fp16-only). Honoring the explicit pin is correct,
        // but it silently departs from the tier's fidelity contract — so surface a conflict note.
        let p = ServeTierPolicy.resolve(tier: .transparent, explicitKVQuant: .int8)
        XCTAssertEqual(p.allowedKVTiers, [.int8], "explicit pin still wins")
        XCTAssertNotNil(p.conflictAnnotation, "but the departure from transparent's fp16 contract is announced")
        XCTAssertTrue(p.conflictAnnotation!.contains("transparent"))
    }

    // MARK: fail-closed value validation

    func testValidatedParsesKnownTiers() throws {
        XCTAssertEqual(try ServeTier.validated("transparent"), .transparent)
        XCTAssertEqual(try ServeTier.validated("balanced"), .balanced)
        XCTAssertEqual(try ServeTier.validated("maxfit"), .maxfit)
    }

    func testValidatedFailsClosedOnUnknownTier() {
        XCTAssertThrowsError(try ServeTier.validated("turbo")) { error in
            XCTAssertEqual(error as? ServeTierError, .unknownTier("turbo"))
            // the message names the valid values so the operator can correct it
            XCTAssertTrue("\(error)".contains("transparent"))
        }
    }

    func testExplicitKVQuantInTierSetHasNoConflict() {
        // int8 IS in balanced's set, so pinning it is not a conflict — no annotation.
        let p = ServeTierPolicy.resolve(tier: .balanced, explicitKVQuant: .int8)
        XCTAssertNil(p.conflictAnnotation)
    }

    // MARK: enforced-path KV advisory (serve dial on the LOADED path — differentiator #4, item #5)

    func testRuntimeWiredKVTiersAreFP16OnlyForNow() {
        // Single source of truth: the serving runtime stores KV in fp16 today (int8 runtime KV is
        // metallib-gated). A future int8-KV runtime flips exactly this member and the advisory below
        // stops firing — the enforced path must never claim a compressed ceiling the runtime can't honor.
        XCTAssertEqual(ServeTierPolicy.runtimeWiredKVTiers, [.fp16])
    }

    func testEnforcedPathKVAdvisoryIsNilWhenTierAsksNothingBeyondWiredTiers() {
        XCTAssertNil(ServeTierPolicy.enforcedPathKVAdvisory(for: nil),
                     "no --tier → no advisory (byte-identical prior serve)")
        XCTAssertNil(ServeTierPolicy.enforcedPathKVAdvisory(for: ServeTierPolicy.resolve(tier: .transparent)),
                     "transparent auto-picks fp16 only → nothing unhonored")
    }

    func testEnforcedPathKVAdvisoryFiresWhenTierWouldEscalateBeyondFP16() throws {
        for tier in [ServeTier.balanced, .maxfit] {
            let advisory = try XCTUnwrap(
                ServeTierPolicy.enforcedPathKVAdvisory(for: ServeTierPolicy.resolve(tier: tier)),
                "\(tier.rawValue) escalates KV to int8, unhonored by the fp16-only runtime")
            XCTAssertTrue(advisory.contains("int8"), advisory)
            XCTAssertTrue(advisory.contains("runtime_not_wired"), advisory)
            XCTAssertTrue(advisory.contains("fp16"), advisory)
            // Honesty: the advisory must NOT claim int8 is applied on the loaded path.
            XCTAssertFalse(advisory.lowercased().contains("applying int8"), advisory)
        }
    }

    // MARK: frozen fit-check ↔ runtime KV-tier agreement (b-step-2, the identity guard)

    // The load-bearing containment chain for the "never ship a quality regression" identity. Three
    // distinct KV-tier sets must stay ordered so the KV-quant moat can grow WITHOUT a wiring change
    // silently promoting an unmeasured-quality format to a served default:
    //   qualityApproved ⊆ runtimeWired ⊆ measured
    // qualityApproved = quality measured & approved to auto-serve; runtimeWired = the runtime can store
    // it; measured = a real reference footprint exists. This test passes trivially today (all three
    // relevant boundaries hold with fp16/int8) and FAILS the instant any future edit drifts a set out of
    // order — e.g. wiring int8 into `runtimeWiredKVTiers` is fine, but ALSO adding int8 to
    // `qualityApprovedKVTiers` without dated big-box quality evidence would (correctly) still pass this
    // chain; what this guard forbids is qualityApproved escaping runtimeWired or runtimeWired escaping
    // measured, and (below) any tier's enforced-served set escaping qualityApproved.
    func testKVTierSetsRespectQualityApprovedSubsetRuntimeWiredSubsetMeasured() {
        let measured = Set(ServeTierPolicy.measuredKVTiers)
        let wired = Set(ServeTierPolicy.runtimeWiredKVTiers)
        let approved = Set(ServeTierPolicy.qualityApprovedKVTiers)
        XCTAssertTrue(approved.isSubset(of: wired),
                      "a KV tier cannot be quality-approved to serve if the runtime cannot even store it")
        XCTAssertTrue(wired.isSubset(of: measured),
                      "the runtime must not be wired for a KV tier with no measured reference footprint")
    }

    // Concrete pins: any drift of the frozen sets trips the test (they are the single sources of truth).
    func testKVTierSetConcreteValuesAreFrozen() {
        XCTAssertEqual(ServeTierPolicy.measuredKVTiers, [.fp16, .int8])
        XCTAssertEqual(ServeTierPolicy.runtimeWiredKVTiers, [.fp16])
        XCTAssertEqual(ServeTierPolicy.qualityApprovedKVTiers, [.fp16],
                       "int8 stays out of the served-default set until a dated big-box quality PASS")
        XCTAssertEqual(ServeTierPolicy.enforcedServableKVTiers, [.fp16],
                       "the enforced path serves fp16 only until int8 is BOTH wired AND quality-approved")
    }

    // The structural fix's whole point: the enforced-served set is the intersection (runtime-wired AND
    // quality-approved), NOT capability alone. For every tier, whatever the planner's allowedKVTiers
    // proposes, the enforced path may only actually serve a subset of the quality-approved set — so a
    // future runtime wiring cannot leak an unapproved format onto a loaded model.
    func testEveryTierEnforcedServedKVIsSubsetOfQualityApproved() {
        let approved = Set(ServeTierPolicy.qualityApprovedKVTiers)
        let servable = Set(ServeTierPolicy.enforcedServableKVTiers)
        XCTAssertTrue(servable.isSubset(of: approved),
                      "the enforced path must never store a KV format outside the quality-approved set")
        for tier in ServeTier.allCases {
            let planned = Set(ServeTierPolicy.resolve(tier: tier).allowedKVTiers)
            let actuallyServed = planned.intersection(servable)
            XCTAssertTrue(actuallyServed.isSubset(of: approved),
                          "\(tier.rawValue): enforced-served KV \(actuallyServed) escaped quality-approved \(approved)")
            XCTAssertTrue(planned.isSubset(of: Set(ServeTierPolicy.measuredKVTiers)),
                          "\(tier.rawValue): auto-pick set \(planned) escaped the measured set (honesty rule)")
        }
    }
}
