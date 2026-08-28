import Foundation
import XCTest
@testable import HarnessCore

/// TDD for the pure serving fit-decision (fit-checked-serve, differentiator #2). The whole
/// proceed/refuse + limit-derivation + context-capping policy is exercised off-box here; the live
/// serve only confirms the wired numbers match.
final class ServingFitPlannerTests: XCTestCase {
    private let gib = 1024 * 1024 * 1024

    private func profile(_ id: String) -> ModelArchProfile {
        guard let m = ModelArchProfile.catalog.first(where: { $0.id == id }) else {
            fatalError("missing catalog entry \(id)")
        }
        return m
    }

    private var qualifiedDedicatedHost: SystemProfile {
        SystemProfile(
            chip: "test dedicated", totalRAMBytes: 128 * gib, wiredLimitBytes: 115 * gib,
            wiredLimitIsMeasured: true, hostUse: .operatorAssertedDedicatedServing())
    }

    // MARK: - fitting model → green, limits from the sizer (NOT flat %)

    func testFittingDenseModel_green_proceeds_withSizerLimits() {
        let host = qualifiedDedicatedHost // 128 GiB, wired 115 GiB (measured)
        let d = ServingFitPlanner.decide(
            profile: profile("Qwen3-32B"), weightsAreMeasured: true, host: host)
        XCTAssertEqual(d.color, .green)
        XCTAssertTrue(d.shouldProceed)
        XCTAssertFalse(d.proceedingUnderForce)
        XCTAssertFalse(d.contextWasCapped)

        // cache limit is the sizer's recommendation, not RAM×10%.
        let recommended = CapacityModel.recommendedCacheLimitBytes(wiredLimitBytes: host.wiredLimitBytes)
        XCTAssertEqual(d.cacheLimitBytes, recommended)
        XCTAssertNotEqual(d.cacheLimitBytes, host.totalRAMBytes / 10, "must not be the old flat RAM×10%")
        // memory limit is the wired/RAM envelope, not RAM×80%.
        XCTAssertEqual(d.memoryLimitBytes, min(host.wiredLimitBytes, host.totalRAMBytes))
        // reserved KV is the sizer's predicted KV for the served context, not RAM×30%.
        let kv = CapacityModel.kvBytesForContext(profile("Qwen3-32B"), context: d.servedContext, kvQuant: .fp16, concurrency: 1)
        XCTAssertEqual(d.maxReservedKVBytes, Int(kv.rounded(.up)))
        XCTAssertNotEqual(d.maxReservedKVBytes, host.totalRAMBytes * 30 / 100)

        // provenance threaded through
        XCTAssertTrue(d.weightsAreMeasured)
        XCTAssertTrue(d.wiredLimitIsMeasured)
    }

    /// Non-regression: with no --context, a box that can hold the full native context serves up to
    /// the native max (== the backend's nil-maxContextTokens cap), NOT the 32K effective default —
    /// otherwise the wired maxContextTokens would newly hard-reject longer requests that used to work.
    func testDefaultServedContextIsMaxThatFits_notFlat32K() {
        let host = qualifiedDedicatedHost
        let m = profile("Qwen3-32B") // nativeMax 40960, fits fully on 128 GiB
        let d = ServingFitPlanner.decide(profile: m, weightsAreMeasured: true, host: host)
        XCTAssertEqual(d.contextCeiling, m.nativeMaxContext, "the full native context fits on this box")
        XCTAssertEqual(d.servedContext, m.nativeMaxContext, "default serves up to native max, not 32K")
        XCTAssertNotEqual(d.servedContext, 32768)
        XCTAssertFalse(d.contextWasCapped)
    }

    // MARK: - too-big model on a small host → red → refuse; --force overrides

    private func smallHost(ramGiB: Int, wiredGiB: Int) -> SystemProfile {
        SystemProfile(
            chip: "test", totalRAMBytes: ramGiB * gib, wiredLimitBytes: wiredGiB * gib,
            wiredLimitIsMeasured: true, hostUse: .operatorAssertedDedicatedServing())
    }

    func testTooBigModel_smallHost_red_refusesByDefault() {
        // GLM-4.5-Air weights ≈ 53 GiB; an 8 GiB box can't even hold the weights.
        let host = smallHost(ramGiB: 8, wiredGiB: 6)
        let d = ServingFitPlanner.decide(profile: profile("GLM-4.5-Air"), weightsAreMeasured: true, host: host)
        XCTAssertEqual(d.color, .red)
        XCTAssertFalse(d.shouldProceed, "red must fail closed without --force")
        XCTAssertTrue([.physicalRAM, .wiredLimit].contains(d.bindingConstraint))
        XCTAssertEqual(d.contextCeiling, 0, "nothing fits at any context")
    }

    func testTooBigModel_forceOverrides() {
        let host = smallHost(ramGiB: 8, wiredGiB: 6)
        let d = ServingFitPlanner.decide(profile: profile("GLM-4.5-Air"), weightsAreMeasured: true, host: host, force: true)
        XCTAssertEqual(d.color, .red)
        XCTAssertTrue(d.shouldProceed, "--force proceeds past red")
        XCTAssertTrue(d.proceedingUnderForce)
    }

    // MARK: - default context beyond the ceiling → auto-cap + announce (not red)

    func testDefaultContext_autoCapsToCeiling_andProceeds() {
        // Phi-4 (nativeMax 16384): weights ≈ 7 GiB. Sized so weights+base fit but 16K KV doesn't,
        // forcing an auto-cap to a smaller context that fits.
        let host = smallHost(ramGiB: 20, wiredGiB: 18)
        let d = ServingFitPlanner.decide(profile: profile("Phi-4-14B"), weightsAreMeasured: true, host: host)
        XCTAssertGreaterThan(d.contextCeiling, 0)
        XCTAssertLessThan(d.contextCeiling, 16384, "the 16K default does not fit here")
        XCTAssertTrue(d.contextWasCapped)
        XCTAssertEqual(d.servedContext, d.contextCeiling)
        XCTAssertNotEqual(d.color, .red, "capping to the ceiling keeps us out of the red")
        XCTAssertTrue(d.shouldProceed)
        XCTAssertFalse(d.explicitContextRequested)
    }

    // MARK: - allowContextCapping=false (transparent/balanced tiers) → refuse a memory-bound cap

    /// The same box that auto-caps Phi-4's default context above now REFUSES instead when the tier
    /// forbids silently shortening context (`allowContextCapping=false`): no partial context is served
    /// behind the operator's back. The ceiling that WOULD have fit is still reported so a re-run under
    /// maxfit or an explicit smaller --context is one step away.
    func testDefaultContext_noCapping_refusesInsteadOfSilentlyShortening() {
        let host = smallHost(ramGiB: 20, wiredGiB: 18)
        let capped = ServingFitPlanner.decide(profile: profile("Phi-4-14B"), weightsAreMeasured: true, host: host)
        XCTAssertTrue(capped.contextWasCapped, "precondition: default behavior caps here")

        let d = ServingFitPlanner.decide(
            profile: profile("Phi-4-14B"), weightsAreMeasured: true, host: host, allowContextCapping: false)
        XCTAssertFalse(d.shouldProceed, "no-cap tier refuses rather than serving a shortened context")
        XCTAssertFalse(d.contextWasCapped, "we are refusing the full ask, not serving a capped one")
        XCTAssertEqual(d.color, .red, "the full ask is assessed (memory-bound red)")
        XCTAssertGreaterThan(d.contextCeiling, 0, "the fitting ceiling is still surfaced for a maxfit re-run")
        XCTAssertLessThan(d.contextCeiling, 16384)
    }

    /// A request that fits WITHOUT any cap is unaffected by the no-cap stance — it still proceeds.
    func testFittingRequest_noCapping_stillProceeds() {
        let host = qualifiedDedicatedHost
        let d = ServingFitPlanner.decide(
            profile: profile("Qwen3-32B"), weightsAreMeasured: true, host: host, allowContextCapping: false)
        XCTAssertTrue(d.shouldProceed, "no cap was needed → no-cap policy changes nothing")
        XCTAssertFalse(d.contextWasCapped)
        XCTAssertEqual(d.servedContext, profile("Qwen3-32B").nativeMaxContext)
    }

    func testLowerPositiveMetalRecommendationDrivesLimitsAndCanBindRedWhenWiredWouldFit() {
        let wiredHost = SystemProfile(
            chip: "test",
            totalRAMBytes: 128 * gib,
            wiredLimitBytes: 96 * gib,
            wiredLimitIsMeasured: true,
            recommendedWorkingSetBytes: nil)
        let metalHost = SystemProfile(
            chip: "test",
            totalRAMBytes: 128 * gib,
            wiredLimitBytes: 96 * gib,
            wiredLimitIsMeasured: true,
            recommendedWorkingSetBytes: 24 * gib)
        let m = profile("Qwen3-32B")

        let wiredDecision = ServingFitPlanner.decide(
            profile: m, weightsAreMeasured: true, host: wiredHost, requestedContext: m.nativeMaxContext)
        let metalDecision = ServingFitPlanner.decide(
            profile: m, weightsAreMeasured: true, host: metalHost, requestedContext: m.nativeMaxContext)

        XCTAssertEqual(metalHost.effectiveMemoryCeiling.bytes, 24 * gib)
        XCTAssertEqual(metalHost.effectiveMemoryCeiling.source, .recommendedWorkingSet)
        XCTAssertNotEqual(metalHost.effectiveMemoryCeiling.source, .wiredLimit)
        XCTAssertNotEqual(wiredDecision.color, .red, "precondition: the model fits under the wired/shared fallback envelope")
        XCTAssertEqual(metalDecision.color, .red)
        XCTAssertEqual(metalDecision.bindingConstraint, .recommendedWorkingSet)
        XCTAssertEqual(metalDecision.memoryLimitBytes, 24 * gib)
        XCTAssertEqual(metalDecision.effectiveMemoryCeilingSource, .recommendedWorkingSet)
        XCTAssertFalse(metalDecision.effectiveMemoryCeilingIsMeasured)
        XCTAssertTrue(metalDecision.wiredLimitIsMeasured)
        XCTAssertTrue(metalDecision.machineReadableFields().contains("fit_estimate_measured=false"))
        XCTAssertEqual(
            metalDecision.cacheLimitBytes,
            CapacityModel.recommendedCacheLimitBytes(wiredLimitBytes: 24 * gib),
            "cache limit must derive from the same effective Metal ceiling as the memory limit")
    }

    func testLowerMeasuredWiredSharedCeilingDrivesPlannerMemoryAndCacheBudgets() {
        let host = SystemProfile(
            chip: "test",
            totalRAMBytes: 128 * gib,
            wiredLimitBytes: 48 * gib,
            wiredLimitIsMeasured: true,
            recommendedWorkingSetBytes: 80 * gib)

        let decision = ServingFitPlanner.decide(
            profile: profile("Qwen3-32B"), weightsAreMeasured: true, host: host)

        XCTAssertEqual(host.effectiveMemoryCeiling.source, .wiredLimit)
        XCTAssertEqual(decision.memoryLimitBytes, 48 * gib)
        XCTAssertEqual(
            decision.cacheLimitBytes,
            CapacityModel.recommendedCacheLimitBytes(wiredLimitBytes: 48 * gib))
        XCTAssertTrue(decision.wiredLimitIsMeasured)
        XCTAssertTrue(decision.effectiveMemoryCeilingIsMeasured)
        XCTAssertTrue(decision.machineReadableFields().contains("fit_estimate_measured=true"))
    }

    // MARK: - --tier dial ↔ serve decision composition (ServeTierPolicy → decide(allowContextCapping:))

    /// The end-to-end contract the serve route relies on when it wires `--tier` into `decide`: each
    /// tier's `ServingPolicy.allowContextCapping` drives a genuinely different serve outcome on the SAME
    /// memory-bound box. On the Phi-4/20-GiB fixture that auto-caps by default, `transparent` and
    /// `balanced` (no silent cap) REFUSE the full ask, while `maxfit` caps to the fitting ceiling and
    /// proceeds. This is what turns `--tier` from an inert validated flag into behavior on the actual
    /// serve path — the pick side already consumes the policy; this locks the serve side's use of it.
    func testServeTierPolicy_drivesContextCappingDecision() {
        let host = smallHost(ramGiB: 20, wiredGiB: 18)
        let p = profile("Phi-4-14B")
        // Precondition: this box caps the default context (the memory-bound case the dial governs).
        XCTAssertTrue(
            ServingFitPlanner.decide(profile: p, weightsAreMeasured: true, host: host).contextWasCapped,
            "precondition: default (cap-allowed) behavior caps here")

        for tier in [ServeTier.transparent, .balanced] {
            let policy = ServeTierPolicy.resolve(tier: tier)
            let d = ServingFitPlanner.decide(
                profile: p, weightsAreMeasured: true, host: host,
                allowContextCapping: policy.allowContextCapping)
            XCTAssertFalse(d.shouldProceed, "\(tier) forbids a silent cap → refuse the full ask")
            XCTAssertFalse(d.contextWasCapped, "\(tier) does not serve a shortened context")
            XCTAssertGreaterThan(d.contextCeiling, 0, "\(tier) still surfaces the fitting ceiling")
        }

        let maxfit = ServeTierPolicy.resolve(tier: .maxfit)
        let d = ServingFitPlanner.decide(
            profile: p, weightsAreMeasured: true, host: host,
            allowContextCapping: maxfit.allowContextCapping)
        XCTAssertTrue(d.shouldProceed, "maxfit caps to fit and proceeds")
        XCTAssertTrue(d.contextWasCapped)
        XCTAssertEqual(d.servedContext, d.contextCeiling, "maxfit serves exactly the fitting ceiling")
    }

    // MARK: - explicit --context beyond ceiling → refuse (honor the ask); --force serves capped

    func testExplicitContextBeyondCeiling_refuses_thenForceServesCapped() {
        let host = smallHost(ramGiB: 20, wiredGiB: 18)
        // Explicitly ask for the full 16K, which we know from the test above exceeds the ceiling.
        let refuse = ServingFitPlanner.decide(
            profile: profile("Phi-4-14B"), weightsAreMeasured: true, host: host, requestedContext: 16384)
        XCTAssertTrue(refuse.explicitContextRequested)
        XCTAssertEqual(refuse.color, .red, "the explicit ask lands red")
        XCTAssertFalse(refuse.shouldProceed, "explicit red ask fails closed")

        let forced = ServingFitPlanner.decide(
            profile: profile("Phi-4-14B"), weightsAreMeasured: true, host: host, requestedContext: 16384, force: true)
        XCTAssertTrue(forced.shouldProceed)
        XCTAssertTrue(forced.contextWasCapped)
        XCTAssertEqual(forced.servedContext, forced.contextCeiling)
    }

    // MARK: - requested context beyond the model's native max → red modelNativeMax

    func testRequestBeyondNativeMax_redModelNativeMax() {
        let host = qualifiedDedicatedHost
        // Phi-4 native max is 16384; ask for 32768.
        let d = ServingFitPlanner.decide(
            profile: profile("Phi-4-14B"), weightsAreMeasured: true, host: host, requestedContext: 32768)
        XCTAssertEqual(d.color, .red)
        XCTAssertEqual(d.bindingConstraint, .modelNativeMax)
        XCTAssertFalse(d.shouldProceed)
    }

    // MARK: - estimated weights provenance flows through

    func testEstimatedWeightsProvenanceSurfaced() {
        let host = qualifiedDedicatedHost
        let d = ServingFitPlanner.decide(profile: profile("Qwen3-32B"), weightsAreMeasured: false, host: host)
        XCTAssertFalse(d.weightsAreMeasured)
        // summary reflects the estimated weights label
        XCTAssertTrue(d.summaryLines().contains { $0.contains("estimated") })
    }

    // MARK: - machine-readable startup-line fields

    func testMachineReadableFields_greenMeasured_omitsForced() {
        let host = qualifiedDedicatedHost
        let d = ServingFitPlanner.decide(
            profile: profile("Qwen3-32B"), weightsAreMeasured: true, host: host)
        let fields = d.machineReadableFields()
        XCTAssertTrue(fields.contains("fit_check=green"))
        XCTAssertTrue(fields.contains("weights_measured=true"))
        XCTAssertTrue(fields.contains("fit_served_context="))
        XCTAssertTrue(fields.contains("fit_context_ceiling="))
        XCTAssertFalse(fields.contains("fit_forced"))
        // folded provenance: both weights and wired-limit measured → estimate is measured
        XCTAssertTrue(fields.contains("fit_estimate_measured=true"))
        // no --quant → none; default context on a box that holds native max → not capped
        XCTAssertTrue(fields.contains("fit_quant_bits=none"))
        XCTAssertTrue(fields.contains("fit_context_capped=false"))
        // listening= is spliced by the serve path AFTER these fields; nothing here may trail it.
        XCTAssertFalse(fields.contains("listening="))
    }

    func testMachineReadableFields_estimateNotMeasured_whenWeightsEstimated() {
        let host = qualifiedDedicatedHost // wired-limit measured, but weights passed as estimated
        let d = ServingFitPlanner.decide(
            profile: profile("Qwen3-32B"), weightsAreMeasured: false, host: host)
        let fields = d.machineReadableFields()
        XCTAssertTrue(fields.contains("weights_measured=false"))
        // folded field is the AND of weights + wired-limit provenance → false when weights estimated.
        XCTAssertTrue(fields.contains("fit_estimate_measured=false"))
    }

    func testMachineReadableFields_capped_reportsQuantBitsAndCapped() {
        // Phi-4 on this host fits at a smaller-than-native context → auto-capped default path
        // (mirrors testDefaultContext_autoCapsToCeiling_andProceeds).
        let host = smallHost(ramGiB: 20, wiredGiB: 18)
        let d = ServingFitPlanner.decide(
            profile: profile("Phi-4-14B"), weightsAreMeasured: true, host: host, quantBits: 4)
        let fields = d.machineReadableFields()
        XCTAssertTrue(d.contextWasCapped, "precondition: this host should auto-cap the default context")
        XCTAssertTrue(fields.contains("fit_context_capped=true"))
        XCTAssertTrue(fields.contains("fit_quant_bits=4"))
    }

    func testMachineReadableFields_redUnderForce_includesForced() {
        let host = smallHost(ramGiB: 8, wiredGiB: 6)
        let d = ServingFitPlanner.decide(
            profile: profile("GLM-4.5-Air"), weightsAreMeasured: true, host: host, force: true)
        let fields = d.machineReadableFields()
        XCTAssertTrue(fields.contains("fit_check=red"))
        XCTAssertTrue(fields.contains("fit_forced=true"))
    }

    // MARK: - quant-bits label in the summary header

    func testSummaryHeader_withQuantBits_showsBitLabel() {
        let host = qualifiedDedicatedHost
        let d = ServingFitPlanner.decide(
            profile: profile("Qwen3-32B"), weightsAreMeasured: true, host: host, quantBits: 4)
        XCTAssertTrue(d.summaryLines()[0].contains("[4-bit]"))
    }

    func testSummaryHeader_withoutQuantBits_unchanged() {
        let host = qualifiedDedicatedHost
        let d = ServingFitPlanner.decide(
            profile: profile("Qwen3-32B"), weightsAreMeasured: true, host: host)
        XCTAssertFalse(d.summaryLines()[0].contains("-bit]"))
    }

    // MARK: - weights provenance is three-state in the summary (measured / declared / estimated)

    /// The weights line distinguishes a *declared* size (from the checkpoint's index metadata) from a
    /// param-count *estimate* — measured still wins. Machine-readable `weights_measured=false` stays
    /// correct for declared (declared is not measured on disk).
    func testWeightsProvenance_threeStateLabel() {
        let host = qualifiedDedicatedHost
        let m = profile("Qwen3-32B")

        let measured = ServingFitPlanner.decide(profile: m, weightsAreMeasured: true, host: host)
        XCTAssertTrue(measured.summaryLines()[1].contains("(measured)"))

        let declared = ServingFitPlanner.decide(
            profile: m, weightsAreMeasured: false, host: host, weightsAreDeclared: true)
        XCTAssertTrue(declared.summaryLines()[1].contains("(declared)"),
            "an index-declared size must read (declared), not (estimated)")
        XCTAssertFalse(declared.summaryLines()[1].contains("(estimated)"))
        XCTAssertTrue(declared.machineReadableFields().contains("weights_measured=false"),
            "declared is still not measured on disk")

        let estimated = ServingFitPlanner.decide(
            profile: m, weightsAreMeasured: false, host: host, weightsAreDeclared: false)
        XCTAssertTrue(estimated.summaryLines()[1].contains("(estimated)"),
            "no measurement and no declared index → param-count estimate")
    }

    // MARK: - --plan-concurrency: opt-in concurrency-aware verdict (…-concurrency-kv-undercount.md,
    // option 2). Default concurrency=1 stays byte-identical to the shipped behavior; an explicit N>1
    // tightens the verdict (KV scales ×N), and the stricter basis is labeled so it is attributable.

    /// The default path is unchanged: computing at concurrency 1 (or passing it explicitly) is
    /// byte-for-byte identical to the shipped call, and emits NO planning-concurrency label. This is
    /// the non-regression lock that keeps the frozen announce contract intact for existing callers.
    func testPlanConcurrency_default1_isByteIdenticalAndUnlabeled() {
        let host = qualifiedDedicatedHost
        let m = profile("Qwen3-32B")
        let base = ServingFitPlanner.decide(profile: m, weightsAreMeasured: true, host: host)
        let explicit1 = ServingFitPlanner.decide(profile: m, weightsAreMeasured: true, host: host, concurrency: 1)
        XCTAssertEqual(explicit1.summaryLines(), base.summaryLines(),
            "concurrency:1 must render byte-identically to the shipped default")
        XCTAssertEqual(explicit1.machineReadableFields(), base.machineReadableFields())
        XCTAssertEqual(base.planningConcurrency, 1)
        XCTAssertFalse(base.machineReadableFields().contains("fit_plan_concurrency"),
            "the single-slot default must not emit a planning-concurrency field")
        XCTAssertFalse(base.summaryLines().contains { $0.contains("planning concurrency") },
            "the single-slot default must not render a planning-concurrency line")
    }

    /// An explicit N>1 tightens the served context: the ceiling shrinks because per-context KV scales
    /// with concurrency, so a box that serves a large context single-slot serves less under N streams.
    func testPlanConcurrency_higherN_tightensCeiling() {
        // Phi-4 auto-caps below its 16K native on this mid host at concurrency 1 (see the auto-cap
        // test); at concurrency 4 the ×4 KV forces a strictly smaller ceiling that still fits.
        let host = smallHost(ramGiB: 20, wiredGiB: 18)
        let m = profile("Phi-4-14B")
        let c1 = ServingFitPlanner.decide(profile: m, weightsAreMeasured: true, host: host, concurrency: 1)
        let c4 = ServingFitPlanner.decide(profile: m, weightsAreMeasured: true, host: host, concurrency: 4)
        XCTAssertGreaterThan(c1.contextCeiling, 0, "precondition: the model fits single-slot on this host")
        XCTAssertGreaterThan(c4.contextCeiling, 0, "precondition: the model still fits at concurrency 4 on this host")
        XCTAssertLessThan(c4.contextCeiling, c1.contextCeiling,
            "×4 concurrent KV must shrink the served-context ceiling")
        XCTAssertLessThanOrEqual(c4.servedContext, c1.servedContext)
    }

    /// A proceeding verdict computed at N>1 labels its basis: the summary carries a planning-concurrency
    /// line and the machine line appends `fit_plan_concurrency=N` (before any `fit_forced`), so a
    /// stricter verdict is never silently attributed to the single-slot default.
    func testPlanConcurrency_higherN_labelsBasis() {
        let host = qualifiedDedicatedHost // a 32B model on 128 GiB still proceeds at 4 streams
        let d = ServingFitPlanner.decide(profile: profile("Qwen3-32B"), weightsAreMeasured: true, host: host, concurrency: 4)
        XCTAssertTrue(d.shouldProceed, "precondition: this model proceeds at concurrency 4 on a 128 GiB box")
        XCTAssertEqual(d.planningConcurrency, 4)
        XCTAssertTrue(d.machineReadableFields().contains("fit_plan_concurrency=4"),
            "a >1 planning concurrency must be attributable on the machine line")
        XCTAssertTrue(d.summaryLines().contains { $0.contains("planning concurrency") },
            "the summary must label the stricter basis")
    }

    // MARK: - decode→decide seam: a real on-disk checkpoint drives the verdict end to end

    // The unit tests above feed `decide` a hand-built profile; the decoder tests decode a directory in
    // isolation. Neither exercises the SEAM the live serve path actually runs
    // (`ModelConfigDecoder.decodeModelDirectory` → `ServingFitPlanner.decide`). These two tests compose
    // them so the differentiator's headline contract — a real config.json + index.json is refused (or
    // served) on the detected host, with provenance carried through — is regression-locked off-box.
    // They are the pure-Swift proxy for the metallib-gated live refusal/green smokes.

    private let seamQwen3Config = """
    { "model_type": "qwen3", "num_hidden_layers": 48, "num_attention_heads": 40,
      "num_key_value_heads": 4, "head_dim": 128, "max_position_embeddings": 262144 }
    """

    private func writeFixtureDirectory(
        config: String, indexTotalSize: Int?, shardByteCounts: [Int], _ file: StaticString = #filePath,
        _ line: UInt = #line
    ) throws -> URL {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("fitseam-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(config.utf8).write(to: dir.appendingPathComponent("config.json"))
        for (i, count) in shardByteCounts.enumerated() {
            try Data(count: count).write(
                to: dir.appendingPathComponent(String(format: "model-%05d.safetensors", i + 1)))
        }
        if let total = indexTotalSize {
            let index = "{ \"metadata\": { \"total_size\": \(total) }, \"weight_map\": {} }"
            try Data(index.utf8).write(to: dir.appendingPathComponent("model.safetensors.index.json"))
        }
        return dir
    }

    /// The headline claim, end to end: a metadata-only checkpoint (config.json + index.json declaring
    /// ~40 GiB of weights, no shards on disk — an interrupted `snapshot_download` of a 70B-4bit model)
    /// decodes to a *declared* size and is REFUSED on a 24 GiB host. The provenance (`weightsAreDeclared`)
    /// must survive the seam into the decision so the refusal reads "(declared)", not a phantom measured fact.
    func testDecodeToDecideSeam_declaredOversizeCheckpoint_refusesOnSmallHost() throws {
        let fortyGiB = 40 * gib
        let dir = try writeFixtureDirectory(
            config: seamQwen3Config, indexTotalSize: fortyGiB, shardByteCounts: [])
        defer { try? FileManager.default.removeItem(at: dir) }

        let parsed = try ModelConfigDecoder.decodeModelDirectory(dir, id: "seam-oversize")
        XCTAssertFalse(parsed.weightsAreMeasured, "no shards on disk — nothing was measured")
        XCTAssertTrue(parsed.weightsAreDeclared, "size came from the index metadata")
        XCTAssertEqual(parsed.profile.weightsBytes4bitEstimate, fortyGiB)

        let host = smallHost(ramGiB: 24, wiredGiB: 20)
        let d = ServingFitPlanner.decide(
            profile: parsed.profile, weightsAreMeasured: parsed.weightsAreMeasured, host: host,
            weightsAreDeclared: parsed.weightsAreDeclared)
        XCTAssertEqual(d.color, .red, "40 GiB of weights cannot fit a 24 GiB box")
        XCTAssertFalse(d.shouldProceed, "red fails closed without --force")
        XCTAssertTrue([.physicalRAM, .wiredLimit].contains(d.bindingConstraint))
        XCTAssertEqual(d.contextCeiling, 0, "the weights alone overflow RAM — nothing fits at any context")
        // Provenance carried through the seam: the refusal is honest about being a declared (not measured) size.
        XCTAssertFalse(d.weightsAreMeasured)
        XCTAssertTrue(d.summaryLines()[1].contains("(declared)"),
            "a declared-size refusal must not masquerade as measured")
    }

    /// The happy path across the same seam: a complete small checkpoint (real on-disk shards) decodes to
    /// a *measured* size and is SERVED on a large host, with the measured provenance surfacing in the
    /// decision. Guards the seam against a regression that drops measured provenance or wrongly refuses a
    /// fitting model.
    func testDecodeToDecideSeam_measuredSmallCheckpoint_proceedsOnLargeHost() throws {
        let dir = try writeFixtureDirectory(
            config: seamQwen3Config, indexTotalSize: 1500, shardByteCounts: [1000, 500])
        defer { try? FileManager.default.removeItem(at: dir) }

        let parsed = try ModelConfigDecoder.decodeModelDirectory(dir, id: "seam-small")
        XCTAssertTrue(parsed.weightsAreMeasured, "on-disk shards were summed — a measured fact")
        XCTAssertFalse(parsed.weightsAreDeclared)

        let host = qualifiedDedicatedHost
        let d = ServingFitPlanner.decide(
            profile: parsed.profile, weightsAreMeasured: parsed.weightsAreMeasured, host: host,
            weightsAreDeclared: parsed.weightsAreDeclared)
        XCTAssertEqual(d.color, .green, "a ~1.5 KB fixture trivially fits a 128 GiB box")
        XCTAssertTrue(d.shouldProceed)
        XCTAssertTrue(d.weightsAreMeasured, "measured provenance must survive the seam")
        XCTAssertTrue(d.summaryLines()[1].contains("(measured)"))
    }
}
