import XCTest
@testable import HarnessCore
@testable import SystemProfiler

final class SystemProfilerTests: XCTestCase {
    private let gib = 1024 * 1024 * 1024

    func testDefaultSharedWiredLimitUsesExactIntegerPolicyForLargeInput() {
        let totalRAMBytes = 9_007_199_254_739_993

        XCTAssertEqual(
            SystemProfiler.defaultSharedWiredLimitBytes(totalRAMBytes: totalRAMBytes),
            6_755_399_441_054_994)
    }

    func testHostReportSystemProfilePreservesSynthesizedLimitAsUnmeasured() {
        let report = HostReport(
            chip: "Apple test",
            totalRAMBytes: 128 * gib,
            wiredLimitBytes: 96 * gib,
            wiredLimitIsDefault: true,
            pCores: 8,
            eCores: 4,
            currentGPUAllocBytes: nil,
            recommendedWorkingSetBytes: nil,
            diskInternal: nil,
            diskFreeBytes: nil,
            hostUse: .automaticShared
        )

        let profile = report.systemProfile

        XCTAssertEqual(profile.wiredLimitBytes, 96 * gib)
        XCTAssertFalse(profile.wiredLimitIsMeasured)
        XCTAssertEqual(profile.hostUse.rawValue, "shared")
        XCTAssertEqual(profile.hostUse.source.rawValue, "automatic")
        XCTAssertEqual(profile.hostUse.policyVersion, HostUseClassification.currentPolicyVersion)
    }

    func testHostReportSystemProfilePreservesMeasuredLimitAndHostUseAssertion() {
        let hostUse = HostUseClassification.operatorAssertedDedicatedServing()
        let report = HostReport(
            chip: "Apple test",
            totalRAMBytes: 128 * gib,
            wiredLimitBytes: 64 * gib,
            wiredLimitIsDefault: false,
            pCores: 8,
            eCores: 4,
            currentGPUAllocBytes: nil,
            recommendedWorkingSetBytes: nil,
            diskInternal: nil,
            diskFreeBytes: nil,
            hostUse: hostUse
        )

        let profile = report.systemProfile

        XCTAssertEqual(profile.wiredLimitBytes, 64 * gib)
        XCTAssertTrue(profile.wiredLimitIsMeasured)
        XCTAssertEqual(profile.hostUse, hostUse)
    }

    func testHostReportSystemProfilePassesRecommendedWorkingSetThroughAndPreservesWiredProvenance() {
        let report = HostReport(
            chip: "Apple test",
            totalRAMBytes: 128 * gib,
            wiredLimitBytes: 96 * gib,
            wiredLimitIsDefault: false,
            pCores: 8,
            eCores: 4,
            currentGPUAllocBytes: nil,
            recommendedWorkingSetBytes: 40 * gib,
            diskInternal: nil,
            diskFreeBytes: nil,
            hostUse: .automaticShared
        )

        let profile = report.systemProfile

        XCTAssertEqual(profile.wiredLimitBytes, 96 * gib)
        XCTAssertTrue(profile.wiredLimitIsMeasured)
        XCTAssertEqual(profile.recommendedWorkingSetBytes, 40 * gib)
        XCTAssertEqual(profile.effectiveMemoryCeiling.bytes, 40 * gib)
        XCTAssertEqual(profile.effectiveMemoryCeiling.source, .recommendedWorkingSet)
        XCTAssertEqual(profile.hostUse.rawValue, "shared")
        XCTAssertEqual(profile.hostUse.source.rawValue, "automatic")
    }

    func testHostReportAppliesExplicitHostUseWithoutLosingOneProbeSnapshot() {
        let gib = 1_073_741_824
        let report = HostReport(
            chip: "Apple Test",
            totalRAMBytes: 24 * gib,
            wiredLimitBytes: 18 * gib,
            wiredLimitIsDefault: true,
            pCores: 4,
            eCores: 6,
            currentGPUAllocBytes: 2 * gib,
            recommendedWorkingSetBytes: 16 * gib,
            diskInternal: true,
            diskFreeBytes: 100 * gib,
            hostUse: .automaticShared)

        let selected = report.applyingHostUse(.operatorAssertedDedicatedServing())

        XCTAssertEqual(selected.chip, report.chip)
        XCTAssertEqual(selected.totalRAMBytes, report.totalRAMBytes)
        XCTAssertEqual(selected.wiredLimitBytes, report.wiredLimitBytes)
        XCTAssertEqual(selected.wiredLimitIsDefault, report.wiredLimitIsDefault)
        XCTAssertEqual(selected.pCores, report.pCores)
        XCTAssertEqual(selected.eCores, report.eCores)
        XCTAssertEqual(selected.currentGPUAllocBytes, report.currentGPUAllocBytes)
        XCTAssertEqual(selected.recommendedWorkingSetBytes, report.recommendedWorkingSetBytes)
        XCTAssertEqual(selected.diskInternal, report.diskInternal)
        XCTAssertEqual(selected.diskFreeBytes, report.diskFreeBytes)
        XCTAssertEqual(selected.hostUse.rawValue, "dedicated-serving")
        XCTAssertEqual(selected.hostUse.source.rawValue, "operator-assertion")

        let explicitShared = report.applyingHostUse(.operatorAssertedShared())
        XCTAssertEqual(explicitShared.hostUse.rawValue, "shared")
        XCTAssertEqual(explicitShared.hostUse.source.rawValue, "operator-assertion")
    }

    func testHostReportMachineFieldsPreserveClassificationObservationsAndEffectiveCeiling() {
        let gib = 1_073_741_824
        let report = HostReport(
            chip: "Apple Test",
            totalRAMBytes: 24 * gib,
            wiredLimitBytes: 18 * gib,
            wiredLimitIsDefault: true,
            pCores: 4,
            eCores: 6,
            currentGPUAllocBytes: 2 * gib,
            recommendedWorkingSetBytes: 16 * gib,
            diskInternal: true,
            diskFreeBytes: 100 * gib,
            hostUse: .automaticShared)

        let expectedFields = [
            "host_use=shared",
            "host_use_source=automatic",
            "host_use_policy_version=host-use/v1",
            "host_physical_ram_bytes=25769803776",
            "host_wired_limit_bytes=19327352832",
            "host_wired_limit_provenance=synthesized",
            "host_metal_recommended_working_set_bytes=17179869184",
            "host_metal_current_allocated_bytes=2147483648",
            "host_effective_memory_ceiling_bytes=17179869184",
            "host_effective_memory_ceiling_source=metal-recommended-working-set",
        ].joined(separator: " ")
        XCTAssertEqual(report.machineReadableMemoryFields(), expectedFields)
    }

    func testHostReportMachineFieldsRenderUnavailableMetalObservationsExplicitly() {
        let gib = 1_073_741_824
        let report = HostReport(
            chip: "Apple Test",
            totalRAMBytes: 24 * gib,
            wiredLimitBytes: 18 * gib,
            wiredLimitIsDefault: false,
            pCores: 4,
            eCores: 6,
            currentGPUAllocBytes: nil,
            recommendedWorkingSetBytes: nil,
            diskInternal: nil,
            diskFreeBytes: nil,
            hostUse: .operatorAssertedDedicatedServing())

        let fields = report.machineReadableMemoryFields()
        XCTAssertTrue(fields.contains("host_use=dedicated-serving"))
        XCTAssertTrue(fields.contains("host_use_source=operator-assertion"))
        XCTAssertTrue(fields.contains("host_wired_limit_provenance=measured"))
        XCTAssertTrue(fields.contains("host_metal_recommended_working_set_bytes=unavailable"))
        XCTAssertTrue(fields.contains("host_metal_current_allocated_bytes=unavailable"))
        XCTAssertTrue(fields.contains("host_effective_memory_ceiling_source=wired-limit"))
    }

    func testHostReportServingFieldsAppendFinalAllocatorAndKVBudgets() {
        let gib = 1_073_741_824
        let report = HostReport(
            chip: "Apple Test",
            totalRAMBytes: 24 * gib,
            wiredLimitBytes: 18 * gib,
            wiredLimitIsDefault: false,
            pCores: 4,
            eCores: 6,
            currentGPUAllocBytes: 2 * gib,
            recommendedWorkingSetBytes: 16 * gib,
            diskInternal: true,
            diskFreeBytes: 100 * gib,
            hostUse: .automaticShared)

        let loaded = report.machineReadableServingFields(
            memoryLimitBytes: 16 * gib,
            cacheLimitBytes: 4 * gib,
            kvBudgetBytes: 3 * gib,
            osServiceReserveBytes: 5 * gib)
        XCTAssertTrue(loaded.hasSuffix(
            "host_os_service_reserve_bytes=5368709120 "
                + "mlx_memory_limit_bytes=17179869184 "
                + "mlx_cache_limit_bytes=4294967296 "
                + "mlx_kv_budget_bytes=3221225472"))

        let scalar = report.machineReadableServingFields(
            memoryLimitBytes: 16 * gib,
            cacheLimitBytes: 4 * gib,
            kvBudgetBytes: nil,
            osServiceReserveBytes: 5 * gib)
        XCTAssertTrue(scalar.hasSuffix(
            "host_os_service_reserve_bytes=5368709120 "
                + "mlx_memory_limit_bytes=17179869184 "
                + "mlx_cache_limit_bytes=4294967296 "
                + "mlx_kv_budget_bytes=not-separately-reserved"))

        let transportOnly = report.machineReadableServingFields(
            memoryLimitBytes: nil,
            cacheLimitBytes: nil,
            kvBudgetBytes: nil,
            osServiceReserveBytes: 5 * gib)
        XCTAssertTrue(transportOnly.hasSuffix(
            "host_os_service_reserve_bytes=5368709120 "
                + "mlx_memory_limit_bytes=not-applicable "
                + "mlx_cache_limit_bytes=not-applicable "
                + "mlx_kv_budget_bytes=not-applicable"))
    }

    // MARK: - operator-budget-envelope-shape decision: the `--host-use` silent-drop guard.
    //
    // The decision's "single most likely silent defect" is a budget stored ON `HostReport` being
    // dropped by `applyingHostUse` (which re-lists every field explicitly) whenever `--host-use` is
    // passed — which the live production serve does. The shipped fix keeps the budget OFF
    // `HostReport` entirely and applies it only to the `SystemProfile` derived AFTER `applyingHostUse`
    // has already resolved the classification — exactly the composition
    // `report.applyingHostUse(explicit).systemProfile.withOperatorMemoryBudget(bytes)` a real
    // `--host-use`-carrying serve invocation performs. This test locks that composition, not merely
    // the two halves in isolation: it is mutation-checked below by proving a dropped `--host-use`
    // classification (or a dropped budget) produces an observably different, WRONG ceiling — so a
    // regression that silently discards either one fails this test rather than passing by
    // coincidence.
    func testOperatorBudgetSurvivesAcrossAnExplicitHostUseClassification() {
        let gib = 1_073_741_824
        // Chosen so shared vs. dedicated-serving classification, and budget-binds vs.
        // budget-inert, are all four distinguishable outcomes:
        //   shared ceiling      = floor(0.75 * 256 GiB) = 192 GiB (wired 220 GiB does not undercut it)
        //   dedicated ceiling   = min(wired, RAM)        = 220 GiB (wired < RAM)
        //   operator budget     = 200 GiB (binds under dedicated: 200 < 220; inert under shared: 200 !< 192)
        let report = HostReport(
            chip: "Apple Test",
            totalRAMBytes: 256 * gib,
            wiredLimitBytes: 220 * gib,
            wiredLimitIsDefault: false,
            pCores: 16,
            eCores: 8,
            currentGPUAllocBytes: nil,
            recommendedWorkingSetBytes: nil,
            diskInternal: nil,
            diskFreeBytes: nil,
            hostUse: .automaticShared)
        let budgetBytes = 200 * gib

        // The seam under test: --host-use dedicated-serving is applied to the ONE probed snapshot
        // BEFORE the operator's memory budget is layered onto the derived profile.
        let dedicatedThenBudgeted = report
            .applyingHostUse(.operatorAssertedDedicatedServing())
            .systemProfile
            .withOperatorMemoryBudget(budgetBytes)

        XCTAssertEqual(dedicatedThenBudgeted.hostUse.rawValue, "dedicated-serving",
            "the --host-use classification must survive into the budgeted profile")
        XCTAssertEqual(dedicatedThenBudgeted.hostCeilingBeforeOperatorBudget.bytes, 220 * gib,
            "precondition: dedicated-serving ceiling is min(wired, RAM) = 220 GiB")
        XCTAssertEqual(dedicatedThenBudgeted.effectiveMemoryCeiling.bytes, budgetBytes,
            "the operator budget must bind against the DEDICATED ceiling, not a dropped/default one")
        XCTAssertEqual(dedicatedThenBudgeted.effectiveMemoryCeiling.source, .operatorBudget)

        // Mutation check #1: if the --host-use classification were silently dropped (the profile
        // stayed on the default `.automaticShared` classification the raw probe returns), the SAME
        // budget would be inert against the shared ceiling — a materially different, wrong outcome
        // this test would catch.
        let sharedThenBudgeted = report.systemProfile.withOperatorMemoryBudget(budgetBytes)
        XCTAssertEqual(sharedThenBudgeted.hostCeilingBeforeOperatorBudget.bytes, 192 * gib,
            "precondition: shared-policy ceiling is floor(0.75 * 256 GiB) = 192 GiB")
        XCTAssertEqual(sharedThenBudgeted.effectiveMemoryCeiling.bytes, 192 * gib,
            "precondition: 200 GiB budget is NOT tighter than 192 GiB, so it must stay inert")
        XCTAssertEqual(sharedThenBudgeted.effectiveMemoryCeiling.source, .sharedPolicy)
        XCTAssertNotEqual(
            dedicatedThenBudgeted.effectiveMemoryCeiling.bytes,
            sharedThenBudgeted.effectiveMemoryCeiling.bytes,
            "dropping --host-use must be observable: the dedicated and (wrongly) shared outcomes differ")

        // Mutation check #2: if the budget itself were dropped (the field never threaded through —
        // the exact hazard the decision calls out for a budget stored ON `HostReport`), the dedicated
        // classification alone would report its own unbudgeted ceiling, not the operator's 200 GiB.
        let dedicatedWithoutBudget = report
            .applyingHostUse(.operatorAssertedDedicatedServing())
            .systemProfile
        XCTAssertEqual(dedicatedWithoutBudget.effectiveMemoryCeiling.bytes, 220 * gib)
        XCTAssertEqual(dedicatedWithoutBudget.effectiveMemoryCeiling.source, .wiredLimit)
        XCTAssertNotEqual(
            dedicatedWithoutBudget.effectiveMemoryCeiling.bytes,
            dedicatedThenBudgeted.effectiveMemoryCeiling.bytes,
            "dropping the budget must be observable: the unbudgeted dedicated ceiling differs from the budgeted one")
    }

    /// The version-skew safety property: an existing deployment can already be passing
    /// `--memory-limit-bytes` EQUAL to its own shared ceiling, because the flag used to be required
    /// and callers supplied the same figure the sizer would have derived. The binding rule is a
    /// strict `<`, so an equal budget must be inert — that is what makes a stale, byte-identical
    /// argv safe across a binary upgrade that makes the flag load-bearing for the first time.
    func testOperatorBudgetExactlyEqualToHostCeilingIsInertNotBinding() {
        let gib = 1_073_741_824
        let report = HostReport(
            chip: "Apple Test",
            totalRAMBytes: 137_438_953_472, // 128 GiB
            wiredLimitBytes: 115 * gib, // measured, but shared policy caps below it on this box
            wiredLimitIsDefault: false,
            pCores: 16,
            eCores: 8,
            currentGPUAllocBytes: nil,
            recommendedWorkingSetBytes: nil,
            diskInternal: nil,
            diskFreeBytes: nil,
            hostUse: .automaticShared)

        // Shared ceiling = floor(0.75 * 137,438,953,472) = 103,079,215,104 — the exact bytes the
        // deployed argv supplies as --memory-limit-bytes.
        let sharedCeiling = report.systemProfile.hostCeilingBeforeOperatorBudget
        XCTAssertEqual(sharedCeiling.bytes, 103_079_215_104, "precondition: matches the deployed argv")

        let budgeted = report
            .applyingHostUse(.operatorAssertedShared())
            .systemProfile
            .withOperatorMemoryBudget(103_079_215_104)

        XCTAssertEqual(budgeted.effectiveMemoryCeiling.bytes, sharedCeiling.bytes,
            "an equal budget must not change the effective ceiling")
        XCTAssertEqual(budgeted.effectiveMemoryCeiling.source, sharedCeiling.source,
            "an equal budget must not relabel the ceiling's source as operator-budget")
        XCTAssertNotEqual(budgeted.effectiveMemoryCeiling.source, .operatorBudget)
    }
}
