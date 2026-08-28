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
}
