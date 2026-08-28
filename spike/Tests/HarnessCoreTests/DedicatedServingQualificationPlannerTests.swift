import Foundation
import XCTest
@testable import HarnessCore

final class DedicatedServingQualificationPlannerTests: XCTestCase {
    private let gib = 1024 * 1024 * 1024

    private func validInput(
        candidateGiB: Int = 224,
        osReserveGiB: Int = 16,
        memoryGiB: Int = 200,
        cacheGiB: Int = 24,
        kvGiB: Int = 160,
        ioGiB: Int = 8
    ) -> DedicatedServingQualificationInput {
        DedicatedServingQualificationInput(
            host: DedicatedServingQualificationHost(
                hostUse: "dedicated-serving",
                hostUseSource: "operator-assertion",
                hostUsePolicyVersion: "host-use/v1",
                physicalRAMBytes: 256 * gib,
                originalWiredLimitBytes: 216 * gib,
                originalWiredLimitProvenance: .measured,
                metalRecommendedWorkingSetBytes: 220 * gib,
                metalCurrentAllocatedBytes: 12 * gib,
                osBuild: "Version 26.6.2 (Build 25G83)",
                osBuildSource: DedicatedServingQualificationHost.processInfoOSBuildSource
            ),
            osServiceReserveBytes: osReserveGiB * gib,
            candidateCeilingMiB: candidateGiB * 1024,
            stagedFollowOnCandidateMiB: [232 * 1024, 240 * 1024],
            mlxBudgets: DedicatedServingQualificationBudgets(
                memoryLimitBytes: memoryGiB * gib,
                cacheLimitBytes: cacheGiB * gib,
                kvBudgetBytes: kvGiB * gib,
                ioPrefetchBudgetBytes: ioGiB * gib
            )
        )
    }

    func testHappyPathBuildsAuditableDryRunArtifactAndRoundTrips() throws {
        let artifact = try DedicatedServingQualificationPlanner.plan(validInput())

        XCTAssertEqual(artifact.schemaVersion, "dedicated-serving-qualification-plan/v1")
        XCTAssertTrue(artifact.dryRunOnly)
        XCTAssertFalse(artifact.authoritative)
        XCTAssertFalse(artifact.executionAuthorized)
        XCTAssertEqual(artifact.host.hostUse, "dedicated-serving")
        XCTAssertEqual(artifact.host.hostUseSource, "operator-assertion")
        XCTAssertEqual(artifact.host.hostUsePolicyVersion, "host-use/v1")
        XCTAssertEqual(artifact.host.physicalRAMBytes, 256 * gib)
        XCTAssertEqual(artifact.host.originalWiredLimitBytes, 216 * gib)
        XCTAssertEqual(artifact.host.originalWiredLimitProvenance, .measured)
        XCTAssertEqual(artifact.host.metalRecommendedWorkingSetBytes, 220 * gib)
        XCTAssertEqual(artifact.host.metalCurrentAllocatedBytes, 12 * gib)
        XCTAssertEqual(artifact.host.osBuild, "Version 26.6.2 (Build 25G83)")
        XCTAssertEqual(artifact.host.osBuildSource, "process-info-operating-system-version")
        XCTAssertEqual(artifact.osServiceReserveBytes, 16 * gib)
        XCTAssertEqual(artifact.proposedCandidateCeiling.mib, 224 * 1024)
        XCTAssertEqual(artifact.proposedCandidateCeiling.bytes, 224 * gib)
        XCTAssertEqual(artifact.stagedFollowOnCandidates.map(\.mib), [232 * 1024, 240 * 1024])
        XCTAssertEqual(artifact.stagedFollowOnCandidates.map(\.bytes), [232 * gib, 240 * gib])
        XCTAssertEqual(artifact.mlxBudgets.memoryLimitBytes, 200 * gib)
        XCTAssertEqual(artifact.mlxBudgets.cacheLimitBytes, 24 * gib)
        XCTAssertEqual(artifact.mlxBudgets.kvBudgetBytes, 160 * gib)
        XCTAssertEqual(artifact.mlxBudgets.ioPrefetchBudgetBytes, 8 * gib)
        XCTAssertEqual(artifact.budgetReconciliation.memoryAvailableAfterReserveBytes, 208 * gib)
        XCTAssertEqual(artifact.budgetReconciliation.reserveSubtractedExactlyOnce, true)
        XCTAssertEqual(artifact.budgetReconciliation.mlxAllocatorLimitFitsCandidateMinusReserve, true)
        XCTAssertEqual(artifact.budgetReconciliation.cacheAndKVFitMLXAllocatorLimit, true)
        XCTAssertEqual(artifact.budgetReconciliation.mlxAllocatorAndIOFitCandidateMinusReserve, true)

        XCTAssertEqual(
            artifact.stages.map(\.id),
            [
                .captureOriginal,
                .temporaryApply,
                .readbackRecommendedWorkingSetCheck,
                .safetyHealthSoakGates,
                .restoreOnAnyFailure,
            ])
        XCTAssertTrue(artifact.stages.allSatisfy(\.dryRunOnly))
        XCTAssertTrue(artifact.stages[0].gates.contains(.osBuildCaptured))
        XCTAssertTrue(artifact.stages[0].gates.contains(.physicalRAMCaptured))
        XCTAssertTrue(artifact.stages[0].gates.contains(.originalWiredLimitCaptured))
        XCTAssertTrue(artifact.stages[0].gates.contains(.metalRecommendedWorkingSetCaptured))
        XCTAssertTrue(artifact.stages[0].gates.contains(.metalCurrentAllocationCaptured))
        XCTAssertTrue(artifact.stages[0].gates.contains(.swapPageoutBaselineCaptured))
        XCTAssertTrue(artifact.stages[0].gates.contains(.memoryPressureBaselineCaptured))
        XCTAssertTrue(artifact.stages[0].gates.contains(.runningServicesBaselineCaptured))
        XCTAssertTrue(artifact.stages[3].gates.contains(.swapAndPageoutsStable))
        XCTAssertTrue(artifact.stages[3].gates.contains(.memoryPressureNormal))
        XCTAssertTrue(artifact.stages[3].gates.contains(.noOOMOrSIGKILL))
        XCTAssertTrue(artifact.stages[3].gates.contains(.responsiveHealth))
        XCTAssertTrue(artifact.stages[3].gates.contains(.boundedAllocationLatencyThermal))

        let json = try artifact.encodedJSON()
        XCTAssertEqual(json, try artifact.encodedJSON(), "encoding should be deterministic for audit diffs")
        let decoded = try JSONDecoder().decode(DedicatedServingQualificationArtifact.self, from: Data(json.utf8))
        XCTAssertEqual(decoded, artifact)
    }

    func testSharedDefaultOrAmbiguousHostUseFailsClosed() {
        let cases = [
            DedicatedServingQualificationHost(
                hostUse: "shared", hostUseSource: "default", hostUsePolicyVersion: "host-use/v1",
                physicalRAMBytes: 256 * gib, originalWiredLimitBytes: 216 * gib,
                originalWiredLimitProvenance: .measured, metalRecommendedWorkingSetBytes: 220 * gib,
                metalCurrentAllocatedBytes: 0, osBuild: "Version 26.6.2 (Build 25G83)",
                osBuildSource: DedicatedServingQualificationHost.processInfoOSBuildSource),
            DedicatedServingQualificationHost(
                hostUse: "dedicated-serving", hostUseSource: "automatic", hostUsePolicyVersion: "host-use/v1",
                physicalRAMBytes: 256 * gib, originalWiredLimitBytes: 216 * gib,
                originalWiredLimitProvenance: .measured, metalRecommendedWorkingSetBytes: 220 * gib,
                metalCurrentAllocatedBytes: 0, osBuild: "Version 26.6.2 (Build 25G83)",
                osBuildSource: DedicatedServingQualificationHost.processInfoOSBuildSource),
            DedicatedServingQualificationHost(
                hostUse: "dedicated-serving", hostUseSource: "operator-assertion", hostUsePolicyVersion: "host-use/v0",
                physicalRAMBytes: 256 * gib, originalWiredLimitBytes: 216 * gib,
                originalWiredLimitProvenance: .measured, metalRecommendedWorkingSetBytes: 220 * gib,
                metalCurrentAllocatedBytes: 0, osBuild: "Version 26.6.2 (Build 25G83)",
                osBuildSource: DedicatedServingQualificationHost.processInfoOSBuildSource),
        ]

        for host in cases {
            var input = validInput()
            input.host = host
            XCTAssertThrowsError(try DedicatedServingQualificationPlanner.plan(input))
        }
    }

    func testUnmeasuredOriginalWiredLimitFailsClosed() {
        var input = validInput()
        input.host.originalWiredLimitProvenance = .synthesized

        XCTAssertThrowsError(try DedicatedServingQualificationPlanner.plan(input))
    }

    func testCandidateMustBeBetweenMeasuredOriginalAndPhysicalRAM() {
        var notRaised = validInput(candidateGiB: 216)
        notRaised.stagedFollowOnCandidateMiB = [224 * 1024, 232 * 1024, 240 * 1024]
        XCTAssertThrowsError(try DedicatedServingQualificationPlanner.plan(notRaised))

        var fullRAM = validInput(candidateGiB: 256)
        fullRAM.stagedFollowOnCandidateMiB = []
        XCTAssertThrowsError(try DedicatedServingQualificationPlanner.plan(fullRAM))

        var overflow = validInput()
        overflow.candidateCeilingMiB = Int.max
        XCTAssertThrowsError(try DedicatedServingQualificationPlanner.plan(overflow))
    }

    func testReserveOSBuildAndBudgetsFailClosedWhenMissingOrNonPositive() {
        var missingOS = validInput()
        missingOS.host.osBuild = "   "
        XCTAssertThrowsError(try DedicatedServingQualificationPlanner.plan(missingOS))

        var wrongOSSource = validInput()
        wrongOSSource.host.osBuildSource = "operator-input"
        XCTAssertThrowsError(try DedicatedServingQualificationPlanner.plan(wrongOSSource))

        var blankOSSource = validInput()
        blankOSSource.host.osBuildSource = ""
        XCTAssertThrowsError(try DedicatedServingQualificationPlanner.plan(blankOSSource))

        let zeroReserve = validInput(osReserveGiB: 0)
        XCTAssertThrowsError(try DedicatedServingQualificationPlanner.plan(zeroReserve))

        let zeroMemory = validInput(memoryGiB: 0)
        XCTAssertThrowsError(try DedicatedServingQualificationPlanner.plan(zeroMemory))

        let zeroCache = validInput(cacheGiB: 0)
        XCTAssertThrowsError(try DedicatedServingQualificationPlanner.plan(zeroCache))

        let zeroKV = validInput(kvGiB: 0)
        XCTAssertThrowsError(try DedicatedServingQualificationPlanner.plan(zeroKV))

        let zeroIO = validInput(ioGiB: 0)
        XCTAssertThrowsError(try DedicatedServingQualificationPlanner.plan(zeroIO))
    }

    func testOvercommitAndCacheBudgetFailuresFailClosedWithoutDoubleCountingReserve() {
        let memoryOverCandidateMinusReserve = validInput(memoryGiB: 209)
        XCTAssertThrowsError(try DedicatedServingQualificationPlanner.plan(memoryOverCandidateMinusReserve))

        let cacheEqualsMemory = validInput(memoryGiB: 24, cacheGiB: 24)
        XCTAssertThrowsError(try DedicatedServingQualificationPlanner.plan(cacheEqualsMemory))

        let cachePlusKVOutsideMemory = validInput(memoryGiB: 180, cacheGiB: 24, kvGiB: 160)
        XCTAssertThrowsError(try DedicatedServingQualificationPlanner.plan(cachePlusKVOutsideMemory))

        let memoryPlusIOOverCandidateMinusReserve = validInput(memoryGiB: 201, ioGiB: 8)
        XCTAssertThrowsError(try DedicatedServingQualificationPlanner.plan(memoryPlusIOOverCandidateMinusReserve))
    }

    func testMetalObservationsAreRequiredAndValidated() throws {
        var negativeRecommended = validInput()
        negativeRecommended.host.metalRecommendedWorkingSetBytes = -1
        XCTAssertThrowsError(try DedicatedServingQualificationPlanner.plan(negativeRecommended))

        var negativeCurrent = validInput()
        negativeCurrent.host.metalCurrentAllocatedBytes = -1
        XCTAssertThrowsError(try DedicatedServingQualificationPlanner.plan(negativeCurrent))

        let artifact = try DedicatedServingQualificationPlanner.plan(validInput())
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try artifact.encodedJSON().utf8)) as? [String: Any])
        var host = try XCTUnwrap(object["host"] as? [String: Any])
        host.removeValue(forKey: "metalRecommendedWorkingSetBytes")
        object["host"] = host
        XCTAssertThrowsError(try decodeArtifact(object))

        host = try XCTUnwrap(object["host"] as? [String: Any])
        host["metalRecommendedWorkingSetBytes"] = 220 * gib
        host.removeValue(forKey: "metalCurrentAllocatedBytes")
        object["host"] = host
        XCTAssertThrowsError(try decodeArtifact(object))
    }

    func testCandidateLadderMustBeExactRemainingSuffix() {
        XCTAssertNoThrow(try DedicatedServingQualificationPlanner.plan(validInput(candidateGiB: 224)))

        var candidate232 = validInput(candidateGiB: 232)
        candidate232.stagedFollowOnCandidateMiB = [240 * 1024]
        XCTAssertNoThrow(try DedicatedServingQualificationPlanner.plan(candidate232))

        var candidate240 = validInput(candidateGiB: 240)
        candidate240.stagedFollowOnCandidateMiB = []
        XCTAssertNoThrow(try DedicatedServingQualificationPlanner.plan(candidate240))

        var emptyFrom224 = validInput(candidateGiB: 224)
        emptyFrom224.stagedFollowOnCandidateMiB = []
        XCTAssertThrowsError(try DedicatedServingQualificationPlanner.plan(emptyFrom224))

        var duplicate = validInput(candidateGiB: 224)
        duplicate.stagedFollowOnCandidateMiB = [232 * 1024, 232 * 1024, 240 * 1024]
        XCTAssertThrowsError(try DedicatedServingQualificationPlanner.plan(duplicate))

        var skipped = validInput(candidateGiB: 224)
        skipped.stagedFollowOnCandidateMiB = [240 * 1024]
        XCTAssertThrowsError(try DedicatedServingQualificationPlanner.plan(skipped))

        var descending = validInput(candidateGiB: 232)
        descending.stagedFollowOnCandidateMiB = [224 * 1024, 240 * 1024]
        XCTAssertThrowsError(try DedicatedServingQualificationPlanner.plan(descending))

        var beyondLadder = validInput(candidateGiB: 248)
        beyondLadder.stagedFollowOnCandidateMiB = []
        XCTAssertThrowsError(try DedicatedServingQualificationPlanner.plan(beyondLadder))
    }

    func testDecodeRejectsAuthoritativeOrForeignSchemaArtifact() throws {
        let artifact = try DedicatedServingQualificationPlanner.plan(validInput())
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try artifact.encodedJSON().utf8)) as? [String: Any])

        object["schemaVersion"] = "dedicated-serving-qualification-plan/v2"
        XCTAssertThrowsError(try decodeArtifact(object))

        object["schemaVersion"] = "dedicated-serving-qualification-plan/v1"
        object["executionAuthorized"] = true
        XCTAssertThrowsError(try decodeArtifact(object))

        object["executionAuthorized"] = false
        object["authoritative"] = true
        XCTAssertThrowsError(try decodeArtifact(object))

        object["authoritative"] = false
        object["dryRunOnly"] = false
        XCTAssertThrowsError(try decodeArtifact(object))
    }

    func testDecodeRejectsTamperedPlannerDerivedPayloads() throws {
        let artifact = try DedicatedServingQualificationPlanner.plan(validInput())
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try artifact.encodedJSON().utf8)) as? [String: Any])
        var proposed = try XCTUnwrap(object["proposedCandidateCeiling"] as? [String: Any])
        proposed["bytes"] = 225 * gib
        object["proposedCandidateCeiling"] = proposed
        XCTAssertThrowsError(try decodeArtifact(object))

        object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try artifact.encodedJSON().utf8)) as? [String: Any])
        var host = try XCTUnwrap(object["host"] as? [String: Any])
        host["osBuildSource"] = "operator-input"
        object["host"] = host
        XCTAssertThrowsError(try decodeArtifact(object))
    }

    private func decodeArtifact(_ object: [String: Any]) throws -> DedicatedServingQualificationArtifact {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        return try JSONDecoder().decode(DedicatedServingQualificationArtifact.self, from: data)
    }
}
