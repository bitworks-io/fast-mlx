import Foundation
import HarnessCore
import XCTest

import ScorecardPairControl

/// Golden test for the argv builder shared by the live adapter and the (future)
/// proof runner: both must launch the self-exec worker subcommand with
/// byte-identical arguments, so the exact list, order, and formatting here is
/// load-bearing, not incidental.
final class Qwen38MTPScorecardWorkerLaunchArgumentsTests: XCTestCase {
    func testLaunchArgumentsMatchExpectedFlagOrderAndFormatting() throws {
        let arguments = Qwen38MTPScorecardLiveRunArguments(
            targetPath: "/models/target",
            drafterPath: "/models/drafter",
            outputPath: "/tmp/scorecard.jsonl",
            authorityOutputPath: "/tmp/authority.json",
            hostUse: "dedicated-serving",
            hostUseSource: "operator-assertion",
            expectedChip: "Apple M3 Ultra",
            memoryBudget: try Qwen38MTPScorecardLiveMemoryBudget(
                memoryLimitBytes: 236_223_201_280,
                cacheLimitBytes: 51_539_607_552,
                reservedKVBytes: 42_949_672_960,
                reservedIOBytes: 2_147_483_648,
                reservedPrefetchBytes: 4_294_967_296,
                osServiceReserveBytes: 8_589_934_592))

        XCTAssertEqual(
            qwen38MTPScorecardWorkerLaunchArguments(role: .candidate, arguments: arguments),
            [
                "qwen38-mtp-scorecard-worker",
                "--role", "candidate",
                "--target", "/models/target",
                "--drafter", "/models/drafter",
                "--host-use", "dedicated-serving",
                "--host-use-source", "operator-assertion",
                "--memory-limit-bytes", "236223201280",
                "--cache-limit-bytes", "51539607552",
                "--reserved-kv-bytes", "42949672960",
                "--reserved-io-bytes", "2147483648",
                "--reserved-prefetch-bytes", "4294967296",
                "--os-service-reserve-bytes", "8589934592",
            ])

        XCTAssertEqual(
            qwen38MTPScorecardWorkerLaunchArguments(role: .reference, arguments: arguments),
            [
                "qwen38-mtp-scorecard-worker",
                "--role", "reference",
                "--target", "/models/target",
                "--drafter", "/models/drafter",
                "--host-use", "dedicated-serving",
                "--host-use-source", "operator-assertion",
                "--memory-limit-bytes", "236223201280",
                "--cache-limit-bytes", "51539607552",
                "--reserved-kv-bytes", "42949672960",
                "--reserved-io-bytes", "2147483648",
                "--reserved-prefetch-bytes", "4294967296",
                "--os-service-reserve-bytes", "8589934592",
            ])
    }
}
