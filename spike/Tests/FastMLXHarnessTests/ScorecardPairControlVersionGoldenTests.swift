import XCTest

import ScorecardPairControl

@testable import fastmlx_harness

/// Golden pin: ScorecardPairControl is MLX-free and can't read Package.swift's
/// mlx-swift `exact:` version, so its own mirrored constant must stay in sync
/// with fastmlx-harness's ProvenanceCLI.mlxSwiftVersion (which is itself pinned
/// to the same Package.swift dependency) by direct comparison here. Kept in its
/// own class (rather than folded into Qwen38MTPScorecardLiveAdapterTests) so
/// that suite's test count stays exactly what it was before this refactor.
final class ScorecardPairControlVersionGoldenTests: XCTestCase {
    func testScorecardPairControlVersionsMirrorsProvenanceCLIMLXSwiftVersion() {
        XCTAssertEqual(ScorecardPairControlVersions.mlxSwiftVersion, ProvenanceCLI.mlxSwiftVersion)
    }

    /// Security-review F4: also cross-pin against Package.swift's literal
    /// `exact:` dependency line, so bumping the dependency cannot silently
    /// drift BOTH mirrored constants at once.
    func testMirroredVersionMatchesPackageSwiftExactPin() throws {
        let packageSwift = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Package.swift")
        let content = try String(contentsOf: packageSwift, encoding: .utf8)
        let expectedPin =
            "mlx-swift.git\", exact: \"\(ScorecardPairControlVersions.mlxSwiftVersion)\""
        XCTAssertTrue(
            content.contains(expectedPin),
            "Package.swift no longer pins mlx-swift at exact: "
                + "\"\(ScorecardPairControlVersions.mlxSwiftVersion)\" — update "
                + "ScorecardPairControlVersions and ProvenanceCLI together"
        )
    }
}
