import Foundation
import HarnessCore
import XCTest

@testable import fastmlx_harness

final class KVTunerQualificationCLITests: XCTestCase {
    func testCandidateCohortRequiresOneCurrentExecutionEnvironment() throws {
        let currentSHA = String(repeating: "a", count: 40)
        let environment = KVTunerCandidateExecutionEnvironment(
            harnessGitSHA: currentSHA,
            buildConfiguration: "Release",
            mlxSwiftVersion: "0.31.6",
            mlxSwiftLMRevision:
                "702e5a0eaf990e1f6d3db2b6e7d8872858a44055",
            hardwareChip: "Apple M3 Ultra",
            hardwareRAMBytes: 256 << 30,
            memoryCacheLimitBytes: 8 << 30,
            hardwareOS: "macOS 26.5",
            modelConfigHash: String(repeating: "1", count: 16),
            modelConfigSHA256: String(repeating: "2", count: 64),
            checkpointManifestHash: String(repeating: "3", count: 64),
            tokenizerSHA256: String(repeating: "4", count: 64))

        XCTAssertNoThrow(try validateKVTunerCandidateExecutionEnvironments(
            [environment, environment], currentHarnessGitSHA: currentSHA))
        XCTAssertThrowsError(
            try validateKVTunerCandidateExecutionEnvironments(
                [], currentHarnessGitSHA: currentSHA))

        var stale = environment
        stale.harnessGitSHA = String(repeating: "b", count: 40)
        XCTAssertThrowsError(
            try validateKVTunerCandidateExecutionEnvironments(
                [environment, stale], currentHarnessGitSHA: currentSHA))

        var differentHardware = environment
        differentHardware.hardwareRAMBytes = 128 << 30
        XCTAssertThrowsError(
            try validateKVTunerCandidateExecutionEnvironments(
                [environment, differentHardware],
                currentHarnessGitSHA: currentSHA))
    }

    func testQualificationSHARequiresCleanLiveOrStampedCommit() throws {
        let live = String(repeating: "a", count: 40)
        let stamp = String(repeating: "b", count: 40)

        XCTAssertEqual(try resolveKVTunerQualificationHarnessGitSHA(
            liveGitOutput: live,
            liveRepositoryPresent: true,
            shaFile: "\(stamp)\n"), live)
        XCTAssertEqual(try resolveKVTunerQualificationHarnessGitSHA(
            liveGitOutput: nil,
            liveRepositoryPresent: false,
            shaFile: "\(stamp)\n"), stamp)
        XCTAssertThrowsError(try resolveKVTunerQualificationHarnessGitSHA(
            liveGitOutput: "\(live)-dirty",
            liveRepositoryPresent: true,
            shaFile: stamp))
        XCTAssertThrowsError(try resolveKVTunerQualificationHarnessGitSHA(
            liveGitOutput: nil,
            liveRepositoryPresent: true,
            shaFile: stamp))
        XCTAssertThrowsError(try resolveKVTunerQualificationHarnessGitSHA(
            liveGitOutput: nil,
            liveRepositoryPresent: false,
            shaFile: "\(stamp)-dirty"))
        XCTAssertThrowsError(try resolveKVTunerQualificationHarnessGitSHA(
            liveGitOutput: nil,
            liveRepositoryPresent: false,
            shaFile: "unknown"))
    }

    func testQualificationSourceRootIgnoresUnrelatedAncestorGit() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let unrelatedGit = root.appendingPathComponent(".git", isDirectory: true)
        let deploy = root.appendingPathComponent(
            "fast-mlx-spike", isDirectory: true)
        try FileManager.default.createDirectory(
            at: unrelatedGit, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: deploy, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data().write(to: deploy.appendingPathComponent("Package.swift"))
        try Data().write(to: deploy.appendingPathComponent(".harness-sha"))

        XCTAssertNil(kvtunerQualificationSourceRepositoryRoot(
            startingAt: deploy))

        let source = root.appendingPathComponent("source", isDirectory: true)
        let sourceGit = source.appendingPathComponent(".git", isDirectory: true)
        let sourceSpike = source.appendingPathComponent("spike", isDirectory: true)
        try FileManager.default.createDirectory(
            at: sourceGit, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: sourceSpike, withIntermediateDirectories: true)
        try Data().write(to: source.appendingPathComponent("AGENTS.md"))
        try Data().write(to: sourceSpike.appendingPathComponent("Package.swift"))

        XCTAssertEqual(
            kvtunerQualificationSourceRepositoryRoot(
                startingAt: sourceSpike)?.standardizedFileURL,
            source.standardizedFileURL)
    }

    func testCandidatePlanSupportsCompleteRunOrOneResumableOrdinal() throws {
        let base = [
            "--model", "/models/qwen",
            "--manifest", "/evidence/manifest.json",
            "--sensitivity", "/evidence/sensitivity.json",
            "--target-pair-bits", "576",
            "--max-candidates", "10000",
            "--output-dir", "/evidence/candidates",
        ]
        let all = try parseKVTunerCandidatePlan(Flags(base))
        XCTAssertNil(all.candidateOrdinal)
        XCTAssertEqual(all.targetPairBitTotal, 576)
        XCTAssertEqual(all.maxCandidates, 10_000)

        let one = try parseKVTunerCandidatePlan(Flags(
            base + ["--candidate-ordinal", "7"]))
        XCTAssertEqual(one.candidateOrdinal, 7)
        XCTAssertEqual(
            kvtunerCandidateArtifactURL(
                directory: one.outputDirectory,
                ordinal: 7).lastPathComponent,
            "candidate-00007.json")

        XCTAssertThrowsError(try parseKVTunerCandidatePlan(Flags(
            base + ["--candidate-ordinal", "-1"])))
        var zeroCandidateLimit = base
        zeroCandidateLimit[zeroCandidateLimit.firstIndex(
            of: "10000")!] = "0"
        XCTAssertThrowsError(try parseKVTunerCandidatePlan(Flags(
            zeroCandidateLimit)))
    }

    func testAllQualificationPlansParseTheirAuthenticatedInputs() throws {
        let manifest = try parseKVTunerManifestPlan(Flags([
            "--model", "/models/qwen",
            "--prompt-fixture", "/evidence/prompts.json",
            "--normalized-targets", "/evidence/targets.json",
            "--output", "/evidence/manifest.json",
        ]))
        XCTAssertEqual(manifest.outputPath, "/evidence/manifest.json")

        let sensitivity = try parseKVTunerSensitivityPlan(Flags([
            "--model", "/models/qwen",
            "--manifest", "/evidence/manifest.json",
            "--matrix-id", "qwen3-32b-kvtuner-g128-v1",
            "--group-size", "128",
            "--output", "/evidence/sensitivity.json",
        ]))
        XCTAssertEqual(sensitivity.groupSize, 128)
        XCTAssertEqual(sensitivity.matrixID, "qwen3-32b-kvtuner-g128-v1")

        let search = try parseKVTunerSearchPlan(Flags([
            "--model", "/models/qwen",
            "--manifest", "/evidence/manifest.json",
            "--sensitivity", "/evidence/sensitivity.json",
            "--target-pair-bits", "576",
            "--max-candidates", "10000",
            "--candidate-dir", "/evidence/candidates",
            "--output", "/evidence/search.json",
            "--schedule-output", "/evidence/schedule.json",
        ]))
        XCTAssertEqual(search.targetPairBitTotal, 576)
        XCTAssertEqual(search.scheduleOutputPath, "/evidence/schedule.json")

        let bundle = try parseKVTunerBundlePlan(Flags([
            "--model", "/models/qwen",
            "--manifest", "/evidence/manifest.json",
            "--sensitivity", "/evidence/sensitivity.json",
            "--search", "/evidence/search.json",
            "--schedule", "/evidence/schedule.json",
            "--candidate-dir", "/evidence/candidates",
            "--output", "/evidence/bundle.json",
        ]))
        XCTAssertEqual(bundle.searchPath, "/evidence/search.json")
        XCTAssertEqual(bundle.outputPath, "/evidence/bundle.json")
    }

    func testQualificationPlansRejectInvalidValuesAndAliasedOutputs() throws {
        XCTAssertThrowsError(try parseKVTunerManifestPlan(Flags([
            "--model", "/models/qwen",
            "--prompt-fixture", "/evidence/prompts.json",
            "--normalized-targets", "/evidence/targets.json",
            "--output", "/evidence/prompts.json",
        ])))

        XCTAssertThrowsError(try parseKVTunerSearchPlan(Flags([
            "--model", "/models/qwen",
            "--manifest", "/evidence/manifest.json",
            "--sensitivity", "/evidence/sensitivity.json",
            "--target-pair-bits", "576",
            "--max-candidates", "10000",
            "--candidate-dir", "/evidence/candidates",
            "--output", "/evidence/search.json",
            "--schedule-output", "/evidence/search.json",
        ])))

        XCTAssertThrowsError(try parseKVTunerSensitivityPlan(Flags([
            "--model", "/models/qwen",
            "--manifest", "/evidence/manifest.json",
            "--matrix-id", "qwen3-32b-kvtuner-g128-v1",
            "--group-size", "32",
            "--output", "/evidence/sensitivity.json",
        ])))

        XCTAssertThrowsError(try parseKVTunerBundlePlan(Flags([
            "--model", "/models/qwen",
            "--manifest", "/evidence/manifest.json",
            "--sensitivity", "/evidence/sensitivity.json",
            "--search", "/evidence/search.json",
            "--schedule", "/evidence/schedule.json",
            "--candidate-dir", "/evidence/candidates",
            "--output", "/evidence/search.json",
        ])))
    }

    func testManifestPlanDetectsAliasedParentPaths() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let real = directory.appendingPathComponent("real", isDirectory: true)
        let alias = directory.appendingPathComponent("alias", isDirectory: true)
        try FileManager.default.createDirectory(
            at: real, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createSymbolicLink(
            at: alias, withDestinationURL: real)
        let prompts = real.appendingPathComponent("prompts.json")
        try Data("[]".utf8).write(to: prompts)

        XCTAssertThrowsError(try parseKVTunerManifestPlan(Flags([
            "--model", directory.appendingPathComponent("model").path,
            "--prompt-fixture", prompts.path,
            "--normalized-targets",
                real.appendingPathComponent("targets.json").path,
            "--output", alias.appendingPathComponent("prompts.json").path,
        ])))
    }

    func testArtifactWriterRequiresFreshRegularDestination() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let output = directory.appendingPathComponent("artifact.json")
        let payload = Data(#"{"schemaVersion":1}"#.utf8)

        try writeFreshKVTunerArtifact(payload, to: output.path)
        XCTAssertEqual(try Data(contentsOf: output), payload)
        XCTAssertThrowsError(try writeFreshKVTunerArtifact(
            Data("replacement".utf8), to: output.path))

        let emptyOutput = directory.appendingPathComponent("empty.json")
        try Data().write(to: emptyOutput)
        try writeFreshKVTunerArtifact(payload, to: emptyOutput.path)
        XCTAssertEqual(try Data(contentsOf: emptyOutput), payload)

        let directoryOutput = directory.appendingPathComponent(
            "directory.json", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directoryOutput, withIntermediateDirectories: false)
        XCTAssertThrowsError(try writeFreshKVTunerArtifact(
            payload, to: directoryOutput.path))

        let target = directory.appendingPathComponent("target.json")
        try Data().write(to: target)
        let symbolicLink = directory.appendingPathComponent("link.json")
        try FileManager.default.createSymbolicLink(
            at: symbolicLink, withDestinationURL: target)
        XCTAssertThrowsError(try writeFreshKVTunerArtifact(
            payload, to: symbolicLink.path))

        let lockedOutput = directory.appendingPathComponent("locked.json")
        let lock = kvtunerArtifactLockURL(for: lockedOutput.path)
        try Data().write(to: lock, options: .withoutOverwriting)
        XCTAssertThrowsError(try writeFreshKVTunerArtifact(
            payload, to: lockedOutput.path))
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: lockedOutput.path))
    }

    func testArtifactResumeAcceptsOnlyExactCanonicalBytes() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let output = directory.appendingPathComponent("artifact.json")
        let expected = Data(#"{"a":1}"#.utf8)

        XCTAssertTrue(try writeOrValidateExactKVTunerArtifact(
            expected, to: output.path))
        XCTAssertFalse(try writeOrValidateExactKVTunerArtifact(
            expected, to: output.path))
        XCTAssertThrowsError(try writeOrValidateExactKVTunerArtifact(
            Data(#"{"a":2}"#.utf8), to: output.path))
    }

    func testCandidateDirectoryRequiresOneCanonicalFilePerOrdinal() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let first = kvtunerCandidateArtifactURL(
            directory: directory.path, ordinal: 0)
        let firstData = Data(#"{"candidateOrdinal":0}"#.utf8)
        try firstData.write(to: first)

        XCTAssertEqual(
            try exactKVTunerCandidateArtifactData(
                directory: directory.path, count: 1),
            [firstData])

        let stale = kvtunerCandidateArtifactURL(
            directory: directory.path, ordinal: 1)
        try Data(#"{"candidateOrdinal":1}"#.utf8).write(to: stale)
        XCTAssertThrowsError(try exactKVTunerCandidateArtifactData(
            directory: directory.path, count: 1))
        try FileManager.default.removeItem(at: stale)
        try FileManager.default.removeItem(at: first)
        XCTAssertThrowsError(try exactKVTunerCandidateArtifactData(
            directory: directory.path, count: 1))
    }

    func testCandidateDirectoryRejectsMissingMiddleAndExtraOrdinals() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let rows = (0..<4).map {
            Data("{\"candidateOrdinal\":\($0)}".utf8)
        }
        for ordinal in [2, 0] {
            try rows[ordinal].write(to: kvtunerCandidateArtifactURL(
                directory: directory.path, ordinal: ordinal))
        }

        XCTAssertThrowsError(try exactKVTunerCandidateArtifactData(
            directory: directory.path, count: 3))
        try rows[1].write(to: kvtunerCandidateArtifactURL(
            directory: directory.path, ordinal: 1))
        XCTAssertEqual(
            try exactKVTunerCandidateArtifactData(
                directory: directory.path, count: 3),
            Array(rows.prefix(3)))

        try rows[3].write(to: kvtunerCandidateArtifactURL(
            directory: directory.path, ordinal: 3))
        XCTAssertThrowsError(try exactKVTunerCandidateArtifactData(
            directory: directory.path, count: 3))
    }
}
