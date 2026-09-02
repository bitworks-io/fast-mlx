import Foundation
import XCTest

/// Slice 4b black-box CLI tests of the built `fastmlx-proof-runner`
/// binary: argument refusal, the deploy-tree-root preflight, and the
/// FAILED status-JSON contract on stdout (design verdict P7 — the status
/// line is the runner's durable audit witness on failure too). The binary
/// is located from the test bundle's products directory (design verdict
/// P8 location rule); ProofControlTests depends on the executable target
/// so it is always built here.
final class Qwen38ScorecardProofRunnerCLITests: XCTestCase {
    private static var productsDirectory: URL {
        Bundle(for: Qwen38ScorecardProofRunnerCLITests.self)
            .bundleURL
            .deletingLastPathComponent()
    }

    private static var runnerURL: URL {
        productsDirectory.appendingPathComponent("fastmlx-proof-runner")
    }

    private struct RunnerResult {
        let exitStatus: Int32
        let stdout: String
        let stderr: String
    }

    private func runRunner(
        arguments: [String],
        currentDirectory: URL? = nil
    ) throws -> RunnerResult {
        let binary = Self.runnerURL
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: binary.path),
            "fastmlx-proof-runner binary missing at \(binary.path); "
                + "ProofControlTests must depend on the executable target"
        )
        let process = Process()
        process.executableURL = binary
        process.arguments = arguments
        if let currentDirectory {
            process.currentDirectoryURL = currentDirectory
        }
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        let stdoutData = stdout.fileHandleForReading.readDataToEndOfFile()
        let stderrData = stderr.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return RunnerResult(
            exitStatus: process.terminationStatus,
            stdout: String(decoding: stdoutData, as: UTF8.self),
            stderr: String(decoding: stderrData, as: UTF8.self)
        )
    }

    private func decodeStatusLine(_ stdout: String) throws -> [String: Any] {
        let lines = stdout.split(separator: "\n")
        XCTAssertEqual(lines.count, 1, "expected exactly one status line")
        let object = try JSONSerialization.jsonObject(
            with: Data(lines[0].utf8)
        )
        return try XCTUnwrap(object as? [String: Any])
    }

    private func fullArguments(prefix: String) -> [String] {
        [
            "--trust-anchor", "\(prefix)/anchor.json",
            "--policy", "\(prefix)/policy.txt",
            "--policy-signature", "\(prefix)/policy.sig",
            "--claim", "\(prefix)/claim.txt",
            "--claim-signature", "\(prefix)/claim.sig",
            "--worker", "\(prefix)/worker",
            "--target", "\(prefix)/target",
            "--drafter", "\(prefix)/drafter",
            "--output", "\(prefix)/scorecard.jsonl",
            "--authority-output", "\(prefix)/authority.json",
            "--host-use", "dedicated-serving",
            "--host-use-source", "operator-assertion",
            "--expected-chip", "Apple M3 Ultra",
            "--memory-limit-bytes", "1073741824",
            "--cache-limit-bytes", "268435456",
            "--reserved-kv-bytes", "1048576",
            "--reserved-io-bytes", "1048576",
            "--reserved-prefetch-bytes", "1048576",
            "--os-service-reserve-bytes", "1048576",
        ]
    }

    func testMissingFlagsFailWithStatusLineAndNonzeroExit() throws {
        let result = try runRunner(arguments: [])
        XCTAssertEqual(result.exitStatus, 1)
        let status = try decodeStatusLine(result.stdout)
        XCTAssertEqual(status["schema"] as? String, "fast-mlx-proof-control-v1")
        XCTAssertEqual(
            status["program"] as? String,
            "qwen38-mtp-scorecard-proof-runner"
        )
        XCTAssertEqual(status["status"] as? String, "FAILED")
        XCTAssertEqual(status["promotable"] as? Bool, false)
        let message = try XCTUnwrap(status["error"] as? String)
        XCTAssertTrue(
            message.contains("--trust-anchor"),
            "unexpected error message: \(message)"
        )
        XCTAssertTrue(result.stderr.contains("qwen38-scorecard-proof-runner"))
    }

    func testUnknownFlagFailsClosed() throws {
        let result = try runRunner(arguments: ["--worker-pid", "123"])
        XCTAssertEqual(result.exitStatus, 1)
        let status = try decodeStatusLine(result.stdout)
        XCTAssertEqual(status["status"] as? String, "FAILED")
        let message = try XCTUnwrap(status["error"] as? String)
        XCTAssertTrue(message.contains("--worker-pid"))
    }

    /// The deploy-tree-root preflight (design verdict P7 rider c) turns a
    /// wrong-cwd invocation into a clear typed error instead of a distant
    /// evidence-ID mismatch. Run from an empty temp dir with no
    /// `.harness-sha`/`.git` reachable above it.
    func testDeployTreeRootPreflightFailsFromUnmarkedDirectory() throws {
        let temp = URL(
            fileURLWithPath: NSTemporaryDirectory(),
            isDirectory: true
        )
        .appendingPathComponent("fast-mlx-runner-cli-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: temp,
            withIntermediateDirectories: false
        )
        defer { try? FileManager.default.removeItem(at: temp) }

        let result = try runRunner(
            arguments: fullArguments(prefix: temp.path),
            currentDirectory: temp
        )
        XCTAssertEqual(result.exitStatus, 1)
        let status = try decodeStatusLine(result.stdout)
        XCTAssertEqual(status["status"] as? String, "FAILED")
        let message = try XCTUnwrap(status["error"] as? String)
        XCTAssertTrue(
            message.contains("deploy tree root"),
            "unexpected error message: \(message)"
        )
    }

    /// With a marked cwd, the pipeline advances to the hardened input
    /// captures and fails closed on the missing trust anchor — proving the
    /// preflight→capture ordering without any crypto material.
    func testMissingTrustAnchorFailsAtHardenedCapture() throws {
        let temp = URL(
            fileURLWithPath: NSTemporaryDirectory(),
            isDirectory: true
        )
        .appendingPathComponent("fast-mlx-runner-cli-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: temp,
            withIntermediateDirectories: false
        )
        defer { try? FileManager.default.removeItem(at: temp) }
        FileManager.default.createFile(
            atPath: temp.appendingPathComponent(".harness-sha").path,
            contents: Data("0000000000000000000000000000000000000000\n".utf8)
        )

        let result = try runRunner(
            arguments: fullArguments(prefix: temp.path),
            currentDirectory: temp
        )
        XCTAssertEqual(result.exitStatus, 1)
        let status = try decodeStatusLine(result.stdout)
        XCTAssertEqual(status["status"] as? String, "FAILED")
        XCTAssertNil(status["claimSHA256"] as? String)
        XCTAssertNil(status["authorizationID"] as? String)
    }
}
