import Foundation
import HarnessCore
import XCTest

import ScorecardPairControl

/// Exercises `Qwen38MTPScorecardProcessPipesTransport` against a real child
/// process (`/bin/cat`) rather than a fake, since the transport's whole job is
/// wiring three already-open `FileHandle`s to an already-running `Process` —
/// behavior a protocol stub can't verify.
final class Qwen38MTPScorecardPipesTransportTests: XCTestCase {
    func testSendReceiveRoundTripsThroughRealChildThenTerminateExitsIt() async throws {
        let (process, input, output, error) = try spawnCat()
        let transport = Qwen38MTPScorecardProcessPipesTransport(
            role: .candidate,
            child: process,
            stdin: input.fileHandleForWriting,
            stdout: output.fileHandleForReading,
            stderr: error.fileHandleForReading)

        try await transport.sendLine("x")
        let echoed = try await transport.receiveLine()

        XCTAssertEqual(echoed, "x")

        await transport.terminate()
        process.waitUntilExit()
        XCTAssertFalse(process.isRunning)
    }

    func testSendLineAfterChildExitThrowsWorkerExited() async throws {
        let (process, input, output, error) = try spawnCat()
        let transport = Qwen38MTPScorecardProcessPipesTransport(
            role: .reference,
            child: process,
            stdin: input.fileHandleForWriting,
            stdout: output.fileHandleForReading,
            stderr: error.fileHandleForReading)

        // Close stdin directly (bypassing the transport) so `cat` sees EOF and
        // exits on its own, independent of the transport's own terminate().
        try input.fileHandleForWriting.close()
        process.waitUntilExit()
        XCTAssertFalse(process.isRunning)

        do {
            try await transport.sendLine("y")
            XCTFail("Expected workerExited after the child exited")
        } catch {
            XCTAssertEqual(
                error as? Qwen38MTPScorecardLiveAdapterError,
                .workerExited(.reference))
        }
    }

    private func spawnCat() throws -> (
        process: Process, input: Pipe, output: Pipe, error: Pipe
    ) {
        let input = Pipe()
        let output = Pipe()
        let error = Pipe()
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/cat")
        process.standardInput = input
        process.standardOutput = output
        process.standardError = error
        try process.run()
        return (process, input, output, error)
    }
}
