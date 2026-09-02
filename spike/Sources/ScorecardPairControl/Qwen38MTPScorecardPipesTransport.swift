import Darwin
import Foundation
import HarnessCore

package protocol Qwen38MTPScorecardChildProcess: AnyObject {
    var isRunning: Bool { get }
    func terminate()
    func interrupt()
    func forceKill()
}

extension Process: Qwen38MTPScorecardChildProcess {
    package func forceKill() {
        guard isRunning else { return }
        _ = Darwin.kill(processIdentifier, SIGKILL)
    }
}

package func qwen38MTPScorecardTerminateChild(
    _ child: Qwen38MTPScorecardChildProcess,
    deadlineChecks: Int,
    pollIntervalMicroseconds: useconds_t = 100_000
) {
    guard child.isRunning else { return }
    child.terminate()
    for _ in 0 ..< max(0, deadlineChecks) {
        guard child.isRunning else { return }
        if pollIntervalMicroseconds > 0 {
            usleep(pollIntervalMicroseconds)
        }
    }
    guard child.isRunning else { return }
    child.interrupt()
    for _ in 0 ..< max(0, deadlineChecks) {
        guard child.isRunning else { return }
        if pollIntervalMicroseconds > 0 {
            usleep(pollIntervalMicroseconds)
        }
    }
    guard child.isRunning else { return }
    child.forceKill()
}

package actor Qwen38MTPScorecardBoundedByteSink {
    private let limit: Int
    private var retained = Data()
    private var droppedByteCount = 0

    package init(limit: Int) {
        self.limit = max(0, limit)
    }

    package func append(_ data: Data) {
        guard !data.isEmpty else { return }
        let remaining = max(0, limit - retained.count)
        if remaining > 0 {
            retained.append(data.prefix(remaining))
        }
        droppedByteCount += max(0, data.count - remaining)
    }

    package func snapshot() -> (retained: Data, droppedByteCount: Int) {
        (retained, droppedByteCount)
    }
}

package func qwen38MTPScorecardDrainStderr(
    from handle: FileHandle,
    into sink: Qwen38MTPScorecardBoundedByteSink
) async {
    while !Task.isCancelled {
        let data = handle.availableData
        guard !data.isEmpty else { break }
        await sink.append(data)
    }
}

package func qwen38MTPScorecardMakeTransportPair<Transport: Qwen38MTPScorecardLineTransport>(
    makeCandidate: () throws -> Transport,
    makeReference: () throws -> Transport
) async throws -> (candidate: Transport, reference: Transport) {
    let candidate = try makeCandidate()
    do {
        return (candidate, try makeReference())
    } catch {
        await candidate.terminate()
        throw error
    }
}

/// Generic over an already-running child so this module never constructs or runs a
/// `Process` itself; the live adapter and (later) the proof runner each own the
/// self-exec launch and hand the running child + its three pipes in here.
package actor Qwen38MTPScorecardProcessPipesTransport: Qwen38MTPScorecardLineTransport {
    private let child: any Qwen38MTPScorecardChildProcess
    private let role: Qwen38MTPPerformanceScorecardEngineRole
    private let stdin: FileHandle
    private let stdout: FileHandle
    private let stderr: FileHandle
    private let stderrDrain: Task<Void, Never>
    private let stderrSink: Qwen38MTPScorecardBoundedByteSink
    private var buffer = Data()

    package init(
        role: Qwen38MTPPerformanceScorecardEngineRole,
        child: any Qwen38MTPScorecardChildProcess,
        stdin: FileHandle,
        stdout: FileHandle,
        stderr: FileHandle
    ) {
        self.role = role
        self.child = child
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        let sink = Qwen38MTPScorecardBoundedByteSink(limit: 64 * 1024)
        self.stderrSink = sink
        let stderr = stderr
        self.stderrDrain = Task.detached {
            await qwen38MTPScorecardDrainStderr(from: stderr, into: sink)
        }
    }

    package func sendLine(_ line: String) async throws {
        guard child.isRunning else {
            throw Qwen38MTPScorecardLiveAdapterError.workerExited(role)
        }
        var data = Data(line.utf8)
        data.append(0x0a)
        try stdin.write(contentsOf: data)
    }

    package func receiveLine() async throws -> String? {
        while true {
            if let newline = buffer.firstIndex(of: 0x0a) {
                let line = buffer[..<newline]
                buffer.removeSubrange(...newline)
                return String(data: Data(line), encoding: .utf8)
            }
            let chunk = stdout.availableData
            guard !chunk.isEmpty else { return nil }
            buffer.append(chunk)
        }
    }

    package func terminate() async {
        try? stdin.close()
        qwen38MTPScorecardTerminateChild(child, deadlineChecks: 20)
        try? stdout.close()
        try? stderr.close()
        stderrDrain.cancel()
    }
}
