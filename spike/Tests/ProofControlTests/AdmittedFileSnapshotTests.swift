import Darwin
import Foundation
import XCTest
@testable import ProofControl

final class AdmittedFileSnapshotTests: XCTestCase {
    private var caseRoot: URL!

    override func setUpWithError() throws {
        let canonicalTemporaryPath = try XCTUnwrap(
            Darwin.realpath(NSTemporaryDirectory(), nil)
        )
        defer { Darwin.free(canonicalTemporaryPath) }

        caseRoot = URL(
            fileURLWithPath: String(cString: canonicalTemporaryPath),
            isDirectory: true
        )
            .appendingPathComponent("fast-mlx-proof-control-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: caseRoot,
            withIntermediateDirectories: false
        )
    }

    override func tearDownWithError() throws {
        if let caseRoot {
            try? FileManager.default.removeItem(at: caseRoot)
        }
    }

    func testCaptureBindsCanonicalRegularFileBytesAndIdentity() throws {
        let input = caseRoot.appendingPathComponent("manifest.json")
        try Data("admitted\n".utf8).write(to: input)

        let admitted = try AdmittedFile.capture(
            absolutePath: input.path,
            maximumBytes: 64
        )

        XCTAssertEqual(admitted.bytes, Data("admitted\n".utf8))
        XCTAssertEqual(
            admitted.sha256,
            "e7635c5d652fd35a6f2c259b32694b78d0aaefc90d611609e6401a76fcf31265"
        )
        XCTAssertEqual(admitted.identity.size, 9)
        XCTAssertEqual(admitted.identity.linkCount, 1)
        XCTAssertGreaterThan(admitted.identity.inode, 0)
    }

    func testCaptureSnapshotDoesNotChangeAfterSameInodeRewrite() throws {
        let input = caseRoot.appendingPathComponent("worker.py")
        try Data("ORIGINAL\n".utf8).write(to: input)
        let admitted = try AdmittedFile.capture(
            absolutePath: input.path,
            maximumBytes: 64
        )

        let handle = try FileHandle(forWritingTo: input)
        try handle.seek(toOffset: 0)
        try handle.write(contentsOf: Data("MALICIOUS\n".utf8))
        try handle.truncate(atOffset: 10)
        try handle.close()

        let rewrittenAttributes = try FileManager.default.attributesOfItem(atPath: input.path)
        let rewrittenInode = try XCTUnwrap(
            rewrittenAttributes[.systemFileNumber] as? NSNumber
        ).uint64Value
        XCTAssertEqual(rewrittenInode, admitted.identity.inode)
        XCTAssertEqual(admitted.bytes, Data("ORIGINAL\n".utf8))
        XCTAssertNotEqual(admitted.bytes, try Data(contentsOf: input))
    }

    func testCaptureRejectsFinalSymlink() throws {
        let real = caseRoot.appendingPathComponent("real.json")
        let link = caseRoot.appendingPathComponent("link.json")
        try Data("real\n".utf8).write(to: real)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: real)

        XCTAssertThrowsError(
            try AdmittedFile.capture(absolutePath: link.path, maximumBytes: 64)
        ) { error in
            guard case AdmissionError.pathComponentRejected = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }

    func testCaptureRejectsAncestorSymlink() throws {
        let realDirectory = caseRoot.appendingPathComponent("real")
        let linkDirectory = caseRoot.appendingPathComponent("linked")
        try FileManager.default.createDirectory(
            at: realDirectory,
            withIntermediateDirectories: false
        )
        try Data("real\n".utf8).write(
            to: realDirectory.appendingPathComponent("manifest.json")
        )
        try FileManager.default.createSymbolicLink(
            at: linkDirectory,
            withDestinationURL: realDirectory
        )

        XCTAssertThrowsError(
            try AdmittedFile.capture(
                absolutePath: linkDirectory.appendingPathComponent("manifest.json").path,
                maximumBytes: 64
            )
        ) { error in
            guard case AdmissionError.pathComponentRejected = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }

    func testCaptureRejectsHardLink() throws {
        let original = caseRoot.appendingPathComponent("original.json")
        let linked = caseRoot.appendingPathComponent("linked.json")
        try Data("real\n".utf8).write(to: original)
        try FileManager.default.linkItem(at: original, to: linked)

        XCTAssertThrowsError(
            try AdmittedFile.capture(absolutePath: original.path, maximumBytes: 64)
        ) { error in
            XCTAssertEqual(error as? AdmissionError, .unexpectedLinkCount(2))
        }
    }

    func testCaptureRejectsDirectoryAndOversizeInput() throws {
        XCTAssertThrowsError(
            try AdmittedFile.capture(absolutePath: caseRoot.path, maximumBytes: 64)
        ) { error in
            XCTAssertEqual(error as? AdmissionError, .notRegularFile)
        }

        let oversized = caseRoot.appendingPathComponent("oversized.json")
        try Data(repeating: 0x41, count: 65).write(to: oversized)
        XCTAssertThrowsError(
            try AdmittedFile.capture(absolutePath: oversized.path, maximumBytes: 64)
        ) { error in
            XCTAssertEqual(
                error as? AdmissionError,
                .exceedsMaximumBytes(maximum: 64, actual: 65)
            )
        }
    }

    func testCaptureRejectsFIFOWithoutWaitingForAWriterAndRejectsDeviceNodes() throws {
        let fifo = caseRoot.appendingPathComponent("input.fifo")
        XCTAssertEqual(Darwin.mkfifo(fifo.path, S_IRUSR | S_IWUSR), 0)

        let delayedWriter = DispatchWorkItem {
            usleep(250_000)
            let descriptor = Darwin.open(fifo.path, O_WRONLY | O_NONBLOCK)
            if descriptor >= 0 {
                Darwin.close(descriptor)
            }
        }
        DispatchQueue.global().async(execute: delayedWriter)

        let started = ContinuousClock.now
        XCTAssertThrowsError(
            try AdmittedFile.capture(absolutePath: fifo.path, maximumBytes: 64)
        ) { error in
            XCTAssertEqual(error as? AdmissionError, .notRegularFile)
        }
        let elapsed = ContinuousClock.now - started
        delayedWriter.wait()
        XCTAssertLessThan(
            elapsed,
            .milliseconds(100),
            "opening an admitted FIFO must not block waiting for a writer"
        )

        XCTAssertThrowsError(
            try AdmittedFile.capture(absolutePath: "/dev/null", maximumBytes: 64)
        ) { error in
            XCTAssertEqual(error as? AdmissionError, .notRegularFile)
        }
    }

    func testCaptureRejectsNonCanonicalOrRelativePath() throws {
        for path in [
            "relative.json",
            "\(caseRoot.path)//manifest.json",
            "\(caseRoot.path)/../manifest.json",
            "\(caseRoot.path)/",
        ] {
            XCTAssertThrowsError(
                try AdmittedFile.capture(absolutePath: path, maximumBytes: 64),
                "expected rejection for \(path)"
            ) { error in
                XCTAssertEqual(error as? AdmissionError, .nonCanonicalAbsolutePath)
            }
        }
    }
}
