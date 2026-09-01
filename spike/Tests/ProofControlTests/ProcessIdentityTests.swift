import Foundation
import XCTest

@testable import ProofControl

final class ProcessIdentityTests: XCTestCase {
    func testBootTimeIsPositiveAndInThePast() throws {
        let boot = try ProcessIdentity.bootTimeUnixSeconds()
        XCTAssertGreaterThan(boot, 0)
        XCTAssertLessThan(boot, Int(Date().timeIntervalSince1970) + 1)
    }

    func testOwnProcessStartUptimeIsPositiveAndStable() throws {
        let pid = getpid()
        let first = try ProcessIdentity.processStartUptimeNanoseconds(pid: pid)
        let second = try ProcessIdentity.processStartUptimeNanoseconds(pid: pid)
        XCTAssertGreaterThan(first, 0)
        XCTAssertEqual(first, second, "same pid must yield a stable start identity")
    }

    func testOwnExecutableSHA256IsCanonicalAndStable() throws {
        let pid = getpid()
        let first = try ProcessIdentity.executableSHA256(pid: pid)
        let second = try ProcessIdentity.executableSHA256(pid: pid)
        XCTAssertEqual(first.count, 64)
        XCTAssertTrue(first.allSatisfy { "0123456789abcdef".contains($0) })
        XCTAssertEqual(first, second)
    }

    func testExecutableSHA256MatchesDirectHashOfReportedPath() throws {
        let pid = getpid()
        let path = try ProcessIdentity.executablePath(pid: pid)
        XCTAssertTrue(FileManager.default.fileExists(atPath: path))
        let expected = try ProcessIdentity.sha256Hex(
            Data(contentsOf: URL(fileURLWithPath: path)))
        XCTAssertEqual(try ProcessIdentity.executableSHA256(pid: pid), expected)
    }

    func testUnavailableProcessFailsClosed() {
        XCTAssertThrowsError(
            try ProcessIdentity.processStartUptimeNanoseconds(pid: -1))
        XCTAssertThrowsError(try ProcessIdentity.executableSHA256(pid: -1))
    }
}
