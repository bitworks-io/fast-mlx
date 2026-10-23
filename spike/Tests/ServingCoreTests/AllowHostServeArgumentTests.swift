import Foundation
import XCTest

@testable import ServingCore

/// Cycle 291: `--allow-host NAME` (repeatable) extends the DNS-rebinding Host guard's allow list on a
/// tokenless server. Mirrors `RequestLogServeArgumentTests`' shape. The key-vs-flag refusal and the
/// startup-line fields live in `FastMLXServeHostGuardStartup` (pure) because `fastmlx-serve` reads
/// `FASTMLX_API_KEY` from the environment, not from `FastMLXServeArguments`.
final class AllowHostServeArgumentTests: XCTestCase {
    func testAbsentFlagDefaultsToEmptyAllowList() throws {
        let arguments = try FastMLXServeArguments.parse(["--scripted"])
        XCTAssertEqual(arguments.allowHosts, [])
    }

    // E9: a valid name parses, is lowercased, and the flag is repeatable (order kept, no duplicates).
    func testAllowHostParsesRepeatsAndLowercases() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--scripted",
            "--allow-host", "My-Box.Example",
            "--allow-host", "203.0.113.5",
            "--allow-host", "my-box.example",
        ])
        XCTAssertEqual(arguments.allowHosts, ["my-box.example", "203.0.113.5"])
    }

    func testAllowHostParsesOnLoadedModelServe() throws {
        let arguments = try FastMLXServeArguments.parse([
            "--model-path", "/models/fixture",
            "--model", "fixture",
            "--memory-limit-bytes", "68719476736",
            "--cache-limit-bytes", "8589934592",
            "--allow-host", "host.docker.internal",
        ])
        XCTAssertEqual(arguments.allowHosts, ["host.docker.internal"])
    }

    // E9: every H5 refusal throws the specific error, fail closed.
    func testAllowHostRefusesWildcardsPortsBracketsPathsAndMalformedNames() {
        let invalid = [
            "*", "*.example.com", "example.*", "example.com:8080", "localhost:80", "[::1]", "::1",
            "a/b", "http://example.com", "example.com/", "exa mple.com", "-bad.example",
            "bad-.example", "a..b", "example.com.", ".example.com", "under_score.example",
            "ex@mple.com", "exämple.com", String(repeating: "a", count: 64) + ".example",
            Array(repeating: String(repeating: "a", count: 62), count: 5).joined(separator: "."),
            "127.1", "2130706433", "0x7f.1", "127.000.000.001", "256.1.1.1", "1.2.3.4.5", "1.2.3.04",
        ]
        for value in invalid {
            XCTAssertThrowsError(
                try FastMLXServeArguments.parse(["--scripted", "--allow-host", value]), value
            ) { error in
                XCTAssertEqual(error as? FastMLXServeArgumentError, .invalidAllowHost, value)
            }
        }
    }

    func testAllowHostAcceptsLongestLegalLabelsAndCanonicalIPv4() throws {
        let label63 = String(repeating: "a", count: 63)
        XCTAssertEqual(FastMLXServeArguments.validatedAllowHost(label63), label63)
        XCTAssertEqual(
            FastMLXServeArguments.validatedAllowHost("0.0.0.0"), "0.0.0.0")
        XCTAssertEqual(
            FastMLXServeArguments.validatedAllowHost("10.20.30.255"), "10.20.30.255")
        XCTAssertEqual(
            FastMLXServeArguments.validatedAllowHost("Xn--Bcher-kva.example"), "xn--bcher-kva.example")
        XCTAssertEqual(FastMLXServeArguments.validatedAllowHost("nas"), "nas")
    }

    func testAllowHostEmptyValueIsRefused() {
        XCTAssertThrowsError(try FastMLXServeArguments.parse(["--scripted", "--allow-host", ""]))
        XCTAssertThrowsError(try FastMLXServeArguments.parse(["--scripted", "--allow-host"])) {
            error in
            XCTAssertEqual(
                error as? FastMLXServeArgumentError, .missingValue("--allow-host"))
        }
        XCTAssertNil(FastMLXServeArguments.validatedAllowHost(""))
    }

    func testAllowHostRefusalAnnounceLineNamesTheFlag() {
        let line = fastMLXServeArgumentRefusalAnnounceLine(.invalidAllowHost)
        XCTAssertTrue(line.contains("--allow-host"), line)
        XCTAssertTrue(line.contains("reason=invalid_arguments"), line)
    }

    func testUsageDocumentsAllowHost() {
        XCTAssertTrue(FastMLXServeArguments.usage.contains("--allow-host NAME"))
    }

    // E10: with a token configured the flag is a startup refusal, never silently ignored.
    func testAllowHostWithApiKeyIsAStartupRefusal() throws {
        let line = try XCTUnwrap(
            FastMLXServeHostGuardStartup.refusalLine(
                apiKeyConfigured: true, allowHosts: ["my-box.example"]))
        XCTAssertTrue(line.hasPrefix("fastmlx-serve configuration=refused reason=allow_host_with_api_key"))
        XCTAssertFalse(line.contains("my-box.example"), "the refusal does not echo the value")
        XCTAssertNil(
            FastMLXServeHostGuardStartup.refusalLine(
                apiKeyConfigured: true, allowHosts: []),
            "a key without --allow-host is fine")
        XCTAssertNil(
            FastMLXServeHostGuardStartup.refusalLine(
                apiKeyConfigured: false, allowHosts: ["my-box.example"]),
            "tokenless with --allow-host is the supported combination")
    }

    // E11: the startup-line fields.
    func testStartupAnnounceFields() {
        XCTAssertEqual(
            FastMLXServeHostGuardStartup.announceFields(apiKeyConfigured: false, allowHosts: []),
            "host_guard=on host_guard_allow=none")
        XCTAssertEqual(
            FastMLXServeHostGuardStartup.announceFields(
                apiKeyConfigured: false, allowHosts: ["a.example", "10.0.0.5"]),
            "host_guard=on host_guard_allow=a.example,10.0.0.5")
        XCTAssertEqual(
            FastMLXServeHostGuardStartup.announceFields(apiKeyConfigured: true, allowHosts: []),
            "host_guard=off host_guard_allow=none")
    }

    // E10/E11 wiring: `fastmlx-serve` must call the shared helpers, refuse before the model load,
    // pass the allow list into the HTTP configuration, and append the fields after the
    // quality-cards fragment (startup-line fields stay one space-separated tail).
    func testServeEntryPointWiresRefusalConfigurationAndAnnouncement() throws {
        let source = try strippedServeSource()
        let refusal = try XCTUnwrap(
            source.range(of: "FastMLXServeHostGuardStartup.refusalLine("),
            "run() never calls the allow-host/API-key refusal")
        let prepare = try XCTUnwrap(source.range(of: "tryawaitprepareBackend(arguments"))
        XCTAssertLessThan(
            refusal.lowerBound, prepare.lowerBound, "the refusal must precede the model load")
        XCTAssertTrue(source.contains("allowedHosts:arguments.allowHosts"))
        let announce = try XCTUnwrap(
            source.range(of: "FastMLXServeHostGuardStartup.announceFields("))
        let quality = try XCTUnwrap(source.range(of: "\\(qualityCardsAnnounce)\""))
        XCTAssertLessThan(
            quality.lowerBound, announce.lowerBound,
            "host_guard fields come after the quality-cards fields")
    }

    /// Whitespace-free, line-comment-free text of `FastMLXServe.swift`, located by SEARCHING the
    /// ancestors of `#filePath` (the fleet deploys `spike/`'s contents as the package root, so a
    /// fixed-depth walk would fail there); neither layout found is a hard failure.
    private func strippedServeSource() throws -> String {
        let suffixes = [
            ["spike", "Sources", "fastmlx-serve", "FastMLXServe.swift"],
            ["Sources", "fastmlx-serve", "FastMLXServe.swift"],
        ]
        var directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        var found: URL?
        for _ in 0..<8 where found == nil {
            for suffix in suffixes {
                let candidate = suffix.reduce(directory) { $0.appendingPathComponent($1) }
                if FileManager.default.fileExists(atPath: candidate.path) {
                    found = candidate
                    break
                }
            }
            directory.deleteLastPathComponent()
        }
        let url = try XCTUnwrap(
            found, "could not locate FastMLXServe.swift by walking up from \(#filePath)")
        let text = try String(contentsOf: url, encoding: .utf8)
        let noComments = text.split(separator: "\n", omittingEmptySubsequences: false)
            .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("//") }
            .joined(separator: "\n")
        return String(noComments.filter { !$0.isWhitespace })
    }
}
