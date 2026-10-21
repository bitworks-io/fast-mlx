import XCTest

/// Structural (source-text) pins for the two production wiring lines that carry
/// `--default-sampling generation-config` onto the continuous-batch route. Neither line has any
/// behavioral unit coverage: `fastmlx-serve` is an executable target with no test target, and
/// `DefaultSamplingServeWiringTests` constructs `ContinuousServingModelLoadConfiguration` and
/// `ContinuousServingBackendConfiguration` DIRECTLY, which proves the helper and the backend honor
/// the field while proving nothing about whether the real call sites set it. Deleting either line
/// would silently turn the flag into a no-op on the continuous route (param-less requests decode
/// greedy) with every other test still green.
///
///  (a) `FastMLXServe.swift`: the `ContinuousServingModelLoadConfiguration(` construction forwards
///      `defaultSampling: arguments.defaultSampling`.
///  (b) `MLXContinuousServing.swift`: `loadContinuousServingModel` resolves the checkpoint defaults
///      through `resolveServingDefaultSampling(` BEFORE the first `Memory.memoryLimit =` (so a bad
///      `generation_config.json` refuses before any process-global mutation or weight load) and
///      assigns that resolved value to `backendConfiguration.samplingDefaults`.
///
/// **Source-text gate, not behavioral:** it proves the wiring exists, not that the served response
/// is sampled. Mirrors `OffloadPlanCheckOnlyServeWiringStructuralTests` (ancestor search from
/// `#filePath` accepting both the checked-out and fleet-synced layouts; `//` comments stripped before
/// matching so a commented-out line cannot satisfy a pin; fails closed rather than vacuously passing).
/// Known limitation shared with that test: the comment stripper is a per-line `//` removal, not a
/// lexer, so a `//` inside a string literal on a scanned line would truncate that line.
final class DefaultSamplingContinuousWiringStructuralTests: XCTestCase {

    /// Thrown (after the descriptive assertion has already recorded the failure) to stop a pin whose
    /// scanned text has an unexpected shape; a thrown error fails the test, it never skips it.
    private struct PinShapeError: Error {}

    private func resolveSourceFile(suffixes: [[String]]) -> URL? {
        var searchDirectory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<8 {
            for suffix in suffixes {
                let candidate = suffix.reduce(searchDirectory) { $0.appendingPathComponent($1) }
                if FileManager.default.fileExists(atPath: candidate.path) {
                    return candidate
                }
            }
            searchDirectory.deleteLastPathComponent()
        }
        return nil
    }

    private func strippedSource(
        spikeRelativeComponents components: [String]
    ) throws -> String {
        let url = try XCTUnwrap(
            resolveSourceFile(suffixes: [["spike"] + components, components]),
            "could not locate \(components.joined(separator: "/")) by walking up from #filePath "
                + "(\(#filePath)); the layout moved relative to this test file. Fix the resolver "
                + "rather than letting this pin skip.")
        let source = try String(contentsOf: url, encoding: .utf8)
        XCTAssertFalse(source.isEmpty, "\(url.path) is empty")
        return stripLineComments(from: source)
    }

    private func stripLineComments(from source: String) -> String {
        source.split(separator: "\n", omittingEmptySubsequences: false)
            .map { line -> Substring in
                if let commentRange = line.range(of: "//") {
                    return line[line.startIndex..<commentRange.lowerBound]
                }
                return line
            }
            .joined(separator: "\n")
    }

    private func isIdentifierCharacter(_ character: Character) -> Bool {
        character.isLetter || character.isNumber || character == "_"
    }

    /// Text from `open` (which must index an `open` delimiter) through its matching `close`.
    private func balancedText(
        in source: String, from start: String.Index, open: Character, close: Character
    ) -> String? {
        var depth = 0
        var index = start
        while index < source.endIndex {
            let character = source[index]
            if character == open {
                depth += 1
            } else if character == close {
                depth -= 1
                if depth == 0 {
                    return String(source[start...index])
                }
            }
            index = source.index(after: index)
        }
        return nil
    }

    // MARK: (a) FastMLXServe.swift forwards defaultSampling into the continuous load configuration

    func testContinuousServingModelLoadConfigurationConstructionForwardsDefaultSampling() throws {
        let source = try strippedSource(
            spikeRelativeComponents: ["Sources", "fastmlx-serve", "FastMLXServe.swift"])
        let marker = "ContinuousServingModelLoadConfiguration("
        var sites: [String] = []
        var searchRange = source.startIndex..<source.endIndex
        while let markerRange = source.range(of: marker, range: searchRange) {
            searchRange = markerRange.upperBound..<source.endIndex
            // Identifier-bounded: skip e.g. `validateContinuousServingModelLoadConfiguration(`.
            if markerRange.lowerBound > source.startIndex,
                isIdentifierCharacter(source[source.index(before: markerRange.lowerBound)])
            {
                continue
            }
            let openParen = source.index(before: markerRange.upperBound)
            let call = try XCTUnwrap(
                balancedText(in: source, from: openParen, open: "(", close: ")"),
                "unbalanced parens in a \(marker) call")
            sites.append(call)
        }
        XCTAssertFalse(
            sites.isEmpty,
            "found zero real \(marker) construction sites in FastMLXServe.swift (comments stripped); "
                + "the call moved or was renamed -- fix this pin rather than letting it pass vacuously.")
        let needle = "defaultSampling:arguments.defaultSampling"
        for (offset, site) in sites.enumerated() {
            let compact = String(site.filter { !$0.isWhitespace })
            var forwarded = false
            var range = compact.startIndex..<compact.endIndex
            while let hit = compact.range(of: needle, range: range) {
                range = hit.upperBound..<compact.endIndex
                let before = hit.lowerBound > compact.startIndex
                    ? compact[compact.index(before: hit.lowerBound)] : Character(" ")
                let after = hit.upperBound < compact.endIndex ? compact[hit.upperBound] : Character(" ")
                if (before == "(" || before == ",") && (after == "," || after == ")") {
                    forwarded = true
                    break
                }
            }
            XCTAssertTrue(
                forwarded,
                "\(marker) construction site #\(offset) does not forward "
                    + "`defaultSampling: arguments.defaultSampling`; `--default-sampling "
                    + "generation-config` would silently be a no-op on the continuous route.")
        }
    }

    // MARK: (b) loadContinuousServingModel resolves, orders, and assigns the defaults

    private func loadContinuousServingModelBody() throws -> String {
        let source = try strippedSource(
            spikeRelativeComponents: [
                "Sources", "SpikeServingAdapters", "MLXContinuousServing.swift",
            ])
        let marker = "func loadContinuousServingModel("
        let markerRange = try XCTUnwrap(
            source.range(of: marker),
            "found no `\(marker)` in MLXContinuousServing.swift (comments stripped)")
        let parameterOpen = source.index(before: markerRange.upperBound)
        let parameters = try XCTUnwrap(
            balancedText(in: source, from: parameterOpen, open: "(", close: ")"))
        let afterParameters = source.index(parameterOpen, offsetBy: parameters.count)
        let bodyOpen = try XCTUnwrap(
            source[afterParameters...].firstIndex(of: "{"),
            "no function body found after the `\(marker)` parameter list")
        let body = try XCTUnwrap(
            balancedText(in: source, from: bodyOpen, open: "{", close: "}"),
            "unbalanced braces in loadContinuousServingModel")
        // Anti-vacuity: the extracted body must be the real, whole function.
        XCTAssertTrue(
            body.contains("Memory.memoryLimit =") && body.contains("return LoadedContinuousServingModel("),
            "the extracted loadContinuousServingModel body is not the whole function; the brace "
                + "scan went wrong -- fix this pin rather than trusting its other assertions.")
        return body
    }

    /// The identifier bound by `let <name> = try resolveServingDefaultSampling(` in `body`.
    private func resolvedDefaultsBinding(in body: String) throws -> (name: String, offset: Int) {
        let call = "resolveServingDefaultSampling("
        let callRange = try XCTUnwrap(
            body.range(of: call),
            "loadContinuousServingModel never calls `\(call)`: --default-sampling would be "
                + "unresolved (and never refused) on the continuous route.")
        let lineStart = body[..<callRange.lowerBound].lastIndex(of: "\n").map {
            body.index(after: $0)
        } ?? body.startIndex
        let prefix = String(body[lineStart..<callRange.lowerBound])
        let compactPrefix = prefix.split(whereSeparator: { $0 == " " || $0 == "\t" })
            .map(String.init)
        XCTAssertEqual(
            compactPrefix.count, 4,
            "expected `let <name> = try ` before `\(call)`, found `\(prefix)`")
        guard compactPrefix.count == 4, compactPrefix[0] == "let", compactPrefix[2] == "=",
            compactPrefix[3] == "try"
        else {
            throw PinShapeError()
        }
        return (compactPrefix[1], body.distance(from: body.startIndex, to: callRange.lowerBound))
    }

    func testLoadContinuousServingModelAssignsResolvedDefaultsToBackendConfiguration() throws {
        let body = try loadContinuousServingModelBody()
        let binding = try resolvedDefaultsBinding(in: body)
        let assignment = "backendConfiguration.samplingDefaults = \(binding.name)"
        var assignmentOffset: Int?
        var range = body.startIndex..<body.endIndex
        while let hit = body.range(of: assignment, range: range) {
            range = hit.upperBound..<body.endIndex
            if hit.upperBound == body.endIndex || !isIdentifierCharacter(body[hit.upperBound]) {
                assignmentOffset = body.distance(from: body.startIndex, to: hit.lowerBound)
                break
            }
        }
        let offset = try XCTUnwrap(
            assignmentOffset,
            "loadContinuousServingModel does not assign `\(assignment)` (the value bound from "
                + "resolveServingDefaultSampling); the resolved defaults never reach the backend "
                + "and the flag is a silent no-op on the continuous route.")
        XCTAssertGreaterThan(
            offset, binding.offset,
            "`\(assignment)` appears before the value is resolved")
    }

    func testLoadContinuousServingModelResolvesDefaultsBeforeFirstMemoryMutation() throws {
        let body = try loadContinuousServingModelBody()
        let binding = try resolvedDefaultsBinding(in: body)
        let memoryMutation = try XCTUnwrap(
            body.range(of: "Memory.memoryLimit ="),
            "loadContinuousServingModel has no `Memory.memoryLimit =` to order against")
        let memoryOffset = body.distance(from: body.startIndex, to: memoryMutation.lowerBound)
        XCTAssertLessThan(
            binding.offset, memoryOffset,
            "resolveServingDefaultSampling( must run BEFORE the first `Memory.memoryLimit =` so a "
                + "bad generation_config.json refuses before any process-global mutation or weight load.")
    }

    // MARK: (c) cycle 281: the startup line announces the active default sampling preset

    /// Body of `PreparedServingBackend.startupLine(localAddress:)` in FastMLXServe.swift.
    private func startupLineBody() throws -> String {
        let source = try strippedSource(
            spikeRelativeComponents: ["Sources", "fastmlx-serve", "FastMLXServe.swift"])
        let marker = "func startupLine(localAddress:"
        let markerRange = try XCTUnwrap(
            source.range(of: marker),
            "found no `\(marker)` in FastMLXServe.swift (comments stripped)")
        let bodyOpen = try XCTUnwrap(
            source[markerRange.upperBound...].firstIndex(of: "{"),
            "no function body after `\(marker)`")
        let body = try XCTUnwrap(
            balancedText(in: source, from: bodyOpen, open: "{", close: "}"),
            "unbalanced braces in startupLine")
        XCTAssertTrue(
            body.contains("case .scalar(let report):") && body.contains("case .continuous(let report):")
                && body.contains("listening=\\(localAddress)"),
            "the extracted startupLine body is not the whole function; fix this pin rather than "
                + "trusting its other assertions.")
        return body
    }

    private func occurrences(of needle: String, in text: String) -> Int {
        var count = 0
        var range = text.startIndex..<text.endIndex
        while let hit = text.range(of: needle, range: range) {
            count += 1
            range = hit.upperBound..<text.endIndex
        }
        return count
    }

    func testStartupLineEmitsDefaultSamplingAnnouncementInBothScalarAndContinuousBranches() throws {
        let body = try startupLineBody()
        let scalarMarker = "case .scalar(let report):"
        let continuousMarker = "case .continuous(let report):"
        let scalarRange = try XCTUnwrap(body.range(of: scalarMarker))
        let continuousRange = try XCTUnwrap(body.range(of: continuousMarker))
        XCTAssertLessThan(scalarRange.lowerBound, continuousRange.lowerBound)
        let scalarBranch = String(body[scalarRange.upperBound..<continuousRange.lowerBound])
        let continuousBranch = String(body[continuousRange.upperBound...])
        let call = "ServingDefaultSamplingAnnouncement.machineReadableFields("
        XCTAssertEqual(
            occurrences(of: call, in: body), 2,
            "startupLine must call `\(call)` exactly twice (scalar and continuous branches)")
        for (name, branch) in [("scalar", scalarBranch), ("continuous", continuousBranch)] {
            XCTAssertEqual(
                occurrences(of: call, in: branch), 1,
                "the \(name) startup-line branch does not emit the default-sampling announcement")
            let compact = String(branch.filter { !$0.isWhitespace })
            XCTAssertTrue(
                compact.contains(
                    "\(call)source:defaultSampling,resolved:report.samplingDefaults)"),
                "the \(name) branch must announce `source: defaultSampling, resolved: "
                    + "report.samplingDefaults` (the resolved defaults on the report).")
            // Placement: the announcement token group sits before the fit fragment so
            // `listening=` stays last.
            let announcementInterpolation = "\\(defaultSamplingFields)"
            let fitInterpolation = "\\(fit)"
            let announcementAt = branch.range(of: announcementInterpolation)
            let fitAt = branch.range(of: fitInterpolation)
            XCTAssertNotNil(
                announcementAt,
                "the \(name) branch's returned line never interpolates the announcement")
            XCTAssertNotNil(fitAt)
            if let announcementAt, let fitAt {
                XCTAssertLessThan(
                    announcementAt.lowerBound, fitAt.lowerBound,
                    "the \(name) announcement must precede the fit fragment")
            }
        }
    }

    /// Every real construction of `<typeName>(` in `fileComponents`' source forwards the SAME
    /// value it assigns to `backendConfiguration.samplingDefaults` as `samplingDefaults:`.
    private func assertReportForwardsResolvedDefaults(
        file: String, typeName: String
    ) throws {
        let source = try strippedSource(
            spikeRelativeComponents: ["Sources", "SpikeServingAdapters", file])
        let assignment = "backendConfiguration.samplingDefaults ="
        let assignmentRange = try XCTUnwrap(
            source.range(of: assignment),
            "\(file) has no `\(assignment)` to derive the resolved-defaults binding from")
        let lineEnd = source[assignmentRange.upperBound...].firstIndex(of: "\n") ?? source.endIndex
        let resolvedName = source[assignmentRange.upperBound..<lineEnd]
            .trimmingCharacters(in: .whitespaces)
        XCTAssertFalse(resolvedName.isEmpty)
        XCTAssertNotEqual(resolvedName, "nil", "\(file) assigns nil defaults to the backend")
        let marker = "\(typeName)("
        var sites: [String] = []
        var searchRange = source.startIndex..<source.endIndex
        while let markerRange = source.range(of: marker, range: searchRange) {
            searchRange = markerRange.upperBound..<source.endIndex
            if markerRange.lowerBound > source.startIndex,
                isIdentifierCharacter(source[source.index(before: markerRange.lowerBound)])
            {
                continue
            }
            let openParen = source.index(before: markerRange.upperBound)
            sites.append(
                try XCTUnwrap(
                    balancedText(in: source, from: openParen, open: "(", close: ")"),
                    "unbalanced parens in a \(marker) call"))
        }
        // The `public init(` declaration is not a `<typeName>(` call, so every hit is a real
        // construction; zero hits means the pin went vacuous.
        XCTAssertFalse(
            sites.isEmpty,
            "found zero \(marker) construction sites in \(file); fix this pin rather than "
                + "letting it pass vacuously.")
        for (offset, site) in sites.enumerated() {
            let compact = String(site.filter { !$0.isWhitespace })
            let needle = "samplingDefaults:\(resolvedName)"
            var forwarded = false
            var range = compact.startIndex..<compact.endIndex
            while let hit = compact.range(of: needle, range: range) {
                range = hit.upperBound..<compact.endIndex
                let after = hit.upperBound < compact.endIndex ? compact[hit.upperBound] : Character(" ")
                if after == "," || after == ")" {
                    forwarded = true
                    break
                }
            }
            XCTAssertTrue(
                forwarded,
                "\(marker) site #\(offset) in \(file) does not pass `samplingDefaults: "
                    + "\(resolvedName)` (the value assigned to the backend configuration); the "
                    + "startup line would announce `default_sampling=off` while defaults are active.")
        }
    }

    func testScalarStartupReportConstructionForwardsResolvedDefaults() throws {
        try assertReportForwardsResolvedDefaults(
            file: "MLXScalarServing.swift", typeName: "ScalarServingModelStartupReport")
    }

    func testContinuousStartupReportConstructionForwardsResolvedDefaults() throws {
        try assertReportForwardsResolvedDefaults(
            file: "MLXContinuousServing.swift", typeName: "ContinuousServingModelStartupReport")
    }
}
