import XCTest

/// Structural (source-text) regression pin for the `--offload-plan-check-only` silent-downgrade
/// defect: `loadScalarServingBackend` in `fastmlx-serve/FastMLXServe.swift` used to construct
/// `ScalarServingModelLoadConfiguration(...)` WITHOUT forwarding the parsed
/// `arguments.offloadPlanCheckOnly` flag. `ScalarServingModelLoadConfiguration.offloadPlanCheckOnly`
/// defaults to `false`, so the omission silently downgraded the advertised "verify the offload plan
/// on this host and exit without loading weights" dry run into a full server start — binding the
/// configured port and sitting in the run loop until SIGTERM. On a production host that means a
/// second model loaded beside the live incumbent, on the incumbent's own port.
///
/// **This is a source-text gate, not a behavioral one.** `fastmlx-serve` is an executable target
/// with NO test target (see `FitCompositionStackTests.swift`'s doc comment for the same structural
/// blind spot, described there for a different call site with the identical shape), so no Swift
/// test in this repository can invoke `loadScalarServingBackend` or `FastMLXServe.main()` directly
/// and observe that the dry run actually exits before loading weights. That blind spot is precisely
/// why this defect shipped and stayed live: `MLXScalarServingTests.swift` exhaustively covers
/// `ScalarServingModelLoadConfiguration.offloadPlanCheckOnly` by constructing the configuration type
/// DIRECTLY with `offloadPlanCheckOnly: true`, which proves the gate itself is correct while proving
/// nothing about whether the one real call site in `FastMLXServe.swift` ever sets that field from the
/// parsed CLI flag. This test reads the ACTUAL source text of that call site and asserts every
/// `ScalarServingModelLoadConfiguration(` construction forwards `offloadPlanCheckOnly:` — it cannot
/// prove the dry run exits cleanly, only that the flag reaches the load path at all. What would
/// defeat this test: a second `ScalarServingModelLoadConfiguration(` construction site added
/// elsewhere in this file (or in a different file this test does not scan) that omits the label the
/// same way the original defect did — this test only ever inspects `FastMLXServe.swift`.
///
/// Naming stays family-neutral throughout (no model-family CamelCase token): this repository's
/// public-projection validator fails closed on such markers in projected source, and
/// `offload`/`plan`/`ngram` is already the established neutral vocabulary for this feature.
final class OffloadPlanCheckOnlyServeWiringStructuralTests: XCTestCase {

    /// The exact construction-site marker this test scans for. Kept as a single source of truth so
    /// the counting logic and the assertion messages below can never drift from each other.
    private let constructionMarker = "ScalarServingModelLoadConfiguration("

    /// The label every construction site found above must forward. Reusing `arguments.` as a
    /// literal prefix would overfit to today's exact call-site spelling (`arguments.` vs. a future
    /// local binding); checking for the label alone is the minimal robust signal that the parsed
    /// flag is actually threaded into the configuration, without over-engineering a Swift parser for
    /// what is, deliberately, a bounded text scan.
    private let forwardedLabel = "offloadPlanCheckOnly:"

    /// Resolve `fastmlx-serve/FastMLXServe.swift` by SEARCHING ancestors of this test file's own
    /// compile-time `#filePath`, not by counting a fixed number of directory levels. The fleet sync
    /// script deploys `spike/`'s CONTENTS as the package root on fleet hosts, so a fixed level-count
    /// walk that works in this checkout lands on a path that does not exist there — see the
    /// identical precedent and rationale in
    /// `FastMLXServeArgumentsTests.testFastMLXServeArgumentErrorCatchArmCallSitePinExists`. Both the
    /// checked-out layout (`spike/Sources/fastmlx-serve/...`) and the fleet-synced layout
    /// (`Sources/fastmlx-serve/...`) are accepted; neither found is a hard failure, never a silent
    /// skip.
    private func resolveFastMLXServeSourceFile() -> URL? {
        let candidateSuffixes = [
            ["spike", "Sources", "fastmlx-serve", "FastMLXServe.swift"],
            ["Sources", "fastmlx-serve", "FastMLXServe.swift"],
        ]
        var searchDirectory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<8 {
            for suffix in candidateSuffixes {
                let candidate = suffix.reduce(searchDirectory) { $0.appendingPathComponent($1) }
                if FileManager.default.fileExists(atPath: candidate.path) {
                    return candidate
                }
            }
            searchDirectory.deleteLastPathComponent()
        }
        return nil
    }

    /// Extract the balanced-paren text of a single call starting at its opening `(`, so the
    /// forwarded-label check below inspects exactly one construction site's own argument list —
    /// never spilling into a sibling call or the surrounding function body. A bounded-window text
    /// scan (rather than a real Swift parser) is deliberately proportionate here: the call sites
    /// this test inspects are simple, non-nested-paren-heavy struct initializers.
    private func balancedCallText(in source: String, openParenIndex: String.Index) -> String? {
        var depth = 0
        var index = openParenIndex
        while index < source.endIndex {
            let character = source[index]
            if character == "(" {
                depth += 1
            } else if character == ")" {
                depth -= 1
                if depth == 0 {
                    return String(source[openParenIndex...index])
                }
            }
            index = source.index(after: index)
        }
        return nil
    }

    /// Strip Swift line comments (`//` to end of line) from `source`, one line at a time, keeping
    /// the newline that separated each line so nothing downstream shifts position. Exists so the
    /// construction-site scan below never counts a comment that merely *mentions* the marker text --
    /// `FastMLXServe.swift` itself carries such a comment, reminding future editors to forward this
    /// field at any new construction site, and without this stripping that reminder sentence was
    /// itself miscounted as a second construction site.
    ///
    /// **Known limitation, stated rather than silently handled:** this is a naive per-line `//`
    /// removal, not a Swift lexer -- it does not know about string literals, so a `//` that appears
    /// inside a Swift string literal would be (incorrectly) treated as the start of a comment and
    /// everything after it on that line would be dropped. None of the text this test scans today
    /// contains such a literal, so the limitation is not currently reachable, but a future edit to
    /// `FastMLXServe.swift` that puts `//` inside a string on the same line as the construction-site
    /// marker could trip it. Fixing that properly needs a real lexer, which is disproportionate for
    /// this bounded text scan.
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

    /// Every `ScalarServingModelLoadConfiguration(` construction site found in
    /// `fastmlx-serve/FastMLXServe.swift`, as balanced-paren call text. Comments are stripped from
    /// `source` (see `stripLineComments(from:)`) before this scan runs, so prose that merely mentions
    /// the marker text is not counted as a real construction site.
    private func constructionSites(in source: String) -> [String]? {
        let source = stripLineComments(from: source)
        var sites: [String] = []
        var searchRange = source.startIndex..<source.endIndex
        while let markerRange = source.range(of: constructionMarker, range: searchRange) {
            // `markerRange.upperBound` sits immediately after the marker's own trailing "(", so
            // stepping back one character lands exactly ON that "(" -- the correct start for the
            // balanced-paren scan below (its own first character re-opens depth 1).
            let openParenIndex = source.index(before: markerRange.upperBound)
            guard let call = balancedCallText(in: source, openParenIndex: openParenIndex) else {
                return nil
            }
            sites.append(call)
            searchRange = markerRange.upperBound..<source.endIndex
        }
        return sites
    }

    /// The regression pin itself. Fails closed at every step (missing file, empty file, zero
    /// construction sites found, unbalanced parens, any site missing the forwarded label) rather
    /// than vacuously passing — this repository has been bitten before by gates that are reachable
    /// but inert (see the anti-vacuity convention documented across this test suite), and a
    /// structural gate that finds zero call sites and "passes" on an empty forall is exactly that
    /// failure mode.
    func testEveryScalarServingModelLoadConfigurationConstructionForwardsOffloadPlanCheckOnly() {
        guard let sourceFile = resolveFastMLXServeSourceFile() else {
            XCTFail(
                "could not locate fastmlx-serve/FastMLXServe.swift by walking up from #filePath "
                    + "(\(#filePath)); the repo/package layout moved relative to this test file. "
                    + "Fix resolveFastMLXServeSourceFile() rather than letting this pin go silent.")
            return
        }

        let source: String
        do {
            source = try String(contentsOf: sourceFile, encoding: .utf8)
        } catch {
            XCTFail(
                "could not read FastMLXServe.swift from the #filePath-derived path "
                    + "\(sourceFile.path): \(error)")
            return
        }

        XCTAssertFalse(
            source.isEmpty,
            "FastMLXServe.swift resolved to \(sourceFile.path) but read as empty content -- "
                + "resolution likely landed on the wrong file")

        guard let sites = constructionSites(in: source) else {
            XCTFail(
                "found a `\(constructionMarker)` marker in \(sourceFile.path) whose parentheses "
                    + "never balance -- the bounded-window scan in balancedCallText(in:openParenIndex:) "
                    + "needs to be revisited, or the source has unbalanced parens")
            return
        }

        // Anti-vacuity: a forall over zero construction sites would trivially "pass" below without
        // ever proving anything, and today's driver has exactly one real construction site (with
        // comment text that merely mentions the marker already stripped out above, so it no longer
        // inflates this count). Pin the count exactly rather than only a lower bound: an exact pin
        // fails loudly both on zero sites (marker drifted, or the load path was removed) and on an
        // unexpected second site, instead of letting a second, unreviewed site slip through into the
        // forall below and surface only as a confusing "N of M forward the label" failure.
        XCTAssertEqual(
            sites.count, 1,
            "found \(sites.count) `\(constructionMarker)` construction site(s) in \(sourceFile.path) "
                + "(expected exactly 1). If this is ZERO: either the marker text drifted "
                + "(rename/reformat) and this scan needs updating, or the scalar serving load path "
                + "was removed entirely -- either way this pin proves nothing until the marker "
                + "resolves to a real call site. If this is MORE than 1: a new "
                + "`\(constructionMarker)` construction site was deliberately added elsewhere in this "
                + "file (for example a new serving route) -- confirm the new site also forwards "
                + "`\(forwardedLabel)`, then update this expected count to match, deliberately, rather "
                + "than loosening this assertion back to a lower bound.")

        let sitesForwardingTheFlag = sites.filter { $0.contains(forwardedLabel) }

        XCTAssertEqual(
            sitesForwardingTheFlag.count, sites.count,
            "of \(sites.count) `\(constructionMarker)` construction site(s) in "
                + "\(sourceFile.path), only \(sitesForwardingTheFlag.count) forward "
                + "`\(forwardedLabel)`. Every construction site MUST forward the parsed "
                + "--offload-plan-check-only flag: omitting it silently keeps the configuration's "
                + "`false` default, downgrading the advertised dry run into a full server start that "
                + "binds the configured port and never exits until SIGTERM -- the exact deploy-safety "
                + "defect this test exists to catch. See loadScalarServingBackend in "
                + "fastmlx-serve/FastMLXServe.swift.")
    }

    // MARK: - The general form of the same defect

    /// Fields on `FastMLXServeArguments` that the serve driver is NOT expected to read, each with a
    /// stated reason. Empty today, and that is the point: at the time this gate was written every
    /// single parsed field was consumed by the driver, so the invariant below holds exactly rather
    /// than approximately. Adding an entry here is a deliberate, reviewable act — it must come with
    /// a reason a reader can check, not merely silence a red test.
    private let fieldsIntentionallyUnreadByTheDriver: [String: String] = [:]

    /// The GENERAL form of the `--offload-plan-check-only` defect, which the test above pins only in
    /// its one specific instance.
    ///
    /// That defect was not "someone mistyped a label". It was that a CLI flag can be fully parsed,
    /// cross-validated against other flags, given its own error cases and its own passing unit
    /// tests, and then never read by the serve driver at all — with nothing anywhere turning red.
    /// The flag-specific gate above would not have caught the same mistake made to any OTHER field,
    /// and there are 30+ of them.
    ///
    /// So this asserts the whole-surface invariant: every `public let` field on
    /// `FastMLXServeArguments` is referenced as `arguments.<field>` at least once in the driver.
    /// `fastmlx-serve` is a single-file executable target (`FastMLXServe.swift` is its only source
    /// file), so scanning that one file is complete coverage of the driver rather than a sample.
    ///
    /// **What this proves and what it does not.** It proves a parsed field is not dropped on the
    /// floor entirely — the exact failure mode that shipped. It does NOT prove the field is used
    /// *correctly*, threaded to the right place, or honored at runtime; a reference inside a
    /// comment or a dead branch would satisfy it. It is a floor, not a ceiling, and it is
    /// deliberately cheap enough to be exhaustive. Validated against real history: run against the
    /// commit immediately before the fix, this invariant flags exactly `offloadPlanCheckOnly` and
    /// nothing else — no false positives across the other 36 fields.
    ///
    /// The `arguments.<field>` match is identifier-boundary bounded (see
    /// `containsIdentifierBoundedOccurrence(of:in:)`), not a bare substring `contains`. Field names on
    /// this struct collide by prefix -- `maximumCompletionTokens` is itself a prefix of
    /// `maximumCompletionTokensWasExplicit`, and likewise for `defaultCompletionTokens` /
    /// `defaultCompletionTokensWasExplicit` -- so a bare substring check would let a read of the
    /// `…WasExplicit` companion field count as proof the base field is read too, even if every real
    /// read of the base field were deleted.
    func testEveryParsedServeArgumentFieldIsReadSomewhereByTheServeDriver() {
        guard let driverFile = resolveFastMLXServeSourceFile(),
            let argumentsFile = resolveSourceFile(
                suffixes: [
                    ["spike", "Sources", "ServingCore", "FastMLXServeArguments.swift"],
                    ["Sources", "ServingCore", "FastMLXServeArguments.swift"],
                ])
        else {
            XCTFail(
                "could not locate FastMLXServe.swift and/or FastMLXServeArguments.swift by walking "
                    + "up from #filePath (\(#filePath)); fix the resolver rather than letting this "
                    + "pin go silent.")
            return
        }

        guard let driverSource = try? String(contentsOf: driverFile, encoding: .utf8),
            let argumentsSource = try? String(contentsOf: argumentsFile, encoding: .utf8)
        else {
            XCTFail("could not read \(driverFile.path) and/or \(argumentsFile.path)")
            return
        }

        // Scope field extraction to the `FastMLXServeArguments` struct itself. The same file also
        // declares several enums ABOVE it; the struct is the last top-level type, so everything
        // after its declaration belongs to it. Without this scoping the scan could attribute an
        // unrelated type's property to the arguments surface and demand the driver read it.
        guard let structDeclarationRange = argumentsSource.range(
            of: "public struct FastMLXServeArguments")
        else {
            XCTFail(
                "could not find the `public struct FastMLXServeArguments` declaration in "
                    + "\(argumentsFile.path) -- the type was renamed and this scan needs updating.")
            return
        }
        let structBody = argumentsSource[structDeclarationRange.lowerBound...]

        var fields: [String] = []
        for line in structBody.split(separator: "\n", omittingEmptySubsequences: false) {
            // Exactly one level of indentation: the struct's own stored properties, not the nested
            // declarations or locals inside its methods (which are more deeply indented).
            guard line.hasPrefix("    public let ") else { continue }
            let afterKeyword = line.dropFirst("    public let ".count)
            let name = String(afterKeyword.prefix { $0.isLetter || $0.isNumber })
            if !name.isEmpty {
                fields.append(name)
            }
        }

        // Anti-vacuity: if the extraction pattern ever stops matching (a reformat, a rename, a
        // property-wrapper migration), `fields` goes empty and the forall below passes while
        // proving nothing at all -- the precise inert-gate failure this repository keeps re-learning.
        XCTAssertGreaterThan(
            fields.count, 20,
            "extracted only \(fields.count) `public let` field(s) from FastMLXServeArguments "
                + "(expected 30+). The extraction pattern has drifted and this gate is now inert -- "
                + "fix the scan, do not lower this bound.")

        let unread = fields.filter { field in
            fieldsIntentionallyUnreadByTheDriver[field] == nil
                && !containsIdentifierBoundedOccurrence(of: "arguments.\(field)", in: driverSource)
        }

        XCTAssertEqual(
            unread, [],
            "these parsed FastMLXServeArguments field(s) are NEVER read by the serve driver "
                + "(\(driverFile.path)): \(unread.joined(separator: ", ")). A field that is parsed "
                + "and validated but never consumed is silently inert -- this is exactly how "
                + "--offload-plan-check-only shipped as a dry run that instead started a full "
                + "server on the operator's configured port. Either thread the field through to "
                + "the behavior it promises, or add it to fieldsIntentionallyUnreadByTheDriver "
                + "with a reason a reviewer can verify.")
    }

    /// True when `text` contains `needle` at an occurrence not immediately followed by another
    /// Swift identifier character (letter, digit, or `_`). Plain substring `contains` would match
    /// `arguments.maximumCompletionTokens` inside the longer
    /// `arguments.maximumCompletionTokensWasExplicit`, so a field-read check built on bare `contains`
    /// cannot tell "the base field is read" apart from "only its `…WasExplicit` companion field is
    /// read" -- exactly the prefix collision the whole-surface gate above exists to not miss. This
    /// is still a bounded text scan, not a parser: it only forbids the character immediately after
    /// the match from extending the identifier, which is sufficient here because `needle` always
    /// starts with `arguments.` and Swift identifiers cannot contain characters other than letters,
    /// digits, and `_` once started.
    private func containsIdentifierBoundedOccurrence(of needle: String, in text: String) -> Bool {
        var searchRange = text.startIndex..<text.endIndex
        while let matchRange = text.range(of: needle, range: searchRange) {
            let characterAfterMatch = matchRange.upperBound
            let isIdentifierBoundary =
                characterAfterMatch == text.endIndex
                || !isIdentifierCharacter(text[characterAfterMatch])
            if isIdentifierBoundary {
                return true
            }
            searchRange = matchRange.upperBound..<text.endIndex
        }
        return false
    }

    private func isIdentifierCharacter(_ character: Character) -> Bool {
        character.isLetter || character.isNumber || character == "_"
    }

    /// Generic ancestor search shared by the gates above. Same rationale as
    /// `resolveFastMLXServeSourceFile()`: search upward for a known path suffix instead of counting
    /// a fixed number of `..` levels, because the fleet sync makes `spike/`'s CONTENTS the package
    /// root and a level-count that is right here is wrong there.
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
}
