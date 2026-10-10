import Foundation
import XCTest
@testable import HarnessCore

/// Swift side of the shared card-resolution conformance table
/// (`Fixtures/card-resolution-conformance-v1.json`). The same table is checked against the Python
/// `fastmlx_launch.resolve_card` by `scripts/tests/test_card_resolution_conformance.py`. This engine
/// is the authority; if the two drift, `fastmlx recommend` judges a pack with a rule the engine does
/// not apply.
///
/// Cards are NOT built as structs: each case's `cards` array is written into a manifest envelope and
/// loaded through the engine's own decoder (`QualityCardStore.loadManifestDetailed`), so a fixture
/// card the decoder would drop (or decode differently) fails here instead of silently shrinking the
/// pool. Scope held fixed: Python's `engine_build_commit` tiebreak and `residency` argument have no
/// counterpart in `resolve`; the host class is injected on both sides.
final class CardResolutionConformanceTests: XCTestCase {
    private struct Table: Decodable {
        struct Ambiguous: Decodable, Equatable {
            let repo: [String]
            let pin: [String]
        }
        struct Expect: Decodable, Equatable {
            let none: Bool
            let card: String?
            let ambiguous: Ambiguous?

            // `expect` is the string "none", {"card": id}, or {"ambiguous": {...}}.
            init(from decoder: Decoder) throws {
                if let single = try? decoder.singleValueContainer().decode(String.self) {
                    guard single == "none" else {
                        throw DecodingError.dataCorrupted(
                            .init(codingPath: decoder.codingPath, debugDescription: "bad expect \(single)"))
                    }
                    none = true
                    card = nil
                    ambiguous = nil
                    return
                }
                let container = try decoder.container(keyedBy: Keys.self)
                none = false
                card = try container.decodeIfPresent(String.self, forKey: .card)
                ambiguous = try container.decodeIfPresent(Ambiguous.self, forKey: .ambiguous)
            }

            private enum Keys: String, CodingKey { case card, ambiguous }
        }
        struct Divergence: Decodable {
            let pythonOutcome: String
            let reason: String
        }
        struct Case: Decodable {
            let id: String
            let why: String
            let repo: String?
            let revision: String?
            let hostHardwareClass: String?
            let expect: Expect
            let knownPythonDivergence: Divergence?
        }
        let schema: String
        let cases: [Case]
    }

    private static let fixtureName = "card-resolution-conformance-v1"

    private func fixtureURL() throws -> URL {
        try XCTUnwrap(Bundle.module.url(forResource: Self.fixtureName, withExtension: "json"))
    }

    private func table() throws -> Table {
        try JSONDecoder().decode(Table.self, from: Data(contentsOf: fixtureURL()))
    }

    /// The raw `cards` array of every case, keyed by case id, as JSON-serialized objects so the
    /// engine decoder (not this test) interprets them.
    private func rawCards() throws -> [String: [Any]] {
        let root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: fixtureURL())) as? [String: Any])
        let cases = try XCTUnwrap(root["cases"] as? [[String: Any]])
        var out: [String: [Any]] = [:]
        for c in cases {
            out[try XCTUnwrap(c["id"] as? String)] = try XCTUnwrap(c["cards"] as? [Any])
        }
        return out
    }

    /// Writes `cards` into a manifest envelope and loads it with the engine's own manifest decoder.
    /// Asserts nothing was dropped, so every fixture card is really in the pool under test.
    private func decode(_ cards: [Any], id: String, dir: URL) throws -> [QualityCard] {
        let manifest: [String: Any] = ["schema": "fast-mlx-quality-card-v1", "cards": cards]
        let url = dir.appendingPathComponent("\(UUID().uuidString).json")
        try JSONSerialization.data(withJSONObject: manifest).write(to: url)
        let loaded = try QualityCardStore.loadManifestDetailed(contentsOf: url)
        XCTAssertEqual(loaded.droppedCardCount, 0, "\(id): the engine decoder dropped a fixture card")
        XCTAssertEqual(loaded.cards.count, cards.count, "\(id): decoded card count")
        return loaded.cards
    }

    func testEngineMatchesTheSharedTable() throws {
        let table = try table()
        XCTAssertEqual(table.schema, Self.fixtureName)
        XCTAssertGreaterThan(table.cases.count, 50, "the table must not be silently truncated")
        let ids = table.cases.map(\.id)
        XCTAssertEqual(ids.count, Set(ids).count, "duplicate case ids")
        let raw = try rawCards()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("cardres-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        var failures: [String] = []
        for c in table.cases {
            let cards = try decode(try XCTUnwrap(raw[c.id]), id: c.id, dir: dir)
            let actual = QualityCardStore.resolve(
                repo: c.repo, revision: c.revision, hostHardwareClass: c.hostHardwareClass, in: cards)
            switch actual {
            case .none:
                if !c.expect.none {
                    failures.append("\(c.id): expected \(c.expect) but got none -- \(c.why)")
                }
            case .resolved(let card):
                if c.expect.card != card.id {
                    failures.append("\(c.id): expected \(c.expect) but got card \(card.id) -- \(c.why)")
                }
            case .ambiguous(let repoIDs, let pinIDs):
                if c.expect.ambiguous != Table.Ambiguous(repo: repoIDs, pin: pinIDs) {
                    failures.append(
                        "\(c.id): expected \(c.expect) but got ambiguous \(repoIDs) / \(pinIDs) -- \(c.why)")
                }
            }
        }
        XCTAssertTrue(failures.isEmpty, "\(failures.count) case(s) diverge:\n" + failures.joined(separator: "\n"))
    }

    func testTableCoversEveryOutcomeKind() throws {
        let cases = try table().cases
        XCTAssertTrue(cases.contains { $0.expect.none }, "no none case")
        XCTAssertTrue(cases.contains { $0.expect.card != nil }, "no resolved-card case")
        XCTAssertTrue(cases.contains { $0.expect.ambiguous != nil }, "no ambiguous case")
        XCTAssertTrue(cases.contains { $0.knownPythonDivergence != nil }, "no python-refuses case")
    }

    func testKnownPythonDivergenceIsOnlyPythonRefusesWhereTheEnginePicksACard() throws {
        for c in try table().cases {
            guard let known = c.knownPythonDivergence else { continue }
            XCTAssertEqual(known.pythonOutcome, "refuses", c.id)
            XCTAssertNotNil(c.expect.card, "\(c.id): a divergence may only excuse python refusing where the engine picks a card")
            XCTAssertFalse(known.reason.isEmpty, c.id)
        }
    }
}
