import Foundation
import XCTest

import HarnessCore
import ServingCore
@testable import SpikeServingAdapters

/// `ContinuousServingRowTokenMask` is the adapter between a `TokenConstraint` and the continuous-batch
/// runtime's `ContinuousBatchRowTokenMask`. It carries the SAME thinking-phase gate and failure
/// recording as the scalar route's `ConstraintMaskingLogitProcessor`, but answers `nil` ("unconstrained
/// for this step") instead of editing logits, and never throws (one row's failure must not fail a batch
/// step other rows share).
final class ContinuousServingRowTokenMaskTests: XCTestCase {
    private static let eosID = 256
    private static let thinkEndID = 257

    private static func bit(_ id: Int, in bitset: [UInt64]) -> Bool {
        let word = id >> 6
        guard word < bitset.count else { return false }
        return bitset[word] & (1 << UInt64(id & 63)) != 0
    }

    /// Byte vocabulary: ids 0...255 raw bytes, 256 EOS, 257 `</think>` (banned for JSON).
    private func booleanConstraint() async throws -> JSONSchemaTokenConstraint {
        var classifications: [TokenByteClassification] = (0...255).map { .bytes([UInt8($0)]) }
        classifications.append(.eos)
        classifications.append(.banned)
        let objectSupport = ScalarServingJSONObjectConstraintSupport(
            classifications: classifications, thinkEndTokenID: Self.thinkEndID)
        let support = try XCTUnwrap(loadScalarServingJSONSchemaConstraintSupport(sharing: objectSupport))
        let raw = try JSONSchemaSubsetCompiler.parseOrdered(Data(#"{"type":"boolean"}"#.utf8))
        let format = try JSONSchemaSubsetCompiler.compile(name: "flag", schema: raw, strict: true)
        return JSONSchemaTokenConstraint(table: await support.table, format: format)
    }

    private func t() -> Int { Int(UInt8(ascii: "t")) }
    private func f() -> Int { Int(UInt8(ascii: "f")) }
    private func x() -> Int { Int(UInt8(ascii: "x")) }

    func testActiveFromStartAnswersTheConstraintBitsetAndAdvancesWithEachToken() async throws {
        let mask = ContinuousServingRowTokenMask(
            constraint: try await booleanConstraint(), activeFromStart: true, thinkEndTokenID: nil)

        var bitset = try XCTUnwrap(mask.allowedTokenBitset())
        XCTAssertTrue(Self.bit(t(), in: bitset))
        XCTAssertTrue(Self.bit(f(), in: bitset))
        XCTAssertFalse(Self.bit(x(), in: bitset))

        mask.advance(token: t())
        bitset = try XCTUnwrap(mask.allowedTokenBitset())
        XCTAssertTrue(Self.bit(Int(UInt8(ascii: "r")), in: bitset), "after `t` only `r` continues `true`")
        XCTAssertFalse(Self.bit(t(), in: bitset))
        XCTAssertNil(mask.recordedFailure)
    }

    /// Mirrors `ConstraintMaskingLogitProcessor`: inactive until `thinkEndTokenID` is emitted; while
    /// inactive the constraint is NOT advanced (a reasoning token that the grammar would reject must
    /// not fail the request), and the think-end token itself is not fed to the constraint.
    func testThinkingGateIsInactiveUntilThinkEndAndDoesNotAdvanceTheConstraintBeforeIt() async throws {
        let mask = ContinuousServingRowTokenMask(
            constraint: try await booleanConstraint(), activeFromStart: false,
            thinkEndTokenID: Self.thinkEndID)

        XCTAssertNil(mask.allowedTokenBitset(), "inactive: unconstrained")
        mask.advance(token: x())  // reasoning text the grammar would reject
        XCTAssertNil(mask.allowedTokenBitset())
        XCTAssertNil(mask.recordedFailure, "an inactive mask must not advance (or fail) the constraint")

        mask.advance(token: Self.thinkEndID)
        let bitset = try XCTUnwrap(mask.allowedTokenBitset(), "active after the think-end token")
        XCTAssertTrue(Self.bit(t(), in: bitset), "the constraint is still at its initial state")
        XCTAssertFalse(Self.bit(x(), in: bitset))
        XCTAssertNil(mask.recordedFailure)
    }

    func testThinkingGateWithoutAThinkEndTokenNeverActivates() async throws {
        let mask = ContinuousServingRowTokenMask(
            constraint: try await booleanConstraint(), activeFromStart: false, thinkEndTokenID: nil)
        mask.advance(token: Self.thinkEndID)
        mask.advance(token: t())
        XCTAssertNil(mask.allowedTokenBitset())
        XCTAssertNil(mask.recordedFailure)
    }

    func testDisallowedTokenRecordsAFailureAndTheMaskIsNilForever() async throws {
        let mask = ContinuousServingRowTokenMask(
            constraint: try await booleanConstraint(), activeFromStart: true, thinkEndTokenID: nil)

        mask.advance(token: x())
        let failure = try XCTUnwrap(mask.recordedFailure, "a disallowed token must record a failure")
        XCTAssertNil(mask.allowedTokenBitset(), "after a failure the row is unconstrained")

        mask.advance(token: t())  // would be legal from the initial state; must be a no-op now
        XCTAssertNil(mask.allowedTokenBitset())
        XCTAssertEqual(
            String(describing: try XCTUnwrap(mask.recordedFailure)), String(describing: failure),
            "the FIRST failure is kept")
    }

    func testAllZeroBitsetRecordsAFailureAndAnswersNil() {
        let mask = ContinuousServingRowTokenMask(
            constraint: StubConstraint(bitset: [0, 0]), activeFromStart: true, thinkEndTokenID: nil)
        XCTAssertNil(mask.recordedFailure)
        XCTAssertNil(mask.allowedTokenBitset())
        XCTAssertNotNil(mask.recordedFailure, "an empty allowed set is a constraint failure")
        XCTAssertNil(mask.allowedTokenBitset(), "and stays nil")
    }

    func testThrowingBitsetRecordsTheThrownFailure() {
        let mask = ContinuousServingRowTokenMask(
            constraint: StubConstraint(bitset: nil), activeFromStart: true, thinkEndTokenID: nil)
        XCTAssertNil(mask.allowedTokenBitset())
        XCTAssertTrue(mask.recordedFailure is StubConstraint.Boom)
    }

    func testNonEmptyBitsetFromAStubIsPassedThroughVerbatim() {
        let mask = ContinuousServingRowTokenMask(
            constraint: StubConstraint(bitset: [0b1010, 1]), activeFromStart: true, thinkEndTokenID: nil)
        XCTAssertEqual(mask.allowedTokenBitset(), [0b1010, 1])
        XCTAssertNil(mask.recordedFailure)
    }
}

private struct StubConstraint: TokenConstraint {
    struct Boom: Error {}
    /// `nil` = `allowedTokenBitset()` throws.
    let bitset: [UInt64]?

    func allowedTokenIds() throws -> [Int] { [] }
    func allowedTokenBitset() throws -> [UInt64] {
        guard let bitset else { throw Boom() }
        return bitset
    }
    mutating func advance(token: Int) throws {}
    var isComplete: Bool { false }
}
