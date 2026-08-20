import XCTest
@testable import HarnessCore

final class ToolCallReliabilityTests: XCTestCase {
    private let corpus = ToolCallReliabilityScore.defaultToolCallReliabilityCorpus

    /// A perfect observation set: every expected call fires with the correct name and complete
    /// arguments, the chit-chat case emits nothing, and nothing needed repair.
    private func perfectObservations() -> [ToolCallObservation] {
        [
            ToolCallObservation(
                id: "weather-single-city", emittedToolName: "get_weather",
                argumentsJSON: #"{"location":"Boston"}"#, neededRepair: false),
            ToolCallObservation(
                id: "verify-identity-two-arg", emittedToolName: "verify_identity",
                argumentsJSON: #"{"email":"a@b.com","order_number":"123"}"#, neededRepair: false),
            ToolCallObservation(
                id: "chit-chat-no-call", emittedToolName: nil, argumentsJSON: nil, neededRepair: false),
            ToolCallObservation(
                id: "list-supported-timezones", emittedToolName: "list_timezones",
                argumentsJSON: "{}", neededRepair: false),
        ]
    }

    func testPerfectObservationsScoreAllOnes() {
        let score = ToolCallReliabilityScore.score(expectations: corpus, observations: perfectObservations())
        XCTAssertEqual(score.caseCount, 4)
        XCTAssertEqual(score.triggerRate, 1.0)
        XCTAssertEqual(score.nameAccuracy, 1.0)
        XCTAssertEqual(score.argumentValidityRate, 1.0)
        XCTAssertEqual(score.repairRate, 0.0)
    }

    func testWrongNameDropsNameAccuracyButKeepsTriggerRate() {
        var observations = perfectObservations()
        observations[0] = ToolCallObservation(
            id: "weather-single-city", emittedToolName: "get_forecast",
            argumentsJSON: #"{"location":"Boston"}"#, neededRepair: false)
        let score = ToolCallReliabilityScore.score(expectations: corpus, observations: observations)
        // A call still fired where one was expected, so trigger rate is unaffected.
        XCTAssertEqual(score.triggerRate, 1.0)
        // 2 of 3 expected-call cases have the correct name.
        XCTAssertEqual(score.nameAccuracy, 2.0 / 3.0, accuracy: 1e-9)
    }

    func testMissedCallDropsTriggerRateAndExcludesFromNameAndArgumentSubsets() {
        var observations = perfectObservations()
        observations[0] = ToolCallObservation(
            id: "weather-single-city", emittedToolName: nil, argumentsJSON: nil, neededRepair: false)
        let score = ToolCallReliabilityScore.score(expectations: corpus, observations: observations)
        // 3 of 4 cases matched expectation (emit-or-withhold).
        XCTAssertEqual(score.triggerRate, 0.75)
        // Of the 3 expected-call cases, the missed one is excluded from the numerator: 2 of 3 correct.
        XCTAssertEqual(score.nameAccuracy, 2.0 / 3.0, accuracy: 1e-9)
        // Of the 3 emitted calls (weather call withheld), all remaining are valid.
        XCTAssertEqual(score.argumentValidityRate, 1.0)
    }

    func testFalsePositiveCallDropsTriggerRate() {
        var observations = perfectObservations()
        observations[2] = ToolCallObservation(
            id: "chit-chat-no-call", emittedToolName: "get_weather",
            argumentsJSON: #"{"location":"Nowhere"}"#, neededRepair: false)
        let score = ToolCallReliabilityScore.score(expectations: corpus, observations: observations)
        // 3 of 4 cases matched expectation.
        XCTAssertEqual(score.triggerRate, 0.75)
        // Name accuracy is unaffected: the false-positive case has no expected name to check.
        XCTAssertEqual(score.nameAccuracy, 1.0)
    }

    func testInvalidArgumentsDropsArgumentValidityRate() {
        var observations = perfectObservations()
        // Missing the required "order_number" key.
        observations[1] = ToolCallObservation(
            id: "verify-identity-two-arg", emittedToolName: "verify_identity",
            argumentsJSON: #"{"email":"a@b.com"}"#, neededRepair: false)
        let score = ToolCallReliabilityScore.score(expectations: corpus, observations: observations)
        // 2 of 3 emitted calls (chit-chat emits none) have fully valid arguments.
        XCTAssertEqual(score.argumentValidityRate, 2.0 / 3.0, accuracy: 1e-9)
        // Trigger rate and name accuracy are unaffected — a call still fired with the right name.
        XCTAssertEqual(score.triggerRate, 1.0)
        XCTAssertEqual(score.nameAccuracy, 1.0)
    }

    func testMalformedJSONArgumentsCountsAsInvalid() {
        var observations = perfectObservations()
        observations[0] = ToolCallObservation(
            id: "weather-single-city", emittedToolName: "get_weather",
            argumentsJSON: #"{"location": "Boston""#, neededRepair: false)
        let score = ToolCallReliabilityScore.score(expectations: corpus, observations: observations)
        XCTAssertEqual(score.argumentValidityRate, 2.0 / 3.0, accuracy: 1e-9)
    }

    func testNilArgumentsOnEmittedCallCountsAsInvalid() {
        var observations = perfectObservations()
        observations[0] = ToolCallObservation(
            id: "weather-single-city", emittedToolName: "get_weather",
            argumentsJSON: nil, neededRepair: false)
        let score = ToolCallReliabilityScore.score(expectations: corpus, observations: observations)
        XCTAssertEqual(score.argumentValidityRate, 2.0 / 3.0, accuracy: 1e-9)
    }

    func testRepairedCallReflectsInRepairRate() {
        var observations = perfectObservations()
        observations[0] = ToolCallObservation(
            id: "weather-single-city", emittedToolName: "get_weather",
            argumentsJSON: #"{"location":"Boston"}"#, neededRepair: true)
        let score = ToolCallReliabilityScore.score(expectations: corpus, observations: observations)
        // 1 of 3 emitted calls needed repair.
        XCTAssertEqual(score.repairRate, 1.0 / 3.0, accuracy: 1e-9)
        // Repair does not affect the other rates.
        XCTAssertEqual(score.triggerRate, 1.0)
        XCTAssertEqual(score.nameAccuracy, 1.0)
        XCTAssertEqual(score.argumentValidityRate, 1.0)
    }

    func testMissingObservationTreatedAsEmittedNothing() {
        // Drop the weather observation entirely rather than supplying an explicit nil-call one.
        let observations = perfectObservations().filter { $0.id != "weather-single-city" }
        let score = ToolCallReliabilityScore.score(expectations: corpus, observations: observations)
        XCTAssertEqual(score.triggerRate, 0.75)
        XCTAssertEqual(score.nameAccuracy, 2.0 / 3.0, accuracy: 1e-9)
    }

    func testUnmatchedObservationIsIgnored() {
        var observations = perfectObservations()
        observations.append(
            ToolCallObservation(
                id: "no-such-expectation", emittedToolName: "get_weather",
                argumentsJSON: #"{"location":"X"}"#, neededRepair: false))
        let score = ToolCallReliabilityScore.score(expectations: corpus, observations: observations)
        XCTAssertEqual(score.caseCount, 4)
        XCTAssertEqual(score.triggerRate, 1.0)
    }

    func testEmptyCorpusIsVacuousPass() {
        let score = ToolCallReliabilityScore.score(expectations: [], observations: [])
        XCTAssertEqual(score.caseCount, 0)
        XCTAssertEqual(score.triggerRate, 1.0)
        XCTAssertEqual(score.nameAccuracy, 1.0)
        XCTAssertEqual(score.argumentValidityRate, 1.0)
        XCTAssertEqual(score.repairRate, 0.0)
    }

    func testNoExpectedCallsGivesVacuousNameAccuracy() {
        let expectations = [
            ToolCallExpectation(id: "a", expectedToolName: nil, requiredArgumentKeys: []),
            ToolCallExpectation(id: "b", expectedToolName: nil, requiredArgumentKeys: []),
        ]
        let observations = [
            ToolCallObservation(id: "a", emittedToolName: nil, argumentsJSON: nil, neededRepair: false),
            ToolCallObservation(id: "b", emittedToolName: nil, argumentsJSON: nil, neededRepair: false),
        ]
        let score = ToolCallReliabilityScore.score(expectations: expectations, observations: observations)
        XCTAssertEqual(score.nameAccuracy, 1.0)
        XCTAssertEqual(score.argumentValidityRate, 1.0)
        XCTAssertEqual(score.repairRate, 0.0)
    }
}
