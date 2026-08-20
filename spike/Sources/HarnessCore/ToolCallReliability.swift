import Foundation

/// One eval case's expected outcome: either a specific tool call (with the argument keys the
/// call must carry) or, when `expectedToolName` is nil, a "should not call" case — the model is
/// expected to answer directly without invoking a tool.
public struct ToolCallExpectation: Equatable, Sendable {
    public let id: String
    public let expectedToolName: String?
    public let requiredArgumentKeys: [String]
    public init(id: String, expectedToolName: String?, requiredArgumentKeys: [String]) {
        self.id = id
        self.expectedToolName = expectedToolName
        self.requiredArgumentKeys = requiredArgumentKeys
    }
}

/// What a model actually produced for one eval case. `argumentsJSON` is the raw arguments JSON
/// string the model emitted (nil when no call was made). `neededRepair` records whether the
/// server had to repair malformed tool-call JSON before it could be used.
public struct ToolCallObservation: Equatable, Sendable {
    public let id: String
    public let emittedToolName: String?
    public let argumentsJSON: String?
    public let neededRepair: Bool
    public init(id: String, emittedToolName: String?, argumentsJSON: String?, neededRepair: Bool) {
        self.id = id
        self.emittedToolName = emittedToolName
        self.argumentsJSON = argumentsJSON
        self.neededRepair = neededRepair
    }
}

/// Aggregate tool-call reliability over an eval corpus. All rate fields are in 0...1.
public struct ToolCallReliabilityScore: Equatable, Sendable {
    public let caseCount: Int
    /// Fraction of cases where emit-or-withhold matched the expectation (a call was made iff one
    /// was expected). Does NOT check the call's name or arguments — see `nameAccuracy` and
    /// `argumentValidityRate` for those.
    public let triggerRate: Double
    /// Among cases where a call was EXPECTED, fraction with the correct emitted tool name.
    /// 1.0 (vacuous pass) when no calls were expected.
    public let nameAccuracy: Double
    /// Among cases where a call was EMITTED, fraction whose arguments parse as a JSON object and
    /// contain every required key for that case. 1.0 (vacuous pass) when no calls were emitted.
    public let argumentValidityRate: Double
    /// Among cases where a call was EMITTED, fraction that needed server-side JSON repair.
    /// 0.0 (vacuous pass) when no calls were emitted.
    public let repairRate: Double
    public init(
        caseCount: Int, triggerRate: Double, nameAccuracy: Double,
        argumentValidityRate: Double, repairRate: Double
    ) {
        self.caseCount = caseCount
        self.triggerRate = triggerRate
        self.nameAccuracy = nameAccuracy
        self.argumentValidityRate = argumentValidityRate
        self.repairRate = repairRate
    }

    /// Scores a set of observations against the expectations they claim to answer, matched by
    /// `id`. Observations with no matching expectation are ignored; an expectation with no
    /// matching observation is treated as an emitted-nothing observation (the model produced no
    /// tool call at all for that case).
    ///
    /// Degenerate-corpus convention: with zero cases, `triggerRate`/`nameAccuracy`/
    /// `argumentValidityRate` report 1.0 (vacuous pass — there is nothing to have gotten wrong)
    /// and `repairRate` reports 0.0 (vacuous pass — no repairs happened). This mirrors the
    /// per-subset vacuous-pass convention used when the relevant subset (expected-call cases,
    /// emitted-call cases) is empty.
    public static func score(
        expectations: [ToolCallExpectation], observations: [ToolCallObservation]
    ) -> ToolCallReliabilityScore {
        let caseCount = expectations.count
        guard caseCount > 0 else {
            return ToolCallReliabilityScore(
                caseCount: 0, triggerRate: 1.0, nameAccuracy: 1.0,
                argumentValidityRate: 1.0, repairRate: 0.0)
        }
        let observationsByID = Dictionary(observations.map { ($0.id, $0) }, uniquingKeysWith: { _, last in last })

        var triggerMatches = 0
        var expectedCallCount = 0, nameCorrectCount = 0
        var emittedCallCount = 0, validArgumentCount = 0, repairedCount = 0

        for expectation in expectations {
            let observation = observationsByID[expectation.id]
                ?? ToolCallObservation(id: expectation.id, emittedToolName: nil, argumentsJSON: nil, neededRepair: false)

            let expectedCall = expectation.expectedToolName != nil
            let emittedCall = observation.emittedToolName != nil
            if expectedCall == emittedCall { triggerMatches += 1 }

            if expectedCall {
                expectedCallCount += 1
                if observation.emittedToolName == expectation.expectedToolName { nameCorrectCount += 1 }
            }

            if emittedCall {
                emittedCallCount += 1
                if observation.neededRepair { repairedCount += 1 }
                if argumentsAreValid(observation.argumentsJSON, requiredKeys: expectation.requiredArgumentKeys) {
                    validArgumentCount += 1
                }
            }
        }

        return ToolCallReliabilityScore(
            caseCount: caseCount,
            triggerRate: Double(triggerMatches) / Double(caseCount),
            nameAccuracy: expectedCallCount == 0 ? 1.0 : Double(nameCorrectCount) / Double(expectedCallCount),
            argumentValidityRate: emittedCallCount == 0 ? 1.0 : Double(validArgumentCount) / Double(emittedCallCount),
            repairRate: emittedCallCount == 0 ? 0.0 : Double(repairedCount) / Double(emittedCallCount))
    }

    private static func argumentsAreValid(_ argumentsJSON: String?, requiredKeys: [String]) -> Bool {
        guard let argumentsJSON, let data = argumentsJSON.data(using: .utf8) else { return false }
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
        return requiredKeys.allSatisfy { object.keys.contains($0) }
    }
}

extension ToolCallReliabilityScore {
    /// Canonical eval cases for the tool-call reliability corpus: a single-required-arg call, a
    /// two-required-arg call, a plain chit-chat case (no call expected), and a zero-argument
    /// call — the four shapes a "should this fire, and does it get the arguments right" score
    /// needs to distinguish.
    public static let defaultToolCallReliabilityCorpus: [ToolCallExpectation] = [
        ToolCallExpectation(
            id: "weather-single-city",
            expectedToolName: "get_weather",
            requiredArgumentKeys: ["location"]),
        ToolCallExpectation(
            id: "verify-identity-two-arg",
            expectedToolName: "verify_identity",
            requiredArgumentKeys: ["email", "order_number"]),
        ToolCallExpectation(
            id: "chit-chat-no-call",
            expectedToolName: nil,
            requiredArgumentKeys: []),
        ToolCallExpectation(
            id: "list-supported-timezones",
            expectedToolName: "list_timezones",
            requiredArgumentKeys: []),
    ]
}
