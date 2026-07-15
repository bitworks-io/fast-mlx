import Foundation

public enum TaskCoherenceDomain: String, Codable, CaseIterable, Hashable, Sendable {
    case math
    case code
    case structuredTool = "structured-tool"
    case longRetrieval = "long-retrieval"
}

public enum TaskCoherenceScoringMode: String, Codable, Equatable, Sendable {
    case restrictedChoice = "restricted-choice"
    case structuredTool = "structured-tool"
}

public struct TaskToolExpectation: Codable, Equatable, Sendable {
    public let name: String
    public let arguments: [String: String]

    public init(name: String, arguments: [String: String]) {
        self.name = name
        self.arguments = arguments
    }
}

public struct TaskCoherenceItem: Codable, Equatable, Sendable {
    public let id: String
    public let domain: TaskCoherenceDomain
    public let scoringMode: TaskCoherenceScoringMode
    public let prefix: String
    public let material: String
    public let suffix: String
    public let query: String
    public let expectedChoice: String?
    public let expectedTool: TaskToolExpectation?

    public var prompt: String { prefix + material + suffix + query }

    public init(
        id: String,
        domain: TaskCoherenceDomain,
        scoringMode: TaskCoherenceScoringMode,
        prefix: String,
        material: String,
        suffix: String,
        query: String,
        expectedChoice: String?,
        expectedTool: TaskToolExpectation?
    ) {
        self.id = id
        self.domain = domain
        self.scoringMode = scoringMode
        self.prefix = prefix
        self.material = material
        self.suffix = suffix
        self.query = query
        self.expectedChoice = expectedChoice
        self.expectedTool = expectedTool
    }
}

public enum TaskCoherenceError: Error, Equatable, Sendable {
    case unsupportedSchema(Int)
    case invalidCorpusID
    case invalidItem(String)
    case duplicateItemID(String)
    case wrongDomainCount(
        domain: TaskCoherenceDomain, got: Int, expected: Int)
    case invalidChoiceTokenIDs
    case choiceTokenOutOfRange(String)
    case nonFiniteChoiceLogit(String)
    case invalidScoreSet(String)
    case mismatchedScoreItems
    case invalidRunIdentity
    case referenceMustBeFP16
    case mismatchedRunIdentity
    case scoreItemsDoNotMatchCorpus
    case invalidReferenceBaseline(TaskCoherenceDomain)
    case invalidReferenceStructuredValidity
}

/// A frozen, deterministic task corpus. The content hash covers the fully expanded prompts and
/// scoring expectations, not just the compact generation recipe.
public struct TaskCoherenceCorpus: Codable, Equatable, Sendable {
    public static let requiredItemsPerDomain = 20

    public let schemaVersion: Int
    public let id: String
    public let items: [TaskCoherenceItem]
    public let contentHash: String

    private enum CodingKeys: String, CodingKey {
        case schemaVersion
        case id
        case items
        case contentHash
    }

    public init(
        schemaVersion: Int,
        id: String,
        items: [TaskCoherenceItem]
    ) throws {
        guard schemaVersion == 1 else {
            throw TaskCoherenceError.unsupportedSchema(schemaVersion)
        }
        guard !id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw TaskCoherenceError.invalidCorpusID
        }

        var seen = Set<String>()
        for item in items {
            guard seen.insert(item.id).inserted else {
                throw TaskCoherenceError.duplicateItemID(item.id)
            }
            guard Self.isValid(item) else {
                throw TaskCoherenceError.invalidItem(item.id)
            }
        }
        for domain in TaskCoherenceDomain.allCases {
            let count = items.lazy.filter { $0.domain == domain }.count
            guard count == Self.requiredItemsPerDomain else {
                throw TaskCoherenceError.wrongDomainCount(
                    domain: domain,
                    got: count,
                    expected: Self.requiredItemsPerDomain)
            }
        }

        self.schemaVersion = schemaVersion
        self.id = id
        self.items = items
        self.contentHash = Self.hash(
            schemaVersion: schemaVersion, id: id, items: items)
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let schemaVersion = try container.decode(Int.self, forKey: .schemaVersion)
        let id = try container.decode(String.self, forKey: .id)
        let items = try container.decode(
            [TaskCoherenceItem].self, forKey: .items)
        let declaredHash = try container.decode(String.self, forKey: .contentHash)

        do {
            let validated = try TaskCoherenceCorpus(
                schemaVersion: schemaVersion, id: id, items: items)
            guard declaredHash == validated.contentHash else {
                throw DecodingError.dataCorruptedError(
                    forKey: .contentHash,
                    in: container,
                    debugDescription: "task-coherence content hash mismatch")
            }
            self = validated
        } catch let error as DecodingError {
            throw error
        } catch {
            throw DecodingError.dataCorruptedError(
                forKey: .items,
                in: container,
                debugDescription: "invalid task-coherence corpus: \(error)")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(schemaVersion, forKey: .schemaVersion)
        try container.encode(id, forKey: .id)
        try container.encode(items, forKey: .items)
        try container.encode(contentHash, forKey: .contentHash)
    }

    /// Exact corpus identity presented to the KVTuner calibration-leakage gate. The shared factory
    /// canonicalizes prompt digests that match the per-case hashes preserved in task evidence.
    public var kvtunerEvaluationCorpusIdentity:
        KVTunerEvaluationCorpusIdentity
    {
        get throws {
            try KVTunerEvaluationCorpusIdentity.taskCoherenceCorpus(self)
        }
    }

    private static func isValid(_ item: TaskCoherenceItem) -> Bool {
        guard !item.id.isEmpty,
            !item.prefix.isEmpty,
            !item.material.isEmpty,
            !item.suffix.isEmpty,
            !item.query.isEmpty
        else { return false }

        switch item.scoringMode {
        case .restrictedChoice:
            guard item.domain != .structuredTool,
                let expected = item.expectedChoice,
                ["A", "B", "C", "D"].contains(expected),
                item.expectedTool == nil
            else { return false }
        case .structuredTool:
            guard item.domain == .structuredTool,
                item.expectedChoice == nil,
                let expected = item.expectedTool,
                !expected.name.isEmpty,
                !expected.arguments.isEmpty,
                expected.arguments.keys.allSatisfy({ !$0.isEmpty })
            else { return false }
        }
        return true
    }

    private static func hash(
        schemaVersion: Int,
        id: String,
        items: [TaskCoherenceItem]
    ) -> String {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(items.reduce(128) { $0 + $1.prompt.utf8.count })

        func appendField(_ value: String) {
            let valueBytes = Array(value.utf8)
            bytes.append(contentsOf: String(valueBytes.count).utf8)
            bytes.append(0x3a)
            bytes.append(contentsOf: valueBytes)
            bytes.append(0x0a)
        }

        appendField(String(schemaVersion))
        appendField(id)
        for item in items {
            appendField(item.id)
            appendField(item.domain.rawValue)
            appendField(item.scoringMode.rawValue)
            appendField(item.prefix)
            appendField(item.material)
            appendField(item.suffix)
            appendField(item.query)
            appendField(item.expectedChoice ?? "")
            if let expected = item.expectedTool {
                appendField(expected.name)
                for key in expected.arguments.keys.sorted() {
                    appendField(key)
                    appendField(expected.arguments[key] ?? "")
                }
            } else {
                appendField("")
            }
        }
        return fnv1a64(bytes)
    }
}

public enum TaskRestrictedChoiceScorer {
    private static let labels = ["A", "B", "C", "D"]
    /// Source-locked tokenizer spellings for the four next-token candidates. Leading spaces are
    /// intentional: every frozen question ends in non-whitespace prose, and common BPE/SentencePiece
    /// vocabularies represent a word-boundary label as one space-prefixed token. The runtime must
    /// still prove each spelling maps to one distinct token for the measured checkpoint.
    public static let labelTokenSpellings = [
        "A": " A", "B": " B", "C": " C", "D": " D",
    ]

    /// Selects the best of four pinned label-token logits. Logits outside those token IDs are
    /// deliberately ignored, so a high-probability free-form continuation cannot change scoring.
    public static func predict(
        logits: [Float],
        labelTokenIDs: [String: Int]
    ) throws -> String {
        guard Set(labelTokenIDs.keys) == Set(labels),
            Set(labelTokenIDs.values).count == labels.count
        else { throw TaskCoherenceError.invalidChoiceTokenIDs }

        var bestLabel = labels[0]
        var bestLogit = -Float.infinity
        for label in labels {
            guard let tokenID = labelTokenIDs[label], logits.indices.contains(tokenID) else {
                throw TaskCoherenceError.choiceTokenOutOfRange(label)
            }
            let logit = logits[tokenID]
            guard logit.isFinite else {
                throw TaskCoherenceError.nonFiniteChoiceLogit(label)
            }
            if logit > bestLogit {
                bestLogit = logit
                bestLabel = label
            }
        }
        return bestLabel
    }
}

public struct TaskStructuredToolScore: Equatable, Sendable {
    public let syntacticallyValid: Bool
    public let correct: Bool
    public let toolName: String?

    public init(
        syntacticallyValid: Bool,
        correct: Bool,
        toolName: String?
    ) {
        self.syntacticallyValid = syntacticallyValid
        self.correct = correct
        self.toolName = toolName
    }
}

public enum TaskStructuredToolScorer {
    public static func score(
        _ raw: String,
        expected: TaskToolExpectation
    ) -> TaskStructuredToolScore {
        let processed = HarnessCorpus.process(raw)
        var candidates: [String] = []
        if let wrapped = processed.toolArgsJSON { candidates.append(wrapped) }
        candidates.append(contentsOf: balancedJSONObjects(in: processed.visibleText))

        guard candidates.count == 1 else {
            return TaskStructuredToolScore(
                syntacticallyValid: false, correct: false, toolName: nil)
        }
        var duplicateKeyDetector = JSONDuplicateKeyDetector(candidates[0])
        guard duplicateKeyDetector.containsDuplicateKeys() == false,
            let data = candidates[0].data(using: .utf8),
            let value = try? JSONSerialization.jsonObject(with: data),
            let object = value as? [String: Any],
            Set(object.keys) == Set(["name", "arguments"]),
            let name = object["name"] as? String,
            let rawArguments = object["arguments"] as? [String: Any]
        else {
            return TaskStructuredToolScore(
                syntacticallyValid: false, correct: false, toolName: nil)
        }

        var arguments: [String: String] = [:]
        for (key, value) in rawArguments {
            guard let string = value as? String else {
                return TaskStructuredToolScore(
                    syntacticallyValid: false, correct: false, toolName: name)
            }
            arguments[key] = string
        }

        return TaskStructuredToolScore(
            syntacticallyValid: true,
            correct: name == expected.name && arguments == expected.arguments,
            toolName: name)
    }

    /// Extract complete top-level JSON objects while respecting quoted braces and escapes. More
    /// than one object is intentionally ambiguous and rejected by `score`.
    private static func balancedJSONObjects(in text: String) -> [String] {
        let bytes = Array(text.utf8)
        var results: [String] = []
        var startIndex: Int?
        var depth = 0
        var inString = false
        var escaped = false

        for (index, byte) in bytes.enumerated() {
            if inString {
                if escaped {
                    escaped = false
                } else if byte == 0x5c {
                    escaped = true
                } else if byte == 0x22 {
                    inString = false
                }
                continue
            }

            if byte == 0x22, depth > 0 {
                inString = true
            } else if byte == 0x7b {
                if depth == 0 { startIndex = index }
                depth += 1
            } else if byte == 0x7d, depth > 0 {
                depth -= 1
                if depth == 0, let objectStart = startIndex {
                    let objectBytes = bytes[objectStart ... index]
                    if let object = String(bytes: objectBytes, encoding: .utf8) {
                        results.append(object)
                    }
                    startIndex = nil
                }
            }
        }
        return results
    }
}

/// Foundation's JSON decoders accept duplicate object keys. Tool-call scoring is stricter: a
/// conflicting duplicate must not be normalized into an apparently correct invocation.
private struct JSONDuplicateKeyDetector {
    private let bytes: [UInt8]
    private var index = 0

    init(_ json: String) { bytes = Array(json.utf8) }

    mutating func containsDuplicateKeys() -> Bool? {
        skipWhitespace()
        guard let duplicate = parseValue() else { return nil }
        skipWhitespace()
        guard index == bytes.count else { return nil }
        return duplicate
    }

    private mutating func parseValue() -> Bool? {
        skipWhitespace()
        guard index < bytes.count else { return nil }
        switch bytes[index] {
        case 0x7b: return parseObject()
        case 0x5b: return parseArray()
        case 0x22: return parseString() == nil ? nil : false
        default: return parseScalar()
        }
    }

    private mutating func parseObject() -> Bool? {
        guard consume(0x7b) else { return nil }
        skipWhitespace()
        if consume(0x7d) { return false }

        var keys = Set<String>()
        var foundDuplicate = false
        while true {
            skipWhitespace()
            guard let key = parseString() else { return nil }
            if !keys.insert(key).inserted { foundDuplicate = true }
            skipWhitespace()
            guard consume(0x3a), let nestedDuplicate = parseValue() else {
                return nil
            }
            foundDuplicate = foundDuplicate || nestedDuplicate
            skipWhitespace()
            if consume(0x7d) { return foundDuplicate }
            guard consume(0x2c) else { return nil }
        }
    }

    private mutating func parseArray() -> Bool? {
        guard consume(0x5b) else { return nil }
        skipWhitespace()
        if consume(0x5d) { return false }

        var foundDuplicate = false
        while true {
            guard let nestedDuplicate = parseValue() else { return nil }
            foundDuplicate = foundDuplicate || nestedDuplicate
            skipWhitespace()
            if consume(0x5d) { return foundDuplicate }
            guard consume(0x2c) else { return nil }
        }
    }

    private mutating func parseString() -> String? {
        guard index < bytes.count, bytes[index] == 0x22 else { return nil }
        let start = index
        index += 1
        var escaped = false
        while index < bytes.count {
            let byte = bytes[index]
            index += 1
            if escaped {
                escaped = false
            } else if byte == 0x5c {
                escaped = true
            } else if byte == 0x22 {
                let encoded = Data(bytes[start ..< index])
                return try? JSONDecoder().decode(String.self, from: encoded)
            }
        }
        return nil
    }

    private mutating func parseScalar() -> Bool? {
        let start = index
        while index < bytes.count {
            switch bytes[index] {
            case 0x20, 0x09, 0x0a, 0x0d, 0x2c, 0x5d, 0x7d:
                return index > start ? false : nil
            default:
                index += 1
            }
        }
        return index > start ? false : nil
    }

    private mutating func skipWhitespace() {
        while index < bytes.count,
            [UInt8(0x20), 0x09, 0x0a, 0x0d].contains(bytes[index])
        {
            index += 1
        }
    }

    private mutating func consume(_ byte: UInt8) -> Bool {
        guard index < bytes.count, bytes[index] == byte else { return false }
        index += 1
        return true
    }
}

public struct TaskItemScore: Codable, Equatable, Sendable {
    public let itemID: String
    public let domain: TaskCoherenceDomain
    public let correct: Bool
    public let syntacticallyValid: Bool?

    public init(
        itemID: String,
        domain: TaskCoherenceDomain,
        correct: Bool,
        syntacticallyValid: Bool?
    ) {
        self.itemID = itemID
        self.domain = domain
        self.correct = correct
        self.syntacticallyValid = syntacticallyValid
    }
}

public struct TaskCoherenceRunIdentity: Codable, Equatable, Sendable {
    public let corpusID: String
    public let corpusContentHash: String
    public let modelConfigHash: String
    public let modelCheckpointManifestHash: String
    public let kvQuantTier: String
    /// Exact authenticated heterogeneous policy. Uniform fp16/affine/KVarN runs must leave this
    /// nil; a KVTuner tier is not a runnable identity without it.
    public let kvtunerSchedule: KVTunerScheduleBinding?

    public init(
        corpusID: String,
        corpusContentHash: String,
        modelConfigHash: String,
        modelCheckpointManifestHash: String,
        kvQuantTier: String,
        kvtunerSchedule: KVTunerScheduleBinding? = nil
    ) {
        self.corpusID = corpusID
        self.corpusContentHash = corpusContentHash
        self.modelConfigHash = modelConfigHash
        self.modelCheckpointManifestHash = modelCheckpointManifestHash
        self.kvQuantTier = kvQuantTier
        self.kvtunerSchedule = kvtunerSchedule
    }
}

public struct TaskCoherenceScoredRun: Codable, Equatable, Sendable {
    public let identity: TaskCoherenceRunIdentity
    public let scores: [TaskItemScore]

    public init(
        identity: TaskCoherenceRunIdentity,
        scores: [TaskItemScore]
    ) {
        self.identity = identity
        self.scores = scores
    }
}

public struct TaskDomainAssessment: Codable, Equatable, Sendable {
    public let domain: TaskCoherenceDomain
    public let correct: Int
    public let denominator: Int
    public let score: Double
    public let referenceScore: Double
    public let deltaPercentagePoints: Double
    public let chanceBaseline: Double
    public let halfReferenceScore: Double
    public let hardFloorPassed: Bool
}

public struct TaskStructuredValidityAssessment: Codable, Equatable, Sendable {
    public let valid: Int
    public let denominator: Int
    public let rate: Double
    public let referenceRate: Double
    public let passed: Bool
}

public struct TaskCoherenceAssessment: Codable, Equatable, Sendable {
    public let domains: [TaskDomainAssessment]
    public let structuredValidity: TaskStructuredValidityAssessment
    public let referenceBaselinePassed: Bool
    public let hardFloorPassed: Bool
    public let balancedTaskDeltaPassed: Bool

    /// The hard floor rejects only a genuine collapse: a domain must be both below its 25% chance
    /// baseline and below half of fp16. Noticeable but still useful loss remains a valid aggressive
    /// dial choice. The Balanced signal is separate and requires every task delta to stay within
    /// five percentage points of fp16.
    public static func derive(
        candidate: TaskCoherenceScoredRun,
        reference: TaskCoherenceScoredRun,
        corpus: TaskCoherenceCorpus
    ) throws -> TaskCoherenceAssessment {
        try validateIdentity(candidate.identity, corpus: corpus)
        try validateIdentity(reference.identity, corpus: corpus)
        guard reference.identity.kvQuantTier == "fp16" else {
            throw TaskCoherenceError.referenceMustBeFP16
        }
        guard candidate.identity.corpusID == corpus.id,
            candidate.identity.corpusContentHash == corpus.contentHash,
            reference.identity.corpusID == corpus.id,
            reference.identity.corpusContentHash == corpus.contentHash,
            candidate.identity.modelConfigHash
                == reference.identity.modelConfigHash,
            candidate.identity.modelCheckpointManifestHash
                == reference.identity.modelCheckpointManifestHash
        else { throw TaskCoherenceError.mismatchedRunIdentity }

        try validateScores(candidate.scores, label: "candidate")
        try validateScores(reference.scores, label: "reference")

        let corpusByID = Dictionary(
            uniqueKeysWithValues: corpus.items.map { ($0.id, $0.domain) })
        guard Set(candidate.scores.map(\.itemID)) == Set(corpusByID.keys),
            Set(reference.scores.map(\.itemID)) == Set(corpusByID.keys),
            candidate.scores.allSatisfy({ corpusByID[$0.itemID] == $0.domain }),
            reference.scores.allSatisfy({ corpusByID[$0.itemID] == $0.domain })
        else { throw TaskCoherenceError.scoreItemsDoNotMatchCorpus }

        let candidateByID = Dictionary(
            uniqueKeysWithValues: candidate.scores.map { ($0.itemID, $0) })
        let referenceByID = Dictionary(
            uniqueKeysWithValues: reference.scores.map { ($0.itemID, $0) })
        guard candidateByID.keys == referenceByID.keys,
            candidateByID.allSatisfy({ id, score in
                referenceByID[id]?.domain == score.domain
            })
        else { throw TaskCoherenceError.mismatchedScoreItems }

        let chance = 0.25
        var domains: [TaskDomainAssessment] = []
        for domain in TaskCoherenceDomain.allCases {
            let candidateRows = candidate.scores.filter { $0.domain == domain }
            let referenceRows = reference.scores.filter { $0.domain == domain }
            let candidateCorrect = candidateRows.filter(\.correct).count
            let referenceCorrect = referenceRows.filter(\.correct).count
            let candidateScore = Double(candidateCorrect) / Double(candidateRows.count)
            let referenceScore = Double(referenceCorrect) / Double(referenceRows.count)
            guard referenceScore > chance else {
                throw TaskCoherenceError.invalidReferenceBaseline(domain)
            }
            let halfReference = 0.5 * referenceScore
            domains.append(TaskDomainAssessment(
                domain: domain,
                correct: candidateCorrect,
                denominator: candidateRows.count,
                score: candidateScore,
                referenceScore: referenceScore,
                deltaPercentagePoints: 100 * (candidateScore - referenceScore),
                chanceBaseline: chance,
                halfReferenceScore: halfReference,
                hardFloorPassed: candidateScore >= chance
                    || candidateScore >= halfReference))
        }

        let candidateStructured = candidate.scores.filter {
            $0.domain == .structuredTool
        }
        let referenceStructured = reference.scores.filter {
            $0.domain == .structuredTool
        }
        let candidateValid = candidateStructured.filter {
            $0.syntacticallyValid == true
        }.count
        let referenceValid = referenceStructured.filter {
            $0.syntacticallyValid == true
        }.count
        let candidateValidity = Double(candidateValid) / Double(candidateStructured.count)
        let referenceValidity = Double(referenceValid) / Double(referenceStructured.count)
        guard referenceValidity >= 0.90 else {
            throw TaskCoherenceError.invalidReferenceStructuredValidity
        }
        let structured = TaskStructuredValidityAssessment(
            valid: candidateValid,
            denominator: candidateStructured.count,
            rate: candidateValidity,
            referenceRate: referenceValidity,
            passed: candidateValidity >= 0.90)

        let epsilon = 1e-12
        let balancedDomains = domains.allSatisfy {
            $0.score + epsilon >= $0.referenceScore - 0.05
        }
        let balancedStructured = candidateValidity + epsilon >= referenceValidity - 0.05
        return TaskCoherenceAssessment(
            domains: domains,
            structuredValidity: structured,
            referenceBaselinePassed: true,
            hardFloorPassed: domains.allSatisfy(\.hardFloorPassed) && structured.passed,
            balancedTaskDeltaPassed: balancedDomains && balancedStructured)
    }

    private static func validateIdentity(
        _ identity: TaskCoherenceRunIdentity,
        corpus: TaskCoherenceCorpus
    ) throws {
        guard !identity.corpusID.isEmpty,
            !identity.corpusContentHash.isEmpty,
            !identity.modelConfigHash.isEmpty,
            !identity.modelCheckpointManifestHash.isEmpty,
            !identity.kvQuantTier.isEmpty
        else { throw TaskCoherenceError.invalidRunIdentity }

        if identity.kvQuantTier.hasPrefix("kvtuner-") {
            guard let schedule = identity.kvtunerSchedule,
                schedule.cellID == identity.kvQuantTier,
                schedule.modelConfigHash == identity.modelConfigHash,
                schedule.checkpointManifestHash
                    == identity.modelCheckpointManifestHash,
                schedule.evaluationCorpora.contains(
                    try corpus.kvtunerEvaluationCorpusIdentity)
            else { throw TaskCoherenceError.invalidRunIdentity }
        } else if identity.kvtunerSchedule != nil {
            throw TaskCoherenceError.invalidRunIdentity
        }
    }

    private static func validateScores(
        _ scores: [TaskItemScore],
        label: String
    ) throws {
        guard scores.count
            == TaskCoherenceCorpus.requiredItemsPerDomain
                * TaskCoherenceDomain.allCases.count
        else { throw TaskCoherenceError.invalidScoreSet(label) }
        guard Set(scores.map(\.itemID)).count == scores.count else {
            throw TaskCoherenceError.invalidScoreSet(label)
        }
        for domain in TaskCoherenceDomain.allCases {
            let rows = scores.filter { $0.domain == domain }
            guard rows.count == TaskCoherenceCorpus.requiredItemsPerDomain else {
                throw TaskCoherenceError.invalidScoreSet(label)
            }
            if domain == .structuredTool {
                guard rows.allSatisfy({
                    $0.syntacticallyValid != nil
                        && (!$0.correct || $0.syntacticallyValid == true)
                }) else {
                    throw TaskCoherenceError.invalidScoreSet(label)
                }
            } else {
                guard rows.allSatisfy({ $0.syntacticallyValid == nil }) else {
                    throw TaskCoherenceError.invalidScoreSet(label)
                }
            }
        }
    }
}

public enum TaskCoherenceCorpusV1 {
    private static let prefixSentence =
        "Archived context may contain distractors; preserve exact identifiers and follow only the final question. "
    private static let suffixSentence =
        "End-of-archive padding carries no answer; retrieve the earlier task record before choosing a response. "
    private static let retrievalFiller =
        "Ledger filler: copper lanterns crossed the north gallery while index clerks recorded an irrelevant checksum. "

    public static func make() throws -> TaskCoherenceCorpus {
        var items: [TaskCoherenceItem] = []
        items.reserveCapacity(
            TaskCoherenceCorpus.requiredItemsPerDomain
                * TaskCoherenceDomain.allCases.count)
        items.append(contentsOf: mathItems())
        items.append(contentsOf: codeItems())
        items.append(contentsOf: structuredToolItems())
        items.append(contentsOf: longRetrievalItems())
        return try TaskCoherenceCorpus(
            schemaVersion: 1,
            id: "kvarn-task-coherence-v1",
            items: items)
    }

    private static var standardPrefix: String {
        String(repeating: prefixSentence, count: 18)
    }

    private static var standardSuffix: String {
        String(repeating: suffixSentence, count: 18)
    }

    private static func mathItems() -> [TaskCoherenceItem] {
        let cases: [(String, [String], Int)] = [
            ("What is 17 + 26?", ["41", "43", "45", "47"], 1),
            ("What is 12 × 7?", ["72", "82", "84", "96"], 2),
            ("What is 144 ÷ 12?", ["10", "11", "13", "12"], 3),
            ("What is 91 - 38?", ["53", "51", "55", "57"], 0),
            ("What is 9²?", ["72", "81", "90", "99"], 1),
            ("What is 15% of 200?", ["20", "25", "30", "35"], 2),
            ("A box has 6 rows of 8 bolts. How many bolts?", ["42", "46", "50", "48"], 3),
            ("What is 3/4 of 64?", ["48", "44", "52", "56"], 0),
            ("What is 125 + 78?", ["201", "203", "205", "207"], 1),
            ("What is 11 × 13?", ["121", "132", "143", "154"], 2),
            ("What is 256 ÷ 8?", ["28", "30", "34", "32"], 3),
            ("What is 1000 - 457?", ["543", "533", "553", "563"], 0),
            ("What is the next value: 5, 10, 20, 40, ...?", ["60", "80", "90", "100"], 1),
            ("A $50 item is reduced by $8. What is its price?", ["40", "41", "42", "43"], 2),
            ("What is the perimeter of a 7 by 5 rectangle?", ["12", "20", "35", "24"], 3),
            ("What is 2 + 4 + 6 + 8?", ["20", "18", "22", "24"], 0),
            ("What is 19 × 5?", ["90", "95", "100", "105"], 1),
            ("What is 7 cubed?", ["294", "336", "343", "350"], 2),
            ("Split 99 equally among 9 people. How many each?", ["9", "10", "12", "11"], 3),
            ("What is 68 - 29?", ["39", "37", "41", "43"], 0),
        ]
        return cases.enumerated().map { index, entry in
            let labels = ["A", "B", "C", "D"]
            let choices = zip(labels, entry.1).map { "\($0): \($1)" }.joined(separator: "; ")
            return TaskCoherenceItem(
                id: String(format: "math-%02d", index),
                domain: .math,
                scoringMode: .restrictedChoice,
                prefix: standardPrefix,
                material: "Task record: \(entry.0) Options: \(choices). ",
                suffix: standardSuffix,
                query: "Answer by selecting the single correct option label.",
                expectedChoice: labels[entry.2],
                expectedTool: nil)
        }
    }

    private static func codeItems() -> [TaskCoherenceItem] {
        let cases: [(String, [String], Int)] = [
            ("x=3; x+=4; print(x)", ["6", "7", "8", "9"], 1),
            ("sum([2,4,6])", ["10", "11", "12", "14"], 2),
            ("len('swift')", ["4", "6", "7", "5"], 3),
            ("[n*n for n in [1,2,3]][-1]", ["9", "6", "8", "12"], 0),
            ("let x = [10,20,30]; print(x[1])", ["10", "20", "30", "1"], 1),
            ("print('ab'.upper())", ["ab", "Ab", "AB", "aB"], 2),
            ("print(17 % 5)", ["0", "1", "3", "2"], 3),
            ("print(min(8, 3, 5))", ["3", "5", "8", "0"], 0),
            ("x=false; print(!x)", ["false", "true", "0", "nil"], 1),
            ("print(sorted([3,1,2])[0])", ["0", "2", "1", "3"], 2),
            ("print(2 ** 4)", ["8", "12", "18", "16"], 3),
            ("print(max([4,9,1]))", ["9", "4", "1", "14"], 0),
            ("s='cache'; print(s[0])", ["a", "c", "e", "h"], 1),
            ("print(len([1,1,2,3]))", ["3", "5", "4", "2"], 2),
            ("x=5; print(x > 7)", ["true", "nil", "5", "false"], 3),
            ("print(10 // 3)", ["3", "3.33", "1", "4"], 0),
            ("print('kv' + 'cache')", ["kv cache", "kvcache", "cachekv", "kv+cache"], 1),
            ("print([1,2,3][::-1])", ["[1,2,3]", "[2,3,1]", "[3,2,1]", "[3,1,2]"], 2),
            ("x=2; for _ in range(3): x*=2; print(x)", ["8", "12", "24", "16"], 3),
            ("print(any([False, True, False]))", ["True", "False", "nil", "1"], 0),
        ]
        return cases.enumerated().map { index, entry in
            let labels = ["A", "B", "C", "D"]
            let choices = zip(labels, entry.1).map { "\($0): \($1)" }.joined(separator: "; ")
            return TaskCoherenceItem(
                id: String(format: "code-%02d", index),
                domain: .code,
                scoringMode: .restrictedChoice,
                prefix: standardPrefix,
                material: "Evaluate this deterministic snippet: \(entry.0). Options: \(choices). ",
                suffix: standardSuffix,
                query: "Select the option label matching the exact printed result.",
                expectedChoice: labels[entry.2],
                expectedTool: nil)
        }
    }

    private static func structuredToolItems() -> [TaskCoherenceItem] {
        let tools = ["lookup_record", "fetch_invoice", "read_sensor", "locate_package"]
        return (0 ..< TaskCoherenceCorpus.requiredItemsPerDomain).map { index in
            let tool = tools[index % tools.count]
            let record = String(format: "R-%03d", 100 + index)
            return TaskCoherenceItem(
                id: String(format: "structured-tool-%02d", index),
                domain: .structuredTool,
                scoringMode: .structuredTool,
                prefix: standardPrefix,
                material: "Available tools are lookup_record, fetch_invoice, read_sensor, and locate_package. The authorized request says to call \(tool) with string argument record equal to \(record). ",
                suffix: standardSuffix,
                query: "Return exactly one JSON tool invocation with keys name and arguments.",
                expectedChoice: nil,
                expectedTool: TaskToolExpectation(
                    name: tool, arguments: ["record": record]))
        }
    }

    private static func longRetrievalItems() -> [TaskCoherenceItem] {
        // Five independent needles at each approximate 4K/8K/16K/24K context depth.
        let repeatCounts = [210, 450, 930, 1_410]
        let labels = ["A", "B", "C", "D"]
        var items: [TaskCoherenceItem] = []
        items.reserveCapacity(TaskCoherenceCorpus.requiredItemsPerDomain)
        for depthIndex in repeatCounts.indices {
            for variant in 0 ..< 5 {
                let index = depthIndex * 5 + variant
                let expectedIndex = index % labels.count
                let archiveID = String(format: "ARCHIVE-%02d", index)
                let answer = String(format: "CODE-%04d", 7000 + index * 17)
                let distractors = (0 ..< 4).map { option -> String in
                    option == expectedIndex
                        ? answer
                        : String(format: "CODE-%04d", 8100 + index * 13 + option)
                }
                let choices = zip(labels, distractors).map {
                    "\($0): \($1)"
                }.joined(separator: "; ")
                let before = repeatCounts[depthIndex] * (variant + 1) / 6
                let after = repeatCounts[depthIndex] - before
                let material = String(repeating: retrievalFiller, count: before)
                    + "Critical archive mapping: \(archiveID) has recovery code \(answer). "
                    + String(repeating: retrievalFiller, count: after)
                    + "Declared answer options for \(archiveID): \(choices). "
                items.append(TaskCoherenceItem(
                    id: String(format: "long-retrieval-%02d", index),
                    domain: .longRetrieval,
                    scoringMode: .restrictedChoice,
                    prefix: standardPrefix,
                    material: material,
                    suffix: standardSuffix,
                    query: "Select the option label containing the recovery code for \(archiveID).",
                    expectedChoice: labels[expectedIndex],
                    expectedTool: nil))
            }
        }
        return items
    }
}
