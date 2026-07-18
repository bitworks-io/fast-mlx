import Foundation

public enum KVTunerSourceItemDigest: Sendable {
    public static let algorithm = "sha256-source-item-v1"

    public static func gsm8kTestItem(
        repository: String,
        commit: String,
        testDataSHA256: String,
        testIndex: Int
    ) -> String {
        var data = Data("fast-mlx.kvtuner-source-item.gsm8k.v1\0".utf8)
        for field in [
            repository, commit, testDataSHA256, "test", String(testIndex),
        ] {
            data.append(contentsOf: field.utf8)
            data.append(0)
        }
        return sha256Hex(data)
    }
}

public enum KVTunerCalibrationManifestError:
    Error, Equatable, Sendable
{
    case unsupportedSchema(Int)
    case invalidProtocol(String)
    case invalidIdentity(String)
    case invalidPromptCount(phase: String, expected: Int, actual: Int)
    case nonCanonicalPrompt(phase: String, position: Int)
    case invalidPromptIdentity(phase: String, position: Int)
    case duplicatePromptIdentity
    case sourcePromptIdentityMismatch(String)
}

/// Compact identity for one reconstructed prompt. Prompt text remains in the authenticated
/// official GSM8K source files; the manifest preserves both cryptographic and schedule-leakage
/// fingerprints plus the exact tokenizer input identity.
public struct KVTunerCalibrationPromptIdentity:
    Codable, Equatable, Sendable
{
    public var ordinal: Int
    public var testIndex: Int
    public var promptDigest: String
    public var promptSHA256: String
    /// Exact UTF-8 tokenizer input. The pinned hashes below authenticate these bytes; retaining
    /// them lets the runtime replay tokenization with the live model tokenizer.
    public var promptUTF8: Data
    public var tokenIDs: [Int]
    public var tokenIDsSHA256: String

    public init(
        ordinal: Int,
        testIndex: Int,
        promptDigest: String,
        promptSHA256: String,
        promptUTF8: Data,
        tokenIDs: [Int],
        tokenIDsSHA256: String
    ) {
        self.ordinal = ordinal
        self.testIndex = testIndex
        self.promptDigest = promptDigest
        self.promptSHA256 = promptSHA256
        self.promptUTF8 = promptUTF8
        self.tokenIDs = tokenIDs
        self.tokenIDsSHA256 = tokenIDsSHA256
    }
}

/// Frozen 20-prompt sensitivity + 200-prompt four-shot search identity for the fast-mlx Qwen3
/// adaptation of KVTuner. Full prompt text is reconstructed from the pinned MIT-licensed source;
/// fixed hashes of the ordered per-prompt hash lists make this compact artifact independently
/// reject substitutions, omissions, and reorderings.
public struct KVTunerCalibrationManifest: Codable, Equatable, Sendable {
    public static let requiredSensitivityCount = 20
    public static let requiredSearchCount = 200

    public var schemaVersion: Int
    public var protocolID: String
    public var corpusID: String
    public var modelConfigHash: String
    public var modelConfigSHA256: String
    public var checkpointManifestHash: String
    public var tokenizerSHA256: String
    public var datasetSourceRepository: String
    public var datasetSourceCommit: String
    public var trainDataSHA256: String
    public var testDataSHA256: String
    public var kvtunerSourceCommit: String
    public var lmEvalSourceCommit: String
    public var lmEvalSourceRepository: String
    public var paperVersion: String
    public var promptExpansionID: String
    public var promptListEncodingID: String
    public var tokenizationProtocolID: String
    public var fewShotSeed: UInt64
    public var sensitivityPromptListSHA256: String
    public var fewShotIndexTableSHA256: String
    public var searchPromptListSHA256: String
    public var sensitivityPromptSHA256ListSHA256: String
    public var sensitivityPromptDigestListSHA256: String
    public var searchPromptSHA256ListSHA256: String
    public var searchPromptDigestListSHA256: String
    public var searchNormalizedTargetListSHA256: String
    public var searchNormalizedTargets: [String]
    public var sensitivityPrompts: [KVTunerCalibrationPromptIdentity]
    public var searchPrompts: [KVTunerCalibrationPromptIdentity]

    public init(
        schemaVersion: Int,
        protocolID: String,
        corpusID: String,
        modelConfigHash: String,
        modelConfigSHA256: String,
        checkpointManifestHash: String,
        tokenizerSHA256: String,
        datasetSourceRepository: String,
        datasetSourceCommit: String,
        trainDataSHA256: String,
        testDataSHA256: String,
        kvtunerSourceCommit: String,
        lmEvalSourceCommit: String,
        lmEvalSourceRepository: String,
        paperVersion: String,
        promptExpansionID: String,
        promptListEncodingID: String,
        tokenizationProtocolID: String,
        fewShotSeed: UInt64,
        sensitivityPromptListSHA256: String,
        fewShotIndexTableSHA256: String,
        searchPromptListSHA256: String,
        sensitivityPromptSHA256ListSHA256: String,
        sensitivityPromptDigestListSHA256: String,
        searchPromptSHA256ListSHA256: String,
        searchPromptDigestListSHA256: String,
        searchNormalizedTargetListSHA256: String,
        searchNormalizedTargets: [String],
        sensitivityPrompts: [KVTunerCalibrationPromptIdentity],
        searchPrompts: [KVTunerCalibrationPromptIdentity]
    ) {
        self.schemaVersion = schemaVersion
        self.protocolID = protocolID
        self.corpusID = corpusID
        self.modelConfigHash = modelConfigHash
        self.modelConfigSHA256 = modelConfigSHA256
        self.checkpointManifestHash = checkpointManifestHash
        self.tokenizerSHA256 = tokenizerSHA256
        self.datasetSourceRepository = datasetSourceRepository
        self.datasetSourceCommit = datasetSourceCommit
        self.trainDataSHA256 = trainDataSHA256
        self.testDataSHA256 = testDataSHA256
        self.kvtunerSourceCommit = kvtunerSourceCommit
        self.lmEvalSourceCommit = lmEvalSourceCommit
        self.lmEvalSourceRepository = lmEvalSourceRepository
        self.paperVersion = paperVersion
        self.promptExpansionID = promptExpansionID
        self.promptListEncodingID = promptListEncodingID
        self.tokenizationProtocolID = tokenizationProtocolID
        self.fewShotSeed = fewShotSeed
        self.sensitivityPromptListSHA256 = sensitivityPromptListSHA256
        self.fewShotIndexTableSHA256 = fewShotIndexTableSHA256
        self.searchPromptListSHA256 = searchPromptListSHA256
        self.sensitivityPromptSHA256ListSHA256 =
            sensitivityPromptSHA256ListSHA256
        self.sensitivityPromptDigestListSHA256 =
            sensitivityPromptDigestListSHA256
        self.searchPromptSHA256ListSHA256 = searchPromptSHA256ListSHA256
        self.searchPromptDigestListSHA256 = searchPromptDigestListSHA256
        self.searchNormalizedTargetListSHA256 =
            searchNormalizedTargetListSHA256
        self.searchNormalizedTargets = searchNormalizedTargets
        self.sensitivityPrompts = sensitivityPrompts
        self.searchPrompts = searchPrompts
    }

    public var calibrationEntryDigests: [String] {
        (sensitivityPrompts + searchPrompts)
            .map(\.promptDigest)
            .sorted()
    }

    /// Wrapper-independent identity for the 200 underlying GSM8K rows. Sensitivity and search
    /// deliberately reuse the first 20 rows, so this set is derived from search indices only.
    public var calibrationSourceItemDigests: [String] {
        searchPrompts.map { prompt in
            KVTunerSourceItemDigest.gsm8kTestItem(
                repository: datasetSourceRepository,
                commit: datasetSourceCommit,
                testDataSHA256: testDataSHA256,
                testIndex: prompt.testIndex)
        }.sorted()
    }

    @discardableResult
    public func validated() throws -> KVTunerCalibrationManifest {
        guard schemaVersion == 1 else {
            throw KVTunerCalibrationManifestError.unsupportedSchema(
                schemaVersion)
        }
        let exactStrings: [(String, String, String)] = [
            ("protocolID", protocolID,
             "gsm8k-kvtuner-qwen3-adaptation-v1"),
            ("corpusID", corpusID,
             "gsm8k-kvtuner-calibration-v1"),
            ("datasetSourceRepository", datasetSourceRepository,
             "openai/grade-school-math"),
            ("datasetSourceCommit", datasetSourceCommit,
             "b0bb162abedc65e1fdd8e93ed090fd7598ee68bc"),
            ("trainDataSHA256", trainDataSHA256,
             "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"),
            ("testDataSHA256", testDataSHA256,
             "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"),
            ("kvtunerSourceCommit", kvtunerSourceCommit,
             "96dd05eb2fe350c72c1a3dfdca04e878506f7c17"),
            ("lmEvalSourceCommit", lmEvalSourceCommit,
             "6ec76a6e28056f5b27715b8a233c13018a6967cc"),
            ("lmEvalSourceRepository", lmEvalSourceRepository,
             "cmd2001/lm-evaluation-harness-X"),
            ("paperVersion", paperVersion, "arxiv-2502.04420v5"),
            ("promptExpansionID", promptExpansionID,
             "lm-eval-gsm8k-question-answer-v1"),
            ("promptListEncodingID", promptListEncodingID,
             "utf8-json-compact-array-v1"),
            ("tokenizationProtocolID", tokenizationProtocolID,
             "mlx-tokenizer-raw-add-special-tokens-true-v1"),
            ("sensitivityPromptListSHA256", sensitivityPromptListSHA256,
             "18d51be3aa1ac8e6ed7028a96c8c05efed1aa88588a2635e517d27a3e4e01730"),
            ("fewShotIndexTableSHA256", fewShotIndexTableSHA256,
             "69dc558179253dcee3d87ab0a95b06911737d483af5e1000038044fb0265728e"),
            ("searchPromptListSHA256", searchPromptListSHA256,
             "5e79ef00e8d8d602ce0b24a9ce49e2522fd5c775ae9e00f1e2c57f84931fb16e"),
            ("sensitivityPromptSHA256ListSHA256",
             sensitivityPromptSHA256ListSHA256,
             "aa4325d7b4d3f1f243b4ab10b4fecc585dcb437b9ee77cd720e3945394b400fa"),
            ("sensitivityPromptDigestListSHA256",
             sensitivityPromptDigestListSHA256,
             "5bb0671a74612092b8c9e417392e7ec133a4a06f6bba6944cfcab15a6a202c3c"),
            ("searchPromptSHA256ListSHA256",
             searchPromptSHA256ListSHA256,
             "52504fe6a2d9342e725d80b7f2f76ba500522085c126e656cb30add18bb41290"),
            ("searchPromptDigestListSHA256",
             searchPromptDigestListSHA256,
             "8e1bfc50a66e97a8dd4149c6afd109387601d166d341d1a83da06f022443716d"),
            ("searchNormalizedTargetListSHA256",
             searchNormalizedTargetListSHA256,
             "4f17cc1a11082c7cbd8e2002e69a5b4a30056b1155ce49731b66f8ea4553903f"),
        ]
        for (field, actual, expected) in exactStrings {
            guard actual == expected else {
                throw KVTunerCalibrationManifestError.invalidProtocol(field)
            }
        }
        guard fewShotSeed == KVTunerScheduleSearch.requiredFewShotSeed else {
            throw KVTunerCalibrationManifestError.invalidProtocol(
                "fewShotSeed")
        }
        guard Self.isIdentityDigest(modelConfigHash) else {
            throw KVTunerCalibrationManifestError.invalidIdentity(
                "modelConfigHash")
        }
        guard Self.isLowercaseHex(modelConfigSHA256, length: 64) else {
            throw KVTunerCalibrationManifestError.invalidIdentity(
                "modelConfigSHA256")
        }
        guard Self.isIdentityDigest(checkpointManifestHash) else {
            throw KVTunerCalibrationManifestError.invalidIdentity(
                "checkpointManifestHash")
        }
        guard Self.isLowercaseHex(tokenizerSHA256, length: 64) else {
            throw KVTunerCalibrationManifestError.invalidIdentity(
                "tokenizerSHA256")
        }

        try Self.validatePrompts(
            sensitivityPrompts,
            phase: "sensitivity",
            expectedCount: Self.requiredSensitivityCount)
        try Self.validatePrompts(
            searchPrompts,
            phase: "search",
            expectedCount: Self.requiredSearchCount)

        let all = sensitivityPrompts + searchPrompts
        guard Set(all.map(\.promptDigest)).count == all.count,
            Set(all.map(\.promptSHA256)).count == all.count,
            Set(all.map(\.tokenIDsSHA256)).count == all.count
        else {
            throw KVTunerCalibrationManifestError.duplicatePromptIdentity
        }

        try Self.validateSourceListHash(
            sensitivityPrompts.map(\.promptSHA256),
            expected: sensitivityPromptSHA256ListSHA256,
            label: "sensitivity-sha256")
        try Self.validateSourceListHash(
            sensitivityPrompts.map(\.promptDigest),
            expected: sensitivityPromptDigestListSHA256,
            label: "sensitivity-fnv1a64")
        try Self.validateSourceListHash(
            searchPrompts.map(\.promptSHA256),
            expected: searchPromptSHA256ListSHA256,
            label: "search-sha256")
        try Self.validateSourceListHash(
            searchPrompts.map(\.promptDigest),
            expected: searchPromptDigestListSHA256,
            label: "search-fnv1a64")
        guard searchNormalizedTargets.count == Self.requiredSearchCount,
            searchNormalizedTargets.allSatisfy({ target in
                !target.isEmpty
                    && KVTunerGSM8KScorer.exactMatchNormalize(target)
                        == target
            })
        else {
            throw KVTunerCalibrationManifestError.invalidProtocol(
                "searchNormalizedTargets")
        }
        try Self.validateSourceListHash(
            searchNormalizedTargets,
            expected: searchNormalizedTargetListSHA256,
            label: "search-normalized-targets")
        guard calibrationSourceItemDigests.count
                == Self.requiredSearchCount,
            Set(calibrationSourceItemDigests).count
                == Self.requiredSearchCount
        else {
            throw KVTunerCalibrationManifestError.invalidProtocol(
                "calibrationSourceItemDigests")
        }
        return self
    }

    private static func validatePrompts(
        _ prompts: [KVTunerCalibrationPromptIdentity],
        phase: String,
        expectedCount: Int
    ) throws {
        guard prompts.count == expectedCount else {
            throw KVTunerCalibrationManifestError.invalidPromptCount(
                phase: phase,
                expected: expectedCount,
                actual: prompts.count)
        }
        for (position, prompt) in prompts.enumerated() {
            guard prompt.ordinal == position,
                prompt.testIndex == position
            else {
                throw KVTunerCalibrationManifestError.nonCanonicalPrompt(
                    phase: phase, position: position)
            }
            guard let promptText = String(
                data: prompt.promptUTF8, encoding: .utf8),
                !prompt.promptUTF8.isEmpty,
                KVTunerPromptDigest.isCanonical(prompt.promptDigest),
                prompt.promptDigest
                    == KVTunerPromptDigest.exactText(promptText),
                isLowercaseHex(prompt.promptSHA256, length: 64),
                prompt.promptSHA256 == sha256Hex(prompt.promptUTF8),
                !prompt.tokenIDs.isEmpty,
                prompt.tokenIDs.allSatisfy({ $0 >= 0 }),
                isLowercaseHex(prompt.tokenIDsSHA256, length: 64),
                prompt.tokenIDsSHA256
                    == taskTokenIDsSHA256(prompt.tokenIDs)
            else {
                throw KVTunerCalibrationManifestError.invalidPromptIdentity(
                    phase: phase, position: position)
            }
        }
    }

    private static func validateSourceListHash(
        _ values: [String],
        expected: String,
        label: String
    ) throws {
        let encoder = JSONEncoder()
        let actual = sha256Hex(try encoder.encode(values))
        guard actual == expected else {
            throw KVTunerCalibrationManifestError
                .sourcePromptIdentityMismatch(label)
        }
    }

    private static func isIdentityDigest(_ value: String) -> Bool {
        isLowercaseHex(value, length: 16)
            || isLowercaseHex(value, length: 64)
    }

    private static func isLowercaseHex(
        _ value: String,
        length: Int
    ) -> Bool {
        guard value.count == length else { return false }
        return value.unicodeScalars.allSatisfy {
            CharacterSet(charactersIn: "0123456789abcdef").contains($0)
        }
    }
}
