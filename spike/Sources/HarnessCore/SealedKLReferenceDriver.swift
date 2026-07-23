import CryptoKit
import Foundation

public struct SealedKLReferenceRuntimeVersions: Codable, Equatable, Hashable, Sendable {
    public let mlx: String
    public let mlxLM: String

    public init(mlx: String, mlxLM: String) {
        self.mlx = mlx
        self.mlxLM = mlxLM
    }
}

public struct SealedKLReferenceBinding: Codable, Equatable, Hashable, Sendable {
    public let model: String
    public let harnessGitSHA: String
    public let modelConfigSHA256: String
    public let checkpointManifestSHA256: String
    public let checkpointContentSHA256: String
    public let tokenizer: String
    public let tokenizerManifestSHA256: String
    /// Existing semantic measurement-corpus identity (16-hex FNV-1a), which stays stable across
    /// JSON formatting churn. `corpusRawFileSHA256` separately binds the exact source bytes.
    public let corpusContentHash: String
    public let corpusRawFileSHA256: String
    public let harness: String
    public let referenceScriptSHA256: String
    public let referenceRuntimeVersions: SealedKLReferenceRuntimeVersions
    public let workloadNonce: String

    public init(
        model: String,
        harnessGitSHA: String,
        modelConfigSHA256: String,
        checkpointManifestSHA256: String,
        checkpointContentSHA256: String,
        tokenizer: String,
        tokenizerManifestSHA256: String,
        corpusContentHash: String,
        corpusRawFileSHA256: String,
        harness: String,
        referenceScriptSHA256: String,
        referenceRuntimeVersions: SealedKLReferenceRuntimeVersions,
        workloadNonce: String
    ) {
        self.model = model
        self.harnessGitSHA = harnessGitSHA
        self.modelConfigSHA256 = modelConfigSHA256
        self.checkpointManifestSHA256 = checkpointManifestSHA256
        self.checkpointContentSHA256 = checkpointContentSHA256
        self.tokenizer = tokenizer
        self.tokenizerManifestSHA256 = tokenizerManifestSHA256
        self.corpusContentHash = corpusContentHash
        self.corpusRawFileSHA256 = corpusRawFileSHA256
        self.harness = harness
        self.referenceScriptSHA256 = referenceScriptSHA256
        self.referenceRuntimeVersions = referenceRuntimeVersions
        self.workloadNonce = workloadNonce
    }
}

public struct SealedKLReferenceManifest: Codable, Equatable, Sendable {
    public let schema: String
    public let version: Int
    public let identity: SealedKLReferenceBinding
    public let corpus: String
    public let referenceVersion: String
    public let maxTokens: Int
    public let sampleSize: Int
    public let entries: [SealedKLReferenceEntry]

    public init(
        schema: String,
        version: Int,
        identity: SealedKLReferenceBinding,
        corpus: String,
        referenceVersion: String,
        maxTokens: Int,
        sampleSize: Int,
        entries: [SealedKLReferenceEntry]
    ) {
        self.schema = schema
        self.version = version
        self.identity = identity
        self.corpus = corpus
        self.referenceVersion = referenceVersion
        self.maxTokens = maxTokens
        self.sampleSize = sampleSize
        self.entries = entries
    }

    private enum CodingKeys: String, CodingKey {
        case schema
        case version
        case identity
        case corpus
        case referenceVersion = "reference-version"
        case maxTokens
        case sampleSize = "sample-size"
        case entries
    }
}

public struct SealedKLReferenceEntry: Codable, Equatable, Sendable {
    public let id: String
    public let tag: String?
    public let promptTokenIDs: [Int]
    public let continuationTokenIDs: [Int]
    public let samplePositions: [Int]?
    public let logitsFile: String
    public let logitsSHA256: String
    public let rowCount: Int
    public let vocabSize: Int
    public let byteCount: Int

    public init(
        id: String,
        tag: String?,
        promptTokenIDs: [Int],
        continuationTokenIDs: [Int],
        samplePositions: [Int]?,
        logitsFile: String,
        logitsSHA256: String,
        rowCount: Int,
        vocabSize: Int,
        byteCount: Int
    ) {
        self.id = id
        self.tag = tag
        self.promptTokenIDs = promptTokenIDs
        self.continuationTokenIDs = continuationTokenIDs
        self.samplePositions = samplePositions
        self.logitsFile = logitsFile
        self.logitsSHA256 = logitsSHA256
        self.rowCount = rowCount
        self.vocabSize = vocabSize
        self.byteCount = byteCount
    }
}

public enum SealedKLReferenceError: Error, Equatable, CustomStringConvertible, Sendable {
    case unsupportedSchema(String)
    case unsupportedVersion(Int)
    case bindingMismatch(field: String, expected: String, actual: String)
    case invalidManifest(String)
    case duplicateID(String)
    case duplicateFile(String)
    case duplicateQuery(String)
    case invalidLogitsFile(String)
    case missingBlob(String)
    case byteCountOverflow(entryID: String)
    case byteCountMismatch(entryID: String, expected: Int, actual: Int)
    case sha256Mismatch(entryID: String)
    case nonFiniteLogit(entryID: String, row: Int, column: Int)
    case unsupportedConfig(String)
    case unsupportedFreeRunningLogprobs
    case missingReplay(String)

    public var description: String {
        switch self {
        case .unsupportedSchema(let schema):
            return "unsupported sealed KL reference schema \(schema)"
        case .unsupportedVersion(let version):
            return "unsupported sealed KL reference version \(version)"
        case .bindingMismatch(let field, let expected, let actual):
            return "sealed KL reference \(field) mismatch: expected \(expected), got \(actual)"
        case .invalidManifest(let reason):
            return "invalid sealed KL reference manifest: \(reason)"
        case .duplicateID(let id):
            return "duplicate sealed KL reference entry id \(id)"
        case .duplicateFile(let file):
            return "duplicate sealed KL reference logits file \(file)"
        case .duplicateQuery(let key):
            return "duplicate sealed KL reference replay query \(key)"
        case .invalidLogitsFile(let file):
            return "invalid sealed KL reference logits file \(file)"
        case .missingBlob(let file):
            return "missing sealed KL reference logits blob \(file)"
        case .byteCountOverflow(let entryID):
            return "sealed KL reference byte-count overflow for entry \(entryID)"
        case .byteCountMismatch(let entryID, let expected, let actual):
            return "sealed KL reference byte-count mismatch for entry \(entryID): expected \(expected), got \(actual)"
        case .sha256Mismatch(let entryID):
            return "sealed KL reference SHA-256 mismatch for entry \(entryID)"
        case .nonFiniteLogit(let entryID, let row, let column):
            return "sealed KL reference entry \(entryID) has non-finite logit at row \(row), column \(column)"
        case .unsupportedConfig(let reason):
            return "sealed KL reference config is unsupported: \(reason)"
        case .unsupportedFreeRunningLogprobs:
            return "sealed KL reference does not support free-running logprobs"
        case .missingReplay(let key):
            return "sealed KL reference has no exact replay for \(key)"
        }
    }
}

public struct SealedKLReferenceBundle: Sendable {
    public let manifest: SealedKLReferenceManifest
    public let manifestSHA256: String
    public let referenceVersions: [String]
    public let referenceRuntimeVersions: SealedKLReferenceRuntimeVersions

    private let continuationsByPrompt: [[Int]: [Int]]
    private let rowsByQuery: [QueryKey: [[Float]]]

    public init(
        manifestData: Data,
        blobs: [String: Data],
        expectedBinding: SealedKLReferenceBinding,
        expectedCorpus: String
    ) throws {
        let manifest = try JSONDecoder().decode(SealedKLReferenceManifest.self, from: manifestData)
        try Self.validateManifestHeader(
            manifest,
            expectedBinding: expectedBinding,
            expectedCorpus: expectedCorpus)
        let validated = try Self.validateEntries(manifest: manifest, blobs: blobs)
        self.manifest = manifest
        self.manifestSHA256 = Self.sha256Hex(manifestData)
        self.referenceVersions = [manifest.referenceVersion]
        self.referenceRuntimeVersions = manifest.identity.referenceRuntimeVersions
        self.continuationsByPrompt = validated.continuations
        self.rowsByQuery = validated.rowsByQuery
    }

    private static func validateEntries(
        manifest: SealedKLReferenceManifest,
        blobs: [String: Data]
    ) throws -> (continuations: [[Int]: [Int]], rowsByQuery: [QueryKey: [[Float]]]) {

        var ids = Set<String>()
        var files = Set<String>()
        var promptKeys = Set<[Int]>()
        var queryKeys = Set<QueryKey>()
        var continuations: [[Int]: [Int]] = [:]
        var rowsByQuery: [QueryKey: [[Float]]] = [:]

        for entry in manifest.entries {
            try Self.validateEntryShape(entry, maxTokens: manifest.maxTokens)
            guard ids.insert(entry.id).inserted else {
                throw SealedKLReferenceError.duplicateID(entry.id)
            }
            guard Self.isSafeBlobFilename(entry.logitsFile) else {
                throw SealedKLReferenceError.invalidLogitsFile(entry.logitsFile)
            }
            guard files.insert(entry.logitsFile).inserted else {
                throw SealedKLReferenceError.duplicateFile(entry.logitsFile)
            }
            if entry.samplePositions == nil {
                guard promptKeys.insert(entry.promptTokenIDs).inserted else {
                    throw SealedKLReferenceError.duplicateQuery(
                        "prompt=\(entry.promptTokenIDs)")
                }
            }
            guard let blob = blobs[entry.logitsFile] else {
                throw SealedKLReferenceError.missingBlob(entry.logitsFile)
            }
            let expectedBytes = try Self.expectedByteCount(
                rows: entry.rowCount,
                vocab: entry.vocabSize,
                entryID: entry.id)
            guard entry.byteCount == expectedBytes else {
                throw SealedKLReferenceError.byteCountMismatch(
                    entryID: entry.id,
                    expected: expectedBytes,
                    actual: entry.byteCount)
            }
            guard blob.count == entry.byteCount else {
                throw SealedKLReferenceError.byteCountMismatch(
                    entryID: entry.id,
                    expected: entry.byteCount,
                    actual: blob.count)
            }
            guard Self.sha256Hex(blob) == entry.logitsSHA256 else {
                throw SealedKLReferenceError.sha256Mismatch(entryID: entry.id)
            }

            let rows = try Self.decodeRows(entry: entry, blob: blob)
            let key = QueryKey(
                prompt: entry.promptTokenIDs,
                continuation: entry.continuationTokenIDs,
                samplePositions: entry.samplePositions)
            guard queryKeys.insert(key).inserted else {
                throw SealedKLReferenceError.duplicateQuery(key.description)
            }
            if entry.samplePositions == nil {
                continuations[entry.promptTokenIDs] = entry.continuationTokenIDs
            }
            rowsByQuery[key] = rows
        }
        guard Set(blobs.keys) == files else {
            let undeclared = Set(blobs.keys).subtracting(files).sorted()
            let missing = files.subtracting(Set(blobs.keys)).sorted()
            throw SealedKLReferenceError.invalidManifest(
                "logits blob set mismatch; undeclared=\(undeclared), missing=\(missing)")
        }

        return (continuations: continuations, rowsByQuery: rowsByQuery)
    }

    public static func canonicalManifestData(_ manifest: SealedKLReferenceManifest) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(manifest)
    }

    public static func canonicalManifestSHA256(_ manifest: SealedKLReferenceManifest) throws -> String {
        try sha256Hex(canonicalManifestData(manifest))
    }

    public static func sha256Hex(_ data: Data) -> String {
        CryptoKit.SHA256.hash(data: data).map {
            String(format: "%02x", $0)
        }.joined()
    }

    func validateConfig(_ config: RunConfig) throws {
        guard config.temperature == 0 else {
            throw SealedKLReferenceError.unsupportedConfig("temperature must be 0")
        }
        guard config.maxTokens == manifest.maxTokens else {
            throw SealedKLReferenceError.unsupportedConfig(
                "maxTokens must be \(manifest.maxTokens)")
        }
        guard config.specDecode == nil, config.specNgram == nil, config.specMaxDraft == nil,
              config.specCompiledVerify == nil, config.kvQuant == nil,
              config.kvtunerSelection == nil, config.compressedKVAttention == nil,
              config.compressedKVAttentionExpectedCheckpointContentSHA256 == nil
        else {
            throw SealedKLReferenceError.unsupportedConfig(
                "candidate-only runtime options are not part of a sealed reference replay")
        }
    }

    func continuation(prompt: [Int], config: RunConfig) throws -> [Int] {
        try validateConfig(config)
        guard let continuation = continuationsByPrompt[prompt] else {
            throw SealedKLReferenceError.missingReplay("prompt=\(prompt)")
        }
        return continuation
    }

    func rows(
        prompt: [Int],
        continuation: [Int],
        positions: [Int]?,
        config: RunConfig
    ) throws -> [[Float]] {
        try validateConfig(config)
        let key = QueryKey(
            prompt: prompt,
            continuation: continuation,
            samplePositions: positions)
        guard let rows = rowsByQuery[key] else {
            throw SealedKLReferenceError.missingReplay(key.description)
        }
        return rows
    }

    private static func validateManifestHeader(
        _ manifest: SealedKLReferenceManifest,
        expectedBinding: SealedKLReferenceBinding,
        expectedCorpus: String
    ) throws {
        guard manifest.schema == "sealed-kl-reference" else {
            throw SealedKLReferenceError.unsupportedSchema(manifest.schema)
        }
        guard manifest.version == 1 else {
            throw SealedKLReferenceError.unsupportedVersion(manifest.version)
        }
        try requireStableIdentifier(field: "corpus", value: manifest.corpus)
        try requireStableIdentifier(field: "reference-version", value: manifest.referenceVersion)
        guard manifest.maxTokens > 0 else {
            throw SealedKLReferenceError.invalidManifest("maxTokens must be positive")
        }
        guard manifest.sampleSize >= 0 else {
            throw SealedKLReferenceError.invalidManifest("sample-size must be non-negative")
        }
        guard !manifest.entries.isEmpty else {
            throw SealedKLReferenceError.invalidManifest("entries must not be empty")
        }
        try validateBinding(manifest.identity)
        guard manifest.identity == expectedBinding else {
            throw SealedKLReferenceError.bindingMismatch(
                field: "identity",
                expected: "\(expectedBinding)",
                actual: "\(manifest.identity)")
        }
        try requireEqual(
            field: "corpus",
            expected: expectedCorpus,
            actual: manifest.corpus)
    }

    private static func validateBinding(_ binding: SealedKLReferenceBinding) throws {
        try requireStableIdentifier(field: "model", value: binding.model)
        try requireStableIdentifier(field: "tokenizer", value: binding.tokenizer)
        try requireStableIdentifier(field: "harness", value: binding.harness)
        try requireStableIdentifier(field: "workloadNonce", value: binding.workloadNonce)
        try requireStableIdentifier(
            field: "referenceRuntimeVersions.mlx",
            value: binding.referenceRuntimeVersions.mlx)
        try requireStableIdentifier(
            field: "referenceRuntimeVersions.mlxLM",
            value: binding.referenceRuntimeVersions.mlxLM)
        try requireHex(
            field: "harnessGitSHA",
            value: binding.harnessGitSHA,
            length: 40)
        try requireHex(field: "modelConfigSHA256", value: binding.modelConfigSHA256, length: 64)
        try requireHex(
            field: "checkpointManifestSHA256",
            value: binding.checkpointManifestSHA256,
            length: 64)
        try requireHex(
            field: "checkpointContentSHA256",
            value: binding.checkpointContentSHA256,
            length: 64)
        try requireHex(
            field: "tokenizerManifestSHA256",
            value: binding.tokenizerManifestSHA256,
            length: 64)
        try requireHex(
            field: "corpusContentHash",
            value: binding.corpusContentHash,
            length: 16)
        try requireHex(field: "corpusRawFileSHA256", value: binding.corpusRawFileSHA256, length: 64)
        try requireHex(field: "referenceScriptSHA256", value: binding.referenceScriptSHA256, length: 64)
    }

    private static func requireEqual(field: String, expected: String, actual: String) throws {
        guard actual == expected else {
            throw SealedKLReferenceError.bindingMismatch(
                field: field,
                expected: expected,
                actual: actual)
        }
    }

    private static func requireStableIdentifier(field: String, value: String) throws {
        guard !value.isEmpty,
              value == value.trimmingCharacters(in: .whitespacesAndNewlines),
              !value.contains("/"),
              !value.contains("\\"),
              !value.contains(".."),
              value.unicodeScalars.allSatisfy({ $0.value >= 0x20 && $0.value != 0x7f })
        else {
            throw SealedKLReferenceError.invalidManifest("\(field) must be a stable identifier")
        }
    }

    private static func requireHex(field: String, value: String, length: Int) throws {
        let hex = Set("0123456789abcdef")
        guard value.count == length, value.allSatisfy({ hex.contains($0) }) else {
            throw SealedKLReferenceError.invalidManifest(
                "\(field) must be lowercase \(length)-hex")
        }
    }

    private static func validateEntryShape(
        _ entry: SealedKLReferenceEntry,
        maxTokens: Int
    ) throws {
        try requireStableIdentifier(field: "entry.id", value: entry.id)
        if let tag = entry.tag {
            try requireStableIdentifier(field: "entry.tag", value: tag)
        }
        guard !entry.promptTokenIDs.isEmpty else {
            throw SealedKLReferenceError.invalidManifest(
                "entry \(entry.id) promptTokenIDs is empty")
        }
        guard !entry.continuationTokenIDs.isEmpty else {
            throw SealedKLReferenceError.invalidManifest(
                "entry \(entry.id) continuationTokenIDs is empty")
        }
        guard entry.promptTokenIDs.allSatisfy({ $0 >= 0 }),
            entry.continuationTokenIDs.allSatisfy({ $0 >= 0 })
        else {
            throw SealedKLReferenceError.invalidManifest(
                "entry \(entry.id) token IDs must be non-negative")
        }
        guard entry.rowCount > 0, entry.vocabSize > 0, entry.byteCount > 0 else {
            throw SealedKLReferenceError.invalidManifest(
                "entry \(entry.id) rowCount, vocabSize, and byteCount must be positive")
        }
        try requireHex(field: "entry \(entry.id) logitsSHA256", value: entry.logitsSHA256, length: 64)

        if let samplePositions = entry.samplePositions {
            guard entry.rowCount == samplePositions.count else {
                throw SealedKLReferenceError.invalidManifest(
                    "entry \(entry.id) rowCount must equal sampled position count")
            }
            var previous: Int?
            for position in samplePositions {
                guard position >= 0, position < entry.continuationTokenIDs.count else {
                    throw SealedKLReferenceError.invalidManifest(
                        "entry \(entry.id) sampled position \(position) is out of range")
                }
                if let previous, position <= previous {
                    throw SealedKLReferenceError.invalidManifest(
                        "entry \(entry.id) sampled positions must be strictly ascending")
                }
                previous = position
            }
        } else {
            guard entry.rowCount == entry.continuationTokenIDs.count else {
                throw SealedKLReferenceError.invalidManifest(
                    "entry \(entry.id) full rowCount must equal continuationTokenIDs count")
            }
            guard entry.continuationTokenIDs.count <= maxTokens else {
                throw SealedKLReferenceError.invalidManifest(
                    "entry \(entry.id) continuation exceeds maxTokens")
            }
        }
    }

    private static func expectedByteCount(rows: Int, vocab: Int, entryID: String) throws -> Int {
        let cellsResult = rows.multipliedReportingOverflow(by: vocab)
        guard !cellsResult.overflow else {
            throw SealedKLReferenceError.byteCountOverflow(entryID: entryID)
        }
        let bytesResult = cellsResult.partialValue.multipliedReportingOverflow(by: 4)
        guard !bytesResult.overflow else {
            throw SealedKLReferenceError.byteCountOverflow(entryID: entryID)
        }
        return bytesResult.partialValue
    }

    private static func decodeRows(
        entry: SealedKLReferenceEntry,
        blob: Data
    ) throws -> [[Float]] {
        var rows: [[Float]] = []
        rows.reserveCapacity(entry.rowCount)
        try blob.withUnsafeBytes { rawBuffer in
            guard let bytes = rawBuffer.bindMemory(to: UInt8.self).baseAddress else {
                throw SealedKLReferenceError.byteCountMismatch(
                    entryID: entry.id,
                    expected: entry.byteCount,
                    actual: 0)
            }
            for rowIndex in 0..<entry.rowCount {
                var row: [Float] = []
                row.reserveCapacity(entry.vocabSize)
                for columnIndex in 0..<entry.vocabSize {
                    let offset = ((rowIndex * entry.vocabSize) + columnIndex) * 4
                    let bitPattern =
                        UInt32(bytes[offset])
                        | (UInt32(bytes[offset + 1]) << 8)
                        | (UInt32(bytes[offset + 2]) << 16)
                        | (UInt32(bytes[offset + 3]) << 24)
                    let value = Float(bitPattern: bitPattern)
                    guard value.isFinite else {
                        throw SealedKLReferenceError.nonFiniteLogit(
                            entryID: entry.id,
                            row: rowIndex,
                            column: columnIndex)
                    }
                    row.append(value)
                }
                rows.append(row)
            }
        }
        return rows
    }

    private static func isSafeBlobFilename(_ filename: String) -> Bool {
        guard !filename.isEmpty, filename != ".", filename != ".." else { return false }
        guard !filename.contains("/"), !filename.contains("\\") else { return false }
        return !filename.contains("..")
    }
}

public struct SealedKLReferenceDriver: EngineDriver {
    public let bundle: SealedKLReferenceBundle

    public init(bundle: SealedKLReferenceBundle) {
        self.bundle = bundle
    }

    public func generate(prompt: [Int], config: RunConfig) async throws -> RunResult {
        RunResult(tokens: try bundle.continuation(prompt: prompt, config: config))
    }

    public func logprobs(prompt: [Int], config: RunConfig) async throws -> [[Float]] {
        throw SealedKLReferenceError.unsupportedFreeRunningLogprobs
    }

    public func logprobs(
        prompt: [Int],
        forcedContinuation: [Int],
        config: RunConfig
    ) async throws -> [[Float]] {
        try bundle.rows(
            prompt: prompt,
            continuation: forcedContinuation,
            positions: nil,
            config: config)
    }

    public func logprobs(
        prompt: [Int],
        forcedContinuation: [Int],
        atPositions positions: [Int],
        config: RunConfig
    ) async throws -> [[Float]] {
        try bundle.rows(
            prompt: prompt,
            continuation: forcedContinuation,
            positions: positions,
            config: config)
    }
}

private struct QueryKey: Hashable, Sendable, CustomStringConvertible {
    let prompt: [Int]
    let continuation: [Int]
    let samplePositions: [Int]?

    var description: String {
        "prompt=\(prompt)|continuation=\(continuation)|positions=\(samplePositions.map(String.init(describing:)) ?? "full")"
    }
}
