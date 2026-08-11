import CryptoKit
import Foundation
import HarnessCore
import Tokenizers

let sealedKLReferenceHarnessIdentity =
    "fastmlx-harness-swiftpm-mlx-\(ProvenanceCLI.mlxSwiftVersion)-lm-\(ProvenanceCLI.mlxSwiftLMRevision)"

struct SealedKLReferenceCapturePlan: Equatable, Sendable {
    let modelPath: String
    let outputPath: String
    let workloadNonce: String
    let corpusPath: String
    let positions: Int
    let longContextSamplePositions: Int
    let pythonPath: String
    let scriptPath: String
}

struct SealedKLReferenceReplayPlan: Equatable, Sendable {
    let modelPath: String
    let directoryPath: String
    let expectedManifestSHA256: String
    let workloadNonce: String
    let corpusPath: String
    let scriptPath: String
}

enum KLReferenceRequest: Equatable, Sendable {
    case livePython
    case sealedReplay(SealedKLReferenceReplayPlan)
}

struct PreparedSealedKLReferenceReplay {
    let driver: SealedKLReferenceDriver
    let referenceVersions: ReferenceDriver.ReferenceVersions
    let manifestSHA256: String
    let sourceSnapshot: CompressedKVAttentionRuntimeSourceSnapshot
}

enum SealedKLReferenceCLIError: Error, Equatable, CustomStringConvertible {
    case missingRequiredFlag(String)
    case missingValue(String)
    case unsupportedFlag(String)
    case invalidPositiveInteger(String, String)
    case invalidSHA256(String)
    case outputPathCollision(String)
    case stagingPathCollision(String)
    case invalidStableIdentifier(String, String)
    case invalidDirectoryEntry(String)
    case symlinkRejected(String)
    case modelSourceDrift
    case referenceScriptDrift
    case corpusFileDrift
    case missingReferenceRuntimeVersions
    case sealedReferenceModelMismatch
    case manifestSHA256Mismatch(expected: String, actual: String)
    case corpusTokenizationMismatch(String)
    case entrySetMismatch
    case nonFiniteLogit
    case injectedFailure(String)

    var description: String {
        switch self {
        case .missingRequiredFlag(let flag):
            return "--\(flag) is required"
        case .missingValue(let flag):
            return "--\(flag) requires a value"
        case .unsupportedFlag(let flag):
            return "--\(flag) is not supported for sealed KL reference"
        case .invalidPositiveInteger(let flag, let value):
            return "--\(flag) must be a positive integer; actual=\(value)"
        case .invalidSHA256(let flag):
            return "--\(flag) must be lowercase 64-hex"
        case .outputPathCollision(let path):
            return "sealed KL reference output already exists: \(path)"
        case .stagingPathCollision(let path):
            return "sealed KL reference staging path already exists: \(path)"
        case .invalidStableIdentifier(let field, let value):
            return "\(field) is not a stable sealed-reference identifier: \(value)"
        case .invalidDirectoryEntry(let path):
            return "sealed KL reference directory contains an invalid entry: \(path)"
        case .symlinkRejected(let path):
            return "sealed KL reference rejects symlinks: \(path)"
        case .modelSourceDrift:
            return "sealed KL reference model source changed during capture"
        case .referenceScriptDrift:
            return "sealed KL reference script changed during capture"
        case .corpusFileDrift:
            return "sealed KL reference corpus changed during capture"
        case .missingReferenceRuntimeVersions:
            return "reference runtime did not report mlx/mlx-lm versions"
        case .sealedReferenceModelMismatch:
            return "sealed KL reference requires --reference-model to resolve to --model"
        case .manifestSHA256Mismatch(let expected, let actual):
            return "sealed KL reference manifest SHA mismatch: expected \(expected), got \(actual)"
        case .corpusTokenizationMismatch(let id):
            return "sealed KL reference corpus tokenization mismatch for \(id)"
        case .entrySetMismatch:
            return "sealed KL reference entry set does not match the current corpus"
        case .nonFiniteLogit:
            return "sealed KL reference logits contain a non-finite value"
        case .injectedFailure(let reason):
            return "injected sealed KL reference failure: \(reason)"
        }
    }
}

func requestedKLReference(
    _ flags: Flags,
    modelPath: String,
    sameResolvedModel: Bool
) throws -> KLReferenceRequest {
    let directoryPath = try flags.strictString("sealed-reference", default: "")
    let expectedManifestSHA256 = try flags.strictString(
        "sealed-reference-sha256",
        default: "")
    guard !directoryPath.isEmpty || !expectedManifestSHA256.isEmpty else {
        return .livePython
    }
    guard sameResolvedModel else {
        throw SealedKLReferenceCLIError.sealedReferenceModelMismatch
    }
    let plan = try parseSealedKLReferenceReplayPlan(flags)
    guard plan.modelPath == modelPath else {
        throw SealedKLReferenceCLIError.sealedReferenceModelMismatch
    }
    return .sealedReplay(plan)
}

func parseSealedKLReferenceCapturePlan(arguments: [String]) throws
    -> SealedKLReferenceCapturePlan
{
    let parsed = try parseStrictSealedKLArguments(
        arguments,
        allowed: [
            "model", "output", "workload-nonce", "corpus", "positions",
            "long-context-sample-positions", "python", "script",
        ])
    let positions = try positiveInteger(
        parsed.value("positions", default: "24"),
        flag: "positions")
    let longContextSamplePositions = try positiveInteger(
        parsed.value("long-context-sample-positions", default: "128"),
        flag: "long-context-sample-positions")
    return SealedKLReferenceCapturePlan(
        modelPath: try parsed.required("model"),
        outputPath: try parsed.required("output"),
        workloadNonce: try parsed.required("workload-nonce"),
        corpusPath: parsed.value("corpus", default: "corpus/measurement-corpus-v2.json"),
        positions: positions,
        longContextSamplePositions: longContextSamplePositions,
        pythonPath: parsed.value("python", default: "~/harness-venv/bin/python"),
        scriptPath: parsed.value("script", default: "scripts/harness_reference.py"))
}

func parseSealedKLReferenceReplayPlan(arguments: [String]) throws
    -> SealedKLReferenceReplayPlan
{
    let parsed = try parseStrictSealedKLArguments(
        arguments,
        allowed: [
            "model", "sealed-reference", "sealed-reference-sha256",
            "workload-nonce", "corpus", "script",
        ])
    let expectedManifestSHA256 = try parsed.required("sealed-reference-sha256")
    guard isLowercaseHex(expectedManifestSHA256, length: 64) else {
        throw SealedKLReferenceCLIError.invalidSHA256("sealed-reference-sha256")
    }
    return SealedKLReferenceReplayPlan(
        modelPath: try parsed.required("model"),
        directoryPath: try parsed.required("sealed-reference"),
        expectedManifestSHA256: expectedManifestSHA256,
        workloadNonce: try parsed.required("workload-nonce"),
        corpusPath: parsed.value("corpus", default: "corpus/measurement-corpus-v2.json"),
        scriptPath: parsed.value("script", default: "scripts/harness_reference.py"))
}

func parseSealedKLReferenceCapturePlan(_ flags: Flags) throws
    -> SealedKLReferenceCapturePlan
{
    let positions = try flags.strictInt("positions", default: 24)
    let longContextSamplePositions = try flags.strictInt(
        "long-context-sample-positions",
        default: 128)
    guard positions > 0 else {
        throw SealedKLReferenceCLIError.invalidPositiveInteger(
            "positions",
            "\(positions)")
    }
    guard longContextSamplePositions > 0 else {
        throw SealedKLReferenceCLIError.invalidPositiveInteger(
            "long-context-sample-positions",
            "\(longContextSamplePositions)")
    }
    let modelPath = try flags.strictString("model", default: "")
    let outputPath = try flags.strictString("output", default: "")
    let workloadNonce = try flags.strictString("workload-nonce", default: "")
    guard !modelPath.isEmpty else { throw SealedKLReferenceCLIError.missingRequiredFlag("model") }
    guard !outputPath.isEmpty else { throw SealedKLReferenceCLIError.missingRequiredFlag("output") }
    guard !workloadNonce.isEmpty else {
        throw SealedKLReferenceCLIError.missingRequiredFlag("workload-nonce")
    }
    return SealedKLReferenceCapturePlan(
        modelPath: modelPath,
        outputPath: outputPath,
        workloadNonce: workloadNonce,
        corpusPath: try flags.strictString(
            "corpus",
            default: "corpus/measurement-corpus-v2.json"),
        positions: positions,
        longContextSamplePositions: longContextSamplePositions,
        pythonPath: try flags.strictString("python", default: "~/harness-venv/bin/python"),
        scriptPath: try flags.strictString("script", default: "scripts/harness_reference.py"))
}

func parseSealedKLReferenceReplayPlan(_ flags: Flags) throws
    -> SealedKLReferenceReplayPlan
{
    let modelPath = try flags.strictString("model", default: "")
    let directoryPath = try flags.strictString("sealed-reference", default: "")
    let expectedManifestSHA256 = try flags.strictString(
        "sealed-reference-sha256",
        default: "")
    let workloadNonce = try flags.strictString("workload-nonce", default: "")
    guard !modelPath.isEmpty else { throw SealedKLReferenceCLIError.missingRequiredFlag("model") }
    guard !directoryPath.isEmpty else {
        throw SealedKLReferenceCLIError.missingRequiredFlag("sealed-reference")
    }
    guard isLowercaseHex(expectedManifestSHA256, length: 64) else {
        throw SealedKLReferenceCLIError.invalidSHA256("sealed-reference-sha256")
    }
    guard !workloadNonce.isEmpty else {
        throw SealedKLReferenceCLIError.missingRequiredFlag("workload-nonce")
    }
    return SealedKLReferenceReplayPlan(
        modelPath: modelPath,
        directoryPath: directoryPath,
        expectedManifestSHA256: expectedManifestSHA256,
        workloadNonce: workloadNonce,
        corpusPath: try flags.strictString(
            "corpus",
            default: "corpus/measurement-corpus-v2.json"),
        scriptPath: try flags.strictString("script", default: "scripts/harness_reference.py"))
}

func runSealedKLReferenceCapture(_ plan: SealedKLReferenceCapturePlan) async throws {
    let fileManager = FileManager.default
    let outputURL = URL(fileURLWithPath: plan.outputPath, isDirectory: true)
    guard !fileManager.fileExists(atPath: outputURL.path) else {
        throw SealedKLReferenceCLIError.outputPathCollision(outputURL.path)
    }

    let sourceBefore = try captureCompressedKVAttentionRuntimeSourceSnapshot(
        modelPath: plan.modelPath)
    let corpusURL = URL(fileURLWithPath: plan.corpusPath)
    let scriptURL = URL(fileURLWithPath: plan.scriptPath)
    let corpusRawData = try readRegularFile(corpusURL)
    let scriptData = try readRegularFile(scriptURL)
    let corpus = try MeasurementCorpusLoader.load(from: corpusRawData)
    let harnessSHA = try ProvenanceCLI.qualificationHarnessGitSHA()

    let tokenizer = try await AutoTokenizer.from(
        modelFolder: URL(fileURLWithPath: plan.modelPath))
    guard let eos = tokenizer.eosToken.flatMap({ tokenizer.convertTokenToId($0) }) else {
        throw KVTunerCLIError.missingEOSToken
    }
    let reference = ReferenceDriver(
        pythonPath: (plan.pythonPath as NSString).expandingTildeInPath,
        scriptPath: plan.scriptPath,
        modelPath: plan.modelPath,
        eos: eos)
    let (entries, blobs) = try await captureSealedKLReferenceEntries(
        corpus: corpus,
        positions: plan.positions,
        longContextSamplePositions: plan.longContextSamplePositions,
        tokenizer: { tokenizer.encode(text: $0) },
        reference: reference)
    guard let reportedVersions = await reference.versionSink.versions else {
        throw SealedKLReferenceCLIError.missingReferenceRuntimeVersions
    }
    let runtimeVersions = SealedKLReferenceRuntimeVersions(
        mlx: reportedVersions.mlx,
        mlxLM: reportedVersions.mlxLM)
    let sourceAfter = try captureCompressedKVAttentionRuntimeSourceSnapshot(
        modelPath: plan.modelPath)
    guard sourceAfter == sourceBefore else {
        throw SealedKLReferenceCLIError.modelSourceDrift
    }
    guard try readRegularFile(corpusURL) == corpusRawData else {
        throw SealedKLReferenceCLIError.corpusFileDrift
    }
    guard try readRegularFile(scriptURL) == scriptData else {
        throw SealedKLReferenceCLIError.referenceScriptDrift
    }

    let binding = try sealedKLReferenceBinding(
        modelPath: plan.modelPath,
        sourceSnapshot: sourceBefore,
        checkpointManifestSHA256: try checkpointManifestSHA256(at: plan.modelPath),
        corpus: corpus,
        corpusRawData: corpusRawData,
        referenceScriptData: scriptData,
        referenceRuntimeVersions: runtimeVersions,
        workloadNonce: plan.workloadNonce,
        harnessGitSHA: harnessSHA)
    let manifest = SealedKLReferenceManifest(
        schema: "sealed-kl-reference",
        version: 1,
        identity: binding,
        corpus: corpus.corpusId,
        referenceVersion: "harness-reference-py-\(String(binding.referenceScriptSHA256.prefix(12)))",
        maxTokens: plan.positions,
        sampleSize: plan.longContextSamplePositions,
        entries: entries)
    try writeSealedKLReferenceBundleAtomically(
        manifest: manifest,
        blobs: blobs,
        to: outputURL)
}

func prepareSealedKLReferenceReplay(_ plan: SealedKLReferenceReplayPlan) async throws
    -> PreparedSealedKLReferenceReplay
{
    let source = try captureCompressedKVAttentionRuntimeSourceSnapshot(
        modelPath: plan.modelPath)
    let corpusRawData = try readRegularFile(URL(fileURLWithPath: plan.corpusPath))
    let scriptData = try readRegularFile(URL(fileURLWithPath: plan.scriptPath))
    let corpus = try MeasurementCorpusLoader.load(from: corpusRawData)
    let tokenizer = try await AutoTokenizer.from(
        modelFolder: URL(fileURLWithPath: plan.modelPath))
    let binding = try sealedKLReferenceBinding(
        modelPath: plan.modelPath,
        sourceSnapshot: source,
        checkpointManifestSHA256: try checkpointManifestSHA256(at: plan.modelPath),
        corpus: corpus,
        corpusRawData: corpusRawData,
        referenceScriptData: scriptData,
        referenceRuntimeVersions: SealedKLReferenceRuntimeVersions(
            mlx: "placeholder",
            mlxLM: "placeholder"),
        workloadNonce: plan.workloadNonce,
        harnessGitSHA: try ProvenanceCLI.qualificationHarnessGitSHA())

    let partiallyPrepared = try loadSealedKLReferenceReplay(
        directory: URL(fileURLWithPath: plan.directoryPath, isDirectory: true),
        expectedManifestSHA256: plan.expectedManifestSHA256,
        expectedBindingWithoutRuntimeVersions: binding,
        sourceSnapshot: source,
        expectedCorpus: corpus.corpusId,
        corpus: corpus,
        tokenize: { tokenizer.encode(text: $0) })
    return partiallyPrepared
}

func loadSealedKLReferenceReplay(
    directory: URL,
    expectedManifestSHA256: String,
    expectedBinding: SealedKLReferenceBinding,
    sourceSnapshot: CompressedKVAttentionRuntimeSourceSnapshot,
    expectedCorpus: String,
    corpus: MeasurementCorpus,
    tokenize: (String) throws -> [Int]
) throws -> PreparedSealedKLReferenceReplay {
    try loadSealedKLReferenceReplay(
        directory: directory,
        expectedManifestSHA256: expectedManifestSHA256,
        expectedBindingWithoutRuntimeVersions: expectedBinding,
        sourceSnapshot: sourceSnapshot,
        expectedCorpus: expectedCorpus,
        corpus: corpus,
        tokenize: tokenize)
}

func loadSealedKLReferenceReplay(
    directory: URL,
    expectedManifestSHA256: String,
    expectedBindingWithoutRuntimeVersions expectedBinding: SealedKLReferenceBinding,
    sourceSnapshot: CompressedKVAttentionRuntimeSourceSnapshot,
    expectedCorpus: String,
    corpus: MeasurementCorpus,
    tokenize: (String) throws -> [Int]
) throws -> PreparedSealedKLReferenceReplay {
    guard isLowercaseHex(expectedManifestSHA256, length: 64) else {
        throw SealedKLReferenceCLIError.invalidSHA256("sealed-reference-sha256")
    }
    let manifestData = try readSealedKLReferenceManifest(directory)
    let actualManifestSHA256 = SealedKLReferenceBundle.sha256Hex(manifestData)
    guard actualManifestSHA256 == expectedManifestSHA256 else {
        throw SealedKLReferenceCLIError.manifestSHA256Mismatch(
            expected: expectedManifestSHA256,
            actual: actualManifestSHA256)
    }
    let manifest = try JSONDecoder().decode(SealedKLReferenceManifest.self, from: manifestData)
    let blobs = try readSealedKLReferenceBlobs(
        directory,
        declaredBy: manifest)
    var exactBinding = expectedBinding
    exactBinding = SealedKLReferenceBinding(
        model: expectedBinding.model,
        harnessGitSHA: expectedBinding.harnessGitSHA,
        modelConfigSHA256: expectedBinding.modelConfigSHA256,
        checkpointManifestSHA256: expectedBinding.checkpointManifestSHA256,
        checkpointContentSHA256: expectedBinding.checkpointContentSHA256,
        tokenizer: expectedBinding.tokenizer,
        tokenizerManifestSHA256: expectedBinding.tokenizerManifestSHA256,
        corpusContentHash: expectedBinding.corpusContentHash,
        corpusRawFileSHA256: expectedBinding.corpusRawFileSHA256,
        harness: expectedBinding.harness,
        referenceScriptSHA256: expectedBinding.referenceScriptSHA256,
        referenceRuntimeVersions: manifest.identity.referenceRuntimeVersions,
        workloadNonce: expectedBinding.workloadNonce)
    let bundle = try SealedKLReferenceBundle(
        manifestData: manifestData,
        blobs: blobs,
        expectedBinding: exactBinding,
        expectedCorpus: expectedCorpus)
    try validateSealedKLReferenceCorpusTokenization(
        manifest: bundle.manifest,
        corpus: corpus,
        tokenize: tokenize)
    let versions = ReferenceDriver.ReferenceVersions(
        mlx: bundle.referenceRuntimeVersions.mlx,
        mlxLM: bundle.referenceRuntimeVersions.mlxLM)
    return PreparedSealedKLReferenceReplay(
        driver: SealedKLReferenceDriver(bundle: bundle),
        referenceVersions: versions,
        manifestSHA256: bundle.manifestSHA256,
        sourceSnapshot: sourceSnapshot)
}

func validateSealedKLReferenceSourceUnchanged(
    before: CompressedKVAttentionRuntimeSourceSnapshot,
    after: CompressedKVAttentionRuntimeSourceSnapshot
) throws {
    guard before == after else {
        throw SealedKLReferenceCLIError.modelSourceDrift
    }
}

func sealedKLReferenceLogitsBlob(_ rows: [[Float]]) throws -> Data {
    var data = Data()
    for row in rows {
        for value in row {
            guard value.isFinite else {
                throw SealedKLReferenceCLIError.nonFiniteLogit
            }
            var littleEndian = value.bitPattern.littleEndian
            withUnsafeBytes(of: &littleEndian) {
                data.append(contentsOf: $0)
            }
        }
    }
    return data
}

func sealedKLReferenceBinding(
    modelPath: String,
    sourceSnapshot: CompressedKVAttentionRuntimeSourceSnapshot,
    corpus: MeasurementCorpus,
    corpusRawData: Data,
    referenceScriptData: Data,
    referenceRuntimeVersions: SealedKLReferenceRuntimeVersions,
    workloadNonce: String,
    harnessGitSHA: String
) throws -> SealedKLReferenceBinding {
    try sealedKLReferenceBinding(
        modelPath: modelPath,
        sourceSnapshot: sourceSnapshot,
        checkpointManifestSHA256: normalizedCheckpointManifestSHA256(
            sourceSnapshot.checkpointManifestHash),
        corpus: corpus,
        corpusRawData: corpusRawData,
        referenceScriptData: referenceScriptData,
        referenceRuntimeVersions: referenceRuntimeVersions,
        workloadNonce: workloadNonce,
        harnessGitSHA: harnessGitSHA)
}

func sealedKLReferenceBinding(
    modelPath: String,
    sourceSnapshot: CompressedKVAttentionRuntimeSourceSnapshot,
    checkpointManifestSHA256: String,
    corpus: MeasurementCorpus,
    corpusRawData: Data,
    referenceScriptData: Data,
    referenceRuntimeVersions: SealedKLReferenceRuntimeVersions,
    workloadNonce: String,
    harnessGitSHA: String
) throws -> SealedKLReferenceBinding {
    let model = try stableIdentifier(
        field: "model",
        value: URL(fileURLWithPath: modelPath).lastPathComponent)
    let tokenizer = try stableIdentifier(
        field: "tokenizer",
        value: "hf-tokenizer-\(sourceSnapshot.tokenizerSHA256.prefix(12))")
    _ = try stableIdentifier(field: "workloadNonce", value: workloadNonce)
    _ = try stableIdentifier(field: "harness", value: sealedKLReferenceHarnessIdentity)
    return SealedKLReferenceBinding(
        model: model,
        harnessGitSHA: harnessGitSHA,
        modelConfigSHA256: SealedKLReferenceBundle.sha256Hex(
            sourceSnapshot.exactModelConfigData),
        checkpointManifestSHA256: checkpointManifestSHA256,
        checkpointContentSHA256: sourceSnapshot.checkpointContentSHA256,
        tokenizer: tokenizer,
        tokenizerManifestSHA256: sourceSnapshot.tokenizerSHA256,
        corpusContentHash: corpus.contentHash,
        corpusRawFileSHA256: SealedKLReferenceBundle.sha256Hex(corpusRawData),
        harness: sealedKLReferenceHarnessIdentity,
        referenceScriptSHA256: SealedKLReferenceBundle.sha256Hex(referenceScriptData),
        referenceRuntimeVersions: referenceRuntimeVersions,
        workloadNonce: workloadNonce)
}

func writeSealedKLReferenceBundleAtomically(
    manifest: SealedKLReferenceManifest,
    blobs: [String: Data],
    to outputURL: URL,
    afterStagingWrite: () throws -> Void = {}
) throws {
    let manager = FileManager.default
    let finalURL = outputURL.standardizedFileURL
    guard !manager.fileExists(atPath: finalURL.path) else {
        throw SealedKLReferenceCLIError.outputPathCollision(finalURL.path)
    }
    let parent = finalURL.deletingLastPathComponent()
    try manager.createDirectory(at: parent, withIntermediateDirectories: true)
    let stagingURL = parent.appendingPathComponent(
        ".\(finalURL.lastPathComponent).staging-\(UUID().uuidString)",
        isDirectory: true)
    guard !manager.fileExists(atPath: stagingURL.path) else {
        throw SealedKLReferenceCLIError.stagingPathCollision(stagingURL.path)
    }
    try manager.createDirectory(at: stagingURL, withIntermediateDirectories: false)
    do {
        for (filename, blob) in blobs {
            try validateSafeBlobFilename(filename)
            try blob.write(to: stagingURL.appendingPathComponent(filename), options: .atomic)
        }
        let manifestData = try SealedKLReferenceBundle.canonicalManifestData(manifest)
        _ = try SealedKLReferenceBundle(
            manifestData: manifestData,
            blobs: blobs,
            expectedBinding: manifest.identity,
            expectedCorpus: manifest.corpus)
        try manifestData.write(
            to: stagingURL.appendingPathComponent("manifest.json"),
            options: .atomic)
        try afterStagingWrite()
        try manager.moveItem(at: stagingURL, to: finalURL)
    } catch {
        try? manager.removeItem(at: stagingURL)
        throw error
    }
}

private func captureSealedKLReferenceEntries(
    corpus: MeasurementCorpus,
    positions: Int,
    longContextSamplePositions: Int,
    tokenizer: (String) throws -> [Int],
    reference: ReferenceDriver
) async throws -> (entries: [SealedKLReferenceEntry], blobs: [String: Data]) {
    let config = RunConfig.greedy(maxTokens: positions)
    var entries: [SealedKLReferenceEntry] = []
    var blobs: [String: Data] = [:]

    for entry in corpus.entries where entry.tag == .prose || entry.tag == .code {
        let prompt = try tokenizer(entry.text)
        let continuation = try await reference.generate(
            prompt: prompt,
            config: config).tokens
        let rows = try await reference.logprobs(
            prompt: prompt,
            forcedContinuation: continuation,
            config: config)
        let blob = try sealedKLReferenceLogitsBlob(rows)
        let filename = "\(entry.id).f32"
        entries.append(SealedKLReferenceEntry(
            id: entry.id,
            tag: entry.tag.rawValue,
            promptTokenIDs: prompt,
            continuationTokenIDs: continuation,
            samplePositions: nil,
            logitsFile: filename,
            logitsSHA256: SealedKLReferenceBundle.sha256Hex(blob),
            rowCount: rows.count,
            vocabSize: rows.first?.count ?? 0,
            byteCount: blob.count))
        blobs[filename] = blob
    }

    for entry in corpus.entries where entry.tag == .longContext {
        let docTokens = try tokenizer(entry.text)
        guard docTokens.count > 1 else {
            throw SealedKLReferenceCLIError.corpusTokenizationMismatch(entry.id)
        }
        let prompt = [docTokens[0]]
        let continuation = Array(docTokens.dropFirst())
        let sampled = evenlySpacedPositions(
            total: continuation.count,
            sampleSize: longContextSamplePositions)
        let rows = try await reference.logprobs(
            prompt: prompt,
            forcedContinuation: continuation,
            atPositions: sampled,
            config: config)
        let blob = try sealedKLReferenceLogitsBlob(rows)
        let filename = "\(entry.id).f32"
        entries.append(SealedKLReferenceEntry(
            id: entry.id,
            tag: entry.tag.rawValue,
            promptTokenIDs: prompt,
            continuationTokenIDs: continuation,
            samplePositions: sampled,
            logitsFile: filename,
            logitsSHA256: SealedKLReferenceBundle.sha256Hex(blob),
            rowCount: rows.count,
            vocabSize: rows.first?.count ?? 0,
            byteCount: blob.count))
        blobs[filename] = blob
    }
    return (entries, blobs)
}

private func validateSealedKLReferenceCorpusTokenization(
    manifest: SealedKLReferenceManifest,
    corpus: MeasurementCorpus,
    tokenize: (String) throws -> [Int]
) throws {
    guard manifest.entries.count == corpus.entries.count else {
        throw SealedKLReferenceCLIError.entrySetMismatch
    }
    let entriesByID = Dictionary(uniqueKeysWithValues: manifest.entries.map { ($0.id, $0) })
    guard entriesByID.count == manifest.entries.count else {
        throw SealedKLReferenceCLIError.entrySetMismatch
    }
    for corpusEntry in corpus.entries {
        guard let sealed = entriesByID[corpusEntry.id],
              sealed.tag == corpusEntry.tag.rawValue
        else {
            throw SealedKLReferenceCLIError.entrySetMismatch
        }
        let tokens = try tokenize(corpusEntry.text)
        switch corpusEntry.tag {
        case .prose, .code:
            guard sealed.promptTokenIDs == tokens,
                  sealed.samplePositions == nil,
                  sealed.rowCount == sealed.continuationTokenIDs.count
            else {
                throw SealedKLReferenceCLIError.corpusTokenizationMismatch(corpusEntry.id)
            }
        case .longContext:
            guard tokens.count > 1 else {
                throw SealedKLReferenceCLIError.corpusTokenizationMismatch(corpusEntry.id)
            }
            let continuation = Array(tokens.dropFirst())
            let sampled = evenlySpacedPositions(
                total: continuation.count,
                sampleSize: manifest.sampleSize)
            guard sealed.promptTokenIDs == [tokens[0]],
                  sealed.continuationTokenIDs == continuation,
                  sealed.samplePositions == sampled
            else {
                throw SealedKLReferenceCLIError.corpusTokenizationMismatch(corpusEntry.id)
            }
        }
    }
}

private struct ParsedSealedKLArguments {
    let values: [String: String]

    func required(_ flag: String) throws -> String {
        guard let value = values[flag], !value.isEmpty else {
            throw SealedKLReferenceCLIError.missingRequiredFlag(flag)
        }
        return value
    }

    func value(_ flag: String, default defaultValue: String) -> String {
        values[flag] ?? defaultValue
    }
}

private func parseStrictSealedKLArguments(
    _ arguments: [String],
    allowed: Set<String>
) throws -> ParsedSealedKLArguments {
    var values: [String: String] = [:]
    var index = 0
    while index < arguments.count {
        let raw = arguments[index]
        guard raw.hasPrefix("--") else {
            throw SealedKLReferenceCLIError.unsupportedFlag(raw)
        }
        let flag = String(raw.dropFirst(2))
        guard allowed.contains(flag) else {
            throw SealedKLReferenceCLIError.unsupportedFlag(flag)
        }
        guard index + 1 < arguments.count,
              !arguments[index + 1].hasPrefix("--")
        else {
            throw SealedKLReferenceCLIError.missingValue(flag)
        }
        values[flag] = arguments[index + 1]
        index += 2
    }
    return ParsedSealedKLArguments(values: values)
}

private func positiveInteger(_ raw: String, flag: String) throws -> Int {
    guard let value = Int(raw), value > 0 else {
        throw SealedKLReferenceCLIError.invalidPositiveInteger(flag, raw)
    }
    return value
}

private func readSealedKLReferenceManifest(_ directory: URL) throws -> Data {
    try rejectSymlink(directory)
    let manifestURL = directory.appendingPathComponent("manifest.json")
    return try readRegularFile(manifestURL)
}

private func readSealedKLReferenceBlobs(
    _ directory: URL,
    declaredBy manifest: SealedKLReferenceManifest
) throws -> [String: Data] {
    var declared = Set<String>()
    for entry in manifest.entries {
        try validateSafeBlobFilename(entry.logitsFile)
        guard declared.insert(entry.logitsFile).inserted else {
            throw SealedKLReferenceError.duplicateFile(entry.logitsFile)
        }
    }
    let urls = try FileManager.default.contentsOfDirectory(
        at: directory,
        includingPropertiesForKeys: [
            .fileSizeKey,
            .isRegularFileKey,
            .isSymbolicLinkKey,
        ],
        options: [])
    let expectedNames = declared.union(["manifest.json"])
    guard Set(urls.map(\.lastPathComponent)) == expectedNames else {
        throw SealedKLReferenceCLIError.entrySetMismatch
    }
    let urlsByName = Dictionary(
        uniqueKeysWithValues: urls.map { ($0.lastPathComponent, $0) })
    var blobs: [String: Data] = [:]
    for entry in manifest.entries {
        guard let url = urlsByName[entry.logitsFile] else {
            throw SealedKLReferenceCLIError.entrySetMismatch
        }
        try rejectSymlink(url)
        let values = try url.resourceValues(
            forKeys: [.fileSizeKey, .isRegularFileKey])
        guard values.isRegularFile == true,
              let fileSize = values.fileSize
        else {
            throw SealedKLReferenceCLIError.invalidDirectoryEntry(url.path)
        }
        guard entry.byteCount == fileSize else {
            throw SealedKLReferenceError.byteCountMismatch(
                entryID: entry.id,
                expected: entry.byteCount,
                actual: fileSize)
        }
        blobs[entry.logitsFile] = try Data(contentsOf: url)
    }
    return blobs
}

private func readRegularFile(_ url: URL) throws -> Data {
    try rejectSymlink(url)
    let values = try url.resourceValues(forKeys: [.isRegularFileKey])
    guard values.isRegularFile == true else {
        throw SealedKLReferenceCLIError.invalidDirectoryEntry(url.path)
    }
    return try Data(contentsOf: url)
}

private func rejectSymlink(_ url: URL) throws {
    let values = try url.resourceValues(forKeys: [.isSymbolicLinkKey])
    if values.isSymbolicLink == true {
        throw SealedKLReferenceCLIError.symlinkRejected(url.path)
    }
}

private func checkpointManifestSHA256(at modelPath: String) throws -> String {
    let directory = URL(fileURLWithPath: modelPath, isDirectory: true)
        .standardizedFileURL
    let manager = FileManager.default
    var hasher = CryptoKit.SHA256()
    hasher.update(data: Data("fastmlx-sealed-reference-checkpoint-manifest-v1\n".utf8))
    func updateField(_ data: Data) {
        var count = UInt64(data.count).bigEndian
        withUnsafeBytes(of: &count) { hasher.update(bufferPointer: $0) }
        hasher.update(data: data)
    }
    updateField(try Data(contentsOf: directory.appendingPathComponent("config.json")))
    let index = directory.appendingPathComponent("model.safetensors.index.json")
    if manager.fileExists(atPath: index.path) {
        updateField(try readRegularFile(index))
    } else {
        updateField(Data())
    }
    let weights = try manager.contentsOfDirectory(
        at: directory,
        includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
        options: [.skipsHiddenFiles])
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    guard !weights.isEmpty else {
        throw ProvenanceCLI.EvidenceIdentityError.missingCheckpointWeights(modelPath)
    }
    for weight in weights {
        try rejectSymlink(weight)
        let values = try weight.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard values.isRegularFile == true, let size = values.fileSize, size >= 0 else {
            throw ProvenanceCLI.EvidenceIdentityError.invalidCheckpointWeight(weight.path)
        }
        updateField(Data("\(weight.lastPathComponent):\(size)\n".utf8))
    }
    return hasher.finalize().map {
        String(format: "%02x", $0)
    }.joined()
}

private func normalizedCheckpointManifestSHA256(_ value: String) -> String {
    if isLowercaseHex(value, length: 64) { return value }
    return SealedKLReferenceBundle.sha256Hex(Data(value.utf8))
}

private func validateSafeBlobFilename(_ filename: String) throws {
    guard !filename.isEmpty,
          filename != ".",
          filename != "..",
          !filename.contains("/"),
          !filename.contains("\\"),
          !filename.contains("..")
    else {
        throw SealedKLReferenceError.invalidLogitsFile(filename)
    }
}

private func stableIdentifier(field: String, value: String) throws -> String {
    guard !value.isEmpty,
          value == value.trimmingCharacters(in: .whitespacesAndNewlines),
          !value.contains("/"),
          !value.contains("\\"),
          !value.contains(".."),
          value.unicodeScalars.allSatisfy({ $0.value >= 0x20 && $0.value != 0x7f })
    else {
        throw SealedKLReferenceCLIError.invalidStableIdentifier(field, value)
    }
    return value
}

private func isLowercaseHex(_ value: String, length: Int) -> Bool {
    let hex = Set("0123456789abcdef")
    return value.count == length && value.allSatisfy { hex.contains($0) }
}
