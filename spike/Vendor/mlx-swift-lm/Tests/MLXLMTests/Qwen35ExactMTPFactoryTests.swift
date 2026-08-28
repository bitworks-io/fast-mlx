// Copyright © 2026 Apple Inc.

import Foundation
import CryptoKit
@testable import MLXLMCommon
import Testing

@_spi(FastMLXExactMTP) @testable import MLXLLM

@Test
func testQwen35MTPGenericRegistryRemainsClosed() async throws {
    do {
        let _: any MTPDrafterModel = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(), modelType: "qwen3_5_mtp")
        Issue.record("qwen3_5_mtp must not be creatable through the generic MTP registry")
    } catch let error as ModelFactoryError {
        guard case .unsupportedModelType(let modelType) = error else {
            Issue.record("unexpected registry error: \(error)")
            return
        }
        #expect(modelType == "qwen3_5_mtp")
    }
}

@Test
func testQwen35ExactMTPKnownArtifactRuntimeMetadata() {
    let lock = Qwen35ExactMTPKnownArtifactLocks.qwen35_9BDepth1

    #expect(lock.target.modelID == "mlx-community/Qwen3.5-9B-MLX-4bit")
    #expect(lock.drafter.modelID == "mlx-community/Qwen3.5-9B-MTP-5bit")
    #expect(lock.runtimeBlockSize == 3)
    #expect(lock.maximumAcceptedDraftTokens == 2)
    #expect(lock.architecture.mtpDepth == 1)
}

@Test
func testQwen38ExactMTPKnownArtifactRuntimeMetadata() {
    let lock = Qwen35ExactMTPKnownArtifactLocks.qwen38_27BMXFP8Depth1

    #expect(lock.target.modelID == "mlx-community/Qwen3.8-27B-mxfp8")
    #expect(lock.drafter.modelID == "mlx-community/Qwen3.8-27B-MTP-mxfp8")
    #expect(lock.target.revision == "d48d163bcdf24acaf656474854ab88ea17d65bd1")
    #expect(lock.drafter.revision == "a50634460045613f166b09b13519466e801c6568")
    #expect(lock.targetQuantization == .init(bits: 8, groupSize: 32, mode: "mxfp8"))
    #expect(lock.drafterQuantization == .init(bits: 8, groupSize: 32, mode: "mxfp8"))
    #expect(lock.drafterTensors.count == 23)
    #expect(lock.runtimeBlockSize == 3)
    #expect(lock.maximumAcceptedDraftTokens == 2)
    #expect(lock.architecture.mtpDepth == 1)
}

@Test
func testQwen35ExactMTPPublicLoaderResolvesFixedPairWithUseLatestFalse() async throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let target = root.appending(component: "target")
    let drafter = root.appending(component: "drafter")
    try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: drafter, withIntermediateDirectories: true)
    let downloader = RecordingDownloader(directories: [
        "mlx-community/Qwen3.5-9B-MLX-4bit": target,
        "mlx-community/Qwen3.5-9B-MTP-5bit": drafter,
    ])

    _ = try? await Qwen35ExactMTPFactory.loadDepth1Pair(
        from: downloader,
        using: UnusedTokenizerLoader()
    ) { _ in }

    let calls = await downloader.calls()
    #expect(calls.count == 2)
    #expect(calls.map(\.id) == [
        "mlx-community/Qwen3.5-9B-MLX-4bit",
        "mlx-community/Qwen3.5-9B-MTP-5bit",
    ])
    #expect(calls.map(\.revision) == [
        "938d8919941c6e7efd3c7150eff7fe9d12afa631",
        "994730d199bff7799aa3ddef33a96723967a3e33",
    ])
    #expect(calls.allSatisfy { !$0.useLatest })
    #expect(calls.allSatisfy { $0.patterns == ["*.safetensors", "*.json", "*.jinja"] })
}

@Test
func testQwen38ExactMTPSelectedLoaderResolvesFixedPairWithUseLatestFalse() async throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let target = root.appending(component: "target")
    let drafter = root.appending(component: "drafter")
    try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: drafter, withIntermediateDirectories: true)
    let downloader = RecordingDownloader(directories: [
        "mlx-community/Qwen3.8-27B-mxfp8": target,
        "mlx-community/Qwen3.8-27B-MTP-mxfp8": drafter,
    ])

    _ = try? await Qwen35ExactMTPFactory.loadDepth1Pair(
        selection: .qwen38_27BMXFP8Depth1,
        from: downloader,
        using: UnusedTokenizerLoader()
    ) { _ in }

    let calls = await downloader.calls()
    #expect(calls.count == 2)
    #expect(calls.map(\.id) == [
        "mlx-community/Qwen3.8-27B-mxfp8",
        "mlx-community/Qwen3.8-27B-MTP-mxfp8",
    ])
    #expect(calls.map(\.revision) == [
        "d48d163bcdf24acaf656474854ab88ea17d65bd1",
        "a50634460045613f166b09b13519466e801c6568",
    ])
    #expect(calls.allSatisfy { !$0.useLatest })
    #expect(calls.allSatisfy { $0.patterns == ["*.safetensors", "*.json", "*.jinja"] })
}

@Test
func testQwen35ExactMTPResolvedAdmissionPassesForSyntheticDepthOnePair() throws {
    let fixture = try ExactMTPFixture()
    defer { fixture.cleanup() }

    let evidence = try fixture.admit()

    #expect(evidence.binding.runtimeBlockSize == 3)
    #expect(evidence.binding.maximumAcceptedDraftTokens == 2)
    #expect(evidence.target.tensors == ExactMTPFixture.targetTensors)
    #expect(evidence.drafter.tensors == ExactMTPFixture.drafterTensors)
}

@Test
func testQwen35ExactMTPTokenizerDigestMatchesJQSortedCompactBytes() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let tokenizer = [
        "model": [
            "vocab": ["a/b": 1, "é": 2, "e": 3, "🐈": 4, "\u{2028}": 5]
        ]
    ]
    let data = try JSONSerialization.data(withJSONObject: tokenizer)
    try data.write(to: root.appending(component: "tokenizer.json"))

    let digest = try Qwen35ExactMTPAdmission.canonicalTokenizerVocabularySHA256(
        in: root, role: .target)

    #expect(digest == "22695b7ac6d7da7e26be0f2277d4299227f3f69e461669cbaa7fd82c3ed36ec3")
}

@Test
func testQwen35ExactMTPDirectoryEvidenceMutationsFailAdmission() throws {
    try expectDirectoryAdmissionError(
        .identityMismatch(role: .drafter, field: "configSHA256")
    ) { fixture in
        try fixture.writeConfig(role: .drafter, data: ExactMTPFixture.drafterConfig(bits: 4))
    }

    try expectDirectoryAdmissionError(
        .identityMismatch(role: .target, field: "tokenizerSHA256")
    ) { fixture in
        try fixture.writeTokenizer(role: .target, vocab: ["token-a": 0, "different": 1])
    }

    try expectDirectoryAdmissionError(
        .identityMismatch(role: .drafter, field: "tensorManifestSHA256")
    ) { fixture in
        try fixture.writeSafetensors(
            role: .drafter,
            tensors: [ExactMTPFixture.drafterTensors[0].renamed("wrong.weight")])
    }

    try expectDirectoryAdmissionError(
        .identityMismatch(role: .target, field: "tensorManifestSHA256")
    ) { fixture in
        try fixture.writeSafetensors(
            role: .target,
            tensors: [ExactMTPFixture.targetTensors[0].reshaped([1, 1])])
    }

    try expectDirectoryAdmissionError(
        .incompleteTensorManifest(role: .drafter)
    ) { fixture in
        try fixture.writeIndex(
            role: .drafter,
            weightMap: ["fc.weight": "missing.safetensors"])
    }

    try expectDirectoryAdmissionError(
        .noSafetensors(role: .drafter)
    ) { fixture in
        try FileManager.default.removeItem(
            at: fixture.directory(role: .drafter).appending(component: "model.safetensors"))
    }

    let swapped = try ExactMTPFixture()
    defer { swapped.cleanup() }
    try expectValidationError(.identityMismatch(role: .target, field: "configSHA256")) {
        _ = try Qwen35ExactMTPFactory.admitResolvedDepth1Pair(
            lock: swapped.lock,
            targetDirectory: swapped.directory(role: .drafter),
            targetTokenizerDirectory: swapped.directory(role: .drafter),
            drafterDirectory: swapped.directory(role: .target),
            drafterTokenizerDirectory: swapped.directory(role: .target))
    }
}

@Test
func testQwen35ExactMTPInternalValidationRejectsSemanticMutations() throws {
    let fixture = try ExactMTPFixture()
    defer { fixture.cleanup() }

    try expectValidationError(
        .unsupportedModelType(role: .drafter, actual: "qwen3_5")
    ) {
        let config = ExactMTPFixture.drafterConfig(modelType: "qwen3_5")
        _ = try Qwen35ExactMTPAdmission.validate(
            lock: fixture.lock.withDrafterConfig(config),
            target: fixture.targetCandidate,
            drafter: fixture.drafterCandidate.withConfig(config))
    }

    try expectValidationError(
        .unsupportedMTPDepth(role: .drafter, actual: 2)
    ) {
        let config = ExactMTPFixture.drafterConfig(mtpLayers: 2)
        _ = try Qwen35ExactMTPAdmission.validate(
            lock: fixture.lock.withDrafterConfig(config),
            target: fixture.targetCandidate,
            drafter: fixture.drafterCandidate.withConfig(config))
    }

    try expectValidationError(.quantizationMismatch(role: .drafter)) {
        let config = ExactMTPFixture.drafterConfig(bits: 4)
        _ = try Qwen35ExactMTPAdmission.validate(
            lock: fixture.lock.withDrafterConfig(config),
            target: fixture.targetCandidate,
            drafter: fixture.drafterCandidate.withConfig(config))
    }

    for blockSize in [2, 4] {
        try expectValidationError(.unsupportedBlockSize(blockSize)) {
            let config = ExactMTPFixture.drafterConfig(blockSize: blockSize)
            _ = try Qwen35ExactMTPAdmission.validate(
                lock: fixture.lock.withDrafterConfig(config),
                target: fixture.targetCandidate,
                drafter: fixture.drafterCandidate.withConfig(config))
        }
    }

    try expectValidationError(
        .architectureMismatch(role: .drafter, field: "hidden_size")
    ) {
        let config = ExactMTPFixture.drafterConfig(hiddenSize: 8192)
        _ = try Qwen35ExactMTPAdmission.validate(
            lock: fixture.lock.withDrafterConfig(config),
            target: fixture.targetCandidate,
            drafter: fixture.drafterCandidate.withConfig(config))
    }

    try expectValidationError(.missingTensor("pre_fc_norm_hidden.weight")) {
        let tensors = Array(ExactMTPFixture.drafterTensors.dropLast())
        _ = try Qwen35ExactMTPAdmission.validate(
            lock: fixture.lock.withDrafterTensorManifest(tensors),
            target: fixture.targetCandidate,
            drafter: fixture.drafterCandidate.withTensors(tensors))
    }

    try expectValidationError(.unexpectedTensor("unexpected.weight")) {
        let tensors = ExactMTPFixture.drafterTensors + [
            .init(name: "unexpected.weight", shape: [1], dtype: "BF16")
        ]
        _ = try Qwen35ExactMTPAdmission.validate(
            lock: fixture.lock.withDrafterTensorManifest(tensors),
            target: fixture.targetCandidate,
            drafter: fixture.drafterCandidate.withTensors(tensors))
    }

    try expectValidationError(.tensorDescriptorMismatch("fc.weight")) {
        var tensors = ExactMTPFixture.drafterTensors
        tensors[0] = tensors[0].reshaped([2, 2])
        _ = try Qwen35ExactMTPAdmission.validate(
            lock: fixture.lock.withDrafterTensorManifest(tensors),
            target: fixture.targetCandidate,
            drafter: fixture.drafterCandidate.withTensors(tensors))
    }

    try expectValidationError(.tensorDescriptorMismatch("fc.weight")) {
        var tensors = ExactMTPFixture.drafterTensors
        tensors[0] = tensors[0].withDType("F32")
        _ = try Qwen35ExactMTPAdmission.validate(
            lock: fixture.lock.withDrafterTensorManifest(tensors),
            target: fixture.targetCandidate,
            drafter: fixture.drafterCandidate.withTensors(tensors))
    }
}

@Test
func testQwen35ExactMTPAuthorizationCallbackRunsBeforeConstructionAndCanRefuse() async throws {
    struct Refusal: Error {}

    let fixture = try ExactMTPFixture()
    defer { fixture.cleanup() }
    let callbackProbe = CallbackProbe()

    do {
        _ = try await Qwen35ExactMTPFactory.loadResolvedDepth1Pair(
            lock: fixture.lock,
            target: fixture.targetResolved,
            drafter: fixture.drafterResolved,
            tokenizerLoader: UnusedTokenizerLoader()
        ) { evidence in
            await callbackProbe.markRan()
            #expect(evidence.binding.runtimeBlockSize == 3)
            #expect(evidence.binding.maximumAcceptedDraftTokens == 2)
            throw Refusal()
        }
        Issue.record("expected callback refusal")
    } catch is Refusal {
        #expect(await callbackProbe.didRun())
    } catch {
        Issue.record("unexpected error: \(error)")
    }
}

private actor CallbackProbe {
    private var ran = false

    func markRan() {
        ran = true
    }

    func didRun() -> Bool {
        ran
    }
}

private actor RecordingDownloader: Downloader {
    struct Call: Equatable, Sendable {
        let id: String
        let revision: String?
        let patterns: [String]
        let useLatest: Bool
    }

    private let directories: [String: URL]
    private var recordedCalls: [Call] = []

    init(directories: [String: URL]) {
        self.directories = directories
    }

    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        progressHandler(Progress(totalUnitCount: 1))
        recordedCalls.append(.init(
            id: id, revision: revision, patterns: patterns, useLatest: useLatest))
        return directories[id] ?? URL(fileURLWithPath: "/dev/null")
    }

    func calls() -> [Call] {
        recordedCalls
    }
}

private struct UnusedTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer {
        TestTokenizer()
    }
}

private func expectDirectoryAdmissionError(
    _ expected: Qwen35ExactMTPAdmissionError,
    mutation: (ExactMTPFixture) throws -> Void
) throws {
    let fixture = try ExactMTPFixture()
    defer { fixture.cleanup() }
    try mutation(fixture)
    try expectValidationError(expected) {
        _ = try fixture.admit()
    }
}

private func expectValidationError(
    _ expected: Qwen35ExactMTPAdmissionError,
    operation: () throws -> Void
) rethrows {
    do {
        try operation()
        Issue.record("expected \(expected)")
    } catch let error as Qwen35ExactMTPAdmissionError {
        #expect(error == expected)
    } catch {
        Issue.record("unexpected error: \(error)")
    }
}

private struct ExactMTPFixture {
    let root: URL
    let targetResolved: ResolvedModelConfiguration
    let drafterResolved: ResolvedModelConfiguration
    let targetCandidate: Qwen35ExactMTPArtifactCandidate
    let drafterCandidate: Qwen35ExactMTPArtifactCandidate
    let lock: Qwen35ExactMTPArtifactLock

    init() throws {
        root = try temporaryDirectory()
        let targetDirectory = root.appending(component: "target")
        let drafterDirectory = root.appending(component: "drafter")
        try FileManager.default.createDirectory(
            at: targetDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: drafterDirectory, withIntermediateDirectories: true)

        targetResolved = ResolvedModelConfiguration(directory: targetDirectory)
        drafterResolved = ResolvedModelConfiguration(directory: drafterDirectory)

        let tokenizerDigest = testSHA256(canonicalVocabularyData(Self.vocab))
        let targetConfig = Self.targetConfig()
        let drafterConfig = Self.drafterConfig()
        let targetTensorDigest = Qwen35ExactMTPAdmission.tensorManifestSHA256(Self.targetTensors)
        let drafterTensorDigest = Qwen35ExactMTPAdmission.tensorManifestSHA256(Self.drafterTensors)
        let targetIdentity = Qwen35ExactMTPArtifactIdentity(
            modelID: "synthetic/qwen35-target",
            revision: String(repeating: "1", count: 40),
            configSHA256: testSHA256(targetConfig),
            tokenizerSHA256: tokenizerDigest,
            tensorManifestSHA256: targetTensorDigest)
        let drafterIdentity = Qwen35ExactMTPArtifactIdentity(
            modelID: "synthetic/qwen35-drafter",
            revision: String(repeating: "2", count: 40),
            configSHA256: testSHA256(drafterConfig),
            tokenizerSHA256: tokenizerDigest,
            tensorManifestSHA256: drafterTensorDigest)

        targetCandidate = .init(
            identity: targetIdentity,
            configJSON: targetConfig,
            tensors: Self.targetTensors)
        drafterCandidate = .init(
            identity: drafterIdentity,
            configJSON: drafterConfig,
            tensors: Self.drafterTensors)
        lock = Qwen35ExactMTPArtifactLock(
            sourceRevision: String(repeating: "3", count: 40),
            target: targetIdentity,
            drafter: drafterIdentity,
            architecture: .init(
                hiddenSize: 4096,
                intermediateSize: 12288,
                vocabularySize: 248_320,
                targetLayerCount: 32,
                fullAttentionInterval: 4,
                attentionHeadCount: 16,
                keyValueHeadCount: 4,
                headDimension: 256,
                mtpDepth: 1),
            targetQuantization: .init(bits: 4, groupSize: 64, mode: "affine"),
            drafterQuantization: .init(bits: 5, groupSize: 64, mode: "affine"),
            drafterTensors: Self.drafterTensors,
            runtimeBlockSize: 3,
            maximumAcceptedDraftTokens: 2)

        try writeConfig(role: .target, data: targetConfig)
        try writeConfig(role: .drafter, data: drafterConfig)
        try writeTokenizer(role: .target, vocab: Self.vocab)
        try writeTokenizer(role: .drafter, vocab: Self.vocab)
        try writeSafetensors(role: .target, tensors: Self.targetTensors)
        try writeSafetensors(role: .drafter, tensors: Self.drafterTensors)
    }

    static let vocab = ["token-a": 0, "token-b": 1]
    static let targetTensors: [Qwen35ExactMTPTensorDescriptor] = [
        .init(name: "model.embed_tokens.weight", shape: [248_320, 4096], dtype: "BF16")
    ]
    static let drafterTensors: [Qwen35ExactMTPTensorDescriptor] = [
        .init(name: "fc.weight", shape: [4096, 1280], dtype: "U32"),
        .init(name: "pre_fc_norm_hidden.weight", shape: [4096], dtype: "BF16"),
    ]

    func cleanup() {
        try? FileManager.default.removeItem(at: root)
    }

    func admit() throws -> Qwen35ExactMTPPreflightEvidence {
        try Qwen35ExactMTPFactory.admitResolvedDepth1Pair(
            lock: lock,
            targetDirectory: targetResolved.modelDirectory,
            targetTokenizerDirectory: targetResolved.tokenizerDirectory,
            drafterDirectory: drafterResolved.modelDirectory,
            drafterTokenizerDirectory: drafterResolved.tokenizerDirectory)
    }

    func directory(role: Qwen35ExactMTPArtifactRole) -> URL {
        switch role {
        case .target: targetResolved.modelDirectory
        case .drafter: drafterResolved.modelDirectory
        }
    }

    func writeConfig(role: Qwen35ExactMTPArtifactRole, data: Data) throws {
        try data.write(to: directory(role: role).appending(component: "config.json"))
    }

    func writeTokenizer(role: Qwen35ExactMTPArtifactRole, vocab: [String: Int]) throws {
        let data = try JSONSerialization.data(
            withJSONObject: ["model": ["vocab": vocab]],
            options: [.sortedKeys])
        try data.write(to: directory(role: role).appending(component: "tokenizer.json"))
    }

    func writeSafetensors(
        role: Qwen35ExactMTPArtifactRole,
        tensors: [Qwen35ExactMTPTensorDescriptor]
    ) throws {
        try safetensorHeaderData(tensors).write(
            to: directory(role: role).appending(component: "model.safetensors"))
    }

    func writeIndex(
        role: Qwen35ExactMTPArtifactRole,
        weightMap: [String: String]
    ) throws {
        let data = try JSONSerialization.data(
            withJSONObject: ["weight_map": weightMap],
            options: [.sortedKeys])
        try data.write(
            to: directory(role: role).appending(component: "model.safetensors.index.json"))
    }

    static func targetConfig() -> Data {
        Data("""
        {"model_type":"qwen3_5","quantization":{"bits":4,"group_size":64,"mode":"affine"},"text_config":\(textConfig())}
        """.utf8)
    }

    static func drafterConfig(
        blockSize: Int = 3,
        modelType: String = "qwen3_5_mtp",
        hiddenSize: Int = 4096,
        mtpLayers: Int = 1,
        bits: Int = 5
    ) -> Data {
        Data("""
        {"block_size":\(blockSize),"model_type":"\(modelType)","quantization":{"bits":\(bits),"group_size":64,"mode":"affine"},"text_config":\(textConfig(hiddenSize: hiddenSize, mtpLayers: mtpLayers))}
        """.utf8)
    }

    static func textConfig(hiddenSize: Int = 4096, mtpLayers: Int = 1) -> String {
        """
        {"model_type":"qwen3_5_text","hidden_size":\(hiddenSize),"intermediate_size":12288,"vocab_size":248320,"num_hidden_layers":32,"full_attention_interval":4,"num_attention_heads":16,"num_key_value_heads":4,"head_dim":256,"mtp_num_hidden_layers":\(mtpLayers),"mtp_use_dedicated_embeddings":false}
        """
    }
}

private extension Qwen35ExactMTPArtifactCandidate {
    func withConfig(_ data: Data) -> Self {
        .init(
            identity: .init(
                modelID: identity.modelID,
                revision: identity.revision,
                configSHA256: testSHA256(data),
                tokenizerSHA256: identity.tokenizerSHA256,
                tensorManifestSHA256: identity.tensorManifestSHA256),
            configJSON: data,
            tensors: tensors)
    }

    func withTensors(_ tensors: [Qwen35ExactMTPTensorDescriptor]) -> Self {
        .init(
            identity: .init(
                modelID: identity.modelID,
                revision: identity.revision,
                configSHA256: identity.configSHA256,
                tokenizerSHA256: identity.tokenizerSHA256,
                tensorManifestSHA256: Qwen35ExactMTPAdmission.tensorManifestSHA256(tensors)),
            configJSON: configJSON,
            tensors: tensors)
    }
}

private extension Qwen35ExactMTPArtifactLock {
    func withDrafterConfig(_ data: Data) -> Self {
        .init(
            sourceRevision: sourceRevision,
            target: target,
            drafter: .init(
                modelID: drafter.modelID,
                revision: drafter.revision,
                configSHA256: testSHA256(data),
                tokenizerSHA256: drafter.tokenizerSHA256,
                tensorManifestSHA256: drafter.tensorManifestSHA256),
            architecture: architecture,
            targetQuantization: targetQuantization,
            drafterQuantization: drafterQuantization,
            drafterTensors: drafterTensors,
            runtimeBlockSize: runtimeBlockSize,
            maximumAcceptedDraftTokens: maximumAcceptedDraftTokens)
    }

    func withDrafterTensorManifest(_ tensors: [Qwen35ExactMTPTensorDescriptor]) -> Self {
        .init(
            sourceRevision: sourceRevision,
            target: target,
            drafter: .init(
                modelID: drafter.modelID,
                revision: drafter.revision,
                configSHA256: drafter.configSHA256,
                tokenizerSHA256: drafter.tokenizerSHA256,
                tensorManifestSHA256: Qwen35ExactMTPAdmission.tensorManifestSHA256(tensors)),
            architecture: architecture,
            targetQuantization: targetQuantization,
            drafterQuantization: drafterQuantization,
            drafterTensors: drafterTensors,
            runtimeBlockSize: runtimeBlockSize,
            maximumAcceptedDraftTokens: maximumAcceptedDraftTokens)
    }
}

private extension Qwen35ExactMTPTensorDescriptor {
    func renamed(_ name: String) -> Self {
        .init(name: name, shape: shape, dtype: dtype)
    }

    func reshaped(_ shape: [Int]) -> Self {
        .init(name: name, shape: shape, dtype: dtype)
    }

    func withDType(_ dtype: String) -> Self {
        .init(name: name, shape: shape, dtype: dtype)
    }
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appending(component: "qwen35-exact-mtp-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func safetensorHeaderData(_ tensors: [Qwen35ExactMTPTensorDescriptor]) throws -> Data {
    let header = Dictionary(uniqueKeysWithValues: tensors.map {
        (
            $0.name,
            [
                "dtype": $0.dtype,
                "shape": $0.shape,
                "data_offsets": [0, 0],
            ] as [String: Any]
        )
    })
    let headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    var length = UInt64(headerData.count).littleEndian
    var data = Data(bytes: &length, count: MemoryLayout<UInt64>.size)
    data.append(headerData)
    return data
}

private func canonicalVocabularyData(_ vocab: [String: Int]) -> Data {
    var data = try! JSONSerialization.data(withJSONObject: vocab, options: [.sortedKeys])
    data.append(10)
    return data
}

private func testSHA256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}
