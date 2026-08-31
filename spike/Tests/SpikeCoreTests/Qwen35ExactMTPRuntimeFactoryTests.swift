import CryptoKit
import Darwin
import Foundation
import HarnessCore
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Metal
import Tokenizers
import XCTest

@testable import SpikeCore

final class Qwen35ExactMTPRuntimeFactoryTests: XCTestCase {
    func testQwen38HarnessRevisionUsesCleanLiveCheckoutAndNeverEnvironmentOverride() throws {
        let liveSHA = String(repeating: "a", count: 40)
        let stampSHA = String(repeating: "b", count: 40)
        XCTAssertEqual(
            try resolveQwen38LiveExactnessHarnessGitSHA(
                liveGitOutput: liveSHA,
                liveRepositoryPresent: true,
                shaFile: stampSHA),
            liveSHA)
        XCTAssertThrowsError(try resolveQwen38LiveExactnessHarnessGitSHA(
            liveGitOutput: "\(liveSHA)-dirty",
            liveRepositoryPresent: true,
            shaFile: stampSHA)) { error in
            XCTAssertEqual(error as? Qwen38LiveExactnessProducerError, .invalidHarnessGitSHA)
        }
        XCTAssertThrowsError(try resolveQwen38LiveExactnessHarnessGitSHA(
            liveGitOutput: nil,
            liveRepositoryPresent: true,
            shaFile: stampSHA)) { error in
            XCTAssertEqual(error as? Qwen38LiveExactnessProducerError, .harnessGitSHAUnavailable)
        }
    }

    func testQwen38HarnessRevisionAllowsOnlyCleanDeployStampWithoutGitCheckout() throws {
        let stampSHA = String(repeating: "c", count: 40)
        XCTAssertEqual(
            try resolveQwen38LiveExactnessHarnessGitSHA(
                liveGitOutput: nil,
                liveRepositoryPresent: false,
                shaFile: " \(stampSHA)\n"),
            stampSHA)
        for invalid in [
            String(repeating: "0", count: 40),
            "\(stampSHA)-dirty",
            String(repeating: "C", count: 40),
        ] {
            XCTAssertThrowsError(try resolveQwen38LiveExactnessHarnessGitSHA(
                liveGitOutput: nil,
                liveRepositoryPresent: false,
                shaFile: invalid))
        }
    }

    func testQwen38HarnessRevisionFindsDeployStampFromCompiledSourceOutsideTestCWD() throws {
        let fixture = try makeQwen38DeploySourceFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let unrelatedWorkingDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: unrelatedWorkingDirectory,
            withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: unrelatedWorkingDirectory) }

        XCTAssertEqual(
            try qwen38HarnessGitSHA(
                currentDirectory: unrelatedWorkingDirectory,
                compiledSourceFile: fixture.compiledSourceFile),
            fixture.sha)
    }

    func testQwen38HarnessRevisionRejectsUnmarkedDeployStampAndArbitraryCWDStamp() throws {
        let fixture = try makeQwen38DeploySourceFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let arbitraryWorkingDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: arbitraryWorkingDirectory,
            withIntermediateDirectories: true)
        try "\(String(repeating: "d", count: 40))\n".write(
            to: arbitraryWorkingDirectory.appendingPathComponent(".harness-sha"),
            atomically: true,
            encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: arbitraryWorkingDirectory) }

        for missingMarker in fixture.requiredMarkers {
            try FileManager.default.removeItem(at: missingMarker)
            XCTAssertThrowsError(try qwen38HarnessGitSHA(
                currentDirectory: arbitraryWorkingDirectory,
                compiledSourceFile: fixture.compiledSourceFile))
            try Data().write(to: missingMarker)
        }
        try FileManager.default.removeItem(
            at: fixture.root.appendingPathComponent(".harness-sha"))
        XCTAssertThrowsError(try qwen38HarnessGitSHA(
            currentDirectory: arbitraryWorkingDirectory,
            compiledSourceFile: fixture.compiledSourceFile))

        let externalStamp = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try "\(fixture.sha)\n".write(
            to: externalStamp,
            atomically: true,
            encoding: .utf8)
        try FileManager.default.createSymbolicLink(
            at: fixture.root.appendingPathComponent(".harness-sha"),
            withDestinationURL: externalStamp)
        defer { try? FileManager.default.removeItem(at: externalStamp) }
        XCTAssertThrowsError(try qwen38HarnessGitSHA(
            currentDirectory: arbitraryWorkingDirectory,
            compiledSourceFile: fixture.compiledSourceFile))
    }

    func testQwen38HarnessRevisionPrefersCompiledSourceLiveGitOverDeployStamp() throws {
        let fixture = try makeQwen38DeploySourceFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        try FileManager.default.createDirectory(
            at: fixture.root.appendingPathComponent(".git"),
            withIntermediateDirectories: true)
        try Data().write(to: fixture.root.appendingPathComponent("AGENTS.md"))
        let repositoryPackageMarker = fixture.root.appendingPathComponent("spike/Package.swift")
        try FileManager.default.createDirectory(
            at: repositoryPackageMarker.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try Data().write(to: repositoryPackageMarker)
        let unrelatedWorkingDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: unrelatedWorkingDirectory,
            withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: unrelatedWorkingDirectory) }
        let liveSHA = String(repeating: "e", count: 40)

        XCTAssertEqual(
            try qwen38HarnessGitSHA(
                currentDirectory: unrelatedWorkingDirectory,
                compiledSourceFile: fixture.compiledSourceFile,
                liveGitSHA: { _ in liveSHA }),
            liveSHA)
        XCTAssertThrowsError(try qwen38HarnessGitSHA(
            currentDirectory: unrelatedWorkingDirectory,
            compiledSourceFile: fixture.compiledSourceFile,
            liveGitSHA: { _ in nil })) { error in
            XCTAssertEqual(error as? Qwen38LiveExactnessProducerError, .harnessGitSHAUnavailable)
        }
        XCTAssertThrowsError(try qwen38HarnessGitSHA(
            currentDirectory: unrelatedWorkingDirectory,
            compiledSourceFile: fixture.compiledSourceFile,
            liveGitSHA: { _ in "\(liveSHA)-dirty" })) { error in
            XCTAssertEqual(error as? Qwen38LiveExactnessProducerError, .invalidHarnessGitSHA)
        }
    }

    func testQwen38ValidatedEvidenceWriterPublishesCompleteRecordAndPreservesExistingFile() throws {
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("jsonl")
        defer { try? FileManager.default.removeItem(at: output) }
        let data = try validQwen38LiveExactnessRecordData()

        try writeValidatedQwen38LiveExactnessEvidenceFile(data, to: output)
        XCTAssertEqual(try Data(contentsOf: output), data)
        XCTAssertNoThrow(try Qwen38MTPLiveExactnessGate.validateJSONL(
            Data(contentsOf: output)))

        XCTAssertThrowsError(try writeValidatedQwen38LiveExactnessEvidenceFile(data, to: output))
        XCTAssertEqual(try Data(contentsOf: output), data)
    }

    func testQwen38ValidatedEvidenceWriterLeavesNoDestinationForInvalidEvidence() throws {
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("jsonl")
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))

        XCTAssertThrowsError(try writeValidatedQwen38LiveExactnessEvidenceFile(
            Data("{\"invalid\":true}\n".utf8),
            to: output))
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: output.path),
            "failed validation must not create even a partial destination")
    }

    func testQwen38LiveExactnessLaunchBindingRequiresExplicitGDNOnEnvironment() throws {
        let process = try qwen38LiveExactnessProcessFacts(
            environment: ["MLX_QWEN_FOUR_GDN": "1"],
            executableSHA256: String(repeating: "a", count: 64),
            harnessGitSHA: String(repeating: "b", count: 40))
        let binding = try qwen38LiveExactnessLaunchBinding(process: process)

        XCTAssertEqual(process.executableIdentitySource, .procPIDPath)
        XCTAssertEqual(binding.mode, .gdnOn)
        XCTAssertEqual(binding.observedEnv, .enabled)
        XCTAssertEqual(binding.sourceDigest, Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID)
        XCTAssertEqual(
            binding.launchDigest,
            Qwen38MTPPerformanceScorecardGate.launchDigest(
                mode: .gdnOn,
                sourceDigest: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID,
                observedEnv: .enabled,
                processIsolationEvidenceID: binding.processIsolationEvidenceID))

        for value in [nil, "", "0", "true", "1,0"] {
            var environment: [String: String] = [:]
            if let value { environment["MLX_QWEN_FOUR_GDN"] = value }
            let process = try qwen38LiveExactnessProcessFacts(
                environment: environment,
                executableSHA256: String(repeating: "a", count: 64),
                harnessGitSHA: String(repeating: "b", count: 40))
            XCTAssertThrowsError(try qwen38LiveExactnessLaunchBinding(process: process)) { error in
                XCTAssertEqual(error as? Qwen38LiveExactnessProducerError, .gdnFusionNotEnabled)
            }
        }
    }

    func testQwen38ExecutableIdentityUsesKernelReportedProcessPath() throws {
        let fixture = URL(fileURLWithPath: "/tmp/qwen38-executable-fixture-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: fixture) }
        try Data("fixture executable bytes".utf8).write(to: fixture)

        var observedPID: pid_t?
        let digest = try qwen38ExecutableSHA256(
            processID: 123,
            processPath: { processID in
                observedPID = processID
                return fixture.path
            })

        XCTAssertEqual(observedPID, 123)
        XCTAssertEqual(digest, sha256Hex(try Data(contentsOf: fixture)))
    }

    func testQwen38LiveExactnessMemoryBudgetRequiresPositiveEnvironmentAndReadback() throws {
        let environment = [
            "FAST_MLX_QWEN38_MLX_MEMORY_LIMIT_BYTES": "4096",
            "FAST_MLX_QWEN38_MLX_CACHE_LIMIT_BYTES": "1024",
        ]
        XCTAssertEqual(
            try qwen38LiveExactnessMemoryBudget(environment: environment),
            .init(memoryLimitBytes: 4096, cacheLimitBytes: 1024))

        for invalid in [
            [:],
            ["FAST_MLX_QWEN38_MLX_MEMORY_LIMIT_BYTES": "4096"],
            [
                "FAST_MLX_QWEN38_MLX_MEMORY_LIMIT_BYTES": "0",
                "FAST_MLX_QWEN38_MLX_CACHE_LIMIT_BYTES": "1024",
            ],
            [
                "FAST_MLX_QWEN38_MLX_MEMORY_LIMIT_BYTES": "4096",
                "FAST_MLX_QWEN38_MLX_CACHE_LIMIT_BYTES": "0",
            ],
            [
                "FAST_MLX_QWEN38_MLX_MEMORY_LIMIT_BYTES": "4096",
                "FAST_MLX_QWEN38_MLX_CACHE_LIMIT_BYTES": "4097",
            ],
        ] {
            XCTAssertThrowsError(try qwen38LiveExactnessMemoryBudget(environment: invalid)) {
                error in
                XCTAssertEqual(error as? Qwen38LiveExactnessProducerError, .invalidMemoryBudget)
            }
        }
    }

    func testQwen38LiveExactnessHostMemoryObservationRequiresDedicatedMeasuredPolicy()
        throws
    {
        let gib = UInt64(1024 * 1024 * 1024)
        let environment = qwen38DedicatedHostMemoryEnvironment()
        let observation = try qwen38LiveExactnessHostMemoryObservation(
            environment: environment,
            physicalRAMBytes: 256 * gib,
            wiredLimitMB: 245_760,
            wiredLimitProvenance: .measured,
            metalRecommendedMaxWorkingSetSizeBytes: 240 * gib,
            metalCurrentAllocatedSizeBytes: 2 * gib)

        XCTAssertEqual(observation.hostUse, "dedicated-serving")
        XCTAssertEqual(observation.hostUseSource, "operator-assertion")
        XCTAssertEqual(observation.hostUsePolicyVersion, Qwen38MTPLiveExactnessGate.requiredHostUsePolicyVersion)
        XCTAssertEqual(observation.memoryLimitBytes, 220 * gib)
        XCTAssertEqual(observation.cacheLimitBytes, 48 * gib)
        XCTAssertEqual(observation.osServiceReserveBytes, 8 * gib)

        var shared = environment
        shared["FAST_MLX_QWEN38_HOST_USE"] = "shared"
        XCTAssertThrowsError(try qwen38LiveExactnessHostMemoryObservation(
            environment: shared,
            physicalRAMBytes: 256 * gib,
            wiredLimitMB: 245_760,
            wiredLimitProvenance: .measured,
            metalRecommendedMaxWorkingSetSizeBytes: 240 * gib,
            metalCurrentAllocatedSizeBytes: 2 * gib)) { error in
            XCTAssertEqual(error as? Qwen38LiveExactnessProducerError, .invalidHostMemoryObservation)
        }

        var unknown = environment
        unknown["FAST_MLX_QWEN38_HOST_USE"] = "auto"
        XCTAssertThrowsError(try qwen38LiveExactnessHostMemoryObservation(
            environment: unknown,
            physicalRAMBytes: 256 * gib,
            wiredLimitMB: 245_760,
            wiredLimitProvenance: .measured,
            metalRecommendedMaxWorkingSetSizeBytes: 240 * gib,
            metalCurrentAllocatedSizeBytes: 2 * gib)) { error in
            XCTAssertEqual(error as? Qwen38LiveExactnessProducerError, .invalidHostMemoryObservation)
        }

        XCTAssertThrowsError(try qwen38LiveExactnessHostMemoryObservation(
            environment: environment,
            physicalRAMBytes: 256 * gib,
            wiredLimitMB: 245_760,
            wiredLimitProvenance: .synthesized,
            metalRecommendedMaxWorkingSetSizeBytes: 240 * gib,
            metalCurrentAllocatedSizeBytes: 2 * gib)) { error in
            XCTAssertEqual(error as? Qwen38LiveExactnessProducerError, .invalidHostMemoryObservation)
        }

        XCTAssertNoThrow(try qwen38LiveExactnessHostMemoryObservation(
            environment: environment,
            physicalRAMBytes: 256 * gib,
            wiredLimitMB: 0,
            wiredLimitProvenance: .measured,
            metalRecommendedMaxWorkingSetSizeBytes: 240 * gib,
            metalCurrentAllocatedSizeBytes: 2 * gib))

        var measuredZeroOverBudget = environment
        measuredZeroOverBudget["FAST_MLX_QWEN38_MLX_MEMORY_LIMIT_BYTES"] = "\(240 * gib)"
        XCTAssertThrowsError(try qwen38LiveExactnessHostMemoryObservation(
            environment: measuredZeroOverBudget,
            physicalRAMBytes: 256 * gib,
            wiredLimitMB: 0,
            wiredLimitProvenance: .measured,
            metalRecommendedMaxWorkingSetSizeBytes: 240 * gib,
            metalCurrentAllocatedSizeBytes: 2 * gib)) { error in
            XCTAssertEqual(error as? Qwen38LiveExactnessProducerError, .invalidHostMemoryObservation)
        }

        XCTAssertThrowsError(try qwen38LiveExactnessHostMemoryObservation(
            environment: environment,
            physicalRAMBytes: 256 * gib,
            wiredLimitMB: 245_760,
            wiredLimitProvenance: .measured,
            metalRecommendedMaxWorkingSetSizeBytes: 0,
            metalCurrentAllocatedSizeBytes: 2 * gib)) { error in
            XCTAssertEqual(error as? Qwen38LiveExactnessProducerError, .invalidHostMemoryObservation)
        }

        var overflow = environment
        overflow["FAST_MLX_QWEN38_OS_SERVICE_RESERVE_BYTES"] = "\(240 * gib)"
        XCTAssertThrowsError(try qwen38LiveExactnessHostMemoryObservation(
            environment: overflow,
            physicalRAMBytes: 256 * gib,
            wiredLimitMB: 245_760,
            wiredLimitProvenance: .measured,
            metalRecommendedMaxWorkingSetSizeBytes: 240 * gib,
            metalCurrentAllocatedSizeBytes: 2 * gib)) { error in
            XCTAssertEqual(error as? Qwen38LiveExactnessProducerError, .invalidHostMemoryObservation)
        }
    }

    func testKnownVendoredLockExactlyMatchesHarnessPreflightLock() throws {
        XCTAssertNoThrow(try Qwen35ExactMTPRuntimeFactory.validateKnownLockParity())
    }

    private func validQwen38LiveExactnessRecordData() throws -> Data {
        let cache = (0 ..< Qwen38MTPLiveExactnessGate.requiredCacheLayerCount).map { layerIndex in
            let isDense = (layerIndex + 1).isMultiple(of: 4)
            return Qwen38MTPLiveExactnessCacheFingerprint(
                layerIndex: layerIndex,
                cacheType: isDense ? "dense-attention" : "recurrent-mamba",
                offset: isDense ? 12 : 0,
                metaStateSHA256: String(repeating: "a", count: 64),
                stateFingerprints: (0 ..< 2).map { stateIndex in
                    let shape = isDense
                        ? [1, 4, 12, 256]
                        : (stateIndex == 0 ? [1, 3, 10_240] : [1, 48, 128, 128])
                    let dtype = isDense || stateIndex == 0 ? "bfloat16" : "float32"
                    let byteCount = shape.reduce(1, *) * (dtype == "float32" ? 4 : 2)
                    return Qwen38MTPLiveExactnessArrayFingerprint(
                        stateIndex: stateIndex,
                        shape: shape,
                        dtype: dtype,
                        byteCount: byteCount,
                        sha256: String(repeating: "b", count: 64))
                })
        }
        let cases = Qwen38MTPLiveExactnessGate.requiredCases.enumerated().map {
            index, specification in
            let bytes = Data("case-\(index)".utf8)
            let tokens = [100 + index, 200 + index]
            return Qwen38MTPLiveExactnessCaseEvidence(
                id: specification.id,
                promptSHA256: specification.promptSHA256,
                maxTokens: specification.maxTokens,
                scalarTokenIDs: tokens,
                mtpTokenIDs: tokens,
                scalarDecodedUTF8Base64: bytes.base64EncodedString(),
                mtpDecodedUTF8Base64: bytes.base64EncodedString(),
                decodedUTF8SHA256: sha256Hex(bytes),
                proposedDraftTokens: 3,
                acceptedDraftTokens: 2,
                passthroughReason: nil,
                scalarCacheFingerprints: cache,
                mtpCacheFingerprints: cache)
        }
        let evidence = Qwen38MTPLiveExactnessEvidence(
            schemaVersion: Qwen38MTPLiveExactnessGate.schemaVersion,
            artifact: Qwen38MTPPerformanceScorecardGate.requiredArtifact,
            artifactID: Qwen38MTPLiveExactnessGate.requiredArtifactID,
            source: Qwen38MTPLiveExactnessGate.requiredSourceIdentity,
            gdnMode: .gdnOn,
            launchBinding: syntheticQwen38LiveExactnessLaunchBinding(),
            processIsolation: syntheticQwen38LiveExactnessProcessFacts(),
            mlxMemoryBudget: Qwen38MTPLiveExactnessMLXMemoryBudget(
                memoryLimitBytes: 220 * 1024 * 1024 * 1024,
                cacheLimitBytes: 48 * 1024 * 1024 * 1024),
            hostMemoryObservation: syntheticQwen38LiveExactnessHostMemoryObservation(),
            cases: cases)
        let provenance = Provenance(
            date: "2026-08-25T00:00:00Z",
            hardwareChip: "Apple M3 Ultra",
            hardwareRAMBytes: Qwen38MTPPerformanceScorecardGate.requiredRAMBytes,
            hardwareOS: "macOS 26.0",
            harnessGitSHA: String(repeating: "e", count: 40),
            mlxSwiftVersion: "0.31.6",
            referenceMLXVersion: nil,
            referenceMLXLMVersion: nil,
            modelPath: Qwen38MTPLiveExactnessGate.modelPathSentinel,
            modelConfigHash:
                Qwen38MTPPerformanceScorecardGate.requiredArtifact.targetConfigSHA256,
            modelCheckpointManifestHash:
                Qwen38MTPPerformanceScorecardGate.requiredArtifact.targetTensorManifestSHA256,
            modelQuant: ModelQuantInfo(bits: 8, groupSize: 32),
            corpusId: nil,
            corpusContentHash: nil,
            nonce: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID)
        let record = ResultRecord(
            subcommand: Qwen38MTPLiveExactnessGate.subcommand,
            provenance: provenance,
            payload: evidence)
        return Data((try record.jsonLine() + "\n").utf8)
    }

    func testKnownQwen38VendoredLockExactlyMatchesHarnessPreflightLock() throws {
        XCTAssertNoThrow(try Qwen35ExactMTPRuntimeFactory.validateKnownLockParity(
            selection: .qwen38_27BMXFP8Depth1))
    }

    func testSelectedRuntimeLockRejectsCrossRowEvidenceBeforeConstruction() throws {
        XCTAssertThrowsError(try Qwen35ExactMTPRuntimeFactory.validateSelectedVendoredLock(
            Qwen35ExactMTPKnownArtifactLocks.qwen35_9BDepth1,
            selection: .qwen38_27BMXFP8Depth1
        )) { error in
            XCTAssertEqual(
                error as? Qwen35ExactMTPRuntimeAdmissionError,
                .vendoredLockDrift)
        }
    }

    func testQwen38SourceLockedPreloadRejectsUnsupportedSelectionBeforeFilesystem() throws {
        XCTAssertThrowsError(try Qwen35ExactMTPRuntimeFactory.preloadSourceLockedDepth1Pair(
            selection: .qwen35_9BDepth1,
            targetDirectory: URL(fileURLWithPath: "/tmp/missing-target", isDirectory: true),
            drafterDirectory: URL(fileURLWithPath: "/tmp/missing-drafter", isDirectory: true)
        )) { error in
            XCTAssertEqual(
                error as? Qwen35ExactMTPRuntimeAdmissionError,
                .unsupportedPreloadSelection(.qwen35_9BDepth1))
        }
    }

    func testQwen38SourceLockedPreloadRejectsSyntheticArtifactDriftWithoutLoader() throws {
        let fixture = try makeQwen38SyntheticPreloadDriftFixture()
        defer { fixture.cleanup() }

        XCTAssertThrowsError(try Qwen35ExactMTPRuntimeFactory.preloadSourceLockedDepth1Pair(
            selection: .qwen38_27BMXFP8Depth1,
            targetDirectory: fixture.targetDirectory,
            drafterDirectory: fixture.drafterDirectory
        )) { error in
            XCTAssertEqual(
                error as? Qwen35ExactMTPAdmissionError,
                .identityMismatch(role: .target, field: "configSHA256"))
        }
    }

    func testQwen38SourceLockedPreloadAuthenticatesExactFixtureWhenTokenizerSnapshotConfigured()
        throws
    {
        let fixture = try makeQwen38ExactPreloadFixture()
        defer { fixture.cleanup() }

        let binding = try Qwen35ExactMTPRuntimeFactory.preloadSourceLockedDepth1Pair(
            selection: .qwen38_27BMXFP8Depth1,
            targetDirectory: fixture.targetDirectory,
            drafterDirectory: fixture.drafterDirectory)
        let lock = QwenMTPKnownArtifactLocks.qwen38_27BMXFP8Depth1

        XCTAssertEqual(binding.targetModelID, lock.targetIdentity.modelID)
        XCTAssertEqual(binding.drafterModelID, lock.drafterIdentity.modelID)
        XCTAssertEqual(binding.targetRevision, lock.targetIdentity.revision)
        XCTAssertEqual(binding.drafterRevision, lock.drafterIdentity.revision)
        XCTAssertEqual(binding.sourceRevision, lock.sourceRevision)
        XCTAssertEqual(binding.runtimeBlockSize, 3)
        XCTAssertEqual(binding.maximumAcceptedDraftTokens, 2)
    }

    func testQwen38SourceLockedPreloadRejectsTargetConfigDriftWhenTokenizerSnapshotConfigured()
        throws
    {
        let fixture = try makeQwen38ExactPreloadFixture()
        defer { fixture.cleanup() }
        let changedConfig = try String(
            decoding: qwen38PreloadFixtureData(named: "qwen38-27b-target-config"),
            as: UTF8.self)
            .replacingOccurrences(of: "\"hidden_size\": 5120", with: "\"hidden_size\": 5121")
        try Data(changedConfig.utf8).write(
            to: fixture.targetDirectory.appending(component: "config.json"))

        XCTAssertThrowsError(try Qwen35ExactMTPRuntimeFactory.preloadSourceLockedDepth1Pair(
            selection: .qwen38_27BMXFP8Depth1,
            targetDirectory: fixture.targetDirectory,
            drafterDirectory: fixture.drafterDirectory
        )) { error in
            XCTAssertEqual(
                error as? Qwen35ExactMTPAdmissionError,
                .identityMismatch(role: .target, field: "configSHA256"))
        }
    }

    func testExactSnapshotDownloaderAllowsOnlySelectedQwen38Rows() async throws {
        let target = URL(fileURLWithPath: "/tmp/qwen38-target", isDirectory: true)
        let drafter = URL(fileURLWithPath: "/tmp/qwen38-drafter", isDirectory: true)
        let downloader = ExactSnapshotDownloader(
            target: target,
            drafter: drafter,
            selection: .qwen38_27BMXFP8Depth1)

        let targetURL = try await downloader.download(
            id: "mlx-community/Qwen3.8-27B-mxfp8",
            revision: "d48d163bcdf24acaf656474854ab88ea17d65bd1",
            matching: exactSnapshotPatterns,
            useLatest: false,
            progressHandler: { _ in })
        let drafterURL = try await downloader.download(
            id: "mlx-community/Qwen3.8-27B-MTP-mxfp8",
            revision: "a50634460045613f166b09b13519466e801c6568",
            matching: exactSnapshotPatterns,
            useLatest: false,
            progressHandler: { _ in })

        XCTAssertEqual(targetURL, target)
        XCTAssertEqual(drafterURL, drafter)
    }

    func testExactSnapshotDownloaderRejectsCrossRowAndLatestRequests() async throws {
        let downloader = ExactSnapshotDownloader(
            target: URL(fileURLWithPath: "/tmp/qwen38-target", isDirectory: true),
            drafter: URL(fileURLWithPath: "/tmp/qwen38-drafter", isDirectory: true),
            selection: .qwen38_27BMXFP8Depth1)

        await XCTAssertThrowsErrorAsync(try await downloader.download(
            id: "mlx-community/Qwen3.5-9B-MLX-4bit",
            revision: "938d8919941c6e7efd3c7150eff7fe9d12afa631",
            matching: exactSnapshotPatterns,
            useLatest: false,
            progressHandler: { _ in })) { error in
            XCTAssertEqual(error as? ExactSnapshotDownloaderError, .unexpectedDownloadRequest)
        }
        await XCTAssertThrowsErrorAsync(try await downloader.download(
            id: "mlx-community/Qwen3.8-27B-mxfp8",
            revision: "d48d163bcdf24acaf656474854ab88ea17d65bd1",
            matching: exactSnapshotPatterns,
            useLatest: true,
            progressHandler: { _ in })) { error in
            XCTAssertEqual(error as? ExactSnapshotDownloaderError, .unexpectedDownloadRequest)
        }
    }

    func testExactArtifactPreflightAgainstLocalSnapshotsWhenConfigured() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let targetPath = environment["FAST_MLX_QWEN35_TARGET_SNAPSHOT"],
            let drafterPath = environment["FAST_MLX_QWEN35_MTP_SNAPSHOT"]
        else {
            throw XCTSkip("exact Qwen3.5 target/drafter snapshot paths are not configured")
        }

        let downloader = ExactSnapshotDownloader(
            target: URL(fileURLWithPath: targetPath, isDirectory: true),
            drafter: URL(fileURLWithPath: drafterPath, isDirectory: true))

        do {
            _ = try await Qwen35ExactMTPRuntimeFactory.loadDepth1Pair(
                from: downloader,
                using: PreflightOnlyTokenizerLoader())
            XCTFail("the preflight-only tokenizer must stop the load")
        } catch ExactSnapshotDownloaderError.tokenizerLoadReached {
            // Fixed resolution, vendored admission, and the mandatory HarnessCore bridge all ran
            // before the wrapper advanced into the model/tokenizer loader.
        }
    }

    func testExactDepthOneArtifactTwoDraftGreedyParityWhenConfigured() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["FAST_MLX_QWEN35_RUN_LIVE_EXACTNESS"] == "1" else {
            throw XCTSkip("live exactness requires FAST_MLX_QWEN35_RUN_LIVE_EXACTNESS=1")
        }
        guard let targetPath = environment["FAST_MLX_QWEN35_TARGET_SNAPSHOT"],
            let drafterPath = environment["FAST_MLX_QWEN35_MTP_SNAPSHOT"]
        else {
            throw XCTSkip("exact Qwen3.5 target/drafter snapshot paths are not configured")
        }

        let pair = try await Qwen35ExactMTPRuntimeFactory.loadDepth1Pair(
            from: ExactSnapshotDownloader(
                target: URL(fileURLWithPath: targetPath, isDirectory: true),
                drafter: URL(fileURLWithPath: drafterPath, isDirectory: true)),
            using: #huggingFaceTokenizerLoader())
        XCTAssertEqual(pair.binding.runtimeBlockSize, 3)
        XCTAssertEqual(pair.binding.maximumAcceptedDraftTokens, 2)
        let promptTokens = pair.target.tokenizer.encode(
            text: "Continue the exact sequence with one concise answer: 2, 3, 5, 7, 11,")
        let input = LMInput(tokens: MLXArray(promptTokens))
        let parameters = GenerateParameters(maxTokens: 16, temperature: 0)

        let scalarCache = pair.target.model.newCache(parameters: parameters)
        let scalar = try await collectTokens(
            generateTokens(
                input: input,
                cache: scalarCache,
                parameters: parameters,
                context: pair.target))

        let mtpCache = pair.target.model.newCache(parameters: parameters)
        let mtp = try await collectTokens(
            generateTokens(
                input: input,
                cache: mtpCache,
                parameters: parameters,
                context: pair.target,
                mtpDrafter: pair.drafter.model,
                blockSize: pair.binding.runtimeBlockSize))

        XCTAssertFalse(scalar.tokens.isEmpty)
        XCTAssertEqual(mtp.tokens, scalar.tokens)
        XCTAssertEqual(
            Data(pair.target.tokenizer.decode(
                tokenIds: mtp.tokens, skipSpecialTokens: false).utf8),
            Data(pair.target.tokenizer.decode(
                tokenIds: scalar.tokens, skipSpecialTokens: false).utf8))
        XCTAssertEqual(mtp.info?.proposedDraftTokens, 10)
        XCTAssertEqual(mtp.info?.acceptedDraftTokens, 9)
        XCTAssertNil(mtp.info?.passthroughReason)
        try assertCacheEquivalent(scalarCache, mtpCache)

        print(
            "QWEN35_MTP_LIVE_EXACTNESS tokens=\(mtp.tokens.count) "
                + "proposed=\(mtp.info?.proposedDraftTokens ?? -1) "
                + "accepted=\(mtp.info?.acceptedDraftTokens ?? -1) "
                + "scalar_tps=\(scalar.info?.tokensPerSecond ?? 0) "
                + "mtp_tps=\(mtp.info?.tokensPerSecond ?? 0)")
    }

    func testQwen38ExactDepthOneArtifactGreedyParityWhenConfigured() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["FAST_MLX_QWEN38_RUN_LIVE_EXACTNESS"] == "1" else {
            throw XCTSkip("live exactness requires FAST_MLX_QWEN38_RUN_LIVE_EXACTNESS=1")
        }
        guard let targetPath = environment["FAST_MLX_QWEN38_TARGET_SNAPSHOT"],
            let drafterPath = environment["FAST_MLX_QWEN38_MTP_SNAPSHOT"]
        else {
            throw XCTSkip("exact Qwen3.8 target/drafter snapshot paths are not configured")
        }
        let harnessGitSHA = try qwen38HarnessGitSHA()
        let hostMemoryObservation = try qwen38LiveExactnessHostMemoryObservation(
            environment: environment)
        let memoryBudget = hostMemoryObservation.mlxMemoryBudget
        try applyQwen38LiveExactnessMemoryBudget(memoryBudget)

        let evidenceOutputPath = environment["FAST_MLX_QWEN38_LIVE_EXACTNESS_JSONL"]
        if let evidenceOutputPath,
            FileManager.default.fileExists(atPath: evidenceOutputPath)
        {
            throw CocoaError(.fileWriteFileExists)
        }
        let processFacts = try qwen38LiveExactnessProcessFacts(
            environment: environment,
            harnessGitSHA: harnessGitSHA)
        let launchBinding = try qwen38LiveExactnessLaunchBinding(process: processFacts)

        let pair = try await Qwen35ExactMTPRuntimeFactory.loadDepth1Pair(
            selection: .qwen38_27BMXFP8Depth1,
            from: ExactSnapshotDownloader(
                target: URL(fileURLWithPath: targetPath, isDirectory: true),
                drafter: URL(fileURLWithPath: drafterPath, isDirectory: true),
                selection: .qwen38_27BMXFP8Depth1),
            using: #huggingFaceTokenizerLoader())
        guard pair.binding.runtimeBlockSize == 3,
            pair.binding.maximumAcceptedDraftTokens == 2
        else {
            throw Qwen38LiveExactnessProducerError.runtimeBindingDrift
        }

        var evidenceCases: [Qwen38MTPLiveExactnessCaseEvidence] = []
        for testCase in qwen38GreedyParityCases {
            let promptTokens = pair.target.tokenizer.encode(text: testCase.prompt)
            let input = LMInput(tokens: MLXArray(promptTokens))
            let parameters = GenerateParameters(maxTokens: testCase.maxTokens, temperature: 0)

            let scalarCache = pair.target.model.newCache(parameters: parameters)
            let scalar = try await collectTokens(
                generateTokens(
                    input: input,
                    cache: scalarCache,
                    parameters: parameters,
                    context: pair.target))

            let mtpCache = pair.target.model.newCache(parameters: parameters)
            let mtp = try await collectTokens(
                generateTokens(
                    input: input,
                    cache: mtpCache,
                    parameters: parameters,
                    context: pair.target,
                    mtpDrafter: pair.drafter.model,
                    blockSize: pair.binding.runtimeBlockSize))

            XCTAssertFalse(mtp.tokens.isEmpty, "MTP generated no tokens for \(testCase.id)")
            XCTAssertEqual(mtp.tokens, scalar.tokens, "token mismatch for \(testCase.id)")
            XCTAssertEqual(
                Data(pair.target.tokenizer.decode(
                    tokenIds: mtp.tokens, skipSpecialTokens: false).utf8),
                Data(pair.target.tokenizer.decode(
                    tokenIds: scalar.tokens, skipSpecialTokens: false).utf8),
                "decoded UTF-8 bytes mismatch for \(testCase.id)")
            XCTAssertGreaterThan(
                mtp.info?.proposedDraftTokens ?? 0,
                0,
                "expected draft proposals for \(testCase.id)")
            XCTAssertGreaterThan(
                mtp.info?.acceptedDraftTokens ?? 0,
                0,
                "expected accepted drafts for \(testCase.id)")
            XCTAssertNil(mtp.info?.passthroughReason, "unexpected passthrough for \(testCase.id)")
            try assertCacheEquivalent(scalarCache, mtpCache)
            evidenceCases.append(makeQwen38LiveExactnessCaseEvidence(
                id: testCase.id,
                prompt: testCase.prompt,
                maxTokens: testCase.maxTokens,
                scalar: scalar,
                mtp: mtp,
                scalarCache: scalarCache,
                mtpCache: mtpCache,
                tokenizer: pair.target.tokenizer))

            print(
                "QWEN38_MTP_LIVE_EXACTNESS identity=qwen38_27BMXFP8Depth1 "
                    + "case=\(testCase.id) "
                    + "tokens=\(mtp.tokens.count) "
                    + "proposed=\(mtp.info?.proposedDraftTokens ?? -1) "
                    + "accepted=\(mtp.info?.acceptedDraftTokens ?? -1) "
                    + "tps=scalar:\(scalar.info?.tokensPerSecond ?? 0)"
                    + "/mtp:\(mtp.info?.tokensPerSecond ?? 0)")
        }

        if let evidenceOutputPath {
            let evidence = Qwen38MTPLiveExactnessEvidence(
                schemaVersion: Qwen38MTPLiveExactnessGate.schemaVersion,
                artifact: Qwen38MTPPerformanceScorecardGate.requiredArtifact,
                artifactID: Qwen38MTPLiveExactnessGate.requiredArtifactID,
                source: Qwen38MTPLiveExactnessGate.requiredSourceIdentity,
                gdnMode: .gdnOn,
                launchBinding: launchBinding,
                processIsolation: processFacts,
                mlxMemoryBudget: memoryBudget,
                hostMemoryObservation: hostMemoryObservation,
                cases: evidenceCases)
            let record = ResultRecord(
                subcommand: Qwen38MTPLiveExactnessGate.subcommand,
                provenance: try qwen38LiveExactnessProvenance(harnessGitSHA: harnessGitSHA),
                payload: evidence)
            let data = Data((try record.jsonLine() + "\n").utf8)
            try writeValidatedQwen38LiveExactnessEvidenceFile(
                data,
                to: URL(fileURLWithPath: evidenceOutputPath))
        }
    }
}

private struct CollectedGeneration {
    let tokens: [Int]
    let info: GenerateCompletionInfo?
}

private struct GreedyParityCase {
    let id: String
    let prompt: String
    let maxTokens: Int
}

private let qwen38GreedyParityCases: [GreedyParityCase] = [
    .init(
        id: "numbers",
        prompt: "Continue the exact sequence with one concise answer: 2, 3, 5, 7, 11,",
        maxTokens: 12),
    .init(
        id: "sentence",
        prompt: "Complete this sentence in a few words: The fastest reliable test is",
        maxTokens: 12),
]

private let exactSnapshotPatterns = ["*.safetensors", "*.json", "*.jinja"]

private func collectTokens(
    _ stream: AsyncStream<TokenGeneration>
) async throws -> CollectedGeneration {
    var tokens: [Int] = []
    var info: GenerateCompletionInfo?
    for await generation in stream {
        if let token = generation.token {
            tokens.append(token)
        }
        if let completion = generation.info {
            info = completion
        }
    }
    return CollectedGeneration(tokens: tokens, info: info)
}

private func assertCacheEquivalent(_ scalar: [KVCache], _ mtp: [KVCache]) throws {
    XCTAssertEqual(mtp.count, scalar.count)
    for (index, pair) in zip(scalar, mtp).enumerated() {
        let scalarCache = pair.0
        let mtpCache = pair.1
        XCTAssertEqual(
            String(reflecting: type(of: mtpCache)),
            String(reflecting: type(of: scalarCache)),
            "cache type mismatch at layer \(index)")
        XCTAssertEqual(mtpCache.offset, scalarCache.offset, "cache offset mismatch at layer \(index)")
        XCTAssertEqual(
            mtpCache.metaState, scalarCache.metaState,
            "cache metadata mismatch at layer \(index)")
        let scalarState = scalarCache.state
        let mtpState = mtpCache.state
        XCTAssertEqual(mtpState.count, scalarState.count, "cache state count mismatch at layer \(index)")
        for (stateIndex, arrays) in zip(scalarState, mtpState).enumerated() {
            eval(arrays.0, arrays.1)
            XCTAssertEqual(
                arrays.1.shape, arrays.0.shape,
                "cache shape mismatch at layer \(index) state \(stateIndex)")
            XCTAssertEqual(
                arrays.1.dtype, arrays.0.dtype,
                "cache dtype mismatch at layer \(index) state \(stateIndex)")
            XCTAssertEqual(
                arrays.1.asData(access: .copy).data,
                arrays.0.asData(access: .copy).data,
                "cache bytes mismatch at layer \(index) state \(stateIndex)")
        }
    }
}

private func makeQwen38LiveExactnessCaseEvidence(
    id: String,
    prompt: String,
    maxTokens: Int,
    scalar: CollectedGeneration,
    mtp: CollectedGeneration,
    scalarCache: [KVCache],
    mtpCache: [KVCache],
    tokenizer: any MLXLMCommon.Tokenizer
) -> Qwen38MTPLiveExactnessCaseEvidence {
    let scalarBytes = Data(tokenizer.decode(
        tokenIds: scalar.tokens,
        skipSpecialTokens: false).utf8)
    let mtpBytes = Data(tokenizer.decode(
        tokenIds: mtp.tokens,
        skipSpecialTokens: false).utf8)
    return Qwen38MTPLiveExactnessCaseEvidence(
        id: id,
        promptSHA256: sha256Hex(Data(prompt.utf8)),
        maxTokens: maxTokens,
        scalarTokenIDs: scalar.tokens,
        mtpTokenIDs: mtp.tokens,
        scalarDecodedUTF8Base64: scalarBytes.base64EncodedString(),
        mtpDecodedUTF8Base64: mtpBytes.base64EncodedString(),
        decodedUTF8SHA256: sha256Hex(scalarBytes),
        proposedDraftTokens: mtp.info?.proposedDraftTokens ?? 0,
        acceptedDraftTokens: mtp.info?.acceptedDraftTokens ?? 0,
        passthroughReason: mtp.info?.passthroughReason,
        scalarCacheFingerprints: qwen38LiveExactnessCacheFingerprints(scalarCache),
        mtpCacheFingerprints: qwen38LiveExactnessCacheFingerprints(mtpCache))
}

private func qwen38LiveExactnessCacheFingerprints(
    _ caches: [KVCache]
) -> [Qwen38MTPLiveExactnessCacheFingerprint] {
    caches.enumerated().map { layerIndex, cache in
        let states = cache.state
        let fingerprints = states.enumerated().map { stateIndex, array in
            eval(array)
            let bytes = array.asData(access: .copy).data
            return Qwen38MTPLiveExactnessArrayFingerprint(
                stateIndex: stateIndex,
                shape: array.shape,
                dtype: String(describing: array.dtype),
                byteCount: bytes.count,
                sha256: sha256Hex(bytes))
        }
        return Qwen38MTPLiveExactnessCacheFingerprint(
            layerIndex: layerIndex,
            cacheType: qwen38LiveExactnessCacheType(cache),
            offset: cache.offset,
            metaStateSHA256: sha256Hex(Data(String(reflecting: cache.metaState).utf8)),
            stateFingerprints: fingerprints)
    }
}

private func qwen38LiveExactnessCacheType(_ cache: KVCache) -> String {
    switch cache {
    case is MambaCache:
        return "recurrent-mamba"
    case is KVCacheSimple:
        return "dense-attention"
    default:
        return "unsupported:\(String(reflecting: type(of: cache)))"
    }
}

private func qwen38LiveExactnessProvenance(harnessGitSHA: String) throws -> Provenance {
    guard let chip = qwen38SysctlString("machdep.cpu.brand_string")
        ?? qwen38SysctlString("hw.model"),
        !chip.isEmpty
    else {
        throw Qwen38LiveExactnessProducerError.hardwareIdentityUnavailable
    }
    return Provenance(
        date: ISO8601DateFormatter().string(from: Date()),
        hardwareChip: chip,
        hardwareRAMBytes: ProcessInfo.processInfo.physicalMemory,
        hardwareOS: ProcessInfo.processInfo.operatingSystemVersionString,
        harnessGitSHA: harnessGitSHA,
        mlxSwiftVersion: "0.31.6",
        referenceMLXVersion: nil,
        referenceMLXLMVersion: nil,
        modelPath: Qwen38MTPLiveExactnessGate.modelPathSentinel,
        modelConfigHash: Qwen38MTPPerformanceScorecardGate.requiredArtifact.targetConfigSHA256,
        modelCheckpointManifestHash:
            Qwen38MTPPerformanceScorecardGate.requiredArtifact.targetTensorManifestSHA256,
        modelQuant: ModelQuantInfo(bits: 8, groupSize: 32),
        corpusId: nil,
        corpusContentHash: nil,
        nonce: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID)
}

private func qwen38LiveExactnessMemoryBudget(
    environment: [String: String]
) throws -> Qwen38MTPLiveExactnessMLXMemoryBudget {
    guard let memory = environment["FAST_MLX_QWEN38_MLX_MEMORY_LIMIT_BYTES"].flatMap(Int.init),
        let cache = environment["FAST_MLX_QWEN38_MLX_CACHE_LIMIT_BYTES"].flatMap(Int.init),
        memory > 0,
        cache > 0,
        cache <= memory
    else {
        throw Qwen38LiveExactnessProducerError.invalidMemoryBudget
    }
    return Qwen38MTPLiveExactnessMLXMemoryBudget(
        memoryLimitBytes: memory,
        cacheLimitBytes: cache)
}

private func qwen38LiveExactnessHostMemoryObservation(
    environment: [String: String]
) throws -> Qwen38MTPLiveExactnessHostMemoryObservation {
    let device = MTLCreateSystemDefaultDevice()
    return try qwen38LiveExactnessHostMemoryObservation(
        environment: environment,
        physicalRAMBytes: ProcessInfo.processInfo.physicalMemory,
        wiredLimitMB: try qwen38WiredLimitMB(),
        wiredLimitProvenance: .measured,
        metalRecommendedMaxWorkingSetSizeBytes:
            UInt64(device?.recommendedMaxWorkingSetSize ?? 0),
        metalCurrentAllocatedSizeBytes: UInt64(device?.currentAllocatedSize ?? 0))
}

private func qwen38LiveExactnessHostMemoryObservation(
    environment: [String: String],
    physicalRAMBytes: UInt64,
    wiredLimitMB: Int,
    wiredLimitProvenance: Qwen38MTPLiveExactnessObservationProvenance,
    metalRecommendedMaxWorkingSetSizeBytes: UInt64,
    metalCurrentAllocatedSizeBytes: UInt64
) throws -> Qwen38MTPLiveExactnessHostMemoryObservation {
    guard environment["FAST_MLX_QWEN38_HOST_USE"] == "dedicated-serving",
        environment["FAST_MLX_QWEN38_HOST_USE_SOURCE"] == "operator-assertion",
        environment["FAST_MLX_QWEN38_HOST_USE_POLICY_VERSION"]
            == Qwen38MTPLiveExactnessGate.requiredHostUsePolicyVersion,
        wiredLimitProvenance == .measured,
        wiredLimitMB >= 0,
        metalRecommendedMaxWorkingSetSizeBytes > 0,
        metalCurrentAllocatedSizeBytes < metalRecommendedMaxWorkingSetSizeBytes
    else {
        throw Qwen38LiveExactnessProducerError.invalidHostMemoryObservation
    }
    let memoryBudget = try qwen38LiveExactnessMemoryBudget(environment: environment)
    let observation = Qwen38MTPLiveExactnessHostMemoryObservation(
        hostUse: environment["FAST_MLX_QWEN38_HOST_USE"]!,
        hostUseSource: environment["FAST_MLX_QWEN38_HOST_USE_SOURCE"]!,
        hostUsePolicyVersion: environment["FAST_MLX_QWEN38_HOST_USE_POLICY_VERSION"]!,
        physicalRAMBytes: physicalRAMBytes,
        wiredLimitMB: wiredLimitMB,
        wiredLimitProvenance: wiredLimitProvenance,
        metalRecommendedMaxWorkingSetSizeBytes: metalRecommendedMaxWorkingSetSizeBytes,
        metalCurrentAllocatedSizeBytes: metalCurrentAllocatedSizeBytes,
        memoryLimitBytes: UInt64(memoryBudget.memoryLimitBytes),
        cacheLimitBytes: UInt64(memoryBudget.cacheLimitBytes),
        reservedKVBytes: try qwen38PositiveUInt64(
            environment, "FAST_MLX_QWEN38_RESERVED_KV_BYTES"),
        reservedIOBytes: try qwen38PositiveUInt64(
            environment, "FAST_MLX_QWEN38_RESERVED_IO_BYTES"),
        reservedPrefetchBytes: try qwen38PositiveUInt64(
            environment, "FAST_MLX_QWEN38_RESERVED_PREFETCH_BYTES"),
        osServiceReserveBytes: try qwen38PositiveUInt64(
            environment, "FAST_MLX_QWEN38_OS_SERVICE_RESERVE_BYTES"))
    do {
        try validateQwen38HostMemoryObservationForProducer(observation, memoryBudget: memoryBudget)
    } catch {
        throw Qwen38LiveExactnessProducerError.invalidHostMemoryObservation
    }
    return observation
}

private func applyQwen38LiveExactnessMemoryBudget(
    _ budget: Qwen38MTPLiveExactnessMLXMemoryBudget
) throws {
    Memory.memoryLimit = budget.memoryLimitBytes
    Memory.cacheLimit = budget.cacheLimitBytes
    guard Memory.memoryLimit == budget.memoryLimitBytes,
        Memory.cacheLimit == budget.cacheLimitBytes
    else {
        throw Qwen38LiveExactnessProducerError.memoryBudgetReadbackMismatch
    }
}

private func qwen38LiveExactnessLaunchBinding(
    process: Qwen38MTPLiveExactnessProcessIsolationEvidence
) throws -> Qwen38MTPPerformanceScorecardLaunchBinding {
    guard process.gdnMode == .gdnOn,
        process.observedEnv == .enabled
    else {
        throw Qwen38LiveExactnessProducerError.gdnFusionNotEnabled
    }
    let processIsolationEvidenceID = Qwen38MTPLiveExactnessGate
        .processIsolationEvidenceID(for: process)
    return Qwen38MTPPerformanceScorecardLaunchBinding(
        mode: .gdnOn,
        sourceDigest: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID,
        observedEnv: .enabled,
        processIsolationEvidenceID: processIsolationEvidenceID,
        launchDigest: Qwen38MTPPerformanceScorecardGate.launchDigest(
            mode: .gdnOn,
            sourceDigest: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID,
            observedEnv: .enabled,
            processIsolationEvidenceID: processIsolationEvidenceID))
}

private func qwen38LiveExactnessProcessFacts(
    environment: [String: String],
    executableSHA256: String? = nil,
    harnessGitSHA: String
) throws -> Qwen38MTPLiveExactnessProcessIsolationEvidence {
    let observedEnv: Qwen38MTPPerformanceScorecardGDNObservedEnv =
        environment["MLX_QWEN_FOUR_GDN"] == "1" ? .enabled : .disabled
    let processInfo = ProcessInfo.processInfo
    let executableSHA256 = try executableSHA256 ?? qwen38ExecutableSHA256()
    return Qwen38MTPLiveExactnessProcessIsolationEvidence(
        processID: Int(processInfo.processIdentifier),
        parentProcessID: Int(getppid()),
        processStartUptimeNanoseconds: try qwen38ProcessStartUptimeNanoseconds(),
        bootTimeUnixSeconds: try qwen38BootTimeUnixSeconds(),
        executableIdentitySource: .procPIDPath,
        executableSHA256: executableSHA256,
        harnessGitSHA: harnessGitSHA,
        sourceID: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID,
        gdnMode: observedEnv == .enabled ? .gdnOn : .gdnOff,
        observedEnv: observedEnv)
}

private func validateQwen38HostMemoryObservationForProducer(
    _ observation: Qwen38MTPLiveExactnessHostMemoryObservation,
    memoryBudget: Qwen38MTPLiveExactnessMLXMemoryBudget
) throws {
    guard observation.hostUse == "dedicated-serving",
        observation.hostUseSource == "operator-assertion",
        observation.hostUsePolicyVersion == Qwen38MTPLiveExactnessGate.requiredHostUsePolicyVersion,
        observation.physicalRAMBytes == Qwen38MTPPerformanceScorecardGate.requiredRAMBytes,
        observation.wiredLimitMB >= 0,
        observation.wiredLimitProvenance == .measured,
        observation.metalRecommendedMaxWorkingSetSizeBytes > 0,
        observation.metalCurrentAllocatedSizeBytes
            < observation.metalRecommendedMaxWorkingSetSizeBytes,
        UInt64(memoryBudget.memoryLimitBytes) == observation.memoryLimitBytes,
        UInt64(memoryBudget.cacheLimitBytes) == observation.cacheLimitBytes,
        observation.cacheLimitBytes <= observation.memoryLimitBytes,
        observation.reservedKVBytes > 0,
        observation.reservedIOBytes > 0,
        observation.reservedPrefetchBytes > 0,
        observation.osServiceReserveBytes > 0
    else {
        throw Qwen38LiveExactnessProducerError.invalidHostMemoryObservation
    }
    guard qwen38CheckedSum([
        observation.cacheLimitBytes,
        observation.reservedKVBytes,
    ]).map({ $0 <= observation.memoryLimitBytes }) == true else {
        throw Qwen38LiveExactnessProducerError.invalidHostMemoryObservation
    }
    let wiredLimitBytes = observation.wiredLimitMB == 0
        ? UInt64.max
        : UInt64(observation.wiredLimitMB) * 1024 * 1024
    let effectiveCeiling = min(
        observation.physicalRAMBytes,
        min(wiredLimitBytes, observation.metalRecommendedMaxWorkingSetSizeBytes))
    guard qwen38CheckedSum([
        observation.metalCurrentAllocatedSizeBytes,
        observation.memoryLimitBytes,
        observation.reservedIOBytes,
        observation.reservedPrefetchBytes,
        observation.osServiceReserveBytes,
    ]).map({ $0 <= effectiveCeiling }) == true else {
        throw Qwen38LiveExactnessProducerError.invalidHostMemoryObservation
    }
}

private func qwen38PositiveUInt64(
    _ environment: [String: String],
    _ key: String
) throws -> UInt64 {
    guard let raw = environment[key],
        let value = UInt64(raw),
        value > 0
    else {
        throw Qwen38LiveExactnessProducerError.invalidHostMemoryObservation
    }
    return value
}

private func qwen38CheckedSum(_ values: [UInt64]) -> UInt64? {
    var total: UInt64 = 0
    for value in values {
        let result = total.addingReportingOverflow(value)
        if result.overflow { return nil }
        total = result.partialValue
    }
    return total
}

private func qwen38WiredLimitMB() throws -> Int {
    var value: Int32 = 0
    var size = MemoryLayout<Int32>.stride
    let result = sysctlbyname("iogpu.wired_limit_mb", &value, &size, nil, 0)
    guard result == 0, value >= 0 else {
        throw Qwen38LiveExactnessProducerError.invalidHostMemoryObservation
    }
    return Int(value)
}

private func qwen38ExecutableSHA256(
    processID: pid_t = getpid(),
    processPath: (pid_t) -> String? = qwen38KernelReportedProcessPath
) throws -> String {
    guard let executable = processPath(processID), !executable.isEmpty else {
        throw Qwen38LiveExactnessProducerError.executableIdentityUnavailable
    }
    return sha256Hex(try Data(contentsOf: URL(fileURLWithPath: executable)))
}

private func qwen38KernelReportedProcessPath(_ processID: pid_t) -> String? {
    var buffer = [CChar](repeating: 0, count: 4096)
    let result = buffer.withUnsafeMutableBufferPointer { pointer in
        proc_pidpath(processID, pointer.baseAddress, UInt32(pointer.count))
    }
    guard result > 0 else {
        return nil
    }
    let bytes = buffer.prefix { $0 != 0 }.map(UInt8.init(bitPattern:))
    guard let path = String(bytes: bytes, encoding: .utf8),
        !path.isEmpty
    else {
        return nil
    }
    return path
}

private func qwen38BootTimeUnixSeconds() throws -> Int64 {
    var bootTime = timeval()
    var size = MemoryLayout<timeval>.stride
    let result = sysctlbyname("kern.boottime", &bootTime, &size, nil, 0)
    guard result == 0, bootTime.tv_sec > 0 else {
        throw Qwen38LiveExactnessProducerError.processIdentityUnavailable
    }
    return Int64(bootTime.tv_sec)
}

private func qwen38ProcessStartUptimeNanoseconds() throws -> UInt64 {
    var info = proc_taskallinfo()
    let size = MemoryLayout<proc_taskallinfo>.stride
    let result = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: CChar.self, capacity: size) { rebound in
            proc_pidinfo(getpid(), PROC_PIDTASKALLINFO, 0, rebound, Int32(size))
        }
    }
    guard result == Int32(size) else {
        throw Qwen38LiveExactnessProducerError.processIdentityUnavailable
    }
    let bootSeconds = try qwen38BootTimeUnixSeconds()
    let startNanoseconds = (Int64(info.pbsd.pbi_start_tvsec) - bootSeconds)
        * 1_000_000_000 + Int64(info.pbsd.pbi_start_tvusec) * 1_000
    guard startNanoseconds > 0 else {
        throw Qwen38LiveExactnessProducerError.processIdentityUnavailable
    }
    return UInt64(startNanoseconds)
}

private func qwen38HarnessGitSHA() throws -> String {
    let currentDirectory = URL(
        fileURLWithPath: FileManager.default.currentDirectoryPath,
        isDirectory: true)
    return try qwen38HarnessGitSHA(
        currentDirectory: currentDirectory,
        compiledSourceFile: URL(fileURLWithPath: #filePath))
}

private func qwen38HarnessGitSHA(
    currentDirectory: URL,
    compiledSourceFile: URL,
    liveGitSHA: (URL) -> String? = qwen38LiveGitSHA
) throws -> String {
    let compiledSourceRepositoryRoot = qwen38LiveExactnessRepositoryRoot(
        startingAt: compiledSourceFile.deletingLastPathComponent())
    let repositoryRoot = compiledSourceRepositoryRoot
        ?? qwen38LiveExactnessRepositoryRoot(startingAt: currentDirectory)
    let deploySourceRoot = qwen38LiveExactnessDeploySourceRoot(
        compiledSourceFile: compiledSourceFile)
    return try resolveQwen38LiveExactnessHarnessGitSHA(
        liveGitOutput: repositoryRoot.flatMap(liveGitSHA),
        liveRepositoryPresent: repositoryRoot != nil,
        shaFile: deploySourceRoot.flatMap(qwen38LiveExactnessDeployStamp))
}

private struct Qwen38DeploySourceFixture {
    var root: URL
    var compiledSourceFile: URL
    var requiredMarkers: [URL]
    var sha: String
}

private func makeQwen38DeploySourceFixture() throws -> Qwen38DeploySourceFixture {
    let manager = FileManager.default
    let root = manager.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let sourceFile = root.appendingPathComponent(
        "Tests/SpikeCoreTests/Qwen35ExactMTPRuntimeFactoryTests.swift")
    let requiredMarkers = [
        root.appendingPathComponent("Package.swift"),
        root.appendingPathComponent("Package.resolved"),
        root.appendingPathComponent("Sources/SpikeCore/Qwen35ExactMTPRuntimeFactory.swift"),
        sourceFile,
        root.appendingPathComponent("Vendor/mlx-swift-lm/Package.swift"),
    ]
    for marker in requiredMarkers {
        try manager.createDirectory(
            at: marker.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try Data().write(to: marker)
    }
    let sha = String(repeating: "c", count: 40)
    try "\(sha)\n".write(
        to: root.appendingPathComponent(".harness-sha"),
        atomically: true,
        encoding: .utf8)
    return Qwen38DeploySourceFixture(
        root: root,
        compiledSourceFile: sourceFile,
        requiredMarkers: requiredMarkers,
        sha: sha)
}

private func qwen38LiveExactnessDeploySourceRoot(
    compiledSourceFile: URL
) -> URL? {
    let sourceFile = compiledSourceFile.standardizedFileURL
    let testTargetDirectory = sourceFile.deletingLastPathComponent()
    let testsDirectory = testTargetDirectory.deletingLastPathComponent()
    let packageRoot = testsDirectory.deletingLastPathComponent()
    guard sourceFile.lastPathComponent == "Qwen35ExactMTPRuntimeFactoryTests.swift",
        testTargetDirectory.lastPathComponent == "SpikeCoreTests",
        testsDirectory.lastPathComponent == "Tests"
    else { return nil }

    let requiredMarkers = [
        packageRoot.appendingPathComponent("Package.swift"),
        packageRoot.appendingPathComponent("Package.resolved"),
        packageRoot.appendingPathComponent(
            "Sources/SpikeCore/Qwen35ExactMTPRuntimeFactory.swift"),
        sourceFile,
        packageRoot.appendingPathComponent("Vendor/mlx-swift-lm/Package.swift"),
    ]
    let manager = FileManager.default
    guard requiredMarkers.allSatisfy({ marker in
        guard let attributes = try? manager.attributesOfItem(atPath: marker.path),
            let type = attributes[.type] as? FileAttributeType
        else { return false }
        return type == .typeRegular
    }) else { return nil }
    return packageRoot
}

private func qwen38LiveExactnessDeployStamp(packageRoot: URL) -> String? {
    let stamp = packageRoot.appendingPathComponent(".harness-sha")
    guard let attributes = try? FileManager.default.attributesOfItem(atPath: stamp.path),
        let type = attributes[.type] as? FileAttributeType,
        type == .typeRegular
    else { return nil }
    return try? String(contentsOf: stamp, encoding: .utf8)
}

private func resolveQwen38LiveExactnessHarnessGitSHA(
    liveGitOutput: String?,
    liveRepositoryPresent: Bool,
    shaFile: String?
) throws -> String {
    let source: String
    if let liveGitOutput {
        source = liveGitOutput
    } else if liveRepositoryPresent {
        throw Qwen38LiveExactnessProducerError.harnessGitSHAUnavailable
    } else if let shaFile {
        source = shaFile
    } else {
        throw Qwen38LiveExactnessProducerError.harnessGitSHAUnavailable
    }
    let candidate = source.trimmingCharacters(in: .whitespacesAndNewlines)
    guard candidate.count == 40,
        candidate != String(repeating: "0", count: 40),
        candidate.utf8.allSatisfy({
            ($0 >= 48 && $0 <= 57) || ($0 >= 97 && $0 <= 102)
        })
    else {
        throw Qwen38LiveExactnessProducerError.invalidHarnessGitSHA
    }
    return candidate
}

private func qwen38LiveExactnessRepositoryRoot(startingAt start: URL) -> URL? {
    let components = start.standardizedFileURL.pathComponents
    let manager = FileManager.default
    for count in stride(from: components.count, through: 1, by: -1) {
        let directory = URL(
            fileURLWithPath: NSString.path(withComponents: Array(components.prefix(count))),
            isDirectory: true)
        guard manager.fileExists(atPath: directory.appendingPathComponent(".git").path),
            manager.fileExists(atPath: directory.appendingPathComponent("AGENTS.md").path),
            manager.fileExists(atPath: directory.appendingPathComponent("spike/Package.swift").path)
        else { continue }
        return directory
    }
    return nil
}

private func qwen38LiveGitSHA(repositoryRoot: URL) -> String? {
    let expectedRoot = repositoryRoot.standardizedFileURL
    guard let reportedRoot = qwen38RunGit([
        "-C", expectedRoot.path, "rev-parse", "--show-toplevel",
    ])?.trimmingCharacters(in: .whitespacesAndNewlines),
        URL(fileURLWithPath: reportedRoot).standardizedFileURL == expectedRoot,
        let sha = qwen38RunGit([
            "-C", expectedRoot.path, "rev-parse", "HEAD",
        ])?.trimmingCharacters(in: .whitespacesAndNewlines),
        let status = qwen38RunGit([
            "-C", expectedRoot.path, "status", "--porcelain",
            "--untracked-files=normal", "--", "spike", "experiments",
        ])
    else { return nil }
    return status.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        ? sha : "\(sha)-dirty"
}

private func qwen38RunGit(_ arguments: [String]) -> String? {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = ["git"] + arguments
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = Pipe()
    do {
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return nil }
        return String(data: data, encoding: .utf8)
    } catch {
        return nil
    }
}

private func qwen38SysctlString(_ name: String) -> String? {
    var size = 0
    let sizeResult = name.withCString { sysctlbyname($0, nil, &size, nil, 0) }
    guard sizeResult == 0, size > 1 else { return nil }
    var bytes = [CChar](repeating: 0, count: size)
    let readResult = name.withCString { namePointer in
        bytes.withUnsafeMutableBytes { buffer in
            sysctlbyname(namePointer, buffer.baseAddress, &size, nil, 0)
        }
    }
    guard readResult == 0 else { return nil }
    if let terminator = bytes.firstIndex(of: 0) {
        bytes.removeSubrange(terminator...)
    }
    return String(decoding: bytes.map(UInt8.init(bitPattern:)), as: UTF8.self)
        .trimmingCharacters(in: .whitespacesAndNewlines)
}

private func writeValidatedQwen38LiveExactnessEvidenceFile(
    _ data: Data,
    to url: URL
) throws {
    _ = try Qwen38MTPLiveExactnessGate.validateJSONL(data)
    guard url.isFileURL else { throw CocoaError(.fileWriteUnsupportedScheme) }
    let directory = url.deletingLastPathComponent()
    let temporaryURL = directory.appendingPathComponent(
        ".\(url.lastPathComponent).\(UUID().uuidString).tmp")
    var descriptor = temporaryURL.withUnsafeFileSystemRepresentation { path -> Int32 in
        guard let path else { return -1 }
        return open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0o600)
    }
    guard descriptor >= 0 else {
        throw CocoaError(.fileWriteUnknown)
    }
    defer {
        if descriptor >= 0 { _ = close(descriptor) }
        temporaryURL.withUnsafeFileSystemRepresentation { path in
            if let path { _ = unlink(path) }
        }
    }

    do {
        try data.withUnsafeBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress else { return }
            var written = 0
            while written < rawBuffer.count {
                let count = write(
                    descriptor,
                    baseAddress.advanced(by: written),
                    rawBuffer.count - written)
                guard count > 0 else { throw CocoaError(.fileWriteUnknown) }
                written += count
            }
        }
        guard fsync(descriptor) == 0 else {
            throw CocoaError(.fileWriteUnknown)
        }
        let closeResult = close(descriptor)
        descriptor = -1
        guard closeResult == 0 else { throw CocoaError(.fileWriteUnknown) }
    } catch {
        throw error
    }

    let linkResult = temporaryURL.withUnsafeFileSystemRepresentation { temporaryPath in
        url.withUnsafeFileSystemRepresentation { destinationPath in
            guard let temporaryPath, let destinationPath else { return Int32(-1) }
            return link(temporaryPath, destinationPath)
        }
    }
    guard linkResult == 0 else {
        if errno == EEXIST { throw CocoaError(.fileWriteFileExists) }
        throw CocoaError(.fileWriteUnknown)
    }

    let directoryDescriptor = directory.withUnsafeFileSystemRepresentation { path -> Int32 in
        guard let path else { return -1 }
        return open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
    }
    guard directoryDescriptor >= 0 else {
        unlinkQwen38EvidenceDestination(url)
        throw CocoaError(.fileWriteUnknown)
    }
    let directorySyncResult = fsync(directoryDescriptor)
    let directoryCloseResult = close(directoryDescriptor)
    guard directorySyncResult == 0, directoryCloseResult == 0 else {
        unlinkQwen38EvidenceDestination(url)
        throw CocoaError(.fileWriteUnknown)
    }
}

private func unlinkQwen38EvidenceDestination(_ url: URL) {
    url.withUnsafeFileSystemRepresentation { path in
        if let path { _ = unlink(path) }
    }
}

private func qwen38DedicatedHostMemoryEnvironment() -> [String: String] {
    let gib = UInt64(1024 * 1024 * 1024)
    return [
        "FAST_MLX_QWEN38_HOST_USE": "dedicated-serving",
        "FAST_MLX_QWEN38_HOST_USE_SOURCE": "operator-assertion",
        "FAST_MLX_QWEN38_HOST_USE_POLICY_VERSION":
            Qwen38MTPLiveExactnessGate.requiredHostUsePolicyVersion,
        "FAST_MLX_QWEN38_MLX_MEMORY_LIMIT_BYTES": "\(220 * gib)",
        "FAST_MLX_QWEN38_MLX_CACHE_LIMIT_BYTES": "\(48 * gib)",
        "FAST_MLX_QWEN38_RESERVED_KV_BYTES": "\(40 * gib)",
        "FAST_MLX_QWEN38_RESERVED_IO_BYTES": "\(2 * gib)",
        "FAST_MLX_QWEN38_RESERVED_PREFETCH_BYTES": "\(4 * gib)",
        "FAST_MLX_QWEN38_OS_SERVICE_RESERVE_BYTES": "\(8 * gib)",
    ]
}

private func syntheticQwen38LiveExactnessProcessFacts()
    -> Qwen38MTPLiveExactnessProcessIsolationEvidence
{
    Qwen38MTPLiveExactnessProcessIsolationEvidence(
        processID: 44_001,
        parentProcessID: 44_000,
        processStartUptimeNanoseconds: 123_456_789,
        bootTimeUnixSeconds: 1_777_000_000,
        executableIdentitySource: .procPIDPath,
        executableSHA256: String(repeating: "a", count: 64),
        harnessGitSHA: String(repeating: "e", count: 40),
        sourceID: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID,
        gdnMode: .gdnOn,
        observedEnv: .enabled)
}

private func syntheticQwen38LiveExactnessLaunchBinding()
    -> Qwen38MTPPerformanceScorecardLaunchBinding
{
    let process = syntheticQwen38LiveExactnessProcessFacts()
    let processIsolationEvidenceID = Qwen38MTPLiveExactnessGate
        .processIsolationEvidenceID(for: process)
    return Qwen38MTPPerformanceScorecardLaunchBinding(
        mode: .gdnOn,
        sourceDigest: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID,
        observedEnv: .enabled,
        processIsolationEvidenceID: processIsolationEvidenceID,
        launchDigest: Qwen38MTPPerformanceScorecardGate.launchDigest(
            mode: .gdnOn,
            sourceDigest: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID,
            observedEnv: .enabled,
            processIsolationEvidenceID: processIsolationEvidenceID))
}

private func syntheticQwen38LiveExactnessHostMemoryObservation()
    -> Qwen38MTPLiveExactnessHostMemoryObservation
{
    let gib = UInt64(1024 * 1024 * 1024)
    return Qwen38MTPLiveExactnessHostMemoryObservation(
        hostUse: "dedicated-serving",
        hostUseSource: "operator-assertion",
        hostUsePolicyVersion: Qwen38MTPLiveExactnessGate.requiredHostUsePolicyVersion,
        physicalRAMBytes: 256 * gib,
        wiredLimitMB: 245_760,
        wiredLimitProvenance: .measured,
        metalRecommendedMaxWorkingSetSizeBytes: 240 * gib,
        metalCurrentAllocatedSizeBytes: 2 * gib,
        memoryLimitBytes: 220 * gib,
        cacheLimitBytes: 48 * gib,
        reservedKVBytes: 40 * gib,
        reservedIOBytes: 2 * gib,
        reservedPrefetchBytes: 4 * gib,
        osServiceReserveBytes: 8 * gib)
}

private enum Qwen38LiveExactnessProducerError: Error, Equatable {
    case runtimeBindingDrift
    case hardwareIdentityUnavailable
    case harnessGitSHAUnavailable
    case invalidHarnessGitSHA
    case gdnFusionNotEnabled
    case invalidMemoryBudget
    case memoryBudgetReadbackMismatch
    case invalidHostMemoryObservation
    case executableIdentityUnavailable
    case processIdentityUnavailable
}

private func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private enum ExactSnapshotDownloaderError: Error {
    case unexpectedDownloadRequest
    case tokenizerLoadReached
}

private struct Qwen38ExactPreloadFixture {
    let root: URL
    let targetDirectory: URL
    let drafterDirectory: URL

    func cleanup() {
        try? FileManager.default.removeItem(at: root)
    }
}

private struct Qwen38PreloadTensorFixture: Decodable {
    let name: String
    let shape: [Int]
    let dtype: String
}

private func makeQwen38SyntheticPreloadDriftFixture() throws -> Qwen38ExactPreloadFixture {
    let root = try qwen38PreloadTemporaryDirectory()
    let targetDirectory = root.appending(component: "target")
    let drafterDirectory = root.appending(component: "drafter")
    try FileManager.default.createDirectory(at: targetDirectory, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: drafterDirectory, withIntermediateDirectories: true)

    try qwen38SyntheticTargetConfig()
        .write(to: targetDirectory.appending(component: "config.json"))
    try qwen38SyntheticDrafterConfig()
        .write(to: drafterDirectory.appending(component: "config.json"))
    try writeQwen38SyntheticTokenizer(to: targetDirectory)
    try writeQwen38SyntheticTokenizer(to: drafterDirectory)
    try qwen38PreloadSafetensorHeaderData([
        .init(name: "model.embed_tokens.weight", shape: [248_320, 4096], dtype: "BF16")
    ]).write(to: targetDirectory.appending(component: "model.safetensors"))
    try qwen38PreloadSafetensorHeaderData([
        .init(name: "fc.weight", shape: [4096, 1280], dtype: "U32")
    ]).write(to: drafterDirectory.appending(component: "model.safetensors"))

    return Qwen38ExactPreloadFixture(
        root: root,
        targetDirectory: targetDirectory,
        drafterDirectory: drafterDirectory)
}

private func makeQwen38ExactPreloadFixture() throws -> Qwen38ExactPreloadFixture {
    let environment = ProcessInfo.processInfo.environment
    let tokenizerPath = environment["FAST_MLX_QWEN38_TOKENIZER_SNAPSHOT"]
        ?? environment["FAST_MLX_QWEN38_TARGET_SNAPSHOT"]
    guard let tokenizerPath else {
        throw XCTSkip(
            "exact qwen38 preload fixture requires a local tokenizer snapshot path")
    }
    let tokenizerDirectory = URL(fileURLWithPath: tokenizerPath, isDirectory: true)
    guard FileManager.default.fileExists(
        atPath: tokenizerDirectory.appending(component: "tokenizer.json").path)
            || FileManager.default.fileExists(
                atPath: tokenizerDirectory.appending(component: "vocab.json").path)
    else {
        throw XCTSkip(
            "exact qwen38 preload fixture tokenizer path has no tokenizer.json or vocab.json")
    }

    let root = try qwen38PreloadTemporaryDirectory()
    let targetDirectory = root.appending(component: "target")
    let drafterDirectory = root.appending(component: "drafter")
    try FileManager.default.createDirectory(at: targetDirectory, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: drafterDirectory, withIntermediateDirectories: true)

    try qwen38PreloadFixtureData(named: "qwen38-27b-target-config")
        .write(to: targetDirectory.appending(component: "config.json"))
    try qwen38PreloadFixtureData(named: "qwen38-27b-mtp-config")
        .write(to: drafterDirectory.appending(component: "config.json"))
    try copyQwen38PreloadTokenizer(from: tokenizerDirectory, to: targetDirectory)
    try copyQwen38PreloadTokenizer(from: tokenizerDirectory, to: drafterDirectory)

    let targetTensors = try qwen38PreloadTensorFixture(named: "qwen38-27b-target-tensors")
    try qwen38PreloadSafetensorHeaderData(targetTensors)
        .write(to: targetDirectory.appending(component: "model.safetensors"))
    try qwen38PreloadSafetensorHeaderData(
        Qwen35ExactMTPKnownArtifactLocks.qwen38_27BMXFP8Depth1.drafterTensors)
        .write(to: drafterDirectory.appending(component: "model.safetensors"))

    return Qwen38ExactPreloadFixture(
        root: root,
        targetDirectory: targetDirectory,
        drafterDirectory: drafterDirectory)
}

private func qwen38PreloadFixtureData(named name: String) throws -> Data {
    let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let url = root.appending(component: "Tests/HarnessCoreTests/Fixtures/mtp/\(name).json")
    var data = try Data(contentsOf: url)
    if data.last == 0x0a {
        data.removeLast()
    }
    return data
}

private func qwen38PreloadTensorFixture(named name: String) throws -> [Qwen38PreloadTensorFixture] {
    try JSONDecoder().decode(
        [Qwen38PreloadTensorFixture].self,
        from: qwen38PreloadFixtureData(named: name))
}

private func copyQwen38PreloadTokenizer(from source: URL, to destination: URL) throws {
    let manager = FileManager.default
    for name in ["tokenizer.json", "vocab.json"] {
        let sourceURL = source.appending(component: name)
        guard manager.fileExists(atPath: sourceURL.path) else { continue }
        try Data(contentsOf: sourceURL).write(to: destination.appending(component: name))
        return
    }
    throw Qwen35ExactMTPAdmissionError.missingFile(role: .target, name: "tokenizer.json")
}

private func writeQwen38SyntheticTokenizer(to directory: URL) throws {
    let data = try JSONSerialization.data(
        withJSONObject: ["model": ["vocab": ["token-a": 0, "token-b": 1]]],
        options: [.sortedKeys])
    try data.write(to: directory.appending(component: "tokenizer.json"))
}

private func qwen38SyntheticTargetConfig() -> Data {
    Data("""
    {"model_type":"qwen3_5","quantization":{"bits":4,"group_size":64,"mode":"affine"},"text_config":\(qwen38SyntheticTextConfig())}
    """.utf8)
}

private func qwen38SyntheticDrafterConfig() -> Data {
    Data("""
    {"block_size":3,"model_type":"qwen3_5_mtp","quantization":{"bits":5,"group_size":64,"mode":"affine"},"text_config":\(qwen38SyntheticTextConfig())}
    """.utf8)
}

private func qwen38SyntheticTextConfig() -> String {
    """
    {"model_type":"qwen3_5_text","hidden_size":4096,"intermediate_size":12288,"vocab_size":248320,"num_hidden_layers":32,"full_attention_interval":4,"num_attention_heads":16,"num_key_value_heads":4,"head_dim":256,"mtp_num_hidden_layers":1,"mtp_use_dedicated_embeddings":false}
    """
}

private func qwen38PreloadSafetensorHeaderData(
    _ tensors: [Qwen38PreloadTensorFixture]
) throws -> Data {
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

private func qwen38PreloadSafetensorHeaderData(
    _ tensors: [Qwen35ExactMTPTensorDescriptor]
) throws -> Data {
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

private func qwen38PreloadTemporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appending(component: "qwen38-exact-preload-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private struct ExactSnapshotDownloader: Downloader {
    let target: URL
    let drafter: URL
    var selection: Qwen35ExactMTPRuntimeSelection = .qwen35_9BDepth1

    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        guard !useLatest, patterns == exactSnapshotPatterns else {
            throw ExactSnapshotDownloaderError.unexpectedDownloadRequest
        }
        progressHandler(Progress(totalUnitCount: 1))
        let row = selectedSnapshotRow(selection)
        switch (id, revision) {
        case (row.targetID, row.targetRevision):
            return target
        case (row.drafterID, row.drafterRevision):
            return drafter
        default:
            throw ExactSnapshotDownloaderError.unexpectedDownloadRequest
        }
    }
}

private struct ExactSnapshotRow {
    let targetID: String
    let targetRevision: String
    let drafterID: String
    let drafterRevision: String
}

private func selectedSnapshotRow(
    _ selection: Qwen35ExactMTPRuntimeSelection
) -> ExactSnapshotRow {
    switch selection {
    case .qwen35_9BDepth1:
        ExactSnapshotRow(
            targetID: "mlx-community/Qwen3.5-9B-MLX-4bit",
            targetRevision: "938d8919941c6e7efd3c7150eff7fe9d12afa631",
            drafterID: "mlx-community/Qwen3.5-9B-MTP-5bit",
            drafterRevision: "994730d199bff7799aa3ddef33a96723967a3e33")
    case .qwen38_27BMXFP8Depth1:
        ExactSnapshotRow(
            targetID: "mlx-community/Qwen3.8-27B-mxfp8",
            targetRevision: "d48d163bcdf24acaf656474854ab88ea17d65bd1",
            drafterID: "mlx-community/Qwen3.8-27B-MTP-mxfp8",
            drafterRevision: "a50634460045613f166b09b13519466e801c6568")
    }
}

private struct PreflightOnlyTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        throw ExactSnapshotDownloaderError.tokenizerLoadReached
    }
}

private func XCTAssertThrowsErrorAsync<T>(
    _ expression: @autoclosure () async throws -> T,
    _ message: @autoclosure () -> String = "",
    file: StaticString = #filePath,
    line: UInt = #line,
    _ errorHandler: (_ error: any Error) -> Void = { _ in }
) async {
    do {
        _ = try await expression()
        XCTFail(message(), file: file, line: line)
    } catch {
        errorHandler(error)
    }
}
