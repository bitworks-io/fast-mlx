import Foundation
import HarnessCore
import ScorecardPairControl
import ServingCore
@_spi(ProductionRouteEvidence) import SpikeServingAdapters
@_spi(ProductionRouteEvidenceTesting) import SpikeServingAdapters
import XCTest

@testable import fastmlx_harness

final class Qwen38ScorecardProductionRouteObservationCLITests: XCTestCase {
    private typealias CLIError = Qwen38ScorecardProductionRouteObservationCLIError

    func testArgumentsFailClosedAndDiagnosticsDoNotEchoInputs() throws {
        let parsed = try parseQwen38ScorecardProductionRouteObservationArguments(
            validArguments())

        XCTAssertEqual(parsed.hostUse, "dedicated-serving")
        XCTAssertEqual(parsed.hostUseSource, "operator-assertion")
        XCTAssertEqual(parsed.expectedChip, "Apple M3 Ultra")
        XCTAssertEqual(parsed.contextTokens, 32_768)

        XCTAssertThrowsError(try parseQwen38ScorecardProductionRouteObservationArguments(
            validArguments(removing: "--target")
        )) { error in
            XCTAssertEqual(error as? CLIError, .missingFlag("--target"))
        }
        XCTAssertThrowsError(try parseQwen38ScorecardProductionRouteObservationArguments(
            validArguments() + ["--target", "/" + "private/operator/model"]
        )) { error in
            XCTAssertEqual(error as? CLIError, .duplicateFlag("--target"))
        }
        XCTAssertThrowsError(try parseQwen38ScorecardProductionRouteObservationArguments(
            validArguments()
                + ["--route-evidence", "/" + "private/operator/manual.json"]
        )) { error in
            XCTAssertEqual(error as? CLIError, .unknownFlag)
            XCTAssertFalse(
                qwen38ScorecardProductionRouteObservationExternalDiagnostic(error)
                    .contains("private"))
        }
        XCTAssertThrowsError(try parseQwen38ScorecardProductionRouteObservationArguments(
            ["operator"] + validArguments()
        )) { error in
            XCTAssertEqual(error as? CLIError, .unexpectedPositional)
        }
        XCTAssertThrowsError(try parseQwen38ScorecardProductionRouteObservationArguments(
            validArguments(replacing: "--context-tokens", with: "0")
        )) { error in
            XCTAssertEqual(error as? CLIError, .invalidInteger("--context-tokens"))
        }
        XCTAssertThrowsError(try parseQwen38ScorecardProductionRouteObservationArguments(
            validArguments(replacing: "--host-use", with: "shared")
        )) { error in
            XCTAssertEqual(error as? CLIError, .invalidHostAssertion)
        }
    }

    func testStrictPeakRSSReadbackUsesZeroForFailureOrInvalidSample() {
        XCTAssertEqual(
            qwen38ProductionRoutePeakRSSBytes(
                getrusageResult: -1,
                maxResidentSetSize: 4_096),
            0)
        XCTAssertEqual(
            qwen38ProductionRoutePeakRSSBytes(
                getrusageResult: 0,
                maxResidentSetSize: 0),
            0)
        XCTAssertEqual(
            qwen38ProductionRoutePeakRSSBytes(
                getrusageResult: 0,
                maxResidentSetSize: 4_096),
            4_096)
    }

    func testRelativePathsFailBeforeSourcePreflight() async throws {
        let sandbox = try Sandbox()
        let trace = CallTrace()

        do {
            _ = try await produceQwen38ScorecardProductionRouteObservation(
                arguments: validArguments(
                    target: "relative-target",
                    drafter: sandbox.drafter.path,
                    output: sandbox.output.path),
                dependencies: .test(sourcePreflight: { _, _ in
                    await trace.record("source")
                    return sourceLockFixture()
                }))
            XCTFail("expected relative target failure")
        } catch {
            XCTAssertEqual(error as? CLIError, .invalidTarget)
        }
        let firstEvents = await trace.snapshot()
        XCTAssertEqual(firstEvents, [])

        do {
            _ = try await produceQwen38ScorecardProductionRouteObservation(
                arguments: validArguments(
                    target: sandbox.target.path,
                    drafter: "relative-drafter",
                    output: sandbox.output.path),
                dependencies: .test(sourcePreflight: { _, _ in
                    await trace.record("source")
                    return sourceLockFixture()
                }))
            XCTFail("expected relative drafter failure")
        } catch {
            XCTAssertEqual(error as? CLIError, .invalidDrafter)
        }

        XCTAssertThrowsError(
            try Qwen38ScorecardProductionRouteObservationFreshOutput
                .validate("relative-output.json")
        ) { error in
            XCTAssertEqual(error as? CLIError, .unsafeOutput)
        }
    }

    func testGDNHostMemoryAndSourcePreflightHappenBeforeModelLoad() async throws {
        let sandbox = try Sandbox()
        let trace = CallTrace()
        let source = sourceLockFixture()

        do {
            _ = try await produceQwen38ScorecardProductionRouteObservation(
                arguments: validArguments(
                    target: sandbox.target.path,
                    drafter: sandbox.drafter.path,
                    output: sandbox.output.path),
                dependencies: .test(
                    environment: {
                        await trace.record("environment")
                        return "1"
                    },
                    hostSnapshot: {
                        await trace.record("host")
                        return hostSnapshot()
                    },
                    sourcePreflight: { _, _ in
                        await trace.record("source")
                        return source
                    },
                    loadModel: { _ in
                        await trace.record("load")
                        throw CLIError.modelLoadFailed
                    }))
            XCTFail("expected load failure")
        } catch {
            XCTAssertEqual(error as? CLIError, .modelLoadFailed)
        }

        let events = await trace.snapshot()
        XCTAssertEqual(events, [
            "environment",
            "host",
            "source",
            "load",
        ])
        XCTAssertFalse(FileManager.default.fileExists(atPath: sandbox.output.path))
    }

    func testInvalidBaselineProcessMemoryFailsBeforeSourcePreflightOrLoad()
        async throws
    {
        let sandbox = try Sandbox()
        let trace = CallTrace()

        do {
            _ = try await produceQwen38ScorecardProductionRouteObservation(
                arguments: validArguments(
                    target: sandbox.target.path,
                    drafter: sandbox.drafter.path,
                    output: sandbox.output.path),
                dependencies: .test(
                    sourcePreflight: { _, _ in
                        await trace.record("source")
                        return sourceLockFixture()
                    },
                    loadModel: { _ in
                        await trace.record("load")
                        throw CLIError.modelLoadFailed
                    },
                    physicalFootprintBytes: { 0 }))
            XCTFail("expected invalid baseline process memory")
        } catch {
            XCTAssertEqual(error as? CLIError, .invalidProcessMemoryReadback)
        }

        let events = await trace.snapshot()
        XCTAssertEqual(events, [])
        XCTAssertFalse(FileManager.default.fileExists(atPath: sandbox.output.path))
    }

    func testUnsafeBaselineThermalStateFailsBeforeSourcePreflightOrLoad()
        async throws
    {
        let sandbox = try Sandbox()
        let trace = CallTrace()

        do {
            _ = try await produceQwen38ScorecardProductionRouteObservation(
                arguments: validArguments(
                    target: sandbox.target.path,
                    drafter: sandbox.drafter.path,
                    output: sandbox.output.path),
                dependencies: .test(
                    sourcePreflight: { _, _ in
                        await trace.record("source")
                        return sourceLockFixture()
                    },
                    loadModel: { _ in
                        await trace.record("load")
                        throw CLIError.modelLoadFailed
                    },
                    thermalState: { "serious" }))
            XCTFail("expected unsafe baseline thermal state")
        } catch {
            XCTAssertEqual(error as? CLIError, .initialThermalUnsafe)
        }

        let events = await trace.snapshot()
        XCTAssertEqual(events, [])
        XCTAssertFalse(FileManager.default.fileExists(atPath: sandbox.output.path))
    }

    func testMissingGDNFailsBeforeSourcePreflightOrLoad() async throws {
        let sandbox = try Sandbox()
        let trace = CallTrace()

        do {
            _ = try await produceQwen38ScorecardProductionRouteObservation(
                arguments: validArguments(
                    target: sandbox.target.path,
                    drafter: sandbox.drafter.path,
                    output: sandbox.output.path),
                dependencies: .test(
                    environment: {
                        await trace.record("environment")
                        return nil
                    },
                    hostSnapshot: {
                        await trace.record("host")
                        return hostSnapshot()
                    },
                    sourcePreflight: { _, _ in
                        await trace.record("source")
                        return sourceLockFixture()
                    },
                    loadModel: { _ in
                        await trace.record("load")
                        throw CLIError.modelLoadFailed
                    }))
            XCTFail("expected GDN failure")
        } catch {
            XCTAssertEqual(error as? CLIError, .invalidGDNEnvironment)
        }

        let events = await trace.snapshot()
        XCTAssertEqual(events, ["environment"])
        XCTAssertFalse(FileManager.default.fileExists(atPath: sandbox.output.path))
    }

    func testSourcePreflightFailurePreventsLoadAndOutput() async throws {
        let sandbox = try Sandbox()
        let trace = CallTrace()

        do {
            _ = try await produceQwen38ScorecardProductionRouteObservation(
                arguments: validArguments(
                    target: sandbox.target.path,
                    drafter: sandbox.drafter.path,
                    output: sandbox.output.path),
                dependencies: .test(
                    sourcePreflight: { _, _ in
                        await trace.record("source")
                        throw CLIError.sourcePreflightFailed
                    },
                    loadModel: { _ in
                        await trace.record("load")
                        throw CLIError.modelLoadFailed
                    }))
            XCTFail("expected source preflight failure")
        } catch {
            XCTAssertEqual(error as? CLIError, .sourcePreflightFailed)
        }

        let events = await trace.snapshot()
        XCTAssertEqual(events, ["source"])
        XCTAssertFalse(FileManager.default.fileExists(atPath: sandbox.output.path))
    }

    func testValidLoadedProvenanceFixtureReachesActualRunnerAndWritesArtifact()
        async throws
    {
        let sandbox = try Sandbox()
        let trace = CallTrace()
        let source = sourceLockFixture()
        let loaded = try makeRouteObservationLoadedFixture()

        let message = try await produceQwen38ScorecardProductionRouteObservation(
            arguments: validArguments(
                target: sandbox.target.path,
                drafter: sandbox.drafter.path,
                output: sandbox.output.path),
            dependencies: .test(
                hostSnapshot: {
                    await trace.record("host")
                    return hostSnapshot()
                },
                sourcePreflight: { _, _ in
                    await trace.record("source")
                    return source
                },
                loadModel: { configuration in
                    await trace.record("load")
                    XCTAssertEqual(configuration.modelDirectory.path, sandbox.target.path)
                    XCTAssertEqual(configuration.memoryLimitBytes, 180_000_000_000)
                    XCTAssertEqual(configuration.cacheLimitBytes, 10_000_000_000)
                    XCTAssertEqual(configuration.maxReservedKVBytes, 20_000_000_000)
                    XCTAssertEqual(configuration.maxContextTokens, 32_768)
                    XCTAssertEqual(configuration.maxReservedContextTokens, 32_768)
                    XCTAssertEqual(configuration.initialDecodeReserve, 8)
                    XCTAssertEqual(configuration.coordinatorConfiguration.maxActiveSlots, 4)
                    XCTAssertEqual(configuration.coordinatorConfiguration.maxPrefillSlots, 4)
                    XCTAssertEqual(configuration.coordinatorConfiguration.maxQueuedRequests, 4)
                    XCTAssertGreaterThanOrEqual(configuration.traceLimit, 4_096)
                    XCTAssertNil(configuration.soloPLDPolicy)
                    XCTAssertEqual(configuration.kvQuantTier, .fp16)
                    XCTAssertTrue(configuration.allowHybridQwen35)
                    switch configuration.backendConfiguration.admission {
                    case .dynamic(let admission, .automatic):
                        XCTAssertFalse(admission.soloPLDQualified)
                        XCTAssertEqual(admission.maximumBatchRequests, 4)
                        XCTAssertEqual(admission.maximumQueuedRequests, 4)
                    default:
                        XCTFail("expected dynamic production coalescing")
                    }
                    return loaded
                },
                shutdownLoaded: { loaded in
                    await trace.record("shutdown")
                    await loaded.backend.shutdown()
                },
                clearMLXCache: {
                    await trace.record("clear")
                }))

        XCTAssertEqual(message, "qwen38-production-route-observation: WROTE unsigned")
        let events = await trace.snapshot()
        XCTAssertEqual(events, [
            "host",
            "source",
            "load",
            "source",
            "source",
            "shutdown",
            "clear",
            "host",
        ])

        let artifact = try JSONDecoder().decode(
            Qwen38ScorecardProductionRouteObservationArtifact.self,
            from: Data(contentsOf: sandbox.output))
        XCTAssertEqual(artifact.kind, "fast-mlx.production-route-observation.v1")
        XCTAssertEqual(artifact.attestationStatus, "unsigned")
        XCTAssertFalse(artifact.isProductionRouteReceipt)
        XCTAssertFalse(artifact.controllerSignatureVerified)
        XCTAssertFalse(artifact.promotionAuthorized)
        XCTAssertFalse(artifact.runtimeAuthorityGranted)
        XCTAssertTrue(artifact.runnerObservationDigestExplanation.contains("not"))
        XCTAssertEqual(artifact.sourceLock, source)
        XCTAssertEqual(
            artifact.sourceLock.lockAuthenticationScope,
            "config-tokenizer-tensor-header-descriptor-metadata-only")
        XCTAssertFalse(
            artifact.sourceLock.safetensorPayloadBytesAuthenticated)
        XCTAssertTrue(
            artifact.sourceLock
                .safetensorPayloadBytesAuthenticationExplanation
                .contains("does not authenticate safetensor payload bytes"))
        XCTAssertEqual(artifact.environment.gdnMode, "gdn-on")
        XCTAssertEqual(artifact.environment.observedGDNEnv, "enabled")
        XCTAssertEqual(artifact.host.hostUse, "dedicated-serving")
        XCTAssertEqual(artifact.host.hostUseSource, "operator-assertion")
        XCTAssertEqual(
            artifact.host.hostUsePolicyVersion,
            Qwen38MTPLiveExactnessGate.requiredHostUsePolicyVersion)
        XCTAssertEqual(artifact.host.chipName, "Apple M3 Ultra")
        XCTAssertEqual(artifact.host.physicalRAMBytes, 274_877_906_944)
        XCTAssertEqual(artifact.host.wiredLimitProvenance, "measured")
        XCTAssertEqual(artifact.memoryBudget.contextTokens, 32_768)
        XCTAssertEqual(artifact.startupReport.defaultMaximumCompletionTokens, 8)
        XCTAssertEqual(artifact.c2.concurrency, 2)
        XCTAssertEqual(artifact.c4.concurrency, 4)
        XCTAssertEqual(artifact.c2.requests.map(\.outputTokenIDs), [[2_000], [2_001]])
        XCTAssertEqual(artifact.c4.requests.map(\.outputTokenIDs), [
            [2_000], [2_001], [2_002], [2_003],
        ])
        XCTAssertEqual(artifact.runnerObservationDigest.count, 64)
        XCTAssertEqual(artifact.artifactDigest.count, 64)
        XCTAssertEqual(
            artifact.artifactDigest,
            try qwen38ScorecardProductionRouteObservationArtifactDigest(artifact))

        let raw = try String(contentsOf: sandbox.output, encoding: .utf8)
        XCTAssertTrue(raw.contains("targetTensorHeaderDescriptorManifestSHA256"))
        XCTAssertTrue(raw.contains("drafterTensorHeaderDescriptorManifestSHA256"))
        XCTAssertTrue(raw.contains("metadataProofVerified"))
        XCTAssertFalse(raw.contains("targetTensorManifestSHA256"))
        XCTAssertFalse(raw.contains("drafterTensorManifestSHA256"))
        XCTAssertFalse(raw.contains("modelProofVerified"))
        XCTAssertFalse(raw.contains(sandbox.target.path))
        XCTAssertFalse(raw.contains(sandbox.drafter.path))
        XCTAssertFalse(raw.contains("Report a stable"))
        XCTAssertFalse(raw.contains("fixture-request"))
        XCTAssertFalse(raw.contains("prompt"))
    }

    func testOutputExistsAndSymlinkFailBeforeSourcePreflight() async throws {
        let sandbox = try Sandbox()
        try Data("old".utf8).write(to: sandbox.output)
        let trace = CallTrace()

        do {
            _ = try await produceQwen38ScorecardProductionRouteObservation(
                arguments: validArguments(
                    target: sandbox.target.path,
                    drafter: sandbox.drafter.path,
                    output: sandbox.output.path),
                dependencies: .test(sourcePreflight: { _, _ in
                    await trace.record("source")
                    return sourceLockFixture()
                }))
            XCTFail("expected existing output failure")
        } catch {
            XCTAssertEqual(error as? CLIError, .outputExists)
        }
        let events = await trace.snapshot()
        XCTAssertEqual(events, [])

        let symlink = sandbox.root.appendingPathComponent("alias.json")
        try FileManager.default.createSymbolicLink(
            at: symlink,
            withDestinationURL: sandbox.output)
        XCTAssertThrowsError(
            try Qwen38ScorecardProductionRouteObservationFreshOutput
                .validate(symlink.path)
        ) { error in
            XCTAssertEqual(error as? CLIError, .unsafeOutput)
        }
    }

    func testInjectedWriteFailurePublishesNothing() async throws {
        let sandbox = try Sandbox()

        do {
            _ = try await produceQwen38ScorecardProductionRouteObservation(
                arguments: validArguments(
                    target: sandbox.target.path,
                    drafter: sandbox.drafter.path,
                    output: sandbox.output.path),
                dependencies: .test(
                    sourcePreflight: { _, _ in sourceLockFixture() },
                    loadModel: { _ in try makeRouteObservationLoadedFixture() },
                    writeFresh: { _, _ in throw CLIError.outputWriteFailed }))
            XCTFail("expected write failure")
        } catch {
            XCTAssertEqual(error as? CLIError, .outputWriteFailed)
        }

        XCTAssertFalse(FileManager.default.fileExists(atPath: sandbox.output.path))
    }

    func testWriterUnlinksDestinationWhenDirectoryFsyncFailsAfterLink()
        throws
    {
        let sandbox = try Sandbox()

        XCTAssertThrowsError(
            try writeFreshQwen38ScorecardProductionRouteObservation(
                Data("{\"ok\":true}".utf8),
                sandbox.output.path,
                directoryFsync: { _ in -1 })
        ) { error in
            XCTAssertEqual(error as? CLIError, .outputWriteFailed)
        }

        XCTAssertFalse(FileManager.default.fileExists(atPath: sandbox.output.path))
        XCTAssertEqual(
            try FileManager.default.contentsOfDirectory(atPath: sandbox.root.path)
                .sorted(),
            ["drafter", "target"])
    }

    func testRollbackPreservesDestinationReplacedBeforeDirectoryFsyncFailure()
        throws
    {
        let sandbox = try Sandbox()
        let replacement = Data("{\"replacement\":true}".utf8)

        XCTAssertThrowsError(
            try writeFreshQwen38ScorecardProductionRouteObservation(
                Data("{\"ok\":true}".utf8),
                sandbox.output.path,
                directoryFsync: { _ in
                    try? FileManager.default.removeItem(at: sandbox.output)
                    try? replacement.write(to: sandbox.output)
                    return -1
                })
        ) { error in
            XCTAssertEqual(error as? CLIError, .outputWriteFailed)
        }

        XCTAssertEqual(try Data(contentsOf: sandbox.output), replacement)
    }

    func testPostLoadSourcePreflightDriftShutsDownClearAndEmitsNoArtifact()
        async throws
    {
        let sandbox = try Sandbox()
        let loaded = try makeRouteObservationLoadedFixture()
        let trace = CallTrace()
        let sources = SourcePreflightSequence([
            sourceLockFixture(),
            sourceLockFixture(
                targetConfigSHA256: String(repeating: "f", count: 64)),
        ])

        do {
            _ = try await produceQwen38ScorecardProductionRouteObservation(
                arguments: validArguments(
                    target: sandbox.target.path,
                    drafter: sandbox.drafter.path,
                    output: sandbox.output.path),
                dependencies: .test(
                    sourcePreflight: { _, _ in
                        await trace.record("source")
                        return await sources.next()
                    },
                    loadModel: { _ in
                        await trace.record("load")
                        return loaded
                    },
                    observeLoaded: { _, _ in
                        await trace.record("observe")
                        throw CLIError.observerFailed
                    },
                    shutdownLoaded: { loaded in
                        await trace.record("shutdown")
                        await loaded.backend.shutdown()
                    },
                    clearMLXCache: { await trace.record("clear") }))
            XCTFail("expected post-load source drift")
        } catch {
            XCTAssertEqual(error as? CLIError, .sourcePreflightFailed)
            XCTAssertEqual(
                qwen38ScorecardProductionRouteObservationExternalDiagnostic(error),
                "source preflight failed")
        }

        let events = await trace.snapshot()
        XCTAssertEqual(events, ["source", "load", "source", "shutdown", "clear"])
        XCTAssertFalse(FileManager.default.fileExists(atPath: sandbox.output.path))
    }

    func testPostObservationSourcePreflightDriftShutsDownClearAndEmitsNoArtifact()
        async throws
    {
        let sandbox = try Sandbox()
        let loaded = try makeRouteObservationLoadedFixture()
        let trace = CallTrace()
        let sources = SourcePreflightSequence([
            sourceLockFixture(),
            sourceLockFixture(),
            sourceLockFixture(
                drafterTensorManifestSHA256: String(repeating: "f", count: 64)),
        ])

        do {
            _ = try await produceQwen38ScorecardProductionRouteObservation(
                arguments: validArguments(
                    target: sandbox.target.path,
                    drafter: sandbox.drafter.path,
                    output: sandbox.output.path),
                dependencies: .test(
                    sourcePreflight: { _, _ in
                        await trace.record("source")
                        return await sources.next()
                    },
                    loadModel: { _ in
                        await trace.record("load")
                        return loaded
                    },
                    observeLoaded: { loaded, traceConfiguration in
                        await trace.record("observe")
                        return try await Qwen38ScorecardProductionRouteRunner
                            .observeLoaded(
                                loaded,
                                tokenTrace: traceConfiguration)
                    },
                    shutdownLoaded: { loaded in
                        await trace.record("shutdown")
                        await loaded.backend.shutdown()
                    },
                    clearMLXCache: { await trace.record("clear") }))
            XCTFail("expected post-observation source drift")
        } catch {
            XCTAssertEqual(error as? CLIError, .sourcePreflightFailed)
            XCTAssertEqual(
                qwen38ScorecardProductionRouteObservationExternalDiagnostic(error),
                "source preflight failed")
        }

        let events = await trace.snapshot()
        XCTAssertEqual(events, [
            "source",
            "load",
            "source",
            "observe",
            "source",
            "shutdown",
            "clear",
        ])
        XCTAssertFalse(FileManager.default.fileExists(atPath: sandbox.output.path))
    }

    func testUnsafePostRunFactsShutDownClearAndEmitNoArtifact() async throws {
        let sandbox = try Sandbox()
        let loaded = try makeRouteObservationLoadedFixture()
        let trace = CallTrace()
        let metal = MetalTrace([
            Qwen38ScorecardProductionRouteMetalMemorySnapshot(
                activeMetalBytes: 10,
                cachedMetalBytes: 10,
                peakMetalBytes: 10),
            Qwen38ScorecardProductionRouteMetalMemorySnapshot(
                activeMetalBytes: 10 + 65_536 + 1,
                cachedMetalBytes: 10,
                peakMetalBytes: 12),
        ])

        do {
            _ = try await produceQwen38ScorecardProductionRouteObservation(
                arguments: validArguments(
                    target: sandbox.target.path,
                    drafter: sandbox.drafter.path,
                    output: sandbox.output.path),
                dependencies: .test(
                    sourcePreflight: { _, _ in sourceLockFixture() },
                    loadModel: { _ in loaded },
                    shutdownLoaded: { loaded in
                        await trace.record("shutdown")
                        await loaded.backend.shutdown()
                    },
                    clearMLXCache: { await trace.record("clear") },
                    cleanupSettle: { await trace.record("settle") },
                    metalMemorySnapshot: { await metal.next() }))
            XCTFail("expected post-run safety failure")
        } catch {
            XCTAssertEqual(error as? CLIError, .postRunActiveMemoryResidual)
        }

        let postRunEvents = await trace.snapshot()
        XCTAssertEqual(postRunEvents, [
            "shutdown",
            "clear", "settle",
            "clear", "settle",
            "clear", "settle",
            "clear", "settle",
            "clear",
        ])
        XCTAssertFalse(FileManager.default.fileExists(atPath: sandbox.output.path))
    }

    func testTransientPostRunMetalResidualSettlesBeforeArtifactPublication()
        async throws
    {
        let sandbox = try Sandbox()
        let loaded = try makeRouteObservationLoadedFixture()
        let trace = CallTrace()
        let metal = MetalTrace([
            Qwen38ScorecardProductionRouteMetalMemorySnapshot(
                activeMetalBytes: 10,
                cachedMetalBytes: 10,
                peakMetalBytes: 10),
            Qwen38ScorecardProductionRouteMetalMemorySnapshot(
                activeMetalBytes: 10 + 65_536 + 1,
                cachedMetalBytes: 10,
                peakMetalBytes: 12),
            Qwen38ScorecardProductionRouteMetalMemorySnapshot(
                activeMetalBytes: 10,
                cachedMetalBytes: 10,
                peakMetalBytes: 12),
        ])

        _ = try await produceQwen38ScorecardProductionRouteObservation(
            arguments: validArguments(
                target: sandbox.target.path,
                drafter: sandbox.drafter.path,
                output: sandbox.output.path),
            dependencies: .test(
                sourcePreflight: { _, _ in sourceLockFixture() },
                loadModel: { _ in loaded },
                shutdownLoaded: { loaded in await loaded.backend.shutdown() },
                clearMLXCache: { await trace.record("clear") },
                cleanupSettle: { await trace.record("settle") },
                metalMemorySnapshot: { await metal.next() }))

        let events = await trace.snapshot()
        XCTAssertEqual(events, ["clear", "settle", "clear"])
        XCTAssertTrue(FileManager.default.fileExists(atPath: sandbox.output.path))
    }

    func testPostRunActiveAndCachedResidualWithinDocumentedTolerancePublishes()
        async throws
    {
        // Reviewed exception 2026-08-31: mlx-swift 0.31.6 compiled-trace
        // teardown leaks kilobyte-scale constant-descriptor buffers per
        // load+serve cycle even with weights passed as compile state, so the
        // post-run gate admits a documented absolute residual per Metal
        // metric. Ratcheted to 64 KiB after the first passing 27B observation
        // measured an 11,274-byte floor (max(64 KiB, 4x floor) per the
        // reviewed ruling). Exactly the bound must pass.
        let sandbox = try Sandbox()
        let loaded = try makeRouteObservationLoadedFixture()
        let trace = CallTrace()
        let metal = MetalTrace([
            Qwen38ScorecardProductionRouteMetalMemorySnapshot(
                activeMetalBytes: 1_000,
                cachedMetalBytes: 1_000,
                peakMetalBytes: 1_000),
            Qwen38ScorecardProductionRouteMetalMemorySnapshot(
                activeMetalBytes: 1_000 + 65_536,
                cachedMetalBytes: 1_000 + 65_536,
                peakMetalBytes: 6_000_000),
        ])

        _ = try await produceQwen38ScorecardProductionRouteObservation(
            arguments: validArguments(
                target: sandbox.target.path,
                drafter: sandbox.drafter.path,
                output: sandbox.output.path),
            dependencies: .test(
                sourcePreflight: { _, _ in sourceLockFixture() },
                loadModel: { _ in loaded },
                shutdownLoaded: { loaded in await loaded.backend.shutdown() },
                clearMLXCache: { await trace.record("clear") },
                cleanupSettle: { await trace.record("settle") },
                metalMemorySnapshot: { await metal.next() }))

        let events = await trace.snapshot()
        XCTAssertEqual(events, ["clear"])
        XCTAssertTrue(FileManager.default.fileExists(atPath: sandbox.output.path))
    }

    func testPostRunCurrentAllocatedResidualWithinDocumentedTolerancePublishes()
        async throws
    {
        let sandbox = try Sandbox()
        let loaded = try makeRouteObservationLoadedFixture()
        let trace = CallTrace()
        let baseline = hostSnapshot()
        let hosts = HostSnapshotTrace([
            baseline,
            hostSnapshot(
                metalCurrentAllocatedSizeBytes:
                    baseline.metalCurrentAllocatedSizeBytes + 65_536),
        ])

        _ = try await produceQwen38ScorecardProductionRouteObservation(
            arguments: validArguments(
                target: sandbox.target.path,
                drafter: sandbox.drafter.path,
                output: sandbox.output.path),
            dependencies: .test(
                hostSnapshot: { await hosts.next() },
                sourcePreflight: { _, _ in sourceLockFixture() },
                loadModel: { _ in loaded },
                shutdownLoaded: { loaded in await loaded.backend.shutdown() },
                clearMLXCache: { await trace.record("clear") },
                cleanupSettle: { await trace.record("settle") }))

        let events = await trace.snapshot()
        XCTAssertEqual(events, ["clear"])
        XCTAssertTrue(FileManager.default.fileExists(atPath: sandbox.output.path))
    }

    func testResidualToleranceExceptionExpiresOnMLXSwiftUpgrade() throws {
        // The 64 KiB residual tolerance is a reviewed exception verified
        // against mlx-swift 0.31.6's compiled-trace teardown defect. Any
        // upgrade must first re-verify the strict zero-residual gate (and
        // restore it if upstream fixed the orphaned-cycle teardown); this
        // tripwire makes the expiry non-forgettable.
        let testsDirectory = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
        let packageResolved = testsDirectory
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Package.resolved")
        let data = try Data(contentsOf: packageResolved)
        let root = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let pins = try XCTUnwrap(root["pins"] as? [[String: Any]])
        let mlxSwift = try XCTUnwrap(
            pins.first { ($0["identity"] as? String) == "mlx-swift" })
        let state = try XCTUnwrap(mlxSwift["state"] as? [String: Any])
        XCTAssertEqual(
            state["version"] as? String,
            "0.31.6",
            """
            mlx-swift changed: the 64 KiB post-run residual tolerance in \
            Qwen38ScorecardProductionRouteObservationCLI was verified against \
            0.31.6 only. Re-verify the strict zero-residual settle gate on the \
            new version, restore it if the upstream orphaned-cycle teardown \
            defect is fixed, and only then update this pin assertion. See \
            docs/task-inbox/2026-08-31-teardown-compiled-state-followups.md.
            """)
    }

    func testPostRunReadbackRetriesCurrentAllocationWithoutDoubleCountingPlan()
        async throws
    {
        let sandbox = try Sandbox()
        let loaded = try makeRouteObservationLoadedFixture()
        let trace = CallTrace()
        let hosts = HostSnapshotTrace([
            hostSnapshot(),
            hostSnapshot(metalCurrentAllocatedSizeBytes: 40_000_000_000),
            hostSnapshot(),
        ])

        _ = try await produceQwen38ScorecardProductionRouteObservation(
            arguments: validArguments(
                target: sandbox.target.path,
                drafter: sandbox.drafter.path,
                output: sandbox.output.path),
            dependencies: .test(
                hostSnapshot: { await hosts.next() },
                sourcePreflight: { _, _ in sourceLockFixture() },
                loadModel: { _ in loaded },
                shutdownLoaded: { loaded in await loaded.backend.shutdown() },
                clearMLXCache: { await trace.record("clear") },
                cleanupSettle: { await trace.record("settle") }))

        let events = await trace.snapshot()
        XCTAssertEqual(events, ["clear", "settle", "clear"])
        XCTAssertTrue(FileManager.default.fileExists(atPath: sandbox.output.path))
    }

    func testPersistentCurrentAllocationEmitsBoundedAttemptTelemetryAndNoArtifact()
        async throws
    {
        let sandbox = try Sandbox()
        let loaded = try makeRouteObservationLoadedFixture()
        let hosts = HostSnapshotTrace([
            hostSnapshot(),
            hostSnapshot(metalCurrentAllocatedSizeBytes: 40_000_000_000),
        ])
        let monotonic = UInt64Trace([100, 200, 300, 400, 500, 600])
        let telemetry = CleanupAttemptTrace()

        do {
            _ = try await produceQwen38ScorecardProductionRouteObservation(
                arguments: validArguments(
                    target: sandbox.target.path,
                    drafter: sandbox.drafter.path,
                    output: sandbox.output.path),
                dependencies: .test(
                    hostSnapshot: { await hosts.next() },
                    sourcePreflight: { _, _ in sourceLockFixture() },
                    loadModel: { _ in loaded },
                    shutdownLoaded: { loaded in await loaded.backend.shutdown() },
                    cleanupMonotonicNanoseconds: { await monotonic.next() },
                    cleanupAttemptTelemetry: { observation in
                        await telemetry.record(observation)
                    }))
            XCTFail("expected current-allocation cleanup failure")
        } catch {
            XCTAssertEqual(
                error as? CLIError,
                .postRunCurrentAllocatedMemoryResidual)
        }

        let observations = await telemetry.snapshot()
        XCTAssertEqual(observations.count, 5)
        XCTAssertEqual(observations.first?.kind,
            "fast-mlx.production-route-cleanup-attempt.v1")
        XCTAssertEqual(observations.first?.attempt, 1)
        XCTAssertEqual(observations.first?.attemptLimit, 5)
        XCTAssertEqual(observations.first?.elapsedNanoseconds, 100)
        XCTAssertEqual(
            observations.first?.before.metalCurrentAllocatedSizeBytes,
            1)
        XCTAssertEqual(
            observations.first?.observed.metalCurrentAllocatedSizeBytes,
            40_000_000_000)
        XCTAssertEqual(observations.first?.before.physicalFootprintBytes, 5)
        XCTAssertEqual(observations.first?.observed.physicalFootprintBytes, 5)
        XCTAssertFalse(observations.first?.currentAllocatedAtOrBelowBaseline ?? true)
        XCTAssertTrue(observations.first?.activeAtOrBelowBaseline ?? false)
        XCTAssertTrue(observations.first?.cachedAtOrBelowBaseline ?? false)
        XCTAssertFalse(observations.first?.residualAdmissionSafe ?? true)
        XCTAssertEqual(observations.last?.attempt, 5)
        XCTAssertEqual(observations.last?.elapsedNanoseconds, 500)
        XCTAssertFalse(FileManager.default.fileExists(atPath: sandbox.output.path))
    }

    func testPersistentActiveAndCurrentResidualReportsActiveMemoryFirst()
        async throws
    {
        let sandbox = try Sandbox()
        let loaded = try makeRouteObservationLoadedFixture()
        let hosts = HostSnapshotTrace([
            hostSnapshot(),
            hostSnapshot(metalCurrentAllocatedSizeBytes: 40_000_000_000),
        ])
        let metal = MetalTrace([
            Qwen38ScorecardProductionRouteMetalMemorySnapshot(
                activeMetalBytes: 10,
                cachedMetalBytes: 10,
                peakMetalBytes: 10),
            Qwen38ScorecardProductionRouteMetalMemorySnapshot(
                activeMetalBytes: 10 + 65_536 + 1,
                cachedMetalBytes: 10,
                peakMetalBytes: 12),
        ])

        do {
            _ = try await produceQwen38ScorecardProductionRouteObservation(
                arguments: validArguments(
                    target: sandbox.target.path,
                    drafter: sandbox.drafter.path,
                    output: sandbox.output.path),
                dependencies: .test(
                    hostSnapshot: { await hosts.next() },
                    sourcePreflight: { _, _ in sourceLockFixture() },
                    loadModel: { _ in loaded },
                    shutdownLoaded: { loaded in await loaded.backend.shutdown() },
                    metalMemorySnapshot: { await metal.next() }))
            XCTFail("expected active-memory cleanup failure")
        } catch {
            XCTAssertEqual(error as? CLIError, .postRunActiveMemoryResidual)
        }

        XCTAssertFalse(FileManager.default.fileExists(atPath: sandbox.output.path))
    }

    func testCleanupSettleCancellationSuppressesArtifact() async throws {
        let sandbox = try Sandbox()
        let loaded = try makeRouteObservationLoadedFixture()
        let metal = MetalTrace([
            Qwen38ScorecardProductionRouteMetalMemorySnapshot(
                activeMetalBytes: 10,
                cachedMetalBytes: 10,
                peakMetalBytes: 10),
            Qwen38ScorecardProductionRouteMetalMemorySnapshot(
                activeMetalBytes: 10 + 65_536 + 1,
                cachedMetalBytes: 10,
                peakMetalBytes: 12),
        ])

        do {
            _ = try await produceQwen38ScorecardProductionRouteObservation(
                arguments: validArguments(
                    target: sandbox.target.path,
                    drafter: sandbox.drafter.path,
                    output: sandbox.output.path),
                dependencies: .test(
                    sourcePreflight: { _, _ in sourceLockFixture() },
                    loadModel: { _ in loaded },
                    shutdownLoaded: { loaded in await loaded.backend.shutdown() },
                    cleanupSettle: { throw CancellationError() },
                    metalMemorySnapshot: { await metal.next() }))
            XCTFail("expected cleanup cancellation")
        } catch is CancellationError {
            // expected
        }

        XCTAssertFalse(FileManager.default.fileExists(atPath: sandbox.output.path))
    }

    func testPostRunHostReadbackCancellationSuppressesArtifact() async throws {
        let sandbox = try Sandbox()
        let loaded = try makeRouteObservationLoadedFixture()
        let hosts = PostRunCancellingHostSnapshotTrace()

        do {
            _ = try await produceQwen38ScorecardProductionRouteObservation(
                arguments: validArguments(
                    target: sandbox.target.path,
                    drafter: sandbox.drafter.path,
                    output: sandbox.output.path),
                dependencies: .test(
                    hostSnapshot: { try await hosts.next() },
                    sourcePreflight: { _, _ in sourceLockFixture() },
                    loadModel: { _ in loaded },
                    shutdownLoaded: { loaded in await loaded.backend.shutdown() }))
            XCTFail("expected post-run host cancellation")
        } catch is CancellationError {
            // expected
        }

        XCTAssertFalse(FileManager.default.fileExists(atPath: sandbox.output.path))
    }

    func testPostRunWiredLimitDriftFailsWithoutCleanupRetry() async throws {
        let sandbox = try Sandbox()
        let loaded = try makeRouteObservationLoadedFixture()
        let hosts = HostSnapshotTrace([
            hostSnapshot(),
            hostSnapshot(wiredLimitMB: 262_143),
        ])
        let trace = CallTrace()

        do {
            _ = try await produceQwen38ScorecardProductionRouteObservation(
                arguments: validArguments(
                    target: sandbox.target.path,
                    drafter: sandbox.drafter.path,
                    output: sandbox.output.path),
                dependencies: .test(
                    hostSnapshot: { await hosts.next() },
                    sourcePreflight: { _, _ in sourceLockFixture() },
                    loadModel: { _ in loaded },
                    shutdownLoaded: { loaded in await loaded.backend.shutdown() },
                    clearMLXCache: { await trace.record("clear") },
                    cleanupSettle: { await trace.record("settle") }))
            XCTFail("expected wired-limit drift failure")
        } catch {
            XCTAssertEqual(error as? CLIError, .postRunWiredLimitDrift)
        }

        let events = await trace.snapshot()
        XCTAssertEqual(events, ["clear"])
        XCTAssertFalse(FileManager.default.fileExists(atPath: sandbox.output.path))
    }

    func testPostRunProcessMemoryReadbackFailureSuppressesArtifact()
        async throws
    {
        let sandbox = try Sandbox()
        let loaded = try makeRouteObservationLoadedFixture()
        let footprints = UInt64Trace([5, 0])
        let telemetry = CleanupAttemptTrace()

        do {
            _ = try await produceQwen38ScorecardProductionRouteObservation(
                arguments: validArguments(
                    target: sandbox.target.path,
                    drafter: sandbox.drafter.path,
                    output: sandbox.output.path),
                dependencies: .test(
                    sourcePreflight: { _, _ in sourceLockFixture() },
                    loadModel: { _ in loaded },
                    shutdownLoaded: { loaded in await loaded.backend.shutdown() },
                    cleanupAttemptTelemetry: { observation in
                        await telemetry.record(observation)
                    },
                    physicalFootprintBytes: { await footprints.next() }))
            XCTFail("expected process-memory readback failure")
        } catch {
            XCTAssertEqual(
                error as? CLIError,
                .postRunProcessMemoryReadbackFailed)
        }

        let observations = await telemetry.snapshot()
        XCTAssertEqual(observations.count, 1)
        XCTAssertFalse(observations[0].processMemoryReadbackValid)
        XCTAssertFalse(FileManager.default.fileExists(atPath: sandbox.output.path))
    }

    func testPersistentPostRunCachedMemoryFailsClosed() async throws {
        let sandbox = try Sandbox()
        let loaded = try makeRouteObservationLoadedFixture()
        let metal = MetalTrace([
            Qwen38ScorecardProductionRouteMetalMemorySnapshot(
                activeMetalBytes: 10,
                cachedMetalBytes: 10,
                peakMetalBytes: 10),
            Qwen38ScorecardProductionRouteMetalMemorySnapshot(
                activeMetalBytes: 10,
                cachedMetalBytes: 10 + 65_536 + 1,
                peakMetalBytes: 12),
        ])

        do {
            _ = try await produceQwen38ScorecardProductionRouteObservation(
                arguments: validArguments(
                    target: sandbox.target.path,
                    drafter: sandbox.drafter.path,
                    output: sandbox.output.path),
                dependencies: .test(
                    sourcePreflight: { _, _ in sourceLockFixture() },
                    loadModel: { _ in loaded },
                    shutdownLoaded: { loaded in await loaded.backend.shutdown() },
                    metalMemorySnapshot: { await metal.next() }))
            XCTFail("expected cached-memory cleanup failure")
        } catch {
            XCTAssertEqual(error as? CLIError, .postRunCachedMemoryResidual)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: sandbox.output.path))
    }

    func testPostRunUnsafeThermalStateFailsWithoutCleanupRetry() async throws {
        let sandbox = try Sandbox()
        let loaded = try makeRouteObservationLoadedFixture()
        let thermal = StringTrace(["nominal", "serious"])
        let trace = CallTrace()

        do {
            _ = try await produceQwen38ScorecardProductionRouteObservation(
                arguments: validArguments(
                    target: sandbox.target.path,
                    drafter: sandbox.drafter.path,
                    output: sandbox.output.path),
                dependencies: .test(
                    sourcePreflight: { _, _ in sourceLockFixture() },
                    loadModel: { _ in loaded },
                    shutdownLoaded: { loaded in await loaded.backend.shutdown() },
                    clearMLXCache: { await trace.record("clear") },
                    cleanupSettle: { await trace.record("settle") },
                    thermalState: { await thermal.next() }))
            XCTFail("expected thermal cleanup failure")
        } catch {
            XCTAssertEqual(error as? CLIError, .postRunThermalUnsafe)
        }

        let events = await trace.snapshot()
        XCTAssertEqual(events, ["clear"])
        XCTAssertFalse(FileManager.default.fileExists(atPath: sandbox.output.path))
    }

    func testLoadedBackendAndRuntimeAreReleasedBeforeMLXCacheClear()
        async throws
    {
        let sandbox = try Sandbox()
        let backendLifetime =
            WeakRouteObservationObjectProbe<ContinuousServingBackend>()
        let runtimeLifetime =
            WeakRouteObservationObjectProbe<RouteObservationFixtureRuntime>()
        let trace = CallTrace()

        _ = try await produceQwen38ScorecardProductionRouteObservation(
            arguments: validArguments(
                target: sandbox.target.path,
                drafter: sandbox.drafter.path,
                output: sandbox.output.path),
            dependencies: .test(
                sourcePreflight: { _, _ in sourceLockFixture() },
                loadModel: { _ in
                    let runtime = RouteObservationFixtureRuntime()
                    runtimeLifetime.capture(runtime)
                    let loaded = try makeRouteObservationLoadedFixture(
                        runtime: runtime)
                    backendLifetime.capture(loaded.backend)
                    return loaded
                },
                shutdownLoaded: { loaded in
                    await loaded.backend.shutdown()
                },
                clearMLXCache: {
                    let allReleased = backendLifetime.isReleased
                        && runtimeLifetime.isReleased
                    await trace.record(
                        allReleased ? "released" : "retained")
                }))

        let events = await trace.snapshot()
        XCTAssertEqual(events, ["released"])
    }

    func testStartupConfigMismatchShutsDownClearAndEmitsNoArtifact()
        async throws
    {
        let sandbox = try Sandbox()
        let loaded = try makeRouteObservationLoadedFixture()
        let trace = CallTrace()
        let source = Qwen38ScorecardProductionRouteSourceLockObservation(
            knownArtifact: "qwen38-27b-mxfp8-depth1",
            selection: "qwen38-27b-mxfp8-depth1",
            lockSourceRevision: "source-revision",
            targetModelID: "target-model",
            targetRevision: "target-revision",
            targetConfigSHA256: String(repeating: "b", count: 64),
            targetTokenizerSHA256: String(repeating: "b", count: 64),
            targetTensorManifestSHA256: String(repeating: "c", count: 64),
            drafterModelID: "drafter-model",
            drafterRevision: "drafter-revision",
            drafterConfigSHA256: String(repeating: "d", count: 64),
            drafterTokenizerSHA256: String(repeating: "b", count: 64),
            drafterTensorManifestSHA256: String(repeating: "e", count: 64),
            targetQuantizationBits: 8,
            targetQuantizationGroupSize: 32,
            targetQuantizationMode: "mxfp8",
            drafterQuantizationBits: 8,
            drafterQuantizationGroupSize: 32,
            drafterQuantizationMode: "mxfp8",
            runtimeBlockSize: 3,
            maximumAcceptedDraftTokens: 2)

        do {
            _ = try await produceQwen38ScorecardProductionRouteObservation(
                arguments: validArguments(
                    target: sandbox.target.path,
                    drafter: sandbox.drafter.path,
                    output: sandbox.output.path),
                dependencies: .test(
                    sourcePreflight: { _, _ in source },
                    loadModel: { _ in loaded },
                    observeLoaded: { _, _ in
                        await trace.record("observe")
                        throw CLIError.observerFailed
                    },
                    shutdownLoaded: { loaded in
                        await trace.record("shutdown")
                        await loaded.backend.shutdown()
                    },
                    clearMLXCache: { await trace.record("clear") }))
            XCTFail("expected startup mismatch")
        } catch {
            XCTAssertEqual(error as? CLIError, .invalidStartupReport)
        }

        let startupMismatchEvents = await trace.snapshot()
        XCTAssertEqual(startupMismatchEvents, ["shutdown", "clear"])
        XCTAssertFalse(FileManager.default.fileExists(atPath: sandbox.output.path))
    }

    func testObserverErrorShutsDownLoadedModelAndClearsCacheWithoutOutput()
        async throws
    {
        let sandbox = try Sandbox()
        let loaded = try makeRouteObservationLoadedFixture()
        let trace = CallTrace()

        do {
            _ = try await produceQwen38ScorecardProductionRouteObservation(
                arguments: validArguments(
                    target: sandbox.target.path,
                    drafter: sandbox.drafter.path,
                    output: sandbox.output.path),
                dependencies: .test(
                    sourcePreflight: { _, _ in sourceLockFixture() },
                    loadModel: { _ in loaded },
                    observeLoaded: { _, _ in throw CLIError.observerFailed },
                    shutdownLoaded: { loaded in
                        await trace.record("shutdown")
                        await loaded.backend.shutdown()
                    },
                    clearMLXCache: {
                        await trace.record("clear")
                    }))
            XCTFail("expected observer failure")
        } catch {
            XCTAssertEqual(error as? CLIError, .observerFailed)
        }

        let events = await trace.snapshot()
        XCTAssertEqual(events, ["shutdown", "clear"])
        XCTAssertFalse(FileManager.default.fileExists(atPath: sandbox.output.path))
    }

    private func validArguments(
        target: String = "/tmp/target",
        drafter: String = "/tmp/drafter",
        output: String = "/tmp/observation.json",
        replacing flag: String? = nil,
        with replacement: String = "",
        removing removed: String? = nil
    ) -> [String] {
        var arguments = [
            "--target", target,
            "--drafter", drafter,
            "--output", output,
            "--host-use", "dedicated-serving",
            "--host-use-source", "operator-assertion",
            "--expected-chip", "Apple M3 Ultra",
            "--memory-limit-bytes", "180000000000",
            "--cache-limit-bytes", "10000000000",
            "--reserved-kv-bytes", "20000000000",
            "--reserved-io-bytes", "1000000000",
            "--reserved-prefetch-bytes", "1000000000",
            "--os-service-reserve-bytes", "1000000000",
            "--context-tokens", "32768",
        ]
        if let flag, let index = arguments.firstIndex(of: flag) {
            arguments[index + 1] = replacement
        }
        if let removed, let index = arguments.firstIndex(of: removed) {
            arguments.removeSubrange(index ... index + 1)
        }
        return arguments
    }
}

private actor CallTrace {
    private var storage: [String] = []

    func record(_ event: String) {
        storage.append(event)
    }

    func snapshot() -> [String] {
        storage
    }
}

private actor MetalTrace {
    private var snapshots:
        [Qwen38ScorecardProductionRouteMetalMemorySnapshot]

    init(_ snapshots: [Qwen38ScorecardProductionRouteMetalMemorySnapshot]) {
        self.snapshots = snapshots
    }

    func next() -> Qwen38ScorecardProductionRouteMetalMemorySnapshot {
        precondition(!snapshots.isEmpty)
        guard snapshots.count > 1 else { return snapshots[0] }
        return snapshots.removeFirst()
    }
}

private actor HostSnapshotTrace {
    private var snapshots: [Qwen38MTPScorecardLiveHostMemorySnapshot]

    init(_ snapshots: [Qwen38MTPScorecardLiveHostMemorySnapshot]) {
        self.snapshots = snapshots
    }

    func next() -> Qwen38MTPScorecardLiveHostMemorySnapshot {
        precondition(!snapshots.isEmpty)
        guard snapshots.count > 1 else { return snapshots[0] }
        return snapshots.removeFirst()
    }
}

private actor PostRunCancellingHostSnapshotTrace {
    private var calls = 0

    func next() throws -> Qwen38MTPScorecardLiveHostMemorySnapshot {
        calls += 1
        guard calls == 1 else { throw CancellationError() }
        return hostSnapshot()
    }
}

private actor StringTrace {
    private var values: [String]

    init(_ values: [String]) {
        self.values = values
    }

    func next() -> String {
        precondition(!values.isEmpty)
        guard values.count > 1 else { return values[0] }
        return values.removeFirst()
    }
}

private actor UInt64Trace {
    private var values: [UInt64]

    init(_ values: [UInt64]) {
        self.values = values
    }

    func next() -> UInt64 {
        precondition(!values.isEmpty)
        guard values.count > 1 else { return values[0] }
        return values.removeFirst()
    }
}

private actor CleanupAttemptTrace {
    private var observations:
        [Qwen38ScorecardProductionRouteCleanupAttemptObservation] = []

    func record(
        _ observation: Qwen38ScorecardProductionRouteCleanupAttemptObservation
    ) {
        observations.append(observation)
    }

    func snapshot()
        -> [Qwen38ScorecardProductionRouteCleanupAttemptObservation]
    {
        observations
    }
}

private actor SourcePreflightSequence {
    private var observations:
        [Qwen38ScorecardProductionRouteSourceLockObservation]

    init(_ observations: [Qwen38ScorecardProductionRouteSourceLockObservation]) {
        self.observations = observations
    }

    func next() -> Qwen38ScorecardProductionRouteSourceLockObservation {
        precondition(!observations.isEmpty)
        return observations.removeFirst()
    }
}

private struct Sandbox {
    let root: URL
    let target: URL
    let drafter: URL
    let output: URL

    init() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen38-route-observation-\(UUID().uuidString)")
        target = root.appendingPathComponent("target", isDirectory: true)
        drafter = root.appendingPathComponent("drafter", isDirectory: true)
        output = root.appendingPathComponent("observation.json")
        try FileManager.default.createDirectory(
            at: target,
            withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: drafter,
            withIntermediateDirectories: true)
    }
}

private func hostSnapshot(
    wiredLimitMB: UInt64 = 262_144,
    metalCurrentAllocatedSizeBytes: UInt64 = 1
) -> Qwen38MTPScorecardLiveHostMemorySnapshot {
    Qwen38MTPScorecardLiveHostMemorySnapshot(
        physicalRAMBytes: 274_877_906_944,
        chipName: "Apple M3 Ultra",
        wiredLimitMB: wiredLimitMB,
        metalRecommendedMaxWorkingSetSizeBytes: 220_000_000_000,
        metalCurrentAllocatedSizeBytes: metalCurrentAllocatedSizeBytes)
}

private func sourceLockFixture(
    targetConfigSHA256: String = String(repeating: "a", count: 64),
    targetTensorManifestSHA256: String = String(repeating: "c", count: 64),
    drafterTensorManifestSHA256: String = String(repeating: "e", count: 64)
)
    -> Qwen38ScorecardProductionRouteSourceLockObservation
{
    Qwen38ScorecardProductionRouteSourceLockObservation(
        knownArtifact: "qwen38-27b-mxfp8-depth1",
        selection: "qwen38-27b-mxfp8-depth1",
        lockSourceRevision: "source-revision",
        targetModelID: "target-model",
        targetRevision: "target-revision",
        targetConfigSHA256: targetConfigSHA256,
        targetTokenizerSHA256: String(repeating: "b", count: 64),
        targetTensorManifestSHA256: targetTensorManifestSHA256,
        drafterModelID: "drafter-model",
        drafterRevision: "drafter-revision",
        drafterConfigSHA256: String(repeating: "d", count: 64),
        drafterTokenizerSHA256: String(repeating: "b", count: 64),
        drafterTensorManifestSHA256: drafterTensorManifestSHA256,
        targetQuantizationBits: 8,
        targetQuantizationGroupSize: 32,
        targetQuantizationMode: "mxfp8",
        drafterQuantizationBits: 8,
        drafterQuantizationGroupSize: 32,
        drafterQuantizationMode: "mxfp8",
        runtimeBlockSize: 3,
        maximumAcceptedDraftTokens: 2)
}

private func makeRouteObservationLoadedFixture(
    runtime: RouteObservationFixtureRuntime = RouteObservationFixtureRuntime()
)
    throws -> LoadedContinuousServingModel
{
    let coordinator = ContinuousBatchCoordinator(
        configuration: try ContinuousBatchConfiguration(
            maxActiveSlots: 4,
            maxPrefillSlots: 4,
            prefillChunkSize: 8,
            maxQueuedRequests: 4),
        runtime: runtime,
        automaticDrive: true,
        publicationCapacity: 8,
        traceLimit: 128)
    let backend = ContinuousServingBackend(
        launchedModel: Qwen38MTPPerformanceScorecardGate.modelArtifactLabel,
        coordinator: coordinator,
        codec: RouteObservationFixtureCodec(),
        stopTokenIDs: [90_000],
        modelStopStrings: [],
        configuration: ContinuousServingBackendConfiguration(
            defaultMaximumCompletionTokens: 8,
            queueRetryAfterSeconds: 2,
            mailboxCapacity: .init(maxDeltas: 8, maxBytes: 4_096),
            scorecardTraceCapacity: 128,
            admission: .dynamic(
                configuration: ServingAdmissionConfiguration(
                    soloPLDQualified: false,
                    maximumBatchRequests: 4,
                    maximumQueuedRequests: 4),
                coalescing: .automatic(.milliseconds(1)))))
    return LoadedContinuousServingModel
        .testingLoadedContinuousServingModelWithLoaderProvenance(
            backend: backend,
            startupReport: ContinuousServingModelStartupReport(
                launchedModel: Qwen38MTPPerformanceScorecardGate.modelArtifactLabel,
                route: .continuousBatchNoSpec,
                memoryLimitBytes: 180_000_000_000,
                cacheLimitBytes: 10_000_000_000,
                maxReservedKVBytes: 20_000_000_000,
                maxContextTokens: 32_768,
                maxReservedContextTokens: 32_768,
                modelFamily: .qwen35,
                modelConfigurationSHA256: String(repeating: "a", count: 64),
                layerCount: 64,
                keyValueHeadCount: 4,
                headDimension: 256,
                stopTokenCount: 1,
                stopStringCount: 0,
                nativeCacheKinds: [.denseAttention],
                startupPromptTokenCount: 3,
                startupGeneratedTokenCount: 1,
                maxActiveSlots: 4,
                maxPrefillSlots: 4,
                prefillChunkSize: 512,
                maxQueuedRequests: 4,
                publicationCapacity: 8,
                soloPLDPolicy: nil,
                modelProofVerified: true))
}

private final class WeakRouteObservationObjectProbe<Object: AnyObject>:
    @unchecked Sendable
{
    private let lock = NSLock()
    private weak var object: Object?

    var isReleased: Bool {
        lock.withLock { object == nil }
    }

    func capture(_ object: Object) {
        lock.withLock { self.object = object }
    }
}

private final class RouteObservationFixtureRuntime:
    ContinuousBatchRuntime,
    @unchecked Sendable
{
    struct Slot {
        var processedTokens = 0
        var ready = false
        var outputCursor = 0
        let outputTokens: [Int]
    }

    private var slots: [BatchRequestID: Slot] = [:]

    func admit(_ admissions: [ContinuousBatchRuntimeAdmission]) throws {
        for admission in admissions {
            guard let requestIndex = admission.submission.promptTokens.first
            else {
                throw Qwen38ScorecardContinuousRouteError.incompleteRequest(index: 0)
            }
            slots[admission.id] = Slot(outputTokens: [
                2_000 + requestIndex,
                90_000,
            ])
        }
    }

    func resourceSnapshot() -> ContinuousBatchRuntimeResourceSnapshot? {
        ContinuousBatchRuntimeResourceSnapshot(
            kvBytesPerToken: 64,
            reservedKVBytes: slots.count * 64,
            maxReservedKVBytes: 4_096)
    }

    func prefill(_ work: ContinuousBatchRuntimePrefill) throws {
        guard var slot = slots[work.id],
            work.startToken == slot.processedTokens
        else {
            throw Qwen38ScorecardContinuousRouteError.incompleteRequest(index: 0)
        }
        slot.processedTokens += work.tokens.count
        slot.ready = work.isFinal
        slots[work.id] = slot
    }

    func decode(
        _ action: BatchDecodeAction
    ) throws -> [ContinuousBatchRuntimeDecodeResult] {
        let ids: [BatchRequestID]
        switch action {
        case .batch(let requestIDs, let speculationAllowed):
            guard !speculationAllowed else {
                throw Qwen38ScorecardContinuousRouteError.speculationEnabled
            }
            ids = requestIDs
        case .drainSoloPipeline, .solo:
            throw Qwen38ScorecardContinuousRouteError.missingSharedBatchDecode
        }
        return try ids.map { id in
            guard var slot = slots[id], slot.ready else {
                throw Qwen38ScorecardContinuousRouteError.incompleteRequest(index: 0)
            }
            guard slot.outputCursor < slot.outputTokens.count else {
                return ContinuousBatchRuntimeDecodeResult(
                    id: id,
                    tokens: [],
                    finished: true,
                    soloPipelineState: .canonical)
            }
            let token = slot.outputTokens[slot.outputCursor]
            slot.outputCursor += 1
            slots[id] = slot
            return ContinuousBatchRuntimeDecodeResult(
                id: id,
                tokens: [token],
                finished: false,
                soloPipelineState: .canonical)
        }
    }

    func remove(_ id: BatchRequestID) {
        slots[id] = nil
    }
}

private struct RouteObservationFixtureCodec: ScalarServingTextCodec {
    func render(
        messages: [OpenAIChatMessage],
        tools: [OpenAIToolSpec],
        enableThinking: Bool?,
        reasoningEffort: String?
    ) throws -> [Int] {
        guard let text = messages.last?.text else {
            throw Qwen38ScorecardContinuousRouteError.incompleteRequest(index: 0)
        }
        for index in 0 ..< 4 where text.hasSuffix("\(index).") {
            return [index]
        }
        throw Qwen38ScorecardContinuousRouteError.incompleteRequest(index: 0)
    }

    func makeDetokenizer() -> any ScalarServingDetokenizer {
        RouteObservationFixtureDetokenizer()
    }
}

private struct RouteObservationFixtureDetokenizer: ScalarServingDetokenizer {
    private var pending: Int?

    mutating func append(token: Int) {
        pending = token
    }

    mutating func next() -> String? {
        defer { pending = nil }
        return pending.map(String.init)
    }
}
