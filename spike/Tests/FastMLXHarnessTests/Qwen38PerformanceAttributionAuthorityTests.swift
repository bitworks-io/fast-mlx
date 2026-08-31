import Foundation
import Darwin
import XCTest

import HarnessCore

@testable import fastmlx_harness

final class Qwen38PerformanceAttributionAuthorityTests: XCTestCase {
    private typealias Loader = Qwen38PerformanceAttributionAuthorityLoader
    private typealias Error = Qwen38PerformanceAttributionAuthorityError
    private typealias Gate = Qwen38PerformanceAttributionScorecardGate

    func testLoadsAuthorityOnlyFromSeparateBodyAndExternalPinChannels() throws {
        let fixture = makeFixture()
        let files = try writeFixture(fixture)
        defer { try? FileManager.default.removeItem(at: files.directory) }

        let captured = try Loader.load(files: files.inputs)

        XCTAssertEqual(
            captured.policyDocumentSHA256,
            Loader.sha256Hex(fixture.bodies.policy))
        XCTAssertEqual(
            captured.policyPinDocumentSHA256,
            Loader.sha256Hex(Data(fixture.policy.digest.utf8)))
        XCTAssertEqual(
            captured.productionRouteReceiptDocumentSHA256,
            Loader.sha256Hex(fixture.bodies.productionRouteReceipt))
        XCTAssertEqual(
            captured.productionRouteReceiptPinDocumentSHA256,
            Loader.sha256Hex(Data(fixture.receipt.digest.utf8)))
        XCTAssertFalse(captured.promotionAuthorized)
        XCTAssertTrue(captured.requiresIndependentControllerSignature)
        XCTAssertNotEqual(captured.policyBodyIdentity, captured.policyPinIdentity)
        XCTAssertNotEqual(
            captured.productionRouteReceiptBodyIdentity,
            captured.productionRouteReceiptPinIdentity)
    }

    func testHappyPathPolicyCoversEveryCorrectedScorecardBand() throws {
        let fixture = makeFixture()
        let expected: [Qwen38PerformanceAttributionClaimKind: Set<String>] = [
            .scalarGDN: ["1:4096:cold"],
            .exactMTP: ["1:4096:cold"],
            .continuousBatchNoSpec: ["2:4096:cold", "4:4096:cold"],
            .prefixMatrix: [
                "1:4096:cold",
                "1:4096:exactWarmPrefix",
                "1:16384:cold",
                "1:16384:exactWarmPrefix",
                "1:32768:cold",
                "1:32768:exactWarmPrefix",
            ],
        ]

        XCTAssertEqual(fixture.policy.claimAuthorities.count, expected.count)
        for authority in fixture.policy.claimAuthorities {
            XCTAssertEqual(
                Set(authority.absoluteAuthority.bands.map(bandKey)),
                expected[authority.claimKind])
        }

        let files = try writeFixture(fixture)
        defer { try? FileManager.default.removeItem(at: files.directory) }
        XCTAssertNoThrow(try Loader.load(files: files.inputs))
    }

    func testRejectsEmbeddedOrSelfIssuedPinsWithoutLeakingCallerText() throws {
        let fixture = makeFixture()
        let files = try writeFixture(fixture)
        defer { try? FileManager.default.removeItem(at: files.directory) }

        let embeddedPin = files.directory.appendingPathComponent("embedded-pin.json")
        try fixture.bodies.policy.write(to: embeddedPin)

        XCTAssertThrowsError(try Loader.load(files: .init(
            policyBody: files.inputs.policyBody,
            policyPin: embeddedPin,
            productionRouteReceiptBody: files.inputs.productionRouteReceiptBody,
            productionRouteReceiptPin: files.inputs.productionRouteReceiptPin))
        ) { error in
            XCTAssertEqual(error as? Error, .oversizedAuthorityInput(.policy))
            XCTAssertFalse(String(describing: error).contains(fixture.policy.digest))
        }

        try Data(Loader.sha256Hex(fixture.bodies.policy).utf8).write(
            to: files.inputs.policyPin)
        XCTAssertThrowsError(try Loader.load(files: files.inputs)) { error in
            XCTAssertEqual(error as? Error, .bodyHashPin(.policy))
        }
    }

    func testCanonicalBodyDigestPinsCaptureButDoNotAuthorizePromotion() throws {
        let fixture = makeFixture()
        let files = try writeFixture(fixture)
        defer { try? FileManager.default.removeItem(at: files.directory) }

        let captured = try Loader.load(files: files.inputs)

        XCTAssertEqual(
            captured.policyDocumentSHA256,
            Loader.sha256Hex(fixture.bodies.policy))
        XCTAssertEqual(
            captured.productionRouteReceiptDocumentSHA256,
            Loader.sha256Hex(fixture.bodies.productionRouteReceipt))
        XCTAssertFalse(captured.promotionAuthorized)
        XCTAssertTrue(captured.requiresIndependentControllerSignature)
    }

    func testRejectsMalformedAndMismatchedPinsWithoutLeakingCallerText() throws {
        let fixture = makeFixture()
        let files = try writeFixture(fixture)
        defer { try? FileManager.default.removeItem(at: files.directory) }

        let rawPin = "not-a-digest-\(UUID().uuidString)"
        try Data(rawPin.utf8).write(to: files.inputs.policyPin)
        XCTAssertThrowsError(try Loader.load(files: files.inputs)) { error in
            XCTAssertEqual(error as? Error, .malformedPin(.policy))
            XCTAssertFalse(String(describing: error).contains(rawPin))
        }

        try Data(hex("a").utf8).write(to: files.inputs.policyPin)
        XCTAssertThrowsError(try Loader.load(files: files.inputs)) { error in
            XCTAssertEqual(error as? Error, .pinMismatch(.policy))
        }
    }

    func testRejectsSameFileHardlinkAndSymlinkAuthorityAliases() throws {
        let fixture = makeFixture()
        let files = try writeFixture(fixture)
        defer { try? FileManager.default.removeItem(at: files.directory) }

        XCTAssertNoThrow(try Loader.load(files: files.inputs))

        XCTAssertThrowsError(try Loader.load(files: .init(
            policyBody: files.inputs.policyBody,
            policyPin: files.inputs.policyBody,
            productionRouteReceiptBody: files.inputs.productionRouteReceiptBody,
            productionRouteReceiptPin: files.inputs.productionRouteReceiptPin))
        ) { error in
            XCTAssertEqual(error as? Error, .aliasedAuthorityInput(.policy))
        }

        let hardlink = files.directory.appendingPathComponent("policy-hardlink.sha256")
        try FileManager.default.linkItem(at: files.inputs.policyBody, to: hardlink)
        XCTAssertThrowsError(try Loader.load(files: .init(
            policyBody: files.inputs.policyBody,
            policyPin: hardlink,
            productionRouteReceiptBody: files.inputs.productionRouteReceiptBody,
            productionRouteReceiptPin: files.inputs.productionRouteReceiptPin))
        ) { error in
            XCTAssertEqual(error as? Error, .aliasedAuthorityInput(.policy))
        }

        let symlink = files.directory.appendingPathComponent("policy-symlink.sha256")
        try FileManager.default.createSymbolicLink(
            at: symlink,
            withDestinationURL: files.inputs.policyBody)
        XCTAssertThrowsError(try Loader.load(files: .init(
            policyBody: files.inputs.policyBody,
            policyPin: symlink,
            productionRouteReceiptBody: files.inputs.productionRouteReceiptBody,
            productionRouteReceiptPin: files.inputs.productionRouteReceiptPin))
        ) { error in
            XCTAssertEqual(error as? Error, .nonRegularAuthorityInput(.policy))
        }

        XCTAssertThrowsError(try Loader.load(files: .init(
            policyBody: files.inputs.policyBody,
            policyPin: files.inputs.policyPin,
            productionRouteReceiptBody: files.inputs.productionRouteReceiptBody,
            productionRouteReceiptPin: files.inputs.policyPin))
        ) { error in
            XCTAssertEqual(
                error as? Error,
                .aliasedAuthorityInput(.productionRouteReceipt))
        }
    }

    func testRejectsTamperedAuthorityBodiesEvenWhenPinnedDigestIsReused() throws {
        let fixture = makeFixture()
        let files = try writeFixture(fixture)
        defer { try? FileManager.default.removeItem(at: files.directory) }
        var policy = fixture.policy
        policy.sealedBeforeMeasurements = false
        try encode(policy).write(to: files.inputs.policyBody)

        XCTAssertThrowsError(try Loader.load(files: files.inputs)) { error in
            XCTAssertEqual(error as? Error, .invalidAuthorityBody(.policy))
        }

        var receipt = fixture.receipt
        receipt.observationDigest = hex("b")
        try fixture.bodies.policy.write(to: files.inputs.policyBody)
        try encode(receipt).write(to: files.inputs.productionRouteReceiptBody)
        XCTAssertThrowsError(try Loader.load(files: files.inputs)) { error in
            XCTAssertEqual(
                error as? Error,
                .invalidAuthorityBody(.productionRouteReceipt))
        }
    }

    func testRejectsSymlinkToDistinctFileAndOversizedInputs() throws {
        let fixture = makeFixture()
        let files = try writeFixture(fixture)
        defer { try? FileManager.default.removeItem(at: files.directory) }

        let distinctTarget = files.directory.appendingPathComponent("distinct.sha256")
        try Data(fixture.policy.digest.utf8).write(to: distinctTarget)
        let symlink = files.directory.appendingPathComponent("distinct-symlink.sha256")
        try FileManager.default.createSymbolicLink(
            at: symlink,
            withDestinationURL: distinctTarget)

        XCTAssertThrowsError(try Loader.load(files: .init(
            policyBody: files.inputs.policyBody,
            policyPin: symlink,
            productionRouteReceiptBody: files.inputs.productionRouteReceiptBody,
            productionRouteReceiptPin: files.inputs.productionRouteReceiptPin))
        ) { error in
            XCTAssertEqual(error as? Error, .nonRegularAuthorityInput(.policy))
        }

        let oversized = Data(
            repeating: UInt8(ascii: "{"),
            count: Loader.maxAuthorityBodyBytes + 1)
        try oversized.write(to: files.inputs.policyBody)
        XCTAssertThrowsError(try Loader.load(files: files.inputs)) { error in
            XCTAssertEqual(error as? Error, .oversizedAuthorityInput(.policy))
        }
    }

    func testRejectsFIFOWithoutWaitingForWriter() throws {
        let fixture = makeFixture()
        let files = try writeFixture(fixture)
        defer { try? FileManager.default.removeItem(at: files.directory) }

        let fifo = files.directory.appendingPathComponent("policy-fifo")
        XCTAssertEqual(Darwin.mkfifo(fifo.path, S_IRUSR | S_IWUSR), 0)

        XCTAssertThrowsError(try Loader.load(files: .init(
            policyBody: fifo,
            policyPin: files.inputs.policyPin,
            productionRouteReceiptBody: files.inputs.productionRouteReceiptBody,
            productionRouteReceiptPin: files.inputs.productionRouteReceiptPin))
        ) { error in
            XCTAssertEqual(error as? Error, .nonRegularAuthorityInput(.policy))
        }
    }

    func testRejectsDuplicateJSONKeysBeforeDecoderNormalization() throws {
        let fixture = makeFixture()
        let files = try writeFixture(fixture)
        defer { try? FileManager.default.removeItem(at: files.directory) }

        let duplicateKeyBody = Data(
            "{\"schemaVersion\":2,\"schemaVersion\":2}".utf8)
        try duplicateKeyBody.write(to: files.inputs.policyBody)
        try Data(Loader.sha256Hex(duplicateKeyBody).utf8).write(
            to: files.inputs.policyPin)

        XCTAssertThrowsError(try Loader.load(files: files.inputs)) { error in
            XCTAssertEqual(error as? Error, .bodyHashPin(.policy))
        }

        let policy = Gate.frozenPromotionPolicy(
            evidenceID: hex("1"),
            artifact: artifact,
            runIdentity: runIdentity,
            claimAuthorities: [])
        var json = String(data: encode(policy), encoding: .utf8)!
        json.insert(
            contentsOf: "\"schemaVersion\":2,",
            at: json.index(after: json.startIndex))
        try Data(json.utf8).write(to: files.inputs.policyBody)
        try Data(policy.digest.utf8).write(to: files.inputs.policyPin)
        XCTAssertThrowsError(try Loader.load(files: files.inputs)) { error in
            XCTAssertEqual(error as? Error, .duplicateJSONKey(.policy))
        }
    }

    func testRejectsPolicyJSONNestedBeyondBoundedPreflightDepth() throws {
        let fixture = makeFixture()
        let files = try writeFixture(fixture)
        defer { try? FileManager.default.removeItem(at: files.directory) }

        let boundaryBody = nestedArrayJSON(
            depth: Loader.maxAuthorityJSONNestingDepth)
        try boundaryBody.write(to: files.inputs.policyBody)
        XCTAssertThrowsError(try Loader.load(files: files.inputs)) { error in
            XCTAssertEqual(error as? Error, .invalidAuthorityBody(.policy))
        }

        let excessiveBody = nestedArrayJSON(
            depth: Loader.maxAuthorityJSONNestingDepth + 1)
        try excessiveBody.write(to: files.inputs.policyBody)
        XCTAssertThrowsError(try Loader.load(files: files.inputs)) { error in
            XCTAssertEqual(error as? Error, .excessiveJSONNesting(.policy))
            XCTAssertFalse(String(describing: error).contains("[[["))
        }
    }

    func testRejectsReceiptJSONNestedBeyondBoundedPreflightDepth() throws {
        let fixture = makeFixture()
        let files = try writeFixture(fixture)
        defer { try? FileManager.default.removeItem(at: files.directory) }

        let boundaryBody = nestedArrayJSON(
            depth: Loader.maxAuthorityJSONNestingDepth)
        try boundaryBody.write(to: files.inputs.productionRouteReceiptBody)
        XCTAssertThrowsError(try Loader.load(files: files.inputs)) { error in
            XCTAssertEqual(
                error as? Error,
                .invalidAuthorityBody(.productionRouteReceipt))
        }

        let excessiveBody = nestedArrayJSON(
            depth: Loader.maxAuthorityJSONNestingDepth + 1)
        try excessiveBody.write(to: files.inputs.productionRouteReceiptBody)
        XCTAssertThrowsError(try Loader.load(files: files.inputs)) { error in
            XCTAssertEqual(
                error as? Error,
                .excessiveJSONNesting(.productionRouteReceipt))
            XCTAssertFalse(String(describing: error).contains("[[["))
        }
    }

    func testRejectsUnknownPolicyKeysBeforeDecoderIgnoresThem() throws {
        let fixture = makeFixture()
        let files = try writeFixture(fixture)
        defer { try? FileManager.default.removeItem(at: files.directory) }

        try insertingUnknownTopLevelField(in: fixture.bodies.policy).write(
            to: files.inputs.policyBody)
        XCTAssertThrowsError(try Loader.load(files: files.inputs)) { error in
            XCTAssertEqual(error as? Error, .invalidAuthorityBody(.policy))
        }

        let nested = try insertingUnknownField(
            in: fixture.bodies.policy,
            replacing: "\"bands\":[{\"claimKind\":",
            with: "\"bands\":[{\"unexpected\":true,\"claimKind\":")
        try nested.write(to: files.inputs.policyBody)
        XCTAssertThrowsError(try Loader.load(files: files.inputs)) { error in
            XCTAssertEqual(error as? Error, .invalidAuthorityBody(.policy))
        }
    }

    func testRejectsUnknownReceiptKeysBeforeDecoderIgnoresThem() throws {
        let fixture = makeFixture()
        let files = try writeFixture(fixture)
        defer { try? FileManager.default.removeItem(at: files.directory) }

        try insertingUnknownTopLevelField(
            in: fixture.bodies.productionRouteReceipt
        ).write(to: files.inputs.productionRouteReceiptBody)
        XCTAssertThrowsError(try Loader.load(files: files.inputs)) { error in
            XCTAssertEqual(
                error as? Error,
                .invalidAuthorityBody(.productionRouteReceipt))
        }

        let nested = try insertingUnknownField(
            in: fixture.bodies.productionRouteReceipt,
            replacing: "\"artifact\":{\"blockSize\":",
            with: "\"artifact\":{\"unexpected\":true,\"blockSize\":")
        try nested.write(to: files.inputs.productionRouteReceiptBody)
        XCTAssertThrowsError(try Loader.load(files: files.inputs)) { error in
            XCTAssertEqual(
                error as? Error,
                .invalidAuthorityBody(.productionRouteReceipt))
        }
    }

    func testRejectsRecomputedPolicyWithInvalidIndependentSemantics() throws {
        let invalidRunIdentity = makeRunIdentity(
            measurementClass: Gate.flagshipMeasurementClass,
            hostIdentityDigest: "not-hex")
        let fixture = makeFixture(runIdentity: invalidRunIdentity)
        let files = try writeFixture(fixture)
        defer { try? FileManager.default.removeItem(at: files.directory) }

        XCTAssertThrowsError(try Loader.load(files: files.inputs)) { error in
            XCTAssertEqual(error as? Error, .invalidAuthorityBody(.policy))
        }
    }

    func testRejectsRecomputedPolicyWithInvalidNestedAbsoluteAuthority() throws {
        var authorities = claimAuthorities()
        let invalid = Gate.absoluteAuthority(
            evidenceID: hex("7"),
            sealedBeforeMeasurements: false,
            bands: bands(kind: authorities[0].claimKind))
        authorities[0] = Qwen38PerformanceAttributionFrozenClaimAuthority(
            claimKind: authorities[0].claimKind,
            absoluteAuthority: invalid,
            cleanupAuthority: authorities[0].cleanupAuthority)
        let fixture = makeFixture(claimAuthorities: authorities)
        let files = try writeFixture(fixture)
        defer { try? FileManager.default.removeItem(at: files.directory) }

        XCTAssertThrowsError(try Loader.load(files: files.inputs)) { error in
            XCTAssertEqual(error as? Error, .invalidAuthorityBody(.policy))
        }
    }

    func testRejectsRecomputedPolicyWithInvalidNestedCleanupAuthority() throws {
        var authorities = claimAuthorities()
        let invalid = Gate.cleanupAuthority(
            evidenceID: hex("8"),
            minIdleSamples: 0,
            cooldownSeconds: 5,
            maxRSSDeltaBytes: 1,
            maxActiveMetalDeltaBytes: 1,
            maxCachedMetalDeltaBytes: 1,
            maxSwapDeltaBytes: 0,
            maxPageoutDelta: 0,
            allowedPressureStates: ["normal"],
            allowedThermalStates: ["nominal"])
        authorities[0] = Qwen38PerformanceAttributionFrozenClaimAuthority(
            claimKind: authorities[0].claimKind,
            absoluteAuthority: authorities[0].absoluteAuthority,
            cleanupAuthority: invalid)
        let fixture = makeFixture(claimAuthorities: authorities)
        let files = try writeFixture(fixture)
        defer { try? FileManager.default.removeItem(at: files.directory) }

        XCTAssertThrowsError(try Loader.load(files: files.inputs)) { error in
            XCTAssertEqual(error as? Error, .invalidAuthorityBody(.policy))
        }
    }

    func testRejectsRecomputedReceiptWithInvalidIndependentSemantics() throws {
        let fixture = makeFixture(receiptEvidenceID: hex("1"))
        let files = try writeFixture(fixture)
        defer { try? FileManager.default.removeItem(at: files.directory) }

        XCTAssertThrowsError(try Loader.load(files: files.inputs)) { error in
            XCTAssertEqual(
                error as? Error,
                .invalidAuthorityBody(.productionRouteReceipt))
        }
    }

    func testRejectsAuthorityInputMutationBetweenDescriptorReadAndFinalStat() throws {
        let fixture = makeFixture()
        let files = try writeFixture(fixture)
        defer { try? FileManager.default.removeItem(at: files.directory) }
#if DEBUG
        Loader.afterDescriptorReadTestHook = { slot in
            guard slot == "policyBody" else { return }
            var mutated = fixture.bodies.policy
            mutated.append(UInt8(ascii: " "))
            try mutated.write(to: files.inputs.policyBody)
        }
        defer { Loader.afterDescriptorReadTestHook = nil }

        XCTAssertThrowsError(try Loader.load(files: files.inputs)) { error in
            XCTAssertEqual(error as? Error, .unstableAuthorityInput(.policy))
        }
#else
        throw XCTSkip("unstable-read hook is debug-only")
#endif
    }

    func testRejectsSameSizeAuthorityInputRewriteWithRestoredModificationTime() throws {
        let fixture = makeFixture()
        let files = try writeFixture(fixture)
        defer { try? FileManager.default.removeItem(at: files.directory) }
#if DEBUG
        let originalStatus = try statFile(files.inputs.policyBody)
        Loader.afterDescriptorReadTestHook = { slot in
            guard slot == "policyBody" else { return }
            var mutated = fixture.bodies.policy
            mutated[mutated.startIndex] = UInt8(ascii: " ")
            XCTAssertEqual(mutated.count, fixture.bodies.policy.count)
            try self.overwriteFileInPlace(
                mutated,
                at: files.inputs.policyBody,
                restoringTimesFrom: originalStatus)
        }
        defer { Loader.afterDescriptorReadTestHook = nil }

        XCTAssertThrowsError(try Loader.load(files: files.inputs)) { error in
            XCTAssertEqual(error as? Error, .unstableAuthorityInput(.policy))
        }
#else
        throw XCTSkip("unstable-read hook is debug-only")
#endif
    }

    private func makeFixture(
        runIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity? = nil,
        receiptEvidenceID: String? = nil,
        claimAuthorities: [Qwen38PerformanceAttributionFrozenClaimAuthority]? = nil
    ) -> Fixture {
        let identity = runIdentity ?? self.runIdentity
        let policy = Gate.frozenPromotionPolicy(
            evidenceID: hex("1"),
            artifact: artifact,
            runIdentity: identity,
            claimAuthorities: claimAuthorities ?? self.claimAuthorities())
        let receipt = Gate.productionRouteReceipt(
            evidenceID: receiptEvidenceID ?? hex("2"),
            artifact: artifact,
            runIdentity: identity,
            backendBuildIdentityDigest: hex("3"),
            observationDigest: hex("4"))
        return Fixture(
            policy: policy,
            receipt: receipt,
            bodies: .init(
                policy: encode(policy),
                productionRouteReceipt: encode(receipt)))
    }

    private struct Fixture {
        var policy: Qwen38PerformanceAttributionFrozenPromotionPolicy
        var receipt: Qwen38PerformanceAttributionProductionRouteReceipt
        var bodies: AuthorityFixtureBodies
    }

    private struct AuthorityFixtureBodies {
        var policy: Data
        var productionRouteReceipt: Data
    }

    private struct FixtureFiles {
        var directory: URL
        var inputs: Qwen38PerformanceAttributionAuthorityFileInputs
    }

    private func writeFixture(_ fixture: Fixture) throws -> FixtureFiles {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "qwen38-authority-\(UUID().uuidString)",
            isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true)

        let policyBody = directory.appendingPathComponent("policy.json")
        let policyPin = directory.appendingPathComponent("policy.sha256")
        let receiptBody = directory.appendingPathComponent("receipt.json")
        let receiptPin = directory.appendingPathComponent("receipt.sha256")
        try fixture.bodies.policy.write(to: policyBody)
        try Data(fixture.policy.digest.utf8).write(to: policyPin)
        try fixture.bodies.productionRouteReceipt.write(to: receiptBody)
        try Data(fixture.receipt.digest.utf8).write(to: receiptPin)
        return FixtureFiles(
            directory: directory,
            inputs: .init(
                policyBody: policyBody,
                policyPin: policyPin,
                productionRouteReceiptBody: receiptBody,
                productionRouteReceiptPin: receiptPin))
    }

    private func statFile(_ url: URL) throws -> stat {
        var status = Darwin.stat()
        let result = url.withUnsafeFileSystemRepresentation { pointer -> Int32 in
            guard let pointer else { return -1 }
            return Darwin.fstatat(AT_FDCWD, pointer, &status, 0)
        }
        guard result == 0 else { throw posixError() }
        return status
    }

    private func overwriteFileInPlace(
        _ data: Data,
        at url: URL,
        restoringTimesFrom originalStatus: stat
    ) throws {
        let descriptor = url.withUnsafeFileSystemRepresentation { pointer -> Int32 in
            guard let pointer else { return -1 }
            return Darwin.open(pointer, O_WRONLY | O_NOFOLLOW)
        }
        guard descriptor >= 0 else { throw posixError() }
        defer { _ = Darwin.close(descriptor) }

        var offset = 0
        try data.withUnsafeBytes { buffer in
            guard let base = buffer.baseAddress else { return }
            while offset < buffer.count {
                let written = Darwin.pwrite(
                    descriptor,
                    base.advanced(by: offset),
                    buffer.count - offset,
                    off_t(offset))
                guard written > 0 else { throw posixError() }
                offset += written
            }
        }
        guard Darwin.fsync(descriptor) == 0 else { throw posixError() }

        var times = [originalStatus.st_atimespec, originalStatus.st_mtimespec]
        guard Darwin.futimens(descriptor, &times) == 0 else { throw posixError() }

        var finalStatus = Darwin.stat()
        guard Darwin.fstat(descriptor, &finalStatus) == 0 else { throw posixError() }
        guard finalStatus.st_dev == originalStatus.st_dev,
            finalStatus.st_ino == originalStatus.st_ino,
            finalStatus.st_size == originalStatus.st_size,
            finalStatus.st_mtimespec.tv_sec == originalStatus.st_mtimespec.tv_sec,
            finalStatus.st_mtimespec.tv_nsec == originalStatus.st_mtimespec.tv_nsec,
            finalStatus.st_ctimespec.tv_sec != originalStatus.st_ctimespec.tv_sec
                || finalStatus.st_ctimespec.tv_nsec != originalStatus.st_ctimespec.tv_nsec
        else {
            throw POSIXError(.EIO)
        }
    }

    private func posixError() -> POSIXError {
        POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
    }

    private var artifact: Qwen38MTPPerformanceScorecardArtifact {
        Qwen38MTPPerformanceScorecardGate.requiredArtifact
    }

    private var runIdentity: Qwen38MTPPerformanceScorecardTrustedRunIdentity {
        makeRunIdentity(measurementClass: Gate.flagshipMeasurementClass)
    }

    private func makeRunIdentity(
        measurementClass: String,
        hostIdentityDigest: String? = nil
    ) -> Qwen38MTPPerformanceScorecardTrustedRunIdentity {
        Qwen38MTPPerformanceScorecardTrustedRunIdentity(
            measurementClass: measurementClass,
            hardwareChip: "Apple M3 Ultra",
            hardwareRAMBytes: Gate.flagshipMinimumRAMBytes,
            hardwareOSBuild: "fixture-os",
            hostIdentityDigest: hostIdentityDigest ?? hex("5"),
            harnessGitSHA: String(repeating: "1", count: 40),
            candidateMLXSwiftVersion: "fixture-mlx-swift",
            referenceMLXVersion: nil,
            referenceMLXLMVersion: nil,
            modelLabel: "fixture-model",
            modelConfigHash: artifact.targetConfigSHA256,
            modelCheckpointManifestHash: artifact.targetTensorManifestSHA256,
            modelQuant: ModelQuantInfo(bits: 8, groupSize: 32),
            corpusID: "fixture-corpus",
            corpusContentHash: hex("6"))
    }

    private func encode<T: Encodable>(_ value: T) -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try! encoder.encode(value)
    }

    private func nestedArrayJSON(depth: Int) -> Data {
        let raw = String(repeating: "[", count: depth)
            + "0"
            + String(repeating: "]", count: depth)
        return Data(raw.utf8)
    }

    private func insertingUnknownTopLevelField(in body: Data) throws -> Data {
        try insertingUnknownField(
            in: body,
            replacing: "{",
            with: "{\"unexpected\":true,")
    }

    private func insertingUnknownField(
        in body: Data,
        replacing target: String,
        with replacement: String
    ) throws -> Data {
        guard let json = String(data: body, encoding: .utf8),
            json.contains(target)
        else {
            throw POSIXError(.EIO)
        }
        return Data(json.replacingOccurrences(
            of: target,
            with: replacement).utf8)
    }

    private func claimAuthorities()
        -> [Qwen38PerformanceAttributionFrozenClaimAuthority]
    {
        Gate.requiredClaimKinds.map { kind in
            Qwen38PerformanceAttributionFrozenClaimAuthority(
                claimKind: kind,
                absoluteAuthority: Gate.absoluteAuthority(
                    evidenceID: hex("7"),
                    bands: bands(kind: kind)),
                cleanupAuthority: Gate.cleanupAuthority(
                    evidenceID: hex("8"),
                    minIdleSamples: 3,
                    cooldownSeconds: 5,
                    maxRSSDeltaBytes: 1,
                    maxActiveMetalDeltaBytes: 1,
                    maxCachedMetalDeltaBytes: 1,
                    maxSwapDeltaBytes: 0,
                    maxPageoutDelta: 0,
                    allowedPressureStates: ["normal"],
                    allowedThermalStates: ["nominal"]))
        }
    }

    private func bands(
        kind: Qwen38PerformanceAttributionClaimKind
    ) -> [Qwen38PerformanceAttributionAbsoluteBand] {
        let cells: [(
            Qwen38MTPPerformanceScorecardBenchmarkContextTokens,
            Qwen38MTPPerformanceScorecardPrefixKind
        )]
        if kind == .prefixMatrix {
            cells = [
                (.tokens4096, .cold),
                (.tokens4096, .exactWarmPrefix),
                (.tokens16384, .cold),
                (.tokens16384, .exactWarmPrefix),
                (.tokens32768, .cold),
                (.tokens32768, .exactWarmPrefix),
            ]
        } else {
            cells = [(.tokens4096, .cold)]
        }
        let concurrencies = kind == .continuousBatchNoSpec ? [2, 4] : [1]
        return cells.flatMap { context, prefix in
            concurrencies.map { concurrency in
                Qwen38PerformanceAttributionAbsoluteBand(
                    claimKind: kind,
                    concurrency: concurrency,
                    contextTokens: context,
                    prefixKind: prefix,
                    maxPrefillSeconds: 1,
                    maxTTFTSeconds: 1,
                    minDecodeTokensPerSecond: 1,
                    minAggregateThroughputTokensPerSecond: 1)
            }
        }
    }

    private func bandKey(
        _ band: Qwen38PerformanceAttributionAbsoluteBand
    ) -> String {
        "\(band.concurrency):\(band.contextTokens.rawValue):\(band.prefixKind.rawValue)"
    }

    private func hex(_ character: Character) -> String {
        String(repeating: String(character), count: 64)
    }
}
