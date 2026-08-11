import CryptoKit
import Foundation
import XCTest

@testable import ProofControl

final class SyntheticSharedCacheDiscoveryRangePlanTests: XCTestCase {
    func testE2BExactFiveTypeExistenceSentinel() {
        _ = SyntheticSharedCacheBoundedReadTranscript.self
        _ = SyntheticSharedCacheRangeConsumerKind.self
        _ = SyntheticSharedCacheDiscoveryRangePlanComparison.self
        _ = SyntheticSharedCacheDiscoveryRangePlanFailure.self
        _ = SyntheticSharedCacheDiscoveryRangePlanVerifier.self
    }

    func testMainOnlyAbsentSignatureFixtureProducesExactInertPlan()
        throws
    {
        let fixture = try Self.fixture()
        let result = try Self.compare(fixture)

        XCTAssertEqual(result.sourceProfile, .reviewed)
        XCTAssertEqual(result.discoveryStores.count, 1)
        XCTAssertEqual(result.discoveryStores[0].fileOrdinal, 0)
        XCTAssertEqual(
            result.discoveryStores[0].bytes,
            fixture.discoveryBytes
        )
        XCTAssertEqual(result.logicalSelectedRanges.count, 3)
        XCTAssertEqual(result.additionalPhysicalRanges.count, 2)
        XCTAssertEqual(
            result.consumerBindings.map(\.consumerKind),
            [
                .setHeader,
                .mappingTable,
                .imageTable,
                .installName,
                .imageNameWindow,
                .machOHeader,
                .loadCommands,
            ]
        )
        XCTAssertEqual(
            result.resourceCounts,
            SyntheticSharedCacheDiscoveryRangePlanComparison.ResourceCounts(
                discoveryBytes: UInt64(fixture.discoveryBytes.count),
                rawProbeRequestedBytes: 4_185,
                probeChunkCount: 3,
                discoveryAttemptCount: 1,
                planProbeAttemptCount: 3,
                totalAttemptCount: 4,
                logicalRangeCount: 3,
                additionalBytes: 4_185,
                combinedRetainedBytes:
                    UInt64(fixture.discoveryBytes.count) + 4_185
            )
        )
        let installBinding = try XCTUnwrap(
            result.consumerBindings.first {
                $0.consumerKind == .installName
            }
        )
        XCTAssertEqual(
            installBinding.pieces,
            [.additional(
                rangeOrdinal: 0,
                relativeStart: 0,
                length: UInt64(Self.imageName.count)
            )]
        )
        XCTAssertEqual(result.runtimeDecision, .noGo)
        Self.assertNoAuthority(result)
    }

    func testDiscoveryNoEOFStateMachineAndExactLimitsRefuse()
        throws
    {
        let fixture = try Self.fixture()
        let metadata = fixture.metadata
        let byteCount = UInt64(fixture.discoveryBytes.count)

        var eof = fixture
        eof.discoveryTranscripts[0] = Self.discoveryTranscript(
            metadata: metadata,
            bytes: fixture.discoveryBytes,
            events: [.endOfFile(offset: 0)]
        )
        Self.assertRefuses(
            eof,
            as: .discoveryTranscript(
                .unexpectedEndOfFile(transcriptOrdinal: 0, eventOrdinal: 0)
            )
        )

        var trailing = fixture
        trailing.discoveryTranscripts[0] = Self.discoveryTranscript(
            metadata: metadata,
            bytes: fixture.discoveryBytes,
            events: [
                .bytes(offset: 0, data: fixture.discoveryBytes),
                .interrupted(offset: byteCount),
            ]
        )
        Self.assertRefuses(
            trailing,
            as: .discoveryTranscript(.trailingEvent(
                transcriptOrdinal: 0,
                eventOrdinal: 1
            ))
        )

        var interrupted = fixture
        interrupted.discoveryTranscripts[0] = Self.discoveryTranscript(
            metadata: metadata,
            bytes: fixture.discoveryBytes,
            events: Array(repeating: .interrupted(offset: 0), count: 9)
        )
        Self.assertRefuses(
            interrupted,
            as: .readInterruptedLimit(
                transcriptOrdinal: 0,
                eventOrdinal: 8
            )
        )

        var fragments = fixture
        let seventeen = Self.fragmentEvents(
            fixture.discoveryBytes,
            fragments: 17
        )
        fragments.discoveryTranscripts[0] = Self.discoveryTranscript(
            metadata: metadata,
            bytes: fixture.discoveryBytes,
            events: seventeen
        )
        Self.assertRefuses(
            fragments,
            as: .readFragmentLimit(
                transcriptOrdinal: 0,
                eventOrdinal: 16
            )
        )

        var explicitError = fixture
        explicitError.discoveryTranscripts[0] = Self.discoveryTranscript(
            metadata: metadata,
            bytes: fixture.discoveryBytes,
            events: [.error(offset: 0, code: 5)]
        )
        Self.assertRefuses(
            explicitError,
            as: .readError(
                transcriptOrdinal: 0,
                eventOrdinal: 0,
                code: 5
            )
        )
    }

    func testDiscoveryMetadataAndContinuityAreCheckedBeforeAnchors()
        throws
    {
        var fixture = try Self.fixture()
        var drift = fixture.metadata
        drift.device += 1
        fixture.discoveryTranscripts[0] =
            SyntheticSharedCacheBoundedReadTranscript(
                purpose: .setHeaderDiscovery,
                fileOrdinal: 0,
                rangeStart: 0,
                requestedByteCount: UInt64(fixture.discoveryBytes.count),
                beforeMetadata: drift,
                afterMetadata: fixture.metadata,
                events: [
                    .bytes(offset: 0, data: fixture.discoveryBytes),
                ]
            )
        fixture.planProbeTranscripts = []

        Self.assertRefuses(
            fixture,
            as: .discoveryTranscript(.metadata(
                transcriptOrdinal: 0,
                position: .before,
                field: .device
            ))
        )

        var mismatch = try Self.fixture()
        var wrong = mismatch.discoveryBytes
        wrong[wrong.startIndex] ^= 0xff
        mismatch.discoveryTranscripts[0] = Self.discoveryTranscript(
            metadata: mismatch.metadata,
            bytes: mismatch.discoveryBytes,
            events: [.bytes(offset: 0, data: wrong)]
        )
        Self.assertRefuses(
            mismatch,
            as: .discoveryContinuity(fileOrdinal: 0, reason: .bytes)
        )
    }

    func testCurrentRolesRejectBeforeAnyPlanProbeInspection() throws {
        var fixture = try Self.fixture()
        fixture.selfGuardExpectation = fixture.gitExpectation
        fixture.planProbeTranscripts = [
            SyntheticSharedCacheBoundedReadTranscript(
                purpose: .planProbe(
                    consumerKind: .setHeader,
                    consumerOrdinal: Int.max
                ),
                fileOrdinal: Int.max,
                rangeStart: UInt64.max,
                requestedByteCount: UInt64.max,
                beforeMetadata: fixture.metadata,
                afterMetadata: fixture.metadata,
                events: [.bytes(offset: 0, data: Data([0xff]))]
            ),
        ]

        Self.assertRefuses(
            fixture,
            as: .anchoredMemberSet(.role(
                expected: .selfGuard,
                actual: .git
            ))
        )
    }

    func testProbeListIsExactAndNoEOF() throws {
        var missing = try Self.fixture()
        missing.planProbeTranscripts.removeLast()
        Self.assertRefuses(
            missing,
            as: .rangePlan(.probeCount(
                expected: 3,
                actual: 2
            ))
        )

        var eof = try Self.fixture()
        let original = eof.planProbeTranscripts[0]
        eof.planProbeTranscripts[0] =
            SyntheticSharedCacheBoundedReadTranscript(
                purpose: original.purpose,
                fileOrdinal: original.fileOrdinal,
                rangeStart: original.rangeStart,
                requestedByteCount: original.requestedByteCount,
                beforeMetadata: original.beforeMetadata,
                afterMetadata: original.afterMetadata,
                events: [.endOfFile(offset: original.rangeStart)]
            )
        Self.assertRefuses(
            eof,
            as: .discoveryTranscript(.unexpectedEndOfFile(
                transcriptOrdinal: 1,
                eventOrdinal: 0
            ))
        )

        var wrongKind = try Self.fixture()
        let probe = wrongKind.planProbeTranscripts[0]
        wrongKind.planProbeTranscripts[0] =
            SyntheticSharedCacheBoundedReadTranscript(
                purpose: .planProbe(
                    consumerKind: .installName,
                    consumerOrdinal: 0
                ),
                fileOrdinal: probe.fileOrdinal,
                rangeStart: probe.rangeStart,
                requestedByteCount: probe.requestedByteCount,
                beforeMetadata: probe.beforeMetadata,
                afterMetadata: probe.afterMetadata,
                events: probe.events
            )
        Self.assertRefuses(
            wrongKind,
            as: .rangePlan(.invalidInputConsumer(
                transcriptOrdinal: 0,
                consumerKind: .installName
            ))
        )
    }

    func testReturnedBytesAreCallerIndependent() throws {
        var fixture = try Self.fixture()
        let expectedDiscovery = fixture.discoveryBytes
        let expectedAdditional = fixture.planProbeTranscripts
            .flatMap(\.events)
            .compactMap { event -> Data? in
                if case .bytes(_, let data) = event { return data }
                return nil
            }
        let result = try Self.compare(fixture)

        fixture.discoveryBytes[0] ^= 0xff
        fixture.discoveryTranscripts.removeAll()
        fixture.planProbeTranscripts.removeAll()

        XCTAssertEqual(result.discoveryStores[0].bytes, expectedDiscovery)
        XCTAssertEqual(
            result.additionalPhysicalRanges.map(\.bytes),
            [expectedAdditional[0], expectedAdditional[1] + expectedAdditional[2]]
        )
        Self.assertNoAuthority(result)
    }

    func testProbeOrderMayDifferFromCanonicalSortedConsumerBindings()
        throws
    {
        let fixture = try Self.multiMemberFixture()

        XCTAssertEqual(
            fixture.planProbeTranscripts.map(\.purpose),
            [
                .planProbe(consumerKind: .imageNameWindow, consumerOrdinal: 0),
                .planProbe(consumerKind: .machOHeader, consumerOrdinal: 1),
                .planProbe(consumerKind: .machOHeader, consumerOrdinal: 0),
                .planProbe(consumerKind: .loadCommands, consumerOrdinal: 1),
                .planProbe(consumerKind: .loadCommands, consumerOrdinal: 0),
            ]
        )

        let result = try Self.compare(fixture)

        XCTAssertEqual(
            result.consumerBindings.map {
                ConsumerBindingKey($0.consumerKind, $0.consumerOrdinal)
            },
            [
                ConsumerBindingKey(.setHeader, 0),
                ConsumerBindingKey(.mappingTable, 0),
                ConsumerBindingKey(.imageTable, 0),
                ConsumerBindingKey(.installName, 1),
                ConsumerBindingKey(.imageNameWindow, 0),
                ConsumerBindingKey(.installName, 0),
                ConsumerBindingKey(.machOHeader, 1),
                ConsumerBindingKey(.loadCommands, 1),
                ConsumerBindingKey(.machOHeader, 0),
                ConsumerBindingKey(.loadCommands, 0),
            ]
        )
        XCTAssertEqual(result.consumerBindings[3].start, 4_096)
        XCTAssertEqual(result.consumerBindings[5].start, 8_192)
        Self.assertNoAuthority(result)
    }

    func testProbeMetadataBeforeAndAfterTaxonomyUsesGlobalOrdinal()
        throws
    {
        var after = try Self.fixture()
        var drift = after.metadata
        drift.inode += 1
        after.planProbeTranscripts[0] = Self.replacingMetadata(
            after.planProbeTranscripts[0],
            before: after.metadata,
            after: drift
        )
        Self.assertRefuses(
            after,
            as: .rangePlan(.metadata(
                transcriptOrdinal: 0,
                position: .after,
                field: .inode
            ))
        )

        var before = try Self.fixture()
        drift = before.metadata
        drift.mode ^= 0o111
        before.planProbeTranscripts[1] = Self.replacingMetadata(
            before.planProbeTranscripts[1],
            before: drift,
            after: before.metadata
        )
        Self.assertRefuses(
            before,
            as: .rangePlan(.metadata(
                transcriptOrdinal: 1,
                position: .before,
                field: .mode
            ))
        )
    }

    func testPresentCodeSignatureUsesLinkeditTranslationAndRetainsBinding()
        throws
    {
        let signed = try Self.signedFixture()
        let result = try Self.compare(signed.fixture)

        XCTAssertEqual(
            result.consumerBindings.map(\.consumerKind),
            [
                .setHeader,
                .mappingTable,
                .imageTable,
                .installName,
                .imageNameWindow,
                .machOHeader,
                .loadCommands,
                .codeSignature,
            ]
        )
        let signatureBinding = try XCTUnwrap(result.consumerBindings.last)
        XCTAssertEqual(signatureBinding.consumerKind, .codeSignature)
        XCTAssertEqual(signatureBinding.start, signed.signatureCacheFileOffset)
        XCTAssertEqual(signatureBinding.length, UInt64(signed.signature.count))
        XCTAssertEqual(result.additionalPhysicalRanges.last?.bytes,
            signed.signature)
        XCTAssertEqual(
            result.resourceCounts.rawProbeRequestedBytes,
            UInt64(4_097 + 32 + signed.loadCommands.count +
                signed.signature.count)
        )
        Self.assertNoAuthority(result)
    }

    func testPresentCodeSignatureRequiresLinkeditTranslation() throws {
        var signed = try Self.signedFixture(includeLinkedit: false).fixture
        signed.planProbeTranscripts.removeLast()

        Self.assertRefuses(
            signed,
            as: .rangePlan(.linkedit(memberOrdinal: 0))
        )
    }

    func testCrossRoleUnionAllowsDisjointMembersAndRejectsConflicts()
        throws
    {
        let disjoint = try Self.multiMemberFixture(
            gitMemberIndexes: [0],
            selfGuardMemberIndexes: [0, 1]
        )
        let result = try Self.compare(disjoint)
        XCTAssertEqual(
            result.consumerBindings.filter {
                $0.consumerKind == .installName
            }.map(\.consumerOrdinal),
            [1, 0]
        )

        var conflict = try Self.multiMemberFixture()
        let cacheSet = try Self.cacheSet(metadata: conflict.metadata)
        let loadCommands = Self.loadCommands(uuid: Self.altImageUUID)
        let member = try Self.memberAndEdge(
            cacheSet: cacheSet,
            name: Self.firstSortedImageName,
            uuid: Self.altImageUUID,
            loadCommands: loadCommands,
            primaryCodeDirectory: .absent
        ).member
        let edge = Self.edge(member)
        conflict.selfGuardExpectation = try Self.anchoredExpectation(
            role: .selfGuard,
            cacheSet: cacheSet,
            members: [member],
            edges: [edge]
        )
        conflict.planProbeTranscripts = []

        Self.assertRefuses(
            conflict,
            as: .anchoredMemberSet(.memberConflict)
        )
    }

    func testMalformedMachOAndLoadCommandsRefuseSpecifically() throws {
        var badHeader = try Self.fixture()
        var header = Self.machHeader(loadCommandBytes: Self.loadCommands().count)
        header[0] = 0
        badHeader.planProbeTranscripts[1] = Self.replacingBytes(
            badHeader.planProbeTranscripts[1],
            with: header
        )
        Self.assertRefuses(
            badHeader,
            as: .rangePlan(.machOHeader(memberOrdinal: 0))
        )

        var badFrame = try Self.fixture()
        var framed = Self.loadCommands()
        Self.writeUInt32LE(25, into: &framed, at: 4)
        badFrame.planProbeTranscripts[2] = Self.replacingBytes(
            badFrame.planProbeTranscripts[2],
            with: framed
        )
        Self.assertRefuses(
            badFrame,
            as: .rangePlan(.loadCommandFrame(
                memberOrdinal: 0,
                commandOrdinal: 0
            ))
        )

        var unknown = try Self.fixture()
        var commands = Self.loadCommands()
        Self.writeUInt32LE(0xffff_ffff, into: &commands, at: 0)
        unknown.planProbeTranscripts[2] = Self.replacingBytes(
            unknown.planProbeTranscripts[2],
            with: commands
        )
        Self.assertRefuses(
            unknown,
            as: .rangePlan(.unknownLoadCommand(
                memberOrdinal: 0,
                commandOrdinal: 0,
                command: 0xffff_ffff
            ))
        )
    }

    func testNormalizationCoalescesOverlapAndRefusesByteConflict()
        throws
    {
        let signed = try Self.signedFixture(
            signatureCacheFileOffset: Self.imageFileOffset + 64
        ).fixture

        Self.assertRefuses(
            signed,
            as: .rangePlan(.overlapBytes)
        )
    }

    func testExactInterruptionAndFragmentBoundariesAcceptAtLimit()
        throws
    {
        var interrupted = try Self.fixture()
        interrupted.discoveryTranscripts[0] = Self.discoveryTranscript(
            metadata: interrupted.metadata,
            bytes: interrupted.discoveryBytes,
            events:
                Array(repeating: .interrupted(offset: 0), count: 8) +
                [.bytes(offset: 0, data: interrupted.discoveryBytes)]
        )
        XCTAssertNoThrow(try Self.compare(interrupted))

        var fragments = try Self.fixture()
        let probe = fragments.planProbeTranscripts[0]
        fragments.planProbeTranscripts[0] = Self.replacingEvents(
            probe,
            with: Self.fragmentEvents(
                Self.bytes(from: probe),
                fragments: 16,
                offsetBase: probe.rangeStart
            )
        )
        XCTAssertNoThrow(try Self.compare(fragments))
    }

    func testSubcacheFixtureTranslatesToFileOrdinalOneWithoutCoalescing()
        throws
    {
        let fixture = try Self.subcacheFixture()
        let result = try Self.compare(fixture)

        XCTAssertEqual(result.discoveryStores.count, 2)
        XCTAssertEqual(result.discoveryStores[0].bytes, fixture.discoveryBytes)
        XCTAssertEqual(
            result.discoveryStores[1].bytes,
            Self.bytes(from: fixture.discoveryTranscripts[1])
        )
        XCTAssertEqual(
            result.discoveryStores.map(\.fileOrdinal),
            [0, 1]
        )
        XCTAssertEqual(
            result.consumerBindings.filter {
                [.setHeader, .mappingTable].contains($0.consumerKind)
            }.map { ConsumerBindingKey($0.consumerKind, $0.fileOrdinal) },
            [
                ConsumerBindingKey(.setHeader, 0),
                ConsumerBindingKey(.mappingTable, 0),
                ConsumerBindingKey(.setHeader, 1),
                ConsumerBindingKey(.mappingTable, 1),
            ]
        )
        XCTAssertEqual(
            result.consumerBindings.filter {
                [.machOHeader, .loadCommands].contains($0.consumerKind)
            }.map(\.fileOrdinal),
            [1, 1]
        )
        XCTAssertTrue(result.logicalSelectedRanges.contains {
            $0.fileOrdinal == 0
        })
        XCTAssertTrue(result.logicalSelectedRanges.contains {
            $0.fileOrdinal == 1
        })
        XCTAssertEqual(
            Set(result.additionalPhysicalRanges.map(\.fileOrdinal)),
            Set([0, 1])
        )
        XCTAssertEqual(result.resourceCounts.discoveryAttemptCount, 2)

        var drift = fixture
        let subcacheBytes = Self.bytes(from: drift.discoveryTranscripts[1])
        var wrongSubcacheBytes = subcacheBytes
        wrongSubcacheBytes[0] ^= 0xff
        drift.discoveryTranscripts[1] = Self.discoveryTranscript(
            fileOrdinal: 1,
            metadata: drift.discoveryTranscripts[1].beforeMetadata,
            bytes: subcacheBytes,
            events: [.bytes(offset: 0, data: wrongSubcacheBytes)]
        )
        Self.assertRefuses(
            drift,
            as: .discoveryContinuity(fileOrdinal: 1, reason: .bytes)
        )
    }

    func testImageTableAndInstallNameFailuresAreTableDriven() throws {
        let duplicateName: (String, () throws -> Fixture,
            SyntheticSharedCacheDiscoveryRangePlanFailure) = (
                "duplicate selected name",
                {
                    var fixture = try Self.multiMemberFixture()
                    let probe = fixture.planProbeTranscripts[0]
                    var window = Self.bytes(from: probe)
                    let firstRelative = 4_096
                    window.replaceSubrange(
                        0..<Self.firstSortedImageName.count,
                        with: Self.firstSortedImageName
                    )
                    window.replaceSubrange(
                        firstRelative..<(firstRelative +
                            Self.firstSortedImageName.count),
                        with: Self.firstSortedImageName
                    )
                    fixture.planProbeTranscripts[0] = Self.replacingBytes(
                        probe,
                        with: window
                    )
                    return fixture
                },
                .imageTable(.selectedNameDuplicate(memberOrdinal: 0))
            )
        let addressAlias: (String, () throws -> Fixture,
            SyntheticSharedCacheDiscoveryRangePlanFailure) = (
                "selected address alias",
                {
                    try Self.multiMemberFixture(addressAlias: true)
                },
                .imageTable(.selectedAddressAlias(memberOrdinal: 1))
            )
        let missingTerminator: (String, () throws -> Fixture,
            SyntheticSharedCacheDiscoveryRangePlanFailure) = (
                "missing terminator",
                {
                    var fixture = try Self.fixture()
                    let probe = fixture.planProbeTranscripts[0]
                    fixture.planProbeTranscripts[0] = Self.replacingBytes(
                        probe,
                        with: Data(repeating: UInt8(ascii: "A"), count: 4_097)
                    )
                    return fixture
                },
                .installName(.terminator(row: 0))
            )
        let invalidGrammar: (String, () throws -> Fixture,
            SyntheticSharedCacheDiscoveryRangePlanFailure) = (
                "install-name grammar",
                {
                    var fixture = try Self.fixture()
                    let probe = fixture.planProbeTranscripts[0]
                    var window = Data(repeating: 0, count: 4_097)
                    window.replaceSubrange(
                        0..<Data("relative/lib.dylib".utf8).count,
                        with: Data("relative/lib.dylib".utf8)
                    )
                    fixture.planProbeTranscripts[0] = Self.replacingBytes(
                        probe,
                        with: window
                    )
                    return fixture
                },
                .installName(.syntax(row: 0))
            )

        for testCase in [
            duplicateName,
            addressAlias,
            missingTerminator,
            invalidGrammar,
        ] {
            Self.assertRefuses(
                try testCase.1(),
                as: testCase.2,
                file: #filePath,
                line: #line
            )
        }
    }

    func testForbiddenRecognizedAndIdentityLoadCommandTaxonomy()
        throws
    {
        for command in Self.forbiddenCommands {
            var fixture = try Self.fixtureWithLoadCommands(
                Self.loadCommands(prefixedBy: [(command, 8)]),
                commandCount: 3
            )
            Self.assertRefuses(
                fixture,
                as: .rangePlan(.forbiddenLoadCommand(
                    memberOrdinal: 0,
                    commandOrdinal: 1,
                    command: command
                ))
            )
            fixture.planProbeTranscripts.removeAll()
        }

        for command in Self.recognizedPassthroughCommands {
            XCTAssertNoThrow(try Self.compare(
                Self.fixtureWithLoadCommands(
                    Self.loadCommands(prefixedBy: [(command, 8)]),
                    commandCount: 3
                )
            ), "recognized command \(String(command, radix: 16))")
        }

        for command in [UInt32(0x0000_7fff), UInt32(0x8000_7fff)] {
            var fixture = try Self.fixture()
            var commands = Self.bytes(from: fixture.planProbeTranscripts[2])
            Self.writeUInt32LE(command, into: &commands, at: 0)
            fixture.planProbeTranscripts[2] = Self.replacingBytes(
                fixture.planProbeTranscripts[2],
                with: commands
            )
            Self.assertRefuses(
                fixture,
                as: .rangePlan(.unknownLoadCommand(
                    memberOrdinal: 0,
                    commandOrdinal: 0,
                    command: command
                ))
            )
        }

        let duplicateUUID = Self.uuidCommand(Self.imageUUID) +
            Self.uuidCommand(Self.altImageUUID) +
            Self.idDylibCommand()
        Self.assertRefuses(
            try Self.fixtureWithLoadCommands(duplicateUUID, commandCount: 3),
            as: .rangePlan(.uuid(memberOrdinal: 0))
        )

        Self.assertRefuses(
            try Self.fixtureWithLoadCommands(
                Self.idDylibCommand(),
                commandCount: 1
            ),
            as: .rangePlan(.uuid(memberOrdinal: 0))
        )

        let duplicateDylib = Self.uuidCommand(Self.imageUUID) +
            Self.idDylibCommand() +
            Self.idDylibCommand()
        Self.assertRefuses(
            try Self.fixtureWithLoadCommands(duplicateDylib, commandCount: 3),
            as: .rangePlan(.dylibIdentity(memberOrdinal: 0))
        )

        Self.assertRefuses(
            try Self.fixtureWithLoadCommands(
                Self.uuidCommand(Self.imageUUID),
                commandCount: 1
            ),
            as: .rangePlan(.dylibIdentity(memberOrdinal: 0))
        )
    }

    func testCodeDirectoryPolicyAndSignatureSizeFailures() throws {
        let signed = try Self.signedFixture()
        let alternatePrimary = Self.codeDirectory(
            hashType: 2,
            flags: 0,
            signingIdentifier: Data("com.example.fixturf".utf8),
            teamIdentifier: Data("TEAM123456".utf8)
        )
        let alternateSignature = Self.superBlob(entries: [
            (0, alternatePrimary),
            (
                0x10000,
                Self.genericBlob(
                    magic: Self.csMagicBlobWrapper,
                    payload: Data([0x30, 0x00])
                )
            ),
        ])
        var mismatch = signed.fixture
        mismatch.planProbeTranscripts[3] = Self.replacingBytes(
            mismatch.planProbeTranscripts[3],
            with: alternateSignature
        )
        Self.assertRefuses(
            mismatch,
            as: .rangePlan(.expectedMember(memberOrdinal: 0))
        )

        let presentZero = try Self.fixtureWithLoadCommands(
            signed.loadCommands,
            commandCount: 4,
            primaryCodeDirectory: .absent
        )
        var presentZeroWithProbe = presentZero
        presentZeroWithProbe.planProbeTranscripts.append(
            signed.fixture.planProbeTranscripts[3]
        )
        Self.assertRefuses(
            presentZeroWithProbe,
            as: .rangePlan(.expectedMember(memberOrdinal: 0))
        )

        Self.assertRefuses(
            try Self.fixtureWithLoadCommands(
                Self.loadCommands(),
                commandCount: 2,
                primaryCodeDirectory: .present(
                    blobSHA256: Self.sha256(Self.codeDirectory(
                        hashType: 2,
                        flags: 0,
                        signingIdentifier: Data("com.example.fixture".utf8),
                        teamIdentifier: Data("TEAM123456".utf8)
                    ))
                )
            ),
            as: .rangePlan(.expectedMember(memberOrdinal: 0))
        )

        let signatureCommand = Data(signed.loadCommands.suffix(16))
        Self.assertRefuses(
            try Self.fixtureWithLoadCommands(
                signed.loadCommands + signatureCommand,
                commandCount: 5,
                primaryCodeDirectory: .present(
                    blobSHA256: Self.sha256(Self.codeDirectory(
                        hashType: 2,
                        flags: 0,
                        signingIdentifier: Data("com.example.fixture".utf8),
                        teamIdentifier: Data("TEAM123456".utf8)
                    ))
                )
            ),
            as: .rangePlan(.codeSignature(memberOrdinal: 0))
        )

        let linkeditCommand = Data(signed.loadCommands[24..<96])
        var duplicateLinkedit = Data(signed.loadCommands.prefix(96))
        duplicateLinkedit.append(linkeditCommand)
        duplicateLinkedit.append(signed.loadCommands.dropFirst(96))
        Self.assertRefuses(
            try Self.fixtureWithLoadCommands(
                duplicateLinkedit,
                commandCount: 5,
                primaryCodeDirectory: .present(
                    blobSHA256: Self.sha256(Self.codeDirectory(
                        hashType: 2,
                        flags: 0,
                        signingIdentifier: Data("com.example.fixture".utf8),
                        teamIdentifier: Data("TEAM123456".utf8)
                    ))
                )
            ),
            as: .rangePlan(.linkedit(memberOrdinal: 0))
        )

        for (size, expected) in [
            (UInt64(0), SyntheticSharedCacheDiscoveryRangePlanFailure
                .rangePlan(.codeSignature(memberOrdinal: 0))),
            (UInt64(262_145), SyntheticSharedCacheDiscoveryRangePlanFailure
                .rangePlan(.codeSignature(memberOrdinal: 0))),
        ] {
            let primary = Self.codeDirectory(
                hashType: 2,
                flags: 0,
                signingIdentifier: Data("com.example.fixture".utf8),
                teamIdentifier: Data("TEAM123456".utf8)
            )
            let commands = Self.signedLoadCommands(
                uuid: Self.imageUUID,
                localSignatureOffset: 20_480,
                signatureBytes: size,
                includeLinkedit: false
            )
            Self.assertRefuses(
                try Self.fixtureWithLoadCommands(
                    commands,
                    commandCount: 3,
                    primaryCodeDirectory: .present(
                        blobSHA256: Self.sha256(primary)
                    )
                ),
                as: expected
            )
        }
    }

    func testPracticalResourceAndMachOBoundaries() throws {
        var tooManyProbes = try Self.fixture()
        tooManyProbes.planProbeTranscripts = Array(
            repeating: tooManyProbes.planProbeTranscripts[0],
            count: 4_097
        )
        Self.assertRefuses(
            tooManyProbes,
            as: .rangePlan(.probeCount(expected: 4_096, actual: 4_097))
        )

        var tooManyCommands = try Self.fixture()
        var header = Self.bytes(from: tooManyCommands.planProbeTranscripts[1])
        Self.writeUInt32LE(257, into: &header, at: 16)
        tooManyCommands.planProbeTranscripts[1] = Self.replacingBytes(
            tooManyCommands.planProbeTranscripts[1],
            with: header
        )
        Self.assertRefuses(
            tooManyCommands,
            as: .rangePlan(.machOHeader(memberOrdinal: 0))
        )

        var tooLargeCommands = try Self.fixture()
        header = Self.bytes(from: tooLargeCommands.planProbeTranscripts[1])
        Self.writeUInt32LE(262_145, into: &header, at: 20)
        tooLargeCommands.planProbeTranscripts[1] = Self.replacingBytes(
            tooLargeCommands.planProbeTranscripts[1],
            with: header
        )
        Self.assertRefuses(
            tooLargeCommands,
            as: .rangePlan(.machOHeader(memberOrdinal: 0))
        )

        let maxLoadCommands = Self.maximalLoadCommands()
        XCTAssertEqual(maxLoadCommands.count, 262_144)
        XCTAssertNoThrow(try Self.compare(
            Self.fixtureWithLoadCommands(
                maxLoadCommands,
                commandCount: 256
            )
        ))
    }

    func testRawProbeByteCeilingAcceptsExactEnvelopeAndRefusesOneOver()
        throws
    {
        let fixture = try Self.fixture()
        func probes(lastLength: UInt64) -> [
            SyntheticSharedCacheBoundedReadTranscript
        ] {
            (0..<4_096).map { index in
                SyntheticSharedCacheBoundedReadTranscript(
                    purpose: .planProbe(
                        consumerKind: .codeSignature,
                        consumerOrdinal: 0
                    ),
                    fileOrdinal: 0,
                    rangeStart: UInt64(index),
                    requestedByteCount:
                        index == 4_095 ? lastLength : 65_536,
                    beforeMetadata: fixture.metadata,
                    afterMetadata: fixture.metadata,
                    events: []
                )
            }
        }

        var exact = fixture
        exact.planProbeTranscripts = probes(lastLength: 65_536)
        Self.assertRefuses(
            exact,
            as: .rangePlan(.requestMismatch(transcriptOrdinal: 0))
        )

        var over = fixture
        over.planProbeTranscripts = probes(lastLength: 65_537)
        Self.assertRefuses(
            over,
            as: .rangePlan(.rawByteLimit(
                actual: 268_435_457,
                maximum: 268_435_456
            ))
        )
    }
}

private extension SyntheticSharedCacheDiscoveryRangePlanTests {
    struct ConsumerBindingKey: Equatable {
        let kind: SyntheticSharedCacheRangeConsumerKind
        let ordinal: Int

        init(_ kind: SyntheticSharedCacheRangeConsumerKind, _ ordinal: Int) {
            self.kind = kind
            self.ordinal = ordinal
        }
    }

    struct Fixture {
        let metadata: SyntheticCaptureFileMetadata
        var discoveryBytes: Data
        let setPlan: SyntheticSharedCacheSetPlanComparison
        var gitExpectation: AnchoredRuntimeClosureExpectationDocument
        var selfGuardExpectation: AnchoredRuntimeClosureExpectationDocument
        var discoveryTranscripts: [SyntheticSharedCacheBoundedReadTranscript]
        var planProbeTranscripts: [SyntheticSharedCacheBoundedReadTranscript]
    }

    struct SignedFixture {
        var fixture: Fixture
        let loadCommands: Data
        let signature: Data
        let signatureCacheFileOffset: UInt64
    }

    static let headerBytes = 552
    static let mappingInfoBytes = 32
    static let mappingWithSlideBytes = 56
    static let mappingWithSlideOffset = headerBytes + mappingWithSlideBytes
    static let mappingTablesEnd = mappingWithSlideOffset + mappingWithSlideBytes
    static let imageTableOffset = mappingTablesEnd
    static let imageTextOffset = imageTableOffset + 32
    static let discoveryByteCount = imageTextOffset + 32
    static let fileBytes: Int64 = 2 * 1_048_576
    static let codeSignatureBytes: UInt64 = 4_096
    static let mappingVMStart: UInt64 = 0x0000_0001_8000_0000
    static let mappingBytes: UInt64 = 1_048_576
    static let nameWindowOffset: UInt64 = 4_096
    static let imageFileOffset: UInt64 = 12_288
    static let imageVMAddress = mappingVMStart + imageFileOffset
    static let imageName = Data("/usr/lib/libFixture.dylib".utf8)
    static let firstSortedImageName = Data("/usr/lib/libA.dylib".utf8)
    static let secondSortedImageName = Data("/usr/lib/libB.dylib".utf8)
    static let imageUUID: Data = {
        var value = Data(repeating: 0, count: 16)
        value[15] = 0x71
        return value
    }()
    static let altImageUUID: Data = {
        var value = Data(repeating: 0, count: 16)
        value[15] = 0x72
        return value
    }()
    static let subcacheUUID: Data = {
        var value = Data(repeating: 0, count: 16)
        value[15] = 0x51
        return value
    }()
    static let headerUUID: Data = {
        var value = Data(repeating: 0, count: 16)
        value[15] = 0x41
        return value
    }()
    static let rootID = String(repeating: "1", count: 64)
    static let loaderID = String(repeating: "2", count: 64)
    static let zeroSHA256 = String(repeating: "0", count: 64)
    static let csMagicEmbeddedSignature: UInt32 = 0xfade0cc0
    static let csMagicCodeDirectory: UInt32 = 0xfade0c02
    static let csMagicBlobWrapper: UInt32 = 0xfade0b01
    static let forbiddenCommands: [UInt32] = [
        0x00000027,
        0x80000028,
        0x0000000e,
        0x00000021,
        0x0000002c,
        0x80000035,
        0x0000000f,
    ]
    static let recognizedCommands: [UInt32] = [
        0x00000001, 0x00000002, 0x00000003, 0x00000004,
        0x00000005, 0x00000006, 0x00000007, 0x00000008,
        0x00000009, 0x0000000a, 0x0000000b, 0x0000000c,
        0x0000000d, 0x0000000e, 0x0000000f, 0x00000010,
        0x00000011, 0x00000012, 0x00000013, 0x00000014,
        0x00000015, 0x00000016, 0x00000017, 0x80000018,
        0x00000019, 0x0000001a, 0x0000001b, 0x8000001c,
        0x0000001d, 0x0000001e, 0x8000001f, 0x00000020,
        0x00000021, 0x00000022, 0x80000022, 0x80000023,
        0x00000024, 0x00000025, 0x00000026, 0x00000027,
        0x80000028, 0x00000029, 0x0000002a, 0x0000002b,
        0x0000002c, 0x0000002d, 0x0000002e, 0x0000002f,
        0x00000030, 0x00000031, 0x00000032, 0x80000033,
        0x80000034, 0x80000035,
    ]
    static let recognizedPassthroughCommands: [UInt32] = {
        let structural: Set<UInt32> = [0x0000000d, 0x00000019,
            0x0000001b, 0x0000001d]
        return recognizedCommands.filter {
            !structural.contains($0) && !forbiddenCommands.contains($0)
        }
    }()

    static func fixture() throws -> Fixture {
        let metadata = metadata()
        let discoveryBytes = discoveryBytes()
        let setPlan = try SyntheticSharedCacheSetPlanVerifier.compare(
            sourceProfile: .reviewed,
            main: SyntheticSharedCacheDiscoveryFile(
                metadata: metadata,
                discoveryBytes: discoveryBytes
            ),
            subcaches: []
        )
        let cacheSet = try cacheSet(metadata: metadata)
        let loadCommands = loadCommands()
        let memberAndEdge = try memberAndEdge(
            cacheSet: cacheSet,
            name: imageName,
            uuid: imageUUID,
            loadCommands: loadCommands,
            primaryCodeDirectory: .absent
        )
        let member = memberAndEdge.member
        let edge = memberAndEdge.edge
        let gitExpectation = try anchoredExpectation(
            role: .git,
            cacheSet: cacheSet,
            member: member,
            edge: edge
        )
        let selfGuardExpectation = try anchoredExpectation(
            role: .selfGuard,
            cacheSet: cacheSet,
            member: member,
            edge: edge
        )

        var nameWindow = Data(repeating: 0, count: 4_097)
        nameWindow.replaceSubrange(0..<imageName.count, with: imageName)
        let machHeader = machHeader(loadCommandBytes: loadCommands.count)
        let probes = [
            probeTranscript(
                kind: .imageNameWindow,
                ordinal: 0,
                start: nameWindowOffset,
                metadata: metadata,
                bytes: nameWindow
            ),
            probeTranscript(
                kind: .machOHeader,
                ordinal: 0,
                start: imageFileOffset,
                metadata: metadata,
                bytes: machHeader
            ),
            probeTranscript(
                kind: .loadCommands,
                ordinal: 0,
                start: imageFileOffset + 32,
                metadata: metadata,
                bytes: loadCommands
            ),
        ]
        return Fixture(
            metadata: metadata,
            discoveryBytes: discoveryBytes,
            setPlan: setPlan,
            gitExpectation: gitExpectation,
            selfGuardExpectation: selfGuardExpectation,
            discoveryTranscripts: [
                discoveryTranscript(
                    metadata: metadata,
                    bytes: discoveryBytes,
                    events: [.bytes(offset: 0, data: discoveryBytes)]
                ),
            ],
            planProbeTranscripts: probes
        )
    }

    static func signedFixture(
        includeLinkedit: Bool = true,
        signatureCacheFileOffset: UInt64 = 32_768
    ) throws -> SignedFixture {
        let primary = codeDirectory(
            hashType: 2,
            flags: 0,
            signingIdentifier: Data("com.example.fixture".utf8),
            teamIdentifier: Data("TEAM123456".utf8)
        )
        let signature = superBlob(entries: [
            (0, primary),
            (
                0x10000,
                genericBlob(
                    magic: csMagicBlobWrapper,
                    payload: Data([0x30, 0x00])
                )
            ),
        ])
        let localSignatureOffset = signatureCacheFileOffset - imageFileOffset
        let loadCommands = signedLoadCommands(
            uuid: imageUUID,
            localSignatureOffset: localSignatureOffset,
            signatureBytes: UInt64(signature.count),
            includeLinkedit: includeLinkedit
        )
        var fixture = try signedFixture(
            loadCommands: loadCommands,
            commandCount: includeLinkedit ? 4 : 3,
            primaryCodeDirectory: .present(blobSHA256: sha256(primary)),
            signatureStart: signatureCacheFileOffset,
            signature: signature
        )
        fixture.planProbeTranscripts.append(
            probeTranscript(
                kind: .codeSignature,
                ordinal: 0,
                start: signatureCacheFileOffset,
                metadata: fixture.metadata,
                bytes: signature
            )
        )
        return SignedFixture(
            fixture: fixture,
            loadCommands: loadCommands,
            signature: signature,
            signatureCacheFileOffset: signatureCacheFileOffset
        )
    }

    static func signedFixture(
        loadCommands: Data,
        commandCount: UInt32,
        primaryCodeDirectory:
            SyntheticSharedCacheImagePrimaryCodeDirectory,
        signatureStart _: UInt64,
        signature _: Data
    ) throws -> Fixture {
        let metadata = metadata()
        let discoveryBytes = discoveryBytes()
        let setPlan = try SyntheticSharedCacheSetPlanVerifier.compare(
            sourceProfile: .reviewed,
            main: SyntheticSharedCacheDiscoveryFile(
                metadata: metadata,
                discoveryBytes: discoveryBytes
            ),
            subcaches: []
        )
        let cacheSet = try cacheSet(metadata: metadata)
        let memberAndEdge = try memberAndEdge(
            cacheSet: cacheSet,
            name: imageName,
            uuid: imageUUID,
            loadCommands: loadCommands,
            primaryCodeDirectory: primaryCodeDirectory
        )
        let machHeader = machHeader(
            commandCount: commandCount,
            loadCommandBytes: loadCommands.count
        )
        var nameWindow = Data(repeating: 0, count: 4_097)
        nameWindow.replaceSubrange(0..<imageName.count, with: imageName)
        return Fixture(
            metadata: metadata,
            discoveryBytes: discoveryBytes,
            setPlan: setPlan,
            gitExpectation: try anchoredExpectation(
                role: .git,
                cacheSet: cacheSet,
                member: memberAndEdge.member,
                edge: memberAndEdge.edge
            ),
            selfGuardExpectation: try anchoredExpectation(
                role: .selfGuard,
                cacheSet: cacheSet,
                member: memberAndEdge.member,
                edge: memberAndEdge.edge
            ),
            discoveryTranscripts: [
                discoveryTranscript(
                    metadata: metadata,
                    bytes: discoveryBytes,
                    events: [.bytes(offset: 0, data: discoveryBytes)]
                ),
            ],
            planProbeTranscripts: [
                probeTranscript(
                    kind: .imageNameWindow,
                    ordinal: 0,
                    start: nameWindowOffset,
                    metadata: metadata,
                    bytes: nameWindow
                ),
                probeTranscript(
                    kind: .machOHeader,
                    ordinal: 0,
                    start: imageFileOffset,
                    metadata: metadata,
                    bytes: machHeader
                ),
                probeTranscript(
                    kind: .loadCommands,
                    ordinal: 0,
                    start: imageFileOffset + 32,
                    metadata: metadata,
                    bytes: loadCommands
                ),
            ]
        )
    }

    static func fixtureWithLoadCommands(
        _ loadCommands: Data,
        commandCount: UInt32,
        primaryCodeDirectory:
            SyntheticSharedCacheImagePrimaryCodeDirectory = .absent
    ) throws -> Fixture {
        try fixtureWith(
            discoveryBytes: discoveryBytes(),
            loadCommands: loadCommands,
            commandCount: commandCount,
            primaryCodeDirectory: primaryCodeDirectory
        )
    }

    static func fixtureWith(
        discoveryBytes: Data,
        loadCommands: Data,
        commandCount: UInt32,
        primaryCodeDirectory:
            SyntheticSharedCacheImagePrimaryCodeDirectory
    ) throws -> Fixture {
        let metadata = metadata()
        let setPlan = try SyntheticSharedCacheSetPlanVerifier.compare(
            sourceProfile: .reviewed,
            main: SyntheticSharedCacheDiscoveryFile(
                metadata: metadata,
                discoveryBytes: discoveryBytes
            ),
            subcaches: []
        )
        let cacheSet = try cacheSet(metadata: metadata)
        let memberAndEdge = try memberAndEdge(
            cacheSet: cacheSet,
            name: imageName,
            uuid: imageUUID,
            loadCommands: loadCommands,
            primaryCodeDirectory: primaryCodeDirectory
        )
        var nameWindow = Data(repeating: 0, count: 4_097)
        nameWindow.replaceSubrange(0..<imageName.count, with: imageName)
        return Fixture(
            metadata: metadata,
            discoveryBytes: discoveryBytes,
            setPlan: setPlan,
            gitExpectation: try anchoredExpectation(
                role: .git,
                cacheSet: cacheSet,
                member: memberAndEdge.member,
                edge: memberAndEdge.edge
            ),
            selfGuardExpectation: try anchoredExpectation(
                role: .selfGuard,
                cacheSet: cacheSet,
                member: memberAndEdge.member,
                edge: memberAndEdge.edge
            ),
            discoveryTranscripts: [
                discoveryTranscript(
                    metadata: metadata,
                    bytes: discoveryBytes,
                    events: [.bytes(offset: 0, data: discoveryBytes)]
                ),
            ],
            planProbeTranscripts: [
                probeTranscript(
                    kind: .imageNameWindow,
                    ordinal: 0,
                    start: nameWindowOffset,
                    metadata: metadata,
                    bytes: nameWindow
                ),
                probeTranscript(
                    kind: .machOHeader,
                    ordinal: 0,
                    start: imageFileOffset,
                    metadata: metadata,
                    bytes: machHeader(
                        commandCount: commandCount,
                        loadCommandBytes: loadCommands.count
                    )
                ),
                probeTranscript(
                    kind: .loadCommands,
                    ordinal: 0,
                    start: imageFileOffset + 32,
                    metadata: metadata,
                    bytes: loadCommands
                ),
            ]
        )
    }

    static func multiMemberFixture(
        gitMemberIndexes: [Int] = [0, 1],
        selfGuardMemberIndexes: [Int] = [0, 1],
        addressAlias: Bool = false
    ) throws -> Fixture {
        let metadata = metadata()
        let lowerImageFileOffset: UInt64 = 20_480
        let higherImageFileOffset: UInt64 = 24_576
        let lowerNameOffset = nameWindowOffset
        let higherNameOffset: UInt64 = 8_192
        var discoveryBytes = discoveryBytes()
        discoveryBytes.append(Data(repeating: 0, count: 64))
        Self.writeUInt32LE(2, into: &discoveryBytes, at: 452)
        Self.writeUInt64LE(
            UInt64(imageTableOffset + 64),
            into: &discoveryBytes,
            at: 136
        )
        Self.writeUInt64LE(2, into: &discoveryBytes, at: 144)
        Self.writeUInt32LE(UInt32(discoveryBytes.count),
            into: &discoveryBytes,
            at: 392)
        writeImageRow(
            into: &discoveryBytes,
            row: 0,
            address: mappingVMStart + lowerImageFileOffset,
            pathFileOffset: lowerNameOffset
        )
        writeImageRow(
            into: &discoveryBytes,
            row: 1,
            address: mappingVMStart +
                (addressAlias ? lowerImageFileOffset : higherImageFileOffset),
            pathFileOffset: higherNameOffset
        )
        let setPlan = try SyntheticSharedCacheSetPlanVerifier.compare(
            sourceProfile: .reviewed,
            main: SyntheticSharedCacheDiscoveryFile(
                metadata: metadata,
                discoveryBytes: discoveryBytes
            ),
            subcaches: []
        )
        let cacheSet = try cacheSet(metadata: metadata)
        let firstCommands = loadCommands(uuid: imageUUID)
        let secondCommands = loadCommands(uuid: altImageUUID)
        let first = try memberAndEdge(
            cacheSet: cacheSet,
            name: firstSortedImageName,
            uuid: imageUUID,
            loadCommands: firstCommands,
            primaryCodeDirectory: .absent
        )
        let second = try memberAndEdge(
            cacheSet: cacheSet,
            name: secondSortedImageName,
            uuid: altImageUUID,
            loadCommands: secondCommands,
            primaryCodeDirectory: .absent
        )
        let members = [first.member, second.member]
        let edges = [first.edge, second.edge]
        var nameWindow = Data(
            repeating: 0,
            count: Int(higherNameOffset - lowerNameOffset + 4_097)
        )
        nameWindow.replaceSubrange(
            0..<secondSortedImageName.count,
            with: secondSortedImageName
        )
        let firstRelative = Int(higherNameOffset - lowerNameOffset)
        nameWindow.replaceSubrange(
            firstRelative..<(firstRelative + firstSortedImageName.count),
            with: firstSortedImageName
        )

        return Fixture(
            metadata: metadata,
            discoveryBytes: discoveryBytes,
            setPlan: setPlan,
            gitExpectation: try anchoredExpectation(
                role: .git,
                cacheSet: cacheSet,
                members: gitMemberIndexes.map { members[$0] },
                edges: gitMemberIndexes.map { edges[$0] }
            ),
            selfGuardExpectation: try anchoredExpectation(
                role: .selfGuard,
                cacheSet: cacheSet,
                members: selfGuardMemberIndexes.map { members[$0] },
                edges: selfGuardMemberIndexes.map { edges[$0] }
            ),
            discoveryTranscripts: [
                discoveryTranscript(
                    metadata: metadata,
                    bytes: discoveryBytes,
                    events: [.bytes(offset: 0, data: discoveryBytes)]
                ),
            ],
            planProbeTranscripts: [
                probeTranscript(
                    kind: .imageNameWindow,
                    ordinal: 0,
                    start: lowerNameOffset,
                    metadata: metadata,
                    bytes: nameWindow
                ),
                probeTranscript(
                    kind: .machOHeader,
                    ordinal: 1,
                    start: lowerImageFileOffset,
                    metadata: metadata,
                    bytes: machHeader(
                        loadCommandBytes: secondCommands.count
                    )
                ),
                probeTranscript(
                    kind: .machOHeader,
                    ordinal: 0,
                    start: higherImageFileOffset,
                    metadata: metadata,
                    bytes: machHeader(
                        loadCommandBytes: firstCommands.count
                    )
                ),
                probeTranscript(
                    kind: .loadCommands,
                    ordinal: 1,
                    start: lowerImageFileOffset + 32,
                    metadata: metadata,
                    bytes: secondCommands
                ),
                probeTranscript(
                    kind: .loadCommands,
                    ordinal: 0,
                    start: higherImageFileOffset + 32,
                    metadata: metadata,
                    bytes: firstCommands
                ),
            ]
        )
    }

    static func subcacheFixture() throws -> Fixture {
        let mainMetadata = metadata()
        var subMetadata = metadata()
        subMetadata.inode += 1
        let subcacheVMOffset: UInt64 = 2 * mappingBytes
        let subcacheImageFileOffset: UInt64 = 12_288
        let subcacheStart = mappingVMStart + subcacheVMOffset
        let mainDiscoveryBytes = mainDiscoveryBytesWithSubcache(
            subcacheVMOffset: subcacheVMOffset,
            subcacheUUID: subcacheUUID,
            subcacheSuffix: Data(".1".utf8),
            imageAddress: subcacheStart + subcacheImageFileOffset
        )
        let subcacheDiscoveryBytes = subcacheDiscoveryBytes(
            uuid: subcacheUUID,
            sharedRegionStart: subcacheStart
        )
        let setPlan = try SyntheticSharedCacheSetPlanVerifier.compare(
            sourceProfile: .reviewed,
            main: SyntheticSharedCacheDiscoveryFile(
                metadata: mainMetadata,
                discoveryBytes: mainDiscoveryBytes
            ),
            subcaches: [
                SyntheticSharedCacheDiscoveryFile(
                    metadata: subMetadata,
                    discoveryBytes: subcacheDiscoveryBytes
                ),
            ]
        )
        let cacheSet = try cacheSet(records: [
            cacheRecord(
                suffix: Data(),
                sha256: String(repeating: "3", count: 64),
                fileBytes: UInt64(mainMetadata.size),
                uuid: headerUUID
            ),
            cacheRecord(
                suffix: Data(".1".utf8),
                sha256: String(repeating: "4", count: 64),
                fileBytes: UInt64(subMetadata.size),
                uuid: subcacheUUID
            ),
        ])
        let loadCommands = loadCommands()
        let memberAndEdge = try memberAndEdge(
            cacheSet: cacheSet,
            name: imageName,
            uuid: imageUUID,
            loadCommands: loadCommands,
            primaryCodeDirectory: .absent
        )
        var nameWindow = Data(repeating: 0, count: 4_097)
        nameWindow.replaceSubrange(0..<imageName.count, with: imageName)

        return Fixture(
            metadata: mainMetadata,
            discoveryBytes: mainDiscoveryBytes,
            setPlan: setPlan,
            gitExpectation: try anchoredExpectation(
                role: .git,
                cacheSet: cacheSet,
                member: memberAndEdge.member,
                edge: memberAndEdge.edge
            ),
            selfGuardExpectation: try anchoredExpectation(
                role: .selfGuard,
                cacheSet: cacheSet,
                member: memberAndEdge.member,
                edge: memberAndEdge.edge
            ),
            discoveryTranscripts: [
                discoveryTranscript(
                    fileOrdinal: 0,
                    metadata: mainMetadata,
                    bytes: mainDiscoveryBytes,
                    events: [.bytes(offset: 0, data: mainDiscoveryBytes)]
                ),
                discoveryTranscript(
                    fileOrdinal: 1,
                    metadata: subMetadata,
                    bytes: subcacheDiscoveryBytes,
                    events: [.bytes(offset: 0, data: subcacheDiscoveryBytes)]
                ),
            ],
            planProbeTranscripts: [
                probeTranscript(
                    kind: .imageNameWindow,
                    ordinal: 0,
                    fileOrdinal: 0,
                    start: nameWindowOffset,
                    metadata: mainMetadata,
                    bytes: nameWindow
                ),
                probeTranscript(
                    kind: .machOHeader,
                    ordinal: 0,
                    fileOrdinal: 1,
                    start: subcacheImageFileOffset,
                    metadata: subMetadata,
                    bytes: machHeader(loadCommandBytes: loadCommands.count)
                ),
                probeTranscript(
                    kind: .loadCommands,
                    ordinal: 0,
                    fileOrdinal: 1,
                    start: subcacheImageFileOffset + 32,
                    metadata: subMetadata,
                    bytes: loadCommands
                ),
            ]
        )
    }

    static func compare(_ fixture: Fixture) throws
        -> SyntheticSharedCacheDiscoveryRangePlanComparison
    {
        try SyntheticSharedCacheDiscoveryRangePlanVerifier.compare(
            setPlan: fixture.setPlan,
            gitExpectation: fixture.gitExpectation,
            selfGuardExpectation: fixture.selfGuardExpectation,
            discoveryTranscripts: fixture.discoveryTranscripts,
            planProbeTranscripts: fixture.planProbeTranscripts
        )
    }

    static func assertRefuses(
        _ fixture: Fixture,
        as expected: SyntheticSharedCacheDiscoveryRangePlanFailure,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(
            try compare(fixture),
            file: file,
            line: line
        ) { error in
            XCTAssertEqual(
                error as? SyntheticSharedCacheDiscoveryRangePlanFailure,
                expected,
                file: file,
                line: line
            )
        }
    }

    static func discoveryTranscript(
        fileOrdinal: Int = 0,
        metadata: SyntheticCaptureFileMetadata,
        bytes: Data,
        events: [SyntheticCaptureReadEvent]
    ) -> SyntheticSharedCacheBoundedReadTranscript {
        SyntheticSharedCacheBoundedReadTranscript(
            purpose: .setHeaderDiscovery,
            fileOrdinal: fileOrdinal,
            rangeStart: 0,
            requestedByteCount: UInt64(bytes.count),
            beforeMetadata: metadata,
            afterMetadata: metadata,
            events: events
        )
    }

    static func probeTranscript(
        kind: SyntheticSharedCacheRangeConsumerKind,
        ordinal: Int,
        fileOrdinal: Int = 0,
        start: UInt64,
        metadata: SyntheticCaptureFileMetadata,
        bytes: Data
    ) -> SyntheticSharedCacheBoundedReadTranscript {
        SyntheticSharedCacheBoundedReadTranscript(
            purpose: .planProbe(
                consumerKind: kind,
                consumerOrdinal: ordinal
            ),
            fileOrdinal: fileOrdinal,
            rangeStart: start,
            requestedByteCount: UInt64(bytes.count),
            beforeMetadata: metadata,
            afterMetadata: metadata,
            events: [.bytes(offset: start, data: bytes)]
        )
    }

    static func fragmentEvents(
        _ bytes: Data,
        fragments: Int
    ) -> [SyntheticCaptureReadEvent] {
        fragmentEvents(bytes, fragments: fragments, offsetBase: 0)
    }

    static func fragmentEvents(
        _ bytes: Data,
        fragments: Int,
        offsetBase: UInt64
    ) -> [SyntheticCaptureReadEvent] {
        precondition(fragments > 0 && fragments <= bytes.count)
        var events: [SyntheticCaptureReadEvent] = []
        var cursor = 0
        for index in 0..<fragments {
            let remainingBytes = bytes.count - cursor
            let remainingFragments = fragments - index
            let count = remainingBytes / remainingFragments
            let data = Data(bytes[cursor..<(cursor + count)])
            events.append(.bytes(
                offset: offsetBase + UInt64(cursor),
                data: data
            ))
            cursor += count
        }
        return events
    }

    static func bytes(
        from transcript: SyntheticSharedCacheBoundedReadTranscript
    ) -> Data {
        for event in transcript.events {
            if case let .bytes(_, data) = event { return data }
        }
        return Data()
    }

    static func replacingBytes(
        _ transcript: SyntheticSharedCacheBoundedReadTranscript,
        with bytes: Data
    ) -> SyntheticSharedCacheBoundedReadTranscript {
        replacingEvents(
            transcript,
            with: [.bytes(offset: transcript.rangeStart, data: bytes)]
        )
    }

    static func replacingEvents(
        _ transcript: SyntheticSharedCacheBoundedReadTranscript,
        with events: [SyntheticCaptureReadEvent]
    ) -> SyntheticSharedCacheBoundedReadTranscript {
        SyntheticSharedCacheBoundedReadTranscript(
            purpose: transcript.purpose,
            fileOrdinal: transcript.fileOrdinal,
            rangeStart: transcript.rangeStart,
            requestedByteCount: transcript.requestedByteCount,
            beforeMetadata: transcript.beforeMetadata,
            afterMetadata: transcript.afterMetadata,
            events: events
        )
    }

    static func replacingMetadata(
        _ transcript: SyntheticSharedCacheBoundedReadTranscript,
        before: SyntheticCaptureFileMetadata,
        after: SyntheticCaptureFileMetadata
    ) -> SyntheticSharedCacheBoundedReadTranscript {
        SyntheticSharedCacheBoundedReadTranscript(
            purpose: transcript.purpose,
            fileOrdinal: transcript.fileOrdinal,
            rangeStart: transcript.rangeStart,
            requestedByteCount: transcript.requestedByteCount,
            beforeMetadata: before,
            afterMetadata: after,
            events: transcript.events
        )
    }

    static func discoveryBytes() -> Data {
        var bytes = Data(repeating: 0, count: discoveryByteCount)
        bytes.replaceSubrange(
            0..<16,
            with: Data([
                0x64, 0x79, 0x6c, 0x64, 0x5f, 0x76, 0x31, 0x20,
                0x20, 0x20, 0x61, 0x72, 0x6d, 0x36, 0x34, 0x00,
            ])
        )
        writeUInt32LE(UInt32(headerBytes), into: &bytes, at: 16)
        writeUInt32LE(1, into: &bytes, at: 20)
        writeUInt64LE(
            UInt64(fileBytes) - codeSignatureBytes,
            into: &bytes,
            at: 40
        )
        writeUInt64LE(codeSignatureBytes, into: &bytes, at: 48)
        bytes.replaceSubrange(88..<104, with: headerUUID)
        writeUInt64LE(2, into: &bytes, at: 104)
        writeUInt32LE(1, into: &bytes, at: 216)
        writeUInt32LE(1 << 12, into: &bytes, at: 220)
        writeUInt64LE(mappingVMStart, into: &bytes, at: 224)
        writeUInt64LE(mappingBytes, into: &bytes, at: 232)
        writeUInt32LE(
            UInt32(mappingWithSlideOffset),
            into: &bytes,
            at: 312
        )
        writeUInt32LE(1, into: &bytes, at: 316)
        writeUInt32LE(UInt32(imageTableOffset), into: &bytes, at: 448)
        writeUInt32LE(1, into: &bytes, at: 452)
        writeUInt64LE(UInt64(imageTextOffset), into: &bytes, at: 136)
        writeUInt64LE(1, into: &bytes, at: 144)
        writeUInt32LE(UInt32(discoveryByteCount), into: &bytes, at: 392)
        writeUInt32LE(0, into: &bytes, at: 396)
        writeUInt32LE(1, into: &bytes, at: 456)

        writeUInt64LE(mappingVMStart, into: &bytes, at: headerBytes)
        writeUInt64LE(mappingBytes, into: &bytes, at: headerBytes + 8)
        writeUInt64LE(0, into: &bytes, at: headerBytes + 16)
        writeUInt32LE(5, into: &bytes, at: headerBytes + 24)
        writeUInt32LE(5, into: &bytes, at: headerBytes + 28)

        writeUInt64LE(mappingVMStart, into: &bytes, at: mappingWithSlideOffset)
        writeUInt64LE(
            mappingBytes,
            into: &bytes,
            at: mappingWithSlideOffset + 8
        )
        writeUInt64LE(0, into: &bytes, at: mappingWithSlideOffset + 16)
        writeUInt64LE(0, into: &bytes, at: mappingWithSlideOffset + 24)
        writeUInt64LE(0, into: &bytes, at: mappingWithSlideOffset + 32)
        writeUInt64LE(0, into: &bytes, at: mappingWithSlideOffset + 40)
        writeUInt32LE(5, into: &bytes, at: mappingWithSlideOffset + 48)
        writeUInt32LE(5, into: &bytes, at: mappingWithSlideOffset + 52)

        writeUInt64LE(imageVMAddress, into: &bytes, at: imageTableOffset)
        writeUInt64LE(0, into: &bytes, at: imageTableOffset + 8)
        writeUInt64LE(0, into: &bytes, at: imageTableOffset + 16)
        writeUInt32LE(UInt32(nameWindowOffset), into: &bytes,
            at: imageTableOffset + 24)
        writeUInt32LE(0, into: &bytes, at: imageTableOffset + 28)
        return bytes
    }

    static func mainDiscoveryBytesWithSubcache(
        subcacheVMOffset: UInt64,
        subcacheUUID: Data,
        subcacheSuffix: Data,
        imageAddress: UInt64
    ) -> Data {
        var bytes = discoveryBytes()
        bytes.append(Data(repeating: 0, count: 56))
        writeUInt64LE(3 * mappingBytes, into: &bytes, at: 232)
        writeUInt32LE(UInt32(discoveryByteCount), into: &bytes, at: 392)
        writeUInt32LE(1, into: &bytes, at: 396)
        writeUInt64LE(imageAddress, into: &bytes, at: imageTableOffset)
        let entryOffset = discoveryByteCount
        bytes.replaceSubrange(
            entryOffset..<(entryOffset + 16),
            with: subcacheUUID
        )
        writeUInt64LE(subcacheVMOffset, into: &bytes, at: entryOffset + 16)
        bytes.replaceSubrange(
            (entryOffset + 24)..<(entryOffset + 24 + subcacheSuffix.count),
            with: subcacheSuffix
        )
        return bytes
    }

    static func subcacheDiscoveryBytes(
        uuid: Data,
        sharedRegionStart: UInt64
    ) -> Data {
        var bytes = Data(repeating: 0, count: mappingTablesEnd)
        bytes.replaceSubrange(
            0..<16,
            with: Data([
                0x64, 0x79, 0x6c, 0x64, 0x5f, 0x76, 0x31, 0x20,
                0x20, 0x20, 0x61, 0x72, 0x6d, 0x36, 0x34, 0x00,
            ])
        )
        writeUInt32LE(UInt32(headerBytes), into: &bytes, at: 16)
        writeUInt32LE(1, into: &bytes, at: 20)
        writeUInt64LE(
            UInt64(fileBytes) - codeSignatureBytes,
            into: &bytes,
            at: 40
        )
        writeUInt64LE(codeSignatureBytes, into: &bytes, at: 48)
        bytes.replaceSubrange(88..<104, with: uuid)
        writeUInt64LE(2, into: &bytes, at: 104)
        writeUInt32LE(1, into: &bytes, at: 216)
        writeUInt32LE(1 << 12, into: &bytes, at: 220)
        writeUInt64LE(sharedRegionStart, into: &bytes, at: 224)
        writeUInt64LE(mappingBytes, into: &bytes, at: 232)
        writeUInt32LE(
            UInt32(mappingWithSlideOffset),
            into: &bytes,
            at: 312
        )
        writeUInt32LE(1, into: &bytes, at: 316)
        writeUInt32LE(1, into: &bytes, at: 456)

        writeUInt64LE(sharedRegionStart, into: &bytes, at: headerBytes)
        writeUInt64LE(mappingBytes, into: &bytes, at: headerBytes + 8)
        writeUInt64LE(0, into: &bytes, at: headerBytes + 16)
        writeUInt32LE(5, into: &bytes, at: headerBytes + 24)
        writeUInt32LE(5, into: &bytes, at: headerBytes + 28)

        writeUInt64LE(
            sharedRegionStart,
            into: &bytes,
            at: mappingWithSlideOffset
        )
        writeUInt64LE(
            mappingBytes,
            into: &bytes,
            at: mappingWithSlideOffset + 8
        )
        writeUInt64LE(0, into: &bytes, at: mappingWithSlideOffset + 16)
        writeUInt64LE(0, into: &bytes, at: mappingWithSlideOffset + 24)
        writeUInt64LE(0, into: &bytes, at: mappingWithSlideOffset + 32)
        writeUInt64LE(0, into: &bytes, at: mappingWithSlideOffset + 40)
        writeUInt32LE(5, into: &bytes, at: mappingWithSlideOffset + 48)
        writeUInt32LE(5, into: &bytes, at: mappingWithSlideOffset + 52)
        return bytes
    }

    static func machHeader(
        commandCount: UInt32 = 2,
        loadCommandBytes: Int
    ) -> Data {
        var bytes = Data(repeating: 0, count: 32)
        writeUInt32LE(0xfeedfacf, into: &bytes, at: 0)
        writeUInt32LE(0x0100000c, into: &bytes, at: 4)
        writeUInt32LE(0, into: &bytes, at: 8)
        writeUInt32LE(0x6, into: &bytes, at: 12)
        writeUInt32LE(commandCount, into: &bytes, at: 16)
        writeUInt32LE(UInt32(loadCommandBytes), into: &bytes, at: 20)
        writeUInt32LE(0, into: &bytes, at: 24)
        writeUInt32LE(0, into: &bytes, at: 28)
        return bytes
    }

    static func loadCommands(uuid: Data = imageUUID) -> Data {
        uuidCommand(uuid) + idDylibCommand()
    }

    static func loadCommands(prefixedBy commands: [(UInt32, Int)]) -> Data {
        var bytes = uuidCommand(imageUUID)
        for (command, size) in commands {
            bytes.append(passthroughCommand(command, size: size))
        }
        bytes.append(idDylibCommand())
        return bytes
    }

    static func uuidCommand(_ uuid: Data) -> Data {
        var bytes = Data()
        appendUInt32LE(&bytes, 0x1b)
        appendUInt32LE(&bytes, 24)
        bytes.append(uuid)
        return bytes
    }

    static func idDylibCommand() -> Data {
        var bytes = Data(repeating: 0, count: 32)
        writeUInt32LE(0x0d, into: &bytes, at: 0)
        writeUInt32LE(32, into: &bytes, at: 4)
        writeUInt32LE(24, into: &bytes, at: 8)
        writeUInt32LE(0, into: &bytes, at: 12)
        writeUInt32LE(1, into: &bytes, at: 16)
        writeUInt32LE(1, into: &bytes, at: 20)
        bytes[24] = 0x78
        return bytes
    }

    static func passthroughCommand(
        _ command: UInt32,
        size: Int
    ) -> Data {
        var bytes = Data(repeating: 0, count: size)
        writeUInt32LE(command, into: &bytes, at: 0)
        writeUInt32LE(UInt32(size), into: &bytes, at: 4)
        return bytes
    }

    static func maximalLoadCommands() -> Data {
        var bytes = uuidCommand(imageUUID)
        for _ in 0..<253 {
            bytes.append(passthroughCommand(0x0000002a, size: 8))
        }
        let finalPassthroughSize = 262_144 - bytes.count -
            idDylibCommand().count
        bytes.append(passthroughCommand(
            0x0000002a,
            size: finalPassthroughSize
        ))
        bytes.append(idDylibCommand())
        return bytes
    }

    static func signedLoadCommands(
        uuid: Data,
        localSignatureOffset: UInt64,
        signatureBytes: UInt64,
        includeLinkedit: Bool
    ) -> Data {
        var bytes = Data()
        appendUInt32LE(&bytes, 0x1b)
        appendUInt32LE(&bytes, 24)
        bytes.append(uuid)

        if includeLinkedit {
            var segment = Data(repeating: 0, count: 72)
            writeUInt32LE(0x19, into: &segment, at: 0)
            writeUInt32LE(72, into: &segment, at: 4)
            segment.replaceSubrange(
                8..<18,
                with: Data("__LINKEDIT".utf8)
            )
            writeUInt64LE(
                imageVMAddress + localSignatureOffset,
                into: &segment,
                at: 24
            )
            writeUInt64LE(signatureBytes, into: &segment, at: 32)
            writeUInt64LE(localSignatureOffset, into: &segment, at: 40)
            writeUInt64LE(signatureBytes, into: &segment, at: 48)
            writeUInt32LE(1, into: &segment, at: 56)
            writeUInt32LE(1, into: &segment, at: 60)
            bytes.append(segment)
        }

        bytes.append(contentsOf: loadCommands(uuid: uuid).dropFirst(24))

        appendUInt32LE(&bytes, 0x1d)
        appendUInt32LE(&bytes, 16)
        appendUInt32LE(&bytes, UInt32(localSignatureOffset))
        appendUInt32LE(&bytes, UInt32(signatureBytes))
        return bytes
    }

    static func writeImageRow(
        into bytes: inout Data,
        row: Int,
        address: UInt64,
        pathFileOffset: UInt64
    ) {
        let offset = imageTableOffset + row * 32
        writeUInt64LE(address, into: &bytes, at: offset)
        writeUInt64LE(0, into: &bytes, at: offset + 8)
        writeUInt64LE(0, into: &bytes, at: offset + 16)
        writeUInt32LE(UInt32(pathFileOffset), into: &bytes, at: offset + 24)
        writeUInt32LE(0, into: &bytes, at: offset + 28)
    }

    static func cacheSet(
        metadata _: SyntheticCaptureFileMetadata
    ) throws -> SyntheticSharedCacheSetIdentityEvidence {
        try cacheSet(records: [
            cacheRecord(
                suffix: Data(),
                sha256: String(repeating: "3", count: 64),
                fileBytes: UInt64(fileBytes),
                uuid: headerUUID
            ),
        ])
    }

    static func cacheSet(
        records: [SyntheticSharedCacheFileRecord]
    ) throws -> SyntheticSharedCacheSetIdentityEvidence {
        try SyntheticSharedCacheSetIdentityVerifier.derive(records: records)
    }

    static func cacheRecord(
        suffix: Data,
        sha256: String,
        fileBytes: UInt64,
        uuid: Data
    ) -> SyntheticSharedCacheFileRecord {
        SyntheticSharedCacheFileRecord(
            suffixBytes: UInt64(suffix.count),
            suffixBase64URL: base64URL(suffix),
            fileSHA256: sha256,
            fileBytes: fileBytes,
            headerUUID: hex(uuid)
        )
    }

    static func memberAndEdge(
        cacheSet: SyntheticSharedCacheSetIdentityEvidence,
        name: Data,
        uuid: Data,
        loadCommands: Data,
        primaryCodeDirectory:
            SyntheticSharedCacheImagePrimaryCodeDirectory
    ) throws -> (
        member: RuntimeClosureExpectationMemberFields,
        edge: RuntimeClosureExpectationEdgeFields
    ) {
        let installName = SyntheticRuntimeClosureInstallName(
            bytes: UInt64(name.count),
            base64URL: base64URL(name)
        )
        let imageEvidence = try
            SyntheticSharedCacheImageContentIdentityVerifier.derive(
                cacheSetEvidence: cacheSet,
                facts: SyntheticSharedCacheImageContentFacts(
                    installNameBytes: installName.bytes,
                    installNameBase64URL: installName.base64URL,
                    machOUUID: hex(uuid),
                    primaryCodeDirectory: primaryCodeDirectory,
                    loadCommandsSHA256: sha256(loadCommands)
                )
            )
        let decoded = try SyntheticRuntimeClosureInstallNameVerifier
            .validate(installName)
        let member = RuntimeClosureExpectationMemberFields(
            contentEvidenceID: imageEvidence.contentEvidenceID.sha256,
            storage: .sharedCache,
            installName: installName,
            decodedInstallName: decoded,
            machOUUID: hex(uuid),
            primaryCodeDirectoryBlobSHA256:
                imageEvidence.primaryCodeDirectoryBlobSHA256,
            loadCommandsSHA256: sha256(loadCommands)
        )
        return (member, edge(member))
    }

    static func edge(
        _ member: RuntimeClosureExpectationMemberFields
    ) -> RuntimeClosureExpectationEdgeFields {
        RuntimeClosureExpectationEdgeFields(
            parentContentEvidenceID: member.contentEvidenceID,
            loadCommandOrdinal: 0,
            kind: .load,
            installName: member.installName,
            decodedInstallName: member.decodedInstallName,
            resolvedContentEvidenceID: member.contentEvidenceID
        )
    }

    static func anchoredExpectation(
        role: RuntimeClosureExpectationArtifactRole,
        cacheSet: SyntheticSharedCacheSetIdentityEvidence,
        member: RuntimeClosureExpectationMemberFields,
        edge: RuntimeClosureExpectationEdgeFields
    ) throws -> AnchoredRuntimeClosureExpectationDocument {
        let bytes = renderExpectation(
            role: role,
            cacheSet: cacheSet,
            members: [member],
            edges: [edge]
        )
        return try anchoredExpectation(role: role, bytes: bytes)
    }

    static func anchoredExpectation(
        role: RuntimeClosureExpectationArtifactRole,
        cacheSet: SyntheticSharedCacheSetIdentityEvidence,
        members: [RuntimeClosureExpectationMemberFields],
        edges: [RuntimeClosureExpectationEdgeFields]
    ) throws -> AnchoredRuntimeClosureExpectationDocument {
        let bytes = renderExpectation(
            role: role,
            cacheSet: cacheSet,
            members: members,
            edges: edges
        )
        return try anchoredExpectation(role: role, bytes: bytes)
    }

    static func anchoredExpectation(
        role: RuntimeClosureExpectationArtifactRole,
        bytes: Data
    ) throws -> AnchoredRuntimeClosureExpectationDocument {
        let file = admittedFile(bytes)
        let anchor = RuntimeClosureExpectationTrustAnchor(
            expectedCurrentDocumentSHA256: file.sha256,
            expectedCurrentDocumentBytes: UInt64(file.bytes.count),
            minimumEvidenceGeneration: 9,
            verificationUnixSeconds: 2_000_000_000,
            expectedArtifactRole: role
        )
        return try RuntimeClosureExpectationVerifier.anchor(
            expectationFile: file,
            trustAnchor: anchor
        )
    }

    static func renderExpectation(
        role: RuntimeClosureExpectationArtifactRole,
        cacheSet: SyntheticSharedCacheSetIdentityEvidence,
        member: RuntimeClosureExpectationMemberFields,
        edge: RuntimeClosureExpectationEdgeFields
    ) -> Data {
        renderExpectation(
            role: role,
            cacheSet: cacheSet,
            members: [member],
            edges: [edge]
        )
    }

    static func renderExpectation(
        role: RuntimeClosureExpectationArtifactRole,
        cacheSet: SyntheticSharedCacheSetIdentityEvidence,
        members: [RuntimeClosureExpectationMemberFields],
        edges: [RuntimeClosureExpectationEdgeFields]
    ) -> Data {
        var lines = [
            RuntimeClosureExpectationVerifier.documentDomain,
            "subject=absorbed-mla-source-import-runtime-closure-identity",
            "evidence_generation=9",
            "valid_from_unix_seconds=1900000000",
            "valid_until_unix_seconds=2100000000",
            "artifact_role=\(role.rawValue)",
            "platform_architecture=arm64",
            "platform_hardware_model=Mac15,14",
            "platform_os_version=26.5.2",
            "platform_os_build=25F84",
            "resolution_profile=absolute-static-graph-v1",
            "environment_profile=no-dyld-environment-v1",
            "root_executable_content_evidence_id=\(rootID)",
            "dynamic_loader_content_evidence_id=\(loaderID)",
            "shared_cache_file_count=\(cacheSet.records.count)",
        ]
        for (index, record) in cacheSet.records.enumerated() {
            let ordinal = String(format: "%04d", index)
            lines.append(
                "shared_cache_file_\(ordinal)_suffix_bytes=\(record.suffixBytes)"
            )
            lines.append(
                "shared_cache_file_\(ordinal)_suffix_base64url=\(record.suffixBase64URL)"
            )
            lines.append(
                "shared_cache_file_\(ordinal)_sha256=\(record.fileSHA256)"
            )
            lines.append(
                "shared_cache_file_\(ordinal)_bytes=\(record.fileBytes)"
            )
            lines.append(
                "shared_cache_file_\(ordinal)_header_uuid=\(record.headerUUID)"
            )
        }
        lines.append(contentsOf: [
            "shared_cache_set_id=\(cacheSet.sharedCacheSetID.sha256)",
        ])
        lines.append("member_count=\(members.count)")
        for (index, member) in members.enumerated() {
            let ordinal = String(format: "%04d", index)
            lines.append(
                "member_\(ordinal)_content_evidence_id=\(member.contentEvidenceID)"
            )
            lines.append("member_\(ordinal)_storage=\(member.storage.rawValue)")
            lines.append(
                "member_\(ordinal)_install_name_bytes=\(member.installName.bytes)"
            )
            lines.append(
                "member_\(ordinal)_install_name_base64url=\(member.installName.base64URL)"
            )
            lines.append("member_\(ordinal)_macho_uuid=\(member.machOUUID)")
            lines.append(
                "member_\(ordinal)_primary_code_directory_blob_sha256=\(member.primaryCodeDirectoryBlobSHA256)"
            )
            lines.append(
                "member_\(ordinal)_load_commands_sha256=\(member.loadCommandsSHA256)"
            )
        }
        lines.append("edge_count=\(edges.count)")
        for (index, edge) in edges.enumerated() {
            let ordinal = String(format: "%04d", index)
            lines.append(
                "edge_\(ordinal)_parent_content_evidence_id=\(edge.parentContentEvidenceID)"
            )
            lines.append(
                "edge_\(ordinal)_load_command_ordinal=\(edge.loadCommandOrdinal)"
            )
            lines.append("edge_\(ordinal)_kind=\(edge.kind.rawValue)")
            lines.append(
                "edge_\(ordinal)_install_name_bytes=\(edge.installName.bytes)"
            )
            lines.append(
                "edge_\(ordinal)_install_name_base64url=\(edge.installName.base64URL)"
            )
            lines.append(
                "edge_\(ordinal)_resolved_content_evidence_id=\(edge.resolvedContentEvidenceID)"
            )
        }
        lines.append(
            "runtime_resolution_outcome=unproved-static-comparison-only"
        )
        lines.append("runtime_authority=none")
        return Data((lines.joined(separator: "\n") + "\n").utf8)
    }

    static func admittedFile(_ bytes: Data) -> AdmittedFile {
        AdmittedFile(
            bytes: bytes,
            sha256: sha256(bytes),
            identity: AdmittedFileIdentity(
                device: 0,
                inode: 0,
                size: UInt64(bytes.count),
                linkCount: 1,
                mode: 0,
                modificationSeconds: 0,
                modificationNanoseconds: 0,
                changeSeconds: 0,
                changeNanoseconds: 0
            )
        )
    }

    static func metadata() -> SyntheticCaptureFileMetadata {
        SyntheticCaptureFileMetadata(
            device: 7,
            inode: 10_000,
            mode: 0o100644,
            linkCount: 1,
            userID: 501,
            groupID: 20,
            size: fileBytes,
            blockCount: fileBytes / 512,
            blockSize: 4_096,
            flags: 0,
            generation: 1,
            modificationTimeSeconds: 1_900_000_000,
            modificationTimeNanoseconds: 0,
            statusChangeTimeSeconds: 1_900_000_000,
            statusChangeTimeNanoseconds: 0,
            birthTimeSeconds: 1_800_000_000,
            birthTimeNanoseconds: 0,
            extendedAttributeSupportMask: 1,
            extendedFlags: 0,
            cloneID: nil,
            cloneReferenceCount: nil
        )
    }

    static func writeUInt32LE(
        _ value: UInt32,
        into bytes: inout Data,
        at offset: Int
    ) {
        for index in 0..<4 {
            bytes[offset + index] = UInt8(
                truncatingIfNeeded: value >> UInt32(index * 8)
            )
        }
    }

    static func writeUInt64LE(
        _ value: UInt64,
        into bytes: inout Data,
        at offset: Int
    ) {
        for index in 0..<8 {
            bytes[offset + index] = UInt8(
                truncatingIfNeeded: value >> UInt64(index * 8)
            )
        }
    }

    static func appendUInt32LE(_ data: inout Data, _ value: UInt32) {
        data.append(UInt8(truncatingIfNeeded: value))
        data.append(UInt8(truncatingIfNeeded: value >> 8))
        data.append(UInt8(truncatingIfNeeded: value >> 16))
        data.append(UInt8(truncatingIfNeeded: value >> 24))
    }

    static func appendUInt32BE(_ data: inout Data, _ value: UInt32) {
        data.append(UInt8(truncatingIfNeeded: value >> 24))
        data.append(UInt8(truncatingIfNeeded: value >> 16))
        data.append(UInt8(truncatingIfNeeded: value >> 8))
        data.append(UInt8(truncatingIfNeeded: value))
    }

    static func writeUInt32BE(
        _ value: UInt32,
        into bytes: inout Data,
        at offset: Int
    ) {
        bytes[offset] = UInt8(truncatingIfNeeded: value >> 24)
        bytes[offset + 1] = UInt8(truncatingIfNeeded: value >> 16)
        bytes[offset + 2] = UInt8(truncatingIfNeeded: value >> 8)
        bytes[offset + 3] = UInt8(truncatingIfNeeded: value)
    }

    static func superBlob(entries: [(UInt32, Data)]) -> Data {
        let indexBytes = entries.count * 8
        var nextOffset = 12 + indexBytes
        var result = Data()
        appendUInt32BE(&result, csMagicEmbeddedSignature)
        appendUInt32BE(&result, 0)
        appendUInt32BE(&result, UInt32(entries.count))
        for (slot, blob) in entries {
            appendUInt32BE(&result, slot)
            appendUInt32BE(&result, UInt32(nextOffset))
            nextOffset += blob.count
        }
        for (_, blob) in entries {
            result.append(blob)
        }
        writeUInt32BE(UInt32(result.count), into: &result, at: 4)
        return result
    }

    static func genericBlob(
        magic: UInt32,
        payload: Data
    ) -> Data {
        var result = Data()
        appendUInt32BE(&result, magic)
        appendUInt32BE(&result, UInt32(8 + payload.count))
        result.append(payload)
        return result
    }

    static func codeDirectory(
        version: UInt32 = 0x20200,
        hashType: UInt8,
        hashSize: UInt8? = nil,
        flags: UInt32,
        signingIdentifier: Data,
        teamIdentifier: Data
    ) -> Data {
        let fixedBytes: Int
        switch version {
        case ..<0x20100:
            fixedBytes = 44
        case ..<0x20200:
            fixedBytes = 48
        case ..<0x20300:
            fixedBytes = 52
        case ..<0x20400:
            fixedBytes = 64
        case ..<0x20500:
            fixedBytes = 88
        case ..<0x20600:
            fixedBytes = 96
        default:
            fixedBytes = 108
        }

        var result = Data(repeating: 0, count: fixedBytes)
        let identifierOffset = result.count
        result.append(signingIdentifier)
        result.append(0)
        let teamOffset: Int
        if version >= 0x20200, !teamIdentifier.isEmpty {
            teamOffset = result.count
            result.append(teamIdentifier)
            result.append(0)
        } else {
            teamOffset = 0
        }
        let hashOffset = result.count

        writeUInt32BE(csMagicCodeDirectory, into: &result, at: 0)
        writeUInt32BE(UInt32(result.count), into: &result, at: 4)
        writeUInt32BE(version, into: &result, at: 8)
        writeUInt32BE(flags, into: &result, at: 12)
        writeUInt32BE(UInt32(hashOffset), into: &result, at: 16)
        writeUInt32BE(UInt32(identifierOffset), into: &result, at: 20)
        writeUInt32BE(0, into: &result, at: 24)
        writeUInt32BE(0, into: &result, at: 28)
        writeUInt32BE(0, into: &result, at: 32)
        result[36] = hashSize ?? expectedHashSize(hashType)
        result[37] = hashType
        result[38] = 0
        result[39] = 0
        writeUInt32BE(0, into: &result, at: 40)
        if version >= 0x20100 {
            writeUInt32BE(0, into: &result, at: 44)
        }
        if version >= 0x20200 {
            writeUInt32BE(UInt32(teamOffset), into: &result, at: 48)
        }
        return result
    }

    static func expectedHashSize(_ hashType: UInt8) -> UInt8 {
        switch hashType {
        case 1:
            20
        case 2:
            32
        case 3:
            20
        case 4:
            48
        default:
            0
        }
    }

    static func sha256(_ bytes: Data) -> String {
        SHA256.hash(data: bytes)
            .map { String(format: "%02x", $0) }
            .joined()
    }

    static func hex(_ bytes: Data) -> String {
        bytes.map { String(format: "%02x", $0) }.joined()
    }

    static func base64URL(_ bytes: Data) -> String {
        bytes.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    static func assertNoAuthority(
        _ result: SyntheticSharedCacheDiscoveryRangePlanComparison,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertFalse(result.canExecute, file: file, line: line)
        XCTAssertFalse(result.canSpawn, file: file, line: line)
        XCTAssertFalse(result.canAccessNetwork, file: file, line: line)
        XCTAssertFalse(result.canConsumePack, file: file, line: line)
        XCTAssertFalse(result.canMutateFileSystem, file: file, line: line)
        XCTAssertFalse(result.canImportGitObjects, file: file, line: line)
        XCTAssertFalse(result.canBuild, file: file, line: line)
        XCTAssertFalse(result.canLoadModel, file: file, line: line)
        XCTAssertFalse(result.canReserveOutput, file: file, line: line)
        XCTAssertFalse(result.canPublish, file: file, line: line)
    }

    /*
     Construction-seal and forbidden-conversion compiler probes intentionally
     remain outside the compiled XCTest suite. They should fail to compile:

     _ = SyntheticSharedCacheDiscoveryRangePlanComparison()
     _ = SyntheticSharedCacheSetPlanComparison(result)
     _ = SyntheticSharedCacheSetIdentityEvidence(result)
     _ = SyntheticSharedCacheImageContentFacts(result)
     _ = SyntheticSharedCacheImageContentIdentityEvidence(result)
     _ = SharedCacheImageContentEvidenceID(result)
     _ = SyntheticFileImageMachOIdentityComparison(result)
     _ = FileImageContentIdentityEvidence(result)
     _ = AnchoredRuntimeClosureExpectationDocument(result)
     _ = AdmittedFile(result)
     */
}
