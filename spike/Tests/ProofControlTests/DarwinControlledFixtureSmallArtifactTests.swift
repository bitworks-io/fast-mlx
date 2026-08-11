import CryptoKit
import Darwin
import Dispatch
import Foundation
import XCTest
@testable import ProofControl

final class DarwinControlledFixtureSmallArtifactTests: XCTestCase {
    func testFutureBoundaryReferencesExactlyFourMissingTypes() {
        let futureTypes: [Any.Type] = [
            DarwinControlledFixtureRootExpectation.self,
            DarwinControlledFixtureTestControl.self,
            DarwinControlledFixtureSmallArtifactFailure.self,
            DarwinControlledFixtureSmallArtifactVerifier.self,
        ]

        XCTAssertEqual(futureTypes.count, 4)
        XCTAssertTrue(
            futureTypes.map { String(reflecting: $0) }
                .allSatisfy { $0.contains("DarwinControlledFixture") }
        )
        // Removed probes: no default/memberwise root construction, no product test-control
        // construction, no direct E1 comparison initializer, no AdmittedFile,
        // executable, loader, file-image, shared-cache, closure, P3, runtime,
        // locator, FD, callback, reopen, or release-module conversion surface.
    }

    func testGoldenFixtureHelperBuildsCompleteRootExpectationFromDescriptorStat()
        throws
    {
        let fixture = try Self.makeFixture(bytes: Self.goldenBytes)
        defer { Self.removeFixtureRoot(fixture.rootURL) }

        XCTAssertEqual(fixture.nonce.count, 32)
        XCTAssertTrue(fixture.nonce.allSatisfy { $0.isLowercaseHex })
        XCTAssertEqual(
            fixture.rootURL.lastPathComponent,
            "fast-mlx-e3a-\(fixture.nonce)"
        )
        XCTAssertEqual(Self.modeBits(for: fixture.rootURL.path), 0o700)
        XCTAssertEqual(Self.goldenBytes.count, 32)
        XCTAssertEqual(
            Self.goldenBytes,
            Data("fast-mlx-e3a-controlled-fixture\n".utf8)
        )
        XCTAssertEqual(Self.sha256Hex(Self.goldenBytes), Self.goldenSHA256)
        _ = fixture.rootExpectation
    }

    func testRegularSmallFixtureHappyPathReturnsOnlyExistingE1Comparison()
        throws
    {
        let fixture = try Self.makeFixture(bytes: Self.goldenBytes)
        defer { Self.removeFixtureRoot(fixture.rootURL) }

        let comparison = try Self.compare(
            fixture: fixture,
            role: .gitRoot,
            expectedFileBytes: 32,
            expectedSHA256: Self.goldenSHA256
        )

        XCTAssertEqual(comparison.role, .gitRoot)
        XCTAssertEqual(comparison.fileBytes, 32)
        XCTAssertEqual(comparison.fileSHA256, Self.goldenSHA256)
        XCTAssertEqual(comparison.retainedBytes, Self.goldenBytes)
        Self.assertAllAuthorityFlagsFalse(comparison)

        try Self.rewriteSameInode(
            fixture.fileURL,
            bytes: Self.replacementBytes
        )
        XCTAssertEqual(comparison.retainedBytes, Self.goldenBytes)
        XCTAssertEqual(comparison.fileSHA256, Self.goldenSHA256)
    }

    func testMultiChunkFixtureHappyPathUsesRealExplicitOffsetStream()
        throws
    {
        var bytes = Data()
        for index in 0..<(Self.readChunkBytes * 2 + 17) {
            bytes.append(UInt8(truncatingIfNeeded: index))
        }
        let fixture = try Self.makeFixture(
            filename: "nested/artifact.bin",
            bytes: bytes
        )
        defer { Self.removeFixtureRoot(fixture.rootURL) }

        let comparison = try Self.compare(
            fixture: fixture,
            role: .fileImage,
            expectedFileBytes: UInt64(bytes.count),
            expectedSHA256: Self.sha256Hex(bytes)
        )

        XCTAssertEqual(comparison.role, .fileImage)
        XCTAssertEqual(comparison.fileBytes, UInt64(bytes.count))
        XCTAssertEqual(comparison.fileSHA256, Self.sha256Hex(bytes))
        XCTAssertEqual(comparison.retainedBytes, bytes)
        Self.assertAllAuthorityFlagsFalse(comparison)
    }

    func testRequestRootLocatorCountDigestDeadlineAndControlTables()
        throws
    {
        let badRoots: [
            (String, (Fixture) -> DarwinControlledFixtureRootExpectation,
             DarwinControlledFixtureSmallArtifactFailure)
        ] = [
            (
                "bad nonce",
                { Self.rootExpectationLike($0, nonce: "ABC") },
                .request(.rootExpectation(.nonce))
            ),
            (
                "relative root",
                { Self.rootExpectationLike(
                    $0,
                    hostAbsolutePathBytes: Data("tmp/root".utf8)
                ) },
                .request(.rootLocator(.leadingSlash))
            ),
            (
                "trailing root slash",
                { fixture in Self.rootExpectationLike(
                    fixture,
                    hostAbsolutePathBytes:
                        Data((fixture.rootURL.path + "/").utf8)
                ) },
                .request(.rootLocator(.trailingSlash))
            ),
            (
                "root nul",
                { Self.rootExpectationLike(
                    $0,
                    hostAbsolutePathBytes:
                        Data([0x2f, 0x74, 0x6d, 0x70, 0x00])
                ) },
                .request(.rootLocator(.nul))
            ),
            (
                "root double slash",
                { Self.rootExpectationLike(
                    $0,
                    hostAbsolutePathBytes: Data("/tmp//root".utf8)
                ) },
                .request(.rootLocator(.emptyComponent(index: 1)))
            ),
            (
                "root dot",
                { Self.rootExpectationLike(
                    $0,
                    hostAbsolutePathBytes: Data("/tmp/./root".utf8)
                ) },
                .request(.rootLocator(.dotComponent(index: 1)))
            ),
            (
                "root dot dot",
                { Self.rootExpectationLike(
                    $0,
                    hostAbsolutePathBytes: Data("/tmp/../root".utf8)
                ) },
                .request(.rootLocator(.dotDotComponent(index: 1)))
            ),
            (
                "root component too long",
                { Self.rootExpectationLike(
                    $0,
                    hostAbsolutePathBytes:
                        Self.absoluteLocator(componentBytes: 256)
                ) },
                .request(.rootLocator(.componentBytes(index: 0)))
            ),
            (
                "root too many components",
                { Self.rootExpectationLike(
                    $0,
                    hostAbsolutePathBytes:
                        Self.absoluteLocator(componentCount: 65)
                ) },
                .request(.rootLocator(.componentCount))
            ),
            (
                "root byte ceiling",
                { Self.rootExpectationLike(
                    $0,
                    hostAbsolutePathBytes:
                        Self.overlongAbsoluteLocator()
                ) },
                .request(.rootLocator(.byteCount))
            ),
            (
                "root invalid utf8",
                { Self.rootExpectationLike(
                    $0,
                    hostAbsolutePathBytes: Data([0x2f, 0xff])
                ) },
                .request(.rootLocator(.utf8))
            ),
            (
                "root basename mismatch",
                { Self.rootExpectationLike(
                    $0,
                    hostAbsolutePathBytes: Data("/tmp/not-e3a".utf8)
                ) },
                .request(.rootExpectation(.basename))
            ),
        ]

        for (label, makeExpectation, expected) in badRoots {
            try Self.withFixture(bytes: Self.goldenBytes) { fixture in
                Self.assertRefuses(label, expected) {
                    try DarwinControlledFixtureSmallArtifactVerifier.compare(
                        rootExpectation: makeExpectation(fixture),
                        logicalLocatorBytes: fixture.logicalLocatorBytes,
                        role: .gitRoot,
                        expectedFileBytes: 32,
                        expectedSHA256: Self.goldenSHA256,
                        deadlineMilliseconds: 10_000,
                        control: nil
                    )
                }
            }
        }

        let badLocators: [
            (String, Data, DarwinControlledFixtureSmallArtifactFailure)
        ] = [
            ("empty", Data(), .request(.logicalLocator(.leadingSlash))),
            ("relative", Data("artifact.bin".utf8),
             .request(.logicalLocator(.leadingSlash))),
            ("double slash", Data("/nested//artifact.bin".utf8),
             .request(.logicalLocator(.emptyComponent(index: 1)))),
            ("dot", Data("/./artifact.bin".utf8),
             .request(.logicalLocator(.dotComponent(index: 0)))),
            ("dot dot", Data("/nested/../artifact.bin".utf8),
             .request(.logicalLocator(.dotDotComponent(index: 1)))),
            ("nul", Data([0x2f, 0x61, 0x00]),
             .request(.logicalLocator(.nul))),
            (
                "component too long",
                Self.absoluteLocator(componentBytes: 256),
                .request(.logicalLocator(.componentBytes(index: 0)))
            ),
            (
                "too many components",
                Self.absoluteLocator(componentCount: 65),
                .request(.logicalLocator(.componentCount))
            ),
            (
                "byte ceiling",
                Self.overlongAbsoluteLocator(),
                .request(.logicalLocator(.byteCount))
            ),
            (
                "invalid utf8",
                Data([0x2f, 0xff]),
                .request(.logicalLocator(.utf8))
            ),
        ]

        for (label, locatorBytes, expected) in badLocators {
            try Self.withFixture(bytes: Self.goldenBytes) { fixture in
                Self.assertRefuses(label, expected) {
                    try DarwinControlledFixtureSmallArtifactVerifier.compare(
                        rootExpectation: fixture.rootExpectation,
                        logicalLocatorBytes: locatorBytes,
                        role: .gitRoot,
                        expectedFileBytes: 32,
                        expectedSHA256: Self.goldenSHA256,
                        deadlineMilliseconds: 10_000,
                        control: nil
                    )
                }
            }
        }

        let badScalars: [
            (String, UInt64, String, UInt64,
             DarwinControlledFixtureSmallArtifactFailure)
        ] = [
            ("zero count", 0, Self.goldenSHA256, 10_000,
             .request(.expectedByteCount)),
            ("too large count", 1_048_577, Self.goldenSHA256, 10_000,
             .request(.expectedByteCount)),
            ("short digest", 32, String(repeating: "a", count: 63), 10_000,
             .request(.expectedSHA256)),
            ("uppercase digest", 32, String(repeating: "A", count: 64),
             10_000, .request(.expectedSHA256)),
            ("zero deadline", 32, Self.goldenSHA256, 0,
             .request(.deadlineDuration)),
            ("long deadline", 32, Self.goldenSHA256, 3_600_001,
             .request(.deadlineDuration)),
        ]

        for (label, count, digest, deadline, expected) in badScalars {
            try Self.withFixture(bytes: Self.goldenBytes) { fixture in
                Self.assertRefuses(label, expected) {
                    try DarwinControlledFixtureSmallArtifactVerifier.compare(
                        rootExpectation: fixture.rootExpectation,
                        logicalLocatorBytes: fixture.logicalLocatorBytes,
                        role: .gitRoot,
                        expectedFileBytes: count,
                        expectedSHA256: digest,
                        deadlineMilliseconds: deadline,
                        control: nil
                    )
                }
            }
        }

        try Self.withFixture(bytes: Self.goldenBytes) { fixture in
            Self.assertRefuses(
                "checked deadline addition overflow",
                .resource(.arithmeticOverflow)
            ) {
                try Self.compare(
                    fixture: fixture,
                    deadlineMilliseconds: 2,
                    control: .init(
                        scriptedClockNanoseconds: [UInt64.max - 1]
                    )
                )
            }
        }

        let badControls: [
            (String, DarwinControlledFixtureTestControl,
             DarwinControlledFixtureSmallArtifactFailure)
        ] = [
            (
                "clock plus barrier",
                .init(
                    scriptedClockNanoseconds: [1, 2, 3],
                    phaseBarrier: .init(phase: .afterRootValidated)
                ),
                .request(.testControl(.clockAndBarrier))
            ),
            (
                "too many clock values",
                .init(
                    scriptedClockNanoseconds:
                        Array(repeating: 1, count: 8_193)
                ),
                .request(.testControl(.scriptedClockCount))
            ),
            (
                "zero interruptions at offset",
                .init(artificialInterruptions: [
                    .init(offset: 0, count: 0),
                ]),
                .request(.testControl(.interruptionCount(index: 0)))
            ),
            (
                "ten interruptions at offset",
                .init(artificialInterruptions: [
                    .init(offset: 0, count: 10),
                ]),
                .request(.testControl(.interruptionCount(index: 0)))
            ),
            (
                "duplicate interruption offset",
                .init(artificialInterruptions: [
                    .init(offset: 0, count: 1),
                    .init(offset: 0, count: 1),
                ]),
                .request(.testControl(.interruptionOrder(index: 1)))
            ),
            (
                "too many interruption entries",
                .init(artificialInterruptions: (0...256).map {
                    .init(offset: UInt64($0), count: 1)
                }),
                .request(.testControl(.interruptionEntryCount))
            ),
            (
                "zero forced read errno",
                .init(forcedNegativeRead: .init(offset: 0, errno: 0)),
                .request(.testControl(.forcedReadErrno))
            ),
            (
                "forced read offset beyond count",
                .init(forcedNegativeRead: .init(offset: 33, errno: EIO)),
                .request(.testControl(.forcedReadOffset))
            ),
            (
                "zero close ordinal",
                .init(reportedCloseFailureOrdinal: 0),
                .request(.testControl(.closeOrdinal))
            ),
            (
                "close ordinal over ceiling",
                .init(reportedCloseFailureOrdinal: 130),
                .request(.testControl(.closeOrdinal))
            ),
            (
                "interruption offset at expected byte count",
                .init(artificialInterruptions: [
                    .init(offset: 32, count: 1),
                ]),
                .request(.testControl(.interruptionOffset(index: 0)))
            ),
            (
                "unreachable interruption",
                .init(
                    artificialInterruptions: [
                        .init(offset: 31, count: 1)
                    ]
                ),
                .testControlDidNotRefuse
            ),
            (
                "unconsumed valid close ordinal",
                .init(reportedCloseFailureOrdinal: 129),
                .testControlDidNotRefuse
            ),
        ]

        for (label, control, expected) in badControls {
            try Self.withFixture(bytes: Self.goldenBytes) { fixture in
                Self.assertRefuses(label, expected) {
                    try Self.compare(fixture: fixture, control: control)
                }
            }
        }
    }

    func testRootIdentityAndFilesystemObjectBranchesRefuse()
        throws
    {
        let rootDrift: [
            (String, (Fixture) -> DarwinControlledFixtureRootExpectation,
             DarwinControlledFixtureSmallArtifactFailure)
        ] = [
            (
                "device",
                { fixture in Self.rootExpectationLike(
                    fixture,
                    expectedDevice:
                        fixture.rootExpectation.expectedDevice ^ 1
                ) },
             .root(.identity(.expectedDevice))),
            (
                "inode",
                { fixture in Self.rootExpectationLike(
                    fixture,
                    expectedInode:
                        fixture.rootExpectation.expectedInode ^ 1
                ) },
             .root(.identity(.expectedInode))),
            ("mode", { Self.rootExpectationLike($0, expectedMode: 0o755) },
             .root(.mode)),
            (
                "user id",
                { fixture in Self.rootExpectationLike(
                    fixture,
                    expectedUserID:
                        fixture.rootExpectation.expectedUserID ^ 1
                ) },
                .root(.identity(.expectedUserID))
            ),
            (
                "group id",
                { fixture in Self.rootExpectationLike(
                    fixture,
                    expectedGroupID:
                        fixture.rootExpectation.expectedGroupID ^ 1
                ) },
                .root(.identity(.expectedGroupID))
            ),
            (
                "flags",
                { fixture in Self.rootExpectationLike(
                    fixture,
                    expectedFlags:
                        fixture.rootExpectation.expectedFlags ^ 1
                ) },
                .root(.identity(.expectedFlags))
            ),
            (
                "generation",
                { fixture in Self.rootExpectationLike(
                    fixture,
                    expectedGeneration:
                        fixture.rootExpectation.expectedGeneration ^ 1
                ) },
                .root(.identity(.expectedGeneration))
            ),
            (
                "modification seconds",
                { fixture in Self.rootExpectationLike(
                    fixture,
                    expectedModificationTimeSeconds:
                        fixture.rootExpectation
                            .expectedModificationTimeSeconds + 1
                ) },
                .root(.identity(.expectedModificationTimeSeconds))
            ),
            (
                "modification nanoseconds",
                { fixture in Self.rootExpectationLike(
                    fixture,
                    expectedModificationTimeNanoseconds:
                        fixture.rootExpectation
                            .expectedModificationTimeNanoseconds ^ 1
                ) },
                .root(.identity(.expectedModificationTimeNanoseconds))
            ),
            (
                "status-change seconds",
                { fixture in Self.rootExpectationLike(
                    fixture,
                    expectedStatusChangeTimeSeconds:
                        fixture.rootExpectation
                            .expectedStatusChangeTimeSeconds + 1
                ) },
                .root(.identity(.expectedStatusChangeTimeSeconds))
            ),
            (
                "status-change nanoseconds",
                { fixture in Self.rootExpectationLike(
                    fixture,
                    expectedStatusChangeTimeNanoseconds:
                        fixture.rootExpectation
                            .expectedStatusChangeTimeNanoseconds ^ 1
                ) },
                .root(.identity(.expectedStatusChangeTimeNanoseconds))
            ),
            (
                "birth seconds",
                { fixture in Self.rootExpectationLike(
                    fixture,
                    expectedBirthTimeSeconds:
                        fixture.rootExpectation.expectedBirthTimeSeconds + 1
                ) },
                .root(.identity(.expectedBirthTimeSeconds))
            ),
            (
                "birth nanoseconds",
                { fixture in Self.rootExpectationLike(
                    fixture,
                    expectedBirthTimeNanoseconds:
                        fixture.rootExpectation
                            .expectedBirthTimeNanoseconds ^ 1
                ) },
                .root(.identity(.expectedBirthTimeNanoseconds))
            ),
        ]

        for (label, makeExpectation, expected) in rootDrift {
            try Self.withFixture(bytes: Self.goldenBytes) { fixture in
                Self.assertRefuses(label, expected) {
                    try DarwinControlledFixtureSmallArtifactVerifier.compare(
                        rootExpectation: makeExpectation(fixture),
                        logicalLocatorBytes: fixture.logicalLocatorBytes,
                        role: .gitRoot,
                        expectedFileBytes: 32,
                        expectedSHA256: Self.goldenSHA256,
                        deadlineMilliseconds: 10_000,
                        control: nil
                    )
                }
            }
        }

        let rootSymlink = try Self.makeFixture(bytes: Self.goldenBytes)
        let rootSymlinkPath = rootSymlink.rootURL
        let rootSymlinkTarget = rootSymlinkPath.deletingLastPathComponent()
            .appendingPathComponent("e3a-root-target-\(rootSymlink.nonce)")
        XCTAssertEqual(
            Darwin.rename(rootSymlinkPath.path, rootSymlinkTarget.path),
            0
        )
        XCTAssertEqual(
            Darwin.symlink(
                rootSymlinkTarget.lastPathComponent,
                rootSymlinkPath.path
            ),
            0
        )
        defer {
            Self.removeFixtureRoot(rootSymlinkPath)
            Self.removeFixtureRoot(rootSymlinkTarget)
        }
        Self.assertRefuses(
            "root symlink",
            .open(.rootComponent(
                index: Self.componentCount(rootSymlinkPath.path) - 1,
                errno: ELOOP
            ))
        ) {
            try Self.compare(fixture: rootSymlink)
        }

        let finalSymlink = try Self.makeFixture(
            filename: "link.bin",
            bytes: Self.goldenBytes
        ) { root in
            let target = root.appendingPathComponent("target.bin")
            try Self.goldenBytes.write(to: target)
            XCTAssertEqual(
                Darwin.symlink("target.bin", root.appendingPathComponent(
                    "link.bin"
                ).path),
                0
            )
        }
        defer { Self.removeFixtureRoot(finalSymlink.rootURL) }
        Self.assertRefuses("final symlink", .open(.logicalComponent(
            index: 0,
            errno: ELOOP
        ))) {
            try Self.compare(fixture: finalSymlink)
        }

        let ancestorSymlink = try Self.makeFixture(
            filename: "alias/artifact.bin",
            bytes: Self.goldenBytes
        ) { root in
            let real = root.appendingPathComponent("real", isDirectory: true)
            try FileManager.default.createDirectory(
                at: real,
                withIntermediateDirectories: false
            )
            try Self.goldenBytes.write(
                to: real.appendingPathComponent("artifact.bin")
            )
            XCTAssertEqual(
                Darwin.symlink("real", root.appendingPathComponent(
                    "alias"
                ).path),
                0
            )
        }
        defer { Self.removeFixtureRoot(ancestorSymlink.rootURL) }
        Self.assertRefuses("ancestor symlink", .open(.logicalComponent(
            index: 0,
            errno: ELOOP
        ))) {
            try Self.compare(fixture: ancestorSymlink)
        }

        let hardlink = try Self.makeFixture(bytes: Self.goldenBytes) { root in
            let artifact = root.appendingPathComponent("artifact.bin")
            try Self.goldenBytes.write(to: artifact)
            XCTAssertEqual(
                Darwin.link(
                    artifact.path,
                    root.appendingPathComponent("second.bin").path
                ),
                0
            )
        }
        defer { Self.removeFixtureRoot(hardlink.rootURL) }
        Self.assertRefuses("hardlink", .file(.linkCount)) {
            try Self.compare(fixture: hardlink)
        }

        let directory = try Self.makeFixture(
            filename: "directory",
            bytes: Data()
        ) { root in
            try FileManager.default.createDirectory(
                at: root.appendingPathComponent("directory"),
                withIntermediateDirectories: false
            )
        }
        defer { Self.removeFixtureRoot(directory.rootURL) }
        Self.assertRefuses("directory", .file(.type)) {
            try Self.compare(
                fixture: directory,
                expectedFileBytes: 32,
                expectedSHA256: Self.goldenSHA256
            )
        }

        let fifo = try Self.makeFixture(filename: "pipe", bytes: Data()) {
            root in
            XCTAssertEqual(
                Darwin.mkfifo(root.appendingPathComponent("pipe").path, 0o600),
                0
            )
        }
        defer { Self.removeFixtureRoot(fifo.rootURL) }
        Self.assertRefuses("fifo", .file(.type)) {
            try Self.compare(fixture: fifo)
        }

        let sparse = try Self.makeSparseFixture()
        defer { Self.removeFixtureRoot(sparse.rootURL) }
        guard let exposesSparse = try Self.probeSparseState(
            for: sparse.fileURL
        ) else {
            throw XCTSkip(
                "fresh test filesystem did not return decodable extended flags"
            )
        }
        guard exposesSparse else {
            throw XCTSkip(
                "fresh test filesystem did not expose EF_IS_SPARSE"
            )
        }
        var sparseBytes = Data(repeating: 0, count: 1_048_576)
        sparseBytes[sparseBytes.index(before: sparseBytes.endIndex)] = 0x5a
        Self.assertRefuses("sparse file", .file(.sparse)) {
            try Self.compare(
                fixture: sparse,
                expectedFileBytes: UInt64(sparseBytes.count),
                expectedSHA256: Self.sha256Hex(sparseBytes)
            )
        }
    }

    func testFileSizeExpectedCountAndDigestBranchesRefuseExactly()
        throws
    {
        let empty = try Self.makeFixture(bytes: Data())
        defer { Self.removeFixtureRoot(empty.rootURL) }
        Self.assertRefuses("empty file", .file(.size)) {
            try Self.compare(
                fixture: empty,
                expectedFileBytes: 1,
                expectedSHA256: Self.goldenSHA256
            )
        }

        let oversizedBytes = Data(repeating: 0x41, count: 1_048_577)
        let oversized = try Self.makeFixture(bytes: oversizedBytes)
        defer { Self.removeFixtureRoot(oversized.rootURL) }
        Self.assertRefuses("oversized file", .file(.size)) {
            try Self.compare(
                fixture: oversized,
                expectedFileBytes: 1_048_576,
                expectedSHA256: Self.sha256Hex(
                    oversizedBytes.prefix(1_048_576)
                )
            )
        }

        try Self.withFixture(bytes: Self.goldenBytes) { fixture in
            Self.assertRefuses(
                "expected count mismatch",
                .expected(.byteCount)
            ) {
                try Self.compare(
                    fixture: fixture,
                    expectedFileBytes: 31,
                    expectedSHA256: Self.goldenSHA256
                )
            }
        }
        try Self.withFixture(bytes: Self.goldenBytes) { fixture in
            Self.assertRefuses(
                "expected digest mismatch",
                .expected(.sha256)
            ) {
                try Self.compare(
                    fixture: fixture,
                    expectedFileBytes: 32,
                    expectedSHA256: String(repeating: "0", count: 64)
                )
            }
        }
    }

    func testPureDescriptorAttributeAndMappedMetadataProbesRefuseExactly()
        throws
    {
        Self.assertRefuses("missing close-on-exec",
                           .descriptor(.closeOnExec(.finalFile))) {
            try DarwinControlledFixtureSmallArtifactVerifier
                .validateDescriptorFlagsForTesting(0)
        }

        let optionalAttributeCases: [
            (String, UInt32, UInt64?, UInt64, UInt32?, UInt64)
        ] = [
            ("ext flags only", Self.extFlagsBit, nil, 0, nil, 0b001),
            (
                "clone id and ext flags",
                Self.cloneIDBit | Self.extFlagsBit,
                41,
                0,
                nil,
                0b011
            ),
            (
                "ext flags and clone refcount",
                Self.extFlagsBit | Self.cloneRefCountBit,
                nil,
                0,
                7,
                0b101
            ),
            (
                "all optional values",
                Self.cloneIDBit | Self.extFlagsBit | Self.cloneRefCountBit,
                41,
                0,
                7,
                0b111
            ),
        ]

        for (label, bits, cloneID, rawFlags, cloneRefCount, supportMask)
            in optionalAttributeCases
        {
            let decoded = try DarwinControlledFixtureSmallArtifactVerifier
                .decodeAttributeBufferForTesting(
                    Self.attributeBuffer(
                        forkBits: bits,
                        cloneID: cloneID,
                        extFlags: rawFlags,
                        cloneReferenceCount: cloneRefCount
                    )
                )
            XCTAssertEqual(decoded.extendedAttributeSupportMask, supportMask,
                           label)
            XCTAssertEqual(decoded.extendedFlags, 0, label)
            XCTAssertEqual(decoded.cloneID, cloneID, label)
            XCTAssertEqual(decoded.cloneReferenceCount, cloneRefCount, label)
        }

        let sparseRawFlags = UInt64(EF_IS_SPARSE)
        let sparseDecoded = try DarwinControlledFixtureSmallArtifactVerifier
            .decodeAttributeBufferForTesting(
                Self.attributeBuffer(
                    forkBits: Self.extFlagsBit,
                    extFlags: sparseRawFlags
                )
            )
        XCTAssertEqual(
            sparseDecoded.extendedFlags,
            (sparseRawFlags << 1) | 1
        )

        let attributeCases: [
            (String, Data, DarwinControlledFixtureSmallArtifactFailure)
        ] = [
            ("too short", Data(repeating: 0, count: 23),
             .metadata(.attributeLength(.before))),
            ("foreign bit", Self.attributeBuffer(forkBits: 0xffff_ffff),
             .metadata(.attributeBits(.before))),
            ("missing ext flags", Self.attributeBuffer(forkBits: 0),
             .metadata(.attributeSupport(.before))),
            ("raw ext flags overflow",
             Self.attributeBuffer(forkBits: Self.extFlagsBit,
                                  extFlags: UInt64.max),
             .metadata(.attributeValue(.before))),
        ]

        for (label, buffer, expected) in attributeCases {
            Self.assertRefuses(label, expected) {
                _ = try DarwinControlledFixtureSmallArtifactVerifier
                    .decodeAttributeBufferForTesting(buffer)
            }
        }

        let invalidMetadata: [
            (String, SyntheticCaptureFileMetadata,
             DarwinControlledFixtureSmallArtifactFailure)
        ] = [
            ("socket", Self.metadata(mode: 0o140000),
             .file(.type)),
            ("character device", Self.metadata(mode: 0o020000),
             .file(.type)),
            ("block device", Self.metadata(mode: 0o060000),
             .file(.type)),
            ("dataless", Self.metadata(flags: 0x40000000),
             .file(.dataless)),
            ("sparse unavailable",
             Self.metadata(extendedAttributeSupportMask: 0),
             .file(.sparseStateUnavailable)),
            ("sparse", Self.metadata(extendedFlags: 1),
             .file(.sparse)),
        ]

        for (label, metadata, expected) in invalidMetadata {
            Self.assertRefuses(label, expected) {
                try DarwinControlledFixtureSmallArtifactVerifier
                    .validateMappedMetadataForTesting(metadata)
            }
        }
    }

    func testReadDeadlineBarrierCleanupAndControlFailures()
        throws
    {
        let interrupted = DarwinControlledFixtureTestControl(
            artificialInterruptions: [.init(offset: 0, count: 8)]
        )
        try Self.withFixture(bytes: Self.goldenBytes) { fixture in
            let interruptedComparison = try Self.compare(
                fixture: fixture,
                control: interrupted
            )
            XCTAssertEqual(
                interruptedComparison.retainedBytes,
                Self.goldenBytes
            )
            Self.assertAllAuthorityFlagsFalse(interruptedComparison)
        }

        try Self.withFixture(bytes: Self.goldenBytes) { fixture in
            Self.assertRefuses(
                "deadline before first open",
                .deadline(.expired(.beforeHostRootOpen))
            ) {
                try Self.compare(
                    fixture: fixture,
                    deadlineMilliseconds: 1,
                    control: .init(
                        scriptedClockNanoseconds: [1, 1_000_001]
                    )
                )
            }
        }

        let controls: [
            (String, DarwinControlledFixtureTestControl,
             DarwinControlledFixtureSmallArtifactFailure)
        ] = [
            (
                "ninth interruption",
                .init(artificialInterruptions: [.init(offset: 0, count: 9)]),
                .read(.interruptedLimit(offset: 0))
            ),
            (
                "forced read",
                .init(forcedNegativeRead: .init(offset: 0, errno: EIO)),
                .read(.system(offset: 0, errno: EIO))
            ),
            (
                "clock failure",
                .init(scriptedClockNanoseconds: [0]),
                .clock(.failure)
            ),
            (
                "clock decrease",
                .init(scriptedClockNanoseconds: [10, 9]),
                .clock(.decreased)
            ),
            (
                "reported close failure",
                .init(reportedCloseFailureOrdinal: 1),
                .cleanup(.close(role: .finalFile, errno: EIO))
            ),
        ]

        for (label, control, expected) in controls {
            try Self.withFixture(bytes: Self.goldenBytes) { fixture in
                Self.assertRefuses(label, expected) {
                    try Self.compare(fixture: fixture, control: control)
                }
            }
        }

        try Self.withFixture(bytes: Self.goldenBytes) { fixture in
            Self.assertRefuses(
                "second close is held fixture root",
                .cleanup(.close(role: .fixtureRoot, errno: EIO))
            ) {
                try Self.compare(
                    fixture: fixture,
                    control: .init(reportedCloseFailureOrdinal: 2)
                )
            }
        }

        let hostRootComponentCount =
            Self.componentCount((try Self.canonicalTemporaryDirectory()).path)
            + 1
        var nestedCloseOrder: [
            DarwinControlledFixtureSmallArtifactFailure
        ] = [
            .cleanup(.close(role: .finalFile, errno: EIO)),
            .cleanup(.close(role: .logicalAncestor(index: 0), errno: EIO)),
            .cleanup(.close(role: .fixtureRoot, errno: EIO)),
        ]
        if hostRootComponentCount > 1 {
            for index in stride(
                from: hostRootComponentCount - 2,
                through: 0,
                by: -1
            ) {
                nestedCloseOrder.append(
                    .cleanup(.close(
                        role: .rootComponent(index: index),
                        errno: EIO
                    ))
                )
            }
        }
        nestedCloseOrder.append(
            .cleanup(.close(role: .hostRoot, errno: EIO))
        )

        for (zeroBasedIndex, expected) in nestedCloseOrder.enumerated() {
            try Self.withFixture(
                filename: "nested/artifact.bin",
                bytes: Self.goldenBytes
            ) { fixture in
                Self.assertRefuses(
                    "nested reverse close ordinal \(zeroBasedIndex + 1)",
                    expected
                ) {
                    try Self.compare(
                        fixture: fixture,
                        control: .init(
                            reportedCloseFailureOrdinal: zeroBasedIndex + 1
                        )
                    )
                }
            }
        }

        try Self.withFixture(bytes: Self.goldenBytes) { fixture in
            Self.assertRefuses(
                "primary read failure wins over cleanup failure",
                .read(.system(offset: 0, errno: EIO))
            ) {
                try Self.compare(
                    fixture: fixture,
                    control: .init(
                        forcedNegativeRead: .init(offset: 0, errno: EIO),
                        reportedCloseFailureOrdinal: 1
                    )
                )
            }
        }

        try Self.withFixture(bytes: Self.goldenBytes) { fixture in
            let beforeE1Barrier =
                DarwinControlledFixtureTestControl.PhaseBarrier(
                    phase: .beforeE1Comparison
                )
            let observer = Self.startBarrierNonArrivalObserver(
                barrier: beforeE1Barrier,
                label: "pre-cleanup read refusal"
            )
            Self.assertRefuses(
                "read refusal never reaches E1 construction phase",
                .read(.system(offset: 0, errno: EIO))
            ) {
                try Self.compare(
                    fixture: fixture,
                    control: .init(
                        forcedNegativeRead: .init(offset: 0, errno: EIO),
                        phaseBarrier: beforeE1Barrier
                    )
                )
            }
            observer.waitAndAssert()
        }

        try Self.withFixture(bytes: Self.goldenBytes) { fixture in
            let beforeE1Barrier =
                DarwinControlledFixtureTestControl.PhaseBarrier(
                    phase: .beforeE1Comparison
                )
            let observer = Self.startBarrierNonArrivalObserver(
                barrier: beforeE1Barrier,
                label: "cleanup refusal"
            )
            Self.assertRefuses(
                "cleanup refusal never reaches E1 construction phase",
                .cleanup(.close(role: .finalFile, errno: EIO))
            ) {
                try Self.compare(
                    fixture: fixture,
                    control: .init(
                        reportedCloseFailureOrdinal: 1,
                        phaseBarrier: beforeE1Barrier
                    )
                )
            }
            observer.waitAndAssert()
        }

        try Self.withFixture(bytes: Self.goldenBytes) { fixture in
            Self.assertRefuses(
                "root identity failure still cleans the held fixture root",
                .root(.identity(.expectedInode))
            ) {
                try DarwinControlledFixtureSmallArtifactVerifier.compare(
                    rootExpectation: Self.rootExpectationLike(
                        fixture,
                        expectedInode:
                            fixture.rootExpectation.expectedInode ^ 1
                    ),
                    logicalLocatorBytes: fixture.logicalLocatorBytes,
                    role: .gitRoot,
                    expectedFileBytes: 32,
                    expectedSHA256: Self.goldenSHA256,
                    deadlineMilliseconds: 10_000,
                    control: .init(reportedCloseFailureOrdinal: 1)
                )
            }
        }

        try Self.withFixture(
            filename: "directory",
            bytes: Data(),
            customCreate: { root in
                try FileManager.default.createDirectory(
                    at: root.appendingPathComponent("directory"),
                    withIntermediateDirectories: false
                )
            }
        ) { fixture in
            Self.assertRefuses(
                "file policy failure wins over final-file close failure",
                .file(.type)
            ) {
                try Self.compare(
                    fixture: fixture,
                    control: .init(reportedCloseFailureOrdinal: 1)
                )
            }
        }

        try Self.withFixture(bytes: Self.goldenBytes) { fixture in
            let beforeE1Barrier =
                DarwinControlledFixtureTestControl.PhaseBarrier(
                    phase: .beforeE1Comparison
                )
            let beforeE1Worker = Self.startBarrierWorker(
                barrier: beforeE1Barrier
            ) {}
            let postCleanupComparison = try Self.compare(
                fixture: fixture,
                control: .init(phaseBarrier: beforeE1Barrier)
            )
            beforeE1Worker.waitAndAssert()
            XCTAssertEqual(
                postCleanupComparison.retainedBytes,
                Self.goldenBytes
            )
            Self.assertAllAuthorityFlagsFalse(postCleanupComparison)
        }
    }

    func testRealBarrierDeadlinePhasesRefuseWithoutSyntheticClock()
        throws
    {
        let cases: [
            (
                String,
                DarwinControlledFixtureTestControl.Phase,
                DarwinControlledFixtureSmallArtifactFailure.DeadlinePhase,
                Data
            )
        ] = [
            (
                "after final descriptor open",
                .afterFinalDescriptorOpened,
                .afterFinalDescriptorOpened,
                Self.goldenBytes
            ),
            (
                "between positive fragments",
                .afterFirstPositiveFragment,
                .afterFirstPositiveFragment,
                Data(repeating: 0x41, count: Self.readChunkBytes + 1)
            ),
            (
                "before eof probe",
                .afterExpectedBytesRead,
                .afterExpectedBytesRead,
                Self.goldenBytes
            ),
        ]

        for (label, phase, deadlinePhase, bytes) in cases {
            let fixture = try Self.makeFixture(bytes: bytes)
            defer { Self.removeFixtureRoot(fixture.rootURL) }
            let barrier = DarwinControlledFixtureTestControl.PhaseBarrier(
                phase: phase
            )
            let worker = Self.startBarrierWorker(barrier: barrier) {
                Darwin.usleep(20_000)
            }

            Self.assertRefuses(
                label,
                .deadline(.expired(deadlinePhase))
            ) {
                try Self.compare(
                    fixture: fixture,
                    expectedFileBytes: UInt64(bytes.count),
                    expectedSHA256: Self.sha256Hex(bytes),
                    deadlineMilliseconds: 1,
                    control: .init(
                        reportedCloseFailureOrdinal: 1,
                        phaseBarrier: barrier
                    )
                )
            }
            worker.waitAndAssert()
        }
    }

    func testBarrierCoordinatedMutationDriftAndPathReplacementRefusalCases()
        throws
    {
        let rewrite = try Self.makeFixture(bytes: Self.goldenBytes)
        defer { Self.removeFixtureRoot(rewrite.rootURL) }
        let rewriteBarrier = DarwinControlledFixtureTestControl.PhaseBarrier(
            phase: .afterBeforeMetadataValidated
        )
        let rewriteWorker = Self.startBarrierWorker(barrier: rewriteBarrier) {
            try Self.rewriteSameInode(
                rewrite.fileURL,
                bytes: Data(repeating: 0x71, count: 32)
            )
        }
        Self.assertRefuses("same size rewrite",
                           .metadata(.drift(.modificationTimeSeconds))) {
            try Self.compare(
                fixture: rewrite,
                control: .init(phaseBarrier: rewriteBarrier)
            )
        }
        rewriteWorker.waitAndAssert()

        let growth = try Self.makeFixture(bytes: Self.goldenBytes)
        defer { Self.removeFixtureRoot(growth.rootURL) }
        let growthBarrier = DarwinControlledFixtureTestControl.PhaseBarrier(
            phase: .afterBeforeMetadataValidated
        )
        let growthWorker = Self.startBarrierWorker(barrier: growthBarrier) {
            try Self.appendFile(growth.fileURL, bytes: Data([0x21]))
        }
        Self.assertRefuses("growth", .read(.trailingByte(offset: 32))) {
            try Self.compare(
                fixture: growth,
                control: .init(phaseBarrier: growthBarrier)
            )
        }
        growthWorker.waitAndAssert()

        let truncation = try Self.makeFixture(bytes: Self.goldenBytes)
        defer { Self.removeFixtureRoot(truncation.rootURL) }
        let truncationBarrier =
            DarwinControlledFixtureTestControl.PhaseBarrier(
                phase: .afterBeforeMetadataValidated
            )
        let truncationWorker = Self.startBarrierWorker(
            barrier: truncationBarrier
        ) {
            guard Darwin.truncate(truncation.fileURL.path, 8) == 0 else {
                throw Self.posixError()
            }
        }
        Self.assertRefuses("truncation", .read(.short(expected: 32,
                                                      actual: 8))) {
            try Self.compare(
                fixture: truncation,
                control: .init(phaseBarrier: truncationBarrier)
            )
        }
        truncationWorker.waitAndAssert()

        let rootDrift = try Self.makeFixture(bytes: Self.goldenBytes)
        defer { Self.removeFixtureRoot(rootDrift.rootURL) }
        let rootDriftBarrier =
            DarwinControlledFixtureTestControl.PhaseBarrier(
                phase: .afterExpectedBytesRead
            )
        let rootDriftWorker = Self.startBarrierWorker(
            barrier: rootDriftBarrier
        ) {
            guard Darwin.chmod(rootDrift.rootURL.path, 0o755) == 0 else {
                throw Self.posixError()
            }
        }
        Self.assertRefuses("root drift", .root(.mode)) {
            try Self.compare(
                fixture: rootDrift,
                control: .init(phaseBarrier: rootDriftBarrier)
            )
        }
        rootDriftWorker.waitAndAssert()

        let replacement = try Self.makeFixture(
            filename: "mutable/artifact.bin",
            bytes: Self.goldenBytes
        )
        defer { Self.removeFixtureRoot(replacement.rootURL) }
        let replacementBarrier =
            DarwinControlledFixtureTestControl.PhaseBarrier(
                phase: .afterFinalDescriptorOpened
            )
        let replacementWorker = Self.startBarrierWorker(
            barrier: replacementBarrier
        ) {
            let original = replacement.fileURL.deletingLastPathComponent()
                .appendingPathComponent("opened-original.bin")
            try FileManager.default.moveItem(
                at: replacement.fileURL,
                to: original
            )
            try Self.replacementBytes.write(to: replacement.fileURL)
        }

        var replacementComparison:
            SyntheticSmallArtifactCaptureComparison?
        var replacementFailure:
            DarwinControlledFixtureSmallArtifactFailure?
        do {
            replacementComparison = try Self.compare(
                fixture: replacement,
                control: .init(phaseBarrier: replacementBarrier)
            )
        } catch let failure as DarwinControlledFixtureSmallArtifactFailure {
            replacementFailure = failure
        }
        replacementWorker.waitAndAssert()

        if let replacementComparison {
            XCTAssertNil(replacementFailure)
            XCTAssertEqual(
                replacementComparison.retainedBytes,
                Self.goldenBytes
            )
            XCTAssertEqual(
                replacementComparison.fileSHA256,
                Self.goldenSHA256
            )
            XCTAssertNotEqual(
                replacementComparison.retainedBytes,
                Self.replacementBytes
            )
            Self.assertAllAuthorityFlagsFalse(replacementComparison)
        } else if let replacementFailure {
            switch replacementFailure {
            case .metadata(.drift(_)), .root(.identity(_)):
                break
            default:
                XCTFail(
                    "path replacement refused outside the frozen drift boundary: \(replacementFailure)"
                )
            }
        } else {
            XCTFail("path replacement produced neither comparison nor refusal")
        }
        XCTAssertEqual(
            try Data(contentsOf: replacement.fileURL),
            Self.replacementBytes
        )
    }

}

private extension DarwinControlledFixtureSmallArtifactTests {
    struct Fixture {
        let rootURL: URL
        let fileURL: URL
        let nonce: String
        let logicalLocatorBytes: Data
        let rootExpectation: DarwinControlledFixtureRootExpectation
    }

    final class BarrierWorkerOutcome: @unchecked Sendable {
        private let lock = NSLock()
        private var recordedFailure: String?
        private var recordedMetadataDrift:
            SyntheticSmallArtifactCaptureFailure.MetadataField?

        func record(_ failure: String) {
            lock.lock()
            defer { lock.unlock() }
            if recordedFailure == nil {
                recordedFailure = failure
            }
        }

        func failure() -> String? {
            lock.lock()
            defer { lock.unlock() }
            return recordedFailure
        }

        func recordMetadataDrift(
            _ field: SyntheticSmallArtifactCaptureFailure.MetadataField?
        ) {
            lock.lock()
            defer { lock.unlock() }
            recordedMetadataDrift = field
        }

        func metadataDrift()
            -> SyntheticSmallArtifactCaptureFailure.MetadataField?
        {
            lock.lock()
            defer { lock.unlock() }
            return recordedMetadataDrift
        }
    }

    struct BarrierWorker {
        let item: DispatchWorkItem
        let outcome: BarrierWorkerOutcome

        func waitAndAssert(
            file: StaticString = #filePath,
            line: UInt = #line
        ) {
            item.wait()
            XCTAssertNil(outcome.failure(), file: file, line: line)
        }
    }

    static let goldenBytes = Data("fast-mlx-e3a-controlled-fixture\n".utf8)
    static let goldenSHA256 =
        "5b42fd3c73c13f475f150d7d91df99e8f2e79a81e2e07cfb0440a0f84cfc83ab"
    static let replacementBytes = Data(
        "fast-mlx-e3a-replacement-bytes!\n".utf8
    )
    static let readChunkBytes = 65_536
    static let cloneIDBit = UInt32(ATTR_CMNEXT_CLONEID)
    static let extFlagsBit = UInt32(ATTR_CMNEXT_EXT_FLAGS)
    static let cloneRefCountBit = UInt32(ATTR_CMNEXT_CLONE_REFCNT)

    static func compare(
        fixture: Fixture,
        role: SyntheticSmallArtifactCaptureComparison.Role = .gitRoot,
        expectedFileBytes: UInt64 = 32,
        expectedSHA256: String = goldenSHA256,
        deadlineMilliseconds: UInt64 = 10_000,
        control: DarwinControlledFixtureTestControl? = nil
    ) throws -> SyntheticSmallArtifactCaptureComparison {
        try DarwinControlledFixtureSmallArtifactVerifier.compare(
            rootExpectation: fixture.rootExpectation,
            logicalLocatorBytes: fixture.logicalLocatorBytes,
            role: role,
            expectedFileBytes: expectedFileBytes,
            expectedSHA256: expectedSHA256,
            deadlineMilliseconds: deadlineMilliseconds,
            control: control
        )
    }

    static func startBarrierWorker(
        barrier: DarwinControlledFixtureTestControl.PhaseBarrier,
        mutation: @escaping @Sendable () throws -> Void
    ) -> BarrierWorker {
        startBarrierWorkerWithOutcome(barrier: barrier) { _ in
            try mutation()
        }
    }

    static func startBarrierWorkerWithOutcome(
        barrier: DarwinControlledFixtureTestControl.PhaseBarrier,
        mutation: @escaping @Sendable (BarrierWorkerOutcome) throws -> Void
    ) -> BarrierWorker {
        let outcome = BarrierWorkerOutcome()
        let item = DispatchWorkItem {
            guard barrier.waitForArrivalForTesting(
                timeoutMilliseconds: 5_000
            ) else {
                outcome.record("phase barrier did not arrive within 5 seconds")
                barrier.signalResumeForTesting()
                return
            }
            defer { barrier.signalResumeForTesting() }
            do {
                try mutation(outcome)
            } catch {
                outcome.record("fixture mutation failed: \(error)")
            }
        }
        DispatchQueue.global(qos: .userInitiated).async(execute: item)
        return BarrierWorker(item: item, outcome: outcome)
    }

    static func startBarrierNonArrivalObserver(
        barrier: DarwinControlledFixtureTestControl.PhaseBarrier,
        label: String
    ) -> BarrierWorker {
        let outcome = BarrierWorkerOutcome()
        let item = DispatchWorkItem {
            if barrier.waitForArrivalForTesting(timeoutMilliseconds: 1_000) {
                outcome.record("unexpected \(label) before-E1 arrival")
                barrier.signalResumeForTesting()
            }
        }
        DispatchQueue.global(qos: .userInitiated).async(execute: item)
        return BarrierWorker(item: item, outcome: outcome)
    }

    static func makeFixture(
        filename: String = "artifact.bin",
        bytes: Data,
        customCreate: ((URL) throws -> Void)? = nil
    ) throws -> Fixture {
        let temp = try canonicalTemporaryDirectory()
        let nonce = lowercaseHexNonce()
        let root = temp.appendingPathComponent(
            "fast-mlx-e3a-\(nonce)",
            isDirectory: true
        )
        XCTAssertEqual(Darwin.mkdir(root.path, 0o700), 0)
        XCTAssertEqual(Darwin.chmod(root.path, 0o700), 0)

        let file = root.appendingPathComponent(filename)
        if let customCreate {
            try customCreate(root)
        } else {
            let parent = file.deletingLastPathComponent()
            if parent.path != root.path {
                try FileManager.default.createDirectory(
                    at: parent,
                    withIntermediateDirectories: true
                )
            }
            try bytes.write(to: file, options: .atomic)
        }

        let expectation = try rootExpectation(for: root, nonce: nonce)
        return Fixture(
            rootURL: root,
            fileURL: file,
            nonce: nonce,
            logicalLocatorBytes: Data(("/" + filename).utf8),
            rootExpectation: expectation
        )
    }

    @discardableResult
    static func withFixture<T>(
        filename: String = "artifact.bin",
        bytes: Data,
        customCreate: ((URL) throws -> Void)? = nil,
        _ body: (Fixture) throws -> T
    ) throws -> T {
        let fixture = try makeFixture(
            filename: filename,
            bytes: bytes,
            customCreate: customCreate
        )
        defer { removeFixtureRoot(fixture.rootURL) }
        return try body(fixture)
    }

    static func makeSparseFixture() throws -> Fixture {
        try makeFixture(filename: "sparse.bin", bytes: Data()) { root in
            let path = root.appendingPathComponent("sparse.bin").path
            let fd = Darwin.open(path, O_CREAT | O_WRONLY | O_TRUNC, 0o600)
            XCTAssertGreaterThanOrEqual(fd, 0)
            defer { XCTAssertEqual(Darwin.close(fd), 0) }
            XCTAssertEqual(Darwin.ftruncate(fd, 1_048_576), 0)
            XCTAssertEqual(Darwin.pwrite(fd, [UInt8(0x5a)], 1, 1_048_575), 1)
        }
    }

    static func probeSparseState(for file: URL) throws -> Bool? {
        let descriptor = Darwin.open(file.path, O_RDONLY | O_CLOEXEC)
        guard descriptor >= 0 else { throw posixError() }
        defer { XCTAssertEqual(Darwin.close(descriptor), 0) }

        var request = attrlist()
        request.bitmapcount = UInt16(ATTR_BIT_MAP_COUNT)
        request.reserved = 0
        request.commonattr = UInt32(ATTR_CMN_RETURNED_ATTRS)
        request.volattr = 0
        request.dirattr = 0
        request.fileattr = 0
        request.forkattr = cloneIDBit | extFlagsBit | cloneRefCountBit

        var bytes = [UInt8](repeating: 0, count: 44)
        let result = bytes.withUnsafeMutableBytes { rawBuffer in
            Darwin.fgetattrlist(
                descriptor,
                &request,
                rawBuffer.baseAddress,
                rawBuffer.count,
                UInt32(FSOPT_ATTR_CMN_EXTENDED)
            )
        }
        guard result == 0 else {
            if errno == EINVAL || errno == ENOTSUP {
                return nil
            }
            throw posixError()
        }

        let length = bytes.withUnsafeBytes { rawBuffer -> UInt32 in
            var value: UInt32 = 0
            withUnsafeMutableBytes(of: &value) { destination in
                destination.copyBytes(from: rawBuffer.prefix(4))
            }
            return UInt32(littleEndian: value)
        }
        guard (24...44).contains(Int(length)) else { return nil }

        do {
            let decoded = try DarwinControlledFixtureSmallArtifactVerifier
                .decodeAttributeBufferForTesting(
                    Data(bytes.prefix(Int(length)))
                )
            return (decoded.extendedFlags & 1) == 1
        } catch let failure as DarwinControlledFixtureSmallArtifactFailure {
            if failure == .metadata(.attributeSupport(.before)) {
                return nil
            }
            throw failure
        }
    }

    static func rootExpectation(
        for root: URL,
        nonce: String
    ) throws -> DarwinControlledFixtureRootExpectation {
        let fd = Darwin.open(root.path, O_RDONLY | O_DIRECTORY | O_CLOEXEC)
        XCTAssertGreaterThanOrEqual(fd, 0)
        defer { XCTAssertEqual(Darwin.close(fd), 0) }

        var statValue = Darwin.stat()
        XCTAssertEqual(Darwin.fstat(fd, &statValue), 0)

        return DarwinControlledFixtureRootExpectation(
            hostAbsolutePathBytes: Data(root.path.utf8),
            nonce: nonce,
            expectedDevice: UInt64(statValue.st_dev),
            expectedInode: UInt64(statValue.st_ino),
            expectedMode: UInt32(statValue.st_mode),
            expectedUserID: UInt32(statValue.st_uid),
            expectedGroupID: UInt32(statValue.st_gid),
            expectedFlags: UInt32(statValue.st_flags),
            expectedGeneration: UInt32(statValue.st_gen),
            expectedModificationTimeSeconds:
                Int64(statValue.st_mtimespec.tv_sec),
            expectedModificationTimeNanoseconds:
                Int64(statValue.st_mtimespec.tv_nsec),
            expectedStatusChangeTimeSeconds:
                Int64(statValue.st_ctimespec.tv_sec),
            expectedStatusChangeTimeNanoseconds:
                Int64(statValue.st_ctimespec.tv_nsec),
            expectedBirthTimeSeconds:
                Int64(statValue.st_birthtimespec.tv_sec),
            expectedBirthTimeNanoseconds:
                Int64(statValue.st_birthtimespec.tv_nsec)
        )
    }

    static func rootExpectationLike(
        _ fixture: Fixture,
        hostAbsolutePathBytes: Data? = nil,
        nonce: String? = nil,
        expectedDevice: UInt64? = nil,
        expectedInode: UInt64? = nil,
        expectedMode: UInt32? = nil,
        expectedUserID: UInt32? = nil,
        expectedGroupID: UInt32? = nil,
        expectedFlags: UInt32? = nil,
        expectedGeneration: UInt32? = nil,
        expectedModificationTimeSeconds: Int64? = nil,
        expectedModificationTimeNanoseconds: Int64? = nil,
        expectedStatusChangeTimeSeconds: Int64? = nil,
        expectedStatusChangeTimeNanoseconds: Int64? = nil,
        expectedBirthTimeSeconds: Int64? = nil,
        expectedBirthTimeNanoseconds: Int64? = nil
    ) -> DarwinControlledFixtureRootExpectation {
        DarwinControlledFixtureRootExpectation(
            hostAbsolutePathBytes:
                hostAbsolutePathBytes ?? Data(fixture.rootURL.path.utf8),
            nonce: nonce ?? fixture.nonce,
            expectedDevice:
                expectedDevice ?? fixture.rootExpectation.expectedDevice,
            expectedInode:
                expectedInode ?? fixture.rootExpectation.expectedInode,
            expectedMode:
                expectedMode ?? fixture.rootExpectation.expectedMode,
            expectedUserID:
                expectedUserID ?? fixture.rootExpectation.expectedUserID,
            expectedGroupID:
                expectedGroupID ?? fixture.rootExpectation.expectedGroupID,
            expectedFlags:
                expectedFlags ?? fixture.rootExpectation.expectedFlags,
            expectedGeneration:
                expectedGeneration ??
                fixture.rootExpectation.expectedGeneration,
            expectedModificationTimeSeconds:
                expectedModificationTimeSeconds ??
                fixture.rootExpectation.expectedModificationTimeSeconds,
            expectedModificationTimeNanoseconds:
                expectedModificationTimeNanoseconds ??
                fixture.rootExpectation.expectedModificationTimeNanoseconds,
            expectedStatusChangeTimeSeconds:
                expectedStatusChangeTimeSeconds ??
                fixture.rootExpectation.expectedStatusChangeTimeSeconds,
            expectedStatusChangeTimeNanoseconds:
                expectedStatusChangeTimeNanoseconds ??
                fixture.rootExpectation.expectedStatusChangeTimeNanoseconds,
            expectedBirthTimeSeconds:
                expectedBirthTimeSeconds ??
                fixture.rootExpectation.expectedBirthTimeSeconds,
            expectedBirthTimeNanoseconds:
                expectedBirthTimeNanoseconds ??
                fixture.rootExpectation.expectedBirthTimeNanoseconds
        )
    }

    static func metadata(
        mode: UInt32 = 0o100600,
        flags: UInt32 = 0,
        extendedAttributeSupportMask: UInt64 = 1,
        extendedFlags: UInt64 = 0
    ) -> SyntheticCaptureFileMetadata {
        SyntheticCaptureFileMetadata(
            device: 7,
            inode: 11,
            mode: mode,
            linkCount: 1,
            userID: 501,
            groupID: 20,
            size: 32,
            blockCount: 8,
            blockSize: 4_096,
            flags: flags,
            generation: 0,
            modificationTimeSeconds: 1_700_000_000,
            modificationTimeNanoseconds: 1,
            statusChangeTimeSeconds: 1_700_000_000,
            statusChangeTimeNanoseconds: 2,
            birthTimeSeconds: 1_700_000_000,
            birthTimeNanoseconds: 3,
            extendedAttributeSupportMask: extendedAttributeSupportMask,
            extendedFlags: extendedFlags,
            cloneID: nil,
            cloneReferenceCount: nil
        )
    }

    static func statMetadata(
        for url: URL
    ) throws -> SyntheticCaptureFileMetadata {
        var statValue = Darwin.stat()
        guard Darwin.lstat(url.path, &statValue) == 0 else {
            throw posixError()
        }
        return SyntheticCaptureFileMetadata(
            device: UInt64(statValue.st_dev),
            inode: UInt64(statValue.st_ino),
            mode: UInt32(statValue.st_mode),
            linkCount: UInt64(statValue.st_nlink),
            userID: UInt32(statValue.st_uid),
            groupID: UInt32(statValue.st_gid),
            size: Int64(statValue.st_size),
            blockCount: Int64(statValue.st_blocks),
            blockSize: Int64(statValue.st_blksize),
            flags: UInt32(statValue.st_flags),
            generation: UInt32(statValue.st_gen),
            modificationTimeSeconds:
                Int64(statValue.st_mtimespec.tv_sec),
            modificationTimeNanoseconds:
                Int64(statValue.st_mtimespec.tv_nsec),
            statusChangeTimeSeconds:
                Int64(statValue.st_ctimespec.tv_sec),
            statusChangeTimeNanoseconds:
                Int64(statValue.st_ctimespec.tv_nsec),
            birthTimeSeconds:
                Int64(statValue.st_birthtimespec.tv_sec),
            birthTimeNanoseconds:
                Int64(statValue.st_birthtimespec.tv_nsec),
            extendedAttributeSupportMask: 1,
            extendedFlags: 0,
            cloneID: nil,
            cloneReferenceCount: nil
        )
    }

    static func firstMetadataDrift(
        _ before: SyntheticCaptureFileMetadata,
        _ after: SyntheticCaptureFileMetadata
    ) -> SyntheticSmallArtifactCaptureFailure.MetadataField? {
        if before.device != after.device { return .device }
        if before.inode != after.inode { return .inode }
        if before.mode != after.mode { return .mode }
        if before.linkCount != after.linkCount { return .linkCount }
        if before.userID != after.userID { return .userID }
        if before.groupID != after.groupID { return .groupID }
        if before.size != after.size { return .size }
        if before.blockCount != after.blockCount { return .blockCount }
        if before.blockSize != after.blockSize { return .blockSize }
        if before.flags != after.flags { return .flags }
        if before.generation != after.generation { return .generation }
        if before.modificationTimeSeconds != after.modificationTimeSeconds {
            return .modificationTimeSeconds
        }
        if before.modificationTimeNanoseconds !=
            after.modificationTimeNanoseconds
        {
            return .modificationTimeNanoseconds
        }
        if before.statusChangeTimeSeconds != after.statusChangeTimeSeconds {
            return .statusChangeTimeSeconds
        }
        if before.statusChangeTimeNanoseconds !=
            after.statusChangeTimeNanoseconds
        {
            return .statusChangeTimeNanoseconds
        }
        if before.birthTimeSeconds != after.birthTimeSeconds {
            return .birthTimeSeconds
        }
        if before.birthTimeNanoseconds != after.birthTimeNanoseconds {
            return .birthTimeNanoseconds
        }
        return nil
    }

    static func attributeBuffer(
        forkBits: UInt32,
        cloneID: UInt64? = nil,
        extFlags: UInt64? = nil,
        cloneReferenceCount: UInt32? = nil
    ) -> Data {
        var data = Data()
        appendUInt32(0, to: &data)
        appendUInt32(UInt32(ATTR_CMN_RETURNED_ATTRS), to: &data)
        appendUInt32(0, to: &data)
        appendUInt32(0, to: &data)
        appendUInt32(0, to: &data)
        appendUInt32(forkBits, to: &data)
        if let cloneID { appendUInt64(cloneID, to: &data) }
        if let extFlags { appendUInt64(extFlags, to: &data) }
        if let cloneReferenceCount {
            appendUInt32(cloneReferenceCount, to: &data)
        }
        data.replaceSubrange(0..<4, with: withUnsafeBytes(of: UInt32(data.count)) {
            Data($0)
        })
        return data
    }

    static func appendUInt32(_ value: UInt32, to data: inout Data) {
        var little = value.littleEndian
        withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
    }

    static func appendUInt64(_ value: UInt64, to data: inout Data) {
        var little = value.littleEndian
        withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
    }

    static func canonicalTemporaryDirectory() throws -> URL {
        guard let pointer = Darwin.realpath(NSTemporaryDirectory(), nil) else {
            throw posixError()
        }
        defer { Darwin.free(pointer) }
        return URL(fileURLWithPath: String(cString: pointer), isDirectory: true)
    }

    static func posixError() -> NSError {
        NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
    }

    static func lowercaseHexNonce() -> String {
        UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
    }

    static func absoluteLocator(
        componentBytes: Int = 1,
        componentCount: Int = 1
    ) -> Data {
        let component = String(repeating: "a", count: componentBytes)
        return Data(("/" + Array(repeating: component,
                                 count: componentCount).joined(separator: "/"))
            .utf8)
    }

    static func overlongAbsoluteLocator() -> Data {
        var bytes = Data([0x2f])
        bytes.append(Data(repeating: 0x61, count: 4_096))
        return bytes
    }

    static func componentCount(_ absolutePath: String) -> Int {
        absolutePath.split(separator: "/", omittingEmptySubsequences: true)
            .count
    }

    static func modeBits(for path: String) -> mode_t {
        var statValue = Darwin.stat()
        XCTAssertEqual(Darwin.lstat(path, &statValue), 0)
        return statValue.st_mode & 0o777
    }

    static func rewriteSameInode(_ url: URL, bytes: Data) throws {
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seek(toOffset: 0)
        try handle.write(contentsOf: bytes)
        try handle.truncate(atOffset: UInt64(bytes.count))
        try FileManager.default.setAttributes(
            [.modificationDate: Date(timeIntervalSince1970: 946_684_800)],
            ofItemAtPath: url.path
        )
    }

    static func appendFile(_ url: URL, bytes: Data) throws {
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: bytes)
    }

    static func removeFixtureRoot(_ root: URL) {
        try? FileManager.default.removeItem(at: root)
    }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func assertAllAuthorityFlagsFalse(
        _ comparison: SyntheticSmallArtifactCaptureComparison,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertFalse(comparison.canExecute, file: file, line: line)
        XCTAssertFalse(comparison.canSpawn, file: file, line: line)
        XCTAssertFalse(comparison.canAccessNetwork, file: file, line: line)
        XCTAssertFalse(comparison.canConsumePack, file: file, line: line)
        XCTAssertFalse(comparison.canMutateFileSystem, file: file, line: line)
        XCTAssertFalse(comparison.canImportGitObjects, file: file, line: line)
        XCTAssertFalse(comparison.canBuild, file: file, line: line)
        XCTAssertFalse(comparison.canLoadModel, file: file, line: line)
        XCTAssertFalse(comparison.canReserveOutput, file: file, line: line)
        XCTAssertFalse(comparison.canPublish, file: file, line: line)
    }

    static func assertRefuses<T>(
        _ label: String,
        _ expected: DarwinControlledFixtureSmallArtifactFailure,
        file: StaticString = #filePath,
        line: UInt = #line,
        _ body: () throws -> T
    ) {
        XCTAssertThrowsError(try body(), label, file: file, line: line) {
            error in
            XCTAssertEqual(
                error as? DarwinControlledFixtureSmallArtifactFailure,
                expected,
                file: file,
                line: line
            )
        }
    }
}

private extension Character {
    var isLowercaseHex: Bool {
        ("0"..."9").contains(self) || ("a"..."f").contains(self)
    }
}

/*
Removed compiler probes. Each snippet is intentionally kept outside XCTest and
must fail when copied into the named debug/release consumer probe:

// ProofControl or runner consumers cannot see any E3-A test-target type.
_ = DarwinControlledFixtureRootExpectation.self
_ = DarwinControlledFixtureTestControl.self
_ = DarwinControlledFixtureSmallArtifactFailure.self
_ = DarwinControlledFixtureSmallArtifactVerifier.self

// No default/root memberwise construction exists outside ProofControlTests.
_ = DarwinControlledFixtureRootExpectation()
_ = DarwinControlledFixtureTestControl()

// Existing E1 success remains sealed.
_ = SyntheticSmallArtifactCaptureComparison()

// No conversion or operational surface exists.
let comparison: SyntheticSmallArtifactCaptureComparison = fatalError()
_ = AdmittedFile(comparison)
_ = ExecutableContentIdentityEvidence(comparison)
_ = DynamicLoaderContentIdentityEvidence(comparison)
_ = SharedCacheImageContentEvidence(comparison)
_ = RuntimeClosureIdentityEvidence(comparison)
_ = FileImageExecutionIdentityComparison(comparison)
_ = comparison.absolutePath
_ = comparison.fileDescriptor
_ = comparison.reopen()
_ = comparison.callback
*/
