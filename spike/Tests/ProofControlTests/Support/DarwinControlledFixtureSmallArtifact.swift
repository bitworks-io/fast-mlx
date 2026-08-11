import CryptoKit
import Darwin
import Dispatch
import Foundation
@testable import ProofControl

struct DarwinControlledFixtureRootExpectation: Equatable {
    let hostAbsolutePathBytes: Data
    let nonce: String
    let expectedDevice: UInt64
    let expectedInode: UInt64
    let expectedMode: UInt32
    let expectedUserID: UInt32
    let expectedGroupID: UInt32
    let expectedFlags: UInt32
    let expectedGeneration: UInt32
    let expectedModificationTimeSeconds: Int64
    let expectedModificationTimeNanoseconds: Int64
    let expectedStatusChangeTimeSeconds: Int64
    let expectedStatusChangeTimeNanoseconds: Int64
    let expectedBirthTimeSeconds: Int64
    let expectedBirthTimeNanoseconds: Int64

    init(
        hostAbsolutePathBytes: Data,
        nonce: String,
        expectedDevice: UInt64,
        expectedInode: UInt64,
        expectedMode: UInt32,
        expectedUserID: UInt32,
        expectedGroupID: UInt32,
        expectedFlags: UInt32,
        expectedGeneration: UInt32,
        expectedModificationTimeSeconds: Int64,
        expectedModificationTimeNanoseconds: Int64,
        expectedStatusChangeTimeSeconds: Int64,
        expectedStatusChangeTimeNanoseconds: Int64,
        expectedBirthTimeSeconds: Int64,
        expectedBirthTimeNanoseconds: Int64
    ) {
        self.hostAbsolutePathBytes = Data(hostAbsolutePathBytes)
        self.nonce = nonce
        self.expectedDevice = expectedDevice
        self.expectedInode = expectedInode
        self.expectedMode = expectedMode
        self.expectedUserID = expectedUserID
        self.expectedGroupID = expectedGroupID
        self.expectedFlags = expectedFlags
        self.expectedGeneration = expectedGeneration
        self.expectedModificationTimeSeconds =
            expectedModificationTimeSeconds
        self.expectedModificationTimeNanoseconds =
            expectedModificationTimeNanoseconds
        self.expectedStatusChangeTimeSeconds =
            expectedStatusChangeTimeSeconds
        self.expectedStatusChangeTimeNanoseconds =
            expectedStatusChangeTimeNanoseconds
        self.expectedBirthTimeSeconds = expectedBirthTimeSeconds
        self.expectedBirthTimeNanoseconds = expectedBirthTimeNanoseconds
    }
}

struct DarwinControlledFixtureTestControl: Equatable {
    struct ArtificialInterruption: Equatable {
        let offset: UInt64
        let count: Int

        init(offset: UInt64, count: Int) {
            self.offset = offset
            self.count = count
        }
    }

    struct ForcedNegativeRead: Equatable {
        let offset: UInt64
        let errno: Int32

        init(offset: UInt64, errno: Int32) {
            self.offset = offset
            self.errno = errno
        }
    }

    enum Phase: Equatable {
        case afterRootValidated
        case afterFinalDescriptorOpened
        case afterBeforeMetadataValidated
        case afterFirstPositiveFragment
        case afterExpectedBytesRead
        case afterAfterMetadataRead
        case beforeCleanup
        case afterCleanup
        case beforeE1Comparison
    }

    final class PhaseBarrier: Equatable, @unchecked Sendable {
        let phase: Phase
        private let arrival = DispatchSemaphore(value: 0)
        private let resume = DispatchSemaphore(value: 0)

        init(phase: Phase) {
            self.phase = phase
        }

        static func == (
            lhs: PhaseBarrier,
            rhs: PhaseBarrier
        ) -> Bool {
            lhs === rhs
        }

        func waitForArrivalForTesting(timeoutMilliseconds: Int) -> Bool {
            let timeout = DispatchTime.now() +
                .milliseconds(timeoutMilliseconds)
            return arrival.wait(timeout: timeout) == .success
        }

        func signalResumeForTesting() {
            resume.signal()
        }

        fileprivate func arrive() {
            arrival.signal()
        }

        fileprivate func waitForResumeStep() -> Bool {
            resume.wait(timeout: DispatchTime.now() + .milliseconds(10)) ==
                .success
        }
    }

    let scriptedClockNanoseconds: [UInt64]?
    let artificialInterruptions: [ArtificialInterruption]
    let forcedNegativeRead: ForcedNegativeRead?
    let reportedCloseFailureOrdinal: Int?
    let phaseBarrier: PhaseBarrier?

    init(
        scriptedClockNanoseconds: [UInt64]? = nil,
        artificialInterruptions: [ArtificialInterruption] = [],
        forcedNegativeRead: ForcedNegativeRead? = nil,
        reportedCloseFailureOrdinal: Int? = nil,
        phaseBarrier: PhaseBarrier? = nil
    ) {
        self.scriptedClockNanoseconds = scriptedClockNanoseconds
        self.artificialInterruptions = artificialInterruptions
        self.forcedNegativeRead = forcedNegativeRead
        self.reportedCloseFailureOrdinal = reportedCloseFailureOrdinal
        self.phaseBarrier = phaseBarrier
    }
}

enum DarwinControlledFixtureSmallArtifactFailure:
    Error,
    Equatable
{
    enum LocatorKind: Equatable {
        case leadingSlash
        case trailingSlash
        case nul
        case emptyComponent(index: Int)
        case dotComponent(index: Int)
        case dotDotComponent(index: Int)
        case componentBytes(index: Int)
        case componentCount
        case byteCount
        case utf8
    }

    enum RootExpectationReason: Equatable {
        case nonce
        case basename
    }

    enum TestControlReason: Equatable {
        case clockAndBarrier
        case scriptedClockCount
        case interruptionCount(index: Int)
        case interruptionOrder(index: Int)
        case interruptionEntryCount
        case interruptionOffset(index: Int)
        case forcedReadErrno
        case forcedReadOffset
        case closeOrdinal
    }

    enum RequestReason: Equatable {
        case rootExpectation(RootExpectationReason)
        case rootLocator(LocatorKind)
        case logicalLocator(LocatorKind)
        case role
        case expectedByteCount
        case expectedSHA256
        case deadlineDuration
        case testControl(TestControlReason)
    }

    enum ResourceReason: Equatable {
        case arithmeticOverflow
        case descriptorStack
        case transcriptLimit
    }

    enum ClockReason: Equatable {
        case failure
        case decreased
    }

    enum DeadlinePhase: Equatable {
        case beforeHostRootOpen
        case afterHostRootOpen
        case beforeRootComponentOpen
        case afterRootComponentOpen
        case beforeDescriptorFlags
        case afterDescriptorFlags
        case beforeRootStat
        case afterRootStat
        case beforeLogicalComponentOpen
        case afterLogicalComponentOpen
        case afterFinalDescriptorOpened
        case beforeMetadataStat
        case afterMetadataStat
        case beforeMetadataAttributes
        case afterMetadataAttributes
        case afterBeforeMetadataValidated
        case beforeRead
        case afterRead
        case afterFirstPositiveFragment
        case afterExpectedBytesRead
        case beforeEOFProbe
        case afterEOFProbe
        case afterAfterMetadataRead
        case beforeRootRevalidation
        case afterRootRevalidation
        case afterRootValidated
        case beforeCleanup
        case afterCleanup
        case beforeE1Comparison
    }

    enum DeadlineReason: Equatable {
        case expired(DeadlinePhase)
    }

    enum DescriptorRole: Equatable {
        case hostRoot
        case rootComponent(index: Int)
        case fixtureRoot
        case logicalAncestor(index: Int)
        case finalFile
    }

    enum OpenReason: Equatable {
        case hostRoot(errno: Int32)
        case rootComponent(index: Int, errno: Int32)
        case logicalComponent(index: Int, errno: Int32)
    }

    enum DescriptorReason: Equatable {
        case closeOnExec(DescriptorRole)
    }

    enum RootIdentityField: Equatable {
        case expectedDevice
        case expectedInode
        case expectedUserID
        case expectedGroupID
        case expectedFlags
        case expectedGeneration
        case expectedModificationTimeSeconds
        case expectedModificationTimeNanoseconds
        case expectedStatusChangeTimeSeconds
        case expectedStatusChangeTimeNanoseconds
        case expectedBirthTimeSeconds
        case expectedBirthTimeNanoseconds
    }

    enum RootReason: Equatable {
        case stat(errno: Int32)
        case type
        case mode
        case identity(RootIdentityField)
    }

    enum MetadataPhase: Equatable {
        case before
        case after
    }

    enum MetadataReason: Equatable {
        case stat(MetadataPhase, errno: Int32)
        case attributes(MetadataPhase, errno: Int32)
        case attributeLength(MetadataPhase)
        case attributeBits(MetadataPhase)
        case attributeSupport(MetadataPhase)
        case attributeValue(MetadataPhase)
        case drift(SyntheticSmallArtifactCaptureFailure.MetadataField)
    }

    enum FileReason: Equatable {
        case type
        case linkCount
        case size
        case dataless
        case sparseStateUnavailable
        case sparse
    }

    enum ReadReason: Equatable {
        case interruptedLimit(offset: UInt64)
        case fragmentLimit(chunkOffset: UInt64)
        case system(offset: UInt64, errno: Int32)
        case eofProbeSystem(errno: Int32)
        case short(expected: UInt64, actual: UInt64)
        case trailingByte(offset: UInt64)
    }

    enum ExpectedReason: Equatable {
        case byteCount
        case sha256
    }

    enum CleanupReason: Equatable {
        case close(role: DescriptorRole, errno: Int32)
    }

    case request(RequestReason)
    case resource(ResourceReason)
    case clock(ClockReason)
    case deadline(DeadlineReason)
    case open(OpenReason)
    case descriptor(DescriptorReason)
    case root(RootReason)
    case metadata(MetadataReason)
    case file(FileReason)
    case read(ReadReason)
    case expected(ExpectedReason)
    case cleanup(CleanupReason)
    case testControlDidNotRefuse
    case e1(SyntheticSmallArtifactCaptureFailure)
}

enum DarwinControlledFixtureSmallArtifactVerifier {
    struct DecodedAttributes: Equatable {
        let extendedAttributeSupportMask: UInt64
        let extendedFlags: UInt64
        let cloneID: UInt64?
        let cloneReferenceCount: UInt32?
    }

    static func compare(
        rootExpectation: DarwinControlledFixtureRootExpectation,
        logicalLocatorBytes: Data,
        role: SyntheticSmallArtifactCaptureComparison.Role,
        expectedFileBytes: UInt64,
        expectedSHA256: String,
        deadlineMilliseconds: UInt64,
        control: DarwinControlledFixtureTestControl?
    ) throws -> SyntheticSmallArtifactCaptureComparison {
        let request = try Request.validate(
            rootExpectation: rootExpectation,
            logicalLocatorBytes: logicalLocatorBytes,
            role: role,
            expectedFileBytes: expectedFileBytes,
            expectedSHA256: expectedSHA256,
            deadlineMilliseconds: deadlineMilliseconds,
            control: control
        )
        var clock = Clock(control?.scriptedClockNanoseconds)
        let start = try clock.now()
        let duration = try multiplyMilliseconds(deadlineMilliseconds)
        let (deadline, overflow) = start.addingReportingOverflow(duration)
        guard !overflow else {
            throw DarwinControlledFixtureSmallArtifactFailure.resource(
                .arithmeticOverflow
            )
        }
        var context = Context(
            deadline: deadline,
            clock: clock,
            control: ControlState(control, expectedFileBytes: expectedFileBytes)
        )
        defer { clock = context.clock }

        var stack = DescriptorStack(control: context.control)
        var primaryFailure:
            DarwinControlledFixtureSmallArtifactFailure?
        var comparisonInput: ComparisonInput?

        do {
            comparisonInput = try capture(
                request: request,
                role: role,
                expectedFileBytes: expectedFileBytes,
                expectedSHA256: expectedSHA256,
                context: &context,
                stack: &stack
            )
        } catch let failure as DarwinControlledFixtureSmallArtifactFailure {
            primaryFailure = failure
        }

        stack.control = context.control
        let cleanupFailure = stack.cleanup()
        context.control = stack.control

        if let primaryFailure {
            throw primaryFailure
        }
        if let cleanupFailure {
            throw cleanupFailure
        }
        try context.reach(.afterCleanup)
        guard var comparisonInput else {
            throw DarwinControlledFixtureSmallArtifactFailure
                .testControlDidNotRefuse
        }
        try context.reach(.beforeE1Comparison)
        if context.clock.usesScriptedValues || !context.control.isSatisfied {
            throw DarwinControlledFixtureSmallArtifactFailure
                .testControlDidNotRefuse
        }

        do {
            comparisonInput.transcript = Array(comparisonInput.transcript)
            return try SyntheticSmallArtifactCaptureVerifier.compare(
                role: role,
                before: comparisonInput.before,
                after: comparisonInput.after,
                expectedFileBytes: expectedFileBytes,
                expectedSHA256: expectedSHA256,
                transcript: comparisonInput.transcript
            )
        } catch let failure as SyntheticSmallArtifactCaptureFailure {
            throw DarwinControlledFixtureSmallArtifactFailure.e1(failure)
        }
    }

    static func decodeAttributeBufferForTesting(
        _ data: Data
    ) throws -> DecodedAttributes {
        try decodeAttributeBuffer(data, phase: .before)
    }

    static func validateDescriptorFlagsForTesting(
        _ descriptorFlags: Int32
    ) throws {
        try validateDescriptorFlagsValue(descriptorFlags, role: .finalFile)
    }

    static func validateMappedMetadataForTesting(
        _ metadata: SyntheticCaptureFileMetadata
    ) throws {
        try validateMappedMetadata(metadata, expectedFileBytes: nil)
    }
}

private extension DarwinControlledFixtureSmallArtifactVerifier {
    struct Request {
        let rootExpectation: DarwinControlledFixtureRootExpectation
        let rootComponents: [Data]
        let logicalComponents: [Data]

        static func validate(
            rootExpectation: DarwinControlledFixtureRootExpectation,
            logicalLocatorBytes: Data,
            role: SyntheticSmallArtifactCaptureComparison.Role,
            expectedFileBytes: UInt64,
            expectedSHA256: String,
            deadlineMilliseconds: UInt64,
            control: DarwinControlledFixtureTestControl?
        ) throws -> Request {
            guard isLowercaseHexNonce(rootExpectation.nonce) else {
                throw DarwinControlledFixtureSmallArtifactFailure.request(
                    .rootExpectation(.nonce)
                )
            }
            let rootComponents = try validateLocator(
                rootExpectation.hostAbsolutePathBytes,
                kind: .root
            )
            guard let basename = rootComponents.last,
                  String(data: basename, encoding: .utf8) ==
                  "fast-mlx-e3a-\(rootExpectation.nonce)"
            else {
                throw DarwinControlledFixtureSmallArtifactFailure.request(
                    .rootExpectation(.basename)
                )
            }
            let logicalComponents = try validateLocator(
                logicalLocatorBytes,
                kind: .logical
            )
            switch role {
            case .gitRoot, .selfGuardRoot, .dynamicLoader, .fileImage:
                break
            }
            guard (1...1_048_576).contains(expectedFileBytes) else {
                throw DarwinControlledFixtureSmallArtifactFailure.request(
                    .expectedByteCount
                )
            }
            guard isLowercaseSHA256(expectedSHA256) else {
                throw DarwinControlledFixtureSmallArtifactFailure.request(
                    .expectedSHA256
                )
            }
            guard (1...3_600_000).contains(deadlineMilliseconds) else {
                throw DarwinControlledFixtureSmallArtifactFailure.request(
                    .deadlineDuration
                )
            }
            try validateControl(control, expectedFileBytes: expectedFileBytes)
            return Request(
                rootExpectation: rootExpectation,
                rootComponents: rootComponents,
                logicalComponents: logicalComponents
            )
        }
    }

    enum LocatorValidationKind {
        case root
        case logical

        func failure(
            _ reason: DarwinControlledFixtureSmallArtifactFailure.LocatorKind
        ) -> DarwinControlledFixtureSmallArtifactFailure {
            switch self {
            case .root:
                return .request(.rootLocator(reason))
            case .logical:
                return .request(.logicalLocator(reason))
            }
        }
    }

    struct Context {
        let deadline: UInt64
        var clock: Clock
        var control: ControlState

        mutating func checkDeadline(
            _ phase: DarwinControlledFixtureSmallArtifactFailure.DeadlinePhase
        ) throws {
            let now = try clock.now()
            if now >= deadline {
                throw DarwinControlledFixtureSmallArtifactFailure.deadline(
                    .expired(phase)
                )
            }
        }

        mutating func reach(
            _ phase: DarwinControlledFixtureTestControl.Phase
        ) throws {
            guard control.phaseBarrier?.phase == phase else { return }
            try checkDeadline(deadlinePhase(for: phase, before: true))
            control.phaseBarrier?.arrive()
            while !(control.phaseBarrier?.waitForResumeStep() ?? true) {
                try checkDeadline(deadlinePhase(for: phase, before: false))
            }
            control.markPhaseConsumed()
            try checkDeadline(deadlinePhase(for: phase, before: false))
        }
    }

    struct Clock {
        private var scripted: [UInt64]?
        private var index = 0
        private var last: UInt64?
        let usesScriptedValues: Bool

        init(_ scripted: [UInt64]?) {
            self.scripted = scripted
            usesScriptedValues = scripted != nil
        }

        mutating func now() throws -> UInt64 {
            let value: UInt64
            if let scripted {
                if index >= scripted.count {
                    throw DarwinControlledFixtureSmallArtifactFailure
                        .testControlDidNotRefuse
                }
                value = scripted[index]
                index += 1
            } else {
                value = Darwin.clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)
            }
            guard value != 0 else {
                throw DarwinControlledFixtureSmallArtifactFailure.clock(
                    .failure
                )
            }
            if let last, value < last {
                throw DarwinControlledFixtureSmallArtifactFailure.clock(
                    .decreased
                )
            }
            last = value
            return value
        }
    }

    struct ControlState {
        struct Interruption {
            let offset: UInt64
            var remaining: Int
        }

        var interruptions: [Interruption]
        var forcedNegativeRead: DarwinControlledFixtureTestControl
            .ForcedNegativeRead?
        var reportedCloseFailureOrdinal: Int?
        var actualCloseOrdinal = 0
        var phaseBarrier: DarwinControlledFixtureTestControl.PhaseBarrier?
        var phaseConsumed = false

        init(
            _ control: DarwinControlledFixtureTestControl?,
            expectedFileBytes _: UInt64
        ) {
            interruptions = control?.artificialInterruptions.map {
                Interruption(offset: $0.offset, remaining: $0.count)
            } ?? []
            forcedNegativeRead = control?.forcedNegativeRead
            reportedCloseFailureOrdinal = control?.reportedCloseFailureOrdinal
            phaseBarrier = control?.phaseBarrier
        }

        var isSatisfied: Bool {
            interruptions.allSatisfy { $0.remaining == 0 } &&
                forcedNegativeRead == nil &&
                reportedCloseFailureOrdinal == nil &&
                (phaseBarrier == nil || phaseConsumed)
        }

        mutating func consumeInterruptions(
            at offset: UInt64
        ) -> Int {
            guard let index = interruptions.firstIndex(where: {
                $0.offset == offset
            }) else { return 0 }
            let count = interruptions[index].remaining
            interruptions[index].remaining = 0
            return count
        }

        mutating func consumeForcedRead(
            at offset: UInt64
        ) -> Int32? {
            guard let forcedNegativeRead,
                  forcedNegativeRead.offset == offset
            else { return nil }
            self.forcedNegativeRead = nil
            return forcedNegativeRead.errno
        }

        mutating func closeErrorIfAny() -> Int32? {
            actualCloseOrdinal += 1
            guard reportedCloseFailureOrdinal == actualCloseOrdinal else {
                return nil
            }
            reportedCloseFailureOrdinal = nil
            return EIO
        }

        mutating func markPhaseConsumed() {
            phaseConsumed = true
        }
    }

    struct DescriptorEntry {
        let fd: Int32
        let role: DarwinControlledFixtureSmallArtifactFailure.DescriptorRole
    }

    struct DescriptorStack {
        var entries: [DescriptorEntry] = []
        var control: ControlState

        mutating func register(
            _ fd: Int32,
            role: DarwinControlledFixtureSmallArtifactFailure.DescriptorRole
        ) throws {
            guard entries.count < 129 else {
                throw DarwinControlledFixtureSmallArtifactFailure.resource(
                    .descriptorStack
                )
            }
            entries.append(DescriptorEntry(fd: fd, role: role))
        }

        mutating func cleanup()
            -> DarwinControlledFixtureSmallArtifactFailure?
        {
            var firstFailure:
                DarwinControlledFixtureSmallArtifactFailure?
            while let entry = entries.popLast() {
                let result = Darwin.close(entry.fd)
                if result != 0, firstFailure == nil {
                    firstFailure = .cleanup(
                        .close(role: entry.role, errno: errno)
                    )
                } else if result == 0,
                          let injected = control.closeErrorIfAny(),
                          firstFailure == nil
                {
                    firstFailure = .cleanup(
                        .close(role: entry.role, errno: injected)
                    )
                }
            }
            return firstFailure
        }
    }

    struct ComparisonInput {
        let before: SyntheticCaptureFileMetadata
        let after: SyntheticCaptureFileMetadata
        var transcript: [SyntheticCaptureReadEvent]
    }

    static func capture(
        request: Request,
        role _: SyntheticSmallArtifactCaptureComparison.Role,
        expectedFileBytes: UInt64,
        expectedSHA256: String,
        context: inout Context,
        stack: inout DescriptorStack
    ) throws -> ComparisonInput {
        try context.checkDeadline(.beforeHostRootOpen)
        let hostRoot = Darwin.open(
            "/",
            O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC
        )
        guard hostRoot >= 0 else {
            throw DarwinControlledFixtureSmallArtifactFailure.open(
                .hostRoot(errno: errno)
            )
        }
        try stack.register(hostRoot, role: .hostRoot)
        try context.checkDeadline(.afterHostRootOpen)
        try validateDescriptor(hostRoot, role: .hostRoot, context: &context)

        var currentRootFD = hostRoot
        let rootLast = request.rootComponents.count - 1
        for (index, component) in request.rootComponents.enumerated() {
            try context.checkDeadline(.beforeRootComponentOpen)
            let fd = component.withNullTerminatedCChars { chars in
                Darwin.openat(
                    currentRootFD,
                    chars,
                    O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC
                )
            }
            guard fd >= 0 else {
                throw DarwinControlledFixtureSmallArtifactFailure.open(
                    .rootComponent(
                        index: index,
                        errno: normalizedDirectoryOpenErrno(errno)
                    )
                )
            }
            let role: DarwinControlledFixtureSmallArtifactFailure
                .DescriptorRole =
                index == rootLast ? .fixtureRoot : .rootComponent(index: index)
            try stack.register(fd, role: role)
            currentRootFD = fd
            try context.checkDeadline(.afterRootComponentOpen)
            try validateDescriptor(fd, role: role, context: &context)
        }
        let fixtureRootFD = currentRootFD
        let expectedRoot = try validateRoot(
            fixtureRootFD,
            expectation: request.rootExpectation,
            context: &context
        )
        try context.reach(.afterRootValidated)

        var currentLogicalFD = fixtureRootFD
        let logicalLast = request.logicalComponents.count - 1
        var finalFD: Int32?
        for (index, component) in request.logicalComponents.enumerated() {
            let isFinal = index == logicalLast
            try context.checkDeadline(.beforeLogicalComponentOpen)
            let flags = isFinal
                ? O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC
                : O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC
            let fd = component.withNullTerminatedCChars { chars in
                Darwin.openat(currentLogicalFD, chars, flags)
            }
            guard fd >= 0 else {
                throw DarwinControlledFixtureSmallArtifactFailure.open(
                    .logicalComponent(
                        index: index,
                        errno: isFinal ? errno :
                            normalizedDirectoryOpenErrno(errno)
                    )
                )
            }
            let descriptorRole:
                DarwinControlledFixtureSmallArtifactFailure.DescriptorRole =
                isFinal ? .finalFile : .logicalAncestor(index: index)
            try stack.register(fd, role: descriptorRole)
            if !isFinal {
                currentLogicalFD = fd
            } else {
                finalFD = fd
            }
            try context.checkDeadline(.afterLogicalComponentOpen)
            try validateDescriptor(fd, role: descriptorRole, context: &context)
            if isFinal {
                try context.reach(.afterFinalDescriptorOpened)
            }
        }
        guard let finalFD else {
            throw DarwinControlledFixtureSmallArtifactFailure.request(
                .logicalLocator(.componentCount)
            )
        }

        let before = try metadata(
            for: finalFD,
            phase: .before,
            context: &context
        )
        try validateMappedMetadata(
            before,
            expectedFileBytes: expectedFileBytes
        )
        try context.reach(.afterBeforeMetadataValidated)

        var transcript = try readFile(
            fd: finalFD,
            expectedFileBytes: expectedFileBytes,
            context: &context
        )

        let after = try metadata(
            for: finalFD,
            phase: .after,
            context: &context
        )
        try context.reach(.afterAfterMetadataRead)
        if let drift = firstMetadataDrift(before, after) {
            throw DarwinControlledFixtureSmallArtifactFailure.metadata(
                .drift(drift)
            )
        }
        try validateExpectedBytesAndDigest(
            transcript: transcript,
            expectedFileBytes: expectedFileBytes,
            expectedSHA256: expectedSHA256
        )
        try revalidateRoot(
            fixtureRootFD,
            expected: expectedRoot,
            expectation: request.rootExpectation,
            context: &context
        )
        try context.reach(.beforeCleanup)
        transcript = Array(transcript)
        return ComparisonInput(before: before, after: after, transcript: transcript)
    }

    static func validateDescriptor(
        _ fd: Int32,
        role: DarwinControlledFixtureSmallArtifactFailure.DescriptorRole,
        context: inout Context
    ) throws {
        try context.checkDeadline(.beforeDescriptorFlags)
        let flags = Darwin.fcntl(fd, F_GETFD)
        guard flags >= 0 else {
            throw DarwinControlledFixtureSmallArtifactFailure.descriptor(
                .closeOnExec(role)
            )
        }
        try context.checkDeadline(.afterDescriptorFlags)
        try validateDescriptorFlagsValue(flags, role: role)
    }

    static func validateDescriptorFlagsValue(
        _ flags: Int32,
        role: DarwinControlledFixtureSmallArtifactFailure.DescriptorRole
    ) throws {
        guard flags & FD_CLOEXEC != 0 else {
            throw DarwinControlledFixtureSmallArtifactFailure.descriptor(
                .closeOnExec(role)
            )
        }
    }

    struct RootTuple: Equatable {
        let device: UInt64
        let inode: UInt64
        let mode: UInt32
        let userID: UInt32
        let groupID: UInt32
        let flags: UInt32
        let generation: UInt32
        let modificationTimeSeconds: Int64
        let modificationTimeNanoseconds: Int64
        let statusChangeTimeSeconds: Int64
        let statusChangeTimeNanoseconds: Int64
        let birthTimeSeconds: Int64
        let birthTimeNanoseconds: Int64
    }

    static func validateRoot(
        _ fd: Int32,
        expectation: DarwinControlledFixtureRootExpectation,
        context: inout Context
    ) throws -> RootTuple {
        let tuple = try rootTuple(fd, context: &context)
        try validateRootTuple(tuple, expectation: expectation)
        return tuple
    }

    static func revalidateRoot(
        _ fd: Int32,
        expected: RootTuple,
        expectation: DarwinControlledFixtureRootExpectation,
        context: inout Context
    ) throws {
        try context.checkDeadline(.beforeRootRevalidation)
        let tuple = try rootTuple(fd, context: &context)
        try context.checkDeadline(.afterRootRevalidation)
        try validateRootTuple(tuple, expectation: expectation)
        if tuple != expected {
            try validateRootTuple(tuple, expectation: expectation)
        }
    }

    static func rootTuple(
        _ fd: Int32,
        context: inout Context
    ) throws -> RootTuple {
        try context.checkDeadline(.beforeRootStat)
        var statValue = Darwin.stat()
        guard Darwin.fstat(fd, &statValue) == 0 else {
            throw DarwinControlledFixtureSmallArtifactFailure.root(
                .stat(errno: errno)
            )
        }
        try context.checkDeadline(.afterRootStat)
        return RootTuple(
            device: UInt64(statValue.st_dev),
            inode: UInt64(statValue.st_ino),
            mode: UInt32(statValue.st_mode),
            userID: UInt32(statValue.st_uid),
            groupID: UInt32(statValue.st_gid),
            flags: UInt32(statValue.st_flags),
            generation: UInt32(statValue.st_gen),
            modificationTimeSeconds: Int64(statValue.st_mtimespec.tv_sec),
            modificationTimeNanoseconds: Int64(statValue.st_mtimespec.tv_nsec),
            statusChangeTimeSeconds: Int64(statValue.st_ctimespec.tv_sec),
            statusChangeTimeNanoseconds: Int64(statValue.st_ctimespec.tv_nsec),
            birthTimeSeconds: Int64(statValue.st_birthtimespec.tv_sec),
            birthTimeNanoseconds: Int64(statValue.st_birthtimespec.tv_nsec)
        )
    }

    static func validateRootTuple(
        _ tuple: RootTuple,
        expectation: DarwinControlledFixtureRootExpectation
    ) throws {
        guard tuple.mode & UInt32(S_IFMT) == UInt32(S_IFDIR) else {
            throw DarwinControlledFixtureSmallArtifactFailure.root(.type)
        }
        guard tuple.mode & 0o777 == 0o700,
              tuple.mode == expectation.expectedMode
        else {
            throw DarwinControlledFixtureSmallArtifactFailure.root(.mode)
        }
        if tuple.device != expectation.expectedDevice {
            throw DarwinControlledFixtureSmallArtifactFailure.root(
                .identity(.expectedDevice)
            )
        }
        if tuple.inode != expectation.expectedInode {
            throw DarwinControlledFixtureSmallArtifactFailure.root(
                .identity(.expectedInode)
            )
        }
        if tuple.userID != expectation.expectedUserID {
            throw DarwinControlledFixtureSmallArtifactFailure.root(
                .identity(.expectedUserID)
            )
        }
        if tuple.groupID != expectation.expectedGroupID {
            throw DarwinControlledFixtureSmallArtifactFailure.root(
                .identity(.expectedGroupID)
            )
        }
        if tuple.flags != expectation.expectedFlags {
            throw DarwinControlledFixtureSmallArtifactFailure.root(
                .identity(.expectedFlags)
            )
        }
        if tuple.generation != expectation.expectedGeneration {
            throw DarwinControlledFixtureSmallArtifactFailure.root(
                .identity(.expectedGeneration)
            )
        }
        if tuple.modificationTimeSeconds !=
            expectation.expectedModificationTimeSeconds
        {
            throw DarwinControlledFixtureSmallArtifactFailure.root(
                .identity(.expectedModificationTimeSeconds)
            )
        }
        if tuple.modificationTimeNanoseconds !=
            expectation.expectedModificationTimeNanoseconds
        {
            throw DarwinControlledFixtureSmallArtifactFailure.root(
                .identity(.expectedModificationTimeNanoseconds)
            )
        }
        if tuple.statusChangeTimeSeconds !=
            expectation.expectedStatusChangeTimeSeconds
        {
            throw DarwinControlledFixtureSmallArtifactFailure.root(
                .identity(.expectedStatusChangeTimeSeconds)
            )
        }
        if tuple.statusChangeTimeNanoseconds !=
            expectation.expectedStatusChangeTimeNanoseconds
        {
            throw DarwinControlledFixtureSmallArtifactFailure.root(
                .identity(.expectedStatusChangeTimeNanoseconds)
            )
        }
        if tuple.birthTimeSeconds != expectation.expectedBirthTimeSeconds {
            throw DarwinControlledFixtureSmallArtifactFailure.root(
                .identity(.expectedBirthTimeSeconds)
            )
        }
        if tuple.birthTimeNanoseconds !=
            expectation.expectedBirthTimeNanoseconds
        {
            throw DarwinControlledFixtureSmallArtifactFailure.root(
                .identity(.expectedBirthTimeNanoseconds)
            )
        }
    }

    static func metadata(
        for fd: Int32,
        phase: DarwinControlledFixtureSmallArtifactFailure.MetadataPhase,
        context: inout Context
    ) throws -> SyntheticCaptureFileMetadata {
        try context.checkDeadline(.beforeMetadataStat)
        var statValue = Darwin.stat()
        guard Darwin.fstat(fd, &statValue) == 0 else {
            throw DarwinControlledFixtureSmallArtifactFailure.metadata(
                .stat(phase, errno: errno)
            )
        }
        try context.checkDeadline(.afterMetadataStat)

        try context.checkDeadline(.beforeMetadataAttributes)
        var request = attrlist()
        request.bitmapcount = UInt16(ATTR_BIT_MAP_COUNT)
        request.reserved = 0
        request.commonattr = UInt32(ATTR_CMN_RETURNED_ATTRS)
        request.volattr = 0
        request.dirattr = 0
        request.fileattr = 0
        request.forkattr = UInt32(
            ATTR_CMNEXT_CLONEID |
                ATTR_CMNEXT_EXT_FLAGS |
                ATTR_CMNEXT_CLONE_REFCNT
        )
        var bytes = [UInt8](repeating: 0, count: 44)
        let result = bytes.withUnsafeMutableBytes { rawBuffer in
            Darwin.fgetattrlist(
                fd,
                &request,
                rawBuffer.baseAddress,
                rawBuffer.count,
                UInt32(FSOPT_ATTR_CMN_EXTENDED)
            )
        }
        guard result == 0 else {
            throw DarwinControlledFixtureSmallArtifactFailure.metadata(
                .attributes(phase, errno: errno)
            )
        }
        try context.checkDeadline(.afterMetadataAttributes)
        let length = try decodeUInt32(bytes, offset: 0, phase: phase)
        guard let lengthInt = Int(exactly: length),
              lengthInt <= bytes.count
        else {
            throw DarwinControlledFixtureSmallArtifactFailure.metadata(
                .attributeLength(phase)
            )
        }
        let decoded = try decodeAttributeBuffer(
            Data(bytes.prefix(lengthInt)),
            phase: phase
        )
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
            modificationTimeSeconds: Int64(statValue.st_mtimespec.tv_sec),
            modificationTimeNanoseconds: Int64(statValue.st_mtimespec.tv_nsec),
            statusChangeTimeSeconds: Int64(statValue.st_ctimespec.tv_sec),
            statusChangeTimeNanoseconds: Int64(statValue.st_ctimespec.tv_nsec),
            birthTimeSeconds: Int64(statValue.st_birthtimespec.tv_sec),
            birthTimeNanoseconds: Int64(statValue.st_birthtimespec.tv_nsec),
            extendedAttributeSupportMask:
                decoded.extendedAttributeSupportMask,
            extendedFlags: decoded.extendedFlags,
            cloneID: decoded.cloneID,
            cloneReferenceCount: decoded.cloneReferenceCount.map(UInt64.init)
        )
    }

    static func decodeAttributeBuffer(
        _ data: Data,
        phase: DarwinControlledFixtureSmallArtifactFailure.MetadataPhase
    ) throws -> DecodedAttributes {
        guard (24...44).contains(data.count) else {
            throw DarwinControlledFixtureSmallArtifactFailure.metadata(
                .attributeLength(phase)
            )
        }
        let totalLength = try decodeUInt32(data, offset: 0, phase: phase)
        guard totalLength == UInt32(data.count) else {
            throw DarwinControlledFixtureSmallArtifactFailure.metadata(
                .attributeLength(phase)
            )
        }
        let common = try decodeUInt32(data, offset: 4, phase: phase)
        let volume = try decodeUInt32(data, offset: 8, phase: phase)
        let directory = try decodeUInt32(data, offset: 12, phase: phase)
        let file = try decodeUInt32(data, offset: 16, phase: phase)
        let fork = try decodeUInt32(data, offset: 20, phase: phase)

        let requestedCommon = UInt32(ATTR_CMN_RETURNED_ATTRS)
        let requestedFork = UInt32(
            ATTR_CMNEXT_CLONEID |
                ATTR_CMNEXT_EXT_FLAGS |
                ATTR_CMNEXT_CLONE_REFCNT
        )
        guard common & ~requestedCommon == 0,
              common & requestedCommon != 0,
              volume == 0,
              directory == 0,
              file == 0,
              fork & ~requestedFork == 0
        else {
            throw DarwinControlledFixtureSmallArtifactFailure.metadata(
                .attributeBits(phase)
            )
        }
        guard fork & UInt32(ATTR_CMNEXT_EXT_FLAGS) != 0 else {
            throw DarwinControlledFixtureSmallArtifactFailure.metadata(
                .attributeSupport(phase)
            )
        }

        var offset = 24
        var cloneID: UInt64?
        var rawExtFlags: UInt64?
        var cloneReferenceCount: UInt32?
        var support: UInt64 = 0

        if fork & UInt32(ATTR_CMNEXT_CLONEID) != 0 {
            cloneID = try decodeUInt64(data, offset: offset, phase: phase)
            offset += 8
            support |= 1 << 1
        }
        if fork & UInt32(ATTR_CMNEXT_EXT_FLAGS) != 0 {
            rawExtFlags = try decodeUInt64(data, offset: offset, phase: phase)
            offset += 8
            support |= 1 << 0
        }
        if fork & UInt32(ATTR_CMNEXT_CLONE_REFCNT) != 0 {
            cloneReferenceCount = try decodeUInt32(
                data,
                offset: offset,
                phase: phase
            )
            offset += 4
            support |= 1 << 2
        }
        guard offset == data.count,
              let rawExtFlags,
              rawExtFlags <= UInt64.max >> 1
        else {
            throw DarwinControlledFixtureSmallArtifactFailure.metadata(
                .attributeValue(phase)
            )
        }
        let sparseBit: UInt64 =
            rawExtFlags & UInt64(EF_IS_SPARSE) == 0 ? 0 : 1
        return DecodedAttributes(
            extendedAttributeSupportMask: support,
            extendedFlags: (rawExtFlags << 1) | sparseBit,
            cloneID: cloneID,
            cloneReferenceCount: cloneReferenceCount
        )
    }

    static func validateMappedMetadata(
        _ metadata: SyntheticCaptureFileMetadata,
        expectedFileBytes: UInt64?
    ) throws {
        guard metadata.mode & UInt32(S_IFMT) == UInt32(S_IFREG) else {
            throw DarwinControlledFixtureSmallArtifactFailure.file(.type)
        }
        guard metadata.linkCount == 1 else {
            throw DarwinControlledFixtureSmallArtifactFailure.file(.linkCount)
        }
        guard metadata.size > 0,
              metadata.size <= 1_048_576
        else {
            throw DarwinControlledFixtureSmallArtifactFailure.file(.size)
        }
        guard metadata.flags & 0x4000_0000 == 0 else {
            throw DarwinControlledFixtureSmallArtifactFailure.file(.dataless)
        }
        guard metadata.extendedAttributeSupportMask & 1 != 0 else {
            throw DarwinControlledFixtureSmallArtifactFailure.file(
                .sparseStateUnavailable
            )
        }
        guard metadata.extendedFlags & 1 == 0 else {
            throw DarwinControlledFixtureSmallArtifactFailure.file(.sparse)
        }
        guard let expectedFileBytes else { return }
        guard UInt64(metadata.size) == expectedFileBytes else {
            throw DarwinControlledFixtureSmallArtifactFailure.expected(
                .byteCount
            )
        }
        _ = try Int(exactly: expectedFileBytes).orThrowResource()
        _ = try off_t(exactly: expectedFileBytes).orThrowResource()
        _ = try SyntheticSmallArtifactCaptureVerifier.checkedReadAttemptLimit(
            forFileBytes: expectedFileBytes
        )
    }

    static func readFile(
        fd: Int32,
        expectedFileBytes: UInt64,
        context: inout Context
    ) throws -> [SyntheticCaptureReadEvent] {
        var transcript: [SyntheticCaptureReadEvent] = []
        let maxEvents = try SyntheticSmallArtifactCaptureVerifier
            .checkedReadAttemptLimit(forFileBytes: expectedFileBytes)
        guard maxEvents <= 2_305 else {
            throw DarwinControlledFixtureSmallArtifactFailure.resource(
                .transcriptLimit
            )
        }
        transcript.reserveCapacity(Int(maxEvents))

        var offset: UInt64 = 0
        var chunkOffset: UInt64 = 0
        var chunkEnd = min(UInt64(65_536), expectedFileBytes)
        var fragmentsInChunk = 0
        var didFirstPositiveFragment = false
        var interruptionOffset: UInt64?
        var interruptionsAtOffset = 0

        while offset < expectedFileBytes {
            if interruptionOffset != offset {
                interruptionOffset = offset
                interruptionsAtOffset = 0
            }
            let requested = try Int(
                exactly: min(UInt64(65_536), chunkEnd - offset)
            ).orThrowResource()
            var interruptions = context.control.consumeInterruptions(at: offset)
            while interruptions > 0 {
                try context.checkDeadline(.beforeRead)
                let (nextCount, overflow) = interruptionsAtOffset
                    .addingReportingOverflow(1)
                guard !overflow else {
                    throw DarwinControlledFixtureSmallArtifactFailure.resource(
                        .arithmeticOverflow
                    )
                }
                interruptionsAtOffset = nextCount
                try context.checkDeadline(.afterRead)
                guard interruptionsAtOffset <= 8 else {
                    throw DarwinControlledFixtureSmallArtifactFailure.read(
                        .interruptedLimit(offset: offset)
                    )
                }
                try appendEvent(
                    .interrupted(offset: offset),
                    to: &transcript,
                    maximum: maxEvents
                )
                interruptions -= 1
            }

            try context.checkDeadline(.beforeRead)
            if let forcedErrno = context.control.consumeForcedRead(at: offset) {
                try context.checkDeadline(.afterRead)
                throw DarwinControlledFixtureSmallArtifactFailure.read(
                    .system(offset: offset, errno: forcedErrno)
                )
            }

            var buffer = [UInt8](repeating: 0, count: requested)
            let result = buffer.withUnsafeMutableBytes { rawBuffer in
                Darwin.pread(
                    fd,
                    rawBuffer.baseAddress,
                    requested,
                    off_t(offset)
                )
            }
            let readErrno = result < 0 ? errno : 0
            try context.checkDeadline(.afterRead)
            if result < 0, readErrno == EINTR {
                let (nextCount, overflow) = interruptionsAtOffset
                    .addingReportingOverflow(1)
                guard !overflow else {
                    throw DarwinControlledFixtureSmallArtifactFailure.resource(
                        .arithmeticOverflow
                    )
                }
                interruptionsAtOffset = nextCount
                guard interruptionsAtOffset <= 8 else {
                    throw DarwinControlledFixtureSmallArtifactFailure.read(
                        .interruptedLimit(offset: offset)
                    )
                }
                try appendEvent(
                    .interrupted(offset: offset),
                    to: &transcript,
                    maximum: maxEvents
                )
                continue
            }
            guard result >= 0 else {
                throw DarwinControlledFixtureSmallArtifactFailure.read(
                    .system(offset: offset, errno: readErrno)
                )
            }
            guard result > 0 else {
                throw DarwinControlledFixtureSmallArtifactFailure.read(
                    .short(expected: expectedFileBytes, actual: offset)
                )
            }
            guard result <= requested else {
                throw DarwinControlledFixtureSmallArtifactFailure.read(
                    .trailingByte(offset: offset)
                )
            }
            let (nextFragmentCount, fragmentOverflow) = fragmentsInChunk
                .addingReportingOverflow(1)
            guard !fragmentOverflow else {
                throw DarwinControlledFixtureSmallArtifactFailure.resource(
                    .arithmeticOverflow
                )
            }
            fragmentsInChunk = nextFragmentCount
            guard fragmentsInChunk <= 16 else {
                throw DarwinControlledFixtureSmallArtifactFailure.read(
                    .fragmentLimit(chunkOffset: chunkOffset)
                )
            }
            let data = Data(buffer.prefix(result))
            try appendEvent(
                .bytes(offset: offset, data: data),
                to: &transcript,
                maximum: maxEvents
            )
            let (nextOffset, offsetOverflow) = offset.addingReportingOverflow(
                UInt64(result)
            )
            guard !offsetOverflow else {
                throw DarwinControlledFixtureSmallArtifactFailure.resource(
                    .arithmeticOverflow
                )
            }
            offset = nextOffset
            if !didFirstPositiveFragment {
                didFirstPositiveFragment = true
                try context.reach(.afterFirstPositiveFragment)
            }
            if offset == chunkEnd, offset < expectedFileBytes {
                chunkOffset = offset
                let remaining = expectedFileBytes - offset
                let (nextChunkEnd, chunkOverflow) = offset
                    .addingReportingOverflow(min(UInt64(65_536), remaining))
                guard !chunkOverflow else {
                    throw DarwinControlledFixtureSmallArtifactFailure.resource(
                        .arithmeticOverflow
                    )
                }
                chunkEnd = nextChunkEnd
                fragmentsInChunk = 0
            }
        }

        try context.reach(.afterExpectedBytesRead)
        try context.checkDeadline(.beforeEOFProbe)
        if let forcedErrno = context.control.consumeForcedRead(at: offset) {
            try context.checkDeadline(.afterEOFProbe)
            throw DarwinControlledFixtureSmallArtifactFailure.read(
                .eofProbeSystem(errno: forcedErrno)
            )
        }
        var eofByte = [UInt8](repeating: 0, count: 1)
        let eof = eofByte.withUnsafeMutableBytes { rawBuffer in
            Darwin.pread(fd, rawBuffer.baseAddress, 1, off_t(offset))
        }
        guard eof == 0 else {
            let code = eof < 0 ? errno : 0
            try context.checkDeadline(.afterEOFProbe)
            if eof < 0 {
                throw DarwinControlledFixtureSmallArtifactFailure.read(
                    .eofProbeSystem(errno: code)
                )
            }
            throw DarwinControlledFixtureSmallArtifactFailure.read(
                .trailingByte(offset: offset)
            )
        }
        try context.checkDeadline(.afterEOFProbe)
        try appendEvent(
            .endOfFile(offset: offset),
            to: &transcript,
            maximum: maxEvents
        )

        return transcript
    }

    static func appendEvent(
        _ event: SyntheticCaptureReadEvent,
        to transcript: inout [SyntheticCaptureReadEvent],
        maximum: UInt64
    ) throws {
        guard UInt64(transcript.count) < maximum else {
            throw DarwinControlledFixtureSmallArtifactFailure.resource(
                .transcriptLimit
            )
        }
        transcript.append(event)
    }

    static func validateExpectedBytesAndDigest(
        transcript: [SyntheticCaptureReadEvent],
        expectedFileBytes: UInt64,
        expectedSHA256: String
    ) throws {
        var retained = Data()
        var hasher = SHA256()
        for event in transcript {
            guard case .bytes(_, let data) = event else { continue }
            retained.append(data)
            hasher.update(data: data)
        }
        guard UInt64(retained.count) == expectedFileBytes else {
            throw DarwinControlledFixtureSmallArtifactFailure.expected(
                .byteCount
            )
        }
        let digest = hasher.finalize().map { String(format: "%02x", $0) }
            .joined()
        guard digest == expectedSHA256 else {
            throw DarwinControlledFixtureSmallArtifactFailure.expected(.sha256)
        }
    }
}

private extension DarwinControlledFixtureSmallArtifactVerifier {
    static func validateLocator(
        _ data: Data,
        kind: LocatorValidationKind
    ) throws -> [Data] {
        guard data.count >= 2, data.first == 0x2f else {
            throw kind.failure(.leadingSlash)
        }
        guard data.count <= 4_096 else {
            throw kind.failure(.byteCount)
        }
        guard data.last != 0x2f else {
            throw kind.failure(.trailingSlash)
        }
        guard !data.contains(0) else {
            throw kind.failure(.nul)
        }
        guard String(data: data, encoding: .utf8) != nil else {
            throw kind.failure(.utf8)
        }

        var components: [Data] = []
        var componentStart = data.index(after: data.startIndex)
        var componentIndex = 0
        var cursor = componentStart
        while cursor <= data.endIndex {
            if cursor == data.endIndex || data[cursor] == 0x2f {
                let component = data[componentStart..<cursor]
                if component.isEmpty {
                    throw kind.failure(.emptyComponent(index: componentIndex))
                }
                if component.count > 255 {
                    throw kind.failure(
                        .componentBytes(index: componentIndex)
                    )
                }
                if component.elementsEqual([0x2e]) {
                    throw kind.failure(.dotComponent(index: componentIndex))
                }
                if component.elementsEqual([0x2e, 0x2e]) {
                    throw kind.failure(
                        .dotDotComponent(index: componentIndex)
                    )
                }
                components.append(Data(component))
                componentIndex += 1
                if cursor == data.endIndex { break }
                componentStart = data.index(after: cursor)
            }
            cursor = data.index(after: cursor)
        }
        guard (1...64).contains(components.count) else {
            throw kind.failure(.componentCount)
        }
        return components
    }

    static func isLowercaseHexNonce(_ value: String) -> Bool {
        let bytes = Array(value.utf8)
        return bytes.count == 32 && bytes.allSatisfy {
            (0x30...0x39).contains($0) || (0x61...0x66).contains($0)
        }
    }

    static func isLowercaseSHA256(_ value: String) -> Bool {
        let bytes = Array(value.utf8)
        return bytes.count == 64 && bytes.allSatisfy {
            (0x30...0x39).contains($0) || (0x61...0x66).contains($0)
        }
    }

    static func validateControl(
        _ control: DarwinControlledFixtureTestControl?,
        expectedFileBytes: UInt64
    ) throws {
        guard let control else { return }
        if control.scriptedClockNanoseconds != nil,
           control.phaseBarrier != nil
        {
            throw DarwinControlledFixtureSmallArtifactFailure.request(
                .testControl(.clockAndBarrier)
            )
        }
        if let values = control.scriptedClockNanoseconds,
           !(1...8_192).contains(values.count)
        {
            throw DarwinControlledFixtureSmallArtifactFailure.request(
                .testControl(.scriptedClockCount)
            )
        }
        guard control.artificialInterruptions.count <= 256 else {
            throw DarwinControlledFixtureSmallArtifactFailure.request(
                .testControl(.interruptionEntryCount)
            )
        }
        var previous: UInt64?
        for (index, interruption) in
            control.artificialInterruptions.enumerated()
        {
            guard (1...9).contains(interruption.count) else {
                throw DarwinControlledFixtureSmallArtifactFailure.request(
                    .testControl(.interruptionCount(index: index))
                )
            }
            guard interruption.offset < expectedFileBytes else {
                throw DarwinControlledFixtureSmallArtifactFailure.request(
                    .testControl(.interruptionOffset(index: index))
                )
            }
            if let previous, interruption.offset <= previous {
                throw DarwinControlledFixtureSmallArtifactFailure.request(
                    .testControl(.interruptionOrder(index: index))
                )
            }
            previous = interruption.offset
        }
        if let forced = control.forcedNegativeRead {
            guard forced.errno > 0 else {
                throw DarwinControlledFixtureSmallArtifactFailure.request(
                    .testControl(.forcedReadErrno)
                )
            }
            guard forced.offset <= expectedFileBytes else {
                throw DarwinControlledFixtureSmallArtifactFailure.request(
                    .testControl(.forcedReadOffset)
                )
            }
        }
        if let ordinal = control.reportedCloseFailureOrdinal,
           !(1...129).contains(ordinal)
        {
            throw DarwinControlledFixtureSmallArtifactFailure.request(
                .testControl(.closeOrdinal)
            )
        }
    }

    static func multiplyMilliseconds(_ milliseconds: UInt64) throws -> UInt64 {
        let (duration, overflow) = milliseconds.multipliedReportingOverflow(
            by: 1_000_000
        )
        guard !overflow else {
            throw DarwinControlledFixtureSmallArtifactFailure.resource(
                .arithmeticOverflow
            )
        }
        return duration
    }

    static func normalizedDirectoryOpenErrno(_ code: Int32) -> Int32 {
        code == ENOTDIR ? ELOOP : code
    }

    static func deadlinePhase(
        for phase: DarwinControlledFixtureTestControl.Phase,
        before: Bool
    ) -> DarwinControlledFixtureSmallArtifactFailure.DeadlinePhase {
        switch phase {
        case .afterRootValidated:
            return .afterRootValidated
        case .afterFinalDescriptorOpened:
            return .afterFinalDescriptorOpened
        case .afterBeforeMetadataValidated:
            return .afterBeforeMetadataValidated
        case .afterFirstPositiveFragment:
            return .afterFirstPositiveFragment
        case .afterExpectedBytesRead:
            return .afterExpectedBytesRead
        case .afterAfterMetadataRead:
            return .afterAfterMetadataRead
        case .beforeCleanup:
            return .beforeCleanup
        case .afterCleanup:
            return .afterCleanup
        case .beforeE1Comparison:
            return .beforeE1Comparison
        }
    }

    static func decodeUInt32(
        _ bytes: [UInt8],
        offset: Int,
        phase: DarwinControlledFixtureSmallArtifactFailure.MetadataPhase
    ) throws -> UInt32 {
        try decodeUInt32(Data(bytes), offset: offset, phase: phase)
    }

    static func decodeUInt32(
        _ data: Data,
        offset: Int,
        phase: DarwinControlledFixtureSmallArtifactFailure.MetadataPhase
    ) throws -> UInt32 {
        guard offset >= 0, offset + 4 <= data.count else {
            throw DarwinControlledFixtureSmallArtifactFailure.metadata(
                .attributeLength(phase)
            )
        }
        var value: UInt32 = 0
        withUnsafeMutableBytes(of: &value) { destination in
            destination.copyBytes(from: data[offset..<(offset + 4)])
        }
        return UInt32(littleEndian: value)
    }

    static func decodeUInt64(
        _ data: Data,
        offset: Int,
        phase: DarwinControlledFixtureSmallArtifactFailure.MetadataPhase
    ) throws -> UInt64 {
        guard offset >= 0, offset + 8 <= data.count else {
            throw DarwinControlledFixtureSmallArtifactFailure.metadata(
                .attributeLength(phase)
            )
        }
        var value: UInt64 = 0
        withUnsafeMutableBytes(of: &value) { destination in
            destination.copyBytes(from: data[offset..<(offset + 8)])
        }
        return UInt64(littleEndian: value)
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
        if before.modificationTimeSeconds !=
            after.modificationTimeSeconds
        {
            return .modificationTimeSeconds
        }
        if before.modificationTimeNanoseconds !=
            after.modificationTimeNanoseconds
        {
            return .modificationTimeNanoseconds
        }
        if before.statusChangeTimeSeconds !=
            after.statusChangeTimeSeconds
        {
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
        if before.extendedAttributeSupportMask !=
            after.extendedAttributeSupportMask
        {
            return .extendedAttributeSupportMask
        }
        if before.extendedFlags != after.extendedFlags {
            return .extendedFlags
        }
        if before.cloneID != after.cloneID { return .cloneID }
        if before.cloneReferenceCount != after.cloneReferenceCount {
            return .cloneReferenceCount
        }
        return nil
    }
}

private extension Data {
    func withNullTerminatedCChars<T>(
        _ body: (UnsafePointer<CChar>) -> T
    ) -> T {
        var chars = map { CChar(bitPattern: $0) }
        chars.append(0)
        return chars.withUnsafeBufferPointer { buffer in
            body(buffer.baseAddress!)
        }
    }
}

private extension Optional where Wrapped == Int {
    func orThrowResource() throws -> Int {
        guard let self else {
            throw DarwinControlledFixtureSmallArtifactFailure.resource(
                .arithmeticOverflow
            )
        }
        return self
    }
}

private extension Optional where Wrapped == off_t {
    func orThrowResource() throws -> off_t {
        guard let self else {
            throw DarwinControlledFixtureSmallArtifactFailure.resource(
                .arithmeticOverflow
            )
        }
        return self
    }
}
