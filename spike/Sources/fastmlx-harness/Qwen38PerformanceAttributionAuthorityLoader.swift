import CryptoKit
import Darwin
import Foundation

import HarnessCore

enum Qwen38PerformanceAttributionAuthorityFileRole:
    Equatable, Sendable
{
    case policy
    case productionRouteReceipt
}

enum Qwen38PerformanceAttributionAuthorityError:
    Error, Equatable, CustomStringConvertible, Sendable
{
    case unreadableAuthorityInput(Qwen38PerformanceAttributionAuthorityFileRole)
    case nonRegularAuthorityInput(Qwen38PerformanceAttributionAuthorityFileRole)
    case oversizedAuthorityInput(Qwen38PerformanceAttributionAuthorityFileRole)
    case unstableAuthorityInput(Qwen38PerformanceAttributionAuthorityFileRole)
    case aliasedAuthorityInput(Qwen38PerformanceAttributionAuthorityFileRole)
    case duplicateJSONKey(Qwen38PerformanceAttributionAuthorityFileRole)
    case excessiveJSONNesting(Qwen38PerformanceAttributionAuthorityFileRole)
    case malformedPin(Qwen38PerformanceAttributionAuthorityFileRole)
    case bodyHashPin(Qwen38PerformanceAttributionAuthorityFileRole)
    case pinMismatch(Qwen38PerformanceAttributionAuthorityFileRole)
    case invalidAuthorityBody(Qwen38PerformanceAttributionAuthorityFileRole)

    var description: String {
        switch self {
        case .unreadableAuthorityInput(let role):
            return "unreadable authority input: \(role.name)"
        case .nonRegularAuthorityInput(let role):
            return "non-regular authority input: \(role.name)"
        case .oversizedAuthorityInput(let role):
            return "oversized authority input: \(role.name)"
        case .unstableAuthorityInput(let role):
            return "unstable authority input: \(role.name)"
        case .aliasedAuthorityInput(let role):
            return "authority inputs alias: \(role.name)"
        case .duplicateJSONKey(let role):
            return "duplicate JSON key in authority input: \(role.name)"
        case .excessiveJSONNesting(let role):
            return "excessive JSON nesting in authority input: \(role.name)"
        case .malformedPin(let role):
            return "malformed authority digest pin: \(role.name)"
        case .bodyHashPin(let role):
            return "authority pin names mutable body hash: \(role.name)"
        case .pinMismatch(let role):
            return "authority digest pin mismatch: \(role.name)"
        case .invalidAuthorityBody(let role):
            return "invalid authority body: \(role.name)"
        }
    }
}

struct Qwen38PerformanceAttributionAuthorityFileInputs: Equatable, Sendable {
    var policyBody: URL
    var policyPin: URL
    var productionRouteReceiptBody: URL
    var productionRouteReceiptPin: URL

    init(
        policyBody: URL,
        policyPin: URL,
        productionRouteReceiptBody: URL,
        productionRouteReceiptPin: URL
    ) {
        self.policyBody = policyBody
        self.policyPin = policyPin
        self.productionRouteReceiptBody = productionRouteReceiptBody
        self.productionRouteReceiptPin = productionRouteReceiptPin
    }
}

struct Qwen38PerformanceAttributionCapturedFileIdentity:
    Equatable, Sendable
{
    var device: UInt64
    var file: UInt64

    init(device: UInt64, file: UInt64) {
        self.device = device
        self.file = file
    }
}

struct Qwen38PerformanceAttributionUnsignedAuthorityCapture:
    Equatable, Sendable
{
    let policyDocumentSHA256: String
    let policyPinDocumentSHA256: String
    let policyBodyIdentity: Qwen38PerformanceAttributionCapturedFileIdentity
    let policyPinIdentity: Qwen38PerformanceAttributionCapturedFileIdentity
    let productionRouteReceiptDocumentSHA256: String
    let productionRouteReceiptPinDocumentSHA256: String
    let productionRouteReceiptBodyIdentity:
        Qwen38PerformanceAttributionCapturedFileIdentity
    let productionRouteReceiptPinIdentity:
        Qwen38PerformanceAttributionCapturedFileIdentity
    let semanticPolicyDigest: String
    let productionRouteReceiptDigest: String
    let runIdentityDigest: String
    let backendBuildIdentityDigest: String
    let observationDigest: String
    let promotionAuthorized: Bool
    let requiresIndependentControllerSignature: Bool

    init(
        policyDocumentSHA256: String,
        policyPinDocumentSHA256: String,
        policyBodyIdentity: Qwen38PerformanceAttributionCapturedFileIdentity,
        policyPinIdentity: Qwen38PerformanceAttributionCapturedFileIdentity,
        productionRouteReceiptDocumentSHA256: String,
        productionRouteReceiptPinDocumentSHA256: String,
        productionRouteReceiptBodyIdentity:
            Qwen38PerformanceAttributionCapturedFileIdentity,
        productionRouteReceiptPinIdentity:
            Qwen38PerformanceAttributionCapturedFileIdentity,
        semanticPolicyDigest: String,
        productionRouteReceiptDigest: String,
        runIdentityDigest: String,
        backendBuildIdentityDigest: String,
        observationDigest: String
    ) {
        self.policyDocumentSHA256 = policyDocumentSHA256
        self.policyPinDocumentSHA256 = policyPinDocumentSHA256
        self.policyBodyIdentity = policyBodyIdentity
        self.policyPinIdentity = policyPinIdentity
        self.productionRouteReceiptDocumentSHA256 =
            productionRouteReceiptDocumentSHA256
        self.productionRouteReceiptPinDocumentSHA256 =
            productionRouteReceiptPinDocumentSHA256
        self.productionRouteReceiptBodyIdentity = productionRouteReceiptBodyIdentity
        self.productionRouteReceiptPinIdentity = productionRouteReceiptPinIdentity
        self.semanticPolicyDigest = semanticPolicyDigest
        self.productionRouteReceiptDigest = productionRouteReceiptDigest
        self.runIdentityDigest = runIdentityDigest
        self.backendBuildIdentityDigest = backendBuildIdentityDigest
        self.observationDigest = observationDigest
        promotionAuthorized = false
        requiresIndependentControllerSignature = true
    }
}

enum Qwen38PerformanceAttributionAuthorityLoader {
    static let maxAuthorityBodyBytes = 8_388_608
    static let maxAuthorityPinBytes = 128
    static let maxAuthorityJSONNestingDepth = 64
#if DEBUG
    nonisolated(unsafe) static var afterDescriptorReadTestHook:
        ((String) throws -> Void)?
#endif

    static func load(
        files: Qwen38PerformanceAttributionAuthorityFileInputs
    ) throws -> Qwen38PerformanceAttributionUnsignedAuthorityCapture {
        let captured = try capture(files: files)

        let policyBody = captured[.policyBody]!.data
        let receiptBody = captured[.productionRouteReceiptBody]!.data
        let policyPinData = captured[.policyPin]!.data
        let receiptPinData = captured[.productionRouteReceiptPin]!.data
        let policyPin = try parsePin(
            policyPinData,
            body: policyBody,
            role: .policy)
        let receiptPin = try parsePin(
            receiptPinData,
            body: receiptBody,
            role: .productionRouteReceipt)

        let policy = try decodePolicy(policyBody)
        let receipt = try decodeReceipt(receiptBody)

        try validate(policy: policy, expectedDigest: policyPin)
        try validate(
            receipt: receipt,
            policy: policy,
            expectedDigest: receiptPin)

        return Qwen38PerformanceAttributionUnsignedAuthorityCapture(
            policyDocumentSHA256: sha256Hex(policyBody),
            policyPinDocumentSHA256: sha256Hex(policyPinData),
            policyBodyIdentity: captured[.policyBody]!.identity.captured,
            policyPinIdentity: captured[.policyPin]!.identity.captured,
            productionRouteReceiptDocumentSHA256: sha256Hex(receiptBody),
            productionRouteReceiptPinDocumentSHA256: sha256Hex(receiptPinData),
            productionRouteReceiptBodyIdentity:
                captured[.productionRouteReceiptBody]!.identity.captured,
            productionRouteReceiptPinIdentity:
                captured[.productionRouteReceiptPin]!.identity.captured,
            semanticPolicyDigest: policy.digest,
            productionRouteReceiptDigest: receipt.digest,
            runIdentityDigest: receipt.runIdentityDigest,
            backendBuildIdentityDigest: receipt.backendBuildIdentityDigest,
            observationDigest: receipt.observationDigest)
    }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func rejectAlias(
        _ lhs: OpenAuthorityInput,
        _ rhs: OpenAuthorityInput
    ) throws {
        guard lhs.identity != rhs.identity else {
            throw Qwen38PerformanceAttributionAuthorityError
                .aliasedAuthorityInput(rhs.slot.role)
        }
    }

    private static func parsePin(
        _ data: Data,
        body: Data,
        role: Qwen38PerformanceAttributionAuthorityFileRole
    ) throws -> String {
        guard let raw = String(data: data, encoding: .utf8) else {
            throw Qwen38PerformanceAttributionAuthorityError.malformedPin(role)
        }
        let pin = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard isLowerHex(pin, count: 64) else {
            throw Qwen38PerformanceAttributionAuthorityError.malformedPin(role)
        }
        guard pin != sha256Hex(body) else {
            throw Qwen38PerformanceAttributionAuthorityError.bodyHashPin(role)
        }
        return pin
    }

    private static func decodePolicy(
        _ body: Data
    ) throws -> Qwen38PerformanceAttributionFrozenPromotionPolicy {
        try preflightAuthorityJSON(
            body,
            role: .policy,
            schema: .policyAuthority)
        do {
            return try JSONDecoder().decode(
                Qwen38PerformanceAttributionFrozenPromotionPolicy.self,
                from: body)
        } catch {
            throw Qwen38PerformanceAttributionAuthorityError
                .invalidAuthorityBody(.policy)
        }
    }

    private static func decodeReceipt(
        _ body: Data
    ) throws -> Qwen38PerformanceAttributionProductionRouteReceipt {
        try preflightAuthorityJSON(
            body,
            role: .productionRouteReceipt,
            schema: .productionRouteReceiptAuthority)
        do {
            return try JSONDecoder().decode(
                Qwen38PerformanceAttributionProductionRouteReceipt.self,
                from: body)
        } catch {
            throw Qwen38PerformanceAttributionAuthorityError
                .invalidAuthorityBody(.productionRouteReceipt)
        }
    }

    private static func validate(
        policy: Qwen38PerformanceAttributionFrozenPromotionPolicy,
        expectedDigest: String
    ) throws {
        guard policy.schemaVersion == Qwen38PerformanceAttributionScorecardGate
            .schemaVersion,
            isLowerHex(policy.evidenceID, count: 64),
            policy.sealedBeforeMeasurements,
            policy.artifact == Qwen38MTPPerformanceScorecardGate.requiredArtifact,
            policy.claimAuthorities.map(\.claimKind)
                == Qwen38PerformanceAttributionScorecardGate.requiredClaimKinds
        else {
            throw Qwen38PerformanceAttributionAuthorityError
                .invalidAuthorityBody(.policy)
        }
        try validateRunIdentity(policy.runIdentity, artifact: policy.artifact)
        try validateClaimAuthorities(policy.claimAuthorities)
        guard policy.digest == expectedDigest else {
            throw Qwen38PerformanceAttributionAuthorityError.pinMismatch(.policy)
        }
        let recomputed = Qwen38PerformanceAttributionScorecardGate
            .frozenPromotionPolicy(
                evidenceID: policy.evidenceID,
                artifact: policy.artifact,
                runIdentity: policy.runIdentity,
                claimAuthorities: policy.claimAuthorities)
        guard policy == recomputed else {
            throw Qwen38PerformanceAttributionAuthorityError
                .invalidAuthorityBody(.policy)
        }
    }

    private static func validateRunIdentity(
        _ run: Qwen38MTPPerformanceScorecardTrustedRunIdentity,
        artifact: Qwen38MTPPerformanceScorecardArtifact
    ) throws {
        guard run.measurementClass
            == Qwen38PerformanceAttributionScorecardGate
                .flagshipMeasurementClass,
            !run.hardwareChip.isEmpty,
            run.hardwareRAMBytes
                >= Qwen38PerformanceAttributionScorecardGate
                    .flagshipMinimumRAMBytes,
            !run.hardwareOSBuild.isEmpty,
            isLowerHex(run.hostIdentityDigest, count: 64),
            isLowerHex(run.harnessGitSHA, count: 40),
            !run.candidateMLXSwiftVersion.isEmpty,
            !run.modelLabel.isEmpty,
            run.modelConfigHash == artifact.targetConfigSHA256,
            run.modelCheckpointManifestHash == artifact.targetTensorManifestSHA256,
            run.modelQuant == ModelQuantInfo(
                bits: artifact.targetQuantizationBits,
                groupSize: artifact.targetQuantizationGroupSize),
            !run.corpusID.isEmpty,
            isLowerHex(run.corpusContentHash, count: 64)
        else {
            throw Qwen38PerformanceAttributionAuthorityError
                .invalidAuthorityBody(.policy)
        }
    }

    private static func validateClaimAuthorities(
        _ authorities: [Qwen38PerformanceAttributionFrozenClaimAuthority]
    ) throws {
        for authority in authorities {
            try validateAbsoluteAuthority(
                authority.absoluteAuthority,
                kind: authority.claimKind)
            try validateCleanupAuthority(
                authority.cleanupAuthority,
                kind: authority.claimKind)
        }
    }

    private static func validateAbsoluteAuthority(
        _ authority: Qwen38PerformanceAttributionAbsoluteAuthority,
        kind: Qwen38PerformanceAttributionClaimKind
    ) throws {
        guard isLowerHex(authority.evidenceID, count: 64),
            isLowerHex(authority.digest, count: 64),
            authority.sealedBeforeMeasurements,
            !authority.bands.isEmpty
        else {
            throw Qwen38PerformanceAttributionAuthorityError
                .invalidAuthorityBody(.policy)
        }
        let recomputed = Qwen38PerformanceAttributionScorecardGate
            .absoluteAuthority(
                evidenceID: authority.evidenceID,
                sealedBeforeMeasurements: authority.sealedBeforeMeasurements,
                bands: authority.bands)
        guard authority == recomputed else {
            throw Qwen38PerformanceAttributionAuthorityError
                .invalidAuthorityBody(.policy)
        }
        var keys = Set<String>()
        for band in authority.bands {
            guard band.claimKind == kind,
                band.concurrency > 0,
                band.maxPrefillSeconds.isFinite,
                band.maxPrefillSeconds > 0,
                band.maxTTFTSeconds.isFinite,
                band.maxTTFTSeconds > 0,
                band.minDecodeTokensPerSecond.isFinite,
                band.minDecodeTokensPerSecond > 0,
                band.minAggregateThroughputTokensPerSecond.isFinite,
                band.minAggregateThroughputTokensPerSecond > 0,
                keys.insert(bandKey(band)).inserted
            else {
                throw Qwen38PerformanceAttributionAuthorityError
                    .invalidAuthorityBody(.policy)
            }
        }
    }

    private static func validateCleanupAuthority(
        _ authority: Qwen38PerformanceAttributionCleanupAuthority,
        kind: Qwen38PerformanceAttributionClaimKind
    ) throws {
        guard isLowerHex(authority.evidenceID, count: 64),
            isLowerHex(authority.digest, count: 64),
            authority.minIdleSamples > 0,
            authority.cooldownSeconds.isFinite,
            authority.cooldownSeconds > 0,
            !authority.allowedPressureStates.isEmpty,
            !authority.allowedThermalStates.isEmpty
        else {
            throw Qwen38PerformanceAttributionAuthorityError
                .invalidAuthorityBody(.policy)
        }
        let recomputed = Qwen38PerformanceAttributionScorecardGate
            .cleanupAuthority(
                evidenceID: authority.evidenceID,
                minIdleSamples: authority.minIdleSamples,
                cooldownSeconds: authority.cooldownSeconds,
                maxRSSDeltaBytes: authority.maxRSSDeltaBytes,
                maxActiveMetalDeltaBytes: authority.maxActiveMetalDeltaBytes,
                maxCachedMetalDeltaBytes: authority.maxCachedMetalDeltaBytes,
                maxSwapDeltaBytes: authority.maxSwapDeltaBytes,
                maxPageoutDelta: authority.maxPageoutDelta,
                allowedPressureStates: authority.allowedPressureStates,
                allowedThermalStates: authority.allowedThermalStates)
        guard authority == recomputed else {
            throw Qwen38PerformanceAttributionAuthorityError
                .invalidAuthorityBody(.policy)
        }
        _ = kind
    }

    private static func validate(
        receipt: Qwen38PerformanceAttributionProductionRouteReceipt,
        policy: Qwen38PerformanceAttributionFrozenPromotionPolicy,
        expectedDigest: String
    ) throws {
        guard receipt.schemaVersion
            == Qwen38PerformanceAttributionScorecardGate.schemaVersion,
            isLowerHex(receipt.evidenceID, count: 64),
            receipt.evidenceID != policy.evidenceID,
            receipt.artifact == policy.artifact,
            isLowerHex(receipt.backendBuildIdentityDigest, count: 64),
            isLowerHex(receipt.observationDigest, count: 64)
        else {
            throw Qwen38PerformanceAttributionAuthorityError
                .invalidAuthorityBody(.productionRouteReceipt)
        }
        guard receipt.digest == expectedDigest else {
            throw Qwen38PerformanceAttributionAuthorityError
                .pinMismatch(.productionRouteReceipt)
        }
        guard receipt.runIdentityDigest == Qwen38PerformanceAttributionScorecardGate
            .canonicalDigest(policy.runIdentity)
        else {
            throw Qwen38PerformanceAttributionAuthorityError
                .invalidAuthorityBody(.productionRouteReceipt)
        }
        let recomputed = Qwen38PerformanceAttributionScorecardGate
            .productionRouteReceipt(
                evidenceID: receipt.evidenceID,
                artifact: receipt.artifact,
                runIdentity: policy.runIdentity,
                backendBuildIdentityDigest: receipt.backendBuildIdentityDigest,
                observationDigest: receipt.observationDigest)
        guard receipt == recomputed else {
            throw Qwen38PerformanceAttributionAuthorityError
                .invalidAuthorityBody(.productionRouteReceipt)
        }
    }

    private static func capture(
        files: Qwen38PerformanceAttributionAuthorityFileInputs
    ) throws -> [AuthorityInputSlot: CapturedAuthorityInput] {
        var opened: [OpenAuthorityInput] = []
        defer {
            for input in opened {
                _ = Darwin.close(input.descriptor)
            }
        }
        var result: [AuthorityInputSlot: CapturedAuthorityInput] = [:]
        for slot in AuthorityInputSlot.allCases {
            opened.append(try openInput(url: slot.url(in: files), slot: slot))
        }
        try rejectPairwiseAliases(opened)
        for input in opened {
            result[input.slot] = try capture(input)
        }
        return result
    }

    private static func rejectPairwiseAliases(
        _ opened: [OpenAuthorityInput]
    ) throws {
        for index in opened.indices {
            for otherIndex in opened.index(after: index) ..< opened.endIndex {
                try rejectAlias(opened[index], opened[otherIndex])
            }
        }
    }

    private static func openInput(
        url: URL,
        slot: AuthorityInputSlot
    ) throws -> OpenAuthorityInput {
        let path = url.path
        let descriptor = Darwin.open(path, O_RDONLY | O_NONBLOCK | O_NOFOLLOW)
        guard descriptor >= 0 else {
            let error = errno == ELOOP
                ? Qwen38PerformanceAttributionAuthorityError
                    .nonRegularAuthorityInput(slot.role)
                : Qwen38PerformanceAttributionAuthorityError
                    .unreadableAuthorityInput(slot.role)
            throw error
        }

        do {
            let before = try statDescriptor(descriptor, role: slot.role)
            guard before.isRegularFile else {
                throw Qwen38PerformanceAttributionAuthorityError
                    .nonRegularAuthorityInput(slot.role)
            }
            return OpenAuthorityInput(
                descriptor: descriptor,
                slot: slot,
                identity: before.stableIdentity,
                before: before)
        } catch {
            _ = Darwin.close(descriptor)
            throw error
        }
    }

    private static func capture(
        _ input: OpenAuthorityInput
    ) throws -> CapturedAuthorityInput {
        let before = input.before
        guard before.size >= 0, before.size <= input.slot.maxBytes else {
            throw Qwen38PerformanceAttributionAuthorityError
                .oversizedAuthorityInput(input.slot.role)
        }

        let data = try readDescriptor(
            input.descriptor,
            byteCount: Int(before.size),
            role: input.slot.role)
#if DEBUG
        try afterDescriptorReadTestHook?(input.slot.name)
#endif
        let after = try statDescriptor(input.descriptor, role: input.slot.role)
        guard before.stableIdentity == after.stableIdentity,
            before.size == after.size,
            before.modifiedSeconds == after.modifiedSeconds,
            before.modifiedNanoseconds == after.modifiedNanoseconds,
            before.changedSeconds == after.changedSeconds,
            before.changedNanoseconds == after.changedNanoseconds
        else {
            throw Qwen38PerformanceAttributionAuthorityError
                .unstableAuthorityInput(input.slot.role)
        }
        return CapturedAuthorityInput(
            role: input.slot.role,
            identity: before.stableIdentity,
            data: data)
    }

    private static func statDescriptor(
        _ descriptor: Int32,
        role: Qwen38PerformanceAttributionAuthorityFileRole
    ) throws -> FileStatSnapshot {
        var status = stat()
        guard Darwin.fstat(descriptor, &status) == 0 else {
            throw Qwen38PerformanceAttributionAuthorityError
                .unreadableAuthorityInput(role)
        }
        return FileStatSnapshot(status)
    }

    private static func readDescriptor(
        _ descriptor: Int32,
        byteCount: Int,
        role: Qwen38PerformanceAttributionAuthorityFileRole
    ) throws -> Data {
        var data = Data(count: byteCount)
        var offset = 0
        while offset < byteCount {
            let count = data.withUnsafeMutableBytes { buffer -> Int in
                guard let base = buffer.baseAddress else { return -1 }
                return Darwin.read(
                    descriptor,
                    base.advanced(by: offset),
                    byteCount - offset)
            }
            guard count > 0 else {
                throw Qwen38PerformanceAttributionAuthorityError
                    .unstableAuthorityInput(role)
            }
            offset += count
        }
        return data
    }

    private static func preflightAuthorityJSON(
        _ data: Data,
        role: Qwen38PerformanceAttributionAuthorityFileRole,
        schema: JSONAuthoritySchema
    ) throws {
        let bytes = Array(data)
        var scanner = JSONAuthorityPreflightScanner(
            bytes: bytes,
            maxDepth: maxAuthorityJSONNestingDepth)
        switch scanner.scan(schema: schema) {
        case nil:
            return
        case .duplicateKey:
            throw Qwen38PerformanceAttributionAuthorityError
                .duplicateJSONKey(role)
        case .excessiveNesting:
            throw Qwen38PerformanceAttributionAuthorityError
                .excessiveJSONNesting(role)
        case .invalid:
            throw Qwen38PerformanceAttributionAuthorityError
                .invalidAuthorityBody(role)
        }
    }

    private static func isLowerHex(_ value: String, count: Int) -> Bool {
        value.count == count && value.utf8.allSatisfy {
            ($0 >= 48 && $0 <= 57) || ($0 >= 97 && $0 <= 102)
        }
    }

    private static func bandKey(
        _ band: Qwen38PerformanceAttributionAbsoluteBand
    ) -> String {
        "\(band.claimKind.rawValue):\(band.concurrency):\(band.contextTokens.rawValue):\(band.prefixKind.rawValue)"
    }
}

private enum AuthorityInputSlot: CaseIterable, Hashable {
    case policyBody
    case policyPin
    case productionRouteReceiptBody
    case productionRouteReceiptPin

    var role: Qwen38PerformanceAttributionAuthorityFileRole {
        switch self {
        case .policyBody, .policyPin:
            return .policy
        case .productionRouteReceiptBody, .productionRouteReceiptPin:
            return .productionRouteReceipt
        }
    }

    var maxBytes: Int64 {
        switch self {
        case .policyBody, .productionRouteReceiptBody:
            return Int64(Qwen38PerformanceAttributionAuthorityLoader
                .maxAuthorityBodyBytes)
        case .policyPin, .productionRouteReceiptPin:
            return Int64(Qwen38PerformanceAttributionAuthorityLoader
                .maxAuthorityPinBytes)
        }
    }

    func url(in files: Qwen38PerformanceAttributionAuthorityFileInputs) -> URL {
        switch self {
        case .policyBody:
            return files.policyBody
        case .policyPin:
            return files.policyPin
        case .productionRouteReceiptBody:
            return files.productionRouteReceiptBody
        case .productionRouteReceiptPin:
            return files.productionRouteReceiptPin
        }
    }

    var name: String {
        switch self {
        case .policyBody:
            return "policyBody"
        case .policyPin:
            return "policyPin"
        case .productionRouteReceiptBody:
            return "productionRouteReceiptBody"
        case .productionRouteReceiptPin:
            return "productionRouteReceiptPin"
        }
    }
}

private struct CapturedAuthorityInput {
    var role: Qwen38PerformanceAttributionAuthorityFileRole
    var identity: FileIdentity
    var data: Data
}

private struct OpenAuthorityInput {
    var descriptor: Int32
    var slot: AuthorityInputSlot
    var identity: FileIdentity
    var before: FileStatSnapshot
}

private struct FileIdentity: Equatable, Hashable {
    var device: UInt64
    var file: UInt64

    var captured: Qwen38PerformanceAttributionCapturedFileIdentity {
        Qwen38PerformanceAttributionCapturedFileIdentity(
            device: device,
            file: file)
    }
}

private struct FileStatSnapshot {
    var stableIdentity: FileIdentity
    var mode: mode_t
    var size: off_t
    var modifiedSeconds: Int
    var modifiedNanoseconds: Int
    var changedSeconds: Int
    var changedNanoseconds: Int

    init(_ status: stat) {
        stableIdentity = FileIdentity(
            device: UInt64(status.st_dev),
            file: UInt64(status.st_ino))
        mode = status.st_mode
        size = status.st_size
        modifiedSeconds = status.st_mtimespec.tv_sec
        modifiedNanoseconds = status.st_mtimespec.tv_nsec
        changedSeconds = status.st_ctimespec.tv_sec
        changedNanoseconds = status.st_ctimespec.tv_nsec
    }

    var isRegularFile: Bool {
        (mode & S_IFMT) == S_IFREG
    }
}

private indirect enum JSONAuthoritySchema: Sendable {
    case scalar
    case array(JSONAuthoritySchema)
    case object([String: JSONAuthoritySchema])
}

private extension JSONAuthoritySchema {
    static let artifact = JSONAuthoritySchema.object([
        "blockSize": .scalar,
        "depth": .scalar,
        "drafterConfigSHA256": .scalar,
        "drafterQuantizationBits": .scalar,
        "drafterQuantizationGroupSize": .scalar,
        "drafterQuantizationMode": .scalar,
        "drafterRevision": .scalar,
        "drafterTensorManifestSHA256": .scalar,
        "lockSourceRevision": .scalar,
        "maxAcceptedDrafts": .scalar,
        "targetConfigSHA256": .scalar,
        "targetQuantizationBits": .scalar,
        "targetQuantizationGroupSize": .scalar,
        "targetQuantizationMode": .scalar,
        "targetRevision": .scalar,
        "targetTensorManifestSHA256": .scalar,
        "tokenizerSHA256": .scalar,
    ])
    static let modelQuant = JSONAuthoritySchema.object([
        "bits": .scalar,
        "groupSize": .scalar,
    ])
    static let runIdentity = JSONAuthoritySchema.object([
        "candidateMLXSwiftVersion": .scalar,
        "corpusContentHash": .scalar,
        "corpusID": .scalar,
        "hardwareChip": .scalar,
        "hardwareOSBuild": .scalar,
        "hardwareRAMBytes": .scalar,
        "harnessGitSHA": .scalar,
        "hostIdentityDigest": .scalar,
        "measurementClass": .scalar,
        "modelCheckpointManifestHash": .scalar,
        "modelConfigHash": .scalar,
        "modelLabel": .scalar,
        "modelQuant": .modelQuant,
        "referenceMLXLMVersion": .scalar,
        "referenceMLXVersion": .scalar,
    ])
    static let absoluteBand = JSONAuthoritySchema.object([
        "claimKind": .scalar,
        "concurrency": .scalar,
        "contextTokens": .scalar,
        "maxPrefillSeconds": .scalar,
        "maxTTFTSeconds": .scalar,
        "minAggregateThroughputTokensPerSecond": .scalar,
        "minDecodeTokensPerSecond": .scalar,
        "prefixKind": .scalar,
    ])
    static let absoluteAuthority = JSONAuthoritySchema.object([
        "bands": .array(.absoluteBand),
        "digest": .scalar,
        "evidenceID": .scalar,
        "sealedBeforeMeasurements": .scalar,
    ])
    static let cleanupAuthority = JSONAuthoritySchema.object([
        "allowedPressureStates": .array(.scalar),
        "allowedThermalStates": .array(.scalar),
        "cooldownSeconds": .scalar,
        "digest": .scalar,
        "evidenceID": .scalar,
        "maxActiveMetalDeltaBytes": .scalar,
        "maxCachedMetalDeltaBytes": .scalar,
        "maxPageoutDelta": .scalar,
        "maxRSSDeltaBytes": .scalar,
        "maxSwapDeltaBytes": .scalar,
        "minIdleSamples": .scalar,
    ])
    static let claimAuthority = JSONAuthoritySchema.object([
        "absoluteAuthority": .absoluteAuthority,
        "claimKind": .scalar,
        "cleanupAuthority": .cleanupAuthority,
    ])
    static let policyAuthority = JSONAuthoritySchema.object([
        "artifact": .artifact,
        "claimAuthorities": .array(.claimAuthority),
        "digest": .scalar,
        "evidenceID": .scalar,
        "runIdentity": .runIdentity,
        "schemaVersion": .scalar,
        "sealedBeforeMeasurements": .scalar,
    ])
    static let productionRouteReceiptAuthority = JSONAuthoritySchema.object([
        "artifact": .artifact,
        "backendBuildIdentityDigest": .scalar,
        "digest": .scalar,
        "evidenceID": .scalar,
        "observationDigest": .scalar,
        "runIdentityDigest": .scalar,
        "schemaVersion": .scalar,
    ])
}

private enum JSONAuthorityPreflightFailure: Sendable {
    case duplicateKey
    case excessiveNesting
    case invalid
}

private struct JSONAuthorityPreflightScanner {
    var bytes: [UInt8]
    var maxDepth: Int
    var index = 0
    var failure: JSONAuthorityPreflightFailure?
    var shapeInvalid = false

    mutating func scan(
        schema: JSONAuthoritySchema
    ) -> JSONAuthorityPreflightFailure? {
        skipWhitespace()
        guard scanValue(schema: schema, depth: 0) else {
            return failure ?? .invalid
        }
        skipWhitespace()
        guard index == bytes.count else { return .invalid }
        if let failure { return failure }
        return shapeInvalid ? .invalid : nil
    }

    private mutating func scanValue(
        schema: JSONAuthoritySchema?,
        depth: Int
    ) -> Bool {
        skipWhitespace()
        guard index < bytes.count else { return false }
        switch bytes[index] {
        case UInt8(ascii: "{"):
            let nextDepth = depth + 1
            guard nextDepth <= maxDepth else {
                failure = .excessiveNesting
                return false
            }
            guard case .object(let fields) = schema else {
                shapeInvalid = true
                return scanObject(fields: nil, depth: nextDepth)
            }
            return scanObject(fields: fields, depth: nextDepth)
        case UInt8(ascii: "["):
            let nextDepth = depth + 1
            guard nextDepth <= maxDepth else {
                failure = .excessiveNesting
                return false
            }
            guard case .array(let elementSchema) = schema else {
                shapeInvalid = true
                return scanArray(elementSchema: nil, depth: nextDepth)
            }
            return scanArray(
                elementSchema: elementSchema,
                depth: nextDepth)
        case UInt8(ascii: "\""):
            if !acceptsScalar(schema) {
                shapeInvalid = true
            }
            return scanString() != nil
        case UInt8(ascii: "t"):
            if !acceptsScalar(schema) {
                shapeInvalid = true
            }
            return consume("true")
        case UInt8(ascii: "f"):
            if !acceptsScalar(schema) {
                shapeInvalid = true
            }
            return consume("false")
        case UInt8(ascii: "n"):
            if !acceptsScalar(schema) {
                shapeInvalid = true
            }
            return consume("null")
        default:
            if !acceptsScalar(schema) {
                shapeInvalid = true
            }
            return scanNumber()
        }
    }

    private mutating func scanObject(
        fields: [String: JSONAuthoritySchema]?,
        depth: Int
    ) -> Bool {
        guard consumeByte(UInt8(ascii: "{")) else { return false }
        skipWhitespace()
        var keys = Set<String>()
        if consumeByte(UInt8(ascii: "}")) { return true }
        while true {
            skipWhitespace()
            guard let key = scanString() else { return false }
            guard keys.insert(key).inserted else {
                failure = .duplicateKey
                return false
            }
            guard let fieldSchema = fields?[key], fields != nil else {
                if fields != nil {
                    shapeInvalid = true
                }
                skipWhitespace()
                guard consumeByte(UInt8(ascii: ":")) else { return false }
                guard scanValue(schema: nil, depth: depth) else {
                    return false
                }
                skipWhitespace()
                if consumeByte(UInt8(ascii: "}")) { return true }
                guard consumeByte(UInt8(ascii: ",")) else { return false }
                continue
            }
            skipWhitespace()
            guard consumeByte(UInt8(ascii: ":")) else { return false }
            guard scanValue(schema: fieldSchema, depth: depth) else {
                return false
            }
            skipWhitespace()
            if consumeByte(UInt8(ascii: "}")) { return true }
            guard consumeByte(UInt8(ascii: ",")) else { return false }
        }
    }

    private mutating func scanArray(
        elementSchema: JSONAuthoritySchema?,
        depth: Int
    ) -> Bool {
        guard consumeByte(UInt8(ascii: "[")) else { return false }
        skipWhitespace()
        if consumeByte(UInt8(ascii: "]")) { return true }
        while true {
            guard scanValue(schema: elementSchema, depth: depth) else {
                return false
            }
            skipWhitespace()
            if consumeByte(UInt8(ascii: "]")) { return true }
            guard consumeByte(UInt8(ascii: ",")) else { return false }
        }
    }

    private mutating func scanString() -> String? {
        guard consumeByte(UInt8(ascii: "\"")) else { return nil }
        var data = Data()
        while index < bytes.count {
            let byte = bytes[index]
            index += 1
            if byte == UInt8(ascii: "\"") {
                return String(data: data, encoding: .utf8)
            }
            if byte == UInt8(ascii: "\\") {
                guard index < bytes.count else { return nil }
                let escaped = bytes[index]
                index += 1
                switch escaped {
                case UInt8(ascii: "\""), UInt8(ascii: "\\"),
                    UInt8(ascii: "/"):
                    data.append(escaped)
                case UInt8(ascii: "b"):
                    data.append(8)
                case UInt8(ascii: "f"):
                    data.append(12)
                case UInt8(ascii: "n"):
                    data.append(10)
                case UInt8(ascii: "r"):
                    data.append(13)
                case UInt8(ascii: "t"):
                    data.append(9)
                case UInt8(ascii: "u"):
                    guard let scalar = scanUnicodeScalar() else { return nil }
                    data.append(contentsOf: String(scalar).utf8)
                default:
                    return nil
                }
            } else {
                guard byte >= 0x20 else { return nil }
                data.append(byte)
            }
        }
        return nil
    }

    private mutating func scanUnicodeScalar() -> Unicode.Scalar? {
        guard index + 4 <= bytes.count else { return nil }
        var value: UInt32 = 0
        for _ in 0 ..< 4 {
            guard let nibble = hexNibble(bytes[index]) else { return nil }
            value = value * 16 + UInt32(nibble)
            index += 1
        }
        return Unicode.Scalar(value)
    }

    private mutating func scanNumber() -> Bool {
        let start = index
        if consumeByte(UInt8(ascii: "-")) {}
        guard scanDigits() else { return false }
        if consumeByte(UInt8(ascii: ".")) {
            guard scanDigits() else { return false }
        }
        if index < bytes.count,
            bytes[index] == UInt8(ascii: "e")
                || bytes[index] == UInt8(ascii: "E")
        {
            index += 1
            if index < bytes.count,
                bytes[index] == UInt8(ascii: "+")
                    || bytes[index] == UInt8(ascii: "-")
            {
                index += 1
            }
            guard scanDigits() else { return false }
        }
        return index > start
    }

    private mutating func scanDigits() -> Bool {
        let start = index
        while index < bytes.count,
            bytes[index] >= UInt8(ascii: "0"),
            bytes[index] <= UInt8(ascii: "9")
        {
            index += 1
        }
        return index > start
    }

    private mutating func consume(_ literal: String) -> Bool {
        for byte in literal.utf8 {
            guard index < bytes.count, bytes[index] == byte else { return false }
            index += 1
        }
        return true
    }

    private mutating func consumeByte(_ byte: UInt8) -> Bool {
        guard index < bytes.count, bytes[index] == byte else { return false }
        index += 1
        return true
    }

    private mutating func skipWhitespace() {
        while index < bytes.count {
            switch bytes[index] {
            case UInt8(ascii: " "), UInt8(ascii: "\n"),
                UInt8(ascii: "\r"), UInt8(ascii: "\t"):
                index += 1
            default:
                return
            }
        }
    }

    private func hexNibble(_ byte: UInt8) -> UInt8? {
        switch byte {
        case UInt8(ascii: "0") ... UInt8(ascii: "9"):
            return byte - UInt8(ascii: "0")
        case UInt8(ascii: "a") ... UInt8(ascii: "f"):
            return byte - UInt8(ascii: "a") + 10
        case UInt8(ascii: "A") ... UInt8(ascii: "F"):
            return byte - UInt8(ascii: "A") + 10
        default:
            return nil
        }
    }

    private func acceptsScalar(_ schema: JSONAuthoritySchema?) -> Bool {
        guard let schema else { return true }
        guard case .scalar = schema else { return false }
        return true
    }
}

private extension Qwen38PerformanceAttributionAuthorityFileRole {
    var name: String {
        switch self {
        case .policy:
            return "policy"
        case .productionRouteReceipt:
            return "productionRouteReceipt"
        }
    }
}
