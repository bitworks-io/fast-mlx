import CryptoKit
import Darwin
import Foundation
import HarnessCore

enum ExactPrefixProofCLIError: Error, Equatable {
    case positionalArgument(String)
    case unknownFlag(String)
    case duplicateFlag(String)
    case missingValue(String)
    case missingRequiredFlag(String)
    case invalidValue(String)
    case harnessIdentityMismatch
    case executableIdentityMismatch
    case outputNotFresh(String)
    case unsafeOutputFilename(String)
    case immutableArtifactExists(String)
    case lockNotHeld
    case invalidRuntimeMetadata
    case invalidArtifact
    case invalidOutputDirectory
    case missingOutputArtifact(String)
    case evidenceHashMismatch
    case completionMismatch
    case terminalStatusMismatch
    case ioFailure(operation: String, code: Int32)
}

struct ExactPrefixProofCommand {
    let modelPath: String
    let plan: ExactPrefixProofCommandPlan
}

struct ExactPrefixProofRawCommand {
    let modelPath: String
    let modelID: String
    let sourceRevision: String
    let expectedHarnessSHA: String
    let expectedExecutableSHA256: String
    let checkpointContentSHA256: String
    let tokenizerSHA256: String
    let workloadNonce: String
    let outputPath: String
    let maxTokens: Int
    let promptRepeat: Int
    let exactPrefixCachePolicy: ExactPrefixCachePolicy
    let templateTokenCachePolicy: TemplateTokenCachePolicy
    let memoryLimitBytes: Int
    let cacheLimitBytes: Int

    func authenticated(
        actualHarnessSHA: String,
        actualExecutableSHA256: String,
        admission: CompressedKVAttentionRuntimeAdmission
    ) throws -> ExactPrefixProofCommand {
        guard expectedHarnessSHA == actualHarnessSHA else {
            throw ExactPrefixProofCLIError.harnessIdentityMismatch
        }
        guard expectedExecutableSHA256 == actualExecutableSHA256 else {
            throw ExactPrefixProofCLIError.executableIdentityMismatch
        }
        let plan = try ExactPrefixProofCommandPlan(
            modelID: modelID,
            sourceRevision: sourceRevision,
            expectedHarnessSHA: expectedHarnessSHA,
            expectedExecutableSHA256: expectedExecutableSHA256,
            admission: admission,
            checkpointContentSHA256: checkpointContentSHA256,
            tokenizerSHA256: tokenizerSHA256,
            workloadNonce: workloadNonce,
            maxTokens: maxTokens,
            promptRepeat: promptRepeat,
            exactPrefixCachePolicy: exactPrefixCachePolicy,
            templateTokenCachePolicy: templateTokenCachePolicy,
            memoryLimitBytes: memoryLimitBytes,
            cacheLimitBytes: cacheLimitBytes,
            outputPath: outputPath)
        return ExactPrefixProofCommand(
            modelPath: modelPath,
            plan: plan)
    }
}

func parseExactPrefixProofRawCommand(
    arguments: [String]
) throws -> ExactPrefixProofRawCommand {
    let allowed: Set<String> = [
        "--model",
        "--model-id",
        "--source-revision",
        "--expected-harness-git-sha",
        "--expected-binary-sha256",
        "--checkpoint-content-sha256",
        "--tokenizer-sha256",
        "--workload-nonce",
        "--output",
        "--max-tokens",
        "--prompt-repeat",
        "--prefix-max-entries",
        "--prefix-max-retained-bytes",
        "--prefix-minimum-reusable-tokens",
        "--template-max-entries",
        "--template-max-retained-bytes",
        "--memory-limit-bytes",
        "--cache-limit-bytes",
    ]
    var values: [String: String] = [:]
    var index = 0
    while index < arguments.count {
        let flag = arguments[index]
        guard flag.hasPrefix("--") else {
            throw ExactPrefixProofCLIError.positionalArgument(flag)
        }
        guard allowed.contains(flag) else {
            throw ExactPrefixProofCLIError.unknownFlag(flag)
        }
        guard values[flag] == nil else {
            throw ExactPrefixProofCLIError.duplicateFlag(flag)
        }
        guard index + 1 < arguments.count else {
            throw ExactPrefixProofCLIError.missingValue(flag)
        }
        let value = arguments[index + 1]
        guard !value.hasPrefix("--") else {
            throw ExactPrefixProofCLIError.missingValue(flag)
        }
        values[flag] = value
        index += 2
    }

    func required(_ flag: String) throws -> String {
        guard let value = values[flag], !value.isEmpty else {
            throw ExactPrefixProofCLIError.missingRequiredFlag(flag)
        }
        return value
    }
    func positiveInteger(_ flag: String) throws -> Int {
        let raw = try required(flag)
        guard let value = Int(raw), value > 0 else {
            throw ExactPrefixProofCLIError.invalidValue(flag)
        }
        return value
    }

    let exactPolicy: ExactPrefixCachePolicy
    let templatePolicy: TemplateTokenCachePolicy
    do {
        exactPolicy = try ExactPrefixCachePolicy(
            maxEntries: positiveInteger("--prefix-max-entries"),
            maxRetainedBytes: positiveInteger(
                "--prefix-max-retained-bytes"),
            minimumReusableTokens: positiveInteger(
                "--prefix-minimum-reusable-tokens"))
        templatePolicy = try TemplateTokenCachePolicy(
            maxEntries: positiveInteger("--template-max-entries"),
            maxRetainedBytes: positiveInteger(
                "--template-max-retained-bytes"))
    } catch let error as ExactPrefixProofCLIError {
        throw error
    } catch {
        throw ExactPrefixProofCLIError.invalidValue(
            "--prefix/template-policy")
    }
    guard (1 ... 64).contains(exactPolicy.maxEntries) else {
        throw ExactPrefixProofCLIError.invalidValue(
            "--prefix-max-entries")
    }
    guard (1 ... 64).contains(templatePolicy.maxEntries) else {
        throw ExactPrefixProofCLIError.invalidValue(
            "--template-max-entries")
    }
    let modelPath = try required("--model")
    guard modelPath == modelPath.trimmingCharacters(
        in: .whitespacesAndNewlines),
        (modelPath as NSString).isAbsolutePath
    else {
        throw ExactPrefixProofCLIError.invalidValue("--model")
    }
    let outputPath = try required("--output")
    guard outputPath == outputPath.trimmingCharacters(
        in: .whitespacesAndNewlines),
        (outputPath as NSString).isAbsolutePath
    else {
        throw ExactPrefixProofCLIError.invalidValue("--output")
    }
    let modelID = try required("--model-id")
    guard exactPrefixProofPathFreeIdentifier(modelID) else {
        throw ExactPrefixProofCLIError.invalidValue("--model-id")
    }
    let sourceRevision = try required("--source-revision")
    guard exactPrefixProofIsLowercaseHex(
        sourceRevision,
        lengths: [64])
    else {
        throw ExactPrefixProofCLIError.invalidValue(
            "--source-revision")
    }
    let expectedHarnessSHA = try required(
        "--expected-harness-git-sha")
    guard exactPrefixProofIsLowercaseHex(
        expectedHarnessSHA,
        lengths: [40, 64])
    else {
        throw ExactPrefixProofCLIError.invalidValue(
            "--expected-harness-git-sha")
    }
    let expectedExecutableSHA256 = try required(
        "--expected-binary-sha256")
    let checkpointContentSHA256 = try required(
        "--checkpoint-content-sha256")
    let tokenizerSHA256 = try required("--tokenizer-sha256")
    for (flag, value) in [
        ("--expected-binary-sha256", expectedExecutableSHA256),
        ("--checkpoint-content-sha256", checkpointContentSHA256),
        ("--tokenizer-sha256", tokenizerSHA256),
    ] where !exactPrefixProofIsLowercaseHex(value, lengths: [64]) {
        throw ExactPrefixProofCLIError.invalidValue(flag)
    }
    let workloadNonce = try required("--workload-nonce")
    do {
        _ = try ServiceWorkloadIdentity(nonce: workloadNonce)
    } catch {
        throw ExactPrefixProofCLIError.invalidValue(
            "--workload-nonce")
    }
    let maxTokens = try positiveInteger("--max-tokens")
    guard (2 ... 128).contains(maxTokens) else {
        throw ExactPrefixProofCLIError.invalidValue("--max-tokens")
    }
    let promptRepeat = try positiveInteger("--prompt-repeat")
    guard (1 ... 256).contains(promptRepeat) else {
        throw ExactPrefixProofCLIError.invalidValue(
            "--prompt-repeat")
    }
    let memoryLimitBytes = try positiveInteger(
        "--memory-limit-bytes")
    let cacheLimitBytes = try positiveInteger(
        "--cache-limit-bytes")
    guard cacheLimitBytes <= memoryLimitBytes else {
        throw ExactPrefixProofCLIError.invalidValue(
            "--memory/cache-limits")
    }
    let (configuredRetainedBytes, retainedBytesOverflow) =
        exactPolicy.maxRetainedBytes.addingReportingOverflow(
            templatePolicy.maxRetainedBytes)
    guard !retainedBytesOverflow,
        configuredRetainedBytes <= memoryLimitBytes
    else {
        throw ExactPrefixProofCLIError.invalidValue(
            "--prefix/template-memory-budget")
    }
    return ExactPrefixProofRawCommand(
        modelPath: modelPath,
        modelID: modelID,
        sourceRevision: sourceRevision,
        expectedHarnessSHA: expectedHarnessSHA,
        expectedExecutableSHA256: expectedExecutableSHA256,
        checkpointContentSHA256: checkpointContentSHA256,
        tokenizerSHA256: tokenizerSHA256,
        workloadNonce: workloadNonce,
        outputPath: outputPath,
        maxTokens: maxTokens,
        promptRepeat: promptRepeat,
        exactPrefixCachePolicy: exactPolicy,
        templateTokenCachePolicy: templatePolicy,
        memoryLimitBytes: memoryLimitBytes,
        cacheLimitBytes: cacheLimitBytes)
}

func parseExactPrefixProofCommand(
    arguments: [String],
    actualHarnessSHA: String,
    actualExecutableSHA256: String,
    admission: CompressedKVAttentionRuntimeAdmission
) throws -> ExactPrefixProofCommand {
    try parseExactPrefixProofRawCommand(
        arguments: arguments
    ).authenticated(
        actualHarnessSHA: actualHarnessSHA,
        actualExecutableSHA256: actualExecutableSHA256,
        admission: admission)
}

private func exactPrefixProofIsLowercaseHex(
    _ value: String,
    lengths: Set<Int>
) -> Bool {
    lengths.contains(value.utf8.count)
        && value.utf8.allSatisfy {
            (48 ... 57).contains($0) || (97 ... 102).contains($0)
        }
}

private func exactPrefixProofPathFreeIdentifier(
    _ value: String
) -> Bool {
    let bytes = value.utf8
    return !bytes.isEmpty && bytes.count <= 128
        && bytes.allSatisfy {
            (33 ... 126).contains($0) && $0 != 47 && $0 != 92
        }
}

struct ExactPrefixExecutableIdentity: Equatable {
    let path: String
    let device: UInt64
    let inode: UInt64
    let size: Int64
    let modifiedSeconds: Int64
    let modifiedNanoseconds: Int64
    let sha256: String
}

func authenticateCurrentExecutable(
    expectedSHA256: String
) throws -> ExactPrefixExecutableIdentity {
    guard let executableURL = Bundle.main.executableURL else {
        throw ExactPrefixProofCLIError.executableIdentityMismatch
    }
    return try authenticateExecutable(
        at: executableURL,
        expectedSHA256: expectedSHA256)
}

func authenticateExecutable(
    at url: URL,
    expectedSHA256: String
) throws -> ExactPrefixExecutableIdentity {
    let url = url.standardizedFileURL
    var pathMetadata = stat()
    guard lstat(url.path, &pathMetadata) == 0 else {
        throw ExactPrefixProofCLIError.ioFailure(
            operation: "lstat-executable",
            code: errno)
    }
    guard pathMetadata.st_mode & S_IFMT == S_IFREG else {
        throw ExactPrefixProofCLIError.executableIdentityMismatch
    }
    let descriptor = open(
        url.path,
        O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
    guard descriptor >= 0 else {
        throw ExactPrefixProofCLIError.ioFailure(
            operation: "open-executable",
            code: errno)
    }
    defer { _ = close(descriptor) }

    var before = stat()
    guard fstat(descriptor, &before) == 0,
        before.st_mode & S_IFMT == S_IFREG
    else {
        throw ExactPrefixProofCLIError.ioFailure(
            operation: "fstat-executable-before",
            code: errno)
    }
    var hasher = SHA256()
    var buffer = [UInt8](repeating: 0, count: 1 << 20)
    while true {
        let count = read(descriptor, &buffer, buffer.count)
        if count == 0 {
            break
        }
        guard count > 0 else {
            throw ExactPrefixProofCLIError.ioFailure(
                operation: "read-executable",
                code: errno)
        }
        hasher.update(data: Data(buffer[..<Int(count)]))
    }
    var after = stat()
    guard fstat(descriptor, &after) == 0 else {
        throw ExactPrefixProofCLIError.ioFailure(
            operation: "fstat-executable-after",
            code: errno)
    }
    guard executableMetadataMatches(before, after) else {
        throw ExactPrefixProofCLIError.executableIdentityMismatch
    }
    let sha256 = hasher.finalize().map {
        String(format: "%02x", $0)
    }.joined()
    guard sha256 == expectedSHA256 else {
        throw ExactPrefixProofCLIError.executableIdentityMismatch
    }
    return ExactPrefixExecutableIdentity(
        path: url.path,
        device: UInt64(before.st_dev),
        inode: UInt64(before.st_ino),
        size: Int64(before.st_size),
        modifiedSeconds: Int64(before.st_mtimespec.tv_sec),
        modifiedNanoseconds: Int64(before.st_mtimespec.tv_nsec),
        sha256: sha256)
}

func validateExecutableUnchanged(
    _ identity: ExactPrefixExecutableIdentity
) throws {
    let current = try authenticateExecutable(
        at: URL(fileURLWithPath: identity.path),
        expectedSHA256: identity.sha256)
    guard current == identity else {
        throw ExactPrefixProofCLIError.executableIdentityMismatch
    }
}

private func executableMetadataMatches(
    _ lhs: stat,
    _ rhs: stat
) -> Bool {
    lhs.st_dev == rhs.st_dev
        && lhs.st_ino == rhs.st_ino
        && lhs.st_mode == rhs.st_mode
        && lhs.st_size == rhs.st_size
        && lhs.st_mtimespec.tv_sec == rhs.st_mtimespec.tv_sec
        && lhs.st_mtimespec.tv_nsec == rhs.st_mtimespec.tv_nsec
}

enum ExactPrefixProofOutputState:
    String, Codable, Equatable, Sendable
{
    case starting
    case running
    case complete
    case failed
}

struct ExactPrefixProofOutputStatus:
    Codable, Equatable, Sendable
{
    let schemaVersion: Int
    let state: ExactPrefixProofOutputState
    let processID: Int32
    let completedCases: Int
    let totalCases: Int
    let elapsedSeconds: Double
    let harnessGitSHA: String
    let executableSHA256: String
    let modelID: String
    let sourceRevision: String
    let workloadNonce: String
    let error: String?

    init(
        state: ExactPrefixProofOutputState,
        processID: Int32,
        completedCases: Int,
        totalCases: Int,
        elapsedSeconds: Double,
        harnessGitSHA: String,
        executableSHA256: String,
        modelID: String,
        sourceRevision: String,
        workloadNonce: String,
        error: String?
    ) {
        schemaVersion = 1
        self.state = state
        self.processID = processID
        self.completedCases = completedCases
        self.totalCases = totalCases
        self.elapsedSeconds = elapsedSeconds
        self.harnessGitSHA = harnessGitSHA
        self.executableSHA256 = executableSHA256
        self.modelID = modelID
        self.sourceRevision = sourceRevision
        self.workloadNonce = workloadNonce
        self.error = error
    }
}

struct ExactPrefixProofRuntimeMetadata:
    Codable, Equatable, Sendable
{
    let schemaVersion: Int
    let capturedAtUTC: String
    let hardwareChip: String
    let hardwareRAMBytes: UInt64
    let hardwareOS: String
    let mlxSwiftVersion: String
    let mlxSwiftLMRevision: String
    let hostEnvironment: BenchQualificationWarmupEnvironment

    init(
        capturedAtUTC: String,
        hardwareChip: String,
        hardwareRAMBytes: UInt64,
        hardwareOS: String,
        mlxSwiftVersion: String,
        mlxSwiftLMRevision: String,
        hostEnvironment: BenchQualificationWarmupEnvironment
    ) throws {
        schemaVersion = 1
        self.capturedAtUTC = capturedAtUTC
        self.hardwareChip = hardwareChip
        self.hardwareRAMBytes = hardwareRAMBytes
        self.hardwareOS = hardwareOS
        self.mlxSwiftVersion = mlxSwiftVersion
        self.mlxSwiftLMRevision = mlxSwiftLMRevision
        self.hostEnvironment = hostEnvironment
        try validate()
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion
        case capturedAtUTC
        case hardwareChip
        case hardwareRAMBytes
        case hardwareOS
        case mlxSwiftVersion
        case mlxSwiftLMRevision
        case hostEnvironment
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try values.decode(
            Int.self, forKey: .schemaVersion)
        capturedAtUTC = try values.decode(
            String.self, forKey: .capturedAtUTC)
        hardwareChip = try values.decode(
            String.self, forKey: .hardwareChip)
        hardwareRAMBytes = try values.decode(
            UInt64.self, forKey: .hardwareRAMBytes)
        hardwareOS = try values.decode(
            String.self, forKey: .hardwareOS)
        mlxSwiftVersion = try values.decode(
            String.self, forKey: .mlxSwiftVersion)
        mlxSwiftLMRevision = try values.decode(
            String.self, forKey: .mlxSwiftLMRevision)
        hostEnvironment = try values.decode(
            BenchQualificationWarmupEnvironment.self,
            forKey: .hostEnvironment)
        try validate()
    }

    func validate() throws {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [
            .withInternetDateTime,
            .withDashSeparatorInDate,
            .withColonSeparatorInTime,
        ]
        guard schemaVersion == 1,
            formatter.date(from: capturedAtUTC) != nil,
            Self.printableNonempty(hardwareChip),
            hardwareRAMBytes > 0,
            Self.printableNonempty(hardwareOS),
            Self.printableNonempty(mlxSwiftVersion),
            mlxSwiftLMRevision.utf8.count == 40,
            mlxSwiftLMRevision.utf8.allSatisfy({
                (48 ... 57).contains($0)
                    || (97 ... 102).contains($0)
            }),
            Self.validHostEnvironment(hostEnvironment)
        else {
            throw ExactPrefixProofCLIError
                .invalidRuntimeMetadata
        }
    }

    private static func printableNonempty(
        _ value: String
    ) -> Bool {
        !value.isEmpty && value.utf8.count <= 256
            && value.utf8.allSatisfy { (32 ... 126).contains($0) }
    }

    private static func validHostEnvironment(
        _ environment: BenchQualificationWarmupEnvironment
    ) -> Bool {
        let before = environment.before
        let after = environment.after
        func safe(
            _ snapshot: BenchQualificationHostSnapshot
        ) -> Bool {
            snapshot.monotonicTimestampSeconds.isFinite
                && snapshot.residentSizeBytes > 0
                && snapshot.physicalFootprintBytes > 0
                && snapshot.powerSource == .acPower
                && !snapshot.lowPowerModeEnabled
                && (
                    snapshot.thermalState == .nominal
                        || snapshot.thermalState == .fair
                )
        }
        return safe(before) && safe(after)
            && after.monotonicTimestampSeconds
                > before.monotonicTimestampSeconds
    }
}

struct ExactPrefixProofArtifact:
    Codable, Equatable, Sendable
{
    let schemaVersion: Int
    let runtime: ExactPrefixProofRuntimeMetadata
    let evidence: ExactPrefixProofEvidence

    init(
        runtime: ExactPrefixProofRuntimeMetadata,
        evidence: ExactPrefixProofEvidence
    ) throws {
        schemaVersion = 1
        self.runtime = runtime
        self.evidence = evidence
        try validate()
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion
        case runtime
        case evidence
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try values.decode(
            Int.self, forKey: .schemaVersion)
        runtime = try values.decode(
            ExactPrefixProofRuntimeMetadata.self,
            forKey: .runtime)
        evidence = try values.decode(
            ExactPrefixProofEvidence.self,
            forKey: .evidence)
        try validate()
    }

    func validate() throws {
        guard schemaVersion == 1 else {
            throw ExactPrefixProofCLIError.invalidArtifact
        }
        try runtime.validate()
        _ = try evidence.validated()
    }
}

struct ExactPrefixProofCompletion:
    Codable, Equatable, Sendable
{
    let schemaVersion: Int
    let state: ExactPrefixProofOutputState
    let evidenceSHA256: String
    let harnessGitSHA: String
    let executableSHA256: String
    let modelID: String
    let sourceRevision: String
    let workloadNonce: String
    let completedCases: Int

    init(
        evidenceSHA256: String,
        evidence: ExactPrefixProofEvidence
    ) {
        schemaVersion = 1
        state = .complete
        self.evidenceSHA256 = evidenceSHA256
        harnessGitSHA = evidence.expectedHarnessSHA
        executableSHA256 =
            evidence.expectedExecutableSHA256
        modelID = evidence.modelID
        sourceRevision = evidence.sourceRevision
        workloadNonce = evidence.workloadNonce
        completedCases = evidence.cases.count
    }
}

@discardableResult
func publishExactPrefixProofArtifact(
    _ artifact: ExactPrefixProofArtifact,
    boundary: ExactPrefixProofOutputBoundary,
    authenticatedExecutableIdentity:
        ExactPrefixExecutableIdentity
) throws -> ExactPrefixProofCompletion {
    try artifact.validate()
    guard authenticatedExecutableIdentity.sha256
        == artifact.evidence.expectedExecutableSHA256
    else {
        throw ExactPrefixProofCLIError.executableIdentityMismatch
    }
    try validateExecutableUnchanged(
        authenticatedExecutableIdentity)
    let evidenceData = try canonicalJSONData(artifact)
    let evidenceSHA256 = sha256Hex(evidenceData)
    let completion = ExactPrefixProofCompletion(
        evidenceSHA256: evidenceSHA256,
        evidence: artifact.evidence)
    try boundary.writeImmutable(
        evidenceData,
        filename: "evidence.json")
    try validateExecutableUnchanged(
        authenticatedExecutableIdentity)
    try boundary.writeImmutable(
        canonicalJSONData(completion),
        filename: "completion.json")
    try validateExecutableUnchanged(
        authenticatedExecutableIdentity)
    return completion
}

func validateExactPrefixProofOutputDirectory(
    at directory: URL
) throws -> ExactPrefixProofArtifact {
    let directory = directory.standardizedFileURL
    let directoryDescriptor = open(
        directory.path,
        O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
    guard directoryDescriptor >= 0 else {
        throw ExactPrefixProofCLIError.invalidOutputDirectory
    }
    defer { _ = close(directoryDescriptor) }
    let statusData = try readExactPrefixProofArtifact(
        directoryDescriptor: directoryDescriptor,
        filename: "status.json")
    let evidenceData = try readExactPrefixProofArtifact(
        directoryDescriptor: directoryDescriptor,
        filename: "evidence.json")
    let completionData = try readExactPrefixProofArtifact(
        directoryDescriptor: directoryDescriptor,
        filename: "completion.json")
    let status = try JSONDecoder().decode(
        ExactPrefixProofOutputStatus.self,
        from: statusData)
    let completion = try JSONDecoder().decode(
        ExactPrefixProofCompletion.self,
        from: completionData)
    guard sha256Hex(evidenceData)
        == completion.evidenceSHA256
    else {
        throw ExactPrefixProofCLIError.evidenceHashMismatch
    }
    let artifact = try JSONDecoder().decode(
        ExactPrefixProofArtifact.self,
        from: evidenceData)
    try artifact.validate()
    let evidence = artifact.evidence
    guard completion.schemaVersion == 1,
        completion.state == .complete,
        completion.harnessGitSHA == evidence.expectedHarnessSHA,
        completion.executableSHA256
            == evidence.expectedExecutableSHA256,
        completion.modelID == evidence.modelID,
        completion.sourceRevision == evidence.sourceRevision,
        completion.workloadNonce == evidence.workloadNonce,
        completion.completedCases == evidence.cases.count
    else {
        throw ExactPrefixProofCLIError.completionMismatch
    }
    guard status.schemaVersion == 1,
        status.state == .complete,
        status.error == nil,
        status.completedCases == evidence.cases.count,
        status.totalCases
            == ExactPrefixProofCaseID.requiredOrder.count,
        status.harnessGitSHA == evidence.expectedHarnessSHA,
        status.executableSHA256
            == evidence.expectedExecutableSHA256,
        status.modelID == evidence.modelID,
        status.sourceRevision == evidence.sourceRevision,
        status.workloadNonce == evidence.workloadNonce
    else {
        throw ExactPrefixProofCLIError.terminalStatusMismatch
    }
    return artifact
}

private func readExactPrefixProofArtifact(
    directoryDescriptor: Int32,
    filename: String
) throws -> Data {
    let descriptor = openat(
        directoryDescriptor,
        filename,
        O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
    guard descriptor >= 0 else {
        throw ExactPrefixProofCLIError
            .missingOutputArtifact(filename)
    }
    defer { _ = close(descriptor) }
    var metadata = stat()
    guard fstat(descriptor, &metadata) == 0,
        metadata.st_mode & S_IFMT == S_IFREG,
        metadata.st_size >= 0
    else {
        throw ExactPrefixProofCLIError
            .missingOutputArtifact(filename)
    }
    var result = Data()
    result.reserveCapacity(Int(metadata.st_size))
    var buffer = [UInt8](repeating: 0, count: 64 << 10)
    while true {
        let count = read(descriptor, &buffer, buffer.count)
        if count == 0 { break }
        guard count > 0 else {
            throw ExactPrefixProofCLIError.ioFailure(
                operation: "read-output-artifact",
                code: errno)
        }
        result.append(contentsOf: buffer[..<Int(count)])
    }
    return result
}

final class ExactPrefixProofOutputBoundary {
    let directoryURL: URL
    private let directoryDescriptor: Int32
    private var lockDescriptor: Int32

    private init(
        directoryURL: URL,
        directoryDescriptor: Int32,
        lockDescriptor: Int32
    ) {
        self.directoryURL = directoryURL
        self.directoryDescriptor = directoryDescriptor
        self.lockDescriptor = lockDescriptor
    }

    static func claim(
        directoryPath: String
    ) throws -> ExactPrefixProofOutputBoundary {
        let directoryURL = URL(
            fileURLWithPath: directoryPath,
            isDirectory: true).standardizedFileURL
        let parent = directoryURL.deletingLastPathComponent()
        let leaf = directoryURL.lastPathComponent
        guard !leaf.isEmpty, leaf != ".", leaf != ".." else {
            throw ExactPrefixProofCLIError.outputNotFresh(
                directoryURL.path)
        }
        let parentDescriptor = open(
            parent.path,
            O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
        guard parentDescriptor >= 0 else {
            throw ExactPrefixProofCLIError.outputNotFresh(
                directoryURL.path)
        }
        defer { _ = close(parentDescriptor) }
        var existing = stat()
        errno = 0
        guard fstatat(
            parentDescriptor,
            leaf,
            &existing,
            AT_SYMLINK_NOFOLLOW) != 0,
            errno == ENOENT,
            mkdirat(parentDescriptor, leaf, S_IRWXU) == 0
        else {
            throw ExactPrefixProofCLIError.outputNotFresh(
                directoryURL.path)
        }
        let directoryDescriptor = openat(
            parentDescriptor,
            leaf,
            O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
        guard directoryDescriptor >= 0 else {
            throw ExactPrefixProofCLIError.ioFailure(
                operation: "open-output-directory",
                code: errno)
        }
        let lockDescriptor = openat(
            directoryDescriptor,
            ".lock",
            O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
            S_IRUSR | S_IWUSR)
        guard lockDescriptor >= 0 else {
            _ = close(directoryDescriptor)
            throw ExactPrefixProofCLIError.ioFailure(
                operation: "create-output-lock",
                code: errno)
        }
        do {
            let pid = Data(
                "\(ProcessInfo.processInfo.processIdentifier)\n".utf8)
            try writeAll(pid, to: lockDescriptor)
            guard fsync(lockDescriptor) == 0,
                flock(lockDescriptor, LOCK_EX | LOCK_NB) == 0,
                fsync(directoryDescriptor) == 0
            else {
                throw ExactPrefixProofCLIError.ioFailure(
                    operation: "seal-output-lock",
                    code: errno)
            }
            return ExactPrefixProofOutputBoundary(
                directoryURL: directoryURL,
                directoryDescriptor: directoryDescriptor,
                lockDescriptor: lockDescriptor)
        } catch {
            _ = close(lockDescriptor)
            _ = close(directoryDescriptor)
            throw error
        }
    }

    func writeStatus(
        _ status: ExactPrefixProofOutputStatus
    ) throws {
        try requireHeldLock()
        let data = try canonicalJSONData(status)
        try writeAtomicReplace(
            data,
            filename: "status.json")
    }

    func writeImmutable(
        _ data: Data,
        filename: String
    ) throws {
        try requireHeldLock()
        let allowed = Set(["evidence.json", "completion.json"])
        guard allowed.contains(filename) else {
            throw ExactPrefixProofCLIError.unsafeOutputFilename(
                filename)
        }
        let temporary = ".\(filename).\(UUID().uuidString).tmp"
        let descriptor = openat(
            directoryDescriptor,
            temporary,
            O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
            S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else {
            throw ExactPrefixProofCLIError.ioFailure(
                operation: "create-immutable-temporary",
                code: errno)
        }
        do {
            try Self.writeAll(data, to: descriptor)
            guard fsync(descriptor) == 0 else {
                throw ExactPrefixProofCLIError.ioFailure(
                    operation: "fsync-immutable-temporary",
                    code: errno)
            }
            _ = close(descriptor)
            guard linkat(
                directoryDescriptor,
                temporary,
                directoryDescriptor,
                filename,
                0) == 0
            else {
                throw ExactPrefixProofCLIError
                    .immutableArtifactExists(filename)
            }
            _ = unlinkat(directoryDescriptor, temporary, 0)
            guard fsync(directoryDescriptor) == 0 else {
                throw ExactPrefixProofCLIError.ioFailure(
                    operation: "fsync-output-directory",
                    code: errno)
            }
        } catch {
            _ = close(descriptor)
            _ = unlinkat(directoryDescriptor, temporary, 0)
            throw error
        }
    }

    private func writeAtomicReplace(
        _ data: Data,
        filename: String
    ) throws {
        guard filename == "status.json" else {
            throw ExactPrefixProofCLIError.unsafeOutputFilename(
                filename)
        }
        let temporary = ".\(filename).\(UUID().uuidString).tmp"
        let descriptor = openat(
            directoryDescriptor,
            temporary,
            O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
            S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else {
            throw ExactPrefixProofCLIError.ioFailure(
                operation: "create-status-temporary",
                code: errno)
        }
        do {
            try Self.writeAll(data, to: descriptor)
            guard fsync(descriptor) == 0 else {
                throw ExactPrefixProofCLIError.ioFailure(
                    operation: "fsync-status-temporary",
                    code: errno)
            }
            _ = close(descriptor)
            guard renameat(
                directoryDescriptor,
                temporary,
                directoryDescriptor,
                filename) == 0,
                fsync(directoryDescriptor) == 0
            else {
                throw ExactPrefixProofCLIError.ioFailure(
                    operation: "publish-status",
                    code: errno)
            }
        } catch {
            _ = close(descriptor)
            _ = unlinkat(directoryDescriptor, temporary, 0)
            throw error
        }
    }

    private func requireHeldLock() throws {
        guard lockDescriptor >= 0 else {
            throw ExactPrefixProofCLIError.lockNotHeld
        }
    }

    private static func writeAll(
        _ data: Data,
        to descriptor: Int32
    ) throws {
        try data.withUnsafeBytes { bytes in
            guard let baseAddress = bytes.baseAddress else { return }
            var offset = 0
            while offset < bytes.count {
                let written = Darwin.write(
                    descriptor,
                    baseAddress.advanced(by: offset),
                    bytes.count - offset)
                if written < 0, errno == EINTR { continue }
                guard written > 0 else {
                    throw ExactPrefixProofCLIError.ioFailure(
                        operation: "write-artifact",
                        code: errno)
                }
                offset += written
            }
        }
    }

    deinit {
        if lockDescriptor >= 0 {
            _ = flock(lockDescriptor, LOCK_UN)
            _ = close(lockDescriptor)
            lockDescriptor = -1
        }
        _ = close(directoryDescriptor)
    }
}

private func canonicalJSONData<T: Encodable>(
    _ value: T
) throws -> Data {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    var data = try encoder.encode(value)
    data.append(0x0A)
    return data
}
