public enum RunAuthorizedInputRole: String, Equatable, Sendable {
    case worker
    case baselineSource = "baseline-source"
    case candidateSource = "candidate-source"
}

public enum RunSourceRole: String, Equatable, Sendable {
    case baseline
    case candidate
}

public enum RunAuthorizedInputsError: Error, Equatable, Sendable {
    case unexpectedPurpose(
        role: RunAuthorizedInputRole,
        expected: OperatorAuthorizationPurpose,
        actual: OperatorAuthorizationPurpose
    )
    case duplicateAuthorizationID(
        first: RunAuthorizedInputRole,
        second: RunAuthorizedInputRole
    )
}

extension RunAuthorizedInputsError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .unexpectedPurpose(let role, let expected, let actual):
            "run input \(role.rawValue) purpose \(actual.rawValue) does not match \(expected.rawValue)"
        case .duplicateAuthorizationID(let first, let second):
            "run inputs \(first.rawValue) and \(second.rawValue) have the same authorization ID"
        }
    }
}

public struct RunAuthorizedSource: Equatable, Sendable {
    public let role: RunSourceRole
    public let sourceManifest: OperatorAuthorizedFile

    fileprivate init(
        role: RunSourceRole,
        sourceManifest: OperatorAuthorizedFile
    ) {
        self.role = role
        self.sourceManifest = sourceManifest
    }
}

public struct RunAuthorizedInputs: Equatable, Sendable {
    public let worker: OperatorAuthorizedFile
    public let baseline: RunAuthorizedSource
    public let candidate: RunAuthorizedSource

    public init(
        worker: OperatorAuthorizedFile,
        baselineSourceManifest: OperatorAuthorizedFile,
        candidateSourceManifest: OperatorAuthorizedFile
    ) throws {
        try Self.requirePurpose(
            worker,
            role: .worker,
            expected: .workerBytes
        )
        try Self.requirePurpose(
            baselineSourceManifest,
            role: .baselineSource,
            expected: .sourceManifest
        )
        try Self.requirePurpose(
            candidateSourceManifest,
            role: .candidateSource,
            expected: .sourceManifest
        )
        try Self.requireDistinctAuthorizationIDs(
            worker,
            role: .worker,
            baselineSourceManifest,
            role: .baselineSource
        )
        try Self.requireDistinctAuthorizationIDs(
            worker,
            role: .worker,
            candidateSourceManifest,
            role: .candidateSource
        )
        try Self.requireDistinctAuthorizationIDs(
            baselineSourceManifest,
            role: .baselineSource,
            candidateSourceManifest,
            role: .candidateSource
        )

        self.worker = worker
        self.baseline = RunAuthorizedSource(
            role: .baseline,
            sourceManifest: baselineSourceManifest
        )
        self.candidate = RunAuthorizedSource(
            role: .candidate,
            sourceManifest: candidateSourceManifest
        )
    }
}

private extension RunAuthorizedInputs {
    static func requirePurpose(
        _ file: OperatorAuthorizedFile,
        role: RunAuthorizedInputRole,
        expected: OperatorAuthorizationPurpose
    ) throws {
        guard file.purpose == expected else {
            throw RunAuthorizedInputsError.unexpectedPurpose(
                role: role,
                expected: expected,
                actual: file.purpose
            )
        }
    }

    static func requireDistinctAuthorizationIDs(
        _ first: OperatorAuthorizedFile,
        role firstRole: RunAuthorizedInputRole,
        _ second: OperatorAuthorizedFile,
        role secondRole: RunAuthorizedInputRole
    ) throws {
        guard first.authorizationID != second.authorizationID else {
            throw RunAuthorizedInputsError.duplicateAuthorizationID(
                first: firstRole,
                second: secondRole
            )
        }
    }
}
