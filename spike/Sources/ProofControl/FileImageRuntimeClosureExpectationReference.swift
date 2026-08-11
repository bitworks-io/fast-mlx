enum FileImageRuntimeClosureExpectationFailure: Error, Equatable {
    case expectationReanchor(RuntimeClosureExpectationError)
    case expectationEvidenceMismatch
    case declaredFileImageMemberCount
}

fileprivate enum FileImageRuntimeClosureExpectationConstructionSeal:
    Equatable
{
    case verified
}

/// Exact current Stage-C1 expectation bytes that declare at least one file row.
/// This comparison-only value grants no runtime or source-import authority.
struct FileImageRuntimeClosureExpectationReference: Equatable {
    let anchoredExpectation: AnchoredRuntimeClosureExpectationDocument
    let declaredFileImageMemberCount: Int
    let runtimeDecision: RuntimeClosureExpectationRuntimeDecision = .noGo

    let canExecute = false
    let canSpawn = false
    let canAccessNetwork = false
    let canConsumePack = false
    let canMutateFileSystem = false
    let canImportGitObjects = false
    let canBuild = false
    let canLoadModel = false
    let canReserveOutput = false
    let canPublish = false

    fileprivate let constructionSeal:
        FileImageRuntimeClosureExpectationConstructionSeal

    fileprivate init(
        anchoredExpectation: AnchoredRuntimeClosureExpectationDocument,
        declaredFileImageMemberCount: Int
    ) {
        self.anchoredExpectation = anchoredExpectation
        self.declaredFileImageMemberCount = declaredFileImageMemberCount
        self.constructionSeal = .verified
    }
}

enum FileImageRuntimeClosureExpectationVerifier {
    static func reference(
        anchoredExpectation: AnchoredRuntimeClosureExpectationDocument,
        currentExpectationAnchor: RuntimeClosureExpectationTrustAnchor
    ) throws -> FileImageRuntimeClosureExpectationReference {
        let reanchored: AnchoredRuntimeClosureExpectationDocument
        do {
            reanchored = try RuntimeClosureExpectationVerifier.anchor(
                expectationFile: anchoredExpectation.expectationFile,
                trustAnchor: currentExpectationAnchor
            )
        } catch let failure as RuntimeClosureExpectationError {
            throw FileImageRuntimeClosureExpectationFailure
                .expectationReanchor(failure)
        }

        guard reanchored == anchoredExpectation else {
            throw FileImageRuntimeClosureExpectationFailure
                .expectationEvidenceMismatch
        }

        var declaredFileImageMemberCount = 0
        for member in reanchored.fields.members
        where member.storage == .file {
            let increment = declaredFileImageMemberCount
                .addingReportingOverflow(1)
            guard !increment.overflow else {
                throw FileImageRuntimeClosureExpectationFailure
                    .declaredFileImageMemberCount
            }
            declaredFileImageMemberCount = increment.partialValue
        }

        guard (1...256).contains(declaredFileImageMemberCount) else {
            throw FileImageRuntimeClosureExpectationFailure
                .declaredFileImageMemberCount
        }

        return FileImageRuntimeClosureExpectationReference(
            anchoredExpectation: reanchored,
            declaredFileImageMemberCount: declaredFileImageMemberCount
        )
    }
}
