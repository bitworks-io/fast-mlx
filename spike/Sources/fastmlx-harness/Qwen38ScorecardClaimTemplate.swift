import Foundation
import HarnessCore
import ProofControl

/// Operator claim-authoring helper (recorded next slice of the Slice 4b
/// ready state): prints the EXACT canonical Qwen38 scorecard claim body
/// with every pinned digest/identity filled from the in-binary constants
/// and unmistakable placeholder markers for the three operator-supplied
/// fields, so claim authoring never involves hand-copied digests. The
/// template is deliberately NOT signable as printed — the markers break
/// canonical parsing until replaced. Lives in fastmlx-harness (not the
/// proof-runner) because the runner's P5 structural gate forbids
/// gate-constant identity references there by design.
enum Qwen38ScorecardClaimTemplateError: Error, CustomStringConvertible {
    case sentinelSubstitutionFailed(String)

    var description: String {
        switch self {
        case .sentinelSubstitutionFailed(let marker):
            return "claim template sentinel substitution failed for \(marker)"
        }
    }
}

let qwen38ScorecardClaimTemplateMarkers = (
    harnessGitSHA1: "<HARNESS_GIT_SHA1_40HEX>",
    hostAdmissionID: "<HOST_ADMISSION_ID_64HEX>",
    resultPairID: "<RESULT_PAIR_ID_64HEX>"
)

func qwen38ScorecardClaimTemplate() throws -> String {
    typealias Gate = Qwen38MTPPerformanceScorecardGate
    // Internal sentinels: valid canonical values so the real claim builder
    // accepts them; replaced with the placeholder markers before printing,
    // with an exactly-once guard so a sentinel colliding with a pinned
    // value can never silently corrupt the template.
    let harnessSentinel = String(repeating: "0f", count: 20)
    let hostSentinel = String(repeating: "0e", count: 32)
    let pairSentinel = String(repeating: "0d", count: 32)

    let bytes = try Qwen38ScorecardRunClaimVerifier.claimBytes(
        fields: Qwen38ScorecardRunClaimFields(
            subject: .mtpScorecardResultPair,
            modelSHA256: Gate.requiredArtifact.targetConfigSHA256,
            tokenizerSHA256: Gate.requiredArtifact.tokenizerSHA256,
            tensorManifestSHA256: Gate.requiredArtifact.targetTensorManifestSHA256,
            chatTemplateSHA256: Gate.requiredWorkload.chatTemplateSHA256,
            quantizationIdentity: Gate.requiredArtifact.targetQuantizationMode,
            target: Qwen38ScorecardModelIdentity(
                modelID: Gate.requiredArtifactLock.targetIdentity.modelID,
                revision: Gate.requiredArtifact.targetRevision),
            drafter: Qwen38ScorecardModelIdentity(
                modelID: Gate.requiredArtifactLock.drafterIdentity.modelID,
                revision: Gate.requiredArtifact.drafterRevision),
            sourceID: Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID,
            hostAdmissionID: hostSentinel,
            harnessGitSHA1: harnessSentinel,
            gdnOnMode: .on,
            gdnOffMode: .off,
            corpusID: Gate.requiredWorkload.id,
            corpusContentSHA256: Gate.requiredWorkload.contentSHA256,
            resultPairID: pairSentinel))

    var text = String(decoding: bytes, as: UTF8.self)
    for (sentinel, marker) in [
        (harnessSentinel, qwen38ScorecardClaimTemplateMarkers.harnessGitSHA1),
        (hostSentinel, qwen38ScorecardClaimTemplateMarkers.hostAdmissionID),
        (pairSentinel, qwen38ScorecardClaimTemplateMarkers.resultPairID),
    ] {
        let components = text.components(separatedBy: sentinel)
        guard components.count == 2 else {
            throw Qwen38ScorecardClaimTemplateError
                .sentinelSubstitutionFailed(marker)
        }
        text = components.joined(separator: marker)
    }
    return text
}
