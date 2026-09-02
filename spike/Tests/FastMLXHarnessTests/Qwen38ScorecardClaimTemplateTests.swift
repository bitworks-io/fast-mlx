import Foundation
import HarnessCore
import ProofControl
import XCTest

@testable import fastmlx_harness

/// The claim template must be format-exact (byte-identical to the real
/// canonical claim builder after placeholder substitution) and must carry
/// the pinned digests so operators never retype them.
final class Qwen38ScorecardClaimTemplateTests: XCTestCase {
    private typealias Gate = Qwen38MTPPerformanceScorecardGate

    func testTemplateCarriesPinnedDigestsAndPlaceholders() throws {
        let template = try qwen38ScorecardClaimTemplate()
        for marker in [
            qwen38ScorecardClaimTemplateMarkers.harnessGitSHA1,
            qwen38ScorecardClaimTemplateMarkers.hostAdmissionID,
            qwen38ScorecardClaimTemplateMarkers.resultPairID,
        ] {
            XCTAssertTrue(template.contains(marker), "missing \(marker)")
        }
        XCTAssertTrue(
            template.contains(
                "model_sha256=\(Gate.requiredArtifact.targetConfigSHA256)"))
        XCTAssertTrue(
            template.contains(
                "tensor_manifest_sha256=\(Gate.requiredArtifact.targetTensorManifestSHA256)"
            ))
        XCTAssertTrue(
            template.contains(
                "tokenizer_sha256=\(Gate.requiredArtifact.tokenizerSHA256)"))
        XCTAssertTrue(
            template.contains(
                "chat_template_sha256=\(Gate.requiredWorkload.chatTemplateSHA256)"
            ))
        XCTAssertTrue(template.contains("corpus_id=\(Gate.requiredWorkload.id)"))
        XCTAssertTrue(
            template.contains(
                "corpus_content_sha256=\(Gate.requiredWorkload.contentSHA256)"))
        XCTAssertTrue(
            template.contains(
                "source_id=\(Qwen38MTPLiveExactnessGate.requiredSourceIdentity.sourceID)"
            ))
        XCTAssertTrue(
            template.contains(
                "target_model_id=\(Gate.requiredArtifactLock.targetIdentity.modelID)"
            ))
    }

    /// Exact-format proof: substituting the three markers with valid
    /// sentinel values must reproduce the REAL canonical claim bytes for
    /// an independently constructed fields value byte-for-byte.
    func testTemplateWithSentinelsIsExactlyCanonicalClaimBytes() throws {
        let harnessSentinel = String(repeating: "f", count: 40)
        let hostSentinel = String(repeating: "e", count: 64)
        let pairSentinel = String(repeating: "d", count: 64)
        let substituted = try qwen38ScorecardClaimTemplate()
            .replacingOccurrences(
                of: qwen38ScorecardClaimTemplateMarkers.harnessGitSHA1,
                with: harnessSentinel)
            .replacingOccurrences(
                of: qwen38ScorecardClaimTemplateMarkers.hostAdmissionID,
                with: hostSentinel)
            .replacingOccurrences(
                of: qwen38ScorecardClaimTemplateMarkers.resultPairID,
                with: pairSentinel)

        let expected = try Qwen38ScorecardRunClaimVerifier.claimBytes(
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

        XCTAssertEqual(Data(substituted.utf8), expected)
    }
}
