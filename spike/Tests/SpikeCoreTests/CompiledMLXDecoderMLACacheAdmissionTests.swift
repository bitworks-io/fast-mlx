import XCTest

@testable import SpikeCore

final class CompiledMLXDecoderMLACacheAdmissionTests: XCTestCase {
    func testAbsorbedDeepseekV3AcceptsOnlyFP16CacheKind() {
        XCTAssertNoThrow(
            try CompiledMLXDecoder.validateKVCacheAdmission(
                kvCache: KVCacheKind.fp16,
                isAbsorbedDeepseekV3: true))

        let unsupportedCacheKinds: [(kind: KVCacheKind, label: String)] = [
            (.affine(.k4v2G64), "affine"),
            (.turboQuant(.tqB2), "TurboQuant"),
            (.kvarn(.k4v2G128I8), "KVarN"),
        ]

        for (kind, label) in unsupportedCacheKinds {
            XCTAssertThrowsError(
                try CompiledMLXDecoder.validateKVCacheAdmission(
                    kvCache: kind,
                    isAbsorbedDeepseekV3: true),
                "absorbed DeepSeek-V3 must reject \(label) before cache construction"
            ) { error in
                XCTAssertEqual(
                    error as? CompiledMLXDecoderCacheAdmissionError,
                    .absorbedDeepseekV3RequiresFP16(requested: kind))
            }
        }
    }

    func testExplicitNonAbsorbedDeepseekFlagKeepsExistingCacheSelectionsAdmitted() {
        let existingCacheKinds: [KVCacheKind] = [
            .fp16,
            .affine(.k4v2G64),
            .turboQuant(.tqB2),
            .kvarn(.k4v2G128I8),
        ]

        for kind in existingCacheKinds {
            XCTAssertNoThrow(
                try CompiledMLXDecoder.validateKVCacheAdmission(
                    kvCache: kind,
                    isAbsorbedDeepseekV3: false),
                "non-DeepSeek or explicitly disabled MLA-family detection must preserve \(kind)")
        }
    }
}
