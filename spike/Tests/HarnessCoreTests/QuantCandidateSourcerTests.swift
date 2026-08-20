import XCTest
@testable import HarnessCore

/// TDD for quant candidate SOURCING (fit-checked-serve #2 full shape, policy decision #3): given a base
/// model repo id, enumerate the ordered mlx-community quant-variant repo ids to probe. Pure + offline —
/// this is only the NAME generator (the network metadata-probe half is a deliberately-deferred
/// follow-on); a wrong pattern costs at most a harmless 404 at probe time, never a fatal.
///
/// The naming conventions are grounded in the mlx-community HF org (verified 2026-08-19): `{base}-Nbit`
/// (the mlx-lm convert default), `{base}-Nbit-DWQ`, and `{base}-bf16` coexist for one base model. DWQ =
/// Distilled Weight Quantization (mlx-lm LEARNED_QUANTS.md): a same-bit-width quality-recovery variant,
/// so at equal bits it is preferred over plain — the enumeration emits DWQ before plain per width.
final class QuantCandidateSourcerTests: XCTestCase {

    // MARK: - a bare base name is qualified under mlx-community and expanded to the variant set

    func testEnumeratesOrderedVariantsForBareBase() {
        let ids = QuantCandidateSourcer.enumerate(baseRepoID: "Qwen3-8B")
        // canonical patterns present, all mlx-community-qualified
        XCTAssertTrue(ids.contains("mlx-community/Qwen3-8B-4bit-DWQ"), "\(ids)")
        XCTAssertTrue(ids.contains("mlx-community/Qwen3-8B-4bit"))
        XCTAssertTrue(ids.contains("mlx-community/Qwen3-8B-8bit"))
        XCTAssertTrue(ids.contains("mlx-community/Qwen3-8B-bf16"))
        XCTAssertTrue(ids.allSatisfy { $0.hasPrefix("mlx-community/Qwen3-8B-") })
    }

    // MARK: - an org-qualified base keeps its org and is expanded

    func testEnumeratesForOrgQualifiedBase() {
        let ids = QuantCandidateSourcer.enumerate(baseRepoID: "mlx-community/Qwen3-32B")
        XCTAssertFalse(ids.isEmpty)
        XCTAssertTrue(ids.allSatisfy { $0.hasPrefix("mlx-community/Qwen3-32B-") },
                      "every candidate extends the org-qualified base: \(ids)")
    }

    // MARK: - a base that ALREADY carries a quant suffix returns itself only (no -4bit-4bit)

    func testAlreadyQuantizedReturnsItself() {
        XCTAssertEqual(
            QuantCandidateSourcer.enumerate(baseRepoID: "mlx-community/Qwen3-8B-4bit"),
            ["mlx-community/Qwen3-8B-4bit"])
        // case variant (mlx-community also publishes -4Bit) is still recognized as already-quantized
        XCTAssertEqual(
            QuantCandidateSourcer.enumerate(baseRepoID: "mlx-community/Mistral-7B-Instruct-v0.3-mlx-4Bit"),
            ["mlx-community/Mistral-7B-Instruct-v0.3-mlx-4Bit"])
        // an unquantized half-precision suffix is also a terminal variant
        XCTAssertEqual(
            QuantCandidateSourcer.enumerate(baseRepoID: "mlx-community/Qwen3-8B-bf16"),
            ["mlx-community/Qwen3-8B-bf16"])
    }

    // MARK: - structured variants: parsed bits + DWQ flag (the future picker tiebreak hook)

    func testAlreadyDWQReturnsItselfWithParsedBitsAndFlag() {
        let vs = QuantCandidateSourcer.variants(baseRepoID: "mlx-community/Qwen3-8B-4bit-DWQ")
        XCTAssertEqual(vs.count, 1)
        XCTAssertEqual(vs.first?.repoID, "mlx-community/Qwen3-8B-4bit-DWQ")
        XCTAssertEqual(vs.first?.quantBits, 4)
        XCTAssertEqual(vs.first?.isDWQ, true)
    }

    func testBf16VariantIsUnquantized() {
        let vs = QuantCandidateSourcer.variants(baseRepoID: "Qwen3-8B")
        let bf16 = vs.first { $0.repoID.hasSuffix("-bf16") }
        XCTAssertNotNil(bf16)
        XCTAssertNil(bf16?.quantBits, "bf16 is unquantized → nil bits")
        XCTAssertEqual(bf16?.isDWQ, false)
    }

    // MARK: - DWQ ranks before plain at equal bit width (quality-recovery precedence, sourced)

    func testDWQRanksBeforePlainAtEqualBits() {
        let ids = QuantCandidateSourcer.enumerate(baseRepoID: "Qwen3-8B")
        for width in [3, 4, 6, 8] {
            guard let dwq = ids.firstIndex(of: "mlx-community/Qwen3-8B-\(width)bit-DWQ"),
                  let plain = ids.firstIndex(of: "mlx-community/Qwen3-8B-\(width)bit")
            else { continue }
            XCTAssertLessThan(dwq, plain, "\(width)bit-DWQ must precede \(width)bit")
        }
    }

    // MARK: - stable, de-duplicated ordering

    func testEnumerationIsDeduplicatedAndStable() {
        let ids = QuantCandidateSourcer.enumerate(baseRepoID: "Qwen3-8B")
        XCTAssertEqual(ids.count, Set(ids).count, "no duplicate repo ids: \(ids)")
        // deterministic: same input → same order
        XCTAssertEqual(ids, QuantCandidateSourcer.enumerate(baseRepoID: "Qwen3-8B"))
    }
}
