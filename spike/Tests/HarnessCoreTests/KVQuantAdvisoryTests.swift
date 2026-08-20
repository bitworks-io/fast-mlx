import XCTest
@testable import HarnessCore

/// Contract tests for the `--kv-quant` sizing-only advisory (fit-checked-serve follow-on:
/// `docs/task-inbox/2026-08-19-kv-quant-serve-flag-and-snapshot-reuse-announce.md`).
///
/// The advisory previews what a NON-fp16 KV-cache tier WOULD buy (a higher context ceiling) without
/// ever claiming the served KV is quantized — the serving runtime stores KV in fp16 today. These
/// tests pin the two honesty guarantees:
///   1. an unknown tier string fails closed (never silently serves fp16 under a typo);
///   2. every preview line is labeled `runtime_not_wired`, and `.fp16` previews nothing (identical to
///      the enforced runtime), so the advisory can never manufacture a phantom GREEN.
///
/// Pure-Swift; no live model. The executable wiring in `fastmlx-serve` is a ~4-line call into this
/// type, deliberately kept thin so the whole policy is unit-tested off-box.
final class KVQuantAdvisoryTests: XCTestCase {
    private func profile(_ id: String) -> ModelArchProfile {
        guard let m = ModelArchProfile.catalog.first(where: { $0.id == id }) else {
            fatalError("missing catalog entry \(id)")
        }
        return m
    }

    private let gib = 1024 * 1024 * 1024
    private func smallHost(ramGiB: Int, wiredGiB: Int) -> SystemProfile {
        SystemProfile(chip: "test", totalRAMBytes: ramGiB * gib, wiredLimitBytes: wiredGiB * gib, wiredLimitIsMeasured: true)
    }

    // MARK: - fail-closed validation

    func testValidateTier_knownTiers_returnMatchingCase() throws {
        XCTAssertEqual(try KVQuantAdvisory.validateTier("fp16"), .fp16)
        XCTAssertEqual(try KVQuantAdvisory.validateTier("int8"), .int8)
        XCTAssertEqual(try KVQuantAdvisory.validateTier("turbo4"), .turbo4)
        XCTAssertEqual(try KVQuantAdvisory.validateTier("tq2_5"), .tq2_5)
        XCTAssertEqual(try KVQuantAdvisory.validateTier("tq3_5"), .tq3_5)
    }

    func testValidateTier_unknownTier_throwsFailClosed() {
        XCTAssertThrowsError(try KVQuantAdvisory.validateTier("int3")) { error in
            XCTAssertEqual(error as? KVQuantAdvisory.Error, .unknownTier("int3"))
        }
    }

    func testValidateTier_emptyString_throwsFailClosed() {
        XCTAssertThrowsError(try KVQuantAdvisory.validateTier("")) { error in
            XCTAssertEqual(error as? KVQuantAdvisory.Error, .unknownTier(""))
        }
    }

    // MARK: - preview lines

    func testPreviewLines_fp16_returnsEmpty() {
        // fp16 == the enforced runtime; there is nothing to preview.
        let lines = KVQuantAdvisory.previewLines(
            tier: .fp16, profile: profile("Qwen3-32B"), host: .m5Max128)
        XCTAssertTrue(lines.isEmpty, "fp16 matches the runtime — no advisory to emit")
    }

    func testPreviewLines_int8_carryRuntimeNotWiredLabelOnEveryLine() {
        let lines = KVQuantAdvisory.previewLines(
            tier: .int8, profile: profile("Qwen3-32B"), host: .m5Max128)
        XCTAssertFalse(lines.isEmpty, "a non-fp16 tier must produce a sizing preview")
        for line in lines {
            XCTAssertTrue(
                line.contains("runtime_not_wired"),
                "every advisory line must be labeled runtime_not_wired — got: \(line)")
        }
        // The tier name is surfaced (in both the human and machine lines) so the operator knows
        // which tier the preview is for.
        XCTAssertTrue(lines.contains { $0.contains("kv_quant=int8") },
            "the requested tier name must appear in the preview")
    }

    func testPreviewLines_machineLine_usesFitAdvisoryPrefix_notFrozenFitCheckKeys() {
        // The advisory must NEVER reuse the frozen `fit_*` keys of `machineReadableFields()` (gate
        // scripts anchor on those). It uses a distinct `fit_advisory_` namespace on its own line.
        let lines = KVQuantAdvisory.previewLines(
            tier: .int8, profile: profile("Qwen3-32B"), host: .m5Max128)
        let machine = lines.first { $0.contains("fit_advisory_kv_quant=") }
        XCTAssertNotNil(machine, "expected a machine-readable advisory line")
        XCTAssertTrue(machine!.contains("fit_advisory_ceiling="))
        XCTAssertTrue(machine!.contains("fit_advisory_verdict="))
        XCTAssertTrue(machine!.contains("runtime_not_wired=true"))
        // Must not collide with the enforced line's frozen keys.
        XCTAssertFalse(machine!.contains("fit_check="), "advisory must not reuse the enforced fit_check= key")
    }

    /// The whole point: a smaller KV tier lifts the context ceiling. On a box where fp16 caps the
    /// ceiling below the model's native max, int8 must model a strictly larger (or equal) ceiling —
    /// this is the mitigation the advisory names for a red fp16 verdict.
    func testPreviewLines_int8_modelsHigherOrEqualCeilingThanFp16_onConstrainedBox() {
        let host = smallHost(ramGiB: 48, wiredGiB: 40)
        let p = profile("Qwen3-32B")
        let fp16Ceiling = CapacityModel.contextCeiling(model: p, profile: host, kvQuant: .fp16, concurrency: 1)
        let int8Ceiling = CapacityModel.contextCeiling(model: p, profile: host, kvQuant: .int8, concurrency: 1)
        XCTAssertGreaterThanOrEqual(int8Ceiling, fp16Ceiling,
            "a smaller KV tier can only raise (or hold) the context ceiling")
        // And the advisory machine line must report the int8 ceiling, not the fp16 one.
        let lines = KVQuantAdvisory.previewLines(tier: .int8, profile: p, host: host)
        let machine = lines.first { $0.contains("fit_advisory_ceiling=") }
        XCTAssertNotNil(machine)
        XCTAssertTrue(machine!.contains("fit_advisory_ceiling=\(int8Ceiling)"),
            "advisory ceiling must be the int8 what-if ceiling — got: \(machine!)")
    }
}
