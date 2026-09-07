import XCTest

@testable import ServingCore

/// Acceptance: item 4's ServingCore-side selection enum carries the pinned per-artifact
/// deployment-policy values the decision doc records, and stays MLX-free (the enum's own
/// definition is the proof the type only names Foundation-visible types -- `ServingCore` has zero
/// package dependencies, see `Package.swift`).
final class InCheckpointMTPSelectionTests: XCTestCase {
    /// Acceptance criterion: the sole serve-eligible case maps to the converted namespace, the
    /// independently-verified source-key count (76), and the pinned revision recorded in
    /// `docs/task-inbox/2026-09-07-qwen4exp-mtp-serving-wiring-DECISION.md`.
    func testConverted4BitSelectionCarriesThePinnedConvertedNamespaceKeyCountAndRevision() {
        let selection = FastMLXInCheckpointMTPSelection.converted4Bit

        XCTAssertEqual(selection.namespace, .converted)
        XCTAssertEqual(selection.expectedSourceKeyCount, 76)
        XCTAssertEqual(selection.revision, "43a82b3f0ff64fa417fd09ca046580f08d19b0d6")
    }

    /// Acceptance criterion: the raw string used on any future CLI surface is a stable, readable
    /// token -- guards against an accidental rename silently changing a serialized/parsed value.
    func testConverted4BitSelectionRawValueIsStable() {
        XCTAssertEqual(FastMLXInCheckpointMTPSelection.converted4Bit.rawValue, "converted-4bit")
    }

    /// Acceptance criterion: the official BF16 namespace mirror exists (the loader-level layout
    /// fact is real), but the decision doc's honest-limit constraint means NO selection case maps
    /// to it -- so `FastMLXInCheckpointMTPSelection.converted4Bit` is exhaustive by construction. The
    /// enforcement is the `switch` itself, NOT the runtime assertions inside it: this `switch` has
    /// NO `default:` clause, so it is exhaustive over `FastMLXInCheckpointMTPSelection`'s cases as they
    /// exist at compile time. Adding a case (e.g. an official-namespace selection) makes this
    /// `switch` non-exhaustive and fails the BUILD, not the test run -- forcing a reviewer to
    /// re-justify the 360 GB claim (and extend this test) rather than silently gaining a serving
    /// path no host can hold. Verified directly: temporarily adding a second case to
    /// `FastMLXInCheckpointMTPSelection` breaks `SpikeTests`' build at this `switch` with "switch must
    /// be exhaustive"; reverting restores a clean build and a green suite.
    func testOfficialNamespaceHasNoServeEligibleSelectionCase() {
        let allRawValues = Set(
            FastMLXInCheckpointMTPNamespace.allCases.map(\.rawValue))
        XCTAssertEqual(allRawValues, ["official", "converted"])

        for selection in FastMLXInCheckpointMTPSelection.allCases {
            switch selection {
            case .converted4Bit:
                XCTAssertEqual(
                    selection.namespace,
                    .converted,
                    "the only selection case must resolve to the converted namespace, never official")
            }
        }
    }
}
