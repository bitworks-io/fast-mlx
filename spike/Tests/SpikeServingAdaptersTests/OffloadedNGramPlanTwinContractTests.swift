// PROJECTION-ONLY: this file lives under `public/sanitized-projection/spike/Tests/
// SpikeServingAdaptersTests/`, not under `spike/Tests/`, so it never compiles in the development
// tree. `public/public-repository.json` copies it into the public checkout at
// `spike/Tests/SpikeServingAdaptersTests/OffloadedNGramPlanTwinContractTests.swift`. It cannot
// live in the development tree because the development implementation it would be testing does
// not exist there: the development `checkOffloadedNGramPlanOnly` really resolves and verifies an
// offloaded n-gram plan and does not throw `ModelFactoryError.unsupportedModelType` on a
// well-formed fixture, so an assertion that it throws that error would simply be wrong against
// development code.
//
// It is the paired replacement for the six cases the public projection excludes from
// `spike/Tests/SpikeServingAdaptersTests/OffloadedNGramPlanCheckTests.swift` (see that file's
// header, and the `exclude` entry for it in `public/public-repository.json`): those cases assert
// real plan resolution, chunk verification, and governance-gate refusal against the development
// offloaded n-gram serving path, which the public projection does not ship. Excluding them without
// a replacement would silently drop coverage of that decision. This file is the replacement: it
// asserts the public-projection twin's actual documented behavior instead, so the public tree's
// offload-plan-check-only route stays exercised by a live test.
//
// It asserts the twin's throw contract through the SAME real call site the excluded dev-tree
// cases used -- `loadScalarServingModel(..., offloadPlanCheckOnly: true)` -- rather than calling
// `checkOffloadedNGramPlanOnly` (`Libraries/MLXLLM/LLMModelFactory.swift`) directly. Direct access
// would need `MLXLLM` added to the `SpikeServingAdaptersTests` target's dependencies in
// `spike/Package.swift`; that file is copied byte-for-byte into the projection (it is not itself a
// sanitized override) and widening its dependency graph is outside the scope of this test and its
// manifest entry. `loadScalarServingModel` reaches `checkOffloadedNGramPlanOnly` directly and
// funnels ANY error it throws, decorated only by `String(describing:)`, into
// `ScalarServingModelLoadError.offloadedNGramPlanLoadFailed(detail:)` -- see that call site's own
// comment in `MLXScalarServing.swift`. So asserting the `detail` string here is, one level
// removed, asserting `checkOffloadedNGramPlanOnly`'s own throw: the development-tree
// implementation never reaches this call site with `.unsupportedModelType` (it does real
// resolution against the fixture below and would instead complete or fail with a DIFFERENT typed
// detail), so this assertion discriminates public-projection behavior from development behavior
// rather than being tautologically true in both trees.
//
// The fixture below is deliberately minimal (a bare `qwen4_exp` `config.json` and an empty plan
// file) precisely because the public-projection twin never reads either: it throws
// unconditionally, before any plan parsing. A fixture built to survive real plan resolution (like
// the excluded dev-tree cases use) is neither required nor available here.

import Foundation
import XCTest

import MLXLMCommon
import ServingCore
import SpikeCore
@testable import SpikeServingAdapters

final class OffloadedNGramPlanTwinContractTests: XCTestCase {
    private func writeMinimalOffloadEligibleModelFixture() throws -> (modelDirectory: URL, planURL: URL) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "offloaded-ngram-plan-twin-contract-\(UUID().uuidString)", isDirectory: true)
        let modelDirectory = root.appendingPathComponent("model", isDirectory: true)
        try FileManager.default.createDirectory(
            at: modelDirectory, withIntermediateDirectories: true)
        try Data(#"{"model_type":"qwen4_exp","num_hidden_layers":4}"#.utf8)
            .write(to: modelDirectory.appendingPathComponent("config.json"))

        // Deliberately empty: the public-projection twin throws before ever reading this file.
        let planURL = root.appendingPathComponent("plan.json")
        try Data().write(to: planURL)

        return (modelDirectory: modelDirectory, planURL: planURL)
    }

    /// THE DECISIVE ASSERTION for the public projection's offloaded n-gram serving facade: driven
    /// through the real `loadScalarServingModel` call site with `offloadPlanCheckOnly: true`, a
    /// `qwen4_exp` fixture in this tree must fail closed with the public-projection twin's throw
    /// -- never resolve, verify, or complete via `OffloadPlanCheckCompleted` -- because the
    /// offloaded n-gram serving path this facade fronts is not part of the public projection. The
    /// `detail` string must name that fact explicitly, not merely be present, so a future edit
    /// that swaps in a DIFFERENT failure (e.g. a generic file-not-found) still fails this test.
    func testCheckOffloadedNGramPlanOnlyThrowsUnsupportedModelTypeNamingThePublicProjection()
        async throws
    {
        let fixture = try writeMinimalOffloadEligibleModelFixture()
        defer {
            try? FileManager.default.removeItem(
                at: fixture.modelDirectory.deletingLastPathComponent())
        }

        do {
            _ = try await loadScalarServingModel(
                configuration: ScalarServingModelLoadConfiguration(
                    launchedModel: "qwen4-exp-offload-plan-check-only",
                    modelDirectory: fixture.modelDirectory,
                    memoryLimitBytes: 8_192,
                    cacheLimitBytes: 1_024,
                    backendConfiguration: fixtureBackendConfiguration(),
                    ngramOffloadPlanURL: fixture.planURL,
                    offloadPlanCheckOnly: true))
            XCTFail(
                "the public-projection twin must fail closed on every input, never complete "
                    + "via OffloadPlanCheckCompleted")
        } catch let error as ScalarServingModelLoadError {
            guard case .offloadedNGramPlanLoadFailed(let detail) = error else {
                XCTFail("expected .offloadedNGramPlanLoadFailed, got \(error)")
                return
            }
            XCTAssertTrue(
                detail.contains("unsupportedModelType"),
                "the public-projection twin throws ModelFactoryError.unsupportedModelType; "
                    + "detail: \(detail)")
            XCTAssertTrue(
                detail.contains("not part of the public projection"),
                "the thrown message must name the public projection as the reason, not just "
                    + "that something failed; detail: \(detail)")
        } catch let completion as OffloadPlanCheckCompleted {
            XCTFail(
                "the public-projection twin must never resolve a real plan: \(completion)")
        }
    }
}
