/// Mirrors `ProvenanceCLI.mlxSwiftVersion` (fastmlx-harness) and Package.swift's
/// mlx-swift `exact:` pin. ScorecardPairControl is MLX-free and cannot depend on
/// mlx-swift or on fastmlx-harness to read the pin directly, so this constant is
/// kept in sync by a golden cross-module test (FastMLXHarnessTests) rather than
/// by a shared source of truth.
package enum ScorecardPairControlVersions {
    package static let mlxSwiftVersion = "0.31.6"
}
