// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "fast-mlx-spike",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "SpikeCore", targets: ["SpikeCore"]),
        .executable(name: "spike-cli", targets: ["spike-cli"]),
        .executable(name: "fastmlx-harness", targets: ["fastmlx-harness"]),
        .executable(name: "fastmlx-capacity", targets: ["fastmlx-capacity"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", exact: "0.31.6"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", revision: "702e5a0eaf990e1f6d3db2b6e7d8872858a44055"),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "SpikeCore",
            dependencies: [
                "HarnessCore",  // pure (no MLX) — TurboQuant codec consumes its Lloyd-Max codebook
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .executableTarget(
            name: "spike-cli",
            dependencies: [
                "SpikeCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "SpikeCoreTests",
            dependencies: ["SpikeCore"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        // HarnessCore is PURE — NO MLX/SpikeCore dependency — so it (and HarnessCoreTests) build+test off-box with `swift test`.
        .target(
            name: "HarnessCore",
            dependencies: [],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        // Only the executable pulls in SpikeCore (MLX); SwiftEngineDriver.swift lives HERE, not in HarnessCore.
        .executableTarget(
            name: "fastmlx-harness",
            dependencies: [
                "HarnessCore",
                "SpikeCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "HarnessCoreTests",
            dependencies: ["HarnessCore"],
            resources: [.process("Fixtures")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        // MLX-coupled harness orchestration tests. These must run through Xcode on the bench Mac
        // so the mlx-swift package plugin emits the required Metal library.
        .testTarget(
            name: "FastMLXHarnessTests",
            dependencies: [
                "fastmlx-harness",
                "HarnessCore",
                "SpikeCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        // MLX-free by design: real host introspection (sysctlbyname, Metal) so the capacity
        // advisor builds and runs on any Mac without the full inference-engine toolchain.
        .target(
            name: "SystemProfiler",
            dependencies: ["HarnessCore"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .executableTarget(
            name: "fastmlx-capacity",
            dependencies: ["SystemProfiler", "HarnessCore"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
