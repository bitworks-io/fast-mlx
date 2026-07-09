// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "fast-mlx-spike",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "SpikeCore", targets: ["SpikeCore"]),
        .executable(name: "spike-cli", targets: ["spike-cli"]),
        .executable(name: "fastmlx-harness", targets: ["fastmlx-harness"]),
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
            dependencies: ["HarnessCore", "SpikeCore"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "HarnessCoreTests",
            dependencies: ["HarnessCore"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
