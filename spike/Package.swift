// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "fast-mlx-spike",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "ServingCore", targets: ["ServingCore"]),
        .library(name: "ServingNIO", targets: ["ServingNIO"]),
        .library(name: "SpikeServingAdapters", targets: ["SpikeServingAdapters"]),
        .library(name: "SpikeCore", targets: ["SpikeCore"]),
        .library(name: "ProofControl", targets: ["ProofControl"]),
        .executable(name: "fastmlx-serve", targets: ["fastmlx-serve"]),
        .executable(name: "spike-cli", targets: ["spike-cli"]),
        .executable(name: "fastmlx-harness", targets: ["fastmlx-harness"]),
        .executable(name: "fastmlx-capacity", targets: ["fastmlx-capacity"]),
        .executable(name: "fastmlx-proof-runner", targets: ["fastmlx-proof-runner"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", exact: "0.31.6"),
        .package(path: "Vendor/mlx-swift-lm"),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/apple/swift-nio.git", exact: "2.101.2"),
    ],
    targets: [
        // ServingCore is PURE — NO MLX/SpikeCore dependency — so protocol, policy, and
        // lifecycle contracts remain independently buildable and testable off-box.
        .target(
            name: "ServingCore",
            dependencies: [],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "ServingCoreTests",
            dependencies: ["ServingCore"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        // ServingSamplingBridge is MLX-free: it imports HarnessCore's tested sampling
        // contract and ServingCore's serving-boundary policy and converts between them.
        // It lives in its own target so both dependencies stay pure and the bridge
        // (and its tests) build+run off-box with `swift test`.
        .target(
            name: "ServingSamplingBridge",
            dependencies: ["HarnessCore", "ServingCore"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "ServingSamplingBridgeTests",
            dependencies: ["ServingSamplingBridge", "HarnessCore", "ServingCore"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        // ServingSnapshotBridge is MLX-free: it maps ServingCore's native-layout snapshot-reuse
        // classification onto HarnessCore's cold-plane restore granularity, so a consumer cannot pair
        // a cache layout with the wrong reuse arithmetic. Own target for the same reason as the
        // sampling bridge — both pure targets stay dependency-free and this builds+tests off-box.
        .target(
            name: "ServingSnapshotBridge",
            dependencies: ["HarnessCore", "ServingCore"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "ServingSnapshotBridgeTests",
            dependencies: ["ServingSnapshotBridge", "HarnessCore", "ServingCore"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        // ServingNIO owns transport only. Model, tokenizer, and MLX state remain behind
        // ServingCore contracts and actor-confined adapters.
        .target(
            name: "ServingNIO",
            dependencies: [
                "ServingCore",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "ServingNIOTests",
            dependencies: [
                "ServingCore",
                "ServingNIO",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOEmbedded", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .target(
            name: "SpikeServingAdapters",
            dependencies: [
                "HarnessCore",
                "ServingCore",
                "SpikeCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "SpikeServingAdaptersTests",
            dependencies: [
                "HarnessCore",
                "ServingCore",
                "ServingNIO",
                "SpikeCore",
                "SpikeServingAdapters",
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOEmbedded", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        // Composition executable: transport-only scripted mode remains available for isolated
        // tests, while model serving is delegated to actor-confined serving adapters.
        .executableTarget(
            name: "fastmlx-serve",
            dependencies: [
                "HarnessCore",
                "ServingCore",
                "ServingNIO",
                "ServingSnapshotBridge",
                "SpikeCore",
                "SpikeServingAdapters",
                "SystemProfiler",
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
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
            dependencies: [
                "SpikeCore",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        // Keep compiled snapshot/restore tests independently filterable. The documented on-box
        // verification invokes this target in a separate xcodebuild process because MLX compiled
        // functions use pointer-keyed backend identities, and rapid construction of unrelated
        // compiled runtime fixtures in one XCTest process can contaminate later test suites.
        .testTarget(
            name: "ExactPrefixMLXTests",
            dependencies: [
                "HarnessCore",
                "SpikeCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
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
                "ProofControl",
                "ServingCore",
                "SpikeCore",
                "SpikeServingAdapters",
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
        // ProofControl is the target-level MLX-free trust root for source/build/evidence
        // admission. Keep its filesystem and process primitives in this dependency-free target.
        .target(
            name: "ProofControl",
            dependencies: [],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .executableTarget(
            name: "fastmlx-proof-runner",
            dependencies: ["ProofControl"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "ProofControlTests",
            // HarnessCore (dependency-free, MLX-free) is a TEST-ONLY dependency:
            // the Slice 3 cross-recipe pin proves the runner-minted process-
            // isolation evidence ID is byte-identical to the worker-side
            // HarnessCore recipe. The ProofControl production target itself
            // stays dependency-free.
            dependencies: ["ProofControl", "HarnessCore"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        // MLX-coupled harness orchestration tests. These must run through Xcode on the bench Mac
        // so the mlx-swift package plugin emits the required Metal library.
        .testTarget(
            name: "FastMLXHarnessTests",
            dependencies: [
                "fastmlx-harness",
                "HarnessCore",
                "ServingCore",
                "SpikeCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
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
        .testTarget(
            name: "SystemProfilerTests",
            dependencies: ["SystemProfiler", "HarnessCore"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .executableTarget(
            name: "fastmlx-capacity",
            dependencies: ["SystemProfiler", "HarnessCore"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
