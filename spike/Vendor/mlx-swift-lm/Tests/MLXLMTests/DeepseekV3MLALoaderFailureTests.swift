import Foundation
import MLX
@testable import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

final class DeepseekV3MLALoaderFailureTests: XCTestCase {
    private static let childModeEnvironmentKey =
        "MLXLM_DEEPSEEK_MLA_LOADER_FAILURE_CHILD"
    private static let fixtureDirectoryEnvironmentKey =
        "MLXLM_DEEPSEEK_MLA_LOADER_FAILURE_FIXTURE_DIR"
    private static let childTestFilter =
        "DeepseekV3MLALoaderFailureTests/testChildModeLoadsMalformedMixedProjectionFixture"
    private static let expectedDiagnostic =
        "DeepSeek-V3 weight conversion failed: source and converted MLA projection weights are both present for layer 0"

    func testChildEnvironmentDropsInheritedXCTestSessionState() {
        let inherited = [
            "HOME": "/tmp/loader-test-home",
            "XCTestBundlePath": "/tmp/parent-tests.xctest",
            "XCTestConfigurationFilePath": "/tmp/parent.xctestconfiguration",
            "XCTestManagerVariant": "IDE",
            "XCTestSessionIdentifier": "parent-session",
        ]

        let child = Self.sanitizedChildEnvironment(inherited)

        XCTAssertEqual(child["HOME"], inherited["HOME"])
        XCTAssertFalse(child.keys.contains { $0.hasPrefix("XCTest") })
    }

    func testMalformedMixedProjectionFixtureDiesBeforeParameterUpdate() throws {
        guard ProcessInfo.processInfo.environment[Self.childModeEnvironmentKey] != "1" else {
            return
        }

        let fixtureDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("DeepseekV3MLALoaderFailureTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: fixtureDirectory, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: fixtureDirectory)
        }

        try Self.writeMalformedMixedProjectionFixture(to: fixtureDirectory)

        let result = try Self.runChildLoaderFailureTest(fixtureDirectory: fixtureDirectory)

        XCTAssertNotEqual(result.terminationStatus, 0)
        XCTAssertTrue(
            result.output.contains("testChildModeLoadsMalformedMixedProjectionFixture"),
            "child output did not show the selected child test:\n\(result.output)")
        XCTAssertFalse(
            result.output.contains("testMalformedMixedProjectionFixtureDiesBeforeParameterUpdate"),
            "child filter also selected the parent test:\n\(result.output)")
        let metalUnavailable =
            result.output.contains("Failed to load the default metallib")
            || (
                result.output.contains("index 0 beyond bounds for empty array")
                    && result.output.contains("load_device")
            )
        guard !metalUnavailable else {
            throw XCTSkip("MLX default metallib is unavailable before the loader reaches sanitize")
        }
        XCTAssertTrue(
            result.output.contains(Self.expectedDiagnostic),
            "child output did not contain expected diagnostic:\n\(result.output)")
    }

    func testChildModeLoadsMalformedMixedProjectionFixture() throws {
        guard ProcessInfo.processInfo.environment[Self.childModeEnvironmentKey] == "1" else {
            return
        }
        guard let fixtureDirectoryPath =
            ProcessInfo.processInfo.environment[Self.fixtureDirectoryEnvironmentKey]
        else {
            fatalError("missing \(Self.fixtureDirectoryEnvironmentKey)")
        }

        print(Self.childTestFilter)
        let fixtureDirectory = URL(fileURLWithPath: fixtureDirectoryPath)
        let model = try DeepseekV3MLASanitizeProbeModel(Self.tinyDeepseekV3Configuration())
        try loadWeights(modelDirectory: fixtureDirectory, model: model)
    }

    private static func runChildLoaderFailureTest(
        fixtureDirectory: URL
    ) throws -> (terminationStatus: Int32, output: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        process.currentDirectoryURL = packageRoot
        process.arguments = [
            "xctest",
            "-XCTest",
            childTestFilter,
            Bundle(for: Self.self).bundlePath,
        ]

        var environment = sanitizedChildEnvironment(
            ProcessInfo.processInfo.environment)
        environment[childModeEnvironmentKey] = "1"
        environment[fixtureDirectoryEnvironmentKey] = fixtureDirectory.path
        process.environment = environment

        let outputURL = fixtureDirectory.appendingPathComponent("child-xctest-output.txt")
        FileManager.default.createFile(atPath: outputURL.path, contents: nil)
        let outputHandle = try FileHandle(forWritingTo: outputURL)
        process.standardOutput = outputHandle
        process.standardError = outputHandle

        try process.run()
        process.waitUntilExit()
        try outputHandle.close()

        let outputData = try Data(contentsOf: outputURL)
        let output = String(data: outputData, encoding: .utf8) ?? ""
        return (process.terminationStatus, output)
    }

    private static func sanitizedChildEnvironment(
        _ inherited: [String: String]
    ) -> [String: String] {
        inherited.filter { !$0.key.hasPrefix("XCTest") }
    }

    private static func writeMalformedMixedProjectionFixture(to directory: URL) throws {
        let prefix = "model.layers.0.self_attn"
        let weightsURL = directory.appendingPathComponent("model.safetensors")
        let sourceByteCount = 8 * 4 * MemoryLayout<Float32>.size
        let convertedByteCount = 2 * 4 * 2 * MemoryLayout<Float32>.size
        let totalByteCount = sourceByteCount + convertedByteCount
        var header = """
            {"\(prefix).kv_b_proj.weight":{"dtype":"F32","shape":[8,4],"data_offsets":[0,\(sourceByteCount)]},"\(prefix).embed_q.weight":{"dtype":"F32","shape":[2,4,2],"data_offsets":[\(sourceByteCount),\(totalByteCount)]}}
            """
        while header.utf8.count % 8 != 0 {
            header.append(" ")
        }

        var data = Data()
        let headerByteCount = UInt64(header.utf8.count)
        for byteOffset in stride(from: 0, to: 64, by: 8) {
            data.append(UInt8(truncatingIfNeeded: headerByteCount >> UInt64(byteOffset)))
        }
        data.append(Data(header.utf8))
        data.append(Data(count: totalByteCount))
        try data.write(to: weightsURL)
    }

    private static var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private static func tinyDeepseekV3Configuration() throws -> DeepseekV3Configuration {
        let json = """
            {
              "model_type": "deepseek_v3",
              "vocab_size": 32,
              "hidden_size": 8,
              "intermediate_size": 16,
              "moe_intermediate_size": 8,
              "num_hidden_layers": 1,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "routed_scaling_factor": 1.0,
              "kv_lora_rank": 4,
              "q_lora_rank": 4,
              "qk_rope_head_dim": 2,
              "v_head_dim": 2,
              "qk_nope_head_dim": 2,
              "norm_topk_prob": false,
              "moe_layer_freq": 1,
              "first_k_dense_replace": 100,
              "max_position_embeddings": 16,
              "rms_norm_eps": 0.000001,
              "rope_theta": 10000.0,
              "attention_bias": false
            }
            """

        return try JSONDecoder().decode(DeepseekV3Configuration.self, from: Data(json.utf8))
    }
}

private final class DeepseekV3MLASanitizeProbeModel: Module, BaseLanguageModel {
    private let configuration: DeepseekV3Configuration

    init(_ configuration: DeepseekV3Configuration) {
        self.configuration = configuration
        super.init()
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        do {
            return try convertDeepseekV3MLAProjectionWeights(
                normalizeDeepseekV3PackedInt4Weights(weights),
                configuration: configuration)
        } catch {
            let diagnostic = "DeepSeek-V3 weight conversion failed: \(error)"
            FileHandle.standardError.write(Data((diagnostic + "\n").utf8))
            fatalError(diagnostic)
        }
    }
}
