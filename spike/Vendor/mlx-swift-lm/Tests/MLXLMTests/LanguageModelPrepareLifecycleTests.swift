// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

final class LanguageModelPrepareLifecycleTests: XCTestCase {

    func testLoadWeightsRunsLanguageModelPrepareOnceAfterParameterUpdate() throws {
        let model = PrepareSpyLanguageModel()
        let directory = try Self.emptyFixtureDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try loadWeights(modelDirectory: directory, model: model)

        XCTAssertEqual(model.events, [.update, .prepare])
    }

    func testLoadWeightsPropagatesThrowingLanguageModelPrepare() throws {
        let model = PrepareSpyLanguageModel(prepareError: PrepareSpyError.expected)
        let directory = try Self.emptyFixtureDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertThrowsError(try loadWeights(modelDirectory: directory, model: model)) { error in
            XCTAssertEqual(error as? PrepareSpyError, .expected)
        }
        XCTAssertEqual(model.events, [.update, .prepare])
    }

    func testLoadWeightsKeepsDefaultNoOpLanguageModelPrepareGreen() throws {
        let model = DefaultPrepareLanguageModel()
        let directory = try Self.emptyFixtureDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try loadWeights(modelDirectory: directory, model: model)

        XCTAssertEqual(model.updateCount, 1)
    }

    func testLoadWeightsDoesNotRequirePrepareFromBaseLanguageModelOnly() throws {
        let model = BaseOnlyLanguageModel()
        let directory = try Self.emptyFixtureDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try loadWeights(modelDirectory: directory, model: model)

        XCTAssertEqual(model.updateCount, 1)
    }

    private static func emptyFixtureDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("LanguageModelPrepareLifecycleTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}

private enum PrepareSpyEvent: Equatable {
    case update
    case prepare
}

private enum PrepareSpyError: Error, Equatable {
    case expected
}

private final class PrepareSpyLanguageModel: Module, LanguageModel {
    private let prepareError: Error?
    private(set) var events: [PrepareSpyEvent] = []

    init(prepareError: Error? = nil) {
        self.prepareError = prepareError
        super.init()
    }

    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        events.append(.update)
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    func prepare() throws {
        events.append(.prepare)
        if let prepareError {
            throw prepareError
        }
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fatalError("not exercised")
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        []
    }
}

private final class DefaultPrepareLanguageModel: Module, LanguageModel {
    private(set) var updateCount = 0

    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        updateCount += 1
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fatalError("not exercised")
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        []
    }
}

private final class BaseOnlyLanguageModel: Module, BaseLanguageModel {
    private(set) var updateCount = 0

    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        updateCount += 1
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }
}
