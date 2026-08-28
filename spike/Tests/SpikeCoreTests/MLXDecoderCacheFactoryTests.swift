import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import SpikeCore

private final class CacheFactorySpyModel: Module, LanguageModel {
    private(set) var nativeCacheBuildCount = 0

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        nativeCacheBuildCount += 1
        return [KVCacheSimple()]
    }

    func prepare(
        _ input: LMInput,
        cache: [KVCache],
        windowSize: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray.zeros([inputs.dim(0), inputs.dim(1), 8])
    }
}

private final class CacheFactorySpy {
    private(set) var callCount = 0

    func makeInt8Caches() -> [KVCache] {
        callCount += 1
        return [QuantizedKVCache(groupSize: 32, bits: 8, mode: .affine)]
    }
}

final class MLXDecoderCacheFactoryTests: XCTestCase {
    func testResetRebuildsTheConfiguredCacheKindWithoutCallingNativeFactory() {
        let model = CacheFactorySpyModel()
        let factory = CacheFactorySpy()
        var decoder = MLXDecoder(model: model, cacheFactory: { factory.makeInt8Caches() })

        XCTAssertEqual(factory.callCount, 1)
        XCTAssertEqual(model.nativeCacheBuildCount, 0)

        decoder.reset()

        XCTAssertEqual(factory.callCount, 2)
        XCTAssertEqual(model.nativeCacheBuildCount, 0)
    }
}
