import Foundation
import XCTest
@testable import HarnessCore

final class KVarNReferenceTests: XCTestCase {
    private struct Fixture: Decodable {
        struct Config: Decodable {
            let groupSize: Int
            let headDimension: Int
            let iterations: Int
            let keyBits: Int
            let rtnQuantile: Float
            let valueBits: Int
        }
        struct Input: Decodable {
            let keysTokenMajor: [Float]
            let valuesTokenMajor: [Float]
        }
        struct Expected: Decodable {
            let keyAbsorbedBiasFP16Bits: [UInt16]
            let keyAbsorbedScaleFP16Bits: [UInt16]
            let keyPacked: [UInt8]
            let keyTokenScaleFP16Bits: [UInt16]
            let keysDequantizedTokenMajor: [Float]
            let valueAbsorbedBiasFP16Bits: [UInt16]
            let valueAbsorbedScaleFP16Bits: [UInt16]
            let valueChannelScaleFP16Bits: [UInt16]
            let valuePacked: [UInt8]
            let valuesDequantizedTokenMajor: [Float]
        }
        struct Source: Decodable {
            struct Generator: Decodable { let path: String; let sha256: String }
            let commit: String
            let files: [String: String]
            let repository: String
            let torchVersion: String
            let generator: Generator
        }
        let schemaVersion: Int
        let source: Source
        let config: Config
        let input: Input
        let expected: Expected
    }

    private func fixture() throws -> Fixture {
        let url = try XCTUnwrap(Bundle.module.url(
            forResource: "kvarn-reference-v1", withExtension: "json"))
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    func testPinnedOfficialFixtureIsByteIdentical() throws {
        let fixture = try fixture()
        XCTAssertEqual(fixture.schemaVersion, 1)
        XCTAssertEqual(fixture.source.commit, "7586257f1c632e63187bfacbbe21ccb51540f7b3")
        XCTAssertEqual(fixture.source.repository, "https://github.com/huawei-csl/KVarN")
        XCTAssertEqual(fixture.source.torchVersion, "2.11.0")
        XCTAssertEqual(fixture.config.rtnQuantile, 0, accuracy: 0)
        XCTAssertEqual(
            fixture.source.generator.path,
            "spike/scripts/generate_kvarn_reference_fixture.py")
        XCTAssertEqual(
            fixture.source.generator.sha256,
            "1d08947f8d9709cccf195edf3ab47031b672773692c9bcb817717f4287a28f56")
        XCTAssertEqual(fixture.source.files, [
            "vllm/model_executor/layers/quantization/kvarn/sinkhorn.py":
                "5b96b0ab28571c855857df0625a5c79f8a916b6bb88722387e3755e99d506910",
            "vllm/v1/attention/ops/kvarn_decode.py":
                "cfb2bf1a70883b0df6d4fbdde13c82afeb98415bd54bf8a2aad0b7425f355653",
            "vllm/v1/attention/ops/kvarn_store.py":
                "7bc98ba130ad84c6e9bf482d03fd9faf3d6595ce55b7a0b6ecf09f40ccf0ed5d",
        ])
        let config = KVarNReferenceConfig(
            headDimension: fixture.config.headDimension,
            groupSize: fixture.config.groupSize,
            keyBits: fixture.config.keyBits,
            valueBits: fixture.config.valueBits,
            iterations: fixture.config.iterations)

        let record = try KVarNReference.quantize(
            keysTokenMajor: fixture.input.keysTokenMajor,
            valuesTokenMajor: fixture.input.valuesTokenMajor,
            config: config)

        XCTAssertEqual(record.keyPacked, fixture.expected.keyPacked)
        XCTAssertEqual(record.keyAbsorbedScale.map(\.bitPattern),
                       fixture.expected.keyAbsorbedScaleFP16Bits)
        XCTAssertEqual(record.keyAbsorbedBias.map(\.bitPattern),
                       fixture.expected.keyAbsorbedBiasFP16Bits)
        XCTAssertEqual(record.keyTokenScale.map(\.bitPattern),
                       fixture.expected.keyTokenScaleFP16Bits)
        XCTAssertEqual(record.valuePacked, fixture.expected.valuePacked)
        XCTAssertEqual(record.valueChannelScale.map(\.bitPattern),
                       fixture.expected.valueChannelScaleFP16Bits)
        XCTAssertEqual(record.valueAbsorbedScale.map(\.bitPattern),
                       fixture.expected.valueAbsorbedScaleFP16Bits)
        XCTAssertEqual(record.valueAbsorbedBias.map(\.bitPattern),
                       fixture.expected.valueAbsorbedBiasFP16Bits)
    }

    func testPinnedFixtureDequantizesToOfficialReference() throws {
        let fixture = try fixture()
        let config = KVarNReferenceConfig(
            headDimension: fixture.config.headDimension,
            groupSize: fixture.config.groupSize,
            keyBits: fixture.config.keyBits,
            valueBits: fixture.config.valueBits,
            iterations: fixture.config.iterations)
        let record = try KVarNReference.quantize(
            keysTokenMajor: fixture.input.keysTokenMajor,
            valuesTokenMajor: fixture.input.valuesTokenMajor,
            config: config)

        let reconstructed = try KVarNReference.dequantize(record)

        XCTAssertEqual(reconstructed.keysTokenMajor.count,
                       fixture.expected.keysDequantizedTokenMajor.count)
        XCTAssertEqual(reconstructed.valuesTokenMajor.count,
                       fixture.expected.valuesDequantizedTokenMajor.count)
        for (actual, expected) in zip(
            reconstructed.keysTokenMajor, fixture.expected.keysDequantizedTokenMajor)
        {
            XCTAssertEqual(actual, expected, accuracy: 2e-5)
        }
        for (actual, expected) in zip(
            reconstructed.valuesTokenMajor, fixture.expected.valuesDequantizedTokenMajor)
        {
            XCTAssertEqual(actual, expected, accuracy: 2e-5)
        }
    }

    func testInvalidPartialNonFiniteAndUnsupportedInputsFailClosed() {
        let valid = KVarNReferenceConfig(
            headDimension: 4, groupSize: 4, keyBits: 4, valueBits: 2, iterations: 16)
        XCTAssertThrowsError(try KVarNReference.quantize(
            keysTokenMajor: [Float](repeating: 0, count: 15),
            valuesTokenMajor: [Float](repeating: 0, count: 16), config: valid))

        var nonFinite = [Float](repeating: 0, count: 16)
        nonFinite[7] = .nan
        XCTAssertThrowsError(try KVarNReference.quantize(
            keysTokenMajor: nonFinite,
            valuesTokenMajor: [Float](repeating: 0, count: 16), config: valid))

        let invalidBits = KVarNReferenceConfig(
            headDimension: 4, groupSize: 4, keyBits: 3, valueBits: 2, iterations: 16)
        XCTAssertThrowsError(try KVarNReference.quantize(
            keysTokenMajor: [Float](repeating: 0, count: 16),
            valuesTokenMajor: [Float](repeating: 0, count: 16), config: invalidBits))

        let invalidDimension = KVarNReferenceConfig(
            headDimension: 6, groupSize: 4, keyBits: 4, valueBits: 2, iterations: 16)
        XCTAssertThrowsError(try KVarNReference.quantize(
            keysTokenMajor: [Float](repeating: 0, count: 24),
            valuesTokenMajor: [Float](repeating: 0, count: 24), config: invalidDimension))
    }

    func testFiniteInputsThatOverflowTheTransformFailClosed() {
        let config = KVarNReferenceConfig(
            headDimension: 4, groupSize: 4, keyBits: 4, valueBits: 2, iterations: 16)
        let extreme = [Float](repeating: .greatestFiniteMagnitude, count: 16)

        XCTAssertThrowsError(try KVarNReference.quantize(
            keysTokenMajor: extreme, valuesTokenMajor: extreme, config: config)) { error in
            XCTAssertEqual(error as? KVarNReferenceError, .nonFiniteOutput)
        }
    }

    func testImpossibleHadamardAllocationFailsAsInvalidConfig() {
        let config = KVarNReferenceConfig(
            headDimension: 1 << 32, groupSize: 2,
            keyBits: 4, valueBits: 2, iterations: 16)

        XCTAssertThrowsError(try KVarNReference.quantize(
            keysTokenMajor: [], valuesTokenMajor: [], config: config)) { error in
            XCTAssertEqual(error as? KVarNReferenceError, .invalidConfig)
        }
    }
}
