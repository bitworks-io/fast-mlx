import XCTest

@testable import HarnessCore

/// The cold-snapshot manifest is a TRUST BOUNDARY: a restored KV snapshot must never be trusted on
/// name alone. These tests pin the schema-tag fail-closed guard (stricter than the quant-reliability
/// renderer — an ABSENT tag is rejected here, not accepted), field validation, and the payload digest
/// verification that a cold restore relies on before reusing on-disk KV bytes.
final class ExactPrefixColdSnapshotManifestTests: XCTestCase {
    private func sampleLayout() -> ColdSnapshotKVLayout {
        ColdSnapshotKVLayout(layerCount: 28, kvHeadCount: 8, headDim: 128, elementBytes: 2)
    }

    private func sampleManifest(payload: Data) -> ColdSnapshotManifest {
        ColdSnapshotManifest.make(
            id: 42,
            blockSize: 256,
            storedBlockCount: 3,
            storedPrefixTokenCount: 768,
            bytes: 12_582_912,
            kvLayout: sampleLayout(),
            payload: payload)
    }

    func testRoundTripEncodeDecode() throws {
        let payload = Data("kv-bytes-here".utf8)
        let manifest = sampleManifest(payload: payload)
        let encoded = try manifest.encode()
        let decoded = try ColdSnapshotManifest.decode(from: encoded)
        XCTAssertEqual(decoded, manifest)
    }

    func testEncodedJSONCarriesSchemaTag() throws {
        let manifest = sampleManifest(payload: Data("x".utf8))
        let encoded = try manifest.encode()
        let json = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        XCTAssertEqual(json["schema"] as? String, "exact-prefix-cold-snapshot/v1")
    }

    func testForeignSchemaTagRejected() throws {
        let manifest = sampleManifest(payload: Data("x".utf8))
        var object = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: manifest.encode()) as? [String: Any])
        object["schema"] = "exact-prefix-cold-snapshot/v2"
        let mutated = try JSONSerialization.data(withJSONObject: object)
        XCTAssertThrowsError(try ColdSnapshotManifest.decode(from: mutated)) { error in
            guard case ColdSnapshotManifestError.unsupportedSchema(let tag) = error else {
                return XCTFail("expected unsupportedSchema, got \(error)")
            }
            XCTAssertEqual(tag, "exact-prefix-cold-snapshot/v2")
        }
    }

    func testAbsentSchemaTagRejected() throws {
        let manifest = sampleManifest(payload: Data("x".utf8))
        var object = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: manifest.encode()) as? [String: Any])
        object.removeValue(forKey: "schema")
        let mutated = try JSONSerialization.data(withJSONObject: object)
        XCTAssertThrowsError(try ColdSnapshotManifest.decode(from: mutated)) { error in
            guard case ColdSnapshotManifestError.unsupportedSchema = error else {
                return XCTFail("expected unsupportedSchema for absent tag, got \(error)")
            }
        }
    }

    func testMalformedJSONRejected() {
        let garbage = Data("not json at all".utf8)
        XCTAssertThrowsError(try ColdSnapshotManifest.decode(from: garbage)) { error in
            guard case ColdSnapshotManifestError.malformedJSON = error else {
                return XCTFail("expected malformedJSON, got \(error)")
            }
        }
    }

    func testDigestMatchesReusablePayload() throws {
        let payload = Data("kv-bytes-here".utf8)
        let manifest = sampleManifest(payload: payload)
        // Same bytes -> verification passes.
        XCTAssertNoThrow(try manifest.verifyDigest(payload: payload))
    }

    func testTamperedPayloadFailsDigest() throws {
        let payload = Data("kv-bytes-here".utf8)
        let manifest = sampleManifest(payload: payload)
        let tampered = Data("kv-bytes-here!".utf8)
        XCTAssertThrowsError(try manifest.verifyDigest(payload: tampered)) { error in
            guard case ColdSnapshotManifestError.digestMismatch(let expected, let actual) = error else {
                return XCTFail("expected digestMismatch, got \(error)")
            }
            XCTAssertEqual(expected, sha256Hex(payload))
            XCTAssertEqual(actual, sha256Hex(tampered))
        }
    }

    func testMakeComputesLowercaseSHA256() {
        let payload = Data("kv-bytes-here".utf8)
        let manifest = sampleManifest(payload: payload)
        XCTAssertEqual(manifest.payloadDigest, sha256Hex(payload))
    }

    func testInvalidBlockSizeRejectedOnDecode() throws {
        let manifest = sampleManifest(payload: Data("x".utf8))
        var object = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: manifest.encode()) as? [String: Any])
        object["blockSize"] = 0
        let mutated = try JSONSerialization.data(withJSONObject: object)
        XCTAssertThrowsError(try ColdSnapshotManifest.decode(from: mutated)) { error in
            guard case ColdSnapshotManifestError.invalidField = error else {
                return XCTFail("expected invalidField, got \(error)")
            }
        }
    }

    func testNonHexDigestRejectedOnDecode() throws {
        let manifest = sampleManifest(payload: Data("x".utf8))
        var object = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: manifest.encode()) as? [String: Any])
        object["payloadDigest"] = "not-a-valid-sha256"
        let mutated = try JSONSerialization.data(withJSONObject: object)
        XCTAssertThrowsError(try ColdSnapshotManifest.decode(from: mutated)) { error in
            guard case ColdSnapshotManifestError.invalidField = error else {
                return XCTFail("expected invalidField for bad digest, got \(error)")
            }
        }
    }

    func testKVLayoutBytesPerTokenDerivation() {
        // 2 (K+V) * 28 layers * 8 kv heads * 128 head dim * 2 bytes = 114_688 bytes/token.
        XCTAssertEqual(sampleLayout().bytesPerToken, 114_688)
    }
}
