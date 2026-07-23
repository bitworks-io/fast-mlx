import Foundation
import XCTest
@testable import HarnessCore

final class SealedKLReferenceDriverTests: XCTestCase {
    func testBundleLoadsAndDriverReplaysExactGenerationFullAndSampledRows() async throws {
        let fullRows: [[Float]] = [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
        ]
        let sampledRows: [[Float]] = [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
        ]
        let fullBlob = blob(fullRows)
        let sampledBlob = blob(sampledRows)
        let manifest = try manifest(entries: [
            entry(
                id: "short-1", prompt: [1, 2], continuation: [3, 4],
                positions: nil, file: "short-1.f32", rows: fullRows,
                blob: fullBlob),
            entry(
                id: "long-1", prompt: [8], continuation: [9, 10, 11, 12],
                positions: [1, 3], file: "long-1.f32", rows: sampledRows,
                blob: sampledBlob),
        ])
        let manifestData = try SealedKLReferenceBundle.canonicalManifestData(manifest)
        let bundle = try SealedKLReferenceBundle(
            manifestData: manifestData,
            blobs: ["short-1.f32": fullBlob, "long-1.f32": sampledBlob],
            expectedBinding: expectedBinding(),
            expectedCorpus: "corpus-a")

        XCTAssertEqual(bundle.manifestSHA256, SealedKLReferenceBundle.sha256Hex(manifestData))
        XCTAssertEqual(bundle.referenceVersions, ["reference-v1"])
        XCTAssertEqual(
            bundle.referenceRuntimeVersions,
            SealedKLReferenceRuntimeVersions(mlx: "mlx-0.31.6", mlxLM: "mlx-lm-local"))
        let driver = SealedKLReferenceDriver(bundle: bundle)

        let generated = try await driver.generate(prompt: [1, 2], config: .greedy(maxTokens: 4))
        XCTAssertEqual(generated.tokens, [3, 4])
        let replayedFullRows = try await driver.logprobs(
            prompt: [1, 2],
            forcedContinuation: [3, 4],
            config: .greedy(maxTokens: 4))
        XCTAssertEqual(replayedFullRows, fullRows)
        let replayedSampledRows = try await driver.logprobs(
            prompt: [8],
            forcedContinuation: [9, 10, 11, 12],
            atPositions: [1, 3],
            config: .greedy(maxTokens: 4))
        XCTAssertEqual(replayedSampledRows, sampledRows)
    }

    func testValidationFailsClosedForWrongBindingCorruptionAndMalformedEntries() throws {
        let rows: [[Float]] = [[0.0, 1.0], [2.0, 3.0]]
        let data = blob(rows)
        let goodEntry = entry(
            id: "entry-1", prompt: [1], continuation: [2, 3],
            positions: nil, file: "entry-1.f32", rows: rows, blob: data)
        let goodManifestData = try SealedKLReferenceBundle.canonicalManifestData(
            manifest(entries: [goodEntry]))

        XCTAssertThrowsError(try SealedKLReferenceBundle(
            manifestData: goodManifestData,
            blobs: ["entry-1.f32": data],
            expectedBinding: expectedBinding(model: "wrong-model"),
            expectedCorpus: "corpus-a"))

        var corrupted = data
        corrupted[0] ^= 0xff
        XCTAssertThrowsError(try SealedKLReferenceBundle(
            manifestData: goodManifestData,
            blobs: ["entry-1.f32": corrupted],
            expectedBinding: expectedBinding(),
            expectedCorpus: "corpus-a"))

        XCTAssertThrowsError(try SealedKLReferenceBundle(
            manifestData: goodManifestData,
            blobs: ["entry-1.f32": data, "extra.f32": data],
            expectedBinding: expectedBinding(),
            expectedCorpus: "corpus-a"))

        XCTAssertThrowsError(try bundle(entries: [
            entry(id: "dup", file: "a.f32", rows: rows, blob: data),
            entry(id: "dup", file: "b.f32", rows: rows, blob: data),
        ], blobs: ["a.f32": data, "b.f32": data]))

        XCTAssertThrowsError(try bundle(entries: [
            entry(id: "a", file: "../a.f32", rows: rows, blob: data),
        ], blobs: ["../a.f32": data]))

        XCTAssertThrowsError(try bundle(entries: [
            entry(id: "a", prompt: [1], continuation: [2, 3], positions: nil,
                  file: "a.f32", rowCount: 1, vocabSize: 2, byteCount: 8,
                  sha: SealedKLReferenceBundle.sha256Hex(data)),
        ], blobs: ["a.f32": data]))

        let nonFinite = blob([[0.0, .nan]])
        XCTAssertThrowsError(try bundle(entries: [
            entry(id: "a", prompt: [1], continuation: [2], positions: nil,
                  file: "a.f32", rows: [[0.0, .nan]], blob: nonFinite),
        ], blobs: ["a.f32": nonFinite]))

        XCTAssertThrowsError(try bundle(entries: [
            entry(id: "a", prompt: [1], continuation: [2, 3, 4],
                  positions: [2, 2], file: "a.f32", rows: rows, blob: data),
        ], blobs: ["a.f32": data]))

        XCTAssertThrowsError(try bundle(entries: [
            entry(
                id: "negative-token",
                prompt: [-1],
                continuation: [2, 3],
                positions: nil,
                file: "negative-token.f32",
                rows: rows,
                blob: data),
        ], blobs: ["negative-token.f32": data]))

        let tooLongRows: [[Float]] = Array(
            repeating: [0.0, 1.0],
            count: 5)
        let tooLongBlob = blob(tooLongRows)
        XCTAssertThrowsError(try bundle(entries: [
            entry(
                id: "too-long-generation",
                prompt: [1],
                continuation: [2, 3, 4, 5, 6],
                positions: nil,
                file: "too-long-generation.f32",
                rows: tooLongRows,
                blob: tooLongBlob),
        ], blobs: ["too-long-generation.f32": tooLongBlob]))

        let sampledRows: [[Float]] = [[0.0, 1.0], [2.0, 3.0]]
        let sampledBlob = blob(sampledRows)
        XCTAssertNoThrow(try bundle(entries: [
            entry(
                id: "long-a",
                prompt: [1],
                continuation: [2, 3, 4, 5, 6],
                positions: [0, 4],
                file: "long-a.f32",
                rows: sampledRows,
                blob: sampledBlob),
            entry(
                id: "long-b",
                prompt: [1],
                continuation: [7, 8, 9, 10, 11],
                positions: [0, 4],
                file: "long-b.f32",
                rows: sampledRows,
                blob: sampledBlob),
        ], blobs: [
            "long-a.f32": sampledBlob,
            "long-b.f32": sampledBlob,
        ]))
    }

    func testDriverFailsClosedForWrongConfigPromptContinuationPositionsAndFreeRunning() async throws {
        let rows: [[Float]] = [[0.0, 1.0], [2.0, 3.0]]
        let data = blob(rows)
        let driver = SealedKLReferenceDriver(bundle: try bundle(entries: [
            entry(
                id: "entry-1", prompt: [1], continuation: [2, 3],
                positions: nil, file: "entry-1.f32", rows: rows, blob: data),
        ], blobs: ["entry-1.f32": data]))

        await XCTAssertThrowsErrorAsync(try await driver.generate(
            prompt: [1], config: .greedy(maxTokens: 3)))
        await XCTAssertThrowsErrorAsync(try await driver.generate(
            prompt: [9], config: .greedy(maxTokens: 4)))
        await XCTAssertThrowsErrorAsync(try await driver.logprobs(
            prompt: [1], config: .greedy(maxTokens: 4)))
        await XCTAssertThrowsErrorAsync(try await driver.logprobs(
            prompt: [1], forcedContinuation: [2, 9], config: .greedy(maxTokens: 4)))
        await XCTAssertThrowsErrorAsync(try await driver.logprobs(
            prompt: [1], forcedContinuation: [2, 3],
            atPositions: [0], config: .greedy(maxTokens: 4)))
    }

    func testSealedReplayPreservesTeacherForcedKLTop1AndPerplexityInputs() async throws {
        let prompt = [1]
        let continuation = [2, 3]
        let referenceRows: [[Float]] = [
            [0.0, 2.0, 1.0, -1.0],
            [1.0, 0.0, -1.0, 2.0],
        ]
        let candidateRows: [[Float]] = [
            [0.0, 1.5, 1.0, -1.0],
            [1.0, 0.0, -0.5, 1.5],
        ]
        let referenceBlob = blob(referenceRows)
        let sealed = SealedKLReferenceDriver(bundle: try bundle(entries: [
            entry(
                id: "entry-1",
                prompt: prompt,
                continuation: continuation,
                positions: nil,
                file: "entry-1.f32",
                rows: referenceRows,
                blob: referenceBlob),
        ], blobs: ["entry-1.f32": referenceBlob]))
        let live = ScriptedDriver(
            tokens: continuation,
            logprobs: referenceRows,
            forcedLogprobs: referenceRows)
        let candidate = ScriptedDriver(
            tokens: [9, 9],
            logprobs: candidateRows,
            forcedLogprobs: candidateRows)
        let config = RunConfig.greedy(maxTokens: 4)

        let liveScores = try await teacherForcedScores(
            driver: candidate,
            reference: live,
            prompt: prompt,
            config: config)
        let sealedScores = try await teacherForcedScores(
            driver: candidate,
            reference: sealed,
            prompt: prompt,
            config: config)

        XCTAssertEqual(sealedScores.continuation, liveScores.continuation)
        XCTAssertEqual(sealedScores.candidateRows, liveScores.candidateRows)
        XCTAssertEqual(sealedScores.referenceRows, liveScores.referenceRows)
        XCTAssertEqual(
            perPositionKLs(
                reference: sealedScores.referenceRows,
                candidate: sealedScores.candidateRows),
            perPositionKLs(
                reference: liveScores.referenceRows,
                candidate: liveScores.candidateRows))
        XCTAssertEqual(
            try teacherForcedTop1Agreement(
                candidate: sealedScores.candidateRows,
                reference: sealedScores.referenceRows),
            try teacherForcedTop1Agreement(
                candidate: liveScores.candidateRows,
                reference: liveScores.referenceRows))
        XCTAssertEqual(
            meanNLL(
                rows: sealedScores.candidateRows,
                tokens: sealedScores.continuation),
            meanNLL(
                rows: liveScores.candidateRows,
                tokens: liveScores.continuation),
            accuracy: 0)
        XCTAssertEqual(
            meanNLL(
                rows: sealedScores.referenceRows,
                tokens: sealedScores.continuation),
            meanNLL(
                rows: liveScores.referenceRows,
                tokens: liveScores.continuation),
            accuracy: 0)
    }

    private func manifest(
        entries: [SealedKLReferenceEntry],
        model: String = "model-a",
        corpus: String = "corpus-a",
        tokenizer: String = "tokenizer-a",
        harness: String = "harness-a",
        referenceVersion: String = "reference-v1",
        maxTokens: Int = 4,
        sampleSize: Int = 2
    ) throws -> SealedKLReferenceManifest {
        SealedKLReferenceManifest(
            schema: "sealed-kl-reference",
            version: 1,
            identity: expectedBinding(
                model: model,
                tokenizer: tokenizer,
                harness: harness),
            corpus: corpus,
            referenceVersion: referenceVersion,
            maxTokens: maxTokens,
            sampleSize: sampleSize,
            entries: entries)
    }

    private func entry(
        id: String,
        tag: String? = nil,
        prompt: [Int] = [1],
        continuation: [Int] = [2, 3],
        positions: [Int]? = nil,
        file: String,
        rows: [[Float]],
        blob: Data
    ) -> SealedKLReferenceEntry {
        entry(
            id: id,
            tag: tag,
            prompt: prompt,
            continuation: continuation,
            positions: positions,
            file: file,
            rowCount: rows.count,
            vocabSize: rows.first?.count ?? 0,
            byteCount: blob.count,
            sha: SealedKLReferenceBundle.sha256Hex(blob))
    }

    private func entry(
        id: String,
        tag: String? = nil,
        prompt: [Int] = [1],
        continuation: [Int] = [2, 3],
        positions: [Int]? = nil,
        file: String,
        rowCount: Int,
        vocabSize: Int,
        byteCount: Int,
        sha: String
    ) -> SealedKLReferenceEntry {
        SealedKLReferenceEntry(
            id: id,
            tag: tag,
            promptTokenIDs: prompt,
            continuationTokenIDs: continuation,
            samplePositions: positions,
            logitsFile: file,
            logitsSHA256: sha,
            rowCount: rowCount,
            vocabSize: vocabSize,
            byteCount: byteCount)
    }

    private func bundle(
        entries: [SealedKLReferenceEntry],
        blobs: [String: Data]
    ) throws -> SealedKLReferenceBundle {
        let data = try SealedKLReferenceBundle.canonicalManifestData(manifest(entries: entries))
        return try SealedKLReferenceBundle(
            manifestData: data,
            blobs: blobs,
            expectedBinding: expectedBinding(),
            expectedCorpus: "corpus-a")
    }

    private func blob(_ rows: [[Float]]) -> Data {
        var data = Data()
        for row in rows {
            for value in row {
                var littleEndian = value.bitPattern.littleEndian
                withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
            }
        }
        return data
    }

    private func hex(_ byte: String) -> String {
        String(repeating: byte, count: 32)
    }

    private func expectedBinding(
        model: String = "model-a",
        tokenizer: String = "tokenizer-a",
        harness: String = "harness-a"
    ) -> SealedKLReferenceBinding {
        SealedKLReferenceBinding(
            model: model,
            harnessGitSHA: String(repeating: "a", count: 40),
            modelConfigSHA256: hex("02"),
            checkpointManifestSHA256: hex("03"),
            checkpointContentSHA256: hex("04"),
            tokenizer: tokenizer,
            tokenizerManifestSHA256: hex("05"),
            corpusContentHash: String(repeating: "6", count: 16),
            corpusRawFileSHA256: hex("07"),
            harness: harness,
            referenceScriptSHA256: hex("08"),
            referenceRuntimeVersions: SealedKLReferenceRuntimeVersions(
                mlx: "mlx-0.31.6",
                mlxLM: "mlx-lm-local"),
            workloadNonce: "workload-1")
    }
}

private func XCTAssertThrowsErrorAsync(
    _ expression: @autoclosure () async throws -> some Any,
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    do {
        _ = try await expression()
        XCTFail("expected error", file: file, line: line)
    } catch {}
}
