import XCTest
@testable import HarnessCore

/// TDD for the serve-path glue that turns a list of on-disk quant candidate directories into a
/// pick: decode each directory's real config.json + summed safetensors bytes, fit-check every
/// candidate against the host, and map the winner back to the directory the serve path should load.
/// Undecodable directories are carried as excluded-with-reason (never fatal), and an all-red set
/// refuses. Pure — exercised entirely off-box with temp fixtures; the live serve only confirms the
/// winning directory loads.
final class QuantCandidateResolverTests: XCTestCase {
    private let gib = 1024 * 1024 * 1024

    private func host(ramGiB: Int, wiredGiB: Int) -> SystemProfile {
        SystemProfile(chip: "test", totalRAMBytes: ramGiB * gib, wiredLimitBytes: wiredGiB * gib, wiredLimitIsMeasured: true)
    }

    /// Uniform-GQA config matching the QuantAutoPicker fixtures (KV@40960 ≈ 5.4 GiB) so headroom, not
    /// geometry, decides the verdict. `bits` nil omits the quantization block (unquantized checkpoint).
    private func config(bits: Int?) -> String {
        let quant = bits.map { ",\n  \"quantization\": { \"group_size\": 64, \"bits\": \($0) }" } ?? ""
        return """
        {
          "model_type": "qwen3",
          "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
          "head_dim": 128, "max_position_embeddings": 40960\(quant)
        }
        """
    }

    /// Write a candidate directory: config.json + one sparse `.safetensors` file whose *logical* size
    /// is `weightsGiB` (truncate sets EOF without allocating blocks, so multi-GiB fixtures cost no
    /// disk). `sumSafetensorsBytes` reads that logical size, exactly as it would a real shard.
    private func makeCandidateDir(_ name: String, bits: Int?, weightsGiB: Double) throws -> URL {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("qcr-\(name)-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(config(bits: bits).utf8).write(to: dir.appendingPathComponent("config.json"))
        let shard = dir.appendingPathComponent("model.safetensors")
        XCTAssertTrue(fm.createFile(atPath: shard.path, contents: nil))
        let fh = try FileHandle(forWritingTo: shard)
        try fh.truncate(atOffset: UInt64(weightsGiB * Double(gib)))
        try fh.close()
        return dir
    }

    /// Write a METADATA-ONLY candidate directory: config.json + a `model.safetensors.index.json`
    /// declaring `declaredGiB` in `metadata.total_size`, but NO `.safetensors` shards — exactly what
    /// `scripts/quant-prefetch.py metadata` materializes (allow_patterns = config + index only). The
    /// decoder must size it from the declared total (conservative) and mark it NOT measured.
    private func makeMetadataOnlyDir(_ name: String, bits: Int?, declaredGiB: Double) throws -> URL {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("qcr-\(name)-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(config(bits: bits).utf8).write(to: dir.appendingPathComponent("config.json"))
        let bytes = Int(declaredGiB * Double(gib))
        let index = "{ \"metadata\": { \"total_size\": \(bytes) }, \"weight_map\": {} }"
        try Data(index.utf8).write(to: dir.appendingPathComponent("model.safetensors.index.json"))
        return dir
    }

    private func rm(_ dirs: [URL]) { for d in dirs { try? FileManager.default.removeItem(at: d) } }

    // MARK: - 8bit red, 4bit green → winner directory is the 4bit dir

    func testPicksFittingDirectory() throws {
        let d8 = try makeCandidateDir("8bit", bits: 8, weightsGiB: 32)
        let d4 = try makeCandidateDir("4bit", bits: 4, weightsGiB: 16)
        defer { rm([d8, d4]) }

        let r = QuantCandidateResolver.resolve(candidateDirectories: [d8, d4], host: host(ramGiB: 40, wiredGiB: 34))
        XCTAssertTrue(r.shouldProceed)
        XCTAssertEqual(r.winnerDirectory, d4)
        XCTAssertEqual(r.winnerParsed?.quantBits, 4)
        // the 8bit candidate was really decoded from disk (measured weights) and found red, not dropped
        let eight = r.pick.evaluations.first { $0.repoID == d8.path }
        XCTAssertEqual(eight?.decision?.color, .red)
        XCTAssertEqual(r.winnerParsed?.weightsAreMeasured, true)
    }

    // MARK: - an undecodable directory is excluded with a reason, does not block a good candidate

    func testUndecodableDirectoryExcluded_notFatal() throws {
        let fm = FileManager.default
        let bad = fm.temporaryDirectory.appendingPathComponent("qcr-bad-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: bad, withIntermediateDirectories: true)  // no config.json
        let good = try makeCandidateDir("4bit", bits: 4, weightsGiB: 16)
        defer { rm([bad, good]) }

        let r = QuantCandidateResolver.resolve(candidateDirectories: [bad, good], host: host(ramGiB: 128, wiredGiB: 115))
        XCTAssertTrue(r.shouldProceed)
        XCTAssertEqual(r.winnerDirectory, good)
        let ex = r.pick.evaluations.first { $0.repoID == bad.path }
        XCTAssertNotNil(ex)
        XCTAssertNil(ex?.decision, "excluded before the fit-check ran")
        XCTAssertFalse(ex?.eligible ?? true)
        XCTAssertNotNil(ex?.exclusionReason)
    }

    // MARK: - nothing fits → refuse, no winner directory

    func testAllRed_refuses() throws {
        let d8 = try makeCandidateDir("8bit", bits: 8, weightsGiB: 18)
        let d4 = try makeCandidateDir("4bit", bits: 4, weightsGiB: 9)
        defer { rm([d8, d4]) }

        let r = QuantCandidateResolver.resolve(candidateDirectories: [d8, d4], host: host(ramGiB: 8, wiredGiB: 6))
        XCTAssertFalse(r.shouldProceed)
        XCTAssertNil(r.winnerDirectory)
        XCTAssertNil(r.winnerParsed)
        XCTAssertTrue(r.summaryLines().joined(separator: "\n").lowercased().contains("no quant"))
    }

    // MARK: - a metadata-only prefetch dir (config + index, no shards) fit-checks with declared provenance

    /// The `scripts/quant-prefetch.py metadata` flow (fit-checked-serve #2 sourcing half) downloads
    /// ONLY config.json + `model.safetensors.index.json` for each candidate — no weights — so the pick
    /// can size candidates cheaply before full-downloading the winner alone. Such a dir must (a) be a
    /// real, fit-checkable candidate (sized from the index's declared total, not thrown out as
    /// weightsUnknown), and (b) surface as (declared) / `weights_measured=false`, never conflated with
    /// a measured on-disk size. Locks decision D2 (docs/task-inbox/2026-08-18-quant-auto-pick-policy.md).
    func testMetadataOnlyDirectory_declaredProvenanceWins() throws {
        let metaOnly = try makeMetadataOnlyDir("4bit-metaonly", bits: 4, declaredGiB: 16)
        defer { rm([metaOnly]) }

        let r = QuantCandidateResolver.resolve(candidateDirectories: [metaOnly], host: host(ramGiB: 128, wiredGiB: 115))
        XCTAssertTrue(r.shouldProceed, "a metadata-only prefetch dir is a real candidate, not excluded")
        XCTAssertEqual(r.winnerDirectory, metaOnly)
        XCTAssertEqual(r.winnerParsed?.weightsAreMeasured, false, "declared size, not measured on disk")
        XCTAssertEqual(r.winnerParsed?.weightsAreDeclared, true, "sized from the index's declared total")
        let d = try XCTUnwrap(r.pick.winnerDecision)
        XCTAssertTrue(d.machineReadableFields().contains("weights_measured=false"))
        XCTAssertTrue(d.summaryLines()[1].contains("(declared)"),
                      "the fit announce must read (declared), not (measured) or (estimated)")
    }

    // MARK: - a config-only "phantom" directory (no shards) never wins over a real, fitting candidate

    /// A directory with config.json but NO safetensors shards and no index (an interrupted or
    /// metadata-only prefetch) used to decode with weightsBytes4bitEstimate == 0 and look GREEN on
    /// any host. It must now throw `.weightsUnknown` in the decoder, which the resolver carries as an
    /// excluded-with-reason candidate — never the winner — while a real, fitting candidate still wins.
    func testConfigOnlyPhantomCandidateExcluded_realWins() throws {
        let real = try makeCandidateDir("4bit", bits: 4, weightsGiB: 16)

        let fm = FileManager.default
        let phantom = fm.temporaryDirectory.appendingPathComponent("qcr-phantom-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: phantom, withIntermediateDirectories: true)
        try Data(config(bits: 4).utf8).write(to: phantom.appendingPathComponent("config.json"))
        // No shards, no index.safetensors.index.json — nothing to honestly size weights from.

        defer { rm([real, phantom]) }

        let r = QuantCandidateResolver.resolve(candidateDirectories: [phantom, real], host: host(ramGiB: 40, wiredGiB: 34))
        XCTAssertTrue(r.shouldProceed)
        XCTAssertEqual(r.winnerDirectory, real)
        let excluded = r.pick.evaluations.first { $0.repoID == phantom.path }
        XCTAssertNotNil(excluded)
        XCTAssertNil(excluded?.decision, "excluded before the fit-check ran — never classified GREEN on phantom zero weights")
        XCTAssertFalse(excluded?.eligible ?? true)
        XCTAssertNotNil(excluded?.exclusionReason)
        XCTAssertTrue(excluded?.exclusionReason?.lowercased().contains("no weights found") ?? false)
    }
}
