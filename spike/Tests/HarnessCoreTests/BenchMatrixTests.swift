import XCTest
@testable import HarnessCore

final class BenchMatrixTests: XCTestCase {
  func testDropsWarmupAndAverages() {
    let agg = aggregateRates([nil, 100.0, 102.0, 104.0]) // index 0 = warmup, dropped; nil = skipped
    XCTAssertEqual(agg.mean, 102.0, accuracy: 1e-9); XCTAssertEqual(agg.runs, 3)
  }
  func testDropsWarmupEvenWhenItIsARealValue() {
    // Index 0 must be dropped unconditionally as the warmup run, not just when it happens to be nil.
    let agg = aggregateRates([99.0, 100.0, 102.0, 104.0])
    XCTAssertEqual(agg.mean, 102.0, accuracy: 1e-9); XCTAssertEqual(agg.runs, 3)
  }

  func testBenchRowCSVIncludesHardwareColumn() {
    // Spec §6.3's mandated durable-evidence dimensions: label, mode, concurrency, model, hardware.
    // label/mode/concurrency/model were already present; hardware was the gap (Task 5).
    let row = BenchRow(
      label: "harness", workload: .decode, mode: .none, model: "Qwen3-32B-4bit",
      decodeTokS: 42.5, ttftMs: 120.3, quant: "int4", concurrency: 1, hardware: "Apple M3 Ultra")
    XCTAssertTrue(BenchRow.csvHeader.hasPrefix(
      "label,workload,mode,model,decode_tok_s,ttft_ms,quant,concurrency,hardware"))
    XCTAssertTrue(row.csvLine.contains(",Apple M3 Ultra,"))
    XCTAssertEqual(
      row.csvLine,
      "harness,decode,none,Qwen3-32B-4bit,42.5,120.3,int4,1,Apple M3 Ultra,,,,,,,,,,")
  }

  func testBenchRowCSVAppendsDirectPrefillMetricsWithoutReorderingExistingColumns() {
    let row = BenchRow(
      label: "harness", workload: .decode, mode: .none, model: "Qwen3-32B-4bit",
      decodeTokS: 42.5, ttftMs: 120.3, quant: "int4", concurrency: 1,
      hardware: "Apple M3 Ultra", prefillTokS: 912.4, prefillMs: 48.2,
      promptTokensMin: 43, promptTokensMax: 44,
      kvQuantTier: "kvtuner-g128-b3.046875",
      matrixID: "kvarn-qwen3-32b-v1",
      cellID: "kvtuner-g128-b3.046875",
      workloadNonce: "kvarn-frontier-20260718",
      kvtunerScheduleSHA256: String(repeating: "a", count: 64),
      kvtunerBundleSHA256: String(repeating: "b", count: 64))

    XCTAssertEqual(
      BenchRow.csvHeader,
      "label,workload,mode,model,decode_tok_s,ttft_ms,quant,concurrency,hardware,prefill_tok_s,prefill_ms,prompt_tokens_min,prompt_tokens_max,kv_quant_tier,matrix_id,cell_id,workload_nonce,kvtuner_schedule_sha256,kvtuner_bundle_sha256")
    XCTAssertEqual(
      row.csvLine,
      "harness,decode,none,Qwen3-32B-4bit,42.5,120.3,int4,1,Apple M3 Ultra,912.4,48.2,43,44,kvtuner-g128-b3.046875,kvarn-qwen3-32b-v1,kvtuner-g128-b3.046875,kvarn-frontier-20260718,aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
  }

  func testPrefillThroughputUsesTheDirectTimedSpan() throws {
    XCTAssertEqual(
      try XCTUnwrap(prefillTokensPerSecond(promptTokens: 512, durationSeconds: 0.25)),
      2_048,
      accuracy: 1e-9)
    XCTAssertNil(prefillTokensPerSecond(promptTokens: 0, durationSeconds: 0.25))
    XCTAssertNil(prefillTokensPerSecond(promptTokens: 512, durationSeconds: 0))
    XCTAssertNil(prefillTokensPerSecond(promptTokens: 512, durationSeconds: -.infinity))
  }

  func testBenchWorkloadIdentityPinsEverySaltedPrompt() throws {
    let workload = try BenchWorkloadIdentity(
      basePrompt: defaultBenchPrompt,
      nonce: "kvarn-frontier-20260718",
      iterations: 4)

    XCTAssertEqual(workload.prompts.count, 4)
    XCTAssertEqual(
      workload.prompt(run: 2),
      "\(defaultBenchPrompt) [run=2 nonce=kvarn-frontier-20260718]")
    XCTAssertEqual(workload.prompts[2], workload.prompt(run: 2))
    XCTAssertEqual(Set(workload.prompts).count, workload.prompts.count)
  }

  func testDefaultBenchWorkloadRequestsEnoughOutputToCrossCompressedTile() {
    XCTAssertEqual(
      defaultBenchPrompt,
      "Explain in at least 250 words how continuous batching improves LLM serving throughput. Cover request scheduling, chunked prefill, decode interleaving, fairness, memory pressure, and cancellation.")
  }

  func testBenchWorkloadIdentityRejectsInvalidNonceAndIterationCount() {
    XCTAssertThrowsError(try BenchWorkloadIdentity(
      basePrompt: defaultBenchPrompt, nonce: "contains space", iterations: 4))
    XCTAssertThrowsError(try BenchWorkloadIdentity(
      basePrompt: defaultBenchPrompt, nonce: "frontier", iterations: 0))
  }

  func testBenchMemoryEvidencePreservesRawSamplesAndRecomputableHighWater() throws {
    let first = try BenchRunMemoryEvidence(samples: [
      ServiceMemorySample(
        timestamp: 10, physicalFootprintBytes: 20_000,
        mlxActiveBytes: 12_000, mlxCacheBytes: 1_000, mlxPeakBytes: 0),
      ServiceMemorySample(
        timestamp: 11, physicalFootprintBytes: 24_000,
        mlxActiveBytes: 14_000, mlxCacheBytes: 2_000, mlxPeakBytes: 18_000),
    ])
    let second = try BenchRunMemoryEvidence(samples: [
      ServiceMemorySample(
        timestamp: 20, physicalFootprintBytes: 21_000,
        mlxActiveBytes: 13_000, mlxCacheBytes: 1_500, mlxPeakBytes: 0),
      ServiceMemorySample(
        timestamp: 21, physicalFootprintBytes: 26_000,
        mlxActiveBytes: 15_000, mlxCacheBytes: 2_500, mlxPeakBytes: 19_000),
    ])
    let aggregate = try BenchMemoryAggregate(runs: [first, second])

    XCTAssertEqual(first.samples.count, 2)
    XCTAssertEqual(first.summary.maxSampledFootprintBytes, 24_000)
    XCTAssertEqual(first.summary.maxMLXPeakBytes, 18_000)
    XCTAssertEqual(aggregate.measuredRuns, 2)
    XCTAssertEqual(aggregate.maxSampledPhysicalFootprintBytes, 26_000)
    XCTAssertEqual(aggregate.maxMLXActiveBytes, 15_000)
    XCTAssertEqual(aggregate.maxMLXCacheBytes, 2_500)
    XCTAssertEqual(aggregate.maxMLXPeakBytes, 19_000)
    XCTAssertEqual(
      try JSONDecoder().decode(
        BenchRunMemoryEvidence.self,
        from: JSONEncoder().encode(first)),
      first)
  }

  func testBenchMemoryEvidenceFailsClosedForMissingOrMalformedSamples() throws {
    let sample = ServiceMemorySample(
      timestamp: 10, physicalFootprintBytes: 20_000,
      mlxActiveBytes: 12_000, mlxCacheBytes: 1_000, mlxPeakBytes: 0)

    XCTAssertThrowsError(try BenchRunMemoryEvidence(samples: [sample]))
    XCTAssertThrowsError(try BenchMemoryAggregate(runs: []))
  }

  func testServiceWorkloadIdentityPinsTheSamePromptAcrossPolicyProcesses() throws {
    let batchIdentity = try ServiceWorkloadIdentity(nonce: "frontier-20260714")
    let soloIdentity = try ServiceWorkloadIdentity(nonce: "frontier-20260714")

    XCTAssertEqual(
      batchIdentity.prompt(basePrompt: "serve", run: 2, request: 3),
      soloIdentity.prompt(basePrompt: "serve", run: 2, request: 3))
    XCTAssertEqual(
      batchIdentity.prompt(basePrompt: "serve", run: 2, request: 3),
      "serve [run=2 nonce=frontier-20260714] [request=3]")
    XCTAssertNotEqual(
      batchIdentity.prompt(basePrompt: "serve", run: 1, request: 3),
      batchIdentity.prompt(basePrompt: "serve", run: 2, request: 3))
    XCTAssertNotEqual(
      batchIdentity.prompt(basePrompt: "serve", run: 2, request: 2),
      batchIdentity.prompt(basePrompt: "serve", run: 2, request: 3))
  }

  func testServiceWorkloadIdentityRejectsUnstableOrPromptChangingNonceValues() {
    for nonce in [
      "", "   ", "contains space", "contains,comma", "-looks-like-a-flag",
      String(repeating: "x", count: 65),
    ] {
      XCTAssertThrowsError(try ServiceWorkloadIdentity(nonce: nonce)) {
        XCTAssertEqual($0 as? ServiceWorkloadIdentityError, .invalidNonce)
      }
    }
  }

  func testCLIFlagsFailClosedWhenAValueIsMissingOrFollowedByAnotherFlag() throws {
    let trailing = CLIFlags(["--workload-nonce"])
    XCTAssertThrowsError(
      try trailing.strictString("workload-nonce", default: "generated")) {
        XCTAssertEqual($0 as? FlagValueError, .missingValue(key: "workload-nonce"))
      }

    let followedByFlag = CLIFlags([
      "--workload-nonce", "--csv", "frontier.csv",
    ])
    XCTAssertThrowsError(
      try followedByFlag.strictString("workload-nonce", default: "generated")) {
        XCTAssertEqual($0 as? FlagValueError, .missingValue(key: "workload-nonce"))
      }
    XCTAssertEqual(followedByFlag.string("csv"), "frontier.csv")

    let explicit = CLIFlags(["--workload-nonce", "shared-identity"])
    XCTAssertEqual(
      try explicit.strictString("workload-nonce", default: "generated"),
      "shared-identity")
  }
}
