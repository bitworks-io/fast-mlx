import Foundation
import CryptoKit
import XCTest

final class LoadedBenchQualificationRunnerTests: XCTestCase {
  func testRunnerExecutesDeclaredBlocksAndWritesAuthenticatedReceipts() throws {
    let fixture = try makeFixture()
    let result = try runRunner(fixture: fixture)

    XCTAssertEqual(result.status, 0, result.output)
    XCTAssertEqual(
      try String(contentsOf: fixture.output.appendingPathComponent(
        "runner.status"), encoding: .utf8),
      "COMPLETE\n")
    let receipts = try FileManager.default.subpathsOfDirectory(
      atPath: fixture.output.appendingPathComponent("runs").path)
      .filter { $0.hasSuffix("/runner-receipt.json") }
    XCTAssertEqual(receipts.count, 6)
    let launches = try String(
      contentsOf: fixture.launches, encoding: .utf8)
      .split(separator: "\n").map(String.init)
    XCTAssertEqual(launches, [
      "0:0:fp16", "0:1:affine-direct",
      "1:0:affine-direct", "1:1:fp16",
      "2:0:fp16", "2:1:affine-direct",
    ])
    let completion = try JSONSerialization.jsonObject(
      with: Data(contentsOf: fixture.output.appendingPathComponent(
        "runner.completion.json"))) as? [String: Any]
    XCTAssertEqual(completion?["completedRows"] as? Int, 6)
    XCTAssertEqual(completion?["blockCount"] as? Int, 3)
    XCTAssertEqual(completion?["cellCount"] as? Int, 2)
    XCTAssertTrue(
      (completion?["runnerScriptSHA256"] as? String)?.count == 64)
    let receiptSet = fixture.output.appendingPathComponent(
      "runner.receipts.sha256")
    let receiptSetData = try Data(contentsOf: receiptSet)
    XCTAssertEqual(
      completion?["receiptSetSHA256"] as? String,
      SHA256.hash(data: receiptSetData).map {
        String(format: "%02x", $0)
      }.joined())
    XCTAssertEqual(
      String(decoding: receiptSetData, as: UTF8.self)
        .split(separator: "\n").count,
      6)
    XCTAssertTrue(FileManager.default.fileExists(
      atPath: fixture.output.appendingPathComponent(
        "runner.manifest.json").path))
    XCTAssertFalse(FileManager.default.fileExists(
      atPath: fixture.output.appendingPathComponent(
        "runner.watchdog.json").path))
  }

  func testRunnerLaunchesHarnessFromAuthenticatedStampDirectory() throws {
    let fixture = try makeFixture(readHarnessSHAFromCurrentDirectory: true)
    let unrelatedDirectory = fixture.root.appendingPathComponent(
      "unrelated-launch-directory", isDirectory: true)
    try FileManager.default.createDirectory(
      at: unrelatedDirectory, withIntermediateDirectories: true)

    let result = try runRunner(
      fixture: fixture, workingDirectory: unrelatedDirectory)

    XCTAssertEqual(result.status, 0, result.output)
    let firstEvidence = try XCTUnwrap(
      FileManager.default.subpathsOfDirectory(
        atPath: fixture.output.appendingPathComponent("runs").path)
        .filter { $0.hasSuffix("/bench.jsonl") }
        .sorted()
        .first)
    let record = try XCTUnwrap(
      JSONSerialization.jsonObject(with: Data(contentsOf:
        fixture.output.appendingPathComponent("runs/\(firstEvidence)")))
        as? [String: Any])
    let provenance = try XCTUnwrap(record["provenance"] as? [String: Any])
    XCTAssertEqual(
      provenance["harnessGitSHA"] as? String,
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
  }

  func testRunnerIgnoresHarnessSHAEnvironmentOverride() throws {
    let fixture = try makeFixture(
      readHarnessSHAFromCurrentDirectory: true,
      spoofHarnessSHAEnvironment: true)

    let result = try runRunner(fixture: fixture)

    XCTAssertEqual(result.status, 0, result.output)
  }

  func testRunnerScrubsGitEnvironmentBeforeLaunchingHarness() throws {
    let fixture = try makeFixture(
      readHarnessSHAFromCurrentDirectory: true,
      spoofGitEnvironment: true)

    let result = try runRunner(fixture: fixture)

    XCTAssertEqual(result.status, 0, result.output)
  }

  func testRunnerRejectsStampMutationBeforeLaunchingTheNextRow() throws {
    let fixture = try makeFixture(
      readHarnessSHAFromCurrentDirectory: true,
      mutateHarnessStampAfterFirstRun: true)

    let result = try runRunner(fixture: fixture)

    XCTAssertNotEqual(result.status, 0)
    XCTAssertTrue(result.output.contains("source stamp changed"))
    XCTAssertEqual(
      try String(contentsOf: fixture.output.appendingPathComponent(
        "runner.status"), encoding: .utf8),
      "INPUT_CHANGED\n")
    let launches = try String(
      contentsOf: fixture.launches, encoding: .utf8)
      .split(separator: "\n")
    XCTAssertEqual(launches.count, 1)
  }

  func testRunnerPreservesCallerRelativeModelAndSchedulePaths() throws {
    let fixture = try makeFixture(
      includeKVTuner: true,
      runtimePathsRelativeToCaller: true,
      requireResolvedRuntimePaths: true,
      harnessStampInSubdirectory: true)

    let result = try runRunner(
      fixture: fixture, workingDirectory: fixture.root)

    XCTAssertEqual(result.status, 0, result.output)
  }

  func testRunnerRejectsAStampNameTheHarnessCannotReadBeforeLaunching()
    throws
  {
    let fixture = try makeFixture(
      readHarnessSHAFromCurrentDirectory: true,
      harnessStampFilename: "source-sha.txt")

    let result = try runRunner(fixture: fixture)

    XCTAssertNotEqual(result.status, 0)
    XCTAssertTrue(result.output.contains(".harness-sha"))
    XCTAssertFalse(FileManager.default.fileExists(
      atPath: fixture.launches.path))
  }

  func testRunnerRefusesOutputReuseAndDoesNotLaunchADuplicate() throws {
    let fixture = try makeFixture()
    XCTAssertEqual(try runRunner(fixture: fixture).status, 0)
    let firstLaunches = try String(
      contentsOf: fixture.launches, encoding: .utf8)

    let duplicate = try runRunner(fixture: fixture)

    XCTAssertNotEqual(duplicate.status, 0)
    XCTAssertTrue(duplicate.output.contains("fresh output"))
    XCTAssertEqual(
      try String(contentsOf: fixture.launches, encoding: .utf8),
      firstLaunches)
  }

  func testRunnerRejectsANonPermutationBeforeLaunching() throws {
    let fixture = try makeFixture(blocks: [
      ["fp16", "affine-direct"],
      ["affine-direct", "affine-direct"],
      ["fp16", "affine-direct"],
    ])

    let result = try runRunner(fixture: fixture)

    XCTAssertNotEqual(result.status, 0)
    XCTAssertTrue(result.output.contains("exact cell permutation"))
    XCTAssertFalse(FileManager.default.fileExists(
      atPath: fixture.launches.path))
  }

  func testRunnerRejectsAPositionBiasedPermutationScheduleBeforeLaunching()
    throws
  {
    let fixture = try makeFixture(blocks: [
      ["fp16", "affine-direct"],
      ["fp16", "affine-direct"],
      ["fp16", "affine-direct"],
    ])

    let result = try runRunner(fixture: fixture)

    XCTAssertNotEqual(result.status, 0)
    XCTAssertTrue(result.output.contains("position-balanced"))
    XCTAssertFalse(FileManager.default.fileExists(
      atPath: fixture.launches.path))
  }

  func testRunnerRequiresAnExplicitRouteForEveryCompressedKVCell()
    throws
  {
    let fixture = try makeFixture()
    var manifest = try XCTUnwrap(
      JSONSerialization.jsonObject(
        with: Data(contentsOf: fixture.manifest)) as? [String: Any])
    var cells = try XCTUnwrap(manifest["cells"] as? [[String: Any]])
    cells[1].removeValue(forKey: "attention")
    cells[1].removeValue(forKey: "checkpointContentSHA256")
    manifest["cells"] = cells
    try JSONSerialization.data(
      withJSONObject: manifest, options: [.sortedKeys])
      .write(to: fixture.manifest)

    let result = try runRunner(fixture: fixture)

    XCTAssertNotEqual(result.status, 0)
    XCTAssertTrue(result.output.contains("manifest schema"))
    XCTAssertFalse(FileManager.default.fileExists(
      atPath: fixture.launches.path))
  }

  func testRunnerRequiresAFrozenTokenizerIdentityBeforeLaunching()
    throws
  {
    let fixture = try makeFixture()
    var manifest = try XCTUnwrap(
      JSONSerialization.jsonObject(
        with: Data(contentsOf: fixture.manifest)) as? [String: Any])
    manifest.removeValue(forKey: "modelTokenizerSHA256")
    try JSONSerialization.data(
      withJSONObject: manifest, options: [.sortedKeys])
      .write(to: fixture.manifest)

    let result = try runRunner(fixture: fixture)

    XCTAssertNotEqual(result.status, 0)
    XCTAssertTrue(result.output.contains("manifest schema"))
    XCTAssertFalse(FileManager.default.fileExists(
      atPath: fixture.launches.path))
  }

  func testRunnerRequiresAnExplicitPostWarmupThermalPolicyBeforeLaunching()
    throws
  {
    let fixture = try makeFixture()
    var manifest = try XCTUnwrap(
      JSONSerialization.jsonObject(
        with: Data(contentsOf: fixture.manifest)) as? [String: Any])
    manifest.removeValue(forKey: "postWarmupThermalTarget")
    try JSONSerialization.data(
      withJSONObject: manifest, options: [.sortedKeys])
      .write(to: fixture.manifest)

    let result = try runRunner(fixture: fixture)

    XCTAssertNotEqual(result.status, 0)
    XCTAssertTrue(result.output.contains("manifest schema"))
    XCTAssertFalse(FileManager.default.fileExists(
      atPath: fixture.launches.path))
  }

  func testRunnerPassesTheFrozenPostWarmupThermalPolicyToEveryHarnessRow()
    throws
  {
    let fixture = try makeFixture(requireThermalPolicyArguments: true)

    let result = try runRunner(fixture: fixture)

    XCTAssertEqual(result.status, 0, result.output)
  }

  func testRunnerRejectsARetainedThermalStateOutsideTheFrozenTarget()
    throws
  {
    let fixture = try makeFixture(changeSecondRowThermal: true)

    let result = try runRunner(fixture: fixture)

    XCTAssertNotEqual(result.status, 0)
    XCTAssertTrue(result.output.contains("identity or qualification"))
    XCTAssertEqual(
      try String(contentsOf: fixture.output.appendingPathComponent(
        "runner.status"), encoding: .utf8),
      "INVALID_EVIDENCE\n")
  }

  func testRunnerRejectsPostWarmupAdmissionBeyondTheFrozenTimeout()
    throws
  {
    let fixture = try makeFixture(lateThermalAdmission: true)

    let result = try runRunner(fixture: fixture)

    XCTAssertNotEqual(result.status, 0)
    XCTAssertTrue(result.output.contains("identity or qualification"))
    XCTAssertEqual(
      try String(contentsOf: fixture.output.appendingPathComponent(
        "runner.status"), encoding: .utf8),
      "INVALID_EVIDENCE\n")
  }

  func testRunnerRejectsARequestedDirectRouteWithoutMatchingObservedOperation()
    throws
  {
    let fixture = try makeFixture(mismatchObservedOperation: true)

    let result = try runRunner(fixture: fixture)

    XCTAssertNotEqual(result.status, 0)
    XCTAssertTrue(result.output.contains("identity or qualification"))
    XCTAssertEqual(
      try String(contentsOf: fixture.output.appendingPathComponent(
        "runner.status"), encoding: .utf8),
      "INVALID_EVIDENCE\n")
  }

  func testRunnerRejectsKVTunerEvidenceThatDoesNotMatchTheFrozenBundle()
    throws
  {
    let fixture = try makeFixture(
      includeKVTuner: true, mismatchKVTunerBinding: true)

    let result = try runRunner(fixture: fixture)

    XCTAssertNotEqual(result.status, 0)
    XCTAssertTrue(result.output.contains("identity or qualification"))
    XCTAssertEqual(
      try String(contentsOf: fixture.output.appendingPathComponent(
        "runner.status"), encoding: .utf8),
      "INVALID_EVIDENCE\n")
  }

  func testRunnerAcceptsKVTunerEvidenceMatchingTheFrozenBundle() throws {
    let fixture = try makeFixture(includeKVTuner: true)

    let result = try runRunner(fixture: fixture)

    XCTAssertEqual(result.status, 0, result.output)
  }

  func testRunnerWritesATerminalStatusWhenHarnessOmitsEvidence() throws {
    let fixture = try makeFixture(omitEvidence: true)

    let result = try runRunner(fixture: fixture)

    XCTAssertNotEqual(result.status, 0)
    XCTAssertEqual(
      try String(contentsOf: fixture.output.appendingPathComponent(
        "runner.status"), encoding: .utf8),
      "INVALID_EVIDENCE\n")
  }

  func testRunnerRejectsEvidenceFromTheWrongModelIdentity() throws {
    let fixture = try makeFixture(mismatchModelIdentity: true)

    let result = try runRunner(fixture: fixture)

    XCTAssertNotEqual(result.status, 0)
    XCTAssertTrue(result.output.contains("identity or qualification"))
    XCTAssertEqual(
      try String(contentsOf: fixture.output.appendingPathComponent(
        "runner.status"), encoding: .utf8),
      "INVALID_EVIDENCE\n")
  }

  func testRunnerRejectsEvidenceFromTheWrongTokenizerIdentity() throws {
    let fixture = try makeFixture(mismatchTokenizerIdentity: true)

    let result = try runRunner(fixture: fixture)

    XCTAssertNotEqual(result.status, 0)
    XCTAssertTrue(result.output.contains("identity or qualification"))
    XCTAssertEqual(
      try String(contentsOf: fixture.output.appendingPathComponent(
        "runner.status"), encoding: .utf8),
      "INVALID_EVIDENCE\n")
  }

  func testRunnerRejectsTruncatedCompressedAttentionAdmissionEvidence()
    throws
  {
    let fixture = try makeFixture(truncateAdmissionEvidence: true)

    let result = try runRunner(fixture: fixture)

    XCTAssertNotEqual(result.status, 0)
    XCTAssertEqual(
      try String(contentsOf: fixture.output.appendingPathComponent(
        "runner.status"), encoding: .utf8),
      "INVALID_EVIDENCE\n")
  }

  func testRunnerRejectsTruncatedKVTunerScheduleBindingEvidence() throws {
    let fixture = try makeFixture(
      includeKVTuner: true, truncateKVTunerEvidence: true)

    let result = try runRunner(fixture: fixture)

    XCTAssertNotEqual(result.status, 0)
    XCTAssertEqual(
      try String(contentsOf: fixture.output.appendingPathComponent(
        "runner.status"), encoding: .utf8),
      "INVALID_EVIDENCE\n")
  }

  func testRunnerRejectsAHarnessBinaryOutsideTheFrozenManifest() throws {
    let fixture = try makeFixture(mismatchBinaryIdentity: true)

    let result = try runRunner(fixture: fixture)

    XCTAssertNotEqual(result.status, 0)
    XCTAssertTrue(result.output.contains("binary digest"))
    XCTAssertFalse(FileManager.default.fileExists(
      atPath: fixture.launches.path))
  }

  func testRunnerWatchdogTerminatesAStalledHarnessAndRetainsTheArtifact()
    throws
  {
    let fixture = try makeFixture(stallHarness: true)

    let result = try runRunner(fixture: fixture, watchdogSeconds: "2")

    XCTAssertNotEqual(result.status, 0)
    XCTAssertTrue(result.output.contains("stalled harness"))
    XCTAssertEqual(
      try String(contentsOf: fixture.output.appendingPathComponent(
        "runner.status"), encoding: .utf8),
      "WATCHDOG\n")
    XCTAssertTrue(FileManager.default.fileExists(
      atPath: fixture.output.appendingPathComponent(
        "runner.watchdog.json").path))
  }

  private struct Fixture {
    let root: URL
    let manifest: URL
    let output: URL
    let binary: URL
    let harnessSHA: URL
    let launches: URL
    let changeSecondRowThermal: Bool
    let mismatchObservedOperation: Bool
    let mismatchKVTunerBinding: Bool
    let omitEvidence: Bool
    let mismatchModelIdentity: Bool
    let mismatchTokenizerIdentity: Bool
    let truncateAdmissionEvidence: Bool
    let truncateKVTunerEvidence: Bool
    let stallHarness: Bool
    let readHarnessSHAFromCurrentDirectory: Bool
    let spoofHarnessSHAEnvironment: Bool
    let spoofGitEnvironment: Bool
    let mutateHarnessStampAfterFirstRun: Bool
    let requireResolvedRuntimePaths: Bool
    let requireThermalPolicyArguments: Bool
    let lateThermalAdmission: Bool
    let kvtunerArtifactSHA256: String
    let kvtunerBundleSHA256: String
  }

  private func makeFixture(
    blocks: [[String]] = [
      ["fp16", "affine-direct"],
      ["affine-direct", "fp16"],
      ["fp16", "affine-direct"],
    ],
    changeSecondRowThermal: Bool = false,
    mismatchObservedOperation: Bool = false,
    includeKVTuner: Bool = false,
    mismatchKVTunerBinding: Bool = false,
    omitEvidence: Bool = false,
    mismatchModelIdentity: Bool = false,
    mismatchTokenizerIdentity: Bool = false,
    truncateAdmissionEvidence: Bool = false,
    truncateKVTunerEvidence: Bool = false,
    mismatchBinaryIdentity: Bool = false,
    stallHarness: Bool = false,
    readHarnessSHAFromCurrentDirectory: Bool = false,
    spoofHarnessSHAEnvironment: Bool = false,
    spoofGitEnvironment: Bool = false,
    mutateHarnessStampAfterFirstRun: Bool = false,
    runtimePathsRelativeToCaller: Bool = false,
    requireResolvedRuntimePaths: Bool = false,
    requireThermalPolicyArguments: Bool = false,
    lateThermalAdmission: Bool = false,
    harnessStampInSubdirectory: Bool = false,
    harnessStampFilename: String = ".harness-sha"
  ) throws -> Fixture {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
      "loaded-bench-runner-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(
      at: root, withIntermediateDirectories: true)
    let manifest = root.appendingPathComponent("manifest.json")
    let output = root.appendingPathComponent("output", isDirectory: true)
    let binary = root.appendingPathComponent("fake-harness.sh")
    let harnessStampDirectory = harnessStampInSubdirectory
      ? root.appendingPathComponent("deploy", isDirectory: true)
      : root
    try FileManager.default.createDirectory(
      at: harnessStampDirectory, withIntermediateDirectories: true)
    let harnessSHA = harnessStampDirectory.appendingPathComponent(
      harnessStampFilename)
    let launches = root.appendingPathComponent("launches.txt")
    let model = root.appendingPathComponent("models/test", isDirectory: true)
    try FileManager.default.createDirectory(
      at: model, withIntermediateDirectories: true)
    let schedule = root.appendingPathComponent("qualification-bundle.json")
    let scheduleData = Data("qualification-bundle".utf8)
    try scheduleData.write(to: schedule)
    let kvtunerBundleSHA256 = SHA256.hash(data: scheduleData).map {
      String(format: "%02x", $0)
    }.joined()
    let kvtunerArtifactSHA256 = String(repeating: "c", count: 64)
    try "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n".write(
      to: harnessSHA, atomically: true, encoding: .utf8)

    let defaultCells: [[String: Any]] = [
      ["id": "fp16", "kvQuant": "fp16"],
      [
        "id": "affine-direct",
        "kvQuant": "affine-k4v2-g64",
        "attention": "split-affine-quantized-mm",
        "checkpointContentSHA256": String(repeating: "b", count: 64),
      ],
    ]
    let kvtunerCells: [[String: Any]] = [
      ["id": "fp16", "kvQuant": "fp16"],
      [
        "id": "kvtuner-direct",
        "kvQuant": "kvtuner-g128-b3.046875",
        "attention": "split-affine-quantized-mm",
        "checkpointContentSHA256": String(repeating: "b", count: 64),
        "kvtunerSchedule": runtimePathsRelativeToCaller
          ? schedule.lastPathComponent : schedule.path,
        "kvtunerBundleSHA256": kvtunerBundleSHA256,
        "kvtunerScheduleArtifactSHA256": kvtunerArtifactSHA256,
      ],
    ]
    let selectedBlocks = includeKVTuner ? [
      ["fp16", "kvtuner-direct"],
      ["kvtuner-direct", "fp16"],
      ["fp16", "kvtuner-direct"],
    ] : blocks
    let fakeHarness = """
      #!/bin/bash
      set -euo pipefail
      if [[ "${1:-}" == "validate-bench-qualification" ]]; then
        evidence_path="${3:-}"
        jq -e '
          .subcommand == "bench" and
          .payload.qualification.schemaVersion == 3 and
          .payload.qualification.context.postWarmupThermalPolicy.target ==
            "nominal" and
          .payload.qualification.warmup.before.thermalState == "nominal" and
          .payload.qualification.postWarmupThermalAdmission.snapshot.thermalState ==
            "nominal" and
          (.payload.qualification.context.tokenizerSHA256 |
            type == "string" and test("^[0-9a-f]{64}$")) and
          (if .payload.compressedKVAttention == null then true else
             (.payload.compressedKVAttention.admission |
               has("family") and has("modelType") and has("architecture") and
               has("modelConfigHash") and has("modelConfigSHA256") and
               has("checkpointManifestHash") and
               has("checkpointContentSHA256") and has("tokenizerSHA256") and
               has("layerCount") and has("queryHeadCount") and
               has("kvHeadCount") and has("headDimension") and
               has("maxPositionEmbeddings")) end) and
          (if .payload.kvtunerSchedule == null then true else
             (.payload.kvtunerSchedule |
               has("schemaVersion") and has("scheduleSchemaVersion") and
               has("artifactSHA256") and has("qualificationBundleSHA256") and
               has("matrixID") and has("cellID") and has("modelConfigHash") and
               has("modelConfigSHA256") and has("checkpointManifestHash") and
               has("checkpointContentSHA256") and has("tokenizerSHA256") and
               has("groupSize") and has("layers")) end)
        ' "$evidence_path" >/dev/null
        exit
      fi
      evidence=""; label=""; matrix=""; nonce=""; tier="fp16"
      manifest_sha=""; block=""; position=""; count=""; memory=""
      cache=""; wired=""; prompt_repeat=""; max_tokens=""; attention=""
      checkpoint=""; model=""; schedule=""; expected_tokenizer=""
      thermal_target=""; thermal_timeout=""; thermal_poll=""
      while (($#)); do
        key="$1"; shift
        if [[ "$key" == "bench" ]]; then continue; fi
        value="$1"; shift
        case "$key" in
          --evidence) evidence="$value" ;;
          --label) label="$value" ;;
          --matrix-id) matrix="$value" ;;
          --workload-nonce) nonce="$value" ;;
          --kv-quant) tier="$value" ;;
          --runner-manifest-sha256) manifest_sha="$value" ;;
          --matrix-block-index) block="$value" ;;
          --matrix-run-position) position="$value" ;;
          --matrix-cell-count) count="$value" ;;
          --memory-limit-bytes) memory="$value" ;;
          --cache-limit-bytes) cache="$value" ;;
          --wired-limit-bytes) wired="$value" ;;
          --model-tokenizer-sha256) expected_tokenizer="$value" ;;
          --post-warmup-thermal-target) thermal_target="$value" ;;
          --post-warmup-thermal-timeout-seconds) thermal_timeout="$value" ;;
          --post-warmup-thermal-poll-milliseconds) thermal_poll="$value" ;;
          --prompt-repeat) prompt_repeat="$value" ;;
          --max-tokens) max_tokens="$value" ;;
          --kv-attention) attention="$value" ;;
          --checkpoint-content-sha256) checkpoint="$value" ;;
          --model) model="$value" ;;
          --kvtuner-schedule) schedule="$value" ;;
        esac
      done
      printf '%s:%s:%s\n' "$block" "$position" "$label" >> "$FAKE_LAUNCHES"
      if [[ "$FAKE_REQUIRE_RESOLVED_RUNTIME_PATHS" == "true" ]]; then
        [[ -d "$model" ]] || exit 41
        [[ -z "$schedule" || -f "$schedule" ]] || exit 42
      fi
      if [[ "$FAKE_REQUIRE_THERMAL_POLICY" == "true" ]]; then
        [[ "$thermal_target" == "nominal" ]] || exit 43
        [[ "$thermal_timeout" == "600" ]] || exit 44
        [[ "$thermal_poll" == "1000" ]] || exit 45
      fi
      if [[ "$FAKE_STALL_HARNESS" == "true" ]]; then sleep 60; fi
      if [[ "$FAKE_OMIT_EVIDENCE" == "true" ]]; then exit 0; fi
      thermal="nominal"
      if [[ "$FAKE_CHANGE_SECOND_THERMAL" == "true" && "$position" == "1" ]]; then
        thermal="fair"
      fi
      observed=""
      case "$attention" in
        materialize) observed="materialized-kv" ;;
        split-affine-quantized-mm) observed="split-quantized-mm" ;;
        split-kvarn-quantized-mm) observed="split-kvarn-quantized-mm" ;;
      esac
      if [[ "$FAKE_MISMATCH_OBSERVED" == "true" && -n "$attention" ]]; then
        observed="materialized-kv"
      fi
      schedule_artifact="$FAKE_KVTUNER_ARTIFACT_SHA"
      if [[ "$FAKE_MISMATCH_KVTUNER" == "true" ]]; then
        schedule_artifact="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
      fi
      config_hash="$FAKE_MODEL_CONFIG_HASH"
      if [[ "$FAKE_MISMATCH_MODEL" == "true" ]]; then
        config_hash="9999999999999999"
      fi
      tokenizer_sha="$expected_tokenizer"
      if [[ "$FAKE_MISMATCH_TOKENIZER" == "true" ]]; then
        tokenizer_sha="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
      fi
      harness_sha="$FAKE_HARNESS_SHA"
      if [[ "$FAKE_READ_HARNESS_SHA_FROM_CWD" == "true" ]]; then
        if [[ -n "${GIT_DIR:-}" || -n "${GIT_WORK_TREE:-}" ||
              -n "${GIT_CEILING_DIRECTORIES:-}" ]]; then
          harness_sha="eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
        elif [[ -n "${HARNESS_GIT_SHA:-}" ]]; then
          harness_sha="$HARNESS_GIT_SHA"
        elif [[ -f .harness-sha ]]; then
          harness_sha="$(tr -d '[:space:]' < .harness-sha)"
        else
          harness_sha="unknown"
        fi
      fi
      jq -cn \
        --arg harness "$harness_sha" --arg label "$label" \
        --arg matrix "$matrix" --arg nonce "$nonce" --arg tier "$tier" \
        --arg manifest "$manifest_sha" --arg thermal "$thermal" \
        --arg thermalTarget "$thermal_target" \
        --arg attention "$attention" --arg observed "$observed" \
        --arg checkpoint "$checkpoint" --arg model "$model" \
        --arg configHash "$config_hash" \
        --arg checkpointManifestHash "$FAKE_MODEL_MANIFEST_HASH" \
        --arg schedule "$schedule" --arg scheduleArtifact "$schedule_artifact" \
        --arg scheduleBundle "$FAKE_KVTUNER_BUNDLE_SHA" \
        --arg modelConfigSHA256 \
          "3333333333333333333333333333333333333333333333333333333333333333" \
        --arg tokenizerSHA256 "$tokenizer_sha" \
        --argjson truncateAdmission "$FAKE_TRUNCATE_ADMISSION" \
        --argjson truncateKVTuner "$FAKE_TRUNCATE_KVTUNER" \
        --argjson lateThermalAdmission "$FAKE_LATE_THERMAL_ADMISSION" \
        --argjson block "$block" --argjson position "$position" \
        --argjson count "$count" --argjson memory "$memory" \
        --argjson cache "$cache" --argjson wired "$wired" \
        --argjson thermalTimeout "$thermal_timeout" \
        --argjson thermalPoll "$thermal_poll" \
        --argjson promptRepeat "$prompt_repeat" --argjson maxTokens "$max_tokens" \
        '{subcommand:"bench",provenance:{date:"2026-07-20T00:00:00Z",
          hardwareChip:"test",hardwareRAMBytes:100000,hardwareOS:"test",
          harnessGitSHA:$harness,mlxSwiftVersion:"test",
          referenceMLXVersion:null,referenceMLXLMVersion:null,modelPath:$model,
          modelConfigHash:$configHash,modelCheckpointManifestHash:$checkpointManifestHash,
          modelQuant:{bits:4,groupSize:64},corpusId:null,corpusContentHash:null,
          nonce:"qualification-test-v1"},payload:{
          label:$label,matrixID:$matrix,cellID:$tier,workloadNonce:$nonce,
          workload:"decode",mode:"none",decodeTokS:10,ttftMs:1000,
          quant:"int4",kvQuantTier:$tier,concurrency:1,
          promptRepeat:$promptRepeat,maxTokens:$maxTokens,
          measuredRuns:1,memoryRuns:[{samples:[
            {timestamp:1,physicalFootprintBytes:18000,mlxActiveBytes:100,
             mlxCacheBytes:50,mlxPeakBytes:150},
            {timestamp:2,physicalFootprintBytes:19000,mlxActiveBytes:110,
             mlxCacheBytes:60,mlxPeakBytes:160}],summary:{
             startFootprintBytes:18000,endFootprintBytes:19000,
             maxSampledFootprintBytes:19000,endFootprintDriftPercent:5.555,
             maxFootprintDriftPercent:5.555,maxMLXActiveBytes:110,
             maxMLXCacheBytes:60,maxMLXPeakBytes:160}}],
          maxSampledPhysicalFootprintBytes:19000,maxMLXActiveBytes:110,
          maxMLXCacheBytes:60,maxMLXPeakBytes:160,
          memoryCacheLimitBytes:$cache,
          workloadPromptSHA256:[
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"],
          promptTokenCountsByRun:[100],prefillDurationSecondsByRun:[1],
          prefillTokSByRun:[100],decodeTokSByRun:[10],ttftMsByRun:[1000],
          generatedTokenCountsByRun:[8],
          compressedKVAttention:(if $attention == "" then null else {
            request:$attention,observedOperation:$observed,admission:({
              family:"qwen3",modelType:"qwen3",architecture:"Qwen3ForCausalLM",
              modelConfigHash:$configHash,modelConfigSHA256:$modelConfigSHA256,
              checkpointManifestHash:$checkpointManifestHash,
              checkpointContentSHA256:$checkpoint,
              tokenizerSHA256:$tokenizerSHA256,layerCount:64,
              queryHeadCount:64,kvHeadCount:8,headDimension:128,
              maxPositionEmbeddings:40960} |
              if $truncateAdmission then del(.family) else . end)}
          end),kvtunerSchedule:(if $schedule == "" then null else {
            schemaVersion:4,scheduleSchemaVersion:4,
            artifactSHA256:$scheduleArtifact,
            qualificationBundleSHA256:$scheduleBundle,
            matrixID:$matrix,cellID:$tier,modelConfigHash:$configHash,
            modelConfigSHA256:$modelConfigSHA256,
            checkpointManifestHash:$checkpointManifestHash,
            checkpointContentSHA256:$checkpoint,tokenizerSHA256:$tokenizerSHA256,
            groupSize:128,layers:[range(0;64) |
              {layer:.,keyBits:8,valueBits:4}]} |
            if $truncateKVTuner then
              del(.modelConfigHash,.checkpointManifestHash,.tokenizerSHA256,
                .groupSize,.layers)
            else . end end),qualification:{schemaVersion:3,
          context:{runnerManifestSHA256:$manifest,matrixBlockIndex:$block,
          matrixRunPosition:$position,matrixCellCount:$count,
          memoryLimitBytes:$memory,cacheLimitBytes:$cache,wiredLimitBytes:$wired,
          tokenizerSHA256:$tokenizerSHA256,
          cacheResetPolicy:"in-place-before-every-generation",
          modelResidencyPolicy:"load-once-per-process",
          processIsolationPolicy:"fresh-process-per-matrix-position",
          postWarmupThermalPolicy:{target:$thermalTarget,
          timeoutSeconds:$thermalTimeout,
          pollIntervalMilliseconds:$thermalPoll}},
          warmup:{before:{monotonicTimestampSeconds:8,residentSizeBytes:19000,
          physicalFootprintBytes:17000,lowPowerModeEnabled:false,
          powerSource:"ac-power",thermalState:"nominal"},
          after:{monotonicTimestampSeconds:9,residentSizeBytes:20000,
          physicalFootprintBytes:18000,lowPowerModeEnabled:false,
          powerSource:"ac-power",thermalState:"nominal"}},
          postWarmupThermalAdmission:{snapshot:{monotonicTimestampSeconds:
          (if $lateThermalAdmission then 610 else 9.5 end),
          residentSizeBytes:20000,physicalFootprintBytes:18000,
          lowPowerModeEnabled:false,powerSource:"ac-power",
          thermalState:$thermalTarget}},runs:[{
          before:{monotonicTimestampSeconds:
          (if $lateThermalAdmission then 611 else 10 end),residentSizeBytes:20000,
          physicalFootprintBytes:18000,lowPowerModeEnabled:false,
          powerSource:"ac-power",thermalState:$thermal},
          after:{monotonicTimestampSeconds:
          (if $lateThermalAdmission then 612 else 11 end),residentSizeBytes:21000,
          physicalFootprintBytes:19000,lowPowerModeEnabled:false,
          powerSource:"ac-power",thermalState:$thermal}}]}}}' > "$evidence"
      if [[ "$FAKE_MUTATE_HARNESS_STAMP" == "true" &&
            "$block" == "0" && "$position" == "0" ]]; then
        printf '%s\n' bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb \
          > "$FAKE_HARNESS_SHA_FILE"
      fi
      """
    try fakeHarness.write(
      to: binary, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
      [.posixPermissions: 0o755], ofItemAtPath: binary.path)
    let actualBinarySHA256 = SHA256.hash(
      data: try Data(contentsOf: binary)).map {
        String(format: "%02x", $0)
      }.joined()
    let manifestObject: [String: Any] = [
      "schemaVersion": 2,
      "harnessGitSHA": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "harnessBinarySHA256": mismatchBinaryIdentity
        ? String(repeating: "f", count: 64) : actualBinarySHA256,
      "matrixID": "qualification-test-v1",
      "workloadNonce": "qualification-test-v1",
      "modelPath": runtimePathsRelativeToCaller
        ? "models/test" : model.path,
      "modelConfigHash": String(repeating: "1", count: 16),
      "modelCheckpointManifestHash": String(repeating: "2", count: 16),
      "modelTokenizerSHA256": String(repeating: "4", count: 64),
      "promptRepeat": 2,
      "maxTokens": 8,
      "memoryLimitBytes": 9_000,
      "cacheLimitBytes": 8_000,
      "wiredLimitBytes": 10_000,
      "postWarmupThermalTarget": "nominal",
      "postWarmupThermalTimeoutSeconds": 600,
      "postWarmupThermalPollMilliseconds": 1_000,
      "cells": includeKVTuner ? kvtunerCells : defaultCells,
      "blocks": selectedBlocks,
    ]
    try JSONSerialization.data(
      withJSONObject: manifestObject, options: [.sortedKeys])
      .write(to: manifest)
    return Fixture(
      root: root, manifest: manifest, output: output, binary: binary,
      harnessSHA: harnessSHA, launches: launches,
      changeSecondRowThermal: changeSecondRowThermal,
      mismatchObservedOperation: mismatchObservedOperation,
      mismatchKVTunerBinding: mismatchKVTunerBinding,
      omitEvidence: omitEvidence,
      mismatchModelIdentity: mismatchModelIdentity,
      mismatchTokenizerIdentity: mismatchTokenizerIdentity,
      truncateAdmissionEvidence: truncateAdmissionEvidence,
      truncateKVTunerEvidence: truncateKVTunerEvidence,
      stallHarness: stallHarness,
      readHarnessSHAFromCurrentDirectory: readHarnessSHAFromCurrentDirectory,
      spoofHarnessSHAEnvironment: spoofHarnessSHAEnvironment,
      spoofGitEnvironment: spoofGitEnvironment,
      mutateHarnessStampAfterFirstRun: mutateHarnessStampAfterFirstRun,
      requireResolvedRuntimePaths: requireResolvedRuntimePaths,
      requireThermalPolicyArguments: requireThermalPolicyArguments,
      lateThermalAdmission: lateThermalAdmission,
      kvtunerArtifactSHA256: kvtunerArtifactSHA256,
      kvtunerBundleSHA256: kvtunerBundleSHA256)
  }

  private func runRunner(
    fixture: Fixture,
    watchdogSeconds: String = "30",
    workingDirectory: URL? = nil
  ) throws -> (
    status: Int32, output: String
  ) {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = [
      runnerScriptURL.path, fixture.manifest.path, fixture.output.path,
    ]
    process.currentDirectoryURL = workingDirectory
    var environment = ProcessInfo.processInfo.environment
    environment["BIN"] = fixture.binary.path
    environment["HARNESS_SHA_FILE"] = fixture.harnessSHA.path
    environment["FAKE_LAUNCHES"] = fixture.launches.path
    environment["FAKE_HARNESS_SHA"] =
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    environment["FAKE_CHANGE_SECOND_THERMAL"] =
      fixture.changeSecondRowThermal ? "true" : "false"
    environment["FAKE_MISMATCH_OBSERVED"] =
      fixture.mismatchObservedOperation ? "true" : "false"
    environment["FAKE_MISMATCH_KVTUNER"] =
      fixture.mismatchKVTunerBinding ? "true" : "false"
    environment["FAKE_OMIT_EVIDENCE"] =
      fixture.omitEvidence ? "true" : "false"
    environment["FAKE_MISMATCH_MODEL"] =
      fixture.mismatchModelIdentity ? "true" : "false"
    environment["FAKE_MISMATCH_TOKENIZER"] =
      fixture.mismatchTokenizerIdentity ? "true" : "false"
    environment["FAKE_TRUNCATE_ADMISSION"] =
      fixture.truncateAdmissionEvidence ? "true" : "false"
    environment["FAKE_TRUNCATE_KVTUNER"] =
      fixture.truncateKVTunerEvidence ? "true" : "false"
    environment["FAKE_STALL_HARNESS"] =
      fixture.stallHarness ? "true" : "false"
    environment["FAKE_READ_HARNESS_SHA_FROM_CWD"] =
      fixture.readHarnessSHAFromCurrentDirectory ? "true" : "false"
    environment["FAKE_REQUIRE_RESOLVED_RUNTIME_PATHS"] =
      fixture.requireResolvedRuntimePaths ? "true" : "false"
    environment["FAKE_REQUIRE_THERMAL_POLICY"] =
      fixture.requireThermalPolicyArguments ? "true" : "false"
    environment["FAKE_LATE_THERMAL_ADMISSION"] =
      fixture.lateThermalAdmission ? "true" : "false"
    environment["FAKE_MUTATE_HARNESS_STAMP"] =
      fixture.mutateHarnessStampAfterFirstRun ? "true" : "false"
    environment["FAKE_HARNESS_SHA_FILE"] = fixture.harnessSHA.path
    if fixture.spoofHarnessSHAEnvironment {
      environment["HARNESS_GIT_SHA"] =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    }
    if fixture.spoofGitEnvironment {
      environment["GIT_DIR"] = "caller-controlled-git-directory"
      environment["GIT_WORK_TREE"] = "caller-controlled-git-work-tree"
      environment["GIT_CEILING_DIRECTORIES"] = fixture.root.path
    }
    environment["FAKE_MODEL_CONFIG_HASH"] = String(repeating: "1", count: 16)
    environment["FAKE_MODEL_MANIFEST_HASH"] = String(repeating: "2", count: 16)
    environment["FAKE_KVTUNER_ARTIFACT_SHA"] =
      fixture.kvtunerArtifactSHA256
    environment["FAKE_KVTUNER_BUNDLE_SHA"] =
      fixture.kvtunerBundleSHA256
    environment["DISABLE_CAFFEINATE"] = "true"
    environment["POLL_SECONDS"] = "1"
    environment["WATCHDOG_SECONDS"] = watchdogSeconds
    process.environment = environment
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = pipe
    try process.run()
    process.waitUntilExit()
    let output = String(
      decoding: pipe.fileHandleForReading.readDataToEndOfFile(),
      as: UTF8.self)
    return (process.terminationStatus, output)
  }

  private var runnerScriptURL: URL {
    URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .appendingPathComponent("scripts/run_loaded_bench_qualification.sh")
  }
}
