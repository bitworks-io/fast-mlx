import XCTest
@testable import HarnessCore

final class BenchQualificationEvidenceTests: XCTestCase {
  private let manifestSHA256 = String(repeating: "a", count: 64)

  func testQualificationEvidenceRoundTripsEveryOrderMemoryAndEnvironmentReceipt()
    throws
  {
    let evidence = try BenchQualificationEvidence(
      context: qualificationContext(),
      runs: [try environmentRun()])

    XCTAssertEqual(evidence.schemaVersion, 2)
    XCTAssertEqual(evidence.context.matrixBlockIndex, 2)
    XCTAssertEqual(evidence.context.matrixRunPosition, 3)
    XCTAssertEqual(evidence.context.matrixCellCount, 7)
    XCTAssertEqual(evidence.context.runnerManifestSHA256, manifestSHA256)
    XCTAssertEqual(
      evidence.context.tokenizerSHA256,
      String(repeating: "b", count: 64))
    XCTAssertEqual(
      evidence.context.cacheResetPolicy,
      .inPlaceBeforeEveryGeneration)
    XCTAssertEqual(
      evidence.context.modelResidencyPolicy,
      .loadOncePerProcess)
    XCTAssertEqual(
      evidence.context.processIsolationPolicy,
      .freshProcessPerMatrixPosition)
    XCTAssertEqual(
      try JSONDecoder().decode(
        BenchQualificationEvidence.self,
        from: JSONEncoder().encode(evidence)),
      evidence)
  }

  func testPostWarmupThermalTargetRoundTripsWarmupAdmissionAndRetainedRun()
    throws
  {
    let policy = try BenchQualificationThermalPolicy(
      target: .nominal,
      timeoutSeconds: 600,
      pollIntervalMilliseconds: 1_000)
    let context = try qualificationContext(
      postWarmupThermalPolicy: policy)
    let warmup = try BenchQualificationWarmupEnvironment(
      before: hostSnapshot(monotonicTimestampSeconds: 10),
      after: hostSnapshot(
        monotonicTimestampSeconds: 11,
        thermalState: .fair))
    let admission = try BenchQualificationThermalAdmission(
      snapshot: hostSnapshot(monotonicTimestampSeconds: 12))
    let retained = try environmentRun(
      before: hostSnapshot(monotonicTimestampSeconds: 13),
      after: hostSnapshot(monotonicTimestampSeconds: 14))

    let evidence = try BenchQualificationEvidence(
      context: context,
      warmup: warmup,
      postWarmupThermalAdmission: admission,
      runs: [retained])

    XCTAssertEqual(evidence.schemaVersion, 3)
    XCTAssertEqual(
      evidence.context.postWarmupThermalPolicy,
      policy)
    XCTAssertEqual(evidence.warmup, warmup)
    XCTAssertEqual(
      evidence.postWarmupThermalAdmission,
      admission)
    XCTAssertEqual(
      try JSONDecoder().decode(
        BenchQualificationEvidence.self,
        from: JSONEncoder().encode(evidence)),
      evidence)
  }

  func testPostWarmupThermalTargetFailsClosedForPartialOrUnsafeEvidence()
    throws
  {
    XCTAssertThrowsError(try BenchQualificationThermalPolicy(
      target: .nominal,
      timeoutSeconds: 0,
      pollIntervalMilliseconds: 1_000)) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidThermalPolicy)
    }
    XCTAssertThrowsError(try BenchQualificationThermalPolicy(
      target: .nominal,
      timeoutSeconds: 60,
      pollIntervalMilliseconds: 60_001)) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidThermalPolicy)
    }

    let policy = try BenchQualificationThermalPolicy(
      target: .nominal,
      timeoutSeconds: 600,
      pollIntervalMilliseconds: 1_000)
    let context = try qualificationContext(
      postWarmupThermalPolicy: policy)
    let warmup = try BenchQualificationWarmupEnvironment(
      before: hostSnapshot(monotonicTimestampSeconds: 10),
      after: hostSnapshot(
        monotonicTimestampSeconds: 11,
        thermalState: .fair))
    let admission = try BenchQualificationThermalAdmission(
      snapshot: hostSnapshot(monotonicTimestampSeconds: 12))

    XCTAssertThrowsError(try BenchQualificationEvidence(
      context: context,
      warmup: nil,
      postWarmupThermalAdmission: admission,
      runs: [try environmentRun(
        before: hostSnapshot(monotonicTimestampSeconds: 13),
        after: hostSnapshot(monotonicTimestampSeconds: 14))])) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidThermalTargetContract)
    }
    XCTAssertThrowsError(try BenchQualificationEvidence(
      context: context,
      warmup: warmup,
      postWarmupThermalAdmission:
        try BenchQualificationThermalAdmission(snapshot: hostSnapshot(
          monotonicTimestampSeconds: 12,
          thermalState: .fair)),
      runs: [try environmentRun(
        before: hostSnapshot(monotonicTimestampSeconds: 13),
        after: hostSnapshot(monotonicTimestampSeconds: 14))])) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidThermalAdmission)
    }
    XCTAssertThrowsError(try BenchQualificationEvidence(
      context: context,
      warmup: warmup,
      postWarmupThermalAdmission: admission,
      runs: [try environmentRun(
        before: hostSnapshot(monotonicTimestampSeconds: 11.5),
        after: hostSnapshot(monotonicTimestampSeconds: 14))])) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidThermalAdmission)
    }
  }

  func testPostWarmupThermalAdmissionCannotExceedManifestTimeout()
    throws
  {
    let policy = try BenchQualificationThermalPolicy(
      target: .nominal,
      timeoutSeconds: 600,
      pollIntervalMilliseconds: 1_000)
    let context = try qualificationContext(
      postWarmupThermalPolicy: policy)
    let warmup = try BenchQualificationWarmupEnvironment(
      before: hostSnapshot(monotonicTimestampSeconds: 10),
      after: hostSnapshot(
        monotonicTimestampSeconds: 11,
        thermalState: .fair))

    XCTAssertThrowsError(try BenchQualificationEvidence(
      context: context,
      warmup: warmup,
      postWarmupThermalAdmission:
        try BenchQualificationThermalAdmission(snapshot: hostSnapshot(
          monotonicTimestampSeconds: 611.001)),
      runs: [try environmentRun(
        before: hostSnapshot(monotonicTimestampSeconds: 612),
        after: hostSnapshot(monotonicTimestampSeconds: 613))])) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidThermalAdmission)
    }
  }

  func testWarmupEvidenceAllowsNominalToFairButRejectsUnknownOrPowerDrift()
    throws
  {
    XCTAssertNoThrow(try BenchQualificationWarmupEnvironment(
      before: hostSnapshot(monotonicTimestampSeconds: 10),
      after: hostSnapshot(
        monotonicTimestampSeconds: 11,
        thermalState: .fair)))
    XCTAssertThrowsError(try BenchQualificationWarmupEnvironment(
      before: hostSnapshot(
        monotonicTimestampSeconds: 10,
        thermalState: .unknown),
      after: hostSnapshot(monotonicTimestampSeconds: 11))) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidWarmupEvidence)
    }
    XCTAssertThrowsError(try BenchQualificationWarmupEnvironment(
      before: hostSnapshot(monotonicTimestampSeconds: 10),
      after: hostSnapshot(
        monotonicTimestampSeconds: 11,
        powerSource: .battery))) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidWarmupEvidence)
    }
  }

  func testQualificationContextRejectsPartialOrInconsistentOrderAndMemoryIdentity() {
    XCTAssertThrowsError(try qualificationContext(matrixBlockIndex: -1)) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidMatrixPosition)
    }
    XCTAssertThrowsError(try qualificationContext(matrixRunPosition: 7)) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidMatrixPosition)
    }
    XCTAssertThrowsError(try qualificationContext(matrixCellCount: 0)) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidMatrixPosition)
    }
    XCTAssertThrowsError(
      try qualificationContext(runnerManifestSHA256: "not-a-digest")
    ) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidRunnerManifestSHA256)
    }
    XCTAssertThrowsError(
      try qualificationContext(tokenizerSHA256: "not-a-digest")
    ) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidTokenizerSHA256)
    }
    XCTAssertThrowsError(
      try qualificationContext(
        memoryLimitBytes: 7_000,
        cacheLimitBytes: 8_000,
        wiredLimitBytes: 9_000)
    ) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidMemorySettings)
    }
    XCTAssertThrowsError(
      try qualificationContext(
        memoryLimitBytes: 10_000,
        cacheLimitBytes: 8_000,
        wiredLimitBytes: 9_000)
    ) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidMemorySettings)
    }
  }

  func testQualificationRunRejectsPowerLowPowerThermalAndClockTransitions() throws {
    let before = hostSnapshot()

    XCTAssertThrowsError(try environmentRun(after: hostSnapshot(
      monotonicTimestampSeconds: 11,
      lowPowerModeEnabled: true))) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidPowerState)
    }
    XCTAssertThrowsError(try environmentRun(after: hostSnapshot(
      monotonicTimestampSeconds: 11,
      powerSource: .battery))) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidPowerState)
    }
    XCTAssertThrowsError(try BenchQualificationRunEnvironment(
      before: hostSnapshot(powerSource: .unavailable),
      after: hostSnapshot(
        monotonicTimestampSeconds: 11,
        powerSource: .unavailable))) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidPowerState)
    }
    XCTAssertThrowsError(try environmentRun(after: hostSnapshot(
      monotonicTimestampSeconds: 11,
      thermalState: .fair))) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidThermalState)
    }
    XCTAssertThrowsError(try BenchQualificationRunEnvironment(
      before: hostSnapshot(thermalState: .unknown),
      after: hostSnapshot(
        monotonicTimestampSeconds: 11,
        thermalState: .unknown))) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidThermalState)
    }
    XCTAssertThrowsError(try environmentRun(after: hostSnapshot(
      monotonicTimestampSeconds: before.monotonicTimestampSeconds))) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidMonotonicTiming)
    }
  }

  func testQualificationEvidenceRejectsMissingOrMalformedRunReceipts() throws {
    XCTAssertThrowsError(try BenchQualificationEvidence(
      context: qualificationContext(),
      runs: [])) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidRunCount(0))
    }
    XCTAssertThrowsError(try BenchQualificationEvidence(
      context: qualificationContext(),
      runs: [try environmentRun(), try environmentRun()])) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidRunCount(2))
    }
    XCTAssertThrowsError(try environmentRun(before: hostSnapshot(
      residentSizeBytes: 0))) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidProcessMemory)
    }
  }

  func testStableLowPowerModeIsRecordedWithoutInventingATransition() throws {
    let run = try BenchQualificationRunEnvironment(
      before: hostSnapshot(lowPowerModeEnabled: true),
      after: hostSnapshot(
        monotonicTimestampSeconds: 11,
        lowPowerModeEnabled: true))

    XCTAssertTrue(run.before.lowPowerModeEnabled)
    XCTAssertTrue(run.after.lowPowerModeEnabled)
  }

  private func qualificationContext(
    runnerManifestSHA256: String? = nil,
    matrixBlockIndex: Int = 2,
    matrixRunPosition: Int = 3,
    matrixCellCount: Int = 7,
    memoryLimitBytes: Int = 9_000,
    cacheLimitBytes: Int = 8_000,
    wiredLimitBytes: Int = 10_000,
    tokenizerSHA256: String = String(repeating: "b", count: 64),
    postWarmupThermalPolicy: BenchQualificationThermalPolicy? = nil
  ) throws -> BenchQualificationContext {
    try BenchQualificationContext(
      runnerManifestSHA256: runnerManifestSHA256 ?? manifestSHA256,
      matrixBlockIndex: matrixBlockIndex,
      matrixRunPosition: matrixRunPosition,
      matrixCellCount: matrixCellCount,
      memoryLimitBytes: memoryLimitBytes,
      cacheLimitBytes: cacheLimitBytes,
      wiredLimitBytes: wiredLimitBytes,
      tokenizerSHA256: tokenizerSHA256,
      cacheResetPolicy: .inPlaceBeforeEveryGeneration,
      modelResidencyPolicy: .loadOncePerProcess,
      processIsolationPolicy: .freshProcessPerMatrixPosition,
      postWarmupThermalPolicy: postWarmupThermalPolicy)
  }

  private func environmentRun(
    before: BenchQualificationHostSnapshot? = nil,
    after: BenchQualificationHostSnapshot? = nil
  ) throws -> BenchQualificationRunEnvironment {
    try BenchQualificationRunEnvironment(
      before: before ?? hostSnapshot(),
      after: after ?? hostSnapshot(monotonicTimestampSeconds: 11))
  }

  private func hostSnapshot(
    monotonicTimestampSeconds: Double = 10,
    residentSizeBytes: Int = 20_000,
    physicalFootprintBytes: Int = 18_000,
    lowPowerModeEnabled: Bool = false,
    powerSource: CompressedAttentionProbePowerSource = .acPower,
    thermalState: CompressedAttentionProbeThermalState = .nominal
  ) -> BenchQualificationHostSnapshot {
    BenchQualificationHostSnapshot(
      monotonicTimestampSeconds: monotonicTimestampSeconds,
      residentSizeBytes: residentSizeBytes,
      physicalFootprintBytes: physicalFootprintBytes,
      lowPowerModeEnabled: lowPowerModeEnabled,
      powerSource: powerSource,
      thermalState: thermalState)
  }
}
