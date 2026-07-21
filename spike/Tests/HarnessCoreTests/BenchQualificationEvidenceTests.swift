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

  func testLegacyThermalPolicyAndAdmissionDecodeWithoutStabilityFields()
    throws
  {
    let policyData = Data("""
      {"target":"nominal","timeoutSeconds":600,"pollIntervalMilliseconds":1000}
      """.utf8)
    let policy = try JSONDecoder().decode(
      BenchQualificationThermalPolicy.self, from: policyData)
    XCTAssertEqual(policy.stabilitySeconds, 0)

    let evidenceData = Data("""
      {"schemaVersion":3,
       "context":{"runnerManifestSHA256":"\(manifestSHA256)",
        "matrixBlockIndex":2,"matrixRunPosition":3,"matrixCellCount":7,
        "memoryLimitBytes":9000,"cacheLimitBytes":8000,"wiredLimitBytes":10000,
        "tokenizerSHA256":"\(String(repeating: "b", count: 64))",
        "cacheResetPolicy":"in-place-before-every-generation",
        "modelResidencyPolicy":"load-once-per-process",
        "processIsolationPolicy":"fresh-process-per-matrix-position",
        "postWarmupThermalPolicy":{"target":"nominal","timeoutSeconds":600,
        "pollIntervalMilliseconds":1000}},
       "warmup":{"before":{"monotonicTimestampSeconds":10,"residentSizeBytes":1,
        "physicalFootprintBytes":1,"lowPowerModeEnabled":false,
        "powerSource":"ac-power","thermalState":"nominal"},
        "after":{"monotonicTimestampSeconds":11,"residentSizeBytes":1,
        "physicalFootprintBytes":1,"lowPowerModeEnabled":false,
        "powerSource":"ac-power","thermalState":"fair"}},
       "postWarmupThermalAdmission":{"snapshot":{"monotonicTimestampSeconds":12,
        "residentSizeBytes":1,"physicalFootprintBytes":1,"lowPowerModeEnabled":false,
        "powerSource":"ac-power","thermalState":"nominal"}},
       "runs":[{"before":{"monotonicTimestampSeconds":13,"residentSizeBytes":1,
        "physicalFootprintBytes":1,"lowPowerModeEnabled":false,
        "powerSource":"ac-power","thermalState":"nominal"},
        "after":{"monotonicTimestampSeconds":14,"residentSizeBytes":1,
        "physicalFootprintBytes":1,"lowPowerModeEnabled":false,
        "powerSource":"ac-power","thermalState":"nominal"}}]}
      """.utf8)
    let evidence = try JSONDecoder().decode(
      BenchQualificationEvidence.self, from: evidenceData)
    XCTAssertEqual(evidence.schemaVersion, 3)
    XCTAssertEqual(evidence.postWarmupThermalAdmission?.stabilityObservations, [
      try XCTUnwrap(evidence.postWarmupThermalAdmission?.snapshot),
    ])
  }

  func testStablePostWarmupThermalTargetRoundTripsSampledNominalDwell()
    throws
  {
    let policy = try BenchQualificationThermalPolicy(
      target: .nominal,
      timeoutSeconds: 600,
      pollIntervalMilliseconds: 1_000,
      stabilitySeconds: 60)
    let context = try qualificationContext(
      postWarmupThermalPolicy: policy)
    let warmup = try BenchQualificationWarmupEnvironment(
      before: hostSnapshot(monotonicTimestampSeconds: 10),
      after: hostSnapshot(
        monotonicTimestampSeconds: 11,
        thermalState: .fair))
    let observations = [
      hostSnapshot(monotonicTimestampSeconds: 12),
      hostSnapshot(monotonicTimestampSeconds: 42),
      hostSnapshot(monotonicTimestampSeconds: 72),
    ]
    let admission = try BenchQualificationThermalAdmission(
      stabilityObservations: observations)
    let retained = try environmentRun(
      before: hostSnapshot(monotonicTimestampSeconds: 73),
      after: hostSnapshot(monotonicTimestampSeconds: 74))

    let evidence = try BenchQualificationEvidence(
      context: context,
      warmup: warmup,
      postWarmupThermalAdmission: admission,
      runs: [retained])

    XCTAssertEqual(evidence.schemaVersion, 4)
    XCTAssertEqual(
      evidence.postWarmupThermalAdmission?.stabilityObservations,
      observations)
    XCTAssertEqual(
      evidence.postWarmupThermalAdmission?.snapshot,
      observations.last)
    XCTAssertEqual(
      try JSONDecoder().decode(
        BenchQualificationEvidence.self,
        from: JSONEncoder().encode(evidence)),
      evidence)
  }

  func testStablePostWarmupThermalTargetRejectsShortOrInterruptedDwell()
    throws
  {
    let policy = try BenchQualificationThermalPolicy(
      target: .nominal,
      timeoutSeconds: 600,
      pollIntervalMilliseconds: 1_000,
      stabilitySeconds: 60)
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
        try BenchQualificationThermalAdmission(
          stabilityObservations: [
            hostSnapshot(monotonicTimestampSeconds: 12),
            hostSnapshot(monotonicTimestampSeconds: 71.999),
          ]),
      runs: [try environmentRun(
        before: hostSnapshot(monotonicTimestampSeconds: 73),
        after: hostSnapshot(monotonicTimestampSeconds: 74))])) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidThermalAdmission)
    }

    XCTAssertThrowsError(try BenchQualificationThermalAdmission(
      stabilityObservations: [
        hostSnapshot(monotonicTimestampSeconds: 12),
        hostSnapshot(
          monotonicTimestampSeconds: 42,
          thermalState: .fair),
        hostSnapshot(monotonicTimestampSeconds: 72),
      ])) {
      XCTAssertEqual(
        $0 as? BenchQualificationEvidenceError,
        .invalidThermalAdmission)
    }
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
