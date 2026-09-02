import CryptoKit
import Foundation
import HarnessCore

/// Both live workers run the same engine build on the same artifact; the sealed
/// authority requires one shared label with the roles split by executionMode.
package let qwen38MTPScorecardSharedEngineLabel = "engine"

package struct Qwen38MTPScorecardWorkerHandshake:
    Codable, Equatable, Sendable
{
    package let role: Qwen38MTPPerformanceScorecardEngineRole
    package let model: Qwen38MTPPerformanceScorecardModel
    package let processIsolation: Qwen38MTPLiveExactnessProcessIsolationEvidence
    package let launchBinding: Qwen38MTPPerformanceScorecardLaunchBinding

    package init(
        role: Qwen38MTPPerformanceScorecardEngineRole,
        model: Qwen38MTPPerformanceScorecardModel,
        processIsolation: Qwen38MTPLiveExactnessProcessIsolationEvidence,
        launchBinding: Qwen38MTPPerformanceScorecardLaunchBinding
    ) {
        self.role = role
        self.model = model
        self.processIsolation = processIsolation
        self.launchBinding = launchBinding
    }
}

package enum Qwen38MTPScorecardWorkerProtocolResponseKind: String, Codable, Sendable {
    case handshake
    case exactness
    case measurement
    case ok
    case error
}

package struct Qwen38MTPScorecardWorkerProtocolResponse: Codable, Sendable {
    package let sequence: Int
    package let kind: Qwen38MTPScorecardWorkerProtocolResponseKind
    package var handshake: Qwen38MTPScorecardWorkerHandshake?
    package var exactnessRecord: ResultRecord<Qwen38MTPLiveExactnessEvidence>?
    package var measurement: Qwen38MTPPerformanceScorecardEngineMeasurement?
    package var error: String?

    package init(
        sequence: Int,
        kind: Qwen38MTPScorecardWorkerProtocolResponseKind,
        handshake: Qwen38MTPScorecardWorkerHandshake? = nil,
        exactnessRecord: ResultRecord<Qwen38MTPLiveExactnessEvidence>? = nil,
        measurement: Qwen38MTPPerformanceScorecardEngineMeasurement? = nil,
        error: String? = nil
    ) {
        self.sequence = sequence
        self.kind = kind
        self.handshake = handshake
        self.exactnessRecord = exactnessRecord
        self.measurement = measurement
        self.error = error
    }

    package func jsonLine() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(self)
        return String(decoding: data, as: UTF8.self)
    }
}

package enum Qwen38MTPScorecardWorkerProtocolRequestKind: String, Codable, Sendable {
    case handshake
    case exactness
    case measure
    case assertReady
    case shutdown
}

package struct Qwen38MTPScorecardWorkerProtocolRequest: Codable, Sendable {
    package let sequence: Int
    package let kind: Qwen38MTPScorecardWorkerProtocolRequestKind
    package let measurement: Qwen38MTPScorecardMeasurementCommand?

    package init(
        sequence: Int,
        kind: Qwen38MTPScorecardWorkerProtocolRequestKind,
        measurement: Qwen38MTPScorecardMeasurementCommand?
    ) {
        self.sequence = sequence
        self.kind = kind
        self.measurement = measurement
    }
}

package struct Qwen38MTPScorecardMeasurementCommand: Codable, Sendable {
    package let role: Qwen38MTPPerformanceScorecardEngineRole
    package let identity: Qwen38MTPPerformanceScorecardModel
    package let schedule: Qwen38MTPPerformanceScorecardPairSchedule
    package let workload: Qwen38MTPPerformanceScorecardWorkload
    package let settings: Qwen38MTPPerformanceScorecardSettings

    package init(_ request: Qwen38MTPPerformanceScorecardMeasurementRequest) {
        role = request.role
        identity = request.identity
        schedule = request.schedule
        workload = request.workload
        settings = request.settings
    }

    package var request: Qwen38MTPPerformanceScorecardMeasurementRequest {
        Qwen38MTPPerformanceScorecardMeasurementRequest(
            role: role,
            identity: identity,
            schedule: schedule,
            workload: workload,
            settings: settings)
    }
}

package protocol Qwen38MTPScorecardLineTransport: Sendable {
    func sendLine(_ line: String) async throws
    func receiveLine() async throws -> String?
    func terminate() async
}

package func qwen38MTPScorecardCanonicalJSON<T: Encodable>(_ value: T) throws -> Data {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    return try encoder.encode(value)
}

package func qwen38MTPScorecardSHA256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}
