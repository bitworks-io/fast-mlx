import Foundation

/// One portable runtime input containing the exact bytes of every artifact needed to re-derive a
/// KVTuner schedule. `Data` fields encode as base64 in JSON, preserving the source-file bytes that
/// each SHA-256 authenticates instead of silently reserializing typed values.
public struct KVTunerQualificationBundle: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    /// Strong identity of the exact config used while authenticating the schedule.
    public var modelConfigSHA256: String
    /// Strong identity of every checkpoint byte used while authenticating the schedule.
    public var checkpointContentSHA256: String
    public var scheduleData: Data
    public var calibrationManifestData: Data
    public var sensitivityArtifactData: Data
    public var searchArtifactData: Data
    public var candidateEvaluationArtifactData: [Data]

    public init(
        schemaVersion: Int,
        modelConfigSHA256: String,
        checkpointContentSHA256: String,
        scheduleData: Data,
        calibrationManifestData: Data,
        sensitivityArtifactData: Data,
        searchArtifactData: Data,
        candidateEvaluationArtifactData: [Data]
    ) {
        self.schemaVersion = schemaVersion
        self.modelConfigSHA256 = modelConfigSHA256
        self.checkpointContentSHA256 = checkpointContentSHA256
        self.scheduleData = scheduleData
        self.calibrationManifestData = calibrationManifestData
        self.sensitivityArtifactData = sensitivityArtifactData
        self.searchArtifactData = searchArtifactData
        self.candidateEvaluationArtifactData =
            candidateEvaluationArtifactData
    }
}
