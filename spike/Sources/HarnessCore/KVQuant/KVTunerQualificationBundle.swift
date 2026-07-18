import Foundation

/// One portable runtime input containing the exact bytes of every artifact needed to re-derive a
/// KVTuner schedule. `Data` fields encode as base64 in JSON, preserving the source-file bytes that
/// each SHA-256 authenticates instead of silently reserializing typed values.
public struct KVTunerQualificationBundle: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var scheduleData: Data
    public var calibrationManifestData: Data
    public var sensitivityArtifactData: Data
    public var searchArtifactData: Data
    public var candidateEvaluationArtifactData: [Data]

    public init(
        schemaVersion: Int,
        scheduleData: Data,
        calibrationManifestData: Data,
        sensitivityArtifactData: Data,
        searchArtifactData: Data,
        candidateEvaluationArtifactData: [Data]
    ) {
        self.schemaVersion = schemaVersion
        self.scheduleData = scheduleData
        self.calibrationManifestData = calibrationManifestData
        self.sensitivityArtifactData = sensitivityArtifactData
        self.searchArtifactData = searchArtifactData
        self.candidateEvaluationArtifactData =
            candidateEvaluationArtifactData
    }
}
