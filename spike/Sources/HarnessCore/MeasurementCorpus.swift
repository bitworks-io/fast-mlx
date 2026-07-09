import Foundation

/// A fixed, checked-in measurement-corpus entry (distinct from `HarnessCorpus` in Corpus.swift,
/// which tests the hermetic post-processing invariants). These entries are the ACTUAL prompts a
/// `kl`/`verify`/`bench` run scores against — versioned so a result can be traced back to the
/// exact text that produced it.
public struct MeasurementCorpusEntry: Sendable, Codable, Hashable {
    public enum Tag: String, Sendable, Codable {
        case prose
        case code
        case longContext = "long-context"
    }
    public let id: String
    public let tag: Tag
    public let text: String
    public init(id: String, tag: Tag, text: String) { self.id = id; self.tag = tag; self.text = text }
}

/// A loaded, hashed corpus. `corpusId` is the declared human-readable version string from the
/// file; `contentHash` is computed here from the entries themselves (not the raw file bytes), so
/// whitespace/formatting churn in the JSON never changes the hash — only entry content does.
public struct MeasurementCorpus: Sendable {
    public let corpusId: String
    public let entries: [MeasurementCorpusEntry]
    public let contentHash: String
    public init(corpusId: String, entries: [MeasurementCorpusEntry], contentHash: String) {
        self.corpusId = corpusId; self.entries = entries; self.contentHash = contentHash
    }
    public func entries(tagged tag: MeasurementCorpusEntry.Tag) -> [MeasurementCorpusEntry] {
        entries.filter { $0.tag == tag }
    }
}

public enum MeasurementCorpusError: Error, CustomStringConvertible, Sendable {
    case decodeFailed(String)
    case empty
    public var description: String {
        switch self {
        case .decodeFailed(let what): return "measurement corpus: decode failed: \(what)"
        case .empty: return "measurement corpus: no entries"
        }
    }
}

/// Pure JSON -> `MeasurementCorpus` loader. Takes `Data`, never touches the filesystem itself —
/// the CLI reads the checked-in file and hands the bytes here — so this stays testable without
/// I/O and works identically off-box.
public enum MeasurementCorpusLoader {
    private struct RawCorpus: Codable {
        let corpusId: String
        let entries: [MeasurementCorpusEntry]
    }

    public static func load(from data: Data) throws -> MeasurementCorpus {
        let raw: RawCorpus
        do {
            raw = try JSONDecoder().decode(RawCorpus.self, from: data)
        } catch {
            throw MeasurementCorpusError.decodeFailed("\(error)")
        }
        guard !raw.entries.isEmpty else { throw MeasurementCorpusError.empty }
        return MeasurementCorpus(
            corpusId: raw.corpusId,
            entries: raw.entries,
            contentHash: contentHash(entries: raw.entries))
    }

    /// FNV-1a 64-bit over `id\0tag\0text` for each entry, sorted by id — a stable, dependency-free
    /// (Foundation-only) content fingerprint. Not cryptographic; the goal is "content changed"
    /// detection and a compact identity to record in provenance, not tamper-resistance.
    public static func contentHash(entries: [MeasurementCorpusEntry]) -> String {
        let bytes = entries
            .sorted(by: { $0.id < $1.id })
            .flatMap { "\($0.id)\u{0}\($0.tag.rawValue)\u{0}\($0.text)".utf8 }
        return fnv1a64(bytes)
    }
}

/// Evenly-spaced ascending sample of `sampleSize` indices in `0..<total` (endpoints included when
/// `sampleSize` allows). For scoring long sequences: materializing a full-vocab row at EVERY
/// position exhausts memory (~0.6MB/row x thousands of positions x 2 drivers), so long-context
/// measurement scores a bounded, reproducible subset instead of the whole sequence.
/// Returns `0..<total` unchanged when `sampleSize <= 0` or `sampleSize >= total`.
public func evenlySpacedPositions(total: Int, sampleSize: Int) -> [Int] {
    guard total > 0 else { return [] }
    guard sampleSize > 0, sampleSize < total else { return Array(0..<total) }
    if sampleSize == 1 { return [0] }
    var result: [Int] = []
    result.reserveCapacity(sampleSize)
    var seen = Set<Int>()
    for i in 0..<sampleSize {
        let pos = Int((Double(i) * Double(total - 1) / Double(sampleSize - 1)).rounded())
        if seen.insert(pos).inserted { result.append(pos) }
    }
    return result
}
