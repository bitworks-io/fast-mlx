// Copyright © 2026 Apple Inc.

import Foundation

/// The decoded `model.safetensors.index.json` for a model artifact.
///
/// This exists so callers outside `MLXLMCommon` -- notably a streaming
/// weight source that must decide which shard holds a given tensor, and
/// which tensor keys the whole artifact declares -- can get that
/// information without opening or reading a single tensor file. `weightKeys`
/// is the artifact's declared key set (the union of every shard's mapped
/// keys per the index), not a per-shard subset: do not mistake it for what
/// any one shard file on disk actually contains.
///
/// `weight_map` values (shard file names) come from a downloaded, untrusted
/// index and are validated at decode time -- see `validateShardName(_:)` --
/// so every successfully constructed instance's `weightMap` is guaranteed
/// to contain only plain, single-component filenames.
public struct SafetensorsWeightIndex: Sendable {
    /// Tensor key -> shard file name, exactly as `model.safetensors.index.json` declares it.
    public let weightMap: [String: String]

    /// The standard index file name inside a model directory.
    public static let indexFileName = "model.safetensors.index.json"

    private struct Decoded: Decodable {
        let weightMap: [String: String]

        enum CodingKeys: String, CodingKey {
            case weightMap = "weight_map"
        }
    }

    public init(decoding data: Data) throws {
        let decoded = try JSONDecoder().decode(Decoded.self, from: data)
        for shardName in decoded.weightMap.values {
            try Self.validateShardName(shardName)
        }
        self.weightMap = decoded.weightMap
    }

    /// Validates that a `weight_map` value is a plain, single-component
    /// filename, before it is ever stored or turned into a URL.
    ///
    /// `weight_map` values come from a downloaded `model.safetensors.index.json`
    /// -- model downloads are automated and unattended, so this input is
    /// genuinely untrusted, not merely malformed-by-accident.
    /// `URL.appendingPathComponent` does not sandbox its argument: a value
    /// like `"../../etc/passwd"` or `"sub/dir/x.safetensors"` produces a URL
    /// outside the model directory that `FileManager.fileExists` (and a
    /// subsequent read) will happily resolve.
    ///
    /// This validation runs at **decode time** (`init(decoding:)`), not only
    /// from `shardURLs(in:)`, and deliberately so: `weightKeys` (the
    /// whole-artifact key set used for checkpoint-namespace resolution) and
    /// `keys(inShard:)` / `shardFileName(forKey:)` are all derived from the
    /// same `weightMap` this validates. An index malformed enough to
    /// contain a traversal is not a trustworthy source for namespace
    /// resolution either, so there is no accessor for which "reject only at
    /// `shardURLs`" is the safer choice -- rejecting the whole artifact up
    /// front means every accessor on a successfully constructed
    /// `SafetensorsWeightIndex` is safe by construction, with no accessor
    /// left needing its own defensive check.
    ///
    /// Rejected: an empty string; any name containing a path separator
    /// (`/`), which in one check also rejects absolute paths and nested
    /// subdirectory components; and a name that is exactly `.` or `..`.
    ///
    /// Deliberately **not** rejected, and why that is not an oversight:
    ///   - Backslash (`\`). This code only ever runs against Darwin/POSIX
    ///     file systems, where backslash is an ordinary filename character,
    ///     not a path separator -- `URL.appendingPathComponent` does not
    ///     treat it specially. A name like `"a\\..\\b"` cannot escape
    ///     `modelDirectory` on this platform, so rejecting it would reject a
    ///     harmless filename rather than close a real hole.
    ///   - Percent-encoding (e.g. `"%2e%2e%2fpasswd"`). `weight_map` values
    ///     are plain JSON strings passed straight to
    ///     `appendingPathComponent`, which treats its argument as a literal
    ///     path component, not as a percent-encoded URL string requiring
    ///     decoding. `%2e%2e` never decodes to `..` on this path -- it
    ///     remains literal percent-sign characters in an unusual but
    ///     harmless filename.
    private static func validateShardName(_ name: String) throws {
        guard !name.isEmpty else {
            throw ModelFactoryError.invalidConfiguration(
                "model.safetensors.index.json weight_map maps a key to an empty shard name")
        }
        guard !name.contains("/") else {
            throw ModelFactoryError.invalidConfiguration(
                "model.safetensors.index.json weight_map maps a key to shard name "
                    + "\"\(name)\", which contains a path separator and is not a plain filename")
        }
        guard name != "." && name != ".." else {
            throw ModelFactoryError.invalidConfiguration(
                "model.safetensors.index.json weight_map maps a key to shard name "
                    + "\"\(name)\", which is a path-traversal component, not a plain filename")
        }
    }

    /// Loads the index from `modelDirectory`.
    ///
    /// Returns `nil` when the directory has no readable
    /// `model.safetensors.index.json` (matching the current unreadable/absent
    /// handling used elsewhere in this module: `try?` on the file read, not
    /// on the decode). Throws when the file is present but malformed, so a
    /// truncated or corrupt index is never silently treated as "no index".
    public static func load(modelDirectory: URL) throws -> SafetensorsWeightIndex? {
        let indexURL = modelDirectory.appendingPathComponent(indexFileName)
        guard let data = try? Data(contentsOf: indexURL) else {
            return nil
        }
        return try SafetensorsWeightIndex(decoding: data)
    }

    /// Every tensor key the whole artifact declares. Built from the index
    /// alone -- no safetensors file is opened.
    public var weightKeys: Set<String> {
        Set(weightMap.keys)
    }

    /// Unique shard file names, sorted.
    public var shardFileNames: [String] {
        Set(weightMap.values).sorted()
    }

    public func shardFileName(forKey key: String) -> String? {
        weightMap[key]
    }

    /// Sorted tensor keys mapped to the given shard.
    public func keys(inShard shardFileName: String) -> [String] {
        weightMap.filter { $0.value == shardFileName }.keys.sorted()
    }

    /// Shard URLs in `shardFileNames` order.
    ///
    /// Throws `ModelFactoryError.invalidConfiguration` naming the first
    /// mapped shard that does not exist on disk. This is what keeps
    /// index-external sidecar safetensors (drafter/vision/calibration
    /// extras) out of the returned set: only shards the index actually maps
    /// are ever considered, and any of those that are missing is a hard
    /// error rather than a silent skip.
    ///
    /// Shard names are already validated by `init(decoding:)` (see
    /// `validateShardName(_:)`), so by the time a `SafetensorsWeightIndex`
    /// exists, every name here is a plain filename -- this method does not
    /// re-validate.
    public func shardURLs(in modelDirectory: URL) throws -> [URL] {
        let urls = shardFileNames.map { modelDirectory.appendingPathComponent($0) }
        for url in urls {
            guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false))
            else {
                throw ModelFactoryError.invalidConfiguration(
                    "model.safetensors.index.json maps missing shard \(url.lastPathComponent)")
            }
        }
        return urls
    }
}
