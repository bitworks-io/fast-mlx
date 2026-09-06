import Foundation

import HarnessCore

/// Errors surfaced while composing the offloaded-n-gram fit profile. Every case names the exact
/// failure mode so a caller (and a test) can assert on the SPECIFIC reason a composition refused,
/// rather than a generic "it threw something" — see the module doc comment on `NGramOffloadFitComposition`
/// for why this composition must fail closed rather than ever return a partial/optimistic reduction.
public enum NGramOffloadFitCompositionError: Error, Equatable, Sendable, CustomStringConvertible {
    /// A safetensors shard could not be opened, or fewer than the required bytes were available
    /// (either the leading 8-byte header-length prefix, or the declared header JSON itself).
    case safetensorsHeaderTruncated(path: String)
    /// The declared 8-byte little-endian header length was non-positive, exceeded the file's own
    /// size, or exceeded the sane upper cap — guards against a corrupt/hostile file forcing a huge
    /// allocation before a single tensor has been inspected.
    case safetensorsHeaderLengthInvalid(path: String)
    /// The header bytes did not parse as JSON.
    case safetensorsHeaderNotJSON(path: String)
    /// A tensor entry in the header was not an object, or its `data_offsets` was not a two-element
    /// `[begin, end]` array with `end >= begin`.
    case safetensorsTensorEntryInvalid(path: String, key: String)
    /// The offloaded-shard-key matcher matched nothing across every scanned shard. This is the
    /// single most important failure mode this file exists to make loud: if the checkpoint-key
    /// prefix is ever wrong (a typo, a namespace this artifact doesn't use, a stale layer index),
    /// the matcher silently matches nothing, the "offload" appears to succeed while saving zero
    /// bytes, and the fit goes wrong silently in the OPTIMISTIC direction. Throwing here — instead
    /// of returning a zero reduction — is the fail-closed answer.
    case noMatchingOffloadedShardKeys
    /// Matched keys disagreed on which PLE layer they belong to. The real artifact has exactly one
    /// PLE layer; matches spanning more than one layer index mean the matcher is over-matching
    /// (or the checkpoint has an unsupported multi-PLE-layer geometry), either of which makes the
    /// resulting byte sum untrustworthy.
    case multipleLayerIndicesInOffloadedShardKeys(indices: [Int])
    /// The matched shard indices were not a contiguous `0..<n` run. A gap means either a shard's
    /// keys were missed by the matcher (under-counting the true offloaded footprint) or an
    /// unexpected on-disk layout — both make the sum untrustworthy.
    case nonContiguousShardIndices(shardCount: Int)
    /// Not every shard carried the same set of fields (e.g. one shard has `weight`/`scales`/
    /// `biases` while another has only `weight`). A real checkpoint's shards are homogeneous —
    /// disagreement means the matcher missed keys on some shards.
    case inconsistentShardFieldSets
    /// The (consistent, per the check above) per-shard field count was neither 1 (dense: `weight`
    /// only) nor 3 (quantized: `weight`/`scales`/`biases`) — the two encodings the embedding
    /// format actually supports.
    case invalidPerShardFieldCount(count: Int)
    /// The same (layer, shard, field) offloaded n-gram tensor was declared by more than one
    /// scanned `*.safetensors` file — e.g. a consolidated `model.safetensors` staged beside the
    /// sharded set, a leftover copy of a shard, or a re-quantized variant left in place alongside
    /// the original. Every structural-audit check below (single layer index, contiguous shard
    /// run, homogeneous per-shard field set, valid field count) is satisfied identically whether a
    /// tensor is counted once or twice, so nothing else in this file is positioned to catch the
    /// duplication. Counting it twice inflates the offloaded byte total, which makes
    /// `adjustedWeights` too SMALL — the optimistic direction: the fit check passes on a figure
    /// that understates true resident memory, and the serve OOMs. The sibling Python repacker
    /// (`ngram_q4_repacker.py`) treats the identical situation as a hard error
    /// (`duplicate tensor name across shard files`) for the same reason; this mirrors that
    /// precedent in Swift rather than silently accepting the second declaration.
    case duplicateOffloadedShardKey(key: String)
    /// The computed offloaded byte total was not strictly less than the base (full-resident)
    /// weights figure. A real offload always shrinks the resident footprint; an offloaded total
    /// at or above the whole-file total means the matcher over-matched, and refusing here prevents
    /// ever classifying a zero-or-negative weight figure as a fit (a phantom-GREEN this repo has
    /// removed twice already).
    case offloadedBytesNotLessThanBaseWeights(offloadedBytes: Int, baseWeightsBytes: Int)
    /// The subtract-then-add formula overflowed `Int`.
    case arithmeticOverflow
    /// The plan file could not be read.
    case planFileUnreadable(path: String)
    /// The plan file's contents did not parse as JSON.
    case planFileNotJSON(path: String)
    /// The plan JSON had no (object-typed) `limits` key.
    case planLimitsMissing(path: String)
    /// `limits.maxResidentBytes` was absent, non-numeric, or not strictly positive.
    case planMaxResidentBytesInvalid(path: String)

    public var description: String {
        switch self {
        case .safetensorsHeaderTruncated(let path):
            "safetensorsHeaderTruncated(path: \(path))"
        case .safetensorsHeaderLengthInvalid(let path):
            "safetensorsHeaderLengthInvalid(path: \(path))"
        case .safetensorsHeaderNotJSON(let path):
            "safetensorsHeaderNotJSON(path: \(path))"
        case .safetensorsTensorEntryInvalid(let path, let key):
            "safetensorsTensorEntryInvalid(path: \(path), key: \(key))"
        case .noMatchingOffloadedShardKeys:
            "noMatchingOffloadedShardKeys"
        case .multipleLayerIndicesInOffloadedShardKeys(let indices):
            "multipleLayerIndicesInOffloadedShardKeys(indices: \(indices))"
        case .nonContiguousShardIndices(let shardCount):
            "nonContiguousShardIndices(shardCount: \(shardCount))"
        case .inconsistentShardFieldSets:
            "inconsistentShardFieldSets"
        case .invalidPerShardFieldCount(let count):
            "invalidPerShardFieldCount(count: \(count))"
        case .duplicateOffloadedShardKey(let key):
            "duplicateOffloadedShardKey(key: \(key))"
        case .offloadedBytesNotLessThanBaseWeights(let offloadedBytes, let baseWeightsBytes):
            "offloadedBytesNotLessThanBaseWeights(offloadedBytes: \(offloadedBytes), baseWeightsBytes: \(baseWeightsBytes))"
        case .arithmeticOverflow:
            "arithmeticOverflow"
        case .planFileUnreadable(let path):
            "planFileUnreadable(path: \(path))"
        case .planFileNotJSON(let path):
            "planFileNotJSON(path: \(path))"
        case .planLimitsMissing(let path):
            "planLimitsMissing(path: \(path))"
        case .planMaxResidentBytesInvalid(let path):
            "planMaxResidentBytesInvalid(path: \(path))"
        }
    }
}

/// Composes a fit-check profile for a `qwen4_exp` (Qwen3.8-Flash-Next) checkpoint served with an
/// n-gram offload plan, whose PLE n-gram embedding shard tensors are streamed from a sealed
/// on-disk row store instead of held resident.
///
/// `base` (from `ModelConfigDecoder.decodeModelDirectory`) sizes `weightsBytes4bitEstimate` as the
/// FULL-RESIDENT total — every safetensors byte on disk, including the offloaded table — because
/// the ordinary decode path has no notion of an offload plan. Serving that checkpoint WITH an
/// offload plan genuinely needs far less resident memory, so a fit-check against the full-resident
/// figure produces a false RED on hosts that would actually fit. This type corrects the figure for
/// that one configuration:
///
/// ```
/// adjustedWeights = wholeFileTotal - offloadedTensorBytes + planLimits.maxResidentBytes
/// ```
///
/// `wholeFileTotal` is `base.profile.weightsBytes4bitEstimate` itself (already the measured/declared
/// full on-disk total). The third term is MANDATORY, not an optional refinement: the offload row
/// store holds its OWN resident budget (`limits.maxResidentBytes` in the plan file) alongside the
/// checkpoint. Omitting it models only the smaller "table gone entirely" figure while an
/// operator-authored large residency budget consumes real memory the fit-check must still account
/// for — an optimistic fit that loads and then OOMs, which is strictly WORSE than the false RED
/// being corrected here.
///
/// This composition is intentionally narrow and self-limiting: it recognizes the offloaded n-gram
/// shard keys by their on-disk checkpoint-key shape (see `matchOffloadedNGramShardKey` below) rather
/// than by trusting the caller's family label. A checkpoint that does not carry those keys at all
/// (i.e. is not the real artifact this plan targets) yields zero matches, which the structural audit
/// below turns into a hard throw rather than a silent zero-byte "reduction" — see
/// `NGramOffloadFitCompositionError.noMatchingOffloadedShardKeys`'s doc comment. The vendored
/// loader-side n-gram key predicate in the vendored MLXLLM PLE checkpoint-keys file runs an
/// equivalent audit for the identical reason: a wrong prefix must fail loudly, not load the full
/// table resident while claiming the offload succeeded.
public enum NGramOffloadFitComposition {
    /// The complete text-module checkpoint-key prefixes (each including its own trailing `.`) under
    /// which the offloaded n-gram table's on-disk keys may appear, mirroring the three prefixes the
    /// vendored loader-side predicate supports (confirmed directly against that file's own prefix
    /// list, which is `["language_model.model.", "model.", "model.language_model."]` — note the
    /// third prefix nests `language_model.` INSIDE `model.`, the opposite inner-segment order from
    /// the first, which is why these are complete prefixes rather than a shared namespace glued
    /// onto one template):
    /// - `"language_model.model."` — the converted artifact this project produces.
    /// - `"model."` — the bare/sanitized root used by fixtures.
    /// - `"model.language_model."` — the real official upstream artifact.
    static let textModulePrefixes = [
        "language_model.model.", "model.", "model.language_model.",
    ]

    /// The field names a shard key's trailing path component may be.
    static let shardFieldNames: Set<String> = ["weight", "scales", "biases"]

    /// A single offloaded n-gram shard-key match: which layer/shard/field it names, and how many
    /// bytes its tensor occupies (from the safetensors header's `data_offsets`, never the blob).
    struct MatchedShardKey {
        let layerIndex: Int
        let shardIndex: Int
        let field: String
        let byteSize: Int
    }

    /// Builds the corrected `ParsedModelArch` for a `qwen4_exp` checkpoint served with an n-gram
    /// offload plan. See the type's doc comment for the formula and why every term is mandatory.
    public static func make(
        base: ParsedModelArch,
        modelDirectory: URL,
        planFileURL: URL
    ) throws -> ParsedModelArch {
        let offloadedTensorBytes = try offloadedNGramTensorBytes(inModelDirectory: modelDirectory)
        let maxResidentBytes = try planMaxResidentBytes(atPlanFileURL: planFileURL)

        let wholeFileTotal = base.profile.weightsBytes4bitEstimate
        guard offloadedTensorBytes < wholeFileTotal else {
            throw NGramOffloadFitCompositionError.offloadedBytesNotLessThanBaseWeights(
                offloadedBytes: offloadedTensorBytes, baseWeightsBytes: wholeFileTotal)
        }

        let (afterOffloadRemoved, subtractOverflow) = wholeFileTotal
            .subtractingReportingOverflow(offloadedTensorBytes)
        let (adjustedWeights, addOverflow) = afterOffloadRemoved
            .addingReportingOverflow(maxResidentBytes)
        guard !subtractOverflow, !addOverflow else {
            throw NGramOffloadFitCompositionError.arithmeticOverflow
        }

        // Copy EVERY field of `base.profile` — a field-by-field rebuild that drops one (as a prior
        // composition in this codebase dropped `auxPerLayerKeyDim`) silently under-counts a term a
        // later reader has no way to notice from this call site alone.
        let profile = base.profile
        return ParsedModelArch(
            profile: ModelArchProfile(
                id: "\(profile.id)+ngram-offload-composition",
                modelType: profile.modelType,
                nLayers: profile.nLayers,
                nAttnLayers: profile.nAttnLayers,
                nKVHeads: profile.nKVHeads,
                headDim: profile.headDim,
                slidingWindow: profile.slidingWindow,
                fixedStateBytes: profile.fixedStateBytes,
                nativeMaxContext: profile.nativeMaxContext,
                weightsBytes4bitEstimate: adjustedWeights,
                license: profile.license,
                mlaHeads: profile.mlaHeads,
                mlaRopeDim: profile.mlaRopeDim,
                mlaNopeDim: profile.mlaNopeDim,
                mlaVDim: profile.mlaVDim,
                swaKVHeads: profile.swaKVHeads,
                swaHeadDim: profile.swaHeadDim,
                vHeadDim: profile.vHeadDim,
                swaVHeadDim: profile.swaVHeadDim,
                auxPerLayerKeyDim: profile.auxPerLayerKeyDim),
            // This figure IS measured from the artifact's own safetensors headers (the whole-file
            // total was already measured/declared by the ordinary decode path, and the offloaded
            // subtraction/residency addition are exact arithmetic on measured/plan-declared
            // quantities) — provenance carries through from `base` unchanged, never regressing to
            // an unmeasured/undeclared state.
            weightsAreMeasured: base.weightsAreMeasured,
            weightsAreDeclared: base.weightsAreDeclared,
            quantBits: base.quantBits)
    }

    /// Scans every `*.safetensors` shard in `modelDirectory`, matches the offloaded n-gram shard
    /// keys, runs the structural audit (see the type's doc comment), and returns the summed byte
    /// size of every matched tensor (from `data_offsets`, never the tensor's own data blob).
    public static func offloadedNGramTensorBytes(inModelDirectory modelDirectory: URL) throws -> Int
    {
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: modelDirectory, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        else {
            throw NGramOffloadFitCompositionError.noMatchingOffloadedShardKeys
        }

        var matches: [MatchedShardKey] = []
        // DEFECT 2: tracks every (layer, shard, field) triple already matched across every shard
        // file scanned so far — see `NGramOffloadFitCompositionError.duplicateOffloadedShardKey`'s
        // doc comment for why a duplicate must throw rather than accumulate twice.
        var seenShardKeyTriples: Set<String> = []
        for url in entries.sorted(by: { $0.path < $1.path }) where url.pathExtension == "safetensors" {
            // Hugging Face snapshot directories store each shard as a symlink into the shared blobs
            // cache — resolve to the real file before opening, mirroring
            // `ModelConfigDecoder.sumSafetensorsBytes`'s own resolution.
            let resolvedURL = url.resolvingSymlinksInPath()
            let tensorSizes = try readSafetensorsTensorByteSizes(at: resolvedURL)
            for (key, byteSize) in tensorSizes {
                guard let match = matchOffloadedNGramShardKey(key) else { continue }
                let triple = "\(match.layerIndex).\(match.shardIndex).\(match.field)"
                guard seenShardKeyTriples.insert(triple).inserted else {
                    throw NGramOffloadFitCompositionError.duplicateOffloadedShardKey(key: key)
                }
                matches.append(
                    MatchedShardKey(
                        layerIndex: match.layerIndex, shardIndex: match.shardIndex,
                        field: match.field, byteSize: byteSize))
            }
        }

        try auditMatches(matches)
        return try sumMatchedByteSizes(matches)
    }

    /// Sums matched tensor byte sizes with overflow-checked arithmetic. A plain `+` reduction would
    /// TRAP on overflow, crashing the serving process rather than surfacing a catchable error — not
    /// the conservative failure mode this file's fail-closed design promises (see the module doc
    /// comment: "any error means no reduction" must hold even when the sum itself is what would
    /// overflow `Int`). Every other arithmetic site in this file already guards with
    /// `addingReportingOverflow`/`subtractingReportingOverflow`; this keeps that guard uniform.
    static func sumMatchedByteSizes(_ matches: [MatchedShardKey]) throws -> Int {
        var total = 0
        for match in matches {
            let (sum, overflowed) = total.addingReportingOverflow(match.byteSize)
            guard !overflowed else {
                throw NGramOffloadFitCompositionError.arithmeticOverflow
            }
            total = sum
        }
        return total
    }

    /// Reads `planFileURL` and returns `limits.maxResidentBytes`, throwing when the plan cannot be
    /// read, does not parse as JSON, has no `limits` object, or that field is absent/non-positive.
    /// Deliberately parses ONLY this one field — the loader that actually resolves the plan runs
    /// its own full, fail-closed validation later; re-implementing that here would just be a second
    /// place for the two to drift apart.
    public static func planMaxResidentBytes(atPlanFileURL planFileURL: URL) throws -> Int {
        guard let data = try? Data(contentsOf: planFileURL) else {
            throw NGramOffloadFitCompositionError.planFileUnreadable(path: planFileURL.path)
        }
        guard let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            throw NGramOffloadFitCompositionError.planFileNotJSON(path: planFileURL.path)
        }
        guard let limits = root["limits"] as? [String: Any] else {
            throw NGramOffloadFitCompositionError.planLimitsMissing(path: planFileURL.path)
        }
        guard let maxResidentBytes = intValue(limits["maxResidentBytes"]), maxResidentBytes > 0 else {
            throw NGramOffloadFitCompositionError.planMaxResidentBytesInvalid(path: planFileURL.path)
        }
        return maxResidentBytes
    }

    // MARK: - Safetensors header reading

    /// The cap on the declared safetensors header-length prefix, chosen generously above any real
    /// header this project's checkpoints produce (thousands of tensors easily fit in low single-digit
    /// MiB of JSON) while still rejecting a corrupt/hostile declared length before it drives a huge
    /// allocation.
    static let maxHeaderBytes = 256 * 1024 * 1024

    /// Reads a safetensors file's header ONLY — never the tensor data blob that follows it, which
    /// can be many GiB per shard for this artifact. Format: an 8-byte little-endian `UInt64` header
    /// length `N`, then `N` bytes of UTF-8 JSON mapping tensor name to
    /// `{"dtype":…,"shape":[…],"data_offsets":[begin,end]}` (the `__metadata__` key, when present,
    /// is not a tensor and is skipped). Returns tensor name → `end - begin` byte size.
    static func readSafetensorsTensorByteSizes(at fileURL: URL) throws -> [String: Int] {
        let path = fileURL.path
        guard let handle = FileHandle(forReadingAtPath: path) else {
            throw NGramOffloadFitCompositionError.safetensorsHeaderTruncated(path: path)
        }
        defer { handle.closeFile() }

        guard
            let attributes = try? FileManager.default.attributesOfItem(atPath: path),
            let fileSize = attributes[.size] as? Int
        else {
            throw NGramOffloadFitCompositionError.safetensorsHeaderTruncated(path: path)
        }

        let lengthPrefix = handle.readData(ofLength: 8)
        guard lengthPrefix.count == 8 else {
            throw NGramOffloadFitCompositionError.safetensorsHeaderTruncated(path: path)
        }
        // Decode as explicit little-endian, byte by byte, rather than a raw bit-reinterpretation of
        // the loaded bytes — safetensors declares this prefix little-endian regardless of host
        // byte order, and an implicit reinterpret is exactly the class of bug that has bitten this
        // codebase before (Double(bitPattern:) over raw bytes).
        let lengthBytes = [UInt8](lengthPrefix)
        var headerLength: UInt64 = 0
        for index in 0..<8 {
            headerLength |= UInt64(lengthBytes[index]) << (8 * index)
        }
        guard headerLength > 0, headerLength <= UInt64(maxHeaderBytes),
            headerLength <= UInt64(max(0, fileSize - 8))
        else {
            throw NGramOffloadFitCompositionError.safetensorsHeaderLengthInvalid(path: path)
        }

        let headerData = handle.readData(ofLength: Int(headerLength))
        guard headerData.count == Int(headerLength) else {
            throw NGramOffloadFitCompositionError.safetensorsHeaderTruncated(path: path)
        }
        guard let root = (try? JSONSerialization.jsonObject(with: headerData)) as? [String: Any]
        else {
            throw NGramOffloadFitCompositionError.safetensorsHeaderNotJSON(path: path)
        }

        // The blob region physically present in the file, immediately after the 8-byte length
        // prefix and the header JSON itself — never negative, because `headerLength` was already
        // bounded above by `fileSize - 8`.
        let blobByteCount = fileSize - 8 - Int(headerLength)

        var sizes: [String: Int] = [:]
        for (key, value) in root where key != "__metadata__" {
            guard let entry = value as? [String: Any] else {
                throw NGramOffloadFitCompositionError.safetensorsTensorEntryInvalid(
                    path: path, key: key)
            }
            // DEFECT 1: a declared span that escapes the actual blob region (corruption, an
            // interrupted rewrite, a hand-edited header) must be rejected here rather than trusted
            // at face value — see `NGramOffloadFitCompositionError.safetensorsTensorEntryInvalid`'s
            // use in this exact check, and the type's doc comment on why an over-declared span is
            // the OPTIMISTIC (too-small `adjustedWeights`) failure direction this file exists to
            // prevent.
            guard let offsets = entry["data_offsets"] as? [Any], offsets.count == 2,
                let begin = intValue(offsets[0]), let end = intValue(offsets[1]), end >= begin,
                begin >= 0, end <= blobByteCount
            else {
                throw NGramOffloadFitCompositionError.safetensorsTensorEntryInvalid(
                    path: path, key: key)
            }
            sizes[key] = end - begin
        }
        return sizes
    }

    // MARK: - Key matching

    /// Matches `key` against every supported text-module prefix followed by
    /// `layers.<layerIndex>.ple.ple_embedding.ngram_embedding.shard_<shardIndex>.<field>`, returning
    /// the parsed components on a match or `nil` for any other key (including near-misses that
    /// share a prefix but differ anywhere in the fixed literal segments).
    static func matchOffloadedNGramShardKey(
        _ key: String
    ) -> (layerIndex: Int, shardIndex: Int, field: String)? {
        for prefix in textModulePrefixes where key.hasPrefix(prefix) {
            let afterPrefix = key.dropFirst(prefix.count)
            guard afterPrefix.hasPrefix("layers.") else { continue }
            let afterLayers = afterPrefix.dropFirst("layers.".count)
            guard let layerDot = afterLayers.firstIndex(of: ".") else { continue }
            guard let layerIndex = Int(afterLayers[afterLayers.startIndex..<layerDot]) else {
                continue
            }
            let afterLayerIndex = afterLayers[afterLayers.index(after: layerDot)...]
            let marker = "ple.ple_embedding.ngram_embedding.shard_"
            guard afterLayerIndex.hasPrefix(marker) else { continue }
            let afterMarker = afterLayerIndex.dropFirst(marker.count)
            guard let shardDot = afterMarker.firstIndex(of: ".") else { continue }
            guard let shardIndex = Int(afterMarker[afterMarker.startIndex..<shardDot]) else {
                continue
            }
            let field = String(afterMarker[afterMarker.index(after: shardDot)...])
            guard shardFieldNames.contains(field) else { continue }
            return (layerIndex, shardIndex, field)
        }
        return nil
    }

    // MARK: - Structural audit

    /// Fails closed unless `matches` (in aggregate, across every scanned shard) describes exactly
    /// one coherent offloaded n-gram table: at least one match; every match sharing one layer
    /// index; shard indices forming a contiguous `0..<n` run; every shard carrying the identical
    /// field set; and that field set's size being 1 (dense) or 3 (quantized). See the error type's
    /// per-case doc comments for why each check exists — collectively, if the checkpoint-key prefix
    /// this matcher is built from is ever wrong, it silently matches nothing (or, in principle, the
    /// wrong subset), and this audit is the only thing positioned to catch that before a fit
    /// decision silently under-counts the resident footprint.
    static func auditMatches(_ matches: [MatchedShardKey]) throws {
        guard !matches.isEmpty else {
            throw NGramOffloadFitCompositionError.noMatchingOffloadedShardKeys
        }

        let layerIndices = Set(matches.map(\.layerIndex))
        guard layerIndices.count == 1 else {
            throw NGramOffloadFitCompositionError.multipleLayerIndicesInOffloadedShardKeys(
                indices: layerIndices.sorted())
        }

        var fieldsByShard: [Int: Set<String>] = [:]
        for match in matches {
            fieldsByShard[match.shardIndex, default: []].insert(match.field)
        }
        let shardIndices = Set(fieldsByShard.keys)
        let expectedShardIndices = Set(0..<shardIndices.count)
        guard shardIndices == expectedShardIndices else {
            throw NGramOffloadFitCompositionError.nonContiguousShardIndices(
                shardCount: shardIndices.count)
        }

        guard let firstShardIndex = shardIndices.min(), let firstFields = fieldsByShard[firstShardIndex]
        else {
            throw NGramOffloadFitCompositionError.noMatchingOffloadedShardKeys
        }
        for (_, fields) in fieldsByShard {
            guard fields == firstFields else {
                throw NGramOffloadFitCompositionError.inconsistentShardFieldSets
            }
        }
        guard firstFields.count == 1 || firstFields.count == 3 else {
            throw NGramOffloadFitCompositionError.invalidPerShardFieldCount(count: firstFields.count)
        }
    }

    // MARK: - JSON helpers

    /// Extracts an `Int` from a JSON-decoded value, accepting the `NSNumber`/`Int`/`Double`
    /// representations `JSONSerialization` may hand back. Fails closed (`nil`) on a non-integral or
    /// out-of-`Int`-range value rather than truncating or clamping into garbage: plain
    /// `NSNumber.intValue` / `Int(_ d:)` on a value like `1e300` silently clamp to `Int.max` on this
    /// platform (verified directly) rather than signaling that the declared number was never a
    /// representable byte count — a clamped-but-still-positive value would sail past every
    /// downstream `> 0` check and substitute a wrong figure into the fit formula instead of
    /// refusing. `Int(exactly:)` rejects those. Mirrors `ModelConfigDecoder.intOf`'s identical
    /// precedent in this codebase.
    static func intValue(_ any: Any?) -> Int? {
        if let intValue = any as? Int { return intValue }
        if let number = any as? NSNumber { return Int(exactly: number.doubleValue) }
        if let doubleValue = any as? Double { return Int(exactly: doubleValue) }
        return nil
    }
}
