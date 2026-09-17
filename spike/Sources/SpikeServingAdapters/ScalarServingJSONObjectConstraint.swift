import Foundation

import MLX
import MLXLMCommon
import ServingCore
import SpikeCore

/// The loaded model's `vocab_size` from `config.json`, or `nil` if it is missing/unreadable/not an
/// integer. This is the dimension the model's OWN `lm_head` projects onto — the size
/// `ByteLevelTokenBytes.classify` must be sized to (NOT the tokenizer vocabulary's own id range,
/// which can be smaller than the padded `lm_head` output some checkpoints ship). An id beyond the
/// tokenizer's real vocabulary but within this size classifies `.banned` (no vocab string), which
/// is the correct outcome for embedding padding. Mirrors this file's sibling loader functions'
/// config.json-without-loading-weights idiom (`MLXScalarServing.swift`).
///
/// Checks `text_config.vocab_size` BEFORE the root field: a VL-wrapped hybrid checkpoint ships its
/// language-model geometry (including `vocab_size`) nested under `text_config`, with the root-level
/// field either absent or describing an unrelated (e.g. vision) dimension — the same
/// `text_config`-then-root fallback `DenseContinuousBatchRuntime`'s own hybrid-checkpoint config
/// derivation already uses for `max_position_embeddings`/`vocab_size` on that route.
func scalarServingModelVocabSize(modelDirectory: URL) -> Int? {
    let configURL = modelDirectory.appendingPathComponent("config.json")
    guard let data = try? Data(contentsOf: configURL),
        let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    else {
        return nil
    }
    let textScope = (root["text_config"] as? [String: Any]) ?? [:]
    return (textScope["vocab_size"] as? NSNumber)?.intValue ?? (root["vocab_size"] as? NSNumber)?.intValue
}

/// Load-time result of attempting to enable `response_format: json_object` for the plain scalar
/// (non-speculative) serving route. Built once at model load by
/// `loadScalarServingJSONObjectConstraintSupport` and stored on
/// `ScalarServingBackendConfiguration`; NEVER lets model loading fail — any problem (missing/
/// malformed tokenizer.json, non-byte-level decoder, no resolvable EOS, or a self-check mismatch)
/// simply disables the feature (one logged line) rather than blocking serving. See that function's
/// doc comment for the full disablement enumeration.
public final class ScalarServingJSONObjectConstraintSupport: @unchecked Sendable {
    /// The `</think>` token id, resolved by `resolveThinkEndTokenID` (see that function's doc
    /// comment for why `tokenizer.convertTokenToId` alone cannot be trusted) — the phase-switch
    /// trigger `JSONObjectMaskingLogitProcessor` activates on for a request that separates
    /// reasoning. `nil` for a tokenizer with no such token (a non-thinking model, most commonly):
    /// `ScalarServingBackend.start` refuses `json_object` only for a THINKING request against a
    /// `nil` value here, never for a non-thinking one. Also what a sibling
    /// `ScalarServingJSONSchemaConstraintSupport` reports for `json_schema`'s OWN thinking-phase
    /// gate — see `loadScalarServingJSONSchemaConstraintSupport`'s doc comment for why json_schema
    /// shares this value rather than resolving it a second time.
    public let thinkEndTokenID: Int?

    /// Started at `init` (i.e. as soon as this support object exists, which only happens on the
    /// SUCCESS path of `loadScalarServingJSONObjectConstraintSupport` — see that function's doc
    /// comment), on a detached background task, NOT lazily on first request: the trie walk over a
    /// ~250k-token vocab is expensive (see the response-format design's "Cost" section), and a
    /// lazy first-access build previously ran ON `ScalarServingBackend`'s actor inside `start()`,
    /// stalling every other in-flight request on that actor for the build's duration. `table`
    /// below awaits this SAME task's `.value` — a suspension, not a blocking lock acquisition — so
    /// a caller on an actor (like `start()`) yields the actor's executor while waiting rather than
    /// occupying its thread.
    ///
    /// Builds BOTH the `json_object` table and a sibling `json_schema` table from ONE
    /// `ServingCore.makeSharedVocabConstraintTables` call (stage 3b requirement: one trie build per
    /// loaded model, shared by both response-format kinds — see that function's doc comment) rather
    /// than two independent `JSONObjectConstraintTable(classifications:)` calls. A sibling
    /// `ScalarServingJSONSchemaConstraintSupport` (built via
    /// `loadScalarServingJSONSchemaConstraintSupport(sharing:)`) reads `schemaTable` below to reuse
    /// this SAME background build instead of re-parsing tokenizer.json / rebuilding the trie a
    /// second time for the same checkpoint.
    private let sharedTablesTask: Task<
        (object: JSONObjectConstraintTable, schema: JSONSchemaConstraintTable), Never
    >

    init(classifications: [TokenByteClassification], thinkEndTokenID: Int?) {
        self.thinkEndTokenID = thinkEndTokenID
        self.sharedTablesTask = Task.detached(priority: .utility) {
            makeSharedVocabConstraintTables(classifications: classifications)
        }
    }

    /// The shared byte-trie constraint table, shared read-only across every `json_object` request
    /// on this backend. `await`-ing this NEVER runs the trie build on the caller's own executor
    /// (see `sharedTablesTask`'s doc comment): if the background build already finished (the common
    /// case, since it started at load time, well before the first request), this returns
    /// immediately; otherwise the caller suspends until it does.
    public var table: JSONObjectConstraintTable {
        get async { await sharedTablesTask.value.object }
    }

    /// The SAME background build's `JSONSchemaConstraintTable` half — internal (not `public`):
    /// reached only via `loadScalarServingJSONSchemaConstraintSupport(sharing:)`, never directly by
    /// a request-serving call site (those go through `ScalarServingJSONSchemaConstraintSupport
    /// .table`).
    var schemaTable: JSONSchemaConstraintTable {
        get async { await sharedTablesTask.value.schema }
    }
}

/// Load-time result of attempting to enable `response_format: json_schema` for the plain scalar
/// serving route. Derived STRICTLY from a sibling `ScalarServingJSONObjectConstraintSupport`
/// (`loadScalarServingJSONSchemaConstraintSupport(sharing:)`) rather than independently re-parsing
/// tokenizer.json: `json_schema`'s eligibility prerequisites (byte-level tokenizer, resolvable EOS,
/// resolvable model vocab size, the BOM self-check) are IDENTICAL to `json_object`'s — see
/// `loadScalarServingJSONObjectConstraintSupport`'s disablement enumeration — so whatever disables
/// one disables the other, and there is nothing for a second, independent check to discover.
/// Reusing the sibling's already-started detached background build
/// (`ScalarServingJSONObjectConstraintSupport.schemaTable`) also means this checkpoint's byte trie
/// is built exactly ONCE for BOTH formats (`ServingCore.makeSharedVocabConstraintTables`), never
/// twice.
public final class ScalarServingJSONSchemaConstraintSupport: @unchecked Sendable {
    /// Same value, same phase-switch role, as `ScalarServingJSONObjectConstraintSupport
    /// .thinkEndTokenID` (see that property's doc comment) — copied from the sibling support at
    /// construction rather than re-resolved.
    public let thinkEndTokenID: Int?
    private let objectSupport: ScalarServingJSONObjectConstraintSupport

    init(objectSupport: ScalarServingJSONObjectConstraintSupport) {
        self.thinkEndTokenID = objectSupport.thinkEndTokenID
        self.objectSupport = objectSupport
    }

    /// The shared byte-trie constraint table backing every `json_schema` request against this
    /// checkpoint, independent of any one request's OWN schema (`JSONSchemaConstraintTable` is
    /// per-model, not per-schema — see that type's doc comment). Awaits the SAME detached build
    /// `ScalarServingJSONObjectConstraintSupport.table` awaits (never re-triggers it).
    public var table: JSONSchemaConstraintTable {
        get async { await objectSupport.schemaTable }
    }
}

/// Builds `ScalarServingJSONSchemaConstraintSupport` from an already-built (possibly `nil`)
/// `ScalarServingJSONObjectConstraintSupport` — see that type's doc comment for why `json_schema`
/// has no independent disablement check of its own. `nil` in, `nil` out, with a matching machine-
/// readable disablement line (`fastmlx-serve response_format=json_schema support=disabled
/// reason=...`), mirroring every OTHER disablement reason's `key=value` shape.
public func loadScalarServingJSONSchemaConstraintSupport(
    sharing objectSupport: ScalarServingJSONObjectConstraintSupport?
) -> ScalarServingJSONSchemaConstraintSupport? {
    guard let objectSupport else {
        print(
            "fastmlx-serve response_format=json_schema support=disabled "
                + "reason=json_object_support_unavailable")
        return nil
    }
    return ScalarServingJSONSchemaConstraintSupport(objectSupport: objectSupport)
}

/// Attempts to build `ScalarServingJSONObjectConstraintSupport` for the checkpoint at
/// `modelDirectory`. NEVER throws: every failure mode is reported as a single machine-readable
/// stdout line (`fastmlx-serve response_format=json_object support=disabled reason=...`, matching
/// this file's sibling loader lines' `key=value` convention) and a `nil` return, so a caller can
/// simply treat `nil` as "feature unsupported for this checkpoint" without a `do`/`catch`.
///
/// Disablement reasons, in the order checked:
/// 1. `tokenizer_json_unreadable` — `tokenizer.json` is missing or unreadable.
/// 2. `tokenizer_json_malformed` — present but `ByteLevelTokenizerDescriptor.parse` cannot recover
///    a vocabulary from it.
/// 3. `non_byte_level_tokenizer` — the decoder is not `ByteLevel` (or a `Sequence` containing one).
/// 4. `no_eos_token_id` — neither the tokenizer's own `eosTokenId` nor `generation_config.json`'s
///    `eos_token_id` resolves to anything: the automaton could never legally terminate.
/// 5. `model_vocab_size_unresolved` — `config.json`'s `vocab_size` is missing/unreadable (see
///    `scalarServingModelVocabSize`'s doc comment for why the constraint needs it specifically,
///    not the tokenizer's own vocabulary size).
/// 6. `eos_out_of_range` — every resolved EOS id is `>= vocabSize`, so none of them could ever
///    classify as `.eos` in `ByteLevelTokenBytes.classify` (which only classifies `0..<vocabSize`):
///    the automaton would then have no legal way to terminate, identically to `no_eos_token_id`.
/// 7. `self_check_mismatch` — `ByteLevelTokenBytes.excludingKnownBOMArtifacts` found at least one id
///    whose recovered bytes disagree with the tokenizer's own `decode([id])` and that disagreement
///    is NOT the known leading-BOM-stripping artifact (see that function's doc comment: Foundation's
///    `String(bytes:encoding:.utf8)` silently drops a leading U+FEFF that a real byte-level
///    `decode([id])` keeps, which would otherwise false-positive on any vocab id whose raw bytes are
///    or start with the UTF-8 BOM). That one recognized shape is excluded and its id reclassified
///    `.banned` rather than disabling the feature; every other mismatch still disables it. The log
///    line also carries the FIRST mismatching id, so a live disablement is diagnosable from stdout
///    alone.
public func loadScalarServingJSONObjectConstraintSupport(
    modelDirectory: URL,
    tokenizer: any MLXLMCommon.Tokenizer
) -> ScalarServingJSONObjectConstraintSupport? {
    func disabled(_ reason: String, detail: String? = nil) -> ScalarServingJSONObjectConstraintSupport? {
        if let detail {
            print("fastmlx-serve response_format=json_object support=disabled reason=\(reason) \(detail)")
        } else {
            print("fastmlx-serve response_format=json_object support=disabled reason=\(reason)")
        }
        return nil
    }

    let tokenizerJSONURL = modelDirectory.appendingPathComponent("tokenizer.json")
    guard let tokenizerJSONData = try? Data(contentsOf: tokenizerJSONURL) else {
        return disabled("tokenizer_json_unreadable")
    }
    let descriptor: ByteLevelTokenizerDescriptor
    do {
        descriptor = try ByteLevelTokenizerDescriptor.parse(tokenizerJSON: tokenizerJSONData)
    } catch {
        return disabled("tokenizer_json_malformed")
    }
    guard descriptor.isByteLevel else {
        return disabled("non_byte_level_tokenizer")
    }

    var eosTokenIds = Set<Int>()
    if let tokenizerEOS = tokenizer.eosTokenId {
        eosTokenIds.insert(tokenizerEOS)
    }
    let generationConfigURL = modelDirectory.appendingPathComponent("generation_config.json")
    if let generationConfigData = try? Data(contentsOf: generationConfigURL) {
        eosTokenIds.formUnion(
            GenerationConfigEOSIds.parse(generationConfigJSON: generationConfigData))
    }
    guard !eosTokenIds.isEmpty else {
        return disabled("no_eos_token_id")
    }

    guard let vocabSize = scalarServingModelVocabSize(modelDirectory: modelDirectory) else {
        return disabled("model_vocab_size_unresolved")
    }

    guard eosTokenIds.contains(where: { $0 < vocabSize }) else {
        return disabled("eos_out_of_range")
    }

    let unfilteredClassifications = ByteLevelTokenBytes.classify(
        vocabSize: vocabSize,
        vocabString: { descriptor.vocabStringsByID[$0] },
        addedTokenIds: descriptor.addedTokenIds,
        eosTokenIds: eosTokenIds)

    let selfCheck = ByteLevelTokenBytes.excludingKnownBOMArtifacts(
        classifications: unfilteredClassifications
    ) { id in
        tokenizer.decode(tokenIds: [id], skipSpecialTokens: false)
    }
    guard selfCheck.mismatches.isEmpty else {
        return disabled("self_check_mismatch", detail: "id=\(selfCheck.mismatches[0].id)")
    }
    let classifications = selfCheck.classifications

    // `</think>` need not exist in every tokenizer's vocabulary — a non-thinking model's tokenizer
    // legitimately has no such token. `nil` here is not itself a disablement; see
    // `thinkEndTokenID`'s doc comment for who refuses on it, and when.
    let thinkEndTokenID = resolveThinkEndTokenID(descriptor: descriptor, tokenizer: tokenizer)

    return ScalarServingJSONObjectConstraintSupport(
        classifications: classifications, thinkEndTokenID: thinkEndTokenID)
}

/// Resolves the `</think>` token id without trusting `Tokenizer.convertTokenToId` on its own: a
/// BPE tokenizer conventionally returns its UNKNOWN-token id (never `nil`) for a string it cannot
/// resolve, so a naive `convertTokenToId("</think>")` call can silently return the WRONG id for a
/// tokenizer that has no such token at all. Accepted only when EITHER:
/// - `tokenizer.convertIdToToken` round-trips the candidate id back to exactly `"</think>"` (rules
///   out the UNKNOWN-id collision above), OR
/// - `tokenizer.json`'s own `added_tokens` table has an entry whose `content` is exactly
///   `"</think>"` (covers a tokenizer whose `convertIdToToken` does not surface added-token text).
/// `nil` in every other case, including when `convertTokenToId` itself returns `nil`.
func resolveThinkEndTokenID(
    descriptor: ByteLevelTokenizerDescriptor,
    tokenizer: any MLXLMCommon.Tokenizer
) -> Int? {
    if let addedTokenID = descriptor.addedTokenIDsByContent["</think>"] {
        return addedTokenID
    }
    guard let candidate = tokenizer.convertTokenToId("</think>") else {
        return nil
    }
    guard tokenizer.convertIdToToken(candidate) == "</think>" else {
        return nil
    }
    return candidate
}

/// Applies a `TokenConstraint`'s grammar mask AFTER any penalty processor `MLXDecoder.setPenalties`
/// already composed — `SpikeCore.MLXDecoder` enforces that ordering itself via its OWN separate
/// `constraintProcessor` slot (`setResponseFormatConstraint`), applied strictly after `processor`
/// (penalties) inside `selectSampleAndAdvance`, so THIS type never wraps or needs to know about the
/// penalty processor.
///
/// GENERIC over `C: TokenConstraint` (stage 3b) so `response_format: json_object` and
/// `response_format: json_schema` share ONE masking/thinking-phase-gate/failure-reporting
/// implementation instead of two copies that could drift apart — `JSONObjectMaskingLogitProcessor`
/// below is a thin, API-preserving wrapper around `ConstraintMaskingLogitProcessor<
/// JSONObjectTokenConstraint>`; `response_format: json_schema` (`ScalarServingBackend.start`)
/// constructs `ConstraintMaskingLogitProcessor<JSONSchemaTokenConstraint>` (aliased
/// `JSONSchemaMaskingLogitProcessor` below) directly, with no wrapper needed since it has no
/// pre-existing call sites to preserve.
///
/// A CLASS (not a struct): `SpikeCore.ConstraintProcessorFailureReporting` is a class-constrained
/// protocol specifically so `MLXDecoder` can read `recordedFailure` through an `as?` cast on the
/// SAME instance it just called `didSample` on — see that protocol's doc comment.
///
/// `@unchecked Sendable`: one instance is built fresh per request (`ScalarServingBackend.start`)
/// and stored on that request's `Sendable` `PendingRequest` before being handed, exactly once,
/// into `InferenceActor.generateBounded` — from then on ONLY that actor's single-owner decode loop
/// ever calls `process`/`didSample` on it (mirrors `JSONObjectConstraintTable`'s own `@unchecked
/// Sendable` justification: every mutation happens from a single, non-overlapping call sequence,
/// never concurrently).
public final class ConstraintMaskingLogitProcessor<C: TokenConstraint>: LogitProcessor,
    ConstraintProcessorFailureReporting, @unchecked Sendable
{
    private var constraint: C
    /// When `false`, `process(logits:)` is a no-op passthrough and `didSample` only watches for
    /// `thinkEndTokenID` — the thinking-phase-gate contract (response-format design item #4):
    /// active from the first token for a request that does NOT separate reasoning; inactive until
    /// `thinkEndTokenID` is sampled for one that does.
    private var active: Bool
    private let thinkEndTokenID: Int?
    public private(set) var recordedFailure: Error?

    /// - Parameters:
    ///   - activeFromStart: `true` for a request that does not separate reasoning (mask applies to
    ///     every token from position 0); `false` for a thinking request (mask stays inactive until
    ///     `thinkEndTokenID` is sampled). Callers must not pass `false` with a `nil`
    ///     `thinkEndTokenID` — see `ScalarServingBackend`'s admission-time refusal for that
    ///     combination, which this type does not itself re-check (it would simply never activate).
    public init(constraint: C, activeFromStart: Bool, thinkEndTokenID: Int?) {
        self.constraint = constraint
        self.active = activeFromStart
        self.thinkEndTokenID = thinkEndTokenID
    }

    public func prompt(_ prompt: MLXArray) {}

    public func process(logits: MLXArray) -> MLXArray {
        guard active, recordedFailure == nil else {
            return logits
        }
        let bitset: [UInt64]
        do {
            bitset = try constraint.allowedTokenBitset()
        } catch {
            recordedFailure = error
            return logits
        }

        let vocab = logits.dim(-1)
        // Build the -inf mask array directly from the bitset's SET bits (word-scan + trailing-zero
        // bit extraction), never materializing `allowedTokenIds()`'s `[Int]` list — see
        // `JSONObjectConstraintTable`'s doc comment for why that list can run to hundreds of
        // thousands of entries at a state deep inside a JSON string.
        var maskValues = [Float](repeating: -Float.infinity, count: vocab)
        for (wordIndex, word) in bitset.enumerated() {
            guard word != 0 else { continue }
            var bits = word
            while bits != 0 {
                let bit = bits.trailingZeroBitCount
                let id = wordIndex * 64 + bit
                if id < vocab {
                    maskValues[id] = 0
                }
                bits &= bits - 1
            }
        }
        let mask = MLXArray(maskValues).reshaped([1, vocab]).asType(logits.dtype)
        return logits + mask
    }

    public func didSample(token: MLXArray) {
        guard recordedFailure == nil else {
            return
        }
        let id = token.item(Int.self)
        guard active else {
            if let thinkEndTokenID, id == thinkEndTokenID {
                active = true
            }
            return
        }
        do {
            try constraint.advance(token: id)
        } catch {
            recordedFailure = error
        }
    }
}

/// `response_format: json_schema`'s masking processor — a plain specialization of
/// `ConstraintMaskingLogitProcessor`, with no wrapper needed (unlike `JSONObjectMaskingLogitProcessor`
/// below, this type has no pre-3b call sites/tests to keep source-compatible).
public typealias JSONSchemaMaskingLogitProcessor = ConstraintMaskingLogitProcessor<JSONSchemaTokenConstraint>

/// `response_format: json_object`'s masking processor. A THIN WRAPPER around
/// `ConstraintMaskingLogitProcessor<JSONObjectTokenConstraint>` (stage 3b): every method/property
/// below simply forwards to `inner`. Kept as its own concrete type — rather than a bare typealias,
/// unlike `JSONSchemaMaskingLogitProcessor` above — SOLELY to preserve this type's pre-existing
/// `init(table:activeFromStart:thinkEndTokenID:)` label (`table:`, not the generic processor's
/// `constraint:`), which every call site and test predating stage 3b (`ScalarServingBackend.start`,
/// `JSONObjectMaskingLogitProcessorTests`, `ScalarServingJSONObjectRequestWiringTests`) already
/// uses; a bare typealias cannot add its own initializer with a different argument label.
public final class JSONObjectMaskingLogitProcessor: LogitProcessor, ConstraintProcessorFailureReporting,
    @unchecked Sendable
{
    private let inner: ConstraintMaskingLogitProcessor<JSONObjectTokenConstraint>

    public init(table: JSONObjectConstraintTable, activeFromStart: Bool, thinkEndTokenID: Int?) {
        self.inner = ConstraintMaskingLogitProcessor(
            constraint: JSONObjectTokenConstraint(table: table),
            activeFromStart: activeFromStart,
            thinkEndTokenID: thinkEndTokenID)
    }

    public var recordedFailure: Error? { inner.recordedFailure }
    public func prompt(_ prompt: MLXArray) { inner.prompt(prompt) }
    public func process(logits: MLXArray) -> MLXArray { inner.process(logits: logits) }
    public func didSample(token: MLXArray) { inner.didSample(token: token) }
}
