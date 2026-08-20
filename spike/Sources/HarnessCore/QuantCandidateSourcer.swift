import Foundation

/// One enumerated quant variant of a base model: the fetchable repo id, its quantization bit width
/// (`nil` = unquantized, e.g. `bf16`), and whether it is a DWQ (Distilled Weight Quantization) build.
/// `isDWQ` is a same-bit-width quality signal: at equal `quantBits`, a DWQ build recovers accuracy the
/// naive quantization loses (mlx-lm LEARNED_QUANTS.md), so it is emitted BEFORE the plain build and can
/// serve as a future equal-bits tiebreak for the picker (not wired into ranking yet).
public struct QuantVariant: Equatable, Sendable {
    public let repoID: String
    public let quantBits: Int?
    public let isDWQ: Bool

    public init(repoID: String, quantBits: Int?, isDWQ: Bool) {
        self.repoID = repoID
        self.quantBits = quantBits
        self.isDWQ = isDWQ
    }
}

/// Candidate SOURCING for quant auto-pick (fit-checked-serve, differentiator #2 full shape — policy
/// decision #3 in docs/task-inbox/2026-08-18-quant-auto-pick-policy.md): given a base model repo id,
/// enumerate the ordered mlx-community quant-variant repo ids to probe. This is ONLY the offline name
/// generator; the network metadata-probe half (which of these actually exist) is a deliberately deferred
/// follow-on. Over-generating is safe by design — a pattern that does not exist costs a harmless 404 at
/// probe time, never a fatal (the picker already carries an excluded-with-reason path).
///
/// **Naming conventions** are grounded in the live mlx-community HF org (verified 2026-08-19): a single
/// base model coexists as `{base}-Nbit` (the mlx-lm convert default), `{base}-Nbit-DWQ`, and
/// `{base}-bf16`. The enumeration emits, per width, the DWQ build before the plain build; the fit-check
/// + `QuantAutoPicker` re-rank the surviving candidates by actual fit and the operator's `--prefer`
/// axis, so this order is only a defensible default probe order + the equal-bits DWQ tiebreak.
public enum QuantCandidateSourcer {

    /// Quantization widths enumerated, highest fidelity first. DWQ is emitted before plain within each.
    private static let widths = [8, 6, 4, 3]
    /// Default HF org for a bare (unqualified) base name.
    public static let defaultOrg = "mlx-community"

    /// The ordered repo ids to probe for `baseRepoID`. See `variants(baseRepoID:)` for the structured
    /// form; this is `variants(...).map(\.repoID)`.
    public static func enumerate(baseRepoID: String, org: String = defaultOrg) -> [String] {
        variants(baseRepoID: baseRepoID, org: org).map(\.repoID)
    }

    /// The ordered, de-duplicated variant set for `baseRepoID`. If the input ALREADY names a specific
    /// quant/precision variant (e.g. `…-4bit`, `…-4bit-DWQ`, `…-bf16`), it is returned as itself only —
    /// never re-suffixed into `…-4bit-4bit`. Otherwise the base is qualified under `org` (unless it
    /// already carries an org prefix) and expanded into the canonical variant set.
    public static func variants(baseRepoID: String, org: String = defaultOrg) -> [QuantVariant] {
        let (resolvedOrg, name) = splitOrg(baseRepoID, defaultOrg: org)

        // Already a specific variant → terminal, return it as-is (parsed).
        if let terminal = parseTerminalVariant(org: resolvedOrg, name: name) {
            return [terminal]
        }

        let base = "\(resolvedOrg)/\(name)"
        var out: [QuantVariant] = []
        var seen: Set<String> = []
        func add(_ v: QuantVariant) {
            guard seen.insert(v.repoID).inserted else { return }
            out.append(v)
        }
        // Unquantized (bf16) is the highest fidelity; probe it first.
        add(QuantVariant(repoID: "\(base)-bf16", quantBits: nil, isDWQ: false))
        for width in widths {
            add(QuantVariant(repoID: "\(base)-\(width)bit-DWQ", quantBits: width, isDWQ: true))
            add(QuantVariant(repoID: "\(base)-\(width)bit", quantBits: width, isDWQ: false))
        }
        return out
    }

    /// Whether `repoID` denotes a DWQ (Distilled Weight Quantization) build — the single DWQ predicate
    /// shared by the enumeration (`parseTerminalVariant`) and the picker's equal-bits tiebreak, so the
    /// two never drift. Scoped to the LAST path component so it works for both an HF repo id
    /// (`mlx-community/Qwen3-8B-4bit-DWQ`) and a local checkpoint directory the resolver passes as an
    /// absolute path (`/models/Qwen3-8B-4bit-DWQ`) — a raw substring test would false-positive on a
    /// parent directory like `~/models/…`.
    public static func denotesDWQ(repoID: String) -> Bool {
        lastComponent(repoID).lowercased().contains("dwq")
    }

    /// The final path/repo component: everything after the last `/` (the checkpoint name), or the whole
    /// string when it carries no slash.
    private static func lastComponent(_ repoID: String) -> String {
        guard let slash = repoID.lastIndex(of: "/") else { return repoID }
        return String(repoID[repoID.index(after: slash)...])
    }

    /// Split an input repo id into (org, name). An input containing `/` keeps its own org (everything
    /// before the FIRST slash); a bare name takes `defaultOrg`. HF repo ids are `org/name`.
    private static func splitOrg(_ repoID: String, defaultOrg: String) -> (org: String, name: String) {
        guard let slash = repoID.firstIndex(of: "/") else { return (defaultOrg, repoID) }
        let org = String(repoID[..<slash])
        let name = String(repoID[repoID.index(after: slash)...])
        return (org, name)
    }

    /// If `name` already carries a quant (`-Nbit`, any case) or half-precision (`-bf16`/`-fp16`) token,
    /// return it as a terminal `QuantVariant` (parsed bits + DWQ flag); otherwise `nil` (it is a base to
    /// expand). Recognizes the case variant `-NBit` that mlx-community also publishes.
    private static func parseTerminalVariant(org: String, name: String) -> QuantVariant? {
        let repoID = "\(org)/\(name)"
        let lower = name.lowercased()
        // -Nbit (case-insensitive): capture the bit width.
        if let match = name.range(of: "-([0-9]+)bit", options: [.regularExpression, .caseInsensitive]) {
            let token = name[match]                       // e.g. "-4bit" / "-4Bit"
            let digits = token.dropFirst().prefix { $0.isNumber }  // strip leading '-', take digits
            let bits = Int(digits)
            return QuantVariant(repoID: repoID, quantBits: bits, isDWQ: denotesDWQ(repoID: name))
        }
        // Unquantized half-precision terminal variant.
        if lower.hasSuffix("-bf16") || lower.hasSuffix("-fp16") {
            return QuantVariant(repoID: repoID, quantBits: nil, isDWQ: false)
        }
        return nil
    }
}
