import Foundation

/// Canonical binary encoding of a compiled `JSONSchemaNode` tree — the ONLY input
/// `JSONSchemaSubsetCompiler`'s fingerprint hash is taken over (see
/// `JSONSchemaResponseFormat.fingerprint`'s doc comment).
///
/// Bug this exists to fix: the fingerprint used to be a SHA-256 of a sorted-keys JSON
/// serialization of the caller's RAW schema — sorted-keys normalization is blind to `properties`
/// DECLARED order, but the compiled IR (`JSONSchemaNode.object`) and the constraint automaton both
/// enforce that order (an object's keys must appear in the order the caller declared them). Two
/// schemas differing only in `properties` order therefore used to compile to the SAME fingerprint
/// but DIFFERENT automata — a masking bug in `JSONSchemaConstraintTable`'s cache, which keys states
/// by fingerprint (see `JSONSchemaAutomaton.MaskCacheKey`), not by `JSONSchemaNode` structural
/// identity. Hashing the COMPILED tree instead — which is what the automaton actually walks — makes
/// two schemas share a fingerprint if and only if they compile to the identical `JSONSchemaNode`
/// (order-sensitive on `.object`, exactly like the automaton), independent of any surface
/// syntax difference (raw key order, annotations like `title`/`description`, `$ref` vs. inlined)
/// that provably compiles to the same tree.
extension JSONSchemaNode {
    /// A deterministic, INJECTIVE binary encoding of this tree: two different `JSONSchemaNode`
    /// values always produce different encodings (see the per-field reasoning below), so hashing
    /// this output can never collide two semantically different compiled schemas.
    ///
    /// Encoding shape — one tag byte per case, followed by that case's fields in a fixed order;
    /// every variable-length field (a byte blob, or a sequence of recursively-encoded children) is
    /// preceded by its own big-endian `UInt32` length. This is a standard length-prefixed (TLV-style)
    /// encoding: because every field's encoded length is determined by a length prefix that precedes
    /// it, and every recursive child's own encoding is in turn self-delimiting by the same argument
    /// (structural induction on tree depth — bounded by `JSONSchemaLimits.maxDepth`), the encoding of
    /// any node can be parsed back into exactly the fields that produced it. Distinct trees can never
    /// share an encoding: a different tag byte alone distinguishes different cases at the same
    /// position; within one case, a different field value changes either a length prefix (making
    /// everything after it mis-align on any attempted re-parse) or the bytes within a length that
    /// stays the same. Two concrete failure modes this specifically guards against, spot-checked in
    /// `ResponseFormatJSONSchemaRequestTests`/`JSONSchemaFingerprintTests`:
    /// - `enum(["ab"])` vs. `enum(["a","b"])`: the LITERAL COUNT prefix (1 vs. 2) differs before any
    ///   literal bytes are compared, so these can never collide even though `"ab"` and `"a"+"b"`
    ///   concatenate to related byte runs.
    /// - `object([a: required])` vs. `object([a: optional])`: the `required` byte (fixed-width, not
    ///   length-prefixed — always exactly one byte) differs at a fixed offset after the property
    ///   name's own length-prefixed bytes.
    /// - `array(items: .string)` vs. `.string`: the tag byte itself (`.array`'s tag vs. `.string`'s
    ///   tag) differs at the very first byte.
    func canonicalEncoding() -> [UInt8] {
        var out: [UInt8] = []
        encode(into: &out)
        return out
    }

    /// Tag bytes — stable and arbitrary; only required to be pairwise distinct across
    /// `JSONSchemaNode`'s cases. Never persisted or compared across builds, so they may be
    /// renumbered freely without any migration concern (the fingerprint is a pure cache key, not a
    /// stored artifact — see `JSONSchemaResponseFormat.fingerprint`'s doc comment).
    private enum Tag: UInt8 {
        case string = 0
        case number = 1
        case integer = 2
        case boolean = 3
        case null = 4
        case enumeration = 5
        case object = 6
        case array = 7
        case anyOf = 8
    }

    private func encode(into out: inout [UInt8]) {
        switch self {
        case .string:
            out.append(Tag.string.rawValue)
        case .number:
            out.append(Tag.number.rawValue)
        case .integer:
            out.append(Tag.integer.rawValue)
        case .boolean:
            out.append(Tag.boolean.rawValue)
        case .null:
            out.append(Tag.null.rawValue)
        case .enumeration(let literals):
            out.append(Tag.enumeration.rawValue)
            Self.appendLength(literals.count, to: &out)
            for literal in literals {
                Self.appendLengthPrefixed(literal.jsonText, to: &out)
            }
        case .object(let properties):
            // Order-sensitive BY CONSTRUCTION: `properties` is encoded in array (declared) order,
            // never sorted — this is the fix's whole point (see this extension's doc comment).
            out.append(Tag.object.rawValue)
            Self.appendLength(properties.count, to: &out)
            for property in properties {
                Self.appendLengthPrefixed(Array(property.name.utf8), to: &out)
                out.append(property.required ? 1 : 0)
                property.value.encode(into: &out)
            }
        case .array(let items):
            out.append(Tag.array.rawValue)
            items.encode(into: &out)
        case .anyOf(let branches):
            out.append(Tag.anyOf.rawValue)
            Self.appendLength(branches.count, to: &out)
            for branch in branches {
                branch.encode(into: &out)
            }
        }
    }

    /// Appends `length` as four big-endian bytes. `JSONSchemaLimits` bounds every count/byte-length
    /// this is used for (nodes, properties-per-object, enum values, anyOf branches, literal bytes)
    /// far below `UInt32.max`, so truncation cannot occur.
    private static func appendLength(_ length: Int, to out: inout [UInt8]) {
        let value = UInt32(length)
        out.append(UInt8((value >> 24) & 0xFF))
        out.append(UInt8((value >> 16) & 0xFF))
        out.append(UInt8((value >> 8) & 0xFF))
        out.append(UInt8(value & 0xFF))
    }

    private static func appendLengthPrefixed(_ bytes: [UInt8], to out: inout [UInt8]) {
        appendLength(bytes.count, to: &out)
        out.append(contentsOf: bytes)
    }
}
