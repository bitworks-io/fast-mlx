import Foundation

/// A compiled `response_format: {"type":"json_schema", ...}` request (response-format slice 2).
///
/// Only a SUBSET of JSON Schema is representable (see `JSONSchemaNode`). A request schema that uses
/// anything outside the subset is refused with a 400 at decode time, never approximated: the server
/// must not return output the caller's validator would reject. Every instance the constraint
/// automaton admits must validate against the caller's ORIGINAL schema.
public struct JSONSchemaResponseFormat: Sendable, Equatable {
    /// `json_schema.name`, validated against `^[A-Za-z0-9_-]{1,64}$`.
    public var name: String
    /// `json_schema.strict` (absent ⇒ `false`). `strict:true` additionally requires every object to
    /// declare `additionalProperties:false` and list every property in `required`.
    public var strict: Bool
    /// The compiled schema, with every local `$ref` already inlined.
    public var root: JSONSchemaNode
    /// Lowercase hex SHA-256 of `root`'s canonical binary encoding (`JSONSchemaNode.canonicalEncoding()`
    /// in `JSONSchemaFingerprint.swift`) — the COMPILED tree, not the caller's raw schema text. Two
    /// schemas share a fingerprint if and only if they compile to the identical `JSONSchemaNode`,
    /// which is order-sensitive on `.object`'s declared property order (matching what the constraint
    /// automaton itself enforces), so two schemas differing only in `properties` order never collide.
    /// Annotation-only differences (`title`, `description`, ...), raw sibling-key order, and an
    /// inlined `$ref` vs. its resolved target all still share a fingerprint when they compile to the
    /// same tree. Mask-cache keys include it so two DIFFERENT compiled schemas never share entries —
    /// see `JSONSchemaAutomaton.MaskCacheKey`.
    public var fingerprint: String

    public init(name: String, strict: Bool, root: JSONSchemaNode, fingerprint: String) {
        self.name = name
        self.strict = strict
        self.root = root
        self.fingerprint = fingerprint
    }
}

/// The supported JSON Schema subset, as a closed tree (no references, no cycles).
///
/// Generation semantics the automaton must enforce, and that the compiler must only emit when they
/// are sound with respect to the caller's schema:
/// - `.object`: only the listed properties may appear, each at most once and in DECLARED order;
///   an optional property may be skipped, a required one may not. No other keys
///   (additionalProperties is always false). At most `JSONSchemaLimits.maxPropertiesPerObject`.
/// - `.array`: zero or more `items`; no length bounds.
/// - `.integer`: a JSON number with no fraction and no exponent.
/// - `.enumeration`: exactly one of the literal byte strings (already canonical JSON text).
/// - `.anyOf`: the branches' FIRST value bytes are pairwise disjoint (the compiler guarantees it), so
///   the automaton commits to a branch on the first byte and never backtracks.
public indirect enum JSONSchemaNode: Sendable, Hashable {
    case string
    case number
    case integer
    case boolean
    case null
    case enumeration([JSONSchemaLiteral])
    case object([JSONSchemaProperty])
    case array(items: JSONSchemaNode)
    case anyOf([JSONSchemaNode])
}

/// Compile-time bounds; a schema exceeding any of them is refused with a 400.
public enum JSONSchemaLimits {
    public static let maxNodes = 512
    public static let maxDepth = 32
    public static let maxPropertiesPerObject = 64
    public static let maxEnumValues = 256
    public static let maxAnyOfBranches = 8
    /// Total bytes across every property name and enum literal.
    public static let maxLiteralBytes = 64 * 1024
}

/// One `enum` value, stored as its canonical JSON text bytes (e.g. `"red"` with quotes, `3`, `true`,
/// `null`). Strings use the minimal escaping `JSONSerialization` would not change: the compiler
/// refuses enum strings containing control characters, `"` or `\`, so the literal bytes are the
/// value's UTF-8 bytes wrapped in quotes and the automaton needs no escape handling for them.
public struct JSONSchemaLiteral: Sendable, Hashable {
    public var jsonText: [UInt8]

    public init(jsonText: [UInt8]) {
        self.jsonText = jsonText
    }
}

/// One object property, in the order the caller declared it.
public struct JSONSchemaProperty: Sendable, Hashable {
    /// The key's UTF-8 bytes (the compiler refuses keys containing control characters, `"` or `\`).
    public var name: String
    public var required: Bool
    public var value: JSONSchemaNode

    public init(name: String, required: Bool, value: JSONSchemaNode) {
        self.name = name
        self.required = required
        self.value = value
    }
}
