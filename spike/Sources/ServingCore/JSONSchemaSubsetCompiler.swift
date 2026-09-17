import CryptoKit
import Foundation

/// One compile error, naming the JSON-pointer PATH within the caller's `json_schema.schema`
/// document (e.g. `/properties/age/pattern`, `""` for the schema root) and a human-readable
/// message. `OpenAIChatCompletions.swift`'s decoder maps this to
/// `OpenAIServingError.invalidRequest("response_format.json_schema.schema<path>: <message>", ...)`.
public struct JSONSchemaCompileError: Error, Equatable, Sendable {
    public let path: String
    public let message: String

    public init(path: String, message: String) {
        self.path = path
        self.message = message
    }
}

/// Compiles a `response_format.json_schema` request into the frozen IR (`JSONSchemaResponseFormat`
/// / `JSONSchemaNode`, `JSONSchemaResponseFormat.swift`). See
/// `docs/task-inbox/2026-09-17-DECISION-response-format-json-schema-subset.md` for the accepted
/// subset. Every schema the compiler ACCEPTS must be one the constraint automaton can enforce
/// soundly — an instance the automaton admits must validate against the caller's ORIGINAL schema.
/// Anything outside the subset is refused with a 400 naming the offending JSON-pointer path, never
/// approximated.
public enum JSONSchemaSubsetCompiler {

    // MARK: - Order-preserving JSON value tree

    /// A JSON value parsed WITHOUT losing information `JSONSerialization` throws away: declared
    /// object-key order (the IR's `JSONSchemaNode.object` requires DECLARED property order) and the
    /// original number literal text (so `enum` numeric literals reproduce exactly what the caller
    /// wrote, rather than a value round-tripped through `Double`/`NSNumber`).
    public indirect enum RawValue: Sendable {
        case object([(key: String, value: RawValue)])
        case array([RawValue])
        case string(String)
        /// The exact JSON number literal text (e.g. `"3"`, `"-0.5e2"`), unparsed.
        case number(String)
        case bool(Bool)
        case null
    }

    /// Parses `json` (RFC 8259) into a `RawValue`, preserving declared object-key order and
    /// refusing a duplicate key within any single object with a `JSONSchemaCompileError`. Callers
    /// in this codebase only ever invoke this on a request body that `JSONSerialization` has
    /// ALREADY parsed successfully (see `OpenAIChatCompletions.decodeChatResponseFormat`), so a
    /// syntax error here should not occur in practice; it is still handled defensively.
    public static func parseOrdered(_ json: Data) throws -> RawValue {
        var parser = OrderedJSONParser(json)
        return try parser.parseRoot()
    }

    // MARK: - Compilation entry point

    /// Compiles `schema` (already parsed with `parseOrdered`, so declared property order survives)
    /// into `JSONSchemaResponseFormat`. `name`/`strict` are the sibling `json_schema.name`/
    /// `json_schema.strict` fields — the caller has already validated their shape.
    public static func compile(
        name: String,
        schema: RawValue,
        strict: Bool
    ) throws -> JSONSchemaResponseFormat {
        guard case .object(let rootPairs) = schema else {
            throw JSONSchemaCompileError(path: "", message: "schema must be a JSON object")
        }

        var defs: [String: RawValue] = [:]
        var definitions: [String: RawValue] = [:]
        for (key, value) in rootPairs {
            if key == "$defs" {
                guard case .object(let pairs) = value else {
                    throw JSONSchemaCompileError(path: "/$defs", message: "$defs must be an object")
                }
                for (defName, defSchema) in pairs { defs[defName] = defSchema }
            } else if key == "definitions" {
                guard case .object(let pairs) = value else {
                    throw JSONSchemaCompileError(path: "/definitions", message: "definitions must be an object")
                }
                for (defName, defSchema) in pairs { definitions[defName] = defSchema }
            }
        }

        let root = RootContext(defs: defs, definitions: definitions)
        var budget = Budget()
        var refStack: Set<String> = []
        let rootNode = try compileNode(
            schema, path: "", depth: 0, root: root, isRoot: true, strict: strict,
            refStack: &refStack, budget: &budget)

        return JSONSchemaResponseFormat(
            name: name, strict: strict, root: rootNode, fingerprint: fingerprint(of: rootNode))
    }

    // MARK: - Node compilation

    private struct RootContext {
        let defs: [String: RawValue]
        let definitions: [String: RawValue]
    }

    private struct Budget {
        var nodeCount = 0
        var literalBytes = 0
    }

    /// Keys ignored everywhere they appear: pure annotations that never affect the shape of a
    /// generated instance.
    private static let ignoredAnnotationKeys: Set<String> = [
        "title", "description", "$schema", "$id", "default", "examples", "$comment",
    ]
    /// Keys recognized ONLY at the schema root (the `$ref` lookup table).
    private static let rootOnlyKeys: Set<String> = ["$defs", "definitions"]
    private static let validTypeNames: Set<String> = [
        "object", "array", "string", "number", "integer", "boolean", "null",
    ]

    private static func compileNode(
        _ value: RawValue,
        path: String,
        depth: Int,
        root: RootContext,
        isRoot: Bool,
        strict: Bool,
        refStack: inout Set<String>,
        budget: inout Budget
    ) throws -> JSONSchemaNode {
        guard depth <= JSONSchemaLimits.maxDepth else {
            throw JSONSchemaCompileError(path: path, message: "schema exceeds JSONSchemaLimits.maxDepth")
        }
        budget.nodeCount += 1
        guard budget.nodeCount <= JSONSchemaLimits.maxNodes else {
            throw JSONSchemaCompileError(path: path, message: "schema exceeds JSONSchemaLimits.maxNodes")
        }

        guard case .object(let pairs) = value else {
            throw JSONSchemaCompileError(path: path, message: "schema node must be a JSON object")
        }
        var seen: [String: RawValue] = [:]
        for (key, keyValue) in pairs { seen[key] = keyValue }
        let keysPresent = Set(pairs.map(\.key))
        let baseAllowed = ignoredAnnotationKeys.union(isRoot ? rootOnlyKeys : [])

        // Universal recognized-keyword check, BEFORE the `$ref`/`anyOf`/`enum`/`type` dispatch
        // below: every keyword this compiler recognizes ANYWHERE (`properties`/`required`/
        // `additionalProperties`/`items` are only meaningful for specific `type`s, checked again
        // below, but must still be recognized HERE so a node with none of `type`/`enum`/`anyOf`/
        // `$ref` present — e.g. a bare `{"pattern":"^a"}` or `{"oneOf":[...]}` — is refused with the
        // OFFENDING KEYWORD's name rather than falling through to a generic "untyped schema" 400.
        // This single check is what refuses every keyword outside the subset (`pattern`, `format`,
        // `minLength`/`maxLength`, `minimum`/`maximum`/`exclusiveMinimum`/`exclusiveMaximum`,
        // `multipleOf`, `minItems`/`maxItems`/`uniqueItems`, `minProperties`/`maxProperties`,
        // `oneOf`/`allOf`/`not`/`const`, `if`/`then`/`else`, `patternProperties`/`propertyNames`/
        // `dependentRequired`, `prefixItems`/`contains`, `unevaluatedProperties`, and any other
        // unlisted keyword) without enumerating them individually.
        let masterAllowed =
            baseAllowed.union(["$ref", "anyOf", "enum", "type", "properties", "required", "additionalProperties", "items"])
        let unrecognized = keysPresent.subtracting(masterAllowed)
        guard unrecognized.isEmpty else {
            throw unsupportedFields(unrecognized, path: path)
        }

        // `$ref` — resolved and inlined, refusing sibling keywords and recursion.
        if let refRaw = seen["$ref"] {
            let allowed = baseAllowed.union(["$ref"])
            let extra = keysPresent.subtracting(allowed)
            guard extra.isEmpty else {
                throw unsupportedFields(extra, path: path)
            }
            guard case .string(let refString) = refRaw else {
                throw JSONSchemaCompileError(path: "\(path)/$ref", message: "$ref must be a string")
            }
            let target = try resolveRef(refString, path: "\(path)/$ref", root: root)
            guard refStack.insert(refString).inserted else {
                throw JSONSchemaCompileError(
                    path: "\(path)/$ref", message: "$ref is recursive: \(refString)")
            }
            defer { refStack.remove(refString) }
            return try compileNode(
                target, path: path, depth: depth + 1, root: root, isRoot: false, strict: strict,
                refStack: &refStack, budget: &budget)
        }

        // `anyOf` — branches must be pairwise disjoint by first byte.
        if let anyOfRaw = seen["anyOf"] {
            let allowed = baseAllowed.union(["anyOf"])
            let extra = keysPresent.subtracting(allowed)
            guard extra.isEmpty else {
                throw unsupportedFields(extra, path: path)
            }
            guard case .array(let branchesRaw) = anyOfRaw else {
                throw JSONSchemaCompileError(path: "\(path)/anyOf", message: "anyOf must be an array")
            }
            guard (2...JSONSchemaLimits.maxAnyOfBranches).contains(branchesRaw.count) else {
                throw JSONSchemaCompileError(
                    path: "\(path)/anyOf",
                    message: "anyOf must have between 2 and \(JSONSchemaLimits.maxAnyOfBranches) branches")
            }
            var branches: [JSONSchemaNode] = []
            for (index, branchRaw) in branchesRaw.enumerated() {
                branches.append(
                    try compileNode(
                        branchRaw, path: "\(path)/anyOf/\(index)", depth: depth + 1, root: root,
                        isRoot: false, strict: strict, refStack: &refStack, budget: &budget))
            }
            try validateDisjointFirstBytes(branches, path: "\(path)/anyOf")
            return .anyOf(branches)
        }

        // `enum` — non-empty scalar literals, optionally cross-checked against a sibling `type`.
        if let enumRaw = seen["enum"] {
            let allowed = baseAllowed.union(["enum", "type"])
            let extra = keysPresent.subtracting(allowed)
            guard extra.isEmpty else {
                throw unsupportedFields(extra, path: path)
            }
            guard case .array(let itemsRaw) = enumRaw, !itemsRaw.isEmpty else {
                throw JSONSchemaCompileError(path: "\(path)/enum", message: "enum must be a non-empty array")
            }
            guard itemsRaw.count <= JSONSchemaLimits.maxEnumValues else {
                throw JSONSchemaCompileError(
                    path: "\(path)/enum", message: "schema exceeds JSONSchemaLimits.maxEnumValues")
            }
            var expectedKinds: Set<String>?
            if let typeRaw = seen["type"] {
                let (primary, nullable) = try parseDeclaredType(typeRaw, path: "\(path)/type")
                expectedKinds = literalKinds(forDeclaredType: primary, nullable: nullable)
            }
            var literals: [JSONSchemaLiteral] = []
            var canonicalKeysSeen: Set<String> = []
            for (index, itemRaw) in itemsRaw.enumerated() {
                let itemPath = "\(path)/enum/\(index)"
                let literal = try canonicalizeEnumLiteral(itemRaw, path: itemPath)
                if let expectedKinds, !expectedKinds.contains(literal.kind) {
                    throw JSONSchemaCompileError(
                        path: itemPath, message: "enum value does not match the declared type")
                }
                guard canonicalKeysSeen.insert(literal.canonicalKey).inserted else {
                    throw JSONSchemaCompileError(path: itemPath, message: "enum contains a duplicate value")
                }
                budget.literalBytes += literal.text.utf8.count
                guard budget.literalBytes <= JSONSchemaLimits.maxLiteralBytes else {
                    throw JSONSchemaCompileError(
                        path: "\(path)/enum", message: "schema exceeds JSONSchemaLimits.maxLiteralBytes")
                }
                literals.append(JSONSchemaLiteral(jsonText: Array(literal.text.utf8)))
            }
            return .enumeration(literals)
        }

        // Otherwise the node must declare `type` (a plain type name, or `[T,"null"]`).
        guard let typeRaw = seen["type"] else {
            throw JSONSchemaCompileError(
                path: path, message: "untyped schema: must declare type, enum, anyOf, or $ref")
        }
        let (primary, nullable) = try parseDeclaredType(typeRaw, path: "\(path)/type")
        let allowed = baseAllowed.union(["type"]).union(extraKeys(forType: primary))
        let extra = keysPresent.subtracting(allowed)
        guard extra.isEmpty else {
            throw unsupportedFields(extra, path: path)
        }

        let node = try compileTypedNode(
            primary: primary, seen: seen, path: path, depth: depth, root: root, isRoot: isRoot,
            strict: strict, refStack: &refStack, budget: &budget)
        return nullable ? .anyOf([node, .null]) : node
    }

    private static func unsupportedFields(_ fields: Set<String>, path: String) -> JSONSchemaCompileError {
        JSONSchemaCompileError(
            path: path, message: "Unsupported field in schema: \(fields.sorted().joined(separator: ", "))")
    }

    private static func extraKeys(forType type: String) -> Set<String> {
        switch type {
        case "object": return ["properties", "required", "additionalProperties"]
        case "array": return ["items"]
        default: return []
        }
    }

    private static func compileTypedNode(
        primary: String,
        seen: [String: RawValue],
        path: String,
        depth: Int,
        root: RootContext,
        isRoot: Bool,
        strict: Bool,
        refStack: inout Set<String>,
        budget: inout Budget
    ) throws -> JSONSchemaNode {
        switch primary {
        case "object":
            return try compileObjectNode(
                seen: seen, path: path, depth: depth, root: root, strict: strict, refStack: &refStack,
                budget: &budget)
        case "array":
            guard let itemsRaw = seen["items"] else {
                throw JSONSchemaCompileError(
                    path: "\(path)/items", message: "items is required for array schemas")
            }
            if case .array = itemsRaw {
                throw JSONSchemaCompileError(
                    path: "\(path)/items", message: "tuple-form items is not supported")
            }
            guard case .object = itemsRaw else {
                throw JSONSchemaCompileError(path: "\(path)/items", message: "items must be a schema object")
            }
            let itemNode = try compileNode(
                itemsRaw, path: "\(path)/items", depth: depth + 1, root: root, isRoot: false,
                strict: strict, refStack: &refStack, budget: &budget)
            return .array(items: itemNode)
        case "string": return .string
        case "number": return .number
        case "integer": return .integer
        case "boolean": return .boolean
        case "null": return .null
        default:
            throw JSONSchemaCompileError(path: "\(path)/type", message: "Unsupported type: \(primary)")
        }
    }

    private static func compileObjectNode(
        seen: [String: RawValue],
        path: String,
        depth: Int,
        root: RootContext,
        strict: Bool,
        refStack: inout Set<String>,
        budget: inout Budget
    ) throws -> JSONSchemaNode {
        if let additionalProperties = seen["additionalProperties"] {
            guard case .bool(false) = additionalProperties else {
                throw JSONSchemaCompileError(
                    path: "\(path)/additionalProperties", message: "additionalProperties must be false")
            }
        } else if strict {
            throw JSONSchemaCompileError(
                path: path, message: "strict requires explicit additionalProperties:false")
        }

        var requiredNames: [String] = []
        if let requiredRaw = seen["required"] {
            guard case .array(let items) = requiredRaw else {
                throw JSONSchemaCompileError(path: "\(path)/required", message: "required must be an array of strings")
            }
            for (index, item) in items.enumerated() {
                guard case .string(let name) = item else {
                    throw JSONSchemaCompileError(
                        path: "\(path)/required/\(index)", message: "required entries must be strings")
                }
                requiredNames.append(name)
            }
            var dedupe: Set<String> = []
            for name in requiredNames {
                guard dedupe.insert(name).inserted else {
                    throw JSONSchemaCompileError(
                        path: "\(path)/required", message: "required contains a duplicate: \(name)")
                }
            }
        }

        var propertyPairs: [(String, RawValue)] = []
        if let propertiesRaw = seen["properties"] {
            guard case .object(let pairs) = propertiesRaw else {
                throw JSONSchemaCompileError(path: "\(path)/properties", message: "properties must be an object")
            }
            propertyPairs = pairs
        }
        guard propertyPairs.count <= JSONSchemaLimits.maxPropertiesPerObject else {
            throw JSONSchemaCompileError(
                path: "\(path)/properties", message: "schema exceeds JSONSchemaLimits.maxPropertiesPerObject")
        }

        let propertyNames = Set(propertyPairs.map(\.0))
        for name in requiredNames {
            guard propertyNames.contains(name) else {
                throw JSONSchemaCompileError(
                    path: "\(path)/required", message: "required names an undeclared property: \(name)")
            }
        }
        let requiredSet = Set(requiredNames)
        if strict {
            guard requiredSet == propertyNames else {
                throw JSONSchemaCompileError(
                    path: "\(path)/required",
                    message: "strict requires every property to be listed in required")
            }
        }

        var jsonProperties: [JSONSchemaProperty] = []
        for (name, rawValue) in propertyPairs {
            let propertyPath = "\(path)/properties/\(name)"
            try validatePropertyName(name, path: propertyPath)
            budget.literalBytes += name.utf8.count
            guard budget.literalBytes <= JSONSchemaLimits.maxLiteralBytes else {
                throw JSONSchemaCompileError(
                    path: propertyPath, message: "schema exceeds JSONSchemaLimits.maxLiteralBytes")
            }
            let propertyNode = try compileNode(
                rawValue, path: propertyPath, depth: depth + 1, root: root, isRoot: false,
                strict: strict, refStack: &refStack, budget: &budget)
            jsonProperties.append(
                JSONSchemaProperty(name: name, required: requiredSet.contains(name), value: propertyNode))
        }
        return .object(jsonProperties)
    }

    private static func validatePropertyName(_ name: String, path: String) throws {
        guard !name.isEmpty else {
            throw JSONSchemaCompileError(path: path, message: "property name must not be empty")
        }
        for scalar in name.unicodeScalars where scalar == "\"" || scalar == "\\" || scalar.value < 0x20 {
            throw JSONSchemaCompileError(
                path: path, message: "property name contains an unsupported character")
        }
    }

    private static func parseDeclaredType(
        _ raw: RawValue, path: String
    ) throws -> (primary: String, nullable: Bool) {
        switch raw {
        case .string(let name):
            guard validTypeNames.contains(name) else {
                throw JSONSchemaCompileError(path: path, message: "Unsupported type: \(name)")
            }
            return (name, false)
        case .array(let elements):
            guard elements.count == 2 else {
                throw JSONSchemaCompileError(
                    path: path, message: "type array must have exactly two elements")
            }
            var names: [String] = []
            for element in elements {
                guard case .string(let name) = element, validTypeNames.contains(name) else {
                    throw JSONSchemaCompileError(
                        path: path, message: "type array elements must be valid type names")
                }
                names.append(name)
            }
            guard names.filter({ $0 == "null" }).count == 1,
                let primary = names.first(where: { $0 != "null" })
            else {
                throw JSONSchemaCompileError(
                    path: path,
                    message: "type array must contain exactly one \"null\" and one other type")
            }
            return (primary, true)
        default:
            throw JSONSchemaCompileError(
                path: path, message: "type must be a string or a two-element array with null")
        }
    }

    private static func literalKinds(forDeclaredType primary: String, nullable: Bool) -> Set<String> {
        var kinds: Set<String>
        switch primary {
        case "number": kinds = ["number", "integer"]
        default: kinds = [primary]
        }
        if nullable { kinds.insert("null") }
        return kinds
    }

    private static func resolveRef(_ ref: String, path: String, root: RootContext) throws -> RawValue {
        for (prefix, table) in [("#/$defs/", root.defs), ("#/definitions/", root.definitions)] {
            guard ref.hasPrefix(prefix) else { continue }
            let name = String(ref.dropFirst(prefix.count))
            guard !name.isEmpty, !name.contains("/") else {
                throw JSONSchemaCompileError(path: path, message: "$ref target name is invalid: \(ref)")
            }
            guard let target = table[name] else {
                throw JSONSchemaCompileError(path: path, message: "$ref target not found: \(ref)")
            }
            return target
        }
        throw JSONSchemaCompileError(
            path: path, message: "$ref must point to #/$defs/<name> or #/definitions/<name>: \(ref)")
    }

    // MARK: - `enum` literal canonicalization

    private struct CanonicalLiteral {
        let text: String
        let kind: String
        let canonicalKey: String
    }

    private static func canonicalizeEnumLiteral(_ value: RawValue, path: String) throws -> CanonicalLiteral {
        switch value {
        case .string(let string):
            for scalar in string.unicodeScalars where scalar == "\"" || scalar == "\\" || scalar.value < 0x20 {
                throw JSONSchemaCompileError(
                    path: path, message: "enum string literal contains an unsupported character")
            }
            return CanonicalLiteral(text: "\"\(string)\"", kind: "string", canonicalKey: "s:\(string)")
        case .number(let text):
            let kind = (text.contains(".") || text.lowercased().contains("e")) ? "number" : "integer"
            return CanonicalLiteral(text: text, kind: kind, canonicalKey: "n:\(text)")
        case .bool(let bool):
            return CanonicalLiteral(
                text: bool ? "true" : "false", kind: "boolean", canonicalKey: "b:\(bool)")
        case .null:
            return CanonicalLiteral(text: "null", kind: "null", canonicalKey: "null")
        case .object, .array:
            throw JSONSchemaCompileError(
                path: path, message: "enum literal must be a scalar (string, number, boolean, or null)")
        }
    }

    // MARK: - `anyOf` disjointness

    private static func firstByteSet(_ node: JSONSchemaNode) -> Set<UInt8> {
        switch node {
        case .string: return [UInt8(ascii: "\"")]
        case .number, .integer:
            return Set((UInt8(ascii: "0")...UInt8(ascii: "9"))).union([UInt8(ascii: "-")])
        case .boolean: return [UInt8(ascii: "t"), UInt8(ascii: "f")]
        case .null: return [UInt8(ascii: "n")]
        case .object: return [UInt8(ascii: "{")]
        case .array: return [UInt8(ascii: "[")]
        case .enumeration(let literals):
            return Set(literals.compactMap(\.jsonText.first))
        case .anyOf(let branches):
            return branches.reduce(into: Set<UInt8>()) { $0.formUnion(firstByteSet($1)) }
        }
    }

    private static func validateDisjointFirstBytes(_ branches: [JSONSchemaNode], path: String) throws {
        var seen: Set<UInt8> = []
        for branch in branches {
            let bytes = firstByteSet(branch)
            guard seen.isDisjoint(with: bytes) else {
                throw JSONSchemaCompileError(
                    path: path,
                    message: "anyOf branches are not distinguishable by their first byte")
            }
            seen.formUnion(bytes)
        }
    }

    // MARK: - Fingerprint

    /// Lowercase hex SHA-256 of `node`'s `canonicalEncoding()` (`JSONSchemaFingerprint.swift`) — the
    /// COMPILED tree, not the caller's raw schema text — so two schemas share a fingerprint iff they
    /// compile to the identical `JSONSchemaNode` (order-sensitive on `.object`, matching what the
    /// constraint automaton actually walks). See `JSONSchemaResponseFormat.fingerprint`'s doc comment.
    private static func fingerprint(of node: JSONSchemaNode) -> String {
        let data = Data(node.canonicalEncoding())
        let digest = CryptoKit.SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}

extension JSONSchemaSubsetCompiler.RawValue: Equatable {
    /// Manual conformance: the compiler auto-synthesizes `Equatable` for an `indirect enum` only
    /// when every associated value is itself directly `Equatable`, and a labeled tuple element
    /// (`(key: String, value: RawValue)`) inside an array does not qualify — so `.object`'s payload
    /// blocks synthesis for the whole enum. Structural equality, recursing through `.object`'s pairs
    /// in order (so two objects with the same keys in a DIFFERENT declared order are NOT equal,
    /// matching the IR's own order-sensitivity).
    public static func == (lhs: JSONSchemaSubsetCompiler.RawValue, rhs: JSONSchemaSubsetCompiler.RawValue) -> Bool {
        switch (lhs, rhs) {
        case (.object(let l), .object(let r)):
            guard l.count == r.count else { return false }
            for (lp, rp) in zip(l, r) where lp.key != rp.key || lp.value != rp.value { return false }
            return true
        case (.array(let l), .array(let r)):
            return l == r
        case (.string(let l), .string(let r)):
            return l == r
        case (.number(let l), .number(let r)):
            return l == r
        case (.bool(let l), .bool(let r)):
            return l == r
        case (.null, .null):
            return true
        default:
            return false
        }
    }
}

// MARK: - Minimal order-preserving JSON parser

/// A hand-rolled RFC 8259 JSON parser producing `JSONSchemaSubsetCompiler.RawValue`. Every caller
/// in this codebase only invokes it on bytes `JSONSerialization` has already accepted, so this is
/// deliberately not a general-purpose hardened parser — it still fails closed (never silently
/// accepts malformed input) rather than assuming validity.
private struct OrderedJSONParser {
    /// Container nesting bound. The parser recurses once per `{`/`[`, and it walks the WHOLE request
    /// body (not only the schema), so a hostile body must not be able to exhaust the event-loop
    /// thread's stack. Kept below Foundation's own JSON nesting limit so any body that
    /// `JSONSerialization` accepted but is deeper than this is refused with a 400, never a crash.
    static let maxNestingDepth = 256

    private let bytes: [UInt8]
    private var index: Int = 0
    private var depth: Int = 0

    init(_ data: Data) { bytes = Array(data) }

    mutating func parseRoot() throws -> JSONSchemaSubsetCompiler.RawValue {
        skipWhitespace()
        let value = try parseValue()
        skipWhitespace()
        guard index == bytes.count else {
            throw JSONSchemaCompileError(path: "", message: "trailing data after JSON value")
        }
        return value
    }

    private mutating func parseValue() throws -> JSONSchemaSubsetCompiler.RawValue {
        skipWhitespace()
        guard let byte = peek() else { throw unexpectedEnd() }
        switch byte {
        case UInt8(ascii: "{"): return try parseObject()
        case UInt8(ascii: "["): return try parseArray()
        case UInt8(ascii: "\""): return .string(try parseStringLiteral())
        case UInt8(ascii: "t"): try expectLiteral("true"); return .bool(true)
        case UInt8(ascii: "f"): try expectLiteral("false"); return .bool(false)
        case UInt8(ascii: "n"): try expectLiteral("null"); return .null
        default: return .number(try parseNumberLiteral())
        }
    }

    private mutating func parseObject() throws -> JSONSchemaSubsetCompiler.RawValue {
        guard depth < Self.maxNestingDepth else {
            throw JSONSchemaCompileError(
                path: "", message: "JSON nesting exceeds \(Self.maxNestingDepth) levels")
        }
        depth += 1
        defer { depth -= 1 }
        index += 1
        var pairs: [(key: String, value: JSONSchemaSubsetCompiler.RawValue)] = []
        var seenKeys: Set<String> = []
        skipWhitespace()
        if peek() == UInt8(ascii: "}") {
            index += 1
            return .object(pairs)
        }
        while true {
            skipWhitespace()
            guard peek() == UInt8(ascii: "\"") else { throw malformed("expected object key") }
            let key = try parseStringLiteral()
            guard seenKeys.insert(key).inserted else {
                throw JSONSchemaCompileError(path: "", message: "duplicate key \"\(key)\" in schema object")
            }
            skipWhitespace()
            guard peek() == UInt8(ascii: ":") else { throw malformed("expected ':'") }
            index += 1
            let value = try parseValue()
            pairs.append((key, value))
            skipWhitespace()
            guard let separator = peek() else { throw unexpectedEnd() }
            if separator == UInt8(ascii: ",") {
                index += 1
                continue
            }
            if separator == UInt8(ascii: "}") {
                index += 1
                break
            }
            throw malformed("expected ',' or '}'")
        }
        return .object(pairs)
    }

    private mutating func parseArray() throws -> JSONSchemaSubsetCompiler.RawValue {
        guard depth < Self.maxNestingDepth else {
            throw JSONSchemaCompileError(
                path: "", message: "JSON nesting exceeds \(Self.maxNestingDepth) levels")
        }
        depth += 1
        defer { depth -= 1 }
        index += 1
        var items: [JSONSchemaSubsetCompiler.RawValue] = []
        skipWhitespace()
        if peek() == UInt8(ascii: "]") {
            index += 1
            return .array(items)
        }
        while true {
            items.append(try parseValue())
            skipWhitespace()
            guard let separator = peek() else { throw unexpectedEnd() }
            if separator == UInt8(ascii: ",") {
                index += 1
                continue
            }
            if separator == UInt8(ascii: "]") {
                index += 1
                break
            }
            throw malformed("expected ',' or ']'")
        }
        return .array(items)
    }

    private mutating func parseStringLiteral() throws -> String {
        index += 1  // consume opening quote
        var scalars = String.UnicodeScalarView()
        while true {
            guard index < bytes.count else { throw unexpectedEnd() }
            let byte = bytes[index]
            if byte == UInt8(ascii: "\"") {
                index += 1
                break
            }
            if byte == UInt8(ascii: "\\") {
                index += 1
                guard index < bytes.count else { throw unexpectedEnd() }
                let escape = bytes[index]
                switch escape {
                case UInt8(ascii: "\""): scalars.append("\""); index += 1
                case UInt8(ascii: "\\"): scalars.append("\\"); index += 1
                case UInt8(ascii: "/"): scalars.append("/"); index += 1
                case UInt8(ascii: "b"): scalars.append(Unicode.Scalar(8)); index += 1
                case UInt8(ascii: "f"): scalars.append(Unicode.Scalar(12)); index += 1
                case UInt8(ascii: "n"): scalars.append("\n"); index += 1
                case UInt8(ascii: "r"): scalars.append("\r"); index += 1
                case UInt8(ascii: "t"): scalars.append("\t"); index += 1
                case UInt8(ascii: "u"):
                    index += 1
                    let codeUnit1 = try parseHex4()
                    if (0xD800...0xDBFF).contains(codeUnit1) {
                        guard index + 1 < bytes.count, bytes[index] == UInt8(ascii: "\\"),
                            bytes[index + 1] == UInt8(ascii: "u")
                        else {
                            throw malformed("invalid surrogate pair")
                        }
                        index += 2
                        let codeUnit2 = try parseHex4()
                        guard (0xDC00...0xDFFF).contains(codeUnit2) else {
                            throw malformed("invalid low surrogate")
                        }
                        let combined = 0x10000 + (codeUnit1 - 0xD800) * 0x400 + (codeUnit2 - 0xDC00)
                        guard let scalar = Unicode.Scalar(combined) else {
                            throw malformed("invalid unicode scalar")
                        }
                        scalars.append(scalar)
                    } else {
                        guard let scalar = Unicode.Scalar(codeUnit1) else {
                            throw malformed("invalid unicode scalar")
                        }
                        scalars.append(scalar)
                    }
                default:
                    throw malformed("invalid escape sequence")
                }
            } else {
                var end = index
                while end < bytes.count, bytes[end] != UInt8(ascii: "\""), bytes[end] != UInt8(ascii: "\\") {
                    end += 1
                }
                guard let chunk = String(bytes: bytes[index..<end], encoding: .utf8) else {
                    throw malformed("invalid UTF-8 in string")
                }
                scalars.append(contentsOf: chunk.unicodeScalars)
                index = end
            }
        }
        return String(scalars)
    }

    private mutating func parseNumberLiteral() throws -> String {
        let start = index
        if peek() == UInt8(ascii: "-") { index += 1 }
        guard let firstDigit = peek(), isDigit(firstDigit) else { throw malformed("invalid number") }
        if firstDigit == UInt8(ascii: "0") {
            index += 1
        } else {
            while let byte = peek(), isDigit(byte) { index += 1 }
        }
        if peek() == UInt8(ascii: ".") {
            index += 1
            guard let byte = peek(), isDigit(byte) else { throw malformed("invalid number") }
            while let byte = peek(), isDigit(byte) { index += 1 }
        }
        if let byte = peek(), byte == UInt8(ascii: "e") || byte == UInt8(ascii: "E") {
            index += 1
            if let signByte = peek(), signByte == UInt8(ascii: "+") || signByte == UInt8(ascii: "-") {
                index += 1
            }
            guard let byte = peek(), isDigit(byte) else { throw malformed("invalid number") }
            while let byte = peek(), isDigit(byte) { index += 1 }
        }
        guard let text = String(bytes: bytes[start..<index], encoding: .utf8) else {
            throw malformed("invalid number")
        }
        return text
    }

    private mutating func parseHex4() throws -> Int {
        guard index + 4 <= bytes.count else { throw unexpectedEnd() }
        var value = 0
        for _ in 0..<4 {
            guard let digit = hexDigitValue(bytes[index]) else { throw malformed("invalid hex digit") }
            value = value * 16 + digit
            index += 1
        }
        return value
    }

    private mutating func expectLiteral(_ literal: String) throws {
        let literalBytes = Array(literal.utf8)
        guard index + literalBytes.count <= bytes.count,
            Array(bytes[index..<(index + literalBytes.count)]) == literalBytes
        else {
            throw malformed("expected literal \(literal)")
        }
        index += literalBytes.count
    }

    private mutating func skipWhitespace() {
        while index < bytes.count {
            switch bytes[index] {
            case 0x20, 0x09, 0x0A, 0x0D: index += 1
            default: return
            }
        }
    }

    private func peek() -> UInt8? {
        index < bytes.count ? bytes[index] : nil
    }

    private func isDigit(_ byte: UInt8) -> Bool {
        (UInt8(ascii: "0")...UInt8(ascii: "9")).contains(byte)
    }

    private func hexDigitValue(_ byte: UInt8) -> Int? {
        switch byte {
        case UInt8(ascii: "0")...UInt8(ascii: "9"): return Int(byte - UInt8(ascii: "0"))
        case UInt8(ascii: "a")...UInt8(ascii: "f"): return Int(byte - UInt8(ascii: "a")) + 10
        case UInt8(ascii: "A")...UInt8(ascii: "F"): return Int(byte - UInt8(ascii: "A")) + 10
        default: return nil
        }
    }

    private func malformed(_ message: String) -> JSONSchemaCompileError {
        JSONSchemaCompileError(path: "", message: "malformed JSON: \(message)")
    }

    private func unexpectedEnd() -> JSONSchemaCompileError {
        JSONSchemaCompileError(path: "", message: "malformed JSON: unexpected end of input")
    }
}
