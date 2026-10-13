import Foundation
import XCTest

import Jinja

/// Conformance of the pinned swift-jinja to Python Jinja2 plus the HF transformers chat-template
/// environment (`ImmutableSandboxedEnvironment(trim_blocks, lstrip_blocks)` and the transformers
/// `tojson` = `json.dumps(ensure_ascii=False, sort_keys=False)`).
///
/// The expected strings were rendered by Python Jinja2 (see the fixture's `jinja2_version` and
/// `generator`; regenerate with `scripts/gen_jinja_conformance_fixture.py`). Templates are
/// synthetic. Swift renders each row with the options swift-transformers uses for chat templates
/// (`lstripBlocks: true, trimBlocks: true`).
///
/// Row contexts are JSON. `JSONSerialization` and swift-jinja's own `Decodable` both lose or sort
/// key order, so this file parses the fixture with a small order-preserving JSON parser and builds
/// `Jinja.Value.object(OrderedDictionary)` directly.
final class JinjaTransformersConformanceTests: XCTestCase {
    private struct Row {
        let name: String
        let template: String
        let context: [String: Value]
        let expected: String
    }

    /// swift-jinja >= 2.5 keys objects by `ObjectKey`; top-level render variables are string-keyed.
    private static func stringKeyed(_ object: OrderedDictionary<ObjectKey, Value>) -> [String: Value] {
        var out: [String: Value] = [:]
        for (key, value) in object {
            if case .string(let name) = key { out[name] = value }
        }
        return out
    }

    private func rows() throws -> [Row] {
        let root = try OrderedJSON.parse(JinjaTransformersConformanceTable.json)
        guard case .object(let doc) = root, case .array(let cases)? = doc[.string("cases")] else {
            throw OrderedJSON.ParseError.malformed("fixture has no cases array")
        }
        return try cases.map { raw in
            guard case .object(let row) = raw,
                case .string(let name)? = row[.string("name")],
                case .string(let template)? = row[.string("template")],
                case .object(let context)? = row[.string("context")],
                case .string(let expected)? = row[.string("expected")]
            else { throw OrderedJSON.ParseError.malformed("bad row") }
            return Row(name: name, template: template, context: Self.stringKeyed(context), expected: expected)
        }
    }

    func testEveryConformanceRowRendersLikeTransformers() throws {
        let rows = try rows()
        XCTAssertGreaterThanOrEqual(rows.count, 12, "fixture lost rows")
        var failed: [String] = []
        for row in rows {
            do {
                let template = try Template(row.template, with: .init(lstripBlocks: true, trimBlocks: true))
                let actual = try template.render(row.context)
                if actual != row.expected {
                    failed.append(row.name)
                    XCTFail("row \(row.name): rendered \(String(reflecting: actual)) expected \(String(reflecting: row.expected))")
                }
            } catch {
                failed.append(row.name)
                XCTFail("row \(row.name): threw \(error)")
            }
        }
        XCTAssertEqual(failed, [], "rows that diverge from the transformers reference")
    }

    /// KNOWN, deliberately out-of-scope divergence: Python Jinja2 (keep_trailing_newline=False)
    /// drops ONE trailing newline of the template source; swift-jinja 2.5.1 keeps it. The
    /// conformance rows avoid templates that end in a newline. If upstream fixes this, this test
    /// fails: flip it to the Python behavior (`"x"`) and add a fixture row.
    func testKnownDivergenceTrailingTemplateNewlineIsKept() throws {
        let template = try Template("x\n", with: .init(lstripBlocks: true, trimBlocks: true))
        XCTAssertEqual(try template.render([:]), "x\n", "Python Jinja2 renders \"x\"")
    }

    func testFixtureRecordsReferenceProvenance() throws {
        guard case .object(let doc) = try OrderedJSON.parse(JinjaTransformersConformanceTable.json) else {
            return XCTFail("fixture is not an object")
        }
        guard case .string(let version)? = doc[.string("jinja2_version")], case .string(let generator)? = doc[.string("generator")] else {
            return XCTFail("fixture lacks jinja2_version/generator")
        }
        XCTAssertFalse(version.isEmpty)
        XCTAssertEqual(generator, "scripts/gen_jinja_conformance_fixture.py")
    }

    /// Discrimination guard: the order-preserving parser really keeps non-alphabetical key order
    /// (otherwise the tojson rows could pass for the wrong reason or fail spuriously).
    func testOrderedParserPreservesInsertionOrderAndEscapes() throws {
        guard case .object(let o) = try OrderedJSON.parse(#"{"z":1,"a":{"y":[true,null,2.5],"b":"\u00e9\/\ud83d\ude00"},"m":-3}"#) else {
            return XCTFail("not an object")
        }
        XCTAssertEqual(Array(o.keys), [.string("z"), .string("a"), .string("m")])
        guard case .object(let inner)? = o[.string("a")] else { return XCTFail("no inner") }
        XCTAssertEqual(Array(inner.keys), [.string("y"), .string("b")])
        XCTAssertEqual(inner[.string("b")], .string("é/\u{1F600}"))
        XCTAssertEqual(o[.string("m")], .int(-3))
    }
}

/// Minimal order-preserving JSON parser producing `Jinja.Value`.
private enum OrderedJSON {
    enum ParseError: Error { case malformed(String) }

    static func parse(_ text: String) throws -> Value {
        var p = Parser(bytes: Array(text.utf8))
        let v = try p.value()
        p.skipWS()
        guard p.i == p.bytes.count else { throw ParseError.malformed("trailing data at \(p.i)") }
        return v
    }

    private struct Parser {
        let bytes: [UInt8]
        var i = 0
        init(bytes: [UInt8]) { self.bytes = bytes }

        mutating func skipWS() {
            while i < bytes.count, [0x20, 0x0A, 0x0D, 0x09].contains(bytes[i]) { i += 1 }
        }

        mutating func expect(_ s: String) throws {
            for b in s.utf8 {
                guard i < bytes.count, bytes[i] == b else { throw ParseError.malformed("expected \(s) at \(i)") }
                i += 1
            }
        }

        mutating func value() throws -> Value {
            skipWS()
            guard i < bytes.count else { throw ParseError.malformed("eof") }
            switch bytes[i] {
            case UInt8(ascii: "{"):
                i += 1
                var dict = OrderedDictionary<ObjectKey, Value>()
                skipWS()
                if bytes[i] == UInt8(ascii: "}") { i += 1; return .object(dict) }
                while true {
                    skipWS()
                    let key = try string()
                    skipWS()
                    try expect(":")
                    dict[.string(key)] = try value()
                    skipWS()
                    if bytes[i] == UInt8(ascii: ",") { i += 1; continue }
                    try expect("}")
                    return .object(dict)
                }
            case UInt8(ascii: "["):
                i += 1
                var items: [Value] = []
                skipWS()
                if bytes[i] == UInt8(ascii: "]") { i += 1; return .array(items) }
                while true {
                    items.append(try value())
                    skipWS()
                    if bytes[i] == UInt8(ascii: ",") { i += 1; continue }
                    try expect("]")
                    return .array(items)
                }
            case UInt8(ascii: "\""):
                return .string(try string())
            case UInt8(ascii: "t"): try expect("true"); return .boolean(true)
            case UInt8(ascii: "f"): try expect("false"); return .boolean(false)
            case UInt8(ascii: "n"): try expect("null"); return .null
            default:
                let start = i
                while i < bytes.count, "+-0123456789.eE".utf8.contains(bytes[i]) { i += 1 }
                let s = String(decoding: bytes[start..<i], as: UTF8.self)
                if let n = Int(s) { return .int(n) }
                if let d = Double(s) { return .double(d) }
                throw ParseError.malformed("bad number \(s)")
            }
        }

        mutating func hex4() throws -> UInt32 {
            guard i + 4 <= bytes.count, let v = UInt32(String(decoding: bytes[i..<i + 4], as: UTF8.self), radix: 16) else {
                throw ParseError.malformed("bad \\u escape at \(i)")
            }
            i += 4
            return v
        }

        mutating func string() throws -> String {
            try expect("\"")
            var out = [UInt8]()
            while i < bytes.count {
                let b = bytes[i]
                i += 1
                if b == UInt8(ascii: "\"") { return String(decoding: out, as: UTF8.self) }
                if b != UInt8(ascii: "\\") { out.append(b); continue }
                let e = bytes[i]
                i += 1
                switch e {
                case UInt8(ascii: "n"): out.append(0x0A)
                case UInt8(ascii: "t"): out.append(0x09)
                case UInt8(ascii: "r"): out.append(0x0D)
                case UInt8(ascii: "b"): out.append(0x08)
                case UInt8(ascii: "f"): out.append(0x0C)
                case UInt8(ascii: "\""), UInt8(ascii: "\\"), UInt8(ascii: "/"): out.append(e)
                case UInt8(ascii: "u"):
                    var scalar = try hex4()
                    if (0xD800..<0xDC00).contains(scalar) {
                        try expect("\\u")
                        let low = try hex4()
                        scalar = 0x10000 + ((scalar - 0xD800) << 10) + (low - 0xDC00)
                    }
                    guard let u = Unicode.Scalar(scalar) else { throw ParseError.malformed("bad scalar") }
                    out.append(contentsOf: Array(String(Character(u)).utf8))
                default: throw ParseError.malformed("bad escape")
                }
            }
            throw ParseError.malformed("unterminated string")
        }
    }
}
