import XCTest

@testable import ServingCore

/// Streaming counterpart to `ReasoningContentSplitterTests`. The streaming splitter must partition
/// raw text deltas into `reasoning` / `content` at the first `</think>` — matching the non-streaming
/// `ReasoningContentSplitter.split` byte-for-byte on any tag-present output regardless of how the
/// stream is chunked — while never tearing the tag across a chunk boundary and never stalling a
/// no-tag stream. The no-`</think>` case is a documented divergence (see the type doc comment).
final class StreamingReasoningSplitterTests: XCTestCase {
    // Fold a sequence of deltas through consume(...) then flush(), concatenating the emitted pieces.
    private func drive(_ chunks: [String]) -> (reasoning: String, content: String) {
        var splitter = StreamingReasoningSplitter()
        var r = "", c = ""
        for chunk in chunks {
            let piece = splitter.consume(chunk)
            r += piece.reasoning ?? ""
            c += piece.content ?? ""
        }
        let tail = splitter.flush()
        r += tail.reasoning ?? ""
        c += tail.content ?? ""
        return (r, c)
    }

    // Every way to cut `full` into two pieces, plus whole-string and one-character chunkings.
    private func chunkings(of full: String) -> [[String]] {
        var out: [[String]] = [[full], full.map(String.init)]
        let chars = Array(full)
        for cut in 0...chars.count {
            let a = String(chars[0..<cut])
            let b = String(chars[cut...])
            out.append([a, b])
        }
        return out
    }

    // MARK: - Basic behavior

    func testSingleChunkSplit() {
        let r = drive(["Let me check.\n</think>\n\nIt is sunny."])
        XCTAssertEqual(r.reasoning, "Let me check.")
        XCTAssertEqual(r.content, "It is sunny.")
    }

    func testTagTornAcrossChunkBoundary() {
        let r = drive(["reason</th", "ink>answer"])
        XCTAssertEqual(r.reasoning, "reason")
        XCTAssertEqual(r.content, "answer")
    }

    func testLeadingThinkStrippedAcrossChunks() {
        let r = drive(["<thi", "nk>reason</think>ans"])
        XCTAssertEqual(r.reasoning, "reason")
        XCTAssertEqual(r.content, "ans")
    }

    func testLeadingContentWhitespaceTrimmed() {
        let r = drive(["reason</think>", "\n\nAnswer"])
        XCTAssertEqual(r.reasoning, "reason")
        XCTAssertEqual(r.content, "Answer")
    }

    func testTrailingWhitespaceTrimmedBothSides() {
        let r = drive(["reason  ", "</think>answer  "])
        XCTAssertEqual(r.reasoning, "reason")
        XCTAssertEqual(r.content, "answer")
    }

    func testSecondThinkCloseInAnswerPassesThrough() {
        let r = drive(["r</think>a</think>b"])
        XCTAssertEqual(r.reasoning, "r")
        XCTAssertEqual(r.content, "a</think>b")
    }

    func testInteriorLoneAngleBracketPreserved() {
        let r = drive(["a < b</think>c < d"])
        XCTAssertEqual(r.reasoning, "a < b")
        XCTAssertEqual(r.content, "c < d")
    }

    func testEmptyAnswerYieldsNilContent() {
        var splitter = StreamingReasoningSplitter()
        _ = splitter.consume("still thinking</think>")
        let tail = splitter.flush()
        // Reasoning already emitted; final content must be nil (empty), matching non-streaming.
        XCTAssertNil(tail.content)
    }

    // MARK: - Documented no-tag divergence

    func testNoTagThinkingActiveYieldsAllReasoning() {
        let r = drive(["still thinking, never closed"])
        XCTAssertEqual(r.reasoning, "still thinking, never closed")
        XCTAssertEqual(r.content, "")
    }

    // MARK: - Latency guarantee

    func testPostTagContentEmittedImmediatelyWithoutTrailingWhitespace() {
        var splitter = StreamingReasoningSplitter()
        _ = splitter.consume("r</think>")
        // A plain content chunk (no whitespace, no tag prefix) must be emitted in full on this call —
        // never held back beyond the <7-byte tag-prefix window.
        let piece = splitter.consume("hello world")
        XCTAssertEqual(piece.content, "hello world")
    }

    func testTrailingTagPrefixHeldBackNotEmitted() {
        var splitter = StreamingReasoningSplitter()
        // In content phase there is no tag to scan for, so this proves the reasoning-phase holdback:
        // a partial "</think" must be held until resolved, never leaked as reasoning.
        let piece = splitter.consume("done</think")
        XCTAssertEqual(piece.reasoning ?? "", "done")
    }

    // MARK: - Parity properties (scoped to tag-present outputs)

    private let tagPresentSamples: [String] = [
        "reasoning here</think>answer here",
        "<think>reasoning here</think>answer here",
        "  leading ws reason  </think>  \n answer trailing  ",
        "multi\nline\nreason</think>multi\nline\nanswer",
        "r</think>a</think>b",
        "think about a < b</think>the answer is x > y",
        "</think>only answer, empty reasoning",
        "only reasoning, empty answer</think>",
        "edge</think>",
    ]

    func testPerFieldParityWithNonStreamingAcrossAllChunkings() {
        for sample in tagPresentSamples {
            let ns = ReasoningContentSplitter.split(sample)
            let expectedR = ns.reasoning ?? ""
            let expectedC = ns.content ?? ""
            for chunks in chunkings(of: sample) {
                let got = drive(chunks)
                XCTAssertEqual(
                    got.reasoning, expectedR,
                    "reasoning mismatch for sample=\(sample.debugDescription) chunks=\(chunks)")
                XCTAssertEqual(
                    got.content, expectedC,
                    "content mismatch for sample=\(sample.debugDescription) chunks=\(chunks)")
            }
        }
    }

    func testBytePreservationAcrossAllChunkings() {
        for sample in tagPresentSamples {
            let ns = ReasoningContentSplitter.split(sample)
            let expected = (ns.reasoning ?? "") + (ns.content ?? "")
            for chunks in chunkings(of: sample) {
                let got = drive(chunks)
                XCTAssertEqual(
                    got.reasoning + got.content, expected,
                    "byte preservation failed for sample=\(sample.debugDescription) chunks=\(chunks)")
            }
        }
    }
}
