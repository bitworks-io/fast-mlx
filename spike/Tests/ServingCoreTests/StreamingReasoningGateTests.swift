import XCTest

@testable import ServingCore

/// `StreamingReasoningGate` decides — from the GENERATION stream itself — whether output is a thinking
/// response (first non-whitespace token is `<think>`) and only then routes through
/// `StreamingReasoningSplitter`. A non-thinking stream (no leading `<think>`) must pass through as raw
/// content, byte-for-byte identical to today, regardless of chunking. These tests pin both halves.
final class StreamingReasoningGateTests: XCTestCase {
    // Fold a sequence of deltas through consume(...) then flush(), concatenating the emitted pieces.
    private func drive(_ chunks: [String]) -> (reasoning: String, content: String) {
        var gate = StreamingReasoningGate()
        var r = "", c = ""
        for chunk in chunks {
            let piece = gate.consume(chunk)
            r += piece.reasoning ?? ""
            c += piece.content ?? ""
        }
        let tail = gate.flush()
        r += tail.reasoning ?? ""
        c += tail.content ?? ""
        return (r, c)
    }

    // Every way to cut `full` into two pieces, plus whole-string and one-character chunkings.
    private func chunkings(of full: String) -> [[String]] {
        var out: [[String]] = [[full], full.map(String.init)]
        let chars = Array(full)
        for cut in 0...chars.count {
            out.append([String(chars[0..<cut]), String(chars[cut...])])
        }
        return out
    }

    // MARK: - Thinking streams (leading <think> present)

    func testLeadingThinkStreamSplits() {
        let r = drive(["<think>Let me check.</think>It is sunny."])
        XCTAssertEqual(r.reasoning, "Let me check.")
        XCTAssertEqual(r.content, "It is sunny.")
    }

    func testLeadingThinkSplitsAcrossEveryChunking() {
        let full = "<think>reasoning here</think>the answer"
        for chunks in chunkings(of: full) {
            let r = drive(chunks)
            XCTAssertEqual(r.reasoning, "reasoning here", "chunking \(chunks)")
            XCTAssertEqual(r.content, "the answer", "chunking \(chunks)")
        }
    }

    func testLeadingThinkTagTornAcrossChunks() {
        // The opening tag itself split mid-tag must still be recognized as a thinking stream.
        let r = drive(["<thi", "nk>reason</think>ans"])
        XCTAssertEqual(r.reasoning, "reason")
        XCTAssertEqual(r.content, "ans")
    }

    func testWhitespacePrefixedThinkStillDetected() {
        // Leading whitespace before <think> is not meaningful; the stream is still a thinking stream.
        let r = drive(["\n\n<think>reason</think>answer"])
        XCTAssertEqual(r.reasoning, "reason")
        XCTAssertEqual(r.content, "answer")
    }

    func testThinkingStreamWithNoCloseIsAllReasoning() {
        // Documented divergence inherited from the splitter: a thinking stream truncated before
        // </think> is all reasoning (those tokens genuinely were reasoning).
        let r = drive(["<think>still thinking when the budget ran out"])
        XCTAssertEqual(r.reasoning, "still thinking when the budget ran out")
        XCTAssertEqual(r.content, "")
    }

    // MARK: - Non-thinking streams (no leading <think>) — must be byte-identical passthrough

    func testPlainAnswerIsRawContentByteIdentical() {
        let full = "The capital of France is Paris."
        for chunks in chunkings(of: full) {
            let r = drive(chunks)
            XCTAssertEqual(r.reasoning, "", "no reasoning for a plain answer; chunking \(chunks)")
            XCTAssertEqual(r.content, full, "content must equal the raw concatenation; chunking \(chunks)")
        }
    }

    func testStrayCloseTagWithoutOpenerStaysAllContent() {
        // The mis-gating case: a plain answer that happens to contain a stray "</think>" must NOT be
        // split (no opener ⇒ not a thinking stream). Byte-identical passthrough, including the stray tag.
        let full = "here is code: if a </think> b then c"
        for chunks in chunkings(of: full) {
            let r = drive(chunks)
            XCTAssertEqual(r.reasoning, "", "chunking \(chunks)")
            XCTAssertEqual(r.content, full, "stray </think> with no opener is literal content; chunking \(chunks)")
        }
    }

    func testAngleBracketContentThatIsNotThinkPassesThrough() {
        // A leading "<" that turns out not to be "<think>" must passthrough verbatim, not stall.
        let full = "<tool_call>{}</tool_call>"
        for chunks in chunkings(of: full) {
            let r = drive(chunks)
            XCTAssertEqual(r.reasoning, "", "chunking \(chunks)")
            XCTAssertEqual(r.content, full, "non-think leading '<' is verbatim content; chunking \(chunks)")
        }
    }

    func testLeadingWhitespaceOnPlainAnswerPreservedVerbatim() {
        // Passthrough must NOT trim: a non-thinking answer keeps its exact bytes (unlike the reasoning path).
        let full = "  \n  hello"
        for chunks in chunkings(of: full) {
            let r = drive(chunks)
            XCTAssertEqual(r.reasoning, "", "chunking \(chunks)")
            XCTAssertEqual(r.content, full, "passthrough preserves leading whitespace; chunking \(chunks)")
        }
    }

    func testEmptyAndWhitespaceOnlyStreams() {
        XCTAssertEqual(drive([]).content, "")
        XCTAssertEqual(drive([""]).content, "")
        XCTAssertEqual(drive(["   "]).content, "   ", "all-whitespace answer passes through verbatim")
        XCTAssertEqual(drive(["   "]).reasoning, "")
    }

    // MARK: - Known limitation: family-blindness (why the gate is not safe standalone)

    // A NON-thinking stream whose answer happens to begin with the literal text `<think>` and never emits
    // `</think>` is mislabeled ENTIRELY as reasoning — the answer vanishes from `content`. This is strictly
    // worse than both the raw-content path (which would emit it as content) and the non-streaming splitter
    // (which returns no-`</think>` text as content). The gate CANNOT tell a reasoning opener from content
    // that starts with the tag; only a family-level thinking-active signal can. This test pins the
    // limitation so a future reader knows the gate must be composed with that signal, never used bare.
    // (It is why the SSE wiring was reverted in 3a806f6.)
    func testLeadingThinkAsContentWithoutCloseIsMislabeled() {
        let r = drive(["<think>this is actually the answer, not reasoning"])
        XCTAssertEqual(r.reasoning, "this is actually the answer, not reasoning",
            "KNOWN LIMITATION: leading <think> as content with no </think> is misrouted to reasoning")
        XCTAssertEqual(r.content, "", "the answer is lost from content — the corruption the wiring revert avoids")
        // Contrast: the non-streaming splitter keeps the WHOLE string verbatim as content, INCLUDING the
        // literal leading `<think>` (it only strips the opener when a `</think>` is present ⇒ no-close means
        // all-content, tag and all). So the gate both loses the answer AND diverges from the non-streaming
        // contract on this input.
        let nonStreaming = ReasoningContentSplitter.split("<think>this is actually the answer, not reasoning")
        XCTAssertEqual(nonStreaming.content, "<think>this is actually the answer, not reasoning")
        XCTAssertNil(nonStreaming.reasoning)
    }

    // MARK: - Parity with the underlying splitters on tag-present, leading-<think> output

    func testGateMatchesStreamingSplitterOnLeadingThinkOutput() {
        // When the stream opens with <think>, the gate's split must equal feeding the same stream
        // straight to StreamingReasoningSplitter (the gate only adds the decision, not new partitioning).
        let full = "<think>weigh the options</think>\n\nfinal answer"
        for chunks in chunkings(of: full) {
            var splitter = StreamingReasoningSplitter()
            var sr = "", sc = ""
            for chunk in chunks {
                let p = splitter.consume(chunk)
                sr += p.reasoning ?? ""; sc += p.content ?? ""
            }
            let t = splitter.flush()
            sr += t.reasoning ?? ""; sc += t.content ?? ""

            let g = drive(chunks)
            XCTAssertEqual(g.reasoning, sr, "reasoning parity; chunking \(chunks)")
            XCTAssertEqual(g.content, sc, "content parity; chunking \(chunks)")
        }
    }

    func testGateMatchesNonStreamingSplitterOnLeadingThinkOutput() {
        // On a leading-<think> output, the gate's concatenated result must match the non-streaming
        // ReasoningContentSplitter.split (the streaming/non-streaming contract, now gated on a real opener).
        let full = "<think>consider carefully</think>the response"
        let expected = ReasoningContentSplitter.split(full)
        for chunks in chunkings(of: full) {
            let g = drive(chunks)
            XCTAssertEqual(g.reasoning, expected.reasoning ?? "", "chunking \(chunks)")
            XCTAssertEqual(g.content, expected.content ?? "", "chunking \(chunks)")
        }
    }
}
