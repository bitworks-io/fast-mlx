import XCTest

@testable import ServingCore

final class ReasoningContentSplitterTests: XCTestCase {
    func testNoThinkMarkerLeavesContentVerbatimAndReasoningNil() {
        let r = ReasoningContentSplitter.split("The weather in Paris is sunny.")
        XCTAssertNil(r.reasoning)
        XCTAssertEqual(r.content, "The weather in Paris is sunny.")
    }

    func testSplitsReasoningFromContentAtThinkClose() {
        let r = ReasoningContentSplitter.split("Let me check the weather.\n</think>\n\nIt is sunny in Paris.")
        XCTAssertEqual(r.reasoning, "Let me check the weather.")
        XCTAssertEqual(r.content, "It is sunny in Paris.")
    }

    func testStripsLeadingThinkOpenTagIfEmitted() {
        let r = ReasoningContentSplitter.split("<think>reasoning here</think>answer here")
        XCTAssertEqual(r.reasoning, "reasoning here")
        XCTAssertEqual(r.content, "answer here")
    }

    func testReasoningOnlyWhenAnswerIsEmpty() {
        let r = ReasoningContentSplitter.split("still thinking about it</think>")
        XCTAssertEqual(r.reasoning, "still thinking about it")
        XCTAssertNil(r.content)
    }

    func testEmptyInputYieldsNilNil() {
        let r = ReasoningContentSplitter.split("")
        XCTAssertNil(r.reasoning)
        XCTAssertNil(r.content)
    }

    // MARK: - separationActive retro-label (non-streaming truncated-thinking parity)

    // When separation is active (thinking-ON, thinks-by-default family) and the budget truncated the
    // stream before `</think>`, the whole output IS reasoning — mirror the streaming Option A contract
    // (all reasoning_content, empty content) instead of retro-labeling raw CoT as the answer.
    func testSeparationActiveNoCloseTagRetroLabelsAllAsReasoning() {
        let r = ReasoningContentSplitter.split("thinking, still no close tag", separationActive: true)
        XCTAssertEqual(r.reasoning, "thinking, still no close tag")
        XCTAssertNil(r.content)
    }

    // Normalization must match streaming's flush: strip a leading `<think>`, trim boundary whitespace.
    func testSeparationActiveNoCloseTagStripsLeadingThinkAndTrims() {
        let r = ReasoningContentSplitter.split("<think>\n  reasoning tail  \n", separationActive: true)
        XCTAssertEqual(r.reasoning, "reasoning tail")
        XCTAssertNil(r.content)
    }

    // Fable trap #1: gate on `</think>` ABSENCE, not `reasoning == nil`. `<think></think>answer` closes
    // thinking with an empty reasoning block and a real answer — the marker path must keep the answer.
    func testSeparationActiveWithEmptyThinkBlockKeepsRealAnswerAsContent() {
        let r = ReasoningContentSplitter.split("<think></think>42", separationActive: true)
        XCTAssertNil(r.reasoning)
        XCTAssertEqual(r.content, "42")
    }

    // Any text CONTAINING `</think>` is byte-identical regardless of the flag.
    func testSeparationActiveWithCloseTagIsByteIdenticalToDefault() {
        for text in [
            "<think>reasoning here</think>answer here",
            "Let me check.\n</think>\n\nIt is sunny.",
            "still thinking</think>",
            "</think>only answer",
        ] {
            let on = ReasoningContentSplitter.split(text, separationActive: true)
            let off = ReasoningContentSplitter.split(text)
            XCTAssertEqual(on.reasoning, off.reasoning, text)
            XCTAssertEqual(on.content, off.content, text)
        }
    }

    // Separation OFF (thinking-OFF / non-thinks-by-default family) is byte-identical to today: a
    // no-`</think>` answer stays verbatim content and is never mislabeled as reasoning.
    func testSeparationInactiveNoCloseTagUnchanged() {
        let r = ReasoningContentSplitter.split("just a plain answer", separationActive: false)
        XCTAssertNil(r.reasoning)
        XCTAssertEqual(r.content, "just a plain answer")
    }

    func testSeparationActiveEmptyOutputYieldsNilNil() {
        let r = ReasoningContentSplitter.split("", separationActive: true)
        XCTAssertNil(r.reasoning)
        XCTAssertNil(r.content)
    }

    // The crisp parity invariant: for any thinking-ON output with no `</think>`, the non-streaming
    // retro-label equals the streaming splitter's reasoning bytes (consume-all + flush) across EVERY
    // chunking, and content is empty on both.
    func testTruncatedThinkingMatchesStreamingSplitterAcrossAllChunkings() {
        let samples = [
            "thinking, still no close tag",
            "<think>held open reasoning that never closes",
            "  leading ws then <think> literal, no close",
            "line one\nline two  \n\n",
            "reason with a lone </thin partial that never completes",
        ]
        for sample in samples {
            let ns = ReasoningContentSplitter.split(sample, separationActive: true)
            XCTAssertNil(ns.content, sample)
            let chars = Array(sample)
            for cut in 0...chars.count {
                var splitter = StreamingReasoningSplitter()
                var r = ""
                for piece in [String(chars[0..<cut]), String(chars[cut...])] {
                    r += splitter.consume(piece).reasoning ?? ""
                }
                r += splitter.flush().reasoning ?? ""
                XCTAssertEqual(ns.reasoning, r.isEmpty ? nil : r, "sample=\(sample) cut=\(cut)")
            }
        }
    }
}
