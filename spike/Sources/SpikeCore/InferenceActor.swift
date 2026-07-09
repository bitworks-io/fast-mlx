import Foundation

/// Abstraction over "one decode step" so the actor's loop is testable without MLX.
public protocol Decoder {
    /// Prefill the prompt and return the first token id.
    mutating func prefill(_ promptTokens: [Int]) -> Int
    /// Given the last token, produce the next. (Greedy; temp=0.)
    mutating func step(last: Int) -> Int
    /// Discard any per-conversation state (e.g. KV cache) so the next `prefill` starts
    /// fresh, without reconstructing the decoder (and re-crossing the actor boundary with
    /// a fresh non-Sendable model reference — see MLXDecoder.reset()).
    mutating func reset()
}

/// Test double: replays a fixed script.
public struct ScriptedDecoder: Decoder {
    let script: [Int]
    let eos: Int
    var i = 0
    public init(script: [Int], eos: Int) { self.script = script; self.eos = eos }
    public mutating func prefill(_ p: [Int]) -> Int { defer { i += 1 }; return script[i] }
    public mutating func step(last: Int) -> Int { defer { i += 1 }; return script[i] }
    public mutating func reset() { i = 0 }
}

/// Single-owner actor: owns the decoder (and, transitively, all MLX state) and streams
/// generated token ids to callers without ever exposing MLX types across the actor boundary.
public actor InferenceActor {
    private var decoder: any Decoder
    public init(decoder: sending any Decoder) { self.decoder = decoder }

    /// Discard per-conversation state (KV cache) so a subsequent `submit` starts fresh.
    /// Lets one actor/decoder/model be reused across bench runs without crossing the
    /// actor boundary again with a non-Sendable model reference (see `MLXDecoder`).
    public func resetForNewRun() {
        decoder.reset()
    }

    /// Non-blocking: returns a stream immediately; decode runs inside the actor.
    public func submit(promptTokens: [Int], maxTokens: Int, eos: Int = 2) -> AsyncThrowingStream<Int, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { await self.run(promptTokens, maxTokens, eos, continuation) }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func run(
        _ prompt: [Int], _ maxTokens: Int, _ eos: Int,
        _ cont: AsyncThrowingStream<Int, Error>.Continuation
    ) {
        var tok = decoder.prefill(prompt)
        var n = 0
        while n < maxTokens {
            if tok == eos { break }
            cont.yield(tok)
            n += 1
            if Task.isCancelled { break }
            tok = decoder.step(last: tok)
        }
        cont.finish()
    }
}
