import Foundation

/// One embedding vector produced for one `OpenAIEmbeddingsRequest.inputs` entry, at the SAME index
/// (`ServingEmbeddingsResult.vectors[i]` answers `request.inputs[i]`). Pure value type -- MLX-free,
/// like the rest of `ServingCore` -- so this protocol can be implemented, mocked, and tested without
/// linking any model runtime.
public struct ServingEmbeddingsResult: Sendable, Equatable {
    /// One vector per input, in request order. Every vector is expected to share the same
    /// (non-zero) length -- the model's native embedding dimension -- though this type itself does
    /// not enforce that; the HTTP layer validates it before encoding a response (a backend that
    /// violates this contract is a defect, not a client error).
    public var vectors: [[Float]]
    /// Total input token count across every input in the batch, for `OpenAIEmbeddingsUsage
    /// .promptTokens`. Embedding requests never produce completion tokens.
    public var promptTokens: Int

    public init(vectors: [[Float]], promptTokens: Int) {
        self.vectors = vectors
        self.promptTokens = promptTokens
    }
}

/// The MLX-free serving-side contract for `POST /v1/embeddings`: turns a decoded
/// `OpenAIEmbeddingsRequest` (text or pre-tokenized inputs) into one vector per input. No HTTP
/// routing, JSON encoding, or model loading happens here -- implementations own tokenization (for
/// `.text` inputs) and the actual embedding computation. Mirrors `ServingGenerationBackend`'s role
/// for chat/completions: a thin seam the HTTP layer depends on so it never links a model runtime
/// directly.
public protocol ServingEmbeddingsBackend: Sendable {
    /// - Parameter request: Already validated by `OpenAIEmbeddingsRequest.decodeStrict` -- `inputs`
    ///   is guaranteed non-empty, and every token-array input's ids and per-input/aggregate token
    ///   counts already passed the documented limits.
    /// - Returns: One vector per `request.inputs` entry, in the same order.
    /// - Throws: Any error on a failure to embed the batch. The HTTP layer treats every thrown error
    ///   as a backend defect (mapped to a 500), never as a client-request problem.
    func embed(_ request: OpenAIEmbeddingsRequest) async throws -> ServingEmbeddingsResult
}
