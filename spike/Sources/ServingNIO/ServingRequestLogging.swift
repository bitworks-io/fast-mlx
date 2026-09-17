import Foundation

/// One line per finished request when `--request-log json` is enabled (`ServingHTTPConfiguration
/// .requestLog` non-`nil`). This is the PUBLIC access-log contract: stable snake_case keys, encoded
/// with `JSONEncoder.openAI`'s `.sortedKeys` formatting so every line has the same deterministic key
/// order.
///
/// Deliberately carries NOTHING an operator would need to redact: no message/prompt text, completion
/// text, tool-call arguments, the `Authorization` header or API key, a raw query string, a client
/// IP, or an unsanitized unknown request-field name. `ignoredFields` reuses
/// `OpenAIChatCompletionRequest.ignoredFields` verbatim -- that array already folds an arbitrary
/// unknown top-level key into a sanitized `"unknown:<key>"` entry (see that type's own doc comment),
/// so this record never has to sanitize anything itself.
struct ServingRequestLogRecord: Encodable, Sendable {
    /// How the request finished. Exactly three buckets by design (not a mirror of every internal
    /// `ServingCancellationReason` case): `completed` (a normal response, streamed or not, fully
    /// written), `error` (any 4xx/5xx, an admission refusal, a backpressure/response-limit timeout,
    /// or an unexpected backend failure), and `clientDisconnected` (the one cancellation reason that
    /// is neither a normal completion nor a server-decided error -- the peer went away).
    enum Outcome: String, Encodable, Sendable {
        case completed
        case error
        case clientDisconnected = "client_disconnected"
    }

    let ts: String
    let requestID: String
    let method: String
    let route: String
    let status: Int
    let outcome: Outcome
    let stream: Bool?
    let model: String?
    let promptTokens: Int?
    let completionTokens: Int?
    let ttftMs: Double?
    let durationMs: Double
    let finishReason: String?
    let errorCode: String?
    let ignoredFields: [String]?

    private enum CodingKeys: String, CodingKey {
        case ts
        case requestID = "request_id"
        case method
        case route
        case status
        case outcome
        case stream
        case model
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case ttftMs = "ttft_ms"
        case durationMs = "duration_ms"
        case finishReason = "finish_reason"
        case errorCode = "error_code"
        case ignoredFields = "ignored_fields"
    }
}

/// Facts captured once per request, before routing/decoding runs, so every finishing path -- an
/// early `validateHead` rejection, a body-stage rejection, a synchronous route response, or the
/// async chat/completions generation task -- can emit the request-log line from the SAME starting
/// point. `route` is already templated (see `servingRequestLogTemplatedRoute`) and `startedAt` uses
/// a fresh `ContinuousClock` reading taken as early as possible in `receiveHead`.
struct ServingRequestLogPreamble: Sendable {
    let method: String
    let route: String
    let startedAt: ContinuousClock.Instant
}

/// Maps a request URI to the PUBLIC route shape the request-log's `route` field reports, stripping
/// any query string (raw query strings are never logged -- see `ServingRequestLogRecord`'s doc
/// comment) and templating the variable `/v1/models/{id}` segment instead of echoing the requested
/// id verbatim. A URI this server does not recognize (an unknown route, which 404s) reports the
/// fixed `<unmatched>` template: an unmatched path is arbitrary, unbounded client input (it can
/// carry tokens or identifiers), so it is never echoed into the log.
func servingRequestLogTemplatedRoute(_ uri: String) -> String {
    let path = String(uri.split(separator: "?", maxSplits: 1, omittingEmptySubsequences: false)[0])
    let modelsPrefix = "/v1/models/"
    if path.hasPrefix(modelsPrefix), path.count > modelsPrefix.count {
        return "/v1/models/{id}"
    }
    return servingRequestLogKnownRoutes.contains(path) ? path : "<unmatched>"
}

private let servingRequestLogKnownRoutes: Set<String> = [
    "/v1/chat/completions", "/v1/completions", "/v1/models", "/v1/embeddings", "/metrics",
    "/healthz", "/readyz",
]

/// Millisecond elapsed time on the monotonic clock, matching the conversion
/// `ServingResponseEvidenceAccumulator.response()` already uses for `ServingEvidence`'s own
/// `durationMilliseconds`, so the two duration figures stay computed the same way.
func servingRequestLogDurationMilliseconds(
    from startedAt: ContinuousClock.Instant,
    to finishedAt: ContinuousClock.Instant
) -> Double {
    let duration = startedAt.duration(to: finishedAt)
    let components = duration.components
    return Double(components.seconds) * 1_000
        + Double(components.attoseconds) / 1_000_000_000_000_000
}

/// ISO-8601 UTC timestamp with millisecond precision (e.g. `2026-09-16T12:34:56.789Z`) for the
/// request-log's `ts` field. A fresh `ISO8601DateFormatter` is constructed per call rather than
/// shared: this only runs when `--request-log json` is opted in (never on the default `off` path),
/// so the extra allocation is not a production hot-path cost, and it avoids relying on
/// `ISO8601DateFormatter`'s undocumented cross-thread reuse safety under concurrent NIO event loops.
func servingRequestLogTimestamp(_ date: Date = Date()) -> String {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    return formatter.string(from: date)
}

/// Shared finishing point for every request this handler serves: ALWAYS records the `/metrics`
/// HTTP dependability series (`configuration.httpMetrics` -- see `ServingHTTPMetricsRecorder`),
/// then, only when `configuration.requestLog` is non-`nil` (`--request-log json`), also encodes
/// and emits one request-log line through it. `preamble` is unconditionally populated by
/// `OpenAIChatCompletionsHTTPHandler` regardless of the `--request-log` setting precisely so this
/// function can feed the metrics recorder on every request, not just when the JSON access log is
/// enabled -- callers may call this unconditionally at every finishing point without their own
/// `if configuration.requestLog != nil` guard. Encoding failure for the JSON line is swallowed
/// (`try?`): a request-log defect must never throw into or block the request path (see the serve
/// flag's own doc comment); metrics recording is a synchronous, non-throwing, in-memory update and
/// carries no equivalent failure mode.
func servingEmitRequestLog(
    configuration: ServingHTTPConfiguration,
    preamble: ServingRequestLogPreamble,
    requestID: String,
    status: Int,
    outcome: ServingRequestLogRecord.Outcome,
    stream: Bool? = nil,
    model: String? = nil,
    promptTokens: Int? = nil,
    completionTokens: Int? = nil,
    ttftMs: Double? = nil,
    finishReason: String? = nil,
    errorCode: String? = nil,
    ignoredFields: [String] = [],
    finishedAt: ContinuousClock.Instant = ContinuousClock().now
) {
    let durationMs = servingRequestLogDurationMilliseconds(from: preamble.startedAt, to: finishedAt)
    configuration.httpMetrics.record(
        route: preamble.route,
        status: status,
        outcome: outcome.rawValue,
        durationSeconds: durationMs / 1_000,
        ttftSeconds: ttftMs.map { $0 / 1_000 })

    guard let sink = configuration.requestLog else {
        return
    }
    let record = ServingRequestLogRecord(
        ts: servingRequestLogTimestamp(),
        requestID: requestID,
        method: preamble.method,
        route: preamble.route,
        status: status,
        outcome: outcome,
        stream: stream,
        model: model,
        promptTokens: promptTokens,
        completionTokens: completionTokens,
        ttftMs: ttftMs,
        durationMs: durationMs,
        finishReason: finishReason,
        errorCode: errorCode,
        ignoredFields: ignoredFields.isEmpty ? nil : ignoredFields)
    guard let data = try? JSONEncoder.openAI.encode(record),
        let line = String(data: data, encoding: .utf8)
    else {
        return
    }
    sink(line)
}

/// The production default sink for `ServingHTTPConfiguration.requestLog` when `--request-log json`
/// is passed with no test override: writes the line plus a trailing newline to standard error.
/// `FileHandle.write` issues the `write(2)` syscall directly (no `FileHandle`-level buffering), so
/// this is already flushed on return -- no separate fsync/flush step is needed.
public func servingRequestLogStandardErrorSink() -> @Sendable (String) -> Void {
    { line in
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }
}
