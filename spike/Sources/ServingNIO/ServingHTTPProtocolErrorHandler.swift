import Foundation
import NIOCore
import NIOHTTP1
import ServingCore

/// Mints the per-request id every response's `x-request-id` header and `--request-log`
/// `request_id` report. One source for both the request handler (`receiveHead`) and the
/// protocol-error handler below, so a parse-error 400 carries the same id format as every other
/// response. Never derived from anything the client sent.
func servingMintRequestID() -> String {
    UUID().uuidString
}

/// Replaces NIO's `HTTPServerProtocolErrorHandler` (the pipeline is built with
/// `withErrorHandling: false`) so fast-mlx answers an HTTP parse error itself: a `400` carrying
/// exactly one `x-request-id`, one `--request-log` line with that id, and one `httpMetrics` count.
///
/// The in-flight guard is NIO's own: outbound writes passing through this handler (everything the
/// request handler writes, since this handler sits in front of it) track whether a response head
/// has been written without its `.end`. A parse error while a response is in flight writes nothing
/// -- a second status line would corrupt the unterminated response -- and only closes.
///
/// Every `HTTPParserError` is fatal for the connection. The error is always forwarded to the
/// request handler afterwards so it cancels any active generation, and the connection is closed.
final class ServingHTTPProtocolErrorHandler: ChannelDuplexHandler, RemovableChannelHandler {
    typealias InboundIn = HTTPServerRequestPart
    typealias InboundOut = HTTPServerRequestPart
    typealias OutboundIn = HTTPServerResponsePart
    typealias OutboundOut = HTTPServerResponsePart

    private let configuration: ServingHTTPConfiguration
    private var hasUnterminatedResponse = false

    init(configuration: ServingHTTPConfiguration) {
        self.configuration = configuration
    }

    func errorCaught(context: ChannelHandlerContext, error: Error) {
        guard error is HTTPParserError else {
            context.fireErrorCaught(error)
            return
        }

        if !hasUnterminatedResponse {
            let requestID = servingMintRequestID()
            let headers = HTTPHeaders([
                ("x-request-id", requestID),
                ("Connection", "close"),
                ("Content-Length", "0"),
            ])
            let head = HTTPResponseHead(version: .http1_1, status: .badRequest, headers: headers)
            Self.emit(configuration: configuration, requestID: requestID)
            context.write(wrapOutboundOut(.head(head)), promise: nil)
            context.writeAndFlush(wrapOutboundOut(.end(nil)), promise: nil)
        }

        // Forward so the request handler cancels any active generation, then close: a parse error
        // is fatal for the connection.
        context.fireErrorCaught(error)
        context.close(promise: nil)
    }

    func write(context: ChannelHandlerContext, data: NIOAny, promise: EventLoopPromise<Void>?) {
        switch unwrapOutboundIn(data) {
        case .head(let head) where Self.isInformational(head):
            break
        case .head:
            hasUnterminatedResponse = true
        case .body:
            break
        case .end:
            hasUnterminatedResponse = false
        }
        context.write(data, promise: promise)
    }

    /// NIO's own definition (`HTTPResponseHead.isInformational` is module-internal there): a 1xx
    /// interim response other than 101 does not start the final response.
    private static func isInformational(_ head: HTTPResponseHead) -> Bool {
        100 <= head.status.code && head.status.code < 200 && head.status.code != 101
    }

    /// Feeds `httpMetrics` (always) and the request log (when on) for the 400. The line carries no
    /// method and no route -- the request never parsed, and raw request bytes are never logged.
    private static func emit(configuration: ServingHTTPConfiguration, requestID: String) {
        configuration.httpMetrics.record(
            route: "<unmatched>",
            status: 400,
            outcome: ServingRequestLogRecord.Outcome.error.rawValue,
            durationSeconds: 0,
            ttftSeconds: nil)
        guard let sink = configuration.requestLog else {
            return
        }
        let record = ServingProtocolErrorLogRecord(
            ts: servingRequestLogTimestamp(),
            requestID: requestID,
            status: 400,
            outcome: .error,
            durationMs: 0,
            errorCode: "malformed_http_request")
        guard let data = try? JSONEncoder.openAI.encode(record),
            let line = String(data: data, encoding: .utf8)
        else {
            return
        }
        sink(line)
    }
}

/// The `--request-log` line for a request that never parsed: the same public keys and encoder as
/// `ServingRequestLogRecord`, minus `method`/`route` (absent, since there is no parsed request).
private struct ServingProtocolErrorLogRecord: Encodable {
    let ts: String
    let requestID: String
    let status: Int
    let outcome: ServingRequestLogRecord.Outcome
    let durationMs: Double
    let errorCode: String

    private enum CodingKeys: String, CodingKey {
        case ts
        case requestID = "request_id"
        case status
        case outcome
        case durationMs = "duration_ms"
        case errorCode = "error_code"
    }
}

@available(*, unavailable)
extension ServingHTTPProtocolErrorHandler: Sendable {}
