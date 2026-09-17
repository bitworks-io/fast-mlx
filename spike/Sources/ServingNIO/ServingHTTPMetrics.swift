import os

/// Prometheus histogram bucket boundaries (seconds), shared by both the request-duration and
/// time-to-first-token histograms `ServingHTTPMetricsRecorder` accumulates. Fixed and
/// non-configurable: a serve flag to change these would let an operator silently blow up label
/// cardinality or drop the ability to compare two deployments' scrapes.
let servingHTTPMetricsHistogramBuckets: [Double] = [
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300,
]

/// Maps an HTTP status code to the Prometheus status-class label (`"2xx"`, `"4xx"`, `"5xx"`, ...).
/// Falls back to `"other"` for anything outside the standard 1xx-5xx range so a malformed or
/// future status code can never grow the label's cardinality without bound.
func servingHTTPMetricsStatusClass(_ status: Int) -> String {
    let hundreds = status / 100
    guard (1...5).contains(hundreds) else {
        return "other"
    }
    return "\(hundreds)xx"
}

/// Maps the access log's templated route (`servingRequestLogTemplatedRoute`) to the metrics route
/// label. Identical to the access-log route string for every KNOWN route (e.g.
/// `/v1/chat/completions`, `/v1/models/{id}`) -- an unmatched/unknown path collapses to the fixed
/// `"other"` label instead of the access log's `<unmatched>` sentinel, which is fine for a
/// human-read JSON line but is an unconventional Prometheus label value (angle brackets).
func servingHTTPMetricsRoute(_ templatedRoute: String) -> String {
    templatedRoute == "<unmatched>" ? "other" : templatedRoute
}

/// Formats a fixed histogram bucket boundary the way Prometheus text-format `le` label values are
/// conventionally rendered: whole numbers with no trailing `.0` (`"1"`, `"300"`), fractional
/// boundaries via `Double`'s own shortest round-tripping description (`"0.005"`, `"2.5"`).
func servingHTTPMetricsFormatBoundary(_ value: Double) -> String {
    if value == value.rounded(.towardZero) {
        return String(Int(value))
    }
    return String(value)
}

/// One route's accumulated latency observations for a single histogram family (either
/// `fastmlx_http_request_duration_seconds` or `fastmlx_http_time_to_first_token_seconds`).
/// `bucketCounts` is the PER-BUCKET (non-cumulative) tally aligned with
/// `servingHTTPMetricsHistogramBuckets` -- `observe` increments exactly one slot (the first
/// boundary the value is `<=`), and `cumulativeBucketCounts` does the running-sum Prometheus
/// text format actually requires at render time. A value greater than every fixed boundary (300s)
/// increments no regular bucket, but is still folded into `sum`/`count` (and therefore into the
/// `+Inf` bucket, which callers render as `count` directly) -- so `count` and the `+Inf` sample
/// always agree by construction, never by a separate reconciliation step.
struct ServingHTTPMetricsHistogram: Sendable {
    private(set) var bucketCounts: [Int]
    private(set) var sum: Double = 0
    private(set) var count: Int = 0

    init() {
        bucketCounts = Array(repeating: 0, count: servingHTTPMetricsHistogramBuckets.count)
    }

    mutating func observe(_ value: Double) {
        if let index = servingHTTPMetricsHistogramBuckets.firstIndex(where: { value <= $0 }) {
            bucketCounts[index] += 1
        }
        sum += value
        count += 1
    }

    var cumulativeBucketCounts: [Int] {
        var running = 0
        return bucketCounts.map {
            running += $0
            return running
        }
    }
}

/// Thread-safe, per-`ServingHTTPConfiguration` recorder backing the `/metrics` HTTP dependability
/// series (`fastmlx_http_requests_total`, `fastmlx_http_request_duration_seconds`,
/// `fastmlx_http_time_to_first_token_seconds`). Fed once per finished request from
/// `servingEmitRequestLog` -- unlike the JSON access-log line that function conditionally writes,
/// this recorder is ALWAYS fed, independently of whether `--request-log json` is enabled: these
/// counters/histograms are the server's baseline dependability signal, not an opt-in diagnostic.
/// `OSAllocatedUnfairLock` matches the lock primitive already used elsewhere in this target
/// (`ServingHTTPEvidenceTracker`, `ServingTransportRequestControl` in
/// `OpenAIChatCompletionsHTTPHandler.swift`) -- no new package dependency. Label cardinality is
/// bounded: route is the same finite templated set `servingRequestLogTemplatedRoute` produces
/// (collapsed further to `"other"` for an unmatched path), status is one of a handful of `NxN`
/// classes, and outcome is `ServingRequestLogRecord.Outcome`'s three raw values.
final class ServingHTTPMetricsRecorder: Sendable {
    struct CounterKey: Hashable, Sendable {
        let route: String
        let statusClass: String
        let outcome: String
    }

    private struct State: Sendable {
        var counters: [CounterKey: Int] = [:]
        var durationHistograms: [String: ServingHTTPMetricsHistogram] = [:]
        var ttftHistograms: [String: ServingHTTPMetricsHistogram] = [:]
    }

    /// A fully-sorted read of the recorder's current state, safe to render without holding the
    /// lock -- `render` (in `OpenAIChatCompletionsHTTPHandler.swift`) never needs to reach back
    /// into the recorder mid-render.
    struct Snapshot: Sendable {
        var counters: [(route: String, statusClass: String, outcome: String, count: Int)]
        var durationHistograms: [(route: String, histogram: ServingHTTPMetricsHistogram)]
        var ttftHistograms: [(route: String, histogram: ServingHTTPMetricsHistogram)]
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    /// Records one finished request. `route` is the RAW templated route as the access log would
    /// report it (e.g. `<unmatched>` for an unrecognized path) -- this function itself applies
    /// `servingHTTPMetricsRoute`'s `"other"` collapse, so every caller can pass the same value it
    /// already has on hand without duplicating that mapping. `ttftSeconds` is `nil` for anything
    /// other than a streaming success (a non-streaming response, or any error/disconnect outcome),
    /// matching `ServingRequestLogRecord.ttftMs`'s own "only ever set on that one path" contract.
    func record(
        route: String,
        status: Int,
        outcome: String,
        durationSeconds: Double,
        ttftSeconds: Double?
    ) {
        let metricsRoute = servingHTTPMetricsRoute(route)
        let statusClass = servingHTTPMetricsStatusClass(status)
        state.withLock { state in
            state.counters[
                CounterKey(route: metricsRoute, statusClass: statusClass, outcome: outcome),
                default: 0
            ] += 1
            state.durationHistograms[metricsRoute, default: ServingHTTPMetricsHistogram()]
                .observe(durationSeconds)
            if let ttftSeconds {
                state.ttftHistograms[metricsRoute, default: ServingHTTPMetricsHistogram()]
                    .observe(ttftSeconds)
            }
        }
    }

    func snapshot() -> Snapshot {
        state.withLock { state in
            Snapshot(
                counters: state.counters
                    .map { (route: $0.key.route, statusClass: $0.key.statusClass, outcome: $0.key.outcome, count: $0.value) }
                    .sorted(by: servingHTTPMetricsCounterOrder),
                durationHistograms: state.durationHistograms
                    .map { (route: $0.key, histogram: $0.value) }
                    .sorted { $0.route < $1.route },
                ttftHistograms: state.ttftHistograms
                    .map { (route: $0.key, histogram: $0.value) }
                    .sorted { $0.route < $1.route })
        }
    }
}

/// Deterministic series ordering for `fastmlx_http_requests_total`: route, then status class, then
/// outcome -- so two scrapes of an unchanged counter state always render byte-identical text.
private func servingHTTPMetricsCounterOrder(
    _ lhs: (route: String, statusClass: String, outcome: String, count: Int),
    _ rhs: (route: String, statusClass: String, outcome: String, count: Int)
) -> Bool {
    if lhs.route != rhs.route {
        return lhs.route < rhs.route
    }
    if lhs.statusClass != rhs.statusClass {
        return lhs.statusClass < rhs.statusClass
    }
    return lhs.outcome < rhs.outcome
}
