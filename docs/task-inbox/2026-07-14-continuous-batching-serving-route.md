---
status: captured
type: serving-integration
priority: required-before-production-default
created: 2026-07-14
source: continuous-batching-phase3-scope-split
planning_ready: false
implementation_ready: false
---

# Continuous batching production route and disconnect propagation

## Raw Capture

Wire the promoted exact dense-Qwen3 continuous runtime into the production serving/API
boundary, implement the measured isolated-PLD versus simultaneous-batch policy without
changing temperature-zero bytes, and prove that an actual client disconnect reaches runtime
cancellation and releases its slot within one configured keepalive interval.

## Light Triage

The 2026-07-14 continuous-batching verdict promotes an explicit engine/probe building block,
not a product default. Runtime cancellation, slot reuse, A/B/A recovery, and the resident soak
pass; HTTP/SSE/WebSocket ownership, backpressure, transport cancellation propagation, and the
dynamic policy handoff are not implemented.

Open questions:

- Which first production surface owns routing: HTTP, CLI `serve`, or Concierge integration?
- How is an isolated request held or migrated when a second request arrives, without violating
  drain-before-batch-join or silently disabling PLD?
- What keepalive and backpressure contract applies across HTTP/SSE/WebSocket transports?
- Which acceptance test observes client close → actor cancellation → slot removal/reuse with
  byte-identical survivor output and zero reservation leak?

## Next Planning Step

Write a serving-boundary design and TDD plan after the service/API owner is selected. Do not
enable a default route until the real disconnect, exact A/B/A, recovery, and policy-transition
acceptance tests pass on a clean SHA.
