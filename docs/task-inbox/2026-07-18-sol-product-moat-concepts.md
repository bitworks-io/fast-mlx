---
status: captured
type: speculative-product-research
priority: unset
created: 2026-07-18
source: Sol ecosystem synthesis
project: fast-mlx
planning_ready: false
implementation_ready: false
---

# Sol product moat concepts

## Raw Capture

Capture only the genuinely non-duplicate speculative product/research concepts from today's
Sol ecosystem synthesis as a planning seed, not a plan:

1. Thermal/memory/network-aware multi-Mac inference control plane and sharding across Apple
   Silicon.
2. Signed model-packaging/provenance pipeline from Hugging Face checkpoint to MLX-ready
   artifact with quant recipe, compatibility assertions, measured dial evidence, and safe
   one-click upgrade/rollback.
3. Policy router selecting MLX, llama.cpp, Core ML, or remote execution per request/model,
   privacy constraint, and latency goal only if this can preserve fast-mlx measurement
   semantics.

Persistent session SSD KV is already owned by exact-prefix/session-cache, and trusted benchmark
network/website/community publication is already owned by website-benchmark-community. Do not
duplicate them.

## Agent Notes

Captured only. Not planned or implemented yet.

These ideas are speculative product/research moat candidates around the existing fast-mlx
measurement thesis. They should not displace the current engine, harness, catalog, dial,
macOS app, exact-prefix cache, or website/community work without a later evidence-backed
planning session.

## Light Triage

Primary user value: operators and advanced local-AI users could get a more resilient Apple
Silicon serving system: scale across multiple Macs when one box is not enough, install and
upgrade measured model artifacts with trustworthy provenance, and route each request to the
best available execution backend without hiding the quality, latency, privacy, or compatibility
tradeoff.

Speculative concepts:

- **Multi-Mac inference control plane/sharding:** explore whether fast-mlx can coordinate
  Apple Silicon hosts using thermal state, memory headroom, queue depth, network bandwidth,
  and model residency to pick placement or split work. This is not the existing single-host
  capacity advisor; it is a possible distributed serving layer above it.
- **Signed model-packaging/provenance pipeline:** explore a reproducible path from an upstream
  HF checkpoint to an MLX-ready fast-mlx artifact, including source revision, license marker,
  conversion toolchain, quant recipe, model-architecture compatibility assertions, measured
  dial-frontier evidence, signatures, and one-click upgrade/rollback metadata.
- **Measured policy router:** explore a router that can choose MLX, llama.cpp/GGUF, Core ML,
  or remote execution based on request shape, model support, privacy policy, latency target,
  and hardware state only if the selected backend remains covered by fast-mlx's
  engine-agnostic conformance and precision-loss semantics.

Out of scope for this capture:

- Persistent session, hot-prefix, or SSD KV cache design; that belongs to
  `2026-07-12-exact-prefix-session-cache.md`.
- Public benchmark publication, trusted community surfaces, or website automation; that
  belongs to `2026-07-18-website-benchmark-community.md`.
- macOS dial UX details; that belongs to `2026-07-18-adaptive-macos-dial-ui.md`.
- Any implementation, architecture plan, vendor choice, or commitment to support non-MLX
  backends.

## Risks / Open Questions

- Can multi-host sharding beat the network and synchronization overhead for realistic
  Apple-Silicon inference paths, or is the valuable shape only placement, failover, and
  model-residency routing?
- What trust root signs model artifacts and dial-frontier evidence, and how are compromised,
  stale, license-invalid, or mis-measured artifacts revoked?
- Which compatibility assertions are mandatory before an artifact can be offered: tokenizer,
  chat template, architecture family, RoPE/context semantics, quant layout, KV format,
  license, and measured hardware profile?
- Can a multi-backend router preserve the same measurement contract when llama.cpp, Core ML,
  remote APIs, or MLX expose different logits, cache semantics, sampling behavior, telemetry,
  and privacy guarantees?
- How should privacy constraints be represented so a "remote allowed" request cannot be
  accidentally sent off-device because of a latency or availability rule?
- Does Core ML add enough supported-model or power-efficiency value to justify another backend
  contract, given fast-mlx's MLX-first product positioning?

## Next Research Step

Run a later, source-backed feasibility spike that treats the three concepts independently:
measure whether multi-Mac placement or sharding has a credible latency/throughput envelope;
define a minimal signed-artifact provenance schema tied to catalog/dial evidence; and test
whether one non-MLX backend can pass the fast-mlx conformance and measurement contract without
weakening the dial's semantics.
