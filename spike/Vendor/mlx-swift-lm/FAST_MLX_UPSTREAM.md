# fast-mlx MLX Swift LM fork provenance

- Upstream: `https://github.com/ml-explore/mlx-swift-lm.git`
- Base revision: `702e5a0eaf990e1f6d3db2b6e7d8872858a44055`
- Upstream license: MIT; retained in [`LICENSE`](LICENSE)
- Snapshot date: 2026-07-18

This repository-vendored fork makes fast-mlx's cache-attention API delta portable and reviewable
without mutating SwiftPM's generated `.build/checkouts` tree. The fork retains the upstream source,
tests, acknowledgments, and license. Local changes must remain narrow, carry tests, and be described
below; a future remote fork may replace this path dependency only at an immutable reviewed revision.

## Local delta

- `MLXLMCommon.KVCache` declares `AttentionKVCacheProtocol`, whose single operation owns cache
  update plus attention without exposing a storage-layout ABI.
- `attentionWithCacheUpdate` checks that protocol before the existing symmetric
  `QuantizedKVCacheProtocol` route. Existing caches and model call sites retain their behavior.
- fast-mlx's affine conformer keeps materialization as its default. The split packed route is an
  explicit per-cache opt-in and records the observed operation in post-run scalar telemetry.
- Reconciled upstream MLX Swift LM PR #351 commit
  `01472a78fca830689ff78246a82c6d31ab111a78` for Qwen3.5 MTP speculative
  decoding. The port keeps fast-mlx's local cache-attention path and adapts the
  PR's Qwen drafter state/one-token hybrid cache rewind semantics to this
  vendor base without importing unrelated staged-cache upstream history. The
  native Qwen drafter types remain package-scoped and intentionally absent from
  model-factory and serving registration; a future loader must first consume
  fast-mlx's exact artifact preflight binding.
- Ported the bounded Qwen3.5 GatedDeltaNet four-input-projection fusion candidate
  from upstream MLX Swift LM PR #572 / commit
  `37688d2cf7d3906e08c74479c9d9949ce6b81136`. In this fork the candidate is
  default-off (`MLX_QWEN_FOUR_GDN=1|true|yes|on`), limited to stock compatible
  `QuantizedLinear` projections, and preserves the existing checkpoint-visible
  Qwen text/VLM projection names by replacing them with storage-sharing views
  only after explicit model preparation.
- `MLXLMCommon.BaseConfiguration.QuantizationContainer` (and its stored properties,
  `init(from:)`, and `encode(to:)`) is widened from `internal` to `public`. This is a pure
  access-level change; the interleaved-quantization decode/encode bodies are byte-identical
  to upstream. A model configuration decoder in the separate `MLXLLM` module names this type
  across the module boundary to decode a checkpoint's root `quantization` /
  `quantization_config` blocks by reusing upstream's interleaved quantization parser
  instead of hand-rolling a second one.

The product remains model-generic. A model whose attention call shape does not use a qualified
router must fail closed for the packed route and continue to use its existing fp16 path.
