# fast-mlx Environments / Hardware Topology

- **Date:** 2026-07-08
- **Purpose:** Pin down which machine is which (the spec/plan refer to "the box"), their roles, and how observability fits. Keep current as infra changes.

## Boxes

| Role | Host | Address | Chip / RAM | Notes |
|---|---|---|---|---|
| **Orchestration / repo / dev-session** | `FluffyMBA` | (separate subnet; reaches .250 + .252) | MacBook Air, **24GB**, macOS 26.5.1, Swift 6.3.3 | Where the fast-mlx **git repo + this Claude session** live, and a local **Grafana (:3000) + OTLP (:4318/:4317)** observability stack runs. Too small to run the engine — it **orchestrates the compute box over SSH**. |
| **Engine compute / test / first deployment** | `llmbench` | **192.168.1.252** | **M5 Max, 128GB**, macOS 26.5.2 | The engine box. Full **Xcode 26.6** (selected), Swift 6.3.3, `iogpu.wired_limit_mb=117760` (115GB), ~3.2TB free. Rich model cache (Qwen3-30B-A3B-2507-4bit, Qwen3-32B 4/8-bit, Qwen3.6-27B, Llama-3.3-70B, Gemma-4 MoE, EAGLE-3 speculator, dspark ckpts). Zig `mlx-serve` prebuilt at `~/mlx-serve-macos-arm64/mlx-serve`. **All Zig perf baselines (e.g. Qwen3-30B-A3B-2507-4bit = 151.8 tok/s decode) were measured here.** Passwordless sudo. The retired `io.mlx-serve` LaunchDaemon (Qwen3-32B-bf16, port 11234) is **stopped + disabled**. **fast-mlx's first production deployment (Concierge) runs here.** SSH: `ssh llmbench@192.168.1.252`. |
| **Dedicated monitoring** | (M5 24GB) | **192.168.1.250** | **M5, 24GB** | Dedicated monitoring node ("can also be utilized"). Candidate home for fast-mlx benchmark + engine observability, and small-model (≤~14B) tasks. SSH creds not yet confirmed — request before use. |

## Execution workflow (spike + engine dev)

Source is version-controlled in the repo on **FluffyMBA**; it **rsyncs to `llmbench` to build (Xcode) and run (model + GPU)**. Results/verdicts come back to the repo for commit. (`swift build` alone cannot compile mlx-swift's Metal shaders — builds use `xcodebuild` on llmbench.)

## Observability integration (opportunity / roadmap)

A Grafana/OTLP stack is already running (locally on FluffyMBA and/or the .250 node), including a **vLLM-compatible serving dashboard**. The mlx-serve lineage already had a `--metrics` (Prometheus) surface + observability work. fast-mlx should emit **OpenTelemetry/Prometheus metrics** (decode/prefill tok/s, TTFT, batch occupancy, KV/cache stats, the dial tier + measured loss per request) and route them to this stack — turning the **benchmark harness and the dial's frontier measurements into live dashboards**, and giving the eventual Concierge production deployment real observability. Wire this in as the harness/engine matures (not v1-blocking, but the plumbing is already here).

## Big-memory / SSD-streaming testing (opportunity)

`llmbench` (128GB) already proved **DeepSeek-V4-Flash 2-bit (86.7GB) via ds4 SSD streaming at 38 tok/s** — so it can prototype "larger-than-fits" quantizer builds + SSD streaming for the dial's **Max-fit tier** and the frontier catalog, even before 256/512GB hardware. Queue this as a follow-on experiment after the Swift gate.
