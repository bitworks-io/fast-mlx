# Continuous serving Phase 5 verdict — the full-day transport boundary passed

- **Date:** 2026-07-28
- **Lane:** exact temperature-zero serving
- **Classification:** EXACT
- **Engine source:** `1c084f305da83c3ab2e399368b5bd32f14dabd09`
- **Decision:** **PROMOTE the explicit, model-scoped `continuous-batch-no-spec` HTTP/SSE route.
  Keep dynamic solo PLD shelved and keep every broader surface fail-closed.**

## Operator story and hard gates

A local operator can keep one source-locked Qwen3-32B-4bit model resident, submit concurrent
temperature-zero text requests over the OpenAI-compatible HTTP/SSE boundary, disconnect a client
after admission, and recover without corrupting survivor output or retaining its slot or KV.

Promotion required more than a short functional smoke. One fresh dedicated-host boundary had to
drop its warmup, retain at least 86,400 measured seconds at C=4, pair every request and response,
record every hostile close as a typed zero-body `clientDisconnected` row within five seconds, and
recover within 30 seconds to zero active requests, coordinator slots, and reserved KV. MLX active
memory and cache had to stay inside 96 GiB and 8 GiB, post-warmup RSS drift had to stay at or below
5%, and the power/thermal contract had to remain stable.

The proof also had to authenticate the complete source snapshot, dependency resolution, clean
Release binary, model files, workload, tools, policy projections, packet, receipts, artifact set,
redaction scan, and final process/port/lock cleanup. Partial or diagnostic boundaries could not
promote.

## What passed

The canonical terminal receipt for
`continuous-serving-phase5-transport-soak-1c084f3-m3ultra-v2` is `COMPLETE`, terminal,
non-diagnostic, and promotable. It records 86,420.985839 measured seconds; an independent monotonic
witness records 86,420.115586 seconds.

After one dropped warmup, 1,727 measured cycles completed:

| Proof | Result |
| --- | ---: |
| Cycle receipts | 1,728 total: 1 warmup + 1,727 measured |
| Request/evidence rows | 10,368 / 10,368 |
| Normal completed responses | 8,640 |
| Typed zero-body disconnect rows | 1,728 |
| Maximum cancel-to-evidence | 2.398127 s against 5 s |
| Maximum recovery | 12.019825 s against 30 s |
| Maximum active requests / coordinator slots | 4 / 4 |
| Final recovery requests / slots / reserved KV | 0 / 0 / 0 |

The request, response, and request/response-pair multiset digests match independently. Every
schema-2 evidence row names only `continuous-batch-no-spec`; there is no mixed route or unmatched
transport row.

## Resource and environment boundary

MLX active/cache/peak memory topped out at 20,442,968,648 / 5,527,599,705 /
22,099,490,330 bytes, inside the declared 96-GiB memory and 8-GiB cache limits. Process RSS moved
from a post-warmup baseline of 18,165,488 KiB to a maximum of 18,168,704 KiB. The exact
peak/baseline ratio is 1.000177039, well below the 1.05 gate.

All 291 retained environment samples report AC power, Foundation low-power false, no performance
warning, no thermal warning, and successful probes. This is a controlled dedicated-host
transport-soak result, not a production-traffic or speed benchmark.

## Why the sibling receipt is canonical

The root receipt intentionally freezes at `FINALIZING`, nonterminal and non-promotable. That
prevents the mutable result root from declaring success before the artifact set, terminal
sensitive scan, and cleanup are finished. Only after those operations does the create-only
immutable sibling receipt publish the terminal decision.

The canonical sibling hashes to
`f910fd570f4fd6d6c1f749dbe16f6d1dfffb8cb7953ebb04cf43aca8b641a608`; its sidecar hashes to
`7d5d2fd0cd801a96de7f5a4c8c21f81b0c003cb1060410895e1cece315c5f714`. All 42 manifest entries
reauthenticate. No artifact-publisher incoming file, retained raw prompt/key/output window,
process, listener, held lock, watchdog, or orphan remains.

The preceding V30 preflight is also terminal `COMPLETE`, but remains diagnostic and non-promotable
by design. It proves the repaired terminal artifact-publication path with one warmup, three
measured C=4 cycles, 24 evidence rows, four typed disconnects, bounded recovery, sealed artifacts,
and a clean sensitive scan.

## Provenance

The soak binds:

- the 527-file source manifest
  `a84b8c9b4de50766c1ec83ed818a4ff76d12128bf77df82a1c831357208bb328`;
- `Package.resolved`
  `7fa8102041ba82ae0347f69f17fb439a55c379420a1f2406b900f7d91020269e`;
- the clean non-profiled Release server
  `f5bfc31fe6cccf8a8aae5cee2c8596052b5a9d594e4ca971df39be4f6ac8a41e`;
- all 12 source-locked model files, totaling 18,445,872,022 bytes, under receipt
  `7853c49616b05e2fd4fa043448959d1c0d6f5a7821967e26cfa80c91865941e5`;
- the exact workload, thermal policy, host contract/admission, Metal toolchain, and 16 runtime
  tool binaries; and
- packet client, launcher, manifest, precondition-audit, stage-admission, and launch-receipt
  hashes.

The terminal sensitive scan is `PASS`, reports `rawValuesPersisted: false`, and retains an empty
scan log with the standard empty-file SHA-256.

## Support and product disposition

| Surface | Disposition |
| --- | --- |
| Explicit Qwen3-32B-4bit `continuous-batch-no-spec` temperature-zero text HTTP/SSE route | promoted for the measured model-scoped boundary |
| Scalar route and fail-closed validation | retained |
| Dynamic solo PLD | shelved; not exposed or authorized |
| Shared-batch speculation | rejected |
| Broad model-family default | not authorized |
| Sampling, tools, media, WebSocket, or unauthenticated remote binding | outside this evidence and not authorized |

This verdict closes Continuous Serving Phase 5. It does not turn the 24-hour stability proof into
a throughput claim, and it does not expand the supported product surface beyond the exact route
and workload that were measured.

## Reopen condition and next action

Reopen this boundary only if its source, binary, model, route, workload, cancellation/recovery
contract, resource limits, or terminal publication protocol changes. A new claim requires a new
fresh root and the same fail-closed evidence discipline; no existing diagnostic or partial root may
be resumed or promoted.

After the required reviewed closeout commit and `--no-ff` integration, the next ranked roadmap
item is absorbed MLA.

Machine-readable criterion mapping:
[`continuous-serving-phase5-verification-2026-07-28.json`](continuous-serving-phase5-verification-2026-07-28.json),
SHA-256 `b1f7bb933e65dfb93c4043d64e1426b1c464b1828edebbcf425eabf6f798e22a`.
