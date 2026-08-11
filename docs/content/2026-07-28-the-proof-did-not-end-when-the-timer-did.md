# The proof did not end when the timer did

**Whitepaper themes:** Building a high-performance MLX inference engine in Swift; Serving big
models on Apple Silicon; Rapid research integration — the flywheel

A short benchmark can show that continuous batching works. It cannot show that an HTTP service
keeps cleaning up after disappearing clients for a full day.

That distinction mattered for fast-mlx. The engine already had exact temperature-zero continuous
decode, bounded admission, an OpenAI-compatible HTTP/SSE route, and tests for cancellation and
resource release. The remaining question was operational: could one source-locked 32B model stay
resident while repeated C=4 workloads mixed ordinary streams with clients that disconnected after
admission but before receiving a body?

The answer required 24 hours of evidence, plus another six minutes of restraint.

## The workload made cancellation part of every cycle

The soak used the explicit `continuous-batch-no-spec` route with a source-locked
Qwen3-32B-4bit model on an Apple M3 Ultra host. It dropped one warmup, then retained
86,420.985839 measured seconds across 1,727 measured cycles.

Each cycle produced six request/evidence rows. One hostile client disconnected after admission
and before body delivery. The other five completed; the final one was a recovery request that had
to leave the runtime at zero active requests, zero coordinator slots, and zero reserved KV bytes.

Across the dropped warmup and 1,727 measured cycles, that produced:

- 10,368 exactly paired request/evidence rows;
- 8,640 completed responses;
- 1,728 typed, zero-body `clientDisconnected` rows;
- a maximum cancel-to-evidence interval of 2.398127 seconds against a five-second gate; and
- a maximum recovery interval of 12.019825 seconds against a 30-second gate.

The request, response, and request/response-pair digests matched independently. Every retained
evidence row named the explicit no-spec route. Active requests and coordinator slots never
exceeded four.

These are transport and recovery numbers, not throughput numbers. The run did not measure a new
tokens-per-second frontier, production traffic diversity, or an automatic routing policy.

## Memory needed a baseline, not an adjective

“No leak” is an attractive phrase and a poor measurement.

fast-mlx instead froze a post-warmup process-RSS baseline of 18,165,488 KiB. The maximum over the
run was 18,168,704 KiB, for a peak/baseline ratio of 1.000177039. That is about 0.0177% above the
baseline and inside the declared 5% gate.

MLX active, cache, and peak memory topped out at 20,442,968,648, 5,527,599,705, and
22,099,490,330 bytes. The run stayed inside its explicit 96-GiB MLX-memory and 8-GiB cache limits.
No resource snapshot failed.

The host contract was measured too. All 291 retained environment samples reported AC power,
Foundation low-power mode disabled, no performance warning, no thermal warning, and successful
probe commands. That does not make every Apple Silicon machine equivalent. It makes this
particular result interpretable.

## The timer was not the terminal state

At 86,400 measured seconds, the workload had satisfied its duration gate. The result still was
not promotable.

The mutable result root first froze a `FINALIZING` receipt. That receipt deliberately remained
nonterminal. The runner then had to snapshot and hash the exact artifact set, scan the retained
files for registered raw prompt, key, and output windows, stop the service, release the port and
locks, and prove that no launcher, client, server, watchdog, or orphan remained.

Only after those checks did a create-only immutable sibling receipt publish `COMPLETE`.

That sequencing prevented a subtle but serious evidence bug: success could not be declared while
the proof was still changing. The final manifest contains 42 authenticated entries. The terminal
sensitive scan passed with no registered raw value retained, and the final process, listener,
lock, watchdog, and orphan counts were all zero.

The canonical terminal receipt appeared at 06:21:58 UTC, roughly six minutes after the root froze
its nonterminal completion data. Those six minutes are part of the result. They are the difference
between “the workload ran long enough” and “the evidence is complete, immutable, and safe to use.”

## The claim stays narrow

This proof qualifies fast-mlx's explicit, temperature-zero text
`continuous-batch-no-spec` HTTP/SSE route for the measured source-locked model and workload.

It does not authorize dynamic prompt-lookup routing, shared-batch speculation, a broad
model-family default, sampling, tools, media, WebSocket transport, or unauthenticated remote
binding. It also does not convert a stability soak into a speed claim.

The general lesson is less glamorous than a benchmark chart: long-running systems need a terminal
protocol for their evidence. Duration, pairing, recovery, resources, environment, provenance,
redaction, and cleanup all have to agree. The proof is not finished when the timer stops. It is
finished when nothing mutable is left that could change what the timer meant.
