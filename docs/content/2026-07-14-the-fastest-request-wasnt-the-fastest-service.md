# The fastest request wasn't the fastest service

**Whitepaper themes:** Building a high-performance MLX inference engine in Swift; Rapid
research integration — the flywheel

On an Apple M5 Max at one request, our fastest exact path was prompt-lookup decoding.
Qwen3-32B-4bit generated
28.30 tokens per second with PLD, versus 26.72 through the new continuous-batching runtime.
If we had optimized the request in front of us, the decision would have been easy: keep PLD.

Then a second request arrived.

At two simultaneous requests, continuous batching delivered 42.70 aggregate tokens per second.
The single-request PLD actor, asked to serve the same two requests exactly, delivered 29.29.
At eight, the gap widened to 56.56 versus 32.37. The fastest request path was no longer the
fastest service policy.

That result sounds obvious in retrospect. It was surprisingly easy to measure dishonestly.

## Our first comparison changed the workload

The benchmark salts every prompt so a repeated run cannot accidentally hit a warm prefix and
masquerade as a faster decoder. The first service sweep launched each policy cell as a separate
process, and each process generated its own random salt nonce. The labels said “batch” and
“solo PLD,” but the tokenized prompts were not identical.

That invalidates a direct performance comparison. A different suffix can change token count,
generation, stopping, and PLD's opportunity to find repeated n-grams. Even a small prompt
difference is unacceptable when the policy decision turns on a measured percentage.

We discarded the preliminary table. It contributes no evidence to the verdict.

The corrected runner creates one workload identity—`frontier-20260714-final`—and passes it to
all eight processes. For a given run and request index, both policies now receive the same
salted prompt. Prompt and visible-output counts match across every paired run. Each cell drops
one warmup and aggregates three measured repetitions from a Release build.

The fix is small. The principle is not: an apples-to-apples benchmark should encode workload
identity as data, not rely on two command lines happening to construct the same input.

## The corrected frontier

The benchmark measures the whole simultaneous burst. Aggregate rate includes queueing and
prefill; TTFT starts when the client submits the burst; TPOT describes a request after its first
token. That matters because solo PLD and continuous batching spend waiting time differently.

| Concurrency | Batch, no spec | Queued solo PLD | Batch change |
| ---: | ---: | ---: | ---: |
| 1 | 26.72 tok/s | **28.30 tok/s** | −5.6% |
| 2 | **42.70** | 29.29 | +45.8% |
| 4 | **51.44** | 32.44 | +58.6% |
| 8 | **56.56** | 32.37 | +74.7% |

Batching scaled 2.12× from one to eight streams. More importantly, it advanced requests
together. At concurrency eight, batch TTFT was 2.63 seconds p50 and 2.71 seconds p95, with a
Jain completion-rate fairness index of 1.000. Queued solo PLD reached 15.39 seconds p50 and
28.08 seconds p95; fairness fell to 0.626 mean and 0.577 minimum.

PLD's per-request TPOT stayed near 33–35 milliseconds once a request started. That is the
single-request speed doing exactly what it promised. The actor still had to serialize those
requests, so later callers waited behind earlier ones. Batching made each stream's TPOT slower
as concurrency rose—120–125 milliseconds at eight—but used one shared model forward to advance
all eight rows. Aggregate service rate and fairness improved even though an individual active
stream stepped less often.

The policy boundary is therefore measured, not ideological: use solo PLD for an isolated
request; for this simultaneous burst at concurrency two or greater, use batch-no-spec. PLD
remains disabled inside a shared batch. A future service router can implement that choice, but
this experiment did not wire one.

## Faster is irrelevant if joining changes the answer

Continuous batching changes tensor shapes and cache membership while a generation is live.
The dangerous transition is solo to batch. Our solo decoder keeps one forwarded token of
lookahead in KV state; dropping or replaying it when another request joins can duplicate a KV
position and corrupt both streams.

The runtime drains that lookahead before forming the shared batch. On the final clean harness,
a Qwen3-32B B1→B2→B1 run matched both independent scalar streams token-for-token and byte-for-byte.
A B3→B2 middle cancellation preserved both survivors and the cancelled request's exact prefix.
A deliberately hostile Qwen3-4B run with one-token prefill chunks also remained exact. MoE
fails closed, and shared batches recorded zero speculative decoding.

This is an exact optimization, so there is no quality-loss dial position to explain away. The
user-visible trade is operational: aggregate throughput, TTFT, TPOT, and fairness. That does
not narrow the broader product philosophy. Intentional lossy techniques can still be useful
when teacher-forced measurements quantify the loss and keep them above the coherence floor.

We also replaced a logical-token memory proxy with byte-denominated dense-Qwen admission. The
reservation includes allocation rounding and a conservative five-copy envelope for a cache
membership rebuild. It protects the transition without pretending to equal process RSS or to
describe MoE, recurrent, or vision state.

## Twenty-four hours later

A short benchmark can prove a crossover and still miss a leak, a poisoned cache, or a request
that never releases its slot. We kept Qwen3-32B resident for a full post-warmup day. The measured
interval was 86,412.85 seconds, across 3,518 post-warmup cycles.

Every cycle mixed known-good chat, agent-recall, tool, and Anthropic-style prompts with a hostile
disconnect/cancellation and then repeated the known-good burst. All 33 predicates passed in all
3,519 total cycles, including the warmup: before/after output stayed byte-identical, replacement
slots were reused, no batched speculation engaged, and KV reservations returned to zero.

Peak resident memory grew 2.2444% from the completed-warmup baseline, below the 5% gate.
Responsiveness topped out at 344.469 milliseconds against a 30-second deadline. Runtime
cancellation averaged 13.789 microseconds and peaked at 28.833 microseconds against a one-second
boundary. The external progress watchdog never fired, and its process and lock were cleaned.

Those are runtime results, not a claim about a production HTTP server. No serving/API route or
default policy is wired yet, and a real client's disconnect still has to be propagated through
that future boundary.

## Measure the queue, not just the kernel

The general lesson is not “batching wins.” At concurrency one, it lost. The lesson is to measure
the unit the user experiences.

A request benchmark rewards the fastest active stream. A service benchmark charges queueing,
prefill, fairness, cancellation, and recovery. Both are legitimate; they answer different
questions. Shared workload identity keeps the comparison honest, exactness ensures the policies
produce the same result, and a resident soak checks that the win survives long enough to matter.

That evidence supports a narrow promotion: exact dense-Qwen continuous batching is ready as a
service-policy building block. The policy, API, additional architectures, sampled generation,
and other Apple hardware still have to earn their own results. The fastest request gave us one
number. The queue told us what to build.

Full measurements and claim boundaries are preserved in the project's reviewed evidence ledger.
