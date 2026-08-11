# When zero speculation costs 2%: making a 2× decoder safe to leave on

**Whitepaper themes:** Building a high-performance MLX inference engine in Swift; Rapid research integration — the flywheel

Prompt-lookup decoding had already given us the result every inference team wants: nearly
twice the decode throughput, with byte-identical output. On a repetition-heavy agent prompt,
Qwen3-32B-4bit rose from about 28 to 56 tokens per second. There was no smaller model, no
sampling approximation, and no precision loss. The target model simply verified several
tokens copied from its own context in one forward pass.

Yet we would not leave it on by default.

The reason was hidden in the boring rows of the benchmark. Code lost 3.7%. Low-repetition
prose lost 2.1%. The prose run was particularly incriminating: the prompt-lookup drafter
proposed zero tokens. The system performed no speculative work at all and still became
slower. An optimization that charges rent while idle is not a safe default, however good its
best case looks.

## The first diagnosis was only half right

We had built a yield gate around speculation. It measured accepted draft tokens and disabled
prompt lookup when proposals stopped paying for their batched verify forwards. The obvious
hypothesis was that the gate was too patient. It required a complete 32-step window before
judging a request, disabled at a low threshold, then probed again after only 16 ordinary
steps. That explained code: repeated identifiers generated plausible drafts, but novel logic
rejected many of them, so the decoder paid for low-yield verifies too long.

It could not explain prose. With no draft, there was no verify forward for the gate to avoid.
The loss came from the loop around it.

The base decoder uses submit-first pipelining. While the host reads the current token back
from the GPU, the next compiled forward is already in flight. Our first speculative loop
could not preserve that rhythm: it synchronously read each token, searched for a draft, and
only then submitted the next forward. Even when the search returned nothing, the overlap was
gone. The feature flag had selected a slower base loop.

That is the more general failure mode. Performance features are often judged by the cost of
their special operation—here, the batched verify. Their control flow can be just as expensive,
especially on Apple Silicon where a host readback can break carefully arranged GPU overlap.

## Two states, one exact sequence

The repair made the decoder's two legal states explicit.

In the pipelined state, committed context is already in the KV cache and the model's next
greedy pick is pending. In the speculative state, the last emitted token has not yet entered
KV and no pick is pending. Cold rounds stay on the exact same `prefill` and `step` path as the
plain decoder. After a speculative round turns cold, one two-deep transition seeds the base
pipeline again; following rounds reuse it unchanged.

Entering speculation from the pipelined state raised a delicate alignment problem. The first
target pick already existed, while the verify forward produced picks only after the proposed
draft tokens. A small accept-walk variant prepends that prefetched pick conceptually, restoring
the same K+1 target sequence used by the original algorithm. Drafted tokens are still emitted
only when they equal the target model's argmax, rejected KV rows are still rolled back in
place, and the bonus token is still the target's own pick.

Then we made the yield gate less sentimental. Its window fell from 32 to 8, it may judge an
obviously bad partial window after four samples, its threshold rose from 0.25 to 0.5 accepted
tokens per enabled step, and its cooldown doubled to 32. Four zeros are enough evidence that
prompt lookup has nothing to offer; the decoder steps aside and gives the base pipeline room
to work.

## The result: the idle tax disappeared

We repeated the same three-shape benchmark on an M5 Max with Qwen3-32B-4bit, 256 generated
tokens, and three post-warmup runs:

| shape | PLD off | PLD on | change |
|---|---:|---:|---:|
| preamble-then-echo | 28.28 tok/s | **56.70 tok/s** | **+100.5%** |
| code with repeated structure | 28.39 | **29.31** | **+3.2%** |
| low-repetition prose | 28.62 | **28.66** | **+0.1%** |

Echo preserved the 2× headline at 98.3% draft acceptance. Code moved from a regression to a
small real win: the gate ignored enough weak periods while retaining useful repeated spans.
Prose proposed and accepted zero drafts and landed within fourteen hundredths of a percent of
the base loop. That last row is the proof of the repair: doing nothing once again costs
nothing measurable.

We reran the property that matters more than speed. With compiled verification both off and
on, PLD emitted byte-identical 120/120-token streams against ordinary greedy decoding at
temperature zero. Restoring overlap did not weaken exactness.

The fixed-K compiled verify path remained slower than ordinary batched verification—52.63
versus 56.70 tokens per second on echo—so it stays non-default. Optimization work includes
declining clever machinery when measurement says the simpler path wins.

## A default is a worst-case promise

A benchmark champion only needs a spectacular best case. A default needs a credible worst
case. That means measuring empty engagement, failed engagement, recovery, and the transitions
between them—not just the path that appears in the technique's paper.

Prompt lookup now clears that performance gate. The product can adopt it by default with an
opt-out, while the current harness continues to expose it explicitly for controlled A/B
measurement. The remaining boundary is named: it assumes one in-flight KV and stays disabled
under continuous batching until that combination is designed and measured.

The lesson for the integration flywheel is simple. Promote a technique only after its fast
path wins, its exactness or quality cost is proven, and its idle path becomes boring. The
boring row is what makes the 2× row shippable.
