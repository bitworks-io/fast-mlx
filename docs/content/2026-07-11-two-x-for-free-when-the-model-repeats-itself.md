# 2× for free, when the model repeats itself: prompt-lookup decoding with a byte-identical proof

**Whitepaper themes:** Building a high-performance MLX engine in Swift; Rapid research integration — the flywheel

> **Follow-up, 2026-07-11:** The two non-target regressions described below were subsequently
> removed: final code measured +3.2%, zero-draft prose +0.1%, and echo +100.5%, while exactness
> remained 120/120 in both verify modes. See the [gate-tuning resolution](../superpowers/verdicts/2026-07-09-pld-firstrun.md#resolved-2026-07-11-performance-gate-cleared-for-a-default-on-product-policy)
> and [the follow-up investigation](2026-07-11-when-zero-speculation-costs-two-percent.md).

Every LLM speedup we had shipped so far traded something. Quantize the weights, pay in
perplexity. Quantize the KV cache, pay in long-context tail divergence — we measured that one
to death and shelved it. So the pitch for speculative decoding sounds like it belongs in the
same bucket: the model emits *several* tokens per forward pass instead of one. Surely
something degrades.

Nothing degrades. That is the whole point of the technique, and the reason we built our first
speculative decoder — prompt-lookup decoding (PLD) — before any of its fancier cousins. This
piece is about the property that makes it safe, the measurement that makes it worth it
(**28 → 56 tokens/sec on our target workload, +97%**), and the two honest asterisks we found
on workloads it doesn't help.

## The trick: the prompt already wrote the next eight tokens

Autoregressive decoding at batch size 1 is memory-bound: every token streams all ~17GB of a
32B 4-bit model's weights through the GPU to produce one token. The compute units are mostly
idle. If you could ask the model to check *eight* proposed tokens in one pass, you'd pay
roughly one pass's bandwidth for up to nine tokens of output.

Where do proposals come from without a second model? From the text itself. Agent and RAG
workloads are drenched in repetition: the model quotes the document it was given, restates
the function signature it's editing, echoes the JSON schema from three turns ago. PLD exploits
this with embarrassing directness — take the last 3 tokens generated, find where that 3-gram
last appeared earlier in the context, and propose whatever followed it. No draft model, no
training, a string search.

The proposals then face a *verify forward*: one batched pass over `[last, draft₁ … draft₈]`
that yields the model's own argmax at every position. Accept drafted tokens exactly while
they match the model's own picks; at the first mismatch, emit what the model wanted instead
(computed for free in the same pass) and roll the KV cache back over the rejected rows. Every
emitted token is one the plain greedy loop would have chosen. Speculation changes *how many*
tokens a forward emits — never *which*.

## "Exact" is a test, not an adjective

That last claim is load-bearing, so we refuse to assert it — we gate on it. Our harness's
spec-decode triad runs the same prompt through the same engine twice, PLD on and PLD off, at
temperature 0, and requires the token streams to be **byte-identical**, plus an engagement
delta proving drafting actually happened (a decoder that never drafts passes equivalence
vacuously). On Qwen3-32B-4bit: 80/80 identical on a repetition-heavy prompt, 120/120 on plain
prose, identical again with the compiled fixed-K verify path. The subtle bits this catches are
exactly the ones you'd fumble silently: the K+1 alignment of the verify argmax, the rollback
count after partial acceptance, the batch replaying the plain loop's stopping rules
(budget and EOS) at the same token boundary.

One implementation detail earned its keep here. Our decode step is wrapped in `MLX.compile`,
which binds to fixed buffer identities — so KV rollback couldn't reallocate anything. We roll
the cache's in-graph offset back in place, leaving the rejected rows as dead weight that the
next update overwrites. The compiled step never notices anything happened.

## The numbers, all of them

Three prompt shapes on Qwen3-32B-4bit (M5 Max, 256 tokens, 3 runs, cold-prefix salted):

| shape | off | on | Δ | drafts accepted |
|---|---|---|---|---|
| preamble-then-echo (agent shape) | 28.25 tok/s | **55.80 tok/s** | **+97.5%** | 98.3% |
| code with repeated structure | 28.41 | 27.36 | −3.7% | 36.6% |
| low-repetition prose | 28.63 | 28.04 | −2.1% | no drafts found |

The echo shape — a passage followed by "a verbatim copy of the passage above:" — is the
distilled version of what agent loops do all day, and it nearly doubled. Acceptance ran at
7.86 of 8 drafted tokens per verify. For calibration, the incumbent engine we're replacing
gained +25% from its PLD on a comparable shape; the batched-verify-plus-rollback core here
more than clears that bar.

The other two rows are the honest part. Code accepted only a third of its drafts — repeated
identifiers, novel logic between them — and each failed speculation is a wasted 9-position
pass. A self-managing yield gate watches a sliding window of acceptance and switches PLD off
when it stops paying (it disabled for 29% of the code run's steps), but its full-window
judgment and eager re-probe leave it a half-step slow. Prose is subtler: the drafter never
found a single match, so its −2.1% is pure loop overhead — and the biggest culprit is that
the speculative loop gives up the plain loop's pipelining trick, where the next forward is
submitted *before* the current token's readback. A host-side drafting decision now sits
between the two. Both fixes are scoped: judge the gate on partial evidence and back off its
re-probes; restore submit-first overlap on the steps where no draft exists.

## The lesson

Two, really. First: **know your workload's entropy.** PLD's frontier isn't
speed-versus-quality — quality is pinned by construction — it's speed-versus-repetition, and
repetition is a property of *your* traffic, not of the model. On agent traffic this is the
cheapest 2× available; on creative prose it's a rounding error below zero until the gate
tightens. That's why it ships as a per-request toggle with the gate as a safety net, not as
an unconditional default.

Second: a technique whose correctness claim is *provable* should be **proven per-run, in CI
shape, forever** — our verify subcommand will re-assert byte-identity on every future engine
change, so a regression in the rollback arithmetic can never masquerade as "the model being
the model." When quality loss is exactly zero, you get to demand exactly zero. And the
scaffolding — drafter protocol, accept-walk, KV rollback, yield gate, the byte-identity triad
— is the reusable half of this work: the next drafters (model-based, block-diffusion) slot
into the same verify loop and inherit the same proof obligations.
