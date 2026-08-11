# “Lossless” wasn't byte-identical: the speculative decoder that failed at generated index seven

**Whitepaper themes:** Rapid research integration — the flywheel; Building a high-performance
MLX engine in Swift

EAGLE-3 looked like the trained speculative decoder we had been waiting for. A public
Qwen3-32B checkpoint matched our production-size target. The draft head was only one decoder
layer. Its published algorithm was familiar: draft several tokens, ask the target to verify
them in one forward pass, keep the longest matching prefix, and emit the target's correction
or bonus. That is exactly how the
[Speculators documentation describes it](https://github.com/vllm-project/speculators/blob/d1a3ff3ed6a48f990584f56efbb06f990e1c7ab2/docs/user_guide/algorithms/eagle3.md#inference-process).

On an M5 Max, the first numbers looked excellent. The 4-bit target appeared to rise from about
27.6 to 38.6 tokens per second. The 8-bit target appeared to rise from 15.1 to 23.9. The head
was engaged and accepted roughly 0.59 and 0.66 draft tokens per target verify round.

We shelved it.

The apparent multiplier generated a different answer.

## First prove the port, then distrust the result

There are many easy ways to port an external drafter incorrectly: choose the wrong hidden
layers, add Q/K normalization that the serialized layer does not have, misalign RoPE by one
position, or reverse a vocabulary map. We removed those explanations first.

The gate authenticated the complete 3.1 GB draft checkpoint, not merely its filename or
Hugging Face metadata. It pinned the config hash, full weight hash, all 16 tensor names,
dtypes, shapes, and payload ranges. It separately hashed every target shard—18.43 GB for
4-bit and 34.81 GB for 8-bit (about 17.2 GiB and 32.4 GiB)—before loading either model.

Then a fixed PyTorch/speculators fixture and the MLX port produced a cosine similarity of
0.9999816 and identical argmax tokens. The head math was faithful. That moved suspicion to
the target verification path.

The first exactness smoke had passed on a short prompt. A more realistic code prompt did not.
The 4-bit stream diverged at generated index 17—the 18th generated token. The 8-bit stream
diverged at index 7, the eighth generated token. Both
failures reproduced on a clean source SHA, with the same authenticated target and temperature
zero.

That is why a single “looks coherent” generation is not an equivalence test. A speculative
decoder can run correctly for several rounds, accept real drafts, and then quietly choose a
different target token.

## Four replays and one uncomfortable answer

We extended the verifier to replay the first mismatch four ways:

1. the live history of full verify batches followed by rollback;
2. a history that processes only the tokens eventually retained;
3. an entirely sequential history followed by the same multi-token probe; and
4. the entirely sequential history followed by an ordinary one-token probe.

Every replay recorded its cache offset. That matters because an off-by-one rollback is the
obvious suspect.

For the 4-bit target, all four offsets were 59. The baseline expected token `12`. Both
sequential probes and the retained-only history predicted `12`; only the live history that had
processed and rolled back rejected future tokens predicted `44364`. The integer cache position
was right, yet the retained state no longer behaved like the autoregressive state.

For the 8-bit target, all offsets were 49. The difference was even more direct. From the same
sequential prefix, a one-token probe predicted the baseline token `279`. Passing
`[current,draft]` instead predicted `264`. No accumulated rollback history was required to
flip the argmax.

MLX's own documentation promises compiled and uncompiled equality only
[up to numerical precision](https://github.com/ml-explore/mlx/blob/4367c73b60541ddd5a266ce4644fd93d20223b6e/docs/src/usage/compile.rst#L39-L40)
and notes that changing input shapes can trigger compilation work. mlx-lm's standard
[`KVCache.trim`](https://github.com/ml-explore/mlx-lm/blob/f3ed856610d3852e41d691b8968021040f9c4a6b/mlx_lm/models/cache.py#L307-L376)
decrements the logical offset; it does not promise that a token computed inside a wider
forward has bitwise-identical cached activations to the same token computed alone.

Those sources describe a plausible finite-precision, shape/order mechanism. They do not prove
the cause of our MLX 0.32 observation, and we should not turn one failed pairing into a general
framework claim. The replay itself is the proof we need: these target computation shapes do
not preserve this target's greedy byte stream.

## Algorithmically lossless is not a product proof

Speculative decoding is lossless in the algorithmic sense when the target distribution owns
every emitted token: accept draft tokens only when the verifier agrees, otherwise emit the
verifier's token. That proof assumes the verifier used for the batched round represents the
same target distribution as ordinary autoregressive decoding.

Our product contract **for greedy speculative decoding** is stronger and more concrete:
temperature-zero output must be byte-identical to the engine's base loop. Different numerical
shapes can be mathematically close and still cross an argmax boundary. Once that happens,
“the target approved every token” is not enough—the two target executions disagreed about
which token the target wanted.

That exactness rule is not a ban on lossy optimization. The fast-mlx dial is explicitly meant
to offer quantization, compressed caches, approximate attention, and other tiers with real,
teacher-forced quality loss when the speed, memory, or runs-at-all benefit is worth it. Those
techniques have a measured quality frontier and a hard coherence/garbage floor. EAGLE was
different: it promised the same target result, and its shape-dependent mismatch provided no
stable or quantified loss control for a user to choose.

There is also an economics trap. Published `acceptance_length` commonly includes the one
target correction or bonus that every verify round emits. The quantity that pays for draft
and verify overhead is accepted **draft** tokens per round, one lower. Our observed inclusive
lengths of 1.59 and 1.66 therefore meant draft yields of only 0.59 and 0.66. Even without the
correctness failure, those numbers must be compared with this pairing's own measured round
cost, not an acceptance headline from another model.

## The benchmark we refused to publish

It is tempting to retain the 1.40× and 1.59× apparent speedups as “promising.” We will not.
A changed continuation can alter subsequent compute and stopping behavior. A throughput ratio
between different outputs is not a speculative-decoding speedup.

An exact repair is conceivable: recompute or commit retained target tokens through the
one-token path, use a shape-stable verifier kernel, or wait for a target/runtime combination
that produces identical state. But replaying retained tokens charges extra target forwards,
which may erase the multiplier. Any repair has to restart the same flywheel: exactness first,
then end-to-end economics.

That is the value of a hardened integration gate. It does not merely tell us when a technique
is fast. It stops an attractive number from becoming a feature when the number answers the
wrong question. EAGLE-3 passed checkpoint fidelity, head parity, and engagement. Generated
index seven still said no.
