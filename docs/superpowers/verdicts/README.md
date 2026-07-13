# How to read technique verdicts

fast-mlx verdicts are **lane-specific**. A technique is not rejected merely because it changes
model behavior: the product's wedge is an optimization dial that lets users exchange measured
quality for speed, memory capacity, power use, or the ability to run a model that otherwise
would not fit. `SHELVE` means the candidate failed the contract of its declared lane, crossed
the garbage floor, or was dominated by another configuration—not that all quality loss is
forbidden.

Every new technique verdict must declare one evaluation lane:

| Lane | Intended behavior | Promotion contract |
| --- | --- | --- |
| `EXACT` | The technique changes execution, not the model result. Examples include exact caching and greedy speculative decoding. | Pass the technique's equivalence contract and deliver a measured operational or performance benefit. Greedy speculative decoding must be byte-identical to the base loop at temperature zero. |
| `LOSSY_FRONTIER` | The technique intentionally approximates model state or computation. Examples include weight/KV quantization, compressed or selective caches, pruning, and approximate attention. | Measure the context-locked speed/memory ↔ quality frontier; promote useful, non-dominated tiers that remain above the garbage floor, even when the loss is real and noticeable. |
| `EXPERIMENTAL` | Evidence is incomplete, unstable, model-specific, or insufficient for an informed product choice. | Keep behind a research flag until it can satisfy either the exact or lossy-frontier evidence contract. Do not expose an unquantified product tier. |

## Lossy-frontier evidence

Loss is measured **teacher-forced against a locked reference context**, never inferred from a
free-running generation that changes its own future inputs. A promotable lossy verdict should
report, where applicable:

- throughput, latency, memory footprint, and fit/capacity on named Apple hardware;
- KL divergence, pooled perplexity change, and long-context tail statistics;
- representative task or domain checks plus coherence and non-finite-output canaries;
- the model, quantization, context lengths, corpus, runtime versions, and clean harness SHA;
- whether another configuration dominates it at equal quality, speed, or actual packed bytes.

The product may expose conservative, balanced, aggressive, or maximum-compression tiers when
those measurements support informed consent. A hard floor still rejects corrupted state,
non-finite logits, severe coherence failure, catastrophic task collapse, or configurations
that are plainly dominated and therefore give the user no rational trade.

## Exact-lane failures are not lossy tiers

An exact candidate that changes output has failed its correctness contract. It cannot inherit
an apparent speedup by being relabeled after the fact: the changed continuation can alter
subsequent work, stopping behavior, and the measured rate. Reconsidering the underlying idea
as an intentional approximation requires a separately named technique, a new plan, and the
full `LOSSY_FRONTIER` measurement contract.

Each verdict should therefore state: evaluation lane, comparator, hard gates, measured
frontier or equivalence result, disposition, user-dial implication, and what evidence could
reopen the decision.
