# Sampling before serving: why the random draw needed its own contract

**Whitepaper themes:** Building a high-performance MLX inference engine in Swift; Rapid research
integration — the flywheel

Sampled generation sounds like a small change: take logits, apply temperature and top-p, draw a
token, and continue. In a serving engine, that “draw a token” step quietly touches request
identity, retries, cancellation, batching, numerical stability, resource limits, and evidence. We
implemented the deterministic CPU foundation first so those decisions can be tested before any
runtime is allowed to depend on them.

The result is intentionally narrower than a sampled-generation product feature. It is not sampled
serving support, not an MLX device path, and it carries no model or tokenizer claim and no
performance claim. It is a dependency-free HarnessCore contract that turns one finite rank-1
logit vector and one explicit policy into one reproducible token selection.

## The random draw is an address, not mutable global state

The foundation does not keep a process-wide random-number generator. A draw is addressed by the
caller's signed 64-bit seed bit pattern, a fixed sampling domain, and a committed sample ordinal.
Those integers are encoded in big-endian order after a versioned domain prefix and hashed with
SHA-256. The first eight digest bytes become an unsigned big-endian word; its high 52 bits map to
the midpoint of one floating-point interval strictly inside zero and one.

That sounds more elaborate than incrementing a generator. It buys an important property: the
same address always produces the same word. A future scheduler can retry work without stealing an
extra draw, then advance the ordinal only after it commits a selection. The current foundation
defines that behavior but does not yet connect it to a scheduler, cancellation path, or request.

Greedy selection remains separate. A zero-temperature policy chooses the first maximum logit and
consumes no draw. Positive-temperature sampling requires an explicit seed.

## Distribution order is part of correctness

The oracle rejects empty, oversized, non-finite, or invalid-policy input before selection. For a
sampled policy it subtracts the maximum logit, applies temperature before top-p, exponentiates,
and accumulates probability mass with sequential Neumaier summation.

Candidates are ranked by descending weight with token ID as the stable tie-breaker. The nucleus
cutoff is inclusive: the first candidate that reaches the requested mass stays in the retained
set. The retained set is then traversed in ascending token-ID order, using a strict
greater-than CDF threshold. These ordering choices are observable. Changing one can change the
token selected at an exact boundary even when the probability distribution looks equivalent.

The tests therefore freeze fixed SHA-256 words, floating-point bit patterns, hand-derived
temperature and top-p cases, half-open CDF boundaries, maximum ordinals, negative-zero policy
normalization, caller independence, and three complete 65,536-seed cohorts. Current revalidation
runs all 13 focused cases with zero failures.

## Bounds come before allocation

The public contract accepts at most 262,144 logits. On the current Swift target, `Double` has an
8-byte stride and an `(Int, Double)` candidate has a 16-byte stride. Before building sampled-path
arrays, checked arithmetic accounts for two Double arrays and four candidate-array equivalents:
20,971,520 bytes at the maximum vocabulary size, below the inherited 256 MiB ceiling.

That ledger is conservative by design. It prevents a later refactor from turning a small policy
surface into unbounded or quadratic memory growth. The returned selection retains only the token
ID, draw address, and random word—not the caller's logits, probability arrays, or an RNG object.

Fresh coverage records 349 of 357 coverable lines, 25 of 27 functions, and 140 of 146 regions.
The remaining invariant and terminal-fallback branches are structurally reviewed without adding
a fault-injection seam that production code could accidentally depend on.

## Evidence can pass now without rewriting a failed past

The implementation arrived in the engineering history before its planned green evidence packet
was completed. The original focused run widened into unrelated package and plugin compilation,
returned a live session, was interrupted, and executed zero tests. That run remains rejected and
non-promotable.

We did not backfill a historical pass. Instead, a corrective checkpoint authenticated the current
source and accepted test blobs, reran focused behavior and coverage, built the dependency-free
target in Debug and Release, ran the broader pure regressions, and preserved the package-wide MLX
metallib failure as an environment boundary rather than sampling evidence.

That distinction matters in a self-improving project. New evidence can tell us what is true now.
It cannot make an earlier procedure happen retroactively.

## What comes next

The next work is integration design, not a broader claim. A request adapter must define automatic
seed timing and API validation. A scheduler must own draw commit, retry, cancellation, and batch
invariance. An MLX implementation must match the CPU oracle on real model logits. Only then can
loaded-model correctness, capacity, and performance be measured.

Publishing the foundation makes that work inspectable. It does not skip it.
