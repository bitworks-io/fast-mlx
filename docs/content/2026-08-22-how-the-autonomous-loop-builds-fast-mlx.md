# How the autonomous loop builds fast-mlx

> **Whitepaper themes:** Rapid research integration — the flywheel; Disciplined proof over
> convenient claims; Building a high-performance MLX inference engine in Swift

fast-mlx is an LLM inference engine for Apple Silicon, written in Swift on MLX. It is also an
experiment in how an engine gets built: most of the day-to-day engineering — research intake,
implementation spikes, benchmarking, regression tests, documentation, even the site you are
reading — is produced by a scheduled autonomous agent loop. Humans steer; the loop builds.
This note explains the development model itself: what the loop does on each cycle, how work is
routed across model tiers, and why the discipline around proof matters more than any single
optimization.

## A loop with a cadence, not a queue of tickets

The loop wakes on a schedule, reads the durable project state — a handoff document, a task
inbox, dated plans and verdicts — and chooses its own next increment. There is no ticket
assignment and no human gate inside a cycle. When the loop is unsure whether a result is real,
it does not wait for a person; it escalates to a stronger reasoning tier, records the verdict
with evidence, and proceeds. People review the record and redirect the loop between cycles,
the way one steers a research group rather than a script.

Because chat history is not a reliable memory, everything that matters is written down as
durable state: what was decided, what was measured, what failed, and what the next safe action
is. Any future session — agent or human — can resume from that record alone. The published
research notes on this site are the public face of that same record.

## Routing work to the cheapest tier that can carry it

Not every step deserves the same reasoning power, so the loop routes work across four model
tiers. A fast, inexpensive tier handles reconnaissance: searching the codebase, reading logs,
summarizing documents. A mid tier executes well-scoped implementation — writing tests,
refactors, boilerplate — from a precise spec. A judgment tier owns planning, design review,
and the final read on any diff. The deepest reasoning tier is reserved for the hard residue:
root-cause debugging, architectural decisions, and long autonomous work where a wrong turn is
expensive. If a task defeats one tier, it moves up exactly one tier rather than being retried
in place. The effect is that heavyweight reasoning is spent where it changes the outcome, and
the routine majority of the work stays fast and cheap.

## Spike first, then prove exactness

Every candidate optimization begins life in a spike harness, not in the serving path. The
spike exists to answer two questions in order: is the technique *correct*, and only then is it
*worth it*. Correctness is held to an exactness standard — token-identical output against a
reference implementation wherever the technique claims to be lossless, and a measured,
quantified quality delta wherever it is not. Parity gates compare the new path against the old
one under the same seeds and inputs before any speed number is allowed to matter.

Speed claims then have to survive service-level measurement, not just a flattering
microbenchmark. A kernel that doubles one phase of decoding can still lose end to end; several
of our published notes are exactly that story, such as
[when a 15× phase win still wasn't fast](2026-07-21-fifteen-times-faster-still-not-fast.md).

## Shelving is a result

Many techniques fail these gates. The loop's rule is that a negative result is not deleted or
quietly abandoned — it is **shelved with dated evidence**: what was tried, the exact
configuration, the numbers observed, and the reason it lost. A shelved verdict is a
first-class artifact. It prevents the backlog from lying about what has been tried, it lets a
technique be revisited when upstream conditions change, and it turns failures into publishable
research instead of folklore. Roughly half of the notes on this site are shelved results, like
[the exact-math quantizer that still lost](2026-07-09-turboquant-exact-math-still-lost.md) —
and that ratio is the point.

## Fit-checked serving: refuse early, fail closed

The serving philosophy follows the same discipline. Before a model is served, the engine
checks that it actually fits the machine — weights, KV cache at the requested context and
concurrency, and working headroom — and refuses up front rather than degrading unpredictably
under load. Features that are not wired end to end are not advertised; requests that ask for
unsupported behavior are rejected explicitly instead of being silently approximated. The
public claim boundary works the same way: only reviewed, evidence-backed results are
published, and the site's build tooling rejects anything that drifts from the reviewed record.

## What this buys

The value of the model is compounding throughput with verifiable claims. The loop converts
research into tested code on a cadence measured in hours, while the exactness gates, shelved
verdicts, and sealed publication boundary keep the resulting claims honest — every published
number traces to a dated, reproducible measurement. The engine you can download and the
development story you can read are products of the same process, and that process, as much as
any single kernel, is what fast-mlx is.
