#!/bin/bash
# PLD three-shape measurement (spec-decode plan, Task 7): decode tok/s PLD-on vs PLD-off on
#   (a) preamble-then-echo  — agent/high-repetition, PLD's best case
#   (b) code                — repeated identifiers/structure
#   (c) low-repetition prose — the yield-gate must keep this ~flat, not regressed
# plus one compile-strategy comparison cell (echo, fixed-K compiled verify forward).
#
# Prompts are RAW COMPLETIONS (the harness feeds prompts untemplated), so each shape is built
# to make greedy continuation itself carry the shape: the echo prompt ends at the start of a
# verbatim re-statement, the code prompt ends at the signature of the next near-identical
# function. NOTE the bench salt is appended to the prompt END (cold-prefix methodology), so
# every prompt ends at a natural "continuation starts here" boundary that survives a trailing
# marker token.
#
# Usage: bench_pld_shapes.sh <model-path> [runs] [max-tokens]
set -euo pipefail

MODEL="${1:?usage: bench_pld_shapes.sh <model-path> [runs] [max-tokens]}"
RUNS="${2:-3}"
MAX_TOKENS="${3:-256}"
CSV="${CSV:-pld-shapes.csv}"
EVIDENCE="${EVIDENCE:-harness-evidence.jsonl}"
BIN="${BIN:-$(ls ~/Library/Developer/Xcode/DerivedData/fast-mlx-spike-*/Build/Products/Release/fastmlx-harness)}"

PASSAGE="The fast-mlx platform measures every optimization before promoting it. Each technique enters through the intake, is rebuilt exactly from its paper, and is wired behind a flag so the default path never changes. The harness then runs the same three gates every time: an equivalence check against the unmodified engine, an engagement check proving the new path actually executed, and a throughput measurement on realistic workloads. Techniques that win are promoted to a dial tier with their measured quality cost attached; techniques that lose are shelved with a dated negative result so the next agent never re-litigates them. The flywheel only turns one way: absorb, measure, promote or shelve."

ECHO_PROMPT="$PASSAGE

A verbatim copy of the passage above, reproduced word for word:"

CODE_PROMPT="import json

def load_config(path):
    with open(path) as f:
        data = json.load(f)
    validate_schema(data, CONFIG_SCHEMA)
    return normalize_keys(data)

def load_manifest(path):
    with open(path) as f:
        data = json.load(f)
    validate_schema(data, MANIFEST_SCHEMA)
    return normalize_keys(data)

def load_inventory(path):"

PROSE_PROMPT="Explain how continuous batching improves LLM serving throughput."

run_cell() { # label prompt extra-flags...
    local label="$1" prompt="$2"; shift 2
    echo "=== $label ==="
    "$BIN" bench --model "$MODEL" --max-tokens "$MAX_TOKENS" --runs "$RUNS" \
        --label "$label" --csv "$CSV" --evidence "$EVIDENCE" --prompt "$prompt" "$@"
}

run_cell pld-echo-off      "$ECHO_PROMPT"
run_cell pld-echo-on       "$ECHO_PROMPT"  --spec pld
run_cell pld-echo-on-cv    "$ECHO_PROMPT"  --spec pld --compiled-verify true
run_cell pld-code-off      "$CODE_PROMPT"
run_cell pld-code-on       "$CODE_PROMPT"  --spec pld
run_cell pld-prose-off     "$PROSE_PROMPT"
run_cell pld-prose-on      "$PROSE_PROMPT" --spec pld

echo "=== $CSV ==="
cat "$CSV"
