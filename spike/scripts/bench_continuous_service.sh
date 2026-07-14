#!/bin/bash
# Exact service-policy frontier for simultaneous dense-model bursts. The batch arm disables
# speculation by contract; the comparison arm queues the same concurrent requests onto the
# existing exact solo PLD actor and includes that queue delay in TTFT/completion.
#
# Usage: bench_continuous_service.sh <model-path> [runs] [max-tokens] [prefill-chunk]
set -euo pipefail

MODEL="${1:?usage: bench_continuous_service.sh <model-path> [runs] [max-tokens] [prefill-chunk]}"
RUNS="${2:-3}"
MAX_TOKENS="${3:-128}"
PREFILL_CHUNK="${4:-16}"
CSV="${CSV:-continuous-service-frontier.csv}"
EVIDENCE="${EVIDENCE:-harness-evidence.jsonl}"
BIN="${BIN:-$(ls ~/Library/Developer/Xcode/DerivedData/fast-mlx-spike-*/Build/Products/Release/fastmlx-harness)}"
WORKLOAD_NONCE="${WORKLOAD_NONCE-frontier-$(date -u +%Y%m%dT%H%M%SZ)-$$}"

for value in "$RUNS" "$MAX_TOKENS" "$PREFILL_CHUNK"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
        echo "runs, max-tokens, and prefill-chunk must be positive integers" >&2
        exit 2
    }
done

[[ "$WORKLOAD_NONCE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || {
    echo "WORKLOAD_NONCE must start with an ASCII letter or digit and contain at most 64 letters, digits, dots, underscores, or hyphens" >&2
    exit 2
}

echo "=== workload nonce: $WORKLOAD_NONCE ==="

for policy in batch-no-spec solo-pld; do
    for concurrency in 1 2 4 8; do
        label="service-${policy}-c${concurrency}"
        echo "=== $label ==="
        args=(
            service-bench
            --model "$MODEL"
            --policy "$policy"
            --scenario burst
            --concurrency "$concurrency"
            --max-tokens "$MAX_TOKENS"
            --runs "$RUNS"
            --label "$label"
            --workload-nonce "$WORKLOAD_NONCE"
            --csv "$CSV"
            --evidence "$EVIDENCE"
        )
        if [[ "$policy" == "batch-no-spec" ]]; then
            args+=(--prefill-chunk "$PREFILL_CHUNK" --max-prefill "$concurrency")
        else
            args+=(--ngram 3 --max-draft 8 --compiled-verify false)
        fi
        "$BIN" "${args[@]}"
    done
done

echo "=== $CSV ==="
cat "$CSV"
