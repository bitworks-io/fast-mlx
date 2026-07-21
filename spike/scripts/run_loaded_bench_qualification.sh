#!/bin/bash
# Run one loaded-model bench process per counterbalanced matrix position and retain authenticated
# environment/RSS receipts. Every invocation requires a new output boundary; interrupted evidence
# is preserved and is never silently resumed or overwritten.
# Usage: run_loaded_bench_qualification.sh <manifest.json> <fresh-output-root>
set -euo pipefail

MANIFEST="${1:?usage: run_loaded_bench_qualification.sh <manifest.json> <fresh-output-root>}"
OUTPUT="${2:?usage: run_loaded_bench_qualification.sh <manifest.json> <fresh-output-root>}"
RUNNER_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$RUNNER_PATH")"
BIN="${BIN:-}"
if [[ -z "$BIN" ]]; then
    binary_candidates=(
        ~/Library/Developer/Xcode/DerivedData/fast-mlx-spike-*/Build/Products/Release/fastmlx-harness
    )
    BIN="${binary_candidates[0]}"
fi
HARNESS_SHA_FILE="${HARNESS_SHA_FILE:-$SCRIPT_DIR/../.harness-sha}"
POLL_SECONDS="${POLL_SECONDS:-5}"
WATCHDOG_SECONDS="${WATCHDOG_SECONDS:-7200}"
KILL_GRACE_SECONDS="${KILL_GRACE_SECONDS:-15}"

fail() {
    echo "loaded bench qualification: $*" >&2
    exit 2
}

for command_name in jq shasum ps stat; do
    command -v "$command_name" >/dev/null 2>&1 \
        || fail "$command_name is required"
done
[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] \
    || fail "manifest must be a regular, non-symbolic-link file"
[[ -f "$RUNNER_PATH" && ! -L "$RUNNER_PATH" ]] \
    || fail "runner must be a regular, non-symbolic-link file"
[[ -n "$BIN" && -x "$BIN" ]] || fail "BIN must name an executable fastmlx-harness"
[[ -f "$HARNESS_SHA_FILE" && ! -L "$HARNESS_SHA_FILE" ]] \
    || fail "HARNESS_SHA_FILE must name a regular source-stamp file"
[[ "$POLL_SECONDS" =~ ^[0-9]{1,4}$ && "$WATCHDOG_SECONDS" =~ ^[0-9]{1,6}$ \
    && "$KILL_GRACE_SECONDS" =~ ^[0-9]{1,3}$ ]] \
    || fail "poll, watchdog, and kill-grace settings must be decimal integers"
POLL_SECONDS=$((10#$POLL_SECONDS))
WATCHDOG_SECONDS=$((10#$WATCHDOG_SECONDS))
KILL_GRACE_SECONDS=$((10#$KILL_GRACE_SECONDS))
(( POLL_SECONDS >= 1 && POLL_SECONDS <= 60 \
    && WATCHDOG_SECONDS > POLL_SECONDS && WATCHDOG_SECONDS <= 86400 \
    && KILL_GRACE_SECONDS >= 1 && KILL_GRACE_SECONDS <= 60 )) \
    || fail "poll must be 1...60s, watchdog > poll and <=86400s, kill grace 1...60s"

HARNESS_SHA="$(tr -d '[:space:]' < "$HARNESS_SHA_FILE")"
[[ "$HARNESS_SHA" =~ ^[0-9a-f]{40}$ ]] \
    || fail "source stamp must be one clean lowercase 40-character git SHA"
MANIFEST_SHA="$(shasum -a 256 "$MANIFEST" | awk '{print $1}')"
BINARY_SHA="$(shasum -a 256 "$BIN" | awk '{print $1}')"
RUNNER_SHA="$(shasum -a 256 "$RUNNER_PATH" | awk '{print $1}')"

# Validate the complete experiment before reserving output or launching a model. JSON integers are
# bounded to IEEE-754's exact range because jq is the manifest parser and the Swift CLI receives
# their decimal representation without any rounding latitude.
jq -e '
  def integer: type == "number" and floor == . and . >= 0 and . <= 9007199254740991;
  def ident: type == "string" and test("^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$");
  def sha256: type == "string" and test("^[0-9a-f]{64}$");
  def identity: type == "string" and test("^([0-9a-f]{16}|[0-9a-f]{64})$");
  .schemaVersion == 1 and
  (.harnessGitSHA | type == "string" and test("^[0-9a-f]{40}$")) and
  (.harnessBinarySHA256 | sha256) and
  (.matrixID | ident) and (.workloadNonce | ident) and
  (.modelPath | type == "string" and length > 0) and
  (.modelConfigHash | type == "string" and test("^[0-9a-f]{16}$")) and
  (.modelCheckpointManifestHash | identity) and
  (.modelTokenizerSHA256 | sha256) and
  (.promptRepeat | integer and . >= 1 and . <= 4096) and
  (.maxTokens | integer and . >= 1 and . <= 1048576) and
  (.memoryLimitBytes | integer and . >= 1) and
  (.cacheLimitBytes | integer and . >= 1) and
  (.wiredLimitBytes | integer and . >= 1) and
  .cacheLimitBytes <= .memoryLimitBytes and
  .memoryLimitBytes <= .wiredLimitBytes and
  (.cells | type == "array" and length >= 1 and length <= 32) and
  ([.cells[].id] | length == (unique | length)) and
  (all(.cells[];
    (.id | ident) and
    (.kvQuant | ident) and
    ((.attention? // "") as $a |
      ($a == "" or $a == "materialize" or
       $a == "split-affine-quantized-mm" or
       $a == "split-kvarn-quantized-mm")) and
    (if .kvQuant == "fp16" then
       (.attention? == null and .checkpointContentSHA256? == null)
     else ((.attention? // "") != "" and
       (.checkpointContentSHA256 | sha256)) end) and
    (if (.kvQuant | startswith("kvtuner-")) then
       (.kvtunerSchedule | type == "string" and length > 0) and
       (.kvtunerBundleSHA256 | sha256) and
       (.kvtunerScheduleArtifactSHA256 | sha256)
     else (.kvtunerSchedule? == null and .kvtunerBundleSHA256? == null and
       .kvtunerScheduleArtifactSHA256? == null) end))) and
  (.blocks | type == "array" and length >= 3 and length <= 1000)
' "$MANIFEST" >/dev/null || fail "manifest schema or bounded field validation failed"

EXPECTED_HARNESS_SHA="$(jq -r '.harnessGitSHA' "$MANIFEST")"
EXPECTED_BINARY_SHA="$(jq -r '.harnessBinarySHA256' "$MANIFEST")"
[[ "$HARNESS_SHA" == "$EXPECTED_HARNESS_SHA" ]] \
    || fail "source stamp does not match the frozen manifest"
[[ "$BINARY_SHA" == "$EXPECTED_BINARY_SHA" ]] \
    || fail "harness binary digest does not match the frozen manifest"

CELL_IDS="$(jq -c '[.cells[].id] | sort' "$MANIFEST")"
if ! jq -e --argjson ids "$CELL_IDS" '
    all(.blocks[]; type == "array" and length == ($ids | length) and
      (sort == $ids))
  ' "$MANIFEST" >/dev/null; then
    fail "every matrix block must be an exact cell permutation"
fi
if ! jq -e --argjson ids "$CELL_IDS" --argjson count "$(jq '.cells | length' "$MANIFEST")" '
    . as $root |
    all($ids[];
      . as $id |
      ([range(0; $count) as $position |
        ([$root.blocks[] | select(.[$position] == $id)] | length)]) as $counts |
      (($counts | max) - ($counts | min) <= 1))
  ' "$MANIFEST" >/dev/null; then
    fail "matrix blocks must be position-balanced for every cell"
fi

CELL_COUNT="$(jq -r '.cells | length' "$MANIFEST")"
BLOCK_COUNT="$(jq -r '.blocks | length' "$MANIFEST")"
MATRIX_ID="$(jq -r '.matrixID' "$MANIFEST")"
WORKLOAD_NONCE="$(jq -r '.workloadNonce' "$MANIFEST")"
MODEL_PATH="$(jq -r '.modelPath' "$MANIFEST")"
MODEL_CONFIG_HASH="$(jq -r '.modelConfigHash' "$MANIFEST")"
MODEL_MANIFEST_HASH="$(jq -r '.modelCheckpointManifestHash' "$MANIFEST")"
MODEL_TOKENIZER_SHA="$(jq -r '.modelTokenizerSHA256' "$MANIFEST")"
PROMPT_REPEAT="$(jq -r '.promptRepeat' "$MANIFEST")"
MAX_TOKENS="$(jq -r '.maxTokens' "$MANIFEST")"
MEMORY_LIMIT="$(jq -r '.memoryLimitBytes' "$MANIFEST")"
CACHE_LIMIT="$(jq -r '.cacheLimitBytes' "$MANIFEST")"
WIRED_LIMIT="$(jq -r '.wiredLimitBytes' "$MANIFEST")"
TOTAL_ROWS=$((BLOCK_COUNT * CELL_COUNT))

# Authenticate immutable schedule inputs before output reservation. The same digest is checked
# again immediately before every KVTuner launch to detect a mid-matrix replacement.
for ((cell_index = 0; cell_index < CELL_COUNT; cell_index++)); do
    kv_quant="$(jq -r ".cells[$cell_index].kvQuant" "$MANIFEST")"
    if [[ "$kv_quant" == kvtuner-* ]]; then
        schedule="$(jq -r ".cells[$cell_index].kvtunerSchedule" "$MANIFEST")"
        schedule_sha="$(jq -r ".cells[$cell_index].kvtunerBundleSHA256" "$MANIFEST")"
        [[ -f "$schedule" && ! -L "$schedule" ]] \
            || fail "KVTuner schedule must be a regular, non-symbolic-link file"
        [[ "$(shasum -a 256 "$schedule" | awk '{print $1}')" == "$schedule_sha" ]] \
            || fail "KVTuner schedule digest does not match the frozen manifest"
    fi
done

[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] \
    || fail "qualification requires a fresh output directory"
OUTPUT_PARENT="$(dirname "$OUTPUT")"
mkdir -p "$OUTPUT_PARENT"
LOCK="${OUTPUT}.lock"
[[ ! -e "$LOCK" && ! -L "$LOCK" ]] \
    || fail "qualification requires a fresh output boundary; runner lock already exists"
if ! mkdir "$LOCK" 2>/dev/null; then
    fail "qualification requires a fresh output boundary; runner lock is held"
fi
lock_owned=true
active_child=""
caffeinate_pid=""

terminate_active_child() {
    if active_child_running; then
        kill -TERM "$active_child" 2>/dev/null || true
        remaining="$KILL_GRACE_SECONDS"
        while active_child_running && (( remaining > 0 )); do
            sleep 1
            remaining=$((remaining - 1))
        done
        if active_child_running; then
            kill -KILL "$active_child" 2>/dev/null || true
        fi
    fi
    if [[ -n "$active_child" ]]; then
        wait "$active_child" 2>/dev/null || true
        active_child=""
    fi
}

active_child_running() {
    local owned_pid
    [[ -n "$active_child" ]] || return 1
    while IFS= read -r owned_pid; do
        [[ "$owned_pid" == "$active_child" ]] && return 0
    done < <(jobs -pr)
    return 1
}

cleanup() {
    cleanup_rc=$?
    terminate_active_child
    if [[ -n "$caffeinate_pid" ]] && kill -0 "$caffeinate_pid" 2>/dev/null; then
        kill "$caffeinate_pid" 2>/dev/null || true
        wait "$caffeinate_pid" 2>/dev/null || true
    fi
    if (( cleanup_rc != 0 )) && [[ -f "${STATUS:-}" ]] \
        && [[ "$(tr -d '[:space:]' < "$STATUS")" == "RUNNING" ]]; then
        printf 'ABORTED\n' > "$STATUS"
    fi
    if [[ "${lock_owned:-false}" == "true" ]]; then
        rmdir "$LOCK" 2>/dev/null || true
    fi
    return "$cleanup_rc"
}
trap cleanup EXIT
trap '[[ -f "${STATUS:-}" ]] && printf "INTERRUPTED\n" > "$STATUS"; exit 130' INT
trap '[[ -f "${STATUS:-}" ]] && printf "INTERRUPTED\n" > "$STATUS"; exit 143' TERM

if ! mkdir "$OUTPUT" 2>/dev/null; then
    fail "qualification requires a fresh output directory"
fi
mkdir "$OUTPUT/runs" "$OUTPUT/blocks"
RUNNER_LOG="$OUTPUT/runner.log"
STATUS="$OUTPUT/runner.status"
PROGRESS="$OUTPUT/runner.progress.json"
MANIFEST_COPY="$OUTPUT/runner.manifest.json"
RECEIPT_SET="$OUTPUT/runner.receipts.sha256"
printf '%s\n' "$$" > "$OUTPUT/runner.pid"
date +%s > "$OUTPUT/runner.started-epoch"
printf 'RUNNING\n' > "$STATUS"
: > "$RUNNER_LOG"
: > "$RECEIPT_SET"
cp "$MANIFEST" "$MANIFEST_COPY"
chmod 0444 "$MANIFEST_COPY"
if [[ "$(shasum -a 256 "$MANIFEST_COPY" | awk '{print $1}')" != "$MANIFEST_SHA" ]]; then
    printf 'INPUT_CHANGED\n' > "$STATUS"
    fail "runner manifest changed while reserving the qualification boundary"
fi
MANIFEST="$MANIFEST_COPY"

if [[ "${DISABLE_CAFFEINATE:-false}" != "true" ]]; then
    command -v caffeinate >/dev/null 2>&1 \
        || fail "caffeinate is required unless DISABLE_CAFFEINATE=true"
    caffeinate -dimsu -w "$$" >/dev/null 2>&1 &
    caffeinate_pid=$!
fi

file_mtime() {
    local path="$1"
    if stat -f %m "$path" >/dev/null 2>&1; then
        stat -f %m "$path"
    else
        stat -c %Y "$path"
    fi
}

write_progress() {
    local state="$1" completed="$2" block="$3" position="$4" cell="$5"
    local child="${6:-}" max_rss="${7:-0}" now tmp
    now="$(date +%s)"
    tmp="${PROGRESS}.tmp.$$"
    jq -cn \
        --arg state "$state" --arg matrixID "$MATRIX_ID" \
        --arg cellID "$cell" --arg harnessGitSHA "$HARNESS_SHA" \
        --arg manifestSHA256 "$MANIFEST_SHA" --arg childPID "$child" \
        --argjson completedRows "$completed" --argjson totalRows "$TOTAL_ROWS" \
        --argjson blockIndex "$block" --argjson runPosition "$position" \
        --argjson heartbeatEpoch "$now" --argjson maxProcessRSSBytes "$max_rss" \
        '{schemaVersion:1,state:$state,matrixID:$matrixID,cellID:$cellID,
          completedRows:$completedRows,totalRows:$totalRows,blockIndex:$blockIndex,
          runPosition:$runPosition,heartbeatEpoch:$heartbeatEpoch,
          childPID:(if $childPID == "" then null else ($childPID | tonumber) end),
          maxProcessRSSBytes:$maxProcessRSSBytes,harnessGitSHA:$harnessGitSHA,
          runnerManifestSHA256:$manifestSHA256}' > "$tmp"
    mv "$tmp" "$PROGRESS"
}

write_progress "starting" 0 0 0 "" "" 0
completed_rows=0

for ((block_index = 0; block_index < BLOCK_COUNT; block_index++)); do
    block_environment=""
    block_receipts="$OUTPUT/blocks/block-$(printf '%03d' "$block_index").receipts.jsonl"
    : > "$block_receipts"
    for ((position = 0; position < CELL_COUNT; position++)); do
        if [[ "$(shasum -a 256 "$MANIFEST" | awk '{print $1}')" != "$MANIFEST_SHA" \
            || "$(shasum -a 256 "$RUNNER_PATH" | awk '{print $1}')" != "$RUNNER_SHA" \
            || "$(shasum -a 256 "$BIN" | awk '{print $1}')" != "$BINARY_SHA" ]]; then
            printf 'INPUT_CHANGED\n' > "$STATUS"
            fail "runner manifest, runner script, or harness binary changed during qualification"
        fi
        cell_id="$(jq -r ".blocks[$block_index][$position]" "$MANIFEST")"
        cell_json="$(jq -c --arg id "$cell_id" '.cells[] | select(.id == $id)' "$MANIFEST")"
        kv_quant="$(jq -r '.kvQuant' <<< "$cell_json")"
        attention="$(jq -r '.attention? // empty' <<< "$cell_json")"
        checkpoint_sha="$(jq -r '.checkpointContentSHA256? // empty' <<< "$cell_json")"
        schedule="$(jq -r '.kvtunerSchedule? // empty' <<< "$cell_json")"
        schedule_bundle_sha="$(jq -r '.kvtunerBundleSHA256? // empty' <<< "$cell_json")"
        schedule_artifact_sha="$(jq -r '.kvtunerScheduleArtifactSHA256? // empty' <<< "$cell_json")"
        run_dir="$OUTPUT/runs/block-$(printf '%03d' "$block_index")/position-$(printf '%03d' "$position")-$cell_id"
        mkdir -p "$run_dir"
        evidence="$run_dir/bench.jsonl"
        run_log="$run_dir/bench.log"
        receipt="$run_dir/runner-receipt.json"
        : > "$run_log"

        args=(
            bench
            --model "$MODEL_PATH"
            --label "$cell_id"
            --matrix-id "$MATRIX_ID"
            # `label` names the experiment route (for example affine materialized versus direct),
            # while the harness's authenticated cell identity must remain the executed KV tier.
            --cell-id "$kv_quant"
            --workload-nonce "$WORKLOAD_NONCE"
            --kv-quant "$kv_quant"
            --prompt-repeat "$PROMPT_REPEAT"
            --max-tokens "$MAX_TOKENS"
            --runs 1
            --qualification-evidence true
            --runner-manifest-sha256 "$MANIFEST_SHA"
            --matrix-block-index "$block_index"
            --matrix-run-position "$position"
            --matrix-cell-count "$CELL_COUNT"
            --memory-limit-bytes "$MEMORY_LIMIT"
            --cache-limit-bytes "$CACHE_LIMIT"
            --wired-limit-bytes "$WIRED_LIMIT"
            --model-tokenizer-sha256 "$MODEL_TOKENIZER_SHA"
            --evidence "$evidence"
        )
        if [[ -n "$attention" ]]; then
            args+=(--kv-attention "$attention" \
                --checkpoint-content-sha256 "$checkpoint_sha")
        fi
        if [[ -n "$schedule" ]]; then
            if [[ "$(shasum -a 256 "$schedule" | awk '{print $1}')" != "$schedule_bundle_sha" ]]; then
                printf 'INPUT_CHANGED\n' > "$STATUS"
                fail "KVTuner qualification bundle changed after manifest authentication"
            fi
            args+=(--kvtuner-schedule "$schedule")
        fi

        printf '[%s] launch block=%s position=%s cell=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$block_index" "$position" "$cell_id" \
            >> "$RUNNER_LOG"
        "$BIN" "${args[@]}" > "$run_log" 2>&1 &
        active_child=$!
        printf '%s\n' "$active_child" > "$run_dir/harness.pid"
        max_rss_bytes=0
        write_progress "running" "$completed_rows" "$block_index" "$position" \
            "$cell_id" "$active_child" "$max_rss_bytes"
        last_log_mtime="$(file_mtime "$run_log")"
        while active_child_running; do
            rss_kb="$(ps -o rss= -p "$active_child" 2>/dev/null | tr -d '[:space:]' || true)"
            if [[ "$rss_kb" =~ ^[0-9]+$ ]]; then
                rss_bytes=$((rss_kb * 1024))
                (( rss_bytes > max_rss_bytes )) && max_rss_bytes=$rss_bytes
            fi
            now="$(date +%s)"
            current_mtime="$(file_mtime "$run_log")"
            (( current_mtime > last_log_mtime )) && last_log_mtime=$current_mtime
            if (( now - last_log_mtime > WATCHDOG_SECONDS )); then
                jq -cn --arg cellID "$cell_id" --argjson blockIndex "$block_index" \
                    --argjson runPosition "$position" --argjson childPID "$active_child" \
                    --argjson detectedEpoch "$now" --argjson lastLogMtime "$last_log_mtime" \
                    '{schemaVersion:1,reason:"log-stalled",cellID:$cellID,
                      blockIndex:$blockIndex,runPosition:$runPosition,childPID:$childPID,
                      detectedEpoch:$detectedEpoch,lastLogMtime:$lastLogMtime}' \
                    > "$OUTPUT/runner.watchdog.json"
                printf 'WATCHDOG\n' > "$STATUS"
                write_progress "watchdog" "$completed_rows" "$block_index" "$position" \
                    "$cell_id" "$active_child" "$max_rss_bytes"
                terminate_active_child
                fail "watchdog detected a stalled harness process"
            fi
            write_progress "running" "$completed_rows" "$block_index" "$position" \
                "$cell_id" "$active_child" "$max_rss_bytes"
            sleep "$POLL_SECONDS"
        done
        set +e
        wait "$active_child"
        child_rc=$?
        set -e
        active_child=""
        cat "$run_log" >> "$RUNNER_LOG"
        if (( child_rc != 0 )); then
            printf 'FAILED\n' > "$STATUS"
            write_progress "failed" "$completed_rows" "$block_index" "$position" \
                "$cell_id" "" "$max_rss_bytes"
            fail "harness failed for block=$block_index position=$position cell=$cell_id"
        fi

        if [[ ! -f "$evidence" || -L "$evidence" ]] \
            || [[ "$(wc -l < "$evidence" 2>/dev/null | tr -d '[:space:]')" != "1" ]]; then
            printf 'INVALID_EVIDENCE\n' > "$STATUS"
            fail "harness evidence must be one regular file containing exactly one row"
        fi
        expected_observed=""
        case "$attention" in
            materialize) expected_observed="materialized-kv" ;;
            split-affine-quantized-mm) expected_observed="split-quantized-mm" ;;
            split-kvarn-quantized-mm) expected_observed="split-kvarn-quantized-mm" ;;
        esac
        if ! jq -e \
            --arg harness "$HARNESS_SHA" --arg matrix "$MATRIX_ID" \
            --arg cell "$cell_id" --arg nonce "$WORKLOAD_NONCE" \
            --arg tier "$kv_quant" --arg manifest "$MANIFEST_SHA" \
            --arg modelPath "$MODEL_PATH" --arg modelConfigHash "$MODEL_CONFIG_HASH" \
            --arg modelManifestHash "$MODEL_MANIFEST_HASH" \
            --arg modelTokenizerSHA "$MODEL_TOKENIZER_SHA" \
            --arg attention "$attention" --arg expectedObserved "$expected_observed" \
            --arg checkpointSHA "$checkpoint_sha" \
            --arg scheduleBundleSHA "$schedule_bundle_sha" \
            --arg scheduleArtifactSHA "$schedule_artifact_sha" \
            --argjson block "$block_index" --argjson position "$position" \
            --argjson count "$CELL_COUNT" --argjson memory "$MEMORY_LIMIT" \
            --argjson cache "$CACHE_LIMIT" --argjson wired "$WIRED_LIMIT" \
            --argjson promptRepeat "$PROMPT_REPEAT" --argjson maxTokens "$MAX_TOKENS" '
              .subcommand == "bench" and
              .provenance.harnessGitSHA == $harness and
              .provenance.modelPath == $modelPath and
              .provenance.modelConfigHash == $modelConfigHash and
              .provenance.modelCheckpointManifestHash == $modelManifestHash and
              .payload.label == $cell and .payload.matrixID == $matrix and
              .payload.cellID == $tier and .payload.workloadNonce == $nonce and
              .payload.kvQuantTier == $tier and .payload.promptRepeat == $promptRepeat and
              .payload.maxTokens == $maxTokens and .payload.measuredRuns == 1 and
              (.payload.memoryRuns | type == "array" and length == 1) and
              (.payload.memoryRuns[0].samples | type == "array" and length == 2) and
              (.payload.memoryRuns[0].samples[0].timestamp <=
                .payload.memoryRuns[0].samples[1].timestamp) and
              (all(.payload.memoryRuns[0].samples[];
                .physicalFootprintBytes > 0 and .mlxActiveBytes >= 0 and
                .mlxCacheBytes >= 0 and .mlxPeakBytes >= 0)) and
              .payload.memoryRuns[0].summary.startFootprintBytes ==
                .payload.memoryRuns[0].samples[0].physicalFootprintBytes and
              .payload.memoryRuns[0].summary.endFootprintBytes ==
                .payload.memoryRuns[0].samples[1].physicalFootprintBytes and
              .payload.memoryRuns[0].summary.maxSampledFootprintBytes ==
                ([.payload.memoryRuns[0].samples[].physicalFootprintBytes] | max) and
              .payload.memoryRuns[0].summary.maxMLXActiveBytes ==
                ([.payload.memoryRuns[0].samples[].mlxActiveBytes] | max) and
              .payload.memoryRuns[0].summary.maxMLXCacheBytes ==
                ([.payload.memoryRuns[0].samples[].mlxCacheBytes] | max) and
              .payload.memoryRuns[0].summary.maxMLXPeakBytes ==
                ([.payload.memoryRuns[0].samples[].mlxPeakBytes] | max) and
              .payload.maxSampledPhysicalFootprintBytes ==
                .payload.memoryRuns[0].summary.maxSampledFootprintBytes and
              .payload.maxMLXActiveBytes ==
                .payload.memoryRuns[0].summary.maxMLXActiveBytes and
              .payload.maxMLXCacheBytes ==
                .payload.memoryRuns[0].summary.maxMLXCacheBytes and
              .payload.maxMLXPeakBytes ==
                .payload.memoryRuns[0].summary.maxMLXPeakBytes and
              .payload.qualification.schemaVersion == 2 and
              .payload.qualification.context.runnerManifestSHA256 == $manifest and
              .payload.qualification.context.matrixBlockIndex == $block and
              .payload.qualification.context.matrixRunPosition == $position and
              .payload.qualification.context.matrixCellCount == $count and
              .payload.qualification.context.memoryLimitBytes == $memory and
              .payload.qualification.context.cacheLimitBytes == $cache and
              .payload.qualification.context.wiredLimitBytes == $wired and
              .payload.qualification.context.tokenizerSHA256 == $modelTokenizerSHA and
              .payload.qualification.context.cacheResetPolicy == "in-place-before-every-generation" and
              .payload.qualification.context.modelResidencyPolicy == "load-once-per-process" and
              .payload.qualification.context.processIsolationPolicy == "fresh-process-per-matrix-position" and
              (.payload.qualification.runs | type == "array" and length == 1) and
              .payload.qualification.runs[0].before.monotonicTimestampSeconds <
                .payload.qualification.runs[0].after.monotonicTimestampSeconds and
              .payload.qualification.runs[0].before.residentSizeBytes > 0 and
              .payload.qualification.runs[0].after.residentSizeBytes > 0 and
              .payload.qualification.runs[0].before.physicalFootprintBytes > 0 and
              .payload.qualification.runs[0].after.physicalFootprintBytes > 0 and
              .payload.qualification.runs[0].before.lowPowerModeEnabled ==
                .payload.qualification.runs[0].after.lowPowerModeEnabled and
              .payload.qualification.runs[0].before.powerSource ==
                .payload.qualification.runs[0].after.powerSource and
              .payload.qualification.runs[0].before.powerSource != "unavailable" and
              .payload.qualification.runs[0].before.thermalState ==
                .payload.qualification.runs[0].after.thermalState and
              .payload.qualification.runs[0].before.thermalState != "unknown" and
              (.payload.workloadPromptSHA256 | type == "array" and length == 2 and
                all(.[]; type == "string" and test("^[0-9a-f]{64}$"))) and
              (.payload.promptTokenCountsByRun | type == "array" and length == 1 and .[0] > 0) and
              (.payload.prefillDurationSecondsByRun | type == "array" and length == 1 and .[0] > 0) and
              (.payload.prefillTokSByRun | type == "array" and length == 1 and .[0] > 0) and
              (.payload.decodeTokSByRun | type == "array" and length == 1 and .[0] > 0) and
              (.payload.ttftMsByRun | type == "array" and length == 1 and .[0] > 0) and
              (.payload.generatedTokenCountsByRun | type == "array" and length == 1 and .[0] > 0) and
              .payload.memoryCacheLimitBytes == $cache and
              (if $attention == "" then .payload.compressedKVAttention == null
               else .payload.compressedKVAttention.request == $attention and
                 .payload.compressedKVAttention.observedOperation == $expectedObserved and
                 .payload.compressedKVAttention.admission.family ==
                   (if .payload.compressedKVAttention.admission.modelType == "qwen3"
                    then "qwen3" else "llama" end) and
                 (.payload.compressedKVAttention.admission.architecture == "Qwen3ForCausalLM" or
                   .payload.compressedKVAttention.admission.architecture == "LlamaForCausalLM") and
                 .payload.compressedKVAttention.admission.modelConfigHash == $modelConfigHash and
                 (.payload.compressedKVAttention.admission.modelConfigSHA256 |
                   type == "string" and test("^[0-9a-f]{64}$")) and
                 .payload.compressedKVAttention.admission.checkpointManifestHash == $modelManifestHash and
                 .payload.compressedKVAttention.admission.checkpointContentSHA256 == $checkpointSHA and
                 .payload.compressedKVAttention.admission.tokenizerSHA256 == $modelTokenizerSHA and
                 .payload.compressedKVAttention.admission.layerCount > 0 and
                 .payload.compressedKVAttention.admission.queryHeadCount > 0 and
                 .payload.compressedKVAttention.admission.kvHeadCount > 0 and
                 .payload.compressedKVAttention.admission.headDimension > 0 and
                 .payload.compressedKVAttention.admission.maxPositionEmbeddings > 0 end) and
              (if $scheduleBundleSHA == "" then .payload.kvtunerSchedule == null
               else .payload.kvtunerSchedule.schemaVersion == 4 and
                 .payload.kvtunerSchedule.scheduleSchemaVersion == 4 and
                 .payload.kvtunerSchedule.qualificationBundleSHA256 == $scheduleBundleSHA and
                 .payload.kvtunerSchedule.artifactSHA256 == $scheduleArtifactSHA and
                 .payload.kvtunerSchedule.matrixID == $matrix and
                 .payload.kvtunerSchedule.cellID == $tier and
                 .payload.kvtunerSchedule.modelConfigHash == $modelConfigHash and
                 .payload.kvtunerSchedule.modelConfigSHA256 ==
                   .payload.compressedKVAttention.admission.modelConfigSHA256 and
                 .payload.kvtunerSchedule.checkpointManifestHash == $modelManifestHash and
                 .payload.kvtunerSchedule.checkpointContentSHA256 == $checkpointSHA and
                 .payload.kvtunerSchedule.tokenizerSHA256 == $modelTokenizerSHA and
                 .payload.kvtunerSchedule.groupSize > 0 and
                 (.payload.kvtunerSchedule.layers | type == "array" and length ==
                   .payload.compressedKVAttention.admission.layerCount) end)
            ' "$evidence" >/dev/null; then
            printf 'INVALID_EVIDENCE\n' > "$STATUS"
            fail "harness row failed exact identity or qualification authentication"
        fi
        typed_validation_log="$run_dir/typed-validation.log"
        if ! "$BIN" validate-bench-qualification --evidence "$evidence" \
            > "$typed_validation_log" 2>&1; then
            cat "$typed_validation_log" >> "$RUNNER_LOG"
            printf 'INVALID_EVIDENCE\n' > "$STATUS"
            fail "harness row failed typed qualification validation"
        fi

        internal_rss="$(jq -r '[.payload.qualification.runs[0].before.residentSizeBytes,
            .payload.qualification.runs[0].after.residentSizeBytes] | max' "$evidence")"
        (( internal_rss > max_rss_bytes )) && max_rss_bytes=$internal_rss
        environment_key="$(jq -c '[
            .payload.qualification.runs[0].before.powerSource,
            .payload.qualification.runs[0].before.lowPowerModeEnabled,
            .payload.qualification.runs[0].before.thermalState,
            .payload.qualification.runs[0].after.powerSource,
            .payload.qualification.runs[0].after.lowPowerModeEnabled,
            .payload.qualification.runs[0].after.thermalState]' "$evidence")"
        if [[ -z "$block_environment" ]]; then
            block_environment="$environment_key"
        elif [[ "$environment_key" != "$block_environment" ]]; then
            invalid="$OUTPUT/blocks/block-$(printf '%03d' "$block_index").invalid.json"
            jq -cn --arg reason "block environment changed" \
                --arg baseline "$block_environment" --arg observed "$environment_key" \
                --arg evidenceSHA256 "$(shasum -a 256 "$evidence" | awk '{print $1}')" \
                --arg cellID "$cell_id" --argjson blockIndex "$block_index" \
                --argjson runPosition "$position" \
                '{schemaVersion:1,reason:$reason,blockIndex:$blockIndex,
                  runPosition:$runPosition,cellID:$cellID,
                  baselineEnvironment:($baseline | fromjson),
                  observedEnvironment:($observed | fromjson),
                  evidenceSHA256:$evidenceSHA256}' > "$invalid"
            printf 'INVALID_BLOCK_ENVIRONMENT\n' > "$STATUS"
            write_progress "invalid-block-environment" "$completed_rows" \
                "$block_index" "$position" "$cell_id" "" "$max_rss_bytes"
            fail "block environment changed across matrix rows"
        fi

        evidence_sha="$(shasum -a 256 "$evidence" | awk '{print $1}')"
        log_sha="$(shasum -a 256 "$run_log" | awk '{print $1}')"
        jq -cn \
            --arg matrixID "$MATRIX_ID" --arg cellID "$cell_id" \
            --arg harnessGitSHA "$HARNESS_SHA" --arg harnessBinarySHA256 "$BINARY_SHA" \
            --arg runnerScriptSHA256 "$RUNNER_SHA" \
            --arg runnerManifestSHA256 "$MANIFEST_SHA" --arg evidenceSHA256 "$evidence_sha" \
            --arg logSHA256 "$log_sha" --argjson blockIndex "$block_index" \
            --argjson runPosition "$position" --argjson maxProcessRSSBytes "$max_rss_bytes" \
            '{schemaVersion:1,matrixID:$matrixID,cellID:$cellID,blockIndex:$blockIndex,
              runPosition:$runPosition,harnessGitSHA:$harnessGitSHA,
              harnessBinarySHA256:$harnessBinarySHA256,
              runnerScriptSHA256:$runnerScriptSHA256,
              runnerManifestSHA256:$runnerManifestSHA256,
              evidenceSHA256:$evidenceSHA256,logSHA256:$logSHA256,
              maxProcessRSSBytes:$maxProcessRSSBytes}' > "$receipt"
        jq -c . "$receipt" >> "$block_receipts"
        receipt_sha="$(shasum -a 256 "$receipt" | awk '{print $1}')"
        printf '%s  %s\n' "$receipt_sha" \
            "${receipt#"$OUTPUT"/}" >> "$RECEIPT_SET"
        completed_rows=$((completed_rows + 1))
        write_progress "row-complete" "$completed_rows" "$block_index" "$position" \
            "$cell_id" "" "$max_rss_bytes"
    done
    jq -s --arg matrixID "$MATRIX_ID" --arg environment "$block_environment" \
        --argjson blockIndex "$block_index" \
        '{schemaVersion:1,matrixID:$matrixID,blockIndex:$blockIndex,
          environment:($environment | fromjson),receipts:.}' \
        "$block_receipts" > "$OUTPUT/blocks/block-$(printf '%03d' "$block_index").complete.json"
done

if [[ "$(shasum -a 256 "$MANIFEST" | awk '{print $1}')" != "$MANIFEST_SHA" \
    || "$(shasum -a 256 "$RUNNER_PATH" | awk '{print $1}')" != "$RUNNER_SHA" \
    || "$(shasum -a 256 "$BIN" | awk '{print $1}')" != "$BINARY_SHA" ]]; then
    printf 'INPUT_CHANGED\n' > "$STATUS"
    fail "qualification inputs changed before finalization"
fi
RECEIPT_SET_SHA="$(shasum -a 256 "$RECEIPT_SET" | awk '{print $1}')"
completion_tmp="$OUTPUT/runner.completion.json.tmp.$$"
jq -cn --arg matrixID "$MATRIX_ID" --arg harnessGitSHA "$HARNESS_SHA" \
    --arg runnerManifestSHA256 "$MANIFEST_SHA" \
    --arg harnessBinarySHA256 "$BINARY_SHA" --arg runnerScriptSHA256 "$RUNNER_SHA" \
    --arg receiptSetSHA256 "$RECEIPT_SET_SHA" --argjson blockCount "$BLOCK_COUNT" \
    --argjson cellCount "$CELL_COUNT" --argjson completedRows "$completed_rows" \
    '{schemaVersion:1,status:"COMPLETE",matrixID:$matrixID,blockCount:$blockCount,
      cellCount:$cellCount,completedRows:$completedRows,harnessGitSHA:$harnessGitSHA,
      harnessBinarySHA256:$harnessBinarySHA256,
      runnerScriptSHA256:$runnerScriptSHA256,
      runnerManifestSHA256:$runnerManifestSHA256,
      receiptSetSHA256:$receiptSetSHA256}' > "$completion_tmp"
mv "$completion_tmp" "$OUTPUT/runner.completion.json"
write_progress "complete" "$completed_rows" "$((BLOCK_COUNT - 1))" \
    "$((CELL_COUNT - 1))" "" "" 0
printf 'COMPLETE\n' > "$STATUS"
printf '[%s] complete rows=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$completed_rows" >> "$RUNNER_LOG"
