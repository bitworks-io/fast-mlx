#!/bin/bash
# Run one loaded-model bench process per counterbalanced matrix position and retain authenticated
# environment/RSS receipts. Every invocation requires a new output boundary; interrupted evidence
# is preserved and is never silently resumed or overwritten.
# Usage: run_loaded_bench_qualification.sh <manifest.json> <fresh-output-root>
set -euo pipefail

# Git repository discovery is part of the evidence identity, so caller-controlled Git
# environment cannot be allowed to redirect either the preflight or the harness child.
clear_caller_git_environment() {
    local variable
    for variable in "${!GIT_@}"; do
        unset "$variable"
    done
}
clear_caller_git_environment
unset HARNESS_GIT_SHA

MANIFEST="${1:?usage: run_loaded_bench_qualification.sh <manifest.json> <fresh-output-root>}"
OUTPUT="${2:?usage: run_loaded_bench_qualification.sh <manifest.json> <fresh-output-root>}"
CALLER_DIR="$(pwd -P)"
absolute_from_caller() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$CALLER_DIR" "$1" ;;
    esac
}
case "$OUTPUT" in
    /*) ;;
    *) OUTPUT="$CALLER_DIR/$OUTPUT" ;;
esac
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

for command_name in jq mktemp shasum ps stat; do
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
[[ "$(basename "$HARNESS_SHA_FILE")" == ".harness-sha" ]] \
    || fail "HARNESS_SHA_FILE must be named .harness-sha so the harness reads the authenticated stamp"
BIN="$(cd "$(dirname "$BIN")" && pwd -P)/$(basename "$BIN")"
HARNESS_SOURCE_DIR="$(cd "$(dirname "$HARNESS_SHA_FILE")" && pwd -P)"
HARNESS_SHA_FILE="$HARNESS_SOURCE_DIR/$(basename "$HARNESS_SHA_FILE")"
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
HARNESS_STAMP_SHA256="$(shasum -a 256 "$HARNESS_SHA_FILE" | awk '{print $1}')"

source_provenance_is_current() {
    local live_git_root live_git_sha live_git_status
    [[ -f "$HARNESS_SHA_FILE" && ! -L "$HARNESS_SHA_FILE" ]] || return 1
    [[ "$(shasum -a 256 "$HARNESS_SHA_FILE" | awk '{print $1}')" \
        == "$HARNESS_STAMP_SHA256" ]] || return 1
    [[ "$(tr -d '[:space:]' < "$HARNESS_SHA_FILE")" == "$HARNESS_SHA" ]] || return 1
    if command -v git >/dev/null 2>&1; then
        live_git_root="$(git -C "$HARNESS_SOURCE_DIR" rev-parse --show-toplevel \
            2>/dev/null || true)"
        if [[ -n "$live_git_root" ]]; then
            live_git_sha="$(git -C "$live_git_root" rev-parse HEAD 2>/dev/null)" \
                || return 1
            live_git_status="$(git -C "$live_git_root" status --porcelain \
                --untracked-files=normal -- spike experiments 2>/dev/null)" \
                || return 1
            [[ "$live_git_sha" =~ ^[0-9a-f]{40}$ \
                && -z "$live_git_status" && "$live_git_sha" == "$HARNESS_SHA" ]] \
                || return 1
        fi
    fi
}
source_provenance_is_current \
    || fail "source stamp and live Git provenance must be immutable, clean, and identical"
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
  .schemaVersion == 3 and
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
  .postWarmupThermalTarget == "nominal" and
  (.postWarmupThermalTimeoutSeconds | integer and . >= 1 and . <= 3600) and
  (.postWarmupThermalPollMilliseconds | integer and . >= 100 and . <= 60000) and
  .postWarmupThermalPollMilliseconds <=
    (.postWarmupThermalTimeoutSeconds * 1000) and
  (.postWarmupThermalStabilitySeconds | integer and . >= 1) and
  .postWarmupThermalStabilitySeconds <=
    .postWarmupThermalTimeoutSeconds and
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
MODEL_PATH="$(absolute_from_caller "$(jq -r '.modelPath' "$MANIFEST")")"
MODEL_CONFIG_HASH="$(jq -r '.modelConfigHash' "$MANIFEST")"
MODEL_MANIFEST_HASH="$(jq -r '.modelCheckpointManifestHash' "$MANIFEST")"
MODEL_TOKENIZER_SHA="$(jq -r '.modelTokenizerSHA256' "$MANIFEST")"
PROMPT_REPEAT="$(jq -r '.promptRepeat' "$MANIFEST")"
MAX_TOKENS="$(jq -r '.maxTokens' "$MANIFEST")"
MEMORY_LIMIT="$(jq -r '.memoryLimitBytes' "$MANIFEST")"
CACHE_LIMIT="$(jq -r '.cacheLimitBytes' "$MANIFEST")"
WIRED_LIMIT="$(jq -r '.wiredLimitBytes' "$MANIFEST")"
POST_WARMUP_THERMAL_TARGET="$(jq -r '.postWarmupThermalTarget' "$MANIFEST")"
POST_WARMUP_THERMAL_TIMEOUT_SECONDS="$(jq -r '.postWarmupThermalTimeoutSeconds' "$MANIFEST")"
POST_WARMUP_THERMAL_POLL_MILLISECONDS="$(jq -r '.postWarmupThermalPollMilliseconds' "$MANIFEST")"
POST_WARMUP_THERMAL_STABILITY_SECONDS="$(jq -r '.postWarmupThermalStabilitySeconds' "$MANIFEST")"
TOTAL_ROWS=$((BLOCK_COUNT * CELL_COUNT))

# Authenticate immutable schedule inputs before output reservation. The same digest is checked
# again immediately before every KVTuner launch to detect a mid-matrix replacement.
for ((cell_index = 0; cell_index < CELL_COUNT; cell_index++)); do
    kv_quant="$(jq -r ".cells[$cell_index].kvQuant" "$MANIFEST")"
    if [[ "$kv_quant" == kvtuner-* ]]; then
        schedule="$(absolute_from_caller \
            "$(jq -r ".cells[$cell_index].kvtunerSchedule" "$MANIFEST")")"
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

file_mtime() {
    local path="$1"
    if stat -f %m "$path" >/dev/null 2>&1; then
        stat -f %m "$path"
    else
        stat -c %Y "$path"
    fi
}

file_identity() {
    local path="$1"
    if stat -f '%d:%i' "$path" >/dev/null 2>&1; then
        stat -f '%d:%i' "$path"
    else
        stat -c '%d:%i' "$path"
    fi
}

original_regular_file() {
    local path="$1" expected_identity="$2"
    [[ -f "$path" && ! -L "$path" ]] \
        && [[ "$(file_identity "$path")" == "$expected_identity" ]]
}

original_directory() {
    local path="$1" expected_identity="$2"
    [[ -d "$path" && ! -L "$path" ]] \
        && [[ "$(file_identity "$path")" == "$expected_identity" ]]
}

# Status and progress are monitor-facing parent artifacts. Publish by rename so a child-created
# symlink can only be replaced, never followed to an unrelated file outside the fresh boundary.
write_status() {
    local value="$1" tmp
    if [[ -n "${OUTPUT_DIRECTORY_IDENTITY:-}" ]] \
        && ! original_directory "$OUTPUT" "$OUTPUT_DIRECTORY_IDENTITY"; then
        return 1
    fi
    tmp="$(mktemp "$OUTPUT/.runner-status.XXXXXX")"
    printf '%s\n' "$value" > "$tmp"
    if [[ -d "$STATUS" ]]; then
        chmod 0444 "$tmp"
        return 1
    fi
    mv -f "$tmp" "$STATUS"
}

cleanup() {
    cleanup_rc=$?
    terminate_active_child
    exec 8>&- 2>/dev/null || true
    exec 9<&- 2>/dev/null || true
    if [[ -n "$caffeinate_pid" ]] && kill -0 "$caffeinate_pid" 2>/dev/null; then
        kill "$caffeinate_pid" 2>/dev/null || true
        wait "$caffeinate_pid" 2>/dev/null || true
    fi
    if (( cleanup_rc != 0 )) && [[ -f "${STATUS:-}" ]] \
        && [[ "$(tr -d '[:space:]' < "$STATUS")" == "RUNNING" ]]; then
        if declare -F write_status >/dev/null 2>&1; then
            write_status "ABORTED" || true
        else
            printf 'ABORTED\n' > "$STATUS"
        fi
    fi
    if [[ "${lock_owned:-false}" == "true" ]]; then
        rmdir "$LOCK" 2>/dev/null || true
    fi
    return "$cleanup_rc"
}
trap cleanup EXIT
trap '[[ -f "${STATUS:-}" ]] && write_status "INTERRUPTED"; exit 130' INT
trap '[[ -f "${STATUS:-}" ]] && write_status "INTERRUPTED"; exit 143' TERM

if ! mkdir "$OUTPUT" 2>/dev/null; then
    fail "qualification requires a fresh output directory"
fi
OUTPUT_DIRECTORY_IDENTITY="$(file_identity "$OUTPUT")"
mkdir "$OUTPUT/runs" "$OUTPUT/blocks"
RUNS_DIRECTORY_IDENTITY="$(file_identity "$OUTPUT/runs")"
BLOCKS_DIRECTORY_IDENTITY="$(file_identity "$OUTPUT/blocks")"
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
RUNNER_LOG_IDENTITY="$(file_identity "$RUNNER_LOG")"
RECEIPT_SET_IDENTITY="$(file_identity "$RECEIPT_SET")"
cp "$MANIFEST" "$MANIFEST_COPY"
chmod 0444 "$MANIFEST_COPY"
if [[ "$(shasum -a 256 "$MANIFEST_COPY" | awk '{print $1}')" != "$MANIFEST_SHA" ]]; then
    write_status "INPUT_CHANGED"
    fail "runner manifest changed while reserving the qualification boundary"
fi
MANIFEST="$MANIFEST_COPY"

if [[ "${DISABLE_CAFFEINATE:-false}" != "true" ]]; then
    command -v caffeinate >/dev/null 2>&1 \
        || fail "caffeinate is required unless DISABLE_CAFFEINATE=true"
    caffeinate -dimsu -w "$$" >/dev/null 2>&1 &
    caffeinate_pid=$!
fi

# A failed child cannot contribute a promotable row, but its exact launch identity, log, optional
# partial evidence, and retained-environment diagnostic must remain independently auditable. The
# receipt intentionally stays outside runner.receipts.sha256 and the block receipt stream.
write_harness_failure_receipt() {
    local child_exit_code="$1" run_directory="$2" evidence_path="$3" log_path="$4"
    local failure_reason="${5:-harness-exit}"
    local failure_receipt="$run_directory/runner-failure.json"
    local failure_receipt_tmp
    local log_sha evidence_present=false evidence_sha="" diagnostic_json="null"
    local diagnostic_candidate="" diagnostic_match_count=0

    log_sha="$(shasum -a 256 "$log_path" | awk '{print $1}')"
    if [[ -f "$evidence_path" && ! -L "$evidence_path" ]]; then
        evidence_present=true
        evidence_sha="$(shasum -a 256 "$evidence_path" | awk '{print $1}')"
    fi
    diagnostic_match_count="$(awk '
        index($0, "# qualification retained environment: ") == 1 { count += 1 }
        END { print count + 0 }
      ' "$log_path")"
    if [[ "$diagnostic_match_count" == "1" ]]; then
        diagnostic_candidate="$(awk '
        index($0, "# qualification retained environment: ") == 1 {
            print substr($0, length("# qualification retained environment: ") + 1)
        }
      ' "$log_path")"
    fi
    if [[ -n "$diagnostic_candidate" ]] && jq -e '
        def snapshot:
          type == "object" and
          (.monotonicTimestampSeconds | type == "number") and
          (.residentSizeBytes | type == "number" and . > 0) and
          (.physicalFootprintBytes | type == "number" and . > 0) and
          (.lowPowerModeEnabled | type == "boolean") and
          (.powerSource == "ac-power" or .powerSource == "battery" or
            .powerSource == "unknown") and
          (.thermalState == "nominal" or .thermalState == "fair" or
            .thermalState == "serious" or .thermalState == "critical" or
            .thermalState == "unknown");
        .schemaVersion == 1 and (.before | snapshot) and (.after | snapshot) and
        .before.monotonicTimestampSeconds < .after.monotonicTimestampSeconds
      ' <<< "$diagnostic_candidate" >/dev/null 2>&1; then
        diagnostic_json="$diagnostic_candidate"
    fi

    failure_receipt_tmp="$(mktemp "$run_directory/.runner-failure.XXXXXX")"
    jq -cn \
        --arg reason "$failure_reason" --arg matrixID "$MATRIX_ID" \
        --arg workloadNonce "$WORKLOAD_NONCE" --arg cellID "$cell_id" \
        --arg kvQuantTier "$kv_quant" --arg requestedAttention "$attention" \
        --arg harnessGitSHA "$HARNESS_SHA" --arg harnessBinarySHA256 "$BINARY_SHA" \
        --arg runnerScriptSHA256 "$RUNNER_SHA" \
        --arg runnerManifestSHA256 "$MANIFEST_SHA" \
        --arg modelConfigHash "$MODEL_CONFIG_HASH" \
        --arg modelCheckpointManifestHash "$MODEL_MANIFEST_HASH" \
        --arg modelTokenizerSHA256 "$MODEL_TOKENIZER_SHA" \
        --arg postWarmupThermalTarget "$POST_WARMUP_THERMAL_TARGET" \
        --arg logSHA256 "$log_sha" --arg logArtifact "${log_path#"$run_directory"/}" \
        --arg evidenceSHA256 "$evidence_sha" \
        --argjson childExitCode "$child_exit_code" \
        --argjson blockIndex "$block_index" --argjson runPosition "$position" \
        --argjson postWarmupThermalTimeoutSeconds \
            "$POST_WARMUP_THERMAL_TIMEOUT_SECONDS" \
        --argjson postWarmupThermalPollMilliseconds \
            "$POST_WARMUP_THERMAL_POLL_MILLISECONDS" \
        --argjson postWarmupThermalStabilitySeconds \
            "$POST_WARMUP_THERMAL_STABILITY_SECONDS" \
        --argjson maxProcessRSSBytes "$max_rss_bytes" \
        --argjson evidencePresent "$evidence_present" \
        --argjson thermalEnvironment "$diagnostic_json" \
        '{schemaVersion:1,status:"FAILED",promotable:false,reason:$reason,
          childExitCode:$childExitCode,matrixID:$matrixID,
          workloadNonce:$workloadNonce,cellID:$cellID,kvQuantTier:$kvQuantTier,
          requestedAttention:(if $requestedAttention == "" then null
            else $requestedAttention end),blockIndex:$blockIndex,
          runPosition:$runPosition,harnessGitSHA:$harnessGitSHA,
          harnessBinarySHA256:$harnessBinarySHA256,
          runnerScriptSHA256:$runnerScriptSHA256,
          runnerManifestSHA256:$runnerManifestSHA256,
          modelConfigHash:$modelConfigHash,
          modelCheckpointManifestHash:$modelCheckpointManifestHash,
          modelTokenizerSHA256:$modelTokenizerSHA256,
          postWarmupThermalPolicy:{target:$postWarmupThermalTarget,
            timeoutSeconds:$postWarmupThermalTimeoutSeconds,
            pollIntervalMilliseconds:$postWarmupThermalPollMilliseconds,
            stabilitySeconds:$postWarmupThermalStabilitySeconds},
          logSHA256:$logSHA256,logArtifact:$logArtifact,
          evidencePresent:$evidencePresent,
          evidenceSHA256:(if $evidenceSHA256 == "" then null
            else $evidenceSHA256 end),thermalEnvironment:$thermalEnvironment,
          maxProcessRSSBytes:$maxProcessRSSBytes}' > "$failure_receipt_tmp"
    if [[ -d "$failure_receipt" ]]; then
        chmod 0444 "$failure_receipt_tmp"
        fail "runner failure receipt boundary is a directory"
    fi
    mv -f "$failure_receipt_tmp" "$failure_receipt"
    chmod 0444 "$failure_receipt"
}

write_progress() {
    local state="$1" completed="$2" block="$3" position="$4" cell="$5"
    local child="${6:-}" max_rss="${7:-0}" now tmp
    original_directory "$OUTPUT" "$OUTPUT_DIRECTORY_IDENTITY" \
        || return 1
    now="$(date +%s)"
    tmp="$(mktemp "$OUTPUT/.runner-progress.XXXXXX")"
    jq -cn \
        --arg state "$state" --arg matrixID "$MATRIX_ID" \
        --arg cellID "$cell" --arg harnessGitSHA "$HARNESS_SHA" \
        --arg postWarmupThermalTarget "$POST_WARMUP_THERMAL_TARGET" \
        --arg manifestSHA256 "$MANIFEST_SHA" --arg childPID "$child" \
        --argjson completedRows "$completed" --argjson totalRows "$TOTAL_ROWS" \
        --argjson blockIndex "$block" --argjson runPosition "$position" \
        --argjson heartbeatEpoch "$now" --argjson maxProcessRSSBytes "$max_rss" \
        --argjson postWarmupThermalStabilitySeconds \
            "$POST_WARMUP_THERMAL_STABILITY_SECONDS" \
        '{schemaVersion:1,state:$state,matrixID:$matrixID,cellID:$cellID,
          completedRows:$completedRows,totalRows:$totalRows,blockIndex:$blockIndex,
          runPosition:$runPosition,heartbeatEpoch:$heartbeatEpoch,
          childPID:(if $childPID == "" then null else ($childPID | tonumber) end),
          maxProcessRSSBytes:$maxProcessRSSBytes,harnessGitSHA:$harnessGitSHA,
          postWarmupThermalTarget:$postWarmupThermalTarget,
          postWarmupThermalStabilitySeconds:$postWarmupThermalStabilitySeconds,
          runnerManifestSHA256:$manifestSHA256}' > "$tmp"
    if [[ -d "$PROGRESS" ]]; then
        chmod 0444 "$tmp"
        return 1
    fi
    mv -f "$tmp" "$PROGRESS"
}

write_progress "starting" 0 0 0 "" "" 0
completed_rows=0

for ((block_index = 0; block_index < BLOCK_COUNT; block_index++)); do
    if ! original_directory "$OUTPUT" "$OUTPUT_DIRECTORY_IDENTITY" \
        || ! original_directory "$OUTPUT/runs" "$RUNS_DIRECTORY_IDENTITY" \
        || ! original_directory "$OUTPUT/blocks" "$BLOCKS_DIRECTORY_IDENTITY"; then
        write_status "INVALID_PARENT_ARTIFACT_BOUNDARY"
        fail "runner aggregate directory boundary changed"
    fi
    block_environment=""
    block_runs_directory="$OUTPUT/runs/block-$(printf '%03d' "$block_index")"
    if [[ -e "$block_runs_directory" || -L "$block_runs_directory" ]]; then
        write_status "INVALID_RUN_DIRECTORY_BOUNDARY"
        fail "matrix block run directory already exists"
    fi
    mkdir "$block_runs_directory"
    block_runs_directory_identity="$(file_identity "$block_runs_directory")"
    block_receipts="$OUTPUT/blocks/block-$(printf '%03d' "$block_index").receipts.jsonl"
    block_receipts_tmp="$(mktemp "$OUTPUT/blocks/.block-receipts.XXXXXX")"
    if [[ -d "$block_receipts" ]]; then
        chmod 0444 "$block_receipts_tmp"
        write_status "INVALID_PARENT_ARTIFACT_BOUNDARY"
        fail "block receipt aggregate boundary is a directory"
    fi
    mv -f "$block_receipts_tmp" "$block_receipts"
    block_receipts_identity="$(file_identity "$block_receipts")"
    for ((position = 0; position < CELL_COUNT; position++)); do
        if ! source_provenance_is_current; then
            write_status "INPUT_CHANGED"
            fail "source stamp changed or live Git provenance drifted during qualification"
        fi
        if [[ "$(shasum -a 256 "$MANIFEST" | awk '{print $1}')" != "$MANIFEST_SHA" \
            || "$(shasum -a 256 "$RUNNER_PATH" | awk '{print $1}')" != "$RUNNER_SHA" \
            || "$(shasum -a 256 "$BIN" | awk '{print $1}')" != "$BINARY_SHA" ]]; then
            write_status "INPUT_CHANGED"
            fail "runner manifest, runner script, or harness binary changed during qualification"
        fi
        cell_id="$(jq -r ".blocks[$block_index][$position]" "$MANIFEST")"
        cell_json="$(jq -c --arg id "$cell_id" '.cells[] | select(.id == $id)' "$MANIFEST")"
        kv_quant="$(jq -r '.kvQuant' <<< "$cell_json")"
        attention="$(jq -r '.attention? // empty' <<< "$cell_json")"
        checkpoint_sha="$(jq -r '.checkpointContentSHA256? // empty' <<< "$cell_json")"
        schedule="$(jq -r '.kvtunerSchedule? // empty' <<< "$cell_json")"
        if [[ -n "$schedule" ]]; then
            schedule="$(absolute_from_caller "$schedule")"
        fi
        schedule_bundle_sha="$(jq -r '.kvtunerBundleSHA256? // empty' <<< "$cell_json")"
        schedule_artifact_sha="$(jq -r '.kvtunerScheduleArtifactSHA256? // empty' <<< "$cell_json")"
        if ! original_directory "$OUTPUT" "$OUTPUT_DIRECTORY_IDENTITY" \
            || ! original_directory "$OUTPUT/runs" "$RUNS_DIRECTORY_IDENTITY" \
            || ! original_directory "$block_runs_directory" \
                "$block_runs_directory_identity" \
            || ! original_directory "$OUTPUT/blocks" \
                "$BLOCKS_DIRECTORY_IDENTITY"; then
            write_status "INVALID_RUN_DIRECTORY_BOUNDARY"
            fail "matrix block run directory boundary changed"
        fi
        run_dir="$block_runs_directory/position-$(printf '%03d' "$position")-$cell_id"
        if [[ -e "$run_dir" || -L "$run_dir" ]]; then
            write_status "INVALID_RUN_DIRECTORY_BOUNDARY"
            fail "matrix position run directory already exists"
        fi
        mkdir "$run_dir"
        run_dir_identity="$(file_identity "$run_dir")"
        evidence="$run_dir/bench.jsonl"
        run_log="$run_dir/bench.log"
        receipt="$run_dir/runner-receipt.json"
        : > "$run_log"
        run_log_identity="$(file_identity "$run_log")"
        exec 9< "$run_log"
        run_log_monitor="/dev/fd/9"

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
            --post-warmup-thermal-target "$POST_WARMUP_THERMAL_TARGET"
            --post-warmup-thermal-timeout-seconds \
                "$POST_WARMUP_THERMAL_TIMEOUT_SECONDS"
            --post-warmup-thermal-poll-milliseconds \
                "$POST_WARMUP_THERMAL_POLL_MILLISECONDS"
            --post-warmup-thermal-stability-seconds \
                "$POST_WARMUP_THERMAL_STABILITY_SECONDS"
            --evidence "$evidence"
        )
        if [[ -n "$attention" ]]; then
            args+=(--kv-attention "$attention" \
                --checkpoint-content-sha256 "$checkpoint_sha")
        fi
        if [[ -n "$schedule" ]]; then
            if [[ "$(shasum -a 256 "$schedule" | awk '{print $1}')" != "$schedule_bundle_sha" ]]; then
                write_status "INPUT_CHANGED"
                fail "KVTuner qualification bundle changed after manifest authentication"
            fi
            args+=(--kvtuner-schedule "$schedule")
        fi

        if ! original_directory "$OUTPUT" "$OUTPUT_DIRECTORY_IDENTITY" \
            || ! original_regular_file "$RUNNER_LOG" "$RUNNER_LOG_IDENTITY"; then
            write_status "INVALID_PARENT_ARTIFACT_BOUNDARY"
            fail "runner log boundary changed before harness launch"
        fi
        harness_pid="$run_dir/harness.pid"
        harness_pid_tmp="$(mktemp "$run_dir/.harness-pid.XXXXXX")"
        exec 8> "$harness_pid_tmp"
        printf '[%s] launch block=%s position=%s cell=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$block_index" "$position" "$cell_id" \
            >> "$RUNNER_LOG"
        (
            exec 8>&-
            exec 9<&-
            clear_caller_git_environment
            unset HARNESS_GIT_SHA
            cd "$HARNESS_SOURCE_DIR"
            exec "$BIN" "${args[@]}"
        ) > "$run_log" 2>&1 &
        active_child=$!
        printf '%s\n' "$active_child" >&8
        exec 8>&-
        if ! original_directory "$OUTPUT" "$OUTPUT_DIRECTORY_IDENTITY" \
            || ! original_directory "$OUTPUT/runs" "$RUNS_DIRECTORY_IDENTITY" \
            || ! original_directory "$block_runs_directory" \
                "$block_runs_directory_identity" \
            || ! original_directory "$run_dir" "$run_dir_identity"; then
            write_status "INVALID_RUN_DIRECTORY_BOUNDARY"
            terminate_active_child
            fail "run directory boundary changed during harness launch"
        fi
        if [[ -d "$harness_pid" ]]; then
            chmod 0444 "$harness_pid_tmp"
            write_status "INVALID_PARENT_ARTIFACT_BOUNDARY"
            terminate_active_child
            fail "harness PID boundary is a directory"
        fi
        mv -f "$harness_pid_tmp" "$harness_pid"
        chmod 0444 "$harness_pid"
        max_rss_bytes=0
        write_progress "running" "$completed_rows" "$block_index" "$position" \
            "$cell_id" "$active_child" "$max_rss_bytes"
        last_log_mtime="$(file_mtime "$run_log_monitor")"
        while active_child_running; do
            rss_kb="$(ps -o rss= -p "$active_child" 2>/dev/null | tr -d '[:space:]' || true)"
            if [[ "$rss_kb" =~ ^[0-9]+$ ]]; then
                rss_bytes=$((rss_kb * 1024))
                (( rss_bytes > max_rss_bytes )) && max_rss_bytes=$rss_bytes
            fi
            now="$(date +%s)"
            current_mtime="$(file_mtime "$run_log_monitor")"
            (( current_mtime > last_log_mtime )) && last_log_mtime=$current_mtime
            if (( now - last_log_mtime > WATCHDOG_SECONDS )); then
                if ! original_directory "$OUTPUT" "$OUTPUT_DIRECTORY_IDENTITY"; then
                    terminate_active_child
                    fail "output root boundary changed during harness execution"
                fi
                watchdog="$OUTPUT/runner.watchdog.json"
                watchdog_tmp="$(mktemp "$OUTPUT/.runner-watchdog.XXXXXX")"
                jq -cn --arg cellID "$cell_id" --argjson blockIndex "$block_index" \
                    --argjson runPosition "$position" --argjson childPID "$active_child" \
                    --argjson detectedEpoch "$now" --argjson lastLogMtime "$last_log_mtime" \
                    '{schemaVersion:1,reason:"log-stalled",cellID:$cellID,
                      blockIndex:$blockIndex,runPosition:$runPosition,childPID:$childPID,
                      detectedEpoch:$detectedEpoch,lastLogMtime:$lastLogMtime}' \
                    > "$watchdog_tmp"
                if [[ -d "$watchdog" ]]; then
                    chmod 0444 "$watchdog_tmp"
                    write_status "INVALID_PARENT_ARTIFACT_BOUNDARY"
                    terminate_active_child
                    fail "watchdog artifact boundary is a directory"
                fi
                mv -f "$watchdog_tmp" "$watchdog"
                chmod 0444 "$watchdog"
                write_status "WATCHDOG"
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
        if ! original_directory "$OUTPUT" "$OUTPUT_DIRECTORY_IDENTITY"; then
            exec 9<&-
            fail "output root boundary changed during harness execution"
        fi
        if ! original_directory "$OUTPUT/runs" "$RUNS_DIRECTORY_IDENTITY" \
            || ! original_directory "$block_runs_directory" \
                "$block_runs_directory_identity" \
            || ! original_directory "$run_dir" "$run_dir_identity" \
            || ! original_directory "$OUTPUT/blocks" \
                "$BLOCKS_DIRECTORY_IDENTITY"; then
            captured_log_tmp="$(mktemp "$OUTPUT/.runner-captured-log.XXXXXX")"
            cat <&9 > "$captured_log_tmp"
            exec 9<&-
            chmod 0444 "$captured_log_tmp"
            write_status "INVALID_RUN_DIRECTORY_BOUNDARY"
            write_progress "invalid-run-directory-boundary" "$completed_rows" \
                "$block_index" "$position" "$cell_id" "" "$max_rss_bytes"
            fail "run directory boundary changed during harness execution"
        fi
        captured_log_tmp="$(mktemp "$run_dir/.runner-captured-log.XXXXXX")"
        cat <&9 > "$captured_log_tmp"
        exec 9<&-
        captured_log_sha="$(shasum -a 256 "$captured_log_tmp" | awk '{print $1}')"
        authenticated_run_log="$run_log"
        log_boundary_valid=false
        if [[ -f "$run_log" && ! -L "$run_log" ]] \
            && [[ "$(file_identity "$run_log")" == "$run_log_identity" ]] \
            && [[ "$(shasum -a 256 "$run_log" | awk '{print $1}')" \
                == "$captured_log_sha" ]]; then
            log_boundary_valid=true
            rm -f "$captured_log_tmp"
        else
            recovered_log="$run_dir/runner-captured.log"
            if [[ ! -e "$recovered_log" && ! -L "$recovered_log" ]]; then
                mv "$captured_log_tmp" "$recovered_log"
                authenticated_run_log="$recovered_log"
            else
                authenticated_run_log="$captured_log_tmp"
            fi
            chmod 0444 "$authenticated_run_log"
        fi
        if ! original_directory "$OUTPUT" "$OUTPUT_DIRECTORY_IDENTITY" \
            || ! original_regular_file "$RUNNER_LOG" "$RUNNER_LOG_IDENTITY"; then
            write_harness_failure_receipt \
                "$child_rc" "$run_dir" "$evidence" "$authenticated_run_log" \
                "parent-artifact-boundary-changed"
            write_status "INVALID_PARENT_ARTIFACT_BOUNDARY"
            write_progress "invalid-parent-artifact-boundary" "$completed_rows" \
                "$block_index" "$position" "$cell_id" "" "$max_rss_bytes"
            fail "runner log boundary changed after harness execution"
        fi
        cat "$authenticated_run_log" >> "$RUNNER_LOG"
        if [[ "$log_boundary_valid" != "true" ]]; then
            write_harness_failure_receipt \
                "$child_rc" "$run_dir" "$evidence" "$authenticated_run_log" \
                "log-boundary-changed"
            write_status "INVALID_LOG_BOUNDARY"
            write_progress "invalid-log-boundary" "$completed_rows" \
                "$block_index" "$position" "$cell_id" "" "$max_rss_bytes"
            fail "harness log boundary changed for block=$block_index position=$position cell=$cell_id"
        fi
        if (( child_rc != 0 )); then
            write_harness_failure_receipt \
                "$child_rc" "$run_dir" "$evidence" "$authenticated_run_log"
            write_status "FAILED"
            write_progress "failed" "$completed_rows" "$block_index" "$position" \
                "$cell_id" "" "$max_rss_bytes"
            fail "harness failed for block=$block_index position=$position cell=$cell_id"
        fi

        if [[ ! -f "$evidence" || -L "$evidence" ]] \
            || [[ "$(wc -l < "$evidence" 2>/dev/null | tr -d '[:space:]')" != "1" ]]; then
            write_status "INVALID_EVIDENCE"
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
            --arg thermalTarget "$POST_WARMUP_THERMAL_TARGET" \
            --argjson block "$block_index" --argjson position "$position" \
            --argjson count "$CELL_COUNT" --argjson memory "$MEMORY_LIMIT" \
            --argjson cache "$CACHE_LIMIT" --argjson wired "$WIRED_LIMIT" \
            --argjson thermalTimeout "$POST_WARMUP_THERMAL_TIMEOUT_SECONDS" \
            --argjson thermalPoll "$POST_WARMUP_THERMAL_POLL_MILLISECONDS" \
            --argjson thermalStability "$POST_WARMUP_THERMAL_STABILITY_SECONDS" \
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
              .payload.qualification.schemaVersion == 4 and
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
              .payload.qualification.context.postWarmupThermalPolicy.target ==
                $thermalTarget and
              .payload.qualification.context.postWarmupThermalPolicy.timeoutSeconds ==
                $thermalTimeout and
              .payload.qualification.context.postWarmupThermalPolicy.pollIntervalMilliseconds ==
                $thermalPoll and
              .payload.qualification.context.postWarmupThermalPolicy.stabilitySeconds ==
                $thermalStability and
              .payload.qualification.warmup.before.monotonicTimestampSeconds <
                .payload.qualification.warmup.after.monotonicTimestampSeconds and
              .payload.qualification.warmup.before.residentSizeBytes > 0 and
              .payload.qualification.warmup.after.residentSizeBytes > 0 and
              .payload.qualification.warmup.before.physicalFootprintBytes > 0 and
              .payload.qualification.warmup.after.physicalFootprintBytes > 0 and
              .payload.qualification.warmup.before.powerSource == "ac-power" and
              .payload.qualification.warmup.after.powerSource == "ac-power" and
              (.payload.qualification.warmup.before.lowPowerModeEnabled | not) and
              (.payload.qualification.warmup.after.lowPowerModeEnabled | not) and
              (.payload.qualification.warmup.before.thermalState == "nominal" or
                .payload.qualification.warmup.before.thermalState == "fair") and
              (.payload.qualification.warmup.after.thermalState == "nominal" or
                .payload.qualification.warmup.after.thermalState == "fair") and
              (.payload.qualification.postWarmupThermalAdmission as $admission |
                ($admission.stabilityObservations | type == "array" and length >= 2) and
                (all($admission.stabilityObservations[];
                  .residentSizeBytes > 0 and .physicalFootprintBytes > 0 and
                  .powerSource == "ac-power" and
                  (.lowPowerModeEnabled | not) and
                  .thermalState == $thermalTarget)) and
                ([$admission.stabilityObservations[].monotonicTimestampSeconds]
                  as $timestamps |
                  all(range(1; ($timestamps | length));
                    $timestamps[.] > $timestamps[. - 1])) and
                $admission.stabilityObservations[0].monotonicTimestampSeconds >=
                  .payload.qualification.warmup.after.monotonicTimestampSeconds and
                $admission.stabilityObservations[-1] == $admission.snapshot and
                ($admission.snapshot.monotonicTimestampSeconds -
                  $admission.stabilityObservations[0].monotonicTimestampSeconds) >=
                  $thermalStability and
                ($admission.snapshot.monotonicTimestampSeconds -
                  .payload.qualification.warmup.after.monotonicTimestampSeconds) <=
                  $thermalTimeout and
                $admission.snapshot.residentSizeBytes > 0 and
                $admission.snapshot.physicalFootprintBytes > 0 and
                $admission.snapshot.powerSource == "ac-power" and
                ($admission.snapshot.lowPowerModeEnabled | not) and
                $admission.snapshot.thermalState == $thermalTarget) and
              (.payload.qualification.runs | type == "array" and length == 1) and
              .payload.qualification.postWarmupThermalAdmission.snapshot.monotonicTimestampSeconds <=
                .payload.qualification.runs[0].before.monotonicTimestampSeconds and
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
              .payload.qualification.runs[0].before.powerSource == "ac-power" and
              (.payload.qualification.runs[0].before.lowPowerModeEnabled | not) and
              (.payload.qualification.runs[0].after.lowPowerModeEnabled | not) and
              .payload.qualification.runs[0].before.thermalState ==
                .payload.qualification.runs[0].after.thermalState and
              .payload.qualification.runs[0].before.thermalState == $thermalTarget and
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
                 (.payload.compressedKVAttention.admission |
                   if .modelType == "qwen3" then
                     .family == "qwen3" and .architecture == "Qwen3ForCausalLM"
                   elif .modelType == "llama" then
                     .family == "llama" and .architecture == "LlamaForCausalLM"
                   elif .modelType == "phi3" then
                     .family == "phi3" and .architecture == "Phi3ForCausalLM"
                   else false end) and
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
                 (.payload.kvtunerSchedule.layers | type == "array") and
                 (.payload.kvtunerSchedule.layers | length) ==
                   .payload.compressedKVAttention.admission.layerCount end)
            ' "$evidence" >/dev/null; then
            write_status "INVALID_EVIDENCE"
            fail "harness row failed exact identity or qualification authentication"
        fi
        typed_validation_log="$run_dir/typed-validation.log"
        typed_validation_tmp="$(mktemp "$run_dir/.typed-validation.XXXXXX")"
        set +e
        "$BIN" validate-bench-qualification --evidence "$evidence" \
            > "$typed_validation_tmp" 2>&1
        typed_validation_rc=$?
        set -e
        if ! original_directory "$OUTPUT" "$OUTPUT_DIRECTORY_IDENTITY" \
            || ! original_directory "$OUTPUT/runs" "$RUNS_DIRECTORY_IDENTITY" \
            || ! original_directory "$block_runs_directory" \
                "$block_runs_directory_identity" \
            || ! original_directory "$run_dir" "$run_dir_identity" \
            || ! original_directory "$OUTPUT/blocks" \
                "$BLOCKS_DIRECTORY_IDENTITY"; then
            write_status "INVALID_RUN_DIRECTORY_BOUNDARY"
            write_progress "invalid-run-directory-boundary" "$completed_rows" \
                "$block_index" "$position" "$cell_id" "" "$max_rss_bytes"
            fail "run directory boundary changed during typed validation"
        fi
        if [[ -d "$typed_validation_log" ]]; then
            chmod 0444 "$typed_validation_tmp"
            write_status "INVALID_PARENT_ARTIFACT_BOUNDARY"
            fail "typed validation log boundary is a directory"
        fi
        mv -f "$typed_validation_tmp" "$typed_validation_log"
        chmod 0444 "$typed_validation_log"
        if ! original_directory "$OUTPUT" "$OUTPUT_DIRECTORY_IDENTITY" \
            || ! original_regular_file "$RUNNER_LOG" "$RUNNER_LOG_IDENTITY"; then
            write_harness_failure_receipt \
                "$typed_validation_rc" "$run_dir" "$evidence" "$authenticated_run_log" \
                "parent-artifact-boundary-changed"
            write_status "INVALID_PARENT_ARTIFACT_BOUNDARY"
            write_progress "invalid-parent-artifact-boundary" "$completed_rows" \
                "$block_index" "$position" "$cell_id" "" "$max_rss_bytes"
            fail "runner log boundary changed during typed validation"
        fi
        if (( typed_validation_rc != 0 )); then
            cat "$typed_validation_log" >> "$RUNNER_LOG"
            write_status "INVALID_EVIDENCE"
            fail "harness row failed typed qualification validation"
        fi

        internal_rss="$(jq -r '[
            .payload.qualification.warmup.before.residentSizeBytes,
            .payload.qualification.warmup.after.residentSizeBytes,
            .payload.qualification.postWarmupThermalAdmission.snapshot.residentSizeBytes,
            .payload.qualification.runs[0].before.residentSizeBytes,
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
            invalid_tmp="$(mktemp "$OUTPUT/blocks/.block-invalid.XXXXXX")"
            jq -cn --arg reason "block environment changed" \
                --arg baseline "$block_environment" --arg observed "$environment_key" \
                --arg evidenceSHA256 "$(shasum -a 256 "$evidence" | awk '{print $1}')" \
                --arg cellID "$cell_id" --argjson blockIndex "$block_index" \
                --argjson runPosition "$position" \
                '{schemaVersion:1,reason:$reason,blockIndex:$blockIndex,
                  runPosition:$runPosition,cellID:$cellID,
                  baselineEnvironment:($baseline | fromjson),
                  observedEnvironment:($observed | fromjson),
                  evidenceSHA256:$evidenceSHA256}' > "$invalid_tmp"
            if [[ -d "$invalid" ]]; then
                chmod 0444 "$invalid_tmp"
                write_status "INVALID_PARENT_ARTIFACT_BOUNDARY"
                fail "invalid-block artifact boundary is a directory"
            fi
            mv -f "$invalid_tmp" "$invalid"
            chmod 0444 "$invalid"
            write_status "INVALID_BLOCK_ENVIRONMENT"
            write_progress "invalid-block-environment" "$completed_rows" \
                "$block_index" "$position" "$cell_id" "" "$max_rss_bytes"
            fail "block environment changed across matrix rows"
        fi

        evidence_sha="$(shasum -a 256 "$evidence" | awk '{print $1}')"
        log_sha="$(shasum -a 256 "$run_log" | awk '{print $1}')"
        receipt_tmp="$(mktemp "$run_dir/.runner-receipt.XXXXXX")"
        jq -cn \
            --arg matrixID "$MATRIX_ID" --arg cellID "$cell_id" \
            --arg harnessGitSHA "$HARNESS_SHA" --arg harnessBinarySHA256 "$BINARY_SHA" \
            --arg runnerScriptSHA256 "$RUNNER_SHA" \
            --arg runnerManifestSHA256 "$MANIFEST_SHA" --arg evidenceSHA256 "$evidence_sha" \
            --arg logSHA256 "$log_sha" --argjson blockIndex "$block_index" \
            --arg postWarmupThermalTarget "$POST_WARMUP_THERMAL_TARGET" \
            --argjson postWarmupThermalTimeoutSeconds \
                "$POST_WARMUP_THERMAL_TIMEOUT_SECONDS" \
            --argjson postWarmupThermalPollMilliseconds \
                "$POST_WARMUP_THERMAL_POLL_MILLISECONDS" \
            --argjson postWarmupThermalStabilitySeconds \
                "$POST_WARMUP_THERMAL_STABILITY_SECONDS" \
            --argjson runPosition "$position" --argjson maxProcessRSSBytes "$max_rss_bytes" \
            '{schemaVersion:1,matrixID:$matrixID,cellID:$cellID,blockIndex:$blockIndex,
              runPosition:$runPosition,harnessGitSHA:$harnessGitSHA,
              harnessBinarySHA256:$harnessBinarySHA256,
              runnerScriptSHA256:$runnerScriptSHA256,
              runnerManifestSHA256:$runnerManifestSHA256,
              postWarmupThermalTarget:$postWarmupThermalTarget,
              postWarmupThermalTimeoutSeconds:$postWarmupThermalTimeoutSeconds,
              postWarmupThermalPollMilliseconds:$postWarmupThermalPollMilliseconds,
              postWarmupThermalStabilitySeconds:$postWarmupThermalStabilitySeconds,
              evidenceSHA256:$evidenceSHA256,logSHA256:$logSHA256,
              maxProcessRSSBytes:$maxProcessRSSBytes}' > "$receipt_tmp"
        if [[ -d "$receipt" ]]; then
            chmod 0444 "$receipt_tmp"
            write_status "INVALID_RECEIPT_BOUNDARY"
            fail "runner receipt boundary is a directory"
        fi
        mv -f "$receipt_tmp" "$receipt"
        chmod 0444 "$receipt"
        if ! original_directory "$OUTPUT" "$OUTPUT_DIRECTORY_IDENTITY" \
            || ! original_directory "$OUTPUT/blocks" "$BLOCKS_DIRECTORY_IDENTITY" \
            || ! original_regular_file "$block_receipts" "$block_receipts_identity" \
            || ! original_regular_file "$RECEIPT_SET" "$RECEIPT_SET_IDENTITY"; then
            write_status "INVALID_PARENT_ARTIFACT_BOUNDARY"
            fail "parent receipt aggregate boundary changed"
        fi
        jq -c . "$receipt" >> "$block_receipts"
        receipt_sha="$(shasum -a 256 "$receipt" | awk '{print $1}')"
        printf '%s  %s\n' "$receipt_sha" \
            "${receipt#"$OUTPUT"/}" >> "$RECEIPT_SET"
        completed_rows=$((completed_rows + 1))
        write_progress "row-complete" "$completed_rows" "$block_index" "$position" \
            "$cell_id" "" "$max_rss_bytes"
    done
    if ! original_directory "$OUTPUT" "$OUTPUT_DIRECTORY_IDENTITY" \
        || ! original_directory "$OUTPUT/blocks" "$BLOCKS_DIRECTORY_IDENTITY" \
        || ! original_regular_file "$block_receipts" "$block_receipts_identity"; then
        write_status "INVALID_PARENT_ARTIFACT_BOUNDARY"
        fail "block receipt aggregate boundary changed before completion"
    fi
    block_completion="$OUTPUT/blocks/block-$(printf '%03d' "$block_index").complete.json"
    block_completion_tmp="$(mktemp "$OUTPUT/blocks/.block-complete.XXXXXX")"
    jq -s --arg matrixID "$MATRIX_ID" --arg environment "$block_environment" \
        --arg postWarmupThermalTarget "$POST_WARMUP_THERMAL_TARGET" \
        --argjson postWarmupThermalStabilitySeconds \
            "$POST_WARMUP_THERMAL_STABILITY_SECONDS" \
        --argjson blockIndex "$block_index" \
        '{schemaVersion:1,matrixID:$matrixID,blockIndex:$blockIndex,
          postWarmupThermalTarget:$postWarmupThermalTarget,
          postWarmupThermalStabilitySeconds:$postWarmupThermalStabilitySeconds,
          environment:($environment | fromjson),receipts:.}' \
        "$block_receipts" > "$block_completion_tmp"
    if [[ -d "$block_completion" ]]; then
        chmod 0444 "$block_completion_tmp"
        write_status "INVALID_PARENT_ARTIFACT_BOUNDARY"
        fail "block completion boundary is a directory"
    fi
    mv -f "$block_completion_tmp" "$block_completion"
    chmod 0444 "$block_completion"
done

if [[ "$(shasum -a 256 "$MANIFEST" | awk '{print $1}')" != "$MANIFEST_SHA" \
    || "$(shasum -a 256 "$RUNNER_PATH" | awk '{print $1}')" != "$RUNNER_SHA" \
    || "$(shasum -a 256 "$BIN" | awk '{print $1}')" != "$BINARY_SHA" ]]; then
    write_status "INPUT_CHANGED"
    fail "qualification inputs changed before finalization"
fi
if ! original_directory "$OUTPUT" "$OUTPUT_DIRECTORY_IDENTITY" \
    || ! original_regular_file "$RUNNER_LOG" "$RUNNER_LOG_IDENTITY" \
    || ! original_regular_file "$RECEIPT_SET" "$RECEIPT_SET_IDENTITY"; then
    write_status "INVALID_PARENT_ARTIFACT_BOUNDARY"
    fail "parent aggregate boundary changed before finalization"
fi
printf '[%s] complete rows=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$completed_rows" >> "$RUNNER_LOG"
RECEIPT_SET_SHA="$(shasum -a 256 "$RECEIPT_SET" | awk '{print $1}')"
completion="$OUTPUT/runner.completion.json"
completion_tmp="$(mktemp "$OUTPUT/.runner-completion.XXXXXX")"
jq -cn --arg matrixID "$MATRIX_ID" --arg harnessGitSHA "$HARNESS_SHA" \
    --arg postWarmupThermalTarget "$POST_WARMUP_THERMAL_TARGET" \
    --arg runnerManifestSHA256 "$MANIFEST_SHA" \
    --arg harnessBinarySHA256 "$BINARY_SHA" --arg runnerScriptSHA256 "$RUNNER_SHA" \
    --arg receiptSetSHA256 "$RECEIPT_SET_SHA" --argjson blockCount "$BLOCK_COUNT" \
    --argjson cellCount "$CELL_COUNT" --argjson completedRows "$completed_rows" \
    --argjson postWarmupThermalStabilitySeconds \
        "$POST_WARMUP_THERMAL_STABILITY_SECONDS" \
    '{schemaVersion:1,status:"COMPLETE",matrixID:$matrixID,blockCount:$blockCount,
      cellCount:$cellCount,completedRows:$completedRows,harnessGitSHA:$harnessGitSHA,
      harnessBinarySHA256:$harnessBinarySHA256,
      runnerScriptSHA256:$runnerScriptSHA256,
      runnerManifestSHA256:$runnerManifestSHA256,
      postWarmupThermalTarget:$postWarmupThermalTarget,
      postWarmupThermalStabilitySeconds:$postWarmupThermalStabilitySeconds,
      receiptSetSHA256:$receiptSetSHA256}' > "$completion_tmp"
if [[ -d "$completion" ]]; then
    chmod 0444 "$completion_tmp"
    write_status "INVALID_PARENT_ARTIFACT_BOUNDARY"
    fail "runner completion boundary is a directory"
fi
mv -f "$completion_tmp" "$completion"
chmod 0444 "$completion"
write_progress "complete" "$completed_rows" "$((BLOCK_COUNT - 1))" \
    "$((CELL_COUNT - 1))" "" "" 0
write_status "COMPLETE"
