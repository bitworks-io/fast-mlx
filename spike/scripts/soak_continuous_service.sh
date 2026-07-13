#!/bin/bash
# Resident continuous-batching soak with a watchdog independent of inference progress.
# Usage: soak_continuous_service.sh <model-path> [measured-duration-seconds]
set -euo pipefail

MODEL="${1:?usage: soak_continuous_service.sh <model-path> [measured-duration-seconds]}"
DURATION_SECONDS="${2:-86400}"
CONCURRENCY="${CONCURRENCY:-4}"
MAX_TOKENS="${MAX_TOKENS:-64}"
PREFILL_CHUNK="${PREFILL_CHUNK:-16}"
KEEPALIVE_MS="${KEEPALIVE_MS:-1000}"
RESPONSIVENESS_MS="${RESPONSIVENESS_MS:-30000}"
MAX_RSS_DRIFT_PERCENT="${MAX_RSS_DRIFT_PERCENT:-5}"
CHECKPOINT_INTERVAL_SECONDS="${CHECKPOINT_INTERVAL_SECONDS:-300}"
WATCHDOG_SECONDS="${WATCHDOG_SECONDS:-300}"
POLL_SECONDS="${POLL_SECONDS:-5}"
KILL_GRACE_SECONDS="${KILL_GRACE_SECONDS:-15}"
PROGRESS="${PROGRESS:-continuous-service-soak-progress.json}"
EVIDENCE="${EVIDENCE:-harness-evidence.jsonl}"
LOG="${LOG:-continuous-service-soak.log}"
WATCHDOG_LOG="${WATCHDOG_LOG:-continuous-service-soak-watchdog.json}"
BIN="${BIN:-$(ls ~/Library/Developer/Xcode/DerivedData/fast-mlx-spike-*/Build/Products/Release/fastmlx-harness)}"
JQ="${JQ:-jq}"

for value in "$DURATION_SECONDS" "$CONCURRENCY" "$MAX_TOKENS" "$PREFILL_CHUNK" \
    "$KEEPALIVE_MS" "$RESPONSIVENESS_MS" "$MAX_RSS_DRIFT_PERCENT" \
    "$CHECKPOINT_INTERVAL_SECONDS" "$WATCHDOG_SECONDS" "$POLL_SECONDS" \
    "$KILL_GRACE_SECONDS"; do
    [[ "$value" =~ ^[0-9]{1,9}$ ]] || {
        echo "soak numeric arguments must be 1-9 decimal digits" >&2
        exit 2
    }
done
for name in DURATION_SECONDS CONCURRENCY MAX_TOKENS PREFILL_CHUNK KEEPALIVE_MS \
    RESPONSIVENESS_MS MAX_RSS_DRIFT_PERCENT WATCHDOG_SECONDS POLL_SECONDS \
    KILL_GRACE_SECONDS CHECKPOINT_INTERVAL_SECONDS; do
    printf -v "$name" '%d' "$((10#${!name}))"
done
(( DURATION_SECONDS >= 1 && DURATION_SECONDS <= 172800 )) || {
    echo "measured duration must be 1...172800 seconds" >&2
    exit 2
}
(( WATCHDOG_SECONDS > POLL_SECONDS && POLL_SECONDS > 0 )) || {
    echo "watchdog must exceed a positive poll interval" >&2
    exit 2
}
(( WATCHDOG_SECONDS <= 3600 && POLL_SECONDS <= 60 \
    && KILL_GRACE_SECONDS >= 1 && KILL_GRACE_SECONDS <= 60 )) || {
    echo "watchdog must be <=3600s, poll <=60s, and kill grace 1...60s" >&2
    exit 2
}
[[ "$CONCURRENCY" == "4" || "$CONCURRENCY" == "8" ]] || {
    echo "soak concurrency must be 4 or 8 so every mixed workload is present" >&2
    exit 2
}
(( MAX_TOKENS >= 2 && MAX_TOKENS <= 256 \
    && CHECKPOINT_INTERVAL_SECONDS >= 60 && CHECKPOINT_INTERVAL_SECONDS <= 3600 )) || {
    echo "soak max tokens must be 2...256 and checkpoint interval 60...3600s" >&2
    exit 2
}
responsiveness_seconds=$(((RESPONSIVENESS_MS + 999) / 1000))
minimum_watchdog_seconds=$((MAX_TOKENS + responsiveness_seconds + 60))
(( WATCHDOG_SECONDS >= minimum_watchdog_seconds )) || {
    echo "watchdog must be at least max-tokens + responsiveness deadline + 60s" >&2
    exit 2
}

mkdir -p "$(dirname "$PROGRESS")" "$(dirname "$EVIDENCE")" \
    "$(dirname "$LOG")" "$(dirname "$WATCHDOG_LOG")"
progress_abs="$(cd "$(dirname "$PROGRESS")" && pwd -P)/$(basename "$PROGRESS")"
evidence_abs="$(cd "$(dirname "$EVIDENCE")" && pwd -P)/$(basename "$EVIDENCE")"
log_abs="$(cd "$(dirname "$LOG")" && pwd -P)/$(basename "$LOG")"
watchdog_log_abs="$(cd "$(dirname "$WATCHDOG_LOG")" && pwd -P)/$(basename "$WATCHDOG_LOG")"
raw_output_paths=("$PROGRESS" "$EVIDENCE" "$LOG" "$WATCHDOG_LOG")
output_paths=("$progress_abs" "$evidence_abs" "$log_abs" "$watchdog_log_abs")
progress_lock="${progress_abs}.lock"
raw_output_paths+=("$progress_lock")
output_paths+=("$progress_lock")
for path in "${raw_output_paths[@]}"; do
    [[ ! -L "$path" ]] || {
        echo "soak output paths must not be symbolic links" >&2
        exit 2
    }
done
for ((left = 0; left < ${#output_paths[@]}; left++)); do
    for ((right = left + 1; right < ${#output_paths[@]}; right++)); do
        shopt -s nocasematch
        paths_differ=true
        if [[ "${output_paths[$left]}" == "${output_paths[$right]}" ]]; then
            paths_differ=false
        fi
        shopt -u nocasematch
        $paths_differ || {
            echo "soak outputs and the internal progress lock must be distinct" >&2
            exit 2
        }
    done
done
command -v "$JQ" >/dev/null 2>&1 || {
    echo "jq is required to validate watchdog heartbeat ownership" >&2
    exit 2
}
soak_pid=""
lock_owned=false
launch_started=false
created_output_paths=()
critical_section=false
pending_signal=""

soak_job_running() {
    local running_jobs
    running_jobs=" $(jobs -pr | tr '\n' ' ') "
    if [[ -n "$soak_pid" ]]; then
        [[ "$running_jobs" == *" $soak_pid "* ]]
    else
        # A signal can arrive after Bash registers the sole async job but before `$!` is
        # assigned. In that window, a nonempty running-job set is still unambiguous ownership.
        [[ "$running_jobs" != "  " ]]
    fi
}

terminate_soak() {
    # The harness is the wrapper's only background job. Signalling Bash's current-job spec
    # keeps ownership tied to the job table, so a recycled numeric PID cannot be targeted.
    if soak_job_running; then
        kill -TERM "%+" 2>/dev/null || true
        remaining="$KILL_GRACE_SECONDS"
        while soak_job_running && (( remaining > 0 )); do
            sleep 1
            remaining=$((remaining - 1))
        done
        if soak_job_running; then
            kill -KILL "%+" 2>/dev/null || true
        fi
    fi
    if [[ -n "$soak_pid" ]]; then
        wait "$soak_pid" 2>/dev/null || true
    elif $launch_started; then
        wait "%+" 2>/dev/null || true
    fi
}

cleanup() {
    terminate_soak
    if ! $launch_started; then
        for path in "${created_output_paths[@]}"; do
            rm -f "$path" 2>/dev/null || true
        done
    fi
    if $lock_owned; then
        rmdir "$progress_lock" 2>/dev/null || true
    fi
}

handle_interrupt() {
    if $critical_section; then
        pending_signal="INT"
        return
    fi
    trap - INT TERM EXIT
    cleanup
    exit 130
}

handle_termination() {
    if $critical_section; then
        pending_signal="TERM"
        return
    fi
    trap - INT TERM EXIT
    cleanup
    exit 143
}

finish_critical_section() {
    critical_section=false
    case "$pending_signal" in
        INT) pending_signal=""; handle_interrupt ;;
        TERM) pending_signal=""; handle_termination ;;
    esac
}

trap handle_interrupt INT
trap handle_termination TERM
trap cleanup EXIT

critical_section=true
set +e
mkdir "$progress_lock" 2>/dev/null
lock_rc=$?
set -e
if (( lock_rc == 0 )); then
    lock_owned=true
fi
finish_critical_section
(( lock_rc == 0 )) || {
    echo "soak progress path is already owned by another watchdog" >&2
    exit 2
}

# Materialize every configured output without truncating an existing file, then ask the
# filesystem whether any pair is the same object. This catches APFS case and Unicode
# normalization aliases that Bash string comparison cannot represent faithfully.
for path in "${raw_output_paths[@]:0:4}"; do
    critical_section=true
    reservation_failed=false
    if [[ ! -e "$path" ]]; then
        if (set -o noclobber; : > "$path") 2>/dev/null; then
            created_output_paths+=("$path")
        elif [[ ! -e "$path" ]]; then
            reservation_failed=true
        fi
    fi
    finish_critical_section
    if $reservation_failed; then
        echo "could not reserve soak output path: $path" >&2
        exit 2
    fi
done
for ((left = 0; left < ${#raw_output_paths[@]}; left++)); do
    for ((right = left + 1; right < ${#raw_output_paths[@]}; right++)); do
        if [[ "${raw_output_paths[$left]}" -ef "${raw_output_paths[$right]}" ]]; then
            echo "soak outputs and the internal progress lock must be distinct" >&2
            exit 2
        fi
    done
done
for path in "${raw_output_paths[@]:0:4}"; do
    [[ -f "$path" ]] || {
        echo "soak output destinations must be regular files" >&2
        exit 2
    }
done
rm -f "$PROGRESS" "$LOG" "$WATCHDOG_LOG"

critical_section=true
"$BIN" service-soak \
    --model "$MODEL" \
    --duration-seconds "$DURATION_SECONDS" \
    --concurrency "$CONCURRENCY" \
    --max-tokens "$MAX_TOKENS" \
    --prefill-chunk "$PREFILL_CHUNK" \
    --keepalive-ms "$KEEPALIVE_MS" \
    --responsiveness-ms "$RESPONSIVENESS_MS" \
    --max-rss-drift-percent "$MAX_RSS_DRIFT_PERCENT" \
    --checkpoint-interval-seconds "$CHECKPOINT_INTERVAL_SECONDS" \
    --progress "$PROGRESS" \
    --evidence "$EVIDENCE" \
    --label "continuous-service-soak-c${CONCURRENCY}" \
    >"$LOG" 2>&1 &
soak_pid=$!
launch_started=true
finish_critical_section
started_epoch=$(date +%s)

while soak_job_running; do
    sleep "$POLL_SECONDS"
    now_epoch=$(date +%s)
    heartbeat_epoch="$started_epoch"
    if [[ -f "$PROGRESS" ]]; then
        heartbeat_pid=$("$JQ" -er '.processID' "$PROGRESS" 2>/dev/null || true)
        if [[ "$heartbeat_pid" == "$soak_pid" ]]; then
            heartbeat_epoch=$(stat -f %m "$PROGRESS")
        fi
    fi
    if (( now_epoch - heartbeat_epoch > WATCHDOG_SECONDS )); then
        printf '{"status":"watchdog_timeout","pid":%d,"stale_seconds":%d}\n' \
            "$soak_pid" "$((now_epoch - heartbeat_epoch))" >"$WATCHDOG_LOG"
        terminate_soak
        trap - INT TERM EXIT
        rmdir "$progress_lock" 2>/dev/null || true
        lock_owned=false
        cat "$LOG"
        echo "soak watchdog FAILED: progress heartbeat stalled" >&2
        exit 1
    fi
done

set +e
wait "$soak_pid"
status=$?
set -e
trap - INT TERM EXIT
rmdir "$progress_lock" 2>/dev/null || true
lock_owned=false
cat "$LOG"
if [[ -f "$PROGRESS" ]]; then
    cat "$PROGRESS"
fi
exit "$status"
