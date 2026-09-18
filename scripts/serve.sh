#!/usr/bin/env bash
# fast-mlx: one-command OpenAI-compatible server for Apple Silicon.
#
# Why this wrapper exists: MLX's Metal kernels compile to a `default.metallib` that a plain
# `swift build` cannot produce (SwiftPM has no Metal build phase, and on macOS 26 Apple made the
# Metal compiler a separate Xcode component). fast-mlx ships a prebuilt metallib
# (spike/prebuilt/mlx.metallib, built from the pinned mlx-swift) and colocates it next to the
# built binary, so deployment "just works" with no Metal Toolchain download and nothing for the
# operator to configure. A pre-load fit-check sizes the model against this host: for a fit-checkable
# arch it derives the MLX memory/cache/reserved-KV limits (and caps the served context) from the
# sizer, and fails closed with a reason if the model can't fit (override with --force). For startup
# before sizing completes, the binary derives its own bootstrap memory/cache limits from the host
# probe when the operator omits --memory-limit-bytes/--cache-limit-bytes; a proven fit overrides
# them, and an unknown fit is refused before load. If the operator passes either flag explicitly,
# it is an operator-set budget that lowers the planning envelope, not a value this wrapper guesses.
# The minimal invocation is:
#
#   ./scripts/serve.sh --model-path /path/to/model-dir --model my-model
#
# To let fast-mlx auto-pick among several already-downloaded quants of a model, pass
# --quant-candidates /dir/4bit,/dir/8bit,... (comma-separated absolute local dirs) with --model for
# identity; the pre-load quant auto-pick loads the best fit for this host and refuses a red-only set.
#
# To auto-pick WITHOUT pre-downloading every quant, pass --quant-repos with a comma-separated list of
# HF repo ids (the different quants of one model), e.g.
#   ./scripts/serve.sh --model my-model --quant-repos mlx-community/M-4bit,mlx-community/M-8bit
# fast-mlx fetches only each candidate's sizing metadata (config + safetensors index, no weights),
# fit-checks them against this host, then downloads the full weights for the WINNER alone and serves it.
#
# Any explicit flag you pass overrides the corresponding default. Pass --context N to request a
# served context (capped to the host's ceiling), or --force to serve past a red fit-check verdict.
# Omit --max-completion-tokens to use the authenticated model + host-fit context maximum; pass it
# only to narrow that model-aware ceiling.
# Pass --plan-concurrency N to size the fit-check for N concurrent decode streams (a stricter,
# concurrency-aware verdict); the default sizes for a single stream.
# Set FASTMLX_API_KEY to require Bearer auth (mandatory when binding a non-loopback --host).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPIKE_DIR="$(cd "$SCRIPT_DIR/../spike" && pwd)"
CONFIG="${FASTMLX_CONFIG:-release}"

# Fill in sensible defaults so the operator only has to name a model.
ARGS=("$@")
have() { local f; for f in "${ARGS[@]}"; do [ "$f" = "$1" ] && return 0; done; return 1; }
# Returns (via stdout) the value following an exact-token flag, mirroring `have`'s exact-match
# scan (this codebase always requires "--flag value" as separate tokens — see validate_host_use's
# explicit rejection of "--flag=value" — so no combined form needs handling here). Prints nothing
# and returns 1 if the flag is absent or has no following token.
value_of() {
  local flag="$1" i
  for i in "${!ARGS[@]}"; do
    if [ "${ARGS[$i]}" = "$flag" ]; then
      if [ "$i" -lt "$(( ${#ARGS[@]} - 1 ))" ]; then
        printf '%s' "${ARGS[$((i + 1))]}"
        return 0
      fi
      return 1
    fi
  done
  return 1
}

# Match Swift's positive 64-bit `Int` admission without evaluating untrusted input in shell
# arithmetic (which can wrap). Leading zeroes do not change the represented value.
is_positive_swift_int() {
  local normalized="$1" max_int="9223372036854775807"
  [[ "$normalized" =~ ^[0-9]+$ ]] || return 1
  while [ "${normalized#0}" != "$normalized" ]; do normalized="${normalized#0}"; done
  [ -n "$normalized" ] || return 1
  [ "${#normalized}" -lt "${#max_int}" ] && return 0
  [ "${#normalized}" -gt "${#max_int}" ] && return 1
  [[ "$normalized" < "$max_int" || "$normalized" == "$max_int" ]]
}

validate_host_use() {
  local seen=0 host_use_value="" reserve_seen=0 reserve_value="" i value
  for i in "${!ARGS[@]}"; do
    case "${ARGS[$i]}" in
      --host-use)
        if [ "$seen" -ne 0 ]; then
          echo "[fast-mlx] error: --host-use may be provided only once." >&2
          exit 2
        fi
        seen=1
        if [ "$i" -eq "$(( ${#ARGS[@]} - 1 ))" ]; then
          echo "[fast-mlx] error: --host-use needs a value: 'shared' or 'dedicated-serving'." >&2
          exit 2
        fi
        value="${ARGS[$((i + 1))]}"
        if [[ "$value" == --* ]]; then
          echo "[fast-mlx] error: --host-use needs a value: 'shared' or 'dedicated-serving'." >&2
          exit 2
        fi
        case "$value" in
          shared|dedicated-serving) host_use_value="$value" ;;
          *)
            echo "[fast-mlx] error: --host-use must be 'shared' or 'dedicated-serving'." >&2
            exit 2
            ;;
        esac
        ;;
      --host-use=*)
        echo "[fast-mlx] error: --host-use must be passed as '--host-use shared' or '--host-use dedicated-serving'." >&2
        exit 2
        ;;
      --os-service-reserve-bytes)
        if [ "$reserve_seen" -ne 0 ]; then
          echo "[fast-mlx] error: --os-service-reserve-bytes may be provided only once." >&2
          exit 2
        fi
        reserve_seen=1
        if [ "$i" -eq "$(( ${#ARGS[@]} - 1 ))" ]; then
          echo "[fast-mlx] error: --os-service-reserve-bytes needs a positive integer value." >&2
          exit 2
        fi
        reserve_value="${ARGS[$((i + 1))]}"
        if ! is_positive_swift_int "$reserve_value"; then
          echo "[fast-mlx] error: --os-service-reserve-bytes needs a positive integer value." >&2
          exit 2
        fi
        ;;
      --os-service-reserve-bytes=*)
        echo "[fast-mlx] error: --os-service-reserve-bytes must be passed as a separate positive integer value." >&2
        exit 2
        ;;
    esac
  done

  if [ "$host_use_value" = "dedicated-serving" ] && [ "$reserve_seen" -eq 0 ]; then
    echo "[fast-mlx] error: --os-service-reserve-bytes is required with --host-use dedicated-serving." >&2
    exit 2
  fi
  if [ "$host_use_value" != "dedicated-serving" ] && [ "$reserve_seen" -ne 0 ]; then
    echo "[fast-mlx] error: --os-service-reserve-bytes requires --host-use dedicated-serving." >&2
    exit 2
  fi
}

validate_host_use

cd "$SPIKE_DIR"

echo "[fast-mlx] building fastmlx-serve ($CONFIG)…" >&2
swift build -c "$CONFIG" --product fastmlx-serve >&2
BIN_DIR="$SPIKE_DIR/.build/$CONFIG"
BIN="$BIN_DIR/fastmlx-serve"

# Colocate the MLX metallib next to the binary (MLX searches <bindir>/mlx.metallib first). Prefer a
# freshly Xcode-built metallib if the maintainer produced one; otherwise the shipped prebuilt.
FRESH="$SPIKE_DIR/.build/xcode/Build/Products/Release/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
SHIPPED="$SPIKE_DIR/prebuilt/mlx.metallib"
SRC=""
[ -f "$FRESH" ] && SRC="$FRESH"
[ -z "$SRC" ] && [ -f "$SHIPPED" ] && SRC="$SHIPPED"
if [ -z "$SRC" ]; then
  echo "[fast-mlx] ERROR: no metallib found (expected $SHIPPED)." >&2
  echo "[fast-mlx] Regenerate it with: scripts/build-metallib.sh" >&2
  exit 1
fi
cp -f "$SRC" "$BIN_DIR/mlx.metallib"

# Exact Qwen3.5 MTP is intentionally a local-artifact-only route. Refuse here, before any wrapper
# branch can enumerate Hugging Face metadata or download a checkpoint; the binary repeats the
# incompatibility checks as defense in depth, but it cannot undo network work already performed by
# this wrapper. `--model` may still accompany `--model-path` as the served identity.
if have --exact-qwen35-mtp; then
  if have --quant-repos || have --auto-quant || have --quant-candidates; then
    echo "[fast-mlx] error: --exact-qwen35-mtp cannot be combined with remote or quant model resolution; pass local --model-path and --mtp-drafter-path directories." >&2
    exit 2
  fi
  if ! have --model-path; then
    echo "[fast-mlx] error: --exact-qwen35-mtp requires an explicit local --model-path DIR; wrapper auto-fetch is disabled for this route." >&2
    exit 2
  fi
fi

# resolve_quant_and_serve <candidate-repos-csv> <model-name-or-empty>: the shared "sourcing half"
# body for both --quant-repos (explicit ids) and --auto-quant (ids enumerated from a base). Fetch ONLY
# each candidate's sizing metadata (config.json + *.safetensors.index.json — no weights), run the
# pre-load fit-check to pick the best-fitting quant for THIS host, then download the FULL weights for
# the WINNER alone and append its --model-path (+ --model when the operator gave none) to the global
# ARGS so it serves as an ordinary local load. A red-only / all-excluded / no-fit set exits non-zero
# (there is no --force in candidates mode). scripts/quant-prefetch.py owns the HF downloads and the
# machine-line parse so this wrapper never re-implements the HF naming or the machine contract.
resolve_quant_and_serve() {
  local qrepos="$1" qmodel="$2"
  local prefetch="$SCRIPT_DIR/quant-prefetch.py"
  echo "[fast-mlx] fetching sizing metadata (config + index only) for: $qrepos" >&2
  local cand_dirs=() cand_repos=() st repo dir
  while IFS=$'\t' read -r st repo dir; do
    if [ "$st" = "ok" ]; then cand_dirs+=("$dir"); cand_repos+=("$repo"); fi
    if [ "$st" = "skip" ]; then echo "[fast-mlx] skipped $repo: $dir" >&2; fi
  done < <(python3 "$prefetch" metadata --repos "$qrepos")
  if [ "${#cand_dirs[@]}" -eq 0 ]; then
    echo "[fast-mlx] no fetchable quant candidates for '$qrepos' (all excluded)." >&2
    exit 1
  fi

  # Dry-run the fit-checked pick over the metadata-only dirs (winner line on STDOUT, per-candidate
  # announce on STDERR — the operator sees the announce live while we capture just the machine line).
  local cand_csv; cand_csv="$(IFS=,; echo "${cand_dirs[*]}")"
  echo "[fast-mlx] fit-checking ${#cand_dirs[@]} candidate(s) against this host…" >&2
  local pick_out winner_dir
  if ! pick_out="$("$BIN" "${ARGS[@]}" --quant-candidates "$cand_csv" --quant-pick-only)"; then
    echo "[fast-mlx] no quant fits this host — the fit-check refused every candidate." >&2
    exit 1
  fi
  if ! winner_dir="$(printf '%s\n' "$pick_out" | python3 "$prefetch" parse-winner)"; then
    echo "[fast-mlx] could not parse the winning quant from the fit-check output." >&2
    exit 1
  fi

  # Map the winning directory back to its repo id (the metadata dir and the eventual full-download dir
  # are the same HF cache path, so a full snapshot_download fills in the winner's weights in place).
  local i winner_repo=""
  for i in "${!cand_dirs[@]}"; do
    if [ "${cand_dirs[$i]}" = "$winner_dir" ]; then winner_repo="${cand_repos[$i]}"; fi
  done
  if [ -z "$winner_repo" ]; then
    echo "[fast-mlx] internal: winner '$winner_dir' did not match any fetched candidate dir." >&2
    exit 1
  fi

  echo "[fast-mlx] winning quant: $winner_repo — downloading full weights…" >&2
  local full_dir
  if ! full_dir="$(python3 "$prefetch" full --repo "$winner_repo")"; then
    echo "[fast-mlx] full download of '$winner_repo' failed." >&2
    exit 1
  fi
  ARGS+=(--model-path "$full_dir")
  if [ -z "$qmodel" ]; then ARGS+=(--model "$winner_repo"); fi
  echo "[fast-mlx] serving $winner_repo from $full_dir" >&2
}

# --quant-repos <id1,id2,…>: the "sourcing half" of fit-checked-serve (differentiator #2). Name a
# model by its candidate quant repos and let this host pick the best-fitting one. We fetch ONLY the
# sizing metadata (config.json + model.safetensors.index.json — no multi-GB weights) for each
# candidate, run the pre-load fit-check to pick the best-fitting quant for THIS host, then download the
# FULL weights for the WINNER alone and serve it. --quant-repos is consumed here (the binary never
# sees it); the winner becomes an ordinary --model-path load. A red-only candidate set refuses (there
# is no --force in candidates mode). scripts/quant-prefetch.py owns the metadata/full HF downloads and
# the winner-line parse so this wrapper never re-implements the HF naming or the machine contract.
if have --quant-repos; then
  if have --model-path || have --quant-candidates; then
    echo "[fast-mlx] error: --quant-repos cannot be combined with --model-path or --quant-candidates." >&2
    exit 2
  fi
  QREPOS=""; QMODEL=""
  for i in "${!ARGS[@]}"; do
    if [ "${ARGS[$i]}" = "--quant-repos" ]; then QREPOS="${ARGS[$((i + 1))]}"; fi
    if [ "${ARGS[$i]}" = "--model" ]; then QMODEL="${ARGS[$((i + 1))]}"; fi
  done
  if [ -z "$QREPOS" ]; then
    echo "[fast-mlx] error: --quant-repos needs a comma-separated list of HF repo ids." >&2
    exit 2
  fi

  # Strip the consumed --quant-repos <value> so the binary never receives an unknown flag.
  NEWARGS=(); skip_next=0
  for a in "${ARGS[@]}"; do
    if [ "$skip_next" = "1" ]; then skip_next=0; continue; fi
    if [ "$a" = "--quant-repos" ]; then skip_next=1; continue; fi
    NEWARGS+=("$a")
  done
  ARGS=("${NEWARGS[@]}")

  echo "[fast-mlx] --quant-repos: resolving best-fitting quant for this host…" >&2
  resolve_quant_and_serve "$QREPOS" "$QMODEL"
fi

# --auto-quant <base>: the same sourcing half, but the candidate quant repos are ENUMERATED from a
# single base repo id instead of listed explicitly. The binary owns the offline name generation
# (QuantCandidateSourcer) and prints them on the `--auto-quant BASE --quant-pick-only` dry-run as
# `quant_enumerate base=.. candidates=..`; we parse that CSV (via quant-prefetch parse-enumerate) and
# feed it into the SAME metadata→pick→download→serve flow as --quant-repos. IMPORTANT: when the
# operator ALSO passes --quant-pick-only, they want the binary's dry-run enumerate, not a served load,
# so we DON'T intercept — the flag passes straight through to the binary below.
if have --auto-quant && ! have --quant-pick-only; then
  if have --model-path || have --quant-candidates || have --quant-repos; then
    echo "[fast-mlx] error: --auto-quant cannot be combined with --model-path, --quant-candidates, or --quant-repos." >&2
    exit 2
  fi
  ABASE=""; AMODEL=""
  for i in "${!ARGS[@]}"; do
    if [ "${ARGS[$i]}" = "--auto-quant" ]; then ABASE="${ARGS[$((i + 1))]}"; fi
    if [ "${ARGS[$i]}" = "--model" ]; then AMODEL="${ARGS[$((i + 1))]}"; fi
  done
  if [ -z "$ABASE" ]; then
    echo "[fast-mlx] error: --auto-quant needs a base HF repo id (e.g. mlx-community/Qwen3-8B)." >&2
    exit 2
  fi
  PREFETCH="$SCRIPT_DIR/quant-prefetch.py"

  # Strip the consumed --auto-quant <base> so the eventual served load never receives it (it is a
  # dry-run-only flag in the binary; the winner serves as a plain --model-path load).
  NEWARGS=(); skip_next=0
  for a in "${ARGS[@]}"; do
    if [ "$skip_next" = "1" ]; then skip_next=0; continue; fi
    if [ "$a" = "--auto-quant" ]; then skip_next=1; continue; fi
    NEWARGS+=("$a")
  done
  ARGS=("${NEWARGS[@]}")

  echo "[fast-mlx] --auto-quant: enumerating candidate quant repos for base '$ABASE'…" >&2
  if ! ENUM_OUT="$("$BIN" --auto-quant "$ABASE" --quant-pick-only)"; then
    echo "[fast-mlx] failed to enumerate quant candidates for '$ABASE'." >&2
    exit 1
  fi
  if ! ACANDS="$(printf '%s\n' "$ENUM_OUT" | python3 "$PREFETCH" parse-enumerate)"; then
    echo "[fast-mlx] could not parse the enumerated candidates for '$ABASE'." >&2
    exit 1
  fi
  echo "[fast-mlx] candidates: $ACANDS" >&2
  resolve_quant_and_serve "$ACANDS" "$AMODEL"
fi

# Resolve the model directory. An explicit --model-path (a local model dir) always wins; otherwise
# treat --model as a Hugging Face repo id (e.g. mlx-community/Qwen3-8B-4bit) and fetch it
# automatically — cached, so subsequent runs are instant. --quant-candidates supplies its own local
# dirs and the binary's pre-load quant auto-pick chooses among them, so skip the fetch branch there.
# A pass-through --auto-quant …--quant-pick-only dry-run (still carrying --auto-quant here, since only
# the intercept-to-serve branch strips it) needs no model dir — the binary just enumerates — so skip.
if ! have --scripted && ! have --model-path && ! have --quant-candidates && ! have --auto-quant; then
  MODEL_ID=""
  for i in "${!ARGS[@]}"; do
    if [ "${ARGS[$i]}" = "--model" ]; then MODEL_ID="${ARGS[$((i + 1))]}"; break; fi
  done
  if [ -z "$MODEL_ID" ]; then
    echo "[fast-mlx] error: pass --model-path DIR for a local model, or --model <hf-repo-id> to auto-fetch." >&2
    exit 2
  fi
  echo "[fast-mlx] no --model-path given; fetching '$MODEL_ID' from Hugging Face (cached after first run)…" >&2
  RESOLVED="$(python3 - "$MODEL_ID" <<'PY'
import sys
try:
    from huggingface_hub import snapshot_download
    print(snapshot_download(sys.argv[1]))
except ModuleNotFoundError:
    sys.stderr.write("huggingface_hub not installed\n"); sys.exit(3)
except Exception as exc:
    sys.stderr.write(f"fetch failed: {exc}\n"); sys.exit(1)
PY
)" || {
    rc=$?
    if [ "$rc" = "3" ]; then
      echo "[fast-mlx] auto-fetch needs Hugging Face: 'pip install -U huggingface_hub' (or 'brew install huggingface-cli'), or pass --model-path DIR." >&2
    else
      echo "[fast-mlx] could not fetch '$MODEL_ID' — check the repo id, or pass --model-path DIR." >&2
    fi
    exit 1
  }
  ARGS+=(--model-path "$RESOLVED")
  echo "[fast-mlx] model ready at $RESOLVED" >&2
fi

if ! have --scripted; then
  RAM="$(sysctl -n hw.memsize)"
  # --ngram-offload-plan is the scalar Flash Next (qwen4_exp) offload route. The binary's parser
  # refuses --ngram-offload-plan together with either continuous-batch route
  # (FastMLXServeArguments.swift, FastMLXServeArgumentError.ngramOffloadPlanWithContinuousBatch), so
  # injecting --continuous-batch-no-spec here would make the documented Flash Next launch impossible
  # to start.
  have --continuous-batch-no-spec || have --continuous-dynamic-pld || have --ngram-offload-plan || have --exact-qwen35-mtp || ARGS+=(--continuous-batch-no-spec)
  # --memory-limit-bytes and --cache-limit-bytes are deliberately left uninjected: an omitted flag
  # must reach the binary as absence, not as a wrapper guess, because the binary's pre-load fit-check
  # derives both from the host probe (Metal working set, host classification) that this wrapper
  # cannot see. If the operator passes either flag, it is an operator-set budget the binary honours
  # directly — this wrapper must not overwrite or duplicate it.
  #
  # --max-reserved-kv-bytes is different: it is genuinely applied at runtime (not superseded by the
  # sizer) and is deliberately generous, since a tighter figure could reject valid concurrent/long
  # requests. So the wrapper still injects a default here, unless the operator already passed one.
  # The only adjustment is a clamp to an explicit --memory-limit-bytes: the binary rejects
  # --max-reserved-kv-bytes > --memory-limit-bytes, and today's unclamped default already satisfies
  # that guard for every host that hasn't been artificially memory-constrained. The clamp exists only
  # so an operator who bounds memory to protect a co-tenant doesn't get a confusing parser rejection
  # naming a flag they never passed.
  #
  # --ngram-offload-plan is also excluded from this default: the binary rejects
  # --max-reserved-kv-bytes whenever continuous batch mode is not selected
  # (FastMLXServeArguments.swift, FastMLXServeArgumentError.optionRequiresContinuousBatchMode), and
  # the scalar offload route above deliberately never selects continuous mode. Injecting the default
  # here would turn the earlier continuous-batch-no-spec suppression into a different launch failure
  # for the same Flash Next command.
  if ! have --max-reserved-kv-bytes && ! have --exact-qwen35-mtp && ! have --ngram-offload-plan; then
    KV_DEFAULT="$(( RAM * 30 / 100 ))"
    if OPERATOR_MEMORY_LIMIT="$(value_of --memory-limit-bytes)" && [ "$OPERATOR_MEMORY_LIMIT" -lt "$KV_DEFAULT" ] 2>/dev/null; then
      KV_DEFAULT="$OPERATOR_MEMORY_LIMIT"
    fi
    ARGS+=(--max-reserved-kv-bytes "$KV_DEFAULT")
  fi
fi

echo "[fast-mlx] starting: fastmlx-serve ${ARGS[*]}" >&2
exec "$BIN" "${ARGS[@]}"
