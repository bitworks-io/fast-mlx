#!/usr/bin/env bash
# Head-to-head: Swift spike decode bench vs the Zig mlx-serve engine, same box, same
# model, same prompt, temp=0, max-tokens=256. Run on llmbench@192.168.1.252.
set -euo pipefail

MODEL="${1:?usage: run_spike_comparison.sh <model_path> [prompt]}"
PROMPT="${2:-Explain how continuous batching improves LLM serving throughput.}"
SPIKE_BIN="${SPIKE_BIN:-$(find "$HOME/Library/Developer/Xcode/DerivedData" -name spike-cli -path "*Release*" -type f 2>/dev/null | head -1)}"
ZIG_PORT="${ZIG_PORT:-11299}"
ARCHIVED_ZIG_TOK_S="151.8" # bench-matrix-2026-07-06.csv, this model, this box

if [[ -z "$SPIKE_BIN" ]]; then
    echo "Could not find a Release spike-cli binary; build with:" >&2
    echo "  xcodebuild -scheme spike-cli -destination 'platform=macOS' -configuration Release -skipPackagePluginValidation build" >&2
    exit 1
fi

echo "== Zig mlx-serve (same-session) =="
nohup "$HOME/mlx-serve-macos-arm64/mlx-serve" --model "$MODEL" --serve --port "$ZIG_PORT" \
    > /tmp/mlx-serve-comparison.log 2>&1 &
ZIG_PID=$!
trap 'kill "$ZIG_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 30); do
    curl -s -m 2 "http://127.0.0.1:${ZIG_PORT}/v1/models" > /dev/null 2>&1 && break
    sleep 1
done

ZIG_TOK_S=$(python3 - "$ZIG_PORT" "$MODEL" "$PROMPT" <<'PY'
import json, sys, time, urllib.request
port, model, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
model_name = model.rstrip("/").split("/")[-1]
url = f"http://127.0.0.1:{port}/v1/chat/completions"

def run(nonce, max_tokens=256):
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": f"[run-{nonce}] {prompt}"}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                  headers={"content-type": "application/json"})
    token_times = []
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            delta = json.loads(payload).get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                token_times.append(time.monotonic())
    if len(token_times) < 2:
        return None
    span = token_times[-1] - token_times[0]
    return (len(token_times) - 1) / span if span > 0 else None

rates = []
for i in range(4):  # run 0 = warmup, dropped
    r = run(i)
    if i > 0 and r:
        rates.append(r)
print(f"{sum(rates)/len(rates):.2f}" if rates else "NaN")
PY
)
echo "zig_decode_tok_s_same_session=${ZIG_TOK_S}"
echo "zig_decode_tok_s_archived=${ARCHIVED_ZIG_TOK_S}"

kill "$ZIG_PID" 2>/dev/null || true
trap - EXIT
sleep 1

echo "== Swift spike =="
SWIFT_LINE=$("$SPIKE_BIN" bench --model "$MODEL" --prompt "$PROMPT" --max-tokens 256 --runs 3 | tail -1)
SWIFT_TOK_S=$(echo "$SWIFT_LINE" | cut -d',' -f5)
echo "swift_decode_tok_s=${SWIFT_TOK_S}"

python3 - "$ZIG_TOK_S" "$SWIFT_TOK_S" "$ARCHIVED_ZIG_TOK_S" <<'PY'
import sys
try:
    z_session = float(sys.argv[1]); s = float(sys.argv[2]); z_archived = float(sys.argv[3])
    print(f"delta vs same-session zig = {100*(s-z_session)/z_session:+.1f}%")
    print(f"delta vs archived zig     = {100*(s-z_archived)/z_archived:+.1f}%")
except ValueError:
    print("record numbers manually")
PY
