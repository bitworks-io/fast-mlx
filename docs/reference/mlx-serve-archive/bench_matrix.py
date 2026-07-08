#!/usr/bin/env python3
"""
Temperature-matched, use-case decode-throughput benchmark across a model ladder,
with enhancement-assessment dimensions:
  --binary   : which mlx-serve to run (optimized perf/integration vs pre-opt baseline)
  --spec     : spec-decode default(on) vs off  -> isolates the spec-decode enhancement
  --include-ds4 : also benchmark DeepSeek-V4-Flash via the embedded ds4 engine (GGUF)

Each cell = streaming decode tok/s (median of repeats, warmup discarded), at the
temperature that use case + that model's vendor recommend (see TEMP). Spec-decode
default = production (PLD on, MTP auto). One model loaded at a time.
"""
import argparse, csv, json, os, signal, statistics, subprocess, time, urllib.request

MODELS_DIR = os.path.expanduser("~/perf-work/models")
DS4_GGUF = os.path.expanduser("~/.mlx-serve/models/antirez/deepseek-v4-gguf/"
                              "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf")
PORT = 11400
BASE = f"http://127.0.0.1:{PORT}"
BIN = os.path.expanduser("~/perf-work/mlx-serve/zig-out/bin/mlx-serve")  # overridden by --binary
SPEC = "default"  # or "off"

# (dir, family, label, tier, abspath|None)
MODELS = [
    ("Qwen3-32B-8bit",                          "qwen3",      "Qwen3-32B (8-bit, production)", "B1", None),
    ("Qwen3-32B-4bit",                          "qwen3",      "Qwen3-32B (4-bit)",             "B1", None),
    ("Qwen3-30B-A3B-Instruct-2507-4bit",        "qwen3_2507", "Qwen3-30B-A3B-2507 (4-bit)",   "A1", None),
    ("gemma-3-27b-it-4bit",                     "gemma",      "Gemma 3 27B (4-bit)",          "A4", None),
    ("Mistral-Small-3.2-24B-Instruct-2506-4bit","mistral",    "Mistral Small 24B (4-bit)",    "B3", None),
    ("Llama-3.3-70B-Instruct-4bit",             "llama",      "Llama 3.3 70B (4-bit)",        "B2", None),
    ("Qwen3.6-27B-4bit",                        "qwen36",     "Qwen3.6-27B (4-bit, MTP)",     "ref", None),
    ("gemma-4-26b-a4b-it-4bit",                 "gemma",      "Gemma 4 26B-A4B (4-bit, MoE)", "ref", None),
    ("Qwen3-8B-8bit",                           "qwen3",      "Qwen3-8B (8-bit)",             "small", None),
    ("gemma-4-e4b-it-4bit",                     "gemma",      "Gemma 4 E4B (4-bit)",          "small", None),
    ("Qwen3-4B-4bit",                           "qwen3",      "Qwen3-4B (4-bit)",             "small", None),
]
DS4_MODEL = ("DeepSeek-V4-Flash", "ds4", "DeepSeek-V4-Flash (IQ2, ds4)", "V4-Flash", DS4_GGUF)

TEMP = {
    "qwen3":      {"precise": (0.6, 0.95, 20), "creative": (0.7, 0.80, 20)},
    "qwen3_2507": {"precise": (0.7, 0.80, 20), "creative": (0.7, 0.80, 20)},
    "qwen36":     {"precise": (0.6, 0.95, 20), "creative": (1.0, 0.95, 20)},
    "gemma":      {"precise": (1.0, 0.95, 64), "creative": (1.0, 0.95, 64)},
    "mistral":    {"precise": (0.15, 1.0, None),"creative": (0.15, 1.0, None)},
    "llama":      {"precise": (0.2, 0.90, None),"creative": (0.6, 0.90, None)},
    "ds4":        {"precise": (0.3, 1.0, None), "creative": (1.0, 1.0, None)},
}

USE_CASES = {
    "coding":    ("precise",
        "Implement a thread-safe LRU cache in Python with get(key) and put(key, value) "
        "methods and a fixed capacity. Include docstrings and a short usage example.", False),
    "math":      ("precise",
        "A tank holds 500 liters. Pipe A fills it at 12 L/min. After A has run for 10 minutes, "
        "pipe B opens and drains at 7 L/min while A keeps filling. How many minutes after B opens "
        "does the tank become full? Show every step of your reasoning.", False),
    "reasoning": ("precise",
        "Five people (Alice, Bob, Carol, Dave, Eve) sit in a row of five chairs. Alice is not at "
        "either end. Bob sits immediately to the left of Carol. Dave is at one of the two ends. "
        "Eve is not seated next to Alice. Find a valid seating order and explain your reasoning "
        "step by step.", False),
    "tools":     ("precise",
        "Use the available tools to answer: what is the current weather in Tokyo and in Paris, "
        "and what is 15% of 2480? Call the tools as needed, then write a short paragraph "
        "summarizing what you found for the user.", True),
    "creative":  ("creative",
        "Write a short story of about 250 words about a lighthouse keeper who discovers a message "
        "in a bottle during a violent storm.", False),
}

TOOLS = [
    {"type": "function", "function": {"name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "calculate",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]
_TOOL_RESULTS = {"get_weather": '{"temp_c": 18, "condition": "cloudy", "wind_kph": 12}',
                 "calculate": '{"result": 372}'}


def apply_spec(body):
    if SPEC == "off":
        body["enable_pld"] = False
        body["enable_drafter"] = False
        body["enable_mtp"] = False
    return body


def _measure_stream(body):
    apply_spec(body)
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; n = 0; ptoks = ctoks = 0
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            ch = obj.get("choices") or []
            if ch and ch[0].get("delta", {}).get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n += 1
            if obj.get("usage"):
                ptoks = obj["usage"].get("prompt_tokens", 0)
                ctoks = obj["usage"].get("completion_tokens", 0)
    total = time.perf_counter() - t0
    if ttft is None:
        ttft = total
    if ctoks == 0:
        ctoks = n
    dt = max(total - ttft, 1e-6)
    return {"ttft": ttft, "decode_tps": (ctoks - 1) / dt if ctoks > 1 else 0.0,
            "prefill_tps": ptoks / ttft if ttft > 0 and ptoks else 0.0, "ptoks": ptoks, "ctoks": ctoks}


def one_request(model, prompt, temp, top_p, top_k, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temp, "top_p": top_p,
            "stream": True, "stream_options": {"include_usage": True}}
    if top_k is not None:
        body["top_k"] = top_k
    return _measure_stream(body)


def two_turn_tools(model, prompt, temp, top_p, top_k, max_tokens):
    base = {"model": model, "temperature": temp, "top_p": top_p, "max_tokens": max_tokens}
    if top_k is not None:
        base["top_k"] = top_k
    b1 = apply_spec(dict(base)); b1["messages"] = [{"role": "user", "content": prompt}]
    b1["tools"] = TOOLS; b1["tool_choice"] = "auto"; b1["stream"] = False
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(b1).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        msg = json.loads(resp.read())["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    if not tcs:
        b = dict(base); b["messages"] = [{"role": "user", "content": prompt}]
        b["tools"] = TOOLS; b["tool_choice"] = "auto"
        b["stream"] = True; b["stream_options"] = {"include_usage": True}
        return _measure_stream(b)
    messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs}]
    for tc in tcs:
        name = tc.get("function", {}).get("name", "")
        messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                         "content": _TOOL_RESULTS.get(name, '{"ok": true}')})
    b2 = dict(base); b2["messages"] = messages; b2["tools"] = TOOLS
    b2["stream"] = True; b2["stream_options"] = {"include_usage": True}
    return _measure_stream(b2)


def wait_health(timeout):
    for _ in range(timeout):
        try:
            urllib.request.urlopen(BASE + "/health", timeout=3).read()
            return True
        except Exception:
            time.sleep(1)
    return False


def boot(path, ds4=False):
    subprocess.run(["pkill", "-f", f"mlx-serve.*--port {PORT}"], capture_output=True)
    time.sleep(3)
    log = open(os.path.expanduser("~/perf-work/bench-matrix-server.log"), "w")
    cmd = [BIN, "--model", path, "--serve", "--port", str(PORT), "--log-level", "warn"]
    if ds4:
        cmd += ["--ctx-size", "8192"]
    p = subprocess.Popen(cmd, stdout=log, stderr=log, preexec_fn=os.setsid)
    return p, wait_health(600 if ds4 else 300)


def kill(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:
        pass
    subprocess.run(["pkill", "-f", f"mlx-serve.*--port {PORT}"], capture_output=True)
    time.sleep(3)


def main():
    global BIN, SPEC
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=BIN)
    ap.add_argument("--spec", choices=["default", "off"], default="default")
    ap.add_argument("--label", default="opt")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--only", default=None)
    ap.add_argument("--include-ds4", action="store_true")
    args = ap.parse_args()
    BIN = os.path.expanduser(args.binary); SPEC = args.spec
    only = args.only.split(",") if args.only else None

    models = list(MODELS)
    if args.include_ds4:
        models.append(DS4_MODEL)

    new = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["label", "spec", "tier", "model", "family", "use_case", "temp",
                        "top_p", "top_k", "decode_tps", "prefill_tps", "ttft_s",
                        "prompt_toks", "completion_toks"])
            f.flush()
        for mdir, fam, label, tier, abspath in models:
            if only and not any(s in mdir for s in only):
                continue
            path = abspath if abspath else os.path.join(MODELS_DIR, mdir)
            is_ds4 = fam == "ds4"
            exists = os.path.exists(path) if is_ds4 else os.path.exists(os.path.join(path, "config.json"))
            if not exists:
                print(f"[skip] {mdir} not present", flush=True); continue
            print(f"\n==== {label} [{args.label}/{SPEC}] ====", flush=True)
            p, ok = boot(path, ds4=is_ds4)
            if not ok:
                print(f"[FAIL boot] {mdir}", flush=True); kill(p); continue
            for uc, (wclass, prompt, uses_tools) in USE_CASES.items():
                temp, top_p, top_k = TEMP[fam][wclass]
                run = (lambda mt: two_turn_tools(mdir, prompt, temp, top_p, top_k, mt)) if uses_tools \
                    else (lambda mt: one_request(mdir, prompt, temp, top_p, top_k, mt))
                try:
                    run(min(args.max_tokens, 48))  # warmup
                except Exception as e:
                    print(f"  [warmup err {uc}] {e}", flush=True)
                samples = []
                for _ in range(args.repeats):
                    try:
                        samples.append(run(args.max_tokens))
                    except Exception as e:
                        print(f"  [err {uc}] {e}", flush=True)
                if not samples:
                    continue
                dt = statistics.median(s["decode_tps"] for s in samples)
                pf = statistics.median(s["prefill_tps"] for s in samples)
                tt = statistics.median(s["ttft"] for s in samples)
                ct = statistics.median(s["ctoks"] for s in samples)
                pt = statistics.median(s["ptoks"] for s in samples)
                print(f"  {uc:10s} T={temp:<4} decode={dt:7.1f} tok/s  ttft={tt*1000:5.0f}ms  ct={ct:.0f}", flush=True)
                w.writerow([args.label, SPEC, tier, label, fam, uc, temp, top_p,
                            top_k if top_k is not None else "", f"{dt:.1f}", f"{pf:.0f}",
                            f"{tt:.3f}", int(pt), int(ct)])
                f.flush()
            kill(p)
    print(f"\nRUN DONE [{args.label}/{SPEC}] -> {args.csv}", flush=True)


if __name__ == "__main__":
    main()
