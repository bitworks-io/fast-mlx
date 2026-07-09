#!/usr/bin/env python3
"""mlx-lm reference driver for the fastmlx harness (extends the spike's reference_tokens.py).

Greedy (temp=0) first-N token ids, plus — with --logits-out — the full-vocab RAW LOGITS at
every generated position.

CONTRACT (must match EngineDriver.logprobs in HarnessCore exactly, or KL is silently
meaningless): each position's vector is the raw, un-softmaxed logits over the FULL vocab in
token-id order (index == token id). NOT top-k. Written as raw little-endian float32, row-major
[positions x vocab], to --logits-out; the small JSON header goes to stdout (last line):

    {"tokens": [...], "positions": P, "vocab": V}

Position semantics: position k is the next-token distribution that PRODUCED generated token k
(context = prompt + generated tokens 0..k-1, each side following its own greedy path).
A terminal eos token is INCLUDED in tokens (and its producing position's logits are included),
then generation stops — the Swift SwiftEngineDriver mirrors this exactly.

Prompts are passed as token ids (--tokens-json) by the harness so tokenizer differences can't
desynchronize the comparison; --prompt (text) is kept for manual use.

Usage:
    harness_reference.py --model PATH (--tokens-json '[1,2,3]' | --prompt TEXT) --n N \
        [--eos-id ID] [--logits-out FILE.f32]
"""
import argparse
import json

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tokens-json", help="prompt as a JSON array of token ids")
    src.add_argument("--prompt", help="prompt as text (tokenized by mlx-lm; manual use only)")
    ap.add_argument("--n", type=int, default=40, help="max generated positions")
    ap.add_argument("--eos-id", type=int, default=None, help="stop AFTER emitting this token id")
    ap.add_argument("--logits-out", default=None,
                    help="write raw little-endian float32 [positions x vocab] logits here")
    args = ap.parse_args()

    model, tok = load(args.model)
    ids = json.loads(args.tokens_json) if args.tokens_json else tok.encode(args.prompt)
    cache = make_prompt_cache(model)

    out_tokens: list[int] = []
    rows: list[np.ndarray] | None = [] if args.logits_out else None
    vocab = None
    y = mx.array(ids)[None]
    for _ in range(args.n):
        logits = model(y, cache=cache)[:, -1, :]  # [1, vocab] raw logits
        vocab = int(logits.shape[-1])
        if rows is not None:
            # fp16 -> float32 is exact; np conversion of an evaluated mx array is a plain copy.
            rows.append(np.array(logits.astype(mx.float32)).reshape(-1))
        t = int(mx.argmax(logits, axis=-1).item())
        out_tokens.append(t)
        if args.eos_id is not None and t == args.eos_id:
            break
        y = mx.array([[t]])

    if rows is not None:
        np.stack(rows).astype("<f4").tofile(args.logits_out)
    print(json.dumps({"tokens": out_tokens, "positions": len(out_tokens), "vocab": vocab}))


if __name__ == "__main__":
    main()
