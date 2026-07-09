#!/usr/bin/env python3
"""Greedy first-N token ids from mlx-lm (the equivalence reference for the Swift spike).

Usage: reference_tokens.py <model_path> <prompt> <n>
Prints a JSON array of N greedy (temp=0) token ids to stdout.
"""
import sys
import json

from mlx_lm import load
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache


def main() -> None:
    model_path, prompt, n_str = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(n_str)

    model, tok = load(model_path)
    ids = tok.encode(prompt)
    cache = make_prompt_cache(model)

    out = []
    y = mx.array(ids)[None]
    for _ in range(n):
        logits = model(y, cache=cache)[:, -1, :]
        t = int(mx.argmax(logits, axis=-1).item())
        out.append(t)
        y = mx.array([[t]])

    print(json.dumps(out))


if __name__ == "__main__":
    main()
