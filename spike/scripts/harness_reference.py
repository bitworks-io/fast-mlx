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

TEACHER-FORCED mode (--force-tokens '[...]'): instead of following its own greedy path, the
model scores the given continuation — position k's logits are computed with context =
prompt + force[0..k-1], then force[k] is fed as the next input REGARDLESS of argmax; eos does
not stop the loop and --n/--eos-id are ignored (the continuation already encodes its stop).
Exactly len(force) positions are emitted and tokens echoes the forced ids. Two drivers scoring
the SAME forced continuation therefore score IDENTICAL contexts at every position — the
context-locked basis the KL/perplexity metrics are defined on.

SAMPLED mode (--sample-positions '[...]', only valid with --force-tokens): the forward loop still
runs over the FULL forced continuation (causal decoding needs every intermediate token as
context regardless), but only positions in the given (ascending) list are converted to numpy and
written out — for a long-context entry with thousands of forced positions, materializing a
full-vocab row at every one of them would exhaust memory (~0.6MB/row x thousands x 2 drivers).
`positions` in the emitted header is then len(sample-positions), not len(force-tokens).

Prompts are passed as token ids (--tokens-json) by the harness so tokenizer differences can't
desynchronize the comparison; --prompt (text) is kept for manual use.

Usage:
    harness_reference.py --model PATH (--tokens-json '[1,2,3]' | --prompt TEXT) --n N \
        [--eos-id ID] [--logits-out FILE.f32] [--force-tokens '[4,5,6]'] \
        [--sample-positions '[0,3,5]']
"""
import argparse
import importlib.metadata
import json

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def _version(pkg: str) -> str:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


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
    ap.add_argument("--force-tokens", default=None,
                    help="teacher-force this JSON array of token ids instead of greedy decoding")
    ap.add_argument("--sample-positions", default=None,
                    help="only materialize+emit rows at these (ascending) indices into "
                         "--force-tokens; the forward loop still runs the full continuation")
    args = ap.parse_args()
    if args.sample_positions and not args.force_tokens:
        ap.error("--sample-positions requires --force-tokens")

    # Bound the allocator's buffer cache. The default limit tracks the (raised) GPU memory
    # limit, so unreusable transients can hoard tens of GB before anything evicts — this
    # process runs co-resident with the Swift harness on the same box, and their combined
    # ballooning was the harness's ~7K-context jetsam SIGKILL. Mirrors the Swift side's bound.
    mx.set_cache_limit(8 << 30)

    model, tok = load(args.model)
    ids = json.loads(args.tokens_json) if args.tokens_json else tok.encode(args.prompt)
    force = json.loads(args.force_tokens) if args.force_tokens else None
    sample_positions = set(json.loads(args.sample_positions)) if args.sample_positions else None
    cache = make_prompt_cache(model)

    out_tokens: list[int] = []
    rows: list[np.ndarray] | None = [] if args.logits_out else None
    vocab = None
    y = mx.array(ids)[None]
    if force is not None:
        # CHUNKED teacher-forced scoring (mirrors SwiftEngineDriver.scoreForced): the row for
        # forced position k is the model output at input index len(ids)-1+k over the full input
        # ids + force[:-1], so multi-token chunk forwards produce every row directly. Single-token
        # stepping made each step's transients slightly larger than the last (growing K/V
        # slices), so the buffer cache could never reuse a freed buffer and grew O(context^2) —
        # the other half of the ~7K jetsam ceiling. Chunks keep transients same-shaped (reused),
        # and score at prefill speed. 512 matches mlx-lm's default prefill step size.
        CHUNK = 512
        p = len(ids)
        full = list(ids) + [int(t) for t in force[:-1]]
        for a in range(0, len(full), CHUNK):
            b = min(a + CHUNK, len(full))
            logits = model(mx.array(full[a:b])[None], cache=cache)  # [1, b-a, vocab] raw logits
            mx.eval(logits)  # bound the lazy graph chunk by chunk
            vocab = int(logits.shape[-1])
            if rows is not None:
                for li in range(a, b):
                    pos = li - (p - 1)
                    if 0 <= pos < len(force) and (sample_positions is None or pos in sample_positions):
                        rows.append(np.array(logits[:, li - a, :].astype(mx.float32)).reshape(-1))
        out_tokens = [int(t) for t in force]  # echo the FORCED ids: they are the context-builders
    else:
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
    # "positions" gates the expected logits byte count on the Swift side: it must be len(rows)
    # (the number of rows actually written), which differs from len(out_tokens) under
    # --sample-positions (the loop still runs the full continuation but only some rows are kept).
    positions = len(rows) if rows is not None else len(out_tokens)
    # mlx/mlx-lm versions (Task 5 provenance): the Swift side reads these off the header so a
    # result record names the EXACT reference-side versions that produced it, not an assumption.
    print(json.dumps({
        "tokens": out_tokens, "positions": positions, "vocab": vocab,
        "mlx_version": _version("mlx"), "mlx_lm_version": _version("mlx-lm"),
    }))


if __name__ == "__main__":
    main()
