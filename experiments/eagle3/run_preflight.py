#!/usr/bin/env python3
"""Qwen3-32B EAGLE-3 exactness and Apple-Silicon economics preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# A typo in a model path must fail locally, never become a repository download.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from eagle_head_mlx import Eagle3Head
from inspect_checkpoint import (
    PINNED_REVISION,
    inspect,
)
from preflight_core import (
    AcceptanceCounters,
    PreflightValidationError,
    RoundTiming,
    accept_greedy,
    authenticate_file_manifest,
    classify_cache_drift,
    committed_accepted_drafts,
    read_harness_git_sha,
    stable_evidence_id,
    trim_emission,
)


WORKLOADS = (
    {
        "name": "code",
        "prompt": (
            "Write a Python function reverse_linked_list(head) that reverses a singly linked "
            "list in place. Include type hints, an iterative implementation, and three tests."
        ),
    },
    {
        "name": "math_reasoning",
        "prompt": (
            "A shop discounts a $240 jacket by 25%, then applies an 8% sales tax to the "
            "discounted price. Explain each step and give the final price."
        ),
    },
    {
        "name": "low_repetition_prose",
        "prompt": (
            "Explain how coastal fog forms and why nearby valleys can remain sunny. Use one "
            "concise paragraph for a general audience."
        ),
    },
)


def _eval_cache(cache) -> None:
    mx.eval([entry.state for entry in cache])


def forward_target_with_taps(model, inputs, cache, tap_layers=(2, 32, 61)):
    inner = model.model
    hidden = inner.embed_tokens(inputs)
    mask = create_attention_mask(hidden, cache[0])
    captured = {}
    for index, (layer, layer_cache) in enumerate(zip(inner.layers, cache)):
        if index in tap_layers:
            captured[index] = hidden
        hidden = layer(hidden, mask, layer_cache)
    hidden = inner.norm(hidden)
    if hasattr(model, "lm_head") and model.lm_head is not None:
        logits = model.lm_head(hidden)
    else:
        logits = inner.embed_tokens.as_linear(hidden)
    missing = [index for index in tap_layers if index not in captured]
    if missing:
        raise PreflightValidationError(f"target did not expose tap layers: {missing}")
    return logits, mx.concatenate([captured[index] for index in tap_layers], axis=-1)


def _normalize_token_ids(value) -> List[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


def encode_prompt(tokenizer, prompt: str, enable_thinking: bool) -> List[int]:
    messages = [{"role": "user", "content": prompt}]
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except (AttributeError, TypeError):
        encoded = tokenizer.encode(prompt)
    tokens = _normalize_token_ids(encoded)
    if not tokens:
        raise PreflightValidationError("prompt tokenized to an empty sequence")
    return tokens


def eos_token_ids(tokenizer) -> set:
    result = set()
    for attribute in ("eos_token_ids", "eos_token_id"):
        value = getattr(tokenizer, attribute, None)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            result.update(int(token) for token in value)
        else:
            result.add(int(value))
    return result


@dataclass
class GenerationResult:
    tokens: List[int]
    text: str
    prompt_tokens: int
    prefill_seconds: float
    decode_seconds: float
    peak_memory_bytes: int
    counters: AcceptanceCounters
    phase_timing: Optional[RoundTiming] = None
    round_trace: Optional[List[dict]] = None

    @property
    def decode_token_count(self) -> int:
        return max(0, len(self.tokens) - 1)

    @property
    def tokens_per_second(self) -> Optional[float]:
        if self.decode_token_count == 0 or self.decode_seconds <= 0:
            return None
        return self.decode_token_count / self.decode_seconds

    def summary(self) -> dict:
        result = {
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": len(self.tokens),
            "decode_tokens": self.decode_token_count,
            "prefill_seconds": self.prefill_seconds,
            "decode_seconds": self.decode_seconds,
            "tokens_per_second": self.tokens_per_second,
            "peak_memory_bytes": self.peak_memory_bytes,
            "proposed_draft_tokens": self.counters.proposed,
            "accepted_draft_tokens": self.counters.accepted,
            "verify_rounds": self.counters.verify_rounds,
            "proposal_acceptance_rate": self.counters.proposal_acceptance_rate,
            "accepted_drafts_per_round": self.counters.accepted_drafts_per_round,
            "inclusive_acceptance_length": self.counters.inclusive_acceptance_length,
        }
        if self.phase_timing:
            result["mean_phase_seconds"] = {
                "draft": self.phase_timing.draft_seconds,
                "target_verify": self.phase_timing.verify_seconds,
                "commit_or_rollback": self.phase_timing.commit_seconds,
                "total": self.phase_timing.total_seconds,
            }
        return result


def baseline_generate(model, tokenizer, prompt_tokens: Sequence[int], max_tokens: int):
    target_cache = make_prompt_cache(model)
    inputs = mx.array(prompt_tokens)[None]
    mx.reset_peak_memory()
    prefill_start = time.perf_counter()
    logits = model(inputs, cache=target_cache)
    mx.eval(logits)
    _eval_cache(target_cache)
    first = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    prefill_seconds = time.perf_counter() - prefill_start

    generated = [first]
    eos_ids = eos_token_ids(tokenizer)
    decode_start = time.perf_counter()
    while len(generated) < max_tokens and generated[-1] not in eos_ids:
        logits = model(mx.array([[generated[-1]]]), cache=target_cache)
        mx.eval(logits)
        _eval_cache(target_cache)
        generated.append(int(mx.argmax(logits[:, -1, :], axis=-1).item()))
    decode_seconds = time.perf_counter() - decode_start
    return GenerationResult(
        tokens=generated,
        text=tokenizer.decode(generated),
        prompt_tokens=len(prompt_tokens),
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        peak_memory_bytes=int(mx.get_peak_memory()),
        counters=AcceptanceCounters(proposed=0, accepted=0, verify_rounds=0),
    )


def diagnose_cache_drift(
    model,
    prompt_tokens: Sequence[int],
    base_tokens: Sequence[int],
    round_trace: Sequence[dict],
    mismatch_index: int,
) -> dict:
    """Replay a mismatch with full, retained-only, and sequential target-cache histories."""
    target_rows = [
        row for row in round_trace
        if row["generated_start_index"] == mismatch_index
    ]
    if mismatch_index <= 0 or len(target_rows) != 1:
        return {
            "status": "not-applicable",
            "reason": "first mismatch is not the first emission of exactly one verify round",
        }
    target_row = target_rows[0]

    def new_cache():
        cache = make_prompt_cache(model)
        logits = model(mx.array(prompt_tokens)[None], cache=cache)
        mx.eval(logits)
        _eval_cache(cache)
        return cache, int(mx.argmax(logits[0, -1]).item())

    full_cache, first = new_cache()
    retained_cache, retained_first = new_cache()
    sequential_batched_cache, sequential_batched_first = new_cache()
    sequential_one_cache, sequential_one_first = new_cache()
    if not (
        first
        == retained_first
        == sequential_batched_first
        == sequential_one_first
        == base_tokens[0]
    ):
        return {
            "status": "inconclusive",
            "reason": "fresh target prefills did not reproduce the base first token",
        }

    generated = [first]
    for row in round_trace:
        if row["generated_start_index"] >= mismatch_index:
            break
        if len(generated) != row["generated_start_index"]:
            return {"status": "inconclusive", "reason": "round trace is not contiguous"}

        verify_input = [generated[-1], *row["draft_tokens"]]
        logits = model(mx.array([verify_input]), cache=full_cache)
        mx.eval(logits)
        _eval_cache(full_cache)
        trimmed = trim_prompt_cache(full_cache, row["rejected"])
        if trimmed != row["rejected"]:
            return {"status": "inconclusive", "reason": "full replay rollback failed"}

        retained_input = [
            generated[-1],
            *row["draft_tokens"][: row["accepted"]],
        ]
        logits = model(mx.array([retained_input]), cache=retained_cache)
        mx.eval(logits)
        _eval_cache(retained_cache)

        for token in retained_input:
            logits = model(mx.array([[token]]), cache=sequential_batched_cache)
            mx.eval(logits)
            _eval_cache(sequential_batched_cache)
            logits = model(mx.array([[token]]), cache=sequential_one_cache)
            mx.eval(logits)
            _eval_cache(sequential_one_cache)

        generated.extend(row["emitted_before_stopping_rules"])
        if generated != list(base_tokens[: len(generated)]):
            return {
                "status": "inconclusive",
                "reason": "trace diverged before the reported first mismatch",
            }

    probe = [generated[-1], *target_row["draft_tokens"]]
    offsets = {
        "full_verify_and_trim": full_cache[0].offset,
        "retained_batches": retained_cache[0].offset,
        "all_sequential_batched_probe": sequential_batched_cache[0].offset,
        "all_sequential_one_token": sequential_one_cache[0].offset,
    }
    argmax = {}
    for name, cache in (
        ("full_verify_and_trim", full_cache),
        ("retained_batches", retained_cache),
        ("all_sequential_batched_probe", sequential_batched_cache),
    ):
        logits = model(mx.array([probe]), cache=cache)
        mx.eval(logits)
        _eval_cache(cache)
        argmax[name] = [int(token) for token in mx.argmax(logits[0], axis=-1).tolist()]
    logits = model(mx.array([[generated[-1]]]), cache=sequential_one_cache)
    mx.eval(logits)
    _eval_cache(sequential_one_cache)
    argmax["all_sequential_one_token"] = [
        int(mx.argmax(logits[0, -1], axis=-1).item())]

    expected = int(base_tokens[mismatch_index])
    classification = classify_cache_drift(
        expected_token=expected,
        full_verify_token=argmax["full_verify_and_trim"][0],
        retained_batch_token=argmax["retained_batches"][0],
        sequential_batched_token=argmax["all_sequential_batched_probe"][0],
        sequential_one_token=argmax["all_sequential_one_token"][0],
    )
    return {
        "status": "classified" if classification != "unclassified" else "inconclusive",
        "classification": classification,
        "mismatch_index": mismatch_index,
        "expected_token": expected,
        "probe_tokens": probe,
        "cache_offsets_before_probe": offsets,
        "argmax_by_cache_history": argmax,
    }


def eagle_generate(
    model,
    head: Eagle3Head,
    tokenizer,
    prompt_tokens: Sequence[int],
    max_tokens: int,
    num_draft: int,
    synchronize_phases: bool,
    capture_trace: bool = False,
):
    if num_draft <= 0 or num_draft > head.spec.max_draft:
        raise PreflightValidationError(
            f"num_draft must be in 1...{head.spec.max_draft}, got {num_draft}")
    target_cache = make_prompt_cache(model)
    draft_cache = head.new_cache()
    inputs = mx.array(prompt_tokens)[None]
    mx.reset_peak_memory()

    prefill_start = time.perf_counter()
    logits, target_hidden = forward_target_with_taps(model, inputs, target_cache)
    _, _ = head(inputs, target_hidden, cache=draft_cache)
    mx.eval(logits, target_hidden, draft_cache.state)
    _eval_cache(target_cache)
    first = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    prefill_seconds = time.perf_counter() - prefill_start

    generated = [first]
    last_target_hidden = target_hidden[:, -1:, :]
    eos_ids = eos_token_ids(tokenizer)
    proposed = 0
    accepted_total = 0
    verify_rounds = 0
    draft_seconds = 0.0
    verify_seconds = 0.0
    commit_seconds = 0.0
    round_trace = [] if capture_trace else None
    decode_start = time.perf_counter()

    while len(generated) < max_tokens and generated[-1] not in eos_ids:
        remaining = max_tokens - len(generated)
        if remaining == 1:
            tail_logits, _ = forward_target_with_taps(
                model, mx.array([[generated[-1]]]), target_cache)
            mx.eval(tail_logits)
            _eval_cache(target_cache)
            generated.append(int(mx.argmax(tail_logits[:, -1, :], axis=-1).item()))
            break

        round_draft = min(num_draft, remaining - 1)
        draft_start = time.perf_counter()
        current = mx.array([[generated[-1]]])
        draft_logits, draft_hidden = head(
            current, last_target_hidden, cache=draft_cache)
        draft_arrays = []
        for draft_index in range(round_draft):
            draft_vocab = mx.argmax(draft_logits[:, -1:, :], axis=-1)
            draft_target = head.map_draft_to_target(draft_vocab)
            draft_arrays.append(draft_target)
            if draft_index + 1 < round_draft:
                draft_logits, draft_hidden = head.forward_fused(
                    draft_target,
                    draft_hidden[:, -1:, :],
                    cache=draft_cache,
                )
        # Cache completeness: consume the final proposal so target and draft caches can roll
        # back the same rejected suffix after verification.
        _, _ = head.forward_fused(
            draft_arrays[-1],
            draft_hidden[:, -1:, :],
            cache=draft_cache,
        )
        if synchronize_phases:
            mx.eval(draft_arrays, draft_cache.state)
            draft_seconds += time.perf_counter() - draft_start

        verify_start = time.perf_counter()
        verify_input = mx.concatenate([current] + draft_arrays, axis=1)
        verify_logits, verify_hidden = forward_target_with_taps(
            model, verify_input, target_cache)
        if synchronize_phases:
            mx.eval(verify_logits, verify_hidden)
            _eval_cache(target_cache)
            verify_seconds += time.perf_counter() - verify_start
        else:
            mx.eval(verify_logits, verify_hidden, draft_arrays, draft_cache.state)
            _eval_cache(target_cache)

        commit_start = time.perf_counter()
        draft_tokens = [int(array.item()) for array in draft_arrays]
        target_tokens = [int(token) for token in mx.argmax(verify_logits[0], axis=-1).tolist()]
        acceptance = accept_greedy(draft_tokens, target_tokens)
        rejected = round_draft - acceptance.accepted
        if round_trace is not None:
            round_trace.append(
                {
                    "generated_start_index": len(generated),
                    "draft_tokens": draft_tokens,
                    "target_verify_argmax": target_tokens,
                    "accepted": acceptance.accepted,
                    "emitted_before_stopping_rules": list(acceptance.emitted),
                    "rejected": rejected,
                })
        target_trimmed = trim_prompt_cache(target_cache, rejected)
        draft_trimmed = draft_cache.trim(rejected)
        if target_trimmed != rejected or draft_trimmed != rejected:
            raise PreflightValidationError(
                "target/draft rollback did not remove the rejected suffix: "
                f"expected={rejected}, target={target_trimmed}, draft={draft_trimmed}")
        if synchronize_phases:
            _eval_cache(target_cache)
            mx.eval(draft_cache.state)

        emitted, done = trim_emission(
            acceptance.emitted,
            already_emitted=len(generated),
            max_tokens=max_tokens,
            eos_ids=eos_ids,
        )
        generated.extend(emitted)
        proposed += round_draft
        accepted_total += committed_accepted_drafts(
            accepted=acceptance.accepted,
            emitted_count=len(emitted),
        )
        verify_rounds += 1
        last_target_hidden = verify_hidden[
            :, acceptance.accepted : acceptance.accepted + 1, :]
        if synchronize_phases:
            mx.eval(last_target_hidden)
            commit_seconds += time.perf_counter() - commit_start
        if done:
            break

    decode_seconds = time.perf_counter() - decode_start
    counters = AcceptanceCounters(
        proposed=proposed,
        accepted=accepted_total,
        verify_rounds=verify_rounds,
    )
    phase_timing = None
    if synchronize_phases and verify_rounds:
        phase_timing = RoundTiming(
            draft_seconds=draft_seconds / verify_rounds,
            verify_seconds=verify_seconds / verify_rounds,
            commit_seconds=commit_seconds / verify_rounds,
        )
    return GenerationResult(
        tokens=generated,
        text=tokenizer.decode(generated),
        prompt_tokens=len(prompt_tokens),
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        peak_memory_bytes=int(mx.get_peak_memory()),
        counters=counters,
        phase_timing=phase_timing,
        round_trace=round_trace,
    )


def token_hash(tokens: Sequence[int]) -> str:
    encoded = json.dumps(list(tokens), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def exactness(base: GenerationResult, speculative: GenerationResult) -> dict:
    mismatch = None
    for index, pair in enumerate(zip(base.tokens, speculative.tokens)):
        if pair[0] != pair[1]:
            mismatch = index
            break
    if mismatch is None and len(base.tokens) != len(speculative.tokens):
        mismatch = min(len(base.tokens), len(speculative.tokens))
    token_identical = base.tokens == speculative.tokens
    base_bytes = base.text.encode("utf-8")
    speculative_bytes = speculative.text.encode("utf-8")
    return {
        "token_ids_identical": token_identical,
        "bytes_identical": base_bytes == speculative_bytes,
        "first_mismatch_index": mismatch,
        "base_token_hash": token_hash(base.tokens),
        "speculative_token_hash": token_hash(speculative.tokens),
        "base_bytes_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "speculative_bytes_sha256": hashlib.sha256(speculative_bytes).hexdigest(),
    }


def _metadata_revision(model_directory: Path) -> str:
    metadata = (
        model_directory / ".cache/huggingface/download/config.json.metadata")
    try:
        revision = metadata.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError) as error:
        raise PreflightValidationError(f"missing target revision metadata: {metadata}") from error
    if not revision:
        raise PreflightValidationError(f"empty target revision metadata: {metadata}")
    return revision


def _sysctl(name: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", name], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance(
    head_directory: Path,
    harness_git_sha: str,
    target_identity: dict,
    head_identity: dict,
) -> dict:
    return {
        "date": datetime.now(timezone.utc).isoformat(),
        "harness_git_sha": harness_git_sha,
        "hardware": {
            "hostname": platform.node(),
            "chip": _sysctl("machdep.cpu.brand_string"),
            "os": platform.platform(),
            "wired_limit_mb": _sysctl("iogpu.wired_limit_mb"),
        },
        "runtime": {
            "python": platform.python_version(),
            "mlx": version("mlx"),
            "mlx_lm": version("mlx-lm"),
            "transformers": version("transformers"),
            "cache_limit_bytes": 8 << 30,
        },
        "target": target_identity,
        "draft": {
            "repository": head_identity["repository"],
            "revision": head_identity["revision"],
            "blob_id": head_identity["model_blob_id"],
            "model_sha256": head_identity["model_sha256"],
            "config_sha256": head_identity["config_sha256"],
            "directory_name": head_directory.name,
        },
    }


def authenticate_target(model_directory: Path) -> dict:
    pins_path = Path(__file__).with_name("pins.json")
    try:
        pins = json.loads(pins_path.read_text(encoding="utf-8"))["targets"]
        pin = pins[model_directory.name]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise PreflightValidationError(
            f"target is not pinned for this gate: {model_directory.name}") from error

    config_path = model_directory / "config.json"
    try:
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightValidationError(
            f"cannot read target config: {config_path}") from error
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if config_sha256 != pin.get("config_sha256"):
        raise PreflightValidationError(
            f"target config SHA-256 changed: expected {pin.get('config_sha256')}, "
            f"got {config_sha256}")
    revision = _metadata_revision(model_directory)
    if revision != pin.get("revision"):
        raise PreflightValidationError(
            f"target revision changed: expected {pin.get('revision')}, got {revision}")
    manifest = authenticate_file_manifest(model_directory, pin.get("weights", {}))
    if manifest["manifest_sha256"] != pin.get("weight_manifest_sha256"):
        raise PreflightValidationError(
            "target weight manifest SHA-256 changed: "
            f"expected {pin.get('weight_manifest_sha256')}, "
            f"got {manifest['manifest_sha256']}")
    return {
        "name": model_directory.name,
        "revision": revision,
        "config_sha256": config_sha256,
        "quantization": config.get("quantization"),
        "weight_manifest_sha256": manifest["manifest_sha256"],
        "weight_file_count": manifest["file_count"],
        "weight_bytes": manifest["total_bytes"],
    }


def load_runtime(model_directory: Path, head_directory: Path):
    if not model_directory.is_dir() or not (model_directory / "config.json").is_file():
        raise PreflightValidationError(
            f"target must be an existing local model directory: {model_directory}")
    target_identity = authenticate_target(model_directory)
    head_identity = inspect(head_directory)
    print(f"loading target {model_directory.name}", flush=True)
    model, tokenizer = mlx_load(str(model_directory))
    print(f"loading EAGLE head {PINNED_REVISION[:12]}", flush=True)
    head = Eagle3Head.load(head_directory)
    return model, tokenizer, head, target_identity, head_identity


def write_json(path: Optional[Path], payload: dict) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


def run_verify(args) -> None:
    model, tokenizer, head, target_identity, head_identity = load_runtime(
        args.model, args.head)
    prompt_tokens = encode_prompt(tokenizer, args.prompt, args.enable_thinking)
    base = baseline_generate(model, tokenizer, prompt_tokens, args.max_tokens)
    speculative = eagle_generate(
        model,
        head,
        tokenizer,
        prompt_tokens,
        args.max_tokens,
        args.num_draft,
        synchronize_phases=args.synchronize_phases,
        capture_trace=True,
    )
    comparison = exactness(base, speculative)
    engaged = speculative.counters.proposed > 0 and speculative.counters.accepted > 0
    payload = {
        "subcommand": "verify",
        "provenance": provenance(
            args.head, args.harness_git_sha, target_identity, head_identity),
        "parameters": {
            "max_tokens": args.max_tokens,
            "num_draft": args.num_draft,
            "temperature": 0,
            "enable_thinking": args.enable_thinking,
            "prompt": args.prompt,
        },
        "base": base.summary(),
        "speculative": speculative.summary(),
        "exactness": comparison,
        "engaged": engaged,
        "round_trace": speculative.round_trace,
        "passed": (
            comparison["token_ids_identical"]
            and comparison["bytes_identical"]
            and engaged
        ),
    }
    mismatch_index = comparison["first_mismatch_index"]
    if mismatch_index is not None:
        start = max(0, mismatch_index - 4)
        end = mismatch_index + 5
        payload["mismatch_context"] = {
            "start_index": start,
            "base_tokens": base.tokens[start:end],
            "speculative_tokens": speculative.tokens[start:end],
        }
        payload["cache_drift_diagnostic"] = diagnose_cache_drift(
            model,
            prompt_tokens,
            base.tokens,
            speculative.round_trace or (),
            mismatch_index,
        )
    payload["evidence_id"] = stable_evidence_id(payload)
    write_json(args.output, payload)
    if not payload["passed"]:
        raise SystemExit("VERIFY FAIL: exactness and non-vacuous engagement are required")


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return statistics.mean(present) if present else None


def _write_csv(path: Path, rows: List[dict], payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance_fields = [
        "evidence_id",
        "evidence_date",
        "harness_git_sha",
        "hardware_chip",
        "runtime_mlx",
        "runtime_mlx_lm",
        "target_name",
        "target_revision",
        "target_config_sha256",
        "target_weight_manifest_sha256",
        "target_quantization",
        "draft_revision",
        "draft_model_sha256",
        "parameters_max_tokens",
        "parameters_num_draft",
        "parameters_runs",
        "parameters_warmup",
    ]
    metric_fields = [
        "workload",
        "num_draft",
        "run",
        "base_tokens_per_second",
        "speculative_tokens_per_second",
        "speedup",
        "proposed_draft_tokens",
        "accepted_draft_tokens",
        "verify_rounds",
        "accepted_drafts_per_round",
        "inclusive_acceptance_length",
        "token_ids_identical",
        "bytes_identical",
    ]
    provenance = payload["provenance"]
    parameters = payload["parameters"]
    context = {
        "evidence_id": payload["evidence_id"],
        "evidence_date": provenance["date"],
        "harness_git_sha": provenance["harness_git_sha"],
        "hardware_chip": provenance["hardware"]["chip"],
        "runtime_mlx": provenance["runtime"]["mlx"],
        "runtime_mlx_lm": provenance["runtime"]["mlx_lm"],
        "target_name": provenance["target"]["name"],
        "target_revision": provenance["target"]["revision"],
        "target_config_sha256": provenance["target"]["config_sha256"],
        "target_weight_manifest_sha256": provenance["target"][
            "weight_manifest_sha256"],
        "target_quantization": json.dumps(
            provenance["target"]["quantization"], sort_keys=True),
        "draft_revision": provenance["draft"]["revision"],
        "draft_model_sha256": provenance["draft"]["model_sha256"],
        "parameters_max_tokens": parameters["max_tokens"],
        "parameters_num_draft": json.dumps(parameters["num_draft"]),
        "parameters_runs": parameters["runs"],
        "parameters_warmup": parameters["warmup"],
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=provenance_fields + metric_fields)
        writer.writeheader()
        writer.writerows(
            {**context, **{key: row.get(key) for key in metric_fields}}
            for row in rows
        )


def run_bench(args) -> None:
    model, tokenizer, head, target_identity, head_identity = load_runtime(
        args.model, args.head)
    encoded_workloads = [
        (item["name"], item["prompt"], encode_prompt(
            tokenizer, item["prompt"], args.enable_thinking))
        for item in WORKLOADS
    ]
    for num_draft in args.num_draft:
        for warmup in range(args.warmup):
            print(
                f"warmup k={num_draft} {warmup + 1}/{args.warmup}", flush=True)
            _, _, tokens = encoded_workloads[0]
            baseline_generate(model, tokenizer, tokens, min(24, args.max_tokens))
            eagle_generate(
                model,
                head,
                tokenizer,
                tokens,
                min(24, args.max_tokens),
                num_draft,
                synchronize_phases=False,
            )

    rows = []
    diagnostics = []
    for num_draft in args.num_draft:
        for workload, prompt, tokens in encoded_workloads:
            for run_index in range(args.runs):
                print(
                    f"measure workload={workload} k={num_draft} run={run_index + 1}/{args.runs}",
                    flush=True,
                )
                if run_index % 2 == 0:
                    base = baseline_generate(model, tokenizer, tokens, args.max_tokens)
                    speculative = eagle_generate(
                        model, head, tokenizer, tokens, args.max_tokens, num_draft, False)
                else:
                    speculative = eagle_generate(
                        model, head, tokenizer, tokens, args.max_tokens, num_draft, False)
                    base = baseline_generate(model, tokenizer, tokens, args.max_tokens)
                equality = exactness(base, speculative)
                if not equality["token_ids_identical"] or not equality["bytes_identical"]:
                    raise SystemExit(
                        f"BENCH FAIL: output mismatch workload={workload} k={num_draft} "
                        f"run={run_index + 1}")
                if speculative.counters.proposed == 0 or speculative.counters.accepted == 0:
                    raise SystemExit(
                        "BENCH FAIL: speculation did not both propose and accept tokens "
                        f"workload={workload} k={num_draft}")
                base_rate = base.tokens_per_second
                spec_rate = speculative.tokens_per_second
                speedup = spec_rate / base_rate
                rows.append(
                    {
                        "workload": workload,
                        "prompt": prompt,
                        "num_draft": num_draft,
                        "run": run_index + 1,
                        "base_tokens_per_second": base_rate,
                        "speculative_tokens_per_second": spec_rate,
                        "speedup": speedup,
                        "proposed_draft_tokens": speculative.counters.proposed,
                        "accepted_draft_tokens": speculative.counters.accepted,
                        "verify_rounds": speculative.counters.verify_rounds,
                        "proposal_acceptance_rate": (
                            speculative.counters.proposal_acceptance_rate),
                        "accepted_drafts_per_round": (
                            speculative.counters.accepted_drafts_per_round),
                        "inclusive_acceptance_length": (
                            speculative.counters.inclusive_acceptance_length),
                        "token_ids_identical": equality["token_ids_identical"],
                        "bytes_identical": equality["bytes_identical"],
                        "base_peak_memory_bytes": base.peak_memory_bytes,
                        "speculative_peak_memory_bytes": speculative.peak_memory_bytes,
                    })

            print(f"diagnostic phases workload={workload} k={num_draft}", flush=True)
            diagnostic = eagle_generate(
                model,
                head,
                tokenizer,
                tokens,
                args.max_tokens,
                num_draft,
                synchronize_phases=True,
            )
            diagnostic_equality = exactness(base, diagnostic)
            if (
                not diagnostic_equality["token_ids_identical"]
                or not diagnostic_equality["bytes_identical"]
            ):
                raise SystemExit(
                    f"BENCH FAIL: phase synchronization changed output "
                    f"workload={workload} k={num_draft}")
            baseline_rates = [
                row["base_tokens_per_second"]
                for row in rows
                if row["workload"] == workload and row["num_draft"] == num_draft
            ]
            baseline_rate = statistics.mean(baseline_rates)
            timing = diagnostic.phase_timing
            economics = timing.economics(
                baseline_token_seconds=1 / baseline_rate,
                accepted_drafts_per_round=(
                    diagnostic.counters.accepted_drafts_per_round or 0),
            )
            diagnostics.append(
                {
                    "workload": workload,
                    "num_draft": num_draft,
                    "speculative": diagnostic.summary(),
                    "serialized_phase_economics": economics,
                    "exactness": diagnostic_equality,
                })

    summaries = []
    for num_draft in args.num_draft:
        for workload, _, _ in encoded_workloads:
            selected = [
                row for row in rows
                if row["workload"] == workload and row["num_draft"] == num_draft
            ]
            summaries.append(
                {
                    "workload": workload,
                    "num_draft": num_draft,
                    "runs": len(selected),
                    "mean_base_tokens_per_second": _mean(
                        row["base_tokens_per_second"] for row in selected),
                    "mean_speculative_tokens_per_second": _mean(
                        row["speculative_tokens_per_second"] for row in selected),
                    "mean_speedup": _mean(row["speedup"] for row in selected),
                    "speedup_ratio_of_means": (
                        _mean(row["speculative_tokens_per_second"] for row in selected)
                        / _mean(row["base_tokens_per_second"] for row in selected)
                    ),
                    "mean_accepted_drafts_per_round": _mean(
                        row["accepted_drafts_per_round"] for row in selected),
                    "mean_inclusive_acceptance_length": _mean(
                        row["inclusive_acceptance_length"] for row in selected),
                    "all_exact": all(
                        row["token_ids_identical"] and row["bytes_identical"]
                        for row in selected),
                })
    payload = {
        "subcommand": "bench",
        "provenance": provenance(
            args.head, args.harness_git_sha, target_identity, head_identity),
        "parameters": {
            "max_tokens": args.max_tokens,
            "num_draft": args.num_draft,
            "runs": args.runs,
            "warmup": args.warmup,
            "temperature": 0,
            "enable_thinking": args.enable_thinking,
        },
        "rows": rows,
        "summaries": summaries,
        "phase_diagnostics": diagnostics,
    }
    payload["evidence_id"] = stable_evidence_id(payload)
    write_json(args.output, payload)
    if args.csv_output:
        _write_csv(args.csv_output, rows, payload)


def common_arguments(parser) -> None:
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--harness-sha-file", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--enable-thinking", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    verify_parser = subparsers.add_parser("verify")
    common_arguments(verify_parser)
    verify_parser.add_argument("--prompt", required=True)
    verify_parser.add_argument("--max-tokens", type=int, default=64)
    verify_parser.add_argument("--num-draft", type=int, default=1)
    verify_parser.add_argument("--synchronize-phases", action="store_true")
    verify_parser.set_defaults(function=run_verify)

    bench_parser = subparsers.add_parser("bench")
    common_arguments(bench_parser)
    bench_parser.add_argument("--max-tokens", type=int, default=128)
    bench_parser.add_argument("--num-draft", type=int, nargs="+", default=[1, 3])
    bench_parser.add_argument("--runs", type=int, default=3)
    bench_parser.add_argument("--warmup", type=int, default=1)
    bench_parser.add_argument("--csv-output", type=Path)
    bench_parser.set_defaults(function=run_bench)

    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if getattr(args, "runs", 1) <= 0:
        parser.error("--runs must be positive")
    args.harness_git_sha = read_harness_git_sha(
        args.harness_sha_file, allow_dirty=args.allow_dirty)
    # llmbench's iogpu.wired_limit_mb is raised; keep MLX cache bounded explicitly.
    mx.set_cache_limit(8 << 30)
    args.function(args)


if __name__ == "__main__":
    main()
