"""MLX-free contracts for the Qwen3-32B EAGLE-3 Phase 0 preflight."""

from __future__ import annotations

import json
import hashlib
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


class PreflightValidationError(ValueError):
    """The checkpoint or measurement violates a fail-closed preflight contract."""


_CLEAN_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_DIRTY_GIT_SHA = re.compile(r"^[0-9a-f]{40}-dirty$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash an artifact from disk without materializing it in memory."""
    if chunk_size <= 0:
        raise PreflightValidationError("hash chunk size must be positive")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as error:
        raise PreflightValidationError(f"cannot hash artifact: {path}") from error
    return digest.hexdigest()


def read_harness_git_sha(path: Path, allow_dirty: bool = False) -> str:
    """Read a full Git SHA, rejecting dirty evidence unless explicitly diagnostic."""
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise PreflightValidationError(f"cannot read harness Git SHA: {path}") from error
    if _CLEAN_GIT_SHA.fullmatch(value):
        return value
    if _DIRTY_GIT_SHA.fullmatch(value):
        if allow_dirty:
            return value
        raise PreflightValidationError(
            "dirty harness Git SHA is diagnostic-only; commit and sync before verdict evidence")
    raise PreflightValidationError(f"invalid harness Git SHA: {value!r}")


def stable_evidence_id(payload: Any) -> str:
    """Bind detached evidence views to one canonical JSON payload."""
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def authenticate_file_manifest(
    directory: Path,
    expected: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Authenticate an exact set of named files and return its canonical manifest digest."""
    observed = {path.name: path for path in directory.glob("*.safetensors") if path.is_file()}
    expected_names = set(expected)
    observed_names = set(observed)
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        unexpected = sorted(observed_names - expected_names)
        raise PreflightValidationError(
            f"weight shard set changed: missing={missing}, unexpected={unexpected}")

    entries = []
    for name in sorted(expected):
        pin = expected[name]
        expected_bytes = pin.get("bytes")
        expected_sha256 = pin.get("sha256")
        if type(expected_bytes) is not int or expected_bytes < 0:
            raise PreflightValidationError(f"invalid pinned byte count for {name}")
        if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
            raise PreflightValidationError(f"invalid pinned SHA-256 for {name}")
        path = observed[name]
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise PreflightValidationError(
                f"weight shard size changed for {name}: "
                f"expected {expected_bytes}, got {actual_bytes}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise PreflightValidationError(
                f"weight shard SHA-256 changed for {name}: "
                f"expected {expected_sha256}, got {actual_sha256}")
        entries.append(
            {"name": name, "bytes": actual_bytes, "sha256": actual_sha256})

    return {
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "manifest_sha256": stable_evidence_id(entries),
        "files": entries,
    }


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    try:
        return mapping[key]
    except KeyError as error:
        raise PreflightValidationError(f"{context} is missing {key!r}") from error


def _require_int(mapping: Mapping[str, Any], key: str, context: str) -> int:
    value = _require(mapping, key, context)
    if type(value) is not int:
        raise PreflightValidationError(
            f"{context}.{key} must be an integer, got {value!r}")
    return value


def _require_number(mapping: Mapping[str, Any], key: str, context: str) -> float:
    value = _require(mapping, key, context)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise PreflightValidationError(
            f"{context}.{key} must be a finite number, got {value!r}")
    return float(value)


def _require_bool(mapping: Mapping[str, Any], key: str, context: str) -> bool:
    value = _require(mapping, key, context)
    if type(value) is not bool:
        raise PreflightValidationError(
            f"{context}.{key} must be a boolean, got {value!r}")
    return value


@dataclass(frozen=True)
class CheckpointSpec:
    """Pinned geometry for RedHatAI/Qwen3-32B-speculator.eagle3."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    target_vocab_size: int
    draft_vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    norm_before_residual: bool
    max_draft: int
    tap_layers: Tuple[int, int, int]

    @property
    def query_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @classmethod
    def from_config(cls, raw: Mapping[str, Any]) -> "CheckpointSpec":
        architectures = _require(raw, "architectures", "config")
        if architectures != ["Eagle3Speculator"]:
            raise PreflightValidationError(
                f"architectures changed: expected ['Eagle3Speculator'], got {architectures!r}")

        speculators = _require(raw, "speculators_config", "config")
        if _require(speculators, "algorithm", "speculators_config") != "eagle3":
            raise PreflightValidationError("speculators_config.algorithm must be 'eagle3'")
        verifier = _require(speculators, "verifier", "speculators_config")
        verifier_name = _require(verifier, "name_or_path", "speculators_config.verifier")
        if verifier_name != "Qwen/Qwen3-32B":
            raise PreflightValidationError(
                f"verifier changed: expected 'Qwen/Qwen3-32B', got {verifier_name!r}")

        proposals = _require(speculators, "proposal_methods", "speculators_config")
        if not isinstance(proposals, list) or len(proposals) != 1:
            raise PreflightValidationError("expected exactly one proposal method")
        max_draft = _require_int(proposals[0], "speculative_tokens", "proposal method")

        layer = _require(raw, "transformer_layer_config", "config")
        model_type = _require(layer, "model_type", "transformer_layer_config")
        if model_type != "llama":
            raise PreflightValidationError(
                f"transformer_layer_config.model_type must stay 'llama', got {model_type!r}")
        if _require_int(layer, "num_hidden_layers", "transformer_layer_config") != 1:
            raise PreflightValidationError("the pinned EAGLE head must contain exactly one layer")
        attention_bias = layer.get("attention_bias", False)
        if type(attention_bias) is not bool:
            raise PreflightValidationError(
                "transformer_layer_config.attention_bias must be a boolean")
        if attention_bias:
            raise PreflightValidationError("the pinned EAGLE head has bias-free attention")

        norm_before_residual = _require_bool(raw, "norm_before_residual", "config")
        if not norm_before_residual:
            raise PreflightValidationError("norm_before_residual must remain true")

        resolved = cls(
            hidden_size=_require_int(layer, "hidden_size", "transformer_layer_config"),
            intermediate_size=_require_int(
                layer, "intermediate_size", "transformer_layer_config"),
            num_attention_heads=_require_int(
                layer, "num_attention_heads", "transformer_layer_config"),
            num_key_value_heads=_require_int(
                layer, "num_key_value_heads", "transformer_layer_config"),
            head_dim=_require_int(layer, "head_dim", "transformer_layer_config"),
            target_vocab_size=_require_int(
                layer, "vocab_size", "transformer_layer_config"),
            draft_vocab_size=_require_int(raw, "draft_vocab_size", "config"),
            rms_norm_eps=_require_number(
                layer, "rms_norm_eps", "transformer_layer_config"),
            rope_theta=_require_number(layer, "rope_theta", "transformer_layer_config"),
            norm_before_residual=norm_before_residual,
            max_draft=max_draft,
            # Current speculators resolves a missing list to {2, N/2, N-3}; Qwen3-32B N=64.
            tap_layers=(2, 32, 61),
        )

        expected_scalars = {
            "hidden_size": 5120,
            "intermediate_size": 25600,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "target_vocab_size": 151936,
            "draft_vocab_size": 32000,
            "max_draft": 3,
        }
        for field_name, expected in expected_scalars.items():
            actual = getattr(resolved, field_name)
            if actual != expected:
                raise PreflightValidationError(
                    f"{field_name} changed: expected {expected}, got {actual}")
        if not math.isclose(resolved.rms_norm_eps, 1e-6, rel_tol=0, abs_tol=1e-12):
            raise PreflightValidationError(
                f"rms_norm_eps changed: expected 1e-6, got {resolved.rms_norm_eps}")
        if resolved.rope_theta != 1_000_000:
            raise PreflightValidationError(
                f"rope_theta changed: expected 1000000, got {resolved.rope_theta}")
        return resolved

    @staticmethod
    def expected_tensor_manifest() -> Dict[str, Tuple[str, Tuple[int, ...]]]:
        return {
            "d2t": ("I64", (32000,)),
            "embed_tokens.weight": ("BF16", (151936, 5120)),
            "fc.weight": ("BF16", (5120, 15360)),
            "layers.0.hidden_norm.weight": ("BF16", (5120,)),
            "layers.0.input_layernorm.weight": ("BF16", (5120,)),
            "layers.0.mlp.down_proj.weight": ("BF16", (5120, 25600)),
            "layers.0.mlp.gate_proj.weight": ("BF16", (25600, 5120)),
            "layers.0.mlp.up_proj.weight": ("BF16", (25600, 5120)),
            "layers.0.post_attention_layernorm.weight": ("BF16", (5120,)),
            "layers.0.self_attn.k_proj.weight": ("BF16", (1024, 10240)),
            "layers.0.self_attn.o_proj.weight": ("BF16", (5120, 8192)),
            "layers.0.self_attn.q_proj.weight": ("BF16", (8192, 10240)),
            "layers.0.self_attn.v_proj.weight": ("BF16", (1024, 10240)),
            "lm_head.weight": ("BF16", (32000, 5120)),
            "norm.weight": ("BF16", (5120,)),
            "t2d": ("BOOL", (151936,)),
        }


def parse_safetensors_header_bytes(blob: bytes) -> Dict[str, Any]:
    if len(blob) < 8:
        raise PreflightValidationError("safetensors file is shorter than its length prefix")
    (header_length,) = struct.unpack("<Q", blob[:8])
    if header_length <= 0 or header_length > 16 * 1024 * 1024:
        raise PreflightValidationError(
            f"invalid safetensors header length: {header_length}")
    if len(blob) < 8 + header_length:
        raise PreflightValidationError(
            f"truncated safetensors header: need {header_length} bytes")
    try:
        parsed = json.loads(blob[8 : 8 + header_length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightValidationError("safetensors header is not valid UTF-8 JSON") from error
    if not isinstance(parsed, dict):
        raise PreflightValidationError("safetensors header must be a JSON object")
    return parsed


def read_safetensors_layout(path: Path) -> Tuple[Dict[str, Any], int]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise PreflightValidationError("safetensors file has no complete length prefix")
        (header_length,) = struct.unpack("<Q", prefix)
        if header_length <= 0 or header_length > 16 * 1024 * 1024:
            raise PreflightValidationError(
                f"invalid safetensors header length: {header_length}")
        header = parse_safetensors_header_bytes(prefix + handle.read(header_length))
    data_size = path.stat().st_size - 8 - header_length
    if data_size < 0:
        raise PreflightValidationError("safetensors payload size is negative")
    return header, data_size


def read_safetensors_header(path: Path) -> Dict[str, Any]:
    return read_safetensors_layout(path)[0]


def resolve_huggingface_revision(head_directory: Path) -> Tuple[str, str]:
    metadata_path = (
        head_directory
        / ".cache"
        / "huggingface"
        / "download"
        / "model.safetensors.metadata"
    )
    try:
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise PreflightValidationError(
            f"missing Hugging Face revision metadata: {metadata_path}") from error
    if len(lines) < 2 or not lines[0].strip() or not lines[1].strip():
        raise PreflightValidationError(
            f"incomplete Hugging Face revision metadata: {metadata_path}")
    return lines[0].strip(), lines[1].strip()


def validate_tensor_manifest(
    header: Mapping[str, Any], data_size: Optional[int] = None
) -> None:
    metadata = header.get("__metadata__", {})
    if metadata.get("format") != "pt":
        raise PreflightValidationError(
            f"expected safetensors metadata format 'pt', got {metadata!r}")

    expected = CheckpointSpec.expected_tensor_manifest()
    observed_names = set(header) - {"__metadata__"}
    missing = sorted(set(expected) - observed_names)
    unexpected = sorted(observed_names - set(expected))
    if missing:
        raise PreflightValidationError(f"missing tensors: {', '.join(missing)}")
    if unexpected:
        raise PreflightValidationError(f"unexpected tensors: {', '.join(unexpected)}")

    element_bytes = {"BF16": 2, "I64": 8, "BOOL": 1}
    mismatches = []
    intervals = []
    for name, (expected_dtype, expected_shape) in expected.items():
        entry = header[name]
        dtype = entry.get("dtype")
        shape = tuple(entry.get("shape", ()))
        if dtype != expected_dtype or shape != expected_shape:
            mismatches.append(
                f"{name}: expected {expected_dtype}{expected_shape}, got {dtype}{shape}")
            continue
        offsets = entry.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
        ):
            mismatches.append(f"{name}: invalid data_offsets {offsets!r}")
            continue
        start, end = offsets
        expected_bytes = math.prod(expected_shape) * element_bytes[expected_dtype]
        if start < 0 or end < start or end - start != expected_bytes:
            mismatches.append(
                f"{name}: invalid data_offsets {offsets!r}, expected {expected_bytes} bytes")
            continue
        intervals.append((start, end, name))
    if mismatches:
        raise PreflightValidationError("tensor manifest mismatch: " + "; ".join(mismatches))

    cursor = 0
    for start, end, name in sorted(intervals):
        if start != cursor:
            raise PreflightValidationError(
                f"tensor data offset gap/overlap before {name}: expected {cursor}, got {start}")
        cursor = end
    if data_size is not None and cursor != data_size:
        raise PreflightValidationError(
            f"tensor payload size mismatch: manifest={cursor}, file={data_size}")


@dataclass(frozen=True)
class AcceptanceResult:
    accepted: int
    bonus: int
    emitted: Tuple[int, ...]


def classify_cache_drift(
    expected_token: int,
    full_verify_token: int,
    retained_batch_token: int,
    sequential_batched_token: int,
    sequential_one_token: int,
) -> str:
    """Classify a mismatch by replaying increasingly conservative target-cache histories."""
    if full_verify_token == expected_token:
        return "not-reproduced"
    if sequential_one_token != expected_token:
        return "sequential-replay-mismatch"
    if sequential_batched_token != expected_token:
        return "immediate-batched-probe-drift"
    if retained_batch_token == expected_token:
        return "rejected-future-cache-drift"
    if retained_batch_token != expected_token:
        return "batched-retained-prefix-drift"
    return "unclassified"


def accept_greedy(draft: Sequence[int], verify_argmax: Sequence[int]) -> AcceptanceResult:
    if len(verify_argmax) != len(draft) + 1:
        raise PreflightValidationError(
            "verify_argmax must contain one target pick per draft plus the trailing bonus")
    accepted = 0
    while accepted < len(draft) and draft[accepted] == verify_argmax[accepted]:
        accepted += 1
    bonus = int(verify_argmax[accepted])
    emitted = tuple(int(token) for token in draft[:accepted]) + (bonus,)
    return AcceptanceResult(accepted=accepted, bonus=bonus, emitted=emitted)


def trim_emission(
    emitted: Sequence[int],
    already_emitted: int,
    max_tokens: int,
    eos_ids: Iterable[int],
) -> Tuple[Tuple[int, ...], bool]:
    if already_emitted < 0 or max_tokens < 0:
        raise PreflightValidationError("token counts must be nonnegative")
    remaining = max(0, max_tokens - already_emitted)
    trimmed = tuple(int(token) for token in emitted[:remaining])
    done = len(trimmed) < len(emitted) or already_emitted + len(trimmed) >= max_tokens
    eos = set(int(token) for token in eos_ids)
    for index, token in enumerate(trimmed):
        if token in eos:
            return trimmed[: index + 1], True
    return trimmed, done


def committed_accepted_drafts(accepted: int, emitted_count: int) -> int:
    """Count accepted drafts that survived EOS/budget trimming and were actually emitted."""
    if accepted < 0 or emitted_count < 0:
        raise PreflightValidationError("accepted and emitted counts must be nonnegative")
    return min(accepted, emitted_count)


@dataclass(frozen=True)
class AcceptanceCounters:
    proposed: int
    accepted: int
    verify_rounds: int

    def __post_init__(self) -> None:
        if self.proposed < 0 or self.accepted < 0 or self.verify_rounds < 0:
            raise PreflightValidationError("acceptance counters must be nonnegative")
        if self.accepted > self.proposed:
            raise PreflightValidationError("accepted drafts cannot exceed proposed drafts")
        if self.proposed > 0 and self.verify_rounds == 0:
            raise PreflightValidationError("proposed drafts require at least one verify round")

    @property
    def proposal_acceptance_rate(self) -> Optional[float]:
        return self.accepted / self.proposed if self.proposed else None

    @property
    def accepted_drafts_per_round(self) -> Optional[float]:
        return self.accepted / self.verify_rounds if self.verify_rounds else None

    @property
    def inclusive_acceptance_length(self) -> Optional[float]:
        yield_per_round = self.accepted_drafts_per_round
        return 1 + yield_per_round if yield_per_round is not None else None


@dataclass(frozen=True)
class RoundTiming:
    draft_seconds: float
    verify_seconds: float
    commit_seconds: float

    def __post_init__(self) -> None:
        for name, value in (
            ("draft_seconds", self.draft_seconds),
            ("verify_seconds", self.verify_seconds),
            ("commit_seconds", self.commit_seconds),
        ):
            if not math.isfinite(value) or value < 0:
                raise PreflightValidationError(f"{name} must be finite and nonnegative")

    @property
    def total_seconds(self) -> float:
        return self.draft_seconds + self.verify_seconds + self.commit_seconds

    def economics(
        self,
        baseline_token_seconds: float,
        accepted_drafts_per_round: float,
    ) -> Dict[str, float]:
        if not math.isfinite(baseline_token_seconds) or baseline_token_seconds <= 0:
            raise PreflightValidationError("baseline_token_seconds must be finite and positive")
        if not math.isfinite(accepted_drafts_per_round) or accepted_drafts_per_round < 0:
            raise PreflightValidationError(
                "accepted_drafts_per_round must be finite and nonnegative")
        if self.total_seconds <= 0:
            raise PreflightValidationError("speculative round time must be positive")
        round_cost_ratio = self.total_seconds / baseline_token_seconds
        emitted = 1 + accepted_drafts_per_round
        return {
            "speculative_round_seconds": self.total_seconds,
            "round_cost_ratio": round_cost_ratio,
            "break_even_accepted_drafts_per_round": round_cost_ratio - 1,
            "emitted_tokens_per_round": emitted,
            "projected_speedup": emitted / round_cost_ratio,
        }
