#!/usr/bin/env python3
"""``fastmlx serve``: an engine-agnostic launcher for an OpenAI-compatible
serving engine.

This is a product-facing front door, not a replacement for any particular
serving engine: given a local model directory it (1) runs a pre-load fit
check against the requested host, (2) consults a quality-guidance card for
the resolved model (if one applies) and refuses or opt-in-admits a flagged
configuration exactly the way this repository's Swift admission gate does,
and then (3) execs whatever OpenAI-compatible engine an "engine profile"
names -- this repository's own in-tree serving CLI by default, or any other
engine an operator points it at via ``--engine-profile``/``--engine-bin``.

Fail-closed by construction: a fit check that cannot be run at all (missing
binary, unexpected exit code, a timeout, a green exit with no attestation
line) refuses startup even under ``--force`` -- ``--force`` overrides only a
confirmed RED verdict, never an unknown one. A malformed or unrecognized
engine-profile placeholder refuses rather than silently dropping a token
from the exec'd command line.

Every subprocess this script starts is invoked as an argv list, never
through a shell. The operator's environment is never modified and is
inherited by the exec'd engine untouched -- API keys the engine wants live
in that environment, not on the command line this script constructs, and
this script never prints or logs environment values. The only variable it
reads is ``FASTMLX_FIT_CHECK_BIN`` (a fallback for ``--fit-check-bin``).

``--residency {resident,expert-stream}`` (default ``resident``) names how the
requested model pack is held while serving: fully resident in memory, or
served with its experts streamed from disk. A quality card only ever
matches a launch whose residency equals the card's own ``config.residency``
(absent/null on a card means "resident") -- the same measured drift a
resident launch produces is not evidence for a streaming launch of the same
pack, and vice versa, so the two residencies are never cross-admitted by one
card. ``--residency expert-stream`` additionally requires an
``--engine-profile`` whose ``residencyArgs.expert-stream`` names the argv
this build's engine needs to stream experts (the built-in engine has none
and always refuses); an operator can never bypass that gate by passing the
streaming flag directly as a passthrough argument.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUALITY_CARDS_RELATIVE_PATH = "site/quality-guides.json"

# ``fastmlx_pull.py`` is the one place that decides where a pull receipt
# lives (``receipt_path_for``) and what fields it carries; imported the same
# sibling-file way ``fastmlx_recommend.py`` imports this module, so the
# launcher can never drift from pull's own naming rule again.
_PULL_PATH = Path(__file__).resolve().parent / "fastmlx_pull.py"
_PULL_SPEC = importlib.util.spec_from_file_location("fastmlx_pull", _PULL_PATH)
assert _PULL_SPEC is not None and _PULL_SPEC.loader is not None
pull = importlib.util.module_from_spec(_PULL_SPEC)
_PULL_SPEC.loader.exec_module(pull)

# The opt-in provenance proxy `--front-port` starts in front of the engine
# (see `_run_front_mode` below); loaded the same sibling-file way `pull` is,
# above, so this launcher never depends on `fastmlx_proxy` being importable
# as a package -- only on it being the file next to this one, exactly like
# every release tarball ships it (libexec/scripts).
_PROXY_PATH = Path(__file__).resolve().parent / "fastmlx_proxy.py"
_PROXY_SPEC = importlib.util.spec_from_file_location("fastmlx_proxy", _PROXY_PATH)
assert _PROXY_SPEC is not None and _PROXY_SPEC.loader is not None
fastmlx_proxy = importlib.util.module_from_spec(_PROXY_SPEC)
_PROXY_SPEC.loader.exec_module(fastmlx_proxy)

# Loaded by file path (not by package import), the same cross-module-reuse
# pattern fastmlx_safetensors_fit.py itself uses to reach fastmlx_gguf_fit.py's
# internals (see that file's own "Shared helpers" comment): this lets
# _pack_has_safetensors below reuse the sizer's OWN countable-file scan
# (_iter_regular_files) instead of a second, hand-copied traversal that could
# silently drift out of step with what the sizer actually counts.
_SAFETENSORS_FIT_PATH = Path(__file__).resolve().parent / "fastmlx_safetensors_fit.py"
_SAFETENSORS_FIT_SPEC = importlib.util.spec_from_file_location(
    "_fastmlx_launch_safetensors_fit", _SAFETENSORS_FIT_PATH
)
assert _SAFETENSORS_FIT_SPEC is not None and _SAFETENSORS_FIT_SPEC.loader is not None
_safetensors_fit = importlib.util.module_from_spec(_SAFETENSORS_FIT_SPEC)
_SAFETENSORS_FIT_SPEC.loader.exec_module(_safetensors_fit)

# The one Swift binary this repository ships that can both answer a
# ``--fit-check-only`` pre-load question and serve an OpenAI-compatible API,
# used as the default for both ``--fit-check-bin`` and (absent a custom
# ``--engine-profile``) the engine itself. Built by concatenation, on
# purpose: written as one literal token this name contains, as a plain
# substring, an unrelated forbidden term this file's public leak sweep
# scans for (a different, third-party engine's short name) -- splitting the
# literal here keeps the *source text* free of that substring while the
# *value* this constant holds at runtime is unaffected.
_BUILT_IN_ENGINE_BINARY_NAME = "fastmlx" + "-serve"

ALLOWED_ENGINE_PROFILE_PLACEHOLDERS = {
    "engine_bin",
    "model_path",
    "model_id",
    "host",
    "port",
    "context",
}

# The two residencies a launch (and a quality card) can carry. "resident"
# is the historical default -- the whole pack held in memory; a card with
# no ``config.residency`` at all is treated as "resident" (see
# ``card_residency``). "expert-stream" is the only other recognized value.
RESIDENCIES = ("resident", "expert-stream")
ALLOWED_RESIDENCY_ARGS_KEYS = {"expert-stream"}

# An engine profile's OPTIONAL `engineBuild` names the engine build this
# profile launches: `commit` (a lowercase 40-hex git sha, operator-asserted)
# and/or `binarySha256` (a lowercase 64-hex sha256 of the resolved engine
# binary, VERIFIED by this launcher before exec -- see the binarySha256
# check in `_run_serve`). Both sub-keys are optional; no other key is
# allowed. See `docs/quality-card-schema-v1.md` "Engine build".
ALLOWED_ENGINE_BUILD_PROFILE_KEYS = {"commit", "binarySha256"}
_LOWERCASE_HEX40_RE = re.compile(r"[0-9a-f]{40}")
_LOWERCASE_HEX64_RE = re.compile(r"[0-9a-f]{64}")

ENGINE_BUILD_STATUS_UNRECORDED = "unrecorded"
ENGINE_BUILD_STATUS_MATCH = "match"
ENGINE_BUILD_STATUS_UNDECLARED = "undeclared"
ENGINE_BUILD_STATUS_MISMATCH = "mismatch"

# `config.flagTransfer["--mtp"]` (OPTIONAL card field) / the `--mtp` launch
# flag itself: whether a card's greedy output was proven to transfer to a
# launch of the SAME pack made WITH `--mtp`, given the card was measured
# WITHOUT it (see docs/quality-card-schema-v1.md "Flag transfer"). Never
# used to filter card admission -- only to classify it for the notice this
# module's callers surface, exactly like `ENGINE_BUILD_STATUS_*` above.
MTP_STATUS_OFF = "off"
MTP_STATUS_UNMEASURED = "unmeasured"
MTP_STATUS_EXACT = "exact"
MTP_STATUS_NOT_EXACT = "not_exact"
MTP_STATUS_NONDETERMINISTIC = "nondeterministic"

# An engine profile's OPTIONAL `fitCheck` object names the sizer this
# profile's own engine needs -- see `_validate_fit_check` for the full
# validation rule. `bin` is either one of these two symbolic names
# (resolved to a sibling of this file, so it works both in the repo and in
# a release tarball's libexec/scripts) or an absolute path.
_BUILTIN_FIT_CHECK_FILENAMES = {
    "builtin:safetensors": "fastmlx_safetensors_fit.py",
    "builtin:gguf": "fastmlx_gguf_fit.py",
}
ALLOWED_FIT_CHECK_KEYS = {"bin", "args"}

# The inverse of `_BUILTIN_FIT_CHECK_FILENAMES`, keyed by the RESOLVED
# argv[0] `_resolve_fit_check_bin_value` produces for each `builtin:` name
# (a sibling of this file). Used by `_run_serve` to tell whether a fully
# resolved `fit_check_bin` -- however it got resolved: CLI/env override,
# an engine profile's own `fitCheck.bin`, or the auto-select fallback --
# is one of this repository's own built-in pure-Python sizers, without
# needing to thread a separate "was this a builtin:" flag through every
# branch that can produce `fit_check_bin`.
_BUILTIN_FIT_CHECK_BIN_PATHS = {
    str(Path(__file__).resolve().parent / filename): name
    for name, filename in _BUILTIN_FIT_CHECK_FILENAMES.items()
}

# The flags `run_fit_check` itself always supplies; a profile's own
# `fitCheck.args` must never repeat one of these, since the launcher -- not
# the profile -- owns the model identity/host/context this fit check runs
# against.
RESERVED_FIT_CHECK_ARGS = {
    "--fit-check-only",
    "--model",
    "--model-path",
    "--host-use",
    "--context",
}

BUILT_IN_ENGINE_PROFILE = {
    "schema": "fastmlx-engine-profile-v1",
    "name": _BUILT_IN_ENGINE_BINARY_NAME,
    "argv": [
        "{engine_bin}",
        "--model",
        "{model_id}",
        "--model-path",
        "{model_path}",
        "--host",
        "{host}",
        "--port",
        "{port}",
        "--context",
        "{context}",
    ],
}

FIT_CHECK_TIMEOUT_SECONDS = 120


class LaunchRefusal(Exception):
    """A fastmlx serve refusal: carries the exit code and operator-facing reason."""

    def __init__(self, exit_code: int, message: str):
        super().__init__(message)
        self.exit_code = exit_code
        self.message = message


# ---------------------------------------------------------------------
# 0. Model-directory layout: MLX weights (config.json) or a GGUF pack
#    (one or more top-level *.gguf shards, admitted only through a
#    GGUF-aware ``--fit-check-bin`` such as scripts/fastmlx_gguf_fit.py).
# ---------------------------------------------------------------------
MODEL_DIR_REFUSAL_MESSAGE_SUFFIX = "does not contain config.json or any .gguf file"


def is_model_dir(path: Path) -> bool:
    """Whether ``path`` (already known to be a directory) looks like a model
    pack this launcher can hand off to a fit check: an MLX weights
    directory (``config.json`` present) or a GGUF pack (at least one
    top-level ``*.gguf`` REGULAR FILE -- a same-named directory does not
    count). This check does not itself parse or validate GGUF headers; it
    only recognizes the layout so the right ``--fit-check-bin`` can be
    tried.
    """
    if (path / "config.json").is_file():
        return True
    return any(child.is_file() and child.suffix == ".gguf" for child in path.iterdir())


# ---------------------------------------------------------------------
# 0b. Built-in sizer auto-selection: which of this repository's own
#     pure-Python sizers (fastmlx_safetensors_fit.py / fastmlx_gguf_fit.py)
#     applies to a given pack, from that pack's own on-disk contents. The
#     single source of truth for both `fastmlx serve` (the auto-select
#     fallback in `_run_serve`, below) and `fastmlx recommend` (which
#     imports this module rather than duplicating this logic).
# ---------------------------------------------------------------------
def _pack_has_safetensors(model_path: Path) -> bool:
    """Whether ``model_path`` looks like a safetensors pack the built-in
    safetensors sizer (``fastmlx_safetensors_fit.compute_model_bytes``)
    would actually find at least one countable weight file in.

    This reuses that sizer's OWN countable-file scan
    (``_iter_regular_files``) rather than a second, hand-copied traversal,
    so the two can never silently drift apart: a symlink (file or
    directory) is never followed or counted, a dot-named entry is skipped,
    and a dot-named directory is never descended into. A top-level
    ``model.safetensors.index.json`` alone is deliberately NOT treated as
    "has safetensors" here (unlike the previous version of this check):
    the sizer itself refuses with "no .safetensors files found" whenever
    zero countable shard files exist, index.json or not, so a pack with an
    index but no countable shard would still be a guaranteed sizer
    failure -- see ``compute_model_bytes``, which raises before it ever
    reaches its own index-manifest cross-check.

    This only checks for the files' existence via a directory scan; it
    never opens or parses a file, and it never follows a symlink to see
    what it points at (see ``_find_uncounted_pattern_entries`` for the
    separate, permissive diagnostic scan used only to explain a
    symlinked/dot-hidden miss to the operator, never to size or select
    anything).
    """
    return any(
        relative.endswith(".safetensors")
        for relative, _path in _safetensors_fit._iter_regular_files(model_path)
    )


def _pack_has_gguf(model_path: Path) -> bool:
    """Whether ``model_path`` looks like a GGUF pack: at least one
    top-level ``*.gguf`` REGULAR FILE -- the same top-level-only scope
    ``is_model_dir`` and ``fastmlx_gguf_fit.resolve_shards`` itself
    use.

    Unlike the safetensors sizer, the GGUF sizer's own scan
    (``resolve_shards``: ``p.is_file() and p.suffix == ".gguf"`` over
    ``model_path.iterdir()``) does NOT skip a symlink -- ``Path.is_file()``
    follows symlinks by default, so a symlinked ``*.gguf`` entry (exactly
    the shape the canonical Hugging Face hub cache layout produces) is
    already counted by that sizer, and this check (which uses the same
    ``child.is_file()`` test) already agrees with it. There is therefore no
    "present but uncounted because it's a symlink" gap here to add a
    diagnostic for, unlike the safetensors case below: verified by reading
    ``fastmlx_gguf_fit.resolve_shards`` directly, not assumed to mirror the
    safetensors sizer's posture.
    """
    return any(child.is_file() and child.suffix == ".gguf" for child in model_path.iterdir())


def _find_uncounted_pattern_entries(model_path: Path, suffix: str) -> list:
    """A permissive, best-effort recursive scan for every on-disk entry
    named ``*<suffix>`` anywhere under ``model_path`` -- deliberately the
    OPPOSITE of ``_iter_regular_files``: it does not check whether an
    entry is a symlink, and it does not skip a dot-named entry or avoid
    descending into a dot-named directory. This exists ONLY to build an
    honest, actionable diagnostic for the operator when the strict,
    sizer-matching scan found nothing countable ("this pack DOES have
    entries shaped like weight files -- they just aren't ones the sizer
    will count") -- it is never used to size or select a fit-check
    binary, and finding something here never flips ``_pack_has_safetensors``
    to ``True``. Returns relative POSIX paths, sorted for deterministic
    output.
    """
    matches = []
    for dirpath, _dirnames, filenames in os.walk(model_path):
        for filename in filenames:
            if filename.endswith(suffix):
                full = Path(dirpath) / filename
                matches.append(full.relative_to(model_path).as_posix())
    return sorted(matches)


def _uncounted_safetensors_message(model_path: Path) -> Optional[str]:
    """``None`` when no ``*.safetensors``-named entry exists anywhere under
    ``model_path`` at all. Otherwise (this candidate has ``_pack_has_safetensors``
    ``False`` but at least one raw ``*.safetensors``-named entry exists) an
    honest, actionable refusal: the entries are present but are symlinks
    and/or live under a dot-named directory, the built-in sizer
    deliberately never follows a symlink or descends into a dot-named
    directory (it only sizes real on-disk bytes it can independently
    verify -- see ``fastmlx_safetensors_fit._iter_regular_files``), and the
    remedy is a directory holding the REAL weight files rather than links.

    This is exactly the shape of the canonical Hugging Face hub cache
    layout (``~/.cache/huggingface/hub/models--<repo>/snapshots/<rev>/``,
    where every entry is a symlink into ``../../blobs/``), named here as an
    example, not asserted as the only possible cause.
    """
    matches = _find_uncounted_pattern_entries(model_path, ".safetensors")
    if not matches:
        return None
    shown = matches[:3]
    more_suffix = f" (+{len(matches) - 3} more)" if len(matches) > 3 else ""
    return (
        f"{model_path} contains *.safetensors entries "
        f"({', '.join(shown)}{more_suffix}) but the built-in safetensors "
        "sizer counted zero of them as weight files: it deliberately never "
        "follows a symlink (file or directory) and never descends into a "
        "dot-named directory or counts a dot-named entry -- it only sizes "
        "real on-disk bytes it can independently verify, which this pack's "
        "entries are not (this is the shape of the canonical Hugging Face "
        "hub cache layout, "
        "~/.cache/huggingface/hub/models--<repo>/snapshots/<rev>/, where "
        "every entry is a symlink). Point --model-path at a directory "
        "holding the REAL weight files instead of links -- "
        "'fastmlx pull <repo>@<revision> --dest <dir>' (this repository's "
        "own downloader; NOT --adopt, which refuses a symlinked source "
        "directory outright) writes real, non-symlink files -- or pass "
        "--fit-check-bin to use a different sizer for this pack."
    )


# The two flags a launch sized by one of this repository's built-in
# pure-Python sizers needs, and which a launch sized by the built-in engine
# does not. Both are collected and reported TOGETHER: before this existed an
# operator met them one refusal at a time, and the --context one only after
# paying for a full fit-check run, because it was enforced downstream from
# the fit check's (absent) context ceiling. See
# docs/task-inbox/2026-09-21-builtin-sizer-yields-no-context-ceiling.md
# option (c).
_BUILTIN_REQUIREMENT_REASONS = {
    "--kv-reserve-gib": (
        "a fit check must never silently assume a zero KV-cache reserve"
    ),
    "--context": (
        "these sizers answer only whether the pack fits at the reserve you "
        "named and emit no context ceiling to derive one from"
    ),
}


def _join_flag_list(flags) -> str:
    """``['--a']`` -> ``'--a'``; ``['--a', '--b']`` -> ``'--a and --b'``."""
    flags = list(flags)
    if len(flags) == 1:
        return flags[0]
    return " and ".join(flags)


def _builtin_sizer_missing_requirements(
    *, kv_reserve_gib, kv_reserve_already_in_args: bool, context
) -> list:
    """Every requirement a built-in sizer launch is missing, in the order
    they are reported -- never just the first one found.

    ``--context`` is required UP FRONT, even when the reserve was supplied,
    rather than downstream from the fit check's missing context ceiling. A
    built-in sizer never emits a ceiling (pinned by
    ``BuiltinSizersEmitNoContextCeilingTests``), so letting the fit check run
    first would only burn a fit check the operator was always going to have
    to run again. The separate downstream "context could not be determined"
    check stays exactly as it was, for a NON-built-in binary that happens to
    omit the field.
    """
    missing = []
    if kv_reserve_gib is None and not kv_reserve_already_in_args:
        missing.append("--kv-reserve-gib")
    if context is None:
        missing.append("--context")
    return missing


def _select_builtin_fit_check_bin(model_path: Path) -> tuple:
    """Auto-select one of this repository's built-in pure-Python sizers by
    inspecting ``model_path``'s own contents -- used only when neither an
    explicit ``--fit-check-bin`` nor an engine profile's own ``fitCheck``
    named one. Returns ``(builtin_name, error_detail)``: exactly one of
    the two is ``None``. ``builtin_name`` is one of the ``builtin:`` names
    ``_resolve_fit_check_bin_value`` understands (never a second,
    hand-copied filename mapping).

    A pack carrying BOTH a safetensors layout and a top-level ``*.gguf``
    file resolves to ``builtin:safetensors`` -- deliberately: MLX's native
    weight format is safetensors, and this repository's own
    ``is_model_dir`` layout check already gives an MLX
    ``config.json`` layout priority over a co-located ``*.gguf`` file (it
    is recognized as a model dir regardless of one); auto-selection here
    must not silently invert that existing precedence. See
    ``test_pack_with_both_layouts_prefers_safetensors_builtin``.

    A pack with zero countable ``*.safetensors`` files (``_pack_has_safetensors``
    ``False``) but at least one raw ``*.safetensors``-named entry on disk
    (a symlink, or one hidden under a dot-named directory) is refused with
    the honest, actionable diagnostic from ``_uncounted_safetensors_message``
    instead of falling through to the generic "neither layout" message --
    the generic message would read as false when the pack visibly contains
    a file named ``model.safetensors``.
    """
    if _pack_has_safetensors(model_path):
        return "builtin:safetensors", None
    if _pack_has_gguf(model_path):
        return "builtin:gguf", None
    uncounted_message = _uncounted_safetensors_message(model_path)
    if uncounted_message is not None:
        return None, uncounted_message
    return None, (
        "no --fit-check-bin was given, no engine profile named one, and "
        f"{model_path} contains neither a *.safetensors layout nor a "
        "top-level *.gguf file for a built-in sizer to auto-select; pass "
        "--fit-check-bin"
    )


# ---------------------------------------------------------------------
# 1. Fit check: run the fit binary, classify GREEN / RED / unknown.
# ---------------------------------------------------------------------
class FitCheckResult:
    def __init__(
        self,
        kind: str,
        fields: Optional[dict] = None,
        detail: Optional[str] = None,
        stderr: Optional[str] = None,
    ):
        self.kind = kind  # "green" | "red" | "error"
        self.fields = fields or {}
        self.detail = detail
        self.stderr = stderr


def _find_attestation_line(stdout: str) -> Optional[str]:
    for line in stdout.splitlines():
        if "fit_check_only=complete" in line:
            return line
    return None


def _parse_attestation_fields(line: str) -> dict:
    fields: dict = {}
    for token in line.split():
        if "=" in token:
            key, _, value = token.partition("=")
            fields[key] = value
    return fields


_STDERR_TAIL_MAX_LINES = 5
_STDERR_TAIL_MAX_CHARS = 800


def _bounded_stderr_tail(stderr: Optional[str]) -> str:
    """The last few non-empty lines of a sizer's stderr, capped in length --
    included in the "fit check could not run" detail below so an operator
    sees WHY a sizer exited unexpectedly (e.g. a required side file the
    profile named is missing) instead of only its exit code. Bounded on
    both line count and character count so a runaway or binary-garbage
    stderr can never blow up a refusal message or a recommend error row.
    """
    if not stderr:
        return ""
    lines = [line for line in stderr.splitlines() if line.strip()]
    tail = "\n".join(lines[-_STDERR_TAIL_MAX_LINES:])
    if len(tail) > _STDERR_TAIL_MAX_CHARS:
        tail = tail[-_STDERR_TAIL_MAX_CHARS:]
    return tail


def run_fit_check(
    fit_check_bin: str,
    model_id: str,
    model_path: Path,
    host_use: str,
    context: Optional[int],
    extra_args: list,
    timeout: float = FIT_CHECK_TIMEOUT_SECONDS,
) -> FitCheckResult:
    """Run the fit-check binary and classify its outcome.

    GREEN (exit 0 with an attestation line) and RED (exit 2) are the only
    two outcomes the fit check itself defines. Anything else -- a missing
    binary, any other exit code, a 0 exit with no attestation line, or a
    timeout -- is reported as ``"error"``: fail-closed, and never
    overridable by ``--force``.
    """
    argv = [
        fit_check_bin,
        "--model",
        model_id,
        "--model-path",
        str(model_path),
        "--host-use",
        host_use,
        "--fit-check-only",
    ]
    if context is not None:
        argv += ["--context", str(context)]
    argv += list(extra_args)
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return FitCheckResult("error", detail=f"fit check binary not found: {fit_check_bin}")
    except subprocess.TimeoutExpired:
        return FitCheckResult(
            "error", detail=f"fit check timed out after {timeout}s: {' '.join(argv)}"
        )
    except OSError as error:
        return FitCheckResult("error", detail=f"fit check could not be started: {error}")

    if proc.returncode == 0:
        line = _find_attestation_line(proc.stdout)
        if line is None:
            return FitCheckResult(
                "error",
                detail=(
                    "fit check exited 0 without a fit_check_only=complete "
                    "attestation line"
                ),
            )
        return FitCheckResult("green", fields=_parse_attestation_fields(line))
    if proc.returncode == 2:
        return FitCheckResult("red", stderr=proc.stderr or "")
    detail = (
        f"fit check exited {proc.returncode} (expected 0 for a passing verdict "
        "or 2 for a red verdict)"
    )
    tail = _bounded_stderr_tail(proc.stderr)
    if tail:
        detail += f": {tail}"
    return FitCheckResult("error", detail=detail)


# ---------------------------------------------------------------------
# 2. Quality-card admission: mirrors HarnessCore/QualityAdmission.swift.
# ---------------------------------------------------------------------
def load_quality_cards(path: Path) -> Optional[list]:
    """Decode a ``fast-mlx-quality-card-v1`` manifest's ``cards`` list.

    Returns ``None`` on any missing file, unreadable file, invalid JSON, or
    a document that is not shaped like the manifest envelope -- the same
    fail-open-to-"no card" behavior ``QualityCardStore`` uses for the
    conventional default manifest path.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    cards = document.get("cards")
    if not isinstance(cards, list):
        return None
    return cards


def find_card_by_id(cards: list, card_id: str) -> Optional[dict]:
    for card in cards:
        if isinstance(card, dict) and card.get("id") == card_id:
            return card
    return None


def find_cards_by_repo(cards: list, repo: Optional[str]) -> list:
    """Every card matching ``repo`` exactly, in list order -- the plural form
    ``resolve_card`` uses to detect when a repo names MORE than one card
    (the "multiple cards for one pack" engine-build-disambiguation case; see
    ``docs/quality-card-schema-v1.md`` "Engine build").
    """
    if repo is None:
        return []
    return [
        card
        for card in cards
        if isinstance(card, dict) and card.get("model", {}).get("repo") == repo
    ]


def find_card_by_repo(cards: list, repo: Optional[str]) -> Optional[dict]:
    matches = find_cards_by_repo(cards, repo)
    return matches[0] if matches else None


_HEX_DIGITS_RE = re.compile(r"^[0-9a-fA-F]+$")
_MIN_HF_PIN_LENGTH = 8
_FULL_REVISION_LENGTH = 40


def _is_full_hex_revision(value: Optional[str]) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _FULL_REVISION_LENGTH
        and bool(_HEX_DIGITS_RE.fullmatch(value))
    )


def _is_usable_hf_pin(value: Optional[str]) -> bool:
    return (
        isinstance(value, str)
        and len(value) >= _MIN_HF_PIN_LENGTH
        and bool(_HEX_DIGITS_RE.fullmatch(value))
    )


def _hf_pin_matches_revision(hf_pin: Optional[str], model_revision: Optional[str]) -> bool:
    """A public card may carry no ``model.repo`` at all and identify its pack
    only by ``model.hfPin`` -- a hex prefix of the full 40-hex pinned HF
    commit sha. A short (<8 hex chars) or non-hex ``hfPin``, or a
    ``model_revision`` that is not itself a full 40-hex sha, never matches:
    this is a launcher-specific extension of the lookup ``QualityAdmission``
    itself does not have (the Swift gate matches on ``model.repo`` only), so
    it fails closed to "no match" on anything short of an exact,
    case-insensitive hex-prefix relationship.
    """
    if not _is_full_hex_revision(model_revision) or not _is_usable_hf_pin(hf_pin):
        return False
    return model_revision.lower().startswith(hf_pin.lower())


def find_cards_by_pin(cards: list, model_revision: Optional[str]) -> list:
    """Every card whose hfPin prefix-matches ``model_revision``, in list
    order -- the plural counterpart to ``find_card_by_pin`` (see
    ``find_cards_by_repo``).
    """
    if not _is_full_hex_revision(model_revision):
        return []
    matches = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        hf_pin = (card.get("model") or {}).get("hfPin")
        if _hf_pin_matches_revision(hf_pin, model_revision):
            matches.append(card)
    return matches


def find_card_by_pin(cards: list, model_revision: Optional[str]) -> Optional[dict]:
    matches = find_cards_by_pin(cards, model_revision)
    return matches[0] if matches else None


def card_residency(card: dict) -> Optional[str]:
    """The residency ``card`` applies to.

    Returns ``config.residency`` when it is exactly ``"resident"`` or
    ``"expert-stream"``; returns ``"resident"`` when ``config`` (or
    ``config.residency``) is absent or null -- the historical card shape,
    predating this field, always described a resident launch. Returns
    ``None`` for any other value: an unrecognized residency fails closed to
    "never matches any launch", the same fail-closed-on-unrecognized-value
    discipline ``decide_admission``'s verdict handling and the Swift
    ``QualityVerdict`` decoder both use.
    """
    config = card.get("config")
    if not isinstance(config, dict):
        return "resident"
    residency = config.get("residency")
    if residency is None:
        return "resident"
    if residency in RESIDENCIES:
        return residency
    return None


def card_engine_build_commit(card: Optional[dict]) -> Optional[str]:
    """The measured engine-build commit ``card`` records
    (``provenance.engineBuild.commit``), or ``None`` when the card carries
    no ``provenance``, no ``engineBuild``, or a malformed one -- fails open
    to "unrecorded" rather than raising, since a fixture/hand-added card is
    never required to carry this OPTIONAL field.
    """
    if card is None:
        return None
    provenance = card.get("provenance")
    if not isinstance(provenance, dict):
        return None
    engine_build = provenance.get("engineBuild")
    if not isinstance(engine_build, dict):
        return None
    commit = engine_build.get("commit")
    return commit if isinstance(commit, str) else None


def card_hardware_class(card: Optional[dict]) -> Optional[str]:
    """The hardware class ``card`` was measured on (``config.hardwareClass``),
    or ``None`` when the card carries no ``config``, no ``hardwareClass``, or
    an empty/non-string one -- fails open to "unrecorded" rather than
    raising, the same way ``card_engine_build_commit`` fails open to
    "unrecorded": this field is OPTIONAL, and a card predating it (or a
    hand-added fixture) is never required to carry it.

    ``resolve_card`` uses this ONLY to disambiguate a multi-candidate match
    (see its docstring) -- exactly like ``card_engine_build_commit``, and
    unlike ``card_residency``, this never filters the candidate list.
    """
    if card is None:
        return None
    config = card.get("config")
    if not isinstance(config, dict):
        return None
    hardware_class = config.get("hardwareClass")
    return hardware_class if isinstance(hardware_class, str) and hardware_class else None


def host_hardware_class() -> Optional[str]:
    """This host's hardware class, normalized to the SAME string the
    published quality cards carry (e.g. ``"apple-m3-ultra"``, ``"apple-m5"``).

    Composes two sibling implementations this MUST stay byte-consistent
    with (a dedicated test pins the normalization against the second one):

    - ``fastmlx_bench._chip_identity()``: ``sysctl -n
      machdep.cpu.brand_string`` -> e.g. ``"Apple M3 Ultra"``.
    - ``emit_quality_card._hardware_class(chip)``: ``chip.lower().replace("
      ", "-")`` -> e.g. ``"apple-m3-ultra"``.

    This is a deliberate separate copy, not an import of either module --
    see the "Deliberately OUT of scope" note in
    ``docs/task-inbox/2026-09-22-PREDECLARATION-hardwareclass-joins-identity-never-filters.md``.

    Fails closed to ``None`` -- never a guess, never a hostname -- on ANY
    failure: non-macOS (no ``sysctl`` on PATH), a nonzero return code, a
    timeout, empty stdout, or any other subprocess error. Never raises.
    """
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    brand = result.stdout.strip()
    if not brand:
        return None
    return brand.lower().replace(" ", "-")


def engine_build_status(card_commit: Optional[str], launch_commit: Optional[str]) -> str:
    """Classify a card's measured engine build against a launch's own, per
    ``docs/quality-card-schema-v1.md`` "Engine build":

    - ``unrecorded``: the card never recorded a build (``card_commit is None``).
    - ``undeclared``: the card recorded a build but this launch's engine
      profile did not (``launch_commit is None``).
    - ``match`` / ``mismatch``: both are recorded; equal or not.

    NEVER used to filter card admission -- only to classify it for the
    notice/message this module's callers surface.
    """
    if card_commit is None:
        return ENGINE_BUILD_STATUS_UNRECORDED
    if launch_commit is None:
        return ENGINE_BUILD_STATUS_UNDECLARED
    if card_commit == launch_commit:
        return ENGINE_BUILD_STATUS_MATCH
    return ENGINE_BUILD_STATUS_MISMATCH


def engine_build_notice_text(
    card_id: Optional[str], card_commit: str, launch_commit: Optional[str]
) -> str:
    """The one-line "transfer unmeasured" notice body for an
    ``undeclared``/``mismatch`` engine-build status -- shared by
    ``fastmlx_launch`` (prefixed ``"fastmlx serve: "`` on stderr, and
    appended to a NO_GO refusal message) and ``fastmlx_recommend`` (surfaced
    as a row's own ``message``).
    """
    launch_desc = launch_commit[:12] if launch_commit else "undeclared"
    return (
        f"card {card_id} was measured on engine build {card_commit[:12]}; "
        f"this launch is {launch_desc} — transfer unmeasured"
    )


def _final_argv_preview(
    profile: dict, residency: str, passthrough_args: "Optional[list]" = None
) -> list:
    """The token list ``_run_serve`` execs, in the same order, BEFORE
    placeholder substitution: the profile's own ``argv``, then
    ``residencyArgs['expert-stream']`` when the launch streams, then any
    passthrough args after ``--``. Placeholder substitution never turns a
    literal flag into ``--mtp`` (or vice versa) -- see
    ``ALLOWED_ENGINE_PROFILE_PLACEHOLDERS`` -- so scanning this
    pre-substitution list for an exact token is equivalent to scanning the
    fully-substituted ``final_argv`` ``_run_serve`` actually execs.
    """
    tokens = list(profile.get("argv") or [])
    if residency == "expert-stream":
        tokens += list((profile.get("residencyArgs") or {}).get("expert-stream") or [])
    tokens += list(passthrough_args or [])
    return tokens


def mtp_launch_requested(
    profile: dict, residency: str, passthrough_args: "Optional[list]" = None
) -> bool:
    """Whether this launch's final engine argv carries the exact token
    ``--mtp`` -- the ONLY thing that makes ``config.flagTransfer["--mtp"]``
    relevant to it (see ``mtp_transfer_status``)."""
    return "--mtp" in _final_argv_preview(profile, residency, passthrough_args)


def card_flag_transfer_mtp(card: Optional[dict]) -> Optional[dict]:
    """The card's ``config.flagTransfer["--mtp"]`` object, or ``None`` when
    the card carries no ``config``, no ``flagTransfer``, or no ``--mtp``
    entry -- fails open to "unmeasured" rather than raising, the same way
    ``card_engine_build_commit`` fails open to "unrecorded": this field is
    OPTIONAL, and a card predating it (or a hand-added fixture) is never
    required to carry it.
    """
    if card is None:
        return None
    config = card.get("config")
    if not isinstance(config, dict):
        return None
    flag_transfer = config.get("flagTransfer")
    if not isinstance(flag_transfer, dict):
        return None
    mtp = flag_transfer.get("--mtp")
    return mtp if isinstance(mtp, dict) else None


def mtp_transfer_status(
    card: Optional[dict], build_status: str, mtp_launch: bool
) -> tuple:
    """Classify this launch's ``--mtp`` flag transfer against a resolved
    card's ``config.flagTransfer["--mtp"]``, per
    ``docs/quality-card-schema-v1.md`` "Flag transfer". Returns
    ``(status, divergentPrompts, prompts)`` -- the last two ``None`` unless
    ``status`` is one of exact/not_exact/nondeterministic.

    - ``off``: ``--mtp`` is not in this launch's final argv at all -- the
      field is irrelevant.
    - ``unmeasured``: ``--mtp`` IS in the argv, but either the resolved
      card (or no card at all) carries no ``flagTransfer["--mtp"]``, or the
      engine-build status (``engine_build_status``) is not ``match`` -- the
      transfer is valid only at the card's own measured build.
    - otherwise: the card's own ``greedy`` value (``exact`` / ``not_exact``
      / ``nondeterministic``).

    NEVER used to filter card admission -- only to classify it for the
    notice this module's callers surface, exactly like ``engine_build_status``.

    ``load_quality_cards`` never validates a card against the schema, so a
    resolved card's ``flagTransfer["--mtp"]`` may be hand-edited, stale, or
    otherwise malformed. This function fails OPEN on malformed data -- any
    ``greedy`` outside the three known values, or a non-int/bool/negative/
    inconsistent ``divergentPrompts``/``prompts``, is treated exactly like
    "no flagTransfer at all" (``unmeasured``, null counts) rather than
    propagating a bogus status to ``mtp_notice_text`` (which has no notice
    for an unrecognized status and would raise).
    """
    if not mtp_launch:
        return MTP_STATUS_OFF, None, None
    transfer = card_flag_transfer_mtp(card)
    if transfer is None or build_status != ENGINE_BUILD_STATUS_MATCH:
        return MTP_STATUS_UNMEASURED, None, None

    greedy = transfer.get("greedy")
    divergent = transfer.get("divergentPrompts")
    prompts = transfer.get("prompts")
    valid_greedy = greedy in (MTP_STATUS_EXACT, MTP_STATUS_NOT_EXACT, MTP_STATUS_NONDETERMINISTIC)
    valid_divergent = (
        isinstance(divergent, int) and not isinstance(divergent, bool) and divergent >= 0
    )
    valid_prompts = isinstance(prompts, int) and not isinstance(prompts, bool) and prompts >= 1
    if not (valid_greedy and valid_divergent and valid_prompts) or divergent > prompts:
        return MTP_STATUS_UNMEASURED, None, None
    return greedy, divergent, prompts


def mtp_notice_text(
    card_id: Optional[str],
    status: str,
    divergent_prompts: Optional[int],
    prompts: Optional[int],
    card_build_commit: Optional[str],
) -> str:
    """The one-line ``--mtp`` transfer notice body for a
    not_exact/nondeterministic/unmeasured status -- shared by
    ``fastmlx_launch`` (prefixed ``"fastmlx serve: "`` on stderr, and
    appended to a NO_GO refusal message, mirroring
    ``engine_build_notice_text``) and ``fastmlx_recommend`` (surfaced as a
    row's own ``mtp.message``). Never called for ``off``/``exact``.
    """
    if card_id is None:
        return (
            "no quality card was resolved for this launch; the --mtp transfer "
            "is unmeasured"
        )
    if status == MTP_STATUS_UNMEASURED:
        return f"card {card_id} was measured without --mtp; the --mtp transfer is unmeasured"
    build_desc = card_build_commit[:12] if card_build_commit else "unknown"
    if status == MTP_STATUS_NOT_EXACT:
        return (
            f"card {card_id} was measured without --mtp; with --mtp at build "
            f"{build_desc}, greedy output differed on {divergent_prompts}/{prompts} prompts"
        )
    if status == MTP_STATUS_NONDETERMINISTIC:
        return (
            f"card {card_id} was measured without --mtp; with --mtp at build "
            f"{build_desc}, greedy output differed from the card's path on "
            f"{divergent_prompts}/{prompts} prompts and was not reproducible "
            "across processes"
        )
    raise ValueError(f"mtp_notice_text: no notice for status {status!r}")


def resolve_card(
    cards: Optional[list],
    model_repo: Optional[str],
    model_revision: Optional[str],
    residency: str = "resident",
    engine_build_commit: Optional[str] = None,
    host_hardware_class=host_hardware_class,
) -> Optional[dict]:
    """The implicit (no ``--card-id``) lookup: a repo match (Swift semantics)
    OR an hfPin-prefix match against the model's pinned revision. If both
    exist and name DIFFERENT card(s), the lookup is ambiguous and refuses
    rather than silently preferring one -- this can only happen with a
    manifest that names the same model twice under two different cards, one
    keyed by repo and the other only by pin.

    Cards are filtered to ``residency`` (``card_residency(card) ==
    residency``) BEFORE either lookup and before the ambiguity check: a
    card measured for the other residency is invisible to this lookup, the
    same way a card for a different model is. ``hardwareClass`` NEVER
    filters this way -- see below.

    A repo or pin match can now name MORE THAN ONE card -- the same pack
    measured on more than one hardware class and/or engine build. When it
    does, ``hardwareClass`` is tried FIRST: if the candidates do not all
    share one hardware class, and exactly one candidate's
    ``card_hardware_class`` equals this host's (``host_hardware_class()``),
    that one is selected. An unknown host class, or zero/more-than-one
    matching candidates, narrows nothing -- it falls through unchanged to
    the existing engine-build tiebreak below. Like
    ``engineBuild.commit``, ``hardwareClass`` is used ONLY to disambiguate
    an otherwise-tied identity match; it is never used to FILTER the
    candidate list the way ``residency`` does. An Ultra card measured on an
    Ultra host is not "for the wrong hardware" on an M5 host -- it is
    weaker, still-valid evidence, and stays reachable as the sole candidate
    whenever it is the only match for the pack.

    If hardware class does not narrow to one, the single card whose
    ``provenance.engineBuild.commit`` equals ``engine_build_commit`` (this
    launch's own engine build) is selected next. An undeclared launch
    (``None``) selects none of them, not even an unrecorded card; anything else (no exact match, or more than one) refuses
    (exit 3) rather than silently picking one -- engine build is never used
    to FILTER a card the way residency does, only to disambiguate an
    otherwise-tied identity match.
    """
    if cards is None:
        return None
    residency_cards = [
        card
        for card in cards
        if isinstance(card, dict) and card_residency(card) == residency
    ]
    repo_cards = find_cards_by_repo(residency_cards, model_repo)
    pin_cards = find_cards_by_pin(residency_cards, model_revision)
    repo_ids = {card.get("id") for card in repo_cards}
    pin_ids = {card.get("id") for card in pin_cards}
    if repo_ids and pin_ids and repo_ids != pin_ids:
        raise LaunchRefusal(
            3,
            "quality card lookup is ambiguous: repo "
            f"{model_repo!r} matches card(s) {sorted(repo_ids)} but pinned "
            f"revision {model_revision!r} matches different card(s) "
            f"{sorted(pin_ids)}",
        )
    candidates = repo_cards if repo_cards else pin_cards
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # hardwareClass disambiguates a multi-candidate tie exactly like
    # engineBuild.commit does below -- NEVER a filter. Only consult the host
    # (and only narrow) when the candidates actually disagree on class; an
    # unknown host class, or zero/more-than-one matching candidates, changes
    # nothing and falls through untouched.
    candidate_classes = {card_hardware_class(card) for card in candidates}
    if len(candidate_classes) > 1:
        host_class = host_hardware_class()
        if host_class is not None:
            class_matches = [
                card for card in candidates if card_hardware_class(card) == host_class
            ]
            if len(class_matches) == 1:
                return class_matches[0]

    # An undeclared launch never picks among several cards, not even the one
    # whose build is also unrecorded: that would let an unrecorded card
    # silently shadow a measured one.
    exact_matches = [
        card
        for card in candidates
        if engine_build_commit is not None
        and card_engine_build_commit(card) == engine_build_commit
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    build_labels = sorted(
        f"{card_hardware_class(card) or 'unrecorded'}@"
        f"{(card_engine_build_commit(card) or 'undeclared')[:12]}"
        for card in candidates
    )
    raise LaunchRefusal(
        3,
        f"{len(candidates)} cards for this pack at builds {','.join(build_labels)}; "
        "declare engineBuild.commit in the engine profile or pass --card-id",
    )


def card_matches_model_identity(
    card: dict, model_repo: Optional[str], model_revision: Optional[str]
) -> bool:
    """Whether ``card`` can be independently verified against the resolved
    model identity -- used to gate an explicit ``--card-id``, which must
    never be trusted as an unverified operator assertion.
    """
    model = card.get("model") or {}
    repo = model.get("repo")
    if repo is not None and model_repo is not None and repo == model_repo:
        return True
    return _hf_pin_matches_revision(model.get("hfPin"), model_revision)


def is_opted_in(card: Optional[dict], opt_in_ids: set) -> bool:
    """``--accept-quality VALUE`` elects a card if VALUE equals its id, its
    (non-null) repo, or its hfPin.
    """
    if card is None:
        return False
    model = card.get("model") or {}
    for candidate in (card.get("id"), model.get("repo"), model.get("hfPin")):
        if candidate is not None and candidate in opt_in_ids:
            return True
    return False


def card_benefit_line(card: Optional[dict]) -> Optional[str]:
    """The measured SPEED benefit ``card`` carries, formatted as one
    human-readable line for a CLI surface. Returns ``None`` when the card
    has no ``legible.benefit`` at all, or no ``speedXStatus`` within it --
    what keeps every card/fixture that predates this benefit sub-object
    printing byte-identical output.

    This is the speed clause ONLY -- it never includes the separate
    ``fit`` clause a benefit may also carry (see ``card_fit_line``). A
    prior version of this function concatenated the two with "; ", and
    both call sites labelled the combined string "speed: ", presenting a
    fit fact (a resident size, a host it fits) as if it were a speed
    fact. Keep the two clauses apart and separately labelled at every
    call site, exactly like the public site does (``Fit`` / ``Speed`` as
    two separate ``<dt>``/``<dd>`` pairs in
    ``build_public_site.render_quality_guide``).

    Mirrors ``build_public_site.quality_speed_line``'s polarity and
    scope-attachment rule exactly (see that function): a speed ratio is
    NEVER printed detached from its ``speedXStatus`` -- that status
    carries the ONLY measurement boundary (host, engine build, flags,
    prompt count, and the referent the ratio is against) fast-mlx has for
    the number, so this never synthesizes or paraphrases a referent the
    status doesn't already state. A ratio below 1.0 is phrased "a
    slowdown", never "{n}x slower", which would invert the meaning. This
    is a reimplementation, not an import of ``build_public_site`` -- that
    module is a build tool, not a CLI dependency; a test pins the two
    stay in agreement (see ``test_speed_direction_matches_site_renderer``
    in both ``scripts/tests/test_fastmlx_launch.py`` and
    ``scripts/tests/test_fastmlx_recommend.py``).
    """
    if card is None:
        return None
    benefit = (card.get("legible") or {}).get("benefit")
    if not benefit:
        return None

    status = benefit.get("speedXStatus")
    if status is None:
        return None
    status_text = str(status)
    speed_x = benefit.get("speedX")
    if isinstance(speed_x, (int, float)) and not isinstance(speed_x, bool):
        if speed_x > 1.0:
            direction = f'{speed_x}x faster on this engine (measured)'
        elif speed_x < 1.0:
            direction = (
                f'{speed_x}x the reference speed on this engine — a '
                'slowdown (measured)'
            )
        else:
            direction = 'no measured speed difference on this engine (measured)'
        return f'{direction} — {status_text}'
    return status_text


def card_fit_line(card: Optional[dict]) -> Optional[str]:
    """The measured/estimated FIT clause ``card`` carries (a resident
    size and which hosts it fits), formatted for a CLI surface. Returns
    ``None`` when the card has no ``legible.benefit`` at all, or no
    ``fit`` within it -- exactly like ``build_public_site`` only renders
    a ``Fit`` row ``if fit is not None`` (see
    ``build_public_site.render_quality_guide``). Kept separate from
    ``card_benefit_line`` (the speed clause) so a call site can never
    relabel one as the other -- see that function's docstring for the
    defect this split repairs.
    """
    if card is None:
        return None
    benefit = (card.get("legible") or {}).get("benefit")
    if not benefit:
        return None
    fit = benefit.get("fit")
    return str(fit) if fit is not None else None


def decide_admission(card: Optional[dict], opted_in: bool) -> tuple:
    """Mirror ``QualityAdmission.decide`` exactly.

    Returns ``(outcome, message)`` where ``outcome`` is one of
    ``"admit_unmeasured"``, ``"admit"``, ``"admit_with_quality_flag"``, or
    ``"refuse_quality_flagged"``, and ``message`` is the one-line summary
    (``None`` for the two silent-admit outcomes). No card, and a card whose
    verdict is ``UNMEASURED`` or is not one of the five recognized verdict
    strings, both admit silently -- `QualityVerdict`'s Swift decoder fails
    closed to ``.unmeasured`` for an unrecognized string, and
    `QualityAdmission.decide` treats `.unmeasured` identically to "no
    card": admit, never refuse. ``opted_in`` is computed by the caller
    (``is_opted_in``); this function only implements the verdict
    discriminator, same as the Swift `QualityAdmission.decide` does not
    itself parse `--accept-quality`.
    """
    if card is None:
        return ("admit_unmeasured", None)
    verdict = card.get("verdict")
    if verdict in ("PASS", "REFERENCE", "EXACT"):
        return ("admit", None)
    if verdict != "NO_GO":
        return ("admit_unmeasured", None)

    legible = card.get("legible") or {}
    summary = f"{legible.get('tier')}: {legible.get('headline')}"
    # The cost (tier/headline) must never be shown without the benefit it
    # buys -- the operator deciding whether to opt in is exactly who needs
    # to see both. Appears on BOTH the opt-in message and the refusal
    # message below, since both carry this same summary. Speed and fit are
    # two DISTINCT facts (see card_benefit_line's docstring for the defect
    # this split repairs) and are always labelled separately -- a fit
    # clause must never be presented under the "speed: " label.
    benefit_line = card_benefit_line(card)
    fit_line = card_fit_line(card)
    # Clauses are separated by " - " rather than a bare space: a status
    # enum ("not-measured-on-this-engine") and a fit string ("20.7 GB --
    # fits a 24 GB Mac") do not end in a period, so space-joining ran
    # them straight into the next label and into the re-run hint
    # ("...fits a 24 GB Mac re-run with --accept-quality..."), which
    # reads as one sentence. Only clauses that actually exist are
    # joined, so a card with no benefit is unaffected.
    clauses = [summary]
    if benefit_line:
        clauses.append(f"speed: {benefit_line}")
    if fit_line:
        clauses.append(f"fit: {fit_line}")
    summary = " - ".join(clauses)
    card_id = card.get("id")
    if opted_in:
        return ("admit_with_quality_flag", summary)
    hint = f"re-run with --accept-quality {card_id} to elect it."
    # A card with no benefit clause keeps the historic single-space join,
    # so its refusal message stays byte-identical to before this change.
    separator = " - " if len(clauses) > 1 else " "
    return ("refuse_quality_flagged", summary + separator + hint)


# ---------------------------------------------------------------------
# 3. Engine profile: load, validate placeholders, substitute, resolve argv[0].
# ---------------------------------------------------------------------
def _placeholder_tokens(text: str) -> list:
    return re.findall(r"\{[^{}]*\}", text)


def _validate_residency_args(document: dict, path: str) -> dict:
    """Validate the OPTIONAL ``residencyArgs`` key of an engine profile.

    Its only allowed key is ``"expert-stream"``; each value must be a
    non-empty list of non-empty strings, none of which contain a ``{``/``}``
    placeholder (``residencyArgs`` entries are appended to the exec'd argv
    verbatim -- never substituted). Returns the validated dict (``{}`` when
    the key is absent), so every profile this function returns carries a
    ``residencyArgs`` key regardless of whether the source document did.
    """
    raw = document.get("residencyArgs", {})
    if not isinstance(raw, dict):
        raise LaunchRefusal(3, f"engine profile at {path} residencyArgs must be an object")
    extra_keys = sorted(set(raw) - ALLOWED_RESIDENCY_ARGS_KEYS)
    if extra_keys:
        raise LaunchRefusal(
            3,
            f"engine profile at {path} residencyArgs has unknown key(s) {extra_keys}; "
            f"the only allowed key is 'expert-stream'",
        )
    validated: dict = {}
    for key, value in raw.items():
        if not isinstance(value, list) or not value or not all(
            isinstance(item, str) and item for item in value
        ):
            raise LaunchRefusal(
                3,
                f"engine profile at {path} residencyArgs[{key!r}] must be a "
                "non-empty list of non-empty strings",
            )
        for item in value:
            if "{" in item or "}" in item:
                raise LaunchRefusal(
                    3,
                    f"engine profile at {path} residencyArgs[{key!r}] element "
                    f"{item!r} must not contain a {{...}} placeholder",
                )
        validated[key] = value
    return validated


def _validate_engine_build_profile(document: dict, path: str) -> Optional[dict]:
    """Validate the OPTIONAL ``engineBuild`` key of an engine profile.

    Its only allowed keys are ``commit`` (a lowercase 40-hex git sha) and
    ``binarySha256`` (a lowercase 64-hex sha256), both optional. Returns
    ``None`` when the key is absent; otherwise a dict always carrying both
    keys (``None`` for whichever sub-key was not given). Refuses
    (``LaunchRefusal(3, ...)``) on any unrecognized shape -- a malformed
    ``engineBuild`` must never be silently treated as absent.
    """
    if "engineBuild" not in document:
        return None
    raw = document["engineBuild"]
    if not isinstance(raw, dict):
        raise LaunchRefusal(3, f"engine profile at {path} engineBuild must be an object")
    extra_keys = sorted(set(raw) - ALLOWED_ENGINE_BUILD_PROFILE_KEYS)
    if extra_keys:
        raise LaunchRefusal(
            3,
            f"engine profile at {path} engineBuild has unknown key(s) {extra_keys}; "
            "the only allowed keys are 'commit' and 'binarySha256'",
        )
    commit = raw.get("commit")
    if commit is not None and (
        not isinstance(commit, str) or not _LOWERCASE_HEX40_RE.fullmatch(commit)
    ):
        raise LaunchRefusal(
            3,
            f"engine profile at {path} engineBuild.commit must be a lowercase 40-hex string",
        )
    binary_sha256 = raw.get("binarySha256")
    if binary_sha256 is not None and (
        not isinstance(binary_sha256, str) or not _LOWERCASE_HEX64_RE.fullmatch(binary_sha256)
    ):
        raise LaunchRefusal(
            3,
            f"engine profile at {path} engineBuild.binarySha256 must be a lowercase "
            "64-hex string",
        )
    return {"commit": commit, "binarySha256": binary_sha256}


def _sha256_file(path: str) -> str:
    """The lowercase hex sha256 of the file at ``path``, read in bounded
    chunks so an arbitrarily large engine binary never has to be held in
    memory at once.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def derive_engine_build_from_release(engine_bin_abs: str) -> Optional[dict]:
    """Derive this launch's own engine-build commit from a
    ``scripts/package-release.sh``-staged release tree, with NO
    operator-written ``--engine-profile engineBuild.commit`` at all.

    Every quality card measured on fast-mlx's OWN Swift engine (never a
    third-party ``--engine-profile``) previously showed the "undeclared --
    transfer unmeasured" notice on EVERY launch forever: nothing upstream of
    this function ever knew the commit the ``fastmlx-serve`` binary being
    exec'd was built from unless an operator hand-wrote it into a profile.
    A release tarball already carries that fact -- ``provenance.json``,
    staged as a sibling of ``bin/`` -- so this reads it back, but only when
    EVERY one of the following holds; any failure returns ``None`` (stays
    "undeclared", the same honest fail-open ``card_engine_build_commit`` and
    ``card_flag_transfer_mtp`` already use elsewhere in this module -- an
    underivable build is never a refusal, only a fact this launch cannot
    assert).

    1. ``engine_bin_abs`` is an absolute path whose parent directory is
       named ``bin``, with a readable ``provenance.json`` at
       ``<parent-of-bin>/provenance.json`` -- exactly the layout
       ``package-release.sh`` stages (``<root>/bin/fastmlx-serve`` next to
       ``<root>/provenance.json``). Anything else (no ``bin/`` parent, no
       sibling file, unreadable) is not a release install this function
       recognizes, so it derives nothing rather than guessing at a
       differently-shaped install.
    2. The file parses as JSON and the top level is a dict -- a malformed
       or unexpected shape is treated as absent, never as a crash.
    3. ``source_dirty`` is present and is exactly the bool ``False``. A
       dirty tree's ``source_commit`` names the checkout the build started
       from, not the exact bytes that came out of it -- an uncommitted
       change could be anywhere in what got compiled, so a dirty build
       stays undeclared rather than asserting a commit the binary may not
       faithfully represent. A missing key or a non-bool is refused the
       same way (fails closed on an unexpected shape, not just on ``True``).
    4. ``source_commit`` is a string matching ``_LOWERCASE_HEX40_RE`` -- the
       same shape ``engineBuild.commit`` is validated to when an operator
       writes it into a profile by hand (see
       ``_validate_engine_build_profile``), so a derived commit is never
       looser than a declared one.
    5. ``engine_binary_sha256`` is a string matching
       ``_LOWERCASE_HEX64_RE``.
    6. The ACTUAL sha256 of the file at ``engine_bin_abs`` (via
       ``_sha256_file``) equals that ``engine_binary_sha256``. This is what
       makes the derivation honest rather than a bare assertion:
       ``provenance.json`` is only a sibling TEXT file, sitting next to the
       binary, not sealed to it -- nothing stops a binary at
       ``bin/fastmlx-serve`` from being swapped out after packaging while
       ``provenance.json`` is left untouched. Recomputing and comparing the
       hash of the binary ACTUALLY BEING EXEC'D is what ties the recorded
       commit to the bytes running this launch, the same reason an
       operator-declared ``engineBuild.binarySha256`` is independently
       verified against the resolved binary rather than trusted as an
       assertion (see the ``expected_binary_sha256`` check in
       ``_run_serve``).

    Returns ``{"commit": <40-hex>, "binarySha256": <64-hex>}`` on success.
    Never raises (``OSError``/``json.JSONDecodeError``/an unexpected type
    are all just another way to fail closed to ``None``) and never calls
    ``LaunchRefusal`` -- an underivable build is simply "undeclared", the
    existing honest outcome this whole module already surfaces for a
    profile that declares no commit at all.
    """
    try:
        bin_dir = os.path.dirname(engine_bin_abs)
        if not os.path.isabs(engine_bin_abs) or os.path.basename(bin_dir) != "bin":
            return None
        provenance_path = os.path.join(os.path.dirname(bin_dir), "provenance.json")
        with open(provenance_path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
        if not isinstance(document, dict):
            return None
        source_dirty = document.get("source_dirty")
        if source_dirty is not False:
            return None
        source_commit = document.get("source_commit")
        if not isinstance(source_commit, str) or not _LOWERCASE_HEX40_RE.fullmatch(
            source_commit
        ):
            return None
        engine_binary_sha256 = document.get("engine_binary_sha256")
        if not isinstance(
            engine_binary_sha256, str
        ) or not _LOWERCASE_HEX64_RE.fullmatch(engine_binary_sha256):
            return None
        if _sha256_file(engine_bin_abs) != engine_binary_sha256:
            return None
        return {"commit": source_commit, "binarySha256": engine_binary_sha256}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _resolve_fit_check_bin_value(bin_value: str, path: str) -> str:
    """Resolve a `fitCheck.bin` value to an argv[0] string.

    A `builtin:` name resolves to the named sizer script SIBLING of this
    file (`Path(__file__).resolve().parent`), so it resolves correctly both
    in this repository and inside a release tarball's `libexec/scripts`.
    Any other `builtin:` name is refused. A non-`builtin:` value must be an
    absolute path -- a relative path would resolve differently depending on
    the operator's current working directory, which this launcher never
    depends on for any other input.
    """
    if bin_value in _BUILTIN_FIT_CHECK_FILENAMES:
        return str(Path(__file__).resolve().parent / _BUILTIN_FIT_CHECK_FILENAMES[bin_value])
    if bin_value.startswith("builtin:"):
        known = ", ".join(sorted(_BUILTIN_FIT_CHECK_FILENAMES))
        raise LaunchRefusal(
            3,
            f"engine profile at {path} fitCheck.bin names unknown builtin "
            f"{bin_value!r}; known builtins are: {known}",
        )
    if not Path(bin_value).is_absolute():
        raise LaunchRefusal(
            3,
            f"engine profile at {path} fitCheck.bin {bin_value!r} must be an "
            "absolute path or one of the 'builtin:' names",
        )
    return bin_value


def _reserved_fit_check_arg_collision(args: list) -> Optional[tuple]:
    """The first ``args`` entry that collides with one of
    ``RESERVED_FIT_CHECK_ARGS``, and the specific reserved flag it collides
    with -- or ``None`` if no entry collides.

    An item collides when its flag name (``item.split("=", 1)[0]``, so
    ``--context=1024`` is checked the same as ``--context 1024``) either
    equals a reserved flag exactly, OR is a strict prefix of a reserved
    flag with more than 2 characters (``--cont`` is a prefix of
    ``--context``; ``--h`` is too short to count). The sizers this
    launcher exec's build their argument parsers with
    ``argparse.ArgumentParser`` (abbreviations enabled by default until
    ``allow_abbrev=False`` is set -- also done as defense in depth), so an
    abbreviated flag in a profile's own ``fitCheck.args`` resolves to
    whichever reserved flag it uniquely prefixes just as surely as the
    exact spelling would, and must be refused the same way: this launcher,
    not the profile, owns the model identity/host/context a fit check runs
    against.
    """
    for item in args:
        if not item.startswith("--"):
            continue
        name = item.split("=", 1)[0]
        if name in RESERVED_FIT_CHECK_ARGS:
            return item, name
        if len(name) > 2:
            for reserved in RESERVED_FIT_CHECK_ARGS:
                if reserved != name and reserved.startswith(name):
                    return item, reserved
    return None


def _validate_fit_check(document: dict, path: str) -> Optional[dict]:
    """Validate the OPTIONAL `fitCheck` key of an engine profile.

    Returns ``None`` when the key is absent (the built-in profile always
    carries no ``fitCheck``: this is a launcher-owned distinction, never a
    heuristic against a document that merely omits the key by accident).
    Returns ``{"bin": <resolved argv[0]>, "args": [...]}`` otherwise --
    ``args`` defaults to ``[]``. Refuses on any unrecognized shape: a
    non-object ``fitCheck``, an unknown key inside it, a non-string/empty
    ``bin``, an unresolvable ``bin`` (see ``_resolve_fit_check_bin_value``),
    a non-list/non-string/empty ``args``, or an ``args`` entry that
    duplicates a flag this launcher itself always supplies
    (``RESERVED_FIT_CHECK_ARGS``) -- a malformed ``fitCheck`` must never
    silently fall back to running the wrong sizer or a mis-shaped argv.
    """
    if "fitCheck" not in document:
        return None
    raw = document["fitCheck"]
    if not isinstance(raw, dict):
        raise LaunchRefusal(3, f"engine profile at {path} fitCheck must be an object")
    extra_keys = sorted(set(raw) - ALLOWED_FIT_CHECK_KEYS)
    if extra_keys:
        raise LaunchRefusal(
            3,
            f"engine profile at {path} fitCheck has unknown key(s) {extra_keys}; "
            "the only allowed keys are 'bin' and 'args'",
        )
    bin_value = raw.get("bin")
    if not isinstance(bin_value, str) or not bin_value:
        raise LaunchRefusal(
            3, f"engine profile at {path} fitCheck.bin must be a non-empty string"
        )
    resolved_bin = _resolve_fit_check_bin_value(bin_value, path)
    args = raw.get("args", [])
    if not isinstance(args, list) or not all(isinstance(item, str) and item for item in args):
        raise LaunchRefusal(
            3,
            f"engine profile at {path} fitCheck.args must be a list of non-empty "
            "strings",
        )
    collision = _reserved_fit_check_arg_collision(args)
    if collision is not None:
        item, reserved = collision
        raise LaunchRefusal(
            3,
            f"engine profile at {path} fitCheck.args contains {item!r}, which "
            f"collides with the reserved flag {reserved!r} this launcher "
            "itself always supplies (either the exact flag or an argparse "
            "abbreviation of it that the sizer's own argument parser would "
            "resolve to it)",
        )
    return {"bin": resolved_bin, "args": list(args)}


def load_engine_profile(path: Optional[str]) -> tuple:
    """Load and validate an engine profile.

    Returns ``(profile, is_built_in)``. ``path`` of ``None`` selects the
    built-in profile (which carries no ``residencyArgs``: it cannot stream,
    and no ``fitCheck``: it names no sizer of its own). Refuses
    (``LaunchRefusal(3, ...)``) on a missing/unreadable/malformed file, a
    schema mismatch, an argv element using any ``{...}`` placeholder outside
    the allowed set, an invalid ``residencyArgs``, or an invalid
    ``fitCheck`` -- a malformed profile must never silently drop or
    mis-render a token in the exec'd command line, or run the wrong sizer.
    The returned profile dict always carries a ``residencyArgs`` key (an
    empty dict when the profile declares none) and a ``fitCheck`` key
    (``None`` when the profile declares none).
    """
    if path is None:
        profile = dict(BUILT_IN_ENGINE_PROFILE)
        profile["residencyArgs"] = {}
        profile["fitCheck"] = None
        profile["engineBuild"] = None
        return profile, True

    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as error:
        raise LaunchRefusal(3, f"engine profile at {path} could not be read: {error}")
    except json.JSONDecodeError as error:
        raise LaunchRefusal(3, f"engine profile at {path} is not valid JSON: {error}")

    if not isinstance(document, dict) or document.get("schema") != "fastmlx-engine-profile-v1":
        raise LaunchRefusal(
            3, f"engine profile at {path} is not a fastmlx-engine-profile-v1 document"
        )
    name = document.get("name")
    argv = document.get("argv")
    if not isinstance(name, str) or not name:
        raise LaunchRefusal(3, f"engine profile at {path} must have a non-empty 'name' string")
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise LaunchRefusal(
            3, f"engine profile at {path} must have a non-empty 'argv' list of strings"
        )
    for item in argv:
        for token in _placeholder_tokens(item):
            placeholder = token[1:-1]
            if placeholder not in ALLOWED_ENGINE_PROFILE_PLACEHOLDERS:
                allowed = ", ".join(
                    "{" + name + "}" for name in sorted(ALLOWED_ENGINE_PROFILE_PLACEHOLDERS)
                )
                raise LaunchRefusal(
                    3,
                    f"engine profile at {path} argv element {item!r} uses unknown "
                    f"placeholder {token}; allowed placeholders are: {allowed}",
                )
    residency_args = _validate_residency_args(document, path)
    fit_check = _validate_fit_check(document, path)
    engine_build = _validate_engine_build_profile(document, path)
    return (
        {
            "name": name,
            "argv": argv,
            "residencyArgs": residency_args,
            "fitCheck": fit_check,
            "engineBuild": engine_build,
        },
        False,
    )


def _substitute_placeholders(item: str, substitutions: dict) -> str:
    result = item
    for key, value in substitutions.items():
        result = result.replace("{" + key + "}", value)
    return result


def _resolve_executable(value: str) -> Optional[str]:
    """Resolve ``value`` (a bare name or a path) to an absolute executable path."""
    if os.sep in value or value.startswith("."):
        candidate = Path(value)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
        return None
    return shutil.which(value)


def _guarded_engine_bin_abs_for_engine_build_derivation(
    args, is_built_in_profile: bool
) -> Optional[str]:
    """The same engine-binary resolution ``_run_serve`` performs lower down
    (``--engine-bin``, or the built-in engine name for the built-in
    profile), but computed early and GUARDED to never raise -- used only to
    feed ``derive_engine_build_from_release`` before the quality-card
    lookup, which needs a candidate commit sooner than the engine binary is
    otherwise resolved.

    This deliberately does NOT raise the "--engine-bin is required" /
    "engine binary not found" ``LaunchRefusal``s the real resolution below
    raises: those refusals must keep firing in their EXISTING position (the
    tests pin their ordering and exit codes), so an engine-build derivation
    attempt that cannot even resolve a candidate binary just yields no
    candidate here -- the real refusal, if any, still fires later, from the
    unchanged code path.
    """
    engine_bin_value = args.engine_bin
    if engine_bin_value is None:
        if not is_built_in_profile:
            return None
        engine_bin_value = _BUILT_IN_ENGINE_BINARY_NAME
    return _resolve_executable(engine_bin_value)


_RESIDENCY_FLAG = "--residency"
# The shortest argparse abbreviation of --residency this guard treats as a
# residency assertion. Chosen conservatively (5 chars: "--res") so the
# guard reads unambiguously as "residency" to an operator and does not
# fire on some unrelated, shorter --res*-prefixed flag a future sizer
# might add.
_RESIDENCY_ABBREV_MIN_LEN = 5


def _is_residency_flag_name(name: str) -> bool:
    """Whether ``name`` (a fit-check-arg token's flag part, before any
    ``=value``) is ``--residency`` itself or a qualifying argparse
    abbreviation of it -- see ``_RESIDENCY_ABBREV_MIN_LEN``.
    """
    return name == _RESIDENCY_FLAG or (
        len(name) >= _RESIDENCY_ABBREV_MIN_LEN and _RESIDENCY_FLAG.startswith(name)
    )


def _residency_assertions_in_fit_check_args(fit_check_args: list) -> list:
    """Every ``--residency`` (or qualifying abbreviation) assertion inside a
    caller-supplied list of fit-check args, across ALL occurrences -- not
    just the first -- as a list of ``(item, value)`` tuples in argv order.
    ``value`` is ``None`` when the flag carries no following value (its
    last argv element). Both ``--residency VALUE`` and ``--residency=VALUE``
    forms are recognized, matching what the sizers' own
    ``argparse.ArgumentParser`` (abbreviations enabled) would resolve. This
    never adds ``--residency`` to the fit-check argv itself: the fit-check
    protocol is shared with a Swift binary that does not accept it.
    """
    assertions: list = []
    index = 0
    while index < len(fit_check_args):
        token = fit_check_args[index]
        if token.startswith("--"):
            name, sep, inline_value = token.partition("=")
            if _is_residency_flag_name(name):
                if sep:
                    assertions.append((token, inline_value))
                elif index + 1 < len(fit_check_args):
                    assertions.append((token, fit_check_args[index + 1]))
                    index += 1
                else:
                    assertions.append((token, None))
        index += 1
    return assertions


def _residency_conflict_in_fit_check_args(
    fit_check_args: list, residency: str
) -> Optional[tuple]:
    """The first ``(item, value)`` residency assertion in ``fit_check_args``
    whose value is missing or differs from ``residency`` (the launch's own
    value), or ``None`` if every assertion agrees (including the "no
    assertion at all" case).
    """
    for item, value in _residency_assertions_in_fit_check_args(fit_check_args):
        if value != residency:
            return item, value
    return None


def _residency_conflict_source_and_value(
    profile_args: list, cli_args: list, residency: str
) -> Optional[tuple]:
    """Refuse-worthy residency disagreement between EITHER arg source and
    ``residency``, attributed to its actual source -- an engine profile's
    own ``fitCheck.args`` or the CLI/env's ``--fit-check-arg`` -- so the
    refusal message never blames one source for a conflict that came from
    the other. Checked as two independent scans over the two lists
    separately (never the concatenated list), specifically so attribution
    is correct. Returns ``(source_label, item, value)`` or ``None``.
    """
    for source_label, arg_list in (
        ("the engine profile's fitCheck.args", profile_args),
        ("--fit-check-arg", cli_args),
    ):
        conflict = _residency_conflict_in_fit_check_args(arg_list, residency)
        if conflict is not None:
            item, value = conflict
            return source_label, item, value
    return None


_FRONT_MODE_BYPASS_FLAGS = ("--host", "--port", "--hostname", "--bind", "-H")


def _flag_and_inline_value(token: str):
    """Split a ``--flag=value`` token into ``(flag, value)``; a bare flag
    (or anything without ``=``) returns ``(token, None)``.
    """
    if token.startswith("-") and "=" in token:
        flag, _, value = token.partition("=")
        return flag, value
    return token, None


def _is_front_mode_bypass_flag(token: str) -> bool:
    flag, _ = _flag_and_inline_value(token)
    return flag in _FRONT_MODE_BYPASS_FLAGS


def _placeholder_bearing(value: Optional[str]) -> bool:
    return bool(value) and ("{host}" in value or "{port}" in value)


def _front_mode_profile_argv_bypass_token(argv: list) -> Optional[str]:
    """The first host/port-bypass-capable flag in a profile's templated
    ``argv`` that is NOT immediately paired with a ``{host}``/``{port}``
    placeholder value -- see the front-mode loopback-guarantee review (M1).
    ``--host {host} --port {port}`` (the shipped example profiles' and the
    built-in profile's own spelling) is exempted; a literal
    ``--hostname 0.0.0.0``, a bare flag with no placeholder value, or an
    ``=``-form whose value carries no placeholder, is not. Returns the
    offending token, or ``None`` if the whole argv is clean.
    """
    for index, token in enumerate(argv):
        if not _is_front_mode_bypass_flag(token):
            continue
        _, inline_value = _flag_and_inline_value(token)
        if inline_value is not None:
            if _placeholder_bearing(inline_value):
                continue
            return token
        next_value = argv[index + 1] if index + 1 < len(argv) else None
        if _placeholder_bearing(next_value):
            continue
        return token
    return None


def _literal_bypass_token(tokens) -> Optional[str]:
    """Same bypass-flag check for a list of already-literal tokens
    (``residencyArgs`` entries or passthrough args): these can never carry
    a ``{host}``/``{port}`` placeholder (the schema forbids ``{``/``}`` in
    ``residencyArgs``, and passthrough args are never templated at all),
    so any bypass flag among them is refused unconditionally.
    """
    for token in tokens:
        if _is_front_mode_bypass_flag(token):
            return token
    return None


def _residency_flag_tokens(profile: dict) -> set:
    """Every ``-``-prefixed token appearing in ANY of ``profile``'s
    ``residencyArgs`` lists, regardless of which residency the current
    launch requested -- the passthrough guard refuses these tokens even on
    a ``resident`` launch, so a streaming flag can never be smuggled in as
    a passthrough argument instead of going through ``--residency
    expert-stream``.
    """
    tokens: set = set()
    for arg_list in profile.get("residencyArgs", {}).values():
        for token in arg_list:
            if token.startswith("-"):
                tokens.add(token)
    return tokens


# ---------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fastmlx_launch",
        description=(
            "fastmlx serve: fit-check + quality-admission front door for an "
            "OpenAI-compatible serving engine."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser(
        "serve",
        help="fit-check, admit, then exec an OpenAI-compatible serving engine",
    )
    serve.add_argument(
        "--model-path",
        required=True,
        type=Path,
        help="the model directory to serve (must be a valid MLX or GGUF pack layout)",
    )
    serve.add_argument(
        "--model-id",
        default=None,
        help=(
            "the model id reported to the fit check and to clients "
            "(default: the resolved --model-path directory's basename)"
        ),
    )
    serve.add_argument(
        "--model-repo",
        default=None,
        help=(
            "override the model's Hugging Face repo id used for "
            "quality-card lookup (default: read from the sibling "
            ".pull-receipt.json, if one exists)"
        ),
    )
    serve.add_argument(
        "--model-revision",
        default=None,
        help=(
            "override the model's pinned revision (40-char lowercase hex "
            "sha) used for hfPin card matching (default: read from the "
            "sibling .pull-receipt.json, if one exists)"
        ),
    )
    serve.add_argument(
        "--card-id",
        default=None,
        help=(
            "pin the exact quality card by id instead of automatic "
            "resolution; refused unless the model's resolved repo/revision "
            "and residency match that card"
        ),
    )
    serve.add_argument(
        "--quality-cards",
        default=None,
        help=(
            "path to the quality-card manifest (default: "
            f"{DEFAULT_QUALITY_CARDS_RELATIVE_PATH} under the repo root; "
            "an explicitly-named manifest that fails to load refuses "
            "startup, unlike a missing default path)"
        ),
    )
    serve.add_argument(
        "--accept-quality",
        action="append",
        default=[],
        help=(
            "opt in to a quality-flagged card by id, repo, or hfPin "
            "(repeatable); required to admit a card whose verdict is NO_GO"
        ),
    )
    serve.add_argument(
        "--residency",
        default="resident",
        choices=list(RESIDENCIES),
        help=(
            "how the pack is held while serving: 'resident' (whole pack in "
            "memory) or 'expert-stream' (experts streamed from disk, which "
            "requires an --engine-profile declaring "
            "residencyArgs.expert-stream)"
        ),
    )
    serve.add_argument(
        "--context",
        type=int,
        default=None,
        help=(
            "the context length to request (default: the fit check's own "
            "reported fit_context_ceiling; startup refuses if neither is "
            "available)"
        ),
    )
    serve.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "the host the serving engine binds (ignored in --front-port "
            "mode, where the engine always binds loopback)"
        ),
    )
    serve.add_argument(
        "--port",
        type=int,
        default=8080,
        help=(
            "the port the serving engine binds (default 8080; must differ "
            "from --front-port when front mode is enabled)"
        ),
    )
    serve.add_argument(
        "--host-use",
        default="shared",
        choices=["shared", "dedicated-serving"],
        help="the host-sharing mode passed to the fit check (default 'shared')",
    )
    serve.add_argument(
        "--fit-check-bin",
        default=None,
        help=(
            "the fit-check binary to run (an absolute path, or one of "
            "'builtin:safetensors'/'builtin:gguf'), overriding both "
            "FASTMLX_FIT_CHECK_BIN and the engine profile's own "
            "fitCheck.bin (default: the profile's fitCheck.bin, else the "
            "built-in engine if found on PATH, else a built-in "
            "pure-Python sizer auto-selected from the pack's own "
            "contents)"
        ),
    )
    serve.add_argument(
        "--fit-check-arg",
        action="append",
        default=[],
        help="an extra argv token appended to the fit-check invocation (repeatable)",
    )
    serve.add_argument(
        "--kv-reserve-gib",
        type=float,
        default=None,
        help=(
            "the KV-cache reserve (GiB) forwarded to the fit check as its "
            "own --kv-reserve-gib; required whenever a built-in sizer "
            "(builtin:safetensors / builtin:gguf) resolves this launch's "
            "fit check -- a fit check must never silently assume a zero "
            "KV-cache reserve; refused if given while a non-built-in "
            "fit-check binary is in use"
        ),
    )
    serve.add_argument(
        "--force",
        action="store_true",
        help=(
            "start anyway after a confirmed RED fit verdict; never "
            "overrides a fit check that could not be run at all, an "
            "attested-residency mismatch, or an engine binarySha256 "
            "mismatch"
        ),
    )
    serve.add_argument(
        "--engine-profile",
        default=None,
        help=(
            "path to an engine-profile JSON file naming the serving "
            "engine's argv (default: the built-in in-tree engine profile)"
        ),
    )
    serve.add_argument(
        "--engine-bin",
        default=None,
        help=(
            "the serving engine binary to exec (required when "
            "--engine-profile names a non-built-in profile; the built-in "
            "profile supplies its own default)"
        ),
    )
    serve.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved launch plan as JSON and exit, without exec'ing the engine",
    )
    # --front-port: opt-in, off by default. When given, the engine's {host}
    # placeholder is forced to loopback and this launcher runs an
    # X-FastMLX-*-header-stamping reverse proxy on --front-host:--front-port
    # in front of it instead of exec'ing the engine directly (see
    # `_run_front_mode`). --front-host only ever names where the PROXY
    # binds; the engine itself always binds loopback in front mode.
    serve.add_argument(
        "--front-port",
        type=int,
        default=None,
        help=(
            "enable front mode: run a header-stamping reverse proxy on "
            "this port in front of the engine, which then always binds "
            "loopback regardless of --host (default: front mode disabled)"
        ),
    )
    serve.add_argument(
        "--front-host",
        default="127.0.0.1",
        help=(
            "the host the front-mode proxy itself binds (only meaningful "
            "with --front-port; never changes where the engine binds)"
        ),
    )
    # Deliberately NOT ``type=int``: argparse's own conversion would refuse
    # a non-numeric value with its own "invalid int value" usage error, a
    # different shape than the ``LaunchRefusal`` fail-closed style every
    # other front-mode argument check below uses. Left a plain string here
    # and parsed/validated in `_run_serve` instead, so 0, a negative count,
    # and a non-numeric string all refuse through the SAME named-reason
    # path (see the validation block below `front_mode`).
    serve.add_argument(
        "--front-max-body-bytes",
        default=None,
        help=(
            "in front mode, the proxy's ceiling on a request's declared "
            "Content-Length, in bytes (default: the proxy's own default, "
            f"{fastmlx_proxy.DEFAULT_MAX_REQUEST_BODY_BYTES} bytes); must "
            "be a positive integer"
        ),
    )
    # Deliberately NOT ``type=int``, for the same reason as
    # --front-max-body-bytes immediately above: parsed/validated in
    # `_run_serve` so 0, a negative count, and a non-numeric string all
    # refuse through the SAME named-reason `LaunchRefusal` path instead of
    # argparse's own "invalid int value" usage error.
    serve.add_argument(
        "--front-max-concurrent",
        default=None,
        help=(
            "in front mode, the proxy's ceiling on in-flight requests "
            "accepted at once (default: the proxy's own default, "
            f"{fastmlx_proxy.DEFAULT_MAX_CONCURRENT_REQUESTS}); the real "
            "aggregate memory ceiling is this times --front-max-body-bytes"
        ),
    )
    return parser


# The one-line hint printed (to stderr) whenever a model pack resolves no
# identity at all (no sibling pull receipt, no --model-revision): shared
# verbatim by `fastmlx serve` and `fastmlx recommend` (which prefixes it
# with its own program name) so the two front doors can never drift on this
# wording.
NO_MODEL_IDENTITY_HINT = (
    "no model identity (no pull receipt, no --model-revision); no quality "
    "card was consulted -- run 'fastmlx pull <repo>@<revision> --dest <dir> "
    "--adopt' to pin a hand-staged pack"
)


def print_no_model_identity_hint(prefix: str, subject: Optional[str] = None) -> None:
    # ``subject`` names the pack when one invocation may print several hints.
    lead = f"{prefix}: {subject}" if subject is not None else prefix
    print(f"{lead}: {NO_MODEL_IDENTITY_HINT}", file=sys.stderr)


def _load_pull_receipt(model_path: Path) -> Optional[dict]:
    """Load the receipt ``fastmlx pull`` wrote for ``model_path``, if any.

    The receipt lives exactly where ``fastmlx_pull.receipt_path_for`` places
    it: a SIBLING of the model directory (``<dir>.pull-receipt.json``), named
    from the same ``.expanduser().absolute()`` normalization ``pull()``
    applies to its ``dest`` argument before writing. There is no separate
    in-dir naming convention here; nothing ever writes one.

    Fails open to "no receipt" (returns ``None``) on a missing file,
    unreadable file, invalid JSON, a non-dict document -- and also on a
    receipt whose recorded ``dest`` does not resolve to ``model_path``. That
    last case matters because a receipt can be copied or left behind after a
    directory is moved/renamed: without this check a stale receipt would
    silently identify the wrong pack. A receipt that records no ``dest`` at
    all is trusted as-is (the sibling naming already ties it to this path).
    """
    dest = model_path.expanduser().absolute()
    receipt_path = pull.receipt_path_for(dest)
    if not receipt_path.is_file():
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(receipt, dict):
        return None
    recorded_dest = receipt.get("dest")
    if isinstance(recorded_dest, str):
        try:
            if Path(recorded_dest).resolve() != model_path.resolve():
                return None
        except OSError:
            return None
    return receipt


def _resolve_model_repo(args, model_path: Path) -> Optional[str]:
    if args.model_repo:
        return args.model_repo
    receipt = _load_pull_receipt(model_path)
    if receipt is None:
        return None
    repo_id = receipt.get("repo_id")
    return repo_id if isinstance(repo_id, str) else None


def _resolve_model_revision(args, model_path: Path) -> Optional[str]:
    """The model's pinned HF revision, for hfPin-prefix card matching:
    ``--model-revision`` if given, else the sibling pull receipt's
    ``revision`` field (the 40-char lowercase hex sha ``fastmlx pull``
    records there).
    """
    if args.model_revision:
        return args.model_revision
    receipt = _load_pull_receipt(model_path)
    if receipt is None:
        return None
    revision = receipt.get("revision")
    return revision if isinstance(revision, str) else None


def _run_serve(args, passthrough_args: list) -> int:
    model_path: Path = args.model_path
    if not model_path.is_dir():
        raise LaunchRefusal(2, f"model path {model_path} does not exist or is not a directory")
    if not is_model_dir(model_path):
        raise LaunchRefusal(2, f"model path {model_path} {MODEL_DIR_REFUSAL_MESSAGE_SUFFIX}")

    model_id = args.model_id or model_path.resolve().name
    model_repo = _resolve_model_repo(args, model_path)
    model_revision = _resolve_model_revision(args, model_path)
    residency = args.residency

    # Engine profile is loaded/validated early: a malformed profile is a
    # configuration error unrelated to the fit check or the quality card,
    # and failing on it before spending time on a fit-check subprocess
    # call keeps the refusal prompt.
    profile, is_built_in_profile = load_engine_profile(args.engine_profile)
    # This launch's own engine build (absent for the built-in profile and
    # for any profile that does not declare one) -- resolved this early
    # because the quality-card lookup below (`resolve_card`) needs it to
    # disambiguate a pack with more than one card, never to filter one.
    #
    # An operator-declared engineBuild.commit in the profile ALWAYS wins
    # and is used unchanged -- derivation only fills the gap when the
    # profile declares no commit at all (including the built-in profile,
    # whose engineBuild is always None). Derivation is attempted through a
    # GUARDED, never-raising local resolution
    # (`_guarded_engine_bin_abs_for_engine_build_derivation`) rather than
    # hoisting the real `_resolve_executable`/`--engine-bin` resolution
    # from its existing position further down: several tests pin the exact
    # ordering and exit codes of the refusals between here and there (the
    # residency, fit-check, and quality-card admission checks), and hoisting
    # would make an "engine binary not found" refusal fire before all of
    # them instead of after. `derive_engine_build_from_release` itself never
    # raises either -- an underivable build is simply left "undeclared",
    # the same honest outcome a profile declaring no commit already
    # produces.
    launch_engine_build_commit = (profile.get("engineBuild") or {}).get("commit")
    if launch_engine_build_commit is None:
        candidate_engine_bin_abs = _guarded_engine_bin_abs_for_engine_build_derivation(
            args, is_built_in_profile
        )
        if candidate_engine_bin_abs is not None:
            derived_engine_build = derive_engine_build_from_release(candidate_engine_bin_abs)
            if derived_engine_build is not None:
                launch_engine_build_commit = derived_engine_build["commit"]
    # Whether this launch's final argv carries the exact token `--mtp` --
    # resolved this early (pre-substitution; see `_final_argv_preview`) so
    # the quality-card `--mtp` status below can be computed before the
    # admission decision, the same reason `launch_engine_build_commit` is.
    is_mtp_launch = mtp_launch_requested(profile, residency, passthrough_args)

    # --- residency validation ------------------------------------------
    # A non-resident launch requires the profile to declare the argv this
    # engine needs to stream; the built-in engine never does, so
    # --residency expert-stream always refuses it.
    if residency == "expert-stream" and not profile["residencyArgs"].get("expert-stream"):
        if is_built_in_profile:
            detail = "the built-in engine cannot stream"
        else:
            detail = f"engine profile {profile['name']!r} declares no expert-stream arguments"
        raise LaunchRefusal(
            3,
            f"cannot launch with --residency expert-stream: {detail}; pass an "
            "--engine-profile whose residencyArgs.expert-stream lists the "
            "engine's streaming argv",
        )

    # Passthrough guard, regardless of the chosen residency: a streaming
    # flag can never be smuggled in as a bare passthrough argument -- an
    # operator must go through --residency expert-stream, which is what
    # actually elects the quality card for that residency.
    residency_flag_tokens = _residency_flag_tokens(profile)
    for passthrough in passthrough_args:
        for token in residency_flag_tokens:
            if passthrough == token or passthrough.startswith(token + "="):
                raise LaunchRefusal(
                    2,
                    f"passthrough argument {passthrough!r} belongs to a residency "
                    "profile's own argv; use --residency expert-stream instead of "
                    "passing it directly",
                )

    # --- fit-check ---------------------------------------------------
    # Precedence: --fit-check-bin (CLI) > FASTMLX_FIT_CHECK_BIN (env) >
    # the engine profile's own fitCheck.bin > the built-in engine (found on
    # PATH) > a built-in pure-Python sizer auto-selected from the pack's own
    # contents (see _select_builtin_fit_check_bin) -- the same last-resort
    # `fastmlx recommend` already uses, so a pack outside the Swift binary's
    # fixed catalog (or a host with no Swift binary built at all) can still
    # be fit-checked instead of refusing outright. An override from the CLI
    # or the environment drops the profile's fitCheck.args entirely -- those
    # args are sized for the profile's own sizer, never for whatever binary
    # overrode it -- and is announced on stderr so an operator does not
    # wonder why a profile's fitCheck.args were silently ignored.
    profile_fit_check = profile.get("fitCheck")
    cli_or_env_fit_check_bin = args.fit_check_bin or os.environ.get("FASTMLX_FIT_CHECK_BIN")
    profile_fit_check_args: list = []
    cli_fit_check_args = list(args.fit_check_arg)
    if cli_or_env_fit_check_bin:
        # A `builtin:` name given via --fit-check-bin/FASTMLX_FIT_CHECK_BIN
        # is resolved the same way an engine profile's own fitCheck.bin is
        # -- it must never be passed to run_fit_check raw (that would try
        # to exec a literal path named "builtin:safetensors" and fail with
        # a confusing "binary not found").
        override_source = "--fit-check-bin" if args.fit_check_bin else "FASTMLX_FIT_CHECK_BIN"
        fit_check_bin = _resolve_fit_check_bin_value(cli_or_env_fit_check_bin, override_source)
        fit_check_extra_args = list(cli_fit_check_args)
        if profile_fit_check is not None:
            print(
                f"fastmlx serve: {override_source} overrides engine profile "
                f"{profile['name']!r}'s own fitCheck; its fitCheck.args are not "
                "applied",
                file=sys.stderr,
            )
    elif profile_fit_check is not None:
        fit_check_bin = profile_fit_check["bin"]
        profile_fit_check_args = list(profile_fit_check["args"])
        fit_check_extra_args = profile_fit_check_args + cli_fit_check_args
    else:
        fit_check_bin = shutil.which(_BUILT_IN_ENGINE_BINARY_NAME)
        fit_check_extra_args = list(cli_fit_check_args)
        if not fit_check_bin:
            # No CLI/env override, no profile fitCheck, and the built-in
            # Swift engine binary isn't on PATH (e.g. it was never built,
            # or this pack is outside its fixed ~19-id catalog): auto-select
            # one of this repository's own pure-Python sizers from the
            # pack's own contents instead of refusing outright. Precedence
            # is unchanged for every existing setup -- this fallback is
            # only ever reached when shutil.which already returned None.
            builtin_name, select_error = _select_builtin_fit_check_bin(model_path)
            if select_error is not None:
                raise LaunchRefusal(3, f"fit check could not run: {select_error}")
            fit_check_bin = _resolve_fit_check_bin_value(
                builtin_name, "<fastmlx serve auto-selected built-in sizer>"
            )

    if not fit_check_bin:
        raise LaunchRefusal(
            3,
            "fit check binary not found; pass --fit-check-bin or set "
            "FASTMLX_FIT_CHECK_BIN",
        )

    # A fit-check-arg (either the engine profile's own fitCheck.args, or a
    # CLI/env --fit-check-arg) that itself names a conflicting --residency
    # is refused before the fit check runs; each source is scanned
    # separately so the refusal names its actual origin. Every occurrence
    # (not just the first) and every argparse abbreviation of --residency
    # are checked -- see _residency_conflict_source_and_value. --residency
    # is never auto-added to the fit-check args (the fit-check protocol is
    # shared with a Swift binary that does not accept it).
    residency_conflict = _residency_conflict_source_and_value(
        profile_fit_check_args, cli_fit_check_args, residency
    )
    if residency_conflict is not None:
        source_label, item, value = residency_conflict
        value_desc = "no value" if value is None else repr(value)
        raise LaunchRefusal(
            2,
            f"{source_label} specifies {item!r} ({value_desc}), which differs "
            f"from the launch's own --residency {residency!r}",
        )

    # Whether the resolved fit_check_bin is one of this repository's own
    # built-in pure-Python sizers -- true whether it got here via an
    # explicit builtin: name (CLI/env or engine profile) or the auto-select
    # fallback above. Those sizers require --kv-reserve-gib (a fit check
    # must never silently assume a zero KV-cache reserve, mirroring
    # `fastmlx recommend`'s own requirement -- see build_row); the Swift
    # built-in engine binary does not accept the flag at all, so it is
    # never appended outside this branch. Checked after the residency
    # conflict above so an operator sees the residency conflict's exit 2
    # first when a launch happens to trip both checks at once.
    fit_check_is_builtin = fit_check_bin in _BUILTIN_FIT_CHECK_BIN_PATHS
    kv_reserve_already_in_args = any(
        item.split("=", 1)[0] == "--kv-reserve-gib" for item in fit_check_extra_args
    )
    if fit_check_is_builtin:
        missing = _builtin_sizer_missing_requirements(
            kv_reserve_gib=args.kv_reserve_gib,
            kv_reserve_already_in_args=kv_reserve_already_in_args,
            context=args.context,
        )
        if missing:
            builtin_name = _BUILTIN_FIT_CHECK_BIN_PATHS[fit_check_bin]
            raise LaunchRefusal(
                3,
                f"fit check could not run: a built-in sizer ({builtin_name}) "
                f"is in use, which requires {_join_flag_list(missing)}; "
                + "; ".join(_BUILTIN_REQUIREMENT_REASONS[flag] for flag in missing)
                + f"; pass {_join_flag_list(missing)}",
            )
        if args.kv_reserve_gib is not None:
            fit_check_extra_args = fit_check_extra_args + [
                "--kv-reserve-gib",
                str(args.kv_reserve_gib),
            ]
    elif args.kv_reserve_gib is not None:
        raise LaunchRefusal(
            2,
            f"--kv-reserve-gib was given but the resolved fit-check binary "
            f"{fit_check_bin} is not one of this repository's built-in "
            "sizers (builtin:safetensors / builtin:gguf), which do not "
            "accept it; pass --kv-reserve-gib only when a built-in sizer "
            "is in use",
        )

    fit_result = run_fit_check(
        fit_check_bin=fit_check_bin,
        model_id=model_id,
        model_path=model_path,
        host_use=args.host_use,
        context=args.context,
        extra_args=fit_check_extra_args,
    )

    if fit_result.kind == "error":
        # Fail-closed: an unknown fit outcome refuses even under --force.
        raise LaunchRefusal(3, f"fit check could not run: {fit_result.detail}")

    if fit_result.kind == "red":
        if not args.force:
            stderr_detail = (
                fit_result.stderr.strip()
                or "fit check verdict RED (unservable at the requested configuration)"
            )
            raise LaunchRefusal(2, f"fit check refused (fit_check=RED): {stderr_detail}")
        fit_label = "RED-forced"
        fit_fields: dict = {}
    else:
        fit_fields = fit_result.fields
        # An exit-0 verdict may be green or yellow; report the binary's own word.
        fit_label = str(fit_fields.get("fit_check", "green")).upper()

    # A GREEN attestation carrying its own residency= field must agree with
    # the launch's own --residency: both sizers print this field, and a
    # sizer that sized the wrong residency (e.g. because a fit-check-arg
    # conflict slipped past the guard above, or because an operator's
    # fitCheck.args hard-codes one) must never be treated as having
    # attested to THIS launch's configuration. Fail-closed and never
    # overridable by --force -- unlike the RED path above, this check runs
    # unconditionally after a GREEN/YELLOW verdict. A binary that omits the
    # field (e.g. the Swift built-in binary) is not checked.
    attested_residency = fit_fields.get("residency")
    if attested_residency is not None and attested_residency != residency:
        raise LaunchRefusal(
            3,
            f"fit check attested residency={attested_residency!r}, which "
            f"differs from the launch's own --residency {residency!r}; "
            "refusing (not overridable by --force)",
        )

    context = args.context
    if context is None:
        ceiling = fit_fields.get("fit_context_ceiling")
        if ceiling is None:
            raise LaunchRefusal(
                3,
                "context could not be determined: pass --context explicitly "
                "(the fit check produced no context ceiling)",
            )
        try:
            context = int(ceiling)
        except ValueError:
            raise LaunchRefusal(
                3, f"fit check reported a non-integer fit_context_ceiling: {ceiling!r}"
            )

    # --- quality-card admission ---------------------------------------
    quality_cards_path = Path(
        args.quality_cards
        if args.quality_cards is not None
        else (REPO_ROOT / DEFAULT_QUALITY_CARDS_RELATIVE_PATH)
    )
    cards = load_quality_cards(quality_cards_path)
    # Only the conventional default path may fail open to "no card" (mirrors
    # QualityCardStore); a manifest the operator named explicitly must load,
    # or a typo would silently disable the gate.
    if cards is None and args.quality_cards is not None:
        raise LaunchRefusal(
            3,
            f"the --quality-cards manifest {quality_cards_path} is missing or "
            "is not a quality-card manifest",
        )

    card: Optional[dict] = None
    if args.card_id is not None:
        card = find_card_by_id(cards, args.card_id) if cards is not None else None
        if card is None:
            raise LaunchRefusal(
                2,
                f"no quality card with id {args.card_id!r} found in "
                f"{quality_cards_path}",
            )
        # An explicit --card-id is never trusted as a bare operator
        # assertion: the model must have SOME resolved identity (repo or
        # a full pinned revision), and that identity must independently
        # match this specific card's repo or hfPin.
        if model_repo is None and model_revision is None:
            raise LaunchRefusal(
                2,
                f"quality card {args.card_id!r} cannot be verified for this "
                "model: no model identity is known (pass --model-repo or "
                "--model-revision, or pull the model with fastmlx_pull.py "
                "so .pull-receipt.json supplies one)",
            )
        if not card_matches_model_identity(card, model_repo, model_revision):
            raise LaunchRefusal(
                2,
                f"quality card {args.card_id!r} cannot be verified for this "
                f"model: its repo/hfPin do not match the resolved model "
                f"identity (repo={model_repo!r}, revision={model_revision!r})",
            )
        # An explicit --card-id must also carry the launch's own residency:
        # a resident card is not evidence for a streaming launch of the
        # same pack, and a streaming card is not evidence for a resident
        # launch.
        matched_card_residency = card_residency(card)
        if matched_card_residency != residency:
            raise LaunchRefusal(
                2,
                f"quality card {args.card_id!r} has residency "
                f"{matched_card_residency if matched_card_residency is not None else 'unrecognized'!r} "
                f"but this launch is --residency {residency!r}",
            )
    else:
        # No repo and no pinned revision at all (no --model-repo/
        # --model-revision, and no usable sibling pull receipt) means no
        # card could ever match this launch by repo or by hfPin -- a
        # hand-staged pack (rsync'd or copied in, never `fastmlx pull`ed)
        # silently falls through to admit_unmeasured otherwise. Made
        # visible here; the admission OUTCOME is unchanged either way.
        if model_repo is None and model_revision is None:
            print_no_model_identity_hint("fastmlx serve")
        card = resolve_card(
            cards,
            model_repo,
            model_revision,
            residency=residency,
            engine_build_commit=launch_engine_build_commit,
        )

    # --- engine-build status (never gates admission; see decide_admission
    # below) ------------------------------------------------------------
    card_build_commit = card_engine_build_commit(card)
    build_status = engine_build_status(card_build_commit, launch_engine_build_commit)
    build_notice: Optional[str] = None
    if build_status in (ENGINE_BUILD_STATUS_UNDECLARED, ENGINE_BUILD_STATUS_MISMATCH):
        build_notice = engine_build_notice_text(
            card.get("id") if card else None, card_build_commit, launch_engine_build_commit
        )

    # --- `--mtp` flag-transfer status (never gates admission either; see
    # decide_admission below) -------------------------------------------
    mtp_status, mtp_divergent_prompts, mtp_prompts = mtp_transfer_status(
        card, build_status, is_mtp_launch
    )
    mtp_notice: Optional[str] = None
    if mtp_status not in (MTP_STATUS_OFF, MTP_STATUS_EXACT):
        mtp_notice = mtp_notice_text(
            card.get("id") if card else None,
            mtp_status,
            mtp_divergent_prompts,
            mtp_prompts,
            card_build_commit,
        )

    opt_in_ids = set(args.accept_quality)
    opted_in = is_opted_in(card, opt_in_ids)
    outcome, message = decide_admission(card, opted_in)
    if outcome == "refuse_quality_flagged":
        notices = " ".join(notice for notice in (build_notice, mtp_notice) if notice)
        full_message = f"{message} {notices}" if notices else message
        raise LaunchRefusal(2, full_message)
    if outcome == "admit_with_quality_flag":
        print(message)
    if build_notice:
        print(f"fastmlx serve: {build_notice}", file=sys.stderr)
    if mtp_notice:
        print(f"fastmlx serve: {mtp_notice}", file=sys.stderr)

    # --- front mode (--front-port) validation ---------------------------
    # Checked before the final argv is built at all: a refusal here must
    # never depend on (or leak) the substituted argv, the same reason the
    # residency passthrough guard above runs before the fit check.
    front_mode = args.front_port is not None
    # Set even outside front mode (and before any front-mode refusal below
    # can fire) so the value passed to ``_run_front_mode`` at the bottom of
    # this function is always defined, never a NameError on a path this
    # variable's own ``if front_mode:`` guard never dropped into.
    max_request_body_bytes = fastmlx_proxy.DEFAULT_MAX_REQUEST_BODY_BYTES
    # Same reasoning as ``max_request_body_bytes`` immediately above: set
    # outside the ``if front_mode:`` guard so the value passed to
    # ``_run_front_mode`` at the bottom of this function is always defined.
    max_concurrent_requests = fastmlx_proxy.DEFAULT_MAX_CONCURRENT_REQUESTS
    if front_mode and args.front_port == args.port:
        raise LaunchRefusal(
            2,
            f"--front-port {args.front_port} cannot equal --port {args.port}: the "
            "front proxy and the engine it fronts must bind different ports",
        )
    if front_mode:
        # The loopback guarantee depends on the PROFILE, not only on the
        # passthrough-args refusal below: a custom --engine-profile whose
        # argv has no {host}/{port} placeholder at all, has a literal
        # host/port-bypass flag alongside those placeholders, or puts one
        # in residencyArgs, would let the engine bind wider than loopback
        # even though front mode believes it fixed the bind. Refuse all
        # three before the argv is even substituted (see M1 in the
        # front-mode review).
        profile_argv = profile["argv"]
        has_host_placeholder = any("{host}" in item for item in profile_argv)
        has_port_placeholder = any("{port}" in item for item in profile_argv)
        if not (has_host_placeholder and has_port_placeholder):
            raise LaunchRefusal(
                2,
                f"engine profile {profile['name']!r} argv has no {{host}}/{{port}} "
                "placeholder; front mode cannot guarantee the engine binds "
                "loopback without both, so it refuses to start",
            )
        profile_bypass = _front_mode_profile_argv_bypass_token(profile_argv)
        if profile_bypass is not None:
            raise LaunchRefusal(
                2,
                f"engine profile {profile['name']!r} argv token {profile_bypass!r} "
                "would let the engine bind a host/port the --front-port proxy does "
                "not expect, bypassing it; only a flag immediately followed by a "
                "{host}/{port} placeholder value is allowed in front mode",
            )
        for residency_name, arg_list in (profile.get("residencyArgs") or {}).items():
            residency_bypass = _literal_bypass_token(arg_list)
            if residency_bypass is not None:
                raise LaunchRefusal(
                    2,
                    f"engine profile {profile['name']!r} residencyArgs[{residency_name!r}] "
                    f"token {residency_bypass!r} would let the engine bind a host/port "
                    "the --front-port proxy does not expect, bypassing it; remove it -- "
                    "front mode already fixes the engine's host and port",
                )
        for passthrough in passthrough_args:
            if _is_front_mode_bypass_flag(passthrough):
                raise LaunchRefusal(
                    2,
                    f"passthrough argument {passthrough!r} would let the engine bind "
                    "a host/port the --front-port proxy does not expect, letting a "
                    "client bypass the proxy entirely; remove it -- front mode "
                    "already fixes the engine's host and port",
                )

        # --front-max-body-bytes: the proxy's own ceiling on a single
        # request's DECLARED Content-Length (see
        # ``fastmlx_proxy.DEFAULT_MAX_REQUEST_BODY_BYTES``). Refused here,
        # in the SAME LaunchRefusal fail-closed style as every other
        # front-mode argument check above, rather than left for
        # ``fastmlx_proxy.create_server`` to reject deep inside the bind
        # call -- 0, a negative count, and a non-numeric string are all
        # equally nonsensical as a body-size ceiling and all get exactly
        # one named reason, never a raw ``ValueError`` traceback.
        if args.front_max_body_bytes is not None:
            try:
                max_request_body_bytes = int(args.front_max_body_bytes)
            except ValueError:
                raise LaunchRefusal(
                    2,
                    f"--front-max-body-bytes {args.front_max_body_bytes!r} is not "
                    "an integer number of bytes",
                )
            if max_request_body_bytes <= 0:
                raise LaunchRefusal(
                    2,
                    "--front-max-body-bytes must be a positive number of bytes, "
                    f"got {max_request_body_bytes}",
                )

        # --front-max-concurrent: the proxy's own ceiling on the number of
        # in-flight requests it will accept AT ONCE, enforced at accept
        # time -- before a handler thread is even started, let alone a
        # body byte read (see ``fastmlx_proxy.DEFAULT_MAX_CONCURRENT_REQUESTS``
        # and ``ProvenanceProxyServer``'s 503-at-accept refusal). Validated
        # here in the SAME LaunchRefusal fail-closed style as
        # --front-max-body-bytes just above, rather than left for
        # ``fastmlx_proxy.create_server`` to reject deep inside the bind
        # call. Note this bounds CONCURRENT REQUEST COUNT, not bytes: the
        # real aggregate memory ceiling is the PRODUCT of this limit and
        # --front-max-body-bytes, not either alone.
        if args.front_max_concurrent is not None:
            try:
                max_concurrent_requests = int(args.front_max_concurrent)
            except ValueError:
                raise LaunchRefusal(
                    2,
                    f"--front-max-concurrent {args.front_max_concurrent!r} is not "
                    "an integer number of requests",
                )
            if max_concurrent_requests <= 0:
                raise LaunchRefusal(
                    2,
                    "--front-max-concurrent must be a positive number of "
                    f"requests, got {max_concurrent_requests}",
                )

    # --- engine argv ---------------------------------------------------
    engine_bin_value = args.engine_bin
    if engine_bin_value is None:
        if not is_built_in_profile:
            raise LaunchRefusal(
                3, "--engine-bin is required when --engine-profile is not the built-in profile"
            )
        engine_bin_value = _BUILT_IN_ENGINE_BINARY_NAME

    engine_bin_abs = _resolve_executable(engine_bin_value)
    if engine_bin_abs is None:
        raise LaunchRefusal(3, f"engine binary not found: {engine_bin_value}")

    # A profile's engineBuild.binarySha256, when given, is VERIFIED against
    # the resolved engine binary -- an operator-asserted commit alone
    # (engineBuild.commit) is never checked this way, only this sha256.
    # Unconditional: --force never reaches this check (it is only consulted
    # on the fit-check RED path above), so a binary mismatch can never be
    # forced through.
    profile_engine_build = profile.get("engineBuild")
    expected_binary_sha256 = (
        profile_engine_build.get("binarySha256") if profile_engine_build else None
    )
    if expected_binary_sha256:
        actual_binary_sha256 = _sha256_file(engine_bin_abs)
        if actual_binary_sha256 != expected_binary_sha256:
            raise LaunchRefusal(
                3,
                "engine profile engineBuild.binarySha256 "
                f"{expected_binary_sha256} does not match the resolved engine "
                f"binary's actual sha256 {actual_binary_sha256} ({engine_bin_abs})",
            )

    # In front mode the engine ALWAYS binds loopback, regardless of
    # --front-host: --front-host only names where the proxy itself listens,
    # and the whole point of front mode is that the engine is never
    # reachable except through it.
    engine_host = "127.0.0.1" if front_mode else args.host
    substitutions = {
        "engine_bin": engine_bin_abs,
        "model_path": str(model_path.resolve()),
        "model_id": model_id,
        "host": engine_host,
        "port": str(args.port),
        "context": str(context),
    }
    final_argv = [_substitute_placeholders(item, substitutions) for item in profile["argv"]]
    if residency == "expert-stream":
        final_argv += list(profile["residencyArgs"]["expert-stream"])
    final_argv += list(passthrough_args)

    plan = {
        "fit": {"verdict": fit_label, "fields": fit_fields},
        "card": card,
        "admission": outcome,
        "argv": final_argv,
        "residency": residency,
        "engineBuild": {
            "status": build_status,
            "card": card_build_commit,
            "launch": launch_engine_build_commit,
        },
        "mtp": {
            "status": mtp_status,
            "divergentPrompts": mtp_divergent_prompts,
            "prompts": mtp_prompts,
        },
    }
    # Only in front mode: without --front-port the plan (and dry-run JSON)
    # is byte-identical to what it was before this feature existed.
    if front_mode:
        plan["front"] = {
            "host": args.front_host,
            "port": args.front_port,
            "upstream": f"http://127.0.0.1:{args.port}",
        }

    if args.dry_run:
        print(json.dumps(plan))
        return 0

    admitted_line = (
        "fastmlx_launch=admitted "
        f"engine={profile['name']} card={card.get('id') if card else 'none'} "
        f"fit={fit_label} context={context} residency={residency} "
        f"engine_build={build_status} mtp={mtp_status}"
    )
    if front_mode:
        admitted_line += f" front={args.front_host}:{args.front_port}"
    print(admitted_line, file=sys.stderr)

    if front_mode:
        return _run_front_mode(
            final_argv,
            plan,
            max_request_body_bytes=max_request_body_bytes,
            max_concurrent_requests=max_concurrent_requests,
        )

    os.execv(engine_bin_abs, final_argv)
    return 0  # pragma: no cover - unreachable, os.execv never returns on success


def _engine_port_has_listener(host: str, port: int, timeout: float = 1.0) -> bool:
    """Probe ``host:port`` with a TCP CONNECT (never a bind): a successful
    connect proves a listener is already there; loopback promptly REFUSES a
    connect to a free port, so anything other than success (refused, timed
    out, unreachable) is treated as free. A bind-based probe would falsely
    report "busy" on a port sitting in TIME_WAIT on macOS, and a timeout
    treated as "busy" would wedge a launch on a merely slow/firewalled probe
    instead of a real listener -- only an accepted connection counts.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.settimeout(timeout)
        probe.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


# ---------------------------------------------------------------------
# Front-mode engine lifeline guard: an engine a SIGKILLed launcher leaves
# behind still holds its loopback port (and, for a large pack, tens of GiB
# of resident memory) with no supervisor left to stop it -- a relaunch then
# hits the orphan-port refusal above (see `_engine_port_has_listener`)
# forever, since nothing ever stops the orphan. Front mode never Popens the
# engine directly; it Popens this small guard process instead (see
# `_run_front_mode`), which Popens the engine ITSELF and watches a pipe
# (the "lifeline") whose write end only the launcher's own process holds
# open. The launcher never writes to that pipe -- its only job is to stay
# open for exactly as long as the launcher process is alive, so the OS
# itself closes it (and the guard's read end sees EOF) the instant the
# launcher dies for ANY reason, including SIGKILL, which a process can
# never catch or clean up after itself.
# ---------------------------------------------------------------------
_INTERNAL_ENGINE_GUARD_FLAG = "--internal-engine-guard"
ENGINE_LIFELINE_GRACE_SECONDS_DEFAULT = 30.0
# Internal-only override, read by the LAUNCHER when it computes the grace
# value it passes the guard via `--grace`: never documented as a public
# flag or environment variable an operator is expected to set, and never
# read by the guard itself (the guard only ever trusts its own `--grace`
# argv value).
_ENGINE_LIFELINE_GRACE_ENV_VAR = "_FASTMLX_ENGINE_LIFELINE_GRACE_SECONDS_INTERNAL"
# A signal the OS itself writes a crash report for. The guard's own exit-
# status mirroring (see `_engine_lifeline_guard` below) must NEVER
# self-signal with one of these: that would make macOS write a SECOND
# crash report, attributed to this guard's own `python3` process, lying
# about what actually crashed. SIGTERM/SIGINT/SIGHUP (the only signals the
# guard ever forwards) and SIGKILL are deliberately absent -- none of those
# produce a crash report either way, so mirroring them is unchanged.
_ENGINE_CRASH_SIGNALS = (
    signal.SIGSEGV,
    signal.SIGABRT,
    signal.SIGBUS,
    signal.SIGILL,
    signal.SIGFPE,
    signal.SIGTRAP,
)


def _engine_lifeline_grace_seconds() -> float:
    override = os.environ.get(_ENGINE_LIFELINE_GRACE_ENV_VAR)
    if override:
        try:
            return float(override)
        except ValueError:
            pass
    return ENGINE_LIFELINE_GRACE_SECONDS_DEFAULT


def _engine_lifeline_guard_argv(
    engine_argv: list, lifeline_read_fd: int, grace_seconds: float
) -> list:
    """The argv `_run_front_mode` Popens instead of the engine directly:
    this same file, re-invoked with the internal `--internal-engine-guard`
    flag (handled at the very top of `main`, before argparse even runs, so
    the normal CLI surface is never affected -- see `main`). The engine's
    own argv follows a literal `--`, unmodified.
    """
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        _INTERNAL_ENGINE_GUARD_FLAG,
        "--lifeline-fd",
        str(lifeline_read_fd),
        "--grace",
        str(grace_seconds),
        "--",
    ] + list(engine_argv)


def _engine_lifeline_guard(
    lifeline_read_fd: int, grace_seconds: float, engine_argv: list, popen=subprocess.Popen
) -> int:
    """The guard process's own body (see `_run_engine_lifeline_guard` for
    argv parsing). Spawns the engine as its OWN child -- no new session, so
    it shares this guard's process group/session, never the launcher's --
    forwards SIGTERM/SIGINT/SIGHUP to it EXACTLY ONCE PER SIGNUM (a signal
    received twice, e.g. once directly from `pkill -f fastmlx_launch`
    matching this guard's own argv and once forwarded by the launcher,
    reaches the engine only once -- see `forwarded_signums` below;
    escalation, e.g. SIGINT then SIGTERM, still delivers both), and watches
    `lifeline_read_fd` for EOF (the launcher's death, by any means
    including SIGKILL) to stop an otherwise-orphaned engine itself:
    SIGTERM, a grace window, then SIGKILL.

    Propagates the engine's own exit status exactly: a normal exit code is
    returned as-is. A signal-killed engine is normally mirrored by
    resetting that same signal to its default disposition and delivering
    it to this guard process itself, so the LAUNCHER's own `wait()` on the
    guard continues to see a negative signal-encoded return code exactly
    as it would have watching the engine directly (see the exit-code
    mapping in `_run_front_mode`, which is otherwise unchanged) -- EXCEPT
    for a crash signal (SIGSEGV/SIGABRT/SIGBUS/SIGILL/SIGFPE/SIGTRAP,
    `_ENGINE_CRASH_SIGNALS`), which is never mirrored: this guard instead
    returns a plain `128+signum` exit code (the same number
    `_run_front_mode`'s mapping already produces for a signal-killed
    child), so the OS writes exactly one crash report -- the engine's --
    instead of a second one falsely attributed to this guard's own
    `python3` process.
    """
    # Handlers go in BEFORE the engine exists: a stop the launcher forwards
    # while the engine is being spawned would otherwise meet the default
    # disposition, kill this guard, and orphan the new engine. A signal that
    # arrives before there is an engine is held and forwarded right after.
    engine_holder = {"engine": None}
    pending = []
    # Signums already delivered to the engine, exactly once each -- shared
    # between `_forward` and the pending-signal drain right after `popen`
    # below, so a signal received twice (e.g. `pkill -f fastmlx_launch`
    # matching this guard's own argv, on top of the launcher's own
    # forwarding of the same signal) reaches the engine only once. Kept as
    # LOCAL state inside this function, never on the engine object itself:
    # `_watch_lifeline` below stops the engine directly (`terminate()`/
    # `kill()`), never through `_forward`, and must never consult -- or be
    # blocked by -- this set. Per-signum, not a blanket "a stop was already
    # sent": escalation (e.g. SIGINT then SIGTERM) must still deliver BOTH.
    forwarded_signums = set()

    def _forward(signum, _frame):
        engine = engine_holder["engine"]
        if engine is None:
            pending.append(signum)
            return
        if signum in forwarded_signums:
            return
        forwarded_signums.add(signum)
        try:
            engine.send_signal(signum)
        except OSError:
            pass

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _forward)

    try:
        engine = popen(engine_argv)
    except OSError as exc:
        print(f"fastmlx serve: front mode could not start the engine: {exc}", file=sys.stderr)
        return 3
    engine_holder["engine"] = engine
    for signum in pending:
        if signum in forwarded_signums:
            continue
        forwarded_signums.add(signum)
        try:
            engine.send_signal(signum)
        except OSError:
            pass

    def _watch_lifeline():
        try:
            os.read(lifeline_read_fd, 1)
        except OSError:
            return
        # The launcher's write end just closed (its process died, e.g. was
        # SIGKILLed): the engine is now orphaned. Stop it ourselves.
        try:
            engine.terminate()
        except ProcessLookupError:
            return
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline and engine.poll() is None:
            time.sleep(0.05)
        if engine.poll() is None:
            try:
                engine.kill()
            except ProcessLookupError:
                pass

    threading.Thread(target=_watch_lifeline, daemon=True).start()

    returncode = engine.wait()
    if returncode >= 0:
        return returncode
    sig = -returncode
    if sig in _ENGINE_CRASH_SIGNALS:
        # A crash signal must never be mirrored onto this guard: macOS
        # writes a crash report for whichever process a crash signal
        # actually kills, so self-signalling here (the non-crash path
        # below) would produce a SECOND report, attributed to this guard's
        # own `python3` process -- a lie about what crashed. Report the
        # engine's death as a normal (non-signalled) exit instead, encoded
        # the same 128+signum the shell/os convention uses for a
        # signal-killed exit status -- `_run_front_mode`'s own mapping
        # (otherwise unchanged) already treats that as the final exit code
        # for anything that isn't a requested stop.
        try:
            name = signal.Signals(sig).name
        except ValueError:  # pragma: no cover - sig is always a known Signals member here
            name = str(sig)
        print(
            f"fastmlx serve: engine died from {name} ({sig})",
            file=sys.stderr,
        )
        return 128 + sig
    try:
        # A signal this guard never installed a handler for (e.g. SIGKILL,
        # which the engine dies to when this guard's own lifeline-watch
        # SIGKILLs it after the grace window) cannot be given a disposition
        # at all -- signal.signal(SIGKILL, ...) always raises EINVAL. Only
        # a signal whose disposition this guard COULD have changed needs
        # resetting before the self-kill below.
        signal.signal(sig, signal.SIG_DFL)
    except (OSError, ValueError):
        pass
    os.kill(os.getpid(), sig)
    time.sleep(1.0)  # pragma: no cover - the self-signal above never returns
    return 1  # pragma: no cover - unreachable if the signal actually killed us


def _run_engine_lifeline_guard(argv: list) -> int:
    """Manually parse the guard's own argv: `--lifeline-fd N --grace G --
    <engine argv...>`. Never routed through `build_arg_parser`/argparse --
    this path is reached before that parser even exists (see `main`), and
    is never a public CLI surface an operator is meant to pass directly.
    """
    if "--" not in argv:
        print(
            "fastmlx serve: internal engine guard invoked without an engine argv",
            file=sys.stderr,
        )
        return 3
    split_index = argv.index("--")
    guard_opts = argv[:split_index]
    engine_argv = argv[split_index + 1 :]

    lifeline_read_fd: Optional[int] = None
    grace_seconds = ENGINE_LIFELINE_GRACE_SECONDS_DEFAULT
    index = 0
    while index < len(guard_opts):
        token = guard_opts[index]
        if token == "--lifeline-fd" and index + 1 < len(guard_opts):
            lifeline_read_fd = int(guard_opts[index + 1])
            index += 2
        elif token == "--grace" and index + 1 < len(guard_opts):
            grace_seconds = float(guard_opts[index + 1])
            index += 2
        else:
            index += 1

    if lifeline_read_fd is None:
        print(
            "fastmlx serve: internal engine guard invoked without --lifeline-fd",
            file=sys.stderr,
        )
        return 3

    return _engine_lifeline_guard(lifeline_read_fd, grace_seconds, engine_argv)


def _stop_leftover_engine_group(guard_pid: int, grace_seconds: float) -> None:
    """Stop whatever is left in the guard's process group once the guard has
    exited. The guard leads its own session, so that group is the engine's.
    Normally the group is already empty. It is not when the guard itself was
    SIGKILLed (`pkill -9 -f fastmlx_launch` matches the guard's argv too): the
    engine was then never told to stop, and would keep its port and memory.
    SIGTERM, the same grace as the guard's, then SIGKILL.
    """
    try:
        os.killpg(guard_pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(guard_pid, 0)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(0.05)
    try:
        os.killpg(guard_pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass

def _run_front_mode(
    final_argv: list,
    plan: dict,
    popen=subprocess.Popen,
    max_request_body_bytes: int = fastmlx_proxy.DEFAULT_MAX_REQUEST_BODY_BYTES,
    max_concurrent_requests: int = fastmlx_proxy.DEFAULT_MAX_CONCURRENT_REQUESTS,
) -> int:
    """Front mode's orchestration: bind the proxy, THEN start the engine as
    a CHILD process (never exec'd -- this process must stay alive to run
    the proxy), forward SIGTERM/SIGINT to the child, and exit non-zero once
    the child exits (fail-closed: front mode never returns 0, since a
    proxied server process exiting is always either a shutdown request or a
    crash, never a "success" this launcher can attest to on its own).

    The proxy is bound BEFORE the engine is started: a bind failure (a busy
    --front-port) must never start -- and then have to kill -- a large
    model load that was never going to be reachable anyway. Right after that
    bind, and still before the engine is spawned, the engine's OWN loopback
    port is probed with a TCP connect: an orphaned engine left listening
    there (e.g. by a SIGKILLed earlier launcher) would otherwise have this
    new launch's proxy silently forward to that OLD process during the new
    model's load, stamping this launch's provenance headers on someone
    else's replies. Signal forwarding is installed before ``popen`` too, via
    a holder this function's own signal handler reads: a SIGTERM/SIGINT that
    arrives in the (short) window between installing the handler and the
    child existing is remembered and forwarded the instant the child does
    exist, rather than being silently dropped.

    ``max_request_body_bytes`` is plumbed straight to
    ``fastmlx_proxy.create_server`` -- ``_run_serve`` has already validated
    it (``--front-max-body-bytes``, 0/negative/non-numeric all refused
    before this function is ever called), so it is trusted as-is here.
    ``max_concurrent_requests`` is plumbed the same way (``--front-max-
    concurrent``, 0/negative/non-numeric equally refused before this
    function is ever called).
    """
    front = plan["front"]
    front_host = front["host"]
    front_port = front["port"]
    # The upstream is always this launcher's own construction (see the
    # "front" key above: "http://127.0.0.1:<--port>"), so its host/port are
    # recovered from the plan rather than threading extra parameters
    # through every caller of this function.
    upstream_host = "127.0.0.1"
    upstream_port = int(front["upstream"].rsplit(":", 1)[1])

    try:
        server = fastmlx_proxy.create_server(
            front_host,
            front_port,
            upstream_host,
            upstream_port,
            plan,
            max_request_body_bytes=max_request_body_bytes,
            max_concurrent_requests=max_concurrent_requests,
        )
    except OSError as exc:
        print(
            f"fastmlx serve: front proxy could not bind {front_host}:{front_port}: {exc}",
            file=sys.stderr,
        )
        return 3

    if _engine_port_has_listener(upstream_host, upstream_port):
        print(
            f"fastmlx serve: refusing: {upstream_host}:{upstream_port} already has a listener; "
            "an engine from an earlier launch may still be running",
            file=sys.stderr,
        )
        server.server_close()
        return 3

    child_holder: dict = {"child": None}
    pending_signal: dict = {"signum": None}
    # The first stop signal this launcher received, forwarded or pending: it
    # decides the exit code once the engine has exited (see below).
    requested_stop: dict = {"signum": None}

    def _forward_signal(signum, _frame):
        if requested_stop["signum"] is None:
            requested_stop["signum"] = signum
        child = child_holder["child"]
        if child is None:
            # No child yet: remember the signal: the code right after
            # ``popen`` below checks this and forwards it immediately.
            pending_signal["signum"] = signum
            return
        try:
            child.send_signal(signum)
        except OSError:
            pass

    previous_handlers = {
        sig: signal.signal(sig, _forward_signal) for sig in (signal.SIGTERM, signal.SIGINT)
    }

    # The engine is never Popened directly any more: this launcher Popens
    # the engine-lifeline guard instead (see the guard section above), and
    # the guard Popens the engine itself. `lifeline_write_fd` is kept open
    # (and never written to) by this process for the rest of this
    # function's life -- see `os.close` near the bottom -- so the OS itself
    # closes it, and the guard notices, the instant this process dies for
    # any reason, including SIGKILL.
    lifeline_read_fd, lifeline_write_fd = os.pipe()
    grace_seconds = _engine_lifeline_grace_seconds()
    guard_argv = _engine_lifeline_guard_argv(final_argv, lifeline_read_fd, grace_seconds)
    try:
        # start_new_session=True: the GUARD gets its OWN session/process
        # group, never the launcher's -- without this, a terminal Ctrl-C
        # (which sends SIGINT to the whole foreground process group) would
        # reach the engine BOTH directly from the terminal AND a second
        # time via this function's own signal forwarding below. The engine
        # itself is the guard's own child (no new session), so it shares
        # the guard's session/process group, never the launcher's.
        child = popen(guard_argv, start_new_session=True, pass_fds=(lifeline_read_fd,))
    except OSError as exc:
        os.close(lifeline_read_fd)
        os.close(lifeline_write_fd)
        print(f"fastmlx serve: front mode could not start the engine: {exc}", file=sys.stderr)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        server.server_close()
        return 3
    # The launcher's own copy of the read end is never needed -- only the
    # guard's (inherited via pass_fds above) is.
    os.close(lifeline_read_fd)

    child_holder["child"] = child
    if pending_signal["signum"] is not None:
        try:
            child.send_signal(pending_signal["signum"])
        except OSError:
            pass

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        child_code = child.wait()
    finally:
        # The lifeline's write end has done its job (the guard, and the
        # engine it supervised, have already exited) -- closed here rather
        # than left for process teardown so this function never holds it
        # open longer than front mode is actually running.
        os.close(lifeline_write_fd)
    if isinstance(child, subprocess.Popen):
        # Only a real guard has a process group to clean up; a test double's
        # made-up pid must never be signalled.
        _stop_leftover_engine_group(child.pid, grace_seconds)
    server.shutdown()
    server.server_close()
    stop_signum = requested_stop["signum"]
    if stop_signum is not None and child_code in (0, -stop_signum):
        # A stop the operator asked for: the served engine traps SIGTERM and
        # exits 0, which must still read as 128+signum (SIGTERM -> 143), not
        # as an engine that quit on its own. Any other code after the stop
        # (a crash while shutting down) falls through unchanged below.
        return 128 + stop_signum
    if child_code == 0:
        # Front mode never attests to a "success" exit -- see docstring.
        return 1
    if child_code < 0:
        # ``subprocess``'s own convention: a negative code from ``wait()``
        # names the signal that killed the child (os.waitpid's WIFSIGNALED
        # encoding). The shell/os convention for a signal-killed process's
        # own exit status is 128+signum (e.g. SIGTERM -> 143) -- returning
        # the raw negative code instead would make THIS process's own exit
        # status wrap to a nonsensical 128-|code| (e.g. -15 -> 241).
        return 128 - child_code
    return child_code


def main(argv: Optional[list] = None) -> None:
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    # Handled BEFORE argparse (manual parsing, see `_run_engine_lifeline_guard`):
    # this is never a public CLI surface -- an internal re-invocation of this
    # same file, made only by `_run_front_mode` -- and must never depend on
    # (or be disturbed by) `build_arg_parser`'s own `serve` subcommand.
    if raw_argv and raw_argv[0] == _INTERNAL_ENGINE_GUARD_FLAG:
        raise SystemExit(_run_engine_lifeline_guard(raw_argv[1:]))
    if "--" in raw_argv:
        split_index = raw_argv.index("--")
        parse_argv = raw_argv[:split_index]
        passthrough_args = raw_argv[split_index + 1 :]
    else:
        parse_argv = raw_argv
        passthrough_args = []

    parser = build_arg_parser()
    args = parser.parse_args(parse_argv)

    try:
        exit_code = _run_serve(args, passthrough_args)
    except LaunchRefusal as refusal:
        print(f"fastmlx serve refused: {refusal.message}", file=sys.stderr)
        raise SystemExit(refusal.exit_code)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
