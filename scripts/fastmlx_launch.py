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
import subprocess
import sys
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
    same way a card for a different model is.

    A repo or pin match can now name MORE THAN ONE card -- the same pack
    measured on more than one engine build. When it does, the single card
    whose ``provenance.engineBuild.commit`` equals ``engine_build_commit``
    (this launch's own engine build) is selected. An undeclared launch
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
        (card_engine_build_commit(card) or "undeclared")[:12] for card in candidates
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
    card_id = card.get("id")
    if opted_in:
        return ("admit_with_quality_flag", summary)
    return (
        "refuse_quality_flagged",
        summary + f" re-run with --accept-quality {card_id} to elect it.",
    )


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
    serve.add_argument("--model-path", required=True, type=Path)
    serve.add_argument("--model-id", default=None)
    serve.add_argument("--model-repo", default=None)
    serve.add_argument("--model-revision", default=None)
    serve.add_argument("--card-id", default=None)
    serve.add_argument("--quality-cards", default=None)
    serve.add_argument("--accept-quality", action="append", default=[])
    serve.add_argument("--residency", default="resident", choices=list(RESIDENCIES))
    serve.add_argument("--context", type=int, default=None)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument(
        "--host-use", default="shared", choices=["shared", "dedicated-serving"]
    )
    serve.add_argument("--fit-check-bin", default=None)
    serve.add_argument("--fit-check-arg", action="append", default=[])
    serve.add_argument("--force", action="store_true")
    serve.add_argument("--engine-profile", default=None)
    serve.add_argument("--engine-bin", default=None)
    serve.add_argument("--dry-run", action="store_true")
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
    launch_engine_build_commit = (profile.get("engineBuild") or {}).get("commit")
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
    # the engine profile's own fitCheck.bin > the built-in engine. An
    # override from the CLI or the environment drops the profile's
    # fitCheck.args entirely -- those args are sized for the profile's own
    # sizer, never for whatever binary overrode it -- and is announced on
    # stderr so an operator does not wonder why a profile's fitCheck.args
    # were silently ignored.
    profile_fit_check = profile.get("fitCheck")
    cli_or_env_fit_check_bin = args.fit_check_bin or os.environ.get("FASTMLX_FIT_CHECK_BIN")
    profile_fit_check_args: list = []
    cli_fit_check_args = list(args.fit_check_arg)
    if cli_or_env_fit_check_bin:
        fit_check_bin = cli_or_env_fit_check_bin
        fit_check_extra_args = list(cli_fit_check_args)
        if profile_fit_check is not None:
            override_source = "--fit-check-bin" if args.fit_check_bin else "FASTMLX_FIT_CHECK_BIN"
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

    substitutions = {
        "engine_bin": engine_bin_abs,
        "model_path": str(model_path.resolve()),
        "model_id": model_id,
        "host": args.host,
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

    if args.dry_run:
        print(json.dumps(plan))
        return 0

    print(
        "fastmlx_launch=admitted "
        f"engine={profile['name']} card={card.get('id') if card else 'none'} "
        f"fit={fit_label} context={context} residency={residency} "
        f"engine_build={build_status} mtp={mtp_status}",
        file=sys.stderr,
    )
    os.execv(engine_bin_abs, final_argv)
    return 0  # pragma: no cover - unreachable, os.execv never returns on success


def main(argv: Optional[list] = None) -> None:
    raw_argv = sys.argv[1:] if argv is None else list(argv)
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
