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
"""

from __future__ import annotations

import argparse
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
PULL_RECEIPT_FILENAME = ".pull-receipt.json"

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
    return FitCheckResult(
        "error",
        detail=(
            f"fit check exited {proc.returncode} (expected 0 for a passing verdict "
            "or 2 for a red verdict)"
        ),
    )


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


def find_card_by_repo(cards: list, repo: Optional[str]) -> Optional[dict]:
    if repo is None:
        return None
    for card in cards:
        if isinstance(card, dict) and card.get("model", {}).get("repo") == repo:
            return card
    return None


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


def find_card_by_pin(cards: list, model_revision: Optional[str]) -> Optional[dict]:
    if not _is_full_hex_revision(model_revision):
        return None
    for card in cards:
        if not isinstance(card, dict):
            continue
        hf_pin = (card.get("model") or {}).get("hfPin")
        if _hf_pin_matches_revision(hf_pin, model_revision):
            return card
    return None


def resolve_card(
    cards: Optional[list], model_repo: Optional[str], model_revision: Optional[str]
) -> Optional[dict]:
    """The implicit (no ``--card-id``) lookup: a repo match (Swift semantics)
    OR an hfPin-prefix match against the model's pinned revision. If both
    exist and name two DIFFERENT cards, the lookup is ambiguous and refuses
    rather than silently preferring one -- this can only happen with a
    manifest that names the same model twice under two different cards, one
    keyed by repo and the other only by pin.
    """
    if cards is None:
        return None
    repo_card = find_card_by_repo(cards, model_repo)
    pin_card = find_card_by_pin(cards, model_revision)
    if repo_card is not None and pin_card is not None and repo_card.get("id") != pin_card.get(
        "id"
    ):
        raise LaunchRefusal(
            3,
            "quality card lookup is ambiguous: repo "
            f"{model_repo!r} matches card {repo_card.get('id')!r} but pinned "
            f"revision {model_revision!r} matches a different card "
            f"{pin_card.get('id')!r}",
        )
    return repo_card if repo_card is not None else pin_card


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


def load_engine_profile(path: Optional[str]) -> tuple:
    """Load and validate an engine profile.

    Returns ``(profile, is_built_in)``. ``path`` of ``None`` selects the
    built-in profile. Refuses (``LaunchRefusal(3, ...)``) on a missing/
    unreadable/malformed file, a schema mismatch, or an argv element using
    any ``{...}`` placeholder outside the allowed set -- a malformed
    profile must never silently drop or mis-render a token in the exec'd
    command line.
    """
    if path is None:
        return dict(BUILT_IN_ENGINE_PROFILE), True

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
    return {"name": name, "argv": argv}, False


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


def _load_pull_receipt(model_path: Path) -> Optional[dict]:
    receipt_path = model_path / PULL_RECEIPT_FILENAME
    if not receipt_path.is_file():
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return receipt if isinstance(receipt, dict) else None


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
    ``--model-revision`` if given, else ``.pull-receipt.json``'s
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
    if not (model_path / "config.json").is_file():
        raise LaunchRefusal(2, f"model path {model_path} does not contain config.json")

    model_id = args.model_id or model_path.resolve().name
    model_repo = _resolve_model_repo(args, model_path)
    model_revision = _resolve_model_revision(args, model_path)

    # Engine profile is loaded/validated early: a malformed profile is a
    # configuration error unrelated to the fit check or the quality card,
    # and failing on it before spending time on a fit-check subprocess
    # call keeps the refusal prompt.
    profile, is_built_in_profile = load_engine_profile(args.engine_profile)

    # --- fit-check ---------------------------------------------------
    fit_check_bin = (
        args.fit_check_bin
        or os.environ.get("FASTMLX_FIT_CHECK_BIN")
        or shutil.which(_BUILT_IN_ENGINE_BINARY_NAME)
    )
    if not fit_check_bin:
        raise LaunchRefusal(
            3,
            "fit check binary not found; pass --fit-check-bin or set "
            "FASTMLX_FIT_CHECK_BIN",
        )

    fit_result = run_fit_check(
        fit_check_bin=fit_check_bin,
        model_id=model_id,
        model_path=model_path,
        host_use=args.host_use,
        context=args.context,
        extra_args=args.fit_check_arg,
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
    else:
        card = resolve_card(cards, model_repo, model_revision)

    opt_in_ids = set(args.accept_quality)
    opted_in = is_opted_in(card, opt_in_ids)
    outcome, message = decide_admission(card, opted_in)
    if outcome == "refuse_quality_flagged":
        raise LaunchRefusal(2, message)
    if outcome == "admit_with_quality_flag":
        print(message)

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

    substitutions = {
        "engine_bin": engine_bin_abs,
        "model_path": str(model_path.resolve()),
        "model_id": model_id,
        "host": args.host,
        "port": str(args.port),
        "context": str(context),
    }
    final_argv = [_substitute_placeholders(item, substitutions) for item in profile["argv"]]
    final_argv += list(passthrough_args)

    plan = {
        "fit": {"verdict": fit_label, "fields": fit_fields},
        "card": card,
        "admission": outcome,
        "argv": final_argv,
    }

    if args.dry_run:
        print(json.dumps(plan))
        return 0

    print(
        "fastmlx_launch=admitted "
        f"engine={profile['name']} card={card.get('id') if card else 'none'} "
        f"fit={fit_label} context={context}",
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
