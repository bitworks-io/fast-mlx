#!/usr/bin/env python3
"""``fastmlx recommend``: rank the local model packs that fit THIS host,
annotated with each pack's measured quality cost from the card store.

This is a read-only advisory front door, a sibling of ``fastmlx serve``
(``fastmlx_launch.py``): it never launches or execs anything. For each
candidate model directory it (1) resolves the model's repo/revision from
``.pull-receipt.json`` if one exists, (2) resolves a quality-guidance card
for that identity using the exact same lookup ``fastmlx serve`` uses
(``resolve_card`` from ``fastmlx_launch``), and (3) runs the same pre-load
fit check ``fastmlx serve`` runs (``run_fit_check``) against the requested
host-use and context. Rows are classified from those two independent
results; nothing is inferred, extrapolated, or fabricated. A candidate this
script cannot resolve cleanly (a card lookup that is ambiguous, a fit check
that cannot be run at all) becomes an explicit ``error`` row, never a
guess and never a crash.

v0 scope: only *carded* packs (a resolved quality-guidance card whose
verdict is one of REFERENCE/EXACT/PASS/NO_GO) are ever labeled with a
measured quality cost. A pack with no card, or a card whose verdict is
UNMEASURED or unrecognized, is reported as "uncarded" -- printed, but never
ranked above a carded pack, and never assigned a speed or accuracy number
this script did not read from a card.

Row classification (see the docstring above each helper for the exact
rule): ``recommended`` (fits, carded PASS/REFERENCE/EXACT), ``opt-in``
(fits, carded NO_GO -- shown with the exact ``--accept-quality`` flag that
would elect it for ``fastmlx serve``), ``uncarded`` (fits, no usable card),
``does-not-fit`` (fit check verdict RED), ``error`` (this candidate could
not be resolved: bad path, ambiguous card lookup, or the fit check itself
could not run).

Exit codes: ``0`` if at least one row is ``recommended``; ``1`` if none is
recommended but at least one row is ``opt-in`` or ``uncarded`` (something
fits, just nothing measured-good); ``2`` on a usage error (no candidates
resolved at all, an explicitly-named ``--quality-cards`` manifest that does
not load) or when every candidate is ``does-not-fit``/``error``.

Every subprocess this script starts is invoked as an argv list, via the
same ``fastmlx_launch.run_fit_check`` helper ``fastmlx serve`` uses --
never through a shell. This script never reads or prints environment
values (unlike ``fastmlx serve``, which may consult ``FASTMLX_FIT_CHECK_BIN``);
the fit-check binary is resolved from ``--fit-check-bin`` or, failing that,
by searching ``PATH`` for this repository's own in-tree serving binary.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Optional


_LAUNCH_PATH = Path(__file__).resolve().parent / "fastmlx_launch.py"
_LAUNCH_SPEC = importlib.util.spec_from_file_location("fastmlx_launch", _LAUNCH_PATH)
assert _LAUNCH_SPEC is not None and _LAUNCH_SPEC.loader is not None
launch = importlib.util.module_from_spec(_LAUNCH_SPEC)
_LAUNCH_SPEC.loader.exec_module(launch)


JSON_SCHEMA = "fastmlx-recommend-v0"

STATUS_RECOMMENDED = "recommended"
STATUS_OPT_IN = "opt-in"
STATUS_UNCARDED = "uncarded"
STATUS_DOES_NOT_FIT = "does-not-fit"
STATUS_ERROR = "error"

_STATUS_RANK = {
    STATUS_RECOMMENDED: 0,
    STATUS_OPT_IN: 1,
    STATUS_UNCARDED: 2,
    STATUS_DOES_NOT_FIT: 3,
    STATUS_ERROR: 4,
}

# Within `recommended`, a REFERENCE/EXACT card outranks a PASS card. The fit
# check's own pre-load attestation line carries no modeled-vs-measured
# headroom pairing (that pairing is a *post-load* drift report a different
# part of this repository emits; see FitCheckMeasuredReport.swift) -- so
# there is no honest headroom number to break ties on, and ties are left in
# stable discovery order rather than inventing one.
_CARD_VERDICT_RANK = {"REFERENCE": 0, "EXACT": 0, "PASS": 1}


# ---------------------------------------------------------------------
# Candidate discovery.
# ---------------------------------------------------------------------
def discover_candidates(model_paths: list, models_dir_paths: list) -> list:
    """The deduplicated candidate set: every explicit ``--model-path``, plus
    every immediate subdirectory of each ``--models-dir`` that contains a
    ``config.json``. Deduplicated by resolved path; order is preserved
    (explicit paths first, then each ``--models-dir``'s children in sorted
    order), since that order is this script's only stable tie-break for
    otherwise-equal rows.
    """
    seen: dict = {}
    ordered: list = []

    def add(path: Path) -> None:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key not in seen:
            seen[key] = path
            ordered.append(key)

    for path in model_paths:
        add(path)
    for models_dir in models_dir_paths:
        if not models_dir.is_dir():
            continue
        for child in sorted(models_dir.iterdir()):
            if child.is_dir() and (child / "config.json").is_file():
                add(child)

    return [seen[key] for key in ordered]


# ---------------------------------------------------------------------
# Per-candidate row.
# ---------------------------------------------------------------------
def _resolve_fit_check_bin(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    return shutil.which(launch._BUILT_IN_ENGINE_BINARY_NAME)


def _card_summary(card: Optional[dict]) -> Optional[dict]:
    if card is None:
        return None
    legible = card.get("legible") or {}
    summary = {
        "id": card.get("id"),
        "verdict": card.get("verdict"),
        "tier": legible.get("tier"),
        "headline": legible.get("headline"),
    }
    top1 = (legible.get("nextWordDrift") or {}).get("top1AgreementPct")
    if isinstance(top1, (int, float)) and not isinstance(top1, bool):
        summary["top1AgreementPct"] = top1
    speed_x = (legible.get("benefit") or {}).get("speedX")
    if isinstance(speed_x, (int, float)) and not isinstance(speed_x, bool):
        summary["speedX"] = speed_x
    return summary


def build_row(
    model_path: Path,
    cards: Optional[list],
    fit_check_bin: Optional[str],
    host_use: str,
    context: Optional[int],
    fit_check_args: list,
) -> dict:
    """Resolve one candidate to a fully-classified row. Never raises: an
    ambiguous card lookup (``LaunchRefusal`` from ``resolve_card``) or a fit
    check that cannot be run both become ``status="error"`` rows carrying
    the operator-facing reason, exactly like ``fastmlx serve`` would refuse
    -- but reported per-candidate here instead of aborting the whole run.
    """
    name = model_path.name
    row: dict = {
        "path": str(model_path),
        "name": name,
        "repo": None,
        "revision": None,
        "status": None,
        "fit": None,
        "card": None,
        "message": None,
        "accept_quality_flag": None,
    }

    if not model_path.is_dir():
        row["status"] = STATUS_ERROR
        row["message"] = f"model path {model_path} does not exist or is not a directory"
        return row
    if not (model_path / "config.json").is_file():
        row["status"] = STATUS_ERROR
        row["message"] = f"model path {model_path} does not contain config.json"
        return row

    receipt = launch._load_pull_receipt(model_path)
    model_repo = receipt.get("repo_id") if receipt else None
    model_repo = model_repo if isinstance(model_repo, str) else None
    model_revision = receipt.get("revision") if receipt else None
    model_revision = model_revision if isinstance(model_revision, str) else None
    row["repo"] = model_repo
    row["revision"] = model_revision

    try:
        card = launch.resolve_card(cards, model_repo, model_revision)
    except launch.LaunchRefusal as refusal:
        row["status"] = STATUS_ERROR
        row["message"] = refusal.message
        return row

    resolved_model_id = model_path.resolve().name
    fit_result = (
        launch.FitCheckResult(
            "error",
            detail=(
                "fit check binary not found; pass --fit-check-bin"
            ),
        )
        if not fit_check_bin
        else launch.run_fit_check(
            fit_check_bin=fit_check_bin,
            model_id=resolved_model_id,
            model_path=model_path,
            host_use=host_use,
            context=context,
            extra_args=fit_check_args,
        )
    )

    if fit_result.kind == "error":
        row["status"] = STATUS_ERROR
        row["message"] = f"fit check could not run: {fit_result.detail}"
        return row

    if fit_result.kind == "red":
        row["status"] = STATUS_DOES_NOT_FIT
        detail = (fit_result.stderr or "").strip() or "fit check verdict RED"
        row["fit"] = {"verdict": "RED", "context": context}
        row["message"] = detail
        return row

    # fit_result.kind == "green": an exit-0, attested verdict -- GREEN or
    # YELLOW are both a pass (mirrors fastmlx serve, which admits either
    # without --force; only RED is a refusal).
    fields = fit_result.fields
    fit_label = str(fields.get("fit_check", "green")).upper()
    reported_context = fields.get("fit_served_context") or (
        context if context is not None else fields.get("fit_context_ceiling")
    )
    row["fit"] = {"verdict": fit_label, "context": reported_context}

    outcome, message = launch.decide_admission(card, opted_in=False)
    if outcome == "admit":
        row["status"] = STATUS_RECOMMENDED
        row["card"] = _card_summary(card)
    elif outcome == "refuse_quality_flagged":
        row["status"] = STATUS_OPT_IN
        row["card"] = _card_summary(card)
        row["message"] = message
        row["accept_quality_flag"] = f"--accept-quality {card.get('id')}"
    else:  # "admit_unmeasured": no card, or an UNMEASURED/unrecognized verdict
        row["status"] = STATUS_UNCARDED
        row["card"] = _card_summary(card)
        row["message"] = "not recommended: no measured quality card"

    return row


# ---------------------------------------------------------------------
# Ranking.
# ---------------------------------------------------------------------
def _rank_key(row: dict) -> tuple:
    status_rank = _STATUS_RANK[row["status"]]
    verdict_rank = 0
    if row["status"] == STATUS_RECOMMENDED:
        verdict = (row.get("card") or {}).get("verdict")
        verdict_rank = _CARD_VERDICT_RANK.get(verdict, 1)
    return (status_rank, verdict_rank)


def rank_rows(rows: list) -> list:
    """Stable sort: recommended, opt-in, uncarded, does-not-fit, error; a
    carded row is never ranked below an uncarded one, and ties (including
    every non-``recommended`` bucket) keep the discovery order
    ``discover_candidates`` produced.
    """
    return sorted(rows, key=_rank_key)


def exit_code_for(rows: list) -> int:
    statuses = {row["status"] for row in rows}
    if STATUS_RECOMMENDED in statuses:
        return 0
    if statuses & {STATUS_OPT_IN, STATUS_UNCARDED}:
        return 1
    return 2


# ---------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------
def _format_row_text(rank: int, row: dict) -> str:
    head = [f"{rank}. {row['name']}"]
    if row["repo"]:
        rev = row["revision"]
        short_rev = f"@{rev[:12]}" if rev else ""
        head.append(f"repo={row['repo']}{short_rev}")
    elif row["revision"]:
        head.append(f"revision={row['revision'][:12]}")

    fit = row.get("fit")
    if fit:
        ctx = fit.get("context")
        ctx_part = f" context={ctx}" if ctx is not None else ""
        head.append(f"fit={fit['verdict']}{ctx_part}")

    card = row.get("card")
    if card:
        head.append(f"card={card['id']} verdict={card['verdict']} tier={card.get('tier')}")
        if "top1AgreementPct" in card:
            head.append(f"top1={card['top1AgreementPct']}%")
        if "speedX" in card:
            head.append(f"speedX={card['speedX']}x")

    lines = ["  ".join(head)]
    tail = f"    [{row['status']}]"
    if row.get("message"):
        tail += f" {row['message']}"
    flag = row.get("accept_quality_flag")
    if flag and flag not in (row.get("message") or ""):
        tail += f" ({flag})"
    lines.append(tail)
    return "\n".join(lines)


def format_text(rows: list) -> str:
    return "\n".join(_format_row_text(index + 1, row) for index, row in enumerate(rows))


# ---------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fastmlx_recommend",
        description=(
            "fastmlx recommend: for this host, rank the local packs that pass "
            "the fit check, annotated with each pack's measured quality cost."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    recommend = subparsers.add_parser(
        "recommend",
        help="rank local model packs that fit this host by measured quality",
    )
    recommend.add_argument("--model-path", action="append", default=[], type=Path)
    recommend.add_argument("--models-dir", action="append", default=[], type=Path)
    recommend.add_argument("--quality-cards", default=None)
    recommend.add_argument("--context", type=int, default=None)
    recommend.add_argument(
        "--host-use", default="shared", choices=["shared", "dedicated-serving"]
    )
    recommend.add_argument("--fit-check-bin", default=None)
    recommend.add_argument("--fit-check-arg", action="append", default=[])
    recommend.add_argument("--json", action="store_true")
    return parser


def _run_recommend(args) -> int:
    candidates = discover_candidates(args.model_path, args.models_dir)
    if not candidates:
        print(
            "fastmlx recommend: no model candidates found; pass --model-path "
            "and/or --models-dir",
            file=sys.stderr,
        )
        return 2

    quality_cards_path = Path(
        args.quality_cards
        if args.quality_cards is not None
        else (launch.REPO_ROOT / launch.DEFAULT_QUALITY_CARDS_RELATIVE_PATH)
    )
    cards = launch.load_quality_cards(quality_cards_path)
    if cards is None and args.quality_cards is not None:
        print(
            f"fastmlx recommend: the --quality-cards manifest {quality_cards_path} "
            "is missing or is not a quality-card manifest",
            file=sys.stderr,
        )
        return 2

    fit_check_bin = _resolve_fit_check_bin(args.fit_check_bin)

    rows = [
        build_row(
            model_path=candidate,
            cards=cards,
            fit_check_bin=fit_check_bin,
            host_use=args.host_use,
            context=args.context,
            fit_check_args=args.fit_check_arg,
        )
        for candidate in candidates
    ]
    rows = rank_rows(rows)

    if args.json:
        print(json.dumps({"schema": JSON_SCHEMA, "rows": rows}))
    else:
        text = format_text(rows)
        if text:
            print(text)

    return exit_code_for(rows)


def main(argv: Optional[list] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    raise SystemExit(_run_recommend(args))


if __name__ == "__main__":
    main()
