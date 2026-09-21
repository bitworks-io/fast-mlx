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

``--residency {resident,expert-stream}`` (default ``resident``) is forwarded
to ``resolve_card`` exactly like ``fastmlx serve`` uses it: a card only ever
matches a candidate when the card's own ``config.residency`` equals this
value (absent/null on a card means ``resident``). Every row also carries
its resolved ``residency``. A ``--fit-check-arg`` that itself names a
conflicting ``--residency`` is a usage error, exactly like ``fastmlx
serve`` refuses one.

Exit codes: ``0`` if at least one row is ``recommended``; ``1`` if none is
recommended but at least one row is ``opt-in`` or ``uncarded`` (something
fits, just nothing measured-good); ``2`` on a usage error (no candidates
resolved at all, an explicitly-named ``--quality-cards`` manifest that does
not load) or when every candidate is ``does-not-fit``/``error``.

Every subprocess this script starts is invoked as an argv list, via the
same ``fastmlx_launch.run_fit_check`` helper ``fastmlx serve`` uses --
never through a shell. This script never reads or prints environment
values (unlike ``fastmlx serve``, which may consult ``FASTMLX_FIT_CHECK_BIN``).
The fit-check binary is resolved, per candidate, from ``--fit-check-bin``
(explicit) > an engine profile's own ``fitCheck.bin`` > one of this
repository's built-in pure-Python sizers (``fastmlx_safetensors_fit.py`` /
``fastmlx_gguf_fit.py``), auto-selected from THAT candidate's own pack
contents (see ``_select_builtin_fit_check_bin``). This is a read-only
advisory front door that must be able to answer with Python alone: it
never falls back to searching ``PATH`` for this repository's Swift engine
binary. A built-in sizer never assumes a zero KV-cache reserve either --
when one is auto-selected, ``--kv-reserve-gib`` (forwarded to the sizer
verbatim) is required, and its absence is an actionable per-candidate
refusal naming the exact flag to pass.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional


_LAUNCH_PATH = Path(__file__).resolve().parent / "fastmlx_launch.py"
_LAUNCH_SPEC = importlib.util.spec_from_file_location("fastmlx_launch", _LAUNCH_PATH)
assert _LAUNCH_SPEC is not None and _LAUNCH_SPEC.loader is not None
launch = importlib.util.module_from_spec(_LAUNCH_SPEC)
_LAUNCH_SPEC.loader.exec_module(launch)

# _pack_has_safetensors / _pack_has_gguf / _uncounted_safetensors_message /
# _select_builtin_fit_check_bin now live in fastmlx_launch.py (the single
# source of truth `fastmlx serve`'s own auto-select fallback and this
# script share -- see that module's "0b. Built-in sizer auto-selection"
# section). Aliased here, unchanged, so every existing call site and test
# in this file that references them on the `recommend` module keeps
# working.
_pack_has_safetensors = launch._pack_has_safetensors
_pack_has_gguf = launch._pack_has_gguf
_uncounted_safetensors_message = launch._uncounted_safetensors_message
_select_builtin_fit_check_bin = launch._select_builtin_fit_check_bin


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
    every immediate subdirectory of each ``--models-dir`` that looks like a
    model pack (``launch.is_model_dir``: a ``config.json`` MLX layout, or a
    GGUF pack with at least one top-level ``*.gguf`` file). Deduplicated by
    resolved path; order is preserved (explicit paths first, then each
    ``--models-dir``'s children in sorted order), since that order is this
    script's only stable tie-break for otherwise-equal rows.
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
            if child.is_dir() and launch.is_model_dir(child):
                add(child)

    return [seen[key] for key in ordered]


# ---------------------------------------------------------------------
# Per-candidate row.
# ---------------------------------------------------------------------
def _resolve_fit_check_bin(explicit: Optional[str], profile_fit_check: Optional[dict]) -> tuple:
    """The fit-check binary and its profile-supplied leading extra args.

    Precedence mirrors ``fastmlx serve``'s (see ``_run_serve``), minus the
    ``FASTMLX_FIT_CHECK_BIN`` environment fallback -- this script never
    reads environment values at all: ``--fit-check-bin`` (explicit) >
    the engine profile's own ``fitCheck.bin``. Returns
    ``(fit_check_bin, profile_extra_args, overridden)`` where
    ``overridden`` is true exactly when an explicit ``--fit-check-bin``
    silently dropped a profile's own ``fitCheck.args``.

    When NEITHER is given, ``fit_check_bin`` is ``None`` -- the signal
    ``build_row`` uses to auto-select one of this repository's built-in
    pure-Python sizers from EACH candidate's own pack contents (see
    ``_select_builtin_fit_check_bin``). This never falls back to
    searching ``PATH`` for this repository's Swift engine binary
    (``launch._BUILT_IN_ENGINE_BINARY_NAME``) -- ``fastmlx recommend`` is a
    read-only advisory front door that must be able to answer with Python
    alone, without the ~25-minute MLX engine build that binary requires.
    """
    if explicit:
        return explicit, [], profile_fit_check is not None
    if profile_fit_check is not None:
        return profile_fit_check["bin"], list(profile_fit_check["args"]), False
    return None, [], False


# ``launch.NO_MODEL_IDENTITY_HINT`` names ``--model-revision``, a flag
# ``fastmlx serve`` accepts but ``fastmlx recommend``'s own parser does
# NOT (recommend's flags are exactly --context/--engine-profile/
# --fit-check-arg/--fit-check-bin/--host-use/--json/--kv-reserve-gib/
# --model-path/--models-dir/--quality-cards/--residency) -- printing the
# shared hint verbatim here would tell an operator to pass a flag this
# command rejects. The other half of the shared hint (the
# ``fastmlx pull ... --adopt`` remedy) IS valid regardless of which
# command is asking, so only the ``--model-revision`` clause is dropped
# here; `fastmlx_launch.py` is outside this file's write set, so the
# shared constant itself is intentionally left untouched. See
# ``RecommendNoIdentityHintFlagsTestCase`` in
# scripts/tests/test_fastmlx_recommend.py, which derives recommend's
# accepted flag set from its own parser so this text can never silently
# drift back to naming a flag recommend does not accept.
RECOMMEND_NO_MODEL_IDENTITY_HINT = (
    "no model identity (no pull receipt); no quality card was consulted -- "
    "run 'fastmlx pull <repo>@<revision> --dest <dir> --adopt' to pin a "
    "hand-staged pack"
)


def _print_recommend_no_model_identity_hint(subject: str) -> None:
    print(f"fastmlx recommend: {subject}: {RECOMMEND_NO_MODEL_IDENTITY_HINT}", file=sys.stderr)


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
    benefit = legible.get("benefit") or {}
    status = benefit.get("speedXStatus")
    if status is not None:
        # speedXStatus carries the only measurement boundary fast-mlx has
        # for a numeric speedX (host, engine build, flags, prompt count,
        # referent) -- a speedX key must NEVER appear in this summary
        # without it, so speedX is only ever added inside this branch.
        speed_x = benefit.get("speedX")
        if isinstance(speed_x, (int, float)) and not isinstance(speed_x, bool):
            summary["speedX"] = speed_x
        summary["speedXStatus"] = status
    fit = benefit.get("fit")
    if isinstance(fit, str) and fit:
        # Named `benefitFit`, deliberately NOT `fit`: `row["fit"]` (see
        # `build_row`) already holds the LIVE fit-check verdict dict
        # (`verdict`/`context`) this host measured for this candidate at
        # the requested context. A card's benefit.fit is a different fact
        # -- a footprint and which Mac CLASSES the pack fits, read from the
        # quality-card store. Two keys both named `fit` in one `--json`
        # document would be exactly the mislabeling class this field
        # exists to repair; keeping this independent of the speedXStatus
        # branch above is deliberate too, since a card can carry a fit
        # sentence with no measured speed at all (most published cards do).
        summary["benefitFit"] = fit
    return summary


def build_row(
    model_path: Path,
    cards: Optional[list],
    fit_check_bin: Optional[str],
    host_use: str,
    context: Optional[int],
    fit_check_args: list,
    residency: str = "resident",
    engine_build_commit: Optional[str] = None,
    mtp_launch: bool = False,
    kv_reserve_gib: Optional[float] = None,
) -> dict:
    """Resolve one candidate to a fully-classified row. Never raises: an
    ambiguous card lookup (``LaunchRefusal`` from ``resolve_card``) or a fit
    check that cannot be run both become ``status="error"`` rows carrying
    the operator-facing reason, exactly like ``fastmlx serve`` would refuse
    -- but reported per-candidate here instead of aborting the whole run.

    ``fit_check_bin is None`` means neither an explicit ``--fit-check-bin``
    nor an engine profile named a sizer -- this candidate's own pack
    contents decide which built-in pure-Python sizer runs (see
    ``_select_builtin_fit_check_bin``), never a PATH search for this
    repository's Swift engine binary. ``kv_reserve_gib``, when given, is
    forwarded verbatim as this fit check's own ``--kv-reserve-gib``; when
    it is ``None`` and a built-in sizer was auto-selected for this
    candidate, that is an actionable ``error`` row (never a silently
    assumed zero KV-cache reserve) naming the exact flag to pass.
    """
    name = model_path.name
    row: dict = {
        "path": str(model_path),
        "name": name,
        "repo": None,
        "revision": None,
        "residency": residency,
        "status": None,
        "fit": None,
        "card": None,
        "message": None,
        "accept_quality_flag": None,
        "engineBuild": None,
        "mtp": None,
    }

    if not model_path.is_dir():
        row["status"] = STATUS_ERROR
        row["message"] = f"model path {model_path} does not exist or is not a directory"
        return row
    if not launch.is_model_dir(model_path):
        row["status"] = STATUS_ERROR
        row["message"] = (
            f"model path {model_path} {launch.MODEL_DIR_REFUSAL_MESSAGE_SUFFIX}"
        )
        return row

    receipt = launch._load_pull_receipt(model_path)
    model_repo = receipt.get("repo_id") if receipt else None
    model_repo = model_repo if isinstance(model_repo, str) else None
    model_revision = receipt.get("revision") if receipt else None
    model_revision = model_revision if isinstance(model_revision, str) else None
    row["repo"] = model_repo
    row["revision"] = model_revision

    # No repo and no pinned revision at all means no card could ever match
    # this candidate by repo or by hfPin -- made visible here for the same
    # reason `fastmlx serve` makes it visible (`launch.NO_MODEL_IDENTITY_HINT`),
    # but through recommend's OWN hint text, not the shared constant: the
    # shared one names --model-revision, a flag this command's parser does
    # not accept (see `RECOMMEND_NO_MODEL_IDENTITY_HINT`). The row's
    # classification is unchanged either way.
    if model_repo is None and model_revision is None:
        _print_recommend_no_model_identity_hint(str(model_path))

    try:
        card = launch.resolve_card(
            cards,
            model_repo,
            model_revision,
            residency=residency,
            engine_build_commit=engine_build_commit,
        )
    except launch.LaunchRefusal as refusal:
        row["status"] = STATUS_ERROR
        row["message"] = refusal.message
        return row

    # Engine-build status/notice: informational only, never gates a row's
    # status/verdict (mirrors fastmlx serve -- see docs/quality-card-schema-v1.md
    # "Engine build"). Computed as soon as the card is known, regardless of
    # what the fit check below decides.
    card_build_commit = launch.card_engine_build_commit(card)
    build_status = launch.engine_build_status(card_build_commit, engine_build_commit)
    build_message = None
    if build_status in (
        launch.ENGINE_BUILD_STATUS_UNDECLARED,
        launch.ENGINE_BUILD_STATUS_MISMATCH,
    ):
        build_message = launch.engine_build_notice_text(
            card.get("id") if card else None, card_build_commit, engine_build_commit
        )
    row["engineBuild"] = {
        "status": build_status,
        "card": card_build_commit,
        "launch": engine_build_commit,
        "message": build_message,
    }

    # `--mtp` flag-transfer status/notice: informational only, never gates
    # a row's status/verdict (mirrors fastmlx serve -- see
    # docs/quality-card-schema-v1.md "Flag transfer"). Computed from the
    # same card/build_status this row already resolved, regardless of what
    # the fit check below decides.
    mtp_status, mtp_divergent_prompts, mtp_prompts = launch.mtp_transfer_status(
        card, build_status, mtp_launch
    )
    mtp_message = None
    if mtp_status not in (launch.MTP_STATUS_OFF, launch.MTP_STATUS_EXACT):
        mtp_message = launch.mtp_notice_text(
            card.get("id") if card else None,
            mtp_status,
            mtp_divergent_prompts,
            mtp_prompts,
            card_build_commit,
        )
    row["mtp"] = {
        "status": mtp_status,
        "divergentPrompts": mtp_divergent_prompts,
        "prompts": mtp_prompts,
        "message": mtp_message,
    }

    resolved_model_id = model_path.resolve().name

    if fit_check_bin:
        resolved_fit_check_bin = fit_check_bin
        resolved_fit_check_args = list(fit_check_args)
    else:
        # Neither --fit-check-bin nor an engine profile's own fitCheck
        # named a sizer: auto-select one of this repository's built-in
        # pure-Python sizers from THIS candidate's own pack contents (see
        # module docstring / _select_builtin_fit_check_bin). Never falls
        # back to searching PATH for the Swift engine binary.
        builtin_name, select_error = _select_builtin_fit_check_bin(model_path)
        if select_error is not None:
            row["status"] = STATUS_ERROR
            row["message"] = f"fit check could not run: {select_error}"
            return row
        if kv_reserve_gib is None:
            row["status"] = STATUS_ERROR
            row["message"] = (
                "fit check could not run: a built-in sizer "
                f"({builtin_name}) was auto-selected for this pack, which "
                "requires --kv-reserve-gib (a fit check must never "
                "silently assume a zero KV-cache reserve); pass "
                "--kv-reserve-gib"
            )
            return row
        resolved_fit_check_bin = launch._resolve_fit_check_bin_value(
            builtin_name, "<fastmlx recommend auto-selected built-in sizer>"
        )
        resolved_fit_check_args = list(fit_check_args)

    if kv_reserve_gib is not None:
        resolved_fit_check_args = resolved_fit_check_args + [
            "--kv-reserve-gib",
            str(kv_reserve_gib),
        ]

    fit_result = launch.run_fit_check(
        fit_check_bin=resolved_fit_check_bin,
        model_id=resolved_model_id,
        model_path=model_path,
        host_use=host_use,
        context=context,
        extra_args=resolved_fit_check_args,
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

    # A GREEN attestation carrying its own residency= field must agree
    # with the requested --residency, exactly like fastmlx serve's
    # fail-closed check: a sizer that sized the wrong residency must never
    # be reported as a recommended/fitting row for THIS residency. A
    # binary that omits the field (e.g. the Swift built-in binary) is not
    # checked.
    attested_residency = fields.get("residency")
    if attested_residency is not None and attested_residency != residency:
        row["status"] = STATUS_ERROR
        row["message"] = (
            f"fit check attested residency={attested_residency!r}, which "
            f"differs from the requested --residency {residency!r}"
        )
        return row

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

    if row.get("residency") not in (None, "resident"):
        head.append(f"residency={row['residency']}")

    card = row.get("card")
    prints_a_quality_number = False
    if card:
        head.append(f"card={card['id']} verdict={card['verdict']} tier={card.get('tier')}")
        if "top1AgreementPct" in card:
            head.append(f"top1={card['top1AgreementPct']}%")
            prints_a_quality_number = True
        if "speedX" in card:
            prints_a_quality_number = True

    lines = ["  ".join(head)]

    # The card's one self-contained sentence: printed for any row that
    # prints a quality number at all, never discarded.
    if prints_a_quality_number and card.get("headline"):
        lines.append(f"    {card['headline']}")

    # A speed ratio must NEVER be shown without its speedXStatus scope
    # (host, engine build, flags, prompt count, and the referent it is a
    # ratio against) -- see launch.card_benefit_line, which this
    # reconstructs a minimal pseudo-card for since only the flattened
    # summary (not the raw card) survives into a row.
    if card:
        benefit_line = launch.card_benefit_line(
            {
                "legible": {
                    "benefit": {
                        "speedX": card.get("speedX"),
                        "speedXStatus": card.get("speedXStatus"),
                    }
                }
            }
        )
        if benefit_line:
            lines.append(f"    speed: {benefit_line}")

        # Labelled `card fit:`, never a bare `fit:` -- the row HEAD above
        # already prints `fit=<verdict> context=<n>`, THIS host's live
        # measured verdict from the fit-check binary at the requested
        # context. The card's sentence is a DIFFERENT fact (a footprint
        # and which Mac classes the pack fits, read from the quality-card
        # store) that can disagree with the head's verdict on this very
        # row -- two bare "fit"s that can disagree is worse than the
        # omission this repairs. `card fit:` attributes the sentence to
        # its source instead of asserting a provenance ("sized from
        # headers" or similar) the card's own sentence does not claim.
        fit_line = launch.card_fit_line(
            {"legible": {"benefit": {"fit": card.get("benefitFit")}}}
        )
        if fit_line:
            lines.append(f"    card fit: {fit_line}")

    tail = f"    [{row['status']}]"
    if row.get("message"):
        tail += f" {row['message']}"
    flag = row.get("accept_quality_flag")
    if flag and flag not in (row.get("message") or ""):
        tail += f" ({flag})"
    engine_build = row.get("engineBuild")
    build_message = engine_build.get("message") if engine_build else None
    if build_message and build_message not in (row.get("message") or ""):
        tail += f" {build_message}"
    mtp = row.get("mtp")
    mtp_message = mtp.get("message") if mtp else None
    if mtp_message and mtp_message not in (row.get("message") or ""):
        tail += f" {mtp_message}"
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
    recommend.add_argument(
        "--model-path",
        action="append",
        default=[],
        type=Path,
        help="a model directory to include as a candidate (repeatable)",
    )
    recommend.add_argument(
        "--models-dir",
        action="append",
        default=[],
        type=Path,
        help=(
            "a directory whose immediate model-pack subdirectories are all "
            "added as candidates (repeatable)"
        ),
    )
    recommend.add_argument(
        "--quality-cards",
        default=None,
        help=(
            "path to the quality-card manifest (default: "
            f"{launch.DEFAULT_QUALITY_CARDS_RELATIVE_PATH} under the repo "
            "root; an explicitly-named manifest that fails to load is a "
            "usage error, unlike the default path)"
        ),
    )
    recommend.add_argument(
        "--residency", default="resident", choices=list(launch.RESIDENCIES),
        help=(
            "the residency each candidate's fit check and card lookup is "
            "evaluated against (default 'resident')"
        ),
    )
    recommend.add_argument(
        "--context",
        type=int,
        default=None,
        help=(
            "the context length forwarded to the fit-check binary's own "
            "--context (omit to use the fit-check binary's default)"
        ),
    )
    recommend.add_argument(
        "--host-use",
        default="shared",
        choices=["shared", "dedicated-serving"],
        help=(
            "the host-sharing mode forwarded to the fit check (default "
            "'shared')"
        ),
    )
    recommend.add_argument(
        "--fit-check-bin",
        default=None,
        help=(
            "the fit-check binary to run, overriding the engine profile's "
            "own fitCheck.bin (default: the profile's fitCheck.bin, else a "
            "built-in pure-Python sizer auto-selected from each "
            "candidate's own pack contents -- never a PATH search for "
            "this repository's Swift engine binary)"
        ),
    )
    recommend.add_argument(
        "--fit-check-arg",
        action="append",
        default=[],
        help="an extra argv token appended to the fit-check invocation (repeatable)",
    )
    recommend.add_argument(
        "--kv-reserve-gib",
        type=float,
        default=None,
        help=(
            "the KV-cache reserve (GiB) forwarded to the fit check as its "
            "own --kv-reserve-gib; required when a built-in sizer is "
            "auto-selected for a candidate (no --fit-check-bin and no "
            "engine-profile fitCheck) -- a fit check must never silently "
            "assume a zero KV-cache reserve"
        ),
    )
    recommend.add_argument(
        "--engine-profile",
        default=None,
        help="path to an engine-profile JSON file (default: the built-in engine profile)",
    )
    recommend.add_argument(
        "--json",
        action="store_true",
        help="print the ranked rows as a JSON object instead of formatted text",
    )
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

    try:
        profile, _ = launch.load_engine_profile(args.engine_profile)
    except launch.LaunchRefusal as refusal:
        print(f"fastmlx recommend: {refusal.message}", file=sys.stderr)
        return refusal.exit_code

    # Read only for engine-build status/notice (see build_row); this script
    # never execs anything, so a profile's engineBuild.binarySha256 is never
    # verified here.
    engine_build_commit = (profile.get("engineBuild") or {}).get("commit")
    # Whether this run's engine profile carries the exact token `--mtp` --
    # recommend never execs anything and has no passthrough concept, so
    # this only ever scans the profile's own (and, for --residency
    # expert-stream, the streaming) argv.
    mtp_launch = launch.mtp_launch_requested(profile, args.residency)

    fit_check_bin, profile_extra_args, overridden = _resolve_fit_check_bin(
        args.fit_check_bin, profile.get("fitCheck")
    )
    if overridden:
        print(
            f"fastmlx recommend: --fit-check-bin overrides engine profile "
            f"{profile['name']!r}'s own fitCheck; its fitCheck.args are not "
            "applied",
            file=sys.stderr,
        )
    combined_fit_check_args = profile_extra_args + args.fit_check_arg

    residency_conflict = launch._residency_conflict_source_and_value(
        profile_extra_args, args.fit_check_arg, args.residency
    )
    if residency_conflict is not None:
        source_label, item, value = residency_conflict
        value_desc = "no value" if value is None else repr(value)
        print(
            f"fastmlx recommend: {source_label} specifies {item!r} "
            f"({value_desc}), which differs from --residency "
            f"{args.residency!r}",
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

    rows = [
        build_row(
            model_path=candidate,
            cards=cards,
            fit_check_bin=fit_check_bin,
            host_use=args.host_use,
            context=args.context,
            fit_check_args=combined_fit_check_args,
            residency=args.residency,
            engine_build_commit=engine_build_commit,
            mtp_launch=mtp_launch,
            kv_reserve_gib=args.kv_reserve_gib,
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
