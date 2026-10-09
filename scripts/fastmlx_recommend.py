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
not load) or when every candidate is ``does-not-fit``/``error``; ``3`` is a
launch refusal (a ``--quality-cards-sha256`` pin refusal or an engine-profile
refusal).

``--pinned <repo>@<40-hex sha>`` (repeatable, combinable with local
candidates) answers the same two questions BEFORE any download: the fit from
the Hugging Face revision manifest's file sizes (the same manifest
``fastmlx pull`` fetches for its free-space check; sizes summed by
``fastmlx_safetensors_fit.weights_from_manifest`` under the local sizer's
rules -- ``*.safetensors`` are weights, dot-prefixed paths are ignored, an
unnamed non-safetensors file of 1 GiB or more is refused) and the card from
the same repo/``hfPin`` lookup. The row uses the same statuses, carries
``source: "revision-manifest"`` and ``downloaded: false``, and prints the
next step ``fastmlx pull <ref> --dest <dir>`` (with the ``--accept-quality``
flag when the row is opt-in). Nothing is downloaded or written. A malformed
ref (branch name, short sha, uppercase hex, no ``@``) is a usage error (exit
2) before any fetch. A manifest verdict is a size-based estimate: shard
headers and ``model.safetensors.index.json`` cannot be read before a
download, so the local checks on those only run once the pack is pulled. A
pinned row is an ``error`` row for ``--residency expert-stream``, an explicit
``--fit-check-bin`` or an engine profile's ``fitCheck.bin``, a missing
``--kv-reserve-gib``, a failed manifest fetch, or a manifest with no
``*.safetensors`` (e.g. a GGUF-only pack); other rows are unaffected.

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
import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import Callable, Optional


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


# The pinned-snapshot downloader and ``<repo>@<sha>`` validation `fastmlx
# pull` itself uses (``launch.pull`` already loaded them by file path), reused
# so a ``--pinned`` ref and its revision manifest are read by exactly the
# code `pull` reads them with -- no separate HTTP path. Exposed as module
# attributes so tests can patch ``downloader.fetch_api``.
downloader = launch.pull.downloader
validate_pinned_reference = launch.pull.validate_pinned_reference
PinnedReferenceError = launch.pull.PinnedReferenceError
_safetensors_fit = launch._safetensors_fit

JSON_SCHEMA = "fastmlx-recommend-v0"
SOURCE_REVISION_MANIFEST = "revision-manifest"

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
    "no model identity (no pull receipt, not a Hugging Face cache snapshot); "
    "no quality card was consulted -- "
    "run 'fastmlx pull <repo>@<revision> --dest <dir> --adopt' to pin a "
    "hand-staged pack"
)


def _print_recommend_no_model_identity_hint(subject: str) -> None:
    print(f"fastmlx recommend: {subject}: {RECOMMEND_NO_MODEL_IDENTITY_HINT}", file=sys.stderr)


def _card_summary(card: Optional[dict]) -> Optional[dict]:
    if card is None:
        return None
    # The loader does no card-shape validation, so every nested field below
    # is guarded with ``launch._as_dict`` (not ``... or {}``, which lets a
    # truthy non-dict value -- a string, a list -- through unchanged and
    # crashes the next ``.get()``): a malformed ``legible``/``benefit``/
    # ``nextWordDrift`` must never crash this summary.
    legible = launch._as_dict(card.get("legible"))
    summary = {
        "id": card.get("id"),
        "verdict": card.get("verdict"),
        "tier": legible.get("tier"),
        "headline": legible.get("headline"),
    }
    top1 = launch._as_dict(legible.get("nextWordDrift")).get("top1AgreementPct")
    if isinstance(top1, (int, float)) and not isinstance(top1, bool):
        summary["top1AgreementPct"] = top1
    # The generation length the card was measured over (boundary.
    # measuredNewTokens). Read through the proxy's strict reader -- the one
    # validation site -- so a bool/float/string/<1 value never surfaces.
    measured_new_tokens = launch.fastmlx_proxy.card_measured_new_tokens(card)
    if measured_new_tokens is not None:
        summary["measuredNewTokens"] = measured_new_tokens
    benefit = launch._as_dict(legible.get("benefit"))
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


def _resolve_card_into_row(
    row: dict,
    cards: Optional[list],
    model_repo: Optional[str],
    model_revision: Optional[str],
    residency: str,
    engine_build_commit: Optional[str],
    mtp_launch: bool,
    resolved_host_hardware_class: Callable[[], Optional[str]],
) -> tuple:
    """The card lookup plus the informational notices every row carries
    (tiebreak/verdict notices, engineBuild, mtp), written into ``row``.
    Shared by local rows (``build_row``) and ``--pinned`` rows
    (``build_pinned_row``) so both resolve identity exactly the same way.
    Returns ``(card, ok)``; ``ok`` is false when the lookup refused (an
    ambiguous match), in which case ``row`` is already a finished ``error``
    row.
    """
    notices: list = []
    try:
        card = launch.resolve_card(
            cards,
            model_repo,
            model_revision,
            residency=residency,
            engine_build_commit=engine_build_commit,
            host_hardware_class=resolved_host_hardware_class,
            notices=notices,
            # Kept apart so `notices` holds only the tiebreak notice.
            shape_notices=[],
        )
    except launch.LaunchRefusal as refusal:
        row["status"] = STATUS_ERROR
        row["message"] = refusal.message
        return None, False

    # Set immediately after resolve_card returns, before the
    # engineBuild/mtp/fit-check logic below, so a does-not-fit or error row
    # (returned further down) still carries whichever notice fired here --
    # mirrors how `row["engineBuild"]`/`row["mtp"]` are always present
    # (initialized above) even on rows that return early.
    row["tiebreakNotice"] = notices[0] if notices else None
    # Same reasoning: an unrecognized/missing-verdict notice must survive
    # into a does-not-fit/error row too, never only a recommended/opt-in/
    # uncarded one -- see the "not recommended: no measured quality card"
    # override further down for the uncarded case specifically.
    row["verdictNotice"] = launch.unrecognized_verdict_notice(card)

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
    return card, True


def _kv_reserve_required_message(builtin_name: str) -> str:
    """The per-candidate refusal for a built-in sizer with no
    ``--kv-reserve-gib`` (shared by local and ``--pinned`` rows)."""
    return (
        "fit check could not run: a built-in sizer "
        f"({builtin_name}) was auto-selected for this pack, which "
        "requires --kv-reserve-gib (a fit check must never "
        "silently assume a zero KV-cache reserve); pass "
        "--kv-reserve-gib"
    )


def _classify_fitting_row(row: dict, card: Optional[dict]) -> None:
    """Status of a row whose fit check passed (GREEN/YELLOW), from its card
    alone: ``recommended`` / ``opt-in`` / ``uncarded`` (see ``build_row``)."""
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
        # A card WITH an unrecognized verdict is a different fact than "no
        # measured quality card at all" (no card, or an explicit
        # UNMEASURED) -- show the verdict notice instead of the generic
        # message so an operator can tell a misspelled/malformed verdict
        # apart from a pack that was simply never measured.
        row["message"] = row["verdictNotice"] or "not recommended: no measured quality card"


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
    host_hardware_class: Optional[Callable[[], Optional[str]]] = None,
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

    ``host_hardware_class``, when given, overrides which callable is passed
    to ``resolve_card`` as ITS OWN ``host_hardware_class`` argument -- used
    by tests to simulate a specific host without touching the real sysctl
    call. When ``None`` (the default, and what every real CLI invocation
    uses), it is resolved to ``launch.host_hardware_class`` HERE, at call
    time, rather than defaulted in this function's own signature: a
    default bound at *def* time would capture launch's host_hardware_class
    as it existed when this module was first imported, so a test that
    monkeypatches ``launch.host_hardware_class`` afterwards would silently
    miss it -- the exact trap ``resolve_card``'s own
    ``host_hardware_class=host_hardware_class`` default parameter has (see
    that function's call sites in ``fastmlx_launch.py``).
    """
    resolved_host_hardware_class = (
        host_hardware_class if host_hardware_class is not None else launch.host_hardware_class
    )
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
        "tiebreakNotice": None,
        "verdictNotice": None,
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

    # The SAME identity `fastmlx serve` and the engine judge a pack by: the
    # sibling pull receipt, else the Hugging Face hub-cache path. A receipt
    # and a path naming different repos refuse (`quality_card_identity_conflict`)
    # -- reported on THIS row only, like a `resolve_card` refusal below.
    try:
        identity = launch.derive_pack_identity(model_path)
    except launch.LaunchRefusal as refusal:
        row["status"] = STATUS_ERROR
        row["message"] = refusal.message
        return row
    model_repo, model_revision = identity if identity is not None else (None, None)
    row["repo"] = model_repo
    row["revision"] = model_revision

    # No repo and no pinned revision at all (no receipt, not a hub-cache
    # snapshot) means no card could ever match this candidate by repo or by
    # hfPin -- made visible here for the same
    # reason `fastmlx serve` makes it visible (`launch.NO_MODEL_IDENTITY_HINT`),
    # but through recommend's OWN hint text, not the shared constant: the
    # shared one names --model-revision, a flag this command's parser does
    # not accept (see `RECOMMEND_NO_MODEL_IDENTITY_HINT`). The row's
    # classification is unchanged either way.
    if identity is None:
        _print_recommend_no_model_identity_hint(str(model_path))

    card, resolved_ok = _resolve_card_into_row(
        row,
        cards,
        model_repo,
        model_revision,
        residency,
        engine_build_commit,
        mtp_launch,
        resolved_host_hardware_class,
    )
    if not resolved_ok:
        return row

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
            row["card"] = _card_summary(card)
            row["message"] = f"fit check could not run: {select_error}"
            return row
        if kv_reserve_gib is None:
            row["status"] = STATUS_ERROR
            row["card"] = _card_summary(card)
            row["message"] = _kv_reserve_required_message(builtin_name)
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
        row["card"] = _card_summary(card)
        row["message"] = f"fit check could not run: {fit_result.detail}"
        return row

    if fit_result.kind == "red":
        row["status"] = STATUS_DOES_NOT_FIT
        detail = (fit_result.stderr or "").strip() or "fit check verdict RED"
        row["fit"] = {"verdict": "RED", "context": context}
        row["card"] = _card_summary(card)
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
        row["card"] = _card_summary(card)
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

    _classify_fitting_row(row, card)
    return row


# ---------------------------------------------------------------------
# Pinned (revision-manifest) row: a fit + card verdict BEFORE any download.
# ---------------------------------------------------------------------
def build_pinned_row(
    ref: str,
    cards: Optional[list],
    fit_check_bin: Optional[str],
    host_use: str,
    context: Optional[int],
    fit_check_args: list,
    residency: str = "resident",
    engine_build_commit: Optional[str] = None,
    mtp_launch: bool = False,
    kv_reserve_gib: Optional[float] = None,
    host_hardware_class: Optional[Callable[[], Optional[str]]] = None,
) -> dict:
    """Resolve one ``--pinned <repo>@<sha>`` ref to a classified row, fetching
    its Hugging Face revision manifest (``downloader.fetch_api`` +
    ``validated_entries``) only AFTER every option refusal below has passed,
    then classifying it with ``_classify_pinned_row`` -- the same core
    ``classify_pinned_entries`` runs over entries a caller already holds.
    See ``_classify_pinned_row`` for the full contract."""

    def fetch_entries() -> list:
        repo, revision = validate_pinned_reference(ref)
        document, _api_data = downloader.fetch_api(repo, revision)
        return downloader.validated_entries(document, repo, revision)

    return _classify_pinned_row(
        ref,
        fetch_entries,
        cards,
        fit_check_bin,
        host_use,
        context,
        fit_check_args,
        residency=residency,
        engine_build_commit=engine_build_commit,
        mtp_launch=mtp_launch,
        kv_reserve_gib=kv_reserve_gib,
        host_hardware_class=host_hardware_class,
    )


def classify_pinned_entries(
    ref: str,
    entries: list,
    cards: Optional[list],
    host_use: str,
    context: Optional[int],
    fit_check_args: list,
    kv_reserve_gib: Optional[float],
    residency: str = "resident",
    fit_check_bin: Optional[str] = None,
    engine_build_commit: Optional[str] = None,
    mtp_launch: bool = False,
    host_hardware_class: Optional[Callable[[], Optional[str]]] = None,
) -> dict:
    """Classify one pinned ref from manifest ``entries`` the caller ALREADY
    holds (``downloader.validated_entries`` output) -- nothing is fetched or
    written. Returns exactly the row ``build_pinned_row`` returns for the
    same manifest (same card lookup, sizing, status and ``next_step``); this
    is how ``fastmlx pull --kv-reserve-gib`` judges the manifest it is about
    to download without a second fetch. Never raises."""
    return _classify_pinned_row(
        ref,
        lambda: entries,
        cards,
        fit_check_bin,
        host_use,
        context,
        fit_check_args,
        residency=residency,
        engine_build_commit=engine_build_commit,
        mtp_launch=mtp_launch,
        kv_reserve_gib=kv_reserve_gib,
        host_hardware_class=host_hardware_class,
    )


def load_default_card_store(notice_prefix: str) -> list:
    """The cards of the default store ``recommend``/``serve`` resolve (the
    newest eligible pulled store, else the bundled one), with the same
    integrity checks. Unlike ``recommend``'s fail-open-to-"no card" default,
    a store that cannot be resolved or read raises ``launch.LaunchRefusal``:
    a caller that asked for a card verdict must not get "uncarded" because
    the store silently failed to load."""
    path, source, notices = launch.resolve_quality_card_store(
        None, notice_prefix=notice_prefix
    )
    for notice in notices:
        print(notice, file=sys.stderr)
    raw_sha256, cards, _identity = launch._inspect_quality_card_store(path)
    launch.enforce_pulled_store_identity(path, source, raw_sha256)
    if cards is None:
        raise launch.LaunchRefusal(
            3, f"the quality-card store {path} is missing or is not a quality-card manifest"
        )
    return cards


def _classify_pinned_row(
    ref: str,
    fetch_entries: Callable[[], list],
    cards: Optional[list],
    fit_check_bin: Optional[str],
    host_use: str,
    context: Optional[int],
    fit_check_args: list,
    residency: str = "resident",
    engine_build_commit: Optional[str] = None,
    mtp_launch: bool = False,
    kv_reserve_gib: Optional[float] = None,
    host_hardware_class: Optional[Callable[[], Optional[str]]] = None,
) -> dict:
    """Classify one ``<repo>@<sha>`` ref into a row from its Hugging Face
    revision manifest -- no file is downloaded and nothing is written to
    disk. ``ref`` must already have passed ``validate_pinned_reference``.
    ``fetch_entries`` yields the manifest entries and is called only after
    every option refusal has passed. Never raises: a manifest that cannot be
    fetched or validated, a pack this sizer cannot size, and every
    unsupported option below becomes an ``error`` row naming the reason.

    The card is resolved by the same lookup as a local row, keyed by the ref's
    repo and its pinned revision (``hfPin``). The fit comes from the built-in
    safetensors sizer (``fastmlx_safetensors_fit.compute_fit``) fed the
    manifest's file sizes (``weights_from_manifest``) instead of a directory
    walk, in-process -- never a subprocess. Refused as ``error`` rows (the
    manifest cannot answer them): ``--residency expert-stream`` (the
    safetensors sizer has no expert-stream sizing), an explicit
    ``--fit-check-bin`` or an engine profile's ``fitCheck.bin`` (an
    arbitrary binary needs a local directory), and a missing
    ``--kv-reserve-gib`` (same refusal as a local auto-selected sizer).
    A GGUF pack is not sized here: it has no ``*.safetensors`` in its
    manifest, or a >= 1 GiB unnamed non-safetensors file, so it errors.
    """
    resolved_host_hardware_class = (
        host_hardware_class if host_hardware_class is not None else launch.host_hardware_class
    )
    repo, revision = validate_pinned_reference(ref)
    row: dict = {
        "path": None,
        "name": ref,
        "repo": repo,
        "revision": revision,
        "residency": residency,
        "status": None,
        "fit": None,
        "card": None,
        "message": None,
        "accept_quality_flag": None,
        "engineBuild": None,
        "mtp": None,
        "tiebreakNotice": None,
        "verdictNotice": None,
        "source": SOURCE_REVISION_MANIFEST,
        "downloaded": False,
        "ref": ref,
        "sizing": None,
        "next_step": None,
    }

    card, resolved_ok = _resolve_card_into_row(
        row,
        cards,
        repo,
        revision,
        residency,
        engine_build_commit,
        mtp_launch,
        resolved_host_hardware_class,
    )
    if not resolved_ok:
        return row

    def refuse(message: str) -> dict:
        row["status"] = STATUS_ERROR
        row["card"] = _card_summary(card)
        row["message"] = message
        return row

    if residency != "resident":
        return refuse(
            f"fit check could not run: --residency {residency} cannot be "
            "judged from a revision manifest (the safetensors sizer has no "
            "expert-stream sizing); pull the pack and run recommend on the "
            "local directory"
        )
    if fit_check_bin:
        return refuse(
            "fit check could not run: --pinned sizes the revision manifest "
            "with the built-in safetensors sizer and cannot run an explicit "
            "--fit-check-bin or an engine profile's fitCheck.bin (those need "
            "a local pack directory); drop it, or pull the pack and run "
            "recommend on the local directory"
        )
    if kv_reserve_gib is None:
        return refuse(_kv_reserve_required_message("builtin:safetensors"))

    sizer_argv = [
        "--model-path",
        "<revision-manifest>",
        "--host-use",
        host_use,
        "--fit-check-only",
    ]
    if context is not None:
        sizer_argv += ["--context", str(context)]
    sizer_argv += list(fit_check_args) + ["--kv-reserve-gib", str(kv_reserve_gib)]
    parser = _safetensors_fit.build_arg_parser()
    parser_stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(parser_stderr):
            sizer_args = parser.parse_args(sizer_argv)
    except SystemExit:
        tail = parser_stderr.getvalue().strip().splitlines()
        return refuse(
            "fit check could not run: the sizer refused the --fit-check-arg "
            "values" + (f": {tail[-1]}" if tail else "")
        )
    if sizer_args.residency not in (None, "resident"):
        return refuse(
            f"fit check could not run: --fit-check-arg names --residency "
            f"{sizer_args.residency}, which cannot be judged from a revision manifest"
        )

    try:
        entries = fetch_entries()
    except Exception as error:  # noqa: BLE001 -- any fetch failure is an error row
        return refuse(
            f"could not fetch the revision manifest for {ref}: {error}"
        )

    try:
        weights = _safetensors_fit.weights_from_manifest(
            entries, getattr(sizer_args, "mmap_side_file", None)
        )
        result = _safetensors_fit.compute_fit(sizer_args, weights=weights)
    except _safetensors_fit.MissingKvReserveError:  # pragma: no cover -- guarded above
        return refuse(_kv_reserve_required_message("builtin:safetensors"))
    except (
        _safetensors_fit.FitCheckError,
        _safetensors_fit.SafetensorsFormatError,
        _safetensors_fit._GGUF.FitCheckError,
    ) as error:
        return refuse(f"fit check could not run: {error}")

    row["sizing"] = {
        "weights_bytes": result["weights_bytes"],
        "kv_reserve_bytes": result["kv_reserve_bytes"],
        "total_bytes": result["total_bytes"],
        "ceiling_bytes": result["ceiling_bytes"],
    }
    if result["fit"] != "green":
        row["status"] = STATUS_DOES_NOT_FIT
        row["fit"] = {"verdict": "RED", "context": context}
        row["card"] = _card_summary(card)
        row["message"] = result["reason_text"]
        return row

    row["fit"] = {"verdict": "GREEN", "context": context}
    _classify_fitting_row(row, card)
    next_step = f"fastmlx pull {ref} --dest <dir>"
    if row["accept_quality_flag"]:
        next_step += f"  (then, to serve it: {row['accept_quality_flag']})"
    row["next_step"] = next_step
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
    pinned = row.get("source") == SOURCE_REVISION_MANIFEST
    if pinned:
        # The name IS the full <repo>@<sha> ref; repeating it as repo=...
        # would only add noise.
        head.append("source=revision-manifest (not downloaded)")
    elif row["repo"]:
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
        if "measuredNewTokens" in card:
            head.append(f"measured_tokens={card['measuredNewTokens']}")
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
    tiebreak_notice = row.get("tiebreakNotice")
    if tiebreak_notice and tiebreak_notice not in (row.get("message") or ""):
        tail += f" {tiebreak_notice}"
    verdict_notice = row.get("verdictNotice")
    if verdict_notice and verdict_notice not in (row.get("message") or ""):
        tail += f" {verdict_notice}"
    lines.append(tail)
    if row.get("next_step"):
        lines.append(f"    next: {row['next_step']}")
    return "\n".join(lines)


def _format_card_store_header(identity: Optional[dict]) -> str:
    """One text-output header line naming the card store the ranking used
    (the same identity ``--json`` carries as ``cardStore``)."""
    if identity is None:
        return "card store: none"
    return "card store: " + launch.card_store_fields(identity)


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
        "--pinned",
        action="append",
        default=[],
        metavar="REPO@SHA",
        help=(
            "a Hugging Face '<repo>@<40-hex lowercase commit sha>' to judge "
            "BEFORE downloading it (repeatable, combinable with local "
            "candidates): its fit comes from the revision manifest's file "
            "sizes and its card from the repo/pin lookup. The row is marked "
            "source=revision-manifest, downloaded=false and prints the next "
            "step, `fastmlx pull <ref> --dest <dir>`. Nothing is downloaded "
            "or written. Needs --kv-reserve-gib; not available with "
            "--residency expert-stream, --fit-check-bin or an engine "
            "profile's fitCheck.bin (those become error rows). A branch "
            "name, short sha, or uppercase hex is a usage error"
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
        "--quality-cards-sha256",
        default=None,
        help=(
            "pin the quality-card manifest by the sha256 of its raw bytes "
            "(64 hex characters, case-insensitive; equals `shasum -a 256 "
            "<file>`); a mismatch, a manifest that does not resolve (even "
            "the default path), or matching bytes that are not a manifest "
            "exit 3"
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
        "--engine-bin",
        default=None,
        help=(
            "the engine binary named for engine-build derivation ONLY "
            "(this command never execs it, unlike fastmlx serve's own "
            "--engine-bin) -- default: the built-in profile's own binary "
            "name, resolved on PATH, when no --engine-profile is given"
        ),
    )
    recommend.add_argument(
        "--json",
        action="store_true",
        help="print the ranked rows as a JSON object instead of formatted text",
    )
    return parser


def _run_recommend(args) -> int:
    # Validate every --pinned ref FIRST: a malformed ref is a usage error
    # (exit 2) before any manifest fetch, card load, or other work.
    pinned_refs: list = []
    for pinned_text in args.pinned:
        try:
            validate_pinned_reference(pinned_text)
        except PinnedReferenceError as error:
            print(
                f"fastmlx recommend: --pinned {pinned_text!r}: {error}", file=sys.stderr
            )
            return 2
        if pinned_text not in pinned_refs:
            pinned_refs.append(pinned_text)

    candidates = discover_candidates(args.model_path, args.models_dir)
    if not candidates and not pinned_refs:
        print(
            "fastmlx recommend: no model candidates found; pass --model-path, "
            "--models-dir and/or --pinned; to see which models carry a quality "
            "card before downloading anything, run `fastmlx cards list`",
            file=sys.stderr,
        )
        return 2

    try:
        profile, is_built_in_profile = launch.load_engine_profile(args.engine_profile)
    except launch.LaunchRefusal as refusal:
        print(f"fastmlx recommend: {refusal.message}", file=sys.stderr)
        return refusal.exit_code

    # Read only for engine-build status/notice (see build_row); this script
    # never execs anything, so a profile's engineBuild.binarySha256 is never
    # verified here. An operator-declared commit ALWAYS wins over a
    # derived one; derivation only fills the gap when the profile declares
    # none at all (including the built-in profile, whose engineBuild is
    # always None) -- see fastmlx_launch._run_serve's identical wiring,
    # which this reuses rather than forks.
    #
    # This script never execs anything, so a derived commit describes
    # ONLY the binary the operator NAMED via --engine-bin (or the
    # built-in profile's own binary name on PATH) -- never a binary this
    # command will run. That is why --engine-bin must be explicit here,
    # and why the built-in-profile PATH fallback is inherited from
    # fastmlx serve's own guarded helper
    # (`_guarded_engine_bin_abs_for_engine_build_derivation`) rather than
    # reinvented: a copied rule would let the two commands disagree about
    # the same binary. Derivation never refuses and never changes this
    # command's exit code -- no candidate means no derived commit, and
    # the launch simply stays "undeclared", exactly as before.
    engine_build_commit = (profile.get("engineBuild") or {}).get("commit")
    if engine_build_commit is None:
        candidate_engine_bin_abs = launch._guarded_engine_bin_abs_for_engine_build_derivation(
            args, is_built_in_profile
        )
        if candidate_engine_bin_abs is not None:
            derived_engine_build = launch.derive_engine_build_from_release(
                candidate_engine_bin_abs
            )
            if derived_engine_build is not None:
                engine_build_commit = derived_engine_build["commit"]
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

    # Malformed hex and every pin refusal exit 3 (never argparse's 2: exit 2
    # collides with the all-does-not-fit verdict), exactly like `fastmlx
    # serve`; the pin never fails open, even for the conventional default.
    try:
        quality_cards_pin = launch.parse_quality_cards_pin(args.quality_cards_sha256)
        # Without --quality-cards the store is resolved from the pulled-cards
        # directory, exactly like `fastmlx serve`.
        quality_cards_path, card_store_source, card_store_notices = (
            launch.resolve_quality_card_store(
                args.quality_cards, notice_prefix="fastmlx recommend"
            )
        )
        for card_store_notice in card_store_notices:
            print(card_store_notice, file=sys.stderr)
        raw_store_sha256, cards, card_store_identity = launch._inspect_quality_card_store(
            quality_cards_path
        )
        launch.enforce_pulled_store_identity(
            quality_cards_path, card_store_source, raw_store_sha256
        )
        launch.enforce_quality_cards_pin(
            quality_cards_pin, quality_cards_path, raw_store_sha256, cards
        )
    except launch.LaunchRefusal as refusal:
        print(f"fastmlx recommend: {refusal.message}", file=sys.stderr)
        return refusal.exit_code
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
    rows += [
        build_pinned_row(
            ref=ref,
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
        for ref in pinned_refs
    ]
    rows = rank_rows(rows)

    if args.json:
        print(
            json.dumps(
                {"schema": JSON_SCHEMA, "cardStore": card_store_identity, "rows": rows}
            )
        )
    else:
        text = format_text(rows)
        if text:
            print(_format_card_store_header(card_store_identity))
            print(text)

    return exit_code_for(rows)


def main(argv: Optional[list] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    raise SystemExit(_run_recommend(args))


if __name__ == "__main__":
    main()
