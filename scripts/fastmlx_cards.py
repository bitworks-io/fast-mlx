#!/usr/bin/env python3
"""``fastmlx cards pull``: fetch a digest-pinned quality-card store that may
only ADD to the bundled one; ``fastmlx cards list``: say, offline, which
models carry a quality card.

Predeclaration (frozen contract, rules R1-R7):
docs/task-inbox/2026-10-03-PREDECLARATION-fastmlx-cards-pull-fetches-a-pinned-monotonic-card-store.md

    fastmlx cards pull --commit <40hex> --sha256 <64hex> [--output PATH]

Fetches ``site/quality-guides.json`` at the named public commit from
``raw.githubusercontent.com`` (HTTPS only, same host on every redirect, body
capped at 4 MiB, timeout set), hashes the body IN MEMORY, compares it to the
pin, then runs rules R1-R7 against the bundled baseline BEFORE anything
touches the disk. On success it writes the store (same-directory temp file,
fsync, ``os.replace``, mode 0644) and prints exactly one shell-quoted line on
stdout, ``--quality-cards <abs> --quality-cards-sha256 <H>``, to paste into
``fastmlx serve`` / ``fastmlx recommend``.

Why the rules exist: the pin proves the bytes are the ones named, not that
they are not a ROLLBACK. An older commit verifies perfectly against its own
digest, and a dropped NO_GO card makes its pack uncarded, which admits
silently. The rules make a pulled store a monotonic superset of the bundled
one. The pin does NOT defend against a compromised repository or maintainer
account (the digest is learned from the same channel); that needs signing.

``fastmlx serve`` and ``fastmlx recommend`` NEVER fetch; this module is the
only network code on the card-store path, and nothing on the serve/recommend
path imports it. They only READ the stores it writes: without
``--quality-cards`` they resolve the newest eligible store from
``~/.fastmlx/cards`` (``fastmlx_launch.resolve_quality_card_store``) and
re-run rules R1-R7, which live in ``fastmlx_launch.check_card_store_rules``
and are re-exported here.

``fastmlx cards list [--model REPO] [--revision REV] [--quality-cards P]
[--quality-cards-sha256 H] [--json]`` never opens the network. It resolves
the store exactly as ``recommend`` does (pulled store, else bundled; pin
enforced the same way), prints ``card store: source=<s> <identity>`` and one
row per card (id, verdict, default-eligible vs opt-in, repo, revision prefix,
residency, hardware class vs this host). Predeclaration:
docs/task-inbox/2026-10-06-PREDECLARATION-cards-list-answers-which-models-are-carded-before-download.md
``--model`` matches ``model.repo`` exactly; ``--revision`` (a full 40-hex
commit) matches ``model.hfPin`` by prefix, the way ``serve`` matches a pack by
its pinned revision (a card with no ``model.repo`` is found only that way); both
together select the union. ``list`` exit codes: 0 listed; 1 nothing matched
(``serve`` would admit it UNMEASURED, except that a ``--model`` alone cannot
rule out a pin-only card, and the message then says so); 2 usage, including a
malformed ``--revision``; 3 store or pin refusal, or ANY card in the store the
built-in engine cannot decode, shown or not (reported after the listing; only
the built-in engine, i.e. ``fastmlx serve`` without a custom engine profile,
refuses on it).

``pull`` exit codes: 0 OK; 1 network/IO failure (including a non-HTTPS or cross-host
redirect and an oversize body); 2 argparse error or malformed ``--commit``;
3 malformed ``--sha256``, digest mismatch, any R1-R7 refusal, a card the
built-in engine cannot decode (``engine_undecodable_card_reason``), or a
conflicting ``--output``.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import importlib.util
import json
import os
import re
import shlex
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import NoReturn, Optional, Sequence
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    HTTPSHandler,
    ProxyHandler,
    Request,
    build_opener,
)


def _load_sibling_module(name: str, filename: str):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The same sibling-file load ``fastmlx_recommend.py`` uses for the launcher:
# the card-store vocabulary (schema/verdicts/identity line) is reused, never
# re-declared here.
launch = _load_sibling_module("fastmlx_launch", "fastmlx_launch.py")

SCHEMA = launch.CARD_STORE_SCHEMA
FETCH_HOST = "raw.githubusercontent.com"
URL_TEMPLATE = (
    "https://" + FETCH_HOST + "/bitworks-io/fast-mlx/{commit}/site/quality-guides.json"
)
MAX_BODY_BYTES = 4 << 20
FETCH_TIMEOUT_SECONDS = 30
GENERATED_AT_FORMAT = launch.CARD_STORE_GENERATED_AT_FORMAT
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
DEFAULT_OUTPUT_DIR = ("~", ".fastmlx", "cards")

# The rule constants are declared once, in the launcher (see
# ``fastmlx_launch.check_card_store_rules``); aliased for this module's tests.
TIGHTENING_VERDICT = launch.CARD_STORE_TIGHTENING_VERDICT
MAX_FUTURE_SKEW_SECONDS = launch.CARD_STORE_MAX_FUTURE_SKEW_SECONDS


class CardsFetchError(Exception):
    """A network/IO failure (exit 1), including a refused redirect."""


class CardsRefusal(Exception):
    """A refusal that writes nothing (exit ``exit_code``)."""

    def __init__(self, exit_code: int, message: str):
        super().__init__(message)
        self.exit_code = exit_code
        self.message = message


def _is_confined_url(url: str) -> bool:
    parts = urlsplit(url)
    return parts.scheme.lower() == "https" and parts.netloc.lower() == FETCH_HOST


class CardsRedirectHandler(HTTPRedirectHandler):
    """HTTPS-only redirects (the ``hf_pinned_snapshot_download`` pattern),
    additionally confined to ``raw.githubusercontent.com`` itself (no other
    host, no userinfo, no explicit port)."""

    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        if urlsplit(new_url).scheme.lower() != "https":
            raise CardsFetchError("network request attempted a non-HTTPS redirect")
        if not _is_confined_url(new_url):
            raise CardsFetchError(
                f"network request was redirected away from {FETCH_HOST}"
            )
        return super().redirect_request(
            request, file_pointer, code, message, headers, new_url
        )


def https_opener():
    """The opener ``fetch`` uses. Module-level so tests patch it with a fake
    and never touch the network."""
    return build_opener(ProxyHandler({}), HTTPSHandler(), CardsRedirectHandler())


def fetch(commit: str) -> bytes:
    url = URL_TEMPLATE.format(commit=commit)
    request = Request(url, headers={"User-Agent": "fastmlx-cards"})
    try:
        with https_opener().open(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
            final_url = response.geturl()
            if not _is_confined_url(final_url):
                raise CardsFetchError(
                    f"response came from {launch._bounded_repr(final_url)}, "
                    f"not https://{FETCH_HOST}"
                )
            body = response.read(MAX_BODY_BYTES + 1)
    except CardsFetchError:
        raise
    except (OSError, ValueError, http.client.HTTPException) as error:
        # URLError/HTTPError/timeouts are OSErrors; IncompleteRead,
        # BadStatusLine and friends are HTTPExceptions, not OSErrors.
        raise CardsFetchError(f"fetching {url} failed: {error}") from error
    if len(body) > MAX_BODY_BYTES:
        raise CardsFetchError(
            f"the response body exceeds the {MAX_BODY_BYTES}-byte cap"
        )
    return body


# ---------------------------------------------------------------------
# Rules R1-R7
#
# The rule logic lives in ``fastmlx_launch.check_card_store_rules`` (pure and
# network-free) so ``serve``/``recommend`` can re-run it on a pulled store at
# use time without importing this fetcher. This module keeps the helper names
# its tests and callers know (aliases) and a thin ``check_rules`` wrapper that
# passes its own baseline path and clock, so patches of ``utc_now`` and
# ``bundled_baseline_path`` keep working.
# ---------------------------------------------------------------------
_parse_store = launch._parse_card_store
_parse_generated_at = launch._parse_card_store_generated_at
_card_ids = launch._card_store_card_ids
_canonical = launch._card_store_canonical


def utc_now() -> datetime:
    """The current UTC time as a naive datetime (comparable with the
    ``strptime`` result for ``generatedAt``). Module-level so tests patch it."""
    return launch.card_store_utc_now()


def bundled_baseline_path() -> Path:
    return launch.REPO_ROOT / launch.DEFAULT_QUALITY_CARDS_RELATIVE_PATH


def check_rules(raw: bytes) -> tuple:
    """Run R1-R7 for ``raw`` against the bundled baseline.

    Returns ``(document, added)`` or raises ``CardsRefusal(3, ...)`` naming
    the rule. Reads only the bundled baseline; writes nothing.
    """
    try:
        return launch.check_card_store_rules(
            raw, baseline_path=bundled_baseline_path(), now=utc_now()
        )
    except launch.CardStoreRuleRefusal as refusal:
        raise CardsRefusal(refusal.exit_code, refusal.message)


def check_engine_decodable(document: dict) -> None:
    """Refuse (exit 3) a store whose cards the BUILT-IN engine cannot decode.

    R5 checks only ``id``, ``model.repo`` and the verdict, so a store missing,
    say, ``legible.headline`` passes R1-R7; plain ``fastmlx serve`` would then
    pick that digest-pinned store by default and refuse every built-in launch
    -- and a re-pull fetches the same bytes. Refuse it here, before anything
    is written. Pull time only: ``launch.check_card_store_rules`` is unchanged
    (it also runs at use time for custom engines and ``recommend``, where the
    Swift decoder is irrelevant).
    """
    found = launch.first_engine_undecodable_card(document["cards"])
    if found is None:
        return
    index, named, reason = found
    raise CardsRefusal(
        3,
        f"{named}(element {index}) cannot be decoded by the built-in engine: "
        f"{reason}; `fastmlx serve` would refuse every launch with this store, "
        "so it was not written",
    )


# ---------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------
def _write_atomically(path: str, raw: bytes) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        dir=directory, prefix=".fastmlx-cards-", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, 0o644)
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def _resolve_output(output: Optional[str], digest: str) -> str:
    if output is None:
        return os.path.abspath(
            os.path.join(os.path.expanduser(os.path.join(*DEFAULT_OUTPUT_DIR)), f"{digest}.json")
        )
    return os.path.abspath(os.path.expanduser(output))


def _store_or_conflict(path: str, raw: bytes) -> bool:
    """``True`` if ``path`` already holds exactly ``raw`` (idempotent
    re-pull: nothing is rewritten); ``False`` if it does not exist; a
    different file there is a conflict (exit 3)."""
    if not os.path.lexists(path):
        return False
    try:
        existing = Path(path).read_bytes()
    except OSError as error:
        raise CardsFetchError(f"cannot read the existing output {path}: {error}") from error
    if existing == raw:
        return True
    raise CardsRefusal(
        3,
        f"--output {path} already exists with different bytes; refusing to "
        "overwrite it",
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fastmlx cards",
        description=(
            "Manage the quality-card store: list which models carry a card "
            "(offline), or fetch a digest-pinned store that may only add to "
            "the bundled one."
        ),
    )
    subparsers = parser.add_subparsers(dest="cards_command", required=True)
    list_parser = subparsers.add_parser(
        "list",
        help="list which models carry a quality card (offline, nothing downloaded)",
        description=(
            "List the quality cards in the store `fastmlx recommend` and "
            "`fastmlx serve` read (a pulled store, else the bundled one): "
            "verdict, whether the card admits by default or needs "
            "--accept-quality, repo, revision, residency and hardware class. "
            "Never opens the network."
        ),
    )
    list_parser.add_argument(
        "--model",
        default=None,
        metavar="REPO",
        help="show only the card(s) whose model.repo is exactly REPO",
    )
    list_parser.add_argument(
        "--revision",
        default=None,
        metavar="REV",
        help=(
            "also show the card(s) whose model.hfPin is a prefix of this full "
            "40-hex pinned revision (how serve finds a card with no model.repo)"
        ),
    )
    list_parser.add_argument(
        "--quality-cards",
        default=None,
        help="path to a card store to list (default: resolved like `fastmlx recommend`)",
    )
    list_parser.add_argument(
        "--quality-cards-sha256",
        default=None,
        help=(
            "pin the store by the sha256 of its raw bytes (64 hex characters); "
            "a mismatch or a store that does not resolve exits 3"
        ),
    )
    list_parser.add_argument(
        "--json", action="store_true", help="print one JSON document instead of text rows"
    )
    pull = subparsers.add_parser(
        "pull",
        help="fetch a pinned card store at a public commit, verify it, and write it",
        description=(
            "Fetch site/quality-guides.json at a 40-hex public commit, verify "
            "its sha256 pin and rules R1-R7 against the bundled store, then "
            "write it and print the --quality-cards flags to use."
        ),
    )
    pull.add_argument(
        "--commit",
        required=True,
        help="the 40-character lowercase hex public commit to fetch the store at",
    )
    pull.add_argument(
        "--sha256",
        required=True,
        help="the expected sha256 (64 hex) of the store file's raw bytes",
    )
    pull.add_argument(
        "--output",
        default=None,
        help=(
            "where to write the store (default: ~/.fastmlx/cards/<sha256>.json); "
            "an existing file with different bytes is refused"
        ),
    )
    return parser


def _refuse(exit_code: int, message: str) -> NoReturn:
    print(f"fastmlx cards pull refused: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def _run_pull(args: argparse.Namespace) -> int:
    if _COMMIT_RE.fullmatch(args.commit) is None:
        _refuse(2, f"--commit must be exactly 40 lowercase hex characters; got {launch._bounded_repr(args.commit)}")
    try:
        pin = launch.parse_quality_cards_pin(args.sha256)
    except launch.LaunchRefusal:
        _refuse(
            3,
            "--sha256 must be exactly 64 hexadecimal characters (the sha256 of "
            f"the store file's raw bytes); got {launch._bounded_repr(args.sha256)}",
        )
    try:
        raw = fetch(args.commit)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != pin:
            raise CardsRefusal(
                3,
                f"the fetched store does not match --sha256: expected {pin}, "
                f"actual {digest}",
            )
        document, added = check_rules(raw)
        check_engine_decodable(document)
        output = _resolve_output(args.output, digest)
        already_there = _store_or_conflict(output, raw)
        if not already_there:
            _write_atomically(output, raw)
    except CardsRefusal as refusal:
        _refuse(refusal.exit_code, refusal.message)
    except (CardsFetchError, OSError) as error:
        print(f"fastmlx cards pull failed: {error}", file=sys.stderr)
        return 1

    identity = {
        "sha256": digest,
        "generatedAt": document["generatedAt"],
        "cards": len(document["cards"]),
    }
    print(
        f"--quality-cards {shlex.quote(output)} --quality-cards-sha256 {digest}"
    )
    print(
        f"fastmlx_cards=pulled commit={args.commit} "
        f"{launch.card_store_fields(identity)} added={added}",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------------
# list
# ---------------------------------------------------------------------
_CELL_MAX_LEN = 80
_SAFE_CELL_RE = re.compile(r"[A-Za-z0-9._@+/\-]{1,80}")
_LIST_PREFIX = "fastmlx cards list"


def _cell(value) -> str:
    """A card-sourced string made safe for one ``" | "``-delimited line: an
    ordinary id/repo/revision prints bare; anything else (spaces, control
    characters, a non-string, over-long) goes through the launcher's
    ``_bounded_repr`` so a hostile card can never forge or extend a row."""
    if isinstance(value, str) and _SAFE_CELL_RE.fullmatch(value):
        return value
    return launch._bounded_repr(value, _CELL_MAX_LEN)


def _host_tag(card_class: Optional[str], host_class: Optional[str]) -> str:
    if card_class is None:
        return "unrecorded"
    if host_class is None:
        return "host unknown"
    return "this host" if card_class == host_class else "other hardware"


def _list_row(index: int, card, host_class: Optional[str]) -> tuple:
    """``(text, record)`` for one card. ``record`` is the ``--json`` shape."""
    undecodable = launch.engine_undecodable_card_reason(card)
    if not isinstance(card, dict):
        record = {
            "id": None, "verdict": None, "admission": None, "repo": None,
            "hfPin": None, "residency": None, "hardwareClass": None,
            "host": "unrecorded", "undecodable": undecodable,
        }
        return f"<element {index}> | UNDECODABLE {undecodable}", record

    card_id = card.get("id")
    verdict = card.get("verdict")
    model = card.get("model")
    model = model if isinstance(model, dict) else None
    config = card.get("config") if isinstance(card.get("config"), dict) else {}

    outcome, _ = launch.decide_admission(card, False)
    opt_in = outcome == "refuse_quality_flagged"
    admission_text = (
        f"opt-in: --accept-quality {_cell(card_id)}" if opt_in else "default-eligible"
    )
    if verdict in launch._RECOGNIZED_VERDICTS:
        verdict_text = verdict
    else:
        raw = "(absent)" if "verdict" not in card else launch._bounded_repr(verdict)
        verdict_text = f"UNRECOGNIZED {raw} (admits as UNMEASURED)"

    repo = model.get("repo") if model is not None else None
    if model is None:
        repo_text = "repo unreadable"
    elif repo is None:
        repo_text = "repo=none (local pack)"
    else:
        repo_text = _cell(repo)
    hf_pin = model.get("hfPin") if model is not None else None
    revision_text = (
        f"revision begins {_cell(hf_pin)}"
        if isinstance(hf_pin, str) and hf_pin
        else "revision unrecorded"
    )

    raw_residency = config.get("residency")
    if raw_residency is None:
        residency_text = "residency unrecorded"
    elif launch.card_residency(card) is not None:
        residency_text = f"residency {launch.card_residency(card)}"
    else:
        residency_text = f"residency unrecognized {launch._bounded_repr(raw_residency)}"

    hardware_class = launch.card_hardware_class(card)
    tag = _host_tag(hardware_class, host_class)
    hardware_text = (
        "hardware unrecorded"
        if hardware_class is None
        else f"hardware {_cell(hardware_class)} ({tag})"
    )

    cells = [
        _cell(card_id), verdict_text, admission_text, repo_text,
        revision_text, residency_text, hardware_text,
    ]
    if undecodable is not None:
        cells.append(f"UNDECODABLE {undecodable}")
    record = {
        "id": card_id if isinstance(card_id, str) else None,
        "verdict": verdict if isinstance(verdict, str) else None,
        "admission": {"default": not opt_in, "optIn": opt_in},
        "repo": repo if isinstance(repo, str) else None,
        "hfPin": hf_pin if isinstance(hf_pin, str) else None,
        "residency": raw_residency if isinstance(raw_residency, str) else None,
        "hardwareClass": hardware_class,
        "host": tag,
        "undecodable": undecodable,
    }
    return " | ".join(cells), record


def _list_refuse(exit_code: int, message: str) -> NoReturn:
    print(f"{_LIST_PREFIX}: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def _run_list(args: argparse.Namespace) -> int:
    # Resolve and verify the store exactly as `fastmlx recommend` does
    # (fastmlx_recommend.py), so both read the identical store with identical
    # refusal messages. Nothing here opens the network.
    try:
        pin = launch.parse_quality_cards_pin(args.quality_cards_sha256)
        path, source, notices = launch.resolve_quality_card_store(
            args.quality_cards, notice_prefix=_LIST_PREFIX
        )
        for notice in notices:
            print(notice, file=sys.stderr)
        raw_sha256, cards, identity = launch._inspect_quality_card_store(path)
        launch.enforce_pulled_store_identity(path, source, raw_sha256)
        launch.enforce_quality_cards_pin(pin, path, raw_sha256, cards)
    except launch.LaunchRefusal as refusal:
        _list_refuse(refusal.exit_code, refusal.message)
    if cards is None or identity is None:
        _list_refuse(
            3, f"the card store {path} is missing or is not a quality-card manifest"
        )

    if args.revision is not None and not launch._is_full_hex_revision(args.revision):
        _list_refuse(
            2,
            f"--revision must be exactly 40 hexadecimal characters, got "
            f"{launch._bounded_repr(args.revision, _CELL_MAX_LEN)}",
        )

    selected = list(enumerate(cards))
    if args.model is not None or args.revision is not None:
        matched = set()
        if args.model is not None:
            matched |= {id(c) for c in launch.find_cards_by_repo(cards, args.model)}
        if args.revision is not None:
            matched |= {id(c) for c in launch.find_cards_by_pin(cards, args.revision)}
        selected = [(i, card) for i, card in selected if id(card) in matched]

    host_class = launch.host_hardware_class()
    rendered = [_list_row(i, card, host_class) for i, card in selected]
    # The built-in engine refuses if ANY card in the store is undecodable, so
    # this looks at the whole store, not only the rows a filter let through.
    undecodable = launch.first_engine_undecodable_card(cards)

    no_match = (args.model is not None or args.revision is not None) and not rendered
    if args.json:
        print(
            json.dumps(
                {
                    "cardStore": {**identity, "source": source},
                    "cards": [record for _, record in rendered],
                }
            )
        )
    else:
        print(f"card store: source={source} {launch.card_store_fields(identity)}")
        for text, _ in rendered:
            print(text)
    if undecodable is not None:
        index, named, reason = undecodable
        print(
            f"{_LIST_PREFIX}: {named}(element {index}) cannot be decoded by the "
            f"built-in engine: {reason}; the built-in engine (`fastmlx serve` "
            "without a custom engine profile) would refuse every launch with "
            "this store",
            file=sys.stderr,
        )
        return 3
    if no_match:
        target = (
            f"{_cell(args.model)} or revision {args.revision}"
            if args.model is not None and args.revision is not None
            else _cell(args.model) if args.revision is None else f"revision {args.revision}"
        )
        pin_only = 0
        repo_keyed = 0
        for card in cards:
            model = card.get("model") if isinstance(card, dict) else None
            if not isinstance(model, dict):
                continue
            if isinstance(model.get("repo"), str):
                repo_keyed += 1
            elif isinstance(model.get("hfPin"), str) and model.get("hfPin"):
                pin_only += 1
        if args.revision is not None:
            pin_only = 0
        if args.model is not None:
            repo_keyed = 0
        if pin_only:
            print(
                f"{_LIST_PREFIX}: no quality card names {target} by repo; "
                f"{pin_only} card(s) in this store are matched only by the "
                "pack's pinned revision, so pass `--revision <40-hex>` to check",
                file=sys.stderr,
            )
        elif repo_keyed:
            print(
                f"{_LIST_PREFIX}: no quality card is pinned to {target}; "
                f"{repo_keyed} card(s) in this store are matched by repo, so "
                "pass `--model <repo>` as well to check",
                file=sys.stderr,
            )
        else:
            print(
                f"{_LIST_PREFIX}: no quality card names {target}; serve would "
                "admit it UNMEASURED",
                file=sys.stderr,
            )
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.cards_command == "list":
        raise SystemExit(_run_list(args))
    raise SystemExit(_run_pull(args))


if __name__ == "__main__":
    main()
