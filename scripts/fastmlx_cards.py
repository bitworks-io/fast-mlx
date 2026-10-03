#!/usr/bin/env python3
"""``fastmlx cards pull``: fetch a digest-pinned quality-card store that may
only ADD to the bundled one.

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
only network code on the card-store path, and nothing imports it.

Exit codes: 0 OK; 1 network/IO failure (including a non-HTTPS or cross-host
redirect and an oversize body); 2 argparse error or malformed ``--commit``;
3 malformed ``--sha256``, digest mismatch, any R1-R7 refusal, or a
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
from datetime import datetime, timedelta, timezone
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

SCHEMA = "fast-mlx-quality-card-v1"
FETCH_HOST = "raw.githubusercontent.com"
URL_TEMPLATE = (
    "https://" + FETCH_HOST + "/bitworks-io/fast-mlx/{commit}/site/quality-guides.json"
)
MAX_BODY_BYTES = 4 << 20
FETCH_TIMEOUT_SECONDS = 30
GENERATED_AT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_GENERATED_AT_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
DEFAULT_OUTPUT_DIR = ("~", ".fastmlx", "cards")

# R6 (strict): every bundled card must reappear, matched by id, DEEP-EQUAL to
# the bundled card (json-equal), with exactly one allowed difference: its
# ``verdict`` may change from an admitting verdict to this one (tightening).
# Anything else on a kept id is refused: ``decide_admission`` only refuses a
# ``NO_GO`` card, but ``resolve_card`` finds the card by ``model.repo`` /
# ``model.hfPin`` and filters it by ``config.residency``, and refuses
# (exit 3) a mixed-verdict tie, so editing any of those, or swapping a verdict
# within the admitting tier, can move a launch's outcome. The launcher's own
# vocabulary is pinned by a test (this must stay one of its recognized verdicts).
TIGHTENING_VERDICT = "NO_GO"
# R2: a ``generatedAt`` further than this past the current UTC time is refused
# (a far-future stamp would make every later honest store an R2 "rollback").
MAX_FUTURE_SKEW_SECONDS = 24 * 60 * 60


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
# ---------------------------------------------------------------------
def _parse_store(raw: bytes) -> Optional[dict]:
    """The decoded document iff ``raw`` is a JSON object whose ``cards`` is a
    list (the same shape test ``fastmlx_launch._inspect_quality_card_store``
    applies; bytes-level here because nothing may touch the disk before the
    checks pass)."""
    try:
        document = json.loads(raw.decode("utf-8"))
    except ValueError:  # JSONDecodeError and UnicodeDecodeError both subclass it
        return None
    except RecursionError:  # pathologically nested JSON: unparsable, not a crash
        return None
    if not isinstance(document, dict) or not isinstance(document.get("cards"), list):
        return None
    return document


def _parse_generated_at(value) -> Optional[datetime]:
    if not isinstance(value, str) or _GENERATED_AT_RE.fullmatch(value) is None:
        return None
    try:
        return datetime.strptime(value, GENERATED_AT_FORMAT)
    except ValueError:
        return None


def _card_ids(cards: list) -> list:
    return [
        card["id"]
        for card in cards
        if isinstance(card, dict) and isinstance(card.get("id"), str)
    ]


def utc_now() -> datetime:
    """The current UTC time as a naive datetime (comparable with the
    ``strptime`` result for ``generatedAt``). Module-level so tests patch it."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _canonical(value) -> str:
    """A json-equal comparison key: unlike ``==``, it keeps ``1``, ``1.0`` and
    ``true`` apart (``1 == 1.0 == True`` in Python)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def bundled_baseline_path() -> Path:
    return launch.REPO_ROOT / launch.DEFAULT_QUALITY_CARDS_RELATIVE_PATH


def _load_baseline() -> tuple:
    """``(raw_sha256, document, generated_at)`` for the bundled store, else R7."""
    path = bundled_baseline_path()
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise CardsRefusal(
            3, f"R7 no resolvable bundled baseline: cannot read {path}: {error}"
        )
    document = _parse_store(raw)
    if document is None or document.get("schema") != SCHEMA:
        raise CardsRefusal(
            3, f"R7 no resolvable bundled baseline: {path} is not a {SCHEMA} store"
        )
    generated_at = _parse_generated_at(document.get("generatedAt"))
    if generated_at is None:
        raise CardsRefusal(
            3, f"R7 no resolvable bundled baseline: {path} has no valid generatedAt"
        )
    return hashlib.sha256(raw).hexdigest(), document, generated_at


def check_rules(raw: bytes) -> tuple:
    """Run R1-R7 for ``raw`` against the bundled baseline.

    Returns ``(document, added)`` or raises ``CardsRefusal(3, ...)`` naming
    the rule. Reads only the bundled baseline; writes nothing.
    """
    baseline_sha, baseline, baseline_generated_at = _load_baseline()  # R7

    document = _parse_store(raw)
    if document is None:
        raise CardsRefusal(3, "R1 the body does not parse as a card store")
    if document.get("schema") != SCHEMA:
        raise CardsRefusal(
            3,
            f"R1 schema is {launch._bounded_repr(document.get('schema'))}, "
            f"expected {SCHEMA!r}",
        )

    generated_at = _parse_generated_at(document.get("generatedAt"))  # R2
    if generated_at is None:
        raise CardsRefusal(
            3,
            "R2 generatedAt "
            f"{launch._bounded_repr(document.get('generatedAt'))} is not strictly "
            "%Y-%m-%dT%H:%M:%SZ",
        )
    if generated_at > utc_now() + timedelta(seconds=MAX_FUTURE_SKEW_SECONDS):
        raise CardsRefusal(
            3, f"R2 generatedAt {document['generatedAt']} is in the future"
        )
    if generated_at < baseline_generated_at:
        raise CardsRefusal(
            3,
            f"R2 generatedAt {document['generatedAt']} is older than the bundled "
            f"store's {baseline['generatedAt']} (rollback)",
        )
    if generated_at == baseline_generated_at and hashlib.sha256(raw).hexdigest() != baseline_sha:
        raise CardsRefusal(
            3,
            f"R2 generatedAt {document['generatedAt']} equals the bundled store's "
            "but the bytes differ",
        )

    cards = document["cards"]
    ids = _card_ids(cards)
    baseline_ids = _card_ids(baseline["cards"])
    dropped = sorted(set(baseline_ids) - set(ids))  # R3
    if dropped:
        raise CardsRefusal(
            3, "R3 drops bundled card id(s): " + ", ".join(dropped)
        )
    duplicates = sorted({i for i in ids if ids.count(i) > 1})  # R4
    if duplicates:
        raise CardsRefusal(3, "R4 duplicate card id(s): " + ", ".join(duplicates))

    for card in cards:  # R5
        if not isinstance(card, dict):
            raise CardsRefusal(3, "R5 a card is not an object")
        label = launch._bounded_repr(card.get("id"))
        if not isinstance(card.get("id"), str):
            raise CardsRefusal(3, f"R5 card {label} has a missing or non-string id")
        model = card.get("model")
        # The schema makes ``model.repo`` required but NULLABLE (hfPin-only
        # and enhancement cards carry ``repo: null``, as five bundled cards
        # do), so "no repo" means the key is absent or not a string/null.
        if (
            not isinstance(model, dict)
            or "repo" not in model
            or not (model["repo"] is None or isinstance(model["repo"], str))
        ):
            raise CardsRefusal(
                3,
                f"R5 card {label} has a missing or non-object 'model', or no "
                "'repo' key: the repo lookup would skip it",
            )
        if card.get("verdict") not in launch._RECOGNIZED_VERDICTS:
            raise CardsRefusal(
                3,
                f"R5 card {label} has an unrecognized verdict "
                f"{launch._bounded_repr(card.get('verdict'))}",
            )

    pulled_by_id = {card["id"]: card for card in cards}  # R6 (ids unique: R4)
    for bundled in baseline["cards"]:
        if not isinstance(bundled, dict) or not isinstance(bundled.get("id"), str):
            continue
        pulled = pulled_by_id[bundled["id"]]  # present: R3
        differing = sorted(
            key
            for key in set(bundled) | set(pulled)
            if key not in bundled
            or key not in pulled
            or _canonical(bundled[key]) != _canonical(pulled[key])
        )
        if not differing:
            continue
        if differing == ["verdict"]:
            if pulled["verdict"] == TIGHTENING_VERDICT:
                continue  # an admitting verdict tightened to NO_GO
            if bundled["verdict"] == TIGHTENING_VERDICT:
                raise CardsRefusal(
                    3,
                    f"R6 card {bundled['id']} relaxes verdict {bundled['verdict']} "
                    f"-> {pulled['verdict']}",
                )
        raise CardsRefusal(
            3,
            f"R6 card {bundled['id']} changes {','.join(differing)} relative to "
            "the bundled store",
        )

    return document, len(set(ids) - set(baseline_ids))


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
            "Manage the quality-card store: fetch a digest-pinned store that "
            "may only add to the bundled one."
        ),
    )
    subparsers = parser.add_subparsers(dest="cards_command", required=True)
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


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    raise SystemExit(_run_pull(args))


if __name__ == "__main__":
    main()
