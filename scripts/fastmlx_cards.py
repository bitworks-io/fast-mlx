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
only network code on the card-store path, and nothing on the serve/recommend
path imports it. They only READ the stores it writes: without
``--quality-cards`` they resolve the newest eligible store from
``~/.fastmlx/cards`` (``fastmlx_launch.resolve_quality_card_store``) and
re-run rules R1-R7, which live in ``fastmlx_launch.check_card_store_rules``
and are re-exported here.

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
