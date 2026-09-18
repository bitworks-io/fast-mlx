#!/usr/bin/env python3
"""Metadata-only quant prefetch helper for fast-mlx's serve.sh (fit-checked-serve
differentiator #2, "sourcing half").

`fastmlx-serve --quant-pick-only --quant-candidates <dirs>` sizes each candidate quant of a
model from its `config.json` (+ `*.safetensors.index.json`) to pick a winner, but must NOT
download multi-GB weights just to COMPARE fits. This module fetches ONLY the sizing metadata
for each candidate HF repo (via `huggingface_hub.snapshot_download(..., allow_patterns=...)`),
so the winner can be picked cheaply, then the FULL weights are downloaded for the winner
ALONE. Calling `snapshot_download` again later for the winner (no `allow_patterns`) fills the
SAME cache snapshot dir with the weights (idempotent) -- so a repo's metadata dir and its
full-download dir are the same path.

Usage:
  python3 scripts/quant-prefetch.py metadata --repos org/M-4bit,org/M-8bit
  python3 scripts/quant-prefetch.py full --repo org/M-4bit
  echo "$harness_stdout" | python3 scripts/quant-prefetch.py parse-winner
"""
from __future__ import annotations

import argparse
import sys
from typing import Callable, NamedTuple

# The allow-patterns handed to `snapshot_download` for a metadata-only fetch: the two files
# the Swift decoder reads to size a candidate (HarnessCore/QuantAutoPicker).
METADATA_PATTERNS = ["config.json", "*.safetensors.index.json"]

# Substrings in a download exception's class name that indicate a not-found / gated repo,
# rather than a transient or unexpected failure.
_NOT_FOUND_CLASS_MARKERS = ("RepositoryNotFound", "EntryNotFound")
_GATED_CLASS_MARKERS = ("GatedRepo",)

_REASON_TRUNCATE_LEN = 200


class PrefetchOutcome(NamedTuple):
    """Result of a metadata-only fetch attempt for one repo.

    `local_dir` is set when the metadata fetched successfully; `excluded_reason` is set when
    the repo was skipped. Exactly one of the two is non-None.
    """

    repo: str
    local_dir: str | None
    excluded_reason: str | None


def parse_winner_line(stdout: str) -> str | None:
    """Scan `stdout` for the frozen machine-readable winner line and return its path.

    The line has the form (HarnessCore/QuantAutoPicker.machineReadableWinnerLine()):
      quant_pick winner=<path> quant_bits=<n> fit_check=<green|yellow|red> fit_served_context=<n> [kv_tier=..] [prefer=..]

    Returns None if no such line is present, or if a matching line has no parseable
    `winner=` value.
    """
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("quant_pick winner="):
            continue
        rest = line[len("quant_pick winner="):]
        if not rest:
            return None
        value = rest.split(None, 1)[0]
        return value or None
    return None


def parse_enumerate_line(stdout: str) -> str | None:
    """Scan `stdout` for the offline candidate-enumeration line and return its candidates CSV.

    The line has the form (fastmlx-serve's `--auto-quant BASE --quant-pick-only` output):
      quant_enumerate base=<base> candidates=<repo1>,<repo2>,...

    `base=` precedes `candidates=`, so we anchor on the `candidates=` marker. Returns None if no
    such line is present or the candidates value is empty.
    """
    marker = "candidates="
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("quant_enumerate "):
            continue
        idx = line.find(marker)
        if idx == -1:
            return None
        value = line[idx + len(marker):].strip()
        # candidates is the final field today; take up to the first whitespace so a future
        # trailing field can't get swept into the CSV.
        value = value.split()[0] if value else ""
        return value or None
    return None


def exclusion_reason_for(exc: Exception) -> str:
    """Map a download exception to a short human reason.

    404 / not-found and gated-repo cases get a fixed short label; anything else falls back to
    the exception's message, truncated to a reasonable length.
    """
    cls_name = type(exc).__name__
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    message = str(exc)

    if any(marker in cls_name for marker in _GATED_CLASS_MARKERS):
        return "gated"
    if (
        any(marker in cls_name for marker in _NOT_FOUND_CLASS_MARKERS)
        or status_code == 404
        or "404" in message
    ):
        return "not found (404)"

    if len(message) > _REASON_TRUNCATE_LEN:
        message = message[:_REASON_TRUNCATE_LEN]
    return message


def _default_downloader(repo: str, allow_patterns: list[str] | None = None) -> str:
    """Lazily import huggingface_hub so importing this module never requires it (tests inject
    a fake downloader instead)."""
    from huggingface_hub import snapshot_download

    if allow_patterns is not None:
        return snapshot_download(repo, allow_patterns=allow_patterns)
    return snapshot_download(repo)


def metadata_snapshot(
    repo: str, downloader: Callable[..., str] = _default_downloader
) -> PrefetchOutcome:
    """Fetch ONLY the sizing metadata for `repo`. Never raises -- exclusions are not fatal;
    they're reported so the picker can proceed with the remaining candidates."""
    try:
        local_dir = downloader(repo, allow_patterns=METADATA_PATTERNS)
    except Exception as exc:  # noqa: BLE001 - deliberately broad; converted to a reason string
        return PrefetchOutcome(repo, None, exclusion_reason_for(exc))
    return PrefetchOutcome(repo, local_dir, None)


def full_download(repo: str, downloader: Callable[..., str] = _default_downloader) -> str:
    """Download the FULL weights for the winning repo. Exceptions propagate -- unlike metadata
    exclusions, a winner download failure is fatal."""
    return downloader(repo)


def prefetch_all(
    repos: list[str], downloader: Callable[..., str] = _default_downloader
) -> list[PrefetchOutcome]:
    """Fetch metadata for each candidate repo, preserving order."""
    return [metadata_snapshot(repo, downloader=downloader) for repo in repos]


def _cmd_metadata(args: argparse.Namespace) -> int:
    repos = [r.strip() for r in args.repos.split(",") if r.strip()]
    outcomes = prefetch_all(repos)
    included = 0
    for outcome in outcomes:
        if outcome.excluded_reason is None:
            print(f"ok\t{outcome.repo}\t{outcome.local_dir}")
            included += 1
        else:
            print(f"skip\t{outcome.repo}\t{outcome.excluded_reason}")
            print(f"quant-prefetch: excluded {outcome.repo}: {outcome.excluded_reason}", file=sys.stderr)
    if included == 0:
        print("quant-prefetch: every candidate was excluded; nothing to pick", file=sys.stderr)
        return 2
    return 0


def _cmd_full(args: argparse.Namespace) -> int:
    try:
        local_dir = full_download(args.repo)
    except Exception as exc:  # noqa: BLE001 - reported to stderr, not swallowed
        print(f"quant-prefetch: full download of {args.repo} failed: {exc}", file=sys.stderr)
        return 1
    print(local_dir)
    return 0


def _cmd_parse_winner(_args: argparse.Namespace) -> int:
    stdout = sys.stdin.read()
    winner = parse_winner_line(stdout)
    if winner is None:
        print("quant-prefetch: no quant_pick winner= line found", file=sys.stderr)
        return 3
    print(winner)
    return 0


def _cmd_parse_enumerate(_args: argparse.Namespace) -> int:
    stdout = sys.stdin.read()
    candidates = parse_enumerate_line(stdout)
    if candidates is None:
        print("quant-prefetch: no quant_enumerate candidates= line found", file=sys.stderr)
        return 4
    print(candidates)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    metadata_parser = subparsers.add_parser(
        "metadata", help="Fetch sizing metadata for candidate repos (no weights)."
    )
    metadata_parser.add_argument("--repos", required=True, help="Comma-separated HF repo ids.")
    metadata_parser.set_defaults(func=_cmd_metadata)

    full_parser = subparsers.add_parser(
        "full", help="Download full weights for the winning repo."
    )
    full_parser.add_argument("--repo", required=True, help="HF repo id of the winner.")
    full_parser.set_defaults(func=_cmd_full)

    parse_winner_parser = subparsers.add_parser(
        "parse-winner", help="Read stdin, print the quant_pick winner= path."
    )
    parse_winner_parser.set_defaults(func=_cmd_parse_winner)

    parse_enumerate_parser = subparsers.add_parser(
        "parse-enumerate", help="Read stdin, print the quant_enumerate candidates= CSV."
    )
    parse_enumerate_parser.set_defaults(func=_cmd_parse_enumerate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
