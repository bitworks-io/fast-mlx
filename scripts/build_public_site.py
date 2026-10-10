#!/usr/bin/env python3
"""Build the dependency-free fast-mlx GitHub Pages site.

Only articles named in site/publications.json are rendered. Markdown is escaped and handled by a
small project-owned renderer so a Pages build does not execute article HTML or depend on a remote
package registry.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import math
import os
import posixpath
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import validate_public_repository

# Single source of truth for private markers -- this used to be a locally
# duplicated tuple that (by omission, not by intent) never included two of
# `validate_public_repository.PRIVATE_MARKERS`' entries, so those two
# markers could leak into a generated card/site artifact undetected even
# though `validate_public_repository.py` (the checkout-level gate) already
# refused them. Importing keeps the two lists from drifting apart again.
PRIVATE_MARKERS: Tuple[str, ...] = validate_public_repository.PRIVATE_MARKERS

# Extended markers for the quality-guide manifest only (not every PRIVATE_MARKERS
# consumer): private serving-engine names, EXCEPT when the occurrence is part of
# the exact public token "fastmlx-serve". The names are assembled from parts so
# this public source never carries them as literals. Host shorthand for the
# fleet's ".25<0-3>" addresses is refused everywhere (never inside a longer
# number, e.g. ".2519").
QUALITY_GUIDE_ENGINE_NAME_MARKERS: Tuple[str, ...] = ("om" + "lx", "mt" + "plx", "ml" + "x-serve")
QUALITY_GUIDE_PUBLIC_ENGINE_TOKEN = "fastmlx-serve"
QUALITY_GUIDE_HOST_SHORTHAND = re.compile(r"(?<![0-9])\.25[0-3](?![0-9])")


def _quality_guide_private_marker(serialized: str) -> Optional[str]:
    """Return the first extended private marker found in `serialized`, or None."""
    sanitized = serialized.casefold().replace(QUALITY_GUIDE_PUBLIC_ENGINE_TOKEN, "")
    for marker in QUALITY_GUIDE_ENGINE_NAME_MARKERS:
        if marker in sanitized:
            return marker
    host_match = QUALITY_GUIDE_HOST_SHORTHAND.search(serialized)
    if host_match:
        return host_match.group(0)
    return None


TABLE_SEPARATOR = re.compile(r"^:?-{3,}:?$")
LIST_ITEM = re.compile(r"^(?:[-+*]\s+|\d+\.\s+)(.+)$")
ORDERED_ITEM = re.compile(r"^\d+\.\s+(.+)$")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
CODE_SPAN = re.compile(r"`([^`]+)`")
SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
QUALITY_CARD_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*@[a-z0-9]+(?:-[a-z0-9]+)*")
WHITEPAPER_THEME = re.compile(
    r"^(?:>\s*)?\*\*Whitepaper themes?:\*\*\s*(.*)$", re.IGNORECASE
)

CAPABILITY_STATUS_DEFINITIONS: Tuple[Tuple[str, str, str], ...] = (
    (
        "implemented",
        "Implemented",
        "The public source and regression contracts exist; this is not automatically a supported default.",
    ),
    (
        "promoted-scoped",
        "Promoted · scoped",
        "A bounded route or result crossed its stated evidence gates only for the named scope.",
    ),
    (
        "experimental",
        "Experimental",
        "The surface is active research and has not earned a production support claim.",
    ),
    (
        "shelved",
        "Shelved",
        "The dated result remains useful evidence, but the capability is not the current production route.",
    ),
)
CAPABILITY_STATUSES = {item[0] for item in CAPABILITY_STATUS_DEFINITIONS}
HIGHLIGHT_DECISIONS = {"promoted-scoped", "shelved"}
QUALITY_GUIDE_SCHEMA = "fast-mlx-quality-card-v1"
QUALITY_VERDICTS = {"NO_GO", "PASS", "REFERENCE", "EXACT", "UNMEASURED"}
QUALITY_PROVENANCE_SOURCES = {"fast-mlx-measured", "vendor-reported", "modeled"}
QUALITY_TIERS = {"Exact", "Near-lossless", "Noticeable", "Significant", "Unquantified"}
QUALITY_CARD_RESIDENCIES = {"resident", "expert-stream"}
QUALITY_CARD_CONFIG_REQUIRED_KEYS = {"quant", "enhancement", "hardwareClass"}
# `flagTransfer` is OPTIONAL (see docs/quality-card-schema-v1.md "Flag
# transfer"); absence means "unmeasured" for any launch that flips the flag.
QUALITY_CARD_CONFIG_ALLOWED_KEYS = (
    QUALITY_CARD_CONFIG_REQUIRED_KEYS | {"residency", "flagTransfer"}
)
QUALITY_CARD_FLAG_TRANSFER_KEYS = {"--mtp"}
QUALITY_CARD_FLAG_TRANSFER_MTP_KEYS = {
    "greedy",
    "divergentPrompts",
    "prompts",
    "maxTokens",
    "evidence",
}
QUALITY_CARD_FLAG_TRANSFER_MTP_GREEDY_VALUES = {"exact", "not_exact", "nondeterministic"}
QUALITY_EXAMPLE_STATUSES = {"measured", "illustrative", "pending"}
QUALITY_CARD_PROVENANCE_REQUIRED_KEYS = {
    "source",
    "vendor",
    "method",
    "confound",
    "hardware",
    "harnessGitSHA",
    "corpusId",
    "sourceVerdict",
    "measuredAt",
}
# `engineBuild` is OPTIONAL: the served-engine build a card's measurement ran
# on (see docs/quality-card-schema-v1.md "Engine build"). Absent means
# "unrecorded" -- never inferred, never defaulted to a fabricated commit.
QUALITY_CARD_PROVENANCE_ALLOWED_KEYS = QUALITY_CARD_PROVENANCE_REQUIRED_KEYS | {"engineBuild"}
QUALITY_CARD_ENGINE_BUILD_COMMIT = re.compile(r"[0-9a-f]{40}")
# config.hardwareClass: the SAME string `fastmlx_launch.host_hardware_class()`
# produces (e.g. "apple-m3-ultra") -- lowercase alphanumeric segments joined
# by single hyphens. A value outside this shape can never host-match by
# construction (see `fastmlx_launch.host_hardware_class()`'s docstring), so
# it is refused here rather than silently never firing at serve time.
#
# This is a deliberate separate copy, not an import of either sibling --
# see the "projection split" note in
# docs/task-inbox/2026-09-22-PREDECLARATION-hardware-class-must-be-canonical.md.
# Must stay byte-consistent with the copy in `validate_public_site.py` and
# with `emit_quality_card.validate_hardware_class` / the argparse-validated
# `--hardware-class` in `emit_quality_card_cell.py`.
QUALITY_CARD_HARDWARE_CLASS = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
# The producer-side fail-closed sentinel: `ProvenanceCLI.chipBrand()`
# (spike/Sources/fastmlx-harness/Provenance+CLI.swift:102,105) and
# `fastmlx_bench._chip_identity` (three call sites) both return the literal
# string "unknown" when chip identification fails, and that value can flow
# unrefused into `emit_quality_card`'s `context["hardware"]["chip"]`. It is
# lowercase alphanumeric, so it PASSES QUALITY_CARD_HARDWARE_CLASS above --
# but the detector, `fastmlx_launch.host_hardware_class()`, fails closed to
# `None` and can NEVER return "unknown". A card carrying "unknown" can
# therefore never host-match: a silent no-op, exactly what this gate exists
# to refuse. This is a separate check from the shape check above (the shape
# is fine; the meaning is "the chip probe failed"), and is compared against
# the case-normalized value so "UNKNOWN" / "Unknown" are refused too, with
# this dedicated message rather than the generic shape message.
# Must stay byte-consistent with the copy in `validate_public_site.py`.
QUALITY_CARD_HARDWARE_CLASS_SENTINEL = "unknown"
# The wire-format PREFIX that names the "decode throughput" claim inside
# `boundary.unmeasured`. A numeric `legible.benefit.speedX` is a measured
# decode-throughput claim, so `boundary.unmeasured` must not also carry an
# entry disclaiming that same claim as unmeasured -- a card cannot both
# publish a throughput number and disclaim throughput as unmeasured.
# Matched by PREFIX, not exact-string membership: the real fixture card
# `qwen38-27b-mtp@m3ultra` disclaims decode throughput with the more
# informative "decode throughput (speedX, tracked in the MTP performance
# scorecard)", which still names the same claim and must still be refused
# beside a numeric speedX; an exact-literal rule would let that phrasing
# sit beside a numeric speedX undetected.
#
# The converse is deliberately NOT enforced: a null speedX does not require
# this entry to be present. `boundary.unmeasured` is not a contractually
# exhaustive list -- docs/quality-card-schema-v1.md specifies it only as a
# "list of non-empty strings" -- and the "why is there no number" guarantee
# is already structural, since `speedXStatus` is required non-empty text
# regardless of speedX.
#
# Must stay byte-consistent with the copy in `validate_public_site.py`.
QUALITY_CARD_SPEEDX_UNMEASURED_PREFIX = "decode throughput (speedX"
QUALITY_VERDICT_LABELS: Dict[str, str] = {
    "NO_GO": "Opt-in only",
    "PASS": "Passes review",
    "REFERENCE": "Production default",
    "EXACT": "Identical output",
    "UNMEASURED": "Not yet measured",
}
RELEASE_CATEGORIES = {"foundation", "operations", "product"}
RELEASE_CATEGORY_LABELS = {
    "foundation": "Foundation",
    "operations": "Operations",
    "product": "Product",
}
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
RELEASE_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})"
)
ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"
SITEMAP_NAMESPACE = "http://www.sitemaps.org/schemas/sitemap/0.9"
PUBLIC_SITE_URL = "https://bitworks-io.github.io/fast-mlx/"
SOCIAL_CARD_PATH = "assets/social-card.png"
SOCIAL_CARD_URL = PUBLIC_SITE_URL + SOCIAL_CARD_PATH
SOCIAL_CARD_ALT = (
    "Abstract emerald data loop connecting research, implementation, testing, "
    "and verified release checkpoints."
)
SOCIAL_CARD_SHA256 = (
    "aa4eaaa35a0dc2280752aab92e6731300e63d272cc5ba6340e0b626f5be610e0"
)
SOCIAL_CARD_BYTES = 1_011_297
SOCIAL_CARD_WIDTH = 1_200
SOCIAL_CARD_HEIGHT = 630
MAX_PUBLICATION_MANIFEST_BYTES = 1_048_576
CORE_PUBLIC_PAGE_PATHS = (
    "",
    "quickstart/",
    "license/",
    "status/",
    "process/",
    "methodology/",
    "capabilities/",
    "benchmarks/",
    "releases/",
    "research/",
)
REVIEWED_BENCHMARK_PUBLIC_PATHS = (
    "benchmarks/pld-echo-throughput/",
    "benchmarks/continuous-batch-c2-throughput/",
    "benchmarks/http-sse-operational-soak/",
)
CAPABILITY_DETAIL_PUBLIC_PATH = re.compile(
    r"capabilities/[a-z0-9]+(?:-[a-z0-9]+)*/"
)
RELEASE_DETAIL_PUBLIC_PATH = re.compile(
    r"releases/[a-z0-9]+(?:-[a-z0-9]+)*/"
)
RELEASE_DETAIL_DESCRIPTION = (
    "A reviewed fast-mlx public milestone with its exact commit, shipped surfaces, "
    "and unchanged claim boundary."
)
PUBLIC_PATH = re.compile(
    r"(?:[a-z0-9][a-z0-9.-]*/)*(?:[a-z0-9][a-z0-9.-]*/|[a-z0-9][a-z0-9.-]*\.(?:atom|html|json))"
)

# --- Served-engine benchmark ledger (site/served-benchmarks.json) ---------
# A sealed, reviewed ledger of fast-mlx measurements made with `fastmlx
# bench` against a serving engine identified by its build commit, not by
# name (see docs/quality-card-schema-v1.md "Engine build" and the internal
# ingest step, which is the only writer of a well-formed entry). Required
# like the release catalog (`load_release_catalog`) --
# never silently-optional like the quality-guide manifest -- because a
# ledger entry makes a load-bearing claim (a pack's decode rate vs its
# card's 8-bit reference) that must never be allowed to drift from the
# quality card it cites.
SERVED_BENCHMARK_POLICY = "reviewed-served-benchmarks-only"
SERVED_BENCHMARK_CLAIM_BOUNDARY = "fast-mlx-owned-results-only"
SERVED_BENCHMARK_TOP_KEYS = {
    "schemaVersion",
    "project",
    "policy",
    "claimBoundary",
    "updatedAt",
    "entries",
}
SERVED_BENCHMARK_ENTRY_KEYS = {
    "id",
    "measuredAt",
    "cardId",
    "packLabel",
    "packRevision",
    "hardwareClass",
    "chip",
    "engineBuild",
    "harness",
    "workload",
    "serving",
    "result",
    "controls",
    "rowSha256",
    "rerun",
}
SERVED_BENCHMARK_ENGINE_BUILD_KEYS = {"commit"}
# `combineCommit` names the public commit whose `scripts/fastmlx_bench.py`
# actually produced the row's `ratio` via `--combine` -- NOT necessarily the
# same commit as `publicCommit` (which names the harness that took the raw
# MEASUREMENT). A `--combine` defect fixed after a measurement was taken
# means the row had to be re-combined at a LATER commit than the one that
# measured it; recording only `publicCommit` would silently misattribute
# the combine step to a harness build that could not have produced this
# ratio (see the internal ingest step's own module docstring, the
# public-ancestry check on both commits).
SERVED_BENCHMARK_HARNESS_KEYS = {"publicCommit", "combineCommit", "benchSha256", "promptSetSha256"}
SERVED_BENCHMARK_WORKLOAD_KEYS = {
    "promptSet",
    "prompts",
    "maxTokens",
    "runs",
    "warmup",
    "temperature",
    "completionTokens",
}
SERVED_BENCHMARK_SERVING_KEYS = {"contextTokens", "mtp", "drafter", "promptLookup"}
# Slice 1 never publishes a serving toggle as "on" -- only whether it was
# explicitly disabled (a captured `--no-*` flag) or left at the engine's own
# default (the flag absent from the captured argv, never inferred further).
SERVED_BENCHMARK_SERVING_TRISTATE = {"off", "engine-default"}
SERVED_BENCHMARK_RESULT_KEYS = {
    "candidateDecodeTokS",
    "referenceDecodeTokS",
    "ratio",
    "candidatePasses",
    "referencePasses",
    "referenceDriftPct",
}
SERVED_BENCHMARK_CONTROLS = {
    "tokens": "verified",
    "owner": "verified",
    "drift": "verified",
    "magnitude": "plausible",
    "anchor": "first-token",
}
SERVED_BENCHMARK_RERUN_KEYS = {"measure", "combine", "publicView"}
SERVED_BENCHMARK_SHA256 = re.compile(r"[0-9a-f]{64}")
SERVED_BENCHMARK_MEASURED_AT = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
SERVED_BENCHMARK_RATE_FLOOR = 25.0
SERVED_BENCHMARK_RATE_CEILING = 2000.0
SERVED_BENCHMARK_DRIFT_CEILING_PCT = 5.0
SERVED_BENCHMARK_RATIO_TOLERANCE = 5e-4
SERVED_BENCHMARK_SPEEDX_TOLERANCE = 0.03
SERVED_BENCHMARK_IPV4 = re.compile(r"(?<!\d)\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?!\d)")
# A packLabel allowlist: starts with a lowercase letter or digit, every
# other character is a lowercase letter, digit, space, dot, slash, or
# hyphen, and the label ends in the literal word "pack" (e.g. "3.3-bit
# pack", "mixed 4/8-bit pack"). This is a SHAPE check, not a marker scan --
# it runs in addition to, never instead of, `_served_benchmark_marker_violation`
# (host shorthand, IPv4, `<path>`, `~`, private/engine markers), which is
# checked first so its specific reason is reported for a packLabel that
# carries one of those markers.
SERVED_BENCHMARK_PACK_LABEL = re.compile(r"[a-z0-9][a-z0-9 ./-]{0,55}pack")
# `--` flag NAMES a rerun string may carry (see the internal ingest step's
# own module docstring for the full `fastmlx bench` CLI surface); the four
# listener-identity flags (`--serve`, `--host`, `--port`, `--ctx-size`, the
# model PATH itself) never appear in a rerun template -- a rerun string is an
# operator INSTRUCTION, not a capture of the exact invocation that produced
# this row.
SERVED_BENCHMARK_RERUN_FLAG_NAMES = {
    "base-url",
    "model",
    "expect-listener-pid",
    "json",
    "runs",
    "warmup",
    "max-tokens",
    "temperature",
    "combine",
    "public-view",
}
# A rerun placeholder is a lowercase angle-bracketed name (e.g. `<pid>`), an
# operator fills in before running the command; `<role>.json` (the redirect
# target of the `measure` rerun) is the same shape with a literal `.json`
# suffix, since the role -- not just the extension -- is still a placeholder.
SERVED_BENCHMARK_RERUN_PLACEHOLDER = re.compile(r"<[a-z][a-z-]*>(?:\.json)?")
SERVED_BENCHMARK_RERUN_NUMBER = re.compile(r"\d+(?:\.\d+)?")
SERVED_BENCHMARK_RERUN_FILENAME = re.compile(r"[a-z0-9-]+\.json")
# --- Served-benchmark ledger ROW (site/served-benchmark-rows/<id>.json) ---
# The published, verifiable evidence a reader's `rowSha256` actually hashes
# (see the ingest tool's own module docstring, item 1). A ledger row is the
# public-view row with exactly one transformation: `--host`/`--port` and
# their values dropped from every role's `controls.flags.listenerFlags`.
# `SERVED_BENCHMARK_LEDGER_ROW_IPV4` is deliberately a SEPARATE, simpler
# regex than `SERVED_BENCHMARK_IPV4` above -- this one is the exact pattern
# the design doc specifies for the ledger-row refusal, independent of that
# other marker scan's own lookaround shape.
SERVED_BENCHMARK_LEDGER_ROW_IPV4 = re.compile(r"\b\d{1,3}(\.\d{1,3}){3}\b")
# Assembled by concatenation, never written as a literal -- this module is
# itself part of the public projection it scans row text for, so a literal
# occurrence here would match this project's OWN private-marker scan on
# every future publish (see `validate_public_repository.PRIVATE_MARKERS`,
# which the same "/" + "Users/" entry already lives in).
_SERVED_BENCHMARK_LEDGER_ROW_HOME_PATH = "/" + "Users/"
SERVED_BENCHMARK_ROWS_DIRNAME = "served-benchmark-rows"
# Mirrors `served_benchmark_entry._BOUNDARY_RE` byte-for-byte -- the ingest
# tool's own I5 refusal already enforces this shape before a row can ever be
# published, so this copy only ever runs against an already-well-formed
# `boundary` string; it exists here too because `derive_served_benchmark_
# fields_from_ledger_row` (below) must re-derive chip/prompts/maxTokens/
# runs/warmup/temperature from a LEDGER ROW file the ingest tool is not
# involved in reading (the build-time loader, and any future re-deriver).
SERVED_BENCHMARK_BOUNDARY_RE = re.compile(
    r"chip=(?P<chip>[^;]+) \([^)]*\); "
    r"promptSet=default-(?P<prompts>\d+)-prompt-set \([^)]*\); "
    r"maxTokens=(?P<maxTokens>\d+); "
    r"runs=(?P<runs>\d+); "
    r"warmup=(?P<warmup>\d+); "
    r"temperature=(?P<temperature>\d+(?:\.\d+)?)$"
)


@dataclass(frozen=True)
class Article:
    source: Path
    source_name: str
    slug: str
    title: str
    date: str
    theme: str
    summary: str
    reviewed_at: str
    body: str

    @property
    def output_file(self) -> str:
        return f"research/{self.slug}/index.html"

    @property
    def public_path(self) -> str:
        return f"research/{self.slug}/"


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="fast-mlx checkout root (defaults to the script's parent checkout)",
    )
    parser.add_argument("--output", type=Path, required=True, help="absent or empty output directory")
    return parser.parse_args(argv)


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"public-site build refused: {message}")


def read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {path}: {exc}")


def read_bounded_regular_json(path: Path, label: str, max_bytes: int) -> object:
    """Read one local JSON authority only after its filesystem boundary is sealed."""

    if path.is_symlink() or not path.is_file():
        fail(f"{label} is missing, not a file, or a symlink")
    try:
        size = path.stat().st_size
    except OSError as exc:
        fail(f"cannot stat {label}: {exc}")
    if size > max_bytes:
        fail(f"{label} exceeds the {max_bytes}-byte limit")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        fail(f"cannot read {label}: {exc}")
    if len(raw) > max_bytes:
        fail(f"{label} exceeds the {max_bytes}-byte limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        fail(f"{label} is not UTF-8: {exc}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        fail(f"cannot parse {label}: {exc}")


def prepare_output(output: Path, repository_root: Path) -> None:
    resolved = output.resolve()
    resolved_root = repository_root.resolve()
    if resolved == resolved_root or resolved_root in resolved.parents:
        fail("output must not be inside the repository root")
    if output.exists():
        if not output.is_dir():
            fail(f"output exists and is not a directory: {output}")
        if any(output.iterdir()):
            fail(f"output directory is not empty: {output}")
    else:
        output.mkdir(parents=True)


def validate_asset_tree(assets: Path) -> None:
    if not assets.is_dir() or assets.is_symlink():
        fail("site/assets is missing, not a directory, or a symlink")
    for path in assets.rglob("*"):
        if path.is_symlink():
            fail(f"site asset is a symlink: {path.relative_to(assets)}")
        if not path.is_file() and not path.is_dir():
            fail(f"site asset is not a regular file or directory: {path.relative_to(assets)}")
    validate_social_card(assets / "social-card.png", "site social card")


def validate_social_card(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        fail(f"{label} is missing, not a regular file, or a symlink")
    try:
        size = path.stat().st_size
    except OSError as exc:
        fail(f"cannot stat {label}: {exc}")
    if size != SOCIAL_CARD_BYTES:
        fail(f"{label} has the wrong byte count")
    raw = path.read_bytes()
    if len(raw) != SOCIAL_CARD_BYTES:
        fail(f"{label} has the wrong byte count")
    if hashlib.sha256(raw).hexdigest() != SOCIAL_CARD_SHA256:
        fail(f"{label} has the wrong SHA-256")
    if (
        raw[:8] != b"\x89PNG\r\n\x1a\n"
        or raw[8:12] != (13).to_bytes(4, "big")
        or raw[12:16] != b"IHDR"
        or int.from_bytes(raw[16:20], "big") != SOCIAL_CARD_WIDTH
        or int.from_bytes(raw[20:24], "big") != SOCIAL_CARD_HEIGHT
        or raw[24] != 8
        or raw[25] != 2
    ):
        fail(f"{label} is not the reviewed 1200x630 RGB PNG")


def strip_front_matter(text: str) -> Tuple[Dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        fail("unterminated article front matter")
    metadata: Dict[str, str] = {}
    for raw_line in text[4:end].splitlines():
        if not raw_line.strip():
            continue
        if ":" not in raw_line:
            fail(f"invalid front-matter line: {raw_line}")
        key, value = raw_line.split(":", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        metadata[key.strip()] = value
    return metadata, text[end + 5 :]


def plain_text(markdown: str) -> str:
    value = LINK.sub(lambda match: match.group(1), markdown)
    value = re.sub(r"[*_`>#]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def find_whitepaper_theme_block(lines: Sequence[str]) -> Optional[Tuple[int, int, str]]:
    """Return the half-open line span and joined value for a wrapped theme paragraph."""

    for start, line in enumerate(lines):
        match = WHITEPAPER_THEME.match(line.strip())
        if not match:
            continue
        values = [match.group(1).strip()] if match.group(1).strip() else []
        end = start + 1
        while end < len(lines):
            stripped = lines[end].strip()
            if (
                not stripped
                or stripped.startswith(("#", ">", "- ", "* ", "```", "|"))
                or re.match(r"^\d+\.\s", stripped)
            ):
                break
            values.append(stripped)
            end += 1
        return start, end, plain_text(" ".join(values))
    return None


def infer_metadata(source: Path, text: str) -> Tuple[str, str, str, str, str]:
    metadata, body = strip_front_matter(text)
    title = metadata.get("title", "")
    if not title:
        title_match = re.search(r"^#\s+(.+)$", body, flags=re.MULTILINE)
        if not title_match:
            fail(f"article has no H1 title: {source}")
        title = plain_text(title_match.group(1))

    date = metadata.get("date", source.name[:10])
    try:
        dt.date.fromisoformat(date)
    except ValueError:
        fail(f"article has invalid ISO date: {source}: {date}")

    lines = body.splitlines()
    theme_block = find_whitepaper_theme_block(lines)
    theme = metadata.get("whitepaper_theme", "")
    if not theme:
        theme = theme_block[2] if theme_block else "Inference research"

    paragraphs: List[str] = []
    current: List[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if theme_block and theme_block[0] <= index < theme_block[1]:
            continue
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if stripped.startswith(("#", ">", "- ", "* ", "```", "|")):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if re.match(r"^\d+\.\s", stripped):
            continue
        current.append(stripped)
    if current:
        paragraphs.append(" ".join(current))
    summary = next((plain_text(item) for item in paragraphs if len(plain_text(item)) > 60), "")
    if not summary:
        fail(f"article has no usable summary paragraph: {source}")
    if len(summary) > 220:
        summary = summary[:217].rsplit(" ", 1)[0] + "…"
    return title, date, theme, summary, body


def require_exact_keys(value: object, required: set[str], label: str) -> Dict[str, object]:
    if not isinstance(value, dict):
        fail(f"{label} is not an object")
    keys = set(value)
    if keys != required:
        missing = sorted(required - keys)
        extra = sorted(keys - required)
        fail(f"{label} keys differ from schema; missing={missing} extra={extra}")
    return value


def require_text(entry: Dict[str, object], key: str, label: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        fail(f"{label} has an empty or non-string {key}")
    if value != value.strip():
        fail(f"{label} {key} contains surrounding whitespace")
    return value


def require_nullable_text(entry: Dict[str, object], key: str, label: str) -> Optional[str]:
    """Require `key` to be a non-empty, non-whitespace-padded string, or explicitly null."""

    value = entry.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        fail(f"{label} {key} must be a non-empty string or null")
    if value != value.strip():
        fail(f"{label} {key} contains surrounding whitespace")
    return value


def require_iso_date(entry: Dict[str, object], key: str, label: str) -> str:
    value = require_text(entry, key, label)
    try:
        dt.date.fromisoformat(value)
    except ValueError:
        fail(f"{label} {key} is not an ISO date")
    return value


def require_iso_timestamp(entry: Dict[str, object], key: str, label: str) -> str:
    value = require_text(entry, key, label)
    try:
        dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        fail(f"{label} {key} is not an ISO-8601 timestamp")
    return value


def load_capability_catalog(
    repository_root: Path, published_slugs: Iterable[str]
) -> Dict[str, object]:
    """Load the strict public capability contract and bind it to reviewed article slugs."""

    manifest_path = repository_root / "site/capabilities.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        fail("site/capabilities.json is missing, not a file, or a symlink")
    catalog = require_exact_keys(
        read_json(manifest_path),
        {
            "schemaVersion",
            "policy",
            "claimBoundary",
            "updatedAt",
            "capabilities",
            "performanceHighlights",
        },
        "site/capabilities.json",
    )
    if catalog.get("schemaVersion") != 1:
        fail("site/capabilities.json must use schemaVersion 1")
    if catalog.get("policy") != "fast-mlx-owned-results-only":
        fail("capability policy must remain fast-mlx-owned-results-only")
    if catalog.get("claimBoundary") != "fast-mlx-owned-results-only":
        fail("capability claim boundary must remain fast-mlx-owned-results-only")
    require_iso_date(catalog, "updatedAt", "site/capabilities.json")

    serialized = json.dumps(catalog, ensure_ascii=False)
    for marker in PRIVATE_MARKERS:
        if marker.casefold() in serialized.casefold():
            fail(f"capability catalog contains private marker {marker!r}")

    published = set(published_slugs)
    capabilities = catalog.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities:
        fail("capability catalog must contain at least one capability")
    seen_ids: set[str] = set()
    for index, raw_entry in enumerate(capabilities):
        label = f"capability entry {index}"
        entry = require_exact_keys(
            raw_entry,
            {"id", "name", "status", "summary", "scope", "evidenceSlugs"},
            label,
        )
        identifier = require_text(entry, "id", label)
        if not SLUG.fullmatch(identifier) or identifier in seen_ids:
            fail(f"{label} has an invalid or duplicate id")
        seen_ids.add(identifier)
        require_text(entry, "name", label)
        require_text(entry, "summary", label)
        require_text(entry, "scope", label)
        status = require_text(entry, "status", label)
        if status not in CAPABILITY_STATUSES:
            fail(f"{label} has unknown status {status!r}")
        evidence_slugs = entry.get("evidenceSlugs")
        if (
            not isinstance(evidence_slugs, list)
            or not evidence_slugs
            or any(not isinstance(slug, str) or not SLUG.fullmatch(slug) for slug in evidence_slugs)
            or len(set(evidence_slugs)) != len(evidence_slugs)
        ):
            fail(f"{label} has invalid evidenceSlugs")
        unavailable = sorted(set(evidence_slugs) - published)
        if unavailable:
            fail(f"{label} cites unpublished evidence: {unavailable}")

    highlights = catalog.get("performanceHighlights")
    if not isinstance(highlights, list) or not highlights:
        fail("capability catalog must contain at least one performance highlight")
    seen_highlight_ids: set[str] = set()
    for index, raw_entry in enumerate(highlights):
        label = f"performance highlight {index}"
        entry = require_exact_keys(
            raw_entry,
            {
                "id",
                "metric",
                "label",
                "model",
                "hardware",
                "workload",
                "date",
                "decision",
                "caveat",
                "evidenceSlug",
            },
            label,
        )
        identifier = require_text(entry, "id", label)
        if not SLUG.fullmatch(identifier) or identifier in seen_highlight_ids:
            fail(f"{label} has an invalid or duplicate id")
        seen_highlight_ids.add(identifier)
        for key in ("metric", "label", "model", "hardware", "workload", "caveat"):
            require_text(entry, key, label)
        require_iso_date(entry, "date", label)
        decision = require_text(entry, "decision", label)
        if decision not in HIGHLIGHT_DECISIONS:
            fail(f"{label} has unknown decision {decision!r}")
        evidence_slug = require_text(entry, "evidenceSlug", label)
        if not SLUG.fullmatch(evidence_slug) or evidence_slug not in published:
            fail(f"{label} cites unpublished evidence {evidence_slug!r}")
    return catalog


def require_release_timestamp(
    entry: Dict[str, object], key: str, label: str
) -> Tuple[str, dt.datetime]:
    value = require_text(entry, key, label)
    if not RELEASE_TIMESTAMP.fullmatch(value):
        fail(f"{label} {key} is not an offset-aware release timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        fail(f"{label} {key} is not an offset-aware release timestamp")
    if parsed.utcoffset() is None:
        fail(f"{label} {key} is not an offset-aware release timestamp")
    return value, parsed


def require_public_link(value: object, label: str) -> Dict[str, str]:
    link = require_exact_keys(value, {"label", "path"}, label)
    link_label = require_text(link, "label", label)
    path = require_text(link, "path", label)
    if not PUBLIC_PATH.fullmatch(path):
        fail(f"{label} has an invalid public path")
    return {"label": link_label, "path": path}


def load_release_catalog(repository_root: Path) -> Dict[str, object]:
    """Load the strict reviewed public-release ledger without consulting Git or the network."""

    manifest_path = repository_root / "site/releases.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        fail("site/releases.json is missing, not a file, or a symlink")
    catalog = require_exact_keys(
        read_json(manifest_path),
        {
            "schemaVersion",
            "project",
            "policy",
            "claimBoundary",
            "updatedAt",
            "currentBoundary",
            "releases",
        },
        "site/releases.json",
    )
    if catalog.get("schemaVersion") != 1:
        fail("site/releases.json must use schemaVersion 1")
    if catalog.get("project") != "fast-mlx":
        fail("release project must remain fast-mlx")
    if catalog.get("policy") != "reviewed-public-releases-only":
        fail("release policy must remain reviewed-public-releases-only")
    if catalog.get("claimBoundary") != "fast-mlx-owned-results-only":
        fail("release claim boundary must remain fast-mlx-owned-results-only")
    require_iso_date(catalog, "updatedAt", "site/releases.json")

    serialized = json.dumps(catalog, ensure_ascii=False)
    for marker in PRIVATE_MARKERS:
        if marker.casefold() in serialized.casefold():
            fail(f"release catalog contains private marker {marker!r}")

    boundary = require_exact_keys(
        catalog.get("currentBoundary"),
        {"id", "label", "state", "summary", "evidence"},
        "current release boundary",
    )
    boundary_id = require_text(boundary, "id", "current release boundary")
    if not SLUG.fullmatch(boundary_id):
        fail("current release boundary has an invalid id")
    require_text(boundary, "label", "current release boundary")
    require_text(boundary, "summary", "current release boundary")
    if require_text(boundary, "state", "current release boundary") != "gated":
        fail("current release boundary state must remain gated")
    require_public_link(boundary.get("evidence"), "current release boundary evidence")

    releases = catalog.get("releases")
    if not isinstance(releases, list) or not releases:
        fail("release catalog must contain at least one release")
    seen_ids: set[str] = set()
    seen_commits: set[str] = set()
    previous_timestamp: Optional[dt.datetime] = None
    for index, raw_entry in enumerate(releases):
        label = f"release entry {index}"
        entry = require_exact_keys(
            raw_entry,
            {
                "id",
                "title",
                "publishedAt",
                "category",
                "state",
                "summary",
                "scope",
                "publicCommit",
                "publicLinks",
            },
            label,
        )
        identifier = require_text(entry, "id", label)
        if not SLUG.fullmatch(identifier) or identifier in seen_ids:
            fail(f"{label} has an invalid or duplicate id")
        seen_ids.add(identifier)
        for key in ("title", "summary", "scope"):
            require_text(entry, key, label)
        category = require_text(entry, "category", label)
        if category not in RELEASE_CATEGORIES:
            fail(f"{label} has unknown category {category!r}")
        if require_text(entry, "state", label) != "released":
            fail(f"{label} is not explicitly released")
        _timestamp, parsed_timestamp = require_release_timestamp(
            entry, "publishedAt", label
        )
        if previous_timestamp is not None and parsed_timestamp >= previous_timestamp:
            fail("release entries are not strictly newest-first")
        previous_timestamp = parsed_timestamp
        commit = require_text(entry, "publicCommit", label)
        if not COMMIT_SHA.fullmatch(commit) or commit in seen_commits:
            fail(f"{label} has an invalid or duplicate publicCommit")
        seen_commits.add(commit)
        links = entry.get("publicLinks")
        if not isinstance(links, list):
            fail(f"{label} publicLinks is not a list")
        seen_paths: set[str] = set()
        for link_index, raw_link in enumerate(links):
            link = require_public_link(raw_link, f"{label} public link {link_index}")
            if link["path"] in seen_paths:
                fail(f"{label} has a duplicate public link path")
            seen_paths.add(link["path"])
    return catalog


def _validate_config_flag_transfer(raw_flag_transfer: object, label: str) -> None:
    """Validate the OPTIONAL `config.flagTransfer` object (see
    docs/quality-card-schema-v1.md "Flag transfer"). `"--mtp"` is the ONLY
    allowed (and required, when `flagTransfer` is present at all) key.
    Enforces every cross-field rule the schema states: `divergentPrompts
    <= prompts`; `greedy == "exact"` requires `divergentPrompts == 0`;
    `greedy == "not_exact"` requires `divergentPrompts >= 1`; `evidence` is
    a non-empty, repo-relative-looking string (no absolute path, no
    `..`) -- private-marker hygiene is enforced separately, by this
    function's caller's own whole-document scan, exactly like every other
    card string.
    """
    flag_transfer = require_exact_keys(
        raw_flag_transfer, QUALITY_CARD_FLAG_TRANSFER_KEYS, f"{label} config.flagTransfer"
    )
    mtp = require_exact_keys(
        flag_transfer["--mtp"],
        QUALITY_CARD_FLAG_TRANSFER_MTP_KEYS,
        f"{label} config.flagTransfer['--mtp']",
    )
    greedy = mtp.get("greedy")
    if greedy not in QUALITY_CARD_FLAG_TRANSFER_MTP_GREEDY_VALUES:
        fail(f"{label} config.flagTransfer['--mtp'].greedy has unknown value {greedy!r}")
    divergent = mtp.get("divergentPrompts")
    if not isinstance(divergent, int) or isinstance(divergent, bool) or divergent < 0:
        fail(f"{label} config.flagTransfer['--mtp'].divergentPrompts must be an int >= 0")
    prompts = mtp.get("prompts")
    if not isinstance(prompts, int) or isinstance(prompts, bool) or prompts < 1:
        fail(f"{label} config.flagTransfer['--mtp'].prompts must be an int >= 1")
    if divergent > prompts:
        fail(
            f"{label} config.flagTransfer['--mtp'].divergentPrompts must be <= "
            "prompts"
        )
    if greedy == "exact" and divergent != 0:
        fail(
            f"{label} config.flagTransfer['--mtp'] greedy=exact requires "
            "divergentPrompts=0"
        )
    if greedy == "not_exact" and divergent < 1:
        fail(
            f"{label} config.flagTransfer['--mtp'] greedy=not_exact requires "
            "divergentPrompts>=1"
        )
    max_tokens = mtp.get("maxTokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        fail(f"{label} config.flagTransfer['--mtp'].maxTokens must be an int >= 1")
    evidence = mtp.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        fail(f"{label} config.flagTransfer['--mtp'].evidence must be a non-empty string")
    evidence_path = Path(evidence)
    if evidence_path.is_absolute() or ".." in evidence_path.parts:
        fail(
            f"{label} config.flagTransfer['--mtp'].evidence must be a repo-relative "
            "path with no absolute path and no '..'"
        )


def validate_quality_card_document(document: object, label: str) -> Dict[str, object]:
    """Validate a decoded `fast-mlx-quality-card-v1` document against the
    exact fail-closed schema `load_quality_guides` enforces for
    `site/quality-guides.json` -- factored out so any OTHER manifest using
    this schema (e.g. an internal, unpublished card set) can be checked with
    the identical rule set, without going through the public manifest's own
    filesystem conventions (path, symlink refusal, "optional/skip if
    absent"). ``label`` names the document in every failure message.
    """

    manifest = require_exact_keys(document, {"schema", "generatedAt", "cards"}, label)
    if manifest.get("schema") != QUALITY_GUIDE_SCHEMA:
        fail(f"{label} must use schema {QUALITY_GUIDE_SCHEMA!r}")
    require_iso_timestamp(manifest, "generatedAt", label)

    serialized = json.dumps(manifest, ensure_ascii=False)
    for marker in PRIVATE_MARKERS:
        if marker.casefold() in serialized.casefold():
            fail(f"{label} contains private marker {marker!r}")
    extended_marker = _quality_guide_private_marker(serialized)
    if extended_marker is not None:
        fail(f"{label} contains private marker {extended_marker!r}")

    cards = manifest.get("cards")
    if not isinstance(cards, list) or not cards:
        fail(f"{label} must contain at least one card")
    seen_ids: set[str] = set()
    # Identity for the (identity, residency, hardwareClass, engineBuild)
    # duplicate check below: (model.repo if non-null else model.hfPin,
    # normalized residency, config.hardwareClass, provenance.engineBuild.commit
    # or None). Two cards sharing all four would be indistinguishable to
    # fastmlx_launch.resolve_card's engine-build disambiguation (see
    # docs/quality-card-schema-v1.md "Engine build") -- refused here instead
    # of silently tying at runtime. hardwareClass is included because an
    # Ultra card and an M5 card for the same pack are two measurements of two
    # different facts, not duplicates (docs/quality-card-schema-v1.md § Card object, `config.hardwareClass`) --
    # but it is NEVER used to FILTER card admission (see "Engine build" in
    # that doc, which states the same never-filters contract); it only
    # disambiguates the key.
    seen_identity_engine_builds: set[tuple] = set()
    validated_cards: List[Dict[str, object]] = []
    for index, raw_card in enumerate(cards):
        card_label = f"{label} card entry {index}"
        card = require_exact_keys(
            raw_card,
            {
                "id",
                "model",
                "config",
                "verdict",
                "admission",
                "legible",
                "rawMetrics",
                "provenance",
                "boundary",
            },
            card_label,
        )
        identifier = require_text(card, "id", card_label)
        if not QUALITY_CARD_ID.fullmatch(identifier) or identifier in seen_ids:
            fail(f"{card_label} has an invalid or duplicate id")
        seen_ids.add(identifier)

        model = require_exact_keys(card.get("model"), {"family", "repo", "hfPin"}, f"{card_label} model")
        require_text(model, "family", f"{card_label} model")
        # repo/hfPin may be null for a card that has no distinct HF-hosted checkpoint
        # (an internal reference build, or an enhancement layered on an already-covered repo).
        for key in ("repo", "hfPin"):
            value = model.get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                fail(f"{card_label} model.{key} must be a non-empty string or null")

        raw_config = card.get("config")
        if not isinstance(raw_config, dict):
            fail(f"{card_label} config is not an object")
        config_keys = set(raw_config)
        if not (
            QUALITY_CARD_CONFIG_REQUIRED_KEYS <= config_keys <= QUALITY_CARD_CONFIG_ALLOWED_KEYS
        ):
            missing = sorted(QUALITY_CARD_CONFIG_REQUIRED_KEYS - config_keys)
            extra = sorted(config_keys - QUALITY_CARD_CONFIG_ALLOWED_KEYS)
            fail(f"{card_label} config keys differ from schema; missing={missing} extra={extra}")
        config = raw_config
        # `residency` is OPTIONAL and absent on every card predating it; absence means
        # "resident" (see fastmlx_launch.card_residency). When present it must be one
        # of the two recognized residencies -- an unrecognized value would silently
        # never match any launch, which a card author should never do unnoticed.
        if "residency" in config and config.get("residency") not in QUALITY_CARD_RESIDENCIES:
            fail(f"{card_label} config.residency has unknown value {config.get('residency')!r}")
        # `quant` is null for a pure enhancement card (e.g. MTP) that carries no distinct
        # quantization variant; `groupSize` is null for a quant format (e.g. mxfp8) that has
        # no blockwise/affine group concept.
        raw_quant = config.get("quant")
        if raw_quant is not None:
            quant = require_exact_keys(
                raw_quant, {"bits", "groupSize", "mixedBit", "note"}, f"{card_label} config.quant"
            )
            if not isinstance(quant.get("bits"), int) or isinstance(quant.get("bits"), bool):
                fail(f"{card_label} config.quant.bits is not an int")
            group_size = quant.get("groupSize")
            if group_size is not None and (
                not isinstance(group_size, int) or isinstance(group_size, bool)
            ):
                fail(f"{card_label} config.quant.groupSize must be an int or null")
            if not isinstance(quant.get("mixedBit"), bool):
                fail(f"{card_label} config.quant.mixedBit is not a bool")
            note = quant.get("note")
            if note is not None and (not isinstance(note, str) or not note.strip()):
                fail(f"{card_label} config.quant.note must be a non-empty string or null")
        # `flagTransfer` is OPTIONAL; absence means every flag's transfer is
        # unmeasured for this card (see "Flag transfer" above).
        if "flagTransfer" in config:
            _validate_config_flag_transfer(config.get("flagTransfer"), card_label)
        require_text(config, "enhancement", f"{card_label} config")
        hardware_class = require_text(config, "hardwareClass", f"{card_label} config")
        if hardware_class.strip().lower() == QUALITY_CARD_HARDWARE_CLASS_SENTINEL:
            fail(
                f"{card_label} config.hardwareClass {hardware_class!r} is the "
                "chip-probe-failed sentinel value ('unknown'); chip identification failed "
                "when this card was measured, so the card must not claim a hardware class "
                "it could not measure"
            )
        if not QUALITY_CARD_HARDWARE_CLASS.fullmatch(hardware_class):
            fail(
                f"{card_label} config.hardwareClass {hardware_class!r} is not a canonical "
                "hardware class (expected lowercase alphanumeric segments joined by single "
                "hyphens, e.g. 'apple-m3-ultra')"
            )

        verdict = require_text(card, "verdict", card_label)
        if verdict not in QUALITY_VERDICTS:
            fail(f"{card_label} has unknown verdict {verdict!r}")

        admission = require_exact_keys(
            card.get("admission"), {"default", "optIn", "reason"}, f"{card_label} admission"
        )
        if not isinstance(admission.get("default"), bool):
            fail(f"{card_label} admission.default is not a bool")
        if not isinstance(admission.get("optIn"), bool):
            fail(f"{card_label} admission.optIn is not a bool")
        require_text(admission, "reason", f"{card_label} admission")
        # Claim-integrity gate (NOT an admission gate: the Swift + Python serve
        # gates key on `verdict` alone -- see "Admission discriminator rules" in
        # docs/quality-card-schema-v1.md). admission.{default,optIn} are
        # validated-for-agreement CLAIMS about verdict that the public renderer
        # surfaces (quality_admission_framing reads admission.reason); a card
        # publishing a NO_GO verdict alongside admission.default=True would
        # render "safe silent default" for a pack the serve gate refuses.
        # R1: a NO_GO card can never be the silent production default.
        # R2: a NO_GO card must stay electable, because the refusal message
        #     tells the operator to elect it with --accept-quality.
        #
        # Deliberately ONE-SIDED. `verdict != "NO_GO"` must NOT imply
        # default=True: "may this be the SILENT production default?" is a
        # rollout/trust question that a passing MEASUREMENT does not settle,
        # and scripts/tests/fixtures/quality-guides.sample.json ships two
        # passing cards that say so -- a PASS card whose quality is
        # vendor-reported rather than independently gated, and an EXACT card
        # whose reason is "opt-in pending broader rollout". A biconditional
        # here would reject both, so it would be a rule about what makes the
        # shipped manifest pass rather than a rule about the claim.
        #
        # UNMEASURED is exempt: no UNMEASURED card has ever been emitted or
        # shipped, so a rule for it would be an unreachable branch.
        #
        # admission.optIn is a FROZEN CONSTANT true on every verdict, not
        # conditioned on NO_GO like R1/R2 above. It is derivable from
        # nothing and carries no per-card information, but the field is
        # retained in the wire format because released fastmlx-serve
        # binaries decode it non-optionally: a missing field fails OPEN
        # (the whole manifest fails to decode and serve announces
        # `quality_cards=none`). Resolved: this used to be left
        # unconstrained on non-NO_GO verdicts pending a decision between
        # two contradictory conventions in the repo; that question is now
        # closed to redefinition. Unlike `default`'s one-sidedness (which
        # is PERMANENT -- two honest passing cards in
        # scripts/tests/fixtures/quality-guides.sample.json carry
        # `default: false`), this exemption was TEMPORARY.
        if admission["optIn"] is not True:
            fail(
                f"{card_label} admission.optIn must be true "
                f"(every card is electable via --accept-quality)"
            )
        if verdict == "NO_GO":
            if admission["default"] is not False:
                fail(
                    f"{card_label} admission.default {admission['default']!r} disagrees "
                    f"with verdict 'NO_GO' (a NO_GO pack is never a silent default)"
                )

        legible = require_exact_keys(
            card.get("legible"),
            {"tier", "headline", "nextWordDrift", "regressionFocus", "example", "benefit"},
            f"{card_label} legible",
        )
        tier = require_text(legible, "tier", f"{card_label} legible")
        if tier not in QUALITY_TIERS:
            fail(f"{card_label} legible has unknown tier {tier!r}")
        require_text(legible, "headline", f"{card_label} legible")
        # regressionFocus is only meaningful for a card that admits a quality trade
        # (NO_GO/PASS); REFERENCE (the reference itself) and EXACT (identical output)
        # have no regression to report and may leave it null.
        if verdict in {"NO_GO", "PASS"}:
            require_text(legible, "regressionFocus", f"{card_label} legible")
        else:
            require_nullable_text(legible, "regressionFocus", f"{card_label} legible")

        # nextWordDrift itself is null for REFERENCE (no drift concept vs itself).
        raw_next_word_drift = legible.get("nextWordDrift")
        if raw_next_word_drift is not None:
            next_word_drift = require_exact_keys(
                raw_next_word_drift,
                {"oneInK", "top1AgreementPct"},
                f"{card_label} legible.nextWordDrift",
            )
            # oneInK is null for EXACT (identical output, so there is no "1 in K" to state).
            one_in_k = next_word_drift.get("oneInK")
            if one_in_k is not None and (
                not isinstance(one_in_k, int) or isinstance(one_in_k, bool)
            ):
                fail(f"{card_label} legible.nextWordDrift.oneInK must be an int or null")
            top1 = next_word_drift.get("top1AgreementPct")
            if not isinstance(top1, (int, float)) or isinstance(top1, bool):
                fail(f"{card_label} legible.nextWordDrift.top1AgreementPct is not a number")

        # "Unquantified" means the output differs from the reference but the size of
        # that difference was never measured: it may ONLY pair with a NO_GO card whose
        # nextWordDrift is null (there is no honest "1 in K" to report), and conversely
        # a NO_GO/PASS card with a null nextWordDrift must be tiered Unquantified --
        # PASS can therefore never carry a null nextWordDrift.
        if tier == "Unquantified":
            if verdict != "NO_GO":
                fail(f"{card_label} legible has tier Unquantified but verdict is not NO_GO")
            if raw_next_word_drift is not None:
                fail(f"{card_label} legible has tier Unquantified but nextWordDrift is not null")
        elif raw_next_word_drift is None and verdict in {"NO_GO", "PASS"}:
            fail(
                f"{card_label} legible has a null nextWordDrift on verdict {verdict!r} "
                "but tier is not Unquantified"
            )

        example = require_exact_keys(
            legible.get("example"),
            {"status", "prompt", "referenceOutput", "configOutput", "note"},
            f"{card_label} legible.example",
        )
        example_status = example.get("status")
        if example_status not in QUALITY_EXAMPLE_STATUSES:
            fail(f"{card_label} legible.example has unknown status {example_status!r}")

        benefit = require_exact_keys(
            legible.get("benefit"), {"fit", "speedX", "speedXStatus"}, f"{card_label} legible.benefit"
        )
        # fit is null for an enhancement card with no footprint of its own (e.g. MTP).
        require_nullable_text(benefit, "fit", f"{card_label} legible.benefit")
        speed_x = benefit.get("speedX")
        if speed_x is not None and (not isinstance(speed_x, (int, float)) or isinstance(speed_x, bool)):
            fail(f"{card_label} legible.benefit.speedX must be a number or null")
        require_text(benefit, "speedXStatus", f"{card_label} legible.benefit")

        # rawMetrics is always an object; it may be empty only for EXACT (identical
        # output has no per-config metric of its own to show).
        raw_metrics = card.get("rawMetrics")
        if not isinstance(raw_metrics, dict):
            fail(f"{card_label} rawMetrics must be an object")
        if not raw_metrics and verdict != "EXACT":
            fail(f"{card_label} rawMetrics must be a non-empty object")

        raw_provenance = card.get("provenance")
        if not isinstance(raw_provenance, dict):
            fail(f"{card_label} provenance is not an object")
        provenance_keys = set(raw_provenance)
        if not (
            QUALITY_CARD_PROVENANCE_REQUIRED_KEYS
            <= provenance_keys
            <= QUALITY_CARD_PROVENANCE_ALLOWED_KEYS
        ):
            missing = sorted(QUALITY_CARD_PROVENANCE_REQUIRED_KEYS - provenance_keys)
            extra = sorted(provenance_keys - QUALITY_CARD_PROVENANCE_ALLOWED_KEYS)
            fail(f"{card_label} provenance keys differ from schema; missing={missing} extra={extra}")
        provenance = raw_provenance
        # `engineBuild` is OPTIONAL; when present its only allowed key is
        # `commit`, a lowercase 40-hex git sha.
        raw_engine_build = provenance.get("engineBuild")
        engine_build_commit: Optional[str] = None
        if raw_engine_build is not None:
            engine_build = require_exact_keys(
                raw_engine_build, {"commit"}, f"{card_label} provenance.engineBuild"
            )
            commit = engine_build.get("commit")
            if not isinstance(commit, str) or not QUALITY_CARD_ENGINE_BUILD_COMMIT.fullmatch(
                commit
            ):
                fail(
                    f"{card_label} provenance.engineBuild.commit must be a lowercase "
                    "40-hex string"
                )
            engine_build_commit = commit
        # `config.flagTransfer` measures how a served-engine flag (e.g.
        # `--mtp`) behaves on THIS card's exact engine build; without a
        # recorded `provenance.engineBuild.commit` the measurement has no
        # build to be true of (see docs/quality-card-schema-v1.md "Flag
        # transfer").
        if "flagTransfer" in config and engine_build_commit is None:
            fail(
                f"{card_label} config.flagTransfer requires "
                "provenance.engineBuild.commit"
            )
        source = require_text(provenance, "source", f"{card_label} provenance")
        if source not in QUALITY_PROVENANCE_SOURCES:
            fail(f"{card_label} provenance has unknown source {source!r}")
        vendor = provenance.get("vendor")
        if source == "vendor-reported":
            if not isinstance(vendor, str) or not vendor.strip():
                fail(f"{card_label} provenance.vendor is required when source is vendor-reported")
        elif vendor is not None:
            fail(f"{card_label} provenance.vendor must be null unless source is vendor-reported")
        for key in ("method", "hardware"):
            require_text(provenance, key, f"{card_label} provenance")
        # harnessGitSHA/corpusId/sourceVerdict/measuredAt are null for a curated card not
        # sourced from a dated gate verdict (e.g. the MTP exactness card).
        for key in ("harnessGitSHA", "corpusId", "sourceVerdict"):
            require_nullable_text(provenance, key, f"{card_label} provenance")
        if provenance.get("measuredAt") is not None:
            require_iso_timestamp(provenance, "measuredAt", f"{card_label} provenance")
        confound = provenance.get("confound")
        if confound is not None and (not isinstance(confound, str) or not confound.strip()):
            fail(f"{card_label} provenance.confound must be a non-empty string or null")

        # Measured-PASS rule. SCOPED to provenance.source == "fast-mlx-measured":
        # a PASS card that fast-mlx itself measured must carry the evidence for
        # the Near-lossless claim and be default-admitting (mirrors
        # scripts/emit_quality_card_teacher.py `pass_card_violations`; see
        # "PASS authoring (teacher path)" in docs/quality-card-schema-v1.md).
        # It is deliberately NOT applied to vendor-reported / modeled PASS
        # cards: scripts/tests/fixtures/quality-guides.sample.json ships a
        # vendor-reported PASS card (Noticeable, admission.default false, no
        # top-1/ppl) that must keep validating, which is also why the
        # one-sided `default` rule above stays permanent for other sources.
        if verdict == "PASS" and source == "fast-mlx-measured":
            measured_pass_label = f"{card_label} (card id {identifier!r}) fast-mlx-measured PASS"

            def measured_number(key: str) -> Optional[float]:
                value = raw_metrics.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return None
                return float(value) if math.isfinite(value) else None

            if tier != "Near-lossless":
                fail(f"{measured_pass_label} requires legible.tier 'Near-lossless', got {tier!r}")
            if legible.get("nextWordDrift") is None:
                fail(f"{measured_pass_label} requires a non-null legible.nextWordDrift")
            top1 = measured_number("top1AgreementPct")
            if top1 is None or top1 < 99.0:
                fail(
                    f"{measured_pass_label} requires numeric rawMetrics.top1AgreementPct >= 99, "
                    f"got {raw_metrics.get('top1AgreementPct')!r}"
                )
            for ppl_key in ("pplDeltaPct", "pplDeltaUpperPct"):
                ppl_value = measured_number(ppl_key)
                if ppl_value is None or ppl_value > 1.0:
                    fail(
                        f"{measured_pass_label} requires numeric rawMetrics.{ppl_key} <= 1, "
                        f"got {raw_metrics.get(ppl_key)!r}"
                    )
            if admission["default"] is not True:
                fail(
                    f"{measured_pass_label} requires admission.default true, "
                    f"got {admission['default']!r}"
                )
            boundary_for_pass = card.get("boundary")
            if not isinstance(boundary_for_pass, dict) or "measuredNewTokens" not in boundary_for_pass:
                fail(f"{measured_pass_label} requires boundary.measuredNewTokens")

        # boundary.measuredNewTokens (optional; required on a fast-mlx-measured
        # PASS card above): the generation length the card was measured over.
        # An integer >= 1; `type(...) is int` excludes bool (an int subclass).
        # Mirrored by scripts/validate_public_site.py.
        boundary_raw = card.get("boundary")
        boundary_keys = {"scope", "unmeasured"}
        if isinstance(boundary_raw, dict) and "measuredNewTokens" in boundary_raw:
            boundary_keys = boundary_keys | {"measuredNewTokens"}
        boundary = require_exact_keys(boundary_raw, boundary_keys, f"{card_label} boundary")
        require_text(boundary, "scope", f"{card_label} boundary")
        if "measuredNewTokens" in boundary:
            measured_new_tokens = boundary["measuredNewTokens"]
            if type(measured_new_tokens) is not int or measured_new_tokens < 1:
                fail(
                    f"{card_label} (card id {identifier!r}) boundary.measuredNewTokens "
                    f"must be an integer >= 1, "
                    f"got {measured_new_tokens!r}"
                )
        unmeasured = boundary.get("unmeasured")
        if not isinstance(unmeasured, list) or any(
            not isinstance(item, str) or not item.strip() for item in unmeasured
        ):
            fail(f"{card_label} boundary.unmeasured must be a list of non-empty strings")
        if not unmeasured:
            # A card claiming nothing is unmeasured is a claim of completeness
            # no measurement in this schema actually makes; the empty-list case
            # slips past the check above (`any(...)` over `[]` is False).
            fail(f"{card_label} boundary.unmeasured must not be empty")
        # A numeric speedX is a measured decode-throughput claim;
        # boundary.unmeasured must not also carry an entry disclaiming that
        # same claim as unmeasured (matched by PREFIX -- see the
        # QUALITY_CARD_SPEEDX_UNMEASURED_PREFIX comment above). The converse
        # (a null speedX requiring the entry) is deliberately NOT enforced;
        # see that comment. This is the producer-side mirror of the
        # identical check in validate_public_site.validate_quality_guide_manifest
        # -- without it, build_site could publish a self-contradictory card
        # that only the separate validator would later catch.
        if isinstance(speed_x, (int, float)) and not isinstance(speed_x, bool):
            offending = [
                item for item in unmeasured if item.strip().startswith(QUALITY_CARD_SPEEDX_UNMEASURED_PREFIX)
            ]
            if offending:
                fail(
                    f"{card_label} legible.benefit.speedX is numeric but "
                    f"boundary.unmeasured still lists {offending[0]!r}"
                )

        repo_value = model.get("repo")
        identity = repo_value if repo_value is not None else model.get("hfPin")
        residency_value = config.get("residency") if isinstance(config, dict) else None
        hardware_class_value = config.get("hardwareClass") if isinstance(config, dict) else None
        identity_key = (
            identity,
            residency_value or "resident",
            hardware_class_value,
            engine_build_commit,
        )
        if identity_key in seen_identity_engine_builds:
            fail(
                f"{card_label} duplicates another card's (identity, residency, "
                f"hardwareClass, engineBuild) combination {identity_key!r}"
            )
        seen_identity_engine_builds.add(identity_key)

        validated_cards.append(card)

    manifest["cards"] = validated_cards
    return manifest


def load_quality_guides(repository_root: Path) -> Optional[Dict[str, object]]:
    """Load the optional public quality-card manifest (`fast-mlx-quality-card-v1`).

    The manifest is owned by a separate emitter (`scripts/emit_quality_card.py`).
    This function's own absence-handling is unchanged: a missing manifest
    returns `None` rather than raising, and the quality-guide page is simply
    skipped, exactly like an optional section. Presence is validated with a
    strict, fail-closed schema (`validate_quality_card_document`) — an
    unknown schema, verdict, provenance source, or a card missing a required
    field refuses the build rather than rendering a partial or fabricated
    card.

    A missing manifest is NO LONGER a no-op for the OVERALL build, though:
    `load_served_benchmarks` (below) is required like the release catalog
    and refuses (`L2`) when its `quality_manifest` argument is `None`, so a
    missing manifest now fails the build one call later, at the served
    ledger's own required load, rather than here.
    """

    manifest_path = repository_root / "site/quality-guides.json"
    if not manifest_path.exists() and not manifest_path.is_symlink():
        return None
    if manifest_path.is_symlink() or not manifest_path.is_file():
        fail("site/quality-guides.json is present but not a regular file")
    return validate_quality_card_document(read_json(manifest_path), "site/quality-guides.json")


def _served_benchmark_marker_violation(text: str) -> Optional[str]:
    """Return a short reason `text` (any serialized fragment of a served-
    benchmark entry) carries a marker that must never reach a published
    entry, or `None` when it is clean. Neutralises this project's own binary
    name before the third-party-engine scan -- the SAME order and the same
    NUL-splice technique `fastmlx_bench.publishability_control` and
    `validate_public_repository.validate_no_third_party_engine_marker` both
    use (see either docstring for why an empty-string replacement would
    manufacture a false match) -- so a positive-control occurrence of the
    public token "fastmlx-serve" never trips this scan.
    """
    lowered = text.casefold()
    for marker in PRIVATE_MARKERS:
        if marker.casefold() in lowered:
            return f"private marker {marker!r}"
    neutralized = lowered.replace(
        validate_public_repository.OWN_BINARY_NAME.casefold(), "\x00"
    )
    for marker in validate_public_repository.THIRD_PARTY_ENGINE_MARKERS:
        if marker.casefold() in neutralized:
            return "a third-party engine name"
    if QUALITY_GUIDE_HOST_SHORTHAND.search(text):
        return "the fleet host shorthand"
    if SERVED_BENCHMARK_IPV4.search(text):
        return "an IPv4 address"
    if "<path>" in text:
        return "a <path> placeholder"
    if "~" in text:
        return "a literal '~'"
    return None


def _served_benchmark_row_listener_flags(row: Dict[str, object]) -> Optional[Dict[str, object]]:
    controls = row.get("controls") if isinstance(row, dict) else None
    flags = controls.get("flags") if isinstance(controls, dict) else None
    listener_flags = flags.get("listenerFlags") if isinstance(flags, dict) else None
    return listener_flags if isinstance(listener_flags, dict) else None


def _served_benchmark_row_path_misplaced(row: Dict[str, object]) -> bool:
    """AMENDMENT A1: `<path>` is the REQUIRED redaction of `--model`'s value
    inside a role's `listenerFlags` list -- the ingest tool's own I4 check
    (`_listener_token_failures`) enforces that `--model`'s value must BE
    `<path>`, so it is the ONE place `<path>` is legitimate anywhere in a
    served-benchmark ledger row. Returns `True` if `<path>` appears anywhere
    ELSE in `row` -- a different flag's value inside a `listenerFlags` list,
    a longer string such as `--key=<path>` (an EXACT-match requirement at
    the permitted position, not a substring one), inside another string
    entirely (e.g. `boundary`), or outside `listenerFlags` altogether.
    Walks the whole row recursively; only a `listenerFlags` role's own list
    (identity-matched against the row's own `controls.flags.listenerFlags`
    dict, never by value) gets the positional exemption.
    """

    listener_flags = _served_benchmark_row_listener_flags(row)

    def is_role_list(candidate: object) -> bool:
        return listener_flags is not None and any(
            candidate is tokens for tokens in listener_flags.values()
        )

    def scan(node: object) -> bool:
        if isinstance(node, dict):
            return any(scan(value) for value in node.values())
        if isinstance(node, list):
            if is_role_list(node):
                for index, token in enumerate(node):
                    if not isinstance(token, str) or "<path>" not in token:
                        continue
                    if token == "<path>" and index > 0 and node[index - 1] == "--model":
                        continue
                    return True
                return False
            return any(scan(item) for item in node)
        if isinstance(node, str):
            return "<path>" in node
        return False

    return scan(row)


def served_benchmark_ledger_row_violation(
    row_text: str, row_obj: Dict[str, object]
) -> Optional[str]:
    """The item-1 marker/IPv4 check for a served-benchmark LEDGER ROW --
    shared, byte-identical, between the ingest tool's I9 refusal (before it
    ever writes `--ledger-row-out`) and the build-time loader's re-check of
    the committed `site/served-benchmark-rows/<id>.json` file. `row_text` is
    the exact canonical serialization of `row_obj` (see
    `served_benchmark_entry._canonical_ledger_row_bytes`).

    Refuses if `row_text` carries an IPv4 literal, `localhost`, a raw
    `--host`/`--port` token (the one transformation a ledger row is supposed
    to have already had applied), an absolute private home-directory path
    (see `_SERVED_BENCHMARK_LEDGER_ROW_HOME_PATH` below -- assembled by
    concatenation for the same reason `PRIVATE_MARKERS` is, so this module's
    own source never carries the literal marker), or a literal `~/`. Then
    (AMENDMENT A1) checks `row_obj` STRUCTURALLY for a misplaced `<path>`
    (`_served_benchmark_row_path_misplaced`, above) -- `<path>` is a
    REQUIRED redaction inside `listenerFlags`, not a marker, so the generic
    entry-level `_served_benchmark_marker_violation` cannot be run against a
    row's raw text unmodified. Only once the row is proven to carry `<path>`
    in nothing but permitted positions does this neutralize exactly those
    occurrences (never any other occurrence, because none can remain) and
    run `_served_benchmark_marker_violation` on the result, so every other
    rule -- private markers, a third-party engine name, the fleet host
    shorthand, and a literal `~` -- still applies in full.
    """
    if SERVED_BENCHMARK_LEDGER_ROW_IPV4.search(row_text):
        return "an IPv4 literal"
    if "localhost" in row_text.casefold():
        return "a 'localhost' literal"
    if "--host" in row_text:
        return "a --host token"
    if "--port" in row_text:
        return "a --port token"
    if _SERVED_BENCHMARK_LEDGER_ROW_HOME_PATH in row_text:
        return "a private home-directory path"
    if "~/" in row_text:
        return "a literal '~/'"
    if _served_benchmark_row_path_misplaced(row_obj):
        return "a misplaced <path> placeholder"
    neutralized_text = row_text.replace("<path>", "\x00")
    return _served_benchmark_marker_violation(neutralized_text)


def _served_benchmark_rerun_failures(key: str, value: object, label: str) -> List[str]:
    """Accumulate-style shape check for one `rerun.<key>` string -- see the
    served-ledger design notes section 3 (`I12`/`L12`): must start
    `fastmlx bench `, every `--`-prefixed token must name an allowed flag,
    and every remaining token must be a placeholder, an integer, the literal
    `>`, or an `[a-z0-9-]+.json` filename -- never a raw path, host, or port.
    """
    failures: List[str] = []
    if not isinstance(value, str) or not value.strip():
        return [f"{label} rerun.{key} has an empty or non-string value"]
    tokens = value.split(" ")
    if len(tokens) < 3 or tokens[0] != "fastmlx" or tokens[1] != "bench":
        return [f"{label} rerun.{key} does not start with 'fastmlx bench '"]
    for token in tokens[2:]:
        if token == ">":
            continue
        if token.startswith("--"):
            if token[2:] not in SERVED_BENCHMARK_RERUN_FLAG_NAMES:
                failures.append(f"{label} rerun.{key} has an unknown flag {token!r}")
            continue
        if (
            SERVED_BENCHMARK_RERUN_PLACEHOLDER.fullmatch(token)
            or SERVED_BENCHMARK_RERUN_NUMBER.fullmatch(token)
            or SERVED_BENCHMARK_RERUN_FILENAME.fullmatch(token)
        ):
            continue
        failures.append(f"{label} rerun.{key} has an unrecognised token {token!r}")
    return failures


def _served_benchmark_number(value: object) -> Optional[float]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return float(value)


def _served_benchmark_row_completion_token_bounds(row: Dict[str, object]) -> Tuple[int, int]:
    values: List[int] = []
    for arm in row.get("arms", []):
        for reading in arm.get("readings", []):
            tokens = reading.get("completionTokens")
            if isinstance(tokens, int) and not isinstance(tokens, bool):
                values.append(tokens)
    if not values:
        fail("served-benchmark ledger row has no completionTokens")
    return min(values), max(values)


def derive_served_benchmark_fields_from_ledger_row(row: Dict[str, object]) -> Dict[str, object]:
    """Re-derive every served-benchmark entry field a LEDGER ROW (the
    public-view row with `--host`/`--port` already stripped from every
    role's `listenerFlags` -- see `served_benchmark_ledger_row_violation`)
    determines, using the exact rounding/parsing rules the ingest tool used
    to derive them (`served_benchmark_entry.build_entry`, prior to this
    function existing). ONE source of truth: the ingest tool calls this in
    place of its own former private copy of this logic, and the build-time
    loader calls it again, independently, against the committed row FILE, so
    an entry's numbers can never silently drift from the row that is
    supposed to justify them. Fails closed (`fail()`) on a malformed row --
    in practice unreachable from the ingest tool, whose own I1/I4/I5 checks
    already refuse a row this malformed before this function is ever
    called.
    """

    arms = row.get("arms")
    if not isinstance(arms, list) or len(arms) < 2:
        fail("served-benchmark ledger row does not have two arms")
    candidate_arm, reference_arm = arms[0], arms[1]
    candidate_rate = round(float(candidate_arm["medianDecodeTokS"]), 2)
    reference_rate = round(float(reference_arm["medianDecodeTokS"]), 2)
    ratio = row.get("ratio")
    if ratio is None:
        fail("served-benchmark ledger row has no ratio")
    ratio = round(float(ratio), 4)
    candidate_passes = len(candidate_arm["passRates"])
    reference_passes = len(reference_arm["passRates"])
    drift = row["controls"]["drift"]
    drift_ratio = float(drift["driftRatio"])
    drift_pct = round(abs(1 - drift_ratio) * 100, 2)

    listener_flags = row["controls"]["flags"]["listenerFlags"]["candidate"]
    context_tokens = None
    for index, token in enumerate(listener_flags):
        if token == "--ctx-size":
            context_tokens = int(listener_flags[index + 1])
    if context_tokens is None:
        fail("served-benchmark ledger row has no --ctx-size in listener flags")
    mtp = "off" if "--no-mtp" in listener_flags else "engine-default"
    drafter = "off" if "--no-drafter" in listener_flags else "engine-default"
    prompt_lookup = "off" if "--no-pld" in listener_flags else "engine-default"

    min_tokens, max_tokens = _served_benchmark_row_completion_token_bounds(row)

    boundary = row.get("boundary")
    if not isinstance(boundary, str):
        fail("served-benchmark ledger row has no boundary")
    match = SERVED_BENCHMARK_BOUNDARY_RE.fullmatch(boundary)
    if match is None:
        fail(f"served-benchmark ledger row boundary {boundary!r} is not well-formed")

    harness = row.get("harness")
    if not isinstance(harness, dict):
        fail("served-benchmark ledger row has no harness")

    return {
        "candidateDecodeTokS": candidate_rate,
        "referenceDecodeTokS": reference_rate,
        "ratio": ratio,
        "candidatePasses": candidate_passes,
        "referencePasses": reference_passes,
        "referenceDriftPct": drift_pct,
        "completionTokens": [min_tokens, max_tokens],
        "contextTokens": context_tokens,
        "mtp": mtp,
        "drafter": drafter,
        "promptLookup": prompt_lookup,
        "benchSha256": harness.get("benchSha256"),
        "promptSetSha256": harness.get("promptSetSha256"),
        "chip": match.group("chip"),
        "prompts": int(match.group("prompts")),
        "maxTokens": int(match.group("maxTokens")),
        "runs": int(match.group("runs")),
        "warmup": int(match.group("warmup")),
        "temperature": float(match.group("temperature")),
    }


def validate_served_benchmark_entry(
    raw_entry: object, cards_by_id: Dict[str, Dict[str, object]], label: str
) -> Dict[str, object]:
    """Validate ONE served-benchmark ledger entry against its own schema
    (`SERVED_BENCHMARK_ENTRY_KEYS` and this function's own checks below)
    and cross-check it against the quality card it cites (`cards_by_id`,
    keyed by `cardId`). Shared, byte-identical code path for both the
    build-time loader (`load_served_benchmarks`, below) and the internal
    ingest step's own `I8` self-check -- an entry this function accepts is
    guaranteed to pass the loader later, and an entry it
    refuses can never reach `site/served-benchmarks.json` in the first
    place. `fail()`s (SystemExit) on the FIRST violation, exactly like every
    other loader in this module (see `load_release_catalog`).
    """

    entry = require_exact_keys(raw_entry, SERVED_BENCHMARK_ENTRY_KEYS, label)
    identifier = require_text(entry, "id", label)
    if not SLUG.fullmatch(identifier):
        fail(f"{label} has an invalid id")
    measured_at = require_text(entry, "measuredAt", label)
    if not SERVED_BENCHMARK_MEASURED_AT.fullmatch(measured_at):
        fail(f"{label} measuredAt is not a UTC Z timestamp")
    card_id = require_text(entry, "cardId", label)
    if not QUALITY_CARD_ID.fullmatch(card_id):
        fail(f"{label} has an invalid cardId")
    card = cards_by_id.get(card_id)
    if card is None:
        fail(f"{label} cardId {card_id!r} is not a published quality card")
    pack_label = require_text(entry, "packLabel", label)
    if len(pack_label) > 60:
        fail(f"{label} packLabel exceeds 60 characters")
    pack_revision = require_text(entry, "packRevision", label)
    if not COMMIT_SHA.fullmatch(pack_revision):
        fail(f"{label} packRevision is not a 40-hex revision")
    card_model = card.get("model") if isinstance(card.get("model"), dict) else {}
    if pack_revision != card_model.get("hfPin"):
        fail(f"{label} packRevision does not match its quality card's hfPin")
    hardware_class = require_text(entry, "hardwareClass", label)
    if (
        not QUALITY_CARD_HARDWARE_CLASS.fullmatch(hardware_class)
        or hardware_class == QUALITY_CARD_HARDWARE_CLASS_SENTINEL
    ):
        fail(f"{label} hardwareClass is invalid")
    card_config = card.get("config") if isinstance(card.get("config"), dict) else {}
    chip = require_text(entry, "chip", label)
    if (
        hardware_class != card_config.get("hardwareClass")
        or chip.lower().replace(" ", "-") != hardware_class
    ):
        fail(f"{label} hardwareClass does not match its quality card or chip")
    engine_build = require_exact_keys(
        entry.get("engineBuild"), SERVED_BENCHMARK_ENGINE_BUILD_KEYS, f"{label} engineBuild"
    )
    engine_commit = require_text(engine_build, "commit", f"{label} engineBuild")
    if not QUALITY_CARD_ENGINE_BUILD_COMMIT.fullmatch(engine_commit):
        fail(f"{label} engineBuild.commit is not a 40-hex sha")
    card_provenance = card.get("provenance") if isinstance(card.get("provenance"), dict) else {}
    card_engine_build = card_provenance.get("engineBuild")
    if not isinstance(card_engine_build, dict) or card_engine_build.get("commit") != engine_commit:
        fail(f"{label} engineBuild.commit does not match its quality card")
    harness = require_exact_keys(
        entry.get("harness"), SERVED_BENCHMARK_HARNESS_KEYS, f"{label} harness"
    )
    harness_commit = require_text(harness, "publicCommit", f"{label} harness")
    if not COMMIT_SHA.fullmatch(harness_commit):
        fail(f"{label} harness.publicCommit is not a 40-hex sha")
    combine_commit = require_text(harness, "combineCommit", f"{label} harness")
    if not COMMIT_SHA.fullmatch(combine_commit):
        fail(f"{label} harness.combineCommit is not a 40-hex sha")
    bench_sha = require_text(harness, "benchSha256", f"{label} harness")
    if not SERVED_BENCHMARK_SHA256.fullmatch(bench_sha):
        fail(f"{label} harness.benchSha256 is not a 64-hex sha256")
    prompt_sha = require_text(harness, "promptSetSha256", f"{label} harness")
    if not SERVED_BENCHMARK_SHA256.fullmatch(prompt_sha):
        fail(f"{label} harness.promptSetSha256 is not a 64-hex sha256")
    workload = require_exact_keys(
        entry.get("workload"), SERVED_BENCHMARK_WORKLOAD_KEYS, f"{label} workload"
    )
    if workload.get("promptSet") != "default-3-prompt-set":
        fail(f"{label} workload.promptSet must be default-3-prompt-set")
    for int_key in ("prompts", "maxTokens", "runs", "warmup"):
        value = workload.get(int_key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            fail(f"{label} workload.{int_key} must be a non-negative int")
    temperature = _served_benchmark_number(workload.get("temperature"))
    if temperature is None or temperature < 0:
        fail(f"{label} workload.temperature must be a non-negative number")
    completion_tokens = workload.get("completionTokens")
    if (
        not isinstance(completion_tokens, list)
        or len(completion_tokens) != 2
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in completion_tokens
        )
        or completion_tokens[0] > completion_tokens[1]
    ):
        fail(f"{label} workload.completionTokens must be a [min, max] pair")
    serving = require_exact_keys(
        entry.get("serving"), SERVED_BENCHMARK_SERVING_KEYS, f"{label} serving"
    )
    context_tokens = serving.get("contextTokens")
    if not isinstance(context_tokens, int) or isinstance(context_tokens, bool) or context_tokens <= 0:
        fail(f"{label} serving.contextTokens must be a positive int")
    for key in ("mtp", "drafter", "promptLookup"):
        if serving.get(key) not in SERVED_BENCHMARK_SERVING_TRISTATE:
            fail(f"{label} serving.{key} is outside the allowed values")
    result = require_exact_keys(
        entry.get("result"), SERVED_BENCHMARK_RESULT_KEYS, f"{label} result"
    )
    candidate_rate = _served_benchmark_number(result.get("candidateDecodeTokS"))
    reference_rate = _served_benchmark_number(result.get("referenceDecodeTokS"))
    ratio = _served_benchmark_number(result.get("ratio"))
    if candidate_rate is None or reference_rate is None or ratio is None:
        fail(f"{label} result rates and ratio must be numeric")
    if (
        candidate_rate < SERVED_BENCHMARK_RATE_FLOOR
        or candidate_rate > SERVED_BENCHMARK_RATE_CEILING
        or reference_rate < SERVED_BENCHMARK_RATE_FLOOR
        or reference_rate > SERVED_BENCHMARK_RATE_CEILING
    ):
        fail(f"{label} result rates are outside the plausible [25, 2000] tok/s range")
    for key in ("candidatePasses", "referencePasses"):
        value = result.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            fail(f"{label} result.{key} must be a positive int")
    drift_pct = _served_benchmark_number(result.get("referenceDriftPct"))
    if drift_pct is None or drift_pct < 0:
        fail(f"{label} result.referenceDriftPct must be a non-negative number")
    if drift_pct > SERVED_BENCHMARK_DRIFT_CEILING_PCT:
        fail(f"{label} result.referenceDriftPct exceeds the 5% ceiling")
    if reference_rate > 0 and abs(ratio - candidate_rate / reference_rate) > SERVED_BENCHMARK_RATIO_TOLERANCE:
        fail(f"{label} result.ratio is inconsistent with its own candidate/reference rates")
    card_benefit = (
        card.get("legible", {}).get("benefit", {}) if isinstance(card.get("legible"), dict) else {}
    )
    card_speed_x = _served_benchmark_number(card_benefit.get("speedX"))
    if card_speed_x is None or card_speed_x == 0 or abs(ratio / card_speed_x - 1) > SERVED_BENCHMARK_SPEEDX_TOLERANCE:
        fail(f"{label} result.ratio disagrees with its quality card's speedX by more than 3%")
    if entry.get("controls") != SERVED_BENCHMARK_CONTROLS:
        fail(f"{label} controls must be exactly the reviewed-slice-1 control set")
    row_sha = require_text(entry, "rowSha256", label)
    if not SERVED_BENCHMARK_SHA256.fullmatch(row_sha):
        fail(f"{label} rowSha256 is not a 64-hex sha256")
    rerun = require_exact_keys(entry.get("rerun"), SERVED_BENCHMARK_RERUN_KEYS, f"{label} rerun")
    rerun_failures: List[str] = []
    for key in sorted(SERVED_BENCHMARK_RERUN_KEYS):
        rerun_failures.extend(_served_benchmark_rerun_failures(key, rerun.get(key), label))
    if rerun_failures:
        fail(rerun_failures[0])
    non_rerun = {key: value for key, value in entry.items() if key != "rerun"}
    if "--" in json.dumps(non_rerun, ensure_ascii=False, sort_keys=True):
        fail(f"{label} carries a raw '--' flag token outside rerun")
    marker_reason = _served_benchmark_marker_violation(
        json.dumps(entry, ensure_ascii=False, sort_keys=True)
    )
    if marker_reason is not None:
        fail(f"{label} contains {marker_reason}")
    # Checked AFTER the marker scan above (deliberately last): a packLabel
    # that carries a private path, an engine name, the fleet host
    # shorthand, an IPv4 address, or a `<path>` placeholder is refused by
    # that marker-specific reason first; only a packLabel that is clean of
    # every marker but still does not look like a pack label reaches this
    # allowlist.
    if not SERVED_BENCHMARK_PACK_LABEL.fullmatch(pack_label):
        fail(f"{label} packLabel does not look like a pack label")
    return entry


def _served_benchmark_row_mismatches(
    entry: Dict[str, object], derived: Dict[str, object], label: str
) -> List[str]:
    """Every field `derive_served_benchmark_fields_from_ledger_row` returns,
    checked against the entry's own nested copy of that same field. Returns
    an accumulate-style list (never raises itself) so the caller can report
    the first one via `fail()`, exactly like every other check in this
    module's fail-on-first-violation style.
    """

    result = entry.get("result", {})
    workload = entry.get("workload", {})
    serving = entry.get("serving", {})
    harness = entry.get("harness", {})
    expected: Dict[str, object] = {
        "candidateDecodeTokS": result.get("candidateDecodeTokS"),
        "referenceDecodeTokS": result.get("referenceDecodeTokS"),
        "ratio": result.get("ratio"),
        "candidatePasses": result.get("candidatePasses"),
        "referencePasses": result.get("referencePasses"),
        "referenceDriftPct": result.get("referenceDriftPct"),
        "completionTokens": workload.get("completionTokens"),
        "contextTokens": serving.get("contextTokens"),
        "mtp": serving.get("mtp"),
        "drafter": serving.get("drafter"),
        "promptLookup": serving.get("promptLookup"),
        "benchSha256": harness.get("benchSha256"),
        "promptSetSha256": harness.get("promptSetSha256"),
        "chip": entry.get("chip"),
        "prompts": workload.get("prompts"),
        "maxTokens": workload.get("maxTokens"),
        "runs": workload.get("runs"),
        "warmup": workload.get("warmup"),
        "temperature": workload.get("temperature"),
    }
    failures: List[str] = []
    for key in sorted(expected):
        if derived.get(key) != expected[key]:
            failures.append(
                f"{label} row file's derived {key} ({derived.get(key)!r}) disagrees with "
                f"the entry's own {key} ({expected[key]!r})"
            )
    return failures


def validate_served_benchmark_row_file(
    repository_root: Path, entry: Dict[str, object], label: str
) -> bytes:
    """Load, verify, and return the raw bytes of the published ledger row
    file (`site/served-benchmark-rows/<entry id>.json`) that `entry["rowSha256"]`
    is supposed to hash -- the check that makes the "sealed row" claim
    actually verifiable by a public reader (see the module's served-
    benchmark-ledger design notes, item 1). Fails closed on the FIRST
    violation, same style as every other loader check in this module.
    """

    row_path = repository_root / "site" / SERVED_BENCHMARK_ROWS_DIRNAME / f"{entry['id']}.json"
    if row_path.is_symlink() or not row_path.is_file():
        fail(f"{label} row file is missing, not a regular file, or a symlink")
    raw = row_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry.get("rowSha256"):
        fail(f"{label} row file sha256 does not match its entry's rowSha256")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        fail(f"{label} row file is not UTF-8: {exc}")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        fail(f"{label} row file is not valid JSON: {exc}")
    if not isinstance(parsed, dict):
        fail(f"{label} row file is not a JSON object")
    canonical = json.dumps(parsed, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if canonical != text:
        fail(f"{label} row file is not canonically serialized")
    violation = served_benchmark_ledger_row_violation(text, parsed)
    if violation is not None:
        fail(f"{label} row file contains {violation}")
    derived = derive_served_benchmark_fields_from_ledger_row(parsed)
    mismatches = _served_benchmark_row_mismatches(entry, derived, label)
    if mismatches:
        fail(mismatches[0])
    return raw


def _check_served_benchmark_row_orphans(repository_root: Path, entry_ids: Iterable[str]) -> None:
    """Refuse any file under `site/served-benchmark-rows/` that no entry
    names (an orphan row nobody's `rowSha256` points at), and refuse a
    subdirectory. Only ever reached once every entry's own row file has
    already been individually verified (see `validate_served_benchmark_row_file`),
    so the directory is guaranteed to exist by this point.
    """

    rows_dir = repository_root / "site" / SERVED_BENCHMARK_ROWS_DIRNAME
    if rows_dir.is_symlink() or not rows_dir.is_dir():
        fail(f"site/{SERVED_BENCHMARK_ROWS_DIRNAME} is missing, not a directory, or a symlink")
    expected_names = {f"{identifier}.json" for identifier in entry_ids}
    for path in sorted(rows_dir.iterdir()):
        if path.is_symlink():
            fail(f"site/{SERVED_BENCHMARK_ROWS_DIRNAME}/{path.name} is a symlink")
        if path.is_dir():
            fail(f"site/{SERVED_BENCHMARK_ROWS_DIRNAME}/{path.name} is a subdirectory")
        if path.name not in expected_names:
            fail(f"site/{SERVED_BENCHMARK_ROWS_DIRNAME}/{path.name} is an orphan row file")


def load_served_benchmarks(
    repository_root: Path, quality_manifest: Optional[Dict[str, object]]
) -> Dict[str, object]:
    """Load the strict, sealed served-engine benchmark ledger
    (`site/served-benchmarks.json`). Required like the release catalog --
    never silently skipped -- because a ledger entry is a load-bearing claim
    about a specific published quality card (see the module-level comment
    above `SERVED_BENCHMARK_TOP_KEYS`). `quality_manifest` is the ALREADY-
    validated return of `load_quality_guides`; its absence (`None`) is
    itself a refusal (`L2`), since an entry can never be cross-checked
    against a card that does not exist.
    """

    if quality_manifest is None:
        fail("served ledger requires a quality-card manifest")
    manifest_path = repository_root / "site/served-benchmarks.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        fail("site/served-benchmarks.json is missing, not a file, or a symlink")
    catalog = require_exact_keys(
        read_json(manifest_path), SERVED_BENCHMARK_TOP_KEYS, "site/served-benchmarks.json"
    )
    if catalog.get("schemaVersion") != 1:
        fail("site/served-benchmarks.json must use schemaVersion 1")
    if catalog.get("project") != "fast-mlx":
        fail("served ledger project must remain fast-mlx")
    if catalog.get("policy") != SERVED_BENCHMARK_POLICY:
        fail(f"served ledger policy must remain {SERVED_BENCHMARK_POLICY}")
    if catalog.get("claimBoundary") != SERVED_BENCHMARK_CLAIM_BOUNDARY:
        fail(f"served ledger claim boundary must remain {SERVED_BENCHMARK_CLAIM_BOUNDARY}")
    require_iso_date(catalog, "updatedAt", "site/served-benchmarks.json")
    raw_entries = catalog.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        fail("served ledger must contain at least one entry")
    cards = quality_manifest.get("cards") if isinstance(quality_manifest, dict) else None
    cards_by_id = {
        str(card["id"]): card for card in cards if isinstance(card, dict) and "id" in card
    } if isinstance(cards, list) else {}
    entries: List[Dict[str, object]] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    previous_measured_at: Optional[str] = None
    for index, raw_entry in enumerate(raw_entries):
        label = f"served benchmark entry {index}"
        entry = validate_served_benchmark_entry(raw_entry, cards_by_id, label)
        identifier = str(entry["id"])
        row_sha = str(entry["rowSha256"])
        if identifier in seen_ids:
            fail("served ledger has a duplicate entry id")
        if row_sha in seen_hashes:
            fail("served ledger has a duplicate entry rowSha256")
        seen_ids.add(identifier)
        seen_hashes.add(row_sha)
        measured_at = str(entry["measuredAt"])
        if previous_measured_at is not None and measured_at >= previous_measured_at:
            fail("served ledger entries are not newest-first")
        previous_measured_at = measured_at
        validate_served_benchmark_row_file(repository_root, entry, label)
        entries.append(entry)
    _check_served_benchmark_row_orphans(repository_root, seen_ids)
    catalog["entries"] = entries
    return catalog


def load_articles(repository_root: Path) -> List[Article]:
    manifest_path = repository_root / "site/publications.json"
    manifest = require_exact_keys(
        read_bounded_regular_json(
            manifest_path,
            "site/publications.json",
            MAX_PUBLICATION_MANIFEST_BYTES,
        ),
        {"schemaVersion", "policy", "articles"},
        "site/publications.json",
    )
    if manifest.get("schemaVersion") != 1:
        fail("site/publications.json must use schemaVersion 1")
    if manifest.get("policy") != "fast-mlx-owned-results-only":
        fail("publication policy must remain fast-mlx-owned-results-only")
    entries = manifest.get("articles")
    if not isinstance(entries, list) or not entries:
        fail("publication manifest must contain at least one article")

    seen_sources: set[str] = set()
    seen_slugs: set[str] = set()
    articles: List[Article] = []
    for index, raw_entry in enumerate(entries):
        entry = require_exact_keys(
            raw_entry,
            {"source", "slug", "status", "reviewedAt"},
            f"article entry {index}",
        )
        source_name = entry.get("source")
        slug = entry.get("slug")
        reviewed_at = entry.get("reviewedAt")
        if entry.get("status") != "published":
            fail(f"article entry {index} is not explicitly published")
        if not isinstance(source_name, str):
            fail(f"article entry {index} source is outside docs/content")
        source_path = Path(source_name)
        if (
            source_path.is_absolute()
            or ".." in source_path.parts
            or source_path.parent != Path("docs/content")
            or source_path.suffix != ".md"
        ):
            fail(f"article entry {index} source is outside docs/content")
        if not isinstance(slug, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug):
            fail(f"article entry {index} has an invalid slug")
        if not isinstance(reviewed_at, str):
            fail(f"article entry {index} is missing reviewedAt")
        try:
            dt.date.fromisoformat(reviewed_at)
        except ValueError:
            fail(f"article entry {index} reviewedAt is not an ISO date")
        if source_name in seen_sources or slug in seen_slugs:
            fail(f"duplicate source or slug in publication manifest: {source_name}")
        seen_sources.add(source_name)
        seen_slugs.add(slug)

        source = repository_root / source_name
        if not source.is_file() or source.is_symlink():
            fail(f"published source is missing, not a file, or a symlink: {source_name}")
        if source.resolve().parent != (repository_root / "docs/content").resolve():
            fail(f"published source escapes docs/content: {source_name}")
        text = source.read_text(encoding="utf-8")
        for marker in PRIVATE_MARKERS:
            if marker.casefold() in text.casefold():
                fail(f"published source contains private marker {marker!r}: {source_name}")
        title, date, theme, summary, body = infer_metadata(source, text)
        articles.append(
            Article(
                source=source,
                source_name=source_name,
                slug=slug,
                title=title,
                date=date,
                theme=theme,
                summary=summary,
                reviewed_at=reviewed_at,
                body=body,
            )
        )
    return sorted(articles, key=lambda article: (article.date, article.slug), reverse=True)


def relative_href(current_file: str, target_path: str) -> str:
    current_dir = posixpath.dirname(current_file)
    target = target_path.rstrip("/") or "."
    value = posixpath.relpath(target, current_dir or ".")
    if value == ".":
        value = "./"
    elif target_path.endswith("/"):
        value += "/"
    return value


class InlineRenderer:
    def __init__(self, article_routes: Dict[str, str], current_file: str) -> None:
        self.article_routes = article_routes
        self.current_file = current_file
        self.tokens: Dict[str, str] = {}

    def stash(self, value: str) -> str:
        token = f"@@FASTMLX{len(self.tokens)}TOKEN@@"
        self.tokens[token] = value
        return token

    def render_link(self, match: re.Match[str]) -> str:
        label = match.group(1)
        raw_url = html.unescape(match.group(2)).strip()
        if raw_url.startswith(("https://", "http://", "mailto:")):
            href = html.escape(raw_url, quote=True)
            return self.stash(f'<a href="{href}" rel="noreferrer">{label}</a>')
        if raw_url.startswith("#"):
            href = html.escape(raw_url, quote=True)
            return self.stash(f'<a href="{href}">{label}</a>')

        path, separator, fragment = raw_url.partition("#")
        route = self.article_routes.get(posixpath.basename(path))
        if route:
            href = relative_href(self.current_file, route)
            if separator:
                href += "#" + fragment
            return self.stash(f'<a href="{html.escape(href, quote=True)}">{label}</a>')

        title = html.escape(
            "Supporting project evidence is retained outside the public projection", quote=True
        )
        return self.stash(f'<span class="source-boundary" title="{title}">{label}</span>')

    def __call__(self, text: str) -> str:
        escaped = html.escape(text, quote=False)
        escaped = CODE_SPAN.sub(
            lambda match: self.stash(f"<code>{match.group(1)}</code>"), escaped
        )
        escaped = LINK.sub(self.render_link, escaped)
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
        escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)
        for token, value in self.tokens.items():
            escaped = escaped.replace(token, value)
        return escaped


def table_cells(line: str) -> List[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def is_table(lines: Sequence[str], index: int) -> bool:
    if index + 1 >= len(lines) or "|" not in lines[index]:
        return False
    separators = table_cells(lines[index + 1])
    return bool(separators) and all(TABLE_SEPARATOR.fullmatch(cell) for cell in separators)


def heading_id(value: str) -> str:
    lowered = plain_text(value).casefold()
    return re.sub(r"[^a-z0-9]+", "-", lowered).strip("-") or "section"


def render_markdown(body: str, article_routes: Dict[str, str], current_file: str) -> str:
    lines = body.splitlines()
    theme_block = find_whitepaper_theme_block(lines)
    inline = InlineRenderer(article_routes, current_file)
    output: List[str] = []
    paragraph: List[str] = []
    title_skipped = False

    def flush_paragraph() -> None:
        if paragraph:
            output.append(f"<p>{inline(' '.join(item.strip() for item in paragraph))}</p>")
            paragraph.clear()

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if theme_block and index == theme_block[0]:
            flush_paragraph()
            output.append(
                f'<p class="eyebrow">{inline("**Whitepaper themes:** " + theme_block[2])}</p>'
            )
            index = theme_block[1]
            continue
        if not stripped:
            flush_paragraph()
            index += 1
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            language = re.sub(r"[^a-zA-Z0-9_+-]", "", stripped[3:].strip())
            code: List[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            if index >= len(lines):
                fail("unterminated fenced code block")
            language_class = f' class="language-{language}"' if language else ""
            output.append(
                f"<pre><code{language_class}>{html.escape(chr(10).join(code))}</code></pre>"
            )
            index += 1
            continue

        heading = HEADING.match(stripped)
        if heading:
            flush_paragraph()
            if len(heading.group(1)) == 1 and not title_skipped:
                title_skipped = True
                index += 1
                continue
            level = min(6, max(2, len(heading.group(1))))
            title = heading.group(2)
            output.append(
                f'<h{level} id="{heading_id(title)}">{inline(title)}</h{level}>'
            )
            index += 1
            continue

        if is_table(lines, index):
            flush_paragraph()
            headers = table_cells(lines[index])
            index += 2
            rows: List[List[str]] = []
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                rows.append(table_cells(lines[index]))
                index += 1
            output.append("<div class=\"table-scroll\"><table><thead><tr>")
            output.extend(f"<th>{inline(cell)}</th>" for cell in headers)
            output.append("</tr></thead><tbody>")
            for row in rows:
                padded = row + [""] * (len(headers) - len(row))
                output.append("<tr>")
                output.extend(f"<td>{inline(cell)}</td>" for cell in padded[: len(headers)])
                output.append("</tr>")
            output.append("</tbody></table></div>")
            continue

        if stripped.startswith(">"):
            flush_paragraph()
            quote: List[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote.append(lines[index].strip().lstrip("> "))
                index += 1
            output.append(f"<blockquote><p>{inline(' '.join(quote))}</p></blockquote>")
            continue

        list_match = LIST_ITEM.match(stripped)
        if list_match:
            flush_paragraph()
            ordered = ORDERED_ITEM.match(stripped) is not None
            tag = "ol" if ordered else "ul"
            items: List[str] = []
            while index < len(lines):
                candidate = lines[index].strip()
                match = ORDERED_ITEM.match(candidate) if ordered else re.match(r"^[-+*]\s+(.+)$", candidate)
                if not match:
                    break
                items.append(match.group(1))
                index += 1
            output.append(f"<{tag}>")
            output.extend(f"<li>{inline(item)}</li>" for item in items)
            output.append(f"</{tag}>")
            continue

        if stripped in {"---", "***", "___"}:
            flush_paragraph()
            output.append("<hr>")
            index += 1
            continue

        paragraph.append(stripped)
        index += 1

    flush_paragraph()
    return "\n".join(output)


def render_template(
    template: str,
    title: str,
    description: str,
    root: str,
    body: str,
    current_page: str = "",
    page_script: str = "",
    *,
    public_path: Optional[str] = None,
    article_section: Optional[str] = None,
    quality_nav_available: bool = False,
) -> str:
    nav_root = html.escape(root, quote=True)

    def nav_link(page: str, label: str) -> str:
        current = ' aria-current="page"' if current_page == page else ""
        return f'<a href="{nav_root}{page}/"{current}>{label}</a>'

    replacements = {
        "{{title}}": html.escape(title),
        "{{description}}": html.escape(description, quote=True),
        "{{head_metadata}}": render_head_metadata(
            title,
            description,
            public_path,
            article_section=article_section,
        ),
        "{{root}}": root,
        "{{body}}": body.replace("{{root}}", root),
        "{{quickstart_nav}}": nav_link("quickstart", "Quickstart"),
        "{{status_nav}}": nav_link("status", "Status"),
        "{{capabilities_nav}}": nav_link("capabilities", "Capabilities"),
        "{{benchmarks_nav}}": nav_link("benchmarks", "Benchmarks"),
        "{{releases_nav}}": nav_link("releases", "Releases"),
        "{{process_nav}}": nav_link("process", "The loop"),
        "{{methodology_nav}}": nav_link("methodology", "Methodology"),
        "{{research_nav}}": nav_link("research", "Research notes"),
        "{{quality_nav}}": nav_link("quality", "Quality guide") if quality_nav_available else "",
        "{{page_script}}": page_script,
    }
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    if re.search(r"{{[a-zA-Z0-9_]+}}", rendered):
        fail(f"template contains an unresolved token for {title}")
    return rendered


def render_head_metadata(
    title: str,
    description: str,
    public_path: Optional[str],
    *,
    article_section: Optional[str],
) -> str:
    if public_path is None:
        if article_section is not None:
            fail("article metadata requires a reviewed public path")
        return ""
    if (
        public_path not in CORE_PUBLIC_PAGE_PATHS
        and public_path != "quality/"
        and not re.fullmatch(
            r"research/[a-z0-9]+(?:-[a-z0-9]+)*/", public_path
        )
        and not CAPABILITY_DETAIL_PUBLIC_PATH.fullmatch(public_path)
        and public_path not in REVIEWED_BENCHMARK_PUBLIC_PATHS
        and not RELEASE_DETAIL_PUBLIC_PATH.fullmatch(public_path)
    ):
        fail(f"invalid metadata public path: {public_path!r}")
    canonical = PUBLIC_SITE_URL + public_path
    page_type = "article" if article_section is not None else "website"
    tags = [
        f'<link rel="canonical" href="{html.escape(canonical, quote=True)}">',
        f'<meta property="og:title" content="{html.escape(title, quote=True)}">',
        f'<meta property="og:type" content="{page_type}">',
        f'<meta property="og:image" content="{SOCIAL_CARD_URL}">',
        f'<meta property="og:image:width" content="{SOCIAL_CARD_WIDTH}">',
        f'<meta property="og:image:height" content="{SOCIAL_CARD_HEIGHT}">',
        f'<meta property="og:image:alt" content="{html.escape(SOCIAL_CARD_ALT, quote=True)}">',
        f'<meta property="og:url" content="{html.escape(canonical, quote=True)}">',
        f'<meta property="og:description" content="{html.escape(description, quote=True)}">',
        '<meta property="og:site_name" content="fast-mlx">',
    ]
    if public_path == "research/" or re.fullmatch(
        r"research/[a-z0-9]+(?:-[a-z0-9]+)*/", public_path
    ):
        current_file = public_path + "index.html"
        research_feed_href = relative_href(current_file, "research/feed.atom")
        tags.append(
            '<link rel="alternate" type="application/atom+xml" '
            'title="fast-mlx reviewed research" '
            f'href="{html.escape(research_feed_href, quote=True)}">'
        )
    if public_path == "":
        tags.append(
            '<link rel="alternate" type="application/atom+xml" '
            'title="fast-mlx reviewed updates" href="feed.atom">'
        )
    if article_section is not None:
        tags.append(
            f'<meta property="article:section" content="{html.escape(article_section, quote=True)}">'
        )
    return "\n    ".join(tags)


def write_page(output: Path, relative_file: str, contents: str) -> None:
    destination = output / relative_file
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(contents.rstrip() + "\n", encoding="utf-8")


def render_release_catalog(
    catalog: Dict[str, object]
) -> Tuple[str, Dict[str, object]]:
    boundary = dict(catalog["currentBoundary"])
    boundary["evidence"] = dict(boundary["evidence"])
    releases: List[Dict[str, object]] = []
    for raw_entry in catalog["releases"]:
        entry = dict(raw_entry)
        entry["publicLinks"] = [dict(link) for link in entry["publicLinks"]]
        entry["sourceUrl"] = (
            "https://github.com/bitworks-io/fast-mlx/commit/"
            f"{entry['publicCommit']}"
        )
        releases.append(entry)

    public_index: Dict[str, object] = {
        "schemaVersion": 1,
        "project": "fast-mlx",
        "policy": catalog["policy"],
        "claimBoundary": catalog["claimBoundary"],
        "updatedAt": catalog["updatedAt"],
        "currentBoundary": boundary,
        "releases": releases,
    }

    boundary_evidence = boundary["evidence"]
    boundary_href = relative_href(
        "releases/index.html", str(boundary_evidence["path"])
    )
    body: List[str] = [
        '<section class="page-hero shell release-hero">',
        '<p class="eyebrow">Reviewed releases</p>',
        '<h1>See what shipped—and what it still does not prove.</h1>',
        '<p class="lede">A generated, newest-first ledger of fast-mlx public milestones. Every entry names the exact public commit, the user-facing surface, and the boundary that stayed in force.</p>',
        '<div class="hero-actions">',
        '<a class="button primary" href="index.json">Read the release JSON</a>',
        '<a class="button secondary" href="feed.atom" type="application/atom+xml">Subscribe to reviewed releases</a>',
        '<a class="button secondary" href="../feed.atom" type="application/atom+xml">Subscribe to all reviewed updates</a>',
        '<a class="button secondary" href="https://github.com/bitworks-io/fast-mlx/commits/main" rel="noreferrer">Inspect public history</a>',
        '</div>',
        '</section>',
        '<section class="section shell release-boundary" aria-labelledby="release-boundary-heading" data-release-boundary '
        f'data-boundary-id="{html.escape(str(boundary["id"]), quote=True)}" '
        f'data-boundary-state="{html.escape(str(boundary["state"]), quote=True)}">',
        '<div class="section-heading">',
        '<p class="eyebrow">Current boundary</p>',
        f'<span class="status-badge status-gated">{html.escape(str(boundary["state"]).upper())}</span>',
        f'<h2 id="release-boundary-heading">{html.escape(str(boundary["label"]))}</h2>',
        f'<p class="section-intro">{html.escape(str(boundary["summary"]))}</p>',
        f'<a class="text-link" href="{html.escape(boundary_href, quote=True)}">{html.escape(str(boundary_evidence["label"]))} →</a>',
        '</div>',
        '</section>',
        '<section class="section shell" aria-labelledby="release-ledger-heading">',
        '<div class="section-heading"><p class="eyebrow">Public history</p><h2 id="release-ledger-heading">One reviewed commit at a time.</h2><p class="section-intro">Operational hardening and product surfaces appear together because both determine what users can trust.</p></div>',
        '<ol class="release-list">',
    ]
    for release in releases:
        category = str(release["category"])
        commit = str(release["publicCommit"])
        published_at = str(release["publishedAt"])
        body.extend(
            [
                '<li>',
                '<article class="release-card" '
                f'id="release-{html.escape(str(release["id"]), quote=True)}" '
                f'data-release-id="{html.escape(str(release["id"]), quote=True)}" '
                f'data-public-commit="{html.escape(commit, quote=True)}">',
                '<div class="card-topline">',
                f'<span class="status-badge status-released">{html.escape(str(release["state"]).upper())}</span>',
                f'<time datetime="{html.escape(published_at, quote=True)}">{html.escape(published_at[:10])}</time>',
                '</div>',
                f'<p class="release-category">{html.escape(RELEASE_CATEGORY_LABELS[category])}</p>',
                f'<h3>{html.escape(str(release["title"]))}</h3>',
                f'<p>{html.escape(str(release["summary"]))}</p>',
                f'<p class="scope-note"><strong>Boundary:</strong> {html.escape(str(release["scope"]))}</p>',
                '<div class="release-links">',
                '<a class="text-link" href="'
                + html.escape(str(release["id"]), quote=True)
                + '/">Open reviewed checkpoint →</a>',
                f'<a class="text-link" href="{html.escape(str(release["sourceUrl"]), quote=True)}" rel="noreferrer">Inspect commit {html.escape(commit[:12])} →</a>',
            ]
        )
        for link in release["publicLinks"]:
            href = relative_href("releases/index.html", str(link["path"]))
            body.append(
                f'<a href="{html.escape(href, quote=True)}">{html.escape(str(link["label"]))} →</a>'
            )
        body.extend(['</div>', '</article>', '</li>'])
    body.extend(['</ol>', '</section>'])
    return "\n".join(body), public_index


def release_detail_title(release: Dict[str, object]) -> str:
    return f'{release["title"]} — fast-mlx release'


def render_release_detail(release: Dict[str, object]) -> str:
    """Render one immutable view of a reviewed release-ledger entry."""

    identifier = str(release["id"])
    commit = str(release["publicCommit"])
    published_at = str(release["publishedAt"])
    category = str(release["category"])
    public_path = f"releases/{identifier}/"
    body = [
        '<section class="page-hero shell benchmark-detail release-detail" '
        'data-release-detail '
        f'data-release-id="{html.escape(identifier, quote=True)}" '
        f'data-public-commit="{html.escape(commit, quote=True)}">',
        '<div class="card-topline">',
        f'<span class="status-badge status-released">{html.escape(str(release["state"]).upper())}</span>',
        f'<time datetime="{html.escape(published_at, quote=True)}">{html.escape(published_at[:10])}</time>',
        '</div>',
        f'<p class="release-category">{html.escape(RELEASE_CATEGORY_LABELS[category])}</p>',
        f'<h1>{html.escape(str(release["title"]))}</h1>',
        f'<p class="lede">{html.escape(str(release["summary"]))}</p>',
        '<p class="scope-note"><strong>Boundary:</strong> '
        + html.escape(str(release["scope"]))
        + '</p>',
        '<div class="release-links">',
        '<a class="button primary" href="https://github.com/bitworks-io/fast-mlx/commit/'
        + html.escape(commit, quote=True)
        + '" rel="noreferrer">Inspect commit '
        + html.escape(commit[:12])
        + ' →</a>',
    ]
    for link in release["publicLinks"]:
        href = relative_href(public_path + "index.html", str(link["path"]))
        body.append(
            f'<a href="{html.escape(href, quote=True)}">{html.escape(str(link["label"]))} →</a>'
        )
    body.extend(
        [
            '<a class="button secondary" href="../">Back to all reviewed releases</a>',
            '</div>',
            '<div class="scope-note">',
            '<strong>Static release boundary.</strong> This page does not create a new release, measurement, ranking, runtime, model, acquisition, or publication authority.',
            ' <a href="../../methodology/">Read the public methodology →</a>',
            '</div>',
            '</section>',
        ]
    )
    return "\n".join(body)


def render_release_feed(release_index: Dict[str, object]) -> str:
    """Render a deterministic Atom 1.0 feed from the reviewed public release index."""

    ET.register_namespace("", ATOM_NAMESPACE)
    atom = lambda name: f"{{{ATOM_NAMESPACE}}}{name}"
    feed = ET.Element(atom("feed"))
    ET.SubElement(feed, atom("title")).text = "fast-mlx reviewed releases"
    ET.SubElement(feed, atom("id")).text = PUBLIC_SITE_URL + "releases/"
    releases = release_index["releases"]
    ET.SubElement(feed, atom("updated")).text = str(releases[0]["publishedAt"])
    author = ET.SubElement(feed, atom("author"))
    ET.SubElement(author, atom("name")).text = "fast-mlx contributors"
    ET.SubElement(
        feed,
        atom("link"),
        {
            "rel": "self",
            "type": "application/atom+xml",
            "href": PUBLIC_SITE_URL + "releases/feed.atom",
        },
    )
    ET.SubElement(
        feed,
        atom("link"),
        {
            "rel": "alternate",
            "type": "text/html",
            "href": PUBLIC_SITE_URL + "releases/",
        },
    )

    for release in releases:
        entry = ET.SubElement(feed, atom("entry"))
        ET.SubElement(entry, atom("title")).text = str(release["title"])
        ET.SubElement(entry, atom("id")).text = (
            "urn:fast-mlx:public-commit:" + str(release["publicCommit"])
        )
        ET.SubElement(entry, atom("published")).text = str(release["publishedAt"])
        ET.SubElement(entry, atom("updated")).text = str(release["publishedAt"])
        ET.SubElement(
            entry, atom("category"), {"term": str(release["category"])}
        )
        ET.SubElement(
            entry,
            atom("link"),
            {
                "rel": "alternate",
                "type": "text/html",
                "href": PUBLIC_SITE_URL + "releases/" + str(release["id"]) + "/",
            },
        )
        ET.SubElement(
            entry,
            atom("link"),
            {"rel": "via", "href": str(release["sourceUrl"])},
        )
        ET.SubElement(entry, atom("summary")).text = (
            str(release["summary"]) + " Boundary: " + str(release["scope"])
        )

    ET.indent(feed, space="  ")
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        + ET.tostring(feed, encoding="unicode", short_empty_elements=True)
    )


def render_research_feed(articles: Sequence[Article]) -> str:
    """Render reviewed research metadata as a deterministic text-only Atom feed."""

    ordered_articles = sorted(
        articles,
        key=lambda article: (article.reviewed_at, article.date, article.slug),
        reverse=True,
    )
    if not ordered_articles:
        fail("research feed requires at least one reviewed article")

    ET.register_namespace("", ATOM_NAMESPACE)
    atom = lambda name: f"{{{ATOM_NAMESPACE}}}{name}"
    feed = ET.Element(atom("feed"))
    ET.SubElement(feed, atom("title")).text = "fast-mlx reviewed research"
    ET.SubElement(feed, atom("id")).text = PUBLIC_SITE_URL + "research/"
    ET.SubElement(feed, atom("updated")).text = (
        ordered_articles[0].reviewed_at + "T00:00:00Z"
    )
    author = ET.SubElement(feed, atom("author"))
    ET.SubElement(author, atom("name")).text = "fast-mlx contributors"
    ET.SubElement(
        feed,
        atom("link"),
        {
            "rel": "self",
            "type": "application/atom+xml",
            "href": PUBLIC_SITE_URL + "research/feed.atom",
        },
    )
    ET.SubElement(
        feed,
        atom("link"),
        {
            "rel": "alternate",
            "type": "text/html",
            "href": PUBLIC_SITE_URL + "research/",
        },
    )

    for article in ordered_articles:
        canonical = PUBLIC_SITE_URL + article.public_path
        entry = ET.SubElement(feed, atom("entry"))
        ET.SubElement(entry, atom("title")).text = article.title
        ET.SubElement(entry, atom("id")).text = canonical
        ET.SubElement(entry, atom("published")).text = (
            article.date + "T00:00:00Z"
        )
        ET.SubElement(entry, atom("updated")).text = (
            article.reviewed_at + "T00:00:00Z"
        )
        ET.SubElement(entry, atom("category"), {"term": article.theme})
        ET.SubElement(
            entry,
            atom("link"),
            {
                "rel": "alternate",
                "type": "text/html",
                "href": canonical,
            },
        )
        ET.SubElement(entry, atom("summary")).text = article.summary

    ET.indent(feed, space="  ")
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        + ET.tostring(feed, encoding="unicode", short_empty_elements=True)
    )


def parse_atom_timestamp(value: str) -> dt.datetime:
    """Parse an already-reviewed RFC 3339 timestamp for deterministic ordering."""

    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        fail(f"invalid reviewed Atom timestamp: {value!r}")
    if parsed.tzinfo is None:
        fail(f"reviewed Atom timestamp is not timezone-aware: {value!r}")
    return parsed


def render_reviewed_updates_feed(
    release_index: Dict[str, object], articles: Sequence[Article]
) -> str:
    """Render one text-only Atom feed from the two reviewed public manifests."""

    updates: List[Dict[str, str]] = []
    for release in release_index["releases"]:
        published_at = str(release["publishedAt"])
        updates.append(
            {
                "kind": "release",
                "title": str(release["title"]),
                "id": "urn:fast-mlx:public-commit:"
                + str(release["publicCommit"]),
                "published": published_at,
                "updated": published_at,
                "href": PUBLIC_SITE_URL
                + "releases/"
                + str(release["id"])
                + "/",
                "via": str(release["sourceUrl"]),
                "summary": str(release["summary"])
                + " Boundary: "
                + str(release["scope"]),
            }
        )
    for article in articles:
        canonical = PUBLIC_SITE_URL + article.public_path
        updates.append(
            {
                "kind": "research",
                "title": article.title,
                "id": canonical,
                "published": article.date + "T00:00:00Z",
                "updated": article.reviewed_at + "T00:00:00Z",
                "href": canonical,
                "via": "",
                "summary": article.summary,
            }
        )
    if not updates:
        fail("reviewed updates feed requires at least one reviewed entry")
    identifiers = [update["id"] for update in updates]
    if len(identifiers) != len(set(identifiers)):
        fail("reviewed updates feed entry IDs are not unique")
    ordered_updates = sorted(
        updates,
        key=lambda update: (
            parse_atom_timestamp(update["updated"]),
            update["id"],
        ),
        reverse=True,
    )

    ET.register_namespace("", ATOM_NAMESPACE)
    atom = lambda name: f"{{{ATOM_NAMESPACE}}}{name}"
    feed = ET.Element(atom("feed"))
    ET.SubElement(feed, atom("title")).text = "fast-mlx reviewed updates"
    ET.SubElement(feed, atom("id")).text = PUBLIC_SITE_URL
    ET.SubElement(feed, atom("updated")).text = ordered_updates[0]["updated"]
    author = ET.SubElement(feed, atom("author"))
    ET.SubElement(author, atom("name")).text = "fast-mlx contributors"
    ET.SubElement(
        feed,
        atom("link"),
        {
            "rel": "self",
            "type": "application/atom+xml",
            "href": PUBLIC_SITE_URL + "feed.atom",
        },
    )
    ET.SubElement(
        feed,
        atom("link"),
        {
            "rel": "alternate",
            "type": "text/html",
            "href": PUBLIC_SITE_URL,
        },
    )
    for update in ordered_updates:
        entry = ET.SubElement(feed, atom("entry"))
        ET.SubElement(entry, atom("title")).text = update["title"]
        ET.SubElement(entry, atom("id")).text = update["id"]
        ET.SubElement(entry, atom("published")).text = update["published"]
        ET.SubElement(entry, atom("updated")).text = update["updated"]
        ET.SubElement(entry, atom("category"), {"term": update["kind"]})
        ET.SubElement(
            entry,
            atom("link"),
            {
                "rel": "alternate",
                "type": "text/html",
                "href": update["href"],
            },
        )
        if update["via"]:
            ET.SubElement(
                entry,
                atom("link"),
                {"rel": "via", "href": update["via"]},
            )
        ET.SubElement(entry, atom("summary")).text = update["summary"]

    ET.indent(feed, space="  ")
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        + ET.tostring(feed, encoding="unicode", short_empty_elements=True)
    )


def render_sitemap(
    articles: Sequence[Article],
    capabilities: Sequence[Dict[str, object]],
    highlights: Sequence[Dict[str, object]],
    releases: Sequence[Dict[str, object]],
) -> str:
    """Render canonical human-facing routes without inventing crawl metadata."""

    ET.register_namespace("", SITEMAP_NAMESPACE)
    sitemap = ET.Element(f"{{{SITEMAP_NAMESPACE}}}urlset")
    public_paths = [
        *CORE_PUBLIC_PAGE_PATHS,
        *[f'capabilities/{capability["id"]}/' for capability in capabilities],
        *[f'benchmarks/{highlight["id"]}/' for highlight in highlights],
        *[f'releases/{release["id"]}/' for release in releases],
        *[article.public_path for article in articles],
    ]
    if len(public_paths) != len(set(public_paths)):
        fail("reviewed sitemap routes are not unique")
    for public_path in public_paths:
        url = ET.SubElement(sitemap, f"{{{SITEMAP_NAMESPACE}}}url")
        ET.SubElement(url, f"{{{SITEMAP_NAMESPACE}}}loc").text = (
            PUBLIC_SITE_URL + public_path
        )
    ET.indent(sitemap, space="  ")
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        + ET.tostring(sitemap, encoding="unicode", short_empty_elements=True)
    )


def render_robots() -> str:
    """Render the exact public crawl policy and canonical sitemap locator."""

    return (
        "User-agent: *\n"
        "Allow: /\n"
        f"Sitemap: {PUBLIC_SITE_URL}sitemap.xml"
    )


def render_capability_catalog(
    catalog: Dict[str, object], articles: Sequence[Article]
) -> Tuple[str, Dict[str, object]]:
    article_by_slug = {article.slug: article for article in articles}

    def evidence(slug: str) -> Dict[str, str]:
        article = article_by_slug[slug]
        return {
            "slug": article.slug,
            "title": article.title,
            "path": article.public_path,
            "reviewedAt": article.reviewed_at,
        }

    capabilities: List[Dict[str, object]] = []
    for raw_entry in catalog["capabilities"]:
        entry = dict(raw_entry)
        evidence_slugs = entry.pop("evidenceSlugs")
        entry["evidence"] = [evidence(slug) for slug in evidence_slugs]
        capabilities.append(entry)

    highlights: List[Dict[str, object]] = []
    for raw_entry in catalog["performanceHighlights"]:
        entry = dict(raw_entry)
        evidence_slug = entry.pop("evidenceSlug")
        entry["evidence"] = evidence(evidence_slug)
        highlights.append(entry)

    status_definitions = [
        {"id": identifier, "label": label, "description": description}
        for identifier, label, description in CAPABILITY_STATUS_DEFINITIONS
    ]
    public_index: Dict[str, object] = {
        "schemaVersion": 1,
        "project": "fast-mlx",
        "claimBoundary": catalog["claimBoundary"],
        "updatedAt": catalog["updatedAt"],
        "statusDefinitions": status_definitions,
        "capabilities": capabilities,
        "performanceHighlights": highlights,
    }

    status_by_id = {item[0]: item[1] for item in CAPABILITY_STATUS_DEFINITIONS}
    body: List[str] = [
        '<section class="page-hero shell capability-hero">',
        '<p class="eyebrow">Capabilities &amp; evidence</p>',
        '<h1>See what exists—and what each claim actually covers.</h1>',
        '<p class="lede">A generated inventory of fast-mlx source, scoped decisions, and measured results. Every evidence link resolves to a reviewed public note; source presence alone is never treated as production support.</p>',
        '<div class="hero-actions">',
        '<a class="button primary" href="https://github.com/bitworks-io/fast-mlx#first-run" rel="noreferrer">Run the scripted server</a>',
        '<a class="button secondary" href="https://github.com/bitworks-io/fast-mlx" rel="noreferrer">Inspect the source</a>',
        '</div>',
        '</section>',
        '<section class="section shell" aria-labelledby="measured-heading">',
        '<div class="section-heading"><p class="eyebrow">Measured highlights</p><h2 id="measured-heading">Useful numbers, attached to their boundaries.</h2><p class="section-intro">These are dated fast-mlx results—not cross-model, cross-hardware, or future-version guarantees.</p></div>',
        '<div class="evidence-grid">',
    ]
    for highlight in highlights:
        evidence_record = highlight["evidence"]
        href = relative_href("capabilities/index.html", evidence_record["path"])
        body.extend(
            [
                '<article class="evidence-card">',
                f'<div class="card-topline"><span class="status-badge status-{html.escape(highlight["decision"])}">{html.escape(status_by_id[highlight["decision"]].upper())}</span><time datetime="{html.escape(highlight["date"], quote=True)}">{html.escape(highlight["date"])}</time></div>',
                f'<div class="metric">{html.escape(highlight["metric"])}</div>',
                f'<h3>{html.escape(highlight["label"])}</h3>',
                '<dl class="evidence-context">',
                f'<div><dt>Model</dt><dd>{html.escape(highlight["model"])}</dd></div>',
                f'<div><dt>Hardware</dt><dd>{html.escape(highlight["hardware"])}</dd></div>',
                f'<div><dt>Workload</dt><dd>{html.escape(highlight["workload"])}</dd></div>',
                '</dl>',
                f'<p class="scope-note"><strong>Boundary:</strong> {html.escape(highlight["caveat"])}</p>',
                f'<a class="text-link" href="{html.escape(href, quote=True)}">Read the measured note →</a>',
                '</article>',
            ]
        )
    body.extend(
        [
            '</div>',
            '</section>',
            '<section class="section shell" aria-labelledby="inventory-heading">',
            '<div class="section-heading"><p class="eyebrow">Current inventory</p><h2 id="inventory-heading">Capabilities carry state, scope, and evidence.</h2></div>',
            '<ul class="status-legend" aria-label="Capability status definitions">',
        ]
    )
    for definition in status_definitions:
        body.append(
            f'<li><span class="status-badge status-{html.escape(definition["id"])}">{html.escape(definition["label"].upper())}</span><p>{html.escape(definition["description"])}</p></li>'
        )
    body.extend(['</ul>', '<div class="capability-grid">'])
    for capability in capabilities:
        detail_href = f'{html.escape(str(capability["id"]), quote=True)}/'
        body.extend(
            [
                '<article class="capability-card" data-capability-card="'
                + html.escape(str(capability["id"]), quote=True)
                + '" data-capability-state="'
                + html.escape(str(capability["status"]), quote=True)
                + '">',
                f'<span class="status-badge status-{html.escape(capability["status"])}">{html.escape(status_by_id[capability["status"]].upper())}</span>',
                f'<h3>{html.escape(capability["name"])}</h3>',
                f'<p>{html.escape(capability["summary"])}</p>',
                f'<p class="scope-note"><strong>Scope:</strong> {html.escape(capability["scope"])}</p>',
                '<div class="evidence-links" aria-label="Published evidence">',
            ]
        )
        for record in capability["evidence"]:
            href = relative_href("capabilities/index.html", record["path"])
            body.append(
                f'<a href="{html.escape(href, quote=True)}">{html.escape(record["title"])} →</a>'
            )
        body.extend(
            [
                '</div>',
                f'<a class="text-link" href="{detail_href}">Open capability details →</a>',
                '</article>',
            ]
        )
    body.extend(
        [
            '</div>',
            '</section>',
            '<section class="section shell callout capability-callout" aria-labelledby="machine-heading">',
            '<p class="eyebrow">Machine-readable contract</p>',
            '<h2 id="machine-heading">Agents can inspect the same bounded inventory.</h2>',
            '<p>The generated JSON mirrors this page and includes resolved public evidence paths. The build refuses unknown states, missing scope, and unpublished evidence slugs.</p>',
            '<a class="text-link" href="index.json">Open capabilities/index.json →</a>',
            '</section>',
        ]
    )
    return "\n".join(body), public_index


def capability_detail_title(capability: Dict[str, object]) -> str:
    return f'{capability["name"]} — fast-mlx capability'


def capability_detail_description(capability: Dict[str, object]) -> str:
    return f'Reviewed fast-mlx capability state and evidence for {capability["name"]}.'


def render_capability_detail(capability: Dict[str, object]) -> str:
    """Render one immutable view of an already-reviewed capability record."""

    status_definitions = {
        identifier: (label, description)
        for identifier, label, description in CAPABILITY_STATUS_DEFINITIONS
    }
    identifier = str(capability["id"])
    state = str(capability["status"])
    status_label, status_description = status_definitions[state]
    evidence_items: List[str] = []
    for record in capability["evidence"]:
        evidence_href = relative_href(
            f"capabilities/{identifier}/index.html",
            str(record["path"]),
        )
        reviewed_at = str(record["reviewedAt"])
        evidence_items.extend(
            [
                '<li data-capability-evidence="'
                + html.escape(str(record["path"]), quote=True)
                + '" data-reviewed-at="'
                + html.escape(reviewed_at, quote=True)
                + '">',
                '<a href="'
                + html.escape(evidence_href, quote=True)
                + '">'
                + html.escape(str(record["title"]))
                + " →</a>",
                '<span class="evidence-meta">Path: '
                + html.escape(str(record["path"]))
                + " · Reviewed: "
                + html.escape(reviewed_at)
                + "</span>",
                "</li>",
            ]
        )
    return "\n".join(
        [
            '<section class="page-hero shell capability-detail" '
            'data-capability-detail data-capability-id="'
            + html.escape(identifier, quote=True)
            + '" data-capability-state="'
            + html.escape(state, quote=True)
            + '">',
            '<p class="eyebrow">Reviewed capability</p>',
            '<div class="card-topline">'
            f'<span class="status-badge status-{html.escape(state, quote=True)}">{html.escape(status_label.upper())}</span>'
            f'<span class="evidence-meta">{html.escape(status_description)}</span>'
            '</div>',
            f'<h1>{html.escape(str(capability["name"]))}</h1>',
            f'<p class="lede">{html.escape(str(capability["summary"]))}</p>',
            f'<p class="scope-note"><strong>Scope:</strong> {html.escape(str(capability["scope"]))}</p>',
            '<section class="section capability-evidence" aria-labelledby="capability-evidence-heading">',
            '<h2 id="capability-evidence-heading">Reviewed evidence</h2>',
            '<ul class="evidence-list">',
            *evidence_items,
            '</ul>',
            '</section>',
            '<div class="hero-actions">',
            '<a class="button secondary" href="../">Back to all capabilities</a>',
            '<a class="button secondary" href="../index.json">Open capabilities/index.json →</a>',
            '<a class="button secondary" href="../../methodology/">Read the methodology →</a>',
            '</div>',
            '</section>',
            '<section class="section shell callout capability-callout" aria-labelledby="capability-detail-boundary-heading">',
            '<p class="eyebrow">Claim boundary</p>',
            '<h2 id="capability-detail-boundary-heading">A permalink is not new authority.</h2>',
            '<p>This page creates no broader support, measurement, runtime, model, acquisition, publication, admission, launchability, or containment authority. It only exposes one reviewed capability record and its already-published evidence.</p>',
            '</section>',
        ]
    )


def render_benchmark_explorer(
    highlights: Sequence[Dict[str, object]],
    served: Sequence[Dict[str, object]] = (),
    cards_by_id: Optional[Dict[str, Dict[str, object]]] = None,
) -> str:
    status_by_id = {item[0]: item[1] for item in CAPABILITY_STATUS_DEFINITIONS}

    def options(values: Iterable[str], all_label: str) -> str:
        rendered = [f'<option value="">{html.escape(all_label)}</option>']
        for value in sorted(set(values), key=str.casefold):
            escaped = html.escape(value, quote=True)
            rendered.append(f'<option value="{escaped}">{html.escape(value)}</option>')
        return "".join(rendered)

    model_options = options(
        (str(highlight["model"]) for highlight in highlights), "All models"
    )
    hardware_options = options(
        (str(highlight["hardware"]) for highlight in highlights), "All hardware"
    )
    decision_options = ['<option value="">All decisions</option>']
    for decision in sorted(
        {str(highlight["decision"]) for highlight in highlights},
        key=lambda value: status_by_id[value].casefold(),
    ):
        decision_options.append(
            f'<option value="{html.escape(decision, quote=True)}">'
            f'{html.escape(status_by_id[decision])}</option>'
        )

    total = len(highlights)
    body: List[str] = [
        '<section class="page-hero shell benchmark-hero">',
        '<p class="eyebrow">Benchmark explorer</p>',
        '<h1>Compare the result, then read its boundary.</h1>',
        '<p class="lede">Filter fast-mlx’s reviewed measurements by model, hardware, or decision. Every result keeps its workload, date, caveat, and evidence attached; these are not cross-system or future-version guarantees.</p>',
        '<div class="hero-actions">',
        '<a class="button secondary" href="../methodology/">Read the methodology</a>',
        '<a class="button secondary" href="../capabilities/index.json">Inspect the source contract</a>',
        '</div>',
        '</section>',
        '<section class="section shell" aria-labelledby="benchmark-results-heading">',
        '<div class="section-heading"><p class="eyebrow">Reviewed fast-mlx evidence only</p><h2 id="benchmark-results-heading">Scope travels with every number.</h2><p class="section-intro">Filters change only what is visible. They do not recompute, normalize, rank, or combine measurements.</p></div>',
        '<form class="benchmark-controls" data-benchmark-controls aria-label="Filter benchmark evidence" action="./" method="get">',
        '<div><label for="benchmark-model">Model</label><select id="benchmark-model" name="model">'
        + model_options
        + '</select></div>',
        '<div><label for="benchmark-hardware">Hardware</label><select id="benchmark-hardware" name="hardware">'
        + hardware_options
        + '</select></div>',
        '<div><label for="benchmark-decision">Decision</label><select id="benchmark-decision" name="decision">'
        + "".join(decision_options)
        + '</select></div>',
        '<button class="button secondary benchmark-reset" data-benchmark-reset type="reset">Clear filters</button>',
        '</form>',
        '<noscript><p class="benchmark-noscript">JavaScript is optional. All reviewed results remain visible below; interactive filters are unavailable.</p></noscript>',
        f'<p class="benchmark-count" data-benchmark-count aria-live="polite">Showing {total} of {total} reviewed results.</p>',
        '<div class="benchmark-results" data-benchmark-results role="list">',
    ]
    for highlight in sorted(
        highlights, key=lambda entry: str(entry["date"]), reverse=True
    ):
        evidence_record = highlight["evidence"]
        href = relative_href("benchmarks/index.html", evidence_record["path"])
        decision = str(highlight["decision"])
        body.extend(
            [
                '<article class="benchmark-result" role="listitem" '
                f'data-benchmark-card data-highlight-id="{html.escape(str(highlight["id"]), quote=True)}" '
                f'data-model="{html.escape(str(highlight["model"]), quote=True)}" '
                f'data-hardware="{html.escape(str(highlight["hardware"]), quote=True)}" '
                f'data-decision="{html.escape(decision, quote=True)}">',
                '<div class="card-topline">'
                f'<span class="status-badge status-{html.escape(decision, quote=True)}">{html.escape(status_by_id[decision].upper())}</span>'
                f'<time datetime="{html.escape(str(highlight["date"]), quote=True)}">{html.escape(str(highlight["date"]))}</time>'
                '</div>',
                f'<div class="metric">{html.escape(str(highlight["metric"]))}</div>',
                f'<h3>{html.escape(str(highlight["label"]))}</h3>',
                '<dl class="evidence-context">',
                f'<div><dt>Model</dt><dd>{html.escape(str(highlight["model"]))}</dd></div>',
                f'<div><dt>Hardware</dt><dd>{html.escape(str(highlight["hardware"]))}</dd></div>',
                f'<div><dt>Workload</dt><dd>{html.escape(str(highlight["workload"]))}</dd></div>',
                '</dl>',
                f'<p class="scope-note"><strong>Boundary:</strong> {html.escape(str(highlight["caveat"]))}</p>',
                '<div class="evidence-links" aria-label="Benchmark evidence links">',
                f'<a class="text-link" href="{html.escape(str(highlight["id"]), quote=True)}/">Open reviewed result →</a>',
                f'<a class="text-link" href="{html.escape(href, quote=True)}">Read {html.escape(str(evidence_record["title"]))} →</a>',
                '</div>',
                '</article>',
            ]
        )
    body.extend(
        [
            '</div>',
            '<div class="benchmark-empty" data-benchmark-empty hidden role="status"><h3>No reviewed results match those filters.</h3><p>Clear the filters to return to the complete evidence set.</p><a class="text-link" href="./">Reset the benchmark explorer →</a></div>',
            '</section>',
            '<section class="section shell callout benchmark-callout" aria-labelledby="benchmark-boundary-heading">',
            '<p class="eyebrow">Claim boundary</p>',
            '<h2 id="benchmark-boundary-heading">This explorer never manufactures a comparison.</h2>',
            '<p>It presents only the reviewed entries already published in the fast-mlx capability contract. It performs no unit conversion, interpolation, aggregation, competitor comparison, or live benchmark execution.</p>',
            '<a class="text-link" href="../capabilities/">See the complete capability inventory →</a>',
            '</section>',
        ]
    )
    if served:
        cards = cards_by_id or {}
        body.append(
            '<section class="section shell" data-served-benchmarks aria-labelledby="served-benchmarks-heading">'
        )
        body.extend(
            [
                '<div class="section-heading"><p class="eyebrow">Served-engine ledger</p>'
                '<h2 id="served-benchmarks-heading">fast-mlx quality-carded packs, measured on a served engine.</h2>'
                '<p class="section-intro">Each row is a fast-mlx measurement, made with fastmlx bench, of a pack '
                'that carries a published quality card, on a serving engine identified by its build commit, not '
                'by name. The ratio compares the pack with its card’s 8-bit reference on the same host, build '
                'and prompt set; it is not a comparison between engines. Each entry’s row sha256 is the sha256 '
                'of its linked, published row file, so a reader can recompute every number shown from that '
                'file.</p></div>',
                '<div class="evidence-grid">',
            ]
        )
        for entry in served:
            card = cards.get(str(entry["cardId"]), {})
            model = card.get("model", {}) if isinstance(card, dict) else {}
            family = str(model.get("family", ""))
            legible = card.get("legible", {}) if isinstance(card, dict) else {}
            benefit = legible.get("benefit", {}) if isinstance(legible, dict) else {}
            speed_x = benefit.get("speedX")
            result = entry["result"]
            serving = entry["serving"]
            harness = entry["harness"]
            workload = entry["workload"]
            engine_commit = str(entry["engineBuild"]["commit"])
            completion_tokens = workload["completionTokens"]
            chip = str(entry["chip"])
            candidate_rate_text = f'{float(result["candidateDecodeTokS"]):.2f}'
            reference_rate_text = f'{float(result["referenceDecodeTokS"]):.2f}'
            serving_text = (
                f'{int(serving["contextTokens"]):,}-token context; '
                f'MTP {serving["mtp"]}; drafter {serving["drafter"]}; '
                f'prompt lookup {serving["promptLookup"]}'
            )
            harness_dd = (
                f'measured with <code>{html.escape(str(harness["publicCommit"])[:8])}</code> '
                f'(bench <code>{html.escape(str(harness["benchSha256"]))}</code>, '
                f'prompts <code>{html.escape(str(harness["promptSetSha256"]))}</code>), '
                f'combined with <code>{html.escape(str(harness["combineCommit"])[:8])}</code>'
            )
            scope_note = (
                f'measured on {chip} at engine build {engine_commit[:8]}: median '
                f'{candidate_rate_text} vs {reference_rate_text} tok/s against the 8-bit '
                f'reference pack, concurrency 1; answers ended after {completion_tokens[0]}–'
                f'{completion_tokens[1]} tokens; reference ran first and last and drifted '
                f'{result["referenceDriftPct"]}%; holds for this host, build and prompt set only.'
            )
            body.extend(
                [
                    '<article class="evidence-card" data-served-benchmark="'
                    + html.escape(str(entry["id"]), quote=True)
                    + '" data-card-id="'
                    + html.escape(str(entry["cardId"]), quote=True)
                    + '">',
                    f'<time datetime="{html.escape(str(entry["measuredAt"]), quote=True)}">'
                    f'{html.escape(str(entry["measuredAt"]))}</time>',
                    '<div class="metric">'
                    + html.escape(f'{float(result["ratio"]):.3f}x')
                    + '</div>',
                    '<p class="metric-note">decode rate vs the 8-bit reference</p>',
                    f'<h3>{html.escape(family)}, {html.escape(str(entry["packLabel"]))}</h3>',
                    '<dl class="evidence-context">',
                    f'<div><dt>Hardware</dt><dd>{html.escape(chip)}</dd></div>',
                    f'<div><dt>Engine build</dt><dd><code>{html.escape(engine_commit[:8])}</code></dd></div>',
                    f'<div><dt>Serving</dt><dd>{html.escape(serving_text)}</dd></div>',
                    f'<div><dt>Harness</dt><dd>{harness_dd}</dd></div>',
                    '<div><dt>Row sha256</dt><dd><a href="'
                    + html.escape(
                        f'{SERVED_BENCHMARK_ROWS_DIRNAME}/{entry["id"]}.json', quote=True
                    )
                    + f'"><code>{html.escape(str(entry["rowSha256"]))}</code></a></dd></div>',
                    '<div><dt>Quality card</dt><dd>'
                    f'<a href="../quality/">{html.escape(str(entry["cardId"]))}</a> — card speedX '
                    f'{html.escape(str(speed_x))}</dd></div>',
                    '</dl>',
                    f'<p class="scope-note">{html.escape(scope_note)}</p>',
                    '<div class="evidence-rerun">',
                ]
            )
            for key in ("measure", "combine", "publicView"):
                body.append(f'<pre><code>{html.escape(str(entry["rerun"][key]))}</code></pre>')
            body.extend(['</div>', '</article>'])
        body.extend(
            [
                '</div>',
                '</section>',
            ]
        )
    return "\n".join(body)


def quality_provenance_label(provenance: Dict[str, object]) -> str:
    source = str(provenance["source"])
    if source == "vendor-reported":
        return f'vendor-reported ({provenance.get("vendor")})'
    return source


def quality_speed_line(benefit: Dict[str, object]) -> str:
    speed_x = benefit.get("speedX")
    status = str(benefit["speedXStatus"])
    if speed_x is None:
        return status
    # speedXStatus carries the only measurement boundary (host, engine build,
    # flags, prompt count) fast-mlx has for a numeric speedX, so it must stay
    # attached to the number, not disappear once a measurement lands.
    if speed_x > 1.0:
        direction = f'{speed_x}x faster on this engine (measured)'
    elif speed_x < 1.0:
        # Not "{n}x slower": for a ratio below 1 that phrasing reads as a
        # factor of slowdown and inverts the meaning. State the ratio, then
        # name the direction separately.
        direction = f'{speed_x}x the reference speed on this engine — a slowdown (measured)'
    else:
        direction = 'no measured speed difference on this engine (measured)'
    return f'{direction} — {status}'


def quality_admission_framing(verdict: str, admission: Dict[str, object]) -> str:
    if verdict == "NO_GO":
        return f'Opt-in choice, not a silent default — {admission["reason"]}'
    if verdict == "REFERENCE":
        return "This is the reference other configs are measured against; not an opt-in."
    if verdict == "EXACT":
        return "Identical output, just faster — proven token-for-token, not a quality trade."
    if verdict == "PASS":
        return "Passes review as an informational result."
    return "Not yet measured; the existing serving path is unchanged for this config."


def render_quality_guide(cards: Sequence[Dict[str, object]]) -> str:
    """Render the public, user-legible quality-guide page from validated quality cards.

    Every card leads with the legible summary (tier, headline, next-word drift,
    regression focus, benefit) and mandatory provenance + boundary; `rawMetrics`
    is only ever available behind a `<details>` expander.
    """

    body: List[str] = [
        '<section class="page-hero shell quality-hero">',
        '<p class="eyebrow">Quality guide</p>',
        '<h1>What a lower-bit or enhanced config actually costs you.</h1>',
        '<p class="lede">Each card translates a measured fast-mlx quality signal into a '
        'plain-language decision: what changes for you on this model, and whether that '
        'trade is one you would elect. Raw metrics stay one click away — never the '
        'headline.</p>',
        '</section>',
        '<section class="section shell" aria-labelledby="quality-cards-heading">',
        '<div class="section-heading"><p class="eyebrow">Reviewed fast-mlx evidence only</p>'
        '<h2 id="quality-cards-heading">Lead with the plain-language cost, not the raw number.</h2>'
        '<p class="section-intro">A NO-GO card is an informed opt-in with a stated cost, never '
        'a defect report. A REFERENCE card is the quality bar the others are measured against. '
        'An EXACT card is a free '
        'speedup with proven identical output.</p></div>',
        '<div class="quality-card-grid">',
    ]
    for card in cards:
        model = card["model"]
        legible = card["legible"]
        benefit = legible["benefit"]
        provenance = card["provenance"]
        boundary = card["boundary"]
        admission = card["admission"]
        verdict = str(card["verdict"])
        tier = str(legible["tier"])
        verdict_label = QUALITY_VERDICT_LABELS[verdict]
        drift = legible.get("nextWordDrift")
        regression_focus = legible.get("regressionFocus")
        fit = benefit.get("fit")
        family = str(model["family"])
        method = str(provenance["method"])

        body.extend(
            [
                '<article class="quality-card" data-quality-card="'
                + html.escape(str(card["id"]), quote=True)
                + '" data-quality-verdict="'
                + html.escape(verdict, quote=True)
                + '">',
                '<div class="card-topline">'
                f'<span class="quality-tier-badge" data-quality-tier="{html.escape(tier, quote=True)}">{html.escape(tier.upper())}</span>'
                f'<span class="quality-verdict-badge" data-quality-verdict-label>{html.escape(verdict_label)}</span>'
                '</div>',
                f'<p class="quality-family">{html.escape(family)}</p>',
                f'<h3>{html.escape(str(legible["headline"]))}</h3>',
            ]
        )
        # nextWordDrift is null for REFERENCE (no drift vs itself); oneInK is null for
        # EXACT (identical output, so there is no "1 in K" to state). Tier
        # "Unquantified" is the third null-drift case: a NO_GO card whose output
        # differed from the reference, but whose difference was never sized -- render
        # that honestly (no fabricated "1 in K" line) instead of silently saying
        # nothing, which would read as "no drift" like the REFERENCE case above.
        if drift is not None:
            one_in_k = drift.get("oneInK")
            top1 = drift["top1AgreementPct"]
            if one_in_k is None:
                body.append(
                    '<p class="quality-drift">Identical output — no next-word drift '
                    f'(top-1 {html.escape(str(top1))}%).</p>'
                )
            else:
                body.append(
                    '<p class="quality-drift">Next-word drift: about 1 word in '
                    f'{html.escape(str(one_in_k))} '
                    f'(top-1 agreement {html.escape(str(top1))}%).</p>'
                )
        elif tier == "Unquantified":
            body.append(
                '<p class="quality-drift">Output differed from the reference; the '
                'size of that difference was not measured.</p>'
            )
        if regression_focus is not None:
            body.append(
                f'<p class="quality-regression"><strong>Regression focus:</strong> {html.escape(str(regression_focus))}</p>'
            )
        # legible.example is "the visceral side-by-side" (schema v1): the one
        # real instance of the measured difference, not just its percentage.
        # A "pending" status must render an explicit line rather than
        # silence -- silence would read as "nothing differs", the same
        # failure mode the Unquantified drift branch above exists to avoid.
        example = legible.get("example")
        if example is not None:
            example_status = str(example.get("status"))
            if example_status == "pending":
                body.append(
                    '<div class="quality-example quality-example-pending">'
                    '<p class="quality-example-label"><strong>Example:</strong> '
                    'No side-by-side has been extracted for this card yet.</p>'
                    '</div>'
                )
            else:
                example_prompt = example.get("prompt")
                example_reference_output = example.get("referenceOutput")
                example_config_output = example.get("configOutput")
                example_note = example.get("note")
                if example_status == "illustrative":
                    example_label = "Illustrative example — not a measured case"
                else:
                    example_label = "Measured example"
                example_parts = [
                    f'<div class="quality-example quality-example-{html.escape(example_status, quote=True)}">',
                    f'<p class="quality-example-label"><strong>{html.escape(example_label)}</strong></p>',
                ]
                if example_prompt is not None:
                    example_parts.append(
                        '<p class="quality-example-context"><strong>Context:</strong> '
                        f'{html.escape(str(example_prompt))}</p>'
                    )
                if example_reference_output is not None:
                    example_parts.append(
                        '<p class="quality-example-reference"><strong>Other arm said:</strong> '
                        f'{html.escape(str(example_reference_output))}</p>'
                    )
                if example_config_output is not None:
                    example_parts.append(
                        '<p class="quality-example-config"><strong>This pack said:</strong> '
                        f'{html.escape(str(example_config_output))}</p>'
                    )
                if example_note:
                    example_parts.append(
                        '<p class="quality-example-note"><strong>Note:</strong> '
                        f'{html.escape(str(example_note))}</p>'
                    )
                example_parts.append('</div>')
                body.extend(example_parts)
        body.extend(
            [
                f'<p class="quality-admission"><strong>{html.escape(verdict_label)}:</strong> '
                f'{html.escape(quality_admission_framing(verdict, admission))}</p>',
                '<dl class="evidence-context">',
            ]
        )
        if fit is not None:
            body.append(f'<div><dt>Fit</dt><dd>{html.escape(str(fit))}</dd></div>')
        body.extend(
            [
                f'<div><dt>Speed</dt><dd>{html.escape(quality_speed_line(benefit))}</dd></div>',
                '</dl>',
                f'<p class="quality-provenance"><strong>Provenance:</strong> {html.escape(quality_provenance_label(provenance))}</p>',
                f'<p class="quality-method"><strong>Method:</strong> {html.escape(method)}</p>',
            ]
        )
        confound = provenance.get("confound")
        if confound:
            body.append(
                f'<p class="quality-confound"><strong>Confound:</strong> {html.escape(str(confound))}</p>'
            )
        body.append(
            f'<p class="scope-note"><strong>Boundary:</strong> {html.escape(str(boundary["scope"]))}</p>'
        )
        measured_new_tokens = boundary.get("measuredNewTokens")
        if measured_new_tokens is not None:
            body.append(
                '<p class="scope-note quality-measured-tokens"><strong>Measured over:</strong> '
                f"{html.escape(str(measured_new_tokens))} generated tokens per prompt; "
                "longer generations are outside what this card measured.</p>"
            )
        body.extend(
            [
                '<p class="scope-note quality-unmeasured-label"><strong>Not measured:</strong></p>',
                '<ul class="quality-unmeasured-list">',
                *[
                    f'<li>{html.escape(str(entry))}</li>'
                    for entry in boundary["unmeasured"]
                ],
                '</ul>',
            ]
        )
        body.extend(
            [
                '<details class="quality-raw-metrics">',
                '<summary>Raw metrics</summary>',
                '<pre>'
                + html.escape(
                    json.dumps(card["rawMetrics"], indent=2, ensure_ascii=False, sort_keys=True)
                )
                + '</pre>',
                '</details>',
                '</article>',
            ]
        )
    body.extend(
        [
            '</div>',
            '</section>',
            '<section class="section shell callout quality-callout" aria-labelledby="quality-boundary-heading">',
            '<p class="eyebrow">Claim boundary</p>',
            '<h2 id="quality-boundary-heading">A quality card never becomes a serve refusal on its own.</h2>',
            '<p>An unmeasured model or config serves exactly as it does today. Only a card carrying '
            'verdict NO_GO gates a silent default, and only for the exact model, config, and hardware '
            'class it was measured on.</p>',
            '</section>',
        ]
    )
    return "\n".join(body)


def research_search_text(article: Article) -> str:
    """Return the exact public fields searched by the research archive."""

    return " ".join(f"{article.title} {article.summary} {article.theme}".split())


def render_research_archive(articles: Sequence[Article]) -> str:
    """Render reviewed notes with a no-JavaScript-readable filter shell."""

    if not articles:
        fail("research archive requires at least one reviewed article")

    themes = sorted({article.theme for article in articles}, key=str.casefold)
    theme_options = ['<option value="">All themes</option>']
    for theme in themes:
        theme_options.append(
            f'<option value="{html.escape(theme, quote=True)}">'
            f"{html.escape(theme)}</option>"
        )

    total = len(articles)
    body: List[str] = [
        '<section class="page-hero shell"><p class="eyebrow">Research notes</p>',
        '<h1>What the measurements changed.</h1>',
        '<p class="lede">Dated fast-mlx investigations, including negative results. Each note is a scoped historical result, not a timeless performance guarantee.</p>',
        '<div class="hero-actions">',
        '<a class="button primary" href="index.json">Read the research JSON</a>',
        '<a class="button secondary" href="feed.atom" type="application/atom+xml">Subscribe to reviewed research</a>',
        '<a class="button secondary" href="../feed.atom" type="application/atom+xml">Subscribe to all reviewed updates</a>',
        '</div></section>',
        '<section class="section shell" aria-labelledby="research-results-heading">',
        '<div class="section-heading"><p class="eyebrow">Reviewed investigations only</p><h2 id="research-results-heading">Find the evidence trail you need.</h2><p class="section-intro">Search checks only each reviewed note’s title, summary, and published theme. Filters change visibility, never order, evidence, or publication state.</p></div>',
        '<form class="research-controls" data-research-controls aria-label="Filter reviewed research" action="./" method="get">',
        '<div><label for="research-query">Search notes</label><input id="research-query" name="q" type="search" maxlength="120" autocomplete="off"></div>',
        '<div><label for="research-theme">Theme</label><select id="research-theme" name="theme">'
        + "".join(theme_options)
        + '</select></div>',
        '<button class="button secondary research-reset" data-research-reset type="reset">Clear filters</button>',
        '</form>',
        '<noscript><p class="research-noscript">JavaScript is optional. All reviewed notes remain visible below; interactive filters are unavailable.</p></noscript>',
        f'<p class="research-count" data-research-count aria-live="polite" aria-atomic="true">Showing {total} of {total} reviewed notes.</p>',
        '<div class="research-grid" data-research-results role="list">',
    ]
    for article in articles:
        body.extend(
            [
                '<article class="note-card" role="listitem" data-research-card '
                f'data-research-path="{html.escape(article.public_path, quote=True)}" '
                f'data-theme="{html.escape(article.theme, quote=True)}" '
                f'data-search="{html.escape(research_search_text(article), quote=True)}">',
                f'<div class="meta">{html.escape(article.date)} · {html.escape(article.theme)}</div>',
                f'<h2>{html.escape(article.title)}</h2>',
                f'<p>{html.escape(article.summary)}</p>',
                f'<a href="{html.escape(article.slug, quote=True)}/">Read the note →</a>',
                '</article>',
            ]
        )
    body.extend(
        [
            '</div>',
            '<div class="research-empty" data-research-empty hidden role="status"><h3>No reviewed notes match those filters.</h3><p>Clear the filters to return to the complete research archive.</p><a class="text-link" href="./">Show all reviewed notes →</a></div>',
            '</section>',
            '<section class="section shell callout research-callout" aria-labelledby="research-boundary-heading">',
            '<p class="eyebrow">Archive boundary</p>',
            '<h2 id="research-boundary-heading">Filtering never creates evidence.</h2>',
            '<p>This page presents only notes already admitted by the reviewed publication manifest. It performs no external request, ingestion, ranking, benchmark recomputation, publication action, or authority transition.</p>',
            '</section>',
        ]
    )
    return "\n".join(body)


def benchmark_detail_title(highlight: Dict[str, object]) -> str:
    return f'{highlight["label"]} — fast-mlx benchmark evidence'


def benchmark_detail_description() -> str:
    return (
        "A reviewed fast-mlx benchmark result with its exact model, hardware, "
        "workload, decision, caveat, and evidence."
    )


def render_benchmark_detail(highlight: Dict[str, object]) -> str:
    """Render one immutable view of an already-reviewed performance highlight."""

    status_by_id = {item[0]: item[1] for item in CAPABILITY_STATUS_DEFINITIONS}
    identifier = str(highlight["id"])
    decision = str(highlight["decision"])
    evidence_record = highlight["evidence"]
    evidence_href = relative_href(
        f"benchmarks/{identifier}/index.html", str(evidence_record["path"])
    )
    return "\n".join(
        [
            '<section class="page-hero shell benchmark-detail" '
            'data-benchmark-detail data-highlight-id="'
            + html.escape(identifier, quote=True)
            + '">',
            '<p class="eyebrow">Reviewed benchmark evidence</p>',
            '<div class="card-topline">'
            f'<span class="status-badge status-{html.escape(decision, quote=True)}">'
            f'{html.escape(status_by_id[decision].upper())}</span>'
            f'<time datetime="{html.escape(str(highlight["date"]), quote=True)}">'
            f'{html.escape(str(highlight["date"]))}</time>'
            '</div>',
            f'<div class="metric">{html.escape(str(highlight["metric"]))}</div>',
            f'<h1>{html.escape(str(highlight["label"]))}</h1>',
            '<p class="lede">One reviewed fast-mlx result, shown with the context and boundary that made it admissible.</p>',
            '<dl class="evidence-context">',
            f'<div><dt>Model</dt><dd>{html.escape(str(highlight["model"]))}</dd></div>',
            f'<div><dt>Hardware</dt><dd>{html.escape(str(highlight["hardware"]))}</dd></div>',
            f'<div><dt>Workload</dt><dd>{html.escape(str(highlight["workload"]))}</dd></div>',
            '</dl>',
            f'<p class="scope-note"><strong>Boundary:</strong> {html.escape(str(highlight["caveat"]))}</p>',
            '<div class="hero-actions">',
            f'<a class="button primary" href="{html.escape(evidence_href, quote=True)}">Read {html.escape(str(evidence_record["title"]))}</a>',
            '<a class="button secondary" href="../">Back to all reviewed results</a>',
            '</div>',
            '</section>',
            '<section class="section shell callout benchmark-callout" aria-labelledby="benchmark-detail-boundary-heading">',
            '<p class="eyebrow">Claim boundary</p>',
            '<h2 id="benchmark-detail-boundary-heading">A permalink is not a broader performance claim.</h2>',
            '<p>This page does not normalize, rank, aggregate, or recompute results. It performs no unit conversion, interpolation, competitor comparison, live benchmark execution, or authority transition.</p>',
            '<a class="text-link" href="../../methodology/">Read the measurement methodology →</a>',
            '</section>',
        ]
    )


def render_home_current_cycle(
    catalog: Dict[str, object],
    articles: Sequence[Article],
    release_catalog: Dict[str, object],
) -> str:
    """Render the home-page snapshot from reviewed public manifests only."""

    capabilities = catalog["capabilities"]
    releases = release_catalog["releases"]
    latest = releases[0]
    boundary = release_catalog["currentBoundary"]
    boundary_evidence = boundary["evidence"]
    status_counts = {
        status: sum(capability["status"] == status for capability in capabilities)
        for status in CAPABILITY_STATUSES
    }
    status_labels = {
        status: label for status, label, _description in CAPABILITY_STATUS_DEFINITIONS
    }
    published_at = str(latest["publishedAt"])
    commit = str(latest["publicCommit"])
    latest_id = str(latest["id"])

    status_summary = "; ".join(
        '<span data-capability-status="'
        + html.escape(status, quote=True)
        + '" data-count="'
        + str(status_counts[status])
        + '">'
        + str(status_counts[status])
        + " "
        + html.escape(status_labels[status].casefold())
        + "</span>"
        for status, _label, _description in CAPABILITY_STATUS_DEFINITIONS
    )

    return "\n".join(
        [
            '<section class="section shell split" aria-labelledby="current-cycle-heading" '
            'data-current-cycle data-latest-release-id="'
            + html.escape(latest_id, quote=True)
            + '" data-boundary-id="'
            + html.escape(str(boundary["id"]), quote=True)
            + '" data-boundary-state="'
            + html.escape(str(boundary["state"]), quote=True)
            + '">',
            '<div>',
            '<p class="eyebrow">Current reviewed cycle</p>',
            '<div class="card-topline"><span class="status-badge status-released">RELEASED</span> '
            f'<time datetime="{html.escape(published_at, quote=True)}">{html.escape(published_at[:10])}</time></div>',
            f'<h2 id="current-cycle-heading">{html.escape(str(latest["title"]))}</h2>',
            f'<p>{html.escape(str(latest["summary"]))}</p>',
            '<p class="scope-note"><strong>Boundary:</strong> '
            + html.escape(str(latest["scope"]))
            + "</p>",
            '<div class="release-links">',
            '<a class="text-link" href="releases/'
            + html.escape(latest_id, quote=True)
            + '/">Inspect the latest reviewed milestone →</a>',
            '<a href="https://github.com/bitworks-io/fast-mlx/commit/'
            + html.escape(commit, quote=True)
            + '" rel="noreferrer">Inspect commit '
            + html.escape(commit[:12])
            + " →</a>",
            '</div>',
            '</div>',
            '<div class="capability-list" role="list" aria-label="Current reviewed evidence inventory">',
            '<article role="listitem">',
            f'<h3>{len(capabilities)} reviewed capabilities</h3>',
            f'<p>{status_summary}</p>',
            '<a class="text-link" href="capabilities/">Inspect capability states →</a>',
            '</article>',
            '<article role="listitem">',
            f'<h3>{len(articles)} published research notes</h3>',
            '<p>Dated investigations preserve promoted, shelved, rejected, and diagnostic outcomes.</p>',
            '<a class="text-link" href="research/">Read the evidence trail →</a>',
            '</article>',
            '<article role="listitem">',
            f'<h3>{len(releases)} reviewed releases</h3>',
            '<p>Each public milestone names its exact commit and unchanged claim boundary.</p>',
            '<a class="text-link" href="releases/">Follow the release ledger →</a>',
            '</article>',
            '<article role="listitem">',
            f'<span class="status-badge status-gated">{html.escape(str(boundary["state"]).upper())}</span>',
            f'<h3>{html.escape(str(boundary["label"]))}</h3>',
            f'<p>{html.escape(str(boundary["summary"]))}</p>',
            '<a class="text-link" href="'
            + html.escape(str(boundary_evidence["path"]), quote=True)
            + '">'
            + html.escape(str(boundary_evidence["label"]))
            + " →</a>",
            '</article>',
            '</div>',
            '</section>',
        ]
    )


def render_status_page(
    catalog: Dict[str, object],
    articles: Sequence[Article],
    release_catalog: Dict[str, object],
) -> str:
    """Render one static current-state reader from reviewed public manifests."""

    article_by_slug = {article.slug: article for article in articles}
    capabilities = catalog["capabilities"]
    highlights = catalog["performanceHighlights"]
    releases = release_catalog["releases"]
    latest = releases[0]
    boundary = release_catalog["currentBoundary"]
    boundary_evidence = boundary["evidence"]
    status_labels = {
        status: label for status, label, _description in CAPABILITY_STATUS_DEFINITIONS
    }
    status_counts = {
        status: sum(capability["status"] == status for capability in capabilities)
        for status in CAPABILITY_STATUSES
    }
    status_summary = "; ".join(
        '<span data-status-count="'
        + html.escape(status, quote=True)
        + '" data-count="'
        + str(status_counts[status])
        + '">'
        + str(status_counts[status])
        + " "
        + html.escape(status_labels[status].casefold())
        + "</span>"
        for status, _label, _description in CAPABILITY_STATUS_DEFINITIONS
    )
    latest_id = str(latest["id"])
    latest_commit = str(latest["publicCommit"])
    latest_published_at = str(latest["publishedAt"])

    body: List[str] = [
        '<div data-status-page data-latest-release-id="'
        + html.escape(latest_id, quote=True)
        + '" data-boundary-id="'
        + html.escape(str(boundary["id"]), quote=True)
        + '" data-boundary-state="'
        + html.escape(str(boundary["state"]), quote=True)
        + '">',
        '<section class="page-hero shell status-hero">',
        '<p class="eyebrow">Reviewed current state</p>',
        '<h1>Know what fast-mlx can do today—and where the proof stops.</h1>',
        '<p class="lede">Current state, not a roadmap. This dashboard is generated from reviewed capability, release, and research manifests; it does not create new measurement, performance, model, runtime, acquisition, or publication authority.</p>',
        '<div class="hero-actions">',
        '<a class="button primary" href="../quickstart/">Run the model-free quickstart</a>',
        '<a class="button secondary" href="../methodology/">Read the evidence rules</a>',
        '</div>',
        '</section>',
        '<section class="section shell" aria-labelledby="status-summary-heading">',
        '<div class="section-heading"><p class="eyebrow">At a glance</p><h2 id="status-summary-heading">One bounded view of the reviewed public record.</h2><p class="section-intro">Counts describe the manifests below, not broad model support or future commitments.</p></div>',
        '<div class="capability-list" role="list" aria-label="Reviewed status summary">',
        '<article role="listitem">',
        f'<h3>{len(capabilities)} reviewed capabilities</h3>',
        f'<p>{status_summary}</p>',
        '<a class="text-link" href="../capabilities/">Inspect the complete capability contract →</a>',
        '</article>',
        '<article role="listitem">',
        f'<h3>{len(highlights)} measured proof points</h3>',
        '<p>Every number retains its model, hardware, workload, decision, caveat, and evidence.</p>',
        '<a class="text-link" href="../benchmarks/">Explore the reviewed measurements →</a>',
        '</article>',
        '<article role="listitem">',
        f'<h3>{len(articles)} published research notes</h3>',
        '<p>Dated investigations preserve promoted, shelved, rejected, and diagnostic outcomes.</p>',
        '<a class="text-link" href="../research/">Read the evidence trail →</a>',
        '</article>',
        '<article role="listitem">',
        f'<h3>{len(releases)} reviewed release records</h3>',
        '<p>The ledger keeps exact public commits and unchanged claim boundaries attached.</p>',
        '<a class="text-link" href="../releases/">Open the release ledger →</a>',
        '</article>',
        '</div>',
        '</section>',
        '<section class="section shell" aria-labelledby="status-capabilities-heading">',
        '<div class="section-heading"><p class="eyebrow">Feature set</p><h2 id="status-capabilities-heading">State and scope travel with every capability.</h2><p class="section-intro">Implemented means source and regression contracts exist. Promoted results remain explicitly scoped; shelved work stays visible as a decision, not a default.</p></div>',
        '<ul class="status-legend" aria-label="Capability status definitions">',
    ]
    for status, label, description in CAPABILITY_STATUS_DEFINITIONS:
        body.append(
            f'<li><span class="status-badge status-{html.escape(status)}">{html.escape(label.upper())}</span><p>{html.escape(description)}</p></li>'
        )
    body.extend(['</ul>', '<div class="capability-grid">'])
    for capability in capabilities:
        status = str(capability["status"])
        body.extend(
            [
                '<article class="capability-card" data-status-capability="'
                + html.escape(str(capability["id"]), quote=True)
                + '" data-capability-state="'
                + html.escape(status, quote=True)
                + '">',
                f'<span class="status-badge status-{html.escape(status)}">{html.escape(status_labels[status].upper())}</span>',
                f'<h3>{html.escape(str(capability["name"]))}</h3>',
                f'<p>{html.escape(str(capability["summary"]))}</p>',
                f'<p class="scope-note"><strong>Scope:</strong> {html.escape(str(capability["scope"]))}</p>',
                '<div class="evidence-links" aria-label="Published evidence">',
            ]
        )
        for evidence_slug in capability["evidenceSlugs"]:
            article = article_by_slug[str(evidence_slug)]
            body.append(
                '<a href="../'
                + html.escape(article.public_path, quote=True)
                + '">'
                + html.escape(article.title)
                + " →</a>"
            )
        body.extend(
            [
                '</div>',
                '<a class="text-link" href="../capabilities/'
                + html.escape(str(capability["id"]), quote=True)
                + '/">Open capability details →</a>',
                '</article>',
            ]
        )
    body.extend(
        [
            '</div>',
            '</section>',
            '<section class="section shell" aria-labelledby="status-performance-heading">',
            '<div class="section-heading"><p class="eyebrow">Measured value</p><h2 id="status-performance-heading">Performance claims keep their boundaries attached.</h2><p class="section-intro">These are dated fast-mlx measurements, not cross-model, cross-hardware, competitor, or future-version guarantees.</p></div>',
            '<div class="evidence-grid">',
        ]
    )
    for highlight in highlights:
        identifier = str(highlight["id"])
        decision = str(highlight["decision"])
        article = article_by_slug[str(highlight["evidenceSlug"])]
        body.extend(
            [
                '<article class="evidence-card" data-status-highlight="'
                + html.escape(identifier, quote=True)
                + '" data-highlight-decision="'
                + html.escape(decision, quote=True)
                + '">',
                '<div class="card-topline">'
                f'<span class="status-badge status-{html.escape(decision)}">{html.escape(status_labels[decision].upper())}</span>'
                f'<time datetime="{html.escape(str(highlight["date"]), quote=True)}">{html.escape(str(highlight["date"]))}</time>'
                '</div>',
                f'<div class="metric">{html.escape(str(highlight["metric"]))}</div>',
                f'<h3>{html.escape(str(highlight["label"]))}</h3>',
                '<dl class="evidence-context">',
                f'<div><dt>Model</dt><dd>{html.escape(str(highlight["model"]))}</dd></div>',
                f'<div><dt>Hardware</dt><dd>{html.escape(str(highlight["hardware"]))}</dd></div>',
                f'<div><dt>Workload</dt><dd>{html.escape(str(highlight["workload"]))}</dd></div>',
                '</dl>',
                f'<p class="scope-note"><strong>Boundary:</strong> {html.escape(str(highlight["caveat"]))}</p>',
                '<div class="evidence-links" aria-label="Measured evidence links">',
                '<a class="text-link" href="../benchmarks/'
                + html.escape(identifier, quote=True)
                + '/">Open the reviewed result →</a>',
                '<a class="text-link" href="../'
                + html.escape(article.public_path, quote=True)
                + '">Read '
                + html.escape(article.title)
                + " →</a>",
                '</div>',
                '</article>',
            ]
        )
    body.extend(
        [
            '</div>',
            '</section>',
            '<section class="section shell split" aria-labelledby="status-release-heading">',
            '<div>',
            '<p class="eyebrow">Latest reviewed release record</p>',
            '<div class="card-topline"><span class="status-badge status-released">'
            + html.escape(str(latest["state"]).upper())
            + '</span> <time datetime="'
            + html.escape(latest_published_at, quote=True)
            + '">'
            + html.escape(latest_published_at[:10])
            + '</time></div>',
            f'<h2 id="status-release-heading">{html.escape(str(latest["title"]))}</h2>',
            f'<p>{html.escape(str(latest["summary"]))}</p>',
            f'<p class="scope-note"><strong>Boundary:</strong> {html.escape(str(latest["scope"]))}</p>',
            '<div class="release-links">',
            '<a class="text-link" href="../releases/'
            + html.escape(latest_id, quote=True)
            + '/">Inspect this reviewed milestone →</a>',
            '<a href="https://github.com/bitworks-io/fast-mlx/commit/'
            + html.escape(latest_commit, quote=True)
            + '" rel="noreferrer">Inspect commit '
            + html.escape(latest_commit[:12])
            + " →</a>",
            '</div>',
            '</div>',
            '<article class="callout">',
            f'<span class="status-badge status-gated">{html.escape(str(boundary["state"]).upper())}</span>',
            f'<h3>{html.escape(str(boundary["label"]))}</h3>',
            f'<p>{html.escape(str(boundary["summary"]))}</p>',
            '<a class="text-link" href="../'
            + html.escape(str(boundary_evidence["path"]), quote=True)
            + '">'
            + html.escape(str(boundary_evidence["label"]))
            + " →</a>",
            '</article>',
            '</section>',
            '<section class="section shell callout" aria-labelledby="status-contract-heading">',
            '<p class="eyebrow">Machine-readable source</p>',
            '<h2 id="status-contract-heading">Inspect the exact records behind this view.</h2>',
            '<p>The capability, release, and research indexes remain the machine-readable contracts. This page performs no live lookup, ranking, aggregation, benchmark execution, or authority transition.</p>',
            '<div class="hero-actions">',
            '<a class="text-link" href="../capabilities/index.json">Capabilities JSON →</a>',
            '<a class="text-link" href="../releases/index.json">Releases JSON →</a>',
            '<a class="text-link" href="../research/index.json">Research JSON →</a>',
            '</div>',
            '</section>',
            '</div>',
        ]
    )
    return "\n".join(body)


def build_site(repository_root: Path, output: Path) -> List[Article]:
    articles = load_articles(repository_root)
    catalog = load_capability_catalog(repository_root, {article.slug for article in articles})
    release_catalog = load_release_catalog(repository_root)
    quality_manifest = load_quality_guides(repository_root)
    quality_available = quality_manifest is not None
    served_ledger = load_served_benchmarks(repository_root, quality_manifest)
    served_cards_by_id: Dict[str, Dict[str, object]] = {
        str(card["id"]): card
        for card in quality_manifest["cards"]
        if isinstance(card, dict) and "id" in card
    }
    template = (repository_root / "site/templates/base.html").read_text(encoding="utf-8")
    assets = repository_root / "site/assets"
    validate_asset_tree(assets)
    shutil.copytree(assets, output / "assets", dirs_exist_ok=True)

    def render_page(*args: object, **kwargs: object) -> str:
        kwargs.setdefault("quality_nav_available", quality_available)
        return render_template(*args, **kwargs)  # type: ignore[arg-type]

    home = (repository_root / "site/fragments/home.html").read_text(encoding="utf-8")
    if home.count("{{current_cycle}}") != 1:
        fail("site/fragments/home.html must contain exactly one current-cycle slot")
    home = home.replace(
        "{{current_cycle}}",
        render_home_current_cycle(catalog, articles, release_catalog),
    )
    write_page(
        output,
        "index.html",
        render_page(
            template,
            "fast-mlx — measured quality for MLX on Apple Silicon",
            "Measures what speed and memory settings cost in output quality for MLX models on Apple Silicon, sizes models before loading, and publishes the evidence from a review-gated research loop.",
            "",
            home,
            public_path="",
        ),
    )

    write_page(
        output,
        "status/index.html",
        render_page(
            template,
            "Current status — fast-mlx",
            "A manifest-derived view of fast-mlx capabilities, measured proof points, reviewed releases, research, and unchanged authority boundaries.",
            "../",
            render_status_page(catalog, articles, release_catalog),
            "status",
            public_path="status/",
        ),
    )

    for name, title, description in (
        (
            "quickstart",
            "Operator quickstart — fast-mlx",
            "Run fast-mlx's model-free HTTP/JSON and HTTP/SSE transport smoke, inspect capacity, and understand the loaded-serving boundary.",
        ),
        (
            "license",
            "Apache-2.0 license — fast-mlx",
            "Commercial-use, proprietary-extension, redistribution, notice, patent, trademark, and third-party boundaries for the fast-mlx public source.",
        ),
        (
            "process",
            "The improvement loop — fast-mlx",
            "How fast-mlx turns research into reviewed, testable inference capabilities.",
        ),
        (
            "methodology",
            "Methodology — fast-mlx",
            "The correctness, comparability, and public-claim boundaries behind fast-mlx results.",
        ),
    ):
        fragment = (repository_root / f"site/fragments/{name}.html").read_text(encoding="utf-8")
        write_page(
            output,
            f"{name}/index.html",
            render_page(
                template,
                title,
                description,
                "../",
                fragment,
                name,
                public_path=f"{name}/",
            ),
        )

    capability_body, capability_index = render_capability_catalog(catalog, articles)
    write_page(
        output,
        "capabilities/index.html",
        render_page(
            template,
            "Capabilities & evidence — fast-mlx",
            "A status-aware inventory of fast-mlx features and scoped measured results.",
            "../",
            capability_body,
            "capabilities",
            public_path="capabilities/",
        ),
    )
    write_page(
        output,
        "capabilities/index.json",
        json.dumps(capability_index, indent=2, ensure_ascii=False),
    )
    for capability in capability_index["capabilities"]:
        identifier = str(capability["id"])
        public_path = f"capabilities/{identifier}/"
        write_page(
            output,
            public_path + "index.html",
            render_page(
                template,
                capability_detail_title(capability),
                capability_detail_description(capability),
                "../../",
                render_capability_detail(capability),
                "capabilities",
                public_path=public_path,
            ),
        )
    capability_detail_lines = "\n".join(
        f'- /capabilities/{capability["id"]}/: {capability["name"]}'
        for capability in capability_index["capabilities"]
    )
    benchmark_body = render_benchmark_explorer(
        capability_index["performanceHighlights"],
        served_ledger["entries"],
        served_cards_by_id,
    )
    write_page(
        output,
        "benchmarks/index.html",
        render_page(
            template,
            "Benchmark explorer — fast-mlx",
            "Filter reviewed fast-mlx measurements without separating results from their scope, caveats, or evidence.",
            "../",
            benchmark_body,
            "benchmarks",
            '<script src="../assets/benchmark-explorer.js" defer></script>',
            public_path="benchmarks/",
        ),
    )
    write_page(
        output,
        "benchmarks/served-benchmarks.json",
        json.dumps(served_ledger, indent=2, ensure_ascii=False),
    )
    for entry in served_ledger["entries"]:
        identifier = str(entry["id"])
        row_source = repository_root / "site" / SERVED_BENCHMARK_ROWS_DIRNAME / f"{identifier}.json"
        row_destination = output / "benchmarks" / SERVED_BENCHMARK_ROWS_DIRNAME / f"{identifier}.json"
        row_destination.parent.mkdir(parents=True, exist_ok=True)
        # Byte-for-byte copy, never re-serialized: `rowSha256` (already
        # verified against this exact source file by `load_served_benchmarks`
        # -> `validate_served_benchmark_row_file`) must keep hashing the
        # PUBLISHED bytes, not a re-encoded copy that could drift from them.
        row_destination.write_bytes(row_source.read_bytes())
    for highlight in capability_index["performanceHighlights"]:
        identifier = str(highlight["id"])
        public_path = f"benchmarks/{identifier}/"
        write_page(
            output,
            public_path + "index.html",
            render_page(
                template,
                benchmark_detail_title(highlight),
                benchmark_detail_description(),
                "../../",
                render_benchmark_detail(highlight),
                "benchmarks",
                public_path=public_path,
            ),
        )
    if quality_manifest is not None:
        write_page(
            output,
            "quality/index.html",
            render_page(
                template,
                "Quality guide — fast-mlx",
                "Plain-language fast-mlx quality cards: what a lower-bit or enhanced config "
                "costs on this model, with raw metrics behind an expander.",
                "../",
                render_quality_guide(quality_manifest["cards"]),
                "quality",
                public_path="quality/",
            ),
        )
        write_page(
            output,
            "quality/index.json",
            json.dumps(quality_manifest, indent=2, ensure_ascii=False),
        )
    release_body, release_index = render_release_catalog(release_catalog)
    write_page(
        output,
        "releases/index.html",
        render_page(
            template,
            "Releases — fast-mlx",
            "A reviewed ledger of fast-mlx public milestones, exact commits, shipped surfaces, and unchanged boundaries.",
            "../",
            release_body,
            "releases",
            public_path="releases/",
        ),
    )
    write_page(
        output,
        "releases/index.json",
        json.dumps(release_index, indent=2, ensure_ascii=False),
    )
    write_page(output, "releases/feed.atom", render_release_feed(release_index))
    for release in release_index["releases"]:
        identifier = str(release["id"])
        public_path = f"releases/{identifier}/"
        write_page(
            output,
            public_path + "index.html",
            render_page(
                template,
                release_detail_title(release),
                RELEASE_DETAIL_DESCRIPTION,
                "../../",
                render_release_detail(release),
                "releases",
                public_path=public_path,
            ),
        )

    write_page(
        output,
        "research/index.html",
        render_page(
            template,
            "Research notes — fast-mlx",
            "Dated fast-mlx investigations and measured negative results.",
            "../",
            render_research_archive(articles),
            "research",
            '<script src="../assets/research-explorer.js" defer></script>',
            public_path="research/",
        ),
    )

    article_routes = {Path(article.source_name).name: article.public_path for article in articles}
    for article in articles:
        markdown_html = render_markdown(article.body, article_routes, article.output_file)
        article_body = (
            '<article class="article shell">'
            '<header class="article-header">'
            f'<div class="article-meta">{html.escape(article.date)} · {html.escape(article.theme)}</div>'
            f'<h1>{html.escape(article.title)}</h1>'
            f'<p class="article-summary">{html.escape(article.summary)}</p>'
            "</header>"
            f'<div class="article-body">{markdown_html}</div>'
            "</article>"
        )
        write_page(
            output,
            article.output_file,
            render_page(
                template,
                f"{article.title} — fast-mlx",
                article.summary,
                "../../",
                article_body,
                "research",
                public_path=article.public_path,
                article_section=article.theme,
            ),
        )

    public_index = {
        "schemaVersion": 1,
        "project": "fast-mlx",
        "claimBoundary": "fast-mlx-owned-results-only",
        "articles": [
            {
                "title": article.title,
                "date": article.date,
                "theme": article.theme,
                "summary": article.summary,
                "path": article.public_path,
                "reviewedAt": article.reviewed_at,
            }
            for article in articles
        ],
    }
    write_page(output, "research/index.json", json.dumps(public_index, indent=2, ensure_ascii=False))
    write_page(output, "research/feed.atom", render_research_feed(articles))
    write_page(
        output,
        "feed.atom",
        render_reviewed_updates_feed(release_index, articles),
    )
    write_page(
        output,
        "sitemap.xml",
        render_sitemap(
            articles,
            capability_index["capabilities"],
            capability_index["performanceHighlights"],
            release_index["releases"],
        ),
    )
    benchmark_detail_lines = "\n".join(
        f'- /benchmarks/{highlight["id"]}/: {highlight["label"]}'
        for highlight in capability_index["performanceHighlights"]
    )
    release_detail_lines = "\n".join(
        f'- /releases/{release["id"]}/: {release["title"]}'
        for release in release_index["releases"]
    )
    write_page(
        output,
        "llms.txt",
        "# fast-mlx\n\n"
        "Measured-quality tooling and evidence-gated Swift/MLX inference research for Apple Silicon.\n\n"
        "## Core pages\n"
        "- /quickstart/: model-free HTTP/SSE operator quickstart\n"
        "- /license/: Apache-2.0 commercial-use and redistribution orientation\n"
        "- /status/: reviewed current-state dashboard\n"
        "- /process/: research-to-publication loop\n"
        "- /methodology/: correctness and claim boundaries\n"
        "- /capabilities/: status-aware feature and evidence inventory\n"
        "- /capabilities/index.json: machine-readable capability contract\n"
        + capability_detail_lines
        + "\n"
        "- /benchmarks/: filterable reviewed performance evidence\n"
        + benchmark_detail_lines
        + "\n"
        "- /benchmarks/served-benchmarks.json: machine-readable served-engine ledger\n"
        "- /releases/: reviewed public milestones and unchanged boundaries\n"
        + release_detail_lines
        + "\n"
        "- /releases/index.json: machine-readable release ledger\n"
        "- /releases/feed.atom: Atom feed of reviewed public milestones\n"
        "- /research/feed.atom: Atom feed of reviewed research notes\n"
        "- /feed.atom: combined Atom feed of all reviewed releases and research notes\n"
        "- /sitemap.xml: reviewed human-facing page inventory\n"
        "- /robots.txt: crawl policy and sitemap location\n"
        "- /research/: dated technical notes\n\n"
        "## Published notes\n"
        + "\n".join(f"- /{article.public_path}: {article.title}" for article in articles),
    )
    write_page(output, "robots.txt", render_robots())
    write_page(
        output,
        "404.html",
        render_page(
            template,
            "Not found — fast-mlx",
            "The requested fast-mlx page was not found.",
            "",
            '<section class="page-hero shell narrow"><p class="eyebrow">404</p><h1>That evidence path is not here.</h1><p class="lede">Return to the <a href="./">fast-mlx home page</a>.</p></section>',
        ),
    )
    (output / ".nojekyll").write_text("", encoding="utf-8")
    return articles


def scan_generated_output(output: Path) -> None:
    validate_social_card(output / SOCIAL_CARD_PATH, "generated social card")
    for path in output.rglob("*"):
        if path.is_symlink():
            fail(f"generated site contains a symlink: {path}")
        if not path.is_file():
            continue
        if path.relative_to(output).as_posix() == SOCIAL_CARD_PATH:
            continue
        text = path.read_text(encoding="utf-8")
        for marker in PRIVATE_MARKERS:
            if marker.casefold() in text.casefold():
                fail(f"generated site contains private marker {marker!r}: {path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_arguments(argv)
    repository_root = arguments.repository_root.resolve()
    output = arguments.output.resolve()
    prepare_output(output, repository_root)
    articles = build_site(repository_root, output)
    scan_generated_output(output)
    print(f"built public site: articles={len(articles)} output={output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
