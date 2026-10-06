#!/usr/bin/env python3
"""Validate generated fast-mlx Pages files and internal links."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html.parser
import json
import math
import posixpath
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlsplit

import validate_public_repository

# Single source of truth for private markers -- see build_public_site.py's
# identical import for why this must not be a locally duplicated tuple.
PRIVATE_MARKERS: Tuple[str, ...] = validate_public_repository.PRIVATE_MARKERS
CAPABILITY_STATUSES = {"implemented", "promoted-scoped", "experimental", "shelved"}
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
QUALITY_CARD_PROVENANCE_ALLOWED_KEYS = QUALITY_CARD_PROVENANCE_REQUIRED_KEYS | {"engineBuild"}
QUALITY_CARD_ENGINE_BUILD_COMMIT = re.compile(r"[0-9a-f]{40}")
QUALITY_CARD_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*@[a-z0-9]+(?:-[a-z0-9]+)*")
# config.hardwareClass: the SAME string `fastmlx_launch.host_hardware_class()`
# produces (e.g. "apple-m3-ultra") -- lowercase alphanumeric segments joined
# by single hyphens. A value outside this shape can never host-match by
# construction (see `fastmlx_launch.host_hardware_class()`'s docstring), so
# it is refused here rather than silently never firing at serve time.
#
# This is a deliberate separate copy, not an import of either sibling --
# see the "projection split" note in
# docs/task-inbox/2026-09-22-PREDECLARATION-hardware-class-must-be-canonical.md.
# Must stay byte-consistent with the copy in `build_public_site.py` and
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
# Must stay byte-consistent with the copy in `build_public_site.py`.
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
# Must stay byte-consistent with the copy in `build_public_site.py`.
QUALITY_CARD_SPEEDX_UNMEASURED_PREFIX = "decode throughput (speedX"
QUALITY_GUIDE_PUBLIC_FILE = "quality/index.html"
CAPABILITY_STATUS_LABELS = {
    "implemented": "Implemented",
    "promoted-scoped": "Promoted · scoped",
    "experimental": "Experimental",
    "shelved": "Shelved",
}
CAPABILITY_STATUS_DESCRIPTIONS = {
    "implemented": (
        "The public source and regression contracts exist; this is not "
        "automatically a supported default."
    ),
    "promoted-scoped": (
        "A bounded route or result crossed its stated evidence gates only for the "
        "named scope."
    ),
    "experimental": (
        "The surface is active research and has not earned a production support "
        "claim."
    ),
    "shelved": (
        "The dated result remains useful evidence, but the capability is not the "
        "current production route."
    ),
}
REVIEWED_CAPABILITIES: Tuple[Dict[str, object], ...] = (
    {
        "id": "openai-http-sse-serving",
        "name": "OpenAI-compatible HTTP/SSE serving",
        "status": "promoted-scoped",
        "summary": "A chat-completions transport with streaming, bounded admission, cancellation, model- and host-fit-aware context/completion limits, authenticated capability discovery, and evidence output.",
        "scope": "The explicit temperature-zero continuous-batch-no-spec route is qualified only for the published source-locked Qwen3-32B-4bit workload. Model-aware limits are implemented and regression-tested across scalar, continuous, exact-MTP, and fallback routes; they are capacity controls, not a model-quality, speed, or universal native-context claim. Scripted mode is transport-only, and no model weights are bundled.",
        "evidenceSlugs": (
            "the-proof-did-not-end-when-the-timer-did",
            "the-4k-limit-was-not-the-model-limit",
        ),
    },
    {
        "id": "exact-continuous-batching",
        "name": "Exact continuous batching",
        "status": "promoted-scoped",
        "summary": "Simultaneous dense-model streams can join, advance, cancel, and release reservations through one explicit no-spec route, with a default-off adaptive solo-to-batch transition.",
        "scope": "The measured policy is a dense-Qwen building block for the named paired workloads. Adaptive late join is an explicit control with regression coverage; it is not a sampled-generation route, an unqualified automatic default, or a general model-family claim.",
        "evidenceSlugs": (
            "the-fastest-request-wasnt-the-fastest-service",
            "the-proof-did-not-end-when-the-timer-did",
        ),
    },
    {
        "id": "deterministic-sampled-generation-foundation",
        "name": "Deterministic sampled-generation foundation",
        "status": "implemented",
        "summary": "A dependency-free CPU oracle defines seeded temperature and top-p token selection with stable counter addressing.",
        "scope": "Internal HarnessCore foundation only. It is not wired to HTTP requests, the serving scheduler, MLX arrays, models, tokenizers, or automatic entropy, and it makes no performance claim.",
        "evidenceSlugs": ("sampling-before-serving",),
    },
    {
        "id": "prompt-lookup-decoding",
        "name": "Prompt-lookup decoding",
        "status": "shelved",
        "summary": "Temperature-zero decoding can verify repeated context spans with byte-identical output on the measured repetition-heavy workload.",
        "scope": "This is a dated solo-path result. Dynamic PLD is not the current production route and remains disabled inside shared continuous batches.",
        "evidenceSlugs": ("when-zero-speculation-costs-two-percent",),
    },
    {
        "id": "exact-cache-lifecycle-controls",
        "name": "Exact cache and lifecycle controls",
        "status": "implemented",
        "summary": "Prefix/session-cache primitives, byte-denominated admission, cancellation, recovery, and reservation release are explicit contracts.",
        "scope": "Source and regression contracts exist. This inventory makes no standalone cache-hit, speedup, or broad architecture claim.",
        "evidenceSlugs": (
            "the-fastest-request-wasnt-the-fastest-service",
            "the-proof-did-not-end-when-the-timer-did",
        ),
    },
    {
        "id": "quality-measurement-harness",
        "name": "Quality and exactness measurement harness",
        "status": "implemented",
        "summary": "Teacher-forced distribution drift, perplexity, tail behavior, exact replay, task checks, throughput, memory, and soak health share one evidence workflow.",
        "scope": "Each measurement remains source-, model-, workload-, and instrument-scoped. A metric value is not a universal quality guarantee.",
        "evidenceSlugs": ("trusting-the-instrument", "the-wall-that-wasnt"),
    },
    {
        "id": "capacity-proof-control-tools",
        "name": "Capacity and proof-control tools",
        "status": "implemented",
        "summary": "Command-line planning, explicit shared or dedicated host profiles, auditable v2 capacity artifacts, and dry-run-only dedicated qualification plans make resource and identity assumptions inspectable.",
        "scope": "Shared or unspecified hosts use a conservative 75% ceiling; dedicated mode requires an explicit reserve. Qualification plans record staged safety gates but grant no privilege and perform no host mutation. Planning and comparison tools do not prove that a model fits every Mac, establish launchability, or prove runtime containment.",
        "evidenceSlugs": ("the-wall-that-wasnt", "lossless-wasnt-byte-identical"),
    },
    {
        "id": "openai-tool-calling",
        "name": "OpenAI-compatible tool calling",
        "status": "implemented",
        "summary": "Function/tool calling on the chat-completions route for the dense Qwen3 family: tools and tool_choice in, OpenAI tool_calls out (streaming and non-streaming), with multi-turn tool results rendered through Qwen3's own chat template.",
        "scope": "Request/response contract, Hermes tool-call parsing, multi-turn chat-template rendering (verified against the real Qwen3 tokenizer), streaming tool-call deltas, and tool_choice resolution are covered by tests without a model. Verified end-to-end on a live served dense Qwen3-8B locally, and independently validated on a production M5 128GB with dense Qwen3-32B-8bit (correct OpenAI tool_calls, the identity verification gate holding without data leak, and the continuous-batch route engaging under concurrency). Tool requests default thinking off for reliability. Targets dense Qwen3 (8B / 32B); the qwen3_5 hybrid line is out of scope.",
        "evidenceSlugs": ("the-tool-call-we-could-prove-without-the-model",),
    },
    {
        "id": "model-sizer",
        "name": "Hardware model sizer",
        "status": "implemented",
        "summary": "Given a Mac's RAM, reports which quantized model builds fit and at what context — reusing the capacity memory model — so an operator can choose a build before downloading it.",
        "scope": "Estimates weight, KV-cache, and prefill memory per model and quantization against a detected or preset machine, plus the largest context that fits. Hand-measured boxes report measured headroom; auto-detected machines report a synthesized wired-memory limit, and every such row is flagged as a modeled estimate, not a measured guarantee.",
        "evidenceSlugs": ("the-checkout-that-couldnt-compile-its-gpu",),
    },
)
REVIEWED_CAPABILITY_PATHS = tuple(
    f'capabilities/{capability["id"]}/' for capability in REVIEWED_CAPABILITIES
)
HIGHLIGHT_DECISION_LABELS = {
    "promoted-scoped": "Promoted · scoped",
    "shelved": "Shelved",
}
BENCHMARK_FILTER_NAMES = ("model", "hardware", "decision")
RELEASE_CATEGORIES = {"foundation", "operations", "product"}
RELEASE_CATEGORY_LABELS = {
    "foundation": "Foundation",
    "operations": "Operations",
    "product": "Product",
}
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
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
SITE_STYLESHEET_PATH = "assets/site.css"
SITE_STYLESHEET_SHA256 = (
    "70fa7d0023bc818143345cff859982b30ea053a3c91069283773a3a7a035764b"
)
RESEARCH_EXPLORER_SCRIPT_PATH = "assets/research-explorer.js"
RESEARCH_EXPLORER_SCRIPT_SHA256 = (
    "cb75f437a56eafc49ce3d0d692183d6f001d4cb8d6cc16df6c66635ce6beb9c2"
)
REVIEWED_HOME_PAGE_BYTES = 11_227
REVIEWED_HOME_PAGE_SHA256 = (
    "f1acc3b606ea362ab41bd03f8f08a8cf7cfa0a53d4eceefe2da3d6b8163a4026"
)
SOCIAL_CARD_BYTES = 1_011_297
SOCIAL_CARD_WIDTH = 1_200
SOCIAL_CARD_HEIGHT = 630
MAX_RELEASE_FEED_BYTES = 1_048_576
MAX_RESEARCH_FEED_BYTES = 1_048_576
MAX_REVIEWED_UPDATES_FEED_BYTES = 1_048_576
MAX_RESEARCH_INDEX_BYTES = 1_048_576
MAX_SITEMAP_BYTES = 1_048_576
MAX_ROBOTS_BYTES = 4_096
MAX_QUICKSTART_BYTES = 131_072
MAX_LICENSE_PAGE_BYTES = 131_072
MAX_STATUS_BYTES = 131_072
MAX_CAPABILITY_CATALOG_BYTES = 131_072
MAX_CAPABILITY_DETAIL_BYTES = 131_072
REVIEWED_QUICKSTART_PAGE_BYTES = 10_009
REVIEWED_QUICKSTART_PAGE_SHA256 = (
    "bd5d84a28a14daacfd1895b75b1175625187e1603dac124cfcb7f0d4cefa2b6f"
)
REVIEWED_LICENSE_PAGE_BYTES = 7_323
REVIEWED_LICENSE_PAGE_SHA256 = (
    "5413029327e71b5472ae598279b119da2e32ece334f791c7295aa0afc638372b"
)
REVIEWED_STATUS_PAGE_BYTES = 23_554
REVIEWED_STATUS_PAGE_SHA256 = (
    "25278505806ec2ab7663cdff7c9e6f8afe6ee65b6371fadff20d736699bec2cb"
)
REVIEWED_CAPABILITIES_PAGE_BYTES = 18_032
REVIEWED_CAPABILITIES_PAGE_SHA256 = (
    "61cc7bc395910f139222934a951ade9d5593b6778b9d5cfb14d6944f8587570b"
)
REVIEWED_CAPABILITY_DETAIL_SEALS: Dict[str, Tuple[int, str]] = {
    "openai-http-sse-serving": (
        5_849,
        "753d212c07fb57a0c6e20be7fa8738d225106392f5dfa19dd707ee694aeb8be6",
    ),
    "exact-continuous-batching": (
        5_685,
        "936b54ca0831bda1d1d60b6ed47d593e5e087f3ddc6bf3361a81bb52feb37ca7",
    ),
    "deterministic-sampled-generation-foundation": (
        5_292,
        "a6119c5e3c2925612b639f956ca63f9f14405f01c77d4d4e697b9577175e13a7",
    ),
    "prompt-lookup-decoding": (
        5_131,
        "f73b2ef019334abe92e4a24a8fed6630b8f99d489182f4802d903b5cbb23ec77",
    ),
    "exact-cache-lifecycle-controls": (
        5_566,
        "20916c9bbd42572fce804ca1aaaef812951af60da307e12aa16bfb1d9885113c",
    ),
    "quality-measurement-harness": (
        5_589,
        "5af2a4a2b2941368c0b6521de0a18bd2d5bf3fdf33d3a246f616e4065bb523c7",
    ),
    "capacity-proof-control-tools": (
        5_833,
        "b189d8dddb5dcb524231051792724b50b769fdb5b7c6c2c7a839f705cdbcb96a",
    ),
    "openai-tool-calling": (
        5_778,
        "b1287b4f9190529756b4d76941092e579118871178ac918d4ade03deb9e6ef33",
    ),
    "model-sizer": (
        5_341,
        "186f8c1c413ccd1fcd503ccda855538885cdc69f830fa05bb7f4df5ea90e9823",
    ),
}
REVIEWED_QUICKSTART_COMMANDS: Tuple[Tuple[str, str], ...] = (
    (
        "clone",
        "git clone https://github.com/bitworks-io/fast-mlx.git\ncd fast-mlx",
    ),
    (
        "serve-scripted",
        "swift run --package-path spike fastmlx-serve --scripted",
    ),
    (
        "request-json",
        "curl http://127.0.0.1:8080/v1/chat/completions \\\n"
        "  -H 'content-type: application/json' \\\n"
        "  -d '{\"model\":\"fastmlx-scripted\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"temperature\":0,\"n\":1,\"stream\":false}'",
    ),
    (
        "request-sse",
        "curl -N http://127.0.0.1:8080/v1/chat/completions \\\n"
        "  -H 'content-type: application/json' \\\n"
        "  -d '{\"model\":\"fastmlx-scripted\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"temperature\":0,\"n\":1,\"stream\":true}'",
    ),
    ("capacity", "swift run --package-path spike fastmlx-capacity"),
    ("serve-help", "swift run --package-path spike fastmlx-serve --help"),
)
REVIEWED_QUICKSTART_TEXT = (
    "Apple Silicon Mac",
    "macOS 14 or newer",
    "Swift 6",
    "Scripted mode loads no model",
    "open another terminal",
    "POST /v1/chat/completions",
    "application/json",
    "text/event-stream",
    "FASTMLX_API_KEY",
    "temperature zero",
    "n = 1",
    "No model weights are bundled",
    "does not prove model compatibility, output quality, capacity fit, or performance",
)
REVIEWED_QUICKSTART_LINKS = (
    "https://github.com/bitworks-io/fast-mlx",
    "../capabilities/",
    "../benchmarks/",
    "../methodology/",
)
QUICKSTART_ALLOWED_TAGS = {
    "a",
    "article",
    "code",
    "div",
    "h1",
    "h2",
    "h3",
    "li",
    "ol",
    "p",
    "pre",
    "section",
    "span",
    "strong",
}
STATUS_ALLOWED_TAGS = {
    "a",
    "article",
    "dd",
    "div",
    "dl",
    "dt",
    "h1",
    "h2",
    "h3",
    "li",
    "p",
    "section",
    "span",
    "strong",
    "time",
    "ul",
}
REVIEWED_STATUS_CAPABILITIES: Tuple[Tuple[str, str], ...] = (
    ("openai-http-sse-serving", "promoted-scoped"),
    ("exact-continuous-batching", "promoted-scoped"),
    ("deterministic-sampled-generation-foundation", "implemented"),
    ("prompt-lookup-decoding", "shelved"),
    ("exact-cache-lifecycle-controls", "implemented"),
    ("quality-measurement-harness", "implemented"),
    ("capacity-proof-control-tools", "implemented"),
    ("openai-tool-calling", "implemented"),
    ("model-sizer", "implemented"),
)
REVIEWED_STATUS_COUNTS: Tuple[Tuple[str, str], ...] = (
    ("implemented", "6"),
    ("promoted-scoped", "2"),
    ("experimental", "0"),
    ("shelved", "1"),
)
REVIEWED_STATUS_LINKS = (
    "../quickstart/",
    "../methodology/",
    "../capabilities/",
    "../benchmarks/",
    "../research/",
    "../releases/",
    "../research/the-proof-did-not-end-when-the-timer-did/",
    "../research/the-4k-limit-was-not-the-model-limit/",
    "../capabilities/openai-http-sse-serving/",
    "../research/the-fastest-request-wasnt-the-fastest-service/",
    "../research/the-proof-did-not-end-when-the-timer-did/",
    "../capabilities/exact-continuous-batching/",
    "../research/sampling-before-serving/",
    "../capabilities/deterministic-sampled-generation-foundation/",
    "../research/when-zero-speculation-costs-two-percent/",
    "../capabilities/prompt-lookup-decoding/",
    "../research/the-fastest-request-wasnt-the-fastest-service/",
    "../research/the-proof-did-not-end-when-the-timer-did/",
    "../capabilities/exact-cache-lifecycle-controls/",
    "../research/trusting-the-instrument/",
    "../research/the-wall-that-wasnt/",
    "../capabilities/quality-measurement-harness/",
    "../research/the-wall-that-wasnt/",
    "../research/lossless-wasnt-byte-identical/",
    "../capabilities/capacity-proof-control-tools/",
    "../research/the-tool-call-we-could-prove-without-the-model/",
    "../capabilities/openai-tool-calling/",
    "../research/the-checkout-that-couldnt-compile-its-gpu/",
    "../capabilities/model-sizer/",
    "../benchmarks/pld-echo-throughput/",
    "../research/when-zero-speculation-costs-two-percent/",
    "../benchmarks/continuous-batch-c2-throughput/",
    "../research/the-fastest-request-wasnt-the-fastest-service/",
    "../benchmarks/http-sse-operational-soak/",
    "../research/the-proof-did-not-end-when-the-timer-did/",
    "../releases/tagged-distribution-v0-1-10/",
    "https://github.com/bitworks-io/fast-mlx/commit/c5be493f132c618f7c083dfd73c5157d938791fb",
    "../methodology/",
    "../capabilities/index.json",
    "../releases/index.json",
    "../research/index.json",
)
REVIEWED_STATUS_TEXT = (
    "Current state, not a roadmap.",
    "does not create new measurement, performance, model, runtime, acquisition, or publication authority",
    "9 reviewed capabilities",
    "3 measured proof points",
    "32 published research notes",
    "33 reviewed release records",
    "Released source and comparison evidence do not grant unreviewed model, acquisition, launchability, containment, or runtime authority.",
    "This page performs no live lookup, ranking, aggregation, benchmark execution, or authority transition.",
)
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
REVIEWED_BENCHMARK_HIGHLIGHTS: Tuple[Dict[str, object], ...] = (
    {
        "id": "pld-echo-throughput",
        "metric": "+100.5%",
        "label": "Repetition-heavy solo PLD, 28.28 → 56.70 tok/s",
        "model": "Qwen3-32B-4bit",
        "hardware": "Apple M5 Max",
        "workload": (
            "Preamble-then-echo; 256 generated tokens; three post-warmup runs; "
            "temperature zero"
        ),
        "date": "2026-07-11",
        "decision": "shelved",
        "caveat": (
            "The 120-token streams were byte-identical, but dynamic PLD is not the "
            "current production route and remains disabled inside shared batches."
        ),
        "evidence": {
            "slug": "when-zero-speculation-costs-two-percent",
            "title": "When zero speculation costs 2%: making a 2× decoder safe to leave on",
            "path": "research/when-zero-speculation-costs-two-percent/",
            "reviewedAt": "2026-08-06",
        },
    },
    {
        "id": "continuous-batch-c2-throughput",
        "metric": "+45.8%",
        "label": "Aggregate C=2 service rate, 29.29 → 42.70 tok/s",
        "model": "Qwen3-32B-4bit",
        "hardware": "Apple M5 Max",
        "workload": (
            "Paired simultaneous burst; one warmup dropped; three measured Release "
            "repetitions; temperature zero"
        ),
        "date": "2026-07-14",
        "decision": "promoted-scoped",
        "caveat": (
            "The narrow dense-Qwen crossover starts at concurrency two for this "
            "workload. No automatic router, sampled route, or broad default was "
            "established."
        ),
        "evidence": {
            "slug": "the-fastest-request-wasnt-the-fastest-service",
            "title": "The fastest request wasn't the fastest service",
            "path": "research/the-fastest-request-wasnt-the-fastest-service/",
            "reviewedAt": "2026-08-11",
        },
    },
    {
        "id": "http-sse-operational-soak",
        "metric": "24 h",
        "label": "HTTP/SSE service soak with 10,368 paired request/evidence rows",
        "model": "Qwen3-32B-4bit",
        "hardware": "Apple M3 Ultra",
        "workload": (
            "C=4 continuous-batch-no-spec; 1,727 measured cycles; ordinary streams "
            "plus pre-body disconnects"
        ),
        "date": "2026-07-28",
        "decision": "promoted-scoped",
        "caveat": (
            "This qualifies transport, cancellation, recovery, resources, and "
            "terminal evidence for the measured route. It is explicitly not a "
            "throughput result."
        ),
        "evidence": {
            "slug": "the-proof-did-not-end-when-the-timer-did",
            "title": "The proof did not end when the timer did",
            "path": "research/the-proof-did-not-end-when-the-timer-did/",
            "reviewedAt": "2026-08-06",
        },
    },
)
REVIEWED_BENCHMARK_PATHS = tuple(
    f'benchmarks/{highlight["id"]}/'
    for highlight in REVIEWED_BENCHMARK_HIGHLIGHTS
)
REVIEWED_ARTICLE_PATHS = (
    "research/streaming-the-experts-changed-the-answers/",
    "research/the-lever-was-worth-ten-percent/",
    "research/the-default-nobody-chose/",
    "research/a-ratio-is-not-a-result/",
    "research/you-had-to-load-the-model-to-learn-it-would-not-load/",
    "research/every-tool-call-took-the-path-we-never-tested/",
    "research/the-fit-check-refused-a-model-that-fit/",
    "research/a-family-name-is-not-evidence-about-a-checkpoint/",
    "research/the-4k-limit-was-not-the-model-limit/",
    "research/how-the-autonomous-loop-builds-fast-mlx/",
    "research/the-state-was-right-the-ledger-was-frozen/",
    "research/the-bytes-were-derivable-we-refused-anyway/",
    "research/measuring-the-premise-before-building-the-engine/",
    "research/the-tool-call-we-could-prove-without-the-model/",
    "research/the-checkout-that-couldnt-compile-its-gpu/",
    "research/sampling-before-serving/",
    "research/the-repository-that-could-reproduce-itself/",
    "research/the-proof-did-not-end-when-the-timer-did/",
    "research/exact-speculation-was-not-fast-enough/",
    "research/exact-prefix-cache-needs-exact-provenance/",
    "research/the-third-geometry-said-no/",
    "research/llama-ran-without-a-speed-tier/",
    "research/fifteen-times-faster-still-not-fast/",
    "research/when-smaller-kv-is-not-faster/",
    "research/the-fastest-request-wasnt-the-fastest-service/",
    "research/lossless-wasnt-byte-identical/",
    "research/when-zero-speculation-costs-two-percent/",
    "research/two-x-for-free-when-the-model-repeats-itself/",
    "research/turboquant-exact-math-still-lost/",
    "research/trusting-the-instrument/",
    "research/the-wall-that-wasnt/",
    "research/one-formula-wrong-for-a-third-of-the-catalog/",
)
REVIEWED_ARTICLE_DATES: Dict[str, Tuple[str, str]] = {
    "research/streaming-the-experts-changed-the-answers/": ("2026-09-18", "2026-09-18"),
    "research/the-lever-was-worth-ten-percent/": ("2026-09-11", "2026-09-11"),
    "research/the-default-nobody-chose/": ("2026-09-10", "2026-09-10"),
    "research/a-ratio-is-not-a-result/": ("2026-09-09", "2026-09-09"),
    "research/you-had-to-load-the-model-to-learn-it-would-not-load/": ("2026-09-08", "2026-09-08"),
    "research/every-tool-call-took-the-path-we-never-tested/": ("2026-09-08", "2026-09-08"),
    "research/the-fit-check-refused-a-model-that-fit/": ("2026-09-06", "2026-09-06"),
    "research/a-family-name-is-not-evidence-about-a-checkpoint/": ("2026-09-06", "2026-09-06"),
    "research/the-4k-limit-was-not-the-model-limit/": ("2026-08-29", "2026-08-29"),
    "research/how-the-autonomous-loop-builds-fast-mlx/": ("2026-08-22", "2026-08-22"),
    "research/the-state-was-right-the-ledger-was-frozen/": ("2026-08-20", "2026-08-22"),
    "research/the-bytes-were-derivable-we-refused-anyway/": ("2026-08-19", "2026-08-22"),
    "research/measuring-the-premise-before-building-the-engine/": ("2026-08-19", "2026-08-22"),
    "research/the-tool-call-we-could-prove-without-the-model/": ("2026-08-17", "2026-08-17"),
    "research/the-checkout-that-couldnt-compile-its-gpu/": ("2026-08-17", "2026-08-17"),
    "research/sampling-before-serving/": ("2026-08-16", "2026-08-16"),
    "research/the-repository-that-could-reproduce-itself/": ("2026-08-14", "2026-08-14"),
    "research/the-proof-did-not-end-when-the-timer-did/": ("2026-07-28", "2026-08-06"),
    "research/exact-speculation-was-not-fast-enough/": ("2026-07-24", "2026-08-22"),
    "research/exact-prefix-cache-needs-exact-provenance/": ("2026-07-24", "2026-08-22"),
    "research/the-third-geometry-said-no/": ("2026-07-23", "2026-08-22"),
    "research/llama-ran-without-a-speed-tier/": ("2026-07-23", "2026-08-22"),
    "research/fifteen-times-faster-still-not-fast/": ("2026-07-21", "2026-08-22"),
    "research/when-smaller-kv-is-not-faster/": ("2026-07-18", "2026-08-22"),
    "research/the-fastest-request-wasnt-the-fastest-service/": ("2026-07-14", "2026-08-11"),
    "research/lossless-wasnt-byte-identical/": ("2026-07-12", "2026-08-06"),
    "research/when-zero-speculation-costs-two-percent/": ("2026-07-11", "2026-08-06"),
    "research/two-x-for-free-when-the-model-repeats-itself/": ("2026-07-11", "2026-08-22"),
    "research/turboquant-exact-math-still-lost/": ("2026-07-09", "2026-08-06"),
    "research/trusting-the-instrument/": ("2026-07-09", "2026-08-06"),
    "research/the-wall-that-wasnt/": ("2026-07-09", "2026-08-06"),
    "research/one-formula-wrong-for-a-third-of-the-catalog/": ("2026-07-09", "2026-08-22"),
}
REVIEWED_RELEASE_INDEX_BYTES = 38_554
REVIEWED_RELEASE_INDEX_SHA256 = (
    "f04b4f9c00cd62ce2eefcf9d8bbb2fb5789d2e121fb41161b65ac0a85876421a"
)
REVIEWED_RELEASE_IDENTITIES: Tuple[Tuple[str, str], ...] = (
    (
        "tagged-distribution-v0-1-10",
        "Publish v0.1.10: serve and cards pull refuse a card store the built-in engine cannot decode, and an unrecognized verdict is announced",
    ),
    (
        "tagged-distribution-v0-1-9",
        "Publish v0.1.9: fastmlx cards pull fetches a digest-pinned card store, and a plain serve uses the newest verified one",
    ),
    (
        "tagged-distribution-v0-1-8",
        "Publish v0.1.8: serve pins its quality-card store by sha256 and names a card measured on another chip",
    ),
    (
        "tagged-distribution-v0-1-7",
        "Publish v0.1.7: the first near-lossless quality card admits by default",
    ),
    (
        "tagged-distribution-v0-1-6",
        "Publish v0.1.6: bench rows name what produced them",
    ),
    (
        "tagged-distribution-v0-1-5",
        "Publish v0.1.5: the quality verdict reaches the client",
    ),
    (
        "tagged-distribution-v0-1-4",
        "Publish v0.1.4: fastmlx bench ships in a release",
    ),
    (
        "tagged-distribution-v0-1-3",
        "Publish v0.1.3: the engine no longer outlives its launcher",
    ),
    (
        "tagged-distribution-v0-1-2",
        "Publish v0.1.2: provenance on every response",
    ),
    (
        "tagged-distribution-v0-1-1",
        "Publish v0.1.1: fit checks that name their own refusal",
    ),
    (
        "tagged-distribution-v0-1-0",
        "Publish v0.1.0: the first tagged distribution",
    ),
    (
        "model-aware-context-completion-budgets",
        "Publish model-aware context and completion budgets",
    ),
    (
        "qwen-gdn-launch-evidence-producer",
        "Publish Qwen GDN launch evidence producer",
    ),
    (
        "qwen-gdn-scorecard-mode-identity",
        "Bind Qwen scorecards to isolated GDN modes",
    ),
    (
        "qwen-gdn-four-projection-fusion",
        "Publish opt-in Qwen GDN projection fusion",
    ),
    (
        "dedicated-serving-qualification-planner",
        "Publish dedicated-serving qualification planning",
    ),
    (
        "seeded-sampled-mtp-diagnostic-provider",
        "Publish seeded sampled-MTP diagnostics",
    ),
    (
        "current-inference-and-capacity-foundations",
        "Publish current inference and capacity foundations",
    ),
    (
        "deterministic-sampled-generation-foundation",
        "Publish deterministic sampled-generation foundation",
    ),
    ("self-reproducing-public-source", "Publish self-reproducing public source"),
    ("reviewed-research-atom-feed", "Publish reviewed research Atom feed"),
    ("reviewed-release-detail-permalinks", "Publish reviewed release detail permalinks"),
    ("reviewed-benchmark-detail-permalinks", "Publish benchmark detail permalinks"),
    ("reviewed-home-current-cycle", "Show current reviewed cycle"),
    ("reviewed-social-metadata", "Publish reviewed social metadata"),
    ("reviewed-sitemap-discovery", "Publish reviewed sitemap discovery"),
    ("reviewed-release-atom-feed", "Publish reviewed release Atom feed"),
    ("reviewed-release-ledger", "Publish reviewed release ledger"),
    ("public-benchmark-explorer", "Publish reviewed benchmark explorer"),
    ("same-commit-pages-quality-gate", "Gate Pages deployment on public quality"),
    ("capabilities-and-evidence", "Publish capabilities and evidence"),
    ("compatible-hosted-swift-runner", "Use compatible GitHub macOS runner"),
    ("initial-public-release", "Initial public release"),
)
REVIEWED_RELEASE_PATHS = tuple(
    f"releases/{identifier}/" for identifier, _title in REVIEWED_RELEASE_IDENTITIES
)
REVIEWED_RELEASE_DETAIL_SEALS: Dict[str, Tuple[int, str]] = {
    "tagged-distribution-v0-1-10": (
        5_622,
        "761729d5f862c57538d5e44d2b31a560cfddd5e1eda00a39bde9394971feec40",
    ),
    "tagged-distribution-v0-1-9": (
        5_352,
        "f5ba93bee8285f09de21548dc093042a66c6dd858c6c1fe40950972a8ee22cf2",
    ),
    "tagged-distribution-v0-1-8": (
        5_333,
        "96283bce82dac1fd3792be222b86f6420bc813a5baf7115f698b5f4e389da37c",
    ),
    "tagged-distribution-v0-1-7": (
        5_187,
        "e8366260ae5da2ae7ed91f1b4ab89b2402ba13ae31edc5c018278c1f7c7b6be3",
    ),
    "tagged-distribution-v0-1-6": (
        5_214,
        "5a8630ea51a6901384db5541c1bb74c0a74782b0613abec9d3d9cc7f1c2f9d58",
    ),
    "tagged-distribution-v0-1-5": (
        4_977,
        "9cc1337d1c93ade1d18f8123d9ab9ab1520c1739a9187fd36af09947058e6f87",
    ),
    "tagged-distribution-v0-1-4": (
        4_893,
        "ec729d9c865fdec122c9ae711b0a5459e8ab310fc8a11f76e2e3f0c8a8fb68c5",
    ),
    "tagged-distribution-v0-1-3": (
        4_884,
        "1e53801b1dd9adf262f61a159ba321cd00d9fef3d65158cf67fd55a496d6cc46",
    ),
    "tagged-distribution-v0-1-2": (
        4_826,
        "f79714a52faff962e93c8b646cbfea2d6e9e91023afa524c9d8dd6dcb909147a",
    ),
    "tagged-distribution-v0-1-1": (
        4_870,
        "84770a0ad3023e24b8c81af1216231033f3853d8aa8649fa9eb4f09bf95f3d28",
    ),
    "tagged-distribution-v0-1-0": (
        4_810,
        "d9eee85da210a2fd402c9a26758d381a02160bf77966585e1d8400a3223c4c88",
    ),
    "model-aware-context-completion-budgets": (
        4_852,
        "6e307e05e3e2eb1b806c8dd7e86fa2ecdfd27eda541e6a8bad304652b6df972a",
    ),
    "qwen-gdn-launch-evidence-producer": (
        4_793,
        "ae64354558c712c3e30a8cb6b2a0bd2b2bb554303a41ce10e6fda18654b14e4f",
    ),
    "qwen-gdn-scorecard-mode-identity": (
        4_743,
        "07154f46b0a403cb6e2336b3cb87c0ee7eb68e2108a09c48fae11b49cf2d4aa1",
    ),
    "qwen-gdn-four-projection-fusion": (
        4_709,
        "6408d9fd66c73ba44b8f059607c0bdd12b8ae31156eeddc17a6b9065311e709b",
    ),
    "dedicated-serving-qualification-planner": (
        4_770,
        "750df5a0bf2fe91613e22bafa16dce9978374be3c83a87f8da834e1d498eb6dc",
    ),
    "seeded-sampled-mtp-diagnostic-provider": (
        4_699,
        "c5126d6be5ac5934a04e4493f3f6a52e912e45df1755342002ea604badbd95f3",
    ),
    "current-inference-and-capacity-foundations": (
        4_802,
        "3cdb994c04f6caa781a26705ab73a936e3edbeb45adad247f516257a05b17799",
    ),
    "deterministic-sampled-generation-foundation": (
        4_782,
        "d475f2b4caa4843ead4bc12295051044523a2328d85dbe00fcf57f8efbc22d13",
    ),
    "self-reproducing-public-source": (
        4_667,
        "a3264db65e9e6dc78f597cb5e92f3b2860aa69ec61446eb289a90063b1216012",
    ),
    "reviewed-research-atom-feed": (
        4_711,
        "20d338fb30c2d72c964c845cf3d6464f0d75e9b430f5bd4207de45a47321ce01",
    ),
    "reviewed-release-detail-permalinks": (
        4_681,
        "aee1a7f0e90a446f118788a14a4b9f5160c2b45082de0959912fd6af4a88a0f7",
    ),
    "reviewed-benchmark-detail-permalinks": (
        4_783,
        "b1e96b23008349d98482ed2c83bf2cd63f2343198f3798e81ea3abb8fed1009c",
    ),
    "reviewed-home-current-cycle": (
        4_611,
        "8c90ddabb53b9c75b762724dc7266a7a56676a3347480ca714ca4b13fc71bfc2",
    ),
    "reviewed-social-metadata": (
        4_509,
        "366ee6115901d9ac8e6f0372891ebac339b95e52740a23fc1332defe46400d68",
    ),
    "reviewed-sitemap-discovery": (
        4_529,
        "15063f88c7b763c33bf093eba93bbf28a9937021f7d158e11c1f52188e04bc0b",
    ),
    "reviewed-release-atom-feed": (
        4_439,
        "2f007a8c0b7f7250f74dbf4f195b972d1663aa6b50a25260ceaab21e9f5a1ece",
    ),
    "reviewed-release-ledger": (
        4_424,
        "c076f33c846036cea883574b37c3e946814b8e201af889524f8312573af62ef2",
    ),
    "public-benchmark-explorer": (
        4_487,
        "02e741116de24e6e0a9891007a1199a9d59dc5c59e315ccf4c2a002d8afae530",
    ),
    "same-commit-pages-quality-gate": (
        4_460,
        "e4fc85daca6cfde589010a992b8b1018d568afca14d962994fc129d3c72be25c",
    ),
    "capabilities-and-evidence": (
        4_511,
        "0ab647211aca7e0a5eb0ed5b355ca3bcda09dd77630267b8febf0cf792793c4c",
    ),
    "compatible-hosted-swift-runner": (
        4_378,
        "3172866d91b6f8180bf79e3cb98fff1ae1f83410addbfaa2d7e1bad90ae3d2eb",
    ),
    "initial-public-release": (
        4_434,
        "88355644c60cbb1064d6f3cfaa6f65842292b058559c9185557b62288ab071e7",
    ),
}
RELEASE_DETAIL_DESCRIPTION = (
    "A reviewed fast-mlx public milestone with its exact commit, shipped surfaces, "
    "and unchanged claim boundary."
)
HTML_VOID_ELEMENTS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
REVIEWED_PAGE_METADATA: Dict[
    str, Tuple[str, str, str, Optional[str]]
] = {
    "": (
        "fast-mlx — measured quality for MLX on Apple Silicon",
        "Measures what speed and memory settings cost in output quality for MLX models on Apple Silicon, sizes models before loading, and publishes the evidence from a review-gated research loop.",
        "website",
        None,
    ),
    "quickstart/": (
        "Operator quickstart — fast-mlx",
        "Run fast-mlx's model-free HTTP/JSON and HTTP/SSE transport smoke, inspect capacity, and understand the loaded-serving boundary.",
        "website",
        None,
    ),
    "license/": (
        "Apache-2.0 license — fast-mlx",
        "Commercial-use, proprietary-extension, redistribution, notice, patent, trademark, and third-party boundaries for the fast-mlx public source.",
        "website",
        None,
    ),
    "status/": (
        "Current status — fast-mlx",
        "A manifest-derived view of fast-mlx capabilities, measured proof points, reviewed releases, research, and unchanged authority boundaries.",
        "website",
        None,
    ),
    "process/": (
        "The improvement loop — fast-mlx",
        "How fast-mlx turns research into reviewed, testable inference capabilities.",
        "website",
        None,
    ),
    "methodology/": (
        "Methodology — fast-mlx",
        "The correctness, comparability, and public-claim boundaries behind fast-mlx results.",
        "website",
        None,
    ),
    "capabilities/": (
        "Capabilities & evidence — fast-mlx",
        "A status-aware inventory of fast-mlx features and scoped measured results.",
        "website",
        None,
    ),
    **{
        f'capabilities/{capability["id"]}/': (
            f'{capability["name"]} — fast-mlx capability',
            f'Reviewed fast-mlx capability state and evidence for {capability["name"]}.',
            "website",
            None,
        )
        for capability in REVIEWED_CAPABILITIES
    },
    "benchmarks/": (
        "Benchmark explorer — fast-mlx",
        "Filter reviewed fast-mlx measurements without separating results from their scope, caveats, or evidence.",
        "website",
        None,
    ),
    "benchmarks/pld-echo-throughput/": (
        "Repetition-heavy solo PLD, 28.28 → 56.70 tok/s — fast-mlx benchmark evidence",
        "A reviewed fast-mlx benchmark result with its exact model, hardware, workload, decision, caveat, and evidence.",
        "website",
        None,
    ),
    "benchmarks/continuous-batch-c2-throughput/": (
        "Aggregate C=2 service rate, 29.29 → 42.70 tok/s — fast-mlx benchmark evidence",
        "A reviewed fast-mlx benchmark result with its exact model, hardware, workload, decision, caveat, and evidence.",
        "website",
        None,
    ),
    "benchmarks/http-sse-operational-soak/": (
        "HTTP/SSE service soak with 10,368 paired request/evidence rows — fast-mlx benchmark evidence",
        "A reviewed fast-mlx benchmark result with its exact model, hardware, workload, decision, caveat, and evidence.",
        "website",
        None,
    ),
    "releases/": (
        "Releases — fast-mlx",
        "A reviewed ledger of fast-mlx public milestones, exact commits, shipped surfaces, and unchanged boundaries.",
        "website",
        None,
    ),
    **{
        path: (
            f"{title} — fast-mlx release",
            RELEASE_DETAIL_DESCRIPTION,
            "website",
            None,
        )
        for path, title in (
            (f"releases/{identifier}/", title)
            for identifier, title in REVIEWED_RELEASE_IDENTITIES
        )
    },
    "research/": (
        "Research notes — fast-mlx",
        "Dated fast-mlx investigations and measured negative results.",
        "website",
        None,
    ),
    "research/streaming-the-experts-changed-the-answers/": (
        "Streaming the experts changed the answers — fast-mlx",
        "A mixture-of-experts model activates only a few of its experts for each token. If the machine cannot hold all of them, a serving engine can keep the rest of the model in memory and read each expert from SSD when the…",
        "article",
        "Serving big models on Apple Silicon",
    ),
    "research/the-lever-was-worth-ten-percent/": (
        "The lever was worth ten percent — fast-mlx",
        "The sparse-attention path in our long-context decode re-pools its key blocks on every single token. At a cache length of 32,768 that is thousands of blocks re-derived per layer per step, and twelve such layers per…",
        "article",
        "Serving big models on Apple Silicon",
    ),
    "research/the-default-nobody-chose/": (
        "The default nobody chose — fast-mlx",
        "A client that sends no temperature gets greedy argmax. That sounds like a reasonable default until you notice two things: the model's authors shipped a recommended sampler inside the checkpoint, and greedy is the…",
        "article",
        "Serving big models on Apple Silicon",
    ),
    "research/a-ratio-is-not-a-result/": (
        "A ratio is not a result — fast-mlx",
        "We had a measured speedup for speculative decoding: 1.31x, honestly obtained, with controls that passed. It was also, for our purposes, close to useless — because it was measured with sampling switched off in a way…",
        "article",
        "Serving big models on Apple Silicon",
    ),
    "research/you-had-to-load-the-model-to-learn-it-would-not-load/": (
        "You had to load the model to learn it wouldn't load — fast-mlx",
        "An earlier note — The fit-check refused a model that fit — was about the fit-check getting an answer wrong. This one is about not being able to ask it the question.",
        "article",
        "Serving big models on Apple Silicon",
    ),
    "research/every-tool-call-took-the-path-we-never-tested/": (
        "Every tool call took the path we never tested — fast-mlx",
        "We had a test for tool calling. We had a lot of tests for speculative decoding. We had no test for tool calling with speculative decoding — and because speculation is bound when the server loads the model rather than…",
        "article",
        "Serving big models on Apple Silicon",
    ),
    "research/the-fit-check-refused-a-model-that-fit/": (
        "The fit-check refused a model that fit — fast-mlx",
        "An earlier note — The bytes were derivable. We refused anyway. — set out the rule the fit-check lives by: fail toward RED. The number the engine computes before it loads anything answers one operator question — will…",
        "article",
        "Serving big models on Apple Silicon",
    ),
    "research/a-family-name-is-not-evidence-about-a-checkpoint/": (
        "A family name is not evidence about a checkpoint — fast-mlx",
        "fast-mlx keeps a served model's chain-of-thought out of the answer a caller sees. The OpenAI-shaped completion carries two fields: content, which is the answer, and reasoningcontent, which is everything the model…",
        "article",
        "Serving big models on Apple Silicon",
    ),
    "research/the-4k-limit-was-not-the-model-limit/": (
        "The 4K limit was not the model limit — fast-mlx",
        "A completion limit can look like a model fact when it is really only a server default.",
        "article",
        "OpenAI-compatible serving; Long-horizon inference; Fail-closed capacity",
    ),
    "research/how-the-autonomous-loop-builds-fast-mlx/": (
        "How the autonomous loop builds fast-mlx — fast-mlx",
        "fast-mlx is an LLM inference engine for Apple Silicon, written in Swift on MLX. It is also an experiment in how an engine gets built: most of the day-to-day engineering — research intake, implementation spikes,…",
        "article",
        "Rapid research integration — the flywheel; Disciplined proof over",
    ),
    "research/the-state-was-right-the-ledger-was-frozen/": (
        "The state was right. The ledger was frozen. — fast-mlx",
        "Continuous batching is how one model serves many users at once: several requests share a single forward pass, their per-request caches stacked side by side, and the engine merges, splits, and re-merges those rows as…",
        "article",
        "Serving big models on Apple Silicon",
    ),
    "research/the-bytes-were-derivable-we-refused-anyway/": (
        "The bytes were derivable. We refused anyway. — fast-mlx",
        "An earlier note — One formula, wrong for a third of the catalog — described the fit-check: the number the engine computes before it loads a model, answering the one operator question that matters on a fixed-memory…",
        "article",
        "Serving big models on Apple Silicon",
    ),
    "research/measuring-the-premise-before-building-the-engine/": (
        "Measuring the premise before building the engine — does MoE routing actually cluster? — fast-mlx",
        "\"Run models larger than your RAM\" is a seductive line to put on a box. For a Mixture-of-Experts model it's even plausible: only a handful of the experts fire on any given token, so — the pitch goes — keep the hot…",
        "article",
        "Running models larger than memory — the spike that gates the bet",
    ),
    "research/the-tool-call-we-could-prove-without-the-model/": (
        "The tool call we could prove without the model — fast-mlx",
        "Tool calling looks like a feature you bolt onto a chat server: accept a tools array, let the model emit a function call, hand it back in OpenAI's shape. Most of that machinery already existed in fast-mlx before this…",
        "article",
        "Building a high-performance MLX inference engine in Swift; Rapid research integration — the flywheel; Disciplined proof over convenient claims",
    ),
    "research/the-checkout-that-couldnt-compile-its-gpu/": (
        "The checkout that couldn't compile its own GPU — fast-mlx",
        "A benchmark nobody can run is not evidence, and an engine nobody can start is not a product. For a stretch, fast-mlx was the second thing: a fresh git clone on a current Mac would build, launch, accept a request, and…",
        "article",
        "Building a high-performance MLX inference engine in Swift; Disciplined proof over convenient claims; Meeting operators where they are",
    ),
    "research/sampling-before-serving/": (
        "Sampling before serving: why the random draw needed its own contract — fast-mlx",
        "Sampled generation sounds like a small change: take logits, apply temperature and top-p, draw a token, and continue. In a serving engine, that “draw a token” step quietly touches request identity, retries,…",
        "article",
        "Building a high-performance MLX inference engine in Swift; Rapid research integration — the flywheel",
    ),
    "research/the-repository-that-could-reproduce-itself/": (
        "The repository that could reproduce itself — but could not publish itself — fast-mlx",
        "A public source release is easy to mistake for a directory copy.",
        "article",
        "Rapid research integration — the flywheel; Building a high-performance MLX inference engine in Swift",
    ),
    "research/the-proof-did-not-end-when-the-timer-did/": (
        "The proof did not end when the timer did — fast-mlx",
        "A short benchmark can show that continuous batching works. It cannot show that an HTTP service keeps cleaning up after disappearing clients for a full day.",
        "article",
        "Building a high-performance MLX inference engine in Swift; Serving big models on Apple Silicon; Rapid research integration — the flywheel",
    ),
    "research/exact-speculation-was-not-fast-enough/": (
        "Exact speculation was not fast enough — fast-mlx",
        "Speculative decoding has an attractive promise: predict several future tokens cheaply, verify them with the target model, and skip some serial decode steps without changing the answer.",
        "article",
        "Building a high-performance MLX inference engine in Swift; Rapid research integration — the flywheel",
    ),
    "research/exact-prefix-cache-needs-exact-provenance/": (
        "An exact prefix cache needs exact provenance — fast-mlx",
        "An inference cache can return the right bytes and still report the wrong reason.",
        "article",
        "Inference research",
    ),
    "research/the-third-geometry-said-no/": (
        "The third geometry said no — fast-mlx",
        "The third-family gate was not supposed to reward novelty. Qwen and Llama had already exercised large uniform-GQA shapes, but both shared the same Q64/KV8/D128 attention geometry. A broader product claim needed a…",
        "article",
        "Serving big models on Apple Silicon; The optimization dial — quantified precision-loss tuning; Rapid research integration — the flywheel",
    ),
    "research/llama-ran-without-a-speed-tier/": (
        "Llama ran, but it did not earn a speed tier — fast-mlx",
        "The second-family gate for fast-mlx was deliberately simple: take a materially different popular model family, bind the exact engine source, checkpoint, and tokenizer, then let the same evidence rules decide whether…",
        "article",
        "Serving big models on Apple Silicon; The optimization dial — quantified precision-loss tuning; Rapid research integration — the flywheel",
    ),
    "research/fifteen-times-faster-still-not-fast/": (
        "Fifteen times faster still was not fast — fast-mlx",
        "Compressed KV attention had one clear job: keep the cache compressed while attention reads it. The previous fast-mlx KVarN path saved memory but reconstructed full K/V tensors on the hot path. That made it a useful…",
        "article",
        "Building a high-performance MLX inference engine in Swift; The optimization dial — quantified precision-loss tuning; Rapid research integration — the flywheel",
    ),
    "research/when-smaller-kv-is-not-faster/": (
        "When smaller KV is not faster — fast-mlx",
        "The KVarN/asymmetric KV-cache gate started with a tempting promise: spend fewer bytes on the attention cache, fit far more context, and maybe get a runtime speedup too. That is what the fast-mlx dial exists to…",
        "article",
        "The optimization dial — quantified precision-loss tuning; Serving big models on Apple Silicon; Rapid research integration — the flywheel",
    ),
    "research/the-fastest-request-wasnt-the-fastest-service/": (
        "The fastest request wasn't the fastest service — fast-mlx",
        "On an Apple M5 Max at one request, our fastest exact path was prompt-lookup decoding. Qwen3-32B-4bit generated 28.30 tokens per second with PLD, versus 26.72 through the new continuous-batching runtime. If we had…",
        "article",
        "Building a high-performance MLX inference engine in Swift; Rapid research integration — the flywheel",
    ),
    "research/lossless-wasnt-byte-identical/": (
        "“Lossless” wasn't byte-identical: the speculative decoder that failed at generated index seven — fast-mlx",
        "EAGLE-3 looked like the trained speculative decoder we had been waiting for. A public Qwen3-32B checkpoint matched our production-size target. The draft head was only one decoder layer. Its published algorithm was…",
        "article",
        "Rapid research integration — the flywheel; Building a high-performance MLX engine in Swift",
    ),
    "research/when-zero-speculation-costs-two-percent/": (
        "When zero speculation costs 2%: making a 2× decoder safe to leave on — fast-mlx",
        "Prompt-lookup decoding had already given us the result every inference team wants: nearly twice the decode throughput, with byte-identical output. On a repetition-heavy agent prompt, Qwen3-32B-4bit rose from about 28…",
        "article",
        "Building a high-performance MLX inference engine in Swift; Rapid research integration — the flywheel",
    ),
    "research/two-x-for-free-when-the-model-repeats-itself/": (
        "2× for free, when the model repeats itself: prompt-lookup decoding with a byte-identical proof — fast-mlx",
        "Every LLM speedup we had shipped so far traded something. Quantize the weights, pay in perplexity. Quantize the KV cache, pay in long-context tail divergence — we measured that one to death and shelved it. So the…",
        "article",
        "Building a high-performance MLX engine in Swift; Rapid research integration — the flywheel",
    ),
    "research/turboquant-exact-math-still-lost/": (
        "We implemented Google's TurboQuant exactly, matched the paper's error tables — and it still lost to plain 4-bit quantization — fast-mlx",
        "The KV cache is the memory bill for long context. On Qwen3-32B, every token you keep costs 256 KiB of fp16 keys and values — 64 layers × 8 KV heads × 128 dims × 2 tensors. At a 24K-token context that's 6 GB per…",
        "article",
        "The optimization dial — quantified precision-loss tuning; Building a high-performance MLX engine in Swift",
    ),
    "research/trusting-the-instrument/": (
        "Who measures the measurer? Auditing a precision-loss harness that was quietly lying — fast-mlx",
        "fast-mlx's product isn't raw speed — it's a dial: turn up the compression, and see exactly how much accuracy you trade. That promise lives or dies on one thing — the instrument that produces the \"how much accuracy\"…",
        "article",
        "The optimization dial — quantified precision-loss tuning",
    ),
    "research/the-wall-that-wasnt/": (
        "The 7K wall that wasn't: jetsam forensics, a quadratic allocator, and the statistic hiding in the tail — fast-mlx",
        "Our precision-loss harness had just been hardened — teacher-forced KL, perplexity, a versioned corpus, provenance records. Then it hit a wall: any measurement past roughly 7,000 tokens of context died with a SIGKILL…",
        "article",
        "The optimization dial — quantified precision-loss tuning",
    ),
    "research/one-formula-wrong-for-a-third-of-the-catalog/": (
        "One formula, wrong for a third of the catalog: a KV-memory model that refuses to lie — fast-mlx",
        "We wanted the engine to answer a simple operator question: before you raise a model's context window from 32K toward its maximum, will this Mac actually hold it? Answering it needs one number — how many bytes of KV…",
        "article",
        "Serving big models on Apple Silicon",
    ),
}


def reviewed_research_articles() -> Tuple[Dict[str, str], ...]:
    """Return the independently pinned public metadata for reviewed research."""

    records: List[Dict[str, str]] = []
    for public_path in REVIEWED_ARTICLE_PATHS:
        title, summary, page_type, theme = REVIEWED_PAGE_METADATA[public_path]
        if page_type != "article" or theme is None or not title.endswith(" — fast-mlx"):
            raise ValueError(f"incomplete reviewed research metadata for {public_path}")
        date, reviewed_at = REVIEWED_ARTICLE_DATES[public_path]
        records.append(
            {
                "title": title[: -len(" — fast-mlx")],
                "date": date,
                "theme": theme,
                "summary": summary,
                "path": public_path,
                "reviewedAt": reviewed_at,
            }
        )
    return tuple(records)
SITEMAP_ARTICLE_PATH = re.compile(r"research/[a-z0-9]+(?:-[a-z0-9]+)*/")
HTML_LIKE_SUFFIXES = {".html", ".htm"}
PUBLIC_PATH = re.compile(
    r"(?:[a-z0-9][a-z0-9.-]*/)*(?:[a-z0-9][a-z0-9.-]*/|[a-z0-9][a-z0-9.-]*\.(?:atom|html|json))"
)
RELEASE_INDEX_KEYS = {
    "schemaVersion",
    "project",
    "policy",
    "claimBoundary",
    "updatedAt",
    "currentBoundary",
    "releases",
}
RELEASE_BOUNDARY_KEYS = {"id", "label", "state", "summary", "evidence"}
RELEASE_LINK_KEYS = {"label", "path"}
RELEASE_ENTRY_KEYS = {
    "id",
    "title",
    "publishedAt",
    "category",
    "state",
    "summary",
    "scope",
    "publicCommit",
    "publicLinks",
    "sourceUrl",
}

# --- Served-engine benchmark ledger (V1-V3; mirrors build_public_site.py's
# `load_served_benchmarks`/`validate_served_benchmark_entry`, a deliberate
# separate copy -- see the "projection split" note on QUALITY_CARD_HARDWARE_CLASS
# above for why this project never imports either sibling's schema). ---------
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
# `combineCommit` mirrors `build_public_site.SERVED_BENCHMARK_HARNESS_KEYS`
# byte-for-byte -- see that constant's own comment for why it is a
# SEPARATE commit from `publicCommit` (the harness that MEASURED the row
# is not necessarily the harness whose `--combine` PRODUCED its ratio).
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
# Mirrors `build_public_site.QUALITY_GUIDE_HOST_SHORTHAND` byte-for-byte --
# a deliberate separate copy, same reason as every other QUALITY_CARD_*
# duplication in this module.
SERVED_BENCHMARK_HOST_SHORTHAND = re.compile(r"(?<![0-9])\.25[0-3](?![0-9])")
# Mirrors `build_public_site.SERVED_BENCHMARK_PACK_LABEL` byte-for-byte --
# see that constant's own comment for the allowlist shape and why it is
# checked AFTER the marker scan below, not instead of it.
SERVED_BENCHMARK_PACK_LABEL = re.compile(r"[a-z0-9][a-z0-9 ./-]{0,55}pack")
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
SERVED_BENCHMARK_RERUN_PLACEHOLDER = re.compile(r"<[a-z][a-z-]*>(?:\.json)?")
SERVED_BENCHMARK_RERUN_NUMBER = re.compile(r"\d+(?:\.\d+)?")
SERVED_BENCHMARK_RERUN_FILENAME = re.compile(r"[a-z0-9-]+\.json")
# Sealed byte count + sha256 of the built `benchmarks/served-benchmarks.json`
# (V1) -- derived with `python3 scripts/build_public_site.py --output` and
# `shasum -a 256`, never transcribed from any other source.
REVIEWED_SERVED_BENCHMARKS_BYTES = 4380
REVIEWED_SERVED_BENCHMARKS_SHA256 = (
    "e750cb065ba08b92d6ca1965afbc500d626d11bdc1a6e56bb6ee16aae1dfa4b8"
)
# (id, cardId) pairs in the exact newest-first order the reviewed ledger
# renders them (V3) -- mirrors REVIEWED_RELEASE_IDENTITIES above.
REVIEWED_SERVED_BENCHMARK_IDENTITIES: Tuple[Tuple[str, str], ...] = (
    (
        "qwen38-flash-next-iq-3p3bpw-m3ultra-2026-09-29",
        "qwen38-flash-next-iq-3p3bpw@m3ultra-v2696",
    ),
    (
        "qwen38-flash-next-mixed-4-8bit-m3ultra-2026-09-29",
        "qwen38-flash-next-mixed-4-8bit@m3ultra-v2696",
    ),
)
# Sealed byte count + sha256 per built `benchmarks/served-benchmark-rows/
# <id>.json` row file (slice 2) -- same derivation as
# REVIEWED_SERVED_BENCHMARKS_BYTES/SHA256 above, one pair per
# REVIEWED_SERVED_BENCHMARK_IDENTITIES entry.
# Mirrors `build_public_site.SERVED_BENCHMARK_ROWS_DIRNAME` byte-for-byte --
# see that constant's own comment for why this is a separate copy.
SERVED_BENCHMARK_ROWS_DIRNAME = "served-benchmark-rows"
REVIEWED_SERVED_BENCHMARK_ROW_FILES: Dict[str, Tuple[int, str]] = {
    "qwen38-flash-next-iq-3p3bpw-m3ultra-2026-09-29": (
        10414,
        "9b5994c9d05262039c0e2a500bfc2e72ed7ca60460f89fa62695e06f4584d882",
    ),
    "qwen38-flash-next-mixed-4-8bit-m3ultra-2026-09-29": (
        10414,
        "a796f8552a1f5860fbb2f68e66c5d76710baac36f4c81af347b23f09d37c35a2",
    ),
}


class LinkCollector(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag not in {"a", "link", "script", "img"}:
            return
        attribute = "href" if tag in {"a", "link"} else "src"
        for key, value in attrs:
            if key == attribute and value:
                self.links.append(value)


class HeadMetadataCollector(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.canonicals: List[str] = []
        self.properties: Dict[str, List[str]] = {}
        self.atom_links: List[Dict[str, str]] = []
        self.metadata_outside_head = False
        self.invalid_head_structure = False
        self._in_head = False
        self._head_seen = False
        self._body_started = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag == "head":
            if self._head_seen or self._body_started:
                self.invalid_head_structure = True
                self._in_head = False
            else:
                self._head_seen = True
                self._in_head = True
            return
        if tag == "body":
            self._body_started = True
            self._in_head = False
        attribute_names = [name.casefold() for name, _value in attrs]
        if tag in {"link", "meta"} and len(attribute_names) != len(
            set(attribute_names)
        ):
            self.invalid_head_structure = True
        attributes = dict(attrs)
        rel_tokens = {
            token.casefold()
            for token in (attributes.get("rel") or "").split()
        }
        is_canonical = tag == "link" and "canonical" in rel_tokens
        is_atom = (
            tag == "link"
            and "alternate" in rel_tokens
            and attributes.get("type") == "application/atom+xml"
        )
        property_name = attributes.get("property") if tag == "meta" else None
        is_social = isinstance(property_name, str) and (
            property_name.casefold().startswith("og:")
            or property_name.casefold().startswith("article:")
        )
        if not is_canonical and not is_social and not is_atom:
            return
        if not self._in_head:
            self.metadata_outside_head = True
        if is_canonical:
            href = attributes.get("href")
            self.canonicals.append(href if isinstance(href, str) else "")
        if is_social and isinstance(property_name, str):
            content = attributes.get("content")
            self.properties.setdefault(property_name, []).append(
                content if isinstance(content, str) else ""
            )
        if is_atom:
            self.atom_links.append(
                {
                    key: value if isinstance(value, str) else ""
                    for key, value in (
                        ("rel", attributes.get("rel")),
                        ("type", attributes.get("type")),
                        ("title", attributes.get("title")),
                        ("href", attributes.get("href")),
                    )
                }
            )

    def handle_endtag(self, tag: str) -> None:
        if tag == "head":
            self._in_head = False


class QuickstartCollector(html.parser.HTMLParser):
    """Collect the visible, static contract inside the reviewed quickstart root."""

    def __init__(self) -> None:
        super().__init__()
        self.roots: List[Dict[str, object]] = []
        self.root_count = 0
        self.page_h1_count = 0
        self.scripts: List[Optional[str]] = []
        self.inline_style_count = 0
        self.stylesheet_links: List[Optional[str]] = []
        self._current: Optional[Dict[str, object]] = None
        self._root_depth = 0
        self._active_command: Optional[Dict[str, object]] = None
        self._element_stack: List[
            Tuple[str, Dict[str, Optional[str]], bool]
        ] = []

    @staticmethod
    def _suppresses_visibility(attributes: Dict[str, Optional[str]]) -> bool:
        classes = set((attributes.get("class") or "").split())
        return (
            "hidden" in attributes
            or "inert" in attributes
            or (attributes.get("aria-hidden") or "").casefold() == "true"
            or "style" in attributes
            or bool(classes & {"benchmark-controls", "research-controls"})
            or "data-benchmark-controls" in attributes
            or "data-research-controls" in attributes
        )

    def handle_starttag(
        self, tag: str, attrs: List[Tuple[str, Optional[str]]]
    ) -> None:
        names = [name for name, _value in attrs]
        has_duplicate_attributes = len(names) != len(set(names))
        attributes = dict(attrs)
        if tag == "h1":
            self.page_h1_count += 1
        elif tag == "script":
            self.scripts.append(attributes.get("src"))
        elif tag == "style":
            self.inline_style_count += 1
        elif tag == "link" and "stylesheet" in {
            token.casefold() for token in (attributes.get("rel") or "").split()
        }:
            self.stylesheet_links.append(attributes.get("href"))

        started_root = False
        if tag == "div" and "data-quickstart" in attributes:
            self.root_count += 1
            if self._current is None:
                self._current = {
                    "rootAttributes": attributes,
                    "ancestry": list(self._element_stack),
                    "hasDuplicateAttributes": has_duplicate_attributes,
                    "hasVisibilitySuppressor": self._suppresses_visibility(attributes),
                    "forbiddenTags": [],
                    "commands": [],
                    "links": [],
                    "h1Count": 0,
                    "text_parts": [],
                }
                self._root_depth = 1
                started_root = True

        if self._current is not None:
            if has_duplicate_attributes:
                self._current["hasDuplicateAttributes"] = True
            if self._suppresses_visibility(attributes):
                self._current["hasVisibilitySuppressor"] = True
            if (
                tag not in QUICKSTART_ALLOWED_TAGS
                or any(name.casefold().startswith("on") for name in names)
            ):
                forbidden = self._current["forbiddenTags"]
                if isinstance(forbidden, list):
                    forbidden.append(tag)
            if tag == "h1":
                count = self._current["h1Count"]
                self._current["h1Count"] = count + 1 if isinstance(count, int) else 1
            elif tag == "a" and attributes.get("href"):
                links = self._current["links"]
                if isinstance(links, list):
                    links.append(attributes["href"])
            if self._active_command is not None:
                self._active_command["hasNestedTag"] = True
            if tag == "code" and "data-command" in attributes:
                if self._active_command is not None:
                    self._active_command["hasNestedCommand"] = True
                else:
                    self._active_command = {
                        "id": attributes.get("data-command"),
                        "attributes": attributes,
                        "hasNestedTag": False,
                        "hasNestedCommand": False,
                        "text_parts": [],
                    }
            if tag not in HTML_VOID_ELEMENTS and not started_root:
                self._root_depth += 1

        if tag not in HTML_VOID_ELEMENTS:
            self._element_stack.append((tag, attributes, has_duplicate_attributes))

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            text_parts = self._current["text_parts"]
            if isinstance(text_parts, list):
                text_parts.append(data)
        if self._active_command is not None:
            text_parts = self._active_command["text_parts"]
            if isinstance(text_parts, list):
                text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "code" and self._active_command is not None:
            text_parts = self._active_command.pop("text_parts")
            self._active_command["text"] = (
                "".join(text_parts) if isinstance(text_parts, list) else ""
            )
            if self._current is not None:
                commands = self._current["commands"]
                if isinstance(commands, list):
                    commands.append(self._active_command)
            self._active_command = None

        if self._current is not None and tag not in HTML_VOID_ELEMENTS:
            self._root_depth -= 1
            if self._root_depth == 0:
                text_parts = self._current.pop("text_parts")
                self._current["text"] = (
                    " ".join("".join(text_parts).split())
                    if isinstance(text_parts, list)
                    else ""
                )
                self.roots.append(self._current)
                self._current = None

        for index in range(len(self._element_stack) - 1, -1, -1):
            if self._element_stack[index][0] == tag:
                del self._element_stack[index:]
                break


class StatusPageCollector(html.parser.HTMLParser):
    """Collect the immutable, static contract inside the reviewed status root."""

    def __init__(self) -> None:
        super().__init__()
        self.roots: List[Dict[str, object]] = []
        self.root_count = 0
        self.page_h1_count = 0
        self.scripts: List[Optional[str]] = []
        self.inline_style_count = 0
        self.stylesheet_links: List[Optional[str]] = []
        self._current: Optional[Dict[str, object]] = None
        self._root_depth = 0
        self._element_stack: List[
            Tuple[str, Dict[str, Optional[str]], bool]
        ] = []

    @staticmethod
    def _suppresses_visibility(attributes: Dict[str, Optional[str]]) -> bool:
        classes = set((attributes.get("class") or "").split())
        return (
            "hidden" in attributes
            or "inert" in attributes
            or (attributes.get("aria-hidden") or "").casefold() == "true"
            or "style" in attributes
            or bool(classes & {"benchmark-controls", "research-controls"})
            or "data-benchmark-controls" in attributes
            or "data-research-controls" in attributes
        )

    def handle_starttag(
        self, tag: str, attrs: List[Tuple[str, Optional[str]]]
    ) -> None:
        names = [name for name, _value in attrs]
        has_duplicate_attributes = len(names) != len(set(names))
        attributes = dict(attrs)
        if tag == "h1":
            self.page_h1_count += 1
        elif tag == "script":
            self.scripts.append(attributes.get("src"))
        elif tag == "style":
            self.inline_style_count += 1
        elif tag == "link" and "stylesheet" in {
            token.casefold() for token in (attributes.get("rel") or "").split()
        }:
            self.stylesheet_links.append(attributes.get("href"))

        started_root = False
        if tag == "div" and "data-status-page" in attributes:
            self.root_count += 1
            if self._current is None:
                self._current = {
                    "rootAttributes": attributes,
                    "ancestry": list(self._element_stack),
                    "hasDuplicateAttributes": has_duplicate_attributes,
                    "hasVisibilitySuppressor": self._suppresses_visibility(attributes),
                    "forbiddenTags": [],
                    "statusCounts": [],
                    "capabilities": [],
                    "highlights": [],
                    "links": [],
                    "h1Count": 0,
                    "text_parts": [],
                }
                self._root_depth = 1
                started_root = True

        if self._current is not None:
            if has_duplicate_attributes:
                self._current["hasDuplicateAttributes"] = True
            if self._suppresses_visibility(attributes):
                self._current["hasVisibilitySuppressor"] = True
            if (
                tag not in STATUS_ALLOWED_TAGS
                or any(name.casefold().startswith("on") for name in names)
            ):
                forbidden = self._current["forbiddenTags"]
                if isinstance(forbidden, list):
                    forbidden.append(tag)
            if tag == "h1":
                count = self._current["h1Count"]
                self._current["h1Count"] = count + 1 if isinstance(count, int) else 1
            elif tag == "a" and attributes.get("href"):
                links = self._current["links"]
                if isinstance(links, list):
                    links.append(attributes["href"])
            if "data-status-count" in attributes:
                counts = self._current["statusCounts"]
                if isinstance(counts, list):
                    counts.append(
                        (
                            attributes.get("data-status-count"),
                            attributes.get("data-count"),
                            attributes,
                        )
                    )
            if "data-status-capability" in attributes:
                capabilities = self._current["capabilities"]
                if isinstance(capabilities, list):
                    capabilities.append(
                        (
                            attributes.get("data-status-capability"),
                            attributes.get("data-capability-state"),
                            attributes,
                        )
                    )
            if "data-status-highlight" in attributes:
                highlights = self._current["highlights"]
                if isinstance(highlights, list):
                    highlights.append(
                        (
                            attributes.get("data-status-highlight"),
                            attributes.get("data-highlight-decision"),
                            attributes,
                        )
                    )
            if tag not in HTML_VOID_ELEMENTS and not started_root:
                self._root_depth += 1

        if tag not in HTML_VOID_ELEMENTS:
            self._element_stack.append((tag, attributes, has_duplicate_attributes))

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            text_parts = self._current["text_parts"]
            if isinstance(text_parts, list):
                text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._current is not None and tag not in HTML_VOID_ELEMENTS:
            self._root_depth -= 1
            if self._root_depth == 0:
                text_parts = self._current.pop("text_parts")
                self._current["text"] = (
                    " ".join("".join(text_parts).split())
                    if isinstance(text_parts, list)
                    else ""
                )
                self.roots.append(self._current)
                self._current = None

        for index in range(len(self._element_stack) - 1, -1, -1):
            if self._element_stack[index][0] == tag:
                del self._element_stack[index:]
                break


class PrimaryNavigationCollector(html.parser.HTMLParser):
    """Collect links structurally contained in the primary navigation landmark."""

    def __init__(self) -> None:
        super().__init__()
        self.nav_count = 0
        self.navs: List[Dict[str, object]] = []
        self._current: Optional[Dict[str, object]] = None
        self._depth = 0
        self._active_link: Optional[Dict[str, object]] = None
        self._element_stack: List[
            Tuple[str, Dict[str, Optional[str]], bool]
        ] = []

    def handle_starttag(
        self, tag: str, attrs: List[Tuple[str, Optional[str]]]
    ) -> None:
        names = [name for name, _value in attrs]
        has_duplicate_attributes = len(names) != len(set(names))
        attributes = dict(attrs)
        started_nav = False
        if tag == "nav" and attributes.get("aria-label") == "Primary navigation":
            self.nav_count += 1
            if self._current is None:
                self._current = {
                    "attributes": attributes,
                    "ancestry": list(self._element_stack),
                    "hasDuplicateAttributes": has_duplicate_attributes,
                    "links": [],
                }
                self._depth = 1
                started_nav = True

        if self._current is not None:
            if has_duplicate_attributes:
                self._current["hasDuplicateAttributes"] = True
            if self._active_link is not None:
                self._active_link["hasNestedTag"] = True
            if tag == "a":
                self._active_link = {
                    "attributes": attributes,
                    "ancestry": list(self._element_stack),
                    "hasNestedTag": False,
                    "text_parts": [],
                }
            if tag not in HTML_VOID_ELEMENTS and not started_nav:
                self._depth += 1

        if tag not in HTML_VOID_ELEMENTS:
            self._element_stack.append((tag, attributes, has_duplicate_attributes))

    def handle_data(self, data: str) -> None:
        if self._active_link is not None:
            text_parts = self._active_link["text_parts"]
            if isinstance(text_parts, list):
                text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._active_link is not None:
            text_parts = self._active_link.pop("text_parts")
            self._active_link["text"] = (
                " ".join("".join(text_parts).split())
                if isinstance(text_parts, list)
                else ""
            )
            if self._current is not None:
                links = self._current["links"]
                if isinstance(links, list):
                    links.append(self._active_link)
            self._active_link = None

        if self._current is not None and tag not in HTML_VOID_ELEMENTS:
            self._depth -= 1
            if self._depth == 0:
                self.navs.append(self._current)
                self._current = None

        for index in range(len(self._element_stack) - 1, -1, -1):
            if self._element_stack[index][0] == tag:
                del self._element_stack[index:]
                break


class BenchmarkCollector(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.cards: List[Dict[str, object]] = []
        self.options: Dict[str, List[Dict[str, Optional[str]]]] = {
            name: [] for name in BENCHMARK_FILTER_NAMES
        }
        self.has_controls = False
        self.has_count = False
        self.has_empty_state = False
        self.has_script = False
        self._current_card: Optional[Dict[str, object]] = None
        self._current_select: Optional[str] = None
        self._current_option: Optional[Dict[str, object]] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "article" and "benchmark-result" in classes:
            self._current_card = {
                "id": attributes.get("data-highlight-id"),
                "model": attributes.get("data-model"),
                "hardware": attributes.get("data-hardware"),
                "decision": attributes.get("data-decision"),
                "hidden": "hidden" in attributes,
                "datetime": None,
                "links": [],
                "text_parts": [],
            }
        elif self._current_card is not None:
            if tag == "time":
                self._current_card["datetime"] = attributes.get("datetime")
            elif tag == "a" and attributes.get("href"):
                links = self._current_card["links"]
                if isinstance(links, list):
                    links.append(attributes["href"])

        if tag == "select" and attributes.get("name") in self.options:
            self._current_select = attributes["name"]
        elif tag == "option" and self._current_select is not None:
            self._current_option = {
                "value": attributes.get("value"),
                "text_parts": [],
            }
        if tag == "form" and "data-benchmark-controls" in attributes:
            self.has_controls = True
        if (
            "data-benchmark-count" in attributes
            and attributes.get("aria-live") == "polite"
        ):
            self.has_count = True
        if (
            "data-benchmark-empty" in attributes
            and "hidden" in attributes
            and attributes.get("role") == "status"
        ):
            self.has_empty_state = True
        if (
            tag == "script"
            and attributes.get("src") == "../assets/benchmark-explorer.js"
        ):
            self.has_script = True

    def handle_data(self, data: str) -> None:
        if self._current_card is not None:
            text_parts = self._current_card["text_parts"]
            if isinstance(text_parts, list):
                text_parts.append(data)
        if self._current_option is not None:
            text_parts = self._current_option["text_parts"]
            if isinstance(text_parts, list):
                text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "article" and self._current_card is not None:
            text_parts = self._current_card.pop("text_parts")
            self._current_card["text"] = " ".join(
                "".join(text_parts).split()
            ) if isinstance(text_parts, list) else ""
            self.cards.append(self._current_card)
            self._current_card = None
        if tag == "option" and self._current_option is not None:
            text_parts = self._current_option.pop("text_parts")
            self._current_option["text"] = " ".join(
                "".join(text_parts).split()
            ) if isinstance(text_parts, list) else ""
            if self._current_select is not None:
                self.options[self._current_select].append(self._current_option)
            self._current_option = None
        elif tag == "select":
            self._current_select = None


class BenchmarkDetailCollector(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sections: List[Dict[str, object]] = []
        self.page_h1_count = 0
        self.page_links: List[str] = []
        self.scripts: List[Optional[str]] = []
        self.text_parts: List[str] = []
        self._current: Optional[Dict[str, object]] = None
        self._section_depth = 0
        self._field_tag: Optional[str] = None
        self._field_parts: List[str] = []
        self._element_stack: List[Tuple[str, bool]] = []

    @staticmethod
    def _suppresses_visibility(
        tag: str, attributes: Dict[str, Optional[str]]
    ) -> bool:
        return (
            tag in {"details", "dialog"}
            or "hidden" in attributes
            or (attributes.get("aria-hidden") or "").casefold() == "true"
            or "style" in attributes
        )

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        names = [name for name, _value in attrs]
        attributes = dict(attrs)
        duplicate_attributes = len(names) != len(set(names))
        suppresses_visibility = self._suppresses_visibility(tag, attributes)

        if tag == "h1":
            self.page_h1_count += 1
        if tag == "a" and attributes.get("href"):
            self.page_links.append(str(attributes["href"]))
        if tag == "script":
            self.scripts.append(attributes.get("src"))

        if tag == "section" and "data-benchmark-detail" in attributes:
            if self._current is None:
                self._current = {
                    "id": attributes.get("data-highlight-id"),
                    "datetime": None,
                    "links": [],
                    "terms": [],
                    "values": [],
                    "text_parts": [],
                    "h1Count": 0,
                    "dlCount": 0,
                    "hasDuplicateAttributes": duplicate_attributes,
                    "hasVisibilitySuppressor": suppresses_visibility
                    or any(item[1] for item in self._element_stack),
                    "hasNestedDetail": False,
                }
                self._section_depth = 1
            else:
                self._current["hasNestedDetail"] = True
                self._section_depth += 1
        elif self._current is not None and tag == "section":
            self._section_depth += 1

        if self._current is not None:
            if duplicate_attributes:
                self._current["hasDuplicateAttributes"] = True
            if suppresses_visibility:
                self._current["hasVisibilitySuppressor"] = True
            if tag == "h1":
                self._current["h1Count"] = int(self._current["h1Count"]) + 1
            elif tag == "time":
                self._current["datetime"] = attributes.get("datetime")
            elif tag == "a" and attributes.get("href"):
                links = self._current["links"]
                if isinstance(links, list):
                    links.append(attributes["href"])
            elif tag == "dl":
                self._current["dlCount"] = int(self._current["dlCount"]) + 1
            elif tag in {"dt", "dd"}:
                self._field_tag = tag
                self._field_parts = []

        if tag not in HTML_VOID_ELEMENTS:
            self._element_stack.append((tag, suppresses_visibility))

    def handle_data(self, data: str) -> None:
        self.text_parts.append(data)
        if self._current is not None:
            parts = self._current["text_parts"]
            if isinstance(parts, list):
                parts.append(data)
        if self._field_tag is not None:
            self._field_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        self.text_parts.append(" ")
        if self._current is not None:
            parts = self._current["text_parts"]
            if isinstance(parts, list):
                parts.append(" ")
        if self._current is not None and tag == self._field_tag:
            key = "terms" if tag == "dt" else "values"
            values = self._current[key]
            if isinstance(values, list):
                values.append(" ".join("".join(self._field_parts).split()))
            self._field_tag = None
            self._field_parts = []

        if self._current is not None and tag == "section":
            self._section_depth -= 1
            if self._section_depth == 0:
                parts = self._current.pop("text_parts")
                self._current["text"] = (
                    " ".join("".join(parts).split())
                    if isinstance(parts, list)
                    else ""
                )
                self.sections.append(self._current)
                self._current = None

        if self._element_stack:
            if self._element_stack[-1][0] == tag:
                self._element_stack.pop()
            else:
                for position in range(len(self._element_stack) - 1, -1, -1):
                    if self._element_stack[position][0] == tag:
                        del self._element_stack[position:]
                        break


class CapabilityCardCollector(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.cards: List[Dict[str, object]] = []
        self._current: Optional[Dict[str, object]] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attributes = dict(attrs)
        if tag == "article" and "data-capability-card" in attributes:
            self._current = {
                "id": attributes.get("data-capability-card"),
                "state": attributes.get("data-capability-state"),
                "hidden": "hidden" in attributes,
                "links": [],
                "text_parts": [],
            }
        elif self._current is not None and tag == "a" and attributes.get("href"):
            links = self._current["links"]
            if isinstance(links, list):
                links.append(attributes["href"])

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            parts = self._current["text_parts"]
            if isinstance(parts, list):
                parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "article" and self._current is not None:
            parts = self._current.pop("text_parts")
            self._current["text"] = (
                " ".join("".join(parts).split()) if isinstance(parts, list) else ""
            )
            self.cards.append(self._current)
            self._current = None


class CapabilityDetailCollector(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sections: List[Dict[str, object]] = []
        self.page_h1_count = 0
        self.page_links: List[str] = []
        self.scripts: List[Optional[str]] = []
        self.text_parts: List[str] = []
        self._current: Optional[Dict[str, object]] = None
        self._section_depth = 0
        self._active_evidence: Optional[Dict[str, object]] = None
        self._element_stack: List[Tuple[str, bool]] = []

    @staticmethod
    def _suppresses_visibility(
        tag: str, attributes: Dict[str, Optional[str]]
    ) -> bool:
        return (
            tag in {"details", "dialog"}
            or "hidden" in attributes
            or (attributes.get("aria-hidden") or "").casefold() == "true"
            or "style" in attributes
        )

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        names = [name for name, _value in attrs]
        attributes = dict(attrs)
        duplicate_attributes = len(names) != len(set(names))
        suppresses_visibility = self._suppresses_visibility(tag, attributes)

        if tag == "h1":
            self.page_h1_count += 1
        if tag == "a" and attributes.get("href"):
            self.page_links.append(str(attributes["href"]))
        if tag == "script":
            self.scripts.append(attributes.get("src"))

        if tag == "section" and "data-capability-detail" in attributes:
            if self._current is None:
                self._current = {
                    "id": attributes.get("data-capability-id"),
                    "state": attributes.get("data-capability-state"),
                    "links": [],
                    "evidence": [],
                    "text_parts": [],
                    "h1Count": 0,
                    "hasDuplicateAttributes": duplicate_attributes,
                    "hasVisibilitySuppressor": suppresses_visibility
                    or any(item[1] for item in self._element_stack),
                    "hasNestedDetail": False,
                }
                self._section_depth = 1
            else:
                self._current["hasNestedDetail"] = True
                self._section_depth += 1
        elif self._current is not None and tag == "section":
            self._section_depth += 1

        if self._current is not None:
            if duplicate_attributes:
                self._current["hasDuplicateAttributes"] = True
            if suppresses_visibility:
                self._current["hasVisibilitySuppressor"] = True
            if tag == "h1":
                self._current["h1Count"] = int(self._current["h1Count"]) + 1
            elif tag == "a" and attributes.get("href"):
                links = self._current["links"]
                if isinstance(links, list):
                    links.append(attributes["href"])
                if self._active_evidence is not None:
                    self._active_evidence["href"] = attributes["href"]
            elif tag == "li" and "data-capability-evidence" in attributes:
                self._active_evidence = {
                    "path": attributes.get("data-capability-evidence"),
                    "reviewedAt": attributes.get("data-reviewed-at"),
                    "href": None,
                    "text_parts": [],
                }

        if tag not in HTML_VOID_ELEMENTS:
            self._element_stack.append((tag, suppresses_visibility))

    def handle_data(self, data: str) -> None:
        self.text_parts.append(data)
        if self._current is not None:
            parts = self._current["text_parts"]
            if isinstance(parts, list):
                parts.append(data)
        if self._active_evidence is not None:
            parts = self._active_evidence["text_parts"]
            if isinstance(parts, list):
                parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        self.text_parts.append(" ")
        if self._current is not None:
            parts = self._current["text_parts"]
            if isinstance(parts, list):
                parts.append(" ")
        if tag == "li" and self._active_evidence is not None:
            parts = self._active_evidence.pop("text_parts")
            self._active_evidence["text"] = (
                " ".join("".join(parts).split()) if isinstance(parts, list) else ""
            )
            if self._current is not None:
                evidence = self._current["evidence"]
                if isinstance(evidence, list):
                    evidence.append(self._active_evidence)
            self._active_evidence = None

        if self._current is not None and tag == "section":
            self._section_depth -= 1
            if self._section_depth == 0:
                parts = self._current.pop("text_parts")
                self._current["text"] = (
                    " ".join("".join(parts).split())
                    if isinstance(parts, list)
                    else ""
                )
                self.sections.append(self._current)
                self._current = None

        if self._element_stack:
            if self._element_stack[-1][0] == tag:
                self._element_stack.pop()
            else:
                for position in range(len(self._element_stack) - 1, -1, -1):
                    if self._element_stack[position][0] == tag:
                        del self._element_stack[position:]
                        break


class ResearchCollector(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.has_json_action = False
        self.atom_actions: List[Dict[str, object]] = []
        self.cards: List[Dict[str, object]] = []
        self.theme_options: List[Dict[str, object]] = []
        self.has_controls = False
        self.has_query = False
        self.has_theme_select = False
        self.has_results = False
        self.has_count = False
        self.has_empty_state = False
        self.reset_actions: List[Dict[str, object]] = []
        self.scripts: List[Dict[str, object]] = []
        self.invalid_archive_structure = False
        self._active_atom_action: Optional[Dict[str, object]] = None
        self._active_reset_action: Optional[Dict[str, object]] = None
        self._current_card: Optional[Dict[str, object]] = None
        self._current_theme_option: Optional[Dict[str, object]] = None
        self._in_controls = False
        self._in_theme_select = False
        self._seen_results = False
        self._element_stack: List[Tuple[str, bool]] = []

    @staticmethod
    def _suppresses_visibility(
        tag: str, attributes: Dict[str, Optional[str]]
    ) -> bool:
        return (
            tag in {"details", "dialog"}
            or "hidden" in attributes
            or "inert" in attributes
            or (attributes.get("aria-hidden") or "").casefold() == "true"
            or "style" in attributes
        )

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        names = [name for name, _value in attrs]
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        has_duplicates = len(names) != len(set(names))
        suppresses_visibility = self._suppresses_visibility(tag, attributes)
        if tag == "a" and attributes.get("href") == "index.json":
            self.has_json_action = True
        if tag == "a" and attributes.get("type") == "application/atom+xml":
            self._active_atom_action = {
                "attributes": attributes,
                "text_parts": [],
                "visible": (
                    len(names) == len(set(names))
                    and not suppresses_visibility
                    and not any(item[1] for item in self._element_stack)
                ),
            }
        elif self._active_atom_action is not None and suppresses_visibility:
            self._active_atom_action["visible"] = False

        if tag == "article" and "data-research-card" in attributes:
            if self._current_card is not None:
                self.invalid_archive_structure = True
            self._current_card = {
                "path": attributes.get("data-research-path"),
                "theme": attributes.get("data-theme"),
                "search": attributes.get("data-search"),
                "hidden": "hidden" in attributes,
                "hasVisibilitySuppressor": suppresses_visibility
                or any(item[1] for item in self._element_stack),
                "role": attributes.get("role"),
                "links": [],
                "text_parts": [],
            }
            if "note-card" not in classes or has_duplicates:
                self.invalid_archive_structure = True
        elif self._current_card is not None:
            if suppresses_visibility:
                self._current_card["hasVisibilitySuppressor"] = True
            if tag == "a" and attributes.get("href"):
                links = self._current_card["links"]
                if isinstance(links, list):
                    links.append(attributes["href"])

        if tag == "form" and "data-research-controls" in attributes:
            self._in_controls = True
            self.has_controls = (
                "research-controls" in classes
                and attributes.get("aria-label") == "Filter reviewed research"
                and attributes.get("action") == "./"
                and attributes.get("method") == "get"
                and not has_duplicates
            )
        if tag == "button" and "data-research-reset" in attributes:
            self._active_reset_action = {
                "attributes": attributes,
                "text_parts": [],
                "visible": (
                    self._in_controls
                    and not has_duplicates
                    and not suppresses_visibility
                    and not any(item[1] for item in self._element_stack)
                ),
            }
        elif self._active_reset_action is not None and suppresses_visibility:
            self._active_reset_action["visible"] = False
        if tag == "input" and attributes.get("name") == "q":
            self.has_query = (
                attributes.get("id") == "research-query"
                and attributes.get("type") == "search"
                and attributes.get("maxlength") == "120"
                and attributes.get("autocomplete") == "off"
                and not has_duplicates
            )
        if tag == "select" and attributes.get("name") == "theme":
            self._in_theme_select = True
            self.has_theme_select = (
                attributes.get("id") == "research-theme" and not has_duplicates
            )
        elif tag == "option" and self._in_theme_select:
            self._current_theme_option = {
                "value": attributes.get("value"),
                "text_parts": [],
            }
            if has_duplicates:
                self.invalid_archive_structure = True
        if (
            "data-research-results" in attributes
            and attributes.get("role") == "list"
            and "research-grid" in classes
            and not has_duplicates
        ):
            self.has_results = True
            self._seen_results = True
        if (
            "data-research-count" in attributes
            and attributes.get("aria-live") == "polite"
            and attributes.get("aria-atomic") == "true"
            and not has_duplicates
        ):
            self.has_count = True
        if (
            "data-research-empty" in attributes
            and "hidden" in attributes
            and attributes.get("role") == "status"
            and not has_duplicates
        ):
            self.has_empty_state = True
        if tag == "script":
            self.scripts.append(
                {
                    "attributes": attributes,
                    "afterResults": self._seen_results,
                    "directBodyChild": bool(
                        self._element_stack and self._element_stack[-1][0] == "body"
                    ),
                    "hasDuplicateAttributes": has_duplicates,
                }
            )

        if tag not in HTML_VOID_ELEMENTS:
            self._element_stack.append((tag, suppresses_visibility))

    def handle_data(self, data: str) -> None:
        if self._active_atom_action is not None:
            parts = self._active_atom_action["text_parts"]
            if isinstance(parts, list):
                parts.append(data)
        if self._active_reset_action is not None:
            parts = self._active_reset_action["text_parts"]
            if isinstance(parts, list):
                parts.append(data)
        if self._current_card is not None:
            parts = self._current_card["text_parts"]
            if isinstance(parts, list):
                parts.append(data)
        if self._current_theme_option is not None:
            parts = self._current_theme_option["text_parts"]
            if isinstance(parts, list):
                parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._active_atom_action is not None:
            parts = self._active_atom_action.pop("text_parts")
            self._active_atom_action["text"] = (
                " ".join("".join(parts).split()) if isinstance(parts, list) else ""
            )
            self.atom_actions.append(self._active_atom_action)
            self._active_atom_action = None
        if tag == "button" and self._active_reset_action is not None:
            parts = self._active_reset_action.pop("text_parts")
            self._active_reset_action["text"] = (
                " ".join("".join(parts).split()) if isinstance(parts, list) else ""
            )
            self.reset_actions.append(self._active_reset_action)
            self._active_reset_action = None
        if tag == "article" and self._current_card is not None:
            parts = self._current_card.pop("text_parts")
            self._current_card["text"] = (
                " ".join("".join(parts).split()) if isinstance(parts, list) else ""
            )
            self.cards.append(self._current_card)
            self._current_card = None
        if tag == "option" and self._current_theme_option is not None:
            parts = self._current_theme_option.pop("text_parts")
            self._current_theme_option["text"] = (
                " ".join("".join(parts).split()) if isinstance(parts, list) else ""
            )
            self.theme_options.append(self._current_theme_option)
            self._current_theme_option = None
        elif tag == "select" and self._in_theme_select:
            self._in_theme_select = False
        if tag == "form" and self._in_controls:
            self._in_controls = False
        if self._element_stack:
            if self._element_stack[-1][0] == tag:
                self._element_stack.pop()
            else:
                for position in range(len(self._element_stack) - 1, -1, -1):
                    if self._element_stack[position][0] == tag:
                        del self._element_stack[position:]
                        break


class ReleaseCollector(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.boundary: Dict[str, object] = {
            "id": None,
            "state": None,
            "links": [],
            "text_parts": [],
        }
        self.cards: List[Dict[str, object]] = []
        self.has_json_link = False
        self.has_atom_link = False
        self.has_atom_action = False
        self._in_boundary = False
        self._current_card: Optional[Dict[str, object]] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "section" and "data-release-boundary" in attributes:
            self._in_boundary = True
            self.boundary["id"] = attributes.get("data-boundary-id")
            self.boundary["state"] = attributes.get("data-boundary-state")
        elif tag == "article" and "release-card" in classes:
            self._current_card = {
                "id": attributes.get("data-release-id"),
                "anchor": attributes.get("id"),
                "publicCommit": attributes.get("data-public-commit"),
                "datetime": None,
                "links": [],
                "text_parts": [],
                "hidden": "hidden" in attributes,
            }

        if tag == "a" and attributes.get("href") == "index.json":
            self.has_json_link = True
        if (
            tag == "a"
            and attributes.get("href") == "feed.atom"
            and attributes.get("type") == "application/atom+xml"
        ):
            self.has_atom_action = True
        if (
            tag == "link"
            and attributes.get("rel") == "alternate"
            and attributes.get("type") == "application/atom+xml"
            and attributes.get("title") == "fast-mlx reviewed releases"
            and attributes.get("href") == "../releases/feed.atom"
        ):
            self.has_atom_link = True
        if self._in_boundary and tag == "a" and attributes.get("href"):
            links = self.boundary["links"]
            if isinstance(links, list):
                links.append(attributes["href"])
        if self._current_card is not None:
            if tag == "time":
                self._current_card["datetime"] = attributes.get("datetime")
            elif tag == "a" and attributes.get("href"):
                links = self._current_card["links"]
                if isinstance(links, list):
                    links.append(attributes["href"])

    def handle_data(self, data: str) -> None:
        if self._in_boundary:
            text_parts = self.boundary["text_parts"]
            if isinstance(text_parts, list):
                text_parts.append(data)
        if self._current_card is not None:
            text_parts = self._current_card["text_parts"]
            if isinstance(text_parts, list):
                text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "section" and self._in_boundary:
            text_parts = self.boundary.pop("text_parts")
            self.boundary["text"] = " ".join(
                "".join(text_parts).split()
            ) if isinstance(text_parts, list) else ""
            self._in_boundary = False
        elif tag == "article" and self._current_card is not None:
            text_parts = self._current_card.pop("text_parts")
            self._current_card["text"] = " ".join(
                "".join(text_parts).split()
            ) if isinstance(text_parts, list) else ""
            self.cards.append(self._current_card)
            self._current_card = None


class ReleaseDetailCollector(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sections: List[Dict[str, object]] = []
        self.page_h1_count = 0
        self.page_links: List[str] = []
        self.scripts: List[Optional[str]] = []
        self.text_parts: List[str] = []
        self._current: Optional[Dict[str, object]] = None
        self._section_depth = 0
        self._element_stack: List[Tuple[str, bool]] = []

    @staticmethod
    def _suppresses_visibility(
        tag: str, attributes: Dict[str, Optional[str]]
    ) -> bool:
        return (
            tag in {"details", "dialog"}
            or "hidden" in attributes
            or (attributes.get("aria-hidden") or "").casefold() == "true"
            or "style" in attributes
        )

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        names = [name for name, _value in attrs]
        attributes = dict(attrs)
        duplicate_attributes = len(names) != len(set(names))
        suppresses_visibility = self._suppresses_visibility(tag, attributes)

        if tag == "h1":
            self.page_h1_count += 1
        if tag == "a" and attributes.get("href"):
            self.page_links.append(str(attributes["href"]))
        if tag == "script":
            self.scripts.append(attributes.get("src"))

        if tag == "section" and "data-release-detail" in attributes:
            if self._current is None:
                self._current = {
                    "id": attributes.get("data-release-id"),
                    "publicCommit": attributes.get("data-public-commit"),
                    "datetime": None,
                    "links": [],
                    "text_parts": [],
                    "h1Count": 0,
                    "hasDuplicateAttributes": duplicate_attributes,
                    "hasVisibilitySuppressor": suppresses_visibility
                    or any(item[1] for item in self._element_stack),
                    "hasNestedDetail": False,
                }
                self._section_depth = 1
            else:
                self._current["hasNestedDetail"] = True
                self._section_depth += 1
        elif self._current is not None and tag == "section":
            self._section_depth += 1

        if self._current is not None:
            if duplicate_attributes:
                self._current["hasDuplicateAttributes"] = True
            if suppresses_visibility:
                self._current["hasVisibilitySuppressor"] = True
            if tag == "h1":
                self._current["h1Count"] = int(self._current["h1Count"]) + 1
            elif tag == "time":
                self._current["datetime"] = attributes.get("datetime")
            elif tag == "a" and attributes.get("href"):
                links = self._current["links"]
                if isinstance(links, list):
                    links.append(attributes["href"])

        if tag not in HTML_VOID_ELEMENTS:
            self._element_stack.append((tag, suppresses_visibility))

    def handle_data(self, data: str) -> None:
        self.text_parts.append(data)
        if self._current is not None:
            parts = self._current["text_parts"]
            if isinstance(parts, list):
                parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        self.text_parts.append(" ")
        if self._current is not None:
            parts = self._current["text_parts"]
            if isinstance(parts, list):
                parts.append(" ")
        if self._current is not None and tag == "section":
            self._section_depth -= 1
            if self._section_depth == 0:
                parts = self._current.pop("text_parts")
                self._current["text"] = (
                    " ".join("".join(parts).split())
                    if isinstance(parts, list)
                    else ""
                )
                self.sections.append(self._current)
                self._current = None

        if self._element_stack:
            if self._element_stack[-1][0] == tag:
                self._element_stack.pop()
            else:
                for position in range(len(self._element_stack) - 1, -1, -1):
                    if self._element_stack[position][0] == tag:
                        del self._element_stack[position:]
                        break


class HomeCurrentCycleCollector(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sections: List[Dict[str, object]] = []
        self._current: Optional[Dict[str, object]] = None
        self._section_depth = 0
        self._element_stack: List[
            Tuple[str, Dict[str, Optional[str]], bool]
        ] = []

    @staticmethod
    def _suppresses_visibility(attributes: Dict[str, Optional[str]]) -> bool:
        classes = set((attributes.get("class") or "").split())
        aria_hidden = (attributes.get("aria-hidden") or "").casefold()
        return (
            "hidden" in attributes
            or aria_hidden == "true"
            or "style" in attributes
            or "benchmark-controls" in classes
            or "data-benchmark-controls" in attributes
        )

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        names = [name for name, _value in attrs]
        has_duplicate_attributes = len(names) != len(set(names))
        attributes = dict(attrs)
        if tag == "section" and "data-current-cycle" in attributes:
            if self._current is not None:
                self.sections.append(self._current)
            self._current = {
                "sectionAttributes": attributes,
                "ancestry": list(self._element_stack),
                "hasDuplicateAttributes": has_duplicate_attributes,
                "hasVisibilitySuppressor": self._suppresses_visibility(attributes),
                "latestReleaseId": attributes.get("data-latest-release-id"),
                "boundaryId": attributes.get("data-boundary-id"),
                "boundaryState": attributes.get("data-boundary-state"),
                "time": None,
                "links": [],
                "statusCounts": [],
                "text_parts": [],
                "inventoryRole": None,
                "listItemCount": 0,
            }
            self._section_depth = 1
        elif self._current is not None:
            if has_duplicate_attributes:
                self._current["hasDuplicateAttributes"] = True
            if self._suppresses_visibility(attributes):
                self._current["hasVisibilitySuppressor"] = True
            if tag == "section":
                self._section_depth += 1
            if tag == "time":
                self._current["time"] = attributes.get("datetime")
            elif tag == "a" and attributes.get("href"):
                links = self._current["links"]
                if isinstance(links, list):
                    links.append(attributes["href"])
            if "data-capability-status" in attributes:
                counts = self._current["statusCounts"]
                if isinstance(counts, list):
                    counts.append(
                        (
                            attributes.get("data-capability-status"),
                            attributes.get("data-count"),
                        )
                    )
            classes = set((attributes.get("class") or "").split())
            if "capability-list" in classes:
                self._current["inventoryRole"] = attributes.get("role")
            if attributes.get("role") == "listitem":
                count = self._current["listItemCount"]
                self._current["listItemCount"] = (
                    count + 1 if isinstance(count, int) else 1
                )
        if tag not in HTML_VOID_ELEMENTS:
            self._element_stack.append((tag, attributes, has_duplicate_attributes))

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            text_parts = self._current["text_parts"]
            if isinstance(text_parts, list):
                text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "section" and self._current is not None:
            self._section_depth -= 1
            if self._section_depth == 0:
                text_parts = self._current.pop("text_parts")
                self._current["text"] = " ".join(
                    "".join(text_parts).split()
                ) if isinstance(text_parts, list) else ""
                self.sections.append(self._current)
                self._current = None
        for index in range(len(self._element_stack) - 1, -1, -1):
            if self._element_stack[index][0] == tag:
                del self._element_stack[index:]
                break


# sysexits.h EX_USAGE: argparse's own ArgumentParser.error() exits 2, which
# this script's own main() reserves for a real RED "not a directory"
# refusal (see main()). A bad invocation -- an unknown flag or a missing
# required positional -- must never be misread as that refusal.
_EXIT_USAGE_ERROR = 64


class _UsageErrorArgumentParser(argparse.ArgumentParser):
    """Same as ``argparse.ArgumentParser``, except a usage error exits 64
    (EX_USAGE) instead of argparse's default of 2 -- see the comment above
    ``_EXIT_USAGE_ERROR`` for why.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(_EXIT_USAGE_ERROR, f"{self.prog}: error: {message}\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = _UsageErrorArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path, help="generated site directory")
    return parser


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def resolve_target(site: Path, page: Path, raw_link: str) -> Optional[Path]:
    parsed = urlsplit(raw_link)
    if parsed.scheme or parsed.netloc or raw_link.startswith(("mailto:", "#")):
        return None
    if parsed.path.startswith("/"):
        target = site / parsed.path.lstrip("/")
    else:
        target = page.parent / unquote(parsed.path)
    if not parsed.path or parsed.path.endswith("/"):
        target /= "index.html"
    return target.resolve()


def validate_evidence_path(site: Path, raw_path: object, label: str) -> List[str]:
    if not isinstance(raw_path, str) or not raw_path:
        return [f"{label} has an invalid evidence path: {raw_path!r}"]
    target = (site / raw_path / "index.html").resolve()
    try:
        target.relative_to(site)
    except ValueError:
        return [f"{label} evidence path escapes site root: {raw_path!r}"]
    if not target.is_file():
        return [f"{label} evidence page is missing: {raw_path!r}"]
    return []


def relative_href(current_file: str, target_path: str) -> str:
    current_dir = posixpath.dirname(current_file)
    target = target_path.rstrip("/") or "."
    value = posixpath.relpath(target, current_dir or ".")
    if value == ".":
        value = "./"
    elif target_path.endswith("/"):
        value += "/"
    return value


def key_failures(value: object, expected: set[str], label: str) -> List[str]:
    if not isinstance(value, dict):
        return [f"{label} is not an object"]
    actual = set(value)
    if actual == expected:
        return []
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return [f"{label} keys differ from schema; missing={missing} extra={extra}"]


def require_str(
    entry: Dict[str, object], key: str, label: str, failures: List[str]
) -> Optional[str]:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        failures.append(f"{label} has an empty or non-string {key}")
        return None
    if value != value.strip():
        failures.append(f"{label} {key} contains surrounding whitespace")
    return value


def require_nullable_str(
    entry: Dict[str, object], key: str, label: str, failures: List[str]
) -> Optional[str]:
    """Require `key` to be a non-empty, non-whitespace-padded string, or explicitly null."""

    value = entry.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        failures.append(f"{label} {key} must be a non-empty string or null")
        return None
    if value != value.strip():
        failures.append(f"{label} {key} contains surrounding whitespace")
    return value


def parse_release_timestamp(
    entry: Dict[str, object], key: str, label: str, failures: List[str]
) -> Optional[dt.datetime]:
    value = require_str(entry, key, label, failures)
    if value is None:
        return None
    if not RELEASE_TIMESTAMP.fullmatch(value):
        failures.append(f"{label} {key} is not an offset-aware release timestamp")
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        failures.append(f"{label} {key} is not an offset-aware release timestamp")
        return None
    if parsed.utcoffset() is None:
        failures.append(f"{label} {key} is not an offset-aware release timestamp")
        return None
    return parsed


def validate_public_path(site: Path, raw_path: object, label: str) -> List[str]:
    failures: List[str] = []
    if not isinstance(raw_path, str) or not raw_path.strip() or raw_path != raw_path.strip():
        return [f"{label} has an invalid public path: {raw_path!r}"]
    if not PUBLIC_PATH.fullmatch(raw_path):
        return [f"{label} has an invalid public path: {raw_path!r}"]
    if raw_path.endswith("/"):
        target = site / raw_path / "index.html"
    else:
        target = site / raw_path
    try:
        target.resolve().relative_to(site)
    except ValueError:
        failures.append(f"{label} public path escapes site root: {raw_path!r}")
    if not target.is_file():
        failures.append(f"{label} public path target is missing: {raw_path!r}")
    return failures


def validate_release_link(site: Path, value: object, label: str) -> List[str]:
    failures = key_failures(value, RELEASE_LINK_KEYS, label)
    if not isinstance(value, dict):
        return failures
    require_str(value, "label", label, failures)
    failures.extend(validate_public_path(site, value.get("path"), label))
    return failures


def load_release_index(site: Path) -> Tuple[Optional[Dict[str, object]], List[str]]:
    failures: List[str] = []
    release_index_path = site / "releases/index.json"
    try:
        raw_release_index = release_index_path.read_bytes()
    except OSError as exc:
        return None, [f"cannot read releases/index.json: {exc}"]
    if (
        len(raw_release_index) != REVIEWED_RELEASE_INDEX_BYTES
        or hashlib.sha256(raw_release_index).hexdigest() != REVIEWED_RELEASE_INDEX_SHA256
    ):
        failures.append("releases/index.json does not match the reviewed release ledger")
    try:
        release_index = json.loads(raw_release_index.decode("utf-8"))
    except UnicodeDecodeError as exc:
        return None, [*failures, f"releases/index.json is not UTF-8: {exc}"]
    except json.JSONDecodeError as exc:
        return None, [*failures, f"invalid releases/index.json: {exc}"]

    failures.extend(key_failures(release_index, RELEASE_INDEX_KEYS, "releases/index.json"))
    if not isinstance(release_index, dict):
        return None, failures
    if release_index.get("schemaVersion") != 1:
        failures.append("releases/index.json does not use schemaVersion 1")
    if release_index.get("project") != "fast-mlx":
        failures.append("releases/index.json has the wrong project")
    if release_index.get("policy") != "reviewed-public-releases-only":
        failures.append("releases/index.json has the wrong policy")
    if release_index.get("claimBoundary") != "fast-mlx-owned-results-only":
        failures.append("releases/index.json has the wrong claim boundary")

    updated_at = release_index.get("updatedAt")
    if not isinstance(updated_at, str):
        failures.append("releases/index.json has an invalid updatedAt")
    else:
        try:
            dt.date.fromisoformat(updated_at)
        except ValueError:
            failures.append("releases/index.json has an invalid updatedAt")

    boundary = release_index.get("currentBoundary")
    failures.extend(key_failures(boundary, RELEASE_BOUNDARY_KEYS, "current release boundary"))
    if isinstance(boundary, dict):
        if boundary.get("id") != "runtime-model-promotion":
            failures.append("current release boundary must remain runtime-model-promotion")
        if boundary.get("state") != "gated":
            failures.append("current release boundary state must remain gated")
        for key in ("label", "summary"):
            require_str(boundary, key, "current release boundary", failures)
        evidence = boundary.get("evidence")
        failures.extend(
            validate_release_link(site, evidence, "current release boundary evidence")
        )
        if isinstance(evidence, dict) and evidence.get("path") != "methodology/":
            failures.append("current release boundary evidence must remain methodology/")

    releases = release_index.get("releases")
    if not isinstance(releases, list) or not releases:
        failures.append("releases/index.json has no releases")
    else:
        identities = tuple(
            (release.get("id"), release.get("title"))
            for release in releases
            if isinstance(release, dict)
        )
        if identities != REVIEWED_RELEASE_IDENTITIES:
            failures.append("releases/index.json does not match reviewed release identities")
        seen_ids: set[str] = set()
        seen_commits: set[str] = set()
        previous_timestamp: Optional[dt.datetime] = None
        for position, release in enumerate(releases):
            label = f"release index entry {position}"
            failures.extend(key_failures(release, RELEASE_ENTRY_KEYS, label))
            if not isinstance(release, dict):
                continue
            identifier = require_str(release, "id", label, failures)
            if identifier is not None:
                if not SLUG.fullmatch(identifier) or identifier in seen_ids:
                    failures.append(f"{label} has an invalid or duplicate id")
                seen_ids.add(identifier)
            for key in ("title", "summary", "scope"):
                require_str(release, key, label, failures)
            category = require_str(release, "category", label, failures)
            if category is not None and category not in RELEASE_CATEGORIES:
                failures.append(f"{label} has unknown category {category!r}")
            state = require_str(release, "state", label, failures)
            if state is not None and state != "released":
                failures.append(f"{label} is not explicitly released")
            parsed_timestamp = parse_release_timestamp(
                release, "publishedAt", label, failures
            )
            if parsed_timestamp is not None:
                if (
                    previous_timestamp is not None
                    and parsed_timestamp >= previous_timestamp
                ):
                    failures.append("release entries are not strictly newest-first")
                previous_timestamp = parsed_timestamp
            commit = require_str(release, "publicCommit", label, failures)
            if commit is not None:
                if not COMMIT_SHA.fullmatch(commit) or commit in seen_commits:
                    failures.append(f"{label} has an invalid or duplicate publicCommit")
                seen_commits.add(commit)
                expected_source_url = (
                    "https://github.com/bitworks-io/fast-mlx/commit/" + commit
                )
                if release.get("sourceUrl") != expected_source_url:
                    failures.append(f"{label} sourceUrl does not match publicCommit")
            links = release.get("publicLinks")
            if not isinstance(links, list):
                failures.append(f"{label} publicLinks is not a list")
            else:
                seen_paths: set[str] = set()
                for link_position, link in enumerate(links):
                    link_label = f"{label} public link {link_position}"
                    failures.extend(validate_release_link(site, link, link_label))
                    if isinstance(link, dict) and isinstance(link.get("path"), str):
                        if link["path"] in seen_paths:
                            failures.append(f"{label} has a duplicate public link path")
                        seen_paths.add(link["path"])
    return release_index, failures


def render_expected_release_feed(release_index: Dict[str, object]) -> str:
    """Recreate the one canonical Atom document accepted by the Pages validator."""

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
                "href": (
                    PUBLIC_SITE_URL
                    + "releases/"
                    + str(release["id"])
                    + "/"
                ),
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
        + "\n"
    )


def validate_release_feed(site: Path, release_index: Dict[str, object]) -> List[str]:
    failures: List[str] = []
    feed_path = site / "releases/feed.atom"
    if feed_path.is_symlink() or not feed_path.is_file():
        return ["releases/feed.atom must be a regular non-symlink file"]
    try:
        feed_size = feed_path.stat().st_size
    except OSError as exc:
        return [f"cannot stat releases/feed.atom: {exc}"]
    if feed_size > MAX_RELEASE_FEED_BYTES:
        return ["releases/feed.atom exceeds the 1048576-byte limit"]
    try:
        raw_feed = feed_path.read_bytes()
    except OSError as exc:
        return [f"cannot read releases/feed.atom: {exc}"]
    if len(raw_feed) > MAX_RELEASE_FEED_BYTES:
        return ["releases/feed.atom exceeds the 1048576-byte limit"]
    try:
        feed_text = raw_feed.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [f"releases/feed.atom is not UTF-8: {exc}"]
    upper_feed = feed_text.upper()
    if "<!DOCTYPE" in upper_feed or "<!ENTITY" in upper_feed:
        return ["releases/feed.atom contains a forbidden XML declaration"]
    try:
        feed = ET.fromstring(feed_text)
    except ET.ParseError as exc:
        return [f"invalid releases/feed.atom: {exc}"]
    if feed.tag != f"{{{ATOM_NAMESPACE}}}feed":
        failures.append("releases/feed.atom is not an Atom 1.0 feed")

    releases = release_index.get("releases")
    if not isinstance(releases, list) or not releases:
        return failures
    if not all(isinstance(release, dict) for release in releases):
        return failures
    try:
        expected_feed = render_expected_release_feed(release_index)
    except (KeyError, IndexError, TypeError):
        return failures
    if feed_text != expected_feed:
        failures.append("releases/feed.atom does not match releases/index.json")
    return failures


def render_expected_research_index() -> Dict[str, object]:
    """Recreate the exact reviewed research JSON contract independently."""

    return {
        "schemaVersion": 1,
        "project": "fast-mlx",
        "claimBoundary": "fast-mlx-owned-results-only",
        "articles": [dict(article) for article in reviewed_research_articles()],
    }


def render_expected_research_feed() -> str:
    """Recreate the one canonical reviewed-research Atom document."""

    articles = sorted(
        reviewed_research_articles(),
        key=lambda article: (
            article["reviewedAt"],
            article["date"],
            article["path"],
        ),
        reverse=True,
    )
    ET.register_namespace("", ATOM_NAMESPACE)
    atom = lambda name: f"{{{ATOM_NAMESPACE}}}{name}"
    feed = ET.Element(atom("feed"))
    ET.SubElement(feed, atom("title")).text = "fast-mlx reviewed research"
    ET.SubElement(feed, atom("id")).text = PUBLIC_SITE_URL + "research/"
    ET.SubElement(feed, atom("updated")).text = (
        articles[0]["reviewedAt"] + "T00:00:00Z"
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
    for article in articles:
        canonical = PUBLIC_SITE_URL + article["path"]
        entry = ET.SubElement(feed, atom("entry"))
        ET.SubElement(entry, atom("title")).text = article["title"]
        ET.SubElement(entry, atom("id")).text = canonical
        ET.SubElement(entry, atom("published")).text = (
            article["date"] + "T00:00:00Z"
        )
        ET.SubElement(entry, atom("updated")).text = (
            article["reviewedAt"] + "T00:00:00Z"
        )
        ET.SubElement(entry, atom("category"), {"term": article["theme"]})
        ET.SubElement(
            entry,
            atom("link"),
            {
                "rel": "alternate",
                "type": "text/html",
                "href": canonical,
            },
        )
        ET.SubElement(entry, atom("summary")).text = article["summary"]
    ET.indent(feed, space="  ")
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        + ET.tostring(feed, encoding="unicode", short_empty_elements=True)
        + "\n"
    )


def validate_research_feed(site: Path) -> List[str]:
    failures: List[str] = []
    feed_path = site / "research/feed.atom"
    if feed_path.is_symlink() or not feed_path.is_file():
        return ["research/feed.atom must be a regular non-symlink file"]
    try:
        feed_size = feed_path.stat().st_size
    except OSError as exc:
        return [f"cannot stat research/feed.atom: {exc}"]
    if feed_size > MAX_RESEARCH_FEED_BYTES:
        return ["research/feed.atom exceeds the 1048576-byte limit"]
    try:
        raw_feed = feed_path.read_bytes()
    except OSError as exc:
        return [f"cannot read research/feed.atom: {exc}"]
    if len(raw_feed) > MAX_RESEARCH_FEED_BYTES:
        return ["research/feed.atom exceeds the 1048576-byte limit"]
    try:
        feed_text = raw_feed.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [f"research/feed.atom is not UTF-8: {exc}"]
    upper_feed = feed_text.upper()
    if "<!DOCTYPE" in upper_feed or "<!ENTITY" in upper_feed:
        return ["research/feed.atom contains a forbidden XML declaration"]
    try:
        feed = ET.fromstring(feed_text)
    except ET.ParseError as exc:
        return [f"invalid research/feed.atom: {exc}"]
    if feed.tag != f"{{{ATOM_NAMESPACE}}}feed":
        failures.append("research/feed.atom is not an Atom 1.0 feed")
    if feed_text != render_expected_research_feed():
        failures.append(
            "research/feed.atom does not match the reviewed research catalog"
        )
    return failures


def parse_reviewed_atom_timestamp(value: str) -> dt.datetime:
    """Parse a pinned RFC 3339 value before comparing combined-feed order."""

    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid reviewed Atom timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"reviewed Atom timestamp is not timezone-aware: {value!r}")
    return parsed


def render_expected_reviewed_updates_feed(
    release_index: Dict[str, object],
) -> str:
    """Reconstruct the combined feed from pinned release and research authorities."""

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
    for article in reviewed_research_articles():
        canonical = PUBLIC_SITE_URL + article["path"]
        updates.append(
            {
                "kind": "research",
                "title": article["title"],
                "id": canonical,
                "published": article["date"] + "T00:00:00Z",
                "updated": article["reviewedAt"] + "T00:00:00Z",
                "href": canonical,
                "via": "",
                "summary": article["summary"],
            }
        )
    identifiers = [update["id"] for update in updates]
    if not updates or len(identifiers) != len(set(identifiers)):
        raise ValueError("reviewed updates feed has invalid entry identities")
    ordered_updates = sorted(
        updates,
        key=lambda update: (
            parse_reviewed_atom_timestamp(update["updated"]),
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
        + "\n"
    )


def validate_reviewed_updates_feed(
    site: Path, release_index: Dict[str, object]
) -> List[str]:
    failures: List[str] = []
    feed_path = site / "feed.atom"
    if feed_path.is_symlink() or not feed_path.is_file():
        return ["feed.atom must be a regular non-symlink file"]
    try:
        feed_size = feed_path.stat().st_size
    except OSError as exc:
        return [f"cannot stat feed.atom: {exc}"]
    if feed_size > MAX_REVIEWED_UPDATES_FEED_BYTES:
        return ["feed.atom exceeds the 1048576-byte limit"]
    try:
        raw_feed = feed_path.read_bytes()
    except OSError as exc:
        return [f"cannot read feed.atom: {exc}"]
    if len(raw_feed) > MAX_REVIEWED_UPDATES_FEED_BYTES:
        return ["feed.atom exceeds the 1048576-byte limit"]
    try:
        feed_text = raw_feed.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [f"feed.atom is not UTF-8: {exc}"]
    upper_feed = feed_text.upper()
    if "<!DOCTYPE" in upper_feed or "<!ENTITY" in upper_feed:
        return ["feed.atom contains a forbidden XML declaration"]
    try:
        feed = ET.fromstring(feed_text)
    except ET.ParseError as exc:
        return [f"invalid feed.atom: {exc}"]
    if feed.tag != f"{{{ATOM_NAMESPACE}}}feed":
        failures.append("feed.atom is not an Atom 1.0 feed")
    try:
        expected_feed = render_expected_reviewed_updates_feed(release_index)
    except (IndexError, KeyError, TypeError, ValueError):
        return failures
    if feed_text != expected_feed:
        failures.append(
            "feed.atom does not match the reviewed release and research catalogs"
        )
    return failures


def render_expected_sitemap(article_paths: Sequence[str]) -> str:
    """Recreate the one canonical sitemap accepted by the Pages validator."""

    ET.register_namespace("", SITEMAP_NAMESPACE)
    sitemap = ET.Element(f"{{{SITEMAP_NAMESPACE}}}urlset")
    for public_path in (
        *CORE_PUBLIC_PAGE_PATHS,
        *REVIEWED_CAPABILITY_PATHS,
        *REVIEWED_BENCHMARK_PATHS,
        *REVIEWED_RELEASE_PATHS,
        *article_paths,
    ):
        url = ET.SubElement(sitemap, f"{{{SITEMAP_NAMESPACE}}}url")
        ET.SubElement(url, f"{{{SITEMAP_NAMESPACE}}}loc").text = (
            PUBLIC_SITE_URL + public_path
        )
    ET.indent(sitemap, space="  ")
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        + ET.tostring(sitemap, encoding="unicode", short_empty_elements=True)
        + "\n"
    )


def validate_sitemap(site: Path) -> List[str]:
    failures: List[str] = []
    sitemap_path = site / "sitemap.xml"
    if sitemap_path.is_symlink() or not sitemap_path.is_file():
        return ["sitemap.xml must be a regular non-symlink file"]
    try:
        sitemap_size = sitemap_path.stat().st_size
    except OSError as exc:
        return [f"cannot stat sitemap.xml: {exc}"]
    if sitemap_size > MAX_SITEMAP_BYTES:
        return ["sitemap.xml exceeds the 1048576-byte limit"]
    try:
        raw_sitemap = sitemap_path.read_bytes()
    except OSError as exc:
        return [f"cannot read sitemap.xml: {exc}"]
    if len(raw_sitemap) > MAX_SITEMAP_BYTES:
        return ["sitemap.xml exceeds the 1048576-byte limit"]
    try:
        sitemap_text = raw_sitemap.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [f"sitemap.xml is not UTF-8: {exc}"]
    upper_sitemap = sitemap_text.upper()
    if "<!DOCTYPE" in upper_sitemap or "<!ENTITY" in upper_sitemap:
        return ["sitemap.xml contains a forbidden XML declaration"]
    try:
        sitemap = ET.fromstring(sitemap_text)
    except ET.ParseError as exc:
        return [f"invalid sitemap.xml: {exc}"]
    if sitemap.tag != f"{{{SITEMAP_NAMESPACE}}}urlset":
        failures.append("sitemap.xml is not a Sitemap protocol urlset")
    if sitemap_text != render_expected_sitemap(REVIEWED_ARTICLE_PATHS):
        failures.append("sitemap.xml does not match reviewed public routes")
    return failures


def validate_robots(site: Path) -> List[str]:
    robots_path = site / "robots.txt"
    if robots_path.is_symlink() or not robots_path.is_file():
        return ["robots.txt must be a regular non-symlink file"]
    try:
        robots_size = robots_path.stat().st_size
    except OSError as exc:
        return [f"cannot stat robots.txt: {exc}"]
    if robots_size > MAX_ROBOTS_BYTES:
        return ["robots.txt exceeds the 4096-byte limit"]
    try:
        raw_robots = robots_path.read_bytes()
    except OSError as exc:
        return [f"cannot read robots.txt: {exc}"]
    if len(raw_robots) > MAX_ROBOTS_BYTES:
        return ["robots.txt exceeds the 4096-byte limit"]
    try:
        robots_text = raw_robots.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [f"robots.txt is not UTF-8: {exc}"]
    expected = (
        "User-agent: *\n"
        "Allow: /\n"
        f"Sitemap: {PUBLIC_SITE_URL}sitemap.xml\n"
    )
    if robots_text != expected:
        return ["robots.txt does not match the canonical crawl policy"]
    return []


def validate_social_card(site: Path) -> List[str]:
    path = site / SOCIAL_CARD_PATH
    if path.is_symlink() or not path.is_file():
        return [f"{SOCIAL_CARD_PATH} must be a regular non-symlink file"]
    try:
        size = path.stat().st_size
    except OSError as exc:
        return [f"cannot stat {SOCIAL_CARD_PATH}: {exc}"]
    if size != SOCIAL_CARD_BYTES:
        return [f"{SOCIAL_CARD_PATH} has the wrong byte count"]
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [f"cannot read {SOCIAL_CARD_PATH}: {exc}"]
    if len(raw) != SOCIAL_CARD_BYTES:
        return [f"{SOCIAL_CARD_PATH} has the wrong byte count"]
    if hashlib.sha256(raw).hexdigest() != SOCIAL_CARD_SHA256:
        return [f"{SOCIAL_CARD_PATH} has the wrong SHA-256"]
    if (
        raw[:8] != b"\x89PNG\r\n\x1a\n"
        or raw[8:12] != (13).to_bytes(4, "big")
        or raw[12:16] != b"IHDR"
        or int.from_bytes(raw[16:20], "big") != SOCIAL_CARD_WIDTH
        or int.from_bytes(raw[20:24], "big") != SOCIAL_CARD_HEIGHT
        or raw[24] != 8
        or raw[25] != 2
    ):
        return [f"{SOCIAL_CARD_PATH} is not the reviewed 1200x630 RGB PNG"]
    return []


def validate_reviewed_stylesheet(site: Path) -> List[str]:
    path = site / SITE_STYLESHEET_PATH
    if path.is_symlink() or not path.is_file():
        return [f"{SITE_STYLESHEET_PATH} must be a regular non-symlink file"]
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [f"cannot read {SITE_STYLESHEET_PATH}: {exc}"]
    if hashlib.sha256(raw).hexdigest() != SITE_STYLESHEET_SHA256:
        return [f"{SITE_STYLESHEET_PATH} does not match the reviewed stylesheet"]
    return []


def validate_quickstart_page(site: Path) -> List[str]:
    failures: List[str] = []
    path = site / "quickstart/index.html"
    if path.is_symlink() or not path.is_file():
        return ["quickstart/index.html must be a regular non-symlink file"]
    try:
        size = path.stat().st_size
    except OSError as exc:
        return [f"cannot stat quickstart/index.html: {exc}"]
    if size > MAX_QUICKSTART_BYTES:
        return ["quickstart/index.html exceeds the 131072-byte limit"]
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [f"cannot read quickstart/index.html: {exc}"]
    if len(raw) > MAX_QUICKSTART_BYTES:
        return ["quickstart/index.html exceeds the 131072-byte limit"]
    if (
        len(raw) != REVIEWED_QUICKSTART_PAGE_BYTES
        or hashlib.sha256(raw).hexdigest() != REVIEWED_QUICKSTART_PAGE_SHA256
    ):
        failures.append("quickstart/index.html does not match the reviewed page seal")
    try:
        page = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [f"quickstart/index.html is not UTF-8: {exc}"]

    collector = QuickstartCollector()
    try:
        collector.feed(page)
        collector.close()
    except Exception as exc:
        return [f"cannot parse quickstart/index.html: {exc}"]
    if collector._current is not None or collector._active_command is not None:
        failures.append("quickstart/index.html has an incomplete content root")
    if collector.root_count != 1 or len(collector.roots) != 1:
        failures.append(
            "quickstart/index.html must contain exactly one complete quickstart root"
        )
        return failures

    root = collector.roots[0]
    if root.get("rootAttributes") != {"data-quickstart": None}:
        failures.append("quickstart root attributes do not match reviewed contract")
    expected_ancestry = [
        ("html", {"lang": "en"}, False),
        ("body", {}, False),
        ("main", {"id": "content"}, False),
    ]
    if root.get("ancestry") != expected_ancestry:
        failures.append("quickstart root ancestry does not match reviewed contract")
    if root.get("hasDuplicateAttributes"):
        failures.append("quickstart content contains duplicate attributes")
    if root.get("hasVisibilitySuppressor"):
        failures.append("quickstart content contains a visibility suppressor")
    if root.get("forbiddenTags"):
        failures.append("quickstart content contains an interactive or executable tag")
    if (
        root.get("h1Count") != 1
        or collector.page_h1_count != 1
    ):
        failures.append("quickstart page must contain exactly one h1")
    if collector.scripts or collector.inline_style_count:
        failures.append("quickstart page must not contain scripts or inline styles")
    if collector.stylesheet_links != ["../assets/site.css"]:
        failures.append("quickstart page must load only the reviewed stylesheet")

    actual_commands: List[Tuple[object, object]] = []
    commands = root.get("commands")
    if isinstance(commands, list):
        for command in commands:
            if not isinstance(command, dict):
                continue
            identifier = command.get("id")
            if (
                command.get("attributes") != {"data-command": identifier}
                or command.get("hasNestedTag")
                or command.get("hasNestedCommand")
            ):
                failures.append(
                    f"quickstart command {identifier!r} has invalid structure"
                )
            actual_commands.append((identifier, command.get("text")))
    if tuple(actual_commands) != REVIEWED_QUICKSTART_COMMANDS:
        failures.append("quickstart commands do not match the reviewed CLI contract")
    if root.get("links") != list(REVIEWED_QUICKSTART_LINKS):
        failures.append("quickstart action links do not match the reviewed contract")
    text = root.get("text")
    normalized_text = text if isinstance(text, str) else ""
    for required_text in REVIEWED_QUICKSTART_TEXT:
        if " ".join(required_text.split()) not in normalized_text:
            failures.append(
                f"quickstart page is missing reviewed text: {required_text!r}"
            )
    return failures


def validate_license_page(site: Path) -> List[str]:
    path = site / "license/index.html"
    if path.is_symlink() or not path.is_file():
        return ["license/index.html must be a regular non-symlink file"]
    try:
        size = path.stat().st_size
    except OSError as exc:
        return [f"cannot stat license/index.html: {exc}"]
    if size > MAX_LICENSE_PAGE_BYTES:
        return ["license/index.html exceeds the 131072-byte limit"]
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [f"cannot read license/index.html: {exc}"]
    if len(raw) > MAX_LICENSE_PAGE_BYTES:
        return ["license/index.html exceeds the 131072-byte limit"]
    if (
        len(raw) != REVIEWED_LICENSE_PAGE_BYTES
        or hashlib.sha256(raw).hexdigest() != REVIEWED_LICENSE_PAGE_SHA256
    ):
        return ["license/index.html does not match the reviewed page seal"]
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [f"license/index.html is not UTF-8: {exc}"]
    return []


def validate_status_page(site: Path) -> List[str]:
    failures: List[str] = []
    path = site / "status/index.html"
    if path.is_symlink() or not path.is_file():
        return ["status/index.html must be a regular non-symlink file"]
    try:
        size = path.stat().st_size
    except OSError as exc:
        return [f"cannot stat status/index.html: {exc}"]
    if size > MAX_STATUS_BYTES:
        return ["status/index.html exceeds the 131072-byte limit"]
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [f"cannot read status/index.html: {exc}"]
    if len(raw) > MAX_STATUS_BYTES:
        return ["status/index.html exceeds the 131072-byte limit"]
    if (
        len(raw) != REVIEWED_STATUS_PAGE_BYTES
        or hashlib.sha256(raw).hexdigest() != REVIEWED_STATUS_PAGE_SHA256
    ):
        failures.append("status/index.html does not match the reviewed page seal")
    try:
        page = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [f"status/index.html is not UTF-8: {exc}"]

    collector = StatusPageCollector()
    try:
        collector.feed(page)
        collector.close()
    except Exception as exc:
        return [f"cannot parse status/index.html: {exc}"]
    if collector._current is not None:
        failures.append("status/index.html has an incomplete content root")
    if collector.root_count != 1 or len(collector.roots) != 1:
        failures.append("status/index.html must contain exactly one complete status root")
        return failures

    root = collector.roots[0]
    expected_root_attributes = {
        "data-status-page": None,
        "data-latest-release-id": "tagged-distribution-v0-1-10",
        "data-boundary-id": "runtime-model-promotion",
        "data-boundary-state": "gated",
    }
    if root.get("rootAttributes") != expected_root_attributes:
        failures.append("status page root attributes do not match reviewed contract")
    expected_ancestry = [
        ("html", {"lang": "en"}, False),
        ("body", {}, False),
        ("main", {"id": "content"}, False),
    ]
    if root.get("ancestry") != expected_ancestry:
        failures.append("status page root ancestry does not match reviewed contract")
    if root.get("hasDuplicateAttributes"):
        failures.append("status page contains duplicate attributes")
    if root.get("hasVisibilitySuppressor"):
        failures.append("status page contains a visibility suppressor")
    if root.get("forbiddenTags"):
        failures.append("status page contains an interactive or executable tag")
    if root.get("h1Count") != 1 or collector.page_h1_count != 1:
        failures.append("status page must contain exactly one h1")
    if collector.scripts or collector.inline_style_count:
        failures.append("status page must not contain scripts or inline styles")
    if collector.stylesheet_links != ["../assets/site.css"]:
        failures.append("status page must load only the reviewed stylesheet")

    counts = root.get("statusCounts")
    actual_counts: List[Tuple[object, object]] = []
    if isinstance(counts, list):
        for status, count, attributes in counts:
            if attributes != {
                "data-status-count": status,
                "data-count": count,
            }:
                failures.append(
                    f"status summary {status!r} attributes do not match reviewed contract"
                )
            actual_counts.append((status, count))
    if tuple(actual_counts) != REVIEWED_STATUS_COUNTS:
        failures.append("status summary counts do not match reviewed contract")

    capabilities = root.get("capabilities")
    actual_capabilities: List[Tuple[object, object]] = []
    if isinstance(capabilities, list):
        for identifier, state, attributes in capabilities:
            if attributes != {
                "class": "capability-card",
                "data-status-capability": identifier,
                "data-capability-state": state,
            }:
                failures.append(
                    f"status capability {identifier!r} attributes do not match reviewed contract"
                )
            actual_capabilities.append((identifier, state))
    if tuple(actual_capabilities) != REVIEWED_STATUS_CAPABILITIES:
        failures.append("status capability set does not match reviewed contract")

    expected_highlights = tuple(
        (highlight["id"], highlight["decision"])
        for highlight in REVIEWED_BENCHMARK_HIGHLIGHTS
    )
    highlights = root.get("highlights")
    actual_highlights: List[Tuple[object, object]] = []
    if isinstance(highlights, list):
        for identifier, decision, attributes in highlights:
            if attributes != {
                "class": "evidence-card",
                "data-status-highlight": identifier,
                "data-highlight-decision": decision,
            }:
                failures.append(
                    f"status highlight {identifier!r} attributes do not match reviewed contract"
                )
            actual_highlights.append((identifier, decision))
    if tuple(actual_highlights) != expected_highlights:
        failures.append("status highlight set does not match reviewed contract")
    if root.get("links") != list(REVIEWED_STATUS_LINKS):
        failures.append("status page links do not match reviewed contract")

    text = root.get("text")
    normalized_text = text if isinstance(text, str) else ""
    for required_text in REVIEWED_STATUS_TEXT:
        if " ".join(required_text.split()) not in normalized_text:
            failures.append(f"status page is missing reviewed text: {required_text!r}")
    return failures


def validate_primary_navigation_link(
    site: Path, *, label: str, link_text: str, public_path: str
) -> List[str]:
    failures: List[str] = []
    reviewed_files = [
        (public_path + "index.html") if public_path else "index.html"
        for public_path in REVIEWED_PAGE_METADATA
    ]
    reviewed_files.append("404.html")
    for relative in reviewed_files:
        path = site / relative
        if path.is_symlink() or not path.is_file():
            continue
        try:
            page = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            failures.append(f"cannot inspect {relative} {label} navigation: {exc}")
            continue
        collector = PrimaryNavigationCollector()
        try:
            collector.feed(page)
            collector.close()
        except Exception as exc:
            failures.append(f"cannot parse {relative} primary navigation: {exc}")
            continue
        if (
            collector._current is not None
            or collector._active_link is not None
            or collector.nav_count != 1
            or len(collector.navs) != 1
        ):
            failures.append(
                f"{relative} {label} navigation does not match reviewed contract"
            )
            continue
        nav = collector.navs[0]
        expected_nav_ancestry = [
            ("html", {"lang": "en"}, False),
            ("body", {}, False),
            ("header", {"class": "site-header"}, False),
        ]
        if (
            nav.get("attributes")
            != {"class": "nav shell", "aria-label": "Primary navigation"}
            or nav.get("ancestry") != expected_nav_ancestry
            or nav.get("hasDuplicateAttributes")
        ):
            failures.append(
                f"{relative} {label} navigation does not match reviewed contract"
            )
            continue
        current_dir = posixpath.dirname(relative)
        depth = len([part for part in current_dir.split("/") if part])
        expected_attributes: Dict[str, Optional[str]] = {
            "href": "../" * depth + public_path
        }
        if relative == public_path + "index.html":
            expected_attributes["aria-current"] = "page"
        expected_link_ancestry = [
            *expected_nav_ancestry,
            (
                "nav",
                {"class": "nav shell", "aria-label": "Primary navigation"},
                False,
            ),
            ("div", {"class": "nav-links"}, False),
        ]
        links = nav.get("links")
        matching_links = [
            link
            for link in links
            if isinstance(link, dict) and link.get("text") == link_text
        ] if isinstance(links, list) else []
        if (
            len(matching_links) != 1
            or matching_links[0].get("attributes") != expected_attributes
            or matching_links[0].get("ancestry") != expected_link_ancestry
            or matching_links[0].get("hasNestedTag")
        ):
            failures.append(
                f"{relative} {label} navigation does not match reviewed contract"
            )
    return failures


def validate_quickstart_navigation(site: Path) -> List[str]:
    return validate_primary_navigation_link(
        site,
        label="quickstart",
        link_text="Quickstart",
        public_path="quickstart/",
    )


def validate_status_navigation(site: Path) -> List[str]:
    return validate_primary_navigation_link(
        site,
        label="status",
        link_text="Status",
        public_path="status/",
    )


def validate_research_explorer_script(site: Path) -> List[str]:
    path = site / RESEARCH_EXPLORER_SCRIPT_PATH
    if path.is_symlink() or not path.is_file():
        return [
            f"{RESEARCH_EXPLORER_SCRIPT_PATH} must be a regular non-symlink file"
        ]
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [f"cannot read {RESEARCH_EXPLORER_SCRIPT_PATH}: {exc}"]
    if hashlib.sha256(raw).hexdigest() != RESEARCH_EXPLORER_SCRIPT_SHA256:
        return [
            f"{RESEARCH_EXPLORER_SCRIPT_PATH} does not match the reviewed script"
        ]
    return []


def validate_reviewed_home_page(site: Path) -> List[str]:
    path = site / "index.html"
    if path.is_symlink() or not path.is_file():
        return ["index.html must be a regular non-symlink file"]
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [f"cannot read index.html: {exc}"]
    if (
        len(raw) != REVIEWED_HOME_PAGE_BYTES
        or hashlib.sha256(raw).hexdigest() != REVIEWED_HOME_PAGE_SHA256
    ):
        return ["index.html does not match the reviewed home page"]
    return []


def expected_social_properties(
    public_path: str,
    metadata: Tuple[str, str, str, Optional[str]],
) -> Dict[str, List[str]]:
    title, description, page_type, article_section = metadata
    canonical = PUBLIC_SITE_URL + public_path
    properties = {
        "og:title": [title],
        "og:type": [page_type],
        "og:image": [SOCIAL_CARD_URL],
        "og:image:width": [str(SOCIAL_CARD_WIDTH)],
        "og:image:height": [str(SOCIAL_CARD_HEIGHT)],
        "og:image:alt": [SOCIAL_CARD_ALT],
        "og:url": [canonical],
        "og:description": [description],
        "og:site_name": ["fast-mlx"],
    }
    if page_type == "article":
        if article_section is None:
            raise ValueError(f"incomplete reviewed article metadata for {public_path}")
        properties["article:section"] = [article_section]
    elif article_section is not None:
        raise ValueError(f"article metadata attached to core page {public_path}")
    return properties


def expected_atom_links(relative: str, public_path: Optional[str]) -> List[Dict[str, str]]:
    links: List[Dict[str, str]] = []
    if public_path == "research/" or (
        isinstance(public_path, str)
        and SITEMAP_ARTICLE_PATH.fullmatch(public_path) is not None
    ):
        links.append(
            {
                "rel": "alternate",
                "type": "application/atom+xml",
                "title": "fast-mlx reviewed research",
                "href": relative_href(relative, "research/feed.atom"),
            }
        )
    if public_path == "":
        links.append(
            {
                "rel": "alternate",
                "type": "application/atom+xml",
                "title": "fast-mlx reviewed updates",
                "href": "feed.atom",
            }
        )
    current_dir = posixpath.dirname(relative)
    root_prefix = "../" * len(
        [part for part in current_dir.split("/") if part]
    )
    links.append(
        {
            "rel": "alternate",
            "type": "application/atom+xml",
            "title": "fast-mlx reviewed releases",
            "href": root_prefix + "releases/feed.atom",
        }
    )
    return links


def validate_reviewed_head_metadata(site: Path) -> List[str]:
    failures: List[str] = []
    expected_files = {
        (public_path + "index.html") if public_path else "index.html": public_path
        for public_path in REVIEWED_PAGE_METADATA
    }
    allowed_html_files = set(expected_files) | {"404.html", QUALITY_GUIDE_PUBLIC_FILE}
    actual_html_files = {
        path.relative_to(site).as_posix()
        for path in site.rglob("*")
        if (
            path.is_file()
            and not path.is_symlink()
            and path.suffix.casefold() in HTML_LIKE_SUFFIXES
        )
    }
    for relative in sorted(actual_html_files - allowed_html_files):
        failures.append(
            f"unexpected HTML page outside the reviewed route set: {relative}"
        )

    for relative, public_path in expected_files.items():
        path = site / relative
        if path.is_symlink() or not path.is_file():
            continue
        collector = HeadMetadataCollector()
        try:
            collector.feed(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"cannot parse {relative} metadata: {exc}")
            continue
        canonical = PUBLIC_SITE_URL + public_path
        expected_properties = expected_social_properties(
            public_path, REVIEWED_PAGE_METADATA[public_path]
        )
        if (
            collector.invalid_head_structure
            or collector.metadata_outside_head
            or collector.canonicals != [canonical]
            or collector.properties != expected_properties
            or collector.atom_links != expected_atom_links(relative, public_path)
        ):
            failures.append(f"{relative} metadata does not match reviewed contract")

    not_found = site / "404.html"
    if not_found.is_file() and not not_found.is_symlink():
        collector = HeadMetadataCollector()
        try:
            collector.feed(not_found.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"cannot parse 404.html metadata: {exc}")
        else:
            if (
                collector.invalid_head_structure
                or collector.metadata_outside_head
                or collector.canonicals
                or collector.properties
                or collector.atom_links != expected_atom_links("404.html", None)
            ):
                failures.append(
                    "404.html must not publish canonical or social metadata"
                )
    return failures


def validate_research_page(site: Path) -> List[str]:
    research_path = site / "research/index.html"
    if research_path.is_symlink() or not research_path.is_file():
        return []
    collector = ResearchCollector()
    try:
        research_text = research_path.read_text(encoding="utf-8")
        collector.feed(research_text)
    except Exception as exc:
        return [f"cannot parse research/index.html: {exc}"]
    failures: List[str] = []
    if not collector.has_json_action:
        failures.append("research/index.html does not link to research/index.json")
    expected_atom_actions = [
        {
            "attributes": {
                "class": "button secondary",
                "href": "feed.atom",
                "type": "application/atom+xml",
            },
            "text": "Subscribe to reviewed research",
            "visible": True,
        },
        {
            "attributes": {
                "class": "button secondary",
                "href": "../feed.atom",
                "type": "application/atom+xml",
            },
            "text": "Subscribe to all reviewed updates",
            "visible": True,
        },
    ]
    if collector.atom_actions != expected_atom_actions:
        failures.append(
            "research/index.html does not expose the reviewed subscription actions"
        )

    expected_articles = list(reviewed_research_articles())
    expected_paths = [article["path"] for article in expected_articles]
    actual_paths = [card.get("path") for card in collector.cards]
    if (
        len(actual_paths) != len(set(actual_paths))
        or set(actual_paths) != set(expected_paths)
    ):
        failures.append(
            "research archive card set does not match reviewed research catalog"
        )
    elif actual_paths != expected_paths:
        failures.append(
            "research archive cards are not ordered by descending article date"
        )

    cards_by_path = {
        str(card["path"]): card
        for card in collector.cards
        if isinstance(card.get("path"), str)
    }
    for article in expected_articles:
        public_path = article["path"]
        card = cards_by_path.get(public_path)
        if card is None:
            continue
        expected_search = " ".join(
            f'{article["title"]} {article["summary"]} {article["theme"]}'.split()
        )
        if card.get("theme") != article["theme"]:
            failures.append(f"research archive card {public_path!r} has the wrong theme")
        if card.get("search") != expected_search:
            failures.append(
                f"research archive card {public_path!r} has the wrong search text"
            )
        if card.get("hidden") or card.get("hasVisibilitySuppressor"):
            failures.append(
                f"research archive card {public_path!r} is hidden before enhancement"
            )
        if card.get("role") != "listitem":
            failures.append(f"research archive card {public_path!r} is not a list item")
        slug = public_path.split("/")[1]
        if card.get("links") != [slug + "/"]:
            failures.append(f"research archive card {public_path!r} has the wrong link")
        expected_text = " ".join(
            (
                f'{article["date"]} · {article["theme"]} {article["title"]} '
                f'{article["summary"]} Read the note →'
            ).split()
        )
        if card.get("text") != expected_text:
            failures.append(
                f"research archive card {public_path!r} has drifted reviewed text"
            )

    expected_theme_options = [
        {"value": "", "text": "All themes"},
        *[
            {"value": theme, "text": theme}
            for theme in sorted(
                {article["theme"] for article in expected_articles}, key=str.casefold
            )
        ],
    ]
    if collector.theme_options != expected_theme_options:
        failures.append(
            "research archive theme options do not match reviewed research catalog"
        )
    if not collector.has_controls or not collector.has_query or not collector.has_theme_select:
        failures.append("research archive has no exact filter controls")
    if not collector.has_results:
        failures.append("research archive has no reviewed result list")
    if not collector.has_count:
        failures.append("research archive has no live result count")
    if not collector.has_empty_state:
        failures.append("research archive has no hidden status empty state")
    expected_reset_actions = [
        {
            "attributes": {
                "class": "button secondary research-reset",
                "data-research-reset": None,
                "type": "reset",
            },
            "text": "Clear filters",
            "visible": True,
        }
    ]
    if collector.reset_actions != expected_reset_actions:
        failures.append("research archive has no exact visible reset control")
    expected_scripts = [
        {
            "attributes": {
                "src": "../assets/research-explorer.js",
                "defer": None,
            },
            "afterResults": True,
            "directBodyChild": True,
            "hasDuplicateAttributes": False,
        }
    ]
    if collector.scripts != expected_scripts:
        failures.append("research archive does not load only its reviewed script")
    if collector.invalid_archive_structure:
        failures.append("research archive has invalid or duplicate structure")
    boundary = (
        "This page presents only notes already admitted by the reviewed publication "
        "manifest. It performs no external request, ingestion, ranking, benchmark "
        "recomputation, publication action, or authority transition."
    )
    if boundary not in research_text:
        failures.append("research archive has the wrong claim boundary")
    return failures


def validate_release_page(site: Path, release_index: Dict[str, object]) -> List[str]:
    failures: List[str] = []
    release_path = site / "releases/index.html"
    collector = ReleaseCollector()
    try:
        collector.feed(release_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"cannot parse releases/index.html: {exc}"]

    if not collector.has_json_link:
        failures.append("releases/index.html does not link to releases/index.json")
    if not collector.has_atom_link:
        failures.append("releases/index.html does not advertise releases/feed.atom")
    if not collector.has_atom_action:
        failures.append("releases/index.html does not link to releases/feed.atom")
    subscription_collector = ResearchCollector()
    try:
        subscription_collector.feed(release_path.read_text(encoding="utf-8"))
    except Exception as exc:
        failures.append(f"cannot parse releases/index.html subscriptions: {exc}")
    else:
        expected_subscription_actions = [
            {
                "attributes": {
                    "class": "button secondary",
                    "href": "feed.atom",
                    "type": "application/atom+xml",
                },
                "text": "Subscribe to reviewed releases",
                "visible": True,
            },
            {
                "attributes": {
                    "class": "button secondary",
                    "href": "../feed.atom",
                    "type": "application/atom+xml",
                },
                "text": "Subscribe to all reviewed updates",
                "visible": True,
            },
        ]
        if subscription_collector.atom_actions != expected_subscription_actions:
            failures.append(
                "releases/index.html does not expose the reviewed subscription actions"
            )

    boundary = release_index.get("currentBoundary")
    if isinstance(boundary, dict):
        if collector.boundary.get("id") != boundary.get("id"):
            failures.append("release page boundary id does not match releases/index.json")
        if collector.boundary.get("state") != boundary.get("state"):
            failures.append("release page boundary state does not match releases/index.json")
        boundary_text = collector.boundary.get("text")
        normalized_boundary_text = (
            boundary_text if isinstance(boundary_text, str) else ""
        )
        for key in ("label", "summary"):
            value = boundary.get(key)
            if isinstance(value, str) and " ".join(value.split()) not in normalized_boundary_text:
                failures.append(f"release page boundary has the wrong {key}")
        evidence = boundary.get("evidence")
        if isinstance(evidence, dict):
            expected_href = relative_href("releases/index.html", str(evidence.get("path", "")))
            links = collector.boundary.get("links")
            if not isinstance(links, list) or expected_href not in links:
                failures.append("release page boundary evidence link does not match JSON")

    releases = release_index.get("releases")
    expected_releases = releases if isinstance(releases, list) else []
    expected_by_id = {
        release["id"]: release
        for release in expected_releases
        if isinstance(release, dict) and isinstance(release.get("id"), str)
    }
    actual_ids = [card.get("id") for card in collector.cards]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_by_id):
        failures.append("release page card set does not match release ledger")
    expected_order = [release.get("id") for release in expected_releases]
    if (
        len(actual_ids) == len(set(actual_ids))
        and set(actual_ids) == set(expected_by_id)
        and actual_ids != expected_order
    ):
        failures.append("release page cards are not ordered like releases/index.json")
    seen_card_commits: set[object] = set()
    for card in collector.cards:
        identifier = card.get("id")
        release = expected_by_id.get(identifier if isinstance(identifier, str) else "")
        if release is None:
            continue
        commit = release.get("publicCommit")
        if card.get("anchor") != "release-" + str(identifier):
            failures.append(f"release page card {identifier!r} has the wrong anchor")
        if card.get("publicCommit") != commit:
            message = "release page card set does not match release ledger"
            if message not in failures:
                failures.append(message)
            failures.append(f"release page card {identifier!r} has the wrong commit")
        if card.get("publicCommit") in seen_card_commits:
            failures.append(f"release page card {identifier!r} duplicates a commit")
        seen_card_commits.add(card.get("publicCommit"))
        if card.get("hidden"):
            failures.append(f"release page card {identifier!r} is hidden")
        if card.get("datetime") != release.get("publishedAt"):
            failures.append(f"release page card {identifier!r} has the wrong timestamp")
        text = card.get("text")
        normalized_text = text if isinstance(text, str) else ""
        expected_text_values = [
            release.get("state", "").upper(),
            str(release.get("publishedAt", ""))[:10],
            RELEASE_CATEGORY_LABELS.get(str(release.get("category", "")), ""),
            release.get("title"),
            release.get("summary"),
            release.get("scope"),
        ]
        for value in expected_text_values:
            if isinstance(value, str) and value:
                normalized_expected = " ".join(value.split())
                if normalized_expected not in normalized_text:
                    failures.append(
                        f"release page card {identifier!r} does not bind text {normalized_expected!r}"
                    )
        links = card.get("links")
        actual_links = links if isinstance(links, list) else []
        source_url = release.get("sourceUrl")
        if not isinstance(source_url, str) or source_url not in actual_links:
            failures.append(f"release page card {identifier!r} has the wrong source link")
        expected_detail_href = str(identifier) + "/"
        if expected_detail_href not in actual_links:
            failures.append(
                f"release page card {identifier!r} has the wrong release detail link"
            )
        for raw_link in release.get("publicLinks", []):
            if not isinstance(raw_link, dict):
                continue
            path = raw_link.get("path")
            if isinstance(path, str):
                expected_href = relative_href("releases/index.html", path)
                if expected_href not in actual_links:
                    failures.append(
                        f"release page card {identifier!r} is missing public link {path!r}"
                    )
    return failures


def validate_home_current_cycle(
    site: Path,
    release_index: Dict[str, object],
    capability_index: Dict[str, object],
    research_index: Dict[str, object],
) -> List[str]:
    failures: List[str] = []
    collector = HomeCurrentCycleCollector()
    try:
        collector.feed((site / "index.html").read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"cannot parse index.html current cycle: {exc}"]
    if len(collector.sections) != 1:
        return ["index.html must contain exactly one home current-cycle section"]

    current = collector.sections[0]
    releases = release_index.get("releases")
    latest = releases[0] if isinstance(releases, list) and releases else None
    boundary = release_index.get("currentBoundary")
    capabilities = capability_index.get("capabilities")
    articles = research_index.get("articles")
    if not isinstance(latest, dict) or not isinstance(boundary, dict):
        return failures
    if not isinstance(capabilities, list) or not isinstance(articles, list):
        return failures

    expected_section_attributes = {
        "class": "section shell split",
        "aria-labelledby": "current-cycle-heading",
        "data-current-cycle": None,
        "data-latest-release-id": latest.get("id"),
        "data-boundary-id": boundary.get("id"),
        "data-boundary-state": boundary.get("state"),
    }
    if current.get("sectionAttributes") != expected_section_attributes:
        failures.append(
            "home current-cycle section attributes do not match the reviewed contract"
        )
    expected_ancestry = [
        ("html", {"lang": "en"}, False),
        ("body", {}, False),
        ("main", {"id": "content"}, False),
    ]
    if current.get("ancestry") != expected_ancestry:
        failures.append(
            "home current-cycle section ancestry does not match the reviewed contract"
        )
    if current.get("hasDuplicateAttributes") is not False:
        failures.append("home current-cycle section contains duplicate attributes")
    if current.get("hasVisibilitySuppressor") is not False:
        failures.append("home current-cycle section contains a visibility suppressor")

    if current.get("latestReleaseId") != latest.get("id"):
        failures.append(
            "home current-cycle latest release does not match releases/index.json"
        )
    if current.get("boundaryId") != boundary.get("id"):
        failures.append(
            "home current-cycle boundary id does not match releases/index.json"
        )
    if current.get("boundaryState") != boundary.get("state"):
        failures.append(
            "home current-cycle boundary state does not match releases/index.json"
        )
    if current.get("time") != latest.get("publishedAt"):
        failures.append(
            "home current-cycle timestamp does not match releases/index.json"
        )

    text = current.get("text")
    normalized_text = text if isinstance(text, str) else ""
    latest_text = (
        ("state", str(latest.get("state", "")).upper()),
        ("date", str(latest.get("publishedAt", ""))[:10]),
        ("title", latest.get("title")),
        ("summary", latest.get("summary")),
        ("scope", latest.get("scope")),
    )
    for label, value in latest_text:
        if isinstance(value, str) and " ".join(value.split()) not in normalized_text:
            failures.append(f"home current-cycle latest release does not bind {label}")
    for label, value in (
        ("label", boundary.get("label")),
        ("summary", boundary.get("summary")),
    ):
        if isinstance(value, str) and " ".join(value.split()) not in normalized_text:
            failures.append(f"home current-cycle boundary has the wrong {label}")

    expected_capability_text = f"{len(capabilities)} reviewed capabilities"
    if expected_capability_text not in normalized_text:
        failures.append(
            "home current-cycle capability count does not match capabilities/index.json"
        )
    expected_research_text = f"{len(articles)} published research notes"
    if expected_research_text not in normalized_text:
        failures.append(
            "home current-cycle research count does not match research/index.json"
        )
    expected_release_text = f"{len(releases)} reviewed releases"
    if expected_release_text not in normalized_text:
        failures.append(
            "home current-cycle release count does not match releases/index.json"
        )

    expected_status_counts = {
        status: sum(
            isinstance(capability, dict) and capability.get("status") == status
            for capability in capabilities
        )
        for status in CAPABILITY_STATUSES
    }
    actual_status_counts: Dict[str, int] = {}
    raw_status_counts = current.get("statusCounts")
    if isinstance(raw_status_counts, list):
        for raw_status, raw_count in raw_status_counts:
            if (
                not isinstance(raw_status, str)
                or raw_status in actual_status_counts
                or not isinstance(raw_count, str)
            ):
                actual_status_counts = {}
                break
            try:
                actual_status_counts[raw_status] = int(raw_count)
            except ValueError:
                actual_status_counts = {}
                break
    if actual_status_counts != expected_status_counts:
        failures.append(
            "home current-cycle capability status counts do not match capabilities/index.json"
        )

    links = current.get("links")
    actual_links = links if isinstance(links, list) else []
    expected_source = latest.get("sourceUrl")
    if not isinstance(expected_source, str) or expected_source not in actual_links:
        failures.append("home current-cycle latest release has the wrong source link")
    expected_release_detail = "releases/" + str(latest.get("id", "")) + "/"
    if expected_release_detail not in actual_links:
        failures.append("home current-cycle latest release has the wrong detail link")
    for expected_link, label in (
        ("capabilities/", "capability"),
        ("research/", "research"),
        ("releases/", "release"),
    ):
        if expected_link not in actual_links:
            failures.append(f"home current-cycle has the wrong {label} inventory link")
    evidence = boundary.get("evidence")
    expected_evidence = evidence.get("path") if isinstance(evidence, dict) else None
    if not isinstance(expected_evidence, str) or expected_evidence not in actual_links:
        failures.append(
            "home current-cycle boundary evidence link does not match releases/index.json"
        )
    if current.get("inventoryRole") != "list" or current.get("listItemCount") != 4:
        failures.append("home current-cycle evidence inventory is not an accessible list")

    status_text = "; ".join(
        f"{expected_status_counts[status]} {CAPABILITY_STATUS_LABELS[status].casefold()}"
        for status in (
            "implemented",
            "promoted-scoped",
            "experimental",
            "shelved",
        )
    )
    expected_text = " ".join(
        [
            "Current reviewed cycle",
            str(latest.get("state", "")).upper(),
            str(latest.get("publishedAt", ""))[:10],
            str(latest.get("title", "")),
            str(latest.get("summary", "")),
            "Boundary:",
            str(latest.get("scope", "")),
            "Inspect the latest reviewed milestone →",
            "Inspect commit",
            str(latest.get("publicCommit", ""))[:12],
            "→",
            expected_capability_text,
            status_text,
            "Inspect capability states →",
            expected_research_text,
            "Dated investigations preserve promoted, shelved, rejected, and diagnostic outcomes.",
            "Read the evidence trail →",
            expected_release_text,
            "Each public milestone names its exact commit and unchanged claim boundary.",
            "Follow the release ledger →",
            str(boundary.get("state", "")).upper(),
            str(boundary.get("label", "")),
            str(boundary.get("summary", "")),
            str(evidence.get("label", "")) if isinstance(evidence, dict) else "",
            "→",
        ]
    )
    expected_text = " ".join(expected_text.split())
    if normalized_text != expected_text:
        failures.append("home current-cycle text does not match reviewed indexes")
    return failures


def reviewed_benchmark_cards() -> Dict[str, Dict[str, str]]:
    cards: Dict[str, Dict[str, str]] = {}
    for highlight in REVIEWED_BENCHMARK_HIGHLIGHTS:
        identifier = str(highlight["id"])
        evidence = highlight["evidence"]
        if not isinstance(evidence, dict):
            raise ValueError(f"reviewed benchmark {identifier!r} has invalid evidence")
        cards[identifier] = {
            **{
                key: str(highlight[key])
                for key in (
                    "metric",
                    "label",
                    "model",
                    "hardware",
                    "workload",
                    "date",
                    "decision",
                    "caveat",
                )
            },
            "detail": f"{identifier}/",
            "evidence": f'../{str(evidence["path"]).rstrip("/")}/',
        }
    return cards


def reviewed_capability_records() -> Tuple[Dict[str, object], ...]:
    """Recreate reviewed capabilities without trusting generated capabilities/index.json."""

    articles_by_slug = {
        article["path"].rstrip("/").split("/")[-1]: article
        for article in reviewed_research_articles()
    }
    records: List[Dict[str, object]] = []
    for capability in REVIEWED_CAPABILITIES:
        evidence_records: List[Dict[str, str]] = []
        for slug in capability["evidenceSlugs"]:
            article = articles_by_slug[str(slug)]
            evidence_records.append(
                {
                    "slug": str(slug),
                    "title": article["title"],
                    "path": article["path"],
                    "reviewedAt": article["reviewedAt"],
                }
            )
        records.append(
            {
                "id": str(capability["id"]),
                "name": str(capability["name"]),
                "status": str(capability["status"]),
                "summary": str(capability["summary"]),
                "scope": str(capability["scope"]),
                "evidence": evidence_records,
            }
        )
    return tuple(records)


def validate_capability_cards(site: Path) -> List[str]:
    failures: List[str] = []
    path = site / "capabilities/index.html"
    if path.is_symlink() or not path.is_file():
        return ["capabilities/index.html must be a regular non-symlink file"]
    try:
        raw_page = path.read_bytes()
    except OSError as exc:
        return [f"cannot read capabilities/index.html: {exc}"]
    if len(raw_page) > MAX_CAPABILITY_CATALOG_BYTES:
        failures.append("capabilities/index.html exceeds the 131072-byte limit")
    if (
        len(raw_page) != REVIEWED_CAPABILITIES_PAGE_BYTES
        or hashlib.sha256(raw_page).hexdigest()
        != REVIEWED_CAPABILITIES_PAGE_SHA256
    ):
        failures.append(
            "capabilities/index.html does not match the reviewed page seal"
        )
    collector = CapabilityCardCollector()
    try:
        collector.feed(raw_page.decode("utf-8"))
        collector.close()
    except Exception as exc:
        return [f"cannot parse capabilities/index.html cards: {exc}"]
    expected_records = reviewed_capability_records()
    actual_ids = [card.get("id") for card in collector.cards]
    expected_ids = [record["id"] for record in expected_records]
    if actual_ids != expected_ids or len(actual_ids) != len(set(actual_ids)):
        failures.append("capability card set does not match reviewed capability records")
    cards_by_id = {
        str(card["id"]): card
        for card in collector.cards
        if isinstance(card.get("id"), str)
    }
    for record in expected_records:
        identifier = str(record["id"])
        card = cards_by_id.get(identifier)
        if card is None:
            continue
        if card.get("state") != record["status"]:
            failures.append(f"capability card {identifier!r} has the wrong state")
        if card.get("hidden"):
            failures.append(f"capability card {identifier!r} is hidden")
        text = card.get("text")
        normalized_text = text if isinstance(text, str) else ""
        expected_text_bits = [
            CAPABILITY_STATUS_LABELS[str(record["status"])].upper(),
            str(record["name"]),
            str(record["summary"]),
            "Scope:",
            str(record["scope"]),
            *[
                str(evidence["title"]) + " →"
                for evidence in record["evidence"]
                if isinstance(evidence, dict)
            ],
            "Open capability details →",
        ]
        expected_text = " ".join(" ".join(expected_text_bits).split())
        if normalized_text != expected_text:
            failures.append(f"capability card {identifier!r} has drifted reviewed text")
        expected_links = [
            relative_href("capabilities/index.html", str(evidence["path"]))
            for evidence in record["evidence"]
            if isinstance(evidence, dict)
        ]
        expected_links.append(identifier + "/")
        if card.get("links") != expected_links:
            failures.append(f"capability card {identifier!r} has the wrong action links")
    return failures


def validate_capability_detail_pages(site: Path) -> List[str]:
    failures: List[str] = []
    capability_root = site / "capabilities"
    expected_entries = {"index.html", "index.json"} | {
        str(capability["id"]) for capability in REVIEWED_CAPABILITIES
    }
    if capability_root.is_symlink() or not capability_root.is_dir():
        return ["capabilities must be a regular non-symlink directory"]
    try:
        actual_entries = {entry.name for entry in capability_root.iterdir()}
    except OSError as exc:
        return [f"cannot inspect capability detail routes: {exc}"]
    for extra in sorted(actual_entries - expected_entries):
        failures.append(f"unexpected capability route outside reviewed set: {extra}")

    records = reviewed_capability_records()
    for record in records:
        identifier = str(record["id"])
        relative = f"capabilities/{identifier}/index.html"
        directory = site / "capabilities" / identifier
        path = directory / "index.html"
        if directory.is_symlink() or not directory.is_dir():
            failures.append(
                f"capability detail route {identifier!r} must be a regular non-symlink directory"
            )
            continue
        if path.is_symlink() or not path.is_file():
            failures.append(f"{relative} must be a regular non-symlink file")
            continue
        try:
            raw_detail = path.read_bytes()
        except OSError as exc:
            failures.append(f"cannot read {relative}: {exc}")
            continue
        if len(raw_detail) > MAX_CAPABILITY_DETAIL_BYTES:
            failures.append(f"{relative} exceeds the 131072-byte limit")
        expected_size, expected_sha256 = REVIEWED_CAPABILITY_DETAIL_SEALS[identifier]
        if (
            len(raw_detail) != expected_size
            or hashlib.sha256(raw_detail).hexdigest() != expected_sha256
        ):
            failures.append(
                f"capability detail {identifier!r} does not match the reviewed page seal"
            )
        try:
            detail_text = raw_detail.decode("utf-8")
        except UnicodeDecodeError as exc:
            failures.append(f"cannot parse {relative}: {exc}")
            continue
        collector = CapabilityDetailCollector()
        try:
            collector.feed(detail_text)
            collector.close()
        except Exception as exc:
            failures.append(f"cannot parse {relative}: {exc}")
            continue
        if len(collector.sections) != 1:
            failures.append(
                f"capability detail {identifier!r} must contain exactly one detail section"
            )
            continue

        detail = collector.sections[0]
        state = str(record["status"])
        expected_text = " ".join(
            [
                "Reviewed capability",
                CAPABILITY_STATUS_LABELS[state].upper(),
                CAPABILITY_STATUS_DESCRIPTIONS[state],
                str(record["name"]),
                str(record["summary"]),
                "Scope:",
                str(record["scope"]),
                "Reviewed evidence",
                *[
                    " ".join(
                        [
                            str(evidence["title"]),
                            "→",
                            "Path:",
                            str(evidence["path"]),
                            "·",
                            "Reviewed:",
                            str(evidence["reviewedAt"]),
                        ]
                    )
                    for evidence in record["evidence"]
                    if isinstance(evidence, dict)
                ],
                "Back to all capabilities",
                "Open capabilities/index.json →",
                "Read the methodology →",
            ]
        )
        expected_text = " ".join(expected_text.split())
        expected_links = [
            relative_href(relative, str(evidence["path"]))
            for evidence in record["evidence"]
            if isinstance(evidence, dict)
        ]
        expected_links.extend(["../", "../index.json", "../../methodology/"])
        expected_evidence = [
            {
                "path": str(evidence["path"]),
                "reviewedAt": str(evidence["reviewedAt"]),
                "href": relative_href(relative, str(evidence["path"])),
                "text": " ".join(
                    [
                        str(evidence["title"]),
                        "→",
                        "Path:",
                        str(evidence["path"]),
                        "·",
                        "Reviewed:",
                        str(evidence["reviewedAt"]),
                    ]
                ),
            }
            for evidence in record["evidence"]
            if isinstance(evidence, dict)
        ]

        if detail.get("id") != identifier:
            failures.append(f"capability detail {identifier!r} has the wrong id")
        if detail.get("state") != state:
            failures.append(f"capability detail {identifier!r} has the wrong state")
        if detail.get("text") != expected_text:
            failures.append(f"capability detail {identifier!r} has drifted reviewed text")
        if detail.get("evidence") != expected_evidence:
            failures.append(f"capability detail {identifier!r} has the wrong evidence")
        for evidence in expected_evidence:
            reviewed_at = evidence["reviewedAt"]
            try:
                dt.date.fromisoformat(reviewed_at)
            except ValueError:
                failures.append(
                    f"capability detail {identifier!r} has a non-date reviewedAt"
                )
            if "T" in reviewed_at:
                failures.append(
                    f"capability detail {identifier!r} has a non-date reviewedAt"
                )
        if detail.get("links") != expected_links:
            failures.append(f"capability detail {identifier!r} has the wrong action links")
        if (
            detail.get("h1Count") != 1
            or collector.page_h1_count != 1
            or detail.get("hasNestedDetail")
        ):
            failures.append(f"capability detail {identifier!r} has the wrong heading structure")
        if detail.get("hasDuplicateAttributes"):
            failures.append(f"capability detail {identifier!r} has duplicate attributes")
        if detail.get("hasVisibilitySuppressor"):
            failures.append(f"capability detail {identifier!r} is hidden")
        if collector.scripts:
            failures.append(f"capability detail {identifier!r} must not load scripts")
        page_text = " ".join("".join(collector.text_parts).split())
        expected_boundary = (
            "Claim boundary A permalink is not new authority. "
            "This page creates no broader support, measurement, runtime, model, "
            "acquisition, publication, admission, launchability, or containment "
            "authority. It only exposes one reviewed capability record and its "
            "already-published evidence."
        )
        if expected_boundary not in page_text:
            failures.append(f"capability detail {identifier!r} has the wrong claim boundary")
        if collector.page_links.count("../../methodology/") != 2:
            failures.append(f"capability detail {identifier!r} has the wrong methodology link")
    return failures


class QualityCardCollector(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.cards: List[Dict[str, object]] = []
        self._current: Optional[Dict[str, object]] = None
        self._details_depth = 0

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attributes = dict(attrs)
        if tag == "article" and "data-quality-card" in attributes:
            self._current = {
                "id": attributes.get("data-quality-card"),
                "verdict": attributes.get("data-quality-verdict"),
                "has_details": False,
                "text_parts": [],
            }
            self._details_depth = 0
        elif self._current is not None and tag == "details":
            self._details_depth += 1
            self._current["has_details"] = True

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            parts = self._current["text_parts"]
            if isinstance(parts, list):
                parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "details" and self._current is not None and self._details_depth > 0:
            self._details_depth -= 1
        if tag == "article" and self._current is not None:
            parts = self._current.pop("text_parts")
            self._current["text"] = (
                " ".join("".join(parts).split()) if isinstance(parts, list) else ""
            )
            self.cards.append(self._current)
            self._current = None


class ServedBenchmarkCollector(html.parser.HTMLParser):
    """Collects each `<article data-served-benchmark="...">` block from
    `benchmarks/index.html` -- mirrors `QualityCardCollector` above (same
    shape, one flag renamed) for the served-engine ledger section (V3)."""

    def __init__(self) -> None:
        super().__init__()
        self.cards: List[Dict[str, object]] = []
        self._current: Optional[Dict[str, object]] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attributes = dict(attrs)
        if tag == "article" and "data-served-benchmark" in attributes:
            self._current = {
                "id": attributes.get("data-served-benchmark"),
                "cardId": attributes.get("data-card-id"),
                "text_parts": [],
                "links": [],
            }
            return
        if tag == "a" and self._current is not None and "href" in attributes:
            links = self._current["links"]
            if isinstance(links, list):
                links.append(attributes.get("href"))

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            parts = self._current["text_parts"]
            if isinstance(parts, list):
                parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "article" and self._current is not None:
            parts = self._current.pop("text_parts")
            self._current["text"] = (
                " ".join("".join(parts).split()) if isinstance(parts, list) else ""
            )
            self.cards.append(self._current)
            self._current = None


def _served_benchmark_marker_violation(text: str) -> Optional[str]:
    """Accumulate-style twin of `build_public_site._served_benchmark_marker_violation`
    -- see that function's docstring for the neutralise-then-scan order this
    mirrors byte-for-byte."""

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
    if SERVED_BENCHMARK_HOST_SHORTHAND.search(text):
        return "the fleet host shorthand"
    if SERVED_BENCHMARK_IPV4.search(text):
        return "an IPv4 address"
    if "<path>" in text:
        return "a <path> placeholder"
    if "~" in text:
        return "a literal '~'"
    return None


def _served_benchmark_rerun_failures(key: str, value: object, label: str) -> List[str]:
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


def _served_benchmark_entry_failures(
    raw_entry: object, cards_by_id: Dict[str, Dict[str, object]], label: str
) -> List[str]:
    """Accumulate-style twin of
    `build_public_site.validate_served_benchmark_entry` (`L3`-`L12`) -- see
    that function's docstring for the full rule set this re-implements
    against an already-BUILT `quality/index.json`, rather than the
    in-process `quality_manifest` object the build-time loader has."""

    failures: List[str] = []
    failures.extend(key_failures(raw_entry, SERVED_BENCHMARK_ENTRY_KEYS, label))
    if failures or not isinstance(raw_entry, dict):
        return failures
    entry = raw_entry

    identifier = require_str(entry, "id", label, failures)
    measured_at = require_str(entry, "measuredAt", label, failures)
    if measured_at is not None and not SERVED_BENCHMARK_MEASURED_AT.fullmatch(measured_at):
        failures.append(f"{label} measuredAt is not a UTC Z timestamp")
    card_id = require_str(entry, "cardId", label, failures)
    card: Optional[Dict[str, object]] = None
    if card_id is not None:
        if not QUALITY_CARD_ID.fullmatch(card_id):
            failures.append(f"{label} has an invalid cardId")
        card = cards_by_id.get(card_id)
        if card is None:
            failures.append(f"{label} cardId {card_id!r} is not a published quality card")
    pack_label = require_str(entry, "packLabel", label, failures)
    if pack_label is not None and len(pack_label) > 60:
        failures.append(f"{label} packLabel exceeds 60 characters")
    pack_revision = require_str(entry, "packRevision", label, failures)
    if pack_revision is not None and not COMMIT_SHA.fullmatch(pack_revision):
        failures.append(f"{label} packRevision is not a 40-hex revision")
    hardware_class = require_str(entry, "hardwareClass", label, failures)
    if hardware_class is not None and (
        not QUALITY_CARD_HARDWARE_CLASS.fullmatch(hardware_class)
        or hardware_class == QUALITY_CARD_HARDWARE_CLASS_SENTINEL
    ):
        failures.append(f"{label} hardwareClass is invalid")
    chip = require_str(entry, "chip", label, failures)

    if card is not None:
        card_model = card.get("model") if isinstance(card.get("model"), dict) else {}
        if pack_revision is not None and pack_revision != card_model.get("hfPin"):
            failures.append(f"{label} packRevision does not match its quality card's hfPin")
        card_config = card.get("config") if isinstance(card.get("config"), dict) else {}
        if (
            hardware_class is not None
            and chip is not None
            and (
                hardware_class != card_config.get("hardwareClass")
                or chip.lower().replace(" ", "-") != hardware_class
            )
        ):
            failures.append(f"{label} hardwareClass does not match its quality card or chip")

    failures.extend(
        key_failures(entry.get("engineBuild"), SERVED_BENCHMARK_ENGINE_BUILD_KEYS, f"{label} engineBuild")
    )
    engine_commit: Optional[str] = None
    if isinstance(entry.get("engineBuild"), dict):
        engine_commit = require_str(entry["engineBuild"], "commit", f"{label} engineBuild", failures)
        if engine_commit is not None and not QUALITY_CARD_ENGINE_BUILD_COMMIT.fullmatch(engine_commit):
            failures.append(f"{label} engineBuild.commit is not a 40-hex sha")
        if card is not None and engine_commit is not None:
            card_provenance = card.get("provenance") if isinstance(card.get("provenance"), dict) else {}
            card_engine_build = card_provenance.get("engineBuild")
            if not isinstance(card_engine_build, dict) or card_engine_build.get("commit") != engine_commit:
                failures.append(f"{label} engineBuild.commit does not match its quality card")

    failures.extend(key_failures(entry.get("harness"), SERVED_BENCHMARK_HARNESS_KEYS, f"{label} harness"))
    if isinstance(entry.get("harness"), dict):
        harness = entry["harness"]
        harness_commit = require_str(harness, "publicCommit", f"{label} harness", failures)
        if harness_commit is not None and not COMMIT_SHA.fullmatch(harness_commit):
            failures.append(f"{label} harness.publicCommit is not a 40-hex sha")
        combine_commit = require_str(harness, "combineCommit", f"{label} harness", failures)
        if combine_commit is not None and not COMMIT_SHA.fullmatch(combine_commit):
            failures.append(f"{label} harness.combineCommit is not a 40-hex sha")
        bench_sha = require_str(harness, "benchSha256", f"{label} harness", failures)
        if bench_sha is not None and not SERVED_BENCHMARK_SHA256.fullmatch(bench_sha):
            failures.append(f"{label} harness.benchSha256 is not a 64-hex sha256")
        prompt_sha = require_str(harness, "promptSetSha256", f"{label} harness", failures)
        if prompt_sha is not None and not SERVED_BENCHMARK_SHA256.fullmatch(prompt_sha):
            failures.append(f"{label} harness.promptSetSha256 is not a 64-hex sha256")

    failures.extend(key_failures(entry.get("workload"), SERVED_BENCHMARK_WORKLOAD_KEYS, f"{label} workload"))
    completion_tokens_bounds: Optional[Tuple[int, int]] = None
    if isinstance(entry.get("workload"), dict):
        workload = entry["workload"]
        if workload.get("promptSet") != "default-3-prompt-set":
            failures.append(f"{label} workload.promptSet must be default-3-prompt-set")
        for int_key in ("prompts", "maxTokens", "runs", "warmup"):
            value = workload.get(int_key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                failures.append(f"{label} workload.{int_key} must be a non-negative int")
        temperature = _served_benchmark_number(workload.get("temperature"))
        if temperature is None or temperature < 0:
            failures.append(f"{label} workload.temperature must be a non-negative number")
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
            failures.append(f"{label} workload.completionTokens must be a [min, max] pair")
        else:
            completion_tokens_bounds = (completion_tokens[0], completion_tokens[1])

    failures.extend(key_failures(entry.get("serving"), SERVED_BENCHMARK_SERVING_KEYS, f"{label} serving"))
    if isinstance(entry.get("serving"), dict):
        serving = entry["serving"]
        context_tokens = serving.get("contextTokens")
        if not isinstance(context_tokens, int) or isinstance(context_tokens, bool) or context_tokens <= 0:
            failures.append(f"{label} serving.contextTokens must be a positive int")
        for key in ("mtp", "drafter", "promptLookup"):
            if serving.get(key) not in SERVED_BENCHMARK_SERVING_TRISTATE:
                failures.append(f"{label} serving.{key} is outside the allowed values")

    failures.extend(key_failures(entry.get("result"), SERVED_BENCHMARK_RESULT_KEYS, f"{label} result"))
    if isinstance(entry.get("result"), dict):
        result = entry["result"]
        candidate_rate = _served_benchmark_number(result.get("candidateDecodeTokS"))
        reference_rate = _served_benchmark_number(result.get("referenceDecodeTokS"))
        ratio = _served_benchmark_number(result.get("ratio"))
        if candidate_rate is None or reference_rate is None or ratio is None:
            failures.append(f"{label} result rates and ratio must be numeric")
        else:
            if (
                candidate_rate < SERVED_BENCHMARK_RATE_FLOOR
                or candidate_rate > SERVED_BENCHMARK_RATE_CEILING
                or reference_rate < SERVED_BENCHMARK_RATE_FLOOR
                or reference_rate > SERVED_BENCHMARK_RATE_CEILING
            ):
                failures.append(f"{label} result rates are outside the plausible [25, 2000] tok/s range")
            if reference_rate > 0 and abs(ratio - candidate_rate / reference_rate) > SERVED_BENCHMARK_RATIO_TOLERANCE:
                failures.append(f"{label} result.ratio is inconsistent with its own candidate/reference rates")
            if card is not None:
                card_benefit = (
                    card.get("legible", {}).get("benefit", {})
                    if isinstance(card.get("legible"), dict)
                    else {}
                )
                card_speed_x = _served_benchmark_number(card_benefit.get("speedX"))
                if (
                    card_speed_x is None
                    or card_speed_x == 0
                    or abs(ratio / card_speed_x - 1) > SERVED_BENCHMARK_SPEEDX_TOLERANCE
                ):
                    failures.append(
                        f"{label} result.ratio disagrees with its quality card's speedX by more than 3%"
                    )
        for key in ("candidatePasses", "referencePasses"):
            value = result.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                failures.append(f"{label} result.{key} must be a positive int")
        drift_pct = _served_benchmark_number(result.get("referenceDriftPct"))
        if drift_pct is None or drift_pct < 0:
            failures.append(f"{label} result.referenceDriftPct must be a non-negative number")
        elif drift_pct > SERVED_BENCHMARK_DRIFT_CEILING_PCT:
            failures.append(f"{label} result.referenceDriftPct exceeds the 5% ceiling")

    if entry.get("controls") != SERVED_BENCHMARK_CONTROLS:
        failures.append(f"{label} controls must be exactly the reviewed-slice-1 control set")
    row_sha = require_str(entry, "rowSha256", label, failures)
    if row_sha is not None and not SERVED_BENCHMARK_SHA256.fullmatch(row_sha):
        failures.append(f"{label} rowSha256 is not a 64-hex sha256")

    failures.extend(key_failures(entry.get("rerun"), SERVED_BENCHMARK_RERUN_KEYS, f"{label} rerun"))
    if isinstance(entry.get("rerun"), dict):
        rerun = entry["rerun"]
        for key in sorted(SERVED_BENCHMARK_RERUN_KEYS):
            failures.extend(_served_benchmark_rerun_failures(key, rerun.get(key), label))

    non_rerun = {key: value for key, value in entry.items() if key != "rerun"}
    if "--" in json.dumps(non_rerun, ensure_ascii=False, sort_keys=True):
        failures.append(f"{label} carries a raw '--' flag token outside rerun")
    marker_reason = _served_benchmark_marker_violation(
        json.dumps(entry, ensure_ascii=False, sort_keys=True)
    )
    if marker_reason is not None:
        failures.append(f"{label} contains {marker_reason}")

    # Checked after the marker scan above, same order as the fail-fast
    # loader (`build_public_site.validate_served_benchmark_entry`): a
    # packLabel that carries a marker is reported by that specific reason
    # first; only a clean-but-malformed packLabel trips this allowlist.
    if pack_label is not None and not SERVED_BENCHMARK_PACK_LABEL.fullmatch(pack_label):
        failures.append(f"{label} packLabel does not look like a pack label")

    if identifier is None:
        pass
    elif not SLUG.fullmatch(identifier):
        failures.append(f"{label} has an invalid id")

    return failures


def validate_served_benchmarks(site: Path) -> List[str]:
    """V1-V3 for the sealed served-engine benchmark ledger. `V1` pins the
    exact built `benchmarks/served-benchmarks.json` bytes; `V2` re-derives
    `L1`, `L3`-`L13` (accumulate style) against the built `quality/
    index.json` (`L2` becomes "served ledger requires quality/index.json",
    since this validator has no in-process `quality_manifest` object to
    check `None`-ness of); `V3` checks the rendered `benchmarks/index.html`
    section against the reviewed identity list.
    """

    failures: List[str] = []
    ledger_path = site / "benchmarks/served-benchmarks.json"
    if not ledger_path.is_file() or ledger_path.is_symlink():
        return ["benchmarks/served-benchmarks.json is missing, not a file, or a symlink"]
    raw = ledger_path.read_bytes()
    if (
        len(raw) != REVIEWED_SERVED_BENCHMARKS_BYTES
        or hashlib.sha256(raw).hexdigest() != REVIEWED_SERVED_BENCHMARKS_SHA256
    ):
        failures.append(
            "benchmarks/served-benchmarks.json does not match the reviewed served-benchmark ledger"
        )
    try:
        ledger = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return failures + [f"invalid benchmarks/served-benchmarks.json: {exc}"]

    failures.extend(key_failures(ledger, SERVED_BENCHMARK_TOP_KEYS, "benchmarks/served-benchmarks.json"))
    if not isinstance(ledger, dict):
        return failures
    if ledger.get("schemaVersion") != 1:
        failures.append("served ledger must use schemaVersion 1")
    if ledger.get("project") != "fast-mlx":
        failures.append("served ledger project must remain fast-mlx")
    if ledger.get("policy") != SERVED_BENCHMARK_POLICY:
        failures.append(f"served ledger policy must remain {SERVED_BENCHMARK_POLICY}")
    if ledger.get("claimBoundary") != SERVED_BENCHMARK_CLAIM_BOUNDARY:
        failures.append(f"served ledger claim boundary must remain {SERVED_BENCHMARK_CLAIM_BOUNDARY}")
    require_str(ledger, "updatedAt", "benchmarks/served-benchmarks.json", failures)

    quality_json_path = site / "quality/index.json"
    if not quality_json_path.is_file() or quality_json_path.is_symlink():
        failures.append("served ledger requires quality/index.json")
        return failures
    try:
        quality_manifest = json.loads(quality_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"served ledger requires quality/index.json: {exc}")
        return failures
    cards = quality_manifest.get("cards") if isinstance(quality_manifest, dict) else None
    cards_by_id: Dict[str, Dict[str, object]] = {
        str(card["id"]): card for card in cards if isinstance(card, dict) and "id" in card
    } if isinstance(cards, list) else {}

    raw_entries = ledger.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        failures.append("served ledger must contain at least one entry")
        return failures

    seen_ids: set = set()
    seen_hashes: set = set()
    previous_measured_at: Optional[str] = None
    entries: List[Dict[str, object]] = []
    for index, raw_entry in enumerate(raw_entries):
        label = f"served benchmark entry {index}"
        entry_failures = _served_benchmark_entry_failures(raw_entry, cards_by_id, label)
        failures.extend(entry_failures)
        if entry_failures or not isinstance(raw_entry, dict):
            continue
        entries.append(raw_entry)
        identifier = str(raw_entry.get("id"))
        row_sha = str(raw_entry.get("rowSha256"))
        if identifier in seen_ids:
            failures.append("served ledger has a duplicate entry id")
        if row_sha in seen_hashes:
            failures.append("served ledger has a duplicate entry rowSha256")
        seen_ids.add(identifier)
        seen_hashes.add(row_sha)
        measured_at = str(raw_entry.get("measuredAt"))
        if previous_measured_at is not None and measured_at >= previous_measured_at:
            failures.append("served ledger entries are not newest-first")
        previous_measured_at = measured_at

    # Cross-check each entry's built row file (item 5): its sha256 must
    # equal the entry's OWN `rowSha256` in the built ledger json (not just
    # the sealed pin below, which only covers the two REVIEWED identities --
    # this check runs for every entry, sealed or not).
    for entry in entries:
        identifier = str(entry.get("id"))
        row_path = site / "benchmarks" / SERVED_BENCHMARK_ROWS_DIRNAME / f"{identifier}.json"
        if row_path.is_symlink() or not row_path.is_file():
            failures.append(
                f"served benchmark entry {identifier!r} row file is missing, not a file, or a symlink"
            )
            continue
        row_bytes = row_path.read_bytes()
        row_sha = hashlib.sha256(row_bytes).hexdigest()
        if row_sha != str(entry.get("rowSha256")):
            failures.append(
                f"served benchmark entry {identifier!r} row file sha256 does not match its own rowSha256"
            )
        sealed = REVIEWED_SERVED_BENCHMARK_ROW_FILES.get(identifier)
        if sealed is not None and (len(row_bytes), row_sha) != sealed:
            failures.append(
                f"benchmarks/{SERVED_BENCHMARK_ROWS_DIRNAME}/{identifier}.json does not match "
                "the reviewed served-benchmark row"
            )

    # V3: the rendered `benchmarks/index.html` section against the reviewed
    # identity list -- only meaningful once the ledger itself is clean.
    if failures:
        return failures

    benchmark_path = site / "benchmarks/index.html"
    if not benchmark_path.is_file():
        return failures + ["benchmarks/index.html is missing"]
    collector = ServedBenchmarkCollector()
    try:
        collector.feed(benchmark_path.read_text(encoding="utf-8"))
        collector.close()
    except Exception as exc:
        return failures + [f"cannot parse benchmarks/index.html for served benchmarks: {exc}"]

    actual_identities = [(str(card.get("id")), str(card.get("cardId"))) for card in collector.cards]
    if actual_identities != list(REVIEWED_SERVED_BENCHMARK_IDENTITIES):
        failures.append("served-engine ledger identity mismatch in benchmarks/index.html")
        return failures

    # Whole-section scan for a raw address or path (V3): the WHOLE
    # `data-served-benchmarks` section's HTML -- text AND attribute values,
    # e.g. a `data-*` attribute or an `href` -- not just the per-article
    # TEXT the collector above exposes (`handle_data` never sees attribute
    # values, so a marker hiding in one would pass a text-only scan
    # silently). This scan runs once, over the raw section markup, rather
    # than per-article.
    page_text = benchmark_path.read_text(encoding="utf-8")
    section_match = re.search(
        r'<section[^>]*\bdata-served-benchmarks\b[^>]*>.*?</section>', page_text, re.S
    )
    if section_match is None:
        failures.append("benchmarks/index.html has no data-served-benchmarks section")
        return failures
    section_html = section_match.group(0)
    if SERVED_BENCHMARK_IPV4.search(section_html) or "<path>" in section_html:
        failures.append(
            "served-engine ledger data-served-benchmarks section leaks a raw address or path"
        )

    by_id = {str(card.get("id")): card for card in collector.cards}
    for entry in entries:
        identifier = str(entry["id"])
        rendered = by_id.get(identifier)
        if rendered is None:
            failures.append(f"served-engine ledger card {identifier!r} is not rendered")
            continue
        text = str(rendered.get("text", ""))
        result = entry["result"]
        ratio_text = f'{float(result["ratio"]):.3f}x'
        if ratio_text not in text:
            failures.append(f"served-engine ledger card {identifier!r} does not render its ratio")
        # Delimited, not a bare substring: `str(rate)` alone could match
        # part of an unrelated number elsewhere in the card's text (e.g. a
        # sha256 fragment or a token count). Both rates are rendered with
        # 2 decimals (see `build_public_site.render_benchmark_explorer`);
        # match that exact rendering, joined by the same delimiters.
        candidate_rate_text = f'{float(result["candidateDecodeTokS"]):.2f}'
        reference_rate_text = f'{float(result["referenceDecodeTokS"]):.2f}'
        rate_phrase = f'median {candidate_rate_text} vs {reference_rate_text} tok/s'
        if rate_phrase not in text:
            failures.append(f"served-engine ledger card {identifier!r} does not render its rates")
        engine_commit = str(entry["engineBuild"]["commit"])
        if f"engine build {engine_commit[:8]}" not in text:
            failures.append(f"served-engine ledger card {identifier!r} does not render its engine build")
        if str(entry["cardId"]) not in text:
            failures.append(f"served-engine ledger card {identifier!r} does not render its cardId")
        if str(entry["rowSha256"]) not in text:
            failures.append(f"served-engine ledger card {identifier!r} does not render its rowSha256")
        expected_row_href = f"{SERVED_BENCHMARK_ROWS_DIRNAME}/{identifier}.json"
        links = rendered.get("links", [])
        if not isinstance(links, list) or expected_row_href not in links:
            failures.append(
                f"served-engine ledger card {identifier!r} rowSha256 does not link to its row file"
            )
        for key in ("measure", "combine", "publicView"):
            if str(entry["rerun"][key]) not in text:
                failures.append(f"served-engine ledger card {identifier!r} does not render its rerun.{key}")

    return failures


def _flag_transfer_failures(raw_flag_transfer: object, label: str) -> List[str]:
    """Accumulate-style mirror of
    `build_public_site._validate_config_flag_transfer` -- see its docstring
    for the full rule set (docs/quality-card-schema-v1.md "Flag transfer").
    """
    failures = key_failures(
        raw_flag_transfer, QUALITY_CARD_FLAG_TRANSFER_KEYS, f"{label} config.flagTransfer"
    )
    if failures or not isinstance(raw_flag_transfer, dict):
        return failures
    failures.extend(
        key_failures(
            raw_flag_transfer.get("--mtp"),
            QUALITY_CARD_FLAG_TRANSFER_MTP_KEYS,
            f"{label} config.flagTransfer['--mtp']",
        )
    )
    mtp = raw_flag_transfer.get("--mtp")
    if not isinstance(mtp, dict):
        return failures
    greedy = mtp.get("greedy")
    if greedy not in QUALITY_CARD_FLAG_TRANSFER_MTP_GREEDY_VALUES:
        failures.append(f"{label} config.flagTransfer['--mtp'].greedy has unknown value {greedy!r}")
    divergent = mtp.get("divergentPrompts")
    if not isinstance(divergent, int) or isinstance(divergent, bool) or divergent < 0:
        failures.append(f"{label} config.flagTransfer['--mtp'].divergentPrompts must be an int >= 0")
    prompts = mtp.get("prompts")
    if not isinstance(prompts, int) or isinstance(prompts, bool) or prompts < 1:
        failures.append(f"{label} config.flagTransfer['--mtp'].prompts must be an int >= 1")
    if (
        isinstance(divergent, int)
        and not isinstance(divergent, bool)
        and isinstance(prompts, int)
        and not isinstance(prompts, bool)
        and divergent > prompts
    ):
        failures.append(f"{label} config.flagTransfer['--mtp'].divergentPrompts must be <= prompts")
    divergent_is_valid_int = isinstance(divergent, int) and not isinstance(divergent, bool)
    if greedy == "exact" and divergent_is_valid_int and divergent != 0:
        failures.append(
            f"{label} config.flagTransfer['--mtp'] greedy=exact requires divergentPrompts=0"
        )
    if greedy == "not_exact" and divergent_is_valid_int and divergent < 1:
        failures.append(
            f"{label} config.flagTransfer['--mtp'] greedy=not_exact requires divergentPrompts>=1"
        )
    max_tokens = mtp.get("maxTokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        failures.append(f"{label} config.flagTransfer['--mtp'].maxTokens must be an int >= 1")
    evidence = mtp.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        failures.append(f"{label} config.flagTransfer['--mtp'].evidence must be a non-empty string")
    else:
        evidence_path = Path(evidence)
        if evidence_path.is_absolute() or ".." in evidence_path.parts:
            failures.append(
                f"{label} config.flagTransfer['--mtp'].evidence must be a repo-relative "
                "path with no absolute path and no '..'"
            )
    return failures


def validate_quality_guide_manifest(value: object) -> List[str]:
    """Fail-closed schema check for the `fast-mlx-quality-card-v1` manifest.

    Mirrors `build_public_site.load_quality_guides` but accumulates failures
    instead of refusing the process, so it can validate an already-generated
    `quality/index.json` artifact.
    """

    failures: List[str] = []
    failures.extend(key_failures(value, {"schema", "generatedAt", "cards"}, "quality-guide manifest"))
    if failures or not isinstance(value, dict):
        return failures
    if value.get("schema") != QUALITY_GUIDE_SCHEMA:
        failures.append(f"quality-guide manifest must use schema {QUALITY_GUIDE_SCHEMA!r}")
    require_str(value, "generatedAt", "quality-guide manifest", failures)

    # Whole-document private-marker scan, mirroring
    # `build_public_site.validate_quality_card_document`'s identical check
    # -- this validator previously only scanned RENDERED Pages output
    # (`validate_site`, below) and never the raw manifest object itself, so
    # a marker inside e.g. `config.flagTransfer["--mtp"].evidence` was
    # invisible to this entry point.
    serialized = json.dumps(value, ensure_ascii=False)
    for marker in PRIVATE_MARKERS:
        if marker.casefold() in serialized.casefold():
            failures.append(f"quality-guide manifest contains private marker {marker!r}")

    cards = value.get("cards")
    if not isinstance(cards, list) or not cards:
        failures.append("quality-guide manifest must contain at least one card")
        return failures

    seen_ids: set[str] = set()
    # See build_public_site.validate_quality_card_document's identical
    # tracking set for the rule this enforces.
    seen_identity_engine_builds: set[tuple] = set()
    for index, raw_card in enumerate(cards):
        label = f"quality card entry {index}"
        failures.extend(
            key_failures(
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
                label,
            )
        )
        if not isinstance(raw_card, dict):
            continue
        card = raw_card
        identifier = require_str(card, "id", label, failures)
        if identifier is not None:
            if not QUALITY_CARD_ID.fullmatch(identifier) or identifier in seen_ids:
                failures.append(f"{label} has an invalid or duplicate id")
            seen_ids.add(identifier)

        model = card.get("model")
        failures.extend(key_failures(model, {"family", "repo", "hfPin"}, f"{label} model"))
        if isinstance(model, dict):
            require_str(model, "family", f"{label} model", failures)
            for key in ("repo", "hfPin"):
                field_value = model.get(key)
                if field_value is not None and (
                    not isinstance(field_value, str) or not field_value.strip()
                ):
                    failures.append(f"{label} model.{key} must be a non-empty string or null")

        config = card.get("config")
        if not isinstance(config, dict):
            failures.append(f"{label} config is not an object")
        else:
            config_keys = set(config)
            if not (
                QUALITY_CARD_CONFIG_REQUIRED_KEYS <= config_keys <= QUALITY_CARD_CONFIG_ALLOWED_KEYS
            ):
                missing = sorted(QUALITY_CARD_CONFIG_REQUIRED_KEYS - config_keys)
                extra = sorted(config_keys - QUALITY_CARD_CONFIG_ALLOWED_KEYS)
                failures.append(
                    f"{label} config keys differ from schema; missing={missing} extra={extra}"
                )
            # `residency` is OPTIONAL; absence means "resident". When present it must be
            # one of the two recognized residencies.
            if "residency" in config and config.get("residency") not in QUALITY_CARD_RESIDENCIES:
                failures.append(
                    f"{label} config.residency has unknown value {config.get('residency')!r}"
                )
            raw_quant = config.get("quant")
            if raw_quant is not None:
                failures.extend(
                    key_failures(
                        raw_quant,
                        {"bits", "groupSize", "mixedBit", "note"},
                        f"{label} config.quant",
                    )
                )
                if isinstance(raw_quant, dict):
                    if not isinstance(raw_quant.get("bits"), int) or isinstance(
                        raw_quant.get("bits"), bool
                    ):
                        failures.append(f"{label} config.quant.bits is not an int")
                    group_size = raw_quant.get("groupSize")
                    if group_size is not None and (
                        not isinstance(group_size, int) or isinstance(group_size, bool)
                    ):
                        failures.append(f"{label} config.quant.groupSize must be an int or null")
                    if not isinstance(raw_quant.get("mixedBit"), bool):
                        failures.append(f"{label} config.quant.mixedBit is not a bool")
                    note = raw_quant.get("note")
                    if note is not None and (not isinstance(note, str) or not note.strip()):
                        failures.append(
                            f"{label} config.quant.note must be a non-empty string or null"
                        )
            # `flagTransfer` is OPTIONAL; absence means every flag's transfer
            # is unmeasured for this card (see "Flag transfer" above).
            if "flagTransfer" in config:
                failures.extend(
                    _flag_transfer_failures(config.get("flagTransfer"), label)
                )
            require_str(config, "enhancement", f"{label} config", failures)
            hardware_class = require_str(config, "hardwareClass", f"{label} config", failures)
            if (
                hardware_class is not None
                and hardware_class.strip().lower() == QUALITY_CARD_HARDWARE_CLASS_SENTINEL
            ):
                failures.append(
                    f"{label} config.hardwareClass {hardware_class!r} is the "
                    "chip-probe-failed sentinel value ('unknown'); chip identification "
                    "failed when this card was measured, so the card must not claim a "
                    "hardware class it could not measure"
                )
            elif hardware_class is not None and not QUALITY_CARD_HARDWARE_CLASS.fullmatch(
                hardware_class
            ):
                failures.append(
                    f"{label} config.hardwareClass {hardware_class!r} is not a canonical "
                    "hardware class (expected lowercase alphanumeric segments joined by single "
                    "hyphens, e.g. 'apple-m3-ultra')"
                )

        verdict = require_str(card, "verdict", label, failures)
        if verdict is not None and verdict not in QUALITY_VERDICTS:
            failures.append(f"{label} has unknown verdict {verdict!r}")

        admission = card.get("admission")
        failures.extend(
            key_failures(admission, {"default", "optIn", "reason"}, f"{label} admission")
        )
        if isinstance(admission, dict):
            require_str(admission, "reason", f"{label} admission", failures)
            default = admission.get("default")
            if not isinstance(default, bool):
                failures.append(f"{label} admission.default is not a bool")
            opt_in = admission.get("optIn")
            if not isinstance(opt_in, bool):
                failures.append(f"{label} admission.optIn is not a bool")
            # Claim-integrity gate (NOT an admission gate: the Swift + Python
            # serve gates key on `verdict` alone -- see "Admission
            # discriminator rules" in docs/quality-card-schema-v1.md).
            # admission.{default,optIn} are validated-for-agreement CLAIMS
            # about verdict that the public renderer surfaces (it reads
            # admission.reason); a card publishing a NO_GO verdict alongside
            # admission.default=True would render "safe silent default" for a
            # pack the serve gate refuses.
            # R1: a NO_GO card can never be the silent production default.
            # R2: a NO_GO card must stay electable, because the refusal
            #     message tells the operator to elect it with
            #     --accept-quality.
            #
            # Deliberately ONE-SIDED. `verdict != "NO_GO"` must NOT imply
            # default=True: "may this be the SILENT production default?" is a
            # rollout/trust question that a passing MEASUREMENT does not
            # settle, and scripts/tests/fixtures/quality-guides.sample.json
            # ships two passing cards that say so -- a PASS card whose quality
            # is vendor-reported rather than independently gated, and an EXACT
            # card whose reason is "opt-in pending broader rollout".
            #
            # UNMEASURED is exempt: no UNMEASURED card has ever been emitted
            # or shipped, so a rule for it would be an unreachable branch.
            #
            # admission.optIn is a FROZEN CONSTANT true on every verdict, not
            # conditioned on NO_GO like R1/R2 above. It is derivable from
            # nothing and carries no per-card information, but the field is
            # retained in the wire format because released fastmlx-serve
            # binaries decode it non-optionally: a missing field fails OPEN
            # (the whole manifest fails to decode and serve announces
            # `quality_cards=none`). Resolved: this used to be left
            # unconstrained on non-NO_GO verdicts pending a decision between
            # two contradictory conventions in the repo; that question is
            # now closed to redefinition. Unlike `default`'s one-sidedness
            # (which is PERMANENT -- two honest passing cards in
            # scripts/tests/fixtures/quality-guides.sample.json carry
            # `default: false`), this exemption was TEMPORARY.
            if isinstance(opt_in, bool) and opt_in is not True:
                failures.append(
                    f"{label} admission.optIn must be true "
                    f"(every card is electable via --accept-quality)"
                )
            if verdict == "NO_GO":
                if isinstance(default, bool) and default is not False:
                    failures.append(
                        f"{label} admission.default {default!r} disagrees with "
                        f"verdict 'NO_GO' (a NO_GO pack is never a silent default)"
                    )

        legible = card.get("legible")
        failures.extend(
            key_failures(
                legible,
                {"tier", "headline", "nextWordDrift", "regressionFocus", "example", "benefit"},
                f"{label} legible",
            )
        )
        benefit: Dict[str, object] = {}
        if isinstance(legible, dict):
            tier = require_str(legible, "tier", f"{label} legible", failures)
            if tier is not None and tier not in QUALITY_TIERS:
                failures.append(f"{label} legible has unknown tier {tier!r}")
            require_str(legible, "headline", f"{label} legible", failures)
            # regressionFocus is required only for a card that admits a quality trade
            # (NO_GO/PASS); REFERENCE and EXACT have no regression to report.
            if verdict in {"NO_GO", "PASS"}:
                require_str(legible, "regressionFocus", f"{label} legible", failures)
            else:
                require_nullable_str(legible, "regressionFocus", f"{label} legible", failures)

            # nextWordDrift itself is null for REFERENCE (no drift vs itself).
            raw_drift = legible.get("nextWordDrift")
            if raw_drift is not None:
                failures.extend(
                    key_failures(
                        raw_drift,
                        {"oneInK", "top1AgreementPct"},
                        f"{label} legible.nextWordDrift",
                    )
                )
                if isinstance(raw_drift, dict):
                    # oneInK is null for EXACT (identical output has no "1 in K" to state).
                    one_in_k = raw_drift.get("oneInK")
                    if one_in_k is not None and (
                        not isinstance(one_in_k, int) or isinstance(one_in_k, bool)
                    ):
                        failures.append(
                            f"{label} legible.nextWordDrift.oneInK must be an int or null"
                        )
                    top1 = raw_drift.get("top1AgreementPct")
                    if not isinstance(top1, (int, float)) or isinstance(top1, bool):
                        failures.append(
                            f"{label} legible.nextWordDrift.top1AgreementPct is not a number"
                        )

            # "Unquantified" means the output differs from the reference but the size of
            # that difference was never measured: it may ONLY pair with a NO_GO card whose
            # nextWordDrift is null, and conversely a NO_GO/PASS card with a null
            # nextWordDrift must be tiered Unquantified.
            if tier == "Unquantified":
                if verdict != "NO_GO":
                    failures.append(
                        f"{label} legible has tier Unquantified but verdict is not NO_GO"
                    )
                if raw_drift is not None:
                    failures.append(
                        f"{label} legible has tier Unquantified but nextWordDrift is not null"
                    )
            elif raw_drift is None and verdict in {"NO_GO", "PASS"}:
                failures.append(
                    f"{label} legible has a null nextWordDrift on verdict {verdict!r} "
                    "but tier is not Unquantified"
                )

            example = legible.get("example")
            failures.extend(
                key_failures(
                    example,
                    {"status", "prompt", "referenceOutput", "configOutput", "note"},
                    f"{label} legible.example",
                )
            )
            if isinstance(example, dict) and example.get("status") not in QUALITY_EXAMPLE_STATUSES:
                failures.append(
                    f'{label} legible.example has unknown status {example.get("status")!r}'
                )
            raw_benefit = legible.get("benefit")
            failures.extend(
                key_failures(raw_benefit, {"fit", "speedX", "speedXStatus"}, f"{label} legible.benefit")
            )
            if isinstance(raw_benefit, dict):
                benefit = raw_benefit
                # fit is null for an enhancement card with no footprint of its own (e.g. MTP).
                require_nullable_str(benefit, "fit", f"{label} legible.benefit", failures)
                require_str(benefit, "speedXStatus", f"{label} legible.benefit", failures)

        # rawMetrics is always an object; it may be empty only for EXACT (identical
        # output has no per-config metric of its own to show).
        raw_metrics = card.get("rawMetrics")
        if not isinstance(raw_metrics, dict):
            failures.append(f"{label} rawMetrics must be an object")
        elif not raw_metrics and verdict != "EXACT":
            failures.append(f"{label} rawMetrics must be a non-empty object")

        provenance = card.get("provenance")
        engine_build_commit: Optional[str] = None
        if not isinstance(provenance, dict):
            failures.append(f"{label} provenance is not an object")
        else:
            provenance_keys = set(provenance)
            if not (
                QUALITY_CARD_PROVENANCE_REQUIRED_KEYS
                <= provenance_keys
                <= QUALITY_CARD_PROVENANCE_ALLOWED_KEYS
            ):
                missing = sorted(QUALITY_CARD_PROVENANCE_REQUIRED_KEYS - provenance_keys)
                extra = sorted(provenance_keys - QUALITY_CARD_PROVENANCE_ALLOWED_KEYS)
                failures.append(
                    f"{label} provenance keys differ from schema; missing={missing} extra={extra}"
                )
            raw_engine_build = provenance.get("engineBuild")
            if raw_engine_build is not None:
                failures.extend(
                    key_failures(
                        raw_engine_build, {"commit"}, f"{label} provenance.engineBuild"
                    )
                )
                if isinstance(raw_engine_build, dict):
                    commit = raw_engine_build.get("commit")
                    if not isinstance(
                        commit, str
                    ) or not QUALITY_CARD_ENGINE_BUILD_COMMIT.fullmatch(commit):
                        failures.append(
                            f"{label} provenance.engineBuild.commit must be a lowercase "
                            "40-hex string"
                        )
                    else:
                        engine_build_commit = commit
        # `config.flagTransfer` measures how a served-engine flag (e.g.
        # `--mtp`) behaves on THIS card's exact engine build; without a
        # recorded `provenance.engineBuild.commit` the measurement has no
        # build to be true of (see docs/quality-card-schema-v1.md "Flag
        # transfer").
        if (
            isinstance(config, dict)
            and "flagTransfer" in config
            and engine_build_commit is None
        ):
            failures.append(
                f"{label} config.flagTransfer requires provenance.engineBuild.commit"
            )
        if isinstance(provenance, dict):
            source = require_str(provenance, "source", f"{label} provenance", failures)
            if source is not None and source not in QUALITY_PROVENANCE_SOURCES:
                failures.append(f"{label} provenance has unknown source {source!r}")
            vendor = provenance.get("vendor")
            if source == "vendor-reported":
                if not isinstance(vendor, str) or not vendor.strip():
                    failures.append(
                        f"{label} provenance.vendor is required when source is vendor-reported"
                    )
            elif vendor is not None:
                failures.append(
                    f"{label} provenance.vendor must be null unless source is vendor-reported"
                )
            for key in ("method", "hardware"):
                require_str(provenance, key, f"{label} provenance", failures)
            # harnessGitSHA/corpusId/sourceVerdict/measuredAt are null for a curated card
            # not sourced from a dated gate verdict (e.g. the MTP exactness card).
            for key in ("harnessGitSHA", "corpusId", "sourceVerdict"):
                require_nullable_str(provenance, key, f"{label} provenance", failures)
            measured_at = require_nullable_str(provenance, "measuredAt", f"{label} provenance", failures)
            if measured_at is not None:
                try:
                    dt.datetime.fromisoformat(measured_at.replace("Z", "+00:00"))
                except ValueError:
                    failures.append(f"{label} provenance measuredAt is not an ISO-8601 timestamp")

        # Measured-PASS rule, mirroring
        # `build_public_site.validate_quality_card_document`'s identical
        # check (and scripts/emit_quality_card_teacher.py
        # `pass_card_violations`). SCOPED to provenance.source ==
        # "fast-mlx-measured": the vendor-reported PASS card in
        # scripts/tests/fixtures/quality-guides.sample.json (Noticeable,
        # admission.default false, no top-1/ppl) must keep validating.
        # Every input is re-read defensively here because the locals above
        # are only bound when their containers are well-formed dicts.
        if (
            verdict == "PASS"
            and isinstance(provenance, dict)
            and provenance.get("source") == "fast-mlx-measured"
        ):
            measured_label = f"{label} (card id {identifier!r}) fast-mlx-measured PASS"
            measured_legible = legible if isinstance(legible, dict) else {}
            measured_metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
            measured_admission = admission if isinstance(admission, dict) else {}

            def measured_number(key: str) -> Optional[float]:
                number = measured_metrics.get(key)
                if isinstance(number, bool) or not isinstance(number, (int, float)):
                    return None
                return float(number) if math.isfinite(number) else None

            measured_tier = measured_legible.get("tier")
            if measured_tier != "Near-lossless":
                failures.append(
                    f"{measured_label} requires legible.tier 'Near-lossless', "
                    f"got {measured_tier!r}"
                )
            if measured_legible.get("nextWordDrift") is None:
                failures.append(f"{measured_label} requires a non-null legible.nextWordDrift")
            measured_top1 = measured_number("top1AgreementPct")
            if measured_top1 is None or measured_top1 < 99.0:
                failures.append(
                    f"{measured_label} requires numeric rawMetrics.top1AgreementPct >= 99, "
                    f"got {measured_metrics.get('top1AgreementPct')!r}"
                )
            for ppl_key in ("pplDeltaPct", "pplDeltaUpperPct"):
                ppl_value = measured_number(ppl_key)
                if ppl_value is None or ppl_value > 1.0:
                    failures.append(
                        f"{measured_label} requires numeric rawMetrics.{ppl_key} <= 1, "
                        f"got {measured_metrics.get(ppl_key)!r}"
                    )
            if measured_admission.get("default") is not True:
                failures.append(
                    f"{measured_label} requires admission.default true, "
                    f"got {measured_admission.get('default')!r}"
                )

        boundary = card.get("boundary")
        # boundary.measuredNewTokens mirrors
        # `build_public_site.validate_quality_card_document`: optional, but
        # required on a fast-mlx-measured PASS card; an integer >= 1 and never
        # a bool (`type(...) is int` excludes bool, an int subclass).
        boundary_keys = {"scope", "unmeasured"}
        if isinstance(boundary, dict) and "measuredNewTokens" in boundary:
            boundary_keys = boundary_keys | {"measuredNewTokens"}
        failures.extend(key_failures(boundary, boundary_keys, f"{label} boundary"))
        if (
            verdict == "PASS"
            and isinstance(provenance, dict)
            and provenance.get("source") == "fast-mlx-measured"
            and not (isinstance(boundary, dict) and "measuredNewTokens" in boundary)
        ):
            failures.append(
                f"{label} (card id {identifier!r}) fast-mlx-measured PASS "
                "requires boundary.measuredNewTokens"
            )
        if isinstance(boundary, dict):
            require_str(boundary, "scope", f"{label} boundary", failures)
            if "measuredNewTokens" in boundary:
                measured_new_tokens = boundary["measuredNewTokens"]
                if type(measured_new_tokens) is not int or measured_new_tokens < 1:
                    failures.append(
                        f"{label} (card id {identifier!r}) boundary.measuredNewTokens "
                        f"must be an integer >= 1, "
                        f"got {measured_new_tokens!r}"
                    )
            unmeasured = boundary.get("unmeasured")
            if not isinstance(unmeasured, list) or any(
                not isinstance(item, str) or not item.strip() for item in unmeasured
            ):
                failures.append(f"{label} boundary.unmeasured must be a list of non-empty strings")
            else:
                # A numeric speedX is a measured decode-throughput claim;
                # boundary.unmeasured must not also carry an entry
                # disclaiming that same claim as unmeasured (matched by
                # PREFIX -- see the QUALITY_CARD_SPEEDX_UNMEASURED_PREFIX
                # comment above). The converse (a null speedX requiring the
                # entry) is deliberately NOT enforced; see that comment.
                # Uses `label` (the same "quality card entry {index}" form
                # every other failure in this loop uses), not `identifier`
                # -- `identifier` can be None when `id` itself is missing or
                # invalid, which would otherwise print the nonsense "quality
                # card None ...".
                speed_x = benefit.get("speedX")
                if isinstance(speed_x, (int, float)) and not isinstance(speed_x, bool):
                    prefix = QUALITY_CARD_SPEEDX_UNMEASURED_PREFIX
                    offending = [item for item in unmeasured if item.strip().startswith(prefix)]
                    if offending:
                        failures.append(
                            f"{label} legible.benefit.speedX is numeric but "
                            f"boundary.unmeasured still lists {offending[0]!r}"
                        )

        repo_value = model.get("repo") if isinstance(model, dict) else None
        identity = repo_value if repo_value is not None else (
            model.get("hfPin") if isinstance(model, dict) else None
        )
        residency_value = config.get("residency") if isinstance(config, dict) else None
        hardware_class_value = config.get("hardwareClass") if isinstance(config, dict) else None
        identity_key = (
            identity,
            residency_value or "resident",
            hardware_class_value,
            engine_build_commit,
        )
        if identity_key in seen_identity_engine_builds:
            failures.append(
                f"{label} duplicates another card's (identity, residency, hardwareClass, "
                f"engineBuild) combination {identity_key!r}"
            )
        else:
            seen_identity_engine_builds.add(identity_key)
    return failures


def validate_quality_guide_page(site: Path) -> List[str]:
    """Validate the optional public quality-guide page and its manifest.

    Absence of both `quality/index.html` and `quality/index.json` is not a
    failure: the page is skipped gracefully when no quality-card manifest was
    available at build time, exactly like the build script.
    """

    failures: List[str] = []
    html_path = site / "quality/index.html"
    json_path = site / "quality/index.json"
    html_present = html_path.is_file() and not html_path.is_symlink()
    json_present = json_path.is_file() and not json_path.is_symlink()
    if not html_present and not json_present:
        return failures
    if html_present != json_present:
        failures.append(
            "quality/index.html and quality/index.json must both be present or both absent"
        )
        return failures

    try:
        manifest = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid quality/index.json: {exc}"]

    manifest_failures = validate_quality_guide_manifest(manifest)
    failures.extend(manifest_failures)
    if manifest_failures or not isinstance(manifest, dict):
        return failures

    cards = manifest.get("cards", [])
    collector = QualityCardCollector()
    html_text = html_path.read_text(encoding="utf-8")
    try:
        collector.feed(html_text)
        collector.close()
    except Exception as exc:
        return failures + [f"cannot parse quality/index.html: {exc}"]

    head_collector = HeadMetadataCollector()
    try:
        head_collector.feed(html_text)
    except Exception as exc:
        failures.append(f"cannot parse quality/index.html metadata: {exc}")
    else:
        expected_canonical = PUBLIC_SITE_URL + "quality/"
        if head_collector.canonicals != [expected_canonical]:
            failures.append("quality/index.html has the wrong canonical metadata")

    rendered_by_id = {
        str(card["id"]): card for card in collector.cards if isinstance(card.get("id"), str)
    }
    expected_ids = {str(card["id"]) for card in cards if isinstance(card, dict)}
    if set(rendered_by_id) != expected_ids:
        failures.append("quality/index.html cards do not match quality/index.json cards")

    for card in cards:
        if not isinstance(card, dict):
            continue
        card_id = str(card["id"])
        rendered = rendered_by_id.get(card_id)
        if rendered is None:
            continue
        if not rendered.get("has_details"):
            failures.append(
                f"quality card {card_id!r} is missing its rawMetrics <details> expander"
            )
        text = str(rendered.get("text", ""))
        legible = card["legible"]
        if str(legible["headline"]) not in text:
            failures.append(f"quality card {card_id!r} does not render its headline")
        if str(legible["tier"]).upper() not in text.upper():
            failures.append(f"quality card {card_id!r} does not render its tier")
        if str(card["verdict"]) == "NO_GO":
            reason = str(card["admission"]["reason"])
            if reason not in text:
                failures.append(
                    f"quality card {card_id!r} (NO_GO) does not render its admission reason"
                )
            if "broken" in text.lower():
                failures.append(
                    f'quality card {card_id!r} (NO_GO) must not read as "broken"'
                )
        family = str(card["model"]["family"])
        if family not in text:
            failures.append(f"quality card {card_id!r} does not render its model family")
        method = str(card["provenance"]["method"])
        if method not in text:
            failures.append(f"quality card {card_id!r} does not render its provenance method")
        source = str(card["provenance"]["source"])
        if source not in text:
            failures.append(f"quality card {card_id!r} does not render its provenance source")
        if source == "vendor-reported":
            vendor = str(card["provenance"].get("vendor"))
            if vendor not in text:
                failures.append(f"quality card {card_id!r} does not render its vendor")
        benefit = legible["benefit"]
        if str(benefit["speedXStatus"]) not in text:
            failures.append(
                f"quality card {card_id!r} does not render its speedXStatus text"
            )
        if str(card["boundary"]["scope"]) not in text:
            failures.append(f"quality card {card_id!r} does not render its boundary scope")
        for entry in card["boundary"]["unmeasured"]:
            if str(entry) not in text:
                failures.append(
                    f"quality card {card_id!r} does not render its "
                    f"boundary.unmeasured entry {entry!r}"
                )
        if re.search(r"\bNone\b", text):
            failures.append(
                f"quality card {card_id!r} leaks a null field as literal text \"None\""
            )
    return failures


def validate_benchmark_detail_pages(site: Path) -> List[str]:
    failures: List[str] = []
    benchmark_root = site / "benchmarks"
    expected_entries = {"index.html", "served-benchmarks.json", SERVED_BENCHMARK_ROWS_DIRNAME} | {
        str(highlight["id"]) for highlight in REVIEWED_BENCHMARK_HIGHLIGHTS
    }
    if benchmark_root.is_symlink() or not benchmark_root.is_dir():
        return ["benchmarks must be a regular non-symlink directory"]
    try:
        actual_entries = {entry.name for entry in benchmark_root.iterdir()}
    except OSError as exc:
        return [f"cannot inspect benchmark detail routes: {exc}"]
    for extra in sorted(actual_entries - expected_entries):
        failures.append(f"unexpected benchmark route outside reviewed set: {extra}")

    # Item 4/5: `benchmarks/served-benchmark-rows/` may hold exactly one file
    # per reviewed served-benchmark identity, never a subdirectory, symlink,
    # or orphan file no ledger entry names. Gated on `expected_row_files`
    # being non-empty: a REAL build always has at least one served-benchmark
    # entry (the loader itself refuses an empty ledger), so the directory
    # always exists there; only a test that substitutes a synthetic EMPTY
    # ledger (e.g. mocking `load_served_benchmarks`/`validate_served_
    # benchmarks` to isolate an unrelated page) can legitimately produce a
    # build with no row directory at all, and that is not this check's
    # concern.
    expected_row_files = {
        f"{identifier}.json" for identifier, _card_id in REVIEWED_SERVED_BENCHMARK_IDENTITIES
    }
    if expected_row_files:
        rows_root = benchmark_root / SERVED_BENCHMARK_ROWS_DIRNAME
        if rows_root.is_symlink() or not rows_root.is_dir():
            failures.append(f"benchmarks/{SERVED_BENCHMARK_ROWS_DIRNAME} must be a regular non-symlink directory")
        else:
            try:
                actual_row_entries = {entry.name: entry for entry in rows_root.iterdir()}
            except OSError as exc:
                failures.append(f"cannot inspect benchmarks/{SERVED_BENCHMARK_ROWS_DIRNAME}: {exc}")
            else:
                for name, entry in sorted(actual_row_entries.items()):
                    if entry.is_symlink():
                        failures.append(f"benchmarks/{SERVED_BENCHMARK_ROWS_DIRNAME}/{name} is a symlink")
                    elif entry.is_dir():
                        failures.append(f"benchmarks/{SERVED_BENCHMARK_ROWS_DIRNAME}/{name} is a subdirectory")
                    if name not in expected_row_files:
                        failures.append(
                            f"unexpected file under benchmarks/{SERVED_BENCHMARK_ROWS_DIRNAME}: {name}"
                        )

    for highlight in REVIEWED_BENCHMARK_HIGHLIGHTS:
        identifier = str(highlight["id"])
        relative = f"benchmarks/{identifier}/index.html"
        directory = site / "benchmarks" / identifier
        path = directory / "index.html"
        if directory.is_symlink() or not directory.is_dir():
            failures.append(
                f"benchmark detail route {identifier!r} must be a regular non-symlink directory"
            )
            continue
        if path.is_symlink() or not path.is_file():
            failures.append(f"{relative} must be a regular non-symlink file")
            continue

        collector = BenchmarkDetailCollector()
        try:
            collector.feed(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"cannot parse {relative}: {exc}")
            continue
        if len(collector.sections) != 1:
            failures.append(
                f"benchmark detail {identifier!r} must contain exactly one detail section"
            )
            continue

        detail = collector.sections[0]
        evidence = highlight["evidence"]
        if not isinstance(evidence, dict):
            failures.append(f"reviewed benchmark {identifier!r} has invalid evidence")
            continue
        decision = str(highlight["decision"])
        expected_text = " ".join(
            [
                "Reviewed benchmark evidence",
                HIGHLIGHT_DECISION_LABELS[decision].upper(),
                str(highlight["date"]),
                str(highlight["metric"]),
                str(highlight["label"]),
                "One reviewed fast-mlx result, shown with the context and boundary that made it admissible.",
                "Model",
                str(highlight["model"]),
                "Hardware",
                str(highlight["hardware"]),
                "Workload",
                str(highlight["workload"]),
                "Boundary:",
                str(highlight["caveat"]),
                "Read",
                str(evidence["title"]),
                "Back to all reviewed results",
            ]
        )
        expected_text = " ".join(expected_text.split())
        evidence_href = f'../../{str(evidence["path"]).rstrip("/")}/'
        expected_fields = [
            ("Model", str(highlight["model"])),
            ("Hardware", str(highlight["hardware"])),
            ("Workload", str(highlight["workload"])),
        ]
        actual_fields = list(zip(detail.get("terms", []), detail.get("values", [])))

        if detail.get("id") != identifier:
            failures.append(f"benchmark detail {identifier!r} has the wrong id")
        if detail.get("datetime") != highlight["date"]:
            failures.append(f"benchmark detail {identifier!r} has the wrong date")
        if detail.get("text") != expected_text:
            failures.append(f"benchmark detail {identifier!r} has drifted reviewed text")
        if actual_fields != expected_fields or detail.get("dlCount") != 1:
            failures.append(f"benchmark detail {identifier!r} has the wrong context fields")
        if detail.get("links") != [evidence_href, "../"]:
            failures.append(f"benchmark detail {identifier!r} has the wrong action links")
        if (
            detail.get("h1Count") != 1
            or collector.page_h1_count != 1
            or detail.get("hasNestedDetail")
        ):
            failures.append(f"benchmark detail {identifier!r} has the wrong heading structure")
        if detail.get("hasDuplicateAttributes"):
            failures.append(f"benchmark detail {identifier!r} has duplicate attributes")
        if detail.get("hasVisibilitySuppressor"):
            failures.append(f"benchmark detail {identifier!r} is hidden")
        if collector.scripts:
            failures.append(f"benchmark detail {identifier!r} must not load scripts")

        page_text = " ".join("".join(collector.text_parts).split())
        expected_boundary = (
            "Claim boundary A permalink is not a broader performance claim. "
            "This page does not normalize, rank, aggregate, or recompute results. "
            "It performs no unit conversion, interpolation, competitor comparison, "
            "live benchmark execution, or authority transition. "
            "Read the measurement methodology →"
        )
        if expected_boundary not in page_text:
            failures.append(
                f"benchmark detail {identifier!r} has the wrong claim boundary"
            )
        if collector.page_links.count("../../methodology/") != 2:
            failures.append(
                f"benchmark detail {identifier!r} has the wrong methodology link"
            )
    return failures


def validate_release_detail_pages(
    site: Path, release_index: Dict[str, object]
) -> List[str]:
    failures: List[str] = []
    release_root = site / "releases"
    expected_entries = {"index.html", "index.json", "feed.atom"} | {
        identifier for identifier, _title in REVIEWED_RELEASE_IDENTITIES
    }
    if release_root.is_symlink() or not release_root.is_dir():
        return ["releases must be a regular non-symlink directory"]
    try:
        actual_entries = {entry.name for entry in release_root.iterdir()}
    except OSError as exc:
        return [f"cannot inspect release detail routes: {exc}"]
    for extra in sorted(actual_entries - expected_entries):
        failures.append(f"unexpected release route outside reviewed set: {extra}")

    releases = release_index.get("releases")
    expected_releases = releases if isinstance(releases, list) else []
    expected_by_id = {
        release["id"]: release
        for release in expected_releases
        if isinstance(release, dict) and isinstance(release.get("id"), str)
    }
    expected_order = tuple(
        release.get("id") for release in expected_releases if isinstance(release, dict)
    )
    if expected_order != tuple(identifier for identifier, _title in REVIEWED_RELEASE_IDENTITIES):
        failures.append("release detail pages do not match reviewed release identities")

    for identifier, _title in REVIEWED_RELEASE_IDENTITIES:
        relative = f"releases/{identifier}/index.html"
        directory = site / "releases" / identifier
        path = directory / "index.html"
        if directory.is_symlink() or not directory.is_dir():
            failures.append(
                f"release detail route {identifier!r} must be a regular non-symlink directory"
            )
            continue
        if path.is_symlink() or not path.is_file():
            failures.append(f"{relative} must be a regular non-symlink file")
            continue
        release = expected_by_id.get(identifier)
        if release is None:
            continue

        try:
            raw_detail = path.read_bytes()
        except OSError as exc:
            failures.append(f"cannot read {relative}: {exc}")
            continue
        expected_size, expected_sha256 = REVIEWED_RELEASE_DETAIL_SEALS[identifier]
        if (
            len(raw_detail) != expected_size
            or hashlib.sha256(raw_detail).hexdigest() != expected_sha256
        ):
            failures.append(
                f"release detail {identifier!r} does not match the reviewed page seal"
            )
        try:
            detail_text = raw_detail.decode("utf-8")
        except UnicodeDecodeError as exc:
            failures.append(f"cannot parse {relative}: {exc}")
            continue
        collector = ReleaseDetailCollector()
        try:
            collector.feed(detail_text)
        except Exception as exc:
            failures.append(f"cannot parse {relative}: {exc}")
            continue
        if len(collector.sections) != 1:
            failures.append(
                f"release detail {identifier!r} must contain exactly one detail section"
            )
            continue

        detail = collector.sections[0]
        commit = str(release.get("publicCommit", ""))
        category = str(release.get("category", ""))
        release_links = release.get("publicLinks")
        public_links = release_links if isinstance(release_links, list) else []
        expected_text = " ".join(
            [
                str(release.get("state", "")).upper(),
                str(release.get("publishedAt", ""))[:10],
                RELEASE_CATEGORY_LABELS.get(category, ""),
                str(release.get("title", "")),
                str(release.get("summary", "")),
                "Boundary:",
                str(release.get("scope", "")),
                "Inspect commit",
                commit[:12],
                "→",
                *[
                    str(link.get("label", "")) + " →"
                    for link in public_links
                    if isinstance(link, dict)
                ],
                "Back to all reviewed releases",
                "Static release boundary. This page does not create a new release, measurement, ranking, runtime, model, acquisition, or publication authority.",
                "Read the public methodology →",
            ]
        )
        expected_text = " ".join(expected_text.split())
        expected_links = [str(release.get("sourceUrl", ""))]
        expected_links.extend(
            relative_href(relative, str(link["path"]))
            for link in public_links
            if isinstance(link, dict) and isinstance(link.get("path"), str)
        )
        expected_links.extend(["../", "../../methodology/"])

        if detail.get("id") != identifier:
            failures.append(f"release detail {identifier!r} has the wrong id")
        if detail.get("publicCommit") != release.get("publicCommit"):
            failures.append(f"release detail {identifier!r} has the wrong commit")
        if detail.get("datetime") != release.get("publishedAt"):
            failures.append(f"release detail {identifier!r} has the wrong timestamp")
        if detail.get("text") != expected_text:
            failures.append(f"release detail {identifier!r} has drifted reviewed text")
        if detail.get("links") != expected_links:
            failures.append(f"release detail {identifier!r} has the wrong action links")
        if (
            detail.get("h1Count") != 1
            or collector.page_h1_count != 1
            or detail.get("hasNestedDetail")
        ):
            failures.append(f"release detail {identifier!r} has the wrong heading structure")
        if detail.get("hasDuplicateAttributes"):
            failures.append(f"release detail {identifier!r} has duplicate attributes")
        if detail.get("hasVisibilitySuppressor"):
            failures.append(f"release detail {identifier!r} is hidden")
        if collector.scripts:
            failures.append(f"release detail {identifier!r} must not load scripts")
    return failures


def validate(site: Path) -> List[str]:
    failures: List[str] = []
    site = site.resolve()
    required = [
        "index.html",
        "quickstart/index.html",
        "license/index.html",
        "status/index.html",
        "process/index.html",
        "methodology/index.html",
        "capabilities/index.html",
        "capabilities/index.json",
        *[
            f'capabilities/{capability["id"]}/index.html'
            for capability in REVIEWED_CAPABILITIES
        ],
        "benchmarks/index.html",
        "benchmarks/served-benchmarks.json",
        *[
            f'benchmarks/{highlight["id"]}/index.html'
            for highlight in REVIEWED_BENCHMARK_HIGHLIGHTS
        ],
        *[
            f'benchmarks/{SERVED_BENCHMARK_ROWS_DIRNAME}/{identifier}.json'
            for identifier, _card_id in REVIEWED_SERVED_BENCHMARK_IDENTITIES
        ],
        "releases/index.html",
        "releases/index.json",
        "releases/feed.atom",
        *[
            f"releases/{identifier}/index.html"
            for identifier, _title in REVIEWED_RELEASE_IDENTITIES
        ],
        "research/index.html",
        "research/index.json",
        "research/feed.atom",
        "feed.atom",
        "sitemap.xml",
        "robots.txt",
        "assets/site.css",
        "assets/benchmark-explorer.js",
        RESEARCH_EXPLORER_SCRIPT_PATH,
        "assets/favicon.svg",
        SOCIAL_CARD_PATH,
        "llms.txt",
        ".nojekyll",
    ]
    for relative in required:
        if not (site / relative).is_file():
            failures.append(f"missing required file: {relative}")

    failures.extend(validate_social_card(site))
    failures.extend(validate_reviewed_stylesheet(site))
    failures.extend(validate_quickstart_page(site))
    failures.extend(validate_license_page(site))
    failures.extend(validate_status_page(site))
    failures.extend(validate_research_explorer_script(site))
    failures.extend(validate_reviewed_home_page(site))
    failures.extend(validate_reviewed_head_metadata(site))
    failures.extend(validate_quickstart_navigation(site))
    failures.extend(validate_status_navigation(site))
    failures.extend(validate_research_page(site))
    failures.extend(validate_quality_guide_page(site))
    failures.extend(validate_served_benchmarks(site))

    expected_benchmark_cards = reviewed_benchmark_cards()

    release_index: Optional[Dict[str, object]] = None
    if (site / "releases/index.json").is_file():
        release_index, release_failures = load_release_index(site)
        failures.extend(release_failures)
    if release_index is not None and (site / "releases/index.html").is_file():
        failures.extend(validate_release_page(site, release_index))
    if release_index is not None and (site / "releases/feed.atom").is_file():
        failures.extend(validate_release_feed(site, release_index))
    if release_index is not None:
        failures.extend(validate_release_detail_pages(site, release_index))

    feed_path = site / "research/feed.atom"
    if feed_path.exists() or feed_path.is_symlink():
        failures.extend(validate_research_feed(site))
    reviewed_updates_feed_path = site / "feed.atom"
    if (
        release_index is not None
        and (
            reviewed_updates_feed_path.exists()
            or reviewed_updates_feed_path.is_symlink()
        )
    ):
        failures.extend(validate_reviewed_updates_feed(site, release_index))

    research_index: Optional[Dict[str, object]] = None
    index_path = site / "research/index.json"
    if index_path.is_symlink() or not index_path.is_file():
        if index_path.exists() or index_path.is_symlink():
            failures.append(
                "research/index.json must be a regular non-symlink file"
            )
    else:
        try:
            index_size = index_path.stat().st_size
        except OSError as exc:
            failures.append(f"cannot stat research/index.json: {exc}")
        else:
            if index_size > MAX_RESEARCH_INDEX_BYTES:
                failures.append(
                    "research/index.json exceeds the 1048576-byte limit"
                )
            else:
                try:
                    raw_index = index_path.read_bytes()
                except OSError as exc:
                    failures.append(f"cannot read research/index.json: {exc}")
                else:
                    if len(raw_index) > MAX_RESEARCH_INDEX_BYTES:
                        failures.append(
                            "research/index.json exceeds the 1048576-byte limit"
                        )
                    else:
                        try:
                            index_text = raw_index.decode("utf-8")
                        except UnicodeDecodeError as exc:
                            failures.append(
                                f"research/index.json is not UTF-8: {exc}"
                            )
                        else:
                            try:
                                index = json.loads(index_text)
                            except json.JSONDecodeError as exc:
                                failures.append(f"invalid research/index.json: {exc}")
                            else:
                                if not isinstance(index, dict):
                                    failures.append(
                                        "research/index.json is not an object"
                                    )
                                else:
                                    research_index = index
                                    if index.get("schemaVersion") != 1:
                                        failures.append(
                                            "research/index.json does not use schemaVersion 1"
                                        )
                                    articles = index.get("articles", [])
                                    if not isinstance(articles, list) or not articles:
                                        failures.append(
                                            "research/index.json has no articles"
                                        )
                                    else:
                                        seen_article_paths: set[str] = set()
                                        sitemap_article_paths: List[str] = []
                                        for article in articles:
                                            path = (
                                                article.get("path")
                                                if isinstance(article, dict)
                                                else None
                                            )
                                            if (
                                                not isinstance(path, str)
                                                or not SITEMAP_ARTICLE_PATH.fullmatch(path)
                                                or path in seen_article_paths
                                            ):
                                                failures.append(
                                                    "invalid or duplicate article path in "
                                                    f"index entry: {path!r}"
                                                )
                                                continue
                                            seen_article_paths.add(path)
                                            sitemap_article_paths.append(path)
                                            if not (site / path / "index.html").is_file():
                                                failures.append(
                                                    "missing article page for index entry: "
                                                    f"{path!r}"
                                                )
                                        if tuple(sitemap_article_paths) != REVIEWED_ARTICLE_PATHS:
                                            failures.append(
                                                "research/index.json does not match "
                                                "reviewed article routes"
                                            )
                                    if index != render_expected_research_index():
                                        failures.append(
                                            "research/index.json does not match the "
                                            "reviewed research catalog"
                                        )

    sitemap_path = site / "sitemap.xml"
    if sitemap_path.exists() or sitemap_path.is_symlink():
        failures.extend(validate_sitemap(site))
    robots_path = site / "robots.txt"
    if robots_path.exists() or robots_path.is_symlink():
        failures.extend(validate_robots(site))

    capability_index: Optional[Dict[str, object]] = None
    capability_index_path = site / "capabilities/index.json"
    if capability_index_path.is_file():
        try:
            capability_index = json.loads(
                capability_index_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            failures.append(f"invalid capabilities/index.json: {exc}")
        else:
            if capability_index.get("schemaVersion") != 1:
                failures.append("capabilities/index.json does not use schemaVersion 1")
            if capability_index.get("project") != "fast-mlx":
                failures.append("capabilities/index.json has the wrong project")
            if capability_index.get("claimBoundary") != "fast-mlx-owned-results-only":
                failures.append("capabilities/index.json has the wrong claim boundary")

            status_definitions = capability_index.get("statusDefinitions")
            definition_ids = {
                item.get("id")
                for item in status_definitions
                if isinstance(item, dict)
            } if isinstance(status_definitions, list) else set()
            if definition_ids != CAPABILITY_STATUSES:
                failures.append("capabilities/index.json has incomplete status definitions")

            capabilities = capability_index.get("capabilities")
            if not isinstance(capabilities, list) or not capabilities:
                failures.append("capabilities/index.json has no capabilities")
            else:
                if capabilities != list(reviewed_capability_records()):
                    failures.append(
                        "capabilities/index.json capabilities do not match reviewed capability records"
                    )
                seen_capability_ids: set[str] = set()
                for position, capability in enumerate(capabilities):
                    label = f"capability index entry {position}"
                    if not isinstance(capability, dict):
                        failures.append(f"{label} is not an object")
                        continue
                    identifier = capability.get("id")
                    if not isinstance(identifier, str) or identifier in seen_capability_ids:
                        failures.append(f"{label} has an invalid or duplicate id")
                    else:
                        seen_capability_ids.add(identifier)
                    if capability.get("status") not in CAPABILITY_STATUSES:
                        failures.append(f"{label} has an unknown status")
                    evidence = capability.get("evidence")
                    if not isinstance(evidence, list) or not evidence:
                        failures.append(f"{label} has no evidence")
                    else:
                        for evidence_position, record in enumerate(evidence):
                            raw_path = record.get("path") if isinstance(record, dict) else None
                            failures.extend(
                                validate_evidence_path(
                                    site,
                                    raw_path,
                                    f"{label} evidence {evidence_position}",
                                )
                            )

            highlights = capability_index.get("performanceHighlights")
            if not isinstance(highlights, list) or not highlights:
                failures.append("capabilities/index.json has no performance highlights")
            else:
                if highlights != list(REVIEWED_BENCHMARK_HIGHLIGHTS):
                    failures.append(
                        "capabilities/index.json performance highlights do not match reviewed benchmark highlights"
                    )
                seen_highlight_ids: set[str] = set()
                for position, highlight in enumerate(highlights):
                    label = f"performance highlight entry {position}"
                    if not isinstance(highlight, dict):
                        failures.append(f"{label} is not an object")
                        continue
                    identifier = highlight.get("id")
                    if not isinstance(identifier, str) or identifier in seen_highlight_ids:
                        failures.append(f"{label} has an invalid or duplicate id")
                    else:
                        seen_highlight_ids.add(identifier)
                    if highlight.get("decision") not in {"promoted-scoped", "shelved"}:
                        failures.append(f"{label} has an unknown decision")
                    for key in (
                        "metric",
                        "label",
                        "model",
                        "hardware",
                        "workload",
                        "date",
                        "caveat",
                    ):
                        if not isinstance(highlight.get(key), str) or not highlight[key].strip():
                            failures.append(f"{label} has an empty {key}")
                    evidence = highlight.get("evidence")
                    raw_path = evidence.get("path") if isinstance(evidence, dict) else None
                    failures.extend(validate_evidence_path(site, raw_path, label))
    failures.extend(validate_capability_cards(site))
    failures.extend(validate_capability_detail_pages(site))

    benchmark_path = site / "benchmarks/index.html"
    if benchmark_path.is_file():
        collector = BenchmarkCollector()
        try:
            collector.feed(benchmark_path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"cannot parse benchmarks/index.html: {exc}")
        else:
            actual_ids = [card.get("id") for card in collector.cards]
            if (
                len(actual_ids) != len(set(actual_ids))
                or set(actual_ids) != set(expected_benchmark_cards)
            ):
                failures.append(
                    "benchmark explorer card set does not match performance highlights"
                )
            expected_order = [
                identifier
                for identifier, _entry in sorted(
                    expected_benchmark_cards.items(),
                    key=lambda item: item[1]["date"],
                    reverse=True,
                )
            ]
            if (
                len(actual_ids) == len(set(actual_ids))
                and set(actual_ids) == set(expected_benchmark_cards)
                and actual_ids != expected_order
            ):
                failures.append(
                    "benchmark explorer cards are not ordered by descending evidence date"
                )
            for card in collector.cards:
                identifier = card.get("id")
                expected = expected_benchmark_cards.get(
                    identifier if isinstance(identifier, str) else ""
                )
                if expected is None:
                    continue
                if card.get("hidden"):
                    failures.append(
                        f"benchmark explorer card {identifier!r} is hidden before enhancement"
                    )
                for key in ("model", "hardware", "decision"):
                    if card.get(key) != expected[key]:
                        failures.append(
                            f"benchmark explorer card {identifier!r} has the wrong {key}"
                        )
                card_text = card.get("text")
                normalized_text = card_text if isinstance(card_text, str) else ""
                for key in (
                    "metric",
                    "label",
                    "model",
                    "hardware",
                    "workload",
                    "date",
                    "caveat",
                ):
                    normalized_expected = " ".join(expected[key].split())
                    if normalized_expected not in normalized_text:
                        message = (
                            f"benchmark explorer card {identifier!r} has the wrong {key}"
                        )
                        if message not in failures:
                            failures.append(message)
                if card.get("datetime") != expected["date"]:
                    message = (
                        f"benchmark explorer card {identifier!r} has the wrong date"
                    )
                    if message not in failures:
                        failures.append(message)
                links = card.get("links")
                if not isinstance(links, list) or expected["detail"] not in links:
                    failures.append(
                        f"benchmark explorer card {identifier!r} has the wrong detail link"
                    )
                if not isinstance(links, list) or expected["evidence"] not in links:
                    failures.append(
                        f"benchmark explorer card {identifier!r} has the wrong evidence"
                    )
                if isinstance(links, list) and links != [
                    expected["detail"],
                    expected["evidence"],
                ]:
                    failures.append(
                        f"benchmark explorer card {identifier!r} has unexpected action links"
                    )
            expected_options = {
                "model": [("", "All models")]
                + [
                    (value, value)
                    for value in sorted(
                        {entry["model"] for entry in expected_benchmark_cards.values()},
                        key=str.casefold,
                    )
                ],
                "hardware": [("", "All hardware")]
                + [
                    (value, value)
                    for value in sorted(
                        {entry["hardware"] for entry in expected_benchmark_cards.values()},
                        key=str.casefold,
                    )
                ],
                "decision": [("", "All decisions")]
                + [
                    (value, HIGHLIGHT_DECISION_LABELS[value])
                    for value in sorted(
                        {entry["decision"] for entry in expected_benchmark_cards.values()},
                        key=lambda value: HIGHLIGHT_DECISION_LABELS[value].casefold(),
                    )
                ],
            }
            for name, expected in expected_options.items():
                actual = [
                    (option.get("value"), option.get("text"))
                    for option in collector.options[name]
                ]
                if actual != expected:
                    failures.append(
                        f"benchmark explorer {name} options do not match performance highlights"
                    )
            if not collector.has_controls:
                failures.append("benchmark explorer has no filter controls")
            if not collector.has_count:
                failures.append("benchmark explorer has no live result count")
            if not collector.has_empty_state:
                failures.append("benchmark explorer has no hidden status empty state")
            if not collector.has_script:
                failures.append("benchmark explorer does not load its reviewed script")

    failures.extend(validate_benchmark_detail_pages(site))

    if (
        release_index is not None
        and capability_index is not None
        and research_index is not None
        and (site / "index.html").is_file()
    ):
        failures.extend(
            validate_home_current_cycle(
                site,
                release_index,
                capability_index,
                research_index,
            )
        )

    for path in site.rglob("*"):
        if path.is_symlink():
            failures.append(f"symlink is forbidden in Pages output: {path.relative_to(site)}")
            continue
        if not path.is_file():
            continue
        if path.relative_to(site).as_posix() == SOCIAL_CARD_PATH:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"non-UTF-8 output file: {path.relative_to(site)}")
            continue
        for marker in PRIVATE_MARKERS:
            if marker.casefold() in text.casefold():
                failures.append(
                    f"private marker {marker!r} in {path.relative_to(site)}"
                )
        if path.suffix.casefold() not in HTML_LIKE_SUFFIXES:
            continue
        collector = LinkCollector()
        try:
            collector.feed(text)
        except Exception as exc:  # HTMLParser should be tolerant; make any failure explicit.
            failures.append(f"cannot parse {path.relative_to(site)}: {exc}")
            continue
        for link in collector.links:
            target = resolve_target(site, path, link)
            if target is None:
                continue
            try:
                target.relative_to(site)
            except ValueError:
                failures.append(
                    f"link escapes site root: {path.relative_to(site)} -> {link}"
                )
                continue
            if not target.exists():
                failures.append(
                    f"broken internal link: {path.relative_to(site)} -> {link}"
                )
    return failures


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_arguments(argv)
    if not arguments.site.is_dir():
        print(f"public-site validation refused: not a directory: {arguments.site}", file=sys.stderr)
        return 2
    failures = validate(arguments.site)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(f"validated public site: {arguments.site.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
