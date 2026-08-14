#!/usr/bin/env python3
"""Validate generated fast-mlx Pages files and internal links."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html.parser
import json
import posixpath
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlsplit


PRIVATE_MARKERS: Tuple[str, ...] = (
    "/" + "Users/",
    "/" + "private/",
    "192" + ".168.",
    "llm" + "bench",
    "passwordless" + " sudo",
    "BEGIN OPENSSH" + " PRIVATE KEY",
    "BEGIN RSA" + " PRIVATE KEY",
)
CAPABILITY_STATUSES = {"implemented", "promoted-scoped", "experimental", "shelved"}
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
        "summary": "A chat-completions transport with streaming, bounded admission, cancellation, and evidence output.",
        "scope": "The explicit temperature-zero continuous-batch-no-spec route is qualified only for the published source-locked Qwen3-32B-4bit workload. Scripted mode is transport-only, and no model weights are bundled.",
        "evidenceSlugs": ("the-proof-did-not-end-when-the-timer-did",),
    },
    {
        "id": "exact-continuous-batching",
        "name": "Exact continuous batching",
        "status": "promoted-scoped",
        "summary": "Simultaneous dense-model streams can join, advance, cancel, and release reservations through one explicit no-spec route.",
        "scope": "The measured policy is a dense-Qwen building block for the named paired workloads. It is not an automatic router, a sampled-generation route, or a general model-family default.",
        "evidenceSlugs": (
            "the-fastest-request-wasnt-the-fastest-service",
            "the-proof-did-not-end-when-the-timer-did",
        ),
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
        "summary": "Command-line planning, provenance, and fail-closed evidence controls make resource and identity assumptions inspectable.",
        "scope": "Planning and comparison tools do not prove that a model fits every Mac, that an artifact is launchable, or that a runtime is contained.",
        "evidenceSlugs": ("the-wall-that-wasnt", "lossless-wasnt-byte-identical"),
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
REVIEWED_HOME_PAGE_BYTES = 9_343
REVIEWED_HOME_PAGE_SHA256 = (
    "a6486910fc68efca6e03e85b76792a950fc9377ecc72ba14463652505d562c6d"
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
MAX_STATUS_BYTES = 131_072
MAX_CAPABILITY_CATALOG_BYTES = 131_072
MAX_CAPABILITY_DETAIL_BYTES = 131_072
REVIEWED_QUICKSTART_PAGE_BYTES = 9_794
REVIEWED_QUICKSTART_PAGE_SHA256 = (
    "c9c990104fe7573d0f84683362c010c3a6945c13423d12dc9c4fd6058bd5506b"
)
REVIEWED_STATUS_PAGE_BYTES = 18_473
REVIEWED_STATUS_PAGE_SHA256 = (
    "d513f3d63bbb07241d4fbbe32f781d82e99c83bfb4ee2ecf211e63be076472e0"
)
REVIEWED_CAPABILITIES_PAGE_BYTES = 13_817
REVIEWED_CAPABILITIES_PAGE_SHA256 = (
    "ad8f887a455282a45014de9a10bcdaf811065aa518153cf6d097771094aaf12f"
)
REVIEWED_CAPABILITY_DETAIL_SEALS: Dict[str, Tuple[int, str]] = {
    "openai-http-sse-serving": (
        5_218,
        "ee0d20469fb5b850c534ee37e940e5d34562ad61536dff3373238f4b47f47b60",
    ),
    "exact-continuous-batching": (
        5_552,
        "816aa47d8282e086085f30ca4f771a95d8404b1e3238a7f2822bf9af2e919e42",
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
        5_559,
        "f5b86d209a73f0c20784ea05fe655f970245ac68f5de5a9d90521649bbe98dd9",
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
    ("prompt-lookup-decoding", "shelved"),
    ("exact-cache-lifecycle-controls", "implemented"),
    ("quality-measurement-harness", "implemented"),
    ("capacity-proof-control-tools", "implemented"),
)
REVIEWED_STATUS_COUNTS: Tuple[Tuple[str, str], ...] = (
    ("implemented", "3"),
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
    "../capabilities/openai-http-sse-serving/",
    "../research/the-fastest-request-wasnt-the-fastest-service/",
    "../research/the-proof-did-not-end-when-the-timer-did/",
    "../capabilities/exact-continuous-batching/",
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
    "../benchmarks/pld-echo-throughput/",
    "../research/when-zero-speculation-costs-two-percent/",
    "../benchmarks/continuous-batch-c2-throughput/",
    "../research/the-fastest-request-wasnt-the-fastest-service/",
    "../benchmarks/http-sse-operational-soak/",
    "../research/the-proof-did-not-end-when-the-timer-did/",
    "../releases/self-reproducing-public-source/",
    "https://github.com/bitworks-io/fast-mlx/commit/c9ba0341a473dccf219f421efefedf3df3e30e2f",
    "../methodology/",
    "../capabilities/index.json",
    "../releases/index.json",
    "../research/index.json",
)
REVIEWED_STATUS_TEXT = (
    "Current state, not a roadmap.",
    "does not create new measurement, performance, model, runtime, acquisition, or publication authority",
    "6 reviewed capabilities",
    "3 measured proof points",
    "7 published research notes",
    "14 reviewed release records",
    "Released source and comparison evidence do not grant unreviewed model, acquisition, launchability, containment, or runtime authority.",
    "This page performs no live lookup, ranking, aggregation, benchmark execution, or authority transition.",
)
CORE_PUBLIC_PAGE_PATHS = (
    "",
    "quickstart/",
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
    "research/the-proof-did-not-end-when-the-timer-did/",
    "research/the-fastest-request-wasnt-the-fastest-service/",
    "research/lossless-wasnt-byte-identical/",
    "research/when-zero-speculation-costs-two-percent/",
    "research/turboquant-exact-math-still-lost/",
    "research/trusting-the-instrument/",
    "research/the-wall-that-wasnt/",
)
REVIEWED_ARTICLE_DATES: Dict[str, Tuple[str, str]] = {
    "research/the-proof-did-not-end-when-the-timer-did/": (
        "2026-07-28",
        "2026-08-06",
    ),
    "research/the-fastest-request-wasnt-the-fastest-service/": (
        "2026-07-14",
        "2026-08-11",
    ),
    "research/lossless-wasnt-byte-identical/": ("2026-07-12", "2026-08-06"),
    "research/when-zero-speculation-costs-two-percent/": (
        "2026-07-11",
        "2026-08-06",
    ),
    "research/turboquant-exact-math-still-lost/": (
        "2026-07-09",
        "2026-08-06",
    ),
    "research/trusting-the-instrument/": ("2026-07-09", "2026-08-06"),
    "research/the-wall-that-wasnt/": ("2026-07-09", "2026-08-06"),
}
REVIEWED_RELEASE_INDEX_BYTES = 13_733
REVIEWED_RELEASE_INDEX_SHA256 = (
    "523ab892cd87c82d5d99a605f585bbee440e11a159b9d5639732090112114a35"
)
REVIEWED_RELEASE_IDENTITIES: Tuple[Tuple[str, str], ...] = (
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
        "fast-mlx — evidence-gated MLX inference",
        "A Swift and MLX inference project that continuously researches, tests, and publishes verified capabilities.",
        "website",
        None,
    ),
    "quickstart/": (
        "Operator quickstart — fast-mlx",
        "Run fast-mlx's model-free HTTP/JSON and HTTP/SSE transport smoke, inspect capacity, and understand the loaded-serving boundary.",
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
    "research/the-proof-did-not-end-when-the-timer-did/": (
        "The proof did not end when the timer did — fast-mlx",
        "A short benchmark can show that continuous batching works. It cannot show that an HTTP service keeps cleaning up after disappearing clients for a full day.",
        "article",
        "Building a high-performance MLX inference engine in Swift; Serving big models on Apple Silicon; Rapid research integration — the flywheel",
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


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path, help="generated site directory")
    return parser.parse_args(argv)


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
        "data-latest-release-id": "self-reproducing-public-source",
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
    allowed_html_files = set(expected_files) | {"404.html"}
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


def validate_benchmark_detail_pages(site: Path) -> List[str]:
    failures: List[str] = []
    benchmark_root = site / "benchmarks"
    expected_entries = {"index.html"} | {
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
        *[
            f'benchmarks/{highlight["id"]}/index.html'
            for highlight in REVIEWED_BENCHMARK_HIGHLIGHTS
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
    failures.extend(validate_status_page(site))
    failures.extend(validate_research_explorer_script(site))
    failures.extend(validate_reviewed_home_page(site))
    failures.extend(validate_reviewed_head_metadata(site))
    failures.extend(validate_quickstart_navigation(site))
    failures.extend(validate_status_navigation(site))
    failures.extend(validate_research_page(site))

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
