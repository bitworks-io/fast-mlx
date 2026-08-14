#!/usr/bin/env python3
"""Offline validation gate for fast-mlx public deployment receipts.

This program intentionally does not import the online deployment verifier.  It
rebuilds the local publication subject from checked-out files and validates a
raw receipt JSON file, or the one exact receipt member inside an artifact ZIP,
without network, subprocess, Git, GitHub, extraction, or mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, NoReturn, Sequence, TextIO

try:
    import validate_public_site
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import validate_public_site


BASE_URL = "https://bitworks-io.github.io/fast-mlx/"
BASE_PATH = "/fast-mlx"
REPOSITORY = "bitworks-io/fast-mlx"
KIND = "fast-mlx-public-deployment-receipt"
SCHEMA_VERSION = 1
RECEIPT_ARTIFACT_MEMBER = "fast-mlx-publication-receipt.json"
RAW_RECEIPT_LIMIT = 2 * 1024 * 1024
ARTIFACT_ZIP_LIMIT = 4 * 1024 * 1024
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_EXPECTED_BYTES = 64 * 1024 * 1024
MAX_ROUTES = 256
MAX_COHORTS = 24
STABLE_COHORTS_REQUIRED = 2

HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
POSITIVE_DECIMAL = re.compile(r"[1-9][0-9]*")

MIME_TYPES = {
    ".atom": "application/atom+xml",
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".txt": "text/plain; charset=utf-8",
    ".xml": "application/xml",
}

AUTHORITY_FLAGS = {
    "acquisition_authority",
    "automatic_evidence_intake",
    "automatic_publication_authority",
    "containment_claim",
    "launchability_claim",
    "model_authority",
    "performance_claim",
    "positive_admission",
    "rollback_authority",
    "runtime_authority",
}

POLICY = {
    "accepted_base_url": BASE_URL,
    "max_abandoned_resolver_threads": 24,
    "max_cohorts": MAX_COHORTS,
    "max_file_bytes": MAX_FILE_BYTES,
    "max_observation_seconds": 600,
    "max_resolved_addresses": 16,
    "max_routes": MAX_ROUTES,
    "max_total_expected_bytes": MAX_TOTAL_EXPECTED_BYTES,
    "per_request_timeout_seconds": 10,
    "resolver_timeout_seconds": 10,
    "retry_delay_seconds": 25,
    "stable_cohorts_required": STABLE_COHORTS_REQUIRED,
}

FAILURE_CODES = {
    "body_too_large",
    "byte_count_mismatch",
    "byte_mismatch",
    "cohort_attempts_exhausted",
    "connect_error",
    "connect_timeout",
    "content_encoding_refused",
    "content_length_refused",
    "content_type_refused",
    "deadline_exhausted",
    "dns_policy_refused",
    "http_status_mismatch",
    "internal_error",
    "read_timeout",
    "resolver_error",
    "resolver_timeout",
    "response_protocol_error",
    "tls_error",
    "transfer_encoding_refused",
}

POLICY_WIDE_FAILURE_CODES = {
    "cohort_attempts_exhausted",
    "deadline_exhausted",
    "dns_policy_refused",
    "resolver_error",
    "resolver_timeout",
}

NULL_STATUS_FAILURE_CODES = {
    "cohort_attempts_exhausted",
    "connect_error",
    "connect_timeout",
    "deadline_exhausted",
    "dns_policy_refused",
    "internal_error",
    "resolver_error",
    "resolver_timeout",
    "tls_error",
}

EXPECTED_STATUS_FAILURE_CODES = {
    "body_too_large",
    "byte_count_mismatch",
    "byte_mismatch",
    "content_encoding_refused",
    "content_length_refused",
    "content_type_refused",
    "read_timeout",
    "transfer_encoding_refused",
}

NULL_BODY_OBSERVATION_FAILURE_CODES = FAILURE_CODES - {
    "body_too_large",
    "byte_count_mismatch",
    "byte_mismatch",
}

ROOT_KEYS = {
    "authority",
    "comparison_sha256",
    "kind",
    "observation",
    "policy",
    "result",
    "schema_version",
    "subject",
    "workflow",
}
SUBJECT_KEYS = {
    "base_url",
    "commit_sha",
    "excluded_reader_control",
    "generated_file_count",
    "repository",
    "route_count",
    "route_manifest_sha256",
    "site_ordered_sha256",
    "verifier_source_sha256",
}
WORKFLOW_KEYS = {"run_attempt", "run_id"}
OBSERVATION_KEYS = {
    "accepted_cohort_sha256",
    "attempted_cohorts",
    "attempts",
    "consecutive_matching_cohorts",
    "failures",
    "routes",
}
ATTEMPT_KEYS = {"attempt", "cohort_sha256", "completed_routes", "failure", "result"}
FAILURE_KEYS = {"code", "observed_bytes", "observed_sha256", "path", "status"}
ROUTE_KEYS = {
    "bytes",
    "content_type",
    "path",
    "request_target",
    "sha256",
    "source",
    "status",
}
ROUTE_SUBJECT_KEYS = {
    "expected_bytes",
    "expected_content_type",
    "expected_sha256",
    "expected_status",
    "path",
    "request_target",
    "source",
}


class UsageError(Exception):
    """Invalid invocation or unsafe local input; maps to exit 2."""


class ValidationError(Exception):
    """Receipt/schema/identity mismatch; maps to exit 1."""


class DuplicateKeyError(ValueError):
    """JSON object contained a duplicate key."""


class ReceiptArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that reports errors through the bounded JSON envelope."""

    def error(self, message: str) -> NoReturn:
        raise UsageError(f"invalid CLI invocation: {message}")


@dataclass(frozen=True)
class FileRecord:
    source: str
    size: int
    sha256_hex: str
    sha256_raw: bytes


@dataclass(frozen=True)
class LocalSubject:
    files: list[FileRecord]
    route_subject: list[dict[str, Any]]
    accepted_routes: list[dict[str, Any]]
    site_ordered_sha256: str
    route_manifest_sha256: str
    verifier_source_sha256: str | None


def canonical_compact(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_pretty(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_hex(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def fail_json(
    stream: TextIO, result: str, errors: Iterable[str], **extra: object
) -> None:
    bounded_errors = []
    for index, error in enumerate(errors):
        if index >= 20:
            bounded_errors.append("additional errors omitted")
            break
        bounded_errors.append(str(error).replace("\n", " ")[:240])
    payload = {"errors": bounded_errors, "result": result, **extra}
    print(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        file=stream,
    )


def success_json(receipt_bytes: bytes, receipt: dict[str, Any]) -> None:
    observation = expect_object(receipt.get("observation"), "observation")
    routes = expect_list(observation.get("routes"), "observation.routes")
    payload = {
        "comparison_sha256": receipt.get("comparison_sha256"),
        "receipt_sha256": sha256_hex(receipt_bytes),
        "result": receipt.get("result"),
        "route_count": len(routes),
    }
    print(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    )


def parser_error(message: str) -> NoReturn:
    raise UsageError(message)


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    allowed_once = {
        "--repository-root",
        "--site",
        "--receipt",
        "--artifact-zip",
        "--commit-sha",
        "--workflow-run-id",
        "--workflow-run-attempt",
        "--require-result",
    }
    seen: dict[str, int] = {}
    for arg in argv:
        for option in allowed_once:
            if arg == option or arg.startswith(option + "="):
                seen[option] = seen.get(option, 0) + 1
    repeated = sorted(key for key, count in seen.items() if count > 1)
    if repeated:
        parser_error(f"CLI option must appear exactly once: {', '.join(repeated)}")

    parser = ReceiptArgumentParser(
        description=__doc__,
        allow_abbrev=False,
    )
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--site", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--receipt", type=Path)
    group.add_argument("--artifact-zip", type=Path)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--workflow-run-id", required=True)
    parser.add_argument("--workflow-run-attempt", required=True)
    parser.add_argument("--require-result", required=True, choices=("PASS", "FAIL"))
    args = parser.parse_args(argv)

    if not HEX40.fullmatch(args.commit_sha):
        parser_error("--commit-sha must be 40 lowercase hexadecimal characters")
    args.workflow_run_id_int = parse_positive_decimal(
        args.workflow_run_id, "--workflow-run-id"
    )
    args.workflow_run_attempt_int = parse_positive_decimal(
        args.workflow_run_attempt, "--workflow-run-attempt"
    )
    return args


def parse_positive_decimal(value: str, label: str) -> int:
    if not POSITIVE_DECIMAL.fullmatch(value):
        parser_error(f"{label} must be a positive decimal integer")
    return int(value, 10)


def require_existing_directory(path: Path, label: str) -> Path:
    try:
        st = path.lstat()
    except OSError as exc:
        raise UsageError(f"{label} is not accessible: {exc}") from exc
    if stat.S_ISLNK(st.st_mode):
        raise UsageError(f"{label} must not be a symlink")
    if not stat.S_ISDIR(st.st_mode):
        raise UsageError(f"{label} must be an existing directory")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise UsageError(f"{label} cannot be resolved: {exc}") from exc


def require_existing_file(path: Path, label: str) -> Path:
    try:
        st = path.lstat()
    except OSError as exc:
        raise UsageError(f"{label} is not accessible: {exc}") from exc
    if stat.S_ISLNK(st.st_mode):
        raise UsageError(f"{label} must not be a symlink")
    if not stat.S_ISREG(st.st_mode):
        raise UsageError(f"{label} must be an existing regular file")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise UsageError(f"{label} cannot be resolved: {exc}") from exc


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def require_outside(path: Path, roots: Sequence[tuple[Path, str]], label: str) -> None:
    for root, root_label in roots:
        if path == root or is_relative_to(path, root):
            raise UsageError(f"{label} must be outside {root_label}")


def open_no_follow(path: Path) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        return os.open(path, flags)
    except OSError as exc:
        raise UsageError(f"cannot open {path}: {exc}") from exc


def read_regular_file(path: Path, label: str, limit: int) -> bytes:
    fd = open_no_follow(path)
    try:
        st = os.fstat(fd)
        if stat.S_ISLNK(st.st_mode):
            raise UsageError(f"{label} must not be a symlink")
        if not stat.S_ISREG(st.st_mode):
            raise UsageError(f"{label} must be a regular file")
        if st.st_size > limit:
            raise UsageError(f"{label} exceeds the {limit}-byte limit")
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining > 0:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > limit:
            raise UsageError(f"{label} exceeds the {limit}-byte limit")
        return raw
    finally:
        os.close(fd)


def read_receipt_bytes(receipt_path: Path | None, artifact_path: Path | None) -> bytes:
    if receipt_path is not None:
        return read_regular_file(receipt_path, "receipt", RAW_RECEIPT_LIMIT)
    if artifact_path is None:
        raise UsageError("exactly one receipt input is required")
    return read_receipt_from_zip(artifact_path)


def read_receipt_from_zip(path: Path) -> bytes:
    raw_fd = open_no_follow(path)
    try:
        st = os.fstat(raw_fd)
        if not stat.S_ISREG(st.st_mode):
            raise UsageError("artifact ZIP must be a regular file")
        if st.st_size > ARTIFACT_ZIP_LIMIT:
            raise UsageError(
                f"artifact ZIP exceeds the {ARTIFACT_ZIP_LIMIT}-byte limit"
            )
        with os.fdopen(raw_fd, "rb") as handle:
            raw_fd = -1
            try:
                with zipfile.ZipFile(handle, "r") as archive:
                    infos = archive.infolist()
                    if len(infos) != 1:
                        raise UsageError("artifact ZIP must contain exactly one member")
                    info = infos[0]
                    validate_zip_member(info)
                    with archive.open(info, "r") as member:
                        raw = member.read(RAW_RECEIPT_LIMIT + 1)
                        if len(raw) > RAW_RECEIPT_LIMIT:
                            raise UsageError(
                                "artifact receipt exceeds the 2097152-byte limit"
                            )
                        if member.read(1):
                            raise UsageError(
                                "artifact receipt exceeds the 2097152-byte limit"
                            )
                        return raw
            except zipfile.BadZipFile as exc:
                raise UsageError(f"invalid artifact ZIP: {exc}") from exc
            except NotImplementedError as exc:
                raise UsageError(f"unsupported artifact ZIP member: {exc}") from exc
    finally:
        if raw_fd != -1:
            os.close(raw_fd)


def validate_zip_member(info: zipfile.ZipInfo) -> None:
    if info.filename != RECEIPT_ARTIFACT_MEMBER:
        raise UsageError("artifact ZIP member name is not exact receipt basename")
    if info.filename.startswith(("/", "\\")):
        raise UsageError("artifact ZIP member must be relative")
    if "/" in info.filename or "\\" in info.filename:
        raise UsageError("artifact ZIP member must not contain separators")
    if info.filename in {".", ".."}:
        raise UsageError("artifact ZIP member must not be dot or dot-dot")
    if info.is_dir():
        raise UsageError("artifact ZIP member must not be a directory")
    if info.flag_bits & 0x1:
        raise UsageError("artifact ZIP member must not be encrypted")
    if info.compress_type != zipfile.ZIP_DEFLATED:
        raise UsageError("artifact ZIP member must use DEFLATE compression")
    if info.file_size > RAW_RECEIPT_LIMIT or info.compress_size > RAW_RECEIPT_LIMIT:
        raise UsageError("artifact ZIP member exceeds the receipt size limit")
    mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(mode)
    if file_type and not stat.S_ISREG(mode):
        raise UsageError("artifact ZIP member mode must be regular or unspecified")


def parse_receipt(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"receipt is not UTF-8: {exc}") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_json_constant,
        )
    except (
        json.JSONDecodeError,
        DuplicateKeyError,
        ValueError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise ValidationError(f"receipt is not valid strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError("receipt top-level value is not an object")
    try:
        canonical = canonical_pretty(value)
    except (
        ValueError,
        TypeError,
        OverflowError,
        RecursionError,
        UnicodeError,
    ) as exc:
        raise ValidationError(
            f"receipt cannot be represented as canonical strict JSON: {exc}"
        ) from exc
    if canonical != raw:
        raise ValidationError("receipt bytes are not canonical pretty JSON plus newline")
    return value


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate key {key!r}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


def build_local_subject(repository_root: Path, site: Path, commit_sha: str) -> LocalSubject:
    files = walk_site(site)
    if not files:
        raise UsageError("site inventory is empty")
    route_subject, accepted_routes = build_routes(files, commit_sha)
    if not route_subject:
        raise UsageError("site has no reader routes")
    site_ordered_sha = compute_site_ordered_sha256(files)
    route_manifest_sha = sha256_hex(canonical_compact(route_subject))

    verifier_path = repository_root / "scripts" / "verify_public_deployment.py"
    verifier_source_sha: str | None = None
    if verifier_path.exists() or verifier_path.is_symlink():
        verifier_source_sha = sha256_hex(
            read_regular_file(
                verifier_path,
                "online verifier source",
                MAX_FILE_BYTES,
            )
        )

    return LocalSubject(
        files=files,
        route_subject=route_subject,
        accepted_routes=accepted_routes,
        site_ordered_sha256=site_ordered_sha,
        route_manifest_sha256=route_manifest_sha,
        verifier_source_sha256=verifier_source_sha,
    )


def walk_site(site: Path) -> list[FileRecord]:
    found_paths: list[Path] = []

    def scan(directory: Path, relative_parent: str) -> None:
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise UsageError(f"cannot scan site directory {relative_parent or '.'}: {exc}") from exc
        entries.sort(key=lambda entry: entry.name.encode("utf-8", "surrogateescape"))
        for entry in entries:
            relative = (
                entry.name if not relative_parent else f"{relative_parent}/{entry.name}"
            )
            validate_relative_source(relative)
            try:
                st = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise UsageError(f"cannot stat site entry {relative}: {exc}") from exc
            if stat.S_ISLNK(st.st_mode):
                raise UsageError(f"site entry must not be a symlink: {relative}")
            if stat.S_ISDIR(st.st_mode):
                scan(Path(entry.path), relative)
            elif stat.S_ISREG(st.st_mode):
                validate_site_file_source(relative)
                found_paths.append(Path(entry.path))
            else:
                raise UsageError(f"site entry must be regular file or directory: {relative}")

    scan(site, "")
    found_sources = {path.relative_to(site).as_posix() for path in found_paths}
    for required in (".nojekyll", "404.html", "index.html"):
        if required not in found_sources:
            raise UsageError(f"site is missing required generated file: {required}")
    records: list[FileRecord] = []
    total_reader_bytes = 0
    for path in sorted(
        found_paths,
        key=lambda item: item.relative_to(site).as_posix().encode("utf-8"),
    ):
        source = path.relative_to(site).as_posix()
        raw = read_regular_file(path, f"site file {source}", MAX_FILE_BYTES)
        digest = hashlib.sha256(raw)
        if source != ".nojekyll":
            total_reader_bytes += len(raw)
            if total_reader_bytes > MAX_TOTAL_EXPECTED_BYTES:
                raise UsageError(
                    "site reader bytes exceed the 67108864-byte aggregate limit"
                )
        records.append(
            FileRecord(
                source=source,
                size=len(raw),
                sha256_hex=digest.hexdigest(),
                sha256_raw=digest.digest(),
            )
        )
    return records


def validate_relative_source(source: str) -> None:
    if "\x00" in source or "\n" in source or "\\" in source:
        raise UsageError(f"unsafe site path: {source!r}")
    if any(ord(char) < 0x20 or ord(char) > 0x7E for char in source):
        raise UsageError(f"site path must be printable ASCII: {source!r}")
    if "%" in source or "?" in source or "#" in source:
        raise UsageError(f"site path contains a refused route delimiter: {source!r}")
    parts = source.split("/")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise UsageError(f"unsafe site path components: {source!r}")


def validate_site_file_source(source: str) -> None:
    if source == ".nojekyll":
        return
    if source.endswith("/.nojekyll") or "/.nojekyll/" in source:
        raise UsageError("nested .nojekyll files are refused")
    suffix = Path(source).suffix
    if suffix not in MIME_TYPES:
        raise UsageError(f"site file has unsupported suffix: {source}")


def build_routes(
    files: list[FileRecord], commit_sha: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    route_subject: list[dict[str, Any]] = []
    accepted_routes: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for record in files:
        if record.source == ".nojekyll":
            continue
        route, request_target, expected_status = route_for_source(
            record.source, commit_sha
        )
        if route in seen_paths:
            raise UsageError(f"duplicate mapped receipt route: {route}")
        seen_paths.add(route)
        if len(seen_paths) > MAX_ROUTES:
            raise UsageError(f"site has more than {MAX_ROUTES} reader routes")
        content_type = MIME_TYPES[Path(record.source).suffix]
        route_subject.append(
            {
                "expected_bytes": record.size,
                "expected_content_type": content_type,
                "expected_sha256": record.sha256_hex,
                "expected_status": expected_status,
                "path": route,
                "request_target": request_target,
                "source": record.source,
            }
        )
        accepted_routes.append(
            {
                "bytes": record.size,
                "content_type": content_type,
                "path": route,
                "request_target": request_target,
                "sha256": record.sha256_hex,
                "source": record.source,
                "status": expected_status,
            }
        )
    return route_subject, accepted_routes


def route_for_source(source: str, commit_sha: str) -> tuple[str, str, int]:
    if source == "index.html":
        route = "/"
        status_code = 200
    elif source == "404.html":
        route = f"/deployment-verification-not-found-{commit_sha[:13]}/"
        status_code = 404
    elif source.endswith("/index.html"):
        route = "/" + source[: -len("index.html")]
        status_code = 200
    else:
        route = "/" + source
        status_code = 200
    validate_route(route)
    request_target = BASE_PATH + route
    return route, request_target, status_code


def validate_route(route: str) -> None:
    if not route.startswith("/") or route.startswith("//"):
        raise UsageError(f"invalid receipt route: {route}")
    if "\x00" in route or "\\" in route or "?" in route or "#" in route or "%" in route:
        raise UsageError(f"unsafe receipt route: {route}")
    if any(ord(char) < 0x20 or ord(char) > 0x7E for char in route):
        raise UsageError(f"receipt route must be printable ASCII: {route}")
    parts = route.split("/")
    if any(part in {".", ".."} for part in parts):
        raise UsageError(f"unsafe receipt route components: {route}")


def compute_site_ordered_sha256(files: list[FileRecord]) -> str:
    digest = hashlib.sha256()
    for record in files:
        digest.update(record.source.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(record.sha256_raw)
        digest.update(b"\n")
    return digest.hexdigest()


def validate_receipt(
    receipt: dict[str, Any],
    subject: LocalSubject,
    commit_sha: str,
    run_id: int,
    run_attempt: int,
    require_result: str,
) -> None:
    failures: list[str] = []
    check_exact_keys(receipt, ROOT_KEYS, "receipt", failures)
    if failures:
        raise ValidationError("; ".join(failures))

    expect_equal(receipt.get("kind"), KIND, "kind", failures)
    expect_equal(receipt.get("schema_version"), SCHEMA_VERSION, "schema_version", failures)
    result = receipt.get("result")
    if not isinstance(result, str) or result not in {"PASS", "FAIL"}:
        failures.append("result must be exact PASS or FAIL")
    elif result != require_result:
        failures.append("result does not match --require-result")

    validate_authority(receipt.get("authority"), failures)
    validate_policy(receipt.get("policy"), failures)
    validate_subject(receipt.get("subject"), subject, commit_sha, failures)
    validate_workflow(receipt.get("workflow"), run_id, run_attempt, failures)

    observation = receipt.get("observation")
    accepted_cohort_sha: str | None = None
    if isinstance(observation, dict):
        accepted_cohort_sha = validate_observation(
            observation,
            result,
            subject.accepted_routes,
            failures,
        )
    else:
        failures.append("observation is not an object")

    if failures:
        raise ValidationError("; ".join(failures))

    if result == "PASS":
        expected_comparison = compute_comparison_sha256(receipt)
        if receipt.get("comparison_sha256") != expected_comparison:
            failures.append("comparison_sha256 does not match recomputed identity")
        if accepted_cohort_sha is None:
            failures.append("PASS receipt has no accepted cohort digest")
    elif result == "FAIL":
        if receipt.get("comparison_sha256") is not None:
            failures.append("FAIL receipt comparison_sha256 must be null")
    else:
        if not (receipt.get("comparison_sha256") is None or is_hex64(receipt.get("comparison_sha256"))):
            failures.append("comparison_sha256 must be 64 lowercase hex or null")

    if failures:
        raise ValidationError("; ".join(failures))


def validate_authority(value: Any, failures: list[str]) -> None:
    if not isinstance(value, dict):
        failures.append("authority is not an object")
        return
    check_exact_keys(value, AUTHORITY_FLAGS, "authority", failures)
    for key in AUTHORITY_FLAGS:
        if value.get(key) is not False:
            failures.append(f"authority.{key} must be literal false")


def validate_policy(value: Any, failures: list[str]) -> None:
    if not isinstance(value, dict):
        failures.append("policy is not an object")
        return
    check_exact_keys(value, set(POLICY), "policy", failures)
    for key, expected in POLICY.items():
        actual = value.get(key)
        if isinstance(expected, int):
            if not is_exact_int(actual) or actual != expected:
                failures.append(f"policy.{key} must be exact integer {expected}")
        elif actual != expected:
            failures.append(f"policy.{key} has the wrong value")


def validate_subject(
    value: Any,
    local: LocalSubject,
    commit_sha: str,
    failures: list[str],
) -> None:
    if not isinstance(value, dict):
        failures.append("subject is not an object")
        return
    check_exact_keys(value, SUBJECT_KEYS, "subject", failures)
    expected = {
        "base_url": BASE_URL,
        "commit_sha": commit_sha,
        "excluded_reader_control": ".nojekyll",
        "generated_file_count": len(local.files),
        "repository": REPOSITORY,
        "route_count": len(local.route_subject),
        "route_manifest_sha256": local.route_manifest_sha256,
        "site_ordered_sha256": local.site_ordered_sha256,
    }
    for key, expected_value in expected.items():
        actual = value.get(key)
        if isinstance(expected_value, int):
            if not is_exact_int(actual) or actual != expected_value:
                failures.append(f"subject.{key} must equal {expected_value}")
        elif actual != expected_value:
            failures.append(f"subject.{key} does not match local subject")
    verifier_hash = value.get("verifier_source_sha256")
    if not is_hex64(verifier_hash):
        failures.append("subject.verifier_source_sha256 must be 64 lowercase hex")
    elif local.verifier_source_sha256 is None:
        failures.append("online verifier source is missing from repository")
    elif verifier_hash != local.verifier_source_sha256:
        failures.append("subject.verifier_source_sha256 does not match local source")


def validate_workflow(
    value: Any, run_id: int, run_attempt: int, failures: list[str]
) -> None:
    if not isinstance(value, dict):
        failures.append("workflow is not an object")
        return
    check_exact_keys(value, WORKFLOW_KEYS, "workflow", failures)
    if not is_exact_int(value.get("run_id")) or value.get("run_id") != run_id:
        failures.append("workflow.run_id does not match CLI")
    if (
        not is_exact_int(value.get("run_attempt"))
        or value.get("run_attempt") != run_attempt
    ):
        failures.append("workflow.run_attempt does not match CLI")


def validate_observation(
    value: dict[str, Any],
    result: Any,
    expected_routes: list[dict[str, Any]],
    failures: list[str],
) -> str | None:
    check_exact_keys(value, OBSERVATION_KEYS, "observation", failures)
    attempted = value.get("attempted_cohorts")
    attempts = value.get("attempts")
    consecutive = value.get("consecutive_matching_cohorts")
    receipt_failures = value.get("failures")
    routes = value.get("routes")
    accepted_cohort = value.get("accepted_cohort_sha256")

    if not is_exact_int(attempted) or attempted < 1 or attempted > MAX_COHORTS:
        failures.append("observation.attempted_cohorts is out of range")
    if not is_exact_int(consecutive) or consecutive < 0 or consecutive > MAX_COHORTS:
        failures.append("observation.consecutive_matching_cohorts is out of range")
    if not isinstance(attempts, list):
        failures.append("observation.attempts is not an array")
        attempts = []
    if not isinstance(receipt_failures, list):
        failures.append("observation.failures is not an array")
        receipt_failures = []
    if not isinstance(routes, list):
        failures.append("observation.routes is not an array")
        routes = []
    if is_exact_int(attempted) and isinstance(attempts, list) and attempted != len(attempts):
        failures.append("observation.attempted_cohorts does not equal attempts length")

    expected_cohort = sha256_hex(canonical_compact(expected_routes))
    route_paths = {route["path"]: route for route in expected_routes}
    route_indexes = {
        route["path"]: index for index, route in enumerate(expected_routes)
    }
    first_route_path = expected_routes[0]["path"] if expected_routes else None

    validated_attempts = validate_attempts(
        attempts,
        expected_cohort,
        len(expected_routes),
        route_paths,
        route_indexes,
        first_route_path,
        failures,
    )

    if result == "PASS":
        if accepted_cohort != expected_cohort:
            failures.append("PASS accepted_cohort_sha256 does not match local routes")
        if routes != expected_routes:
            failures.append("PASS observation.routes do not match local accepted routes")
        if sha256_hex(canonical_compact(routes)) != expected_cohort:
            failures.append("PASS observation.routes digest mismatch")
        if receipt_failures != []:
            failures.append("PASS observation.failures must be empty")
        if consecutive != STABLE_COHORTS_REQUIRED:
            failures.append("PASS requires exactly two consecutive matching cohorts")
        if len(validated_attempts) < STABLE_COHORTS_REQUIRED:
            failures.append("PASS requires at least two attempt summaries")
        else:
            tail = validated_attempts[-STABLE_COHORTS_REQUIRED:]
            if not all(item.get("result") == "MATCH" for item in tail):
                failures.append("PASS attempts must end in two MATCH summaries")
            if not all(
                item.get("cohort_sha256") == expected_cohort for item in tail
            ):
                failures.append("PASS tail MATCH summaries must share accepted digest")
            actual_consecutive = 0
            for attempt in reversed(validated_attempts):
                if attempt.get("result") != "MATCH":
                    break
                actual_consecutive += 1
            if actual_consecutive != STABLE_COHORTS_REQUIRED:
                failures.append("PASS must terminate after exactly two MATCH summaries")
        return expected_cohort

    if result == "FAIL":
        if accepted_cohort is not None:
            failures.append("FAIL accepted_cohort_sha256 must be null")
        if routes != []:
            failures.append("FAIL observation.routes must be empty")
        if not is_exact_int(consecutive) or consecutive not in {0, 1}:
            failures.append("FAIL consecutive_matching_cohorts must be zero or one")
        if len(receipt_failures) != 1:
            failures.append("FAIL observation.failures must contain exactly one failure")
        else:
            terminal_failure = receipt_failures[0]
            validate_failure(
                terminal_failure,
                route_paths,
                first_route_path,
                failures,
                "observation.failures[0]",
            )
            if validated_attempts:
                last = validated_attempts[-1]
                terminal_code = (
                    terminal_failure.get("code")
                    if isinstance(terminal_failure, dict)
                    else None
                )
                if last.get("result") == "FAIL":
                    if (
                        last.get("failure") != terminal_failure
                        and terminal_code != "deadline_exhausted"
                    ):
                        failures.append(
                            "terminal observation failure must equal final attempt failure"
                        )
                else:
                    if not isinstance(terminal_code, str) or terminal_code not in {
                        "deadline_exhausted",
                        "cohort_attempts_exhausted",
                    }:
                        failures.append(
                            "post-attempt terminal failures are limited to deadline or cohort exhaustion"
                        )
                if terminal_code == "cohort_attempts_exhausted" and (
                    len(validated_attempts) != MAX_COHORTS
                    or last.get("result") != "MATCH"
                ):
                    failures.append(
                        "cohort exhaustion requires the final allowed attempt to MATCH"
                    )
        actual_consecutive = 0
        for attempt in reversed(validated_attempts):
            if attempt.get("result") != "MATCH":
                break
            actual_consecutive += 1
        if is_exact_int(consecutive) and consecutive != actual_consecutive:
            failures.append(
                "FAIL consecutive_matching_cohorts does not match attempt suffix"
            )
        return None

    if not (accepted_cohort is None or is_hex64(accepted_cohort)):
        failures.append("accepted_cohort_sha256 must be 64 lowercase hex or null")
    return None


def validate_attempts(
    attempts: list[Any],
    expected_cohort: str,
    route_count: int,
    route_paths: dict[str, dict[str, Any]],
    route_indexes: dict[str, int],
    first_route_path: str | None,
    failures: list[str],
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for index, attempt in enumerate(attempts, start=1):
        label = f"observation.attempts[{index - 1}]"
        if not isinstance(attempt, dict):
            failures.append(f"{label} is not an object")
            continue
        has_exact_keys = set(attempt) == ATTEMPT_KEYS
        check_exact_keys(attempt, ATTEMPT_KEYS, label, failures)
        if not is_exact_int(attempt.get("attempt")) or attempt.get("attempt") != index:
            failures.append(f"{label}.attempt is not contiguous from 1")
        result = attempt.get("result")
        completed_routes = attempt.get("completed_routes")
        if result == "MATCH":
            if attempt.get("cohort_sha256") != expected_cohort:
                failures.append(f"{label}.cohort_sha256 does not match accepted route digest")
            if not is_exact_int(completed_routes) or completed_routes != route_count:
                failures.append(f"{label}.completed_routes must equal route count")
            if attempt.get("failure") is not None:
                failures.append(f"{label}.failure must be null for MATCH")
        elif result == "FAIL":
            if attempt.get("cohort_sha256") is not None:
                failures.append(f"{label}.cohort_sha256 must be null for FAIL")
            if (
                not is_exact_int(completed_routes)
                or completed_routes < 0
                or completed_routes >= route_count
            ):
                failures.append(f"{label}.completed_routes is out of failure range")
            attempt_failure = attempt.get("failure")
            validate_failure(
                attempt_failure,
                route_paths,
                first_route_path,
                failures,
                f"{label}.failure",
            )
            if isinstance(attempt_failure, dict):
                failure_code = attempt_failure.get("code")
                failure_path = attempt_failure.get("path")
                if failure_code == "cohort_attempts_exhausted":
                    failures.append(
                        f"{label}.failure cohort_attempts_exhausted is terminal-only"
                    )
                elif isinstance(failure_code, str) and failure_code in {
                    "dns_policy_refused",
                    "resolver_error",
                    "resolver_timeout",
                }:
                    if completed_routes != 0:
                        failures.append(
                            f"{label}.completed_routes must be zero for resolver failure"
                        )
                elif (
                    isinstance(failure_code, str)
                    and failure_code not in POLICY_WIDE_FAILURE_CODES
                    and isinstance(failure_path, str)
                    and failure_path in route_indexes
                    and completed_routes != route_indexes[failure_path]
                ):
                    failures.append(
                        f"{label}.completed_routes does not match failure path"
                    )
        else:
            failures.append(f"{label}.result must be MATCH or FAIL")
        if has_exact_keys:
            validated.append(attempt)
    return validated


def validate_failure(
    value: Any,
    route_paths: dict[str, dict[str, Any]],
    first_route_path: str | None,
    failures: list[str],
    label: str,
) -> None:
    if not isinstance(value, dict):
        failures.append(f"{label} is not an object")
        return
    check_exact_keys(value, FAILURE_KEYS, label, failures)
    code = value.get("code")
    code_is_allowed = isinstance(code, str) and code in FAILURE_CODES
    if not code_is_allowed:
        failures.append(f"{label}.code is not allowed")
    path = value.get("path")
    path_is_route = isinstance(path, str) and path in route_paths
    if not path_is_route:
        failures.append(f"{label}.path is not an accepted receipt route")
        expected_limit = MAX_FILE_BYTES + 1
    else:
        expected_limit = route_paths[path]["bytes"] + 1
    if (
        code_is_allowed
        and code in POLICY_WIDE_FAILURE_CODES
        and path != first_route_path
    ):
        failures.append(f"{label}.path must be the first route for policy-wide failure")
    status_value = value.get("status")
    if status_value is not None and (
        not is_exact_int(status_value) or status_value < 100 or status_value > 599
    ):
        failures.append(f"{label}.status must be null or HTTP status 100..599")
    observed_bytes = value.get("observed_bytes")
    if observed_bytes is not None and (
        not is_exact_int(observed_bytes)
        or observed_bytes < 0
        or observed_bytes > expected_limit
    ):
        failures.append(f"{label}.observed_bytes is out of range")
    observed_sha = value.get("observed_sha256")
    if observed_sha is not None and not is_hex64(observed_sha):
        failures.append(f"{label}.observed_sha256 must be 64 lowercase hex or null")
    if observed_sha is not None and observed_bytes is None:
        failures.append(f"{label}.observed_sha256 requires observed_bytes")
    expected_bytes = route_paths[path]["bytes"] if path_is_route else None
    expected_status = route_paths[path]["status"] if path_is_route else None
    if (
        code_is_allowed
        and code in NULL_STATUS_FAILURE_CODES
        and status_value is not None
    ):
        failures.append(f"{label}.status must be null for pre-response failure")
    if (
        code_is_allowed
        and code in EXPECTED_STATUS_FAILURE_CODES
        and status_value != expected_status
    ):
        failures.append(f"{label}.status must equal the expected route status")
    if code == "http_status_mismatch" and (
        not is_exact_int(status_value) or status_value == expected_status
    ):
        failures.append(
            f"{label}.status must be a parsed status different from the expected route status"
        )
    if code_is_allowed and code in NULL_BODY_OBSERVATION_FAILURE_CODES and (
        observed_bytes is not None or observed_sha is not None
    ):
        failures.append(f"{label} body observations must be null for this failure code")
    if code == "body_too_large" and (
        expected_bytes is None
        or observed_bytes != expected_bytes + 1
        or observed_sha is not None
    ):
        failures.append(f"{label} body_too_large observations are inconsistent")
    if code == "byte_count_mismatch" and (
        expected_bytes is None
        or not is_exact_int(observed_bytes)
        or observed_bytes >= expected_bytes
        or observed_sha is None
    ):
        failures.append(f"{label} byte_count_mismatch observations are inconsistent")
    if code == "byte_mismatch" and (
        expected_bytes is None
        or observed_bytes != expected_bytes
        or observed_sha is None
    ):
        failures.append(f"{label} byte_mismatch observations are inconsistent")


def compute_comparison_sha256(receipt: dict[str, Any]) -> str:
    observation = expect_object(receipt.get("observation"), "observation")
    comparison_subject = {
        "authority": receipt["authority"],
        "kind": receipt["kind"],
        "observation": {
            "accepted_cohort_sha256": observation["accepted_cohort_sha256"],
            "consecutive_matching_cohorts": observation[
                "consecutive_matching_cohorts"
            ],
            "routes": observation["routes"],
        },
        "policy": receipt["policy"],
        "result": receipt["result"],
        "schema_version": receipt["schema_version"],
        "subject": receipt["subject"],
    }
    return sha256_hex(canonical_compact(comparison_subject))


def check_exact_keys(
    value: dict[str, Any], expected: set[str], label: str, failures: list[str]
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        failures.append(f"{label} keys differ; missing={missing} extra={extra}")


def expect_equal(actual: Any, expected: Any, label: str, failures: list[str]) -> None:
    if isinstance(expected, int):
        if not is_exact_int(actual) or actual != expected:
            failures.append(f"{label} must be integer {expected}")
    elif actual != expected:
        failures.append(f"{label} has the wrong value")


def expect_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{label} is not an object")
    return value


def expect_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValidationError(f"{label} is not an array")
    return value


def is_exact_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_hex64(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_arguments(sys.argv[1:] if argv is None else argv)
        repository_root = require_existing_directory(
            args.repository_root, "repository root"
        )
        site = require_existing_directory(args.site, "site")
        if (
            site == repository_root
            or is_relative_to(site, repository_root)
            or is_relative_to(repository_root, site)
        ):
            raise UsageError("site and repository roots must not overlap")

        local_site_failures = validate_public_site.validate(site)
        if local_site_failures:
            raise UsageError("generated site failed strict public-site validation")

        receipt_path = None
        artifact_path = None
        if args.receipt is not None:
            receipt_path = require_existing_file(args.receipt, "receipt")
            require_outside(
                receipt_path,
                ((repository_root, "repository root"), (site, "site")),
                "receipt",
            )
        if args.artifact_zip is not None:
            artifact_path = require_existing_file(args.artifact_zip, "artifact ZIP")
            require_outside(
                artifact_path,
                ((repository_root, "repository root"), (site, "site")),
                "artifact ZIP",
            )

        local_subject = build_local_subject(repository_root, site, args.commit_sha)
        receipt_bytes = read_receipt_bytes(receipt_path, artifact_path)
        receipt = parse_receipt(receipt_bytes)
        validate_receipt(
            receipt,
            local_subject,
            args.commit_sha,
            args.workflow_run_id_int,
            args.workflow_run_attempt_int,
            args.require_result,
        )
        success_json(receipt_bytes, receipt)
        return 0
    except ValidationError as exc:
        fail_json(sys.stderr, "INVALID", [str(exc)])
        return 1
    except UsageError as exc:
        fail_json(sys.stderr, "ERROR", [str(exc)])
        return 2
    except OSError as exc:
        fail_json(sys.stderr, "ERROR", [f"I/O error: {exc}"])
        return 2
    except Exception:
        fail_json(sys.stderr, "ERROR", ["unexpected internal validation error"])
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
