#!/usr/bin/env python3
"""Create a fail-closed public GitHub Pages deployment receipt.

The public CLI intentionally exposes no policy knobs. Tests may exercise the
module-internal seams by calling ``verify_public_deployment`` with fake
resolver, transport, clock, and sleeper callables.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import hashlib
import hmac
import http.client
import io
import ipaddress
import json
import os
from pathlib import Path
import queue
import re
import socket
import ssl
import stat
import sys
import threading
import time
from typing import Callable, NamedTuple, Sequence
from urllib.parse import urlsplit

try:
    import validate_public_site
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import validate_public_site


SCHEMA_VERSION = 1
KIND = "fast-mlx-public-deployment-receipt"
REPOSITORY = "bitworks-io/fast-mlx"
ACCEPTED_BASE_URL = "https://bitworks-io.github.io/fast-mlx/"
ACCEPTED_HOST = "bitworks-io.github.io"
BASE_PATH = "/fast-mlx"
HTTPS_PORT = 443
COMMIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
POSITIVE_DECIMAL_RE = re.compile(r"[1-9][0-9]*")
CONTENT_LENGTH_RE = re.compile(r"0|[1-9][0-9]*")

MAX_COHORTS = 24
MAX_ROUTES = 256
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_EXPECTED_BYTES = 64 * 1024 * 1024
MAX_RESOLVED_ADDRESSES = 16
MAX_ABANDONED_RESOLVER_THREADS = 24
MAX_OBSERVATION_SECONDS = 600
RESOLVER_TIMEOUT_SECONDS = 10
PER_REQUEST_TIMEOUT_SECONDS = 10
RETRY_DELAY_SECONDS = 25
STABLE_COHORTS_REQUIRED = 2

POLICY: dict[str, object] = {
    "accepted_base_url": ACCEPTED_BASE_URL,
    "max_abandoned_resolver_threads": MAX_ABANDONED_RESOLVER_THREADS,
    "max_cohorts": MAX_COHORTS,
    "max_file_bytes": MAX_FILE_BYTES,
    "max_observation_seconds": MAX_OBSERVATION_SECONDS,
    "max_resolved_addresses": MAX_RESOLVED_ADDRESSES,
    "max_routes": MAX_ROUTES,
    "max_total_expected_bytes": MAX_TOTAL_EXPECTED_BYTES,
    "per_request_timeout_seconds": PER_REQUEST_TIMEOUT_SECONDS,
    "resolver_timeout_seconds": RESOLVER_TIMEOUT_SECONDS,
    "retry_delay_seconds": RETRY_DELAY_SECONDS,
    "stable_cohorts_required": STABLE_COHORTS_REQUIRED,
}

AUTHORITY: dict[str, bool] = {
    "acquisition_authority": False,
    "automatic_evidence_intake": False,
    "automatic_publication_authority": False,
    "containment_claim": False,
    "launchability_claim": False,
    "model_authority": False,
    "performance_claim": False,
    "positive_admission": False,
    "rollback_authority": False,
    "runtime_authority": False,
}

MIME_BY_SUFFIX = {
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

ALLOWED_FAILURE_CODES = {
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

class Refusal(Exception):
    """A local, reject-before-network policy refusal."""


class InvocationError(Exception):
    """Invalid public CLI usage."""


class RouteSubject(NamedTuple):
    source: str
    path: str
    request_target: str
    expected_status: int
    expected_content_type: str
    expected_bytes: int
    expected_sha256: str
    expected_body: bytes

    def manifest_object(self) -> dict[str, object]:
        return {
            "source": self.source,
            "path": self.path,
            "request_target": self.request_target,
            "expected_status": self.expected_status,
            "expected_content_type": self.expected_content_type,
            "expected_bytes": self.expected_bytes,
            "expected_sha256": self.expected_sha256,
        }

    def accepted_object(self) -> dict[str, object]:
        return {
            "source": self.source,
            "path": self.path,
            "request_target": self.request_target,
            "status": self.expected_status,
            "content_type": self.expected_content_type,
            "bytes": self.expected_bytes,
            "sha256": self.expected_sha256,
        }


class Inventory(NamedTuple):
    generated_file_count: int
    route_count: int
    site_ordered_sha256: str
    route_manifest_sha256: str
    routes: tuple[RouteSubject, ...]


class LocalInputs(NamedTuple):
    repository_root: Path
    site: Path
    output: Path
    commit_sha: str
    workflow_run_id: int
    workflow_run_attempt: int


class RouteFetch(NamedTuple):
    record: dict[str, object] | None
    failure: dict[str, object] | None


Resolver = Callable[[str, int], object]
Transport = Callable[
    [RouteSubject, tuple[str, ...], float, Callable[[], float]], RouteFetch
]


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def receipt_json_bytes(value: object) -> bytes:
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


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_failure(
    code: str,
    path: str,
    *,
    status: int | None = None,
    observed_bytes: int | None = None,
    observed_sha256: str | None = None,
) -> dict[str, object]:
    if code not in ALLOWED_FAILURE_CODES:
        code = "internal_error"
    return {
        "code": code,
        "observed_bytes": observed_bytes,
        "observed_sha256": observed_sha256,
        "path": path,
        "status": status,
    }


def _contains_or_equals(root: Path, candidate: Path) -> bool:
    return candidate == root or root in candidate.parents


def _resolve_existing_directory(raw: str, label: str) -> Path:
    path = Path(raw)
    if path.is_symlink() or not path.is_dir():
        raise Refusal(f"{label} must be an existing non-symlink directory")
    return path.resolve(strict=True)


def parse_fixed_cli(argv: Sequence[str]) -> argparse.Namespace:
    required_once = (
        "--repository-root",
        "--site",
        "--deployment-url",
        "--commit-sha",
        "--workflow-run-id",
        "--workflow-run-attempt",
        "--output",
    )
    counts = {name: 0 for name in required_once}
    for token in argv:
        for name in required_once:
            if token == name or token.startswith(name + "="):
                counts[name] += 1
    if any(count != 1 for count in counts.values()):
        raise InvocationError("each required CLI argument must appear exactly once")

    parser = argparse.ArgumentParser(
        description="Verify the exact fast-mlx public GitHub Pages deployment.",
        allow_abbrev=False,
    )
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--deployment-url", required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--workflow-run-id", required=True)
    parser.add_argument("--workflow-run-attempt", required=True)
    parser.add_argument("--output", required=True)
    try:
        return parser.parse_args(list(argv))
    except SystemExit as exc:
        raise InvocationError("invalid CLI arguments") from exc


def validate_local_inputs(arguments: argparse.Namespace) -> LocalInputs:
    repository_root = _resolve_existing_directory(
        arguments.repository_root, "repository root"
    )

    site_raw = Path(arguments.site)
    if site_raw.is_symlink() or not site_raw.is_dir():
        raise Refusal("site must be an existing non-symlink directory")
    site = site_raw.resolve(strict=True)
    if _contains_or_equals(repository_root, site) or _contains_or_equals(
        site, repository_root
    ):
        raise Refusal("site and repository roots must not overlap")

    output_raw = Path(arguments.output)
    if output_raw.exists() or output_raw.is_symlink():
        raise Refusal("output must be fresh")
    output_parent_raw = output_raw.parent
    if output_parent_raw.is_symlink() or not output_parent_raw.is_dir():
        raise Refusal("output parent must be an existing non-symlink directory")
    output_parent = output_parent_raw.resolve(strict=True)
    output = output_parent / output_raw.name
    output_resolved = output.resolve(strict=False)
    if _contains_or_equals(repository_root, output_resolved) or _contains_or_equals(
        site, output_resolved
    ):
        raise Refusal("output must be outside the repository and site roots")

    commit_sha = arguments.commit_sha
    if not isinstance(commit_sha, str) or not COMMIT_SHA_RE.fullmatch(commit_sha):
        raise Refusal("commit SHA must be exactly 40 lowercase hexadecimal characters")

    run_id_raw = arguments.workflow_run_id
    run_attempt_raw = arguments.workflow_run_attempt
    if not POSITIVE_DECIMAL_RE.fullmatch(run_id_raw):
        raise Refusal("workflow run ID must be a positive decimal integer")
    if not POSITIVE_DECIMAL_RE.fullmatch(run_attempt_raw):
        raise Refusal("workflow run attempt must be a positive decimal integer")
    workflow_run_id = int(run_id_raw)
    workflow_run_attempt = int(run_attempt_raw)

    deployment_url = arguments.deployment_url
    parsed = urlsplit(deployment_url)
    if (
        deployment_url != ACCEPTED_BASE_URL
        or parsed.scheme != "https"
        or parsed.hostname != ACCEPTED_HOST
        or parsed.netloc != ACCEPTED_HOST
        or parsed.port is not None
        or parsed.path != "/fast-mlx/"
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise Refusal("deployment URL is outside the fixed accepted authority")

    local_failures = validate_public_site.validate(site)
    if local_failures:
        raise Refusal("generated site failed local public-site validation")

    return LocalInputs(
        repository_root=repository_root,
        site=site,
        output=output,
        commit_sha=commit_sha,
        workflow_run_id=workflow_run_id,
        workflow_run_attempt=workflow_run_attempt,
    )


def _validate_relative_source(source: str) -> None:
    if source.startswith("/") or source.endswith("/"):
        raise Refusal("generated source path is not site-relative")
    if "\\" in source or "\x00" in source or "%" in source:
        raise Refusal("generated source path contains a refused byte")
    if "?" in source or "#" in source:
        raise Refusal("generated source path contains a URL delimiter")
    for character in source:
        ordinal = ord(character)
        if ordinal > 0x7F or ordinal < 0x20 or ordinal == 0x7F:
            raise Refusal("generated source path contains a control or non-ASCII byte")
    components = source.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise Refusal("generated source path contains an unsafe component")


def _file_sources(site: Path) -> tuple[str, ...]:
    stack = [site]
    sources: list[str] = []
    while stack:
        directory = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise Refusal("cannot inspect generated site") from exc
        for entry in entries:
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as exc:
                raise Refusal("cannot inspect generated site entry") from exc
            if stat.S_ISLNK(mode):
                raise Refusal("symlink is forbidden in generated site")
            path = Path(entry.path)
            if stat.S_ISDIR(mode):
                stack.append(path)
            elif stat.S_ISREG(mode):
                source = path.relative_to(site).as_posix()
                _validate_relative_source(source)
                sources.append(source)
            else:
                raise Refusal("special file is forbidden in generated site")
    return tuple(sorted(sources, key=lambda value: value.encode("utf-8")))


def _route_for_source(source: str, commit_sha: str) -> tuple[str, int] | None:
    if source == ".nojekyll":
        return None
    if source.endswith("/.nojekyll"):
        raise Refusal("nested .nojekyll is forbidden")
    if source == "index.html":
        return "/", 200
    if source.endswith("/index.html"):
        return "/" + source[: -len("index.html")], 200
    if source == "404.html":
        return f"/deployment-verification-not-found-{commit_sha[:13]}/", 404
    return "/" + source, 200


def inventory_site(site: Path, commit_sha: str) -> Inventory:
    sources = _file_sources(site)
    site_hash = hashlib.sha256()
    route_subjects: list[RouteSubject] = []
    seen_routes: set[str] = set()
    total_expected_bytes = 0

    for source in sources:
        path = site / source
        try:
            file_bytes = path.read_bytes()
        except OSError as exc:
            raise Refusal("cannot read generated site file") from exc
        byte_count = len(file_bytes)
        if byte_count > MAX_FILE_BYTES:
            raise Refusal("generated file exceeds fixed byte cap")
        total_expected_bytes += byte_count
        if total_expected_bytes > MAX_TOTAL_EXPECTED_BYTES:
            raise Refusal("generated site exceeds fixed total byte cap")

        digest = hashlib.sha256(file_bytes).digest()
        site_hash.update(source.encode("utf-8"))
        site_hash.update(b"\x00")
        site_hash.update(digest)
        site_hash.update(b"\n")

        route_info = _route_for_source(source, commit_sha)
        if route_info is None:
            continue
        suffix = Path(source).suffix
        expected_content_type = MIME_BY_SUFFIX.get(suffix)
        if expected_content_type is None:
            raise Refusal("generated source uses an unknown published suffix")
        route, expected_status = route_info
        if not route.startswith("/") or route.startswith("//"):
            raise Refusal("generated route is not exactly rooted")
        if route in seen_routes:
            raise Refusal("duplicate generated route")
        seen_routes.add(route)
        request_target = BASE_PATH + route
        route_subjects.append(
            RouteSubject(
                source=source,
                path=route,
                request_target=request_target,
                expected_status=expected_status,
                expected_content_type=expected_content_type,
                expected_bytes=byte_count,
                expected_sha256=digest.hex(),
                expected_body=file_bytes,
            )
        )
        if len(route_subjects) > MAX_ROUTES:
            raise Refusal("generated site exceeds fixed route cap")

    if not route_subjects:
        raise Refusal("generated site has no reader routes")
    route_manifest = [route.manifest_object() for route in route_subjects]
    return Inventory(
        generated_file_count=len(sources),
        route_count=len(route_subjects),
        site_ordered_sha256=site_hash.hexdigest(),
        route_manifest_sha256=sha256_hex(canonical_json_bytes(route_manifest)),
        routes=tuple(route_subjects),
    )


def verifier_source_sha256() -> str:
    return sha256_hex(Path(__file__).read_bytes())


def build_subject(
    local: LocalInputs, inventory: Inventory, verifier_sha256: str
) -> dict[str, object]:
    return {
        "base_url": ACCEPTED_BASE_URL,
        "commit_sha": local.commit_sha,
        "excluded_reader_control": ".nojekyll",
        "generated_file_count": inventory.generated_file_count,
        "repository": REPOSITORY,
        "route_count": inventory.route_count,
        "route_manifest_sha256": inventory.route_manifest_sha256,
        "site_ordered_sha256": inventory.site_ordered_sha256,
        "verifier_source_sha256": verifier_sha256,
    }


def _normalize_resolver_result(raw: object) -> tuple[str, ...]:
    addresses: list[str] = []
    if not isinstance(raw, Iterable):
        raise Refusal("resolver returned a malformed result")
    for item in raw:
        address: str | None = None
        if isinstance(item, str):
            address = item
        elif isinstance(item, tuple):
            if len(item) >= 5 and isinstance(item[4], tuple) and item[4]:
                candidate = item[4][0]
                if isinstance(candidate, str):
                    address = candidate
            elif item and isinstance(item[0], str):
                address = item[0]
        if address is None:
            raise Refusal("resolver returned a malformed address")
        addresses.append(address)
    try:
        return tuple(sorted(set(addresses), key=lambda value: value.encode("ascii")))
    except UnicodeEncodeError as exc:
        raise Refusal("resolver returned a malformed address") from exc


def is_allowed_global_unicast(address: str) -> bool:
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(parsed, ipaddress.IPv6Address):
        if parsed.ipv4_mapped or parsed.sixtofour or parsed.teredo:
            return False
    return (
        parsed.is_global
        and not parsed.is_multicast
        and not parsed.is_unspecified
        and not parsed.is_loopback
        and not parsed.is_link_local
        and not parsed.is_private
        and not parsed.is_reserved
    )


def production_resolver(host: str, port: int) -> object:
    return socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)


def resolve_cohort_addresses(
    *,
    resolver: Resolver,
    deadline: float,
    clock: Callable[[], float],
) -> tuple[tuple[str, ...] | None, str | None]:
    remaining = deadline - clock()
    if remaining <= 0:
        return None, "deadline_exhausted"
    timeout = min(float(RESOLVER_TIMEOUT_SECONDS), remaining)
    result_queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            result_queue.put(("ok", resolver(ACCEPTED_HOST, HTTPS_PORT)), block=False)
        except BaseException:
            try:
                result_queue.put(("error", None), block=False)
            except BaseException:
                pass

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        status_name, payload = result_queue.get(timeout=timeout)
    except queue.Empty:
        return None, "resolver_timeout"
    if status_name != "ok":
        return None, "resolver_error"
    try:
        addresses = _normalize_resolver_result(payload)
    except Refusal:
        return None, "dns_policy_refused"
    if not addresses or len(addresses) > MAX_RESOLVED_ADDRESSES:
        return None, "dns_policy_refused"
    if any(not is_allowed_global_unicast(address) for address in addresses):
        return None, "dns_policy_refused"
    return addresses, None


class _DeadlineReader(io.RawIOBase):
    def __init__(
        self,
        sock: socket.socket,
        *,
        deadline: float,
        clock: Callable[[], float],
    ) -> None:
        super().__init__()
        self._socket = sock
        self._deadline = deadline
        self._clock = clock

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray | memoryview) -> int:
        remaining = self._deadline - self._clock()
        if remaining <= 0:
            raise TimeoutError("route deadline exhausted")
        self._socket.settimeout(remaining)
        return self._socket.recv_into(buffer)


class _DeadlineSocket:
    def __init__(
        self,
        sock: socket.socket,
        *,
        deadline: float,
        clock: Callable[[], float],
    ) -> None:
        self._socket = sock
        self._deadline = deadline
        self._clock = clock

    def _remaining(self) -> float:
        remaining = self._deadline - self._clock()
        if remaining <= 0:
            raise TimeoutError("route deadline exhausted")
        return remaining

    def settimeout(self, timeout: float | None) -> None:
        remaining = self._remaining()
        self._socket.settimeout(remaining if timeout is None else min(timeout, remaining))

    def sendall(self, data: bytes, flags: int = 0) -> None:
        self._socket.settimeout(self._remaining())
        self._socket.sendall(data, flags)

    def makefile(
        self,
        mode: str,
        buffering: int | None = None,
        *_args: object,
        **_kwargs: object,
    ) -> io.BufferedReader:
        if mode != "rb":
            raise ValueError("deadline socket supports binary reads only")
        raw = _DeadlineReader(
            self._socket, deadline=self._deadline, clock=self._clock
        )
        buffer_size = (
            buffering
            if isinstance(buffering, int) and buffering > 0
            else io.DEFAULT_BUFFER_SIZE
        )
        return io.BufferedReader(raw, buffer_size=buffer_size)

    def close(self) -> None:
        self._socket.close()


class DirectHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        address: str,
        *,
        timeout: float,
        context: ssl.SSLContext,
        deadline: float,
        clock: Callable[[], float],
    ) -> None:
        super().__init__(
            ACCEPTED_HOST,
            port=HTTPS_PORT,
            timeout=timeout,
            context=context,
        )
        self._direct_address = address
        self._deadline = deadline
        self._clock = clock

    def connect(self) -> None:
        parsed_address = ipaddress.ip_address(self._direct_address)
        family = socket.AF_INET6 if parsed_address.version == 6 else socket.AF_INET
        raw_socket = socket.socket(family, socket.SOCK_STREAM)
        tls_socket: socket.socket | None = None
        try:
            remaining = min(float(self.timeout), self._deadline - self._clock())
            if remaining <= 0:
                raise TimeoutError("route deadline exhausted before connect")
            raw_socket.settimeout(remaining)
            if family == socket.AF_INET6:
                raw_socket.connect((str(parsed_address), HTTPS_PORT, 0, 0))
            else:
                raw_socket.connect((str(parsed_address), HTTPS_PORT))
            remaining = min(float(self.timeout), self._deadline - self._clock())
            if remaining <= 0:
                raise TimeoutError("route deadline exhausted before TLS")
            raw_socket.settimeout(remaining)
            tls_socket = self._context.wrap_socket(
                raw_socket, server_hostname=ACCEPTED_HOST
            )
            remaining = min(float(self.timeout), self._deadline - self._clock())
            if remaining <= 0:
                raise TimeoutError("route deadline exhausted after TLS")
            tls_socket.settimeout(remaining)
            self.sock = _DeadlineSocket(
                tls_socket, deadline=self._deadline, clock=self._clock
            )
        except BaseException:
            (tls_socket if tls_socket is not None else raw_socket).close()
            raise


def _remaining_timeout(deadline: float, clock: Callable[[], float]) -> float | None:
    remaining = min(float(PER_REQUEST_TIMEOUT_SECONDS), deadline - clock())
    if remaining <= 0:
        return None
    return remaining


def _read_bounded_body(
    response: http.client.HTTPResponse,
    route: RouteSubject,
) -> tuple[bytes | None, dict[str, object] | None]:
    read_limit = route.expected_bytes + 1
    try:
        body = response.read(read_limit)
    except TimeoutError:
        return None, make_failure("read_timeout", route.path, status=response.status)
    except OSError:
        return None, make_failure("read_timeout", route.path, status=response.status)
    except http.client.HTTPException:
        return None, make_failure(
            "response_protocol_error", route.path, status=response.status
        )
    observed_bytes = len(body)
    if observed_bytes > route.expected_bytes:
        return None, make_failure(
            "body_too_large",
            route.path,
            status=response.status,
            observed_bytes=observed_bytes,
        )
    observed_sha256 = sha256_hex(body)
    if observed_bytes != route.expected_bytes:
        return None, make_failure(
            "byte_count_mismatch",
            route.path,
            status=response.status,
            observed_bytes=observed_bytes,
            observed_sha256=observed_sha256,
        )
    if not hmac.compare_digest(observed_sha256, route.expected_sha256):
        return None, make_failure(
            "byte_mismatch",
            route.path,
            status=response.status,
            observed_bytes=observed_bytes,
            observed_sha256=observed_sha256,
        )
    if body != route.expected_body:
        return None, make_failure(
            "byte_mismatch",
            route.path,
            status=response.status,
            observed_bytes=observed_bytes,
            observed_sha256=observed_sha256,
        )
    return body, None


def evaluate_http_response(
    route: RouteSubject,
    response: http.client.HTTPResponse,
) -> RouteFetch:
    status_value = response.status
    if type(status_value) is not int or status_value < 100 or status_value > 599:
        return RouteFetch(
            None,
            make_failure("response_protocol_error", route.path),
        )
    if status_value != route.expected_status:
        return RouteFetch(
            None,
            make_failure(
                "http_status_mismatch", route.path, status=status_value
            ),
        )

    content_types = response.headers.get_all("Content-Type", [])
    if len(content_types) != 1 or content_types[0] != route.expected_content_type:
        return RouteFetch(
            None,
            make_failure("content_type_refused", route.path, status=status_value),
        )

    encodings = response.headers.get_all("Content-Encoding", [])
    if len(encodings) > 1:
        return RouteFetch(
            None,
            make_failure("content_encoding_refused", route.path, status=status_value),
        )
    if len(encodings) == 1:
        token = encodings[0]
        if "," in token or token.strip().lower() != "identity":
            return RouteFetch(
                None,
                make_failure(
                    "content_encoding_refused", route.path, status=status_value
                ),
            )

    if response.headers.get_all("Transfer-Encoding", []):
        return RouteFetch(
            None,
            make_failure("transfer_encoding_refused", route.path, status=status_value),
        )

    lengths = response.headers.get_all("Content-Length", [])
    if (
        len(lengths) != 1
        or len(lengths[0]) > len(str(MAX_FILE_BYTES))
        or not CONTENT_LENGTH_RE.fullmatch(lengths[0])
    ):
        return RouteFetch(
            None,
            make_failure("content_length_refused", route.path, status=status_value),
        )
    try:
        content_length = int(lengths[0])
    except ValueError:
        return RouteFetch(
            None,
            make_failure("content_length_refused", route.path, status=status_value),
        )
    if content_length != route.expected_bytes:
        return RouteFetch(
            None,
            make_failure("content_length_refused", route.path, status=status_value),
        )

    body, failure = _read_bounded_body(response, route)
    if failure is not None:
        return RouteFetch(None, failure)
    assert body is not None
    return RouteFetch(route.accepted_object(), None)


def production_transport(
    route: RouteSubject,
    addresses: tuple[str, ...],
    deadline: float,
    clock: Callable[[], float],
) -> RouteFetch:
    last_code = "connect_error"
    route_deadline = min(deadline, clock() + float(PER_REQUEST_TIMEOUT_SECONDS))
    for address in addresses:
        timeout = _remaining_timeout(route_deadline, clock)
        if timeout is None:
            return RouteFetch(None, make_failure("connect_timeout", route.path))
        connection: DirectHTTPSConnection | None = None
        response: http.client.HTTPResponse | None = None
        try:
            context = ssl.create_default_context()
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            connection = DirectHTTPSConnection(
                address,
                timeout=timeout,
                context=context,
                deadline=route_deadline,
                clock=clock,
            )
            connection.putrequest(
                "GET", route.request_target, skip_accept_encoding=True
            )
            connection.putheader("Accept", "*/*")
            connection.putheader("Accept-Encoding", "identity")
            connection.putheader("User-Agent", "fast-mlx-publication-receipt/1")
            connection.endheaders()
            if connection.sock is not None:
                response_timeout = _remaining_timeout(route_deadline, clock)
                if response_timeout is None:
                    return RouteFetch(
                        None, make_failure("connect_timeout", route.path)
                    )
                connection.sock.settimeout(response_timeout)
            response = connection.getresponse()
            result = evaluate_http_response(route, response)
            if result.failure is None and clock() >= route_deadline:
                result = RouteFetch(
                    None,
                    make_failure(
                        "read_timeout", route.path, status=response.status
                    ),
                )
            cleanup_failure = _close_response_and_connection(response, connection)
            response = None
            connection = None
            if cleanup_failure:
                return RouteFetch(
                    None,
                    make_failure(
                        "response_protocol_error",
                        route.path,
                        status=(
                            int(result.failure["status"])
                            if result.failure
                            and isinstance(result.failure.get("status"), int)
                            else route.expected_status
                        ),
                    ),
                )
            return result
        except TimeoutError:
            last_code = "connect_timeout"
        except socket.timeout:
            last_code = "connect_timeout"
        except ssl.SSLError:
            last_code = "tls_error"
        except (http.client.HTTPException, ValueError):
            last_code = "response_protocol_error"
        except OSError:
            last_code = "connect_error"
        finally:
            if response is not None or connection is not None:
                cleanup_failed = _close_response_and_connection(response, connection)
                if cleanup_failed and last_code == "connect_error":
                    last_code = "response_protocol_error"
    return RouteFetch(None, make_failure(last_code, route.path))


def _close_response_and_connection(
    response: http.client.HTTPResponse | None,
    connection: http.client.HTTPSConnection | None,
) -> bool:
    failed = False
    if response is not None:
        try:
            response.close()
        except Exception:
            failed = True
    if connection is not None:
        try:
            connection.close()
        except Exception:
            failed = True
    return failed


def cohort_sha256(routes: Sequence[dict[str, object]]) -> str:
    return sha256_hex(canonical_json_bytes(list(routes)))


def observe_routes(
    routes: Sequence[RouteSubject],
    *,
    resolver: Resolver = production_resolver,
    transport: Transport = production_transport,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    deadline = clock() + float(MAX_OBSERVATION_SECONDS)
    attempts: list[dict[str, object]] = []
    consecutive = 0
    first_matching_hash: str | None = None
    accepted_routes: list[dict[str, object]] = []
    terminal_failure: dict[str, object] | None = None
    first_route = routes[0] if routes else None

    while len(attempts) < MAX_COHORTS and clock() < deadline:
        attempt_number = len(attempts) + 1
        addresses, resolver_code = resolve_cohort_addresses(
            resolver=resolver, deadline=deadline, clock=clock
        )
        if resolver_code is not None:
            assert first_route is not None
            failure = make_failure(resolver_code, first_route.path)
            attempts.append(
                {
                    "attempt": attempt_number,
                    "cohort_sha256": None,
                    "completed_routes": 0,
                    "failure": failure,
                    "result": "FAIL",
                }
            )
            consecutive = 0
            first_matching_hash = None
            terminal_failure = failure
        else:
            assert addresses is not None
            cohort_records: list[dict[str, object]] = []
            failure = None
            completed_routes = 0
            for route in routes:
                if clock() >= deadline:
                    assert first_route is not None
                    failure = make_failure("deadline_exhausted", first_route.path)
                    break
                fetch = transport(route, addresses, deadline, clock)
                if fetch.failure is not None:
                    failure = fetch.failure
                    break
                if fetch.record is None:
                    failure = make_failure("internal_error", route.path)
                    break
                cohort_records.append(fetch.record)
                completed_routes += 1

            if failure is not None:
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "cohort_sha256": None,
                        "completed_routes": completed_routes,
                        "failure": failure,
                        "result": "FAIL",
                    }
                )
                consecutive = 0
                first_matching_hash = None
                terminal_failure = failure
            else:
                digest = cohort_sha256(cohort_records)
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "cohort_sha256": digest,
                        "completed_routes": len(routes),
                        "failure": None,
                        "result": "MATCH",
                    }
                )
                if first_matching_hash == digest:
                    consecutive += 1
                else:
                    first_matching_hash = digest
                    consecutive = 1
                terminal_failure = None
                accepted_routes = cohort_records
                if consecutive == STABLE_COHORTS_REQUIRED:
                    return {
                        "accepted_cohort_sha256": digest,
                        "attempted_cohorts": len(attempts),
                        "attempts": attempts,
                        "consecutive_matching_cohorts": consecutive,
                        "failures": [],
                        "routes": accepted_routes,
                    }

        if len(attempts) >= MAX_COHORTS:
            break
        if attempts[-1]["result"] == "MATCH":
            continue
        remaining = deadline - clock()
        if remaining <= 0:
            terminal_failure = make_failure(
                "deadline_exhausted",
                first_route.path if first_route is not None else "/",
            )
            break
        sleeper(min(float(RETRY_DELAY_SECONDS), remaining))
        if clock() >= deadline:
            terminal_failure = make_failure(
                "deadline_exhausted",
                first_route.path if first_route is not None else "/",
            )
            break

    if terminal_failure is None:
        if attempts and attempts[-1]["result"] == "FAIL":
            terminal_failure = attempts[-1]["failure"]  # type: ignore[assignment]
        elif len(attempts) >= MAX_COHORTS:
            terminal_failure = make_failure(
                "cohort_attempts_exhausted",
                first_route.path if first_route is not None else "/",
            )
        else:
            terminal_failure = make_failure(
                "deadline_exhausted",
                first_route.path if first_route is not None else "/",
            )
    return {
        "accepted_cohort_sha256": None,
        "attempted_cohorts": len(attempts),
        "attempts": attempts,
        "consecutive_matching_cohorts": consecutive,
        "failures": [terminal_failure],
        "routes": [],
    }


def build_receipt(
    *,
    result: str,
    subject: dict[str, object],
    workflow: dict[str, object],
    observation: dict[str, object],
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "authority": dict(AUTHORITY),
        "comparison_sha256": None,
        "kind": KIND,
        "observation": observation,
        "policy": dict(POLICY),
        "result": result,
        "schema_version": SCHEMA_VERSION,
        "subject": subject,
        "workflow": workflow,
    }
    if result == "PASS":
        comparison_basis = {
            "authority": dict(AUTHORITY),
            "kind": KIND,
            "observation": {
                "accepted_cohort_sha256": observation["accepted_cohort_sha256"],
                "consecutive_matching_cohorts": observation[
                    "consecutive_matching_cohorts"
                ],
                "routes": observation["routes"],
            },
            "policy": dict(POLICY),
            "result": "PASS",
            "schema_version": SCHEMA_VERSION,
            "subject": subject,
        }
        receipt["comparison_sha256"] = sha256_hex(
            canonical_json_bytes(comparison_basis)
        )
    return receipt


def write_receipt_exclusive(output: Path, receipt: dict[str, object]) -> None:
    data = receipt_json_bytes(receipt)
    with output.open("xb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())


def verify_public_deployment(
    local: LocalInputs,
    inventory: Inventory,
    *,
    resolver: Resolver = production_resolver,
    transport: Transport = production_transport,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[int, dict[str, object]]:
    subject = build_subject(local, inventory, verifier_source_sha256())
    workflow = {
        "run_attempt": local.workflow_run_attempt,
        "run_id": local.workflow_run_id,
    }
    observation = observe_routes(
        inventory.routes,
        resolver=resolver,
        transport=transport,
        clock=clock,
        sleeper=sleeper,
    )
    result = "PASS" if observation["accepted_cohort_sha256"] is not None else "FAIL"
    return (0 if result == "PASS" else 1), build_receipt(
        result=result,
        subject=subject,
        workflow=workflow,
        observation=observation,
    )


def internal_error_receipt(
    local: LocalInputs, inventory: Inventory | None
) -> dict[str, object]:
    if inventory is None or not inventory.routes:
        raise Refusal("cannot construct a bounded internal error receipt")
    route = inventory.routes[0]
    subject = build_subject(
        local,
        inventory,
        verifier_source_sha256(),
    )
    failure = make_failure("internal_error", route.path)
    observation = {
        "accepted_cohort_sha256": None,
        "attempted_cohorts": 1,
        "attempts": [
            {
                "attempt": 1,
                "cohort_sha256": None,
                "completed_routes": 0,
                "failure": failure,
                "result": "FAIL",
            }
        ],
        "consecutive_matching_cohorts": 0,
        "failures": [failure],
        "routes": [],
    }
    return build_receipt(
        result="FAIL",
        subject=subject,
        workflow={
            "run_attempt": local.workflow_run_attempt,
            "run_id": local.workflow_run_id,
        },
        observation=observation,
    )


def main(argv: Sequence[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        arguments = parse_fixed_cli(argv)
        local = validate_local_inputs(arguments)
        inventory = inventory_site(local.site, local.commit_sha)
    except (InvocationError, Refusal):
        return 2
    except Exception:
        return 2

    try:
        exit_code, receipt = verify_public_deployment(local, inventory)
    except Exception:
        try:
            receipt = internal_error_receipt(local, inventory)
            write_receipt_exclusive(local.output, receipt)
        except Exception:
            return 2
        return 2

    try:
        write_receipt_exclusive(local.output, receipt)
    except Exception:
        return 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
