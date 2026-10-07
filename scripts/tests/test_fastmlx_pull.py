import argparse
import contextlib
import errno
import hashlib
import importlib.util
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import URLError
from urllib.parse import unquote


FASTMLX_PULL_PATH = Path(__file__).resolve().parents[1] / "fastmlx_pull.py"
_SPEC = importlib.util.spec_from_file_location("fastmlx_pull", FASTMLX_PULL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
FASTMLX_PULL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(FASTMLX_PULL)

# The exact module instance fastmlx_pull.py uses internally for network
# calls; patch attributes on *this* object, not on a separately-loaded copy.
DOWNLOADER = FASTMLX_PULL.downloader


# The downloader publishes with renameatx_np(RENAME_EXCL), which exists only on
# macOS; elsewhere it refuses by design. These cases exercise that real rename,
# so they skip off macOS. Public CI runs them in its macOS job and fails on any
# skip there.
REQUIRES_MACOS_EXCLUSIVE_RENAME = unittest.skipUnless(
    sys.platform == "darwin", "exclusive rename (renameatx_np) is macOS-only"
)

REPO_ID = "example/Test-Model"
REVISION = "e" * 40


def git_blob_sha1(data: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {len(data)}\0".encode())
    digest.update(data)
    return digest.hexdigest()


class FakeResponse:
    def __init__(self, data: bytes, url: str = "https://huggingface.co/fake"):
        self._data = data
        self._pos = 0
        self._url = url
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def geturl(self) -> str:
        return self._url

    def read(self, size=None):
        if size is None:
            size = len(self._data) - self._pos
        chunk = self._data[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk


class FakeOpener:
    """Serves the source-API document and per-file resolve requests without
    ever touching the network. Injected in place of ``downloader.https_opener``.
    """

    def __init__(self, api_bytes: bytes, content_by_name: dict):
        self.api_bytes = api_bytes
        self.content_by_name = content_by_name
        self.resolve_calls: list[str] = []

    def _name_from_resolve_url(self, url: str) -> str:
        marker = "/resolve/"
        index = url.index(marker)
        after = url[index + len(marker) :]
        rev_and_name = after.split("?", 1)[0]
        _, _, encoded_name = rev_and_name.partition("/")
        return unquote(encoded_name)

    def open(self, request, timeout=60):
        url = request.full_url
        if url.startswith("https://huggingface.co/api/models/"):
            return FakeResponse(self.api_bytes)
        name = self._name_from_resolve_url(url)
        self.resolve_calls.append(name)
        return FakeResponse(self.content_by_name[name])


class HardFailThenSucceedOpener(FakeOpener):
    """Like FakeOpener, but a named file raises a hard network error on its
    first ``hard_fail_calls`` resolve attempts, then serves real content.

    ``hard_fail_calls`` set to 5 exhausts the downloader's own internal
    per-file retry budget (``for attempt in range(1, 6)`` in
    ``hf_pinned_snapshot_download.download_entry``), so the *first*
    top-level ``acquire()`` call this feeds fails outright with an
    ``AcquisitionError`` -- simulating a supervisor-level attempt that must
    be retried, not a transient blip absorbed inside one attempt.
    """

    def __init__(self, api_bytes, content_by_name, hard_fail_name, hard_fail_calls=5):
        super().__init__(api_bytes, content_by_name)
        self.hard_fail_name = hard_fail_name
        self.hard_fail_calls = hard_fail_calls
        self.hard_fail_seen = 0

    def open(self, request, timeout=60):
        url = request.full_url
        if url.startswith("https://huggingface.co/") and "/resolve/" in url:
            name = self._name_from_resolve_url(url)
            if name == self.hard_fail_name and self.hard_fail_seen < self.hard_fail_calls:
                self.hard_fail_seen += 1
                self.resolve_calls.append(name)
                raise URLError("simulated hard network failure")
        return super().open(request, timeout=timeout)


class MultiHardFailOpener(FakeOpener):
    """Like ``HardFailThenSucceedOpener``, but hard-fails more than one
    named file, each for its own first ``hard_fail_calls`` resolve
    attempts, then serves real content for that name afterward. Used to
    script a chain of supervisor attempts that each fail on a different
    file.
    """

    def __init__(self, api_bytes, content_by_name, hard_fail_names, hard_fail_calls=5):
        super().__init__(api_bytes, content_by_name)
        self.hard_fail_calls = hard_fail_calls
        self.hard_fail_seen = {name: 0 for name in hard_fail_names}

    def open(self, request, timeout=60):
        url = request.full_url
        if url.startswith("https://huggingface.co/") and "/resolve/" in url:
            name = self._name_from_resolve_url(url)
            seen = self.hard_fail_seen.get(name)
            if seen is not None and seen < self.hard_fail_calls:
                self.hard_fail_seen[name] = seen + 1
                self.resolve_calls.append(name)
                raise URLError("simulated hard network failure")
        return super().open(request, timeout=timeout)


class SyntheticRepo:
    """A 3-file public-repo fixture, mirroring the downloader's own tests."""

    def __init__(self):
        self.repo_id = REPO_ID
        self.revision = REVISION
        self.content_by_name = {
            "README.md": b"# synthetic\n",
            "config.json": b'{"hello":"world"}',
            "model.safetensors": (b"synthetic-shard-bytes-" * 50),
        }
        self.document = self._build_document()
        self.api_bytes = self._serialize()

    def _build_document(self) -> dict:
        siblings = []
        for name, data in self.content_by_name.items():
            if name.endswith(".safetensors"):
                sibling = {
                    "rfilename": name,
                    "size": len(data),
                    "blobId": "0" * 40,
                    "lfs": {
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "size": len(data),
                    },
                }
            else:
                sibling = {
                    "rfilename": name,
                    "size": len(data),
                    "blobId": git_blob_sha1(data),
                }
            siblings.append(sibling)
        return {
            "id": self.repo_id,
            "sha": self.revision,
            "private": False,
            "siblings": siblings,
        }

    def _serialize(self) -> bytes:
        return json.dumps(self.document, sort_keys=True).encode()


class NestedRepo(SyntheticRepo):
    """A repo fixture whose manifest names files inside a subdirectory, to
    exercise a manifest-named subdirectory that also contains an unlisted
    extra file."""

    def __init__(self):
        self.repo_id = REPO_ID
        self.revision = REVISION
        self.content_by_name = {
            "weights/config.json": b'{"hello":"world"}',
            "weights/model.safetensors": (b"synthetic-shard-bytes-" * 50),
        }
        self.document = self._build_document()
        self.api_bytes = self._serialize()


class TwoFileRepo(SyntheticRepo):
    """A minimal 2-file public-repo fixture for staging-dir resume tests."""

    def __init__(self):
        self.repo_id = REPO_ID
        self.revision = REVISION
        self.content_by_name = {
            "config.json": b'{"hello":"world"}',
            "model.safetensors": (b"synthetic-shard-bytes-" * 50),
        }
        self.document = self._build_document()
        self.api_bytes = self._serialize()


class ValidatePinnedReferenceTests(unittest.TestCase):
    """Acceptance criterion 1: only a 40-char lowercase hex sha is accepted."""

    def test_accepts_a_canonical_pinned_reference(self):
        repo_id, revision = FASTMLX_PULL.validate_pinned_reference(
            f"{REPO_ID}@{REVISION}"
        )
        self.assertEqual(repo_id, REPO_ID)
        self.assertEqual(revision, REVISION)

    def test_refuses_a_branch_name(self):
        with self.assertRaises(FASTMLX_PULL.PinnedReferenceError) as ctx:
            FASTMLX_PULL.validate_pinned_reference(f"{REPO_ID}@main")
        message = str(ctx.exception)
        self.assertIn("40-character", message)
        self.assertIn("'main'", message)

    def test_refuses_a_short_sha(self):
        with self.assertRaises(FASTMLX_PULL.PinnedReferenceError) as ctx:
            FASTMLX_PULL.validate_pinned_reference(f"{REPO_ID}@{'a' * 7}")
        self.assertIn("40-character", str(ctx.exception))

    def test_refuses_uppercase_hex(self):
        with self.assertRaises(FASTMLX_PULL.PinnedReferenceError) as ctx:
            FASTMLX_PULL.validate_pinned_reference(f"{REPO_ID}@{'A' * 40}")
        message = str(ctx.exception)
        self.assertIn("not all lowercase", message)

    def test_refuses_a_missing_at_sign(self):
        with self.assertRaises(FASTMLX_PULL.PinnedReferenceError) as ctx:
            FASTMLX_PULL.validate_pinned_reference(f"{REPO_ID}{REVISION}")
        self.assertIn("'@'", str(ctx.exception))

    def test_cli_main_exits_nonzero_with_reason_on_stderr(self):
        import contextlib
        import io

        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "model"
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as ctx:
                    FASTMLX_PULL.main([f"{REPO_ID}@main", "--dest", str(dest)])
        self.assertNotEqual(ctx.exception.code, 0)
        self.assertIn("40-character", stderr.getvalue())
        self.assertFalse(dest.exists())


class PullSupervisorTests(unittest.TestCase):
    def opener_for(self, repo: SyntheticRepo) -> FakeOpener:
        return FakeOpener(repo.api_bytes, dict(repo.content_by_name))

    # ------------------------------------------------------------------
    # Happy path: one attempt, nothing reused, full receipt written.
    # ------------------------------------------------------------------
    @REQUIRES_MACOS_EXCLUSIVE_RENAME
    def test_happy_path_single_attempt_writes_a_complete_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            dest = root / "model"
            opener = self.opener_for(repo)

            with mock.patch.object(DOWNLOADER, "https_opener", return_value=opener):
                receipt_path = FASTMLX_PULL.pull(
                    repo_id=repo.repo_id, revision=repo.revision, dest=dest
                )

            self.assertTrue(dest.is_dir())
            self.assertEqual(receipt_path, FASTMLX_PULL.receipt_path_for(dest))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

            self.assertEqual(receipt["repo_id"], repo.repo_id)
            self.assertEqual(receipt["revision"], repo.revision)
            self.assertEqual(receipt["attempts"], 1)
            self.assertEqual(receipt["reused_files"], 0)
            self.assertEqual(receipt["reused_bytes"], 0)
            self.assertEqual(receipt["total_files"], 3)
            self.assertEqual(
                receipt["downloader_script_sha256"],
                FASTMLX_PULL.downloader_script_sha256(),
            )
            self.assertEqual(
                receipt["files"]["README.md"],
                {
                    "sha256": hashlib.sha256(
                        repo.content_by_name["README.md"]
                    ).hexdigest(),
                    "size": len(repo.content_by_name["README.md"]),
                },
            )
            self.assertEqual(set(receipt["files"]), set(repo.content_by_name))

    # ------------------------------------------------------------------
    # Acceptance criterion 4: resume across attempts, reusing what a
    # failed attempt already verified; every staging tree is preserved.
    # ------------------------------------------------------------------
    @REQUIRES_MACOS_EXCLUSIVE_RENAME
    def test_second_attempt_reuses_the_first_attempts_verified_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            dest = root / "model"
            # "config.json" sorts before "model.safetensors" and after
            # "README.md", so attempt 1 downloads README.md, then fails
            # hard on config.json, and never reaches model.safetensors.
            opener = HardFailThenSucceedOpener(
                repo.api_bytes,
                dict(repo.content_by_name),
                hard_fail_name="config.json",
                hard_fail_calls=5,
            )

            with mock.patch.object(
                DOWNLOADER, "https_opener", return_value=opener
            ), mock.patch.object(DOWNLOADER.time, "sleep", return_value=None):
                receipt_path = FASTMLX_PULL.pull(
                    repo_id=repo.repo_id,
                    revision=repo.revision,
                    dest=dest,
                    max_attempts=3,
                )

            self.assertTrue(dest.is_dir())
            for name, data in repo.content_by_name.items():
                self.assertEqual((dest / name).read_bytes(), data)

            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["attempts"], 2)
            self.assertEqual(receipt["reused_files"], 1)
            self.assertEqual(
                receipt["reused_bytes"], len(repo.content_by_name["README.md"])
            )

            # README.md's resolve endpoint was hit exactly once, ever: it
            # was never re-fetched on the second attempt.
            self.assertEqual(
                opener.resolve_calls.count("README.md"), 1
            )

            staging_dirs = list(root.glob(f".{dest.name}.acquiring-*"))
            self.assertEqual(len(staging_dirs), 1, staging_dirs)
            self.assertTrue(staging_dirs[0].is_dir())

    # ------------------------------------------------------------------
    # Review defect: receipt hashing must stream, not load whole files.
    # ------------------------------------------------------------------
    @REQUIRES_MACOS_EXCLUSIVE_RENAME
    def test_receipt_hashing_streams_published_files_instead_of_reading_them_whole(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            dest = root / "model"
            opener = self.opener_for(repo)

            original_read_bytes = Path.read_bytes

            def guarded_read_bytes(self):
                # Only the receipt phase's reads of *published* files are
                # under test here; the downloader's own script-hash read
                # (a small script file, not shard data) is unaffected.
                try:
                    self.relative_to(dest)
                except ValueError:
                    return original_read_bytes(self)
                raise AssertionError(
                    "receipt computation must not Path.read_bytes() a "
                    f"published file into memory whole: {self}"
                )

            with mock.patch.object(
                DOWNLOADER, "https_opener", return_value=opener
            ), mock.patch.object(Path, "read_bytes", guarded_read_bytes):
                receipt_path = FASTMLX_PULL.pull(
                    repo_id=repo.repo_id, revision=repo.revision, dest=dest
                )

            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            for name, data in repo.content_by_name.items():
                self.assertEqual(
                    receipt["files"][name],
                    {
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "size": len(data),
                    },
                )

    # ------------------------------------------------------------------
    # Review defect: resume across a killed *process*, not just a failed
    # in-process attempt. A preserved staging tree left by an earlier,
    # separately-invoked pull() must be used as attempt 1's reuse source.
    # ------------------------------------------------------------------
    @REQUIRES_MACOS_EXCLUSIVE_RENAME
    def test_pull_resumes_from_a_staging_tree_preserved_by_an_earlier_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = TwoFileRepo()
            dest = root / "model"
            opener = self.opener_for(repo)

            good_name = "config.json"
            corrupt_name = "model.safetensors"
            staging = root / f".{dest.name}.acquiring-99999-deadbeef"
            staging.mkdir()
            (staging / good_name).write_bytes(repo.content_by_name[good_name])
            corrupt_bytes = bytearray(repo.content_by_name[corrupt_name])
            corrupt_bytes[0] ^= 0xFF
            (staging / corrupt_name).write_bytes(bytes(corrupt_bytes))

            with mock.patch.object(DOWNLOADER, "https_opener", return_value=opener):
                receipt_path = FASTMLX_PULL.pull(
                    repo_id=repo.repo_id, revision=repo.revision, dest=dest
                )

            self.assertTrue(dest.is_dir())
            for name, data in repo.content_by_name.items():
                self.assertEqual((dest / name).read_bytes(), data)

            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["attempts"], 1)
            self.assertEqual(receipt["reused_files"], 1)
            self.assertEqual(
                receipt["reused_bytes"], len(repo.content_by_name[good_name])
            )

            # Only the corrupt file was ever fetched from the network; the
            # already-correct staging-tree file was reused by move.
            self.assertEqual(opener.resolve_calls, [corrupt_name])

    # ------------------------------------------------------------------
    # Acceptance criterion 4 (extended): a 3-attempt resume chain, each of
    # the first two attempts downloading a new file before hard-failing.
    #
    # Reuse moves the verified candidate (os.rename) instead of hardlinking
    # it, so a file keeps a single link (nlink == 1) wherever it currently
    # lives and stays eligible for a *later* hop of reuse: README.md,
    # downloaded fresh in attempt 1, is moved into attempt 2's tree and then
    # moved again into attempt 3's tree (multi-hop reuse) without ever being
    # refetched. config.json, downloaded fresh in attempt 2, is moved into
    # attempt 3's tree. model.safetensors is never verified in any earlier
    # tree, so it is downloaded fresh in attempt 3.
    # ------------------------------------------------------------------
    @REQUIRES_MACOS_EXCLUSIVE_RENAME
    def test_third_attempt_in_a_chain_succeeds_and_preserves_both_earlier_staging_trees(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            dest = root / "model"
            hard_fail_calls = 5
            opener = MultiHardFailOpener(
                repo.api_bytes,
                dict(repo.content_by_name),
                hard_fail_names=["config.json", "model.safetensors"],
                hard_fail_calls=hard_fail_calls,
            )

            with mock.patch.object(
                DOWNLOADER, "https_opener", return_value=opener
            ), mock.patch.object(DOWNLOADER.time, "sleep", return_value=None):
                receipt_path = FASTMLX_PULL.pull(
                    repo_id=repo.repo_id,
                    revision=repo.revision,
                    dest=dest,
                    max_attempts=3,
                )

            self.assertTrue(dest.is_dir())
            for name, data in repo.content_by_name.items():
                self.assertEqual((dest / name).read_bytes(), data)
                # Every published file is a move target, never a hardlink:
                # a single link, wherever it currently lives.
                self.assertEqual((dest / name).stat().st_nlink, 1, name)

            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["attempts"], 3)
            # README.md (moved twice, attempt1 -> attempt2 -> attempt3) and
            # config.json (moved once, attempt2 -> attempt3) are both
            # reused; model.safetensors is a fresh download in attempt 3.
            self.assertEqual(receipt["reused_files"], 2)
            self.assertEqual(
                receipt["reused_bytes"],
                len(repo.content_by_name["README.md"])
                + len(repo.content_by_name["config.json"]),
            )

            # README.md: fetched exactly once, ever -- reused by move
            # through both later attempts, never refetched.
            self.assertEqual(opener.resolve_calls.count("README.md"), 1)
            # config.json: 5 hard failures in attempt 1, then one real
            # success in attempt 2; reused (moved) into attempt 3, never
            # refetched.
            self.assertEqual(
                opener.resolve_calls.count("config.json"), hard_fail_calls + 1
            )
            # model.safetensors: 5 hard failures in attempt 2, then one
            # real success in attempt 3.
            self.assertEqual(
                opener.resolve_calls.count("model.safetensors"), hard_fail_calls + 1
            )

            staging_dirs = list(root.glob(f".{dest.name}.acquiring-*"))
            self.assertEqual(len(staging_dirs), 2, staging_dirs)
            for staging_dir in staging_dirs:
                self.assertTrue(staging_dir.is_dir())

    # ------------------------------------------------------------------
    # Acceptance criterion 4: attempt bound; all-fail names the count and
    # preserves every staging tree.
    # ------------------------------------------------------------------
    def test_exhausting_max_attempts_fails_and_preserves_every_staging_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            dest = root / "model"
            content_by_name = dict(repo.content_by_name)
            real = content_by_name["config.json"]
            content_by_name["config.json"] = real[:-1] + (
                b"!" if real[-1:] != b"!" else b"?"
            )
            opener = FakeOpener(repo.api_bytes, content_by_name)

            with mock.patch.object(DOWNLOADER, "https_opener", return_value=opener):
                with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
                    FASTMLX_PULL.pull(
                        repo_id=repo.repo_id,
                        revision=repo.revision,
                        dest=dest,
                        max_attempts=2,
                    )

            self.assertIn("2 attempt", str(ctx.exception))
            self.assertFalse(dest.exists())
            self.assertFalse(FASTMLX_PULL.receipt_path_for(dest).exists())

            staging_dirs = list(root.glob(f".{dest.name}.acquiring-*"))
            self.assertEqual(len(staging_dirs), 2, staging_dirs)

    # ------------------------------------------------------------------
    # Free-space floor check
    # ------------------------------------------------------------------
    def test_refuses_before_any_attempt_when_free_space_is_insufficient(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            dest = root / "model"
            opener = self.opener_for(repo)

            with mock.patch.object(DOWNLOADER, "https_opener", return_value=opener):
                with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
                    FASTMLX_PULL.pull(
                        repo_id=repo.repo_id,
                        revision=repo.revision,
                        dest=dest,
                        disk_usage_bytes=lambda path: 1,
                    )

            self.assertIn("insufficient free space", str(ctx.exception))
            self.assertEqual(opener.resolve_calls, [])
            self.assertFalse(dest.exists())
            staging_dirs = list(root.glob(f".{dest.name}.acquiring-*"))
            self.assertEqual(staging_dirs, [])

    def test_explicit_min_free_bytes_overrides_the_computed_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            dest = root / "model"
            opener = self.opener_for(repo)
            total_bytes = sum(len(data) for data in repo.content_by_name.values())
            # Above the computed default (1.2x total) but below the
            # explicit floor we pass, proving the explicit floor is used.
            available = int(total_bytes * DOWNLOADER.FREE_SPACE_SAFETY_MULTIPLIER) + 1

            with mock.patch.object(DOWNLOADER, "https_opener", return_value=opener):
                with self.assertRaises(FASTMLX_PULL.PullError):
                    FASTMLX_PULL.pull(
                        repo_id=repo.repo_id,
                        revision=repo.revision,
                        dest=dest,
                        min_free_bytes=available + 1,
                        disk_usage_bytes=lambda path: available,
                    )

    # F0: a --dest whose parent does not exist yet must not escape the real
    # free-space probe as a raw FileNotFoundError (the probe statvfs()s the
    # parent). The nearest existing ancestor is probed instead.
    # The downloader itself never creates the output parent, so a plain pull
    # into a missing parent still fails -- but as a PullError naming the
    # parent, not as a raw FileNotFoundError out of the free-space probe.
    def test_f0_missing_dest_parent_is_a_pull_error_not_a_raw_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            dest = root / "not" / "yet" / "model"
            opener = self.opener_for(repo)

            with mock.patch.object(
                DOWNLOADER, "https_opener", return_value=opener
            ), mock.patch.object(DOWNLOADER.time, "sleep", return_value=None):
                with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
                    FASTMLX_PULL.pull(
                        repo_id=repo.repo_id, revision=repo.revision, dest=dest
                    )

            self.assertIn("missing output parent", str(ctx.exception))
            self.assertFalse(dest.exists())

    def test_f0_missing_dest_parent_with_insufficient_space_is_a_pull_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            dest = root / "not" / "yet" / "model"
            opener = self.opener_for(repo)

            with mock.patch.object(DOWNLOADER, "https_opener", return_value=opener):
                with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
                    FASTMLX_PULL.pull(
                        repo_id=repo.repo_id,
                        revision=repo.revision,
                        dest=dest,
                        min_free_bytes=1 << 62,
                    )

            self.assertIn("insufficient free space", str(ctx.exception))
            self.assertFalse((root / "not").exists())

    # F3: plain pull keeps an effective --max-attempts default of 3, and an
    # explicit value still reaches pull().
    def test_f3_plain_pull_keeps_default_max_attempts_three(self):
        argv_base = [f"{REPO_ID}@{REVISION}", "--dest", "/nonexistent-dest/model"]
        with mock.patch.object(FASTMLX_PULL, "pull") as pull_mock:
            FASTMLX_PULL.main(list(argv_base))
            self.assertEqual(pull_mock.call_args.kwargs["max_attempts"], 3)
            FASTMLX_PULL.main(argv_base + ["--max-attempts", "5"])
            self.assertEqual(pull_mock.call_args.kwargs["max_attempts"], 5)

    # ------------------------------------------------------------------
    # Acceptance criterion 5: refuse to overwrite an existing receipt.
    # ------------------------------------------------------------------
    def test_refuses_to_overwrite_an_existing_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            dest = root / "model"
            receipt_path = FASTMLX_PULL.receipt_path_for(dest)
            receipt_path.write_text("{}", encoding="utf-8")
            opener = self.opener_for(repo)

            with mock.patch.object(DOWNLOADER, "https_opener", return_value=opener):
                with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
                    FASTMLX_PULL.pull(
                        repo_id=repo.repo_id, revision=repo.revision, dest=dest
                    )

            self.assertIn("existing pull receipt", str(ctx.exception))
            self.assertFalse(dest.exists())
            self.assertEqual(opener.resolve_calls, [])


class AdoptTests(unittest.TestCase):
    """``fastmlx pull --adopt``: verify an already-staged directory against
    a pinned revision's manifest instead of downloading it.
    """

    def opener_for(self, repo: SyntheticRepo) -> FakeOpener:
        return FakeOpener(repo.api_bytes, dict(repo.content_by_name))

    def stage_correctly(self, dest: Path, repo: SyntheticRepo) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        for name, data in repo.content_by_name.items():
            path = dest / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    # ------------------------------------------------------------------
    # Happy path: a correctly hand-staged directory is adopted, and the
    # resulting receipt is what the launcher actually reads.
    # ------------------------------------------------------------------
    def test_adopt_success_writes_a_receipt_with_acquisition_adopted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            dest = root / "model"
            self.stage_correctly(dest, repo)
            opener = self.opener_for(repo)

            with mock.patch.object(DOWNLOADER, "https_opener", return_value=opener):
                receipt_path = FASTMLX_PULL.adopt(
                    repo_id=repo.repo_id, revision=repo.revision, dest=dest
                )

            # Adopt never fetches file bytes over the network -- only the
            # source-API manifest request is made (served by FakeOpener's
            # api-bytes branch); no /resolve/ URL is ever hit.
            self.assertEqual(opener.resolve_calls, [])

            self.assertEqual(receipt_path, FASTMLX_PULL.receipt_path_for(dest))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["repo_id"], repo.repo_id)
            self.assertEqual(receipt["revision"], repo.revision)
            self.assertEqual(receipt["acquisition"], "adopted")
            self.assertEqual(receipt["dest"], str(dest.absolute()))
            self.assertEqual(receipt["total_files"], 3)
            self.assertEqual(receipt["ignored_local_paths"], [])
            for name, data in repo.content_by_name.items():
                self.assertEqual(
                    receipt["files"][name]["sha256"],
                    hashlib.sha256(data).hexdigest(),
                )
                self.assertEqual(receipt["files"][name]["size"], len(data))
                self.assertIn(
                    receipt["files"][name]["verified"],
                    ("lfs-sha256", "git-blob-sha1"),
                )
            # The LFS entry (model.safetensors) is verified by streamed
            # sha256; the two small non-LFS entries are verified by the
            # downloader's own git-blob-sha1 identity.
            self.assertEqual(
                receipt["files"]["model.safetensors"]["verified"], "lfs-sha256"
            )
            self.assertEqual(
                receipt["files"]["config.json"]["verified"], "git-blob-sha1"
            )

            # End-to-end: the launcher resolves this receipt's pinned
            # revision exactly the way a real pull's receipt is resolved.
            launch_spec = importlib.util.spec_from_file_location(
                "fastmlx_launch",
                Path(__file__).resolve().parents[1] / "fastmlx_launch.py",
            )
            assert launch_spec is not None and launch_spec.loader is not None
            launch_module = importlib.util.module_from_spec(launch_spec)
            launch_spec.loader.exec_module(launch_module)
            args = argparse.Namespace(model_revision=None)
            self.assertEqual(
                launch_module._resolve_model_revision(args, dest), repo.revision
            )

    # ------------------------------------------------------------------
    # Refusals: missing file, size mismatch, sha mismatch, symlink.
    # ------------------------------------------------------------------
    def test_refuses_a_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            dest = root / "model"
            self.stage_correctly(dest, repo)
            (dest / "README.md").unlink()
            opener = self.opener_for(repo)

            with mock.patch.object(DOWNLOADER, "https_opener", return_value=opener):
                with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
                    FASTMLX_PULL.adopt(
                        repo_id=repo.repo_id, revision=repo.revision, dest=dest
                    )
            self.assertIn("missing file", str(ctx.exception))
            self.assertIn("README.md", str(ctx.exception))
            self.assertFalse(FASTMLX_PULL.receipt_path_for(dest).exists())

    def test_refuses_a_size_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            dest = root / "model"
            self.stage_correctly(dest, repo)
            (dest / "config.json").write_bytes(b"{}")
            opener = self.opener_for(repo)

            with mock.patch.object(DOWNLOADER, "https_opener", return_value=opener):
                with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
                    FASTMLX_PULL.adopt(
                        repo_id=repo.repo_id, revision=repo.revision, dest=dest
                    )
            self.assertIn("size mismatch", str(ctx.exception))
            self.assertIn("config.json", str(ctx.exception))
            self.assertFalse(FASTMLX_PULL.receipt_path_for(dest).exists())

    def test_refuses_a_sha_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            dest = root / "model"
            self.stage_correctly(dest, repo)
            original = repo.content_by_name["model.safetensors"]
            corrupted = bytearray(original)
            corrupted[0] ^= 0xFF
            (dest / "model.safetensors").write_bytes(bytes(corrupted))
            opener = self.opener_for(repo)

            with mock.patch.object(DOWNLOADER, "https_opener", return_value=opener):
                with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
                    FASTMLX_PULL.adopt(
                        repo_id=repo.repo_id, revision=repo.revision, dest=dest
                    )
            self.assertIn("content mismatch", str(ctx.exception))
            self.assertIn("model.safetensors", str(ctx.exception))
            self.assertFalse(FASTMLX_PULL.receipt_path_for(dest).exists())

    @REQUIRES_MACOS_EXCLUSIVE_RENAME
    def test_refuses_a_symlinked_file_even_if_content_is_correct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            dest = root / "model"
            self.stage_correctly(dest, repo)
            real_target = root / "real-config.json"
            real_target.write_bytes(repo.content_by_name["config.json"])
            (dest / "config.json").unlink()
            (dest / "config.json").symlink_to(real_target)
            opener = self.opener_for(repo)

            with mock.patch.object(DOWNLOADER, "https_opener", return_value=opener):
                with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
                    FASTMLX_PULL.adopt(
                        repo_id=repo.repo_id, revision=repo.revision, dest=dest
                    )
            self.assertIn("not a regular file", str(ctx.exception))
            self.assertIn("config.json", str(ctx.exception))
            self.assertFalse(FASTMLX_PULL.receipt_path_for(dest).exists())

    # ------------------------------------------------------------------
    # An extra file not in the manifest is refused, at any depth; a
    # harmless extra (dotfile / dot-directory) is ignored and recorded.
    # ------------------------------------------------------------------
    def test_refuses_an_extra_top_level_safetensors_file_not_in_the_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            dest = root / "model"
            self.stage_correctly(dest, repo)
            (dest / "extra-shard.safetensors").write_bytes(b"unverified-bytes")
            opener = self.opener_for(repo)

            with mock.patch.object(DOWNLOADER, "https_opener", return_value=opener):
                with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
                    FASTMLX_PULL.adopt(
                        repo_id=repo.repo_id, revision=repo.revision, dest=dest
                    )
            self.assertIn("extra-shard.safetensors", str(ctx.exception))
            self.assertFalse(FASTMLX_PULL.receipt_path_for(dest).exists())

    def test_refuses_an_unlisted_non_loadable_top_level_file(self):
        # Not a *.safetensors/*.gguf/*.bin file -- under the old top-level
        # suffix-based check this was silently ignored; it must now be
        # refused like any other unlisted file.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            dest = root / "model"
            self.stage_correctly(dest, repo)
            (dest / "chat_template.jinja").write_bytes(b"{{ messages }}")
            opener = self.opener_for(repo)

            with mock.patch.object(DOWNLOADER, "https_opener", return_value=opener):
                with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
                    FASTMLX_PULL.adopt(
                        repo_id=repo.repo_id, revision=repo.revision, dest=dest
                    )
            self.assertIn("chat_template.jinja", str(ctx.exception))
            self.assertFalse(FASTMLX_PULL.receipt_path_for(dest).exists())

    def test_refuses_an_unlisted_file_nested_in_a_manifest_named_subdirectory(self):
        # Under the old top-level-component skip, an entire manifest-named
        # subdirectory was never looked into again once its top component
        # matched; an extra file hidden inside it was silently ignored.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = NestedRepo()
            dest = root / "model"
            self.stage_correctly(dest, repo)
            (dest / "weights" / "extra.bin").write_bytes(b"unverified-bytes")
            opener = self.opener_for(repo)

            with mock.patch.object(DOWNLOADER, "https_opener", return_value=opener):
                with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
                    FASTMLX_PULL.adopt(
                        repo_id=repo.repo_id, revision=repo.revision, dest=dest
                    )
            self.assertIn("weights/extra.bin", str(ctx.exception))
            self.assertFalse(FASTMLX_PULL.receipt_path_for(dest).exists())

    def test_refuses_an_unlisted_file_in_a_non_manifest_subdirectory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            dest = root / "model"
            self.stage_correctly(dest, repo)
            (dest / "extra_dir").mkdir()
            (dest / "extra_dir" / "notes.txt").write_bytes(b"notes")
            opener = self.opener_for(repo)

            with mock.patch.object(DOWNLOADER, "https_opener", return_value=opener):
                with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
                    FASTMLX_PULL.adopt(
                        repo_id=repo.repo_id, revision=repo.revision, dest=dest
                    )
            self.assertIn("extra_dir/notes.txt", str(ctx.exception))
            self.assertFalse(FASTMLX_PULL.receipt_path_for(dest).exists())

    def test_ignores_and_records_a_nested_dot_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            dest = root / "model"
            self.stage_correctly(dest, repo)
            cache_dir = dest / ".cache" / "x"
            cache_dir.mkdir(parents=True)
            (cache_dir / "y").write_bytes(b"cache-data")
            opener = self.opener_for(repo)

            with mock.patch.object(DOWNLOADER, "https_opener", return_value=opener):
                receipt_path = FASTMLX_PULL.adopt(
                    repo_id=repo.repo_id, revision=repo.revision, dest=dest
                )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            # The walk never descends past the dot-directory itself, so
            # only ".cache" is recorded -- not the file inside it.
            self.assertEqual(receipt["ignored_local_paths"], [".cache"])

    @REQUIRES_MACOS_EXCLUSIVE_RENAME
    def test_refuses_an_unlisted_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            dest = root / "model"
            self.stage_correctly(dest, repo)
            real_target = root / "real-extra.bin"
            real_target.write_bytes(b"extra-content")
            (dest / "extra-link").symlink_to(real_target)
            opener = self.opener_for(repo)

            with mock.patch.object(DOWNLOADER, "https_opener", return_value=opener):
                with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
                    FASTMLX_PULL.adopt(
                        repo_id=repo.repo_id, revision=repo.revision, dest=dest
                    )
            self.assertIn("extra-link", str(ctx.exception))
            self.assertFalse(FASTMLX_PULL.receipt_path_for(dest).exists())

    def test_ignores_and_records_a_harmless_extra_dotfile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            dest = root / "model"
            self.stage_correctly(dest, repo)
            (dest / ".DS_Store").write_bytes(b"junk")
            opener = self.opener_for(repo)

            with mock.patch.object(DOWNLOADER, "https_opener", return_value=opener):
                receipt_path = FASTMLX_PULL.adopt(
                    repo_id=repo.repo_id, revision=repo.revision, dest=dest
                )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["ignored_local_paths"], [".DS_Store"])

    # ------------------------------------------------------------------
    # An existing receipt is never overwritten by --adopt either.
    # ------------------------------------------------------------------
    def test_refuses_to_overwrite_an_existing_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            dest = root / "model"
            self.stage_correctly(dest, repo)
            receipt_path = FASTMLX_PULL.receipt_path_for(dest)
            receipt_path.write_text("{}", encoding="utf-8")
            opener = self.opener_for(repo)

            with mock.patch.object(DOWNLOADER, "https_opener", return_value=opener):
                with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
                    FASTMLX_PULL.adopt(
                        repo_id=repo.repo_id, revision=repo.revision, dest=dest
                    )
            self.assertIn("existing pull receipt", str(ctx.exception))

    # ------------------------------------------------------------------
    # A non-existent directory cannot be adopted (--adopt is for a
    # directory that is already there, never a fresh download destination).
    # ------------------------------------------------------------------
    def test_refuses_a_dest_that_does_not_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            dest = root / "not-there"
            opener = self.opener_for(repo)

            with mock.patch.object(DOWNLOADER, "https_opener", return_value=opener):
                with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
                    FASTMLX_PULL.adopt(
                        repo_id=repo.repo_id, revision=repo.revision, dest=dest
                    )
            self.assertIn("existing directory", str(ctx.exception))

    # ------------------------------------------------------------------
    # CLI wiring: --adopt routes through fastmlx_pull.main to adopt(), not
    # pull().
    # ------------------------------------------------------------------
    def test_cli_adopt_flag_calls_adopt_not_pull(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            dest = root / "model"
            self.stage_correctly(dest, repo)
            opener = self.opener_for(repo)

            with mock.patch.object(DOWNLOADER, "https_opener", return_value=opener):
                FASTMLX_PULL.main(
                    [
                        f"{repo.repo_id}@{repo.revision}",
                        "--dest",
                        str(dest),
                        "--adopt",
                    ]
                )
            receipt = json.loads(
                FASTMLX_PULL.receipt_path_for(dest).read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["acquisition"], "adopted")


# ---------------------------------------------------------------------
# `fastmlx pull --from-hub-cache [DIR]`: import a pinned pack from the local
# Hugging Face hub cache (blobs/ + snapshots/<rev>/ relative symlinks) into a
# fresh directory as REGULAR files, verified like --adopt. The cache is only
# ever read. No network: fetch_api is stubbed, acquire/https_opener raise.
# ---------------------------------------------------------------------
class HubCacheImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.hub = self.root / "hub"
        self.work = self.root / "work"
        self.work.mkdir()
        self.dest = self.work / "model"
        self.repo = SyntheticRepo()
        self.repo_dir = self.hub / ("models--" + self.repo.repo_id.replace("/", "--"))
        self.blobs = self.repo_dir / "blobs"
        self.snapshot = self.repo_dir / "snapshots" / self.repo.revision
        self.blob_for = {}
        self.build_hub()

        self.fetch_api = mock.patch.object(
            DOWNLOADER, "fetch_api", return_value=(self.repo.document, b"")
        ).start()
        self.acquire = mock.patch.object(
            DOWNLOADER,
            "acquire",
            side_effect=AssertionError("the downloader must never be invoked"),
        ).start()
        self.opener = mock.patch.object(
            DOWNLOADER,
            "https_opener",
            side_effect=AssertionError("no network request may be made"),
        ).start()
        self.addCleanup(mock.patch.stopall)

    def build_hub(self):
        self.blobs.mkdir(parents=True)
        for name, data in self.repo.content_by_name.items():
            blob = self.blobs / hashlib.sha256(b"blob:" + name.encode() + data).hexdigest()
            blob.write_bytes(data)
            self.blob_for[name] = blob
            link = self.snapshot / name
            link.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(os.path.relpath(blob, link.parent), link)

    def run_import(self, hub=None, preflight=None, dest=None, **kwargs):
        return FASTMLX_PULL.import_from_hub_cache(
            repo_id=self.repo.repo_id,
            revision=self.repo.revision,
            dest=dest or self.dest,
            hub_dir=hub or self.hub,
            preflight=preflight,
            **kwargs,
        )

    def total_bytes(self):
        return sum(len(d) for d in self.repo.content_by_name.values())

    def default_required(self):
        return int(self.total_bytes() * DOWNLOADER.FREE_SPACE_SAFETY_MULTIPLIER)

    def assert_nothing_written(self):
        self.assertEqual(sorted(self.work.iterdir()), [])

    def cache_fingerprint(self):
        out = {}
        for path in sorted(self.blobs.iterdir()):
            info = path.lstat()
            out[path.name] = (
                hashlib.sha256(path.read_bytes()).hexdigest(),
                info.st_nlink,
                info.st_ino,
                info.st_mtime_ns,
            )
        return out

    # --- H1 -----------------------------------------------------------
    def test_h1_imports_regular_files_and_writes_a_hub_cache_receipt(self):
        before = self.cache_fingerprint()
        receipt_path = self.run_import()
        self.assertEqual(receipt_path, FASTMLX_PULL.receipt_path_for(self.dest))
        for name, data in self.repo.content_by_name.items():
            target = self.dest / name
            self.assertFalse(target.is_symlink(), name)
            self.assertTrue(target.is_file(), name)
            self.assertEqual(target.read_bytes(), data)
            self.assertEqual(target.lstat().st_nlink, 1, name)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["repo_id"], self.repo.repo_id)
        self.assertEqual(receipt["revision"], self.repo.revision)
        self.assertEqual(receipt["acquisition"], "hub-cache")
        self.assertEqual(receipt["hub_cache_dir"], str(self.hub))
        self.assertEqual(receipt["dest"], str(self.dest.absolute()))
        self.assertEqual(receipt["total_files"], 3)
        self.assertEqual(
            receipt["total_bytes"],
            sum(len(d) for d in self.repo.content_by_name.values()),
        )
        self.assertEqual(receipt["ignored_local_paths"], [])
        for name, data in self.repo.content_by_name.items():
            self.assertEqual(receipt["files"][name]["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(receipt["files"][name]["size"], len(data))
        # Only the dest and its receipt exist; no staging dir is left.
        self.assertEqual(
            sorted(path.name for path in self.work.iterdir()),
            ["model", "model.pull-receipt.json"],
        )
        # The cache was only read.
        self.assertEqual(self.cache_fingerprint(), before)
        self.acquire.assert_not_called()
        self.opener.assert_not_called()

    def test_h1_nested_manifest_names_import_into_subdirectories(self):
        nested = NestedRepo()
        self.repo = nested
        self.fetch_api.return_value = (nested.document, b"")
        self.repo_dir = self.hub / ("models--" + nested.repo_id.replace("/", "--"))
        self.blobs = self.repo_dir / "blobs"
        self.snapshot = self.repo_dir / "snapshots" / nested.revision
        # Fresh hub: drop the default fixture's tree, build the nested one.
        import shutil as _shutil

        _shutil.rmtree(self.hub)
        self.build_hub()
        self.run_import()
        for name, data in nested.content_by_name.items():
            self.assertEqual((self.dest / name).read_bytes(), data)
            self.assertFalse((self.dest / name).is_symlink())

    # --- H2 -----------------------------------------------------------
    def test_h2_missing_file_is_named_and_nothing_is_left(self):
        (self.snapshot / "README.md").unlink()
        with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
            self.run_import()
        self.assertIn("README.md", str(ctx.exception))
        self.assert_nothing_written()

    def test_h2_dangling_link_is_named_and_nothing_is_left(self):
        self.blob_for["config.json"].unlink()
        with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
            self.run_import()
        self.assertIn("config.json", str(ctx.exception))
        self.assert_nothing_written()

    def test_h2_later_failure_removes_the_staging_dir_it_created(self):
        # model.safetensors is verified after the earlier files were copied;
        # corrupt it so the failure happens mid-way, not at the first file.
        blob = self.blob_for["model.safetensors"]
        blob.write_bytes(b"Z" * blob.stat().st_size)
        with self.assertRaises(FASTMLX_PULL.PullError):
            self.run_import()
        self.assert_nothing_written()

    # --- H3 -----------------------------------------------------------
    def test_h3_same_size_different_bytes_is_a_content_mismatch(self):
        for name in ("model.safetensors", "config.json"):
            with self.subTest(name=name):
                blob = self.blob_for[name]
                original = blob.read_bytes()
                blob.write_bytes(b"Z" * len(original))
                try:
                    with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
                        self.run_import()
                finally:
                    blob.write_bytes(original)
                self.assertIn("content mismatch", str(ctx.exception))
                self.assertIn(name, str(ctx.exception))
                self.assert_nothing_written()

    def test_h3_wrong_size_is_refused(self):
        blob = self.blob_for["config.json"]
        blob.write_bytes(blob.read_bytes() + b"x")
        with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
            self.run_import()
        self.assertIn("size mismatch", str(ctx.exception))
        self.assert_nothing_written()

    # --- H4 -----------------------------------------------------------
    def test_h4_link_escaping_blobs_is_refused(self):
        outside = self.root / "outside"
        outside.mkdir()
        evil = outside / "config.json"
        evil.write_bytes(self.repo.content_by_name["config.json"])  # correct bytes
        link = self.snapshot / "config.json"
        link.unlink()
        os.symlink(evil, link)
        with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
            self.run_import()
        self.assertIn("config.json", str(ctx.exception))
        self.assertIn("blobs", str(ctx.exception))
        self.assert_nothing_written()

    def test_h4_link_into_another_repos_blobs_is_refused(self):
        other_blobs = self.hub / "models--other--repo" / "blobs"
        other_blobs.mkdir(parents=True)
        other = other_blobs / "abc"
        other.write_bytes(self.repo.content_by_name["config.json"])
        link = self.snapshot / "config.json"
        link.unlink()
        os.symlink(other, link)
        with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
            self.run_import()
        self.assertIn("config.json", str(ctx.exception))
        self.assert_nothing_written()

    def test_h4_a_plain_file_in_the_snapshot_is_refused(self):
        link = self.snapshot / "config.json"
        link.unlink()
        link.write_bytes(self.repo.content_by_name["config.json"])
        with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
            self.run_import()
        self.assertIn("config.json", str(ctx.exception))
        self.assert_nothing_written()

    def test_h4_path_traversal_manifest_name_is_refused(self):
        entries = [
            {
                "name": "../escape.json",
                "size": 1,
                "blob_id": "0" * 40,
                "lfs_sha256": None,
            }
        ]
        with mock.patch.object(DOWNLOADER, "validated_entries", return_value=entries):
            with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
                self.run_import()
        self.assertIn("escape.json", str(ctx.exception))
        self.assert_nothing_written()

    # --- H5 -----------------------------------------------------------
    def test_h5_absent_snapshot_refuses_and_suggests_a_plain_pull(self):
        import shutil as _shutil

        _shutil.rmtree(self.repo_dir / "snapshots")
        with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
            self.run_import()
        message = str(ctx.exception)
        self.assertIn(self.repo.revision, message)
        self.assertIn("fastmlx pull", message)
        self.assertIn("--dest", message)
        self.assert_nothing_written()

    def test_h5_absent_hub_dir_refuses_and_suggests_a_plain_pull(self):
        with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
            self.run_import(hub=self.root / "no-such-hub")
        self.assertIn("fastmlx pull", str(ctx.exception))
        self.assert_nothing_written()

    def test_h5_adopt_with_from_hub_cache_is_a_usage_error(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as ctx:
            FASTMLX_PULL.main(
                [
                    f"{self.repo.repo_id}@{self.repo.revision}",
                    "--dest",
                    str(self.dest),
                    "--adopt",
                    "--from-hub-cache",
                    str(self.hub),
                ]
            )
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("not allowed with", stderr.getvalue())
        self.fetch_api.assert_not_called()
        self.assert_nothing_written()

    # --- fresh-destination refusals ------------------------------------
    def test_existing_dest_is_refused_and_left_untouched(self):
        self.dest.mkdir()
        (self.dest / "keep.txt").write_text("mine")
        with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
            self.run_import()
        self.assertIn("already exists", str(ctx.exception))
        self.assertEqual((self.dest / "keep.txt").read_text(), "mine")
        self.assertEqual(
            sorted(path.name for path in self.work.iterdir()), ["model"]
        )

    def test_existing_receipt_is_never_overwritten(self):
        receipt = FASTMLX_PULL.receipt_path_for(self.dest)
        receipt.write_text("old")
        with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
            self.run_import()
        self.assertIn("existing pull receipt", str(ctx.exception))
        self.assertEqual(receipt.read_text(), "old")
        self.assertFalse(self.dest.exists())

    # --- H6 -----------------------------------------------------------
    def test_h6_hub_dir_precedence(self):
        resolve = FASTMLX_PULL.resolve_hub_cache_dir
        env = {"HF_HUB_CACHE": "/a/hubcache", "HF_HOME": "/b/home", "HOME": "/c"}
        self.assertEqual(resolve("/x/explicit", env), Path("/x/explicit"))
        self.assertEqual(resolve("", env), Path("/a/hubcache"))
        self.assertEqual(resolve(None, env), Path("/a/hubcache"))
        del env["HF_HUB_CACHE"]
        self.assertEqual(resolve("", env), Path("/b/home/hub"))
        del env["HF_HOME"]
        self.assertEqual(resolve("", env), Path("/c/.cache/huggingface/hub"))
        # Empty env values do not count as set.
        self.assertEqual(
            resolve("", {"HF_HUB_CACHE": "", "HF_HOME": "", "HOME": "/c"}),
            Path("/c/.cache/huggingface/hub"),
        )

    def test_h6_cli_flag_without_value_uses_the_environment(self):
        with mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(self.hub)}):
            FASTMLX_PULL.main(
                [
                    f"{self.repo.repo_id}@{self.repo.revision}",
                    "--dest",
                    str(self.dest),
                    "--from-hub-cache",
                ]
            )
        receipt = json.loads(
            FASTMLX_PULL.receipt_path_for(self.dest).read_text(encoding="utf-8")
        )
        self.assertEqual(receipt["acquisition"], "hub-cache")
        self.assertEqual(receipt["hub_cache_dir"], str(self.hub))
        self.acquire.assert_not_called()

    # ---------------------------------------------------------------------
    # `--from-hub-cache` clones each blob copy-on-write where the volume supports
    # it (macOS clonefile(2)), else byte-copies it; never links. The clone
    # primitive is resolved by `_resolve_clone_primitive()` (None when
    # unavailable), so these tests inject fake primitives through that seam.
    # ---------------------------------------------------------------------
    def patch_primitive(self, primitive):
        # create=True keeps a missing seam a behavioural RED, not an AttributeError.
        patcher = mock.patch.object(
            FASTMLX_PULL,
            "_resolve_clone_primitive",
            return_value=primitive,
            create=True,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_capturing_stderr(self, dest=None):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            receipt_path = self.run_import(dest=dest)
        return receipt_path, stderr.getvalue()

    def progress_methods(self, stderr_text):
        methods = {}
        for line in stderr_text.splitlines():
            match = re.match(r"hub-cache (clone|copy) file=\d+/\d+ name=(.+)$", line)
            if match:
                methods[match.group(2)] = match.group(1)
        return methods

    def full_fingerprint(self):
        out = {}
        for path in sorted(self.blobs.iterdir()):
            info = path.lstat()
            out[path.name] = (
                path.read_bytes(),
                info.st_size,
                info.st_mtime_ns,
                info.st_nlink,
                info.st_ino,
            )
        return out

    @staticmethod
    def copying_primitive(calls=None):
        def clone(src, dst):
            if calls is not None:
                calls.append((src, dst))
            with open(src, "rb") as source, open(dst, "xb") as sink:
                sink.write(source.read())

        return clone

    # --- C1 -----------------------------------------------------------
    def test_c1_clone_path_imports_every_file_and_keeps_the_receipt_shape(self):
        calls = []
        self.patch_primitive(self.copying_primitive(calls))
        receipt_path, stderr_text = self.run_capturing_stderr()
        self.assertEqual(len(calls), 3)
        self.assertEqual(
            self.progress_methods(stderr_text),
            {name: "clone" for name in self.repo.content_by_name},
        )
        self.assertNotIn("hub-cache copy", stderr_text)
        for name, data in self.repo.content_by_name.items():
            self.assertEqual((self.dest / name).read_bytes(), data)

        # Same receipt as the byte-copy path, apart from the dest it names.
        copy_dest = self.work / "model-copy"
        self.patch_primitive(None)
        copy_receipt_path, _ = self.run_capturing_stderr(dest=copy_dest)
        cloned = json.loads(receipt_path.read_text(encoding="utf-8"))
        copied = json.loads(copy_receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(cloned["acquisition"], "hub-cache")
        self.assertEqual(sorted(cloned), sorted(copied))
        self.assertEqual(cloned.pop("dest"), str(self.dest.absolute()))
        self.assertEqual(copied.pop("dest"), str(copy_dest.absolute()))
        self.assertEqual(cloned, copied)

    # --- C2 -----------------------------------------------------------
    def test_c2_clone_failure_falls_back_to_a_byte_copy_for_that_file(self):
        for code in (errno.ENOTSUP, errno.EXDEV, errno.EPERM):
            with self.subTest(errno=errno.errorcode[code]):
                good = self.copying_primitive()

                def clone(src, dst, good=good, code=code):
                    if dst.endswith("config.json"):
                        raise OSError(code, os.strerror(code), dst)
                    good(src, dst)

                self.patch_primitive(clone)
                dest = self.work / f"model-{errno.errorcode[code]}"
                receipt_path, stderr_text = self.run_capturing_stderr(dest=dest)
                methods = self.progress_methods(stderr_text)
                self.assertEqual(methods["config.json"], "copy")
                self.assertEqual(methods["README.md"], "clone")
                self.assertEqual(methods["model.safetensors"], "clone")
                for name, data in self.repo.content_by_name.items():
                    self.assertEqual((dest / name).read_bytes(), data)
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                self.assertEqual(receipt["acquisition"], "hub-cache")

    def test_c2_a_partial_target_left_by_a_failed_clone_is_replaced(self):
        def clone(src, dst):
            with open(dst, "xb") as sink:
                sink.write(b"partial")
            raise OSError(errno.ENOTSUP, os.strerror(errno.ENOTSUP), dst)

        self.patch_primitive(clone)
        _, stderr_text = self.run_capturing_stderr()
        self.assertEqual(
            self.progress_methods(stderr_text),
            {name: "copy" for name in self.repo.content_by_name},
        )
        for name, data in self.repo.content_by_name.items():
            self.assertEqual((self.dest / name).read_bytes(), data)

    # --- C3 -----------------------------------------------------------
    def test_c3_no_clone_primitive_byte_copies_every_file(self):
        self.patch_primitive(None)
        _, stderr_text = self.run_capturing_stderr()
        self.assertEqual(
            self.progress_methods(stderr_text),
            {name: "copy" for name in self.repo.content_by_name},
        )
        self.assertNotIn("hub-cache clone", stderr_text)
        for name, data in self.repo.content_by_name.items():
            self.assertEqual((self.dest / name).read_bytes(), data)

    def test_c3_the_primitive_is_unavailable_off_macos(self):
        with mock.patch.object(sys, "platform", "linux"):
            self.assertIsNone(FASTMLX_PULL._resolve_clone_primitive())

    # --- C4 -----------------------------------------------------------
    def test_c4_eexist_refuses_and_leaves_nothing(self):
        good = self.copying_primitive()

        def clone(src, dst):
            if dst.endswith("config.json"):
                raise OSError(errno.EEXIST, os.strerror(errno.EEXIST), dst)
            good(src, dst)

        self.patch_primitive(clone)
        with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
            self.run_import()
        self.assertIn("config.json", str(ctx.exception))
        self.assert_nothing_written()
        self.assertFalse(self.dest.exists())
        self.assertFalse(FASTMLX_PULL.receipt_path_for(self.dest).exists())

    # --- C5 -----------------------------------------------------------
    def test_c5_imported_files_never_share_an_inode_with_the_cache(self):
        for label, primitive in (
            ("clone", self.copying_primitive()),
            ("copy", None),
        ):
            with self.subTest(method=label):
                self.patch_primitive(primitive)
                dest = self.work / f"model-{label}"
                self.run_capturing_stderr(dest=dest)
                for name in self.repo.content_by_name:
                    mine = os.stat(dest / name)
                    theirs = os.stat(self.blob_for[name])
                    self.assertNotEqual(mine.st_ino, theirs.st_ino, name)
                    self.assertEqual(mine.st_nlink, 1, name)

    def test_c5_a_hard_link_is_refused_and_leaves_nothing(self):
        def link(src, dst):
            os.link(src, dst)

        self.patch_primitive(link)
        with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
            self.run_import()
        self.assertIn("link", str(ctx.exception))
        self.assert_nothing_written()
        for blob in self.blob_for.values():
            self.assertEqual(blob.lstat().st_nlink, 1)

    def test_c5_the_inode_guard_refuses_a_link_without_the_verifiers_help(self):
        # The verifier also refuses multi-link files; stub it so the inode
        # guard is the only defence in this case.
        self.patch_primitive(lambda src, dst: os.link(src, dst))
        with mock.patch.object(
            FASTMLX_PULL, "_verify_adopted_entry", return_value={}
        ):
            with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
                self.run_import()
        self.assertIn("link to the cache blob", str(ctx.exception))
        self.assert_nothing_written()

    # --- C6 -----------------------------------------------------------
    def test_c6_the_cache_is_untouched_by_either_path(self):
        for label, primitive in (
            ("clone", self.copying_primitive()),
            ("copy", None),
        ):
            with self.subTest(method=label):
                self.patch_primitive(primitive)
                before = self.full_fingerprint()
                self.run_capturing_stderr(dest=self.work / f"model-{label}")
                self.assertEqual(self.full_fingerprint(), before)

    # --- C7 -----------------------------------------------------------
    def test_c7_a_clone_that_writes_wrong_bytes_is_caught_by_the_verifier(self):
        def clone(src, dst):
            with open(dst, "xb") as sink:
                sink.write(b"Z" * os.path.getsize(src))

        self.patch_primitive(clone)
        with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
            self.run_import()
        self.assertIn("content mismatch", str(ctx.exception))
        self.assert_nothing_written()

    def test_c7_a_clone_that_reports_success_without_writing_is_refused(self):
        self.patch_primitive(lambda src, dst: None)
        with self.assertRaises(FASTMLX_PULL.PullError):
            self.run_import()
        self.assert_nothing_written()

    # --- C8 -----------------------------------------------------------
    @unittest.skipUnless(sys.platform == "darwin", "clonefile(2) is macOS-only")
    def test_c8_a_real_clonefile_import_uses_the_clone_method(self):
        before = self.full_fingerprint()
        _, stderr_text = self.run_capturing_stderr()
        self.assertEqual(
            self.progress_methods(stderr_text),
            {name: "clone" for name in self.repo.content_by_name},
        )
        for name, data in self.repo.content_by_name.items():
            self.assertEqual((self.dest / name).read_bytes(), data)
            self.assertNotEqual(
                os.stat(self.dest / name).st_ino, os.stat(self.blob_for[name]).st_ino
            )
        self.assertEqual(self.full_fingerprint(), before)

    def test_h6_cli_explicit_dir_beats_the_environment(self):
        with mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(self.root / "nowhere")}):
            FASTMLX_PULL.main(
                [
                    f"{self.repo.repo_id}@{self.repo.revision}",
                    "--dest",
                    str(self.dest),
                    "--from-hub-cache",
                    str(self.hub),
                ]
            )
        self.assertTrue(FASTMLX_PULL.receipt_path_for(self.dest).exists())

    def test_cli_refusal_exits_1_and_writes_nothing(self):
        (self.snapshot / "README.md").unlink()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as ctx:
            FASTMLX_PULL.main(
                [
                    f"{self.repo.repo_id}@{self.repo.revision}",
                    "--dest",
                    str(self.dest),
                    "--from-hub-cache",
                    str(self.hub),
                ]
            )
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("README.md", stderr.getvalue())
        self.assert_nothing_written()

    # --- preflight ------------------------------------------------------
    def test_preflight_runs_before_any_copy_and_can_refuse(self):
        seen = []

        def refuse(entries):
            seen.append([entry["name"] for entry in entries])
            self.assertEqual(sorted(self.work.iterdir()), [])  # nothing copied yet
            raise FASTMLX_PULL.PreflightRefused("does not fit")

        with self.assertRaises(FASTMLX_PULL.PreflightRefused):
            self.run_import(preflight=refuse)
        self.assertEqual(len(seen), 1)
        self.assertEqual(sorted(seen[0]), sorted(self.repo.content_by_name))
        self.assert_nothing_written()

    def test_cli_kv_reserve_gib_builds_and_runs_the_preflight(self):
        calls = []

        def fake_builder(repo_id, revision, kv, context, host_use):
            calls.append((repo_id, revision, kv, context, host_use))

            def preflight(entries):
                raise FASTMLX_PULL.PreflightRefused("simulated does-not-fit")

            return preflight

        stderr = io.StringIO()
        with mock.patch.object(
            FASTMLX_PULL, "_build_fit_and_card_preflight", side_effect=fake_builder
        ), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as ctx:
            FASTMLX_PULL.main(
                [
                    f"{self.repo.repo_id}@{self.repo.revision}",
                    "--dest",
                    str(self.dest),
                    "--from-hub-cache",
                    str(self.hub),
                    "--kv-reserve-gib",
                    "0.5",
                ]
            )
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("simulated does-not-fit", stderr.getvalue())
        self.assertEqual(calls, [(self.repo.repo_id, self.repo.revision, 0.5, None, "shared")])
        self.assert_nothing_written()

    # --- F0: dest parent does not exist, real probe -----------------------
    def test_f0_import_into_a_missing_dest_parent_with_the_real_probe(self):
        dest = self.work / "not" / "yet" / "model"
        receipt_path = self.run_import(dest=dest)
        self.assertEqual(receipt_path, FASTMLX_PULL.receipt_path_for(dest))
        self.assertTrue(dest.is_dir())
        self.assertTrue(receipt_path.is_file())

    def test_f0_import_refusal_into_a_missing_dest_parent_creates_no_directories(self):
        dest = self.work / "not" / "yet" / "model"
        with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
            self.run_import(dest=dest, min_free_bytes=1 << 62)
        self.assertIn("insufficient free space", str(ctx.exception))
        self.assert_nothing_written()

    # --- F1: insufficient free space refuses, leaves nothing ---------------
    def test_f1_insufficient_free_space_refuses_and_leaves_nothing(self):
        needed = self.default_required()
        with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
            self.run_import(disk_usage_bytes=lambda path: needed - 1)
        message = str(ctx.exception)
        self.assertIn("insufficient free space", message)
        self.assertIn(str(needed), message)
        self.assertIn(str(needed - 1), message)
        self.assert_nothing_written()
        self.assertFalse(FASTMLX_PULL.receipt_path_for(self.dest).exists())

    def test_f1_exactly_the_floor_is_admitted(self):
        needed = self.default_required()
        self.run_import(disk_usage_bytes=lambda path: needed)
        self.assertTrue(FASTMLX_PULL.receipt_path_for(self.dest).is_file())

    def test_f1_cli_refusal_exits_1_with_the_reason_and_writes_nothing(self):
        stderr = io.StringIO()
        with mock.patch.object(
            DOWNLOADER, "probe_free_space_bytes", return_value=1
        ), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as ctx:
            FASTMLX_PULL.main(
                [
                    f"{self.repo.repo_id}@{self.repo.revision}",
                    "--dest",
                    str(self.dest),
                    "--from-hub-cache",
                    str(self.hub),
                ]
            )
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("insufficient free space", stderr.getvalue())
        self.assert_nothing_written()

    # --- F2: --min-free-bytes overrides the floor in both directions -------
    def test_f2_min_free_bytes_above_available_refuses(self):
        available = self.default_required() * 10  # far above the default floor
        with self.assertRaises(FASTMLX_PULL.PullError) as ctx:
            self.run_import(
                min_free_bytes=available + 1, disk_usage_bytes=lambda path: available
            )
        message = str(ctx.exception)
        self.assertIn("insufficient free space", message)
        self.assertIn(str(available + 1), message)
        self.assert_nothing_written()

    def test_f2_min_free_bytes_below_available_admits_despite_the_default_floor(self):
        available = self.default_required() - 1  # would refuse by default
        self.run_import(min_free_bytes=available, disk_usage_bytes=lambda path: available)
        self.assertTrue(FASTMLX_PULL.receipt_path_for(self.dest).is_file())

    def test_f2_cli_min_free_bytes_reaches_the_import(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as ctx:
            FASTMLX_PULL.main(
                [
                    f"{self.repo.repo_id}@{self.repo.revision}",
                    "--dest",
                    str(self.dest),
                    "--from-hub-cache",
                    str(self.hub),
                    "--min-free-bytes",
                    str(1 << 62),
                ]
            )
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("insufficient free space", stderr.getvalue())
        self.assert_nothing_written()

    # --- F3: --max-attempts has no meaning for an import -------------------
    def test_f3_explicit_max_attempts_with_from_hub_cache_is_a_usage_error(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as ctx:
            FASTMLX_PULL.main(
                [
                    f"{self.repo.repo_id}@{self.repo.revision}",
                    "--dest",
                    str(self.dest),
                    "--from-hub-cache",
                    str(self.hub),
                    "--max-attempts",
                    "3",
                ]
            )
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("--max-attempts", stderr.getvalue())
        self.fetch_api.assert_not_called()
        self.assert_nothing_written()

    # --- F4: ordering ------------------------------------------------------
    def test_f4_preflight_refusal_fires_before_the_probe_is_called(self):
        probe_calls = []

        def probe(path):
            probe_calls.append(path)
            return 1 << 62

        def refuse(entries):
            raise FASTMLX_PULL.PreflightRefused("does not fit")

        with self.assertRaises(FASTMLX_PULL.PreflightRefused):
            self.run_import(preflight=refuse, disk_usage_bytes=probe)
        self.assertEqual(probe_calls, [])
        self.assert_nothing_written()

    def test_f4_probe_runs_after_preflight_and_before_any_staging_or_copy(self):
        events = []
        work = self.work

        def preflight(entries):
            events.append("preflight")

        def probe(path):
            events.append("probe")
            self.assertEqual(path, work)
            self.assertEqual(sorted(work.iterdir()), [])  # no staging, no dest
            return 1 << 62

        self.run_import(preflight=preflight, disk_usage_bytes=probe)
        self.assertEqual(events, ["preflight", "probe"])

    def test_f4_probe_sees_the_nearest_existing_ancestor_of_a_missing_parent(self):
        seen = []
        dest = self.work / "not" / "yet" / "model"

        def probe(path):
            seen.append(path)
            return 1 << 62

        self.run_import(dest=dest, disk_usage_bytes=probe)
        self.assertEqual(seen, [self.work])

    # --- H7 -----------------------------------------------------------
    def test_h7_launcher_receipt_reader_resolves_repo_and_revision(self):
        self.run_import()
        launch_spec = importlib.util.spec_from_file_location(
            "fastmlx_launch",
            Path(__file__).resolve().parents[1] / "fastmlx_launch.py",
        )
        assert launch_spec is not None and launch_spec.loader is not None
        launch_module = importlib.util.module_from_spec(launch_spec)
        launch_spec.loader.exec_module(launch_module)
        receipt = launch_module._load_pull_receipt(self.dest)
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt["acquisition"], "hub-cache")
        args = argparse.Namespace(model_revision=None, model_repo=None)
        self.assertEqual(
            launch_module._resolve_model_revision(args, self.dest), self.repo.revision
        )
        self.assertEqual(
            launch_module._resolve_model_repo(args, self.dest), self.repo.repo_id
        )


# ---------------------------------------------------------------------
# `fastmlx pull --kv-reserve-gib N`: judge fit and the quality card from the
# manifest pull already fetched, BEFORE the free-space probe and any staging
# directory. No real network: `downloader.fetch_api`, `acquire` and the
# free-space probe are faked, the card store is a local manifest, and the
# host ceiling is pinned through the sizer's own environment inputs.
# ---------------------------------------------------------------------
_RECOMMEND_PATH = Path(__file__).resolve().parents[1] / "fastmlx_recommend.py"
_RECOMMEND_MODULE = None


def _recommend_module():
    """One shared instance of ``fastmlx_recommend`` loaded by file path (the
    same idiom the scripts use for siblings), only when a test asks."""
    global _RECOMMEND_MODULE
    if _RECOMMEND_MODULE is None:
        spec = importlib.util.spec_from_file_location("fastmlx_recommend", _RECOMMEND_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _RECOMMEND_MODULE = module
    return _RECOMMEND_MODULE


PREFLIGHT_PASS_REPO = "example/PassModel"
PREFLIGHT_PASS_CARD = "fixture-pass@test"
PREFLIGHT_NO_GO_REPO = "example/NoGoModel"
PREFLIGHT_NO_GO_CARD = "fixture-no-go@test"
PREFLIGHT_UNCARDED_REPO = "example/NoCardModel"
PREFLIGHT_SHA = "a" * 40
# kv 0.5 GiB over a 2 GiB ceiling (4096 MiB wired limit - 2 GiB margin).
PREFLIGHT_CEILING_BYTES = (4096 << 20) - (2 << 30)


def preflight_card_manifest() -> dict:
    return {
        "schema": "fast-mlx-quality-card-v1",
        "generatedAt": "2026-01-01T00:00:00Z",
        "cards": [
            {
                "id": PREFLIGHT_PASS_CARD,
                "model": {"repo": PREFLIGHT_PASS_REPO, "hfPin": "cafebabe"},
                "verdict": "PASS",
                "legible": {
                    "tier": "Reference",
                    "headline": "Matches the reference closely.",
                    "nextWordDrift": {"top1AgreementPct": 91.2},
                    "benefit": {"speedX": None, "speedXStatus": "not-measured"},
                },
            },
            {
                "id": PREFLIGHT_NO_GO_CARD,
                "model": {"repo": PREFLIGHT_NO_GO_REPO, "hfPin": "deadbeef"},
                "verdict": "NO_GO",
                "legible": {
                    "tier": "Noticeable",
                    "headline": "About 1 word in 6 differs from the reference model.",
                },
            },
        ],
    }


def preflight_document(repo: str, sha: str, files: dict) -> dict:
    return {
        "id": repo,
        "sha": sha,
        "private": False,
        "siblings": [
            {"rfilename": name, "size": size, "blobId": "b" * 40}
            for name, size in files.items()
        ],
    }


SMALL_PACK = {
    "config.json": 10,
    "model-00001-of-00002.safetensors": 1000,
    "model-00002-of-00002.safetensors": 2000,
}


class PullPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.dest = self.work / "model"
        self.cards_path = self.root / "quality-guides.json"
        self.cards_path.write_text(json.dumps(preflight_card_manifest()), encoding="utf-8")
        self.recommend = _recommend_module()
        self.document = None
        self.files = dict(SMALL_PACK)
        self.repo = PREFLIGHT_PASS_REPO

        env = mock.patch.dict(
            os.environ,
            {"FASTMLX_WIRED_LIMIT_MIB": "4096", "FASTMLX_WIRED_MARGIN_GIB": "2"},
        )
        env.start()
        self.addCleanup(env.stop)

        self.fetch_api = mock.patch.object(
            DOWNLOADER, "fetch_api", side_effect=self._fake_fetch
        ).start()
        self.acquire = mock.patch.object(
            DOWNLOADER, "acquire", side_effect=self._fake_acquire
        ).start()
        self.probe = mock.patch.object(
            DOWNLOADER, "probe_free_space_bytes", return_value=1 << 60
        ).start()
        self.loader = mock.patch.object(
            FASTMLX_PULL, "_load_recommend_module", side_effect=_recommend_module
        ).start()
        self.store = mock.patch.object(
            self.recommend.launch,
            "resolve_quality_card_store",
            return_value=(self.cards_path, "explicit", []),
        ).start()
        self.addCleanup(mock.patch.stopall)

    def _fake_fetch(self, repo_id, revision):
        return preflight_document(repo_id, revision, self.files), b""

    def _fake_acquire(self, namespace):
        output = Path(namespace.output)
        output.mkdir()
        for name, size in self.files.items():
            (output / name).write_bytes(b"x" * size)

    def run_main(self, *extra, repo=None, revision=PREFLIGHT_SHA):
        argv = [f"{repo or self.repo}@{revision}", "--dest", str(self.dest), *extra]
        stdout, stderr = io.StringIO(), io.StringIO()
        code = None
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                FASTMLX_PULL.main(argv)
            except SystemExit as exit_:
                code = exit_.code
        return code, stdout.getvalue(), stderr.getvalue()

    def assert_nothing_written(self):
        self.assertEqual(sorted(self.work.iterdir()), [])
        self.acquire.assert_not_called()
        self.probe.assert_not_called()

    # --- A1 -----------------------------------------------------------
    def test_does_not_fit_refuses_before_probe_staging_or_download(self):
        self.files = {"config.json": 10, "model.safetensors": 3 << 30}
        code, _out, err = self.run_main("--kv-reserve-gib", "0.5")
        self.assertEqual(code, 1)
        self.assertIn("exceeds ceiling", err)  # the sizer's own reason
        self.assertIn("over by 1610612736 bytes", err)
        self.assertIn("drop --kv-reserve-gib", err)
        self.assert_nothing_written()
        self.fetch_api.assert_called_once()

    # --- A2 -----------------------------------------------------------
    def test_pass_card_prints_one_verdict_line_then_downloads_and_writes_receipt(self):
        code, _out, err = self.run_main("--kv-reserve-gib", "0.5")
        self.assertIsNone(code, err)
        verdict_lines = [line for line in err.splitlines() if PREFLIGHT_PASS_CARD in line]
        self.assertEqual(len(verdict_lines), 1, err)
        self.assertIn("PASS", verdict_lines[0])
        self.assertNotIn("--accept-quality", err)
        self.fetch_api.assert_called_once()  # the manifest is never fetched twice
        self.acquire.assert_called_once()
        receipt = json.loads(
            FASTMLX_PULL.receipt_path_for(self.dest).read_text(encoding="utf-8")
        )
        self.assertEqual(receipt["total_files"], 3)
        self.assertTrue(self.dest.is_dir())

    # --- A3 -----------------------------------------------------------
    def test_no_go_card_names_accept_quality_and_still_downloads(self):
        code, _out, err = self.run_main(
            "--kv-reserve-gib", "0.5", repo=PREFLIGHT_NO_GO_REPO
        )
        self.assertIsNone(code, err)
        self.assertIn(f"--accept-quality {PREFLIGHT_NO_GO_CARD}", err)
        self.assertIn("NO_GO", err)
        self.acquire.assert_called_once()
        self.assertTrue(FASTMLX_PULL.receipt_path_for(self.dest).exists())

    def test_uncarded_pack_says_unmeasured_and_still_downloads(self):
        code, _out, err = self.run_main(
            "--kv-reserve-gib", "0.5", repo=PREFLIGHT_UNCARDED_REPO
        )
        self.assertIsNone(code, err)
        self.assertIn("unmeasured", err)
        self.assertNotIn("--accept-quality", err)
        self.acquire.assert_called_once()
        self.assertTrue(FASTMLX_PULL.receipt_path_for(self.dest).exists())

    # --- A4 -----------------------------------------------------------
    def test_gguf_only_manifest_refuses_naming_the_reason_and_writes_nothing(self):
        self.files = {"config.json": 10, "model.gguf": 5000}
        code, _out, err = self.run_main("--kv-reserve-gib", "0.5")
        self.assertEqual(code, 1)
        self.assertIn("no .safetensors files found", err)
        self.assertIn("drop --kv-reserve-gib", err)
        self.assert_nothing_written()

    def test_big_unnamed_non_safetensors_file_refuses_and_writes_nothing(self):
        self.files = {"a.safetensors": 10, "weights.bin": 2 << 30}
        code, _out, err = self.run_main("--kv-reserve-gib", "0.5")
        self.assertEqual(code, 1)
        self.assertIn("weights.bin", err)
        self.assert_nothing_written()

    def test_default_card_store_that_refuses_to_load_is_exit_1_and_writes_nothing(self):
        self.store.side_effect = self.recommend.launch.LaunchRefusal(
            3, "pulled card store /x refused: simulated"
        )
        code, _out, err = self.run_main("--kv-reserve-gib", "0.5")
        self.assertEqual(code, 1)
        self.assertIn("simulated", err)
        self.assert_nothing_written()

    def test_default_card_store_that_is_not_a_manifest_is_exit_1_and_writes_nothing(self):
        junk = self.root / "junk.json"
        junk.write_text("not json", encoding="utf-8")
        self.store.return_value = (junk, "default", [])
        code, _out, err = self.run_main("--kv-reserve-gib", "0.5")
        self.assertEqual(code, 1)
        self.assertIn("quality-card", err)
        self.assert_nothing_written()

    # --- A5 -----------------------------------------------------------
    def test_without_the_flag_pull_is_unchanged_and_never_loads_recommend(self):
        self.loader.side_effect = AssertionError("recommend must not load without the flag")
        code, _out, err = self.run_main()
        self.assertIsNone(code, err)
        self.loader.assert_not_called()
        self.fetch_api.assert_called_once()
        self.acquire.assert_called_once()
        self.assertNotIn("quality card", err)
        self.assertTrue(FASTMLX_PULL.receipt_path_for(self.dest).exists())

    # --- A6 -----------------------------------------------------------
    def test_adopt_with_the_check_flag_is_a_usage_error(self):
        self.dest.mkdir()
        code, _out, err = self.run_main("--adopt", "--kv-reserve-gib", "0.5")
        self.assertEqual(code, 2)
        self.assertIn("--adopt cannot be combined with --kv-reserve-gib", err)
        self.fetch_api.assert_not_called()
        self.loader.assert_not_called()

    def test_context_or_host_use_without_the_flag_is_a_usage_error(self):
        for extra in (["--context", "4096"], ["--host-use", "dedicated-serving"]):
            with self.subTest(extra=extra):
                code, _out, err = self.run_main(*extra)
                self.assertEqual(code, 2)
                self.assertIn("--kv-reserve-gib", err)
        self.fetch_api.assert_not_called()
        self.acquire.assert_not_called()
        self.assertEqual(sorted(self.work.iterdir()), [])

    def test_host_use_and_context_reach_the_classifier(self):
        spy = mock.patch.object(
            self.recommend,
            "classify_pinned_entries",
            wraps=self.recommend.classify_pinned_entries,
        ).start()
        code, _out, err = self.run_main(
            "--kv-reserve-gib", "0.5", "--context", "4096", "--host-use", "dedicated-serving"
        )
        self.assertIsNone(code, err)
        kwargs = spy.call_args.kwargs
        self.assertEqual(kwargs["host_use"], "dedicated-serving")
        self.assertEqual(kwargs["context"], 4096)
        self.assertEqual(kwargs["kv_reserve_gib"], 0.5)

    # --- A7 -----------------------------------------------------------
    def test_preflight_sizing_equals_build_pinned_row_sizing_for_the_same_manifest(self):
        rows = []
        real = self.recommend.classify_pinned_entries

        def capture(*args, **kwargs):
            row = real(*args, **kwargs)
            rows.append(row)
            return row

        mock.patch.object(self.recommend, "classify_pinned_entries", side_effect=capture).start()
        code, _out, err = self.run_main("--kv-reserve-gib", "0.5")
        self.assertIsNone(code, err)
        self.assertEqual(len(rows), 1)
        pull_row = rows[0]

        ref = f"{PREFLIGHT_PASS_REPO}@{PREFLIGHT_SHA}"
        document = preflight_document(PREFLIGHT_PASS_REPO, PREFLIGHT_SHA, self.files)
        with mock.patch.object(
            self.recommend.downloader, "fetch_api", return_value=(document, b"")
        ):
            recommend_row = self.recommend.build_pinned_row(
                ref=ref,
                cards=self.recommend.launch._inspect_quality_card_store(self.cards_path)[1],
                fit_check_bin=None,
                host_use="shared",
                context=None,
                fit_check_args=[],
                kv_reserve_gib=0.5,
            )
        self.assertEqual(pull_row["sizing"], recommend_row["sizing"])
        self.assertEqual(pull_row, recommend_row)
        # Independent expected values, so a classifier that ignores the
        # manifest cannot pass by agreeing with itself.
        self.assertEqual(pull_row["sizing"]["weights_bytes"], 3000)
        self.assertEqual(pull_row["sizing"]["kv_reserve_bytes"], 1 << 29)
        self.assertEqual(pull_row["sizing"]["total_bytes"], 3000 + (1 << 29))
        self.assertEqual(pull_row["sizing"]["ceiling_bytes"], PREFLIGHT_CEILING_BYTES)


if __name__ == "__main__":
    unittest.main()

