import argparse
import hashlib
import importlib.util
import json
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


if __name__ == "__main__":
    unittest.main()
