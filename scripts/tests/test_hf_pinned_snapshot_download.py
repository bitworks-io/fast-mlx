import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import unquote


SCRIPT = (
    Path(__file__).resolve().parents[1] / "hf_pinned_snapshot_download.py"
)
REVISION = "d" * 40
REPO_ID = "example/Test-Model"

SPEC = importlib.util.spec_from_file_location(
    "hf_pinned_snapshot_download", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
DOWNLOADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOADER)


# The downloader publishes with renameatx_np(RENAME_EXCL), which exists only on
# macOS; elsewhere it refuses by design. These cases exercise that real rename,
# so they skip off macOS. Public CI runs them in its macOS job and fails on any
# skip there.
REQUIRES_MACOS_EXCLUSIVE_RENAME = unittest.skipUnless(
    sys.platform == "darwin", "exclusive rename (renameatx_np) is macOS-only"
)


def git_blob_sha1(data: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {len(data)}\0".encode())
    digest.update(data)
    return digest.hexdigest()


class FakeResponse:
    """A minimal stand-in for the object returned by opener.open(...)."""

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
    ever touching the network. Injected in place of ``https_opener()``.
    """

    def __init__(
        self,
        api_bytes: bytes,
        content_by_name: dict,
        serve_overrides: dict = None,
        flaky_name: str = None,
        deny_resolve: bool = False,
    ):
        self.api_bytes = api_bytes
        self.content_by_name = content_by_name
        self.serve_overrides = serve_overrides or {}
        self.flaky_name = flaky_name
        self.deny_resolve = deny_resolve
        self.attempt_counts = {}
        self.resolve_calls = []

    def open(self, request, timeout=60):
        url = request.full_url
        if url.startswith("https://huggingface.co/api/models/"):
            return FakeResponse(self.api_bytes)
        marker = "/resolve/"
        index = url.index(marker)
        after = url[index + len(marker) :]
        rev_and_name = after.split("?", 1)[0]
        _, _, encoded_name = rev_and_name.partition("/")
        name = unquote(encoded_name)
        self.resolve_calls.append(name)
        if self.deny_resolve:
            raise AssertionError(
                "a download was issued when none was expected"
            )
        data = self.serve_overrides.get(name, self.content_by_name[name])
        if name == self.flaky_name:
            count = self.attempt_counts.get(name, 0) + 1
            self.attempt_counts[name] = count
            if count == 1:
                # Short read: fewer bytes than the declared size, which
                # trips the TransientTransferError retry path.
                data = data[: len(data) // 2]
        return FakeResponse(data)


class SyntheticRepo:
    """A 3-file public-repo fixture: one small text file, one small JSON
    config, and one LFS-tracked "shard" file, mirroring a real snapshot.
    """

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

    def resync_api_bytes(self) -> None:
        """Call after mutating ``self.document`` in a test."""
        self.api_bytes = self._serialize()


class HFPinnedSnapshotDownloadTests(unittest.TestCase):
    def test_exclusive_rename_refuses_where_the_platform_lacks_it(self) -> None:
        # Runs on every platform: without renameatx_np the downloader must refuse
        # rather than fall back to a replacing rename.
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            destination = Path(directory) / "destination"
            source.write_bytes(b"x")
            with mock.patch.object(DOWNLOADER.ctypes, "CDLL", return_value=object()):
                with self.assertRaisesRegex(
                    DOWNLOADER.AcquisitionError, "exclusive rename is unavailable"
                ):
                    DOWNLOADER.exclusive_rename(source, destination)
            self.assertTrue(source.exists())
            self.assertFalse(destination.exists())

    def opener_for(self, repo: SyntheticRepo, **kwargs) -> FakeOpener:
        return FakeOpener(repo.api_bytes, dict(repo.content_by_name), **kwargs)

    def acquire(
        self,
        repo: SyntheticRepo,
        output: Path,
        manifest: Path,
        opener: FakeOpener,
        reuse_verified_from: Path = None,
    ):
        args = argparse.Namespace(
            repo_id=repo.repo_id,
            revision=repo.revision,
            output=output,
            source_api_manifest=manifest,
            plan_only=False,
            include_prefix=None,
            reuse_verified_from=reuse_verified_from,
        )
        with mock.patch.object(DOWNLOADER, "https_opener", return_value=opener):
            DOWNLOADER.acquire(args)

    # ------------------------------------------------------------------
    # 1. Happy path
    # ------------------------------------------------------------------
    @REQUIRES_MACOS_EXCLUSIVE_RENAME
    def test_happy_path_publishes_exact_files_and_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            output = root / "model"
            manifest = root / "source-api.json"
            opener = self.opener_for(repo)

            self.acquire(repo, output, manifest, opener)

            for name, data in repo.content_by_name.items():
                published = output / name
                self.assertTrue(published.is_file(), name)
                self.assertEqual(published.read_bytes(), data)

            self.assertEqual(
                json.loads(manifest.read_text(encoding="utf-8")),
                repo.document,
            )
            tree_path = (
                output
                / ".cache/huggingface/trees"
                / f"{REVISION}.json"
            )
            tree = json.loads(tree_path.read_text(encoding="utf-8"))
            self.assertEqual(set(tree["files"]), set(repo.content_by_name))
            for name, data in repo.content_by_name.items():
                metadata_path = (
                    output
                    / ".cache/huggingface/download"
                    / f"{name}.metadata"
                )
                self.assertTrue(metadata_path.is_file(), name)
                lines = metadata_path.read_text(encoding="utf-8").splitlines()
                self.assertEqual(lines[0], REVISION)

    # ------------------------------------------------------------------
    # 2. Revision mismatch
    # ------------------------------------------------------------------
    def test_rejects_source_api_revision_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            repo.document["sha"] = "a" * 40
            repo.resync_api_bytes()
            output = root / "model"
            manifest = root / "source-api.json"
            opener = self.opener_for(repo, deny_resolve=True)

            with self.assertRaises(DOWNLOADER.AcquisitionError) as ctx:
                self.acquire(repo, output, manifest, opener)

            self.assertIn("revision identity mismatch", str(ctx.exception))
            self.assertFalse(output.exists())
            self.assertEqual(opener.resolve_calls, [])

    # ------------------------------------------------------------------
    # 3. Corrupted byte content
    # ------------------------------------------------------------------
    @REQUIRES_MACOS_EXCLUSIVE_RENAME
    def test_rejects_corrupted_content_and_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            output = root / "model"
            manifest = root / "source-api.json"
            tampered = repo.content_by_name["config.json"].replace(
                b"world", b"WORLD"
            )
            self.assertEqual(
                len(tampered), len(repo.content_by_name["config.json"])
            )
            opener = self.opener_for(
                repo, serve_overrides={"config.json": tampered}
            )

            with self.assertRaises(DOWNLOADER.AcquisitionError) as ctx:
                self.acquire(repo, output, manifest, opener)

            self.assertIn("identity mismatch", str(ctx.exception))
            self.assertFalse(output.exists())

    # ------------------------------------------------------------------
    # 4. Unsafe rfilename values
    # ------------------------------------------------------------------
    def test_rejects_unsafe_repository_paths(self):
        unsafe_names = ["../escape", "/etc/passwd", ".cache/x"]
        for unsafe_name in unsafe_names:
            with self.subTest(unsafe_name=unsafe_name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    repo = SyntheticRepo()
                    repo.document["siblings"].append(
                        {
                            "rfilename": unsafe_name,
                            "size": 1,
                            "blobId": git_blob_sha1(b"x"),
                        }
                    )
                    repo.resync_api_bytes()
                    output = root / "model"
                    manifest = root / "source-api.json"
                    opener = self.opener_for(repo, deny_resolve=True)

                    with self.assertRaises(DOWNLOADER.AcquisitionError) as ctx:
                        self.acquire(repo, output, manifest, opener)

                    message = str(ctx.exception).lower()
                    self.assertTrue(
                        "unsafe" in message or "reserved" in message,
                        message,
                    )
                    self.assertFalse(output.exists())

    # ------------------------------------------------------------------
    # 5. On failure, output is never created; a staging dir remains
    # ------------------------------------------------------------------
    def test_failure_leaves_only_acquiring_staging_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            output = root / "model"
            manifest = root / "source-api.json"
            tampered = repo.content_by_name["model.safetensors"] + b"x"
            opener = self.opener_for(
                repo, serve_overrides={"model.safetensors": tampered}
            )

            with self.assertRaises(DOWNLOADER.AcquisitionError):
                self.acquire(repo, output, manifest, opener)

            self.assertFalse(output.exists())
            staging = list(root.glob(f".{output.name}.acquiring-*"))
            self.assertEqual(len(staging), 1, staging)
            self.assertTrue(staging[0].is_dir())

    # ------------------------------------------------------------------
    # 6. FIX A: no *.partial residue survives into a published acquisition
    # ------------------------------------------------------------------
    @REQUIRES_MACOS_EXCLUSIVE_RENAME
    def test_retry_leaves_no_partial_residue_in_published_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            output = root / "model"
            manifest = root / "source-api.json"
            opener = self.opener_for(repo, flaky_name="model.safetensors")

            with mock.patch.object(DOWNLOADER.time, "sleep", return_value=None):
                self.acquire(repo, output, manifest, opener)

            self.assertEqual(opener.attempt_counts["model.safetensors"], 2)
            published = output / "model.safetensors"
            self.assertEqual(
                published.read_bytes(),
                repo.content_by_name["model.safetensors"],
            )
            partial_files = list(output.rglob("*.partial"))
            self.assertEqual(partial_files, [])

    @REQUIRES_MACOS_EXCLUSIVE_RENAME
    def test_retry_unlinks_the_failed_attempt_file_immediately(self):
        # Narrower unit check on download_entry itself: after a transient
        # failure, the failed attempt file must not still be on disk once
        # the retry loop moves on, independent of whether the eventual
        # acquisition as a whole succeeds.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_root = root / ".cache/huggingface/download"
            cache_root.mkdir(parents=True)
            repo = SyntheticRepo()
            data = repo.content_by_name["config.json"]
            entry = {
                "name": "config.json",
                "size": len(data),
                "blob_id": git_blob_sha1(data),
                "lfs_sha256": None,
                "lfs_size": None,
            }
            opener = self.opener_for(repo, flaky_name="config.json")

            with mock.patch.object(
                DOWNLOADER, "https_opener", return_value=opener
            ), mock.patch.object(DOWNLOADER.time, "sleep", return_value=None):
                DOWNLOADER.download_entry(
                    repo.repo_id, repo.revision, root, cache_root, entry, 1, 1
                )

            transfer_root = (
                root
                / ".cache/huggingface/failed-downloads"
                / hashlib.sha256(b"config.json").hexdigest()
            )
            self.assertEqual(
                list(transfer_root.glob("*.partial")), []
            )
            first_attempt = transfer_root / "attempt-1.partial"
            self.assertFalse(first_attempt.exists())

    # ------------------------------------------------------------------
    # FIX B: free-space preflight
    # ------------------------------------------------------------------
    def test_refuses_when_free_space_preflight_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            output = root / "model"
            manifest = root / "source-api.json"
            opener = self.opener_for(repo, deny_resolve=True)

            with mock.patch.object(
                DOWNLOADER, "probe_free_space_bytes", return_value=1
            ):
                with self.assertRaises(DOWNLOADER.AcquisitionError) as ctx:
                    self.acquire(repo, output, manifest, opener)

            self.assertIn("insufficient free space", str(ctx.exception))
            self.assertFalse(output.exists())
            self.assertEqual(opener.resolve_calls, [])
            staging = list(root.glob(f".{output.name}.acquiring-*"))
            self.assertEqual(staging, [])

    def test_free_space_multiplier_is_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            output = root / "model"
            manifest = root / "source-api.json"
            total_bytes = sum(len(data) for data in repo.content_by_name.values())
            opener = self.opener_for(repo, deny_resolve=True)

            just_under_required = int(
                total_bytes * DOWNLOADER.FREE_SPACE_SAFETY_MULTIPLIER
            ) - 1

            with mock.patch.object(
                DOWNLOADER,
                "probe_free_space_bytes",
                return_value=just_under_required,
            ):
                with self.assertRaises(DOWNLOADER.AcquisitionError):
                    self.acquire(repo, output, manifest, opener)

    # ------------------------------------------------------------------
    # v5: --reuse-verified-from
    # ------------------------------------------------------------------
    def test_parse_args_reuse_verified_from_flag_defaults_to_none_and_is_wired(self):
        base = ["prog", "--repo-id", REPO_ID, "--revision", REVISION, "--plan-only"]
        with mock.patch.object(sys, "argv", base):
            args = DOWNLOADER.parse_args()
        self.assertIsNone(args.reuse_verified_from)

        with mock.patch.object(
            sys, "argv", base + ["--reuse-verified-from", "/tmp/some-stage"]
        ):
            args = DOWNLOADER.parse_args()
        self.assertEqual(args.reuse_verified_from, Path("/tmp/some-stage"))

    @REQUIRES_MACOS_EXCLUSIVE_RENAME
    def test_reuse_verified_from_moves_only_the_fully_verified_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            extra_data = b"extra-tokenizer-bytes-value"
            repo.content_by_name["extra.txt"] = extra_data
            repo.document["siblings"].append(
                {
                    "rfilename": "extra.txt",
                    "size": len(extra_data),
                    "blobId": git_blob_sha1(extra_data),
                }
            )
            repo.resync_api_bytes()

            output = root / "model"
            manifest = root / "source-api.json"
            opener = self.opener_for(repo)

            reuse_root = root / "preserved-stage"
            reuse_root.mkdir()

            # 1. GOOD: exact bytes -> must be reused (moved, not downloaded).
            good_data = repo.content_by_name["README.md"]
            (reuse_root / "README.md").write_bytes(good_data)
            os.chmod(reuse_root / "README.md", 0o444)
            candidate_identity = (reuse_root / "README.md").stat()

            # 2. Same size, wrong bytes -> hash mismatch -> must be downloaded.
            real_config = repo.content_by_name["config.json"]
            corrupt_config = real_config[:-1] + (
                b"!" if real_config[-1:] != b"!" else b"?"
            )
            self.assertEqual(len(corrupt_config), len(real_config))
            (reuse_root / "config.json").write_bytes(corrupt_config)

            # 3. Truncated (short) -> size mismatch -> must be downloaded.
            real_shard = repo.content_by_name["model.safetensors"]
            (reuse_root / "model.safetensors").write_bytes(real_shard[:10])

            # 4. Symlink to exactly-correct bytes -> refused for being a
            #    symlink (not a regular file), independent of its content
            #    matching -> must be downloaded.
            symlink_target = reuse_root / "extra-target.txt"
            symlink_target.write_bytes(extra_data)
            (reuse_root / "extra.txt").symlink_to(symlink_target)

            corrupt_bytes_before = (reuse_root / "config.json").read_bytes()
            truncated_bytes_before = (reuse_root / "model.safetensors").read_bytes()
            symlink_target_before = os.readlink(reuse_root / "extra.txt")

            self.acquire(repo, output, manifest, opener, reuse_verified_from=reuse_root)

            # Exactly 3 downloads went through the fake transport; README.md
            # was reused and never resolved.
            self.assertEqual(
                sorted(opener.resolve_calls),
                sorted(["config.json", "model.safetensors", "extra.txt"]),
            )

            for name, data in repo.content_by_name.items():
                published = output / name
                self.assertTrue(published.is_file(), name)
                self.assertEqual(published.read_bytes(), data)

            # Reuse moves the verified candidate: the published file keeps
            # the exact inode the candidate had (a move, not a copy), has a
            # single link (a move, not a hardlink), and the candidate path
            # no longer exists in the preserved staging tree (so the file
            # remains eligible for a later, second hop of reuse elsewhere,
            # and no second writable-path link into a failed staging tree
            # survives publication).
            reused_published = (output / "README.md").stat()
            self.assertEqual(reused_published.st_dev, candidate_identity.st_dev)
            self.assertEqual(reused_published.st_ino, candidate_identity.st_ino)
            self.assertEqual(reused_published.st_nlink, 1)
            self.assertFalse((reuse_root / "README.md").exists())

            # Every downloaded (non-reused) file also publishes at nlink 1.
            for name in ("config.json", "model.safetensors", "extra.txt"):
                self.assertEqual((output / name).stat().st_nlink, 1, name)

            # The unusable candidates (hash mismatch, size mismatch,
            # symlink) are left exactly as they were: still present, still
            # untouched, at their original paths in the preserved tree.
            self.assertEqual(
                (reuse_root / "config.json").read_bytes(), corrupt_bytes_before
            )
            self.assertEqual(
                (reuse_root / "model.safetensors").read_bytes(),
                truncated_bytes_before,
            )
            self.assertTrue((reuse_root / "extra.txt").is_symlink())
            self.assertEqual(
                os.readlink(reuse_root / "extra.txt"), symlink_target_before
            )


    @REQUIRES_MACOS_EXCLUSIVE_RENAME
    def test_without_reuse_flag_every_file_is_downloaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = SyntheticRepo()
            output = root / "model"
            manifest = root / "source-api.json"
            opener = self.opener_for(repo)

            self.acquire(repo, output, manifest, opener, reuse_verified_from=None)

            self.assertEqual(
                sorted(opener.resolve_calls), sorted(repo.content_by_name)
            )


if __name__ == "__main__":
    unittest.main()
