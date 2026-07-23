import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = (
    Path(__file__).resolve().parents[1] / "authenticate_hf_snapshot.py"
)
REVISION = "d" * 40
REPO_ID = "example/Test-Model"

SPEC = importlib.util.spec_from_file_location(
    "authenticate_hf_snapshot", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
AUTHENTICATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUTHENTICATOR)


def git_blob_sha1(data: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {len(data)}\0".encode())
    digest.update(data)
    return digest.hexdigest()


def length_field(data: bytes) -> bytes:
    return len(data).to_bytes(8, "big") + data


def fnv1a64(data: bytes) -> str:
    value = 0xCBF29CE484222325
    for byte in data:
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


class SnapshotFixture:
    def __init__(self, root: Path):
        self.model = root / "model"
        self.hub = root / "models--example--Test-Model"
        self.model.mkdir()
        (self.model / ".cache/huggingface/download").mkdir(
            parents=True
        )
        (self.model / ".cache/huggingface/trees").mkdir(parents=True)
        (self.hub / "refs").mkdir(parents=True)

        config = {
            "architectures": ["LlamaForCausalLM"],
            "head_dim": 8,
            "hidden_size": 32,
            "max_position_embeddings": 128,
            "model_type": "llama",
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "quantization": {"bits": 4, "group_size": 8},
        }
        shard = b"synthetic-weights"
        files = {
            ".gitattributes": b"*.safetensors filter=lfs\n",
            "README.md": b"# synthetic\n",
            "config.json": json.dumps(config, sort_keys=True).encode(),
            "model-00001-of-00001.safetensors": shard,
            "model.safetensors.index.json": json.dumps(
                {
                    "metadata": {"total_size": len(shard)},
                    "weight_map": {
                        "model.layers.0.self_attn.k_proj.weight":
                            "model-00001-of-00001.safetensors"
                    },
                },
                sort_keys=True,
            ).encode(),
            "special_tokens_map.json": b'{"eos_token":"</s>"}',
            "tokenizer.json": b'{"version":"1.0"}',
            "tokenizer_config.json": b'{"model_max_length":128}',
        }
        tree_files = {}
        for name, data in files.items():
            path = self.model / name
            path.write_bytes(data)
            if name.endswith(".safetensors") or name == "tokenizer.json":
                content_identity = hashlib.sha256(data).hexdigest()
                entry = {
                    "size": len(data),
                    "blob_id": "0" * 40,
                    "lfs_sha256": content_identity,
                    "lfs_size": len(data),
                }
            else:
                content_identity = git_blob_sha1(data)
                entry = {
                    "size": len(data),
                    "blob_id": content_identity,
                }
            tree_files[name] = entry
            metadata = (
                self.model
                / ".cache/huggingface/download"
                / f"{name}.metadata"
            )
            metadata.parent.mkdir(parents=True, exist_ok=True)
            metadata.write_text(
                f"{REVISION}\n{content_identity}\n0\n",
                encoding="utf-8",
            )

        tree = {"format_version": 1, "files": tree_files}
        self.tree = (
            self.model
            / ".cache/huggingface/trees"
            / f"{REVISION}.json"
        )
        self.tree.write_text(
            json.dumps(tree, sort_keys=True), encoding="utf-8"
        )
        (self.hub / "refs/main").write_text(
            REVISION, encoding="utf-8"
        )
        self.source_api = root / "source-api.json"
        siblings = []
        for name, entry in tree_files.items():
            sibling = {
                "rfilename": name,
                "size": entry["size"],
                "blobId": entry["blob_id"],
            }
            if "lfs_sha256" in entry:
                sibling["lfs"] = {
                    "sha256": entry["lfs_sha256"],
                    "size": entry["lfs_size"],
                }
            siblings.append(sibling)
        self.source_api.write_text(
            json.dumps(
                {
                    "id": REPO_ID,
                    "sha": REVISION,
                    "private": False,
                    "gated": False,
                    "siblings": siblings,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def arguments(self, output: Path) -> argparse.Namespace:
        return argparse.Namespace(
            model_path=str(self.model),
            hub_cache_path=str(self.hub),
            repo_id=REPO_ID,
            revision=REVISION,
            source_api_manifest=str(self.source_api),
            output=str(output),
            expected_model_type="llama",
            expected_architecture="LlamaForCausalLM",
            expected_max_context=128,
            expected_layers=2,
            expected_query_heads=4,
            expected_kv_heads=2,
            expected_head_dim=8,
            expected_weight_bits=4,
            expected_weight_group_size=8,
            expected_rope_type=None,
        )

    def command(self, output: Path) -> list[str]:
        args = self.arguments(output)
        return [
            sys.executable,
            str(SCRIPT),
            "--model-path",
            args.model_path,
            "--hub-cache-path",
            args.hub_cache_path,
            "--repo-id",
            args.repo_id,
            "--revision",
            args.revision,
            "--source-api-manifest",
            args.source_api_manifest,
            "--output",
            args.output,
            "--expected-model-type",
            args.expected_model_type,
            "--expected-architecture",
            args.expected_architecture,
            "--expected-max-context",
            str(args.expected_max_context),
            "--expected-layers",
            str(args.expected_layers),
            "--expected-query-heads",
            str(args.expected_query_heads),
            "--expected-kv-heads",
            str(args.expected_kv_heads),
            "--expected-head-dim",
            str(args.expected_head_dim),
            "--expected-weight-bits",
            str(args.expected_weight_bits),
            "--expected-weight-group-size",
            str(args.expected_weight_group_size),
        ]


class AuthenticateHFSnapshotTests(unittest.TestCase):
    def run_fixture(
        self, fixture: SnapshotFixture, output: Path
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            fixture.command(output),
            check=False,
            capture_output=True,
            text=True,
        )

    def test_authenticates_exact_snapshot_and_writes_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = SnapshotFixture(root)
            output = root / "receipt.json"

            result = self.run_fixture(fixture, output)

            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "PASS")
            self.assertFalse(receipt["promotable"])
            self.assertFalse(receipt["runtimeEvidence"])
            self.assertEqual(receipt["source"]["repoID"], REPO_ID)
            self.assertEqual(receipt["source"]["revision"], REVISION)
            self.assertEqual(
                receipt["source"]["sourceAPIRepoID"], REPO_ID
            )
            self.assertEqual(
                receipt["source"]["sourceAPIRevision"], REVISION
            )
            self.assertEqual(receipt["fileCount"], 8)
            self.assertEqual(receipt["weightShardCount"], 1)
            self.assertEqual(
                receipt["geometry"]["architecture"],
                "LlamaForCausalLM",
            )
            config = (fixture.model / "config.json").read_bytes()
            index = (
                fixture.model / "model.safetensors.index.json"
            ).read_bytes()
            shard_name = "model-00001-of-00001.safetensors"
            shard = (fixture.model / shard_name).read_bytes()
            checkpoint_bytes = (
                b"fastmlx-checkpoint-content-manifest-v2\n"
                + length_field(config)
                + length_field(index)
                + length_field(shard_name.encode())
                + len(shard).to_bytes(8, "big")
                + shard
            )
            self.assertEqual(
                receipt["checkpointContentSHA256"],
                hashlib.sha256(checkpoint_bytes).hexdigest(),
            )
            self.assertEqual(
                receipt["modelConfigHash"], fnv1a64(config)
            )
            manifest_bytes = (
                config
                + index
                + f"{shard_name}:{len(shard)}\n".encode()
            )
            self.assertEqual(
                receipt["checkpointManifestHash"],
                fnv1a64(manifest_bytes),
            )
            tokenizer_bytes = b""
            for name in [
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ]:
                data = (fixture.model / name).read_bytes()
                tokenizer_bytes += (
                    length_field(name.encode())
                    + len(data).to_bytes(8, "big")
                    + data
                )
            self.assertEqual(
                receipt["tokenizerSHA256"],
                hashlib.sha256(tokenizer_bytes).hexdigest(),
            )
            sidecar = Path(f"{output}.sha256")
            self.assertTrue(sidecar.is_file())
            expected = hashlib.sha256(output.read_bytes()).hexdigest()
            self.assertEqual(
                sidecar.read_text(encoding="utf-8").split()[0],
                expected,
            )

    def test_rejects_tampered_lfs_content_without_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = SnapshotFixture(root)
            output = root / "receipt.json"
            shard = (
                fixture.model / "model-00001-of-00001.safetensors"
            )
            shard.write_bytes(b"tampered-weights")

            result = self.run_fixture(fixture, output)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("size mismatch", result.stderr)
            self.assertFalse(output.exists())
            self.assertFalse(Path(f"{output}.sha256").exists())

    def test_rejects_symlinked_source_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = SnapshotFixture(root)
            output = root / "receipt.json"
            tokenizer = fixture.model / "tokenizer_config.json"
            target = root / "external-tokenizer.json"
            target.write_bytes(tokenizer.read_bytes())
            tokenizer.unlink()
            tokenizer.symlink_to(target)

            result = self.run_fixture(fixture, output)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not a regular non-symlink file", result.stderr)
            self.assertFalse(output.exists())

    def test_rejects_revision_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = SnapshotFixture(root)
            output = root / "receipt.json"
            (fixture.hub / "refs/main").write_text(
                "a" * 40, encoding="utf-8"
            )

            result = self.run_fixture(fixture, output)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("revision ref mismatch", result.stderr)
            self.assertFalse(output.exists())

    def test_accepts_exact_commit_snapshot_without_branch_ref(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = SnapshotFixture(root)
            output = root / "receipt.json"
            (fixture.hub / "refs/main").unlink()

            result = self.run_fixture(fixture, output)

            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(
                receipt["source"]["revisionRefPresent"]
            )
            self.assertIsNone(receipt["source"]["revisionRef"])

    def test_rejects_source_api_revision_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = SnapshotFixture(root)
            output = root / "receipt.json"
            source_api = json.loads(
                fixture.source_api.read_text(encoding="utf-8")
            )
            source_api["sha"] = "a" * 40
            fixture.source_api.write_text(
                json.dumps(source_api), encoding="utf-8"
            )

            result = self.run_fixture(fixture, output)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "source API revision mismatch", result.stderr
            )
            self.assertFalse(output.exists())

    def test_refuses_to_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = SnapshotFixture(root)
            output = root / "receipt.json"
            output.write_text("preserve-me", encoding="utf-8")

            result = self.run_fixture(fixture, output)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("output already exists", result.stderr)
            self.assertEqual(
                output.read_text(encoding="utf-8"), "preserve-me"
            )

    def test_rejects_normalized_duplicate_tree_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = SnapshotFixture(root)
            output = root / "receipt.json"
            tree = json.loads(
                fixture.tree.read_text(encoding="utf-8")
            )
            tree["files"]["./README.md"] = tree["files"][
                "README.md"
            ]
            fixture.tree.write_text(
                json.dumps(tree), encoding="utf-8"
            )

            result = self.run_fixture(fixture, output)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "normalized duplicate tree path", result.stderr
            )
            self.assertFalse(output.exists())

    def test_interrupt_between_receipt_and_sidecar_cleans_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "receipt.json"
            real_link = AUTHENTICATOR.os.link
            calls = 0

            def interrupt_second_link(
                source, destination, *, follow_symlinks
            ):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt()
                return real_link(
                    source,
                    destination,
                    follow_symlinks=follow_symlinks,
                )

            with mock.patch.object(
                AUTHENTICATOR.os,
                "link",
                side_effect=interrupt_second_link,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    AUTHENTICATOR.write_receipt(
                        output, {"status": "PASS"}
                    )

            self.assertFalse(output.exists())
            self.assertFalse(Path(f"{output}.sha256").exists())

    def test_interrupt_after_each_publish_cleans_owned_outputs(self):
        for interrupted_call in (1, 2):
            with self.subTest(interrupted_call=interrupted_call):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    output = root / "receipt.json"
                    real_link = AUTHENTICATOR.os.link
                    calls = 0

                    def link_then_interrupt(
                        source, destination, *, follow_symlinks
                    ):
                        nonlocal calls
                        calls += 1
                        result = real_link(
                            source,
                            destination,
                            follow_symlinks=follow_symlinks,
                        )
                        if calls == interrupted_call:
                            raise KeyboardInterrupt()
                        return result

                    with mock.patch.object(
                        AUTHENTICATOR.os,
                        "link",
                        side_effect=link_then_interrupt,
                    ):
                        with self.assertRaises(KeyboardInterrupt):
                            AUTHENTICATOR.write_receipt(
                                output, {"status": "PASS"}
                            )

                    self.assertFalse(output.exists())
                    self.assertFalse(
                        Path(f"{output}.sha256").exists()
                    )

    def test_receipt_hashes_the_exact_metadata_bytes_it_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = SnapshotFixture(root)
            output = root / "receipt.json"
            original_tree = fixture.tree.read_bytes()
            original_source_api = fixture.source_api.read_bytes()
            real_validate = AUTHENTICATOR.validate_source_api_manifest

            def validate_then_mutate(*args, **kwargs):
                result = real_validate(*args, **kwargs)
                fixture.tree.write_text(
                    '{"mutated":"tree"}', encoding="utf-8"
                )
                fixture.source_api.write_text(
                    '{"mutated":"source-api"}', encoding="utf-8"
                )
                return result

            with mock.patch.object(
                AUTHENTICATOR,
                "validate_source_api_manifest",
                side_effect=validate_then_mutate,
            ):
                receipt, _ = AUTHENTICATOR.authenticate(
                    fixture.arguments(output)
                )

            self.assertEqual(
                receipt["source"]["treeMetadataSHA256"],
                hashlib.sha256(original_tree).hexdigest(),
            )
            self.assertEqual(
                receipt["source"]["sourceAPIManifestSHA256"],
                hashlib.sha256(original_source_api).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
