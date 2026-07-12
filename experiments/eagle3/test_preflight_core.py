import json
import hashlib
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import inspect_checkpoint as checkpoint_inspector

from preflight_core import (
    AcceptanceCounters,
    CheckpointSpec,
    PreflightValidationError,
    RoundTiming,
    accept_greedy,
    authenticate_file_manifest,
    classify_cache_drift,
    committed_accepted_drafts,
    sha256_file,
    parse_safetensors_header_bytes,
    read_harness_git_sha,
    stable_evidence_id,
    resolve_huggingface_revision,
    trim_emission,
    validate_tensor_manifest,
)


def pinned_config():
    return {
        "architectures": ["Eagle3Speculator"],
        "draft_vocab_size": 32000,
        "norm_before_residual": True,
        "speculators_config": {
            "algorithm": "eagle3",
            "proposal_methods": [{"speculative_tokens": 3}],
            "verifier": {"name_or_path": "Qwen/Qwen3-32B"},
        },
        "transformer_layer_config": {
            "model_type": "llama",
            "hidden_size": 5120,
            "intermediate_size": 25600,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "num_hidden_layers": 1,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000,
            "vocab_size": 151936,
            "attention_bias": False,
        },
    }


def pinned_header():
    element_bytes = {"BF16": 2, "I64": 8, "BOOL": 1}
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, (dtype, shape) in CheckpointSpec.expected_tensor_manifest().items():
        count = 1
        for extent in shape:
            count *= extent
        end = offset + count * element_bytes[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, end],
        }
        offset = end
    return header, offset


class CheckpointTests(unittest.TestCase):
    def test_file_manifest_authenticates_every_named_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first = directory / "model-00001.safetensors"
            second = directory / "model-00002.safetensors"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            expected = {
                first.name: {
                    "bytes": first.stat().st_size,
                    "sha256": hashlib.sha256(b"first").hexdigest(),
                },
                second.name: {
                    "bytes": second.stat().st_size,
                    "sha256": hashlib.sha256(b"second").hexdigest(),
                },
            }

            result = authenticate_file_manifest(directory, expected)
            self.assertEqual(result["file_count"], 2)
            self.assertEqual(result["total_bytes"], 11)

            second.write_bytes(b"SECOND")
            with self.assertRaisesRegex(PreflightValidationError, "SHA-256"):
                authenticate_file_manifest(directory, expected)

    def test_inspector_rejects_same_size_weight_payload_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            head = Path(temporary)
            config_bytes = json.dumps(pinned_config()).encode("utf-8")
            (head / "config.json").write_bytes(config_bytes)
            weights = head / "model.safetensors"
            weights.write_bytes(b"x")
            metadata = head / ".cache/huggingface/download/model.safetensors.metadata"
            metadata.parent.mkdir(parents=True)
            expected_weight_sha = hashlib.sha256(b"x").hexdigest()
            metadata.write_text(
                "revision\n" + expected_weight_sha + "\n", encoding="utf-8")
            header, data_size = pinned_header()

            with mock.patch.object(checkpoint_inspector, "PINNED_REVISION", "revision"), \
                    mock.patch.object(
                        checkpoint_inspector, "PINNED_BLOB_ID", expected_weight_sha), \
                    mock.patch.object(checkpoint_inspector, "PINNED_FILE_SIZE", 1), \
                    mock.patch.object(
                        checkpoint_inspector,
                        "PINNED_CONFIG_SHA256",
                        hashlib.sha256(config_bytes).hexdigest(),
                        create=True,
                    ), \
                    mock.patch.object(
                        checkpoint_inspector,
                        "read_safetensors_layout",
                        return_value=(header, data_size),
                        create=True,
                    ):
                checkpoint_inspector.inspect(head)
                weights.write_bytes(b"y")
                with self.assertRaisesRegex(PreflightValidationError, "SHA-256"):
                    checkpoint_inspector.inspect(head)

    def test_sha256_file_hashes_the_complete_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "weights.safetensors"
            content = (b"checkpoint-content" * 7) + b"tail"
            path.write_bytes(content)

            self.assertEqual(sha256_file(path, chunk_size=11), hashlib.sha256(content).hexdigest())

    def test_clean_harness_sha_is_required_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".harness-sha"
            clean = "a" * 40
            path.write_text(clean + "\n", encoding="utf-8")
            self.assertEqual(read_harness_git_sha(path), clean)

            path.write_text(clean + "-dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(PreflightValidationError, "dirty"):
                read_harness_git_sha(path)
            self.assertEqual(read_harness_git_sha(path, allow_dirty=True), clean + "-dirty")

    def test_unknown_or_malformed_harness_sha_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".harness-sha"
            for value in ("", "unknown", "ABCDEF", "a" * 39):
                path.write_text(value + "\n", encoding="utf-8")
                with self.subTest(value=value):
                    with self.assertRaisesRegex(PreflightValidationError, "harness Git SHA"):
                        read_harness_git_sha(path, allow_dirty=True)

    def test_pinned_checkpoint_config_resolves_exact_geometry_and_taps(self):
        spec = CheckpointSpec.from_config(pinned_config())

        self.assertEqual(spec.hidden_size, 5120)
        self.assertEqual(spec.head_dim, 128)
        self.assertEqual(spec.query_width, 8192)
        self.assertEqual(spec.kv_width, 1024)
        self.assertEqual(spec.tap_layers, (2, 32, 61))
        self.assertEqual(spec.max_draft, 3)
        self.assertTrue(spec.norm_before_residual)

    def test_qwen_style_draft_layer_is_rejected_for_the_pinned_head(self):
        config = pinned_config()
        config["transformer_layer_config"]["model_type"] = "qwen3"

        with self.assertRaisesRegex(PreflightValidationError, "model_type"):
            CheckpointSpec.from_config(config)

    def test_config_scalar_coercions_fail_closed(self):
        cases = (
            ("norm_before_residual", "false"),
            ("draft_vocab_size", 32000.9),
        )
        for key, value in cases:
            config = pinned_config()
            config[key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaises(PreflightValidationError):
                    CheckpointSpec.from_config(config)

        config = pinned_config()
        config["speculators_config"]["proposal_methods"][0]["speculative_tokens"] = 3.9
        with self.assertRaises(PreflightValidationError):
            CheckpointSpec.from_config(config)

        config = pinned_config()
        config["transformer_layer_config"]["hidden_size"] = 5120.9
        with self.assertRaises(PreflightValidationError):
            CheckpointSpec.from_config(config)

    def test_safetensors_header_parser_reads_little_endian_json_length(self):
        header = {
            "__metadata__": {"format": "pt"},
            "d2t": {"dtype": "I64", "shape": [32000], "data_offsets": [0, 8]},
        }
        encoded = json.dumps(header).encode("utf-8")
        blob = struct.pack("<Q", len(encoded)) + encoded + b"ignored tensor bytes"

        self.assertEqual(parse_safetensors_header_bytes(blob), header)

    def test_pinned_manifest_accepts_all_sixteen_tensors(self):
        header, data_size = pinned_header()

        validate_tensor_manifest(header, data_size=data_size)

    def test_manifest_schema_drift_fails_closed(self):
        header, data_size = pinned_header()
        header["layers.0.self_attn.q_norm.weight"] = {
            "dtype": "BF16",
            "shape": [128],
            "data_offsets": [data_size, data_size + 256],
        }

        with self.assertRaisesRegex(PreflightValidationError, "unexpected tensors"):
            validate_tensor_manifest(header)

    def test_manifest_rejects_overlapping_or_truncated_tensor_payloads(self):
        header, data_size = pinned_header()
        header["fc.weight"]["data_offsets"][0] -= 2
        with self.assertRaisesRegex(PreflightValidationError, "offset"):
            validate_tensor_manifest(header, data_size=data_size)

        header, data_size = pinned_header()
        with self.assertRaisesRegex(PreflightValidationError, "payload size"):
            validate_tensor_manifest(header, data_size=data_size - 1)

    def test_huggingface_revision_comes_from_model_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            head = Path(temporary)
            metadata = head / ".cache/huggingface/download/model.safetensors.metadata"
            metadata.parent.mkdir(parents=True)
            metadata.write_text("abc123\nblob456\n123.0\n", encoding="utf-8")

            self.assertEqual(resolve_huggingface_revision(head), ("abc123", "blob456"))


class AcceptanceTests(unittest.TestCase):
    def test_evidence_id_is_canonical_and_content_bound(self):
        left = {"parameters": {"k": 3}, "rows": [{"speedup": 1.2}]}
        reordered = {"rows": [{"speedup": 1.2}], "parameters": {"k": 3}}
        changed = {"parameters": {"k": 3}, "rows": [{"speedup": 1.21}]}

        self.assertEqual(stable_evidence_id(left), stable_evidence_id(reordered))
        self.assertNotEqual(stable_evidence_id(left), stable_evidence_id(changed))

    def test_cache_drift_classifier_isolates_rejected_future_processing(self):
        self.assertEqual(
            classify_cache_drift(
                expected_token=12,
                full_verify_token=44364,
                retained_batch_token=12,
                sequential_token=12,
            ),
            "rejected-future-cache-drift",
        )
        self.assertEqual(
            classify_cache_drift(
                expected_token=12,
                full_verify_token=44364,
                retained_batch_token=44364,
                sequential_token=12,
            ),
            "batched-retained-prefix-drift",
        )
        self.assertEqual(
            classify_cache_drift(
                expected_token=12,
                full_verify_token=12,
                retained_batch_token=12,
                sequential_token=12,
            ),
            "not-reproduced",
        )

    def test_greedy_accept_walk_emits_matching_prefix_plus_target_bonus(self):
        result = accept_greedy(draft=[10, 11, 12], verify_argmax=[10, 11, 7, 99])

        self.assertEqual(result.accepted, 2)
        self.assertEqual(result.bonus, 7)
        self.assertEqual(result.emitted, (10, 11, 7))

    def test_trim_emission_matches_budget_then_eos_rules(self):
        self.assertEqual(
            trim_emission([1, 2, 99, 4], already_emitted=8, max_tokens=12, eos_ids={99}),
            ((1, 2, 99), True),
        )
        self.assertEqual(
            trim_emission([1, 2, 99], already_emitted=9, max_tokens=11, eos_ids={99}),
            ((1, 2), True),
        )

    def test_committed_acceptance_excludes_drafts_trimmed_after_eos_or_budget(self):
        self.assertEqual(committed_accepted_drafts(accepted=3, emitted_count=2), 2)
        self.assertEqual(committed_accepted_drafts(accepted=2, emitted_count=3), 2)
        with self.assertRaisesRegex(PreflightValidationError, "nonnegative"):
            committed_accepted_drafts(accepted=-1, emitted_count=0)

    def test_counter_metrics_keep_proposal_and_round_denominators_distinct(self):
        counters = AcceptanceCounters(proposed=9, accepted=4, verify_rounds=3)

        self.assertAlmostEqual(counters.proposal_acceptance_rate, 4 / 9)
        self.assertAlmostEqual(counters.accepted_drafts_per_round, 4 / 3)
        self.assertAlmostEqual(counters.inclusive_acceptance_length, 1 + 4 / 3)

    def test_counter_metrics_are_none_without_work(self):
        counters = AcceptanceCounters(proposed=0, accepted=0, verify_rounds=0)

        self.assertIsNone(counters.proposal_acceptance_rate)
        self.assertIsNone(counters.accepted_drafts_per_round)
        self.assertIsNone(counters.inclusive_acceptance_length)

    def test_round_timing_reproduces_pair_specific_dspark_break_even(self):
        timing = RoundTiming(draft_seconds=0.0157, verify_seconds=0.0369, commit_seconds=0.0010)
        economics = timing.economics(
            baseline_token_seconds=0.016,
            accepted_drafts_per_round=1.46,
        )

        self.assertAlmostEqual(economics["round_cost_ratio"], 3.35)
        self.assertAlmostEqual(economics["break_even_accepted_drafts_per_round"], 2.35)
        self.assertAlmostEqual(economics["projected_speedup"], 2.46 / 3.35)


if __name__ == "__main__":
    unittest.main()
