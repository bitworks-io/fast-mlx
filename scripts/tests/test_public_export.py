from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import export_public_repository  # noqa: E402
import validate_public_repository  # noqa: E402


PUBLIC_VENDOR_SOURCE_OVERRIDES = {
    "spike/Vendor/mlx-swift-lm/Libraries/MLXLLM/LLMModelFactory.swift": {
        "source": "public/sanitized-projection/spike/Vendor/mlx-swift-lm/Libraries/MLXLLM/LLMModelFactory.swift",
        "sha256": "97010b716dc2e47b77046d7510db8bf3f3c672bc052029ce35c754f3e51cdf90",
    },
    "spike/Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift": {
        "source": "public/sanitized-projection/spike/Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35.swift",
        "sha256": "e9b3a174ed61aa172b0f1d4a51312ba9536f8d4cc09ccd0e3dbf28f06a94808f",
    },
    "spike/Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen3MoELazyModel.swift": {
        "source": "public/sanitized-projection/spike/Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen3MoELazyModel.swift",
        "sha256": "9f5c926ffe8625b6b17dd7a48056f2d638f29b1a9036b188a6cf1c5e16230d44",
    },
    "spike/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift": {
        "source": "public/sanitized-projection/spike/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift",
        "sha256": "300e934c43eeffbbac0f90d3befdb8818907aa761ed34ef2cec2ed25d808f1c7",
    },
    "spike/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/MTPDrafterModel.swift": {
        "source": "public/sanitized-projection/spike/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/MTPDrafterModel.swift",
        "sha256": "9cc537a054f0d609406aaeefc1ff0969ea11de9a23fe349585415b2e7f311ff0",
    },
    "spike/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift": {
        "source": "public/sanitized-projection/spike/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift",
        "sha256": "8f104d0955510b979d994d29fda922fe4ab9a2158f029ca136749df02078aaa2",
    },
    "spike/Vendor/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift": {
        "source": "public/sanitized-projection/spike/Vendor/mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift",
        "sha256": "0e0cd9862862ab59759c1496db85b217a2adb195f4ae6f25ac18288ea378b275",
    },
    "spike/Vendor/mlx-swift-lm/Tests/MLXLMTests/KVCacheTests.swift": {
        "source": "public/sanitized-projection/spike/Vendor/mlx-swift-lm/Tests/MLXLMTests/KVCacheTests.swift",
        "sha256": "af9608bdc3b308a4e4eb61577408ae0568b05ea0d3e95399e6c3a989aeee182f",
    },
    "spike/Vendor/mlx-swift-lm/Tests/MLXLMTests/MTPSpeculativeTokenIteratorTests.swift": {
        "source": "public/sanitized-projection/spike/Vendor/mlx-swift-lm/Tests/MLXLMTests/MTPSpeculativeTokenIteratorTests.swift",
        "sha256": "67a88575c2cf0a0834eb1822333b6b8b7151030884b31a3aa8e43fa9eb81591b",
    },
    "spike/Vendor/mlx-swift-lm/Tests/MLXLMTests/Qwen35MTPTests.swift": {
        "source": "public/sanitized-projection/spike/Vendor/mlx-swift-lm/Tests/MLXLMTests/Qwen35MTPTests.swift",
        "sha256": "3996179002b61ec022fe1f208aece97bb4cbc6ab28cee155b9d2c5bcf3b3521d",
    },
    "spike/Vendor/mlx-swift-lm/Tests/MLXLMTests/Qwen3MoELazyModelTests.swift": {
        "source": "public/sanitized-projection/spike/Vendor/mlx-swift-lm/Tests/MLXLMTests/Qwen3MoELazyModelTests.swift",
        "sha256": "9d66385dbf031beaf34dffb3d379e7a6e7cd4e5df30b7ad045793a5a2d5e4ac4",
    },
}


# DELIBERATELY HARDCODED, not recomputed: this is the tripwire that forces a conscious
# decision whenever the public path set changes. Recomputing it here would make it agree
# with any projection, including one that leaked a file. It is defined exactly ONCE, as a
# module-level constant, because keeping the same literal in two places inside
# test_public_projection_uses_sanitized_vendor_overrides -- the publicIndex assertion and
# the reexport_count assertion -- and updating them out of step has turned HEAD red three
# times (see the comment above the publicIndex assertion for the full history). Both call
# sites below read this constant; there is no longer a second literal to drift.
SEALED_PUBLIC_INDEX = {
    "pathCount": 908,
    "pathModeSha256": "19cbed1e282d7dea4fa14da066dcf02fe90e2729f9fb34d7d67bf7754b135666",
}


def public_index_seal(entries: dict[str, str]) -> dict[str, object]:
    digest = hashlib.sha256()
    for path in sorted(entries):
        digest.update(entries[path].encode("ascii"))
        digest.update(b"\0")
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
    return {
        "pathCount": len(entries),
        "pathModeSha256": digest.hexdigest(),
    }


class PublicExportTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> None:
        files = {
            "public/public-repository.json": json.dumps(
                {
                    "schemaVersion": 1,
                    "files": [
                        {"source": "README.md", "destination": "README.md"},
                        {
                            "source": "public/public-repository-public.json",
                            "destination": "public/public-repository.json",
                        },
                    ],
                    "trees": [
                        {"source": "site", "destination": "site"}
                    ],
                }
            ),
            "public/public-repository-public.json": json.dumps(
                {
                    "schemaVersion": 1,
                    "publicIndex": public_index_seal(
                        {
                            "README.md": "100644",
                            "public/public-repository.json": "100644",
                            "site/publications.json": "100644",
                            "site/assets/site.css": "100644",
                            "docs/content/2026-08-06-note.md": "100644",
                        }
                    ),
                    "files": [
                        {"source": "README.md", "destination": "README.md"},
                        {
                            "source": "public/public-repository.json",
                            "destination": "public/public-repository.json",
                        },
                    ],
                    "trees": [{"source": "site", "destination": "site"}],
                }
            ),
            "site/publications.json": json.dumps(
                {
                    "schemaVersion": 1,
                    "articles": [
                        {
                            "source": "docs/content/2026-08-06-note.md",
                            "slug": "note",
                            "status": "published",
                            "reviewedAt": "2026-08-06",
                        }
                    ],
                }
            ),
            "site/assets/site.css": "body {}\n",
            "docs/content/2026-08-06-note.md": "# Note\n\nPublic body.\n",
            "README.md": "# Fixture\n",
        }
        for name, contents in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
        (root / "untracked-secret.txt").write_text("must not copy", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "add", "README.md", "public", "site", "docs"], cwd=root, check=True)

    def test_project_manifest_exports_projection_toolchain(self) -> None:
        canonical_manifest = json.loads(
            (REPOSITORY_ROOT / "public/public-repository.json").read_text(
                encoding="utf-8"
            )
        )
        public_manifest_path = REPOSITORY_ROOT / "public/public-repository-public.json"
        public_manifest = (
            json.loads(public_manifest_path.read_text(encoding="utf-8"))
            if public_manifest_path.is_file()
            else canonical_manifest
        )
        public_mappings = {
            (entry.get("source"), entry.get("destination"))
            for entry in public_manifest.get("files", [])
            if isinstance(entry, dict)
        }

        if public_manifest_path.is_file():
            engineering_mappings = {
                (entry.get("source"), entry.get("destination"))
                for entry in canonical_manifest.get("files", [])
                if isinstance(entry, dict)
            }
            self.assertIn(
                (
                    "public/public-repository-public.json",
                    "public/public-repository.json",
                ),
                engineering_mappings,
            )
            self.assertIn(
                ("public/docs-content-README.md", "docs/content/README.md"),
                engineering_mappings,
            )
        for path in (
            "scripts/export_public_repository.py",
            "scripts/tests/test_public_export.py",
        ):
            self.assertIn((path, path), public_mappings)
        self.assertIn(
            ("public/public-repository.json", "public/public-repository.json"),
            public_mappings,
        )
        for entry in public_manifest.get("files", []):
            self.assertEqual(entry.get("source"), entry.get("destination"))
        for entry in public_manifest.get("trees", []):
            self.assertEqual(entry.get("source"), entry.get("destination"))
        self.assertEqual(
            set(public_manifest.get("publicIndex", {})),
            {"pathCount", "pathModeSha256"},
        )

    def test_current_development_projection_passes_public_validator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "public"
            output.mkdir()
            export_public_repository.export(
                REPOSITORY_ROOT,
                output,
                allow_development_manifest=(
                    REPOSITORY_ROOT
                    / "public/public-repository-public.json"
                ).is_file(),
            )

            failures = validate_public_repository.validate(output)

        self.assertEqual(failures, [])

    def test_sampled_generation_foundation_is_exported_byte_for_byte(self) -> None:
        development_manifest = json.loads(
            (REPOSITORY_ROOT / "public/public-repository.json").read_text(
                encoding="utf-8"
            )
        )
        tree_entries = {
            entry["source"]: entry
            for entry in development_manifest["trees"]
        }
        source_path = "HarnessCore/Sampling/SamplingContractV1.swift"
        test_path = "HarnessCoreTests/SamplingContractV1Tests.swift"
        self.assertNotIn(
            source_path,
            tree_entries["spike/Sources"].get("exclude", []),
        )
        self.assertNotIn(
            test_path,
            tree_entries["spike/Tests"].get("exclude", []),
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "public"
            output.mkdir()
            export_public_repository.export(
                REPOSITORY_ROOT,
                output,
                allow_development_manifest=(
                    REPOSITORY_ROOT
                    / "public/public-repository-public.json"
                ).is_file(),
            )
            for relative in (
                "spike/Sources/HarnessCore/Sampling/SamplingContractV1.swift",
                "spike/Tests/HarnessCoreTests/SamplingContractV1Tests.swift",
            ):
                self.assertEqual(
                    (output / relative).read_bytes(),
                    (REPOSITORY_ROOT / relative).read_bytes(),
                )

    def test_vendor_tree_excludes_are_absent_from_export(self) -> None:
        # Development checkouts only: the public checkout's remapped manifest
        # intentionally carries no development-side exclusion list. Exclusion
        # REMOVAL is separately guarded by the sealed publicIndex pathCount.
        public_manifest_path = (
            REPOSITORY_ROOT / "public/public-repository-public.json"
        )
        if not public_manifest_path.is_file():
            return

        development_manifest = json.loads(
            (REPOSITORY_ROOT / "public/public-repository.json").read_text(
                encoding="utf-8"
            )
        )
        vendor_root = "spike/Vendor/mlx-swift-lm"
        vendor_tree = next(
            entry
            for entry in development_manifest["trees"]
            if entry.get("source") == vendor_root
            and entry.get("destination") == vendor_root
        )
        override_destinations = {
            entry.get("destination")
            for entry in development_manifest.get("files", [])
        }
        vendor_excludes = [
            value
            for value in vendor_tree.get("exclude", [])
            if (REPOSITORY_ROOT / vendor_root / value).is_file()
            and f"{vendor_root}/{value}" not in override_destinations
        ]
        self.assertTrue(vendor_excludes)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "public"
            output.mkdir()
            export_public_repository.export(
                REPOSITORY_ROOT,
                output,
                allow_development_manifest=True,
            )
            for excluded in vendor_excludes:
                self.assertFalse(
                    (output / vendor_root / excluded).exists(),
                    "internal-only vendor exclusion leaked into the public "
                    f"export: {excluded}",
                )

    def test_public_projection_uses_sanitized_vendor_overrides(self) -> None:
        development_manifest = json.loads(
            (REPOSITORY_ROOT / "public/public-repository.json").read_text(
                encoding="utf-8"
            )
        )
        public_manifest_path = REPOSITORY_ROOT / "public/public-repository-public.json"
        has_development_manifest = public_manifest_path.is_file()
        public_manifest = (
            json.loads(public_manifest_path.read_text(encoding="utf-8"))
            if has_development_manifest
            else development_manifest
        )
        # DELIBERATELY HARDCODED, not recomputed: the SEALED_PUBLIC_INDEX constant above this
        # class is the tripwire that forces a conscious decision whenever the public path set
        # changes. Recomputing it here would make it agree with any projection, including one
        # that leaked a file. When it fails, do NOT just paste the actual value -- first confirm
        # the added/removed path belongs in public, then update the TWO places that still encode
        # the path count separately:
        #   1. SEALED_PUBLIC_INDEX (above this class),
        #   2. public/public-repository-public.json's stored seal.
        # The former third place -- the `reexport_count` assertion in
        # test_public_projection_uses_sanitized_vendor_overrides (below) -- now derives from
        # SEALED_PUBLIC_INDEX["pathCount"] instead of carrying its own literal, so it can no
        # longer drift out of step on its own.
        # Historically, updating these out of step left the suite red at HEAD: 4a15e78 missed (1)
        # and the (then-separate) reexport_count literal, and the repair that fixed (1) still
        # missed reexport_count.
        #
        # 871 -> 872 (4a15e78) added spike/Tests/SpikeCoreTests/MLXDecoderEvaluateThrowingPropagationTests.swift.
        # That commit resealed the manifest but missed this literal, so the full suite failed while
        # the narrow projection regression still passed -- which is exactly why a change touching
        # projected files must run this whole suite, not just the narrow regression.
        #
        # 872 -> 873 added spike/Vendor/mlx-swift-lm/Tests/MLXLMTests/LLMModelFactoryLoadTests.swift,
        # the coverage for the `LLMModelFactory._load` extraction seam. Confirmed to belong in
        # public before reseal: it is ordinary XCTest coverage of a projected loading contract,
        # every sibling MLXLMTests file is already projected, and it carries no internal family
        # marker, infrastructure detail, or machine-local path. All three places moved together
        # this time.
        # 879 -> 880 added spike/Tests/ServingCoreTests/InCheckpointMTPSelectionTests.swift,
        # the coverage for the in-checkpoint MTP selection's pinned deployment values. Confirmed to
        # belong in public before reseal: it is ordinary XCTest coverage of a projected serving
        # contract, its sibling ServingCoreTests files are already projected, and it carries no
        # internal family marker, infrastructure detail, or machine-local path. The two vendored
        # facade files added in the same increment are NOT here because they carry exclusion
        # entries -- only this one reaches the projection. All three places moved together.
        # 886 -> 887 added
        # spike/Tests/SpikeCoreTests/MTPSpeculativeDecoderPrefillChunkParityTests.swift, the
        # regression for a real prefill chunk-size defect: the speculative decoder silently
        # inherited the vendored 512 default while the scalar route and the capacity fit both
        # use 2048, so the two serving routes chunked prefill differently for reasons unrelated
        # to speculation. Confirmed to belong in public before reseal: it is ordinary coverage
        # of projected serving-core behavior, it drives weight-free synthetic mocks rather than
        # any real checkpoint, and it carries no internal family marker, infrastructure detail,
        # or machine-local path. All three places moved together.
        # 885 -> 886 added
        # spike/Vendor/mlx-swift-lm/Tests/MLXLMTests/TokenIteratorThrowingEntryPointTests.swift,
        # the coverage for TokenIterator's new throwing entry point -- the half of the startup
        # gate's abort hazard that was specified but never landed. Confirmed to belong in public
        # before reseal: it is ordinary coverage of a projected, non-excluded vendored contract
        # (Evaluate.swift is itself projected), it drives a synthetic stub model rather than any
        # real checkpoint, and it carries no internal family marker, infrastructure detail, or
        # machine-local path -- unlike the target-family-specific throwing-entry-point sibling in
        # the same directory, which carries an exclusion entry. All three places moved together.
        # 884 -> 885 added
        # spike/Tests/SpikeServingAdaptersTests/MTPDecoderBridgeSelectionTests.swift, which
        # covers the serving-load wiring that gives the speculative decoder its first
        # production consumer: the fail-closed decoder-strategy guard, the cache-factory
        # shared between the speculative and plain decoder branches, and the decoder
        # selection itself. Confirmed to belong in public before reseal: ordinary XCTest
        # coverage of an already-projected serving contract, carrying no internal family
        # marker, infrastructure detail, or machine-local path. All three places moved
        # together.
        # 881 -> 884 added, in one increment: the speculative serving decoder
        # (spike/Sources/SpikeCore/MTPSpeculativeDecoder.swift), its XCTest coverage
        # (spike/Tests/SpikeServingAdaptersTests/MTPSpeculativeDecoderTests.swift), and the
        # mock target/drafter fixtures extracted, access-level-only, out of the startup-gate
        # test file (.../InCheckpointMTPMockFixtures.swift) so both suites can share them.
        # Confirmed to belong in public before reseal: all three are ordinary source/XCTest
        # coverage of an already-projected serving contract, and each was checked to carry no
        # internal family marker, infrastructure detail, or machine-local path -- the first
        # projection attempt FAILED the marker gate on two of them and they were reworded to
        # describe the target family by role. All three places moved together.
        # 880 -> 881 added
        # spike/Tests/SpikeServingAdaptersTests/InCheckpointMTPStartupGateEndToEndTests.swift,
        # the first end-to-end coverage of the in-checkpoint MTP startup equivalence gate --
        # it drives the real MLX token iterators through a weight-free mock target/drafter pair
        # rather than asserting on hand-written struct literals. Confirmed to belong in public
        # before reseal: it is ordinary XCTest coverage of a projected serving contract, its
        # sibling SpikeServingAdaptersTests files are already projected, and it carries no
        # internal family marker, infrastructure detail, or machine-local path. All three places
        # moved together.
        # 887 -> 889 added, in one increment:
        # spike/Sources/SpikeServingAdapters/InCheckpointMTPFitComposition.swift and its
        # spike/Tests/SpikeServingAdaptersTests/InCheckpointMTPFitCompositionTests.swift coverage
        # -- the fit-check correction that counts the in-checkpoint MTP drafter's own additional
        # growing attention cache (nAttnLayers 12 -> 13) when --qwen4exp-mtp is passed. Confirmed
        # to belong in public before reseal: they are ordinary source/XCTest coverage of an
        # already-projected serving/capacity contract (the sibling NGramOffloadFitComposition and
        # the fastmlx-serve call site are both projected), they are pure geometry over synthetic
        # profile fixtures rather than any real checkpoint, and they carry no infrastructure
        # detail or machine-local path. As with 881 -> 884, the FIRST projection attempt FAILED
        # the marker gate: the type was originally named with the internal family marker, which
        # also dragged the marker into the already-projected fastmlx-serve call site. Renaming it
        # to InCheckpointMTPFitComposition and describing the family by role (lowercase
        # qwen4_exp, as the projected sibling already does) cleared the gate. All three places
        # moved together.
        # 889 -> 890 added, in one increment:
        # spike/Tests/SpikeServingAdaptersTests/FitCompositionStackTests.swift -- coverage for the
        # STACK of the two fit compositions (in-checkpoint MTP, then n-gram offload), which is the
        # only reachable MTP serving shape because --qwen4exp-mtp requires --ngram-offload-plan at
        # argument-parse time. Ordinary XCTest coverage of an already-projected contract; pure
        # geometry over synthetic fixtures, no real checkpoint, no infrastructure detail. The same
        # increment edited three already-projected files in place (byte-only, no reseal of their
        # own): the composition gained a sentinel guard, its tests gained the refusal cases, and the
        # catalog's now-stale "treat MTP-on as an under-count" warning was scoped to the raw-catalog
        # consumers that still under-count. For the THIRD time the marker gate caught a real leak
        # first: a doc-comment citation reintroduced the CamelCase family marker into this very
        # file, and was rewritten to name the cache by role. All three places moved together.
        # 894 -> 895 added, in one increment: the cycle's published engineering note on why a
        # pure fit-check function whose only callers were the loads it gated could not be run
        # without paying for the load, on refusing to combine a dry run with the --force flag
        # that suppresses the very verdict it exists to learn, and on reporting
        # offload_path_resolvable=unproven rather than letting an arithmetic-only green read as
        # a serving guarantee. Articles are projected only when named in site/publications.json;
        # this note reports fast-mlx's own work only and names no third party, no host and no
        # machine-local path. The same increment edited one already-projected file in place
        # byte-only, no reseal of its own: site/publications.json gained the entry. All three
        # places moved together.
        #
        # 893 -> 894 added, in one increment:
        # docs/content/2026-09-08-every-tool-call-took-the-path-we-never-tested.md -- the cycle's
        # published engineering note, registered in site/publications.json, on why a default that
        # binds at load rather than per request meant every tool call took an untested path, and on
        # building the gate to measure the discriminating quantity rather than the observable one.
        # Articles are projected only when named in site/publications.json; this one reports
        # fast-mlx's own measurements only and names no third party, no host and no machine-local
        # path. The same increment edited one already-projected file in place (byte-only, no reseal
        # of its own): site/publications.json gained the entry. All three places moved together.
        #
        # 892 -> 893 added, in one increment:
        # spike/Tests/SpikeServingAdaptersTests/ToolCallChunkBoundaryTests.swift -- chunk-boundary
        # invariance for the XML-function tool-call streaming parser, the format ToolCallFormat.infer
        # selects for the served hybrid MoE family. Speculative decoding can change how generated text
        # is segmented into chunks without changing the text, so the parser must produce the same
        # parsed call, and the same ABSENCE of one, under every segmentation. Deliberately
        # unconditional: it uses no tokenizer and no checkpoint, so it is the CI-visible control for
        # the env-gated template suite next to it, which skips wherever no local tokenizer fixture
        # exists. It holds only hand-written wire-format string literals -- no checkpoint text, no
        # infrastructure detail, no machine-local path, and it reads no deployment asset at runtime.
        # The same increment edited one already-projected file in place (byte-only, no reseal of its
        # own): the template render suite gained the tools-with-a-non-leading-system-message case,
        # the intersection its two neighbouring tests each covered only one axis of. All three
        # places moved together.
        #
        # 903 -> 904 added, in one increment (one path, mode 100644):
        # docs/content/2026-09-09-a-ratio-is-not-a-result.md -- a new article registered as
        # published in site/publications.json, which article_pairs() requires before an article's
        # source is added to the projected path set. It is ordinary already-reviewed content in
        # the docs/content/ tree every sibling article already publishes from, and it
        # carries no checkpoint text, no infrastructure detail, no machine-local path, and no
        # internal implementation-family CamelCase name. The same increment edited one
        # already-projected file in place (byte-only, no reseal of its own): site/publications.json
        # gained the entry. All three places moved together.
        #
        # 901 -> 903 added, in one increment (two paths, mode 100644):
        # spike/Tests/ServingCoreTests/SampledMTPServeArgumentTests.swift and
        # spike/Vendor/mlx-swift-lm/Tests/MLXLMTests/TruncatedSamplingProbabilitiesTests.swift --
        # the opt-in sampled-MTP serve flag's argument tests and the vendored truncation helper's
        # tests, from the truncation-parity increment recorded in
        # docs/task-inbox/2026-09-09-sampled-mtp-truncation-parity-IMPLEMENTED.md. The same
        # increment edited several already-projected files in place (byte-only, no reseal of their
        # own). All three places moved together.
        #
        # 899 -> 901 added, in one increment (two paths, mode 100644):
        # spike/Sources/fastmlx-harness/InCheckpointSampledMTPThroughputCLI.swift and
        # spike/Tests/FastMLXHarnessTests/InCheckpointSampledMTPThroughputArithmeticTests.swift --
        # the direct end-to-end speculative-vs-scalar throughput instrument predeclared in
        # docs/task-inbox/2026-09-09-sampled-mtp-direct-throughput-PREDECLARATION.md, plus its 31
        # pure-arithmetic tests. It exists because the sampled-MTP speedup on record is composed
        # from separately timed parts rather than observed, and its denominator omits phase costs
        # the acceptance instrument measures and discards. All three places moved together.
        #
        # 898 -> 899 added, in one increment (one path, mode 100644):
        # spike/Tests/FastMLXHarnessTests/RowsHaveNoCorruptValuesTests.swift -- coverage for the
        # harness verify path's logprob corruption predicate. That check previously rejected any
        # non-finite value, which fails closed on a checkpoint that masks unsupported vocabulary
        # indices to -infinity on every forward: a masked token's logprob is legitimately -infinity
        # because its probability is exactly zero. The predicate now rejects only NaN and
        # +infinity, and these tests pin both directions including the happy-path control, so the
        # relaxation cannot silently widen into accepting real corruption.
        # It is ordinary harness test source inside an already-projected tree: no checkpoint text,
        # no infrastructure detail, no machine-local path, and every identifier is family-neutral
        # because the projected-marker gate forbids the internal implementation-family CamelCase
        # name in a projected path or its bytes. The same increment edited already-projected files
        # in place (byte-only, no reseal of their own): Harness.swift swapped the predicate and its
        # printed label, and the speculative iterator now carries the provider's own failure cause
        # into its passthrough reason instead of a single hardcoded string. All three places moved
        # together.
        #
        # 896 -> 898 added, in one increment (two paths, both mode 100644):
        # spike/Sources/fastmlx-harness/InCheckpointSampledMTPAcceptanceCLI.swift -- the
        # release-only instrument that measures the real sampled-MTP per-step acceptance rate `a`
        # together with the decide() cost, the target step cost and an independently re-derived
        # Sigma_x min(p(x), q(x)), all in one run so no quantity is inherited from a prior
        # measurement.
        # spike/Tests/FastMLXHarnessTests/InCheckpointSampledMTPAcceptanceArithmeticTests.swift --
        # 28 hand-derived unit tests for that CLI's pure arithmetic: softmax (including the masked
        # -infinity row this model really emits), the sigma overlap, the per-step reached/accepted
        # split, the pooled ratio, the break-even solve checked by substitution, and the
        # conditioned-vs-unconditional sigma distinction.
        # Both are ordinary harness source/test inside already-projected trees: no checkpoint text,
        # no infrastructure detail, no machine-local path, and every identifier is family-neutral
        # because the projected-marker gate forbids the internal implementation-family CamelCase
        # name in a projected path or its bytes. The same increment edited one already-projected
        # file in place (byte-only, no reseal of its own): Harness.swift gained the subcommand
        # dispatch and its usage text. All three places moved together.
        #
        # 907 -> 908 added, in one increment:
        # spike/Tests/SpikeCoreTests/TopPFilterRankGenericityTests.swift -- direct coverage of
        # `applyTopPFilter`, which is a public function whose doc comment asserts a rank contract
        # ("no rank precondition", unlike `applyTopKFilter`) that no test had ever exercised: both
        # live call sites pass rank 2, and the existing suites reach the function only through
        # them. It pins rank-1/rank-2 exact agreement, and pins that the `min_tokens_to_keep`
        # floor is the sole survivor mechanism in a regime an unconditional anti-vacuity
        # precondition MEASURES rather than assumes -- an earlier draft claimed that regime at a
        # narrower width where float32 `cumsum` rounding overshot the threshold, so deleting the
        # floor term left the suite green and the test passed for the wrong reason. Confirmed to
        # belong in public before reseal: ordinary XCTest coverage of an already-projected tree,
        # driving synthetic in-test logit fixtures rather than any checkpoint, with no
        # infrastructure detail, no machine-local path, and deliberately family-neutral
        # identifiers, since the projected-marker gate fails closed on the internal
        # implementation-family CamelCase name in a projected path or its bytes. No other
        # projected file moved in this increment.
        #
        # 906 -> 907 added, in one increment:
        # spike/Tests/SpikeServingAdaptersTests/OffloadPlanCheckOnlyServeWiringStructuralTests.swift
        # -- a source-text structural pin asserting that every `ScalarServingModelLoadConfiguration(`
        # construction site in the serve driver forwards `offloadPlanCheckOnly:`, plus a
        # whole-surface companion gate asserting every parsed `public let` argument field is read
        # somewhere by that driver. It exists because the serve driver is an executable target with
        # no test target, so no behavioral Swift test can reach its call sites: the parsed and
        # cross-validated `--offload-plan-check-only` flag was dropped at the one bridge into the
        # load path, silently downgrading an advertised dry run into a full server bound to the
        # operator's configured port, and nothing anywhere turned red. Confirmed to belong in public
        # before reseal: it is ordinary XCTest coverage of already-projected trees, it reads
        # repository source text through a `#filePath` ancestor search rather than any checkpoint or
        # deployment asset, it carries no checkpoint text, no infrastructure detail and no
        # machine-local path, and its identifiers are deliberately family-neutral (the
        # `offload`/`plan`/`ngram` vocabulary, never the internal implementation-family CamelCase
        # name) because the projected-marker gate fails closed on such a marker in a projected path
        # or its bytes. The same increment edited two already-projected files in place (byte-only,
        # no reseal of their own): the serve driver gained the one-line forwarding argument, and
        # spike/Tests/ServingCoreTests/FastMLXServeArgumentsTests.swift gained three parse-level
        # rows pinning that the flag is refused on the continuous routes and parses on the scalar
        # route. All three places moved together.
        #
        # 904 -> 906 added, across two commits in one cycle:
        # spike/Tests/SpikeCoreTests/SampledMTPDegenerateTopPCharacterizationTests.swift
        # (904 -> 905) and
        # spike/Tests/SpikeCoreTests/ScalarTopPSamplerDegenerateTopPCharacterizationTests.swift
        # (905 -> 906). Both characterize how a degenerate `top_p` behaves: the first on the
        # speculative sampled-MTP truncation bridge, the second on the scalar `TopPSampler` arm,
        # where a permuted-peak discriminator shows the scalar arm emits a WRONG token rather than
        # failing closed. Confirmed to belong in public before reseal: both are ordinary XCTest
        # coverage of already-projected SpikeCore contracts, they drive synthetic in-test logit
        # fixtures rather than any real checkpoint, and each was marker-scanned to carry no
        # internal family marker, infrastructure detail, or machine-local path. The stored seal had
        # already moved to 906 while THIS literal and the `reexport_count` assertion below were
        # both left at 904 -- the exact split-update failure this comment warns about, and it
        # turned the public `public-boundary` CI job red at `bd28880a`. All three places move
        # together here.
        #
        # 891 -> 892 added, in one increment:
        # spike/Tests/SpikeServingAdaptersTests/ServingDeveloperRoleMappingTests.swift -- end-to-end
        # coverage for the render boundary's wire-role mapping: OpenAI's `developer` role is now
        # translated to the template's `system` vocabulary before rendering, because no served
        # template has a `developer` branch and the role therefore hit the template's own terminal
        # raise_exception(...). Ordinary XCTest coverage of an already-projected serving contract. It
        # inlines a small Jinja excerpt as a string literal rather than reading any deployment asset
        # at runtime, precisely so this published test carries no dependency on a path the public
        # distribution does not ship; it holds no checkpoint text, no infrastructure detail and no
        # machine-local path. The same increment edited two already-projected files in place
        # (byte-only, no reseal of their own): the codec gained the role-mapping seam and its
        # existing fixture test flipped to pin the mapped role. All three places moved together.
        #
        # 890 -> 891 added, in one increment:
        # spike/Tests/SpikeServingAdaptersTests/ServingChatTemplateRefusalTests.swift -- coverage
        # for ServingChatTemplateRefusal, which converts a chat template's own raise_exception(...)
        # refusal into a typed 400 instead of an opaque 500. Ordinary XCTest coverage of an
        # already-projected serving contract; it renders small hand-written Jinja templates inline,
        # so it carries no checkpoint text, no infrastructure detail and no machine-local path. The
        # same increment edited two already-projected files in place (byte-only, no reseal of their
        # own): the codec gained the translation seam and Package.swift declared the Jinja product
        # that was already resolved transitively. All three places moved together.
        self.assertEqual(
            public_manifest.get("publicIndex"),
            SEALED_PUBLIC_INDEX,
        )

        if has_development_manifest:
            vendor_root = "spike/Vendor/mlx-swift-lm"
            vendor_tree = next(
                entry
                for entry in development_manifest["trees"]
                if entry.get("source") == vendor_root
                and entry.get("destination") == vendor_root
            )
            excluded = set(vendor_tree.get("exclude", []))
            expected_excludes = {
                destination.removeprefix(vendor_root + "/")
                for destination in PUBLIC_VENDOR_SOURCE_OVERRIDES
            }
            self.assertLessEqual(expected_excludes, excluded)

            file_mappings = {
                (entry.get("source"), entry.get("destination"))
                for entry in development_manifest["files"]
            }
            expected_mappings = {
                (metadata["source"], destination)
                for destination, metadata in PUBLIC_VENDOR_SOURCE_OVERRIDES.items()
            }
            self.assertLessEqual(expected_mappings, file_mappings)

            for destination, metadata in PUBLIC_VENDOR_SOURCE_OVERRIDES.items():
                source_path = REPOSITORY_ROOT / metadata["source"]
                source_bytes = source_path.read_bytes()
                destination_bytes = (REPOSITORY_ROOT / destination).read_bytes()
                self.assertEqual(
                    hashlib.sha256(source_bytes).hexdigest(),
                    metadata["sha256"],
                )
                self.assertNotEqual(source_bytes, destination_bytes)
        else:
            for metadata in PUBLIC_VENDOR_SOURCE_OVERRIDES.values():
                self.assertFalse((REPOSITORY_ROOT / metadata["source"]).exists())

        self.assertEqual(
            set(public_manifest.get("publicIndex", {})),
            {"pathCount", "pathModeSha256"},
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "public"
            reexport = Path(directory) / "reexport"
            output.mkdir()
            export_public_repository.export(
                REPOSITORY_ROOT,
                output,
                allow_development_manifest=has_development_manifest,
            )

            self.assertFalse((output / "public/sanitized-projection").exists())
            internal_family_marker = "Qwen" + "4Exp"
            internal_family_paths = sorted(
                str(path.relative_to(output))
                for path in output.rglob(f"*{internal_family_marker}*")
            )
            self.assertEqual(internal_family_paths, [])
            internal_family_content = sorted(
                str(path.relative_to(output))
                for path in output.rglob("*")
                if path.is_file()
                and internal_family_marker.encode("utf-8") in path.read_bytes()
            )
            self.assertEqual(internal_family_content, [])
            for destination, metadata in PUBLIC_VENDOR_SOURCE_OVERRIDES.items():
                output_bytes = (output / destination).read_bytes()
                self.assertEqual(
                    hashlib.sha256(output_bytes).hexdigest(),
                    metadata["sha256"],
                )
                if has_development_manifest:
                    source_bytes = (REPOSITORY_ROOT / metadata["source"]).read_bytes()
                    self.assertEqual(output_bytes, source_bytes)
                else:
                    checkout_bytes = (REPOSITORY_ROOT / destination).read_bytes()
                    self.assertEqual(output_bytes, checkout_bytes)

            subprocess.run(["git", "init", "-q"], cwd=output, check=True)
            subprocess.run(["git", "add", "."], cwd=output, check=True)
            reexport_count = export_public_repository.export(output, reexport)
            # Same hardcoded-tripwire rule as SEALED_PUBLIC_INDEX (see the comment above that
            # constant and the comment above the publicIndex assertion in
            # test_public_projection_uses_sanitized_vendor_overrides for the update obligation and
            # history). This used to carry its own separate literal, which was the one that kept
            # getting missed: 4a15e78 moved the projection 871 -> 872 and updated neither literal,
            # and the follow-up repair updated the other one but not this one, leaving the suite
            # red at HEAD a second time. Re-exporting the already-projected tree must reproduce the
            # same path count -- that idempotence is what this asserts -- so this now reads
            # SEALED_PUBLIC_INDEX["pathCount"] directly and can no longer drift out of step.
            self.assertEqual(reexport_count, SEALED_PUBLIC_INDEX["pathCount"])
            for destination, metadata in PUBLIC_VENDOR_SOURCE_OVERRIDES.items():
                output_bytes = (output / destination).read_bytes()
                reexport_bytes = (reexport / destination).read_bytes()
                self.assertEqual(reexport_bytes, output_bytes)
                self.assertEqual(
                    hashlib.sha256(reexport_bytes).hexdigest(),
                    metadata["sha256"],
                )
            self.assertFalse((reexport / "public/sanitized-projection").exists())

    def test_mtp_processor_pin_does_not_reintroduce_the_fictional_copy(self) -> None:
        """The MTP logit-processor pin must not describe a copy that does not exist.

        Until 2026-09-09 the emit-only pin -- in BOTH the vendored test and its published
        sanitized projection -- explained itself in terms of ``var verifyProcessorCopy =
        processor`` inside ``speculateRound``. That line has never existed in the vendored
        production source; the sequential verify loop breaks at the first draft mismatch, so
        positions after it are never sampled and there is nothing to leak. The assertion held
        for the right value and the wrong stated reason, and the false explanation shipped
        publicly.

        The identifier is still real in ``spike/.build/checkouts/mlx-swift-lm/``, a stale
        leftover from when ``mlx-swift-lm`` was a URL dependency -- a DIFFERENT revision that a
        grep without ``--exclude-dir=.build`` will happily surface. So this pins the three files
        that actually matter by exact path rather than by search.

        This is the only gate that reads the vendored source and its published projection
        together: the sha256 pin above hashes the projection alone, so a vendored edit whose
        hand-port was forgotten leaves every other test green.
        """
        fictional = "verifyProcessorCopy"
        sanitized = (
            "public/sanitized-projection/spike/Vendor/mlx-swift-lm/Tests/MLXLMTests/"
            "MTPSpeculativeTokenIteratorTests.swift"
        )
        # (path, token that MUST be present) -- the second element is the anti-vacuity control.
        # Without it, a repointed or truncated path would satisfy the absence check by reading
        # a file with no relevant content at all, which is an INERT pass rather than a passing
        # one. Note this deliberately does not scan scripts/ -- this very file names the
        # identifier, and a gate that swept its own reminder text would count itself.
        #
        # These two exist in BOTH the development checkout and the public projection. The
        # sanitized source does NOT: the exporter strips `public/sanitized-projection/` from
        # the candidate by design, which is asserted a few tests above. Checking it
        # unconditionally is what turned this gate red in the public-boundary job on its first
        # publication; it is now handled explicitly below rather than assumed.
        always_present = (
            (
                "spike/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/"
                "MTPSpeculativeTokenIterator.swift",
                "func speculateRound",
            ),
            (
                "spike/Vendor/mlx-swift-lm/Tests/MLXLMTests/"
                "MTPSpeculativeTokenIteratorTests.swift",
                "private struct EmissionLog",
            ),
        )

        # assertTrue/assertFalse on an explicit `in`, NOT assertIn/assertNotIn: the latter
        # interpolate the whole haystack into the failure message, which for these files is a
        # ~77 KB single-line dump that buries the actual message.
        def check(relative: str, required: str) -> None:
            path = REPOSITORY_ROOT / relative
            self.assertTrue(path.is_file(), f"{relative} is missing")
            text = path.read_text(encoding="utf-8")
            self.assertTrue(
                required in text,
                f"{relative} does not contain {required!r} -- this gate is reading the "
                "wrong file, so its absence check below proves nothing",
            )
            self.assertFalse(
                fictional in text,
                f"{relative} names {fictional!r}, an identifier that does not exist in the "
                "vendored production source. The emit-only invariant holds because the "
                "verify loop breaks at the first draft mismatch, not because of a struct "
                "value-copy.",
            )

        for relative, required in always_present:
            check(relative, required)

        if (REPOSITORY_ROOT / sanitized).is_file():
            # Development checkout: the hand-ported copy is the one that actually publishes,
            # and the sha256 pin above hashes it in isolation -- so a vendored edit whose
            # hand-port was forgotten leaves every other test green. This is the only gate
            # that reads both copies together.
            check(sanitized, "private struct EmissionLog")
        else:
            # Public checkout. NOT a silent skip: assert the whole sanitized tree is absent,
            # so a mistyped path in the branch above fails loudly here instead of quietly
            # taking the "must be the public checkout" route and asserting nothing.
            self.assertFalse(
                (REPOSITORY_ROOT / "public/sanitized-projection").exists(),
                f"{sanitized} is missing even though public/sanitized-projection/ exists -- "
                "this is a development checkout with a broken path, not a public one",
            )

    def test_export_copies_only_indexed_allowlist_and_published_articles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            output = Path(directory) / "output"
            root.mkdir()
            output.mkdir()
            self.make_fixture(root)
            (root / "README.md").write_text("# Unstaged content\n", encoding="utf-8")

            count = export_public_repository.export(
                root,
                output,
                allow_development_manifest=True,
            )

            self.assertEqual(count, 5)
            self.assertTrue((output / "README.md").is_file())
            self.assertTrue((output / "site/publications.json").is_file())
            self.assertTrue((output / "site/assets/site.css").is_file())
            self.assertTrue(
                (output / "docs/content/2026-08-06-note.md").is_file()
            )
            self.assertFalse((output / "untracked-secret.txt").exists())
            self.assertTrue((output / "public/public-repository.json").is_file())
            self.assertEqual((output / "README.md").read_text(), "# Fixture\n")

    def test_development_projection_is_strictly_self_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            output = Path(directory) / "output"
            reexport = Path(directory) / "reexport"
            root.mkdir()
            output.mkdir()
            reexport.mkdir()
            self.make_fixture(root)

            export_public_repository.export(
                root,
                output,
                allow_development_manifest=True,
            )
            subprocess.run(["git", "init", "-q"], cwd=output, check=True)
            subprocess.run(["git", "add", "."], cwd=output, check=True)

            count = export_public_repository.export(output, reexport)

            self.assertEqual(count, 5)
            expected = {
                path.relative_to(output).as_posix(): (
                    path.read_bytes(),
                    path.stat().st_mode & 0o111,
                )
                for path in output.rglob("*")
                if path.is_file() and ".git" not in path.relative_to(output).parts
            }
            actual = {
                path.relative_to(reexport).as_posix(): (
                    path.read_bytes(),
                    path.stat().st_mode & 0o111,
                )
                for path in reexport.rglob("*")
                if path.is_file()
            }
            self.assertEqual(actual, expected)

    def test_development_manifest_requires_explicit_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            output = Path(directory) / "output"
            root.mkdir()
            self.make_fixture(root)

            with self.assertRaisesRegex(SystemExit, "must use identity paths"):
                export_public_repository.export(root, output)

    def test_development_mode_requires_indexed_identity_manifest_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            output = Path(directory) / "output"
            root.mkdir()
            (root / "README.md").write_text("# Fixture\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "add", "README.md"], cwd=root, check=True)

            with self.assertRaisesRegex(
                SystemExit,
                "--development-projection requires indexed",
            ):
                export_public_repository.export(
                    root,
                    output,
                    allow_development_manifest=True,
                )

    def test_nonempty_output_is_refused_without_deleting_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            output = Path(directory) / "output"
            root.mkdir()
            output.mkdir()
            marker = output / "preserve.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(SystemExit):
                export_public_repository.prepare_output(output, root)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_output_inside_engineering_checkout_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            with self.assertRaises(SystemExit):
                export_public_repository.prepare_output(root / "public-output", root)

    def test_published_article_source_cannot_traverse(self) -> None:
        manifest = {
            "schemaVersion": 1,
            "articles": [{"source": "docs/content/../../README.md"}],
        }
        with self.assertRaises(SystemExit):
            export_public_repository.article_pairs(manifest, {"README.md": "100644"})

    def test_destination_cannot_target_git_metadata(self) -> None:
        manifest = {
            "schemaVersion": 1,
            "files": [
                {
                    "source": "README.md",
                    "destination": ".git/hooks/post-checkout",
                }
            ],
            "trees": [],
        }

        with self.assertRaisesRegex(SystemExit, "Git metadata"):
            export_public_repository.manifest_pairs(
                manifest, {"README.md": "100644"}
            )

    def test_destination_cannot_target_private_public_path(self) -> None:
        private_destination = "docs/" + "superpowers/verdicts/private.md"
        manifest = {
            "schemaVersion": 1,
            "files": [
                {
                    "source": "README.md",
                    "destination": private_destination,
                }
            ],
            "trees": [],
        }

        with self.assertRaisesRegex(SystemExit, "private public path"):
            export_public_repository.manifest_pairs(
                manifest,
                {"README.md": "100644"},
                allow_remapped_manifest=True,
            )

    def test_expanded_tree_destination_cannot_target_private_public_path(self) -> None:
        source = "safe/.github/workflows/ci.yml"
        manifest = {
            "schemaVersion": 1,
            "files": [],
            "trees": [
                {
                    "source": "safe",
                    "destination": "spike/Vendor/mlx-swift-lm",
                }
            ],
        }

        with self.assertRaisesRegex(SystemExit, "private public path"):
            export_public_repository.manifest_pairs(
                manifest,
                {source: "100644"},
                allow_remapped_manifest=True,
            )

    def test_development_source_cannot_read_private_public_path(self) -> None:
        private_source = "docs/" + "superpowers/private.md"
        vendored_github_source = (
            "spike/Vendor/mlx-swift-lm/.github/ISSUE_TEMPLATE/bug_report.md"
        )
        cases = (
            (
                {
                    "schemaVersion": 1,
                    "files": [
                        {
                            "source": private_source,
                            "destination": "docs/content/README.md",
                        }
                    ],
                    "trees": [],
                },
                {private_source: "100644"},
            ),
            (
                {
                    "schemaVersion": 1,
                    "files": [],
                    "trees": [
                        {
                            "source": "docs/" + "superpowers",
                            "destination": "docs/content",
                        }
                    ],
                },
                {private_source: "100644"},
            ),
            (
                {
                    "schemaVersion": 1,
                    "files": [
                        {
                            "source": vendored_github_source,
                            "destination": "docs/content/README.md",
                        }
                    ],
                    "trees": [],
                },
                {vendored_github_source: "100644"},
            ),
            (
                {
                    "schemaVersion": 1,
                    "files": [],
                    "trees": [
                        {
                            "source": "spike/Vendor/mlx-swift-lm/.github",
                            "destination": "docs/content",
                        }
                    ],
                },
                {vendored_github_source: "100644"},
            ),
        )

        for manifest, tracked in cases:
            with self.subTest(manifest=manifest):
                with self.assertRaisesRegex(SystemExit, "private public source"):
                    export_public_repository.manifest_pairs(
                        manifest,
                        tracked,
                        allow_remapped_manifest=True,
                    )

    def test_destination_must_use_canonical_relative_spelling(self) -> None:
        manifest = {
            "schemaVersion": 1,
            "files": [
                {
                    "source": "README.md",
                    "destination": "docs/./content/README.md",
                }
            ],
            "trees": [],
        }

        with self.assertRaisesRegex(SystemExit, "canonical relative path"):
            export_public_repository.manifest_pairs(
                manifest, {"README.md": "100644"}
            )

    def test_tree_exclude_omits_nested_metadata(self) -> None:
        manifest = {
            "schemaVersion": 1,
            "files": [],
            "trees": [
                {
                    "source": "vendor",
                    "destination": "public-vendor",
                    "exclude": [".github"],
                }
            ],
        }
        tracked = {
            "vendor/.github/workflows/test.yml": "100644",
            "vendor/Sources/Library.swift": "100644",
        }
        self.assertEqual(
            export_public_repository.manifest_pairs(
                manifest,
                tracked,
                allow_remapped_manifest=True,
            ),
            [
                (
                    "vendor/Sources/Library.swift",
                    "public-vendor/Sources/Library.swift",
                )
            ],
        )

    def test_tree_exclude_cannot_traverse(self) -> None:
        manifest = {
            "schemaVersion": 1,
            "files": [],
            "trees": [
                {
                    "source": "vendor",
                    "destination": "vendor",
                    "exclude": ["../private"],
                }
            ],
        }
        with self.assertRaises(SystemExit):
            export_public_repository.manifest_pairs(
                manifest, {"vendor/Sources/Library.swift": "100644"}
            )

    def test_manifest_rejects_unknown_top_level_key(self) -> None:
        manifest = {
            "schemaVersion": 1,
            "files": [],
            "trees": [],
            "file": [],
        }

        with self.assertRaisesRegex(SystemExit, "unknown keys"):
            export_public_repository.manifest_pairs(manifest, {})

    def test_tree_manifest_rejects_unknown_key(self) -> None:
        manifest = {
            "schemaVersion": 1,
            "files": [],
            "trees": [
                {
                    "source": "vendor",
                    "destination": "vendor",
                    "excludes": [".github"],
                }
            ],
        }

        with self.assertRaisesRegex(SystemExit, "unknown keys"):
            export_public_repository.manifest_pairs(
                manifest, {"vendor/Sources/Library.swift": "100644"}
            )

    def test_identity_manifest_requires_exact_public_index(self) -> None:
        expected = {"README.md": "100644"}
        manifest = {
            "schemaVersion": 1,
            "publicIndex": public_index_seal(expected),
            "files": [{"source": "README.md", "destination": "README.md"}],
            "trees": [],
        }

        with self.assertRaisesRegex(SystemExit, "public index seal"):
            export_public_repository.manifest_pairs(
                manifest,
                {
                    "README.md": "100644",
                    "development-only.md": "100644",
                },
            )

    def test_identity_manifest_requires_public_index_seal(self) -> None:
        manifest = {
            "schemaVersion": 1,
            "files": [{"source": "README.md", "destination": "README.md"}],
            "trees": [],
        }

        with self.assertRaisesRegex(SystemExit, "requires publicIndex"):
            export_public_repository.manifest_pairs(
                manifest,
                {"README.md": "100644"},
            )

    def test_identity_manifest_seal_rejects_extra_tracked_tree_member(self) -> None:
        expected = {
            "public/public-repository.json": "100644",
            "site/index.html": "100644",
        }
        manifest = {
            "schemaVersion": 1,
            "publicIndex": public_index_seal(expected),
            "files": [
                {
                    "source": "public/public-repository.json",
                    "destination": "public/public-repository.json",
                }
            ],
            "trees": [{"source": "site", "destination": "site"}],
        }
        tracked = dict(expected)
        tracked["site/ambient-extra.txt"] = "100644"

        with self.assertRaisesRegex(SystemExit, "public index seal"):
            export_public_repository.manifest_pairs(manifest, tracked)

    def test_identity_manifest_seal_rejects_mode_drift(self) -> None:
        expected = {"scripts/tool.py": "100644"}
        manifest = {
            "schemaVersion": 1,
            "publicIndex": public_index_seal(expected),
            "files": [
                {
                    "source": "scripts/tool.py",
                    "destination": "scripts/tool.py",
                }
            ],
            "trees": [],
        }

        with self.assertRaisesRegex(SystemExit, "public index seal"):
            export_public_repository.manifest_pairs(
                manifest,
                {"scripts/tool.py": "100755"},
            )

    def test_public_checkout_manifest_rejects_remapped_paths(self) -> None:
        manifest = {
            "schemaVersion": 1,
            "files": [
                {
                    "source": "README.md",
                    "destination": "docs/README.md",
                }
            ],
            "trees": [],
        }

        with self.assertRaisesRegex(SystemExit, "must use identity paths"):
            export_public_repository.manifest_pairs(
                manifest,
                {"README.md": "100644"},
            )

    def test_file_manifest_rejects_public_source_key(self) -> None:
        manifest = {
            "schemaVersion": 1,
            "files": [
                {
                    "source": "public/docs-content-README.md",
                    "destination": "docs/content/README.md",
                    "publicSource": "docs/content/README.md",
                }
            ],
            "trees": [],
        }

        with self.assertRaisesRegex(SystemExit, "unknown keys"):
            export_public_repository.manifest_pairs(
                manifest, {"public/docs-content-README.md": "100644"}
            )

    def test_article_overlap_refuses_before_copying_any_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            output = Path(directory) / "output"
            root.mkdir()
            output.mkdir()
            self.make_fixture(root)
            manifest_path = root / "public/public-repository.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"].append(
                {
                    "source": "docs/content/2026-08-06-note.md",
                    "destination": "docs/content/2026-08-06-note.md",
                }
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            subprocess.run(
                ["git", "add", "public/public-repository.json"],
                cwd=root,
                check=True,
            )

            with self.assertRaisesRegex(
                SystemExit,
                "duplicate destination across public manifests",
            ):
                export_public_repository.export(
                    root,
                    output,
                    allow_development_manifest=True,
                )

            self.assertEqual(list(output.iterdir()), [])

    def test_validator_rejects_internal_family_marker(self) -> None:
        # Never spell this out as a literal: the marker must be assembled by
        # concatenation, both here and in the validator, so this test file
        # (itself part of the public projection) does not trip the scan it
        # is exercising.
        internal_family_marker = "Qwen" + "4Exp"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean_file = root / "clean.txt"
            clean_file.write_text("nothing to see here\n", encoding="utf-8")

            self.assertEqual(
                validate_public_repository.validate_no_internal_family_marker(root),
                [],
            )

            tainted_file = root / "tainted.txt"
            tainted_file.write_text(
                f"internal note referencing {internal_family_marker}\n",
                encoding="utf-8",
            )

            failures = validate_public_repository.validate_no_internal_family_marker(
                root
            )
            self.assertTrue(
                any("tainted.txt" in failure for failure in failures),
                failures,
            )

if __name__ == "__main__":
    unittest.main()
