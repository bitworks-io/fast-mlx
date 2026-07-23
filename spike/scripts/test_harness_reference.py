#!/usr/bin/env python3
"""Pure contract tests for chunked reference scoring helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest


def _load_reference_module():
    # The production script imports MLX at module scope. These tests exercise only its pure
    # LongRoPE preparation contract, so lightweight stubs keep the off-box test MLX-free.
    mlx = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx.core = mlx_core
    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.load = lambda _: None
    mlx_lm_models = types.ModuleType("mlx_lm.models")
    mlx_lm_cache = types.ModuleType("mlx_lm.models.cache")
    mlx_lm_cache.make_prompt_cache = lambda _: None
    numpy = types.ModuleType("numpy")

    stubs = {
        "mlx": mlx,
        "mlx.core": mlx_core,
        "mlx_lm": mlx_lm,
        "mlx_lm.models": mlx_lm_models,
        "mlx_lm.models.cache": mlx_lm_cache,
        "numpy": numpy,
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        path = Path(__file__).with_name("harness_reference.py")
        spec = importlib.util.spec_from_file_location("harness_reference_under_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class OrdinaryModule:
    pass


class SuScaledRoPE:
    def __init__(self, threshold: int = 4096):
        self.original_max_position_embeddings = threshold
        self._short_freqs = object()
        self._long_freqs = object()
        self._short_scale = object()
        self._long_scale = object()


class FakeModel:
    def __init__(self, modules):
        self._modules = modules

    def named_modules(self):
        return [(f"module.{index}", module) for index, module in enumerate(self._modules)]


class ChunkedLongRoPEContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = _load_reference_module()

    def test_long_sequence_pins_every_su_scaled_rope_before_chunk_zero(self):
        ropes = [SuScaledRoPE(), SuScaledRoPE()]
        original_short = [(rope._short_freqs, rope._short_scale) for rope in ropes]

        result = self.reference.prepare_chunked_longrope(FakeModel(ropes), 27_145)

        self.assertEqual(
            result,
            {
                "fullInputTokens": 27_145,
                "suScaledRoPEModules": 2,
                "longRegimeModules": 2,
            },
        )
        for rope, (short_freqs, short_scale) in zip(ropes, original_short):
            self.assertIsNot(rope._short_freqs, short_freqs)
            self.assertIsNot(rope._short_scale, short_scale)
            self.assertIs(rope._short_freqs, rope._long_freqs)
            self.assertIs(rope._short_scale, rope._long_scale)

    def test_short_sequence_also_pins_long_regime_to_match_swift(self):
        rope = SuScaledRoPE()
        short_freqs = rope._short_freqs
        short_scale = rope._short_scale

        result = self.reference.prepare_chunked_longrope(FakeModel([rope]), 4_096)

        self.assertEqual(result["longRegimeModules"], 1)
        self.assertIsNot(rope._short_freqs, short_freqs)
        self.assertIsNot(rope._short_scale, short_scale)
        self.assertIs(rope._short_freqs, rope._long_freqs)
        self.assertIs(rope._short_scale, rope._long_scale)

    def test_non_longrope_model_is_unchanged(self):
        result = self.reference.prepare_chunked_longrope(
            FakeModel([OrdinaryModule()]), 27_145
        )

        self.assertEqual(result["suScaledRoPEModules"], 0)
        self.assertEqual(result["longRegimeModules"], 0)

    def test_malformed_longrope_fails_closed(self):
        rope = SuScaledRoPE()
        del rope._long_scale

        with self.assertRaisesRegex(RuntimeError, "_long_scale"):
            self.reference.prepare_chunked_longrope(FakeModel([rope]), 27_145)

    def test_invalid_sequence_length_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "full_input_tokens"):
            self.reference.prepare_chunked_longrope(FakeModel([]), 0)


if __name__ == "__main__":
    unittest.main()
