import argparse
import contextlib
import importlib.util
import io
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock


FASTMLX_PATH = Path(__file__).resolve().parents[1] / "fastmlx.py"
_SPEC = importlib.util.spec_from_file_location("fastmlx", FASTMLX_PATH)
assert _SPEC is not None and _SPEC.loader is not None
FASTMLX = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(FASTMLX)


class UsageAndDispatchTests(unittest.TestCase):
    def test_no_args_prints_usage_and_exits_2(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as ctx:
                FASTMLX.main([])
        self.assertEqual(ctx.exception.code, 2)
        for subcommand in FASTMLX.SUBCOMMANDS:
            self.assertIn(subcommand, stdout.getvalue())

    def test_help_flag_prints_usage_and_exits_0(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as ctx:
                FASTMLX.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)
        for subcommand in FASTMLX.SUBCOMMANDS:
            self.assertIn(subcommand, stdout.getvalue())

    def test_short_help_flag_prints_usage_and_exits_0(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as ctx:
                FASTMLX.main(["-h"])
        self.assertEqual(ctx.exception.code, 0)

    def test_unknown_subcommand_exits_2_with_message(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                FASTMLX.main(["bogus"])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("bogus", stderr.getvalue())

    def test_pull_routes_to_fastmlx_pull_main_with_pull_word_stripped(self):
        with mock.patch.object(FASTMLX._pull, "main") as fake_main:
            FASTMLX.main(["pull", "org/name@" + "a" * 40, "--dest", "/tmp/x"])
        fake_main.assert_called_once_with(["org/name@" + "a" * 40, "--dest", "/tmp/x"])

    def test_serve_routes_to_fastmlx_launch_main_with_serve_word_prepended(self):
        with mock.patch.object(FASTMLX._launch, "main") as fake_main:
            FASTMLX.main(["serve", "--model-path", "/tmp/model"])
        fake_main.assert_called_once_with(["serve", "--model-path", "/tmp/model"])

    def test_recommend_routes_to_fastmlx_recommend_main_with_recommend_word_prepended(
        self,
    ):
        with mock.patch.object(FASTMLX._recommend, "main") as fake_main:
            FASTMLX.main(["recommend", "--model-path", "/tmp/model"])
        fake_main.assert_called_once_with(["recommend", "--model-path", "/tmp/model"])

    def test_pull_argv_passthrough_is_exact_even_with_no_extra_args(self):
        with mock.patch.object(FASTMLX._pull, "main") as fake_main:
            FASTMLX.main(["pull", "--help"])
        fake_main.assert_called_once_with(["--help"])

    def test_bench_routes_to_fastmlx_bench_main_with_bench_word_stripped(self):
        with mock.patch.object(FASTMLX._bench, "main") as fake_main:
            FASTMLX.main(["bench", "--base-url", "http://localhost:8080"])
        fake_main.assert_called_once_with(["--base-url", "http://localhost:8080"])


class SubcommandHelpEquivalenceTests(unittest.TestCase):
    """fastmlx <subcommand> --help must behave exactly like invoking the
    sibling module directly with the same argv (argparse's own --help path).
    """

    def test_pull_help_exits_0(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                FASTMLX.main(["pull", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_serve_help_exits_0(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                FASTMLX.main(["serve", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_recommend_help_exits_0(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                FASTMLX.main(["recommend", "--help"])
        self.assertEqual(ctx.exception.code, 0)


class CapacityAndEngineExecTests(unittest.TestCase):
    """capacity/engine are resolved and exec'd (never through a shell), with
    a clean, non-crashing refusal when the binary cannot be found.
    """

    def test_capacity_execs_the_resolved_binary_with_argv_passthrough(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_bin_dir = Path(directory) / "bin"
            fake_bin_dir.mkdir()
            fake_capacity = fake_bin_dir / FASTMLX.CAPACITY_BINARY_NAME
            fake_capacity.write_text("#!/bin/sh\necho capacity\n", encoding="utf-8")
            fake_capacity.chmod(fake_capacity.stat().st_mode | stat.S_IXUSR)

            with mock.patch.object(FASTMLX, "_SIBLING_BIN_DIR", fake_bin_dir):
                with mock.patch.object(FASTMLX.os, "execv") as fake_execv:
                    FASTMLX.main(["capacity", "--fit-check-only", "--model", "x"])

            fake_execv.assert_called_once_with(
                str(fake_capacity),
                [str(fake_capacity), "--fit-check-only", "--model", "x"],
            )

    def test_engine_execs_the_built_in_engine_binary_name(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_bin_dir = Path(directory) / "bin"
            fake_bin_dir.mkdir()
            fake_engine = fake_bin_dir / FASTMLX.ENGINE_BINARY_NAME
            fake_engine.write_text("#!/bin/sh\necho engine\n", encoding="utf-8")
            fake_engine.chmod(fake_engine.stat().st_mode | stat.S_IXUSR)

            with mock.patch.object(FASTMLX, "_SIBLING_BIN_DIR", fake_bin_dir):
                with mock.patch.object(FASTMLX.os, "execv") as fake_execv:
                    FASTMLX.main(["engine", "--help"])

            fake_execv.assert_called_once_with(
                str(fake_engine), [str(fake_engine), "--help"]
            )

    def test_capacity_refuses_cleanly_when_binary_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            empty_bin_dir = Path(directory) / "bin"
            empty_bin_dir.mkdir()
            stderr = io.StringIO()
            with mock.patch.object(FASTMLX, "_SIBLING_BIN_DIR", empty_bin_dir):
                with mock.patch.object(FASTMLX.shutil, "which", return_value=None):
                    with contextlib.redirect_stderr(stderr):
                        with self.assertRaises(SystemExit) as ctx:
                            FASTMLX.main(["capacity"])
            self.assertEqual(ctx.exception.code, 2)
            self.assertIn(FASTMLX.CAPACITY_BINARY_NAME, stderr.getvalue())

    def test_engine_refuses_cleanly_when_binary_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            empty_bin_dir = Path(directory) / "bin"
            empty_bin_dir.mkdir()
            stderr = io.StringIO()
            with mock.patch.object(FASTMLX, "_SIBLING_BIN_DIR", empty_bin_dir):
                with mock.patch.object(FASTMLX.shutil, "which", return_value=None):
                    with contextlib.redirect_stderr(stderr):
                        with self.assertRaises(SystemExit) as ctx:
                            FASTMLX.main(["engine"])
            self.assertEqual(ctx.exception.code, 2)
            self.assertIn(FASTMLX.ENGINE_BINARY_NAME, stderr.getvalue())

    def test_capacity_falls_back_to_path_when_not_in_sibling_bin_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            empty_bin_dir = Path(directory) / "bin"
            empty_bin_dir.mkdir()
            path_bin = str(Path(directory) / "on-path-fastmlx-capacity")

            with mock.patch.object(FASTMLX, "_SIBLING_BIN_DIR", empty_bin_dir):
                with mock.patch.object(FASTMLX.shutil, "which", return_value=path_bin):
                    with mock.patch.object(FASTMLX.os, "execv") as fake_execv:
                        FASTMLX.main(["capacity"])

            fake_execv.assert_called_once_with(path_bin, [path_bin])


class NoShellExecutionTests(unittest.TestCase):
    def test_engine_binary_name_reuses_launch_constant_not_a_literal(self):
        # This assertion pins the intended reuse (never a re-declared literal in
        # this file) rather than merely checking the runtime string value.
        self.assertIs(FASTMLX.ENGINE_BINARY_NAME, FASTMLX._launch._BUILT_IN_ENGINE_BINARY_NAME)


class CLIHelpTextCoverageTests(unittest.TestCase):
    """``--help`` is this project's onboarding/discoverability surface: a
    bare flag name with no explanatory text tells an operator nothing.

    This asserts every ``add_argument()`` flag (and positional) on every
    ``fastmlx*`` CLI parser -- including ones nested under a subcommand,
    e.g. ``fastmlx serve``/``fastmlx recommend`` -- carries a non-empty
    ``help=`` string. It never asserts anything about the CONTENT of a
    help string (accuracy is a human-review concern, not a lint), only
    that one is present.
    """

    # Loaded the same sibling-file way ``scripts/fastmlx.py`` loads its own
    # subcommand modules (see ``FASTMLX._load_sibling_module``), so this
    # test never depends on any of these modules being importable as a
    # package. ``_launch``, ``_pull``, ``_recommend``, and ``_bench`` are
    # reused from the already-loaded dispatcher instead of being loaded a
    # second time.
    MODULES = {
        "fastmlx_bench.py": FASTMLX._bench,
        "fastmlx_gguf_fit.py": FASTMLX._load_sibling_module(
            "fastmlx_gguf_fit", "fastmlx_gguf_fit.py"
        ),
        "fastmlx_launch.py": FASTMLX._launch,
        "fastmlx_pull.py": FASTMLX._pull,
        "fastmlx_recommend.py": FASTMLX._recommend,
        "fastmlx_safetensors_fit.py": FASTMLX._load_sibling_module(
            "fastmlx_safetensors_fit", "fastmlx_safetensors_fit.py"
        ),
    }

    @staticmethod
    def _iter_flag_actions(parser: argparse.ArgumentParser):
        """Every flag/positional action on ``parser``, recursing into any
        subparser (e.g. ``fastmlx serve``'s ``serve`` subparser) so a flag
        that only exists one level down is still checked. The auto-added
        ``-h``/``--help`` action is skipped (argparse supplies its help
        text, not this project), and the ``add_subparsers()`` action
        itself is skipped (it is not an ``add_argument()`` flag) -- but
        its children are still walked.
        """
        for action in parser._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            if isinstance(action, argparse._SubParsersAction):
                for subparser in action.choices.values():
                    yield from CLIHelpTextCoverageTests._iter_flag_actions(subparser)
                continue
            yield action

    def test_every_flag_has_a_nonempty_help_string(self):
        missing = []
        for module_name, module in sorted(self.MODULES.items()):
            parser = module.build_arg_parser()
            for action in self._iter_flag_actions(parser):
                if not (action.help and action.help.strip()):
                    option_strings = action.option_strings or [action.dest]
                    missing.append(f"{module_name}: {'/'.join(option_strings)}")
        self.assertEqual(
            missing,
            [],
            "flags with no (or blank) --help text: " + ", ".join(missing),
        )


if __name__ == "__main__":
    unittest.main()
