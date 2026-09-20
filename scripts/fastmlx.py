#!/usr/bin/env python3
"""``fastmlx``: the single product command.

This is a thin dispatcher, not a reimplementation: each subcommand's real
behavior lives in its own sibling module (or, for ``capacity``/``engine``, in
a compiled binary this repository/formula ships alongside this dispatcher).
This file only routes ``sys.argv`` to the right place, the same way
``fastmlx_recommend.py`` already imports ``fastmlx_launch.py`` -- by loading
the sibling module from its file path, never by duplicating its logic.

Subcommands:

- ``fastmlx pull ...``      -> ``fastmlx_pull.main`` (a verified, resumable,
  pinned Hugging Face snapshot pull).
- ``fastmlx serve ...``     -> ``fastmlx_launch.main`` with the ``serve``
  subcommand word prepended (fit-check + quality-admission front door).
- ``fastmlx recommend ...`` -> ``fastmlx_recommend.main`` with the
  ``recommend`` subcommand word prepended (rank local packs by fit +
  measured quality).
- ``fastmlx capacity ...``  -> exec the ``fastmlx-capacity`` binary found
  next to this dispatcher's own install (a sibling ``bin`` dir) or on
  ``PATH``, argv passthrough, never through a shell.
- ``fastmlx engine ...``    -> exec this repository's Swift serving binary
  directly (the research engine, an escape hatch for an operator who wants
  to bypass the fit-check/quality-admission front door on purpose), same
  resolution and passthrough rules as ``capacity``.
- ``fastmlx bench ...``     -> ``fastmlx_bench.main`` (measure decode
  throughput against any OpenAI-compatible endpoint).

``pull`` and ``bench`` each have no subparser of their own (each is a bare
``argparse.ArgumentParser`` with its own positional/flag arguments), so this
dispatcher strips the leading ``pull``/``bench`` word before calling it.
``serve`` and ``recommend`` each require their own subcommand word (they use
``add_subparsers(..., required=True)``), so this dispatcher re-adds it --
this is why ``fastmlx serve --model-path X`` behaves exactly like
``python3 scripts/fastmlx_launch.py serve --model-path X``, and likewise for
``recommend``.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import NoReturn, Optional, Sequence


def _load_sibling_module(name: str, filename: str) -> ModuleType:
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_pull = _load_sibling_module("fastmlx_pull", "fastmlx_pull.py")
_launch = _load_sibling_module("fastmlx_launch", "fastmlx_launch.py")
_recommend = _load_sibling_module("fastmlx_recommend", "fastmlx_recommend.py")
_bench = _load_sibling_module("fastmlx_bench", "fastmlx_bench.py")


# The capacity-check binary this repository ships. Unlike
# ``fastmlx_launch._BUILT_IN_ENGINE_BINARY_NAME`` (built by string
# concatenation to keep an unrelated third-party engine's short name out of
# this repository's source text -- see that constant's own docstring), this
# literal contains no such substring and is written plainly.
CAPACITY_BINARY_NAME = "fastmlx-capacity"

# The Swift serving binary ``fastmlx engine`` execs directly: the same
# binary ``fastmlx serve`` uses as its own default engine and fit-check
# binary. Reused, never re-declared, so this file never repeats the
# concatenation trick or risks drifting from it.
ENGINE_BINARY_NAME = _launch._BUILT_IN_ENGINE_BINARY_NAME

# This dispatcher's own install directory is ``<prefix>/libexec/scripts`` in
# the Homebrew layout (and ``<repo>/scripts`` in a development checkout).
# The compiled binaries live in the sibling ``<prefix>/bin`` directory --
# three parents up from this file, then back down into ``bin``. In a
# development checkout this path simply does not exist, and resolution
# falls through to ``PATH``.
_SIBLING_BIN_DIR = Path(__file__).resolve().parent.parent.parent / "bin"

SUBCOMMANDS = ("pull", "serve", "recommend", "capacity", "engine", "bench")

USAGE = """usage: fastmlx <subcommand> [args ...]

subcommands:
  pull        pull a pinned Hugging Face model snapshot
  serve       fit-check, admit, then serve an OpenAI-compatible engine
  recommend   rank local model packs that fit this host by measured quality
  capacity    run the capacity-check binary directly
  engine      run the Swift serving engine binary directly (escape hatch)
  bench       measure decode throughput against any OpenAI-compatible endpoint

Run "fastmlx <subcommand> --help" for subcommand-specific help.
"""


def _resolve_sibling_or_path_binary(name: str) -> Optional[str]:
    sibling = _SIBLING_BIN_DIR / name
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return shutil.which(name)


def _refuse(exit_code: int, message: str) -> NoReturn:
    print(f"fastmlx: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def _exec_binary(subcommand: str, binary_name: str, args: Sequence[str]) -> None:
    """Resolve ``binary_name`` and exec it (argv passthrough, no shell).

    Refuses cleanly (exit 2) if the binary cannot be found next to this
    dispatcher's own install or on ``PATH`` -- this never raises an
    unhandled exception, and it never falls back to silently doing nothing.
    ``os.execv`` never returns on success (the calling process image is
    replaced); this function only returns when a test double stands in for
    it, which is why it has no ``NoReturn`` annotation.
    """
    resolved = _resolve_sibling_or_path_binary(binary_name)
    if resolved is None:
        _refuse(
            2,
            f"{subcommand} refused: binary {binary_name!r} not found "
            f"(checked {_SIBLING_BIN_DIR} and PATH)",
        )
    os.execv(resolved, [resolved, *args])


def main(argv: Optional[Sequence[str]] = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)

    if not raw_argv:
        sys.stdout.write(USAGE)
        raise SystemExit(2)

    if raw_argv[0] in ("-h", "--help"):
        sys.stdout.write(USAGE)
        raise SystemExit(0)

    subcommand, rest = raw_argv[0], list(raw_argv[1:])

    if subcommand == "pull":
        _pull.main(rest)
        return
    if subcommand == "bench":
        _bench.main(rest)
        return
    if subcommand == "serve":
        _launch.main(["serve", *rest])
        return
    if subcommand == "recommend":
        _recommend.main(["recommend", *rest])
        return
    if subcommand == "capacity":
        _exec_binary("capacity", CAPACITY_BINARY_NAME, rest)
        return
    if subcommand == "engine":
        _exec_binary("engine", ENGINE_BINARY_NAME, rest)
        return

    sys.stderr.write(f"fastmlx: unknown subcommand: {subcommand!r}\n\n{USAGE}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
