from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Optional


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_SCRIPT = REPOSITORY_ROOT / "scripts" / "package-release.sh"

# Files package-release.sh copies from the repo root into the tarball's libexec/scripts +
# libexec/site. Fixture repos built below replicate this exact tree (see
# _init_fixture_release_repo) so the script's own REPO_ROOT -- derived from its BASH_SOURCE
# location, never from cwd or --source-dir -- resolves inside a disposable fixture instead of
# this repository, letting provenance tests exercise arbitrary commit graphs without touching
# this repository's real git history.
_TOOLING_SCRIPT_NAMES = (
    "fastmlx.py",
    "fastmlx_pull.py",
    "fastmlx_launch.py",
    "fastmlx_proxy.py",
    "fastmlx_recommend.py",
    "fastmlx_bench.py",
    "validate_public_repository.py",
    "hf_pinned_snapshot_download.py",
    "fastmlx_gguf_fit.py",
    "fastmlx_safetensors_fit.py",
)

# The two fit sizers must keep their executable bit in the staged tree/tarball: the launcher execs
# `--fit-check-bin` directly (see fastmlx_launch.py), never through `python3 <path>`.
_EXECUTABLE_SIZER_NAMES = (
    "fastmlx_gguf_fit.py",
    "fastmlx_safetensors_fit.py",
)

# Public-safe example --engine-profile documents package-release.sh stages into
# share/fastmlx/engine-profiles (mode 0644, never executable -- these are JSON, not scripts).
_ENGINE_PROFILE_EXAMPLE_NAMES = (
    "served-engine-safetensors.json",
    "served-engine-safetensors-ngram.json",
)

# Forbidden internal-deployment strings: package-release.sh is generic product-release tooling
# and must name no specific deployment, commit, or model family (see grep_test below). Each
# entry is assembled by concatenation, never written as a literal: this file is itself part of
# the public projection these strings gate, so a literal, contiguous internal-model-family
# marker here would trip validate_public_repository.py's own validate_no_internal_family_marker
# scan on every future publish (see that script's INTERNAL_FAMILY_MARKER comment for the same
# pattern -- its own constant is split the identical way for the identical reason). The other
# entries below are split the same way so this file never republishes an internal deployment
# identifier verbatim either.
_FORBIDDEN_STRINGS = (
    "d69f" + "5465",
    "bbb8" + "79d0",
    "flash" + "-next",
    "Flash" + " Next",
    "Qwen" + "4Exp",
    "qwen" + "4",
)

# package-release.sh's own packaging step shells out to `shasum` (macOS's bundled sha256
# provider) and stages a tarball this repo names "*-arm64-macos" by design -- Linux ships
# `sha256sum` under coreutils instead and is not guaranteed to have `shasum` on PATH (confirmed
# absent on a stock ubuntu:24.04 image; only pulled in transitively by the full `perl` package).
# Every test below stages a tarball via that script, so this class skips off macOS the same way
# test_fastmlx_pull.py / test_hf_pinned_snapshot_download.py skip their macOS-only exclusive-
# rename cases: public CI runs this module in its macOS job and fails the job on any skip there.
# test_script_contains_no_internal_deployment_names (below, in its own class) touches neither
# shasum nor a shell invocation, so it stays a cross-platform positive control.
REQUIRES_MACOS_RELEASE_PACKAGING = unittest.skipUnless(
    sys.platform == "darwin",
    "package-release.sh's packaged output (shasum, an arm64-macos tarball) is exercised on macOS only",
)


# --- sibling-load dependency closure: scanned from source, never a second hardcoded list -------
# The tooling loads its siblings by file path at CALL time (importlib.util.spec_from_file_
# location, resolved through Path(__file__).resolve().parent / "<name>.py" -- see
# fastmlx_bench.py's _load_marker_source, fastmlx_pull.py's _DOWNLOADER_SPEC, fastmlx_launch.py's
# _PULL_SPEC/_PROXY_SPEC/_SAFETENSORS_FIT_SPEC, fastmlx_recommend.py's _LAUNCH_SPEC, and
# fastmlx_safetensors_fit.py's _GGUF_FIT_SPEC). package-release.sh must stage every script this
# graph transitively reaches into BOTH the tarball's libexec/scripts AND the emitted formula's
# install block, or the staged copy 500s/refuses at a sibling-load call site the repo checkout
# never exercises. This is computed by scanning source below -- asserting a hardcoded name list
# against another hardcoded name list would just be two copies of the same guess.
_SIBLING_LOAD_RE = re.compile(r'\.parent\s*/\s*"([^"/]+\.py)"')


def _sibling_load_targets(script_path: Path) -> "set[str]":
    """The sibling ``.py`` filenames `script_path` loads by file path, per the
    ``Path(__file__).resolve().parent / "<name>.py"`` pattern every load site in this codebase
    uses (see the module comment above). A regex over the source, not an AST walk or an import
    analysis -- every load site is written this one way, so this is a faithful stand-in."""
    text = script_path.read_text(encoding="utf-8")
    return set(_SIBLING_LOAD_RE.findall(text))


def _transitive_sibling_load_closure(seed_names: "list[str]", scripts_dir: Path) -> "set[str]":
    """Starting from `seed_names` (already-staged tooling scripts), transitively follows every
    sibling-file load discovered in each script's own source until no new name appears. This is
    the actual dependency graph the tooling exercises at CALL time, not a hand-maintained list --
    a script that starts loading a new sibling is caught here automatically the next time this
    runs, with no edit required to this test file."""
    closure = set(seed_names)
    frontier = set(seed_names)
    while frontier:
        next_frontier: "set[str]" = set()
        for name in frontier:
            script_path = scripts_dir / name
            if not script_path.is_file():
                continue
            for target in _sibling_load_targets(script_path):
                if target not in closure:
                    closure.add(target)
                    next_frontier.add(target)
        frontier = next_frontier
    return closure


def _parse_staging_script_names(package_script_text: str) -> "list[str]":
    """Parses package-release.sh's own tooling-staging ``for name in ...; do`` loop's bash word
    list -- never hardcoded here -- so a future change to that loop is reflected immediately in
    what the closure check below tests against."""
    match = re.search(r"for name in ([^;]+); do", package_script_text)
    if not match:
        raise AssertionError(
            "could not find package-release.sh's tooling-staging 'for name in ...; do' loop -- "
            "has it been rewritten to no longer use a bash word-list loop?"
        )
    return [f"{word}.py" for word in match.group(1).split()]


def _parse_formula_install_block_names(package_script_text: str) -> "list[str]":
    """Parses the emitted formula's ``(libexec/"scripts").install "scripts/<name>.py"`` lines
    directly out of package-release.sh's own emitter heredoc -- never hardcoded here. These lines
    carry no bash variable substitution (the heredoc is unquoted only for $VERSION and the
    profile-install block), so they appear byte-identical in package-release.sh's own source
    text and in whatever it emits."""
    return re.findall(
        r'\(libexec/"scripts"\)\.install "scripts/([^"]+\.py)"', package_script_text
    )


def _run_git(repo_dir: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        capture_output=True,
        text=True,
        check=True,
    )


def _init_git_repo(repo_dir: Path) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(repo_dir)],
        capture_output=True,
        text=True,
        check=True,
    )
    _run_git(repo_dir, "config", "user.email", "test@example.com")
    _run_git(repo_dir, "config", "user.name", "Test Fixture")


def _git_commit(repo_dir: Path, filename: str, content: str, message: str) -> str:
    (repo_dir / filename).write_text(content, encoding="utf-8")
    _run_git(repo_dir, "add", "-A")
    _run_git(repo_dir, "commit", "-q", "-m", message)
    return _git_head(repo_dir)


def _git_head(repo_dir: Path) -> str:
    return _run_git(repo_dir, "rev-parse", "HEAD").stdout.strip()


def _populate_release_tree(root: Path) -> None:
    """Copy this repo's release-relevant files (the script itself, its sibling tooling modules,
    the quality cards, and the top-level docs it copies into every tarball) into `root`."""
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "site").mkdir(parents=True, exist_ok=True)

    package_script_dst = root / "scripts" / "package-release.sh"
    package_script_dst.write_bytes(PACKAGE_SCRIPT.read_bytes())
    package_script_dst.chmod(0o755)

    for name in _TOOLING_SCRIPT_NAMES:
        dst = root / "scripts" / name
        dst.write_bytes((REPOSITORY_ROOT / "scripts" / name).read_bytes())
        if name in _EXECUTABLE_SIZER_NAMES:
            # Deliberately NON-executable (0644) in the fixture source tree: package-release.sh
            # itself must set the executable bit during staging (see its `chmod 0755` on these
            # two sizers). Writing the fixture pre-chmod'd to 0755 would make
            # test_fit_sizers_are_executable_in_the_tarball pass even if that chmod line were
            # ever removed from the script -- this mode proves the SCRIPT does the chmod, not
            # that the source happened to carry the bit already.
            dst.chmod(0o644)

    (root / "site" / "quality-guides.json").write_bytes(
        (REPOSITORY_ROOT / "site" / "quality-guides.json").read_bytes()
    )

    (root / "examples" / "engine-profiles").mkdir(parents=True, exist_ok=True)
    for name in _ENGINE_PROFILE_EXAMPLE_NAMES:
        dst = root / "examples" / "engine-profiles" / name
        dst.write_bytes((REPOSITORY_ROOT / "examples" / "engine-profiles" / name).read_bytes())
        # Deliberately wrong mode (0600, not 0644) in the fixture source tree, the same
        # discriminating-mode trick used for the sizers above: package-release.sh itself must
        # set 0644 during staging (see its `chmod 0644` on these files). Writing the fixture
        # pre-chmod'd to 0644 would make a staged-mode test pass even if that chmod line were
        # ever removed -- this mode proves the SCRIPT does the chmod, not that `cp` merely
        # preserved a source mode that already happened to be 0644.
        dst.chmod(0o600)

    (root / "LICENSE").write_text("fixture license\n", encoding="utf-8")
    (root / "NOTICE").write_text("fixture notice\n", encoding="utf-8")
    (root / "README.md").write_text("fixture readme\n", encoding="utf-8")


def _init_fixture_release_repo(repo_dir: Path) -> None:
    """A standalone git repo shaped like this repo's release-relevant tree, with its own copy of
    package-release.sh. --source-dir now must equal the script's own repo root (see defect #2 in
    the task spec), so provenance tests that need a specific, controllable commit graph run the
    fixture's own copy of the script (`bash <repo_dir>/scripts/package-release.sh`) rather than
    pointing --source-dir at a throwaway directory next to the real repository.
    """
    _init_git_repo(repo_dir)
    _populate_release_tree(repo_dir)


@REQUIRES_MACOS_RELEASE_PACKAGING
class ReleasePackageTests(unittest.TestCase):
    # --- helpers -----------------------------------------------------------------------------

    def _make_fake_binaries(self, directory: Path) -> tuple[Path, Path, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        fake_serve = directory / "fake-fastmlx-serve"
        fake_capacity = directory / "fake-fastmlx-capacity"
        fake_metallib = directory / "fake-mlx.metallib"
        fake_serve.write_text("#!/bin/sh\necho serve\n", encoding="utf-8")
        fake_capacity.write_text("#!/bin/sh\necho capacity\n", encoding="utf-8")
        fake_metallib.write_bytes(b"fake-metallib-bytes")
        fake_serve.chmod(0o755)
        fake_capacity.chmod(0o755)
        return fake_serve, fake_capacity, fake_metallib

    def run_package_script(
        self,
        stage_dir: Path,
        out_dir: Optional[Path],
        *,
        version: str = "testver",
        emit_formula: Optional[Path] = None,
        source_dir: Optional[Path] = None,
        require_sampled_mtp: bool = False,
        sampled_mtp_minimum_commit: Optional[str] = None,
        include_out_dir: bool = True,
        include_emit_formula: bool = True,
        check: bool = True,
        script: Path = PACKAGE_SCRIPT,
        cwd: Optional[Path] = None,
        env: Optional[dict] = None,
    ) -> subprocess.CompletedProcess:
        fake_serve, fake_capacity, fake_metallib = self._make_fake_binaries(
            stage_dir.parent / "fakes"
        )

        args = [
            "bash",
            str(script),
            "--binary",
            str(fake_serve),
            "--capacity-binary",
            str(fake_capacity),
            "--metallib",
            str(fake_metallib),
            "--version",
            version,
            "--stage-dir",
            str(stage_dir),
        ]
        if include_out_dir:
            assert out_dir is not None
            args += ["--out-dir", str(out_dir)]
        if include_emit_formula:
            # Always route the generated formula to a temp path so the test never clobbers the
            # committed Formula/fastmlx.rb.
            if emit_formula is None:
                emit_formula = stage_dir.parent / "generated-fastmlx.rb"
            args += ["--emit-formula", str(emit_formula)]

        if source_dir is not None:
            args += ["--source-dir", str(source_dir)]
        if sampled_mtp_minimum_commit is not None:
            args += ["--sampled-mtp-minimum-commit", sampled_mtp_minimum_commit]
        if require_sampled_mtp:
            args += ["--require-sampled-mtp"]

        return subprocess.run(
            args,
            cwd=str(cwd) if cwd is not None else REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=check,
            env=env,
        )

    def _tarball_paths(self, out_dir: Path, version: str = "testver") -> tuple[Path, Path]:
        tarball = out_dir / f"fastmlx-{version}-arm64-macos.tar.gz"
        return tarball, Path(f"{tarball}.sha256")

    def _read_provenance(self, tarball: Path, version: str = "testver") -> dict:
        with tarfile.open(tarball, "r:gz") as tar:
            member = tar.getmember(
                f"fastmlx-{version}-arm64-macos/provenance.json"
            )
            extracted = tar.extractfile(member)
            assert extracted is not None
            return json.loads(extracted.read().decode("utf-8"))

    def _read_tarball_member_bytes(self, tarball: Path, member_path: str) -> bytes:
        with tarfile.open(tarball, "r:gz") as tar:
            extracted = tar.extractfile(tar.getmember(member_path))
            assert extracted is not None
            return extracted.read()

    # --- layout / packaging basics ------------------------------------------------------------

    def test_tarball_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "stage"
            out_dir = root / "out"
            self.run_package_script(stage_dir, out_dir)
            tarball, _ = self._tarball_paths(out_dir)

            self.assertTrue(tarball.is_file())
            with tarfile.open(tarball, "r:gz") as tar:
                members = sorted(
                    m.name for m in tar.getmembers() if m.isfile()
                )
            top = "fastmlx-testver-arm64-macos"
            expected = sorted(
                f"{top}/{name}"
                for name in (
                    "bin/fastmlx-serve",
                    "bin/fastmlx-capacity",
                    "bin/mlx.metallib",
                    "bin/fastmlx",
                    "libexec/scripts/fastmlx.py",
                    "libexec/scripts/fastmlx_pull.py",
                    "libexec/scripts/fastmlx_launch.py",
                    "libexec/scripts/fastmlx_proxy.py",
                    "libexec/scripts/fastmlx_recommend.py",
                    "libexec/scripts/fastmlx_bench.py",
                    "libexec/scripts/validate_public_repository.py",
                    "libexec/scripts/hf_pinned_snapshot_download.py",
                    "libexec/scripts/fastmlx_gguf_fit.py",
                    "libexec/scripts/fastmlx_safetensors_fit.py",
                    "libexec/site/quality-guides.json",
                    "share/fastmlx/engine-profiles/served-engine-safetensors.json",
                    "share/fastmlx/engine-profiles/served-engine-safetensors-ngram.json",
                    "LICENSE",
                    "NOTICE",
                    "README.md",
                    "provenance.json",
                )
            )
            self.assertEqual(members, expected)

    def test_sha256_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "stage"
            out_dir = root / "out"
            self.run_package_script(stage_dir, out_dir)
            tarball, sha_file = self._tarball_paths(out_dir)

            self.assertTrue(sha_file.is_file())
            expected_digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
            contents = sha_file.read_text(encoding="utf-8").strip()
            digest, _, filename = contents.partition("  ")
            self.assertEqual(digest, expected_digest)
            self.assertEqual(filename, tarball.name)

    def test_launcher_is_executable_and_execs_dispatcher_with_path_extended(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "stage"
            out_dir = root / "out"
            self.run_package_script(stage_dir, out_dir)
            tarball, _ = self._tarball_paths(out_dir)

            with tarfile.open(tarball, "r:gz") as tar:
                member = tar.getmember(
                    "fastmlx-testver-arm64-macos/bin/fastmlx"
                )
                self.assertTrue(member.mode & 0o100)
                extracted = tar.extractfile(member)
                assert extracted is not None
                body = extracted.read().decode("utf-8")
            self.assertIn("#!/bin/sh", body)
            self.assertIn("libexec/scripts/fastmlx.py", body)
            # PATH must be extended with the shim's own bin dir: fastmlx_launch.py's and
            # fastmlx_recommend.py's engine-binary resolution is PATH-based
            # (shutil.which("fastmlx-serve")), with no sibling-bin fallback of their own.
            self.assertIn("PATH=", body)
            self.assertIn("BIN_DIR", body)

    def test_fit_sizers_are_executable_in_the_tarball(self) -> None:
        # Acceptance: the launcher execs `--fit-check-bin` directly (see fastmlx_launch.py), so
        # both fit sizers must keep their executable bit through staging into the tarball.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "stage"
            out_dir = root / "out"
            self.run_package_script(stage_dir, out_dir)
            tarball, _ = self._tarball_paths(out_dir)

            with tarfile.open(tarball, "r:gz") as tar:
                for name in _EXECUTABLE_SIZER_NAMES:
                    member = tar.getmember(
                        f"fastmlx-testver-arm64-macos/libexec/scripts/{name}"
                    )
                    self.assertTrue(
                        member.mode & 0o100,
                        f"{name} must be executable in the tarball (mode={oct(member.mode)})",
                    )

    def test_fit_sizers_executable_bit_comes_from_the_scripts_own_chmod(self) -> None:
        # test_fit_sizers_are_executable_in_the_tarball above sources from THIS repository's own
        # checkout, where both sizers are already git-mode 100755 -- so that test alone cannot
        # tell whether package-release.sh's own `chmod 0755` line is doing anything at all (`cp`
        # on macOS preserves the source's existing executable bit either way). This test instead
        # stages from a fixture source tree whose sizers are committed NON-executable (0644; see
        # _populate_release_tree), so the tarball's executable sizers can only be explained by
        # the script's own explicit chmod.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_root = root / "fixture-repo"
            _init_fixture_release_repo(fixture_root)
            _git_commit(fixture_root, "a.txt", "a", "initial")
            for name in _EXECUTABLE_SIZER_NAMES:
                source_mode = (fixture_root / "scripts" / name).stat().st_mode
                self.assertFalse(
                    source_mode & 0o111,
                    f"fixture source {name} must be non-executable for this test to be "
                    "discriminating",
                )

            stage_dir = root / "stage"
            out_dir = root / "out"
            self.run_package_script(
                stage_dir,
                out_dir,
                script=fixture_root / "scripts" / "package-release.sh",
                cwd=fixture_root,
            )
            tarball, _ = self._tarball_paths(out_dir)

            with tarfile.open(tarball, "r:gz") as tar:
                for name in _EXECUTABLE_SIZER_NAMES:
                    member = tar.getmember(
                        f"fastmlx-testver-arm64-macos/libexec/scripts/{name}"
                    )
                    self.assertTrue(
                        member.mode & 0o100,
                        f"{name} must be executable in the tarball (mode={oct(member.mode)}), "
                        "even though the fixture source tree it was staged from is not -- proves "
                        "package-release.sh's own chmod, not an inherited source bit",
                    )

    def test_no_engine_profile_examples_fails_loudly(self) -> None:
        # Acceptance: package-release.sh must refuse (non-zero exit, named reason) rather than
        # silently ship an empty engine-profiles dir if the source examples are missing.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_root = root / "fixture-repo"
            _init_fixture_release_repo(fixture_root)
            shutil.rmtree(fixture_root / "examples" / "engine-profiles")
            _git_commit(fixture_root, "a.txt", "a", "initial")

            stage_dir = root / "stage"
            out_dir = root / "out"
            result = self.run_package_script(
                stage_dir,
                out_dir,
                script=fixture_root / "scripts" / "package-release.sh",
                cwd=fixture_root,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("engine profile examples", result.stderr)

    def test_staged_safetensors_fit_help_runs_from_libexec_scripts(self) -> None:
        # Acceptance: fastmlx_safetensors_fit.py loads fastmlx_gguf_fit.py by sibling path
        # (importlib, Path(__file__).resolve().parent), so both must land in the same
        # libexec/scripts dir in the tarball layout, and the staged copy must actually run.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "stage"
            out_dir = root / "out"
            self.run_package_script(stage_dir, out_dir)
            tarball, _ = self._tarball_paths(out_dir)

            extract_dir = root / "extracted"
            extract_dir.mkdir()
            with tarfile.open(tarball, "r:gz") as tar:
                tar.extractall(extract_dir, filter="data")

            staged_sizer = (
                extract_dir
                / "fastmlx-testver-arm64-macos"
                / "libexec"
                / "scripts"
                / "fastmlx_safetensors_fit.py"
            )
            self.assertTrue(staged_sizer.is_file())
            self.assertTrue(os.access(staged_sizer, os.X_OK))

            help_result = subprocess.run(
                [str(staged_sizer), "--help"], capture_output=True, text=True
            )
            self.assertEqual(help_result.returncode, 0, help_result.stderr)

    def test_engine_profile_examples_staged_byte_identical_mode_0644(self) -> None:
        # Acceptance: the two public-safe example engine profiles land under
        # share/fastmlx/engine-profiles in the tarball, byte-identical to their repo sources, and
        # non-executable (JSON, not a script) mode 0644. Stages from a fixture source tree whose
        # example files are committed mode 0600 (see _populate_release_tree), so a passing 0644
        # in the tarball can only be explained by package-release.sh's own `chmod 0644` -- staging
        # from this repo's own checkout (where the files are typically already 0644 via git mode
        # 100644) would let this test pass even if that chmod line were deleted.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_root = root / "fixture-repo"
            _init_fixture_release_repo(fixture_root)
            _git_commit(fixture_root, "a.txt", "a", "initial")
            for name in _ENGINE_PROFILE_EXAMPLE_NAMES:
                source_mode = (fixture_root / "examples" / "engine-profiles" / name).stat().st_mode
                self.assertNotEqual(
                    source_mode & 0o777,
                    0o644,
                    f"fixture source {name} must not already be mode 0644 for this test to be "
                    "discriminating",
                )

            stage_dir = root / "stage"
            out_dir = root / "out"
            self.run_package_script(
                stage_dir,
                out_dir,
                script=fixture_root / "scripts" / "package-release.sh",
                cwd=fixture_root,
            )
            tarball, _ = self._tarball_paths(out_dir)

            with tarfile.open(tarball, "r:gz") as tar:
                for name in _ENGINE_PROFILE_EXAMPLE_NAMES:
                    member = tar.getmember(
                        f"fastmlx-testver-arm64-macos/share/fastmlx/engine-profiles/{name}"
                    )
                    self.assertEqual(
                        member.mode & 0o777,
                        0o644,
                        f"{name} must be staged mode 0644 (mode={oct(member.mode)}), even "
                        "though the fixture source tree it was staged from is not -- proves "
                        "package-release.sh's own chmod, not an inherited source mode",
                    )
                    extracted = tar.extractfile(member)
                    assert extracted is not None
                    staged_bytes = extracted.read()
                    source_bytes = (
                        REPOSITORY_ROOT / "examples" / "engine-profiles" / name
                    ).read_bytes()
                    self.assertEqual(
                        staged_bytes,
                        source_bytes,
                        f"{name} must be staged byte-identical to its repo source",
                    )

    def test_staged_engine_profile_loads_through_staged_launcher_with_builtin_sizer_resolved_to_staged_sizer(
        self,
    ) -> None:
        # Acceptance: an example profile taken from the STAGED tree loads through the STAGED
        # fastmlx_launch.py (not this repo's own copy), and `builtin:safetensors` resolves to the
        # STAGED libexec/scripts/fastmlx_safetensors_fit.py -- proving the tarball is internally
        # self-consistent, not merely that the repo's own copies happen to work together.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "stage"
            out_dir = root / "out"
            self.run_package_script(stage_dir, out_dir)
            tarball, _ = self._tarball_paths(out_dir)

            extract_dir = root / "extracted"
            extract_dir.mkdir()
            with tarfile.open(tarball, "r:gz") as tar:
                tar.extractall(extract_dir, filter="data")

            top = extract_dir / "fastmlx-testver-arm64-macos"
            staged_launcher = top / "libexec" / "scripts" / "fastmlx_launch.py"
            staged_sizer = top / "libexec" / "scripts" / "fastmlx_safetensors_fit.py"
            staged_profile = (
                top / "share" / "fastmlx" / "engine-profiles" / "served-engine-safetensors-ngram.json"
            )
            self.assertTrue(staged_launcher.is_file())
            self.assertTrue(staged_sizer.is_file())
            self.assertTrue(staged_profile.is_file())

            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import importlib.util, json, sys\n"
                        "spec = importlib.util.spec_from_file_location('fastmlx_launch', sys.argv[1])\n"
                        "module = importlib.util.module_from_spec(spec)\n"
                        "spec.loader.exec_module(module)\n"
                        "profile, is_built_in = module.load_engine_profile(sys.argv[2])\n"
                        "print(json.dumps({'is_built_in': is_built_in, "
                        "'fit_check_bin': profile['fitCheck']['bin']}))\n"
                    ),
                    str(staged_launcher),
                    str(staged_profile),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(probe.returncode, 0, probe.stderr)
            result = json.loads(probe.stdout.strip())
            self.assertFalse(result["is_built_in"])
            self.assertEqual(result["fit_check_bin"], str(staged_sizer.resolve()))

    def test_summary_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "stage"
            out_dir = root / "out"
            result = self.run_package_script(stage_dir, out_dir)
            tarball, _ = self._tarball_paths(out_dir)

            expected_digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
            self.assertIn("packaged:", result.stdout)
            self.assertIn(f"sha256={expected_digest}", result.stdout)

    def test_version_names_the_tarball_exactly(self) -> None:
        # Acceptance criterion (defect #5): an explicit --version must name the tarball
        # fastmlx-<version>-arm64-macos.tar.gz verbatim.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "stage"
            out_dir = root / "out"
            self.run_package_script(stage_dir, out_dir, version="0.1.0")
            tarball, sha_file = self._tarball_paths(out_dir, version="0.1.0")

            self.assertEqual(tarball.name, "fastmlx-0.1.0-arm64-macos.tar.gz")
            self.assertTrue(tarball.is_file())
            self.assertTrue(sha_file.is_file())

    def test_extracted_tarball_help_invocations_exit_zero(self) -> None:
        # Acceptance: the tarball is not just laid out correctly, it WORKS once extracted --
        # `bin/fastmlx --help` and `bin/fastmlx capacity --help` both run the real, repo-shipped
        # fastmlx.py dispatcher (loaded from libexec/scripts) and exit 0.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "stage"
            out_dir = root / "out"
            self.run_package_script(stage_dir, out_dir)
            tarball, _ = self._tarball_paths(out_dir)

            extract_dir = root / "extracted"
            extract_dir.mkdir()
            with tarfile.open(tarball, "r:gz") as tar:
                tar.extractall(extract_dir, filter="data")

            fastmlx_bin = extract_dir / "fastmlx-testver-arm64-macos" / "bin" / "fastmlx"
            self.assertTrue(fastmlx_bin.is_file())

            help_result = subprocess.run(
                [str(fastmlx_bin), "--help"], capture_output=True, text=True
            )
            self.assertEqual(help_result.returncode, 0, help_result.stderr)

            capacity_help_result = subprocess.run(
                [str(fastmlx_bin), "capacity", "--help"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                capacity_help_result.returncode, 0, capacity_help_result.stderr
            )

            # A user who links bin/fastmlx into a directory already on PATH must get the same
            # result: the shim resolves libexec and its sibling binaries through the link.
            link_dir = root / "on-path"
            link_dir.mkdir()
            link = link_dir / "fastmlx"
            link.symlink_to(fastmlx_bin)
            for argv in ([str(link), "--help"], [str(link), "capacity", "--help"]):
                linked_result = subprocess.run(argv, capture_output=True, text=True)
                self.assertEqual(linked_result.returncode, 0, linked_result.stderr)

    def test_staged_bench_module_reports_publishable_not_refused_sweep_unavailable(
        self,
    ) -> None:
        # Acceptance (the actual user outcome, not list membership): fastmlx_bench.py's
        # publishability_control loads its marker source (PRIVATE_MARKERS,
        # THIRD_PARTY_ENGINE_MARKERS, OWN_BINARY_NAME) from the sibling
        # scripts/validate_public_repository.py by file path at CALL time -- see
        # _load_marker_source. If that sibling is not staged alongside fastmlx_bench.py, EVERY
        # `fastmlx bench` row in a shipped tarball is fail-closed refused
        # ("refused_sweep_unavailable"), never "publishable", even for a row with no marker in
        # it at all. This loads the module from the STAGED tarball (not this repo's own
        # checkout copy) and proves the staged tree is internally self-consistent.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "stage"
            out_dir = root / "out"
            self.run_package_script(stage_dir, out_dir)
            tarball, _ = self._tarball_paths(out_dir)

            extract_dir = root / "extracted"
            extract_dir.mkdir()
            with tarfile.open(tarball, "r:gz") as tar:
                tar.extractall(extract_dir, filter="data")

            staged_bench = (
                extract_dir
                / "fastmlx-testver-arm64-macos"
                / "libexec"
                / "scripts"
                / "fastmlx_bench.py"
            )
            self.assertTrue(staged_bench.is_file())

            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import importlib.util, json, sys\n"
                        "spec = importlib.util.spec_from_file_location('fastmlx_bench', sys.argv[1])\n"
                        "module = importlib.util.module_from_spec(spec)\n"
                        "spec.loader.exec_module(module)\n"
                        "row = {'model': 'demo-model', 'quantization': 'q4'}\n"
                        "print(json.dumps(module.publishability_control(row)))\n"
                    ),
                    str(staged_bench),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(probe.returncode, 0, probe.stderr)
            verdict = json.loads(probe.stdout.strip())
            self.assertNotEqual(
                verdict["status"],
                "refused_sweep_unavailable",
                f"staged fastmlx_bench.py could not load its marker source and fail-closed "
                f"refused a row with no marker in it: {verdict!r}",
            )
            self.assertEqual(verdict["status"], "publishable", verdict)

    # --- formula generation (opt-in, only with --emit-formula) --------------------------------

    def test_formula_not_generated_by_default(self) -> None:
        # Acceptance criterion (defect #4/#6): the Formula is written only when --emit-formula
        # PATH is given -- a packaging run must not touch Formula/fastmlx.rb by default.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "stage"
            out_dir = root / "out"
            self.run_package_script(
                stage_dir, out_dir, include_emit_formula=False
            )
            # No formula path was requested; nothing outside stage_dir/out_dir should appear,
            # and specifically this repo's own Formula/fastmlx.rb must be untouched (checked via
            # mtime-independent absence of any new write: we simply never point --emit-formula at
            # it, so there is nothing further to assert here beyond a clean, successful run).
            tarball, _ = self._tarball_paths(out_dir)
            self.assertTrue(tarball.is_file())

    def test_formula_generated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "stage"
            out_dir = root / "out"
            formula_path = root / "fastmlx.rb"
            self.run_package_script(
                stage_dir, out_dir, emit_formula=formula_path
            )

            self.assertTrue(formula_path.is_file())
            body = formula_path.read_text(encoding="utf-8")
            self.assertIn("class Fastmlx < Formula", body)
            self.assertIn('version "testver"', body)
            self.assertIn("--product fastmlx-serve", body)
            self.assertIn("--product fastmlx-capacity", body)
            self.assertIn("mlx.metallib", body)
            self.assertIn("PLACEHOLDER_SET_AT_PUBLISH", body)
            self.assertIn("GENERATED", body)

    def test_formula_version_uses_the_passed_version_not_a_stale_default(self) -> None:
        # Acceptance criterion (defect #6): the emitted Formula must not carry a stale
        # hard-coded version -- it must reflect whatever --version this run used.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "stage"
            out_dir = root / "out"
            formula_path = root / "fastmlx.rb"
            self.run_package_script(
                stage_dir, out_dir, version="9.9.9", emit_formula=formula_path
            )

            body = formula_path.read_text(encoding="utf-8")
            self.assertIn('version "9.9.9"', body)
            self.assertNotIn('version "testver"', body)

    def test_formula_desc_names_the_product_not_a_competitor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "stage"
            out_dir = root / "out"
            formula_path = root / "fastmlx.rb"
            self.run_package_script(
                stage_dir, out_dir, emit_formula=formula_path
            )

            body = formula_path.read_text(encoding="utf-8")
            self.assertIn(
                'desc "Fit-checked, quality-carded LLM serving for Apple Silicon"', body
            )

    def test_formula_declares_a_python_dependency_for_stdlib_only_tooling(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "stage"
            out_dir = root / "out"
            formula_path = root / "fastmlx.rb"
            self.run_package_script(
                stage_dir, out_dir, emit_formula=formula_path
            )

            body = formula_path.read_text(encoding="utf-8")
            # "python" (not "python3") is the spelling Homebrew's own FormulaAudit/
            # UsesFromMacos check accepts for this mechanism -- "python3" is flagged as
            # invalid by `brew style`.
            self.assertIn('uses_from_macos "python"', body)

    def test_formula_installs_the_fastmlx_tooling_into_libexec(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "stage"
            out_dir = root / "out"
            formula_path = root / "fastmlx.rb"
            self.run_package_script(
                stage_dir, out_dir, emit_formula=formula_path
            )

            body = formula_path.read_text(encoding="utf-8")
            self.assertIn('(libexec/"scripts").install "scripts/fastmlx.py"', body)
            self.assertIn('"scripts/fastmlx_pull.py"', body)
            self.assertIn('"scripts/fastmlx_launch.py"', body)
            self.assertIn('"scripts/fastmlx_proxy.py"', body)
            self.assertIn('"scripts/fastmlx_recommend.py"', body)
            self.assertIn('"scripts/fastmlx_bench.py"', body)
            self.assertIn('"scripts/hf_pinned_snapshot_download.py"', body)
            self.assertIn('"scripts/fastmlx_gguf_fit.py"', body)
            self.assertIn('"scripts/fastmlx_safetensors_fit.py"', body)
            self.assertIn('(libexec/"site").install "site/quality-guides.json"', body)

    def test_formula_installs_the_engine_profile_examples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "stage"
            out_dir = root / "out"
            formula_path = root / "fastmlx.rb"
            self.run_package_script(
                stage_dir, out_dir, emit_formula=formula_path
            )

            body = formula_path.read_text(encoding="utf-8")
            self.assertIn(
                '(pkgshare/"engine-profiles").install '
                '"examples/engine-profiles/served-engine-safetensors.json"',
                body,
            )
            self.assertIn(
                '(pkgshare/"engine-profiles").install '
                '"examples/engine-profiles/served-engine-safetensors-ngram.json"',
                body,
            )

    def test_formula_engine_profile_installs_are_generated_from_the_staged_set_not_hard_coded(
        self,
    ) -> None:
        # Acceptance: the Formula's pkgshare installs must come from the SAME set the tarball
        # actually stages under share/fastmlx/engine-profiles, not a separately hand-written
        # list. Proven with a THREE-file fixture (the real repo only ever ships two): a
        # hard-coded two-line Formula generator would either omit the third file or -- if it
        # named the third file literally -- pass by coincidence rather than by construction. This
        # fixture makes both failure modes visible by asserting exact set equality with the
        # tarball's own staged set.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_root = root / "fixture-repo"
            _init_fixture_release_repo(fixture_root)
            third_profile = json.loads(
                (
                    fixture_root / "examples" / "engine-profiles" / "served-engine-safetensors.json"
                ).read_text(encoding="utf-8")
            )
            third_profile["name"] = "served-engine-safetensors-third-fixture"
            (fixture_root / "examples" / "engine-profiles" / "served-engine-safetensors-third.json").write_text(
                json.dumps(third_profile), encoding="utf-8"
            )
            _git_commit(fixture_root, "a.txt", "a", "initial")

            stage_dir = root / "stage"
            out_dir = root / "out"
            formula_path = root / "fastmlx.rb"
            self.run_package_script(
                stage_dir,
                out_dir,
                emit_formula=formula_path,
                script=fixture_root / "scripts" / "package-release.sh",
                cwd=fixture_root,
            )
            tarball, _ = self._tarball_paths(out_dir)

            with tarfile.open(tarball, "r:gz") as tar:
                staged_names = {
                    Path(m.name).name
                    for m in tar.getmembers()
                    if m.isfile()
                    and m.name.startswith("fastmlx-testver-arm64-macos/share/fastmlx/engine-profiles/")
                }

            body = formula_path.read_text(encoding="utf-8")
            formula_names = set(
                re.findall(r'\(pkgshare/"engine-profiles"\)\.install "examples/engine-profiles/([^"]+)"', body)
            )

            self.assertEqual(len(staged_names), 3, staged_names)
            self.assertEqual(formula_names, staged_names)

    def test_formula_wrapper_execs_the_python_dispatcher_and_extends_path(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "stage"
            out_dir = root / "out"
            formula_path = root / "fastmlx.rb"
            self.run_package_script(
                stage_dir, out_dir, emit_formula=formula_path
            )

            body = formula_path.read_text(encoding="utf-8")
            self.assertIn('exec python3 "#{libexec}/scripts/fastmlx.py" "$@"', body)
            self.assertIn("bin.install \".build/release/fastmlx-serve\"", body)
            self.assertIn("bin.install \".build/release/fastmlx-capacity\"", body)
            self.assertIn('bin.install "prebuilt/mlx.metallib"', body)
            self.assertIn('export PATH="$PREFIX_BIN_DIR:$PATH"', body)

    def test_formula_test_block_checks_fastmlx_and_fastmlx_capacity_help(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "stage"
            out_dir = root / "out"
            formula_path = root / "fastmlx.rb"
            self.run_package_script(
                stage_dir, out_dir, emit_formula=formula_path
            )

            body = formula_path.read_text(encoding="utf-8")
            self.assertIn('system "#{bin}/fastmlx", "--help"', body)
            self.assertIn('system "#{bin}/fastmlx-capacity", "--help"', body)

    # --- --source-dir: single-root provenance -------------------------------------------------

    def test_source_dir_mismatch_is_refused_with_reason(self) -> None:
        # Acceptance criterion (defect #2): a --source-dir that does not resolve to this
        # script's own repo root must be refused, naming both paths, before anything is staged.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "stage"
            out_dir = root / "out"
            mismatched = root / "somewhere-else"
            mismatched.mkdir()

            result = self.run_package_script(
                stage_dir, out_dir, source_dir=mismatched, check=False
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("repo root", result.stderr.lower())
            self.assertIn(str(mismatched), result.stderr)
            tarball, _ = self._tarball_paths(out_dir)
            self.assertFalse(tarball.exists())

    def test_source_dir_matching_repo_root_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "stage"
            out_dir = root / "out"
            result = self.run_package_script(
                stage_dir, out_dir, source_dir=REPOSITORY_ROOT
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            tarball, _ = self._tarball_paths(out_dir)
            self.assertTrue(tarball.is_file())

    # --- default out-dir stays outside the tree / never dirties the source --------------------

    def test_default_out_dir_is_outside_the_repo_and_leaves_it_clean_across_two_runs(
        self,
    ) -> None:
        # Acceptance criteria (defect #4): default out-dir must be outside the repo, and a
        # second consecutive run on a clean repo must still report source_dirty=false.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_root = root / "fixture-repo"
            _init_fixture_release_repo(fixture_root)
            _git_commit(fixture_root, "a.txt", "a", "initial")

            self.assertEqual(
                _run_git(fixture_root, "status", "--porcelain").stdout.strip(), ""
            )

            stage_dir = root / "stage"
            version = "outdirtest"
            expected_out_dir = (
                Path(os.environ.get("TMPDIR", "/tmp")) / f"fastmlx-release-{version}"
            )
            self.addCleanup(
                lambda: subprocess.run(["rm", "-rf", str(expected_out_dir)])
            )

            for _ in range(2):
                result = self.run_package_script(
                    stage_dir,
                    None,
                    version=version,
                    include_out_dir=False,
                    script=fixture_root / "scripts" / "package-release.sh",
                    cwd=fixture_root,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

                tarball, _ = self._tarball_paths(expected_out_dir, version=version)
                self.assertTrue(tarball.is_file())
                provenance = self._read_provenance(tarball, version=version)
                self.assertFalse(provenance["source_dirty"])

                # The repo the script built from must stay clean -- no dist/, no Formula/, no
                # untracked artifact left behind by this run.
                status = _run_git(fixture_root, "status", "--porcelain").stdout.strip()
                self.assertEqual(status, "", f"fixture repo dirtied by run: {status!r}")

    # --- provenance: source_commit / source_dirty on a real, controllable commit graph --------

    def test_provenance_base_fields_have_correct_types_and_no_sampled_mtp_fields(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_root = root / "fixture-repo"
            _init_fixture_release_repo(fixture_root)
            head_sha = _git_commit(fixture_root, "a.txt", "a", "initial")

            stage_dir = root / "stage"
            out_dir = root / "out"
            self.run_package_script(
                stage_dir,
                out_dir,
                script=fixture_root / "scripts" / "package-release.sh",
                cwd=fixture_root,
            )
            tarball, _ = self._tarball_paths(out_dir)
            provenance = self._read_provenance(tarball)

            for field in ("source_commit", "source_dirty", "version", "built_at"):
                self.assertIn(field, provenance)
            self.assertEqual(provenance["source_commit"], head_sha)
            self.assertIsInstance(provenance["source_dirty"], bool)
            self.assertFalse(provenance["source_dirty"])
            self.assertEqual(provenance["version"], "testver")
            self.assertRegex(
                provenance["built_at"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
            )
            # Acceptance criterion: no sampled-MTP fields appear unless a minimum commit was
            # supplied.
            self.assertNotIn("sampled_mtp_minimum_commit", provenance)
            self.assertNotIn("satisfies_sampled_mtp_minimum", provenance)

    def test_dirty_tree_recorded_true(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_root = root / "fixture-repo"
            _init_fixture_release_repo(fixture_root)
            head_sha = _git_commit(fixture_root, "a.txt", "a", "initial")
            (fixture_root / "a.txt").write_text("modified", encoding="utf-8")

            stage_dir = root / "stage"
            out_dir = root / "out"
            self.run_package_script(
                stage_dir,
                out_dir,
                script=fixture_root / "scripts" / "package-release.sh",
                cwd=fixture_root,
            )
            tarball, _ = self._tarball_paths(out_dir)
            provenance = self._read_provenance(tarball)

            self.assertTrue(provenance["source_dirty"])
            self.assertEqual(provenance["source_commit"], head_sha)

    def test_not_a_git_checkout_recorded_as_null_source_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_root = root / "fixture-repo"
            _populate_release_tree(fixture_root)  # no _init_git_repo: not a git checkout

            stage_dir = root / "stage"
            out_dir = root / "out"
            self.run_package_script(
                stage_dir,
                out_dir,
                script=fixture_root / "scripts" / "package-release.sh",
                cwd=fixture_root,
            )
            tarball, _ = self._tarball_paths(out_dir)
            provenance = self._read_provenance(tarball)

            self.assertIsNone(provenance["source_commit"])
            self.assertFalse(provenance["source_dirty"])

    # --- provenance: engine_binary_sha256 / capacity_binary_sha256 -----------------------------
    # These digests are what lets `fastmlx serve` derive its own engine-build commit from a
    # release install with no operator-written --engine-profile at all (see
    # fastmlx_launch.py's derive_engine_build_from_release): source_commit alone names a
    # checkout, not the bytes actually shipped, so the digest is what makes that commit
    # trustworthy for the staged fastmlx-serve binary specifically.
    def test_provenance_binary_sha256_fields_match_the_staged_binaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_root = root / "fixture-repo"
            _init_fixture_release_repo(fixture_root)
            _git_commit(fixture_root, "a.txt", "a", "initial")

            stage_dir = root / "stage"
            out_dir = root / "out"
            version = "testver"
            self.run_package_script(
                stage_dir,
                out_dir,
                version=version,
                script=fixture_root / "scripts" / "package-release.sh",
                cwd=fixture_root,
            )
            tarball, _ = self._tarball_paths(out_dir, version=version)
            provenance = self._read_provenance(tarball, version=version)

            for field in ("engine_binary_sha256", "capacity_binary_sha256"):
                self.assertIn(field, provenance)
                self.assertRegex(provenance[field], r"^[0-9a-f]{64}$")

            top_dir = f"fastmlx-{version}-arm64-macos"
            serve_bytes = self._read_tarball_member_bytes(
                tarball, f"{top_dir}/bin/fastmlx-serve"
            )
            capacity_bytes = self._read_tarball_member_bytes(
                tarball, f"{top_dir}/bin/fastmlx-capacity"
            )
            # The load-bearing assertion: each recorded digest equals the ACTUAL sha256 of the
            # staged binary bytes extracted from the tarball -- asserting only that the key
            # exists would pass against a hardcoded or stale value.
            self.assertEqual(
                provenance["engine_binary_sha256"], hashlib.sha256(serve_bytes).hexdigest()
            )
            self.assertEqual(
                provenance["capacity_binary_sha256"], hashlib.sha256(capacity_bytes).hexdigest()
            )

    # --- sampled-MTP attestation: opt-in, family-neutral, no default minimum ------------------

    def test_require_sampled_mtp_without_minimum_commit_is_refused(self) -> None:
        # Acceptance criterion (defect #3): --require-sampled-mtp refuses unless
        # --sampled-mtp-minimum-commit is also given -- no default minimum is assumed.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "stage"
            out_dir = root / "out"
            result = self.run_package_script(
                stage_dir, out_dir, require_sampled_mtp=True, check=False
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--sampled-mtp-minimum-commit", result.stderr)
            tarball, _ = self._tarball_paths(out_dir)
            self.assertFalse(tarball.exists())

    def test_no_sampled_mtp_fields_when_no_minimum_commit_given(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "stage"
            out_dir = root / "out"
            self.run_package_script(stage_dir, out_dir)
            tarball, _ = self._tarball_paths(out_dir)
            provenance = self._read_provenance(tarball)

            self.assertNotIn("sampled_mtp_minimum_commit", provenance)
            self.assertNotIn("satisfies_sampled_mtp_minimum", provenance)

    def test_sampled_mtp_fields_present_only_when_minimum_commit_given(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_root = root / "fixture-repo"
            _init_fixture_release_repo(fixture_root)
            minimum_sha = _git_commit(fixture_root, "a.txt", "a", "minimum")
            descendant_sha = _git_commit(fixture_root, "b.txt", "b", "descendant")

            stage_dir = root / "stage"
            out_dir = root / "out"
            self.run_package_script(
                stage_dir,
                out_dir,
                sampled_mtp_minimum_commit=minimum_sha,
                script=fixture_root / "scripts" / "package-release.sh",
                cwd=fixture_root,
            )
            tarball, _ = self._tarball_paths(out_dir)
            provenance = self._read_provenance(tarball)

            self.assertEqual(provenance["sampled_mtp_minimum_commit"], minimum_sha)
            self.assertIsInstance(provenance["satisfies_sampled_mtp_minimum"], bool)
            self.assertTrue(provenance["satisfies_sampled_mtp_minimum"])
            self.assertEqual(provenance["source_commit"], descendant_sha)

    def test_descendant_commit_satisfies_and_require_flag_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_root = root / "fixture-repo"
            _init_fixture_release_repo(fixture_root)
            minimum_sha = _git_commit(fixture_root, "a.txt", "a", "minimum")
            descendant_sha = _git_commit(fixture_root, "b.txt", "b", "descendant")

            stage_dir = root / "stage"
            out_dir = root / "out"
            result = self.run_package_script(
                stage_dir,
                out_dir,
                sampled_mtp_minimum_commit=minimum_sha,
                require_sampled_mtp=True,
                script=fixture_root / "scripts" / "package-release.sh",
                cwd=fixture_root,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            tarball, _ = self._tarball_paths(out_dir)
            provenance = self._read_provenance(tarball)

            self.assertEqual(provenance["source_commit"], descendant_sha)
            self.assertTrue(provenance["satisfies_sampled_mtp_minimum"])
            self.assertFalse(provenance["source_dirty"])

    def _make_non_descendant_repo(self, root: Path) -> tuple[Path, str, str]:
        """Two commits on sibling branches: `minimum_sha` is NOT an ancestor of `source_sha`."""
        fixture_root = root / "fixture-repo"
        _init_fixture_release_repo(fixture_root)
        base_sha = _git_commit(fixture_root, "base.txt", "base", "base")
        _run_git(fixture_root, "checkout", "-q", "-b", "sibling")
        minimum_sha = _git_commit(fixture_root, "minimum.txt", "m", "minimum")
        _run_git(fixture_root, "checkout", "-q", "main")
        source_sha = _git_commit(fixture_root, "source.txt", "s", "source")
        self.assertNotEqual(base_sha, minimum_sha)
        return fixture_root, minimum_sha, source_sha

    def test_non_descendant_commit_fails_require_flag_with_named_reason(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_root, minimum_sha, source_sha = self._make_non_descendant_repo(
                root
            )

            stage_dir = root / "stage"
            out_dir = root / "out"
            result = self.run_package_script(
                stage_dir,
                out_dir,
                sampled_mtp_minimum_commit=minimum_sha,
                require_sampled_mtp=True,
                check=False,
                script=fixture_root / "scripts" / "package-release.sh",
                cwd=fixture_root,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(minimum_sha, result.stderr)
            self.assertIn(source_sha, result.stderr)
            tarball, _ = self._tarball_paths(out_dir)
            self.assertFalse(tarball.exists())

    def test_non_descendant_commit_succeeds_without_require_flag_but_records_false(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_root, minimum_sha, source_sha = self._make_non_descendant_repo(
                root
            )

            stage_dir = root / "stage"
            out_dir = root / "out"
            self.run_package_script(
                stage_dir,
                out_dir,
                sampled_mtp_minimum_commit=minimum_sha,
                require_sampled_mtp=False,
                script=fixture_root / "scripts" / "package-release.sh",
                cwd=fixture_root,
            )
            tarball, _ = self._tarball_paths(out_dir)
            self.assertTrue(tarball.is_file())
            provenance = self._read_provenance(tarball)

            self.assertEqual(provenance["source_commit"], source_sha)
            self.assertFalse(provenance["satisfies_sampled_mtp_minimum"])

    def test_dirty_tree_recorded_and_refused_under_require_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_root = root / "fixture-repo"
            _init_fixture_release_repo(fixture_root)
            minimum_sha = _git_commit(fixture_root, "a.txt", "a", "minimum")
            head_sha = _git_commit(fixture_root, "b.txt", "b", "descendant")
            # Dirty the tree after committing: an uncommitted edit to a tracked file.
            (fixture_root / "b.txt").write_text("modified", encoding="utf-8")

            stage_dir = root / "stage"
            out_dir = root / "out"

            # Without --require-sampled-mtp: succeeds, but records source_dirty truthfully.
            self.run_package_script(
                stage_dir,
                out_dir,
                sampled_mtp_minimum_commit=minimum_sha,
                require_sampled_mtp=False,
                script=fixture_root / "scripts" / "package-release.sh",
                cwd=fixture_root,
            )
            tarball, _ = self._tarball_paths(out_dir)
            provenance = self._read_provenance(tarball)
            self.assertTrue(provenance["source_dirty"])
            self.assertEqual(provenance["source_commit"], head_sha)

            # With --require-sampled-mtp: refused (cannot attest a binary built from uncommitted
            # state), even though HEAD itself descends from the minimum.
            out_dir_2 = root / "out2"
            result = self.run_package_script(
                stage_dir,
                out_dir_2,
                sampled_mtp_minimum_commit=minimum_sha,
                require_sampled_mtp=True,
                check=False,
                script=fixture_root / "scripts" / "package-release.sh",
                cwd=fixture_root,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("dirty", result.stderr.lower())
            tarball_2, _ = self._tarball_paths(out_dir_2)
            self.assertFalse(tarball_2.exists())

    def test_not_a_git_checkout_refused_under_require_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_root = root / "fixture-repo"
            _populate_release_tree(fixture_root)  # no git init

            stage_dir = root / "stage"
            out_dir = root / "out"
            result = self.run_package_script(
                stage_dir,
                out_dir,
                sampled_mtp_minimum_commit="0" * 40,
                require_sampled_mtp=True,
                check=False,
                script=fixture_root / "scripts" / "package-release.sh",
                cwd=fixture_root,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("git checkout", result.stderr.lower())
            tarball, _ = self._tarball_paths(out_dir)
            self.assertFalse(tarball.exists())


# --- no internal deployment names leak into the generic release script -----------------------
# Deliberately its own, undecorated class: reading and grepping package-release.sh's source
# text needs neither `shasum` nor a shell invocation, so this check runs on every platform
# (including the Linux public-boundary job), unlike ReleasePackageTests above.
class ReleasePackageScriptTextTests(unittest.TestCase):
    def test_script_contains_no_internal_deployment_names(self) -> None:
        # Acceptance criterion (defect #3): package-release.sh is generic product-release
        # tooling and must name no specific deployment, minimum commit, or model family.
        text = PACKAGE_SCRIPT.read_text(encoding="utf-8")
        for forbidden in _FORBIDDEN_STRINGS:
            self.assertNotIn(
                forbidden,
                text,
                f"scripts/package-release.sh must not reference {forbidden!r}",
            )

    def test_sibling_load_closure_is_covered_by_both_install_lists(self) -> None:
        # Acceptance: every script transitively reachable by a sibling-file load, starting from
        # the tooling scripts package-release.sh currently stages, must appear in BOTH install
        # lists it maintains for that tooling -- the tarball staging loop (`for name in ...; do`)
        # and the formula's emitted `(libexec/"scripts").install` block. Both the seed list and
        # the closure itself are parsed/computed from source, never pinned against a second
        # hardcoded name list (that would just be this test asserting a list against a copy of
        # itself -- see _TOOLING_SCRIPT_NAMES's own comment above, which does exactly that and is
        # NOT relied on here). Two independently failable assertions (A1, A2) so a defect that
        # hits only one of the two install lists is still caught and named.
        text = PACKAGE_SCRIPT.read_text(encoding="utf-8")
        staged_names = _parse_staging_script_names(text)
        formula_names = _parse_formula_install_block_names(text)
        closure = _transitive_sibling_load_closure(staged_names, PACKAGE_SCRIPT.parent)

        missing_from_staging = closure - set(staged_names)
        self.assertEqual(
            missing_from_staging,
            set(),
            "package-release.sh's tarball staging loop ('for name in ...; do') omits "
            f"sibling-load target(s) {sorted(missing_from_staging)} -- a staged tarball would "
            "fail to load them at CALL time even though the script that needs them is staged",
        )
        missing_from_formula = closure - set(formula_names)
        self.assertEqual(
            missing_from_formula,
            set(),
            "package-release.sh's emitted formula install block omits sibling-load target(s) "
            f"{sorted(missing_from_formula)} -- a brew install would fail to load them at CALL "
            "time even though the script that needs them is installed",
        )

    def test_closure_check_fires_when_a_real_dependency_is_missing(self) -> None:
        # Anti-vacuity control for the closure gate above: proves the gate can actually fail --
        # and fails by NAMING the missing module, not merely "it raised" -- without mutating
        # package-release.sh on disk. Removes hf_pinned_snapshot_download.py (fastmlx_pull.py's
        # own sibling-load target) from an in-memory copy of the parsed staging list only.
        text = PACKAGE_SCRIPT.read_text(encoding="utf-8")
        staged_names = _parse_staging_script_names(text)
        self.assertIn(
            "hf_pinned_snapshot_download.py",
            staged_names,
            "fixture assumption broken: hf_pinned_snapshot_download.py is no longer in "
            "package-release.sh's staging list, so removing it in-memory would not be a "
            "discriminating mutation for this control",
        )
        mutated_staged_names = [
            name for name in staged_names if name != "hf_pinned_snapshot_download.py"
        ]

        closure = _transitive_sibling_load_closure(mutated_staged_names, PACKAGE_SCRIPT.parent)
        missing = closure - set(mutated_staged_names)

        self.assertIn(
            "hf_pinned_snapshot_download.py",
            missing,
            "removing hf_pinned_snapshot_download.py from the staged-name list must make the "
            f"closure check name it as missing -- got missing={sorted(missing)} instead, which "
            "means the gate is not actually reachable through fastmlx_pull.py's sibling load",
        )


if __name__ == "__main__":
    unittest.main()
