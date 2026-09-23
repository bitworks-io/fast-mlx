#!/usr/bin/env bash
# fast-mlx: package a self-contained arm64 macOS release tarball (and, optionally, a generated
# Homebrew formula) for the `fastmlx` product.
#
# The product this packages is the `fastmlx` command (pull | serve | recommend | capacity |
# engine) plus its quality cards -- not the Swift engine on its own. The Swift binaries
# (fastmlx-serve, fastmlx-capacity) ship as the fit checker and research engine that `bin/fastmlx`
# and its libexec tooling depend on. By default this script builds both binaries with
# `swift build -c release`; pass --binary/--capacity-binary to inject prebuilt binaries instead
# (the test hook, so tests don't need a multi-minute release build).
#
# Why the metallib is colocated: the operator needs a distributable artifact that colocates the
# prebuilt MLX metallib (spike/prebuilt/mlx.metallib) next to the binaries the same way
# scripts/serve.sh does at dev time (MLX searches <bindir>/mlx.metallib first), so a downloaded
# tarball "just works" with no Metal Toolchain download.
#
# This script only STAGES artifacts locally. It does not create a tap repo, push a GitHub
# release, or publish anything -- publication is human-gated.
#
# Every staged tarball carries a top-level provenance.json recording the source commit, whether
# the tree was dirty, and the sha256 of each staged binary (engine_binary_sha256,
# capacity_binary_sha256) -- the digests let `fastmlx serve` derive its own engine-build commit
# from a release install with no operator-written --engine-profile at all (see
# fastmlx_launch.py's derive_engine_build_from_release), verified against the actual binary bytes
# rather than trusted from the sibling text file alone. Sampled-MTP build attestation is opt-in
# and family-neutral: pass
# --sampled-mtp-minimum-commit <sha> to record whether the source commit descends from that
# minimum, and --require-sampled-mtp (only valid together with the minimum flag) to make the
# script refuse (non-zero exit) to package a build that cannot back that claim. Neither flag has
# a default minimum commit -- no internal deployment's commit is baked into this script.
#
#   scripts/package-release.sh --stage-dir /tmp/fastmlx-stage
#   scripts/package-release.sh --stage-dir /tmp/fastmlx-stage \
#     --sampled-mtp-minimum-commit <sha> --require-sampled-mtp
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPIKE_DIR="$REPO_ROOT/spike"

STAGE_DIR=""
OUT_DIR=""
BINARY=""
CAPACITY_BINARY=""
METALLIB="$REPO_ROOT/spike/prebuilt/mlx.metallib"
VERSION=""
EMIT_FORMULA=""
# Build provenance always comes from this script's own checkout (REPO_ROOT, above) -- see the
# --source-dir handling below for why a mismatched value is refused rather than honored.
SOURCE_DIR="$REPO_ROOT"
SAMPLED_MTP_MIN_COMMIT=""
REQUIRE_SAMPLED_MTP=0

while [ $# -gt 0 ]; do
  case "$1" in
    --stage-dir) STAGE_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --binary) BINARY="$2"; shift 2 ;;
    --capacity-binary) CAPACITY_BINARY="$2"; shift 2 ;;
    --metallib) METALLIB="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    --emit-formula) EMIT_FORMULA="$2"; shift 2 ;;
    --source-dir) SOURCE_DIR="$2"; shift 2 ;;
    --sampled-mtp-minimum-commit) SAMPLED_MTP_MIN_COMMIT="$2"; shift 2 ;;
    --require-sampled-mtp) REQUIRE_SAMPLED_MTP=1; shift ;;
    *) echo "[package-release] unknown flag: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$STAGE_DIR" ]; then
  echo "[package-release] --stage-dir is required" >&2
  exit 2
fi

# --source-dir is accepted for CLI stability, but everything this script produces (the build, the
# README/LICENSE/NOTICE it copies, and the provenance it records) must come from ONE root: this
# script's own checkout. A --source-dir that points somewhere else would let provenance.json name
# a commit other than the one actually built, so any mismatch is refused outright rather than
# silently honored.
SOURCE_DIR_REAL="$(cd "$SOURCE_DIR" 2>/dev/null && pwd -P || true)"
REPO_ROOT_REAL="$(cd "$REPO_ROOT" && pwd -P)"
if [ "$SOURCE_DIR_REAL" != "$REPO_ROOT_REAL" ]; then
  echo "[package-release] ERROR: --source-dir ($SOURCE_DIR) must resolve to this script's own repo root ($REPO_ROOT) -- provenance, the build, and the copied README/LICENSE/NOTICE must all come from the same checkout" >&2
  exit 2
fi

if [ "$REQUIRE_SAMPLED_MTP" -eq 1 ] && [ -z "$SAMPLED_MTP_MIN_COMMIT" ]; then
  echo "[package-release] ERROR: --require-sampled-mtp requires --sampled-mtp-minimum-commit <sha> (no default minimum commit is assumed)" >&2
  exit 2
fi

if [ -z "$VERSION" ]; then
  VERSION="$(cd "$REPO_ROOT" && git describe --always --dirty 2>/dev/null || echo dev)"
fi
if [ -z "$OUT_DIR" ]; then
  # Outside the repo by default so a packaging run never dirties the source tree (a dirty tree
  # would be recorded, truthfully but misleadingly, in the very provenance.json this run emits).
  OUT_DIR="${TMPDIR:-/tmp}/fastmlx-release-${VERSION}"
fi

# --- Build provenance ---------------------------------------------------------------------------
# Feature-in-tree is not feature-in-binary: a checkout containing some capability proves nothing
# about the binary actually built from it. Record what this artifact was actually built from so
# the fact travels with the tarball rather than being lost once it is copied between hosts.
SOURCE_COMMIT_JSON="null"
SOURCE_COMMIT_RAW=""
SOURCE_DIRTY="false"
IS_GIT_CHECKOUT=0
SATISFIES_SAMPLED_MTP_MIN="false"

if git -C "$SOURCE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  IS_GIT_CHECKOUT=1
  SOURCE_COMMIT_RAW="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
  SOURCE_COMMIT_JSON="\"$SOURCE_COMMIT_RAW\""
  if [ -n "$(git -C "$SOURCE_DIR" status --porcelain 2>/dev/null)" ]; then
    SOURCE_DIRTY="true"
  fi
  if [ -n "$SAMPLED_MTP_MIN_COMMIT" ] && git -C "$SOURCE_DIR" merge-base --is-ancestor "$SAMPLED_MTP_MIN_COMMIT" "$SOURCE_COMMIT_RAW" 2>/dev/null; then
    SATISFIES_SAMPLED_MTP_MIN="true"
  fi
fi

if [ "$REQUIRE_SAMPLED_MTP" -eq 1 ]; then
  if [ "$IS_GIT_CHECKOUT" -ne 1 ]; then
    echo "[package-release] ERROR: --require-sampled-mtp requires a git checkout to attest build provenance (source is not a git checkout: $SOURCE_DIR)" >&2
    exit 1
  fi
  if [ "$SOURCE_DIRTY" = "true" ]; then
    echo "[package-release] ERROR: --require-sampled-mtp refuses a dirty working tree — cannot attest a binary built from uncommitted state (source_commit=$SOURCE_COMMIT_RAW)" >&2
    exit 1
  fi
  if [ "$SATISFIES_SAMPLED_MTP_MIN" != "true" ]; then
    echo "[package-release] ERROR: --require-sampled-mtp requires a build at or descending from the given --sampled-mtp-minimum-commit $SAMPLED_MTP_MIN_COMMIT; source_commit=$SOURCE_COMMIT_RAW does not descend from it" >&2
    exit 1
  fi
elif [ -n "$SAMPLED_MTP_MIN_COMMIT" ]; then
  if [ "$IS_GIT_CHECKOUT" -ne 1 ]; then
    echo "[package-release] WARNING: source is not a git checkout ($SOURCE_DIR); build provenance is unknown, satisfies_sampled_mtp_minimum will be recorded false" >&2
  else
    if [ "$SOURCE_DIRTY" = "true" ]; then
      echo "[package-release] WARNING: working tree is dirty at $SOURCE_COMMIT_RAW; provenance.json will record source_dirty=true" >&2
    fi
    if [ "$SATISFIES_SAMPLED_MTP_MIN" != "true" ]; then
      echo "[package-release] WARNING: source_commit=$SOURCE_COMMIT_RAW does not descend from the given --sampled-mtp-minimum-commit $SAMPLED_MTP_MIN_COMMIT; provenance.json will record satisfies_sampled_mtp_minimum=false" >&2
    fi
  fi
fi

BUILT_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Build the two products unless prebuilt binaries were injected (test hook).
if [ -z "$BINARY" ] || [ -z "$CAPACITY_BINARY" ]; then
  echo "[package-release] building fastmlx-serve and fastmlx-capacity (release)…" >&2
  # Build each product in its own invocation. A single `swift build --product A --product B`
  # honors only the LAST --product (observed on Swift 6.3.3: the dual-product form built only
  # fastmlx-capacity and exited 0, leaving fastmlx-serve absent and tripping the not-found guard
  # below). Separate invocations reliably produce both binaries.
  (cd "$SPIKE_DIR" && swift build -c release --product fastmlx-serve) >&2
  (cd "$SPIKE_DIR" && swift build -c release --product fastmlx-capacity) >&2
  BIN_DIR="$SPIKE_DIR/.build/release"
  [ -z "$BINARY" ] && BINARY="$BIN_DIR/fastmlx-serve"
  [ -z "$CAPACITY_BINARY" ] && CAPACITY_BINARY="$BIN_DIR/fastmlx-capacity"
fi

if [ ! -f "$BINARY" ]; then
  echo "[package-release] ERROR: binary not found: $BINARY" >&2
  exit 1
fi
if [ ! -f "$CAPACITY_BINARY" ]; then
  echo "[package-release] ERROR: capacity binary not found: $CAPACITY_BINARY" >&2
  exit 1
fi
if [ ! -f "$METALLIB" ]; then
  echo "[package-release] ERROR: metallib not found: $METALLIB" >&2
  exit 1
fi

TOP_DIR_NAME="fastmlx-${VERSION}-arm64-macos"
STAGE_ROOT="$STAGE_DIR/$TOP_DIR_NAME"

# Idempotent: clean out only the top dir we own inside the stage dir before staging.
rm -rf "$STAGE_ROOT"
mkdir -p "$STAGE_ROOT/bin" "$STAGE_ROOT/libexec/scripts" "$STAGE_ROOT/libexec/site"

cp -f "$BINARY" "$STAGE_ROOT/bin/fastmlx-serve"
cp -f "$CAPACITY_BINARY" "$STAGE_ROOT/bin/fastmlx-capacity"
cp -f "$METALLIB" "$STAGE_ROOT/bin/mlx.metallib"
chmod +x "$STAGE_ROOT/bin/fastmlx-serve" "$STAGE_ROOT/bin/fastmlx-capacity"

# Hashed AFTER staging (the staged copy is what a downloaded tarball actually execs) so
# provenance.json's binary digests describe the exact bytes an operator's tarball ships, not
# whatever transient path $BINARY/$CAPACITY_BINARY pointed at during the build. fastmlx serve's
# engine-build derivation (fastmlx_launch.py's derive_engine_build_from_release) treats
# engine_binary_sha256 as the one fact that makes source_commit trustworthy for the shipped
# fastmlx-serve -- a mismatch there means the binary was swapped after packaging.
ENGINE_BINARY_SHA256="$(shasum -a 256 "$STAGE_ROOT/bin/fastmlx-serve" | awk '{print $1}')"
CAPACITY_BINARY_SHA256="$(shasum -a 256 "$STAGE_ROOT/bin/fastmlx-capacity" | awk '{print $1}')"

# The `fastmlx` tooling: a thin Python dispatcher (fastmlx.py) plus the sibling modules it loads
# by file path (pull/launch/recommend/bench/the front-mode proxy/the HF downloader/the release
# publishability validator/the two fit sizers). All ten must stay siblings in the same
# directory -- fastmlx_launch.py loads fastmlx_pull.py and fastmlx_proxy.py, fastmlx_recommend.py
# loads fastmlx_launch.py, fastmlx_pull.py loads hf_pinned_snapshot_download.py,
# fastmlx_bench.py loads validate_public_repository.py (its sole source for PRIVATE_MARKERS,
# THIRD_PARTY_ENGINE_MARKERS, and OWN_BINARY_NAME -- see fastmlx_bench.py's
# _load_marker_source; a staged tree missing this sibling fail-closed refuses every bench row
# with "refused_sweep_unavailable" rather than reporting a clean sweep), and
# fastmlx_safetensors_fit.py loads fastmlx_gguf_fit.py (via importlib, resolving
# Path(__file__).resolve().parent), each resolving the sibling path relative to its own
# __file__.
for name in fastmlx fastmlx_pull fastmlx_launch fastmlx_proxy fastmlx_recommend fastmlx_bench validate_public_repository hf_pinned_snapshot_download fastmlx_gguf_fit fastmlx_safetensors_fit; do
  cp -f "$REPO_ROOT/scripts/${name}.py" "$STAGE_ROOT/libexec/scripts/${name}.py"
done
# The two fit sizers are exec'd directly (fastmlx_launch.py's --fit-check-bin), unlike the other
# five tooling modules above which are only ever invoked via `python3 <path>` -- so, unlike them,
# these two need their executable bit set explicitly rather than relying on cp to preserve it.
chmod 0755 "$STAGE_ROOT/libexec/scripts/fastmlx_gguf_fit.py" "$STAGE_ROOT/libexec/scripts/fastmlx_safetensors_fit.py"
# fastmlx_launch.py's REPO_ROOT resolves one parent up from its own scripts/ dir, so its default
# quality-cards path lands on <that parent>/site/quality-guides.json -- libexec/site here, mirroring
# this repository's own scripts/ + site/ layout.
cp -f "$REPO_ROOT/site/quality-guides.json" "$STAGE_ROOT/libexec/site/quality-guides.json"

# Public-safe example engine profiles (--engine-profile documents), for an operator fronting
# another OpenAI-compatible engine. Fails loudly rather than silently shipping an empty
# engine-profiles dir if the repo's examples/engine-profiles has nothing to stage.
mkdir -p "$STAGE_ROOT/share/fastmlx/engine-profiles"
shopt -s nullglob
PROFILE_EXAMPLES=("$REPO_ROOT"/examples/engine-profiles/*.json)
shopt -u nullglob
if [ "${#PROFILE_EXAMPLES[@]}" -eq 0 ]; then
  echo "[package-release] ERROR: no engine profile examples found under $REPO_ROOT/examples/engine-profiles" >&2
  exit 1
fi
for profile in "${PROFILE_EXAMPLES[@]}"; do
  cp -f "$profile" "$STAGE_ROOT/share/fastmlx/engine-profiles/$(basename "$profile")"
  chmod 0644 "$STAGE_ROOT/share/fastmlx/engine-profiles/$(basename "$profile")"
done

cp -f "$REPO_ROOT/LICENSE" "$STAGE_ROOT/LICENSE"
cp -f "$REPO_ROOT/NOTICE" "$STAGE_ROOT/NOTICE"
cp -f "$REPO_ROOT/README.md" "$STAGE_ROOT/README.md"

PROVENANCE_JSON="{
  \"source_commit\": $SOURCE_COMMIT_JSON,
  \"source_dirty\": $SOURCE_DIRTY,
  \"version\": \"$VERSION\",
  \"built_at\": \"$BUILT_AT\",
  \"engine_binary_sha256\": \"$ENGINE_BINARY_SHA256\",
  \"capacity_binary_sha256\": \"$CAPACITY_BINARY_SHA256\""
if [ -n "$SAMPLED_MTP_MIN_COMMIT" ]; then
  PROVENANCE_JSON="$PROVENANCE_JSON,
  \"sampled_mtp_minimum_commit\": \"$SAMPLED_MTP_MIN_COMMIT\",
  \"satisfies_sampled_mtp_minimum\": $SATISFIES_SAMPLED_MTP_MIN"
fi
PROVENANCE_JSON="$PROVENANCE_JSON
}"
printf '%s\n' "$PROVENANCE_JSON" > "$STAGE_ROOT/provenance.json"

cat > "$STAGE_ROOT/bin/fastmlx" <<'LAUNCHER'
#!/bin/sh
# Slim POSIX-sh shim: resolves its own real directory (through symlinks), puts it on PATH (fastmlx_launch.py's and
# fastmlx_recommend.py's engine-binary resolution is PATH-based, falling back to
# `shutil.which("fastmlx-serve")` when no --fit-check-bin/--engine-binary is given), then execs
# the real dispatcher out of libexec/scripts. mlx.metallib is already colocated in bin/ (MLX
# searches <bindir>/mlx.metallib first), so no build step or copy is needed here.
set -eu
# Follow symlinks (a user may link bin/fastmlx into a directory already on PATH) so libexec and
# the sibling binaries resolve from the real install, not from the link's directory.
SELF="$0"
while [ -L "$SELF" ]; do
  LINK="$(readlink "$SELF")"
  case "$LINK" in
    /*) SELF="$LINK" ;;
    *) SELF="$(dirname "$SELF")/$LINK" ;;
  esac
done
BIN_DIR="$(cd "$(dirname "$SELF")" && pwd -P)"
ROOT_DIR="$(cd "$BIN_DIR/.." && pwd)"
export PATH="$BIN_DIR:$PATH"
exec /usr/bin/env python3 "$ROOT_DIR/libexec/scripts/fastmlx.py" "$@"
LAUNCHER
chmod +x "$STAGE_ROOT/bin/fastmlx"

mkdir -p "$OUT_DIR"
TARBALL="$OUT_DIR/${TOP_DIR_NAME}.tar.gz"
SHA_FILE="$TARBALL.sha256"
rm -f "$TARBALL" "$SHA_FILE"

COPYFILE_DISABLE=1 tar -czf "$TARBALL" -C "$STAGE_DIR" "$TOP_DIR_NAME"

SHA_HEX="$(shasum -a 256 "$TARBALL" | awk '{print $1}')"
printf '%s  %s\n' "$SHA_HEX" "$(basename "$TARBALL")" > "$SHA_FILE"

# --- Generate the source-build Homebrew formula (opt-in: only when --emit-formula is given, so a
# packaging run never touches this repo's own Formula/fastmlx.rb by accident) -----------------
if [ -n "$EMIT_FORMULA" ]; then
  mkdir -p "$(dirname "$EMIT_FORMULA")"

  # Generated from the SAME $PROFILE_EXAMPLES array the tarball's own staging loop above uses
  # (not a separately hand-written list), so the Formula's pkgshare installs can never drift from
  # the set of example files actually staged into share/fastmlx/engine-profiles.
  FORMULA_ENGINE_PROFILE_INSTALL_LINES=""
  for profile in "${PROFILE_EXAMPLES[@]}"; do
    profile_name="$(basename "$profile")"
    profile_install_line="    (pkgshare/\"engine-profiles\").install \"examples/engine-profiles/$profile_name\""
    if [ -z "$FORMULA_ENGINE_PROFILE_INSTALL_LINES" ]; then
      FORMULA_ENGINE_PROFILE_INSTALL_LINES="$profile_install_line"
    else
      FORMULA_ENGINE_PROFILE_INSTALL_LINES="$FORMULA_ENGINE_PROFILE_INSTALL_LINES
$profile_install_line"
    fi
  done

  cat > "$EMIT_FORMULA" <<FORMULA
# GENERATED by scripts/package-release.sh — do not hand-edit.
#
# This is a source-build Homebrew formula: it runs \`swift build\` from source rather than
# shipping an unsigned prebuilt binary through brew, which sidesteps Gatekeeper/notarization.
# The in-repo prebuilt metallib (spike/prebuilt/mlx.metallib) is what makes a source build work
# with no Metal Toolchain install on the installing machine.
#
# The \`url\` and \`sha256\` below are PLACEHOLDER_SET_AT_PUBLISH — the release/source archive
# address and its digest are unknowable until the operator publishes a tagged release. Fill
# them in at publish time. Publication (tap repo, GitHub release, tap update) is human-gated
# and is not performed by this script.
#
# The product this formula installs is the single \`fastmlx\` command (pull/serve/recommend
# front door), not the Swift engine on its own -- the engine (fastmlx-serve), the capacity
# binary (fastmlx-capacity), and the metallib are components \`bin/fastmlx\` and its
# \`libexec/scripts\` tooling depend on, installed alongside it.
class Fastmlx < Formula
  desc "Fit-checked, quality-carded LLM serving for Apple Silicon"
  homepage "https://github.com/bitworks-io/fast-mlx"
  url "PLACEHOLDER_SET_AT_PUBLISH"
  sha256 "PLACEHOLDER_SET_AT_PUBLISH"
  version "${VERSION}"
  license "Apache-2.0"

  # Swift 6 toolchain required; Xcode 16+ provides it on macOS.
  depends_on xcode: ["16.0", :build]
  # The fastmlx.py tooling (pull/serve/recommend) is stdlib-only Python with no pip imports,
  # so it only needs an interpreter, never a managed Python keg. uses_from_macos is Homebrew's
  # own mechanism for exactly that case (a lighter runtime footprint than pulling in a whole
  # extra keg just to run a wrapper script), and any machine able to satisfy this formula's
  # own Xcode 16+ build requirement already has a working system Python 3.
  uses_from_macos "python"

  def install
    cd "spike" do
      system "swift build --disable-sandbox -c release --product fastmlx-serve"
      system "swift build --disable-sandbox -c release --product fastmlx-capacity"
      bin.install ".build/release/fastmlx-serve"
      bin.install ".build/release/fastmlx-capacity"
      # Colocate the prebuilt metallib next to the binaries (MLX searches <bindir>/mlx.metallib
      # first), avoiding any Metal Toolchain requirement on the installing machine.
      bin.install "prebuilt/mlx.metallib"
    end

    # Install the engine-agnostic tooling into libexec, mirroring this repository's own
    # scripts/ + site/ layout: fastmlx_launch.py's REPO_ROOT resolves one parent up from
    # libexec/scripts, so its default quality-guidance cards path lands on
    # libexec/site/quality-guides.json, exactly like site/quality-guides.json does in
    # a development checkout.
    (libexec/"scripts").install "scripts/fastmlx.py"
    (libexec/"scripts").install "scripts/fastmlx_pull.py"
    (libexec/"scripts").install "scripts/fastmlx_launch.py"
    (libexec/"scripts").install "scripts/fastmlx_proxy.py"
    (libexec/"scripts").install "scripts/fastmlx_recommend.py"
    (libexec/"scripts").install "scripts/fastmlx_bench.py"
    (libexec/"scripts").install "scripts/validate_public_repository.py"
    (libexec/"scripts").install "scripts/hf_pinned_snapshot_download.py"
    (libexec/"scripts").install "scripts/fastmlx_gguf_fit.py"
    (libexec/"scripts").install "scripts/fastmlx_safetensors_fit.py"
    (libexec/"site").install "site/quality-guides.json"

    # Public-safe example engine profiles (--engine-profile documents), installed under
    # share/fastmlx/engine-profiles -- the Homebrew-idiomatic \`pkgshare\` location (share/<name>).
$FORMULA_ENGINE_PROFILE_INSTALL_LINES

    (bin/"fastmlx").write <<~EOS
      #!/usr/bin/env bash
      set -euo pipefail
      PREFIX_BIN_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
      # Make this same directory (fastmlx-serve, fastmlx-capacity) discoverable on PATH: the
      # tooling's own binary resolution (fastmlx.py's capacity/engine subcommands, and
      # fastmlx_launch.py's fit-check/engine binary defaults) falls back to a PATH lookup for
      # "fastmlx-serve"/"fastmlx-capacity" when it is not found next to the dispatcher's own
      # install.
      export PATH="\$PREFIX_BIN_DIR:\$PATH"
      exec python3 "#{libexec}/scripts/fastmlx.py" "\$@"
    EOS
    chmod "+x", bin/"fastmlx"
  end

  test do
    system "#{bin}/fastmlx", "--help"
    system "#{bin}/fastmlx-capacity", "--help"
  end
end
FORMULA
fi

echo "packaged: $TARBALL sha256=$SHA_HEX"
