#!/usr/bin/env bash
# Regenerate spike/prebuilt/mlx.metallib from the pinned mlx-swift.
#
# MAINTAINER script — run this only when the mlx-swift version pin in spike/Package.swift changes.
# End users never run this; they use the shipped prebuilt via scripts/serve.sh.
#
# Background: `swift build` cannot compile MLX's Metal kernels (no SwiftPM Metal phase). Xcode can,
# but on macOS 26 the Metal compiler is a separate on-demand component. This script installs that
# component if needed, builds the kernels into default.metallib via Xcode, and stages it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPIKE_DIR="$(cd "$SCRIPT_DIR/../spike" && pwd)"
cd "$SPIKE_DIR"

if ! xcrun --find metal >/dev/null 2>&1; then
  echo "[fast-mlx] Metal compiler not found — installing the Metal Toolchain component (one-time, ~690MB, no sudo)…" >&2
  xcodebuild -downloadComponent MetalToolchain
fi

echo "[fast-mlx] building the Metal kernels via Xcode…" >&2
xcodebuild -scheme fastmlx-serve -configuration Release \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath .build/xcode \
  -skipPackagePluginValidation -skipMacroValidation \
  build >&2

SRC=".build/xcode/Build/Products/Release/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
if [ ! -f "$SRC" ]; then
  echo "[fast-mlx] ERROR: expected metallib not produced at $SRC" >&2
  exit 1
fi
mkdir -p prebuilt
cp -f "$SRC" prebuilt/mlx.metallib
echo "[fast-mlx] refreshed prebuilt/mlx.metallib ($(du -h prebuilt/mlx.metallib | cut -f1)); commit it alongside the mlx-swift pin change." >&2
