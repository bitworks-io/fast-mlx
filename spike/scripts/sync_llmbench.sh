#!/bin/bash
# Deploy the spike and reproducible experiments to the bench host, carrying git provenance.
#
# An rsync'd tree has no .git, so the harness binary cannot resolve its own SHA there —
# every evidence record used to say harnessGitSHA="unknown". This script captures
# `git rev-parse HEAD` (with a "-dirty" suffix when the working tree differs from HEAD,
# so a hand-edited deploy can never masquerade as a committed one) into .harness-sha,
# which ProvenanceCLI reads as its second source (after the HARNESS_GIT_SHA env override).
#
# Usage: spike/scripts/sync_llmbench.sh [user@host] [remote-dir]
set -euo pipefail

HOST="${1:-llmbench@192.168.1.252}"
REMOTE_DIR="${2:-fast-mlx-spike}"
SPIKE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(git -C "$SPIKE_DIR" rev-parse --show-toplevel)"

SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
if ! git -C "$REPO_ROOT" diff --quiet HEAD -- spike experiments \
    || [[ -n "$(git -C "$REPO_ROOT" ls-files --others --exclude-standard -- spike experiments)" ]]; then
    SHA="${SHA}-dirty"
fi
echo "$SHA" > "$SPIKE_DIR/.harness-sha"

STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/fast-mlx-sync.XXXXXX")"
trap 'rm -rf "$STAGING_DIR"' EXIT
mkdir -p "$STAGING_DIR/source" "$STAGING_DIR/deploy"

# Stage only tracked files plus non-ignored development files. This keeps global or project
# gitignored machine artifacts from crossing the provenance boundary under a clean SHA.
git -C "$REPO_ROOT" ls-files -z --cached --others --exclude-standard -- spike experiments \
    | rsync -a --from0 --files-from=- "$REPO_ROOT/" "$STAGING_DIR/source/"
rsync -a "$STAGING_DIR/source/spike/" "$STAGING_DIR/deploy/"
if [[ -d "$STAGING_DIR/source/experiments" ]]; then
    mkdir -p "$STAGING_DIR/deploy/experiments"
    rsync -a "$STAGING_DIR/source/experiments/" "$STAGING_DIR/deploy/experiments/"
fi
cp "$SPIKE_DIR/.harness-sha" "$STAGING_DIR/deploy/.harness-sha"

# Preserve bench-only build caches and generated evidence; delete every other file that is not
# in the staged source manifest so an ignored local artifact cannot influence execution.
rsync -a --delete \
    --filter='protect .build/' \
    --filter='protect .swiftpm/' \
    --filter='protect Package.resolved' \
    --filter='protect default.profraw' \
    --filter='protect harness-evidence.jsonl' \
    --filter='protect experiments/eagle3/artifacts/' \
    "$STAGING_DIR/deploy/" "$HOST:~/$REMOTE_DIR/"

echo "synced $SPIKE_DIR + experiments -> $HOST:~/$REMOTE_DIR (harness SHA $SHA)"
