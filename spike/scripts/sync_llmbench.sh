#!/bin/bash
# Deploy the spike to the bench host, carrying git provenance with it.
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

SHA="$(git -C "$SPIKE_DIR" rev-parse HEAD)"
if ! git -C "$SPIKE_DIR" diff --quiet HEAD -- .; then
    SHA="${SHA}-dirty"
fi
echo "$SHA" > "$SPIKE_DIR/.harness-sha"

rsync -a --delete \
    --exclude '.build' --exclude '.swiftpm' --exclude 'default.profraw' \
    --exclude 'harness-evidence.jsonl' \
    "$SPIKE_DIR/" "$HOST:~/$REMOTE_DIR/"

echo "synced $SPIKE_DIR -> $HOST:~/$REMOTE_DIR (harness SHA $SHA)"
