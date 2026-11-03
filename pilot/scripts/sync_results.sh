#!/usr/bin/env bash
# Copies live runner output (data/pilot/raw, untracked) into pilot/results/raw
# and commits pilot/results. Usage: pilot/scripts/sync_results.sh "message"
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p pilot/results/raw
cp -a data/pilot/raw/. pilot/results/raw/
git add pilot/results
if ! git diff --cached --quiet; then
  git commit -q -m "Pilot results: ${1:-checkpoint}

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_018ARa1fsyMMbRFTkCJwUdTh"
  git push -q origin "$(git rev-parse --abbrev-ref HEAD)"
fi
