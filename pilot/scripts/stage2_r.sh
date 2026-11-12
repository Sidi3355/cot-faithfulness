#!/usr/bin/env bash
# Arm R main run at the chosen N for the three lengths, no instruction first.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
for length in empty very_concise very_comprehensive; do
  for model in Qwen2.5-0.5B-Instruct Qwen2.5-1.5B-Instruct Qwen2.5-3B-Instruct; do
    $PY pilot/scripts/run_pilot.py "pilot/configs/R_${model}_${length}.yaml"
    pilot/scripts/sync_results.sh "arm R ${model} ${length} (N = 150)" || true
  done
done
echo "== stage 2 R done $(date -Is)"
