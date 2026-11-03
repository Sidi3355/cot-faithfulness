#!/usr/bin/env bash
# Stage 1 of the pilot: arm R pre-pilot and its sample-size table.
# (The interventions were frozen beforehand by pilot/scripts/judge.py.)
# Usage: setsid nohup pilot/scripts/stage1.sh > pilot/logs/stage1.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
$PY pilot/scripts/make_configs.py --r-examples 20
$PY pilot/scripts/run_pilot.py pilot/configs/R_Qwen2.5-0.5B-Instruct_empty.yaml \
  pilot/configs/R_Qwen2.5-1.5B-Instruct_empty.yaml pilot/configs/R_Qwen2.5-3B-Instruct_empty.yaml
$PY pilot/scripts/analyze.py prepilot
echo "== stage 1 done $(date -Is)"
