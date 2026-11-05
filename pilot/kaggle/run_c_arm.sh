#!/usr/bin/env bash
# Runs the pilot's chain-of-thought arm (C1, C2, C3) on a Kaggle GPU with the
# same code and configs as the CPU runs, then pushes the raw rows to the branch
# `pilot-c-results` every 10 minutes and at the end. Re-running it in a new
# session resumes from what that branch already holds.
#
# Run from the repo root, with GITHUB_TOKEN set (see pilot/kaggle/README.md).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src"
# Lets the allocator reuse freed GPU memory across batches of different shapes.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RESULTS_BRANCH=pilot-c-results
# The results branch is checked out in its own worktree, so the code that runs
# is always this checkout's.
RES="$PWD/../pilot-results"
mkdir -p pilot/logs

echo "== installing (Kaggle's torch; tokenizer/generation libraries pinned as in uv.lock)"
pip install -q "transformers==5.19.0" "tokenizers==$(grep -A1 '^name = "tokenizers"' uv.lock | sed -n 's/version = "\(.*\)"/\1/p')" \
  accelerate "pandas<3" pyarrow nltk "pydantic>=2.9" pyyaml huggingface_hub
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| GPUs', torch.cuda.device_count())"

echo "== data (ComVE only) and Qwen3-0.6B weights"
DATASETS=comve PYTHON=python scripts/fetch_data.sh
python - <<'EOF'
from huggingface_hub import snapshot_download
from cotfaith import models
m = models.get("Qwen/Qwen3-0.6B")
snapshot_download(m.name, revision=m.revision, cache_dir="data/hf",
                  allow_patterns=["*.safetensors", "*.json", "merges.txt"])
EOF

echo "== results branch"
git config user.name "pilot-runner"
git config user.email "pilot-runner@users.noreply.github.com"
git worktree remove --force "$RES" 2> /dev/null || true
git worktree prune
if git ls-remote --exit-code --heads origin "$RESULTS_BRANCH" > /dev/null; then
  git fetch -q origin "$RESULTS_BRANCH"
  git worktree add -q -B "$RESULTS_BRANCH" "$RES" "origin/$RESULTS_BRANCH"
else
  git worktree add -q -B "$RESULTS_BRANCH" "$RES" HEAD
fi
mkdir -p data/pilot/raw "$RES/pilot/results/raw"
cp -a "$RES/pilot/results/raw/." data/pilot/raw/          # Resume from earlier sessions.
cp "$RES"/pilot/results/kaggle_*.log pilot/logs/ 2> /dev/null || true  # Logs are appended to.

push_results() {
  cp -a data/pilot/raw/. "$RES/pilot/results/raw/"
  cp pilot/logs/kaggle_*.log "$RES/pilot/results/" 2> /dev/null || true
  git -C "$RES" add -f pilot/results
  git -C "$RES" commit -q -m "Pilot arm C results from Kaggle ($(date -u +%FT%TZ))" > /dev/null || true
  git -C "$RES" push -q origin "HEAD:$RESULTS_BRANCH" || echo "push failed; will retry"
}

# Two queues of about equal length once C2 (~9 h) is done; each config runs in
# its own process, so GPU memory is freed between configs, and a failed config
# does not stop the rest of its queue.
Q0=(pilot/configs/C2_thinking.yaml pilot/configs/C3A_very_comprehensive.yaml
    pilot/configs/C3A_comprehensive.yaml pilot/configs/C3R.yaml)
Q1=(pilot/configs/C1_*.yaml pilot/configs/C3A_empty.yaml pilot/configs/C3A_concise.yaml
    pilot/configs/C3A_very_concise.yaml)
run_each() {  # device configs...
  local status=0 c bs
  for c in "${@:2}"; do
    # Thinking runs: 3 prompts x 3 samples per batch keeps the worst-case fp32 KV
    # cache (~4,700 tokens x 230 KB per row) within a 16 GB GPU.
    if [[ "$c" == *C2_* ]]; then bs=9; else bs=32; fi
    python pilot/scripts/run_pilot.py --device "$1" --batch-size "$bs" "$c" || status=1
  done
  return $status
}

echo "== running"
N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "== session $(date -u +%FT%TZ)" | tee -a pilot/logs/kaggle_c2.log pilot/logs/kaggle_c13.log > /dev/null
if [ "$N_GPUS" -ge 2 ]; then
  run_each cuda:0 "${Q0[@]}" >> pilot/logs/kaggle_c2.log 2>&1 &
  P1=$!
  run_each cuda:1 "${Q1[@]}" >> pilot/logs/kaggle_c13.log 2>&1 &
  P2=$!
else
  run_each cuda:0 "${Q0[@]}" "${Q1[@]}" >> pilot/logs/kaggle_c13.log 2>&1 &
  P1=$!
  P2=$P1
fi
while kill -0 "$P1" 2> /dev/null || kill -0 "$P2" 2> /dev/null; do
  sleep 600
  push_results
  grep -h -E "^== " pilot/logs/kaggle_*.log | tail -3
done
S1=0; wait "$P1" || S1=$?
S2=0; if [ "$P2" != "$P1" ]; then wait "$P2" || S2=$?; fi
push_results
echo "== finished (exit codes $S1 $S2; non-zero means check pilot/logs/kaggle_*.log)"
