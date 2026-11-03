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
RESULTS_BRANCH=pilot-c-results
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
if git ls-remote --exit-code --heads origin "$RESULTS_BRANCH" > /dev/null; then
  git fetch -q origin "$RESULTS_BRANCH"
  git checkout -q -B "$RESULTS_BRANCH" "origin/$RESULTS_BRANCH"
else
  git checkout -q -B "$RESULTS_BRANCH"
fi
mkdir -p data/pilot/raw pilot/results/raw
cp -a pilot/results/raw/. data/pilot/raw/   # Resume from earlier sessions.

push_results() {
  cp -a data/pilot/raw/. pilot/results/raw/
  cp pilot/logs/kaggle_*.log pilot/results/ 2>/dev/null || true
  git add -f pilot/results
  git commit -q -m "Pilot arm C results from Kaggle ($(date -u +%FT%TZ))" || return 0
  git push -q origin "HEAD:$RESULTS_BRANCH" || echo "push failed; will retry"
}

C2=(pilot/configs/C2_thinking.yaml)
C13=(pilot/configs/C1_*.yaml pilot/configs/C3A_*.yaml pilot/configs/C3R.yaml)
run() {  # device batch_size configs...
  python pilot/scripts/run_pilot.py --device "$1" --batch-size "$2" "${@:3}"
}

echo "== running"
N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
# Thinking runs: 3 prompts x 3 samples per batch keeps the worst-case fp32 KV
# cache (~4,700 tokens x 230 KB per row) within a 16 GB GPU.
if [ "$N_GPUS" -ge 2 ]; then
  run cuda:0 9 "${C2[@]}" > pilot/logs/kaggle_c2.log 2>&1 &
  P1=$!
  run cuda:1 32 "${C13[@]}" > pilot/logs/kaggle_c13.log 2>&1 &
  P2=$!
else
  (run cuda:0 9 "${C2[@]}" && run cuda:0 32 "${C13[@]}") > pilot/logs/kaggle_all.log 2>&1 &
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
