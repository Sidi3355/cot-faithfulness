# Running the chain-of-thought arm on Kaggle

Runs C1, C2 and C3 (all on Qwen3-0.6B, 40 examples) on a free Kaggle GPU with the same code and
configs as the CPU runs. Results are pushed to the branch `pilot-c-results` every 10 minutes.
Expected time: a few hours on 2× T4.

## One-time setup

1. **GitHub token** (the repo is private). Go to GitHub → Settings → Developer settings →
   Fine-grained tokens → Generate new token.
   - Repository access: only `Sidi3355/cot-faithfulness`.
   - Permissions: **Contents: Read and write**.
2. **Kaggle notebook.** Create a new notebook, then in the right-hand panel:
   - Session options → Accelerator: **GPU T4 x2**.
   - Session options → Internet: **On** (Kaggle asks for phone verification the first time).
   - Add-ons → Secrets → add `GITHUB_TOKEN` with the token from step 1, and tick it for this
     notebook.

## Cells to run

Cell 1 (Python):

```python
from kaggle_secrets import UserSecretsClient
import os
os.environ["GITHUB_TOKEN"] = UserSecretsClient().get_secret("GITHUB_TOKEN")
```

Cell 2:

```bash
%%bash
cd /kaggle/working
git clone -q -b claude/happy-lamport-gzsfmv "https://x-access-token:${GITHUB_TOKEN}@github.com/Sidi3355/cot-faithfulness.git"
cd cot-faithfulness
bash pilot/kaggle/run_c_arm.sh
```

Cell 2 prints progress every 10 minutes. If the session ends before `== finished`, start a new
session and run both cells again: the script resumes from the results already pushed.

## Notes

- Uses Kaggle's preinstalled PyTorch (with CUDA) and pins `transformers` and `tokenizers` to the
  versions in `uv.lock`. The versions actually used are recorded in each run's `meta.json`.
- Runs in float32 like the CPU pilot runs; T4 GPUs have no bf16.
