#!/usr/bin/env bash
# Downloads the three datasets into data/ at pinned commits, in the layout the
# upstream corr_faith loaders expect (see third_party/corr_faith/Dockerfile).
# Also downloads the WordNet corpus used for intervention words.
#
# Usage: scripts/fetch_data.sh [DATA_DIR]   (default: data)
set -euo pipefail

DATA_DIR="$(realpath -m "${1:-data}")"
# Absolute path (without resolving the venv symlink), since we cd below.
PYTHON="$(realpath -s "$(command -v "${PYTHON:-python}")")"
mkdir -p "$DATA_DIR"

clone_pinned() {  # url dir commit
  if [[ ! -d "$DATA_DIR/$2/.git" ]]; then
    git clone -q "$1" "$DATA_DIR/$2"
  fi
  git -C "$DATA_DIR/$2" checkout -q "$3"
}

clone_pinned https://github.com/OanaMariaCamburu/e-SNLI.git \
  e-SNLI 7b585a3f077fdea899780eb0473940522ae44a2e
clone_pinned https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation.git \
  SemEval2020-Task4-Commonsense-Validation-and-Explanation c55b1a9e6cb66997675f85073e17a9522cf38c2c
clone_pinned https://github.com/dair-iitd/ECQA-Dataset.git \
  ECQA-Dataset 2ba8f0d3cf64ba5beb4605808ee1285c895af50d

# ECQA ships explanations only; join them with CommonsenseQA using the
# dataset's own script, as the upstream Dockerfile does.
ECQA="$DATA_DIR/ECQA-Dataset"
mkdir -p "$ECQA/cqa"
for split in train_rand_split dev_rand_split; do
  if [[ ! -f "$ECQA/cqa/$split.jsonl" ]]; then
    curl -sSfL -o "$ECQA/cqa/$split.jsonl" "https://s3.amazonaws.com/commensenseqa/$split.jsonl"
  fi
done
if [[ ! -f "$ECQA/cqa_data_test.csv" ]]; then
  # -I: don't put the dataset directory on sys.path.
  (cd "$ECQA" && "$PYTHON" -I generate_data.py > /dev/null)
fi

# WordNet 3.0, fetched directly at a pinned nltk_data commit and checksummed
# (NLTK's own downloader refuses to run behind an egress proxy).
WN_DIR="$DATA_DIR/nltk_data/corpora"
WN_ZIP="$WN_DIR/wordnet.zip"
WN_COMMIT=550b6625bcef1f2abff2ff770a5a0d272c9c6b2a
WN_SHA256=cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59
mkdir -p "$WN_DIR"
if [[ ! -f "$WN_ZIP" ]]; then
  curl -sSfL -o "$WN_ZIP" \
    "https://raw.githubusercontent.com/nltk/nltk_data/$WN_COMMIT/packages/corpora/wordnet.zip"
fi
echo "$WN_SHA256  $WN_ZIP" | sha256sum -c --quiet
unzip -q -o "$WN_ZIP" -d "$WN_DIR"
# NLTK ignores corpus files that other accounts can write to.
chmod -R go-w "$DATA_DIR/nltk_data"

echo "Datasets ready in $DATA_DIR"
