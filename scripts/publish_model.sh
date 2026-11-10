#!/usr/bin/env bash
# Upload the shipping checkpoint to the Hugging Face Hub.
# Prerequisites: `huggingface-cli login` (or HF_TOKEN), `uv pip install huggingface_hub`.
set -euo pipefail
REPO="${1:?Usage: publish_model.sh <org/repo> [checkpoint-dir]}"
CHECKPOINT="${2:-$(cd "$(dirname "$0")/.." && pwd)/.models/owl-int4}"
if [ ! -f "$CHECKPOINT/model.safetensors" ]; then
  echo "no checkpoint at $CHECKPOINT" >&2
  exit 1
fi
CARD="$(cd "$(dirname "$0")/.." && pwd)/MODEL_CARD.md"
python3 - "$REPO" "$CHECKPOINT" "$CARD" <<'PY'
import sys
from huggingface_hub import HfApi
repo, folder, card = sys.argv[1], sys.argv[2], sys.argv[3]
api = HfApi()
api.create_repo(repo, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=folder, repo_id=repo, repo_type="model")
api.upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=repo, repo_type="model")
print(f"uploaded to https://huggingface.co/{repo}")
PY
