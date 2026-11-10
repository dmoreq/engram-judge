#!/usr/bin/env bash
# Upload the shipping checkpoint to the Hugging Face Hub.
#
# Fails fast on three conditions before transferring 85 MB:
#   1. no valid token (run `hf auth login --force` or export HF_TOKEN),
#   2. a target namespace the token cannot write to,
#   3. a missing local checkpoint.
set -euo pipefail
REPO="${1:?Usage: publish_model.sh <namespace/name> [checkpoint-dir]}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT="${2:-$ROOT/.models/owl-int4}"
CARD="$ROOT/MODEL_CARD.md"

if [ ! -f "$CHECKPOINT/model.safetensors" ]; then
  echo "no checkpoint at $CHECKPOINT" >&2
  exit 1
fi
if [ ! -f "$CARD" ]; then
  echo "no MODEL_CARD.md at $CARD" >&2
  exit 1
fi

python3 - "$REPO" "$CHECKPOINT" "$CARD" <<'PY'
import sys
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

repo, folder, card = sys.argv[1], sys.argv[2], sys.argv[3]
namespace = repo.split("/")[0]
api = HfApi()

try:
    who = api.whoami()
except HfHubHTTPError as exc:
    sys.exit(f"no valid token ({exc}); run `hf auth login --force` or export HF_TOKEN")

allowed = {who.get("name")} | {org.get("name") for org in who.get("orgs", [])}
print(f"authenticated as {who.get('name')!r}; writable namespaces: {sorted(n for n in allowed if n)}")
if namespace not in allowed:
    sys.exit(
        f"namespace {namespace!r} is not writable by this token; "
        f"use one of {sorted(n for n in allowed if n)}"
    )

api.create_repo(repo, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=folder, repo_id=repo, repo_type="model")
api.upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=repo, repo_type="model")
print(f"uploaded to https://huggingface.co/{repo}")
PY
