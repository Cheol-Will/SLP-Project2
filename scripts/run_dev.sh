#!/usr/bin/env bash
# Usage:
#   bash scripts/run_dev.sh -p configs/prompt1.json
#   bash scripts/run_dev.sh --prompt-path configs/prompt2.json
set -euo pipefail

TOP_K_LIST=(7 6 5 4 3)
PROMPT_IDS=(2 15 25 30 31 32 33 34 35)
MAX_TOKENS=(100 200)

for TOP_K in "${TOP_K_LIST[@]}"; do
  for PROMPT_ID in "${PROMPT_IDS[@]}"; do
    for MAX_TOKEN in "${MAX_TOKENS[@]}"; do
      PROMPT_PATH="configs/prompt$PROMPT_ID.json"

      EXP_NAME="dev-260608-topk${TOP_K}-p_id${PROMPT_ID}-n_token$MAX_TOKEN"
      echo "=== top-k=${TOP_K}  exp=${EXP_NAME} ==="
      uv run python main.py \
        --split dev \
        --top-k "${TOP_K}" \
        --exp-name "${EXP_NAME}" \
        --prompt-path "${PROMPT_PATH}" \
        --max_token "${MAX_TOKEN}"
    done
  done
done
