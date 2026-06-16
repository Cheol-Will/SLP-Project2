#!/usr/bin/env bash
TOP_K=4
PROMPT_ID=30
MAX_TOKEN=200

EXP_NAME="results/test-260616-topk${TOP_K}-p_id${PROMPT_ID}-n_token$MAX_TOKEN"
PROMPT_PATH="configs/prompt$PROMPT_ID.json"

python main.py \
  --split test \
  --exp-name "${EXP_NAME}" \
  --prompt-path "${PROMPT_PATH}" \
  --top-k "${TOP_K}" \
  --max_token "${MAX_TOKEN}"