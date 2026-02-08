#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_PATH="${1:-}"
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "Usage: scripts/validate.sh /path/to/checkpoint.ckpt"
  exit 1
fi

python -m src.validate \
  --dataset-config configs/dataset.yaml \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --checkpoint "${CHECKPOINT_PATH}"
