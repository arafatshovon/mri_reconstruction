#!/usr/bin/env bash
set -euo pipefail

python -m src.train \
  --dataset-config configs/dataset.yaml \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml
