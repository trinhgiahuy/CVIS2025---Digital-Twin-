#!/usr/bin/env bash
set -euo pipefail
MANIFEST=${1:?Usage: bash scripts/run_all.sh manifest.csv}
for TASK in binary counting; do
  if [ "$TASK" = binary ]; then SEED=40; else SEED=42; fi
  for METHOD in naive random_dr cdr; do
    python -m cvis_cdr.experiment train --manifest "$MANIFEST" --task "$TASK" --method "$METHOD" --seed "$SEED" --epochs 30 --input-size 224
  done
done
