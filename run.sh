#!/bin/bash
# Mapping model training (Vibravox, K-Fold CV)
# Run from mfa project root: bash enhancement/run.sh

CUDA_VISIBLE_DEVICES=0 python -m enhancement.src.train \
  +model=dpcrn \
  cv.enabled=true cv.fold_index=1
