#!/usr/bin/env bash
# Table 2 variants. All start from runs/discrete_init (see train_cradius_rse.sh);
# the continuous + CRC variant is the proposed model (runs/continuous_crc).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

for variant in discrete_crc continuous_wo_crc; do
    python train.py --config "configs/ablation/${variant}.json" --run-dir "runs/${variant}"
done
