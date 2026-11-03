#!/usr/bin/env bash
# Proposed model: (1) discrete-radius initialization, (2) continuous radius + CRC.
# To skip step (1), use the released initialization:
#   python train.py --config configs/cradius_rse/continuous_crc.json --run-dir runs/continuous_crc \
#       --init-checkpoint checkpoints/discrete_init.pt
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

python train.py --config configs/cradius_rse/discrete_init.json --run-dir runs/discrete_init
python train.py --config configs/cradius_rse/continuous_crc.json --run-dir runs/continuous_crc
