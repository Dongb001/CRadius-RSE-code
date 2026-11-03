#!/usr/bin/env bash
# Evaluate the released checkpoints on the shared scenes (Tables 1-2, Fig. 4).
# Extra arguments are passed to every evaluate.py call, e.g. measured RIRs:
#   bash scripts/evaluate_paper.sh --split test --rir-root data/rirs/dechorate \
#       --speech-dir data/speech/test --noise-dir data/noise/test
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
OUT=results
OURS=(--config configs/cradius_rse/continuous_crc.json --checkpoint checkpoints/continuous_crc.pt)

# Tables 1 and 2: proposed model (continuous radius encoding + CRC).
python evaluate.py "${OURS[@]}" --output-dir "$OUT/continuous_crc" "$@"

# Table 2: radius encoding and CRC ablation.
for variant in discrete_crc continuous_wo_crc; do
    python evaluate.py --config "configs/ablation/${variant}.json" --checkpoint "checkpoints/${variant}.pt" \
        --output-dir "$OUT/${variant}" "$@"
done

# Fig. 4(a): 0.1-m query sweep (output_rms / adjacent_change in per_output.csv).
python evaluate.py "${OURS[@]}" --radii 0:4:0.1 --seed 20260903 --output-dir "$OUT/fig4a_dense_sweep" "$@"

# Fig. 4(b): dense-query statistics on 500 clean two-speaker test mixtures.
python scripts/dense_radius_stats.py "${OURS[@]}" --output-dir "$OUT/fig4b_dense_stats"
