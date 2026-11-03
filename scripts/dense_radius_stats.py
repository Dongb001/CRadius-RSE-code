#!/usr/bin/env python3
"""Dense-query statistics of continuous-radius extraction (Fig. 4b).

Clean two-speaker test mixtures with source distances d1 < d2 < 4 m are
queried on a 0.1-m grid over [0, R_max]. Each query falls in the empty
[0, d1), partial [d1, d2) or full [d2, R_max] region; its position is
normalized to [0, 1] within that region and grouped into bins. Per mixture
and bin the median metric is taken; the summary reports the median and the
interquartile range over mixtures (attenuation / SI-SDRi / SI-SDR).

The 10 distance profiles (pairs of the bins [0,1), [1,2), [2,3), [3,4) m)
are cycled; a run of ``--samples-per-profile N`` covers indices
``offset .. offset + 10 N``. Shards can be merged with
``scripts/merge_dense_radius_stats.py``.

    python scripts/dense_radius_stats.py --config configs/cradius_rse/continuous_crc.json \
        --checkpoint runs/continuous_crc/checkpoints/best.pt --output-dir results/fig4b
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torchmetrics.functional import scale_invariant_signal_distortion_ratio as si_sdr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cradius_rse.data import RegionalMixtureDataset  # noqa: E402
from cradius_rse.utils import build_model, load_json, load_model_weights  # noqa: E402

CATEGORIES = ("empty", "partial", "full")
METRIC_BY_CATEGORY = {"empty": "attenuation_db", "partial": "si_sdri_db", "full": "si_sdr_db"}
PROFILES = [list(pair) for pair in itertools.combinations_with_replacement(range(4), 2)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-config", default="configs/evaluation/scenes.json")
    parser.add_argument("--rir-root", default=None)
    parser.add_argument("--speech-dir", default="data/speech/test")
    parser.add_argument("--noise-dir", default="data/noise/test")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples-per-profile", type=int, default=50)
    parser.add_argument("--sample-index-offset", type=int, default=0)
    parser.add_argument("--infer-batch-size", type=int, default=16)
    parser.add_argument("--scene-batch-size", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--query-step-m", type=float, default=0.1)
    parser.add_argument("--max-radius-m", type=float, default=4.0)
    parser.add_argument("--normalized-bins", type=int, default=10)
    return parser.parse_args()


def write_csv(path: Path, rows):
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def query_grid(step: float, max_radius: float) -> tuple[float, ...]:
    count = round(max_radius / step)
    if count <= 0 or not math.isclose(count * step, max_radius, abs_tol=1e-9):
        raise ValueError("query-step-m must divide the maximum radius exactly")
    return tuple(round(index * step, 10) for index in range(count + 1))


def category_from_count(count: int) -> str:
    return {0: "empty", 1: "partial", 2: "full"}[count]


def normalized_position(category, radius, d1, d2, max_radius):
    if category == "empty":
        value = radius / d1
    elif category == "partial":
        value = (radius - d1) / (d2 - d1)
    else:
        value = (radius - d2) / (max_radius - d2)
    if value < -1e-5 or value > 1.0 + 1e-5:
        raise ValueError(f"Normalized position outside [0, 1]: {category} r={radius} d1={d1} d2={d2}")
    return min(max(float(value), 0.0), 1.0)


def primary_metrics(outputs, mixtures, targets, counts):
    values = [None] * outputs.shape[0]
    empty = torch.nonzero(counts == 0, as_tuple=False).flatten()
    if empty.numel():
        mixture_power = mixtures[empty].double().square().sum(dim=(-2, -1))
        output_power = outputs[empty].double().square().sum(dim=(-2, -1))
        for index, value in zip(empty.tolist(), 10.0 * torch.log10(mixture_power / (output_power + 1e-12))):
            values[index] = float(value)
    positive = torch.nonzero(counts > 0, as_tuple=False).flatten()
    if positive.numel():
        sdr = si_sdr(outputs[positive], targets[positive]).reshape(-1)
        for offset, index in enumerate(positive.tolist()):
            values[index] = float(sdr[offset])
    partial = torch.nonzero(counts == 1, as_tuple=False).flatten()
    if partial.numel():
        baseline = si_sdr(mixtures[partial], targets[partial]).reshape(-1)
        for offset, index in enumerate(partial.tolist()):
            values[index] -= float(baseline[offset])
    if any(value is None or not math.isfinite(value) for value in values):
        raise ValueError("A primary metric is missing or non-finite")
    return values


def main():
    args = parse_args()
    if args.sample_index_offset % len(PROFILES):
        raise ValueError(f"sample-index-offset must be divisible by {len(PROFILES)}")
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    radii = query_grid(args.query_step_m, args.max_radius_m)
    scene_count = len(PROFILES) * args.samples_per_profile
    data_args = load_json(args.data_config)
    data_args.update(
        speech_dir=args.speech_dir, noise_dir=args.noise_dir,
        base_mixtures_per_epoch=args.sample_index_offset + scene_count,
        query_radii=list(radii), query_sampling="fixed",
        source_profiles=[{"source_count": 2, "distance_bin_ids": p} for p in PROFILES],
        profile_weight_per_source_count=None, noise_probability=0.0, seed=args.seed,
    )
    if args.rir_root:
        data_args["rir_root"] = args.rir_root
    dataset = RegionalMixtureDataset(**data_args, split=args.split)
    dataset.set_epoch(0)

    config = load_json(args.config)
    model, _ = load_model_weights(build_model(config), args.checkpoint)
    model.to(device).eval()

    query_rows, sample_rows = [], []
    per_bin = defaultdict(list)
    start_time = time.time()
    for scene_start in range(0, scene_count, args.scene_batch_size):
        group = []
        for local in range(scene_start, min(scene_count, scene_start + args.scene_batch_size)):
            index = args.sample_index_offset + local
            inputs, targets = dataset[index]
            distances = inputs["source_distance_m"][0].tolist()
            order = sorted(range(2), key=lambda i: distances[i])
            d1, d2 = (distances[i] for i in order)
            if not 0.0 < d1 < d2 < args.max_radius_m:
                raise ValueError(f"Scene {index} violates 0 < d1 < d2 < R_max: {distances}")
            profile_id = int(inputs["profile_id"][0])
            gains = inputs["source_gain_db"][0].tolist()
            sample_rows.append({
                "sample_index": index, "profile_id": profile_id,
                "distance_bin_1": PROFILES[profile_id][0], "distance_bin_2": PROFILES[profile_id][1],
                "room_id": inputs["room_id"][0], "d1_m": d1, "d2_m": d2,
                "gain_d1_db": gains[order[0]], "gain_d2_db": gains[order[1]],
            })
            group.append((index, profile_id, d1, d2, inputs, targets))

        mixtures = torch.cat([g[4]["mixture"][0].unsqueeze(0).repeat(len(radii), 1, 1) for g in group])
        targets = torch.cat([g[5]["target"] for g in group])
        counts = torch.cat([g[5]["num_target_speakers"] for g in group])
        radius_tensor = torch.tensor(radii, dtype=torch.float32).repeat(len(group)).view(-1, 1)
        outputs = []
        with torch.inference_mode():
            for start in range(0, mixtures.shape[0], args.infer_batch_size):
                stop = min(start + args.infer_batch_size, mixtures.shape[0])
                y = model({"mixture": mixtures[start:stop].to(device),
                           "query_radius": radius_tensor[start:stop].to(device)})["output"].cpu()
                outputs.append(y[..., : mixtures.shape[-1]].clamp(-1.0, 1.0))
        values = primary_metrics(torch.cat(outputs), mixtures, targets, counts)

        flat = 0
        for index, profile_id, d1, d2, _, _ in group:
            for radius in radii:
                count = int(counts[flat])
                category = category_from_count(count)
                position = normalized_position(category, radius, d1, d2, args.max_radius_m)
                bin_index = min(int(position * args.normalized_bins), args.normalized_bins - 1)
                query_rows.append({
                    "sample_index": index, "profile_id": profile_id, "d1_m": d1, "d2_m": d2,
                    "query_radius_m": radius, "target_speakers": count, "category": category,
                    "normalized_u": position, "normalized_bin": bin_index,
                    "metric_name": METRIC_BY_CATEGORY[category], "metric_db": values[flat],
                })
                per_bin[(index, profile_id, category, bin_index)].append(values[flat])
                flat += 1
        done = min(scene_count, scene_start + args.scene_batch_size)
        rate = done / (time.time() - start_time)
        print(f"evaluated {done}/{scene_count} scenes ({rate:.2f} scenes/s)", flush=True)

    mixture_bin_rows = []
    for (index, profile_id, category, bin_index), vals in sorted(per_bin.items()):
        mixture_bin_rows.append({
            "sample_index": index, "profile_id": profile_id, "category": category,
            "normalized_bin": bin_index, "bin_left": bin_index / args.normalized_bins,
            "bin_right": (bin_index + 1) / args.normalized_bins,
            "metric_name": METRIC_BY_CATEGORY[category], "query_count": len(vals),
            "mixture_bin_median_db": statistics.median(vals),
        })
    write_csv(output_dir / "per_query.csv", query_rows)
    write_csv(output_dir / "samples.csv", sample_rows)
    write_csv(output_dir / "per_mixture_bin.csv", mixture_bin_rows)
    write_csv(output_dir / "summary.csv", summarize(mixture_bin_rows, args.normalized_bins))
    manifest = {
        "config": args.config, "checkpoint": args.checkpoint, "split": args.split,
        "rir_root": data_args["rir_root"], "speech_dir": data_args["speech_dir"],
        "seed": args.seed, "query_radii_m": list(radii), "queries_per_scene": len(radii),
        "distance_profiles": PROFILES, "samples_per_profile": args.samples_per_profile,
        "sample_index_offset": args.sample_index_offset,
        "sample_index_stop_exclusive": args.sample_index_offset + scene_count,
        "normalized_bins": args.normalized_bins,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"output_dir={output_dir.resolve()}")


def summarize(mixture_bin_rows, normalized_bins):
    grouped = defaultdict(list)
    query_counts = defaultdict(int)
    for row in mixture_bin_rows:
        key = (row["category"], int(row["normalized_bin"]))
        grouped[key].append(float(row["mixture_bin_median_db"]))
        query_counts[key] += int(row["query_count"])
    rows = []
    for category in CATEGORIES:
        for bin_index in range(normalized_bins):
            values = np.asarray(grouped[(category, bin_index)], dtype=np.float64)
            if not values.size:
                raise RuntimeError(f"No values for category={category}, bin={bin_index}")
            rows.append({
                "category": category, "normalized_bin": bin_index,
                "bin_left": bin_index / normalized_bins, "bin_right": (bin_index + 1) / normalized_bins,
                "metric_name": METRIC_BY_CATEGORY[category], "mixture_count": values.size,
                "query_point_count": query_counts[(category, bin_index)],
                "median_db": float(np.quantile(values, 0.50)),
                "q25_db": float(np.quantile(values, 0.25)),
                "q75_db": float(np.quantile(values, 0.75)),
            })
    return rows


if __name__ == "__main__":
    main()
