#!/usr/bin/env python3
"""Evaluate a trained model on deterministic regional-extraction scenes (Tables 1-2, Fig. 4a).

Examples
--------
Proposed model (released checkpoint):
    python evaluate.py --config configs/cradius_rse/continuous_crc.json         --checkpoint checkpoints/continuous_crc.pt --output-dir results/continuous_crc

Table 2 variants: ``--config configs/ablation/<variant>.json --checkpoint checkpoints/<variant>.pt``.
Dense query grid (Fig. 4a): ``--radii 0:4:0.1 --seed 20260903``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import soundfile as sf
import torch

from cradius_rse.data.regional_mixtures import profile_label
from cradius_rse.evaluation import (
    RadiusConditionedSystem,
    consistency_error,
    evaluate_output,
    evaluation_dataset,
    summarize_rce,
    summarize_rows,
    target_category,
    target_set,
)
from cradius_rse.utils import load_json


def parse_radii(text: str) -> tuple[float, ...]:
    if ":" in text:
        start, stop, step = (float(part) for part in text.split(":"))
        count = round((stop - start) / step)
        if count < 0 or not math.isclose(start + count * step, stop, abs_tol=1e-9):
            raise ValueError("--radii start:stop:step must reach stop exactly")
        radii = tuple(round(start + index * step, 10) for index in range(count + 1))
    else:
        radii = tuple(float(part) for part in text.split(",") if part.strip())
    if not radii or tuple(sorted(set(radii))) != radii:
        raise ValueError("--radii must be strictly increasing")
    return radii


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Experiment config of the model")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-config", default="configs/evaluation/scenes.json",
                        help="Dataset arguments of the evaluation scenes")
    parser.add_argument("--split", default="validation", choices=("validation", "test"))
    parser.add_argument("--rir-root", default=None, help="Override the RIR root (e.g. measured RIRs)")
    parser.add_argument("--speech-dir", default=None)
    parser.add_argument("--noise-dir", default=None)
    parser.add_argument("--source-counts", default="0,1,2")
    parser.add_argument("--mixtures-per-profile", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--radii", default="1,2,3,4", help="Comma list or start:stop:step")
    parser.add_argument("--noise-snr", type=float, default=None, help="Add noise at this SNR (dB)")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--save-audio", action="store_true", help="Write input and output WAV files")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def main():
    args = parse_args()
    if args.batch_size < 1 or args.mixtures_per_profile < 1 or args.torch_threads < 1:
        raise ValueError("Batch size, mixtures per profile and threads must be positive")
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory {output_dir}")

    radii = parse_radii(args.radii)
    source_counts = sorted({int(c) for c in args.source_counts.split(",")})
    data_args = load_json(args.data_config)
    for key, value in (("rir_root", args.rir_root), ("speech_dir", args.speech_dir), ("noise_dir", args.noise_dir)):
        if value is not None:
            data_args[key] = value
    dataset = evaluation_dataset(
        data_args, split=args.split, source_counts=source_counts,
        mixtures_per_profile=args.mixtures_per_profile, seed=args.seed,
        radii=radii, noise_snr_db=args.noise_snr,
    )
    system = RadiusConditionedSystem(args.config, args.checkpoint, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = dataset.source_profiles

    rows, pair_rows, scenes = [], [], []
    for start in range(0, len(dataset), args.batch_size):
        items = [dataset[index] for index in range(start, min(len(dataset), start + args.batch_size))]
        mixtures = torch.stack([inputs["mixture"][0] for inputs, _ in items])  # [B, 1, T]
        predictions = system.predict(mixtures, radii)  # one [B, 1, T] tensor per radius
        for offset, (inputs, targets) in enumerate(items):
            index = start + offset
            profile_id = int(inputs["profile_id"][0])
            label = profile_label(profiles[profile_id])
            source_count = int(inputs["source_count"][0])
            distances = inputs["source_distance_m"][0, :source_count].tolist()
            mixture = inputs["mixture"][0]
            scene_dir = Path(f"profile_{profile_id:02d}_{label}")
            mix_name = f"mix_{index // len(profiles) + 1:02d}"
            scenes.append({
                "mixture_index": index, "profile_id": profile_id, "profile": label,
                "source_count": source_count, "source_distances_m": distances,
                "source_gains_db": inputs["source_gain_db"][0, :source_count].tolist(),
                "room_id": inputs["room_id"][0], "noise_added": int(inputs["noise_added"][0]),
                "mixture_sha256": hashlib.sha256(mixture.numpy().astype("<f4").tobytes()).hexdigest(),
            })
            if args.save_audio:
                (output_dir / scene_dir).mkdir(parents=True, exist_ok=True)
                sf.write(output_dir / scene_dir / f"{mix_name}_input.wav", mixture.squeeze().numpy(), 16000, subtype="FLOAT")
            previous = None
            for q, radius in enumerate(radii):
                output = predictions[q][offset]
                target_count = int(targets["num_target_speakers"][q])
                metrics = evaluate_output(output, mixture, targets["target"][q], target_count)
                row = {
                    "mixture_index": index, "profile_id": profile_id, "profile": label,
                    "source_count": source_count, "query_radius_m": radius,
                    "category": target_category(source_count, target_count), **metrics,
                    "output_rms": float(torch.sqrt(torch.mean(output.double() ** 2))),
                    "adjacent_change": None,
                }
                if previous is not None:
                    row["adjacent_change"] = consistency_error(previous[1], output, mixture)
                    if target_set(distances, previous[0]) == target_set(distances, radius):
                        pair_rows.append({
                            "mixture_index": index, "profile_id": profile_id,
                            "radius_a": previous[0], "radius_b": radius,
                            "target_speakers": len(target_set(distances, radius)),
                            "rce": row["adjacent_change"],
                        })
                rows.append(row)
                previous = (radius, output)
                if args.save_audio:
                    sf.write(output_dir / scene_dir / f"{mix_name}_r{radius:g}m.wav",
                             output.squeeze().numpy(), 16000, subtype="FLOAT")
        print(f"evaluated {min(len(dataset), start + args.batch_size)}/{len(dataset)} mixtures", flush=True)

    aggregated = summarize_rows(rows)
    rce = summarize_rce(pair_rows)
    summary = {
        "config": args.config,
        "provenance": system.provenance,
        "protocol": {
            "split": args.split, "rir_root": data_args["rir_root"],
            "speech_dir": data_args["speech_dir"], "noise_dir": data_args["noise_dir"],
            "source_counts": source_counts, "profiles": len(profiles),
            "mixtures_per_profile": args.mixtures_per_profile, "mixtures": len(dataset),
            "query_radii_m": list(radii), "seed": args.seed, "noise_snr_db": args.noise_snr,
            "target_rule": "source_distance_m < query_radius_m",
        },
        "metrics": aggregated["summary"],
        "rce": rce,
    }
    write_csv(output_dir / "per_output.csv", rows)
    write_csv(output_dir / "cells.csv", aggregated["cells"])
    write_csv(output_dir / "rce_pairs.csv", pair_rows)
    (output_dir / "scenes.json").write_text(json.dumps(scenes, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cm = aggregated["summary"]["cell_median"]
    pm = aggregated["summary"]["profile_mean"]
    lines = [
        f"# {Path(args.config).stem} on {len(dataset)} mixtures x {len(radii)} radii ({args.split})",
        "",
        "| Aggregation | Full SI-SDR (dB) | Partial SI-SDRi (dB) | Empty Atten. (dB) |",
        "|---|---:|---:|---:|",
        f"| cell median | {fmt(cm['full'])} | {fmt(cm['partial'])} | {fmt(cm['empty'])} |",
        f"| profile mean | {fmt(pm['full'])} | {fmt(pm['partial'])} | {fmt(pm['empty'])} |",
        "",
        "| RCE (%) | n | median | mean |",
        "|---|---:|---:|---:|",
        f"| adjacent pairs | {rce['adjacent_pairs']['pairs']} | {fmt(rce['adjacent_pairs']['median_percent'])} | {fmt(rce['adjacent_pairs']['mean_percent'])} |",
        f"| per mixture (non-empty) | {rce['per_mixture_nonempty']['mixtures']} | {fmt(rce['per_mixture_nonempty']['median_percent'])} | {fmt(rce['per_mixture_nonempty']['mean_percent'])} |",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
