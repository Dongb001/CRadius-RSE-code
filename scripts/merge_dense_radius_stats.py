#!/usr/bin/env python3
"""Merge shards of ``dense_radius_stats.py`` and recompute the Fig. 4b summary.

    python scripts/merge_dense_radius_stats.py --input-dir results/fig4b_s0 \
        --input-dir results/fig4b_s1 --output-dir results/fig4b
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dense_radius_stats import CATEGORIES, summarize, write_csv  # noqa: E402

INVARIANT_KEYS = ("config", "checkpoint", "split", "rir_root", "speech_dir", "seed",
                  "query_radii_m", "distance_profiles", "normalized_bins")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    inputs = [Path(value) for value in args.input_dir]
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory {output}")

    manifests = [json.loads((path / "manifest.json").read_text(encoding="utf-8")) for path in inputs]
    for key in INVARIANT_KEYS:
        if any(item[key] != manifests[0][key] for item in manifests[1:]):
            raise ValueError(f"Shard manifests differ in {key}")
    ranges = sorted((m["sample_index_offset"], m["sample_index_stop_exclusive"]) for m in manifests)
    if any(left[1] != right[0] for left, right in zip(ranges, ranges[1:])):
        raise ValueError(f"Shard sample ranges are not contiguous: {ranges}")

    samples, queries, mixture_bins = [], [], []
    for path in inputs:
        samples += read_csv(path / "samples.csv")
        queries += read_csv(path / "per_query.csv")
        mixture_bins += read_csv(path / "per_mixture_bin.csv")
    samples.sort(key=lambda row: int(row["sample_index"]))
    queries.sort(key=lambda row: (int(row["sample_index"]), float(row["query_radius_m"])))
    mixture_bins.sort(key=lambda row: (int(row["sample_index"]), CATEGORIES.index(row["category"]),
                                       int(row["normalized_bin"])))
    if len({row["sample_index"] for row in samples}) != len(samples):
        raise ValueError("Duplicate sample indices across shards")
    if len(queries) != len(samples) * len(manifests[0]["query_radii_m"]):
        raise ValueError("Query row count does not match the number of scenes")

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "samples.csv", samples)
    write_csv(output / "per_query.csv", queries)
    write_csv(output / "per_mixture_bin.csv", mixture_bins)
    write_csv(output / "summary.csv", summarize(mixture_bins, int(manifests[0]["normalized_bins"])))
    merged = dict(manifests[0])
    merged.update(sample_index_offset=ranges[0][0], sample_index_stop_exclusive=ranges[-1][1],
                  scenes=len(samples), merged_shards=[str(path) for path in inputs])
    merged.pop("samples_per_profile", None)
    (output / "manifest.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"merged {len(inputs)} shards: scenes={len(samples)} queries={len(queries)} -> {output}")


if __name__ == "__main__":
    main()
