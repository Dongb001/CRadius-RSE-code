#!/usr/bin/env python3
"""Convert measured dEchorate RIRs into the RIR format used by this repository.

Input: a directory with the 16 kHz dEchorate RIR WAV files and a
``metadata.csv`` with (at least) the columns ``relpath, room_code, src_id,
mic_type, mic_id, dist_m``, where ``dist_m`` is the source-to-capsule distance
computed from the official source/microphone coordinates.

For every capsule RIR (loopback channels are skipped) the first second is kept
and the response is scaled so that its peak equals 1 / (4 pi d), the free-field
amplitude at the labelled distance d. Files are written as

    <output>/<split>/room_dechorate_<room_code>/rir_XXX.npz   {rir: [1, 16000], distance}

ordered by (source ID, microphone ID). By default the rooms 000000, 011111 and
020002 form the test split.

    python data_preparation/prepare_dechorate.py --dechorate-dir /path/to/dEchorate \
        --output data/rirs/dechorate
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


def convert(wav_path: Path, distance: float, length: int) -> np.ndarray:
    audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if sample_rate != 16000:
        raise ValueError(f"{wav_path}: expected 16 kHz, got {sample_rate}")
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio[:, 0]
    segment = audio[:length]
    peak = float(np.max(np.abs(segment)))
    if not math.isfinite(peak) or peak <= 0:
        raise ValueError(f"{wav_path}: silent RIR")
    scale = (1.0 / (4.0 * math.pi * distance)) / peak
    return (segment * scale).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dechorate-dir", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, default=None, help="Default: <dechorate-dir>/metadata.csv")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--rooms", default="000000,011111,020002")
    parser.add_argument("--length", type=int, default=16000, help="Samples kept per RIR (1 s)")
    args = parser.parse_args()

    metadata = args.metadata or args.dechorate_dir / "metadata.csv"
    with metadata.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["mic_type"] == "capsule"]
    rooms = [room.strip() for room in args.rooms.split(",") if room.strip()]
    by_room = defaultdict(list)
    for row in rows:
        if row["room_code"] in rooms:
            by_room[row["room_code"]].append(row)

    for room in rooms:
        items = sorted(by_room[room], key=lambda row: (int(row["src_id"]), int(row["mic_id"])))
        if not items:
            raise FileNotFoundError(f"No capsule RIRs for room {room} in {metadata}")
        room_dir = args.output / args.split / f"room_dechorate_{room}"
        room_dir.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(items):
            distance = float(row["dist_m"])
            rir = convert(args.dechorate_dir / row["relpath"], distance, args.length)
            np.savez_compressed(room_dir / f"rir_{index:03d}.npz", rir=rir.reshape(1, -1), distance=distance)
        print(f"room {room}: {len(items)} RIRs -> {room_dir}")


if __name__ == "__main__":
    main()
