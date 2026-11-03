#!/usr/bin/env python3
"""Simulate single-channel RIRs with known source distances using gpuRIR (Sec. 3.1).

1,000 shoebox rooms are split into 800 training, 100 validation and 100 test
rooms with 64 RIRs each. Room length and width are drawn from [2, 12] m,
height from [2, 5] m, RT60 from [0.1, 1.0] s and source distances are spread
uniformly over [0.3, 5.0] m. The output layout is

    <rir_dir>/rir_cfg.npz                       room geometry of all splits
    <rir_dir>/{train,validation,test}/room_XXXX/rir_YYY.npz   {rir, distance}

If ``rir_cfg.npz`` already exists it is reused, so the RIR files can be
re-rendered from a released room configuration.

    python data_preparation/simulate_rirs.py --rir-dir data/rirs/simulated --gpus 0
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from functools import partial

import numpy as np
import tqdm
from numpy.linalg import norm
from numpy.random import uniform
from scipy.optimize import minimize


# ----------------------------------------------------------------- room acoustics
def estimate_minimal_rt60(room_sz) -> float:
    volume = float(np.prod(room_sz))
    surface = 2 * (room_sz[0] * room_sz[1] + room_sz[0] * room_sz[2] + room_sz[1] * room_sz[2])
    return 0.161 * volume / surface


def is_valid_rt60_for_room(room_sz, rt60: float, eps: float = 1e-4) -> bool:
    return rt60 >= estimate_minimal_rt60(room_sz) + eps


def beta_sabine_estimation(room_sz, t60, abs_weights=(1.0,) * 6):
    """Wall reflection coefficients reaching ``t60`` by Sabine's formula.

    Same estimator as ``gpuRIR.beta_SabineEstimation``; additionally returns the
    residual RT60 error so that infeasible absorption weights can be redrawn.
    """

    def t60error(x, t60, room_sz, abs_weights):
        alpha = x * abs_weights
        sa = (alpha[0] + alpha[1]) * room_sz[1] * room_sz[2] + \
             (alpha[2] + alpha[3]) * room_sz[0] * room_sz[2] + \
             (alpha[4] + alpha[5]) * room_sz[0] * room_sz[1]
        if sa == 0:
            return t60
        return abs(t60 - 0.161 * np.prod(room_sz) / sa)

    abs_weights = np.asarray(abs_weights, dtype=np.float64)
    abs_weights /= abs_weights.max()
    result = minimize(t60error, 0.5, args=(t60, room_sz, abs_weights), bounds=[[0, 1]])
    return np.sqrt(1 - result.x * abs_weights).astype(np.float32), result.fun


def simulate_rir_gpu(room_sz, pos_src, pos_rcv, rt60, fs, beta, att_diff=15.0, att_max=60.0, c=343.0):
    import gpuRIR

    gpuRIR.activateMixedPrecision(False)
    gpuRIR.activateLUT(False)
    room_sz = np.asarray(room_sz).reshape(1, -1)
    pos_src = np.asarray(pos_src).reshape(-1, 3)
    pos_rcv = np.asarray(pos_rcv).reshape(-1, 3)
    t_max = gpuRIR.att2t_SabineEstimator(att_max, rt60)
    t_diff = gpuRIR.att2t_SabineEstimator(att_diff, rt60)
    nb_img = gpuRIR.t2n(t_diff, room_sz[0])
    return gpuRIR.simulateRIR(room_sz=room_sz[0], beta=beta, pos_src=pos_src, pos_rcv=pos_rcv,
                              nb_img=nb_img, Tmax=t_max, Tdiff=t_diff, fs=fs, c=c)


# ---------------------------------------------------------------- source layout
def position_at_distance(room_sz, mic_pos, target_distance, wall_margin=0.5, z_lim=(1.2, 2.0),
                         max_attempts=1000, tolerance=0.1):
    """A source position inside the room at ``target_distance`` (+- tolerance)."""
    low, high = target_distance - tolerance, target_distance + tolerance
    for _ in range(max_attempts):
        theta = uniform(0, 2 * np.pi)
        phi = uniform(0, np.pi)
        offset = target_distance * np.array(
            [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
        )
        pos = mic_pos + offset
        if (wall_margin <= pos[0] <= room_sz[0] - wall_margin
                and wall_margin <= pos[1] <= room_sz[1] - wall_margin
                and z_lim[0] <= pos[2] <= z_lim[1]
                and low <= norm(pos - mic_pos) <= high):
            return pos
    for _ in range(max_attempts):
        pos = np.array([uniform(wall_margin, room_sz[0] - wall_margin),
                        uniform(wall_margin, room_sz[1] - wall_margin),
                        uniform(z_lim[0], z_lim[1])])
        if low <= norm(pos - mic_pos) <= high:
            return pos
    return None


def distance_uniform_positions(room_sz, mic_pos, num_positions, distance_range=(0.3, 5.0),
                               wall_margin=0.5, spk_zlim=(1.2, 2.0)):
    positions, distances = [], []
    targets = np.linspace(distance_range[0], distance_range[1], num_positions)
    targets += uniform(-0.05, 0.05, num_positions)
    for target in targets:
        pos = position_at_distance(room_sz, mic_pos, target, wall_margin, spk_zlim,
                                   max_attempts=500, tolerance=0.15)
        if pos is not None:
            positions.append(pos)
            distances.append(norm(pos - mic_pos))
            continue
        for _ in range(100):  # fallback: any position inside the distance range
            pos = np.array([uniform(wall_margin, room_sz[0] - wall_margin),
                            uniform(wall_margin, room_sz[1] - wall_margin),
                            uniform(spk_zlim[0], spk_zlim[1])])
            dist = norm(pos - mic_pos)
            if distance_range[0] <= dist <= distance_range[1]:
                positions.append(pos)
                distances.append(dist)
                break
    return positions, distances


def room_config(room_sz, mic_pos, rt60, num_rirs, distance_range, wall_abs_weights_lims,
                wall_margin, spk_zlim, seed):
    np.random.seed(seed)
    abs_weights = [uniform(*lim) for lim in wall_abs_weights_lims]
    beta, error = beta_sabine_estimation(room_sz, rt60, abs_weights)
    while error > 0.05:
        abs_weights = [uniform(*lim) for lim in wall_abs_weights_lims]
        beta, error = beta_sabine_estimation(room_sz, rt60, abs_weights)
    positions, distances = distance_uniform_positions(
        room_sz, mic_pos, num_rirs, distance_range, wall_margin, spk_zlim
    )
    return {"room_sz": room_sz, "pos_rcv": mic_pos[np.newaxis, :].astype(np.float32),
            "pos_src": positions, "distances": distances, "RT60": rt60, "beta": beta,
            "num_rirs": len(positions)}


def make_room_configs(args):
    save_to = os.path.join(args.rir_dir, "rir_cfg.npz")
    if os.path.exists(save_to):
        print(f"Loading the existing room configuration {save_to}")
        return dict(np.load(save_to, allow_pickle=True))

    np.random.seed(args.seed)
    wall_abs = [(0.5, 1.0)] * 6
    total = args.num_rooms_train + args.num_rooms_val + args.num_rooms_test
    rooms = []
    for room_idx in range(total):
        room_sz = [uniform(*args.room_length), uniform(*args.room_width), uniform(*args.room_height)]
        mic_pos = np.array([uniform(args.wall_margin, room_sz[0] - args.wall_margin),
                            uniform(args.wall_margin, room_sz[1] - args.wall_margin),
                            uniform(*args.mic_height)])
        rt60 = uniform(*args.rt60)
        while not is_valid_rt60_for_room(room_sz, rt60):
            rt60 = uniform(*args.rt60)
        if room_idx < args.num_rooms_train:
            split = "train"
        elif room_idx < args.num_rooms_train + args.num_rooms_val:
            split = "validation"
        else:
            split = "test"
        cfg = room_config(room_sz, mic_pos, rt60, args.rir_per_room, tuple(args.distance_range),
                          wall_abs, args.wall_margin, tuple(args.source_height), args.seed + room_idx)
        cfg["dataset"] = split
        cfg["room_idx"] = room_idx
        rooms.append(cfg)
    cfg = {
        "args": np.array(vars(args)),
        "train_rooms": [r for r in rooms if r["dataset"] == "train"],
        "val_rooms": [r for r in rooms if r["dataset"] == "validation"],
        "test_rooms": [r for r in rooms if r["dataset"] == "test"],
    }
    os.makedirs(args.rir_dir, exist_ok=True)
    np.savez_compressed(save_to, **cfg)
    print(f"Saved the room configuration to {save_to}")
    return cfg


# -------------------------------------------------------------------- rendering
def render_room(room, fs, rir_dir, att_diff, att_max):
    room_dir = os.path.join(rir_dir, room["dataset"], f"room_{room['room_idx']:04d}")
    os.makedirs(room_dir, exist_ok=True)
    config_file = os.path.join(room_dir, "room_config.npz")
    if os.path.exists(config_file):
        return
    room_sz = np.array(room["room_sz"], dtype=np.float32)
    pos_rcv = np.array(room["pos_rcv"], dtype=np.float32)
    beta = np.array(room["beta"], dtype=np.float32)
    rir_files = []
    for index, (pos_src, dist) in enumerate(zip(room["pos_src"], room["distances"])):
        rir = simulate_rir_gpu(room_sz, np.array(pos_src, dtype=np.float32).reshape(1, -1), pos_rcv,
                               room["RT60"], fs, beta, att_diff, att_max)
        name = f"rir_{index:03d}.npz"
        np.savez_compressed(os.path.join(room_dir, name), rir=rir[0].astype(np.float32), distance=float(dist))
        rir_files.append(name)
    np.savez(config_file, fs=fs, RT60=room["RT60"], room_sz=room_sz, pos_rcv=pos_rcv,
             pos_src=np.array(room["pos_src"], dtype=np.float32),
             distances=np.array(room["distances"], dtype=np.float32), rir_files=rir_files,
             room_idx=room["room_idx"], dataset=room["dataset"])


def _init_worker(queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(queue.get())


def render_all(cfg, rir_dir, gpus, att_diff, att_max, fs):
    for split, key in (("train", "train_rooms"), ("validation", "val_rooms"), ("test", "test_rooms")):
        rooms = list(cfg[key])
        os.makedirs(os.path.join(rir_dir, split), exist_ok=True)
        queue = mp.Queue()
        for gpu in gpus:
            queue.put(gpu)
        with mp.Pool(processes=len(gpus), initializer=_init_worker, initargs=(queue,)) as pool:
            worker = partial(render_room, fs=fs, rir_dir=rir_dir, att_diff=att_diff, att_max=att_max)
            for _ in tqdm.tqdm(pool.imap_unordered(worker, rooms, chunksize=5), total=len(rooms), desc=split):
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rir-dir", required=True)
    parser.add_argument("--seed", type=int, default=694235206)
    parser.add_argument("--num-rooms-train", type=int, default=800)
    parser.add_argument("--num-rooms-val", type=int, default=100)
    parser.add_argument("--num-rooms-test", type=int, default=100)
    parser.add_argument("--rir-per-room", type=int, default=64)
    parser.add_argument("--room-length", type=float, nargs=2, default=(2.0, 12.0))
    parser.add_argument("--room-width", type=float, nargs=2, default=(2.0, 12.0))
    parser.add_argument("--room-height", type=float, nargs=2, default=(2.0, 5.0))
    parser.add_argument("--rt60", type=float, nargs=2, default=(0.1, 1.0))
    parser.add_argument("--distance-range", type=float, nargs=2, default=(0.3, 5.0))
    parser.add_argument("--wall-margin", type=float, default=0.5)
    parser.add_argument("--mic-height", type=float, nargs=2, default=(1.0, 1.5))
    parser.add_argument("--source-height", type=float, nargs=2, default=(1.2, 2.0))
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--att-diff", type=float, default=15.0)
    parser.add_argument("--att-max", type=float, default=60.0)
    args = parser.parse_args()

    start = time.time()
    cfg = make_room_configs(args)
    render_all(cfg, args.rir_dir, args.gpus, args.att_diff, args.att_max, args.fs)
    print(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
