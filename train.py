#!/usr/bin/env python3
"""Train a model from an experiment config.

    python train.py --config configs/cradius_rse/continuous_crc.json --run-dir runs/continuous_crc

The config is copied to ``<run-dir>/config.json``. ``--init-checkpoint`` replaces the
initialization of the config, e.g. the released ``checkpoints/discrete_init.pt``. Checkpoints are written to
``<run-dir>/checkpoints``: ``last.pt`` (resumed automatically), ``best.pt``
and the three best epochs according to the configured monitor.
"""

import argparse
import multiprocessing
import os
import shutil
import time
from functools import partial

import numpy as np
import torch
import torch.utils.data

from cradius_rse.training import train_epoch, validate_epoch
from cradius_rse.utils import import_attr, load_json, seed_all

VAL_SEED = 0
CURRENT_EPOCH = 0
DEFAULT_MODULE = "cradius_rse.training.TrainingModule"


def seed_train_worker(seed, worker_id=0):
    seed_all(seed + CURRENT_EPOCH * 1009 + int(worker_id))


def seed_validation_worker(worker_id):
    seed_all(VAL_SEED + int(worker_id))


def update_early_stopping(value, best, stale_epochs, mode, min_delta):
    if not np.isfinite(value):
        return best, stale_epochs + 1, False
    if mode == "min":
        improved = best is None or value < best - min_delta
    elif mode == "max":
        improved = best is None or value > best + min_delta
    else:
        raise ValueError(f"early_stopping.mode must be min or max, got {mode!r}")
    if improved:
        return float(value), 0, True
    return best, stale_epochs + 1, False


def build_module(config):
    module_cls = import_attr(config.get("training_module", DEFAULT_MODULE))
    return module_cls(model=config["model"], model_args=config.get("model_args", {}), **config["training_args"])


def train(args):
    global CURRENT_EPOCH
    seed_all(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = not args.nondeterministic_cudnn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    config = load_json(args.config)
    if args.init_checkpoint is not None:
        config["training_args"]["init_checkpoint"] = args.init_checkpoint
    data_train = import_attr(config["train_dataset"])(**config["train_dataset_args"], split="train")
    data_val = import_attr(config["val_dataset"])(**config["val_dataset_args"], split="val")

    loader_args = {
        "num_workers": min(multiprocessing.cpu_count(), config["num_workers"]),
        "worker_init_fn": partial(seed_train_worker, args.seed),
        "pin_memory": device.type == "cuda",
    }
    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=config["batch_size"], shuffle=config.get("train_shuffle", True),
        collate_fn=getattr(data_train, "collate_fn", None), **loader_args,
    )
    loader_args["worker_init_fn"] = seed_validation_worker
    val_loader = torch.utils.data.DataLoader(
        data_val, batch_size=config["eval_batch_size"],
        collate_fn=getattr(data_val, "collate_fn", None), **loader_args,
    )

    module = build_module(config)
    module.model.to(device)

    checkpoints_dir = os.path.join(args.run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    if not os.path.exists(os.path.join(args.run_dir, "config.json")):
        shutil.copyfile(args.config, os.path.join(args.run_dir, "config.json"))
    best_path = os.path.join(checkpoints_dir, "best.pt")
    last_path = os.path.join(checkpoints_dir, "last.pt")
    if os.path.exists(last_path):
        module.load_state(last_path)
        print(f"Resumed from {last_path} at epoch {module.epoch}")

    early = config.get("early_stopping", {})
    early_enabled = bool(early.get("enabled", False))
    early_metric = early.get("metric", "val/loss")
    early_mode = early.get("mode", "min")
    early_patience = int(early.get("patience", 10))
    early_min_delta = float(early.get("min_delta", 0.0))
    if early_enabled and early_patience <= 0:
        raise ValueError("early_stopping.patience must be positive")
    early_best, early_stale = None, 0
    if early_enabled:
        for epoch in sorted(module.metric_values):
            if early_metric in module.metric_values[epoch]:
                value = module.get_avg_metric_at_epoch(early_metric, epoch)
                early_best, early_stale, _ = update_early_stopping(
                    value, early_best, early_stale, early_mode, early_min_delta
                )

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, name=os.path.basename(args.run_dir.rstrip("/")))

    for epoch in range(module.epoch, config["epochs"]):
        CURRENT_EPOCH = epoch
        seed_train_worker(args.seed)
        for dataset in (data_train, data_val):
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)
        module.on_epoch_start()
        print(f"Learning rate: {module.get_current_lr():.8f}")

        start = time.time()
        train_loss = train_epoch(module, train_loader, device)
        print(f"Train epoch time: {time.time() - start:.1f}s, average loss: {train_loss:.4f}")

        # Fixed seed for every validation pass.
        seed_all(VAL_SEED)
        val_loss = validate_epoch(module, val_loader, device)
        print(f"Validation average loss: {val_loss:.4f}")

        module.on_epoch_end(best_path, wandb_run)
        module.dump_state(last_path)

        stop = False
        if early_enabled:
            metric_epoch = module.epoch - 1
            if early_metric not in module.metric_values[metric_epoch]:
                raise KeyError(f"Early-stopping metric {early_metric!r} was not logged at epoch {metric_epoch}")
            value = module.get_avg_metric_at_epoch(early_metric, metric_epoch)
            early_best, early_stale, improved = update_early_stopping(
                value, early_best, early_stale, early_mode, early_min_delta
            )
            print(f"[EARLY_STOP] {early_metric}={value:.6f} best={early_best:.6f} "
                  f"stale={early_stale}/{early_patience} improved={improved}")
            stop = early_stale >= early_patience

        print()
        print("=" * 25, "FINISHED EPOCH", epoch, "=" * 25)
        print()
        if stop:
            print(f"[EARLY_STOP] No improvement of {early_metric} for {early_patience} epochs.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Experiment config (JSON)")
    parser.add_argument("--run-dir", required=True, help="Output directory of the run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-checkpoint", default=None,
                        help="Override training_args.init_checkpoint (e.g. checkpoints/discrete_init.pt)")
    parser.add_argument("--nondeterministic-cudnn", action="store_true",
                        help="Allow non-deterministic cuDNN kernels (faster)")
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb-project", default="CRadius-RSE")
    train(parser.parse_args())
