"""Training module: model, loss, optimizer, scheduler, metrics and checkpoints."""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn
from torchmetrics.functional import (
    scale_invariant_signal_distortion_ratio,
    scale_invariant_signal_noise_ratio,
    signal_noise_ratio,
)

from ..utils import import_attr

try:
    import wandb
except ImportError:  # wandb is optional
    wandb = None


def _improvement_metric(name):
    functions = {
        "snr_i": signal_noise_ratio,
        "si_snr_i": scale_invariant_signal_noise_ratio,
        "si_sdr_i": scale_invariant_signal_distortion_ratio,
    }
    if name not in functions:
        raise ValueError(f"Unknown training metric {name!r}; choose from {sorted(functions)}")
    function = functions[name]
    return lambda est, gt, mix: (function(preds=est, target=gt) - function(preds=mix, target=gt)).mean(dim=-1)


def attenuation_db(est: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    """Input-to-output power ratio in dB, ``[*, C, T] -> [*]``."""
    p_est = 10 * torch.log10(torch.sum(est**2, dim=-1))
    p_mix = 10 * torch.log10(torch.sum(mix**2, dim=-1))
    return (p_mix - p_est).mean(dim=-1)


class TrainingModule:
    """Wraps one experiment; ``train.py`` drives it epoch by epoch.

    Args:
        model / model_args: dotted path and arguments of the network.
        loss / loss_args: dotted path and arguments of the training objective.
        optimizer / optimizer_args, scheduler / scheduler_args: as in ``torch.optim``.
        warmup_steps: linear learning-rate warm-up in optimizer steps.
        grad_clip: gradient-norm clipping.
        metrics: improvement metrics logged for non-empty targets.
        init_checkpoint: optional weights used to initialize the model.
        init_checkpoint_skip_prefixes: parameters (e.g. ``radius_encoder.``) left
            randomly initialized when loading ``init_checkpoint``.
        monitor / monitor_mode: metric that selects the best checkpoint.
        data_parallel: wrap the model in ``nn.DataParallel``.
    """

    def __init__(
        self,
        model,
        model_args,
        optimizer,
        optimizer_args,
        loss,
        loss_args,
        scheduler=None,
        scheduler_args=None,
        warmup_steps=0,
        grad_clip=None,
        metrics=("snr_i", "si_snr_i", "si_sdr_i"),
        init_checkpoint=None,
        init_checkpoint_skip_prefixes=None,
        monitor="val/loss",
        monitor_mode="min",
        data_parallel=False,
        sample_rate=16000,
        logged_samples_per_count=3,
    ):
        self.model = import_attr(model)(**model_args)
        self.data_parallel = bool(data_parallel)
        if self.data_parallel:
            self.model = nn.DataParallel(self.model)
        self.sample_rate = int(sample_rate)
        self.logged_samples_per_count = int(logged_samples_per_count)
        self.metrics = {name: _improvement_metric(name) for name in metrics}
        self.metric_values = {}
        self.statistics = {}
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        if self.monitor_mode not in ("min", "max"):
            raise ValueError("monitor_mode must be 'min' or 'max'")
        self.val_samples = {}
        self.train_samples = {}
        self.loss_fn = import_attr(loss)(**(loss_args or {}))

        if init_checkpoint is not None:
            self._load_initial_checkpoint(init_checkpoint, tuple(init_checkpoint_skip_prefixes or ()))

        self.optimizer_name = optimizer
        self.optimizer_args = optimizer_args
        self.optimizer = import_attr(optimizer)(self.model.parameters(), **optimizer_args)
        self.warmup_steps = int(warmup_steps)
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        self.optimizer_step = 0
        self.base_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        if self.warmup_steps:
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = base_lr / self.warmup_steps
            print(f"Linear LR warm-up over {self.warmup_steps} optimizer steps, base_lr={self.base_lrs}")

        self.grad_clip = grad_clip
        print(f"Gradient clipping: {self.grad_clip}")
        self.scheduler_name = scheduler
        self.scheduler_args = scheduler_args
        self.scheduler = self._init_scheduler()
        self.epoch = 0

    # ------------------------------------------------------------------ state
    @property
    def bare_model(self):
        return self.model.module if self.data_parallel else self.model

    @staticmethod
    def _normalize_checkpoint_key(key):
        key = str(key)
        for _ in range(3):
            changed = False
            for prefix in ("module.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
            if not changed:
                break
        return key

    def _load_initial_checkpoint(self, path, skip_prefixes):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            source_state = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            source_state = checkpoint["state_dict"]
        else:
            source_state = checkpoint
        target_state = self.bare_model.state_dict()
        load_state, skipped, incompatible, unexpected = {}, [], [], []
        for raw_key, value in source_state.items():
            key = self._normalize_checkpoint_key(raw_key)
            if any(key.startswith(prefix) for prefix in skip_prefixes):
                skipped.append(key)
            elif key not in target_state:
                unexpected.append(key)
            elif target_state[key].shape != value.shape:
                incompatible.append(f"{key}: {tuple(value.shape)} vs {tuple(target_state[key].shape)}")
            else:
                load_state[key] = value
        result = self.bare_model.load_state_dict(load_state, strict=False)
        missing = [k for k in result.missing_keys if not any(k.startswith(p) for p in skip_prefixes)]
        print(f"Initialized from {path}: loaded={len(load_state)} skipped={len(skipped)}")
        if incompatible:
            print("  shape-mismatched parameters ignored: " + "; ".join(incompatible[:8]))
        if unexpected:
            print(f"  unexpected checkpoint parameters ignored: {len(unexpected)}")
        if missing:
            print("  WARNING: missing parameters: " + ", ".join(missing[:12]))

    def _init_scheduler(self):
        if self.scheduler_name is None:
            return None
        return import_attr(self.scheduler_name)(self.optimizer, **(self.scheduler_args or {}))

    def load_state(self, path, map_location=None):
        state = torch.load(path, map_location=map_location, weights_only=False)
        self.bare_model.load_state_dict(state["model"])
        self.optimizer = import_attr(self.optimizer_name)(self.model.parameters(), **self.optimizer_args)
        if self.scheduler is not None:
            self.scheduler = self._init_scheduler()
        self.optimizer.load_state_dict(state["optimizer"])
        self.optimizer_step = int(state.get("optimizer_step", 0))
        self.base_lrs = list(state.get("base_lrs", self.base_lrs))
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.epoch = state["current_epoch"]
        self.metric_values = state["metric_values"]
        self.statistics = state.get("statistics", {})

    def dump_state(self, path):
        state = dict(
            model=self.bare_model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            optimizer_step=self.optimizer_step,
            base_lrs=self.base_lrs,
            current_epoch=self.epoch,
            metric_values=self.metric_values,
            statistics=self.statistics,
        )
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        torch.save(state, path)

    # ---------------------------------------------------------------- metrics
    def get_current_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def get_avg_metric_at_epoch(self, metric, epoch=None):
        epoch = self.epoch if epoch is None else epoch
        entry = self.metric_values[epoch][metric]
        return entry["epoch"] / entry["num_elements"]

    def has_metric_at_epoch(self, metric, epoch=None):
        epoch = self.epoch if epoch is None else epoch
        return epoch in self.metric_values and metric in self.metric_values[epoch]

    def log_metric(self, name, value, batch_size=1, on_step=False):
        """Accumulate a metric; ``value`` must be the average over ``batch_size`` items."""
        epoch_values = self.metric_values.setdefault(self.epoch, {})
        entry = epoch_values.setdefault(name, dict(step=None, epoch=None))
        if isinstance(value, torch.Tensor):
            value = value.item()
        if on_step:
            entry["step"] = (entry["step"] or []) + [value]
        if entry["epoch"] is None:
            entry["epoch"] = 0
            entry["num_elements"] = 0
        entry["epoch"] += value * batch_size
        entry["num_elements"] += batch_size

    def log_statistic(self, name, value, reduction="mean"):
        if name not in self.statistics:
            self.statistics[name] = dict(logged=False, data=[], reduction=reduction)
        self.statistics[name]["data"].append(value)

    def _log_loss_terms(self, step, aux, batch_size, group_batch_size):
        for name, value in aux.items():
            if name.endswith("_count"):
                continue
            if f"{name}_count" in aux:
                weight = int(aux[f"{name}_count"])
            elif name.startswith("nonempty_"):
                weight = int(aux.get("nonempty_count", 0))
            elif name.startswith("empty_"):
                weight = int(aux.get("empty_count", 0))
            elif name.startswith("group_") or name.startswith("crc"):
                weight = group_batch_size
            else:
                weight = batch_size
            if weight > 0:
                self.log_metric(f"{step}/loss_{name}", value, batch_size=weight)

    # ------------------------------------------------------------------ steps
    def _step(self, batch, batch_idx, step="train"):
        inputs, targets = batch
        batch_size = inputs["mixture"].shape[0]
        group_batch_size = batch_size
        outputs = self.model(inputs)

        mix = inputs["mixture"][:, 0:1].clone()
        est = outputs["output"].clone()
        gt = targets["target"].clone()
        n_speakers = targets["num_target_speakers"].clone()

        if getattr(self.loss_fn, "requires_query_groups", False):
            queries = int(self.loss_fn.queries_per_mixture)
            if batch_size % queries != 0:
                raise ValueError(f"Batch of {batch_size} is not a whole number of {queries}-query groups")
            group_batch_size = batch_size // queries
            loss_values, aux = self.loss_fn(
                est=est, gt=gt, num_target_speakers=n_speakers, query_radius=inputs["query_radius"]
            )
        else:
            loss_values, aux = self.loss_fn(est=est, gt=gt)
        loss = loss_values.mean()

        est_detached = est.detach().clone()
        with torch.no_grad():
            self.log_metric(f"{step}/loss", loss.item(), batch_size=batch_size, on_step=(step == "train"))
            self._log_loss_terms(step, aux, batch_size, group_batch_size)
            for name, metric in self.metrics.items():
                values = metric(est_detached, gt, mix)
                for i in range(batch_size):
                    if n_speakers[i] > 0:
                        self.log_metric(f"{step}/{name}", values[i].item())
                        self.log_metric(f"{step}/{name}_{int(n_speakers[i])}spk", values[i].item())
            for i in range(batch_size):
                if n_speakers[i] == 0:
                    value = attenuation_db(est_detached[i].unsqueeze(0), mix[i].unsqueeze(0)).item()
                    self.log_metric(f"{step}/empty_attenuation", value)

        sample = {"mixture": mix, "output": est_detached, "target": gt, "n_tgt_speakers": n_speakers}
        return loss, sample

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def _remember_samples(self, store, sample, limit):
        n_speakers = sample["n_tgt_speakers"]
        for i in range(n_speakers.shape[0]):
            bucket = store.setdefault(n_speakers[i].item(), [])
            if len(bucket) < limit:
                bucket.append({k: sample[k][i] for k in sample})

    def training_step(self, batch, batch_idx):
        loss, sample = self._step(batch, batch_idx, step="train")
        self._remember_samples(self.train_samples, sample, 3)
        return loss, sample["n_tgt_speakers"].shape[0]

    def validation_step(self, batch, batch_idx):
        loss, sample = self._step(batch, batch_idx, step="val")
        self._remember_samples(self.val_samples, sample, self.logged_samples_per_count)
        return loss, sample["n_tgt_speakers"].shape[0]

    def reset_grad(self):
        self.optimizer.zero_grad()

    def backprop(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip, error_if_nonfinite=False)
        self.optimizer.step()
        self.optimizer_step += 1
        if self.warmup_steps and self.optimizer_step < self.warmup_steps:
            scale = float(self.optimizer_step + 1) / float(self.warmup_steps)
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = base_lr * scale
        elif self.warmup_steps and self.optimizer_step == self.warmup_steps:
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = base_lr

    # ------------------------------------------------------------ epoch hooks
    def on_epoch_start(self):
        print()
        print("=" * 25, "STARTING EPOCH", self.epoch, "=" * 25)
        print()

    def _ranked_epochs(self, top_k=3):
        ranked = []
        for epoch in range(len(self.metric_values)):
            if self.has_metric_at_epoch(self.monitor, epoch):
                value = self.get_avg_metric_at_epoch(self.monitor, epoch)
                if np.isfinite(value):
                    ranked.append((epoch, value))
        return sorted(ranked, key=lambda x: x[1], reverse=self.monitor_mode == "max")[:top_k]

    def _save_top_k(self, checkpoints_dir, top_k=3):
        top_epochs = {epoch for epoch, _ in self._ranked_epochs(top_k)}
        if self.epoch in top_epochs:
            value = self.get_avg_metric_at_epoch(self.monitor)
            name = self.monitor.replace("/", "_")
            self.dump_state(os.path.join(checkpoints_dir, f"best_epoch_{self.epoch:04d}_{name}_{value:.6f}.pt"))
        for filename in os.listdir(checkpoints_dir):
            if filename.startswith("best_epoch_") and filename.endswith(".pt"):
                try:
                    epoch = int(filename[len("best_epoch_"):len("best_epoch_") + 4])
                except ValueError:
                    continue
                if epoch not in top_epochs:
                    os.remove(os.path.join(checkpoints_dir, filename))

    def on_epoch_end(self, best_path, wandb_run=None):
        assert self.epoch + 1 == len(self.metric_values), "Metrics must be logged for every epoch"
        current = self.get_avg_metric_at_epoch(self.monitor)
        is_best = bool(np.isfinite(current))
        for epoch in range(len(self.metric_values) - 1):
            if not is_best:
                break
            if not self.has_metric_at_epoch(self.monitor, epoch):
                continue
            previous = self.get_avg_metric_at_epoch(self.monitor, epoch)
            if not np.isfinite(previous):
                continue
            if (self.monitor_mode == "max" and current < previous) or (
                self.monitor_mode == "min" and current > previous
            ):
                is_best = False
        if is_best:
            print("Current checkpoint is the best! Saving it...")
            self.dump_state(best_path)
        self._save_top_k(os.path.dirname(best_path), top_k=3)

        print(f"Val loss: {self.get_avg_metric_at_epoch('val/loss'):.4f}")
        for key in ("val/loss_nonempty_rec_loss", "val/si_sdr_i", "val/snr_i", "val/empty_attenuation"):
            if self.has_metric_at_epoch(key):
                print(f"{key}: {self.get_avg_metric_at_epoch(key):.4f}")

        if wandb_run is not None and wandb is not None:
            self._log_wandb(wandb_run)
        self.train_samples.clear()
        self.val_samples.clear()

        if self.scheduler is not None and self.optimizer_step >= self.warmup_steps:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(current)
            else:
                self.scheduler.step()
        self.epoch += 1

    def _log_wandb(self, run):
        step = self.epoch + 1
        run.log({"lr": self.get_current_lr()}, commit=False, step=step)
        for metric in self.metric_values[self.epoch]:
            run.log({metric: self.get_avg_metric_at_epoch(metric)}, commit=False, step=step)
        for split, store in (("train", self.train_samples), ("val", self.val_samples)):
            for count, samples in store.items():
                audio = [
                    wandb.Audio(sample[k].permute(1, 0).cpu().numpy(), sample_rate=self.sample_rate, caption=f"{i}/{k}")
                    for i, sample in enumerate(samples)
                    for k in ("mixture", "target", "output")
                ]
                run.log({f"{split}/audio_{count}spk": audio}, commit=False, step=step)
        run.log({"epoch": self.epoch}, commit=True, step=step)
