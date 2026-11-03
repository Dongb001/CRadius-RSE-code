"""Small shared helpers."""

import importlib
import json
import random

import numpy as np
import torch


def import_attr(import_path: str):
    """Import ``package.module.Attribute`` from a dotted path."""
    module, attr = import_path.rsplit(".", 1)
    return getattr(importlib.import_module(module), attr)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_model(config: dict) -> torch.nn.Module:
    """Instantiate the network described by an experiment config."""
    return import_attr(config["model"])(**config.get("model_args", {}))


def load_model_weights(model: torch.nn.Module, checkpoint_path, strict: bool = True):
    """Load either a full training state (``{"model": ...}``) or a bare state dict."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(weights, strict=strict)
    epoch = state.get("current_epoch") if isinstance(state, dict) else None
    return model, epoch
