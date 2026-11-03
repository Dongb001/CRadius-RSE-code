"""Query a trained radius-conditioned model on a batch of mixtures."""

from __future__ import annotations

import torch

from ..utils import build_model, load_json, load_model_weights


class RadiusConditionedSystem:
    """A trained model queried with a continuous (or discrete) radius.

    Estimates are clipped to [-1, 1].
    """

    def __init__(self, config_path, checkpoint_path, device):
        config = load_json(config_path)
        model, self.epoch = load_model_weights(build_model(config), checkpoint_path)
        self.model = model.to(device).eval()
        self.config = config
        self.device = device
        self.provenance = {"config": str(config_path), "checkpoint": str(checkpoint_path), "epoch": self.epoch}

    @torch.inference_mode()
    def predict(self, mixtures: torch.Tensor, radii) -> list[torch.Tensor]:
        """mixtures: ``[B, 1, T]``; returns one ``[B, 1, T]`` estimate per radius."""
        outputs = []
        x = mixtures.to(self.device)
        for radius in radii:
            query = torch.full((x.shape[0], 1), float(radius), dtype=torch.float32, device=self.device)
            y = self.model({"mixture": x, "query_radius": query})["output"].detach().cpu()
            outputs.append(y[..., : mixtures.shape[-1]].clamp(-1.0, 1.0))
        return outputs
