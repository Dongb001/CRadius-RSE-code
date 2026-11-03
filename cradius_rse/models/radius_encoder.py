"""Radius encoders that map a query radius to the FiLM conditioning tensor.

Both encoders return a frequency-dependent representation ``e_r`` of shape
``[B, condition_dim, F]``:

* ``ContinuousRadiusEncoder`` (proposed): the radius is normalized as
  ``r / R_max`` and mapped by a three-layer MLP (Eq. 10). Any radius in
  ``[0, R_max]`` can be queried at inference time.
* ``DiscreteRadiusEncoder`` (ablation, Table 2): the radius is replaced by a
  one-hot code over a fixed set of training radii.
"""

import math

import torch
from torch import nn


def _as_radius_column(radius: torch.Tensor) -> torch.Tensor:
    if radius.ndim == 1:
        radius = radius.unsqueeze(-1)
    if radius.ndim != 2 or radius.shape[-1] != 1:
        raise ValueError(
            f"Query radius must have shape [B, 1] or [B], got {tuple(radius.shape)}"
        )
    if not torch.is_floating_point(radius):
        radius = radius.float()
    if torch.any(~torch.isfinite(radius)):
        raise ValueError("Query radius contains NaN or Inf")
    return radius


class ContinuousRadiusEncoder(nn.Module):
    """``e_r = E(r / R_max)``: Linear-GELU-Linear-GELU-Linear + LayerNorm."""

    def __init__(
        self,
        n_freqs: int,
        max_radius_m: float = 4.0,
        hidden_dim: int = 64,
        condition_dim: int = 4,
        activation: str = "gelu",
    ):
        super().__init__()
        self.n_freqs = int(n_freqs)
        self.condition_dim = int(condition_dim)
        self.max_radius_m = float(max_radius_m)
        if self.n_freqs <= 0 or int(hidden_dim) <= 0 or self.condition_dim <= 0:
            raise ValueError("n_freqs, hidden_dim and condition_dim must be positive")
        if not math.isfinite(self.max_radius_m) or self.max_radius_m <= 0:
            raise ValueError("max_radius_m must be finite and positive")

        activations = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}
        if str(activation).lower() not in activations:
            raise ValueError(f"activation must be one of {sorted(activations)}")
        activation_layer = activations[str(activation).lower()]
        hidden_dim = int(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            activation_layer(),
            nn.Linear(hidden_dim, hidden_dim),
            activation_layer(),
            nn.Linear(hidden_dim, self.condition_dim * self.n_freqs),
        )
        self.output_norm = nn.LayerNorm(self.condition_dim)

    def forward(self, radius: torch.Tensor) -> torch.Tensor:
        radius = _as_radius_column(radius)
        if torch.any(radius < 0) or torch.any(radius > self.max_radius_m):
            raise ValueError(f"Query radius must be in [0, {self.max_radius_m:g}] m")
        encoded = self.mlp(radius / self.max_radius_m)
        encoded = encoded.view(encoded.shape[0], self.n_freqs, self.condition_dim)
        encoded = self.output_norm(encoded)
        return encoded.transpose(1, 2)


class DiscreteRadiusEncoder(nn.Module):
    """One-hot radius code followed by a linear map and LayerNorm."""

    def __init__(self, n_freqs: int, radii_m=(1.0, 2.0, 3.0, 4.0), condition_dim: int = 4):
        super().__init__()
        self.n_freqs = int(n_freqs)
        self.condition_dim = int(condition_dim)
        radii = torch.tensor([float(value) for value in radii_m], dtype=torch.float32)
        if radii.numel() == 0 or torch.any(radii[1:] <= radii[:-1]):
            raise ValueError("radii_m must be a non-empty strictly increasing list")
        self.register_buffer("radii_m", radii, persistent=False)
        self.projection = nn.Linear(radii.numel(), self.n_freqs * self.condition_dim, bias=False)
        self.output_norm = nn.LayerNorm(self.condition_dim)

    def one_hot(self, radius: torch.Tensor) -> torch.Tensor:
        radius = _as_radius_column(radius)
        grid = self.radii_m.to(device=radius.device, dtype=radius.dtype).view(1, -1)
        matches = torch.isclose(radius, grid, rtol=0.0, atol=1e-6)
        if not torch.all(matches.sum(dim=1) == 1):
            raise ValueError(
                f"The discrete radius encoder only accepts radii {self.radii_m.tolist()} m"
            )
        return matches.to(dtype=radius.dtype)

    def forward(self, radius: torch.Tensor) -> torch.Tensor:
        encoded = self.projection(self.one_hot(radius))
        encoded = encoded.view(encoded.shape[0], self.n_freqs, self.condition_dim)
        encoded = self.output_norm(encoded)
        return encoded.transpose(1, 2)
