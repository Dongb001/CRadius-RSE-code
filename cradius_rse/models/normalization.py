"""Recursive magnitude normalization of the input spectrum."""

import torch
from torch import Tensor


def recursive_magnitude_normalization(magnitude: Tensor, sliding_window_len: int = 250) -> Tensor:
    """Frame-wise recursive magnitude reference (the scheme used by CleanMel).

    Args:
        magnitude: Magnitude spectrogram with shape ``[B, F, T]``.
        sliding_window_len: Effective forgetting window in frames.

    Returns:
        A detached magnitude reference with shape ``[B, 1, T]``.
    """
    if magnitude.ndim != 3:
        raise ValueError(f"Expected magnitude spectrogram [B, F, T], got {magnitude.shape}")
    if int(sliding_window_len) <= 0:
        raise ValueError("sliding_window_len must be positive")

    _, _, frames = magnitude.shape
    if frames == 0:
        return magnitude.mean(dim=1, keepdim=True).detach()

    alpha = (int(sliding_window_len) - 1) / (int(sliding_window_len) + 1)
    mean_mag = magnitude.mean(dim=1, keepdim=True).detach()

    # The first two recursion coefficients are special cases (alpha(0) = -1,
    # alpha(1) = 0); the remaining recurrence is evaluated with one cumulative
    # product/sum on the device.
    first = 2.0 * mean_mag[..., :1]
    if frames == 1:
        return first

    second = mean_mag[..., 1:2]
    if frames == 2:
        return torch.cat((first, second), dim=-1)

    indices = torch.arange(2, frames, device=magnitude.device, dtype=magnitude.dtype)
    coefficients = torch.minimum(
        (indices - 1.0) / (indices + 1.0),
        torch.full_like(indices, alpha),
    )
    products = torch.cumprod(coefficients, dim=0)
    weighted = (1.0 - coefficients).view(1, 1, -1) * mean_mag[..., 2:]
    products_view = products.view(1, 1, -1)
    rest = products_view * torch.cumsum(weighted / products_view, dim=-1)
    rest = rest + products_view * second
    return torch.cat((first, second, rest), dim=-1)
