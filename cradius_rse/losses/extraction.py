"""Per-radius extraction loss L_q (Eqs. 15-16 of the paper).

For a non-empty target region the normalized reconstruction loss

    l_rec(y_hat, y) = 10 log10(1 + MSE(y_hat - y) / (MSE(y) + eps))

is used. For an empty region (the target is silent) the output magnitude is
penalized with the zero-target suppression weight alpha:

    L_q = alpha * mean(|y_hat|).
"""

import torch
from torch import nn


class RegionalExtractionLoss(nn.Module):
    """Returns per-item losses ``[N]`` and a dictionary of logging values.

    Args:
        zero_target_weight: alpha in Eq. (16).
        eps: epsilon of the reconstruction loss.
        zero_target_threshold: targets whose peak is below this value are empty.
        max_loss: the reconstruction loss is capped at this value (dB).
        straight_through_clip: keep the gradient of the uncapped loss when capped.
    """

    def __init__(
        self,
        zero_target_weight: float = 100.0,
        eps: float = 1e-8,
        zero_target_threshold: float = 1e-7,
        max_loss: float = 40.0,
        straight_through_clip: bool = False,
    ) -> None:
        super().__init__()
        self.zero_target_weight = float(zero_target_weight)
        self.eps = float(eps)
        self.zero_target_threshold = float(zero_target_threshold)
        self.max_loss = float(max_loss)
        self.straight_through_clip = bool(straight_through_clip)

    def reconstruction_loss(self, est: torch.Tensor, gt: torch.Tensor, return_unclipped=False):
        """Bounded, non-negative l_rec computed independently per item."""
        batch_size = est.shape[0]
        est_flat = est.reshape(batch_size, -1).double()
        gt_flat = gt.reshape(batch_size, -1).double()
        error_power = torch.mean((est_flat - gt_flat) ** 2, dim=1)
        target_power = torch.mean(gt_flat**2, dim=1)
        raw_loss = 10.0 * torch.log10(1.0 + error_power / (target_power + self.eps))
        raw_loss = torch.nan_to_num(
            raw_loss, nan=self.max_loss, posinf=self.max_loss, neginf=0.0
        ).clamp_min(0.0)
        clipped_loss = raw_loss.clamp(max=self.max_loss)
        if self.straight_through_clip:
            clipped_loss = raw_loss + (clipped_loss - raw_loss).detach()
        loss = clipped_loss.to(dtype=est.dtype)
        if return_unclipped:
            return loss, raw_loss.to(dtype=est.dtype)
        return loss

    def forward(self, est: torch.Tensor, gt: torch.Tensor, **kwargs):
        if est.shape != gt.shape:
            raise ValueError(f"Estimate/target shapes differ: {est.shape} vs {gt.shape}")

        loss = torch.zeros(est.shape[0], device=est.device, dtype=est.dtype)
        target_peak = torch.amax(torch.abs(gt).reshape(gt.shape[0], -1), dim=1)
        empty = target_peak <= self.zero_target_threshold
        aux = {"zero_target_ratio": empty.float().mean().detach()}

        if torch.any(empty):
            empty_count = int(empty.sum().item())
            empty_l1 = torch.mean(torch.abs(est[empty]).reshape(empty_count, -1), dim=1)
            empty_weighted = empty_l1 * self.zero_target_weight
            loss[empty] = empty_weighted
            aux["empty_l1"] = empty_l1.mean().detach()
            aux["empty_weighted_l1"] = empty_weighted.mean().detach()
            aux["empty_count"] = empty.sum().detach()

        if torch.any(~empty):
            rec, rec_unclipped = self.reconstruction_loss(
                est[~empty], gt[~empty], return_unclipped=True
            )
            loss[~empty] = rec
            aux["nonempty_rec_loss"] = rec.mean().detach()
            aux["nonempty_rec_unclipped"] = rec_unclipped.mean().detach()
            aux["nonempty_rec_saturation_ratio"] = (
                (rec_unclipped >= self.max_loss).float().mean().detach()
            )
            aux["nonempty_count"] = (~empty).sum().detach()

        loss = torch.nan_to_num(
            loss, nan=self.max_loss, posinf=self.max_loss, neginf=0.0
        ).clamp_min(0.0)
        return loss, aux
