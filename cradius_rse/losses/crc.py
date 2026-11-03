"""Cross-radius consistency (CRC) learning (Sec. 2.3-2.4 of the paper).

For every training mixture the data loader provides Q strictly increasing
query radii r_1 < ... < r_Q, flattened contiguously in the batch. Because the
target-speaker sets are nested, S(r_q) = S(r_{q+1}) holds exactly when the
number of target speakers does not change between adjacent radii. For every
adjacent pair with an unchanged, non-empty target set whose common-target RMS
exceeds tau (Eq. 12), CRC minimizes

    || y_hat(r_a) - y_hat(r_b) ||_1 / (N_s * max(RMS(y(r_a)), eps))     (Eq. 13)

averaged over valid pairs (L_con = 0 if there is none). The total loss is

    L = L_ext + lambda_con * L_con                                         (Eq. 18)

where L_ext is the mean of the per-radius losses of ``RegionalExtractionLoss``.
"""

import torch
from torch import nn

from .extraction import RegionalExtractionLoss


class CRCObjective(nn.Module):
    """Extraction loss plus cross-radius consistency."""

    # The training module must pass complete, ordered query groups.
    requires_query_groups = True

    def __init__(
        self,
        lambda_con: float = 0.1,
        queries_per_mixture: int = 4,
        rms_threshold: float = 1e-4,
        zero_target_weight: float = 100.0,
        eps: float = 1e-8,
        zero_target_threshold: float = 1e-7,
        max_loss: float = 40.0,
        straight_through_clip: bool = False,
    ) -> None:
        super().__init__()
        self.lambda_con = float(lambda_con)
        self.queries_per_mixture = int(queries_per_mixture)
        self.rms_threshold = float(rms_threshold)
        self.eps = float(eps)
        if self.lambda_con < 0:
            raise ValueError("lambda_con must be non-negative")
        if self.queries_per_mixture < 2:
            raise ValueError("queries_per_mixture must be at least two")
        if self.rms_threshold < 0:
            raise ValueError("rms_threshold must be non-negative")
        self.extraction_loss = RegionalExtractionLoss(
            zero_target_weight=zero_target_weight,
            eps=eps,
            zero_target_threshold=zero_target_threshold,
            max_loss=max_loss,
            straight_through_clip=straight_through_clip,
        )

    def consistency(self, est_grouped, gt_grouped, counts, aux):
        """L_con for grouped tensors ``[M, Q, C, T]`` and target counts ``[M, Q]``."""
        # Nested target sets imply non-decreasing counts; skip a pair instead of
        # failing a long run if a degenerate draw ever violates this.
        nonmonotonic = counts[:, 1:] < counts[:, :-1]
        if torch.any(nonmonotonic):
            aux["crc_skipped_nonmonotonic_pairs"] = nonmonotonic.sum().detach()

        pair_losses = []
        valid_counts = []
        for pair in range(est_grouped.shape[1] - 1):
            delta = est_grouped[:, pair + 1] - est_grouped[:, pair]
            unchanged = ~(counts[:, pair + 1] > counts[:, pair]) & ~nonmonotonic[:, pair]
            # For an unchanged pair the smaller-radius target is the common target.
            common_target = gt_grouped[:, pair]
            common_rms = torch.sqrt(
                torch.mean(common_target.double() ** 2, dim=(1, 2))
            ).to(dtype=est_grouped.dtype)
            valid = unchanged & (counts[:, pair] > 0) & (common_rms > self.rms_threshold)
            if torch.any(valid):
                values = (
                    delta[valid].abs().reshape(-1, delta.shape[1] * delta.shape[2]).mean(dim=1)
                    / common_rms[valid].clamp_min(self.eps)
                )
                pair_loss = values.mean()
                pair_losses.append(pair_loss)
            else:
                pair_loss = delta.new_zeros(())
            aux[f"crc_q{pair}to{pair + 1}"] = pair_loss.detach()
            aux[f"crc_q{pair}to{pair + 1}_count"] = valid.sum().detach()
            valid_counts.append(valid.sum())

        if pair_losses:
            # Average inside each adjacent slot, then over slots with valid pairs.
            loss = torch.stack(pair_losses).mean()
        else:
            loss = est_grouped.new_zeros(())
        aux["crc_valid_pairs"] = torch.stack(valid_counts).sum().detach()
        return loss

    def forward(self, est, gt, num_target_speakers, query_radius, **kwargs):
        if est.shape != gt.shape or est.ndim != 3:
            raise ValueError(f"Expected matching waveforms [N, C, T], got {est.shape}/{gt.shape}")
        queries = self.queries_per_mixture
        if est.shape[0] % queries != 0:
            raise ValueError(
                f"Batch of {est.shape[0]} items is not a whole number of {queries}-query groups"
            )
        if num_target_speakers.numel() != est.shape[0] or query_radius.shape[0] != est.shape[0]:
            raise ValueError("Target counts and query radii must align with the flattened batch")

        mixtures = est.shape[0] // queries
        radius = query_radius.reshape(mixtures, queries, -1)
        if not torch.all(radius[:, 1:] > radius[:, :-1]):
            raise ValueError("Query radii must be strictly increasing within each mixture")
        est_grouped = est.reshape(mixtures, queries, *est.shape[1:])
        gt_grouped = gt.reshape(mixtures, queries, *gt.shape[1:])
        counts = num_target_speakers.reshape(mixtures, queries)

        per_query, ext_aux = self.extraction_loss(est=est, gt=gt)
        extraction = per_query.mean()
        aux = dict(ext_aux)
        aux["extraction"] = extraction.detach()
        for query in range(queries):
            aux[f"group_extraction_q{query}"] = per_query[query::queries].mean().detach()
            aux[f"group_radius_mean_q{query}"] = radius[:, query].mean().detach()

        consistency = self.consistency(est_grouped, gt_grouped, counts, aux)
        total = extraction + self.lambda_con * consistency
        aux["crc"] = consistency.detach()
        aux["crc_weighted"] = (self.lambda_con * consistency).detach()
        aux["total"] = total.detach()
        return total, aux
