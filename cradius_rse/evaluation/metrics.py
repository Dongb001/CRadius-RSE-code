"""Evaluation metrics (Sec. 3.3 of the paper).

* Full target (all sources inside the radius): SI-SDR.
* Partial target (some sources inside): SI-SDR improvement over the mixture.
* Empty target (no source inside): attenuation, 10 log10(P(x) / P(y_hat)).
* Radius consistency error (RCE) between adjacent query radii with the same
  target-speaker set: mean |y_hat(r_a) - y_hat(r_b)| / max(RMS(x), eps).
"""

from __future__ import annotations

import math

import torch
from torchmetrics.functional import scale_invariant_signal_distortion_ratio as si_sdr

CATEGORY_METRIC = {"full": "si_sdr_db", "partial": "si_sdri_db", "empty": "attenuation_db"}


def target_category(source_count: int, target_count: int) -> str:
    if target_count == 0:
        return "empty"
    return "full" if target_count == source_count else "partial"


def rms(audio: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(audio.double() ** 2) + 1e-12)


def evaluate_output(output: torch.Tensor, mixture: torch.Tensor, target: torch.Tensor, target_count: int) -> dict:
    """Metrics of one estimate ``[1, T]`` against its target and input mixture."""
    result = {
        "target_speakers": int(target_count),
        "output_rms_db_vs_target": (
            float(20.0 * torch.log10(rms(output) / rms(target)).item()) if target_count else None
        ),
    }
    if target_count:
        error_power = torch.sum((output - target).double() ** 2)
        target_power = torch.sum(target.double() ** 2)
        result["snr_db"] = float(10.0 * torch.log10(target_power / (error_power + 1e-12)).item())
        result["si_sdr_db"] = float(si_sdr(preds=output.unsqueeze(0), target=target.unsqueeze(0)).mean().item())
        if torch.max(torch.abs(mixture - target)).item() <= 1e-6:
            # The clean input already equals the target (full target, no noise):
            # SI-SDRi has an infinite baseline and is undefined.
            result["input_si_sdr_db"] = None
            result["si_sdri_db"] = None
        else:
            result["input_si_sdr_db"] = float(
                si_sdr(preds=mixture.unsqueeze(0), target=target.unsqueeze(0)).mean().item()
            )
            result["si_sdri_db"] = result["si_sdr_db"] - result["input_si_sdr_db"]
        result["attenuation_db"] = None
    else:
        mixture_power = torch.sum(mixture.double() ** 2)
        output_power = torch.sum(output.double() ** 2)
        result.update(
            snr_db=None,
            si_sdr_db=None,
            input_si_sdr_db=None,
            si_sdri_db=None,
            attenuation_db=float(10.0 * torch.log10(mixture_power / (output_power + 1e-12)).item()),
        )
    return result


def consistency_error(output_a: torch.Tensor, output_b: torch.Tensor, mixture: torch.Tensor, eps: float = 1e-8) -> float:
    """||y_hat(r_a) - y_hat(r_b)||_1 / (N_s * max(RMS(x), eps)) for one pair."""
    difference = (output_b.double() - output_a.double()).abs().mean()
    mixture_rms = torch.sqrt(torch.mean(mixture.double() ** 2))
    return float(difference / max(float(mixture_rms), eps))


def target_set(distances, radius: float) -> tuple[int, ...]:
    return tuple(index for index, distance in enumerate(distances) if 0 <= distance < radius)


def finite(value) -> bool:
    return value is not None and math.isfinite(float(value))
