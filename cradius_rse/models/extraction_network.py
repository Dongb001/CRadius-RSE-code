"""Continuous-radius extraction network (Fig. 2 of the paper).

mixture x --STFT--> [real, imag, log-magnitude] (magnitude-normalized)
  -> Conv2D + LayerNorm -> H(0)
  -> TF block 0 -> FiLM 0 -> TF block 1 -> ... -> FiLM L-1 -> TF block L
  -> ConvTranspose2D -> complex spectrum -> iSTFT -> estimate y_r

The query radius r is encoded by the radius encoder E into e_r, which
conditions the transitions between adjacent TF blocks through FiLM (Eq. 11).
"""

import torch
import torch.nn.functional as F
from asteroid_filterbanks import make_enc_dec
from torch import nn

from .normalization import recursive_magnitude_normalization
from .radius_encoder import ContinuousRadiusEncoder, DiscreteRadiusEncoder
from .tf_block import FiLM, LayerNormPermuted, TFBlock


def log_magnitude(real: torch.Tensor, imag: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    magnitude = torch.sqrt(torch.square(real) + torch.square(imag) + eps)
    return torch.log(magnitude + eps)


class ContinuousRadiusExtractionNetwork(nn.Module):
    """Monaural regional speech extraction conditioned on a query radius.

    Args:
        radius_encoding: ``"continuous"`` (proposed) or ``"discrete"`` (one-hot
            radius codes, ablation in Table 2).
        max_radius_m: R_max used to normalize the continuous radius.
        discrete_radii_m: radii represented by the one-hot codes.
        n_fft / hop_length: STFT size and hop (192 / 128 samples at 16 kHz).
        num_tf_blocks: number of TF blocks (L + 1 = 6).
        feature_dim: backbone feature dimension C.
        lstm_hidden_dim: hidden size of each BiLSTM direction.
        condition_dim: per-frequency size of the radius representation e_r.
        radius_encoder_hidden_dim: hidden size of the continuous radius MLP.
        norm_window_frames / norm_eps: recursive magnitude normalization.
    """

    def __init__(
        self,
        radius_encoding: str = "continuous",
        max_radius_m: float = 4.0,
        discrete_radii_m=(1.0, 2.0, 3.0, 4.0),
        n_fft: int = 192,
        hop_length: int = 128,
        num_tf_blocks: int = 6,
        feature_dim: int = 32,
        lstm_hidden_dim: int = 64,
        condition_dim: int = 4,
        radius_encoder_hidden_dim: int = 64,
        radius_encoder_activation: str = "gelu",
        norm_window_frames: int = 250,
        norm_eps: float = 1e-8,
    ):
        super().__init__()
        if n_fft % 2 != 0 or not 0 < hop_length <= n_fft:
            raise ValueError("n_fft must be even and 0 < hop_length <= n_fft")
        if num_tf_blocks < 1:
            raise ValueError("num_tf_blocks must be positive")
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_freqs = self.n_fft // 2 + 1
        self.norm_window_frames = int(norm_window_frames)
        self.norm_eps = float(norm_eps)
        if self.norm_window_frames <= 0 or self.norm_eps <= 0:
            raise ValueError("norm_window_frames and norm_eps must be positive")

        self.encoder, self.decoder = make_enc_dec(
            "stft",
            n_filters=self.n_fft,
            kernel_size=self.n_fft,
            stride=self.hop_length,
            window_type="hann",
        )

        # Normalized real, imaginary and log-magnitude planes -> H(0).
        self.input_conv = nn.Sequential(
            nn.Conv2d(3, feature_dim, (3, 3), padding=(1, 1)),
            LayerNormPermuted(feature_dim),
        )

        radius_encoding = str(radius_encoding).lower()
        if radius_encoding == "continuous":
            self.radius_encoder = ContinuousRadiusEncoder(
                n_freqs=self.n_freqs,
                max_radius_m=max_radius_m,
                hidden_dim=radius_encoder_hidden_dim,
                condition_dim=condition_dim,
                activation=radius_encoder_activation,
            )
        elif radius_encoding == "discrete":
            self.radius_encoder = DiscreteRadiusEncoder(
                n_freqs=self.n_freqs, radii_m=discrete_radii_m, condition_dim=condition_dim
            )
        else:
            raise ValueError("radius_encoding must be 'continuous' or 'discrete'")
        self.radius_encoding = radius_encoding

        # FiLM only between adjacent TF blocks: num_tf_blocks - 1 modules.
        # (Modules are created block by block so that a given seed yields the
        # same initialization as the original training runs.)
        self.tf_blocks = nn.ModuleList()
        self.films = nn.ModuleList()
        for index in range(num_tf_blocks):
            self.tf_blocks.append(TFBlock(feature_dim, lstm_hidden_dim))
            if index > 0:
                self.films.append(FiLM(condition_dim, feature_dim))
        self.output_deconv = nn.ConvTranspose2d(feature_dim, 2, (3, 3), padding=(1, 1))

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        remainder = x.shape[-1] % self.hop_length
        if remainder:
            x = F.pad(x, (0, self.hop_length - remainder))
        return F.pad(x, (0, self.n_fft - self.hop_length))

    def _extract_padded(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        n_samples = x.shape[-1]
        spec = self.encoder(x)
        if spec.dim() == 3:
            spec = spec.unsqueeze(1)
        real = spec[..., : self.n_freqs, :]
        imag = spec[..., self.n_freqs :, :]

        magnitude = torch.sqrt(torch.square(real) + torch.square(imag))
        reference = recursive_magnitude_normalization(
            magnitude[:, 0], self.norm_window_frames
        ).unsqueeze(2)  # [B, 1, 1, N]
        safe_reference = reference.clamp_min(self.norm_eps)
        real = real / safe_reference
        imag = imag / safe_reference

        features = torch.cat((real, imag, log_magnitude(real, imag)), dim=1)  # [B, 3, F, N]
        features = features.transpose(2, 3)  # [B, 3, N, F]
        n_batch, _, n_frames, n_freqs = features.shape
        hidden = self.input_conv(features)

        for index, block in enumerate(self.tf_blocks):
            if index > 0:
                hidden = hidden.transpose(2, 3)
                hidden = self.films[index - 1](hidden, condition)
                hidden = hidden.transpose(2, 3)
            hidden = block(hidden)

        out = self.output_deconv(hidden)  # [B, 2, N, F]
        out = out.view(n_batch, 1, 2, n_frames, n_freqs).transpose(3, 4)
        out = torch.cat([out[:, :, 0], out[:, :, 1]], dim=2)  # [B, 1, 2F, N]
        # The network operates on normalized spectra; restore the input scale.
        out = out * reference
        return self.decoder(out)[..., :n_samples]

    def extract(self, mixture: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        """mixture: ``[B, T]`` or ``[B, 1, T]``; radius: ``[B, 1]`` in metres."""
        if mixture.dim() == 2:
            mixture = mixture.unsqueeze(1)
        if mixture.dim() != 3 or mixture.shape[1] != 1:
            raise ValueError(f"Expected mono input [B, T] or [B, 1, T], got {tuple(mixture.shape)}")
        length = mixture.shape[-1]
        condition = self.radius_encoder(radius)
        estimate = self._extract_padded(self._pad(mixture), condition)
        return estimate[..., :length]

    def forward(self, inputs: dict) -> dict:
        return {"output": self.extract(inputs["mixture"], inputs["query_radius"])}
