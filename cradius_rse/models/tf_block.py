"""TF block and FiLM conditioning of the continuous-radius extraction network.

The TF block follows the offline TF-GridNet block (Wang et al., 2023) without
the attention branch: a bidirectional LSTM along frequency followed by a
bidirectional LSTM along time, each with layer normalization, a linear
projection and a residual connection (Eqs. 8-9 of the paper).
The implementation is adapted from the TF-GridNet code in ESPnet (Apache-2.0).
"""

import torch
from torch import nn


class LayerNormPermuted(nn.LayerNorm):
    """Layer normalization over the channel axis of a ``[B, C, T, F]`` tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)  # [B, T, F, C]
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)  # [B, C, T, F]


class TFBlock(nn.Module):
    """Frequency-path and time-path bidirectional sequence modelling."""

    def __init__(self, feature_dim: int, lstm_hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        # Frequency path: U = H + P^f(BiLSTM^f(LN^f(H)))
        self.freq_norm = nn.LayerNorm(feature_dim, eps=eps)
        self.freq_lstm = nn.LSTM(
            feature_dim, lstm_hidden_dim, 1, batch_first=True, bidirectional=True
        )
        self.freq_proj = nn.Linear(lstm_hidden_dim * 2, feature_dim)
        # Time path: V = U + P^t(BiLSTM^t(LN^t(U)))
        self.time_norm = nn.LayerNorm(feature_dim, eps=eps)
        self.time_lstm = nn.LSTM(
            feature_dim, lstm_hidden_dim, 1, batch_first=True, bidirectional=True
        )
        self.time_proj = nn.Linear(lstm_hidden_dim * 2, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: ``[B, C, T, F]`` -> ``[B, C, T, F]``."""
        batch, channels, frames, freqs = x.shape
        residual = x.permute(0, 2, 3, 1)  # [B, T, F, C]

        freq = self.freq_norm(residual)
        freq = freq.reshape(batch * frames, freqs, channels)
        self.freq_lstm.flatten_parameters()
        freq, _ = self.freq_lstm(freq)
        freq = self.freq_proj(freq)
        freq = freq.reshape(batch, frames, freqs, channels) + residual

        time = self.time_norm(freq)
        time = time.transpose(1, 2).reshape(batch * freqs, frames, channels)
        self.time_lstm.flatten_parameters()
        time, _ = self.time_lstm(time)
        time = self.time_proj(time)
        time = time.view(batch, freqs, frames, channels).transpose(1, 2)
        out = time + freq
        return out.permute(0, 3, 1, 2)


class FiLM(nn.Module):
    """Feature-wise linear modulation, H = Gamma(e_r) * V + B(e_r) (Eq. 11).

    The radius representation ``e_r`` has shape ``[B, condition_dim, F]``; the
    scale and shift vary over channels and frequency and are broadcast over time.
    """

    def __init__(self, condition_dim: int, feature_dim: int):
        super().__init__()
        self.scale = nn.Conv1d(condition_dim, feature_dim, 1)
        self.shift = nn.Conv1d(condition_dim, feature_dim, 1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """x: ``[B, C, F, T]``, condition: ``[B, condition_dim, F]``."""
        scale = self.scale(condition).unsqueeze(3)  # [B, C, F, 1]
        shift = self.shift(condition).unsqueeze(3)  # [B, C, F, 1]
        return x * scale + shift
