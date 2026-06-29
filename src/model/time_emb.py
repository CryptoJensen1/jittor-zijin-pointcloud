import jittor as jt
import numpy as np
from jittor import nn


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps.

    Input: t (B,) int32 in [0, T-1]
    Output: scale (B, d_out), shift (B, d_out)
    """

    def __init__(self, d_out=256):
        super().__init__()
        L = 10
        self.freq_bands = (2.0 ** np.arange(L)) * np.pi  # [pi, 2pi, 4pi, ..., 512pi]
        self.freq_bands = jt.array(self.freq_bands.astype(np.float32)).stop_grad()
        self.mlp = nn.Sequential(
            nn.Linear(2 * L, 64),
            nn.Sigmoid(),
            nn.Linear(64, d_out * 2),
        )

    def execute(self, t, T=30):
        """t: (B,) int32 time steps"""
        t_norm = (t.float() - T / 2.0) / (T / 2.0)  # [-1, 1]
        t_norm = t_norm.reshape(-1, 1)  # (B, 1)
        emb = []
        for freq in self.freq_bands:
            freq_val = freq.item()
            emb.append(jt.sin(freq_val * t_norm))
            emb.append(jt.cos(freq_val * t_norm))
        emb = jt.concat(emb, dim=1)  # (B, 20)
        out = self.mlp(emb)  # (B, d_out*2)
        scale, shift = out.chunk(2, dim=1)
        return scale, shift
