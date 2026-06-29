"""DRD (Deterministic Residual Diffusion) denoising model for Jittor.

Based on TVCG 2026 "Deterministic Point Cloud Diffusion for Denoising".
Hybrid approach: reuses DGCNN FeatureExtraction from StraightPCF baseline,
adds time embedding + iterative reverse diffusion + alpha schedule.
"""

import jittor as jt
import numpy as np
from jittor import nn

from .time_emb import TimeEmbedding
from .vm import VelocityModule, get_random_indices


class DrdDecoder(nn.Module):
    """Deeper decoder with an extra hidden layer for time-conditioned residual prediction."""

    def __init__(self, z_dim=256, hidden_size=64):
        super().__init__()
        self.lin_1 = nn.Linear(z_dim, z_dim)
        self.bn_1 = nn.BatchNorm1d(z_dim)

        self.lin_2 = nn.Linear(z_dim, hidden_size)
        self.bn_2 = nn.BatchNorm1d(hidden_size)

        self.lin_3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn_3 = nn.BatchNorm1d(hidden_size // 2)

        self.lin_4 = nn.Linear(hidden_size // 2, 3)

        self.actvn = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def execute(self, c):
        """c: (B*N, F)"""
        net = self.lin_1(c)
        net = self.bn_1(net)
        net = self.actvn(net)
        net = self.dropout(net)

        net = self.lin_2(net)
        net = self.bn_2(net)
        net = self.actvn(net)
        net = self.dropout(net)

        net = self.lin_3(net)
        net = self.bn_3(net)
        net = self.actvn(net)

        net = self.lin_4(net)
        return net


def gen_alphas_np(T=30, schedule="decreased"):
    """Generate diffusion step coefficients (numpy, pre-computed).

    decreased: large steps first, small steps last. sum = 1.
    """
    if schedule == "decreased":
        x = np.linspace(T, 1, T, dtype=np.float32)
    else:
        x = np.ones(T, dtype=np.float32) / T
    scale = 0.5 * T * (T + 1)
    alphas = x / scale
    alpha_cumsum = np.cumsum(alphas).clip(0.0, 1.0)
    return alphas, alpha_cumsum


class DrdDenoiseNet(VelocityModule):
    """DRD-style denoising with DGCNN encoder + iterative residual diffusion."""

    def __init__(self, model_config, transform_config):
        # VelocityModule.__init__ creates encoder (DGCNN) + old Decoder
        # We replace the decoder and add time embedding
        super().__init__(model_config, transform_config)

        cfg = self.model_config
        F = self.encoder.embedding_dim

        # Replace decoder with deeper time-conditioned version
        self.decoder = DrdDecoder(
            z_dim=F,
            hidden_size=cfg.get('decoder_hidden_dim', 128),
        )

        # Time embedding: injects timestep information into features
        self.time_emb = TimeEmbedding(d_out=F)

        # Diffusion schedule (kept as numpy for safe indexing, jt.Var for compute)
        T = cfg.get('T', 30)
        schedule = cfg.get('schedule', 'decreased')
        alphas, cumsum = gen_alphas_np(T, schedule)
        self.T = T
        self._alphas_np = alphas
        self._cumsum_np = cumsum
        self.alphas = jt.array(alphas).stop_grad()
        self.alpha_cumsum = jt.array(cumsum).stop_grad()

    # ---------- training ----------

    def get_supervised_loss(self, pc_noisy, pc_mix, pc_clean):
        """DRD training loss: random timestep, forward diffuse, predict residual.

        pc_noisy: (B, N, 3) — fully noisy
        pc_clean: (B, N, 3) — ground truth clean
        pc_mix:   unused (kept for API compatibility)
        """
        B, N_noisy, d = pc_noisy.shape

        # Random point subsampling (same as baseline)
        pnt_idx = get_random_indices(N_noisy, self.num_train_points)
        pc_noisy = pc_noisy[:, pnt_idx, :]
        pc_clean = pc_clean[:, pnt_idx, :]
        B, N, _ = pc_noisy.shape

        # 1. Random timestep per batch element
        t_np = np.random.randint(0, self.T, size=(B,))
        t = jt.array(t_np).int32()

        # 2. True residual (noise → clean direction)
        p_res = pc_noisy - pc_clean  # (B, N, 3)

        # 3. Forward diffusion: P_t = P_clean + alpha_cumsum[t] * P_res
        alpha_t = jt.array(self._cumsum_np[t_np]).reshape(B, 1, 1)  # (B, 1, 1)
        p_t = pc_clean + alpha_t * p_res  # (B, N, 3)

        # 4. Feature extraction + time injection
        feat = self.encoder(p_t)  # (B, N, F)
        F_dim = feat.shape[2]
        scale, shift = self.time_emb(t, T=self.T)  # (B, F), (B, F)

        # Inject time before decoder
        feat = feat * (scale.unsqueeze(1) + 1.0) + shift.unsqueeze(1)  # (B, N, F)

        # 5. Predict residual
        pred_res = self.decoder(
            feat.reshape(B * N, F_dim)
        ).reshape(B, N, d)

        # 6. MSE loss
        loss = ((pred_res - p_res) ** 2.0).sum(dim=-1).mean()

        return loss

    # ---------- inference ----------

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps=None):
        """Override: use DRD reverse diffusion instead of Langevin dynamics.

        pcl_noisy: (B, N, 3)
        """
        if num_steps is None:
            num_steps = self.T
        return self._reverse_diffusion(pcl_noisy, num_steps)

    def _reverse_diffusion(self, pcl_noisy, num_steps):
        """Iteratively remove predicted residuals over `num_steps` steps.

        pcl_noisy: (B, N, 3)
        Returns: (p_denoised, None)  — second element is for API compat
        """
        B, N, d = pcl_noisy.shape
        assert num_steps <= self.T, f"num_steps {num_steps} > T {self.T}"

        p_t = pcl_noisy.clone()
        with jt.no_grad():
            for step_idx in range(num_steps - 1, -1, -1):
                # Time embedding for current step
                t = jt.array([step_idx] * B).int32()
                scale, shift = self.time_emb(t, T=self.T)  # (B, F), (B, F)

                # Feature extraction + time injection
                feat = self.encoder(p_t)  # (B, N, F)
                F_dim = feat.shape[2]
                feat = feat * (scale.unsqueeze(1) + 1.0) + shift.unsqueeze(1)

                # Predict residual
                pred_res = self.decoder(
                    feat.reshape(B * N, F_dim)
                ).reshape(B, N, d)

                # Remove partial residual (safe numpy indexing)
                if step_idx > 0:
                    step_val = self._cumsum_np[step_idx] - self._cumsum_np[step_idx - 1]
                else:
                    step_val = self._cumsum_np[0]
                step = jt.array(step_val).reshape(1, 1, 1)

                p_t = p_t - step * pred_res

        return p_t, None

    # training_step, predict_step, process_fn inherited from VelocityModule
