"""StraightPCF DistanceModule — ultra-simple MLP, zero reshape/squeeze."""
from typing import Dict, List, Optional

import numpy as np
import jittor as jt
from jittor import nn

from .spec import ModelSpec
from ..data.asset import Asset


class DistanceModule(ModelSpec):

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config
        h = cfg.get('decoder_hidden_dim', 64)
        emb = cfg.get('feat_embedding_dim', 256)

        self.conv1 = nn.Conv1d(3, h, 1, bias=False)
        self.conv2 = nn.Conv1d(h, emb, 1, bias=False)
        self.fc = nn.Linear(emb, 1, bias=False)

    def execute(self, x):
        """x: (B, N, 3) → d_phi: (B, 1)"""
        feat = nn.relu(self.conv1(x.permute(0, 2, 1)))  # (B, 3, N) → (B, h, N)
        feat = nn.relu(self.conv2(feat))                 # (B, h, N) → (B, emb, N)
        feat = jt.max(feat, dim=2)                       # (B, emb, N) → (B, emb)
        out = jt.sigmoid(self.fc(feat))                  # (B, emb) → (B, 1)
        return out  # (B, 1)

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch['pc_noisy'].shape[-2]
        pc_noisy = batch['pc_noisy'].reshape(-1, patch_size, 3)
        pc_mix = batch['pc_mix'].reshape(-1, patch_size, 3)
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)

        dist_mix = jt.sqrt(((pc_clean - pc_mix) ** 2).sum(dim=-1)).mean(dim=-1, keepdims=True)
        dist_noisy = jt.sqrt(((pc_clean - pc_noisy) ** 2).sum(dim=-1)).mean(dim=-1, keepdims=True)
        d_gt = (dist_mix / (dist_noisy + 1e-8)).clamp(0.0, 1.0)  # (B, 1)

        d_pred = self.execute(pc_mix)  # (B, 1)
        loss = ((d_pred - d_gt) ** 2).mean()
        return {"loss": loss}

    def predict_step(self, batch: Dict) -> List[Dict]:
        raise NotImplementedError("Use DistanceModule.forward() directly")

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        res = []
        for b in batch:
            assert b.meta is not None
            res.append({
                "pc_noisy": b.meta['pc_noisy'],
                "pc_clean": b.meta['pc_clean'],
                "pc_mix": b.meta['pc_mix'],
            })
        return res

    def forward(self, x):
        """x: (B, N, 3) → d_phi (B, 1)"""
        return self.execute(x)
