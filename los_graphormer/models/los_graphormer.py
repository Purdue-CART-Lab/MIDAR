import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden)
        self.w2 = nn.Linear(d_model, hidden)
        self.w3 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.silu(self.w1(x))
        b = self.w2(x)
        y = self.w3(a * b)
        return self.drop(y)


class RelGeomBias(nn.Module):
    def __init__(self, n_heads: int, use_rbf: bool = True, rbf_k: int = 8,
                 rbf_gamma: float = 40.0, hidden: int = 96):
        super().__init__()
        self.n_heads = n_heads
        self.use_rbf = use_rbf
        self.rbf_k = rbf_k

        mu = torch.linspace(0.0, 1.0, rbf_k)
        self.register_buffer("rbf_mu", mu, persistent=False)
        self.log_rbf_gamma = nn.Parameter(torch.log(torch.tensor(rbf_gamma)))

        base_dim = 8
        in_dim = base_dim + (rbf_k if use_rbf else 0)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_heads),
        )

    @staticmethod
    def _ang_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        d = a[..., None] - b[..., None, :]
        return (d + math.pi) % (2 * math.pi) - math.pi

    def _rbf(self, r01: torch.Tensor) -> torch.Tensor:
        gamma = self.log_rbf_gamma.exp()
        return torch.exp(-gamma * (r01[..., None] - self.rbf_mu) ** 2)

    @staticmethod
    def _local_frame_deltas(pos_batch: torch.Tensor,
                            yaw_batch: torch.Tensor,
                            eps: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, _ = pos_batch.shape
        pi = pos_batch[:, :, None, :]
        pj = pos_batch[:, None, :, :]
        d = pj - pi
        dx, dy = d[..., 0], d[..., 1]

        yi = yaw_batch
        c = torch.cos(yi)[:, :, None]
        s = torch.sin(yi)[:, :, None]

        dx_l = c * dx + s * dy
        dy_l = -s * dx + c * dy

        r = torch.sqrt(dx_l ** 2 + dy_l ** 2 + eps)
        return dx_l, dy_l, r

    @classmethod
    def compute_geom_feat(cls,
                          pos_batch: torch.Tensor,
                          yaw_batch: torch.Tensor,
                          z_batch: torch.Tensor,
                          use_rbf: bool = True,
                          rbf_k: int = 8,
                          rbf_mu: Optional[torch.Tensor] = None,
                          log_rbf_gamma: Optional[torch.Tensor] = None,
                          scale_xy: float = 60.0,
                          scale_z: float = 5.0,
                          eps: float = 1e-4) -> Dict:
        B, S, _ = pos_batch.shape

        dx_l, dy_l, r = cls._local_frame_deltas(pos_batch, yaw_batch, eps=eps)
        r_scaled = (r / scale_xy).clamp(0.0, 2.0)
        r01 = (r_scaled / 2.0).clamp(0.0, 1.0)

        inv_r = (1.0 / (r_scaled + 1e-3)).clamp(max=10.0)
        d_yaw = cls._ang_diff(yaw_batch, yaw_batch)
        d_yaw_cos, d_yaw_sin = torch.cos(d_yaw), torch.sin(d_yaw)

        zi = z_batch[:, :, None]
        zj = z_batch[:, None, :]
        dz = (zj - zi) / scale_z
        adz = dz.abs()

        base_feat = torch.stack([
            dx_l / scale_xy,
            dy_l / scale_xy,
            r_scaled,
            inv_r,
            d_yaw_cos, d_yaw_sin,
            dz, adz,
        ], dim=-1)

        out = {
            "feat_base": base_feat,
            "r01": r01,
            "S": S,
            "rbf_mu": rbf_mu,
            "log_rbf_gamma": log_rbf_gamma,
            "use_rbf": use_rbf,
            "rbf_k": rbf_k,
        }
        return out

    def forward(self,
                pos_batch: Optional[torch.Tensor] = None,
                yaw_batch: Optional[torch.Tensor] = None,
                z_batch: Optional[torch.Tensor] = None,
                *,
                geom_cache: Optional[Dict] = None,
                scale_xy: float = 60.0,
                scale_z: float = 5.0,
                eps: float = 1e-4) -> torch.Tensor:
        if geom_cache is None:
            assert pos_batch is not None and yaw_batch is not None and z_batch is not None, \
                "Provide either geom_cache or (pos_batch, yaw_batch, z_batch)."
            gc = RelGeomBias.compute_geom_feat(
                pos_batch, yaw_batch, z_batch,
                use_rbf=self.use_rbf,
                rbf_k=self.rbf_k,
                rbf_mu=self.rbf_mu,
                log_rbf_gamma=self.log_rbf_gamma,
                scale_xy=scale_xy,
                scale_z=scale_z,
                eps=eps,
            )
        else:
            gc = geom_cache

        feat = gc["feat_base"]
        if self.use_rbf:
            r01 = gc["r01"]
            rbf = self._rbf(r01)
            feat = torch.cat([feat, rbf], dim=-1)

        bias = self.mlp(feat)
        bias = bias.permute(0, 3, 1, 2).contiguous()

        B, H, S, _ = bias.shape
        eye = torch.eye(S, device=bias.device, dtype=torch.bool).expand(B, H, S, S)
        bias = bias.masked_fill(eye, 0.0)
        return bias


class RelMHABlock(nn.Module):
    def __init__(self, d_model=128, nhead=4, dim_feedforward=256, dropout=0.2):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.nhead = nhead
        self.dk = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.bias_mod = RelGeomBias(n_heads=nhead, use_rbf=True, rbf_k=8, rbf_gamma=40.0, hidden=96)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, hidden=dim_feedforward, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.log_tau = nn.Parameter(torch.zeros(nhead))

    def _attend(self, q, k, v, attn_bias, key_padding_mask=None, neighbor_mask: Optional[torch.Tensor] = None):
        B, H, S, D = q.shape

        qk = torch.matmul(q, k.transpose(-2, -1))
        logits = qk / math.sqrt(D)
        logits = logits / self.log_tau.exp()[None, :, None, None]
        logits = logits + attn_bias

        if key_padding_mask is not None:
            padmask = key_padding_mask[:, None, None, :].to(torch.bool)
            logits = logits.masked_fill(padmask, float('-inf'))

        if neighbor_mask is not None:
            keep = neighbor_mask[:, None, :, :].to(torch.bool)
            logits = logits.masked_fill(~keep, float('-inf'))

        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out

    def forward(self, h, *,
                geom_cache: Dict,
                key_padding_mask: Optional[torch.Tensor] = None,
                neighbor_mask: Optional[torch.Tensor] = None):
        B, S, d = h.shape
        H, D = self.nhead, self.dk

        x = self.norm1(h)
        q = self.q_proj(x).view(B, S, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, S, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, D).transpose(1, 2)

        bias = self.bias_mod(geom_cache=geom_cache)
        out = self._attend(q, k, v, bias, key_padding_mask=key_padding_mask, neighbor_mask=neighbor_mask)
        out = out.transpose(1, 2).contiguous().view(B, S, H * D)

        h = h + self.dropout(self.o_proj(out))
        h = h + self.ff(self.norm2(h))
        return h


class RelTransformerEncoder(nn.Module):
    def __init__(self, num_layers=3, d_model=128, nhead=4, dim_feedforward=256, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList([
            RelMHABlock(d_model=d_model, nhead=nhead,
                        dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, h, pos_batch, yaw_batch, z_batch,
                key_padding_mask: Optional[torch.Tensor] = None,
                neighbor_mask: Optional[torch.Tensor] = None):
        geom_cache = RelGeomBias.compute_geom_feat(
            pos_batch, yaw_batch, z_batch,
            use_rbf=self.layers[0].bias_mod.use_rbf,
            rbf_k=self.layers[0].bias_mod.rbf_k,
            rbf_mu=self.layers[0].bias_mod.rbf_mu,
            log_rbf_gamma=self.layers[0].bias_mod.log_rbf_gamma,
        )
        for layer in self.layers:
            h = layer(h, geom_cache=geom_cache,
                      key_padding_mask=key_padding_mask,
                      neighbor_mask=neighbor_mask)
        return h


class LoSGraphormer(nn.Module):
    def __init__(self,
                 in_feats=5,
                 d_model=128,
                 nhead=4,
                 num_layers=3,
                 dim_feedforward=256,
                 dropout=0.2,
                 max_len=32,
                 num_classes=2):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.in_lin = nn.Linear(in_feats, d_model)

        self.encoder = RelTransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        nn.init.xavier_uniform_(self.in_lin.weight)
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, data, neighbor_mask: Optional[torch.Tensor] = None):
        x_nodes = data.x
        pos_all = data.pos
        yaw_all = data.yaw
        z_all = data.z
        chains = data.chains

        B = len(chains)
        if B == 0:
            empty_logits = torch.empty((0, 0, self.head.out_features), device=x_nodes.device)
            empty_mask = torch.empty((0, 0), dtype=torch.bool, device=x_nodes.device)
            return empty_logits, empty_mask

        lengths = [c.numel() for c in chains]
        Smax = max(lengths)
        Fdim = x_nodes.size(1)

        x_batch = x_nodes.new_zeros((B, Smax, Fdim))
        pos_batch = pos_all.new_zeros((B, Smax, 2))
        yaw_batch = yaw_all.new_zeros((B, Smax))
        z_batch = z_all.new_zeros((B, Smax))
        pad_mask = torch.ones((B, Smax), dtype=torch.bool, device=x_nodes.device)

        for i, c in enumerate(chains):
            L = c.numel()
            x_batch[i, :L] = x_nodes[c]
            pos_batch[i, :L] = pos_all[c]
            yaw_batch[i, :L] = yaw_all[c]
            z_batch[i, :L] = z_all[c]
            pad_mask[i, :L] = False

        pos_ids = torch.arange(Smax, device=x_nodes.device).clamp(max=self.pos_emb.num_embeddings - 1)
        pos_tok = self.pos_emb(pos_ids)[None, :, :]

        h = self.in_lin(x_batch) + pos_tok
        h = self.encoder(h, pos_batch, yaw_batch, z_batch,
                         key_padding_mask=pad_mask, neighbor_mask=neighbor_mask)
        h = self.norm(h)
        logits = self.head(h)
        return logits, pad_mask
