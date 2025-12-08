from typing import Optional

import math
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

from .los_graphormer import RMSNorm  # reuse if you used RMSNorm in vanilla

# ──────────────────────────────────────────────────────────────────────────────
# MLP baseline
# ──────────────────────────────────────────────────────────────────────────────

class MLPBaseline(nn.Module):
    """
    Per-vehicle baseline model.
    Input: last-node features (dist, bin_score, w, l, h)
    Output: logits for 2 classes (occluded vs not).
    """
    def __init__(self, in_feats=5, hidden=128, num_classes=2, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feats, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )
        self.is_mlp = True  # flag so we can detect this model type

    def forward(self, data):
        # data.x is already normalized in train_epoch / evaluate
        x = data.x
        chains = data.chains
        feats = []

        # take the last node feature of each chain
        for c in chains:
            last_idx = c[-1].item()
            feats.append(x[last_idx])

        feats = torch.stack(feats, dim=0)  # (B, 5)
        logits = self.net(feats)          # (B, 2)
        return logits  # NOTE: no extra dims, no tuple

# ──────────────────────────────────────────────────────────────────────────────
# Plain Transformer Encoder (no RelGeomBias)
# ──────────────────────────────────────────────────────────────────────────────

class PlainMHABlock(nn.Module):
    """
    Standard MHA + SwiGLU FFN, PreNorm, NO geometric bias.
    """
    def __init__(self, d_model=128, nhead=4, dim_feedforward=256, dropout=0.2):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.nhead = nhead
        self.dk = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ff    = SwiGLU(d_model, hidden=dim_feedforward, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def _attend(self, q, k, v, key_padding_mask=None):
        """
        q,k,v: (B,H,S,D)
        key_padding_mask: (B,S) True = pad
        """
        B, H, S, D = q.shape

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # (B,H,S,S)

        if key_padding_mask is not None:
            padmask = key_padding_mask[:, None, None, :].to(torch.bool)  # (B,1,1,S)
            logits = logits.masked_fill(padmask, float('-inf'))

        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out  = torch.matmul(attn, v)  # (B,H,S,D)
        return out

    def forward(self, h, key_padding_mask=None):
        """
        h: (B,S,d)
        key_padding_mask: (B,S) True=pad
        """
        B, S, d = h.shape
        H, D = self.nhead, self.dk

        # PreNorm
        x = self.norm1(h)

        q = self.q_proj(x).view(B, S, H, D).transpose(1, 2)  # (B,H,S,D)
        k = self.k_proj(x).view(B, S, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, D).transpose(1, 2)

        out = self._attend(q, k, v, key_padding_mask=key_padding_mask)
        out = out.transpose(1, 2).contiguous().view(B, S, H * D)

        h = h + self.dropout(self.o_proj(out))
        h = h + self.ff(self.norm2(h))
        return h


class PlainTransformerEncoder(nn.Module):
    """
    Stack of PlainMHABlock, no geometry bias.
    """
    def __init__(self, num_layers=3, d_model=128, nhead=4, dim_feedforward=256, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList([
            PlainMHABlock(d_model=d_model, nhead=nhead,
                          dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, h, key_padding_mask=None):
        for layer in self.layers:
            h = layer(h, key_padding_mask=key_padding_mask)
        return h

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
    
# ──────────────────────────────────────────────────────────────────────────────
# LoS-VanillaTransformer (no RelGeomBias; same chain packing)
# ──────────────────────────────────────────────────────────────────────────────

class LoSVanillaTransformer(nn.Module):
    """
    Sequence model baseline:
      - Same chain construction as LoSGraphormer
      - Uses only node features + positional index embedding
      - NO relative geometry bias in attention
    """
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
        self.in_lin  = nn.Linear(in_feats, d_model)

        self.encoder = PlainTransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        nn.init.xavier_uniform_(self.in_lin.weight)
        nn.init.xavier_uniform_(self.head.weight)

        self.is_mlp = False  # just to be explicit

    def forward(self, data, neighbor_mask=None):
        """
        Pack chains into (B,S,F).
        Ignore pos/yaw/z except for chain structure; only x + positional index.
        Return logits with shape (B,S,C) to reuse last-token logic.
        """
        x_nodes = data.x          # (N+1, F=5)
        pos_all = data.pos        # (N+1, 2)  # not used, but kept for symmetry
        yaw_all = data.yaw        # (N+1,)    # not used
        z_all   = data.z          # (N+1,)    # not used
        chains  = data.chains

        B = len(chains)
        if B == 0:
            empty_logits = torch.empty((0, 0, self.head.out_features), device=x_nodes.device)
            empty_mask   = torch.empty((0, 0), dtype=torch.bool, device=x_nodes.device)
            return empty_logits, empty_mask

        lengths = [c.numel() for c in chains]
        Smax = max(lengths)
        Fdim = x_nodes.size(1)

        x_batch  = x_nodes.new_zeros((B, Smax, Fdim))
        pad_mask = torch.ones((B, Smax), dtype=torch.bool, device=x_nodes.device)

        for i, c in enumerate(chains):
            L = c.numel()
            x_batch[i, :L] = x_nodes[c]
            pad_mask[i, :L] = False

        pos_ids = torch.arange(Smax, device=x_nodes.device).clamp(max=self.pos_emb.num_embeddings - 1)
        pos_tok = self.pos_emb(pos_ids)[None, :, :]  # (1,S,d)

        h = self.in_lin(x_batch) + pos_tok
        h = self.encoder(h, key_padding_mask=pad_mask)
        h = self.norm(h)
        logits = self.head(h)  # (B,S,C)
        return logits, pad_mask

# ──────────────────────────────────────────────────────────────────────────────
# GCN baseline
# ──────────────────────────────────────────────────────────────────────────────
class GCNOnChains(nn.Module):
    """
    GCN baseline that builds its own RM-LoS graph from `data.chains`,
    ignoring `data.edge_index` coming from the dataset.

    - Nodes: same as others (ego + vehicles).
    - Edges: for each chain [n0, n1, ..., nk], connect consecutive nodes
             (n0<->n1, n1<->n2, ..., nk-1<->nk), undirected.
    - Outputs per-node logits; training/eval pick last node per chain.
    """
    def __init__(self, in_feats=5, hidden=64, num_layers=2, num_classes=2, dropout=0.1):
        super().__init__()
        self.is_gcn = True

        layers = []
        in_dim = in_feats
        for _ in range(num_layers - 1):
            layers.append(GCNConv(in_dim, hidden))
            in_dim = hidden
        self.convs = nn.ModuleList(layers)

        self.out_conv = GCNConv(in_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.head = nn.Linear(hidden, num_classes)

    def _build_chain_edge_index(self, chains, num_nodes, device):
        """
        Build edge_index from chains:

        For each chain [n0, n1, ..., nk], add undirected edges
        (n0<->n1), (n1<->n2), ..., (nk-1<->nk).
        """
        edge_pairs = set()
        for c in chains:
            idxs = c.tolist()
            for u, v in zip(idxs[:-1], idxs[1:]):
                if u == v:
                    continue
                edge_pairs.add((u, v))
                edge_pairs.add((v, u))

        if not edge_pairs:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        src, dst = zip(*edge_pairs)
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
        return edge_index

    def forward(self, data):
        x = data.x  # (N_nodes, F)
        chains = data.chains
        device = x.device
        num_nodes = x.size(0)

        # Build RM-LoS edge_index from chains, ignore data.edge_index
        edge_index = self._build_chain_edge_index(chains, num_nodes, device)

        # Standard GCN stack
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)

        x = self.out_conv(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)

        logits = self.head(x)  # (N_nodes, num_classes)
        return logits

