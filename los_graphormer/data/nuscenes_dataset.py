import math
from typing import List, Dict

import numpy as np
import pandas as pd
from shapely.geometry import Polygon, LineString, Point

import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset


class RMLoSDataset_nuscenes(Dataset):
    """
    Returns Data with:
      x:   (N+1, F)  node features [ego first]
           either:
             F=5: [dist, ray_hit, w, l, h]
             F=4: [dist, w, l, h]
      pos: (N+1, 2)  raw XY (ego=(0,0))
      yaw: (N+1,)    heading (rad), ego=0
      z:   (N+1,)    center z (m), ego=0
      y:   (N+1,)    full labels (0 for ego, occluded (i.e., False Negative) for vehicles)
      chains: list[LongTensor] indices into x/pos/yaw/z
      seq_targets: list[LongTensor] per chain (only last token labeled)
      edge_index: (2,E) ego-star edges (ego→member) for inspection

    Splitting by scene is done using an integer 'scene_index' column if present;
    otherwise all frames are treated as a single scene.
    """
    def __init__(self,
                 csv_path: str,
                 ray_width: float = 1.0,
                 use_ray_hit: bool = True,
                 dist_scale: float = 60.0):
        raw = pd.read_csv(csv_path)

        base_need = {'timestamp', 'x', 'y', 'z', 'w', 'l', 'h', 'heading', 'occluded'}
        if use_ray_hit:
            base_need.add('ray_hit')
        missing = base_need - set(raw.columns)
        if missing:
            raise KeyError(f"CSV missing columns: {sorted(missing)}")

        self.use_ray_hit = use_ray_hit
        self.dist_scale = float(dist_scale)

        # group by timestamp
        self.frames: List[pd.DataFrame] = []
        self.frame_metadata: List[Dict] = []

        for _, g in raw.groupby('timestamp'):
            g = g.reset_index(drop=True)
            self.frames.append(g)
            scene_idx = g['scene_index'].iloc[0] if 'scene_index' in g.columns else -1
            self.frame_metadata.append({
                'timestamp': g['timestamp'].iloc[0],
                'scene_index': int(scene_idx),
            })

        self.ray_width = float(ray_width)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]

        xs = frame['x'].to_numpy(np.float32)
        ys = frame['y'].to_numpy(np.float32)
        zs = frame['z'].to_numpy(np.float32)
        ws = frame['w'].to_numpy(np.float32)
        ls = frame['l'].to_numpy(np.float32)
        hs = frame['h'].to_numpy(np.float32)
        yaws = frame['heading'].to_numpy(np.float32)
        occ = frame['occluded'].to_numpy(np.int64)
        ray_hit = frame['ray_hit'].to_numpy(np.float32) if self.use_ray_hit else None

        N = len(xs)
        centers = np.stack([xs, ys], axis=1).astype(np.float32)
        ego_xy = np.array([0.0, 0.0], dtype=np.float32)

        # polygons for blocker test
        polys = []
        for cx, cy, w, l, yaw in zip(xs, ys, ws, ls, yaws):
            dx, dy = w / 2.0, l / 2.0
            rect = np.array([[-dx, -dy], [-dx, dy], [dx, dy], [dx, -dy]], dtype=np.float32)
            c, s = math.cos(yaw), math.sin(yaw)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)
            poly = Polygon((rect @ R.T) + np.array([cx, cy], dtype=np.float32))
            polys.append(poly)

        # ----- node features & scaling -----
        dists = np.linalg.norm(centers, axis=1).astype(np.float32)
        d_norm = dists / self.dist_scale
        w_norm = ws / 4.0
        l_norm = ls / 8.0
        h_norm = hs / 3.0

        if self.use_ray_hit:
            node_feats = np.stack([d_norm, ray_hit, w_norm, l_norm, h_norm], axis=1)  # (N,5)
            ego_feat = np.zeros((1, 5), dtype=np.float32)
        else:
            node_feats = np.stack([d_norm, w_norm, l_norm, h_norm], axis=1)  # (N,4)
            ego_feat = np.zeros((1, 4), dtype=np.float32)

        all_feats = np.vstack([ego_feat, node_feats])  # (N+1,F)

        # positions & yaw/z (ego first)
        pos = np.vstack([ego_xy[None, :], centers])
        yaw_all = np.concatenate([[0.0], yaws]).astype(np.float32)
        z_all = np.concatenate([[0.0], zs]).astype(np.float32)

        chains, seq_targets = [], []
        edge_src, edge_dst = [], []

        for j in range(N):
            seg = LineString([tuple(ego_xy), tuple(centers[j])])
            corr = seg.buffer(self.ray_width, cap_style=2)

            blockers = []
            for k in range(N):
                if k == j:
                    continue
                if corr.intersects(polys[k]):
                    d = seg.project(Point(centers[k]))
                    blockers.append((k, d))
            blockers.sort(key=lambda x: x[1])

            chain_idx = [0] + [b[0] + 1 for b in blockers] + [j + 1]
            if chain_idx[0] != 0:
                chain_idx = [0] + [n for n in chain_idx if n != 0]

            chains.append(torch.tensor(chain_idx, dtype=torch.long))

            lab = [-100] * (len(chain_idx) - 1) + [int(occ[j])]
            seq_targets.append(torch.tensor(lab, dtype=torch.long))

            for v in chain_idx[1:]:
                edge_src.append(0)
                edge_dst.append(v)

        x = torch.from_numpy(all_feats).float()
        y_full = torch.from_numpy(np.concatenate([[0], occ])).long()
        pos_t = torch.from_numpy(pos).float()
        yaw_t = torch.from_numpy(yaw_all).float()
        z_t = torch.from_numpy(z_all).float()

        if len(edge_src) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

        return Data(
            x=x,
            pos=pos_t,
            yaw=yaw_t,
            z=z_t,
            y=y_full,
            chains=chains,
            seq_targets=seq_targets,
            edge_index=edge_index,
        )