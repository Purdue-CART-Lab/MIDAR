#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 04:35:59 2025

@author: Tianheng Zhu
"""

import math
from typing import Optional
import numpy as np

import torch
from torch_geometric.data import Data
from shapely.geometry import Polygon, LineString, Point

# --------------------------
# Utility: feature normalizer (same as training script)
# --------------------------
def normalize_batch(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-6)

# =============================================================================
# Graph building
# =============================================================================
def _box_corners_xy(x: float, y: float, l: float, w: float, yaw: float) -> np.ndarray:
    """Bottom face 4 corners (x,y) for an oriented box."""
    hl, hw = 0.5 * l, 0.5 * w
    local = np.array([[ hl,  hw],
                      [ hl, -hw],
                      [-hl, -hw],
                      [-hl,  hw]], dtype=float)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s],
                  [s,  c]], dtype=float)
    world = (R @ local.T).T
    world[:, 0] += x
    world[:, 1] += y
    return world  # (4,2)

def _angular_span_from_corners(corners: np.ndarray) -> tuple[float, float]:
    """
    Minimal continuous azimuth span [tmin, tmax] covering all 4 corners.
    May return tmax < tmin (wrap). Caller handles wrap via tiling.
    """
    th = np.arctan2(corners[:, 1], corners[:, 0])           # [-π, π)
    th = (th + np.pi) % (2*np.pi)                           # [0, 2π)
    th.sort()
    gaps = np.diff(np.concatenate([th, th[:1] + 2*np.pi]))
    k = int(np.argmax(gaps))
    start = th[(k + 1) % len(th)]                           # [0, 2π)
    width = (2*np.pi) - gaps[k]
    end = start + width                                     # (start, start+2π]
    start = (start - np.pi)
    end   = (end   - np.pi)
    return float(start), float(end)

def _span_to_bins_robust(tmin: float, tmax: float, n_theta: int) -> np.ndarray:
    """
    Map shortest arc [tmin, tmax] to bin centers in [-π, π).
    Works across the wrap and guarantees ≥1 bin.
    """
    centers = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    arc_len = tmax - tmin                     # in (0, 2π]
    # Modular distance forward from tmin
    delta = (centers - tmin + 2*np.pi) % (2*np.pi)
    mask = (delta <= arc_len) | np.isclose(delta, arc_len)
    idx = np.where(mask)[0]
    if idx.size == 0:
        # Safety net: choose bin nearest the arc midpoint
        mid = (tmin + tmax) * 0.5
        mid = (mid + np.pi) % (2*np.pi) - np.pi
        idx = np.array([int(np.argmin(np.abs(((centers - mid + np.pi) % (2*np.pi)) - np.pi)))], dtype=int)
    return idx

# ===============================================================
# ray_hit calculation: k-depth ray casting
# ===============================================================
def _ray_hit_for_cav(
    centers_xy: np.ndarray,     # (N,2) relative to ego (ego faces +x)
    widths: np.ndarray,         # (N,)
    lengths: np.ndarray,        # (N,)
    heights: np.ndarray,        # (N,)
    headings: np.ndarray,       # (N,) ego-relative, radians
    n_theta: int = 720,
    thresholds: tuple | list | None = (1.8, 3.6)
) -> np.ndarray:
    """
    AOF-style azimuth z-buffer with optional k-depth peeling.
    Depth proxy = distance to box CENTER.

    Returns:
        counts or peeled counts: (N,) float32

    If debug_plot=True, also shows:
      - Cartesian plot of boxes (ego at 0,0), annotated with per-vehicle bin counts from the *final* pass
      - Polar plot of per-bin ownership (owner index color; -1 omitted)
    """
    import numpy as np

    N = int(0 if centers_xy is None else len(centers_xy))
    if N == 0:
        return np.zeros((0,), dtype=np.float32)

    # --- sanitize and shapes
    centers_xy = np.asarray(centers_xy, dtype=np.float32).reshape(N, 2)
    widths     = np.asarray(widths,     dtype=np.float32).reshape(N)
    lengths    = np.asarray(lengths,    dtype=np.float32).reshape(N)
    heights    = np.asarray(heights,    dtype=np.float32).reshape(N)
    headings   = np.asarray(headings,   dtype=np.float32).reshape(N)

    # ensure headings are radians and wrapped (harmless if already so)
    if np.max(np.abs(headings)) > 3.2:
        headings = np.deg2rad(headings)
    headings = (headings + np.pi) % (2*np.pi) - np.pi

    depths_all = np.linalg.norm(centers_xy, axis=1).astype(np.float32)

    # --- helpers (use your existing geometry helpers for consistency) ---
    def _single_aof_pass(spans_idx, depths, n_th):
        theta_depth = np.full(n_th, np.inf, dtype=np.float32)
        theta_owner = np.full(n_th, -1,    dtype=np.int32)
        for i, bins in spans_idx:
            d = depths[i]
            cur = theta_depth[bins]
            upd = d < cur
            if np.any(upd):
                theta_depth[bins[upd]] = d
                theta_owner[bins[upd]] = i
        # counts for this pass
        mask = theta_owner >= 0
        if np.any(mask):
            vi, vc = np.unique(theta_owner[mask], return_counts=True)
            counts = np.zeros(N, dtype=np.float32)
            counts[vi] = vc.astype(np.float32)
        else:
            counts = np.zeros(N, dtype=np.float32)
        return counts, theta_owner

    # Precompute angular spans once (they don't depend on peeling)
    # NOTE: call uses your helper ordering (l, w)
    spans_all = []
    for i in range(N):
        cx, cy = float(centers_xy[i, 0]), float(centers_xy[i, 1])
        l, w, yaw = float(lengths[i]), float(widths[i]), float(headings[i])
        corners = _box_corners_xy(cx, cy, l, w, yaw)
        tmin, tmax = _angular_span_from_corners(corners)
        bins = _span_to_bins_robust(tmin, tmax, n_theta)
        spans_all.append((i, bins))

    # ----------------- SINGLE PASS (no peeling) -----------------
    if not thresholds:
        counts_final, theta_owner_final = _single_aof_pass(spans_all, depths_all, n_theta)
        return counts_final.astype(np.float32)

    # ----------------- K-DEPTH PEELING -----------------
    counts_total = np.zeros(N, dtype=np.float32)
    prev_t = 0.0
    all_thresholds = list(thresholds) + [np.inf]

    # Keep last pass' owner for debugging visuals
    theta_owner_final = None
    counts_last_round = None

    for t in all_thresholds:
        mask_keep = heights > prev_t
        if not np.any(mask_keep):
            prev_t = t
            continue

        kept = set(np.nonzero(mask_keep)[0])
        spans_kept = [(i, bins) for (i, bins) in spans_all if i in kept]
        depths_kept = depths_all  # depths are per-vehicle; _single_aof_pass indexes by i

        round_counts, theta_owner = _single_aof_pass(spans_kept, depths_kept, n_theta)
        counts_last_round = round_counts.copy()
        theta_owner_final = theta_owner.copy()
        

        # per-vehicle height slice thickness
        if np.isfinite(t):
            inc = np.clip(heights - prev_t, 0.0, t - prev_t).astype(np.float32)
        else:
            inc = np.maximum(heights - prev_t, 0.0).astype(np.float32)

        counts_total += round_counts * inc
        prev_t = t
        if not np.isfinite(prev_t):
            break
    #print(counts_total)
    return counts_total.astype(np.float32)

# ===============================================================
# Build inputs for LoS-Graphormer
# ===============================================================
def _build_los_frame_for_cav(
    cav_id: int,
    neighbour_ids: list[int],
    positions: dict[int, tuple[float, float]],
    dims: dict[int, tuple[float, float, float]],
    headings: dict[int, float],
    corridor_width: float = 1.0,
    use_ray_hit: bool = True,
    z: Optional[dict] = None,
    lidar_height: float = 1.75,
) -> Data:
    """
    Build a single-frame torch_geometric.Data object for the CAV and its neighbours.

    Matches MultiHopLoSDataset semantics:
      x:   (N+1, 5) = [dist, bin_score, w, l, h], ego at index 0
      pos: (N+1, 2) = [ego_xy; centers]
      yaw: (N+1,)   = [0; ego-relative yaws]
      z:   (N+1,)   = [0; box-centre heights relative to the ego LiDAR]
      y:   dummy labels (zeros)
      chains, seq_targets, edge_index: same structure as dataset.

    z: optional {id: box-centre height (m) relative to the ego's LiDAR sensor}
       for the neighbours, as in the training data. If None, it is estimated
       for a flat road as h/2 - lidar_height.
    """

    def _to_rad_arr(a):
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            return a
        if np.max(np.abs(a)) > 3.2:   # likely degrees
            a = np.deg2rad(a)
        return a

    # -------- ego pose (global) --------
    cav_x, cav_y = positions[cav_id]
    cav_yaw = _to_rad_arr([headings[cav_id]])[0]

    # -------- neighbors list (no ego) --------
    ids = np.asarray(neighbour_ids, dtype=int)
    if ids.size == 0:
        # Graph with just the ego node
        ego_xy = np.array([0.0, 0.0], dtype=np.float32)
        x      = torch.zeros((1, 5), dtype=torch.float32)
        pos_t  = torch.from_numpy(ego_xy[None, :]).float()
        yaw_t  = torch.tensor([0.0], dtype=torch.float32)
        z_t    = torch.tensor([0.0], dtype=torch.float32)
        data = Data(
            x=x, pos=pos_t, yaw=yaw_t, z=z_t,
            y=torch.zeros(1, dtype=torch.long),
            chains=[], seq_targets=[],
            edge_index=torch.empty((2, 0), dtype=torch.long)
        )
        return data

    # -------- gather neighbor attributes (global frame) --------
    gx_all = np.array([positions[i][0] for i in ids], dtype=float)
    gy_all = np.array([positions[i][1] for i in ids], dtype=float)
    w_all  = np.array([dims[i][0]      for i in ids], dtype=float)
    l_all  = np.array([dims[i][1]      for i in ids], dtype=float)
    h_all  = np.array([dims[i][2]      for i in ids], dtype=float)
    yaw_all= np.array([headings[i]     for i in ids], dtype=float)

    # ego-relative headings
    yaws = _to_rad_arr(yaw_all) - cav_yaw
    yaws = (yaws + np.pi) % (2*np.pi) - np.pi   # wrap to [-pi, pi)

    # -------- translate & rotate into ego frame --------
    rel = np.column_stack([gx_all - cav_x, gy_all - cav_y])
    c, s = np.cos(-cav_yaw), np.sin(-cav_yaw)
    R = np.array([[c, -s],
                  [s,  c]], dtype=float)
    centers = (rel @ R.T).astype(np.float32)     # (N,2)

    ws = w_all.astype(np.float32)
    ls = l_all.astype(np.float32)
    hs = h_all.astype(np.float32)
    yaws = yaws.astype(np.float32)

    # -------- polygons for blocker test (IDENTICAL to dataset) --------
    polys = []
    for cx, cy, w, l, yaw in zip(centers[:, 0], centers[:, 1], ws, ls, yaws):
        dx, dy = w / 2.0, l / 2.0
        rect = np.array(
            [[-dx, -dy],
             [-dx,  dy],
             [ dx,  dy],
             [ dx, -dy]],
            dtype=np.float32
        )
        c, s = np.cos(yaw), np.sin(yaw)
        Rloc = np.array([[c, -s],
                         [s,  c]], dtype=np.float32)
        poly = Polygon((rect @ Rloc.T) + np.array([cx, cy], dtype=np.float32))
        polys.append(poly)


    N = centers.shape[0]
    ego_xy = np.array([0.0, 0.0], dtype=np.float32)
    if z is None:
        zs = (0.5 * hs - lidar_height).astype(np.float32)
    else:
        zs = np.array([z[i] for i in ids], dtype=np.float32)
    # -------------------------------------------------
    # Node features
    #   use_ray_hit=True  -> [dist, bin_score, w, l, h]  (F=5)
    #   use_ray_hit=False -> [dist, w, l, h]             (F=4)
    # -------------------------------------------------
    dists = np.linalg.norm(centers, axis=1).astype(np.float32)

    if use_ray_hit:
        ray_hit = _ray_hit_for_cav(
            centers_xy=centers,
            widths=ws,
            lengths=ls,
            heights=hs,
            headings=yaws,
            n_theta=720,
            thresholds=(1.8, 3.6),
        ).astype(np.float32)

        # -------- node features [dist, ray_hit, w, l, h] + scaling --------
        node_feats = np.stack([dists, ray_hit, ws, ls, hs], axis=1)  # (N,5)
        ego_feat = np.zeros((1, 5), dtype=np.float32)
        all_feats = np.vstack([ego_feat, node_feats])                  # (N+1,5)

        all_feats[:, 0] /= 80.0   # dist
        all_feats[:, 2] /= 4.0    # w
        all_feats[:, 3] /= 8.0    # l
        all_feats[:, 4] /= 3.0    # h
    else:
        # -------- node features [dist, w, l, h] + scaling --------
        node_feats = np.stack([dists, ws, ls, hs], axis=1)  # (N,5)
        ego_feat = np.zeros((1, 4), dtype=np.float32)
        all_feats = np.vstack([ego_feat, node_feats])                  # (N+1,5)

        all_feats[:, 0] /= 80.0   # dist
        all_feats[:, 1] /= 4.0    # w
        all_feats[:, 2] /= 8.0    # l
        all_feats[:, 3] /= 3.0    # h


    # -------- pos / yaw / z with ego at index 0 --------
    pos     = np.vstack([ego_xy[None, :], centers])                # (N+1,2)
    yaw_all = np.concatenate([[0.0], yaws]).astype(np.float32)     # (N+1,)
    z_all   = np.concatenate([[0.0], zs]).astype(np.float32)       # (N+1,)

    # -------- chains & ego-star edges (same as dataset) --------
    chains, seq_targets = [], []
    edge_src, edge_dst = [], []

    for j in range(N):
        seg = LineString([tuple(ego_xy), tuple(centers[j])])
        corr = seg.buffer(corridor_width, cap_style=2)

        blockers = []
        for k in range(N):
            if k == j:
                continue
            if corr.intersects(polys[k]):
                d = seg.project(Point(centers[k]))
                blockers.append((k, d))
        blockers.sort(key=lambda t: t[1])

        chain_idx = [0] + [k + 1 for (k, _) in blockers] + [j + 1]
        chains.append(torch.tensor(chain_idx, dtype=torch.long))

        # inference-time: label last token 0 (placeholder, not used by loss)
        seq_targets.append(
            torch.tensor([-100] * (len(chain_idx) - 1) + [0], dtype=torch.long)
        )

        for v in chain_idx[1:]:
            edge_src.append(0)
            edge_dst.append(v)

    # -------- pack into Data --------
    x        = torch.from_numpy(all_feats).float()
    pos_t    = torch.from_numpy(pos).float()
    yaw_t    = torch.from_numpy(yaw_all).float()
    z_t      = torch.from_numpy(z_all).float()
    edge_index = (torch.empty((2, 0), dtype=torch.long)
                  if not edge_src else
                  torch.tensor([edge_src, edge_dst], dtype=torch.long))

    data = Data(
        x=x,
        pos=pos_t,
        yaw=yaw_t,
        z=z_t,
        y=torch.zeros(x.size(0), dtype=torch.long),   # dummy labels for inference
        chains=chains,
        seq_targets=seq_targets,
        edge_index=edge_index
    )
    return data

def los_last_token_probs(data, logits):
    # logits is (N,C) or (S,C) after normalization in caller
    if logits.ndim != 2 or logits.shape[1] != 2 or logits.shape[0] == 0:
        return [], np.array([], dtype=float)

    chains = getattr(data, "chains", [])
    if not chains:
        return [], np.array([], dtype=float)

    per_node = (logits.shape[0] == data.x.size(0))
    p = torch.softmax(logits, dim=-1)[:, 1]  # occlusion prob

    targets, p_last = [], []
    if per_node:
        for ch in chains:
            tgt = int(ch[-1].item())
            if 0 <= tgt < p.shape[0]:
                targets.append(tgt)
                p_last.append(p[tgt].item())
    else:
        # Treat row index as token index (already flattened/squeezed).
        # If you need true (S) mapping, extend this to track per-chain lengths.
        for ch in chains:
            tgt = int(ch[-1].item())
            if 0 <= tgt < p.shape[0]:
                targets.append(tgt)
                p_last.append(p[tgt].item())

    return targets, np.asarray(p_last, dtype=np.float32)

def los_visible_ids_from_graph(data, model, occ_thresh=0.5):
    chains = getattr(data, "chains", None)
    if not chains or len(chains) == 0:
        return {0}

    # IMPORTANT: match training preprocessing
    data = data.clone()  # avoid in-place side effects if you reuse data
    data.x = normalize_batch(data.x).clamp(-5, 5)

    model.eval()
    with torch.no_grad():
        out = model(data)
        logits = out[0] if isinstance(out, (tuple, list)) else out

    # ---- normalize logits shape ----
    if logits.ndim == 3:
        # (B,S,C) → pick last token per chain (same as training/eval)
        B, S, C = logits.shape
        if B != len(chains):
            raise RuntimeError(
                f"Expected B == len(chains) but got B={B}, len(chains)={len(chains)}"
            )
        last_pos = torch.tensor(
            [ch.numel() - 1 for ch in chains],
            device=logits.device,
            dtype=torch.long
        )
        last_logits = logits[torch.arange(B, device=logits.device), last_pos, :]  # (B,C)

    elif logits.ndim == 2:
        # (B,C): already one logit per chain (e.g., pooled)
        B, C = logits.shape
        if B != len(chains):
            raise RuntimeError(
                f"Expected B == len(chains) but got B={B}, len(chains)={len(chains)}"
            )
        last_logits = logits
    else:
        raise RuntimeError(f"Unexpected logits shape {tuple(logits.shape)}")

    # ---- occlusion probability for each chain's target node ----
    p_occ = torch.softmax(last_logits, dim=-1)[:, 1]  # (B,)
    p_occ = p_occ.cpu().numpy()

    # chain last index = target node index (compatible with Data.x/pos/yaw/z)
    target_nodes = [int(ch[-1].item()) for ch in chains]

    # ---- Visibility decision per node ----
    visible = {0}  # ego always visible
    for tgt, p in zip(target_nodes, p_occ):
        if p < occ_thresh:
            visible.add(tgt)

    return visible