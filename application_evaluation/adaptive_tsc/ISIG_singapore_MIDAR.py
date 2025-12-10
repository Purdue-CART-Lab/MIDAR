#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISIG + MIDAR for Singapore CAV Adaptive Control Demo

- Builds a per-CAV "frame" compatible with RMLoSDataset:
  x:   [dist, ray_hit, w, l, h]
  pos: XY (ego first)
  yaw: heading (rad), ego=0
  z:   center z (m), ego=0  (flat BEV)
  LoS chains: [ego, blockers..., target]

Created: Mon Nov  3 2025
"""

# %% imports
import time
import math
import numpy as np
import random
import traci
import os
import sys

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from shapely.geometry import LineString, Point
from shapely.geometry import Polygon as ShpPolygon

# Import MIDAR implementation
sys.path.append('../../los_graphormer/models')
from los_graphormer import LoSGraphormer

# =============================================================================
# Utility: feature normalizer
# =============================================================================
def _normalize_batch(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(0)) / (x.std(0) + 1e-6)

# =============================================================================
# Graph building
# =============================================================================
def _box_corners_xy(x: float, y: float, l: float, w: float, yaw: float) -> np.ndarray:
    # Bottom face 4 corners (x,y) for an oriented box
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
    return world

def _box_polygon(cx, cy, w, l, yaw):
    # Reuse the same exact corner generator & (l,w) convention:
    corners = _box_corners_xy(cx, cy, l, w, yaw)
    return ShpPolygon(corners)

def _angular_span_from_corners(corners: np.ndarray) -> tuple[float, float]:
    # Minimal continuous azimuth span [tmin, tmax] covering all 4 corners.
    # May return tmax < tmin (wrap). Caller handles wrap via tiling.
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

def _span_to_rays_robust(tmin: float, tmax: float, n_theta: int) -> np.ndarray:
    # Map shortest arc [tmin, tmax] to [-π, π).
    # Works across the wrap and guarantees ≥1 ray.
    centers = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    arc_len = tmax - tmin                     # in (0, 2π]
    # Modular distance forward from tmin
    delta = (centers - tmin + 2*np.pi) % (2*np.pi)
    mask = (delta <= arc_len) | np.isclose(delta, arc_len)
    idx = np.where(mask)[0]
    if idx.size == 0:
        # Safety net: choose ray nearest the arc midpoint
        mid = (tmin + tmax) * 0.5
        mid = (mid + np.pi) % (2*np.pi) - np.pi
        idx = np.array([int(np.argmin(np.abs(((centers - mid + np.pi) % (2*np.pi)) - np.pi)))], dtype=int)
    return idx

# ===============================================================
# K-DEPTH Ray Casting
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

    N = int(0 if centers_xy is None else len(centers_xy))
    if N == 0:
        return np.zeros((0,), dtype=np.float32)

    # --- sanitize and shapes
    centers_xy = np.asarray(centers_xy, dtype=np.float32).reshape(N, 2)
    widths     = np.asarray(widths,     dtype=np.float32).reshape(N)
    lengths    = np.asarray(lengths,    dtype=np.float32).reshape(N)
    heights    = np.asarray(heights,    dtype=np.float32).reshape(N)
    headings   = np.asarray(headings,   dtype=np.float32).reshape(N)

    # ensure headings are radians and wrapped
    if np.max(np.abs(headings)) > 3.2:
        headings = np.deg2rad(headings)
    headings = (headings + np.pi) % (2*np.pi) - np.pi

    depths_all = np.linalg.norm(centers_xy, axis=1).astype(np.float32)

    # --- helpers ---
    def _single_ray_casting(spans_idx, depths, n_th):
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
        bins = _span_to_rays_robust(tmin, tmax, n_theta)
        spans_all.append((i, bins))

    # ----------------- SINGLE Ray Casting -----------------
    if not thresholds:
        counts_final, theta_owner_final = _single_ray_casting(spans_all, depths_all, n_theta)
        return counts_final.astype(np.float32)

    # ----------------- K-DEPTH Ray Casting -----------------
    counts_total = np.zeros(N, dtype=np.float32)
    prev_t = 0.0
    all_thresholds = list(thresholds) + [np.inf]

    for t in all_thresholds:
        mask_keep = heights > prev_t
        if not np.any(mask_keep):
            prev_t = t
            continue

        kept = set(np.nonzero(mask_keep)[0])
        spans_kept = [(i, bins) for (i, bins) in spans_all if i in kept]
        depths_kept = depths_all  # depths are per-vehicle; _single_ray_casting indexes by i

        round_counts, theta_owner = _single_ray_casting(spans_kept, depths_kept, n_theta)

        # per-vehicle height slice thickness
        if np.isfinite(t):
            inc = np.clip(heights - prev_t, 0.0, t - prev_t).astype(np.float32)
        else:
            inc = np.maximum(heights - prev_t, 0.0).astype(np.float32)

        counts_total += round_counts * inc
        prev_t = t
        if not np.isfinite(prev_t):
            break
    
    return counts_total.astype(np.float32)

# ===============================================================
# Build RMLoS and LoS Chains
# ===============================================================
def _build_los_frame_for_cav(cav_id: str,
                             neighbour_ids: list[str],
                             positions: dict[str, tuple[float, float]],
                             dims: dict[str, tuple[float, float, float]],
                             headings: dict[str, float],
                             corridor_width: float = 1.0) -> Data:
    # Build a single-frame torch_geometric.Data object for the CAV and its neighbours.
    cav_x, cav_y = positions[cav_id]
    N = len(neighbour_ids)

    xs, ys, zs, ws, ls, hs, yaws = [], [], [], [], [], [], []
    for vid in neighbour_ids:
        x, y = positions[vid]
        rel_x, rel_y = x - cav_x, y - cav_y
        w, l, h = dims[vid]
        yaw = headings[vid]
        xs.append(rel_x); ys.append(rel_y); zs.append(0.0)
        ws.append(w); ls.append(l); hs.append(h); yaws.append(yaw)

    xs, ys, zs, ws, ls, hs, yaws = map(np.asarray,
        [xs, ys, zs, ws, ls, hs, yaws], [np.float32]*7)
    centers = np.stack([xs, ys], axis=1)
    ego_xy = np.array([0.0, 0.0], dtype=np.float32)

    # --- polygons for corridor intersection ---
    polys = []
    for cx, cy, w, l, yaw in zip(xs, ys, ws, ls, yaws):
        polys.append(_box_polygon(cx, cy, w, l, yaw))

    # --- ray_hit ---
    bin_score = _ray_hit_for_cav(
        centers_xy=centers,
        widths=ws,
        lengths=ls,
        heights=hs,
        headings=yaws,
        n_theta=720,
        thresholds=(1.8,3.6)
    ).astype(np.float32)

    # --- node features ---
    dists = np.linalg.norm(centers, axis=1).astype(np.float32)
    node_feats = np.stack([dists, bin_score, ws, ls, hs], axis=1)
    ego_feat = np.zeros((1, 5), dtype=np.float32)
    all_feats = np.vstack([ego_feat, node_feats])
    all_feats[:, 0] /= 80.0
    all_feats[:, 2] /= 4.0
    all_feats[:, 3] /= 8.0
    all_feats[:, 4] /= 3.0

    # --- pos/yaw/z arrays ---
    pos = np.vstack([ego_xy[None, :], centers])
    yaw_all = np.concatenate([[0.0], yaws]).astype(np.float32)
    z_all = np.concatenate([[0.0], zs]).astype(np.float32)

    # --- chains and edges (same as before) ---
    chains, seq_targets, edge_src, edge_dst = [], [], [], []
    for j in range(N):
        seg = LineString([tuple(ego_xy), tuple(centers[j])])
        corr = seg.buffer(corridor_width, cap_style=2)
        blockers = []
        for k in range(N):
            if k == j: continue
            if corr.intersects(polys[k]):
                d = seg.project(Point(centers[k]))
                blockers.append((k, d))
        blockers.sort(key=lambda x: x[1])
        chain_idx = [0] + [b[0] + 1 for b in blockers] + [j + 1]
        chains.append(torch.tensor(chain_idx, dtype=torch.long))
        seq_targets.append(torch.tensor([-100]*(len(chain_idx)-1)+[0], dtype=torch.long))
        for v in chain_idx[1:]:
            edge_src.append(0); edge_dst.append(v)

    x = torch.from_numpy(all_feats).float()
    pos_t = torch.from_numpy(pos).float()
    yaw_t = torch.from_numpy(yaw_all).float()
    z_t = torch.from_numpy(z_all).float()
    edge_index = (torch.empty((2,0), dtype=torch.long) if not edge_src
                  else torch.tensor([edge_src, edge_dst], dtype=torch.long))

    data = Data(
        x=x, pos=pos_t, yaw=yaw_t, z=z_t,
        y=torch.zeros(x.size(0), dtype=torch.long),
        chains=chains, seq_targets=seq_targets,
        edge_index=edge_index
    )
    return data

@torch.no_grad()
def _predict_visibility_graphormer(model: LoSGraphormer, data: Data, device) -> np.ndarray:
    
    # Returns an array probabilities of FNs for all nodes (vehicles, index-aligned with data.x),
    # where node 0 is ego. We fill non-target nodes with nan and targets
    # (last token of each chain) with the corresponding probability.
    
    data = data.to(device)
    data.x = _normalize_batch(data.x).clamp(-5, 5)

    logits, _pad = model(data) # (B=NumChainsPerFrame, S, C) but B==#chains since we batch frames=1
    # In our packing, each chain is a separate sequence inside the same frame-batch.
    B, S, C = logits.shape
    # We need last token of each chain
    last_pos = torch.tensor([c.numel() - 1 for c in data.chains], device=logits.device, dtype=torch.long)
    last_logits = logits[torch.arange(B, device=logits.device), last_pos, :]  # (B, C)
    probs = F.softmax(last_logits, dim=-1)[:, 1].flatten().cpu().numpy()      # P(occluded) per chain

    # Map back to node indices (each chain ends at a unique node)
    p_occ_nodes = np.full((data.num_nodes,), np.nan, dtype=np.float32)
    for ci, chain in enumerate(data.chains):
        target_node = chain[-1].item()
        p_occ_nodes[target_node] = probs[ci]
    # Fill ego with 1.0 (irrelevant) so downstream filtering ignores it
    p_occ_nodes[0] = 1.0
    return p_occ_nodes

# =============================================================================
# CAV helpers
# =============================================================================
def update_CAV_flags(veh_ids: list[str]) -> None:
    for vid in veh_ids:
        if vid not in cav_flag:
            cav_flag[vid] = random.random() < PENETRATION_RATE

def get_observed_vehicle_ids(veh_ids: list[str]) -> tuple[list[str], list[float]]:

    # Returns ([observed_ids], [per-inference time]) using LoS-Graphormer.
    # A vehicle is considered observed if at least one present CAV predicts it

    if not veh_ids:
        return [], []

    # Bulk TraCI pulls
    positions = {v: traci.vehicle.getPosition(v) for v in veh_ids}

    dims = {}
    for v in veh_ids:
        try:
            dims[v] = (traci.vehicle.getWidth(v),
                       traci.vehicle.getLength(v),
                       traci.vehicle.getHeight(v))
        except Exception:
            dims[v] = (traci.vehicle.getWidth(v),
                       traci.vehicle.getLength(v),
                       1.5)

    headings = {v: math.radians(traci.vehicle.getAngle(v)) for v in veh_ids}

    observed: set[str] = set()
    current_cavs = [v for v in veh_ids if cav_flag.get(v, False)]
    gnn_time_list = []

    for cav in current_cavs:
        cx, cy = positions[cav]
        neigh = [v for v in veh_ids
                 if v != cav and math.hypot(positions[v][0]-cx, positions[v][1]-cy) <= PERCEPTION_RANGE]
        if not neigh:
            observed.add(cav)
            continue

        # Build one frame (chains) and run the transformer once
        start = time.perf_counter()

        data = _build_los_frame_for_cav(cav, neigh, positions, dims, headings)
        p_occ_nodes = _predict_visibility_graphormer(_gnn, data, device)

        gnn_time_list.append(time.perf_counter() - start)

        # Nodes map: 0=ego; 1..N correspond to neigh list order
        visible_neigh = [
            neigh[i-1]
            for i in range(1, data.num_nodes)
            if (not np.isnan(p_occ_nodes[i])) and (p_occ_nodes[i] < OCCLUDED_THRESH)
        ]
        observed.update(visible_neigh)
        observed.add(cav)

    return list(observed), gnn_time_list

# =============================================================================
# ISIG Adaptive Signal Control Code
# =============================================================================

def find_last_effective_element(veh_routes):
    for element in reversed(veh_routes):
        if not element.startswith(":"):
            return element

def mapping_route2phase(veh_routes):
    lane_id = find_last_effective_element(veh_routes)
    if lane_id in ['74859235#1_2','74859235#1_1','652556221_2']:
        return 0
    elif lane_id in ['74859235#1_0','744913575_1']:
        return 1
    elif lane_id in ['744913575_0',]:
        return 2
    elif lane_id in ['652556221_1']:
        return 0 if random.random() <= 0.5 else 1
    elif lane_id in ['652556221_0','652556226_0']:
        return 1 if random.random() <= 0.5 else 2
    elif lane_id in ['652556226_1']:
        return 0 if random.random() <= 0.67 else 1
    elif lane_id in ['840414102_4','840414102_3','840414102_2','652291042_3','652291042_2','652291042_1','660362915_2','660362915_1']:
        return 3
    elif lane_id in ['840414102_1','840414102_0','652291042_0','660362915_0']:
        return 4
    elif lane_id in ['840414101_4','840414101_3','479616091_3','479616091_2','654999305_2']:
        return 5
    elif lane_id in ['840414101_2','840414101_1','479616091_1','654999305_1']:
        return 6
    elif lane_id in ['840414101_0']:
        return 7
    elif lane_id in ['479616091_0','654999305_0']:
        return 6 if random.random() <= 0.5 else 7
    elif lane_id in ['173767662_5','173767662_4','654985989_4']:
        return 8
    elif lane_id in ['173767662_3','173767662_2','173767662_1','654985989_3','654985989_2','654985989_1','630390098_1','630390098_2']:
        return 9
    elif lane_id in ['173767662_0','654985989_0','630390098_0']:
        return 10
    elif lane_id in ['630390098_3']:
        return 8 if random.random() <= 0.67 else 9

def phase2ETA(veh_id, intersection_center):
    veh_speed = traci.vehicle.getSpeed(veh_id)
    x2, y2 = intersection_center
    if veh_speed <= 0.9:
        return 0
    else:
        x1, y1 = traci.vehicle.getPosition(veh_id)
        return round(math.sqrt((x2 - x1)**2 + (y2 - y1)**2)/veh_speed)

def return_ETA_cell(veh_id, intersection_center, plan_horizon, routes):
    phase_id = mapping_route2phase(routes[veh_id])
    ETA = phase2ETA(veh_id, intersection_center)
    if ETA >= plan_horizon-1:
        return False
    else:
        if phase_id is not None:
            return [ETA, phase_id]

def return_signal(phase):
    if phase == "012":
        return [6,7,8]
    elif phase == '34':
        return [3, 4, 5]
    elif phase == '567':
        return [9,10,11]
    elif phase == '8910':
        return [0,1,2]

def signal_DP_sup_2(s, x, phase, prev_stage_calculation, G_min_T, Arrival_Table, offset):
    prev_stage_bu = prev_stage_calculation.copy()[s-x-(Yellow+Red)*2-offset]
    phase_loc_1, phase_loc_2, phase_loc_3 = return_phase_loc_spec(phase)
    for i in range(s-5-x+1, s-5+1):
        prev_stage_bu = np.vstack((prev_stage_bu, prev_stage_bu[i-1,]+np.append(Arrival_Table[i, :], [0, 0])))
        prev_stage_bu[i, phase_loc_1] = np.maximum(prev_stage_bu[i, phase_loc_1] - departure_rate, 0)
        prev_stage_bu[i, phase_loc_2] = np.maximum(prev_stage_bu[i, phase_loc_2] - departure_rate*2, 0)
        prev_stage_bu[i, phase_loc_3] = np.maximum(prev_stage_bu[i, phase_loc_3] - departure_rate*3, 0)
        prev_stage_bu[i, -1] = prev_stage_bu[i-1, -1] + np.sum(prev_stage_bu[i, :approaches])
    for i in range(s-5+1, s+1):
        prev_stage_bu = np.vstack((prev_stage_bu, prev_stage_bu[i-1,]+np.append(Arrival_Table[i, :], [0, 0])))
        prev_stage_bu[i, -1] = prev_stage_bu[i-1, -1] + np.sum(prev_stage_bu[i, :approaches])
    return x, prev_stage_bu[-1, -1], prev_stage_bu

def signal_DP_sup_1(s, phase, prev_stage_calculation, G_min_T, G_max_T, Arrival_Table, s_start, offset):
    upload = [-1, np.inf]
    X = []
    for i in range(G_min_T[phase], G_max_T+1):
        if s-5-i >= s_start-Yellow-Red-G_min_T[phase]:
            X.append(i)
    for x_minor in X:
        x, value, stage = signal_DP_sup_2(s, x_minor, phase, prev_stage_calculation, G_min_T, Arrival_Table, offset)
        if value < upload[1]:
            upload = [x, value, stage, False]
    return upload

def return_phase_loc(phase):
    if phase == '012': return [0,1,2]
    elif phase == '34': return [3,4]
    elif phase == '567': return [5,6,7]
    elif phase == '8910': return [8,9,10]

def return_phase_loc_spec(phase):
    if phase == '012':
        return [2],[0,1],[]
    elif phase == '34':
        return [],[4],[3]
    elif phase == '567':
        return [7],[5,6],[]
    elif phase == '8910':
        return [10],[8],[9]

def return_phase_char(index):
    if index == 0: return '8910'
    elif index == 3: return '34'
    elif index == 6: return '012'
    elif index == 9: return '567'

def signal_DP(Arrival_Table, G_min_T, G_max_T, phase_sequence, plan_horizon, approaches, departure_rate):
    phase_sequence_index = 0
    flag = True
    while flag:
        phase = phase_sequence[phase_sequence_index % len(phase_sequence)]
        phase_loc = return_phase_loc(phase)
        all_zero_columns = np.all(Arrival_Table == 0, axis=0)
        if all_zero_columns[phase_loc].all():
            phase_sequence_index += 1
            continue
        else:
            flag = False
        phase_loc_1, phase_loc_2, phase_loc_3 = return_phase_loc_spec(phase)
        prev_stage_calculation = []
        for i in range(G_min_T[phase]+Yellow+Red+1, G_max_T+Yellow+Red+1+1):
            s1c_temp = np.zeros((i, approaches+2))
            s1c_temp[G_min_T[phase]+Yellow+Red:i, -2] = np.arange(G_min_T[phase], i-5)
            s1c_temp[0, :approaches] = Arrival_Table[0, :approaches]
            s1c_temp[0, -1] = np.sum(s1c_temp[0, :approaches])
            for j in range(1, i-5):
                s1c_temp[j, :approaches] = s1c_temp[j-1,:approaches]+Arrival_Table[j, :approaches]
                s1c_temp[j, phase_loc_1] = np.maximum(s1c_temp[j, phase_loc_1] - departure_rate, 0)
                s1c_temp[j, phase_loc_2] = np.maximum(s1c_temp[j, phase_loc_2] - departure_rate*2, 0)
                s1c_temp[j, phase_loc_3] = np.maximum(s1c_temp[j, phase_loc_3] - departure_rate*3, 0)
                s1c_temp[j, -1] = s1c_temp[j-1, -1] + np.sum(s1c_temp[j, :approaches])
            for j in range(i-5, len(s1c_temp)):
                s1c_temp[j, :approaches] = s1c_temp[j - 1,:approaches] + Arrival_Table[j, :approaches]
                s1c_temp[j, -1] = s1c_temp[j-1, -1] + np.sum(s1c_temp[j, :approaches])
            prev_stage_calculation.append(s1c_temp)
    historical_decision_table = np.zeros((plan_horizon+1, 1))
    historical_decision_table[0:len(prev_stage_calculation[-1])] = prev_stage_calculation[-1][:, -2].reshape(-1, 1)
    historical_value_table = np.zeros((plan_horizon+1, 1))
    historical_value_table[0:len(prev_stage_calculation[-1])] = prev_stage_calculation[-1][:, -1].reshape(-1, 1)
    historical_signals = [phase]
    flag = True
    s_start=G_min_T[phase] + Yellow + Red
    phase_sequence_index += 1
    offset = G_min_T[phase]
    while flag:
        phase = phase_sequence[phase_sequence_index % len(phase_sequence)]
        phase_loc = return_phase_loc(phase)
        all_zero_columns = np.all(Arrival_Table == 0, axis=0)
        if all_zero_columns[phase_loc].all():
            phase_sequence_index += 1
            continue

        s_start+=G_min_T[phase]+Yellow+Red
        historical_signals.append(phase)
        decision_table = np.zeros((plan_horizon+1, 1))
        value_table = np.zeros((plan_horizon+1, 1))

        prev_stage_calculation_previous = prev_stage_calculation.copy()
        for s in range(s_start, min(s_start+G_max_T-offset+1, 121)):
            x, v, stage, replacement = signal_DP_sup_1(s, phase, prev_stage_calculation_previous, G_min_T, G_max_T, Arrival_Table, s_start, offset)
            decision_table[s] = x
            value_table[s] = v
            if len(stage) <= len(prev_stage_calculation[-1]):
                prev_stage_calculation[s-Yellow-Red-offset] = stage.copy()
            else:
                prev_stage_calculation.append(stage.copy())
        historical_decision_table = np.concatenate((historical_decision_table, decision_table), axis=1)
        historical_value_table = np.concatenate((historical_value_table, value_table), axis=1)
        if s == 120:
            flag = False
        phase_sequence_index += 1
    return historical_decision_table, historical_value_table, historical_signals

def optimal_policy_generation(result, plan_horizon):
    stage_index = len(result[2])-1
    time_index = plan_horizon
    time_all = []
    for i in range(len(result[2])):
        time_temp = result[0][int(time_index), int(stage_index)]
        time_all.append(time_temp)
        stage_index -= 1
        if time_temp == 0:
            time_index = max(time_index-time_temp, 0)
        else:
            time_index = max(time_index-time_temp-5, 0)
    time_all.reverse()
    signal_time = [[a, b*10] for a, b in zip(result[2], time_all)]
    flag = True
    while flag:
        if signal_time[0][0] != signal_time[1][0]:
            flag = False
        else:
            signal_time[1][1] += signal_time[0][1]
            signal_time.pop(0)
            if signal_time[0][1] >= 400:
                signal_time[0][1] = 400
                flag = False
    new_list = []
    for i in range(len(signal_time)):
        if signal_time[i][1] != 0:
            signal_return = return_signal(signal_time[i][0])
            new_list.append([signal_return[0], signal_time[i][1]])
            new_list.append([signal_return[1], 40])
            new_list.append([signal_return[2], 10])
    return new_list

# =============================================================================
# Main
# =============================================================================
if __name__=='__main__':

    if 'SUMO_HOME' in os.environ:
        sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

    # -----------------------  LoS-Graphormer PARAMETERS  --------------------------
    MODEL_PATH = '../../trained_model/nuscenes_los_graphormer_8647.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _gnn = LoSGraphormer(
        in_feats=5,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.2,
        max_len=32,
        num_classes=2
    ).to(device)
    if os.path.isfile(MODEL_PATH):
        _gnn.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    _gnn.eval()

    VIS_GRAPH = False
    OCCLUDED_THRESH = 0.315 # a vehicle is considered observed if P(occ) < OCCLUDED_THRESH.

    # -----------------------  CAV PARAMETERS  --------------------------------
    PENETRATION_RATE  = 0.03
    PERCEPTION_RANGE  = 54.0
    RANDOM_SEED       = 11
    random.seed(RANDOM_SEED)

    cav_flag: dict[str, bool] = {}

    # -----------------------  SUMO / ISIG PARAMETERS  --------------------------
    traffic_light_id = '79'
    junction_id = "79"
    G_min_T = {'012': 5, '34': 5, '567': 5, '8910': 5}
    G_max_T = 40
    Yellow = 4
    Red = 1
    depart_lane_list = ['173166881_0','173166881_1','173166881_2',
                        'E0_0','E0_1',
                        '174262747_0','174262747_1','174262747_2',
                        '173896171_0','173896171_1','173896171_2',
                        '655620659_0','655620659_1','655620659_2','655620659_3']
    intersection_center = [238.96, 255.79]
    plan_horizon = 120
    approaches = 11
    departure_rate = 0.5
    phase_sequence = ['8910', '34', '012', '567']

    sumoCmd = ["sumo-gui", "-c", "./osm.sumocfg"]
    traci.start(sumoCmd)

    routes = {}
    flag = False
    signal_list = []
    Arrival_Table_list = []
    result_list = []
    all_gnn_time = []
    marker = 1000

    for step in range(36000):  # stepwidth=0.1
        print(step)
        traci.simulationStep()

        # -------------------  Visibility with LoS-Graphormer  ----------------
        all_ids = traci.vehicle.getIDList()
        update_CAV_flags(all_ids)
        veh_id_list, gnn_time_list = get_observed_vehicle_ids(all_ids)
        all_gnn_time += gnn_time_list

        # Colouring in SUMO GUI
        observed_set = set(veh_id_list)
        for vid in all_ids:
            if cav_flag[vid]:
                traci.vehicle.setColor(vid, (255, 0, 0, 255))      # red = CAV
            elif vid in observed_set:
                traci.vehicle.setColor(vid, (0, 0, 255, 255))      # blue = observed
            else:
                traci.vehicle.setColor(vid, (255, 255, 255, 255))  # light gray

        # ------------ route history for ISIG -------------
        if step == 1000:
            flag = True
        if step % 10 == 0:
            for veh_id in veh_id_list:
                if veh_id not in routes:
                    routes[veh_id] = []
                lane_id = traci.vehicle.getLaneID(veh_id)
                if lane_id in depart_lane_list or lane_id[:3] == ':79':
                    if veh_id in routes:
                        del routes[veh_id]
                else:
                    routes[veh_id].append(lane_id)

        # ------------- ISIG scheduling -------------------
        if flag and step == marker:
            Arrival_Table = np.zeros((plan_horizon+1, approaches))
            for veh_id in routes:
                if veh_id in veh_id_list:
                    loc = return_ETA_cell(veh_id, intersection_center, plan_horizon, routes)
                    if loc:
                        Arrival_Table[loc[0], loc[1]] += 1
            Arrival_Table_list.append(Arrival_Table)
            if np.all(Arrival_Table==0):
                signal=[[return_signal(phase_sequence[0])[0],G_min_T[phase_sequence[0]]*10],
                        [return_signal(phase_sequence[0])[0]+1,40],
                        [return_signal(phase_sequence[0])[0]+2,10]]
            else:
                result = signal_DP(Arrival_Table, G_min_T, G_max_T, phase_sequence, plan_horizon, approaches, departure_rate)
                result_list.append(result)
                signal = optimal_policy_generation(result, plan_horizon)
            print(signal)
            for i in range(3):
                marker += signal[i][1]
            index=phase_sequence.index(return_phase_char(signal[0][0]))
            phase_sequence = phase_sequence[index+1:]+phase_sequence[:index+1]
            signal_list.append(signal)

            time_list = [signal[0][1]+step]
            for i in range(1, len(signal)):
                time_list.append(signal[i][1]+time_list[i-1])

        if flag:
            for i in range(len(time_list)):
                if i == 0 and step < time_list[i]:
                    traci.trafficlight.setPhase(traffic_light_id, signal[i][0])
                elif step >= time_list[i-1] and step < time_list[i]:
                    traci.trafficlight.setPhase(traffic_light_id, signal[i][0])

    print(f"Avg per-CAV per-frame inference time: {np.mean(all_gnn_time)*1000:.4f} ms")

    traci.close()
