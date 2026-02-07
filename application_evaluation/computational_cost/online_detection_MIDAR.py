#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detection-only MIDAR (LoS-Graphormer) online pipeline in SUMO.

- No signal control logic.
- Deterministic CAV selection:
    type == 'vehicle.lincoln.mkz_2017'
    AND current edge in {'40','1185','1184','39'}
- SUMO config (*.sumocfg) is expected in the same folder as this script unless
  a path is provided.

Output per tick:
- observed vehicle IDs (union across all active CAVs in this tick)
- timing stats
"""

import os
import sys
import time
import math
import argparse
from typing import Any, Optional

import numpy as np
import traci

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from shapely.geometry import LineString, Point
from shapely.geometry import Polygon as ShpPolygon

import csv
import psutil
from pynvml import (
    nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetComputeRunningProcesses_v2
)

sys.path.append('../../los_graphormer/models')
from los_graphormer import LoSGraphormer


def find_pid_by_exact_process_name(name: str):
    target = name.lower()
    for p in psutil.process_iter(attrs=["pid", "name"]):
        try:
            if (p.info["name"] or "").lower() == target:
                return p.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


# =============================================================================
# Deterministic CAV selection
# =============================================================================
CAV_TYPE_ID = "vehicle.lincoln.mkz_2017"
CAV_EDGE_PREFIXES = {"40.0.00", "39.0.00", "-40.0.00", "-39.0.00", ":669_6",":669_1",":720_1",":720_6"}  # adjust to your real ones

def is_cav_now(veh_id: str) -> bool:
    """
    Detect CAV if:
      1) type matches CAV_TYPE_ID
      2) edge_id (derived from lane_id) is in CAV_EDGE_PREFIXES

    Works for:
      - normal lanes:  "-40.0.00_3"   -> edge "-40.0.00"
      - internal lanes:":669_6_1"     -> edge ":669_6"
    """
    try:
        if traci.vehicle.getTypeID(veh_id) != CAV_TYPE_ID:
            return False

        lane_id = traci.vehicle.getLaneID(veh_id)

        # If lane_id has a lane index suffix, strip it; otherwise keep as-is
        # "-40.0.00_3" -> "-40.0.00"
        # ":669_6_1"   -> ":669_6"
        # ":669_6"     -> ":669_6" (already an edge-like id)
        edge_id = lane_id.rsplit("_", 1)[0] if "_" in lane_id else lane_id

        return edge_id in CAV_EDGE_PREFIXES

    except traci.TraCIException:
        return False




# =============================================================================
# MIDAR / LoS-Graphormer helpers (copied/cleaned from your script)
# =============================================================================
def _normalize_batch(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def _vehicle_bbox_polygon(x: float, y: float, w: float, l: float, yaw: float) -> ShpPolygon:
    """
    Create an oriented rectangle polygon centered at (x,y) with width w and length l.
    yaw in radians.
    """
    # corners in vehicle frame (length along x, width along y)
    dx = l / 2.0
    dy = w / 2.0
    corners = np.array([
        [ dx,  dy],
        [ dx, -dy],
        [-dx, -dy],
        [-dx,  dy],
    ], dtype=np.float64)

    c = math.cos(yaw)
    s = math.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    rot = corners @ R.T
    rot[:, 0] += x
    rot[:, 1] += y
    return ShpPolygon(rot)

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

def get_observed_vehicle_ids(
    veh_ids: list[str],
    gnn: torch.nn.Module,
    device: torch.device,
    perception_range: float,
    occluded_thresh: float,
) -> tuple[list[str], list[float], list[str]]:
    """
    Returns:
      observed_ids: vehicles considered observed by at least one active CAV this tick
      per_cav_times: list of per-CAV inference times (seconds)
      cavs_now: list of active CAV IDs this tick
    """
    if not veh_ids:
        return [], [], []

    # Bulk pulls
    positions = {v: traci.vehicle.getPosition(v) for v in veh_ids}
    headings = {v: math.radians(traci.vehicle.getAngle(v)) for v in veh_ids}

    dims = {}
    
    for v in veh_ids:
        try:
            dims[v] = (traci.vehicle.getWidth(v),
                       traci.vehicle.getLength(v),
                       traci.vehicle.getHeight(v))
        except Exception:
            # fallback
            dims[v] = (traci.vehicle.getWidth(v), traci.vehicle.getLength(v), 1.5)

    cavs_now = [v for v in veh_ids if is_cav_now(v)]
    observed: set[str] = set()
    per_cav_times: list[float] = []

    for cav in cavs_now:
        cx, cy = positions[cav]
        neigh = [
            v for v in veh_ids
            if v != cav and math.hypot(positions[v][0] - cx, positions[v][1] - cy) <= perception_range
        ]
        # If no neighbors, at least ego is "observed"
        if not neigh:
            observed.add(cav)
            continue

        start = time.perf_counter()
        data = _build_los_frame_for_cav(cav, neigh, positions, dims, headings)
        p_occ = _predict_visibility_graphormer(gnn, data, device)
        per_cav_times.append(time.perf_counter() - start)

        # Node mapping: 0=ego, 1..N correspond to neigh in order
        for i in range(1, len(neigh) + 1):
            if (not np.isnan(p_occ[i])) and (p_occ[i] < occluded_thresh):
                observed.add(neigh[i - 1])

        observed.add(cav)

    return list(observed), per_cav_times, cavs_now

class TickResourceMonitor:
    """
    Logs per-tick CPU/RAM for selected PIDs + GPU util/VRAM (global + per-PID if possible).
    """
    def __init__(self, csv_path: str, pids: dict, gpu_index: int = 0):
        """
        pids: dict like {"python": <pid>, "sumo": <pid or None>, "carla": <pid or None>, ...}
        """
        self.csv_path = csv_path
        self.pids = {k: v for k, v in pids.items() if v is not None}
        self.procs = {k: psutil.Process(pid) for k, pid in self.pids.items()}
        self.gpu_ok = False
        self.gpu_index = gpu_index
        self.gpu_handle = None

        # Prime cpu_percent counters (psutil needs an initial call)
        for p in self.procs.values():
            try:
                p.cpu_percent(interval=None)
            except Exception:
                pass

        # GPU init (optional)
        try:
            nvmlInit()
            self.gpu_handle = nvmlDeviceGetHandleByIndex(gpu_index)
            self.gpu_ok = True
        except Exception:
            self.gpu_ok = False

        self.f = open(self.csv_path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow([
            "wall_time_s", "step", "n_cav", "n_observed",
            "proc", "pid", "cpu_percent", "rss_mb",
            "gpu_global_util_percent", "gpu_global_vram_used_mb", "gpu_global_vram_total_mb",
            "gpu_proc_vram_mb"
        ])

    def close(self):
        try:
            self.f.close()
        finally:
            if self.gpu_ok:
                try:
                    nvmlShutdown()
                except Exception:
                    pass

    def _gpu_global(self):
        if not self.gpu_ok:
            return (None, None, None)
        try:
            util = nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu  # %
            mem = nvmlDeviceGetMemoryInfo(self.gpu_handle)
            used_mb = mem.used / (1024**2)
            total_mb = mem.total / (1024**2)
            return (float(util), float(used_mb), float(total_mb))
        except Exception:
            return (None, None, None)

    def _gpu_vram_by_pid(self):
        """
        Returns dict pid->vram_mb for running compute processes (best-effort).
        Note: graphics processes may not show up here; it’s still useful for PyTorch.
        """
        out = {}
        if not self.gpu_ok:
            return out
        try:
            procs = nvmlDeviceGetComputeRunningProcesses_v2(self.gpu_handle)
            for p in procs:
                # p.pid, p.usedGpuMemory (bytes or NVML_VALUE_NOT_AVAILABLE)
                used = getattr(p, "usedGpuMemory", None)
                if used is None or used < 0:
                    continue
                out[int(p.pid)] = float(used) / (1024**2)
        except Exception:
            pass
        return out

    def log_tick(self, *, step: int, wall_time_s: float, n_cav: int = 0, n_observed: int = 0):
        gpu_util, gpu_used_mb, gpu_total_mb = self._gpu_global()
        vram_by_pid = self._gpu_vram_by_pid()

        for name, proc in self.procs.items():
            try:
                cpu = proc.cpu_percent(interval=None)  # percent since last call
                rss_mb = proc.memory_info().rss / (1024**2)
            except Exception:
                cpu, rss_mb = None, None

            pid = proc.pid
            gpu_proc_vram_mb = vram_by_pid.get(pid, None)

            self.w.writerow([
                wall_time_s, step, n_cav, n_observed,
                name, pid, cpu, rss_mb,
                gpu_util, gpu_used_mb, gpu_total_mb,
                gpu_proc_vram_mb
            ])
        self.f.flush()

# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sumocfg", nargs="?", default=None,
                        help="SUMO config (.sumocfg). If omitted, uses a .sumocfg in the script folder.")
    parser.add_argument("--sumo-gui", action="store_true", help="Use sumo-gui instead of sumo.")
    parser.add_argument("--steps", type=int, default=18000, help="Number of simulation steps.")
    parser.add_argument("--step-length", type=float, default=0.05, help="SUMO step-length if you want to set it.")
    parser.add_argument("--perception-range", type=float, default=80.0, help="Perception range (m).")
    parser.add_argument("--occ-thresh", type=float, default=0.22167, help="Observed if p(occ) < threshold.")
    parser.add_argument("--model-path", type=str, default="../../trained_model/carla_los_graphormer_9385.pth",
                        help="LoSGraphormer checkpoint path.")
    parser.add_argument("--util-log", type=str, default=None,
                    help="If set, write per-tick CPU/GPU utilization CSV to this path.")
    parser.add_argument("--gpu-index", type=int, default=0,
                        help="GPU index for NVML (default 0).")
    
    parser.add_argument("--track-procname", type=str, default="sumo:sumo-gui",
                help='Comma list like "sumo:sumo-gui"')


    args = parser.parse_args()

    this_dir = os.path.dirname(os.path.abspath(__file__))

    # Pick sumocfg
    if args.sumocfg is None:
        # choose the first *.sumocfg in the same folder if user didn’t pass one
        candidates = [f for f in os.listdir(this_dir) if f.endswith(".sumocfg")]
        if not candidates:
            raise FileNotFoundError(f"No .sumocfg found in {this_dir} and none provided.")
        sumocfg_path = os.path.join(this_dir, candidates[0])
    else:
        sumocfg_path = args.sumocfg
        if not os.path.isabs(sumocfg_path):
            sumocfg_path = os.path.join(this_dir, sumocfg_path)

    # Build MIDAR model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn = LoSGraphormer(
        in_feats=5,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.2,
        max_len=32,
        num_classes=2
    ).to(device)
    if os.path.isfile(args.model_path):
        gnn.load_state_dict(torch.load(args.model_path, map_location=device))
    gnn.eval()

    # Warmup (optional but recommended)
    # If you have a cached frame you can warmup; here we skip since inputs depend on SUMO state.

    # Start SUMO
    sumo_bin = "sumo-gui"
    cmd = [sumo_bin, "-c", sumocfg_path]
    if args.step_length is not None:
        cmd += ["--step-length", str(args.step_length)]

    traci.start(cmd)

    all_per_cav_times = []
    all_tick_times = []
    all_cav_counts = []
    
    monitor = None
    if args.util_log is not None:
        pids = {"python": os.getpid()}
        
        # Resolve processes by name substring (carla / sumo)
        if args.track_procname.strip():
            for token in args.track_procname.split(","):
                token = token.strip()
                if not token or ":" not in token:
                    continue
                label, pattern = token.split(":", 1)
                label = label.strip()
                pattern = pattern.strip()
        
                pid = find_pid_by_exact_process_name(pattern)
                if pid is not None:
                    pids[label] = pid
                else:
                    # Keep it missing for now; monitor can log blanks instead of crashing
                    pids[label] = None
    
        monitor = TickResourceMonitor(
            csv_path=args.util_log,
            pids=pids,
            gpu_index=args.gpu_index
        )
    

    try:
        for step in range(args.steps):
            t0 = time.perf_counter()
            traci.simulationStep()

            veh_ids = traci.vehicle.getIDList()
            observed_ids, per_cav_times, cavs_now = get_observed_vehicle_ids(
                veh_ids=veh_ids,
                gnn=gnn,
                device=device,
                perception_range=args.perception_range,
                occluded_thresh=args.occ_thresh,
            )

            monitor.log_tick(
                step=step,
                wall_time_s=time.time(),
                n_cav=len(cavs_now),
                n_observed=len(observed_ids)
            )

            tick_s = time.perf_counter() - t0
            all_tick_times.append(tick_s)
            all_per_cav_times.extend(per_cav_times)
            all_cav_counts.append(len(cavs_now))

            # Minimal logging (edit as you like)
            if step % 10 == 0:
                tick_ms = tick_s * 1000.0
                cav_n = len(cavs_now)
                obs_n = len(observed_ids)
                avg_cav_ms = (np.mean(per_cav_times) * 1000.0) if per_cav_times else 0.0
                print(f"[step {step:05d}] tick_ms={tick_ms:7.2f}  cavs={cav_n:2d}  observed={obs_n:3d}  avg_cav_midar_ms={avg_cav_ms:7.3f}")

        # Summary
        print("\n=== Summary ===")
        if all_per_cav_times:
            print(f"Avg per-CAV MIDAR time: {np.mean(all_per_cav_times)*1000:.3f} ms")
            print(f"P95 per-CAV MIDAR time: {np.percentile(all_per_cav_times, 95)*1000:.3f} ms")
        print(f"Avg tick time: {np.mean(all_tick_times)*1000:.3f} ms")
        print(f"P95 tick time: {np.percentile(all_tick_times, 95)*1000:.3f} ms")
        print(f"Avg #CAVs per tick: {np.mean(all_cav_counts):.3f}")

    finally:
        traci.close()
        if monitor is not None:
            monitor.close()

if __name__ == "__main__":
    main()
    
    
'''
python online_detection_MIDAR.py Town04_highway_bn_6000.sumocfg --steps 18000 --util-log util_10%.csv
'''
