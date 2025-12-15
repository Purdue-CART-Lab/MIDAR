#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import argparse
import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt


# --------------------------- helpers ---------------------------

def _load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def _to_1d_float_array(x, name="arr", expected_len=None):
    arr = np.asarray(x, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    if expected_len is not None and arr.size != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got {arr.size}")
    return arr

def _extract_case_ids(obj):
    """
    Try common names. Return None if not found.
    """
    if isinstance(obj, dict):
        for k in ["case_ids", "case_id_list", "case_list", "case_number_list", "case_numbers", "case_idx"]:
            if k in obj:
                return list(obj[k])
        # nested
        for kk in ["meta", "config", "results"]:
            if kk in obj and isinstance(obj[kk], dict):
                for k in ["case_ids", "case_number_list", "case_numbers"]:
                    if k in obj[kk]:
                        return list(obj[kk][k])
    return None

def align_by_case_ids(series_dict):
    """
    series_dict: {name: (case_ids, values)}
    Returns:
      common_ids, aligned_values_dict
    """
    # intersection of all non-None id sets
    id_sets = []
    for name, (ids, vals) in series_dict.items():
        if ids is None:
            raise ValueError(f"{name} has no case_ids stored in pkl. Can't align safely by IDs.")
        id_sets.append(set(ids))

    common = sorted(set.intersection(*id_sets))
    if len(common) == 0:
        raise ValueError("No common cases across pkls.")

    aligned = {}
    for name, (ids, vals) in series_dict.items():
        id_to_val = {i: v for i, v in zip(ids, vals)}
        aligned[name] = np.array([id_to_val[i] for i in common], dtype=float)

    return common, aligned


def _extract_case_metric(obj, metric_key: str):
    """
    Tries common structures:
      A) dict with keys like 'mae_list', 'rmse_list', 'mape_list'
      B) tuple where the second / fourth / sixth element are list outputs
         (your earlier function returns overall_mae, mae_list, overall_mape, mape_list, overall_rmse, rmse_list, ...)
    """
    if isinstance(obj, dict):
        # expected: "mae_list" / "rmse_list" / "mape_list"
        if metric_key in obj:
            return obj[metric_key]
        # sometimes nested
        for k in ["lists", "metrics", "results"]:
            if k in obj and isinstance(obj[k], dict) and metric_key in obj[k]:
                return obj[k][metric_key]
        raise KeyError(f"Cannot find '{metric_key}' in dict keys: {list(obj.keys())[:30]}")

    if isinstance(obj, (list, tuple)):
        # Your earlier assignment:
        # overall_mae, mae_list, overall_mape, mape_list, overall_rmse, rmse_list, overall_mae_l, mae_l_list = ...
        # So: mae_list at index 1, mape_list at index 3, rmse_list at index 5, mae_l_list at index 7
        lookup = {
            "mae_list": 1,
            "mape_list": 3,
            "rmse_list": 5,
            "mae_l_list": 7,
        }
        if metric_key not in lookup:
            raise KeyError(f"metric_key '{metric_key}' not supported for tuple parsing.")
        idx = lookup[metric_key]
        if len(obj) <= idx:
            raise ValueError(f"Tuple/list too short ({len(obj)}) to get {metric_key} at index {idx}.")
        return obj[idx]

    raise TypeError(f"Unsupported pkl object type: {type(obj)}")

def paired_tests(x_ref, x_cmp, ref_name, cmp_name):
    """
    Paired t-test + effect sizes.
    """
    d = x_cmp - x_ref
    n = len(d)

    tstat, p = stats.ttest_rel(x_cmp, x_ref, nan_policy="raise")

    # Cohen's dz for paired design: mean(diff)/std(diff)
    dz = float(np.mean(d) / (np.std(d, ddof=1) + 1e-12))

    # 95% CI for mean difference using t interval
    md = float(np.mean(d))
    se = float(stats.sem(d))
    ci = stats.t.interval(0.95, df=n-1, loc=md, scale=se)

    return {
        "ref": ref_name,
        "cmp": cmp_name,
        "n": n,
        "mean_ref": float(np.mean(x_ref)),
        "mean_cmp": float(np.mean(x_cmp)),
        "mean_diff_cmp_minus_ref": md,
        "t": float(tstat),
        "p": float(p),
        "cohens_dz": dz,
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
    }

def plot_case_bars_3rows(case_idx, series_dict, ylabel, outbase, title=None):
    """
    39 cases -> 3 rows x 13 cases.
    series_dict: {label: np.ndarray of length 39}
    """
    import numpy as np
    import matplotlib.pyplot as plt

    labels = list(series_dict.keys())
    Y = np.vstack([series_dict[k] for k in labels])  # (M, L)
    M, L = Y.shape
    assert L == 39, f"Expected 39 cases, got {L}"

    fig, axes = plt.subplots(
        3, 1, figsize=(7.2, 5.6), sharey=True
    )

    # global y-limits (same across rows)
    y_max = np.max(Y) * 1.08

    # bar geometry
    group_w = 0.86
    bar_w = group_w / M
    offsets = (np.arange(M) - (M - 1) / 2.0) * bar_w

    for r in range(3):
        ax = axes[r]
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=8, length=3, width=0.8)

        start = r * 13
        end = start + 13

        x = np.arange(13)  # local positions for this row

        for i, lab in enumerate(labels):
            ax.bar(
                x + offsets[i],
                Y[i, start:end],
                width=bar_w * 0.95,
                linewidth=0,
                label=lab if r == 0 else None  # legend only once
            )

        ax.set_ylim(0, y_max)
        ax.set_xlim(-0.8, 12.8)

        # x ticks = case indices
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(c)) for c in case_idx[start:end]])

        if r == 1:
            ax.set_ylabel(ylabel, fontsize=9)

        # small row label (optional but helpful)
        ax.text(0.01, 0.92, f"Cases {int(case_idx[start])}–{int(case_idx[end-1])}",
                transform=ax.transAxes, fontsize=8, va="top")

    if title:
        fig.suptitle(title, fontsize=10, y=0.995)

    # shared legend at top, Nature-like (no box)
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center",
               ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.975))

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(
        outbase + ".pdf",
        bbox_inches="tight",
        dpi=600,              # high DPI for any rasterized elements
        backend="pdf"
    )
    fig.savefig(outbase + ".png", dpi=600, bbox_inches="tight")
    plt.close(fig)




def nature_axes(ax):
    # Nature-ish: clean, minimal ink, readable
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=9, length=3, width=0.8)
    ax.tick_params(axis="both", which="minor", labelsize=8, length=2, width=0.6)
    ax.grid(False)

def save_fig(fig, outpath_base):
    fig.savefig(outpath_base + ".pdf", bbox_inches="tight")
    fig.savefig(outpath_base + ".png", dpi=600, bbox_inches="tight")


# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--true", required=True, help="pkl for True LiDAR detection")
    ap.add_argument("--perfect", required=True, help="pkl for Perfect detection")
    ap.add_argument("--drop", required=True, help="pkl for Random dropout")
    ap.add_argument("--midar", required=True, help="pkl for MIDAR")
    ap.add_argument("--midar-norh", required=True, help="pkl for MIDAR_noRH")
    ap.add_argument("--metric", default="mae_list",
                    choices=["mae_list", "rmse_list", "mape_list", "mae_l_list"],
                    help="which per-case metric list to plot/test (length 42)")
    ap.add_argument("--outdir", default="./fig_out", help="output folder")
    ap.add_argument("--title", default="", help="optional figure title")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # load
    obj_true = _load_pkl(args.true)
    obj_perf = _load_pkl(args.perfect)
    obj_drop = _load_pkl(args.drop)
    obj_midar = _load_pkl(args.midar)
    obj_norh = _load_pkl(args.midar_norh)

    metric = args.metric

    # extract values
    v_true  = _extract_case_metric(obj_true, metric)
    v_perf  = _extract_case_metric(obj_perf, metric)
    v_drop  = _extract_case_metric(obj_drop, metric)
    v_midar = _extract_case_metric(obj_midar, metric)
    v_norh  = _extract_case_metric(obj_norh, metric)

    y_true  = _to_1d_float_array(v_true,  f"true/{metric}")
    y_perf  = _to_1d_float_array(v_perf,  f"perfect/{metric}")
    y_drop  = _to_1d_float_array(v_drop,  f"drop/{metric}")
    y_midar = _to_1d_float_array(v_midar, f"midar/{metric}")
    y_norh  = _to_1d_float_array(v_norh,  f"midar_noRH/{metric}")

    ids_true  = _extract_case_ids(obj_true)
    ids_perf  = _extract_case_ids(obj_perf)
    ids_drop  = _extract_case_ids(obj_drop)
    ids_midar = _extract_case_ids(obj_midar)
    ids_norh  = _extract_case_ids(obj_norh)

    # if case ids exist → align by ids (best)
    if all(x is not None for x in [ids_true, ids_perf, ids_drop, ids_midar, ids_norh]):
        common_ids, aligned = align_by_case_ids({
            "True LiDAR": (ids_true,  y_true),
            "Perfect":    (ids_perf,  y_perf),
            "Random Dropout": (ids_drop, y_drop),
            "MIDAR":      (ids_midar, y_midar),
            "MIDAR-noRH": (ids_norh,  y_norh),
        })
        y_true  = aligned["True LiDAR"]
        y_perf  = aligned["Perfect"]
        y_drop  = aligned["Random Dropout"]
        y_midar = aligned["MIDAR"]
        y_norh  = aligned["MIDAR-noRH"]
        case_idx = np.arange(1, len(common_ids) + 1)

        # helpful diagnostics
        print(f"[INFO] Align by case IDs: using {len(common_ids)} common cases.")
        print(f"[INFO] Common case IDs (first 10): {common_ids[:10]}")
    else:
        # fallback: align by min length (still valid pairing if lists are in same order)
        L = min(len(y_true), len(y_perf), len(y_drop), len(y_midar), len(y_norh))
        print(f"[WARN] No case IDs found in some pkls. Falling back to first {L} cases by order.")
        y_true, y_perf, y_drop, y_midar, y_norh = y_true[:L], y_perf[:L], y_drop[:L], y_midar[:L], y_norh[:L]
        case_idx = np.arange(1, L + 1)


    # ---------- plotting (Nature-friendly) ----------
    fig, ax = plt.subplots(figsize=(7.0, 2.6))  # wide, compact (works as 1-col+)
    nature_axes(ax)

    # Use grayscale + one accent is typical, but I won’t hardcode colors.
    # Matplotlib defaults are fine; Nature mainly cares about readability & export.
    ax.plot(case_idx, y_true, marker="o", markersize=2.5, linewidth=1.0, label="True LiDAR")
    ax.plot(case_idx, y_perf, marker="o", markersize=2.5, linewidth=1.0, label="Perfect")
    ax.plot(case_idx, y_drop, marker="o", markersize=2.5, linewidth=1.0, label="Random Dropout")
    ax.plot(case_idx, y_midar, marker="o", markersize=2.5, linewidth=1.0, label="MIDAR")
    ax.plot(case_idx, y_norh, marker="o", markersize=2.5, linewidth=1.0, label="MIDAR-noRH")

    ax.set_xlabel("Case index", fontsize=10)
    ax.set_ylabel(args.metric.replace("_list", "").upper(), fontsize=10)
    ax.set_xlim(1, 42)
    ax.set_xticks([1, 7, 14, 21, 28, 35, 42])

    if args.title:
        ax.set_title(args.title, fontsize=10)

    leg = ax.legend(frameon=False, fontsize=8, ncol=3, handlelength=2.0, columnspacing=1.2)
    fig.tight_layout()

    outbase = os.path.join(args.outdir, f"cases42_{args.metric}")
    save_fig(fig, outbase)

    # ---------- paired t-tests ----------
    # Choose your “reference”:
    # Common choice: compare each method vs True LiDAR
    ref_name = "True LiDAR"
    ref = y_true

    tests = []
    tests.append(paired_tests(ref, y_perf, ref_name, "Perfect"))
    tests.append(paired_tests(ref, y_drop, ref_name, "Random Dropout"))
    tests.append(paired_tests(ref, y_midar, ref_name, "MIDAR"))
    tests.append(paired_tests(ref, y_norh, ref_name, "MIDAR-noRH"))

    df = pd.DataFrame(tests)

    # multiple comparison correction (Holm is a good default; BH/FDR also common)
    reject, p_holm, _, _ = multipletests(df["p"].values, alpha=0.05, method="holm")
    df["p_holm"] = p_holm
    df["significant_holm_0p05"] = reject

    # Save table
    csv_path = os.path.join(args.outdir, f"paired_ttests_vs_true_{args.metric}.csv")
    df.to_csv(csv_path, index=False)

    # Print a compact summary
    print("\nPaired t-tests (cmp vs True LiDAR):")
    cols = ["cmp", "n", "mean_ref", "mean_cmp", "mean_diff_cmp_minus_ref", "t", "p", "p_holm", "cohens_dz", "significant_holm_0p05"]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df[cols].to_string(index=False))

    L = 39
    y_true  = y_true[:L]
    y_perf  = y_perf[:L]
    y_drop  = y_drop[:L]
    y_midar = y_midar[:L]
    y_norh  = y_norh[:L]
    case_idx = np.arange(1, L + 1)

    bar_data = {
        "True LiDAR": y_true,
        "Perfect": y_perf,
        "Random Dropout": y_drop,
        "MIDAR": y_midar,
        "MIDAR-noRH": y_norh,
    }

    plot_case_bars_3rows(
        case_idx,
        bar_data,
        ylabel=args.metric.replace("_list", "").upper(),
        outbase=os.path.join(args.outdir, f"bars_3rows_cases{L}_{args.metric}"),
        title=args.title
    )
    print(f"Saved table : {csv_path}")


if __name__ == "__main__":
    main()
'''
python evaluation.py \
  --true ./results/carla_true_evaluation_results.pkl \
  --perfect ./results/carla_Perfect_evaluation_results.pkl \
  --drop ./results/carla_Drop_evaluation_results.pkl \
  --midar ./results/carla_MIDAR_023_evaluation_results.pkl \
  --midar-norh ./results/carla_MIDAR_noRH_023_evaluation_results.pkl \
  --metric mae_list \
  --outdir ./evaluation_results \
  --title "MAE of trajectory reconstruction across all planning horizons"
'''