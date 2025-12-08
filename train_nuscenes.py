#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import random
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

import torch

from los_graphormer import (
    RMLoSDataset,
    LoSGraphormer,
    MLPBaseline,
    GCNOnChains,
    LoSVanillaTransformer,
    FocalLoss,
    EarlyStopping,
    train_epoch,
    evaluate,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train MIDAR to mimic CenterPoint detections on nuScenes dataset."
    )
    parser.add_argument("--csv-path", type=str, required=True,
                        help="Path to nuScenes CSV file.")
    parser.add_argument("--use-ray-hit", action="store_true",
                        help="Use 5F [dist, ray_hit, w, l, h]. "
                             "If not set, use 4F [dist, w, l, h].")
    parser.add_argument("--ray-width", type=float, default=1.0)

    parser.add_argument(
        "--model-type",
        type=str,
        default="los_graphormer",
        choices=["los_graphormer", "mlp", "gcn", "vanilla"],
        help="Which model to train."
    )

    # model hyperparams
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-len", type=int, default=32)

    parser.add_argument("--mlp-hidden", type=int, default=128)
    parser.add_argument("--gcn-hidden", type=int, default=64)
    parser.add_argument("--gcn-layers", type=int, default=2)

    # training hyperparams
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--test-size", type=float, default=0.15,
                        help="Fraction of scenes for test.")
    parser.add_argument("--val-size", type=float, default=0.1765, #0.85*0.1765=0.15
                        help="Fraction of trainval scenes for validation.")
    parser.add_argument("--ckpt-path", type=str,
                        default="./trained_model/nuscenes_losgraphormer.pth")

    args = parser.parse_args()

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # dataset
    ds = RMLoSDataset(
        args.csv_path,
        ray_width=args.ray_width,
        use_ray_hit=args.use_ray_hit,
        dist_scale=60.0,
    )

    scene_indices = [meta["scene_index"] for meta in ds.frame_metadata]
    unique_scenes = sorted(set(scene_indices))

    trainval_scenes, test_scenes = train_test_split(
        unique_scenes, test_size=args.test_size, random_state=args.seed
    )
    train_scenes, val_scenes = train_test_split(
        trainval_scenes, test_size=args.val_size, random_state=args.seed
    )

    train_indices = [i for i, s in enumerate(scene_indices) if s in train_scenes]
    val_indices = [i for i, s in enumerate(scene_indices) if s in val_scenes]
    test_indices = [i for i, s in enumerate(scene_indices) if s in test_scenes]

    train_ds = Subset(ds, train_indices)
    val_ds = Subset(ds, val_indices)
    test_ds = Subset(ds, test_indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # class imbalance
    df = pd.read_csv(args.csv_path)
    pos_frac = (df["occluded"] == 1).mean()
    neg_frac = 1.0 - pos_frac
    print("pos_frac:", pos_frac, "neg_frac:", neg_frac)

    alpha = torch.tensor([neg_frac, pos_frac], dtype=torch.float32)
    criterion = FocalLoss(gamma=2.0, alpha=alpha)

    in_feats = 5 if args.use_ray_hit else 4

    if args.model_type == "los_graphormer":
        model = LoSGraphormer(
            in_feats=in_feats,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_ff,
            dropout=args.dropout,
            max_len=args.max_len,
            num_classes=2,
        ).to(device)

    elif args.model_type == "mlp":
        model = MLPBaseline(
            in_feats=in_feats,
            hidden=args.mlp_hidden,
            num_classes=2,
            dropout=args.dropout,
        ).to(device)

    elif args.model_type == "gcn":
        model = GCNOnChains(
            in_feats=in_feats,
            hidden=args.gcn_hidden,
            num_layers=args.gcn_layers,
            num_classes=2,
            dropout=args.dropout,
        ).to(device)

    elif args.model_type == "vanilla":
        model = LoSVanillaTransformer(
            in_feats=in_feats,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_ff,
            dropout=args.dropout,
            max_len=args.max_len,
            num_classes=2,
        ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
    )

    from torch.optim.lr_scheduler import LambdaLR
    warmup_steps = args.warmup_steps

    def lr_lambda(step):
        return float(step + 1) / warmup_steps if step < warmup_steps else 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)

    stopper = EarlyStopping(
        patience=args.patience,
        min_delta=1e-4,
        verbose=True,
        path=args.ckpt_path,
    )

    best_val_metric = -1.0
    for epoch in range(1, args.max_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device, scheduler=scheduler)
        (val_loss, val_acc,
         val_prec, val_rec, val_f1, val_auc,
         val_bprec, val_brec, val_bf1, val_bthr) = evaluate(model, val_loader, criterion, device)

        metric = val_auc if not math.isnan(val_auc) else val_f1
        best_val_metric = max(best_val_metric, metric)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}, val_f1@0.5={val_f1:.4f}, val_auc={val_auc:.4f}, "
            f"best_f1={val_bf1:.4f} @ thr={val_bthr:.3f}"
        )

        stopper(current_score=metric, model=model)
        if stopper.early_stop:
            print(f"Early stop at epoch {epoch}, best val metric={best_val_metric:.4f}")
            break

    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    (te_loss, te_acc,
     te_prec, te_rec, te_f1, te_auc,
     te_bprec, te_brec, te_bf1, te_bthr) = evaluate(model, test_loader, criterion, device)
    print(
        f"[{args.model_type}] TEST: "
        f"loss={te_loss:.4f}, acc={te_acc:.4f}, "
        f"prec@0.5={te_prec:.4f}, rec@0.5={te_rec:.4f}, f1@0.5={te_f1:.4f}, auc={te_auc:.4f}\n"
        f"best_f1={te_bf1:.4f} @ thr={te_bthr:.3f} "
        f"(prec={te_bprec:.4f}, rec={te_brec:.4f})"
    )


if __name__ == "__main__":
    main()
