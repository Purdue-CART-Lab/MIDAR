import math
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha, dtype=torch.float32) if alpha is not None else None
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits,
            target,
            weight=(self.alpha.to(logits.device) if self.alpha is not None else None),
            ignore_index=self.ignore_index,
            reduction='none'
        )
        pt = torch.exp(-ce)
        fl = (1 - pt) ** self.gamma * ce
        if self.reduction == 'mean':
            return fl.mean()
        if self.reduction == 'sum':
            return fl.sum()
        return fl


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score, model):
        if self.best_score is None:
            self.best_score = current_score
            self._save_checkpoint(model)
        elif current_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self._save_checkpoint(model)
            self.counter = 0

    def _save_checkpoint(self, model):
        if self.verbose:
            print(f"Validation metric improved to {self.best_score:.4f}. Saved → {self.path}")
        import os
        ckpt_dir = os.path.dirname(self.path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), self.path)


def train_epoch(model, model_type, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss, total_correct, total_items = 0.0, 0, 0

    for data in loader:
        data = data.to(device)
        data.x = normalize_batch(data.x).clamp(-5, 5)

        optimizer.zero_grad()
        out = model(data)

        # unify logits
        if isinstance(out, tuple):
            # Graphormer / vanilla TF: (logits, pad_mask)
            logits, _ = out
        else:
            logits = out

        # select last-token logits depending on model type
        if model_type == "mlp" and logits.dim() == 2:
            # MLP: logits already (B, C) = one per chain
            last_logits = logits

        elif model_type == "gcn" and logits.dim() == 2:
            # GCN: logits per node (N_nodes, C)
            node_logits = logits
            feats = []
            for c in data.chains:
                last_idx = c[-1].item()
                feats.append(node_logits[last_idx])
            last_logits = torch.stack(feats, dim=0)  # (B, C)

        else:
            # Graphormer / vanilla TF: logits (B, S, C)
            B, S, C = logits.shape
            last_pos = torch.tensor(
                [c.numel() - 1 for c in data.chains],
                device=logits.device,
                dtype=torch.long
            )
            last_logits = logits[torch.arange(B, device=logits.device), last_pos, :]  # (B, C)

        # targets = last label of each chain
        last_targets = torch.tensor(
            [t[-1].item() for t in data.seq_targets],
            device=last_logits.device,
            dtype=torch.long
        )

        loss = criterion(last_logits, last_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            pred = last_logits.argmax(-1)
            total_correct += (pred == last_targets).sum().item()
            total_items   += last_targets.numel()
            total_loss    += loss.item() * last_targets.numel()

    mean_loss = total_loss / max(total_items, 1)
    acc       = total_correct / max(total_items, 1)
    return mean_loss, acc

@torch.no_grad()
def evaluate(model, model_type, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_items = 0.0, 0, 0
    all_preds, all_probs, all_true = [], [], []

    for data in loader:
        data = data.to(device)
        data.x = normalize_batch(data.x).clamp(-5, 5)

        out = model(data)
        if isinstance(out, tuple):
            logits, _ = out
        else:
            logits = out

        # ---- select last logits ----
        if model_type == "mlp" and logits.dim() == 2:
            last_logits = logits

        elif model_type == "gcn" and logits.dim() == 2:
            node_logits = logits
            feats = []
            for c in data.chains:
                last_idx = c[-1].item()
                feats.append(node_logits[last_idx])
            last_logits = torch.stack(feats, dim=0)  # (B, C)

        else:
            B, S, C = logits.shape
            last_pos = torch.tensor(
                [c.numel() - 1 for c in data.chains],
                device=logits.device,
                dtype=torch.long
            )
            last_logits = logits[torch.arange(B, device=logits.device), last_pos, :]  # (B, C)

        # ---- targets ----
        last_targets = torch.tensor(
            [t[-1].item() for t in data.seq_targets],
            device=last_logits.device,
            dtype=torch.long
        )

        # ---- loss ----
        loss = criterion(last_logits, last_targets)

        probs = F.softmax(last_logits, dim=-1)[:, 1]
        preds = last_logits.argmax(-1)

        total_correct += (preds == last_targets).sum().item()
        total_items   += last_targets.numel()
        total_loss    += loss.item() * last_targets.numel()

        all_preds.append(preds.flatten().cpu())
        all_probs.append(probs.flatten().cpu())
        all_true.append(last_targets.flatten().cpu())

    if total_items == 0:
        # loss, acc@0.5, prec@0.5, rec@0.5, f1@0.5, auc,
        # best_prec, best_rec, best_f1, best_thr, acc_at_best_f1
        return (0.0, 0.0,
                0.0, 0.0, 0.0, float('nan'),
                0.0, 0.0, 0.0, 0.5, 0.0)

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    # ---- fixed-threshold (0.5) metrics ----
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')

    # ---- sweep thresholds to maximize F1, and record acc at that thr ----
    thresholds = np.linspace(0.0, 1.0, 201)
    best_f1, best_thr = -1.0, 0.5
    best_prec, best_rec = 0.0, 0.0
    acc_at_best_f1 = 0.0

    for thr in thresholds:
        y_pred_thr = (y_prob >= thr).astype(int)

        # metrics at this thr
        p = precision_score(y_true, y_pred_thr, zero_division=0)
        r = recall_score(y_true, y_pred_thr, zero_division=0)
        f = f1_score(y_true, y_pred_thr, zero_division=0)
        acc_t = (y_pred_thr == y_true).mean()

        if f > best_f1 + 1e-8:
            best_f1 = float(f)
            best_thr = float(thr)
            best_prec = float(p)
            best_rec = float(r)
            acc_at_best_f1 = float(acc_t)

    acc_05 = total_correct / max(total_items, 1)

    return (
        total_loss / max(total_items, 1),   # mean loss
        acc_05,                             # accuracy at 0.5
        precision, recall, f1, auc,
        best_prec, best_rec, best_f1, best_thr,
        acc_at_best_f1,                     # accuracy at best-F1 threshold
    )

