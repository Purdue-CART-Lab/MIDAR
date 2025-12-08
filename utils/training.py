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


def train_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss, total_correct, total_items = 0.0, 0, 0

    for data in loader:
        data = data.to(device)
        data.x = normalize_batch(data.x).clamp(-5, 5)

        optimizer.zero_grad()
        logits, _ = model(data)

        B, S, C = logits.shape
        last_pos = torch.tensor([c.numel() - 1 for c in data.chains],
                                device=logits.device, dtype=torch.long)
        last_logits = logits[torch.arange(B, device=logits.device), last_pos, :]
        last_targets = torch.tensor([t[-1].item() for t in data.seq_targets],
                                    device=logits.device, dtype=torch.long)

        loss = criterion(last_logits, last_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            pred = last_logits.argmax(-1)
            total_correct += (pred == last_targets).sum().item()
            total_items += last_targets.numel()
            total_loss += loss.item() * last_targets.numel()

    mean_loss = total_loss / max(total_items, 1)
    acc = total_correct / max(total_items, 1)
    return mean_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Returns:
      mean_loss, acc,
      precision@0.5, recall@0.5, f1@0.5, auc,
      best_precision, best_recall, best_f1, best_threshold
    """
    model.eval()
    total_loss, total_correct, total_items = 0.0, 0, 0
    all_preds, all_probs, all_true = [], [], []

    for data in loader:
        data = data.to(device)
        data.x = normalize_batch(data.x).clamp(-5, 5)
        logits, _ = model(data)

        B, S, C = logits.shape
        last_pos = torch.tensor([c.numel() - 1 for c in data.chains],
                                device=logits.device, dtype=torch.long)
        last_logits = logits[torch.arange(B, device=logits.device), last_pos, :]
        last_targets = torch.tensor([t[-1].item() for t in data.seq_targets],
                                    device=logits.device, dtype=torch.long)

        loss = criterion(last_logits, last_targets)

        probs = F.softmax(last_logits, dim=-1)[:, 1]
        preds = last_logits.argmax(-1)

        total_correct += (preds == last_targets).sum().item()
        total_items += last_targets.numel()
        total_loss += loss.item() * last_targets.numel()

        all_preds.append(preds.flatten().cpu())
        all_probs.append(probs.flatten().cpu())
        all_true.append(last_targets.flatten().cpu())

    if total_items == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, float('nan'),
                0.0, 0.0, 0.0, None)

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    # Fixed threshold 0.5
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')

    # Sweep thresholds for best F1
    thresholds = np.unique(y_prob)
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0
    best_thr = None

    for thr in thresholds:
        y_hat = (y_prob >= thr).astype(int)
        prec_t = precision_score(y_true, y_hat, zero_division=0)
        rec_t = recall_score(y_true, y_hat, zero_division=0)
        f1_t = f1_score(y_true, y_hat, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_prec = prec_t
            best_rec = rec_t
            best_thr = thr

    return (total_loss / max(total_items, 1),
            total_correct / max(total_items, 1),
            precision, recall, f1, auc,
            best_prec, best_rec, best_f1, best_thr)
