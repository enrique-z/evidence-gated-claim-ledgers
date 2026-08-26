"""Attention-based Multiple Instance Learning (Ilse et al. ICML 2018).

Gated attention pooling over 1024-d Phikon-v2 patch embeddings, followed by
a linear classifier over 6 ISUP classes.

Architecture (equations 8-9 of Ilse et al. 2018):
    a_i = w^T * (tanh(V * h_i) ⊙ sigmoid(U * h_i))
    α_i = softmax(a)_i
    z   = Σ α_i * h_i
    logits = W_c * z + b_c
    p   = softmax(logits)

Hyper-parameters follow the paper's defaults:
    input dim D = 1024
    hidden dim L = 128
    classes K = 6
    dropout on attention 0.25
    Adam optimiser, lr=1e-3, weight_decay=1e-4, 50 epochs (matches the
    mean-pool trainer so cross-head runtime is directly comparable).

The head exposes the same `predict(features) → SlidePrediction`-shaped
tuple as `p5.mil_classifier.MeanPoolMIL`, so it plugs into the same
Claim → Evidence Gate → LedgerRow path unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


ABMIL_VERSION_TEMPLATE = "trained@{run_id}-cv{n_folds}-seed{seed}-abmil-gated-v1@2026-08-11"


class GatedAttentionMIL(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 128,
        n_classes: int = 6,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.V = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)
        self.attn_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, n_classes)

    def attention(self, h: torch.Tensor) -> torch.Tensor:
        # h: (N, D)
        v = torch.tanh(self.V(h))       # (N, L)
        u = torch.sigmoid(self.U(h))    # (N, L)
        a = self.w(self.attn_dropout(v * u)).squeeze(-1)  # (N,)
        return torch.softmax(a, dim=0)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # h: (N, D)  →  logits (K,), attn (N,)
        attn = self.attention(h)
        z = (attn.unsqueeze(-1) * h).sum(dim=0)   # (D,)
        logits = self.classifier(z)               # (K,)
        return logits, attn


@dataclass(frozen=True)
class ABMILConfig:
    input_dim: int = 1024
    hidden_dim: int = 128
    n_classes: int = 6
    dropout: float = 0.25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    seed: int = 42


def train_and_predict_cv_abmil(
    slide_ids: list[str],
    features_per_slide: list,
    labels: list[int],
    n_folds: int = 5,
    cfg: ABMILConfig = ABMILConfig(),
    device: str = "cpu",
    run_id: str = "expansion-500",
) -> tuple[list[tuple[int, float, list[float]]], str]:
    """5-fold CV: for each fold, train ABMIL on the other 4 folds, predict on
    the held-out fold. Fold assignment is `i % n_folds` — matches the
    mean-pool baseline for cross-head comparability.
    """
    version = ABMIL_VERSION_TEMPLATE.format(
        run_id=run_id, n_folds=n_folds, seed=cfg.seed
    )

    n = len(slide_ids)
    fold_of = {i: i % n_folds for i in range(n)}

    predictions: list = [None] * n
    for fold in range(n_folds):
        train_idx = [i for i in range(n) if fold_of[i] != fold]
        val_idx = [i for i in range(n) if fold_of[i] == fold]
        if not train_idx or not val_idx:
            continue

        torch.manual_seed(cfg.seed + fold)  # per-fold seed for reproducibility
        model = GatedAttentionMIL(
            input_dim=cfg.input_dim, hidden_dim=cfg.hidden_dim,
            n_classes=cfg.n_classes, dropout=cfg.dropout,
        ).to(device)
        opt = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()

        # Pre-move training bags to device once.
        train_bags = [torch.as_tensor(features_per_slide[i]).float().to(device)
                      for i in train_idx]
        train_labels = torch.tensor([labels[i] for i in train_idx],
                                    dtype=torch.long, device=device)

        model.train()
        for _epoch in range(cfg.epochs):
            opt.zero_grad()
            logits_batch = torch.stack([model(b)[0] for b in train_bags])  # (B, K)
            loss = loss_fn(logits_batch, train_labels)
            loss.backward()
            opt.step()

        # Predict the held-out fold.
        model.eval()
        with torch.no_grad():
            for i in val_idx:
                h = torch.as_tensor(features_per_slide[i]).float().to(device)
                if h.numel() == 0:
                    probs = [1.0 / cfg.n_classes] * cfg.n_classes
                    predictions[i] = (0, probs[0], probs)
                    continue
                logits, _ = model(h)
                probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
                grade = int(max(range(len(probs)), key=lambda k: probs[k]))
                predictions[i] = (grade, float(probs[grade]),
                                   [float(p) for p in probs])

    missing = [i for i, p in enumerate(predictions) if p is None]
    if missing:
        raise RuntimeError(f"CV left {len(missing)} slides unpredicted")

    return predictions, version
