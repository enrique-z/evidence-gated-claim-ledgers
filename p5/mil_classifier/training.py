"""Trainer for the MeanPoolMIL head.

Small module — the head is (INPUT_DIM=1024 → N_CLASSES=6) so training is fast.
Owns:
- Cross-entropy loss + Adam optimiser with fixed seed
- Per-fold training loop (one bag per slide, one label per slide)
- Version string generation for trained heads: `trained@<run_id>-cv<n>-seed<s>`
- Save/load of trained weights via torch.save/torch.load

The trained head has the same public predict() surface as the untrained one
(`p5.mil_classifier.MeanPoolMIL`) — same input shape `(N, 1024)`, same
`SlidePrediction` output. Only the VERSION string differs, and that is the
signal the orchestrator uses to allow non-tracer cohorts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class TrainingConfig:
    lr: float = 1e-3
    epochs: int = 50
    weight_decay: float = 1e-4
    seed: int = 42


def _train_one_head(bags: List, labels: List[int], cfg: TrainingConfig, device: str):
    """Train one linear head on a set of (bag, label) pairs. Returns the head."""
    import torch

    torch.manual_seed(cfg.seed)
    head = torch.nn.Linear(1024, 6).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Pre-pool each bag (mean pooling — matches inference)
    pooled = torch.stack([
        torch.as_tensor(b).float().to(device).mean(dim=0) for b in bags
    ])
    y = torch.tensor(labels, dtype=torch.long, device=device)

    head.train()
    for _ in range(cfg.epochs):
        opt.zero_grad()
        logits = head(pooled)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

    head.eval()
    return head


def train_and_predict_cv(
    slide_ids: List[str],
    features_per_slide: List,           # list[Tensor(N_i, 1024)]
    labels: List[int],                  # ground-truth ISUP per slide
    n_folds: int = 5,
    cfg: TrainingConfig = TrainingConfig(),
    device: str = "cuda",
    run_id: str = "pilot-50",
):
    """Run k-fold CV, produce honest per-slide predictions.

    Returns:
        predictions: list of (isup_grade, confidence, per_grade_probs) — one per
            slide, in the same order as slide_ids. Each prediction comes from a
            model that DID NOT see that slide during training.
        version: string identifying this training run for ledger pinning.
    """
    import torch

    version = f"trained@{run_id}-cv{n_folds}-seed{cfg.seed}"

    # Deterministic fold assignment: hash of slide_id % n_folds (via sorted index)
    indices = list(range(len(slide_ids)))
    fold_of = {i: i % n_folds for i in indices}

    predictions = [None] * len(slide_ids)
    for fold in range(n_folds):
        train_idx = [i for i in indices if fold_of[i] != fold]
        val_idx = [i for i in indices if fold_of[i] == fold]
        if not train_idx or not val_idx:
            continue
        head = _train_one_head(
            [features_per_slide[i] for i in train_idx],
            [labels[i] for i in train_idx],
            cfg,
            device,
        )
        # Predict for val slides
        for i in val_idx:
            feats = torch.as_tensor(features_per_slide[i]).float().to(device)
            pooled = feats.mean(dim=0, keepdim=True)
            with torch.no_grad():
                logits = head(pooled)
            probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().tolist()
            grade = int(max(range(len(probs)), key=lambda k: probs[k]))
            predictions[i] = (grade, float(probs[grade]), [float(p) for p in probs])

    # Any slide without a fold assignment (shouldn't happen for balanced folds)
    missing = [i for i, p in enumerate(predictions) if p is None]
    if missing:
        raise RuntimeError(f"CV left {len(missing)} slides unpredicted; check fold assignment")

    return predictions, version
