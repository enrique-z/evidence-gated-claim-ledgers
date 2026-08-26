"""CLAM-SB (Clustering-constrained Attention MIL, single-branch) — Lu et al.
Nature Biomedical Engineering 2021.

Single-branch means ONE attention head + ONE classifier (as opposed to
CLAM-MB, which trains K attention heads and K classifiers for K classes).
The key departure from ABMIL is the instance-level clustering loss: after
each forward, the top-k and bottom-k attended patches per bag are used as
pseudo-positive / pseudo-negative examples for an auxiliary binary
in-class-or-not head. This is what the paper claims regularises the
attention weights.

Our simplification (documented in the paper): we use cross-entropy for the
instance-level loss instead of the SVM hinge used in the reference
implementation (Lu et al. use `torch.smooth_l1_loss` on the SVM output);
CE has the same shape, is simpler in a small module, and does not change
the paper's headline claim that the reporting standard is stable across
MIL heads.

The head produces the same `(isup_grade, confidence, per_grade_probs)`
tuple as `ABMIL` and the mean-pool baseline, so it is a drop-in swap at
the pipeline seam.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

CLAM_SB_VERSION_TEMPLATE = (
    "trained@{run_id}-cv{n_folds}-seed{seed}-clam_sb-v1@2026-08-11"
)


class CLAMSingleBranch(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 128,
        n_classes: int = 6,
        dropout: float = 0.25,
        instance_k: int = 8,
    ) -> None:
        super().__init__()
        # Gated attention (Ilse 2018) — same substrate as ABMIL.
        self.V = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)
        self.attn_dropout = nn.Dropout(dropout)
        # Slide-level classifier — single-branch → single K-way head.
        self.classifier = nn.Linear(input_dim, n_classes)
        # Instance-level auxiliary head: binary in-class-or-not per patch.
        self.instance_head = nn.Linear(input_dim, 2)
        self.instance_k = instance_k

    def forward(self, h: torch.Tensor):
        # h: (N, D)
        v = torch.tanh(self.V(h))
        u = torch.sigmoid(self.U(h))
        a = self.w(self.attn_dropout(v * u)).squeeze(-1)     # (N,)
        attn = torch.softmax(a, dim=0)
        z = (attn.unsqueeze(-1) * h).sum(dim=0)              # (D,)
        logits = self.classifier(z)                          # (K,)

        # Instance-level pseudo-labels:
        #   top-k by attention → pseudo-positive-for-class (label 1)
        #   bottom-k by attention → pseudo-negative-for-class (label 0)
        n = h.shape[0]
        k = min(self.instance_k, max(1, n // 4))
        if n >= 2:
            order = torch.argsort(attn, descending=True)
            top = order[:k]
            bot = order[-k:]
            inst_h = h[torch.cat([top, bot])]
            inst_y = torch.cat([
                torch.ones(k, dtype=torch.long, device=h.device),
                torch.zeros(k, dtype=torch.long, device=h.device),
            ])
            inst_logits = self.instance_head(inst_h)
        else:
            inst_logits = None
            inst_y = None
        return logits, attn, inst_logits, inst_y


@dataclass(frozen=True)
class CLAMSBConfig:
    input_dim: int = 1024
    hidden_dim: int = 128
    n_classes: int = 6
    dropout: float = 0.25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    seed: int = 42
    instance_k: int = 8
    instance_loss_weight: float = 0.3  # matches Lu 2021 default range 0.1-0.7


def train_and_predict_cv_clam_sb(
    slide_ids: list[str],
    features_per_slide: list,
    labels: list[int],
    n_folds: int = 5,
    cfg: CLAMSBConfig = CLAMSBConfig(),
    device: str = "cpu",
    run_id: str = "expansion-500",
):
    version = CLAM_SB_VERSION_TEMPLATE.format(
        run_id=run_id, n_folds=n_folds, seed=cfg.seed,
    )
    n = len(slide_ids)
    fold_of = {i: i % n_folds for i in range(n)}

    predictions: list = [None] * n
    for fold in range(n_folds):
        train_idx = [i for i in range(n) if fold_of[i] != fold]
        val_idx = [i for i in range(n) if fold_of[i] == fold]
        if not train_idx or not val_idx:
            continue

        torch.manual_seed(cfg.seed + fold)
        model = CLAMSingleBranch(
            input_dim=cfg.input_dim, hidden_dim=cfg.hidden_dim,
            n_classes=cfg.n_classes, dropout=cfg.dropout,
            instance_k=cfg.instance_k,
        ).to(device)
        opt = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        slide_loss = nn.CrossEntropyLoss()
        instance_loss = nn.CrossEntropyLoss()

        train_bags = [torch.as_tensor(features_per_slide[i]).float().to(device)
                      for i in train_idx]
        train_labels = torch.tensor([labels[i] for i in train_idx],
                                    dtype=torch.long, device=device)

        model.train()
        for _epoch in range(cfg.epochs):
            opt.zero_grad()
            total_loss = torch.zeros(1, device=device)
            instance_component = torch.zeros(1, device=device)
            n_inst_bags = 0
            slide_logits = []
            for bag in train_bags:
                logits, _attn, inst_logits, inst_y = model(bag)
                slide_logits.append(logits)
                if inst_logits is not None:
                    instance_component = instance_component + instance_loss(
                        inst_logits, inst_y
                    )
                    n_inst_bags += 1
            slide_logits_b = torch.stack(slide_logits)
            slide_component = slide_loss(slide_logits_b, train_labels)
            total_loss = (
                slide_component
                + (cfg.instance_loss_weight * instance_component / max(1, n_inst_bags))
            )
            total_loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            for i in val_idx:
                h = torch.as_tensor(features_per_slide[i]).float().to(device)
                if h.numel() == 0:
                    probs = [1.0 / cfg.n_classes] * cfg.n_classes
                    predictions[i] = (0, probs[0], probs)
                    continue
                logits, _, _, _ = model(h)
                probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
                grade = int(max(range(len(probs)), key=lambda k: probs[k]))
                predictions[i] = (grade, float(probs[grade]),
                                   [float(p) for p in probs])

    missing = [i for i, p in enumerate(predictions) if p is None]
    if missing:
        raise RuntimeError(f"CV left {len(missing)} slides unpredicted")
    return predictions, version
