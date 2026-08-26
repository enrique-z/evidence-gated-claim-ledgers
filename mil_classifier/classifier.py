"""MIL classifier: bag of patch features → per-slide ISUP grade prediction.

**For the gate-oneslide tracer, this is a random-init mean-pool + linear head.**
Its prediction quality is not meaningful; its job is to prove the pipeline's
shape works end-to-end. An untrained model will output ~uniform probabilities
across the 6 ISUP grades, so the Evidence Gate will (correctly) produce
`UNSUPPORTED` verdicts. That is the anti-fabrication regime working as
designed: an untrained model is not to be trusted, and the ledger says so.

A trained MIL head (ABMIL, TransMIL, CLAM-SB) plugs in behind the same
`SlidePrediction` interface at slice #29 (pilot-50 training).

The `MEAN_POOL_MIL_VERSION` string is deterministic per seed so ledger entries
are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

MEAN_POOL_MIL_VERSION = "mean-pool-linear-untrained-seed42@2026-08-05"


@dataclass(frozen=True)
class SlidePrediction:
    isup_grade: int              # argmax class in 0..5
    confidence: float            # softmax probability of the argmax class
    per_grade_probs: List[float] # length 6, sums to 1.0


class MeanPoolMIL:
    """Bag → SlidePrediction. Deterministic random-init for the tracer."""

    INPUT_DIM = 1024
    N_CLASSES = 6  # ISUP grades 0..5
    VERSION = MEAN_POOL_MIL_VERSION

    def __init__(self, seed: int = 42, device: str = "cuda"):
        import torch
        torch.manual_seed(seed)
        self._torch = torch
        self.device = device
        # Fixed-seed random-init linear head; NOT trained.
        self.head = torch.nn.Linear(self.INPUT_DIM, self.N_CLASSES).to(device).eval()
        self.seed = seed

    def predict(self, features) -> SlidePrediction:
        """features: (N, 1024) tensor of patch embeddings."""
        torch = self._torch
        if hasattr(features, "detach"):
            x = features.detach().to(self.device).float()
        else:
            x = torch.as_tensor(features).to(self.device).float()

        if x.numel() == 0:
            # Degenerate case: no patches. Emit a NOT_APPLICABLE-shaped
            # prediction (uniform, isup_grade=0, confidence=1/N).
            probs = [1.0 / self.N_CLASSES] * self.N_CLASSES
            return SlidePrediction(isup_grade=0, confidence=probs[0], per_grade_probs=probs)

        pooled = x.mean(dim=0, keepdim=True)
        with torch.no_grad():
            logits = self.head(pooled)
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().tolist()
        grade = int(max(range(len(probs)), key=lambda i: probs[i]))
        return SlidePrediction(
            isup_grade=grade,
            confidence=float(probs[grade]),
            per_grade_probs=[float(p) for p in probs],
        )
