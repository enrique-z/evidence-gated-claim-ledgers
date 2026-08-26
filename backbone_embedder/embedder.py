"""Phikon-v2 backbone embedder.

Wraps `owkin/phikon-v2` (DINOv2, ~303 M params) and exposes a batched
`embed(patches) -> Tensor[N, 1024]` interface.

Owns the M3 runtime contract from `~/.claude/projects/.../memory/kaggle_setup.md`
notes: on macOS, unset OpenMP / MKL threading env vars cause the process to
crash silently at the first forward pass ("leaked semaphore"). We refuse to
construct the embedder unless the three required env vars are set, so a Mac
dry-run cannot silently crash mid-batch. On Linux (Kaggle), the check is a
no-op.

Carries a `BACKBONE_VERSION` string that every ledger entry pins.
"""

from __future__ import annotations

import os
import platform
from typing import Iterable, List

BACKBONE_VERSION = "phikon-v2-owkin@2026-08-05"


REQUIRED_MAC_ENV = {
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


class BackboneRuntimeError(RuntimeError):
    """Raised when the runtime contract (env vars, device availability) is violated."""


def _enforce_mac_runtime_contract():
    """On macOS, refuse to load the model unless the three env vars are set."""
    if platform.system() != "Darwin":
        return
    missing = [k for k, v in REQUIRED_MAC_ENV.items() if os.environ.get(k) != v]
    if missing:
        raise BackboneRuntimeError(
            "Mac runtime contract violated: set the following env vars before importing "
            "torch or constructing the embedder — " + ", ".join(missing) + ". "
            "See ~/.claude/projects/-Users-apple-code-neurips-26-workshop-papers/memory/kaggle_setup.md."
        )


class PhikonV2Embedder:
    """Batched Phikon-v2 CLS embedder.

    Usage:
        emb = PhikonV2Embedder(device="cuda")
        features = emb.embed(patches)  # -> torch.Tensor of shape (N, 1024)
    """

    HF_ID = "owkin/phikon-v2"
    OUTPUT_DIM = 1024

    def __init__(self, device: str = "cuda"):
        _enforce_mac_runtime_contract()

        # Imports deferred so `import p5.backbone_embedder` is cheap in tests.
        import torch
        from transformers import AutoModel, AutoImageProcessor

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(self.HF_ID)
        self.model = AutoModel.from_pretrained(self.HF_ID).to(device).eval()
        self._torch = torch
        self.backbone_version = BACKBONE_VERSION

    def embed(self, patches: Iterable, batch_size: int = 16):
        """Embed an iterable of numpy patches → (N, 1024) torch tensor on CPU."""
        torch = self._torch

        # Normalize input: accept a list of Patch objects OR raw numpy arrays
        tensors: List = []
        for p in patches:
            arr = getattr(p, "tensor", p)  # duck-type: Patch or ndarray
            tensors.append(arr)

        if not tensors:
            return torch.zeros((0, self.OUTPUT_DIM), dtype=torch.float32)

        out = []
        for start in range(0, len(tensors), batch_size):
            batch = tensors[start:start + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                res = self.model(**inputs)
            # Phikon-v2 (DINOv2) exposes the CLS token as last_hidden_state[:, 0]
            cls = res.last_hidden_state[:, 0, :].detach().cpu()
            out.append(cls)
        return torch.cat(out, dim=0)
