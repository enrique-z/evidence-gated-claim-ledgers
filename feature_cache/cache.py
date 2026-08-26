"""Persistent slide-keyed feature store.

Keys are `(slide_id, backbone_name, backbone_version)`. Values are numpy
arrays of shape (N_patches, 1024) stored as `.npz` files.

Warm cache turns embedding into a no-op; cold cache never regresses
correctness (the caller re-computes and calls `put`).

For the gate-oneslide tracer, this is a local file store. For pilot / expansion
runs on Kaggle, the same cache lives in `/kaggle/working/` and gets uploaded
to a Kaggle Dataset for cross-session reuse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class FeatureCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, slide_id: str, backbone_name: str, backbone_version: str) -> Path:
        safe = backbone_version.replace("/", "_").replace(":", "_").replace("@", "_")
        return self.root / f"{slide_id}__{backbone_name}__{safe}.npz"

    def get(self, slide_id: str, backbone_name: str, backbone_version: str):
        """Return the cached feature array, or None if absent."""
        import numpy as np
        p = self._path(slide_id, backbone_name, backbone_version)
        if not p.exists():
            return None
        with np.load(p) as npz:
            return npz["features"]

    def put(self, slide_id: str, backbone_name: str, backbone_version: str, features) -> None:
        """Persist a (N, D) feature array to disk."""
        import numpy as np
        p = self._path(slide_id, backbone_name, backbone_version)
        arr = features
        # Accept torch tensors or numpy arrays
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        np.savez_compressed(p, features=np.asarray(arr))

    def has(self, slide_id: str, backbone_name: str, backbone_version: str) -> bool:
        return self._path(slide_id, backbone_name, backbone_version).exists()
