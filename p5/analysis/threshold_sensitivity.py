"""Phase A5 — Threshold sensitivity heatmap for confidence-threshold-v1.

Reads: evidence/expansion_500/expansion_500_ledger.jsonl
Writes:
  - papers/p5-asci/figures/fig6_threshold_heatmap.pdf  (2-panel)
  - evidence/expansion_500/analysis/threshold_sensitivity.json

Grid:
  high ∈ {0.50, 0.55, ..., 0.90}          (9 values)
  low  ∈ {0.20, 0.25, ..., 0.45}          (6 values)
  Only (low, high) pairs with low < high are populated; the rest is NaN.

For each populated cell we compute:
  - supported_count       — # slides with confidence >= high
  - supported_accuracy    — # correct in the supported subset / supported_count
  - unsupported_count     — # slides with confidence <= low
  - contested_count       — 500 - supported_count - unsupported_count
  - contested_accuracy    — # correct in the contested subset / contested_count
The high/low argmax-confidence definitions are identical to the reference
gate rule `confidence-threshold-v1`.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# NeurIPS 2026 formatting: submissions must contain only Type 1 or embedded
# TrueType fonts. Matplotlib emits Type 3 for PDF by default; 42 selects
# TrueType. Font encoding only -- no figure content is affected.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXPANSION = REPO_ROOT / "evidence" / "expansion_500" / "expansion_500_ledger.jsonl"
OUT_PDF = REPO_ROOT / "papers" / "p5-asci" / "figures" / "fig6_threshold_heatmap.pdf"
OUT_JSON = REPO_ROOT / "evidence" / "expansion_500" / "analysis" / "threshold_sensitivity.json"

HIGH_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
LOW_GRID = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]


def _load(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def main() -> int:
    if not EXPANSION.exists():
        print("ERROR: missing expansion ledger", file=sys.stderr)
        return 2

    rows = _load(EXPANSION)
    conf = np.array([float(r["extra"]["confidence"]) for r in rows])
    pred = np.array([int(r["extra"]["isup_prediction"]) for r in rows])
    truth = np.array([int(r["extra"]["ground_truth_isup"]) for r in rows])
    correct = (pred == truth).astype(np.int64)
    n = len(rows)

    n_high = len(HIGH_GRID)
    n_low = len(LOW_GRID)
    sup_count = np.full((n_low, n_high), np.nan)
    sup_acc = np.full((n_low, n_high), np.nan)
    unsup_count = np.full((n_low, n_high), np.nan)
    contested_count = np.full((n_low, n_high), np.nan)
    contested_acc = np.full((n_low, n_high), np.nan)

    cells = []
    for i, lo in enumerate(LOW_GRID):
        for j, hi in enumerate(HIGH_GRID):
            if not lo < hi:
                continue
            sup_mask = conf >= hi
            unsup_mask = conf <= lo
            cont_mask = (~sup_mask) & (~unsup_mask)

            s_n = int(sup_mask.sum())
            u_n = int(unsup_mask.sum())
            c_n = int(cont_mask.sum())
            assert s_n + u_n + c_n == n

            s_acc = float(correct[sup_mask].mean()) if s_n else None
            c_acc = float(correct[cont_mask].mean()) if c_n else None

            sup_count[i, j] = s_n
            sup_acc[i, j] = s_acc if s_acc is not None else np.nan
            unsup_count[i, j] = u_n
            contested_count[i, j] = c_n
            contested_acc[i, j] = c_acc if c_acc is not None else np.nan
            cells.append({
                "low": lo, "high": hi,
                "supported_count": s_n,
                "supported_accuracy": s_acc,
                "unsupported_count": u_n,
                "contested_count": c_n,
                "contested_accuracy": c_acc,
            })

    # ---------- figure ----------
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.9))

    ax = axs[0]
    im = ax.imshow(sup_count, cmap="Blues", origin="lower", aspect="auto")
    ax.set_xticks(range(n_high)); ax.set_xticklabels([f"{h:.2f}" for h in HIGH_GRID])
    ax.set_yticks(range(n_low));  ax.set_yticklabels([f"{l:.2f}" for l in LOW_GRID])
    ax.set_xlabel("high threshold τ_high")
    ax.set_ylabel("low threshold τ_low")
    ax.set_title("SUPPORTED count on expansion-500")
    for i in range(n_low):
        for j in range(n_high):
            if not np.isnan(sup_count[i, j]):
                ax.text(j, i, f"{int(sup_count[i, j])}", ha="center", va="center",
                        fontsize=7.5, color="black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = axs[1]
    im = ax.imshow(sup_acc, cmap="Greens", origin="lower", aspect="auto",
                   vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n_high)); ax.set_xticklabels([f"{h:.2f}" for h in HIGH_GRID])
    ax.set_yticks(range(n_low));  ax.set_yticklabels([f"{l:.2f}" for l in LOW_GRID])
    ax.set_xlabel("high threshold τ_high")
    ax.set_ylabel("low threshold τ_low")
    ax.set_title("SUPPORTED accuracy on expansion-500")
    for i in range(n_low):
        for j in range(n_high):
            if not np.isnan(sup_acc[i, j]):
                ax.text(j, i, f"{sup_acc[i, j]:.2f}", ha="center", va="center",
                        fontsize=7.5, color="black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    # Annotate the paper's default cell (0.70, 0.30).
    if 0.30 in LOW_GRID and 0.70 in HIGH_GRID:
        i0 = LOW_GRID.index(0.30); j0 = HIGH_GRID.index(0.70)
        for a in axs:
            a.add_patch(plt.Rectangle((j0 - 0.5, i0 - 0.5), 1, 1, fill=False,
                                       edgecolor="red", lw=1.6))
    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    # ---------- JSON companion ----------
    payload = {
        "cohort_id": "expansion-500",
        "gate_rule_id": "confidence-threshold-v1@2026-08-05",
        "high_grid": HIGH_GRID,
        "low_grid": LOW_GRID,
        "cells": cells,
        "default_operating_point": {"high": 0.70, "low": 0.30},
        "regenerated_at": datetime.now(timezone.utc).isoformat(),
    }
    with OUT_JSON.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"Wrote {OUT_PDF.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_JSON.relative_to(REPO_ROOT)}  ({len(cells)} cells)")
    # Sample of cells including the default.
    for c in cells:
        if c["low"] == 0.30 and c["high"] == 0.70:
            print(f"  default (low=0.30, high=0.70): SUPPORTED N={c['supported_count']} "
                  f"acc={c['supported_accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
