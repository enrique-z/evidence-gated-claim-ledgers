"""Phase B3 — Consolidated MIL-head comparison.

Reads:
  - evidence/expansion_500/expansion_500_ledger.jsonl   (mean-pool baseline)
  - evidence/expansion_500/abmil_ledger.jsonl           (ABMIL, Phase B1)
  - evidence/expansion_500/clam_sb_ledger.jsonl         (CLAM-SB, Phase B2)

Writes:
  - evidence/expansion_500/analysis/mil_head_comparison.json
      Cross-head verdict distribution, per-verdict accuracy, ECE, Brier, NLL,
      and per-verdict grade counts. The paper's §5.5 MIL-comparison table
      pulls its numbers from here.

  - papers/p5-asci/figures/fig7_risk_coverage_per_head.pdf
      Risk-coverage curves (max-softmax score) for the three heads on a
      single axis, with the EGCL (0.70) SUPPORTED point annotated for each.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
from collections import Counter, defaultdict
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
LEDGERS = {
    "mean-pool": REPO_ROOT / "evidence" / "expansion_500" / "expansion_500_ledger.jsonl",
    "abmil":     REPO_ROOT / "evidence" / "expansion_500" / "abmil_ledger.jsonl",
    "clam-sb":   REPO_ROOT / "evidence" / "expansion_500" / "clam_sb_ledger.jsonl",
}
OUT_JSON = REPO_ROOT / "evidence" / "expansion_500" / "analysis" / "mil_head_comparison.json"
OUT_PDF = REPO_ROOT / "papers" / "p5-asci" / "figures" / "fig7_risk_coverage_per_head.pdf"


def _load(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def _summary(rows: list[dict]) -> dict:
    verdict_n = Counter()
    verdict_correct = Counter()
    verdict_conf_sum = defaultdict(float)
    verdict_pred_grade: dict[str, Counter] = defaultdict(Counter)
    verdict_truth_grade: dict[str, Counter] = defaultdict(Counter)
    all_correct = 0
    total = len(rows)
    per_class_total = Counter()
    per_class_correct = Counter()
    P = []
    y = []
    for r in rows:
        v = r["verdict"]
        pred = int(r["extra"]["isup_prediction"])
        truth = int(r["extra"]["ground_truth_isup"])
        conf = float(r["extra"]["confidence"])
        probs = r["extra"]["per_grade_probs"]
        verdict_n[v] += 1
        verdict_conf_sum[v] += conf
        verdict_pred_grade[v][pred] += 1
        verdict_truth_grade[v][truth] += 1
        per_class_total[truth] += 1
        if pred == truth:
            verdict_correct[v] += 1
            all_correct += 1
            per_class_correct[truth] += 1
        P.append(probs)
        y.append(truth)

    P = np.array(P, dtype=np.float64)
    y = np.array(y, dtype=np.int64)

    # ECE / Brier / NLL over the whole ledger (10 equal-width bins).
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    correct_arr = (pred == y).astype(np.float64)
    n_bins = 10
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = ((conf >= lo) & (conf <= hi)) if b == n_bins - 1 else ((conf >= lo) & (conf < hi))
        n_b = int(m.sum())
        if n_b:
            ece += (n_b / total) * abs(correct_arr[m].mean() - conf[m].mean())
    one_hot = np.zeros_like(P); one_hot[np.arange(len(y)), y] = 1.0
    brier = float(((P - one_hot) ** 2).sum(axis=1).mean())
    nll = float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0)).mean())

    recalls = [per_class_correct[c] / per_class_total[c]
               for c in sorted(per_class_total) if per_class_total[c]]
    balanced = sum(recalls) / len(recalls) if recalls else None
    raw = all_correct / total

    per_verdict = {}
    for v in ["SUPPORTED", "UNSUPPORTED", "NOT_APPLICABLE", "CONTESTED"]:
        n = verdict_n[v]
        per_verdict[v] = {
            "n": n,
            "correct": verdict_correct[v],
            "accuracy": (verdict_correct[v] / n) if n else None,
            "mean_confidence": (verdict_conf_sum[v] / n) if n else None,
            "predicted_grade_counts": {str(g): verdict_pred_grade[v].get(g, 0)
                                       for g in range(6)},
            "truth_grade_counts": {str(g): verdict_truth_grade[v].get(g, 0)
                                   for g in range(6)},
        }
    return {
        "n_slides": total,
        "raw_accuracy": raw,
        "balanced_accuracy": balanced,
        "ece": float(ece),
        "brier": brier,
        "nll": nll,
        "per_verdict": per_verdict,
    }


def _rank_and_sweep(scores: list[float], correct: list[int]) -> dict:
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    n = len(order)
    cov, acc = [], []
    cum = 0
    for k, idx in enumerate(order):
        cum += correct[idx]
        cov.append((k + 1) / n)
        acc.append(cum / (k + 1))
    return {"coverage": cov, "accuracy": acc}


def main() -> int:
    for name, p in LEDGERS.items():
        if not p.exists():
            print(f"ERROR: missing {name} ledger at {p}", file=sys.stderr)
            return 2

    summaries = {}
    curves = {}
    egcl_points = {}
    for name, p in LEDGERS.items():
        rows = _load(p)
        if len(rows) != 500:
            print(f"ERROR: {name} ledger has {len(rows)} rows (want 500)",
                  file=sys.stderr)
            return 3
        summaries[name] = _summary(rows)
        conf = [float(r["extra"]["confidence"]) for r in rows]
        correct = [int(r["extra"]["isup_prediction"] == r["extra"]["ground_truth_isup"])
                   for r in rows]
        curves[name] = _rank_and_sweep(conf, correct)
        # EGCL default operating point (>= 0.70) per head:
        sup_idx = [i for i, c in enumerate(conf) if c >= 0.70]
        egcl_points[name] = {
            "supported_count": len(sup_idx),
            "supported_coverage": len(sup_idx) / len(rows),
            "supported_accuracy": (
                sum(correct[i] for i in sup_idx) / len(sup_idx) if sup_idx else None
            ),
        }

    # ---------- figure ----------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    style = {
        "mean-pool": {"color": "#1f77b4", "ls": "-",  "label": "mean-pool"},
        "abmil":     {"color": "#2ca02c", "ls": "--", "label": "ABMIL (gated)"},
        "clam-sb":   {"color": "#d62728", "ls": ":",  "label": "CLAM-SB"},
    }
    for name, s in style.items():
        c = curves[name]
        ax.plot(c["coverage"], c["accuracy"], color=s["color"], ls=s["ls"],
                lw=1.4, label=s["label"])
    balanced_bar = summaries["mean-pool"]["balanced_accuracy"]
    ax.axhline(balanced_bar, color="gray", ls=(0, (1, 2)), lw=0.9,
               label=f"mean-pool balanced acc ({balanced_bar:.3f})")

    for name, pt in egcl_points.items():
        if pt["supported_accuracy"] is None:
            continue
        s = style[name]
        ax.scatter([pt["supported_coverage"]], [pt["supported_accuracy"]],
                   marker="*", s=110, color=s["color"], edgecolor="black", lw=0.5,
                   zorder=6,
                   label=f"{s['label']} EGCL @ 0.70: "
                         f"cov={pt['supported_coverage']:.3f}, "
                         f"acc={pt['supported_accuracy']:.3f}")

    ax.set_xlabel("Coverage (fraction of expansion-500 retained)")
    ax.set_ylabel("Accuracy on retained subset")
    ax.set_xlim(0, 1); ax.set_ylim(0.2, 1.02)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.set_title("Risk-coverage across MIL heads (Phikon-v2 backbone, expansion-500)",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    # ---------- JSON ----------
    payload = {
        "cohort_id": "expansion-500",
        "heads": {name: summaries[name] for name in LEDGERS},
        "egcl_operating_points": egcl_points,
        "regenerated_at": datetime.now(timezone.utc).isoformat(),
    }
    with OUT_JSON.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True); fh.write("\n")

    print(f"Wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_PDF.relative_to(REPO_ROOT)}")
    print("\nHead                balanced_acc  ECE     Brier   NLL     SUPPORTED N/acc")
    for name in ["mean-pool", "abmil", "clam-sb"]:
        s = summaries[name]
        pt = egcl_points[name]
        sup_acc = f"{pt['supported_accuracy']:.4f}" if pt["supported_accuracy"] is not None else "N/A"
        print(f"  {name:16s}  {s['balanced_accuracy']:.4f}      {s['ece']:.4f}  "
              f"{s['brier']:.4f}  {s['nll']:.4f}  N={pt['supported_count']:3d}/acc={sup_acc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
