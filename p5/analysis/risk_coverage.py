"""Phase A3 — Risk-coverage curve (the paper's load-bearing figure, Fig 2).

For each of {max-softmax, entropy (inverse), margin} rank all 500 expansion
slides by the score and sweep the coverage from 1/500 to 500/500. At each
coverage k, the retained subset is the top-k slides by that score;
retained-subset accuracy is the fraction of those k whose predicted ISUP
matches the ground-truth ISUP.

Additionally: three conformal (LAC) operating points at alpha ∈ {0.05, 0.10,
0.20}, calibrated on pilot-50, plotted as (singleton coverage, singleton
accuracy). If the LAC calibration produces zero singletons at a given alpha,
that point is reported with coverage=0 / accuracy=undefined and omitted from
the plot but recorded in the JSON companion.

The current EGCL default (confidence-threshold-v1 at high=0.70, low=0.30)
is annotated as a point on the max-softmax curve (its SUPPORTED coverage is
9/500 = 0.018, SUPPORTED accuracy 7/9 = 0.7778).

Writes:
  - papers/p5-asci/figures/fig2_risk_coverage.pdf
  - evidence/expansion_500/analysis/risk_coverage.json
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

from p5.evidence_gate.alt_gates import ConformalGate, calibrate_lac
from p5.evidence_gate.gate import Claim

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXPANSION = REPO_ROOT / "evidence" / "expansion_500" / "expansion_500_ledger.jsonl"
PILOT = REPO_ROOT / "evidence" / "pilot_50" / "pilot_50_ledger.jsonl"
OUT_PDF = REPO_ROOT / "papers" / "p5-asci" / "figures" / "fig2_risk_coverage.pdf"
OUT_JSON = REPO_ROOT / "evidence" / "expansion_500" / "analysis" / "risk_coverage.json"


def _load(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def _entropy(p: list[float]) -> float:
    return -sum(x * math.log(x) for x in p if x > 0)


def _margin(p: list[float]) -> float:
    s = sorted(p, reverse=True)
    return s[0] - (s[1] if len(s) > 1 else 0.0)


def _rank_and_sweep(scores: list[float], correct: list[int]) -> dict:
    """Rank slides by score DESCENDING (higher = more confident); sweep k=1..N.

    Returns coverage, accuracy, and score-threshold arrays.
    """
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    n = len(order)
    cov = [(k + 1) / n for k in range(n)]
    cum_correct = 0
    acc = []
    thresh = []
    for k, idx in enumerate(order):
        cum_correct += correct[idx]
        acc.append(cum_correct / (k + 1))
        thresh.append(scores[idx])
    return {"coverage": cov, "accuracy": acc, "score_threshold_at_k": thresh}


def main() -> int:
    if not EXPANSION.exists() or not PILOT.exists():
        print("ERROR: missing ledgers", file=sys.stderr)
        return 2

    exp = _load(EXPANSION)
    pil = _load(PILOT)
    n = len(exp)
    correct = [
        int(r["extra"]["isup_prediction"] == r["extra"]["ground_truth_isup"])
        for r in exp
    ]

    # Score families (all higher = more confident):
    #   max_softmax    — the top probability
    #   inv_entropy    — log(K) - H(p)
    #   margin         — top1 - top2
    max_softmax = [float(max(r["extra"]["per_grade_probs"])) for r in exp]
    logK = math.log(6)
    inv_entropy = [logK - _entropy(r["extra"]["per_grade_probs"]) for r in exp]
    margins = [_margin(r["extra"]["per_grade_probs"]) for r in exp]

    curves = {
        "max_softmax": _rank_and_sweep(max_softmax, correct),
        "inv_entropy": _rank_and_sweep(inv_entropy, correct),
        "margin": _rank_and_sweep(margins, correct),
    }

    # EGCL default point: SUPPORTED coverage = |{i: p_top >= 0.70}| / N,
    # accuracy = correct rate on that subset.
    sup_mask = [i for i, p in enumerate(max_softmax) if p >= 0.70]
    egcl_cov = len(sup_mask) / n
    egcl_acc = (
        sum(correct[i] for i in sup_mask) / len(sup_mask) if sup_mask else None
    )

    # Conformal LAC points at alpha ∈ {0.05, 0.10, 0.20}, calibrated on pilot-50.
    cals = calibrate_lac(pil, alphas=(0.05, 0.10, 0.20),
                         calibration_source="pilot-50")
    conformal_points = {}
    for alpha, cal in cals.items():
        gate = ConformalGate(cal)
        n_singleton = 0
        n_singleton_correct = 0
        n_empty = 0
        for r in exp:
            claim = Claim(
                claim_id=r["claim_id"], claim_text=r["claim_text"],
                evidence_ref=r["evidence_ref"],
                model_output={
                    "per_grade_probs": r["extra"]["per_grade_probs"],
                    "confidence": r["extra"]["confidence"],
                    "isup_grade": r["extra"]["isup_prediction"],
                },
                backbone_version=r["backbone_version"],
                cohort_id=r["cohort_id"],
            )
            _, j = gate.gate(claim)
            pset = j.thresholds.get("prediction_set", [])
            if len(pset) == 1:
                n_singleton += 1
                if pset[0] == int(r["extra"]["ground_truth_isup"]):
                    n_singleton_correct += 1
            elif len(pset) == 0:
                n_empty += 1
        conformal_points[f"alpha={alpha:.2f}"] = {
            "alpha": alpha,
            "q_alpha": cal.q_alpha,
            "singleton_count": n_singleton,
            "singleton_coverage": n_singleton / n,
            "singleton_accuracy": (
                n_singleton_correct / n_singleton if n_singleton else None
            ),
            "empty_count": n_empty,
        }

    # ---------- figure ----------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    style = {
        "max_softmax": {"label": "max-softmax", "color": "#1f77b4", "ls": "-"},
        "inv_entropy": {"label": "inverse entropy", "color": "#2ca02c", "ls": "--"},
        "margin":      {"label": "margin (top1 - top2)", "color": "#d62728", "ls": ":"},
    }
    for name, s in style.items():
        c = curves[name]
        ax.plot(c["coverage"], c["accuracy"], color=s["color"], ls=s["ls"],
                lw=1.4, label=s["label"])

    # Chance / base rate baselines: raw accuracy and balanced accuracy.
    raw_acc = sum(correct) / n
    balanced_acc = 0.3652  # 6-class recall mean (from A1, deterministic)
    ax.axhline(raw_acc, color="gray", ls=(0, (1, 2)), lw=0.9,
               label=f"raw accuracy on all N={n} (base rate {raw_acc:.3f})")
    ax.axhline(balanced_acc, color="gray", ls=(0, (2, 2)), lw=0.9,
               label=f"balanced accuracy ({balanced_acc:.3f})")

    # EGCL default annotation.
    if egcl_acc is not None:
        ax.scatter([egcl_cov], [egcl_acc], marker="*", s=140,
                   color="black", zorder=5,
                   label=f"EGCL default (0.70/0.30): "
                         f"cov={egcl_cov:.3f}, acc={egcl_acc:.3f}")
        ax.annotate(f"EGCL @ (0.70, 0.30)",
                    xy=(egcl_cov, egcl_acc),
                    xytext=(egcl_cov + 0.04, egcl_acc + 0.015),
                    fontsize=8, color="black",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="black"))

    # Conformal LAC points at alpha ∈ {0.05, 0.10, 0.20}.
    plotted_alphas = []
    for key, cp in conformal_points.items():
        if cp["singleton_count"] > 0 and cp["singleton_accuracy"] is not None:
            ax.scatter([cp["singleton_coverage"]], [cp["singleton_accuracy"]],
                       marker="D", s=45, color="#9467bd",
                       label=f"LAC {key} (n=50)"
                             f"  cov={cp['singleton_coverage']:.3f}")
            plotted_alphas.append(key)
    if not plotted_alphas:
        # All zero singletons — leave a note on the plot.
        ax.text(0.98, 0.05,
                "LAC (pilot-50 calibration): 0 singletons at α ∈ {0.05,0.10,0.20}",
                fontsize=7.5, ha="right", va="bottom",
                transform=ax.transAxes,
                bbox=dict(fc="white", ec="#9467bd", lw=0.5, alpha=0.9))

    ax.set_xlabel("Coverage (fraction of expansion-500 slides retained)")
    ax.set_ylabel("Accuracy on retained subset")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 1.02)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.set_title("Risk--coverage on the expansion-500 cohort (Phikon-v2 + mean-pool MIL)",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    # ---------- JSON companion ----------
    payload = {
        "cohort_id": "expansion-500",
        "n_slides": n,
        "raw_accuracy": raw_acc,
        "balanced_accuracy": balanced_acc,
        "egcl_default": {
            "high_threshold": 0.70,
            "low_threshold": 0.30,
            "supported_count": len(sup_mask),
            "supported_coverage": egcl_cov,
            "supported_accuracy": egcl_acc,
        },
        "curves": {
            name: {
                "coverage": c["coverage"],
                "accuracy": c["accuracy"],
            } for name, c in curves.items()
        },
        "conformal_points": conformal_points,
        "regenerated_at": datetime.now(timezone.utc).isoformat(),
    }
    with OUT_JSON.open("w") as fh:
        json.dump(payload, fh, sort_keys=True)  # compact — arrays are long
        fh.write("\n")

    print(f"Wrote {OUT_PDF.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"EGCL default: coverage={egcl_cov:.4f}  accuracy={egcl_acc:.4f}")
    for name in curves:
        c = curves[name]
        # Report accuracy at 10% coverage as a summary comparator.
        k10 = max(1, int(0.10 * n))
        print(f"  {name:14s}  acc @ 10% coverage = {c['accuracy'][k10-1]:.4f}")
    print(f"Conformal LAC (pilot-50, n=50) singletons:")
    for k, cp in conformal_points.items():
        print(f"  {k}: q_alpha={cp['q_alpha']:.4f}  singletons={cp['singleton_count']}"
              f"  empty={cp['empty_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
