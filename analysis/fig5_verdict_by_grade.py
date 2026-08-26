"""Phase G Fig 5 — Per-verdict × per-grade stacked bar chart.

Reads evidence/expansion_500/analysis/per_verdict_per_grade.json (from A1)
and produces papers/p5-asci/figures/fig5_verdict_by_grade.pdf: a stacked
bar over the 6 ISUP grades, one bar per verdict, coloured by verdict.
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
IN_JSON = REPO_ROOT / "evidence" / "expansion_500" / "analysis" / "per_verdict_per_grade.json"
OUT_PDF = REPO_ROOT / "papers" / "p5-asci" / "figures" / "fig5_verdict_by_grade.pdf"

VERDICTS = ["SUPPORTED", "CONTESTED", "UNSUPPORTED"]
COLORS = {"SUPPORTED": "#2ca02c", "CONTESTED": "#ff7f0e", "UNSUPPORTED": "#d62728"}


def main() -> int:
    if not IN_JSON.exists():
        print(f"ERROR: {IN_JSON} not found — run Phase A1 first.", file=sys.stderr)
        return 2
    payload = json.loads(IN_JSON.read_text())
    per_verdict = payload["per_verdict"]

    grades = list(range(6))
    # y[verdict][grade] = count of predicted-grade-`grade` slides among
    # slides with this verdict.
    y = {v: [per_verdict[v]["predicted_grade_counts"].get(str(g), 0)
             for g in grades] for v in VERDICTS}

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(grades))
    bottom = np.zeros(len(grades))
    for v in ["UNSUPPORTED", "CONTESTED", "SUPPORTED"]:
        vals = np.array(y[v])
        ax.bar(x, vals, bottom=bottom, color=COLORS[v], edgecolor="black",
               lw=0.4, label=f"{v.title()} (n={per_verdict[v]['n']})")
        for xi, (val, bi) in enumerate(zip(vals, bottom)):
            if val > 0:
                ax.text(xi, bi + val / 2, str(int(val)),
                        ha="center", va="center", fontsize=7, color="white",
                        fontweight="bold")
        bottom = bottom + vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"grade {g}" for g in grades])
    ax.set_ylabel("Number of slides")
    ax.set_xlabel("Predicted ISUP grade")
    ax.set_title(
        "Verdict distribution by predicted ISUP grade (expansion-500, mean-pool MIL)",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PDF.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
