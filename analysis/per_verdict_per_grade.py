"""Phase A1 — Per-verdict × per-grade accuracy stratification.

Reads `evidence/expansion_500/expansion_500_ledger.jsonl` (frozen, 500 rows)
and writes:

  - evidence/expansion_500/analysis/per_verdict_per_grade.json
      Structured, deterministic JSON. Every numeric field is derived from the
      ledger row set — no external sources, no smoothing, no rounding beyond
      float64 precision.

Downstream figure `papers/p5-asci/figures/fig5_verdict_by_grade.py` reads the
JSON and renders the paper's Fig 5.

Exit code: 0 on success, nonzero on any read/aggregation failure. Deterministic
across reruns (same input ledger → identical bytes on disk aside from the
regenerated_at timestamp, which is the only non-reproducible field).
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LEDGER = REPO_ROOT / "evidence" / "expansion_500" / "expansion_500_ledger.jsonl"
OUT_JSON = (
    REPO_ROOT / "evidence" / "expansion_500" / "analysis" / "per_verdict_per_grade.json"
)

VERDICTS = ["SUPPORTED", "UNSUPPORTED", "NOT_APPLICABLE", "CONTESTED"]
ISUP_GRADES = [0, 1, 2, 3, 4, 5]


def load_ledger(path: pathlib.Path) -> list[dict]:
    with path.open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if len(rows) != 500:
        raise RuntimeError(
            f"expansion_500 ledger expected 500 rows, got {len(rows)} at {path}"
        )
    return rows


def stratify(rows: list[dict]) -> dict:
    """Compute the (verdict, prediction, truth) joint distribution + aggregates."""
    verdict_n: Counter = Counter()
    verdict_correct: Counter = Counter()
    verdict_conf_sum: dict[str, float] = defaultdict(float)
    verdict_pred_grade: dict[str, Counter] = {v: Counter() for v in VERDICTS}
    verdict_truth_grade: dict[str, Counter] = {v: Counter() for v in VERDICTS}
    verdict_joint: dict[str, dict[str, int]] = {
        v: {f"pred={p}|truth={t}": 0 for p in ISUP_GRADES for t in ISUP_GRADES}
        for v in VERDICTS
    }

    per_grade_recall_correct: Counter = Counter()
    per_grade_recall_total: Counter = Counter()

    supported_rows: list[dict] = []

    for r in rows:
        v = r["verdict"]
        pred = int(r["extra"]["isup_prediction"])
        truth = int(r["extra"]["ground_truth_isup"])
        conf = float(r["extra"]["confidence"])
        correct = int(pred == truth)

        verdict_n[v] += 1
        verdict_correct[v] += correct
        verdict_conf_sum[v] += conf
        verdict_pred_grade[v][pred] += 1
        verdict_truth_grade[v][truth] += 1
        verdict_joint[v][f"pred={pred}|truth={truth}"] += 1
        per_grade_recall_total[truth] += 1
        if correct:
            per_grade_recall_correct[truth] += 1

        if v == "SUPPORTED":
            supported_rows.append(
                {
                    "claim_id": r["claim_id"],
                    "predicted_isup": pred,
                    "ground_truth_isup": truth,
                    "confidence": conf,
                    "correct": bool(correct),
                }
            )

    per_verdict = {}
    for v in VERDICTS:
        n = verdict_n[v]
        if n == 0:
            per_verdict[v] = {
                "n": 0,
                "correct": 0,
                "accuracy": None,
                "mean_confidence": None,
                "predicted_grade_counts": {str(g): 0 for g in ISUP_GRADES},
                "truth_grade_counts": {str(g): 0 for g in ISUP_GRADES},
            }
            continue
        per_verdict[v] = {
            "n": n,
            "correct": verdict_correct[v],
            "accuracy": verdict_correct[v] / n,
            "mean_confidence": verdict_conf_sum[v] / n,
            "predicted_grade_counts": {
                str(g): verdict_pred_grade[v].get(g, 0) for g in ISUP_GRADES
            },
            "truth_grade_counts": {
                str(g): verdict_truth_grade[v].get(g, 0) for g in ISUP_GRADES
            },
            "joint_pred_truth": verdict_joint[v],
        }

    per_grade_recall = {
        str(g): {
            "n_slides_with_true_grade": per_grade_recall_total[g],
            "correct": per_grade_recall_correct[g],
            "recall": (
                per_grade_recall_correct[g] / per_grade_recall_total[g]
                if per_grade_recall_total[g]
                else None
            ),
        }
        for g in ISUP_GRADES
    }

    recalls = [
        per_grade_recall_correct[g] / per_grade_recall_total[g]
        for g in ISUP_GRADES
        if per_grade_recall_total[g]
    ]
    balanced_accuracy = sum(recalls) / len(recalls) if recalls else None

    raw_accuracy = sum(verdict_correct.values()) / len(rows)

    return {
        "cohort_id": "expansion-500",
        "ledger_path": str(LEDGER.relative_to(REPO_ROOT)),
        "n_slides": len(rows),
        "backbone_version": rows[0]["backbone_version"],
        "gate_rule_id": rows[0]["gate_rule_id"],
        "mil_version": rows[0]["extra"]["mil_version"],
        "raw_accuracy": raw_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "per_verdict": per_verdict,
        "per_grade_recall": per_grade_recall,
        "supported_slide_details": supported_rows,
        "regenerated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    if not LEDGER.exists():
        print(f"ERROR: ledger not found at {LEDGER}", file=sys.stderr)
        return 2

    rows = load_ledger(LEDGER)
    result = stratify(rows)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")

    # Console summary — the numbers below appear as-is in Table 1 of the paper.
    pv = result["per_verdict"]
    print(f"Wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"Cohort: {result['cohort_id']}  N={result['n_slides']}")
    print(f"Raw accuracy: {result['raw_accuracy']:.4f}")
    print(f"Balanced accuracy: {result['balanced_accuracy']:.4f}")
    print("Per-verdict:")
    for v in VERDICTS:
        d = pv[v]
        if d["n"] == 0:
            print(f"  {v:16s}  N={d['n']:4d}")
        else:
            print(
                f"  {v:16s}  N={d['n']:4d}  correct={d['correct']:4d}  "
                f"acc={d['accuracy']:.4f}  mean_conf={d['mean_confidence']:.4f}"
            )
    print(f"SUPPORTED predicted-grade distribution: "
          f"{pv['SUPPORTED']['predicted_grade_counts']}")
    print(f"SUPPORTED truth-grade distribution:     "
          f"{pv['SUPPORTED']['truth_grade_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
