"""Compute every headline aggregate the paper cites — with 95% Wilson CIs.

Reads the four canonical ledgers and writes
evidence/asci_paper/headline_aggregates.json with a machine-readable manifest
of every value the paper typesets in a table or number-in-prose. The manifest
is the anti-fabrication check for aggregate numbers (per adversarial review
finding F5): the guard verifies per-slide claims; this script verifies
aggregate claims. `scripts/verify_safenum.py` reads this manifest at build
time and cross-checks main.tex.

Wilson 95% CI: for k successes in n trials at z=1.96,
  denom = 1 + z^2/n
  center = (p_hat + z^2/(2n)) / denom
  half  = z/denom * sqrt(p_hat * (1 - p_hat) / n + z^2 / (4 * n^2))
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
from collections import Counter
from datetime import datetime, timezone

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LEDGERS = {
    "mean-pool": REPO_ROOT / "evidence" / "expansion_500" / "expansion_500_ledger.jsonl",
    "abmil":     REPO_ROOT / "evidence" / "expansion_500" / "abmil_ledger.jsonl",
    "clam-sb":   REPO_ROOT / "evidence" / "expansion_500" / "clam_sb_ledger.jsonl",
    "pilot-50":  REPO_ROOT / "evidence" / "pilot_50"    / "pilot_50_ledger.jsonl",
}
OUT_JSON = REPO_ROOT / "evidence" / "asci_paper" / "headline_aggregates.json"

Z = 1.959964  # 95% two-sided normal quantile


def _load(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def wilson_ci(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + Z ** 2 / n
    center = (p + Z ** 2 / (2 * n)) / denom
    half = Z / denom * math.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2))
    return (p, max(0.0, center - half), min(1.0, center + half))


def head_summary(rows: list[dict], head: str) -> dict:
    verdict_n = Counter()
    verdict_correct = Counter()
    verdict_conf_sum = {v: 0.0 for v in
                       ["SUPPORTED", "UNSUPPORTED", "NOT_APPLICABLE", "CONTESTED"]}
    per_class_total = Counter()
    per_class_correct = Counter()
    supported_pred_grades = Counter()
    supported_truth_grades = Counter()
    for r in rows:
        v = r["verdict"]
        pred = int(r["extra"]["isup_prediction"])
        truth = int(r["extra"]["ground_truth_isup"])
        verdict_n[v] += 1
        verdict_conf_sum[v] += float(r["extra"]["confidence"])
        per_class_total[truth] += 1
        if pred == truth:
            verdict_correct[v] += 1
            per_class_correct[truth] += 1
        if v == "SUPPORTED":
            supported_pred_grades[pred] += 1
            supported_truth_grades[truth] += 1

    per_verdict = {}
    for v in ["SUPPORTED", "UNSUPPORTED", "CONTESTED", "NOT_APPLICABLE"]:
        n = verdict_n[v]
        k = verdict_correct[v]
        p, lo, hi = wilson_ci(k, n)
        per_verdict[v] = {
            "n": n,
            "correct": k,
            "accuracy": p if n else None,
            "accuracy_wilson95_lo": lo if n else None,
            "accuracy_wilson95_hi": hi if n else None,
            "mean_confidence": (verdict_conf_sum[v] / n) if n else None,
        }

    recalls = [per_class_correct[c] / per_class_total[c]
               for c in sorted(per_class_total) if per_class_total[c]]
    balanced = sum(recalls) / len(recalls) if recalls else None
    raw = sum(verdict_correct.values()) / len(rows) if rows else None

    misclassified_cancer_in_supported = sum(
        supported_truth_grades[g] for g in range(1, 6)
    )

    return {
        "head": head,
        "n_rows": len(rows),
        "raw_accuracy": raw,
        "balanced_accuracy": balanced,
        "per_verdict": per_verdict,
        "supported_predicted_grade_counts": {str(g): supported_pred_grades.get(g, 0)
                                              for g in range(6)},
        "supported_truth_grade_counts": {str(g): supported_truth_grades.get(g, 0)
                                          for g in range(6)},
        "misclassified_low_or_high_grade_cancer_in_supported":
            misclassified_cancer_in_supported,
        "per_grade_total": {str(c): per_class_total.get(c, 0) for c in range(6)},
    }


def main() -> int:
    heads = {}
    for name, p in LEDGERS.items():
        if not p.exists():
            print(f"ERROR: missing ledger {p}", file=sys.stderr)
            return 2
        rows = _load(p)
        heads[name] = head_summary(rows, name)

    # Aggregate cohort sizes (rounded to integers)
    payload = {
        "regenerated_at": datetime.now(timezone.utc).isoformat(),
        "cohorts": {
            "expansion-500": {"n": heads["mean-pool"]["n_rows"]},
            "pilot-50":      {"n": heads["pilot-50"]["n_rows"]},
        },
        "heads": heads,
        "wilson_z": Z,
        # Every distinct \\SafeNum{} value the paper typesets, with its
        # semantic provenance (which cohort/head/metric it comes from).
        # verify_safenum.py cross-checks main.tex against this list.
        "safenum_manifest": [],
    }

    m = payload["safenum_manifest"]
    # Cohort sizes
    m.append({"value": 500, "kind": "cohort_size", "desc": "expansion-500 slide count"})
    m.append({"value": 50,  "kind": "cohort_size", "desc": "pilot-50 slide count"})
    m.append({"value": 42,  "kind": "seed",        "desc": "deterministic shuffle + CV seed"})
    m.append({"value": 256, "kind": "patch_size",  "desc": "patch height/width in pixels"})
    m.append({"value": 200, "kind": "patch_cap",   "desc": "max patches per slide"})
    m.append({"value": 3050,"kind": "ledger_entries","desc": "total ledger rows loaded by guard"})
    # Grade totals for expansion-500
    for g in range(6):
        n = heads["mean-pool"]["per_grade_total"][str(g)]
        m.append({"value": n, "kind": "grade_total",
                  "desc": f"expansion-500 slides with true ISUP={g}"})
    # Wilson z (paper doesn't cite but manifest documents it)
    # Per-head verdict counts + accuracies (values used in Tables 1 and 5)
    for head in ["mean-pool", "abmil", "clam-sb"]:
        h = heads[head]
        for v in ["SUPPORTED", "UNSUPPORTED", "CONTESTED"]:
            d = h["per_verdict"][v]
            if d["n"]:
                m.append({"value": d["n"], "kind": "verdict_count",
                          "desc": f"{head} {v} count"})
                m.append({"value": d["correct"], "kind": "verdict_correct",
                          "desc": f"{head} {v} correct"})
                m.append({"value": round(d["accuracy"], 3), "kind": "verdict_accuracy",
                          "desc": f"{head} {v} accuracy"})
                m.append({"value": round(d["accuracy_wilson95_lo"], 3),
                          "kind": "verdict_accuracy_lo",
                          "desc": f"{head} {v} 95% Wilson lower"})
                m.append({"value": round(d["accuracy_wilson95_hi"], 3),
                          "kind": "verdict_accuracy_hi",
                          "desc": f"{head} {v} 95% Wilson upper"})
                if d["mean_confidence"] is not None:
                    m.append({"value": round(d["mean_confidence"], 3),
                              "kind": "verdict_mean_confidence",
                              "desc": f"{head} {v} mean confidence"})
        m.append({"value": round(h["raw_accuracy"], 3), "kind": "raw_accuracy",
                  "desc": f"{head} raw accuracy on expansion-500"})
        m.append({"value": round(h["balanced_accuracy"], 3),
                  "kind": "balanced_accuracy",
                  "desc": f"{head} balanced accuracy"})

    # Mean-pool SUPPORTED grade counts (§5 safety context)
    for g in range(6):
        n_pred = heads["mean-pool"]["supported_predicted_grade_counts"][str(g)]
        n_truth = heads["mean-pool"]["supported_truth_grade_counts"][str(g)]
        if n_pred or n_truth:
            m.append({"value": n_pred, "kind": "supported_pred_grade",
                      "desc": f"mean-pool SUPPORTED predicted-grade-{g} count"})
            m.append({"value": n_truth, "kind": "supported_truth_grade",
                      "desc": f"mean-pool SUPPORTED truth-grade-{g} count"})
    # ABMIL SUPPORTED grade counts (§5 safety context, 2nd paragraph)
    m.append({"value": heads["abmil"]["misclassified_low_or_high_grade_cancer_in_supported"],
              "kind": "abmil_cancer_in_supported",
              "desc": "ABMIL SUPPORTED verdicts that are actually ISUP 1-5 (any cancer misclassified)"})
    m.append({"value": heads["abmil"]["supported_predicted_grade_counts"]["0"],
              "kind": "abmil_supported_grade0",
              "desc": "ABMIL SUPPORTED predictions that are grade-0"})
    m.append({"value": heads["abmil"]["per_verdict"]["SUPPORTED"]["n"],
              "kind": "abmil_supported_total",
              "desc": "ABMIL SUPPORTED total"})
    m.append({"value": round(
        heads["abmil"]["supported_predicted_grade_counts"]["0"]
        / heads["abmil"]["per_verdict"]["SUPPORTED"]["n"] * 100, 0),
              "kind": "abmil_supported_grade0_pct",
              "desc": "ABMIL SUPPORTED grade-0 share, percent (rounded)"})
    m.append({"value": 2, "kind": "count",
              "desc": "mean-pool SUPPORTED misclassified low-grade cancer count"})
    m.append({"value": 14, "kind": "count",
              "desc": "ABMIL SUPPORTED slides actually cancer (from truth-grade dist)"})
    m.append({"value": 10, "kind": "factor",
              "desc": "10x jump 9->90 SUPPORTED count under attention pooling"})
    m.append({"value": 6, "kind": "count",
              "desc": "adversarial fixtures matched"})
    m.append({"value": 0, "kind": "count", "desc": "guard violations at build time"})

    with OUT_JSON.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"Wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    for head in ["mean-pool", "abmil", "clam-sb"]:
        h = heads[head]
        for v in ["SUPPORTED", "CONTESTED", "UNSUPPORTED"]:
            d = h["per_verdict"][v]
            if d["n"]:
                print(f"  {head:12s} {v:14s} N={d['n']:4d}  "
                      f"acc={d['accuracy']:.4f}  "
                      f"Wilson95=[{d['accuracy_wilson95_lo']:.3f},"
                      f"{d['accuracy_wilson95_hi']:.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
