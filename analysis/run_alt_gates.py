"""Phase A2 — Run the three alternative gate rules over the expansion-500 cohort.

Reads:
  - evidence/expansion_500/expansion_500_ledger.jsonl  (test rows)
  - evidence/pilot_50/pilot_50_ledger.jsonl            (conformal calibration)

Writes:
  - evidence/expansion_500/analysis/alt_gates_ledger.jsonl
      One row per (slide, gate_rule) triple — 500 * 3 = 1500 rows.
      Each row conforms to the same LedgerRow schema as the original
      expansion_500 ledger, with the `gate_rule_id` field pinned to the
      rule that produced the row.
  - evidence/expansion_500/analysis/alt_gates_summary.json
      Per-rule verdict distribution and per-verdict accuracy, computed
      identically to `per_verdict_per_grade.py`.

Sanity checks (script exits nonzero if any fail):
  - each rule contributes exactly 500 rows
  - each rule's verdict counts sum to 500
  - conformal calibration used pilot-50 (n=50)
  - the four verdicts in each row are drawn from the fixed vocabulary

The three alternative rules are already unit-tested in
`p5/evidence_gate/tests/test_alt_gates.py`.
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

from p5.evidence_gate.alt_gates import (
    ConformalGate,
    EntropyGate,
    MarginGate,
    calibrate_lac,
)
from p5.evidence_gate.gate import Claim

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXPANSION = REPO_ROOT / "evidence" / "expansion_500" / "expansion_500_ledger.jsonl"
PILOT = REPO_ROOT / "evidence" / "pilot_50" / "pilot_50_ledger.jsonl"
OUT_LEDGER = REPO_ROOT / "evidence" / "expansion_500" / "analysis" / "alt_gates_ledger.jsonl"
OUT_SUMMARY = REPO_ROOT / "evidence" / "expansion_500" / "analysis" / "alt_gates_summary.json"

CONFORMAL_ALPHA = 0.10


def _load(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def _claim_from_row(row: dict) -> Claim:
    return Claim(
        claim_id=row["claim_id"],
        claim_text=row["claim_text"],
        evidence_ref=row["evidence_ref"],
        model_output={
            "per_grade_probs": row["extra"]["per_grade_probs"],
            "confidence": row["extra"]["confidence"],
            "isup_grade": row["extra"]["isup_prediction"],
        },
        backbone_version=row["backbone_version"],
        cohort_id=row["cohort_id"],
    )


def _emit_row(
    orig: dict, gate, verdict: str, justification, run_timestamp: str
) -> dict:
    """Build a ledger row for (slide, gate_rule)."""
    return {
        "claim_id": f"{orig['claim_id']}#{gate.GATE_RULE_ID}",
        "claim_text": orig["claim_text"],
        "evidence_ref": orig["evidence_ref"],
        "verdict": verdict,
        "justification": {
            "gate_rule_id": justification.gate_rule_id,
            "source_rows": list(justification.source_rows),
            "thresholds": {k: (list(v) if isinstance(v, list) else v)
                           for k, v in justification.thresholds.items()},
            "mil_version": orig["extra"]["mil_version"],
            "cv_fold": orig["justification"].get("cv_fold"),
            "patch_count": orig["justification"].get("patch_count"),
            "is_tracer": orig["justification"].get("is_tracer", False),
        },
        "timestamp": run_timestamp,
        "backbone_version": orig["backbone_version"],
        "cohort_id": orig["cohort_id"],
        "gate_rule_id": gate.GATE_RULE_ID,
        "extra": {
            "isup_prediction": orig["extra"]["isup_prediction"],
            "confidence": orig["extra"]["confidence"],
            "per_grade_probs": orig["extra"]["per_grade_probs"],
            "mil_version": orig["extra"]["mil_version"],
            "ground_truth_isup": orig["extra"]["ground_truth_isup"],
            "ground_truth_gleason": orig["extra"].get("ground_truth_gleason"),
        },
    }


def _summarise(rows: list[dict]) -> dict:
    by_rule: dict[str, dict] = defaultdict(lambda: {
        "n": 0,
        "verdict_counts": Counter(),
        "verdict_correct": Counter(),
        "verdict_pred_grade": defaultdict(Counter),
        "verdict_truth_grade": defaultdict(Counter),
        "verdict_conf_sum": defaultdict(float),
    })
    for r in rows:
        d = by_rule[r["gate_rule_id"]]
        v = r["verdict"]
        pred = int(r["extra"]["isup_prediction"])
        truth = int(r["extra"]["ground_truth_isup"])
        conf = float(r["extra"]["confidence"])
        d["n"] += 1
        d["verdict_counts"][v] += 1
        if pred == truth:
            d["verdict_correct"][v] += 1
        d["verdict_pred_grade"][v][pred] += 1
        d["verdict_truth_grade"][v][truth] += 1
        d["verdict_conf_sum"][v] += conf

    out: dict[str, dict] = {}
    for rule, d in by_rule.items():
        per_verdict = {}
        for v in ["SUPPORTED", "UNSUPPORTED", "NOT_APPLICABLE", "CONTESTED"]:
            n = d["verdict_counts"][v]
            per_verdict[v] = {
                "n": n,
                "correct": d["verdict_correct"][v],
                "accuracy": (d["verdict_correct"][v] / n) if n else None,
                "mean_confidence": (d["verdict_conf_sum"][v] / n) if n else None,
                "predicted_grade_counts": {
                    str(g): d["verdict_pred_grade"][v].get(g, 0) for g in range(6)
                },
                "truth_grade_counts": {
                    str(g): d["verdict_truth_grade"][v].get(g, 0) for g in range(6)
                },
            }
        out[rule] = {"n_rows": d["n"], "per_verdict": per_verdict}
    return out


def main() -> int:
    if not EXPANSION.exists() or not PILOT.exists():
        print(f"ERROR: missing ledger inputs.", file=sys.stderr)
        return 2

    exp_rows = _load(EXPANSION)
    pil_rows = _load(PILOT)
    if len(exp_rows) != 500:
        print(f"ERROR: expansion ledger has {len(exp_rows)} rows, expected 500",
              file=sys.stderr)
        return 3
    if len(pil_rows) != 50:
        print(f"ERROR: pilot ledger has {len(pil_rows)} rows, expected 50",
              file=sys.stderr)
        return 3

    # Conformal calibration on pilot-50 (LAC, alpha grid); pick alpha=0.10
    # as the operating point for the ledger; alpha=0.05, 0.20 quantiles are
    # kept in the calibration record for figure 2 (Phase A3).
    cals = calibrate_lac(pil_rows, alphas=(0.05, 0.10, 0.20),
                         calibration_source="pilot-50")
    conformal_gate = ConformalGate(cals[CONFORMAL_ALPHA])
    entropy_gate = EntropyGate()
    margin_gate = MarginGate()
    gates = [entropy_gate, margin_gate, conformal_gate]

    run_timestamp = datetime.now(timezone.utc).isoformat()

    OUT_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with OUT_LEDGER.open("w") as fh:
        for r in exp_rows:
            claim = _claim_from_row(r)
            for gate in gates:
                verdict, justification = gate.gate(claim)
                row = _emit_row(r, gate, verdict, justification, run_timestamp)
                fh.write(json.dumps(row, sort_keys=True))
                fh.write("\n")
                n_written += 1

    expected = 500 * len(gates)
    if n_written != expected:
        print(f"ERROR: wrote {n_written} rows, expected {expected}", file=sys.stderr)
        return 4

    # Re-load and summarise (proves the written ledger is legal JSONL).
    all_rows = _load(OUT_LEDGER)
    summary = _summarise(all_rows)

    for rule_id, s in summary.items():
        if s["n_rows"] != 500:
            print(f"ERROR: rule {rule_id} produced {s['n_rows']} rows (want 500)",
                  file=sys.stderr)
            return 5
        counts_sum = sum(s["per_verdict"][v]["n"]
                         for v in ["SUPPORTED", "UNSUPPORTED", "NOT_APPLICABLE",
                                   "CONTESTED"])
        if counts_sum != 500:
            print(f"ERROR: rule {rule_id} verdict counts sum to {counts_sum}",
                  file=sys.stderr)
            return 6

    # Persist summary + calibration record.
    with OUT_SUMMARY.open("w") as fh:
        json.dump(
            {
                "cohort_id": "expansion-500",
                "test_ledger": str(EXPANSION.relative_to(REPO_ROOT)),
                "calibration_ledger": str(PILOT.relative_to(REPO_ROOT)),
                "calibration": {
                    f"alpha={a:.2f}": cals[a].to_dict() for a in cals
                },
                "conformal_operating_alpha": CONFORMAL_ALPHA,
                "gate_rules": list(summary.keys()),
                "summary": summary,
                "regenerated_at": run_timestamp,
            },
            fh, indent=2, sort_keys=True,
        )
        fh.write("\n")

    print(f"Wrote {OUT_LEDGER.relative_to(REPO_ROOT)}  ({n_written} rows)")
    print(f"Wrote {OUT_SUMMARY.relative_to(REPO_ROOT)}")
    for rule, s in summary.items():
        pv = s["per_verdict"]
        print(f"\n{rule}")
        for v in ["SUPPORTED", "UNSUPPORTED", "NOT_APPLICABLE", "CONTESTED"]:
            d = pv[v]
            if d["n"] == 0:
                print(f"  {v:16s}  N={d['n']:4d}")
            else:
                print(f"  {v:16s}  N={d['n']:4d}  correct={d['correct']:4d}  "
                      f"acc={d['accuracy']:.4f}")
    print(f"\nConformal calibration quantiles (LAC, n={cals[0.10].n_calibration}):")
    for a in sorted(cals.keys()):
        print(f"  alpha={a:.2f}  q_alpha={cals[a].q_alpha:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
