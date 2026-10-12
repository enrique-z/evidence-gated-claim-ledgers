"""Generate camera-ready numbers JSON for the ASCI EGCL paper.

Computes:
  - ABMIL tau_high sweep table (for R2 reviewer appendix)
  - Key keyed aggregates for \AggNum macro verification
  - Per-institution mean-pool breakdown (for R1 reviewer appendix)

Writes evidence/asci_paper/camera_ready_numbers.json.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
from collections import Counter
from datetime import datetime, timezone

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ABMIL_LEDGER = REPO_ROOT / "evidence" / "expansion_500" / "abmil_ledger.jsonl"
MEANPOOL_LEDGER = REPO_ROOT / "evidence" / "expansion_500" / "expansion_500_ledger.jsonl"
MIL_COMPARISON = REPO_ROOT / "evidence" / "expansion_500" / "analysis" / "mil_head_comparison.json"
PER_INSTITUTION = REPO_ROOT / "evidence" / "expansion_500" / "analysis" / "per_institution.json"
OUT_JSON = REPO_ROOT / "evidence" / "asci_paper" / "camera_ready_numbers.json"

Z = 1.959964  # 95% two-sided


def _load(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                j = row.get("justification", {})
                if j.get("is_tracer", False):
                    continue
                rows.append(row)
    return rows


def wilson_ci(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + Z ** 2 / n
    center = (p + Z ** 2 / (2 * n)) / denom
    half = Z / denom * math.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2))
    return (p, max(0.0, center - half), min(1.0, center + half))


def abmil_threshold_sweep(rows: list[dict], tau_high_values: list[float], tau_low: float = 0.30) -> list[dict]:
    """Sweep tau_high for ABMIL ledger, tau_low fixed at 0.30."""
    results = []
    n_total = len(rows)
    for tau_high in tau_high_values:
        supported = []
        for r in rows:
            conf = float(r["extra"]["confidence"])
            pred = int(r["extra"]["isup_prediction"])
            truth = int(r["extra"]["ground_truth_isup"])
            if conf >= tau_high:
                supported.append({"pred": pred, "truth": truth})
        n_supp = len(supported)
        coverage = n_supp / n_total if n_total else 0.0
        correct = sum(1 for s in supported if s["pred"] == s["truth"])
        # Missed cancers: predicted grade 0, truth >= 1 (in SUPPORTED)
        missed_cancer = sum(1 for s in supported if s["pred"] == 0 and s["truth"] >= 1)
        accuracy = correct / n_supp if n_supp else None
        results.append({
            "tau_high": tau_high,
            "tau_low": tau_low,
            "released_n": n_supp,
            "coverage": round(coverage, 4),
            "coverage_pct": round(coverage * 100, 1),
            "released_accuracy": round(accuracy, 3) if accuracy is not None else None,
            "missed_cancers_released": missed_cancer,
            "correct": correct,
        })
    return results


def compute_keyed_aggregates(abmil_rows: list[dict], meanpool_rows: list[dict]) -> dict:
    """Compute a keyed dict for \AggNum macro verification."""
    agg = {}

    # ABMIL per-verdict at default tau_high=0.70, tau_low=0.30
    for tau_high, tau_low in [(0.70, 0.30)]:
        supp_n = supp_correct = 0
        supp_pred0_cancer = 0
        for r in abmil_rows:
            conf = float(r["extra"]["confidence"])
            pred = int(r["extra"]["isup_prediction"])
            truth = int(r["extra"]["ground_truth_isup"])
            v = r["verdict"]  # use stored verdict (threshold already applied)
            if v == "SUPPORTED":
                supp_n += 1
                if pred == truth:
                    supp_correct += 1
                if pred == 0 and truth >= 1:
                    supp_pred0_cancer += 1

        acc_supp, lo_supp, hi_supp = wilson_ci(supp_correct, supp_n)
        # ABMIL supported = grade-0 predictions
        abmil_supported_grade0 = sum(
            1 for r in abmil_rows
            if r["verdict"] == "SUPPORTED" and int(r["extra"]["isup_prediction"]) == 0
        )
        agg["abmil.supported.n"] = supp_n
        agg["abmil.supported.acc"] = round(acc_supp, 3)
        agg["abmil.supported.wilson.lo"] = round(lo_supp, 3)
        agg["abmil.supported.wilson.hi"] = round(hi_supp, 3)
        agg["abmil.supported.grade0.n"] = abmil_supported_grade0
        agg["abmil.supported.missed_cancer"] = supp_pred0_cancer
        mc_pct, mc_lo, mc_hi = wilson_ci(supp_pred0_cancer, abmil_supported_grade0)
        agg["abmil.supported.missed_cancer.pct"] = round(mc_pct * 100, 1)
        agg["abmil.supported.missed_cancer.wilson.lo.pct"] = round(mc_lo * 100, 1)
        agg["abmil.supported.missed_cancer.wilson.hi.pct"] = round(mc_hi * 100, 1)

    # Mean-pool per-verdict at default tau_high=0.70, tau_low=0.30
    mp_supp_n = mp_supp_correct = 0
    for r in meanpool_rows:
        if r["verdict"] == "SUPPORTED":
            mp_supp_n += 1
            if int(r["extra"]["isup_prediction"]) == int(r["extra"]["ground_truth_isup"]):
                mp_supp_correct += 1
    mp_acc, mp_lo, mp_hi = wilson_ci(mp_supp_correct, mp_supp_n)
    agg["meanpool.supported.n"] = mp_supp_n
    agg["meanpool.supported.acc"] = round(mp_acc, 3)
    agg["meanpool.supported.wilson.lo"] = round(mp_lo, 3)
    agg["meanpool.supported.wilson.hi"] = round(mp_hi, 3)

    # ABMIL contested Wilson (from headline_aggregates)
    c_n = c_correct = 0
    for r in abmil_rows:
        if r["verdict"] == "CONTESTED":
            c_n += 1
            if int(r["extra"]["isup_prediction"]) == int(r["extra"]["ground_truth_isup"]):
                c_correct += 1
    _, c_lo, c_hi = wilson_ci(c_correct, c_n)
    agg["abmil.contested.wilson.lo"] = round(c_lo, 3)
    agg["abmil.contested.wilson.hi"] = round(c_hi, 3)

    # Cohort
    agg["expansion500.n"] = len(abmil_rows)

    # ABMIL raw accuracy
    abmil_correct = sum(
        1 for r in abmil_rows
        if int(r["extra"]["isup_prediction"]) == int(r["extra"]["ground_truth_isup"])
    )
    agg["abmil.raw_accuracy"] = round(abmil_correct / len(abmil_rows), 3)

    # Mean-pool total release fraction
    agg["meanpool.coverage_pct"] = round(mp_supp_n / len(meanpool_rows) * 100, 1)

    # ABMIL total release count at 90
    agg["abmil.release_count_default"] = sum(1 for r in abmil_rows if r["verdict"] == "SUPPORTED")

    return agg


def main() -> int:
    if not ABMIL_LEDGER.exists():
        print(f"ERROR: missing {ABMIL_LEDGER}", file=sys.stderr)
        return 2
    if not MEANPOOL_LEDGER.exists():
        print(f"ERROR: missing {MEANPOOL_LEDGER}", file=sys.stderr)
        return 2

    abmil_rows = _load(ABMIL_LEDGER)
    meanpool_rows = _load(MEANPOOL_LEDGER)

    # ABMIL threshold sweep (tau_low=0.30 fixed, per PLAN.md reviewer appendix)
    sweep = abmil_threshold_sweep(
        abmil_rows,
        tau_high_values=[0.50, 0.55, 0.60, 0.65, 0.70],
        tau_low=0.30,
    )

    # Keyed aggregates for \AggNum macro
    keyed = compute_keyed_aggregates(abmil_rows, meanpool_rows)

    # Per-institution (mean-pool only; ABMIL lacks data_provider)
    per_inst = {}
    if PER_INSTITUTION.exists():
        with PER_INSTITUTION.open() as f:
            per_inst = json.load(f)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "abmil_tau_high_sweep": sweep,
        "keyed_aggregates": keyed,
        "per_institution_meanpool": per_inst.get("mean_pool", {}),
        "note": "Numbers derived from committed ledgers; skip rows where justification.is_tracer=True",
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    print(f"Wrote {OUT_JSON}")
    print("\nABMIL tau_high sweep:")
    for row in sweep:
        acc_str = f"{row['released_accuracy']:.3f}" if row['released_accuracy'] is not None else "N/A"
        print(f"  tau_high={row['tau_high']:.2f}  N={row['released_n']:3d}  "
              f"coverage={row['coverage_pct']:.1f}%  "
              f"acc={acc_str}  "
              f"missed_cancer={row['missed_cancers_released']}")
    print("\nKeyed aggregates:")
    for k, v in sorted(keyed.items()):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
