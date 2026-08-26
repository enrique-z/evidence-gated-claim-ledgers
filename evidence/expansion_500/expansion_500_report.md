# expansion-500 report

> **This is the paper's headline table.** The verdict distribution below
> IS what the Aug 29 paper reports. Every number here traces to a real
> 5-fold CV prediction on a real PANDA slide; every ledger row pins its
> backbone_version, mil_version, and gate_rule_id. The Anti-Fabrication
> Guard runs on the manuscript at build time (#34) and requires every
> numeric claim in the paper to resolve to a `SUPPORTED` ledger entry —
> or, in the case of a verdict-distribution table, to a directly-derivable
> aggregation over ledger rows.

- generated: 2026-08-10T16:17:38.553920+00:00
- cohort_id: `expansion-500`
- slides processed: 500
- slides skipped (no tissue / read error): 0
- backbone_version: `phikon-v2-owkin@2026-08-05`
- mil_version: `trained@expansion-500-cv5-seed42`
- gate_rule_id: `confidence-threshold-v1@2026-08-05`

## Verdict distribution (HEADLINE)

- SUPPORTED: 9
- UNSUPPORTED: 126
- NOT_APPLICABLE: 0
- CONTESTED: 365

## Per-grade metrics (one-vs-rest, 5-fold CV honest predictions)

- macro_accuracy: 0.3651533255012949
- per_grade_accuracy: [0.7238805970149254, 0.46511627906976744, 0.07692307692307693, 0.3125, 0.21666666666666667, 0.3958333333333333]
- per_grade_auroc: [0.8451390588043389, 0.7199063916922627, 0.6659946949602122, 0.724663130733945, 0.7231818181818181, 0.846469395280236]

## Wall-clocks

- embed_wall_clock_seconds: 11144.45
- mil_wall_clock_seconds: 0.41
- total_wall_minutes: 185.75

## Cost accounting

- backend: Kaggle Notebook (free tier)
- cost_actual_usd: 0.00  (Kaggle is free at expansion volume)
- cost_proxy_usd: 7.7395  (wall-time × $2.50/hr nominal T4-equivalent)
- cost_meter_budget_cap_usd: 300.00
- budget_remaining_usd: 292.2605

## Consistency with pilot-50

This run uses the same method as pilot-50 (`p5/orchestrator/notebooks/pilot_50.py`):
- Same backbone (Phikon-v2, same version pin)
- Same MIL architecture (mean-pool + linear head)
- Same CV protocol (5 folds, seed 42)
- Same gate thresholds (high=0.70, low=0.30)
- Same version-pinning convention
Only differences: cohort slice, cohort_id, MIL run_id in the version string.

## Anti-fabrication provenance

- Every per-slide prediction from a MIL head trained on the OTHER 4 folds
- Every ledger row pins backbone_version, mil_version, gate_rule_id
- Total ledger rows this run: 500
