# pilot-50 report

> **⚠️ NOT FOR ABSTRACT.** The Aug 20 abstract is framework-first with
> zero numeric claims. No number in this report may appear in the abstract.
> Numbers here seed downstream expansion-500 (#32) and inform the paper's
> Method / Experiments section only after full expansion.

- generated: 2026-08-10T11:47:41.135200+00:00
- cohort_id: `pilot-50`
- slides processed: 50
- backbone_version: `phikon-v2-owkin@2026-08-05`
- mil_version: `trained@pilot-50-cv5-seed42`
- gate_rule_id: `confidence-threshold-v1@2026-08-05`

## Verdict distribution

- SUPPORTED: 0
- UNSUPPORTED: 6
- NOT_APPLICABLE: 0
- CONTESTED: 44

## Per-grade metrics (one-vs-rest, 5-fold CV honest predictions)

- macro_accuracy: 0.20454545454545456
- per_grade_accuracy: [0.3, 0.2, 0.18181818181818182, 0.5454545454545454, 0.0, 0.0]
- per_grade_auroc: [0.5625, 0.725, 0.45454545454545453, 0.6317016317016317, 0.6195652173913043, 0.7282608695652174]

## Wall-clocks

- embed_wall_clock_seconds: 1086.97
- mil_wall_clock_seconds: 0.24
- total_wall_minutes: 18.12

## Cost accounting

- backend: Kaggle Notebook (free tier)
- cost_actual_usd: 0.00  (Kaggle is free at pilot volume)
- cost_proxy_usd: 0.7550  (wall-time × $2.50/hr nominal T4-equivalent)
- cost_meter_budget_cap_usd: 300.00
- budget_remaining_usd: 299.2450

## Anti-fabrication provenance

- Every per-slide prediction comes from a MIL head trained on the OTHER
  4 folds — the slide was NOT in its own training set.
- Every ledger row pins backbone_version, mil_version, gate_rule_id.
- The MIL version string includes the training-run id and seed so any
  prediction can be traced back to an exact model.
- Total ledger rows this run: 50
