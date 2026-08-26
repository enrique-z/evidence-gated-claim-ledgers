# expansion-500 — phikon-v1 report

> **Second-backbone validation run.** Same cohort as expansion-500
> (same 500 slides, same CV protocol, same gate thresholds). Only
> change: backbone swapped Phikon-v2 (1024-dim, pathology-specific)
> -> Phikon-v1 (1024-dim, general-purpose Meta AI ViT-L/14).
> Compare verdict distributions with expansion_500_ledger.jsonl via
> p5/analysis/cross_backbone_agreement.py.

- generated: 2026-08-23T04:44:17.062506+00:00
- cohort_id: `expansion-500`
- backbone_version: `phikon-v1-owkin@2026-08-23`
- output_dim: `768`
- mil_version: `trained@expansion-500-phikonv1-cv5-seed42-phikonv1`
- gate_rule_id: `confidence-threshold-v1@2026-08-05`
- slides processed: 500
- slides skipped: 0

## Verdict distribution

- SUPPORTED: 60
- UNSUPPORTED: 50
- NOT_APPLICABLE: 0
- CONTESTED: 390

## Per-grade metrics (5-fold CV honest predictions)

- macro_accuracy: 0.38768726248008617
- per_grade_accuracy: [0.7014925373134329, 0.4728682170542636, 0.15384615384615385, 0.21875, 0.3, 0.4791666666666667]
- per_grade_auroc: [0.8608596362450045, 0.7347416368917027, 0.643395225464191, 0.7117975917431193, 0.7596969696969696, 0.8655051622418879]

## Wall-clocks

- embed_wall_clock_seconds: 4034.69
- mil_wall_clock_seconds: 0.35
- total_wall_minutes: 67.25

## Cost accounting

- backend: Kaggle Notebook (free tier)
- cost_actual_usd: 0.00  (Kaggle is free)
- cost_proxy_usd: 2.8021
- cost_meter_budget_cap_usd: 300.00
- budget_remaining_usd: 297.1979

## Next step

Download expansion_500_phikonv1_ledger.jsonl to
evidence/expansion_500_hoptimus0/expansion_500_hoptimus0_ledger.jsonl
then run: python3 p5/analysis/cross_backbone_agreement.py
