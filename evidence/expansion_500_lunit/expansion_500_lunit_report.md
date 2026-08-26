# expansion-500 — lunit-dino report

> **Second-backbone validation run.** Same cohort as expansion-500
> (same 500 slides, same CV protocol, same gate thresholds). Only
> change: backbone swapped Phikon-v2 (1024-dim, pathology-specific)
> -> Lunit-DINO (1024-dim, general-purpose Meta AI ViT-L/14).
> Compare verdict distributions with expansion_500_ledger.jsonl via
> p5/analysis/cross_backbone_agreement.py.

- generated: 2026-08-23T04:44:33.716113+00:00
- cohort_id: `expansion-500`
- backbone_version: `lunit-dino-1aurent@2026-08-23`
- output_dim: `384`
- mil_version: `trained@expansion-500-lunit-cv5-seed42-lunit`
- gate_rule_id: `confidence-threshold-v1@2026-08-05`
- slides processed: 500
- slides skipped: 0

## Verdict distribution

- SUPPORTED: 24
- UNSUPPORTED: 75
- NOT_APPLICABLE: 0
- CONTESTED: 401

## Per-grade metrics (5-fold CV honest predictions)

- macro_accuracy: 0.38080424224287035
- per_grade_accuracy: [0.7089552238805971, 0.5193798449612403, 0.18461538461538463, 0.234375, 0.2, 0.4375]
- per_grade_auroc: [0.860309110186771, 0.7429741532418145, 0.6982847038019452, 0.7458787270642202, 0.7038257575757576, 0.9061578171091446]

## Wall-clocks

- embed_wall_clock_seconds: 1880.68
- mil_wall_clock_seconds: 0.32
- total_wall_minutes: 31.35

## Cost accounting

- backend: Kaggle Notebook (free tier)
- cost_actual_usd: 0.00  (Kaggle is free)
- cost_proxy_usd: 1.3062
- cost_meter_budget_cap_usd: 300.00
- budget_remaining_usd: 298.6938

## Next step

Download expansion_500_lunit_ledger.jsonl to
evidence/expansion_500_hoptimus0/expansion_500_hoptimus0_ledger.jsonl
then run: python3 p5/analysis/cross_backbone_agreement.py
