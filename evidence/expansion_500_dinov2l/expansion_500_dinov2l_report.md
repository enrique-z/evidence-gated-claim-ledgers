# expansion-500 — dinov2-large report

> **Second-backbone validation run.** Same cohort as expansion-500
> (same 500 slides, same CV protocol, same gate thresholds). Only
> change: backbone swapped Phikon-v2 (1024-dim, pathology-specific)
> -> DINOv2-L (1024-dim, general-purpose Meta AI ViT-L/14).
> Compare verdict distributions with expansion_500_ledger.jsonl via
> p5/analysis/cross_backbone_agreement.py.

- generated: 2026-08-23T00:33:29.304098+00:00
- cohort_id: `expansion-500`
- backbone_version: `dinov2-large-meta@2026-08-11`
- output_dim: `1024`
- mil_version: `trained@expansion-500-dinov2l-cv5-seed42-dinov2l`
- gate_rule_id: `confidence-threshold-v1@2026-08-05`
- slides processed: 500
- slides skipped: 0

## Verdict distribution

- SUPPORTED: 0
- UNSUPPORTED: 189
- NOT_APPLICABLE: 0
- CONTESTED: 311

## Per-grade metrics (5-fold CV honest predictions)

- macro_accuracy: 0.30033771493605316
- per_grade_accuracy: [0.753731343283582, 0.37209302325581395, 0.12307692307692308, 0.078125, 0.1, 0.375]
- per_grade_auroc: [0.7988337003507054, 0.6945611065839236, 0.6697435897435897, 0.6823752866972477, 0.6633712121212121, 0.8597437315634219]

## Wall-clocks

- embed_wall_clock_seconds: 14643.69
- mil_wall_clock_seconds: 0.37
- total_wall_minutes: 244.07

## Cost accounting

- backend: Kaggle Notebook (free tier)
- cost_actual_usd: 0.00  (Kaggle is free)
- cost_proxy_usd: 10.1695
- cost_meter_budget_cap_usd: 300.00
- budget_remaining_usd: 289.8305

## Next step

Download expansion_500_dinov2l_ledger.jsonl to
evidence/expansion_500_hoptimus0/expansion_500_hoptimus0_ledger.jsonl
then run: python3 p5/analysis/cross_backbone_agreement.py
