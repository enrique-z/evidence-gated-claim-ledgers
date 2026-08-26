# READ FIRST — this folder's NAME is stale; its CONTENTS are a DINOv2-L run

**Status: not fabricated, not a fourth backbone. A stale-named duplicate.**
Written 2026-08-25 during the p5-asci R6 finalization pass, so that no future auditor
re-trips on the folder name.

## What this folder actually contains

`expansion_500_hoptimus0_ledger.jsonl` (500 rows) is the **DINOv2-L** run, not H-optimus-0.
Proof, re-derivable in one command:

```
python3 -c "import json;print({json.loads(l)['backbone_version'] for l in open('evidence/expansion_500_hoptimus0/expansion_500_hoptimus0_ledger.jsonl')})"
# -> {'dinov2-large-meta@2026-08-11'}
```

Every row also carries `extra.mil_version = trained@expansion-500-dinov2l-cv5-seed42-dinov2l`.
`cross_backbone_agreement.json` in this folder states `"second_backbone": "dinov2-large"`
and its numbers (agreement 369/500 = 0.738; S/C/U = 0/311/189) are identical to the
canonical DINOv2-L run in `evidence/expansion_500_dinov2l/`.

## Why the name is wrong

H-optimus-0 was the intended fourth backbone. It is **HuggingFace access-gated** (HTTP 401
without an approved licence request), so it was never run. The Kaggle notebook
`p5/orchestrator/notebooks/expansion_500_hoptimus0.py` was repurposed to the open
`facebook/dinov2-large` (Apache-2.0, no gate) — its *docstring is already correct*
("DINOv2-L backbone", line 1) — but the **filename and this output directory kept the old
name**. The analysis script `p5/analysis/cross_backbone_agreement.py` likewise hardcodes
this path and still says "H-optimus-0" in two operator-facing strings.

## Why the folder was NOT renamed

`p5/analysis/cross_backbone_agreement.py` hardcodes this directory at lines 27--28.
Renaming would break a committed analysis script that the paper's Reproducibility
Statement points at ("every table and figure regenerates from a script under
`p5/analysis/`"). Documenting in place gives the same protection at zero risk.

## What the paper says about it

Nothing. The manuscript cites **three** additional backbones (Phikon-v1, Lunit-DINO,
DINOv2-L) from `evidence/expansion_500_phikonv1|_lunit|_dinov2l/`, and lists H-optimus-0
nowhere. This folder is not in `papers/p5-asci/build.sh`'s guard ledger list and contributes
no number to the manuscript. **Do not cite it as a fourth backbone.**
