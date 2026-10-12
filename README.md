# EGCL — Evidence-Gated Claim Ledgers

Code, ledgers, and manuscript source for *"Evidence-Gated Claim Ledgers: A Reporting Standard
for Pathology Foundation Models"*, Enrique Zueco, David Scott Lewis, Haley Yi (AIXC Research).
AI at Scale for Clinical Impact (ASCI) workshop @ NeurIPS 2026, PMLR.

- Camera-ready paper: [`papers/p5-asci/main_camera_ready.pdf`](papers/p5-asci/main_camera_ready.pdf)
- The commit the camera-ready PDF is built from is tagged `v1.0-camera-ready`.
- How to cite: see [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" button).

Everything a reader needs to re-derive **every number, table, and figure in the paper** is in
this repository. No external download is required for the analyses.

## Layout

```
p5/                     the EGCL implementation
  evidence_gate/        Claim -> Verdict gate rules (confidence-threshold, entropy,
                        margin, split-conformal LAC) + the four-value vocabulary
  claim_ledger/         append-only ledger writer + schema
  antifab_guard/        build-time guard: refuses any manuscript number that cannot
                        resolve to a ledger row (+ its adversarial fixtures and tests)
  analysis/             one script per table/figure in the paper
  backbone_embedder/, patch_extractor/, mil_classifier/, feature_cache/, data_loader/
  orchestrator/         Kaggle notebook scripts used to produce the ledgers

evidence/               the ledgers the paper is bound to (8,550 rows total)
  expansion_500/        Phikon-v2 mean-pool (500), ABMIL (500), CLAM-SB (500),
                        analysis/ (alt-gate + conformal-variant ledgers and summaries)
  pilot_50/             pilot cohort (50)
  expansion_500_phikonv1|_lunit|_dinov2l/   cross-backbone runs (500 each)
  cross_backbone/       joined 4-way agreement analysis
  asci_paper/           headline aggregates, guard-ablation record, a guard run log

scripts/verify_safenum.py   record-scoped check: every \AggNum{key}{value} in the manuscript
                        must equal the value stored under that key in
                        evidence/asci_paper/camera_ready_numbers.json

papers/p5-asci/         main_camera_ready.tex/.pdf (camera-ready, refs_cr.bib,
                        build_camera_ready.sh); main.tex/build.sh are the version
                        submitted for review, kept unchanged for the record
```

## Reproducing the paper's numbers

```bash
# every headline aggregate (per-verdict accuracy, Wilson intervals, class counts)
python3 p5/analysis/headline_aggregates.py

# MIL-head comparison (Table 2) and cross-backbone agreement (Table 4)
python3 p5/analysis/mil_head_comparison.py
python3 p5/analysis/cross_backbone_agreement.py

# alternative gate rules, and the cross-fold + Mondrian conformal variants (Appendix F)
python3 p5/analysis/run_alt_gates.py
python3 p5/analysis/run_conformal_variants.py

# figures
python3 p5/analysis/fig5_verdict_by_grade.py
python3 p5/analysis/risk_coverage.py
python3 p5/analysis/threshold_sensitivity.py
```

Each script reads the JSONL ledgers under `evidence/` and writes to `evidence/.../analysis/`.
No script contains hard-coded result arrays.

## Rebuilding the manuscript (runs the anti-fabrication guard)

```bash
bash papers/p5-asci/build_camera_ready.sh
```

This runs, in order: regeneration of `camera_ready_numbers.json` from the ledgers; the keyed
`\AggNum` check (`scripts/verify_safenum.py`); the anti-fabrication guard over all 8,550 ledger entries; a check that the
five enforcement layers are active; a 4-pass `pdflatex` + `bibtex` build; a page-budget check
against the venue's 9-page body cap; and an author-block check. **The build fails if any numeric
claim in the manuscript cannot be resolved to a ledger row.** Requires a TeX distribution and
`pdftotext`/`pdfinfo` (poppler).

## Data

Slide images come from the public **PANDA** challenge cohort
(https://www.kaggle.com/c/prostate-cancer-grade-assessment). They are not redistributed here.
Patch extraction and backbone embedding were run on Kaggle; the resulting per-slide predictions
and verdicts are what this repository ships, as ledgers. Cached `.npz` feature tensors
(≈69 MB) are omitted from this repository for size — they are needed only to *re-train* the MIL
heads, not to reproduce any number, table, or figure in the paper.

## A note on one directory name

`evidence/expansion_500_hoptimus0/` is a **stale directory name**: it holds a DINOv2-L run, not
H-optimus-0 (which is access-gated and was never run). See the `README.md` inside it. The paper
cites three additional backbones — Phikon-v1, Lunit-DINO, and DINOv2-L — and does not use this
directory.

## Tests

```bash
python3 -m pytest -q p5/antifab_guard/tests
```

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). The ledgers under `evidence/`
are derived from the PANDA challenge data and are shared for research reproducibility; use of
the underlying slides and labels is governed by the PANDA data terms.
