"""Phase B — Train ABMIL and CLAM-SB on cached Phikon-v2 features.

Uses the 500 cached feature files at evidence/expansion_500/feature_cache/
(each is `(N_patches, 1024)` float32). Fold assignment matches the mean-pool
baseline (`i % 5`) so per-slide predictions are directly comparable across
heads.

Writes:
  - evidence/expansion_500/abmil_ledger.jsonl        (500 rows)
  - evidence/expansion_500/clam_sb_ledger.jsonl      (500 rows)

Each row has the same schema as expansion_500_ledger.jsonl. The `verdict`
comes from the same reference gate rule (confidence-threshold-v1 at
high=0.70, low=0.30) so a head-to-head verdict-distribution comparison is
meaningful.

Runtime: ~10-15 min per head on CPU (500 slides × 5 folds × 50 epochs on a
small linear-attention model). Deterministic per seed. No re-embedding.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LEDGER = REPO_ROOT / "evidence" / "expansion_500" / "expansion_500_ledger.jsonl"
CACHE_DIR = REPO_ROOT / "evidence" / "expansion_500" / "feature_cache"
BACKBONE_TAG = "phikon-v2__phikon-v2-owkin_2026-08-05"

OUT_ABMIL = REPO_ROOT / "evidence" / "expansion_500" / "abmil_ledger.jsonl"
OUT_CLAM = REPO_ROOT / "evidence" / "expansion_500" / "clam_sb_ledger.jsonl"


def _load_ledger(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def _load_feature(slide_id: str) -> np.ndarray:
    fname = f"{slide_id}__{BACKBONE_TAG}.npz"
    d = np.load(CACHE_DIR / fname, allow_pickle=True)
    return d["features"]


def _load_slide_data(ledger_rows: list[dict]):
    slide_ids = []
    labels = []
    features = []
    ref_rows = []
    for r in ledger_rows:
        sid = r["claim_id"].split("#")[0]
        slide_ids.append(sid)
        labels.append(int(r["extra"]["ground_truth_isup"]))
        features.append(_load_feature(sid))
        ref_rows.append(r)
    return slide_ids, features, labels, ref_rows


def _emit_row(orig: dict, pred_tuple, mil_version: str, gate_rule_id: str,
              run_timestamp: str, cv_fold: int) -> dict:
    grade, conf, probs = pred_tuple
    high, low = 0.70, 0.30
    if conf >= high:
        verdict = "SUPPORTED"
    elif conf <= low:
        verdict = "UNSUPPORTED"
    else:
        verdict = "CONTESTED"
    return {
        "claim_id": f"{orig['claim_id'].split('#')[0]}#expansion-500#{run_timestamp}#{mil_version}",
        "claim_text": (
            f"slide={orig['claim_id'].split('#')[0]} "
            f"isup_prediction={grade} confidence={conf:.4f} "
            f"mil={mil_version}"
        ),
        "evidence_ref": orig["evidence_ref"],
        "verdict": verdict,
        "justification": {
            "gate_rule_id": gate_rule_id,
            "thresholds": {
                "high": high,
                "low": low,
                "observed_confidence": conf,
            },
            "mil_version": mil_version,
            "patch_count": orig["justification"].get("patch_count"),
            "is_tracer": False,
            "cv_fold": cv_fold,
        },
        "timestamp": run_timestamp,
        "backbone_version": orig["backbone_version"],
        "cohort_id": orig["cohort_id"],
        "gate_rule_id": gate_rule_id,
        "extra": {
            "isup_prediction": grade,
            "confidence": conf,
            "per_grade_probs": probs,
            "mil_version": mil_version,
            "ground_truth_isup": orig["extra"]["ground_truth_isup"],
            "ground_truth_gleason": orig["extra"].get("ground_truth_gleason"),
        },
    }


def _run_and_write(name: str, train_fn, out_path: pathlib.Path,
                   slide_ids, features, labels, ref_rows,
                   device: str) -> str:
    print(f"[{name}] training 5-fold CV on {len(slide_ids)} slides ...")
    t0 = datetime.now(timezone.utc)
    predictions, version = train_fn(
        slide_ids=slide_ids, features_per_slide=features, labels=labels,
        n_folds=5, device=device, run_id="expansion-500",
    )
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    print(f"[{name}] done in {elapsed:.1f}s — version = {version}")

    run_timestamp = datetime.now(timezone.utc).isoformat()
    gate_rule_id = "confidence-threshold-v1@2026-08-05"
    n_written = 0
    with out_path.open("w") as fh:
        for i, (pred, orig) in enumerate(zip(predictions, ref_rows)):
            row = _emit_row(
                orig=orig, pred_tuple=pred, mil_version=version,
                gate_rule_id=gate_rule_id, run_timestamp=run_timestamp,
                cv_fold=i % 5,
            )
            fh.write(json.dumps(row, sort_keys=True))
            fh.write("\n")
            n_written += 1
    print(f"[{name}] wrote {n_written} rows to {out_path.relative_to(REPO_ROOT)}")

    # Short summary — mirrors run_alt_gates.py output for A2.
    from collections import Counter
    v_counts = Counter()
    v_correct = Counter()
    v_pred_grade = {}
    v_truth_grade = {}
    for pred, orig in zip(predictions, ref_rows):
        grade, conf, _ = pred
        truth = int(orig["extra"]["ground_truth_isup"])
        if conf >= 0.70:
            v = "SUPPORTED"
        elif conf <= 0.30:
            v = "UNSUPPORTED"
        else:
            v = "CONTESTED"
        v_counts[v] += 1
        if grade == truth:
            v_correct[v] += 1
        v_pred_grade.setdefault(v, Counter())[grade] += 1
        v_truth_grade.setdefault(v, Counter())[truth] += 1
    print(f"[{name}] verdict distribution:")
    for v in ["SUPPORTED", "UNSUPPORTED", "NOT_APPLICABLE", "CONTESTED"]:
        n = v_counts.get(v, 0)
        if n == 0:
            print(f"  {v:16s}  N={n:4d}")
            continue
        c = v_correct.get(v, 0)
        pg = dict(v_pred_grade.get(v, {}))
        tg = dict(v_truth_grade.get(v, {}))
        print(f"  {v:16s}  N={n:4d}  correct={c:4d}  acc={c/n:.4f}  "
              f"pred_grade={pg}  truth_grade={tg}")
    balanced = _balanced_accuracy(predictions, ref_rows)
    print(f"[{name}] balanced accuracy: {balanced:.4f}")
    return version


def _balanced_accuracy(predictions, ref_rows) -> float:
    from collections import Counter
    total = Counter()
    correct = Counter()
    for pred, orig in zip(predictions, ref_rows):
        grade, _, _ = pred
        truth = int(orig["extra"]["ground_truth_isup"])
        total[truth] += 1
        if grade == truth:
            correct[truth] += 1
    if not total:
        return float("nan")
    recalls = [correct[c] / total[c] for c in total if total[c]]
    return sum(recalls) / len(recalls)


def main() -> int:
    if not LEDGER.exists() or not CACHE_DIR.exists():
        print("ERROR: missing inputs.", file=sys.stderr)
        return 2

    ledger_rows = _load_ledger(LEDGER)
    slide_ids, features, labels, ref_rows = _load_slide_data(ledger_rows)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loaded {len(slide_ids)} slides. Device = {device}")

    from p5.mil_classifier.abmil import train_and_predict_cv_abmil
    from p5.mil_classifier.clam_sb import train_and_predict_cv_clam_sb

    abmil_version = _run_and_write(
        "ABMIL", train_and_predict_cv_abmil, OUT_ABMIL,
        slide_ids, features, labels, ref_rows, device,
    )
    clam_version = _run_and_write(
        "CLAM-SB", train_and_predict_cv_clam_sb, OUT_CLAM,
        slide_ids, features, labels, ref_rows, device,
    )

    print(f"\nSUCCESS — ABMIL {abmil_version}\n         CLAM-SB {clam_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
