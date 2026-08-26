"""Round-5 pre-declared experiment: larger-split (cross-fold) LAC conformal and
Mondrian (class-conditional) LAC variants, on the existing per-slide out-of-fold
probabilities. See papers/p5-asci/CHANGELOG_p5_round5.md for the pre-declared
analysis plan (arms, alpha grid, metrics, reporting commitment) — written BEFORE
this script was first run.

Reads (ledgers only, no feature access, no model runs):
  - evidence/expansion_500/expansion_500_ledger.jsonl                 (Phikon-v2 mean-pool)
  - evidence/expansion_500_phikonv1/expansion_500_phikonv1_ledger.jsonl
  - evidence/expansion_500_lunit/expansion_500_lunit_ledger.jsonl
  - evidence/expansion_500_dinov2l/expansion_500_dinov2l_ledger.jsonl

Writes:
  - evidence/expansion_500/analysis/conformal_variants_summary.json
  - evidence/expansion_500/analysis/conformal_variants_ledger.jsonl
      (operating alpha=0.10 only, both arms, all 4 backbones -> 4000 rows)

Protocol (cross-fold split conformal): for each fold f in {0..4}, calibrate on
the 400 rows with cv_fold != f, test on the 100 rows with cv_fold == f; each
slide is tested exactly once; metrics pooled over folds. LAC score
s(x,y) = 1 - p_y; finite-sample quantile k = ceil((n+1)(1-alpha)) clamped to
[1, n] (identical to p5.evidence_gate.alt_gates.calibrate_lac). The Mondrian
arm computes the quantile per TRUE class from the calibration rows of that
class. Gate semantics identical to ConformalGate: singleton -> SUPPORTED,
empty -> UNSUPPORTED, |set| >= 2 -> CONTESTED.

Sanity checks (exit nonzero on failure):
  - each ledger has exactly 500 rows, folds exactly {0..4} x 100
  - each arm x alpha produces exactly 500 verdicts summing to 500
  - Mondrian calibration bins are non-empty for every class at every fold
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

LEDGERS = {
    "phikon-v2-meanpool": REPO_ROOT / "evidence/expansion_500/expansion_500_ledger.jsonl",
    "phikon-v1": REPO_ROOT / "evidence/expansion_500_phikonv1/expansion_500_phikonv1_ledger.jsonl",
    "lunit-dino": REPO_ROOT / "evidence/expansion_500_lunit/expansion_500_lunit_ledger.jsonl",
    "dinov2-large": REPO_ROOT / "evidence/expansion_500_dinov2l/expansion_500_dinov2l_ledger.jsonl",
}

ALPHAS = (0.05, 0.10, 0.20)
OPERATING_ALPHA = 0.10
N_CLASSES = 6

OUT_DIR = REPO_ROOT / "evidence/expansion_500/analysis"
OUT_SUMMARY = OUT_DIR / "conformal_variants_summary.json"
OUT_LEDGER = OUT_DIR / "conformal_variants_ledger.jsonl"


def _load(path: pathlib.Path) -> list[dict]:
    rows = [json.loads(l) for l in path.open() if l.strip()]
    if len(rows) != 500:
        sys.exit(f"ERROR: {path} has {len(rows)} rows, expected 500")
    folds = Counter(r["justification"].get("cv_fold") for r in rows)
    if dict(folds) != {f: 100 for f in range(5)}:
        sys.exit(f"ERROR: {path} fold structure {dict(folds)} != 5x100")
    return rows


def _lac_quantile(scores: list[float], alpha: float) -> float:
    """ceil((n+1)(1-alpha))-th order statistic, clamped to [1, n] (Vovk)."""
    s = sorted(scores)
    n = len(s)
    if n == 0:
        raise ValueError("empty calibration bin")
    k = math.ceil((n + 1) * (1.0 - alpha))
    k = max(1, min(n, k))
    return s[k - 1]


def _score(row: dict, y: int) -> float:
    return 1.0 - float(row["extra"]["per_grade_probs"][y])


def _true_class(row: dict) -> int:
    return int(row["extra"]["ground_truth_isup"])


def _pred_set_pooled(row: dict, q: float) -> list[int]:
    probs = row["extra"]["per_grade_probs"]
    return [y for y in range(N_CLASSES) if (1.0 - float(probs[y])) <= q]


def _pred_set_mondrian(row: dict, q_by_class: dict[int, float]) -> list[int]:
    probs = row["extra"]["per_grade_probs"]
    return [y for y in range(N_CLASSES)
            if (1.0 - float(probs[y])) <= q_by_class[y]]


def run_arm(rows: list[dict], arm: str, alpha: float) -> dict:
    """Cross-fold conformal for one (arm, alpha). Returns metrics + per-row sets."""
    per_row: list[dict] = []
    quantiles_by_fold: dict[int, object] = {}
    for f in range(5):
        cal = [r for r in rows if r["justification"]["cv_fold"] != f]
        test = [r for r in rows if r["justification"]["cv_fold"] == f]
        if arm == "pooled":
            q = _lac_quantile([_score(r, _true_class(r)) for r in cal], alpha)
            quantiles_by_fold[f] = q
            for r in test:
                per_row.append({"row": r, "pset": _pred_set_pooled(r, q)})
        elif arm == "mondrian":
            by_class: dict[int, list[float]] = defaultdict(list)
            for r in cal:
                by_class[_true_class(r)].append(_score(r, _true_class(r)))
            missing = [c for c in range(N_CLASSES) if not by_class.get(c)]
            if missing:
                sys.exit(f"ERROR: empty Mondrian bin(s) {missing} at fold {f}")
            q_by_class = {c: _lac_quantile(by_class[c], alpha)
                          for c in range(N_CLASSES)}
            quantiles_by_fold[f] = {str(c): q_by_class[c] for c in range(N_CLASSES)}
            quantiles_by_fold[f]["n_per_class"] = {
                str(c): len(by_class[c]) for c in range(N_CLASSES)}
            for r in test:
                per_row.append({"row": r, "pset": _pred_set_mondrian(r, q_by_class)})
        else:
            raise ValueError(arm)

    if len(per_row) != 500:
        sys.exit(f"ERROR: {arm} alpha={alpha} produced {len(per_row)} test rows")

    verdicts = Counter()
    correct_singleton = 0
    covered = 0
    set_sizes = 0
    for pr in per_row:
        pset = pr["pset"]
        truth = _true_class(pr["row"])
        set_sizes += len(pset)
        if truth in pset:
            covered += 1
        if len(pset) == 1:
            v = "SUPPORTED"
            if pset[0] == truth:
                correct_singleton += 1
        elif len(pset) == 0:
            v = "UNSUPPORTED"
        else:
            v = "CONTESTED"
        verdicts[v] += 1
        pr["verdict"] = v

    n_s = verdicts["SUPPORTED"]
    if sum(verdicts.values()) != 500:
        sys.exit(f"ERROR: verdict counts sum {sum(verdicts.values())} != 500")
    return {
        "arm": arm,
        "alpha": alpha,
        "verdict_counts": {v: verdicts.get(v, 0)
                           for v in ["SUPPORTED", "UNSUPPORTED", "CONTESTED"]},
        "singleton_accuracy": (correct_singleton / n_s) if n_s else None,
        "singleton_correct": correct_singleton,
        "empirical_coverage": covered / 500.0,
        "mean_set_size": set_sizes / 500.0,
        "quantiles_by_fold": quantiles_by_fold,
        "_per_row": per_row,
    }


def _ledger_row(orig: dict, arm: str, alpha: float, pset: list[int],
                verdict: str, ts: str) -> dict:
    rule_id = (f"{'cross' if arm == 'pooled' else 'mondrian'}-conformal-lac"
               f"-alpha{alpha:.2f}-v1@2026-08-24")
    return {
        "claim_id": f"{orig['claim_id']}#{rule_id}",
        "claim_text": orig["claim_text"],
        "evidence_ref": orig["evidence_ref"],
        "verdict": verdict,
        "justification": {
            "gate_rule_id": rule_id,
            "source_rows": [orig["evidence_ref"]],
            "thresholds": {"alpha": alpha,
                           "prediction_set": pset,
                           "prediction_set_size": len(pset)},
            "mil_version": orig["extra"]["mil_version"],
            "cv_fold": orig["justification"].get("cv_fold"),
            "patch_count": orig["justification"].get("patch_count"),
            "is_tracer": orig["justification"].get("is_tracer", False),
        },
        "timestamp": ts,
        "backbone_version": orig["backbone_version"],
        "cohort_id": orig["cohort_id"],
        "gate_rule_id": rule_id,
        "extra": {
            "isup_prediction": orig["extra"]["isup_prediction"],
            "confidence": orig["extra"]["confidence"],
            "per_grade_probs": orig["extra"]["per_grade_probs"],
            "mil_version": orig["extra"]["mil_version"],
            "ground_truth_isup": orig["extra"]["ground_truth_isup"],
            "ground_truth_gleason": orig["extra"].get("ground_truth_gleason"),
        },
    }


def main() -> int:
    ts = datetime.now(timezone.utc).isoformat()
    summary: dict = {
        "pre_declared_plan": "papers/p5-asci/CHANGELOG_p5_round5.md (E-R5-1, E-R5-2)",
        "protocol": "cross-fold split conformal (calibrate on 4 folds' out-of-fold "
                    "scores n=400, test on the held-out fold n=100, pooled over folds)",
        "alphas": list(ALPHAS),
        "operating_alpha": OPERATING_ALPHA,
        "regenerated_at": ts,
        "backbones": {},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_ledger = 0
    with OUT_LEDGER.open("w") as fh:
        for backbone, path in LEDGERS.items():
            rows = _load(path)
            summary["backbones"][backbone] = {
                "ledger": str(path.relative_to(REPO_ROOT)), "arms": {}}
            for arm in ("pooled", "mondrian"):
                for alpha in ALPHAS:
                    res = run_arm(rows, arm, alpha)
                    per_row = res.pop("_per_row")
                    summary["backbones"][backbone]["arms"][
                        f"{arm}@alpha={alpha:.2f}"] = res
                    if alpha == OPERATING_ALPHA:
                        for pr in per_row:
                            fh.write(json.dumps(_ledger_row(
                                pr["row"], arm, alpha, pr["pset"],
                                pr["verdict"], ts), sort_keys=True) + "\n")
                            n_ledger += 1

    expected = len(LEDGERS) * 2 * 500
    if n_ledger != expected:
        sys.exit(f"ERROR: wrote {n_ledger} ledger rows, expected {expected}")

    with OUT_SUMMARY.open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"Wrote {OUT_LEDGER.relative_to(REPO_ROOT)} ({n_ledger} rows)")
    print(f"Wrote {OUT_SUMMARY.relative_to(REPO_ROOT)}")
    for backbone, b in summary["backbones"].items():
        print(f"\n== {backbone} ==")
        for key, res in b["arms"].items():
            vc = res["verdict_counts"]
            acc = res["singleton_accuracy"]
            print(f"  {key:24s} S/U/C = {vc['SUPPORTED']:3d}/{vc['UNSUPPORTED']:3d}/"
                  f"{vc['CONTESTED']:3d}  singleton_acc="
                  f"{'--' if acc is None else f'{acc:.3f}'}  "
                  f"coverage={res['empirical_coverage']:.3f}  "
                  f"mean|set|={res['mean_set_size']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
