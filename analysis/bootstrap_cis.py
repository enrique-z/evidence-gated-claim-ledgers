"""Bootstrap 95% CIs on every verdict-distribution count in the expansion-500 cohort.

Post-hoc analysis on the checked-in ledgers under evidence/expansion_500/. NO new
ledger entries — this script only produces derived aggregates.

Deterministic (seed=42) 1000-resample non-parametric bootstrap over slide-level
verdict + accuracy assignments per MIL head. Outputs
evidence/expansion_500/analysis/bootstrap_cis.json which the paper's tables can
cite via new \\SafeNum{...} macros (percentile columns).

Reviewer motivation (from the P5 acceptance-strategy notes 2026-08-11): kill the
"9/500 SUPPORTED is noise" objection with actual uncertainty on every headline
verdict count.

Run: python3 p5/analysis/bootstrap_cis.py
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

SEED = 42
N_BOOT = 1000
CI_LOW = 2.5
CI_HIGH = 97.5

LEDGERS = {
    "mean_pool": "evidence/expansion_500/expansion_500_ledger.jsonl",
    "abmil": "evidence/expansion_500/abmil_ledger.jsonl",
    "clam_sb": "evidence/expansion_500/clam_sb_ledger.jsonl",
}
OUT = Path("evidence/expansion_500/analysis/bootstrap_cis.json")


def load_ledger(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def percentile(values: List[float], p: float) -> float:
    """Linear-interp percentile matching numpy's default."""
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] * (c - k) + xs[c] * (k - f)


def bootstrap_verdict_counts(rows: List[dict], rng: random.Random) -> Dict[str, List[int]]:
    """Return {verdict: [count_boot_1, ..., count_boot_N]} across N_BOOT resamples."""
    n = len(rows)
    verdicts = ["SUPPORTED", "UNSUPPORTED", "CONTESTED", "NOT_APPLICABLE"]
    dist: Dict[str, List[int]] = {v: [] for v in verdicts}
    for _ in range(N_BOOT):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        c = Counter(r["verdict"] for r in sample)
        for v in verdicts:
            dist[v].append(c.get(v, 0))
    return dist


def bootstrap_supported_accuracy(rows: List[dict], rng: random.Random) -> List[float]:
    """Return N_BOOT resampled SUPPORTED-verdict accuracies (predicted grade == truth)."""
    n = len(rows)
    accs: List[float] = []
    for _ in range(N_BOOT):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        sup = [r for r in sample if r["verdict"] == "SUPPORTED"]
        if not sup:
            accs.append(float("nan"))
            continue
        correct = sum(
            1 for r in sup
            if r["extra"]["isup_prediction"] == r["extra"]["ground_truth_isup"]
        )
        accs.append(correct / len(sup))
    return accs


def bootstrap_supported_cancer_misclass(rows: List[dict], rng: random.Random) -> List[int]:
    """Bootstrap the count of SUPPORTED verdicts that are actually cancer (truth ISUP 1-5)."""
    n = len(rows)
    counts: List[int] = []
    for _ in range(N_BOOT):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        counts.append(sum(
            1 for r in sample
            if r["verdict"] == "SUPPORTED" and r["extra"]["ground_truth_isup"] >= 1
        ))
    return counts


def summary(values: List[float]) -> Dict[str, float]:
    finite = [v for v in values if v == v]
    if not finite:
        return {"mean": None, "ci_low": None, "ci_high": None, "n_finite": 0}
    mean = sum(finite) / len(finite)
    return {
        "mean": mean,
        "ci_low": percentile(finite, CI_LOW),
        "ci_high": percentile(finite, CI_HIGH),
        "n_finite": len(finite),
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = {"seed": SEED, "n_boot": N_BOOT, "ci_percentiles": [CI_LOW, CI_HIGH]}

    for head_name, ledger_path in LEDGERS.items():
        rng = random.Random(SEED)  # same seed per head → reproducible
        rows = load_ledger(ledger_path)
        n = len(rows)

        verdict_dist = bootstrap_verdict_counts(rows, rng)
        sup_acc = bootstrap_supported_accuracy(rows, rng)
        sup_cancer = bootstrap_supported_cancer_misclass(rows, rng)

        # Observed point estimates
        observed_verdicts = Counter(r["verdict"] for r in rows)
        observed_sup = [r for r in rows if r["verdict"] == "SUPPORTED"]
        observed_sup_acc = (
            sum(1 for r in observed_sup
                if r["extra"]["isup_prediction"] == r["extra"]["ground_truth_isup"])
            / len(observed_sup)
        ) if observed_sup else None
        observed_sup_cancer_misclass = sum(
            1 for r in rows
            if r["verdict"] == "SUPPORTED" and r["extra"]["ground_truth_isup"] >= 1
        )

        out[head_name] = {
            "n_rows": n,
            "observed": {
                "verdict_counts": {v: observed_verdicts.get(v, 0) for v in verdict_dist.keys()},
                "supported_accuracy": observed_sup_acc,
                "supported_cancer_misclass": observed_sup_cancer_misclass,
            },
            "bootstrap": {
                "verdict_counts": {v: summary(vals) for v, vals in verdict_dist.items()},
                "supported_accuracy": summary(sup_acc),
                "supported_cancer_misclass": summary([float(x) for x in sup_cancer]),
            },
        }

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"WROTE {OUT}")

    # Human-readable summary
    print()
    print("=" * 90)
    print(f"{'head':<12} {'verdict':<18} {'observed':>10} {'boot mean':>12} {'95% CI':>16}")
    print("=" * 90)
    for head_name in LEDGERS:
        for v in ["SUPPORTED", "UNSUPPORTED", "CONTESTED", "NOT_APPLICABLE"]:
            obs = out[head_name]["observed"]["verdict_counts"][v]
            b = out[head_name]["bootstrap"]["verdict_counts"][v]
            if b["ci_low"] is None:
                ci = "n/a"
            else:
                ci = f"[{b['ci_low']:.0f}, {b['ci_high']:.0f}]"
            print(f"{head_name:<12} {v:<18} {obs:>10} {b['mean']:>12.2f} {ci:>16}")
    print()
    print(f"{'head':<12} {'metric':<32} {'observed':>10} {'boot mean':>12} {'95% CI':>18}")
    print("=" * 90)
    for head_name in LEDGERS:
        b = out[head_name]["bootstrap"]["supported_accuracy"]
        obs = out[head_name]["observed"]["supported_accuracy"]
        obs_s = f"{obs:.3f}" if obs is not None else "n/a"
        ci = (f"[{b['ci_low']:.3f}, {b['ci_high']:.3f}]"
              if b["ci_low"] is not None else "n/a")
        print(f"{head_name:<12} {'SUPPORTED accuracy':<32} {obs_s:>10} {b['mean']:>12.3f} {ci:>18}")

        b2 = out[head_name]["bootstrap"]["supported_cancer_misclass"]
        obs2 = out[head_name]["observed"]["supported_cancer_misclass"]
        ci2 = (f"[{b2['ci_low']:.1f}, {b2['ci_high']:.1f}]"
               if b2["ci_low"] is not None else "n/a")
        print(f"{head_name:<12} {'SUPPORTED cancer misclass':<32} {obs2:>10} {b2['mean']:>12.2f} {ci2:>18}")


if __name__ == "__main__":
    main()
