"""Gleason-grade-stratified verdict distributions on expansion-500.

Post-hoc analysis. Groups every slide in every MIL-head ledger by
ground_truth_isup (0-5) and reports per-grade verdict counts + accuracies.
Answers the reviewer question: "does the reporting standard degrade gracefully
across rarity, or does it break on high-grade cases?"

Output: evidence/expansion_500/analysis/gleason_stratified.json

Run: python3 p5/analysis/gleason_stratified.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

LEDGERS = {
    "mean_pool": "evidence/expansion_500/expansion_500_ledger.jsonl",
    "abmil": "evidence/expansion_500/abmil_ledger.jsonl",
    "clam_sb": "evidence/expansion_500/clam_sb_ledger.jsonl",
}
OUT = Path("evidence/expansion_500/analysis/gleason_stratified.json")

VERDICTS = ["SUPPORTED", "UNSUPPORTED", "CONTESTED", "NOT_APPLICABLE"]


def load_ledger(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def stratify(rows: List[dict]) -> Dict[int, Dict]:
    """Return {isup_grade: {verdict: count, ...}} plus accuracies."""
    by_grade: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        by_grade[r["extra"]["ground_truth_isup"]].append(r)

    out = {}
    for grade in sorted(by_grade.keys()):
        rs = by_grade[grade]
        counts = Counter(r["verdict"] for r in rs)
        supported = [r for r in rs if r["verdict"] == "SUPPORTED"]
        contested = [r for r in rs if r["verdict"] == "CONTESTED"]
        unsupported = [r for r in rs if r["verdict"] == "UNSUPPORTED"]

        def acc(sub: List[dict]) -> float | None:
            if not sub:
                return None
            correct = sum(
                1 for r in sub
                if r["extra"]["isup_prediction"] == r["extra"]["ground_truth_isup"]
            )
            return correct / len(sub)

        # For each verdict subset, count how many are actually cancer (grade >= 1)
        # under the "released" verdicts (SUPPORTED). Grade-0 subset trivially = 0.
        cancer_misclass_supported = sum(
            1 for r in supported
            if r["extra"]["ground_truth_isup"] >= 1
        )

        out[grade] = {
            "n_total": len(rs),
            "counts": {v: counts.get(v, 0) for v in VERDICTS},
            "fractions": {v: counts.get(v, 0) / len(rs) for v in VERDICTS},
            "SUPPORTED_accuracy": acc(supported),
            "CONTESTED_accuracy": acc(contested),
            "UNSUPPORTED_accuracy": acc(unsupported),
            "SUPPORTED_cancer_misclass": cancer_misclass_supported,
        }
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result = {}
    for head_name, path in LEDGERS.items():
        rows = load_ledger(path)
        result[head_name] = {
            "n_rows": len(rows),
            "by_isup_grade": stratify(rows),
        }

    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"WROTE {OUT}")

    # Human-readable summary
    print()
    for head_name, res in result.items():
        print(f"\n=== {head_name} (N={res['n_rows']}) ===")
        print(f"{'ISUP':<5} {'N':>5} {'SUPP':>5} {'CONT':>5} {'UNSUPP':>7} "
              f"{'SUPP_acc':>10} {'CONT_acc':>10}")
        for grade, d in sorted(res["by_isup_grade"].items()):
            sacc = f"{d['SUPPORTED_accuracy']:.3f}" if d["SUPPORTED_accuracy"] is not None else "n/a"
            cacc = f"{d['CONTESTED_accuracy']:.3f}" if d["CONTESTED_accuracy"] is not None else "n/a"
            print(f"{grade:<5} {d['n_total']:>5} "
                  f"{d['counts']['SUPPORTED']:>5} "
                  f"{d['counts']['CONTESTED']:>5} "
                  f"{d['counts']['UNSUPPORTED']:>7} "
                  f"{sacc:>10} {cacc:>10}")


if __name__ == "__main__":
    main()
