"""Per-institution verdict stability: Karolinska vs Radboudumc.

Post-hoc analysis on the enriched expansion-500 ledger (after adding
data_provider to the extra field). Groups each ledger row by data_provider
and reports per-institution verdict distributions for all three MIL heads.

Requires: expansion_500.py re-run on Kaggle with data_provider in extra field
(TASK 2 enrichment — see p5/orchestrator/notebooks/expansion_500.py line ~487).

Output:
  evidence/expansion_500/analysis/per_institution.json

Usage:
  python3 p5/analysis/per_institution.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

LEDGERS = {
    "mean_pool": Path("evidence/expansion_500/expansion_500_ledger.jsonl"),
    "abmil": Path("evidence/expansion_500/abmil_ledger.jsonl"),
    "clam_sb": Path("evidence/expansion_500/clam_sb_ledger.jsonl"),
}
OUT = Path("evidence/expansion_500/analysis/per_institution.json")

VERDICTS = ["SUPPORTED", "UNSUPPORTED", "CONTESTED", "NOT_APPLICABLE"]
PROVIDERS = ["karolinska", "radboud"]


def load_ledger(path: Path) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def has_data_provider(rows: List[dict]) -> bool:
    return any("data_provider" in r.get("extra", {}) for r in rows)


def stratify_by_provider(rows: List[dict]) -> Dict[str, Dict]:
    by_provider: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        provider = r.get("extra", {}).get("data_provider", "unknown").lower()
        by_provider[provider].append(r)

    out = {}
    for provider in sorted(by_provider.keys()):
        rs = by_provider[provider]
        counts = Counter(r["verdict"] for r in rs)
        n = len(rs)

        supported = [r for r in rs if r["verdict"] == "SUPPORTED"]
        supported_cancer = [
            r for r in supported
            if r["extra"].get("ground_truth_isup", 0) >= 1
        ]

        out[provider] = {
            "n_total": n,
            "counts": {v: counts.get(v, 0) for v in VERDICTS},
            "fractions": {v: counts.get(v, 0) / n for v in VERDICTS},
            "SUPPORTED_cancer_count": len(supported_cancer),
            "SUPPORTED_total": len(supported),
        }
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result = {}
    missing_field = []

    for head_name, ledger_path in LEDGERS.items():
        if not ledger_path.exists():
            print(f"WARN: {ledger_path} not found — skipping {head_name}")
            continue
        rows = load_ledger(ledger_path)
        if not has_data_provider(rows):
            missing_field.append(head_name)
            print(
                f"WARN: {head_name} ledger has no data_provider in extra field.\n"
                f"  Re-run expansion_500.py on Kaggle with the data_provider fix,\n"
                f"  then re-download the ledger."
            )
            result[head_name] = {"error": "data_provider not in extra field"}
            continue

        result[head_name] = {
            "n_rows": len(rows),
            "by_provider": stratify_by_provider(rows),
        }

    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"WROTE {OUT}")

    if missing_field:
        print(
            f"\nNOTE: {missing_field} had no data_provider. "
            f"The current ledgers predate the TASK 2 enrichment. "
            f"Re-run expansion_500.py on Kaggle and replace the ledger files."
        )
        return

    # Human-readable summary
    print()
    for head_name, res in result.items():
        if "error" in res:
            continue
        print(f"\n=== {head_name} (N={res['n_rows']}) ===")
        print(f"{'Provider':<14} {'N':>5} {'SUPP':>5} {'CONT':>5} "
              f"{'UNSUPP':>7} {'SUPP_cancer':>12}")
        for provider, d in sorted(res["by_provider"].items()):
            print(
                f"{provider:<14} {d['n_total']:>5} "
                f"{d['counts']['SUPPORTED']:>5} "
                f"{d['counts']['CONTESTED']:>5} "
                f"{d['counts']['UNSUPPORTED']:>7} "
                f"{d['SUPPORTED_cancer_count']:>12}"
            )


if __name__ == "__main__":
    main()
