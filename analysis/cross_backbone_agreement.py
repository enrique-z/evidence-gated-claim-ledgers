"""Cross-backbone agreement: Phikon-v2 vs H-optimus-0 on expansion-500.

Joins the two mean-pool ledgers on slide_id and computes:
  - Overall verdict agreement rate (same verdict both backbones)
  - Agreement broken down by verdict pair (both SUPPORTED, both CONTESTED, etc.)
  - Disagreement rate and disagreement types
  - Agreement stratified by ground-truth ISUP grade

NOTE (2026-08-25): the "hoptimus0" path below is a STALE NAME. H-optimus-0 was
HF-access-gated (401) and never ran; the notebook was repurposed to DINOv2-L and the
output directory kept the old name. The ledger it loads is the DINOv2-L run
(backbone_version=dinov2-large-meta@2026-08-11). Path left unchanged because it is
hardcoded here and in the notebook; see that folder's README.md. Not renamed, not
fabricated, not a fourth backbone.

Run AFTER expansion_500_hoptimus0_ledger.jsonl is downloaded from Kaggle to:
  evidence/expansion_500_hoptimus0/expansion_500_hoptimus0_ledger.jsonl

Usage:
  python3 p5/analysis/cross_backbone_agreement.py

Output:
  evidence/expansion_500_hoptimus0/cross_backbone_agreement.json
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

PHIKON_LEDGER = Path("evidence/expansion_500/expansion_500_ledger.jsonl")
SECOND_LEDGER = Path("evidence/expansion_500_hoptimus0/expansion_500_hoptimus0_ledger.jsonl")
OUT = Path("evidence/expansion_500_hoptimus0/cross_backbone_agreement.json")

VERDICTS = ["SUPPORTED", "UNSUPPORTED", "CONTESTED", "NOT_APPLICABLE"]


def load_ledger(path: Path) -> Dict[str, dict]:
    """Return {slide_id: row} for easy joining."""
    by_slide = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            # slide_id is in claim_text: "slide=<id> isup_prediction=..."
            sid = row["extra"].get("isup_prediction")  # fallback
            # Robust extraction: parse claim_text
            for part in row["claim_text"].split():
                if part.startswith("slide="):
                    sid = part[len("slide="):]
                    break
            if sid is None:
                sid = row.get("claim_id", "unknown").split("#")[0]
            by_slide[sid] = row
    return by_slide


def verdict_pair_key(v1: str, v2: str) -> str:
    return f"{v1}|{v2}"


def main():
    if not PHIKON_LEDGER.exists():
        raise SystemExit(f"Phikon-v2 ledger not found: {PHIKON_LEDGER}")
    if not SECOND_LEDGER.exists():
        raise SystemExit(
            f"H-optimus-0 ledger not found: {SECOND_LEDGER}\n"
            f"Run expansion_500_hoptimus0.py on Kaggle first, then download output."
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)

    phikon = load_ledger(PHIKON_LEDGER)
    hibou = load_ledger(SECOND_LEDGER)

    common = sorted(set(phikon.keys()) & set(hibou.keys()))
    phikon_only = sorted(set(phikon.keys()) - set(hibou.keys()))
    hibou_only = sorted(set(hibou.keys()) - set(phikon.keys()))

    print(f"Phikon-v2 ledger:    {len(phikon)} slides")
    print(f"H-optimus-0 ledger:  {len(hibou)} slides")
    print(f"Common slides:       {len(common)}")
    print(f"Phikon-only:         {len(phikon_only)}")
    print(f"H-optimus-0-only:    {len(hibou_only)}")

    # ------------------------------------------------------------------
    # Per-slide comparison
    # ------------------------------------------------------------------
    pair_counts: Counter = Counter()
    agree = 0
    disagree = 0
    by_isup: Dict[int, List[Tuple[str, str]]] = defaultdict(list)

    for sid in common:
        vp = phikon[sid]["verdict"]
        vh = hibou[sid]["verdict"]
        isup = phikon[sid]["extra"].get("ground_truth_isup", -1)
        pair_counts[verdict_pair_key(vp, vh)] += 1
        by_isup[isup].append((vp, vh))
        if vp == vh:
            agree += 1
        else:
            disagree += 1

    agreement_rate = agree / len(common) if common else 0.0

    # ------------------------------------------------------------------
    # Agreement by ISUP grade
    # ------------------------------------------------------------------
    by_isup_summary = {}
    for grade in sorted(by_isup.keys()):
        pairs = by_isup[grade]
        grade_agree = sum(1 for vp, vh in pairs if vp == vh)
        by_isup_summary[grade] = {
            "n": len(pairs),
            "agree": grade_agree,
            "disagree": len(pairs) - grade_agree,
            "agreement_rate": grade_agree / len(pairs) if pairs else 0.0,
            "pair_counts": dict(Counter(verdict_pair_key(vp, vh) for vp, vh in pairs)),
        }

    # ------------------------------------------------------------------
    # Disagreement breakdown by verdict pair
    # ------------------------------------------------------------------
    disagree_breakdown = {
        k: v for k, v in pair_counts.items()
        if k.split("|")[0] != k.split("|")[1]
    }
    agree_breakdown = {
        k: v for k, v in pair_counts.items()
        if k.split("|")[0] == k.split("|")[1]
    }

    # ------------------------------------------------------------------
    # SUPPORTED-specific agreement (the clinically critical verdict)
    # ------------------------------------------------------------------
    both_supported = pair_counts.get("SUPPORTED|SUPPORTED", 0)
    phikon_supported_total = sum(
        1 for sid in common if phikon[sid]["verdict"] == "SUPPORTED"
    )
    hibou_supported_total = sum(
        1 for sid in common if hibou[sid]["verdict"] == "SUPPORTED"
    )

    result = {
        "phikon_ledger": str(PHIKON_LEDGER),
        "second_ledger": str(SECOND_LEDGER),
        "second_backbone": "dinov2-large",
        "n_phikon": len(phikon),
        "n_second": len(hibou),
        "n_common": len(common),
        "n_phikon_only": len(phikon_only),
        "n_second_only": len(hibou_only),
        "overall_agreement": {
            "agree": agree,
            "disagree": disagree,
            "agreement_rate": agreement_rate,
        },
        "agree_breakdown": agree_breakdown,
        "disagree_breakdown": disagree_breakdown,
        "pair_counts": dict(pair_counts),
        "supported_overlap": {
            "both_supported": both_supported,
            "phikon_supported_total": phikon_supported_total,
            "second_supported_total": hibou_supported_total,
        },
        "by_isup_grade": by_isup_summary,
    }

    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWROTE {OUT}")

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Cross-backbone agreement summary")
    print("=" * 70)
    print(f"  Common slides:         {len(common)}")
    print(f"  Overall agreement:     {agree}/{len(common)} "
          f"({agreement_rate:.1%})")
    print(f"  Overall disagreement:  {disagree}/{len(common)} "
          f"({1 - agreement_rate:.1%})")
    print()
    print("  Agreement by verdict:")
    for k, v in sorted(agree_breakdown.items()):
        vv = k.split("|")[0]
        print(f"    {vv}: {v}")
    print()
    print("  Disagreement pairs (phikon|h-optimus-0: count):")
    for k, v in sorted(disagree_breakdown.items(), key=lambda kv: -kv[1]):
        print(f"    {k}: {v}")
    print()
    print("  SUPPORTED overlap:")
    print(f"    Phikon-v2   SUPPORTED: {phikon_supported_total}")
    print(f"    H-optimus-0 SUPPORTED: {hibou_supported_total}")
    print(f"    Both        SUPPORTED: {both_supported}")
    print()
    print("  Agreement by ISUP grade:")
    print(f"  {'ISUP':>4} {'N':>5} {'Agree':>7} {'Rate':>8}")
    for grade, d in sorted(by_isup_summary.items()):
        print(f"  {grade:>4} {d['n']:>5} {d['agree']:>7} "
              f"{d['agreement_rate']:>8.1%}")


if __name__ == "__main__":
    main()
