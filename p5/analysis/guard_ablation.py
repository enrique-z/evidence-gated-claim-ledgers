"""Guard-disabled ablation: measure how many numeric claims the Anti-Fabrication
Guard actually blocks vs how many would slip through if the guard were disabled
(or the ledger were empty).

Runs the guard against:
  (A) papers/p5-asci/main.tex with the FULL ledger snapshot (guard enabled).
  (B) papers/p5-asci/main.tex with an EMPTY ledger snapshot (guard blind).
  (C) An adversarial-fixture manuscript with 3 intentional fabrications, under
      both modes, to show the guard catches them (mode A) and misses them (mode B).

Output: evidence/asci_paper/guard_ablation.json + a summary printed to stdout.

Reviewer motivation: turns the guard from "clever engineering" to "measurably
necessary" — quantitative argument for the guard's value.

Run: python3 p5/analysis/guard_ablation.py
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import List, Tuple

from p5.antifab_guard.guard import check_manuscript, extract_safenum_ranges, SAFENUM_PATTERN, LATEX_NUM_PATTERN

MANUSCRIPT = Path("papers/p5-asci/main.tex")
LEDGERS = [
    "evidence/expansion_500/expansion_500_ledger.jsonl",
    "evidence/expansion_500/abmil_ledger.jsonl",
    "evidence/expansion_500/clam_sb_ledger.jsonl",
    "evidence/pilot_50/pilot_50_ledger.jsonl",
    "evidence/gate_oneslide/gate_oneslide_ledger.jsonl",
]
OUT = Path("evidence/asci_paper/guard_ablation.json")


def load_all_ledgers() -> dict:
    entries = []
    for p in LEDGERS:
        path = Path(p)
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
    return {"entries": entries}


NUMERIC_TOKEN_RE = re.compile(
    r"(?<![A-Za-z_])"       # not preceded by letter/underscore (skip macros)
    r"[+-]?\d+(?:\.\d+)?"   # int or decimal
    r"(?:%)?"               # optional trailing %
)


def count_numeric_tokens_in_manuscript(path: Path) -> Tuple[int, int]:
    """Return (total_numeric_tokens_scanned, tokens_wrapped_in_SafeNum).

    Total is a lower-bound: any bare number that could be a claim. SafeNum count
    is the number of tokens the guard whitelists without ledger check.
    """
    total = 0
    safenum = 0
    with open(path) as f:
        for line in f:
            # Strip LaTeX comments
            code_line = re.sub(r"(?<!\\)%.*$", "", line)
            # SafeNum-wrapped tokens
            for m in SAFENUM_PATTERN.finditer(code_line):
                inner = m.group(1)
                for _ in NUMERIC_TOKEN_RE.finditer(inner):
                    safenum += 1
                    total += 1
            # Un-wrapped tokens (outside SafeNum ranges)
            safe_ranges = extract_safenum_ranges(code_line)
            for m in NUMERIC_TOKEN_RE.finditer(code_line):
                start = m.start()
                inside = any(lo <= start < hi for lo, hi in safe_ranges)
                if not inside:
                    total += 1
    return total, safenum


def write_adversarial_fixture(tmp_path: Path) -> Path:
    """Write a synthetic manuscript with 3 intentional fabrications and 1 SafeNum control."""
    fixture = tmp_path / "adversarial_fixture.tex"
    fixture.write_text(r"""
Fabricated percentage: 87.3%.
Fabricated N: 42 slides.
Fabricated AUROC: 0.9999.
Legitimate aggregate: \SafeNum{500}.
""".lstrip())
    return fixture


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    if not MANUSCRIPT.exists():
        raise SystemExit(f"Manuscript not found: {MANUSCRIPT}")

    full_ledger = load_all_ledgers()
    empty_ledger = {"entries": []}

    # ----- Numeric-token census on the actual manuscript
    total_tokens, safenum_tokens = count_numeric_tokens_in_manuscript(MANUSCRIPT)
    bare_tokens = total_tokens - safenum_tokens

    # ----- Manuscript-level A/B
    print(f"Loaded {len(full_ledger['entries'])} ledger entries from {len(LEDGERS)} files.")
    print(f"Manuscript token census: total_numeric={total_tokens}, "
          f"SafeNum-wrapped={safenum_tokens}, bare={bare_tokens}")
    print()

    print("--- Run A: guard ENABLED (full ledger, current build) ---")
    violations_A = check_manuscript(str(MANUSCRIPT), full_ledger)
    print(f"  violations: {len(violations_A)}")
    for v in violations_A[:5]:
        print(f"    line {v.line_number}: {v.reason} — {v.token}")

    print()
    print("--- Run B: guard BLIND (empty ledger) ---")
    violations_B = check_manuscript(str(MANUSCRIPT), empty_ledger)
    print(f"  violations: {len(violations_B)} "
          f"(unresolved numbers if guard were disabled / ledger absent)")
    if violations_B:
        reason_counts = {}
        for v in violations_B:
            reason_counts[v.reason] = reason_counts.get(v.reason, 0) + 1
        for reason, n in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {reason}: {n}")

    # ----- Adversarial-fixture A/B
    with tempfile.TemporaryDirectory() as td:
        fixture = write_adversarial_fixture(Path(td))
        print()
        print("--- Run C: adversarial fixture, guard ENABLED (full ledger) ---")
        adv_A = check_manuscript(str(fixture), full_ledger)
        print(f"  violations detected: {len(adv_A)} (should be 3: 3 fabrications, 1 SafeNum ignored)")
        for v in adv_A:
            print(f"    line {v.line_number}: {v.reason} — {v.token}")

        print()
        print("--- Run D: adversarial fixture, guard BLIND (empty ledger) ---")
        adv_B = check_manuscript(str(fixture), empty_ledger)
        print(f"  violations detected: {len(adv_B)} "
              f"(same fabrications should also flag under empty ledger)")

    # ----- Persist
    result = {
        "manuscript": str(MANUSCRIPT),
        "ledger_files": LEDGERS,
        "ledger_entries_loaded": len(full_ledger["entries"]),
        "manuscript_census": {
            "total_numeric_tokens": total_tokens,
            "safenum_wrapped": safenum_tokens,
            "bare_tokens_subject_to_guard": bare_tokens,
        },
        "manuscript_guard_enabled": {
            "violations": len(violations_A),
            "reasons": {},
        },
        "manuscript_guard_disabled_equivalent": {
            "violations": len(violations_B),
            "reasons": {},
        },
        "adversarial_fixture_guard_enabled": len(adv_A),
        "adversarial_fixture_guard_blind": len(adv_B),
    }
    for v in violations_A:
        result["manuscript_guard_enabled"]["reasons"][v.reason] = (
            result["manuscript_guard_enabled"]["reasons"].get(v.reason, 0) + 1
        )
    for v in violations_B:
        result["manuscript_guard_disabled_equivalent"]["reasons"][v.reason] = (
            result["manuscript_guard_disabled_equivalent"]["reasons"].get(v.reason, 0) + 1
        )

    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print()
    print(f"WROTE {OUT}")
    print()
    print("=" * 70)
    print("Guard ablation summary (for paper):")
    print("=" * 70)
    print(f"  Numeric tokens in manuscript (excluding SafeNum): {bare_tokens}")
    print(f"  SafeNum-wrapped aggregations (guard-whitelisted): {safenum_tokens}")
    print(f"  Guard-enabled violations on current draft:        {len(violations_A)}")
    print(f"  Guard-blind (empty-ledger) violations:            {len(violations_B)}")
    print(f"  → the guard resolves {bare_tokens - len(violations_A)} numeric tokens "
          f"to ledger rows.")
    print(f"  → without the guard OR without the ledger, {len(violations_B)} bare numeric "
          f"tokens would ship un-checked.")


if __name__ == "__main__":
    main()
