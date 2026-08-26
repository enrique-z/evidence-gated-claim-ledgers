"""Phase C2 — Guard evaluation on the 6 adversarial fixtures.

Builds a synthetic ledger snapshot with a handful of SUPPORTED and CONTESTED
claims (so the six fixtures can be checked in isolation from the 550-entry
production ledger). Then runs `check_manuscript` against each fixture and
compares the returned violations to the expected verdicts recorded in
`fixtures/adversarial/expected.json`.

Writes:
  - evidence/antifab_guard_eval/confusion_matrix.json
      A row per fixture with (expected, observed, matched?) and a global
      confusion matrix (rows = expected class, cols = observed class).

Exits 0 iff every fixture produced the expected verdict class. The paper's
§Guard evaluation table pulls its numbers from the JSON.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from collections import Counter
from datetime import datetime, timezone

from p5.antifab_guard import LedgerSnapshot, check_manuscript

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "adversarial"
EXPECTED_JSON = FIXTURE_DIR / "expected.json"
OUT_JSON = REPO_ROOT / "evidence" / "antifab_guard_eval" / "confusion_matrix.json"

CLASSES = ["clean", "no_claim_id", "verdict_not_supported", "sakana_layer_bypass"]


def _build_ledger() -> LedgerSnapshot:
    """Small in-memory ledger — supplies the two rows the fixtures reference."""
    return {"entries": [
        {"claim_id": "supp-balanced-acc-0.365",
         "claim_text": "balanced_accuracy=0.365 cohort=expansion-500",
         "verdict": "SUPPORTED"},
        {"claim_id": "cont-auroc-0.75",
         "claim_text": "AUROC=0.75 cohort=pilot-50",
         "verdict": "CONTESTED"},
    ]}


def _observed_class(violations) -> str:
    if not violations:
        return "clean"
    # Prioritise SAKANA bypass — the plan treats it as the most serious.
    for v in violations:
        if v.reason == "sakana_layer_bypass":
            return "sakana_layer_bypass"
    for v in violations:
        if v.reason == "verdict_not_supported":
            return "verdict_not_supported"
    return "no_claim_id"


def _with_sakana_off(fn):
    """Run fn() with the SAKANA layer 2 env var flipped off; restore on exit."""
    keys = ("SYNTHETIC_DATA_FORBIDDEN", "USE_GLENS_DATA", "USE_GEOMIP_DATA")
    prev = {k: os.environ.get(k) for k in keys}
    try:
        os.environ["SYNTHETIC_DATA_FORBIDDEN"] = "false"
        return fn()
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _with_sakana_on(fn):
    keys = ("SYNTHETIC_DATA_FORBIDDEN", "USE_GLENS_DATA", "USE_GEOMIP_DATA")
    prev = {k: os.environ.get(k) for k in keys}
    try:
        os.environ["SYNTHETIC_DATA_FORBIDDEN"] = "true"
        os.environ["USE_GLENS_DATA"] = "true"
        os.environ["USE_GEOMIP_DATA"] = "true"
        return fn()
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main() -> int:
    expected = json.loads(EXPECTED_JSON.read_text())["fixtures"]
    ledger = _build_ledger()

    fixture_results = []
    matrix: dict[str, Counter] = {c: Counter() for c in CLASSES}

    for fname, exp in sorted(expected.items()):
        fpath = FIXTURE_DIR / fname
        if not fpath.exists():
            print(f"ERROR: missing fixture {fpath}", file=sys.stderr)
            return 2

        expected_class = exp["expected_violation"] or "clean"

        if fname == "f6_sakana_bypass.tex":
            violations = _with_sakana_off(lambda: check_manuscript(str(fpath), ledger))
        else:
            violations = _with_sakana_on(lambda: check_manuscript(str(fpath), ledger))

        observed_class = _observed_class(violations)
        matched = observed_class == expected_class
        fixture_results.append({
            "fixture": fname,
            "expected": expected_class,
            "observed": observed_class,
            "matched": matched,
            "n_violations": len(violations),
            "violations": [
                {"token": v.token, "line": v.line_number,
                 "reason": v.reason, "claim_id": v.claim_id_or_none}
                for v in violations
            ],
            "description": exp.get("description"),
        })
        matrix[expected_class][observed_class] += 1

    all_matched = all(r["matched"] for r in fixture_results)

    payload = {
        "regenerated_at": datetime.now(timezone.utc).isoformat(),
        "n_fixtures": len(fixture_results),
        "n_matched": sum(1 for r in fixture_results if r["matched"]),
        "all_matched": all_matched,
        "classes": CLASSES,
        "confusion_matrix": {
            expected: {obs: matrix[expected].get(obs, 0) for obs in CLASSES}
            for expected in CLASSES
        },
        "fixture_results": fixture_results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"Wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    for r in fixture_results:
        mark = "OK" if r["matched"] else "FAIL"
        print(f"  {mark:4s}  {r['fixture']:34s}  expected={r['expected']:22s}  "
              f"observed={r['observed']}")
    print(f"\n{payload['n_matched']}/{payload['n_fixtures']} matched")
    return 0 if all_matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
