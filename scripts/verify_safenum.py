#!/usr/bin/env python3
r"""Record-scoped \AggNum checker for the ASCI EGCL paper.

Usage:
    python3 scripts/verify_safenum.py papers/p5-asci/main_camera_ready.tex

What it checks:
    1. Every \AggNum{key}{value} in the source:
       (a) The key must exist in the manifest (headline_aggregates.json + camera_ready_numbers.json).
       (b) The printed value must match the manifest value at the printed precision.
    2. Every \SafeNum{value} is counted but NOT cross-checked (legacy, unkeyed).

Exit codes:
    0  = all \AggNum keys known and values match; reports \SafeNum count
    1  = unknown_key or value_mismatch violations found
    2  = manifest file(s) missing

Unlike the old value-level guard which only checks whether the VALUE exists anywhere
in the ledger, \AggNum checks that KEY 'abmil.supported.acc' resolves to the
value documented for ABMIL Supported accuracy --- not for mean-pool Supported accuracy.
This catches swapped-attribution errors (model A's number printed under model B's key).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

HEADLINE_JSON = REPO_ROOT / "evidence" / "asci_paper" / "headline_aggregates.json"
CR_NUMBERS_JSON = REPO_ROOT / "evidence" / "asci_paper" / "camera_ready_numbers.json"

# Tolerance: one unit in the last decimal place of the printed value.
# e.g. "0.811" printed -> tolerance 0.0005; "18.1" printed -> 0.05
def _tolerance(printed: str) -> float:
    """Return half-ULP tolerance for the printed string."""
    # Strip sign, percent signs, etc.
    s = printed.lstrip("-")
    if "." in s:
        decimals = len(s.split(".")[1])
        return 0.5 * 10 ** (-decimals)
    return 0.5


def _load_manifest() -> dict[str, float]:
    """Load and merge both JSON manifests into a keyed float dict."""
    missing = []
    if not HEADLINE_JSON.exists():
        missing.append(str(HEADLINE_JSON))
    if not CR_NUMBERS_JSON.exists():
        missing.append(str(CR_NUMBERS_JSON))
    if missing:
        print(f"ERROR: manifest file(s) missing: {missing}", file=sys.stderr)
        sys.exit(2)

    manifest: dict[str, float] = {}

    # From headline_aggregates.json: safenum_manifest list (value-only, no keys)
    # We expose the heads dict under structured keys.
    with HEADLINE_JSON.open() as f:
        ha = json.load(f)

    heads = ha.get("heads", {})
    for head, data in heads.items():
        pv = data.get("per_verdict", {})
        for verdict, vdata in pv.items():
            vkey = verdict.lower().replace("_", ".")
            if vdata.get("n"):
                manifest[f"{head}.{vkey}.n"] = float(vdata["n"])
                manifest[f"{head}.{vkey}.correct"] = float(vdata["correct"])
                if vdata.get("accuracy") is not None:
                    manifest[f"{head}.{vkey}.acc"] = vdata["accuracy"]
                if vdata.get("accuracy_wilson95_lo") is not None:
                    manifest[f"{head}.{vkey}.wilson.lo"] = vdata["accuracy_wilson95_lo"]
                if vdata.get("accuracy_wilson95_hi") is not None:
                    manifest[f"{head}.{vkey}.wilson.hi"] = vdata["accuracy_wilson95_hi"]
                if vdata.get("mean_confidence") is not None:
                    manifest[f"{head}.{vkey}.mean_conf"] = vdata["mean_confidence"]
        if data.get("raw_accuracy") is not None:
            manifest[f"{head}.raw_accuracy"] = data["raw_accuracy"]
        if data.get("balanced_accuracy") is not None:
            manifest[f"{head}.balanced_accuracy"] = data["balanced_accuracy"]

    # Cohort sizes
    for cname, cdata in ha.get("cohorts", {}).items():
        ckey = cname.replace("-", "_")
        manifest[f"cohort.{ckey}.n"] = float(cdata["n"])

    # From camera_ready_numbers.json: keyed_aggregates
    with CR_NUMBERS_JSON.open() as f:
        crn = json.load(f)

    keyed = crn.get("keyed_aggregates", {})
    for k, v in keyed.items():
        if v is not None:
            manifest[k] = float(v)

    # ABMIL threshold sweep: expose as abmil.sweep.tau{tau}.{field}
    for row in crn.get("abmil_tau_high_sweep", []):
        tau_str = f"{row['tau_high']:.2f}".replace(".", "")
        prefix = f"abmil.sweep.tau{tau_str}"
        manifest[f"{prefix}.released_n"] = float(row["released_n"])
        manifest[f"{prefix}.coverage_pct"] = float(row["coverage_pct"])
        if row["released_accuracy"] is not None:
            manifest[f"{prefix}.acc"] = float(row["released_accuracy"])
        manifest[f"{prefix}.missed_cancer"] = float(row["missed_cancers_released"])

    return manifest


def _parse_tex(tex_path: Path) -> tuple[list[tuple[int, str, str]], int]:
    """Return list of (lineno, key, printed_value) for \\AggNum and count of \\SafeNum."""
    aggnum_pattern = re.compile(r"\\AggNum\{([^}]+)\}\{([^}]+)\}")
    safenum_pattern = re.compile(r"\\SafeNum\{")
    results = []
    safenum_count = 0
    with tex_path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = re.sub(r"(?<!\\)%.*", "", line)  # ignore LaTeX comments
            for m in aggnum_pattern.finditer(line):
                results.append((lineno, m.group(1), m.group(2)))
            safenum_count += len(safenum_pattern.findall(line))
    return results, safenum_count


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(f"Usage: {argv[0]} <main_camera_ready.tex>", file=sys.stderr)
        return 2
    tex_path = Path(argv[1])
    if not tex_path.exists():
        print(f"ERROR: {tex_path} not found", file=sys.stderr)
        return 2

    manifest = _load_manifest()
    aggnum_entries, safenum_count = _parse_tex(tex_path)

    violations: list[str] = []
    ok_count = 0

    for lineno, key, printed in aggnum_entries:
        if key not in manifest:
            violations.append(
                f"  unknown_key  line {lineno}: \\AggNum{{{key}}}{{{printed}}} — key not in manifest"
            )
            continue
        try:
            printed_val = float(printed.replace("%", "").replace(",", ""))
        except ValueError:
            violations.append(
                f"  non_numeric  line {lineno}: \\AggNum{{{key}}}{{{printed}}} — cannot parse value"
            )
            continue
        manifest_val = manifest[key]
        tol = _tolerance(printed.replace("%", ""))
        if abs(printed_val - manifest_val) > tol:
            violations.append(
                f"  value_mismatch  line {lineno}: \\AggNum{{{key}}}{{{printed}}} "
                f"— manifest says {manifest_val} (diff={abs(printed_val - manifest_val):.6f}, tol={tol})"
            )
        else:
            ok_count += 1

    print(f"verify_safenum: checked {len(aggnum_entries)} \\AggNum entries against manifest "
          f"({len(manifest)} keys)")
    print(f"  OK: {ok_count}  violations: {len(violations)}")
    print(f"  \\SafeNum remaining (unchecked): {safenum_count}")

    if violations:
        print("VIOLATIONS:", file=sys.stderr)
        for v in violations:
            print(v, file=sys.stderr)
        return 1

    print("  All \\AggNum values match manifest.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
