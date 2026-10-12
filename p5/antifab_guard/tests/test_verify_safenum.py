"""Tests for scripts/verify_safenum.py record-scoped \AggNum checker.

Key test: swapped-attribution fixture.
  - mean-pool Supported accuracy is 0.778.
  - ABMIL Supported accuracy is 0.811.
  - Printing 0.778 as \AggNum{abmil.supported.acc}{0.778} should FAIL (wrong key).
  - The old value-level guard would PASS because 0.778 exists somewhere in the ledger.
  - The new key-scoped check REJECTS it.

A clean case (correct key/value) must PASS.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

# Locate repo root (3 levels up from this file: tests/ -> antifab_guard/ -> p5/ -> repo)
REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify_safenum.py"

# Pull in the verify_safenum module by path so we don't need it on sys.path
import importlib.util
spec = importlib.util.spec_from_file_location("verify_safenum", VERIFY_SCRIPT)
verify_safenum = importlib.util.util = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify_safenum)


@pytest.fixture()
def minimal_manifests(tmp_path: Path) -> tuple[Path, Path]:
    """Create minimal headline and camera-ready manifests for testing."""
    # headline_aggregates.json: defines abmil.supported.acc = 0.811, mean-pool.supported.acc = 0.778
    ha = {
        "heads": {
            "mean-pool": {
                "raw_accuracy": 0.428,
                "balanced_accuracy": 0.365,
                "per_verdict": {
                    "SUPPORTED": {
                        "n": 9,
                        "correct": 7,
                        "accuracy": 0.7777777777,
                        "accuracy_wilson95_lo": 0.453,
                        "accuracy_wilson95_hi": 0.937,
                        "mean_confidence": 0.747,
                    },
                    "CONTESTED": {
                        "n": 365,
                        "correct": 173,
                        "accuracy": 0.4739726,
                        "accuracy_wilson95_lo": 0.423,
                        "accuracy_wilson95_hi": 0.525,
                        "mean_confidence": 0.434,
                    },
                    "UNSUPPORTED": {
                        "n": 126,
                        "correct": 34,
                        "accuracy": 0.2698,
                        "accuracy_wilson95_lo": 0.200,
                        "accuracy_wilson95_hi": 0.353,
                        "mean_confidence": 0.259,
                    },
                    "NOT_APPLICABLE": {"n": 0, "correct": 0, "accuracy": None,
                                       "accuracy_wilson95_lo": None, "accuracy_wilson95_hi": None,
                                       "mean_confidence": None},
                },
            },
            "abmil": {
                "raw_accuracy": 0.504,
                "balanced_accuracy": 0.439,
                "per_verdict": {
                    "SUPPORTED": {
                        "n": 90,
                        "correct": 73,
                        "accuracy": 0.8111111111,
                        "accuracy_wilson95_lo": 0.718,
                        "accuracy_wilson95_hi": 0.879,
                        "mean_confidence": 0.809,
                    },
                    "CONTESTED": {
                        "n": 361,
                        "correct": 167,
                        "accuracy": 0.4626,
                        "accuracy_wilson95_lo": 0.412,
                        "accuracy_wilson95_hi": 0.514,
                        "mean_confidence": 0.447,
                    },
                    "UNSUPPORTED": {
                        "n": 49,
                        "correct": 12,
                        "accuracy": 0.2449,
                        "accuracy_wilson95_lo": 0.146,
                        "accuracy_wilson95_hi": 0.381,
                        "mean_confidence": 0.276,
                    },
                    "NOT_APPLICABLE": {"n": 0, "correct": 0, "accuracy": None,
                                       "accuracy_wilson95_lo": None, "accuracy_wilson95_hi": None,
                                       "mean_confidence": None},
                },
            },
        },
        "cohorts": {
            "expansion-500": {"n": 500},
            "pilot-50": {"n": 50},
        },
    }
    crn = {
        "keyed_aggregates": {
            "abmil.supported.n": 90,
            "abmil.supported.acc": 0.811,
            "abmil.supported.grade0.n": 83,
            "abmil.supported.missed_cancer": 15,
        },
        "abmil_tau_high_sweep": [],
    }
    ha_path = tmp_path / "headline_aggregates.json"
    crn_path = tmp_path / "camera_ready_numbers.json"
    ha_path.write_text(json.dumps(ha, indent=2))
    crn_path.write_text(json.dumps(crn, indent=2))
    return ha_path, crn_path


def _run_check(tex_content: str, ha_path: Path, crn_path: Path, tmp_path: Path) -> int:
    """Run verify_safenum.main on a tex snippet with patched manifest paths."""
    tex_file = tmp_path / "test.tex"
    tex_file.write_text(textwrap.dedent(tex_content))
    # Patch module globals to use our tmp manifests
    original_ha = verify_safenum.HEADLINE_JSON
    original_crn = verify_safenum.CR_NUMBERS_JSON
    verify_safenum.HEADLINE_JSON = ha_path
    verify_safenum.CR_NUMBERS_JSON = crn_path
    try:
        return verify_safenum.main(["verify_safenum", str(tex_file)])
    finally:
        verify_safenum.HEADLINE_JSON = original_ha
        verify_safenum.CR_NUMBERS_JSON = original_crn


class TestSwappedAttribution:
    """The core regression: mean-pool accuracy quoted under ABMIL key."""

    def test_swapped_attribution_rejected(self, minimal_manifests, tmp_path):
        """0.778 is mean-pool Supported accuracy; printing it as abmil.supported.acc must FAIL."""
        ha_path, crn_path = minimal_manifests
        tex = r"Under ABMIL \AggNum{abmil.supported.acc}{0.778}."
        result = _run_check(tex, ha_path, crn_path, tmp_path)
        assert result != 0, (
            "Expected non-zero exit for swapped attribution "
            "(mean-pool acc=0.778 printed under abmil.supported.acc key)"
        )

    def test_old_value_level_guard_would_pass(self, minimal_manifests, tmp_path):
        """Demonstrate that 0.778 exists in the manifest under the correct key, so a
        value-only check (ignoring keys) would pass it.  Our keyed check rejects it."""
        ha_path, crn_path = minimal_manifests
        # Load manifest to verify 0.778 is present under mean-pool key
        manifest = {}
        verify_safenum.HEADLINE_JSON = ha_path
        verify_safenum.CR_NUMBERS_JSON = crn_path
        try:
            manifest = verify_safenum._load_manifest()
        finally:
            pass
        # 0.778 appears in manifest under mean-pool.supported.acc
        meanpool_acc = manifest.get("mean-pool.supported.acc")
        assert meanpool_acc is not None
        assert abs(meanpool_acc - 0.778) < 0.001, (
            f"mean-pool supported acc should be ~0.778, got {meanpool_acc}"
        )
        # But abmil.supported.acc is 0.811, not 0.778
        abmil_acc = manifest.get("abmil.supported.acc")
        assert abs(abmil_acc - 0.811) < 0.001


class TestCleanCase:
    """Correct key/value pairings must pass."""

    def test_correct_abmil_acc(self, minimal_manifests, tmp_path):
        """abmil.supported.acc = 0.811 must pass."""
        ha_path, crn_path = minimal_manifests
        tex = r"Under ABMIL \AggNum{abmil.supported.acc}{0.811}."
        result = _run_check(tex, ha_path, crn_path, tmp_path)
        assert result == 0, "Expected exit 0 for correct key/value pair"

    def test_correct_meanpool_acc(self, minimal_manifests, tmp_path):
        """mean-pool.supported.acc = 0.778 must pass."""
        ha_path, crn_path = minimal_manifests
        tex = r"Mean-pool releases 9 slides at \AggNum{mean-pool.supported.acc}{0.778}."
        result = _run_check(tex, ha_path, crn_path, tmp_path)
        assert result == 0, "Expected exit 0 for correct mean-pool key/value pair"

    def test_unknown_key_rejected(self, minimal_manifests, tmp_path):
        """An unknown key must be rejected even if the value is valid."""
        ha_path, crn_path = minimal_manifests
        tex = r"\AggNum{nonexistent.metric}{0.811}."
        result = _run_check(tex, ha_path, crn_path, tmp_path)
        assert result != 0, "Expected non-zero exit for unknown key"

    def test_no_aggnum_clean(self, minimal_manifests, tmp_path):
        """A file with only \SafeNum and no \AggNum should pass cleanly."""
        ha_path, crn_path = minimal_manifests
        tex = r"Cohort size \SafeNum{500}, accuracy \SafeNum{0.811}."
        result = _run_check(tex, ha_path, crn_path, tmp_path)
        assert result == 0, "Expected exit 0 when no \\AggNum present"
