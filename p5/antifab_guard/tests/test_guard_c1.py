"""Phase C1 tests — canonical numeric matching, \\num{} support, \\SafeNum{}.

Every test here is designed to fail against the pre-C1 implementation
(substring `if token in claim_text or claim_text in line`) and pass against
the C1 implementation (canonical value + tolerance).
"""

import os
import tempfile
from pathlib import Path

import pytest

from p5.antifab_guard import check_manuscript, LedgerSnapshot
from p5.antifab_guard.guard import canonical_value


# All Phase C tests assume SAKANA layers are enabled — the SAKANA bypass
# checks are validated in test_guard.py separately.
@pytest.fixture(autouse=True)
def _sakana_env():
    prev = {k: os.environ.get(k) for k in [
        "SYNTHETIC_DATA_FORBIDDEN", "USE_GLENS_DATA", "USE_GEOMIP_DATA"
    ]}
    os.environ["SYNTHETIC_DATA_FORBIDDEN"] = "true"
    os.environ["USE_GLENS_DATA"] = "true"
    os.environ["USE_GEOMIP_DATA"] = "true"
    yield
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _tmp_manuscript(text: str, suffix: str = ".tex") -> str:
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as fh:
        fh.write(text)
        return fh.name


# ---------- canonicalisation unit tests ----------

def test_canonicalise_bare_decimal():
    assert canonical_value("0.734") == pytest.approx(0.734)


def test_canonicalise_percentage():
    assert canonical_value("87.3%") == pytest.approx(0.873)


def test_canonicalise_n_count():
    assert canonical_value("N=500") == pytest.approx(500.0)


def test_canonicalise_auroc_form():
    assert canonical_value("AUROC=0.92") == pytest.approx(0.92)


def test_canonicalise_num_macro():
    assert canonical_value("\\num{0.365}") == pytest.approx(0.365)


def test_canonicalise_returns_none_for_garbage():
    assert canonical_value("banana") is None
    assert canonical_value("") is None


# ---------- substring-false-positive elimination ----------

def test_pre_c1_substring_fp_no_longer_matches():
    """A ledger claim with '0.7778' MUST NOT support a manuscript literal
    '0.7' (which would silently pass under the pre-C1 substring match)."""
    ledger: LedgerSnapshot = {
        "entries": [{
            "claim_id": "c-supported-0.7778",
            "claim_text": "slide=X accuracy=0.7778 cohort=expansion-500",
            "verdict": "SUPPORTED",
        }],
    }
    manuscript = _tmp_manuscript(
        "The macro-accuracy was 0.7 in the pilot cohort.\n"
    )
    try:
        violations = check_manuscript(manuscript, ledger)
        no_claim = [v for v in violations if v.reason == "no_claim_id"]
        assert any(v.token == "0.7" for v in no_claim), (
            "0.7 must be flagged as no_claim_id since only 0.7778 is in the ledger."
        )
    finally:
        os.unlink(manuscript)


def test_supported_matches_within_tolerance():
    """A ledger with value 0.365 SUPPORTS a manuscript literal 0.3652
    to within NUMERIC_TOL — 4-decimal-place rounding is OK."""
    ledger: LedgerSnapshot = {
        "entries": [{
            "claim_id": "c-supported",
            "claim_text": "balanced_accuracy=0.365",
            "verdict": "SUPPORTED",
        }],
    }
    manuscript = _tmp_manuscript(
        "The balanced-accuracy of \\num{0.3652} was reported.\n"
    )
    try:
        violations = check_manuscript(manuscript, ledger)
        blocking = [v for v in violations
                    if v.reason in ("no_claim_id", "verdict_not_supported")
                    and "0.3652" in v.token or "0.365" in v.token]
        assert blocking == [], (
            f"0.3652 should match 0.365 to within tolerance, got {blocking}"
        )
    finally:
        os.unlink(manuscript)


# ---------- \\num{} macro support ----------

def test_num_macro_number_is_checked():
    ledger: LedgerSnapshot = {"entries": []}
    manuscript = _tmp_manuscript(
        "The accuracy was \\num{0.777}.\n"
    )
    try:
        violations = check_manuscript(manuscript, ledger)
        assert any(
            v.reason == "no_claim_id" and "0.777" in v.token for v in violations
        ), "0.777 inside \\num{} must be flagged when no ledger row exists."
    finally:
        os.unlink(manuscript)


def test_num_macro_number_passes_when_supported():
    ledger: LedgerSnapshot = {"entries": [{
        "claim_id": "c-supp",
        "claim_text": "accuracy=0.777",
        "verdict": "SUPPORTED",
    }]}
    manuscript = _tmp_manuscript(
        "The accuracy was \\num{0.777}.\n"
    )
    try:
        violations = check_manuscript(manuscript, ledger)
        no_claim = [v for v in violations if v.reason == "no_claim_id"]
        assert no_claim == [], "0.777 should now resolve to the SUPPORTED claim"
    finally:
        os.unlink(manuscript)


# ---------- \\SafeNum{} whitelist ----------

def test_safenum_whitelist():
    ledger: LedgerSnapshot = {"entries": []}
    manuscript = _tmp_manuscript(
        "We report \\SafeNum{9} SUPPORTED verdicts on expansion-500.\n"
    )
    try:
        violations = check_manuscript(manuscript, ledger)
        numeric_v = [v for v in violations if v.reason != "sakana_layer_bypass"]
        assert numeric_v == [], (
            f"SafeNum-wrapped 9 should not be flagged, got {numeric_v}"
        )
    finally:
        os.unlink(manuscript)


def test_safenum_only_whitelists_wrapped_numbers():
    """Numbers outside \\SafeNum{...} on the same line remain checked."""
    ledger: LedgerSnapshot = {"entries": []}
    manuscript = _tmp_manuscript(
        "We report \\SafeNum{9} SUPPORTED verdicts, but the raw accuracy 0.42 "
        "still needs a claim.\n"
    )
    try:
        violations = check_manuscript(manuscript, ledger)
        no_claim = [v for v in violations if v.reason == "no_claim_id"]
        assert any(v.token == "0.42" for v in no_claim), (
            "0.42 outside SafeNum must still be flagged."
        )
        assert not any(v.token == "9" for v in no_claim), (
            "9 inside SafeNum must NOT be flagged."
        )
    finally:
        os.unlink(manuscript)


# ---------- verdict-not-supported still flagged ----------

def test_contested_verdict_still_flags():
    ledger: LedgerSnapshot = {"entries": [{
        "claim_id": "c-contested",
        "claim_text": "accuracy=0.365",
        "verdict": "CONTESTED",
    }]}
    manuscript = _tmp_manuscript(
        "The reported accuracy was 0.365.\n"
    )
    try:
        violations = check_manuscript(manuscript, ledger)
        verdict_fail = [v for v in violations
                        if v.reason == "verdict_not_supported"]
        assert any(v.token == "0.365" for v in verdict_fail), (
            "0.365 matches a CONTESTED claim — must be reported."
        )
    finally:
        os.unlink(manuscript)
