"""
Test suite for the Anti-Fabrication Guard.

Covers:
- Numeric token recognition (all types)
- Manuscript validation against ledger
- SAKANA layer bypass detection
- Edge cases and error handling
"""

import os
import pytest
from pathlib import Path

from p5.antifab_guard import (
    check_manuscript,
    Violation,
    NumericTokenMatcher,
    LedgerSnapshot,
    check_sakana_status,
)


# Fixture directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def supported_ledger() -> LedgerSnapshot:
    """Ledger with SUPPORTED verdicts for all typical tokens."""
    return {
        "entries": [
            {
                "claim_id": "claim_1",
                "claim_text": "N=500",
                "verdict": "SUPPORTED",
                "backbone_version": "phikon-v2",
            },
            {
                "claim_id": "claim_2",
                "claim_text": "AUROC=0.92",
                "verdict": "SUPPORTED",
                "backbone_version": "phikon-v2",
            },
            {
                "claim_id": "claim_3",
                "claim_text": "87.3%",
                "verdict": "SUPPORTED",
                "backbone_version": "phikon-v2",
            },
            {
                "claim_id": "claim_4",
                "claim_text": "0.734",
                "verdict": "SUPPORTED",
                "backbone_version": "phikon-v2",
            },
            {
                "claim_id": "claim_5",
                "claim_text": "N=50",
                "verdict": "SUPPORTED",
                "backbone_version": "phikon-v2",
            },
            {
                "claim_id": "claim_6",
                "claim_text": "95.2%",
                "verdict": "SUPPORTED",
                "backbone_version": "phikon-v2",
            },
            {
                "claim_id": "claim_7",
                "claim_text": "AUC 0.881",
                "verdict": "SUPPORTED",
                "backbone_version": "phikon-v2",
            },
            {
                "claim_id": "claim_8",
                "claim_text": "0.512",
                "verdict": "SUPPORTED",
                "backbone_version": "phikon-v2",
            },
            {
                "claim_id": "claim_9",
                "claim_text": "N=30",
                "verdict": "SUPPORTED",
                "backbone_version": "phikon-v2",
            },
        ]
    }


@pytest.fixture
def contested_ledger() -> LedgerSnapshot:
    """Ledger with a CONTESTED verdict."""
    return {
        "entries": [
            {
                "claim_id": "contested_1",
                "claim_text": "N=100",
                "verdict": "CONTESTED",
                "backbone_version": "phikon-v2",
            },
            {
                "claim_id": "contested_2",
                "claim_text": "AUROC=0.75",
                "verdict": "CONTESTED",
                "backbone_version": "phikon-v2",
            },
        ]
    }


@pytest.fixture
def unsupported_ledger() -> LedgerSnapshot:
    """Ledger with UNSUPPORTED verdicts."""
    return {
        "entries": [
            {
                "claim_id": "unsupported_1",
                "claim_text": "N=50",
                "verdict": "UNSUPPORTED",
                "backbone_version": "phikon-v2",
            },
            {
                "claim_id": "unsupported_2",
                "claim_text": "AUC 0.65",
                "verdict": "UNSUPPORTED",
                "backbone_version": "phikon-v2",
            },
        ]
    }


@pytest.fixture
def not_applicable_ledger() -> LedgerSnapshot:
    """Ledger with NOT_APPLICABLE verdicts."""
    return {
        "entries": [
            {
                "claim_id": "na_1",
                "claim_text": "N=200",
                "verdict": "NOT_APPLICABLE",
                "backbone_version": "phikon-v2",
            },
            {
                "claim_id": "na_2",
                "claim_text": "0.78",
                "verdict": "NOT_APPLICABLE",
                "backbone_version": "phikon-v2",
            },
        ]
    }


# ============================================================================
# Token Recognition Tests
# ============================================================================


class TestNumericTokenMatcher:
    """Test the NumericTokenMatcher pattern recognition."""

    def test_recognizes_percentages(self):
        """Percentages like 12.3%, 12%, 100% are recognized."""
        tokens = NumericTokenMatcher.find_tokens("Accuracy was 87.3%", 1)
        assert any(t[0] == "87.3%" for t in tokens)

        tokens = NumericTokenMatcher.find_tokens("Performance: 95%", 1)
        assert any(t[0] == "95%" for t in tokens)

    def test_recognizes_auroc_values(self):
        """AUROC and AUC patterns are recognized."""
        tokens = NumericTokenMatcher.find_tokens("AUROC=0.87 on test set", 1)
        assert any("0.87" in t[0] for t in tokens)

        tokens = NumericTokenMatcher.find_tokens("The AUC 0.912 was excellent", 1)
        assert any("0.912" in t[0] for t in tokens)

    def test_recognizes_n_counts(self):
        """N counts in various formats are recognized."""
        tokens = NumericTokenMatcher.find_tokens("We used N=500 samples", 1)
        assert any("500" in t[0] for t in tokens)

        tokens = NumericTokenMatcher.find_tokens("The cohort: n = 50", 1)
        assert any("50" in t[0] for t in tokens)

    def test_recognizes_bare_decimals_near_keywords(self):
        """Bare decimals near result keywords are recognized."""
        tokens = NumericTokenMatcher.find_tokens("The accuracy was 0.734", 1)
        assert any("0.734" in t[0] for t in tokens)

        tokens = NumericTokenMatcher.find_tokens("F1 score of 0.512 achieved", 1)
        assert any("0.512" in t[0] for t in tokens)

    def test_skips_latex_commands(self):
        """LaTeX commands with numbers are skipped."""
        assert NumericTokenMatcher.should_skip_line("\\documentclass{article}")
        assert NumericTokenMatcher.should_skip_line("\\usepackage{neurips_2026}")
        assert NumericTokenMatcher.should_skip_line("\\section{Introduction}")

    def test_skips_comments(self):
        """Comment lines are skipped."""
        assert NumericTokenMatcher.should_skip_line("% This is a comment")

    def test_skips_page_numbers(self):
        """Lines with only digits (page numbers) are skipped."""
        assert NumericTokenMatcher.should_skip_line("1")
        assert NumericTokenMatcher.should_skip_line("42")

    def test_line_numbers_are_correct(self):
        """Token line numbers match the input line number."""
        tokens = NumericTokenMatcher.find_tokens("AUROC=0.92", 5)
        assert all(t[1] == 5 for t in tokens)


# ============================================================================
# Manuscript Validation Tests
# ============================================================================


class TestCheckManuscript:
    """Test manuscript validation against ledger."""

    def test_all_supported_passes(self, supported_ledger):
        """Manuscript with only SUPPORTED claims passes."""
        manuscript = FIXTURES_DIR / "all_supported.tex"
        violations = check_manuscript(str(manuscript), supported_ledger)
        # Note: some violations may be empty list or may not match exactly
        # because token matching is simplified in this version.
        # We expect no "no_claim_id" violations for this fixture.
        no_claim_violations = [v for v in violations if v.reason == "no_claim_id"]
        assert len(no_claim_violations) == 0, f"Expected no missing claims, got {no_claim_violations}"

    def test_one_fabricated_fails(self, supported_ledger):
        """Manuscript with one fabricated claim fails."""
        manuscript = FIXTURES_DIR / "one_fabricated.md"
        violations = check_manuscript(str(manuscript), supported_ledger)
        # Should find the 99.9% that has no ledger entry
        fabricated = [v for v in violations if v.reason == "no_claim_id"]
        assert len(fabricated) > 0, "Should find at least one unsupported claim"
        assert any("99.9%" in v.token for v in fabricated)

    def test_contested_verdict_fails(self, contested_ledger):
        """Manuscript with CONTESTED verdict fails."""
        manuscript = FIXTURES_DIR / "contested_verdict.tex"
        violations = check_manuscript(str(manuscript), contested_ledger)
        # Should report verdict_not_supported
        contested = [v for v in violations if v.reason == "verdict_not_supported"]
        assert len(contested) > 0, "Should flag CONTESTED verdicts"

    def test_unsupported_verdict_fails(self, unsupported_ledger):
        """Manuscript with UNSUPPORTED verdict fails."""
        manuscript = FIXTURES_DIR / "unsupported_verdict.md"
        violations = check_manuscript(str(manuscript), unsupported_ledger)
        unsupported = [v for v in violations if v.reason == "verdict_not_supported"]
        assert len(unsupported) > 0, "Should flag UNSUPPORTED verdicts"

    def test_not_applicable_verdict_fails(self, not_applicable_ledger):
        """Manuscript with NOT_APPLICABLE verdict fails."""
        manuscript = FIXTURES_DIR / "not_applicable_verdict.md"
        violations = check_manuscript(str(manuscript), not_applicable_ledger)
        na = [v for v in violations if v.reason == "verdict_not_supported"]
        assert len(na) > 0, "Should flag NOT_APPLICABLE verdicts"

    def test_no_numeric_tokens_passes(self):
        """Pure prose manuscript with no numbers passes."""
        manuscript = FIXTURES_DIR / "no_numeric_tokens.tex"
        empty_ledger: LedgerSnapshot = {"entries": []}
        violations = check_manuscript(str(manuscript), empty_ledger)
        # Should have at most a sakana layer bypass, but no numeric claim violations
        numeric_violations = [v for v in violations if v.reason != "sakana_layer_bypass"]
        assert len(numeric_violations) == 0

    def test_all_token_types_recognized(self):
        """Fixture with all token types is properly scanned."""
        manuscript = FIXTURES_DIR / "all_token_types.md"
        # Create a ledger with all token types
        ledger: LedgerSnapshot = {
            "entries": [
                {"claim_id": "c1", "claim_text": "87.3%", "verdict": "SUPPORTED"},
                {"claim_id": "c2", "claim_text": "AUROC=0.92", "verdict": "SUPPORTED"},
                {"claim_id": "c3", "claim_text": "AUC 0.915", "verdict": "SUPPORTED"},
                {"claim_id": "c4", "claim_text": "N=500", "verdict": "SUPPORTED"},
                {"claim_id": "c5", "claim_text": "N=50", "verdict": "SUPPORTED"},
                {"claim_id": "c6", "claim_text": "n = 250", "verdict": "SUPPORTED"},
                {"claim_id": "c7", "claim_text": "0.734", "verdict": "SUPPORTED"},
                {"claim_id": "c8", "claim_text": "0.891", "verdict": "SUPPORTED"},
                {"claim_id": "c9", "claim_text": "0.756", "verdict": "SUPPORTED"},
            ]
        }
        violations = check_manuscript(str(manuscript), ledger)
        # All tokens should be supported
        numeric_failures = [v for v in violations if v.reason != "sakana_layer_bypass"]
        assert len(numeric_failures) == 0

    def test_file_not_found_raises(self):
        """Non-existent manuscript raises FileNotFoundError."""
        empty_ledger: LedgerSnapshot = {"entries": []}
        with pytest.raises(FileNotFoundError):
            check_manuscript("/nonexistent/path.tex", empty_ledger)

    def test_violation_structure(self, supported_ledger):
        """Violation objects have required fields."""
        manuscript = FIXTURES_DIR / "one_fabricated.md"
        violations = check_manuscript(str(manuscript), supported_ledger)
        if violations:
            v = violations[0]
            assert hasattr(v, "token")
            assert hasattr(v, "line_number")
            assert hasattr(v, "reason")
            assert hasattr(v, "claim_id_or_none")


# ============================================================================
# SAKANA Layer Bypass Tests
# ============================================================================


class TestSakanaLayerBypassDetection:
    """Test detection of disabled SAKANA enforcement layers."""

    def test_disabling_layer_1_triggers_bypass(self, supported_ledger):
        """Disabling layer 1 (RNG hook) triggers bypass violation."""
        # Save original env var
        orig_val = os.environ.get("SYNTHETIC_DATA_FORBIDDEN")
        try:
            # Disable layer 1
            os.environ["SYNTHETIC_DATA_FORBIDDEN"] = "false"

            manuscript = FIXTURES_DIR / "all_supported.tex"
            violations = check_manuscript(str(manuscript), supported_ledger)

            # Should have a sakana_layer_bypass violation
            bypass_violations = [v for v in violations if v.reason == "sakana_layer_bypass"]
            assert len(bypass_violations) > 0, "Should detect layer 1 bypass"
        finally:
            # Restore original
            if orig_val is None:
                os.environ.pop("SYNTHETIC_DATA_FORBIDDEN", None)
            else:
                os.environ["SYNTHETIC_DATA_FORBIDDEN"] = orig_val

    def test_disabling_layer_2_triggers_bypass(self, supported_ledger):
        """Disabling layer 2 (env vars) triggers bypass violation."""
        orig_use_glens = os.environ.get("USE_GLENS_DATA")
        orig_use_geomip = os.environ.get("USE_GEOMIP_DATA")
        try:
            os.environ["USE_GLENS_DATA"] = "false"
            os.environ["USE_GEOMIP_DATA"] = "true"

            manuscript = FIXTURES_DIR / "all_supported.tex"
            violations = check_manuscript(str(manuscript), supported_ledger)

            bypass_violations = [v for v in violations if v.reason == "sakana_layer_bypass"]
            assert len(bypass_violations) > 0, "Should detect layer 2 bypass"
        finally:
            if orig_use_glens is None:
                os.environ.pop("USE_GLENS_DATA", None)
            else:
                os.environ["USE_GLENS_DATA"] = orig_use_glens
            if orig_use_geomip is None:
                os.environ.pop("USE_GEOMIP_DATA", None)
            else:
                os.environ["USE_GEOMIP_DATA"] = orig_use_geomip

    def test_all_layers_enabled_passes(self, supported_ledger):
        """With all layers enabled, no bypass violations occur."""
        # Set all required env vars
        os.environ["SYNTHETIC_DATA_FORBIDDEN"] = "true"
        os.environ["USE_GLENS_DATA"] = "true"
        os.environ["USE_GEOMIP_DATA"] = "true"

        manuscript = FIXTURES_DIR / "all_supported.tex"
        violations = check_manuscript(str(manuscript), supported_ledger)

        # Should have no bypass violations (may have no_claim_id for unmatched tokens)
        bypass_violations = [v for v in violations if v.reason == "sakana_layer_bypass"]
        assert len(bypass_violations) == 0, f"Should not have bypasses when all layers enabled, got {bypass_violations}"

    def test_check_sakana_status(self):
        """check_sakana_status() returns correct layer status."""
        # Set a consistent state
        os.environ["SYNTHETIC_DATA_FORBIDDEN"] = "true"
        os.environ["USE_GLENS_DATA"] = "true"
        os.environ["USE_GEOMIP_DATA"] = "true"

        status = check_sakana_status()

        assert status.layer_1_rng_hook is True
        assert status.layer_2_env_lock is True
        assert status.any_layer_disabled() is False
        assert status.disabled_layers() == []


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_ledger(self):
        """Empty ledger causes all tokens to fail."""
        manuscript = FIXTURES_DIR / "all_supported.tex"
        empty_ledger: LedgerSnapshot = {"entries": []}
        violations = check_manuscript(str(manuscript), empty_ledger)

        # All numeric tokens should fail with no_claim_id
        no_claim = [v for v in violations if v.reason == "no_claim_id"]
        assert len(no_claim) > 0, "Should find unsupported tokens in empty ledger"

    def test_empty_manuscript_passes(self, supported_ledger):
        """Empty manuscript passes (no tokens to check)."""
        # Create an empty file
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            violations = check_manuscript(temp_path, supported_ledger)
            numeric_violations = [v for v in violations if v.reason != "sakana_layer_bypass"]
            assert len(numeric_violations) == 0
        finally:
            os.unlink(temp_path)

    def test_multiple_tokens_per_line(self):
        """Lines with multiple tokens are all checked."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("Results: N=100 with AUROC=0.87 and 92.5% accuracy\n")
            temp_path = f.name

        try:
            ledger: LedgerSnapshot = {
                "entries": [
                    {"claim_id": "c1", "claim_text": "N=100", "verdict": "SUPPORTED"},
                    {"claim_id": "c2", "claim_text": "AUROC=0.87", "verdict": "SUPPORTED"},
                    {"claim_id": "c3", "claim_text": "92.5%", "verdict": "SUPPORTED"},
                ]
            }
            violations = check_manuscript(temp_path, ledger)
            numeric_violations = [v for v in violations if v.reason != "sakana_layer_bypass"]
            assert len(numeric_violations) == 0
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
