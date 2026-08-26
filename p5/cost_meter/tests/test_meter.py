"""
Pytest tests for Cost Meter module.

Covers acceptance criteria:
  - Property tests: sum of recorded amounts equals total()
  - Property tests: budget_remaining() = cap − total()
  - Property tests: authorize(x) returns False iff x + total() > cap
  - Persistence: write, close, re-open, read yields the same rows
  - Negative amount raises loudly with a clear message
  - Cap loaded from config (test with a custom config path)
"""

import json
import os
import tempfile
from pathlib import Path
from typing import List

import pytest

from p5.cost_meter import CostMeter, NegativeAmountError


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for config and ledger files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def default_config_path(temp_config_dir):
    """Create a default config.toml file."""
    config_path = temp_config_dir / "config.toml"
    config_path.write_text(
        """[budget]
cap = 300.0

[storage]
ledger_path = "{ledger_path}"
""".format(ledger_path=str(temp_config_dir / "cost_meter.jsonl"))
    )
    return config_path


@pytest.fixture
def custom_cap_config_path(temp_config_dir):
    """Create a config.toml file with a custom cap."""
    config_path = temp_config_dir / "custom_config.toml"
    config_path.write_text(
        """[budget]
cap = 100.0

[storage]
ledger_path = "{ledger_path}"
""".format(ledger_path=str(temp_config_dir / "custom_cost_meter.jsonl"))
    )
    return config_path


@pytest.fixture
def meter(default_config_path):
    """Create a CostMeter instance with a temporary ledger."""
    return CostMeter(config_path=str(default_config_path))


@pytest.fixture
def meter_custom_cap(custom_cap_config_path):
    """Create a CostMeter instance with a custom cap."""
    return CostMeter(config_path=str(custom_cap_config_path))


class TestRecordAndTotal:
    """Test recording and total calculation."""

    def test_empty_ledger_has_zero_total(self, meter):
        """An empty ledger should have total() = 0."""
        assert meter.total() == 0.0

    def test_record_single_item(self, meter):
        """Recording one item should update total."""
        meter.record("compute", 50.0, "2026-08-05T10:00:00Z", "pilot-50")
        assert meter.total() == 50.0

    def test_record_multiple_items_sum(self, meter):
        """Recording multiple items should sum correctly."""
        meter.record("compute", 50.0, "2026-08-05T10:00:00Z", "pilot-50")
        meter.record("storage", 25.5, "2026-08-05T10:30:00Z", "pilot-50")
        meter.record("egress", 12.25, "2026-08-05T11:00:00Z", "pilot-50")
        assert meter.total() == 87.75

    def test_property_sum_of_amounts_equals_total(self, meter):
        """Property: sum of recorded amounts equals total()."""
        amounts = [10.0, 20.5, 30.75, 5.0]
        for i, amount in enumerate(amounts):
            meter.record(f"item-{i}", amount, f"2026-08-05T{i:02d}:00:00Z", "test")
        assert meter.total() == sum(amounts)


class TestBudgetRemaining:
    """Test budget remaining calculation."""

    def test_full_budget_when_empty(self, meter):
        """With no spend, budget_remaining should equal cap."""
        assert meter.budget_remaining() == 300.0

    def test_budget_remaining_after_spend(self, meter):
        """budget_remaining = cap - total()."""
        meter.record("compute", 75.0, "2026-08-05T10:00:00Z", "pilot-50")
        assert meter.budget_remaining() == 225.0

    def test_property_budget_remaining_equals_cap_minus_total(self, meter):
        """Property: budget_remaining() = cap - total()."""
        cap = meter.cap
        total = 0.0
        amounts = [10.0, 20.5, 30.75]
        for i, amount in enumerate(amounts):
            meter.record(f"item-{i}", amount, f"2026-08-05T{i:02d}:00:00Z", "test")
            total += amount
            assert meter.budget_remaining() == cap - total


class TestAuthorize:
    """Test authorization logic."""

    def test_authorize_within_budget(self, meter):
        """authorize(x) returns True if x + total() <= cap."""
        meter.record("compute", 100.0, "2026-08-05T10:00:00Z", "pilot-50")
        assert meter.authorize(100.0) is True
        assert meter.authorize(199.99) is True
        assert meter.authorize(200.0) is True

    def test_authorize_exceeds_budget(self, meter):
        """authorize(x) returns False if x + total() > cap."""
        meter.record("compute", 100.0, "2026-08-05T10:00:00Z", "pilot-50")
        assert meter.authorize(200.01) is False
        assert meter.authorize(300.0) is False

    def test_authorize_at_boundary(self, meter):
        """authorize(x) at exact boundary."""
        meter.record("compute", 100.0, "2026-08-05T10:00:00Z", "pilot-50")
        # Exactly at cap: 100 + 200 = 300
        assert meter.authorize(200.0) is True
        # Just over cap: 100 + 200.01 = 300.01
        assert meter.authorize(200.01) is False

    def test_property_authorize_false_iff_exceeds_cap(self, meter):
        """Property: authorize(x) returns False iff x + total() > cap."""
        meter.record("compute", 150.0, "2026-08-05T10:00:00Z", "pilot-50")
        current_total = meter.total()
        cap = meter.cap

        # Test various amounts
        test_amounts = [50.0, 100.0, 149.99, 150.0, 150.01, 200.0]
        for amount in test_amounts:
            expected = (amount + current_total) <= cap
            assert meter.authorize(amount) == expected


class TestNegativeAmount:
    """Test that negative amounts are rejected."""

    def test_negative_amount_raises_error(self, meter):
        """Recording a negative amount should raise NegativeAmountError."""
        with pytest.raises(NegativeAmountError) as exc_info:
            meter.record("refund", -50.0, "2026-08-05T10:00:00Z", "pilot-50")
        assert "negative" in str(exc_info.value).lower()
        assert "non-negative" in str(exc_info.value).lower()

    def test_zero_amount_is_allowed(self, meter):
        """Recording a zero amount should be allowed."""
        meter.record("zero-cost-item", 0.0, "2026-08-05T10:00:00Z", "pilot-50")
        assert meter.total() == 0.0

    def test_error_message_includes_amount(self, meter):
        """Error message should include the problematic amount."""
        with pytest.raises(NegativeAmountError) as exc_info:
            meter.record("item", -42.5, "2026-08-05T10:00:00Z", "test")
        assert "-42.5" in str(exc_info.value)


class TestPersistence:
    """Test that ledger persists across process restarts."""

    def test_write_close_reopen_read(self, default_config_path):
        """Write, close, re-open, read should yield the same rows."""
        # First meter: write data
        meter1 = CostMeter(config_path=str(default_config_path))
        meter1.record("compute", 50.0, "2026-08-05T10:00:00Z", "pilot-50")
        meter1.record("storage", 25.5, "2026-08-05T10:30:00Z", "pilot-50")
        total1 = meter1.total()

        # Second meter: re-open and read
        meter2 = CostMeter(config_path=str(default_config_path))
        total2 = meter2.total()

        assert total1 == total2 == 75.5

    def test_ledger_is_append_only(self, default_config_path):
        """Ledger should be append-only; old rows should never be rewritten."""
        meter1 = CostMeter(config_path=str(default_config_path))
        meter1.record("item-1", 10.0, "2026-08-05T10:00:00Z", "run-1")

        meter2 = CostMeter(config_path=str(default_config_path))
        meter2.record("item-2", 20.0, "2026-08-05T11:00:00Z", "run-2")

        meter3 = CostMeter(config_path=str(default_config_path))
        assert meter3.total() == 30.0

    def test_ledger_file_contains_json_lines(self, default_config_path):
        """Ledger file should be JSON-lines format (one JSON object per line)."""
        meter = CostMeter(config_path=str(default_config_path))
        meter.record("compute", 50.0, "2026-08-05T10:00:00Z", "pilot-50")
        meter.record("storage", 25.5, "2026-08-05T10:30:00Z", "pilot-50")

        # Read ledger file directly
        ledger_path = Path(default_config_path).parent / "cost_meter.jsonl"
        with open(ledger_path, "r") as f:
            lines = f.readlines()

        assert len(lines) == 2
        for line in lines:
            record = json.loads(line)
            assert "item" in record
            assert "amount" in record
            assert "timestamp" in record
            assert "cohort_id" in record


class TestConfigLoading:
    """Test that cap is loaded from config."""

    def test_cap_from_default_config(self, meter):
        """Cap should be loaded from config.toml (default 300.0)."""
        assert meter.cap == 300.0

    def test_cap_from_custom_config(self, meter_custom_cap):
        """Cap should be loaded from a custom config.toml."""
        assert meter_custom_cap.cap == 100.0

    def test_custom_cap_affects_budget_remaining(self, meter_custom_cap):
        """Custom cap should affect budget_remaining() calculations."""
        meter_custom_cap.record("compute", 50.0, "2026-08-05T10:00:00Z", "test")
        assert meter_custom_cap.budget_remaining() == 50.0

    def test_custom_cap_affects_authorize(self, meter_custom_cap):
        """Custom cap should affect authorize() logic."""
        meter_custom_cap.record("compute", 50.0, "2026-08-05T10:00:00Z", "test")
        # With cap=100 and total=50, authorize(50) should pass
        assert meter_custom_cap.authorize(50.0) is True
        # authorize(50.01) should fail
        assert meter_custom_cap.authorize(50.01) is False

    def test_config_not_found_raises_error(self, temp_config_dir):
        """Loading a non-existent config should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            CostMeter(config_path=str(temp_config_dir / "does_not_exist.toml"))


class TestIntegration:
    """End-to-end integration tests."""

    def test_typical_workflow(self, meter):
        """Simulate a typical workflow: record, check budget, authorize next step."""
        # Record first step
        meter.record("compute", 100.0, "2026-08-05T10:00:00Z", "pilot-50")
        assert meter.total() == 100.0
        assert meter.budget_remaining() == 200.0

        # Check if we can authorize next step
        projected_step_2 = 120.0
        assert meter.authorize(projected_step_2) is True

        # Record second step
        meter.record("compute", 120.0, "2026-08-05T11:00:00Z", "pilot-50")
        assert meter.total() == 220.0

        # Check if we can authorize third step
        projected_step_3 = 100.0
        assert meter.authorize(projected_step_3) is False  # Would exceed cap

        # Check if smaller step fits
        projected_step_3_alt = 80.0
        assert meter.authorize(projected_step_3_alt) is True

    def test_multiple_cohorts(self, meter):
        """Ledger should track multiple cohorts under a single budget."""
        meter.record("compute", 50.0, "2026-08-05T10:00:00Z", "gate-oneslide")
        meter.record("compute", 100.0, "2026-08-05T11:00:00Z", "pilot-50")
        meter.record("compute", 130.0, "2026-08-05T12:00:00Z", "expansion-500")

        assert meter.total() == 280.0
        assert meter.authorize(20.0) is True
        assert meter.authorize(20.01) is False
