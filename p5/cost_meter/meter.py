"""
Cost Meter: append-only ledger of cloud spend with a hard $300 cap and authorization gate.

Public API:
  - record(item: str, amount: float, timestamp: str, cohort_id: str) → None
  - total() → float
  - budget_remaining() → float
  - authorize(projected_amount: float) → bool

The ledger persists across process restarts. Every record is appended to a JSON-lines file
on disk. The cap is loaded from config.toml and cannot be overridden at runtime.
"""

import json
import os
from pathlib import Path
from typing import Optional

from .config import load_config


class CostMeterError(Exception):
    """Base exception for Cost Meter errors."""
    pass


class NegativeAmountError(CostMeterError):
    """Raised when attempting to record a negative amount."""
    pass


class CostMeter:
    """Append-only ledger of cloud spend with budget enforcement."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Cost Meter.

        Args:
            config_path: Path to config.toml. If None, uses default.
                        Raises FileNotFoundError if path does not exist.
        """
        self.config = load_config(config_path)
        self.cap = self.config["budget"]["cap"]
        self.ledger_path = Path(self.config["storage"]["ledger_path"]).expanduser()

        # Ensure parent directory exists
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, item: str, amount: float, timestamp: str, cohort_id: str) -> None:
        """
        Record a spend item to the ledger.

        Args:
            item: Description of the spend item (e.g., 'compute-hours', 'storage').
            amount: Amount in USD. Must be >= 0.
            timestamp: ISO 8601 timestamp of the spend.
            cohort_id: Identifier of the cohort or run this item belongs to.

        Raises:
            NegativeAmountError: If amount < 0.
        """
        if amount < 0:
            raise NegativeAmountError(
                f"Cannot record negative amount: {amount}. "
                "Spend items must be non-negative."
            )

        record = {
            "item": item,
            "amount": amount,
            "timestamp": timestamp,
            "cohort_id": cohort_id,
        }

        # Append to ledger
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def total(self) -> float:
        """
        Return the sum of all recorded amounts.

        Returns:
            Total spend in USD.
        """
        if not self.ledger_path.exists():
            return 0.0

        total = 0.0
        with open(self.ledger_path, "r") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    total += record["amount"]

        return total

    def budget_remaining(self) -> float:
        """
        Return the remaining budget.

        Returns:
            budget_remaining = cap - total()
        """
        return self.cap - self.total()

    def authorize(self, projected_amount: float) -> bool:
        """
        Check if a projected spend is within budget.

        Returns False iff projected_amount + total() > cap.

        Args:
            projected_amount: Projected spend in USD for the operation.

        Returns:
            True if the operation can be authorized, False if it would breach the cap.
        """
        return projected_amount + self.total() <= self.cap
