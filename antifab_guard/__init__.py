"""
Anti-Fabrication Guard: validates numeric claims in manuscripts.

This module wraps (not replaces) the five SAKANA enforcement layers
to ensure every numeric claim in a manuscript resolves to a SUPPORTED
verdict in the claim ledger.

Public surface:
    - check_manuscript(path, ledger_snapshot) → list[Violation]
    - Violation type with (token, line_number, reason, claim_id_or_none)
    - LedgerEntry / LedgerSnapshot types for consumers
"""

from .guard import Violation, check_manuscript, NumericTokenMatcher
from .ledger_types import LedgerEntry, LedgerSnapshot, Verdict
from .sakana_status import SakanaLayerStatus, check_sakana_status

__all__ = [
    "check_manuscript",
    "Violation",
    "NumericTokenMatcher",
    "LedgerEntry",
    "LedgerSnapshot",
    "Verdict",
    "SakanaLayerStatus",
    "check_sakana_status",
]
