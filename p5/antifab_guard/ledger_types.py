"""
Minimal type definitions for ledger entries consumed by the Anti-Fabrication Guard.

The real Claim Ledger is a separate module; here we define only the shapes
that the guard needs to validate manuscripts.
"""

from typing import TypedDict, Literal


Verdict = Literal["SUPPORTED", "UNSUPPORTED", "NOT_APPLICABLE", "CONTESTED"]


class LedgerEntry(TypedDict, total=False):
    """
    A single row in the claim ledger.

    The guard validates that every numeric token in a manuscript
    has a ledger entry with verdict=SUPPORTED.
    """
    claim_id: str
    claim_text: str
    evidence_ref: str
    verdict: Verdict
    justification: str
    timestamp: str
    backbone_version: str
    cohort_id: str
    gate_rule_id: str


class LedgerSnapshot(TypedDict, total=False):
    """
    A snapshot of ledger entries for validation.

    Passed to check_manuscript() to validate numeric tokens.
    The guard treats this as immutable during a single check.
    """
    entries: list[LedgerEntry]
