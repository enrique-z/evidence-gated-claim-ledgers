"""Tests for the Claim Ledger."""

from pathlib import Path

from p5.claim_ledger.ledger import ClaimLedger, LedgerRow


def _row(claim_id="c-1", verdict="SUPPORTED", cohort="gate-oneslide"):
    return LedgerRow(
        claim_id=claim_id,
        claim_text="ISUP grade = 3",
        evidence_ref="ref://x",
        verdict=verdict,
        justification={"gate_rule_id": "gr-1"},
        timestamp="2026-08-05T18:00:00Z",
        backbone_version="phikon-v2-owkin@test",
        cohort_id=cohort,
        gate_rule_id="gr-1",
    )


def test_append_and_read_round_trip(tmp_path):
    p = tmp_path / "l.jsonl"
    L = ClaimLedger(p)
    L.append(_row("c-1"))
    L.append(_row("c-2", verdict="UNSUPPORTED"))
    rows = list(L.rows())
    assert len(rows) == 2
    assert rows[0].claim_id == "c-1" and rows[0].verdict == "SUPPORTED"
    assert rows[1].claim_id == "c-2" and rows[1].verdict == "UNSUPPORTED"


def test_append_only_survives_reopen(tmp_path):
    p = tmp_path / "l.jsonl"
    L1 = ClaimLedger(p)
    L1.append(_row("c-1"))
    del L1
    L2 = ClaimLedger(p)
    L2.append(_row("c-2"))
    rows = list(L2.rows())
    assert [r.claim_id for r in rows] == ["c-1", "c-2"]


def test_verdict_distribution(tmp_path):
    p = tmp_path / "l.jsonl"
    L = ClaimLedger(p)
    L.append(_row("c-1", verdict="SUPPORTED"))
    L.append(_row("c-2", verdict="UNSUPPORTED"))
    L.append(_row("c-3", verdict="UNSUPPORTED"))
    L.append(_row("c-4", verdict="CONTESTED"))
    dist = L.verdict_distribution()
    assert dist == {"SUPPORTED": 1, "UNSUPPORTED": 2, "CONTESTED": 1}


def test_verdict_distribution_filtered_by_cohort(tmp_path):
    p = tmp_path / "l.jsonl"
    L = ClaimLedger(p)
    L.append(_row("c-1", cohort="gate-oneslide"))
    L.append(_row("c-2", cohort="pilot-50", verdict="UNSUPPORTED"))
    assert L.count("gate-oneslide") == 1
    assert L.count("pilot-50") == 1
    assert L.verdict_distribution("gate-oneslide") == {"SUPPORTED": 1}
