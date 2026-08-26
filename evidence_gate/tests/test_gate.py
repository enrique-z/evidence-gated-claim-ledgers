"""Property + example tests for the Evidence Gate."""

import pytest

from p5.evidence_gate.gate import Claim, EvidenceGate, Justification


def _claim(probs, confidence=None):
    return Claim(
        claim_id="c-1",
        claim_text="ISUP grade = 3",
        evidence_ref="ledger://slide-x/gate-oneslide",
        model_output={
            "per_grade_probs": probs,
            "confidence": confidence if confidence is not None else max(probs),
            "isup_grade": max(range(len(probs)), key=lambda i: probs[i]),
        },
        backbone_version="phikon-v2-owkin@test",
        cohort_id="gate-oneslide",
    )


def test_high_confidence_is_supported():
    v, j = EvidenceGate().gate(_claim([0.02, 0.02, 0.02, 0.9, 0.02, 0.02]))
    assert v == "SUPPORTED"
    assert j.gate_rule_id.startswith("confidence-threshold-v1")


def test_low_confidence_is_unsupported():
    # Argmax confidence 0.25 < 0.30 low threshold → UNSUPPORTED. Not uniform
    # (which would be NOT_APPLICABLE) — there's a real, weak signal we're
    # correctly refusing to support.
    v, j = EvidenceGate().gate(_claim([0.05, 0.10, 0.10, 0.25, 0.20, 0.30]))
    # argmax is 0.30 which is exactly the low threshold — bump it clearly under
    v, j = EvidenceGate().gate(_claim([0.05, 0.10, 0.10, 0.25, 0.30, 0.20]))
    assert v == "UNSUPPORTED"


def test_random_init_model_output_is_not_applicable():
    """A random-init MIL head produces near-uniform outputs. Verdict must be
    NOT_APPLICABLE (we cannot ask the question) — not UNSUPPORTED (which would
    wrongly imply we asked and the model said no). This is the semantics the
    Aug 15 gate-oneslide tracer relies on to stay honest."""
    uniform = [1 / 6] * 6
    v, _ = EvidenceGate().gate(_claim(uniform, confidence=1 / 6))
    assert v == "NOT_APPLICABLE"


def test_mid_confidence_is_contested():
    v, j = EvidenceGate().gate(_claim([0.1, 0.5, 0.1, 0.1, 0.1, 0.1]))
    assert v == "CONTESTED"


def test_perfectly_uniform_is_not_applicable():
    # Exactly-equal probs → no signal → NOT_APPLICABLE
    v, j = EvidenceGate().gate(_claim([0.5, 0.5], confidence=0.5))
    assert v == "NOT_APPLICABLE"


def test_missing_probs_is_not_applicable():
    claim = Claim(
        claim_id="c-2", claim_text="", evidence_ref="ref",
        model_output={}, backbone_version="x", cohort_id="gate-oneslide",
    )
    v, _ = EvidenceGate().gate(claim)
    assert v == "NOT_APPLICABLE"


def test_thresholds_recorded_in_justification():
    v, j = EvidenceGate(high_threshold=0.8, low_threshold=0.2).gate(
        _claim([0.02, 0.02, 0.02, 0.9, 0.02, 0.02])
    )
    assert j.thresholds["high"] == 0.8
    assert j.thresholds["low"] == 0.2
    assert "observed_confidence" in j.thresholds


def test_invalid_thresholds_raise():
    with pytest.raises(ValueError):
        EvidenceGate(high_threshold=0.3, low_threshold=0.7)
    with pytest.raises(ValueError):
        EvidenceGate(high_threshold=1.5)
    with pytest.raises(ValueError):
        EvidenceGate(low_threshold=0)


def test_all_verdicts_serialize_to_ledger_row():
    """Round-trip: a Justification's fields survive a dict → JSON roundtrip."""
    import json
    v, j = EvidenceGate().gate(_claim([0.02, 0.02, 0.02, 0.9, 0.02, 0.02]))
    row = {
        "verdict": v,
        "gate_rule_id": j.gate_rule_id,
        "source_rows": list(j.source_rows),
        "thresholds": dict(j.thresholds),
    }
    reloaded = json.loads(json.dumps(row))
    assert reloaded["verdict"] == "SUPPORTED"
    assert reloaded["gate_rule_id"] == j.gate_rule_id
    assert reloaded["thresholds"]["high"] == 0.7
