"""Property + example tests for EntropyGate, MarginGate, ConformalGate."""

import math

import pytest

from p5.evidence_gate.alt_gates import (
    ConformalCalibration,
    ConformalGate,
    EntropyGate,
    MarginGate,
    calibrate_lac,
)
from p5.evidence_gate.gate import Claim


def _claim(probs, cid="c-1"):
    return Claim(
        claim_id=cid,
        claim_text="ISUP grade = X",
        evidence_ref="ledger://slide-x/alt-gates",
        model_output={
            "per_grade_probs": list(probs),
            "confidence": max(probs),
            "isup_grade": max(range(len(probs)), key=lambda i: probs[i]),
        },
        backbone_version="phikon-v2-owkin@test",
        cohort_id="alt-gates-test",
    )


# ---------- EntropyGate ----------

def test_entropy_low_entropy_is_supported():
    v, j = EntropyGate().gate(_claim([0.02, 0.02, 0.02, 0.9, 0.02, 0.02]))
    assert v == "SUPPORTED"
    assert j.gate_rule_id.startswith("entropy-threshold-v1")
    assert j.thresholds["observed_entropy"] < 1.10


def test_entropy_high_entropy_is_unsupported():
    v, _ = EntropyGate().gate(_claim([0.10, 0.10, 0.20, 0.20, 0.20, 0.20]))
    assert v == "UNSUPPORTED"


def test_entropy_mid_entropy_is_contested():
    v, _ = EntropyGate().gate(_claim([0.05, 0.05, 0.05, 0.55, 0.15, 0.15]))
    assert v == "CONTESTED"


def test_entropy_uniform_is_not_applicable():
    v, _ = EntropyGate().gate(_claim([1 / 6] * 6))
    assert v == "NOT_APPLICABLE"


def test_entropy_invalid_thresholds_raise():
    with pytest.raises(ValueError):
        EntropyGate(entropy_low=1.5, entropy_high=1.0)


# ---------- MarginGate ----------

def test_margin_large_gap_is_supported():
    v, j = MarginGate().gate(_claim([0.02, 0.02, 0.02, 0.9, 0.02, 0.02]))
    assert v == "SUPPORTED"
    assert j.gate_rule_id.startswith("margin-threshold-v1")


def test_margin_small_gap_is_unsupported():
    v, _ = MarginGate().gate(_claim([0.18, 0.19, 0.16, 0.17, 0.15, 0.15]))
    assert v == "UNSUPPORTED"


def test_margin_mid_gap_is_contested():
    # top1=0.4, top2=0.2 → gap 0.2 (between low=0.05 and high=0.55)
    v, _ = MarginGate().gate(_claim([0.05, 0.4, 0.2, 0.15, 0.15, 0.05]))
    assert v == "CONTESTED"


def test_margin_uniform_is_not_applicable():
    v, _ = MarginGate().gate(_claim([1 / 6] * 6))
    assert v == "NOT_APPLICABLE"


def test_margin_invalid_thresholds_raise():
    with pytest.raises(ValueError):
        MarginGate(margin_low=0.9, margin_high=0.3)


# ---------- ConformalGate ----------

def _fake_cal_rows(pairs):
    """pairs = [(probs, truth_isup), ...]"""
    return [
        {"extra": {"per_grade_probs": list(p), "ground_truth_isup": int(y)}}
        for p, y in pairs
    ]


def test_lac_calibration_returns_a_calibration_per_alpha():
    rows = _fake_cal_rows(
        [
            ([0.7, 0.1, 0.05, 0.05, 0.05, 0.05], 0),
            ([0.1, 0.7, 0.05, 0.05, 0.05, 0.05], 1),
            ([0.1, 0.1, 0.7, 0.05, 0.03, 0.02], 2),
            ([0.1, 0.1, 0.1, 0.6, 0.05, 0.05], 3),
            ([0.05, 0.05, 0.1, 0.1, 0.6, 0.1], 4),
            ([0.05, 0.05, 0.1, 0.1, 0.1, 0.6], 5),
            ([0.4, 0.3, 0.1, 0.1, 0.05, 0.05], 0),
            ([0.3, 0.4, 0.1, 0.1, 0.05, 0.05], 1),
            ([0.2, 0.2, 0.4, 0.1, 0.05, 0.05], 2),
            ([0.1, 0.1, 0.1, 0.4, 0.2, 0.1], 3),
        ]
    )
    cals = calibrate_lac(rows, alphas=(0.10, 0.20))
    assert set(cals.keys()) == {0.10, 0.20}
    # Larger alpha → smaller quantile (less coverage, tighter set).
    assert cals[0.20].q_alpha <= cals[0.10].q_alpha
    assert cals[0.10].calibration_source == "pilot-50"


def test_conformal_singleton_is_supported():
    # Force calibration where q_alpha is small (0.4). A test point with a
    # single probability >= 0.6 (so score = 0.4) is on the boundary of the
    # set; make it clearly under: top prob 0.9 → score 0.1.
    cal = ConformalCalibration(
        alpha=0.10, q_alpha=0.4, n_calibration=100, calibration_source="test",
        per_alpha_q={"alpha=0.10": 0.4},
    )
    v, j = ConformalGate(cal).gate(_claim([0.02, 0.02, 0.02, 0.9, 0.02, 0.02]))
    assert v == "SUPPORTED"
    assert j.thresholds["prediction_set_size"] == 1
    assert j.gate_rule_id.startswith("split-conformal-lac-alpha0.10")


def test_conformal_multiclass_set_is_contested():
    cal = ConformalCalibration(
        alpha=0.10, q_alpha=0.6, n_calibration=100, calibration_source="test",
        per_alpha_q={"alpha=0.10": 0.6},
    )
    # Two classes at 0.4 each → scores 0.6, 0.6 both <= q → set size 2.
    v, j = ConformalGate(cal).gate(_claim([0.4, 0.4, 0.05, 0.05, 0.05, 0.05]))
    assert v == "CONTESTED"
    assert j.thresholds["prediction_set_size"] >= 2


def test_conformal_empty_set_is_unsupported():
    # Very restrictive q_alpha → nothing qualifies.
    cal = ConformalCalibration(
        alpha=0.01, q_alpha=0.05, n_calibration=100, calibration_source="test",
        per_alpha_q={"alpha=0.01": 0.05},
    )
    v, _ = ConformalGate(cal).gate(_claim([0.3, 0.2, 0.2, 0.1, 0.1, 0.1]))
    assert v == "UNSUPPORTED"


def test_conformal_uniform_is_not_applicable():
    cal = ConformalCalibration(
        alpha=0.10, q_alpha=0.5, n_calibration=50, calibration_source="test",
        per_alpha_q={"alpha=0.10": 0.5},
    )
    v, _ = ConformalGate(cal).gate(_claim([1 / 6] * 6))
    assert v == "NOT_APPLICABLE"


def test_all_verdicts_serialize_to_ledger_row():
    import json
    cal = ConformalCalibration(
        alpha=0.10, q_alpha=0.4, n_calibration=50, calibration_source="pilot-50",
        per_alpha_q={"alpha=0.10": 0.4},
    )
    for gate in [EntropyGate(), MarginGate(), ConformalGate(cal)]:
        v, j = gate.gate(_claim([0.02, 0.02, 0.02, 0.9, 0.02, 0.02]))
        row = {
            "verdict": v,
            "gate_rule_id": j.gate_rule_id,
            "source_rows": list(j.source_rows),
            "thresholds": {k: (list(v) if isinstance(v, list) else v)
                           for k, v in j.thresholds.items()},
        }
        reloaded = json.loads(json.dumps(row))
        assert reloaded["verdict"] in {"SUPPORTED", "UNSUPPORTED",
                                      "NOT_APPLICABLE", "CONTESTED"}
        assert reloaded["gate_rule_id"] == j.gate_rule_id


def test_entropy_matches_manual_computation():
    probs = [0.5, 0.3, 0.1, 0.05, 0.03, 0.02]
    expected = -sum(p * math.log(p) for p in probs)
    _, j = EntropyGate().gate(_claim(probs))
    assert abs(j.thresholds["observed_entropy"] - expected) < 1e-9


def test_margin_matches_manual_computation():
    probs = [0.4, 0.35, 0.1, 0.05, 0.05, 0.05]
    _, j = MarginGate().gate(_claim(probs))
    assert abs(j.thresholds["observed_margin"] - (0.4 - 0.35)) < 1e-9
