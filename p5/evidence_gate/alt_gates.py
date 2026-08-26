"""Alternative Evidence-Gate rules for Phase A2.

Each rule implements the same `gate(Claim) -> (Verdict, Justification)`
contract as `p5.evidence_gate.gate.EvidenceGate`, so they are drop-in
replacements at the pipeline seam. Every rule has a versioned
`GATE_RULE_ID` so ledger rows pin the exact rule that produced them.

Three rules:

  EntropyGate    — Shannon entropy of the softmax vector, in nats.
                   H(p) <= entropy_low  → SUPPORTED  (low entropy = confident)
                   H(p) >= entropy_high → UNSUPPORTED
                   otherwise             → CONTESTED
                   Degenerate (uniform / empty) → NOT_APPLICABLE.

  MarginGate     — top1 - top2 gap of the softmax vector.
                   gap >= margin_high → SUPPORTED
                   gap <= margin_low  → UNSUPPORTED
                   otherwise           → CONTESTED
                   Degenerate → NOT_APPLICABLE.

  ConformalGate  — Split-conformal prediction set at level α (LAC score
                   s(x,y) = 1 - p_y). Calibrated on a separate held-out
                   split (pilot-50 by default).
                   singleton set → SUPPORTED
                   empty set     → UNSUPPORTED
                   |set| ≥ 2      → CONTESTED
                   Reference: Sadinle, Lei, Wasserman (2019); Romano,
                   Sesia, Candès (2020) for the class-conditional variant.

Notes on defaults
-----------------
The entropy and margin thresholds are picked so the induced verdict
distribution is qualitatively comparable to the confidence-threshold
default (0.70 / 0.30) on a 6-class problem, so head-to-head comparison
in Table 1 and Fig 2 is meaningful without a per-rule threshold sweep.
The full risk-coverage sweep in `p5/analysis/risk_coverage.py` (Phase A3)
characterises the family of operating points for each rule and does not
depend on these defaults.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from p5.evidence_gate.gate import Claim, Justification, Verdict


def _degenerate(probs: list[float] | None) -> bool:
    if not probs:
        return True
    return all(abs(p - probs[0]) < 1e-6 for p in probs)


def _entropy_nats(probs: Iterable[float]) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0)


def _top1_top2(probs: list[float]) -> tuple[int, float, float]:
    idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    top1_i = idx[0]
    top1 = float(probs[top1_i])
    top2 = float(probs[idx[1]]) if len(idx) > 1 else 0.0
    return top1_i, top1, top2


class EntropyGate:
    GATE_RULE_ID = "entropy-threshold-v1@2026-08-11"

    def __init__(self, entropy_low: float = 1.10, entropy_high: float = 1.65):
        if not 0.0 <= entropy_low <= entropy_high:
            raise ValueError(
                f"Require 0 <= entropy_low <= entropy_high, got low={entropy_low}"
                f" high={entropy_high}"
            )
        self.entropy_low = float(entropy_low)
        self.entropy_high = float(entropy_high)

    def gate(self, claim: Claim) -> tuple[Verdict, Justification]:
        probs = claim.model_output.get("per_grade_probs")
        if _degenerate(probs):
            return (
                "NOT_APPLICABLE",
                Justification(
                    gate_rule_id=self.GATE_RULE_ID,
                    source_rows=[claim.evidence_ref],
                    thresholds={
                        "entropy_low": self.entropy_low,
                        "entropy_high": self.entropy_high,
                    },
                ),
            )
        H = _entropy_nats(probs)
        if H <= self.entropy_low:
            v: Verdict = "SUPPORTED"
        elif H >= self.entropy_high:
            v = "UNSUPPORTED"
        else:
            v = "CONTESTED"
        return (
            v,
            Justification(
                gate_rule_id=self.GATE_RULE_ID,
                source_rows=[claim.evidence_ref],
                thresholds={
                    "entropy_low": self.entropy_low,
                    "entropy_high": self.entropy_high,
                    "observed_entropy": H,
                },
            ),
        )


class MarginGate:
    GATE_RULE_ID = "margin-threshold-v1@2026-08-11"

    def __init__(self, margin_low: float = 0.05, margin_high: float = 0.55):
        if not 0.0 <= margin_low <= margin_high <= 1.0:
            raise ValueError(
                f"Require 0 <= margin_low <= margin_high <= 1, got low={margin_low}"
                f" high={margin_high}"
            )
        self.margin_low = float(margin_low)
        self.margin_high = float(margin_high)

    def gate(self, claim: Claim) -> tuple[Verdict, Justification]:
        probs = claim.model_output.get("per_grade_probs")
        if _degenerate(probs):
            return (
                "NOT_APPLICABLE",
                Justification(
                    gate_rule_id=self.GATE_RULE_ID,
                    source_rows=[claim.evidence_ref],
                    thresholds={
                        "margin_low": self.margin_low,
                        "margin_high": self.margin_high,
                    },
                ),
            )
        _, top1, top2 = _top1_top2(list(probs))
        gap = top1 - top2
        if gap >= self.margin_high:
            v: Verdict = "SUPPORTED"
        elif gap <= self.margin_low:
            v = "UNSUPPORTED"
        else:
            v = "CONTESTED"
        return (
            v,
            Justification(
                gate_rule_id=self.GATE_RULE_ID,
                source_rows=[claim.evidence_ref],
                thresholds={
                    "margin_low": self.margin_low,
                    "margin_high": self.margin_high,
                    "observed_margin": gap,
                },
            ),
        )


@dataclass(frozen=True)
class ConformalCalibration:
    """LAC (Least Ambiguous set-valued Classifier) calibration.

    Score: s(x,y) = 1 - p_y (probability of the *true* class).
    q_alpha = ceil((n+1)(1-alpha))/n-th empirical quantile of calibration
    scores.
    At test time, prediction set = {y : 1 - p_y <= q_alpha}.

    Reference: Sadinle, Lei, Wasserman (2019). "Least Ambiguous Set-Valued
    Classifiers with Bounded Error Levels." JASA.
    """

    alpha: float
    q_alpha: float
    n_calibration: int
    calibration_source: str
    per_alpha_q: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "q_alpha": self.q_alpha,
            "n_calibration": self.n_calibration,
            "calibration_source": self.calibration_source,
            "per_alpha_q": dict(self.per_alpha_q),
        }


def calibrate_lac(
    calibration_rows: list[dict],
    alphas: Iterable[float] = (0.05, 0.10, 0.20),
    calibration_source: str = "pilot-50",
) -> dict[float, ConformalCalibration]:
    """Compute LAC quantile q_alpha for each alpha from calibration ledger rows.

    Each row must have `extra.per_grade_probs` and `extra.ground_truth_isup`.
    """
    scores: list[float] = []
    for r in calibration_rows:
        probs = r["extra"]["per_grade_probs"]
        y = int(r["extra"]["ground_truth_isup"])
        scores.append(1.0 - float(probs[y]))
    scores.sort()
    n = len(scores)
    if n == 0:
        raise ValueError("empty calibration set")

    out: dict[float, ConformalCalibration] = {}
    per_alpha_q: dict[str, float] = {}
    for alpha in alphas:
        # ceil((n+1)(1-alpha)) / n quantile (Vovk correction), clamped to [1, n].
        k = math.ceil((n + 1) * (1.0 - alpha))
        k = max(1, min(n, k))
        q = scores[k - 1]  # 1-indexed k -> 0-indexed k-1
        per_alpha_q[f"alpha={alpha:.2f}"] = q

    for alpha in alphas:
        out[float(alpha)] = ConformalCalibration(
            alpha=float(alpha),
            q_alpha=per_alpha_q[f"alpha={alpha:.2f}"],
            n_calibration=n,
            calibration_source=calibration_source,
            per_alpha_q=per_alpha_q,
        )
    return out


class ConformalGate:
    """Split-conformal LAC gate with pre-computed calibration.

    Emits SUPPORTED iff the prediction set at level alpha is a singleton
    (the classifier is certain about a specific class at the specified
    coverage level), UNSUPPORTED iff the set is empty (no class is
    plausible at that level — this happens when the top score exceeds
    q_alpha), and CONTESTED for sets of size >= 2.
    """

    def __init__(self, calibration: ConformalCalibration):
        self.calibration = calibration
        self.GATE_RULE_ID = (
            f"split-conformal-lac-alpha{calibration.alpha:.2f}"
            f"-v1@2026-08-11"
        )

    def gate(self, claim: Claim) -> tuple[Verdict, Justification]:
        probs = claim.model_output.get("per_grade_probs")
        if _degenerate(probs):
            return (
                "NOT_APPLICABLE",
                Justification(
                    gate_rule_id=self.GATE_RULE_ID,
                    source_rows=[claim.evidence_ref],
                    thresholds={
                        "alpha": self.calibration.alpha,
                        "q_alpha": self.calibration.q_alpha,
                        "n_calibration": self.calibration.n_calibration,
                    },
                ),
            )
        q = self.calibration.q_alpha
        pset = [i for i, p in enumerate(probs) if (1.0 - float(p)) <= q]
        if len(pset) == 1:
            v: Verdict = "SUPPORTED"
        elif len(pset) == 0:
            v = "UNSUPPORTED"
        else:
            v = "CONTESTED"
        return (
            v,
            Justification(
                gate_rule_id=self.GATE_RULE_ID,
                source_rows=[claim.evidence_ref],
                thresholds={
                    "alpha": self.calibration.alpha,
                    "q_alpha": self.calibration.q_alpha,
                    "n_calibration": self.calibration.n_calibration,
                    "prediction_set": pset,
                    "prediction_set_size": len(pset),
                },
            ),
        )
