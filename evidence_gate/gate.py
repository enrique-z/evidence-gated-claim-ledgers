"""Evidence Gate — Claim + model output → Verdict.

The paper's headline contribution. A per-slide MIL prediction becomes a
`Claim`; the gate returns a typed `Verdict` in the exact enum from PRD #22:

    SUPPORTED | UNSUPPORTED | NOT_APPLICABLE | CONTESTED

Gate rules for the tracer are threshold-based on the softmax confidence of
the model's argmax class:

- confidence >= high_threshold  → SUPPORTED
- confidence <= low_threshold   → UNSUPPORTED
- otherwise                     → CONTESTED
- if the prediction is degenerate (no patches / uniform probs) → NOT_APPLICABLE

For the gate-oneslide tracer with a random-init model, confidence will be
~0.17 (uniform over 6 classes), so verdicts will (correctly) be UNSUPPORTED
until a trained head plugs in.

The rule id is stamped on every Justification so ledger rows are traceable
to the exact gate rule that produced them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

Verdict = Literal["SUPPORTED", "UNSUPPORTED", "NOT_APPLICABLE", "CONTESTED"]


@dataclass(frozen=True)
class Claim:
    claim_id: str
    claim_text: str
    evidence_ref: str
    model_output: dict
    backbone_version: str
    cohort_id: str


@dataclass(frozen=True)
class Justification:
    gate_rule_id: str
    source_rows: List[str]
    thresholds: Dict[str, float]


class EvidenceGate:
    """Threshold-confidence gate for the P5 (ASCI) gate-oneslide tracer."""

    GATE_RULE_ID = "confidence-threshold-v1@2026-08-05"

    def __init__(self, high_threshold: float = 0.70, low_threshold: float = 0.30):
        if not 0.0 < low_threshold <= high_threshold <= 1.0:
            raise ValueError(
                f"Thresholds must satisfy 0 < low <= high <= 1, got low={low_threshold} high={high_threshold}"
            )
        self.high_threshold = float(high_threshold)
        self.low_threshold = float(low_threshold)

    def gate(self, claim: Claim) -> tuple[Verdict, Justification]:
        """Return (Verdict, Justification) for one claim."""
        model_output = claim.model_output
        probs = model_output.get("per_grade_probs")
        if not probs or all(abs(p - probs[0]) < 1e-6 for p in probs):
            # Degenerate / no-signal — cannot make a claim.
            return (
                "NOT_APPLICABLE",
                Justification(
                    gate_rule_id=self.GATE_RULE_ID,
                    source_rows=[claim.evidence_ref],
                    thresholds={"high": self.high_threshold, "low": self.low_threshold},
                ),
            )

        confidence = float(model_output.get("confidence", max(probs)))

        if confidence >= self.high_threshold:
            verdict: Verdict = "SUPPORTED"
        elif confidence <= self.low_threshold:
            verdict = "UNSUPPORTED"
        else:
            verdict = "CONTESTED"

        return (
            verdict,
            Justification(
                gate_rule_id=self.GATE_RULE_ID,
                source_rows=[claim.evidence_ref],
                thresholds={
                    "high": self.high_threshold,
                    "low": self.low_threshold,
                    "observed_confidence": confidence,
                },
            ),
        )
