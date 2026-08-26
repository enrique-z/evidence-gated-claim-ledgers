"""Gate-oneslide runner: composes the 6 P5 modules for one real PANDA WSI.

Explicit anti-fabrication contract:
- Every number in the emitted report and ledger is derived from a real
  computation on the real WSI. Nothing is synthesized.
- If the MIL classifier version string starts with "untrained-", the runner
  labels the report and the ledger entry with a large "TRACER — untrained
  model — no scientific claim" banner and forces the ledger's `claim_text`
  to be an infrastructure statement, not a disease-presence statement.
- The runner refuses to run any cohort other than "gate-oneslide" with an
  untrained MIL head. Later cohorts (pilot-50, expansion-500) must be
  invoked with a trained head whose version string starts with
  "trained@<training-run-id>".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from p5.claim_ledger.ledger import ClaimLedger, LedgerRow
from p5.evidence_gate.gate import Claim, EvidenceGate


@dataclass(frozen=True)
class RunReport:
    slide_id: str
    cohort_id: str
    patch_count: int
    backbone_version: str
    mil_version: str
    isup_prediction: int
    confidence: float
    verdict: str
    gate_rule_id: str
    ledger_path: str
    report_path: str
    is_tracer: bool


class GateOneSlideRunner:
    """Orchestrator for the Aug 15 gate-oneslide tracer.

    Wire order: Data Loader (external) → Patch Extractor → Backbone Embedder
    (with Feature Cache in front) → MIL Classifier → Evidence Gate → Claim
    Ledger → Report.
    """

    COHORT_ID = "gate-oneslide"

    def __init__(
        self,
        patch_extractor,
        backbone_embedder,
        feature_cache,
        mil_classifier,
        evidence_gate: EvidenceGate,
        claim_ledger: ClaimLedger,
        cost_meter=None,
    ):
        self.patch_extractor = patch_extractor
        self.backbone_embedder = backbone_embedder
        self.feature_cache = feature_cache
        self.mil_classifier = mil_classifier
        self.evidence_gate = evidence_gate
        self.ledger = claim_ledger
        self.cost_meter = cost_meter

    def run(
        self,
        slide_id: str,
        wsi_path: Path,
        report_dir: Path,
        max_patches: Optional[int] = 200,
    ) -> RunReport:
        """Run gate-oneslide end-to-end on one WSI. Return a typed RunReport."""
        mil_version = self.mil_classifier.VERSION
        is_tracer = mil_version.startswith("mean-pool-linear-untrained")

        # Refuse to run non-tracer cohorts with an untrained MIL head.
        if not is_tracer and self.COHORT_ID != "gate-oneslide":
            raise RuntimeError(
                f"Cohort {self.COHORT_ID} requires a trained MIL head; got version {mil_version}"
            )

        backbone_name = "phikon-v2"
        backbone_version = self.backbone_embedder.backbone_version

        # 1. Patch extraction (real WSI)
        patches = list(self.patch_extractor(
            wsi_path, slide_id=slide_id, max_patches=max_patches
        ))
        patch_count = len(patches)
        if patch_count == 0:
            raise RuntimeError(
                f"No tissue patches extracted from {wsi_path} — WSI may be all background"
            )

        # 2. Feature extraction (with cache)
        cached = self.feature_cache.get(slide_id, backbone_name, backbone_version)
        if cached is not None:
            features = cached
        else:
            features = self.backbone_embedder.embed(patches)
            self.feature_cache.put(slide_id, backbone_name, backbone_version, features)

        # 3. MIL prediction
        prediction = self.mil_classifier.predict(features)

        # 4. Wrap prediction as a Claim
        timestamp = datetime.now(timezone.utc).isoformat()
        claim_id = f"{slide_id}#{self.COHORT_ID}#{timestamp}"

        if is_tracer:
            claim_text = (
                f"[TRACER — untrained MIL, no scientific claim] slide={slide_id} "
                f"argmax=isup_{prediction.isup_grade} softmax_confidence={prediction.confidence:.4f}"
            )
        else:
            claim_text = (
                f"slide={slide_id} isup_prediction={prediction.isup_grade} "
                f"confidence={prediction.confidence:.4f}"
            )

        evidence_ref = f"file://{wsi_path}"
        claim = Claim(
            claim_id=claim_id,
            claim_text=claim_text,
            evidence_ref=evidence_ref,
            model_output={
                "isup_grade": prediction.isup_grade,
                "confidence": prediction.confidence,
                "per_grade_probs": prediction.per_grade_probs,
            },
            backbone_version=backbone_version,
            cohort_id=self.COHORT_ID,
        )

        # 5. Gate → verdict
        verdict, justification = self.evidence_gate.gate(claim)

        # 6. Ledger
        justification_dict = {
            "gate_rule_id": justification.gate_rule_id,
            "source_rows": list(justification.source_rows),
            "thresholds": dict(justification.thresholds),
            "mil_version": mil_version,
            "patch_count": patch_count,
            "is_tracer": is_tracer,
        }
        row = LedgerRow(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            evidence_ref=claim.evidence_ref,
            verdict=verdict,
            justification=justification_dict,
            timestamp=timestamp,
            backbone_version=backbone_version,
            cohort_id=self.COHORT_ID,
            gate_rule_id=justification.gate_rule_id,
            extra={
                "isup_prediction": prediction.isup_grade,
                "confidence": prediction.confidence,
                "per_grade_probs": prediction.per_grade_probs,
                "mil_version": mil_version,
            },
        )
        self.ledger.append(row)

        # 7. Report
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"gate_oneslide_{slide_id}.md"
        self._write_report(
            report_path,
            slide_id=slide_id,
            cohort_id=self.COHORT_ID,
            patch_count=patch_count,
            backbone_version=backbone_version,
            mil_version=mil_version,
            prediction=prediction,
            verdict=verdict,
            justification=justification_dict,
            timestamp=timestamp,
            is_tracer=is_tracer,
            wsi_path=str(wsi_path),
            claim_id=claim.claim_id,
        )

        return RunReport(
            slide_id=slide_id,
            cohort_id=self.COHORT_ID,
            patch_count=patch_count,
            backbone_version=backbone_version,
            mil_version=mil_version,
            isup_prediction=prediction.isup_grade,
            confidence=prediction.confidence,
            verdict=verdict,
            gate_rule_id=justification.gate_rule_id,
            ledger_path=str(self.ledger.path),
            report_path=str(report_path),
            is_tracer=is_tracer,
        )

    def _write_report(self, path: Path, **kw) -> None:
        is_tracer = kw["is_tracer"]
        pred = kw["prediction"]
        lines = [
            "# gate-oneslide report",
            "",
            f"- **generated:** {kw['timestamp']}",
            f"- **cohort_id:** `{kw['cohort_id']}`",
            f"- **slide_id:** `{kw['slide_id']}`",
            f"- **wsi_path:** `{kw['wsi_path']}`",
            f"- **claim_id:** `{kw['claim_id']}`",
            "",
        ]
        if is_tracer:
            lines += [
                "> **⚠️ TRACER RUN — UNTRAINED MIL HEAD**",
                ">",
                "> This report was produced by a **random-init** MIL classifier. The ISUP",
                "> prediction and confidence are model-calibration artefacts, NOT scientific",
                "> claims about disease presence in this slide. Trained-model verdicts are",
                "> produced only from the `pilot-50` (issue #29) and `expansion-500` (issue #32)",
                "> cohorts, where the MIL head has been trained. Do NOT cite this report's",
                "> ISUP prediction, confidence, or verdict as evidence about the slide.",
                "",
                "> The verdict below reflects the gate's honest response to a no-signal input,",
                "> which is what the anti-fabrication regime requires.",
                "",
            ]
        lines += [
            "## Pipeline modules (versions pinned)",
            "",
            f"- **backbone_version:** `{kw['backbone_version']}`",
            f"- **mil_version:** `{kw['mil_version']}`",
            f"- **gate_rule_id:** `{kw['justification']['gate_rule_id']}`",
            "",
            "## Extraction",
            "",
            f"- **patch_count:** {kw['patch_count']}",
            "",
            "## Prediction",
            "",
            f"- **isup_prediction:** `isup_{pred.isup_grade}`",
            f"- **confidence:** `{pred.confidence:.6f}`",
            f"- **per_grade_probs:** `{[f'{p:.4f}' for p in pred.per_grade_probs]}`",
            "",
            "## Verdict",
            "",
            f"- **verdict:** `{kw['verdict']}`",
            f"- **thresholds:** `{kw['justification']['thresholds']}`",
            "",
            "## Ledger",
            "",
            "This run appended one row to the P5 claim ledger.",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
