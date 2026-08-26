"""Composition tests for the GateOneSlideRunner.

Uses lightweight fake modules so the composition contract is verified without
loading Phikon-v2 or requiring a GPU. The REAL end-to-end run happens in the
Kaggle notebook (recorded under evidence/gate_oneslide/).
"""

from dataclasses import dataclass
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from p5.claim_ledger.ledger import ClaimLedger
from p5.evidence_gate.gate import EvidenceGate
from p5.feature_cache.cache import FeatureCache
from p5.orchestrator.runner import GateOneSlideRunner


class _FakeEmbedder:
    backbone_version = "phikon-v2-owkin@test"

    def embed(self, patches):
        n = len(list(patches))
        return torch.zeros((max(n, 1), 1024))


@dataclass
class _FakePred:
    isup_grade: int
    confidence: float
    per_grade_probs: list


class _UntrainedFakeMIL:
    VERSION = "mean-pool-linear-untrained-seed42@test"

    def predict(self, features):
        # Uniform output — mimics an untrained head
        return _FakePred(0, 1 / 6, [1 / 6] * 6)


class _HighConfFakeMIL:
    VERSION = "trained@fake-run-id"

    def predict(self, features):
        return _FakePred(3, 0.9, [0.02, 0.02, 0.02, 0.9, 0.02, 0.02])


def _fake_patch_extractor(wsi_path, slide_id, max_patches=None):
    # Yield 3 fake patch objects; runner only counts them and passes to embedder
    for i in range(3):
        yield type("P", (), {"slide_id": slide_id, "x": i, "y": 0, "level": 0, "tensor": None})


def test_untrained_head_produces_not_applicable(tmp_path):
    cache = FeatureCache(tmp_path / "cache")
    ledger = ClaimLedger(tmp_path / "ledger.jsonl")
    runner = GateOneSlideRunner(
        patch_extractor=_fake_patch_extractor,
        backbone_embedder=_FakeEmbedder(),
        feature_cache=cache,
        mil_classifier=_UntrainedFakeMIL(),
        evidence_gate=EvidenceGate(),
        claim_ledger=ledger,
    )
    report = runner.run(
        slide_id="fake-slide-1",
        wsi_path=Path("/nonexistent.tiff"),
        report_dir=tmp_path / "reports",
    )
    assert report.verdict == "NOT_APPLICABLE"
    assert report.is_tracer is True
    assert Path(report.report_path).exists()
    # Ledger row landed with the tracer marker
    rows = list(ledger.rows())
    assert len(rows) == 1
    assert rows[0].justification["is_tracer"] is True
    assert "[TRACER" in rows[0].claim_text


def test_high_confidence_trained_head_produces_supported(tmp_path):
    cache = FeatureCache(tmp_path / "cache")
    ledger = ClaimLedger(tmp_path / "ledger.jsonl")
    runner = GateOneSlideRunner(
        patch_extractor=_fake_patch_extractor,
        backbone_embedder=_FakeEmbedder(),
        feature_cache=cache,
        mil_classifier=_HighConfFakeMIL(),
        evidence_gate=EvidenceGate(),
        claim_ledger=ledger,
    )
    report = runner.run(
        slide_id="fake-slide-2",
        wsi_path=Path("/nonexistent.tiff"),
        report_dir=tmp_path / "reports",
    )
    assert report.verdict == "SUPPORTED"
    assert report.is_tracer is False
    rows = list(ledger.rows())
    assert rows[0].verdict == "SUPPORTED"
    assert "[TRACER" not in rows[0].claim_text


def test_feature_cache_hit_skips_embed(tmp_path):
    cache = FeatureCache(tmp_path / "cache")
    ledger = ClaimLedger(tmp_path / "ledger.jsonl")

    calls = {"embed": 0}

    class _CountingEmbedder(_FakeEmbedder):
        def embed(self, patches):
            calls["embed"] += 1
            return super().embed(patches)

    embedder = _CountingEmbedder()
    runner = GateOneSlideRunner(
        patch_extractor=_fake_patch_extractor,
        backbone_embedder=embedder,
        feature_cache=cache,
        mil_classifier=_UntrainedFakeMIL(),
        evidence_gate=EvidenceGate(),
        claim_ledger=ledger,
    )
    runner.run("s", Path("/x.tiff"), report_dir=tmp_path / "r1")
    assert calls["embed"] == 1
    runner.run("s", Path("/x.tiff"), report_dir=tmp_path / "r2")
    # Second run must hit cache — no additional embed
    assert calls["embed"] == 1
