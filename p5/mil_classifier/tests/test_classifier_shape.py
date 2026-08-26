"""Contract tests for the MIL classifier — shape and version pinning only.

The head is intentionally untrained; accuracy is not tested. What we assert:
argmax is a valid ISUP grade (0..5), softmax sums to 1, and the version
string carries the `untrained` marker so downstream (the orchestrator) can
refuse to run non-tracer cohorts with it.
"""

import pytest

torch = pytest.importorskip("torch")

from p5.mil_classifier.classifier import MEAN_POOL_MIL_VERSION, MeanPoolMIL, SlidePrediction


def test_version_string_pins_untrained():
    assert MEAN_POOL_MIL_VERSION.startswith("mean-pool-linear-untrained")


def test_predict_returns_valid_shape():
    mil = MeanPoolMIL(seed=42, device="cpu")
    features = torch.zeros((10, 1024))
    p = mil.predict(features)
    assert isinstance(p, SlidePrediction)
    assert p.isup_grade in range(6)
    assert 0.0 <= p.confidence <= 1.0
    assert len(p.per_grade_probs) == 6
    assert abs(sum(p.per_grade_probs) - 1.0) < 1e-4


def test_empty_bag_is_degenerate_uniform():
    mil = MeanPoolMIL(seed=42, device="cpu")
    p = mil.predict(torch.zeros((0, 1024)))
    # per_grade_probs must be perfectly uniform → downstream gate emits NOT_APPLICABLE
    assert all(abs(x - p.per_grade_probs[0]) < 1e-6 for x in p.per_grade_probs)
