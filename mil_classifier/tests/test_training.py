"""Contract tests for the CV trainer.

Does not test accuracy (untrained data). Tests shape, honesty (val-fold slides
never seen during their own training), and version-string format.
"""

import pytest

torch = pytest.importorskip("torch")

from p5.mil_classifier.training import TrainingConfig, train_and_predict_cv


def _fake_bag(n=8, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, 1024)


def test_returns_prediction_per_slide():
    n_slides = 15
    slide_ids = [f"s{i}" for i in range(n_slides)]
    features = [_fake_bag(seed=i) for i in range(n_slides)]
    labels = [i % 6 for i in range(n_slides)]
    preds, version = train_and_predict_cv(
        slide_ids, features, labels, n_folds=5,
        cfg=TrainingConfig(epochs=2), device="cpu", run_id="test",
    )
    assert len(preds) == n_slides
    assert all(p is not None for p in preds)
    for grade, conf, probs in preds:
        assert grade in range(6)
        assert 0.0 <= conf <= 1.0
        assert len(probs) == 6
        assert abs(sum(probs) - 1.0) < 1e-4


def test_version_string_marks_trained():
    slide_ids = [f"s{i}" for i in range(10)]
    features = [_fake_bag(n=4, seed=i) for i in range(10)]
    labels = [0] * 10
    _, version = train_and_predict_cv(
        slide_ids, features, labels, n_folds=5,
        cfg=TrainingConfig(epochs=1), device="cpu", run_id="pilot-50",
    )
    assert version.startswith("trained@pilot-50-cv5-seed")


def test_deterministic_across_runs():
    slide_ids = [f"s{i}" for i in range(10)]
    features = [_fake_bag(seed=i) for i in range(10)]
    labels = [i % 6 for i in range(10)]
    p1, v1 = train_and_predict_cv(
        slide_ids, features, labels, n_folds=5,
        cfg=TrainingConfig(epochs=3, seed=42), device="cpu", run_id="test",
    )
    p2, v2 = train_and_predict_cv(
        slide_ids, features, labels, n_folds=5,
        cfg=TrainingConfig(epochs=3, seed=42), device="cpu", run_id="test",
    )
    assert v1 == v2
    for (g1, c1, _), (g2, c2, _) in zip(p1, p2):
        assert g1 == g2
        assert abs(c1 - c2) < 1e-5


def test_val_slide_never_seen_in_its_own_training():
    """Honesty check: each slide's prediction comes from a head trained on the
    OTHER folds. If it didn't, predictions would be perfect and confidence 1.0.
    """
    slide_ids = [f"s{i}" for i in range(20)]
    features = [_fake_bag(seed=i) for i in range(20)]
    # Adversarial labels: random noise → an honest CV predictor cannot fit
    torch.manual_seed(0)
    labels = torch.randint(0, 6, (20,)).tolist()
    preds, _ = train_and_predict_cv(
        slide_ids, features, labels, n_folds=5,
        cfg=TrainingConfig(epochs=20), device="cpu", run_id="test",
    )
    # Expect at most modest agreement, NOT perfect fit
    correct = sum(1 for i, (g, _, _) in enumerate(preds) if g == labels[i])
    assert correct < len(preds), (
        f"CV predictions perfectly matched random labels ({correct}/{len(preds)}) — "
        "val slides are leaking into their own training folds"
    )
