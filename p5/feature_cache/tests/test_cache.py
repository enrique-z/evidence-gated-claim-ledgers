"""Tests for the Feature Cache."""

import numpy as np

from p5.feature_cache.cache import FeatureCache


def test_get_after_put_round_trips_bytes(tmp_path):
    cache = FeatureCache(tmp_path)
    features = np.random.RandomState(0).rand(32, 1024).astype(np.float32)
    cache.put("slide-a", "phikon-v2", "v1@test", features)
    out = cache.get("slide-a", "phikon-v2", "v1@test")
    assert out is not None
    np.testing.assert_array_equal(out, features)


def test_get_unknown_key_returns_none(tmp_path):
    cache = FeatureCache(tmp_path)
    assert cache.get("nope", "phikon-v2", "v1@test") is None


def test_has_reflects_presence(tmp_path):
    cache = FeatureCache(tmp_path)
    assert not cache.has("s", "b", "v")
    cache.put("s", "b", "v", np.zeros((1, 4), dtype=np.float32))
    assert cache.has("s", "b", "v")


def test_keys_are_isolated_by_backbone_version(tmp_path):
    cache = FeatureCache(tmp_path)
    f1 = np.ones((2, 4), dtype=np.float32)
    f2 = np.zeros((2, 4), dtype=np.float32)
    cache.put("s", "phikon-v2", "v1", f1)
    cache.put("s", "phikon-v2", "v2", f2)
    np.testing.assert_array_equal(cache.get("s", "phikon-v2", "v1"), f1)
    np.testing.assert_array_equal(cache.get("s", "phikon-v2", "v2"), f2)


def test_version_string_with_special_chars_is_safe(tmp_path):
    cache = FeatureCache(tmp_path)
    version = "phikon-v2-owkin@2026-08-05"
    cache.put("s", "phikon-v2", version, np.ones((1, 4), dtype=np.float32))
    assert cache.has("s", "phikon-v2", version)
