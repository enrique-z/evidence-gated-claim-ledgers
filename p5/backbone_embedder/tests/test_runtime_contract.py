"""Tests for the Mac runtime contract on the Phikon-v2 embedder.

The contract check runs BEFORE torch import, so these tests need no torch.
"""

import platform

import pytest

from p5.backbone_embedder.embedder import (
    REQUIRED_MAC_ENV,
    BackboneRuntimeError,
    _enforce_mac_runtime_contract,
)


@pytest.mark.skipif(platform.system() != "Darwin", reason="Mac-only contract")
def test_missing_env_vars_raise(monkeypatch):
    for k in REQUIRED_MAC_ENV:
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(BackboneRuntimeError) as e:
        _enforce_mac_runtime_contract()
    # The error must name every missing var so the fix is obvious
    for k in REQUIRED_MAC_ENV:
        assert k in str(e.value)


@pytest.mark.skipif(platform.system() != "Darwin", reason="Mac-only contract")
def test_correct_env_vars_pass(monkeypatch):
    for k, v in REQUIRED_MAC_ENV.items():
        monkeypatch.setenv(k, v)
    _enforce_mac_runtime_contract()  # must not raise


@pytest.mark.skipif(platform.system() != "Darwin", reason="Mac-only contract")
def test_wrong_value_raises(monkeypatch):
    for k, v in REQUIRED_MAC_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")  # wrong value
    with pytest.raises(BackboneRuntimeError):
        _enforce_mac_runtime_contract()
