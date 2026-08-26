"""MIL Classifier for the P5 (ASCI) pipeline."""

from .classifier import MEAN_POOL_MIL_VERSION, MeanPoolMIL, SlidePrediction

__all__ = ["MeanPoolMIL", "SlidePrediction", "MEAN_POOL_MIL_VERSION"]
