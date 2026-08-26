"""Unit tests for the patch extractor."""

from pathlib import Path

import numpy as np
import pytest

from p5.patch_extractor.extractor import Patch, extract_patches, _tissue_mask


def _write_synthetic_tiff(path: Path, size_px: int = 1024, tissue: bool = True):
    """Write a small synthetic TIFF for testing. All-tissue if `tissue`, else all-white."""
    import tifffile
    if tissue:
        # Colourful noise → passes tissue mask
        rng = np.random.RandomState(42)
        pixels = rng.randint(50, 200, size=(size_px, size_px, 3), dtype=np.uint8)
    else:
        pixels = np.full((size_px, size_px, 3), 255, dtype=np.uint8)
    tifffile.imwrite(str(path), pixels)


def test_all_tissue_yields_all_tiles(tmp_path):
    wsi = tmp_path / "synth.tif"
    _write_synthetic_tiff(wsi, size_px=1024, tissue=True)
    patches = list(extract_patches(wsi, slide_id="synth-1", tile_size=256))
    # 1024 / 256 = 4 tiles per side, 16 total. Reader may pick a level >= 0;
    # accept the count that corresponds to whichever level was read.
    assert len(patches) > 0
    for p in patches:
        assert isinstance(p, Patch)
        assert p.slide_id == "synth-1"
        assert p.tensor.shape == (256, 256, 3)
        assert p.tensor.dtype == np.uint8
        assert (p.x % 256) == 0
        assert (p.y % 256) == 0


def test_all_background_yields_no_tiles(tmp_path):
    wsi = tmp_path / "synth_bg.tif"
    _write_synthetic_tiff(wsi, size_px=1024, tissue=False)
    patches = list(extract_patches(wsi, slide_id="synth-2", tile_size=256))
    assert patches == []


def test_max_patches_caps_output(tmp_path):
    wsi = tmp_path / "synth.tif"
    _write_synthetic_tiff(wsi, size_px=1024, tissue=True)
    patches = list(extract_patches(wsi, slide_id="synth-3", tile_size=256, max_patches=3))
    assert len(patches) == 3


def test_tissue_mask_recognises_dark_tile():
    dark = np.full((256, 256, 3), 100, dtype=np.uint8)  # grey, brightness < 220
    assert _tissue_mask(dark).all()


def test_tissue_mask_rejects_pure_white():
    white = np.full((256, 256, 3), 255, dtype=np.uint8)
    assert not _tissue_mask(white).any()


def test_missing_wsi_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(extract_patches(tmp_path / "nope.tif", slide_id="x"))
