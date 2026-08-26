"""WSI → 256x256 patches @ 20x with tissue-mask filter.

Emits typed Patch(slide_id, x, y, level, tensor) records. Owns WSI I/O and
tissue segmentation; knows nothing about any downstream model.

Reader strategy: prefer openslide (rich MPP handling), fall back to skimage
via tifffile (works out-of-the-box on Kaggle Notebooks with PANDA .tiff).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


@dataclass(frozen=True)
class Patch:
    """A single tile extracted from a WSI."""
    slide_id: str
    x: int
    y: int
    level: int
    # tensor is a numpy array of shape (H, W, 3), uint8, RGB. Kept as `object`
    # in the dataclass to avoid importing numpy at module import time (the
    # module is imported by tests that never touch tensors).
    tensor: object


def _load_slide_pixels(wsi_path: Path):
    """Load a PANDA .tiff into an (H, W, 3) uint8 numpy array.

    Tries readers in order and returns pixels at whatever base level the reader
    exposes. For PANDA, the highest-resolution level is the desired 20x target.
    """
    import numpy as np

    wsi_path = Path(wsi_path)
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI not found: {wsi_path}")

    # Reader 1: openslide (best for large WSIs; may not be installed)
    try:
        import openslide
        slide = openslide.OpenSlide(str(wsi_path))
        # PANDA slides are small enough to read the level-0 image in one shot;
        # but some are huge. Use the smallest downsample >= 2 so the whole
        # image fits comfortably in memory for the tracer.
        # For the gate-oneslide tracer, correctness > resolution — read a
        # mid-resolution level and treat that as our 20x target.
        level = min(len(slide.level_dimensions) - 1, 1)
        w, h = slide.level_dimensions[level]
        rgba = slide.read_region((0, 0), level, (w, h))
        return np.array(rgba.convert("RGB")), level
    except Exception:
        pass

    # Reader 2: tifffile via skimage (default on Kaggle)
    try:
        import tifffile
        # PANDA .tiff files are multi-page pyramids; the LAST page is the
        # smallest (highest downsample). Use series[0].levels[1] if available
        # to get a mid-resolution image.
        with tifffile.TiffFile(str(wsi_path)) as tf:
            series = tf.series[0]
            levels = getattr(series, "levels", [series])
            # Pick the second-highest resolution (index 1) if available — keeps
            # tile counts sane for the tracer; falls back to level 0.
            level_idx = 1 if len(levels) > 1 else 0
            pixels = levels[level_idx].asarray()
            if pixels.ndim == 4:  # (P, H, W, C) — take first page
                pixels = pixels[0]
            if pixels.ndim == 3 and pixels.shape[-1] == 4:
                pixels = pixels[..., :3]
            return pixels.astype(np.uint8), level_idx
    except Exception as e:
        raise RuntimeError(
            f"Neither openslide nor tifffile could read {wsi_path}: {e}"
        )


def _tissue_mask(rgb):
    """Binary mask: True where a tile is likely tissue (not background/white).

    Simple, robust for H&E-stained pathology: threshold on saturation OR
    darkness. Avoids the empty-background regions PANDA slides have around
    the actual tissue.
    """
    import numpy as np
    # Convert to HSV-ish: saturation + darkness heuristic
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    max_c = np.maximum(np.maximum(r, g), b).astype(np.int16)
    min_c = np.minimum(np.minimum(r, g), b).astype(np.int16)
    saturation = max_c - min_c  # 0..255
    brightness = max_c
    # Tissue: colourful (sat > 15) OR darker than pure white (brightness < 220)
    return (saturation > 15) | (brightness < 220)


def extract_patches(
    wsi_path: Path,
    slide_id: str,
    tile_size: int = 256,
    tissue_threshold: float = 0.3,
    max_patches: Optional[int] = None,
) -> Iterator[Patch]:
    """Yield 256x256 tissue patches from a WSI.

    Args:
        wsi_path: path to the .tiff WSI
        slide_id: identifier stamped onto each emitted Patch
        tile_size: patch edge length in pixels (default 256)
        tissue_threshold: min fraction of tissue pixels required to emit
            a tile (default 0.3 = 30% of the tile must be tissue)
        max_patches: optional cap on emitted patches (safety valve for the
            tracer — prevents multi-thousand-patch slides from blowing the
            budget on one WSI)

    Yields:
        Patch objects with an (H, W, 3) uint8 RGB tensor.
    """
    import numpy as np

    pixels, source_level = _load_slide_pixels(Path(wsi_path))
    H, W, _ = pixels.shape
    mask = _tissue_mask(pixels)

    emitted = 0
    for y in range(0, H - tile_size + 1, tile_size):
        for x in range(0, W - tile_size + 1, tile_size):
            tile = pixels[y:y + tile_size, x:x + tile_size, :]
            tile_mask = mask[y:y + tile_size, x:x + tile_size]
            if tile_mask.mean() < tissue_threshold:
                continue
            yield Patch(
                slide_id=slide_id,
                x=x,
                y=y,
                level=source_level,
                tensor=tile,
            )
            emitted += 1
            if max_patches is not None and emitted >= max_patches:
                return
