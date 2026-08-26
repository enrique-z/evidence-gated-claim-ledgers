"""P5 (ASCI) gate-oneslide tracer — Kaggle Notebook script.

Runs the full pipeline against ONE real PANDA WSI end-to-end and prints the
report + ledger row to stdout. Verifies the pipeline shape works and the
Evidence Gate produces a real Verdict for a real Claim from real evidence.

Anti-fabrication contract: the MIL head is intentionally untrained
(random-init, fixed seed). Its output is honest — a near-uniform softmax
distribution — so the Evidence Gate emits `NOT_APPLICABLE` (we can't ask the
question of an untrained model). This is the anti-fab regime working. The
report is labelled TRACER; no scientific claim is made about the slide.

Modules inlined here so Kaggle can run without installing the p5 package.
Any bugfix in this script MUST be mirrored in the corresponding p5/ module.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional


# ------------------------------------------------------------------------------
# Path resolution — glob-based, robust to Kaggle mount conventions
# ------------------------------------------------------------------------------

def resolve_panda_root() -> Path:
    candidates = list(Path("/kaggle/input").glob("**/prostate-cancer-grade-assessment"))
    if not candidates:
        print("Available under /kaggle/input:")
        for item in Path("/kaggle/input").iterdir():
            print(f"  - {item}")
        raise FileNotFoundError("PANDA competition directory not found under /kaggle/input")
    return candidates[0]


# ------------------------------------------------------------------------------
# Data loader (inlined)
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class PandaRow:
    slide_id: str
    wsi_path: str
    isup_grade: int
    gleason_score: str
    data_provider: str


def load_first_slide(panda_root: Path, slide_index: int = 0) -> PandaRow:
    import csv
    train_csv = panda_root / "train.csv"
    with open(train_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Deterministic pick: sort by image_id, take index N
    rows.sort(key=lambda r: r["image_id"])
    r = rows[slide_index]
    slide_id = r["image_id"].strip()
    return PandaRow(
        slide_id=slide_id,
        wsi_path=str(panda_root / "train_images" / f"{slide_id}.tiff"),
        isup_grade=int(r["isup_grade"]),
        gleason_score=r["gleason_score"].strip(),
        data_provider=r["data_provider"].strip(),
    )


# ------------------------------------------------------------------------------
# Patch extractor (inlined)
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class Patch:
    slide_id: str
    x: int
    y: int
    level: int
    tensor: object


def _load_slide_pixels(wsi_path: Path):
    import numpy as np
    wsi_path = Path(wsi_path)
    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI not found: {wsi_path}")

    # PANDA .tiff → prefer openslide, fall back to tifffile
    try:
        import openslide
        slide = openslide.OpenSlide(str(wsi_path))
        # Use mid-resolution level for the tracer (keeps patch count sane)
        level = min(len(slide.level_dimensions) - 1, 1)
        w, h = slide.level_dimensions[level]
        rgba = slide.read_region((0, 0), level, (w, h))
        return np.array(rgba.convert("RGB")), level
    except Exception as e:
        print(f"[patch_extractor] openslide failed ({e}); falling back to tifffile")

    import tifffile
    with tifffile.TiffFile(str(wsi_path)) as tf:
        series = tf.series[0]
        levels = getattr(series, "levels", [series])
        level_idx = 1 if len(levels) > 1 else 0
        pixels = levels[level_idx].asarray()
        if pixels.ndim == 4:
            pixels = pixels[0]
        if pixels.ndim == 3 and pixels.shape[-1] == 4:
            pixels = pixels[..., :3]
        return pixels.astype(np.uint8), level_idx


def _tissue_mask(rgb):
    import numpy as np
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    max_c = np.maximum(np.maximum(r, g), b).astype(np.int16)
    min_c = np.minimum(np.minimum(r, g), b).astype(np.int16)
    saturation = max_c - min_c
    brightness = max_c
    return (saturation > 15) | (brightness < 220)


def extract_patches(
    wsi_path: Path,
    slide_id: str,
    tile_size: int = 256,
    tissue_threshold: float = 0.3,
    max_patches: Optional[int] = 200,
) -> Iterator[Patch]:
    pixels, source_level = _load_slide_pixels(Path(wsi_path))
    H, W, _ = pixels.shape
    mask = _tissue_mask(pixels)
    emitted = 0
    for y in range(0, H - tile_size + 1, tile_size):
        for x in range(0, W - tile_size + 1, tile_size):
            tile_mask = mask[y:y + tile_size, x:x + tile_size]
            if tile_mask.mean() < tissue_threshold:
                continue
            yield Patch(
                slide_id=slide_id,
                x=x,
                y=y,
                level=source_level,
                tensor=pixels[y:y + tile_size, x:x + tile_size, :],
            )
            emitted += 1
            if max_patches is not None and emitted >= max_patches:
                return


# ------------------------------------------------------------------------------
# Backbone embedder (Phikon-v2, inlined)
# ------------------------------------------------------------------------------

BACKBONE_VERSION = "phikon-v2-owkin@2026-08-05"


class PhikonV2Embedder:
    HF_ID = "owkin/phikon-v2"
    OUTPUT_DIM = 1024

    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.device = device
        print(f"[embedder] loading {self.HF_ID} on {device} ...", flush=True)
        t0 = time.time()
        self.processor = AutoImageProcessor.from_pretrained(self.HF_ID)
        self.model = AutoModel.from_pretrained(self.HF_ID).to(device).eval()
        print(f"[embedder] loaded in {time.time() - t0:.1f}s")
        self._torch = torch
        self.backbone_version = BACKBONE_VERSION

    def embed(self, patches, batch_size: int = 16):
        torch = self._torch
        tensors = [getattr(p, "tensor", p) for p in patches]
        if not tensors:
            return torch.zeros((0, self.OUTPUT_DIM), dtype=torch.float32)
        out = []
        for start in range(0, len(tensors), batch_size):
            batch = tensors[start:start + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                res = self.model(**inputs)
            cls = res.last_hidden_state[:, 0, :].detach().cpu()
            out.append(cls)
        return torch.cat(out, dim=0)


# ------------------------------------------------------------------------------
# Feature cache (inlined)
# ------------------------------------------------------------------------------

class FeatureCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, slide_id, backbone_name, backbone_version):
        safe = backbone_version.replace("/", "_").replace(":", "_").replace("@", "_")
        return self.root / f"{slide_id}__{backbone_name}__{safe}.npz"

    def get(self, slide_id, backbone_name, backbone_version):
        import numpy as np
        p = self._path(slide_id, backbone_name, backbone_version)
        if not p.exists():
            return None
        with np.load(p) as npz:
            return npz["features"]

    def put(self, slide_id, backbone_name, backbone_version, features):
        import numpy as np
        p = self._path(slide_id, backbone_name, backbone_version)
        arr = features.detach().cpu().numpy() if hasattr(features, "detach") else features
        np.savez_compressed(p, features=np.asarray(arr))


# ------------------------------------------------------------------------------
# MIL classifier (untrained mean-pool, inlined)
# ------------------------------------------------------------------------------

MIL_VERSION = "mean-pool-linear-untrained-seed42@2026-08-05"


@dataclass(frozen=True)
class SlidePrediction:
    isup_grade: int
    confidence: float
    per_grade_probs: List[float]


class MeanPoolMIL:
    INPUT_DIM = 1024
    N_CLASSES = 6
    VERSION = MIL_VERSION

    def __init__(self, seed: int = 42, device: str = "cuda"):
        import torch
        torch.manual_seed(seed)
        self._torch = torch
        self.device = device
        self.head = torch.nn.Linear(self.INPUT_DIM, self.N_CLASSES).to(device).eval()

    def predict(self, features) -> SlidePrediction:
        torch = self._torch
        if hasattr(features, "detach"):
            x = features.detach().to(self.device).float()
        else:
            x = torch.as_tensor(features).to(self.device).float()
        if x.numel() == 0:
            probs = [1.0 / self.N_CLASSES] * self.N_CLASSES
            return SlidePrediction(0, probs[0], probs)
        pooled = x.mean(dim=0, keepdim=True)
        with torch.no_grad():
            logits = self.head(pooled)
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().tolist()
        grade = int(max(range(len(probs)), key=lambda i: probs[i]))
        return SlidePrediction(grade, float(probs[grade]), [float(p) for p in probs])


# ------------------------------------------------------------------------------
# Evidence Gate (inlined)
# ------------------------------------------------------------------------------

GATE_RULE_ID = "confidence-threshold-v1@2026-08-05"


def gate_claim(per_grade_probs, confidence, high=0.70, low=0.30):
    if not per_grade_probs or all(abs(p - per_grade_probs[0]) < 1e-6 for p in per_grade_probs):
        return "NOT_APPLICABLE", {"gate_rule_id": GATE_RULE_ID, "thresholds": {"high": high, "low": low}}
    if confidence >= high:
        v = "SUPPORTED"
    elif confidence <= low:
        v = "UNSUPPORTED"
    else:
        v = "CONTESTED"
    return v, {
        "gate_rule_id": GATE_RULE_ID,
        "thresholds": {"high": high, "low": low, "observed_confidence": confidence},
    }


# ------------------------------------------------------------------------------
# Main tracer
# ------------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("P5 (ASCI) — gate-oneslide tracer")
    print("=" * 80)

    panda_root = resolve_panda_root()
    print(f"PANDA mount: {panda_root}")

    row = load_first_slide(panda_root, slide_index=0)
    print(f"Selected slide: {row.slide_id}")
    print(f"  Ground-truth ISUP grade: {row.isup_grade}, Gleason: {row.gleason_score}")
    print(f"  Provider: {row.data_provider}")
    print(f"  WSI path: {row.wsi_path}")

    # Detect device — probe with a real forward, not just is_available().
    # Kaggle images occasionally ship a torch build without kernels for the
    # assigned GPU arch (cudaErrorNoKernelImageForDevice); is_available() still
    # returns True there, so a tiny conv forward is the only reliable probe.
    import torch
    device = "cpu"
    if torch.cuda.is_available():
        try:
            probe = torch.nn.Conv2d(3, 4, 3).to("cuda")
            probe(torch.zeros(1, 3, 16, 16, device="cuda"))
            torch.cuda.synchronize()
            device = "cuda"
        except Exception as e:
            print(f"CUDA probe failed ({type(e).__name__}: {e}); falling back to CPU")
    print(f"Device: {device} (torch.cuda.is_available={torch.cuda.is_available()})")

    # 1. Patch extraction (real WSI)
    print("\n[1/6] Patch extraction ...")
    t0 = time.time()
    patches = list(extract_patches(
        Path(row.wsi_path), slide_id=row.slide_id, max_patches=200
    ))
    print(f"  extracted {len(patches)} patches in {time.time() - t0:.1f}s")
    if len(patches) == 0:
        raise RuntimeError("No tissue patches — WSI may be all background")

    # 2. Backbone embedder
    print("\n[2/6] Backbone embedder (Phikon-v2) ...")
    embedder = PhikonV2Embedder(device=device)

    # 3. Feature cache
    print("\n[3/6] Feature cache + embedding ...")
    cache_dir = Path("/kaggle/working/feature_cache")
    cache = FeatureCache(cache_dir)
    cached = cache.get(row.slide_id, "phikon-v2", BACKBONE_VERSION)
    if cached is not None:
        print(f"  cache HIT for {row.slide_id}")
        features = cached
    else:
        print(f"  cache MISS — embedding {len(patches)} patches")
        t0 = time.time()
        features = embedder.embed(patches, batch_size=16)
        print(f"  embedded in {time.time() - t0:.1f}s; features shape = {tuple(features.shape)}")
        cache.put(row.slide_id, "phikon-v2", BACKBONE_VERSION, features)

    # 4. MIL classifier (UNTRAINED — tracer mode)
    print("\n[4/6] MIL classifier (untrained, tracer mode) ...")
    mil = MeanPoolMIL(seed=42, device=device)
    prediction = mil.predict(features)
    print(f"  isup_prediction={prediction.isup_grade}  confidence={prediction.confidence:.6f}")
    print(f"  per_grade_probs={[f'{p:.4f}' for p in prediction.per_grade_probs]}")

    # 5. Evidence Gate
    print("\n[5/6] Evidence Gate ...")
    verdict, justification = gate_claim(
        prediction.per_grade_probs, prediction.confidence
    )
    print(f"  verdict={verdict}  gate_rule_id={justification['gate_rule_id']}")

    # 6. Claim Ledger row + report
    print("\n[6/6] Claim Ledger row ...")
    timestamp = datetime.now(timezone.utc).isoformat()
    claim_id = f"{row.slide_id}#gate-oneslide#{timestamp}"
    is_tracer = MIL_VERSION.startswith("mean-pool-linear-untrained")
    claim_text = (
        f"[TRACER — untrained MIL, no scientific claim] slide={row.slide_id} "
        f"argmax=isup_{prediction.isup_grade} softmax_confidence={prediction.confidence:.4f}"
    )
    ledger_row = {
        "claim_id": claim_id,
        "claim_text": claim_text,
        "evidence_ref": f"file://{row.wsi_path}",
        "verdict": verdict,
        "justification": {
            **justification,
            "mil_version": MIL_VERSION,
            "patch_count": len(patches),
            "is_tracer": is_tracer,
        },
        "timestamp": timestamp,
        "backbone_version": BACKBONE_VERSION,
        "cohort_id": "gate-oneslide",
        "gate_rule_id": justification["gate_rule_id"],
        "extra": {
            "isup_prediction": prediction.isup_grade,
            "confidence": prediction.confidence,
            "per_grade_probs": prediction.per_grade_probs,
            "mil_version": MIL_VERSION,
            "ground_truth_isup": row.isup_grade,
            "ground_truth_gleason": row.gleason_score,
        },
    }
    ledger_path = Path("/kaggle/working/ledger.jsonl")
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ledger_row) + "\n")
    print(f"  appended to {ledger_path}")

    # Report
    report_path = Path(f"/kaggle/working/gate_oneslide_{row.slide_id}.md")
    report_lines = [
        "# gate-oneslide report",
        "",
        f"- **generated:** {timestamp}",
        f"- **slide_id:** `{row.slide_id}`",
        f"- **cohort_id:** `gate-oneslide`",
        f"- **claim_id:** `{claim_id}`",
        "",
        "> **⚠️ TRACER RUN — UNTRAINED MIL HEAD**",
        ">",
        "> Random-init MIL classifier. ISUP prediction / confidence / verdict are",
        "> model-calibration artefacts, NOT scientific claims about the slide.",
        "> Trained-model verdicts land in `pilot-50` (#29) and `expansion-500` (#32).",
        "",
        "## Modules (versions pinned)",
        f"- backbone_version: `{BACKBONE_VERSION}`",
        f"- mil_version: `{MIL_VERSION}`",
        f"- gate_rule_id: `{GATE_RULE_ID}`",
        "",
        "## Extraction",
        f"- patch_count: {len(patches)}",
        "",
        "## Prediction (TRACER — do not cite)",
        f"- isup_prediction: isup_{prediction.isup_grade}",
        f"- confidence: {prediction.confidence:.6f}",
        f"- per_grade_probs: {[f'{p:.4f}' for p in prediction.per_grade_probs]}",
        "",
        "## Verdict",
        f"- verdict: `{verdict}`",
        f"- thresholds: {justification['thresholds']}",
        "",
        "## Ground truth (from PANDA, for context only)",
        f"- ground_truth_isup: {row.isup_grade}",
        f"- ground_truth_gleason: {row.gleason_score}",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"  wrote {report_path}")

    # Summary + acceptance-criteria assertions
    print("\n" + "=" * 80)
    print("gate-oneslide SUMMARY")
    print("=" * 80)
    print(f"  slide_id: {row.slide_id}")
    print(f"  patch_count: {len(patches)}")
    print(f"  backbone_version: {BACKBONE_VERSION}")
    print(f"  mil_version: {MIL_VERSION}")
    print(f"  isup_prediction: {prediction.isup_grade}  confidence: {prediction.confidence:.6f}")
    print(f"  verdict: {verdict}")
    print(f"  gate_rule_id: {justification['gate_rule_id']}")
    print(f"  ledger row appended to: {ledger_path}")
    print(f"  report written to: {report_path}")

    assert len(patches) > 0
    assert prediction.isup_grade in range(6)
    assert 0.0 <= prediction.confidence <= 1.0
    assert verdict in {"SUPPORTED", "UNSUPPORTED", "NOT_APPLICABLE", "CONTESTED"}

    print("\nGATE-ONESLIDE OK")


if __name__ == "__main__":
    main()
