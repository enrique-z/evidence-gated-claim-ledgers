"""P5 (ASCI) expansion-500 — Phikon-v1 backbone — Kaggle Notebook script.

Second-backbone validation: runs the identical pipeline as expansion_500.py
(same 500 slides, same 5-fold CV, same gate thresholds, same MIL head) but
replaces the Phikon-v2 embedder (owkin/phikon-v2, 1024-dim) with Phikon-v1arge
(owkin/phikon, MIT-style license, no gate, 768-dim CLS, iBOT ViT-B/16).

Phikon-v1 is a general-purpose vision transformer trained by Meta AI on a
curated 142M-image dataset. Using it as a second backbone tests whether EGCL's
evidence-gating mechanism is truly backbone-agnostic — including for models not
specifically pre-trained on histopathology.

Produces:
  - /kaggle/working/expansion_500_phikonv1_ledger.jsonl  — 500 rows
  - /kaggle/working/expansion_500_phikonv1_report.md     — headline stats

Download and save outputs to evidence/expansion_500_hoptimus0/ in the repo.
Then run: python3 p5/analysis/cross_backbone_agreement.py

Anti-fab contract:
- Same cohort slice as expansion-500 (indices [51:551], seed=42).
- Same 5-fold CV, same gate thresholds {high=0.70, low=0.30}.
- ONLY change: backbone embedding model (Phikon-v1 instead of Phikon-v2).
- Every ledger row is pinned to backbone_version, mil_version, gate_rule_id.
- Do NOT modify DECISIONS.md — this run EXTENDS the claim vocabulary.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List


COHORT_ID = "expansion-500"
COHORT_SLICE = (51, 551)
COHORT_SIZE = COHORT_SLICE[1] - COHORT_SLICE[0]

BACKBONE_NAME = "phikon-v1"
BACKBONE_VERSION = "phikon-v1-owkin@2026-08-23"


# ------------------------------------------------------------------------------
# P100 CUDA fix — must run BEFORE any "import torch" in the process
# Kaggle sometimes assigns P100 (sm_60) which modern PyTorch dropped.
# We reinstall torch with CUDA 11.7 (last version with sm_60 support).
# ------------------------------------------------------------------------------

def _maybe_fix_pytorch_for_p100():
    """Reinstall PyTorch with sm_60 support if on P100. Runs before torch import."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            cap_str = parts[1].strip()  # e.g. "6.0"
            try:
                major = int(cap_str.split(".")[0])
            except ValueError:
                continue
            if major < 7:
                print(f"GPU: {name} (sm_{cap_str.replace('.','')}) — not supported by current PyTorch.")
                print("Reinstalling PyTorch with CUDA 11.7 (sm_60 support) ...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q",
                     "torch==2.0.1+cu117",
                     "--index-url", "https://download.pytorch.org/whl/cu117"],
                    timeout=600,
                )
                print("PyTorch reinstalled — sm_60 now supported.")
                return
        print("[cuda-fix] GPU is sm_70+ or no nvidia-smi — no reinstall needed.")
    except Exception as e:
        print(f"[cuda-fix] skipped: {e}")


# ------------------------------------------------------------------------------
# Deterministic RNG — identical to expansion_500.py (same cohort)
# ------------------------------------------------------------------------------

def _deterministic_shuffle(items, seed):
    result = items.copy()
    rng_state = seed
    for i in range(len(result) - 1, 0, -1):
        rng_state = (rng_state * 1103515245 + 12345) & 0x7fffffff
        j = rng_state % (i + 1)
        result[i], result[j] = result[j], result[i]
    return result


def resolve_panda_root() -> Path:
    candidates = list(Path("/kaggle/input").glob("**/prostate-cancer-grade-assessment"))
    if not candidates:
        raise FileNotFoundError("PANDA competition directory not found under /kaggle/input")
    return candidates[0]


# ------------------------------------------------------------------------------
# Data loader — identical to expansion_500.py
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class PandaRow:
    slide_id: str
    wsi_path: str
    isup_grade: int
    gleason_score: str
    data_provider: str


def load_expansion_500(panda_root: Path, seed: int = 42) -> List[PandaRow]:
    train_csv = panda_root / "train.csv"
    with open(train_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_id = {}
    for r in rows:
        try:
            isup = int(r["isup_grade"])
        except (ValueError, KeyError):
            continue
        sid = r["image_id"].strip()
        by_id[sid] = PandaRow(
            slide_id=sid,
            wsi_path=str(panda_root / "train_images" / f"{sid}.tiff"),
            isup_grade=isup,
            gleason_score=r["gleason_score"].strip(),
            data_provider=r["data_provider"].strip(),
        )

    sorted_ids = sorted(by_id.keys())
    shuffled = _deterministic_shuffle(sorted_ids, seed)
    start, end = COHORT_SLICE
    cohort_ids = shuffled[start:end]
    return [by_id[i] for i in cohort_ids]


# ------------------------------------------------------------------------------
# Patch extractor — identical to expansion_500.py
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
    try:
        import openslide
        slide = openslide.OpenSlide(str(wsi_path))
        level = min(len(slide.level_dimensions) - 1, 1)
        w, h = slide.level_dimensions[level]
        rgba = slide.read_region((0, 0), level, (w, h))
        return np.array(rgba.convert("RGB")), level
    except Exception:
        pass
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


def extract_patches(wsi_path, slide_id, tile_size=256, tissue_threshold=0.3, max_patches=200):
    pixels, level = _load_slide_pixels(Path(wsi_path))
    H, W, _ = pixels.shape
    mask = _tissue_mask(pixels)
    emitted = 0
    for y in range(0, H - tile_size + 1, tile_size):
        for x in range(0, W - tile_size + 1, tile_size):
            tile_mask = mask[y:y + tile_size, x:x + tile_size]
            if tile_mask.mean() < tissue_threshold:
                continue
            yield Patch(slide_id, x, y, level, pixels[y:y + tile_size, x:x + tile_size, :])
            emitted += 1
            if emitted >= max_patches:
                return


# ------------------------------------------------------------------------------
# Phikon-v1 backbone embedder
# owkin/phikon — MIT-style license, no gate, 768-dim CLS, iBOT ViT-B/16
# Standard transformers AutoModel API; no trust_remote_code needed.
# ------------------------------------------------------------------------------

class PhikonV1Embedder:
    HF_ID = "owkin/phikon"
    OUTPUT_DIM = 768  # CLS token (Phikon-v1 iBOT ViT-B hidden size)

    def __init__(self, device="cuda"):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        print(f"[embedder] loading {self.HF_ID} on {device} ...", flush=True)
        t0 = time.time()
        self.processor = AutoImageProcessor.from_pretrained(self.HF_ID)
        self.model = AutoModel.from_pretrained(self.HF_ID).to(device).eval()
        self._torch = torch
        self.backbone_version = BACKBONE_VERSION
        print(f"[embedder] loaded in {time.time() - t0:.1f}s, output_dim={self.OUTPUT_DIM}")

    def embed(self, patches, batch_size=16):
        torch = self._torch
        if not patches:
            return torch.zeros((0, self.OUTPUT_DIM), dtype=torch.float32)
        out = []
        for start in range(0, len(patches), batch_size):
            batch = patches[start:start + batch_size]
            inputs = self.processor(images=[p.tensor for p in batch], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                res = self.model(**inputs)
            # CLS token at position 0 of last_hidden_state
            out.append(res.last_hidden_state[:, 0, :].detach().cpu())
        return torch.cat(out, dim=0)


# ------------------------------------------------------------------------------
# Feature cache
# ------------------------------------------------------------------------------

class FeatureCache:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, slide_id, backbone_name, version):
        safe = version.replace("/", "_").replace(":", "_").replace("@", "_")
        return self.root / f"{slide_id}__{backbone_name}__{safe}.npz"

    def get(self, slide_id, backbone_name, version):
        import numpy as np
        p = self._p(slide_id, backbone_name, version)
        if not p.exists():
            return None
        with np.load(p) as npz:
            return npz["features"]

    def put(self, slide_id, backbone_name, version, features):
        import numpy as np
        p = self._p(slide_id, backbone_name, version)
        arr = features.detach().cpu().numpy() if hasattr(features, "detach") else features
        np.savez_compressed(p, features=np.asarray(arr))


# ------------------------------------------------------------------------------
# CV training + inference
# ------------------------------------------------------------------------------

def train_and_predict_cv(
    slide_ids, features_per_slide, labels, n_folds, seed, run_id, device,
    input_dim=1024,
):
    import torch

    version = f"trained@{run_id}-cv{n_folds}-seed{seed}-phikonv1"
    n = len(slide_ids)
    fold_of = {i: i % n_folds for i in range(n)}

    predictions = [None] * n
    for fold in range(n_folds):
        train_idx = [i for i in range(n) if fold_of[i] != fold]
        val_idx = [i for i in range(n) if fold_of[i] == fold]
        if not train_idx or not val_idx:
            continue

        torch.manual_seed(seed)
        head = torch.nn.Linear(input_dim, 6).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss()

        pooled = torch.stack([
            torch.as_tensor(features_per_slide[i]).float().to(device).mean(dim=0)
            for i in train_idx
        ])
        y = torch.tensor([labels[i] for i in train_idx], dtype=torch.long, device=device)

        head.train()
        for _ in range(50):
            opt.zero_grad()
            logits = head(pooled)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
        head.eval()

        for i in val_idx:
            feats = torch.as_tensor(features_per_slide[i]).float().to(device)
            p = feats.mean(dim=0, keepdim=True)
            with torch.no_grad():
                logits = head(p)
            probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().tolist()
            grade = int(max(range(len(probs)), key=lambda k: probs[k]))
            predictions[i] = (grade, float(probs[grade]), [float(p) for p in probs])

    return predictions, version


# ------------------------------------------------------------------------------
# Evidence Gate — identical thresholds as expansion_500.py
# ------------------------------------------------------------------------------

GATE_RULE_ID = "confidence-threshold-v1@2026-08-05"


def gate_claim(per_grade_probs, confidence, high=0.70, low=0.30):
    if not per_grade_probs or all(abs(p - per_grade_probs[0]) < 1e-6 for p in per_grade_probs):
        return "NOT_APPLICABLE", {"gate_rule_id": GATE_RULE_ID,
                                  "thresholds": {"high": high, "low": low}}
    if confidence >= high:
        v = "SUPPORTED"
    elif confidence <= low:
        v = "UNSUPPORTED"
    else:
        v = "CONTESTED"
    return v, {"gate_rule_id": GATE_RULE_ID,
               "thresholds": {"high": high, "low": low, "observed_confidence": confidence}}


# ------------------------------------------------------------------------------
# Per-grade metrics — identical to expansion_500.py
# ------------------------------------------------------------------------------

def per_grade_metrics(predictions, labels, n_classes=6):
    per_grade_acc = []
    for k in range(n_classes):
        idx_k = [i for i, y in enumerate(labels) if y == k]
        if not idx_k:
            per_grade_acc.append(None)
            continue
        correct = sum(1 for i in idx_k if predictions[i][0] == k)
        per_grade_acc.append(correct / len(idx_k))

    per_grade_auroc = []
    for k in range(n_classes):
        pos_scores = [predictions[i][2][k] for i, y in enumerate(labels) if y == k]
        neg_scores = [predictions[i][2][k] for i, y in enumerate(labels) if y != k]
        if not pos_scores or not neg_scores:
            per_grade_auroc.append(None)
            continue
        merged = [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores]
        merged.sort(key=lambda t: t[0])
        rank_sum_pos = 0
        for rank, (_, label) in enumerate(merged, start=1):
            if label == 1:
                rank_sum_pos += rank
        n_pos = len(pos_scores)
        n_neg = len(neg_scores)
        auroc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        per_grade_auroc.append(auroc)

    valid_acc = [a for a in per_grade_acc if a is not None]
    macro_acc = sum(valid_acc) / len(valid_acc) if valid_acc else None
    return {
        "per_grade_accuracy": per_grade_acc,
        "per_grade_auroc": per_grade_auroc,
        "macro_accuracy": macro_acc,
    }


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    print("=" * 80)
    print(f"P5 (ASCI) — {COHORT_ID} — backbone: {BACKBONE_NAME}")
    print("=" * 80)

    # Fix P100 CUDA before torch is imported anywhere
    _maybe_fix_pytorch_for_p100()

    panda_root = resolve_panda_root()
    print(f"PANDA mount: {panda_root}")

    slides = load_expansion_500(panda_root, seed=42)
    print(f"Loaded {len(slides)} slides for cohort {COHORT_ID}")

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
    print(f"Device: {device}")

    cache_dir = Path("/kaggle/working/feature_cache_phikonv1")
    cache = FeatureCache(cache_dir)

    # ---------------------------------------------------------- 1. Embedding
    print("\n[1/4] Patch extraction + Phikon-v1 embedding (with cache) ...")
    t0 = time.time()
    embedder = None
    features_per_slide = []
    labels = []
    slide_ids = []
    patch_counts = []
    embed_wall_clock = 0.0
    skipped = 0

    for idx, r in enumerate(slides):
        if idx % 25 == 0:
            print(f"  progress: {idx}/{len(slides)}  elapsed={time.time() - t0:.0f}s",
                  flush=True)
        cached = cache.get(r.slide_id, BACKBONE_NAME, BACKBONE_VERSION)
        if cached is not None:
            feats = cached
            n_patches = int(feats.shape[0])
        else:
            if embedder is None:
                embedder = PhikonV1Embedder(device=device)
            t_slide = time.time()
            try:
                patches = list(extract_patches(Path(r.wsi_path), r.slide_id,
                                               max_patches=200))
            except Exception as e:
                print(f"    WARN: patch extraction failed for {r.slide_id}: {e}")
                skipped += 1
                continue
            n_patches = len(patches)
            if n_patches == 0:
                print(f"    WARN: no tissue patches for {r.slide_id} — skipping")
                skipped += 1
                continue
            feats = embedder.embed(patches, batch_size=16)
            elapsed = time.time() - t_slide
            embed_wall_clock += elapsed
            cache.put(r.slide_id, BACKBONE_NAME, BACKBONE_VERSION, feats)

        features_per_slide.append(feats)
        labels.append(r.isup_grade)
        slide_ids.append(r.slide_id)
        patch_counts.append(n_patches)

    print(f"[1/4] embedding pass done in {time.time() - t0:.1f}s "
          f"(embed_wall_clock={embed_wall_clock:.1f}s; {skipped} slides skipped)")

    if len(slide_ids) < 5:
        raise RuntimeError(
            f"Too few usable slides ({len(slide_ids)}) — CV cannot proceed"
        )

    # ---------------------------------------------------------- 2. CV training
    print(f"\n[2/4] 5-fold CV training + prediction (Phikon-v1 {PhikonV1Embedder.OUTPUT_DIM}-dim head) ...")
    t_mil = time.time()
    predictions, mil_version = train_and_predict_cv(
        slide_ids, features_per_slide, labels,
        n_folds=5, seed=42, run_id="expansion-500-phikonv1",
        device=device, input_dim=PhikonV1Embedder.OUTPUT_DIM,
    )
    mil_wall_clock = time.time() - t_mil
    print(f"  mil_version = {mil_version}")
    print(f"  MIL wall-clock = {mil_wall_clock:.1f}s")

    # ---------------------------------------------------------- 3. Gate + ledger
    print("\n[3/4] Gate + ledger append ...")
    ledger_path = Path("/kaggle/working/expansion_500_phikonv1_ledger.jsonl")
    if ledger_path.exists():
        ledger_path.unlink()
    timestamp_run = datetime.now(timezone.utc).isoformat()
    verdict_dist = {"SUPPORTED": 0, "UNSUPPORTED": 0, "NOT_APPLICABLE": 0, "CONTESTED": 0}

    id_to_slide = {r.slide_id: r for r in slides}

    for i, sid in enumerate(slide_ids):
        if predictions[i] is None:
            continue
        r = id_to_slide[sid]
        grade, confidence, per_grade_probs = predictions[i]
        verdict, justification = gate_claim(per_grade_probs, confidence)
        verdict_dist[verdict] = verdict_dist.get(verdict, 0) + 1
        claim_id = f"{sid}#{COHORT_ID}#{BACKBONE_NAME}#{timestamp_run}"
        row = {
            "claim_id": claim_id,
            "claim_text": (
                f"slide={sid} isup_prediction={grade} "
                f"confidence={confidence:.4f} mil={mil_version}"
            ),
            "evidence_ref": f"file://{r.wsi_path}",
            "verdict": verdict,
            "justification": {
                **justification,
                "mil_version": mil_version,
                "patch_count": patch_counts[i],
                "is_tracer": False,
                "cv_fold": i % 5,
            },
            "timestamp": timestamp_run,
            "backbone_version": BACKBONE_VERSION,
            "cohort_id": COHORT_ID,
            "gate_rule_id": justification["gate_rule_id"],
            "extra": {
                "isup_prediction": grade,
                "confidence": confidence,
                "per_grade_probs": per_grade_probs,
                "mil_version": mil_version,
                "ground_truth_isup": r.isup_grade,
                "ground_truth_gleason": r.gleason_score,
                "data_provider": r.data_provider,
            },
        }
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    print(f"  ledger appended: {ledger_path}")
    print(f"  verdict distribution: {verdict_dist}")

    # ---------------------------------------------------------- 4. Metrics + report
    print("\n[4/4] Metrics + report ...")
    metrics = per_grade_metrics(predictions, labels, n_classes=6)

    total_wall_minutes = (embed_wall_clock + mil_wall_clock) / 60.0
    cost_proxy_usd = total_wall_minutes / 60.0 * 2.50

    report_path = Path("/kaggle/working/expansion_500_phikonv1_report.md")
    lines = [
        f"# {COHORT_ID} — {BACKBONE_NAME} report",
        "",
        "> **Second-backbone validation run.** Same cohort as expansion-500",
        "> (same 500 slides, same CV protocol, same gate thresholds). Only",
        "> change: backbone swapped Phikon-v2 (1024-dim, pathology-specific)",
        "> -> Phikon-v1 (1024-dim, general-purpose Meta AI ViT-L/14).",
        "> Compare verdict distributions with expansion_500_ledger.jsonl via",
        "> p5/analysis/cross_backbone_agreement.py.",
        "",
        f"- generated: {timestamp_run}",
        f"- cohort_id: `{COHORT_ID}`",
        f"- backbone_version: `{BACKBONE_VERSION}`",
        f"- output_dim: `{PhikonV1Embedder.OUTPUT_DIM}`",
        f"- mil_version: `{mil_version}`",
        f"- gate_rule_id: `{GATE_RULE_ID}`",
        f"- slides processed: {len(slide_ids)}",
        f"- slides skipped: {skipped}",
        "",
        "## Verdict distribution",
        "",
        f"- SUPPORTED: {verdict_dist['SUPPORTED']}",
        f"- UNSUPPORTED: {verdict_dist['UNSUPPORTED']}",
        f"- NOT_APPLICABLE: {verdict_dist['NOT_APPLICABLE']}",
        f"- CONTESTED: {verdict_dist['CONTESTED']}",
        "",
        "## Per-grade metrics (5-fold CV honest predictions)",
        "",
        f"- macro_accuracy: {metrics['macro_accuracy']}",
        f"- per_grade_accuracy: {metrics['per_grade_accuracy']}",
        f"- per_grade_auroc: {metrics['per_grade_auroc']}",
        "",
        "## Wall-clocks",
        "",
        f"- embed_wall_clock_seconds: {embed_wall_clock:.2f}",
        f"- mil_wall_clock_seconds: {mil_wall_clock:.2f}",
        f"- total_wall_minutes: {total_wall_minutes:.2f}",
        "",
        "## Cost accounting",
        "",
        "- backend: Kaggle Notebook (free tier)",
        "- cost_actual_usd: 0.00  (Kaggle is free)",
        f"- cost_proxy_usd: {cost_proxy_usd:.4f}",
        "- cost_meter_budget_cap_usd: 300.00",
        f"- budget_remaining_usd: {300.00 - cost_proxy_usd:.4f}",
        "",
        "## Next step",
        "",
        "Download expansion_500_phikonv1_ledger.jsonl to",
        "evidence/expansion_500_hoptimus0/expansion_500_hoptimus0_ledger.jsonl",
        "then run: python3 p5/analysis/cross_backbone_agreement.py",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  report written: {report_path}")

    print("\n" + "=" * 80)
    print(f"{COHORT_ID.upper()} / {BACKBONE_NAME.upper()} SUMMARY")
    print("=" * 80)
    print(f"  slides: {len(slide_ids)}")
    print(f"  skipped: {skipped}")
    print(f"  verdicts: {verdict_dist}")
    print(f"  macro_accuracy: {metrics['macro_accuracy']}")
    print(f"  embed_wall_clock: {embed_wall_clock:.1f}s")
    print(f"  mil_wall_clock: {mil_wall_clock:.1f}s")
    print(f"  cost_proxy: ${cost_proxy_usd:.4f}")

    min_expected = int(0.95 * COHORT_SIZE)
    assert len(slide_ids) >= min_expected, (
        f"Only {len(slide_ids)} slides processed; expected >= {min_expected}"
    )
    assert sum(verdict_dist.values()) == len(slide_ids), "Verdict count mismatch"
    assert cost_proxy_usd <= 300.0, "Cost proxy exceeds budget"

    print(f"\n{COHORT_ID.upper()} / {BACKBONE_NAME.upper()} OK")


if __name__ == "__main__":
    main()
