# PANDA Data Loader

Metadata parser and deterministic cohort splitting for the PANDA (Prostate cANcer graDe Assessment) dataset.

## Overview

The PANDA Data Loader owns Kaggle credential handling, metadata parsing, and deterministic train/val/pilot/expansion splits. It exposes a typed row iterator and a reproducible `splits()` accessor for the three canonical cohorts: `gate-oneslide` (1 slide), `pilot-50` (50 slides), and `expansion-500` (500 slides).

All splits are derived from a single sorted slide list and a single named seed, ensuring reproducibility across re-runs and pairwise disjointness.

## Usage

```python
from p5.data_loader import PandaRow, load_rows, splits

# Parse metadata from Kaggle mount or local path
for row in load_rows("/kaggle/input/prostate-cancer-grade-assessment/train.csv"):
    print(f"{row.slide_id}: {row.isup_grade}, {row.gleason_score}")

# Get deterministic splits
gate_slides = splits("gate-oneslide", metadata_csv, seed=42)  # 1 slide
pilot_slides = splits("pilot-50", metadata_csv, seed=42)      # 50 slides
expansion_slides = splits("expansion-500", metadata_csv, seed=42)  # 500 slides
```

## API

### `PandaRow` (NamedTuple)

A typed row from the PANDA metadata CSV.

**Fields:**
- `slide_id: str` — Unique slide identifier
- `wsi_path: str` — Path to the whole-slide image (resolves to `train_images/{slide_id}.tiff`)
- `isup_grade: int` — ISUP Gleason grade (0–5)
- `gleason_score: str` — Gleason pattern, e.g., `"3+4"`
- `tissue_provider: str` — Data provider (karolinska, radboudumc, etc.)

### `load_rows(metadata_csv: Path) -> Iterator[PandaRow]`

Parse PANDA metadata CSV and yield typed rows.

**Args:**
- `metadata_csv`: Path to the PANDA metadata CSV (e.g., `/kaggle/input/prostate-cancer-grade-assessment/train.csv`)

**Yields:**
- `PandaRow` objects with `wsi_path` resolved relative to the CSV's parent directory

**Raises:**
- `FileNotFoundError`: If the CSV does not exist
- `ValueError`: If required columns are missing

**Notes:**
- Rows with missing or non-numeric `isup_grade` are silently skipped
- `wsi_path` follows the convention `{csv_parent}/train_images/{slide_id}.tiff`

### `splits(cohort_id: Literal["gate-oneslide", "pilot-50", "expansion-500"], metadata_csv: Path, seed: int = 42) -> list[str]`

Produce deterministic cohort splits from metadata.

**Args:**
- `cohort_id`: One of `"gate-oneslide"` (1 slide), `"pilot-50"` (50 slides), or `"expansion-500"` (500 slides)
- `metadata_csv`: Path to the PANDA metadata CSV
- `seed`: Random seed for reproducibility (default 42)

**Returns:**
- List of slide_ids belonging to the requested cohort, in sorted order

**Raises:**
- `FileNotFoundError`: If metadata_csv does not exist
- `ValueError`: If cohort_id is not recognized or there are insufficient slides

**Guarantees:**
- **Determinism:** Two calls with the same seed return identical lists
- **Disjointness:** The three cohorts are pairwise disjoint (no slide appears in two cohorts)
- **Reproducibility:** Splits depend only on the slide_id list and seed, not on CSV row order

## Cohort Sizes and Schedule

| Cohort | Size | Deadline | Purpose |
|--------|------|----------|---------|
| `gate-oneslide` | 1 slide | Aug 15 | End-to-end sanity check on the imaging pipeline |
| `pilot-50` | 50 slides | Aug 19 | De-risk expansion; seed the evidence ledger (NOT in abstract) |
| `expansion-500` | 500 slides | Aug 27 | Headline verdict distribution for the final paper |

## Data Format

The PANDA metadata CSV has columns: `slide_id`, `image_width`, `image_height`, `isup_grade`, `gleason_score`, `data_provider`. The loader requires at least `slide_id`, `isup_grade`, `gleason_score`, and `data_provider`.

Example row:
```
slide_id,image_width,image_height,isup_grade,gleason_score,data_provider
prostate_001,69120,71680,0,0+0,karolinska
```

## Local Testing

Run pytest with:

```bash
python3 -m pytest p5/data_loader/tests -v
```

Tests use a small fixture CSV (`tiny_train.csv`, 60 slides) to validate behavior without requiring the full Kaggle dataset.

**Test coverage:**
- Metadata parsing and type correctness
- Split sizes and determinism
- Pairwise disjointness of cohorts
- Error handling for invalid inputs
- WSI path resolution

## Files

- `loader.py` — Core data loader implementation
- `splits.py` — Cohort splitting logic
- `__init__.py` — Public API exports
- `tests/test_loader.py` — Comprehensive pytest tests
- `tests/fixtures/tiny_train.csv` — Small fixture CSV for unit tests
- `requirements.txt` — Module dependencies (pytest)
- `notebooks/smoke.py` — Kaggle Notebook smoke script

## Kaggle Notebook Integration

The `notebooks/smoke.py` script imports the loader and runs a smoke test against `/kaggle/input/prostate-cancer-grade-assessment/train.csv`. Push it via:

```bash
export PATH="$HOME/.local/bin:$PATH"
cd p5/data_loader/notebooks
kaggle kernels push -p .
```

The smoke test prints the first 5 rows, the three cohort sizes, and confirms they are exactly 1/50/500.

## Design Notes

- **Deterministic shuffling:** Splits are computed from a sorted slide list using a deterministic Fisher-Yates shuffle seeded by the `seed` parameter. This ensures reproducibility across Python sessions and machines.
- **No data fabrication:** The loader reads only what the PANDA metadata provides. Missing grades or invalid entries are silently skipped (their rows are not yielded).
- **Relative path resolution:** `wsi_path` is resolved relative to the CSV's parent directory, allowing the same code to work with Kaggle mounts (`/kaggle/input/...`) and local paths.
- **Iterator pattern:** `load_rows()` returns an iterator, not a list, to support streaming through large CSVs without loading all rows into memory.
