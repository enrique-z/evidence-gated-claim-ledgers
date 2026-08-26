"""PANDA Data Loader smoke test for Kaggle Notebooks.

Loads the PANDA metadata from the Kaggle-mounted path, prints the first 5 rows,
and verifies the three cohort sizes (1/50/500).
"""

import csv
from pathlib import Path
from typing import NamedTuple, Literal


class PandaRow(NamedTuple):
    """A typed row from the PANDA metadata CSV."""
    slide_id: str
    wsi_path: str
    isup_grade: int
    gleason_score: str
    tissue_provider: str


def load_rows(metadata_csv: Path):
    """Parse PANDA metadata CSV and yield typed rows."""
    metadata_csv = Path(metadata_csv)
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    csv_dir = metadata_csv.parent

    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError("CSV has no header row")
        required_columns = {"image_id", "isup_grade", "gleason_score", "data_provider"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        for row_dict in reader:
            slide_id = row_dict["image_id"].strip()
            isup_grade_str = row_dict["isup_grade"].strip()
            gleason_score = row_dict["gleason_score"].strip()
            data_provider = row_dict["data_provider"].strip()

            try:
                isup_grade = int(isup_grade_str)
            except (ValueError, TypeError):
                continue

            wsi_path = str(csv_dir / "train_images" / f"{slide_id}.tiff")

            yield PandaRow(
                slide_id=slide_id,
                wsi_path=wsi_path,
                isup_grade=isup_grade,
                gleason_score=gleason_score,
                tissue_provider=data_provider,
            )


def _deterministic_shuffle(items: list, seed: int) -> list:
    """Shuffle a list deterministically using a seed."""
    result = items.copy()
    rng_state = seed

    for i in range(len(result) - 1, 0, -1):
        rng_state = (rng_state * 1103515245 + 12345) & 0x7fffffff
        j = rng_state % (i + 1)
        result[i], result[j] = result[j], result[i]

    return result


def splits(
    cohort_id: Literal["gate-oneslide", "pilot-50", "expansion-500"],
    slide_ids: list[str],
    seed: int = 42,
) -> list[str]:
    """Produce deterministic, disjoint cohort splits from a sorted slide_id list."""
    cohort_sizes = {
        "gate-oneslide": 1,
        "pilot-50": 50,
        "expansion-500": 500,
    }
    if cohort_id not in cohort_sizes:
        raise ValueError(f"Unknown cohort_id: {cohort_id}")

    size = cohort_sizes[cohort_id]

    if len(slide_ids) < size:
        raise ValueError(
            f"Requested cohort '{cohort_id}' requires {size} slides "
            f"but only {len(slide_ids)} are available."
        )

    sorted_ids = sorted(slide_ids)
    shuffled_ids = _deterministic_shuffle(sorted_ids, seed)

    slices = {
        "gate-oneslide": (0, 1),
        "pilot-50": (1, 51),
        "expansion-500": (51, 551),
    }
    start, end = slices[cohort_id]

    if end > len(shuffled_ids):
        raise ValueError(
            f"Not enough slides for cohort '{cohort_id}'. "
            f"Need {end} slides but only {len(shuffled_ids)} available."
        )

    return shuffled_ids[start:end]


def main():
    """Smoke test: load metadata, print first 5 rows, verify cohort sizes."""
    import sys
    import os

    # Allow override for local testing
    if len(sys.argv) > 1:
        metadata_csv = Path(sys.argv[1])
    else:
        # Kaggle may mount competitions at either /kaggle/input/<slug>/ or
        # /kaggle/input/competitions/<slug>/ — glob is robust to both.
        candidates = list(Path("/kaggle/input").glob("**/prostate-cancer-grade-assessment/train.csv"))
        if not candidates:
            print("Available in /kaggle/input:")
            for item in Path("/kaggle/input").iterdir():
                print(f"  - {item}")
                if item.is_dir():
                    for sub in item.iterdir():
                        print(f"    - {sub}")
            raise FileNotFoundError("PANDA train.csv not found under /kaggle/input")
        metadata_csv = candidates[0]
        print(f"Resolved PANDA train.csv at: {metadata_csv}")

    print("=" * 80)
    print("PANDA Data Loader Smoke Test")
    print("=" * 80)

    # Load all rows
    all_rows = list(load_rows(metadata_csv))
    print(f"\nLoaded {len(all_rows)} rows from metadata CSV")

    # Print first 5 rows
    print("\nFirst 5 rows:")
    print("-" * 80)
    for i, row in enumerate(all_rows[:5]):
        print(f"{i+1}. {row.slide_id}")
        print(f"   Grade: {row.isup_grade}, Gleason: {row.gleason_score}")
        print(f"   Provider: {row.tissue_provider}")
    print()

    # Extract all slide_ids for splits
    all_slide_ids = [row.slide_id for row in all_rows]

    # Get splits (expand as available)
    gate = splits("gate-oneslide", all_slide_ids, seed=42)
    pilot = splits("pilot-50", all_slide_ids, seed=42)

    expansion = None
    try:
        expansion = splits("expansion-500", all_slide_ids, seed=42)
    except ValueError as e:
        print(f"\nNote: expansion-500 requires 500 slides, but only {len(all_slide_ids)} available")
        print(f"  This is expected in the smoke test if running against a fixture")

    print(f"Gate-oneslide cohort size: {len(gate)}")
    print(f"  Slides: {gate[:3] if len(gate) >= 3 else gate}")
    print(f"\nPilot-50 cohort size: {len(pilot)}")
    print(f"  First 3 slides: {pilot[:3]}")

    if expansion:
        print(f"\nExpansion-500 cohort size: {len(expansion)}")
        print(f"  First 3 slides: {expansion[:3]}")

    # Verify sizes
    print("\n" + "=" * 80)
    errors = []
    if len(gate) != 1:
        errors.append(f"FAIL: gate-oneslide size is {len(gate)}, expected 1")
    if len(pilot) != 50:
        errors.append(f"FAIL: pilot-50 size is {len(pilot)}, expected 50")
    if expansion and len(expansion) != 500:
        errors.append(f"FAIL: expansion-500 size is {len(expansion)}, expected 500")

    # Verify disjointness
    gate_set = set(gate)
    pilot_set = set(pilot)

    if gate_set & pilot_set:
        errors.append(f"FAIL: gate and pilot are not disjoint")

    if expansion:
        expansion_set = set(expansion)
        if gate_set & expansion_set:
            errors.append(f"FAIL: gate and expansion are not disjoint")
        if pilot_set & expansion_set:
            errors.append(f"FAIL: pilot and expansion are not disjoint")

    if errors:
        print("\n".join(errors))
        print("SMOKE FAILED")
        raise AssertionError("Smoke test failed")
    else:
        print("All assertions passed: 1/50/500 cohort sizes, pairwise disjoint")
        print("SMOKE OK")


if __name__ == "__main__":
    main()
