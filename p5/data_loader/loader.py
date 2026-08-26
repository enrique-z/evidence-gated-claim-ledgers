"""PANDA Data Loader — metadata parsing and deterministic cohort splits."""

from typing import Iterator, NamedTuple, Literal
from pathlib import Path
import csv

from .splits import splits as compute_splits


class PandaRow(NamedTuple):
    """A typed row from the PANDA metadata CSV."""
    slide_id: str
    wsi_path: str
    isup_grade: int
    gleason_score: str
    tissue_provider: str  # data_provider in the CSV


def load_rows(metadata_csv: Path) -> Iterator[PandaRow]:
    """
    Parse PANDA metadata CSV and yield typed PandaRow objects.

    Expects the CSV to have columns: image_id, data_provider, isup_grade, gleason_score
    (PANDA's actual header). `image_id` is exposed as `slide_id` on PandaRow to match the
    project's vocabulary. Resolves wsi_path relative to the metadata CSV's parent directory
    (the standard Kaggle mount structure: csv lives at the dataset root, images in
    train_images/ subdirectory).

    Args:
        metadata_csv: Path to the PANDA metadata CSV.

    Yields:
        PandaRow with fully resolved wsi_path (relative to the Kaggle mount root).

    Raises:
        FileNotFoundError: If metadata_csv does not exist.
        ValueError: If the CSV is missing required columns.
    """
    metadata_csv = Path(metadata_csv)
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    csv_dir = metadata_csv.parent

    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Validate required columns
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

            # Parse isup_grade as int; skip rows with missing/invalid grades
            try:
                isup_grade = int(isup_grade_str)
            except (ValueError, TypeError):
                # Skip rows with missing or non-numeric isup_grade
                continue

            # Construct wsi_path relative to CSV parent (Kaggle mount root)
            # Convention: train_images/{slide_id}.tiff
            wsi_path = str(csv_dir / "train_images" / f"{slide_id}.tiff")

            yield PandaRow(
                slide_id=slide_id,
                wsi_path=wsi_path,
                isup_grade=isup_grade,
                gleason_score=gleason_score,
                tissue_provider=data_provider,
            )


def splits(
    cohort_id: Literal["gate-oneslide", "pilot-50", "expansion-500"],
    metadata_csv: Path,
    seed: int = 42,
) -> list[str]:
    """
    Produce deterministic cohort splits from PANDA metadata.

    Reads the metadata CSV to extract all slide_ids, then computes a deterministic
    split of the requested cohort from a sorted, shuffled slide list.

    Args:
        cohort_id: One of "gate-oneslide" (1 slide), "pilot-50" (50 slides),
                   or "expansion-500" (500 slides).
        metadata_csv: Path to the PANDA metadata CSV.
        seed: Random seed for reproducibility (default 42).

    Returns:
        List of slide_ids belonging to the requested cohort, in sorted order.

    Raises:
        FileNotFoundError: If metadata_csv does not exist.
        ValueError: If the cohort is not recognized or there are not enough slides.
    """
    # Extract all slide_ids from the CSV
    slide_ids = []
    for row in load_rows(metadata_csv):
        slide_ids.append(row.slide_id)

    if not slide_ids:
        raise ValueError("No valid slides found in metadata CSV")

    # Compute splits from the sorted slide_id list
    return compute_splits(cohort_id, slide_ids, seed=seed)
