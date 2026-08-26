"""Deterministic cohort splitting logic for PANDA metadata."""

from typing import Literal
import hashlib


def splits(
    cohort_id: Literal["gate-oneslide", "pilot-50", "expansion-500"],
    slide_ids: list[str],
    seed: int = 42,
) -> list[str]:
    """
    Produce deterministic, disjoint cohort splits from a sorted slide_id list.

    All three cohorts are derived from a single sorted list and a single seed,
    ensuring they are reproducible and pairwise disjoint. The seed is used to
    produce a deterministic shuffle; splits are carved sequentially from the
    shuffled list.

    Args:
        cohort_id: One of "gate-oneslide" (1 slide), "pilot-50" (50 slides),
                   or "expansion-500" (500 slides).
        slide_ids: Full sorted list of all available slide_ids.
        seed: Random seed for reproducibility (default 42).

    Returns:
        List of slide_ids belonging to the requested cohort.

    Raises:
        ValueError: If the requested cohort is not recognized or if there are
                    not enough slides to form the requested cohort.
    """
    # Validate cohort_id
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

    # Ensure input is sorted for determinism
    sorted_ids = sorted(slide_ids)

    # Produce deterministic shuffle using seed
    shuffled_ids = _deterministic_shuffle(sorted_ids, seed)

    # Map cohort_id to its slice of the shuffled list
    slices = {
        "gate-oneslide": (0, 1),
        "pilot-50": (1, 51),
        "expansion-500": (51, 551),
    }
    start, end = slices[cohort_id]

    # Check that we have enough slides for this cohort at this position
    if end > len(shuffled_ids):
        raise ValueError(
            f"Not enough slides for cohort '{cohort_id}'. "
            f"Need {end} slides but only {len(shuffled_ids)} available."
        )

    return shuffled_ids[start:end]


def _deterministic_shuffle(items: list, seed: int) -> list:
    """
    Shuffle a list deterministically using a seed.

    Uses a simple seed-based Fisher-Yates shuffle that produces the same
    permutation for the same seed every time.

    Args:
        items: List to shuffle.
        seed: Random seed.

    Returns:
        Shuffled copy of items.
    """
    result = items.copy()
    rng_state = seed

    for i in range(len(result) - 1, 0, -1):
        # Deterministic pseudo-random number in range [0, i]
        rng_state = (rng_state * 1103515245 + 12345) & 0x7fffffff
        j = rng_state % (i + 1)
        result[i], result[j] = result[j], result[i]

    return result
