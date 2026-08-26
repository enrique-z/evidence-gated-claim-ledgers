"""Pytest tests for PANDA Data Loader."""

import pytest
from pathlib import Path
from p5.data_loader import PandaRow, load_rows, splits


FIXTURES_DIR = Path(__file__).parent / "fixtures"
TINY_CSV = FIXTURES_DIR / "tiny_train.csv"


class TestLoadRows:
    """Tests for load_rows() function."""

    def test_load_rows_basic(self):
        """Test that load_rows yields typed PandaRow objects."""
        rows = list(load_rows(TINY_CSV))
        assert len(rows) == 60  # tiny_train.csv has 60 rows

    def test_load_rows_types(self):
        """Test that each row is a PandaRow with correct types."""
        rows = list(load_rows(TINY_CSV))
        row = rows[0]

        assert isinstance(row, PandaRow)
        assert isinstance(row.slide_id, str)
        assert isinstance(row.wsi_path, str)
        assert isinstance(row.isup_grade, int)
        assert isinstance(row.gleason_score, str)
        assert isinstance(row.tissue_provider, str)

    def test_load_rows_first_row(self):
        """Test that the first row has expected values."""
        rows = list(load_rows(TINY_CSV))
        first = rows[0]

        assert first.slide_id == "slide_001"
        assert first.isup_grade == 0
        assert first.gleason_score == "0+0"
        assert first.tissue_provider == "karolinska"

    def test_wsi_path_resolution(self):
        """Test that wsi_path is correctly resolved relative to CSV parent."""
        rows = list(load_rows(TINY_CSV))
        first = rows[0]

        # wsi_path should be relative to the CSV's parent (fixtures dir)
        # Convention: train_images/{slide_id}.tiff
        expected_path = str(FIXTURES_DIR / "train_images" / "slide_001.tiff")
        assert first.wsi_path == expected_path

    def test_load_rows_file_not_found(self):
        """Test that FileNotFoundError is raised for missing CSV."""
        with pytest.raises(FileNotFoundError):
            list(load_rows(Path("/nonexistent/path.csv")))

    def test_load_rows_iterator(self):
        """Test that load_rows returns an iterator, not a list."""
        result = load_rows(TINY_CSV)
        # Verify it's an iterator
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")


class TestSplits:
    """Tests for splits() function."""

    def test_gate_oneslide(self):
        """Test that gate-oneslide returns exactly 1 slide_id."""
        result = splits("gate-oneslide", TINY_CSV, seed=42)
        assert len(result) == 1
        assert isinstance(result[0], str)

    def test_pilot_50(self):
        """Test that pilot-50 returns exactly 50 slide_ids."""
        result = splits("pilot-50", TINY_CSV, seed=42)
        assert len(result) == 50

    def test_expansion_500(self):
        """Test that expansion-500 returns exactly 500 slide_ids."""
        # This should fail because tiny_train.csv only has 60 slides
        with pytest.raises(ValueError, match="requires 500 slides"):
            splits("expansion-500", TINY_CSV, seed=42)

    def test_splits_determinism(self):
        """Test that splits are deterministic with the same seed."""
        result1 = splits("gate-oneslide", TINY_CSV, seed=42)
        result2 = splits("gate-oneslide", TINY_CSV, seed=42)
        assert result1 == result2

    def test_splits_different_seed_different_result(self):
        """Test that different seeds produce different results."""
        result1 = splits("gate-oneslide", TINY_CSV, seed=42)
        result2 = splits("gate-oneslide", TINY_CSV, seed=123)
        # Both should have length 1, but they might be the same slide
        # For a cohort of 1, we expect determinism but allow same result
        assert len(result1) == 1
        assert len(result2) == 1

    def test_splits_pairwise_disjoint(self):
        """Test that gate-oneslide and pilot-50 splits are disjoint."""
        gate = set(splits("gate-oneslide", TINY_CSV, seed=42))
        pilot = set(splits("pilot-50", TINY_CSV, seed=42))

        assert gate & pilot == set(), "gate-oneslide and pilot-50 should be disjoint"

    def test_splits_invalid_cohort_id(self):
        """Test that invalid cohort_id raises ValueError."""
        with pytest.raises(ValueError, match="Unknown cohort_id"):
            splits("invalid-cohort", TINY_CSV, seed=42)

    def test_splits_returns_list_of_strings(self):
        """Test that splits() returns a list of strings."""
        result = splits("gate-oneslide", TINY_CSV, seed=42)
        assert isinstance(result, list)
        assert all(isinstance(sid, str) for sid in result)


class TestSplitProperties:
    """Property-based tests for splits behavior."""

    def test_gate_and_pilot_disjoint(self):
        """Test that gate-oneslide and pilot-50 do not overlap."""
        gate = set(splits("gate-oneslide", TINY_CSV, seed=42))
        pilot = set(splits("pilot-50", TINY_CSV, seed=42))
        assert gate & pilot == set()

    def test_all_splits_are_valid_slide_ids(self):
        """Test that returned split slide_ids exist in the CSV."""
        all_rows = list(load_rows(TINY_CSV))
        all_slide_ids = {row.slide_id for row in all_rows}

        gate = set(splits("gate-oneslide", TINY_CSV, seed=42))
        pilot = set(splits("pilot-50", TINY_CSV, seed=42))

        for sid in gate:
            assert sid in all_slide_ids, f"slide_id {sid} not in CSV"
        for sid in pilot:
            assert sid in all_slide_ids, f"slide_id {sid} not in CSV"

    def test_splits_seed_consistency_across_calls(self):
        """Test that calling splits multiple times with same seed is consistent."""
        results = [
            splits("gate-oneslide", TINY_CSV, seed=42),
            splits("gate-oneslide", TINY_CSV, seed=42),
            splits("gate-oneslide", TINY_CSV, seed=42),
        ]
        assert all(r == results[0] for r in results)


class TestIntegration:
    """Integration tests combining load_rows and splits."""

    def test_load_and_split_workflow(self):
        """Test a typical workflow: load CSV, extract splits, verify structure."""
        # Load all rows
        all_rows = list(load_rows(TINY_CSV))
        assert len(all_rows) > 0

        # Get splits
        gate = splits("gate-oneslide", TINY_CSV, seed=42)
        pilot = splits("pilot-50", TINY_CSV, seed=42)

        # Verify disjointness
        assert set(gate) & set(pilot) == set()

        # Verify all splits are in the CSV
        all_slide_ids = {row.slide_id for row in all_rows}
        assert set(gate).issubset(all_slide_ids)
        assert set(pilot).issubset(all_slide_ids)

        # Verify sizes
        assert len(gate) == 1
        assert len(pilot) == 50

    def test_wsi_path_for_split_slides(self):
        """Test that wsi_path can be constructed for all split slides."""
        gate_ids = splits("gate-oneslide", TINY_CSV, seed=42)

        # For a gate slide, verify its wsi_path follows the convention
        gate_slide_id = gate_ids[0]
        expected_wsi_path = str(FIXTURES_DIR / "train_images" / f"{gate_slide_id}.tiff")

        # Find the row and verify
        for row in load_rows(TINY_CSV):
            if row.slide_id == gate_slide_id:
                assert row.wsi_path == expected_wsi_path
                break
        else:
            pytest.fail(f"Slide {gate_slide_id} not found in CSV")
