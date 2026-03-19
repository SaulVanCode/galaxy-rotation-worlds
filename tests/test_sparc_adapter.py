"""
MetaLaw Discovery System v0.2-FROZEN — SPARC Adapter Tests

Verify data loading, unit validation, and basic sanity of SPARC data.
"""

import numpy as np
import pytest

from src.data.sparc_adapter import load_sparc, get_sparc_velocity_contributions
from src.data.split import split_galaxies, log_split


# Skip all tests if SPARC data not present
from pathlib import Path
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sparc"

try:
    _galaxies = load_sparc(DATA_DIR, quality_filter=(1, 2))
    HAS_DATA = len(_galaxies) > 0
except Exception:
    _galaxies = []
    HAS_DATA = False

skipif_no_data = pytest.mark.skipif(not HAS_DATA, reason="SPARC data files not found")


@skipif_no_data
class TestSPARCLoading:
    def test_loads_galaxies(self):
        """Should load >100 galaxies with Q=1,2."""
        assert len(_galaxies) >= 100, f"Only loaded {len(_galaxies)} galaxies"

    def test_galaxy_has_required_fields(self):
        """Each galaxy has non-empty arrays of correct length."""
        for gal in _galaxies[:5]:
            assert gal.n_points >= 5
            assert len(gal.radii) == gal.n_points
            assert len(gal.v_obs) == gal.n_points
            assert len(gal.v_err) == gal.n_points
            assert len(gal.v_gas) == gal.n_points

    def test_radii_positive(self):
        """All radii are positive [kpc]."""
        for gal in _galaxies:
            assert np.all(gal.radii > 0), f"{gal.name}: negative radii"

    def test_velocities_reasonable(self):
        """Observed velocities in [0, 500] km/s range."""
        for gal in _galaxies:
            assert np.all(np.abs(gal.v_obs) < 500), f"{gal.name}: velocity > 500 km/s"

    def test_errors_positive(self):
        """All velocity errors are > 0 (floored at 1 km/s)."""
        for gal in _galaxies:
            assert np.all(gal.v_err > 0), f"{gal.name}: non-positive errors"

    def test_quality_filter(self):
        """Only Q=1,2 galaxies are loaded."""
        for gal in _galaxies:
            assert gal.quality in (1, 2), f"{gal.name}: quality={gal.quality}"

    def test_has_sparc_velocity_contributions(self):
        """Each galaxy has v_disk_ml1 and v_bul_ml1 attached."""
        for gal in _galaxies[:5]:
            v_disk, v_bul = get_sparc_velocity_contributions(gal)
            assert len(v_disk) == gal.n_points
            assert len(v_bul) == gal.n_points

    def test_luminosities_positive(self):
        """Disk luminosity is always positive."""
        for gal in _galaxies:
            assert gal.luminosity_disk > 0, f"{gal.name}: non-positive luminosity"

    def test_scale_lengths_positive(self):
        """Disk scale length is always positive."""
        for gal in _galaxies:
            assert gal.scale_length_disk > 0, f"{gal.name}: non-positive R_disk"


@skipif_no_data
class TestSPARCSplit:
    def test_split_sizes(self):
        """60/20/20 split approximately correct."""
        split = split_galaxies(_galaxies)
        total = len(split.all_galaxies)
        assert total == len(_galaxies)
        # Allow ±5% from target ratios
        assert abs(len(split.train) / total - 0.6) < 0.05
        assert abs(len(split.validation) / total - 0.2) < 0.05
        assert abs(len(split.test) / total - 0.2) < 0.05

    def test_no_overlap(self):
        """No galaxy appears in two splits."""
        split = split_galaxies(_galaxies)
        train_names = {g.name for g in split.train}
        val_names = {g.name for g in split.validation}
        test_names = {g.name for g in split.test}
        assert len(train_names & val_names) == 0
        assert len(train_names & test_names) == 0
        assert len(val_names & test_names) == 0

    def test_deterministic(self):
        """Same input produces same split."""
        split1 = split_galaxies(_galaxies)
        split2 = split_galaxies(_galaxies)
        names1 = [g.name for g in split1.train]
        names2 = [g.name for g in split2.train]
        assert names1 == names2

    def test_log_split(self):
        """Log produces serializable record."""
        split = split_galaxies(_galaxies)
        record = log_split(split)
        assert "train" in record
        assert "validation" in record
        assert "test" in record
        assert record["counts"]["total"] == len(_galaxies)

    def test_luminosity_coverage(self):
        """Each split covers similar luminosity range (stratification check)."""
        split = split_galaxies(_galaxies)

        def lum_range(gals):
            lums = [g.luminosity_disk for g in gals]
            return min(lums), max(lums)

        train_range = lum_range(split.train)
        val_range = lum_range(split.validation)
        test_range = lum_range(split.test)

        # All splits should contain low-luminosity galaxies (< 1e9)
        assert train_range[0] < 1e9
        assert val_range[0] < 1e9
        assert test_range[0] < 1e9
