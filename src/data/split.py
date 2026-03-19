"""
MetaLaw Discovery System v0.2-FROZEN — Deterministic Train/Val/Test Split

Split is:
    60% train, 20% validation, 20% test

Stratified by galaxy luminosity (proxy for mass) to ensure each split
covers the full range of galaxy types.

Assignment is deterministic: sorted by name, stratified by luminosity
tercile, then assigned round-robin within each tercile.

The split is logged and must not change after Phase 2 is complete.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.types import GalaxyData


@dataclass(frozen=True)
class DataSplit:
    """Container for train/validation/test splits."""
    train: list[GalaxyData]
    validation: list[GalaxyData]
    test: list[GalaxyData]

    @property
    def all_galaxies(self) -> list[GalaxyData]:
        return self.train + self.validation + self.test

    def summary(self) -> str:
        lines = [
            f"Train:      {len(self.train)} galaxies",
            f"Validation: {len(self.validation)} galaxies",
            f"Test:       {len(self.test)} galaxies",
            f"Total:      {len(self.all_galaxies)} galaxies",
        ]
        return "\n".join(lines)


def split_galaxies(
    galaxies: list[GalaxyData],
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> DataSplit:
    """
    Deterministic stratified split.

    Strategy:
    1. Sort galaxies by name (deterministic ordering)
    2. Divide into 3 luminosity terciles
    3. Within each tercile, assign round-robin: train, train, train, val, test
       (60/20/20 ratio: 3 train, 1 val, 1 test per 5 galaxies)

    Args:
        galaxies: list of GalaxyData (order doesn't matter, will be sorted)
        train_frac: fraction for training (default 0.6)
        val_frac: fraction for validation (default 0.2)

    Returns:
        DataSplit with train/validation/test lists
    """
    assert len(galaxies) >= 5, f"Need at least 5 galaxies, got {len(galaxies)}"

    # Sort by name for deterministic ordering
    sorted_gals = sorted(galaxies, key=lambda g: g.name)

    # Compute luminosities for stratification
    luminosities = np.array([g.luminosity_disk + g.luminosity_bulge for g in sorted_gals])

    # Divide into 3 luminosity terciles
    tercile_33 = np.percentile(luminosities, 33.3)
    tercile_67 = np.percentile(luminosities, 66.7)

    terciles: list[list[GalaxyData]] = [[], [], []]
    for gal, lum in zip(sorted_gals, luminosities):
        if lum <= tercile_33:
            terciles[0].append(gal)
        elif lum <= tercile_67:
            terciles[1].append(gal)
        else:
            terciles[2].append(gal)

    # Within each tercile, assign round-robin
    train: list[GalaxyData] = []
    val: list[GalaxyData] = []
    test: list[GalaxyData] = []

    for tercile in terciles:
        for i, gal in enumerate(tercile):
            slot = i % 5
            if slot < 3:   # 0, 1, 2 → train (60%)
                train.append(gal)
            elif slot == 3:  # 3 → validation (20%)
                val.append(gal)
            else:            # 4 → test (20%)
                test.append(gal)

    return DataSplit(train=train, validation=val, test=test)


def log_split(split: DataSplit) -> dict:
    """
    Produce a serializable record of the split for reproducibility.

    Returns dict suitable for JSON serialization.
    """
    return {
        "train": [g.name for g in sorted(split.train, key=lambda g: g.name)],
        "validation": [g.name for g in sorted(split.validation, key=lambda g: g.name)],
        "test": [g.name for g in sorted(split.test, key=lambda g: g.name)],
        "counts": {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
            "total": len(split.all_galaxies),
        },
    }
