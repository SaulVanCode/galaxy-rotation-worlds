"""
MetaLaw Discovery System v0.2-FROZEN — Core Data Types

All dataclasses use internal units (see constants.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class GalaxyData:
    """Observed rotation curve data for one galaxy, in internal units."""

    name: str
    radii: np.ndarray          # [kpc] observed radial points
    v_obs: np.ndarray          # [km/s] observed circular velocities
    v_err: np.ndarray          # [km/s] observational uncertainties
    v_gas: np.ndarray          # [km/s] gas contribution (from SPARC, not modeled)
    luminosity_disk: float     # [L_sun] 3.6 micron disk luminosity
    luminosity_bulge: float    # [L_sun] 3.6 micron bulge luminosity
    scale_length_disk: float   # [kpc] exponential disk scale length
    scale_length_bulge: float  # [kpc] Hernquist bulge scale length (0 if no bulge)
    distance: float            # [Mpc] galaxy distance
    inclination: float         # [degrees] disk inclination
    quality: int               # SPARC quality flag (1=best, 2=good, 3=poor)

    def __post_init__(self) -> None:
        n = len(self.radii)
        assert len(self.v_obs) == n, f"v_obs length {len(self.v_obs)} != radii length {n}"
        assert len(self.v_err) == n, f"v_err length {len(self.v_err)} != radii length {n}"
        assert len(self.v_gas) == n, f"v_gas length {len(self.v_gas)} != radii length {n}"
        assert self.quality in (1, 2, 3), f"Invalid quality flag: {self.quality}"

    @property
    def n_points(self) -> int:
        return len(self.radii)


@dataclass(frozen=True)
class FitResult:
    """Result of fitting one law family to one galaxy."""

    galaxy_name: str
    family_name: str
    params: np.ndarray         # best-fit parameter vector
    param_names: list[str]     # names for each parameter
    chi2: float                # sum of (v_obs - v_pred)^2 / v_err^2
    chi2_reduced: float        # chi2 / (n_points - n_params)
    bic: float                 # chi2 + k * ln(n)
    n_params: int              # k = number of free parameters
    n_points: int              # n = number of data points
    v_pred: np.ndarray         # [km/s] predicted velocities at observed radii
    residuals: np.ndarray      # (v_obs - v_pred) / v_err
    converged: bool            # optimizer reported convergence
    optimizer_seed: int        # random seed used
    smoothness: float          # curve smoothness metric [0, 1]
    passed_smoothness: bool    # smoothness >= 0.7


@dataclass(frozen=True)
class ComparisonRow:
    """One row in the per-galaxy comparison table."""

    galaxy_name: str
    family_name: str
    chi2_reduced: float
    bic: float
    n_params: int
    upsilon_disk: float        # best-fit disk mass-to-light ratio
    upsilon_bulge: float       # best-fit bulge mass-to-light ratio
    passed_smoothness: bool


@dataclass(frozen=True)
class AggregateResult:
    """Aggregate comparison across all galaxies for one family."""

    family_name: str
    median_chi2: float
    median_bic: float
    iqr_bic: float             # interquartile range of BIC
    n_galaxies: int
    n_wins: int                # galaxies where this family has lowest BIC
    n_passed_smoothness: int
    median_upsilon_disk: float
    median_upsilon_bulge: float


@dataclass(frozen=True)
class SimResult:
    """Output of simulating a rotation curve for one world."""

    radii: np.ndarray          # [kpc]
    v_pred: np.ndarray         # [km/s]
    a_pred: np.ndarray         # [km^2 s^-2 kpc^-1]
    smoothness: float          # curve smoothness [0, 1]
