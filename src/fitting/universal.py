"""
MetaLaw Discovery System v0.2-FROZEN — Universal Parameter Fitting (Level B)

Fits PiecewiseRegime with law parameters FIXED to universal values,
optimizing ONLY upsilon_disk and upsilon_bulge per galaxy.

This tests: "Can a single set of law parameters work across all galaxies?"

For BIC per-galaxy: k=2 (only Υ values are free).
For aggregate comparison: must add 4 universal params to total budget
(handled in cli.py reporting).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import differential_evolution

from src.data.sparc_adapter import get_sparc_velocity_contributions
from src.fitting.per_galaxy import (
    UPSILON_BOUNDS,
    _enclosed_baryonic_mass,
    _predict_modified_gravity,
)
from src.laws.piecewise_regime import PiecewiseRegimeLaw
from src.scoring.bic import build_fit_result, compute_chi2
from src.simulator import _compute_smoothness
from src.types import FitResult, GalaxyData, SimResult


def fit_piecewise_universal(
    galaxy: GalaxyData,
    universal_law_params: np.ndarray,
    seed: int = 42,
) -> FitResult:
    """
    Fit PiecewiseRegime with fixed universal law parameters.

    Only Υ_disk and Υ_bulge are optimized per galaxy.

    Args:
        galaxy: observed galaxy data
        universal_law_params: [log10_a0, alpha_inner, alpha_outer, w]
        seed: optimizer seed

    Returns:
        FitResult with n_params=2 (only upsilon values are free)
    """
    log_a0, ai, ao, w = universal_law_params
    law = PiecewiseRegimeLaw(a0=10.0**log_a0, alpha_inner=ai, alpha_outer=ao, w=w)

    bounds = [UPSILON_BOUNDS, UPSILON_BOUNDS]

    def objective(params):
        ud, ub = params
        v_pred = _predict_modified_gravity(galaxy, law, ud, ub)
        if np.any(np.isnan(v_pred)):
            return 1e20
        return compute_chi2(galaxy.v_obs, v_pred, galaxy.v_err)

    result = differential_evolution(
        objective, bounds, seed=seed, tol=1e-8, maxiter=500, polish=True,
    )

    v_pred = _predict_modified_gravity(galaxy, law, *result.x)
    sim = SimResult(
        radii=galaxy.radii, v_pred=v_pred,
        a_pred=v_pred**2 / galaxy.radii,
        smoothness=_compute_smoothness(v_pred),
    )

    # Full param vector includes universal params for record-keeping
    full_params = np.concatenate([result.x, universal_law_params])

    return build_fit_result(
        galaxy=galaxy, sim=sim, family_name="PiecewiseRegime-universal",
        params=full_params,
        param_names=["upsilon_disk", "upsilon_bulge", "log10_a0", "alpha_inner", "alpha_outer", "w"],
        n_free_params=2,  # Only Υ values are free per galaxy
        optimizer_seed=seed, converged=result.success,
    )
