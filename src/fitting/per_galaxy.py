"""
MetaLaw Discovery System v0.2-FROZEN — Per-Galaxy Fitting (Level A)

Fits each law family to each galaxy by optimizing over the family's
free parameters using scipy differential_evolution.

For each family, the free parameters are:
    Newton baryons-only:   [Υ_disk, Υ_bulge]                   (2 params)
    Newton+NFW:            [Υ_disk, Υ_bulge, log10_M200, c]    (4 params)
    Simple MOND:           [Υ_disk, Υ_bulge]                   (2 params)
    PiecewiseRegime:       [Υ_disk, Υ_bulge, log10_a0, α_in, α_out, w]  (6 params)

SPARC provides Vdisk and Vbul at M/L=1. For Newtonian families:
    V²_total = V²_gas + Υ_d * V²_disk + Υ_b * V²_bul (+ V²_halo for NFW)

For modified gravity (MOND, PiecewiseRegime):
    M_bary(r) = r/G * (V²_gas + Υ_d * V²_disk + Υ_b * V²_bul)
    then apply the gravity law to get a(r), then v(r) = sqrt(r * a(r))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution

from src.constants import G_INTERNAL, SOFTENING_KPC2
from src.data.sparc_adapter import get_sparc_velocity_contributions
from src.laws.interface import GravityLaw
from src.laws.newtonian import NewtonianLaw
from src.laws.newtonian_nfw import NewtonianNFWLaw
from src.laws.mond_simple import SimpleMONDLaw
from src.laws.piecewise_regime import PiecewiseRegimeLaw
from src.scoring.bic import build_fit_result, compute_chi2
from src.simulator import SMOOTHNESS_THRESHOLD, _compute_smoothness
from src.types import FitResult, GalaxyData, SimResult


# Υ bounds (same for all families)
UPSILON_BOUNDS = (0.1, 1.5)  # M_sun / L_sun at 3.6 micron


def _predict_newtonian(
    galaxy: GalaxyData,
    upsilon_disk: float,
    upsilon_bulge: float,
) -> np.ndarray:
    """
    Predict v(r) for Newtonian baryons-only using SPARC velocity decomposition.

    V²_total = V²_gas + Υ_d * V²_disk + Υ_b * V²_bul
    [km²/s²] = [km²/s²] + [dimless] * [km²/s²]  ✓

    Returns v_pred [km/s].
    """
    v_disk_ml1, v_bul_ml1 = get_sparc_velocity_contributions(galaxy)
    v2 = (
        galaxy.v_gas**2
        + upsilon_disk * v_disk_ml1**2
        + upsilon_bulge * v_bul_ml1**2
    )
    return np.sqrt(np.maximum(v2, 0.0))


def _predict_newton_nfw(
    galaxy: GalaxyData,
    upsilon_disk: float,
    upsilon_bulge: float,
    log10_M200: float,
    c: float,
) -> np.ndarray:
    """
    Predict v(r) for Newton+NFW using SPARC decomposition + NFW halo.

    V²_total = V²_bary + V²_halo
    V²_halo = G * M_NFW(<r) / r
    """
    v_bary = _predict_newtonian(galaxy, upsilon_disk, upsilon_bulge)
    nfw_law = NewtonianNFWLaw(log10_M200, c)

    v2_halo = np.zeros(len(galaxy.radii))
    for i, r in enumerate(galaxy.radii):
        M_nfw = nfw_law._nfw_enclosed_mass(r)
        # V²_halo = G * M_NFW / r
        # [km²/s²] = [kpc km² s⁻² M_sun⁻¹] * [M_sun] / [kpc] = [km²/s²]  ✓
        v2_halo[i] = G_INTERNAL * M_nfw / (r + 1e-10)

    v2_total = v_bary**2 + v2_halo
    return np.sqrt(np.maximum(v2_total, 0.0))


def _enclosed_baryonic_mass(
    galaxy: GalaxyData,
    upsilon_disk: float,
    upsilon_bulge: float,
    index: int,
) -> float:
    """
    Compute enclosed baryonic mass at radial point i using SPARC decomposition.

    M_bary(r) = r/G * (V²_gas + Υ_d * V²_disk + Υ_b * V²_bul)
    [M_sun] = [kpc] / [kpc km² s⁻² M_sun⁻¹] * [km²/s²]
            = [M_sun]  ✓
    """
    v_disk_ml1, v_bul_ml1 = get_sparc_velocity_contributions(galaxy)
    r = galaxy.radii[index]
    v2_bary = (
        galaxy.v_gas[index] ** 2
        + upsilon_disk * v_disk_ml1[index] ** 2
        + upsilon_bulge * v_bul_ml1[index] ** 2
    )
    return r * v2_bary / G_INTERNAL


def _predict_modified_gravity(
    galaxy: GalaxyData,
    law: GravityLaw,
    upsilon_disk: float,
    upsilon_bulge: float,
) -> np.ndarray:
    """
    Predict v(r) for a modified gravity law (MOND or PiecewiseRegime).

    1. Compute M_bary(r) from SPARC velocity decomposition
    2. Apply gravity law: a(r) = law.acceleration(r, M_bary, rho)
    3. v(r) = sqrt(r * a(r))
    """
    v_pred = np.zeros(len(galaxy.radii))

    for i, r in enumerate(galaxy.radii):
        M_enc = _enclosed_baryonic_mass(galaxy, upsilon_disk, upsilon_bulge, i)

        # Density estimate (crude, from disk surface brightness proxy)
        # Not critical for MOND/PiecewiseRegime which depend on acceleration, not density
        rho = 1e4  # placeholder, not used by current law families

        a = law.acceleration(r, M_enc, rho)
        v_pred[i] = math.sqrt(abs(a) * r)

    return v_pred


# --- Family-specific fitting functions ---

def _fit_objective(
    params: np.ndarray,
    galaxy: GalaxyData,
    predict_fn: Callable[..., np.ndarray],
) -> float:
    """Generic objective: chi2 from predicted vs observed velocities."""
    v_pred = predict_fn(galaxy, *params)
    chi2 = compute_chi2(galaxy.v_obs, v_pred, galaxy.v_err)
    # Penalize unphysical results
    if np.any(np.isnan(v_pred)) or np.any(np.isinf(v_pred)):
        return 1e20
    return chi2


def fit_newtonian(galaxy: GalaxyData, seed: int = 42) -> FitResult:
    """Fit Newton baryons-only: optimize [Υ_disk, Υ_bulge]."""
    bounds = [UPSILON_BOUNDS, UPSILON_BOUNDS]

    def predict(gal, ud, ub):
        return _predict_newtonian(gal, ud, ub)

    result = differential_evolution(
        _fit_objective, bounds, args=(galaxy, predict),
        seed=seed, tol=1e-8, maxiter=500, polish=True,
    )

    v_pred = _predict_newtonian(galaxy, *result.x)
    sim = SimResult(
        radii=galaxy.radii, v_pred=v_pred,
        a_pred=v_pred**2 / galaxy.radii,
        smoothness=_compute_smoothness(v_pred),
    )

    return build_fit_result(
        galaxy=galaxy, sim=sim, family_name="Newton-baryons",
        params=result.x, param_names=["upsilon_disk", "upsilon_bulge"],
        n_free_params=2, optimizer_seed=seed, converged=result.success,
    )


def fit_newton_nfw(galaxy: GalaxyData, seed: int = 42) -> FitResult:
    """Fit Newton+NFW: optimize [Υ_disk, Υ_bulge, log10_M200, c]."""
    bounds = [
        UPSILON_BOUNDS,   # Υ_disk
        UPSILON_BOUNDS,   # Υ_bulge
        (9.0, 14.0),     # log10(M200 / M_sun)
        (1.0, 50.0),     # concentration
    ]

    def predict(gal, ud, ub, lm, c):
        return _predict_newton_nfw(gal, ud, ub, lm, c)

    result = differential_evolution(
        _fit_objective, bounds, args=(galaxy, predict),
        seed=seed, tol=1e-8, maxiter=1000, polish=True,
    )

    v_pred = _predict_newton_nfw(galaxy, *result.x)
    sim = SimResult(
        radii=galaxy.radii, v_pred=v_pred,
        a_pred=v_pred**2 / galaxy.radii,
        smoothness=_compute_smoothness(v_pred),
    )

    return build_fit_result(
        galaxy=galaxy, sim=sim, family_name="Newton+NFW",
        params=result.x,
        param_names=["upsilon_disk", "upsilon_bulge", "log10_M200", "c"],
        n_free_params=4, optimizer_seed=seed, converged=result.success,
    )


def fit_mond(galaxy: GalaxyData, seed: int = 42) -> FitResult:
    """Fit Simple MOND: optimize [Υ_disk, Υ_bulge] (a0 fixed)."""
    bounds = [UPSILON_BOUNDS, UPSILON_BOUNDS]
    law = SimpleMONDLaw()

    def predict(gal, ud, ub):
        return _predict_modified_gravity(gal, law, ud, ub)

    result = differential_evolution(
        _fit_objective, bounds, args=(galaxy, predict),
        seed=seed, tol=1e-8, maxiter=500, polish=True,
    )

    v_pred = _predict_modified_gravity(galaxy, law, *result.x)
    sim = SimResult(
        radii=galaxy.radii, v_pred=v_pred,
        a_pred=v_pred**2 / galaxy.radii,
        smoothness=_compute_smoothness(v_pred),
    )

    return build_fit_result(
        galaxy=galaxy, sim=sim, family_name="MOND-simple",
        params=result.x, param_names=["upsilon_disk", "upsilon_bulge"],
        n_free_params=2, optimizer_seed=seed, converged=result.success,
    )


def fit_piecewise_regime(galaxy: GalaxyData, seed: int = 42) -> FitResult:
    """Fit PiecewiseRegime: optimize [Υ_disk, Υ_bulge, log10_a0, α_in, α_out, w]."""
    bounds = [
        UPSILON_BOUNDS,   # Υ_disk
        UPSILON_BOUNDS,   # Υ_bulge
        (2.5, 4.5),      # log10(a0) in internal units
        (0.7, 1.3),      # alpha_inner
        (0.3, 0.8),      # alpha_outer
        (0.1, 5.0),      # transition width
    ]

    def predict(gal, ud, ub, log_a0, ai, ao, w):
        law = PiecewiseRegimeLaw(a0=10.0**log_a0, alpha_inner=ai, alpha_outer=ao, w=w)
        return _predict_modified_gravity(gal, law, ud, ub)

    result = differential_evolution(
        _fit_objective, bounds, args=(galaxy, predict),
        seed=seed, tol=1e-8, maxiter=1500, polish=True,
    )

    log_a0, ai, ao, w = result.x[2], result.x[3], result.x[4], result.x[5]
    law = PiecewiseRegimeLaw(a0=10.0**log_a0, alpha_inner=ai, alpha_outer=ao, w=w)
    v_pred = _predict_modified_gravity(galaxy, law, result.x[0], result.x[1])

    sim = SimResult(
        radii=galaxy.radii, v_pred=v_pred,
        a_pred=v_pred**2 / galaxy.radii,
        smoothness=_compute_smoothness(v_pred),
    )

    return build_fit_result(
        galaxy=galaxy, sim=sim, family_name="PiecewiseRegime",
        params=result.x,
        param_names=["upsilon_disk", "upsilon_bulge", "log10_a0", "alpha_inner", "alpha_outer", "w"],
        n_free_params=6, optimizer_seed=seed, converged=result.success,
    )


# Registry of all fitting functions
FAMILY_FITTERS = {
    "Newton-baryons": fit_newtonian,
    "Newton+NFW": fit_newton_nfw,
    "MOND-simple": fit_mond,
    "PiecewiseRegime": fit_piecewise_regime,
}


def fit_all_families(
    galaxy: GalaxyData,
    seed: int = 42,
    families: list[str] | None = None,
) -> dict[str, FitResult]:
    """
    Fit all (or selected) law families to one galaxy.

    Returns dict mapping family name to FitResult.
    """
    if families is None:
        families = list(FAMILY_FITTERS.keys())

    results: dict[str, FitResult] = {}
    for name in families:
        fitter = FAMILY_FITTERS[name]
        results[name] = fitter(galaxy, seed=seed)

    return results
