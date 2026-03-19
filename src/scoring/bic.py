"""
MetaLaw Discovery System v0.2-FROZEN — BIC Scoring

Primary metric: Bayesian Information Criterion
    BIC = chi2 + k * ln(n)

where:
    chi2 = sum((v_obs - v_pred)^2 / v_err^2)
    k = number of free parameters
    n = number of data points

BIC automatically penalizes complexity. Lower is better.
"""

from __future__ import annotations

import math

import numpy as np

from src.types import FitResult, GalaxyData, SimResult


def compute_chi2(
    v_obs: np.ndarray,
    v_pred: np.ndarray,
    v_err: np.ndarray,
) -> float:
    """
    Sum of squared standardized residuals.

    chi2 = sum((v_obs - v_pred)^2 / v_err^2)
    [dimless] = [km^2/s^2] / [km^2/s^2]  ✓
    """
    residuals = (v_obs - v_pred) / v_err
    return float(np.sum(residuals**2))


def compute_bic(chi2: float, k: int, n: int) -> float:
    """
    Bayesian Information Criterion.

    BIC = chi2 + k * ln(n)
    [dimless] = [dimless] + [dimless] * [dimless]  ✓

    Args:
        chi2: sum of squared standardized residuals
        k: number of free parameters
        n: number of data points
    """
    assert n > 0, "Need at least 1 data point"
    return chi2 + k * math.log(n)


def compute_chi2_reduced(chi2: float, k: int, n: int) -> float:
    """
    Reduced chi-squared.

    chi2_red = chi2 / (n - k)

    Returns inf if n <= k (underdetermined).
    """
    dof = n - k
    if dof <= 0:
        return float("inf")
    return chi2 / dof


def build_fit_result(
    galaxy: GalaxyData,
    sim: SimResult,
    family_name: str,
    params: np.ndarray,
    param_names: list[str],
    n_free_params: int,
    optimizer_seed: int,
    converged: bool,
) -> FitResult:
    """
    Assemble a FitResult from simulation output and fit metadata.

    Args:
        galaxy: observed data
        sim: simulation result
        family_name: name of the law family
        params: best-fit parameter vector (all params including upsilon)
        param_names: names for params
        n_free_params: number of free parameters (k for BIC)
        optimizer_seed: seed used
        converged: whether optimizer reported convergence
    """
    chi2 = compute_chi2(galaxy.v_obs, sim.v_pred, galaxy.v_err)
    n = galaxy.n_points

    return FitResult(
        galaxy_name=galaxy.name,
        family_name=family_name,
        params=params,
        param_names=param_names,
        chi2=chi2,
        chi2_reduced=compute_chi2_reduced(chi2, n_free_params, n),
        bic=compute_bic(chi2, n_free_params, n),
        n_params=n_free_params,
        n_points=n,
        v_pred=sim.v_pred.copy(),
        residuals=(galaxy.v_obs - sim.v_pred) / galaxy.v_err,
        converged=converged,
        optimizer_seed=optimizer_seed,
        smoothness=sim.smoothness,
        passed_smoothness=sim.smoothness >= 0.7,
    )
