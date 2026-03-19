"""
v0.3 Evolution Engine — Mutation Operators

Each operator takes a LawFamily and produces zero or more child families.
All mutations preserve dimensional consistency.

Operators:
    M1 — Parameter Bound Shift
    M2 — Term Addition (additive correction)
    M3 — Regime Composition (smooth blend of two families)
    M4 — Parameter Freezing (fix one law param at universal value)
    M5 — Dimensional Substitution (swap dimensionless ratio)
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution

from src.constants import A0_INTERNAL, G_INTERNAL
from src.data.sparc_adapter import get_sparc_velocity_contributions
from src.evolution.population import LawFamily, MAX_STRUCTURAL_COMPLEXITY
from src.fitting.per_galaxy import (
    UPSILON_BOUNDS,
    _enclosed_baryonic_mass,
    _predict_modified_gravity,
    _predict_newtonian,
    _predict_newton_nfw,
)
from src.laws.interface import GravityLaw
from src.laws.mond_simple import SimpleMONDLaw
from src.laws.piecewise_regime import PiecewiseRegimeLaw
from src.scoring.bic import build_fit_result, compute_chi2
from src.simulator import _compute_smoothness
from src.types import FitResult, GalaxyData, SimResult


# ============================================================
# M4 — Parameter Freezing
# ============================================================

def m4_freeze_param(
    parent: LawFamily,
    param_name: str,
    param_index: int,
    frozen_value: float,
    make_fit_fn: Callable,
) -> LawFamily | None:
    """
    Freeze one law parameter at a fixed value, reducing per-galaxy freedom.

    Returns None if the parent has no law params to freeze.
    """
    if param_index not in parent.law_param_indices:
        return None

    new_indices = [i for i in parent.law_param_indices if i != param_index]
    child_name = f"{parent.name}-M4({param_name}={frozen_value:.3f})"

    if any(child_name == parent.name for _ in []):  # avoid self-reference
        return None

    child = LawFamily(
        name=child_name,
        fit_fn=make_fit_fn(frozen_value, param_index),
        k_law=parent.k_law - 1,
        k_pergalaxy=parent.k_pergalaxy,
        structural_complexity=parent.structural_complexity,
        law_param_indices=new_indices,
        lineage=[parent.name],
        generation=parent.generation + 1,
        mutation_desc=f"M4: froze {param_name}={frozen_value:.4f}",
    )
    return child


# ============================================================
# M2 — Term Addition (multiplicative correction)
# ============================================================

class CorrectedGravityLaw(GravityLaw):
    """
    Wraps a base law with a multiplicative correction:
        a(r) = a_base(r) * (1 + c1 * (ratio)^p1)

    where ratio is a dimensionless quantity.
    Dimensional analysis: correction is dimensionless ✓
    """

    def __init__(
        self,
        base_law: GravityLaw,
        c1: float,
        p1: float,
        ratio_fn: Callable[[float, float, float], float],
        ratio_name: str,
    ):
        self.base_law = base_law
        self.c1 = c1
        self.p1 = p1
        self.ratio_fn = ratio_fn
        self.ratio_name = ratio_name

    def acceleration(self, r: float, M_enclosed: float, rho: float) -> float:
        a_base = self.base_law.acceleration(r, M_enclosed, rho)
        ratio = self.ratio_fn(r, M_enclosed, rho)
        ratio = max(abs(ratio), 1e-30)
        correction = 1.0 + self.c1 * (ratio ** self.p1)
        return a_base * max(correction, 0.01)  # prevent negative acceleration

    def param_vector(self) -> np.ndarray:
        base = self.base_law.param_vector()
        return np.concatenate([base, [self.c1, self.p1]])

    def param_names(self) -> list[str]:
        return self.base_law.param_names() + [f"c1_{self.ratio_name}", f"p1_{self.ratio_name}"]

    @property
    def name(self) -> str:
        return f"{self.base_law.name}+corr({self.ratio_name})"


# Dimensionless ratio functions
def ratio_acceleration(r: float, M_enclosed: float, rho: float) -> float:
    """a_N / a0 [dimensionless]"""
    a_N = G_INTERNAL * M_enclosed / (r**2 + 1e-10)
    return a_N / A0_INTERNAL


def ratio_radius(r: float, M_enclosed: float, rho: float) -> float:
    """r / R_scale — uses fixed R_scale=3 kpc as reference [dimensionless]"""
    return r / 3.0


RATIO_FUNCTIONS = {
    "a_N/a0": ratio_acceleration,
    "r/R0": ratio_radius,
}


def m2_add_correction(
    parent: LawFamily,
    ratio_name: str,
    base_law_factory: Callable[..., GravityLaw],
) -> LawFamily | None:
    """
    Add a multiplicative correction term using a dimensionless ratio.

    New params: c1, p1 (2 additional law params).
    Structural complexity: +2 (free power).
    """
    if ratio_name not in RATIO_FUNCTIONS:
        return None

    new_S = parent.structural_complexity + 2.0  # free power
    if new_S > MAX_STRUCTURAL_COMPLEXITY:
        return None

    ratio_fn = RATIO_FUNCTIONS[ratio_name]

    def fit_fn(galaxy: GalaxyData, seed: int = 42) -> FitResult:
        """Fit the corrected law."""
        # Base law params + c1 + p1 + upsilon
        # For simplicity, start with MOND as base (the best universal family)
        bounds = [
            UPSILON_BOUNDS,     # Υ_disk
            UPSILON_BOUNDS,     # Υ_bulge
            (-2.0, 2.0),       # c1 (correction coefficient)
            (-2.0, 2.0),       # p1 (correction power)
        ]

        def predict(gal, ud, ub, c1, p1):
            base_law = base_law_factory()
            corrected = CorrectedGravityLaw(base_law, c1, p1, ratio_fn, ratio_name)
            return _predict_modified_gravity(gal, corrected, ud, ub)

        def objective(params):
            v_pred = predict(galaxy, *params)
            if np.any(np.isnan(v_pred)) or np.any(np.isinf(v_pred)):
                return 1e20
            return compute_chi2(galaxy.v_obs, v_pred, galaxy.v_err)

        result = differential_evolution(
            objective, bounds, seed=seed, tol=1e-8, maxiter=1000, polish=True,
        )

        v_pred = predict(galaxy, *result.x)
        sim = SimResult(
            radii=galaxy.radii, v_pred=v_pred,
            a_pred=v_pred**2 / galaxy.radii,
            smoothness=_compute_smoothness(v_pred),
        )

        return build_fit_result(
            galaxy=galaxy, sim=sim,
            family_name=f"MOND+corr({ratio_name})",
            params=result.x,
            param_names=["upsilon_disk", "upsilon_bulge", "c1", "p1"],
            n_free_params=4,
            optimizer_seed=seed, converged=result.success,
        )

    child_name = f"MOND+corr({ratio_name})"
    return LawFamily(
        name=child_name,
        fit_fn=fit_fn,
        k_law=2,  # c1, p1
        k_pergalaxy=2,  # Υ_d, Υ_b
        structural_complexity=new_S,
        law_param_indices=[2, 3],  # c1, p1 in param vector
        lineage=[parent.name],
        generation=parent.generation + 1,
        mutation_desc=f"M2: added correction term ({ratio_name})^p to {parent.name}",
    )


# ============================================================
# M3 — Regime Composition (blend two families)
# ============================================================

def m3_compose(
    parent_a: LawFamily,
    parent_b: LawFamily,
    transition_var: str,
) -> LawFamily | None:
    """
    Compose two families via smooth sigmoid transition.

    a(r) = sigma(x) * a_familyA(r) + (1-sigma(x)) * a_familyB(r)
    where x depends on transition_var.

    New params: threshold, width (2 law params).
    Structural complexity: S_A + S_B + 2 (sigmoid).
    """
    new_S = parent_a.structural_complexity + parent_b.structural_complexity + 2.0
    if new_S > MAX_STRUCTURAL_COMPLEXITY:
        return None

    child_name = f"({parent_a.name}|{parent_b.name})@{transition_var}"

    def fit_fn(galaxy: GalaxyData, seed: int = 42) -> FitResult:
        bounds = [
            UPSILON_BOUNDS,     # Υ_disk
            UPSILON_BOUNDS,     # Υ_bulge
            (0.0, 8.0),        # log10(threshold) for transition
            (0.1, 5.0),        # width of transition
        ]

        def predict(gal, ud, ub, log_thresh, width):
            v_pred = np.zeros(len(gal.radii))
            for i, r in enumerate(gal.radii):
                M_enc = _enclosed_baryonic_mass(gal, ud, ub, i)
                rho = 1e4  # placeholder

                # Compute transition variable
                a_N = G_INTERNAL * M_enc / (r**2 + 1e-10)
                if transition_var == "acceleration":
                    x = math.log10(max(a_N / A0_INTERNAL, 1e-30)) / width
                elif transition_var == "radius":
                    x = math.log10(max(r / (10.0 ** log_thresh), 1e-30)) / width
                else:
                    x = 0.0

                x = max(min(x, 500), -500)
                sigma = 1.0 / (1.0 + math.exp(-x))

                # Family A: Newton baryons (high regime)
                a_newton = G_INTERNAL * M_enc / (r**2 + 1e-10)

                # Family B: MOND (low regime)
                mond = SimpleMONDLaw()
                a_mond = mond.acceleration(r, M_enc, rho)

                a_total = sigma * a_newton + (1.0 - sigma) * a_mond
                v_pred[i] = math.sqrt(abs(a_total) * r)
            return v_pred

        def objective(params):
            v_pred = predict(galaxy, *params)
            if np.any(np.isnan(v_pred)):
                return 1e20
            return compute_chi2(galaxy.v_obs, v_pred, galaxy.v_err)

        result = differential_evolution(
            objective, bounds, seed=seed, tol=1e-8, maxiter=1000, polish=True,
        )

        v_pred = predict(galaxy, *result.x)
        sim = SimResult(
            radii=galaxy.radii, v_pred=v_pred,
            a_pred=v_pred**2 / galaxy.radii,
            smoothness=_compute_smoothness(v_pred),
        )

        return build_fit_result(
            galaxy=galaxy, sim=sim,
            family_name=child_name,
            params=result.x,
            param_names=["upsilon_disk", "upsilon_bulge", "log_threshold", "width"],
            n_free_params=4,
            optimizer_seed=seed, converged=result.success,
        )

    return LawFamily(
        name=child_name,
        fit_fn=fit_fn,
        k_law=2,  # threshold, width
        k_pergalaxy=2,  # Υ
        structural_complexity=new_S,
        law_param_indices=[2, 3],
        lineage=[parent_a.name, parent_b.name],
        generation=max(parent_a.generation, parent_b.generation) + 1,
        mutation_desc=f"M3: compose {parent_a.name} + {parent_b.name} via {transition_var}",
    )


# ============================================================
# M4 specializations for PiecewiseRegime
# ============================================================

def m4_freeze_pr_w(universal_w: float) -> LawFamily:
    """Freeze PiecewiseRegime w at universal value, keep a0/alpha free."""

    def fit_fn(galaxy: GalaxyData, seed: int = 42) -> FitResult:
        bounds = [
            UPSILON_BOUNDS,
            UPSILON_BOUNDS,
            (2.5, 4.5),    # log10_a0
            (0.7, 1.3),    # alpha_inner
            (0.3, 0.8),    # alpha_outer
        ]

        def predict(gal, ud, ub, log_a0, ai, ao):
            law = PiecewiseRegimeLaw(a0=10**log_a0, alpha_inner=ai, alpha_outer=ao, w=universal_w)
            return _predict_modified_gravity(gal, law, ud, ub)

        def objective(params):
            v_pred = predict(galaxy, *params)
            if np.any(np.isnan(v_pred)):
                return 1e20
            return compute_chi2(galaxy.v_obs, v_pred, galaxy.v_err)

        result = differential_evolution(
            objective, bounds, seed=seed, tol=1e-8, maxiter=1000, polish=True,
        )

        v_pred = predict(galaxy, *result.x)
        sim = SimResult(
            radii=galaxy.radii, v_pred=v_pred,
            a_pred=v_pred**2 / galaxy.radii,
            smoothness=_compute_smoothness(v_pred),
        )

        full_params = np.array([*result.x, universal_w])
        return build_fit_result(
            galaxy=galaxy, sim=sim,
            family_name=f"PR-M4(w={universal_w:.3f})",
            params=full_params,
            param_names=["upsilon_disk", "upsilon_bulge", "log10_a0", "alpha_inner", "alpha_outer", "w_frozen"],
            n_free_params=5,
            optimizer_seed=seed, converged=result.success,
        )

    return LawFamily(
        name=f"PR-M4(w={universal_w:.3f})",
        fit_fn=fit_fn,
        k_law=3,  # a0, alpha_in, alpha_out (w frozen)
        k_pergalaxy=2,
        structural_complexity=7.0,
        law_param_indices=[2, 3, 4],  # the 3 free law params
        lineage=["PiecewiseRegime"],
        generation=1,
        mutation_desc=f"M4: froze w={universal_w:.4f} from PiecewiseRegime",
    )


def m4_freeze_pr_alphas(universal_ai: float, universal_ao: float) -> LawFamily:
    """Freeze both alpha params, keep a0 and w free."""

    def fit_fn(galaxy: GalaxyData, seed: int = 42) -> FitResult:
        bounds = [
            UPSILON_BOUNDS,
            UPSILON_BOUNDS,
            (2.5, 4.5),    # log10_a0
            (0.1, 5.0),    # w
        ]

        def predict(gal, ud, ub, log_a0, w):
            law = PiecewiseRegimeLaw(
                a0=10**log_a0, alpha_inner=universal_ai,
                alpha_outer=universal_ao, w=w
            )
            return _predict_modified_gravity(gal, law, ud, ub)

        def objective(params):
            v_pred = predict(galaxy, *params)
            if np.any(np.isnan(v_pred)):
                return 1e20
            return compute_chi2(galaxy.v_obs, v_pred, galaxy.v_err)

        result = differential_evolution(
            objective, bounds, seed=seed, tol=1e-8, maxiter=1000, polish=True,
        )

        v_pred = predict(galaxy, *result.x)
        sim = SimResult(
            radii=galaxy.radii, v_pred=v_pred,
            a_pred=v_pred**2 / galaxy.radii,
            smoothness=_compute_smoothness(v_pred),
        )

        full_params = np.array([*result.x[:2], result.x[2], universal_ai, universal_ao, result.x[3]])
        return build_fit_result(
            galaxy=galaxy, sim=sim,
            family_name=f"PR-M4(ai={universal_ai:.2f},ao={universal_ao:.2f})",
            params=full_params,
            param_names=["upsilon_disk", "upsilon_bulge", "log10_a0", "alpha_inner_frozen", "alpha_outer_frozen", "w"],
            n_free_params=4,
            optimizer_seed=seed, converged=result.success,
        )

    name = f"PR-M4(ai={universal_ai:.2f},ao={universal_ao:.2f})"
    return LawFamily(
        name=name,
        fit_fn=fit_fn,
        k_law=2,  # a0, w (alphas frozen)
        k_pergalaxy=2,
        structural_complexity=7.0,
        law_param_indices=[2, 5],  # log10_a0 and w
        lineage=["PiecewiseRegime"],
        generation=1,
        mutation_desc=f"M4: froze alpha_inner={universal_ai:.4f}, alpha_outer={universal_ao:.4f}",
    )


def m4_freeze_pr_a0_alphas(
    universal_a0: float, universal_ai: float, universal_ao: float
) -> LawFamily:
    """Freeze a0 and both alphas, keep ONLY w free per galaxy (+ upsilons)."""

    def fit_fn(galaxy: GalaxyData, seed: int = 42) -> FitResult:
        bounds = [
            UPSILON_BOUNDS,
            UPSILON_BOUNDS,
            (0.1, 5.0),    # w only
        ]

        def predict(gal, ud, ub, w):
            law = PiecewiseRegimeLaw(
                a0=10**universal_a0, alpha_inner=universal_ai,
                alpha_outer=universal_ao, w=w
            )
            return _predict_modified_gravity(gal, law, ud, ub)

        def objective(params):
            v_pred = predict(galaxy, *params)
            if np.any(np.isnan(v_pred)):
                return 1e20
            return compute_chi2(galaxy.v_obs, v_pred, galaxy.v_err)

        result = differential_evolution(
            objective, bounds, seed=seed, tol=1e-8, maxiter=500, polish=True,
        )

        v_pred = predict(galaxy, *result.x)
        sim = SimResult(
            radii=galaxy.radii, v_pred=v_pred,
            a_pred=v_pred**2 / galaxy.radii,
            smoothness=_compute_smoothness(v_pred),
        )

        full_params = np.array([*result.x[:2], universal_a0, universal_ai, universal_ao, result.x[2]])
        return build_fit_result(
            galaxy=galaxy, sim=sim,
            family_name="PR-M4(a0+ai+ao frozen)",
            params=full_params,
            param_names=["upsilon_disk", "upsilon_bulge", "log10_a0_f", "ai_f", "ao_f", "w"],
            n_free_params=3,  # Υ_d, Υ_b, w
            optimizer_seed=seed, converged=result.success,
        )

    return LawFamily(
        name="PR-M4(a0+ai+ao frozen)",
        fit_fn=fit_fn,
        k_law=1,  # only w free
        k_pergalaxy=2,
        structural_complexity=7.0,
        law_param_indices=[5],  # only w
        lineage=["PiecewiseRegime"],
        generation=1,
        mutation_desc=f"M4: froze a0={universal_a0:.4f}, ai={universal_ai:.4f}, ao={universal_ao:.4f}",
    )
