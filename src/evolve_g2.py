"""
v0.3 Evolution Engine — Generation 2

Mutates the most promising G1 families:
- PR-M4(w=0.615) is closest to gap (F1=0.419, F2=0.693)
- MOND+corr(r/R0) has best fit (F1=0.520)

Strategy: push PR-M4(w) into the gap by freezing more params,
and try to universalize MOND+corr variants.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from src.constants import A0_INTERNAL, G_INTERNAL
from src.data.sparc_adapter import load_sparc
from src.data.split import split_galaxies
from src.evolution.generation import run_generation
from src.evolution.mutations import (
    RATIO_FUNCTIONS,
    CorrectedGravityLaw,
    m4_freeze_pr_alphas,
    m4_freeze_pr_w,
)
from src.evolution.population import LawFamily
from src.fitting.per_galaxy import (
    UPSILON_BOUNDS,
    _enclosed_baryonic_mass,
    _predict_modified_gravity,
    fit_mond,
    fit_newton_nfw,
    fit_newtonian,
    fit_piecewise_regime,
)
from src.fitting.universal import fit_piecewise_universal
from src.laws.mond_simple import SimpleMONDLaw
from src.laws.piecewise_regime import PiecewiseRegimeLaw
from src.scoring.bic import build_fit_result, compute_chi2
from src.simulator import _compute_smoothness
from src.types import FitResult, GalaxyData, SimResult


ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sparc"


# ============================================================
# G2 Mutant 1: PR with w AND alpha_outer frozen
# (from PR-M4(w=0.615) → freeze alpha_outer too)
# ============================================================

def make_pr_w_ao_frozen(w: float, ao: float) -> LawFamily:
    """PR with w and alpha_outer frozen. Free: a0, alpha_inner."""

    def fit_fn(galaxy: GalaxyData, seed: int = 42) -> FitResult:
        bounds = [
            UPSILON_BOUNDS,
            UPSILON_BOUNDS,
            (2.5, 4.5),    # log10_a0
            (0.7, 1.3),    # alpha_inner
        ]

        def predict(gal, ud, ub, log_a0, ai):
            law = PiecewiseRegimeLaw(a0=10**log_a0, alpha_inner=ai, alpha_outer=ao, w=w)
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

        full_params = np.array([*result.x, ao, w])
        return build_fit_result(
            galaxy=galaxy, sim=sim,
            family_name="PR-M4(w+ao frozen)",
            params=full_params,
            param_names=["upsilon_disk", "upsilon_bulge", "log10_a0", "alpha_inner", "ao_frozen", "w_frozen"],
            n_free_params=4,
            optimizer_seed=seed, converged=result.success,
        )

    return LawFamily(
        name="PR-M4(w+ao frozen)",
        fit_fn=fit_fn,
        k_law=2,  # a0, alpha_inner
        k_pergalaxy=2,
        structural_complexity=7.0,
        law_param_indices=[2, 3],  # log10_a0, alpha_inner
        lineage=["PR-M4(w=0.615)"],
        generation=2,
        mutation_desc=f"M4: froze w={w:.4f} and alpha_outer={ao:.4f}",
    )


# ============================================================
# G2 Mutant 2: PR with w, alpha_outer, alpha_inner ALL frozen
# Only a0 free (+ upsilons). Most aggressive freezing.
# ============================================================

def make_pr_only_a0_free(a_inner: float, a_outer: float, w: float) -> LawFamily:
    """PR with only a0 free per galaxy. Tests if a0 alone varies."""

    def fit_fn(galaxy: GalaxyData, seed: int = 42) -> FitResult:
        bounds = [
            UPSILON_BOUNDS,
            UPSILON_BOUNDS,
            (2.5, 4.5),    # log10_a0
        ]

        def predict(gal, ud, ub, log_a0):
            law = PiecewiseRegimeLaw(a0=10**log_a0, alpha_inner=a_inner, alpha_outer=a_outer, w=w)
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

        full_params = np.array([*result.x, a_inner, a_outer, w])
        return build_fit_result(
            galaxy=galaxy, sim=sim,
            family_name="PR-only-a0-free",
            params=full_params,
            param_names=["upsilon_disk", "upsilon_bulge", "log10_a0", "ai_f", "ao_f", "w_f"],
            n_free_params=3,
            optimizer_seed=seed, converged=result.success,
        )

    return LawFamily(
        name="PR-only-a0-free",
        fit_fn=fit_fn,
        k_law=1,  # only a0
        k_pergalaxy=2,
        structural_complexity=7.0,
        law_param_indices=[2],  # only log10_a0
        lineage=["PR-M4(w+ao frozen)"],
        generation=2,
        mutation_desc=f"M4: froze ai={a_inner:.4f}, ao={a_outer:.4f}, w={w:.4f}. Only a0 free.",
    )


# ============================================================
# G2 Mutant 3: MOND+corr(r/R0) with FROZEN correction params
# (universal c1, p1 from G1 median)
# ============================================================

def make_mond_corr_r_universal(c1_univ: float, p1_univ: float) -> LawFamily:
    """MOND+corr(r/R0) with correction params frozen at universal values."""

    ratio_fn = RATIO_FUNCTIONS["r/R0"]

    def fit_fn(galaxy: GalaxyData, seed: int = 42) -> FitResult:
        bounds = [UPSILON_BOUNDS, UPSILON_BOUNDS]
        base_law = SimpleMONDLaw()
        corrected = CorrectedGravityLaw(base_law, c1_univ, p1_univ, ratio_fn, "r/R0")

        def predict(gal, ud, ub):
            return _predict_modified_gravity(gal, corrected, ud, ub)

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

        full_params = np.array([*result.x, c1_univ, p1_univ])
        return build_fit_result(
            galaxy=galaxy, sim=sim,
            family_name="MOND+corr(r/R0)-universal",
            params=full_params,
            param_names=["upsilon_disk", "upsilon_bulge", "c1_frozen", "p1_frozen"],
            n_free_params=2,
            optimizer_seed=seed, converged=result.success,
        )

    return LawFamily(
        name="MOND+corr(r/R0)-universal",
        fit_fn=fit_fn,
        k_law=0,  # correction params frozen
        k_pergalaxy=2,
        structural_complexity=4.0,
        law_param_indices=[],
        lineage=["MOND+corr(r/R0)"],
        generation=2,
        mutation_desc=f"M4: froze c1={c1_univ:.4f}, p1={p1_univ:.4f} from MOND+corr(r/R0)",
    )


# ============================================================
# G2 Mutant 4: MOND+corr(a_N/a0) with frozen correction
# ============================================================

def make_mond_corr_accel_universal(c1_univ: float, p1_univ: float) -> LawFamily:
    """MOND+corr(a_N/a0) with correction params frozen."""

    ratio_fn = RATIO_FUNCTIONS["a_N/a0"]

    def fit_fn(galaxy: GalaxyData, seed: int = 42) -> FitResult:
        bounds = [UPSILON_BOUNDS, UPSILON_BOUNDS]
        base_law = SimpleMONDLaw()
        corrected = CorrectedGravityLaw(base_law, c1_univ, p1_univ, ratio_fn, "a_N/a0")

        def predict(gal, ud, ub):
            return _predict_modified_gravity(gal, corrected, ud, ub)

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

        full_params = np.array([*result.x, c1_univ, p1_univ])
        return build_fit_result(
            galaxy=galaxy, sim=sim,
            family_name="MOND+corr(a/a0)-universal",
            params=full_params,
            param_names=["upsilon_disk", "upsilon_bulge", "c1_frozen", "p1_frozen"],
            n_free_params=2,
            optimizer_seed=seed, converged=result.success,
        )

    return LawFamily(
        name="MOND+corr(a/a0)-universal",
        fit_fn=fit_fn,
        k_law=0,
        k_pergalaxy=2,
        structural_complexity=4.0,
        law_param_indices=[],
        lineage=["MOND+corr(a_N/a0)"],
        generation=2,
        mutation_desc=f"M4: froze c1={c1_univ:.4f}, p1={p1_univ:.4f}",
    )


def main():
    max_galaxies = int(sys.argv[1]) if len(sys.argv) > 1 else None

    galaxies = load_sparc(DATA_DIR)
    split = split_galaxies(galaxies)
    train = split.train
    if max_galaxies:
        train = train[:max_galaxies]

    print(f"Train set: {len(train)} galaxies")

    # Load universal params
    with open(ARTIFACTS_DIR / "aggregate" / "universal_params.json") as f:
        univ = json.load(f)
    log_a0 = univ["params"]["log10_a0"]
    ai = univ["params"]["alpha_inner"]
    ao = univ["params"]["alpha_outer"]
    w = univ["params"]["w"]

    # We need G1 correction param medians for MOND+corr universalization.
    # Extract from G1 fit results.
    g1_path = ARTIFACTS_DIR / "evolution" / "gen_001.json"
    assert g1_path.exists(), "Run generation 1 first!"

    # Re-fit MOND+corr variants on a small sample to get median correction params
    print("\nEstimating universal correction params from 20-galaxy subsample...")
    from src.evolution.mutations import m2_add_correction
    mond_family = LawFamily(
        name="MOND-simple", fit_fn=fit_mond,
        k_law=0, k_pergalaxy=2, structural_complexity=2.0,
        law_param_indices=[], generation=0,
    )

    # Get correction params for r/R0
    m2_r = m2_add_correction(mond_family, "r/R0", base_law_factory=SimpleMONDLaw)
    c1_r_values, p1_r_values = [], []
    c1_a_values, p1_a_values = [], []

    m2_a = m2_add_correction(mond_family, "a_N/a0", base_law_factory=SimpleMONDLaw)

    for gal in train[:20]:
        r = m2_r.fit_fn(gal, seed=42)
        c1_r_values.append(r.params[2])
        p1_r_values.append(r.params[3])

        r2 = m2_a.fit_fn(gal, seed=42)
        c1_a_values.append(r2.params[2])
        p1_a_values.append(r2.params[3])

    c1_r_med = float(np.median(c1_r_values))
    p1_r_med = float(np.median(p1_r_values))
    c1_a_med = float(np.median(c1_a_values))
    p1_a_med = float(np.median(p1_a_values))

    print(f"  MOND+corr(r/R0):   c1={c1_r_med:.4f}, p1={p1_r_med:.4f}")
    print(f"  MOND+corr(a_N/a0): c1={c1_a_med:.4f}, p1={p1_a_med:.4f}")

    # Build G0 baselines (carry forward)
    univ_params = np.array([log_a0, ai, ao, w])
    g0 = [
        LawFamily("Newton-baryons", fit_newtonian, 0, 2, 0.0, [], generation=0),
        LawFamily("Newton+NFW", fit_newton_nfw, 0, 4, 1.0, [2, 3], generation=0),
        LawFamily("MOND-simple", fit_mond, 0, 2, 2.0, [], generation=0),
        LawFamily("PiecewiseRegime", fit_piecewise_regime, 4, 2, 7.0, [2, 3, 4, 5], generation=0),
        LawFamily("PiecewiseRegime-universal",
                  lambda gal, seed=42: fit_piecewise_universal(gal, univ_params, seed),
                  0, 2, 7.0, [], generation=0),
    ]

    # G1 survivors (Pareto-optimal from G1)
    g1 = [
        m4_freeze_pr_w(w),
        m4_freeze_pr_alphas(ai, ao),
        m2_r,
        m2_a,
    ]

    # G2 mutants
    g2 = [
        make_pr_w_ao_frozen(w, ao),
        make_pr_only_a0_free(ai, ao, w),
        make_mond_corr_r_universal(c1_r_med, p1_r_med),
        make_mond_corr_accel_universal(c1_a_med, p1_a_med),
    ]

    population = g0 + g1 + g2
    print(f"\nPopulation: {len(g0)} G0 + {len(g1)} G1 + {len(g2)} G2 = {len(population)}")
    for f in population:
        tag = f"G{f.generation}"
        print(f"  [{tag}] {f.name:45s} k_law={f.k_law} k_pg={f.k_pergalaxy} S={f.structural_complexity}")

    # Run
    vectors = run_generation(population, train, generation_num=2, seed=42)
    print("\nDone.")


if __name__ == "__main__":
    main()
