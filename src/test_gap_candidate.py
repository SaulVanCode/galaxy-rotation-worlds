"""
v0.3 — Test set evaluation of gap candidate PR-only-a0-free

One-time evaluation on 30 held-out galaxies.
Compares against all baselines + key mutants.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from src.constants import A0_INTERNAL, G_INTERNAL
from src.data.sparc_adapter import load_sparc
from src.data.split import split_galaxies
from src.evolution.mutations import (
    RATIO_FUNCTIONS,
    CorrectedGravityLaw,
)
from src.fitting.per_galaxy import (
    UPSILON_BOUNDS,
    _predict_modified_gravity,
    fit_all_families,
)
from src.fitting.universal import fit_piecewise_universal
from src.laws.mond_simple import SimpleMONDLaw
from src.laws.piecewise_regime import PiecewiseRegimeLaw
from src.scoring.bic import build_fit_result, compute_chi2
from src.scoring.multi_objective import (
    compute_f1,
    compute_f2,
    compute_f4_rar,
    pareto_front,
    FitnessVector,
    STRUCTURAL_COMPLEXITY,
)
from src.simulator import _compute_smoothness
from src.types import FitResult, GalaxyData, SimResult


ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sparc"


def _sf(v):
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    return v


def fit_pr_only_a0(
    galaxy: GalaxyData,
    ai: float, ao: float, w: float,
    seed: int = 42,
) -> FitResult:
    """Fit PR with only a0 free (+ upsilons)."""
    bounds = [UPSILON_BOUNDS, UPSILON_BOUNDS, (2.5, 4.5)]

    def predict(gal, ud, ub, log_a0):
        law = PiecewiseRegimeLaw(a0=10**log_a0, alpha_inner=ai, alpha_outer=ao, w=w)
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
    full_params = np.array([*result.x, ai, ao, w])
    return build_fit_result(
        galaxy=galaxy, sim=sim, family_name="PR-only-a0-free",
        params=full_params,
        param_names=["upsilon_disk", "upsilon_bulge", "log10_a0", "ai_f", "ao_f", "w_f"],
        n_free_params=3, optimizer_seed=seed, converged=result.success,
    )


def fit_mond_corr_r_univ(
    galaxy: GalaxyData,
    c1: float, p1: float,
    seed: int = 42,
) -> FitResult:
    """Fit MOND+corr(r/R0) with frozen correction params."""
    bounds = [UPSILON_BOUNDS, UPSILON_BOUNDS]
    ratio_fn = RATIO_FUNCTIONS["r/R0"]
    law = CorrectedGravityLaw(SimpleMONDLaw(), c1, p1, ratio_fn, "r/R0")

    def predict(gal, ud, ub):
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
    return build_fit_result(
        galaxy=galaxy, sim=sim, family_name="MOND+corr(r/R0)-universal",
        params=np.array([*result.x, c1, p1]),
        param_names=["upsilon_disk", "upsilon_bulge", "c1_f", "p1_f"],
        n_free_params=2, optimizer_seed=seed, converged=result.success,
    )


def main():
    print("=" * 70)
    print("v0.3 TEST SET EVALUATION — GAP CANDIDATE")
    print("=" * 70)

    galaxies = load_sparc(DATA_DIR)
    split = split_galaxies(galaxies)
    test = split.test
    print(f"\nTest set: {len(test)} galaxies")

    # Load params from training
    with open(ARTIFACTS_DIR / "aggregate" / "universal_params.json") as f:
        univ = json.load(f)
    ai = univ["params"]["alpha_inner"]
    ao = univ["params"]["alpha_outer"]
    w_val = univ["params"]["w"]
    univ_params = np.array([univ["params"]["log10_a0"], ai, ao, w_val])

    # MOND+corr universal params (from G2 estimation)
    c1_r, p1_r = -0.2502, -0.2838
    c1_a, p1_a = -0.0974, 0.1164

    # Families to evaluate
    families_info = {
        "Newton-baryons":           {"k_law": 0, "k_pg": 2, "S": 0.0, "law_idx": []},
        "Newton+NFW":               {"k_law": 0, "k_pg": 4, "S": 1.0, "law_idx": [2, 3]},
        "MOND-simple":              {"k_law": 0, "k_pg": 2, "S": 2.0, "law_idx": []},
        "PiecewiseRegime":          {"k_law": 4, "k_pg": 2, "S": 7.0, "law_idx": [2, 3, 4, 5]},
        "PR-only-a0-free":          {"k_law": 1, "k_pg": 2, "S": 7.0, "law_idx": [2]},
        "MOND+corr(r/R0)-universal": {"k_law": 0, "k_pg": 2, "S": 4.0, "law_idx": []},
    }

    # Fit all
    print(f"\nFitting 6 families to {len(test)} test galaxies...")
    all_results: dict[str, list[FitResult]] = {k: [] for k in families_info}
    t0 = time.time()

    for i, gal in enumerate(test):
        # Standard families
        std = fit_all_families(gal, seed=42)
        all_results["Newton-baryons"].append(std["Newton-baryons"])
        all_results["Newton+NFW"].append(std["Newton+NFW"])
        all_results["MOND-simple"].append(std["MOND-simple"])
        all_results["PiecewiseRegime"].append(std["PiecewiseRegime"])

        # Gap candidate
        all_results["PR-only-a0-free"].append(
            fit_pr_only_a0(gal, ai, ao, w_val, seed=42)
        )

        # MOND+corr universal
        all_results["MOND+corr(r/R0)-universal"].append(
            fit_mond_corr_r_univ(gal, c1_r, p1_r, seed=42)
        )

        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(test) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(test)}] {gal.name:15s} ({elapsed:.0f}s, ~{eta:.0f}s remaining)")

    print(f"  Done in {time.time()-t0:.1f}s")

    # Compute 4D fitness
    print("\n" + "=" * 90)
    print(f"{'Family':35s} {'F1(fit)':>8s} {'F2(univ)':>8s} {'F3(simple)':>10s} {'F4(RAR)':>8s} {'Pareto':>7s}")
    print("-" * 90)

    vectors = []
    for fam_name, info in families_info.items():
        results = all_results[fam_name]
        f1 = compute_f1(results)
        f2 = compute_f2(results, info["law_idx"])
        f3 = 1.0 / (info["k_law"] + info["k_pg"] + info["S"]) if (info["k_law"] + info["k_pg"] + info["S"]) > 0 else 1.0
        f4 = compute_f4_rar(results, test)
        fv = FitnessVector(fam_name, f1, f2, f3, f4)
        vectors.append(fv)

    front = pareto_front(vectors)
    front_names = {fv.family_name for fv in front}

    for fv in vectors:
        is_pareto = "YES" if fv.family_name in front_names else "no"
        marker = " <-- GAP" if fv.family_name == "PR-only-a0-free" else ""
        print(f"  {fv.family_name:33s} {fv.f1:8.4f} {fv.f2:8.4f} {fv.f3:10.4f} {fv.f4:8.4f} {is_pareto:>7s}{marker}")

    print("=" * 90)

    # Per-galaxy BIC comparison
    print(f"\n{'Galaxy':15s} {'Newton':>7s} {'NFW':>7s} {'MOND':>7s} {'PR-free':>7s} {'PR-a0':>7s} {'MOND-cu':>7s} {'Best':>20s}")
    print("-" * 95)

    wins = {k: 0 for k in families_info}
    for gal in test:
        bics = {}
        for fam_name in families_info:
            results = all_results[fam_name]
            r = [x for x in results if x.galaxy_name == gal.name][0]
            bics[fam_name] = r.bic
        best = min(bics, key=bics.get)
        wins[best] += 1
        print(f"{gal.name:15s} "
              f"{bics['Newton-baryons']:7.1f} "
              f"{bics['Newton+NFW']:7.1f} "
              f"{bics['MOND-simple']:7.1f} "
              f"{bics['PiecewiseRegime']:7.1f} "
              f"{bics['PR-only-a0-free']:7.1f} "
              f"{bics['MOND+corr(r/R0)-universal']:7.1f} "
              f"{'<-'+best:>20s}")

    print("-" * 95)
    print(f"{'WINS':15s}", end="")
    for fam_name in families_info:
        print(f" {wins[fam_name]:7d}", end="")
    print()

    # Aggregate BIC
    print(f"\n{'Aggregate':35s} {'med(BIC)':>10s} {'med(chi2r)':>10s} {'Wins':>5s} {'k_pg':>5s}")
    print("-" * 70)
    agg = {}
    for fam_name in families_info:
        results = all_results[fam_name]
        bics = [r.bic for r in results]
        chi2s = [r.chi2_reduced for r in results]
        agg[fam_name] = {
            "median_bic": _sf(np.median(bics)),
            "median_chi2r": _sf(np.median(chi2s)),
            "wins": wins[fam_name],
            "k_pg": families_info[fam_name]["k_pg"] + families_info[fam_name]["k_law"],
        }
        print(f"  {fam_name:33s} {np.median(bics):10.1f} {np.median(chi2s):10.2f} {wins[fam_name]:5d} {agg[fam_name]['k_pg']:5d}")

    # Gap check
    pr_a0 = [fv for fv in vectors if fv.family_name == "PR-only-a0-free"][0]
    print(f"\n{'='*70}")
    print(f"GAP CANDIDATE CHECK: PR-only-a0-free")
    print(f"  F1={pr_a0.f1:.4f} (threshold: >=0.35) -> {'PASS' if pr_a0.f1 >= 0.35 else 'FAIL'}")
    print(f"  F2={pr_a0.f2:.4f} (threshold: >=0.70) -> {'PASS' if pr_a0.f2 >= 0.70 else 'FAIL'}")
    print(f"  Pareto-optimal: {'YES' if pr_a0.family_name in front_names else 'NO'}")

    is_gap = pr_a0.f1 >= 0.35 and pr_a0.f2 >= 0.70 and pr_a0.family_name in front_names
    if is_gap:
        print(f"\n  RESULT: GAP CANDIDATE CONFIRMED ON TEST SET")
        print(f"  This family occupies a region of the Pareto front that no")
        print(f"  baseline (Newton, NFW, MOND) covers: moderate fit quality")
        print(f"  with high parameter universality.")
    else:
        print(f"\n  RESULT: GAP CANDIDATE NOT CONFIRMED ON TEST SET")

    # a0 distribution for PR-only-a0-free
    pr_results = all_results["PR-only-a0-free"]
    a0_values = [r.params[2] for r in pr_results]  # log10_a0
    print(f"\n  a0 distribution across test galaxies:")
    print(f"    median(log10_a0) = {np.median(a0_values):.4f}")
    print(f"    IQR(log10_a0)    = {np.percentile(a0_values, 75) - np.percentile(a0_values, 25):.4f}")
    print(f"    range            = [{min(a0_values):.4f}, {max(a0_values):.4f}]")

    print(f"{'='*70}")

    # Save report
    report = {
        "version": "v0.3-gen2",
        "n_test_galaxies": len(test),
        "fitness_vectors": [fv.to_dict() for fv in vectors],
        "pareto_front": [fv.family_name for fv in front],
        "aggregate_bic": agg,
        "gap_candidate": {
            "name": "PR-only-a0-free",
            "confirmed": is_gap,
            "f1": _sf(pr_a0.f1),
            "f2": _sf(pr_a0.f2),
            "f3": _sf(pr_a0.f3),
            "f4": _sf(pr_a0.f4),
            "a0_median": _sf(np.median(a0_values)),
            "a0_iqr": _sf(np.percentile(a0_values, 75) - np.percentile(a0_values, 25)),
        },
    }
    report_path = ARTIFACTS_DIR / "v03_test_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
