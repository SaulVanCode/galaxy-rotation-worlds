"""
v0.3 — a0 Correlation Analysis

Key question: does the per-galaxy a0 from PR-only-a0-free correlate
with galaxy properties? If yes, a0 is not a "variable constant" but
a scaling relation — potentially more fundamental than MOND's fixed a0.

Candidate correlators:
    - L_total (total luminosity, proxy for baryonic mass)
    - R_disk (disk scale length, proxy for size)
    - Sigma_eff = L_total / (2*pi*R_eff^2) (effective surface density proxy)
    - V_flat (asymptotic rotation velocity)
    - M_bary_proxy = Upsilon_disk * L_disk (fitted baryonic mass)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from src.constants import G_INTERNAL
from src.data.sparc_adapter import load_sparc
from src.data.split import split_galaxies
from src.fitting.per_galaxy import UPSILON_BOUNDS, _predict_modified_gravity
from src.laws.piecewise_regime import PiecewiseRegimeLaw
from src.scoring.bic import compute_chi2, build_fit_result
from src.simulator import _compute_smoothness
from src.types import FitResult, GalaxyData, SimResult

from scipy.optimize import differential_evolution


ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sparc"


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


def main():
    print("=" * 70)
    print("a0 CORRELATION ANALYSIS")
    print("=" * 70)

    galaxies = load_sparc(DATA_DIR)
    split = split_galaxies(galaxies)

    # Use ALL galaxies for maximum statistical power
    all_gals = split.train + split.validation + split.test
    print(f"\nUsing all {len(all_gals)} galaxies")

    # Load universal params
    with open(ARTIFACTS_DIR / "aggregate" / "universal_params.json") as f:
        univ = json.load(f)
    ai = univ["params"]["alpha_inner"]
    ao = univ["params"]["alpha_outer"]
    w = univ["params"]["w"]

    # Fit PR-only-a0-free to all galaxies
    print(f"\nFitting PR-only-a0-free to all galaxies...")
    results = []
    for i, gal in enumerate(all_gals):
        r = fit_pr_only_a0(gal, ai, ao, w, seed=42)
        results.append((gal, r))
        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(all_gals)}]")

    print(f"  Done. {len(results)} galaxies fitted.")

    # Extract a0 and galaxy properties
    data = []
    for gal, r in results:
        log_a0 = r.params[2]
        upsilon_d = r.params[0]

        L_total = gal.luminosity_disk + gal.luminosity_bulge
        M_bary = upsilon_d * gal.luminosity_disk  # rough baryonic mass proxy
        R_d = gal.scale_length_disk
        V_max = float(np.max(gal.v_obs))

        # Surface density proxy: L / (2 pi R_d^2)
        Sigma = L_total / (2 * math.pi * R_d**2) if R_d > 0 else 0

        # Surface brightness proxy for mass surface density
        Sigma_bary = M_bary / (2 * math.pi * R_d**2) if R_d > 0 else 0

        data.append({
            "name": gal.name,
            "log10_a0": log_a0,
            "a0": 10**log_a0,
            "log10_L": math.log10(max(L_total, 1)),
            "log10_M_bary": math.log10(max(M_bary, 1)),
            "log10_R_d": math.log10(max(R_d, 0.01)),
            "log10_Sigma": math.log10(max(Sigma, 1)),
            "log10_Sigma_bary": math.log10(max(Sigma_bary, 1)),
            "log10_Vmax": math.log10(max(V_max, 1)),
            "upsilon_d": upsilon_d,
            "chi2r": r.chi2_reduced,
        })

    # Correlation analysis
    correlators = [
        ("log10_L", "Total luminosity (log L)"),
        ("log10_M_bary", "Baryonic mass (log M_bary)"),
        ("log10_R_d", "Disk scale length (log R_d)"),
        ("log10_Sigma", "Surface luminosity density (log Sigma_L)"),
        ("log10_Sigma_bary", "Surface mass density (log Sigma_bary)"),
        ("log10_Vmax", "Max rotation velocity (log V_max)"),
    ]

    log_a0_values = np.array([d["log10_a0"] for d in data])

    print(f"\n{'='*75}")
    print(f"{'Correlator':40s} {'r':>7s} {'p-value':>10s} {'slope':>8s} {'Verdict':>10s}")
    print(f"{'-'*75}")

    best_r = 0
    best_name = ""
    best_slope = 0
    best_intercept = 0

    for key, label in correlators:
        x = np.array([d[key] for d in data])

        # Filter out any -inf or nan
        mask = np.isfinite(x) & np.isfinite(log_a0_values)
        x_clean = x[mask]
        y_clean = log_a0_values[mask]

        if len(x_clean) < 10:
            print(f"  {label:40s} {'N/A':>7s} {'too few':>10s}")
            continue

        slope, intercept, r_val, p_val, std_err = stats.linregress(x_clean, y_clean)

        verdict = ""
        if abs(r_val) > 0.7:
            verdict = "STRONG"
        elif abs(r_val) > 0.4:
            verdict = "moderate"
        else:
            verdict = "weak"

        if p_val < 0.001:
            p_str = f"{p_val:.1e}"
        else:
            p_str = f"{p_val:.4f}"

        print(f"  {label:40s} {r_val:7.3f} {p_str:>10s} {slope:8.4f} {verdict:>10s}")

        if abs(r_val) > abs(best_r):
            best_r = r_val
            best_name = label
            best_slope = slope
            best_intercept = intercept
            best_key = key

    print(f"{'='*75}")

    # Detailed analysis of best correlator
    print(f"\n--- Best correlator: {best_name} ---")
    print(f"  r = {best_r:.4f}")
    print(f"  log10(a0) = {best_slope:.4f} * {best_key} + {best_intercept:.4f}")
    print(f"  Interpretation: a0 scales as 10^({best_slope:.3f} * {best_key})")

    # If best correlation is strong, compute residual scatter
    x_best = np.array([d[best_key] for d in data])
    mask = np.isfinite(x_best) & np.isfinite(log_a0_values)
    x_clean = x_best[mask]
    y_clean = log_a0_values[mask]
    y_pred = best_slope * x_clean + best_intercept
    residuals = y_clean - y_pred
    scatter = float(np.std(residuals))
    print(f"  Residual scatter: {scatter:.4f} dex")
    print(f"  Original a0 scatter: {float(np.std(y_clean)):.4f} dex")
    print(f"  Variance explained: {best_r**2 * 100:.1f}%")

    # ASCII scatter plot of best correlation
    print(f"\n  --- Scatter: {best_key} vs log10(a0) ---")
    x_min, x_max = float(np.min(x_clean)), float(np.max(x_clean))
    y_min, y_max = float(np.min(y_clean)), float(np.max(y_clean))

    width, height = 60, 20
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    for xi, yi in zip(x_clean, y_clean):
        col = int((xi - x_min) / (x_max - x_min + 1e-10) * (width - 1))
        row = int((1 - (yi - y_min) / (y_max - y_min + 1e-10)) * (height - 1))
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        grid[row][col] = '*'

    # Draw regression line
    for c in range(width):
        xi = x_min + c * (x_max - x_min) / width
        yi = best_slope * xi + best_intercept
        row = int((1 - (yi - y_min) / (y_max - y_min + 1e-10)) * (height - 1))
        if 0 <= row < height and grid[row][c] == ' ':
            grid[row][c] = '-'

    for row_idx, row in enumerate(grid):
        y_val = y_max - row_idx * (y_max - y_min) / height
        print(f"  {y_val:5.2f} |{''.join(row)}|")
    print(f"        {'_' * width}")
    x_labels = f"  {x_min:.1f}{' ' * (width - 10)}{x_max:.1f}"
    print(f"       {x_labels}")
    print(f"       {' ' * (width // 2 - len(best_key) // 2)}{best_key}")

    # Save
    output = {
        "n_galaxies": len(data),
        "universal_params": {"alpha_inner": ai, "alpha_outer": ao, "w": w},
        "a0_stats": {
            "median_log10": float(np.median(log_a0_values)),
            "std_log10": float(np.std(log_a0_values)),
            "iqr_log10": float(np.percentile(log_a0_values, 75) - np.percentile(log_a0_values, 25)),
        },
        "correlations": {
            key: {
                "label": label,
                "r": float(stats.linregress(
                    np.array([d[key] for d in data]),
                    log_a0_values
                ).rvalue),
                "p": float(stats.linregress(
                    np.array([d[key] for d in data]),
                    log_a0_values
                ).pvalue),
                "slope": float(stats.linregress(
                    np.array([d[key] for d in data]),
                    log_a0_values
                ).slope),
            }
            for key, label in correlators
        },
        "best_correlator": {
            "key": best_key,
            "label": best_name,
            "r": best_r,
            "slope": best_slope,
            "intercept": best_intercept,
            "residual_scatter_dex": scatter,
            "variance_explained_pct": best_r**2 * 100,
        },
        "per_galaxy": data,
    }

    out_path = ARTIFACTS_DIR / "a0_correlation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
