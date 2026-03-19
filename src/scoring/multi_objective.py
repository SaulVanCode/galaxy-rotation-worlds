"""
MetaLaw Discovery System v0.3 — Multi-Objective Fitness Scoring

Four independent axes:
    F1: Fit quality (median chi2_reduced, lower is better)
    F2: Parameter universality (how consistent are law params across galaxies)
    F3: Structural simplicity (inverse of complexity)
    F4: RAR transfer (predicts Radial Acceleration Relation scatter)

No axis is collapsed into a weighted sum. Pareto dominance determines
which families represent genuine trade-offs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.constants import G_INTERNAL
from src.types import FitResult, GalaxyData


@dataclass(frozen=True)
class FitnessVector:
    """4D fitness for one law family. All axes: higher is better (normalized)."""

    family_name: str
    f1: float  # fit quality: 1/(1+median_chi2r). Range (0,1]. Higher=better fit.
    f2: float  # universality: 1/(1+mean_CV). Range (0,1]. Higher=more universal.
    f3: float  # simplicity: 1/(k_law+k_pergalaxy+S). Higher=simpler.
    f4: float  # RAR transfer: 1/(1+scatter). Range (0,1]. Higher=better transfer.

    def dominates(self, other: FitnessVector) -> bool:
        """True if self is >= on all axes and > on at least one."""
        axes_self = (self.f1, self.f2, self.f3, self.f4)
        axes_other = (other.f1, other.f2, other.f3, other.f4)
        at_least_as_good = all(s >= o for s, o in zip(axes_self, axes_other))
        strictly_better = any(s > o for s, o in zip(axes_self, axes_other))
        return at_least_as_good and strictly_better

    def to_dict(self) -> dict:
        return {
            "family": self.family_name,
            "f1_fit": round(self.f1, 4),
            "f2_universality": round(self.f2, 4),
            "f3_simplicity": round(self.f3, 4),
            "f4_transfer": round(self.f4, 4),
        }


# --- Structural complexity definitions ---

STRUCTURAL_COMPLEXITY: dict[str, float] = {
    # S = sum of operation costs in the acceleration formula
    # Newton: a = G*M/r^2 → only mult/div → S=0
    "Newton-baryons": 0.0,

    # NFW: a = G*(M_bary+M_NFW)/r^2, M_NFW has ln(1+x) → S=1 (log)
    "Newton+NFW": 1.0,

    # MOND: a = a_N * nu(x), nu = 1/(1-exp(-sqrt(x))) → S = 1(exp) + 1(sqrt) = 2
    "MOND-simple": 2.0,

    # PiecewiseRegime: sigmoid + two free powers + log →
    #   2(free power alpha_in) + 2(free power alpha_out) + 1(log10) + 2(sigmoid) = 7
    "PiecewiseRegime": 7.0,

    # Universal: same structure, fewer free per-galaxy params
    "PiecewiseRegime-universal": 7.0,
}

PARAM_COUNTS: dict[str, tuple[int, int]] = {
    # (k_law, k_pergalaxy)
    "Newton-baryons": (0, 2),
    "Newton+NFW": (0, 4),       # halo params are per-galaxy
    "MOND-simple": (0, 2),      # a0 is fixed, not fit
    "PiecewiseRegime": (4, 2),  # 4 law + 2 upsilon per galaxy
    "PiecewiseRegime-universal": (0, 2),  # law params frozen from training
}


def compute_f1(fit_results: list[FitResult]) -> float:
    """
    F1 = 1/(1 + median(chi2_reduced))

    Range: (0, 1]. F1=0.5 when median chi2r=1 (perfect fit).
    Higher is better.
    """
    chi2s = [r.chi2_reduced for r in fit_results]
    med = float(np.median(chi2s))
    return 1.0 / (1.0 + med)


def compute_f2(fit_results: list[FitResult], law_param_indices: list[int]) -> float:
    """
    F2 = 1/(1 + mean(CV_j)) where CV_j = IQR(theta_j)/|median(theta_j)|

    For families with no law params: F2 = 1.0 (perfectly universal by definition).

    Args:
        fit_results: per-galaxy fit results
        law_param_indices: indices into params array that are law parameters
                          (not upsilon values)
    """
    if not law_param_indices:
        return 1.0  # no law params → trivially universal

    cvs = []
    for j in law_param_indices:
        values = np.array([r.params[j] for r in fit_results])
        median_val = np.median(values)
        iqr = float(np.percentile(values, 75) - np.percentile(values, 25))

        if abs(median_val) < 1e-10:
            cvs.append(10.0)  # effectively infinite CV
        else:
            cvs.append(iqr / abs(median_val))

    mean_cv = float(np.mean(cvs))
    return 1.0 / (1.0 + mean_cv)


def compute_f3(family_name: str) -> float:
    """
    F3 = 1/(k_law + k_pergalaxy + S)

    Higher = simpler.
    """
    k_law, k_pg = PARAM_COUNTS.get(family_name, (0, 2))
    S = STRUCTURAL_COMPLEXITY.get(family_name, 0.0)
    total = k_law + k_pg + S
    if total == 0:
        return 1.0
    return 1.0 / total


def compute_f4_rar(
    fit_results: list[FitResult],
    galaxies: list[GalaxyData],
) -> float:
    """
    F4: Radial Acceleration Relation transfer score.

    The RAR is the tight empirical relation between:
        g_obs = v_obs^2 / r  (observed total acceleration)
        g_bar = v_bar^2 / r  (baryonic acceleration from mass model)

    For each galaxy+radius, we compute:
        g_pred = v_pred^2 / r  (predicted by the law)

    F4 = 1/(1 + RMS(log10(g_pred) - log10(g_obs)))

    This measures whether the law reproduces the correct total acceleration
    at each point, which is a separate test from matching v(r) curves.

    Higher = better RAR reproduction.
    """
    log_residuals = []

    # Build galaxy lookup
    gal_by_name = {g.name: g for g in galaxies}

    for result in fit_results:
        gal = gal_by_name.get(result.galaxy_name)
        if gal is None:
            continue

        for i in range(len(gal.radii)):
            r = gal.radii[i]
            if r < 0.1:  # skip innermost points (softening effects)
                continue

            v_obs = gal.v_obs[i]
            v_pred = result.v_pred[i]

            if v_obs <= 0 or v_pred <= 0:
                continue

            # g = v^2 / r  [km^2 s^-2 kpc^-1]
            g_obs = v_obs**2 / r
            g_pred = v_pred**2 / r

            if g_obs > 0 and g_pred > 0:
                log_residuals.append(np.log10(g_pred) - np.log10(g_obs))

    if not log_residuals:
        return 0.0

    rms = float(np.sqrt(np.mean(np.array(log_residuals)**2)))
    return 1.0 / (1.0 + rms)


def compute_fitness(
    family_name: str,
    fit_results: list[FitResult],
    galaxies: list[GalaxyData],
    law_param_indices: list[int],
) -> FitnessVector:
    """Compute full 4D fitness vector for one family."""
    return FitnessVector(
        family_name=family_name,
        f1=compute_f1(fit_results),
        f2=compute_f2(fit_results, law_param_indices),
        f3=compute_f3(family_name),
        f4=compute_f4_rar(fit_results, galaxies),
    )


def pareto_front(vectors: list[FitnessVector]) -> list[FitnessVector]:
    """
    Compute the Pareto front from a list of fitness vectors.

    A vector is Pareto-optimal if no other vector dominates it.
    """
    front = []
    for v in vectors:
        dominated = False
        for other in vectors:
            if other.family_name == v.family_name:
                continue
            if other.dominates(v):
                dominated = True
                break
        if not dominated:
            front.append(v)
    return front
