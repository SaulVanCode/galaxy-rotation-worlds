"""
MetaLaw Discovery System v0.2-FROZEN — Rotation Curve Simulator

Computes v(r) from a gravity law + baryonic mass model + gas data.

Assumption: circular orbits in a static axisymmetric potential.
This is the same assumption used in SPARC analysis.

v(r) = sqrt(r * |a(r)|)

Dimensional analysis:
    [km/s] = sqrt([kpc] * [km^2 s^-2 kpc^-1]) = sqrt([km^2 s^-2]) = [km/s]  ✓
"""

from __future__ import annotations

import math

import numpy as np

from src.laws.interface import GravityLaw
from src.mass_model import BaryonicMassModel, total_enclosed_mass
from src.types import GalaxyData, SimResult


def simulate_rotation_curve(
    law: GravityLaw,
    mass_model: BaryonicMassModel,
    galaxy: GalaxyData,
) -> SimResult:
    """
    Simulate rotation curve for a given gravity law and mass model.

    Uses the galaxy's observed radii and gas velocities.

    Args:
        law: gravity law to evaluate
        mass_model: stellar mass model (parameterized by upsilon values)
        galaxy: observed galaxy data (provides radii, v_gas)

    Returns:
        SimResult with predicted velocities, accelerations, and smoothness
    """
    n = galaxy.n_points
    v_pred = np.zeros(n)
    a_pred = np.zeros(n)

    for i in range(n):
        r = galaxy.radii[i]
        v_gas = galaxy.v_gas[i]

        # Total baryonic enclosed mass at this radius
        M_enc = total_enclosed_mass(mass_model, r, v_gas)

        # Local density estimate (stellar only — gas density not modeled)
        rho = mass_model.density_disk(r)

        # Gravitational acceleration from the law
        a = law.acceleration(r, M_enc, rho)
        a_pred[i] = a

        # Circular velocity: v = sqrt(r * |a|)
        # [km/s] = sqrt([kpc] * [km^2 s^-2 kpc^-1])  ✓
        v_pred[i] = math.sqrt(abs(a) * r)

    # Smoothness metric: fraction of radial intervals without sign reversal in dv/dr
    smoothness = _compute_smoothness(v_pred)

    return SimResult(
        radii=galaxy.radii.copy(),
        v_pred=v_pred,
        a_pred=a_pred,
        smoothness=smoothness,
    )


def _compute_smoothness(v: np.ndarray) -> float:
    """
    Curve smoothness: 1 - (sign changes in dv/dr) / (N-2).

    Returns 1.0 for monotonic or smooth curves, lower for oscillatory.
    This is NOT dynamical stability — it is a plausibility filter.

    A smoothness < 0.7 causes the world to be rejected (not scored).
    """
    if len(v) < 3:
        return 1.0
    dv = np.diff(v)
    # Count sign changes in dv (ignoring zeros)
    nonzero = dv[dv != 0]
    if len(nonzero) < 2:
        return 1.0
    sign_changes = np.sum(np.diff(np.sign(nonzero)) != 0)
    return 1.0 - sign_changes / (len(v) - 2)


SMOOTHNESS_THRESHOLD: float = 0.7
