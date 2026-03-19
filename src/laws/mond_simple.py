"""
MetaLaw Discovery System v0.2-FROZEN — Simple MOND Law

Baseline 3: MOND with simple interpolating function.
a0 is FIXED (not fit per galaxy).

Parameters fit per galaxy: upsilon_disk, upsilon_bulge (2 total).

Interpolating function (simple):
    a = a_N * nu(a_N / a0)
    nu(x) = 1 / (1 - exp(-sqrt(x)))

Deep MOND limit (a_N << a0, x → 0):
    nu(x) → 1/sqrt(x)
    a → a_N / sqrt(a_N/a0) = sqrt(a_N * a0)
    v^4 = r^2 * a^2 = r^2 * a_N * a0 = G * M * a0
    This is the Tully-Fisher relation. ✓

Newtonian limit (a_N >> a0, x → ∞):
    nu(x) → 1
    a → a_N
    Standard Newtonian gravity. ✓
"""

from __future__ import annotations

import math

import numpy as np

from src.constants import A0_INTERNAL, G_INTERNAL, SOFTENING_KPC2
from src.laws.interface import GravityLaw


class SimpleMONDLaw(GravityLaw):
    """
    a(r) = a_N(r) * nu(a_N / a0)

    where a_N = G * M_bary / r^2 (standard Newtonian)
    and nu(x) = 1 / (1 - exp(-sqrt(x)))

    Dimensional analysis:
        a_N: [km^2 s^-2 kpc^-1]  ✓
        a_N / a0: [dimless]  ✓
        nu: [dimless]  ✓
        a = a_N * nu: [km^2 s^-2 kpc^-1]  ✓
    """

    def acceleration(
        self,
        r: float,
        M_enclosed: float,
        rho: float,
    ) -> float:
        # a_N = G * M / (r^2 + eps)
        # [km^2 s^-2 kpc^-1] = [kpc km^2 s^-2 M_sun^-1] * [M_sun] / [kpc^2]  ✓
        a_N = G_INTERNAL * M_enclosed / (r**2 + SOFTENING_KPC2)

        # x = a_N / a0  [dimless]  ✓
        x = a_N / A0_INTERNAL

        # nu(x) = 1 / (1 - exp(-sqrt(x)))  [dimless]  ✓
        # Guard against x=0: sqrt(0)=0, exp(0)=1, denominator=0 → use limit
        sqrt_x = math.sqrt(max(x, 1e-30))
        exp_term = math.exp(-sqrt_x)
        denom = 1.0 - exp_term
        if denom < 1e-30:
            # Deep MOND limit: nu → 1/sqrt(x) → a = sqrt(a_N * a0)
            return math.sqrt(a_N * A0_INTERNAL)
        nu = 1.0 / denom

        # a = a_N * nu  [km^2 s^-2 kpc^-1]  ✓
        return a_N * nu

    def param_vector(self) -> np.ndarray:
        return np.array([])  # a0 is fixed, not fit

    def param_names(self) -> list[str]:
        return []

    @property
    def name(self) -> str:
        return "MOND-simple"
