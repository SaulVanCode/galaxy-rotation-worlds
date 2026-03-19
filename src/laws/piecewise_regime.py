"""
MetaLaw Discovery System v0.2-FROZEN — PiecewiseRegime Generated Law

The one generated family in v0.2. A smooth two-regime law with
sigmoid crossover at an acceleration threshold.

Corrected formulation (dimensionally consistent):

    a(r) = a0 * [sigma(x) * (a_N/a0)^alpha_inner + (1-sigma(x)) * (a_N/a0)^alpha_outer]

    where:
        a_N = G * M_bary(<r) / r^2
        x = log10(a_N / a0) / w
        sigma(x) = 1 / (1 + exp(-x))

Dimensional analysis:
    a_N / a0: [dimless]  ✓
    (a_N / a0)^alpha: [dimless]  ✓
    a0 * [dimless]: [km^2 s^-2 kpc^-1]  ✓

Nesting properties:
    alpha_inner=1, alpha_outer=1 → a = a0 * (a_N/a0) = a_N (Newtonian)  ✓
    alpha_inner=1, alpha_outer=0.5, w→0 → deep MOND: a = a0 * sqrt(a_N/a0) = sqrt(a_N*a0)  ✓

Parameters fit per galaxy: upsilon_disk, upsilon_bulge, a0, alpha_inner, alpha_outer, w (6 total).
For universal evaluation: a0, alpha_inner, alpha_outer, w are fixed globally; only upsilon values fit (2 per galaxy).
"""

from __future__ import annotations

import math

import numpy as np

from src.constants import G_INTERNAL, LOG_FLOOR, SOFTENING_KPC2
from src.laws.interface import GravityLaw, RuleGenerator


class PiecewiseRegimeLaw(GravityLaw):
    """
    Two-regime gravity law with smooth sigmoid transition.

    All parameters are in internal units.
    """

    def __init__(
        self,
        a0: float,
        alpha_inner: float,
        alpha_outer: float,
        w: float,
    ) -> None:
        """
        Args:
            a0: transition acceleration [km^2 s^-2 kpc^-1]
            alpha_inner: power law exponent in high-acceleration regime [dimless]
            alpha_outer: power law exponent in low-acceleration regime [dimless]
            w: transition width in decades of acceleration [dimless, > 0]
        """
        self.a0 = a0
        self.alpha_inner = alpha_inner
        self.alpha_outer = alpha_outer
        self.w = w

    def acceleration(
        self,
        r: float,
        M_enclosed: float,
        rho: float,
    ) -> float:
        # a_N = G * M / (r^2 + eps)
        # [km^2 s^-2 kpc^-1] = [kpc km^2 s^-2 M_sun^-1] * [M_sun] / [kpc^2]  ✓
        a_N = G_INTERNAL * M_enclosed / (r**2 + SOFTENING_KPC2)

        # ratio = a_N / a0  [dimless]  ✓
        ratio = a_N / self.a0
        ratio = max(ratio, LOG_FLOOR)  # prevent log(0)

        # x = log10(ratio) / w  [dimless] / [dimless] = [dimless]  ✓
        x = math.log10(ratio) / self.w

        # sigma(x) = 1 / (1 + exp(-x))  [dimless]  ✓
        # Clamp x to prevent overflow in exp
        x_clamped = max(min(x, 500.0), -500.0)
        sigma = 1.0 / (1.0 + math.exp(-x_clamped))

        # inner term: (a_N/a0)^alpha_inner  [dimless]  ✓
        inner = ratio**self.alpha_inner

        # outer term: (a_N/a0)^alpha_outer  [dimless]  ✓
        outer = ratio**self.alpha_outer

        # a = a0 * [sigma * inner + (1-sigma) * outer]
        # [km^2 s^-2 kpc^-1] = [km^2 s^-2 kpc^-1] * [dimless]  ✓
        return self.a0 * (sigma * inner + (1.0 - sigma) * outer)

    def param_vector(self) -> np.ndarray:
        return np.array([self.a0, self.alpha_inner, self.alpha_outer, self.w])

    def param_names(self) -> list[str]:
        return ["a0", "alpha_inner", "alpha_outer", "w"]

    @property
    def name(self) -> str:
        return "PiecewiseRegime"


class PiecewiseRegimeGenerator(RuleGenerator):
    """
    Generator for the PiecewiseRegime family.

    Parameter bounds are chosen to include known physics as special cases
    while preventing unphysical extremes.
    """

    def generate(self, params: np.ndarray) -> PiecewiseRegimeLaw:
        """
        Args:
            params: [log10_a0, alpha_inner, alpha_outer, w]
                    log10_a0 is in log10 of internal units
        """
        assert len(params) == 4, f"Expected 4 params, got {len(params)}"
        return PiecewiseRegimeLaw(
            a0=10.0**params[0],
            alpha_inner=params[1],
            alpha_outer=params[2],
            w=params[3],
        )

    def param_bounds(self) -> list[tuple[float, float]]:
        return [
            (2.5, 4.5),    # log10(a0) in internal units ≈ 10^{-11.5} to 10^{-9.5} m/s^2
            (0.7, 1.3),    # alpha_inner: 1.0 = Newtonian
            (0.3, 0.8),    # alpha_outer: 0.5 = deep MOND
            (0.1, 5.0),    # transition width in decades
        ]

    def param_names(self) -> list[str]:
        return ["log10_a0", "alpha_inner", "alpha_outer", "w"]

    @property
    def name(self) -> str:
        return "PiecewiseRegime-generator"
