"""
MetaLaw Discovery System v0.2-FROZEN — Newtonian Baryons-Only Law

Baseline 1: Pure Newtonian gravity with baryonic mass only.
Expected to fail for most galaxies (rotation curves fall off too fast).
Exists as the null hypothesis.

Parameters fit per galaxy: upsilon_disk, upsilon_bulge (2 total).
No law-level parameters.
"""

from __future__ import annotations

import numpy as np

from src.constants import G_INTERNAL, SOFTENING_KPC2
from src.laws.interface import GravityLaw


class NewtonianLaw(GravityLaw):
    """
    a(r) = G * M_enclosed / r^2

    Dimensional analysis:
        [km^2 s^-2 kpc^-1] = [kpc km^2 s^-2 M_sun^-1] * [M_sun] / [kpc^2]
        = km^2 s^-2 kpc^-1  ✓
    """

    def acceleration(
        self,
        r: float,
        M_enclosed: float,
        rho: float,
    ) -> float:
        # a = G * M / (r^2 + eps)
        # [km^2 s^-2 kpc^-1] = [kpc km^2 s^-2 M_sun^-1] * [M_sun] / [kpc^2]  ✓
        return G_INTERNAL * M_enclosed / (r**2 + SOFTENING_KPC2)

    def param_vector(self) -> np.ndarray:
        return np.array([])  # no law-level parameters

    def param_names(self) -> list[str]:
        return []

    @property
    def name(self) -> str:
        return "Newton-baryons"
