"""
MetaLaw Discovery System v0.2-FROZEN — Newtonian + NFW Halo Law

Baseline 2: Newtonian gravity with baryonic mass + NFW dark matter halo.
Standard ΛCDM-motivated approach.

Parameters fit per galaxy: upsilon_disk, upsilon_bulge, log10_M200, c (4 total).

NFW profile:
    M_NFW(<r) = M_200 * f(c) * [ln(1 + r/r_s) - (r/r_s)/(1 + r/r_s)]
    where r_s = R_200/c, f(c) = 1/[ln(1+c) - c/(1+c)]

    R_200 is derived from M_200 assuming critical density:
    R_200 = (3 M_200 / (4π * 200 * ρ_crit))^(1/3)

    ρ_crit ≈ 127.4 M_sun/kpc^3 (at z=0, H0=70 km/s/Mpc)
"""

from __future__ import annotations

import math

import numpy as np

from src.constants import G_INTERNAL, SOFTENING_KPC2
from src.laws.interface import GravityLaw


# Critical density at z=0 in internal units
# rho_crit = 3 H0^2 / (8 pi G)
# H0 = 70 km/s/Mpc = 70 / 1e3 km/s/kpc = 0.07 km/s/kpc
# rho_crit = 3 * (0.07)^2 / (8 * pi * 4.3009e-3) = 0.0147 / 0.10808 = 135.9
# More precisely using standard value:
RHO_CRIT: float = 127.4  # M_sun / kpc^3 (H0=70 km/s/Mpc)


class NewtonianNFWLaw(GravityLaw):
    """
    a(r) = G * [M_bary(<r) + M_NFW(<r)] / r^2

    The M_bary(<r) is provided as M_enclosed by the caller.
    This law adds the NFW halo contribution.

    Dimensional analysis:
        M_NFW: [M_sun] ✓
        a = G * M_total / r^2: same as NewtonianLaw ✓
    """

    def __init__(self, log10_M200: float, c: float) -> None:
        """
        Args:
            log10_M200: log10 of halo virial mass [M_sun]
            c: NFW concentration parameter [dimensionless]
        """
        self.log10_M200 = log10_M200
        self.c = c

        # Derived quantities
        self.M200 = 10.0**log10_M200  # [M_sun]

        # R_200 from M_200 = (4/3) * pi * 200 * rho_crit * R_200^3
        # R_200 = (3 M_200 / (800 pi rho_crit))^(1/3)
        # [kpc] = ([M_sun] / [M_sun kpc^-3])^(1/3) = [kpc^3]^(1/3) = [kpc]  ✓
        self.R200 = (3.0 * self.M200 / (800.0 * math.pi * RHO_CRIT)) ** (1.0 / 3.0)

        # Scale radius
        # [kpc] = [kpc] / [dimensionless]  ✓
        self.r_s = self.R200 / c

        # NFW normalization: f(c) = 1 / [ln(1+c) - c/(1+c)]
        # [dimensionless]  ✓
        self.f_c = 1.0 / (math.log(1.0 + c) - c / (1.0 + c))

    def _nfw_enclosed_mass(self, r: float) -> float:
        """
        NFW enclosed mass at radius r.

        M_NFW(<r) = M_200 * f(c) * [ln(1 + r/r_s) - (r/r_s)/(1 + r/r_s)]
        [M_sun] = [M_sun] * [dimless] * [dimless]  ✓
        """
        x = r / self.r_s  # [dimensionless] = [kpc] / [kpc]  ✓
        return self.M200 * self.f_c * (math.log(1.0 + x) - x / (1.0 + x))

    def acceleration(
        self,
        r: float,
        M_enclosed: float,
        rho: float,
    ) -> float:
        # M_total = M_bary + M_NFW
        # [M_sun] = [M_sun] + [M_sun]  ✓
        M_total = M_enclosed + self._nfw_enclosed_mass(r)

        # a = G * M_total / (r^2 + eps)
        # [km^2 s^-2 kpc^-1] = [kpc km^2 s^-2 M_sun^-1] * [M_sun] / [kpc^2]  ✓
        return G_INTERNAL * M_total / (r**2 + SOFTENING_KPC2)

    def param_vector(self) -> np.ndarray:
        return np.array([self.log10_M200, self.c])

    def param_names(self) -> list[str]:
        return ["log10_M200", "c"]

    @property
    def name(self) -> str:
        return "Newton+NFW"
