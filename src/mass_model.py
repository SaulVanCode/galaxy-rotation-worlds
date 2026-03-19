"""
MetaLaw Discovery System v0.2-FROZEN — Baryonic Mass Model

Baryonic modeling protocol:
    - Stellar disk: exponential surface density (analytic enclosed mass)
    - Stellar bulge: Hernquist profile (analytic enclosed mass)
    - Gas: included via SPARC-provided V_gas contribution, not a separate
      fitted gas density model

Mass-to-light ratios (upsilon_disk, upsilon_bulge) are fit parameters.
Scale lengths are fixed from SPARC photometry.

All quantities in internal units (kpc, M_sun, km/s).
"""

from __future__ import annotations

import math

import numpy as np

from src.constants import G_INTERNAL, SOFTENING_KPC2


class BaryonicMassModel:
    """
    Computes enclosed baryonic mass and local density at any radius.

    The gas contribution is handled separately via V_gas interpolation,
    not through this mass model.
    """

    def __init__(
        self,
        M_disk: float,
        R_disk: float,
        M_bulge: float,
        R_bulge: float,
    ) -> None:
        """
        Args:
            M_disk: total stellar disk mass [M_sun] = upsilon_disk * L_disk
            R_disk: exponential scale length [kpc]
            M_bulge: total stellar bulge mass [M_sun] = upsilon_bulge * L_bulge
            R_bulge: Hernquist scale length [kpc] (0 if no bulge)
        """
        self.M_disk = M_disk
        self.R_disk = R_disk
        self.M_bulge = M_bulge
        self.R_bulge = R_bulge

    def enclosed_mass_disk(self, r: float) -> float:
        """
        Enclosed mass of exponential disk at radius r.

        M_disk(<r) = M_disk * [1 - (1 + r/R_d) * exp(-r/R_d)]

        Dimensional analysis:
            [M_sun] = [M_sun] * [dimless]  ✓
        """
        x = r / self.R_disk  # [dimless] = [kpc] / [kpc]  ✓
        return self.M_disk * (1.0 - (1.0 + x) * math.exp(-x))

    def enclosed_mass_bulge(self, r: float) -> float:
        """
        Enclosed mass of Hernquist bulge at radius r.

        M_bulge(<r) = M_bulge * r^2 / (r + R_b)^2

        Dimensional analysis:
            [M_sun] = [M_sun] * [kpc^2] / [kpc^2]  ✓
        """
        if self.M_bulge <= 0 or self.R_bulge <= 0:
            return 0.0
        return self.M_bulge * r**2 / (r + self.R_bulge) ** 2

    def enclosed_mass_stellar(self, r: float) -> float:
        """Total stellar enclosed mass at radius r [M_sun]."""
        return self.enclosed_mass_disk(r) + self.enclosed_mass_bulge(r)

    def density_disk(self, r: float) -> float:
        """
        Local volume density estimate for the disk (thin disk approximation).

        Sigma(r) = (M_disk / 2*pi*R_d^2) * exp(-r/R_d)    [M_sun / kpc^2]
        rho(r) = Sigma(r) / (2*h)                           [M_sun / kpc^3]

        where h = 0.3 kpc is an assumed disk half-thickness.

        Dimensional analysis:
            Sigma: [M_sun] / [kpc^2]  ✓
            rho: [M_sun kpc^-2] / [kpc] = [M_sun kpc^-3]  ✓
        """
        h = 0.3  # disk half-thickness [kpc] — fixed assumption
        Sigma = (self.M_disk / (2.0 * math.pi * self.R_disk**2)) * math.exp(
            -r / self.R_disk
        )
        return Sigma / (2.0 * h)


def gas_enclosed_mass(r: float, v_gas: float) -> float:
    """
    Infer gas enclosed mass from observed gas rotation velocity.

    From circular orbit: v_gas^2 = G * M_gas(<r) / r
    Therefore: M_gas(<r) = v_gas^2 * r / G

    Dimensional analysis:
        [M_sun] = [km^2/s^2] * [kpc] / [kpc km^2 s^-2 M_sun^-1]
        = [km^2 s^-2 kpc] * [M_sun kpc^-1 km^-2 s^2]
        = [M_sun]  ✓

    Args:
        r: radius [kpc]
        v_gas: gas circular velocity contribution at r [km/s]

    Returns:
        Enclosed gas mass [M_sun]
    """
    return v_gas**2 * r / G_INTERNAL


def total_enclosed_mass(
    model: BaryonicMassModel,
    r: float,
    v_gas: float,
) -> float:
    """
    Total baryonic enclosed mass: stellar + gas.

    Args:
        model: stellar mass model
        r: radius [kpc]
        v_gas: gas velocity at r [km/s] (interpolated from SPARC)

    Returns:
        Total enclosed baryonic mass [M_sun]
    """
    return model.enclosed_mass_stellar(r) + gas_enclosed_mass(r, v_gas)
