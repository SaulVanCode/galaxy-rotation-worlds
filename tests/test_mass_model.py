"""
MetaLaw Discovery System v0.2-FROZEN — Mass Model Tests

Verify baryonic mass model against analytic limits.
"""

import math

import numpy as np
import pytest

from src.constants import G_INTERNAL
from src.mass_model import BaryonicMassModel, gas_enclosed_mass


class TestExponentialDisk:
    def test_zero_at_origin(self):
        """Enclosed mass is 0 at r=0."""
        model = BaryonicMassModel(M_disk=1e11, R_disk=3.0, M_bulge=0, R_bulge=0)
        assert model.enclosed_mass_disk(0.0) == 0.0

    def test_asymptotic_total(self):
        """At large r, enclosed mass → M_disk."""
        model = BaryonicMassModel(M_disk=1e11, R_disk=3.0, M_bulge=0, R_bulge=0)
        M_enc = model.enclosed_mass_disk(300.0)  # 100 scale lengths
        assert abs(M_enc / 1e11 - 1.0) < 1e-6

    def test_half_mass_radius(self):
        """
        For exponential disk, half-mass radius ≈ 1.678 * R_d.
        M(<1.678*R_d) / M_total ≈ 0.5.
        """
        R_d = 3.0
        model = BaryonicMassModel(M_disk=1e11, R_disk=R_d, M_bulge=0, R_bulge=0)
        r_half = 1.678 * R_d
        fraction = model.enclosed_mass_disk(r_half) / 1e11
        assert abs(fraction - 0.5) < 0.01

    def test_monotonic(self):
        """Enclosed mass always increases with radius."""
        model = BaryonicMassModel(M_disk=1e11, R_disk=3.0, M_bulge=0, R_bulge=0)
        radii = np.linspace(0.1, 100.0, 200)
        masses = [model.enclosed_mass_disk(r) for r in radii]
        for i in range(len(masses) - 1):
            assert masses[i + 1] >= masses[i]


class TestHernquistBulge:
    def test_zero_at_origin(self):
        """Enclosed mass is 0 at r=0."""
        model = BaryonicMassModel(M_disk=0, R_disk=3.0, M_bulge=1e10, R_bulge=0.5)
        assert model.enclosed_mass_bulge(0.0) == 0.0

    def test_asymptotic_total(self):
        """At large r, enclosed mass → M_bulge."""
        model = BaryonicMassModel(M_disk=0, R_disk=3.0, M_bulge=1e10, R_bulge=0.5)
        M_enc = model.enclosed_mass_bulge(500.0)  # 1000 scale lengths
        assert abs(M_enc / 1e10 - 1.0) < 0.005

    def test_no_bulge(self):
        """Zero bulge mass/scale → zero contribution."""
        model = BaryonicMassModel(M_disk=1e11, R_disk=3.0, M_bulge=0, R_bulge=0)
        assert model.enclosed_mass_bulge(10.0) == 0.0


class TestGasEnclosedMass:
    def test_consistency(self):
        """M_gas = v_gas^2 * r / G, verified numerically."""
        r = 10.0  # kpc
        v_gas = 50.0  # km/s
        M = gas_enclosed_mass(r, v_gas)
        # Reverse: v = sqrt(G * M / r)
        v_check = math.sqrt(G_INTERNAL * M / r)
        assert abs(v_check - v_gas) < 1e-10

    def test_zero_gas(self):
        """No gas velocity → no gas mass."""
        assert gas_enclosed_mass(10.0, 0.0) == 0.0
