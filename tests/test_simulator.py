"""
MetaLaw Discovery System v0.2-FROZEN — Simulator Tests

Verify rotation curve simulation against analytic cases.
"""

import math

import numpy as np
import pytest

from src.constants import G_INTERNAL
from src.laws.newtonian import NewtonianLaw
from src.mass_model import BaryonicMassModel
from src.simulator import simulate_rotation_curve, _compute_smoothness
from src.types import GalaxyData


def _make_galaxy(
    radii: np.ndarray,
    v_gas: np.ndarray | None = None,
) -> GalaxyData:
    """Helper to create a minimal GalaxyData for testing."""
    n = len(radii)
    if v_gas is None:
        v_gas = np.zeros(n)
    return GalaxyData(
        name="test",
        radii=radii,
        v_obs=np.zeros(n),  # not used in simulation
        v_err=np.ones(n),
        v_gas=v_gas,
        luminosity_disk=1e10,
        luminosity_bulge=0.0,
        scale_length_disk=3.0,
        scale_length_bulge=0.0,
        distance=10.0,
        inclination=60.0,
        quality=1,
    )


class TestKeplerianRecovery:
    """
    For a point-like mass (very small scale length), Newtonian gravity
    should produce v(r) ∝ 1/sqrt(r) (Keplerian decline).
    """

    def test_keplerian_falloff(self):
        """v(r) ∝ r^{-0.5} at r >> R_d."""
        radii = np.linspace(20.0, 80.0, 30)
        galaxy = _make_galaxy(radii)

        # Very compact mass (R_d = 0.1 kpc) → effectively point mass at large r
        mass_model = BaryonicMassModel(M_disk=1e11, R_disk=0.1, M_bulge=0, R_bulge=0)
        law = NewtonianLaw()

        result = simulate_rotation_curve(law, mass_model, galaxy)

        # v should scale as r^{-0.5}
        # Check: v(r2)/v(r1) ≈ sqrt(r1/r2)
        for i in range(len(radii) - 1):
            v_ratio = result.v_pred[i + 1] / result.v_pred[i]
            r_ratio = math.sqrt(radii[i] / radii[i + 1])
            assert abs(v_ratio / r_ratio - 1.0) < 0.02

    def test_absolute_velocity(self):
        """v = sqrt(G*M/r) for point mass."""
        r = 30.0
        galaxy = _make_galaxy(np.array([r]))
        mass_model = BaryonicMassModel(M_disk=1e11, R_disk=0.01, M_bulge=0, R_bulge=0)
        law = NewtonianLaw()

        result = simulate_rotation_curve(law, mass_model, galaxy)

        v_expected = math.sqrt(G_INTERNAL * 1e11 / r)
        assert abs(result.v_pred[0] - v_expected) / v_expected < 0.01


class TestSmoothness:
    def test_monotonic_curve(self):
        """Monotonically increasing curve has smoothness = 1."""
        v = np.linspace(50, 200, 50)
        assert _compute_smoothness(v) == 1.0

    def test_flat_curve(self):
        """Flat curve has smoothness = 1."""
        v = np.full(50, 150.0)
        assert _compute_smoothness(v) == 1.0

    def test_oscillatory_curve(self):
        """Highly oscillatory curve has low smoothness."""
        v = np.array([100, 200, 100, 200, 100, 200, 100, 200, 100, 200])
        s = _compute_smoothness(v)
        assert s < 0.5

    def test_short_array(self):
        """Arrays with < 3 points return 1.0."""
        assert _compute_smoothness(np.array([100.0, 200.0])) == 1.0
        assert _compute_smoothness(np.array([100.0])) == 1.0
