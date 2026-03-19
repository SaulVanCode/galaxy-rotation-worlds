"""
MetaLaw Discovery System v0.2-FROZEN — Nesting Tests

Verify that PiecewiseRegime strictly nests known physics:
1. Newton is a special case (alpha_inner=1, alpha_outer=1)
2. MOND is a special case (alpha_inner=1, alpha_outer=0.5)

These tests use full rotation curve simulation, not just single-point checks.
"""

import math

import numpy as np
import pytest

from src.constants import A0_INTERNAL, G_INTERNAL
from src.laws.newtonian import NewtonianLaw
from src.laws.mond_simple import SimpleMONDLaw
from src.laws.piecewise_regime import PiecewiseRegimeLaw


# Test over a range of radii and masses representative of real galaxies
RADII = np.linspace(0.5, 50.0, 100)  # kpc
M_ENCLOSED = 5e10 * (1 - (1 + RADII / 3.0) * np.exp(-RADII / 3.0))  # exponential disk, R_d=3 kpc
RHO = 1e6 * np.exp(-RADII / 3.0)  # approximate density


class TestNewtonianNesting:
    """PiecewiseRegime(alpha_inner=1, alpha_outer=1) ≡ Newton."""

    @pytest.mark.parametrize("w", [0.1, 1.0, 3.0, 5.0])
    def test_matches_newton_all_radii(self, w):
        """Full curve match within 0.1% at all radii."""
        newton = NewtonianLaw()
        piecewise = PiecewiseRegimeLaw(
            a0=A0_INTERNAL, alpha_inner=1.0, alpha_outer=1.0, w=w
        )

        for i, r in enumerate(RADII):
            a_newton = newton.acceleration(r, M_ENCLOSED[i], RHO[i])
            a_piecewise = piecewise.acceleration(r, M_ENCLOSED[i], RHO[i])
            rel_err = abs(a_piecewise - a_newton) / a_newton
            assert rel_err < 0.001, (
                f"Nesting violated at r={r:.1f} kpc, w={w}: "
                f"Newton={a_newton:.6e}, Piecewise={a_piecewise:.6e}, "
                f"rel_err={rel_err:.6e}"
            )


class TestMONDNesting:
    """
    PiecewiseRegime(alpha_inner=1, alpha_outer=0.5, w→small) ≈ MOND
    in the deep-MOND regime (a_N << a0).
    """

    def test_deep_mond_regime(self):
        """
        In the deep MOND limit (a_N << a0), both PiecewiseRegime(alpha_outer=0.5)
        and SimpleMOND converge to the same asymptote: a = sqrt(a_N * a0).

        Note: PiecewiseRegime uses a sigmoid transition while MOND uses
        nu(x) = 1/(1-exp(-sqrt(x))). These are DIFFERENT interpolating functions
        that share the same deep-MOND and Newtonian limits. We test the shared
        asymptotic limit, not exact functional match in the transition zone.
        """
        piecewise = PiecewiseRegimeLaw(
            a0=A0_INTERNAL, alpha_inner=1.0, alpha_outer=0.5, w=0.1
        )

        # Very low mass at large radius → a_N/a0 < 0.001 (truly deep MOND)
        M_low = 1e5  # M_sun
        radii_outer = np.linspace(50.0, 200.0, 30)

        tested = 0
        for r in radii_outer:
            M_enc = M_low * (1 - (1 + r / 2.0) * math.exp(-r / 2.0))
            if M_enc < 1.0:
                continue
            rho = 1e2 * math.exp(-r / 2.0)

            a_N = G_INTERNAL * M_enc / r**2

            # Only test deep MOND: a_N/a0 < 0.001
            if a_N > 0.001 * A0_INTERNAL:
                continue

            # Asymptotic prediction: a = sqrt(a_N * a0)
            a_asymptotic = math.sqrt(a_N * A0_INTERNAL)
            a_piecewise = piecewise.acceleration(r, M_enc, rho)

            rel_err = abs(a_piecewise - a_asymptotic) / a_asymptotic
            assert rel_err < 0.05, (
                f"Deep MOND asymptote not recovered at r={r:.1f} kpc: "
                f"asymptotic={a_asymptotic:.6e}, Piecewise={a_piecewise:.6e}, "
                f"rel_err={rel_err:.4f}"
            )
            tested += 1

        assert tested >= 5, f"Only tested {tested} points in deep MOND regime"

    def test_newtonian_regime_with_mond_params(self):
        """
        Even with alpha_outer=0.5, at high acceleration (inner galaxy),
        PiecewiseRegime with alpha_inner=1 → Newtonian.
        """
        newton = NewtonianLaw()
        piecewise = PiecewiseRegimeLaw(
            a0=A0_INTERNAL, alpha_inner=1.0, alpha_outer=0.5, w=0.1
        )

        # High mass, inner radius → a_N >> a0
        M_high = 1e12
        r_inner = 1.0
        rho = 1e8

        a_N = G_INTERNAL * M_high / r_inner**2
        assert a_N > 10 * A0_INTERNAL, "Test setup: should be Newtonian regime"

        a_newton = newton.acceleration(r_inner, M_high, rho)
        a_piecewise = piecewise.acceleration(r_inner, M_high, rho)

        rel_err = abs(a_piecewise - a_newton) / a_newton
        assert rel_err < 0.01, (
            f"Newtonian regime not recovered: "
            f"Newton={a_newton:.6e}, Piecewise={a_piecewise:.6e}"
        )
