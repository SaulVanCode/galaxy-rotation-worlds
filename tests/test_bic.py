"""
MetaLaw Discovery System v0.2-FROZEN — BIC Scoring Tests

Verify BIC computation against hand-calculated values.
"""

import math

import numpy as np
import pytest

from src.scoring.bic import compute_bic, compute_chi2, compute_chi2_reduced


class TestChi2:
    def test_perfect_fit(self):
        """Zero residuals → chi2 = 0."""
        v_obs = np.array([100.0, 150.0, 180.0])
        v_pred = np.array([100.0, 150.0, 180.0])
        v_err = np.array([5.0, 5.0, 5.0])
        assert compute_chi2(v_obs, v_pred, v_err) == 0.0

    def test_known_value(self):
        """Hand-calculated chi2."""
        v_obs = np.array([100.0, 150.0])
        v_pred = np.array([105.0, 145.0])
        v_err = np.array([5.0, 10.0])
        # residuals: (100-105)/5 = -1, (150-145)/10 = 0.5
        # chi2 = 1 + 0.25 = 1.25
        assert abs(compute_chi2(v_obs, v_pred, v_err) - 1.25) < 1e-10

    def test_scales_with_error(self):
        """Doubling errors quarters chi2."""
        v_obs = np.array([100.0])
        v_pred = np.array([110.0])
        chi2_small = compute_chi2(v_obs, v_pred, np.array([5.0]))
        chi2_large = compute_chi2(v_obs, v_pred, np.array([10.0]))
        assert abs(chi2_large / chi2_small - 0.25) < 1e-10


class TestBIC:
    def test_known_value(self):
        """BIC = chi2 + k*ln(n), hand-calculated."""
        chi2 = 10.0
        k = 3
        n = 50
        expected = 10.0 + 3 * math.log(50)
        assert abs(compute_bic(chi2, k, n) - expected) < 1e-10

    def test_penalizes_more_params(self):
        """More parameters → higher BIC for same chi2."""
        chi2 = 10.0
        n = 50
        bic_2 = compute_bic(chi2, k=2, n=n)
        bic_4 = compute_bic(chi2, k=4, n=n)
        assert bic_4 > bic_2

    def test_rewards_better_fit(self):
        """Lower chi2 → lower BIC for same params."""
        n = 50
        k = 2
        bic_good = compute_bic(5.0, k, n)
        bic_bad = compute_bic(20.0, k, n)
        assert bic_good < bic_bad


class TestChi2Reduced:
    def test_known_value(self):
        """chi2_red = chi2 / (n-k)."""
        assert abs(compute_chi2_reduced(10.0, 2, 12) - 1.0) < 1e-10

    def test_underdetermined(self):
        """n <= k → inf."""
        assert compute_chi2_reduced(10.0, 5, 5) == float("inf")
        assert compute_chi2_reduced(10.0, 6, 5) == float("inf")
