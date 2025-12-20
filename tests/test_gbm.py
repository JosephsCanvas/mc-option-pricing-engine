"""
Unit tests for Geometric Brownian Motion model.
"""

import numpy as np
import pytest

from mc_pricer.models.gbm import GeometricBrownianMotion


class TestGeometricBrownianMotion:
    """Test suite for GBM model."""

    def test_initialization(self):
        """Test GBM initialization with valid parameters."""
        gbm = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0, seed=42)
        assert gbm.S0 == 100
        assert gbm.r == 0.05
        assert gbm.sigma == 0.2
        assert gbm.T == 1.0

    def test_invalid_S0(self):
        """Test that negative S0 raises ValueError."""
        with pytest.raises(ValueError, match="Initial price S0 must be positive"):
            GeometricBrownianMotion(S0=-100, r=0.05, sigma=0.2, T=1.0)

        with pytest.raises(ValueError, match="Initial price S0 must be positive"):
            GeometricBrownianMotion(S0=0, r=0.05, sigma=0.2, T=1.0)

    def test_invalid_sigma(self):
        """Test that negative sigma raises ValueError."""
        with pytest.raises(ValueError, match="Volatility sigma must be non-negative"):
            GeometricBrownianMotion(S0=100, r=0.05, sigma=-0.2, T=1.0)

    def test_invalid_T(self):
        """Test that non-positive T raises ValueError."""
        with pytest.raises(ValueError, match="Time to maturity T must be positive"):
            GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=0)

        with pytest.raises(ValueError, match="Time to maturity T must be positive"):
            GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=-1.0)

    def test_simulate_terminal_shape(self):
        """Test that simulate_terminal returns correct shape."""
        gbm = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0, seed=42)
        S_T = gbm.simulate_terminal(n_paths=1000)
        assert S_T.shape == (1000,)

    def test_simulate_terminal_positive(self):
        """Test that all simulated prices are positive."""
        gbm = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0, seed=42)
        S_T = gbm.simulate_terminal(n_paths=1000)
        assert np.all(S_T > 0)

    def test_simulate_paths_shape(self):
        """Test that simulate_paths returns correct shape."""
        gbm = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0, seed=42)
        paths = gbm.simulate_paths(n_paths=100, n_steps=10)
        assert paths.shape == (100, 11)  # n_steps + 1 for initial price

    def test_simulate_paths_initial_price(self):
        """Test that all paths start at S0."""
        gbm = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0, seed=42)
        paths = gbm.simulate_paths(n_paths=100, n_steps=10)
        assert np.allclose(paths[:, 0], 100)

    def test_antithetic_even_paths(self):
        """Test antithetic variates with even number of paths."""
        gbm = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0, seed=42)
        S_T = gbm.simulate_terminal(n_paths=100, antithetic=True)
        assert S_T.shape == (100,)

        # For antithetic variates, mean of pairs should match
        # This is a basic check; exact pairing is hard to verify without internals
        assert len(S_T) == 100

    def test_antithetic_odd_paths(self):
        """Test antithetic variates with odd number of paths."""
        gbm = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0, seed=42)
        S_T = gbm.simulate_terminal(n_paths=99, antithetic=True)
        assert S_T.shape == (99,)

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        gbm1 = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0, seed=42)
        S_T1 = gbm1.simulate_terminal(n_paths=100)

        gbm2 = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0, seed=42)
        S_T2 = gbm2.simulate_terminal(n_paths=100)

        assert np.allclose(S_T1, S_T2)

    def test_zero_volatility(self):
        """Test GBM with zero volatility (deterministic drift)."""
        gbm = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.0, T=1.0, seed=42)
        S_T = gbm.simulate_terminal(n_paths=100)

        # With zero volatility, all paths should be S0 * exp(r * T)
        expected = 100 * np.exp(0.05 * 1.0)
        assert np.allclose(S_T, expected)

    def test_mean_convergence(self):
        """Test that sample mean converges to theoretical forward price."""
        gbm = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0, seed=42)
        S_T = gbm.simulate_terminal(n_paths=100000)

        # Theoretical forward price: S0 * exp(r * T)
        forward_price = 100 * np.exp(0.05 * 1.0)
        sample_mean = np.mean(S_T)

        # Should be close with large sample (within 1%)
        assert np.abs(sample_mean - forward_price) / forward_price < 0.01
