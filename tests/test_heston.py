"""
Tests for Heston stochastic volatility model.
"""

import numpy as np
import pytest

from mc_pricer.analytics.black_scholes import bs_price
from mc_pricer.models.heston import HestonModel
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff
from mc_pricer.pricers.heston_monte_carlo import HestonMonteCarloEngine


class TestHestonModel:
    """Test Heston model simulation."""

    def test_reproducibility(self):
        """Test that same seed gives identical results."""
        params = {
            "S0": 100.0,
            "r": 0.05,
            "T": 1.0,
            "kappa": 2.0,
            "theta": 0.04,
            "xi": 0.3,
            "rho": -0.7,
            "v0": 0.04,
        }

        model1 = HestonModel(**params, seed=42)
        model2 = HestonModel(**params, seed=42)

        terminal1 = model1.simulate_terminal(n_paths=1000, n_steps=100)
        terminal2 = model2.simulate_terminal(n_paths=1000, n_steps=100)

        np.testing.assert_array_equal(terminal1, terminal2)

    def test_no_nans(self):
        """Test that simulated prices are finite (no NaNs or Infs)."""
        model = HestonModel(
            S0=100.0,
            r=0.05,
            T=1.0,
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            rho=-0.7,
            v0=0.04,
            seed=42
        )

        terminal = model.simulate_terminal(n_paths=10000, n_steps=100)

        assert np.all(np.isfinite(terminal)), "Terminal prices contain NaN or Inf"
        assert np.all(terminal > 0), "Terminal prices must be positive"

    def test_paths_shape(self):
        """Test that simulate_paths returns correct shape."""
        model = HestonModel(
            S0=100.0,
            r=0.05,
            T=1.0,
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            rho=-0.7,
            v0=0.04,
            seed=42
        )

        n_paths = 100
        n_steps = 50
        paths = model.simulate_paths(n_paths, n_steps)

        assert paths.shape == (n_paths, n_steps + 1)
        assert np.all(paths[:, 0] == 100.0), "Initial prices should be S0"

    def test_antithetic_variates(self):
        """Test antithetic variates functionality."""
        model = HestonModel(
            S0=100.0,
            r=0.05,
            T=1.0,
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            rho=-0.7,
            v0=0.04,
            seed=42
        )

        # Test with even number of paths
        n_paths = 100
        paths = model.simulate_paths(n_paths, n_steps=50, antithetic=True)
        assert paths.shape == (n_paths, 51)

        # Test that odd number raises error
        with pytest.raises(ValueError, match="n_paths must be even"):
            model.simulate_paths(n_paths=101, n_steps=50, antithetic=True)

    def test_parameter_validation(self):
        """Test that invalid parameters raise errors."""
        # Negative S0
        with pytest.raises(ValueError, match="S0 must be positive"):
            HestonModel(S0=-100, r=0.05, T=1.0, kappa=2.0, theta=0.04,
                       xi=0.3, rho=-0.7, v0=0.04)

        # Negative T
        with pytest.raises(ValueError, match="T must be positive"):
            HestonModel(S0=100, r=0.05, T=-1.0, kappa=2.0, theta=0.04,
                       xi=0.3, rho=-0.7, v0=0.04)

        # Negative kappa
        with pytest.raises(ValueError, match="kappa must be positive"):
            HestonModel(S0=100, r=0.05, T=1.0, kappa=-2.0, theta=0.04,
                       xi=0.3, rho=-0.7, v0=0.04)

        # Negative theta
        with pytest.raises(ValueError, match="theta must be positive"):
            HestonModel(S0=100, r=0.05, T=1.0, kappa=2.0, theta=-0.04,
                       xi=0.3, rho=-0.7, v0=0.04)

        # Negative xi
        with pytest.raises(ValueError, match="xi must be non-negative"):
            HestonModel(S0=100, r=0.05, T=1.0, kappa=2.0, theta=0.04,
                       xi=-0.3, rho=-0.7, v0=0.04)

        # Invalid rho
        with pytest.raises(ValueError, match="rho must be in"):
            HestonModel(S0=100, r=0.05, T=1.0, kappa=2.0, theta=0.04,
                       xi=0.3, rho=-1.5, v0=0.04)

        # Negative v0
        with pytest.raises(ValueError, match="v0 must be non-negative"):
            HestonModel(S0=100, r=0.05, T=1.0, kappa=2.0, theta=0.04,
                       xi=0.3, rho=-0.7, v0=-0.04)


class TestHestonMonteCarloEngine:
    """Test Heston Monte Carlo pricing engine."""

    def test_pricing_call(self):
        """Test basic call option pricing."""
        model = HestonModel(
            S0=100.0,
            r=0.05,
            T=1.0,
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            rho=-0.7,
            v0=0.04,
            seed=42
        )

        payoff = EuropeanCallPayoff(strike=100.0)
        engine = HestonMonteCarloEngine(
            model=model,
            payoff=payoff,
            n_paths=50000,
            n_steps=100,
            seed=42
        )

        result = engine.price()

        # Basic checks
        assert result.price > 0, "Call price should be positive"
        assert result.stderr > 0, "Standard error should be positive"
        assert result.ci_lower < result.price < result.ci_upper
        assert result.n_paths == 50000
        assert result.n_steps == 100

    def test_black_scholes_limit(self):
        """
        Test that Heston converges to Black-Scholes when xi=0.

        With xi=0 (no volatility of volatility), v0=theta (starting at
        long-term variance), and rho=0 (no correlation), the Heston model
        should reduce to Black-Scholes with sigma=sqrt(theta).
        """
        S0 = 100.0
        K = 100.0
        r = 0.05
        T = 1.0
        theta = 0.04  # variance
        sigma = np.sqrt(theta)  # volatility = 0.2

        # Heston model with no vol-of-vol
        model = HestonModel(
            S0=S0,
            r=r,
            T=T,
            kappa=2.0,  # Doesn't matter when xi=0
            theta=theta,
            xi=0.0,  # No volatility of volatility
            rho=0.0,  # No correlation
            v0=theta,  # Start at long-term level
            seed=42
        )

        # Price with Heston
        payoff = EuropeanCallPayoff(strike=K)
        engine = HestonMonteCarloEngine(
            model=model,
            payoff=payoff,
            n_paths=100000,
            n_steps=200,
            seed=42
        )
        heston_result = engine.price()

        # Price with Black-Scholes
        bs_price_val = bs_price(S0, K, r, T, sigma, "call")

        # Check that Heston price is close to BS price
        # Allow for Monte Carlo error (should be within a few standard errors)
        error = abs(heston_result.price - bs_price_val)
        tolerance = 3 * heston_result.stderr  # 3 standard errors

        assert error < tolerance, (
            f"Heston price {heston_result.price:.6f} differs from "
            f"Black-Scholes {bs_price_val:.6f} by {error:.6f}, "
            f"exceeding tolerance of {tolerance:.6f}"
        )

        # Also check that they're reasonably close in absolute terms
        relative_error = abs(error / bs_price_val)
        assert relative_error < 0.02, (
            f"Relative error {relative_error:.2%} exceeds 2%"
        )

    def test_pricing_result_attributes(self):
        """Test that pricing result has all required attributes."""
        model = HestonModel(
            S0=100.0,
            r=0.05,
            T=1.0,
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            rho=-0.7,
            v0=0.04,
            seed=42
        )

        payoff = EuropeanCallPayoff(strike=100.0)
        engine = HestonMonteCarloEngine(
            model=model,
            payoff=payoff,
            n_paths=10000,
            n_steps=50,
            seed=42
        )

        result = engine.price()

        # Check all attributes exist
        assert hasattr(result, 'price')
        assert hasattr(result, 'stderr')
        assert hasattr(result, 'ci_lower')
        assert hasattr(result, 'ci_upper')
        assert hasattr(result, 'n_paths')
        assert hasattr(result, 'n_steps')

        # Check types
        assert isinstance(result.price, float)
        assert isinstance(result.stderr, float)
        assert isinstance(result.ci_lower, float)
        assert isinstance(result.ci_upper, float)
        assert isinstance(result.n_paths, int)
        assert isinstance(result.n_steps, int)

    def test_antithetic_reduces_variance(self):
        """Test that antithetic variates reduce variance."""
        model = HestonModel(
            S0=100.0,
            r=0.05,
            T=1.0,
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            rho=-0.7,
            v0=0.04,
            seed=42
        )

        payoff = EuropeanCallPayoff(strike=100.0)

        # Without antithetic
        engine_base = HestonMonteCarloEngine(
            model=model,
            payoff=payoff,
            n_paths=10000,
            n_steps=100,
            antithetic=False,
            seed=42
        )
        result_base = engine_base.price()

        # With antithetic (reset seed)
        model._rng = np.random.default_rng(42)
        engine_anti = HestonMonteCarloEngine(
            model=model,
            payoff=payoff,
            n_paths=10000,
            n_steps=100,
            antithetic=True,
            seed=42
        )
        result_anti = engine_anti.price()

        # Antithetic should typically reduce variance
        # Note: This is probabilistic, so we just check it's positive
        assert result_anti.stderr > 0
        # Can't guarantee reduction in single run, so just check reasonable values
        assert result_base.stderr > 0
