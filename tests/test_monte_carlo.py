"""
Unit tests for Monte Carlo pricing engine.
"""

import numpy as np
import pytest

from mc_pricer.models.gbm import GeometricBrownianMotion
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff, EuropeanPutPayoff
from mc_pricer.pricers.monte_carlo import MonteCarloEngine


class TestMonteCarloEngine:
    """Test suite for Monte Carlo pricing engine."""

    def test_initialization(self):
        """Test engine initialization."""
        model = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0, seed=42)
        payoff = EuropeanCallPayoff(strike=100)
        engine = MonteCarloEngine(model=model, payoff=payoff, n_paths=10000)

        assert engine.n_paths == 10000
        assert engine.antithetic is False

    def test_invalid_n_paths(self):
        """Test that non-positive n_paths raises ValueError."""
        model = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0)
        payoff = EuropeanCallPayoff(strike=100)

        with pytest.raises(ValueError, match="n_paths must be positive"):
            MonteCarloEngine(model=model, payoff=payoff, n_paths=0)

        with pytest.raises(ValueError, match="n_paths must be positive"):
            MonteCarloEngine(model=model, payoff=payoff, n_paths=-1000)

    def test_call_price_positive(self):
        """Test that call option price is positive."""
        model = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0, seed=42)
        payoff = EuropeanCallPayoff(strike=100)
        engine = MonteCarloEngine(model=model, payoff=payoff, n_paths=10000)

        result = engine.price()
        assert result.price > 0

    def test_put_price_positive(self):
        """Test that put option price is positive."""
        model = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0, seed=42)
        payoff = EuropeanPutPayoff(strike=100)
        engine = MonteCarloEngine(model=model, payoff=payoff, n_paths=10000)

        result = engine.price()
        assert result.price > 0

    def test_confidence_interval(self):
        """Test that confidence interval bounds are reasonable."""
        model = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0, seed=42)
        payoff = EuropeanCallPayoff(strike=100)
        engine = MonteCarloEngine(model=model, payoff=payoff, n_paths=10000)

        result = engine.price()

        # CI bounds should bracket the price
        assert result.ci_lower < result.price < result.ci_upper

        # Width of CI should be approximately 2 * 1.96 * stderr
        ci_width = result.ci_upper - result.ci_lower
        expected_width = 2 * 1.96 * result.stderr
        assert np.abs(ci_width - expected_width) < 1e-6

    def test_call_monotonicity_in_spot(self):
        """Test that call price increases with spot price."""
        payoff = EuropeanCallPayoff(strike=100)

        prices = []
        for S0 in [80, 90, 100, 110, 120]:
            model = GeometricBrownianMotion(S0=S0, r=0.05, sigma=0.2, T=1.0, seed=42)
            engine = MonteCarloEngine(model=model, payoff=payoff, n_paths=50000)
            result = engine.price()
            prices.append(result.price)

        # Check monotonicity
        assert np.all(np.diff(prices) > 0)

    def test_put_monotonicity_in_spot(self):
        """Test that put price decreases with spot price."""
        payoff = EuropeanPutPayoff(strike=100)

        prices = []
        for S0 in [80, 90, 100, 110, 120]:
            model = GeometricBrownianMotion(S0=S0, r=0.05, sigma=0.2, T=1.0, seed=42)
            engine = MonteCarloEngine(model=model, payoff=payoff, n_paths=50000)
            result = engine.price()
            prices.append(result.price)

        # Check monotonicity (decreasing)
        assert np.all(np.diff(prices) < 0)

    def test_deep_itm_call(self):
        """Test deep in-the-money call approximates intrinsic value."""
        model = GeometricBrownianMotion(S0=150, r=0.05, sigma=0.2, T=1.0, seed=42)
        payoff = EuropeanCallPayoff(strike=100)
        engine = MonteCarloEngine(model=model, payoff=payoff, n_paths=50000)

        result = engine.price()

        # Deep ITM call should be close to S0 - K * exp(-rT)
        intrinsic = 150 - 100 * np.exp(-0.05 * 1.0)

        # Should be within 5% of intrinsic value
        assert np.abs(result.price - intrinsic) / intrinsic < 0.05

    def test_deep_otm_call(self):
        """Test deep out-of-the-money call has small price."""
        model = GeometricBrownianMotion(S0=50, r=0.05, sigma=0.2, T=1.0, seed=42)
        payoff = EuropeanCallPayoff(strike=100)
        engine = MonteCarloEngine(model=model, payoff=payoff, n_paths=50000)

        result = engine.price()

        # Deep OTM call should have small value
        assert result.price < 1.0

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        model1 = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0, seed=42)
        payoff = EuropeanCallPayoff(strike=100)
        engine1 = MonteCarloEngine(model=model1, payoff=payoff, n_paths=10000, seed=42)
        result1 = engine1.price()

        model2 = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0, seed=42)
        engine2 = MonteCarloEngine(model=model2, payoff=payoff, n_paths=10000, seed=42)
        result2 = engine2.price()

        assert result1.price == result2.price

    def test_price_with_details(self):
        """Test price_with_details returns correct outputs."""
        model = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0, seed=42)
        payoff = EuropeanCallPayoff(strike=100)
        engine = MonteCarloEngine(model=model, payoff=payoff, n_paths=1000)

        result, payoffs = engine.price_with_details()

        assert len(payoffs) == 1000
        assert result.price == np.mean(payoffs)
        assert np.all(payoffs >= 0)
