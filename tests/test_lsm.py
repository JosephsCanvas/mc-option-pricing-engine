"""
Tests for Longstaff-Schwartz (LSM) American option pricing.
"""

import pytest

from mc_pricer.models.gbm import GeometricBrownianMotion
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff, EuropeanPutPayoff
from mc_pricer.pricers.lsm import price_american_lsm
from mc_pricer.pricers.monte_carlo import MonteCarloEngine


class TestLSMBasics:
    """Test basic LSM functionality and input validation."""

    def test_american_put_pricing(self):
        """Test that American put can be priced."""
        model = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0, seed=42)
        result = price_american_lsm(
            model=model,
            strike=100,
            option_type="put",
            n_paths=10000,
            n_steps=50,
            basis="poly2",
            seed=42,
        )
        assert result.price > 0
        assert result.stderr > 0
        assert result.ci_lower < result.price < result.ci_upper
        assert result.n_paths == 10000
        assert result.n_steps == 50
        assert result.basis == "poly2"

    def test_american_call_pricing(self):
        """Test that American call can be priced."""
        model = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0, seed=42)
        result = price_american_lsm(
            model=model,
            strike=100,
            option_type="call",
            n_paths=10000,
            n_steps=50,
            basis="poly2",
            seed=42,
        )
        assert result.price > 0
        assert result.stderr > 0
        assert result.ci_lower < result.price < result.ci_upper

    def test_poly3_basis(self):
        """Test pricing with poly3 basis functions."""
        model = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0, seed=42)
        result = price_american_lsm(
            model=model,
            strike=100,
            option_type="put",
            n_paths=10000,
            n_steps=50,
            basis="poly3",
            seed=42,
        )
        assert result.price > 0
        assert result.basis == "poly3"

    def test_invalid_strike(self):
        """Test that invalid strike raises error."""
        model = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0, seed=42)
        with pytest.raises(ValueError, match="Strike must be positive"):
            price_american_lsm(
                model=model, strike=-100, option_type="put", n_paths=1000, n_steps=50
            )

    def test_invalid_option_type(self):
        """Test that invalid option type raises error."""
        model = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0, seed=42)
        with pytest.raises(ValueError, match="option_type must be"):
            price_american_lsm(
                model=model, strike=100, option_type="invalid", n_paths=1000, n_steps=50
            )

    def test_invalid_basis(self):
        """Test that invalid basis raises error."""
        model = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0, seed=42)
        with pytest.raises(ValueError, match="basis must be"):
            price_american_lsm(
                model=model,
                strike=100,
                option_type="put",
                n_paths=1000,
                n_steps=50,
                basis="invalid",
            )


class TestAmericanVsEuropean:
    """Test American vs European option relationships."""

    def test_american_put_geq_european_put(self):
        """Test that American put >= European put (early exercise premium)."""
        S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
        seed = 42
        n_paths = 20000
        n_steps = 50

        # European put
        model_euro = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        payoff_euro = EuropeanPutPayoff(strike=K)
        engine_euro = MonteCarloEngine(
            model=model_euro,
            payoff=payoff_euro,
            n_paths=n_paths,
            antithetic=False,
            control_variate=False,
        )
        result_euro = engine_euro.price()

        # American put
        model_amer = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        result_amer = price_american_lsm(
            model=model_amer,
            strike=K,
            option_type="put",
            n_paths=n_paths,
            n_steps=n_steps,
            basis="poly2",
            seed=seed,
        )

        # American should be >= European (allowing small negative tolerance for MC error)
        assert result_amer.price >= result_euro.price - 1e-3, (
            f"American put ({result_amer.price:.6f}) should be >= "
            f"European put ({result_euro.price:.6f})"
        )

    def test_american_call_approx_european_call_no_dividend(self):
        """Test that American call ≈ European call in no-dividend GBM."""
        S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
        seed = 42
        n_paths = 20000
        n_steps = 50

        # European call
        model_euro = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        payoff_euro = EuropeanCallPayoff(strike=K)
        engine_euro = MonteCarloEngine(
            model=model_euro,
            payoff=payoff_euro,
            n_paths=n_paths,
            antithetic=False,
            control_variate=False,
        )
        result_euro = engine_euro.price()

        # American call
        model_amer = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        result_amer = price_american_lsm(
            model=model_amer,
            strike=K,
            option_type="call",
            n_paths=n_paths,
            n_steps=n_steps,
            basis="poly2",
            seed=seed,
        )

        # Should be approximately equal (within 3*stderr + small constant)
        diff = abs(result_amer.price - result_euro.price)
        tolerance = 3 * max(result_amer.stderr, result_euro.stderr) + 0.05
        assert diff < tolerance, (
            f"American call ({result_amer.price:.6f}) and "
            f"European call ({result_euro.price:.6f}) differ by {diff:.6f}, "
            f"tolerance: {tolerance:.6f}"
        )


class TestMonotonicity:
    """Test monotonicity properties of American options."""

    def test_american_put_monotonic_in_spot(self):
        """Test that American put price decreases as spot increases."""
        K, r, sigma, T = 100, 0.05, 0.2, 1.0
        seed = 42
        n_paths = 20000
        n_steps = 50

        prices = []
        for S0 in [90, 100, 110]:
            model = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
            result = price_american_lsm(
                model=model,
                strike=K,
                option_type="put",
                n_paths=n_paths,
                n_steps=n_steps,
                basis="poly2",
                seed=seed,
            )
            prices.append(result.price)

        # Put price should decrease as spot increases: P(90) > P(100) > P(110)
        # Allow small tolerance for MC noise
        assert prices[0] > prices[1] - 0.1, (
            f"P(90)={prices[0]:.4f} should be > P(100)={prices[1]:.4f}"
        )
        assert prices[1] > prices[2] - 0.1, (
            f"P(100)={prices[1]:.4f} should be > P(110)={prices[2]:.4f}"
        )

    def test_american_call_monotonic_in_spot(self):
        """Test that American call price increases as spot increases."""
        K, r, sigma, T = 100, 0.05, 0.2, 1.0
        seed = 42
        n_paths = 20000
        n_steps = 50

        prices = []
        for S0 in [90, 100, 110]:
            model = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
            result = price_american_lsm(
                model=model,
                strike=K,
                option_type="call",
                n_paths=n_paths,
                n_steps=n_steps,
                basis="poly2",
                seed=seed,
            )
            prices.append(result.price)

        # Call price should increase as spot increases: C(90) < C(100) < C(110)
        # Allow small tolerance for MC noise
        assert prices[0] < prices[1] + 0.1, (
            f"C(90)={prices[0]:.4f} should be < C(100)={prices[1]:.4f}"
        )
        assert prices[1] < prices[2] + 0.1, (
            f"C(100)={prices[1]:.4f} should be < C(110)={prices[2]:.4f}"
        )


class TestReproducibility:
    """Test that results are reproducible with same seed."""

    def test_reproducible_with_seed(self):
        """Test that same seed produces same results."""
        model1 = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0, seed=42)
        result1 = price_american_lsm(
            model=model1,
            strike=100,
            option_type="put",
            n_paths=10000,
            n_steps=50,
            basis="poly2",
            seed=42,
        )

        model2 = GeometricBrownianMotion(S0=100, r=0.05, sigma=0.2, T=1.0, seed=42)
        result2 = price_american_lsm(
            model=model2,
            strike=100,
            option_type="put",
            n_paths=10000,
            n_steps=50,
            basis="poly2",
            seed=42,
        )

        assert result1.price == result2.price
        assert result1.stderr == result2.stderr
