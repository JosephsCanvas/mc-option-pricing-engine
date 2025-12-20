"""
Validation tests comparing Monte Carlo pricing against Black-Scholes.
"""

import numpy as np
import pytest

from mc_pricer.models.gbm import GeometricBrownianMotion
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff, EuropeanPutPayoff
from mc_pricer.pricers.monte_carlo import MonteCarloEngine

from .utils.black_scholes import black_scholes_call, black_scholes_put


class TestBlackScholesValidation:
    """Validate Monte Carlo pricing against Black-Scholes analytical prices."""

    @pytest.mark.parametrize("S0,K,r,sigma,T", [
        (100, 100, 0.05, 0.2, 1.0),  # ATM
        (100, 90, 0.05, 0.2, 1.0),   # ITM
        (100, 110, 0.05, 0.2, 1.0),  # OTM
        (120, 100, 0.03, 0.25, 0.5), # ITM, shorter maturity
        (80, 100, 0.02, 0.15, 2.0),  # OTM, longer maturity
    ])
    def test_call_vs_black_scholes(self, S0, K, r, sigma, T):
        """Test MC call price converges to Black-Scholes price."""
        # Black-Scholes analytical price
        bs_price = black_scholes_call(S0=S0, K=K, r=r, sigma=sigma, T=T)

        # Monte Carlo price with large sample
        model = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=42)
        payoff = EuropeanCallPayoff(strike=K)
        engine = MonteCarloEngine(model=model, payoff=payoff, n_paths=200000)
        mc_result = engine.price()

        # MC price should be within 3 standard errors of BS price
        assert np.abs(mc_result.price - bs_price) < 3 * mc_result.stderr

        # Also check relative error is small (within 2%)
        rel_error = np.abs(mc_result.price - bs_price) / bs_price
        assert rel_error < 0.02, f"Relative error {rel_error:.4f} too large"

    @pytest.mark.parametrize("S0,K,r,sigma,T", [
        (100, 100, 0.05, 0.2, 1.0),  # ATM
        (100, 110, 0.05, 0.2, 1.0),  # ITM
        (100, 90, 0.05, 0.2, 1.0),   # OTM
        (80, 100, 0.03, 0.25, 0.5),  # ITM, shorter maturity
        (120, 100, 0.02, 0.15, 2.0), # OTM, longer maturity
    ])
    def test_put_vs_black_scholes(self, S0, K, r, sigma, T):
        """Test MC put price converges to Black-Scholes price."""
        # Black-Scholes analytical price
        bs_price = black_scholes_put(S0=S0, K=K, r=r, sigma=sigma, T=T)

        # Monte Carlo price with large sample
        model = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=42)
        payoff = EuropeanPutPayoff(strike=K)
        engine = MonteCarloEngine(model=model, payoff=payoff, n_paths=200000)
        mc_result = engine.price()

        # MC price should be within 3 standard errors of BS price
        assert np.abs(mc_result.price - bs_price) < 3 * mc_result.stderr

        # Also check relative error is small (within 2%)
        rel_error = np.abs(mc_result.price - bs_price) / bs_price
        assert rel_error < 0.02, f"Relative error {rel_error:.4f} too large"

    def test_put_call_parity(self):
        """Test that MC pricing satisfies put-call parity."""
        S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0

        model_call = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=42)
        call_payoff = EuropeanCallPayoff(strike=K)
        call_engine = MonteCarloEngine(model=model_call, payoff=call_payoff, n_paths=200000)
        call_price = call_engine.price().price

        model_put = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=42)
        put_payoff = EuropeanPutPayoff(strike=K)
        put_engine = MonteCarloEngine(model=model_put, payoff=put_payoff, n_paths=200000)
        put_price = put_engine.price().price

        # Put-call parity: C - P = S0 - K * exp(-rT)
        parity_lhs = call_price - put_price
        parity_rhs = S0 - K * np.exp(-r * T)

        # Should hold within small tolerance
        assert np.abs(parity_lhs - parity_rhs) < 0.1


class TestAntitheticVarianceReduction:
    """Test antithetic variates variance reduction."""

    def test_antithetic_reduces_variance(self):
        """Test that antithetic variates reduce variance."""
        S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
        n_paths = 10000

        # Standard MC
        model_std = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=42)
        payoff = EuropeanCallPayoff(strike=K)
        engine_std = MonteCarloEngine(model=model_std, payoff=payoff, n_paths=n_paths)
        result_std = engine_std.price()

        # Antithetic MC
        model_anti = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=42)
        engine_anti = MonteCarloEngine(
            model=model_anti, payoff=payoff, n_paths=n_paths, antithetic=True
        )
        result_anti = engine_anti.price()

        # Antithetic should have smaller or equal standard error
        # (with same n_paths, antithetic should reduce variance)
        assert result_anti.stderr <= result_std.stderr * 1.1  # Allow 10% tolerance

    def test_antithetic_preserves_mean(self):
        """Test that antithetic variates preserve the mean estimate."""
        S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0

        # Black-Scholes analytical price
        bs_price = black_scholes_call(S0=S0, K=K, r=r, sigma=sigma, T=T)

        # Antithetic MC
        model = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=42)
        payoff = EuropeanCallPayoff(strike=K)
        engine = MonteCarloEngine(
            model=model, payoff=payoff, n_paths=200000, antithetic=True
        )
        result = engine.price()

        # Should still be close to BS price
        rel_error = np.abs(result.price - bs_price) / bs_price
        assert rel_error < 0.02

    def test_antithetic_confidence_interval(self):
        """Test that antithetic CI is narrower for same n_paths."""
        S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
        n_paths = 50000

        # Standard MC
        model_std = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=42)
        payoff = EuropeanCallPayoff(strike=K)
        engine_std = MonteCarloEngine(model=model_std, payoff=payoff, n_paths=n_paths)
        result_std = engine_std.price()
        ci_width_std = result_std.ci_upper - result_std.ci_lower

        # Antithetic MC
        model_anti = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=43)
        engine_anti = MonteCarloEngine(
            model=model_anti, payoff=payoff, n_paths=n_paths, antithetic=True
        )
        result_anti = engine_anti.price()
        ci_width_anti = result_anti.ci_upper - result_anti.ci_lower

        # Antithetic CI should be narrower (or similar)
        assert ci_width_anti <= ci_width_std * 1.1  # Allow 10% tolerance
