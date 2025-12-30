"""
Tests for implied volatility solver and Black-Scholes analytics.
"""

import math

import pytest

from mc_pricer.analytics.black_scholes import bs_price
from mc_pricer.analytics.implied_vol import implied_vol


class TestImpliedVolRecovery:
    """Test that implied_vol recovers the true volatility."""

    @pytest.mark.parametrize(
        "S0,K,r,T,sigma,option_type",
        [
            (100, 100, 0.05, 1.0, 0.20, "call"),
            (100, 100, 0.05, 1.0, 0.20, "put"),
            (100, 110, 0.05, 1.0, 0.25, "call"),  # OTM call
            (100, 90, 0.05, 1.0, 0.25, "put"),  # OTM put
            (100, 90, 0.05, 1.0, 0.15, "call"),  # ITM call
            (100, 110, 0.05, 1.0, 0.15, "put"),  # ITM put
            (50, 50, 0.03, 0.5, 0.30, "call"),  # Different params
            (150, 150, 0.02, 2.0, 0.18, "put"),  # Longer maturity
            (80, 100, 0.04, 0.25, 0.40, "call"),  # High vol
            (120, 100, 0.01, 0.5, 0.15, "put"),  # Moderate maturity OTM put
        ],
    )
    def test_recovery_accuracy(self, S0, K, r, T, sigma, option_type):
        """Test that IV solver recovers true volatility accurately."""
        # Compute price with true volatility
        price = bs_price(S0, K, r, T, sigma, option_type)

        # Recover implied volatility
        iv = implied_vol(price, S0, K, r, T, option_type, tol=1e-8)

        # Check recovery accuracy
        assert abs(iv - sigma) < 1e-6, (
            f"IV recovery failed: got {iv:.8f}, expected {sigma:.8f}, error: {abs(iv - sigma):.2e}"
        )

    def test_atm_options_various_vols(self):
        """Test ATM options with various volatilities."""
        S0, K, r, T = 100, 100, 0.05, 1.0

        for sigma in [0.05, 0.10, 0.20, 0.30, 0.50, 1.0]:
            for option_type in ["call", "put"]:
                price = bs_price(S0, K, r, T, sigma, option_type)
                iv = implied_vol(price, S0, K, r, T, option_type, tol=1e-8)
                assert abs(iv - sigma) < 1e-6


class TestArbitrageBounds:
    """Test that arbitrage bounds are enforced."""

    def test_call_below_intrinsic(self):
        """Test that call price below intrinsic raises error."""
        S0, K, r, T = 100, 90, 0.05, 1.0
        intrinsic = S0 - K * math.exp(-r * T)
        invalid_price = intrinsic - 0.1

        with pytest.raises(ValueError, match="below arbitrage lower bound"):
            implied_vol(invalid_price, S0, K, r, T, "call")

    def test_call_above_spot(self):
        """Test that call price above spot raises error."""
        S0, K, r, T = 100, 100, 0.05, 1.0
        invalid_price = S0 + 0.1

        with pytest.raises(ValueError, match="exceeds arbitrage upper bound"):
            implied_vol(invalid_price, S0, K, r, T, "call")

    def test_put_below_intrinsic(self):
        """Test that put price below intrinsic raises error."""
        S0, K, r, T = 100, 110, 0.05, 1.0
        intrinsic = K * math.exp(-r * T) - S0
        invalid_price = intrinsic - 0.1

        with pytest.raises(ValueError, match="below arbitrage lower bound"):
            implied_vol(invalid_price, S0, K, r, T, "put")

    def test_put_above_discounted_strike(self):
        """Test that put price above discounted strike raises error."""
        S0, K, r, T = 100, 100, 0.05, 1.0
        max_put = K * math.exp(-r * T)
        invalid_price = max_put + 0.1

        with pytest.raises(ValueError, match="exceeds arbitrage upper bound"):
            implied_vol(invalid_price, S0, K, r, T, "put")

    def test_valid_boundary_prices(self):
        """Test that prices at boundaries work correctly."""
        S0, K, r, T = 100, 100, 0.05, 1.0

        # Call at lower bound (intrinsic)
        call_intrinsic = max(S0 - K * math.exp(-r * T), 0)
        if call_intrinsic > 0:
            iv_call = implied_vol(call_intrinsic, S0, K, r, T, "call", tol=1e-6)
            assert iv_call < 0.01  # Should be very low vol

        # Put at lower bound (intrinsic)
        put_intrinsic = max(K * math.exp(-r * T) - S0, 0)
        if put_intrinsic > 0:
            iv_put = implied_vol(put_intrinsic, S0, K, r, T, "put", tol=1e-6)
            assert iv_put < 0.01


class TestMonotonicity:
    """Test monotonicity properties of implied volatility."""

    def test_call_price_increases_with_vol(self):
        """Test that call price increases with volatility."""
        S0, K, r, T = 100, 100, 0.05, 1.0

        sigmas = [0.10, 0.20, 0.30, 0.40]
        prices = [bs_price(S0, K, r, T, sigma, "call") for sigma in sigmas]

        # Prices should be increasing
        for i in range(len(prices) - 1):
            assert prices[i] < prices[i + 1]

        # Implied vols should match
        for price, true_sigma in zip(prices, sigmas):
            iv = implied_vol(price, S0, K, r, T, "call")
            assert abs(iv - true_sigma) < 1e-6

    def test_put_price_increases_with_vol(self):
        """Test that put price increases with volatility."""
        S0, K, r, T = 100, 100, 0.05, 1.0

        sigmas = [0.10, 0.20, 0.30, 0.40]
        prices = [bs_price(S0, K, r, T, sigma, "put") for sigma in sigmas]

        # Prices should be increasing
        for i in range(len(prices) - 1):
            assert prices[i] < prices[i + 1]

        # Implied vols should match
        for price, true_sigma in zip(prices, sigmas):
            iv = implied_vol(price, S0, K, r, T, "put")
            assert abs(iv - true_sigma) < 1e-6

    def test_higher_price_gives_higher_iv(self):
        """Test that higher option price implies higher volatility."""
        S0, K, r, T = 100, 100, 0.05, 1.0

        price1 = bs_price(S0, K, r, T, 0.15, "call")
        price2 = bs_price(S0, K, r, T, 0.30, "call")

        iv1 = implied_vol(price1, S0, K, r, T, "call")
        iv2 = implied_vol(price2, S0, K, r, T, "call")

        assert iv1 < iv2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_inputs(self):
        """Test that invalid inputs raise appropriate errors."""
        S0, K, r, T, price = 100, 100, 0.05, 1.0, 10.0

        # Negative spot
        with pytest.raises(ValueError, match="S0 must be positive"):
            implied_vol(price, -1, K, r, T, "call")

        # Negative strike
        with pytest.raises(ValueError, match="K must be positive"):
            implied_vol(price, S0, -1, r, T, "call")

        # Zero maturity
        with pytest.raises(ValueError, match="T must be positive"):
            implied_vol(price, S0, K, r, 0, "call")

        # Negative price
        with pytest.raises(ValueError, match="Price must be non-negative"):
            implied_vol(-1, S0, K, r, T, "call")

        # Invalid option type
        with pytest.raises(ValueError, match="option_type must be"):
            implied_vol(price, S0, K, r, T, "invalid")

    def test_near_zero_price(self):
        """Test handling of very small prices."""
        S0, K, r, T = 100, 150, 0.05, 1.0  # Far OTM call
        small_price = 1e-10

        # Should return low vol without crashing
        iv = implied_vol(small_price, S0, K, r, T, "call", tol=1e-8)
        assert iv >= 0
        assert iv < 0.1  # Should be reasonably small

    def test_high_volatility_prices(self):
        """Test that high volatility prices can be handled."""
        S0, K, r, T = 100, 100, 0.05, 1.0

        # Price from very high vol (should expand search range)
        high_vol = 2.0
        price = bs_price(S0, K, r, T, high_vol, "call")

        iv = implied_vol(price, S0, K, r, T, "call", tol=1e-6)
        assert abs(iv - high_vol) < 1e-4

    def test_different_tolerances(self):
        """Test that different tolerances work correctly."""
        S0, K, r, T, sigma = 100, 100, 0.05, 1.0, 0.25
        price = bs_price(S0, K, r, T, sigma, "call")

        # Looser tolerance
        iv_loose = implied_vol(price, S0, K, r, T, "call", tol=1e-4)
        assert abs(iv_loose - sigma) < 1e-3

        # Tighter tolerance
        iv_tight = implied_vol(price, S0, K, r, T, "call", tol=1e-10)
        assert abs(iv_tight - sigma) < 1e-8

    def test_extreme_otm_numerical_limits(self):
        """
        Test numerical limits for deep OTM short-dated options.

        For extremely small prices (< 1e-9), floating-point precision limits
        prevent exact recovery. This test documents the expected behavior.
        """
        # Deep OTM put with very short maturity
        S0, K, r, T = 120, 100, 0.01, 0.1
        sigma = 0.10

        # Price is near floating-point precision limits (~1e-9)
        price = bs_price(S0, K, r, T, sigma, "put")
        assert price < 1e-8  # Confirm price is extremely small

        # Recovery is limited by numerical precision
        # Use lenient tolerance for such extreme cases
        iv = implied_vol(price, S0, K, r, T, "put", tol=1e-8)
        # Accept larger error for extreme cases
        assert abs(iv - sigma) < 0.05  # Within 5% absolute


class TestConvergence:
    """Test convergence properties of the bisection algorithm."""

    def test_max_iterations(self):
        """Test that max_iter is respected."""
        S0, K, r, T, sigma = 100, 100, 0.05, 1.0, 0.25
        price = bs_price(S0, K, r, T, sigma, "call")

        # Very tight tolerance with few iterations should still work or fail gracefully
        try:
            iv = implied_vol(price, S0, K, r, T, "call", tol=1e-12, max_iter=10)
            # If it converges, check accuracy
            assert abs(iv - sigma) < 1e-3
        except ValueError as e:
            # Should mention iteration limit
            assert "did not converge" in str(e)

    def test_custom_search_bounds(self):
        """Test that custom search bounds work correctly."""
        S0, K, r, T, sigma = 100, 100, 0.05, 1.0, 0.25
        price = bs_price(S0, K, r, T, sigma, "call")

        # Use custom bounds that bracket the solution
        iv = implied_vol(price, S0, K, r, T, "call", sigma_low=0.1, sigma_high=0.5, tol=1e-8)
        assert abs(iv - sigma) < 1e-6
