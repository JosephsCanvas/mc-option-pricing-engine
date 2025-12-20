"""
Unit tests for plain vanilla option payoffs.
"""

import numpy as np
import pytest

from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff, EuropeanPutPayoff


class TestEuropeanCallPayoff:
    """Test suite for European call payoff."""

    def test_initialization(self):
        """Test call payoff initialization."""
        payoff = EuropeanCallPayoff(strike=100)
        assert payoff.strike == 100

    def test_invalid_strike(self):
        """Test that non-positive strike raises ValueError."""
        with pytest.raises(ValueError, match="Strike price must be positive"):
            EuropeanCallPayoff(strike=0)

        with pytest.raises(ValueError, match="Strike price must be positive"):
            EuropeanCallPayoff(strike=-100)

    def test_itm_payoff(self):
        """Test in-the-money call payoff."""
        payoff = EuropeanCallPayoff(strike=100)
        S_T = np.array([110, 120, 150])
        expected = np.array([10, 20, 50])
        assert np.allclose(payoff(S_T), expected)

    def test_otm_payoff(self):
        """Test out-of-the-money call payoff."""
        payoff = EuropeanCallPayoff(strike=100)
        S_T = np.array([90, 80, 50])
        expected = np.array([0, 0, 0])
        assert np.allclose(payoff(S_T), expected)

    def test_atm_payoff(self):
        """Test at-the-money call payoff."""
        payoff = EuropeanCallPayoff(strike=100)
        S_T = np.array([100])
        expected = np.array([0])
        assert np.allclose(payoff(S_T), expected)

    def test_monotonicity(self):
        """Test that call payoff is monotonically increasing in spot."""
        payoff = EuropeanCallPayoff(strike=100)
        S_T = np.linspace(50, 150, 100)
        payoffs = payoff(S_T)

        # Check monotonicity
        assert np.all(np.diff(payoffs) >= 0)

    def test_repr(self):
        """Test string representation."""
        payoff = EuropeanCallPayoff(strike=100)
        assert "EuropeanCallPayoff" in repr(payoff)
        assert "100" in repr(payoff)


class TestEuropeanPutPayoff:
    """Test suite for European put payoff."""

    def test_initialization(self):
        """Test put payoff initialization."""
        payoff = EuropeanPutPayoff(strike=100)
        assert payoff.strike == 100

    def test_invalid_strike(self):
        """Test that non-positive strike raises ValueError."""
        with pytest.raises(ValueError, match="Strike price must be positive"):
            EuropeanPutPayoff(strike=0)

        with pytest.raises(ValueError, match="Strike price must be positive"):
            EuropeanPutPayoff(strike=-100)

    def test_itm_payoff(self):
        """Test in-the-money put payoff."""
        payoff = EuropeanPutPayoff(strike=100)
        S_T = np.array([90, 80, 50])
        expected = np.array([10, 20, 50])
        assert np.allclose(payoff(S_T), expected)

    def test_otm_payoff(self):
        """Test out-of-the-money put payoff."""
        payoff = EuropeanPutPayoff(strike=100)
        S_T = np.array([110, 120, 150])
        expected = np.array([0, 0, 0])
        assert np.allclose(payoff(S_T), expected)

    def test_atm_payoff(self):
        """Test at-the-money put payoff."""
        payoff = EuropeanPutPayoff(strike=100)
        S_T = np.array([100])
        expected = np.array([0])
        assert np.allclose(payoff(S_T), expected)

    def test_monotonicity(self):
        """Test that put payoff is monotonically decreasing in spot."""
        payoff = EuropeanPutPayoff(strike=100)
        S_T = np.linspace(50, 150, 100)
        payoffs = payoff(S_T)

        # Check monotonicity (decreasing)
        assert np.all(np.diff(payoffs) <= 0)

    def test_repr(self):
        """Test string representation."""
        payoff = EuropeanPutPayoff(strike=100)
        assert "EuropeanPutPayoff" in repr(payoff)
        assert "100" in repr(payoff)

    def test_put_call_parity_payoff(self):
        """Test put-call parity at payoff level: C - P = S_T - K."""
        K = 100
        call = EuropeanCallPayoff(strike=K)
        put = EuropeanPutPayoff(strike=K)

        S_T = np.linspace(50, 150, 100)
        call_payoffs = call(S_T)
        put_payoffs = put(S_T)

        # At payoff level: C - P = S_T - K (undiscounted)
        assert np.allclose(call_payoffs - put_payoffs, S_T - K)
