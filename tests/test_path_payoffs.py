"""Tests for path-dependent option payoffs."""

import numpy as np
import pytest

from mc_pricer.payoffs.path_dependent import (
    AsianArithmeticCallPayoff,
    AsianArithmeticPutPayoff,
    DownAndOutPutPayoff,
    UpAndOutCallPayoff,
)


class TestAsianArithmeticCallPayoff:
    """Test AsianArithmeticCallPayoff."""

    def test_strike_validation(self):
        """Test that invalid strike raises ValueError."""
        with pytest.raises(ValueError, match="Strike price must be positive"):
            AsianArithmeticCallPayoff(strike=0)

        with pytest.raises(ValueError, match="Strike price must be positive"):
            AsianArithmeticCallPayoff(strike=-10)

    def test_itm_payoff(self):
        """Test in-the-money Asian call payoff."""
        payoff = AsianArithmeticCallPayoff(strike=100)

        # Simple path: [100, 110, 120]
        # Average = 110, Payoff = max(110 - 100, 0) = 10
        paths = np.array([[100, 110, 120]])

        result = payoff(paths)
        expected = np.array([10.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_otm_payoff(self):
        """Test out-of-the-money Asian call payoff."""
        payoff = AsianArithmeticCallPayoff(strike=150)

        # Simple path: [100, 110, 120]
        # Average = 110, Payoff = max(110 - 150, 0) = 0
        paths = np.array([[100, 110, 120]])

        result = payoff(paths)
        expected = np.array([0.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_atm_payoff(self):
        """Test at-the-money Asian call payoff."""
        payoff = AsianArithmeticCallPayoff(strike=110)

        # Simple path: [100, 110, 120]
        # Average = 110, Payoff = max(110 - 110, 0) = 0
        paths = np.array([[100, 110, 120]])

        result = payoff(paths)
        expected = np.array([0.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_multiple_paths(self):
        """Test multiple paths simultaneously."""
        payoff = AsianArithmeticCallPayoff(strike=100)

        # Three paths with different averages
        paths = np.array([
            [100, 110, 120],  # avg=110, payoff=10
            [100, 90, 80],    # avg=90, payoff=0
            [100, 105, 110],  # avg=105, payoff=5
        ])

        result = payoff(paths)
        expected = np.array([10.0, 0.0, 5.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_repr(self):
        """Test string representation."""
        payoff = AsianArithmeticCallPayoff(strike=100.5)
        assert repr(payoff) == "AsianArithmeticCallPayoff(strike=100.5)"


class TestAsianArithmeticPutPayoff:
    """Test AsianArithmeticPutPayoff."""

    def test_strike_validation(self):
        """Test that invalid strike raises ValueError."""
        with pytest.raises(ValueError, match="Strike price must be positive"):
            AsianArithmeticPutPayoff(strike=0)

    def test_itm_payoff(self):
        """Test in-the-money Asian put payoff."""
        payoff = AsianArithmeticPutPayoff(strike=100)

        # Simple path: [100, 90, 80]
        # Average = 90, Payoff = max(100 - 90, 0) = 10
        paths = np.array([[100, 90, 80]])

        result = payoff(paths)
        expected = np.array([10.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_otm_payoff(self):
        """Test out-of-the-money Asian put payoff."""
        payoff = AsianArithmeticPutPayoff(strike=80)

        # Simple path: [100, 90, 80]
        # Average = 90, Payoff = max(80 - 90, 0) = 0
        paths = np.array([[100, 90, 80]])

        result = payoff(paths)
        expected = np.array([0.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_multiple_paths(self):
        """Test multiple paths simultaneously."""
        payoff = AsianArithmeticPutPayoff(strike=100)

        paths = np.array([
            [100, 90, 80],    # avg=90, payoff=10
            [100, 110, 120],  # avg=110, payoff=0
            [100, 95, 90],    # avg=95, payoff=5
        ])

        result = payoff(paths)
        expected = np.array([10.0, 0.0, 5.0])

        np.testing.assert_array_almost_equal(result, expected)


class TestUpAndOutCallPayoff:
    """Test UpAndOutCallPayoff."""

    def test_validation(self):
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError, match="Strike price must be positive"):
            UpAndOutCallPayoff(strike=0, barrier=150)

        with pytest.raises(ValueError, match="Barrier must be positive"):
            UpAndOutCallPayoff(strike=100, barrier=0)

    def test_not_knocked_out(self):
        """Test payoff when barrier is not breached."""
        payoff = UpAndOutCallPayoff(strike=100, barrier=150)

        # Path never reaches 150, terminal=120
        # Payoff = max(120 - 100, 0) = 20
        paths = np.array([[100, 110, 120, 115, 120]])

        result = payoff(paths)
        expected = np.array([20.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_knocked_out(self):
        """Test payoff when barrier is breached."""
        payoff = UpAndOutCallPayoff(strike=100, barrier=150)

        # Path reaches 150, knocked out
        # Payoff = 0
        paths = np.array([[100, 120, 150, 145, 140]])

        result = payoff(paths)
        expected = np.array([0.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_knocked_out_exact_barrier(self):
        """Test payoff when path exactly touches barrier."""
        payoff = UpAndOutCallPayoff(strike=100, barrier=150)

        # Path exactly touches 150
        paths = np.array([[100, 120, 149.99, 150.0, 145]])

        result = payoff(paths)
        expected = np.array([0.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_multiple_paths(self):
        """Test multiple paths with mixed outcomes."""
        payoff = UpAndOutCallPayoff(strike=100, barrier=150)

        paths = np.array([
            [100, 120, 140, 135, 145],  # Not knocked out, terminal=145, payoff=45
            [100, 130, 150, 140, 135],  # Knocked out, payoff=0
            [100, 110, 105, 108, 112],  # Not knocked out, terminal=112, payoff=12
            [100, 145, 149, 151, 148],  # Knocked out at 151, payoff=0
        ])

        result = payoff(paths)
        expected = np.array([45.0, 0.0, 12.0, 0.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_repr(self):
        """Test string representation."""
        payoff = UpAndOutCallPayoff(strike=100, barrier=150)
        assert repr(payoff) == "UpAndOutCallPayoff(strike=100, barrier=150)"


class TestDownAndOutPutPayoff:
    """Test DownAndOutPutPayoff."""

    def test_validation(self):
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError, match="Strike price must be positive"):
            DownAndOutPutPayoff(strike=0, barrier=50)

        with pytest.raises(ValueError, match="Barrier must be positive"):
            DownAndOutPutPayoff(strike=100, barrier=0)

    def test_not_knocked_out(self):
        """Test payoff when barrier is not breached."""
        payoff = DownAndOutPutPayoff(strike=100, barrier=50)

        # Path never reaches 50, terminal=80
        # Payoff = max(100 - 80, 0) = 20
        paths = np.array([[100, 90, 80, 85, 80]])

        result = payoff(paths)
        expected = np.array([20.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_knocked_out(self):
        """Test payoff when barrier is breached."""
        payoff = DownAndOutPutPayoff(strike=100, barrier=50)

        # Path reaches 50, knocked out
        # Payoff = 0
        paths = np.array([[100, 80, 50, 55, 60]])

        result = payoff(paths)
        expected = np.array([0.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_knocked_out_exact_barrier(self):
        """Test payoff when path exactly touches barrier."""
        payoff = DownAndOutPutPayoff(strike=100, barrier=50)

        # Path exactly touches 50
        paths = np.array([[100, 80, 50.01, 50.0, 55]])

        result = payoff(paths)
        expected = np.array([0.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_multiple_paths(self):
        """Test multiple paths with mixed outcomes."""
        payoff = DownAndOutPutPayoff(strike=100, barrier=50)

        paths = np.array([
            [100, 80, 60, 65, 70],    # Not knocked out, terminal=70, payoff=30
            [100, 70, 50, 60, 65],    # Knocked out, payoff=0
            [100, 90, 85, 82, 80],    # Not knocked out, terminal=80, payoff=20
            [100, 75, 51, 49, 55],    # Knocked out at 49, payoff=0
        ])

        result = payoff(paths)
        expected = np.array([30.0, 0.0, 20.0, 0.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_repr(self):
        """Test string representation."""
        payoff = DownAndOutPutPayoff(strike=100, barrier=50)
        assert repr(payoff) == "DownAndOutPutPayoff(strike=100, barrier=50)"
