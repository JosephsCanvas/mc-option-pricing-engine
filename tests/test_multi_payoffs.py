"""Tests for multi-asset payoffs."""

import numpy as np
import pytest

from mc_pricer.payoffs.multi_asset import (
    BasketArithmeticCallPayoff,
    BasketArithmeticPutPayoff,
    SpreadCallPayoff,
    SpreadPutPayoff,
)


def test_basket_call_correctness():
    """Test basket call payoff computation."""
    payoff = BasketArithmeticCallPayoff(strike=100.0)

    # Case 1: Average above strike
    S_T = np.array([[110.0, 120.0, 130.0], [95.0, 105.0, 115.0]])
    expected = np.array([20.0, 5.0])  # means: 120, 105
    result = payoff(S_T)
    np.testing.assert_array_almost_equal(result, expected)

    # Case 2: Average below strike
    S_T = np.array([[90.0, 95.0, 85.0]])
    expected = np.array([0.0])  # mean: 90
    result = payoff(S_T)
    np.testing.assert_array_almost_equal(result, expected)

    # Case 3: Average at strike
    S_T = np.array([[100.0, 100.0, 100.0]])
    expected = np.array([0.0])
    result = payoff(S_T)
    np.testing.assert_array_almost_equal(result, expected)


def test_basket_put_correctness():
    """Test basket put payoff computation."""
    payoff = BasketArithmeticPutPayoff(strike=100.0)

    # Case 1: Average below strike
    S_T = np.array([[80.0, 90.0, 70.0], [95.0, 85.0, 105.0]])
    expected = np.array([20.0, 5.0])  # means: 80, 95
    result = payoff(S_T)
    np.testing.assert_array_almost_equal(result, expected)

    # Case 2: Average above strike
    S_T = np.array([[110.0, 120.0, 130.0]])
    expected = np.array([0.0])  # mean: 120
    result = payoff(S_T)
    np.testing.assert_array_almost_equal(result, expected)


def test_spread_call_correctness():
    """Test spread call payoff computation."""
    payoff = SpreadCallPayoff(strike=10.0)

    # Case 1: Spread > strike
    S_T = np.array([[120.0, 95.0], [110.0, 90.0]])
    expected = np.array([15.0, 10.0])  # spreads: 25, 20
    result = payoff(S_T)
    np.testing.assert_array_almost_equal(result, expected)

    # Case 2: Spread < strike
    S_T = np.array([[100.0, 95.0]])
    expected = np.array([0.0])  # spread: 5
    result = payoff(S_T)
    np.testing.assert_array_almost_equal(result, expected)

    # Case 3: Negative spread (S1 < S2)
    S_T = np.array([[90.0, 100.0]])
    expected = np.array([0.0])  # spread: -10
    result = payoff(S_T)
    np.testing.assert_array_almost_equal(result, expected)


def test_spread_put_correctness():
    """Test spread put payoff computation."""
    payoff = SpreadPutPayoff(strike=10.0)

    # Case 1: Spread < strike (put in the money)
    S_T = np.array([[100.0, 95.0], [105.0, 110.0]])
    expected = np.array([5.0, 15.0])  # spreads: 5, -5
    result = payoff(S_T)
    np.testing.assert_array_almost_equal(result, expected)

    # Case 2: Spread > strike (put out of money)
    S_T = np.array([[130.0, 100.0]])
    expected = np.array([0.0])  # spread: 30
    result = payoff(S_T)
    np.testing.assert_array_almost_equal(result, expected)


def test_spread_requires_two_assets():
    """Test that spread payoffs require exactly 2 assets."""
    payoff_call = SpreadCallPayoff(strike=10.0)
    payoff_put = SpreadPutPayoff(strike=10.0)

    # Too many assets
    S_T_three = np.array([[100.0, 95.0, 110.0]])

    with pytest.raises(ValueError, match="exactly 2 assets"):
        payoff_call(S_T_three)

    with pytest.raises(ValueError, match="exactly 2 assets"):
        payoff_put(S_T_three)

    # Single asset
    S_T_one = np.array([[100.0]])

    with pytest.raises(ValueError, match="exactly 2 assets"):
        payoff_call(S_T_one)

    with pytest.raises(ValueError, match="exactly 2 assets"):
        payoff_put(S_T_one)


def test_invalid_strike():
    """Test that negative strikes raise ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        BasketArithmeticCallPayoff(strike=-1.0)

    with pytest.raises(ValueError, match="non-negative"):
        BasketArithmeticPutPayoff(strike=-1.0)

    with pytest.raises(ValueError, match="non-negative"):
        SpreadCallPayoff(strike=-1.0)

    with pytest.raises(ValueError, match="non-negative"):
        SpreadPutPayoff(strike=-1.0)


def test_zero_strike():
    """Test payoffs with zero strike."""
    # Basket call with K=0 is always positive
    payoff = BasketArithmeticCallPayoff(strike=0.0)
    S_T = np.array([[100.0, 110.0, 120.0]])
    result = payoff(S_T)
    expected = np.mean(S_T)
    np.testing.assert_array_almost_equal(result, expected)

    # Spread call with K=0 is max(S1 - S2, 0)
    payoff_spread = SpreadCallPayoff(strike=0.0)
    S_T_spread = np.array([[120.0, 100.0], [90.0, 110.0]])
    result_spread = payoff_spread(S_T_spread)
    expected_spread = np.array([20.0, 0.0])
    np.testing.assert_array_almost_equal(result_spread, expected_spread)


def test_payoff_shapes():
    """Test that payoffs return correct shapes."""
    basket_call = BasketArithmeticCallPayoff(strike=100.0)
    spread_call = SpreadCallPayoff(strike=10.0)

    # Multiple paths, multiple assets
    S_T_basket = np.random.uniform(80, 120, size=(1000, 5))
    result_basket = basket_call(S_T_basket)
    assert result_basket.shape == (1000,)
    assert np.all(result_basket >= 0)

    # Multiple paths, 2 assets
    S_T_spread = np.random.uniform(80, 120, size=(500, 2))
    result_spread = spread_call(S_T_spread)
    assert result_spread.shape == (500,)
    assert np.all(result_spread >= 0)


def test_invalid_input_dimensions():
    """Test that 1D input raises ValueError."""
    payoff = BasketArithmeticCallPayoff(strike=100.0)

    # 1D input
    with pytest.raises(ValueError, match="S_T must be 2D"):
        payoff(np.array([100.0, 110.0, 120.0]))


def test_basket_with_two_assets():
    """Test that basket payoffs work with minimum 2 assets."""
    payoff = BasketArithmeticCallPayoff(strike=100.0)

    S_T = np.array([[110.0, 90.0]])
    expected = np.array([0.0])  # mean: 100
    result = payoff(S_T)
    np.testing.assert_array_almost_equal(result, expected)


def test_repr_methods():
    """Test string representation of payoffs."""
    payoff1 = BasketArithmeticCallPayoff(strike=105.5)
    assert "BasketArithmeticCallPayoff" in repr(payoff1)
    assert "105.5" in repr(payoff1)

    payoff2 = SpreadCallPayoff(strike=20.0)
    assert "SpreadCallPayoff" in repr(payoff2)
    assert "20.0" in repr(payoff2)


def test_large_number_of_assets():
    """Test basket payoff with many assets."""
    payoff = BasketArithmeticCallPayoff(strike=100.0)

    # 20 assets
    n_paths = 100
    n_assets = 20
    S_T = np.random.uniform(90, 110, size=(n_paths, n_assets))

    result = payoff(S_T)
    assert result.shape == (n_paths,)

    # Verify computation
    expected = np.maximum(np.mean(S_T, axis=1) - 100.0, 0.0)
    np.testing.assert_array_almost_equal(result, expected)


def test_put_call_parity_like_relationship():
    """Test relationship between call and put basket options."""
    strike = 100.0
    call_payoff = BasketArithmeticCallPayoff(strike=strike)
    put_payoff = BasketArithmeticPutPayoff(strike=strike)

    S_T = np.array([[110.0, 120.0, 130.0], [80.0, 90.0, 70.0], [100.0, 100.0, 100.0]])

    call_result = call_payoff(S_T)
    put_result = put_payoff(S_T)

    # For each path, at most one is non-zero (or both zero at strike)
    # call + put >= |mean - K|
    avg = np.mean(S_T, axis=1)
    assert np.all(call_result + put_result >= np.abs(avg - strike) - 1e-10)


def test_spread_symmetry():
    """Test that spread call/put have expected symmetry."""
    strike = 5.0
    call_payoff = SpreadCallPayoff(strike=strike)
    put_payoff = SpreadPutPayoff(strike=strike)

    S_T = np.array([[120.0, 100.0], [100.0, 120.0], [110.0, 105.0]])

    call_result = call_payoff(S_T)
    put_result = put_payoff(S_T)

    # Spreads: 20, -20, 5
    # Call: max(20-5, 0) = 15, max(-20-5, 0) = 0, max(5-5, 0) = 0
    # Put: max(5-20, 0) = 0, max(5-(-20), 0) = 25, max(5-5, 0) = 0

    np.testing.assert_array_almost_equal(call_result, np.array([15.0, 0.0, 0.0]))
    np.testing.assert_array_almost_equal(put_result, np.array([0.0, 25.0, 0.0]))
