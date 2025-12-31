"""Tests for multi-asset Monte Carlo pricer."""

import math

import numpy as np
import pytest

from mc_pricer.models.multi_gbm import MultiAssetGeometricBrownianMotion
from mc_pricer.payoffs.multi_asset import (
    BasketArithmeticCallPayoff,
    SpreadCallPayoff,
    SpreadPutPayoff,
)
from mc_pricer.pricers.multi_asset_monte_carlo import (
    MultiAssetMonteCarloEngine,
    MultiAssetPricingResult,
)

# Utility functions for analytic formulas


def norm_cdf(x: float) -> float:
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def margrabe_exchange_price(
    S1: float, S2: float, sigma1: float, sigma2: float, rho: float, T: float
) -> float:
    """
    Margrabe formula for exchange option: max(S1_T - S2_T, 0).

    This is equivalent to a spread call with K=0.

    Parameters
    ----------
    S1 : float
        Initial price of asset 1
    S2 : float
        Initial price of asset 2
    sigma1 : float
        Volatility of asset 1
    sigma2 : float
        Volatility of asset 2
    rho : float
        Correlation between assets
    T : float
        Time to maturity

    Returns
    -------
    float
        Exchange option price
    """
    # Composite volatility
    sigma_M = math.sqrt(sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2)

    if sigma_M < 1e-10:
        # Degenerate case: assets perfectly correlated with same vol
        return max(S1 - S2, 0.0)

    d1 = (math.log(S1 / S2) + 0.5 * sigma_M**2 * T) / (sigma_M * math.sqrt(T))
    d2 = d1 - sigma_M * math.sqrt(T)

    price = S1 * norm_cdf(d1) - S2 * norm_cdf(d2)
    return price


# Test fixtures


@pytest.fixture
def simple_two_asset_model():
    """Simple two-asset model with moderate correlation."""
    return MultiAssetGeometricBrownianMotion(
        S0=np.array([100.0, 100.0]),
        r=0.05,
        sigma=np.array([0.2, 0.25]),
        T=1.0,
        corr=np.array([[1.0, 0.5], [0.5, 1.0]]),
        seed=42,
    )


@pytest.fixture
def three_asset_model():
    """Three-asset model for basket options."""
    return MultiAssetGeometricBrownianMotion(
        S0=np.array([100.0, 95.0, 110.0]),
        r=0.05,
        sigma=np.array([0.2, 0.25, 0.18]),
        T=1.0,
        corr=np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]]),
        seed=123,
    )


# Basic functionality tests


def test_pricer_initialization(simple_two_asset_model):
    """Test that pricer initializes correctly."""
    payoff = SpreadCallPayoff(strike=10.0)

    engine = MultiAssetMonteCarloEngine(
        model=simple_two_asset_model,
        payoff=payoff,
        n_paths=1000,
        antithetic=False,
        rng_type="pseudo",
        seed=42,
    )

    assert engine.n_paths == 1000
    assert engine.model.n_assets == 2
    assert not engine.antithetic


def test_invalid_n_paths(simple_two_asset_model):
    """Test that invalid n_paths raises ValueError."""
    payoff = SpreadCallPayoff(strike=10.0)

    with pytest.raises(ValueError, match="n_paths must be positive"):
        MultiAssetMonteCarloEngine(model=simple_two_asset_model, payoff=payoff, n_paths=0)

    with pytest.raises(ValueError, match="n_paths must be positive"):
        MultiAssetMonteCarloEngine(model=simple_two_asset_model, payoff=payoff, n_paths=-100)


def test_antithetic_requires_even_paths(simple_two_asset_model):
    """Test that antithetic requires even n_paths."""
    payoff = SpreadCallPayoff(strike=10.0)

    with pytest.raises(ValueError, match="even"):
        MultiAssetMonteCarloEngine(
            model=simple_two_asset_model,
            payoff=payoff,
            n_paths=1001,
            antithetic=True,
        )


def test_pricing_returns_result(simple_two_asset_model):
    """Test that pricing returns valid result."""
    payoff = SpreadCallPayoff(strike=5.0)

    engine = MultiAssetMonteCarloEngine(
        model=simple_two_asset_model,
        payoff=payoff,
        n_paths=10000,
        rng_type="pseudo",
        seed=42,
    )

    result = engine.price()

    assert isinstance(result, MultiAssetPricingResult)
    assert result.price > 0  # Spread call should have positive value
    assert result.stderr > 0
    assert result.ci_lower < result.price < result.ci_upper
    assert result.n_paths == 10000
    assert result.n_assets == 2
    assert result.rng_type == "pseudo"


def test_ci_bounds_correct(simple_two_asset_model):
    """Test that confidence interval bounds are correctly computed."""
    payoff = SpreadCallPayoff(strike=10.0)

    engine = MultiAssetMonteCarloEngine(
        model=simple_two_asset_model,
        payoff=payoff,
        n_paths=5000,
        seed=42,
    )

    result = engine.price()

    # CI should be price ± 1.96 * stderr
    expected_ci_lower = result.price - 1.96 * result.stderr
    expected_ci_upper = result.price + 1.96 * result.stderr

    np.testing.assert_almost_equal(result.ci_lower, expected_ci_lower, decimal=10)
    np.testing.assert_almost_equal(result.ci_upper, expected_ci_upper, decimal=10)


def test_reproducibility_with_seed(simple_two_asset_model):
    """Test that same seed gives identical results."""
    payoff = SpreadCallPayoff(strike=5.0)

    engine1 = MultiAssetMonteCarloEngine(
        model=simple_two_asset_model,
        payoff=payoff,
        n_paths=1000,
        seed=999,
    )

    engine2 = MultiAssetMonteCarloEngine(
        model=simple_two_asset_model,
        payoff=payoff,
        n_paths=1000,
        seed=999,
    )

    result1 = engine1.price()
    result2 = engine2.price()

    assert result1.price == result2.price
    assert result1.stderr == result2.stderr


def test_basket_option_pricing(three_asset_model):
    """Test basket option pricing."""
    payoff = BasketArithmeticCallPayoff(strike=100.0)

    engine = MultiAssetMonteCarloEngine(
        model=three_asset_model,
        payoff=payoff,
        n_paths=20000,
        seed=456,
    )

    result = engine.price()

    assert result.price > 0
    assert result.n_assets == 3
    # Basket with average S0 ~102, K=100, should have some value
    assert result.price > 5.0  # Reasonable lower bound


# Variance reduction tests


def test_antithetic_reduces_stderr(simple_two_asset_model):
    """Test that antithetic variates reduce standard error."""
    payoff = SpreadCallPayoff(strike=10.0)

    # Standard
    engine_standard = MultiAssetMonteCarloEngine(
        model=simple_two_asset_model,
        payoff=payoff,
        n_paths=10000,
        antithetic=False,
        seed=111,
    )

    # Antithetic
    engine_antithetic = MultiAssetMonteCarloEngine(
        model=simple_two_asset_model,
        payoff=payoff,
        n_paths=10000,
        antithetic=True,
        seed=111,
    )

    result_standard = engine_standard.price()
    result_antithetic = engine_antithetic.price()

    # Antithetic should have smaller or equal stderr (with tolerance)
    # Note: Not guaranteed for every seed, but should be true statistically
    assert result_antithetic.stderr <= result_standard.stderr * 1.1


# QMC tests


def test_sobol_pricing(simple_two_asset_model):
    """Test pricing with Sobol QMC."""
    payoff = SpreadCallPayoff(strike=5.0)

    engine = MultiAssetMonteCarloEngine(
        model=simple_two_asset_model,
        payoff=payoff,
        n_paths=10000,
        rng_type="sobol",
        scramble=False,
        seed=222,
    )

    result = engine.price()

    assert result.price > 0
    assert result.rng_type == "sobol"
    assert not result.scramble


def test_sobol_scramble_affects_result(simple_two_asset_model):
    """Test that scrambling affects Sobol results."""
    payoff = SpreadCallPayoff(strike=5.0)

    engine_no_scramble = MultiAssetMonteCarloEngine(
        model=simple_two_asset_model,
        payoff=payoff,
        n_paths=5000,
        rng_type="sobol",
        scramble=False,
        seed=333,
    )

    engine_scramble = MultiAssetMonteCarloEngine(
        model=simple_two_asset_model,
        payoff=payoff,
        n_paths=5000,
        rng_type="sobol",
        scramble=True,
        seed=333,
    )

    result_no_scramble = engine_no_scramble.price()
    result_scramble = engine_scramble.price()

    # Should produce different results
    assert result_no_scramble.price != result_scramble.price


def test_qmc_dimension_guard():
    """Test that QMC raises error for dimension > 21."""
    # 22 assets exceeds limit
    S0 = np.ones(22) * 100.0
    sigma = np.ones(22) * 0.2
    corr = np.eye(22)

    model = MultiAssetGeometricBrownianMotion(S0=S0, r=0.05, sigma=sigma, T=1.0, corr=corr)

    payoff = BasketArithmeticCallPayoff(strike=100.0)

    engine = MultiAssetMonteCarloEngine(model=model, payoff=payoff, n_paths=1000, rng_type="sobol")

    with pytest.raises(ValueError, match="exceeds maximum of 21"):
        engine.price()


# Analytic validation: Margrabe formula


def test_spread_K0_matches_margrabe():
    """
    Test that spread call with K=0 matches Margrabe exchange option formula.

    This is a high-value test for credibility.
    """
    # Parameters
    S1 = 110.0
    S2 = 100.0
    sigma1 = 0.25
    sigma2 = 0.30
    rho = 0.6
    r = 0.05
    T = 1.0

    # Analytic price via Margrabe
    analytic_price = margrabe_exchange_price(S1, S2, sigma1, sigma2, rho, T)

    # Monte Carlo price
    model = MultiAssetGeometricBrownianMotion(
        S0=np.array([S1, S2]),
        r=r,
        sigma=np.array([sigma1, sigma2]),
        T=T,
        corr=np.array([[1.0, rho], [rho, 1.0]]),
        seed=12345,
    )

    payoff = SpreadCallPayoff(strike=0.0)  # K=0 for exchange option

    engine = MultiAssetMonteCarloEngine(
        model=model,
        payoff=payoff,
        n_paths=100000,  # Large n_paths for accuracy
        rng_type="sobol",
        scramble=True,
        seed=12345,
    )

    result = engine.price()

    # MC should be within 3*stderr of analytic (99.7% confidence)
    tolerance = 3 * result.stderr + 0.1  # Small buffer
    assert abs(result.price - analytic_price) < tolerance, (
        f"MC price {result.price:.4f} differs from Margrabe "
        f"{analytic_price:.4f} by more than {tolerance:.4f}"
    )

    # Also check that analytic is within CI
    assert result.ci_lower <= analytic_price <= result.ci_upper


def test_margrabe_multiple_correlations():
    """Test Margrabe validation across different correlations."""
    S1 = 100.0
    S2 = 100.0
    sigma1 = 0.2
    sigma2 = 0.2
    r = 0.05
    T = 1.0

    for rho in [-0.5, 0.0, 0.5]:
        analytic_price = margrabe_exchange_price(S1, S2, sigma1, sigma2, rho, T)

        model = MultiAssetGeometricBrownianMotion(
            S0=np.array([S1, S2]),
            r=r,
            sigma=np.array([sigma1, sigma2]),
            T=T,
            corr=np.array([[1.0, rho], [rho, 1.0]]),
            seed=54321,
        )

        payoff = SpreadCallPayoff(strike=0.0)

        engine = MultiAssetMonteCarloEngine(
            model=model,
            payoff=payoff,
            n_paths=50000,
            rng_type="pseudo",
            seed=54321,
        )

        result = engine.price()

        tolerance = 4 * result.stderr + 0.15
        assert abs(result.price - analytic_price) < tolerance, (
            f"rho={rho}: MC {result.price:.4f} vs Margrabe {analytic_price:.4f}"
        )


def test_margrabe_independent_assets():
    """Test Margrabe for independent assets (rho=0)."""
    S1 = 105.0
    S2 = 95.0
    sigma1 = 0.3
    sigma2 = 0.3
    rho = 0.0
    r = 0.05
    T = 0.5

    analytic_price = margrabe_exchange_price(S1, S2, sigma1, sigma2, rho, T)

    model = MultiAssetGeometricBrownianMotion(
        S0=np.array([S1, S2]),
        r=r,
        sigma=np.array([sigma1, sigma2]),
        T=T,
        corr=np.array([[1.0, 0.0], [0.0, 1.0]]),
        seed=11111,
    )

    payoff = SpreadCallPayoff(strike=0.0)

    engine = MultiAssetMonteCarloEngine(
        model=model,
        payoff=payoff,
        n_paths=80000,
        rng_type="pseudo",  # Use pseudo for more reliable convergence
        seed=11111,
    )

    result = engine.price()

    tolerance = 4 * result.stderr + 0.2  # Increased tolerance for independent case
    assert abs(result.price - analytic_price) < tolerance


# Edge cases


def test_zero_volatility():
    """Test with zero volatility (deterministic)."""
    model = MultiAssetGeometricBrownianMotion(
        S0=np.array([100.0, 110.0]),
        r=0.05,
        sigma=np.array([0.0, 0.0]),
        T=1.0,
        corr=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )

    payoff = SpreadCallPayoff(strike=5.0)

    engine = MultiAssetMonteCarloEngine(model=model, payoff=payoff, n_paths=1000)

    result = engine.price()

    # With zero vol, S_T = S0 * exp(r*T)
    S1_T = 100.0 * math.exp(0.05 * 1.0)
    S2_T = 110.0 * math.exp(0.05 * 1.0)
    spread = S1_T - S2_T  # ~-10
    expected_payoff = max(spread - 5.0, 0.0)  # 0
    expected_price = expected_payoff * math.exp(-0.05 * 1.0)

    # Should be very close to deterministic value
    np.testing.assert_almost_equal(result.price, expected_price, decimal=10)
    # Stderr should be exactly zero
    np.testing.assert_almost_equal(result.stderr, 0.0, decimal=10)


def test_deep_itm_call(simple_two_asset_model):
    """Test deep in-the-money spread call."""
    # S1=100, S2=100, but with high correlation and vol
    # Set very low strike to ensure ITM
    payoff = SpreadCallPayoff(strike=0.0)

    engine = MultiAssetMonteCarloEngine(
        model=simple_two_asset_model,
        payoff=payoff,
        n_paths=20000,
        seed=777,
    )

    result = engine.price()

    # Should have positive value
    assert result.price > 0


def test_deep_otm_call():
    """Test deep out-of-the-money spread call."""
    model = MultiAssetGeometricBrownianMotion(
        S0=np.array([100.0, 100.0]),
        r=0.05,
        sigma=np.array([0.1, 0.1]),
        T=0.25,
        corr=np.array([[1.0, 0.9], [0.9, 1.0]]),
        seed=888,
    )

    # Very high strike makes it OTM
    payoff = SpreadCallPayoff(strike=50.0)

    engine = MultiAssetMonteCarloEngine(model=model, payoff=payoff, n_paths=10000, seed=888)

    result = engine.price()

    # Should be close to zero
    assert result.price < 1.0


def test_repr_methods(simple_two_asset_model):
    """Test string representations."""
    payoff = SpreadCallPayoff(strike=10.0)

    engine = MultiAssetMonteCarloEngine(
        model=simple_two_asset_model,
        payoff=payoff,
        n_paths=1000,
    )

    engine_repr = repr(engine)
    assert "MultiAssetMonteCarloEngine" in engine_repr
    assert "n_assets=2" in engine_repr

    result = engine.price()
    result_repr = repr(result)
    assert "MultiAssetPricingResult" in result_repr
    assert "price=" in result_repr


def test_payoff_validation():
    """Test that invalid payoff output raises error."""
    model = MultiAssetGeometricBrownianMotion(
        S0=np.array([100.0, 95.0]),
        r=0.05,
        sigma=np.array([0.2, 0.25]),
        T=1.0,
        corr=np.array([[1.0, 0.3], [0.3, 1.0]]),
    )

    # Payoff that returns wrong shape
    def bad_payoff(S_T: np.ndarray) -> np.ndarray:
        return np.ones((S_T.shape[0], 2))  # 2D instead of 1D

    engine = MultiAssetMonteCarloEngine(model=model, payoff=bad_payoff, n_paths=100)

    with pytest.raises(ValueError, match="must return array of shape"):
        engine.price()


def test_spread_put_pricing(simple_two_asset_model):
    """Test spread put option pricing."""
    payoff = SpreadPutPayoff(strike=5.0)

    engine = MultiAssetMonteCarloEngine(
        model=simple_two_asset_model,
        payoff=payoff,
        n_paths=20000,
        seed=999,
    )

    result = engine.price()

    assert result.price > 0
    assert result.stderr > 0
