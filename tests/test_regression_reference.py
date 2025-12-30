"""
Regression tests against frozen reference values.

These tests ensure that core functionality remains stable across code changes
by comparing outputs to frozen reference values within tolerance.
"""

import json
from pathlib import Path

import pytest

from mc_pricer.analytics.black_scholes import bs_delta, bs_price, bs_vega
from mc_pricer.models.gbm import GeometricBrownianMotion
from mc_pricer.models.heston import HestonModel
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff, EuropeanPutPayoff
from mc_pricer.pricers.heston_monte_carlo import HestonMonteCarloEngine
from mc_pricer.pricers.lsm import price_american_lsm
from mc_pricer.pricers.monte_carlo import MonteCarloEngine


@pytest.fixture
def reference_values():
    """Load reference values from JSON."""
    ref_path = Path(__file__).parent / "reference_values.json"
    with open(ref_path) as f:
        return json.load(f)


class TestBlackScholesReferences:
    """Test Black-Scholes analytical values."""

    def test_bs_atm_call_price(self, reference_values):
        """Test that BS analytical price matches reference."""
        ref = reference_values["test_cases"]["black_scholes_atm_call"]
        params = ref["parameters"]

        price = bs_price(
            params["S0"],
            params["K"],
            params["r"],
            params["T"],
            params["sigma"],
            params["option_type"],
        )

        assert abs(price - ref["reference_price"]) < 1e-10

    def test_bs_atm_call_delta(self, reference_values):
        """Test that BS delta matches reference."""
        ref = reference_values["test_cases"]["black_scholes_atm_call"]
        params = ref["parameters"]

        delta = bs_delta(
            params["S0"],
            params["K"],
            params["r"],
            params["T"],
            params["sigma"],
            params["option_type"],
        )

        assert abs(delta - ref["reference_delta"]) < 1e-10

    def test_bs_atm_call_vega(self, reference_values):
        """Test that BS vega matches reference."""
        ref = reference_values["test_cases"]["black_scholes_atm_call"]
        params = ref["parameters"]

        vega = bs_vega(params["S0"], params["K"], params["r"], params["T"], params["sigma"])

        assert abs(vega - ref["reference_vega"]) < 1e-10


class TestGBMMonteCarloReferences:
    """Test GBM Monte Carlo stability."""

    def test_gbm_plain_mc_reproducibility(self, reference_values):
        """Test that plain MC with fixed seed produces stable results."""
        ref = reference_values["test_cases"]["gbm_european_call_plain"]
        params = ref["parameters"]

        model = GeometricBrownianMotion(
            S0=params["S0"],
            r=params["r"],
            sigma=params["sigma"],
            T=params["T"],
            seed=params["seed"],
        )

        payoff = EuropeanCallPayoff(strike=params["K"])
        engine = MonteCarloEngine(
            model=model,
            payoff=payoff,
            n_paths=params["n_paths"],
            antithetic=params["antithetic"],
            control_variate=params["control_variate"],
        )

        result = engine.price()

        # Check within tolerance (MC noise + small epsilon)
        tolerance = ref["tolerance_stderr_multiple"] * result.stderr + ref["tolerance_absolute"]
        assert abs(result.price - ref["reference_price"]) < tolerance

    def test_gbm_antithetic_reproducibility(self, reference_values):
        """Test that antithetic variates with fixed seed produces stable results."""
        ref = reference_values["test_cases"]["gbm_european_call_antithetic"]
        params = ref["parameters"]

        model = GeometricBrownianMotion(
            S0=params["S0"],
            r=params["r"],
            sigma=params["sigma"],
            T=params["T"],
            seed=params["seed"],
        )

        payoff = EuropeanCallPayoff(strike=params["K"])
        engine = MonteCarloEngine(
            model=model,
            payoff=payoff,
            n_paths=params["n_paths"],
            antithetic=params["antithetic"],
            control_variate=params["control_variate"],
        )

        result = engine.price()

        # Check within Monte Carlo tolerance
        # With fixed seed, should be very close to reference
        assert abs(result.price - ref["reference_price"]) < 0.5

    def test_control_variate_reduces_variance(self, reference_values):
        """Test that control variate reduces standard error."""
        ref_plain = reference_values["test_cases"]["gbm_european_call_plain"]
        params_plain = ref_plain["parameters"]

        # Plain MC
        model_plain = GeometricBrownianMotion(
            S0=params_plain["S0"],
            r=params_plain["r"],
            sigma=params_plain["sigma"],
            T=params_plain["T"],
            seed=params_plain["seed"],
        )
        payoff = EuropeanCallPayoff(strike=params_plain["K"])
        engine_plain = MonteCarloEngine(
            model=model_plain,
            payoff=payoff,
            n_paths=params_plain["n_paths"],
            antithetic=False,
            control_variate=False,
        )
        result_plain = engine_plain.price()

        # Control variate
        model_cv = GeometricBrownianMotion(
            S0=params_plain["S0"],
            r=params_plain["r"],
            sigma=params_plain["sigma"],
            T=params_plain["T"],
            seed=params_plain["seed"],
        )
        engine_cv = MonteCarloEngine(
            model=model_cv,
            payoff=payoff,
            n_paths=params_plain["n_paths"],
            antithetic=False,
            control_variate=True,
        )
        result_cv = engine_cv.price()

        # Control variate should reduce stderr
        assert result_cv.stderr < result_plain.stderr
        assert result_cv.control_variate_beta is not None
        assert abs(result_cv.control_variate_beta) > 0

    def test_pathwise_delta_approximation(self, reference_values):
        """Test that pathwise delta approximates analytical delta."""
        ref = reference_values["test_cases"]["gbm_pathwise_delta_call"]
        params = ref["parameters"]

        model = GeometricBrownianMotion(
            S0=params["S0"],
            r=params["r"],
            sigma=params["sigma"],
            T=params["T"],
            seed=params["seed"],
        )

        payoff = EuropeanCallPayoff(strike=params["K"])
        engine = MonteCarloEngine(
            model=model,
            payoff=payoff,
            n_paths=params["n_paths"],
            antithetic=False,
            control_variate=False,
        )

        greeks = engine.compute_greeks(option_type=params["option_type"], method=params["method"])

        assert greeks.delta is not None
        assert abs(greeks.delta.value - ref["reference_delta"]) < ref["tolerance"]


class TestHestonReferences:
    """Test Heston model stability."""

    def test_heston_black_scholes_limit(self, reference_values):
        """Test that Heston degenerates to Black-Scholes when xi=0."""
        ref = reference_values["test_cases"]["heston_black_scholes_limit"]
        params = ref["parameters"]

        # Heston with no volatility of volatility
        model = HestonModel(
            S0=params["S0"],
            r=params["r"],
            T=params["T"],
            kappa=params["kappa"],
            theta=params["theta"],
            xi=params["xi"],
            rho=params["rho"],
            v0=params["v0"],
            seed=params["seed"],
        )

        payoff = EuropeanCallPayoff(strike=params["K"])
        engine = HestonMonteCarloEngine(
            model=model,
            payoff=payoff,
            n_paths=params["n_paths"],
            n_steps=params["n_steps"],
            seed=params["seed"],
        )

        result = engine.price()

        # Check convergence to Black-Scholes
        bs_price_val = ref["black_scholes_price"]
        error = abs(result.price - bs_price_val)

        # Should be within 3 standard errors
        assert error < ref["tolerance_stderr_multiple"] * result.stderr

        # Should be within 2% relative error
        relative_error = error / bs_price_val * 100
        assert relative_error < ref["tolerance_absolute_percent"]


class TestAmericanReferences:
    """Test American option bounds."""

    def test_american_put_early_exercise_premium(self, reference_values):
        """Test that American put >= European put."""
        ref = reference_values["test_cases"]["american_put_vs_european"]
        params = ref["parameters"]

        # American put
        model_am = GeometricBrownianMotion(
            S0=params["S0"],
            r=params["r"],
            sigma=params["sigma"],
            T=params["T"],
            seed=params["seed"],
        )

        american_result = price_american_lsm(
            model=model_am,
            strike=params["K"],
            option_type=params["option_type"],
            n_paths=params["n_paths"],
            n_steps=params["n_steps"],
            basis="poly2",
            seed=params["seed"],
        )

        # European put
        model_eu = GeometricBrownianMotion(
            S0=params["S0"],
            r=params["r"],
            sigma=params["sigma"],
            T=params["T"],
            seed=params["seed"],
        )

        payoff = EuropeanPutPayoff(strike=params["K"])
        engine_eu = MonteCarloEngine(
            model=model_eu,
            payoff=payoff,
            n_paths=params["n_paths"],
            antithetic=False,
            control_variate=False,
        )

        european_result = engine_eu.price()

        # American should be >= European (allowing for MC noise)
        # Use conservative tolerance due to different MC samples
        assert american_result.price >= european_result.price - 3 * european_result.stderr
