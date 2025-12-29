"""Tests for Heston calibration module."""

import numpy as np
import pytest

from mc_pricer.analytics.implied_vol import implied_vol
from mc_pricer.calibration import (
    CalibrationConfig,
    CalibrationResult,
    HestonCalibrator,
    MarketQuote,
)
from mc_pricer.models.heston import HestonModel
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff
from mc_pricer.pricers.heston_monte_carlo import HestonMonteCarloEngine


@pytest.fixture
def simple_quotes():
    """Small set of synthetic quotes for fast testing."""
    # Generate from known params
    S0 = 100.0
    r = 0.05
    true_params = {
        "kappa": 2.0,
        "theta": 0.04,
        "xi": 0.3,
        "rho": -0.7,
        "v0": 0.04,
    }

    strikes = [90.0, 100.0, 110.0]
    maturities = [0.5, 1.0]

    quotes = []
    for T in maturities:
        model = HestonModel(
            S0=S0,
            r=r,
            T=T,
            kappa=true_params["kappa"],
            theta=true_params["theta"],
            xi=true_params["xi"],
            rho=true_params["rho"],
            v0=true_params["v0"],
            seed=42,
            scheme="qe",
        )

        for K in strikes:
            payoff = EuropeanCallPayoff(strike=K)
            engine = HestonMonteCarloEngine(
                model=model,
                payoff=payoff,
                n_paths=10000,
                n_steps=50,
                antithetic=True,
                seed=42,
            )
            result = engine.price()

            try:
                iv = implied_vol(
                    price=result.price,
                    S0=S0,
                    K=K,
                    r=r,
                    T=T,
                    option_type="call",
                )
                quotes.append(
                    MarketQuote(
                        strike=K,
                        maturity=T,
                        option_type="call",
                        implied_vol=iv,
                    )
                )
            except (ValueError, RuntimeError):
                continue

    return quotes, true_params


class TestMarketQuote:
    """Test MarketQuote dataclass."""

    def test_valid_quote(self):
        """Test valid quote creation."""
        quote = MarketQuote(
            strike=100.0,
            maturity=1.0,
            option_type="call",
            implied_vol=0.2,
        )
        assert quote.strike == 100.0
        assert quote.maturity == 1.0
        assert quote.option_type == "call"
        assert quote.implied_vol == 0.2
        assert quote.bid_ask_width is None

    def test_with_bid_ask(self):
        """Test quote with bid-ask width."""
        quote = MarketQuote(
            strike=100.0,
            maturity=1.0,
            option_type="call",
            implied_vol=0.2,
            bid_ask_width=0.01,
        )
        assert quote.bid_ask_width == 0.01

    def test_invalid_strike(self):
        """Test that negative strike raises ValueError."""
        with pytest.raises(ValueError, match="strike must be positive"):
            MarketQuote(
                strike=-100.0,
                maturity=1.0,
                option_type="call",
                implied_vol=0.2,
            )

    def test_invalid_option_type(self):
        """Test that invalid option type raises ValueError."""
        with pytest.raises(ValueError, match="option_type must be"):
            MarketQuote(
                strike=100.0,
                maturity=1.0,
                option_type="invalid",
                implied_vol=0.2,
            )


class TestCalibrationConfig:
    """Test CalibrationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = CalibrationConfig()
        assert config.n_paths == 10000
        assert config.n_steps == 50
        assert config.rng_type == "pseudo"
        assert config.scramble is False
        assert config.seeds == [42]
        assert config.max_iter == 200
        assert config.tol == 1e-6
        assert "kappa" in config.bounds

    def test_custom_config(self):
        """Test custom configuration."""
        config = CalibrationConfig(
            n_paths=5000,
            n_steps=30,
            seeds=[1, 2, 3],
            max_iter=50,
        )
        assert config.n_paths == 5000
        assert config.n_steps == 30
        assert config.seeds == [1, 2, 3]
        assert config.max_iter == 50

    def test_invalid_n_paths(self):
        """Test that invalid n_paths raises ValueError."""
        with pytest.raises(ValueError, match="n_paths must be positive"):
            CalibrationConfig(n_paths=-100)

    def test_invalid_seeds(self):
        """Test that empty seeds list raises ValueError."""
        with pytest.raises(ValueError, match="seeds list cannot be empty"):
            CalibrationConfig(seeds=[])


class TestHestonCalibrator:
    """Test HestonCalibrator."""

    def test_initialization(self, simple_quotes):
        """Test calibrator initialization."""
        quotes, _ = simple_quotes
        config = CalibrationConfig(n_paths=1000, n_steps=20, seeds=[42])

        calibrator = HestonCalibrator(S0=100.0, r=0.05, quotes=quotes, config=config)

        assert calibrator.S0 == 100.0
        assert calibrator.r == 0.05
        assert len(calibrator.quotes) == len(quotes)
        assert calibrator.n_evals == 0

    def test_weights_uniform(self, simple_quotes):
        """Test uniform weights when no bid-ask provided."""
        quotes, _ = simple_quotes
        config = CalibrationConfig(n_paths=1000, n_steps=20, seeds=[42])

        calibrator = HestonCalibrator(S0=100.0, r=0.05, quotes=quotes, config=config)

        # Weights should be normalized to sum to n_quotes
        assert np.allclose(np.sum(calibrator.weights), len(quotes))

    def test_reproducibility(self, simple_quotes):
        """Test that same seed produces identical results."""
        quotes, _ = simple_quotes
        config = CalibrationConfig(
            n_paths=2000,
            n_steps=20,
            seeds=[42],
            max_iter=10,  # Small for speed
        )

        initial_guess = {
            "kappa": 1.5,
            "theta": 0.05,
            "xi": 0.4,
            "rho": -0.6,
            "v0": 0.05,
        }

        # Run twice with same seed
        calibrator1 = HestonCalibrator(S0=100.0, r=0.05, quotes=quotes, config=config)
        result1 = calibrator1.calibrate(initial_guess=initial_guess)

        calibrator2 = HestonCalibrator(S0=100.0, r=0.05, quotes=quotes, config=config)
        result2 = calibrator2.calibrate(initial_guess=initial_guess)

        # Should get identical results
        assert result1.objective_value == result2.objective_value
        for name in ["kappa", "theta", "xi", "rho", "v0"]:
            assert result1.best_params[name] == result2.best_params[name]

    def test_objective_decreases(self, simple_quotes):
        """Test that objective value decreases from initial guess."""
        quotes, _ = simple_quotes
        config = CalibrationConfig(
            n_paths=2000,
            n_steps=20,
            seeds=[42],
            max_iter=20,
        )

        # Poor initial guess
        initial_guess = {
            "kappa": 5.0,
            "theta": 0.1,
            "xi": 0.8,
            "rho": -0.3,
            "v0": 0.1,
        }

        calibrator = HestonCalibrator(S0=100.0, r=0.05, quotes=quotes, config=config)

        # Compute initial objective
        x0 = calibrator._dict_to_params(initial_guess)
        initial_obj = calibrator.objective(x0, config.seeds[0])

        # Run calibration
        result = calibrator.calibrate(initial_guess=initial_guess)

        # Final objective should be lower
        assert result.objective_value < initial_obj

    def test_parameter_recovery(self, simple_quotes):
        """Test recovery of known parameters from noiseless surface.

        This test may be sensitive to random sampling, so we use loose tolerances.
        """
        quotes, true_params = simple_quotes

        # Use more paths and steps for better accuracy
        config = CalibrationConfig(
            n_paths=5000,
            n_steps=30,
            seeds=[42, 123],  # 2 restarts
            max_iter=50,
            tol=1e-5,
            heston_scheme="qe",
        )

        # Start from true params with small perturbation
        initial_guess = {
            "kappa": true_params["kappa"] * 1.2,
            "theta": true_params["theta"] * 1.1,
            "xi": true_params["xi"] * 0.9,
            "rho": true_params["rho"] * 0.95,
            "v0": true_params["v0"] * 1.05,
        }

        calibrator = HestonCalibrator(S0=100.0, r=0.05, quotes=quotes, config=config)
        result = calibrator.calibrate(initial_guess=initial_guess)

        # Check that we recovered parameters reasonably well
        # Use loose tolerances due to MC noise
        for name in ["kappa", "theta", "xi", "rho", "v0"]:
            recovered = result.best_params[name]
            true_val = true_params[name]
            rel_error = abs((recovered - true_val) / true_val)

            # Allow up to 30% error due to MC noise and small sample
            error_msg = (
                f"{name}: {rel_error:.2%} error "
                f"(recovered={recovered:.4f}, true={true_val:.4f})"
            )
            assert rel_error < 0.3, error_msg

        # Objective should be small for noiseless data
        assert result.objective_value < 0.05

    def test_multiple_restarts(self, simple_quotes):
        """Test that multiple restarts work correctly."""
        quotes, _ = simple_quotes
        config = CalibrationConfig(
            n_paths=2000,
            n_steps=20,
            seeds=[42, 123, 456],  # 3 restarts
            max_iter=10,
        )

        initial_guess = {
            "kappa": 2.0,
            "theta": 0.04,
            "xi": 0.3,
            "rho": -0.7,
            "v0": 0.04,
        }

        calibrator = HestonCalibrator(S0=100.0, r=0.05, quotes=quotes, config=config)
        result = calibrator.calibrate(initial_guess=initial_guess)

        # Check diagnostics
        assert result.diagnostics["n_restarts"] == 3
        assert len(result.diagnostics["restart_results"]) == 3

        # Each restart should have run
        for restart_result in result.diagnostics["restart_results"]:
            assert restart_result["n_iterations"] > 0
            assert restart_result["final_value"] > 0

    def test_result_structure(self, simple_quotes):
        """Test that CalibrationResult has correct structure."""
        quotes, _ = simple_quotes
        config = CalibrationConfig(
            n_paths=1000,
            n_steps=20,
            seeds=[42],
            max_iter=5,  # Very small for speed
        )

        calibrator = HestonCalibrator(S0=100.0, r=0.05, quotes=quotes, config=config)
        result = calibrator.calibrate()

        # Check result attributes
        assert isinstance(result, CalibrationResult)
        assert isinstance(result.best_params, dict)
        assert "kappa" in result.best_params
        assert "theta" in result.best_params
        assert "xi" in result.best_params
        assert "rho" in result.best_params
        assert "v0" in result.best_params

        assert result.objective_value > 0
        assert result.n_evals > 0
        assert result.runtime_sec > 0

        assert len(result.fitted_vols) == len(quotes)
        assert len(result.target_vols) == len(quotes)
        assert len(result.residuals) == len(quotes)

        assert "n_restarts" in result.diagnostics
        assert "convergence_histories" in result.diagnostics
        assert "restart_results" in result.diagnostics

    def test_bounds_enforcement(self, simple_quotes):
        """Test that parameter bounds are enforced."""
        quotes, _ = simple_quotes
        config = CalibrationConfig(
            n_paths=1000,
            n_steps=20,
            seeds=[42],
            max_iter=10,
            bounds={
                "kappa": (0.5, 3.0),
                "theta": (0.01, 0.1),
                "xi": (0.1, 0.5),
                "rho": (-0.9, -0.5),
                "v0": (0.01, 0.1),
            },
        )

        calibrator = HestonCalibrator(S0=100.0, r=0.05, quotes=quotes, config=config)
        result = calibrator.calibrate()

        # Check that final params respect bounds
        for name, (lb, ub) in config.bounds.items():
            value = result.best_params[name]
            assert lb <= value <= ub, f"{name}={value} outside bounds [{lb}, {ub}]"
