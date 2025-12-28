"""
Tests for Heston QE (Quadratic-Exponential) variance discretization scheme.
"""

import numpy as np
import pytest

from mc_pricer.analytics.black_scholes import bs_price
from mc_pricer.models.heston import HestonModel, sample_variance_qe
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff
from mc_pricer.pricers.heston_monte_carlo import HestonMonteCarloEngine


class TestQEVarianceSampling:
    """Test QE variance sampling function."""

    def test_qe_reproducibility(self):
        """Test that QE sampling is reproducible with same seed."""
        v = np.array([0.04, 0.05, 0.03])
        kappa, theta, xi, dt = 2.0, 0.04, 0.3, 0.01

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        z1 = rng1.standard_normal(3)
        z2 = rng2.standard_normal(3)

        v_next1 = sample_variance_qe(v, kappa, theta, xi, dt, z1)
        v_next2 = sample_variance_qe(v, kappa, theta, xi, dt, z2)

        np.testing.assert_array_equal(v_next1, v_next2)

    def test_qe_non_negative(self):
        """Test that QE always produces non-negative variance."""
        rng = np.random.default_rng(42)
        v = np.abs(rng.standard_normal(1000)) * 0.05
        kappa, theta, xi, dt = 2.0, 0.04, 0.3, 0.01
        z = rng.standard_normal(1000)

        v_next = sample_variance_qe(v, kappa, theta, xi, dt, z)

        assert np.all(v_next >= 0), "QE scheme produced negative variance"
        assert np.all(np.isfinite(v_next)), "QE scheme produced non-finite variance"

    def test_qe_mean_reversion(self):
        """Test that QE exhibits mean reversion towards theta."""
        rng = np.random.default_rng(42)
        n_paths = 10000
        n_steps = 100

        v = np.full(n_paths, 0.10)  # Start far from theta
        kappa, theta, xi, dt = 2.0, 0.04, 0.3, 0.01

        # Simulate forward
        for _ in range(n_steps):
            z = rng.standard_normal(n_paths)
            v = sample_variance_qe(v, kappa, theta, xi, dt, z)

        # After many steps, mean should be close to theta
        mean_v = np.mean(v)
        assert abs(mean_v - theta) < 0.01, "QE does not show mean reversion"

    def test_qe_regimes(self):
        """Test that both quadratic and exponential regimes are exercised."""
        rng = np.random.default_rng(42)

        # Low variance -> high psi -> exponential regime
        v_low = np.full(100, 0.001)
        z_low = rng.standard_normal(100)
        v_next_low = sample_variance_qe(v_low, 2.0, 0.04, 0.5, 0.1, z_low)

        # High variance -> low psi -> quadratic regime
        v_high = np.full(100, 0.20)
        z_high = rng.standard_normal(100)
        v_next_high = sample_variance_qe(v_high, 2.0, 0.04, 0.3, 0.01, z_high)

        # Both should produce valid results
        assert np.all(np.isfinite(v_next_low))
        assert np.all(np.isfinite(v_next_high))
        assert np.all(v_next_low >= 0)
        assert np.all(v_next_high >= 0)


class TestHestonModelQE:
    """Test Heston model with QE scheme."""

    def test_qe_scheme_initialization(self):
        """Test that QE scheme can be initialized."""
        model = HestonModel(
            S0=100.0,
            r=0.05,
            T=1.0,
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            rho=-0.7,
            v0=0.04,
            seed=42,
            scheme="qe"
        )
        assert model.scheme == "qe"

    def test_invalid_scheme_raises_error(self):
        """Test that invalid scheme raises ValueError."""
        with pytest.raises(ValueError, match="Invalid scheme"):
            HestonModel(
                S0=100.0,
                r=0.05,
                T=1.0,
                kappa=2.0,
                theta=0.04,
                xi=0.3,
                rho=-0.7,
                v0=0.04,
                scheme="invalid"
            )

    def test_qe_reproducibility(self):
        """Test that QE scheme produces reproducible results."""
        params = {
            "S0": 100.0,
            "r": 0.05,
            "T": 1.0,
            "kappa": 2.0,
            "theta": 0.04,
            "xi": 0.3,
            "rho": -0.7,
            "v0": 0.04,
            "scheme": "qe"
        }

        model1 = HestonModel(**params, seed=42)
        model2 = HestonModel(**params, seed=42)

        terminal1 = model1.simulate_terminal(n_paths=1000, n_steps=100)
        terminal2 = model2.simulate_terminal(n_paths=1000, n_steps=100)

        np.testing.assert_array_equal(terminal1, terminal2)

    def test_qe_no_nans(self):
        """Test that QE scheme produces finite positive prices."""
        model = HestonModel(
            S0=100.0,
            r=0.05,
            T=1.0,
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            rho=-0.7,
            v0=0.04,
            seed=42,
            scheme="qe"
        )

        terminal = model.simulate_terminal(n_paths=10000, n_steps=100)

        assert np.all(np.isfinite(terminal)), "QE produced non-finite prices"
        assert np.all(terminal > 0), "QE produced non-positive prices"

    def test_qe_antithetic_variates(self):
        """Test that QE scheme works with antithetic variates."""
        model = HestonModel(
            S0=100.0,
            r=0.05,
            T=1.0,
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            rho=-0.7,
            v0=0.04,
            seed=42,
            scheme="qe"
        )

        # Test with even number of paths
        n_paths = 100
        paths = model.simulate_paths(n_paths, n_steps=50, antithetic=True)
        assert paths.shape == (n_paths, 51)

        # Test that odd number raises error
        with pytest.raises(ValueError, match="n_paths must be even"):
            model.simulate_paths(n_paths=101, n_steps=50, antithetic=True)

    def test_qe_convergence_to_ft_euler(self):
        """Test that QE and FT Euler converge to similar prices."""
        n_paths = 50000
        n_steps = 200

        params = {
            "S0": 100.0,
            "r": 0.05,
            "T": 1.0,
            "kappa": 2.0,
            "theta": 0.04,
            "xi": 0.3,
            "rho": -0.7,
            "v0": 0.04
        }

        # Price with FT Euler
        model_ft = HestonModel(**params, seed=42, scheme="full_truncation_euler")
        payoff = EuropeanCallPayoff(strike=100.0)
        engine_ft = HestonMonteCarloEngine(
            model=model_ft, payoff=payoff, n_paths=n_paths, n_steps=n_steps
        )
        result_ft = engine_ft.price()

        # Price with QE
        model_qe = HestonModel(**params, seed=42, scheme="qe")
        engine_qe = HestonMonteCarloEngine(
            model=model_qe, payoff=payoff, n_paths=n_paths, n_steps=n_steps
        )
        result_qe = engine_qe.price()

        # Prices should be close (within 3 combined standard errors)
        combined_stderr = np.sqrt(result_ft.stderr**2 + result_qe.stderr**2)
        price_diff = abs(result_ft.price - result_qe.price)

        assert price_diff <= 3 * combined_stderr, (
            f"FT Euler and QE prices differ by {price_diff:.6f}, "
            f"which exceeds 3σ = {3 * combined_stderr:.6f}"
        )

        # Also check relative difference is reasonable
        rel_diff = price_diff / result_ft.price
        assert rel_diff < 0.02, (
            f"Relative difference {rel_diff:.4f} exceeds 2%"
        )

    def test_qe_black_scholes_limit(self):
        """Test that QE converges to Black-Scholes when xi=0."""
        # When xi=0, v0=theta, rho=0, Heston reduces to GBM
        S0, K, r, T = 100.0, 100.0, 0.05, 1.0
        sigma = 0.2
        theta = sigma**2

        # Compute BS price
        bs_call = bs_price(S0, K, r, T, sigma, "call")

        # Price with QE
        model = HestonModel(
            S0=S0,
            r=r,
            T=T,
            kappa=2.0,
            theta=theta,
            xi=0.0,  # No vol of vol
            rho=0.0,  # No correlation
            v0=theta,  # Start at long-term variance
            seed=42,
            scheme="qe"
        )

        payoff = EuropeanCallPayoff(strike=K)
        engine = HestonMonteCarloEngine(
            model=model,
            payoff=payoff,
            n_paths=100000,
            n_steps=200
        )
        result = engine.price()

        # Should match BS within 3 standard errors
        assert abs(result.price - bs_call) <= 3 * result.stderr, (
            f"QE price {result.price:.6f} differs from BS {bs_call:.6f} "
            f"by more than 3σ = {3 * result.stderr:.6f}"
        )

        # Also check relative error
        rel_error = abs(result.price - bs_call) / bs_call
        assert rel_error < 0.02, (
            f"Relative error {rel_error:.4f} exceeds 2%"
        )


class TestHestonPricingResultWithScheme:
    """Test that HestonPricingResult includes scheme information."""

    def test_result_includes_scheme(self):
        """Test that pricing result includes scheme field."""
        model = HestonModel(
            S0=100.0,
            r=0.05,
            T=1.0,
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            rho=-0.7,
            v0=0.04,
            seed=42,
            scheme="qe"
        )

        payoff = EuropeanCallPayoff(strike=100.0)
        engine = HestonMonteCarloEngine(
            model=model, payoff=payoff, n_paths=10000, n_steps=50
        )
        result = engine.price()

        assert hasattr(result, "scheme")
        assert result.scheme == "qe"

    def test_default_scheme_in_result(self):
        """Test that default scheme is recorded correctly."""
        model = HestonModel(
            S0=100.0,
            r=0.05,
            T=1.0,
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            rho=-0.7,
            v0=0.04,
            seed=42
            # No scheme specified -> default to full_truncation_euler
        )

        payoff = EuropeanCallPayoff(strike=100.0)
        engine = HestonMonteCarloEngine(
            model=model, payoff=payoff, n_paths=10000, n_steps=50
        )
        result = engine.price()

        assert result.scheme == "full_truncation_euler"
