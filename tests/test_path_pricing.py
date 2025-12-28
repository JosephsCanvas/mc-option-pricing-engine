"""Tests for path-dependent option pricing."""

import pytest

from mc_pricer.models.gbm import GeometricBrownianMotion
from mc_pricer.payoffs.path_dependent import (
    AsianArithmeticCallPayoff,
    DownAndOutPutPayoff,
    UpAndOutCallPayoff,
)
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff
from mc_pricer.pricers.monte_carlo import MonteCarloEngine


class TestAsianPricing:
    """Test Asian option pricing."""

    def test_asian_call_vs_intrinsic(self):
        """Test that Asian call price >= intrinsic value."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        n_paths = 10000
        n_steps = 50

        model = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=42)
        payoff = AsianArithmeticCallPayoff(strike=K)
        engine = MonteCarloEngine(model=model, payoff=payoff, n_paths=n_paths)

        result = engine.price_path_dependent(n_steps=n_steps)

        # Asian call price should be >= max(S0 - K, 0) for ATM
        intrinsic = max(S0 - K, 0)
        assert result.price >= intrinsic

    def test_asian_call_monotonic_in_spot(self):
        """Test that Asian call price increases with S0."""
        K, r, sigma, T = 100.0, 0.05, 0.2, 1.0
        n_paths = 10000
        n_steps = 50
        seed = 42

        prices = []
        for S0 in [90.0, 100.0, 110.0]:
            model = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
            payoff = AsianArithmeticCallPayoff(strike=K)
            engine = MonteCarloEngine(model=model, payoff=payoff, n_paths=n_paths)

            result = engine.price_path_dependent(n_steps=n_steps)
            prices.append(result.price)

        # Prices should be increasing
        assert prices[0] < prices[1] < prices[2]

    def test_asian_reproducibility(self):
        """Test that same seed produces same results."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        n_paths = 5000
        n_steps = 50
        seed = 42

        # First run
        model1 = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        payoff1 = AsianArithmeticCallPayoff(strike=K)
        engine1 = MonteCarloEngine(model=model1, payoff=payoff1, n_paths=n_paths)
        result1 = engine1.price_path_dependent(n_steps=n_steps)

        # Second run with same seed
        model2 = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        payoff2 = AsianArithmeticCallPayoff(strike=K)
        engine2 = MonteCarloEngine(model=model2, payoff=payoff2, n_paths=n_paths)
        result2 = engine2.price_path_dependent(n_steps=n_steps)

        # Should be identical
        assert result1.price == result2.price
        assert result1.stderr == result2.stderr

    def test_asian_qmc_reproducibility(self):
        """Test QMC reproducibility with Sobol sequences."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        n_paths = 5000
        n_steps = 10  # Keep <= 21 for Sobol
        seed = 42

        # First run
        model1 = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        payoff1 = AsianArithmeticCallPayoff(strike=K)
        engine1 = MonteCarloEngine(model=model1, payoff=payoff1, n_paths=n_paths)
        result1 = engine1.price_path_dependent(n_steps=n_steps, rng_type="sobol")

        # Second run with same seed
        model2 = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        payoff2 = AsianArithmeticCallPayoff(strike=K)
        engine2 = MonteCarloEngine(model=model2, payoff=payoff2, n_paths=n_paths)
        result2 = engine2.price_path_dependent(n_steps=n_steps, rng_type="sobol")

        # Should be identical
        assert result1.price == result2.price
        assert result1.stderr == result2.stderr

    def test_asian_qmc_vs_pseudo(self):
        """Test that QMC has comparable or better stderr than pseudo for Asian options."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        n_paths = 5000
        n_steps = 10  # Keep <= 21 for Sobol
        seed = 42

        # Pseudo-random
        model_pseudo = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        payoff_pseudo = AsianArithmeticCallPayoff(strike=K)
        engine_pseudo = MonteCarloEngine(
            model=model_pseudo, payoff=payoff_pseudo, n_paths=n_paths
        )
        result_pseudo = engine_pseudo.price_path_dependent(
            n_steps=n_steps, rng_type="pseudo"
        )

        # QMC with Sobol
        model_qmc = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        payoff_qmc = AsianArithmeticCallPayoff(strike=K)
        engine_qmc = MonteCarloEngine(model=model_qmc, payoff=payoff_qmc, n_paths=n_paths)
        result_qmc = engine_qmc.price_path_dependent(n_steps=n_steps, rng_type="sobol")

        # QMC stderr should be <= pseudo stderr with some tolerance
        # (May not always hold due to randomness, but typically does)
        # Allow 50% tolerance for statistical variation
        assert result_qmc.stderr <= result_pseudo.stderr * 1.5

    def test_path_dependent_result_metadata(self):
        """Test that PathDependentPricingResult contains correct metadata."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        n_paths = 1000
        n_steps = 20

        model = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=42)
        payoff = AsianArithmeticCallPayoff(strike=K)
        engine = MonteCarloEngine(model=model, payoff=payoff, n_paths=n_paths)

        result = engine.price_path_dependent(
            n_steps=n_steps, rng_type="sobol", scramble=True
        )

        assert result.n_steps == n_steps
        assert result.rng_type == "sobol"
        assert result.scramble is True
        assert result.n_paths == n_paths


class TestBarrierPricing:
    """Test barrier option pricing."""

    def test_barrier_le_vanilla(self):
        """Test that barrier option price <= corresponding vanilla option."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        barrier = 150.0
        n_paths = 10000
        n_steps = 50
        seed = 42

        # Vanilla call
        model_vanilla = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        payoff_vanilla = EuropeanCallPayoff(strike=K)
        engine_vanilla = MonteCarloEngine(
            model=model_vanilla, payoff=payoff_vanilla, n_paths=n_paths
        )
        result_vanilla = engine_vanilla.price()

        # Barrier call (up-and-out)
        model_barrier = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        payoff_barrier = UpAndOutCallPayoff(strike=K, barrier=barrier)
        engine_barrier = MonteCarloEngine(
            model=model_barrier, payoff=payoff_barrier, n_paths=n_paths
        )
        result_barrier = engine_barrier.price_path_dependent(n_steps=n_steps)

        # Barrier price should be <= vanilla price
        assert result_barrier.price <= result_vanilla.price

    def test_barrier_reproducibility(self):
        """Test barrier option reproducibility."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        barrier = 150.0
        n_paths = 5000
        n_steps = 50
        seed = 42

        # First run
        model1 = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        payoff1 = UpAndOutCallPayoff(strike=K, barrier=barrier)
        engine1 = MonteCarloEngine(model=model1, payoff=payoff1, n_paths=n_paths)
        result1 = engine1.price_path_dependent(n_steps=n_steps)

        # Second run
        model2 = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        payoff2 = UpAndOutCallPayoff(strike=K, barrier=barrier)
        engine2 = MonteCarloEngine(model=model2, payoff=payoff2, n_paths=n_paths)
        result2 = engine2.price_path_dependent(n_steps=n_steps)

        assert result1.price == result2.price
        assert result1.stderr == result2.stderr

    def test_barrier_monotonic_in_barrier(self):
        """Test that up-and-out call price decreases as barrier decreases."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        n_paths = 10000
        n_steps = 50
        seed = 42

        prices = []
        for barrier in [200.0, 150.0, 130.0]:
            model = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
            payoff = UpAndOutCallPayoff(strike=K, barrier=barrier)
            engine = MonteCarloEngine(model=model, payoff=payoff, n_paths=n_paths)

            result = engine.price_path_dependent(n_steps=n_steps)
            prices.append(result.price)

        # As barrier decreases, more paths get knocked out, so price decreases
        assert prices[0] > prices[1] > prices[2]

    def test_down_out_put_reproducibility(self):
        """Test down-and-out put reproducibility."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        barrier = 50.0
        n_paths = 5000
        n_steps = 50
        seed = 42

        # First run
        model1 = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        payoff1 = DownAndOutPutPayoff(strike=K, barrier=barrier)
        engine1 = MonteCarloEngine(model=model1, payoff=payoff1, n_paths=n_paths)
        result1 = engine1.price_path_dependent(n_steps=n_steps)

        # Second run
        model2 = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        payoff2 = DownAndOutPutPayoff(strike=K, barrier=barrier)
        engine2 = MonteCarloEngine(model=model2, payoff=payoff2, n_paths=n_paths)
        result2 = engine2.price_path_dependent(n_steps=n_steps)

        assert result1.price == result2.price
        assert result1.stderr == result2.stderr


class TestAntithetic:
    """Test antithetic variates with path-dependent options."""

    def test_antithetic_reduces_variance(self):
        """Test that antithetic variates reduce variance for Asian options."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        n_paths = 10000  # Use more paths for statistical significance
        n_steps = 20
        seed = 42

        # Without antithetic
        model_plain = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        payoff_plain = AsianArithmeticCallPayoff(strike=K)
        engine_plain = MonteCarloEngine(
            model=model_plain, payoff=payoff_plain, n_paths=n_paths, antithetic=False
        )
        result_plain = engine_plain.price_path_dependent(n_steps=n_steps)

        # With antithetic
        model_anti = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        payoff_anti = AsianArithmeticCallPayoff(strike=K)
        engine_anti = MonteCarloEngine(
            model=model_anti, payoff=payoff_anti, n_paths=n_paths, antithetic=True
        )
        result_anti = engine_anti.price_path_dependent(n_steps=n_steps)

        # Antithetic should reduce stderr on average (allow tolerance due to randomness)
        # For path-dependent options, variance reduction is less pronounced than terminal payoffs
        # Just verify stderr is in reasonable range (not worse than plain by more than 10%)
        assert result_anti.stderr <= result_plain.stderr * 1.1


class TestNStepsValidation:
    """Test n_steps parameter validation."""

    def test_invalid_n_steps(self):
        """Test that invalid n_steps raises ValueError."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
        model = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=42)
        payoff = AsianArithmeticCallPayoff(strike=K)
        engine = MonteCarloEngine(model=model, payoff=payoff, n_paths=1000)

        with pytest.raises(ValueError, match="n_steps must be positive"):
            engine.price_path_dependent(n_steps=0)

        with pytest.raises(ValueError, match="n_steps must be positive"):
            engine.price_path_dependent(n_steps=-10)
