"""
Tests for Greeks computation (Delta and Vega).
"""

from mc_pricer.models.gbm import GeometricBrownianMotion
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff, EuropeanPutPayoff
from mc_pricer.pricers.monte_carlo import MonteCarloEngine
from tests.utils.black_scholes import (
    black_scholes_delta_call,
    black_scholes_delta_put,
    black_scholes_vega,
)


class TestPathwiseGreeks:
    """Tests for pathwise Greeks estimators."""

    def test_pw_greeks_match_black_scholes_call(self):
        """Test pathwise Greeks for call match Black-Scholes."""
        # Parameters
        S0 = 100.0
        K = 100.0
        r = 0.05
        sigma = 0.2
        T = 1.0
        n_paths = 100000
        seed = 42

        # Black-Scholes Greeks
        delta_bs = black_scholes_delta_call(S0, K, r, sigma, T)
        vega_bs = black_scholes_vega(S0, K, r, sigma, T)

        # Monte Carlo with pathwise Greeks
        model = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        payoff = EuropeanCallPayoff(strike=K)
        engine = MonteCarloEngine(
            model=model,
            payoff=payoff,
            n_paths=n_paths,
            control_variate=True,
            seed=seed
        )

        greeks = engine.compute_greeks(option_type='call', method='pw')

        # Check Delta
        assert greeks.delta is not None
        delta_pw = greeks.delta.value
        stderr_delta = greeks.delta.standard_error

        delta_error = abs(delta_pw - delta_bs)
        tolerance_delta = 3 * stderr_delta + 5e-3

        assert delta_error < tolerance_delta, (
            f"Delta mismatch: PW={delta_pw:.6f}, BS={delta_bs:.6f}, "
            f"error={delta_error:.6f}, tolerance={tolerance_delta:.6f}"
        )

        # Check Vega
        assert greeks.vega is not None
        vega_pw = greeks.vega.value
        stderr_vega = greeks.vega.standard_error

        vega_error = abs(vega_pw - vega_bs)
        tolerance_vega = 3 * stderr_vega + 0.2

        assert vega_error < tolerance_vega, (
            f"Vega mismatch: PW={vega_pw:.6f}, BS={vega_bs:.6f}, "
            f"error={vega_error:.6f}, tolerance={tolerance_vega:.6f}"
        )

    def test_pw_greeks_match_black_scholes_put(self):
        """Test pathwise Greeks for put match Black-Scholes."""
        # Parameters
        S0 = 100.0
        K = 100.0
        r = 0.05
        sigma = 0.2
        T = 1.0
        n_paths = 100000
        seed = 42

        # Black-Scholes Greeks
        delta_bs = black_scholes_delta_put(S0, K, r, sigma, T)
        vega_bs = black_scholes_vega(S0, K, r, sigma, T)

        # Monte Carlo with pathwise Greeks
        model = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        payoff = EuropeanPutPayoff(strike=K)
        engine = MonteCarloEngine(
            model=model,
            payoff=payoff,
            n_paths=n_paths,
            control_variate=True,
            seed=seed
        )

        greeks = engine.compute_greeks(option_type='put', method='pw')

        # Check Delta
        assert greeks.delta is not None
        delta_pw = greeks.delta.value
        stderr_delta = greeks.delta.standard_error

        delta_error = abs(delta_pw - delta_bs)
        tolerance_delta = 3 * stderr_delta + 5e-3

        assert delta_error < tolerance_delta, (
            f"Delta mismatch: PW={delta_pw:.6f}, BS={delta_bs:.6f}, "
            f"error={delta_error:.6f}, tolerance={tolerance_delta:.6f}"
        )

        # Check Vega
        assert greeks.vega is not None
        vega_pw = greeks.vega.value
        stderr_vega = greeks.vega.standard_error

        vega_error = abs(vega_pw - vega_bs)
        tolerance_vega = 3 * stderr_vega + 0.2

        assert vega_error < tolerance_vega, (
            f"Vega mismatch: PW={vega_pw:.6f}, BS={vega_bs:.6f}, "
            f"error={vega_error:.6f}, tolerance={tolerance_vega:.6f}"
        )


class TestFiniteDifferenceGreeks:
    """Tests for finite difference Greeks estimators."""

    def test_fd_agrees_with_pw_at_scale_call(self):
        """Test FD and PW Greeks agree for call options."""
        # Parameters
        S0 = 100.0
        K = 100.0
        r = 0.05
        sigma = 0.2
        T = 1.0
        n_paths = 50000  # Smaller for FD to keep runtime reasonable
        seed = 42

        # Monte Carlo engine
        model = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        payoff = EuropeanCallPayoff(strike=K)
        engine = MonteCarloEngine(
            model=model,
            payoff=payoff,
            n_paths=n_paths,
            control_variate=True,
            seed=seed
        )

        # Compute PW Greeks
        greeks_pw = engine.compute_greeks(option_type='call', method='pw')

        # Compute FD Greeks
        greeks_fd = engine.compute_greeks(
            option_type='call',
            method='fd',
            fd_seeds=10,
            fd_step_spot=1e-4,
            fd_step_sigma=1e-4
        )

        # Check Delta agreement
        delta_pw = greeks_pw.delta.value
        delta_fd = greeks_fd.delta.value
        stderr_delta_pw = greeks_pw.delta.standard_error
        stderr_delta_fd = greeks_fd.delta.standard_error

        delta_error = abs(delta_fd - delta_pw)
        tolerance_delta = 3 * max(stderr_delta_fd, stderr_delta_pw) + 1e-2

        assert delta_error < tolerance_delta, (
            f"Delta mismatch: FD={delta_fd:.6f}, PW={delta_pw:.6f}, "
            f"error={delta_error:.6f}, tolerance={tolerance_delta:.6f}"
        )

        # Check Vega agreement (looser tolerance)
        vega_pw = greeks_pw.vega.value
        vega_fd = greeks_fd.vega.value
        stderr_vega_pw = greeks_pw.vega.standard_error
        stderr_vega_fd = greeks_fd.vega.standard_error

        vega_error = abs(vega_fd - vega_pw)
        tolerance_vega = 3 * max(stderr_vega_fd, stderr_vega_pw) + 0.5

        assert vega_error < tolerance_vega, (
            f"Vega mismatch: FD={vega_fd:.6f}, PW={vega_pw:.6f}, "
            f"error={vega_error:.6f}, tolerance={tolerance_vega:.6f}"
        )

    def test_fd_agrees_with_pw_at_scale_put(self):
        """Test FD and PW Greeks agree for put options."""
        # Parameters
        S0 = 100.0
        K = 100.0
        r = 0.05
        sigma = 0.2
        T = 1.0
        n_paths = 50000
        seed = 42

        # Monte Carlo engine
        model = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        payoff = EuropeanPutPayoff(strike=K)
        engine = MonteCarloEngine(
            model=model,
            payoff=payoff,
            n_paths=n_paths,
            control_variate=True,
            seed=seed
        )

        # Compute PW Greeks
        greeks_pw = engine.compute_greeks(option_type='put', method='pw')

        # Compute FD Greeks
        greeks_fd = engine.compute_greeks(
            option_type='put',
            method='fd',
            fd_seeds=10,
            fd_step_spot=1e-4,
            fd_step_sigma=1e-4
        )

        # Check Delta agreement
        delta_pw = greeks_pw.delta.value
        delta_fd = greeks_fd.delta.value
        stderr_delta_pw = greeks_pw.delta.standard_error
        stderr_delta_fd = greeks_fd.delta.standard_error

        delta_error = abs(delta_fd - delta_pw)
        tolerance_delta = 3 * max(stderr_delta_fd, stderr_delta_pw) + 1e-2

        assert delta_error < tolerance_delta, (
            f"Delta mismatch: FD={delta_fd:.6f}, PW={delta_pw:.6f}, "
            f"error={delta_error:.6f}, tolerance={tolerance_delta:.6f}"
        )

        # Check Vega agreement
        vega_pw = greeks_pw.vega.value
        vega_fd = greeks_fd.vega.value
        stderr_vega_pw = greeks_pw.vega.standard_error
        stderr_vega_fd = greeks_fd.vega.standard_error

        vega_error = abs(vega_fd - vega_pw)
        tolerance_vega = 3 * max(stderr_vega_fd, stderr_vega_pw) + 0.5

        assert vega_error < tolerance_vega, (
            f"Vega mismatch: FD={vega_fd:.6f}, PW={vega_pw:.6f}, "
            f"error={vega_error:.6f}, tolerance={tolerance_vega:.6f}"
        )


class TestGreeksEdgeCases:
    """Tests for Greeks edge cases."""

    def test_greeks_itm_call(self):
        """Test Greeks for ITM call option."""
        S0 = 110.0
        K = 100.0
        r = 0.05
        sigma = 0.2
        T = 1.0
        n_paths = 100000
        seed = 42

        model = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        payoff = EuropeanCallPayoff(strike=K)
        engine = MonteCarloEngine(
            model=model,
            payoff=payoff,
            n_paths=n_paths,
            control_variate=True,
            seed=seed
        )

        greeks = engine.compute_greeks(option_type='call', method='pw')

        # ITM call should have delta close to 1
        assert greeks.delta.value > 0.7
        assert greeks.delta.value < 1.0

        # Vega should be positive
        assert greeks.vega.value > 0

    def test_greeks_otm_put(self):
        """Test Greeks for OTM put option."""
        S0 = 110.0
        K = 100.0
        r = 0.05
        sigma = 0.2
        T = 1.0
        n_paths = 100000
        seed = 42

        model = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        payoff = EuropeanPutPayoff(strike=K)
        engine = MonteCarloEngine(
            model=model,
            payoff=payoff,
            n_paths=n_paths,
            control_variate=True,
            seed=seed
        )

        greeks = engine.compute_greeks(option_type='put', method='pw')

        # OTM put should have delta close to 0
        assert greeks.delta.value < 0
        assert greeks.delta.value > -0.3

        # Vega should be positive
        assert greeks.vega.value > 0
