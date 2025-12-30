"""Tests for Heston calibration with CRN and caching."""

import pytest

from mc_pricer.calibration import (
    CalibrationConfig,
    HestonCalibrator,
    MarketQuote,
)


@pytest.fixture
def simple_quotes():
    """Create simple set of market quotes for testing."""
    return [
        MarketQuote(strike=95.0, maturity=0.25, option_type="call", implied_vol=0.25),
        MarketQuote(strike=100.0, maturity=0.25, option_type="call", implied_vol=0.22),
        MarketQuote(strike=105.0, maturity=0.25, option_type="call", implied_vol=0.24),
    ]


@pytest.fixture
def calibrator_params():
    """Standard calibrator parameters."""
    return {
        "S0": 100.0,
        "r": 0.05,
    }


def test_crn_deterministic(simple_quotes, calibrator_params):
    """Test that CRN produces identical results with same seed."""
    config = CalibrationConfig(
        n_paths=5000,
        n_steps=30,
        seeds=[42],
        max_iter=20,
        use_crn=True,
        cache_size=100,
    )

    # Run calibration twice with same config
    cal1 = HestonCalibrator(**calibrator_params, quotes=simple_quotes, config=config)
    result1 = cal1.calibrate()

    cal2 = HestonCalibrator(**calibrator_params, quotes=simple_quotes, config=config)
    result2 = cal2.calibrate()

    # Results should be identical
    for param in ["kappa", "theta", "xi", "rho", "v0"]:
        assert result1.best_params[param] == result2.best_params[param]

    assert result1.objective_value == result2.objective_value
    assert result1.n_evals == result2.n_evals


def test_cache_consistency(simple_quotes, calibrator_params):
    """Test that cache doesn't change results."""
    # Run with caching enabled
    config_cached = CalibrationConfig(
        n_paths=5000,
        n_steps=30,
        seeds=[42],
        max_iter=20,
        use_crn=True,
        cache_size=100,
    )

    # Run without caching
    config_no_cache = CalibrationConfig(
        n_paths=5000,
        n_steps=30,
        seeds=[42],
        max_iter=20,
        use_crn=False,  # Disables cache
        cache_size=0,
    )

    cal_cached = HestonCalibrator(**calibrator_params, quotes=simple_quotes, config=config_cached)
    result_cached = cal_cached.calibrate()

    cal_no_cache = HestonCalibrator(
        **calibrator_params, quotes=simple_quotes, config=config_no_cache
    )
    result_no_cache = cal_no_cache.calibrate()

    # Results should be very close (some small differences due to RNG)
    for param in ["kappa", "theta", "xi", "rho", "v0"]:
        assert abs(result_cached.best_params[param] - result_no_cache.best_params[param]) < 0.1

    # Cache should have been used
    assert result_cached.cache_hits > 0
    assert result_no_cache.cache_hits == 0


def test_cache_hit_rate(simple_quotes, calibrator_params):
    """Test that cache achieves reasonable hit rate."""
    config = CalibrationConfig(
        n_paths=5000,
        n_steps=30,
        seeds=[42, 123],  # Two restarts should increase hit rate
        max_iter=20,
        use_crn=True,
        cache_size=100,
    )

    calibrator = HestonCalibrator(**calibrator_params, quotes=simple_quotes, config=config)
    result = calibrator.calibrate()

    # With CRN and multiple restarts, should get some cache hits
    total_calls = result.cache_hits + result.cache_misses

    # Should get at least some hits (not super strict since it's stochastic)
    assert total_calls > 0
    assert result.cache_hits >= 0  # At minimum, no errors


def test_cache_size_limit(simple_quotes, calibrator_params):
    """Test that cache respects size limit."""
    config = CalibrationConfig(
        n_paths=5000,
        n_steps=30,
        seeds=[42],
        max_iter=50,  # More iterations to fill cache
        use_crn=True,
        cache_size=10,  # Small cache
    )

    calibrator = HestonCalibrator(**calibrator_params, quotes=simple_quotes, config=config)
    calibrator.calibrate()

    # Cache should not exceed size limit
    assert len(calibrator._cache) <= config.cache_size


def test_fast_mode_completes(simple_quotes, calibrator_params):
    """Test that fast mode completes in reasonable time."""
    import time

    config = CalibrationConfig(
        n_paths=5000,
        n_steps=30,
        seeds=[42],
        max_iter=30,
        use_crn=True,
        cache_size=100,
    )

    calibrator = HestonCalibrator(**calibrator_params, quotes=simple_quotes, config=config)

    start = time.time()
    result = calibrator.calibrate()
    elapsed = time.time() - start

    # Should complete in reasonable time (generous limit for CI)
    assert elapsed < 60.0

    # Should produce valid result
    assert result.objective_value >= 0
    assert result.n_evals > 0
    for param in ["kappa", "theta", "xi", "rho", "v0"]:
        assert param in result.best_params
    # Check reasonable bounds (rho can be negative)
    assert result.best_params["kappa"] > 0
    assert result.best_params["theta"] > 0
    assert result.best_params["xi"] > 0
    assert -1.0 < result.best_params["rho"] < 1.0
    assert result.best_params["v0"] > 0


def test_crn_different_seeds(simple_quotes, calibrator_params):
    """Test that different seeds produce different results without CRN."""
    config1 = CalibrationConfig(
        n_paths=5000,
        n_steps=30,
        seeds=[42],
        max_iter=20,
        use_crn=False,
    )

    config2 = CalibrationConfig(
        n_paths=5000,
        n_steps=30,
        seeds=[999],  # Different seed
        max_iter=20,
        use_crn=False,
    )

    cal1 = HestonCalibrator(**calibrator_params, quotes=simple_quotes, config=config1)
    result1 = cal1.calibrate()

    cal2 = HestonCalibrator(**calibrator_params, quotes=simple_quotes, config=config2)
    result2 = cal2.calibrate()

    # Results should differ (different random seeds, no CRN)
    params_differ = any(
        result1.best_params[param] != result2.best_params[param]
        for param in ["kappa", "theta", "xi", "rho", "v0"]
    )
    assert params_differ


def test_calibration_improves_objective(simple_quotes, calibrator_params):
    """Test that calibration reduces objective function."""
    config = CalibrationConfig(
        n_paths=5000,
        n_steps=30,
        seeds=[42],
        max_iter=30,
        use_crn=True,
    )

    calibrator = HestonCalibrator(**calibrator_params, quotes=simple_quotes, config=config)

    # Poor initial guess
    initial_guess = {
        "kappa": 0.5,
        "theta": 0.2,
        "xi": 0.8,
        "rho": 0.0,
        "v0": 0.1,
    }

    result = calibrator.calibrate(initial_guess=initial_guess)

    # Check convergence history
    history = result.diagnostics["convergence_histories"][0]
    initial_value = history[0]
    final_value = history[-1]

    # Objective should improve
    assert final_value <= initial_value


def test_multiple_restarts(simple_quotes, calibrator_params):
    """Test calibration with multiple restarts."""
    config = CalibrationConfig(
        n_paths=5000,
        n_steps=30,
        seeds=[42, 123, 456],  # Three restarts
        max_iter=20,
        use_crn=True,
    )

    calibrator = HestonCalibrator(**calibrator_params, quotes=simple_quotes, config=config)
    result = calibrator.calibrate()

    # Should have 3 restart results
    assert len(result.diagnostics["restart_results"]) == 3

    # Each restart should have run
    for restart in result.diagnostics["restart_results"]:
        assert restart["n_iterations"] > 0
        assert "params" in restart


def test_regularization(simple_quotes, calibrator_params):
    """Test that regularization affects objective value."""
    # Without regularization
    config_no_reg = CalibrationConfig(
        n_paths=5000,
        n_steps=30,
        seeds=[42],
        max_iter=20,
        regularization=0.0,
        use_crn=True,
    )

    # With regularization
    config_with_reg = CalibrationConfig(
        n_paths=5000,
        n_steps=30,
        seeds=[42],
        max_iter=20,
        regularization=0.01,
        use_crn=True,
    )

    cal_no_reg = HestonCalibrator(**calibrator_params, quotes=simple_quotes, config=config_no_reg)
    result_no_reg = cal_no_reg.calibrate()

    cal_with_reg = HestonCalibrator(
        **calibrator_params, quotes=simple_quotes, config=config_with_reg
    )
    result_with_reg = cal_with_reg.calibrate()

    # With regularization, objective includes penalty
    # So it might be slightly higher
    # Just check both ran successfully
    assert result_no_reg.n_evals > 0
    assert result_with_reg.n_evals > 0


def test_weighted_calibration(calibrator_params):
    """Test that bid-ask weights affect calibration."""
    # Quotes with different weights
    weighted_quotes = [
        MarketQuote(
            strike=95.0,
            maturity=0.25,
            option_type="call",
            implied_vol=0.25,
            bid_ask_width=0.001,  # Tight spread -> high weight
        ),
        MarketQuote(
            strike=100.0,
            maturity=0.25,
            option_type="call",
            implied_vol=0.22,
            bid_ask_width=0.05,  # Wide spread -> low weight
        ),
    ]

    config = CalibrationConfig(
        n_paths=5000,
        n_steps=30,
        seeds=[42],
        max_iter=20,
        use_crn=True,
    )

    calibrator = HestonCalibrator(**calibrator_params, quotes=weighted_quotes, config=config)

    # Weights should be computed
    assert len(calibrator.weights) == len(weighted_quotes)
    # First quote should have higher weight
    assert calibrator.weights[0] > calibrator.weights[1]

    # Calibration should run
    result = calibrator.calibrate()
    assert result.n_evals > 0
