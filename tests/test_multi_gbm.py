"""Tests for multi-asset GBM model."""

import numpy as np
import pytest

from mc_pricer.models.multi_gbm import MultiAssetGeometricBrownianMotion


def test_basic_initialization():
    """Test that valid parameters initialize correctly."""
    S0 = np.array([100.0, 95.0, 110.0])
    r = 0.05
    sigma = np.array([0.2, 0.25, 0.18])
    T = 1.0
    corr = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])

    model = MultiAssetGeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, corr=corr, seed=42)

    assert model.n_assets == 3
    assert model.chol.shape == (3, 3)
    np.testing.assert_array_equal(model.S0, S0)


def test_invalid_dimensions():
    """Test that invalid dimensions raise ValueError."""
    # Single asset
    with pytest.raises(ValueError, match="at least 2 assets"):
        MultiAssetGeometricBrownianMotion(
            S0=np.array([100.0]),
            r=0.05,
            sigma=np.array([0.2]),
            T=1.0,
            corr=np.array([[1.0]]),
        )

    # Mismatched sigma
    with pytest.raises(ValueError, match="sigma must have shape"):
        MultiAssetGeometricBrownianMotion(
            S0=np.array([100.0, 95.0]),
            r=0.05,
            sigma=np.array([0.2]),
            T=1.0,
            corr=np.array([[1.0, 0.0], [0.0, 1.0]]),
        )

    # Mismatched corr
    with pytest.raises(ValueError, match="corr must have shape"):
        MultiAssetGeometricBrownianMotion(
            S0=np.array([100.0, 95.0]),
            r=0.05,
            sigma=np.array([0.2, 0.25]),
            T=1.0,
            corr=np.array([[1.0]]),
        )


def test_invalid_values():
    """Test that invalid parameter values raise ValueError."""
    # Negative S0
    with pytest.raises(ValueError, match="S0 must be positive"):
        MultiAssetGeometricBrownianMotion(
            S0=np.array([100.0, -95.0]),
            r=0.05,
            sigma=np.array([0.2, 0.25]),
            T=1.0,
            corr=np.array([[1.0, 0.0], [0.0, 1.0]]),
        )

    # Negative sigma
    with pytest.raises(ValueError, match="sigma must be non-negative"):
        MultiAssetGeometricBrownianMotion(
            S0=np.array([100.0, 95.0]),
            r=0.05,
            sigma=np.array([0.2, -0.25]),
            T=1.0,
            corr=np.array([[1.0, 0.0], [0.0, 1.0]]),
        )

    # Negative T
    with pytest.raises(ValueError, match="T must be non-negative"):
        MultiAssetGeometricBrownianMotion(
            S0=np.array([100.0, 95.0]),
            r=0.05,
            sigma=np.array([0.2, 0.25]),
            T=-1.0,
            corr=np.array([[1.0, 0.0], [0.0, 1.0]]),
        )


def test_invalid_correlation_matrix():
    """Test that invalid correlation matrices raise ValueError."""
    # Non-symmetric
    with pytest.raises(ValueError, match="symmetric"):
        MultiAssetGeometricBrownianMotion(
            S0=np.array([100.0, 95.0]),
            r=0.05,
            sigma=np.array([0.2, 0.25]),
            T=1.0,
            corr=np.array([[1.0, 0.5], [0.3, 1.0]]),
        )

    # Diagonal not 1
    with pytest.raises(ValueError, match="diagonal must be 1"):
        MultiAssetGeometricBrownianMotion(
            S0=np.array([100.0, 95.0]),
            r=0.05,
            sigma=np.array([0.2, 0.25]),
            T=1.0,
            corr=np.array([[0.9, 0.0], [0.0, 1.0]]),
        )

    # Values out of [-1, 1]
    with pytest.raises(ValueError, match="must be in"):
        MultiAssetGeometricBrownianMotion(
            S0=np.array([100.0, 95.0]),
            r=0.05,
            sigma=np.array([0.2, 0.25]),
            T=1.0,
            corr=np.array([[1.0, 1.5], [1.5, 1.0]]),
        )

    # Not positive semidefinite
    # Use a matrix that's actually not PSD: has negative eigenvalue
    with pytest.raises(ValueError, match="positive semidefinite"):
        MultiAssetGeometricBrownianMotion(
            S0=np.array([100.0, 95.0, 110.0]),
            r=0.05,
            sigma=np.array([0.2, 0.25, 0.18]),
            T=1.0,
            # This correlation matrix has eigenvalues: ~2.7, ~0.3, ~-0.04 (not PSD)
            corr=np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]]),
        )


def test_terminal_shapes():
    """Test that terminal simulation returns correct shapes."""
    model = MultiAssetGeometricBrownianMotion(
        S0=np.array([100.0, 95.0, 110.0]),
        r=0.05,
        sigma=np.array([0.2, 0.25, 0.18]),
        T=1.0,
        corr=np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]]),
        seed=42,
    )

    S_T = model.simulate_terminal(n_paths=1000, rng_type="pseudo")
    assert S_T.shape == (1000, 3)
    assert np.all(S_T > 0)


def test_paths_shapes():
    """Test that path simulation returns correct shapes."""
    model = MultiAssetGeometricBrownianMotion(
        S0=np.array([100.0, 95.0]),
        r=0.05,
        sigma=np.array([0.2, 0.25]),
        T=1.0,
        corr=np.array([[1.0, 0.5], [0.5, 1.0]]),
        seed=42,
    )

    paths = model.simulate_paths(n_paths=500, n_steps=10, rng_type="pseudo")
    assert paths.shape == (500, 11, 2)  # 11 = n_steps + 1
    np.testing.assert_array_equal(paths[:, 0, :], np.broadcast_to(model.S0, (500, 2)))
    assert np.all(paths > 0)


def test_reproducibility_with_seed():
    """Test that same seed gives identical results."""
    params = {
        "S0": np.array([100.0, 95.0]),
        "r": 0.05,
        "sigma": np.array([0.2, 0.25]),
        "T": 1.0,
        "corr": np.array([[1.0, 0.3], [0.3, 1.0]]),
        "seed": 123,
    }

    model1 = MultiAssetGeometricBrownianMotion(**params)
    model2 = MultiAssetGeometricBrownianMotion(**params)

    S_T1 = model1.simulate_terminal(n_paths=100, rng_type="pseudo")
    S_T2 = model2.simulate_terminal(n_paths=100, rng_type="pseudo")

    np.testing.assert_array_equal(S_T1, S_T2)


def test_correlation_enforcement():
    """Test that empirical correlation approximates specified correlation."""
    # Use high correlation to make effect visible
    corr_target = np.array([[1.0, 0.8], [0.8, 1.0]])

    model = MultiAssetGeometricBrownianMotion(
        S0=np.array([100.0, 100.0]),
        r=0.05,
        sigma=np.array([0.2, 0.2]),
        T=1.0,
        corr=corr_target,
        seed=42,
    )

    # Simulate large number of paths
    S_T = model.simulate_terminal(n_paths=50000, rng_type="pseudo")

    # Compute log returns: ln(S_T / S0)
    log_returns = np.log(S_T / model.S0)

    # Standardize to approximate Z
    Z_empirical = (log_returns - (model.r - 0.5 * model.sigma**2) * model.T) / (
        model.sigma * np.sqrt(model.T)
    )

    # Compute empirical correlation
    corr_empirical = np.corrcoef(Z_empirical.T)

    # Check off-diagonal correlation within tolerance
    # Use relaxed tolerance for MC
    np.testing.assert_allclose(corr_empirical[0, 1], corr_target[0, 1], atol=0.05, rtol=0.05)


def test_antithetic_requires_even_paths():
    """Test that antithetic requires even number of paths."""
    model = MultiAssetGeometricBrownianMotion(
        S0=np.array([100.0, 95.0]),
        r=0.05,
        sigma=np.array([0.2, 0.25]),
        T=1.0,
        corr=np.array([[1.0, 0.3], [0.3, 1.0]]),
    )

    with pytest.raises(ValueError, match="even"):
        model.simulate_terminal(n_paths=1001, antithetic=True)

    with pytest.raises(ValueError, match="even"):
        model.simulate_paths(n_paths=999, n_steps=10, antithetic=True)


def test_sobol_dimension_guard_terminal():
    """Test that Sobol raises error for dimension > 21."""
    # Create model with 22 assets (exceeds limit)
    S0 = np.ones(22) * 100.0
    sigma = np.ones(22) * 0.2
    corr = np.eye(22)  # Independent for simplicity

    model = MultiAssetGeometricBrownianMotion(S0=S0, r=0.05, sigma=sigma, T=1.0, corr=corr)

    with pytest.raises(ValueError, match="exceeds maximum of 21"):
        model.simulate_terminal(n_paths=100, rng_type="sobol")


def test_sobol_dimension_guard_paths():
    """Test that Sobol raises error for dimension > 21 in path simulation."""
    model = MultiAssetGeometricBrownianMotion(
        S0=np.array([100.0, 95.0, 110.0]),
        r=0.05,
        sigma=np.array([0.2, 0.25, 0.18]),
        T=1.0,
        corr=np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]]),
    )

    # 3 assets * 10 steps = 30 dimensions > 21
    with pytest.raises(ValueError, match="exceeds maximum of 21"):
        model.simulate_paths(n_paths=100, n_steps=10, rng_type="sobol")


def test_sobol_dimension_override():
    """Test qmc_dim_override parameter."""
    model = MultiAssetGeometricBrownianMotion(
        S0=np.array([100.0, 95.0]),
        r=0.05,
        sigma=np.array([0.2, 0.25]),
        T=1.0,
        corr=np.array([[1.0, 0.3], [0.3, 1.0]]),
    )

    # Valid override (>= required dimension of 2)
    S_T = model.simulate_terminal(n_paths=100, rng_type="sobol", qmc_dim_override=5)
    assert S_T.shape == (100, 2)

    # Invalid override (< required dimension)
    with pytest.raises(ValueError, match="qmc_dim_override"):
        model.simulate_terminal(n_paths=100, rng_type="sobol", qmc_dim_override=1)

    # Override still must be <= 21
    with pytest.raises(ValueError, match="exceeds maximum of 21"):
        model.simulate_terminal(n_paths=100, rng_type="sobol", qmc_dim_override=22)


def test_sobol_reproducibility():
    """Test Sobol sequence reproducibility with seed."""
    model = MultiAssetGeometricBrownianMotion(
        S0=np.array([100.0, 95.0]),
        r=0.05,
        sigma=np.array([0.2, 0.25]),
        T=1.0,
        corr=np.array([[1.0, 0.3], [0.3, 1.0]]),
        seed=456,
    )

    S_T1 = model.simulate_terminal(n_paths=100, rng_type="sobol", scramble=False)
    S_T2 = model.simulate_terminal(n_paths=100, rng_type="sobol", scramble=False)

    np.testing.assert_array_equal(S_T1, S_T2)


def test_scramble_affects_sobol():
    """Test that scramble produces different sequences."""
    model = MultiAssetGeometricBrownianMotion(
        S0=np.array([100.0, 95.0]),
        r=0.05,
        sigma=np.array([0.2, 0.25]),
        T=1.0,
        corr=np.array([[1.0, 0.3], [0.3, 1.0]]),
        seed=789,
    )

    S_T_no_scramble = model.simulate_terminal(n_paths=100, rng_type="sobol", scramble=False)
    S_T_scramble = model.simulate_terminal(n_paths=100, rng_type="sobol", scramble=True)

    # Should be different due to scrambling
    assert not np.allclose(S_T_no_scramble, S_T_scramble)


def test_independence_with_zero_correlation():
    """Test that zero correlation gives independent assets."""
    model = MultiAssetGeometricBrownianMotion(
        S0=np.array([100.0, 100.0]),
        r=0.05,
        sigma=np.array([0.2, 0.2]),
        T=1.0,
        corr=np.array([[1.0, 0.0], [0.0, 1.0]]),
        seed=111,
    )

    S_T = model.simulate_terminal(n_paths=50000, rng_type="pseudo")
    log_returns = np.log(S_T / model.S0)
    Z_empirical = (log_returns - (model.r - 0.5 * model.sigma**2) * model.T) / (
        model.sigma * np.sqrt(model.T)
    )

    corr_empirical = np.corrcoef(Z_empirical.T)

    # Off-diagonal should be near zero
    np.testing.assert_allclose(corr_empirical[0, 1], 0.0, atol=0.05)


def test_positive_prices():
    """Test that all simulated prices are positive."""
    model = MultiAssetGeometricBrownianMotion(
        S0=np.array([100.0, 95.0, 110.0]),
        r=0.05,
        sigma=np.array([0.3, 0.35, 0.28]),
        T=2.0,
        corr=np.array([[1.0, -0.5, 0.2], [-0.5, 1.0, -0.3], [0.2, -0.3, 1.0]]),
    )

    S_T = model.simulate_terminal(n_paths=10000)
    assert np.all(S_T > 0)

    paths = model.simulate_paths(n_paths=1000, n_steps=20)
    assert np.all(paths > 0)


def test_antithetic_produces_correct_count():
    """Test that antithetic produces exactly n_paths."""
    model = MultiAssetGeometricBrownianMotion(
        S0=np.array([100.0, 95.0]),
        r=0.05,
        sigma=np.array([0.2, 0.25]),
        T=1.0,
        corr=np.array([[1.0, 0.3], [0.3, 1.0]]),
        seed=222,
    )

    n_paths = 1000
    S_T = model.simulate_terminal(n_paths=n_paths, antithetic=True)
    assert S_T.shape[0] == n_paths

    paths = model.simulate_paths(n_paths=n_paths, n_steps=5, antithetic=True)
    assert paths.shape[0] == n_paths
