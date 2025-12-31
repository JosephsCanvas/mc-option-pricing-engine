"""
QMC Dimension Guard Demo

Demonstrates the dimensional limits of Sobol QMC sequences and how to handle them.
The Sobol generator supports up to 21 dimensions.
"""

import numpy as np

from mc_pricer.models.multi_gbm import MultiAssetGeometricBrownianMotion
from mc_pricer.payoffs.multi_asset import BasketArithmeticCallPayoff
from mc_pricer.pricers.multi_asset_monte_carlo import MultiAssetMonteCarloEngine


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"{title:^80}")
    print("=" * 80)


def terminal_dimension_limits() -> None:
    """Demonstrate dimension limits for terminal simulation."""
    print_section("TERMINAL SIMULATION: Dimension Limits")

    print("\nSobol sequences support up to 21 dimensions.")
    print("For terminal simulation, required_dim = n_assets\n")

    # Test cases
    test_cases = [
        (5, "OK", "Well within limit"),
        (10, "OK", "Acceptable"),
        (20, "OK", "Near limit"),
        (21, "OK", "At limit"),
        (22, "FAIL", "Exceeds limit - use pseudo RNG"),
    ]

    print(f"{'N Assets':<12} {'Dimension':<12} {'Sobol?':<10} {'Status'}")
    print("-" * 60)

    for n_assets, expected_status, note in test_cases:
        # Create model
        S0 = np.ones(n_assets) * 100.0
        sigma = np.ones(n_assets) * 0.2
        corr = np.eye(n_assets)  # Independent for simplicity

        try:
            model = MultiAssetGeometricBrownianMotion(
                S0=S0, r=0.05, sigma=sigma, T=1.0, corr=corr, seed=42
            )

            payoff = BasketArithmeticCallPayoff(strike=100.0)

            engine = MultiAssetMonteCarloEngine(
                model=model, payoff=payoff, n_paths=1000, rng_type="sobol", seed=42
            )

            _ = engine.price()  # Successfully priced - result unused for this demo
            status = f"✓ {expected_status}"
            print(f"{n_assets:<12} {n_assets:<12} {status:<10} {note}")

        except ValueError as e:
            status = f"✗ {expected_status}"
            print(f"{n_assets:<12} {n_assets:<12} {status:<10} {note}")
            if n_assets == 22:
                print(f"  Error: {str(e)[:70]}...")


def path_dimension_limits() -> None:
    """Demonstrate dimension limits for path simulation."""
    print_section("PATH SIMULATION: Dimension Limits")

    print("\nFor path simulation, required_dim = n_assets × n_steps")
    print("This can quickly exceed the 21-dimensional limit.\n")

    # Test cases: (n_assets, n_steps, expected_status, note)
    test_cases = [
        (2, 5, "OK", "2×5 = 10 dimensions"),
        (3, 5, "OK", "3×5 = 15 dimensions"),
        (2, 10, "OK", "2×10 = 20 dimensions"),
        (3, 7, "OK", "3×7 = 21 dimensions"),
        (3, 8, "FAIL", "3×8 = 24 dimensions (exceeds limit)"),
        (5, 5, "FAIL", "5×5 = 25 dimensions (exceeds limit)"),
    ]

    print(f"{'N Assets':<12} {'N Steps':<12} {'Dimension':<12} {'Sobol?':<10} {'Status'}")
    print("-" * 75)

    for n_assets, n_steps, expected_status, note in test_cases:
        dim = n_assets * n_steps

        # Create simple model
        S0 = np.ones(n_assets) * 100.0
        sigma = np.ones(n_assets) * 0.2
        corr = np.eye(n_assets)

        try:
            model = MultiAssetGeometricBrownianMotion(
                S0=S0, r=0.05, sigma=sigma, T=1.0, corr=corr, seed=42
            )

            _ = model.simulate_paths(  # Successfully simulated - paths unused for this demo
                n_paths=1000, n_steps=n_steps, rng_type="sobol", scramble=False
            )

            status = f"✓ {expected_status}"
            print(f"{n_assets:<12} {n_steps:<12} {dim:<12} {status:<10} {note}")

        except ValueError:
            status = f"✗ {expected_status}"
            print(f"{n_assets:<12} {n_steps:<12} {dim:<12} {status:<10} {note}")


def workarounds() -> None:
    """Demonstrate workarounds for dimension limits."""
    print_section("WORKAROUNDS FOR DIMENSION LIMITS")

    print("\nWhen you exceed 21 dimensions, you have three options:\n")

    print("1. REDUCE N_STEPS (for path-dependent options)")
    print("   - Use coarser time discretization")
    print("   - Example: Use 5 steps instead of 10\n")

    print("2. REDUCE N_ASSETS")
    print("   - Focus on fewer underlying assets")
    print("   - Consider synthetic baskets or factor models\n")

    print("3. USE PSEUDO-RANDOM RNG")
    print("   - Falls back to standard Monte Carlo")
    print("   - Still effective with antithetic variates")
    print("   - Higher paths may be needed for same accuracy\n")

    # Practical example
    print("Example: 5-asset basket with many steps\n")

    n_assets = 5
    S0 = np.ones(n_assets) * 100.0
    sigma = np.ones(n_assets) * 0.2
    corr = np.eye(n_assets)

    model = MultiAssetGeometricBrownianMotion(S0=S0, r=0.05, sigma=sigma, T=1.0, corr=corr, seed=42)

    payoff = BasketArithmeticCallPayoff(strike=100.0)

    # Sobol would fail with many steps, so use pseudo
    print("  ✓ Solution: Use pseudo RNG with more paths")

    engine = MultiAssetMonteCarloEngine(
        model=model, payoff=payoff, n_paths=100000, rng_type="pseudo", antithetic=True, seed=42
    )

    result = engine.price()

    print("\n  Results:")
    print("    Method: Pseudo-random + Antithetic")
    print(f"    N_paths: {result.n_paths:,}")
    print(f"    Price: {result.price:.4f}")
    print(f"    Stderr: {result.stderr:.6f}")
    print(f"    95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")


def main() -> None:
    """Run QMC dimension guard demos."""
    print("\n" + "=" * 80)
    print(" " * 20 + "QMC DIMENSION LIMITS AND WORKAROUNDS")
    print("=" * 80)
    print("\nSobol quasi-random sequences are limited to 21 dimensions.")
    print("This demo shows how this affects multi-asset pricing.")

    terminal_dimension_limits()
    path_dimension_limits()
    workarounds()

    print("\n" + "=" * 80)
    print("Demo complete! Key takeaway: Plan dimensions carefully with QMC.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
