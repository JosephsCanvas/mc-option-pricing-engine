#!/usr/bin/env python
"""
Convergence demonstration comparing variance reduction techniques.

Compares plain MC, antithetic variates, control variates, and combined methods
across different sample sizes.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mc_pricer.models.gbm import GeometricBrownianMotion
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff
from mc_pricer.pricers.monte_carlo import MonteCarloEngine


def main():
    """Run convergence analysis across different methods and sample sizes."""
    # Fixed parameters for reproducibility
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    seed = 42

    # Sample sizes to test
    n_paths_list = [1000, 5000, 20000, 100000]

    # Print header
    print("=" * 90)
    print("Monte Carlo Convergence Analysis")
    print("=" * 90)
    print(f"\nParameters: S0={S0}, K={K}, r={r}, sigma={sigma}, T={T}")
    print(f"Seed: {seed}\n")
    print("-" * 90)
    print(f"{'Method':<30} {'n_paths':<10} {'Price':<12} {'Std Error':<12} {'CI Width':<12}")
    print("-" * 90)

    methods = [
        ("Plain MC", False, False),
        ("Antithetic", True, False),
        ("Control Variate", False, True),
        ("Antithetic + Control Variate", True, True),
    ]

    for n_paths in n_paths_list:
        for method_name, antithetic, control_variate in methods:
            # Create model and payoff
            model = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
            payoff = EuropeanCallPayoff(strike=K)

            # Create engine with specified variance reduction techniques
            engine = MonteCarloEngine(
                model=model,
                payoff=payoff,
                n_paths=n_paths,
                antithetic=antithetic,
                control_variate=control_variate,
                seed=seed,
            )

            # Price the option
            result = engine.price()

            # Calculate CI width
            ci_width = result.ci_upper - result.ci_lower

            # Print results
            print(
                f"{method_name:<30} {n_paths:<10,} "
                f"{result.price:<12.6f} {result.stderr:<12.6f} {ci_width:<12.6f}"
            )

        print("-" * 90)

    print("\nObservations:")
    print("  • Standard error decreases as n_paths increases (O(1/√n) convergence)")
    print("  • Antithetic variates reduce variance for symmetric payoffs")
    print("  • Control variates leverage correlation with S_T (known expectation)")
    print("  • Combined methods can achieve further variance reduction")
    print("=" * 90)


if __name__ == "__main__":
    main()
