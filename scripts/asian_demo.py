#!/usr/bin/env python
"""
Demo: Asian option pricing with convergence analysis.

Demonstrates:
- Asian arithmetic call option pricing
- Convergence across different path counts
- Comparison of pseudo-random vs Sobol sequences (QMC)
"""

import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mc_pricer.models.gbm import GeometricBrownianMotion
from mc_pricer.payoffs.path_dependent import AsianArithmeticCallPayoff
from mc_pricer.pricers.monte_carlo import MonteCarloEngine


def main():
    """Run Asian option pricing demo."""
    print("\n" + "=" * 80)
    print("Asian Arithmetic Call Option - Convergence Analysis")
    print("=" * 80)

    # Market parameters
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 20
    seed = 42

    print("\nMarket Parameters:")
    print(f"  Spot Price (S0):        {S0:,.2f}")
    print(f"  Strike Price (K):       {K:,.2f}")
    print(f"  Risk-free Rate (r):     {r:.4f}")
    print(f"  Volatility (σ):         {sigma:.4f}")
    print(f"  Time to Maturity (T):   {T:.4f} years")
    print(f"  Time Steps (n_steps):   {n_steps}")
    print(f"  Random Seed:            {seed}")

    # Path counts to test
    path_counts = [2000, 5000, 10000, 50000]

    print("\n" + "-" * 80)
    print("Convergence Table")
    print("-" * 80)
    print(
        f"{'n_paths':<10} {'RNG':<10} {'Price':<12} "
        f"{'Stderr':<12} {'CI Width':<12} {'Time (s)':<10}"
    )
    print("-" * 80)

    results = {}

    for n_paths in path_counts:
        for rng_type in ["pseudo", "sobol"]:
            # Create model
            model = GeometricBrownianMotion(
                S0=S0, r=r, sigma=sigma, T=T, seed=seed
            )

            # Create payoff and engine
            payoff = AsianArithmeticCallPayoff(strike=K)
            engine = MonteCarloEngine(
                model=model,
                payoff=payoff,
                n_paths=n_paths,
                antithetic=False
            )

            # Time the pricing
            import time
            start = time.time()
            result = engine.price_path_dependent(n_steps=n_steps, rng_type=rng_type)
            elapsed = time.time() - start

            ci_width = result.ci_upper - result.ci_lower

            print(
                f"{n_paths:<10} {rng_type:<10} {result.price:<12.6f} "
                f"{result.stderr:<12.6f} {ci_width:<12.6f} {elapsed:<10.4f}"
            )

            # Store for comparison
            key = (n_paths, rng_type)
            results[key] = result

    # Analysis
    print("\n" + "=" * 80)
    print("Analysis")
    print("=" * 80)

    # Compare QMC vs Pseudo at highest path count
    n_high = path_counts[-1]
    pseudo_result = results[(n_high, "pseudo")]
    sobol_result = results[(n_high, "sobol")]

    print(f"\nAt n_paths={n_high}:")
    print(f"  Pseudo-random:  Price={pseudo_result.price:.6f}, Stderr={pseudo_result.stderr:.6f}")
    print(f"  Sobol (QMC):    Price={sobol_result.price:.6f}, Stderr={sobol_result.stderr:.6f}")

    stderr_reduction = (pseudo_result.stderr - sobol_result.stderr) / pseudo_result.stderr * 100
    if stderr_reduction > 0:
        print(f"  → QMC reduces stderr by {stderr_reduction:.1f}%")
    else:
        print(f"  → Pseudo has {-stderr_reduction:.1f}% lower stderr in this run")

    # Convergence rate
    print("\nConvergence (Pseudo-random):")
    for i in range(1, len(path_counts)):
        n1, n2 = path_counts[i-1], path_counts[i]
        stderr1 = results[(n1, "pseudo")].stderr
        stderr2 = results[(n2, "pseudo")].stderr
        ratio = stderr1 / stderr2
        expected_ratio = np.sqrt(n2 / n1)
        print(
            f"  {n1:>6} → {n2:>6}: stderr ratio = {ratio:.3f} "
            f"(expected √{n2/n1:.1f} = {expected_ratio:.3f})"
        )

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("\nAsian options are path-dependent:")
    print("  • Payoff based on average price over the path")
    print("  • Requires full path simulation (n_steps time points)")
    print("  • Lower variance than vanilla options (averaging reduces volatility)")
    print("\nQMC (Sobol sequences) typically provides:")
    print("  • Better convergence rate: O(1/N) vs O(1/√N)")
    print("  • Lower standard errors for the same path count")
    print("  • Deterministic results (reproducible with same seed)")
    print()


if __name__ == "__main__":
    main()
