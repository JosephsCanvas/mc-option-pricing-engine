#!/usr/bin/env python
"""
Benchmark comparing pseudo-random and Quasi-Monte Carlo (Sobol) pricing.

Compares pricing accuracy and convergence for:
- GBM European call options
- Heston European call options

across different path counts: [2000, 5000, 20000, 100000]
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mc_pricer.analytics.black_scholes import bs_price
from mc_pricer.models.gbm import GeometricBrownianMotion
from mc_pricer.models.heston import HestonModel
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff
from mc_pricer.pricers.heston_monte_carlo import HestonMonteCarloEngine
from mc_pricer.pricers.monte_carlo import MonteCarloEngine


def benchmark_gbm():
    """Benchmark GBM pricing with pseudo-random vs Sobol."""
    print("\n" + "=" * 70)
    print("GBM European Call Option Benchmark")
    print("=" * 70)

    # Parameters
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    seed = 42

    # Analytical reference
    ref_price = bs_price(S0=S0, K=K, r=r, sigma=sigma, T=T, option_type="call")
    print(f"\nBlack-Scholes Reference Price: {ref_price:.6f}")

    # Path counts to test
    path_counts = [2000, 5000, 20000, 100000]

    print(f"\n{'Paths':<10} {'RNG':<8} {'Price':<12} {'Error':<12} {'Time (s)':<10}")
    print("-" * 70)

    for n_paths in path_counts:
        for rng_type in ["pseudo", "sobol"]:
            # Create model
            model = GeometricBrownianMotion(
                S0=S0, r=r, sigma=sigma, T=T, seed=seed, rng_type=rng_type, scramble=False
            )

            # Create payoff and engine
            payoff = EuropeanCallPayoff(strike=K)
            engine = MonteCarloEngine(
                model=model,
                payoff=payoff,
                n_paths=n_paths,
                antithetic=False,
                control_variate=False
            )

            # Time the pricing
            start = time.time()
            result = engine.price()
            elapsed = time.time() - start

            # Calculate error
            error = abs(result.price - ref_price)

            print(
                f"{n_paths:<10} {rng_type:<8} {result.price:<12.6f} "
                f"{error:<12.6f} {elapsed:<10.4f}"
            )


def benchmark_heston():
    """Benchmark Heston pricing with pseudo-random vs Sobol."""
    print("\n" + "=" * 70)
    print("Heston European Call Option Benchmark")
    print("=" * 70)

    # Parameters
    S0 = 100.0
    K = 100.0
    r = 0.05
    T = 1.0
    kappa = 2.0
    theta = 0.04
    xi = 0.3
    rho = -0.7
    v0 = 0.04
    n_steps = 100
    seed = 42

    # Reference price using high path count pseudo-random
    print("\nComputing reference price (1M paths, pseudo-random)...")
    ref_model = HestonModel(
        S0=S0, r=r, T=T, kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0,
        seed=seed, scheme="qe", rng_type="pseudo", scramble=False
    )
    ref_payoff = EuropeanCallPayoff(strike=K)
    ref_engine = HestonMonteCarloEngine(
        model=ref_model,
        payoff=ref_payoff,
        n_paths=1000000,
        n_steps=n_steps,
        antithetic=False
    )
    ref_result = ref_engine.price()
    ref_price = ref_result.price
    print(f"Reference Price: {ref_price:.6f}")

    # Path counts to test
    path_counts = [2000, 5000, 20000, 100000]

    print(f"\n{'Paths':<10} {'RNG':<8} {'Price':<12} {'Error':<12} {'Time (s)':<10}")
    print("-" * 70)

    for n_paths in path_counts:
        for rng_type in ["pseudo", "sobol"]:
            # Create model
            model = HestonModel(
                S0=S0, r=r, T=T, kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0,
                seed=seed, scheme="qe", rng_type=rng_type, scramble=False
            )

            # Create payoff and engine
            payoff = EuropeanCallPayoff(strike=K)
            engine = HestonMonteCarloEngine(
                model=model,
                payoff=payoff,
                n_paths=n_paths,
                n_steps=n_steps,
                antithetic=False
            )

            # Time the pricing
            start = time.time()
            result = engine.price()
            elapsed = time.time() - start

            # Calculate error
            error = abs(result.price - ref_price)

            print(
                f"{n_paths:<10} {rng_type:<8} {result.price:<12.6f} "
                f"{error:<12.6f} {elapsed:<10.4f}"
            )


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print("QMC vs Pseudo-Random Monte Carlo Benchmark")
    print("=" * 70)
    print("\nComparing convergence and timing for Sobol sequences vs")
    print("pseudo-random numbers across different path counts.")

    benchmark_gbm()
    benchmark_heston()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nQMC (Sobol sequences) typically provides:")
    print("  • Better convergence: O(1/N) vs O(1/√N) for pseudo-random")
    print("  • Lower pricing errors for the same number of paths")
    print("  • Deterministic results (with fixed seed)")
    print("\nBest used when:")
    print("  • Pricing vanilla options with low dimensionality")
    print("  • Variance reduction techniques are not already effective")
    print("  • Deterministic results are required")
    print()


if __name__ == "__main__":
    main()
