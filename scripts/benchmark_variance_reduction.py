#!/usr/bin/env python
"""
Benchmark variance reduction techniques across multiple seeds.

Quantifies the effectiveness of different variance reduction methods by running
repeated trials with different random seeds and analyzing the distribution of
standard errors and confidence interval widths.
"""

import sys
from pathlib import Path

import numpy as np

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mc_pricer.models.gbm import GeometricBrownianMotion
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff
from mc_pricer.pricers.monte_carlo import MonteCarloEngine


def run_benchmark():
    """Run variance reduction benchmark across multiple seeds."""
    # Fixed parameters
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0

    # Test configurations
    n_paths_list = [5000, 20000, 100000]
    seeds = range(20)

    # Methods to test
    methods = [
        ("Plain MC", False, False),
        ("Antithetic", True, False),
        ("Control Variate", False, True),
        ("Antithetic + Control Variate", True, True),
    ]

    print("=" * 100)
    print("Variance Reduction Benchmark - Statistical Analysis Across Seeds")
    print("=" * 100)
    print(f"\nParameters: S0={S0}, K={K}, r={r}, sigma={sigma}, T={T}")
    print(f"Number of trials per configuration: {len(seeds)}")
    print(f"Seeds: {min(seeds)} to {max(seeds)}\n")

    # Store results for each configuration
    results = {}

    # Run simulations
    for n_paths in n_paths_list:
        print(f"\nRunning simulations with n_paths = {n_paths:,}...")

        for method_name, antithetic, control_variate in methods:
            stderr_list = []
            ci_width_list = []
            price_list = []
            beta_list = []

            for seed in seeds:
                # Create model and payoff
                model = GeometricBrownianMotion(
                    S0=S0, r=r, sigma=sigma, T=T, seed=seed
                )
                payoff = EuropeanCallPayoff(strike=K)

                # Create engine
                engine = MonteCarloEngine(
                    model=model,
                    payoff=payoff,
                    n_paths=n_paths,
                    antithetic=antithetic,
                    control_variate=control_variate,
                    seed=seed
                )

                # Price the option
                result = engine.price()

                # Record metrics
                price_list.append(result.price)
                stderr_list.append(result.stderr)
                ci_width_list.append(result.ci_upper - result.ci_lower)
                if result.control_variate_beta is not None:
                    beta_list.append(result.control_variate_beta)

            # Store results
            key = (method_name, n_paths)
            results[key] = {
                "prices": np.array(price_list),
                "stderrs": np.array(stderr_list),
                "ci_widths": np.array(ci_width_list),
                "betas": np.array(beta_list) if beta_list else None,
            }

    # Print results table
    print("\n" + "=" * 100)
    print("RESULTS: Statistical Summary Across Seeds")
    print("=" * 100)

    for n_paths in n_paths_list:
        print(f"\nn_paths = {n_paths:,}")
        print("-" * 100)
        print(
            f"{'Method':<30} {'Mean StdErr':<15} {'StdDev StdErr':<15} "
            f"{'Mean CI Width':<16} {'StdDev CI Width':<16}"
        )
        print("-" * 100)

        for method_name, _, _ in methods:
            key = (method_name, n_paths)
            data = results[key]

            mean_stderr = np.mean(data["stderrs"])
            std_stderr = np.std(data["stderrs"], ddof=1)
            mean_ci_width = np.mean(data["ci_widths"])
            std_ci_width = np.std(data["ci_widths"], ddof=1)

            print(
                f"{method_name:<30} {mean_stderr:<15.6f} {std_stderr:<15.6f} "
                f"{mean_ci_width:<16.6f} {std_ci_width:<16.6f}"
            )

        print("-" * 100)

    # Print variance reduction summary
    print("\n" + "=" * 100)
    print("VARIANCE REDUCTION SUMMARY")
    print("=" * 100)
    print("\nReduction in Mean Standard Error (compared to Plain MC):\n")

    for n_paths in n_paths_list:
        print(f"n_paths = {n_paths:,}")
        plain_key = ("Plain MC", n_paths)
        plain_mean_stderr = np.mean(results[plain_key]["stderrs"])

        for method_name, _, _ in methods:
            if method_name == "Plain MC":
                continue

            key = (method_name, n_paths)
            method_mean_stderr = np.mean(results[key]["stderrs"])
            reduction = (1 - method_mean_stderr / plain_mean_stderr) * 100

            print(f"  {method_name:<30} {reduction:>6.2f}% reduction")

        print()

    # Print beta statistics for control variate methods
    print("=" * 100)
    print("CONTROL VARIATE BETA COEFFICIENTS")
    print("=" * 100)

    for n_paths in n_paths_list:
        print(f"\nn_paths = {n_paths:,}")
        print("-" * 100)
        print(f"{'Method':<30} {'Mean Beta':<15} {'StdDev Beta':<15}")
        print("-" * 100)

        for method_name, _, control_variate in methods:
            if not control_variate:
                continue

            key = (method_name, n_paths)
            betas = results[key]["betas"]

            if betas is not None and len(betas) > 0:
                mean_beta = np.mean(betas)
                std_beta = np.std(betas, ddof=1)
                print(f"{method_name:<30} {mean_beta:<15.6f} {std_beta:<15.6f}")

        print("-" * 100)

    print("\n" + "=" * 100)
    print("Observations:")
    print("  • Lower mean standard error indicates better variance reduction")
    print("  • Consistent beta coefficients indicate stable control variate effectiveness")
    print("  • Combined methods (Antithetic + Control Variate) can provide cumulative benefits")
    print("=" * 100)


if __name__ == "__main__":
    run_benchmark()
