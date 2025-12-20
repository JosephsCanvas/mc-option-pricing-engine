#!/usr/bin/env python
"""
Greeks convergence demonstration.

Shows how Delta and Vega estimates converge with increasing sample size.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from mc_pricer.models.gbm import GeometricBrownianMotion
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff
from mc_pricer.pricers.monte_carlo import MonteCarloEngine
from tests.utils.black_scholes import (
    black_scholes_delta_call,
    black_scholes_vega,
)


def main():
    """Run Greeks convergence analysis."""
    # Fixed parameters
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    seed = 42

    # Sample sizes to test
    n_paths_list = [1000, 5000, 20000, 100000]

    # Compute Black-Scholes target values
    bs_delta = black_scholes_delta_call(S0, K, r, sigma, T)
    bs_vega = black_scholes_vega(S0, K, r, sigma, T)

    print("=" * 100)
    print("Greeks Convergence Analysis - Pathwise Estimator")
    print("=" * 100)
    print(f"\nParameters: S0={S0}, K={K}, r={r}, sigma={sigma}, T={T}")
    print("Option Type: European Call")
    print(f"Seed: {seed}")
    print(f"\nBlack-Scholes Targets: Delta={bs_delta:.6f}, Vega={bs_vega:.4f}\n")

    print("-" * 120)
    print(
        f"{'n_paths':<12} {'Delta':<10} {'DeltaErr':<10} {'Delta SE':<10} "
        f"{'Vega':<10} {'VegaErr':<10} {'Vega SE':<10} "
        f"{'Delta CI':<22} {'Vega CI':<22}"
    )
    print("-" * 120)

    for n_paths in n_paths_list:
        # Create model and payoff
        model = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        payoff = EuropeanCallPayoff(strike=K)

        # Create engine with control variate for better convergence
        engine = MonteCarloEngine(
            model=model,
            payoff=payoff,
            n_paths=n_paths,
            control_variate=True,
            seed=seed
        )

        # Compute pathwise Greeks
        greeks = engine.compute_greeks(option_type='call', method='pw')

        # Extract results
        delta = greeks.delta.value
        delta_se = greeks.delta.standard_error
        delta_ci = f"[{greeks.delta.ci_lower:.6f}, {greeks.delta.ci_upper:.6f}]"
        delta_err = delta - bs_delta

        vega = greeks.vega.value
        vega_se = greeks.vega.standard_error
        vega_ci = f"[{greeks.vega.ci_lower:.4f}, {greeks.vega.ci_upper:.4f}]"
        vega_err = vega - bs_vega

        # Print row
        print(
            f"{n_paths:<12,} {delta:<10.6f} {delta_err:>10.6f} {delta_se:<10.6f} "
            f"{vega:<10.4f} {vega_err:>10.4f} {vega_se:<10.6f} "
            f"{delta_ci:<22} {vega_ci:<22}"
        )

    print("-" * 120)

    # Show FD estimates for largest n_paths
    print("\n" + "=" * 100)
    print("Finite Difference Comparison (n_paths=100,000)")
    print("=" * 100)

    n_paths = 100000
    model = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
    payoff = EuropeanCallPayoff(strike=K)
    engine = MonteCarloEngine(
        model=model,
        payoff=payoff,
        n_paths=n_paths,
        control_variate=True,
        seed=seed
    )

    # Compute FD Greeks
    greeks_fd = engine.compute_greeks(
        option_type='call',
        method='fd',
        fd_seeds=10,
        fd_step_spot=1e-4,
        fd_step_sigma=1e-4
    )

    print("\nMethod: Finite Difference")
    print(f"  Delta: {greeks_fd.delta.value:.6f} ± {greeks_fd.delta.standard_error:.6f}")
    print(f"    95% CI: [{greeks_fd.delta.ci_lower:.6f}, {greeks_fd.delta.ci_upper:.6f}]")
    print(f"  Vega:  {greeks_fd.vega.value:.4f} ± {greeks_fd.vega.standard_error:.4f}")
    print(f"    95% CI: [{greeks_fd.vega.ci_lower:.4f}, {greeks_fd.vega.ci_upper:.4f}]")
    print(f"  FD steps: h_spot={greeks_fd.fd_step_spot}, h_sigma={greeks_fd.fd_step_sigma}")

    print("\n" + "=" * 100)
    print("Analysis Complete")
    print("=" * 100)
    print("\nKey Observations:")
    print("  • Standard error decreases as O(1/√n)")
    print(f"  • Delta converges to Black-Scholes value ({bs_delta:.6f} for ATM call)")
    print(f"  • Vega converges to Black-Scholes value ({bs_vega:.4f} for these parameters)")
    print("  • Pathwise (PW) and Finite Difference (FD) estimates agree within CI")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
