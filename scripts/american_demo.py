#!/usr/bin/env python
"""
American option pricing demo comparing European vs American put prices.

Demonstrates the early exercise premium for American options using LSM.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mc_pricer.models.gbm import GeometricBrownianMotion
from mc_pricer.payoffs.plain_vanilla import EuropeanPutPayoff
from mc_pricer.pricers.lsm import price_american_lsm
from mc_pricer.pricers.monte_carlo import MonteCarloEngine


def main():
    """Run American option pricing demo."""
    # Parameters
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    seed = 42

    # LSM parameters
    n_steps = 50
    basis = "poly2"

    # Different path counts for convergence
    path_counts = [5000, 20000, 100000]

    print("=" * 100)
    print("American Option Pricing Demo - Longstaff-Schwartz (LSM) Algorithm")
    print("=" * 100)
    print(f"\nParameters: S0={S0}, K={K}, r={r}, sigma={sigma}, T={T}")
    print(f"LSM: n_steps={n_steps}, basis={basis}, seed={seed}")
    print("\nComparing European Put vs American Put prices:")
    print("-" * 100)
    print(f"{'n_paths':<12} {'Euro Put Price':<18} {'Euro SE':<12} "
          f"{'Amer Put Price':<18} {'Amer SE':<12} {'Premium':<12}")
    print("-" * 100)

    for n_paths in path_counts:
        # European put
        model_euro = GeometricBrownianMotion(
            S0=S0, r=r, sigma=sigma, T=T, seed=seed
        )
        payoff_euro = EuropeanPutPayoff(strike=K)
        engine_euro = MonteCarloEngine(
            model=model_euro,
            payoff=payoff_euro,
            n_paths=n_paths,
            antithetic=False,
            control_variate=False
        )
        result_euro = engine_euro.price()

        # American put
        model_amer = GeometricBrownianMotion(
            S0=S0, r=r, sigma=sigma, T=T, seed=seed
        )
        result_amer = price_american_lsm(
            model=model_amer,
            strike=K,
            option_type="put",
            n_paths=n_paths,
            n_steps=n_steps,
            basis=basis,
            seed=seed
        )

        # Premium
        premium = result_amer.price - result_euro.price

        print(f"{n_paths:<12,} "
              f"{result_euro.price:<18.6f} {result_euro.stderr:<12.6f} "
              f"{result_amer.price:<18.6f} {result_amer.stderr:<12.6f} "
              f"{premium:<12.6f}")

    print("-" * 100)
    print("\nKey Observations:")
    print("  • American put price >= European put price (early exercise premium)")
    print("  • Premium reflects the value of early exercise right")
    print("  • Standard error decreases with more paths (O(1/√n))")
    print("  • LSM algorithm captures optimal exercise boundary")
    print("=" * 100)


if __name__ == "__main__":
    main()
