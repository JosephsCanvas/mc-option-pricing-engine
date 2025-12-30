#!/usr/bin/env python
"""
Heston volatility smile demonstration.

This script prices call options across a range of strikes using the Heston
stochastic volatility model and converts the prices to implied volatilities,
demonstrating the volatility smile/skew characteristic of stochastic volatility
models.
"""

import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mc_pricer.analytics.implied_vol import implied_vol
from mc_pricer.models.heston import HestonModel
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff
from mc_pricer.pricers.heston_monte_carlo import HestonMonteCarloEngine


def main():
    """Generate and display Heston volatility smile."""
    # Model parameters - calibrated to show typical smile
    S0 = 100.0
    r = 0.05
    T = 1.0
    kappa = 2.0  # Mean reversion speed
    theta = 0.04  # Long-term variance (vol = 0.2)
    xi = 0.3  # Volatility of volatility
    rho = -0.7  # Negative correlation (typical for equity)
    v0 = 0.04  # Initial variance

    # Simulation parameters
    n_paths = 100000
    n_steps = 200
    seed = 42

    # Strike prices
    strikes = [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]

    print("=" * 100)
    print("Heston Volatility Smile Demonstration")
    print("=" * 100)
    print("\nModel Parameters:")
    print(f"  S0={S0}, r={r}, T={T}, option_type=call")
    print(f"  kappa={kappa}, theta={theta}, xi={xi}, rho={rho}, v0={v0}")
    print(f"\nSimulation: n_paths={n_paths:,}, n_steps={n_steps}")
    print(f"\nLong-term volatility: σ∞ = √θ = {np.sqrt(theta):.4f}")

    print("\n" + "-" * 100)
    print(
        f"{'Strike':<10} {'Moneyness':<12} {'Price':<15} {'Stderr':<12} "
        f"{'Implied Vol':<15} {'Time (s)':<10}"
    )
    print("-" * 100)

    results = []
    import time

    for K in strikes:
        start_time = time.time()

        # Create model
        model = HestonModel(
            S0=S0, r=r, T=T, kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0, seed=seed
        )

        # Price option
        payoff = EuropeanCallPayoff(strike=K)
        engine = HestonMonteCarloEngine(
            model=model, payoff=payoff, n_paths=n_paths, n_steps=n_steps, seed=seed
        )
        result = engine.price()

        # Compute implied volatility
        try:
            iv = implied_vol(price=result.price, S0=S0, K=K, r=r, T=T, option_type="call", tol=1e-6)
        except ValueError as e:
            iv = np.nan
            print(f"Warning: Could not compute IV for K={K}: {e}")

        elapsed = time.time() - start_time

        moneyness = K / S0
        print(
            f"{K:<10.1f} {moneyness:<12.4f} {result.price:<15.6f} "
            f"{result.stderr:<12.6f} {iv:<15.6f} {elapsed:<10.2f}"
        )

        results.append(
            {
                "strike": K,
                "moneyness": moneyness,
                "price": result.price,
                "stderr": result.stderr,
                "iv": iv,
            }
        )

    print("-" * 100)

    # Analysis
    valid_ivs = [r["iv"] for r in results if not np.isnan(r["iv"])]
    if valid_ivs:
        print("\nImplied Volatility Statistics:")
        print(f"  Minimum IV:  {min(valid_ivs):.6f}")
        print(f"  Maximum IV:  {max(valid_ivs):.6f}")
        atm_iv = [r["iv"] for r in results if abs(r["moneyness"] - 1.0) < 0.01][0]
        print(f"  ATM IV:      {atm_iv:.6f}")
        print(f"  Smile/Skew:  {max(valid_ivs) - min(valid_ivs):.6f}")

    print("\n" + "=" * 100)
    print("\nKey Observations:")
    print("  • Heston model produces a volatility smile/skew naturally")
    print("  • Negative correlation (ρ < 0) creates higher IV for low strikes (OTM puts)")
    print("  • Volatility of volatility (ξ > 0) creates curvature in the smile")
    print("  • This demonstrates the advantage of Heston over Black-Scholes")
    print("  • Black-Scholes requires different σ for each strike;")
    print("    Heston uses one set of parameters")
    print("=" * 100)


if __name__ == "__main__":
    main()
