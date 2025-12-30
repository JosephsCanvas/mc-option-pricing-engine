#!/usr/bin/env python
"""
Demo: Barrier option pricing analysis.

Demonstrates:
- Up-and-out barrier call option pricing
- Comparison with vanilla call option
- Knock-out rate analysis across different barriers
"""

import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mc_pricer.models.gbm import GeometricBrownianMotion
from mc_pricer.payoffs.path_dependent import UpAndOutCallPayoff
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff
from mc_pricer.pricers.monte_carlo import MonteCarloEngine


def compute_knockout_rate(model, barrier, n_paths, n_steps, seed):
    """Compute the knock-out rate for a given barrier."""
    # Simulate paths
    model_temp = GeometricBrownianMotion(
        S0=model.S0, r=model.r, sigma=model.sigma, T=model.T, seed=seed
    )
    paths = model_temp.simulate_paths(n_paths=n_paths, n_steps=n_steps)

    # Check how many paths breach the barrier
    max_prices = np.max(paths, axis=1)
    knocked_out = max_prices >= barrier

    return np.mean(knocked_out)


def main():
    """Run barrier option pricing demo."""
    print("\n" + "=" * 80)
    print("Up-and-Out Barrier Call Option Analysis")
    print("=" * 80)

    # Market parameters
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 20
    n_paths = 50000
    seed = 42

    print("\nMarket Parameters:")
    print(f"  Spot Price (S0):        {S0:,.2f}")
    print(f"  Strike Price (K):       {K:,.2f}")
    print(f"  Risk-free Rate (r):     {r:.4f}")
    print(f"  Volatility (σ):         {sigma:.4f}")
    print(f"  Time to Maturity (T):   {T:.4f} years")
    print(f"  Time Steps (n_steps):   {n_steps}")
    print(f"  Monte Carlo Paths:      {n_paths:,}")
    print(f"  Random Seed:            {seed}")

    # First, price vanilla call for reference
    print("\n" + "-" * 80)
    print("Vanilla European Call (Reference)")
    print("-" * 80)

    model_vanilla = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
    payoff_vanilla = EuropeanCallPayoff(strike=K)
    engine_vanilla = MonteCarloEngine(
        model=model_vanilla, payoff=payoff_vanilla, n_paths=n_paths, antithetic=False
    )

    result_vanilla = engine_vanilla.price()

    print(f"\nVanilla Call Price:    {result_vanilla.price:.6f}")
    print(f"Standard Error:        {result_vanilla.stderr:.6f}")
    print(f"95% CI:                [{result_vanilla.ci_lower:.6f}, {result_vanilla.ci_upper:.6f}]")

    # Now price barrier options with different barriers
    print("\n" + "-" * 80)
    print("Up-and-Out Barrier Calls")
    print("-" * 80)

    barriers = [110, 120, 130, 150, 200]

    print(f"\n{'Barrier':<10} {'Price':<12} {'% of Vanilla':<15} {'Knockout Rate':<15}")
    print("-" * 80)

    for barrier in barriers:
        # Price barrier option
        model_barrier = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        payoff_barrier = UpAndOutCallPayoff(strike=K, barrier=barrier)
        engine_barrier = MonteCarloEngine(
            model=model_barrier, payoff=payoff_barrier, n_paths=n_paths, antithetic=False
        )

        result_barrier = engine_barrier.price_path_dependent(n_steps=n_steps)

        # Compute knock-out rate
        knockout_rate = compute_knockout_rate(model_vanilla, barrier, n_paths, n_steps, seed)

        # Percentage of vanilla price
        pct_vanilla = (result_barrier.price / result_vanilla.price) * 100

        print(
            f"{barrier:<10.1f} {result_barrier.price:<12.6f} "
            f"{pct_vanilla:<15.2f}% {knockout_rate:<15.2%}"
        )

    # Detailed analysis for one barrier
    print("\n" + "=" * 80)
    print("Detailed Analysis: Barrier = 130")
    print("=" * 80)

    barrier_detail = 130
    model_detail = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
    payoff_detail = UpAndOutCallPayoff(strike=K, barrier=barrier_detail)
    engine_detail = MonteCarloEngine(
        model=model_detail, payoff=payoff_detail, n_paths=n_paths, antithetic=False
    )

    result_detail = engine_detail.price_path_dependent(n_steps=n_steps)
    knockout_rate_detail = compute_knockout_rate(
        model_vanilla, barrier_detail, n_paths, n_steps, seed
    )

    print(f"\nBarrier Level:         {barrier_detail}")
    print(f"Barrier Call Price:    {result_detail.price:.6f}")
    print(f"Vanilla Call Price:    {result_vanilla.price:.6f}")
    print(f"Discount (barrier):    {(1 - result_detail.price / result_vanilla.price) * 100:.2f}%")
    print(f"Knock-out Rate:        {knockout_rate_detail:.2%}")
    print(f"Standard Error:        {result_detail.stderr:.6f}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("\nUp-and-out barrier call options:")
    print("  • Knocked out if price ever reaches or exceeds the barrier")
    print("  • Always cheaper than corresponding vanilla call")
    print("  • Price decreases as barrier decreases (more knock-outs)")
    print("  • Knock-out rate increases as barrier approaches spot")
    print("\nKey observations:")
    ko_rate_110 = compute_knockout_rate(model_vanilla, 110, n_paths, n_steps, seed)
    ko_rate_200 = compute_knockout_rate(model_vanilla, 200, n_paths, n_steps, seed)
    print(f"  • Barrier near spot (110): High knock-out rate (~{ko_rate_110:.1%})")
    print(f"  • Barrier far from spot (200): Low knock-out rate (~{ko_rate_200:.1%})")
    print("  • Barrier very far: Price approaches vanilla call price")
    print()


if __name__ == "__main__":
    main()
