#!/usr/bin/env python
"""
Comprehensive demonstration of all features.

Shows basic pricing, variance reduction techniques, and model validation.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mc_pricer.models.gbm import GeometricBrownianMotion
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff, EuropeanPutPayoff
from mc_pricer.pricers.monte_carlo import MonteCarloEngine


def demo_basic_pricing():
    """Demonstrate basic option pricing."""
    print("=" * 80)
    print("DEMO 1: Basic European Call Option Pricing")
    print("=" * 80)

    # Market parameters
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0

    print("\nMarket Parameters:")
    print(f"  S0 = {S0}, K = {K}, r = {r}, σ = {sigma}, T = {T}")

    # Create model and payoff
    model = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=42)
    payoff = EuropeanCallPayoff(strike=K)

    # Price with plain MC
    engine = MonteCarloEngine(model=model, payoff=payoff, n_paths=50000, seed=42)
    result = engine.price()

    print("\nResults (50k paths):")
    print(f"  Price:  {result.price:.6f}")
    print(f"  StdErr: {result.stderr:.6f}")
    print(f"  95% CI: [{result.ci_lower:.6f}, {result.ci_upper:.6f}]")


def demo_variance_reduction():
    """Demonstrate variance reduction techniques."""
    print("\n" + "=" * 80)
    print("DEMO 2: Variance Reduction Techniques")
    print("=" * 80)

    # Fixed parameters
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_paths = 50000
    seed = 42

    methods = [
        ("Plain MC", False, False),
        ("Antithetic Variates", True, False),
        ("Control Variate", False, True),
        ("Antithetic + Control Variate", True, True),
    ]

    print(f"\nComparing methods with {n_paths:,} paths:")
    print("-" * 80)
    print(f"{'Method':<30} {'Price':<12} {'StdErr':<12} {'Reduction':<12}")
    print("-" * 80)

    baseline_stderr = None

    for method_name, antithetic, control_variate in methods:
        model = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=seed)
        payoff = EuropeanCallPayoff(strike=K)

        engine = MonteCarloEngine(
            model=model,
            payoff=payoff,
            n_paths=n_paths,
            antithetic=antithetic,
            control_variate=control_variate,
            seed=seed,
        )

        result = engine.price()

        if baseline_stderr is None:
            baseline_stderr = result.stderr
            reduction_str = "baseline"
        else:
            reduction = (1 - result.stderr / baseline_stderr) * 100
            reduction_str = f"{reduction:.1f}%"

        print(f"{method_name:<30} {result.price:<12.6f} {result.stderr:<12.6f} {reduction_str:<12}")


def demo_black_scholes_validation():
    """Demonstrate validation against Black-Scholes."""
    print("\n" + "=" * 80)
    print("DEMO 3: Black-Scholes Validation")
    print("=" * 80)

    from tests.utils.black_scholes import black_scholes_call

    # Parameters
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0

    # Black-Scholes analytical price
    bs_price = black_scholes_call(S0=S0, K=K, r=r, sigma=sigma, T=T)

    # Monte Carlo price with control variate
    model = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=42)
    payoff = EuropeanCallPayoff(strike=K)
    engine = MonteCarloEngine(
        model=model, payoff=payoff, n_paths=200000, control_variate=True, seed=42
    )
    result = engine.price()

    print("\nATM Call Option:")
    print(f"  Black-Scholes Price: {bs_price:.6f}")
    print(f"  Monte Carlo Price:   {result.price:.6f}")
    print(f"  Difference:          {abs(result.price - bs_price):.6f}")
    print(f"  Relative Error:      {abs(result.price - bs_price) / bs_price * 100:.4f}%")
    print(f"  Within 3σ?           {abs(result.price - bs_price) < 3 * result.stderr}")


def demo_put_call_parity():
    """Demonstrate put-call parity."""
    print("\n" + "=" * 80)
    print("DEMO 4: Put-Call Parity")
    print("=" * 80)

    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_paths = 200000

    # Price call
    model_call = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=42)
    call_payoff = EuropeanCallPayoff(strike=K)
    call_engine = MonteCarloEngine(
        model=model_call, payoff=call_payoff, n_paths=n_paths, control_variate=True, seed=42
    )
    call_price = call_engine.price().price

    # Price put
    model_put = GeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, seed=42)
    put_payoff = EuropeanPutPayoff(strike=K)
    put_engine = MonteCarloEngine(
        model=model_put, payoff=put_payoff, n_paths=n_paths, control_variate=True, seed=42
    )
    put_price = put_engine.price().price

    # Put-call parity: C - P = S0 - K*exp(-rT)
    import numpy as np

    parity_lhs = call_price - put_price
    parity_rhs = S0 - K * np.exp(-r * T)

    print("\nPut-Call Parity Check:")
    print(f"  Call Price:          {call_price:.6f}")
    print(f"  Put Price:           {put_price:.6f}")
    print(f"  C - P:               {parity_lhs:.6f}")
    print(f"  S0 - K*exp(-rT):     {parity_rhs:.6f}")
    print(f"  Difference:          {abs(parity_lhs - parity_rhs):.6f}")
    print(f"  Parity holds?        {abs(parity_lhs - parity_rhs) < 0.01}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("MONTE CARLO OPTION PRICING ENGINE - COMPLETE DEMONSTRATION")
    print("=" * 80)
    print()

    demo_basic_pricing()
    demo_variance_reduction()
    demo_black_scholes_validation()
    demo_put_call_parity()

    print("\n" + "=" * 80)
    print("DEMONSTRATIONS COMPLETE")
    print("=" * 80)
    print("\nFor more examples, see:")
    print("  • scripts/mc_price.py          - CLI interface")
    print("  • scripts/convergence_demo.py  - Convergence analysis")
    print("  • scripts/benchmark_variance_reduction.py - Statistical benchmarks")
    print("  • scripts/plot_convergence.py  - Visualization")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
