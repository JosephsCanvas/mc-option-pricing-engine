#!/usr/bin/env python
"""
Command-line interface for Monte Carlo option pricing.

Example usage:
    python scripts/mc_price.py --S0 100 --K 100 --r 0.05 --sigma 0.2 --T 1.0 \\
        --n_paths 100000
    python scripts/mc_price.py --S0 100 --K 100 --r 0.05 --sigma 0.2 --T 1.0 \\
        --n_paths 200000 --antithetic --option_type put
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mc_pricer.models.gbm import GeometricBrownianMotion
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff, EuropeanPutPayoff
from mc_pricer.pricers.monte_carlo import MonteCarloEngine


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Monte Carlo option pricing engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Market parameters
    parser.add_argument("--S0", type=float, required=True, help="Initial spot price")
    parser.add_argument("--K", type=float, required=True, help="Strike price")
    parser.add_argument("--r", type=float, required=True, help="Risk-free rate")
    parser.add_argument("--sigma", type=float, required=True, help="Volatility")
    parser.add_argument("--T", type=float, required=True, help="Time to maturity (years)")

    # Simulation parameters
    parser.add_argument(
        "--n_paths",
        type=int,
        default=100000,
        help="Number of Monte Carlo paths"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--antithetic",
        action="store_true",
        help="Use antithetic variates for variance reduction"
    )
    parser.add_argument(
        "--control_variate",
        action="store_true",
        help="Use control variate variance reduction"
    )

    # Option parameters
    parser.add_argument(
        "--option_type",
        type=str,
        choices=["call", "put"],
        default="call",
        help="Option type: call or put"
    )

    return parser.parse_args()


def main():
    """Main entry point for CLI."""
    args = parse_args()

    # Print input parameters
    print("=" * 70)
    print("Monte Carlo Option Pricing Engine")
    print("=" * 70)
    print("\nInput Parameters:")
    print(f"  Spot Price (S0):        {args.S0:,.2f}")
    print(f"  Strike Price (K):       {args.K:,.2f}")
    print(f"  Risk-free Rate (r):     {args.r:.4f}")
    print(f"  Volatility (σ):         {args.sigma:.4f}")
    print(f"  Time to Maturity (T):   {args.T:.4f} years")
    print(f"  Option Type:            {args.option_type.upper()}")
    print("\nSimulation Parameters:")
    print(f"  Number of Paths:        {args.n_paths:,}")
    print(f"  Antithetic Variates:    {args.antithetic}")
    print(f"  Control Variate:        {args.control_variate}")
    print(f"  Random Seed:            {args.seed if args.seed is not None else 'None (random)'}")

    # Create model
    model = GeometricBrownianMotion(
        S0=args.S0,
        r=args.r,
        sigma=args.sigma,
        T=args.T,
        seed=args.seed
    )

    # Create payoff
    if args.option_type == "call":
        payoff = EuropeanCallPayoff(strike=args.K)
    else:
        payoff = EuropeanPutPayoff(strike=args.K)

    # Create pricing engine
    engine = MonteCarloEngine(
        model=model,
        payoff=payoff,
        n_paths=args.n_paths,
        antithetic=args.antithetic,
        control_variate=args.control_variate
    )

    # Price the option
    print("\n" + "=" * 70)
    print("Pricing...")
    print("=" * 70)
    result = engine.price()

    # Print results
    print("\nResults:")
    print(f"  Option Price:           {result.price:.6f}")
    print(f"  Standard Error:         {result.stderr:.6f}")
    print(f"  95% Confidence Interval: [{result.ci_lower:.6f}, {result.ci_upper:.6f}]")
    print(f"  CI Width:               {result.ci_upper - result.ci_lower:.6f}")
    print(f"  Relative Error (σ/μ):   {result.stderr / result.price * 100:.4f}%")
    if result.control_variate_beta is not None:
        print(f"  Control Variate Beta:   {result.control_variate_beta:.6f}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
