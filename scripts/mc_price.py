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
from mc_pricer.pricers.lsm import price_american_lsm
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
    parser.add_argument(
        "--style",
        type=str,
        choices=["european", "american"],
        default="european",
        help="Option style: european or american"
    )

    # American option parameters
    parser.add_argument(
        "--lsm_steps",
        type=int,
        default=50,
        help="Number of time steps for LSM (American options only)"
    )
    parser.add_argument(
        "--lsm_basis",
        type=str,
        choices=["poly2", "poly3"],
        default="poly2",
        help="Basis functions for LSM: poly2 or poly3 (American options only)"
    )

    # Greeks parameters
    parser.add_argument(
        "--greeks",
        type=str,
        choices=["none", "delta", "vega", "all"],
        default="none",
        help="Greeks to compute: none, delta, vega, or all"
    )
    parser.add_argument(
        "--greeks_method",
        type=str,
        choices=["pw", "fd", "both"],
        default="pw",
        help="Method for Greeks computation: pw (pathwise), fd (finite difference), or both"
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
    print(f"  Option Style:           {args.style.upper()}")
    print("\nSimulation Parameters:")
    print(f"  Number of Paths:        {args.n_paths:,}")
    if args.style == "american":
        print(f"  LSM Time Steps:         {args.lsm_steps}")
        print(f"  LSM Basis Functions:    {args.lsm_basis}")
    print(f"  Antithetic Variates:    {args.antithetic}")
    if args.style == "european":
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

    # Price the option
    print("\n" + "=" * 70)
    print("Pricing...")
    print("=" * 70)

    if args.style == "american":
        # American option pricing with LSM
        if args.control_variate or args.greeks != "none":
            print("\nNote: Control variate and Greeks not supported for American options (LSM)")

        result = price_american_lsm(
            model=model,
            strike=args.K,
            option_type=args.option_type,
            n_paths=args.n_paths,
            n_steps=args.lsm_steps,
            basis=args.lsm_basis,
            seed=args.seed
        )

        # Print results
        print("\nResults (LSM American):")
        print(f"  Option Price:           {result.price:.6f}")
        print(f"  Standard Error:         {result.stderr:.6f}")
        print(f"  95% Confidence Interval: [{result.ci_lower:.6f}, {result.ci_upper:.6f}]")
        print(f"  CI Width:               {result.ci_upper - result.ci_lower:.6f}")
        print(f"  Relative Error (σ/μ):   {result.stderr / result.price * 100:.4f}%")

        print("\n" + "=" * 70)
        return

    # European option pricing
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

    # Compute Greeks if requested
    if args.greeks != "none":
        print("\n" + "=" * 70)
        print("Computing Greeks...")
        print("=" * 70)

        if args.greeks_method == "both":
            # Compute both PW and FD
            greeks_pw = engine.compute_greeks(
                option_type=args.option_type,
                method='pw'
            )
            greeks_fd = engine.compute_greeks(
                option_type=args.option_type,
                method='fd'
            )

            print("\nGreeks (Pathwise):")
            if args.greeks in ["delta", "all"] and greeks_pw.delta is not None:
                print(
                    f"  Delta:  {greeks_pw.delta.value:.6f} "
                    f"± {greeks_pw.delta.standard_error:.6f}"
                )
                print(
                    f"    95% CI: [{greeks_pw.delta.ci_lower:.6f}, "
                    f"{greeks_pw.delta.ci_upper:.6f}]"
                )
            if args.greeks in ["vega", "all"] and greeks_pw.vega is not None:
                print(f"  Vega:   {greeks_pw.vega.value:.4f} ± {greeks_pw.vega.standard_error:.4f}")
                print(f"    95% CI: [{greeks_pw.vega.ci_lower:.4f}, {greeks_pw.vega.ci_upper:.4f}]")

            print("\nGreeks (Finite Difference):")
            if args.greeks in ["delta", "all"] and greeks_fd.delta is not None:
                print(
                    f"  Delta:  {greeks_fd.delta.value:.6f} "
                    f"± {greeks_fd.delta.standard_error:.6f}"
                )
                print(
                    f"    95% CI: [{greeks_fd.delta.ci_lower:.6f}, "
                    f"{greeks_fd.delta.ci_upper:.6f}]"
                )
            if args.greeks in ["vega", "all"] and greeks_fd.vega is not None:
                print(f"  Vega:   {greeks_fd.vega.value:.4f} ± {greeks_fd.vega.standard_error:.4f}")
                print(f"    95% CI: [{greeks_fd.vega.ci_lower:.4f}, {greeks_fd.vega.ci_upper:.4f}]")

        else:
            # Compute single method
            greeks = engine.compute_greeks(
                option_type=args.option_type,
                method=args.greeks_method
            )

            method_name = "Pathwise" if args.greeks_method == "pw" else "Finite Difference"
            print(f"\nGreeks ({method_name}):")
            if args.greeks in ["delta", "all"] and greeks.delta is not None:
                print(f"  Delta:  {greeks.delta.value:.6f} ± {greeks.delta.standard_error:.6f}")
                print(f"    95% CI: [{greeks.delta.ci_lower:.6f}, {greeks.delta.ci_upper:.6f}]")
            if args.greeks in ["vega", "all"] and greeks.vega is not None:
                print(f"  Vega:   {greeks.vega.value:.4f} ± {greeks.vega.standard_error:.4f}")
                print(f"    95% CI: [{greeks.vega.ci_lower:.4f}, {greeks.vega.ci_upper:.4f}]")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
