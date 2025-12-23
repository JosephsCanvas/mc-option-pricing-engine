#!/usr/bin/env python
"""
Implied volatility smile demonstration.

Generates a synthetic volatility smile and recovers implied volatility
from market prices. Optionally plots the smile if matplotlib is available.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mc_pricer.analytics.black_scholes import bs_price
from mc_pricer.analytics.implied_vol import implied_vol


def main():
    """Run IV smile demonstration."""
    # Parameters
    S0 = 100.0
    r = 0.05
    T = 1.0
    option_type = "call"

    # Volatility smile parameters
    base_vol = 0.20  # ATM volatility
    skew = -0.15  # Volatility skew (negative for call smile)
    curvature = 0.25  # Smile curvature

    # Strike range
    strikes = [70, 80, 90, 100, 110, 120, 130]

    print("=" * 100)
    print("Implied Volatility Smile Demonstration")
    print("=" * 100)
    print(f"\nParameters: S0={S0}, r={r}, T={T}, option_type={option_type}")
    print(f"Volatility model: σ(K) = {base_vol} + {skew}*(K/S0 - 1) + {curvature}*(K/S0 - 1)²")
    print("\n" + "-" * 100)
    print(f"{'Strike':<10} {'Moneyness':<12} {'True Vol':<12} "
          f"{'Market Price':<15} {'Implied Vol':<15} {'Abs Error':<12}")
    print("-" * 100)

    results = []
    for K in strikes:
        # Compute moneyness
        moneyness = K / S0

        # True volatility with smile
        deviation = moneyness - 1.0
        true_sigma = base_vol + skew * deviation + curvature * deviation**2

        # Generate "market price" using true volatility
        market_price = bs_price(S0, K, r, T, true_sigma, option_type)

        # Recover implied volatility
        try:
            iv = implied_vol(
                price=market_price,
                S0=S0,
                K=K,
                r=r,
                T=T,
                option_type=option_type,
                tol=1e-8
            )
            error = abs(iv - true_sigma)
            results.append((K, moneyness, true_sigma, market_price, iv, error))

            print(f"{K:<10.1f} {moneyness:<12.4f} {true_sigma:<12.6f} "
                  f"{market_price:<15.6f} {iv:<15.6f} {error:<12.2e}")

        except ValueError as e:
            print(f"{K:<10.1f} {moneyness:<12.4f} {true_sigma:<12.6f} "
                  f"{market_price:<15.6f} {'ERROR':<15} {str(e)}")

    print("-" * 100)

    # Summary statistics
    if results:
        errors = [r[5] for r in results]
        max_error = max(errors)
        avg_error = sum(errors) / len(errors)
        print("\nRecovery Statistics:")
        print(f"  Maximum error:  {max_error:.2e}")
        print(f"  Average error:  {avg_error:.2e}")
        print(f"  All errors < 1e-6: {'✓' if all(e < 1e-6 for e in errors) else '✗'}")

    # Optional plotting
    try:
        import matplotlib.pyplot as plt

        print("\n" + "=" * 100)
        print("Generating plot...")
        print("=" * 100)

        strikes_list = [r[0] for r in results]
        true_vols = [r[2] for r in results]
        implied_vols = [r[4] for r in results]

        plt.figure(figsize=(10, 6))
        plt.plot(strikes_list, true_vols, 'b-o', label='True Volatility', linewidth=2)
        plt.plot(strikes_list, implied_vols, 'r--s', label='Implied Volatility', linewidth=2)
        plt.axvline(S0, color='gray', linestyle=':', alpha=0.7, label=f'ATM (S0={S0})')
        plt.xlabel('Strike Price (K)', fontsize=12)
        plt.ylabel('Volatility', fontsize=12)
        plt.title('Volatility Smile: True vs Recovered Implied Volatility', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        plots_dir = Path(__file__).parent.parent / "plots"
        plots_dir.mkdir(exist_ok=True)
        output_path = plots_dir / "iv_smile.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
        print("=" * 100)

    except ImportError:
        print("\n" + "=" * 100)
        print("Note: matplotlib not available - skipping plot generation")
        print("Install with: pip install matplotlib")
        print("=" * 100)

    print("\nKey Observations:")
    print("  • Implied volatility solver successfully recovers true volatility")
    print("  • Smile shape (volatility varies with strike) is preserved")
    print("  • Bisection method is robust and accurate")
    print("  • Call options show typical smile with lower vol at higher strikes")
    print("=" * 100)


if __name__ == "__main__":
    main()
