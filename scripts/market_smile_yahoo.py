#!/usr/bin/env python3
"""Fetch and analyze implied volatility smile from Yahoo Finance.

This script demonstrates market data ingestion and IV computation using
the mc-pricer analytics library with real Yahoo Finance options chains.

Requires: pip install -e ".[marketdata]"

Example usage:
    python scripts/market_smile_yahoo.py --ticker SPY --expiry 2026-01-16 --option_type call
    python scripts/market_smile_yahoo.py --ticker AAPL --expiry 2026-03-20 --option_type put --calibrate_heston
"""

import argparse
import sys

import numpy as np

from mc_pricer.analytics.market_iv import expiry_to_years, quote_to_iv
from mc_pricer.data.yahoo_options import compute_mids, fetch_options_chain, filter_quotes


def print_table(quotes_with_iv: list[tuple], max_rows: int = 25) -> None:
    """Print formatted table of option quotes with implied volatilities.

    Parameters
    ----------
    quotes_with_iv : list[tuple]
        List of (quote, iv) tuples sorted by strike.
    max_rows : int
        Maximum number of rows to print.
    """
    print("\n" + "=" * 100)
    print("OPTION QUOTES WITH IMPLIED VOLATILITIES")
    print("=" * 100)
    print(
        f"{'Strike':>10} {'Bid':>10} {'Ask':>10} {'Mid':>10} "
        f"{'IV (Ours)':>12} {'IV (Yahoo)':>12} {'Volume':>10} {'OI':>10}"
    )
    print("-" * 100)

    for quote, iv in quotes_with_iv[:max_rows]:
        mid = quote.mid()
        iv_ours = f"{iv * 100:.2f}%" if iv is not None else "N/A"
        iv_yahoo = f"{quote.iv_yahoo * 100:.2f}%" if quote.iv_yahoo is not None else "N/A"
        volume = str(quote.volume) if quote.volume is not None else "N/A"
        oi = str(quote.open_interest) if quote.open_interest is not None else "N/A"

        print(
            f"{quote.strike:>10.2f} {quote.bid:>10.2f} {quote.ask:>10.2f} {mid:>10.2f} "
            f"{iv_ours:>12} {iv_yahoo:>12} {volume:>10} {oi:>10}"
        )

    if len(quotes_with_iv) > max_rows:
        print(f"... ({len(quotes_with_iv) - max_rows} more rows omitted)")

    print("=" * 100)


def print_diagnostics(
    quotes_raw: list,
    quotes_filtered: list,
    quotes_with_iv: list[tuple],
    spot: float,
) -> None:
    """Print summary diagnostics.

    Parameters
    ----------
    quotes_raw : list
        Raw quotes fetched.
    quotes_filtered : list
        Filtered quotes.
    quotes_with_iv : list[tuple]
        Quotes with computed IVs.
    spot : float
        Underlying spot price.
    """
    ivs = [iv for _, iv in quotes_with_iv if iv is not None]

    print("\n" + "=" * 100)
    print("DIAGNOSTICS")
    print("=" * 100)
    print(f"Underlying spot:      ${spot:.2f}")
    print(f"Quotes fetched:       {len(quotes_raw)}")
    print(f"Quotes after filter:  {len(quotes_filtered)}")
    print(f"Quotes with IV:       {len(ivs)}")

    if ivs:
        ivs_pct = [iv * 100 for iv in ivs]
        print(f"IV range:             {min(ivs_pct):.2f}% - {max(ivs_pct):.2f}%")
        print(f"IV mean:              {np.mean(ivs_pct):.2f}%")
        print(f"IV median:            {np.median(ivs_pct):.2f}%")
        print(f"IV std dev:           {np.std(ivs_pct):.2f}%")

        # Find ATM IV
        strikes = [q.strike for q, iv in quotes_with_iv if iv is not None]
        atm_idx = np.argmin([abs(K - spot) for K in strikes])
        atm_iv = ivs[atm_idx] * 100
        print(f"ATM IV (nearest):     {atm_iv:.2f}% at strike ${strikes[atm_idx]:.2f}")

        # Smile width (OTM put IV - OTM call IV as rough measure)
        low_strikes = [K for K in strikes if K < spot * 0.95]
        high_strikes = [K for K in strikes if K > spot * 1.05]

        if low_strikes and high_strikes:
            low_idx = strikes.index(min(low_strikes, key=lambda k: abs(k - spot * 0.90)))
            high_idx = strikes.index(min(high_strikes, key=lambda k: abs(k - spot * 1.10)))
            skew = (ivs[low_idx] - ivs[high_idx]) * 100
            print(f"IV skew (10% OTM):    {skew:+.2f}% (put IV - call IV)")

    print("=" * 100)


def calibrate_heston_to_smile(
    quotes_with_iv: list[tuple],
    spot: float,
    r: float,
    T: float,  # noqa: N803
    n_paths: int,
    n_steps: int,
    rng_type: str,
    scramble: bool,
    seed: int,
) -> None:
    """Calibrate Heston model to the implied volatility smile.

    Parameters
    ----------
    quotes_with_iv : list[tuple]
        List of (quote, iv) tuples.
    spot : float
        Underlying spot price.
    r : float
        Risk-free rate.
    T : float
        Time to maturity.
    n_paths : int
        Number of MC paths for calibration.
    n_steps : int
        Number of time steps.
    rng_type : str
        'pseudo' or 'sobol'.
    scramble : bool
        Whether to scramble Sobol.
    seed : int
        Random seed.
    """
    print("\n" + "=" * 100)
    print("HESTON CALIBRATION")
    print("=" * 100)

    from mc_pricer.calibration.heston_calibration import (
        CalibrationConfig,
        HestonCalibrator,
        MarketQuote,
    )

    # Build market quotes for calibration
    market_quotes = []
    for quote, iv in quotes_with_iv:
        if iv is not None:
            market_quotes.append(
                MarketQuote(
                    strike=quote.strike,
                    maturity=T,
                    option_type=quote.option_type,
                    implied_vol=iv,
                )
            )

    if len(market_quotes) < 3:
        print("Not enough quotes with valid IV for calibration (need at least 3)")
        print("=" * 100)
        return

    print(f"Calibrating to {len(market_quotes)} market quotes...")
    print(f"Using {n_paths} paths, {n_steps} steps, {rng_type} RNG")

    # Configure calibration (use fewer iterations for demo)
    config = CalibrationConfig(
        n_paths=n_paths,
        n_steps=n_steps,
        rng_type=rng_type,
        scramble=scramble,
        seeds=[seed],
        max_iter=100,  # Reduced for demo
        tol=1e-6,
        bounds={
            "kappa": (0.1, 10.0),
            "theta": (0.01, 1.0),
            "xi": (0.01, 2.0),
            "rho": (-0.99, 0.99),
            "v0": (0.01, 1.0),
        },
        heston_scheme="full_truncation_euler",
    )

    # Create calibrator and run
    calibrator = HestonCalibrator(
        S0=spot,
        r=r,
        quotes=market_quotes,
        config=config,
    )

    result = calibrator.calibrate()

    # Print results
    print("\nCalibrated Parameters:")
    print(f"  kappa (mean reversion):   {result.best_params['kappa']:.4f}")
    print(f"  theta (long-run var):     {result.best_params['theta']:.4f}")
    print(f"  xi (vol of vol):          {result.best_params['xi']:.4f}")
    print(f"  rho (correlation):        {result.best_params['rho']:.4f}")
    print(f"  v0 (initial variance):    {result.best_params['v0']:.4f}")
    print("\nCalibration quality:")
    print(f"  RMSE:                     {result.objective_value:.6f}")
    print(f"  Function evals:           {result.n_evals}")
    print(f"  Time elapsed:             {result.runtime_sec:.2f}s")
    if result.cache_hits > 0 or result.cache_misses > 0:
        print(f"  Cache hits/misses:        {result.cache_hits}/{result.cache_misses}")

    print("=" * 100)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch and analyze option smile from Yahoo Finance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument(
        "--expiry",
        type=str,
        required=True,
        help="Expiry date in YYYY-MM-DD format (e.g., 2026-01-16)",
    )
    parser.add_argument(
        "--option_type",
        type=str,
        choices=["call", "put"],
        default="call",
        help="Option type (default: call)",
    )
    parser.add_argument("--r", type=float, default=0.05, help="Risk-free rate (default: 0.05)")
    parser.add_argument(
        "--max_rows", type=int, default=25, help="Max rows to display (default: 25)"
    )
    parser.add_argument(
        "--min_bid", type=float, default=0.05, help="Min bid price filter (default: 0.05)"
    )
    parser.add_argument(
        "--max_rel_spread",
        type=float,
        default=0.30,
        help="Max relative spread filter (default: 0.30)",
    )
    parser.add_argument("--min_volume", type=int, default=None, help="Min volume filter (optional)")
    parser.add_argument(
        "--min_oi", type=int, default=None, help="Min open interest filter (optional)"
    )
    parser.add_argument(
        "--calibrate_heston",
        action="store_true",
        help="Calibrate Heston model to IV smile",
    )
    parser.add_argument(
        "--calib_paths", type=int, default=20000, help="MC paths for calibration (default: 20000)"
    )
    parser.add_argument(
        "--calib_steps", type=int, default=100, help="MC steps for calibration (default: 100)"
    )
    parser.add_argument(
        "--rng",
        type=str,
        choices=["pseudo", "sobol"],
        default="pseudo",
        help="RNG type for calibration (default: pseudo)",
    )
    parser.add_argument(
        "--scramble", action="store_true", help="Use scrambling for Sobol (default: False)"
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed (default: 123)")

    args = parser.parse_args()

    # Fetch options chain
    print(f"Fetching options chain for {args.ticker} expiry {args.expiry}...")
    try:
        quotes_raw = fetch_options_chain(args.ticker, args.expiry)
    except (ImportError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not quotes_raw:
        print("No quotes found", file=sys.stderr)
        return 1

    spot = quotes_raw[0].underlying_spot
    print(f"Fetched {len(quotes_raw)} quotes (spot: ${spot:.2f})")

    # Filter by option type
    quotes_type = [q for q in quotes_raw if q.option_type == args.option_type]
    print(f"Filtered to {len(quotes_type)} {args.option_type}s")

    # Apply liquidity filters
    quotes_filtered = filter_quotes(
        quotes_type,
        min_bid=args.min_bid,
        max_rel_spread=args.max_rel_spread,
        min_volume=args.min_volume,
        min_oi=args.min_oi,
    )
    print(f"After liquidity filters: {len(quotes_filtered)} quotes")

    # Ensure we have mid prices
    quotes_filtered = compute_mids(quotes_filtered)

    if not quotes_filtered:
        print("No quotes remain after filtering", file=sys.stderr)
        return 1

    # Compute time to maturity
    T = expiry_to_years(args.expiry)
    print(f"Time to maturity: {T:.4f} years")

    # Compute implied volatilities
    quotes_with_iv = []
    for quote in quotes_filtered:
        iv = quote_to_iv(quote, r=args.r, T=T)
        quotes_with_iv.append((quote, iv))

    # Sort by strike
    quotes_with_iv.sort(key=lambda x: x[0].strike)

    # Print table
    print_table(quotes_with_iv, max_rows=args.max_rows)

    # Print diagnostics
    print_diagnostics(quotes_raw, quotes_filtered, quotes_with_iv, spot)

    # Calibrate Heston if requested
    if args.calibrate_heston:
        calibrate_heston_to_smile(
            quotes_with_iv=quotes_with_iv,
            spot=spot,
            r=args.r,
            T=T,
            n_paths=args.calib_paths,
            n_steps=args.calib_steps,
            rng_type=args.rng,
            scramble=args.scramble,
            seed=args.seed,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
