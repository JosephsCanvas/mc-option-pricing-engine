#!/usr/bin/env python
"""
Command-line interface for Monte Carlo option pricing.

This is a thin wrapper around mc_pricer.cli for backward compatibility.
Prefer using the installed 'mc-price' command or 'python -m mc_pricer.cli'.

Example usage:
    mc-price --S0 100 --K 100 --r 0.05 --sigma 0.2 --T 1.0 --n_paths 100000
    python scripts/mc_price.py --S0 100 --K 100 --r 0.05 --sigma 0.2 --T 1.0 \
        --n_paths 200000 --antithetic --option_type put
"""

import sys
from pathlib import Path

# Add parent directory to path to allow running without installation
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mc_pricer.cli import main

if __name__ == "__main__":
    sys.exit(main())
