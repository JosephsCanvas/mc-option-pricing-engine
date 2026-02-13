"""Market data ingestion and handling.

This subpackage provides tools for fetching and processing real market data
from external sources. Modules are NOT imported automatically to avoid
side effects or network calls at import time.

Available modules (import explicitly):
    - yahoo_options: Options chain data from Yahoo Finance
    - historical: Historical OHLCV data and technical indicators

Note: This module is part of the mc-option-pricing-engine for options
pricing research. For LLM signal extraction projects, see the separate
llm-signals repository.
"""
