"""
Multi-asset payoffs for basket and spread options.

Implements:
- Basket options: payoff based on arithmetic average of asset prices
- Spread options: payoff based on difference between two asset prices
"""

from typing import Protocol

import numpy as np


class MultiAssetPayoff(Protocol):
    """Protocol for multi-asset payoff functions."""

    strike: float

    def __call__(self, S_T: np.ndarray) -> np.ndarray:
        """
        Compute payoff for terminal prices.

        Parameters
        ----------
        S_T : np.ndarray
            Terminal prices, shape (n_paths, n_assets)

        Returns
        -------
        np.ndarray
            Payoffs, shape (n_paths,)
        """
        ...


class BasketArithmeticCallPayoff:
    """
    Basket call option payoff: max(mean(S_T) - K, 0).

    The payoff is based on the arithmetic average of all asset prices at maturity.
    """

    def __init__(self, strike: float) -> None:
        """
        Initialize basket call payoff.

        Parameters
        ----------
        strike : float
            Strike price (must be non-negative)
        """
        if strike < 0:
            raise ValueError("Strike must be non-negative")
        self.strike = strike

    def __call__(self, S_T: np.ndarray) -> np.ndarray:
        """
        Compute basket call payoff.

        Parameters
        ----------
        S_T : np.ndarray
            Terminal prices, shape (n_paths, n_assets)

        Returns
        -------
        np.ndarray
            Payoffs, shape (n_paths,)
        """
        if S_T.ndim != 2:
            raise ValueError(f"S_T must be 2D, got shape {S_T.shape}")

        # Arithmetic average across assets
        avg_price = np.mean(S_T, axis=1)
        return np.maximum(avg_price - self.strike, 0.0)

    def __repr__(self) -> str:
        return f"BasketArithmeticCallPayoff(strike={self.strike})"


class BasketArithmeticPutPayoff:
    """
    Basket put option payoff: max(K - mean(S_T), 0).

    The payoff is based on the arithmetic average of all asset prices at maturity.
    """

    def __init__(self, strike: float) -> None:
        """
        Initialize basket put payoff.

        Parameters
        ----------
        strike : float
            Strike price (must be non-negative)
        """
        if strike < 0:
            raise ValueError("Strike must be non-negative")
        self.strike = strike

    def __call__(self, S_T: np.ndarray) -> np.ndarray:
        """
        Compute basket put payoff.

        Parameters
        ----------
        S_T : np.ndarray
            Terminal prices, shape (n_paths, n_assets)

        Returns
        -------
        np.ndarray
            Payoffs, shape (n_paths,)
        """
        if S_T.ndim != 2:
            raise ValueError(f"S_T must be 2D, got shape {S_T.shape}")

        # Arithmetic average across assets
        avg_price = np.mean(S_T, axis=1)
        return np.maximum(self.strike - avg_price, 0.0)

    def __repr__(self) -> str:
        return f"BasketArithmeticPutPayoff(strike={self.strike})"


class SpreadCallPayoff:
    """
    Spread call option payoff: max(S1_T - S2_T - K, 0).

    Requires exactly 2 assets. Payoff is based on the difference between
    the first and second asset prices.
    """

    def __init__(self, strike: float) -> None:
        """
        Initialize spread call payoff.

        Parameters
        ----------
        strike : float
            Strike price (must be non-negative)
        """
        if strike < 0:
            raise ValueError("Strike must be non-negative")
        self.strike = strike

    def __call__(self, S_T: np.ndarray) -> np.ndarray:
        """
        Compute spread call payoff.

        Parameters
        ----------
        S_T : np.ndarray
            Terminal prices, shape (n_paths, 2)

        Returns
        -------
        np.ndarray
            Payoffs, shape (n_paths,)
        """
        if S_T.ndim != 2:
            raise ValueError(f"S_T must be 2D, got shape {S_T.shape}")

        if S_T.shape[1] != 2:
            raise ValueError(f"Spread option requires exactly 2 assets, got {S_T.shape[1]}")

        spread = S_T[:, 0] - S_T[:, 1]
        return np.maximum(spread - self.strike, 0.0)

    def __repr__(self) -> str:
        return f"SpreadCallPayoff(strike={self.strike})"


class SpreadPutPayoff:
    """
    Spread put option payoff: max(K - (S1_T - S2_T), 0) = max(S2_T - S1_T + K, 0).

    Requires exactly 2 assets. Payoff is based on the difference between
    the second and first asset prices, plus the strike.
    """

    def __init__(self, strike: float) -> None:
        """
        Initialize spread put payoff.

        Parameters
        ----------
        strike : float
            Strike price (must be non-negative)
        """
        if strike < 0:
            raise ValueError("Strike must be non-negative")
        self.strike = strike

    def __call__(self, S_T: np.ndarray) -> np.ndarray:
        """
        Compute spread put payoff.

        Parameters
        ----------
        S_T : np.ndarray
            Terminal prices, shape (n_paths, 2)

        Returns
        -------
        np.ndarray
            Payoffs, shape (n_paths,)
        """
        if S_T.ndim != 2:
            raise ValueError(f"S_T must be 2D, got shape {S_T.shape}")

        if S_T.shape[1] != 2:
            raise ValueError(f"Spread option requires exactly 2 assets, got {S_T.shape[1]}")

        spread = S_T[:, 0] - S_T[:, 1]
        return np.maximum(self.strike - spread, 0.0)

    def __repr__(self) -> str:
        return f"SpreadPutPayoff(strike={self.strike})"
