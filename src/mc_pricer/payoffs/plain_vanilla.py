"""
Plain vanilla option payoffs (European call and put).
"""

import numpy as np


class EuropeanCallPayoff:
    """
    European call option payoff: max(S_T - K, 0)
    """

    def __init__(self, strike: float):
        """
        Initialize European call payoff.

        Parameters
        ----------
        strike : float
            Strike price K (must be > 0)
        """
        if strike <= 0:
            raise ValueError("Strike price must be positive")
        self.strike = strike

    def __call__(self, spot_prices: np.ndarray) -> np.ndarray:
        """
        Compute call option payoff.

        Parameters
        ----------
        spot_prices : np.ndarray
            Terminal spot prices S_T

        Returns
        -------
        np.ndarray
            Payoffs max(S_T - K, 0)
        """
        return np.maximum(spot_prices - self.strike, 0.0)

    def __repr__(self) -> str:
        return f"EuropeanCallPayoff(strike={self.strike})"


class EuropeanPutPayoff:
    """
    European put option payoff: max(K - S_T, 0)
    """

    def __init__(self, strike: float):
        """
        Initialize European put payoff.

        Parameters
        ----------
        strike : float
            Strike price K (must be > 0)
        """
        if strike <= 0:
            raise ValueError("Strike price must be positive")
        self.strike = strike

    def __call__(self, spot_prices: np.ndarray) -> np.ndarray:
        """
        Compute put option payoff.

        Parameters
        ----------
        spot_prices : np.ndarray
            Terminal spot prices S_T

        Returns
        -------
        np.ndarray
            Payoffs max(K - S_T, 0)
        """
        return np.maximum(self.strike - spot_prices, 0.0)

    def __repr__(self) -> str:
        return f"EuropeanPutPayoff(strike={self.strike})"
