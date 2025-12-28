"""
Path-dependent option payoffs (Asian and Barrier options).
"""

import numpy as np


class AsianArithmeticCallPayoff:
    """
    Asian arithmetic call option payoff: max(mean(S_path) - K, 0)

    The payoff is based on the arithmetic average of all prices in the path.
    """

    def __init__(self, strike: float):
        """
        Initialize Asian arithmetic call payoff.

        Parameters
        ----------
        strike : float
            Strike price K (must be > 0)
        """
        if strike <= 0:
            raise ValueError("Strike price must be positive")
        self.strike = strike

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute Asian arithmetic call payoff.

        Parameters
        ----------
        paths : np.ndarray
            Asset price paths of shape (n_paths, n_steps+1)

        Returns
        -------
        np.ndarray
            Payoffs max(mean(S_path) - K, 0) for each path
        """
        # Compute arithmetic average across time steps for each path
        avg_prices = np.mean(paths, axis=1)
        return np.maximum(avg_prices - self.strike, 0.0)

    def __repr__(self) -> str:
        return f"AsianArithmeticCallPayoff(strike={self.strike})"


class AsianArithmeticPutPayoff:
    """
    Asian arithmetic put option payoff: max(K - mean(S_path), 0)

    The payoff is based on the arithmetic average of all prices in the path.
    """

    def __init__(self, strike: float):
        """
        Initialize Asian arithmetic put payoff.

        Parameters
        ----------
        strike : float
            Strike price K (must be > 0)
        """
        if strike <= 0:
            raise ValueError("Strike price must be positive")
        self.strike = strike

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute Asian arithmetic put payoff.

        Parameters
        ----------
        paths : np.ndarray
            Asset price paths of shape (n_paths, n_steps+1)

        Returns
        -------
        np.ndarray
            Payoffs max(K - mean(S_path), 0) for each path
        """
        # Compute arithmetic average across time steps for each path
        avg_prices = np.mean(paths, axis=1)
        return np.maximum(self.strike - avg_prices, 0.0)

    def __repr__(self) -> str:
        return f"AsianArithmeticPutPayoff(strike={self.strike})"


class UpAndOutCallPayoff:
    """
    Up-and-out barrier call option payoff.

    The option is knocked out if the asset price ever reaches or exceeds
    the barrier during the path. If not knocked out, the payoff is max(S_T - K, 0).
    """

    def __init__(self, strike: float, barrier: float):
        """
        Initialize up-and-out call payoff.

        Parameters
        ----------
        strike : float
            Strike price K (must be > 0)
        barrier : float
            Upper barrier level (must be > strike for typical use)
        """
        if strike <= 0:
            raise ValueError("Strike price must be positive")
        if barrier <= 0:
            raise ValueError("Barrier must be positive")
        self.strike = strike
        self.barrier = barrier

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute up-and-out call payoff.

        Parameters
        ----------
        paths : np.ndarray
            Asset price paths of shape (n_paths, n_steps+1)

        Returns
        -------
        np.ndarray
            Payoffs: max(S_T - K, 0) if max(path) < barrier, else 0
        """
        # Check if barrier was breached (max of path >= barrier)
        max_prices = np.max(paths, axis=1)
        knocked_out = max_prices >= self.barrier

        # Terminal prices
        terminal_prices = paths[:, -1]

        # Vanilla call payoff
        call_payoff = np.maximum(terminal_prices - self.strike, 0.0)

        # Zero out payoff where knocked out
        return np.where(knocked_out, 0.0, call_payoff)

    def __repr__(self) -> str:
        return f"UpAndOutCallPayoff(strike={self.strike}, barrier={self.barrier})"


class DownAndOutPutPayoff:
    """
    Down-and-out barrier put option payoff.

    The option is knocked out if the asset price ever reaches or falls below
    the barrier during the path. If not knocked out, the payoff is max(K - S_T, 0).
    """

    def __init__(self, strike: float, barrier: float):
        """
        Initialize down-and-out put payoff.

        Parameters
        ----------
        strike : float
            Strike price K (must be > 0)
        barrier : float
            Lower barrier level (must be < strike for typical use)
        """
        if strike <= 0:
            raise ValueError("Strike price must be positive")
        if barrier <= 0:
            raise ValueError("Barrier must be positive")
        self.strike = strike
        self.barrier = barrier

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute down-and-out put payoff.

        Parameters
        ----------
        paths : np.ndarray
            Asset price paths of shape (n_paths, n_steps+1)

        Returns
        -------
        np.ndarray
            Payoffs: max(K - S_T, 0) if min(path) > barrier, else 0
        """
        # Check if barrier was breached (min of path <= barrier)
        min_prices = np.min(paths, axis=1)
        knocked_out = min_prices <= self.barrier

        # Terminal prices
        terminal_prices = paths[:, -1]

        # Vanilla put payoff
        put_payoff = np.maximum(self.strike - terminal_prices, 0.0)

        # Zero out payoff where knocked out
        return np.where(knocked_out, 0.0, put_payoff)

    def __repr__(self) -> str:
        return f"DownAndOutPutPayoff(strike={self.strike}, barrier={self.barrier})"
