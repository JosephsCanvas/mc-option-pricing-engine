"""
Quasi-Monte Carlo (QMC) random number generators using Sobol sequences.

This module provides a pure NumPy implementation of Sobol sequences with
optional scrambling for variance reduction in Monte Carlo simulations.
"""

import numpy as np

# Sobol direction numbers (primitive polynomials and direction numbers)
# These are for the first 21 dimensions (sufficient for most applications)
# Source: Joe & Kuo (2008), ACM TOMS
SOBOL_DIRECTION_NUMBERS = {
    1: (1, [1]),
    2: (2, [1, 3]),
    3: (3, [1, 3, 1]),
    4: (3, [1, 1, 1]),
    5: (4, [1, 1, 3, 3]),
    6: (4, [1, 3, 5, 13]),
    7: (5, [1, 1, 5, 5, 17]),
    8: (5, [1, 1, 5, 5, 5]),
    9: (5, [1, 1, 7, 11, 19]),
    10: (5, [1, 1, 5, 1, 1]),
    11: (5, [1, 1, 1, 3, 11]),
    12: (5, [1, 3, 5, 5, 31]),
    13: (5, [1, 3, 3, 9, 7]),
    14: (6, [1, 1, 5, 13, 29, 3]),
    15: (6, [1, 3, 1, 13, 11, 5]),
    16: (6, [1, 1, 3, 5, 25, 25]),
    17: (6, [1, 1, 7, 11, 19, 23]),
    18: (6, [1, 3, 7, 13, 3, 13]),
    19: (6, [1, 3, 7, 5, 7, 33]),
    20: (7, [1, 1, 1, 9, 23, 37, 67]),
    21: (7, [1, 3, 7, 15, 29, 15, 21]),
}


def inverse_normal_cdf(u: np.ndarray) -> np.ndarray:
    """
    Convert uniform (0,1) samples to standard normal N(0,1) using inverse CDF.

    Uses the Acklam approximation for high accuracy without requiring scipy.

    Parameters
    ----------
    u : np.ndarray
        Uniform random variables in (0, 1)

    Returns
    -------
    np.ndarray
        Standard normal random variables

    Notes
    -----
    This is an implementation of Peter John Acklam's inverse normal CDF
    algorithm, which has relative error less than 1.15e-9 for all inputs.

    Reference:
    https://web.archive.org/web/20150912080806/http://home.online.no/~pjacklam/notes/invnorm/
    """
    # Validate input
    if np.any(u < 0) or np.any(u > 1):
        raise ValueError("probabilities must be in [0, 1]")

    # Clip to avoid numerical issues at boundaries
    u = np.clip(u, 1e-10, 1 - 1e-10)

    # Coefficients for rational approximation
    a = np.array(
        [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
    )

    b = np.array(
        [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
    )

    c = np.array(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
    )

    d = np.array(
        [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    )

    # Define break-points
    u_low = 0.02425
    u_high = 1 - u_low

    # Allocate output
    z = np.zeros_like(u)

    # Region 1: u < u_low (lower tail)
    mask_low = u < u_low
    if np.any(mask_low):
        q = np.sqrt(-2.0 * np.log(u[mask_low]))
        z[mask_low] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )

    # Region 2: u_low <= u <= u_high (central region)
    mask_central = (u >= u_low) & (u <= u_high)
    if np.any(mask_central):
        q = u[mask_central] - 0.5
        r = q * q
        z[mask_central] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)

    # Region 3: u > u_high (upper tail)
    mask_high = u > u_high
    if np.any(mask_high):
        q = np.sqrt(-2.0 * np.log(1.0 - u[mask_high]))
        z[mask_high] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )

    return z


class SobolGenerator:
    """
    Sobol quasi-random sequence generator with optional scrambling.

    Generates low-discrepancy sequences for improved Monte Carlo convergence.
    Supports up to 21 dimensions using direction numbers from Joe & Kuo (2008).

    Parameters
    ----------
    dimension : int
        Dimension of the Sobol sequence (1-21)
    seed : int, optional
        Seed for digital shift scrambling (default: None, no scrambling)
    scramble : bool, optional
        Whether to apply digital shift scrambling (default: False)

    Notes
    -----
    The implementation uses the Gray code construction for efficiency and
    supports optional digital shift scrambling for additional variance reduction.

    For Owen scrambling, a more sophisticated implementation would be needed,
    so we use digital shift scrambling as a simpler alternative that still
    provides scrambling benefits.
    """

    def __init__(self, dimension: int, seed: int | None = None, scramble: bool = False):
        """Initialize Sobol generator."""
        if dimension < 1 or dimension > 21:
            raise ValueError("dimension must be between 1 and 21")

        self.dimension = dimension
        self.seed = seed
        self.scramble = scramble

        # Initialize direction numbers for each dimension
        self._direction_numbers = self._initialize_direction_numbers()

        # Initialize scrambling shift if requested
        if scramble and seed is not None:
            rng = np.random.default_rng(seed)
            # Digital shift: random integer shift for each dimension
            self._shift = rng.integers(0, 2**31, size=dimension, dtype=np.uint32)
        else:
            self._shift = np.zeros(dimension, dtype=np.uint32)

        # Track current index
        self._index = 0

    def _initialize_direction_numbers(self) -> list:
        """
        Initialize direction numbers for Sobol sequence generation.

        Returns
        -------
        list
            List of direction number arrays for each dimension
        """
        direction_numbers = []

        for dim in range(1, self.dimension + 1):
            if dim == 1:
                # First dimension uses simple powers of 2
                # v_i = 2^{32-i} for i=1,2,...,32
                v = np.array([1 << (32 - i) for i in range(1, 33)], dtype=np.uint32)
            else:
                # Get primitive polynomial and initialization values
                poly, m_init = SOBOL_DIRECTION_NUMBERS[dim]
                degree = len(m_init)

                # Initialize direction numbers
                v = np.zeros(32, dtype=np.uint32)

                # Set initial values (multiply m_i by 2^{32-i})
                for i in range(degree):
                    v[i] = m_init[i] << (32 - i - 1)

                # Generate remaining direction numbers using recurrence
                # v_i = 2^degree * v_{i-degree} XOR (sum of a_k * v_{i-k} for k=1..degree)
                for i in range(degree, 32):
                    v[i] = v[i - degree] << degree
                    for k in range(1, degree + 1):
                        if (poly >> (k - 1)) & 1:
                            v[i] ^= v[i - k]

            direction_numbers.append(v)

        return direction_numbers

    def generate(self, n_points: int, skip: int = 0) -> np.ndarray:
        """
        Generate Sobol sequence points.

        Parameters
        ----------
        n_points : int
            Number of points to generate
        skip : int, optional
            Number of initial points to skip (default: 0)

        Returns
        -------
        np.ndarray
            Array of shape (n_points, dimension) with values in (0, 1)

        Notes
        -----
        Uses Gray code construction for efficient generation.
        """
        if n_points <= 0:
            raise ValueError("n_points must be positive")

        points = np.zeros((n_points, self.dimension), dtype=np.float64)

        # Initialize with skip offset
        x = np.zeros(self.dimension, dtype=np.uint32)
        if skip > 0:
            for i in range(skip):
                # Update using Gray code
                c = self._gray_code_position(i + 1)
                for d in range(self.dimension):
                    x[d] ^= self._direction_numbers[d][c]

        # Generate points
        for i in range(n_points):
            # Find position of rightmost zero bit in Gray code
            c = self._gray_code_position(skip + i + 1)

            # Update x by XOR with direction number
            for d in range(self.dimension):
                x[d] ^= self._direction_numbers[d][c]

            # Apply digital shift scrambling if enabled
            if self.scramble:
                x_scrambled = x ^ self._shift
                points[i] = x_scrambled / (2**32)
            else:
                points[i] = x / (2**32)

        # Ensure values are strictly in (0, 1)
        points = np.clip(points, 1e-10, 1 - 1e-10)

        return points

    def _gray_code_position(self, n: int) -> int:
        """
        Find the position of the rightmost zero bit in the binary representation of n-1.

        This gives the index of which direction number to use for the Gray code
        construction of the Sobol sequence.

        Parameters
        ----------
        n : int
            Index in the sequence (1-indexed)

        Returns
        -------
        int
            Position of rightmost zero bit in n-1 (0-indexed)
        """
        # Find position of rightmost zero bit in (n-1)
        # Equivalent to counting trailing 1-bits in (n-1)
        x = n - 1
        c = 0
        while (x & 1) == 1:
            x >>= 1
            c += 1
        return c

    def generate_normal(self, n_points: int, skip: int = 0) -> np.ndarray:
        """
        Generate Sobol sequence points transformed to standard normal.

        Parameters
        ----------
        n_points : int
            Number of points to generate
        skip : int, optional
            Number of initial points to skip (default: 0)

        Returns
        -------
        np.ndarray
            Array of shape (n_points, dimension) with N(0,1) samples

        Notes
        -----
        Uses inverse normal CDF transformation to convert uniform Sobol
        points to standard normal variates.
        """
        uniform_points = self.generate(n_points, skip)
        return inverse_normal_cdf(uniform_points)

    def reset(self):
        """Reset the generator to start from the beginning of the sequence."""
        self._index = 0
