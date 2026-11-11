"""Full sampled Gaussian responses for reflective and periodic filters."""

import numpy as np


def _sampled_gaussian_transfer(n: int, width: float, periodic: bool = False) -> np.ndarray:
    """Evaluate the normalized infinite sampled Gaussian to float64 precision.

    Parameters
    ----------
    n : int
        Axis length.
    width : float
        Effective grid width, with ``sigma = width / sqrt(12)``.
    periodic : bool, default=False
        Use RFFT frequencies for a periodic axis, otherwise DCT-II frequencies.

    Returns
    -------
    ndarray
        Response of length ``n // 2 + 1`` (periodic) or ``n`` (reflective).

    Notes
    -----
    For sigma < 1, sum the spatial samples directly. The omitted two-sided
    mass beyond offset 9 is below 4e-22. For sigma >= 1, Poisson summation
    gives an equivalent sum of frequency-shifted Gaussians. The nearest
    three aliases suffice: omitted terms are below float64 relative precision
    throughout the represented frequency interval [0, pi]. Neither evaluation
    imposes a finite spatial support or allocates memory proportional to width.
    """
    n_freq = n // 2 + 1 if periodic else n
    sigma = width / np.sqrt(12)
    if sigma == 0 or n_freq == 0:
        return np.ones(n_freq)
    omega = (2 if periodic else 1) * np.pi * np.arange(n_freq) / n
    with np.errstate(over="ignore", under="ignore"):
        if sigma < 1:
            offsets = np.arange(1, 10, dtype=float)
            weights = np.exp(-0.5 * (offsets / sigma) ** 2)
            response = 1 + 2 * np.sum(
                weights[:, None] * np.cos(offsets[:, None] * omega), axis=0
            )
        else:
            response = np.zeros(n_freq)
            for alias in (-1, 0, 1):
                response += np.exp(-0.5 * ((omega + alias * 2 * np.pi) * sigma) ** 2)
    response /= response[0]
    response[0] = 1.0
    return response
