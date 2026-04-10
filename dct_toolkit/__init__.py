"""Public API for DCT-based convolution and statistical operations.

The public top-level surface is intentionally scoped to smoothing and
normalized-convolution statistics for the publication-focused release track.
"""

from typing import Any

import numpy as np

from .core import dct_convolve_1d, get_dct_transfer_function
from .cartesian import smooth_cartesian
from .polar import smooth_polar
from .stats import dct_count, dct_mean, dct_prefill, dct_std, dct_variance

__version__ = "0.4.0"


__all__ = [
    "__version__",
    "get_dct_transfer_function",
    "dct_convolve_1d",
    "smooth_cartesian",
    "smooth_polar",
    "dct_smooth",
    "dct_count",
    "dct_mean",
    "dct_prefill",
    "dct_variance",
    "dct_std",
]


def dct_smooth(
    data: np.ndarray,
    width: float,
    coordinates: str = "cartesian",
    **kwargs: Any,
) -> np.ndarray:
    """
    Apply DCT-based smoothing.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    width : float
        Smoothing width.
    coordinates : str, default='cartesian'
        'cartesian' or 'polar'.
    **kwargs
        Additional arguments (kernel_type, az_res_deg, etc.)

    Returns
    -------
    smoothed : np.ndarray
        Smoothed data.
    """
    if coordinates == "cartesian":
        return smooth_cartesian(data, width, **kwargs)
    if coordinates == "polar":
        # Polar func takes width_pixels
        return smooth_polar(data, width_pixels=width, **kwargs)
    raise ValueError(f"Unknown coordinates: {coordinates}")
