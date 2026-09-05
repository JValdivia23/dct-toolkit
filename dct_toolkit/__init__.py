"""Public API for DCT-based convolution and statistical operations.

The public top-level surface is intentionally scoped to smoothing and
normalized-convolution statistics for the publication-focused release track.
"""

from typing import Any, Optional

import numpy as np

from .core import dct_convolve_1d, get_dct_transfer_function
from ._widths import WidthLike
from .cartesian import smooth_cartesian
from .polar import smooth_polar
from .stats import dct_count, dct_mean, dct_prefill, dct_std, dct_variance

__version__ = "0.5.0"


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
    width: WidthLike,
    coordinates: str = "cartesian",
    prefill_max_iter: Optional[int] = 3,
    **kwargs: Any,
) -> np.ndarray:
    """
    Apply DCT-based smoothing.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    width : float or sequence of float
        Smoothing width. A scalar applies the same width to every axis.
        A sequence allows per-axis widths. Cartesian widths are in grid cells
        and follow array axis order; Gaussian ``sigma = width / sqrt(12)``.
    coordinates : str, default='cartesian'
        'cartesian' or 'polar'.
    prefill_max_iter : int or None, default=3
        Number of normalized-convolution prefill iterations used when input
        contains NaNs. If None, prefill iterates until convergence or a
        safety cap of 20 iterations.
    **kwargs
        Additional arguments (kernel_type, az_res_deg, etc.). For Cartesian
        data, ``kernel_type`` accepts a string or a sequence of supported
        kernel names with length ``data.ndim``, in array axis order.
        The same kernels are used for prefill and final smoothing.
        Polar data requires a single kernel string. The default is 'gaussian'.

    Returns
    -------
    smoothed : np.ndarray
        Smoothed data.
    """
    data_array = np.asarray(data)
    nan_mask = np.isnan(data_array)

    # Support legacy alias in wrapper calls.
    if "max_iter" in kwargs:
        prefill_max_iter = kwargs.pop("max_iter")

    if not np.any(nan_mask):
        if coordinates == "cartesian":
            return smooth_cartesian(data_array, width, **kwargs)
        if coordinates == "polar":
            return smooth_polar(data_array, width_pixels=width, **kwargs)
        raise ValueError(f"Unknown coordinates: {coordinates}")

    # No support exists to estimate an all-NaN field; preserve mask semantics.
    if np.all(nan_mask):
        return np.full(data_array.shape, np.nan, dtype=np.float64)

    filled = dct_prefill(
        data_array,
        width=width,
        coordinates=coordinates,
        max_iter=prefill_max_iter,
        **kwargs,
    )

    if coordinates == "cartesian":
        smoothed = smooth_cartesian(filled, width, **kwargs)
    elif coordinates == "polar":
        smoothed = smooth_polar(filled, width_pixels=width, **kwargs)
    else:
        raise ValueError(f"Unknown coordinates: {coordinates}")

    smoothed[nan_mask] = np.nan
    return smoothed
