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
from .stats import (
    _resolve_smoothing_geometry,
    _smooth_completed_field,
    _smoothing_kwargs,
    dct_count,
    dct_mean,
    dct_prefill,
    dct_std,
    dct_variance,
)

__version__ = "1.0.0"


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
        Explicit N-D polar widths also follow array order, with units set by
        ``width_units``. Legacy polar calls use (azimuth, range) widths.
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
        Explicit polar geometry accepts ``axis_types``, ``axis_coords``,
        ``boundary``, ``width_units='physical'``, ``angle_units='degrees'``,
        and ``filter_axes``. Per-axis widths, kernels, boundaries, and units
        follow array axis order. Unfiltered axes are isolated domains, including
        geometry axes; all-missing domains remain NaN without affecting others.
        Legacy polar data accepts a string or an (azimuth, range) kernel pair,
        and ``az_res_deg`` must be a finite, positive real scalar.
        Geometry and kernels are validated even for all-NaN inputs. The same
        operator is used for prefill and final smoothing. Default kernel is
        'gaussian'. ``min_effective_density`` and private mean controls apply
        to prefill only and are not passed to the low-level smoother.

    Returns
    -------
    smoothed : np.ndarray
        Smoothed data with the input shape, including a zero-dimensional array
        for scalar input.
    """
    data_array = np.asarray(data)
    nan_mask = np.isnan(data_array)

    # Support legacy alias in wrapper calls.
    if "max_iter" in kwargs:
        prefill_max_iter = kwargs.pop("max_iter")

    smooth_kwargs = _smoothing_kwargs(kwargs)
    smooth_kwargs.pop("min_effective_density", None)
    geometry = _resolve_smoothing_geometry(data_array.shape, width, coordinates, **smooth_kwargs)

    if not np.any(nan_mask):
        return _smooth_completed_field(
            data_array, width, coordinates, geometry=geometry, **smooth_kwargs
        )

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

    smoothed = _smooth_completed_field(
        filled, width, coordinates, geometry=geometry, **smooth_kwargs
    )

    smoothed[nan_mask] = np.nan
    return smoothed
