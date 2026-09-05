"""
N-D Cartesian Smoothing.

This module provides separable smoothing for N-D Cartesian data.
It applies a separable N-D transfer function in the DCT domain.
"""

from typing import Sequence, Tuple, Union

import numpy as np
import scipy.fft

from ._widths import WidthLike, normalize_widths
from .core import get_dct_transfer_function


KernelLike = Union[str, Sequence[str], np.ndarray]


def _normalize_kernel_types(kernel_type: KernelLike, ndim: int) -> Tuple[str, ...]:
    """Validate kernel names and expand a single name to every axis."""
    if isinstance(kernel_type, str):
        kernels = (kernel_type,) * ndim
    else:
        values = np.asarray(kernel_type, dtype=object)
        if values.ndim != 1:
            raise ValueError("kernel_type must be a string or 1-D sequence of strings")
        if values.size != ndim:
            raise ValueError(
                f"kernel_type must be a string or length-{ndim} sequence, "
                f"got length {values.size}"
            )
        kernels = tuple(values.tolist())

    for axis, kernel in enumerate(kernels):
        if not isinstance(kernel, str):
            raise ValueError(f"kernel_type entries must be strings, got {kernel!r} at axis {axis}")
        if kernel not in ("boxcar", "boxcar_discrete", "gaussian"):
            raise ValueError(f"Unknown kernel type: {kernel!r} at axis {axis}")
    return kernels


def smooth_cartesian(
    data: np.ndarray,
    width: WidthLike,
    kernel_type: KernelLike = "gaussian",
) -> np.ndarray:
    """
    Apply separable DCT smoothing to N-D Cartesian data.

    The smoothing uses a single N-D forward DCT and inverse DCT. The
    separable transfer function is applied by multiplying the spectrum by
    one 1-D transfer function per axis using broadcasting.

    Parameters
    ----------
    data : np.ndarray
        Input data array (any dimension).
    width : float or sequence of float
        Smoothing width in grid cells along each axis. A scalar applies the
        same width to every axis. A sequence must have length ``data.ndim``
        and follows array axis order. For Gaussian kernels,
        ``sigma = width / sqrt(12)`` along the corresponding axis.
    kernel_type : str or sequence of str, default='gaussian'
        Kernel type ('boxcar', 'boxcar_discrete', 'gaussian'). A string applies
        to every axis. A sequence must have length ``data.ndim`` and selects
        one kernel per axis in array order, independently of ``width``.

    Returns
    -------
    smoothed : np.ndarray
        Smoothed data with same shape as input.

    Raises
    ------
    ValueError
        If a kernel sequence is not one-dimensional, its length differs
        from ``data.ndim``, or a kernel entry is not a supported string.

    Notes
    -----
    Boundaries are reflective on every axis. Equal widths with different
    kernel types can still produce anisotropic smoothing. Input should be
    finite; use ``dct_mean`` or ``dct_smooth`` for data containing NaNs.

    Examples
    --------
    Smooth a volume stored in (z, y, x) order with a Gaussian along z and
    boxcars along y and x:

    >>> volume = np.ones((8, 16, 16))
    >>> smoothed = smooth_cartesian(
    ...     volume, width=(3.0, 5.0, 5.0),
    ...     kernel_type=("gaussian", "boxcar", "boxcar"),
    ... )
    >>> np.allclose(smoothed, volume)
    True
    """
    data_array = np.asarray(data)
    if data_array.ndim == 0:
        return data_array.copy()

    widths = normalize_widths(width, data_array.ndim, name="width")
    kernels = _normalize_kernel_types(kernel_type, data_array.ndim)

    spectrum = scipy.fft.dctn(data_array, type=2, norm="ortho")

    for axis, (n, width_axis, kernel_axis) in enumerate(zip(data_array.shape, widths, kernels)):
        H = get_dct_transfer_function(n, kernel_axis, float(width_axis))
        transfer_shape = [1] * data_array.ndim
        transfer_shape[axis] = n
        spectrum *= H.reshape(transfer_shape)

    return scipy.fft.idctn(spectrum, type=2, norm="ortho")
