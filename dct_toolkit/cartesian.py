"""
N-D Cartesian Smoothing.

This module provides separable smoothing for N-D Cartesian data.
It applies a separable N-D transfer function in the DCT domain.
"""

import numpy as np
import scipy.fft

from .core import get_dct_transfer_function
from ._widths import WidthLike, normalize_widths


def smooth_cartesian(
    data: np.ndarray,
    width: WidthLike,
    kernel_type: str = "gaussian",
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
        Smoothing width. Scalar input applies isotropic smoothing. Sequence
        input must have length ``data.ndim`` and applies per-axis widths.
    kernel_type : str, default='gaussian'
        Kernel type ('boxcar', 'boxcar_discrete', 'gaussian').

    Returns
    -------
    smoothed : np.ndarray
        Smoothed data with same shape as input.
    """
    data_array = np.asarray(data)
    if data_array.ndim == 0:
        return data_array.copy()

    widths = normalize_widths(width, data_array.ndim, name="width")

    spectrum = scipy.fft.dctn(data_array, type=2, norm="ortho")

    for axis, (n, width_axis) in enumerate(zip(data_array.shape, widths)):
        H = get_dct_transfer_function(n, kernel_type, float(width_axis))
        transfer_shape = [1] * data_array.ndim
        transfer_shape[axis] = n
        spectrum *= H.reshape(transfer_shape)

    return scipy.fft.idctn(spectrum, type=2, norm="ortho")
