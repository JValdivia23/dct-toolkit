"""
N-D Cartesian Smoothing.

This module provides separable smoothing for N-D Cartesian data.
It applies a separable N-D transfer function in the DCT domain.
"""

import numpy as np
import scipy.fft

from .core import get_dct_transfer_function


def smooth_cartesian(
    data: np.ndarray, width: float, kernel_type: str = "gaussian"
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
    width : float
        Smoothing width.
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

    spectrum = scipy.fft.dctn(data_array, type=2, norm="ortho")

    for axis, n in enumerate(data_array.shape):
        H = get_dct_transfer_function(n, kernel_type, width)
        transfer_shape = [1] * data_array.ndim
        transfer_shape[axis] = n
        spectrum *= H.reshape(transfer_shape)

    return scipy.fft.idctn(spectrum, type=2, norm="ortho")
