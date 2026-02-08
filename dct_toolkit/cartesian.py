"""
2D Cartesian Smoothing.

This module provides separable smoothing for N-D Cartesian data.
It applies 1D smoothing sequentially along each dimension.
"""

import numpy as np
from .core import get_dct_transfer_function, dct_convolve_1d

def smooth_cartesian(data: np.ndarray, width: float, kernel_type: str = 'boxcar') -> np.ndarray:
    """
    Apply separable DCT smoothing to N-D Cartesian data.
    
    The smoothing is applied sequentially along each axis using the same
    kernel type and width (isotropic smoothing).
    
    Parameters
    ----------
    data : np.ndarray
        Input data array (any dimension).
    width : float
        Smoothing width.
    kernel_type : str, default='boxcar'
        Kernel type ('boxcar', 'boxcar_discrete', 'gaussian').
        
    Returns
    -------
    smoothed : np.ndarray
        Smoothed data with same shape as input.
    """
    result = data.copy()
    ndim = data.ndim
    
    # Apply 1D smoothing along each axis
    for axis in range(ndim):
        n = data.shape[axis]
        H = get_dct_transfer_function(n, kernel_type, width)
        result = dct_convolve_1d(result, H, axis=axis)
        
    return result
