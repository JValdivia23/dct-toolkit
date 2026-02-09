"""
DCT-Toolkit: Discrete Cosine Transform Statistical Operations.

This package provides robust statistical primitives (smoothing, mean, variance)
based on the Discrete Cosine Transform (DCT). It supports:
- 1D Data
- 2D Cartesian Data
- 2D Polarimetric Data (with adaptive kernels)
- Iterative gap filling with linear interpolation initialization

Key Functions
-------------
- dct_smooth: General purpose smoothing
- dct_count: Effective sample size calculation
- dct_mean: Robust local mean (Normalized Convolution)
- dct_variance: Robust local variance
- dct_std: Robust local standard deviation
- iterative_gap_fill: Fill gaps using iterative DCT smoothing

Modules
-------
- core: Transfer functions and 1D primitives
- cartesian: Separable N-D smoothing
- polar: Polar coordinate smoothing
- stats: Statistical operations
- gap_filling: Iterative gap filling
"""

from .core import get_dct_transfer_function, dct_convolve_1d
from .cartesian import smooth_cartesian
from .polar import smooth_polar
from .stats import dct_count, dct_mean, dct_variance, dct_std
from .gap_filling import iterative_gap_fill

__version__ = "0.2.1"

def dct_smooth(
    data, 
    width, 
    coordinates='cartesian', 
    **kwargs
):
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
    if coordinates == 'cartesian':
        return smooth_cartesian(data, width, **kwargs)
    elif coordinates == 'polar':
        # Polar func takes width_pixels
        return smooth_polar(data, width_pixels=width, **kwargs)
    else:
        raise ValueError(f"Unknown coordinates: {coordinates}")
