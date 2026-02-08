"""
DCT Statistical Operations.

This module implements robust local statistics (Mean, Variance, Std, Count)
using Normalized Convolution. This approach naturally handles gaps (NaNs)
without requiring explicit pre-filling.
"""

import numpy as np
from .core import get_dct_transfer_function, dct_convolve_1d
from .cartesian import smooth_cartesian
from .polar import smooth_polar

def _get_smooth_func(coordinates: str):
    """Select smoothing function based on coordinate system."""
    if coordinates == 'cartesian':
        return smooth_cartesian
    elif coordinates == 'polar':
        return smooth_polar
    else:
        raise ValueError(f"Unknown coordinates: {coordinates}")

def dct_count(
    mask: np.ndarray, 
    width: float, 
    coordinates: str = 'cartesian', 
    **kwargs
) -> np.ndarray:
    """
    Compute effective sample count (local density * window area).
    
    Parameters
    ----------
    mask : np.ndarray
        Boolean or binary mask (1=valid, 0=invalid).
    width : float
        Smoothing width.
    coordinates : str
        'cartesian' or 'polar'.
    **kwargs
        Additional arguments passed to smoothing function (e.g. az_res_deg).
        
    Returns
    -------
    count : np.ndarray
        Effective count of valid samples within the smoothing window.
    """
    smooth_func = _get_smooth_func(coordinates)
    
    # Density = Smooth(Indicator)
    density = smooth_func(mask.astype(float), width, **kwargs)
    
    # Area Calculation
    if coordinates == 'cartesian':
        # Area = width^ndim (assuming isotropic width)
        area = width ** mask.ndim
    elif coordinates == 'polar':
        # Area varies with range: w_az(r) * w_range
        # w_range = width
        # w_az(r) = width / (r * d_theta) [in beam units]
        # But wait, w_beams was used in smoothing. 
        # Area in index-space (which DCT operates on) is what matters for "count".
        # DCT smoothing effectively averages over a window defined in index space 
        # but weighted by the kernel.
        # For polar, the azimuth kernel width in indices is w_beams[r].
        # So Area[r] = width * w_beams[r]
        
        n_az, n_range = mask.shape
        az_res_deg = kwargs.get('az_res_deg', 1.0)
        az_res_rad = np.deg2rad(az_res_deg)
        r_indices = np.arange(1, n_range + 1)
        
        # w_beams[r] is width in azimuth indices
        w_beams = width / (r_indices * az_res_rad)
        
        # Effective area in (az, range) index space
        area_1d = width * w_beams
        area = np.tile(area_1d, (n_az, 1))
    else:
        area = 1.0
        
    return density * area

def dct_mean(
    data: np.ndarray, 
    width: float, 
    coordinates: str = 'cartesian', 
    mask: np.ndarray = None,
    **kwargs
) -> np.ndarray:
    """
    Compute robust local mean using Normalized Convolution.
    
    Mean = Smooth(Data * Mask) / Smooth(Mask)
    
    Parameters
    ----------
    data : np.ndarray
        Input data (can contain NaNs).
    width : float
        Smoothing width.
    coordinates : str
        'cartesian' or 'polar'.
    mask : np.ndarray, optional
        Valid data mask. If None, inferred from ~isnan(data).
        
    Returns
    -------
    mean : np.ndarray
        Local mean.
    """
    smooth_func = _get_smooth_func(coordinates)
    
    if mask is None:
        mask = ~np.isnan(data)
    
    # 1. Numerator: Smooth(Data * Mask)
    # Fill NaNs with 0 for the convolution (they are masked out anyway)
    data_filled = data.copy()
    data_filled[~mask] = 0.0
    numerator = smooth_func(data_filled, width, **kwargs)
    
    # 2. Denominator: Smooth(Mask)
    denominator = smooth_func(mask.astype(float), width, **kwargs)
    
    # 3. Normalized Ratio
    # Handle division by zero where denominator is very small (no valid data nearby)
    valid_den = denominator > 1e-10
    mean = np.full_like(data, np.nan)
    
    mean[valid_den] = numerator[valid_den] / denominator[valid_den]
    
    return mean

def dct_variance(
    data: np.ndarray, 
    width: float, 
    coordinates: str = 'cartesian', 
    mask: np.ndarray = None,
    **kwargs
) -> np.ndarray:
    """
    Compute robust local variance.
    
    Var = E[X^2] - (E[X])^2
    Both expectations are computed using Normalized Convolution.
    
    Parameters
    ----------
    data : np.ndarray
        Input data.
    width : float
        Smoothing width.
        
    Returns
    -------
    variance : np.ndarray
    """
    if mask is None:
        mask = ~np.isnan(data)
        
    # E[X]
    mean = dct_mean(data, width, coordinates, mask, **kwargs)
    
    # E[X^2]
    data_sq = data ** 2
    mean_sq = dct_mean(data_sq, width, coordinates, mask, **kwargs)
    
    # Var = E[X^2] - E[X]^2
    # Use maximum(0) to avoid negative variance due to numerical precision
    variance = np.maximum(mean_sq - mean**2, 0.0)
    
    return variance

def dct_std(
    data: np.ndarray, 
    width: float, 
    coordinates: str = 'cartesian', 
    mask: np.ndarray = None,
    **kwargs
) -> np.ndarray:
    """Compute robust local standard deviation."""
    var = dct_variance(data, width, coordinates, mask, **kwargs)
    return np.sqrt(var)
