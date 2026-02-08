"""
Core DCT Primitives and Transfer Functions.

This module provides the low-level building blocks for DCT-based smoothing:
1. Analytical and discrete transfer functions (H[k])
2. 1D convolution primitive using DCT-II/Ortho
"""

import numpy as np
import scipy.fft

def get_dct_transfer_function(n: int, kernel_type: str, width: float) -> np.ndarray:
    """
    Generate the 1D DCT Transfer Function H[k].

    Parameters
    ----------
    n : int
        Number of points.
    kernel_type : str
        'boxcar', 'boxcar_discrete', or 'gaussian'.
    width : float
        Kernel width.
        - boxcar: full width
        - gaussian: equivalent boxcar width (sigma = width / sqrt(12))

    Returns
    -------
    H : np.ndarray
        Transfer function of shape (n,).
    """
    k = np.arange(n)
    
    if kernel_type == 'boxcar':
        # Analytical Boxcar: H = sin(W * theta) / (W * sin(theta))
        # where theta = pi * k / (2*n)
        theta_half = (np.pi * k) / (2 * n)
        
        H = np.zeros(n)
        H[0] = 1.0
        
        mask = k > 0
        if np.any(mask):
            numerator = np.sin(width * theta_half[mask])
            denominator = np.sin(theta_half[mask])
            
            # Avoid division by zero
            valid = np.abs(denominator) > 1e-15
            safe_mask = np.zeros_like(mask)
            safe_mask[mask] = valid
            
            if np.any(safe_mask):
                H[safe_mask] = numerator[safe_mask[mask]] / (denominator[safe_mask[mask]] * width)
                
        return H
        
    elif kernel_type == 'boxcar_discrete':
        # Discrete Boxcar Sum
        # H[k] = (1 + 2*sum_{j=1}^{m} cos(j*w_k)) / w_int
        w_int = int(np.round(width))
        if w_int < 1: w_int = 1
        if w_int % 2 == 0: w_int += 1
        
        w_k = (np.pi * k) / n
        H = np.ones(n)
        half_width = (w_int - 1) // 2
        
        for j in range(1, half_width + 1):
            H += 2 * np.cos(j * w_k)
            
        H /= w_int
        return H
        
    elif kernel_type == 'gaussian':
        # Gaussian: sigma = width / sqrt(12)
        sigma = width / np.sqrt(12)
        if sigma < 1e-9:
            return np.ones(n)
            
        # omega_k = pi * k / n
        omega_k = (np.pi * k) / n
        return np.exp(-0.5 * (omega_k * sigma)**2)
    
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

def dct_convolve_1d(data: np.ndarray, H: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Apply 1D DCT-based convolution.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    H : np.ndarray
        Transfer function (must match data.shape[axis]).
    axis : int
        Axis to smooth along.

    Returns
    -------
    smoothed : np.ndarray
        Smoothed data.
    """
    # Validate shapes
    if data.shape[axis] != len(H):
        raise ValueError(f"Transfer function length {len(H)} does not match data dimension {data.shape[axis]}")

    # Broadcast H to data shape
    shape = [1] * data.ndim
    shape[axis] = len(H)
    H_reshaped = H.reshape(shape)
    
    # Forward DCT (Type-II, Ortho)
    X = scipy.fft.dct(data, axis=axis, type=2, norm='ortho')
    
    # Apply transfer function
    Y = X * H_reshaped
    
    # Inverse DCT (Type-II, Ortho)
    y = scipy.fft.idct(Y, axis=axis, type=2, norm='ortho')
    
    return y
