"""
2D Polarimetric Smoothing.

This module provides smoothing for polar coordinate data (Azimuth x Range).
It handles the non-uniform physical width of azimuth beams by adapting
the smoothing kernel in the spectral domain.
"""

import numpy as np
import scipy.fft
import warnings
from typing import Tuple, Optional
from .core import get_dct_transfer_function


def compute_polar_transfer_functions(
    shape: Tuple[int, int],
    az_res_deg: float,
    width_pixels: float,
    kernel_type: str = "gaussian",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute transfer functions for polar smoothing.

    The azimuth kernel width adapts with range to maintain constant physical width.

    Parameters
    ----------
    shape : tuple
        (n_azimuth, n_range).
    az_res_deg : float
        Azimuth resolution in degrees.
    width_pixels : float
        Target smoothing width in pixels (at reference range).
    kernel_type : str
        'boxcar', 'boxcar_discrete', 'gaussian'.

    Returns
    -------
    H_az : np.ndarray
        Azimuth transfer function of shape (n_az, n_range).
    H_range : np.ndarray
        Range transfer function of shape (n_range,).
    """
    n_az, n_range = shape
    az_res_rad = np.deg2rad(az_res_deg)

    # 1. Range Transfer Function (fixed width)
    H_range = get_dct_transfer_function(n_range, kernel_type, width_pixels)

    # 2. Azimuth Transfer Function (varies with range)
    # We construct it column-wise (per range gate)
    H_az_T = np.zeros((n_range, n_az))

    # Pre-calculate range indices (1-based to avoid div by zero)
    # Range gate index roughly correlates with distance
    r_indices = np.arange(1, n_range + 1)

    # Calculate effective beam width for each range
    # w_beams(r) = width_pixels / (r * az_res_rad)
    # This assumes range resolution corresponds to r=1 unit distance
    # and we want physical arc length ~ width_pixels * range_resolution
    w_beams = width_pixels / (r_indices * az_res_rad)

    # Generate H for each range bin
    # Vectorized generation is hard because 'width' changes, loop is safer/clearer
    # Optimization: group ranges with similar widths if needed, but for now loop is fine
    for i in range(n_range):
        H_az_T[i, :] = get_dct_transfer_function(n_az, kernel_type, w_beams[i])

    H_az = H_az_T.T  # (n_az, n_range)

    return H_az, H_range


def _smooth_azimuth_periodic(data: np.ndarray, H_az: np.ndarray) -> np.ndarray:
    """
    Smooth azimuth using Real FFT (Periodic Boundary Condition).

    H_az needs to be adapted for DFT frequencies: k = 0, ..., n/2
    Current get_dct_transfer_function returns DCT frequencies (k=0..n-1).
    We need to re-compute H for DFT frequencies if using periodic BC.

    Wait, get_dct_transfer_function computes H based on k.
    For DCT: theta = pi * k / (2n) or pi * k / n
    For DFT: theta = 2 * pi * k / n

    So we cannot reuse H_az directly if it was computed for DCT.
    We need a separate transfer function generator for Periodic (DFT).
    """
    # This helper logic handles the re-computation internally for now
    n_az, n_range = data.shape
    n_rfft = n_az // 2 + 1

    # We need to regenerate H_az for DFT frequencies
    # This is a bit inefficient to do here, but cleaner than complicating the public API
    # Let's extract the width parameter from H_az? No, passed implicitly.
    # Ideally, compute_polar_transfer_functions should know about BC.
    pass


# Revised approach: compute_polar_transfer_functions should return
# the correct H based on the boundary condition (DCT vs DFT).


def compute_polar_transfer_functions_v2(
    shape: Tuple[int, int],
    az_res_deg: float,
    width_pixels: float,
    kernel_type: str = "gaussian",
    az_boundary: str = "reflective",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute transfer functions for polar smoothing.

    Parameters
    ----------
    ...
    az_boundary : str
        'reflective' (DCT) or 'periodic' (DFT/RFFT).
    """
    n_az, n_range = shape
    az_res_rad = np.deg2rad(az_res_deg)

    # Range H (Always DCT/Reflective for now)
    H_range = get_dct_transfer_function(n_range, kernel_type, width_pixels)

    # Azimuth H
    r_indices = np.arange(1, n_range + 1)
    w_beams = width_pixels / (r_indices * az_res_rad)

    if az_boundary == "reflective":
        # Use standard DCT transfer function generator
        H_az_T = np.zeros((n_range, n_az))
        for i in range(n_range):
            H_az_T[i, :] = get_dct_transfer_function(n_az, kernel_type, w_beams[i])
        H_az = H_az_T.T

    elif az_boundary == "periodic":
        # Use DFT transfer function generator
        # RFFT frequencies: k = 0, 1, ..., n/2
        n_freq = n_az // 2 + 1
        H_az_T = np.zeros((n_range, n_freq))

        k = np.arange(n_freq)

        for i in range(n_range):
            width = w_beams[i]

            # Compute DFT Transfer Function
            if kernel_type == "boxcar":
                # H = sin(W * theta) / (W * sin(theta))
                # DFT theta = pi * k / n (half period? No, full period 2pi)
                # Sinc is sin(W * omega/2) / ...
                # Let's match the standard definition:
                # continuous: sin(f*W)/...
                # DFT freq bin k corresponds to frequency k/N

                # Adapting from examples/dct_smoothing.py:
                # theta_half = (np.pi * k) / n_az  (Note: DCT was pi*k/(2n))
                theta_half = (np.pi * k) / n_az

                H = np.zeros(n_freq)
                H[0] = 1.0
                mask = k > 0

                num = np.sin(width * theta_half[mask])
                den = np.sin(theta_half[mask])
                valid = np.abs(den) > 1e-15

                H[mask] = np.where(valid, num / (den * width), 0.0)
                H_az_T[i, :] = H

            elif kernel_type == "gaussian":
                # omega = 2 * pi * k / n
                omega = (2 * np.pi * k) / n_az
                sigma = width / np.sqrt(12)
                H_az_T[i, :] = np.exp(-0.5 * (omega * sigma) ** 2)

            else:
                # Fallback for others (discrete boxcar not implemented for periodic yet)
                # Use Gaussian approx
                omega = (2 * np.pi * k) / n_az
                sigma = width / np.sqrt(12)
                H_az_T[i, :] = np.exp(-0.5 * (omega * sigma) ** 2)

        H_az = H_az_T.T

    else:
        raise ValueError(f"Unknown azimuth boundary: {az_boundary}")

    return H_az, H_range


def smooth_polar(
    data: np.ndarray,
    width_pixels: float,
    az_res_deg: float = 1.0,
    az_boundary: str = "reflective",
    range_boundary: str = "reflective",
    kernel_type: str = "gaussian",
) -> np.ndarray:
    """
    Apply smoothing to 2D polar data (Azimuth x Range).

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (n_azimuth, n_range).
    width_pixels : float
        Smoothing width in pixels (at reference range).
    az_res_deg : float
        Azimuth resolution in degrees.
    az_boundary : str
        'reflective' (default) or 'periodic'.
    range_boundary : str
        'reflective' (default).
    kernel_type : str
        Kernel type.

    Returns
    -------
    smoothed : np.ndarray
    """
    # Validation
    if data.ndim != 2:
        raise ValueError(f"Data must be 2D (az, range), got {data.ndim}D")

    n_az, n_range = data.shape
    if n_az > 720 and n_range < n_az:
        warnings.warn(
            f"Data shape ({n_az}, {n_range}) has large first dimension. "
            "Ensure format is (n_azimuth, n_range).",
            UserWarning,
        )

    # Get transfer functions
    H_az, H_range = compute_polar_transfer_functions_v2(
        data.shape, az_res_deg, width_pixels, kernel_type, az_boundary
    )

    # 1. Azimuth Smoothing (Axis 0)
    if az_boundary == "reflective":
        # DCT
        X = scipy.fft.dct(data, axis=0, type=2, norm="ortho")
        Y = X * H_az
        step1 = scipy.fft.idct(Y, axis=0, type=2, norm="ortho")

    elif az_boundary == "periodic":
        # Real FFT
        X = scipy.fft.rfft(data, axis=0, norm="ortho")
        # H_az shape matches rfft output (n_az//2 + 1, n_range)
        Y = X * H_az
        step1 = scipy.fft.irfft(Y, n=n_az, axis=0, norm="ortho")

    else:
        raise ValueError(f"Unknown azimuth boundary: {az_boundary}")

    # 2. Range Smoothing (Axis 1)
    # Always reflective (DCT) for now
    if range_boundary != "reflective":
        raise NotImplementedError("Only 'reflective' range boundary supported")

    # H_range is (n_range,), broadcast to (n_az, n_range)
    X = scipy.fft.dct(step1, axis=1, type=2, norm="ortho")
    Y = X * H_range.reshape(1, -1)
    result = scipy.fft.idct(Y, axis=1, type=2, norm="ortho")

    return result
