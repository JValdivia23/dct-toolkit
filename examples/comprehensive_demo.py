"""
# DCT Toolkit: Comprehensive Demonstration

This script demonstrates the end-to-end workflow of `dct-toolkit`, from basic smoothing
to advanced gap filling on polar data.

## 1. Setup
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add toolkit to path (if not installed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dct_toolkit as dct

print(f"DCT Toolkit Version: {dct.__version__}")

"""
## 2. 1D Smoothing & Derivatives
We'll generate a noisy signal and recover it.
"""
# Generate synthetic signal
np.random.seed(42)
t = np.linspace(0, 10, 200)
signal = np.sin(t) + 0.5 * np.sin(3*t)
noise = 0.2 * np.random.randn(len(t))
data = signal + noise

# Apply DCT smoothing (Gaussian kernel)
smooth_width = 5.0
smoothed = dct.dct_smooth(data, width=smooth_width, kernel_type='gaussian')

# Compute approximate derivative using central differences of smoothed signal
deriv = np.gradient(smoothed, t)

print(f"1D Smoothing (width={smooth_width}):")
print(f"  RMSE Raw:      {np.sqrt(np.mean((data - signal)**2)):.4f}")
print(f"  RMSE Smoothed: {np.sqrt(np.mean((smoothed - signal)**2)):.4f}")

"""
## 3. Robust Statistics with Gaps (Normalized Convolution)
We'll introduce gaps and show how `dct_mean` handles them naturally.
"""
# Introduce 40% random gaps
mask = np.random.rand(len(t)) > 0.4
data_gappy = data.copy()
data_gappy[~mask] = np.nan

# Compute robust mean (recovers signal despite gaps)
robust_mean = dct.dct_mean(data_gappy, width=smooth_width)

print("\nRobust Statistics:")
print(f"  Gap Fraction:  {np.sum(~mask)/len(mask):.1%}")
print(f"  RMSE Robust:   {np.sqrt(np.mean((robust_mean - signal)**2)):.4f}")

"""
## 4. 2D Polar Smoothing (Radar Use Case)
Demonstrating adaptive kernels and periodic boundaries.
"""
# Create a synthetic polar field (360 azimuth x 100 range)
n_az = 360
n_range = 100
az_res = 1.0

# Create a spiral pattern
az_grid, r_grid = np.meshgrid(np.deg2rad(np.arange(n_az)), np.arange(n_range), indexing='ij')
polar_signal = np.sin(3*az_grid + r_grid/10.0)

# Add noise
polar_noisy = polar_signal + 0.5 * np.random.randn(n_az, n_range)

# Smooth with PERIODIC boundary (correct for azimuth)
smooth_periodic = dct.dct_smooth(
    polar_noisy, 
    width=5.0, 
    coordinates='polar', 
    az_boundary='periodic',
    az_res_deg=az_res
)

# Smooth with REFLECTIVE boundary (incorrect for azimuth)
smooth_reflective = dct.dct_smooth(
    polar_noisy, 
    width=5.0, 
    coordinates='polar', 
    az_boundary='reflective',
    az_res_deg=az_res
)

# Check boundary discontinuity at Az=0 vs Az=359
disc_periodic = np.abs(smooth_periodic[0,:] - smooth_periodic[-1,:])
disc_reflective = np.abs(smooth_reflective[0,:] - smooth_reflective[-1,:])

print("\nPolar Smoothing:")
print(f"  Mean Discontinuity (Periodic):   {np.mean(disc_periodic):.4f} (Expected: Low)")
print(f"  Mean Discontinuity (Reflective): {np.mean(disc_reflective):.4f} (Expected: Higher)")

"""
## 5. Experimental: Iterative Gap Filling
Constructing missing data from valid surroundings.
"""
# Create large gaps (sectors)
polar_gappy = polar_noisy.copy()
polar_gappy[50:100, :] = np.nan  # 50-degree sector gap

# Import experimental filler
try:
    from dct_toolkit.experimental.gap_filling.iterative_fill import iterative_gap_fill
    
    # Fill gaps iteratively
    print("\nIterative Gap Filling...")
    filled = iterative_gap_fill(polar_gappy, width=10.0, coordinates='polar', max_iter=20)
    
    # Measure error in the gap
    gap_mask = np.isnan(polar_gappy)
    mae_gap = np.mean(np.abs(filled[gap_mask] - polar_signal[gap_mask]))
    print(f"  MAE in Gap: {mae_gap:.4f}")
    
except ImportError:
    print("\n(Experimental module not found in path, skipping gap filling demo)")

print("\nDemo Complete.")
