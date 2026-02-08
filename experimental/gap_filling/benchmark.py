"""
Benchmark: Iterative Gap Filling (Constructive) vs Linear Interpolation.
"""

import numpy as np
import time
import os
import sys

# Ensure we can import dct_toolkit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import local experimental module
try:
    from iterative_fill import iterative_gap_fill
except ImportError:
    # Fallback if running from different directory
    sys.path.append(os.path.dirname(__file__))
    from iterative_fill import iterative_gap_fill

from scipy.interpolate import griddata

def create_synthetic_field(n=50):
    """Create a smooth 2D field."""
    x = np.linspace(0, 10, n)
    y = np.linspace(0, 10, n)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X/2) * np.cos(Y/2) + 0.1 * np.sin(2*X)
    return Z

def add_gaps(data, fraction=0.3):
    """Add random gaps."""
    mask = np.random.rand(*data.shape) > fraction
    data_gappy = data.copy()
    data_gappy[~mask] = np.nan
    return data_gappy, mask

def linear_interp(data):
    """Linear interpolation using griddata."""
    n = data.shape[0]
    x = np.arange(n)
    y = np.arange(n)
    grid_x, grid_y = np.meshgrid(x, y)
    
    points = []
    values = []
    
    # Slow loop-based extraction (optimize if needed, but griddata needs points)
    # Actually faster:
    valid_mask = ~np.isnan(data)
    points = np.column_stack(np.where(valid_mask))
    values = data[valid_mask]
    
    if len(values) < 4:
        return data
        
    filled = griddata(points, values, (grid_x, grid_y), method='linear')
    
    # Fill remaining NaNs (edges) with nearest
    nan_mask = np.isnan(filled)
    if np.any(nan_mask):
        filled[nan_mask] = griddata(points, values, 
                                   (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')
    return filled

def main():
    print("BENCHMARK: Constructive Gap Filling vs Linear")
    print("=============================================")
    
    n = 100
    print(f"Grid size: {n}x{n} ({n*n} points)")
    
    # Data
    np.random.seed(42)
    Z_true = create_synthetic_field(n)
    Z_gappy, mask = add_gaps(Z_true, fraction=0.3) # 30% gaps
    
    # 1. Linear Interpolation
    t0 = time.time()
    Z_linear = linear_interp(Z_gappy)
    t_linear = (time.time() - t0) * 1000
    mae_linear = np.mean(np.abs(Z_linear - Z_true)[~mask])
    print(f"Linear Interp:     {t_linear:.1f} ms, MAE = {mae_linear:.4f}")
    
    # 2. Iterative DCT Filling (Constructive)
    t0 = time.time()
    # Width parameter needs tuning relative to gap size
    Z_dct = iterative_gap_fill(Z_gappy, width=5.0, max_iter=20, tol=1e-4)
    t_dct = (time.time() - t0) * 1000
    mae_dct = np.mean(np.abs(Z_dct - Z_true)[~mask])
    print(f"Iterative DCT:     {t_dct:.1f} ms, MAE = {mae_dct:.4f}")
    
    # Improvement
    improvement = (1 - mae_dct / mae_linear) * 100
    print(f"Improvement:       {improvement:+.1f}%")

if __name__ == "__main__":
    main()
