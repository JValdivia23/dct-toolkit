"""
Iterative Gap Filling using DCT Statistics.

This experimental module demonstrates how to build a gap filling algorithm
from the `dct_mean` primitive.

Algorithm:
1. Initialize filled data (e.g. with global mean or 0).
2. Iteratively replace ONLY the missing values with the robust local mean
   calculated from the current filled data.
3. Converges to a solution where missing values are consistent with the
   smooth trends of the surrounding valid data.
"""

import numpy as np
from dct_toolkit.stats import dct_mean

def iterative_gap_fill(
    data: np.ndarray, 
    width: float, 
    coordinates: str = 'cartesian',
    max_iter: int = 50,
    tol: float = 1e-4,
    **kwargs
) -> np.ndarray:
    """
    Fill gaps using iterative robust smoothing.
    
    Parameters
    ----------
    data : np.ndarray
        Input data with NaNs.
    width : float
        Smoothing width.
    coordinates : str
        'cartesian' or 'polar'.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance (relative change).
        
    Returns
    -------
    filled : np.ndarray
        Data with gaps filled.
    """
    # Identify gaps
    mask = ~np.isnan(data)
    if np.all(mask):
        return data.copy()
        
    # Initialize gaps
    # A good initialization helps convergence.
    # Simple strategy: Fill with 0.0 (masked out anyway in first step of dct_mean)
    # But for the "feedback" part, we need values.
    # dct_mean handles NaNs in input, so we can pass 'data' directly in first iter?
    # Yes, dct_mean(data) returns a smooth field where valid data dominates.
    
    filled = data.copy()
    # Initial fill with 0 or global mean for the gaps
    # (Though dct_mean handles NaNs, having values helps if we use filled in next step)
    # Actually, in the loop:
    # Estimate = Smooth(Current_State)
    # Update Gaps = Estimate[Gaps]
    
    # First estimate using only valid data
    estimate = dct_mean(data, width, coordinates, mask=mask, **kwargs)
    
    # Fill gaps with first estimate
    filled[~mask] = estimate[~mask]
    
    # Iterate
    for i in range(max_iter):
        prev_filled = filled.copy()
        
        # Compute smooth trend of the CURRENT filled state
        # Note: We now treat the filled values as "valid" for the smoothing
        # to propagate information into large gaps.
        # However, we must trust the ORIGINAL valid data more?
        # Standard approach: Smooth the full 'filled' array.
        
        # We use dct_mean but now we can consider ALL data valid (mask=None)
        # or we can stick to Normalized Convolution but that ignores the filled values.
        # We WANT to use the filled values to support further filling (diffusion).
        # So we treat all points as valid.
        
        trend = dct_mean(filled, width, coordinates, mask=np.ones_like(filled, dtype=bool), **kwargs)
        
        # Update only the gaps
        filled[~mask] = trend[~mask]
        
        # Check convergence
        diff = np.linalg.norm(filled - prev_filled)
        norm = np.linalg.norm(filled)
        rel_change = diff / (norm + 1e-10)
        
        if rel_change < tol:
            break
            
    return filled
