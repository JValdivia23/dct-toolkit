"""
Basic Polar Smoothing Example using dct-toolkit.

This script demonstrates:
1. Creating synthetic polar data (Azimuth x Range)
2. Applying smoothing with both 'reflective' and 'periodic' boundary conditions
3. Comparing the results
"""

import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dct_toolkit as dct


def main() -> None:
    print("DCT Toolkit - Polar Smoothing Demo")
    print("----------------------------------")

    n_az = 360
    n_range = 100
    az_res = 1.0

    az_idx = np.arange(n_az)
    field = (az_idx / n_az)[:, np.newaxis] * np.ones((1, n_range))

    np.random.seed(42)
    noisy_field = field + 0.1 * np.random.randn(n_az, n_range)

    print(f"Data shape: {noisy_field.shape}")
    print("Applying smoothing (width=10 pixels)...")

    smooth_periodic = dct.dct_smooth(
        noisy_field,
        width=10.0,
        coordinates="polar",
        az_boundary="periodic",
        az_res_deg=az_res,
    )

    smooth_reflective = dct.dct_smooth(
        noisy_field,
        width=10.0,
        coordinates="polar",
        az_boundary="reflective",
        az_res_deg=az_res,
    )

    # Periodic: value at 0 is pulled UP by value at 359 (which is ~1.0).
    # Reflective: value at 0 is pulled DOWN by value at 1 (which is ~0.0).
    val_0_periodic = smooth_periodic[0, 50]
    val_0_reflective = smooth_reflective[0, 50]

    print("\nResults at Azimuth 0 (Boundary):")
    print(f"  Periodic (Correct):   {val_0_periodic:.4f} (Influenced by Az=359 -> 1.0)")
    print(f"  Reflective (Wrong):   {val_0_reflective:.4f} (Influenced by Az=1 -> 0.0)")

    diff = val_0_periodic - val_0_reflective
    print(f"  Difference:           {diff:.4f}")

    if diff > 0.1:
        print("\nSUCCESS: Periodic boundary condition is working correctly!")
    else:
        print("\nFAILURE: Boundary conditions indistinguishable.")


if __name__ == "__main__":
    main()
