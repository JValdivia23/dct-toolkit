"""Comprehensive demo for the stats-first public API surface."""

import os
import sys

import numpy as np

# Ensure import works without installation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dct_toolkit as dct


def main() -> None:
    print(f"DCT Toolkit Version: {dct.__version__}")

    np.random.seed(42)

    # 1) 1D smoothing
    t = np.linspace(0, 10, 200)
    signal = np.sin(t) + 0.5 * np.sin(3 * t)
    noise = 0.2 * np.random.randn(len(t))
    data = signal + noise

    smooth_width = 5.0
    smoothed = dct.dct_smooth(data, width=smooth_width, kernel_type="gaussian")

    print("1D Smoothing:")
    print(f"  RMSE raw:      {np.sqrt(np.mean((data - signal) ** 2)):.4f}")
    print(f"  RMSE smoothed: {np.sqrt(np.mean((smoothed - signal) ** 2)):.4f}")

    # 2) Robust statistics with gaps
    mask = np.random.rand(len(t)) > 0.4
    data_gappy = data.copy()
    data_gappy[~mask] = np.nan

    robust_mean = dct.dct_mean(data_gappy, width=smooth_width)
    robust_std = dct.dct_std(data_gappy, width=smooth_width)

    print("\nRobust Statistics:")
    print(f"  Gap fraction:  {np.sum(~mask) / len(mask):.1%}")
    print(f"  RMSE mean:     {np.sqrt(np.mean((robust_mean - signal) ** 2)):.4f}")
    print(f"  Median std:    {np.nanmedian(robust_std):.4f}")

    # 3) 2D polar smoothing with periodic azimuth
    n_az = 360
    n_range = 100
    az_res = 1.0

    az_grid, r_grid = np.meshgrid(
        np.deg2rad(np.arange(n_az)),
        np.arange(n_range),
        indexing="ij",
    )
    polar_signal = np.sin(3 * az_grid + r_grid / 10.0)
    polar_noisy = polar_signal + 0.5 * np.random.randn(n_az, n_range)

    smooth_periodic = dct.dct_smooth(
        polar_noisy,
        width=5.0,
        coordinates="polar",
        az_boundary="periodic",
        az_res_deg=az_res,
    )
    smooth_reflective = dct.dct_smooth(
        polar_noisy,
        width=5.0,
        coordinates="polar",
        az_boundary="reflective",
        az_res_deg=az_res,
    )

    disc_periodic = np.abs(smooth_periodic[0, :] - smooth_periodic[-1, :])
    disc_reflective = np.abs(smooth_reflective[0, :] - smooth_reflective[-1, :])

    print("\nPolar Smoothing:")
    print(f"  Mean discontinuity (periodic):   {np.mean(disc_periodic):.4f}")
    print(f"  Mean discontinuity (reflective): {np.mean(disc_reflective):.4f}")
    print("\nDemo complete.")


if __name__ == "__main__":
    main()
