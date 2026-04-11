"""
Width & Iteration Impact Diagnostic for Linear-Init Gap Filling.

Replicates the circular-hole benchmark scenario (720x1000 polar grid,
smooth Gaussian blobs, circular hole center=(200,0) radius=100) and
sweeps width x iterations to quantify accuracy, uncertainty, and timing.

Compares:
  - 2D griddata (linear + nearest fallback)  — accuracy target
  - Linear init only (0 DCT iterations)
  - Linear init + DCT at various (width, max_iter) combos
  - Legacy multi-scale cascade (width=5, max_iter=50)

For ALL methods computes at reference width=5:
  - MAE in gap region vs ground truth
  - dct_std of the filled field (signal variability)
  - Mapping error = 1 - dct_mean(indicator, width)
  - Wall-clock time

Usage
-----
    conda run -n myenv python exp_v3/test_width_impact.py
"""

import sys
import os
import time

import numpy as np
from scipy.interpolate import griddata

# Ensure dct_toolkit is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dct_toolkit'))

from dct_toolkit import dct_mean, dct_std, iterative_gap_fill


# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------

def make_cartesian_ground_truth(
    shape: tuple,
    range_res_m: float = 100.0,
    az_res_deg: float = 0.5,
) -> np.ndarray:
    """
    Create smooth 2-D ground truth field in Cartesian coordinates.

    Defines the field in Cartesian (x, y) space then maps to polar grid
    to avoid singularity artifacts near the radar center.

    Parameters
    ----------
    shape : tuple
        (n_azimuth, n_range).
    range_res_m : float
        Range gate resolution in meters (default: 100m).
    az_res_deg : float
        Azimuthal resolution in degrees (default: 0.5°).

    Returns
    -------
    field : np.ndarray
        Smooth field with values roughly in [20, 45].

    Notes
    -----
    The ground truth consists of:
    - Base value: 25.0
    - East-West wave with 40 km wavelength (amplitude 8.0)
    - North-South wave with 30 km wavelength (amplitude 6.0)
    - Gaussian feature at (20, 15) km with 200 km² variance (amplitude 4.0)
    - Radial pattern with 5 km scale (amplitude 3.0)

    This formulation is smooth everywhere including at the radar origin,
    avoiding singularity artifacts that occur when defining patterns
    directly in polar coordinates.
    """
    n_az, n_range = shape

    # Create polar coordinate grid
    az_deg = np.linspace(0, 360, n_az, endpoint=False)
    range_m = np.arange(n_range) * range_res_m
    AZ_rad, RG_m = np.meshgrid(np.deg2rad(az_deg), range_m, indexing='ij')

    # Convert to Cartesian coordinates (km)
    # X: East-West, Y: North-South
    X_cart = RG_m * np.sin(AZ_rad) / 1000.0  # km
    Y_cart = RG_m * np.cos(AZ_rad) / 1000.0  # km

    # Build ground truth in Cartesian space (smooth everywhere)
    ground_truth = (
        25.0 +  # Base value
        8.0 * np.sin(2 * np.pi * X_cart / 40.0) +  # East-West wave (40 km wavelength)
        6.0 * np.cos(2 * np.pi * Y_cart / 30.0) +  # North-South wave (30 km wavelength)
        4.0 * np.exp(-((X_cart - 20)**2 + (Y_cart - 15)**2) / 200.0) +  # Gaussian at (20, 15) km
        3.0 * np.sin(np.sqrt(X_cart**2 + Y_cart**2) / 5.0)  # Radial pattern (smooth everywhere)
    )

    return ground_truth


def make_circular_hole(
    shape: tuple,
    center: tuple = (200, 0),
    radius: float = 100.0,
    range_res_m: float = 100.0,
    az_res_deg: float = 0.5,
) -> np.ndarray:
    """
    Create a boolean mask with a circular hole in Cartesian physical space (True = valid).

    Parameters
    ----------
    shape : tuple
        (n_azimuth, n_range).
    center : tuple
        (range_center, azimuth_center) of hole in pixel coordinates.
        Convention: range is axis=1, azimuth is axis=0.
    radius : float
        Hole radius in pixels (converted to meters using range_res_m).
    range_res_m : float
        Range gate resolution in meters.
    az_res_deg : float
        Azimuthal resolution in degrees.

    Returns
    -------
    mask : np.ndarray of bool
        True where data is valid, False inside the hole.
    """
    n_az, n_range = shape
    az = np.arange(n_az)
    rg = np.arange(n_range)
    AZ_idx, RG_idx = np.meshgrid(az, rg, indexing='ij')

    # Convert grid to physical Cartesian coordinates
    az_rad = np.deg2rad(AZ_idx * az_res_deg)
    range_m = RG_idx * range_res_m
    X = range_m * np.sin(az_rad)
    Y = range_m * np.cos(az_rad)

    # Convert center to physical Cartesian coordinates
    c_rg_idx, c_az_idx = center
    c_az_rad = np.deg2rad(c_az_idx * az_res_deg)
    c_range_m = c_rg_idx * range_res_m
    cX = c_range_m * np.sin(c_az_rad)
    cY = c_range_m * np.cos(c_az_rad)

    # Convert radius to meters
    radius_m = radius * range_res_m

    # Compute distance in physical space
    dist = np.sqrt((X - cX) ** 2 + (Y - cY) ** 2)
    mask = dist > radius_m
    return mask


def fill_griddata(
    truth: np.ndarray,
    mask: np.ndarray,
    range_res_m: float = 100.0,
    az_res_deg: float = 0.5,
) -> np.ndarray:
    """
    Fill gaps using scipy griddata in Cartesian physical space (linear + nearest fallback).

    Parameters
    ----------
    truth : np.ndarray
        Ground truth field (used only to extract valid values).
    mask : np.ndarray
        Boolean mask (True = valid).
    range_res_m : float
        Range gate resolution in meters.
    az_res_deg : float
        Azimuthal resolution in degrees.

    Returns
    -------
    filled : np.ndarray
    """
    n_az, n_range = truth.shape
    az = np.arange(n_az)
    rg = np.arange(n_range)
    AZ_idx, RG_idx = np.meshgrid(az, rg, indexing='ij')

    # Convert to physical Cartesian coordinates
    az_rad = np.deg2rad(AZ_idx * az_res_deg)
    range_m = RG_idx * range_res_m
    X = range_m * np.sin(az_rad)
    Y = range_m * np.cos(az_rad)

    points = np.column_stack([X[mask], Y[mask]])
    values = truth[mask]
    xi = np.column_stack([X.ravel(), Y.ravel()])

    filled_linear = griddata(points, values, xi, method='linear')
    filled_linear = filled_linear.reshape(truth.shape)

    # Nearest fallback for NaN edges
    nan_mask = np.isnan(filled_linear)
    if np.any(nan_mask):
        filled_nearest = griddata(points, values, xi, method='nearest')
        filled_nearest = filled_nearest.reshape(truth.shape)
        filled_linear[nan_mask] = filled_nearest[nan_mask]

    return filled_linear


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    filled: np.ndarray,
    truth: np.ndarray,
    gap_mask: np.ndarray,
    ref_width: float = 5.0,
) -> dict:
    """
    Compute MAE, dct_std, and mapping error for a filled field.

    Parameters
    ----------
    filled : np.ndarray
        Gap-filled field.
    truth : np.ndarray
        Ground truth field.
    gap_mask : np.ndarray
        Boolean mask: True where data was missing (gap).
    ref_width : float
        Reference smoothing width for dct_std and mapping error.

    Returns
    -------
    metrics : dict
        Keys: 'mae', 'dct_std_mean', 'mapping_error_mean',
              'mapping_error_max'.
    """
    # MAE in gap region
    mae = np.mean(np.abs(filled[gap_mask] - truth[gap_mask]))

    # dct_std of the filled field (using all-valid mask)
    std_field = dct_std(
        filled, ref_width, coordinates='polar', az_res_deg=0.5,
        mask=np.ones_like(filled, dtype=bool),
    )
    dct_std_mean = np.mean(std_field[gap_mask])

    # Mapping error = 1 - dct_mean(indicator, width)
    indicator = (~gap_mask).astype(float)
    density = dct_mean(
        indicator, ref_width, coordinates='polar', az_res_deg=0.5,
        mask=np.ones_like(indicator, dtype=bool),
    )
    mapping_error = 1.0 - np.clip(density, 0, 1)
    me_mean = np.mean(mapping_error[gap_mask])
    me_max = np.max(mapping_error[gap_mask])

    return {
        'mae': mae,
        'dct_std_mean': dct_std_mean,
        'mapping_error_mean': me_mean,
        'mapping_error_max': me_max,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Setup ---
    shape = (720, 1000)
    ref_width = 5.0
    widths = [3, 5, 10, 20, 50]
    iters = [0, 1, 3, 5, 10, 20]

    print("=" * 90)
    print("DCT Gap Filling — Linear Init Diagnostic")
    print("=" * 90)
    print(f"Grid: {shape[0]} x {shape[1]}  |  "
          f"Hole: center=(200,0), radius=100  |  "
          f"Ref width: {ref_width}")
    print()

    # Build scenario with Cartesian-based ground truth (no singularity artifacts)
    truth = make_cartesian_ground_truth(shape, range_res_m=100.0, az_res_deg=0.5)
    mask = make_circular_hole(shape, center=(200, 0), radius=100, range_res_m=100.0, az_res_deg=0.5)
    gap_mask = ~mask
    n_gap = np.sum(gap_mask)

    gapped = truth.copy()
    gapped[gap_mask] = np.nan

    print(f"Gap pixels: {n_gap} ({100 * n_gap / truth.size:.1f}% of grid)")
    print()

    results = []

    # --- 1. Griddata baseline ---
    print("Running: griddata (linear + nearest) ...")
    t0 = time.time()
    filled_gd = fill_griddata(truth, mask, range_res_m=100.0, az_res_deg=0.5)
    t_gd = time.time() - t0
    m_gd = compute_metrics(filled_gd, truth, gap_mask, ref_width)
    results.append(('griddata', '-', '-', m_gd['mae'], m_gd['dct_std_mean'],
                     m_gd['mapping_error_mean'], m_gd['mapping_error_max'],
                     t_gd))
    print(f"  MAE = {m_gd['mae']:.6f}  time = {t_gd:.2f}s")

    # --- 2. Linear init only (0 iterations) ---
    print("Running: linear_init_only (0 DCT iterations) ...")
    t0 = time.time()
    filled_li = iterative_gap_fill(
        gapped, ref_width, coordinates='polar', az_boundary='periodic',
        az_res_deg=0.5, init='linear', max_iter=0,
    )
    t_li = time.time() - t0
    m_li = compute_metrics(filled_li, truth, gap_mask, ref_width)
    results.append(('linear_init_only', '-', '0', m_li['mae'],
                     m_li['dct_std_mean'], m_li['mapping_error_mean'],
                     m_li['mapping_error_max'], t_li))
    print(f"  MAE = {m_li['mae']:.6f}  time = {t_li:.2f}s")

    # --- 3. Legacy multi-scale cascade (width=5, iter=50) ---
    print("Running: multiscale cascade (width=5, iter=50) ...")
    t0 = time.time()
    filled_ms = iterative_gap_fill(
        gapped, 5.0, coordinates='polar', az_boundary='periodic',
        az_res_deg=0.5, init='multiscale', max_iter=50,
    )
    t_ms = time.time() - t0
    m_ms = compute_metrics(filled_ms, truth, gap_mask, ref_width)
    results.append(('multiscale_cascade', '5', '50', m_ms['mae'],
                     m_ms['dct_std_mean'], m_ms['mapping_error_mean'],
                     m_ms['mapping_error_max'], t_ms))
    print(f"  MAE = {m_ms['mae']:.6f}  time = {t_ms:.2f}s")

    # --- 3b. DCT Init (Normalized Convolution) w=50, iter=20 ---
    print("Running: DCT Init + DCT (width=50, iter=20) ...")
    t0 = time.time()
    filled_dct = iterative_gap_fill(
        gapped, 50.0, coordinates='polar', az_boundary='periodic',
        az_res_deg=0.5, init='dct', max_iter=20,
    )
    t_dct = time.time() - t0
    m_dct = compute_metrics(filled_dct, truth, gap_mask, ref_width)
    results.append(('dct_init+DCT', '50', '20', m_dct['mae'],
                     m_dct['dct_std_mean'], m_dct['mapping_error_mean'],
                     m_dct['mapping_error_max'], t_dct))
    print(f"  MAE = {m_dct['mae']:.6f}  time = {t_dct:.2f}s")

    # --- 4. Width x Iteration sweep (linear init) ---
    print()
    print("Running width x iteration sweep (linear init) ...")
    for w in widths:
        for n_iter in iters:
            label = f"linear+DCT w={w} iter={n_iter}"
            t0 = time.time()
            filled = iterative_gap_fill(
                gapped, float(w), coordinates='polar', az_boundary='periodic',
                az_res_deg=0.5, init='linear', max_iter=n_iter,
            )
            t_elapsed = time.time() - t0
            m = compute_metrics(filled, truth, gap_mask, ref_width)
            results.append((
                f'linear+DCT', str(w), str(n_iter),
                m['mae'], m['dct_std_mean'],
                m['mapping_error_mean'], m['mapping_error_max'],
                t_elapsed,
            ))
            print(f"  {label:35s}  MAE={m['mae']:.6f}  "
                  f"std={m['dct_std_mean']:.6f}  "
                  f"me={m['mapping_error_mean']:.4f}  "
                  f"t={t_elapsed:.2f}s")

    # --- Print summary table ---
    print()
    print("=" * 90)
    print(f"{'Method':25s} {'Width':>6s} {'Iter':>5s} "
          f"{'MAE':>10s} {'dct_std':>10s} {'ME_mean':>10s} "
          f"{'ME_max':>10s} {'Time(s)':>8s}")
    print("-" * 90)
    for row in results:
        method, w, it, mae, std, me_mean, me_max, t = row
        print(f"{method:25s} {w:>6s} {it:>5s} "
              f"{mae:10.6f} {std:10.6f} {me_mean:10.4f} "
              f"{me_max:10.4f} {t:8.2f}")
    print("=" * 90)

    # --- Key comparisons ---
    print()
    gd_mae = results[0][3]
    li_mae = results[1][3]
    ms_mae = results[2][3]
    dct_init_mae = results[3][3]
    print("Key comparisons:")
    print(f"  griddata MAE:           {gd_mae:.6f}")
    print(f"  linear init only MAE:   {li_mae:.6f}  "
          f"(ratio vs griddata: {li_mae / gd_mae:.1f}x)")
    print(f"  DCT init (w=50, i=20):  {dct_init_mae:.6f}  "
          f"(ratio vs griddata: {dct_init_mae / gd_mae:.1f}x)")
    print(f"  multiscale cascade MAE: {ms_mae:.6f}  "
          f"(ratio vs griddata: {ms_mae / gd_mae:.1f}x)")

    # Find best linear+DCT within 10 iterations
    best_mae_10 = float('inf')
    best_config = None
    for row in results:
        method, w, it, mae, *_ = row
        if method == 'linear+DCT' and it.isdigit() and int(it) <= 10:
            if mae < best_mae_10:
                best_mae_10 = mae
                best_config = (w, it)

    if best_config:
        print(f"  best linear+DCT (≤10 iter): w={best_config[0]}, "
              f"iter={best_config[1]}, MAE={best_mae_10:.6f}  "
              f"(ratio vs griddata: {best_mae_10 / gd_mae:.1f}x)")

    # --- Experiment 2 Reference (Manual Entry) ---
    print()
    print("Reference: Non-Wrapping Hole Experiment (Separate Run)")
    print("  griddata MAE:           1.133601")
    print("  linear init only MAE:   1.117276  (ratio vs griddata: 0.99x) <-- WINNER")
    print("  DCT (w=50, iter=20) MAE: 1.580311")
    print("  Conclusion: Linear init fails only at periodic boundaries.")


if __name__ == '__main__':
    main()

