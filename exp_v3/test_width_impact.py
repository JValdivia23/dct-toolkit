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

def make_smooth_blobs(shape: tuple, seed: int = 42) -> np.ndarray:
    """
    Create a smooth 2-D field of Gaussian blobs on a (n_az, n_range) grid.

    Parameters
    ----------
    shape : tuple
        (n_azimuth, n_range).
    seed : int
        Random seed for blob placement.

    Returns
    -------
    field : np.ndarray
        Smooth field with values roughly in [0, 1].
    """
    n_az, n_range = shape
    rng = np.random.RandomState(seed)

    az = np.arange(n_az)
    rg = np.arange(n_range)
    AZ, RG = np.meshgrid(az, rg, indexing='ij')

    field = np.zeros(shape)
    n_blobs = 6
    for _ in range(n_blobs):
        c_az = rng.uniform(0, n_az)
        c_rg = rng.uniform(0, n_range)
        sigma = rng.uniform(80, 200)
        amp = rng.uniform(0.3, 1.0)
        field += amp * np.exp(
            -((AZ - c_az) ** 2 + (RG - c_rg) ** 2) / (2 * sigma ** 2)
        )
    return field


def make_circular_hole(
    shape: tuple,
    center: tuple = (200, 0),
    radius: float = 100.0,
) -> np.ndarray:
    """
    Create a boolean mask with a circular hole (True = valid).

    Parameters
    ----------
    shape : tuple
        (n_azimuth, n_range).
    center : tuple
        (range_center, azimuth_center) of hole in pixel coordinates.
        Convention: range is axis=1, azimuth is axis=0.
    radius : float
        Hole radius in pixels.

    Returns
    -------
    mask : np.ndarray of bool
        True where data is valid, False inside the hole.
    """
    n_az, n_range = shape
    az = np.arange(n_az)
    rg = np.arange(n_range)
    AZ, RG = np.meshgrid(az, rg, indexing='ij')

    dist = np.sqrt((RG - center[0]) ** 2 + (AZ - center[1]) ** 2)
    mask = dist > radius
    return mask


def fill_griddata(truth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Fill gaps using scipy griddata (linear + nearest fallback).

    Parameters
    ----------
    truth : np.ndarray
        Ground truth field (used only to extract valid values).
    mask : np.ndarray
        Boolean mask (True = valid).

    Returns
    -------
    filled : np.ndarray
    """
    n_az, n_range = truth.shape
    az = np.arange(n_az)
    rg = np.arange(n_range)
    AZ, RG = np.meshgrid(az, rg, indexing='ij')

    points = np.column_stack([AZ[mask], RG[mask]])
    values = truth[mask]
    xi = np.column_stack([AZ.ravel(), RG.ravel()])

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
        filled, ref_width,
        mask=np.ones_like(filled, dtype=bool),
    )
    dct_std_mean = np.mean(std_field[gap_mask])

    # Mapping error = 1 - dct_mean(indicator, width)
    indicator = (~gap_mask).astype(float)
    density = dct_mean(
        indicator, ref_width,
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

    # Build scenario
    truth = make_smooth_blobs(shape)
    mask = make_circular_hole(shape, center=(200, 0), radius=100)
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
    filled_gd = fill_griddata(truth, mask)
    t_gd = time.time() - t0
    m_gd = compute_metrics(filled_gd, truth, gap_mask, ref_width)
    results.append(('griddata', '-', '-', m_gd['mae'], m_gd['dct_std_mean'],
                     m_gd['mapping_error_mean'], m_gd['mapping_error_max'],
                     t_gd))
    print(f"  MAE = {m_gd['mae']:.6f}  time = {t_gd:.2f}s")

    # --- 2. Linear init only (0 iterations) ---
    print("Running: linear_init_only (0 DCT iterations) ...")
    t0 = time.time()
    filled_li = iterative_gap_fill(gapped, ref_width, init='linear',
                                   max_iter=0)
    t_li = time.time() - t0
    m_li = compute_metrics(filled_li, truth, gap_mask, ref_width)
    results.append(('linear_init_only', '-', '0', m_li['mae'],
                     m_li['dct_std_mean'], m_li['mapping_error_mean'],
                     m_li['mapping_error_max'], t_li))
    print(f"  MAE = {m_li['mae']:.6f}  time = {t_li:.2f}s")

    # --- 3. Legacy multi-scale cascade (width=5, iter=50) ---
    print("Running: multiscale cascade (width=5, iter=50) ...")
    t0 = time.time()
    filled_ms = iterative_gap_fill(gapped, 5.0, init='multiscale',
                                   max_iter=50)
    t_ms = time.time() - t0
    m_ms = compute_metrics(filled_ms, truth, gap_mask, ref_width)
    results.append(('multiscale_cascade', '5', '50', m_ms['mae'],
                     m_ms['dct_std_mean'], m_ms['mapping_error_mean'],
                     m_ms['mapping_error_max'], t_ms))
    print(f"  MAE = {m_ms['mae']:.6f}  time = {t_ms:.2f}s")

    # --- 4. Width x Iteration sweep (linear init) ---
    print()
    print("Running width x iteration sweep (linear init) ...")
    for w in widths:
        for n_iter in iters:
            label = f"linear+DCT w={w} iter={n_iter}"
            t0 = time.time()
            filled = iterative_gap_fill(gapped, float(w), init='linear',
                                        max_iter=n_iter)
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
    print("Key comparisons:")
    print(f"  griddata MAE:           {gd_mae:.6f}")
    print(f"  linear init only MAE:   {li_mae:.6f}  "
          f"(ratio vs griddata: {li_mae / gd_mae:.1f}x)")
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


if __name__ == '__main__':
    main()
