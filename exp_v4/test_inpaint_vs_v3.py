"""
Benchmark: dct_inpaint (v4) vs iterative_gap_fill (v3) vs griddata.

Reproduces the exp_v3 benchmark scenario (720x1000 polar grid, circular hole)
and adds dct_inpaint with various width/order configurations.

Usage
-----
    conda activate myenv
    export PYTHONPATH=$PYTHONPATH:$(pwd)/dct_toolkit
    python exp_v4/test_inpaint_vs_v3.py
"""

import sys
import os
import time

import numpy as np

# Ensure dct_toolkit is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dct_toolkit"))

from dct_toolkit import iterative_gap_fill, dct_inpaint

# Reuse v3 scenario construction
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp_v3"))
from test_width_impact import (
    make_cartesian_ground_truth,
    make_circular_hole,
    fill_griddata,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_gap_metrics(
    filled: np.ndarray,
    truth: np.ndarray,
    gap_mask: np.ndarray,
) -> dict:
    """
    Compute accuracy metrics in the gap region.

    Parameters
    ----------
    filled : np.ndarray
        Gap-filled field.
    truth : np.ndarray
        Ground truth.
    gap_mask : np.ndarray of bool
        True where data was missing.

    Returns
    -------
    dict with keys: 'mae', 'rmse', 'max_error'.
    """
    err = filled[gap_mask] - truth[gap_mask]
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_error": float(np.max(np.abs(err))),
    }


# ---------------------------------------------------------------------------
# Experiment 1: Wrapping hole (azimuth=0, reproducing v3)
# ---------------------------------------------------------------------------


def experiment_wrapping_hole():
    """Circular hole centred at range=200, azimuth=0 (wraps around 360/0)."""
    print("=" * 90)
    print("EXPERIMENT 1: Wrapping Hole (center=(200,0), radius=100)")
    print("=" * 90)

    shape = (720, 1000)
    truth = make_cartesian_ground_truth(shape, range_res_m=100.0, az_res_deg=0.5)
    mask = make_circular_hole(
        shape,
        center=(200, 0),
        radius=100,
        range_res_m=100.0,
        az_res_deg=0.5,
    )
    gap_mask = ~mask
    n_gap = np.sum(gap_mask)

    gapped = truth.copy()
    gapped[gap_mask] = np.nan
    print(f"Gap pixels: {n_gap} ({100 * n_gap / truth.size:.1f}%)")
    print()

    results = []

    # --- Griddata baseline ---
    print("Running: griddata (linear + nearest) ...")
    t0 = time.time()
    filled_gd = fill_griddata(truth, mask, range_res_m=100.0, az_res_deg=0.5)
    t_gd = time.time() - t0
    m_gd = compute_gap_metrics(filled_gd, truth, gap_mask)
    results.append(("griddata", "-", "-", m_gd, t_gd))
    print(f"  MAE = {m_gd['mae']:.6f}  RMSE = {m_gd['rmse']:.6f}  time = {t_gd:.2f}s")

    # --- v3: iterative_gap_fill (best configs from exp_v3 report) ---
    for init_mode, w, n_iter in [
        ("linear", 50, 20),
        ("dct", 50, 20),
        ("linear", 50, 100),
    ]:
        label = f"v3 ({init_mode}, w={w}, i={n_iter})"
        print(f"Running: {label} ...")
        t0 = time.time()
        filled = iterative_gap_fill(
            gapped,
            float(w),
            coordinates="polar",
            az_boundary="periodic",
            az_res_deg=0.5,
            init=init_mode,
            max_iter=n_iter,
        )
        elapsed = time.time() - t0
        m = compute_gap_metrics(filled, truth, gap_mask)
        results.append((label, str(w), str(n_iter), m, elapsed))
        print(f"  MAE = {m['mae']:.6f}  RMSE = {m['rmse']:.6f}  time = {elapsed:.2f}s")

    # --- v4: dct_inpaint (various configurations) ---
    for w, order in [(10, 2), (20, 2), (50, 2), (100, 2), (50, 1), (20, 1)]:
        label = f"v4 inpaint (w={w}, p={order})"
        print(f"Running: {label} ...")
        t0 = time.time()
        filled = dct_inpaint(
            gapped,
            float(w),
            coordinates="polar",
            az_boundary="periodic",
            az_res_deg=0.5,
            order=order,
            max_iter=100,
            tol=1e-6,
        )
        elapsed = time.time() - t0
        m = compute_gap_metrics(filled, truth, gap_mask)
        results.append((label, str(w), str(order), m, elapsed))
        print(f"  MAE = {m['mae']:.6f}  RMSE = {m['rmse']:.6f}  time = {elapsed:.2f}s")

    # --- Summary Table ---
    print()
    print("=" * 90)
    print(f"{'Method':40s} {'MAE':>10s} {'RMSE':>10s} {'MaxErr':>10s} {'Time(s)':>8s}")
    print("-" * 90)
    for label, _, _, m, t in results:
        print(
            f"{label:40s} {m['mae']:10.6f} {m['rmse']:10.6f} "
            f"{m['max_error']:10.4f} {t:8.2f}"
        )
    print("=" * 90)

    gd_mae = results[0][3]["mae"]
    print()
    print("Ratios vs griddata:")
    for label, _, _, m, _ in results[1:]:
        ratio = m["mae"] / gd_mae
        print(f"  {label:40s}  {ratio:.2f}x")

    return results


# ---------------------------------------------------------------------------
# Experiment 2: Non-wrapping hole (centered at azimuth=180)
# ---------------------------------------------------------------------------


def experiment_centered_hole():
    """Circular hole centred at range=200, azimuth=360 (interior, no wrapping)."""
    print()
    print("=" * 90)
    print("EXPERIMENT 2: Centered Hole (center=(200,360), radius=100)")
    print("=" * 90)

    shape = (720, 1000)
    truth = make_cartesian_ground_truth(shape, range_res_m=100.0, az_res_deg=0.5)
    mask = make_circular_hole(
        shape,
        center=(200, 360),
        radius=100,
        range_res_m=100.0,
        az_res_deg=0.5,
    )
    gap_mask = ~mask
    n_gap = np.sum(gap_mask)

    gapped = truth.copy()
    gapped[gap_mask] = np.nan
    print(f"Gap pixels: {n_gap} ({100 * n_gap / truth.size:.1f}%)")
    print()

    results = []

    # Griddata
    print("Running: griddata ...")
    t0 = time.time()
    filled_gd = fill_griddata(truth, mask, range_res_m=100.0, az_res_deg=0.5)
    t_gd = time.time() - t0
    m_gd = compute_gap_metrics(filled_gd, truth, gap_mask)
    results.append(("griddata", m_gd, t_gd))
    print(f"  MAE = {m_gd['mae']:.6f}  time = {t_gd:.2f}s")

    # v3 best
    print("Running: v3 (linear, w=50, i=20) ...")
    t0 = time.time()
    filled_v3 = iterative_gap_fill(
        gapped,
        50.0,
        coordinates="polar",
        az_boundary="periodic",
        az_res_deg=0.5,
        init="linear",
        max_iter=20,
    )
    t_v3 = time.time() - t0
    m_v3 = compute_gap_metrics(filled_v3, truth, gap_mask)
    results.append(("v3 (linear, w=50, i=20)", m_v3, t_v3))
    print(f"  MAE = {m_v3['mae']:.6f}  time = {t_v3:.2f}s")

    # Linear init only
    print("Running: linear init only ...")
    t0 = time.time()
    filled_li = iterative_gap_fill(
        gapped,
        5.0,
        coordinates="polar",
        az_boundary="periodic",
        az_res_deg=0.5,
        init="linear",
        max_iter=0,
    )
    t_li = time.time() - t0
    m_li = compute_gap_metrics(filled_li, truth, gap_mask)
    results.append(("linear init only", m_li, t_li))
    print(f"  MAE = {m_li['mae']:.6f}  time = {t_li:.2f}s")

    # v4 inpaint
    for w in [10, 20, 50]:
        label = f"v4 inpaint (w={w}, p=2)"
        print(f"Running: {label} ...")
        t0 = time.time()
        filled = dct_inpaint(
            gapped,
            float(w),
            coordinates="polar",
            az_boundary="periodic",
            az_res_deg=0.5,
            order=2,
            max_iter=100,
            tol=1e-6,
        )
        elapsed = time.time() - t0
        m = compute_gap_metrics(filled, truth, gap_mask)
        results.append((label, m, elapsed))
        print(f"  MAE = {m['mae']:.6f}  time = {elapsed:.2f}s")

    # Summary
    print()
    print(f"{'Method':40s} {'MAE':>10s} {'RMSE':>10s} {'Time(s)':>8s}")
    print("-" * 70)
    for row in results:
        label, m, t = row
        print(f"{label:40s} {m['mae']:10.6f} {m['rmse']:10.6f} {t:8.2f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print()
    print("DCT Spectral Inpainting Benchmark (v4 vs v3)")
    print("Based on Garcia (2010) DCT-PLS with width parameterisation")
    print()

    r1 = experiment_wrapping_hole()
    r2 = experiment_centered_hole()

    print()
    print("=" * 90)
    print("CONCLUSION")
    print("=" * 90)
    gd_wrap = r1[0][3]["mae"]
    best_v3_wrap = min(r[3]["mae"] for r in r1[1:4])
    best_v4_wrap = min(r[3]["mae"] for r in r1[4:])
    print(
        f"Wrapping hole:   griddata={gd_wrap:.4f}  "
        f"best_v3={best_v3_wrap:.4f}  best_v4={best_v4_wrap:.4f}"
    )
    print(f"  v4 improvement over v3: {best_v3_wrap / best_v4_wrap:.1f}x")
    print(f"  v4 vs griddata:         {best_v4_wrap / gd_wrap:.2f}x")


if __name__ == "__main__":
    main()
