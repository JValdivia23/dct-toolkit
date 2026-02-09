"""
Generate benchmark figures for DCT gap filling with linear initialization.

Produces 4 figures (PNG + PDF):
1. Spatial comparison: Truth | Gapped | Linear Init + DCT | Griddata
2. Width impact: MAE vs width for fixed iterations
3. Iteration convergence: MAE vs iterations for selected widths
4. Uncertainty / mapping error: spatial maps inside the hole

Usage
-----
    conda run -n myenv python exp_v3/plot_gap_filling_results.py
"""

import os
import sys
import time

import numpy as np
from scipy.interpolate import griddata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Ensure dct_toolkit is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dct_toolkit'))

from dct_toolkit import dct_mean, dct_std, iterative_gap_fill

EXP_ROOT = os.path.abspath(os.path.dirname(__file__))
FIG_DIR = os.path.join(EXP_ROOT, 'figures')


# ---------------------------------------------------------------------------
# Scenario construction (same as test_width_impact.py)
# ---------------------------------------------------------------------------

def make_smooth_blobs(
    shape: tuple, seed: int = 42,
) -> np.ndarray:
    """
    Generate a smooth field of Gaussian blobs in polar coordinates.

    Parameters
    ----------
    shape : tuple
        Shape of the output array (n_azimuth, n_range).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Smooth field of shape `shape`.
    """
    n_az, n_range = shape
    rng = np.random.RandomState(seed)
    az = np.arange(n_az)
    rg = np.arange(n_range)
    # Map azimuth to radians (0 to 2π)
    theta = az * (2 * np.pi / n_az)
    # Create meshgrid in polar coordinates
    THETA, R = np.meshgrid(theta, rg, indexing='ij')
    # Convert to Cartesian coordinates
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)

    field = np.zeros(shape)
    for _ in range(6):
        c_x = rng.uniform(-n_range, n_range)
        c_y = rng.uniform(-n_range, n_range)
        sigma = rng.uniform(80, 200)
        amp = rng.uniform(0.3, 1.0)
        field += amp * np.exp(
            -((X - c_x) ** 2 + (Y - c_y) ** 2) / (2 * sigma ** 2)
        )
    return field


def make_circular_hole(
    shape: tuple,
    center: tuple = (200, 0),
    radius: float = 100.0,
) -> np.ndarray:
    """
    Create a boolean mask with a circular hole in polar coordinates, using X/Y distance.

    Parameters
    ----------
    shape : tuple
        Shape of the output array (n_azimuth, n_range).
    center : tuple, optional
        Center of the hole in (range, azimuth) index coordinates.
    radius : float, optional
        Radius of the hole in pixels.

    Returns
    -------
    np.ndarray
        Boolean mask: True = valid (outside hole), False = inside hole.
    """
    n_az, n_range = shape
    az = np.arange(n_az)
    rg = np.arange(n_range)
    # Map azimuth to radians (0 to 2π)
    theta = az * (2 * np.pi / n_az)
    # Create meshgrid in polar coordinates
    THETA, R = np.meshgrid(theta, rg, indexing='ij')
    # Convert to Cartesian coordinates
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    # Center in X/Y
    center_r, center_az = center
    center_theta = center_az * (2 * np.pi / n_az)
    center_x = center_r * np.cos(center_theta)
    center_y = center_r * np.sin(center_theta)
    dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    return dist > radius


def fill_griddata_fn(
    truth: np.ndarray, mask: np.ndarray,
) -> np.ndarray:
    n_az, n_range = truth.shape
    az = np.arange(n_az)
    rg = np.arange(n_range)
    AZ, RG = np.meshgrid(az, rg, indexing='ij')

    points = np.column_stack([AZ[mask], RG[mask]])
    values = truth[mask]
    xi = np.column_stack([AZ.ravel(), RG.ravel()])

    filled = griddata(points, values, xi, method='linear')
    filled = filled.reshape(truth.shape)
    nan_mask = np.isnan(filled)
    if np.any(nan_mask):
        filled_nearest = griddata(points, values, xi, method='nearest')
        filled_nearest = filled_nearest.reshape(truth.shape)
        filled[nan_mask] = filled_nearest[nan_mask]
    return filled


def save_fig(fig: plt.Figure, name: str) -> None:
    """Save figure as PNG and PDF."""
    path_png = os.path.join(FIG_DIR, f'{name}.png')
    path_pdf = os.path.join(FIG_DIR, f'{name}.pdf')
    fig.savefig(path_png, dpi=150, bbox_inches='tight')
    fig.savefig(path_pdf, bbox_inches='tight')
    print(f"  Saved: {name}.png, {name}.pdf")


# ---------------------------------------------------------------------------
# Helper functions for polar coordinates
# ---------------------------------------------------------------------------

def create_polar_mesh(shape: tuple) -> tuple:
    """
    Create polar coordinate mesh for plotting.
    
    Returns theta (azimuth in radians) and r (range) meshgrids.
    Azimuth: 0-720 indices mapped to 0-2π radians
    Range: 0-1000 indices mapped to 0-1000 radius
    """
    n_az, n_range = shape
    # Azimuth: 0 to 2π (0 to 360 degrees)
    theta = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    # Range: 0 to max range
    r = np.arange(n_range)
    THETA, R = np.meshgrid(theta, r, indexing='ij')
    return THETA, R


def mark_hole_polar(ax, center_az: float = 0, center_r: float = 200, radius: float = 100) -> None:
    """Mark circular hole boundary on polar plot."""
    # Convert center from indices to polar coordinates
    # center_az is already in azimuth indices (0-720), convert to radians
    theta_center = center_az * (2 * np.pi / 720)
    
    # Create circle in polar coordinates
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    # Circle in Cartesian around center
    x_circle = center_r + radius * np.cos(theta_circle)
    y_circle = center_az + radius * np.sin(theta_circle)
    
    # Convert back to polar (this is approximate for visualization)
    r_circle = np.sqrt(x_circle**2 + y_circle**2)
    theta_circle_plot = np.arctan2(y_circle, x_circle)
    
    ax.plot(theta_circle_plot, r_circle, 'k--', linewidth=1.5)


# ---------------------------------------------------------------------------
# Figure 1: Spatial comparison (Polar)
# ---------------------------------------------------------------------------

def plot_spatial_comparison(
    truth: np.ndarray,
    gapped: np.ndarray,
    filled_dct: np.ndarray,
    filled_gd: np.ndarray,
    gap_mask: np.ndarray,
) -> None:
    """4-panel spatial comparison in polar coordinates: Truth | Gapped | DCT Fill | Griddata."""
    n_az, n_range = truth.shape
    THETA, R = create_polar_mesh((n_az, n_range))
    
    panels = [
        ('Ground Truth', truth),
        ('Gapped Data', gapped),
        ('Linear Init + DCT (w=50, iter=10)', filled_dct),
        ('Griddata (linear)', filled_gd),
    ]

    vmin = np.nanmin(truth)
    vmax = np.nanmax(truth)

    fig = plt.figure(figsize=(16, 12))
    for i, (title, data) in enumerate(panels):
        ax = fig.add_subplot(2, 2, i + 1, projection='polar')
        # Use pcolormesh for polar plot
        im = ax.pcolormesh(THETA, R, data, shading='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_title(title, fontsize=12, pad=20)
        ax.set_ylim(0, n_range)
        ax.set_theta_zero_location('N')  # 0 degrees at top
        ax.set_theta_direction(-1)  # Clockwise
        # Add radial labels
        ax.set_rticks([200, 400, 600, 800])
        ax.set_rlabel_position(22.5)
        plt.colorbar(im, ax=ax, shrink=0.6, pad=0.1)

    fig.suptitle(
        'Gap Filling: Circular Hole Benchmark (Polar View, 720x1000 grid)',
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout()
    save_fig(fig, 'gap_filling_spatial_comparison')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Width impact
# ---------------------------------------------------------------------------

def plot_width_impact(
    gapped: np.ndarray,
    truth: np.ndarray,
    gap_mask: np.ndarray,
    mae_griddata: float,
) -> None:
    """MAE vs width for fixed iteration counts."""
    widths = [3, 5, 10, 20, 50, 75, 100]
    iter_counts = [1, 5, 10, 20]

    fig, ax = plt.subplots(figsize=(10, 6))

    for n_iter in iter_counts:
        maes = []
        for w in widths:
            filled = iterative_gap_fill(
                gapped, float(w), init='linear', max_iter=n_iter,
            )
            mae = np.mean(np.abs(filled[gap_mask] - truth[gap_mask]))
            maes.append(mae)
        ax.plot(widths, maes, 'o-', label=f'iter={n_iter}', linewidth=2)

    ax.axhline(
        mae_griddata, color='gray', linestyle='--', linewidth=1.5,
        label=f'griddata (MAE={mae_griddata:.4f})',
    )

    # Linear init only (0 iter)
    filled_li = iterative_gap_fill(
        gapped, 5.0, init='linear', max_iter=0,
    )
    mae_li = np.mean(np.abs(filled_li[gap_mask] - truth[gap_mask]))
    ax.axhline(
        mae_li, color='orange', linestyle=':', linewidth=1.5,
        label=f'linear init only (MAE={mae_li:.4f})',
    )

    ax.set_xlabel('Smoothing Width (pixels)', fontsize=12)
    ax.set_ylabel('MAE in Gap Region', fontsize=12)
    ax.set_title(
        'Effect of Smoothing Width on Gap Fill Accuracy',
        fontsize=13, fontweight='bold',
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])

    plt.tight_layout()
    save_fig(fig, 'gap_filling_width_impact')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Iteration convergence
# ---------------------------------------------------------------------------

def plot_iteration_convergence(
    gapped: np.ndarray,
    truth: np.ndarray,
    gap_mask: np.ndarray,
    mae_griddata: float,
) -> None:
    """MAE vs iteration count for selected widths."""
    iters = [0, 1, 2, 3, 5, 10, 20, 50]
    selected_widths = [5, 20, 50]

    fig, ax = plt.subplots(figsize=(10, 6))

    for w in selected_widths:
        maes = []
        for n_iter in iters:
            filled = iterative_gap_fill(
                gapped, float(w), init='linear', max_iter=n_iter,
            )
            mae = np.mean(np.abs(filled[gap_mask] - truth[gap_mask]))
            maes.append(mae)
        ax.plot(iters, maes, 'o-', label=f'w={w}', linewidth=2)

    ax.axhline(
        mae_griddata, color='gray', linestyle='--', linewidth=1.5,
        label=f'griddata (MAE={mae_griddata:.4f})',
    )

    ax.set_xlabel('Number of DCT Iterations', fontsize=12)
    ax.set_ylabel('MAE in Gap Region', fontsize=12)
    ax.set_title(
        'Iteration Convergence from Linear Initialization',
        fontsize=13, fontweight='bold',
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 'gap_filling_iteration_convergence')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Uncertainty and mapping error (Polar)
# ---------------------------------------------------------------------------

def plot_uncertainty_maps(
    filled_dct: np.ndarray,
    gap_mask: np.ndarray,
    ref_width: float = 5.0,
) -> None:
    """Spatial maps of dct_std and mapping error in polar coordinates."""
    n_az, n_range = filled_dct.shape
    THETA, R = create_polar_mesh((n_az, n_range))

    # dct_std of filled field
    std_field = dct_std(
        filled_dct, ref_width,
        mask=np.ones_like(filled_dct, dtype=bool),
    )

    # Mapping error = 1 - dct_mean(indicator, width)
    indicator = (~gap_mask).astype(float)
    density = dct_mean(
        indicator, ref_width,
        mask=np.ones_like(indicator, dtype=bool),
    )
    mapping_error = np.clip(1.0 - density, 0.0, 1.0)

    fig = plt.figure(figsize=(16, 7))

    # dct_std
    ax0 = fig.add_subplot(1, 2, 1, projection='polar')
    im0 = ax0.pcolormesh(THETA, R, std_field, shading='auto', cmap='viridis')
    ax0.set_title(f'dct_std (width={ref_width})', fontsize=12, pad=20)
    ax0.set_ylim(0, n_range)
    ax0.set_theta_zero_location('N')
    ax0.set_theta_direction(-1)
    ax0.set_rticks([200, 400, 600, 800])
    ax0.set_rlabel_position(22.5)
    plt.colorbar(im0, ax=ax0, shrink=0.6, pad=0.1)

    # Mapping error
    ax1 = fig.add_subplot(1, 2, 2, projection='polar')
    im1 = ax1.pcolormesh(THETA, R, mapping_error, shading='auto', cmap='Reds', vmin=0, vmax=1)
    ax1.set_title(f'Mapping Error (width={ref_width})', fontsize=12, pad=20)
    ax1.set_ylim(0, n_range)
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_rticks([200, 400, 600, 800])
    ax1.set_rlabel_position(22.5)
    plt.colorbar(im1, ax=ax1, shrink=0.6, pad=0.1)

    fig.suptitle(
        'Uncertainty Metrics for Gap-Filled Field (Polar View)',
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout()
    save_fig(fig, 'gap_filling_uncertainty_maps')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 60)
    print("Generating Gap Filling Figures")
    print("=" * 60)

    shape = (720, 1000)
    truth = make_smooth_blobs(shape)
    mask = make_circular_hole(shape, center=(200, 0), radius=100)
    gap_mask = ~mask
    gapped = truth.copy()
    gapped[gap_mask] = np.nan

    # Compute fills
    print("Computing griddata fill ...")
    t0 = time.time()
    filled_gd = fill_griddata_fn(truth, mask)
    print(f"  Done in {time.time() - t0:.1f}s")

    print("Computing DCT fill (linear init, w=50, iter=10) ...")
    t0 = time.time()
    filled_dct = iterative_gap_fill(
        gapped, 50.0, init='linear', max_iter=10,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    mae_gd = np.mean(np.abs(filled_gd[gap_mask] - truth[gap_mask]))
    mae_dct = np.mean(np.abs(filled_dct[gap_mask] - truth[gap_mask]))
    print(f"  Griddata MAE: {mae_gd:.6f}")
    print(f"  DCT MAE:      {mae_dct:.6f}")
    print()

    # Figure 1: Spatial comparison
    print("Figure 1: Spatial comparison ...")
    plot_spatial_comparison(truth, gapped, filled_dct, filled_gd, gap_mask)

    # Figure 2: Width impact
    print("Figure 2: Width impact ...")
    plot_width_impact(gapped, truth, gap_mask, mae_gd)

    # Figure 3: Iteration convergence
    print("Figure 3: Iteration convergence ...")
    plot_iteration_convergence(gapped, truth, gap_mask, mae_gd)

    # Figure 4: Uncertainty maps
    print("Figure 4: Uncertainty maps ...")
    plot_uncertainty_maps(filled_dct, gap_mask, ref_width=5.0)

    print()
    print("All figures saved to exp_v3/figures/")
    print("=" * 60)


if __name__ == '__main__':
    main()
