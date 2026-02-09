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
    n_az, n_range = shape
    rng = np.random.RandomState(seed)
    az = np.arange(n_az)
    rg = np.arange(n_range)
    AZ, RG = np.meshgrid(az, rg, indexing='ij')

    field = np.zeros(shape)
    for _ in range(6):
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
    n_az, n_range = shape
    az = np.arange(n_az)
    rg = np.arange(n_range)
    AZ, RG = np.meshgrid(az, rg, indexing='ij')
    dist = np.sqrt((RG - center[0]) ** 2 + (AZ - center[1]) ** 2)
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
    for ext in ['png', 'pdf']:
        path = os.path.join(FIG_DIR, f'{name}.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {name}.png, {name}.pdf")


# ---------------------------------------------------------------------------
# Figure 1: Spatial comparison
# ---------------------------------------------------------------------------

def plot_spatial_comparison(
    truth: np.ndarray,
    gapped: np.ndarray,
    filled_dct: np.ndarray,
    filled_gd: np.ndarray,
    gap_mask: np.ndarray,
) -> None:
    """4-panel spatial comparison: Truth | Gapped | DCT Fill | Griddata."""
    # Zoom into the hole region for better visibility
    az_slice = slice(0, 200)
    rg_slice = slice(0, 350)

    panels = [
        ('Ground Truth', truth),
        ('Gapped Data', gapped),
        ('Linear Init + DCT (w=50, iter=10)', filled_dct),
        ('Griddata (linear)', filled_gd),
    ]

    vmin = np.nanmin(truth[az_slice, rg_slice])
    vmax = np.nanmax(truth[az_slice, rg_slice])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (title, data) in zip(axes.ravel(), panels):
        im = ax.imshow(
            data[az_slice, rg_slice].T,
            aspect='auto', origin='lower',
            vmin=vmin, vmax=vmax, cmap='RdBu_r',
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Azimuth index')
        ax.set_ylabel('Range index')
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Mark hole boundary on all panels
    for ax in axes.ravel():
        circle = plt.Circle(
            (0, 200), 100,
            fill=False, color='k', linestyle='--', linewidth=1.5,
        )
        ax.add_patch(circle)

    fig.suptitle(
        'Gap Filling: Circular Hole Benchmark (720x1000 grid)',
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
# Figure 4: Uncertainty and mapping error
# ---------------------------------------------------------------------------

def plot_uncertainty_maps(
    filled_dct: np.ndarray,
    gap_mask: np.ndarray,
    ref_width: float = 5.0,
) -> None:
    """Spatial maps of dct_std and mapping error inside the hole."""
    az_slice = slice(0, 200)
    rg_slice = slice(0, 350)

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

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # dct_std
    im0 = axes[0].imshow(
        std_field[az_slice, rg_slice].T,
        aspect='auto', origin='lower', cmap='viridis',
    )
    axes[0].set_title(f'dct_std (width={ref_width})', fontsize=12)
    axes[0].set_xlabel('Azimuth index')
    axes[0].set_ylabel('Range index')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Mapping error
    im1 = axes[1].imshow(
        mapping_error[az_slice, rg_slice].T,
        aspect='auto', origin='lower', cmap='Reds',
        vmin=0, vmax=1,
    )
    axes[1].set_title(f'Mapping Error (width={ref_width})', fontsize=12)
    axes[1].set_xlabel('Azimuth index')
    axes[1].set_ylabel('Range index')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # Mark hole boundary
    for ax in axes:
        circle = plt.Circle(
            (0, 200), 100,
            fill=False, color='k', linestyle='--', linewidth=1.5,
        )
        ax.add_patch(circle)

    fig.suptitle(
        'Uncertainty Metrics for Gap-Filled Field',
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
