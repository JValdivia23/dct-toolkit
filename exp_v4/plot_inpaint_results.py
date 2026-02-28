"""
Generate benchmark figures for DCT spectral inpainting (v4).

Produces 5 figures (PNG + PDF):
1. Spatial comparison (polar): Truth | Gapped | v3 | v4 | Griddata
2. Error maps (polar): |v3 - truth| vs |v4 - truth| inside gap
3. Width impact: MAE vs width for v4 (p=1 and p=2) with baselines
4. Convergence: MAE vs iteration for v4 (selected widths)
5. Method bar chart: MAE comparison across all methods

Usage
-----
    conda activate myenv
    export PYTHONPATH=$PYTHONPATH:$(pwd)/dct_toolkit
    python exp_v4/plot_inpaint_results.py
"""

import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Ensure dct_toolkit is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dct_toolkit"))

from dct_toolkit import iterative_gap_fill, dct_inpaint

# Import scenario functions from exp_v3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp_v3"))
from test_width_impact import (
    make_cartesian_ground_truth,
    make_circular_hole,
    fill_griddata,
)

EXP_ROOT = os.path.abspath(os.path.dirname(__file__))
FIG_DIR = os.path.join(EXP_ROOT, "figures")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_polar_mesh(shape: tuple) -> tuple:
    """
    Create polar coordinate mesh for plotting.

    Parameters
    ----------
    shape : tuple
        (n_azimuth, n_range).

    Returns
    -------
    THETA : np.ndarray
        Azimuth mesh in radians.
    R : np.ndarray
        Range mesh.
    """
    n_az, n_range = shape
    theta = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    r = np.arange(n_range)
    THETA, R = np.meshgrid(theta, r, indexing="ij")
    return THETA, R


def save_fig(fig: plt.Figure, name: str) -> None:
    """Save figure as PNG and PDF."""
    path_png = os.path.join(FIG_DIR, f"{name}.png")
    path_pdf = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.savefig(path_png, dpi=150, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    print(f"  Saved: {name}.png, {name}.pdf")


# ---------------------------------------------------------------------------
# Figure 1: Spatial comparison (5 panels, polar)
# ---------------------------------------------------------------------------


def plot_spatial_comparison(
    truth: np.ndarray,
    gapped: np.ndarray,
    filled_v3: np.ndarray,
    filled_v4: np.ndarray,
    filled_gd: np.ndarray,
    gap_mask: np.ndarray,
    mae_v3: float,
    mae_v4: float,
    mae_gd: float,
) -> None:
    """5-panel polar comparison: Truth | Gapped | v3 | v4 | Griddata."""
    n_az, n_range = truth.shape
    THETA, R = create_polar_mesh((n_az, n_range))

    vmin = np.nanmin(truth)
    vmax = np.nanmax(truth)

    panels = [
        ("Ground Truth", truth),
        ("Gapped Data", gapped),
        (f"v3: iterative_gap_fill\nMAE={mae_v3:.4f}", filled_v3),
        (f"v4: dct_inpaint (w=50, p=2)\nMAE={mae_v4:.4f}", filled_v4),
        (f"Griddata (linear)\nMAE={mae_gd:.4f}", filled_gd),
    ]

    fig = plt.figure(figsize=(20, 8))
    for i, (title, data) in enumerate(panels):
        ax = fig.add_subplot(1, 5, i + 1, projection="polar")
        im = ax.pcolormesh(
            THETA,
            R,
            data,
            shading="auto",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        ax.set_title(title, fontsize=10, pad=18)
        ax.set_ylim(0, n_range)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rticks([200, 400, 600, 800])
        ax.set_rlabel_position(22.5)
        ax.tick_params(labelsize=7)

    # Single colorbar on the right
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    fig.suptitle(
        "Spectral Inpainting Benchmark: Wrapping Hole (720x1000 polar grid)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout(rect=[0, 0, 0.92, 1.0])
    save_fig(fig, "inpaint_spatial_comparison")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Error maps (v3 vs v4, polar)
# ---------------------------------------------------------------------------


def plot_error_maps(
    filled_v3: np.ndarray,
    filled_v4: np.ndarray,
    filled_gd: np.ndarray,
    truth: np.ndarray,
    gap_mask: np.ndarray,
) -> None:
    """Side-by-side absolute error maps inside the gap region."""
    n_az, n_range = truth.shape
    THETA, R = create_polar_mesh((n_az, n_range))

    err_v3 = np.full_like(truth, np.nan)
    err_v4 = np.full_like(truth, np.nan)
    err_gd = np.full_like(truth, np.nan)
    err_v3[gap_mask] = np.abs(filled_v3[gap_mask] - truth[gap_mask])
    err_v4[gap_mask] = np.abs(filled_v4[gap_mask] - truth[gap_mask])
    err_gd[gap_mask] = np.abs(filled_gd[gap_mask] - truth[gap_mask])

    # Shared color scale
    emax = max(np.nanmax(err_v3), np.nanmax(err_v4), np.nanmax(err_gd))

    panels = [
        ("v3: iterative_gap_fill", err_v3),
        ("v4: dct_inpaint", err_v4),
        ("griddata", err_gd),
    ]

    fig = plt.figure(figsize=(16, 6))
    for i, (title, err) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1, projection="polar")
        im = ax.pcolormesh(
            THETA,
            R,
            err,
            shading="auto",
            cmap="hot_r",
            vmin=0,
            vmax=emax,
            rasterized=True,
        )
        mae = np.nanmean(err[gap_mask])
        ax.set_title(f"{title}\nMAE={mae:.4f}", fontsize=11, pad=18)
        ax.set_ylim(0, n_range)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rticks([200, 400, 600, 800])
        ax.set_rlabel_position(22.5)
        ax.tick_params(labelsize=8)

    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="|error|")

    fig.suptitle(
        "Absolute Error in Gap Region (wrapping hole, r=100)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout(rect=[0, 0, 0.90, 1.0])
    save_fig(fig, "inpaint_error_maps")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Width impact (v4 MAE vs width for p=1 and p=2)
# ---------------------------------------------------------------------------


def plot_width_impact(
    gapped: np.ndarray,
    truth: np.ndarray,
    gap_mask: np.ndarray,
    mae_gd: float,
    mae_v3: float,
) -> None:
    """MAE vs width for v4 inpaint with order=1 and order=2, plus baselines."""
    widths = [5, 10, 20, 50, 75, 100, 150]

    fig, ax = plt.subplots(figsize=(10, 6))

    for order, marker, color in [(2, "o", "#2196F3"), (1, "s", "#FF9800")]:
        maes = []
        for w in widths:
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
            mae = np.mean(np.abs(filled[gap_mask] - truth[gap_mask]))
            maes.append(mae)
        ax.plot(
            widths,
            maes,
            f"{marker}-",
            color=color,
            label=f"v4 dct_inpaint (order={order})",
            linewidth=2,
            markersize=7,
        )

    ax.axhline(
        mae_gd,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"griddata (MAE={mae_gd:.4f})",
    )
    ax.axhline(
        mae_v3,
        color="#E91E63",
        linestyle=":",
        linewidth=1.5,
        label=f"v3 best (MAE={mae_v3:.4f})",
    )

    ax.set_xlabel("Smoothing Width (pixels)", fontsize=12)
    ax.set_ylabel("MAE in Gap Region", fontsize=12)
    ax.set_title(
        "v4 Inpainting Accuracy vs Width (wrapping hole benchmark)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])

    plt.tight_layout()
    save_fig(fig, "inpaint_width_impact")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Convergence (MAE vs iteration for selected widths)
# ---------------------------------------------------------------------------


def plot_convergence(
    gapped: np.ndarray,
    truth: np.ndarray,
    gap_mask: np.ndarray,
    mae_gd: float,
) -> None:
    """MAE vs max_iter for v4 inpaint at selected widths."""
    iters_list = [1, 2, 3, 5, 10, 20, 50, 100]
    selected_widths = [10, 20, 50, 100]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for w, color in zip(selected_widths, colors):
        maes = []
        for n_iter in iters_list:
            filled = dct_inpaint(
                gapped,
                float(w),
                coordinates="polar",
                az_boundary="periodic",
                az_res_deg=0.5,
                order=2,
                max_iter=n_iter,
                tol=1e-10,
            )
            mae = np.mean(np.abs(filled[gap_mask] - truth[gap_mask]))
            maes.append(mae)
        ax.plot(
            iters_list,
            maes,
            "o-",
            color=color,
            label=f"w={w}",
            linewidth=2,
            markersize=6,
        )

    ax.axhline(
        mae_gd,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"griddata (MAE={mae_gd:.4f})",
    )

    ax.set_xlabel("Max Iterations", fontsize=12)
    ax.set_ylabel("MAE in Gap Region", fontsize=12)
    ax.set_title(
        "v4 Convergence: dct_inpaint (order=2) from Linear Init",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "inpaint_convergence")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Method bar chart
# ---------------------------------------------------------------------------


def plot_method_barchart(results: list) -> None:
    """Horizontal bar chart comparing MAE across methods."""
    labels = [r[0] for r in results]
    maes = [r[1] for r in results]

    # Color: best = green, worst = red, others = blue
    sorted_maes = sorted(maes)
    colors = []
    for m in maes:
        if m == sorted_maes[0]:
            colors.append("#4CAF50")
        elif m == sorted_maes[-1]:
            colors.append("#F44336")
        else:
            colors.append("#2196F3")

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, maes, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("MAE in Gap Region", fontsize=12)
    ax.set_title(
        "Method Comparison: Wrapping Hole Benchmark",
        fontsize=13,
        fontweight="bold",
    )

    # Add value labels on bars
    for bar, mae in zip(bars, maes):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{mae:.4f}",
            va="center",
            fontsize=9,
        )

    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "inpaint_method_comparison")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 70)
    print("Generating v4 Inpainting Figures")
    print("=" * 70)
    print()

    # --- Build scenario ---
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
    gapped = truth.copy()
    gapped[gap_mask] = np.nan
    n_gap = np.sum(gap_mask)
    print(f"Scenario: 720x1000 polar grid, wrapping hole (r=100)")
    print(f"Gap pixels: {n_gap} ({100 * n_gap / truth.size:.1f}%)")
    print()

    # --- Compute fills ---
    print("Computing griddata fill ...")
    t0 = time.time()
    filled_gd = fill_griddata(truth, mask, range_res_m=100.0, az_res_deg=0.5)
    print(f"  Done in {time.time() - t0:.1f}s")
    mae_gd = float(np.mean(np.abs(filled_gd[gap_mask] - truth[gap_mask])))

    print("Computing v3 fill (linear init, w=50, iter=20) ...")
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
    print(f"  Done in {time.time() - t0:.1f}s")
    mae_v3 = float(np.mean(np.abs(filled_v3[gap_mask] - truth[gap_mask])))

    print("Computing v4 fill (w=50, p=2, max_iter=100) ...")
    t0 = time.time()
    filled_v4 = dct_inpaint(
        gapped,
        50.0,
        coordinates="polar",
        az_boundary="periodic",
        az_res_deg=0.5,
        order=2,
        max_iter=100,
        tol=1e-6,
    )
    print(f"  Done in {time.time() - t0:.1f}s")
    mae_v4 = float(np.mean(np.abs(filled_v4[gap_mask] - truth[gap_mask])))

    print()
    print(f"  griddata MAE: {mae_gd:.6f}")
    print(f"  v3 MAE:       {mae_v3:.6f}")
    print(f"  v4 MAE:       {mae_v4:.6f}")
    print(f"  v4 improvement over v3: {mae_v3 / mae_v4:.1f}x")
    print(f"  v4 improvement over griddata: {mae_gd / mae_v4:.1f}x")
    print()

    # --- Figure 1: Spatial comparison ---
    print("Figure 1: Spatial comparison ...")
    plot_spatial_comparison(
        truth,
        gapped,
        filled_v3,
        filled_v4,
        filled_gd,
        gap_mask,
        mae_v3,
        mae_v4,
        mae_gd,
    )

    # --- Figure 2: Error maps ---
    print("Figure 2: Error maps ...")
    plot_error_maps(filled_v3, filled_v4, filled_gd, truth, gap_mask)

    # --- Figure 3: Width impact ---
    print("Figure 3: Width impact ...")
    plot_width_impact(gapped, truth, gap_mask, mae_gd, mae_v3)

    # --- Figure 4: Convergence ---
    print("Figure 4: Convergence ...")
    plot_convergence(gapped, truth, gap_mask, mae_gd)

    # --- Figure 5: Method bar chart ---
    print("Figure 5: Method comparison bar chart ...")
    bar_results = [
        ("v4: dct_inpaint (w=50, p=2)", mae_v4),
        ("griddata (linear)", mae_gd),
        ("v3: iterative_gap_fill (w=50)", mae_v3),
    ]
    # Also compute v4 with order=1 and a few more widths for context
    for w, p in [(20, 2), (100, 2), (50, 1)]:
        filled = dct_inpaint(
            gapped,
            float(w),
            coordinates="polar",
            az_boundary="periodic",
            az_res_deg=0.5,
            order=p,
            max_iter=100,
            tol=1e-6,
        )
        mae = float(np.mean(np.abs(filled[gap_mask] - truth[gap_mask])))
        bar_results.append((f"v4: w={w}, p={p}", mae))

    # Sort by MAE for cleaner display
    bar_results.sort(key=lambda x: x[1])
    plot_method_barchart(bar_results)

    print()
    print(f"All figures saved to {FIG_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
