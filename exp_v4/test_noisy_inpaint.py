"""
Benchmark: dct_inpaint performance on noisy data.

Tests how noise in the observed (valid) pixels affects gap-filling accuracy.
Compares:
  1. dct_inpaint (smooth_output=False) — exact data fidelity, no denoising
  2. dct_inpaint (smooth_output=True)  — joint denoise + inpaint
  3. griddata (linear) baseline

Produces 3 figures:
  1. Spatial comparison at moderate noise (SNR=10)
  2. MAE vs noise level (SNR sweep)
  3. Effect of width on noisy gap fill

Usage
-----
    conda activate myenv
    export PYTHONPATH=$PYTHONPATH:$(pwd)/dct_toolkit
    python exp_v4/test_noisy_inpaint.py
"""

import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure dct_toolkit is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dct_toolkit"))

from dct_toolkit import dct_inpaint

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
    """Create polar (theta, r) meshgrid for polar plots."""
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


def add_noise(truth: np.ndarray, snr: float, rng: np.random.Generator) -> np.ndarray:
    """
    Add white Gaussian noise to the truth field at a given SNR.

    Parameters
    ----------
    truth : np.ndarray
        Clean ground truth field.
    snr : float
        Signal-to-noise ratio (ratio of signal std to noise std).
        Higher = less noise.  SNR=10 means noise std is 10% of signal std.
    rng : np.random.Generator
        Random number generator (for reproducibility).

    Returns
    -------
    noisy : np.ndarray
        Truth + noise.
    """
    signal_std = np.std(truth)
    noise_std = signal_std / snr
    noise = rng.normal(0, noise_std, truth.shape)
    return truth + noise


def gap_mae(filled: np.ndarray, truth: np.ndarray, gap_mask: np.ndarray) -> float:
    """MAE in gap region vs clean truth."""
    return float(np.mean(np.abs(filled[gap_mask] - truth[gap_mask])))


def total_mae(filled: np.ndarray, truth: np.ndarray) -> float:
    """MAE over entire field vs clean truth."""
    return float(np.mean(np.abs(filled - truth)))


# ---------------------------------------------------------------------------
# Figure 1: Spatial comparison at moderate noise
# ---------------------------------------------------------------------------


def plot_noisy_spatial(
    truth: np.ndarray,
    noisy_gapped: np.ndarray,
    filled_exact: np.ndarray,
    filled_smooth: np.ndarray,
    filled_gd: np.ndarray,
    gap_mask: np.ndarray,
    noise_std: float,
    snr: float,
) -> None:
    """6-panel polar comparison showing noisy data and gap fills."""
    n_az, n_range = truth.shape
    THETA, R = create_polar_mesh((n_az, n_range))

    vmin = np.nanmin(truth)
    vmax = np.nanmax(truth)

    # Compute metrics
    mae_exact_gap = gap_mae(filled_exact, truth, gap_mask)
    mae_smooth_gap = gap_mae(filled_smooth, truth, gap_mask)
    mae_gd_gap = gap_mae(filled_gd, truth, gap_mask)
    mae_exact_all = total_mae(filled_exact, truth)
    mae_smooth_all = total_mae(filled_smooth, truth)

    panels = [
        ("Clean Ground Truth", truth),
        (f"Noisy + Gapped (SNR={snr:.0f})", noisy_gapped),
        (f"v4 exact fidelity\ngap MAE={mae_exact_gap:.3f}", filled_exact),
        (
            f"v4 smooth_output=True\ngap MAE={mae_smooth_gap:.3f}\ntotal MAE={mae_smooth_all:.3f}",
            filled_smooth,
        ),
        (f"griddata\ngap MAE={mae_gd_gap:.3f}", filled_gd),
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
        ax.set_title(title, fontsize=9, pad=18)
        ax.set_ylim(0, n_range)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rticks([200, 400, 600, 800])
        ax.set_rlabel_position(22.5)
        ax.tick_params(labelsize=7)

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    fig.suptitle(
        f"Noisy Inpainting Benchmark (SNR={snr:.0f}, noise std={noise_std:.2f})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout(rect=[0, 0, 0.92, 1.0])
    save_fig(fig, "noisy_spatial_comparison")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: MAE vs SNR (noise sweep)
# ---------------------------------------------------------------------------


def plot_snr_sweep(
    truth: np.ndarray,
    mask: np.ndarray,
    gap_mask: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """MAE vs SNR for different methods, evaluated in gap region and total."""
    snr_values = [3, 5, 7, 10, 15, 20, 50, 100]
    width = 50.0

    gap_maes_exact = []
    gap_maes_smooth = []
    gap_maes_gd = []
    total_maes_exact = []
    total_maes_smooth = []

    print()
    print(
        f"  {'SNR':>5s}  {'exact_gap':>10s}  {'smooth_gap':>10s}  "
        f"{'gd_gap':>10s}  {'exact_tot':>10s}  {'smooth_tot':>10s}"
    )
    print("  " + "-" * 60)

    for snr in snr_values:
        noisy = add_noise(truth, snr, rng)
        noisy_gapped = noisy.copy()
        noisy_gapped[gap_mask] = np.nan

        # v4 exact fidelity
        filled_exact = dct_inpaint(
            noisy_gapped,
            width,
            coordinates="polar",
            az_boundary="periodic",
            az_res_deg=0.5,
            order=2,
            max_iter=100,
            tol=1e-6,
            smooth_output=False,
        )
        # v4 smooth output (denoise + inpaint)
        filled_smooth = dct_inpaint(
            noisy_gapped,
            width,
            coordinates="polar",
            az_boundary="periodic",
            az_res_deg=0.5,
            order=2,
            max_iter=100,
            tol=1e-6,
            smooth_output=True,
        )
        # griddata
        filled_gd = fill_griddata(noisy, mask, range_res_m=100.0, az_res_deg=0.5)

        g_exact = gap_mae(filled_exact, truth, gap_mask)
        g_smooth = gap_mae(filled_smooth, truth, gap_mask)
        g_gd = gap_mae(filled_gd, truth, gap_mask)
        t_exact = total_mae(filled_exact, truth)
        t_smooth = total_mae(filled_smooth, truth)

        gap_maes_exact.append(g_exact)
        gap_maes_smooth.append(g_smooth)
        gap_maes_gd.append(g_gd)
        total_maes_exact.append(t_exact)
        total_maes_smooth.append(t_smooth)

        print(
            f"  {snr:5.0f}  {g_exact:10.4f}  {g_smooth:10.4f}  "
            f"{g_gd:10.4f}  {t_exact:10.4f}  {t_smooth:10.4f}"
        )

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Gap MAE
    ax1.plot(
        snr_values,
        gap_maes_exact,
        "o-",
        color="#2196F3",
        linewidth=2,
        label="v4 exact fidelity",
    )
    ax1.plot(
        snr_values,
        gap_maes_smooth,
        "s-",
        color="#4CAF50",
        linewidth=2,
        label="v4 smooth_output=True",
    )
    ax1.plot(
        snr_values, gap_maes_gd, "^--", color="gray", linewidth=1.5, label="griddata"
    )
    ax1.set_xlabel("SNR (signal std / noise std)", fontsize=12)
    ax1.set_ylabel("MAE in Gap Region (vs clean truth)", fontsize=12)
    ax1.set_title("Gap Fill Accuracy vs Noise Level", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")
    ax1.set_xticks(snr_values)
    ax1.set_xticklabels([str(s) for s in snr_values])

    # Right: Total MAE (gap + valid pixels)
    ax2.plot(
        snr_values,
        total_maes_exact,
        "o-",
        color="#2196F3",
        linewidth=2,
        label="v4 exact fidelity",
    )
    ax2.plot(
        snr_values,
        total_maes_smooth,
        "s-",
        color="#4CAF50",
        linewidth=2,
        label="v4 smooth_output=True",
    )
    ax2.set_xlabel("SNR (signal std / noise std)", fontsize=12)
    ax2.set_ylabel("MAE over Entire Field (vs clean truth)", fontsize=12)
    ax2.set_title(
        "Total Field Error (denoising effect)", fontsize=13, fontweight="bold"
    )
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")
    ax2.set_xticks(snr_values)
    ax2.set_xticklabels([str(s) for s in snr_values])

    plt.tight_layout()
    save_fig(fig, "noisy_snr_sweep")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Width impact on noisy gap fill
# ---------------------------------------------------------------------------


def plot_noisy_width_impact(
    truth: np.ndarray,
    mask: np.ndarray,
    gap_mask: np.ndarray,
    snr: float,
    rng: np.random.Generator,
) -> None:
    """MAE vs width at a fixed noise level for exact vs smooth modes."""
    noisy = add_noise(truth, snr, rng)
    noisy_gapped = noisy.copy()
    noisy_gapped[gap_mask] = np.nan

    widths = [5, 10, 20, 50, 75, 100, 150]

    gap_maes_exact = []
    gap_maes_smooth = []
    total_maes_smooth = []

    for w in widths:
        filled_exact = dct_inpaint(
            noisy_gapped,
            float(w),
            coordinates="polar",
            az_boundary="periodic",
            az_res_deg=0.5,
            order=2,
            max_iter=100,
            tol=1e-6,
            smooth_output=False,
        )
        filled_smooth = dct_inpaint(
            noisy_gapped,
            float(w),
            coordinates="polar",
            az_boundary="periodic",
            az_res_deg=0.5,
            order=2,
            max_iter=100,
            tol=1e-6,
            smooth_output=True,
        )
        gap_maes_exact.append(gap_mae(filled_exact, truth, gap_mask))
        gap_maes_smooth.append(gap_mae(filled_smooth, truth, gap_mask))
        total_maes_smooth.append(total_mae(filled_smooth, truth))

    # griddata baseline
    filled_gd = fill_griddata(noisy, mask, range_res_m=100.0, az_res_deg=0.5)
    mae_gd = gap_mae(filled_gd, truth, gap_mask)

    # noise floor (expected MAE of noise at valid pixels)
    noise_std = np.std(truth) / snr
    noise_floor = noise_std * np.sqrt(2 / np.pi)  # E[|N(0,sigma)|] = sigma*sqrt(2/pi)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Gap MAE
    ax1.plot(
        widths,
        gap_maes_exact,
        "o-",
        color="#2196F3",
        linewidth=2,
        label="v4 exact fidelity",
    )
    ax1.plot(
        widths,
        gap_maes_smooth,
        "s-",
        color="#4CAF50",
        linewidth=2,
        label="v4 smooth_output=True",
    )
    ax1.axhline(
        mae_gd,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"griddata (MAE={mae_gd:.3f})",
    )
    ax1.axhline(
        noise_floor,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label=f"noise floor (MAE={noise_floor:.3f})",
    )
    ax1.set_xlabel("Smoothing Width (pixels)", fontsize=12)
    ax1.set_ylabel("MAE in Gap Region (vs clean truth)", fontsize=12)
    ax1.set_title(
        f"Gap Fill Accuracy vs Width (SNR={snr:.0f})", fontsize=13, fontweight="bold"
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")
    ax1.set_xticks(widths)
    ax1.set_xticklabels([str(w) for w in widths])

    # Right: Total MAE (denoising quality)
    ax2.plot(
        widths,
        total_maes_smooth,
        "s-",
        color="#4CAF50",
        linewidth=2,
        label="v4 smooth_output=True",
    )
    ax2.axhline(
        noise_floor,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label=f"noise floor (MAE={noise_floor:.3f})",
    )
    ax2.set_xlabel("Smoothing Width (pixels)", fontsize=12)
    ax2.set_ylabel("Total MAE (vs clean truth)", fontsize=12)
    ax2.set_title(
        f"Denoising Quality vs Width (SNR={snr:.0f})", fontsize=13, fontweight="bold"
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")
    ax2.set_xticks(widths)
    ax2.set_xticklabels([str(w) for w in widths])

    plt.tight_layout()
    save_fig(fig, "noisy_width_impact")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("Noisy Inpainting Benchmark")
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
    n_gap = np.sum(gap_mask)
    signal_std = np.std(truth)
    print(f"Scenario: 720x1000 polar grid, wrapping hole (r=100)")
    print(f"Gap pixels: {n_gap} ({100 * n_gap / truth.size:.1f}%)")
    print(f"Signal std: {signal_std:.4f}")
    print()

    # --- Figure 1: Spatial comparison at SNR=10 ---
    snr_demo = 10.0
    noise_std = signal_std / snr_demo
    noisy = add_noise(truth, snr_demo, rng)
    noisy_gapped = noisy.copy()
    noisy_gapped[gap_mask] = np.nan

    print(f"Computing fills at SNR={snr_demo:.0f} (noise std={noise_std:.3f}) ...")
    t0 = time.time()
    filled_exact = dct_inpaint(
        noisy_gapped,
        50.0,
        coordinates="polar",
        az_boundary="periodic",
        az_res_deg=0.5,
        order=2,
        max_iter=100,
        tol=1e-6,
        smooth_output=False,
    )
    print(f"  v4 exact fidelity: {time.time() - t0:.2f}s")

    t0 = time.time()
    filled_smooth = dct_inpaint(
        noisy_gapped,
        50.0,
        coordinates="polar",
        az_boundary="periodic",
        az_res_deg=0.5,
        order=2,
        max_iter=100,
        tol=1e-6,
        smooth_output=True,
    )
    print(f"  v4 smooth_output=True: {time.time() - t0:.2f}s")

    t0 = time.time()
    filled_gd = fill_griddata(noisy, mask, range_res_m=100.0, az_res_deg=0.5)
    print(f"  griddata: {time.time() - t0:.2f}s")

    print()
    print(f"  Gap MAE (vs clean truth):")
    print(f"    v4 exact:    {gap_mae(filled_exact, truth, gap_mask):.4f}")
    print(f"    v4 smooth:   {gap_mae(filled_smooth, truth, gap_mask):.4f}")
    print(f"    griddata:    {gap_mae(filled_gd, truth, gap_mask):.4f}")
    print(f"  Total MAE (vs clean truth):")
    print(f"    v4 exact:    {total_mae(filled_exact, truth):.4f}")
    print(f"    v4 smooth:   {total_mae(filled_smooth, truth):.4f}")
    print()

    print("Figure 1: Noisy spatial comparison ...")
    plot_noisy_spatial(
        truth,
        noisy_gapped,
        filled_exact,
        filled_smooth,
        filled_gd,
        gap_mask,
        noise_std,
        snr_demo,
    )

    # --- Figure 2: SNR sweep ---
    print("Figure 2: SNR sweep ...")
    rng2 = np.random.default_rng(123)  # fresh rng for reproducibility
    plot_snr_sweep(truth, mask, gap_mask, rng2)

    # --- Figure 3: Width impact at SNR=10 ---
    print("Figure 3: Noisy width impact ...")
    rng3 = np.random.default_rng(456)
    plot_noisy_width_impact(truth, mask, gap_mask, snr=10.0, rng=rng3)

    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()
    print("Key observations:")
    print("  1. With smooth_output=False, observed noisy pixels are preserved")
    print("     exactly -> noise propagates into the gap fill.")
    print("  2. With smooth_output=True, the spectral filter denoises the")
    print("     entire field AND fills gaps simultaneously.")
    print("  3. The width parameter controls the bias-variance tradeoff:")
    print("     larger width = more denoising but more smoothing of signal.")
    print()
    print(f"All figures saved to {FIG_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
