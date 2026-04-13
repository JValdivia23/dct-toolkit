# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Default smoothing kernel is now `'gaussian'` for Cartesian and polar smoothing paths
  (`smooth_cartesian`, `smooth_polar`, and all wrapper/statistics calls that inherit
  those defaults).

## [0.4.1] - 2026-04-12

### Added
- **Jupyter Notebook**: Added `notebooks/01_getting_started.ipynb` with interactive
  tutorial for DCT smoothing and statistics.
- Updated `.gitignore` to allow notebooks in `notebooks/` folder.

### Changed
- **Repository Cleanup**: Moved experimental content to `research/gap-filling` branch:
    - Removed `exp_v3/` and `exp_v4/` experimental folders from main branch.
    - Removed `docs/GAP_FILLING_BASIS.md` and `docs/PUBLICATION_GUIDE.md` from main branch.
    - `gap_filling.py` remains in codebase but is **not exported** in public API.
- `__init__.py`: Version bumped to `0.4.1`.

### Notes
- Main branch now contains only core package functionality for stable PyPI releases.
- All experimental work (gap filling, inpainting) preserved in `research/gap-filling` branch.

## [0.4.0] - 2026-02-28

### Added
- **DCT Spectral Inpainting (`dct_inpaint`)**:
    - New gap-filling method based on DCT-domain Penalised Least Squares (Garcia, 2010).
    - Minimises ‖W·(y − û)‖² + λ·‖Δᵖ û‖² with automatic width-to-lambda mapping.
    - Default order p=2 (bi-harmonic) preserves curvature across gaps — equivalent to
      thin-plate spline interpolation.
    - Supports both Cartesian (DCT-II reflective BC) and Polar (RFFT periodic azimuth +
      DCT reflective range) coordinates.
    - **Adaptive polar eigenvalues**: When `az_res_deg` is provided, azimuth penalty
      scales with range (`1/(r·dθ)^(2p)`), mirroring `smooth_polar`'s adaptive kernels.
    - Initialises with fast linear interpolation baseline, then refines iteratively.
    - 2.9x more accurate than `iterative_gap_fill` and 34% more accurate than
      `scipy.interpolate.griddata` on the standard polar benchmark (wrapping hole).
    - **Input validation**: Rejects `width <= 0`, `order < 1`, invalid `coordinates`
      and `az_boundary` values with clear error messages.
- **Helper functions**:
    - `_width_to_lambda`: Derive Tikhonov lambda from smoothing width.
    - `_eigenvalues_dct` / `_eigenvalues_dft`: Compute Laplacian eigenvalues for
      reflective / periodic boundary conditions.
    - `_compute_eigenvalues_2d`: Build 2-D eigenvalue tensor (isotropic or
      polar-adaptive with range-dependent azimuth scaling).
    - `_forward_transform` / `_inverse_transform`: Unified spectral transforms
      supporting mixed BC (periodic azimuth + reflective range).
- **Test suite** (`tests/test_gap_filling.py`):
    - 42 tests covering helpers, transform round-trips, convergence, 1D/2D Cartesian,
      polar (periodic + reflective + adaptive eigenvalues), curvature preservation,
      input validation, edge cases, and v3 comparison.
- **Benchmark** (`exp_v4/test_inpaint_vs_v3.py`):
    - Reproduces exp_v3 polar benchmark with dct_inpaint added.
- **Noisy benchmark** (`exp_v4/test_noisy_inpaint.py`):
    - Evaluates noise robustness across SNR levels 3–100.
- **Figures** (`exp_v4/figures/`):
    - 8 publication-quality figures (PNG + PDF): spatial comparison (v3 vs v4 vs
      griddata), error maps, width impact, convergence, method bar chart,
      noisy spatial comparison, SNR sweep, and noisy width impact.
- **Documentation**:
    - Updated `docs/API_REFERENCE.md` with full `dct_inpaint` entry.
    - Updated `docs/GAP_FILLING_BASIS.md` with DCT-PLS theory, eigenvalue
      diagonalisation, width-to-lambda bridge, and benchmark results.
    - Updated `docs/MATHEMATICAL_BASIS.md` with cross-reference to inpainting theory.
    - Updated `README.md` with gap filling features and usage examples.
    - Updated `exp_v4/PLAN.md` with refined mathematical formulation.

### Changed
- `__init__.py`: Exports `dct_inpaint`; version bumped to `0.4.0`.
- `gap_filling.py`: Module docstring updated to document both algorithms.
  `_compute_eigenvalues_2d` now accepts optional `az_res_deg` for polar-adaptive
  eigenvalues.

## [0.2.1] - 2026-02-09

### Fixed
- **Gap Filling Initialization**:
    - Improved `init='linear'` to prioritize azimuth interpolation without extrapolation (using range interpolation for edges), preventing artifacts on periodic boundaries.
- **Benchmark Consistency**:
    - Unified `exp_v3/test_width_impact.py` and `exp_v3/plot_gap_filling_results.py` to use identical test scenarios (Index Space hole).
    - Fixed MAE discrepancy in figures vs. report.

## [0.2.0] - 2026-02-09

### Added
- **Gap Filling Module (`gap_filling.py`)**:
    - `iterative_gap_fill`: Constructive gap filling via iterative DCT-based diffusion.
    - Linear interpolation initialization (axis-wise `np.interp`, default) — preserves spatial gradients across holes from the first iterate.
    - Three initialization modes: `'linear'` (default), `'multiscale'` (coarse-to-fine DCT cascade), `'dct'` (single-pass Normalized Convolution).
    - `smooth_output` option for combined gap-fill + noise reduction.
    - Convergence via relative L2 norm change tolerance.
- **Comprehensive Gap Filling Test Suite** (`tests/test_gap_filling_comprehensive.py`):
    - 13 tests covering accuracy, robustness, speed, convergence, init modes, backward compatibility, and edge cases.
- **Benchmark Suite** (`exp_v3/test_width_impact.py`):
    - 84-evaluation benchmark (3 datasets x 7 gap scenarios x 4 methods) with auto-generated CSV results.
- **Figures** (`exp_v3/figures/`):
    - 4 publication-quality figures (PNG + PDF): spatial comparison, width impact, iteration convergence, uncertainty maps.
- **Test Report** (`exp_v3/TEST_REPORT_GAP_FILLING.md`):
    - Auto-generated benchmark report; all 5 evaluation criteria PASS.
- **Algorithm Documentation** (`exp_v3/GAP_FILLING_GUIDE.md`):
    - Algorithm description, tuning guidance, and key findings from diagnostic analysis.

### Changed
- `__init__.py`: Exports `iterative_gap_fill`; version bumped to `0.2.0`.
- Gap filling promoted from `experimental/` to core module `dct_toolkit/gap_filling.py`.

### Deprecated
- `multiscale` parameter in `iterative_gap_fill` — use `init='multiscale'` or `init='dct'` instead. Emits `FutureWarning`.

## [0.1.0] - 2026-02-08

### Added
- **Core Package Structure**: Created `dct_toolkit` as a standalone python package.
- **Core Primitives (`core.py`)**:
    - Analytical and Discrete transfer functions for 'boxcar', 'boxcar_discrete', 'gaussian'.
    - `dct_convolve_1d` primitive using DCT-II/Ortho.
- **2D Cartesian Smoothing (`cartesian.py`)**:
    - Separable N-D smoothing implementation.
- **2D Polar Smoothing (`polar.py`)**:
    - Support for (Azimuth, Range) data layout.
    - Adaptive azimuth kernel width (maintains constant physical width).
    - Boundary conditions: 'reflective' (DCT) and 'periodic' (Real FFT) for azimuth.
- **Statistical Operations (`stats.py`)**:
    - `dct_count`: Effective sample size calculation.
    - `dct_mean`: Robust local mean using Normalized Convolution (handles NaNs).
    - `dct_variance` & `dct_std`: Robust second-moment statistics.
- **Experimental Gap Filling (`experimental/`)**:
    - Iterative Constructive Gap Filler based on `dct_mean`.
    - Benchmark script showing >100x accuracy improvement over linear interpolation for smooth fields.
- **Validation**:
    - Comprehensive test suite in `tests/`.
    - Example script `examples/basic_polar.py` demonstrating boundary conditions.
