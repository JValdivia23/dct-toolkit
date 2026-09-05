# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Added anisotropic width support while keeping scalar width as the default path:
  - Cartesian smoothing/stats now accept `width` as scalar or length-`ndim` sequence.
  - Polar smoothing/stats now accept `width` as scalar or
    `(width_azimuth, width_range)`.
  - `dct_count` now uses anisotropic window area (`prod(widths)` in Cartesian,
    `width_range * width_azimuth/(r*dtheta)` in polar).
- `dct_prefill` now gates per-iteration mean estimates by the local valid-sample
  density (`smooth(mask)`) with `min_effective_density=0.35` by default. This
  stabilizes iterative gap-filling in low-support regions and uses the same
  density signal that `dct_count / area` represents internally. The gate is
  applied uniformly across Cartesian and polar coordinates and can be disabled
  by passing `min_effective_density=None`.

### Added
- Added mixed polar kernels in `(azimuth, range)` order for `smooth_polar`,
  both polar transfer-function entry points, and high-level smoothing,
  statistics, and prefill. A single string still applies to both dimensions.
  Both azimuth boundary modes retain range-adaptive widths and apply azimuth
  smoothing before reflective range smoothing.
- Added independent polar reference tests covering all mixed kernel pairs,
  circular discrete averaging, odd/even azimuth lengths, singleton axes,
  input validation, normalized statistics, prefill, and NaN handling.
- Added per-axis Cartesian kernel types: `kernel_type` accepts a string or a
  length-`ndim` sequence of `'boxcar'`, `'boxcar_discrete'`, and `'gaussian'`.
  Mixed kernels work with scalar/per-axis widths in `smooth_cartesian`,
  `dct_smooth`, and Cartesian statistics/prefill. Existing single-string calls
  retain their behavior; the backend still uses one forward/inverse N-D DCT pair.
- Added mixed-kernel reference, validation, and NaN-handling tests, plus a 3D
  usage example and documentation of array axis order and Gaussian width units.
- Added parity and artifact-regression tests for scalar-vs-vector isotropic widths,
  plus anisotropic Cartesian/polar smoke coverage.
- Added `min_effective_density` keyword argument to `dct_prefill` (default
  `0.35`) and `dct_mean` (default `None`, opt-in) for count-based stability
  gating. When set, `dct_mean` returns NaN wherever the local valid-sample
  density is below the threshold. New tests in `tests/test_stats.py` cover
  the default, the disabled, the custom threshold, and the polar-coordinates
  paths.

### Fixed
- Polar nearest-neighbor prefill now respects periodic azimuth in both axis-wise
  propagation and the global fallback, correcting seam-dependent results on
  sparse sweeps. Range propagation remains nonperiodic. Added sparse-sweep
  rotation, targeted global fallback, and singleton-axis regression tests.
- Polar smoothing, statistics, and prefill now require a finite, positive real
  scalar `az_res_deg`, including calls that return early without smoothing.
  Shared spacing validation prevents negative counts and invalid width arithmetic;
  tests cover invalid inputs and valid fractional/NumPy scalar spacings.
- Periodic azimuth with `'boxcar_discrete'` now uses an actual centered discrete
  boxcar with an odd integer window, matching the reflective width convention.
  It previously substituted a Gaussian, so results for this option intentionally
  change for both single-string and mixed-kernel calls.

## [0.5.0] - 2026-04-14

### Changed
- `smooth_cartesian` now uses a single N-D spectral pass (`dctn/idctn`) with
  broadcasted separable transfer functions, preserving output equivalence while
  reducing transform overhead for higher-dimensional Cartesian arrays.
- Default smoothing kernel is now `'gaussian'` for Cartesian and polar smoothing paths
  (`smooth_cartesian`, `smooth_polar`, and all wrapper/statistics calls that inherit
  those defaults).
- `dct_smooth` now enforces a NaN-safe smoothing pipeline: if NaNs are present, it
  automatically pre-fills missing values before spectral smoothing and restores the
  original NaN mask afterward.
- `dct_prefill` now applies mandatory nearest-neighbor residual fill for unresolved
  targets, guaranteeing finite arrays for downstream spectral smoothing when support
  exists.
- Default prefill iteration count is now `max_iter=3` for predictable runtime. You can
  still set `max_iter=None` to iterate until convergence (safety cap: 20).
- `dct_mean`, `dct_variance`, and `dct_std` now include stable fallback handling for
  poorly conditioned normalized-convolution regions (heavy-gap volumes), preventing
  NaN/overflow behavior in low-support areas when valid support exists.
- `dct_count` now clips smoothed density to `[0, 1]` before area scaling to keep
  effective sample counts physically valid.
- `dct_count`, `dct_mean`, `dct_variance`, and `dct_std` now default to
  `restore_input_nan=True`, masking outputs at original invalid locations while
  keeping valid locations stable and finite.

### Added
- Added NaN-policy regression tests for `dct_smooth` and new `dct_prefill` iteration/
  residual behavior (`tests/test_api.py`, updates in `tests/test_stats.py`).
- Added Cartesian N-D equivalence tests to verify artifact-free 3D/4D outputs and
  new N-D statistics smoke tests (`tests/test_cartesian.py`, `tests/test_stats_nd.py`).

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
