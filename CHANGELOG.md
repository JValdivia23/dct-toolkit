# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
