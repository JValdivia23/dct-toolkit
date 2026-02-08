# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
