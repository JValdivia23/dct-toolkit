# DCT-Toolkit: Convolutional Statistics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`dct_toolkit` provides DCT-based smoothing and normalized-convolution statistics for
data with gaps. The current publication-prep track is intentionally focused on the
convolution/statistics surface.

## Current Scope (Stats-First)

- DCT-based smoothing for 1D/N-D Cartesian data.
- Polar smoothing with adaptive azimuth kernels.
- Reflective boundaries via DCT and periodic azimuth boundaries via real FFT.
- Robust local statistics (`dct_mean`, `dct_variance`, `dct_std`, `dct_count`) with NaN support.

## Installation

### pip

```bash
pip install .
```

When the package is published on PyPI, installation will be:

```bash
pip install dct-toolkit
```

### from source

```bash
git clone <your-repo-url>
cd dct_toolkit
pip install .
```

### conda (planned)

Conda-forge packaging is in preparation. Until the feedstock is published, you can install with
pip inside a conda environment.

```bash
conda create -n dct-toolkit python=3.11 -y
conda activate dct-toolkit
pip install dct-toolkit
```

## Quick Start

### 1) Robust Mean with Missing Data

```python
import numpy as np
import dct_toolkit as dct

data = np.sin(np.linspace(0, 10, 200))
data[70:110] = np.nan

mu = dct.dct_mean(data, width=10.0)
```

### 1b) Prefill Before Full-Field Smoothing

```python
prefilled = dct.dct_prefill(data, width=10.0, max_iter=3)
smoothed = dct.dct_smooth(prefilled, width=10.0)
```

### 2) Polar Smoothing with Periodic Azimuth

```python
smoothed = dct.dct_smooth(
    radar_data,
    width=5.0,
    coordinates="polar",
    az_boundary="periodic",
    az_res_deg=1.0,
)
```

### 3) Local Variability Estimates

```python
var = dct.dct_variance(data, width=10.0)
std = dct.dct_std(data, width=10.0)
```

## Public API (Top Level)

- `dct_smooth`
- `dct_count`
- `dct_mean`
- `dct_prefill`
- `dct_variance`
- `dct_std`
- `smooth_cartesian`
- `smooth_polar`
- `get_dct_transfer_function`
- `dct_convolve_1d`

## Notes on Gap Filling

Gap-filling methods remain in the repository for internal research, but they are
de-scoped from the top-level public API for the initial convolution/statistics release.

## Documentation

- `docs/MATHEMATICAL_BASIS.md`
- `docs/API_REFERENCE.md`

## License

MIT. See `LICENSE`.
