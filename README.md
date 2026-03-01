# DCT-Toolkit: Robust Statistical Operations

[![Development Status](https://img.shields.io/badge/status-alpha-orange)](https://github.com/yourusername/dct-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**DCT-Toolkit** provides robust statistical primitives (smoothing, mean, variance) and spectral gap filling based on the Discrete Cosine Transform (DCT). It is designed to be a constructive, composable alternative to "black box" gap filling methods.

## Key Features

- **Spectral Gap Filling**: Fill missing data (NaN gaps) via DCT-domain Penalised Least Squares (Garcia, 2010). Default bi-harmonic penalty preserves curvature across gaps — 3x more accurate than diffusion-based methods.
- **Robust Statistics**: Compute local Mean, Variance, and Count even with missing data (NaNs) using Normalized Convolution.
- **Polar Support**: Native handling of 2D Polarimetric data (Azimuth x Range) with adaptive kernels, correct boundary conditions, and range-dependent azimuth penalty.
- **Fast & Exact**: Leverages FFT-based DCT for $O(N \log N)$ performance with analytical transfer functions.
- **Zero Dependencies**: Built on pure `numpy` and `scipy`.

## Installation

```bash
git clone https://github.com/yourusername/dct-toolkit.git
cd dct-toolkit
pip install .
```

## Quick Start

### 1. Gap Filling (1D/2D)

```python
import numpy as np
from dct_toolkit import dct_inpaint

# Create data with a gap
x = np.linspace(0, 2 * np.pi, 200)
data = np.sin(x)
data[80:120] = np.nan

# Fill the gap — curvature-preserving spectral inpainting
filled = dct_inpaint(data, width=10.0)
```

### 2. Gap Filling (Polar/Radar)

```python
from dct_toolkit import dct_inpaint

# Polar radar data (Azimuth x Range) with NaN gaps
filled = dct_inpaint(
    radar_data,
    width=50.0,
    coordinates='polar',
    az_res_deg=1.0,          # adaptive azimuth penalty
    az_boundary='periodic',  # correct 0/360 wrapping
)
```

### 3. Robust Smoothing

```python
import numpy as np
import dct_toolkit as dct

# Create data with gaps
data = np.sin(np.linspace(0, 10, 100))
data[40:60] = np.nan  # Add a gap

# Smooth (automatically handles NaNs via normalized convolution)
smoothed = dct.dct_mean(data, width=10.0)
```

### 4. Polar Smoothing (Radar/Lidar)

```python
# Load polar data (Azimuth x Range)
# 360 azimuths, 1000 range gates
data = load_radar_data(...)

# Smooth with correct physics
# - Adaptive azimuth width (constant physical size)
# - Periodic boundary for azimuth (0-360 wrap)
smoothed = dct.dct_smooth(
    data,
    width=5.0,              # 5 pixels at reference range
    coordinates='polar',
    az_boundary='periodic', # Correctly handles 0/360 discontinuity
    az_res_deg=1.0
)
```

## Documentation

- [Mathematical Basis](docs/MATHEMATICAL_BASIS.md): Theory of DCT smoothing and Normalized Convolution.
- [Gap Filling Basis](docs/GAP_FILLING_BASIS.md): Mathematical basis for DCT-PLS spectral inpainting.
- [API Reference](docs/API_REFERENCE.md): Detailed function documentation.
- [Gap Filling Report](exp_v3/TEST_REPORT_GAP_FILLING.md): Experimental benchmark report.

## Project Structure

```
dct_toolkit/
├── dct_toolkit/           # Core package
│   ├── core.py            # Transfer functions
│   ├── cartesian.py       # Separable smoothing
│   ├── polar.py           # Polar smoothing
│   ├── stats.py           # Statistical ops
│   └── gap_filling.py     # Gap filling (dct_inpaint + iterative_gap_fill)
├── tests/                 # Unit tests (47+)
├── examples/              # Usage scripts
├── exp_v3/                # Experiments v3 (diffusion-based)
├── exp_v4/                # Experiments v4 (spectral inpainting)
└── docs/                  # Theory docs (API + math)
```

## License

MIT License. See [LICENSE](LICENSE) for details.
