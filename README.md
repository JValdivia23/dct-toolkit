# DCT-Toolkit: Convolutional Statistics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`dct_toolkit` provides DCT/FFT-based smoothing and normalized-convolution statistics
for data with gaps. Runtime dependencies are NumPy and SciPy only.

**Version: v1.0.0.** Geometry-aware N-D polar processing requires version 1.0.0
or newer. Existing 2D polar calls retain their index-based width conventions.

## Scope

- DCT-based smoothing for 1D/N-D Cartesian data.
- Per-axis Cartesian widths and kernel types, including mixed boxcar/Gaussian smoothing.
- Geometry-aware N-D polar/spherical smoothing with named axes in any array order,
  physical or grid widths, independent kernels/boundaries, and batch isolation.
- Backward-compatible 2D polar smoothing with index-based adaptive azimuth widths.
- Reflective boundaries via DCT and periodic boundaries via real FFT.
- Robust local statistics (`dct_mean`, `dct_variance`, `dct_std`, `dct_count`) with NaN support.

## Installation

### pip

```bash
python -m pip install dct-toolkit
```

### conda

```bash
conda install -c conda-forge dct-toolkit
```

To update an existing installation:

```bash
python -m pip install --upgrade dct-toolkit
# Or, in an environment managed by conda:
conda update -c conda-forge dct-toolkit
```

PyPI and conda-forge publish separately; the conda update becomes available after
the feedstock update has passed its builds and been merged.

For development, install from this checkout:

```bash
python -m pip install -e .
python examples/nd_polar.py
```

The example uses synthetic data and needs no radar files or plotting packages.
Install the package first, then run the example from the repository root.

## Quick Start

### 1) Robust Mean with Missing Data

```python
import numpy as np
import dct_toolkit as dct

data = np.sin(np.linspace(0, 10, 200))
data[70:110] = np.nan

mu = dct.dct_mean(data, width=10.0)
```

### 1b) Prefill and Smoothing

```python
# Automatic prefill, then smoothing; the original NaN mask is restored.
smoothed = dct.dct_smooth(data, width=10.0)

# To obtain a filled field instead, explicitly prefill first.
prefilled = dct.dct_prefill(data, width=10.0)
full_field = dct.dct_smooth(prefilled, width=10.0)
```

### 2) Legacy 2D Polar Smoothing with Periodic Azimuth

For data in **(azimuth, range)** order, combine a boxcar along azimuth with a
Gaussian along range:

```python
radar_data = np.random.default_rng(42).standard_normal((360, 100))
smoothed = dct.dct_smooth(
    radar_data,
    width=(5.0, 3.0),
    coordinates="polar",
    kernel_type=("boxcar", "gaussian"),
    az_boundary="periodic",
    az_res_deg=1.0,
)
```

With `axis_types=None` (the default), both sequences follow `(azimuth, range)`
order. Azimuth width adapts with range:
`width_azimuth / (r * dtheta)` beams, where `r` starts at 1 and `dtheta` is the
azimuth spacing in radians. Range width is in range gates. Gaussian
`sigma = effective_width / sqrt(12)` on the selected axis. Filtering applies
azimuth first, then range; both periodic and reflective azimuth support all
three kernel types, including `'boxcar_discrete'`.
`az_res_deg` must be a finite, positive scalar in degrees. For gapped data,
periodic azimuth also wraps the nearest-neighbor prefill fallback across the
sweep seam; range distances remain nonperiodic.

This compatibility path does **not** infer actual range coordinates or elevation.
Its widths are not reinterpreted as physical distances in v1.0.0. Nondefault new
geometry arguments require explicit `axis_types`.

### 3) Local Variability Estimates

```python
var = dct.dct_variance(data, width=10.0)
std = dct.dct_std(data, width=10.0)
```

### 4) Mixed Kernels for a 3D Volume

For a volume stored in **(z, y, x)** order, apply Gaussian smoothing vertically
and boxcar smoothing horizontally:

```python
rng = np.random.default_rng(42)
volume = rng.standard_normal((16, 32, 32))  # (z, y, x)

smoothed = dct.dct_smooth(
    volume,
    width=(3.0, 5.0, 5.0),
    kernel_type=("gaussian", "boxcar", "boxcar"),
)

# The same kernels work with normalized statistics on data containing NaNs.
volume[5:7, 12:15, 12:15] = np.nan
mean = dct.dct_mean(
    volume,
    width=(3.0, 5.0, 5.0),
    kernel_type=("gaussian", "boxcar", "boxcar"),
)
```

Kernel and width sequences follow NumPy axis order and must have one entry per
dimension. Widths are in grid cells along each axis; Gaussian
`sigma = width / sqrt(12)`. Convert physical widths using the corresponding
grid spacing before calling. A scalar width applies equal widths, and a single
kernel string applies that kernel to every axis. Equal widths with different
kernel types can still produce anisotropic smoothing. Legacy polar smoothing
accepts a kernel pair, using the `(azimuth, range)` convention shown above.

### 5) N-D Polar Smoothing (v1.0.0)

#### Which inputs are required?

Use `coordinates="polar"` to select the polar backend and supply `axis_types`
to enable explicit N-D geometry. You do **not** need to supply all three of
`axis_types`, `axis_coords`, and `width_units` in every call:

| Argument | Required for explicit N-D polar processing? |
|----------|--------------------------------------------|
| `axis_types` | **Yes:** one role per array axis, in array order. |
| `axis_coords` | Only where physical width conversion or geometry dependencies need them. |
| `width_units` | **No:** defaults to `"physical"`; use `"grid"` for widths in cells. |

With `width_units="grid"`, no coordinate arrays are required. Physical filtering
needs coordinates on selected, non-singleton axes. Physical angular filtering
also needs actual positive range coordinates; physical azimuth filtering needs
elevation coordinates when that role exists. These geometry dependencies remain
required even when range/elevation are unfiltered. Coordinate entries for batch
or unfiltered axes may otherwise be `None`; supplied coordinates are always
validated.

Omitting `axis_types` does **not** infer N-D geometry: it retains the legacy
2D `(azimuth, range)` mode described above.

#### Start with grid widths: no coordinates needed

This self-contained example filters a `(time, elevation, azimuth, range)` array
using widths in cells, with periodic azimuth and reflective boundaries elsewhere:

```python
import numpy as np
import dct_toolkit as dct

data = np.random.default_rng(42).standard_normal((3, 3, 36, 24))
data[0, 1, :3, 8:12] = np.nan

mean_grid = dct.dct_mean(
    data,
    width=(3.0, 3.0, 5.0, 7.0),             # time, elevation, azimuth, range cells
    coordinates="polar",
    axis_types=("time", "elevation", "azimuth", "range"),
    width_units="grid",
    boundary=("reflective", "reflective", "periodic", "reflective"),
    restore_input_nan=False,                # retain supported estimates inside gaps
)
```

By default, `dct_mean` restores input NaNs. Setting `restore_input_nan=False`
keeps estimates in gaps where support exists; wholly missing independent domains
still remain NaN. To keep time slices independent, add `filter_axes=(1, 2, 3)`.
Still supply one positive width per array axis; the unfiltered time width is then
an unused placeholder.

#### How does this differ from Cartesian N-D?

Both backends already handle N-dimensional arrays. The polar extension adds
geometry and filtering controls, not just more dimensions:

| Feature | Cartesian N-D | Explicit polar N-D |
|---------|---------------|--------------------|
| Axis roles | Not specified | Named roles in array order |
| Width units | Grid cells | Grid or coordinate-based physical units |
| Angular widths | No angular metric | Physical widths adapt with range/elevation |
| Boundaries | Reflective on every axis | Reflective or periodic per axis |
| Filtered axes | All axes | `filter_axes`; batch axes excluded |

Cartesian calls need no axis metadata: `dct.dct_mean(data, width=(3, 3, 5, 7))`.
Convert physical Cartesian widths to cells yourself using each axis's spacing.
Polar physical angular widths additionally account for range and elevation, as
shown below; they are distances, not degrees or radians.

With grid widths, the same widths/kernels, every axis filtered, and all boundaries
reflective, **finite-data smoothing matches Cartesian smoothing** up to numerical
rounding. This equivalence concerns the smoothing operator, not nominal count
conventions or missing-data fallback rules.

The same polar geometry options work with `dct_smooth`, `dct_variance`, `dct_std`,
`dct_count`, and `dct_prefill`. The low-level `smooth_polar` uses `width_pixels=`
instead of `width=` and does not take `coordinates=`. See the
[API reference](docs/API_REFERENCE.md#dct_toolkitpolar) for full syntax and validation.

#### Physical widths with actual coordinates

For a regular array in **(time, elevation, azimuth, range)** order:

```python
time = np.arange(3) * 30.0                       # seconds
elevation = np.array([2.0, 4.0, 6.0])            # degrees
azimuth = np.arange(0.0, 360.0, 10.0)            # full circle, no duplicate endpoint
ranges = 2.0 + 0.25 * np.arange(24)             # km; first gate is really 2 km
data = np.random.default_rng(42).standard_normal((3, 3, 36, 24))
data[0, 1, :3, 8:12] = np.nan

mean = dct.dct_mean(
    data,
    (60.0, 0.3, 0.9, 0.75),                    # seconds, km, km, km
    coordinates="polar",
    axis_types=("time", "elevation", "azimuth", "range"),
    axis_coords=(time, elevation, azimuth, ranges),
    width_units="physical",
    angle_units="degrees",
    kernel_type=("gaussian", "boxcar_discrete", "boxcar_discrete", "gaussian"),
    boundary=("reflective", "reflective", "periodic", "reflective"),
)
```

- Every per-axis sequence follows **array order**, not a fixed spherical order.
  Reorder `axis_types`, coordinates, widths, kernels, boundaries, and per-axis
  width units together when transposing the data.
- `axis_types` supports `azimuth`, `elevation`, `range`, `time`, `linear`,
  `cartesian` (alias of `linear`), and `batch`; spherical roles cannot repeat.
- Explicit geometry defaults to `width_units="physical"`. Time/linear/range
  widths are divided by coordinate spacing. Angular widths are **distances in
  the range coordinate's units**, not degrees: in grid cells,
  `w_az = L_az / (R * cos(elevation) * delta_az)` and
  `w_el = L_el / (R * delta_el)`, using radians internally. Without an elevation
  axis, the azimuth cosine factor is 1. `width_units="grid"` means literal cells;
  a length-`ndim` sequence can mix physical and grid units.
- Supply actual coordinates where physical conversion needs them. Coordinates
  must be finite real 1D arrays, strictly increasing and uniformly spaced.
  Physical angular filtering requires positive actual ranges and the relevant
  angular coordinates; singular origins/poles are rejected, not capped.
  Non-singleton physical periodic azimuth filtering requires a full 360-degree
  or `2*pi`-radian grid without repeating the endpoint.
- `filter_axes=None` smooths all non-batch axes. For independent time slices,
  pass `filter_axes=(1, 2, 3)` in this example. Unselected and `batch` axes never
  exchange values, even during prefill or low-support fallback. A wholly missing
  independent domain stays missing. Batch axes cannot be selected for filtering.
  Singleton axes are identities with nominal support width 1.

Filtering uses canonical role order: **azimuth, elevation, range, time, linear**,
regardless of storage order. Geometry-dependent widths make this operator
generally nonseparable and noncommuting. It is not an exact isotropic Cartesian
or geodesic convolution, nor a cell-volume-weighted average. Only regular
coordinate grids are supported, not irregular concatenated multi-sweep radar
acquisitions. See [the complete example](examples/nd_polar.py) for missing-data,
axis-permutation, and independent-domain demonstrations.

## Statistics and Support

Let `S` be the complete smoothing operator and `m` the finite-data validity mask.
The mean uses `S(m*x) / S(m)`, dividing **once after all filters**, with invalid
data zero-filled before filtering. The normal ratio path requires a finite
denominator **strictly greater than `1e-3`**; it does not add an epsilon.
Low-support locations use the existing prefill-and-smooth fallback when their
independent domain has support. Variance independently estimates the first and
second moments, including separate prefill of `x` and `x**2` in fallback, then
clamps `E[x**2] - E[x]**2` to zero from below.

`dct_count` keeps its API name but reports **nominal weighted support**:
`clip(S(m), 0, 1)` times the product of local grid widths on filtered axes.
Batch/unfiltered axes contribute no support factor. This is not a literal number
of distinct neighbors, a cell volume, or squared-weight statistical effective
sample size (ESS). Reflection/wrapping can revisit the same input sample.
Nonnegative-weight variance guarantees require actual nonnegative spatial
weights; arbitrary fractional spectral kernels are not automatically positive.

## Quick Cheat Sheet

| Function | Description |
|----------|-------------|
| `get_dct_transfer_function` | Build 1D spectral kernel `H[k]` from width and kernel type. |
| `dct_convolve_1d` | Apply one 1D DCT convolution along one axis using a precomputed `H`. |
| `smooth_cartesian` | Low-level separable smoothing for N-D Cartesian arrays. |
| `smooth_polar` | Low-level N-D geometry-aware or legacy 2D polar smoothing. |
| `dct_smooth` | Top-level "just smooth this" wrapper (auto-pre-fills NaNs, then restores NaN mask). |
| `dct_mean` | NaN-robust local mean via normalized convolution. |
| `dct_count` | Nominal weighted support from a validity mask (not statistical ESS). |
| `dct_prefill` | Iterative normalized-convolution gap fill (often used before full-field smoothing). |
| `dct_variance` | NaN-robust local variance from normalized-convolution moments. |
| `dct_std` | NaN-robust local standard deviation (`sqrt(dct_variance)`). |

## Pre-filling (`dct_prefill`)

`dct_prefill` provides an **iterative normalized-convolution gap fill** based on `dct_mean`:

- Fills gaps (NaNs or a specified mask) using iterative local averaging.
- Preserves non-target values exactly.
- Intended as a **pre-processing step** before full-field smoothing.
- `max_iter=3` by default; `max_iter=None` iterates until convergence or a safety
  cap of 20 iterations. Remaining targets use nearest-neighbor fallback only
  within their filtered domain; domains with no finite support remain missing.
- `min_effective_density=0.35` gates iterative estimates using the smoothed
  validity mask, not statistical ESS. Set it to `None` to disable the gate.

See the Quick Start section above for usage examples.

Research-internal inpainting/diffusion algorithms are documented in the
[gap-filling module](dct_toolkit/gap_filling.py); they are not exported by the
top-level public API and do not inherit the new N-D geometry contract.

## Documentation

- [Mathematical basis](docs/MATHEMATICAL_BASIS.md) — Operators, moments, and support
- [API reference](docs/API_REFERENCE.md) — Parameters, compatibility, and limitations

## License

MIT. See `LICENSE`.
