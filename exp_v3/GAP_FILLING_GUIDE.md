# Gap Filling Guide

## Overview

The `iterative_gap_fill` function fills missing data (NaN values) in 1-D or
2-D arrays using iterative DCT-based diffusion.  It is designed for fields
where the underlying signal is smooth or band-limited — typical of
atmospheric, oceanographic, and radar datasets.

```python
from dct_toolkit import iterative_gap_fill

filled = iterative_gap_fill(data, width=10.0, max_iter=10)
```

## Algorithm

The algorithm has two phases:

### Phase 1: Initialization

Gap values are filled with an initial estimate before iterative refinement.
The `init` parameter controls which strategy is used:

| Mode | Description | Best for |
|------|-------------|----------|
| `'linear'` (default) | Axis-wise linear interpolation (row-wise, then column-wise, then global-mean fallback) | General use; contiguous holes |
| `'multiscale'` | Coarse-to-fine DCT cascade from `max(shape)/4` down to target width | Legacy compatibility |
| `'dct'` | Single-pass Normalized Convolution at target width | Small scattered gaps |

**Linear initialization** is equivalent to `scipy.interpolate.griddata`
(linear) for most hole geometries.  It preserves spatial gradients across
holes from the very first iterate, giving the iterative phase a better
starting point.

### Phase 2: Iterative DCT Diffusion

At each iteration *k*:

1. **Smooth** the entire current field (gaps + valid) using `dct_mean` at
   the user's target `width`.
2. **Reset** valid data to its original values; update **only** gap pixels
   with the smooth trend.
3. **Check convergence**: stop when the relative L2 change falls below `tol`.

$$
\text{Trend}^{(k)} = \text{dct\_mean}(\hat{y}^{(k)}, \sigma)
$$

$$
\hat{y}^{(k+1)} = M \cdot y_{\text{obs}} + (1 - M) \cdot \text{Trend}^{(k)}
$$

This is equivalent to solving the heat equation (diffusion) with Dirichlet
boundary conditions at the valid data points.

## Tuning Guide

### Choosing `width`

The `width` parameter controls the **stiffness** of the interpolated surface.

| Scenario | Recommended `width` | Notes |
|----------|---------------------|-------|
| Small scattered gaps (< few pixels) | 3–5 | Linear init already near-optimal; DCT adds little |
| Moderate holes (~20 px diameter) | 5–10 | Good balance of speed and accuracy |
| Large contiguous holes (~100+ px) | 25–50 | Width ≈ 25% of hole diameter works well |
| Very large holes (> 200 px) | 50–100 | Diminishing returns; too large causes over-smoothing |

**Key finding**: For large contiguous holes, the iterative DCT phase
provides a meaningful improvement over linear interpolation **only** when
the smoothing width is a significant fraction of the hole diameter.
With small widths (3–5), the DCT iterations barely change the linear-init
result.

### Choosing `max_iter`

- **5–10 iterations** is usually sufficient for convergence.
- More iterations never hurt accuracy (the method converges monotonically
  in the fixed-point sense), but cost more time.
- The `tol` parameter (default `1e-4`) provides early stopping.

### `smooth_output`

Set `smooth_output=True` to apply a final smoothing pass to the **entire**
field (valid + filled).  This is useful when you want combined gap filling
and noise reduction in a single step.  When `False` (default), valid data
is preserved exactly.

## Performance

On a 720 x 1000 grid with 30% random gaps (smooth field):

| Method | Time (ms) | MAE |
|--------|-----------|-----|
| 1D Linear (az-only) | 29 | 5.4e-5 |
| **Iterative DCT** (w=5, 10 iter) | **111** | **2.4e-5** |
| 2D Griddata (Delaunay) | 26,120 | 5.0e-6 |

- DCT is **~234x faster** than griddata with comparable accuracy.
- Griddata wins on raw accuracy for random gaps, but is impractical for
  large grids or real-time applications.
- DCT outperforms Astropy convolution on smooth fields and handles large
  holes where Astropy's kernel is too small.

## Limitations

1. **Width must match hole scale**.  A width of 5 pixels does almost nothing
   to improve the fill inside a 200-pixel hole.  The user must choose a width
   appropriate to the gap geometry.

2. **Low-pass nature**.  DCT smoothing acts as a low-pass filter.  Sharp edges
   and discontinuities in the true field will be smoothed over.  For data with
   sharp features, consider masking them separately.

3. **Linear init is not a magic bullet**.  For contiguous holes, linear
   interpolation (the default init) gives a piecewise-linear surface that is
   essentially identical to griddata.  The real accuracy improvement comes
   from the iterative DCT diffusion step with an appropriately sized width.

4. **2-D only** (for now).  The `'linear'` init mode supports 1-D and 2-D
   arrays.  3-D support is planned for v0.3.0.

## Uncertainty Estimation

The `dct_std` function can be used to estimate local uncertainty in the
filled values.  The mapping error (ratio of dct_std to the field's standard
deviation) depends on mask geometry and smoothing width:

```python
from dct_toolkit import dct_std
uncertainty = dct_std(data, width=10.0, mask=valid_mask)
```

See the benchmark figures in `exp_v3/figures/` for spatial maps of uncertainty.

## References

- Mathematical derivation: [`docs/GAP_FILLING_BASIS.md`](../docs/GAP_FILLING_BASIS.md)
- Benchmark results: [`exp_v3/TEST_REPORT_GAP_FILLING.md`](TEST_REPORT_GAP_FILLING.md)
- API details: [`docs/API_REFERENCE.md`](../docs/API_REFERENCE.md)
- Garcia (2010): DCT-PLS for comparison context
- Knutsson et al.: Normalized Convolution foundation
