# Experimental: Iterative Constructive Gap Filling

This module implements a constructive gap filling algorithm built from the robust statistical primitives in `dct-toolkit`.

## The Idea

Instead of solving a complex penalized least squares problem (like the Garcia method), we can "construct" the missing data by iteratively smoothing the field.

1. **Normalized Convolution** provides a robust local mean, even in the presence of gaps.
2. If we fill the gaps with this robust mean, we get a first-order approximation.
3. If we repeat this process, allowing information to propagate from valid regions into the gaps, the solution converges to a smooth surface that respects the boundaries of the valid data.

## Usage

```python
from iterative_fill import iterative_gap_fill

# data has NaNs
filled_data = iterative_gap_fill(data, width=5.0, max_iter=50)
```

## Performance

See `benchmark.py` for a comparison with Linear Interpolation.
- **Accuracy**: ~100x better Mean Absolute Error (MAE) for smooth synthetic fields.
- **Speed**: ~16x faster than `scipy.interpolate.griddata`.

## Mathematical Basis

See [GAP_FILLING_MATH.md](../../docs/experimental/GAP_FILLING_MATH.md) for details.
