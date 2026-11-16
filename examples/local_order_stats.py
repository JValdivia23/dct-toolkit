"""Demonstrate local weighted distributions, support, and strict smoothing."""

import numpy as np

from dct_toolkit import dct_mean, local_mad, local_quantile, local_support


def main() -> None:
    """Compute isolated polar distributions without creating synthetic support."""
    rng = np.random.default_rng(2301)
    data = rng.normal(size=(2, 36, 20))  # independent batch, azimuth, range
    data[0, 3:5, 8:10] = np.nan
    data[1] = np.nan
    widths = (1., 4., 3.)
    geometry = {
        "coordinates": "polar", "axis_types": ("batch", "azimuth", "range"),
        "axis_coords": (None, np.arange(36) * 10., np.arange(20) + 2.),
        "boundary": ("reflective", "periodic", "reflective"),
        "kernel_type": ("gaussian", "gaussian", "boxcar_discrete"),
    }
    quantiles = local_quantile(data, [.25, .5, .75], widths, bins=64,
                               exclude_center=True, return_info=True, **geometry)
    mad = local_mad(data, widths, bins=64, exclude_center=True,
                    return_info=True, max_memory_mb=8, **geometry)
    support = local_support(np.isfinite(data), widths, exclude_center=True, **geometry)
    mean = dct_mean(data, widths, fallback="none", exclude_center=True, **geometry)
    assert quantiles.values.shape == (3,) + data.shape
    assert mad.values.shape == data.shape
    assert np.all(np.isnan(mean[1])) and not np.any(support.supported[1])
    assert np.all(np.isnan(quantiles.values[:, 1])) and np.all(np.isnan(mad.values[1]))
    print("Quantile axis:", quantiles.values.shape)
    print("Histogram bins:", quantiles.bin_count, "; no support crosses batch axes")
    print("Maximum MAD value-quantization bound:", np.nanmax(mad.error_bound))
    print("Supported external neighborhoods:", np.count_nonzero(support.supported))
    print("Strict smoothing and order statistics preserve all-missing batch isolation.")


if __name__ == "__main__":
    main()
