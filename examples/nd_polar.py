"""Demonstrate the v1.0.0 N-D polar API on a synthetic regular grid.

Install the package first (for development, ``python -m pip install -e .``),
then run ``python examples/nd_polar.py``. The example uses the installed package
without changing the import path. No external radar files, plotting packages,
or dependencies beyond NumPy/SciPy and this package are needed.
"""

import numpy as np

import dct_toolkit as dct


def main() -> None:
    """Run geometry, missing-data, permutation, and independent-domain examples.

    Returns
    -------
    None
        Print a short summary after checking the demonstrated properties.
    """
    time = np.arange(3) * 30.0                  # seconds
    elevation = np.array([2.0, 4.0, 6.0])       # degrees; away from the poles
    azimuth = np.arange(0.0, 360.0, 10.0)       # full circle, no repeated endpoint
    ranges = 2.0 + 0.25 * np.arange(24)         # km; actual nonzero range origin
    t, el, az, r = np.meshgrid(time, elevation, azimuth, ranges, indexing="ij")
    signal = 3.0 + 0.01 * t + np.cos(np.deg2rad(az)) * np.cos(np.deg2rad(el)) + 0.2 * r
    data = signal + np.random.default_rng(42).normal(0.0, 0.15, signal.shape)
    data[0, 1, :3, 8:12] = np.nan

    # Every per-axis sequence follows ARRAY order: time, elevation, azimuth, range.
    # Angular physical widths are distances in the range coordinate's units (km).
    widths = (60.0, 0.3, 0.9, 0.75)            # seconds, km, km, km
    geometry = {
        "axis_types": ("time", "elevation", "azimuth", "range"),
        "axis_coords": (time, elevation, azimuth, ranges),
        "width_units": "physical",
        "angle_units": "degrees",
        "kernel_type": ("gaussian", "boxcar_discrete", "boxcar_discrete", "gaussian"),
        "boundary": ("reflective", "reflective", "periodic", "reflective"),
    }

    # Normalize only after the complete operator. False allows estimates in gaps.
    mean = dct.dct_mean(
        data, widths, coordinates="polar", restore_input_nan=False, **geometry
    )
    std = dct.dct_std(data, widths, coordinates="polar", **geometry)
    support = dct.dct_count(
        np.isfinite(data), widths, coordinates="polar", restore_input_nan=False, **geometry
    )
    assert mean.shape == data.shape
    assert np.isfinite(mean).all()
    assert np.isnan(std[np.isnan(data)]).all()  # default restores missing input support

    # Transpose data AND every per-axis option; canonical role order is unchanged.
    order = (3, 0, 2, 1)                       # range, time, azimuth, elevation
    permuted_geometry = dict(geometry)
    for key in ("axis_types", "axis_coords", "kernel_type", "boundary"):
        permuted_geometry[key] = tuple(geometry[key][axis] for axis in order)
    permuted_mean = dct.dct_mean(
        data.transpose(order),
        tuple(widths[axis] for axis in order),
        coordinates="polar",
        restore_input_nan=False,
        **permuted_geometry,
    )
    np.testing.assert_allclose(
        permuted_mean, mean.transpose(order), rtol=1e-10, atol=1e-10, equal_nan=True
    )

    # Exclude time: even iterative filling and nearest fallback cannot borrow from
    # another time slice. Tagging an extra dimension "batch" also isolates it.
    independent = data.copy()
    independent[1] = np.nan
    filled = dct.dct_prefill(
        independent,
        widths,
        coordinates="polar",
        filter_axes=(1, 2, 3),
        **geometry,
    )
    assert np.isnan(filled[1]).all()
    assert np.isfinite(filled[[0, 2]]).all()
    valid = np.isfinite(independent)
    np.testing.assert_array_equal(filled[valid], independent[valid])

    print(f"Array order: (time, elevation, azimuth, range); shape: {data.shape}")
    print(f"Actual first range gate: {ranges[0]:.2f} km")
    print(f"Estimated {np.isnan(data).sum()} missing values with dct_mean")
    print(f"Nominal weighted support range: {support.min():.3f} to {support.max():.3f}")
    print("Axis permutation agrees; unfiltered all-missing time slice stays missing.")
    print("Support is not statistical ESS; smoothing is not an exact isotropic convolution.")


if __name__ == "__main__":
    main()
