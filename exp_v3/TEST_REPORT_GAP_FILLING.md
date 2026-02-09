# Test Report: Gap Filling

**Date:** 2026-02-09
**Grid:** 720 x 1000 (azimuth x range)
**Azimuth Resolution:** 0.5 deg
**Smoothing Width:** 5.0 pixels
**Algorithm:** Linear interpolation initialization + iterative DCT diffusion
**Timing Repeats:** 1 (median)

## Summary

This test suite evaluates the performance, accuracy, and robustness of the
`iterative_gap_fill` algorithm (with linear interpolation initialization) against three
standard baselines:

1. **1D Linear Interpolation**: Along-azimuth `np.interp` (fast, ignores 2D structure).
2. **2D Linear Interpolation**: Unstructured `scipy.interpolate.griddata` (accurate, slow).
3. **Astropy Gaussian**: Convolution with `astropy.convolution` (standard, fails on large holes).

Results are saved to `exp_v3/gap_filling_results.csv`.

## Criteria Evaluation

Evaluation against the 5 success criteria from the gap filling test plan:

| # | Criterion | Verdict | Evidence |
| - | :--- | :---: | :--- |
| 1 | Beat Astropy on smooth; beat 1D Linear on mixed fields (<50% gaps) | **PASS** | DCT MAE=0.000024, Astropy MAE=0.001043 (random_30, smooth); DCT MAE=0.000386, 1D Linear MAE=0.003233 (random_30, mixed) |
| 2 | Not crash or diverge for 70% gaps | **PASS** | MAE=0.000052, coverage=1.000000 |
| 3 | Faster than griddata for N>=128 | **PASS** | DCT=111ms vs griddata=26120ms (234.4x speedup) |
| 4 | Error monotonically decreases with iterations | **PASS** | Validated by `test_monotonic_convergence` pytest |
| 5 | Valid data preserved exactly (0.0 difference) | **PASS** | Validated by `test_preservation_of_valid_data` pytest |

## Key Findings

### 1. Initialization: Linear vs Multi-Scale

Linear interpolation initialization preserves spatial gradients across holes from the first iterate.
This provides a better starting point for iterative DCT refinement compared to the multi-scale cascade,
which tends to fill large holes with near-constant values (global average).

### 2. Accuracy on Random Gaps

For smooth, band-limited fields with random gaps:
- **2D Griddata** achieves the lowest MAE (exact linear interpolation).
- **Iterative DCT** outperforms **Astropy** on smooth fields.
- On **mixed-frequency** fields, DCT outperforms 1D Linear because the 2D smoothing captures structure that 1D azimuth interpolation misses.

### 3. Computational Speed

On a 720x1000 grid (random_30, smooth field):
- **Iterative DCT**: ~111 ms
- **2D Griddata**: ~26120 ms
- Iterative DCT is **~234x faster** than griddata.

### 4. Limitations

- **Width must match hole scale**: For large contiguous holes, the smoothing width should be a significant fraction of the hole diameter for iterative DCT to improve on linear init.
- **Edge preservation**: DCT acts as a low-pass filter and will smooth sharp edges.

## Detailed Results

### Dataset: smooth

| Gap Scenario | Method | MAE | RMSE | Max Error | Coverage | Time (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| random_10 | linear_1d_az | 0.000033 | 0.000054 | 0.000866 | 1.000000 | 25.846583 |
| random_10 | linear_2d_griddata | 0.000003 | 0.000010 | 0.001031 | 1.000000 | 38982.863042 |
| random_10 | astropy_gaussian | 0.000891 | 0.010893 | 0.265073 | 1.000000 | 158.406417 |
| random_10 | iterative_dct | 0.000017 | 0.000051 | 0.001342 | 1.000000 | 53.473750 |
| random_30 | linear_1d_az | 0.000054 | 0.000105 | 0.002930 | 1.000000 | 28.882125 |
| random_30 | linear_2d_griddata | 0.000005 | 0.000014 | 0.001296 | 1.000000 | 26119.663125 |
| random_30 | astropy_gaussian | 0.001043 | 0.010581 | 0.329731 | 1.000000 | 328.832292 |
| random_30 | iterative_dct | 0.000024 | 0.000062 | 0.001617 | 1.000000 | 111.448083 |
| random_50 | linear_1d_az | 0.000106 | 0.000238 | 0.005585 | 1.000000 | 35.297583 |
| random_50 | linear_2d_griddata | 0.000008 | 0.000025 | 0.003350 | 1.000000 | 19433.026375 |
| random_50 | astropy_gaussian | 0.001236 | 0.010863 | 0.399832 | 1.000000 | 475.766125 |
| random_50 | iterative_dct | 0.000030 | 0.000072 | 0.003701 | 1.000000 | 111.484542 |
| random_70 | linear_1d_az | 0.000293 | 0.000757 | 0.031216 | 1.000000 | 28.725750 |
| random_70 | linear_2d_griddata | 0.000016 | 0.000040 | 0.004527 | 1.000000 | 9707.576167 |
| random_70 | astropy_gaussian | 0.001581 | 0.011750 | 0.530192 | 1.000000 | 296.349458 |
| random_70 | iterative_dct | 0.000052 | 0.000118 | 0.009307 | 1.000000 | 144.725250 |
| hole_circular | linear_1d_az | 0.030737 | 0.035412 | 0.062595 | 1.000000 | 16.974334 |
| hole_circular | linear_2d_griddata | 0.016155 | 0.018463 | 0.032647 | 1.000000 | 36572.437416 |
| hole_circular | astropy_gaussian | 0.006683 | 0.007800 | 0.021440 | 0.197580 | 98.624959 |
| hole_circular | iterative_dct | 0.031417 | 0.036572 | 0.065174 | 1.000000 | 58.657875 |
| hole_rect | linear_1d_az | 0.042816 | 0.049595 | 0.104130 | 1.000000 | 24.632208 |
| hole_rect | linear_2d_griddata | 0.015863 | 0.019344 | 0.036714 | 1.000000 | 35701.846209 |
| hole_rect | astropy_gaussian | 0.006603 | 0.007931 | 0.023445 | 0.194461 | 87.568917 |
| hole_rect | iterative_dct | 0.034950 | 0.041965 | 0.076703 | 1.000000 | 41.575583 |
| sector_10deg | linear_1d_az | 0.001492 | 0.002174 | 0.005571 | 1.000000 | 16.860750 |
| sector_10deg | linear_2d_griddata | 0.000995 | 0.001440 | 0.003918 | 1.000000 | 34229.651458 |
| sector_10deg | astropy_gaussian | 0.016555 | 0.022849 | 0.204241 | 0.600000 | 90.494833 |
| sector_10deg | iterative_dct | 0.001495 | 0.002175 | 0.005571 | 1.000000 | 64.116875 |

### Dataset: mixed

| Gap Scenario | Method | MAE | RMSE | Max Error | Coverage | Time (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| random_10 | linear_1d_az | 0.002062 | 0.003444 | 0.082299 | 1.000000 | 25.171750 |
| random_10 | linear_2d_griddata | 0.000078 | 0.000134 | 0.007067 | 1.000000 | 30902.193833 |
| random_10 | astropy_gaussian | 0.004325 | 0.012244 | 0.281321 | 1.000000 | 158.072291 |
| random_10 | iterative_dct | 0.000294 | 0.000388 | 0.007392 | 1.000000 | 109.597459 |
| random_30 | linear_1d_az | 0.003233 | 0.005962 | 0.105959 | 1.000000 | 29.098209 |
| random_30 | linear_2d_griddata | 0.000123 | 0.000225 | 0.013599 | 1.000000 | 23201.010791 |
| random_30 | astropy_gaussian | 0.004492 | 0.012040 | 0.338585 | 1.000000 | 331.178958 |
| random_30 | iterative_dct | 0.000386 | 0.000513 | 0.012668 | 1.000000 | 147.678834 |
| random_50 | linear_1d_az | 0.005791 | 0.011338 | 0.163086 | 1.000000 | 32.936542 |
| random_50 | linear_2d_griddata | 0.000220 | 0.000439 | 0.027655 | 1.000000 | 15784.175375 |
| random_50 | astropy_gaussian | 0.004869 | 0.012510 | 0.426895 | 1.000000 | 474.817167 |
| random_50 | iterative_dct | 0.000556 | 0.000776 | 0.041208 | 1.000000 | 196.128459 |
| random_70 | linear_1d_az | 0.011671 | 0.021234 | 0.189134 | 1.000000 | 28.350417 |
| random_70 | linear_2d_griddata | 0.000485 | 0.001059 | 0.090075 | 1.000000 | 9353.544417 |
| random_70 | astropy_gaussian | 0.005657 | 0.013822 | 0.548276 | 1.000000 | 296.475916 |
| random_70 | iterative_dct | 0.001069 | 0.001540 | 0.070375 | 1.000000 | 354.566500 |
| hole_circular | linear_1d_az | 0.053576 | 0.071708 | 0.213526 | 1.000000 | 16.830458 |
| hole_circular | linear_2d_griddata | 0.048387 | 0.060148 | 0.153096 | 1.000000 | 35772.632875 |
| hole_circular | astropy_gaussian | 0.016596 | 0.022919 | 0.078881 | 0.197580 | 89.101042 |
| hole_circular | iterative_dct | 0.073815 | 0.091834 | 0.206747 | 1.000000 | 85.147625 |
| hole_rect | linear_1d_az | 0.058823 | 0.073906 | 0.213827 | 1.000000 | 17.258458 |
| hole_rect | linear_2d_griddata | 0.046068 | 0.059835 | 0.176398 | 1.000000 | 32766.617292 |
| hole_rect | astropy_gaussian | 0.016971 | 0.023073 | 0.076578 | 0.194461 | 95.712916 |
| hole_rect | iterative_dct | 0.064943 | 0.084872 | 0.206440 | 1.000000 | 78.068875 |
| sector_10deg | linear_1d_az | 0.037395 | 0.052438 | 0.153579 | 1.000000 | 17.383833 |
| sector_10deg | linear_2d_griddata | 0.037410 | 0.052688 | 0.153128 | 1.000000 | 33160.486166 |
| sector_10deg | astropy_gaussian | 0.040645 | 0.056209 | 0.324771 | 0.600000 | 88.860083 |
| sector_10deg | iterative_dct | 0.037383 | 0.052402 | 0.153440 | 1.000000 | 53.509083 |

### Dataset: edge

| Gap Scenario | Method | MAE | RMSE | Max Error | Coverage | Time (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| random_10 | linear_1d_az | 0.000184 | 0.000896 | 0.030933 | 1.000000 | 22.042500 |
| random_10 | linear_2d_griddata | 0.000002 | 0.000012 | 0.000562 | 1.000000 | 29258.142458 |
| random_10 | astropy_gaussian | 0.001010 | 0.009821 | 0.240803 | 1.000000 | 156.959958 |
| random_10 | iterative_dct | 0.000071 | 0.000240 | 0.004147 | 1.000000 | 98.692208 |
| random_30 | linear_1d_az | 0.000303 | 0.001705 | 0.086352 | 1.000000 | 28.705666 |
| random_30 | linear_2d_griddata | 0.000004 | 0.000027 | 0.005755 | 1.000000 | 23990.449583 |
| random_30 | astropy_gaussian | 0.001093 | 0.009639 | 0.290391 | 1.000000 | 326.700291 |
| random_30 | iterative_dct | 0.000083 | 0.000277 | 0.004610 | 1.000000 | 107.712250 |
| random_50 | linear_1d_az | 0.000595 | 0.003772 | 0.181960 | 1.000000 | 33.047042 |
| random_50 | linear_2d_griddata | 0.000008 | 0.000080 | 0.025472 | 1.000000 | 16918.831500 |
| random_50 | astropy_gaussian | 0.001244 | 0.010066 | 0.358447 | 1.000000 | 475.675625 |
| random_50 | iterative_dct | 0.000110 | 0.000350 | 0.005383 | 1.000000 | 151.254000 |
| random_70 | linear_1d_az | 0.001488 | 0.009049 | 0.296539 | 1.000000 | 28.626834 |
| random_70 | linear_2d_griddata | 0.000023 | 0.000208 | 0.063998 | 1.000000 | 10076.637334 |
| random_70 | astropy_gaussian | 0.001533 | 0.011041 | 0.458900 | 1.000000 | 295.580042 |
| random_70 | iterative_dct | 0.000174 | 0.000510 | 0.007009 | 1.000000 | 236.476875 |
| hole_circular | linear_1d_az | 0.000307 | 0.000498 | 0.001468 | 1.000000 | 17.088625 |
| hole_circular | linear_2d_griddata | 0.000370 | 0.000597 | 0.001635 | 1.000000 | 34391.436500 |
| hole_circular | astropy_gaussian | 0.000282 | 0.000538 | 0.001786 | 0.197580 | 87.544792 |
| hole_circular | iterative_dct | 0.002126 | 0.002492 | 0.004623 | 1.000000 | 40.313292 |
| hole_rect | linear_1d_az | 0.001208 | 0.002089 | 0.005934 | 1.000000 | 17.328250 |
| hole_rect | linear_2d_griddata | 0.001010 | 0.001489 | 0.003143 | 1.000000 | 34987.670375 |
| hole_rect | astropy_gaussian | 0.000440 | 0.000807 | 0.002529 | 0.194461 | 88.582875 |
| hole_rect | iterative_dct | 0.002479 | 0.002844 | 0.004664 | 1.000000 | 44.172417 |
| sector_10deg | linear_1d_az | 0.106569 | 0.140452 | 0.336914 | 1.000000 | 17.058333 |
| sector_10deg | linear_2d_griddata | 0.106547 | 0.141207 | 0.587568 | 1.000000 | 34029.391792 |
| sector_10deg | astropy_gaussian | 0.103421 | 0.162609 | 0.504618 | 0.600000 | 87.869750 |
| sector_10deg | iterative_dct | 0.106569 | 0.140451 | 0.336854 | 1.000000 | 52.394667 |
