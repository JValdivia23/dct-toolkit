# Local-statistics performance (v2.1.0)

These are measurements, not a guarantee that a nonlinear quantile costs one
linear convolution. The three evaluation methods use the same weighted
statistic, with explicit value quantization for histograms. Application accuracy
requirements must be checked along with elapsed time and memory.

## Reproduce

```bash
python examples/benchmark_order_stats.py --repeat 2
```

The script emits one JSON record per method/workload. Each method runs in a
fresh subprocess, performs one warm-up, then reports the minimum elapsed time
of two measured calls and that process's high-water RSS. Data generation is
deterministic. This run used Python 3.11.13, NumPy 2.4.4, SciPy 1.17.1 on macOS,
64 histogram bins, and a 16 MiB working-array budget. Peak RSS includes the
interpreter, imported libraries, allocator retention, outputs and FFT scratch;
it is not limited to the histogram workspace budget.

## Measurements

| Workload | Evaluation | Time (ms) | Peak RSS (MiB) |
|---|---|---:|---:|
| small_boxcar | `mean` | 2.622 | 81.0 |
| small_boxcar | `variance` | 6.998 | 88.7 |
| small_boxcar | `strict_variance` | 4.356 | 89.4 |
| small_boxcar | `median_histogram` | 59.565 | 189.2 |
| small_boxcar | `mad_histogram` | 199.652 | 225.6 |
| small_boxcar | `median_spatial` | 2.800 | 75.5 |
| small_boxcar | `mad_spatial` | 22.765 | 128.0 |
| large_boxcar | `mean` | 0.240 | 65.7 |
| large_boxcar | `variance` | 0.418 | 66.2 |
| large_boxcar | `strict_variance` | 0.217 | 66.6 |
| large_boxcar | `median_histogram` | 3.718 | 67.7 |
| large_boxcar | `mad_histogram` | 11.202 | 84.8 |
| large_boxcar | `median_exact` | 281.526 | 76.6 |
| large_boxcar | `mad_exact` | 548.139 | 80.0 |
| large_boxcar | `median_spatial` | 50.884 | 122.7 |
| large_boxcar | `mad_spatial` | 98.364 | 163.9 |
| polar | `mean` | 1.595 | 66.4 |
| polar | `variance` | 3.138 | 66.2 |
| polar | `strict_variance` | 0.878 | 66.0 |
| polar | `median_histogram` | 4.894 | 66.9 |
| polar | `mad_histogram` | 7.957 | 74.0 |
| polar | `median_exact` | 96.791 | 70.0 |
| polar | `mad_exact` | 140.844 | 72.7 |

Workloads:

- `small_boxcar`: 65,536 fully valid samples, discrete width 3.
- `large_boxcar`: 64 x 64, 15% missing, discrete widths 15 x 15.
- `polar`: 3 elevations x 32 azimuths x 16 ranges, 15% missing, physical angular
  widths, periodic azimuth, and a mixed discrete/Gaussian/discrete operator.

The mean/variance rows are existing spectral baselines. `strict_variance` uses
original first/second moments and mask in a cached batch. These workloads have
sufficient original support, so the existing completion fallback is not needed.
Histogram and spatial/exact rows are different evaluations of the declared
weighted quantile/MAD, not substitutes for a mean or variance.

## Interpretation

On this run, the histogram median was about 76x faster than the exact operator-row
reference for the large boxcar and 20x faster for physical polar geometry.
Histogram MAD was about 49x and 18x faster, respectively. These speedups include
value quantization; consult returned `error_bound` and validate the bins needed
for the application. Exact categorical histograms have no value quantization.

The compiled spatial median substantially outperformed histogram evaluation for
the small boxcar, with peak memory close to the mean baseline. For the broader,
masked 2D box, histogram evaluation was faster and used less peak RSS than tiled
spatial evaluation. Strict batched variance was about 1.9x faster than the
existing variance path on that 2D case and 3.6x faster on the polar case.

A histogram uses up to B indicator convolutions, so matching the cost of one
mean convolution is not a general promise. Geometry/response caching, transform
batching, reuse of the original mask response, and shared median/MAD bin masses
avoid repeated setup and redundant transforms. Explicit method selection keeps
these cost/accuracy choices reviewable. There is no automatic kernel-size or
multiscale selector in this milestone.

## Memory and storage

Quantiles stream bin masses and do not retain B full output fields. MAD needs a
shared distribution: small histograms stay in memory, while larger ones use a
temporary file of up to 8*B*N bytes and are evaluated in bounded working tiles.
The temporary file is closed/deleted even on errors. `max_memory_mb` controls
approximate algorithm workspace, not total RSS or a hard allocation ceiling.
Quantization encodings, sort buffers, metadata, outputs, allocator behavior and
FFT scratch must be budgeted too. Insufficient space for even one estimated
transform/row batch raises ValueError instead of allocating an unbounded batch.

The exact reference uses bounded batches rather than an N x N stored matrix,
but still requires quadratic operator-row work. It is intended for independent
small-array verification. Spatial evaluation is exact for fixed Cartesian
boxcars, including repeated boundary visits and center exclusion; its footprint
cost grows with requested width. The sampled Gaussian is never truncated at an
arbitrary spatial radius to improve a benchmark.
