"""Measure elapsed time and fresh-process peak RSS for local statistics.

Run after installing the checkout. Output is JSON; no optional runtime packages
are required. A fresh subprocess for each case/method prevents an earlier dense
reference allocation from contaminating another method's peak memory reading.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import scipy

from dct_toolkit import dct_mean, dct_variance, local_mad, local_median


def _case(name: str) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
    """Construct deterministic dense, sparse, and nonstationary polar workloads."""
    rng = np.random.default_rng(2201)
    if name == "small_boxcar":
        return rng.standard_normal(65536), 3., {"kernel_type": "boxcar_discrete"}
    if name == "large_boxcar":
        data = rng.standard_normal((64, 64))
        data[rng.random(data.shape) < .15] = np.nan
        return data, 15., {"kernel_type": "boxcar_discrete"}
    data = rng.standard_normal((3, 32, 16))
    data[rng.random(data.shape) < .15] = np.nan
    return data, (3., 5., 3.), {
        "coordinates": "polar", "axis_types": ("elevation", "azimuth", "range"),
        "axis_coords": (np.array([-15., 0., 15.]), np.arange(32) * 11.25,
                        np.arange(16) + 1.),
        "boundary": ("reflective", "periodic", "reflective"),
        "kernel_type": ("boxcar_discrete", "gaussian", "boxcar_discrete"),
    }


def _measure(case: str, method: str, repeat: int) -> Dict[str, Any]:
    """Time one algorithm and report its process-wide high-water resident memory."""
    data, width, options = _case(case)
    options["restore_input_nan"] = False
    if method == "mean":
        function = dct_mean
    elif method in ("variance", "strict_variance"):
        function = dct_variance
        if method == "strict_variance":
            options["fallback"] = "none"
    else:
        function = local_mad if method.startswith("mad") else local_median
        options.update(method=method.split("_")[1], bins=64, max_memory_mb=16)
    function(data, width, **options)
    durations = []
    for _ in range(repeat):
        start = time.perf_counter()
        function(data, width, **options)
        durations.append(time.perf_counter() - start)
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mb = rss / (1024**2 if sys.platform == "darwin" else 1024)
    except ImportError:
        peak_mb = None
    return {"case": case, "method": method, "shape": data.shape,
            "seconds": min(durations), "peak_rss_mb": peak_mb,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "python": sys.version.split()[0]}


def main() -> None:
    """Run isolated benchmarks without assuming every backend supports every operator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--child", nargs=2, metavar=("CASE", "METHOD"))
    arguments = parser.parse_args()
    if arguments.repeat < 1:
        parser.error("repeat must be positive")
    if arguments.child:
        print(json.dumps(_measure(*arguments.child, arguments.repeat)))
        return
    for case in ("small_boxcar", "large_boxcar", "polar"):
        methods = ["mean", "variance", "strict_variance", "median_histogram", "mad_histogram"]
        if case != "small_boxcar":
            methods += ["median_exact", "mad_exact"]
        if case != "polar":
            methods += ["median_spatial", "mad_spatial"]
        for method in methods:
            output = subprocess.check_output([
                sys.executable, str(Path(__file__).resolve()), "--repeat", str(arguments.repeat),
                "--child", case, method,
            ], text=True)
            print(output.strip(), flush=True)


if __name__ == "__main__":
    main()
