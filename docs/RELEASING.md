# Release procedure

Use this procedure for the `1.0.0` release and subsequent updates. Changing
version strings does not publish a release.
Do not tag, push, upload to PyPI/conda-forge, or create an archival record without
explicit approval.

## Before building

1. Run the complete test suite, including the N-D geometry and missing-data tests.
2. Obtain an independent implementation and documentation review; resolve its
   findings and rerun affected tests.
3. Check the documented examples and compatibility of existing 2D calls.
4. Verify that `pyproject.toml` and `dct_toolkit.__version__` both report `1.0.0`.
5. Inspect `git diff --check` and the intended source changes. A dirty development
   checkout is not an immutable source identifier for manuscript computations.

### Refresh editable installation metadata

After changing the version, reinstall the editable package instead of relying on
its source import to update distribution metadata:

```bash
conda run -n dct-publish python -m pip install --no-deps -e .
conda run -n dct-publish python -c 'from importlib.metadata import version; import dct_toolkit as dct; assert dct.__version__ == version("dct-toolkit") == "1.0.0"'
```

Run the suite with both current dependencies and the declared minimum versions:
Python 3.8, NumPy 1.20.0, and SciPy 1.8.0. CI checks both dependency configurations
against an installed package, from outside the source checkout.

## Build and verify artifacts

Use a dedicated build environment, not the general development environment:

```bash
python -m build --outdir dist/1.0.0
python -m twine check dist/1.0.0/*.whl dist/1.0.0/*.tar.gz
```

Install the wheel into a separate test environment and run from outside the
source checkout. The following inline smoke test verifies that the imported
package comes from the environment, not a source-tree import override:

```python
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np
import dct_toolkit as dct

module = Path(dct.__file__).resolve()
module.relative_to(Path(sys.prefix).resolve())  # raises if outside the environment
assert "site-packages" in module.parts, module
assert dct.__version__ == version("dct-toolkit") == "1.0.0"
print("Testing installed artifact:", module)

data = np.ones((2, 8, 4))  # elevation, azimuth, range
mean = dct.dct_mean(
    data, (50.0, 100.0, 50.0), coordinates="polar",
    axis_types=("elevation", "azimuth", "range"),
    axis_coords=(np.array([-5.0, 5.0]), np.arange(8) * 45.0,
                 100.0 + np.arange(4) * 25.0),
    boundary=("reflective", "periodic", "reflective"),
    kernel_type=("gaussian", "boxcar_discrete", "gaussian"),
)
np.testing.assert_allclose(mean, data)
```

After this check, run `examples/nd_polar.py` against that same installed package;
the example deliberately does not modify `sys.path`. Also test a wheel built from
the source archive, so missing source-distribution files cannot be hidden by the
working tree.

Record SHA-256 digests of the exact wheel and source archive. Rebuild and update
digests after any change to a packaged file, including documentation or tests.

## Conda recipes

The two recipes under `conda.recipe/` must use the same version and the SHA-256
of the exact source archive intended for publication. A checksum from v0.6.0
must never be reused for v1.0.0. The PyPI source URL is a publication destination,
not evidence that the archive already exists there. Publish that exact archive
before submitting a recipe that downloads it.

## Publish an approved release

1. Commit the reviewed software, tests, documentation, and recipe changes. Merge
   to `main` without discarding unrelated work, then push the approved source and
   annotated `v1.0.0` tag. Preserve all historical tags.
2. Upload only the validated artifacts, using a project-scoped PyPI token stored
   securely or a configured trusted publisher:

   ```bash
   python -m twine upload dist/1.0.0/dct_toolkit-1.0.0-py3-none-any.whl dist/1.0.0/dct_toolkit-1.0.0.tar.gz
   ```

   Never put tokens in source files, shell command arguments, logs, or chat.
3. Verify the PyPI version and both published SHA-256 values against the local
   artifacts. Test a fresh `pip install dct-toolkit==1.0.0` outside the checkout.
4. Create a pull request in `conda-forge/dct-toolkit-feedstock` updating the
   version and the SHA-256 of that exact PyPI source archive. Keep the build
   number at zero for a new upstream version; follow feedstock CI/rerender rules.
5. Report the feedstock PR separately from conda channel availability. A submitted
   or merged PR is not proof that the new package is already downloadable.
   Verify the conda-forge package index after its build/upload completes.

PyPI uploads are immutable: do not rebuild or replace files under an already
published version. If a packaged file needs changing afterward, prepare a new
version. A GitHub release may attach the same validated artifacts and checksum
manifest, but must not substitute a different archive for the one on PyPI.

## Reproducibility handoff

Before using v1.0.0 for final paper computations, record:

- the approved immutable source commit and whether the checkout is clean;
- the exact installed artifact and its digest;
- runtime and distribution versions, NumPy/SciPy versions, and execution settings;
- resolved axis roles, coordinate vectors/units, filtering selection, kernels,
  boundaries, widths, and fallback settings.

Require the exact chosen release or artifact rather than a minimum-version check.
Keep results generated with earlier versions identifiable as historical results;
do not relabel them as v1.0.0. Updating manuscripts and review plans is a separate
task after software validation, not part of this release preparation.
