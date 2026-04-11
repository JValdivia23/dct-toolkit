# AGENTS.md - DCT Toolkit Agent Guidelines

**Project**: dct-toolkit  
**Version**: 0.4.0  
**Status**: Published — PyPI live, conda-forge PR submitted  
**Last Updated**: 2026-04-11

---

## Critical Rule: Standalone Project

⚠️ **THIS IS A COMPLETELY STANDALONE PROJECT**

- ✅ ONLY import from within `dct_toolkit/dct_toolkit/` directory
- ✅ External dependencies: `numpy`, `scipy` ONLY
- ❌ NEVER import from parent directories (`../src/`, `../dct_stats/`, legacy code)
- ❌ NEVER import from any code outside this repository
- ❌ NEVER assume other projects are available

**Verification Command**:
```bash
grep -r "^from \.\./" dct_toolkit/
grep -r "^import .*\.\." dct_toolkit/
```
Should return NO matches.

---

## Environment Setup

### Conda Environment
```bash
conda activate myenv
python --version  # Should be 3.8+
```

### PYTHONPATH
When running tests or examples from project root:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/dct_toolkit
```

### Isolated Conda Environment for Testing/Building (Recommended)
To avoid dependency conflicts when building or testing the package, use an isolated conda environment:

```bash
# Create isolated environment
conda create -n dct-toolkit-dev python=3.11 -y
conda activate dct-toolkit-dev

# Install package in development mode
pip install -e .

# Run tests
python -m pytest tests -q

# Build package artifacts (for PyPI)
conda install -c conda-forge build twine -y
python -m build
python -m twine check dist/*

# Clean up when done
conda deactivate
```

**Why isolated?** Building tools (`build`, `twine`) and their dependencies can conflict with other packages in your base environment (e.g., Sphinx, Streamlit). Always use an isolated environment for package operations.

---

## Code Style - Scientific Repository Standards

### Docstrings (REQUIRED)
Use NumPy style:
```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Short description.
    
    Longer description if needed.
    
    Parameters
    ----------
    param1 : type
        Description
    param2 : type
        Description
        
    Returns
    -------
    return_type
        Description
        
    Examples
    --------
    >>> result = function_name(1.0, 2.0)
    """
```

### Type Hints (REQUIRED)
All function parameters and returns must be typed:
```python
def smooth_data(data: np.ndarray, width: float) -> np.ndarray:
    ...
```

### Naming Conventions
- **Functions/Variables**: `snake_case` (e.g., `get_transfer_function`)
- **Classes**: `CamelCase` (e.g., `DCTSmoother`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_KERNEL`)
- **Private**: `_leading_underscore` (e.g., `_helper_func`)

### Formatting
- **Indentation**: 4 spaces (NO tabs)
- **Line Length**: 100 characters (soft limit)
- **Blank Lines**: 
  - 2 lines between top-level functions/classes
  - 1 line between methods
- **Operators**: Spaces around operators (`x = y + 1`)

### Import Organization
```python
# 1. Standard library
import os
import sys
from typing import Tuple, Optional

# 2. Third-party (numpy, scipy)
import numpy as np
import scipy.fft

# 3. Local modules (relative imports)
from .core import get_dct_transfer_function
from .stats import dct_mean
```

---

## Project Structure

```
dct_toolkit/                    # PROJECT ROOT
│
├── dct_toolkit/               # INSTALLABLE PACKAGE ⚠️ CODE GOES HERE
│   ├── __init__.py           # Public API exports
│   ├── core.py               # Transfer functions & 1D primitives
│   ├── cartesian.py          # 2D Cartesian separable smoothing
│   ├── polar.py              # 2D Polar smoothing (adaptive kernels)
│   ├── stats.py              # Statistical ops (Normalized Convolution)
│   └── gap_filling.py         # Gap-filling methods (deferred for public stats-first release)
│
├── tests/                     # TEST SUITE ⚠️ TESTS GO HERE
│   ├── test_core.py
│   ├── test_cartesian.py
│   ├── test_polar.py
│   ├── test_stats.py
│   └── test_gap_filling.py
│
├── examples/                  # USAGE EXAMPLES
│   ├── basic_polar.py
│   └── comprehensive_demo.py
│
├── exp_v3/                    # EXPERIMENTS (reports + code)
│   ├── figures/               # Generated figures (PNG/PDF)
│   ├── test_width_impact.py
│   ├── plot_gap_filling_results.py
│   ├── GAP_FILLING_GUIDE.md
│   ├── TEST_REPORT_GAP_FILLING.md
│   ├── TEST_REPORT_CORE.md
│   └── gap_filling_results.csv
│
├── docs/                      # DOCUMENTATION
│   ├── API_REFERENCE.md
│   ├── MATHEMATICAL_BASIS.md
│   └── GAP_FILLING_BASIS.md
│
├── README.md                  # Main documentation
├── CHANGELOG.md               # Version history
└── AGENTS.md                  # This file
```

---

## Development Workflow

### Before Making Changes

1. **Activate Environment**
   ```bash
   conda activate myenv
   ```

2. **Verify Baseline**
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)/dct_toolkit
   python -m pytest tests/ -v
   ```
   All tests in `tests/` must pass before you start.

3. **Review Documentation**
   - Check `docs/MATHEMATICAL_BASIS.md` for theory
   - Check `docs/GAP_FILLING_BASIS.md` for gap filling theory
   - Check `docs/API_REFERENCE.md` for existing API

### Making Changes

1. **Edit Code**
   - Modify appropriate module in `dct_toolkit/dct_toolkit/`
   - Follow code style guidelines
   - Add docstrings and type hints

2. **Add Tests**
   - Add/update tests in `dct_toolkit/tests/`
   - Tests should validate correctness with synthetic data
   - Include edge cases (empty arrays, all NaN, etc.)

3. **Update Documentation**
   - Update `API_REFERENCE.md` if API changes
   - Update `CHANGELOG.md` with changes
   - Update this file if workflow changes

### Testing (MANDATORY)

**Run Tests**:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/dct_toolkit
python -m pytest tests/ -v
```

**Expected Output**: all tests passing

**If Tests Fail**:
- Fix the code, not the tests
- Tests represent ground truth requirements
- Never commit with failing tests

### Pre-Completion Checklist

Before declaring task complete:

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] No imports from outside `dct_toolkit/`
- [ ] All public functions have NumPy-style docstrings
- [ ] All functions have type hints
- [ ] Code follows naming conventions
- [ ] No tabs, 4-space indentation
- [ ] Line length ≤ 100 characters
- [ ] CHANGELOG.md updated
- [ ] Documentation updated if API changed

---

## Current Progress & Roadmap

### ✅ Completed (v0.4.0)
- [x] Core primitives (`core.py`)
  - Transfer functions (boxcar, boxcar_discrete, gaussian)
  - 1D DCT convolution
- [x] Cartesian smoothing (`cartesian.py`)
  - Separable N-D smoothing
- [x] Polar smoothing (`polar.py`)
  - Adaptive azimuth kernels
  - Boundary conditions (reflective/periodic)
- [x] Statistical operations (`stats.py`)
  - Normalized Convolution
  - dct_count, dct_mean, dct_prefill, dct_variance, dct_std
- [x] Gap-filling module (`gap_filling.py`)
  - iterative diffusion fill (`iterative_gap_fill`)
  - DCT-PLS inpainting (`dct_inpaint`)
- [x] Test suite (`tests/`) passing
- [x] Documentation suite

### ✅ Completed (Publication Readiness)
- [x] Fix immediate correctness issues in core statistics path (integer dtype bug)
- [x] Create dedicated publication-prep branches
- [x] Update `AGENTS.md` / docs to reflect publication plan
- [x] Build clean public-facing scope focused on convolution/statistics
- [x] Add packaging metadata (`pyproject.toml`)
- [x] Add `LICENSE` (MIT) and align README badges/claims
- [x] Define public API surface (exclude gap filling for initial public release)
- [x] Remove experimental assets from public-facing surface
- [x] Add CI for tests (`.github/workflows/tests.yml`)
- [x] Publish to PyPI (v0.4.0 live at https://pypi.org/project/dct-toolkit/)
- [x] Submit conda-forge recipe (PR #32930 in review)
- [x] Merge publication work to `main` branch
- [x] Tag `v0.4.0` release
- [x] Create `research/gap-filling` branch for long-term research

### 🔭 Future (v0.5.0 and beyond)
- [ ] 3D support (volumetric data)
- [ ] Additional kernels (Savitzky-Golay, Hanning)
- [ ] Performance optimizations (numba?)
- [ ] Jupyter notebook examples
- [ ] Advanced gap filling (constrained optimization)
- [ ] GPU acceleration (CuPy)

---

## Key Technical Details

### Transfer Functions
- **Boxcar (Analytical)**: `H[k] = sin(W·θ)/(W·sin(θ))`, θ = πk/(2n)
- **Boxcar (Discrete)**: Sum of cosines for discrete kernels
- **Gaussian**: `H[k] = exp(-0.5·(ω·σ)²)`, σ = width/√12

### Normalized Convolution
Formula for robust mean with gaps:
```
μ = smooth(data * mask) / (smooth(mask) + ε)
```

### Polar Coordinates
- **Data Layout**: (n_azimuth, n_range)
- **Adaptive Width**: w_az(r) = width / (r · dθ)
- **Boundary Conditions**:
  - Reflective: DCT-II (even symmetry)
  - Periodic: Real FFT (circular convolution)

### Testing Philosophy
- Synthetic data with known ground truth
- Verify theoretical properties (DC preservation, linearity)
- Edge case coverage (all NaN, single point, etc.)

---

## Common Pitfalls & Solutions

### 1. Import Errors
**Problem**: `ModuleNotFoundError: No module named 'dct_toolkit'`
**Solution**: 
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/dct_toolkit
```

### 2. Path Confusion
**Problem**: Tests run from wrong directory
**Solution**: Always run from project root (`dct_toolkit/`)

### 3. Polar Boundary Issues
**Problem**: Discontinuity at 360°/0°
**Solution**: Use `az_boundary='periodic'` for azimuth

### 4. Test Failures
**Problem**: Tests fail after changes
**Solution**: Fix the code, don't modify tests to pass

### 5. Legacy Code Contamination
**Problem**: Accidentally importing from `../src/`
**Solution**: Check with grep command in Critical Rule section

---

## Repository Management

### Branch Strategy

We use a **trunk-based workflow** with dedicated branches for different purposes:

| Branch | Purpose | Stability |
|--------|---------|-----------|
| `main` | Public releases, stable code | **Production-ready** |
| `research/gap-filling` | Long-term gap-filling research | Experimental |
| `feature/*` | Short-lived feature development | In-progress |

#### `main` Branch (The Public Package)
- Always matches the latest PyPI release
- Contains only public API: core, cartesian, polar, stats
- Gap-filling code exists in `gap_filling.py` but is **not exported** in `__init__.py`
- All tests must pass before merging

#### `research/gap-filling` Branch
- Dedicated branch for long-term gap-filling research
- Safe to experiment with `dct_inpaint`, `iterative_gap_fill`
- Regularly merge `main` into this branch to get bug fixes and new features
- Not merged back to `main` until research matures

**To sync research branch with latest main:**
```bash
git checkout research/gap-filling
git merge main
```

#### Feature Branches
For new development (Hanning window, 3D support, notebooks):
```bash
# Create feature branch from main
git checkout main
git checkout -b feature/hanning-window

# Work, commit, push
git push origin feature/hanning-window

# When done, merge to main via PR
git checkout main
git merge feature/hanning-window
```

### Repository Status
- GitHub: Private (core team access only)
- PyPI: **Public** — `pip install dct-toolkit` works
- conda-forge: **Submitted** — PR #32930 in review at `conda-forge/staged-recipes`

### Current Version
- `v0.4.0` — Public stats-first release
- Tag: `git checkout v0.4.0`

### Gap-Filling Status
- Code exists in `dct_toolkit/gap_filling.py` (available for import directly)
- **Not exported** in public API (`__init__.py`) — research-internal
- Full gap-filling to be released in future v1.x after research matures

---

## Emergency Contacts & References

### Documentation
- **Math Theory**: `docs/MATHEMATICAL_BASIS.md`
- **Gap Filling Basis**: `docs/GAP_FILLING_BASIS.md`
- **API Details**: `docs/API_REFERENCE.md`
- **Core Test Report**: `exp_v3/TEST_REPORT_CORE.md`
- **Gap Filling Report**: `exp_v3/TEST_REPORT_GAP_FILLING.md`

### Code Examples
- **Polar Smoothing**: `examples/basic_polar.py`
- **Full Demo**: `examples/comprehensive_demo.py`
- **Gap Filling**: `exp_v3/test_width_impact.py`

### Key Papers & References
- Garcia (2010): DCT-PLS (contrast/background)
- Knutsson et al.: Normalized Convolution
- Ahmed et al.: DCT fundamentals
- Lecture 17 Notes: Objective Mapping / Kriging (for context)

---

## Quick Reference Card

```bash
# Install (pip)
pip install dct-toolkit

# Install (conda - once feed stock is merged)
conda install -c conda-forge dct-toolkit

# Development install from source
git clone https://github.com/JValdivia23/dct-toolkit.git
cd dct-toolkit
pip install -e .

# Test
python -m pytest tests -v

# Run example
python examples/basic_polar.py

# Check imports (should be empty)
grep -r "^from \.\./" dct_toolkit/dct_toolkit/
```

---

**Remember**: This is a standalone scientific package. Quality, correctness, and clarity are more important than speed. When in doubt, refer to the test suite - it defines the expected behavior.
