# Publication Guide (pip + conda-forge)

This guide describes how to publish `dct-toolkit` for both pip and conda users.

## 1) Pre-release checklist

- `python -m pytest tests -q` passes.
- `python -m pip install .` succeeds in a clean environment.
- `LICENSE` exists (MIT).
- `pyproject.toml` metadata is correct.
- `README.md` install instructions match reality.

## 2) Build and validate package artifacts

From repository root:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

You should see both sdist and wheel in `dist/` and a successful twine check.

## 3) Publish to PyPI

1. Create or log in to your account at https://pypi.org.
2. Create an API token in account settings.
3. Upload artifacts:

```bash
python -m twine upload dist/*
```

After upload, verify installation in a fresh env:

```bash
python -m pip install dct-toolkit
python -c "import dct_toolkit as d; print(d.__version__)"
```

## 4) Prepare conda-forge submission

The starter recipe is in `conda.recipe/meta.yaml`.

Before submitting, update:

- `source.url` to the real PyPI tarball URL.
- `source.sha256` using:

```bash
python -m pip download --no-binary :all: dct-toolkit==0.4.0 -d /tmp
python - <<'PY'
import hashlib
from pathlib import Path
p = next(Path('/tmp').glob('dct_toolkit-*.tar.gz'))
h = hashlib.sha256(p.read_bytes()).hexdigest()
print(h)
PY
```

## 5) Submit to conda-forge

1. Fork `conda-forge/staged-recipes`.
2. Add recipe under `recipes/dct-toolkit/meta.yaml`.
3. Open PR to `conda-forge/staged-recipes`.
4. Address bot/reviewer comments.
5. After merge, conda-forge creates `dct-toolkit-feedstock`.

Future version updates are usually handled by a bot PR in the feedstock.

## 6) Recommended release order

1. Tag release in GitHub.
2. Publish to PyPI.
3. Submit conda-forge recipe.
