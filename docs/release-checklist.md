# Release Checklist

This checklist defines the local gates required before tagging a `protrepair`
release.

## Required Verification

Install release-constrained dependencies before running release gates:

```bash
python -m pip install -c constraints/release.txt ".[dev,refinement]"
```

Run the permanent code-quality and unit surface:

```bash
python -m ruff check
python -m basedpyright
python -m pytest tests/unit -q
```

Run the release-facing API and documentation smoke surface:

```bash
python -m pytest \
  tests/unit/test_release_public_api_contract.py \
  tests/unit/test_readme_usage.py \
  tests/unit/test_release_artifacts.py \
  -q
```

Run the lean optional-backend retained-ligand gate without RDKit:

```bash
rdkit_blocker="$(mktemp -d)"
printf 'raise ModuleNotFoundError("No module named rdkit")\n' > "$rdkit_blocker/rdkit.py"
python -m pip install -c constraints/release.txt ".[dev]"
PYTHONPATH="${rdkit_blocker}:${PYTHONPATH:-}" python -m pytest \
  tests/unit/test_retained_non_polymer_no_rdkit_release.py \
  -q
```

## Build And Install Smoke

Run the installed-wheel functional smoke from a clean worktree:

```bash
python scripts/run_installed_wheel_smoke.py
```

Run the optional RDKit refinement smoke before releases that advertise the
`refinement` extra:

```bash
python scripts/run_installed_wheel_smoke.py --with-refinement
```

The script builds a wheel with `hatchling`, installs it into a temporary
virtual environment, and verifies installed-package imports, bundled chemistry
resources, coordinate read/write, `process_structure()`, packaged FASPR
execution, and optional RDKit local refinement. It installs the wheel under
`constraints/release.txt` by default.

Bundled FASPR assets are an installed-package/wheel contract. Source-tree
execution may use an explicit FASPR `executable_path`, but release verification
should not rely on speculative build-directory probing.

## Artifact Contents

Confirm that the wheel contains these non-code assets:

- `protrepair/chemistry/resources/nonstandard_components.json.gz`
- `protrepair/chemistry/resources/retained_non_polymer_components.json.gz`
- `protrepair/packing/faspr/bin/FASPR`
- `protrepair/packing/faspr/bin/dun2010bbdep.bin`

## Release Metadata

Before tagging:

- `CHANGELOG.md` describes the shipped scope.
- `README.md` examples pass through `tests/unit/test_readme_usage.py`.
- `THIRD_PARTY_NOTICES.md` covers bundled third-party assets.
- `constraints/release.txt` matches the dependency set used by CI and release
  smoke checks.
- `pyproject.toml` classifiers and optional dependency groups match the release.
- Release-facing documentation contains no stale historical-plan, old import-root,
  or removed package-path references.
- `git status --short` is empty.
- No `tk ready`, `tk list --status=open`, or `tk list --status=in_progress`
  tickets remain for the release scope.
