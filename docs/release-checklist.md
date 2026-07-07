# Release Checklist

This checklist defines the local gates required before tagging a `protrepair`
release.

## Supported Release Matrix

The current release-supported surface is intentionally capped to the platforms
and Python versions exercised by release CI:

- CPython 3.10, 3.11, and 3.12.
- Linux through GitHub Actions `ubuntu-latest`.
- Lean optional-backend coverage on Linux CPython 3.12.

Python 3.13+, macOS, and Windows are not advertised as release-supported
surfaces until matching CI lanes, constraints, native FASPR checks, and optional
backend gates are added.

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

## RDKit Release Version Policy

The full release CI lane installs RDKit through `constraints/release.txt` and
runs representative RDKit coordinate-digest checks with
`PROTREPAIR_RELEASE_STRICT_RDKIT_DIGESTS=1`. Under that strict release gate, an
unregistered RDKit backend version is a failure instead of a skip.

Local compatibility runs and no-RDKit lanes may still skip version-bound
coordinate digests when RDKit is absent or unregistered. Current release
constraints pin `rdkit==2026.3.2`, which corresponds to the registered backend
digest version `2026.03.2`. The digest registry also carries `2026.03.1` for
known reviewer/environment parity.

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

Run the fresh wheel/sdist archive content gate:

```bash
python -m pytest tests/release/test_artifact_contents.py -q
```

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
