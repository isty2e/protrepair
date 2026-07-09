# Release Checklist

This checklist defines the local gates required before tagging a `protrepair`
release.

## Supported Release Matrix

The current release-supported surface is intentionally capped to the platforms
and Python versions exercised by release CI:

- CPython 3.10, 3.11, and 3.12.
- Linux through GitHub Actions `ubuntu-latest`.

Python 3.13+, macOS, and Windows are not advertised as release-supported
surfaces until matching CI lanes, constraints, native FASPR checks, and RDKit
version gates are added.

## Required Verification

Install release-constrained dependencies before running release gates:

```bash
python -m pip install -c constraints/release.txt ".[dev]"
```

Run the permanent code-quality and unit surface:

```bash
python -m ruff check
python -m basedpyright
PROTREPAIR_RELEASE_STRICT_RDKIT=1 python -m pytest \
  tests/unit tests/corpus tests/workflow -m "not benchmark" -q
```

Run the release-facing API and documentation smoke surface:

```bash
python -m pytest \
  tests/unit/test_release_public_api_contract.py \
  tests/unit/test_readme_usage.py \
  tests/unit/test_release_artifacts.py \
  -q
```

## RDKit Release Version Policy

The full release CI lane installs RDKit through `constraints/release.txt` and
runs all RDKit-version-bound scientific checks with
`PROTREPAIR_RELEASE_STRICT_RDKIT=1`. Under that strict release gate, an
unregistered RDKit backend version fails both coordinate-digest and live radius
snapshot verification instead of skipping either contract.

Local compatibility runs may still skip version-bound coordinate digests when
RDKit is present but unregistered. Missing RDKit is a broken required-dependency
installation, not a supported runtime mode. Current release constraints pin
`rdkit==2026.3.2`, which corresponds to RDKit backend version `2026.03.2`. The
digest registry may carry more than one known coordinate digest for the same
RDKit version when constrained scientific-stack or platform differences preserve
topology and atom ordering but move coordinates. The registry also carries
`2026.03.1` and `2026.03.3` for known reviewer/environment parity.

Element-radius diagnostics use a static `rdkit==2026.3.2` PeriodicTable
snapshot generated from `GetRvdw` and `GetRcovalent`, not runtime RDKit lookup.
See `docs/radius-policy.md` for the canonical unknown-element and batch-lookup
contract. Update `src/protrepair/chemistry/radii.py`, `THIRD_PARTY_NOTICES.md`,
`vendor/rdkit/LICENSE`, and the radius sentinel/quality tests together if the
release RDKit pin changes.
`tests/unit/test_radii.py` includes a live RDKit PeriodicTable verifier that
compares all atomic numbers 1 through 118 when the installed RDKit backend
version matches the committed snapshot version; the release constraints make
that verifier active in release CI.

## Build And Install Smoke

Build wheel and sdist artifacts from a clean worktree, then smoke-test each exact
artifact path:

```bash
python -m hatchling build
python scripts/run_installed_artifact_smoke.py --artifact-path /absolute/path/to/protrepair-<version>-<tag>.whl \
  --venv-path .tmp/release-wheel-smoke
python scripts/run_installed_artifact_smoke.py --artifact-path /absolute/path/to/protrepair-<version>.tar.gz \
  --venv-path .tmp/release-sdist-smoke
```

The script requires one explicit built wheel or source distribution. It
recreates a safe virtual environment under `.tmp`, force-installs that exact
artifact under `constraints/release.txt`, and verifies that `protrepair` imports
from the installed distribution rather than the source checkout. The install
environment also exports an absolute `PIP_CONSTRAINT` so isolated PEP 517 sdist
builds remain constrained with older pip versions, plus the dedicated
`PIP_BUILD_CONSTRAINT` used by pip 25.3 and newer. The shared smoke covers
bundled chemistry resources, coordinate read/write,
`process_structure()`, packaged FASPR execution, and RDKit local refinement.
Release CI builds wheel and sdist artifacts and runs this installed-artifact
smoke for both on Python 3.12.

Bundled FASPR assets are an installed-package artifact contract. Source-tree
execution may use an explicit FASPR `executable_path`, but artifact verification
should not rely on speculative build-directory probing.

Source installs and GitHub installs build the vendored FASPR executable through
`scikit-build-core` and CMake. The documented native-build prerequisite is CMake
3.18 or newer plus a working C++ compiler toolchain. Prefer wheels for users who
need FASPR but do not want to build native assets locally.

## Artifact Contents

Run the fresh wheel/sdist archive content gate:

```bash
python -m pytest tests/release/test_artifact_contents.py -q
```

The archive gate compares packaged `radii.py`, `THIRD_PARTY_NOTICES.md`, the
RDKit license, and the sdist radius policy byte-for-byte with their repository
owners. The repository test also pins the complete chosen RDKit license payload
by SHA-256, so a truncated local copy cannot become the new expected artifact.

Confirm that the wheel contains these non-code assets:

- `protrepair/chemistry/resources/nonstandard_components.json.gz`
- `protrepair/chemistry/resources/retained_non_polymer_components.json.gz`
- `protrepair/chemistry/radii.py`
- `protrepair/packing/faspr/bin/FASPR`
- `protrepair/packing/faspr/bin/dun2010bbdep.bin`

Confirm that the sdist contains the release gate sources:

- `constraints/release.txt`
- `docs/release-checklist.md`
- `docs/radius-policy.md`
- `scripts/run_installed_artifact_smoke.py`
- `THIRD_PARTY_NOTICES.md`
- `vendor/rdkit/LICENSE`

## GitHub Actions Policy

CI may use first-party GitHub actions pinned to reviewed major-version tags for
readability, currently `actions/checkout@v4` and `actions/setup-python@v5`.
Review those major tags before each release. Any other action reference must be
pinned to a full 40-character commit SHA before it is added to release CI.
Workflow token permissions are constrained to `contents: read`; widen them only
for a concrete release job that needs additional repository access.

## Release Metadata

Before tagging:

- `CHANGELOG.md` describes the shipped scope.
- `README.md` examples pass through `tests/unit/test_readme_usage.py`.
- `THIRD_PARTY_NOTICES.md` covers bundled third-party assets.
- `constraints/release.txt` matches the dependency set used by CI and release
  smoke checks.
- `pyproject.toml` classifiers and dependency groups match the release.
- Release-facing documentation contains no stale historical-plan, old import-root,
  or removed package-path references.
- `git status --short` is empty.
- No `tk ready`, `tk list --status=open`, or `tk list --status=in_progress`
  tickets remain for the release scope.
