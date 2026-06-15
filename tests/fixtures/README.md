# Fixture Taxonomy

`tests/fixtures/` contains repository-local fixture material used by the redesign
test suite.

The fixture set is intentionally realistic rather than toy-only. The corpus is
therefore heavier than a minimal unit-test fixture set, but it preserves CI
reproducibility for behavior that only fails on deposited structures.

## Classification Axes

Fixture classification is not encoded by directory alone. The authoritative
case metadata lives in `tests/support/refinement_corpus.py` and uses orthogonal
axes:

- Surface: whole deposited structures versus extracted local refinement
  neighborhoods.
- Panel: active, extended, or exploratory breadth.
- Execution policy: must-pass gating cases versus tracking-only pressure cases.
- Evidence basis and coverage tags: internal regression, literature-backed, or
  known problematic deposition with chemistry/topology/operator coverage tags.

The current metadata classifies 47 cases: 27 whole-structure representative
cases and 20 local refinement benchmark cases. Of those, 19 are must-pass
gating cases and 28 are tracking-only cases. Panel coverage counts: 12 active,
7 extended, and 28 exploratory cases.

## Repository-Resident Fixture Families

- `pdb/1aho.pdb`: small whole-structure smoke fixture used by README examples
  and direct API tests.
- `pdb/1aho_faspr_input.pdb`: small side-chain packing fixture for the packaged
  FASPR backend.
- `pdb/refinement/*.pdb`: extracted local neighborhoods for local refinement,
  chemistry-admissibility, and literature-backed regression. These are small
  derived fixtures rather than full deposited structures.
- `corpus/*.ent` and `corpus/*.cif`: deposited whole-structure fixtures for
  representative, exploratory, and known-problematic behavior. These are the
  heavy realistic cases that keep CI and benchmark registries reproducible.

## Test Execution Surfaces

The test tree separates execution meaning from fixture storage:

- `tests/unit/`: small contract and model tests. These may use tiny fixtures,
  but they should not own corpus breadth.
- `tests/corpus/`: realistic fixture inventory, representative regression, and
  corpus metadata contract tests.
- `tests/workflow/`: end-to-end `process_structure` workflow tests.
- `tests/benchmark/`: opt-in benchmark and executable quality-comparison tests.
  These tests are marked `benchmark` and are not part of the default CI gate.

## Retention Policy

Keep fixture material in Git when it is required by one of these contracts:

- README or public API smoke behavior uses it directly.
- CI unit tests require it without network access.
- Corpus metadata marks it as active or must-pass.
- Tracking-only metadata documents a recurring realistic failure pressure that
  should remain reproducible while the implementation evolves.

Do not add one-off exploratory source files here by default. Prefer local
scratch storage plus a metadata entry only after the case exercises a recurring
coverage gap or a specific release-risk boundary.

## PDB-derived fixture data

- `pdb/1aho.pdb`
- `pdb/1aho_faspr_input.pdb`
- `pdb/refinement/*.pdb`
- `corpus/*.ent`
- `corpus/*.cif`
- any future coordinate fixtures copied from PDB archive entries

These files are structure-data fixtures, not original source code. Their usage
is governed by the Protein Data Bank archive policy rather than the repository
MIT code license.

- Upstream data source: wwPDB / RCSB PDB archive
- Data policy: CC0 1.0
- Reference: <https://www.rcsb.org/pages/usage-policy>
