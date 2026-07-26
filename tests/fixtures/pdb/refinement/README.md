# Local Refinement Fixtures

This directory contains small neighborhoods extracted from wwPDB structures for
local refinement, chemistry-admissibility, and repair regression tests. The
files are derived fixtures, not complete deposited structures.

## Canonical Inventory

`tests/support/refinement_cases.py` owns the executable case definitions,
including fixture path, focus residues or atoms, and reference geometry.
`tests/support/refinement_corpus.py` owns panel membership, execution policy,
evidence basis, and coverage tags. Tests in
`tests/corpus/test_refinement_registry.py` verify that registered fixture paths
exist and retain their expected anchor content.

Do not maintain a second filename inventory here. Add or remove a fixture
through the owning registry and its corpus metadata so execution and
documentation cannot drift independently.

## Fixture Families

The registry currently covers:

- residue-local side-chain and amide-orientation cases
- paired perturbation/reference cases
- nonstandard peptide-linking components
- backbone-window and multi-residue correction pressure cases
- glycan, metal-site, and other chemistry-admissibility cases

Filenames retain the source PDB identifier and local focus as a human-readable
lookup aid; registry metadata defines their test meaning.
