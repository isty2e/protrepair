# Span Reconstruction Fixtures

These heavy-atom excerpts retain the deposited coordinates and author residue
numbers. Tests delete an internal span, perturb a donor torsion independently,
and check the resulting covalent geometry against the original coordinates.

- [1UBQ](https://www.rcsb.org/structure/1UBQ): chain A residues 3-13 and 34-40.
  The first window exercises the fixed carbonyl-O boundary; the second contains
  consecutive Pro37/Pro38 and covers internal and source-anchor amide nitrogen.
- [2CI2](https://www.rcsb.org/structure/2CI2): chain I residues 78-83, including
  Pro80 at the start of a reconstructed span.
- [1CRN](https://www.rcsb.org/structure/1CRN): chain A heavy atoms and deposited
  disulfide records. Removing Cys4 tests whether the workflow recognizes and
  resolves new disulfide evidence after insertion, rather than copying a donor
  crosslink in the closure kernel.

Source PDB files were retrieved from RCSB on 2026-09-08. Header, solvent, and
unrelated atom records were omitted. These are kinematic regression fixtures,
not a held-out loop-prediction benchmark. Some deposited residues have existing
restraint outliers, so the carbonyl-boundary test isolates the closure kernel
without bypassing the public workflow's geometry checks.

## Short-Gap Donor Comparison

`1ubq-short-gaps.pdb` and `af-p0cg48-short-gaps.pdb` contain residues 3-5,
25-27, 48-50, and 71-73 from 1UBQ chain A and
[AF-P0CG48-F1-model_v6](https://alphafold.ebi.ac.uk/entry/P0CG48), respectively.
The first 76 AlphaFold residues map to ubiquitin in 1UBQ; these excerpts use
that explicit mapping and retain the original heavy-atom coordinates.
The AlphaFold model was retrieved on 2026-09-08.

Tests remove residues 4, 26, 49, or 72 from the experimental structure and use
the corresponding donor residue for reconstruction. The reference's missing
coordinates are used only to assess the result, never as fitting input. These
cases distinguish rigid-frame mismatch from changes to covalent geometry and
include an unresolved closure. They are regression examples from one protein,
not independent evidence of loop-prediction accuracy.
