# Analysis Policy

ProtRepair can attach structured Ramachandran and coarse secondary-structure
results to a workflow result. These analyses describe the final canonical
structure; they do not direct repair planning.

## Ramachandran Categories

When both torsions are available, ProtRepair assigns one closed
`RamachandranCategory`:

- `helix`: phi in `[-160, -20]` and psi in `[-90, 45]`
- `beta`: phi in `[-180, -40]` and psi at least `90` or at most `-120`
- `left_handed`: phi in `[20, 120]` and psi in `[-20, 120]`
- `other`: outside the regions above

A residue with a missing torsion receives no Ramachandran category.
Peptide-neighbor topology, rather than residue numbering alone, determines
whether the adjacent geometry needed for a torsion is available.

## Coarse Secondary Structure

Secondary-structure output projects the Ramachandran categories to:

- `H` for `helix`
- `E` for `beta`
- `C` for left-handed, other, missing-torsion, and gap-disconnected residues

This is not a DSSP replacement. It does not infer hydrogen-bond patterns,
turns, bends, strand registration, or a separate polyproline-II assignment.
