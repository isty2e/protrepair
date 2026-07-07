# Element Radius Policy

ProtRepair uses one canonical radius basis for element-radius diagnostics:
a static snapshot of RDKit PeriodicTable radii generated from `rdkit==2026.3.2`
(`rdBase.rdkitVersion == "2026.03.2"`).

- Clash and steric scoring use `GetRvdw`.
- Bond-length, near-covalent, parser-witness, and retained-ligand evidence
  checks use `GetRcovalent`.
- Deuterium and tritium are isotope aliases of hydrogen at the radius boundary.
- Blank, pseudo, invalid, or unsupported symbols are unresolved elements. They
  must not silently fall back to carbon-like radii.
- Hot atom-pair loops should consume prepared `ElementRadiusLookup` values.
  Callers that work on batches should raise one aggregate unresolved-element
  error at the preparation boundary instead of warning per atom pair.
- Spatial search envelopes must be derived from the active radius table and
  policy thresholds. Fixed legacy cutoffs are acceptable only for non-radius
  scoring terms such as the rotatable-H potential-energy horizon.

This policy is intentionally separate from RDKit force-field behavior. UFF and
RDKit PDB proximity bonding are algorithm-specific mechanisms, not the canonical
per-element diagnostic radius table.

Near-covalent checks currently consume clash output, so changing vdW radii can
still change which near-covalent contacts are considered. That coupling is
tracked separately from the radius lookup contract.
