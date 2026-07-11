# Element Radius Policy

ProtRepair uses one canonical radius basis for element-radius diagnostics:
a static snapshot of RDKit PeriodicTable radii generated from `rdkit==2026.3.2`
(`rdBase.rdkitVersion == "2026.03.2"`).

- Clash and steric scoring use `GetRvdw`.
- Bond-length, near-covalent, parser-witness, and retained-ligand evidence
  checks use `GetRcovalent`.
- Deuterium and tritium are isotope aliases of hydrogen at the radius boundary.
- Resolution availability, isotope aliasing, and upstream data quality are
  orthogonal facts. `ElementRadiusResolution` marks resolved values as either
  `SOURCE_REPORTED` or `SOURCE_DEFAULT`; prepared lookups retain the requested
  symbols whose values are upstream defaults.
- RDKit attributes covalent radii to Cordero et al.
  ([DOI 10.1039/B801115J](https://doi.org/10.1039/B801115J)) and documents
  `1.9` as its unknown-value default. In the pinned table this applies to
  Bk-Cn (atomic numbers 97-112). Equal `1.9` values elsewhere are not inferred
  to be defaults by numeric equality.
- RDKit attributes van der Waals radii to BODR v10.1 and explicitly documents
  `2.0` defaults for Ds, Rg, Cn, Nh, Fl, Mc, Lv, Ts, and Og. Other equal `2.0`
  values remain source-reported.
- Blank, pseudo, invalid, or unsupported symbols are unresolved elements. They
  must not silently fall back to carbon-like radii.
- Hot atom-pair loops should consume prepared `ElementRadiusLookup` values.
  Callers that work on batches should raise one aggregate unresolved-element
  error at the preparation boundary instead of warning per atom pair.
- Source-default radii remain usable so diagnostics retain broad element
  coverage. They are lower-confidence source facts, not missing elements and
  not silent substitutes for carbon-like radii.
- Spatial search envelopes must be derived from the active radius table and
  policy thresholds. Fixed legacy cutoffs are acceptable only for non-radius
  scoring terms such as the rotatable-H potential-energy horizon. A
  donor-centered rotatable-H environment covers the largest active
  candidate-centered scoring horizon plus the maximum candidate displacement
  from the donor.

This policy is intentionally separate from RDKit force-field behavior. UFF and
RDKit PDB proximity bonding are algorithm-specific mechanisms, not the canonical
per-element diagnostic radius table.

The snapshot is selected for deterministic lookup, broad element coverage, and
consistency with the required RDKit interoperability boundary. It is not a
global scientific ranking oracle. RDKit readability and valence checks answer
whether downstream RDKit consumers can interpret an output; protein-specific
quality evidence such as topology completeness, steric contacts, bond and angle
geometry, Ramachandran behavior, and realistic repair regressions remains on
independent axes. PRAS-derived protein repair behavior and its empirical lineage
therefore remain relevant evidence rather than being overridden by UFF or a
PeriodicTable radius value.

The copied RDKit radius snapshot is distributed with the complete pinned RDKit
BSD 3-Clause text in `vendor/rdkit/LICENSE`. `THIRD_PARTY_NOTICES.md` records
the runtime and snapshot provenance separately from the underlying scientific
sources.

Radius aliases do not own general element semantics. Canonical structure atom
sites retain their source element symbol and derive a separate
`ElementIdentity`: `D` and `T` keep their source spelling and isotope mass
number while sharing hydrogen chemical identity with `H`. Diagnostics,
readiness, completion, packing, and refinement classify source atoms through
that model-owned identity rather than literal element comparisons. Coordinate
egress restores explicit isotope symbols after Gemmi projection so the
canonical source identity is not collapsed to an unknown element. Polymer
hydrogen completion also restores D/T labels when the same residue-local atom
name survives rebuilding; it does not guess an isotope reassignment when a
source hydrogen identity is replaced by a different materialized atom name.

Near-covalent checks use covalent radii plus an explicit distance margin to
generate candidate contacts. They may reuse clash atom-pair scope policy for
hydrogen, ligand, and bonded-neighbor exclusions, but they must not depend on
vdW clash overlap or on the output of steric clash filtering. Changing vdW
radii alone must not suppress a contact that satisfies the covalent threshold.
Canonical covalent, disulfide, and metal-coordination topology relationships
explain their endpoint proximity and therefore do not become unexpected
near-covalent pathology; an `UNKNOWN` relationship does not suppress a
finding. Same-residue heavy-atom eligibility follows the explicit
near-covalent policy while known template-bond neighbors remain excluded.
