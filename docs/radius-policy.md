# Element Radius Policy

## Radius Basis

ProtRepair uses one canonical radius basis for element-radius diagnostics: a
static snapshot of RDKit PeriodicTable radii generated from `rdkit==2026.3.2`
(`rdBase.rdkitVersion == "2026.03.2"`).

- Clash and steric scoring use `GetRvdw`.
- Bond-length, near-covalent, parser-witness, and retained-ligand evidence
  checks use `GetRcovalent`.
- Deuterium and tritium are isotope aliases of hydrogen at the radius boundary.

This radius table is separate from RDKit force-field behavior. UFF and RDKit
PDB proximity bonding are algorithm-specific mechanisms, not the canonical
per-element diagnostic radius table.

## Resolution And Upstream Defaults

Resolution availability, isotope aliasing, and upstream data quality are
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
- Source-default radii remain usable so diagnostics retain broad element
  coverage. They are lower-confidence source facts, not missing elements or
  silent substitutes for carbon-like radii.
- Blank, pseudo, invalid, or unsupported symbols are unresolved elements. They
  must not silently fall back to carbon-like radii.

The copied RDKit radius snapshot is distributed with the complete pinned RDKit
BSD 3-Clause text in `vendor/rdkit/LICENSE`. `THIRD_PARTY_NOTICES.md` records
runtime and snapshot provenance separately from the underlying scientific
sources.

## Prepared Lookup And Search Bounds

Hot atom-pair loops consume prepared `ElementRadiusLookup` values. Batch callers
raise one aggregate unresolved-element error at the preparation boundary
instead of warning for each atom pair.

Spatial search envelopes are derived from the active radius table and policy
thresholds. Fixed legacy cutoffs are used only for non-radius scoring terms such
as the rotatable-H potential-energy horizon. A donor-centered rotatable-H
environment covers the largest active candidate-centered scoring horizon plus
the maximum candidate displacement from the donor.

## Scientific Scope

The snapshot provides deterministic lookup, broad element coverage, and
consistency with the required RDKit interoperability boundary. It is not a
global scientific ranking criterion. RDKit readability and valence checks
answer whether downstream RDKit consumers can interpret an output.
Protein-specific quality evidence remains separate: topology completeness,
steric contacts, bond and angle geometry, Ramachandran behavior, and realistic
repair regressions. PRAS-derived protein repair behavior and its empirical
lineage therefore remain relevant rather than being overridden by UFF or a
PeriodicTable radius value.

## Isotope Identity

Radius aliases do not own general element semantics. Canonical atom sites retain
their source element symbol and derive a separate `ElementIdentity`: `D` and `T`
keep their source spelling and isotope mass number while sharing hydrogen
chemical identity with `H`.

Diagnostics, readiness, completion, packing, and refinement classify atoms
through that model-owned identity rather than literal element comparisons.
Coordinate egress restores explicit isotope symbols after Gemmi projection so
canonical source identity is not collapsed to an unknown element. Polymer
hydrogen completion also restores D/T labels when the same residue-local atom
name survives rebuilding; it does not guess an isotope reassignment when a
source hydrogen identity is replaced by a different materialized atom name.

## Near-Covalent Detection

Near-covalent checks use covalent radii plus an explicit distance margin to
generate candidate contacts. They may reuse clash atom-pair scope policy for
hydrogen, ligand, and bonded-neighbor exclusions, but they must not depend on
vdW clash overlap or the output of steric clash filtering. Changing vdW radii
alone must not suppress a contact that satisfies the covalent threshold.

Canonical covalent, disulfide, and metal-coordination relationships explain
their endpoint proximity and do not become unexpected near-covalent pathology.
An `UNKNOWN` relationship does not suppress a finding. Same-residue heavy-atom
eligibility follows the explicit near-covalent policy while known template-bond
neighbors remain excluded.

## Polar Contacts

A close H contact is not necessarily a steric clash. Clash checks, near-covalent
diagnostics and H-orientation scoring share a conservative hydrogen-bond
exemption. It applies within a residue as well as between residues, after each
consumer's bonded-neighbor exclusions.

The H must have a supported N, O or S parent, and the other atom must have
compatible acceptor chemistry. Formal charge, bond order, attached H and
resonance context matter: an amide N or protonated amine is not an acceptor just
because it is nitrogen. Known canonical bonds take precedence. Supported
templates can fill absent bonds or interpret an exact match to an untyped
CONECT record; they cannot move H to a nearer atom or override a conflicting
attachment. Standard H-numbering variants are recognized only when their
template parent is unambiguous, including branched methylene names such as ILE
HG12/HG13. These projections do not change topology.

Carboxyl oxygens can be recognized from their carbon/oxygen bond context before
formal charges are available, including stock ASP, GLU and terminal groups. This
does not assign charges. An observed acid O-H or conflicting source chemistry
still limits acceptor eligibility.

The current geometry envelope is H...acceptor distance 1.6-2.4 A, donor-H
distance at most 1.45 A, and donor-H...acceptor angle greater than 120 degrees.
These are diagnostic heuristics, not universal chemical limits or a force-field
energy model. Metal-bound, contradictory or insufficiently specified chemistry
receives no exemption. A remaining warning is therefore not proof that the
contact is physically impossible, and RDKit readability is assessed separately.
