# Ingress Residue Role Policy

ProtRepair normalizes each source residue slot to one canonical role before it
constructs `StructureConstitution`:

- polymer
- retained non-polymer
- water

The chain/ligand partition and `ResidueSite.is_hetero` are projections of this
single decision. Completion, topology, diagnostics, and ligand handling must not
reclassify residues independently.

## Evidence Precedence

Ingress applies evidence in this order:

1. Water identity is excluded from both polymer and retained-ligand buckets.
2. A residue `entity_id` that resolves to a declared mmCIF/Gemmi entity is
   authoritative. Polymer entities remain polymer even when `group_PDB` is
   `HETATM`; non-polymer and branched entities remain retained non-polymers even
   when the component is chemically peptide-compatible.
3. Without a declared entity, an `ATOM` residue is a polymer anchor. If one
   microheterogeneous slot mixes that anchor with an unsupported `HETATM`
   component, ingress rejects the contradictory role instead of selecting a
   role by occupancy.
4. An entity-less `HETATM` slot is a polymer candidate only when every component
   variant is a standard residue or a bundled peptide-linking nonstandard
   component. It becomes polymer only when residue numbering places it next to
   known polymer context in the observed chain-slot order or a source-explicit
   inter-residue covalent/unknown C-N connection links it into that context.
   Insertion-code adjacency is conservative: blank-to-`A` and immediately
   successive one-character codes are neighbors, while distant codes are not.
   Resolution is transitive across consecutive candidates. A source-explicit
   C-N link between two candidates is sufficient linkage evidence even when no
   `ATOM` anchor survives; an intra-residue C-N record is not polymer evidence.
5. Unsupported, isolated, or otherwise unresolved `HETATM` slots remain retained
   non-polymers.

Gemmi's residue-level `entity_type` is not authoritative by itself when no
declared entity exists. PDB parsing derives that value from `ATOM`/`HETATM`, so
treating it as independent evidence would reproduce the ambiguity this policy
removes.

## Egress Projection

Canonical role controls the emitted entity type: polymer residues emit polymer
entities, while retained residues emit non-polymer entities. Record spelling is
a separate format projection. Standard polymer components use `ATOM`; bundled or
peptide-linking nonstandard polymer components use `HETATM`; retained
non-polymers use `HETATM`. Unknown polymer components retain the existing `ATOM`
fallback because ingress has no source-record provenance axis for reconstructing
their original spelling. This preserves the conventional PDB spelling of SEP,
TPO, PTR, and MSE without moving those residues back into the ligand ontology.

An entity-less PDB chain made entirely of peptide-compatible `HETATM` residues
without explicit C-N links remains intentionally unresolved as retained
non-polymer content. Chemistry support alone is not evidence that a component
occupies a polymer chain slot.
