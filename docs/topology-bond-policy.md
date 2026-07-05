# Topology Bond Policy

ProtRepair keeps bond truth in `StructureTopology.bonds`. Coordinates, topology,
readiness, and serialized output must not carry separate chemistry stories.

## Canonical Axes

`TopologyBond.relationship_type` records the physical relationship:

- `COVALENT`
- `DISULFIDE`
- `HYDROGEN_BOND`
- `METAL_COORDINATION`
- `UNKNOWN`

`TopologyBond.provenance` records the support mode for the endpoint pair:

- `SOURCE_EXPLICIT`: the source file explicitly reported the bond through PDB
  `LINK`, PDB `CONECT`, or mmCIF `_struct_conn`.
- `TEMPLATE_RESOLVED`: a component or residue template resolved the bond.
- `SEQUENCE_INFERRED`: polymer sequence context resolved the bond.
- `EVIDENCE_RESOLVED`: external chemistry evidence, such as retained-ligand
  atom mapping, resolved the bond.
- `REPAIR_INFERRED`: a repair transformer introduced the bond without stronger
  source, template, sequence, or evidence support.

These are orthogonal axes. Provenance is not an execution flag, not a writer
flag, and not a lifecycle flag.

## Projection Rules

Execution may treat only covalent-like relationship types as force-field planned
bonds. `SOURCE_EXPLICIT` metal coordination remains topology truth, but it is
not a covalent force-field bond merely because the source reported it.

Egress must project from canonical topology instead of inventing writer-local
connectivity. Source-explicit bonds preserve source roundtrip behavior.
Covalent-like model-resolved bonds are egress-emittable: PDB writes them as
`CONECT`, and mmCIF writes them as `_struct_conn`. Model-resolved non-covalent
or unknown relationships are not promoted to PDB `CONECT`; source-explicit
non-covalent relationships remain source-preserving boundary records where the
format supports them.

PDB egress intentionally uses two boundary projections. Gemmi connection
records preserve typed source relationships, so PDB `LINK`/`SSBOND`-like records
carry source-explicit bonds except source records that originally came from PDB
`CONECT`. PDB `CONECT` is an untyped connectivity table, so it is appended from
canonical topology after gemmi serialization and includes source PDB `CONECT`
records plus covalent-like model-resolved bonds. mmCIF has a single
`_struct_conn` projection because that boundary can carry typed relationships
and repaired/model-resolved covalent-like bonds in the same table.

Readiness must compare expected bond endpoint pairs against
`StructureTopology.bonds`. Atom coordinates alone do not prove topology
readiness. If a residue or retained non-polymer has all expected atoms but is
missing the expected covalent-like endpoint pairs, readiness must report
incomplete topology instead of treating coordinates as connectivity.

## Repair Requirements

Any transformer that creates atoms or changes chemistry must also update
canonical topology, or deliberately report that topology is incomplete. This
applies to missing heavy atoms, polymer hydrogens, and retained non-polymer
hydrogens.

Topology rebuilds must preserve existing `SOURCE_EXPLICIT` bonds unless their
endpoints are removed by a canonical normalization or rewrite. Source endpoint
identity includes the source component id and altloc before lowering; a
`LINK`, `CONECT`, or `_struct_conn` endpoint is not lowered onto the selected
canonical atom when the source component variant or altloc did not survive
normalization.

Source bond metadata is canonical metadata, not raw boundary text. Reported
distances are stored only as finite positive numeric angstrom values; corrupt,
non-finite, or non-numeric boundary distances are ignored before lowering.

Polymer hydrogen completion uses the strongest available support mode for each
new H anchor:

- static or rotatable component-template anchors are `TEMPLATE_RESOLVED`;
- backbone and N-terminal polymer context anchors are `SEQUENCE_INFERRED`;
- request-driven histidine delta protonation is `REPAIR_INFERRED` until a
  stronger microstate template or external evidence model exists.

Existing source-explicit H bonds remain authoritative when the same endpoint
pair is regenerated during hydrogen completion.

Retained non-polymer hydrogen completion uses the strongest available support
mode for each generated H anchor:

- component-template anchors are `TEMPLATE_RESOLVED`;
- explicit retained-ligand chemistry evidence and atom mapping are
  `EVIDENCE_RESOLVED`;
- template-less RDKit fallback anchors are `REPAIR_INFERRED`.

Existing source-explicit retained-ligand H bonds remain authoritative when the
same endpoint pair is regenerated during hydrogen completion.

Retained non-polymer readiness uses the same expected H atom naming policy as
hydrogen coverage. When RDKit fallback infers generated hydrogens for a residue
that already has complete H names, the generated anchor graph is projected onto
those preferred names before comparing endpoint pairs against canonical
topology.
