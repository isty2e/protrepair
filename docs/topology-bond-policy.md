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
  `SSBOND`, PDB `LINK`, PDB `CONECT`, or mmCIF `_struct_conn`.
- `TEMPLATE_RESOLVED`: a component or residue template resolved the bond.
- `SEQUENCE_INFERRED`: polymer sequence context resolved the bond.
- `EVIDENCE_RESOLVED`: external chemistry evidence, such as retained-ligand
  atom mapping, resolved the bond.
- `REPAIR_INFERRED`: a repair transformer introduced the bond without stronger
  source, template, sequence, or evidence support.

These are orthogonal axes. Provenance is not an execution flag, not a writer
flag, and not a lifecycle flag.

`TopologyBond.order` is a positive integer, or `None` when the order is
unresolved. A source record can establish connectivity without supplying a
bond order. In that case, ingress can use matching component or sequence
chemistry without discarding the source provenance or metadata.

## Source Bond Orders

An explicit order takes precedence over a default template. A missing order
does not: for example, a connectivity-only record for a known backbone C-O
pair retains the component's double bond. Source charges, hydrogen atoms,
and coordinates are unchanged by this resolution. It does not establish
that an explicitly supplied microstate is chemically valid.

PDB `LINK` records supply no order; `SSBOND` identifies a single disulfide
bond. For `CONECT`, ProtRepair accepts the repeated-neighbor convention used
by [RDKit's PDB parser](https://github.com/rdkit/rdkit/blob/Release_2026_03_2/Code/GraphMol/FileParsers/PDBParser.cpp).
Two, three, or four occurrences in one direction supply a double, triple,
or quadruple order. Counts include distinct continuation rows, but identical
rows are treated as redundant. Reciprocal records are not added together.
Contradictory multiple orders in the two directions are rejected. One
occurrence supplies connectivity only, not an explicit single-bond assertion.
The reader accepts RDKit's extension through the sixth neighbor field; the
writer uses four fields per row and keeps each multiplicity together.
Legacy pre-3.2 hydrogen/salt-bridge column semantics are not supported.

mmCIF carries explicit `sing`, `doub`, `trip`, and `quad` values in
[`_struct_conn.pdbx_value_order`](https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_struct_conn.pdbx_value_order.html),
including connections within a residue. An absent field or `?`/`.` supplies
no order evidence. Unsupported explicit values are rejected rather than
replaced with single bonds. The writer projects canonical orders into this
field and writes `?` for unresolved orders.

These records do not provide a general aromaticity encoding. Ingress retains
aromatic support from a matching component definition, but arbitrary ligand
aromaticity recovery or writer-side Kekulization is outside this policy.
PDB also cannot distinguish unresolved order from an unannotated single bond.
Use mmCIF when that distinction matters. Neither format is a lossless archive
of all internal chemistry provenance.

## Standard Component Chemistry

The built-in standard residues assign double bonds to backbone C=O and the
ASN CG=OD1 and GLN CD=OE1 amides, matching the corresponding
[wwPDB chemical component definitions](https://www.wwpdb.org/data/ccd).
C-OXT remains single. These definitions feed topology construction and
force-field binding; the backend does not repair their chemistry privately.

This is not a complete polymer microstate model. Charge-dependent groups,
including guanidinium, carboxylates, and histidine, still need coordinated
bond-order, charge, and hydrogen resolution. Source charges are not changed by
the carbonyl correction.

## Projection Rules

Execution may treat only covalent-like relationship types as force-field planned
bonds. `SOURCE_EXPLICIT` metal coordination remains topology truth, but it is
not a covalent force-field bond merely because the source reported it.
Resolved canonical orders take precedence over execution fallback templates.
An unresolved covalent order can remain in region planning, but cannot be
bound to RDKit as an assumed single bond. An unresolved bond outside the
included region does not block local refinement. An `UNKNOWN` relationship
is not proof of a non-covalent relationship; existing supported ligand
chemistry inference remains available for those endpoints.

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
`CONECT`. Source hydrogen bonds are omitted because PDB has no corresponding
typed connection record. PDB `CONECT` is an untyped connectivity table, so it
is appended from canonical topology after gemmi serialization and includes
source PDB `CONECT` records plus covalent-like bonds, whether source-reported
or model-resolved. mmCIF has a
single `_struct_conn` projection because that boundary can carry typed
relationships and repaired/model-resolved covalent-like bonds in the same
table. Serialized connection distances are derived from current coordinates;
the source-reported distance remains provenance metadata and is not emitted as
a stale geometric claim after coordinates change.

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

### Source Bond Preservation

Topology rebuilds must preserve existing `SOURCE_EXPLICIT` bonds unless their
endpoints are removed by a canonical normalization or rewrite. Source endpoint
identity includes the source component id and altloc before lowering; a
`LINK`, `CONECT`, or `_struct_conn` endpoint is not lowered onto the selected
canonical atom when the source component variant or altloc did not survive
normalization.

PDB `SSBOND` is the deliberate exception to altloc-specific endpoint matching.
The record identifies two cysteine residues but has no alternate-location
fields; the wwPDB format guide explicitly calls this out as a known ambiguity
for disordered SG atoms. ProtRepair therefore lowers a source `SSBOND` onto the
selected canonical SG variants when the residue component identities survive.
PDB `LINK` and mmCIF `_struct_conn`, which can carry endpoint altloc identity,
continue to require the declared variants to survive. Typed records take
precedence over `CONECT` for relationship type and source metadata. A matching
`CONECT` record can still supply a missing order. Conflicting explicit orders
or typed declarations remain invalid rather than silently becoming first-wins.
Disulfide declarations cannot be combined with multiple-bond orders.

Source bond metadata is canonical metadata, not raw boundary text. Reported
distances are stored only as finite positive numeric angstrom values; corrupt,
non-finite, or non-numeric boundary distances are ignored before lowering.

### Hydrogen Bond Provenance

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

### Disulfide Chemistry

A canonical `COVALENT` or `DISULFIDE` bond between two CYS `SG` atoms defines
disulfide chemistry independently of provenance, source record class, distance,
or polymer/retained placement. Polymer and retained-non-polymer hydrogen
expectation and completion therefore omit hydrogens anchored to those sulfur
atoms. Geometry-only SG proximity remains diagnostic evidence and must not
silently remove thiol hydrogen expectation or materialization.

This follows the chemical distinction between reduced cysteine thiols and the
oxidized covalent S-S linkage described in the protein-disulfide literature. It
does not claim that proximity alone establishes oxidation state.

If a canonical disulfide endpoint still contains an H, D, or T atom assigned to
its thiol site, ProtRepair records a typed chemistry contradiction independently
of missing-hydrogen coverage. Standard `HG`/`DG`/`TG` names are recognized;
nonstandard names require an explicit covalent-like SG-H topology bond. The
planner removes only the contradictory isotope atoms and their incident bonds,
then re-observes the structure while preserving the S-S relationship, other
hydrogens, formal charges, and source coordinates. It does not infer a thiolate
or delete an SG-SG bond from metal coordination, proximity, or charge alone.

The atom-identity rule deliberately covers mixed H/D models. Neutron structures
can model H and D simultaneously at one exchangeable site and commonly use
distinct names such as `HG` and `DG`; each explicitly present isotope is part of
the canonical atom inventory and must be normalized when the same sulfur is in
a canonical disulfide.

### Disulfide Candidate Resolution

Continuous relaxation likewise projects inter-residue disulfide constraints
from canonical topology only. A likely geometry candidate must pass through an
explicit topology-writing transformer before it becomes an execution bond; the
continuous backend does not maintain a hidden geometry-derived bond graph.

ProtRepair uses an SG-SG distance of at most 2.5 angstrom as candidate evidence,
not as direct oxidation-state truth. Marino and Gladyshev describe 2.5 angstrom
as a commonly employed structure-screening cutoff, while broader disulfide
modeling work also considers bond geometry, torsions, energetics, and steric
context. A unique candidate may therefore be promoted only by the explicit
topology-resolution action, after ambiguity and conflicting canonical
relationships have been excluded. Source-explicit or otherwise conflicting
topology is preserved and reported rather than overwritten by proximity.

### Retained-Ligand Readiness

Retained non-polymer readiness uses the same expected H atom naming policy as
hydrogen coverage. When RDKit fallback infers generated hydrogens for a residue
that already has complete H names, the generated anchor graph is projected onto
those preferred names before comparing endpoint pairs against canonical
topology.

## Scientific and Format References

- wwPDB, [PDB Format v2.3 Connectivity Annotation Section](https://www.wwpdb.org/documentation/file-format-content/format23/sect6.html),
  including the `SSBOND` altloc limitation and `LINK` endpoint fields.
- Dombkowski, Sultana, and Craig, "Protein disulfide engineering," *FEBS
  Letters* 588 (2014), [DOI 10.1016/j.febslet.2013.11.024](https://doi.org/10.1016/j.febslet.2013.11.024).
- Marino and Gladyshev, "Redox Biology: Computational Approaches to the
  Investigation of Functional Cysteine Residues," *Antioxidants & Redox
  Signaling* 15 (2011), [DOI 10.1089/ars.2010.3561](https://doi.org/10.1089/ars.2010.3561).
- Fass and Thorpe, "Chemistry and Enzymology of Disulfide Cross-Linking in
  Proteins," *Chemical Reviews* 118 (2018),
  [DOI 10.1021/acs.chemrev.7b00123](https://doi.org/10.1021/acs.chemrev.7b00123).
- Liebschner et al., "Evaluation of models determined by neutron diffraction
  and proposed improvements to their validation and deposition," *Acta
  Crystallographica D* 74 (2018),
  [DOI 10.1107/S2059798318004588](https://doi.org/10.1107/S2059798318004588).
