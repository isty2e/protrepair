# Ingress Policy

ProtRepair converts PDB and mmCIF input into one canonical structure
realization before repair begins. Validation, model selection, source-variant
selection, and residue-role classification all belong to this boundary.
Downstream code operates on the normalized structure and must not reinterpret
the source independently.

## Model And Variant Selection

ProtRepair currently reads the first model from a multi-model PDB or mmCIF
file. Select a different model before calling ProtRepair when another
realization is required.

Alternate locations and microheterogeneous residue variants are resolved
during ingress. A selected atom cohort is kept consistently for the residue;
discarded source variants and connectivity that points only to them do not
enter canonical topology.

PDB output leaves alternate-location labels blank on the selected atoms and
their `LINK` endpoints. It does not combine discarded alternatives with the
selected coordinates. This keeps downstream readers from dropping a selected
atom because it still carries a `B` or lowercase label. Coordinates, occupancy
and B factors are unchanged apart from normal output rounding. The structure
object and mmCIF output retain the selected labels; use mmCIF if they matter,
because the PDB alone cannot recover them.

## PDB Identifiers And Output Limits

ProtRepair supports Gemmi's two-character chain IDs and uppercase hybrid-36
numbering. Residue numbers can range from `-999` to `1223055`; emitted atom
serials, including numbered `TER` records, cannot exceed `43770015`. Lowercase
hybrid-36 is not supported: Gemmi reads it as a different numbering scheme.
Malformed or unsupported serial fields raise `StructureNormalizationError`
with the field and token, including invalid nonblank `CONECT` fields.
See [Topology Bond Policy](topology-bond-policy.md#source-bond-orders) for
ambiguous and missing connection endpoints.

Public PDB writers keep coordinates at three decimal places when they fit,
or round to two places when needed to stay within eight columns. B factors
use two, one, or zero decimal places within six columns; zero-place fields
retain a decimal point. Values that still do not fit are refused, not
clamped or truncated. These are output-format limits, not ingress validation
or scientific quality thresholds.

PDB cannot preserve component IDs longer than three characters, atom names
longer than four, chains longer than two, or formal charges outside one digit.
Writers raise `StructureSerializationError` instead of changing these facts;
use mmCIF for those cases. Insertion and altloc codes must be a single
printable ASCII character in the current coordinate adapter, so switching
to mmCIF does not resolve that adapter limit. H/D/T and explicit zero charges
are preserved in public PDB output.

PDB extensions are not supported by every downstream tool. RDKit parser
probes and FASPR use internal calculation projections with separate contracts;
successful public serialization does not establish native-tool compatibility.
Both projections use the same blank-altloc policy as public PDB output.

RDKit diagnostics associate parsed atoms with the original canonical atoms by
the serials in the emitted PDB, not RDKit's one-character chain labels or atom
indices. This keeps diagnostic and repair targets distinct for chains such as
`A` and `AA` without renaming chains or renumbering residues, which could change
proximity bonds. Selected altloc labels are omitted. RDKit still requires decimal
atom serials and residue numbers: emitted hybrid-36 fields cause a nonlocalized
parser failure. The serial limit applies
to the emitted records, not just the atom count, since `TER` records consume
serials too. A successful probe describes the atoms RDKit retained; it is not
a guarantee that RDKit read every canonical atom or alternate-location record.
Canonical chain IDs or residue numbers outside Gemmi's PDB capacity cannot reach
that reader unchanged. For hydrogenated structures, ProtRepair keeps the workflow
result and reports the unsupported projection as one structure-level parser
error, with the original field and location. This does not mean the source
chemistry is invalid.

## Numeric Atom Values

Atom coordinates, occupancy, and B factors must be finite.

- Occupancy must be in the closed interval `[0.0, 1.0]`.
- B factors must be non-negative. There is no additional upper bound.
- Coordinates outside the canonical geometry model's finite domain are
  rejected.

Ingress rejects values outside these domains, including boundary-adjacent
values such as occupancy `1.0000001` and B factor `-0.0000001`. It does not
clamp them. Clamping would replace source evidence and hide a data-quality
problem before repair starts.

## Original Chemistry Observations

The selected input is retained at `structure.provenance.ingress.observation`.
It keeps original atom identities, coordinates, reported charges, and explicit
connections, including observed H, D, and T. Repair can remove or recreate an
atom without erasing that observation. Original labels stay original even if
the current structure renames an atom.

An explicitly reported charge of zero differs from an unspecified charge.
ProtRepair preserves that distinction when reading and writing both formats;
PDB output uses `0+` for an explicit zero. Missing source hydrogens do not show
that a residue is neutral or deprotonated. Connections and their reported
semantics are evidence, not a guarantee that the source chemistry is valid.
See [Topology Bond Policy](topology-bond-policy.md#source-bond-orders).

Passing a repaired `ProteinStructure` back into the workflow keeps its original
observation; generated atoms do not become source observations. Manually built
structures may have no observation. Output files describe the current structure,
not the original snapshot. Reading an exported file starts a new observation;
PDB and mmCIF output do not archive the processing history.

## Canonical Residue Roles

Each source residue slot receives one canonical role before
`StructureConstitution` is constructed:

- polymer
- retained non-polymer
- water

The chain/ligand partition and `ResidueSite.is_hetero` are projections of this
decision. Completion, topology, diagnostics, and ligand handling must not
reclassify residues independently.

### Evidence Precedence

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
4. An entity-less `HETATM` slot is a polymer candidate only when every
   component variant is a standard residue or a bundled peptide-linking
   nonstandard component. It becomes polymer only when residue numbering
   places it next to known polymer context in the observed chain-slot order or
   a source-explicit inter-residue covalent/unknown C-N connection links it
   into that context. Insertion-code adjacency is conservative: blank-to-`A`
   and immediately successive one-character codes are neighbors, while
   distant codes are not. Resolution is transitive across consecutive
   candidates. A source-explicit C-N link between two candidates is sufficient
   linkage evidence even when no `ATOM` anchor survives; an intra-residue C-N
   record is not polymer evidence.
5. Unsupported, isolated, or otherwise unresolved `HETATM` slots remain
   retained non-polymers.

Gemmi's residue-level `entity_type` is not authoritative by itself when no
declared entity exists. PDB parsing derives that value from `ATOM`/`HETATM`, so
treating it as independent evidence would reproduce the ambiguity this policy
removes.

An entity-less PDB chain made entirely of peptide-compatible `HETATM` residues
without explicit C-N links remains retained non-polymer content. Chemistry
support alone is not evidence that a component occupies a polymer chain slot.

## Egress Projection

Canonical role controls the emitted entity type: polymer residues emit polymer
entities, while retained residues emit non-polymer entities. Record spelling is
a separate format projection. Standard polymer components use `ATOM`; bundled
or peptide-linking nonstandard polymer components use `HETATM`; retained
non-polymers use `HETATM`. Unknown polymer components retain the existing
`ATOM` fallback because ingress has no source-record provenance axis for
reconstructing their original spelling.

This preserves the conventional PDB spelling of SEP, TPO, PTR, and MSE without
moving those residues back into the ligand ontology.
