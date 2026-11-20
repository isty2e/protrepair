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

## Terminal Oxygen Names

After selecting atoms and residue variants, ingress normalizes complete terminal
oxygen pairs in recognized peptide residues with canonical backbone templates.
If one alias has a declared charge of `-1` and the other is explicitly `0` or
unspecified, the charged atom becomes `OXT` and its partner becomes `O`. The
charge stays on the same physical atom. Otherwise ingress uses this naming table:

| Source names | Canonical names |
| --- | --- |
| `OT1`, `OT2` | `O`, `OXT` |
| `OC1`, `OC2` | `OXT`, `O` |

The fallback mappings follow the GROMACS v2026.0
[CHARMM27 terminal rules](https://raw.githubusercontent.com/gromacs/gromacs/v2026.0/share/top/charmm27.ff/aminoacids.c.tdb)
and [Amber99SB-ILDN atom renaming table](https://raw.githubusercontent.com/gromacs/gromacs/v2026.0/share/top/amber99sb-ildn.ff/aminoacids.arn).

Both atoms must be oxygen, with no `O` or `OXT` already present. Lone aliases,
mixed families and canonical/alias mixtures raise `StructureNormalizationError`
at read time. Retained ligands and names defined by the component template keep
their identities. Other naming schemes, including `O1`/`O2`, are not reconciled.
The mapping is a naming convention, not a determination of bond order or
protonation from coordinates.

Coordinates, selected atom order, occupancy, B factors, isotopes and reported
charges stay with each physical atom. Surviving source connection endpoints use
the same name mapping and retain their component and altloc matching rules.
Excluded coordination declarations keep their raw endpoint names.
Normalization precedes template bonds and sequence-link inference, so the
resulting `OXT` has the same terminal meaning as an input named `OXT`.

An alias-bearing residue is refused if any source H/D/T name would identify one
of the newly created `O`/`OXT` parents under the existing polymer naming rules,
even when an explicit source bond is also present. These include `HO`, `HXT`
and their isotope variants. Explicit parents of otherwise unrecognized H names
are preserved by the name mapping. Resolving ambiguous source H names is outside
this normalization step.

The original observation is captured after this normalization. Output uses
`O`/`OXT`; the raw alias spelling is not archived. Already constructed
`ProteinStructure` objects apply the same checks to selected recognized peptide
residues. Malformed or H-dependent aliases are refused; otherwise admissible
pairs require re-reading the coordinate input to establish a normalized
observation. Canonical objects already using `O`/`OXT` keep their existing
observation.

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

### Zero-Occupancy Side Chains

By default the reader keeps selected coordinates with zero occupancy. Set
`omit_zero_occupancy_sidechain_heavy_atoms=True` on `StructureIngressOptions`
for `process_structure()` to treat eligible positions as missing.

The option runs after alternate-location and residue-variant selection. It
checks for an explicit zero in the input field, even when the atom has no
alternate location. Blank or unspecified fields and nonzero values that round
to zero are not declarations of zero. The reader does not fall back to another
alternate when it withholds the selected coordinate.

The option covers standard polymer side chains attached to the backbone only
at `CA`, with `N`, `CA`, `C` and `O` present and none declared zero. A hydrogen
is withheld only if it is also explicitly zero and its parent is withheld.
Whole-zero residues, backbone and terminal atoms, Pro rings, and retained
non-polymers are outside this option's scope.

Ingress refuses ambiguous source records rather than guessing which coordinate
carried a zero. It also refuses a projection that would remove a declared connection
(including `CONECT`) or charge, affect a declared side-chain link or nonzero
side-chain charge, leave a retained hydrogen without its parent, or disconnect
a retained template fragment. Applying the option does not request repair.

The withheld coordinates are excluded from the active source observation.
Inspect their history in `structure.provenance.ingress.zero_occupancy_omissions`:
`None` means the option was not applied, and an empty tuple means it was
applied without omissions. This history describes the original source, not
the current atom inventory. A canonical structure without that history
requires re-reading the original input to apply the option. Reusing a
structure with a receipt never removes a rebuilt atom a second time.

### Whole Zero-Occupancy Residues

Set `omit_zero_occupancy_polymer_residues=True` on `StructureIngressOptions`
to withhold isolated, internal standard polymer residues with only explicit
zero occupancies. This is independent of the side-chain option. All atom
records at the slot in the first model must declare zero, including H/D/T and
discarded alternate locations. Both flanks must have positively declared
`N`, `CA`, `C` and `O` and follow the source polymer sequence through the
slot. Terminal slots, consecutive zero runs, insertion codes and ambiguous
or conflicting sequence evidence leave the selected coordinates in place.
An otherwise eligible omission that would discard a declared connection or
formal charge is refused.

The reader does not align incomplete PDB coordinate records to `SEQRES`.
An unmodeled residue or an extra record in the same chain, such as water,
can prevent an exact match and leave a zero-occupancy residue in place.
For mmCIF, the reader checks the declared sequence position, subchain and
entity labels on the original atom rows for agreement with the polymer path.

Inspect the original residue ID, component, flanks and withheld coordinates
in `structure.provenance.ingress.zero_occupancy_residue_omissions`. An empty
tuple means the policy ran without omissions; `None` means it did not run.
The original coordinates are not used for repair. To fill a recorded gap,
request an explicit span reconstruction matching its slot and flanks and
provide a donor of the same component with positive occupancy on the heavy
positions used for the insertion and the backbone fit.

The in-memory record prevents geometrical peptide adjacency across the
recorded gap. Coordinate exports contain neither the withheld atoms nor that
history. A fresh read can still treat unusually close flanks as geometrical
neighbors even without a C-N topology bond. To retain the source gap
semantics, continue with the returned structure or reread the original input
with the option enabled.

## Original Chemistry Observations

The selected input is retained at `structure.provenance.ingress.observation`.
It keeps normalized selected atom identities, coordinates, reported charges, and
explicit connections, including observed H, D, and T. Repair can remove or recreate an
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

For structures built through the Python API, `StructureConstitution` takes
membership in `chains` or `ligands` as the role and normalizes `is_hetero`
accordingly. Supplied member objects are not mutated. This does not classify
components chemically or infer missing connections.

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

Existing peptide caps supply context only at their open ends: `ACE.C` can
link to the next backbone N, and `NME.N` can link to the preceding carbonyl
C. `NME.C` is a methyl carbon, not an outgoing peptide site. A cap's closed
end cannot promote a neighboring `HETATM` residue into the chain.

Hydrogen completion supports linked NME with methyl atom `C` in a chain
or under `LigandPolicy.KEEP` with a typed covalent link or an already
resolved bond. It adds one amide H and three methyl H when all four are
missing, without changing existing H/D/T. A bare retained `CONECT` does
not establish the cap's chemistry or change `DROP` behavior.

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
