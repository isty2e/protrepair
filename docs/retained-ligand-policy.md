# Retained-Ligand Chemistry Policy

Retained non-polymers include ligands, cofactors, and other non-polymer
components that remain in the canonical structure. Their coordinates are
preserved unless a selected transformation has explicit authority to change
them.

## Chemistry Resolution

ProtRepair resolves retained-ligand chemistry in this order:

1. explicit chemistry evidence supplied at ingress
2. a bundled component template that can add hydrogens
3. RDKit fallback inference when fallback is allowed

Existing hydrogen names are preferred when they are compatible with the
resolved chemistry. Explicit overrides are validated against the retained
heavy-atom set before workflow execution. Invalid SMILES/evidence mappings or
mismatched overrides raise `ValueError`; they do not silently fall through to
another chemistry source.

## Connectivity And Existing Hydrogens

A source CONECT record can identify two connected atoms without declaring a bond
order. After validating the ligand chemistry, ProtRepair fills missing bond
attributes from the selected template, override or RDKit inference. The source
record and its reported attributes remain separate from the resolved graph.
An explicitly reported order or relationship is not silently replaced.

A complete H inventory does not necessarily have a complete bond graph. In that
case, the planner can select a topology-only transformer. It checks H attachment
against the admitted chemistry and publishes compatible bonds without moving,
renaming or replacing atoms, including source H isotopes. Matching the total H
count is not enough: each H needs a compatible local heavy-atom anchor, without
extra covalent attachments to another H or another residue. Source hydrogen bonds
and metal coordination records remain separate from covalent H attachments.

Hydrogen correspondence uses compatible local parent groups before comparing
coordinates. This preserves existing H/D/T identities without considering every
hydrogen pair in a large ligand. Ambiguous attachments do not become evidence
of chemical completeness. If a required correspondence cannot be found within
the per-residue resource bound, the attempt reports the reason and leaves the
residue unchanged; it does not discard source hydrogens to continue.

Declared source H attachments remain preservation constraints. Model-derived
parents can guide correspondence when their H counts fit the admitted chemistry;
incompatible model groups can be reassigned without replacing source atoms. If
nearest-pair matching is incomplete, bounded augmenting paths look for a complete
compatible assignment before refusing the residue. This is identity matching,
not a protonation selector or a global geometry optimization.

If the proposed graph contradicts a source connection, that ligand stays
unchanged and receives a chemistry diagnostic. Other ligands can still be
prepared. Local FF readiness reports unresolved chemistry, missing heavy-atom
topology and missing H topology separately.

### Stereochemistry In RDKit Placement

For mapped SMILES, preparation and readiness compare declared stereochemistry
with the current 3D pose. A confirmed opposite tetrahedral center or double-bond
configuration leaves the ligand unchanged with a diagnostic. Unspecified SMILES
stereo and geometry that RDKit cannot assign are not treated as contradictions.
This check does not choose a protonation state.

RDKit fallback uses stereochemistry observed from the input pose. Both placement
paths recheck the generated coordinates. If a new tetrahedral H is on the wrong
side, ProtRepair tests both normals of the three fixed neighbors' plane and selects
the side consistent with the declaration, retaining the native parent distance.
It does not use an unassigned center's small displacement to choose that side.
A mismatch that cannot be corrected with the original atoms fixed is refused;
the operation does not move source atoms to make them agree with the selected graph.

After correspondence restores original H/D/T names and coordinates, native
configuration is checked again. This is not a separate H-geometry test: when
heavy neighbors already determine the configuration, RDKit may not use the H
position. A displaced or reflected source H can therefore pass this check.
See the [native tetrahedral observer](https://github.com/rdkit/rdkit/blob/Release_2026_03_2/Code/GraphMol/Chirality.cpp#L3282-L3308).

Retained-H completion keeps existing H/D/T fixed during its clash cleanup,
including on the template path, not only with overrides. Newly generated eligible
H can still undergo cleanup.

These are placement and source-preservation checks, not an optimized-geometry
certificate or a general validation of template-reference stereochemistry.

### Fixed-Template Aromatic Bond Orders

Fixed-template preparation currently requires reported bond orders to match the
template's particular Kekule form. A chemically equivalent alternative can pass
ingress but still be refused during preparation. For example, FMN with repeated
CONECT records selecting the opposite aromatic-ring double bonds receives
`CHEMISTRY_CONTRADICTION`, even though the difference need not be a chemical
contradiction. The ligand's atoms and bonds remain unchanged.

This also affects inputs with complete hydrogens: topology-only repair cannot
resolve that representation, and local force-field readiness remains blocked.
Connectivity-only records and orders matching the template do not have this
limitation.

Fixed-template resonance selection is deferred rather than handled by ignoring
reported orders. It needs a shared resolution rule for the template graph,
hydrogen inventory, charges and stereochemistry, used by both preparation and
readiness. Revisit this limitation before extending the resolver's support to
alternate fixed-template aromatic forms.

## Incomplete Templates

A template can place hydrogens on compatible heavy atoms that are present, even
when part of the component is missing. This applies to templates with nonzero
formal charges as well as neutral templates. Partial placement is not evidence
that the original hydrogens are wrong. If it cannot preserve all of them, ProtRepair
leaves the residue unchanged and reports `UNSUPPORTED_HYDROGENATION`, identifying
the missing heavy atoms and unmatched hydrogens.

Failed hydrogenation leaves the residue out of the subsequent H clash cleanup.
Other retained components can still be completed. This does not certify the
unchanged residue as chemically complete or ready for force-field refinement.
Validated chemistry overrides remain a separate route for changing the selected
chemistry, but they do not authorize silently dropping a source connection.

## Formal Charges And Template Admission

Selected chemistry includes formal-charge requirements as well as bonds and H
inventory. Preparation checks those requirements before publishing a residue's
changes. A known input charge that disagrees with the selected chemistry leaves
the residue unchanged with a diagnostic; an unspecified charge is not evidence
of neutrality.

An explicit mapped SMILES supplies the heavy atoms' formal charges, including
zero. Successful preparation writes missing assignments into canonical topology,
and readiness requires them to be present. Bundled templates fill missing
nonzero charges only when the charged atom and its template heavy-atom neighbors
are present. Other parts of an incomplete component can still receive H. A missing
local neighbor leaves a new charge unassigned; it does not erase a reported charge.
Templates leave unspecified neutral atoms unassigned. A template's
neutral requirement still rejects a contradictory known input charge. PDB and
mmCIF output use the resulting topology, not a separate writer charge policy.

All templates require compatible present heavy-atom names and, where the template
declares elements, matching elements. A partial scaffold can receive supported H
and charge assignments, but is not considered complete or ready for refinement.
Extra heavy names and contradictory declared elements are rejected even when the
template does not declare formal charges. The local-neighbor charge rule is a
structural applicability check, not a protonation prediction or a claim that a
fragment is a complete chemical species.

Retained components use the full component inventory. For example, SAH and SAM
contain an atom named `OXT`; it is not removed by the polymer free-terminal
projection.

The bundle includes sulfate (`SO4`), a zero-H component with a selected -2
graph/charge assignment, not an organic fallback or a generic metal/ion exemption.
Unspecified ionic charges can be resolved without changing its atoms or
coordinates. The isolated zero-H template cannot admit source H, external
covalent attachments or untyped external connections, whether those connections
come from the source or an earlier transformation. Typed hydrogen bonds and
metal coordination are preserved separately. This template does not
admit covalent derivatives, infer pH-dependent populations or validate a metal
potential.

## RDKit Fallback

RDKit fallback is enabled by default for retained components without usable
explicit or bundled chemistry. A successful fallback emits the structured
warning `RETAINED_NON_POLYMER_FALLBACK_USED` and records the RDKit backend
version in the resolved chemistry facts.

The fallback is conservative:

- Existing hydrogens are reused only when local anchor geometry and source
  topology are compatible.
- Unsupported hetero multiple-bond inference is rejected.
- Inference that would change retained-ligand stereochemistry is rejected.
- A rejected inference leaves the retained ligand unchanged and reports the
  reason instead of guessing.

Set
`WorkflowTransformRequests(allow_retained_non_polymer_rdkit_fallback=False)`
to require a bundled template or explicit chemistry evidence. If requested
hydrogen completion would otherwise need RDKit, the workflow emits
`RETAINED_NON_POLYMER_FALLBACK_BLOCKED` and leaves that retained component
unchanged.

## Scope Boundaries

The fallback infers chemistry for the heavy atoms already present. It is not a
general heavy-atom completion mechanism for arbitrary nonstandard components.
Single-atom metals and ions are retained as not-applicable chemistry rather
than passed through organic-molecule hydrogenation.

Template H coverage still has a separate limitation: hydrogens anchored to atoms
named `N` or `OXT` share polymer-only exclusions. This leaves SAH/SAM amine H and
SAH HXT unplaced even on complete heavy scaffolds. Full component-specific H
coverage needs those exclusions separated from retained-component chemistry.

## Component Dictionaries

Coordinate files and chemical-component dictionaries have different roles.
PDB/mmCIF ingress keeps typed source coordination records, including declarations
to partners removed by normalization. The component-dictionary reader instead
builds covalent templates with integer atom charges. It rejects coordination or
metal-pi flags, unresolved metal flags, and incomplete bond columns rather than silently
converting them into ordinary single bonds.

Explicit unknown atom charges (`?` or `.`) are also rejected by that reader. An
absent charge column uses the dictionary's
[default of zero](https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_chem_comp_atom.charge.html);
an explicit null cell does not. This is dictionary parsing, not a policy for
assigning charges to atoms in a deposited structure.

## Excluded Coordination Partners

If normalization removes a source-declared partner of a retained metal, the
original observation keeps its identity and connection metadata without adding
an atom or bond to the selected structure. Final processing emits a non-blocking
`SOURCE_COORDINATION_PARTNER_OMITTED` warning while that partner remains absent.

This warning does not restore water, infer oxidation or protonation, or certify
a complete metal site. It covers typed source declarations, not coordination
guessed from distances or untyped CONECT records. A missing endpoint, another
model, or an unchosen alternate location is not reported as a policy exclusion.

Output contains only selected atoms and their topology. Declarations to excluded
partners are not written as dangling bonds, so re-reading the output does not
recover that original off-graph evidence. Keep the processing result and original
input when the warning's source history matters.
