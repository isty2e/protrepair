# Retained-Ligand Chemistry Policy

Retained non-polymers include ligands, cofactors, and other non-polymer
components that remain in the canonical structure. Their coordinates are
preserved unless a selected transformation has explicit authority to change
them.

## Chemistry Resolution

ProtRepair resolves retained-ligand chemistry in this order:

1. explicit chemistry evidence supplied at ingress
2. an active component template that supports the requested preparation
3. a registered coordination family, when its scaffold and site are supported
4. RDKit fallback inference when fallback is allowed

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

### Connected Components

An isolated template or residue-local SMILES does not describe a covalent
adduct. Every external covalent-like or untyped edge must be explained by
the selected chemistry, regardless of its provenance. Unsupported attachments
leave the residue unchanged with a diagnostic rather than adding the isolated
component's H inventory. Typed hydrogen bonds and metal coordination are not
covalent substitutions; preserving them does not certify a coordination site.
These connection checks inspect relationships already present in canonical
topology; they do not infer missing external bonds from spatial proximity.

For a complete neutral aldehyde scaffold, one deposited carbonyl `C`--polymer
backbone `N` single link can select an acyl-amide application. The carbonyl H
is absent from that application; the methyl H inventory is retained. Existing
carbonyl H, conflicting charges, ambiguous links or unsupported partners are
not removed to make it fit. The original isolated template remains available
when there is no covalent attachment.
The polymer N must have no other external covalent or untyped attachment;
an existing peptide bond is not displaced to admit a cap.

The topology/H transformer realizes the admitted external order and preserves
the source link metadata. Readiness requires that realized graph, and polymer
N chemistry recognizes the cap only after its neutral acyl-amide graph exists.
An unresolved deposited link alone is not enough. Canonical CYS SG--SG
disulfides retain their no-thiol-H inventory; mapped evidence must describe a
compatible neutral terminal-thiol precursor.
Both sulfur endpoints must have compatible source charges and no competing
external covalent or untyped attachment at SG. Links at another atom on the
partner residue do not consume the sulfur's attachment site.

### Explicit Coordination Chemistry

Mapped SMILES can describe covalent bonds and directed donor-to-metal bonds in
the same component. ProtRepair keeps those relationships separate: a dative
bond does not consume the donor's covalent valence. An untyped source connection
can acquire a compatible selected relationship while retaining its original
metadata. An explicitly covalent declaration is not silently changed to dative.

Retained nitrogen preparation supports unprotonated aromatic N with two carbon
neighbors for Fe, Zn, Cu, Co, Ni, Mn, Cd and Hg, and primary sulfonamidate
`NH(-1)` for Zn. These are bounded preparation capabilities, not a complete
catalogue of chemically possible donors. Active templates, mapped overrides and
fallback inference must fit the admitted donor chemistry and preserve current
and original H/D/T, known charges and declared partners. Unsupported donors or
missing original partners leave that residue unchanged. Generated donor H can
use a validated partner position as a placement anchor; this does not move the
heavy atoms or authorize moving deposited H.

Standard polymer sites have their own rules. Typed HIS coordination excludes H
on the donating ring N. Supported ASP/GLU oxygen and linked peptide carbonyl
coordination do not select an acid/anion state. Coordinated CYS can retain
source-supported thiol or thiolate chemistry. With one typed `SG--HEM.FE` link,
polymer preparation prefers thiolate (`SG -1`, no sulfur H) when the active HEM
template is stock-compatible and its complete heavy scaffold preserves the
original atom identities. Source H/D/T or a known neutral sulfur charge instead
selects neutral thiol. Other admitted metal partners, incomplete or custom HEM
chemistry supply no default; without source evidence or an explicit choice,
sulfur protonation remains unresolved.

This removes the neutral thiol default at non-heme metal sites, including Zn--CYS.
With neither source H nor known SG charge, preparation adds no HG, leaves SG
charge unknown and warns for each unresolved residue. Unresolved sites do not
contribute an expected H inventory, so the H goal can still be satisfied; that
does not certify their protonation. Local refinement refuses an execution region
containing an unresolved polymer microstate, including fixed context residues.
For example, in 6NBB, selecting A:46 also includes unresolved A:174 and is refused.

The public API has no CYS protonation selector. To resolve an admitted site,
provide chemically consistent SG formal charge or attached H in the input:
`SG -1` selects thiolate, while `SG 0` or source H selects neutral thiol.
For PDB input, charge columns 79 and 80 accept `1-` for thiolate and `0+` or
`0-` for explicit zero; a blank field is unknown, not neutral. Supplying HG is
another way to specify a thiol. In mmCIF, use `-1` or `0` in
`_atom_site.pdbx_formal_charge`.
Contradictory charge and H are reported as conflicts. An explicit choice here
refers to the internal microstate model, not an additional public request option.
Resolving protonation alone does not establish metal-site force-field support.

This is a preparation convention, not a universal rule for heme proteins:
[neutral thiol ligation has experimental support](https://doi.org/10.1073/pnas.0737142100).
Saved compatible choices retain their authority; resetting a choice reconsiders
the original evidence and current preparation preferences. Ordinary uncoordinated
CYS keeps its existing H placement path. None of these rules estimates pKa or
metal oxidation state.
Multiple HIS ring donors or unsupported donor/partner combinations remain
unresolved rather than forcing the standard site into a different chemistry.

Readiness checks the selected coordination relationships even when all H are
already present. The H or topology-only transformer publishes the admitted graph
without losing source connection metadata. This does not establish metal-site
force-field support.

### Automatic Coordinated-Component Preparation

For components without an active template or explicit override, a packaged
catalogue supplies finite mapped alternatives for IMD, PIM, CPZ, HME and AZM.
Component codes select chemical data; the same scaffold, donor and source checks
apply to each family. External donors need a supported typed coordination
relationship. The process does not infer missing partners from distance.
This preparation also applies when organic RDKit fallback is disabled.

The complete heavy scaffold, elements, declared stereo, original and current
H/D/T attachments, known charges and reported bond orders must agree with an
alternative. Other registered site members and supported polymer partners also
constrain that choice. A contradictory or unsupported recognized site stays
unchanged with a diagnostic; it does not fall through to organic inference.
Explicit mapped evidence and active component templates retain precedence.

Preparation first prefers compatible neutral bound imidazoles and deprotonated
HME propionates. These are preparation defaults, not pKa predictions. It then
publishes the H inventory, bonds and atom charges shared by the remaining
metal-state alternatives for that ligand chemistry, not the intersection of
every catalogue state. In particular, HME's Fe(II)/Fe(III) alternatives do not
assign an unknown Fe charge. A source-assigned charge is retained when
compatible, and a native H-placement representative cannot publish its
otherwise unresolved metal state. The result carries a coordinated-preparation
informational diagnostic, not an external-override or organic-fallback label.

The HME catalogue fixes negative pyrrole charges at `NA` and `NC`. Other
placements, including `NB`/`ND`, are not normalized and may be refused even when
they describe chemically equivalent forms. Carboxylate alternatives do not
extend that coverage to macrocycle resonance representations.

The ligand catalogue and the polymer heme--CYS preference are separate chemistry
decisions. A complete H inventory does not validate the metal site's protonation,
oxidation state or potential; a source-supported neutral heme-bound CYS may still
retain its sulfur H.

Fresh readiness checks the realized H, charges and typed relationships, including
when no new H is needed. Supported ligands can supply a fixed passive calculation
graph even with organic fallback disabled; coordination is not converted to a
covalent FF bond. This path does not authorize movable template-less ligands,
metal optimization, or oxidation/spin-state prediction. A broader predictor or
metal potential would need a separate chemistry and execution contract.

H completion does not guarantee that a proximity-only PDB parser will infer the
same graph: it can still misread heme N--Fe relationships as covalent bonds.
Topology-preserving RDKit export also requires an assigned metal charge; this
preparation does not invent one to make export succeed.

### Atomic Ions

A retained single metal or halogen atom has a zero-H inventory, even when organic
RDKit fallback is disabled. Its charge stays as supplied, including unknown;
zero-H applicability is not an oxidation-state assignment. If H/D/T accompanies
that heavy atom, a molecular template or mapped evidence is needed. The atomic
rule does not invent a hydride or hydrogen halide, or hide source atoms removed
from the current inventory.

### Glycosidic Links

An explicit single covalent link can select a pyranose product inventory.
The selected component graph must contain one saturated six-membered sugar
ring and an aldose-like anomeric carbon with declared stereochemistry.
The donor's leaving hydroxyl is replaced by the deposited partner: a neutral
alcohol oxygen on another supported pyranose, or polymer ASN ND2. An occupied
acceptor oxygen receives no hydroxyl H; linked ASN receives one amide H rather
than two. Methyl glycosides can act as acceptors, but not as donors through
their anomeric O--methyl group.

Sugar roles come from the selected graph, not the component ID or atom names.
The protein acceptor is specifically the standard ASN ND2 carboxamide.
The bundle includes NAG, MAN, BMA, FUC, FUL, GAL, GNX and IDY; a compatible
user template uses the same path. FUC and GAL use their
[CCD](https://www.wwpdb.org/data/ccd) graphs and ideal coordinates, without
component-specific placement rules. Isolated templates retain their original inventories.
Both preparation and readiness use the linked product's heavy atoms, H,
charges and bonds. Local FF context retains covalently connected partners as
fixed atoms even when they lie outside the spatial shell; it does not enlarge
the requested motion. This rule also applies to disulfides and other admitted
non-peptide covalent links, so local refinement that previously failed at such
a boundary can now run.

Admission is residue-local. It checks the residue's scaffold and actual 3D
orientation, including the donor's substituted partner, and the chemistry and
exclusive occupancy of both ends of each incident link. New stereocenter H is
checked before publication. A stereo error or unrelated missing substituent in
another residue does not veto an otherwise supported product. An ambiguous
link, incompatible endpoint or unknown partner role still blocks preparation
at that link.

Original leaving atoms, excess source H/D/T, incompatible charges or
contradictory geometry are not removed or moved to make a link fit. Only neutral,
non-source H on a newly occupied hydroxyl can be removed. If an ASN link cannot
be interpreted, its ND2 H inventory stays unresolved; independently known CA/CB
hydrogens can still be added. Readiness remains partial rather than treating
that known subset as complete.

This is not general carbohydrate reconstruction. Missing heavy atoms, absent
or untyped links, furanose/ketose donors, ambiguous rings and other acceptor
families remain unsupported. Proximity alone does not establish a glycosidic
link. Covalent-adduct preparation also remains unsupported by this resolver.
The actual-pose stereo checks described here apply to the linked path. Isolated
fixed-template preparation does not yet certify every declared center against
the deposited pose; that broader admission policy remains separate work.

### Unresolved Polymer H At External Links

A polymer template describes an isolated residue's fixed H inventory. It does
not explain every substituent in a deposited structure. If a fixed H parent
has an external covalent, disulfide or untyped heavy-atom connection that no
supported product model explains, ProtRepair leaves that parent's H inventory
unresolved. Other parents in the same residue can still receive H. Existing
H/D/T, coordinates, known charges and source connections are preserved; the
unresolved H is not rotated by rotatable-H cleanup.

An H-completion goal remains unmet at these sites, with a warning on the final
result, even when there is no further supported H to add. The
`unresolved_polymer_hydrogen` warning describes the current residue, not a failed
attempt. Branch scores and final output include it once per affected residue
within the requested H or local preparation scope; repeated H preparation does
not add copies, and resolving the site removes the warning. Other execution
warnings keep their existing counts. Local refinement
including the unresolved site is not ready. This prevents an isolated hydroxyl
H from being added to an oxygen already linked to another heavy atom; it does
not infer the product's charge or remove an H already present in the input.

This also applies to metal connections recorded only as untyped PDB `CONECT`.
A metal element or a short distance does not turn an untyped edge into
coordination. A chemically appropriate `LINK` or mmCIF `_struct_conn` record can
resolve that relationship ambiguity. Typed metal coordination follows its
existing preparation policy; typing a bond does not by itself add support for
an otherwise unsupported product.

SER/THR O-linked glycans remain unsupported by the glycosidic product model,
which currently admits polymer ASN ND2 only. Their linked oxygen therefore
receives no new hydroxyl H and remains unresolved. Supporting O-glycosylation
requires a separate chemical interpretation, not a component-specific exception.

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

The final cleanup in `add_hydrogens` also keeps its input ligand H/D/T fixed,
including when retained completion runs as a local-refinement prerequisite.
An explicitly requested FF stage may move those atoms within its declared domain;
cleanup preserves that resulting pose rather than restoring earlier coordinates.

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

## Cysteine-Linked Terminal Acrylamides

A complete selected terminal acrylamide can describe a product when its beta
carbon has one explicit covalent link to polymer CYS SG. ProtRepair identifies
the neutral N--C(=O)--CH=CH2 motif from the graph, not the component ID or ligand
atom names. A close C--S contact alone does not establish this chemistry.

The product has two methylene carbons, a single alpha--beta bond, and neutral
sulfur with two single heavy bonds and no thiol H. For example, the isolated
[QS7 CCD record](https://files.rcsb.org/ligands/download/QS7.cif) has 23 H; its
Cys552-linked product in [6V9C](https://www.rcsb.org/structure/6V9C) has 24 ligand
H. The covalent modification is reported in the
[FGFR4 study](https://doi.org/10.1021/acsmedchemlett.9b00601).

Reaction-site H are placed against the current carbon and sulfur coordinates.
The all-protium methylene groups introduce no new stereocenter; declared ligand
centers elsewhere must retain their mapped 3D orientation. Deposited atoms, H/D/T,
coordinates, scalar values, known charges and link metadata are preserved.
Only provably generated thiol H can be removed, together with successful ligand
product publication. Unresolved SG chemistry does not suppress independently
known CYS carbon H, but it keeps the residue's H readiness partial.

Readiness includes the partner-local CB--SG bond as well as the ligand graph
and external C--S link. A complete H inventory does not hide a missing bond,
unresolved relationship or unresolved order: the existing topology-only action
can publish the admitted single bond without moving atoms. A noncovalent CB--SG
relationship is a contradiction, regardless of provenance, and is not replaced
silently.

Once the product graph is realized, the matching bundled dictionary excludes
the isolated alkene's reaction-site bond, angle and plane targets. Native FF
parameters act on the product graph; unrelated targets remain in place. A caller
dictionary equal in value to the bundled precursor receives the same projection;
dictionaries with different target values are unchanged. Native parameter coverage
does not establish good optimization or conformational quality.

Missing scaffold atoms, incompatible source H, charges, reported orders or stereo,
and multiple covalent attachments leave preparation unresolved with a diagnostic.
Residue-local SMILES overrides cannot describe this coupled product. Other
electrophiles, substituted reaction-site carbons, other nucleophiles, pKa and
reaction kinetics are outside this support boundary.

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

The bundle also includes standalone sulfanilamide (`SAN`) and cyclic AMP
(`CMP`), using their neutral CCD graphs and ideal references. These are
preparation defaults, not physiological protonation predictions. Explicit
source-compatible overrides can select other chemistry. Their standalone
support does not imply covalent-adduct or coordination support. CCD `ACE`
is stored as its full isolated acetaldehyde graph; the linked acyl application
above is selected from graph and context, not the component identifier.

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

## Native RDKit Handoff

`protrepair.io.to_rdkit_molecule()` copies selected canonical chemistry without
PDB parsing. It retains explicit H/D/T, assigned charges and full-precision
coordinates, and directs typed coordination from donor to metal. Unknown metal
charge is refused, not interpreted as a neutral metal.

This is representation, not a method for assigning oxidation, spin or
protonation states, and it does not establish metal-site FF support. Unresolved
chemical bonds, unselected H/radicals and sanitation that changes the selected
chemistry are refused. Noncovalent hydrogen-bond interactions are omitted with
an explicit count. The [API reference](public-api.md#native-rdkit-export)
describes the native properties and provisional representation of unknown
nonmetal charges.

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
