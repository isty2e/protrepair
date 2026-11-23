# Public API Reference

Use the import paths listed here for application code. Planner, transformer,
chemistry, and diagnostics modules below these facades are internal.

## Main Entrypoint

`protrepair.process_structure()` accepts a source path or an existing
`ProteinStructure` and returns a `ProcessResult`.

| Parameter | Contract |
| --- | --- |
| `source` | `Path`, path-like string, or canonical `ProteinStructure` |
| `ingress` | Optional `StructureIngressOptions` for source normalization |
| `selected_source_chain_ids` | Optional tuple of source chain IDs, valid only for path-like input |
| `requested_goals` | Sequence of scoped target state values |
| `transform_requests` | Optional `WorkflowTransformRequests` for explicit operator choices |
| `planning_context` | Optional advanced planning context |
| `analyses` | Iterable of `AnalysisKind` values attached to the final result |

Path-based processing infers PDB or mmCIF from the source suffix. Use the I/O
facade when an explicit in-memory string or output format is required.

## Requested Goals

`requested_process_goal(scope=..., value=...)` validates and returns one
canonical workflow goal. Supported target axes include:

- `BackboneHeavyAtomCompletenessState`
- `SidechainHeavyAtomCompletenessState`
- `HydrogenCoverageState`
- `BondedGeometryState.WITHIN_TOLERANCE` with `WholeStructureScope`,
  `ChainSetScope`, or `ResidueSetScope`
- `OxtPresenceState` with a C-terminal `ResidueBoundaryScope`
- `ClashState` for local scopes
- `ClashPresenceState` for whole-structure scope

Scopes are imported from `protrepair.scope`; target state values are imported
from `protrepair.state`. Invalid scope and state combinations raise at the
request boundary.

`BondedGeometryState.WITHIN_TOLERANCE` asks the workflow to bring heavy-atom
bond lengths and angles within the diagnostic tolerances for standard polymer
residues in the selected scope. The tolerances also apply to deposited
coordinates. If a requested chain has no polymer residues or a requested
residue lacks qualified reference coverage, the goal remains unmet.

Only intra-residue terms in the selected scope can trigger an attempt for
this goal. The scope does not freeze other residues or chains: a candidate
can include atoms outside it when a bond or steric contact couples them to
the selected defect and the candidate passes the existing safeguards.

Request hydrogen completion separately if it is needed. Geometry repair
uses the current atom inventory and chemistry, and can return an unmet goal
when prerequisites or safe motion are unavailable. Final warnings cover the
whole polymer regardless of scope; those warnings do not trigger motion.

## Existing Hydrogen Lengths

`process_structure()` reports the X-H lengths of existing H, D, and T in
`result.hydrogen_length_report`. The workflow selects these sites at input
and measures them on the returned structure. If there are no input H/D/T,
an ordinary call omits the report. Parent-position conflicts also appear in
`result.issues`. A measured length without a reference remains unassessed.

Use `analyze_hydrogen_lengths()` to make the same read-only observations on
a structure without running the workflow. To grade a length, supply monomer
CIF text with a standard side-chain graph, distance targets and ESDs. Loading
the reference uses RDKit to validate its graph. No reference is downloaded
or bundled for this check.

```python
from pathlib import Path

from protrepair.analysis import HydrogenLengthReference, analyze_hydrogen_lengths
from protrepair.io import read_structure

structure = read_structure("model.cif")
observations = analyze_hydrogen_lengths(structure)

cysteine_reference = HydrogenLengthReference(
    Path("CYS.cif").read_text(encoding="utf-8"),
    component_id="CYS",
    source="my monomer library, pinned version",
)
graded = analyze_hydrogen_lengths(structure, references=(cysteine_reference,))
```

The result has one entry per selected present H, D, or T in structure order;
an empty result means no sites were selected. Use `scope=` to restrict the
selection. Parent evidence and the nearest-heavy veto still use the supplied
snapshot, including heavy atoms outside the selection. The veto does not
reassign a parent. If a heavy coordinate is nonfinite, the check cannot
establish which heavy atom is nearest. Backbone lengths may be measurable, but
this check only grades standard polymer side-chain terms whose scaffold,
attachments, and protonation state match the reference. Contradictory
attachment evidence can leave the whole residue unassessed; an independently
measurable length can also lack an applicable reference.
The side-chain graph can join the backbone only through CA, so proline is
outside the supported reference domain.

The default convention for this check is `HydrogenLengthConvention.UNKNOWN`.
With two applicable profiles it uses the outer limits of the two closed
`target +/- 4*ESD` intervals. Values between disjoint intervals are reported
separately from values inside an individual interval. The comparison reports
which individual intervals match and a signed `outer_margin_angstrom`; a
positive margin can still fall between intervals. Set `convention=` to
`ELECTRON_CLOUD` or `NUCLEAR` only when the coordinate convention is known.
Unassessed sites and parent-position conflicts are distinct from length
outliers. The comparison uses the stored coordinates rather than a projection
to PDB precision; a value near a bound can change grade after serialization.

Pass a `HydrogenLengthRequest` to narrow the workflow's observation scope,
grade lengths against supplied references, or request a repair:

```python
from protrepair import process_structure
from protrepair.analysis import HydrogenLengthConvention
from protrepair.workflow.contracts import HydrogenLengthRequest

checked = process_structure(
    structure,
    hydrogen_lengths=HydrogenLengthRequest((cysteine_reference,)),
)
adjusted = process_structure(
    structure,
    hydrogen_lengths=HydrogenLengthRequest(
        (cysteine_reference,),
        convention=HydrogenLengthConvention.NUCLEAR,
        repair=True,
    ),
)
```

Use `NUCLEAR` in the second call only if the model uses that coordinate
convention. A request without references can narrow the scope without
grading lengths. The scope limits both measurements and X-H warnings.
Reference outliers appear in `result.issues`; a gap between two convention
intervals is not a length warning.

Repair requires a declared convention and moves only out-of-window sites
whose assigned parent and input pose are unchanged. It runs once after the
workflow selects a branch, without adding hydrogens or changing their chemistry.
The selected hydrogen moves along its original bond direction to the
reference target. The batch is kept only if it preserves the assigned parents
on a no-CONECT PDB readback, meets the length limits at stored and PDB precision,
and does not introduce or worsen an assessed clash or near-covalent contact. If the
structure cannot be represented on that parser surface, repair is unavailable.
See `repair_outcomes` for in-window, unavailable, rejected, and repaired sites.
`branch_report` describes the search before this terminal operation.

Length observation and repair are separate from hydrogen completeness and
the heavy bonded-geometry goal.

## Ingress Options

`StructureIngressOptions` controls source interpretation before canonical
execution:

| Field | Default | Meaning |
| --- | --- | --- |
| `occupancy_policy` | `OccupancyPolicy.HIGHEST` | Select alternate atom cohorts |
| `mutation_policy` | `MutationPolicy.HIGHEST_OCCUPANCY` | Resolve residue variants |
| `ligand_policy` | `LigandPolicy.DROP` | Drop, keep, or reject non-water non-polymers when no retention rules are supplied |
| `non_polymer_retention` | `None` | Select whole waters and other non-polymers using [ordered retention rules](ingress-policy.md#keeping-selected-waters-cofactors-and-ions) |
| `retained_non_polymer_chemistry_overrides` | empty | Supply explicit retained-component chemistry |
| `use_ccd_orthophosphate_models` | `False` | Use the deposited chemical component model for retained inorganic phosphate |
| `omit_zero_occupancy_sidechain_heavy_atoms` | `False` | Treat eligible, explicitly zero side-chain coordinates as missing without requesting repair |
| `omit_zero_occupancy_polymer_residues` | `False` | Treat verified all-zero internal singleton residues as missing without requesting reconstruction |

Exact normalization precedence is defined in the [ingress policy](ingress-policy.md).

To prepare retained phosphate under a hydrogen-completion goal, set
`StructureIngressOptions(ligand_policy=LigandPolicy.KEEP, use_ccd_orthophosphate_models=True)`.
The [PO4](https://www.rcsb.org/ligand/PO4) model assigns charge -3 and no
hydrogens. [PI](https://www.rcsb.org/ligand/PI) assigns charge -2 and an H
on O4; [2HP](https://www.rcsb.org/ligand/2HP) assigns charge -1 and an H on
each of O3 and O4. These assignments use the deposited atom names, not the
order of the coordinate records. Added H and their model-assigned parents
appear in the hydrogen-addition repair event. Compatible observed H/D/T keep
their names and coordinates. If neither the atom name nor a recorded bond
identifies the parent, the repair uses the coordinates only when one
model-allowed oxygen fits the O-H distance and is the unique nearest heavy
atom in the phosphate.

A site-specific override takes precedence over the CCD model. Both must agree
with the observed atoms and their attachments. A phosphate linked to a metal
or covalently attached to another residue stays unchanged, even if the partner
was filtered out. If the selected graph conflicts with the input, that site
leaves the H goal unmet; other sites can still be prepared. Hydrogen coverage
counts atoms. If all H are present but their coordinates prevent bond
assignment, the repair reports that failure separately. Selecting a model
neither keeps a dropped ion nor requests a repair.

## Transform Requests

`WorkflowTransformRequests` carries explicit method or operator choices that
are not target states:

- orphan-fragment handling
- external span reconstruction specifications
- reference or committed side-chain packing
- backbone-window and repair refinement specifications
- histidine protonation
- retained non-polymer RDKit fallback permission

`ExternalSpanReconstructionSpec` pairs one canonical
`AbsentResidueSpanScope` with a contiguous donor residue span. Use
`build_alphafold_span_reconstruction_specs()` when a UniProt alignment and an
`AlphaFoldStructureArtifact` already identify the missing coverage gaps.

When the workflow requests at least two spans, one donor per distinct scope,
and no goals or other transforms, it tries pending spans in request order. A
refusal leaves the span absent but does not stop later requests. Each attempt
uses the current structure and the existing search budget. Requests that
offer competing donors for the same scope, or combine spans with other work
such as terminal H length repair, keep the ordinary branching search.

If a refusal is on the returned branch, `issues` includes its
`SPAN_RECONSTRUCTION_FAILED` error. In a span-only run, `has_errors()` can
therefore be true even if the workflow reconstructed all later spans.

Import `PackingMode`, `PackingScope`, and `PackingSpec` from the same
`protrepair.workflow.contracts` facade when configuring reference or committed
side-chain packing.

Default construction requests no optional packing, refinement, external span,
or histidine operation. RDKit fallback remains enabled for retained components
without a full active chemical reference. It cannot force geometric inference
for a known reference without admitted preparation.

## Retained H Facts

`RetainedNonPolymerChemistryReadinessFact`, exported from `protrepair.state`,
separates observed coverage from generation capability:

- `hydrogen_coverage_state` is `HydrogenCoverageState | None`. `None` means
  coverage cannot be established under the admitted interpretation; it does
  not mean no H are present or that H generation is required. Residue-level
  facts and projections evaluating that inventory propagate the same nullable
  coverage rather than treating an empty expectation as completion.
- `hydrogen_generation_supported` reports whether the selected chemistry
  admits whole-component generation, independently of the observed coverage.
- `hydrogen_updates_pending` identifies concrete admitted updates in the current
  snapshot, including independent amino-acid sites when whole coverage is
  indeterminate. `requires_hydrogen_completion()` includes those updates; a
  nullable coverage value alone never requests generation or admits refinement.
  Such independent updates can exist while whole-component generation support
  is false.
- `reference_hydrogens_verified` reports whether read-only reference comparison
  established the observed coverage. It can be true for partial coverage and
  does not expose the internal reference graph or parent correspondence.
- `supports_hydrogen_verification()` accepts admitted preparation or a comparable
  read-only inventory. It does not imply `is_supported()` or H generation.
- `has_complete_reference_hydrogens()` identifies complete reference-verified H
  without implying generation, topology or refinement support.
- `chemistry_failure` is an optional `RetainedNonPolymerChemistryFailure`
  with a `RetainedNonPolymerChemistryFailureKind` and readable `detail`.
  This typed field is new since 0.2.0.

A verified reference inventory can satisfy a whole-structure H goal even when
preparation remains unsupported. `chemistry_failure` may still explain why
missing H cannot be generated; it is not a blocker for already complete verified
H. Reference-backed parent correspondence can accept different H names without
changing source identities or coordinates. No topology is published, and local
refinement remains refused without its existing preparation/realization support. See
[reference validation](retained-ligand-policy.md#known-references-without-preparation)
for naming, source-state and stereo limits.

The `hydrogen_coverage_state` fields on `ProteinStructureObservation`,
`AtomScopeObservation` and `ResidueChemistryReadinessFacts` also allow `None`.
Check for `None` before reading `.value`. The polymer-only aggregate
`StructureChemistryReadinessFacts.hydrogen_coverage_state` also allows `None`
when an included polymer inventory cannot be verified. Retained-component facts
are evaluated separately. Known polymer components without admitted H preparation
are not treated as complete merely because only their backbone sites have H.
Existing H can be verified against the active reference and supported peptide
boundaries without authorizing generation, topology completion or refinement.

An indeterminate whole-structure H goal uses `RequestedGoalOutcome`:
`observed_state=None`, unmet status, blocking residue scopes and explanatory
`details`. The requested `HydrogenCoverageState` enum still has only
`NONE`, `PARTIAL` and `COMPLETE`; indeterminate coverage is not a target.

The default workflow capability registry does not currently admit residue- or
atom-scoped H goals containing retained ligands, even for an admitted component.
Such requests report `UNSUPPORTED`; use a whole-structure H goal instead.

## Process Result

`ProcessResult` exposes:

| Attribute | Meaning |
| --- | --- |
| `structure` | Preferred final canonical structure |
| `repairs` | Ordered repair events applied to the selected result |
| `issues` | Structured warnings and errors |
| `analyses` | Optional `AnalysisBundle` |
| `requested_goal_report` | Per-goal outcomes; `process_structure()` returns an empty report when no goals were requested |
| `external_span_reconstruction_report` | Per-donor request outcomes, or `None` when no spans were requested |
| `branch_report` | Retained candidates, selected node, and search stop reason |
| `relaxation_executions` | Observed native FF returns from batches along the selected ancestry, including sibling calls |

Useful methods include `has_errors()`, `has_warnings()`, `repair_count()`,
`error_count()`, `warning_count()`, `issue_count()`,
`requested_goal_completion_verdict()`, and
`requested_goals_fully_satisfied()`.

If search reaches `max_speculative_nodes`, processing returns the preferred
assessed candidate with a `WORKFLOW_SEARCH_EXHAUSTED` warning. Check the goal
report and remaining diagnostics before using it: reaching the limit says
nothing about whether the requested goals were met. `branch_report` distinguishes
planner-terminal candidates from candidates still on the frontier, and records
why search stopped. It replaces the former `terminal_branch_report` attribute.
There is no compatibility alias for that attribute. Terminal means that the
planner has no further proposals, not that the requested goals were met.
When terminal and frontier candidates remain together, the same configured
quality ordering compares them all. A frontier candidate can therefore win
despite pending preparation, including missing H. Any unmet requested goals
remain visible in its goal report; the exhaustion warning is not a success verdict.

The span report has one outcome per normalized request, in order, with a
zero-based `request_index`. Exact duplicate specifications share one outcome.
Each row lists the target positions still missing from the returned structure.

| Span status | Meaning |
| --- | --- |
| `APPLIED` | This request filled the target on the selected branch. |
| `ALREADY_PRESENT` | The target was complete at input and is still present. |
| `PRESENT_WITHOUT_THIS_REQUEST` | The target is complete, but this request did not fill it on the selected branch. |
| `NOT_RETAINED` | A trial succeeded, but the returned target is incomplete. |
| `FAILED` | The returned target is incomplete and every executed trial refused. |
| `NOT_ATTEMPTED` | The returned target is incomplete and no trial ran. |

The counts and `last_failure` include trials from branches that were not
selected. A refusal in the report is not necessarily an error in `issues`;
it describes what happened in that trial, not whether reconstruction is
possible under different conditions.

Native execution records describe returned calls, not successful repairs or
adopted coordinates. Each batch along the selected ancestry contributes all
its sibling calls, including rejected or unselected siblings. Calls in other
descendant subtrees are not included, so this is not a complete workflow-wide
execution log. Exceptions may leave native work unobserved. The records contain
settings, movable atoms, minimizer status, and backend version, not resumable
solver state.

Final `INVALID_GEOMETRY` warnings describe residual bonded geometry on the
returned polymer, including outside reconstructed spans. A warning does not
change the outcome of a separately requested hydrogen goal.

## Structure I/O

Import these functions from `protrepair.io`:

| Function | Contract |
| --- | --- |
| `read_structure(path)` | Normalize a PDB or mmCIF file into `ProteinStructure` |
| `read_structure_string(text, file_format)` | Normalize an in-memory coordinate block |
| `write_structure(structure, path, file_format=None)` | Atomically write a canonical structure |
| `write_structure_string(structure, file_format)` | Serialize to an in-memory coordinate block |
| `to_rdkit_molecule(structure)` | Copy selected canonical chemistry into a sanitized RDKit molecule |

Writers project canonical topology and residue roles. They do not maintain a
separate writer-owned connectivity model.

PDB output supports two-character chains and uppercase hybrid-36 numbers.
It permits coordinate rounding down to two decimal places and variable
B-factor precision, but does not silently shorten identifiers or clamp values.
Unrepresentable output raises `protrepair.errors.StructureSerializationError`,
a `ProtrepairError`; path-based failure leaves an existing file unchanged.
See [PDB Output Limits](ingress-policy.md#pdb-identifiers-and-output-limits)
for the supported envelope and when to use mmCIF.

### Native RDKit Export

`to_rdkit_molecule()` takes a prepared `ProteinStructure`, not a path or PDB
block. It returns an independent mutable `rdkit.Chem.Mol` with a full-precision
3D conformer. Atom indices match the structure's current `AtomIndex` values;
residue identity, explicit H/D/T and assigned charges are copied. Editing the
molecule does not update the structure.

Typed coordination becomes donor-to-metal `DATIVE` when exactly one endpoint
matches RDKit's M-atom metal grammar. Compatible aromatic representation may
normalize; declared aromatic double bonds must admit a compatible Kekule form.
This includes a complete alternating single/double ring with input aromatic
flags set to false: RDKit may return aromatic bonds while the original orders
and flags remain in the `protrepair_bond_order` and `protrepair_aromatic`
properties. Exact native aromatic-flag equality is not required; the canonical
structure is unchanged.
Stereo is assigned from coordinates. Export does not prepare chemistry, add H,
optimize geometry or establish force-field support.

Unresolved bonds, unknown metal charge, ambiguous coordination, unselected
H/radical chemistry and failed or chemistry-changing sanitation raise
`protrepair.errors.RdkitProjectionError`, a `ProtrepairError`. Unknown nonmetal
charge uses native zero with a knowledge flag; it is not assigned back to
canonical topology or original observations.

If the required RDKit runtime cannot be imported, the package and non-native
I/O remain importable; calling this conversion raises
`protrepair.errors.RdkitUnavailableError`.

The following native properties retain distinctions absent from standard
RDKit fields:

| Property | Location | Meaning |
| --- | --- | --- |
| `protrepair_formal_charge_known` | Atom | Whether current canonical charge is assigned |
| `protrepair_b_factor_known` | Atom | Whether the copied B factor is known; otherwise native temperature factor is zero |
| `protrepair_source_element` | Atom | Source element symbol, including D/T aliases |
| `protrepair_relationship_type` | Bond | Canonical chemical relationship |
| `protrepair_bond_provenance` | Bond | Provenance of that relationship |
| `protrepair_bond_order_known` | Bond | Whether canonical integral order is resolved |
| `protrepair_bond_order` | Bond | Input integral order, present only when resolved |
| `protrepair_aromatic` | Bond | Input aromatic flag before native normalization |
| `protrepair_omitted_hydrogen_bonds` | Molecule | Number of omitted noncovalent hydrogen-bond interactions |

Hydrogen-bond interactions are not chemical edges in this export. A known-charge
flag describes current selected chemistry, not whether the original file
reported that charge. Native export and PDB/mmCIF serialization remain separate
operations; successful export does not guarantee equivalent PDB re-reading.
Native atom metadata retains selected altloc labels. RDKit's own PDB writer
can recreate `B` or lowercase records that its default reader skips; use
ProtRepair's PDB writer on the `ProteinStructure` when blank selected-altloc
labels are needed.

`protrepair_formal_charge_known=False` means canonical topology has no assigned
formal charge. This is common for neutral C and H; it is not a confidence score
or a warning that the native zero is invalid. Successful export has checked
that zero against the explicit graph and H inventory through RDKit sanitation
and rejected unselected radical chemistry. These checks do not establish a
unique or biologically correct protonation state.

RDKit's default binary and pickle settings omit custom properties. Preserve
these fields explicitly when creating a binary payload:

```python
from rdkit import Chem

payload = molecule.ToBinary(Chem.PropertyPickleOptions.AllProps)
restored = Chem.Mol(payload)
```

This selects properties for that payload without changing process-wide pickle
defaults. Native atom-name metadata uses Gemmi's element-aware PDB padding, so
protein `CA` and elemental calcium retain distinct alignments. Longer atom
names remain intact in native metadata; the PDB writer's format limits still
apply when a caller later writes a PDB block.

## Supported Facades

| Facade | Purpose |
| --- | --- |
| `protrepair` | Main workflow entrypoint and common request/result contracts |
| `protrepair.workflow.contracts` | Workflow requests, policies, reports, and analysis result contracts |
| `protrepair.scope` | Semantic scopes and scope relations |
| `protrepair.state` | Target state axes and observation read models |
| `protrepair.analysis` | Read-only hydrogen length checks, analysis kinds, and structured results |
| `protrepair.io` | PDB/mmCIF ingress and egress, and native RDKit export |
| `protrepair.sources` | UniProt and AlphaFold source records, retrieval outcomes, and sequence projection |
| `protrepair.structure` | Canonical structure, topology, provenance, and labels |
| `protrepair.geometry` | Public geometric value objects and transforms |

These are the supported application import paths. Symbols from deeper modules
are implementation details unless a facade re-exports them.
