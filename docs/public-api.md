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
- `OxtPresenceState` with a C-terminal `ResidueBoundaryScope`
- `ClashState` for local scopes
- `ClashPresenceState` for whole-structure scope

Scopes are imported from `protrepair.scope`; target state values are imported
from `protrepair.state`. Invalid scope and state combinations raise at the
request boundary.

## Ingress Options

`StructureIngressOptions` controls source interpretation before canonical
execution:

| Field | Default | Meaning |
| --- | --- | --- |
| `occupancy_policy` | `OccupancyPolicy.HIGHEST` | Select alternate atom cohorts |
| `mutation_policy` | `MutationPolicy.HIGHEST_OCCUPANCY` | Resolve residue variants |
| `ligand_policy` | `LigandPolicy.DROP` | Drop, keep, or reject retained non-polymers |
| `retained_non_polymer_chemistry_overrides` | empty | Supply explicit retained-component chemistry |

Exact normalization precedence is defined in the [ingress policy](ingress-policy.md).

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
  admits generation, independently of the observed coverage.
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
`StructureChemistryReadinessFacts.hydrogen_coverage_state` remains non-nullable;
retained-component facts are evaluated separately.

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

Native execution records describe returned calls, not successful repairs or
adopted coordinates. Each batch along the selected ancestry contributes all
its sibling calls, including rejected or unselected siblings. Calls in other
descendant subtrees are not included, so this is not a complete workflow-wide
execution log. Exceptions may leave native work unobserved. The records contain
settings, movable atoms, minimizer status, and backend version, not resumable
solver state.

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
| `protrepair.analysis` | Analysis kinds and structured analysis results |
| `protrepair.io` | PDB/mmCIF ingress and egress, and native RDKit export |
| `protrepair.sources` | UniProt and AlphaFold source records, retrieval outcomes, and sequence projection |
| `protrepair.structure` | Canonical structure, topology, provenance, and labels |
| `protrepair.geometry` | Public geometric value objects and transforms |

These are the supported application import paths. Symbols from deeper modules
are implementation details unless a facade re-exports them.
