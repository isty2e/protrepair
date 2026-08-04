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

Import `PackingMode`, `PackingScope`, and `PackingSpec` from the same
`protrepair.workflow.contracts` facade when configuring reference or committed
side-chain packing.

Default construction requests no optional packing, refinement, external span,
or histidine operation. RDKit fallback remains enabled for retained components
whose chemistry cannot otherwise be resolved.

## Process Result

`ProcessResult` exposes:

| Attribute | Meaning |
| --- | --- |
| `structure` | Preferred final canonical structure |
| `repairs` | Ordered repair events applied to the selected result |
| `issues` | Structured warnings and errors |
| `analyses` | Optional `AnalysisBundle` |
| `requested_goal_report` | Per-goal outcomes; `process_structure()` returns an empty report when no goals were requested |
| `terminal_branch_report` | Branch evaluation details when available |

Useful methods include `has_errors()`, `has_warnings()`, `repair_count()`,
`error_count()`, `warning_count()`, `issue_count()`,
`requested_goal_completion_verdict()`, and
`requested_goals_fully_satisfied()`.

## Structure I/O

Import these functions from `protrepair.io`:

| Function | Contract |
| --- | --- |
| `read_structure(path)` | Normalize a PDB or mmCIF file into `ProteinStructure` |
| `read_structure_string(text, file_format)` | Normalize an in-memory coordinate block |
| `write_structure(structure, path, file_format=None)` | Atomically write a canonical structure |
| `write_structure_string(structure, file_format)` | Serialize to an in-memory coordinate block |

Writers project canonical topology and residue roles. They do not maintain a
separate writer-owned connectivity model.

## Supported Facades

| Facade | Purpose |
| --- | --- |
| `protrepair` | Main workflow entrypoint and common request/result contracts |
| `protrepair.workflow.contracts` | Workflow requests, policies, reports, and analysis result contracts |
| `protrepair.scope` | Semantic scopes and scope relations |
| `protrepair.state` | Target state axes and observation read models |
| `protrepair.analysis` | Analysis kinds and structured analysis results |
| `protrepair.io` | PDB/mmCIF ingress and egress |
| `protrepair.structure` | Canonical structure, topology, provenance, and labels |
| `protrepair.geometry` | Public geometric value objects and transforms |

These facades are covered by the release public-export contract. Symbols from
deeper modules are implementation details unless a facade re-exports them.
