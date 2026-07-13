# Changelog

## Unreleased

### Breaking changes

- require RDKit at runtime through the `rdkit` package and remove the former
  `refinement` extra and no-RDKit execution mode
- replace the partial public Bondi/Cordero radius tables with a bundled RDKit PeriodicTable radius snapshot
  of `GetRvdw` and `GetRcovalent` values for elements 1-118
- remove `BONDI_VAN_DER_WAALS_RADII_ANGSTROM`,
  `CORDERO_COVALENT_RADII_ANGSTROM`, `DEFAULT_COVALENT_RADIUS_ANGSTROM`, and
  `DEFAULT_VAN_DER_WAALS_RADIUS_ANGSTROM`; radius lookup now raises
  `UnknownElementRadiusError` instead of silently using a default for unsupported
  elements

### Changed

- reuse workflow branch evaluations across sibling selection, regression
  retention, budget pruning, and terminal reporting while evicting discarded
  branch results from the runtime cache
- consolidate retained-ligand RDKit hydrogen patch projection while rejecting
  invalid generated names and non-finite coordinates before materialization
- centralize NumPy geometry-placement vector degeneracy rules while preserving
  domain-specific error and diagnostic semantics, and reject non-finite or
  overflowing placement frames before they emit invalid coordinates
- report recoverable heavy-atom, terminal OXT, and polymer-hydrogen geometry
  placement failures as atom-scoped validation issues while preserving any
  successfully placed partial repair
- harden ingress normalization by selecting coherent residue-level altloc
  cohorts by mean occupancy with lexical tie-breaks, validating non-finite atom
  coordinates/occupancy/B-factors at the boundary, preserving first-model PDB
  CONECT records safely, and dropping source connectivity that refers to
  discarded altloc or source component variants
- tighten source-bond metadata validation so public topology metadata requires
  finite positive numeric reported distances rather than float-like strings
- expose unsupported topology as an explicit readiness state for polymer
  components whose expected topology cannot be resolved from supported chemistry
  evidence, including separate absent-vs-unsupported workflow deficit fields
- wrap public structure parsing and unsupported-format failures in project-owned
  errors so callers can catch `ProtrepairError` subclasses consistently
- model thiol H/D/T atoms on canonical CYS disulfides as typed chemistry
  contradictions, remove them through planner-selected topology-preserving
  normalization, and report chemistry work under the broader
  `CHEMISTRY_NORMALIZATION` workflow phase

## 0.1.0

Initial public release candidate of the rewritten `protrepair` package.

- expose a small release-facing API centered on `process_structure()`,
  `StructureIngressOptions`, `requested_process_goal()`, and typed workflow
  result contracts
- support canonical PDB/mmCIF ingress and egress through `gemmi`, including
  topology-backed connection emission
- model structure constitution, geometry, topology, provenance, semantic
  scopes, and observed state axes as typed canonical domain objects
- implement workflow planning over explicit heavy-atom completion, hydrogen
  completion, retained non-polymer hydrogenation, stereochemistry correction,
  side-chain packing, and local refinement actions
- support standard residues, bundled polymer-like nonstandard components, and
  bundled retained non-polymer/cofactor component templates
- support RDKit-backed local refinement and parser-readability diagnostics when
  the optional `refinement` dependency is installed
- package the vendored FASPR source and rotamer library for the optional
  side-chain packing backend
- include representative workflow regressions, fixture-backed refinement
  benchmarks, public API smoke tests, release-facing README examples, typed
  package metadata, and third-party licensing notices
