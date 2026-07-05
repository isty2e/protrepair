# Changelog

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
- harden ingress normalization by selecting coherent residue-level altloc
  cohorts, validating non-finite atom coordinates/occupancy/B-factors at the
  boundary, preserving first-model PDB CONECT records safely, and dropping source
  connectivity that refers to discarded altloc or ligand variants
- expose unsupported topology as an explicit readiness state for polymer
  components whose expected topology cannot be resolved from supported chemistry
  evidence, including separate absent-vs-unsupported workflow deficit fields
- wrap public structure parsing and unsupported-format failures in project-owned
  errors so callers can catch `ProtrepairError` subclasses consistently
