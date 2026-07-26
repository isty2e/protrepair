# ProtRepair Documentation

The root [README](../README.md) covers installation and a first workflow. The
documents here explain the behavior behind specific options and the contracts
maintainers need when changing them.

## User-Facing Behavior

- [Ingress normalization](ingress-policy.md): model selection, source variants,
  numeric validation, and polymer/retained residue roles.
- [Retained-ligand chemistry](retained-ligand-policy.md): templates, explicit
  chemistry evidence, RDKit fallback, and strict opt-out behavior.
- [Histidine protonation](histidine-protonation.md): the opt-in PRAS ratio
  method and its assignment contract.
- [Analyses](analysis-policy.md): Ramachandran categories and coarse
  secondary-structure labels.

## Engineering And Scientific Reference

- [Topology and bond egress](topology-bond-policy.md): canonical bond truth,
  provenance, repair requirements, and PDB/mmCIF projection.
- [Atomic radii](radius-policy.md): radius sources, defaults, isotope handling,
  and near-covalent detection.
- [FASPR runtime](faspr-runtime-policy.md): hydrogen ownership, retained
  components, packaged assets, and custom executable use.
- [Release checklist](release-checklist.md): supported matrix, verification,
  artifact contents, and tagging prerequisites.

## Public API Map

- `protrepair`: `process_structure()` and top-level workflow entrypoints.
- `protrepair.workflow.contracts`: ingress options, requested goals, transform
  requests, and result contracts.
- `protrepair.scope`: semantic scopes used by requested goals.
- `protrepair.state`: target and observed state axes.
- `protrepair.analysis`: analysis requests and structured results.
- `protrepair.io`: PDB/mmCIF read and write boundaries.
- `protrepair.structure`: canonical structure and label types for callers that
  need lower-level inspection.

These package facades are the supported import paths. Internal transformer,
planner, chemistry, and diagnostics modules may change without serving as
public API.
