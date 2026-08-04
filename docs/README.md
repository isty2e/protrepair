# ProtRepair Documentation

The root [README](../README.md) introduces the package. Start with the tutorial
for a first run, use a how-to guide for a specific task, or go directly to the
explanation and reference material when you need more detail.

## Tutorial

- [Getting started](getting-started.md): install from a repository checkout,
  run a deterministic repair, inspect the result, and write a PDB file.

## How-To Guides

- [Common repair tasks](how-to.md): select repair goals, retain ligands, control
  RDKit fallback, request histidine protonation, run FASPR, attach analyses, and
  convert structure formats.

## Explanation

- [Workflow concepts](concepts.md): canonical structure axes, observations,
  requested goals, planner-selected transformations, and partial outcomes.

## Reference

- [Public API](public-api.md): supported facades such as
  `protrepair.workflow.contracts` and `protrepair.structure`, plus workflow
  entrypoints, request and result contracts, and I/O functions.
- [Ingress normalization](ingress-policy.md): model selection, source variants,
  numeric validation, and polymer/retained residue roles.
- [Retained-ligand chemistry](retained-ligand-policy.md): templates, explicit
  chemistry evidence, RDKit fallback, and strict opt-out behavior.
- [Histidine protonation](histidine-protonation.md): the opt-in PRAS ratio
  method and its assignment contract.
- [Analyses](analysis-policy.md): Ramachandran categories and coarse
  secondary-structure labels.
- [Topology and bond egress](topology-bond-policy.md): canonical bond truth,
  provenance, repair requirements, and PDB/mmCIF projection.
- [Atomic radii](radius-policy.md): radius sources, defaults, isotope handling,
  and near-covalent detection.
- [FASPR runtime](faspr-runtime-policy.md): hydrogen ownership, retained
  components, packaged assets, and custom executable use.

## Maintainer Guide

- [Release checklist](release-checklist.md): supported matrix, verification,
  artifact contents, and tagging prerequisites.
