# Workflow Concepts

ProtRepair plans from the state you want rather than running a fixed list of
repair commands. The workflow observes the structure, selects a transformation,
and evaluates the new result before deciding what to do next.

## From Source To Canonical Structure

PDB and mmCIF files can contain multiple models, alternate locations,
microheterogeneous residues, format-specific connectivity, and ambiguous
polymer labels. ProtRepair resolves those source concerns once at ingress:

```text
source file -> normalization -> canonical structure -> workflow -> egress
```

Downstream code does not reinterpret raw records. The canonical structure keeps
four distinct parts of the model together:

- **Constitution** identifies chains, residues, atoms, and retained components.
- **Geometry** stores atom coordinates and residue-local geometric facts.
- **Topology** stores bonds and their physical relationship and provenance.
- **Provenance** records where the structure came from and how it was selected.

Keeping these axes separate prevents coordinates, bond inference, and source
records from becoming competing descriptions of the same structure.

## Observation Is Not Execution

State values describe what is currently true. Examples include backbone
heavy-atom completeness, hydrogen coverage, clash presence, and topology
availability. Observation derives those values from the canonical structure;
it does not modify the structure.

A requested goal pairs one state value with a scope. The same whole-structure
scope can carry separate goals for backbone atoms, side-chain atoms, and
hydrogens because those are independent state axes.

## Goals And Transform Requests

Goals express desired end states. Transform requests name method choices that
cannot be inferred from a state alone. Histidine protonation, an explicit FASPR
packing request, and a particular refinement specification belong to the second
category.

This distinction keeps the workflow from treating implementation choices as
facts about the protein. A caller can request complete hydrogens without naming
the transformer that should achieve them, while still explicitly selecting a
protonation or packing method when that choice carries scientific meaning.

## Planning And Transformation

The planner compares requested goals with observed state and available
chemistry. It selects transformers that are admissible for the current
structure and can reconsider the plan after each result.

Transformers own structure changes. A transformer that adds or removes atoms
must update topology with the same change or report incomplete topology. A
continuous relaxation backend may move coordinates, but it does not silently
create a separate bond graph. FASPR may alter side-chain heavy atoms, but
hydrogen completion remains a separate transformer responsibility.

Partial standard-residue completion keeps connected atoms already present in
the source and places only missing atoms. Available side-chain anchors guide
the missing branches; otherwise the standard recipe supplies a fallback.
Ring closure uses available component dictionary targets; partial HIS rings have
no state-independent closure target. If the observed anchors cannot accommodate
a closure, completion leaves those atoms absent and reports a placement warning
rather than moving the anchors. The workflow may keep the unchanged source
instead; a complete-atom goal then remains unmet, with no automatic packing rescue.
Explicit orphan-fragment rebuilding and later packing or refinement are separate
operations that may change existing coordinates.

The workflow may evaluate more than one candidate result. Candidate preference
considers requested-goal satisfaction together with hard errors, parser
compatibility, clashes, stereochemistry, geometry, issue burden, and search
depth. The returned structure is the preferred terminal candidate, not merely
the last action executed.

## Local Refinement Assessment

Local refinement measures contacts involving the selected residues, including
contacts with retained ligands and cofactors. Heavy atoms and H/D/T participate;
H-H pairs keep their usual exclusion. This default does not depend on the force
field. Low-level callers supplying a contact basis keep its explicit policy.

The no-CONECT parser probe runs full RDKit sanitation. A successful parse can
still attach H to a different heavy atom, so the probe separately checks
attachments against declared canonical H/D/T parents. It does not guess missing
parents or rewrite the structure's topology.
Observed noncanonical attachments also contribute to the parser deficits used
by correction planning, even when sanitation succeeds.

Local candidate comparison puts confirmed parser failure first, then observed
noncanonical parser bonds, steric clashes and overlap, and soft contact proxies.
Unknown parser status is not evidence of success or failure. Equal quality is
not regression, and a local improvement need not fix a remote parser failure.
Existing checks for new stereo violations and severe restraint-backed bond
distortion still apply. This is endpoint assessment, not selection of additional
intermediate optimization frames.

## Complete, Partial, And Unsupported Outcomes

A processing run can return a useful structure without satisfying every goal.
Each requested goal receives one outcome:

- already satisfied before execution
- satisfied by execution
- unmet after execution
- blocked by current evidence or prerequisites
- unsupported by available capabilities

Warnings and partial repair do not automatically invalidate the whole
structure. `ProcessResult` keeps the selected structure, repair events, issues,
and requested-goal report together so callers can decide what they will accept.
`has_errors()` reports error-severity issues;
`requested_goals_fully_satisfied()` reports whether the requested outcomes were
achieved.

## Where To Read Next

- [Ingress policy](ingress-policy.md) defines source normalization.
- [Topology bond policy](topology-bond-policy.md) defines canonical bond truth
  and egress projection.
- [Retained-ligand policy](retained-ligand-policy.md) defines chemistry evidence
  and fallback behavior.
- [Public API reference](public-api.md) lists the supported request and result
  boundaries.
