# Common Repair Tasks

These recipes cover common `process_structure()` options. Unless a recipe says
otherwise, call it with a `pathlib.Path` source and the request fragment shown.

## Retain Or Reject Ligands

Ligands are dropped by default. Keep retained non-polymers with an ingress
option:

```python
from protrepair.workflow.contracts import LigandPolicy, StructureIngressOptions

ingress = StructureIngressOptions(ligand_policy=LigandPolicy.KEEP)
```

Use `LigandPolicy.REJECT` when any retained non-polymer should make ingress
fail. See the [retained-ligand policy](retained-ligand-policy.md) for chemistry
resolution and scope boundaries.

## Request Only The State Changes You Need

Each goal combines a semantic scope with one target state. This example asks
for hydrogen completion without requesting heavy-atom completion:

```python
from protrepair.scope import WholeStructureScope
from protrepair.state import HydrogenCoverageState
from protrepair.workflow.contracts import requested_process_goal

requested_goals = (
    requested_process_goal(
        scope=WholeStructureScope(),
        value=HydrogenCoverageState.COMPLETE,
    ),
)
```

Hydrogen completion can still be blocked when required heavy atoms or chemistry
are unavailable. Inspect `result.requested_goal_report` rather than assuming
that a requested action succeeded.

## Disable Retained-Ligand Chemistry Fallback

RDKit fallback is enabled by default for retained components that lack explicit
or bundled chemistry. Disable it when ProtRepair must not infer chemistry for
an unrecognized retained component:

```python
from protrepair.workflow.contracts import WorkflowTransformRequests

transform_requests = WorkflowTransformRequests(
    allow_retained_non_polymer_rdkit_fallback=False,
)
```

If retained-ligand hydrogen completion needs that fallback, the goal remains
blocked and the result contains a structured issue explaining why.

## Request PRAS-Ratio Histidine Protonation

Histidine protonation is opt-in. Supply an explicit request through the
transform request boundary:

```python
from protrepair.workflow.contracts import (
    PrasRatioHistidineProtonationRequest,
    WorkflowTransformRequests,
)

transform_requests = WorkflowTransformRequests(
    histidine_protonation=PrasRatioHistidineProtonationRequest(ratio=0.2),
)
```

This is a deterministic chain-order ratio method, not a pKa or environment
model. Read [histidine protonation](histidine-protonation.md) before changing
the ratio.

## Run FASPR Side-Chain Packing

Use committed packing when the returned structure should include the FASPR
side-chain result:

```python
from protrepair.transformer.packing import PackingSpec
from protrepair.workflow.contracts import WorkflowTransformRequests

transform_requests = WorkflowTransformRequests(
    committed_sidechain_packing=PackingSpec(backend_name="faspr"),
)
```

Installed packages include the FASPR executable and rotamer library. Source
build and custom executable requirements are documented in the
[FASPR runtime policy](faspr-runtime-policy.md).

## Attach And Read Analyses

Request analyses independently of repair goals:

```python
from pathlib import Path

from protrepair import process_structure
from protrepair.analysis import AnalysisKind

result = process_structure(
    Path("input.pdb"),
    analyses=(
        AnalysisKind.RAMACHANDRAN,
        AnalysisKind.SECONDARY_STRUCTURE,
    ),
)

if result.analyses is not None:
    ramachandran = result.analyses.ramachandran
    secondary_structure = result.analyses.secondary_structure
```

These analyses describe the final canonical structure and do not direct repair
planning. Their category definitions are in the [analysis policy](analysis-policy.md).

## Convert Between PDB And mmCIF

Use the I/O boundary directly when no repair is needed:

```python
from pathlib import Path

from protrepair.io import read_structure, write_structure

structure = read_structure(Path("input.cif"))
write_structure(structure, Path("output.pdb"))
```

The writer carries coordinates, canonical residue roles, and topology into the
selected format. See the [topology bond policy](topology-bond-policy.md) for
format-specific connectivity rules.
