# Getting Started

This tutorial repairs the 1AHO structure included in the repository. You will
request complete backbone atoms, side-chain atoms, and hydrogens, inspect the
result, and write a new PDB file.

## Prepare The Environment

Clone the repository and install ProtRepair in a virtual environment:

```bash
git clone https://github.com/isty2e/protrepair.git
cd protrepair
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Copy the tutorial input so the output remains separate from the test fixture:

```bash
cp tests/fixtures/pdb/1aho.pdb input.pdb
```

## Run A Repair

Save the following as `repair_example.py`:

```python
from pathlib import Path

from protrepair import process_structure
from protrepair.io import write_structure
from protrepair.scope import WholeStructureScope
from protrepair.state import (
    BackboneHeavyAtomCompletenessState,
    HydrogenCoverageState,
    SidechainHeavyAtomCompletenessState,
)
from protrepair.workflow.contracts import requested_process_goal

input_path = Path("input.pdb")
output_path = Path("repaired.pdb")
whole_structure = WholeStructureScope()

result = process_structure(
    input_path,
    requested_goals=(
        requested_process_goal(
            scope=whole_structure,
            value=BackboneHeavyAtomCompletenessState.COMPLETE,
        ),
        requested_process_goal(
            scope=whole_structure,
            value=SidechainHeavyAtomCompletenessState.COMPLETE,
        ),
        requested_process_goal(
            scope=whole_structure,
            value=HydrogenCoverageState.COMPLETE,
        ),
    ),
)

if result.has_errors():
    raise RuntimeError(result.issues)
if result.requested_goals_fully_satisfied() is not True:
    raise RuntimeError(result.requested_goal_report)

write_structure(result.structure, output_path)
print(f"Wrote {output_path}")
print(f"Recorded {result.repair_count()} repair events")
```

Run it from the repository root:

```bash
python repair_example.py
```

The command creates `repaired.pdb`. Warnings do not stop the workflow, but this
example raises if ProtRepair reports an error or cannot satisfy every requested
goal.

## Read The Result

`process_structure()` returns a `ProcessResult`:

- `result.structure` is the final canonical structure.
- `result.repairs` records transformations applied to the structure.
- `result.issues` records warnings and errors with structured kinds and scopes.
- `result.requested_goal_report` records whether each requested goal was already
  satisfied, satisfied by execution, unmet, blocked, or unsupported.

The output includes connectivity derived from the canonical topology, not just
the final coordinates. Re-reading `repaired.pdb` therefore does not require the
writer to infer those bonds again.

## Continue

- Use the [how-to recipes](how-to.md) to retain ligands, request analyses, or
  enable side-chain packing.
- Read [workflow concepts](concepts.md) before configuring more advanced goals
  and transformations.
- Use the [public API reference](public-api.md) for exact entrypoint and result
  contracts.
