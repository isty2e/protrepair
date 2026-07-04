# ProtRepair

`protrepair` repairs protein structure files for downstream parsers and
modeling workflows. It reads PDB/mmCIF structures, completes supported missing
atoms and hydrogens, preserves supported retained ligands and cofactors, and
writes topology-aware structures back out.

Current implemented scope:

- PDB and mmCIF ingress and egress through `gemmi`
- missing heavy-atom repair for supported residues
- hydrogen placement for supported polymer and retained non-polymer chemistry
- retained ligand/cofactor preservation where component support is bundled or
  supplied
- topology-backed connection emission
- structured Ramachandran and coarse secondary-structure analyses
- canonical workflow entrypoint via `process_structure()`
- optional packaged FASPR backend for side-chain packing guidance
- optional RDKit-backed hydrogenation and local refinement through the
  `refinement` extra

Topology policy is documented in
[`docs/topology-bond-policy.md`](docs/topology-bond-policy.md). In short,
canonical topology owns bond truth; execution, readiness, and PDB/mmCIF egress
project from that topology instead of carrying separate connectivity stories.

Current deferred scope:

- generic arbitrary nonstandard chemistry beyond supported component templates

## Installation

Install from the repository root:

```bash
pip install .
```

Or install directly from GitHub:

```bash
pip install "git+https://github.com/isty2e/protrepair.git"
```

Built packages and wheels include the vendored FASPR executable and rotamer
library used by the optional side-chain packing backend. Direct source-tree
imports are not guaranteed to have compiled FASPR assets available; install the
package/wheel first, or pass an explicit FASPR `executable_path` for local
development.

Install optional RDKit-backed refinement support with:

```bash
pip install ".[refinement]"
```

## Usage

The release-facing import surface is intentionally small:

- `protrepair` exposes the workflow entrypoint and top-level request helpers.
- `protrepair.scope` exposes semantic scopes used by requested goals.
- `protrepair.state` exposes closed state axes used by requested goals.
- `protrepair.workflow.contracts` exposes ingress policies and request/result contracts.
- `protrepair.io` exposes coordinate-format read/write boundaries.

```python
from pathlib import Path

from protrepair import (
    StructureIngressOptions,
    process_structure,
    requested_process_goal,
)
from protrepair.scope import WholeStructureScope
from protrepair.state import (
    BackboneHeavyAtomCompletenessState,
    HydrogenCoverageState,
    SidechainHeavyAtomCompletenessState,
)
from protrepair.workflow.contracts import LigandPolicy

result = process_structure(
    Path("tests/fixtures/pdb/1aho.pdb"),
    ingress=StructureIngressOptions(
        ligand_policy=LigandPolicy.KEEP,
    ),
    requested_goals=(
        requested_process_goal(
            scope=WholeStructureScope(),
            value=BackboneHeavyAtomCompletenessState.COMPLETE,
        ),
        requested_process_goal(
            scope=WholeStructureScope(),
            value=SidechainHeavyAtomCompletenessState.COMPLETE,
        ),
        requested_process_goal(
            scope=WholeStructureScope(),
            value=HydrogenCoverageState.COMPLETE,
        ),
    ),
)

structure = result.structure
if result.has_errors():
    raise RuntimeError(result.issues)
```

If you want structured analyses in the result:

```python
from protrepair import AnalysisKind

analysis_result = process_structure(
    Path("tests/fixtures/pdb/1aho.pdb"),
    analyses=frozenset(
        {
            AnalysisKind.SECONDARY_STRUCTURE,
            AnalysisKind.RAMACHANDRAN,
        }
    ),
)

assert analysis_result.analyses is not None
```

If you want to write the repaired structure back out:

```python
from pathlib import Path

from protrepair.io import write_structure

write_structure(structure, Path("output.pdb"))
```

## Development

Run the permanent verification surface with:

```bash
ruff check src/protrepair tests --extend-select=I,UP --fix
basedpyright src/protrepair tests
pytest tests/unit -q
```

## License

This repository is licensed under [MIT](LICENSE).
Third-party provenance is summarized in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Provenance

ProtRepair descends from the original
[PRAS](https://github.com/osita-sunday-nnyigide/Pras_Server) project. The public
distribution name and import package are now `protrepair`.

## Changelog

Release notes are tracked in [CHANGELOG.md](CHANGELOG.md).
