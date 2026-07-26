# ProtRepair

[![CI](https://github.com/isty2e/protrepair/actions/workflows/ci.yml/badge.svg)](https://github.com/isty2e/protrepair/actions/workflows/ci.yml)

ProtRepair prepares protein structure files for downstream modeling. It reads
PDB and mmCIF files, repairs supported missing heavy atoms, adds hydrogens,
preserves retained ligands and cofactors, and writes the repaired structure
with its connectivity.

## What It Does

- Repairs missing backbone and side-chain heavy atoms for supported residues.
- Adds polymer and retained-ligand hydrogens using component chemistry or
  RDKit-backed inference.
- Preserves supported ligands, cofactors, metals, ions, and source
  connectivity.
- Uses topology-aware local refinement and optional FASPR side-chain packing.
- Reports structured issues and optional Ramachandran or coarse
  secondary-structure analyses.

ProtRepair repairs the atoms present in a structure model. It does not
currently build missing residue spans or arbitrary nonstandard heavy-atom
chemistry.

## Installation

Install from the repository root:

```bash
pip install .
```

Or install directly from GitHub:

```bash
pip install "git+https://github.com/isty2e/protrepair.git"
```

Built wheels include the FASPR executable and rotamer library. Building from
source requires CMake 3.18 or newer and a C++ compiler; see the
[FASPR runtime policy](docs/faspr-runtime-policy.md) for source-tree and custom
executable setups.

## Quick Start

The example below keeps retained ligands, requests complete heavy atoms and
hydrogens, runs both available analyses, and writes a repaired PDB file.

```python
from pathlib import Path

from protrepair import process_structure
from protrepair.analysis import AnalysisKind
from protrepair.io import write_structure
from protrepair.scope import WholeStructureScope
from protrepair.state import (
    BackboneHeavyAtomCompletenessState,
    HydrogenCoverageState,
    SidechainHeavyAtomCompletenessState,
)
from protrepair.workflow.contracts import (
    LigandPolicy,
    StructureIngressOptions,
    requested_process_goal,
)

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
    analyses=frozenset(
        {
            AnalysisKind.SECONDARY_STRUCTURE,
            AnalysisKind.RAMACHANDRAN,
        }
    ),
)

if result.has_errors():
    raise RuntimeError(result.issues)

write_structure(result.structure, Path("output.pdb"))
```

`result.structure` is the canonical repaired structure. `result.issues`
contains structured warnings and errors, and `result.analyses` contains the
requested analysis results.

## Common Choices

Retained ligands are kept in the example above. Use
`LigandPolicy.REJECT` when ligand-bearing input should fail at ingress instead.
Known components use bundled chemistry; unknown retained ligands can use the
default RDKit fallback. You can require templates or explicit chemistry
evidence by setting
`WorkflowTransformRequests(allow_retained_non_polymer_rdkit_fallback=False)`.
See the [retained-ligand policy](docs/retained-ligand-policy.md) for fallback,
override, and diagnostic behavior.

Histidine protonation is disabled by default. Opt into deterministic PRAS-style
delta protonation with `PrasRatioHistidineProtonationRequest`; see
[histidine protonation](docs/histidine-protonation.md) for its ratio semantics
and limitations.

ProtRepair reads the first model from multi-model inputs and resolves source
variants into one canonical realization. Invalid coordinates, occupancies, and
B factors are rejected rather than silently repaired. The exact ingress rules
are documented in the [ingress policy](docs/ingress-policy.md).

## Further Reading

- [Ingress normalization](docs/ingress-policy.md)
- [Retained-ligand chemistry](docs/retained-ligand-policy.md)
- [Histidine protonation](docs/histidine-protonation.md)
- [Analysis categories](docs/analysis-policy.md)
- [Topology and bond egress](docs/topology-bond-policy.md)
- [FASPR runtime and hydrogen ownership](docs/faspr-runtime-policy.md)
- [Atomic radii](docs/radius-policy.md)
- [Release checklist](docs/release-checklist.md)

## Development

Run the permanent verification surface with:

```bash
ruff check src/protrepair tests --extend-select=I,UP
basedpyright src/protrepair tests
pytest tests/unit -q
```

## License And Provenance

This repository is licensed under [MIT](LICENSE). Third-party provenance is
listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

ProtRepair descends from the original
[PRAS](https://github.com/osita-sunday-nnyigide/Pras_Server) project. Donor
hydrogen geometry follows the protein parameters in
[AMBER ff14SB](https://doi.org/10.1021/acs.jctc.5b00255), while donor-hydrogen
ranking retains PRAS's AMBER-derived nonbonded and torsion-search lineage.
RDKit readability is an interoperability check, not the scientific ranking
criterion.

Release notes are tracked in [CHANGELOG.md](CHANGELOG.md).
