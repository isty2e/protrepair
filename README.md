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
- RDKit-backed hydrogenation and local refinement

Topology policy is documented in
[`docs/topology-bond-policy.md`](docs/topology-bond-policy.md). In short,
canonical topology owns bond truth; execution, readiness, and PDB/mmCIF egress
project from that topology instead of carrying separate connectivity stories.

Ingress normalizes each source into one canonical structure realization.
For multi-model PDB or mmCIF files, ProtRepair currently reads the first model
only; choose a different model upstream before calling `process_structure()` if
you need another realization.

Numeric atom scalar validation at ingress is strict. Occupancy must be finite
and within the closed interval `[0.0, 1.0]`; B factors must be finite and
non-negative. ProtRepair rejects boundary-adjacent invalid values such as
`1.0000001` occupancy or `-0.0000001` B factor instead of clamping them, because
silent normalization would hide source data quality problems.

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

Prefer a built wheel when you need the optional FASPR side-chain packing
backend. Built packages and wheels include the vendored FASPR executable and
rotamer library.

Source installs, including direct GitHub installs, build the vendored FASPR
executable through `scikit-build-core` and CMake. They require CMake 3.18 or
newer and a working C++ compiler toolchain. Direct source-tree imports are not
guaranteed to have compiled FASPR assets available; install the package/wheel
first. For advanced transformer-layer development with a separately built
FASPR binary, construct the FASPR backend with an explicit FASPR
`executable_path` whose directory also contains `dun2010bbdep.bin`.

## Usage

The release-facing import surface is intentionally small:

- `protrepair` exposes the workflow entrypoint and top-level request helpers.
- `protrepair.structure` exposes canonical structure and label types.
- `protrepair.scope` exposes semantic scopes used by requested goals.
- `protrepair.state` exposes closed state axes used by requested goals.
- `protrepair.workflow.contracts` exposes ingress policies and request/result contracts.
- `protrepair.analysis` exposes structured analysis request and result types.
- `protrepair.io` exposes coordinate-format read/write boundaries.

```python
from pathlib import Path

from protrepair import process_structure
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
)

structure = result.structure
if result.has_errors():
    raise RuntimeError(result.issues)
```

`StructureIngressOptions(ligand_policy=LigandPolicy.REJECT)` rejects selected
ligand-bearing inputs during ingress instead of silently dropping ligands.

Retained non-polymer hydrogen completion uses supported bundled templates or
explicit chemistry overrides when available. For otherwise unknown retained
ligands, the default RDKit coordinate/proximity fallback is permissive and
emits a `RETAINED_NON_POLYMER_FALLBACK_USED` warning issue when used. To require
templates or explicit chemistry evidence, keep retained ligands, request hydrogen
completion, and disable that fallback:

The fallback is still conservative: existing retained-ligand hydrogens are reused
only when local H-anchor geometry and any source topology agree, and unsupported
hetero multiple-bond or stereochemistry-changing fallback inferences leave the
retained ligand unchanged with a warning instead of guessing chemistry.

Explicit retained-ligand chemistry overrides are validated at ingress. Invalid
SMILES/evidence mappings or overrides that do not match the kept heavy-atom set
raise `ValueError` before workflow execution rather than falling back silently.

```python
from protrepair.workflow.contracts import WorkflowTransformRequests

strict_result = process_structure(
    Path("tests/fixtures/pdb/1aho.pdb"),
    ingress=StructureIngressOptions(
        ligand_policy=LigandPolicy.KEEP,
    ),
    requested_goals=(
        requested_process_goal(
            scope=WholeStructureScope(),
            value=HydrogenCoverageState.COMPLETE,
        ),
    ),
    transform_requests=WorkflowTransformRequests(
        allow_retained_non_polymer_rdkit_fallback=False,
    ),
)
```

When strict policy blocks an otherwise required RDKit fallback, the result emits
a `RETAINED_NON_POLYMER_FALLBACK_BLOCKED` warning issue and leaves the retained
ligand unchanged.

Histidine delta protonation is available as an explicit workflow request, not as
an automatic pKa or hydrogen-bond-network inference. The legacy PRAS behavior
assumes roughly pH 7 and protonates the first `floor(total_chain_HIS * 0.2)`
histidines in chain order. That ratio is exposed as
`PrasRatioHistidineProtonationRequest` so callers can opt in, choose a different
ratio in `[0.0, 1.0]`, or keep the default disabled behavior. The older
`protonate_histidines=True` flag remains a shorthand for the default PRAS-ratio
request at the workflow boundary.

```python
from protrepair.workflow.contracts import (
    PrasRatioHistidineProtonationRequest,
    WorkflowTransformRequests,
)

his_result = process_structure(
    Path("tests/fixtures/pdb/1aho.pdb"),
    requested_goals=(
        requested_process_goal(
            scope=WholeStructureScope(),
            value=HydrogenCoverageState.COMPLETE,
        ),
    ),
    transform_requests=WorkflowTransformRequests(
        histidine_protonation=PrasRatioHistidineProtonationRequest(ratio=0.2),
    ),
)
```

The request resolves to explicit histidine assignments before hydrogen
directives are built. Future explicit-residue, tautomer, pKa-backed, or
environment-aware methods should produce the same assignment contract instead of
changing the PRAS-ratio request into a general protonation policy.

If you want structured analyses in the result:

```python
from protrepair.analysis import AnalysisKind

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

Analysis categories are intentionally coarse. Ramachandran points report a
closed `RamachandranCategory` value: `helix` for phi in `[-160, -20]` and psi
in `[-90, 45]`, `beta` for phi in `[-180, -40]` with psi at least `90` or at
most `-120`, `left_handed` for phi in `[20, 120]` and psi in `[-20, 120]`,
and `other` outside those broad regions when both torsions are available.
Coarse secondary-structure output projects those categories to `H` for helix,
`E` for beta, and `C` for everything else, including left-handed, other,
missing-torsion, and gap-disconnected residues. This analysis is not a DSSP
replacement: it does not infer hydrogen-bond patterns, turns, bends, strand
registration, or a separate PPII assignment.

If you want to write the repaired structure back out:

```python
from pathlib import Path

from protrepair.io import write_structure

write_structure(structure, Path("output.pdb"))
```

## Development

Run the permanent verification surface with:

```bash
ruff check src/protrepair tests --extend-select=I,UP
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
