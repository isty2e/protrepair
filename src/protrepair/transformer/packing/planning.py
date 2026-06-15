"""Committed side-chain packing target planning."""

from protrepair.chemistry import ComponentLibrary
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.transformer.packing.spec import PackingScope, PackingSpec

PACKING_REQUIRED_BACKBONE_ATOM_NAMES: frozenset[str] = frozenset(
    {"N", "CA", "C", "O"}
)


def _missing_heavy_atom_names(
    present_atom_names: tuple[str, ...],
    *,
    expected_heavy_atom_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Return missing heavy atom names for one residue/template pairing."""

    present_atom_name_set = {
        atom_name.strip().upper() for atom_name in present_atom_names
    }
    return tuple(
        atom_name
        for atom_name in expected_heavy_atom_names
        if atom_name not in present_atom_name_set
    )


def planned_committed_packing_spec(
    structure: ProteinStructure,
    spec: PackingSpec,
    *,
    component_library: ComponentLibrary,
) -> PackingSpec | None:
    """Return the effective committed-packing spec for one workflow run."""

    if spec.mutable_residue_ids is not None or spec.target_sequence is not None:
        return spec

    if spec.scope is PackingScope.LOCAL:
        return spec

    mutable_residue_ids = committed_packing_target_residue_ids(
        structure,
        component_library=component_library,
    )
    if not mutable_residue_ids:
        return None

    return spec.with_mutable_residue_ids(mutable_residue_ids).with_scope(
        PackingScope.LOCAL
    )


def committed_packing_target_residue_ids(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
) -> tuple[ResidueId, ...]:
    """Return residues whose sidechain-heavy state is eligible for packing."""

    target_residue_ids: list[ResidueId] = []
    for chain in structure.constitution.chains:
        for residue in chain.residues:
            if residue.is_hetero:
                continue

            template = component_library.get(residue.component_id)
            if template is None or not template.can_repair_heavy_atoms():
                continue

            present_atom_names = residue.atom_site_names()
            if not PACKING_REQUIRED_BACKBONE_ATOM_NAMES.issubset(present_atom_names):
                continue

            missing_heavy_atom_names = _missing_heavy_atom_names(
                present_atom_names,
                expected_heavy_atom_names=template.expected_heavy_atom_names(),
            )
            if not missing_heavy_atom_names:
                continue

            if any(
                atom_name in PACKING_REQUIRED_BACKBONE_ATOM_NAMES
                for atom_name in missing_heavy_atom_names
            ):
                continue

            target_residue_ids.append(residue.residue_id)

    return tuple(target_residue_ids)
