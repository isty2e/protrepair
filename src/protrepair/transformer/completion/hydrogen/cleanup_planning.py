"""Residual rotatable-hydrogen cleanup planning helpers."""

from dataclasses import dataclass

from protrepair.chemistry import ComponentLibrary
from protrepair.diagnostics import (
    ClashPolicy,
    ClashReport,
    StericClash,
    detect_clashes_involving_residues,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import ResidueId
from protrepair.transformer.completion.hydrogen.rotatable import (
    RotatableHydrogenPlacementSpec,
    rotatable_hydrogen_placement_spec,
)


@dataclass(frozen=True, slots=True)
class RotatableHydrogenCleanupTarget:
    """One placed rotatable hydrogen eligible for cleanup search."""

    residue_id: ResidueId
    hydrogen_atom_name: str
    placement_spec: RotatableHydrogenPlacementSpec


def structure_residue_site_or_ligand(
    structure: ProteinStructure, residue_id: ResidueId
) -> ResidueSite | None:
    """Return one constitution residue or ligand site by identifier if present."""

    return structure.constitution.residue_or_ligand(residue_id)


def cleanup_targets_from_report(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
    report: ClashReport,
) -> tuple[RotatableHydrogenCleanupTarget, ...]:
    """Return unique rotatable-hydrogen cleanup targets from one clash report."""

    targets_by_key: dict[tuple[str, int, str, str], RotatableHydrogenCleanupTarget] = {}
    for clash in report.clashes:
        for residue_id, atom_name in (
            (clash.left_residue_id, clash.left_atom_name),
            (clash.right_residue_id, clash.right_atom_name),
        ):
            residue_site = structure_residue_site_or_ligand(structure, residue_id)
            if residue_site is None or not residue_site.has_atom_site(atom_name):
                continue

            atom_site = residue_site.atom_site(atom_name)
            if not atom_site.is_hydrogen():
                continue

            template = component_library.get(residue_site.component_id)
            if template is None:
                continue

            placement_spec = rotatable_hydrogen_placement_spec(
                template.hydrogen_semantics
            )
            if (
                placement_spec is None
                or atom_site.name != placement_spec.hydrogen_atom_name
            ):
                continue

            key = (
                residue_id.chain_id,
                residue_id.seq_num,
                residue_id.insertion_code or "",
                atom_site.name,
            )
            targets_by_key.setdefault(
                key,
                RotatableHydrogenCleanupTarget(
                    residue_id=residue_id,
                    hydrogen_atom_name=atom_site.name,
                    placement_spec=placement_spec,
                ),
            )

    return tuple(targets_by_key.values())


def targeted_hydrogen_clash_score(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
    residue_id: ResidueId,
    hydrogen_atom_name: str,
    policy: ClashPolicy,
) -> tuple[int, float, float]:
    """Return a clash score tuple for one specific hydrogen atom."""

    report = detect_clashes_involving_residues(
        structure,
        residue_ids=frozenset((residue_id,)),
        component_library=component_library,
        policy=policy,
    )
    relevant_overlaps = [
        clash.overlap_angstrom
        for clash in report.clashes
        if clash_involves_hydrogen(
            clash,
            residue_id=residue_id,
            hydrogen_atom_name=hydrogen_atom_name,
        )
    ]
    if not relevant_overlaps:
        return (0, 0.0, 0.0)

    return (
        len(relevant_overlaps),
        sum(relevant_overlaps),
        max(relevant_overlaps),
    )


def unresolved_rotatable_hydrogen_report(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
    report: ClashReport,
) -> ClashReport:
    """Return one report containing unresolved rotatable-hydrogen clashes only."""

    unresolved_clashes = tuple(
        clash
        for clash in report.clashes
        if clash_involves_residual_rotatable_hydrogen(
            clash,
            structure=structure,
            component_library=component_library,
        )
    )
    return ClashReport(clashes=unresolved_clashes)


def clash_involves_residual_rotatable_hydrogen(
    clash: StericClash,
    *,
    structure: ProteinStructure,
    component_library: ComponentLibrary,
) -> bool:
    """Return whether one clash involves a supported rotatable donor hydrogen."""

    return any(
        hydrogen_is_supported_cleanup_target(
            structure,
            component_library=component_library,
            residue_id=residue_id,
            atom_name=atom_name,
        )
        for residue_id, atom_name in (
            (clash.left_residue_id, clash.left_atom_name),
            (clash.right_residue_id, clash.right_atom_name),
        )
    )


def hydrogen_is_supported_cleanup_target(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
    residue_id: ResidueId,
    atom_name: str,
) -> bool:
    """Return whether one residue-local atom is a supported rotatable hydrogen."""

    residue_site = structure_residue_site_or_ligand(structure, residue_id)
    if residue_site is None or not residue_site.has_atom_site(atom_name):
        return False

    atom_site = residue_site.atom_site(atom_name)
    if not atom_site.is_hydrogen():
        return False

    template = component_library.get(residue_site.component_id)
    if template is None:
        return False

    placement_spec = rotatable_hydrogen_placement_spec(template.hydrogen_semantics)
    return (
        placement_spec is not None
        and atom_site.name == placement_spec.hydrogen_atom_name
    )


def clash_involves_hydrogen(
    clash: StericClash,
    *,
    residue_id: ResidueId,
    hydrogen_atom_name: str,
) -> bool:
    """Return whether one clash involves a specific residue-local hydrogen."""

    return (
        clash.left_residue_id == residue_id
        and clash.left_atom_name == hydrogen_atom_name
    ) or (
        clash.right_residue_id == residue_id
        and clash.right_atom_name == hydrogen_atom_name
    )
