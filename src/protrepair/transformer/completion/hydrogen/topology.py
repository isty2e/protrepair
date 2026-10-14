"""Materialization of supported, already-present polymer hydrogen bonds."""

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.preparation import PolymerMicrostatePreparation
from protrepair.diagnostics.events import RepairEvent, ValidationIssue
from protrepair.diagnostics.kinds import (
    IssueSeverity,
    RepairEventKind,
    ValidationIssueKind,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.structure.topology import StructureTopology
from protrepair.transformer.result import TransformationResult


def resolve_fixed_hydrogen_topology(
    structure: ProteinStructure,
    component_library: ComponentLibrary,
    *,
    target_residue_ids: frozenset[ResidueId] | None = None,
) -> TransformationResult:
    """Attach present fixed H without changing atom identity or geometry.

    Parameters
    ----------
    structure : ProteinStructure
        Current canonical structure and preserved source evidence.
    component_library : ComponentLibrary
        Component definitions used to interpret names and fixed H inventory.
    target_residue_ids : frozenset[ResidueId] or None
        Polymer residues to update; None selects all polymer residues.

    Returns
    -------
    TransformationResult
        Topology-only edits and residue-local diagnostics. Contradictory residues
        remain unchanged. Coupled microstate sites retain their separate atomic
        realization path; this operation neither selects nor changes protonation.
    """
    prepared: PolymerMicrostatePreparation | None = None
    updates = []
    repairs = []
    issues = []
    for chain in structure.constitution.chains:
        for residue in chain.residues:
            if (
                target_residue_ids is not None
                and residue.residue_id not in target_residue_ids
            ):
                continue
            if not any(atom.is_hydrogen() for atom in residue.atom_sites):
                continue
            if prepared is None:
                prepared = PolymerMicrostatePreparation(
                    PolymerMicrostateContext(structure), component_library
                )
            try:
                bonds = prepared.fixed_hydrogen_topology_updates(residue.residue_id)
            except ValueError as error:
                issues.append(
                    ValidationIssue.for_residue(
                        kind=ValidationIssueKind.UNSUPPORTED_HYDROGENATION,
                        severity=IssueSeverity.WARNING,
                        message=f"Preserved H topology was not resolved: {error}",
                        residue_id=residue.residue_id,
                        component_id=residue.component_id,
                    )
                )
                continue
            if not bonds:
                continue
            updates.extend(bonds)
            repairs.append(
                RepairEvent.for_residue(
                    kind=RepairEventKind.HYDROGEN_TOPOLOGY_RESOLVED,
                    residue_id=residue.residue_id,
                    component_id=residue.component_id,
                    atom_names=tuple(
                        structure.constitution.atom_ref_at(index).atom_name
                        for bond in bonds
                        for index in bond.endpoint_pair()
                        if structure.constitution.atom_site_at(index).is_hydrogen()
                    ),
                )
            )
    if not updates:
        return TransformationResult(structure, repairs=(), issues=tuple(issues))

    bonds_by_pair = {bond.endpoint_pair(): bond for bond in structure.topology.bonds}
    bonds_by_pair.update((bond.endpoint_pair(), bond) for bond in updates)
    topology = StructureTopology(
        constitution=structure.constitution,
        atom_topologies=structure.topology.atom_topologies,
        bonds=tuple(bonds_by_pair.values()),
    )
    return TransformationResult(
        structure=ProteinStructure.from_payload(
            constitution=structure.constitution,
            geometry=structure.geometry,
            topology=topology,
            polymer_blueprint=structure.polymer_blueprint,
            provenance=structure.provenance,
        ),
        repairs=tuple(repairs),
        issues=tuple(issues),
    )
