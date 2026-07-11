"""Terminal heavy-atom augmentation over canonical protein structures."""

from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.diagnostics.events import RepairEvent, ValidationIssue
from protrepair.diagnostics.kinds import RepairEventKind
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ChainSite
from protrepair.structure.geometry import StructureGeometry
from protrepair.structure.labels import ResidueId
from protrepair.structure.slots import ChainIndex, ResidueIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.structure.topology import AtomTopology, StructureTopology
from protrepair.transformer.completion.atom.terminal import (
    TerminalAtomPlacementTransformer,
)
from protrepair.transformer.completion.diagnostics import (
    skipped_geometry_placement_issue,
)
from protrepair.transformer.completion.heavy.policy import should_add_terminal_oxt
from protrepair.transformer.completion.shared.domain import (
    CompletionResiduePayload,
    ResidueCompletionSite,
)
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.result import TransformationResult


def augment_c_terminal_oxt(
    structure: ProteinStructure,
    component_library: ComponentLibrary | None = None,
    *,
    target_residue_ids: frozenset[ResidueId] | None = None,
) -> TransformationResult:
    """Add terminal OXT to eligible C-terminal residues in the active scope."""

    library = (
        build_default_component_library()
        if component_library is None
        else component_library
    )
    augmented_structure = structure
    repairs: list[RepairEvent] = []
    issues: list[ValidationIssue] = []
    for chain_offset, chain_site in enumerate(structure.constitution.chains):
        if not chain_site.residues:
            continue

        chain_residue_indices = (
            augmented_structure.constitution.residue_indices_for_chain_index(
                ChainIndex(chain_offset)
            )
        )
        terminal_residue_index = chain_residue_indices[-1]
        terminal_residue_site = augmented_structure.constitution.residue_site_at(
            terminal_residue_index
        )
        terminal_residue_geometry = augmented_structure.residue_geometry(
            terminal_residue_index
        )
        terminal_residue = CompletionResiduePayload(
            residue_site=terminal_residue_site,
            residue_geometry=terminal_residue_geometry,
            formal_charge_by_atom_name=(
                augmented_structure.residue_formal_charge_by_atom_name(
                    terminal_residue_index
                )
            ),
        )
        if (
            target_residue_ids is not None
            and terminal_residue.residue_id not in target_residue_ids
        ) or not should_add_terminal_oxt(
            terminal_residue.residue_site,
            component_library=library,
        ):
            continue

        template = library.get(terminal_residue.component_id)
        assert template is not None
        terminal_site = ResidueCompletionSite(
            residue_index=ResidueIndex(len(chain_residue_indices) - 1),
            template=template,
            original_payload=terminal_residue,
        )
        chain_residue_payloads: list[CompletionResiduePayload] = []
        for chain_residue_index in chain_residue_indices:
            residue_site = augmented_structure.constitution.residue_site_at(
                chain_residue_index
            )
            chain_residue_geometry = augmented_structure.residue_geometry(
                chain_residue_index
            )
            chain_residue_payloads.append(
                CompletionResiduePayload(
                    residue_site=residue_site,
                    residue_geometry=chain_residue_geometry,
                    formal_charge_by_atom_name=(
                        augmented_structure.residue_formal_charge_by_atom_name(
                            chain_residue_index
                        )
                    ),
                )
            )
        terminal_snapshot = _chain_snapshot(
            chain_id=chain_site.chain_id,
            chain_residues=tuple(chain_residue_payloads),
            source_structure=augmented_structure,
        )
        placement_outcome = TerminalAtomPlacementTransformer(
            terminal_site
        ).placement_outcome(
            ProteinTransformationContext.from_snapshot_atom_input(
                terminal_snapshot,
                terminal_site.atom_input(terminal_snapshot),
            )
        )
        augmented_terminal_snapshot = placement_outcome.snapshot
        augmented_terminal_residue = terminal_site.payload(augmented_terminal_snapshot)
        assert augmented_terminal_residue is not None
        if placement_outcome.has_skipped_atoms():
            assert placement_outcome.failure_reason is not None
            issues.append(
                skipped_geometry_placement_issue(
                    augmented_terminal_residue,
                    atom_names=placement_outcome.skipped_atom_names,
                    reason=placement_outcome.failure_reason,
                )
            )
        if not augmented_terminal_residue.has_atom_site("OXT"):
            continue

        augmented_structure = augmented_structure.with_updated_residue_facets(
            augmented_terminal_residue.residue_site,
            residue_geometry=augmented_terminal_residue.residue_geometry,
            formal_charge_by_atom_name=(
                augmented_terminal_residue.formal_charge_by_atom_name
            ),
        )
        repairs.append(
            RepairEvent.for_residue(
                kind=RepairEventKind.C_TERMINAL_OXT_ADDED,
                residue_id=augmented_terminal_residue.residue_id,
                component_id=augmented_terminal_residue.component_id,
                atom_names=("OXT",),
            )
        )

    if not repairs and not issues:
        return TransformationResult(
            structure=structure,
            repairs=(),
            issues=(),
        )

    return TransformationResult(
        structure=augmented_structure,
        repairs=tuple(repairs),
        issues=tuple(issues),
    )


def _chain_snapshot(
    *,
    chain_id: str,
    chain_residues: tuple[CompletionResiduePayload, ...],
    source_structure: ProteinStructure,
) -> ProteinStructureSnapshot:
    """Return one chain-local snapshot for terminal augmentation."""

    chain_constitution = ChainSite(
        chain_id=chain_id,
        residues=tuple(residue.residue_site for residue in chain_residues),
    )
    chain_structure_constitution = source_structure.constitution.with_chains(
        (chain_constitution,)
    ).with_ligands(())
    chain_geometry = StructureGeometry(
        constitution=chain_structure_constitution,
        atom_geometries=tuple(
            atom_geometry
            for residue in chain_residues
            for atom_geometry in residue.residue_geometry.atoms_by_name.values()
        ),
    )
    chain_topology = StructureTopology(
        constitution=chain_structure_constitution,
        atom_topologies=tuple(
            (
                None
                if formal_charge is None
                else AtomTopology(formal_charge=formal_charge)
            )
            for residue in chain_residues
            for atom_site in residue.residue_site.atom_sites
            for formal_charge in (
                dict(residue.formal_charge_by_atom_name).get(atom_site.name),
            )
        ),
        bonds=source_structure.topology.bonds_for_constitution(
            source_constitution=source_structure.constitution,
            target_constitution=chain_structure_constitution,
        ),
    )
    return ProteinStructureSnapshot.from_structure(
        ProteinStructure.from_payload(
            constitution=chain_structure_constitution,
            geometry=chain_geometry,
            topology=chain_topology,
            polymer_blueprint=(
                None
                if source_structure.polymer_blueprint is None
                else source_structure.polymer_blueprint.select_chains((chain_id,))
            ),
            provenance=source_structure.provenance,
        )
    )
