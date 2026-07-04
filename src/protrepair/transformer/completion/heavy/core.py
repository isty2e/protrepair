"""Primitive heavy-atom completion over direct atom transforms."""

from dataclasses import dataclass

from protrepair.chemistry import (
    ComponentLibrary,
    IdealGeometryHeavyAtomSemantics,
    ResidueTemplate,
    build_default_component_library,
)
from protrepair.chemistry.component.topology import (
    template_resolved_topology_bonds_for_new_atoms,
)
from protrepair.diagnostics.component_support import (
    diagnose_component_support,
    missing_component_definition_issue,
    unsupported_template_repair_issue,
)
from protrepair.diagnostics.events import RepairEvent, ValidationIssue
from protrepair.diagnostics.kinds import (
    IssueSeverity,
    RepairEventKind,
    ValidationIssueKind,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ChainSite
from protrepair.structure.geometry import StructureGeometry
from protrepair.structure.labels import ResidueId
from protrepair.structure.slots import ChainIndex, ResidueIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.structure.topology import AtomTopology, StructureTopology
from protrepair.transformer.completion.atom.graft import ReferenceAtomGraftTransformer
from protrepair.transformer.completion.atom.internal_coordinates import (
    InternalCoordinatePlacementTransformer,
)
from protrepair.transformer.completion.atom.rigid_frame import (
    RigidFramePlacementTransformer,
)
from protrepair.transformer.completion.atom.terminal import (
    TerminalAtomPlacementTransformer,
)
from protrepair.transformer.completion.fragment_matching import (
    ResidueFragmentMatch,
    match_residue_fragment,
)
from protrepair.transformer.completion.heavy.orphan_fragments import (
    apply_orphan_fragment_policy,
    prepare_residue_for_fragment_repair,
)
from protrepair.transformer.completion.heavy.policy import (
    allows_refinement_gated_component_repair,
    should_add_terminal_oxt,
)
from protrepair.transformer.completion.heavy.reconstruction import (
    build_component_reconstruction_plan,
)
from protrepair.transformer.completion.normalization import (
    normalize_residue_inputs,
    normalized_reference_residue_by_id,
)
from protrepair.transformer.completion.policies import OrphanFragmentPolicy
from protrepair.transformer.completion.shared.domain import (
    CompletionResiduePayload,
    ResidueBackboneNeighborhood,
    ResidueCompletionSite,
    resolve_completion_residue_payload,
)
from protrepair.transformer.completion.targets import (
    BACKBONE_ATOM_NAMES,
    missing_atoms_for_completion,
)
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.refinement.directive import RepairLocalRefinementDirective
from protrepair.transformer.result import TransformationResult


@dataclass(frozen=True, slots=True)
class _ResidueRepairStageResult:
    """One residue-local heavy-completion stage result."""

    residue: CompletionResiduePayload
    added_atom_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ChainHeavyRepairStageResult:
    """One chain-local heavy-completion stage result."""

    chain_id: str
    residues: tuple[CompletionResiduePayload, ...]
    repairs: tuple[RepairEvent, ...]
    issues: tuple[ValidationIssue, ...]


def repair_heavy_atoms_core(
    structure: ProteinStructure,
    component_library: ComponentLibrary | None = None,
    reference_structure: ProteinStructure | None = None,
    *,
    augment_c_terminal_oxt: bool = True,
    target_residue_ids: frozenset[ResidueId] | None = None,
    orphan_fragment_policy: OrphanFragmentPolicy = OrphanFragmentPolicy.REBUILD,
    local_refinement: RepairLocalRefinementDirective | None = None,
) -> TransformationResult:
    """Return the repaired heavy-atom structure before optional local refinement."""

    library = (
        build_default_component_library()
        if component_library is None
        else component_library
    )
    stripped_structure = (
        structure.without_hydrogens()
        if target_residue_ids is None
        else structure.without_hydrogens_in_residues(target_residue_ids)
    )
    stripped_reference_structure = (
        None
        if reference_structure is None
        else (
            reference_structure.without_hydrogens()
            if target_residue_ids is None
            else reference_structure.without_hydrogens_in_residues(target_residue_ids)
        )
    )
    repaired_chain_results: list[_ChainHeavyRepairStageResult] = []
    repairs: list[RepairEvent] = []
    issues: list[ValidationIssue] = []

    for chain_offset, chain_site in enumerate(stripped_structure.constitution.chains):
        chain_result = _repair_heavy_atom_chain(
            chain_id=chain_site.chain_id,
            chain_residue_indices=(
                stripped_structure.constitution.residue_indices_for_chain_index(
                    ChainIndex(chain_offset)
                )
            ),
            source_structure=stripped_structure,
            reference_structure=stripped_reference_structure,
            component_library=library,
            augment_c_terminal_oxt=augment_c_terminal_oxt,
            target_residue_ids=target_residue_ids,
            orphan_fragment_policy=orphan_fragment_policy,
            local_refinement=local_refinement,
        )
        repaired_chain_results.append(chain_result)
        repairs.extend(chain_result.repairs)
        issues.extend(chain_result.issues)

    repaired_structure = _structure_from_repaired_chain_results(
        source_structure=stripped_structure,
        chain_results=tuple(repaired_chain_results),
        component_library=library,
    )
    return TransformationResult(
        structure=repaired_structure,
        repairs=tuple(repairs),
        issues=tuple(issues),
    )


def _is_targeted_residue(
    residue_id: ResidueId,
    *,
    target_residue_ids: frozenset[ResidueId] | None,
) -> bool:
    """Return whether one residue is in the active workflow-stage scope."""

    return target_residue_ids is None or residue_id in target_residue_ids


def _chain_snapshot(
    *,
    chain_id: str,
    chain_residues: tuple[CompletionResiduePayload, ...],
    source_structure: ProteinStructure,
) -> ProteinStructureSnapshot:
    """Return one chain-local snapshot for residue-local completion transforms."""

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


def _structure_from_repaired_chain_results(
    *,
    source_structure: ProteinStructure,
    chain_results: tuple[_ChainHeavyRepairStageResult, ...],
    component_library: ComponentLibrary,
) -> ProteinStructure:
    """Return one structure rebuilt from repaired chain residue payload."""

    repaired_chain_result_by_id = {
        chain_result.chain_id: chain_result for chain_result in chain_results
    }
    repaired_chain_sites: list[ChainSite] = []
    updated_residue_entries_by_index: list[
        tuple[CompletionResiduePayload, tuple[tuple[str, int | None], ...]]
    ] = []

    for source_chain_site in source_structure.constitution.chains:
        chain_result = repaired_chain_result_by_id[source_chain_site.chain_id]
        repaired_chain_sites.append(
            ChainSite(
                chain_id=chain_result.chain_id,
                residues=tuple(
                    residue.residue_site for residue in chain_result.residues
                ),
            )
        )
        updated_residue_entries_by_index.extend(
            (residue, residue.formal_charge_by_atom_name)
            for residue in chain_result.residues
        )

    updated_constitution = source_structure.constitution.with_chains(
        repaired_chain_sites
    )
    updated_residue_entries_by_index.extend(
        (
            CompletionResiduePayload(
                residue_site=ligand_site,
                residue_geometry=source_structure.residue_geometry(
                    source_structure.constitution.residue_index(ligand_site.residue_id)
                ),
                formal_charge_by_atom_name=(
                    source_structure.residue_formal_charge_by_atom_name(
                        source_structure.constitution.residue_index(
                            ligand_site.residue_id
                        )
                    )
                ),
            ),
            source_structure.residue_formal_charge_by_atom_name(
                source_structure.constitution.residue_index(ligand_site.residue_id)
            ),
        )
        for ligand_site in source_structure.constitution.ligands
    )
    preserved_bonds = source_structure.topology.bonds_for_constitution(
        source_constitution=source_structure.constitution,
        target_constitution=updated_constitution,
    )
    repaired_atom_bonds = template_resolved_topology_bonds_for_new_atoms(
        source_constitution=source_structure.constitution,
        target_constitution=updated_constitution,
        component_library=component_library,
    )
    return ProteinStructure.from_payload(
        constitution=updated_constitution,
        geometry=StructureGeometry(
            constitution=updated_constitution,
            atom_geometries=tuple(
                residue.residue_geometry.atom_geometry(atom_site.name)
                for residue, _formal_charge_payload in updated_residue_entries_by_index
                for atom_site in residue.residue_site.atom_sites
            ),
        ),
        topology=StructureTopology(
            constitution=updated_constitution,
            atom_topologies=tuple(
                (
                    None
                    if formal_charge is None
                    else AtomTopology(formal_charge=formal_charge)
                )
                for residue, formal_charge_payload in updated_residue_entries_by_index
                for atom_site in residue.residue_site.atom_sites
                for formal_charge in (dict(formal_charge_payload).get(atom_site.name),)
            ),
            bonds=(*preserved_bonds, *repaired_atom_bonds),
        ),
        polymer_blueprint=source_structure.polymer_blueprint,
        provenance=source_structure.provenance,
    )


def _repair_heavy_atom_chain(
    *,
    chain_id: str,
    chain_residue_indices: tuple[ResidueIndex, ...],
    source_structure: ProteinStructure,
    reference_structure: ProteinStructure | None,
    component_library: ComponentLibrary,
    augment_c_terminal_oxt: bool,
    target_residue_ids: frozenset[ResidueId] | None,
    orphan_fragment_policy: OrphanFragmentPolicy,
    local_refinement: RepairLocalRefinementDirective | None,
) -> _ChainHeavyRepairStageResult:
    """Execute the chain-local heavy-completion stage for one chain."""

    normalized_residues: list[CompletionResiduePayload] = []
    repairs: list[RepairEvent] = []
    issues: list[ValidationIssue] = []

    for residue_index in chain_residue_indices:
        source_residue_site = source_structure.constitution.residue_site_at(
            residue_index
        )
        source_residue_geometry = source_structure.residue_geometry(residue_index)
        residue = CompletionResiduePayload(
            residue_site=source_residue_site,
            residue_geometry=source_residue_geometry,
            formal_charge_by_atom_name=(
                source_structure.residue_formal_charge_by_atom_name(residue_index)
            ),
        )
        if not _is_targeted_residue(
            residue.residue_id,
            target_residue_ids=target_residue_ids,
        ):
            normalized_residues.append(residue)
            continue

        normalized_residue, normalization_details = normalize_residue_inputs(
            residue,
            component_library,
        )
        if normalization_details is not None:
            repairs.append(
                RepairEvent.for_residue(
                    kind=RepairEventKind.COMPONENT_NORMALIZED,
                    residue_id=residue.residue_id,
                    component_id=normalized_residue.component_id,
                    atom_names=(),
                    details=normalization_details,
                )
            )
        normalized_residues.append(normalized_residue)

    reference_residue_by_id = normalized_reference_residue_by_id(
        reference_structure,
        chain_id=chain_id,
        component_library=component_library,
    )
    working_snapshot = _chain_snapshot(
        chain_id=chain_id,
        chain_residues=tuple(normalized_residues),
        source_structure=source_structure,
    )

    for residue_index_value, original_residue in enumerate(normalized_residues):
        residue_index = ResidueIndex(residue_index_value)
        current_residue = resolve_completion_residue_payload(
            working_snapshot,
            residue_index,
        )
        assert current_residue is not None
        if not _is_targeted_residue(
            current_residue.residue_id,
            target_residue_ids=target_residue_ids,
        ):
            continue

        template = component_library.get(current_residue.component_id)
        if template is None:
            diagnosis = diagnose_component_support(
                current_residue.component_id,
                component_library,
            )
            issues.append(
                missing_component_definition_issue(
                    current_residue.residue_site,
                    diagnosis=diagnosis,
                    action="leaving residue unchanged",
                )
            )
            continue

        fragment_match = match_residue_fragment(
            current_residue,
            template,
            exclude_atom_names=frozenset({"OXT"}),
        )
        prepared_residue = prepare_residue_for_fragment_repair(
            current_residue,
            fragment_match=fragment_match,
            orphan_fragment_policy=orphan_fragment_policy,
        )
        if prepared_residue != current_residue:
            working_snapshot = _snapshot_with_residue_payload(
                working_snapshot,
                prepared_residue,
            )

        missing_atom_names = missing_atoms_for_completion(
            prepared_residue.residue_site,
            template,
        )
        if missing_atom_names and any(
            atom_name in BACKBONE_ATOM_NAMES for atom_name in missing_atom_names
        ):
            issues.append(
                ValidationIssue.for_residue(
                    kind=ValidationIssueKind.INVALID_BACKBONE,
                    severity=IssueSeverity.ERROR,
                    message=(
                        f"{prepared_residue.residue_id.display_token()} "
                        "is missing required backbone atoms and cannot be repaired"
                    ),
                    residue_id=prepared_residue.residue_id,
                )
            )
            continue

        heavy_atom_semantics = template.heavy_atom_semantics
        if missing_atom_names and heavy_atom_semantics is None:
            issues.append(
                unsupported_template_repair_issue(
                    prepared_residue.residue_site,
                    diagnosis=diagnose_component_support(
                        prepared_residue.component_id,
                        component_library,
                    ),
                )
            )
            continue

        if missing_atom_names and isinstance(
            heavy_atom_semantics,
            IdealGeometryHeavyAtomSemantics,
        ):
            lineage_parent_component_id = template.lineage_parent_component_id
            residue_result, working_snapshot = _repair_supported_component_residue(
                snapshot=working_snapshot,
                residue_index=residue_index,
                original_residue=original_residue,
                residue=prepared_residue,
                template=template,
                reference_payload=reference_residue_by_id.get(
                    prepared_residue.residue_id
                ),
                fragment_match=fragment_match,
                orphan_fragment_policy=orphan_fragment_policy,
                heavy_atom_semantics=heavy_atom_semantics,
                parent_template=(
                    None
                    if lineage_parent_component_id is None
                    else component_library.get(lineage_parent_component_id)
                ),
                local_refinement=local_refinement,
            )
            if residue_result is None:
                issues.append(
                    unsupported_template_repair_issue(
                        prepared_residue.residue_site,
                        diagnosis=diagnose_component_support(
                            prepared_residue.component_id,
                            component_library,
                        ),
                    )
                )
                continue
        else:
            residue_result, working_snapshot = _repair_standard_residue(
                snapshot=working_snapshot,
                residue_index=residue_index,
                original_residue=original_residue,
                template=template,
                reference_payload=reference_residue_by_id.get(
                    prepared_residue.residue_id
                ),
                fragment_match=fragment_match,
                orphan_fragment_policy=orphan_fragment_policy,
            )

        if residue_result.added_atom_names:
            repairs.append(
                RepairEvent.for_residue(
                    kind=RepairEventKind.HEAVY_ATOMS_ADDED,
                    residue_id=residue_result.residue.residue_id,
                    component_id=residue_result.residue.component_id,
                    atom_names=residue_result.added_atom_names,
                )
            )

    repaired_residues = _chain_residue_payloads_from_snapshot(
        working_snapshot,
        residue_count=len(normalized_residues),
    )
    if (
        augment_c_terminal_oxt
        and repaired_residues
        and should_add_terminal_oxt(
            repaired_residues[-1].residue_site,
            component_library=component_library,
        )
    ):
        terminal_template = component_library.get(repaired_residues[-1].component_id)
        assert terminal_template is not None
        terminal_site = ResidueCompletionSite(
            residue_index=ResidueIndex(len(repaired_residues) - 1),
            template=terminal_template,
            original_payload=repaired_residues[-1],
        )
        working_snapshot = TerminalAtomPlacementTransformer(terminal_site).transform(
            ProteinTransformationContext.from_snapshot_atom_input(
                working_snapshot,
                terminal_site.atom_input(working_snapshot),
            )
        )
        repaired_terminal_residue = terminal_site.payload(working_snapshot)
        assert repaired_terminal_residue is not None
        repairs.append(
            RepairEvent.for_residue(
                kind=RepairEventKind.C_TERMINAL_OXT_ADDED,
                residue_id=repaired_terminal_residue.residue_id,
                component_id=repaired_terminal_residue.component_id,
                atom_names=("OXT",),
            )
        )
        repaired_residues = _chain_residue_payloads_from_snapshot(
            working_snapshot,
            residue_count=len(normalized_residues),
        )

    return _ChainHeavyRepairStageResult(
        chain_id=chain_id,
        residues=repaired_residues,
        repairs=tuple(repairs),
        issues=tuple(issues),
    )


def _snapshot_with_residue_payload(
    snapshot: ProteinStructureSnapshot,
    residue: CompletionResiduePayload,
) -> ProteinStructureSnapshot:
    """Return a snapshot updated with one residue payload."""

    return snapshot.with_updated_residue_facets(
        residue.residue_site,
        residue_geometry=residue.residue_geometry,
        formal_charge_by_atom_name=residue.formal_charge_by_atom_name,
    )


def _chain_residue_payloads_from_snapshot(
    snapshot: ProteinStructureSnapshot,
    *,
    residue_count: int,
) -> tuple[CompletionResiduePayload, ...]:
    """Resolve the current chain residue payloads from one working snapshot."""

    payloads: list[CompletionResiduePayload] = []
    for residue_index_value in range(residue_count):
        residue = resolve_completion_residue_payload(
            snapshot,
            ResidueIndex(residue_index_value),
        )
        assert residue is not None
        payloads.append(residue)

    return tuple(payloads)


def _repair_standard_residue(
    *,
    snapshot: ProteinStructureSnapshot,
    residue_index: ResidueIndex,
    original_residue: CompletionResiduePayload,
    template: ResidueTemplate,
    reference_payload: CompletionResiduePayload | None,
    fragment_match: ResidueFragmentMatch,
    orphan_fragment_policy: OrphanFragmentPolicy,
) -> tuple[_ResidueRepairStageResult, ProteinStructureSnapshot]:
    """Return one standard heavy-atom completion result and updated snapshot."""

    original_atom_name_set = frozenset(original_residue.atom_names())
    chain_length = len(snapshot.structure.constitution.residue_slots)
    site = ResidueCompletionSite(
        residue_index=residue_index,
        template=template,
        original_payload=original_residue,
        reference_payload=reference_payload,
        neighborhood=ResidueBackboneNeighborhood(
            previous_residue_index=ResidueIndex(
                (residue_index.value - 1) % chain_length
            ),
            next_residue_index=ResidueIndex((residue_index.value + 1) % chain_length),
        ),
    )
    current_residue = site.payload(snapshot)
    assert current_residue is not None
    if not site.missing_atom_names(snapshot):
        return (
            _ResidueRepairStageResult(
                residue=current_residue,
                added_atom_names=(),
            ),
            snapshot,
        )

    atom_input = site.atom_input(snapshot)
    source_context = ProteinTransformationContext.from_snapshot_atom_input(
        snapshot,
        atom_input,
    )

    guided_transformer = ReferenceAtomGraftTransformer(site)
    guided_snapshot = (
        guided_transformer.transform(source_context)
        if guided_transformer.is_applicable(source_context)
        else snapshot
    )
    guided_residue = site.payload(guided_snapshot)
    assert guided_residue is not None

    if not site.missing_atom_names(guided_snapshot):
        finalized_residue = _finalize_guided_residue(
            original_residue=original_residue,
            repaired_residue=guided_residue,
            template=template,
            fragment_match=fragment_match,
            orphan_fragment_policy=orphan_fragment_policy,
        )
        finalized_snapshot = (
            guided_snapshot
            if finalized_residue == guided_residue
            else _snapshot_with_residue_payload(guided_snapshot, finalized_residue)
        )
        return (
            _ResidueRepairStageResult(
                residue=finalized_residue,
                added_atom_names=_added_atom_names(
                    finalized_residue,
                    original_atom_name_set=original_atom_name_set,
                ),
            ),
            finalized_snapshot,
        )

    repaired_snapshot = InternalCoordinatePlacementTransformer(site).transform(
        ProteinTransformationContext.from_snapshot_atom_input(
            guided_snapshot,
            atom_input,
        )
    )
    repaired_residue = site.payload(repaired_snapshot)
    assert repaired_residue is not None
    finalized_residue = _apply_fragment_policy(
        original_residue=original_residue,
        repaired_residue=repaired_residue,
        fragment_match=fragment_match,
        orphan_fragment_policy=orphan_fragment_policy,
    )
    finalized_snapshot = (
        repaired_snapshot
        if finalized_residue == repaired_residue
        else _snapshot_with_residue_payload(repaired_snapshot, finalized_residue)
    )
    return (
        _ResidueRepairStageResult(
            residue=finalized_residue,
            added_atom_names=_added_atom_names(
                finalized_residue,
                original_atom_name_set=original_atom_name_set,
            ),
        ),
        finalized_snapshot,
    )


def _repair_supported_component_residue(
    *,
    snapshot: ProteinStructureSnapshot,
    residue_index: ResidueIndex,
    original_residue: CompletionResiduePayload,
    residue: CompletionResiduePayload,
    template: ResidueTemplate,
    reference_payload: CompletionResiduePayload | None,
    fragment_match: ResidueFragmentMatch,
    orphan_fragment_policy: OrphanFragmentPolicy,
    heavy_atom_semantics: IdealGeometryHeavyAtomSemantics,
    parent_template: ResidueTemplate | None,
    local_refinement: RepairLocalRefinementDirective | None,
) -> tuple[_ResidueRepairStageResult | None, ProteinStructureSnapshot]:
    """Return one supported component result and updated snapshot."""

    original_atom_name_set = frozenset(original_residue.atom_names())
    site = ResidueCompletionSite(
        residue_index=residue_index,
        template=template,
        original_payload=original_residue,
        reference_payload=reference_payload,
    )
    current_residue = site.payload(snapshot)
    assert current_residue is not None
    if not site.missing_atom_names(snapshot):
        return (
            _ResidueRepairStageResult(
                residue=current_residue,
                added_atom_names=(),
            ),
            snapshot,
        )

    atom_input = site.atom_input(snapshot)
    source_context = ProteinTransformationContext.from_snapshot_atom_input(
        snapshot,
        atom_input,
    )

    guided_transformer = ReferenceAtomGraftTransformer(site)
    guided_snapshot = (
        guided_transformer.transform(source_context)
        if guided_transformer.is_applicable(source_context)
        else snapshot
    )
    guided_residue = site.payload(guided_snapshot)
    assert guided_residue is not None

    if not site.missing_atom_names(guided_snapshot):
        finalized_residue = _finalize_guided_residue(
            original_residue=original_residue,
            repaired_residue=guided_residue,
            template=template,
            fragment_match=fragment_match,
            orphan_fragment_policy=orphan_fragment_policy,
        )
        finalized_snapshot = (
            guided_snapshot
            if finalized_residue == guided_residue
            else _snapshot_with_residue_payload(guided_snapshot, finalized_residue)
        )
        return (
            _ResidueRepairStageResult(
                residue=finalized_residue,
                added_atom_names=_added_atom_names(
                    finalized_residue,
                    original_atom_name_set=original_atom_name_set,
                ),
            ),
            finalized_snapshot,
        )

    if not allows_refinement_gated_component_repair(
        template,
        local_refinement=local_refinement,
        residue_id=guided_residue.residue_id,
    ):
        return None, guided_snapshot

    reconstruction_plan = build_component_reconstruction_plan(
        guided_residue,
        heavy_atom_semantics.component,
        template=template,
        parent_template=parent_template,
        orphan_fragment_policy=orphan_fragment_policy,
    )
    placement_directive = reconstruction_plan.rigid_component_placement_directive(
        original_residue=original_residue,
        residue=residue,
        semantics=heavy_atom_semantics,
        target_atom_names=site.missing_atom_names(guided_snapshot),
        orphan_fragment_policy=orphan_fragment_policy,
    )
    if placement_directive is None:
        return None, guided_snapshot

    placement_transformer = RigidFramePlacementTransformer(
        site,
        placement_directive,
    )
    guided_context = ProteinTransformationContext.from_snapshot_atom_input(
        guided_snapshot,
        atom_input,
    )
    if not placement_transformer.is_applicable(guided_context):
        return None, guided_snapshot

    repaired_snapshot = placement_transformer.transform(guided_context)
    repaired_residue = site.payload(repaired_snapshot)
    assert repaired_residue is not None
    finalized_residue = _apply_fragment_policy(
        original_residue=original_residue,
        repaired_residue=repaired_residue,
        fragment_match=fragment_match,
        orphan_fragment_policy=orphan_fragment_policy,
    )
    finalized_snapshot = (
        repaired_snapshot
        if finalized_residue == repaired_residue
        else _snapshot_with_residue_payload(repaired_snapshot, finalized_residue)
    )
    return (
        _ResidueRepairStageResult(
            residue=finalized_residue,
            added_atom_names=_added_atom_names(
                finalized_residue,
                original_atom_name_set=original_atom_name_set,
            ),
        ),
        finalized_snapshot,
    )


def _finalize_guided_residue(
    *,
    original_residue: CompletionResiduePayload,
    repaired_residue: CompletionResiduePayload,
    template: ResidueTemplate,
    fragment_match: ResidueFragmentMatch,
    orphan_fragment_policy: OrphanFragmentPolicy,
) -> CompletionResiduePayload:
    """Finalize a guidance-only residue using template heavy-atom order."""

    return _apply_fragment_policy(
        original_residue=original_residue,
        repaired_residue=repaired_residue.reordered(template.ordered_atom_names()),
        fragment_match=fragment_match,
        orphan_fragment_policy=orphan_fragment_policy,
    )


def _apply_fragment_policy(
    *,
    original_residue: CompletionResiduePayload,
    repaired_residue: CompletionResiduePayload,
    fragment_match: ResidueFragmentMatch,
    orphan_fragment_policy: OrphanFragmentPolicy,
) -> CompletionResiduePayload:
    """Apply orphan-fragment policy without altering transform materialization order."""

    return apply_orphan_fragment_policy(
        original_residue=original_residue,
        repaired_residue=repaired_residue,
        fragment_match=fragment_match,
        orphan_fragment_policy=orphan_fragment_policy,
    )


def _added_atom_names(
    residue: CompletionResiduePayload,
    *,
    original_atom_name_set: frozenset[str],
) -> tuple[str, ...]:
    """Return newly added atom names relative to one original residue."""

    return tuple(
        atom_site.name
        for atom_site in residue.atom_sites
        if atom_site.name not in original_atom_name_set
    )
