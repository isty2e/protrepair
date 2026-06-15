"""Primitive hydrogen materialization over prepared heavy-atom structures."""

from dataclasses import dataclass

from protrepair.chemistry import (
    ComponentLibrary,
    ResidueTemplate,
    build_default_component_library,
)
from protrepair.diagnostics.component_support import (
    diagnose_component_support,
    missing_component_definition_issue,
    unsupported_hydrogenation_issue,
)
from protrepair.diagnostics.events import RepairEvent, ValidationIssue
from protrepair.diagnostics.kinds import RepairEventKind
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ChainSite
from protrepair.structure.geometry import StructureGeometry
from protrepair.structure.labels import ResidueId
from protrepair.structure.slots import ChainIndex, ResidueIndex
from protrepair.structure.topology import AtomTopology, StructureTopology
from protrepair.transformer.completion.hydrogen.component_patch import (
    generate_component_hydrogen_patch,
)
from protrepair.transformer.completion.hydrogen.directives import (
    BackboneHydrogenPropagationDirective,
    HistidineDeltaProtonationDirective,
    HydrogenCompletionDirective,
    NTerminalHydrogenPlacementDirective,
    RigidHydrogenPlacementDirective,
    StaticHydrogenPlacementDirective,
    derive_hydrogen_directives,
    hydrogen_placement_directive,
)
from protrepair.transformer.completion.hydrogen.domain import (
    HydrogenCompletionEnvironment,
    HydrogenResidueSite,
)
from protrepair.transformer.completion.hydrogen.geometry import backbone_hydrogen
from protrepair.transformer.completion.hydrogen.static_patch import (
    generate_hydrogen_patch,
    histidine_delta_hydrogen,
    n_terminal_hydrogen_coordinates,
)
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload
from protrepair.transformer.completion.shared.patch import OrderedAtomPatch
from protrepair.transformer.result import TransformationResult


@dataclass(frozen=True, slots=True)
class _HydrogenChainStageResult:
    """One chain-local hydrogen-completion stage result."""

    chain_id: str
    residues: tuple[CompletionResiduePayload, ...]
    repairs: tuple[RepairEvent, ...]
    issues: tuple[ValidationIssue, ...]


def materialize_hydrogens_core(
    structure: ProteinStructure,
    component_library: ComponentLibrary | None = None,
    *,
    target_residue_ids: frozenset[ResidueId] | None = None,
    protonate_histidines: bool = False,
) -> TransformationResult:
    """Materialize hydrogens on the current heavy-atom structure."""

    library = (
        build_default_component_library()
        if component_library is None
        else component_library
    )
    return _execute_hydrogen_placement_stage(
        TransformationResult(
            structure=structure,
            repairs=(),
            issues=(),
        ),
        protonate_histidines=protonate_histidines,
        component_library=library,
        target_residue_ids=target_residue_ids,
        original_structure=structure,
    )


def _hydrogenate_chain_stage(
    *,
    chain_id: str,
    chain_residues: tuple[CompletionResiduePayload, ...],
    protonate_histidines: bool,
    component_library: ComponentLibrary,
    source_structure: ProteinStructure,
    target_residue_ids: frozenset[ResidueId] | None = None,
) -> _HydrogenChainStageResult:
    """Hydrogenate one chain and return chain-local repairs and issues."""

    if not chain_residues:
        return _HydrogenChainStageResult(
            chain_id=chain_id,
            residues=(),
            repairs=(),
            issues=(),
        )

    templates = [
        component_library.get(residue.component_id) for residue in chain_residues
    ]
    issues: list[ValidationIssue] = []

    environment = HydrogenCompletionEnvironment.from_payloads(
        chain_residues,
        templates=tuple(templates),
    )
    working_residues = list(chain_residues)

    for residue_index, residue in enumerate(chain_residues):
        if not _is_targeted_residue(
            residue.residue_id,
            target_residue_ids=target_residue_ids,
        ):
            continue
        template = templates[residue_index]
        placement_directive = hydrogen_placement_directive(
            residue_index=ResidueIndex(residue_index),
            template=template,
        )
        if template is None:
            diagnosis = diagnose_component_support(
                residue.component_id,
                component_library,
            )
            issues.append(
                missing_component_definition_issue(
                    residue.residue_site,
                    diagnosis=diagnosis,
                    action="leaving residue unchanged during hydrogenation",
                )
            )
            continue

        if placement_directive is None:
            diagnosis = diagnose_component_support(
                residue.component_id,
                component_library,
            )
            issues.append(
                unsupported_hydrogenation_issue(
                    residue.residue_site,
                    diagnosis=diagnosis,
                )
            )
    directives = derive_hydrogen_directives(
        source_structure.constitution.chain(chain_id),
        templates=tuple(templates),
        protonate_histidines=protonate_histidines,
    )
    protonated_histidines = {
        _directive_residue_index(directive)
        for directive in directives
        if _is_targeted_residue(
            chain_residues[_directive_residue_index(directive).value].residue_id,
            target_residue_ids=target_residue_ids,
        )
        if isinstance(directive, HistidineDeltaProtonationDirective)
    }
    for directive in directives:
        if not _is_targeted_residue(
            chain_residues[_directive_residue_index(directive).value].residue_id,
            target_residue_ids=target_residue_ids,
        ):
            continue
        _apply_hydrogen_directive(
            directive,
            chain_residues_by_index=working_residues,
            environment=environment,
        )

    hydrogenated_residues = tuple(working_residues)
    ordered_residues = tuple(
        hydrogenated_residue.reordered(
            _ordered_hydrogenated_atom_names(
                residue_index,
                chain_residues=hydrogenated_residues,
                templates=tuple(templates),
            )
        )
        if template is not None
        else hydrogenated_residue
        for residue_index, (hydrogenated_residue, template) in enumerate(
            zip(
                hydrogenated_residues,
                templates,
                strict=True,
            )
        )
    )
    repairs: list[RepairEvent] = []
    for residue_index, (residue, hydrogenated_residue) in enumerate(
        zip(
            chain_residues,
            ordered_residues,
            strict=True,
        )
    ):
        if not _is_targeted_residue(
            residue.residue_id,
            target_residue_ids=target_residue_ids,
        ):
            continue
        added_atoms = tuple(
            atom_name
            for atom_name in hydrogenated_residue.atom_names()
            if atom_name not in residue.atom_names()
        )
        if added_atoms:
            details = None
            if ResidueIndex(residue_index) in protonated_histidines:
                details = "histidine protonation (+1 charge) applied"
            repairs.append(
                RepairEvent.for_residue(
                    kind=RepairEventKind.HYDROGENS_ADDED,
                    residue_id=residue.residue_id,
                    component_id=residue.component_id,
                    atom_names=added_atoms,
                    details=details,
                )
            )

    return _HydrogenChainStageResult(
        chain_id=chain_id,
        residues=ordered_residues,
        repairs=tuple(repairs),
        issues=tuple(issues),
    )


def _execute_hydrogen_placement_stage(
    prepared_result: TransformationResult,
    *,
    protonate_histidines: bool,
    component_library: ComponentLibrary,
    target_residue_ids: frozenset[ResidueId] | None,
    original_structure: ProteinStructure,
) -> TransformationResult:
    """Execute the chain-local hydrogen placement stage over one structure."""

    if target_residue_ids is not None:
        return _execute_targeted_polymer_hydrogen_placement_stage(
            prepared_result,
            protonate_histidines=protonate_histidines,
            component_library=component_library,
            target_residue_ids=target_residue_ids,
        )

    placement_input = prepared_result.structure.without_hydrogens()
    repaired_chain_results: list[_HydrogenChainStageResult] = []
    repairs = list(prepared_result.repairs)
    issues = list(prepared_result.issues)

    for chain_offset, chain_site in enumerate(placement_input.constitution.chains):
        chain_result = _hydrogenate_chain_stage(
            chain_id=chain_site.chain_id,
            chain_residues=_chain_residue_payloads(
                placement_input,
                chain_index=ChainIndex(chain_offset),
            ),
            protonate_histidines=protonate_histidines,
            component_library=component_library,
            source_structure=placement_input,
            target_residue_ids=target_residue_ids,
        )
        repaired_chain_results.append(chain_result)
        repairs.extend(chain_result.repairs)
        issues.extend(chain_result.issues)

    repaired_structure = _structure_from_hydrogen_chain_results(
        source_structure=placement_input,
        original_structure=original_structure,
        chain_results=tuple(repaired_chain_results),
    )
    return TransformationResult(
        structure=repaired_structure,
        repairs=tuple(repairs),
        issues=tuple(issues),
    )


def _execute_targeted_polymer_hydrogen_placement_stage(
    prepared_result: TransformationResult,
    *,
    protonate_histidines: bool,
    component_library: ComponentLibrary,
    target_residue_ids: frozenset[ResidueId],
) -> TransformationResult:
    """Execute hydrogen placement only on polymer chains containing target residues."""

    source_structure = prepared_result.structure
    target_chain_ids = _target_polymer_chain_ids(
        source_structure,
        target_residue_ids=target_residue_ids,
    )
    if not target_chain_ids:
        return prepared_result

    repaired_structure = prepared_result.structure
    repairs = list(prepared_result.repairs)
    issues = list(prepared_result.issues)
    repaired_target_residues: list[CompletionResiduePayload] = []
    for chain_offset, chain_site in enumerate(source_structure.constitution.chains):
        if chain_site.chain_id not in target_chain_ids:
            continue

        chain_result = _hydrogenate_chain_stage(
            chain_id=chain_site.chain_id,
            chain_residues=_chain_residue_payloads(
                source_structure,
                chain_index=ChainIndex(chain_offset),
                hydrogen_stripped_residue_ids=target_residue_ids,
            ),
            protonate_histidines=protonate_histidines,
            component_library=component_library,
            source_structure=source_structure,
            target_residue_ids=target_residue_ids,
        )
        repairs.extend(chain_result.repairs)
        issues.extend(chain_result.issues)
        for residue in chain_result.residues:
            if not _is_targeted_residue(
                residue.residue_id,
                target_residue_ids=target_residue_ids,
            ):
                continue

            repaired_target_residues.append(residue)

    if repaired_target_residues:
        repaired_structure = repaired_structure.with_updated_residue_facets_batch(
            (
                residue.residue_site,
                residue.residue_geometry,
                residue.formal_charge_by_atom_name,
            )
            for residue in repaired_target_residues
        )

    return TransformationResult(
        structure=repaired_structure,
        repairs=tuple(repairs),
        issues=tuple(issues),
    )


def _target_polymer_chain_ids(
    structure: ProteinStructure,
    *,
    target_residue_ids: frozenset[ResidueId],
) -> frozenset[str]:
    """Return polymer chain ids that contain at least one target residue."""

    chain_ids: set[str] = set()
    for chain_site in structure.constitution.chains:
        if any(
            residue.residue_id in target_residue_ids for residue in chain_site.residues
        ):
            chain_ids.add(chain_site.chain_id)

    return frozenset(chain_ids)


def _chain_residue_payloads(
    structure: ProteinStructure,
    *,
    chain_index: ChainIndex,
    hydrogen_stripped_residue_ids: frozenset[ResidueId] = frozenset(),
) -> tuple[CompletionResiduePayload, ...]:
    """Return completion payloads for one polymer chain."""

    payloads: list[CompletionResiduePayload] = []
    for residue_index in structure.constitution.residue_indices_for_chain_index(
        chain_index
    ):
        residue_site = structure.constitution.residue_site_at(residue_index)
        residue_geometry = structure.residue_geometry(residue_index)
        formal_charge_by_atom_name = structure.residue_formal_charge_by_atom_name(
            residue_index
        )
        if residue_site.residue_id in hydrogen_stripped_residue_ids:
            hydrogen_atom_names = {
                atom_site.name
                for atom_site in residue_site.atom_sites
                if atom_site.element == "H"
            }
            residue_site = residue_site.without_atom_sites(hydrogen_atom_names)
            residue_geometry = residue_geometry.without_atoms(hydrogen_atom_names)
            formal_charge_by_atom_name = tuple(
                (atom_name, formal_charge)
                for atom_name, formal_charge in formal_charge_by_atom_name
                if atom_name not in hydrogen_atom_names
            )

        payloads.append(
            CompletionResiduePayload(
                residue_site=residue_site,
                residue_geometry=residue_geometry,
                formal_charge_by_atom_name=formal_charge_by_atom_name,
            )
        )

    return tuple(payloads)


def _is_targeted_residue(
    residue_id: ResidueId,
    *,
    target_residue_ids: frozenset[ResidueId] | None,
) -> bool:
    """Return whether one residue is in the active workflow-stage scope."""

    return target_residue_ids is None or residue_id in target_residue_ids


def _directive_residue_index(
    directive: HydrogenCompletionDirective,
) -> ResidueIndex:
    """Return the receiving residue slot for one hydrogen directive."""

    if isinstance(directive, StaticHydrogenPlacementDirective):
        return directive.residue_index
    if isinstance(directive, RigidHydrogenPlacementDirective):
        return directive.residue_index
    if isinstance(directive, HistidineDeltaProtonationDirective):
        return directive.residue_index
    if isinstance(directive, BackboneHydrogenPropagationDirective):
        return directive.next_residue_index
    if isinstance(directive, NTerminalHydrogenPlacementDirective):
        return directive.residue_index

    raise TypeError(f"unsupported hydrogen directive type {type(directive)!r}")


def _ordered_hydrogenated_atom_names(
    residue_index: int,
    *,
    chain_residues: tuple[CompletionResiduePayload, ...],
    templates: tuple[ResidueTemplate | None, ...],
) -> tuple[str, ...]:
    """Return canonical heavy-plus-hydrogen atom order for one chain residue."""

    template = templates[residue_index]
    if template is None:
        return ()

    ordered_atom_names = list(template.ordered_atom_names())
    ordered_atom_names.extend(template.expected_hydrogen_atom_names())

    residue = chain_residues[residue_index]
    if (
        residue_index == 0
        and template.can_add_hydrogens()
        and _supports_peptide_backbone_hydrogens(residue)
    ):
        backbone_family_component_id = template.backbone_family_component_id
        ordered_atom_names.extend(
            ("H1", "H2")
            if backbone_family_component_id == "PRO"
            else ("H1", "H2", "H3")
        )

    if residue_index > 0:
        previous_template = templates[residue_index - 1]
        if (
            previous_template is not None
            and previous_template.can_add_hydrogens()
            and template.can_add_hydrogens()
            and _supports_peptide_backbone_hydrogens(chain_residues[residue_index - 1])
            and _supports_peptide_backbone_hydrogens(residue)
        ):
            backbone_family_component_id = template.backbone_family_component_id
            if backbone_family_component_id != "PRO":
                ordered_atom_names.append("H")

    return tuple(ordered_atom_names)


def _supports_peptide_backbone_hydrogens(
    residue: CompletionResiduePayload,
) -> bool:
    """Return whether one residue can participate in peptide-H ordering."""

    return all(residue.has_atom(atom_name) for atom_name in ("N", "CA", "C"))


def _structure_from_hydrogen_chain_results(
    *,
    source_structure: ProteinStructure,
    original_structure: ProteinStructure,
    chain_results: tuple[_HydrogenChainStageResult, ...],
) -> ProteinStructure:
    """Return one structure rebuilt from hydrogenated chain residue payload."""

    chain_result_by_id = {
        chain_result.chain_id: chain_result for chain_result in chain_results
    }
    repaired_chain_sites: list[ChainSite] = []
    updated_residue_entries_by_index: list[
        tuple[CompletionResiduePayload, tuple[tuple[str, int | None], ...]]
    ] = []

    for source_chain_site in source_structure.constitution.chains:
        chain_result = chain_result_by_id[source_chain_site.chain_id]
        repaired_chain_sites.append(
            ChainSite(
                chain_id=source_chain_site.chain_id,
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
        tuple(repaired_chain_sites)
    ).with_ligands(tuple(original_structure.constitution.ligands))
    updated_residue_entries_by_index.extend(
        (
            CompletionResiduePayload(
                residue_site=ligand,
                residue_geometry=original_structure.residue_geometry(
                    original_structure.constitution.residue_index(ligand.residue_id)
                ),
                formal_charge_by_atom_name=(
                    original_structure.residue_formal_charge_by_atom_name(
                        original_structure.constitution.residue_index(ligand.residue_id)
                    )
                ),
            ),
            original_structure.residue_formal_charge_by_atom_name(
                original_structure.constitution.residue_index(ligand.residue_id)
            ),
        )
        for ligand in original_structure.constitution.ligands
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
            bonds=source_structure.topology.bonds_for_constitution(
                source_constitution=source_structure.constitution,
                target_constitution=updated_constitution,
            ),
        ),
        polymer_blueprint=source_structure.polymer_blueprint,
        provenance=original_structure.provenance,
    )


def _apply_hydrogen_directive(
    directive: HydrogenCompletionDirective,
    *,
    chain_residues_by_index: list[CompletionResiduePayload],
    environment: HydrogenCompletionEnvironment,
) -> None:
    """Apply one hydrogen directive onto the current chain-local payload state."""

    if isinstance(directive, StaticHydrogenPlacementDirective):
        current_residue = chain_residues_by_index[directive.residue_index.value]
        site = HydrogenResidueSite(
            residue_index=directive.residue_index,
            template=directive.template,
            environment=environment,
            next_residue_index=directive.next_residue_index,
        )
        patch = OrderedAtomPatch.from_residue_payload(
            current_residue.residue_site,
            residue_geometry=current_residue.residue_geometry,
        )
        chain_residues_by_index[directive.residue_index.value] = (
            current_residue.apply_patch(
                generate_hydrogen_patch(
                    site=site,
                    patch=patch,
                    semantics=directive.semantics,
                )
            )
        )
        return

    if isinstance(directive, RigidHydrogenPlacementDirective):
        current_residue = chain_residues_by_index[directive.residue_index.value]
        patch = generate_component_hydrogen_patch(
            residue=current_residue,
            patch=OrderedAtomPatch.from_residue_payload(
                current_residue.residue_site,
                residue_geometry=current_residue.residue_geometry,
            ),
            semantics=directive.semantics,
        )
        if patch is None:
            return

        chain_residues_by_index[directive.residue_index.value] = (
            current_residue.apply_patch(patch)
        )
        return

    if isinstance(directive, HistidineDeltaProtonationDirective):
        current_residue = chain_residues_by_index[directive.residue_index.value]
        if current_residue.component_id != "HIS" or current_residue.has_atom("HD1"):
            return

        current_patch = OrderedAtomPatch.from_residue_payload(
            current_residue.residue_site,
            residue_geometry=current_residue.residue_geometry,
        )
        chain_residues_by_index[directive.residue_index.value] = (
            current_residue.apply_patch(
                current_patch.append_atoms(
                    ("HD1",),
                    (histidine_delta_hydrogen(current_patch),),
                )
            )
        )
        return

    if isinstance(directive, BackboneHydrogenPropagationDirective):
        current_residue = chain_residues_by_index[directive.residue_index.value]
        next_residue = chain_residues_by_index[directive.next_residue_index.value]
        if next_residue.has_atom("H"):
            return

        if not current_residue.has_atom("C") or not all(
            next_residue.has_atom(atom_name) for atom_name in ("CA", "N")
        ):
            return

        position = backbone_hydrogen(
            next_residue.residue_geometry.position("CA"),
            next_residue.residue_geometry.position("N"),
            current_residue.residue_geometry.position("C"),
        )
        next_patch = OrderedAtomPatch.from_residue_payload(
            next_residue.residue_site,
            residue_geometry=next_residue.residue_geometry,
        )
        chain_residues_by_index[directive.next_residue_index.value] = (
            next_residue.apply_patch(next_patch.append_atoms(("H",), (position,)))
        )
        return

    if isinstance(directive, NTerminalHydrogenPlacementDirective):
        current_residue = chain_residues_by_index[directive.residue_index.value]
        current_patch = OrderedAtomPatch.from_residue_payload(
            current_residue.residue_site,
            residue_geometry=current_residue.residue_geometry,
        )
        atom_coordinates = tuple(
            n_terminal_hydrogen_coordinates(
                current_patch,
                directive.backbone_family_component_id,
            )
        )
        atom_names = (
            ("H1", "H2")
            if directive.backbone_family_component_id == "PRO"
            else ("H1", "H2", "H3")
        )
        chain_residues_by_index[directive.residue_index.value] = (
            current_residue.apply_patch(
                current_patch.append_atoms(atom_names, atom_coordinates)
            )
        )
        return

    raise TypeError(f"unsupported hydrogen directive type {type(directive)!r}")
