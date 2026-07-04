"""Hydrogen completion over retained non-polymer components."""

from collections.abc import Mapping
from dataclasses import dataclass

from protrepair.chemistry import (
    ComponentLibrary,
    IdealGeometryHydrogenSemantics,
    ResidueTemplate,
    build_default_component_library,
)
from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.component.topology import (
    template_hydrogen_bond_definitions_for_names,
)
from protrepair.chemistry.inference.retained_non_polymer_fallback import (
    retained_non_polymer_rdkit_fallback_hydrogen_bond_definitions,
    retained_non_polymer_rdkit_fallback_hydrogenated_molecule,
)
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
    evidence_by_residue_id,
)
from protrepair.diagnostics.chemistry_contradictions import (
    diagnose_retained_non_polymer_template_chemistry_contradictions,
)
from protrepair.diagnostics.component_support import (
    diagnose_component_support,
    missing_component_definition_issue,
)
from protrepair.diagnostics.events import RepairEvent, ValidationIssue
from protrepair.diagnostics.kinds import (
    IssueSeverity,
    RepairEventKind,
    ValidationIssueKind,
)
from protrepair.errors import RdkitUnavailableError
from protrepair.geometry import Vec3
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import ResidueId
from protrepair.structure.slots import ResidueIndex
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.hydrogen.cleanup import (
    cleanup_residual_hydrogen_clashes,
)
from protrepair.transformer.completion.hydrogen.component_patch import (
    generate_component_hydrogen_patch,
)
from protrepair.transformer.completion.hydrogen.directives import (
    RigidHydrogenPlacementDirective,
    StaticHydrogenPlacementDirective,
    hydrogen_placement_directive,
)
from protrepair.transformer.completion.hydrogen.domain import (
    HydrogenCompletionEnvironment,
    HydrogenResidueSite,
)
from protrepair.transformer.completion.hydrogen.static_patch import (
    generate_hydrogen_patch,
)
from protrepair.transformer.completion.retained_non_polymer_hydrogen import (
    rdkit_evidence,
)
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload
from protrepair.transformer.completion.shared.patch import OrderedAtomPatch
from protrepair.transformer.result import TransformationResult


@dataclass(frozen=True, slots=True)
class _RetainedNonPolymerHydrogenStageResult:
    """One retained non-polymer hydrogenation stage result."""

    structure: ProteinStructure
    repairs: tuple[RepairEvent, ...]
    issues: tuple[ValidationIssue, ...]


@dataclass(frozen=True, slots=True)
class _RetainedNonPolymerHydrogenPayloadResult:
    """One retained non-polymer hydrogenation payload result."""

    payload: CompletionResiduePayload | None
    repairs: tuple[RepairEvent, ...]
    issues: tuple[ValidationIssue, ...]
    topology_plan: "_RetainedNonPolymerHydrogenTopologyPlan | None" = None


@dataclass(frozen=True, slots=True)
class _RetainedNonPolymerHydrogenTopologyPlan:
    """Residue-local retained ligand H-anchor topology to apply after rebuild."""

    residue_id: ResidueId
    bond_definitions: tuple[BondDefinition, ...]
    provenance: BondProvenance


@dataclass(frozen=True, slots=True)
class _RdkitFallbackHydrogenationResult:
    """Fallback hydrogenated payload plus RDKit-derived H-heavy anchors."""

    payload: CompletionResiduePayload
    hydrogen_bond_definitions: tuple[BondDefinition, ...]


def add_retained_non_polymer_hydrogens(
    structure: ProteinStructure,
    component_library: ComponentLibrary | None = None,
    *,
    target_residue_ids: frozenset[ResidueId] | None = None,
    chemistry_evidence: tuple[RetainedNonPolymerChemistryEvidence, ...] = (),
) -> TransformationResult:
    """Add hydrogens to supported retained non-polymer components."""

    library = (
        build_default_component_library()
        if component_library is None
        else component_library
    )
    active_target_residue_ids = (
        frozenset(ligand.residue_id for ligand in structure.constitution.ligands)
        if target_residue_ids is None
        else frozenset(target_residue_ids)
    )
    if not active_target_residue_ids:
        return TransformationResult(
            structure=structure,
            repairs=(),
            issues=(),
        )

    stripped_structure = structure.without_hydrogens_in_residues(
        active_target_residue_ids
    )
    stage_result = _execute_retained_non_polymer_hydrogen_stage(
        stripped_structure,
        topology_source_structure=structure,
        component_library=library,
        target_residue_ids=active_target_residue_ids,
        chemistry_evidence_by_residue_id=evidence_by_residue_id(chemistry_evidence),
    )
    cleaned_structure, cleanup_issues = cleanup_residual_hydrogen_clashes(
        stage_result.structure,
        component_library=library,
        target_residue_ids=active_target_residue_ids,
    )
    return TransformationResult(
        structure=cleaned_structure,
        repairs=stage_result.repairs,
        issues=stage_result.issues + cleanup_issues,
    )


def _execute_retained_non_polymer_hydrogen_stage(
    structure: ProteinStructure,
    *,
    topology_source_structure: ProteinStructure,
    component_library: ComponentLibrary,
    target_residue_ids: frozenset[ResidueId],
    chemistry_evidence_by_residue_id: Mapping[
        ResidueId,
        RetainedNonPolymerChemistryEvidence,
    ],
) -> _RetainedNonPolymerHydrogenStageResult:
    """Apply hydrogen patches to targeted retained non-polymer components."""

    hydrogenated_payloads: list[CompletionResiduePayload] = []
    topology_plans: list[_RetainedNonPolymerHydrogenTopologyPlan] = []
    repairs: list[RepairEvent] = []
    issues: list[ValidationIssue] = []
    for ligand in structure.constitution.ligands:
        if ligand.residue_id not in target_residue_ids:
            continue

        result = _hydrogenate_retained_non_polymer_payload(
            _payload_for_structure(structure, ligand.residue_id),
            component_library=component_library,
            chemistry_evidence=chemistry_evidence_by_residue_id.get(ligand.residue_id),
        )
        if result.payload is not None:
            hydrogenated_payloads.append(result.payload)
        if result.topology_plan is not None:
            topology_plans.append(result.topology_plan)
        repairs.extend(result.repairs)
        issues.extend(result.issues)

    updated_structure = (
        structure.with_updated_residue_facets_batch(
            (
                payload.residue_site,
                payload.residue_geometry,
                payload.formal_charge_by_atom_name,
            )
            for payload in hydrogenated_payloads
        )
        if hydrogenated_payloads
        else structure
    )
    if hydrogenated_payloads:
        updated_structure = _with_retained_non_polymer_hydrogen_topology_bonds(
            topology_source_structure=topology_source_structure,
            target_structure=updated_structure,
            topology_plans=tuple(topology_plans),
        )
    return _RetainedNonPolymerHydrogenStageResult(
        structure=updated_structure,
        repairs=tuple(repairs),
        issues=tuple(issues),
    )


def _hydrogenate_retained_non_polymer_payload(
    payload: CompletionResiduePayload,
    *,
    component_library: ComponentLibrary,
    chemistry_evidence: RetainedNonPolymerChemistryEvidence | None = None,
) -> _RetainedNonPolymerHydrogenPayloadResult:
    """Hydrogenate one retained non-polymer residue when supported."""

    if chemistry_evidence is not None:
        hydrogenation_result = (
            rdkit_evidence.hydrogenate_retained_non_polymer_payload_with_evidence_result(
                payload,
                evidence=chemistry_evidence,
            )
        )
        hydrogenated_payload = hydrogenation_result.payload
        added_atom_names = tuple(
            atom_name
            for atom_name in hydrogenated_payload.atom_names()
            if atom_name not in payload.atom_names()
        )
        return _RetainedNonPolymerHydrogenPayloadResult(
            payload=hydrogenated_payload,
            repairs=(
                ()
                if not added_atom_names
                else (
                    RepairEvent.for_residue(
                        kind=RepairEventKind.HYDROGENS_ADDED,
                        residue_id=payload.residue_id,
                        component_id=payload.component_id,
                        atom_names=added_atom_names,
                    ),
                )
            ),
            issues=(),
            topology_plan=_topology_plan_for_new_hydrogens(
                residue_id=payload.residue_id,
                bond_definitions=hydrogenation_result.hydrogen_bond_definitions,
                added_atom_names=added_atom_names,
                provenance=BondProvenance.EVIDENCE_RESOLVED,
            ),
        )

    template = component_library.get(payload.component_id)
    if template is not None:
        contradiction_issues = (
            diagnose_retained_non_polymer_template_chemistry_contradictions(
                payload.residue_site,
                residue_geometry=payload.residue_geometry,
                source_formal_charge_by_atom_name=dict(
                    payload.formal_charge_by_atom_name
                ),
                template=template,
            )
        )
        placement_directive = hydrogen_placement_directive(
            residue_index=ResidueIndex(0),
            template=template,
        )
        if placement_directive is not None:
            hydrogenated_payload = _apply_retained_non_polymer_hydrogen_directive(
                payload,
                template=template,
                placement_directive=placement_directive,
            )
            if hydrogenated_payload is None:
                return _RetainedNonPolymerHydrogenPayloadResult(
                    payload=None,
                    repairs=(),
                    issues=contradiction_issues,
                )

            ordered_payload = hydrogenated_payload.reordered(
                _retained_non_polymer_template_atom_order(
                    hydrogenated_payload,
                    template=template,
                )
            )
            added_atom_names = tuple(
                atom_name
                for atom_name in ordered_payload.atom_names()
                if atom_name not in payload.atom_names()
            )
            return _RetainedNonPolymerHydrogenPayloadResult(
                payload=ordered_payload,
                repairs=(
                    ()
                    if not added_atom_names
                    else (
                        RepairEvent.for_residue(
                            kind=RepairEventKind.HYDROGENS_ADDED,
                            residue_id=payload.residue_id,
                            component_id=payload.component_id,
                            atom_names=added_atom_names,
                        ),
                    )
                ),
                issues=contradiction_issues,
                topology_plan=_topology_plan_for_new_hydrogens(
                    residue_id=payload.residue_id,
                    bond_definitions=template_hydrogen_bond_definitions_for_names(
                        template,
                        hydrogen_atom_names=added_atom_names,
                    ),
                    added_atom_names=added_atom_names,
                    provenance=BondProvenance.TEMPLATE_RESOLVED,
                ),
            )

    try:
        hydrogenation_result = (
            _hydrogenate_retained_non_polymer_payload_with_rdkit_fallback_result(
                payload
            )
        )
        hydrogenated_payload = hydrogenation_result.payload
    except (RdkitUnavailableError, RuntimeError, ValueError) as error:
        return _RetainedNonPolymerHydrogenPayloadResult(
            payload=None,
            repairs=(),
            issues=(
                _retained_non_polymer_hydrogen_fallback_issue(
                    payload.residue_site,
                    component_library=component_library,
                    error=error,
                ),
            ),
        )

    ordered_payload = hydrogenated_payload
    added_atom_names = tuple(
        atom_name
        for atom_name in ordered_payload.atom_names()
        if atom_name not in payload.atom_names()
    )
    return _RetainedNonPolymerHydrogenPayloadResult(
        payload=ordered_payload,
        repairs=(
            ()
            if not added_atom_names
            else (
                RepairEvent.for_residue(
                    kind=RepairEventKind.HYDROGENS_ADDED,
                    residue_id=payload.residue_id,
                    component_id=payload.component_id,
                    atom_names=added_atom_names,
                ),
            )
        ),
        issues=(),
        topology_plan=_topology_plan_for_new_hydrogens(
            residue_id=payload.residue_id,
            bond_definitions=hydrogenation_result.hydrogen_bond_definitions,
            added_atom_names=added_atom_names,
            provenance=BondProvenance.REPAIR_INFERRED,
        ),
    )


def hydrogenate_retained_non_polymer_payload_with_rdkit_fallback(
    payload: CompletionResiduePayload,
) -> CompletionResiduePayload:
    """Hydrogenate one retained non-polymer payload with RDKit fallback."""

    return _hydrogenate_retained_non_polymer_payload_with_rdkit_fallback_result(
        payload
    ).payload


def _hydrogenate_retained_non_polymer_payload_with_rdkit_fallback_result(
    payload: CompletionResiduePayload,
) -> _RdkitFallbackHydrogenationResult:
    """Hydrogenate one payload and return fallback-inferred H anchors."""

    hydrogenated_molecule = retained_non_polymer_rdkit_fallback_hydrogenated_molecule(
        payload.residue_site,
        payload.residue_geometry,
        formal_charge_by_atom_name=dict(payload.formal_charge_by_atom_name),
    )
    return _RdkitFallbackHydrogenationResult(
        payload=payload.apply_patch(
            _rdkit_hydrogen_append_patch(
                payload,
                hydrogenated_molecule=hydrogenated_molecule,
            )
        ),
        hydrogen_bond_definitions=(
            retained_non_polymer_rdkit_fallback_hydrogen_bond_definitions(
                hydrogenated_molecule
            )
        ),
    )


def _with_retained_non_polymer_hydrogen_topology_bonds(
    *,
    topology_source_structure: ProteinStructure,
    target_structure: ProteinStructure,
    topology_plans: tuple[_RetainedNonPolymerHydrogenTopologyPlan, ...],
) -> ProteinStructure:
    """Return target structure with source-first retained ligand H topology."""

    source_bonds = topology_source_structure.topology.bonds_for_constitution(
        source_constitution=topology_source_structure.constitution,
        target_constitution=target_structure.constitution,
    )
    generated_bonds = tuple(
        topology_bond
        for plan in topology_plans
        for topology_bond in _retained_non_polymer_topology_bonds_for_plan(
            target_structure,
            plan,
        )
    )
    topology = StructureTopology(
        constitution=target_structure.constitution,
        atom_topologies=target_structure.topology.atom_topologies,
        bonds=_merge_source_first_topology_bonds(source_bonds, generated_bonds),
    )
    return type(target_structure).from_payload(
        constitution=target_structure.constitution,
        geometry=target_structure.geometry,
        topology=topology,
        polymer_blueprint=target_structure.polymer_blueprint,
        provenance=target_structure.provenance,
    )


def _retained_non_polymer_topology_bonds_for_plan(
    structure: ProteinStructure,
    plan: _RetainedNonPolymerHydrogenTopologyPlan,
) -> tuple[TopologyBond, ...]:
    """Project one residue-local H-anchor plan into canonical atom slots."""

    residue_index = structure.constitution.residue_index(plan.residue_id)
    residue_site = structure.constitution.residue_site_at(residue_index)
    present_atom_names = frozenset(residue_site.atom_site_names())
    bonds: list[TopologyBond] = []
    for bond_definition in plan.bond_definitions:
        if (
            bond_definition.atom_name_1 not in present_atom_names
            or bond_definition.atom_name_2 not in present_atom_names
        ):
            continue

        bonds.append(
            TopologyBond(
                atom_index_1=structure.constitution.atom_index_in_residue(
                    residue_index,
                    bond_definition.atom_name_1,
                ),
                atom_index_2=structure.constitution.atom_index_in_residue(
                    residue_index,
                    bond_definition.atom_name_2,
                ),
                order=bond_definition.order,
                aromatic=bond_definition.aromatic,
                relationship_type=BondRelationshipType.COVALENT,
                provenance=plan.provenance,
            )
        )

    return tuple(bonds)


def _merge_source_first_topology_bonds(
    source_bonds: tuple[TopologyBond, ...],
    generated_bonds: tuple[TopologyBond, ...],
) -> tuple[TopologyBond, ...]:
    """Return topology bonds with source endpoint pairs taking precedence."""

    merged: list[TopologyBond] = []
    seen_endpoint_pairs = set()
    for bond in (*source_bonds, *generated_bonds):
        endpoint_pair = bond.endpoint_pair()
        if endpoint_pair in seen_endpoint_pairs:
            continue

        seen_endpoint_pairs.add(endpoint_pair)
        merged.append(bond)

    return tuple(merged)


def _topology_plan_for_new_hydrogens(
    *,
    residue_id: ResidueId,
    bond_definitions: tuple[BondDefinition, ...],
    added_atom_names: tuple[str, ...],
    provenance: BondProvenance,
) -> _RetainedNonPolymerHydrogenTopologyPlan | None:
    """Return retained ligand H topology plan for newly materialized hydrogens."""

    added_hydrogen_atom_names = frozenset(added_atom_names)
    filtered_bond_definitions = tuple(
        bond_definition
        for bond_definition in bond_definitions
        if bond_definition.atom_name_1 in added_hydrogen_atom_names
        or bond_definition.atom_name_2 in added_hydrogen_atom_names
    )
    if not filtered_bond_definitions:
        return None

    return _RetainedNonPolymerHydrogenTopologyPlan(
        residue_id=residue_id,
        bond_definitions=filtered_bond_definitions,
        provenance=provenance,
    )


def _apply_retained_non_polymer_hydrogen_directive(
    payload: CompletionResiduePayload,
    *,
    template: ResidueTemplate,
    placement_directive: StaticHydrogenPlacementDirective
    | RigidHydrogenPlacementDirective,
) -> CompletionResiduePayload | None:
    """Return one hydrogenated retained non-polymer payload."""

    environment = HydrogenCompletionEnvironment.from_payloads(
        (payload,),
        templates=(template,),
    )
    if isinstance(placement_directive, StaticHydrogenPlacementDirective):
        site = HydrogenResidueSite(
            residue_index=ResidueIndex(0),
            template=template,
            environment=environment,
        )
        patch = OrderedAtomPatch.from_residue_payload(
            payload.residue_site,
            residue_geometry=payload.residue_geometry,
        )
        return payload.apply_patch(
            generate_hydrogen_patch(
                site=site,
                patch=patch,
                semantics=placement_directive.semantics,
            )
        )

    semantics = placement_directive.semantics
    if not isinstance(semantics, IdealGeometryHydrogenSemantics):
        raise TypeError(
            "retained non-polymer rigid hydrogen placement requires "
            "ideal-geometry semantics"
        )

    patch = generate_component_hydrogen_patch(
        residue=payload,
        patch=OrderedAtomPatch.from_residue_payload(
            payload.residue_site,
            residue_geometry=payload.residue_geometry,
        ),
        semantics=semantics,
        skip_hydrogens_with_absent_anchors=True,
    )
    if patch is None:
        return None

    return payload.apply_patch(patch)


def _retained_non_polymer_template_atom_order(
    payload: CompletionResiduePayload,
    *,
    template: ResidueTemplate,
) -> tuple[str, ...]:
    """Return template atom order projected onto the materialized retained surface."""

    payload_atom_names = payload.atom_names()
    template_order = (
        *template.ordered_atom_names(),
        *template.expected_hydrogen_atom_names(),
    )
    ordered_atom_names = [
        atom_name for atom_name in template_order if atom_name in payload_atom_names
    ]
    ordered_atom_names.extend(
        atom_name
        for atom_name in payload_atom_names
        if atom_name not in ordered_atom_names
    )
    return tuple(ordered_atom_names)


def _payload_for_structure(
    structure: ProteinStructure,
    residue_id: ResidueId,
) -> CompletionResiduePayload:
    """Return the current retained non-polymer payload from one structure."""

    residue_index = structure.constitution.residue_index(residue_id)
    return CompletionResiduePayload(
        residue_site=structure.constitution.residue_site_at(residue_index),
        residue_geometry=structure.residue_geometry(residue_index),
        formal_charge_by_atom_name=(
            structure.residue_formal_charge_by_atom_name(residue_index)
        ),
    )


def _rdkit_hydrogen_append_patch(
    payload: CompletionResiduePayload,
    *,
    hydrogenated_molecule,
) -> OrderedAtomPatch:
    """Return one patch that appends RDKit-generated hydrogens."""

    conformer = hydrogenated_molecule.GetConformer()
    hydrogen_positions: list[Vec3] = []
    for atom in hydrogenated_molecule.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue

        coordinates = conformer.GetAtomPosition(atom.GetIdx())
        hydrogen_positions.append(
            Vec3(
                float(coordinates.x),
                float(coordinates.y),
                float(coordinates.z),
            )
        )

    hydrogen_atom_names = tuple(
        f"H{index:03d}" for index in range(1, len(hydrogen_positions) + 1)
    )
    return OrderedAtomPatch.from_residue_payload(
        payload.residue_site,
        residue_geometry=payload.residue_geometry,
    ).append_atoms(
        hydrogen_atom_names,
        hydrogen_positions,
    )


def _retained_non_polymer_hydrogen_fallback_issue(
    residue_site: ResidueSite,
    *,
    component_library: ComponentLibrary,
    error: Exception,
) -> ValidationIssue:
    """Return one validation issue for a failed retained non-polymer fallback."""

    diagnosis = diagnose_component_support(
        residue_site.component_id,
        component_library,
    )
    if component_library.get(residue_site.component_id) is None:
        return missing_component_definition_issue(
            residue_site,
            diagnosis=diagnosis,
            action=(
                "RDKit fallback hydrogenation failed "
                f"({type(error).__name__}); leaving retained non-polymer unchanged"
            ),
        )

    return ValidationIssue.for_residue(
        kind=ValidationIssueKind.UNSUPPORTED_HYDROGENATION,
        severity=IssueSeverity.WARNING,
        message=(
            f"{residue_site.residue_id.display_token()} could not materialize "
            "retained non-polymer hydrogens via template or RDKit fallback "
            f"({type(error).__name__}); leaving residue unchanged"
        ),
        residue_id=residue_site.residue_id,
    )
