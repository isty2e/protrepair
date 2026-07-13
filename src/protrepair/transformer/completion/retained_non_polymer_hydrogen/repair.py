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
    infer_retained_non_polymer_rdkit_fallback,
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
from protrepair.geometry import GeometryPlacementError, Vec3
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.disulfide import disulfide_bonded_cysteine_residue_ids
from protrepair.structure.labels import ResidueId
from protrepair.structure.slots import AtomIndex, ResidueIndex
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
    rdkit_patch,
)
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload
from protrepair.transformer.completion.shared.patch import OrderedAtomPatch
from protrepair.transformer.result import TransformationResult

_RDKIT_FALLBACK_VISIBILITY_DETAILS = (
    "retained non-polymer hydrogens inferred via RDKit "
    "coordinate/proximity fallback"
)
_SOURCE_HYDROGEN_RECONCILIATION_CANDIDATE_LIMIT = 4096
_SOURCE_HYDROGEN_ANCHOR_DISTANCE_TOLERANCE_ANGSTROM = 0.35


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
    topology_plan: "_RetainedNonPolymerTopologyPlan | None" = None


@dataclass(frozen=True, slots=True)
class _RetainedNonPolymerTopologyPlan:
    """Residue-local retained ligand topology to apply after rebuild."""

    residue_id: ResidueId
    bond_definitions: tuple[BondDefinition, ...]
    provenance: BondProvenance


@dataclass(frozen=True, slots=True)
class _RdkitFallbackHydrogenationResult:
    """Fallback hydrogenated payload plus RDKit-derived topology bonds."""

    payload: CompletionResiduePayload
    rdkit_backend_version: str
    heavy_bond_definitions: tuple[BondDefinition, ...]
    hydrogen_bond_definitions: tuple[BondDefinition, ...]


@dataclass(frozen=True, slots=True)
class _HydrogenatedPayloadProjection:
    """Hydrogenated payload aligned with its residue-local hydrogen bonds."""

    payload: CompletionResiduePayload
    hydrogen_bond_definitions: tuple[BondDefinition, ...]

    def without_hydrogens_anchored_to(
        self,
        anchor_atom_names: frozenset[str],
    ) -> "_HydrogenatedPayloadProjection":
        """Return this projection without H atoms on selected heavy anchors."""

        excluded_hydrogen_names = {
            bonded_atom_name
            for bond_definition in self.hydrogen_bond_definitions
            for anchor_atom_name, bonded_atom_name in (
                (bond_definition.atom_name_1, bond_definition.atom_name_2),
                (bond_definition.atom_name_2, bond_definition.atom_name_1),
            )
            if anchor_atom_name in anchor_atom_names
        }
        if not excluded_hydrogen_names:
            return self

        return type(self)(
            payload=self.payload.without_atom_sites(excluded_hydrogen_names),
            hydrogen_bond_definitions=tuple(
                bond_definition
                for bond_definition in self.hydrogen_bond_definitions
                if bond_definition.atom_name_1 not in excluded_hydrogen_names
                and bond_definition.atom_name_2 not in excluded_hydrogen_names
            ),
        )


@dataclass(frozen=True, slots=True)
class _SourceHydrogenReconciliationResult:
    """Source-preserved payload plus generated-to-final atom-name projection."""

    payload: CompletionResiduePayload
    atom_name_projection: tuple[tuple[str, str], ...] = ()

    def project_bond_definitions(
        self,
        bond_definitions: tuple[BondDefinition, ...],
    ) -> tuple[BondDefinition, ...]:
        """Return bond definitions projected onto final atom names."""

        if not self.atom_name_projection:
            return bond_definitions

        atom_name_projection = dict(self.atom_name_projection)
        return tuple(
            BondDefinition(
                atom_name_1=atom_name_projection.get(
                    bond_definition.atom_name_1,
                    bond_definition.atom_name_1,
                ),
                atom_name_2=atom_name_projection.get(
                    bond_definition.atom_name_2,
                    bond_definition.atom_name_2,
                ),
                order=bond_definition.order,
                aromatic=bond_definition.aromatic,
            )
            for bond_definition in bond_definitions
        )


def add_retained_non_polymer_hydrogens(
    structure: ProteinStructure,
    component_library: ComponentLibrary | None = None,
    *,
    target_residue_ids: frozenset[ResidueId] | None = None,
    chemistry_evidence: tuple[RetainedNonPolymerChemistryEvidence, ...] = (),
    allow_retained_non_polymer_rdkit_fallback: bool = True,
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
        allow_retained_non_polymer_rdkit_fallback=(
            allow_retained_non_polymer_rdkit_fallback
        ),
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
    allow_retained_non_polymer_rdkit_fallback: bool,
) -> _RetainedNonPolymerHydrogenStageResult:
    """Apply hydrogen patches to targeted retained non-polymer components."""

    disulfide_residue_ids = disulfide_bonded_cysteine_residue_ids(
        topology_source_structure
    )
    hydrogenated_payloads: list[CompletionResiduePayload] = []
    topology_plans: list[_RetainedNonPolymerTopologyPlan] = []
    repairs: list[RepairEvent] = []
    issues: list[ValidationIssue] = []
    for ligand in structure.constitution.ligands:
        if ligand.residue_id not in target_residue_ids:
            continue

        result = _hydrogenate_retained_non_polymer_payload(
            _payload_for_structure(structure, ligand.residue_id),
            component_library=component_library,
            chemistry_evidence=chemistry_evidence_by_residue_id.get(ligand.residue_id),
            source_payload=_payload_for_structure(
                topology_source_structure,
                ligand.residue_id,
            ),
            source_hydrogen_anchor_by_name=(
                _source_explicit_hydrogen_anchor_by_name(
                    topology_source_structure,
                    ligand.residue_id,
                )
            ),
            is_disulfide_bonded=ligand.residue_id in disulfide_residue_ids,
            allow_retained_non_polymer_rdkit_fallback=(
                allow_retained_non_polymer_rdkit_fallback
            ),
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
        updated_structure = _with_retained_non_polymer_topology_bonds(
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
    source_payload: CompletionResiduePayload | None = None,
    source_hydrogen_anchor_by_name: Mapping[str, str] | None = None,
    is_disulfide_bonded: bool,
    allow_retained_non_polymer_rdkit_fallback: bool = True,
) -> _RetainedNonPolymerHydrogenPayloadResult:
    """Hydrogenate one retained non-polymer residue when supported."""

    if chemistry_evidence is not None:
        try:
            hydrogenation_result = (
                rdkit_evidence.hydrogenate_retained_non_polymer_payload_with_evidence_result(
                    payload,
                    evidence=chemistry_evidence,
                )
            )
        except (RuntimeError, ValueError) as error:
            return _RetainedNonPolymerHydrogenPayloadResult(
                payload=source_payload or payload,
                repairs=(),
                issues=(
                    _retained_non_polymer_evidence_hydrogenation_issue(
                        payload.residue_site,
                        error=error,
                    ),
                ),
            )

        projection = _HydrogenatedPayloadProjection(
            payload=hydrogenation_result.payload,
            hydrogen_bond_definitions=(
                hydrogenation_result.hydrogen_bond_definitions
            ),
        )
        if is_disulfide_bonded:
            projection = projection.without_hydrogens_anchored_to(
                frozenset(("SG",))
            )
        hydrogenated_payload = projection.payload
        reconciliation = _reconcile_source_hydrogens(
            hydrogenated_payload,
            source_payload=source_payload,
            source_hydrogen_anchor_by_name=source_hydrogen_anchor_by_name,
            generated_hydrogen_anchor_by_name=_hydrogen_anchor_by_name(
                projection.hydrogen_bond_definitions,
                hydrogenated_payload,
            ),
        )
        hydrogenated_payload = reconciliation.payload
        added_atom_names = tuple(
            atom_name
            for atom_name in hydrogenated_payload.atom_names()
            if atom_name not in (source_payload or payload).atom_names()
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
            topology_plan=_topology_plan_for_bond_definitions(
                residue_id=payload.residue_id,
                bond_definitions=reconciliation.project_bond_definitions(
                    hydrogenation_result.heavy_bond_definitions
                    + projection.hydrogen_bond_definitions,
                ),
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
                is_disulfide_bonded=is_disulfide_bonded,
            )
            if hydrogenated_payload is None:
                return _RetainedNonPolymerHydrogenPayloadResult(
                    payload=source_payload or payload,
                    repairs=(),
                    issues=contradiction_issues
                    + (
                        _retained_non_polymer_template_hydrogen_placement_issue(
                            payload.residue_site,
                            atom_names=tuple(
                                atom_name
                                for atom_name in template.expected_hydrogen_atom_names()
                                if not payload.has_atom(atom_name)
                            ),
                        ),
                    ),
                )

            ordered_payload = hydrogenated_payload.reordered(
                _retained_non_polymer_template_atom_order(
                    hydrogenated_payload,
                    template=template,
                )
            )
            generated_hydrogen_atom_names = _hydrogen_atom_names(ordered_payload)
            template_hydrogen_bond_definitions = (
                template_hydrogen_bond_definitions_for_names(
                    template,
                    hydrogen_atom_names=generated_hydrogen_atom_names,
                )
            )
            reconciliation = _reconcile_source_hydrogens(
                ordered_payload,
                source_payload=source_payload,
                source_hydrogen_anchor_by_name=source_hydrogen_anchor_by_name,
                generated_hydrogen_anchor_by_name=_hydrogen_anchor_by_name(
                    template_hydrogen_bond_definitions,
                    ordered_payload,
                ),
            )
            ordered_payload = reconciliation.payload.reordered(
                _retained_non_polymer_template_atom_order(
                    reconciliation.payload,
                    template=template,
                )
            )
            added_atom_names = tuple(
                atom_name
                for atom_name in ordered_payload.atom_names()
                if atom_name not in (source_payload or payload).atom_names()
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
                topology_plan=_topology_plan_for_bond_definitions(
                    residue_id=payload.residue_id,
                    bond_definitions=reconciliation.project_bond_definitions(
                        template_hydrogen_bond_definitions
                    ),
                    provenance=BondProvenance.TEMPLATE_RESOLVED,
                ),
            )

    if not allow_retained_non_polymer_rdkit_fallback:
        return _RetainedNonPolymerHydrogenPayloadResult(
            payload=source_payload or payload,
            repairs=(),
            issues=(
                _retained_non_polymer_hydrogen_fallback_disabled_issue(
                    payload.residue_site
                ),
            ),
        )

    try:
        hydrogenation_result = (
            _hydrogenate_retained_non_polymer_payload_with_rdkit_fallback_result(
                payload,
                source_payload=source_payload,
            )
        )
        hydrogenated_payload = hydrogenation_result.payload
    except (RuntimeError, ValueError) as error:
        return _RetainedNonPolymerHydrogenPayloadResult(
            payload=source_payload or payload,
            repairs=(),
            issues=(
                _retained_non_polymer_hydrogen_fallback_issue(
                    payload.residue_site,
                    component_library=component_library,
                    error=error,
                ),
            ),
        )

    projection = _HydrogenatedPayloadProjection(
        payload=hydrogenated_payload,
        hydrogen_bond_definitions=hydrogenation_result.hydrogen_bond_definitions,
    )
    if is_disulfide_bonded:
        projection = projection.without_hydrogens_anchored_to(frozenset(("SG",)))
    ordered_payload = projection.payload
    reconciliation = _reconcile_source_hydrogens(
        ordered_payload,
        source_payload=source_payload,
        source_hydrogen_anchor_by_name=source_hydrogen_anchor_by_name,
        generated_hydrogen_anchor_by_name=_hydrogen_anchor_by_name(
            projection.hydrogen_bond_definitions,
            ordered_payload,
        ),
    )
    ordered_payload = reconciliation.payload
    added_atom_names = tuple(
        atom_name
        for atom_name in ordered_payload.atom_names()
        if atom_name not in (source_payload or payload).atom_names()
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
                    details=_rdkit_fallback_visibility_details(
                        hydrogenation_result.rdkit_backend_version
                    ),
                ),
            )
        ),
        issues=(
            _retained_non_polymer_hydrogen_fallback_visibility_issue(
                payload.residue_site,
                rdkit_backend_version=hydrogenation_result.rdkit_backend_version,
            ),
        ),
        topology_plan=_topology_plan_for_bond_definitions(
            residue_id=payload.residue_id,
            bond_definitions=reconciliation.project_bond_definitions(
                hydrogenation_result.heavy_bond_definitions
                + projection.hydrogen_bond_definitions,
            ),
            provenance=BondProvenance.REPAIR_INFERRED,
        ),
    )


def _retained_non_polymer_evidence_hydrogenation_issue(
    residue_site: ResidueSite,
    *,
    error: Exception,
) -> ValidationIssue:
    """Return one validation issue for failed explicit chemistry evidence."""

    reason = _retained_non_polymer_hydrogen_failure_reason(error)
    return ValidationIssue.for_residue(
        kind=ValidationIssueKind.CHEMISTRY_CONTRADICTION,
        severity=IssueSeverity.WARNING,
        message=(
            f"{residue_site.residue_id.display_token()} {residue_site.component_id} "
            "retained non-polymer chemistry evidence could not be projected "
            f"({reason}); leaving residue unchanged"
        ),
        residue_id=residue_site.residue_id,
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
    *,
    source_payload: CompletionResiduePayload | None = None,
) -> _RdkitFallbackHydrogenationResult:
    """Hydrogenate one payload and return fallback-inferred H anchors."""

    fallback_inference_result = infer_retained_non_polymer_rdkit_fallback(
        payload.residue_site,
        payload.residue_geometry,
        formal_charge_by_atom_name=dict(payload.formal_charge_by_atom_name),
        present_hydrogen_atom_names=_hydrogen_atom_names(source_payload or payload),
        hydrogen_position_by_name=_hydrogen_position_by_name(source_payload),
    )
    return _RdkitFallbackHydrogenationResult(
        payload=payload.apply_patch(
            rdkit_patch.rdkit_hydrogen_append_patch(
                payload,
                hydrogenated_molecule=fallback_inference_result.hydrogenated_molecule,
                hydrogen_atom_names=(
                    fallback_inference_result.projected_hydrogen_atom_names()
                ),
            )
        ),
        rdkit_backend_version=fallback_inference_result.rdkit_backend_version,
        heavy_bond_definitions=fallback_inference_result.heavy_bond_definitions,
        hydrogen_bond_definitions=fallback_inference_result.hydrogen_bond_definitions,
    )


def _hydrogen_position_by_name(
    payload: CompletionResiduePayload | None,
) -> Mapping[str, Vec3] | None:
    """Return present source H coordinates keyed by atom name when available."""

    if payload is None:
        return None

    return {
        atom_site.name: payload.position(atom_site.name)
        for atom_site in payload.residue_site.atom_sites
        if atom_site.is_hydrogen()
    }


def _reconcile_source_hydrogens(
    hydrogenated_payload: CompletionResiduePayload,
    *,
    source_payload: CompletionResiduePayload | None,
    source_hydrogen_anchor_by_name: Mapping[str, str] | None = None,
    generated_hydrogen_anchor_by_name: Mapping[str, str] | None = None,
) -> _SourceHydrogenReconciliationResult:
    """Return hydrogenated payload with compatible source H atoms preserved."""

    if source_payload is None:
        return _SourceHydrogenReconciliationResult(payload=hydrogenated_payload)

    source_hydrogen_atom_names = _hydrogen_atom_names(source_payload)
    generated_hydrogen_atom_names = _hydrogen_atom_names(hydrogenated_payload)
    if (
        not source_hydrogen_atom_names
        or not generated_hydrogen_atom_names
        or len(source_hydrogen_atom_names) > len(generated_hydrogen_atom_names)
    ):
        return _SourceHydrogenReconciliationResult(payload=hydrogenated_payload)

    if (
        len(source_hydrogen_atom_names) * len(generated_hydrogen_atom_names)
        > _SOURCE_HYDROGEN_RECONCILIATION_CANDIDATE_LIMIT
    ):
        return _SourceHydrogenReconciliationResult(payload=hydrogenated_payload)

    generated_name_by_source_name = _source_hydrogen_generated_name_assignment(
        hydrogenated_payload,
        source_payload=source_payload,
        source_hydrogen_atom_names=source_hydrogen_atom_names,
        generated_hydrogen_atom_names=generated_hydrogen_atom_names,
        source_hydrogen_anchor_by_name=source_hydrogen_anchor_by_name or {},
        generated_hydrogen_anchor_by_name=generated_hydrogen_anchor_by_name or {},
    )
    source_name_by_generated_name = {
        generated_name: source_name
        for source_name, generated_name in generated_name_by_source_name.items()
    }
    reconciled_payload = hydrogenated_payload.without_atom_sites(
        generated_hydrogen_atom_names
    )
    atom_name_projection: dict[str, str] = {}
    used_atom_names = set(reconciled_payload.atom_names())
    used_atom_names.update(source_hydrogen_atom_names)

    for generated_name in generated_hydrogen_atom_names:
        source_name = source_name_by_generated_name.get(generated_name)
        if source_name is not None:
            reconciled_payload = reconciled_payload.with_atom_payload(
                source_payload.atom_site(source_name),
                atom_geometry=source_payload.atom_geometry(source_name),
                formal_charge=source_payload.formal_charge(source_name),
            )
            atom_name_projection[generated_name] = source_name
            continue

        materialized_name = generated_name
        if materialized_name in used_atom_names:
            materialized_name = _next_generated_hydrogen_atom_name(used_atom_names)
        generated_atom_site = hydrogenated_payload.atom_site(generated_name)
        reconciled_payload = reconciled_payload.with_atom_payload(
            generated_atom_site.with_name(materialized_name),
            atom_geometry=hydrogenated_payload.atom_geometry(generated_name),
            formal_charge=hydrogenated_payload.formal_charge(generated_name),
        )
        used_atom_names.add(materialized_name)
        if materialized_name != generated_name:
            atom_name_projection[generated_name] = materialized_name

    return _SourceHydrogenReconciliationResult(
        payload=reconciled_payload,
        atom_name_projection=tuple(atom_name_projection.items()),
    )


def _source_hydrogen_generated_name_assignment(
    hydrogenated_payload: CompletionResiduePayload,
    *,
    source_payload: CompletionResiduePayload,
    source_hydrogen_atom_names: tuple[str, ...],
    generated_hydrogen_atom_names: tuple[str, ...],
    source_hydrogen_anchor_by_name: Mapping[str, str],
    generated_hydrogen_anchor_by_name: Mapping[str, str],
) -> Mapping[str, str]:
    """Assign source H names to generated H slots by nearest coordinates."""

    candidates: list[tuple[float, int, int, str, str]] = []
    for source_index, source_name in enumerate(source_hydrogen_atom_names):
        source_position = source_payload.position(source_name)
        for generated_index, generated_name in enumerate(generated_hydrogen_atom_names):
            if not _source_hydrogen_anchor_is_compatible(
                source_name,
                generated_name,
                source_hydrogen_anchor_by_name=source_hydrogen_anchor_by_name,
                generated_hydrogen_anchor_by_name=generated_hydrogen_anchor_by_name,
            ):
                continue
            if not _source_hydrogen_anchor_distance_is_compatible(
                source_name,
                generated_name,
                source_payload=source_payload,
                hydrogenated_payload=hydrogenated_payload,
                source_hydrogen_anchor_by_name=source_hydrogen_anchor_by_name,
                generated_hydrogen_anchor_by_name=generated_hydrogen_anchor_by_name,
            ):
                continue

            distance = source_position.distance_to(
                hydrogenated_payload.position(generated_name)
            )
            candidates.append(
                (
                    distance,
                    source_index,
                    generated_index,
                    source_name,
                    generated_name,
                )
            )

    candidates.sort()
    assignment: dict[str, str] = {}
    used_generated_names: set[str] = set()
    for _, _, _, source_name, generated_name in candidates:
        if source_name in assignment or generated_name in used_generated_names:
            continue

        assignment[source_name] = generated_name
        used_generated_names.add(generated_name)
        if len(assignment) == len(source_hydrogen_atom_names):
            break

    return assignment


def _source_hydrogen_anchor_is_compatible(
    source_name: str,
    generated_name: str,
    *,
    source_hydrogen_anchor_by_name: Mapping[str, str],
    generated_hydrogen_anchor_by_name: Mapping[str, str],
) -> bool:
    """Return whether source topology agrees with the generated H anchor."""

    source_anchor = source_hydrogen_anchor_by_name.get(source_name)
    if source_anchor is None:
        return True

    return generated_hydrogen_anchor_by_name.get(generated_name) == source_anchor


def _source_hydrogen_anchor_distance_is_compatible(
    source_name: str,
    generated_name: str,
    *,
    source_payload: CompletionResiduePayload,
    hydrogenated_payload: CompletionResiduePayload,
    source_hydrogen_anchor_by_name: Mapping[str, str],
    generated_hydrogen_anchor_by_name: Mapping[str, str],
) -> bool:
    """Return whether source and generated H-anchor distances agree."""

    generated_anchor = generated_hydrogen_anchor_by_name.get(generated_name)
    if generated_anchor is None:
        return True

    source_anchor = source_hydrogen_anchor_by_name.get(source_name, generated_anchor)
    if source_anchor not in source_payload.atom_names():
        return False

    source_anchor_distance = source_payload.position(source_name).distance_to(
        source_payload.position(source_anchor)
    )
    generated_anchor_distance = hydrogenated_payload.position(
        generated_name
    ).distance_to(hydrogenated_payload.position(generated_anchor))
    return (
        abs(source_anchor_distance - generated_anchor_distance)
        <= _SOURCE_HYDROGEN_ANCHOR_DISTANCE_TOLERANCE_ANGSTROM
    )


def _hydrogen_anchor_by_name(
    bond_definitions: tuple[BondDefinition, ...],
    payload: CompletionResiduePayload,
) -> Mapping[str, str]:
    """Return generated hydrogen anchors by residue-local H atom name."""

    hydrogen_atom_names = set(_hydrogen_atom_names(payload))
    anchor_by_hydrogen_name: dict[str, str] = {}
    for bond_definition in bond_definitions:
        if (
            bond_definition.atom_name_1 in hydrogen_atom_names
            and bond_definition.atom_name_2 not in hydrogen_atom_names
        ):
            anchor_by_hydrogen_name[bond_definition.atom_name_1] = (
                bond_definition.atom_name_2
            )
        elif (
            bond_definition.atom_name_2 in hydrogen_atom_names
            and bond_definition.atom_name_1 not in hydrogen_atom_names
        ):
            anchor_by_hydrogen_name[bond_definition.atom_name_2] = (
                bond_definition.atom_name_1
            )

    return anchor_by_hydrogen_name


def _source_explicit_hydrogen_anchor_by_name(
    structure: ProteinStructure,
    residue_id: ResidueId,
) -> Mapping[str, str]:
    """Return source-explicit retained-ligand H anchors by H atom name."""

    residue = structure.constitution.residue_or_ligand(residue_id)
    if residue is None:
        return {}

    residue_atom_names = set(residue.atom_site_names())
    anchor_by_hydrogen_name: dict[str, str] = {}
    for bond in structure.topology.bonds:
        if (
            bond.provenance is not BondProvenance.SOURCE_EXPLICIT
            or bond.relationship_type is not BondRelationshipType.COVALENT
        ):
            continue

        atom_ref_1 = structure.constitution.atom_ref_at(bond.atom_index_1)
        atom_ref_2 = structure.constitution.atom_ref_at(bond.atom_index_2)
        if atom_ref_1.residue_id != residue_id or atom_ref_2.residue_id != residue_id:
            continue
        if (
            atom_ref_1.atom_name not in residue_atom_names
            or atom_ref_2.atom_name not in residue_atom_names
        ):
            continue

        atom_site_1 = structure.constitution.atom_site_at(bond.atom_index_1)
        atom_site_2 = structure.constitution.atom_site_at(bond.atom_index_2)
        if atom_site_1.is_hydrogen() and not atom_site_2.is_hydrogen():
            anchor_by_hydrogen_name[atom_ref_1.atom_name] = atom_ref_2.atom_name
        elif atom_site_2.is_hydrogen() and not atom_site_1.is_hydrogen():
            anchor_by_hydrogen_name[atom_ref_2.atom_name] = atom_ref_1.atom_name

    return anchor_by_hydrogen_name


def _hydrogen_atom_names(payload: CompletionResiduePayload) -> tuple[str, ...]:
    """Return hydrogen atom names in payload order."""

    return tuple(
        atom_site.name
        for atom_site in payload.residue_site.atom_sites
        if atom_site.is_hydrogen()
    )


def _next_generated_hydrogen_atom_name(used_atom_names: set[str]) -> str:
    """Return the next deterministic generated H name not already used."""

    index = 1
    while True:
        atom_name = f"H{index:03d}"
        if atom_name not in used_atom_names:
            return atom_name
        index += 1


def _with_retained_non_polymer_topology_bonds(
    *,
    topology_source_structure: ProteinStructure,
    target_structure: ProteinStructure,
    topology_plans: tuple[_RetainedNonPolymerTopologyPlan, ...],
) -> ProteinStructure:
    """Return target structure with source-first retained ligand topology."""

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
        bonds=_merge_source_first_topology_bonds(
            source_bonds,
            generated_bonds,
            target_structure=target_structure,
        ),
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
    plan: _RetainedNonPolymerTopologyPlan,
) -> tuple[TopologyBond, ...]:
    """Project one residue-local retained-ligand plan into canonical atom slots."""

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
    *,
    target_structure: ProteinStructure,
) -> tuple[TopologyBond, ...]:
    """Return topology bonds with source endpoint pairs taking precedence."""

    merged: list[TopologyBond] = []
    seen_endpoint_pairs = set()
    source_anchored_hydrogen_indices = _source_explicit_hydrogen_anchor_indices(
        source_bonds,
        target_structure=target_structure,
    )
    for bond in source_bonds:
        endpoint_pair = bond.endpoint_pair()
        if endpoint_pair in seen_endpoint_pairs:
            continue

        seen_endpoint_pairs.add(endpoint_pair)
        merged.append(bond)

    for bond in generated_bonds:
        endpoint_pair = bond.endpoint_pair()
        if endpoint_pair in seen_endpoint_pairs:
            continue
        if (
            bond.atom_index_1 in source_anchored_hydrogen_indices
            or bond.atom_index_2 in source_anchored_hydrogen_indices
        ):
            continue

        seen_endpoint_pairs.add(endpoint_pair)
        merged.append(bond)

    return tuple(merged)


def _source_explicit_hydrogen_anchor_indices(
    source_bonds: tuple[TopologyBond, ...],
    *,
    target_structure: ProteinStructure,
) -> frozenset[AtomIndex]:
    """Return H atom indices that already have source-explicit heavy anchors."""

    source_anchored_hydrogen_indices: set[AtomIndex] = set()
    for bond in source_bonds:
        if bond.provenance is not BondProvenance.SOURCE_EXPLICIT:
            continue

        atom_site_1 = target_structure.constitution.atom_site_at(bond.atom_index_1)
        atom_site_2 = target_structure.constitution.atom_site_at(bond.atom_index_2)
        if atom_site_1.is_hydrogen() and not atom_site_2.is_hydrogen():
            source_anchored_hydrogen_indices.add(bond.atom_index_1)
        elif atom_site_2.is_hydrogen() and not atom_site_1.is_hydrogen():
            source_anchored_hydrogen_indices.add(bond.atom_index_2)

    return frozenset(source_anchored_hydrogen_indices)


def _topology_plan_for_bond_definitions(
    *,
    residue_id: ResidueId,
    bond_definitions: tuple[BondDefinition, ...],
    provenance: BondProvenance,
) -> _RetainedNonPolymerTopologyPlan | None:
    """Return retained ligand topology plan for validated bond definitions."""

    if not bond_definitions:
        return None

    return _RetainedNonPolymerTopologyPlan(
        residue_id=residue_id,
        bond_definitions=bond_definitions,
        provenance=provenance,
    )


def _apply_retained_non_polymer_hydrogen_directive(
    payload: CompletionResiduePayload,
    *,
    template: ResidueTemplate,
    placement_directive: StaticHydrogenPlacementDirective
    | RigidHydrogenPlacementDirective,
    is_disulfide_bonded: bool,
) -> CompletionResiduePayload | None:
    """Return one hydrogenated retained non-polymer payload."""

    environment = HydrogenCompletionEnvironment.from_payloads(
        (payload,),
        templates=(template,),
        disulfide_bonded_residue_ids=(
            frozenset((payload.residue_id,))
            if is_disulfide_bonded
            else frozenset()
        ),
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
        try:
            hydrogen_patch = generate_hydrogen_patch(
                site=site,
                patch=patch,
                semantics=placement_directive.semantics,
            )
        except GeometryPlacementError:
            return None

        return payload.apply_patch(hydrogen_patch)

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


def _retained_non_polymer_hydrogen_fallback_issue(
    residue_site: ResidueSite,
    *,
    component_library: ComponentLibrary,
    error: Exception,
) -> ValidationIssue:
    """Return one validation issue for a failed retained non-polymer fallback."""

    reason = _retained_non_polymer_hydrogen_failure_reason(error)
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
                f"({reason}); leaving retained non-polymer unchanged"
            ),
        )

    return ValidationIssue.for_residue(
        kind=ValidationIssueKind.UNSUPPORTED_HYDROGENATION,
        severity=IssueSeverity.WARNING,
        message=(
            f"{residue_site.residue_id.display_token()} could not materialize "
            "retained non-polymer hydrogens via template or RDKit fallback "
            f"({reason}); leaving residue unchanged"
        ),
        residue_id=residue_site.residue_id,
    )


def _retained_non_polymer_hydrogen_failure_reason(error: Exception) -> str:
    """Return stable user-facing retained-ligand hydrogenation failure wording."""

    if isinstance(error, ValueError):
        detail = str(error).strip()
        if detail:
            return (
                "chemistry evidence or inferred fallback chemistry is invalid: "
                f"{detail}"
            )

        return "chemistry evidence or inferred fallback chemistry is invalid"

    return "RDKit chemistry projection failed"


def _retained_non_polymer_hydrogen_fallback_visibility_issue(
    residue_site: ResidueSite,
    *,
    rdkit_backend_version: str,
) -> ValidationIssue:
    """Return one warning that RDKit fallback chemistry was used."""

    return ValidationIssue.for_residue(
        kind=ValidationIssueKind.RETAINED_NON_POLYMER_FALLBACK_USED,
        severity=IssueSeverity.WARNING,
        message=(
            f"{residue_site.residue_id.display_token()} "
            f"{residue_site.component_id} "
            f"{_rdkit_fallback_visibility_details(rdkit_backend_version)}; "
            "provide a template or explicit chemistry evidence for stricter "
            "hydrogenation"
        ),
        residue_id=residue_site.residue_id,
    )


def _rdkit_fallback_visibility_details(rdkit_backend_version: str) -> str:
    """Return stable user-facing RDKit fallback provenance details."""

    return (
        f"{_RDKIT_FALLBACK_VISIBILITY_DETAILS} "
        f"(RDKit {rdkit_backend_version})"
    )


def _retained_non_polymer_template_hydrogen_placement_issue(
    residue_site: ResidueSite,
    *,
    atom_names: tuple[str, ...],
) -> ValidationIssue:
    """Return one warning for failed template-backed ligand H placement."""

    return ValidationIssue.for_residue(
        kind=ValidationIssueKind.UNSUPPORTED_HYDROGENATION,
        severity=IssueSeverity.WARNING,
        message=(
            f"{residue_site.residue_id.display_token()} "
            f"{residue_site.component_id} template-backed hydrogen placement "
            "failed; leaving residue unchanged"
        ),
        residue_id=residue_site.residue_id,
        component_id=residue_site.component_id,
        atom_names=atom_names,
    )


def _retained_non_polymer_hydrogen_fallback_disabled_issue(
    residue_site: ResidueSite,
) -> ValidationIssue:
    """Return one warning that strict policy blocked RDKit fallback."""

    return ValidationIssue.for_residue(
        kind=ValidationIssueKind.RETAINED_NON_POLYMER_FALLBACK_BLOCKED,
        severity=IssueSeverity.WARNING,
        message=(
            f"{residue_site.residue_id.display_token()} "
            f"{residue_site.component_id} retained non-polymer hydrogens require "
            "RDKit coordinate/proximity fallback, but RDKit coordinate/proximity "
            "fallback is disabled; provide a template or explicit chemistry "
            "evidence for strict hydrogenation"
        ),
        residue_id=residue_site.residue_id,
    )
