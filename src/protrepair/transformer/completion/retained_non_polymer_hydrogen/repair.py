"""Hydrogen completion over retained non-polymer components."""

from collections import Counter, deque
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from itertools import chain

from protrepair.chemistry import (
    ComponentLibrary,
    IdealGeometryHydrogenSemantics,
    ResidueTemplate,
    build_default_component_library,
)
from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.component.topology import (
    template_heavy_bond_definitions_for_present_atoms,
    template_hydrogen_bond_definitions_for_names,
)
from protrepair.chemistry.inference.retained_non_polymer_evidence import (
    retained_non_polymer_evidence_formal_charge_requirements,
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
    AtomTopology,
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
    is_covalent_like_relationship,
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
    rdkit_patch,
)
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload
from protrepair.transformer.completion.shared.patch import OrderedAtomPatch
from protrepair.transformer.result import TransformationResult

from .rdkit_evidence import (
    hydrogenate_retained_non_polymer_payload_with_evidence_result as hydrogenate,
)

_RDKIT_FALLBACK_VISIBILITY_DETAILS = (
    "retained non-polymer hydrogens inferred via RDKit coordinate/proximity fallback"
)
_SOURCE_HYDROGEN_RECONCILIATION_CANDIDATE_LIMIT = 4096
_SOURCE_HYDROGEN_ANCHOR_DISTANCE_TOLERANCE_ANGSTROM = 0.35


@dataclass(frozen=True, slots=True)
class _RetainedNonPolymerHydrogenPayloadResult:
    """Admitted residue mutation, or no payload when the source must stay intact."""

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
    formal_charge_requirements: tuple[tuple[str, int], ...] = ()

    def requires_unlinked_scaffold(self, heavy_atom_names: Collection[str]) -> bool:
        """An admitted zero-H graph cannot acquire extra covalent bonds."""
        heavy_names = frozenset(heavy_atom_names)
        return bool(self.formal_charge_requirements) and all(
            bond.atom_name_1 in heavy_names and bond.atom_name_2 in heavy_names
            for bond in self.bond_definitions
        )


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

    def unpreserved_source_hydrogen_names(
        self,
        source: CompletionResiduePayload,
    ) -> tuple[str, ...]:
        """Return source H identities whose complete atom payload was not retained."""

        return tuple(
            atom.name
            for atom in source.residue_site.atom_sites
            if atom.is_hydrogen()
            and (
                not self.payload.has_atom(atom.name)
                or self.payload.atom_site(atom.name) != atom
                or self.payload.atom_geometry(atom.name)
                != source.atom_geometry(atom.name)
                or self.payload.formal_charge(atom.name)
                != source.formal_charge(atom.name)
            )
        )

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

    return _execute_retained_non_polymer_hydrogen_stage(
        structure,
        component_library=library,
        target_residue_ids=active_target_residue_ids,
        chemistry_evidence_by_residue_id=evidence_by_residue_id(chemistry_evidence),
        allow_retained_non_polymer_rdkit_fallback=(
            allow_retained_non_polymer_rdkit_fallback
        ),
    )


def _execute_retained_non_polymer_hydrogen_stage(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
    target_residue_ids: frozenset[ResidueId],
    chemistry_evidence_by_residue_id: Mapping[
        ResidueId,
        RetainedNonPolymerChemistryEvidence,
    ],
    allow_retained_non_polymer_rdkit_fallback: bool,
    topology_only: bool = False,
) -> TransformationResult:
    """Validate retained chemistry, then publish atoms or only their graph."""

    disulfide_residue_ids = disulfide_bonded_cysteine_residue_ids(structure)
    hydrogenated_payloads: list[CompletionResiduePayload] = []
    topology_plans: list[_RetainedNonPolymerTopologyPlan] = []
    repairs: list[RepairEvent] = []
    issues: list[ValidationIssue] = []
    source_bonds_by_residue = _source_bonds_by_residue(structure)
    for ligand in structure.constitution.ligands:
        if ligand.residue_id not in target_residue_ids:
            continue

        source_payload = _payload_for_structure(structure, ligand.residue_id)
        source_hydrogen_anchor_by_name = (
            structure.topology.residue_hydrogen_anchor_by_name(
                constitution=structure.constitution,
                residue_index=structure.constitution.residue_index(ligand.residue_id),
            )
        )
        source_explicit_hydrogen_anchor_by_name = (
            structure.topology.residue_hydrogen_anchor_by_name(
                constitution=structure.constitution,
                residue_index=structure.constitution.residue_index(ligand.residue_id),
                provenance=BondProvenance.SOURCE_EXPLICIT,
            )
        )
        try:
            result = _hydrogenate_retained_non_polymer_payload(
                source_payload.without_atom_sites(_hydrogen_atom_names(source_payload)),
                component_library=component_library,
                chemistry_evidence=chemistry_evidence_by_residue_id.get(
                    ligand.residue_id
                ),
                source_payload=source_payload,
                source_hydrogen_anchor_by_name=source_hydrogen_anchor_by_name,
                source_explicit_hydrogen_anchor_by_name=(
                    source_explicit_hydrogen_anchor_by_name
                ),
                is_disulfide_bonded=ligand.residue_id in disulfide_residue_ids,
                allow_retained_non_polymer_rdkit_fallback=(
                    allow_retained_non_polymer_rdkit_fallback
                ),
            )
        except _SourceHydrogenReconciliationError as error:
            issues.append(
                ValidationIssue.for_residue(
                    kind=ValidationIssueKind.UNSUPPORTED_HYDROGENATION,
                    severity=IssueSeverity.WARNING,
                    residue_id=ligand.residue_id,
                    component_id=ligand.component_id,
                    atom_names=error.atom_names,
                    message=f"Retained source H reconciliation failed ({error}); "
                    "leaving residue unchanged",
                )
            )
            continue
        if result.payload is not None and result.topology_plan is not None:
            try:
                _require_source_compatible_topology_plan(
                    structure,
                    result.topology_plan,
                    source_bonds=source_bonds_by_residue.get(ligand.residue_id, ()),
                )
            except ValueError as error:
                issues.extend(result.issues)
                issues.append(_retained_chemistry_conflict_issue(ligand, error))
                continue
        if topology_only and result.payload is not None:
            if not _preserves_atom_payload(source_payload, result.payload):
                issues.extend(result.issues)
                issues.append(
                    _retained_chemistry_conflict_issue(
                        ligand,
                        ValueError(
                            "topology-only resolution would change atom payload"
                        ),
                    )
                )
                continue
        elif result.payload is not None and not _preserves_atom_payload(
            source_payload, result.payload
        ):
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
    if topology_plans:
        updated_structure = _with_retained_non_polymer_topology_bonds(
            topology_source_structure=structure,
            target_structure=updated_structure,
            topology_plans=tuple(topology_plans),
        )
    changed_payload_ids = {payload.residue_id for payload in hydrogenated_payloads}
    graph_only_residue_ids = tuple(
        plan.residue_id
        for plan in topology_plans
        if plan.residue_id not in changed_payload_ids
    )
    if graph_only_residue_ids and updated_structure is not structure:
        repairs.extend(
            _retained_topology_resolution_repairs(
                structure,
                updated_structure,
                residue_ids=graph_only_residue_ids,
            )
        )
    if hydrogenated_payloads:
        updated_structure, cleanup_issues = cleanup_residual_hydrogen_clashes(
            updated_structure,
            component_library=component_library,
            target_residue_ids=frozenset(
                payload.residue_id for payload in hydrogenated_payloads
            ),
        )
        issues.extend(cleanup_issues)

    return TransformationResult(
        structure=updated_structure,
        repairs=tuple(repairs),
        issues=tuple(issues),
    )


def _retained_topology_resolution_repairs(
    source: ProteinStructure,
    target: ProteinStructure,
    *,
    residue_ids: tuple[ResidueId, ...],
) -> tuple[RepairEvent, ...]:
    """Attribute graph/charge-only edits across neighboring inventory changes."""

    previous_bonds = (
        source.topology.bonds
        if source.constitution is target.constitution
        else source.topology.bonds_for_constitution(
            source_constitution=source.constitution,
            target_constitution=target.constitution,
        )
    )
    previous_by_pair = {bond.endpoint_pair(): bond for bond in previous_bonds}
    changed_names_by_residue: dict[ResidueId, set[str]] = {
        residue_id: set() for residue_id in residue_ids
    }
    for bond in chain(previous_bonds, target.topology.bonds):
        pair = bond.endpoint_pair()
        if previous_by_pair.get(pair) == target.topology.bond_between(*pair):
            continue
        for index in pair:
            reference = target.constitution.atom_ref_at(index)
            names = changed_names_by_residue.get(reference.residue_id)
            if names is not None:
                names.add(reference.atom_name)

    for residue_id, names in changed_names_by_residue.items():
        source_index = source.constitution.residue_index(residue_id)
        target_index = target.constitution.residue_index(residue_id)
        for atom in source.constitution.residue_site_at(source_index).atom_sites:
            previous = source.constitution.atom_index_in_residue(
                source_index, atom.name
            )
            current = target.constitution.atom_index_in_residue(target_index, atom.name)
            if source.topology.formal_charge(previous) != target.topology.formal_charge(
                current
            ):
                names.add(atom.name)

    return tuple(
        RepairEvent.for_residue(
            kind=RepairEventKind.RETAINED_NON_POLYMER_TOPOLOGY_RESOLVED,
            residue_id=residue_id,
            component_id=target.constitution.residue_site_at(
                target.constitution.residue_index(residue_id)
            ).component_id,
            atom_names=tuple(sorted(names)),
        )
        for residue_id, names in changed_names_by_residue.items()
        if names
    )


def resolve_retained_non_polymer_topology(
    structure: ProteinStructure,
    component_library: ComponentLibrary,
    *,
    target_residue_ids: frozenset[ResidueId] | None = None,
    chemistry_evidence: tuple[RetainedNonPolymerChemistryEvidence, ...] = (),
    allow_retained_non_polymer_rdkit_fallback: bool = True,
) -> TransformationResult:
    """Publish admitted bonds and missing charges without replacing atoms.

    Parameters
    ----------
    structure : ProteinStructure
        Structure whose atom identities, geometry and reported charges are preserved.
    component_library : ComponentLibrary
        Active component definitions.
    target_residue_ids : frozenset[ResidueId] or None
        Selected retained residues; None selects all retained residues.
    chemistry_evidence : tuple[RetainedNonPolymerChemistryEvidence, ...]
        Explicit chemistry overrides, validated against the current pose.
    allow_retained_non_polymer_rdkit_fallback : bool, default=True
        Whether coordinate-based inference is allowed without template/evidence.

    Returns
    -------
    TransformationResult
        Topology-only changes, including previously unspecified ionic charges.
        Failed or atom-changing proposals leave their
        residue unchanged and report a diagnostic. Placement is used to validate
        source H correspondence, never to replace its coordinates in this path.
    """
    return _execute_retained_non_polymer_hydrogen_stage(
        structure,
        component_library=component_library,
        target_residue_ids=(
            frozenset(ligand.residue_id for ligand in structure.constitution.ligands)
            if target_residue_ids is None
            else target_residue_ids
        ),
        chemistry_evidence_by_residue_id=evidence_by_residue_id(chemistry_evidence),
        allow_retained_non_polymer_rdkit_fallback=allow_retained_non_polymer_rdkit_fallback,
        topology_only=True,
    )


def _preserves_atom_payload(
    source: CompletionResiduePayload, candidate: CompletionResiduePayload
) -> bool:
    return set(source.atom_names()) == set(candidate.atom_names()) and all(
        candidate.atom_site(name) == source.atom_site(name)
        and candidate.atom_geometry(name) == source.atom_geometry(name)
        and candidate.formal_charge(name) == source.formal_charge(name)
        for name in source.atom_names()
    )


def _hydrogenate_retained_non_polymer_payload(
    payload: CompletionResiduePayload,
    *,
    component_library: ComponentLibrary,
    chemistry_evidence: RetainedNonPolymerChemistryEvidence | None = None,
    source_payload: CompletionResiduePayload | None = None,
    source_hydrogen_anchor_by_name: Mapping[str, str] | None = None,
    source_explicit_hydrogen_anchor_by_name: Mapping[str, str] | None = None,
    is_disulfide_bonded: bool,
    allow_retained_non_polymer_rdkit_fallback: bool = True,
) -> _RetainedNonPolymerHydrogenPayloadResult:
    """Hydrogenate one retained non-polymer residue when supported."""

    if chemistry_evidence is not None:
        try:
            hydrogenation_result = hydrogenate(
                payload,
                evidence=chemistry_evidence,
            )
        except (RuntimeError, ValueError) as error:
            return _RetainedNonPolymerHydrogenPayloadResult(
                payload=None,
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
            hydrogen_bond_definitions=(hydrogenation_result.hydrogen_bond_definitions),
        )
        if is_disulfide_bonded:
            projection = projection.without_hydrogens_anchored_to(frozenset(("SG",)))
        hydrogenated_payload = projection.payload
        reconciliation = _reconcile_source_hydrogens(
            hydrogenated_payload,
            source_payload=source_payload,
            source_hydrogen_anchor_by_name=source_hydrogen_anchor_by_name,
            source_explicit_hydrogen_anchor_by_name=(
                source_explicit_hydrogen_anchor_by_name
            ),
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
                formal_charge_requirements=retained_non_polymer_evidence_formal_charge_requirements(
                    chemistry_evidence
                ),
            ),
        )

    template = component_library.get(payload.component_id)
    if template is not None:
        if not template.has_compatible_heavy_scaffold(payload.residue_site):
            return _RetainedNonPolymerHydrogenPayloadResult(
                payload=None,
                repairs=(),
                issues=(
                    _retained_chemistry_conflict_issue(
                        payload.residue_site,
                        ValueError(
                            "template requires compatible present heavy names "
                            "and elements"
                        ),
                    ),
                ),
            )

        try:
            _require_compatible_formal_charges(
                dict(payload.formal_charge_by_atom_name),
                template.heavy_atom_charge_requirements(),
            )
        except ValueError as error:
            return _RetainedNonPolymerHydrogenPayloadResult(
                payload=None,
                repairs=(),
                issues=(
                    _retained_chemistry_conflict_issue(payload.residue_site, error),
                ),
            )
        charge_requirements = template.applicable_heavy_atom_charge_requirements(
            payload.residue_site
        )
        if (
            template.can_add_hydrogens()
            and not template.expected_hydrogen_atom_names()
            and source_payload is not None
            and _hydrogen_atom_names(source_payload)
        ):
            raise _SourceHydrogenReconciliationError(
                "source H conflicts with the selected zero-H template",
                atom_names=_hydrogen_atom_names(source_payload),
            )
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
                    payload=None,
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
            missing_heavy_atom_names = tuple(
                atom_name
                for atom_name in template.expected_heavy_atom_names()
                if not payload.has_atom(atom_name)
            )
            try:
                reconciliation = _reconcile_source_hydrogens(
                    ordered_payload,
                    source_payload=source_payload,
                    source_hydrogen_anchor_by_name=source_hydrogen_anchor_by_name,
                    source_explicit_hydrogen_anchor_by_name=(
                        source_explicit_hydrogen_anchor_by_name
                    ),
                    generated_hydrogen_anchor_by_name=_hydrogen_anchor_by_name(
                        template_hydrogen_bond_definitions,
                        ordered_payload,
                    ),
                )
            except _SourceHydrogenReconciliationError as error:
                if not missing_heavy_atom_names:
                    raise
                raise _SourceHydrogenReconciliationError(
                    "template-backed H reconciliation on an incomplete "
                    "heavy-atom scaffold "
                    f"(missing {', '.join(missing_heavy_atom_names)}) failed: {error}",
                    atom_names=error.atom_names,
                ) from error
            if missing_heavy_atom_names and source_payload is not None:
                unpreserved_names = reconciliation.unpreserved_source_hydrogen_names(
                    source_payload
                )
                if unpreserved_names:
                    return _RetainedNonPolymerHydrogenPayloadResult(
                        payload=None,
                        repairs=(),
                        issues=contradiction_issues
                        + (
                            ValidationIssue.for_residue(
                                kind=ValidationIssueKind.UNSUPPORTED_HYDROGENATION,
                                severity=IssueSeverity.WARNING,
                                message=(
                                    f"{payload.residue_id.display_token()} "
                                    f"{payload.component_id} template-backed hydrogen "
                                    "placement cannot reconcile all source hydrogens "
                                    "with an incomplete heavy-atom scaffold "
                                    f"(missing {', '.join(missing_heavy_atom_names)}); "
                                    "leaving residue unchanged"
                                ),
                                residue_id=payload.residue_id,
                                component_id=payload.component_id,
                                atom_names=unpreserved_names,
                            ),
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
                        template_heavy_bond_definitions_for_present_atoms(
                            payload.residue_site, template=template
                        )
                        + template_hydrogen_bond_definitions
                    ),
                    provenance=BondProvenance.TEMPLATE_RESOLVED,
                    formal_charge_requirements=charge_requirements,
                ),
            )

    if not allow_retained_non_polymer_rdkit_fallback:
        return _RetainedNonPolymerHydrogenPayloadResult(
            payload=None,
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
                source_hydrogen_anchor_by_name=source_hydrogen_anchor_by_name,
            )
        )
        hydrogenated_payload = hydrogenation_result.payload
    except (RuntimeError, ValueError) as error:
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
        source_explicit_hydrogen_anchor_by_name=(
            source_explicit_hydrogen_anchor_by_name
        ),
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
    source_hydrogen_anchor_by_name: Mapping[str, str] | None = None,
) -> _RdkitFallbackHydrogenationResult:
    """Hydrogenate one payload and return fallback-inferred H anchors."""

    fallback_inference_result = infer_retained_non_polymer_rdkit_fallback(
        payload.residue_site,
        payload.residue_geometry,
        formal_charge_by_atom_name=dict(payload.formal_charge_by_atom_name),
        present_hydrogen_atom_names=_hydrogen_atom_names(source_payload or payload),
        hydrogen_position_by_name=_hydrogen_position_by_name(source_payload),
        hydrogen_anchor_by_name=source_hydrogen_anchor_by_name,
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


class _SourceHydrogenReconciliationError(ValueError):
    """Source H cannot be preserved by a coherent bounded correspondence."""

    def __init__(self, message: str, *, atom_names: tuple[str, ...]) -> None:
        super().__init__(message)
        self.atom_names = atom_names


def _reconcile_source_hydrogens(
    hydrogenated_payload: CompletionResiduePayload,
    *,
    source_payload: CompletionResiduePayload | None,
    source_hydrogen_anchor_by_name: Mapping[str, str] | None = None,
    source_explicit_hydrogen_anchor_by_name: Mapping[str, str] | None = None,
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

    generated_name_by_source_name = _source_hydrogen_generated_name_assignment(
        hydrogenated_payload,
        source_payload=source_payload,
        source_hydrogen_atom_names=source_hydrogen_atom_names,
        generated_hydrogen_atom_names=generated_hydrogen_atom_names,
        source_hydrogen_anchor_by_name=source_hydrogen_anchor_by_name or {},
        source_explicit_hydrogen_anchor_by_name=(
            source_explicit_hydrogen_anchor_by_name or {}
        ),
        generated_hydrogen_anchor_by_name=generated_hydrogen_anchor_by_name or {},
    )
    if len(generated_name_by_source_name) != len(source_hydrogen_atom_names):
        raise _SourceHydrogenReconciliationError(
            "source connectivity or coordinates cannot be reconciled",
            atom_names=tuple(
                name
                for name in source_hydrogen_atom_names
                if name not in generated_name_by_source_name
            ),
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
    source_explicit_hydrogen_anchor_by_name: Mapping[str, str],
    generated_hydrogen_anchor_by_name: Mapping[str, str],
) -> Mapping[str, str]:
    """Match compatible H slots without promoting model hints to source authority."""

    candidates: list[tuple[float, int, int, str, str]] = []
    current_anchors = dict(source_hydrogen_anchor_by_name)
    current_anchors.update(source_explicit_hydrogen_anchor_by_name)
    observed_counts = Counter(
        current_anchors[name]
        for name in source_hydrogen_atom_names
        if name in current_anchors
    )
    generated_counts = Counter(generated_hydrogen_anchor_by_name.values())
    source_hydrogen_anchor_by_name = {
        name: parent
        for name, parent in current_anchors.items()
        if name in source_explicit_hydrogen_anchor_by_name
        or observed_counts[parent] <= generated_counts[parent]
    }
    generated_names_by_anchor: dict[str, list[tuple[int, str]]] = {}
    for generated_index, generated_name in enumerate(generated_hydrogen_atom_names):
        anchor = generated_hydrogen_anchor_by_name.get(generated_name)
        if anchor is not None:
            generated_names_by_anchor.setdefault(anchor, []).append(
                (generated_index, generated_name)
            )
    all_generated = tuple(enumerate(generated_hydrogen_atom_names))
    candidates_by_source = {
        source_name: (
            generated_names_by_anchor.get(
                source_hydrogen_anchor_by_name[source_name], ()
            )
            if source_name in source_hydrogen_anchor_by_name
            else all_generated
        )
        for source_name in source_hydrogen_atom_names
    }
    if sum(len(names) for names in candidates_by_source.values()) > (
        _SOURCE_HYDROGEN_RECONCILIATION_CANDIDATE_LIMIT
    ):
        raise _SourceHydrogenReconciliationError(
            "source hydrogen reconciliation exceeds the per-residue resource limit",
            atom_names=source_hydrogen_atom_names,
        )
    for source_index, source_name in enumerate(source_hydrogen_atom_names):
        source_position = source_payload.position(source_name)
        for generated_index, generated_name in candidates_by_source[source_name]:
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

    candidates.sort(
        key=lambda pair: (pair[3] not in source_hydrogen_anchor_by_name, *pair)
    )
    return _maximum_hydrogen_assignment(
        source_hydrogen_atom_names,
        tuple((source, generated) for _, _, _, source, generated in candidates),
    )


def _maximum_hydrogen_assignment(
    source_names: tuple[str, ...],
    candidate_pairs: tuple[tuple[str, str], ...],
) -> Mapping[str, str]:
    """Preserve complete greedy matches; otherwise augment without recursion."""

    assignment: dict[str, str] = {}
    source_by_generated: dict[str, str] = {}
    for source_name, generated_name in candidate_pairs:
        if source_name in assignment or generated_name in source_by_generated:
            continue

        assignment[source_name] = generated_name
        source_by_generated[generated_name] = source_name
        if len(assignment) == len(source_names):
            return assignment

    if len(assignment) == len(source_names):
        return assignment

    candidates_by_source: dict[str, list[str]] = {}
    for source_name, generated_name in candidate_pairs:
        candidates_by_source.setdefault(source_name, []).append(generated_name)

    for source_name in source_names:
        if source_name in assignment:
            continue
        pending = deque((source_name,))
        predecessor_by_generated: dict[str, str] = {}
        available_name: str | None = None
        while pending and available_name is None:
            current = pending.popleft()
            for generated_name in candidates_by_source.get(current, ()):
                if generated_name in predecessor_by_generated:
                    continue
                predecessor_by_generated[generated_name] = current
                occupant = source_by_generated.get(generated_name)
                if occupant is None:
                    available_name = generated_name
                    break
                pending.append(occupant)
        if available_name is None:
            continue

        while True:
            current = predecessor_by_generated[available_name]
            previous_name = assignment.get(current)
            assignment[current] = available_name
            source_by_generated[available_name] = current
            if previous_name is None:
                break
            available_name = previous_name

        if len(assignment) == len(source_names):
            break

    return assignment


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
    planned_residue_ids = {plan.residue_id for plan in topology_plans}
    retained_source_bonds = []
    for bond in source_bonds:
        residue_ids = {
            target_structure.constitution.atom_ref_at(index).residue_id
            for index in bond.endpoint_pair()
        }
        if (
            bond.provenance is not BondProvenance.SOURCE_EXPLICIT
            and len(residue_ids) == 1
            and residue_ids.issubset(planned_residue_ids)
            and (
                is_covalent_like_relationship(bond)
                or bond.relationship_type is BondRelationshipType.UNKNOWN
            )
        ):
            continue
        retained_source_bonds.append(bond)
    generated_bonds = tuple(
        topology_bond
        for plan in topology_plans
        for topology_bond in _retained_non_polymer_topology_bonds_for_plan(
            target_structure,
            plan,
        )
    )
    atom_topologies = list(target_structure.topology.atom_topologies)
    for plan in topology_plans:
        residue_index = target_structure.constitution.residue_index(plan.residue_id)
        for name, charge in plan.formal_charge_requirements:
            index = target_structure.constitution.atom_index_in_residue(
                residue_index, name
            )
            if (
                charge != 0 or plan.provenance is BondProvenance.EVIDENCE_RESOLVED
            ) and target_structure.topology.formal_charge(index) is None:
                atom_topologies[index.value] = AtomTopology(formal_charge=charge)
    topology = StructureTopology(
        constitution=target_structure.constitution,
        atom_topologies=tuple(atom_topologies),
        bonds=_merge_source_first_topology_bonds(
            tuple(retained_source_bonds),
            generated_bonds,
            target_structure=target_structure,
        ),
    )
    if topology == target_structure.topology:
        return target_structure
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
    """Publish admitted chemistry while preserving explicit endpoint evidence."""

    merged = {bond.endpoint_pair(): bond for bond in source_bonds}
    source_anchored_hydrogen_indices = _source_explicit_hydrogen_anchor_indices(
        source_bonds,
        target_structure=target_structure,
    )
    for bond in generated_bonds:
        endpoint_pair = bond.endpoint_pair()
        existing = merged.get(endpoint_pair)
        if existing is not None:
            merged[endpoint_pair] = (
                existing.with_resolved_chemistry(bond)
                if existing.provenance is BondProvenance.SOURCE_EXPLICIT
                else bond
            )
            continue
        if (
            bond.atom_index_1 in source_anchored_hydrogen_indices
            or bond.atom_index_2 in source_anchored_hydrogen_indices
        ):
            continue

        merged[endpoint_pair] = bond

    return tuple(merged.values())


def _source_bonds_by_residue(
    structure: ProteinStructure,
) -> dict[ResidueId, tuple[TopologyBond, ...]]:
    by_residue: dict[ResidueId, list[TopologyBond]] = {}
    for bond in structure.topology.bonds:
        if bond.provenance is not BondProvenance.SOURCE_EXPLICIT:
            continue
        for residue_id in {
            structure.constitution.atom_ref_at(index).residue_id
            for index in bond.endpoint_pair()
        }:
            by_residue.setdefault(residue_id, []).append(bond)
    return {key: tuple(value) for key, value in by_residue.items()}


def _require_source_compatible_topology_plan(
    structure: ProteinStructure,
    plan: _RetainedNonPolymerTopologyPlan,
    *,
    source_bonds: tuple[TopologyBond, ...],
) -> None:
    residue_index = structure.constitution.residue_index(plan.residue_id)
    _require_compatible_formal_charges(
        dict(structure.residue_formal_charge_by_atom_name(residue_index)),
        plan.formal_charge_requirements,
    )
    residue = structure.constitution.residue_site_at(residue_index)
    requires_unlinked_scaffold = plan.requires_unlinked_scaffold(
        tuple(atom.name for atom in residue.atom_sites if not atom.is_hydrogen())
    )
    definitions = {
        frozenset((bond.atom_name_1, bond.atom_name_2)): bond
        for bond in plan.bond_definitions
    }
    for source_bond in source_bonds:
        references = tuple(
            structure.constitution.atom_ref_at(index)
            for index in source_bond.endpoint_pair()
        )
        if any(ref.residue_id != plan.residue_id for ref in references):
            if requires_unlinked_scaffold and (
                is_covalent_like_relationship(source_bond)
                or source_bond.relationship_type is BondRelationshipType.UNKNOWN
            ):
                raise ValueError(
                    "zero-H chemistry cannot admit an external "
                    f"{source_bond.relationship_type.value} relationship"
                )
            # External source attachments cannot be redirected to local parents.
            if any(
                ref.residue_id == plan.residue_id
                and structure.constitution.atom_site_at(index).is_hydrogen()
                for ref, index in zip(
                    references, source_bond.endpoint_pair(), strict=True
                )
            ) and (
                is_covalent_like_relationship(source_bond)
                or source_bond.relationship_type is BondRelationshipType.UNKNOWN
            ):
                raise ValueError("source hydrogen has an external attachment")
            continue
        expected = definitions.get(frozenset(ref.atom_name for ref in references))
        if expected is None:
            if is_covalent_like_relationship(source_bond) or (
                source_bond.relationship_type is BondRelationshipType.UNKNOWN
            ):
                raise ValueError(
                    "source edge is absent from admitted chemistry: "
                    + "-".join(ref.atom_name for ref in references)
                )
            continue
        source_bond.with_resolved_chemistry(
            TopologyBond(
                *source_bond.endpoint_pair(),
                order=expected.order,
                aromatic=expected.aromatic,
                provenance=plan.provenance,
            )
        )


def _require_compatible_formal_charges(
    reported_charges: Mapping[str, int | None],
    requirements: tuple[tuple[str, int], ...],
) -> None:
    for name, charge in requirements:
        reported = reported_charges.get(name)
        if reported is not None and reported != charge:
            raise ValueError(
                f"{name} charge {reported:+d} conflicts with "
                f"admitted chemistry {charge:+d}"
            )


def _retained_chemistry_conflict_issue(
    residue: ResidueSite, error: ValueError
) -> ValidationIssue:
    return ValidationIssue.for_residue(
        kind=ValidationIssueKind.CHEMISTRY_CONTRADICTION,
        severity=IssueSeverity.WARNING,
        residue_id=residue.residue_id,
        component_id=residue.component_id,
        message=f"Retained chemistry preparation failed ({error}); "
        "leaving residue unchanged",
    )


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
        if bond.relationship_type not in {
            BondRelationshipType.COVALENT,
            BondRelationshipType.UNKNOWN,
        }:
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
    formal_charge_requirements: tuple[tuple[str, int], ...] = (),
) -> _RetainedNonPolymerTopologyPlan | None:
    """Return retained ligand topology plan for validated bond definitions."""

    if not bond_definitions and not formal_charge_requirements:
        return None

    return _RetainedNonPolymerTopologyPlan(
        residue_id=residue_id,
        bond_definitions=bond_definitions,
        provenance=provenance,
        formal_charge_requirements=formal_charge_requirements,
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
            frozenset((payload.residue_id,)) if is_disulfide_bonded else frozenset()
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
        except (GeometryPlacementError, KeyError):
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

    return f"{_RDKIT_FALLBACK_VISIBILITY_DETAILS} (RDKit {rdkit_backend_version})"


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
