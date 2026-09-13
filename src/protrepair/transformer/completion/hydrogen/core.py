"""Hydrogen materialization with one authority for coupled polymer chemistry."""

from collections import Counter

from protrepair.chemistry import (
    ComponentLibrary,
    IdealGeometryHydrogenSemantics,
    build_default_component_library,
)
from protrepair.chemistry.microstate.catalog import PolymerChemicalSite
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.preparation import (
    PolymerMicrostatePreparation,
    PolymerSitePreparation,
    pras_microstate_preferences,
)
from protrepair.chemistry.microstate.resolution import MicrostateResolutionStatus
from protrepair.diagnostics.component_support import (
    diagnose_component_support,
    missing_component_definition_issue,
    unsupported_hydrogenation_issue,
)
from protrepair.diagnostics.events import RepairEvent, ValidationIssue
from protrepair.diagnostics.kinds import (
    IssueSeverity,
    RepairEventKind,
    ValidationIssueKind,
)
from protrepair.geometry import GeometryPlacementError
from protrepair.state.structure_topology import StructureDisulfideHydrogenFacts
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.disulfide import disulfide_bonded_cysteine_residue_ids
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.slots import ResidueIndex
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.diagnostics import (
    skipped_geometry_placement_issue,
)
from protrepair.transformer.completion.hydrogen.component_patch import (
    generate_component_hydrogen_patch,
)
from protrepair.transformer.completion.hydrogen.domain import (
    HydrogenCompletionEnvironment,
    HydrogenResidueSite,
)
from protrepair.transformer.completion.hydrogen.microstate import (
    place_polymer_microstate_hydrogens,
)
from protrepair.transformer.completion.hydrogen.protonation import (
    DisabledHistidineProtonationRequest,
    HistidineProtonationRequest,
    histidine_microstate_requests,
    normalize_histidine_protonation_request,
)
from protrepair.transformer.completion.hydrogen.static_patch import (
    generate_hydrogen_patch,
)
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload
from protrepair.transformer.completion.shared.patch import OrderedAtomPatch
from protrepair.transformer.polymer_microstate import apply_polymer_microstate_patches
from protrepair.transformer.result import TransformationResult

_NO_NEW_HISTIDINE_REQUEST = DisabledHistidineProtonationRequest()


def materialize_hydrogens_core(
    structure: ProteinStructure,
    component_library: ComponentLibrary | None = None,
    *,
    source_isotope_structure: ProteinStructure | None = None,
    target_residue_ids: frozenset[ResidueId] | None = None,
    histidine_protonation: HistidineProtonationRequest = _NO_NEW_HISTIDINE_REQUEST,
    rebuild_existing: bool = False,
) -> TransformationResult:
    """Place fixed-template H and atomically realize resolved polymer sites.

    Parameters
    ----------
    structure : ProteinStructure
        Current heavy scaffold. Missing heavy atoms, including OXT, are diagnosed
        rather than manufactured by this hydrogen-only operation.
    component_library : ComponentLibrary or None
        Active component definitions; None selects the default library.
    source_isotope_structure : ProteinStructure or None
        Pre-repair geometry used to detect moved H anchors. Original chemical
        evidence comes from provenance, not this argument.
    target_residue_ids : frozenset[ResidueId] or None
        Polymer residues to change. None selects all polymer residues.
    histidine_protonation : HistidineProtonationRequest
        New ratio selection or no-new-request, normalized at this boundary.
    rebuild_existing : bool
        Regenerate H coordinates while preserving identities and scalars.
        Moved pre-repair anchors also trigger residue-local rebuilding.

    Returns
    -------
    TransformationResult
        Complete site graph edits and local diagnostics. Unresolved sites remain
        unchanged; unrelated successful sites are retained. No FF runs here.

    Raises
    ------
    TypeError
        A request or rebuilding flag is not a supported value.
    RdkitUnavailableError
        Required native H placement is unavailable.
    """
    if type(rebuild_existing) is not bool:
        raise TypeError("hydrogen rebuilding must be a bool")
    library = component_library or build_default_component_library()
    request = normalize_histidine_protonation_request(histidine_protonation)
    targets = frozenset(
        residue.residue_id
        for chain in structure.constitution.chains
        for residue in chain.residues
        if target_residue_ids is None or residue.residue_id in target_residue_ids
    )
    rebuilding = frozenset(
        residue_id
        for residue_id in targets
        if rebuild_existing
        or _heavy_anchors_changed(structure, source_isotope_structure, residue_id)
    )
    # Heavy completion may strip H. Restore identity/scalars first; the explicit
    # rebuilding set controls coordinate replacement before any FF can run.
    structure = _restore_preparation_hydrogens(
        structure, source_isotope_structure, targets
    )
    forbidden = tuple(
        atom_ref
        for atom_ref in StructureDisulfideHydrogenFacts.from_structure(
            structure
        ).forbidden_hydrogen_atom_refs()
        if atom_ref.residue_id in targets
    )
    normalized = structure.without_atom_refs(forbidden) if forbidden else structure
    prepared = PolymerMicrostatePreparation(
        PolymerMicrostateContext(normalized),
        library,
        requests=histidine_microstate_requests(structure, request),
    )
    result = _fixed_hydrogens(
        normalized, library, prepared, targets=targets, rebuilding=rebuilding
    )
    if forbidden:
        result = TransformationResult(
            result.structure,
            (
                *result.repairs,
                *(
                    RepairEvent.for_residue(
                        kind=RepairEventKind.HYDROGENS_REMOVED,
                        residue_id=residue_id,
                        component_id="CYS",
                        atom_names=tuple(
                            ref.atom_name
                            for ref in forbidden
                            if ref.residue_id == residue_id
                        ),
                        details=(
                            "removed thiol H incompatible with canonical "
                            "disulfide topology"
                        ),
                    )
                    for residue_id in sorted({ref.residue_id for ref in forbidden})
                ),
            ),
            result.issues,
        )

    # Carbonyl orders are prerequisites for neighboring amide-H geometry.
    # These immutable staging snapshots never escape to FF or the workflow.
    for carbonyls in (True, False):
        prepared = PolymerMicrostatePreparation(
            PolymerMicrostateContext(result.structure),
            library,
            requests=histidine_microstate_requests(result.structure, request),
        )
        patches = []
        repairs = list(result.repairs)
        issues = list(result.issues)
        for residue_id in sorted(targets):
            for target in prepared.targets_for(residue_id):
                if (target.site.kind is PolymerChemicalSite.BACKBONE_C) != carbonyls:
                    continue
                try:
                    patch = place_polymer_microstate_hydrogens(
                        target.context,
                        residue_id,
                        target.site,
                        override=target.override,
                        preferences=pras_microstate_preferences(target.site),
                        retain_applied_override=target.retain_applied_override,
                        rebuild_existing=residue_id in rebuilding,
                    )
                except GeometryPlacementError as error:
                    issues.append(
                        skipped_geometry_placement_issue(
                            _payload(result.structure, residue_id),
                            atom_names=tuple(
                                atom.name for atom, _ in target.hydrogen_atom_sites()
                            ),
                            reason=str(error),
                        )
                    )
                    continue
                except ValueError as error:
                    issues.append(_site_failure(target, str(error)))
                    continue

                patches.append(patch)
                added = tuple(
                    entry.atom.name
                    for entry in patch.hydrogens
                    if not result.structure.constitution.residue_site_at(
                        result.structure.constitution.residue_index(residue_id)
                    ).has_atom_site(entry.atom.name)
                )
                if added:
                    repairs.append(
                        RepairEvent.for_residue(
                            kind=RepairEventKind.HYDROGENS_ADDED,
                            residue_id=residue_id,
                            component_id=target.site.template.component_id,
                            atom_names=added,
                        )
                    )
                if not target.is_realized():
                    graph = patch.resolution.graph
                    assert graph is not None and patch.resolution.basis is not None
                    repairs.append(
                        RepairEvent.for_residue(
                            kind=RepairEventKind.POLYMER_MICROSTATE_APPLIED,
                            residue_id=residue_id,
                            component_id=target.site.template.component_id,
                            atom_names=tuple(atom.name for atom in graph.atoms),
                            details=(
                                f"{target.site.kind.value}: "
                                f"{patch.resolution.basis.value}; "
                                + "; ".join(patch.resolution.details)
                            ),
                        )
                    )
        result = TransformationResult(
            structure=apply_polymer_microstate_patches(
                result.structure, tuple(patches)
            ),
            repairs=tuple(repairs),
            issues=tuple(issues),
        )
    return result


def _fixed_hydrogens(
    structure: ProteinStructure,
    library: ComponentLibrary,
    prepared: PolymerMicrostatePreparation,
    *,
    targets: frozenset[ResidueId],
    rebuilding: frozenset[ResidueId],
) -> TransformationResult:
    updates = []
    repairs = []
    issues = []
    fixed_bond_refs = []
    disulfides = disulfide_bonded_cysteine_residue_ids(structure)
    for chain in structure.constitution.chains:
        if not any(residue.residue_id in targets for residue in chain.residues):
            continue
        payloads = tuple(
            _payload(structure, residue.residue_id) for residue in chain.residues
        )
        heavy = tuple(
            payload.without_atom_sites(
                {atom.name for atom in payload.atom_sites if atom.is_hydrogen()}
            )
            for payload in payloads
        )
        templates = tuple(library.get(payload.component_id) for payload in payloads)
        environment = HydrogenCompletionEnvironment.from_payloads(
            heavy,
            templates=templates,
            disulfide_bonded_residue_ids=disulfides.intersection(
                payload.residue_id for payload in payloads
            ),
        )
        for offset, (payload, template) in enumerate(
            zip(payloads, templates, strict=True)
        ):
            if payload.residue_id not in targets:
                continue
            diagnosis = diagnose_component_support(payload.component_id, library)
            if template is None:
                issues.append(
                    missing_component_definition_issue(
                        payload.residue_site,
                        diagnosis=diagnosis,
                        action="leaving residue unchanged during hydrogenation",
                    )
                )
                continue
            if not template.can_add_hydrogens():
                issues.append(
                    unsupported_hydrogenation_issue(
                        payload.residue_site, diagnosis=diagnosis
                    )
                )
                continue
            site_targets = prepared.targets_for(payload.residue_id)
            controlled = frozenset(
                name
                for target in site_targets
                for name in target.controlled_parent_names()
            )
            fixed_template_names = frozenset(
                name
                for name, parent in template.template_hydrogen_anchor_by_name(
                    template.expected_hydrogen_atom_names()
                ).items()
                if parent not in controlled
            )
            patch = OrderedAtomPatch.from_residue_payload(
                heavy[offset].residue_site,
                residue_geometry=heavy[offset].residue_geometry,
            )
            semantics = template.hydrogen_semantics
            assert semantics is not None
            try:
                fixed_identities = prepared.fixed_hydrogen_atom_sites(
                    payload.residue_id
                )
                named = prepared.context.hydrogen_parents(
                    payload.residue_id,
                    PolymerMicrostateSite(template, PolymerChemicalSite.SIDECHAIN),
                )
                if isinstance(semantics, IdealGeometryHydrogenSemantics):
                    generated = generate_component_hydrogen_patch(
                        residue=heavy[offset],
                        patch=patch,
                        semantics=semantics,
                        selected_hydrogen_names=fixed_template_names,
                    )
                    if generated is None:
                        raise GeometryPlacementError("insufficient heavy-atom anchors")
                else:
                    generated = generate_hydrogen_patch(
                        site=HydrogenResidueSite(
                            ResidueIndex(offset),
                            template,
                            environment,
                        ),
                        patch=patch,
                        semantics=semantics,
                        selected_hydrogen_names=fixed_template_names,
                    )
            except (GeometryPlacementError, KeyError, ValueError) as error:
                issues.append(
                    skipped_geometry_placement_issue(
                        payload,
                        atom_names=tuple(sorted(fixed_template_names)),
                        reason=str(error),
                    )
                )
                continue

            candidate = heavy[offset].apply_patch(generated)
            anchors = template.template_hydrogen_anchor_by_name(candidate.atom_names())
            unknown = tuple(
                atom.name
                for atom in candidate.atom_sites
                if atom.is_hydrogen() and atom.name not in anchors
            )
            if unknown:
                issues.append(
                    skipped_geometry_placement_issue(
                        payload,
                        atom_names=unknown,
                        reason="template H placement lacks canonical parent bonds",
                    )
                )
                continue
            generated_by_parent = {}
            for atom in candidate.atom_sites:
                if atom.is_hydrogen() and anchors[atom.name] not in controlled:
                    generated_by_parent.setdefault(anchors[atom.name], []).append(
                        atom.name
                    )
            selected_names = {atom.name for atom, _ in fixed_identities}
            fixed_parents = {parent for _, parent in fixed_identities}
            required_counts = Counter(parent for _, parent in fixed_identities)
            if any(
                len(generated_by_parent.get(parent, ())) < count
                for parent, count in required_counts.items()
            ):
                issues.append(
                    skipped_geometry_placement_issue(
                        payload,
                        atom_names=tuple(sorted(selected_names)),
                        reason="fixed H placement omitted a selected parent inventory",
                    )
                )
                continue
            updated = payload.without_atom_sites(
                {
                    name
                    for name, parent in named.items()
                    if parent in fixed_parents and name not in selected_names
                }
            )
            added = []
            for atom, parent in fixed_identities:
                generated_names = generated_by_parent.get(parent, [])
                generated_name = (
                    atom.name if atom.name in generated_names else generated_names[0]
                )
                generated_names.remove(generated_name)
                fixed_bond_refs.append(
                    (
                        AtomRef(payload.residue_id, parent),
                        AtomRef(payload.residue_id, atom.name),
                    )
                )
                if payload.has_atom_site(atom.name):
                    if payload.residue_id not in rebuilding:
                        continue
                    original_atom = payload.atom_site(atom.name)
                    if not original_atom.is_hydrogen():
                        continue
                    atom = original_atom
                    geometry = payload.atom_geometry(atom.name).with_position(
                        candidate.position(generated_name)
                    )
                else:
                    geometry = candidate.atom_geometry(generated_name)
                    observation = structure.provenance.ingress.observation
                    if observation is not None:
                        source_index = observation.constitution.resolve_atom_index(
                            AtomRef(payload.residue_id, atom.name)
                        )
                        if source_index is not None:
                            geometry = observation.geometry.atom_geometry(
                                source_index
                            ).with_position(geometry.position)
                    added.append(atom.name)
                updated = updated.with_atom_payload(
                    atom,
                    atom_geometry=geometry,
                    formal_charge=payload.formal_charge(atom.name),
                )
            if updated != payload:
                updates.append(
                    (
                        updated.residue_site,
                        updated.residue_geometry,
                        updated.formal_charge_by_atom_name,
                    )
                )
            if added:
                repairs.append(
                    RepairEvent.for_residue(
                        kind=RepairEventKind.HYDROGENS_ADDED,
                        residue_id=payload.residue_id,
                        component_id=payload.component_id,
                        atom_names=tuple(added),
                    )
                )
    result = (
        structure.with_updated_residue_facets_batch(updates) if updates else structure
    )
    added_bonds = tuple(
        TopologyBond(
            result.constitution.atom_index(parent),
            result.constitution.atom_index(hydrogen),
            relationship_type=BondRelationshipType.COVALENT,
            provenance=BondProvenance.TEMPLATE_RESOLVED,
        )
        for parent, hydrogen in fixed_bond_refs
        if result.topology.bond_between(
            result.constitution.atom_index(parent),
            result.constitution.atom_index(hydrogen),
        )
        is None
    )
    if added_bonds:
        result = ProteinStructure.from_payload(
            constitution=result.constitution,
            geometry=result.geometry,
            topology=StructureTopology(
                constitution=result.constitution,
                atom_topologies=result.topology.atom_topologies,
                bonds=(*result.topology.bonds, *added_bonds),
            ),
            polymer_blueprint=result.polymer_blueprint,
            provenance=result.provenance,
        )
    return TransformationResult(result, tuple(repairs), tuple(issues))


def _payload(
    structure: ProteinStructure, residue_id: ResidueId
) -> CompletionResiduePayload:
    index = structure.constitution.residue_index(residue_id)
    return CompletionResiduePayload(
        structure.constitution.residue_site_at(index),
        structure.residue_geometry(index),
        structure.residue_formal_charge_by_atom_name(index),
    )


def _restore_preparation_hydrogens(
    structure: ProteinStructure,
    before: ProteinStructure | None,
    residue_ids: frozenset[ResidueId],
) -> ProteinStructure:
    if before is None or before is structure:
        return structure
    updates = []
    restored_refs = set()
    for residue_id in sorted(residue_ids):
        source = before.constitution.residue_or_ligand(residue_id)
        if source is None:
            continue
        source_payload = _payload(before, residue_id)
        payload = _payload(structure, residue_id)
        for atom in source.atom_sites:
            if not atom.is_hydrogen() or payload.has_atom_site(atom.name):
                continue
            payload = payload.with_atom_payload(
                atom,
                atom_geometry=source_payload.atom_geometry(atom.name),
                formal_charge=source_payload.formal_charge(atom.name),
            )
            restored_refs.add(AtomRef(residue_id, atom.name))
        updates.append(
            (
                payload.residue_site,
                payload.residue_geometry,
                payload.formal_charge_by_atom_name,
            )
        )
    if not restored_refs:
        return structure
    result = structure.with_updated_residue_facets_batch(updates)
    bonds = tuple(
        bond
        for bond in before.topology.bonds_for_constitution(
            source_constitution=before.constitution,
            target_constitution=result.constitution,
        )
        if any(
            result.constitution.atom_ref_at(index) in restored_refs
            for index in bond.endpoint_pair()
        )
    )
    return ProteinStructure.from_payload(
        constitution=result.constitution,
        geometry=result.geometry,
        topology=StructureTopology(
            constitution=result.constitution,
            atom_topologies=result.topology.atom_topologies,
            bonds=(*result.topology.bonds, *bonds),
        ),
        provenance=result.provenance,
        polymer_blueprint=result.polymer_blueprint,
    )


def _heavy_anchors_changed(
    structure: ProteinStructure,
    before: ProteinStructure | None,
    residue_id: ResidueId,
) -> bool:
    if before is None or before is structure:
        return False
    current = _payload(structure, residue_id)
    original = before.constitution.residue_or_ligand(residue_id)
    if original is None:
        return True
    geometry = before.residue_geometry(before.constitution.residue_index(residue_id))
    return any(
        not original.has_atom_site(atom.name)
        or current.position(atom.name) != geometry.position(atom.name)
        for atom in current.atom_sites
        if not atom.is_hydrogen()
    )


def _site_failure(target: PolymerSitePreparation, reason: str) -> ValidationIssue:
    return ValidationIssue.for_residue(
        kind=(
            ValidationIssueKind.CHEMISTRY_CONTRADICTION
            if target.resolution.status is MicrostateResolutionStatus.CONFLICT
            else ValidationIssueKind.UNSUPPORTED_HYDROGENATION
        ),
        severity=IssueSeverity.WARNING,
        residue_id=target.residue_id,
        component_id=target.site.template.component_id,
        atom_names=tuple(sorted(target.controlled_parent_names())),
        message=f"polymer {target.site.kind.value} left unchanged: {reason}",
    )


__all__ = ["materialize_hydrogens_core"]
