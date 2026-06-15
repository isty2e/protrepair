"""Tests for relation and transformation boundary contracts."""

from typing import cast

import pytest
from tests.support.refinement_type_fixtures import (
    build_toy_component_library,
    build_toy_structure,
)

from protrepair.diagnostics import (
    IssueSeverity,
    RepairEvent,
    RepairEventKind,
    ValidationIssue,
    ValidationIssueKind,
)
from protrepair.relation import (
    AnchorDistanceConstraint,
    DensityEvidence,
    ExternalCorrespondenceEvidence,
    StructureEndpoint,
    StructureEndpointMapping,
    StructureRealizationSelector,
    SupportingStructureAuthorityAspect,
    SupportingStructureAuthorityGrant,
    SupportingStructureCorrespondence,
    SymmetryContactEvidence,
)
from protrepair.relation.reference import (
    StructureRealizationReference,
    StructureReference,
    StructureReferenceKind,
)
from protrepair.relation.supporting_role import SupportingStructureRole
from protrepair.scope import (
    AtomSetScope,
    ResidueSetScope,
)
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import StructureProvenanceOrigin
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.context import (
    ProteinTransformationContext,
    SupportingStructureContext,
)
from protrepair.transformer.continuous.settings import ContinuousRelaxationForceField
from protrepair.transformer.local import (
    DirectRegionTransformationSpec,
    LocalScopeSpec,
    LocalTransformationContextSpec,
    SupportingStructureAuthorityGrantSpec,
    SupportingStructureCorrespondenceSpec,
    SupportingStructureMappingSpec,
    SupportingStructureSpec,
    atom_input_from_local_scope_spec,
)


def test_local_transformation_context_models_supporting_structures() -> None:
    """Boundary context should represent supporting donor/reference structures."""

    residue_scope_spec = LocalScopeSpec.from_residues(
        (ResidueId(chain_id="A", seq_num=1),)
    )
    residue_scope = residue_scope_spec.as_scope()
    donor_scope = LocalScopeSpec.from_atoms(
        (AtomRef(ResidueId(chain_id="B", seq_num=9), "CA"),)
    ).as_scope()
    correspondence = SupportingStructureCorrespondenceSpec(
        source_scope=residue_scope,
        supporting_scope=donor_scope,
        mappings=(
            SupportingStructureMappingSpec(
                source_scope=residue_scope,
                supporting_scope=donor_scope,
            ),
        ),
    )
    context = LocalTransformationContextSpec(
        supporting_structures=(
            SupportingStructureSpec(
                role=SupportingStructureRole.DONOR,
                structure=build_toy_structure(),
                scope=residue_scope,
                authority_grants=(
                    SupportingStructureAuthorityGrantSpec(
                        correspondence=correspondence,
                        authoritative_aspects=(
                            SupportingStructureAuthorityAspect.COORDINATES,
                            SupportingStructureAuthorityAspect.HEAVY_ATOM_TOPOLOGY,
                        ),
                    ),
                ),
            ),
        )
    )
    spec = DirectRegionTransformationSpec(
        scope_spec=residue_scope_spec,
        force_field=ContinuousRelaxationForceField.UFF,
        context=context,
    )

    assert not context.is_source_only()
    assert spec.context.supporting_structures[0].role is SupportingStructureRole.DONOR
    assert spec.context.supporting_structures[0].correspondences == (correspondence,)
    assert spec.context.supporting_structures[0].correspondences[0].mappings == (
        SupportingStructureMappingSpec(
            source_scope=residue_scope,
            supporting_scope=donor_scope,
        ),
    )
    assert spec.context.supporting_structures[0].authority_grants[
        0
    ].authoritative_aspects == (
        SupportingStructureAuthorityAspect.COORDINATES,
        SupportingStructureAuthorityAspect.HEAVY_ATOM_TOPOLOGY,
    )

    with pytest.raises(TypeError, match="SupportingStructureSpec"):
        LocalTransformationContextSpec(
            supporting_structures=("donor",)  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="at least one authoritative aspect"):
        SupportingStructureAuthorityGrantSpec(
            correspondence=correspondence,
            authoritative_aspects=(),
        )


def test_local_transformation_context_models_external_evidence_and_constraints() -> (
    None
):
    """Boundary context should model non-structure evidence and constraints."""

    component_library = build_toy_component_library()
    source_snapshot = ProteinStructureSnapshot.from_structure(build_toy_structure())
    source_atom_domain = atom_input_from_local_scope_spec(
        source_snapshot,
        LocalScopeSpec.from_residues((ResidueId(chain_id="A", seq_num=1),)),
        component_library=component_library,
    )
    donor_snapshot = ProteinStructureSnapshot.from_structure(build_toy_structure())
    donor_atom_domain = atom_input_from_local_scope_spec(
        donor_snapshot,
        LocalScopeSpec.from_atoms((AtomRef(ResidueId(chain_id="A", seq_num=1), "C1"),)),
        component_library=component_library,
    )
    donor_correspondence = SupportingStructureCorrespondence(
        source_structure_endpoint=StructureEndpoint.source(
            AtomSetScope(atom_refs=(AtomRef(ResidueId(chain_id="A", seq_num=1), "C1"),))
        ),
        supporting_structure_endpoint=StructureEndpoint.supporting(
            AtomSetScope(atom_refs=(AtomRef(ResidueId(chain_id="A", seq_num=1), "C1"),))
        ),
        mappings=(
            StructureEndpointMapping(
                source_structure_endpoint=StructureEndpoint.source(
                    AtomSetScope(
                        atom_refs=(AtomRef(ResidueId(chain_id="A", seq_num=1), "C1"),)
                    )
                ),
                supporting_structure_endpoint=StructureEndpoint.supporting(
                    AtomSetScope(
                        atom_refs=(AtomRef(ResidueId(chain_id="A", seq_num=1), "C1"),)
                    )
                ),
            ),
        ),
    )
    source_endpoint = StructureEndpoint.source(
        ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),)),
        realization_selector=StructureRealizationSelector(model_index=0),
    )
    donor_origin = StructureProvenanceOrigin(
        structure_endpoint=StructureEndpoint.supporting(
            AtomSetScope(
                atom_refs=(AtomRef(ResidueId(chain_id="A", seq_num=1), "C1"),)
            ),
            token="supporting-0",
            realization_selector=StructureRealizationSelector(
                model_index=2,
                altloc_label=" B ",
            ),
        )
    )
    source_anchor_origin = StructureProvenanceOrigin(
        structure_endpoint=StructureEndpoint.source(
            AtomSetScope(atom_refs=(AtomRef(ResidueId(chain_id="A", seq_num=1), "C1"),))
        )
    )
    context = LocalTransformationContextSpec(
        external_evidence=(
            DensityEvidence(
                target_structure_endpoint=source_endpoint,
                map_label=" 2fo-fc ",
            ),
            SymmetryContactEvidence(
                target_structure_endpoint=source_endpoint,
                symmetry_operator_label=" x,y,z ",
            ),
            ExternalCorrespondenceEvidence(
                source_origin=source_anchor_origin,
                counterpart_origin=donor_origin,
                evidence_label=" manual-correspondence ",
            ),
        ),
        external_constraints=(
            AnchorDistanceConstraint(
                left_anchor_origin=source_anchor_origin,
                right_anchor_origin=donor_origin,
                target_distance_angstrom=3.0,
                tolerance_angstrom=0.25,
            ),
        ),
    )
    transformer_context = ProteinTransformationContext(
        source_snapshot=source_snapshot,
        atom_input=source_atom_domain,
        supporting_structures=(
            SupportingStructureContext(
                role=SupportingStructureRole.DONOR,
                snapshot=donor_snapshot,
                atom_input=donor_atom_domain,
                authority_grants=(
                    SupportingStructureAuthorityGrant(
                        correspondence=donor_correspondence,
                        authoritative_aspects=(
                            SupportingStructureAuthorityAspect.COORDINATES,
                        ),
                    ),
                ),
            ),
        ),
        external_evidence=context.external_evidence,
        external_constraints=context.external_constraints,
    )
    density_evidence = context.external_evidence[0]
    symmetry_evidence = context.external_evidence[1]
    correspondence_evidence = context.external_evidence[2]

    assert not context.is_source_only()
    assert isinstance(density_evidence, DensityEvidence)
    assert isinstance(symmetry_evidence, SymmetryContactEvidence)
    assert isinstance(correspondence_evidence, ExternalCorrespondenceEvidence)
    assert density_evidence.map_label == "2fo-fc"
    assert density_evidence.target_structure_endpoint.realization_selector == (
        StructureRealizationSelector(model_index=0)
    )
    assert symmetry_evidence.symmetry_operator_label == "x,y,z"
    assert correspondence_evidence.evidence_label == "manual-correspondence"
    assert donor_origin.structure_endpoint.realization_selector == (
        StructureRealizationSelector(model_index=2, altloc_label="B")
    )
    assert transformer_context.external_constraints == context.external_constraints
    assert not transformer_context.is_source_only()

    with pytest.raises(ValueError, match="atom-local provenance origins"):
        AnchorDistanceConstraint(
            left_anchor_origin=StructureProvenanceOrigin(
                structure_endpoint=StructureEndpoint.source(
                    ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),))
                )
            ),
            right_anchor_origin=donor_origin,
            target_distance_angstrom=3.0,
            tolerance_angstrom=0.25,
        )

    with pytest.raises(ValueError, match="two distinct provenance origins"):
        ExternalCorrespondenceEvidence(
            source_origin=source_anchor_origin,
            counterpart_origin=source_anchor_origin,
        )


def test_structure_reference_supports_model_and_altloc_specific_realizations() -> None:
    """Structure references should model one specific model/altloc realization."""

    realization = StructureRealizationReference(
        model_index=3,
        altloc_label=" A ",
    )
    residue_reference = StructureReference.from_residues(
        (ResidueId(chain_id="A", seq_num=7),),
        realization=realization,
    )
    atom_reference = StructureReference.from_atoms(
        (AtomRef(ResidueId(chain_id="A", seq_num=7), "CA"),),
        realization=realization,
    )

    assert realization.model_index == 3
    assert realization.altloc_label == "A"
    assert residue_reference.realization == realization
    assert atom_reference.realization == realization
    assert atom_reference.referenced_residue_ids() == (
        ResidueId(chain_id="A", seq_num=7),
    )

    with pytest.raises(ValueError, match="at least one of model_index or altloc_label"):
        StructureRealizationReference()

    with pytest.raises(ValueError, match="non-negative"):
        StructureRealizationReference(model_index=-1)

    with pytest.raises(TypeError, match="StructureRealizationReference or None"):
        StructureReference(
            kind=residue_reference.kind,
            residue_ids=(ResidueId(chain_id="A", seq_num=7),),
            realization=cast("StructureRealizationReference | None", "A"),
        )


def test_structure_reference_supports_absent_anchor_and_composite_basis() -> None:
    """Structure references should align with absent, anchor, and composite targets."""

    absent_reference = StructureReference.from_absent_residue_span(
        preceding_residue_id=ResidueId(chain_id="A", seq_num=10),
        following_residue_id=ResidueId(chain_id="A", seq_num=14),
        absent_residue_ids=(
            ResidueId(chain_id="A", seq_num=11),
            ResidueId(chain_id="A", seq_num=12),
            ResidueId(chain_id="A", seq_num=13),
        ),
    )
    anchor_reference = StructureReference.from_anchor_atom_pair(
        AtomRef(ResidueId(chain_id="A", seq_num=10), "C"),
        AtomRef(ResidueId(chain_id="A", seq_num=14), "N"),
    )
    residue_reference = StructureReference.from_residues(
        (
            ResidueId(chain_id="A", seq_num=10),
            ResidueId(chain_id="A", seq_num=14),
        )
    )
    composite_reference = StructureReference.composite(
        (
            absent_reference,
            anchor_reference,
            residue_reference,
        )
    )

    assert absent_reference.referenced_residue_ids() == (
        ResidueId(chain_id="A", seq_num=10),
        ResidueId(chain_id="A", seq_num=11),
        ResidueId(chain_id="A", seq_num=12),
        ResidueId(chain_id="A", seq_num=13),
        ResidueId(chain_id="A", seq_num=14),
    )
    assert absent_reference.cardinality() == 5
    assert anchor_reference.referenced_residue_ids() == (
        ResidueId(chain_id="A", seq_num=10),
        ResidueId(chain_id="A", seq_num=14),
    )
    assert anchor_reference.cardinality() == 2
    assert composite_reference.referenced_residue_ids() == (
        ResidueId(chain_id="A", seq_num=10),
        ResidueId(chain_id="A", seq_num=11),
        ResidueId(chain_id="A", seq_num=12),
        ResidueId(chain_id="A", seq_num=13),
        ResidueId(chain_id="A", seq_num=14),
    )
    assert composite_reference.cardinality() == 3
    assert composite_reference.covers(anchor_reference)
    assert absent_reference.covers(
        StructureReference.from_absent_residue_span(
            preceding_residue_id=ResidueId(chain_id="A", seq_num=10),
            following_residue_id=ResidueId(chain_id="A", seq_num=14),
            absent_residue_ids=(
                ResidueId(chain_id="A", seq_num=11),
                ResidueId(chain_id="A", seq_num=12),
            ),
        )
    )

    with pytest.raises(ValueError, match="at least one anchor residue"):
        StructureReference.from_absent_residue_span()

    with pytest.raises(ValueError, match="require both anchor atom refs"):
        StructureReference(
            kind=StructureReferenceKind.ANCHOR_ATOM_PAIR,
            left_anchor_atom_ref=AtomRef(ResidueId(chain_id="A", seq_num=10), "C"),
        )

    with pytest.raises(ValueError, match="must stay flat"):
        StructureReference.composite((composite_reference, anchor_reference))

    with pytest.raises(ValueError, match="must not carry a top-level realization"):
        StructureReference(
            kind=StructureReferenceKind.COMPOSITE,
            member_references=(anchor_reference, residue_reference),
            realization=StructureRealizationReference(model_index=0),
        )


def test_protein_transformation_context_supports_additional_structures() -> None:
    """Canonical transformer context should preserve supporting structure evidence."""

    component_library = build_toy_component_library()
    source_snapshot = ProteinStructureSnapshot.from_structure(build_toy_structure())
    source_atom_domain = atom_input_from_local_scope_spec(
        source_snapshot,
        LocalScopeSpec.from_residues((ResidueId(chain_id="A", seq_num=1),)),
        component_library=component_library,
    )
    donor_snapshot = ProteinStructureSnapshot.from_structure(build_toy_structure())
    donor_atom_domain = atom_input_from_local_scope_spec(
        donor_snapshot,
        LocalScopeSpec.from_atoms((AtomRef(ResidueId(chain_id="A", seq_num=1), "C1"),)),
        component_library=component_library,
    )
    correspondence = SupportingStructureCorrespondence(
        source_structure_endpoint=StructureEndpoint.source(
            ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),))
        ),
        supporting_structure_endpoint=StructureEndpoint.supporting(
            AtomSetScope(atom_refs=(AtomRef(ResidueId(chain_id="A", seq_num=1), "C1"),))
        ),
        mappings=(
            StructureEndpointMapping(
                source_structure_endpoint=StructureEndpoint.source(
                    ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),))
                ),
                supporting_structure_endpoint=StructureEndpoint.supporting(
                    AtomSetScope(
                        atom_refs=(AtomRef(ResidueId(chain_id="A", seq_num=1), "C1"),)
                    )
                ),
            ),
        ),
    )
    context = ProteinTransformationContext(
        source_snapshot=source_snapshot,
        atom_input=source_atom_domain,
        supporting_structures=(
            SupportingStructureContext(
                role=SupportingStructureRole.DONOR,
                snapshot=donor_snapshot,
                atom_input=donor_atom_domain,
                authority_grants=(
                    SupportingStructureAuthorityGrant(
                        correspondence=correspondence,
                        authoritative_aspects=(
                            SupportingStructureAuthorityAspect.COORDINATES,
                        ),
                    ),
                ),
            ),
        ),
    )

    assert not context.is_source_only()
    assert (
        context.supporting_structures_of_role(SupportingStructureRole.DONOR)
        == context.supporting_structures
    )
    assert context.supporting_structures[0].correspondences == (correspondence,)
    assert context.supporting_structures[0].correspondences[0].mappings[
        0
    ].cardinality_signature() == (1, 1)
    assert context.supporting_structures[0].has_authority_for(
        SupportingStructureAuthorityAspect.COORDINATES
    )
    assert (
        context.supporting_structures_authoritative_for(
            SupportingStructureAuthorityAspect.COORDINATES
        )
        == context.supporting_structures
    )


def test_supporting_structure_context_rejects_ungrounded_supporting_references() -> (
    None
):
    """Support contexts should reject correspondences outside the support snapshot."""

    with pytest.raises(
        ValueError,
        match=(
            "supporting_structure_endpoint must ground in the active supporting "
            "snapshot"
        ),
    ):
        SupportingStructureContext(
            role=SupportingStructureRole.DONOR,
            snapshot=ProteinStructureSnapshot.from_structure(build_toy_structure()),
            correspondences=(
                SupportingStructureCorrespondence(
                    source_structure_endpoint=StructureEndpoint.source(
                        ResidueSetScope(
                            residue_ids=(ResidueId(chain_id="A", seq_num=1),)
                        )
                    ),
                    supporting_structure_endpoint=StructureEndpoint.supporting(
                        AtomSetScope(
                            atom_refs=(
                                AtomRef(
                                    ResidueId(chain_id="A", seq_num=9),
                                    "C1",
                                ),
                            )
                        )
                    ),
                ),
            ),
        )


def test_protein_transformation_context_rejects_ungrounded_external_references() -> (
    None
):
    """External evidence should ground against the active source snapshot."""

    source_snapshot = ProteinStructureSnapshot.from_structure(build_toy_structure())
    source_atom_domain = atom_input_from_local_scope_spec(
        source_snapshot,
        LocalScopeSpec.from_residues((ResidueId(chain_id="A", seq_num=1),)),
        component_library=build_toy_component_library(),
    )

    with pytest.raises(
        ValueError,
        match=(
            "DensityEvidence target_structure_endpoint must ground in the active "
            "source snapshot"
        ),
    ):
        ProteinTransformationContext(
            source_snapshot=source_snapshot,
            atom_input=source_atom_domain,
            external_evidence=(
                DensityEvidence(
                    target_structure_endpoint=StructureEndpoint.source(
                        ResidueSetScope(
                            residue_ids=(ResidueId(chain_id="A", seq_num=9),)
                        )
                    )
                ),
            ),
        )


def test_protein_transformation_context_rejects_ambiguous_supporting_origins() -> None:
    """Supporting provenance origins should disambiguate repeated support roles."""

    component_library = build_toy_component_library()
    source_snapshot = ProteinStructureSnapshot.from_structure(build_toy_structure())
    source_atom_domain = atom_input_from_local_scope_spec(
        source_snapshot,
        LocalScopeSpec.from_residues((ResidueId(chain_id="A", seq_num=1),)),
        component_library=component_library,
    )
    donor_snapshot = ProteinStructureSnapshot.from_structure(build_toy_structure())
    donor_correspondence = SupportingStructureCorrespondence(
        source_structure_endpoint=StructureEndpoint.source(
            AtomSetScope(atom_refs=(AtomRef(ResidueId(chain_id="A", seq_num=1), "C1"),))
        ),
        supporting_structure_endpoint=StructureEndpoint.supporting(
            AtomSetScope(atom_refs=(AtomRef(ResidueId(chain_id="A", seq_num=1), "C1"),))
        ),
    )
    source_origin = StructureProvenanceOrigin(
        structure_endpoint=StructureEndpoint.source(
            AtomSetScope(atom_refs=(AtomRef(ResidueId(chain_id="A", seq_num=1), "C1"),))
        )
    )
    ambiguous_donor_origin = StructureProvenanceOrigin(
        structure_endpoint=StructureEndpoint.supporting(
            AtomSetScope(
                atom_refs=(AtomRef(ResidueId(chain_id="A", seq_num=1), "C1"),)
            ),
        ),
    )

    with pytest.raises(ValueError, match="require a unique carrier token"):
        ProteinTransformationContext(
            source_snapshot=source_snapshot,
            atom_input=source_atom_domain,
            supporting_structures=(
                SupportingStructureContext(
                    role=SupportingStructureRole.DONOR,
                    snapshot=donor_snapshot,
                    authority_grants=(
                        SupportingStructureAuthorityGrant(
                            correspondence=donor_correspondence,
                            authoritative_aspects=(
                                SupportingStructureAuthorityAspect.COORDINATES,
                            ),
                        ),
                    ),
                ),
                SupportingStructureContext(
                    role=SupportingStructureRole.DONOR,
                    snapshot=donor_snapshot,
                    authority_grants=(
                        SupportingStructureAuthorityGrant(
                            correspondence=donor_correspondence,
                            authoritative_aspects=(
                                SupportingStructureAuthorityAspect.COORDINATES,
                            ),
                        ),
                    ),
                ),
            ),
            external_constraints=(
                AnchorDistanceConstraint(
                    left_anchor_origin=source_origin,
                    right_anchor_origin=ambiguous_donor_origin,
                    target_distance_angstrom=3.0,
                    tolerance_angstrom=0.25,
                ),
            ),
        )


def test_supporting_anchor_constraints_require_coordinate_authority() -> None:
    """Supporting anchor constraints should require coordinate authority."""

    component_library = build_toy_component_library()
    source_snapshot = ProteinStructureSnapshot.from_structure(build_toy_structure())
    source_atom_domain = atom_input_from_local_scope_spec(
        source_snapshot,
        LocalScopeSpec.from_residues((ResidueId(chain_id="A", seq_num=1),)),
        component_library=component_library,
    )
    donor_snapshot = ProteinStructureSnapshot.from_structure(build_toy_structure())
    source_origin = StructureProvenanceOrigin(
        structure_endpoint=StructureEndpoint.source(
            AtomSetScope(atom_refs=(AtomRef(ResidueId(chain_id="A", seq_num=1), "C1"),))
        )
    )
    donor_origin = StructureProvenanceOrigin(
        structure_endpoint=StructureEndpoint.supporting(
            AtomSetScope(
                atom_refs=(AtomRef(ResidueId(chain_id="A", seq_num=1), "C1"),)
            ),
            token="supporting-0",
        )
    )
    donor_correspondence = SupportingStructureCorrespondence(
        source_structure_endpoint=StructureEndpoint.source(
            AtomSetScope(atom_refs=(AtomRef(ResidueId(chain_id="A", seq_num=1), "C1"),))
        ),
        supporting_structure_endpoint=StructureEndpoint.supporting(
            AtomSetScope(atom_refs=(AtomRef(ResidueId(chain_id="A", seq_num=1), "C1"),))
        ),
    )

    with pytest.raises(
        ValueError,
        match="require a declared coordinate authority grant",
    ):
        ProteinTransformationContext(
            source_snapshot=source_snapshot,
            atom_input=source_atom_domain,
            supporting_structures=(
                SupportingStructureContext(
                    role=SupportingStructureRole.DONOR,
                    snapshot=donor_snapshot,
                    correspondences=(donor_correspondence,),
                ),
            ),
            external_constraints=(
                AnchorDistanceConstraint(
                    left_anchor_origin=source_origin,
                    right_anchor_origin=donor_origin,
                    target_distance_angstrom=3.0,
                    tolerance_angstrom=0.25,
                ),
            ),
        )


def test_repair_and_issue_provenance_can_reference_supporting_structures() -> None:
    """Diagnostics provenance should distinguish source and supporting origins."""

    source_origin = StructureProvenanceOrigin(
        structure_endpoint=StructureEndpoint.source(
            ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),)),
            token="source-structure",
        )
    )
    donor_origin = StructureProvenanceOrigin(
        structure_endpoint=StructureEndpoint.supporting(
            AtomSetScope(
                atom_refs=(AtomRef(ResidueId(chain_id="B", seq_num=9), "CA"),)
            ),
            token="donor-fragment",
        ),
    )
    repair = RepairEvent.for_residue(
        kind=RepairEventKind.HEAVY_ATOMS_ADDED,
        residue_id=ResidueId(chain_id="A", seq_num=1),
        component_id="MOV",
        atom_names=("H1",),
        provenance_origins=(source_origin, donor_origin, donor_origin),
    )
    issue = ValidationIssue.for_residue(
        kind=ValidationIssueKind.REFINEMENT_REJECTED,
        severity=IssueSeverity.WARNING,
        residue_id=ResidueId(chain_id="A", seq_num=1),
        message="support-guided rejection",
        provenance_origins=(donor_origin,),
    )

    assert repair.provenance_origins == (source_origin, donor_origin)
    assert repair.supporting_provenance_origins() == (donor_origin,)
    assert issue.provenance_origins == (donor_origin,)
    assert issue.supporting_provenance_origins() == (donor_origin,)
    assert source_origin.is_source()
    assert donor_origin.is_supporting()

    with pytest.raises(TypeError, match="StructureEndpoint"):
        StructureProvenanceOrigin(
            structure_endpoint=cast(StructureEndpoint, "not-a-carrier-scope"),
        )


def test_supporting_structure_correspondence_rejects_out_of_scope_mappings() -> None:
    """Mapped support correspondences should stay inside their parent references."""

    with pytest.raises(ValueError, match="must fall inside the correspondence"):
        SupportingStructureCorrespondence(
            source_structure_endpoint=StructureEndpoint.source(
                ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),))
            ),
            supporting_structure_endpoint=StructureEndpoint.supporting(
                AtomSetScope(
                    atom_refs=(AtomRef(ResidueId(chain_id="B", seq_num=9), "CA"),)
                )
            ),
            mappings=(
                StructureEndpointMapping(
                    source_structure_endpoint=StructureEndpoint.source(
                        ResidueSetScope(
                            residue_ids=(ResidueId(chain_id="A", seq_num=2),)
                        )
                    ),
                    supporting_structure_endpoint=StructureEndpoint.supporting(
                        AtomSetScope(
                            atom_refs=(
                                AtomRef(ResidueId(chain_id="B", seq_num=9), "CA"),
                            )
                        )
                    ),
                ),
            ),
        )
