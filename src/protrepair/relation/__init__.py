"""Canonical relation basis for scoped carriers and external links."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from protrepair.relation.blueprint import (
        StructureBlueprintCoverage,
        StructureBlueprintCoverageGap,
        StructureBlueprintResidueMapping,
    )
    from protrepair.relation.boundary import (
        SupportingStructureAuthorityGrantSpec,
        SupportingStructureCorrespondenceSpec,
        SupportingStructureMappingSpec,
        boundary_structure_reference_from_scope,
    )
    from protrepair.relation.endpoint import (
        StructureCarrierHandle,
        StructureCarrierKind,
        StructureEndpoint,
        StructureRealizationSelector,
    )
    from protrepair.relation.evidence import (
        AnchorDistanceConstraint,
        DensityEvidence,
        ExternalCorrespondenceEvidence,
        SymmetryContactEvidence,
    )
    from protrepair.relation.reference import (
        StructureRealizationReference,
        StructureReference,
        StructureReferenceKind,
        reference_from_scope,
        scope_from_reference,
    )
    from protrepair.relation.structure import (
        StructureEndpointMapping,
        SupportingStructureAuthorityAspect,
        SupportingStructureAuthorityGrant,
        SupportingStructureCorrespondence,
    )
    from protrepair.relation.supporting_role import SupportingStructureRole
    from protrepair.relation.uniprot import (
        StructureUniProtCorrespondence,
        StructureUniProtCorrespondenceSet,
        UniProtCorrespondenceConfidence,
    )
    from protrepair.relation.uniprot_reference import UniProtSequenceReference

__all__ = [
    "AnchorDistanceConstraint",
    "DensityEvidence",
    "ExternalCorrespondenceEvidence",
    "StructureBlueprintCoverage",
    "StructureBlueprintCoverageGap",
    "StructureBlueprintResidueMapping",
    "StructureCarrierHandle",
    "StructureCarrierKind",
    "StructureEndpoint",
    "StructureEndpointMapping",
    "StructureRealizationReference",
    "StructureRealizationSelector",
    "StructureReference",
    "StructureReferenceKind",
    "SupportingStructureAuthorityAspect",
    "SupportingStructureAuthorityGrant",
    "SupportingStructureAuthorityGrantSpec",
    "SupportingStructureCorrespondence",
    "SupportingStructureCorrespondenceSpec",
    "SupportingStructureMappingSpec",
    "StructureUniProtCorrespondence",
    "StructureUniProtCorrespondenceSet",
    "SupportingStructureRole",
    "UniProtCorrespondenceConfidence",
    "UniProtSequenceReference",
    "SymmetryContactEvidence",
    "boundary_structure_reference_from_scope",
    "reference_from_scope",
    "scope_from_reference",
]


def __getattr__(name: str):
    """Lazily expose relation entities across the structure provenance cycle.

    This package cannot eager-import the full relation basis while
    ``protrepair.structure.provenance`` imports ``protrepair.relation.endpoint``:
    ``relation.evidence`` also imports ``structure.provenance`` for provenance
    evidence, creating a concrete package-init cycle.
    """

    if name in {
        "SupportingStructureAuthorityGrantSpec",
        "SupportingStructureCorrespondenceSpec",
        "SupportingStructureMappingSpec",
        "boundary_structure_reference_from_scope",
    }:
        from protrepair.relation import boundary

        return getattr(boundary, name)
    if name in {
        "StructureBlueprintCoverage",
        "StructureBlueprintCoverageGap",
        "StructureBlueprintResidueMapping",
    }:
        from protrepair.relation import blueprint

        return getattr(blueprint, name)

    if name in {
        "StructureCarrierHandle",
        "StructureCarrierKind",
        "StructureEndpoint",
        "StructureRealizationSelector",
    }:
        from protrepair.relation import endpoint

        return getattr(endpoint, name)

    if name in {
        "AnchorDistanceConstraint",
        "DensityEvidence",
        "ExternalCorrespondenceEvidence",
        "SymmetryContactEvidence",
    }:
        from protrepair.relation import evidence

        return getattr(evidence, name)

    if name in {
        "StructureRealizationReference",
        "StructureReference",
        "StructureReferenceKind",
        "reference_from_scope",
        "scope_from_reference",
    }:
        from protrepair.relation import reference

        return getattr(reference, name)

    if name in {
        "StructureEndpointMapping",
        "SupportingStructureAuthorityAspect",
        "SupportingStructureAuthorityGrant",
        "SupportingStructureCorrespondence",
    }:
        from protrepair.relation import structure

        return getattr(structure, name)

    if name in {
        "StructureUniProtCorrespondence",
        "StructureUniProtCorrespondenceSet",
        "UniProtCorrespondenceConfidence",
    }:
        from protrepair.relation import uniprot

        return getattr(uniprot, name)

    if name == "UniProtSequenceReference":
        from protrepair.relation import uniprot_reference

        return uniprot_reference.UniProtSequenceReference

    if name == "SupportingStructureRole":
        from protrepair.relation import supporting_role

        return supporting_role.SupportingStructureRole

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
