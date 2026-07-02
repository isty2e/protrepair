"""Canonical relation basis for scoped carriers and external links."""

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
from protrepair.structure.endpoint import (
    StructureCarrierHandle,
    StructureCarrierKind,
    StructureEndpoint,
    StructureRealizationSelector,
)

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
