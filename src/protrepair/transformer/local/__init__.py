"""Public boundary for local selected-domain transformations."""

from protrepair.relation import (
    SupportingStructureAuthorityGrantSpec,
    SupportingStructureCorrespondenceSpec,
    SupportingStructureMappingSpec,
)
from protrepair.relation.supporting_role import SupportingStructureRole
from protrepair.transformer.local.api import (
    LocalRegionTransformationRuntimeProfile,
    profile_transform_local_region,
    transform_local_region,
)
from protrepair.transformer.local.lowering import atom_input_from_local_scope_spec
from protrepair.transformer.local.models import (
    DirectRegionTransformationSpec,
    LocalScope,
    LocalScopeLowering,
    LocalScopeSpec,
    LocalTransformationContextSpec,
    SupportingStructureSpec,
)

__all__ = [
    "DirectRegionTransformationSpec",
    "LocalScope",
    "LocalTransformationContextSpec",
    "LocalScopeLowering",
    "LocalScopeSpec",
    "SupportingStructureRole",
    "SupportingStructureAuthorityGrantSpec",
    "SupportingStructureCorrespondenceSpec",
    "SupportingStructureMappingSpec",
    "SupportingStructureSpec",
    "LocalRegionTransformationRuntimeProfile",
    "atom_input_from_local_scope_spec",
    "profile_transform_local_region",
    "transform_local_region",
]
