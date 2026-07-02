"""Boundary relation specs over semantic scopes."""

from dataclasses import dataclass

from protrepair.relation.reference import StructureReference, reference_from_scope
from protrepair.relation.structure import (
    StructureEndpointMapping,
    SupportingStructureAuthorityAspect,
    SupportingStructureAuthorityGrant,
    SupportingStructureCorrespondence,
)
from protrepair.scope.base import (
    Scope,
)
from protrepair.scope.relations import scope_refines
from protrepair.structure.endpoint import StructureEndpoint


def _require_scope(scope: Scope, *, field_name: str) -> Scope:
    """Return one validated semantic scope."""

    if not isinstance(scope, Scope):
        raise TypeError(f"{field_name} must be a Scope value")

    return scope


def boundary_structure_reference_from_scope(scope: Scope) -> StructureReference:
    """Return the canonical structure reference for one semantic scope."""

    return reference_from_scope(scope, realization_selector=None)


@dataclass(frozen=True, slots=True)
class SupportingStructureMappingSpec:
    """Boundary mapped scope correspondence inside one broader support relation."""

    source_scope: Scope
    supporting_scope: Scope

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_scope",
            _require_scope(self.source_scope, field_name="source_scope"),
        )
        object.__setattr__(
            self,
            "supporting_scope",
            _require_scope(
                self.supporting_scope,
                field_name="supporting_scope",
            ),
        )

    def to_relation(self) -> StructureEndpointMapping:
        """Return the canonical relation value for this mapping spec."""

        return StructureEndpointMapping(
            source_structure_endpoint=StructureEndpoint.source(self.source_scope),
            supporting_structure_endpoint=StructureEndpoint.supporting(
                self.supporting_scope
            ),
        )


@dataclass(frozen=True, slots=True)
class SupportingStructureCorrespondenceSpec:
    """Boundary correspondence between one source scope and one support scope."""

    source_scope: Scope
    supporting_scope: Scope
    mappings: tuple[SupportingStructureMappingSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_scope",
            _require_scope(self.source_scope, field_name="source_scope"),
        )
        object.__setattr__(
            self,
            "supporting_scope",
            _require_scope(
                self.supporting_scope,
                field_name="supporting_scope",
            ),
        )

        mappings = tuple(dict.fromkeys(self.mappings))
        for mapping in mappings:
            if not isinstance(mapping, SupportingStructureMappingSpec):
                raise TypeError(
                    "supporting structure correspondence specs require "
                    "SupportingStructureMappingSpec values"
                )
            if not scope_refines(mapping.source_scope, self.source_scope):
                raise ValueError(
                    "supporting structure correspondence mapping source_scope "
                    "must refine the correspondence source_scope"
                )
            if not scope_refines(
                mapping.supporting_scope,
                self.supporting_scope,
            ):
                raise ValueError(
                    "supporting structure correspondence mapping "
                    "supporting_scope must refine the correspondence "
                    "supporting_scope"
                )

        object.__setattr__(self, "mappings", mappings)

    def to_relation(self) -> SupportingStructureCorrespondence:
        """Return the canonical relation value for this correspondence spec."""

        return SupportingStructureCorrespondence(
            source_structure_endpoint=StructureEndpoint.source(self.source_scope),
            supporting_structure_endpoint=StructureEndpoint.supporting(
                self.supporting_scope
            ),
            mappings=tuple(mapping.to_relation() for mapping in self.mappings),
        )


@dataclass(frozen=True, slots=True)
class SupportingStructureAuthorityGrantSpec:
    """Boundary authority grant over one support correspondence and its aspects."""

    correspondence: SupportingStructureCorrespondenceSpec
    authoritative_aspects: tuple[SupportingStructureAuthorityAspect, ...]

    def __post_init__(self) -> None:
        if not isinstance(
            self.correspondence,
            SupportingStructureCorrespondenceSpec,
        ):
            raise TypeError(
                "supporting structure authority grant specs require one "
                "SupportingStructureCorrespondenceSpec"
            )

        authoritative_aspects = tuple(dict.fromkeys(self.authoritative_aspects))
        if not authoritative_aspects:
            raise ValueError(
                "supporting structure authority grant specs require at least one "
                "authoritative aspect"
            )
        for aspect in authoritative_aspects:
            if not isinstance(aspect, SupportingStructureAuthorityAspect):
                raise TypeError(
                    "supporting structure authority grant specs require "
                    "SupportingStructureAuthorityAspect values"
                )

        object.__setattr__(self, "authoritative_aspects", authoritative_aspects)

    def to_relation(self) -> SupportingStructureAuthorityGrant:
        """Return the canonical authority relation value for this grant spec."""

        return SupportingStructureAuthorityGrant(
            correspondence=self.correspondence.to_relation(),
            authoritative_aspects=self.authoritative_aspects,
        )
